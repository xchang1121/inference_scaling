"""R7: request-local, epoch-based width learning around the unchanged Uno engine.

Python 3.10 stdlib only. This updates policy statistics, NOT LoRA weights.
"""
from __future__ import annotations

import math
from time import perf_counter_ns


class NativeWidthPolicy:
    def __init__(self, widths=(4, 8, 16), preferred=8, epoch_cycles=2,
                 retention=0.75, switch_margin=0.03, probe_every=16):
        self.widths = tuple(sorted(widths))
        if (not self.widths or len(set(self.widths)) != len(self.widths)
                or any(type(b) is not int or b < 1 for b in self.widths)
                or preferred not in self.widths or type(epoch_cycles) is not int or epoch_cycles < 1
                or not 0 <= retention < 1 or not math.isfinite(switch_margin) or switch_margin < 0
                or type(probe_every) is not int or probe_every < 1):
            raise ValueError("invalid native policy configuration")
        self.preferred = preferred
        self.order = (preferred,) + tuple(b for b in self.widths if b != preferred)
        self.epoch_cycles = epoch_cycles
        self.retention = retention
        self.switch_margin = switch_margin
        self.probe_every = probe_every
        self.counts = dict.fromkeys(self.widths, 0)
        self.mean_tokens = dict.fromkeys(self.widths, 0.0)
        self.mean_seconds = dict.fromkeys(self.widths, 0.0)
        self.total_tokens = dict.fromkeys(self.widths, 0)
        self.total_seconds = dict.fromkeys(self.widths, 0.0)
        self.completed_epochs = 0
        self.adaptive_epochs = 0
        self.probe_cursor = 0
        self.current = preferred
        self.current_reason = "initial_probe"
        self.epoch_count = 0
        self.epoch_tokens = 0
        self.epoch_seconds = 0.0
        self.pending = None

    def choose(self):
        if self.pending is not None:
            raise RuntimeError("must observe the previous completed cycle before choosing")
        if self.epoch_count == 0:
            unseen = next((b for b in self.order if self.counts[b] == 0), None)
            if unseen is not None:
                self.current, self.current_reason = unseen, "initial_probe"
            else:
                self.adaptive_epochs += 1
                if self.adaptive_epochs % self.probe_every == 0:
                    self.current = self.order[self.probe_cursor % len(self.order)]
                    self.probe_cursor += 1
                    self.current_reason = "refresh_probe"
                else:
                    scores = {b: self.mean_tokens[b] / self.mean_seconds[b] for b in self.widths}
                    best = max(self.order, key=scores.__getitem__)
                    if scores[best] <= scores[self.preferred] * (1 + self.switch_margin):
                        best = self.preferred
                    self.current, self.current_reason = best, "guarded_exploit"
        self.pending = self.current
        return self.current, self.current_reason

    def observe(self, width, tokens, seconds):
        if self.pending is None or self.pending != width:
            raise RuntimeError("feedback does not match the pending width")
        if type(tokens) is not int or not 1 <= tokens <= width + 1 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("valid committed count and finite positive cycle time required")
        self.pending = None
        self.epoch_count += 1
        self.epoch_tokens += tokens
        self.epoch_seconds += seconds
        self.total_tokens[width] += tokens
        self.total_seconds[width] += seconds
        if self.epoch_count == self.epoch_cycles:
            weight = self.retention if self.counts[width] else 0.0
            self.mean_tokens[width] = (weight * self.mean_tokens[width]
                                      + (1 - weight) * self.epoch_tokens / self.epoch_count)
            self.mean_seconds[width] = (weight * self.mean_seconds[width]
                                       + (1 - weight) * self.epoch_seconds / self.epoch_count)
            self.counts[width] += 1
            self.completed_epochs += 1
            self.epoch_count, self.epoch_tokens, self.epoch_seconds = 0, 0, 0.0

    def snapshot(self):
        return {
            "algorithm": "R7 request-local guarded EMA of committed tokens / exposed cycle seconds",
            "widths": self.widths, "preferred": self.preferred, "epoch_cycles": self.epoch_cycles,
            "retention": self.retention, "switch_margin": self.switch_margin, "probe_every": self.probe_every,
            "completed_epochs_by_width": self.counts.copy(), "completed_epochs": self.completed_epochs,
            "mean_tokens": self.mean_tokens.copy(), "mean_seconds": self.mean_seconds.copy(),
            "total_tokens": self.total_tokens.copy(), "total_seconds": self.total_seconds.copy(),
            "incomplete_epoch_cycles": self.epoch_count, "pending": self.pending,
            "optimizer_steps": 0, "model_weight_updates": 0,
            "counterfactual_feedback": False, "unbiased_tps_or_regret_claim": False,
        }


def generate_online(engine, prompt_ids, params, budget, policy):
    """Wrap one batch-1 request, retaining official generate/finalize and kernels.

    No additional CUDA synchronization and no mutation within a decoder cycle.
    The caller must time this ENTIRE function, not just engine.generate.
    """
    if not engine.is_finished() or engine.config.max_num_seqs != 1:
        raise ValueError("native online wrapper requires an idle batch-1 engine")
    if (engine.config.tree_verify_size is not None or engine.config.enforce_eager
            or engine.config.tensor_parallel_size != 1):
        raise ValueError("R7 currently requires graph-enabled, single-GPU linear Uno")
    if policy.pending is not None or policy.completed_epochs or policy.epoch_count:
        raise ValueError("each request requires a fresh policy")
    captured = set(engine.config.cuda_graph_block_sizes)
    if not set(policy.widths) <= captured or max(policy.widths) > engine.config.max_diffusion_block_size:
        raise ValueError("every allowed width must already have a CUDA graph")
    original_width = params.diffusion_block_size
    original_step = engine.step
    missing = object()
    previous_override = engine.__dict__.get("step", missing)
    traces = []
    measured_policy_ns = 0

    def step():
        nonlocal measured_policy_ns
        # Exactly one request, unchunked prefill. Do not learn from prefill time.
        if engine.scheduler.waiting:
            return original_step()
        start = perf_counter_ns()
        width, reason = policy.choose()
        params.diffusion_block_size = width
        dispatch = perf_counter_ns()
        output, token_count = original_step()
        finish = perf_counter_ns()
        if token_count >= 0:
            raise RuntimeError("expected a completed decode step")
        seconds = (finish - start) / 1e9
        policy.observe(width, -token_count, seconds)
        updated = perf_counter_ns()
        measured_policy_ns += dispatch - start + updated - finish
        traces.append({"width": width, "reason": reason, "tokens": -token_count,
                       "exposed_cycle_seconds": seconds})
        return output, token_count

    engine.step = step
    try:
        outputs = engine.generate([prompt_ids], params, use_tqdm=False, request_max_tokens=[budget])
    finally:
        params.diffusion_block_size = original_width
        if previous_override is missing:
            del engine.step
        else:
            engine.step = previous_override
    return outputs[0], {"policy": policy.snapshot(), "cycles": traces,
                        "instrumented_choice_update_seconds": measured_policy_ns / 1e9,
                        "cost_label": "exposed host cycle through official committed-row transfer; includes choice, excludes following update/trace; full function E2E includes all",
                        "additional_cuda_synchronizations": 0}
