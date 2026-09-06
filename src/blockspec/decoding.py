"""Independent batch-one AR and two-pass speculative decoding reference paths."""

from dataclasses import dataclass, field
import time

import torch

from .model import trim_cache
from .diffusion import UniformNoise
from .online import Feedback, synchronize
from .sampling import (SamplingConfig, draw, greedy_tokens, probabilities,
                       sample_logits, verify_greedy, verify_linear)


@dataclass
class Generation:
    tokens: list[int]
    seconds: float
    decode_forwards: int
    rounds: int
    accepted: int = 0
    proposed: int = 0
    updates: int = 0
    update_seconds: float = 0.0
    accepted_per_round: list[int] = field(default_factory=list)
    feedback_blocks: int = 0
    fully_covered_rounds: int = 0
    coverage_skips: int = 0

    @property
    def tps(self):
        return len(self.tokens) / self.seconds if self.seconds else 0.0

    def summary(self):
        return {"tokens": len(self.tokens), "seconds": self.seconds, "tps": self.tps,
                "decode_forwards": self.decode_forwards, "rounds": self.rounds,
                "accepted": self.accepted, "proposed": self.proposed,
                "updates": self.updates, "update_seconds": self.update_seconds,
                "feedback_blocks": self.feedback_blocks,
                "fully_covered_rounds": self.fully_covered_rounds, "coverage_skips": self.coverage_skips}


def _check(model, prompt, max_new_tokens, eos_id):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("a nonempty batch-one token prompt is required")
    if max_new_tokens < 0 or (eos_id is not None and not 0 <= eos_id < model.config.vocab_size):
        raise ValueError("invalid output budget or EOS token")


def _inference_forward(model, executor):
    if executor is None:
        return model
    executor.validate(model)
    return executor._forward


def _prefill(forward, prompt):
    if prompt.shape[1] == 1:
        return None
    _, cache = forward(prompt[:, :-1], return_cache=True)
    return trim_cache(cache, prompt.shape[1] - 1)


@torch.no_grad()
def generate_ar(model, prompt, max_new_tokens, *, sampling=SamplingConfig(), eos_id=None,
                generator=None, executor=None, sampler_executor=None):
    _check(model, prompt, max_new_tokens, eos_id)
    forward = _inference_forward(model, executor)
    if sampler_executor is not None:
        sampler_executor.validate(model, sampling)
    synchronize(model)
    start = time.perf_counter()
    cache = _prefill(forward, prompt) if max_new_tokens else None
    seed = prompt[:, -1:]
    output = []
    for _ in range(max_new_tokens):
        logits, cache = forward(seed, cache=cache, return_cache=True)
        token = (sampler_executor.sample_ar(logits[0, -1], generator) if sampler_executor is not None
                 else int(sample_logits(logits[0, -1], sampling, generator)))
        output.append(token)
        seed = prompt.new_tensor([[token]])
        if token == eos_id:
            break
    synchronize(model)
    return Generation(output, time.perf_counter() - start, len(output), len(output))


@torch.no_grad()
def generate_speculative(model, prompt, max_new_tokens, *, block_size=8,
                         sampling=SamplingConfig(), eos_id=None, generator=None,
                         learner=None, executor=None, noise=UniformNoise(), calibrator=None,
                         sampler_executor=None):
    """One clean root + B-1 independent noisy proposals, then base verification.

    cache always describes committed history excluding its last token. A replay
    update runs only AFTER accepting/rejecting with the saved proposal version.
    Replay learners skip end-of-request updates. Sparse calibration accumulators
    persist across requests; each update consumes the original proposal version.
    """
    _check(model, prompt, max_new_tokens, eos_id)
    forward = _inference_forward(model, executor)
    if sampler_executor is not None:
        sampler_executor.validate(model, sampling, block_size, calibrator)
    if block_size < 2:
        raise ValueError("speculative blocks require B>=2; use generate_ar for B=1")
    if learner is not None and learner.model is not model:
        raise ValueError("learner and decoder must share the same model")
    if calibrator is not None and (learner is not None or sampling.temperature <= 0
                                   or sampling.top_k != calibrator.top_k
                                   or block_size != calibrator.block_size):
        raise ValueError("calibration requires matched positive-temperature top-k sampling and a frozen adapter")
    synchronize(model)
    start = time.perf_counter()
    calibration_initial = ((calibrator.updates, calibrator.update_seconds, calibrator.feedback_blocks)
                           if calibrator is not None else (0, 0., 0))
    continuation = getattr(calibrator, "kind", None) == "continuation"
    if continuation:
        calibrator.begin_request(prompt[0].tolist())
    initial_updates = learner.updates if learner is not None else 0
    initial_update_seconds = learner.update_seconds if learner is not None else 0.0
    initial_feedback = learner.feedback_blocks if learner is not None else 0
    initial_skips = learner.coverage_skips if learner is not None else 0
    if learner is not None:
        learner.clear_replay()
    cache = _prefill(forward, prompt) if max_new_tokens else None
    seed = prompt[:, -1:]
    output, accepts = [], []
    rounds = forwards = accepted = proposed = 0
    fully_covered_rounds = 0
    while len(output) < max_new_tokens:
        rounds += 1
        remaining = max_new_tokens - len(output)
        if remaining == 1:
            logits, cache = forward(seed, cache=cache, return_cache=True)
            output.append(sampler_executor.sample_ar(logits[0, -1], generator) if sampler_executor is not None
                          else int(sample_logits(logits[0, -1], sampling, generator)))
            if continuation:
                calibrator.commit(output[-1:])
            forwards += 1
            break
        b = min(block_size, remaining)
        noisy = noise.sample((1, b - 1), model.config.vocab_size, device=prompt.device, generator=generator)
        inputs = torch.cat((seed, noisy), dim=1)
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[:, 0] = False
        old_cache = cache
        collect = learner is not None and learner.needs_decoder_feedback
        capture = learner.capture_layer if collect else None
        result = forward(inputs, cache=cache, adapter_mask=mask, return_cache=True, capture_layer=capture)
        draft, temporary_cache = result[:2]
        boundary = result[2] if capture is not None else None
        forwards += 1
        if sampler_executor is not None:
            candidates, draft_distribution, calibration_feedback = sampler_executor.draft(draft[0], generator, calibrator)
        elif sampling.temperature == 0:
            candidates = greedy_tokens(draft[0])
        else:
            draft_distribution = probabilities(draft[0], sampling)
            if continuation:
                candidates, draft_distribution, calibration_feedback = calibrator.draft(draft_distribution, generator)
            elif calibrator is not None:
                draft_distribution, calibration_feedback = calibrator.propose(draft_distribution)
                candidates = draw(draft_distribution, generator)
            else:
                candidates = draw(draft_distribution, generator)
        root = int(candidates[0])
        if root == eos_id:
            output.append(root)
            if continuation:
                calibrator.commit([root])
            break
        # Only the seed's KV is clean. No adapted/noisy KV is ever committed.
        clean_length = prompt.shape[1] + len(output)
        clean_cache = trim_cache(temporary_cache, clean_length)
        teacher, verified_cache = forward(candidates[None], cache=clean_cache, return_cache=True)
        forwards += 1
        if sampler_executor is not None:
            verified, target = sampler_executor.verify(candidates[1:], draft_distribution[1:], teacher[0], generator)
        elif sampling.temperature == 0:
            verified = verify_greedy(candidates[1:], greedy_tokens(teacher[0]))
        else:
            target = probabilities(teacher[0], sampling)
            verified = verify_linear(candidates[1:], draft_distribution[1:], target, generator=generator)
        committed = [root] + verified.tokens
        if eos_id is not None and eos_id in committed:
            committed = committed[:committed.index(eos_id) + 1]
        committed = committed[:remaining]
        # Count only noise decisions corresponding to tokens actually returned.
        used = min(verified.supervised, max(0, len(committed) - 1))
        kept = min(verified.accepted, max(0, len(committed) - 1))
        accepted += kept
        proposed += b - 1
        accepts.append(kept)
        fully_covered = kept == b - 1
        fully_covered_rounds += fully_covered
        output.extend(committed)
        if continuation:
            calibrator.commit(committed)
        cache = trim_cache(verified_cache, prompt.shape[1] + len(output) - 1)
        seed = prompt.new_tensor([[output[-1]]])
        done = len(output) >= max_new_tokens or output[-1] == eos_id
        if learner is not None:
            if collect:
                feedback = Feedback(inputs, old_cache, teacher[0, :used], used, boundary, fully_covered)
                learner.observe(feedback, may_update=not done)
            else:
                learner._skip_decoder_feedback(used)
        if calibrator is not None:
            if continuation:
                calibrator.observe(calibration_feedback, target[:used], root=root)
            else:
                calibrator.observe(calibration_feedback, target[:used])
        if done:
            break
    if learner is not None:
        learner.clear_replay()
    synchronize(model)
    elapsed = time.perf_counter() - start
    if calibrator is not None:
        return Generation(output, elapsed, forwards, rounds, accepted, proposed,
                          calibrator.updates - calibration_initial[0],
                          calibrator.update_seconds - calibration_initial[1], accepts,
                          calibrator.feedback_blocks - calibration_initial[2], fully_covered_rounds)
    return Generation(output, elapsed, forwards, rounds, accepted, proposed,
                      (learner.updates - initial_updates) if learner is not None else 0,
                      (learner.update_seconds - initial_update_seconds) if learner is not None else 0.0,
                      accepts, learner.feedback_blocks - initial_feedback if learner is not None else 0,
                      fully_covered_rounds, learner.coverage_skips - initial_skips if learner is not None else 0)
