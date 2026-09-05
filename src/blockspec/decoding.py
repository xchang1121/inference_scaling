"""Independent batch-one AR and two-pass speculative decoding reference paths."""

from dataclasses import dataclass, field
import time

import torch

from .model import trim_cache
from .online import Feedback, synchronize
from .sampling import SamplingConfig, draw, probabilities, verify_linear


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

    @property
    def tps(self):
        return len(self.tokens) / self.seconds if self.seconds else 0.0

    def summary(self):
        return {"tokens": len(self.tokens), "seconds": self.seconds, "tps": self.tps,
                "decode_forwards": self.decode_forwards, "rounds": self.rounds,
                "accepted": self.accepted, "proposed": self.proposed,
                "updates": self.updates, "update_seconds": self.update_seconds}


def _check(model, prompt, max_new_tokens, eos_id):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("a nonempty batch-one token prompt is required")
    if max_new_tokens < 0 or (eos_id is not None and not 0 <= eos_id < model.config.vocab_size):
        raise ValueError("invalid output budget or EOS token")


def _prefill(model, prompt):
    if prompt.shape[1] == 1:
        return None
    _, cache = model(prompt[:, :-1], return_cache=True)
    return trim_cache(cache, prompt.shape[1] - 1)


@torch.no_grad()
def generate_ar(model, prompt, max_new_tokens, *, sampling=SamplingConfig(), eos_id=None,
                generator=None):
    _check(model, prompt, max_new_tokens, eos_id)
    synchronize(model)
    start = time.perf_counter()
    cache = _prefill(model, prompt) if max_new_tokens else None
    seed = prompt[:, -1:]
    output = []
    for _ in range(max_new_tokens):
        logits, cache = model(seed, cache=cache, return_cache=True)
        token = int(draw(probabilities(logits[0, -1], sampling), generator))
        output.append(token)
        seed = prompt.new_tensor([[token]])
        if token == eos_id:
            break
    synchronize(model)
    return Generation(output, time.perf_counter() - start, len(output), len(output))


@torch.no_grad()
def generate_speculative(model, prompt, max_new_tokens, *, block_size=8,
                         sampling=SamplingConfig(), eos_id=None, generator=None,
                         learner=None):
    """One clean root + B-1 independent noisy proposals, then base verification.

    cache always describes committed history excluding its last token. A replay
    update runs only AFTER accepting/rejecting with the saved proposal version.
    End-of-request updates are skipped; weights/optimizer persist, replay does not.
    """
    _check(model, prompt, max_new_tokens, eos_id)
    if block_size < 2:
        raise ValueError("speculative blocks require B>=2; use generate_ar for B=1")
    if learner is not None and learner.model is not model:
        raise ValueError("learner and decoder must share the same model")
    synchronize(model)
    start = time.perf_counter()
    initial_updates = learner.updates if learner is not None else 0
    initial_update_seconds = learner.update_seconds if learner is not None else 0.0
    if learner is not None:
        learner.clear_replay()
    cache = _prefill(model, prompt) if max_new_tokens else None
    seed = prompt[:, -1:]
    output, accepts = [], []
    rounds = forwards = accepted = proposed = 0
    while len(output) < max_new_tokens:
        rounds += 1
        remaining = max_new_tokens - len(output)
        if remaining == 1:
            logits, cache = model(seed, cache=cache, return_cache=True)
            output.append(int(draw(probabilities(logits[0, -1], sampling), generator)))
            forwards += 1
            break
        b = min(block_size, remaining)
        noise = torch.randint(model.config.vocab_size, (1, b - 1), device=prompt.device,
                              generator=generator)
        inputs = torch.cat((seed, noise), dim=1)
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask[:, 0] = False
        old_cache = cache
        draft, temporary_cache = model(inputs, cache=cache, adapter_mask=mask, return_cache=True)
        forwards += 1
        draft_distribution = probabilities(draft[0], sampling)
        candidates = draw(draft_distribution, generator)
        root = int(candidates[0])
        if root == eos_id:
            output.append(root)
            break
        # Only the seed's KV is clean. No adapted/noisy KV is ever committed.
        clean_length = prompt.shape[1] + len(output)
        clean_cache = trim_cache(temporary_cache, clean_length)
        teacher, verified_cache = model(candidates[None], cache=clean_cache, return_cache=True)
        forwards += 1
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
        output.extend(committed)
        cache = trim_cache(verified_cache, prompt.shape[1] + len(output) - 1)
        seed = prompt.new_tensor([[output[-1]]])
        done = len(output) >= max_new_tokens or output[-1] == eos_id
        if learner is not None:
            feedback = Feedback(inputs, old_cache, teacher[0, :used], used)
            learner.observe(feedback, may_update=not done)
        if done:
            break
    if learner is not None:
        learner.clear_replay()
    synchronize(model)
    elapsed = time.perf_counter() - start
    return Generation(output, elapsed, forwards, rounds, accepted, proposed,
                      (learner.updates - initial_updates) if learner is not None else 0,
                      (learner.update_seconds - initial_update_seconds) if learner is not None else 0.0,
                      accepts)
