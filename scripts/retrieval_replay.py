"""Shared logical IS replay; no inference or experiment dispatch."""
import math
from types import SimpleNamespace
from inference_scaling.arllm.types import SequenceSample
from inference_scaling.retrieval.cot_guidance import GuidanceLedger
from inference_scaling.shared.rng import SeedStream

def canonical(value):
    """Compare semantic search records, excluding hardware-only slot counters."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if k != 'forward_token_slots'}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def same_search_record(left, right):
    """Tokens/choices are exact; tolerate platform libm rounding of weights."""
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-10)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_search_record(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_search_record(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


class ReplayLedger(GuidanceLedger):
    def __init__(self, row, cfg, eos, close):
        backend = SimpleNamespace(model_id='offline', tokenizer=SimpleNamespace(eos_token_id=eos))
        seed = SeedStream(SeedStream(row['seed']).derive('question', row['question_id'])).derive('search')
        super().__init__(backend, None, row['root_ids'], (close,), seed, cfg)
        self.recorded = iter(row['calls'])

    def sample_batch(self, requests):
        samples = []
        for r in requests:
            c = next(self.recorded)
            assert (list(r.prefix), r.seed, r.uniform_offset, r.max_new_tokens, r.request_id) == (
                c['prefix_ids'], c['seed'], c['uniform_offset'], c['max_new_tokens'], c['request_id'])
            self.calls.append(c)
            self.work += len(c['token_ids'])
            samples.append(SequenceSample(tuple(r.prefix), tuple(c['token_ids']), tuple(c['lp']),
                r.sampling.policy_id, self.model_id, r.request_id, c['finish_reason']))
        return samples
