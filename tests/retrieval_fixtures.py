"""Deterministic CPU-only native backend and evidence fixtures."""
from types import SimpleNamespace
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, SequenceSample

class Native:
    model_id = 'test-online'
    tokenizer = SimpleNamespace(eos_token_id=900, convert_tokens_to_ids=lambda s: 902 if s == '<think>' else 901)

    def __init__(self, stop=9, eos=False):
        self.stop, self.eos = stop, eos
        self.requests, self.stats = [], {}

    def encode(self, text, **kwargs):
        return (10,) if text == '\n' else tuple(30+ord(c) % 20 for c in text)

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))

    def sample_batch(self, requests):
        self.requests.extend(requests)
        samples = []
        for r in requests:
            physical, ids, sc = list(r.prefix), [], []
            for i in range(r.max_new_tokens):
                count = sum(100 <= t < 200 for t in physical)
                if count >= self.stop:
                    token = 900 if self.eos else 901
                else:
                    # Changing injected context changes both sampling and reward.
                    token = 100+(r.seed+r.uniform_offset+i+sum(physical)) % 73
                sc.append(1+sum(physical) % 17/10)
                physical.append(token);ids.append(token)
                if token in {900, 901}:
                    break
            sample = SequenceSample(tuple(r.prefix), tuple(ids), (-1.,)*len(ids), r.sampling.policy_id,
                                    self.model_id, r.request_id, 'eos' if ids[-1] == 900 else 'stop' if ids[-1] == 901 else 'length')
            self.stats[r.request_id] = tuple(sc), (3.,)*len(ids)
            samples.append(sample)
        return samples

    def take(self, request, sample):
        return self.stats.pop(request.request_id)


class Facts:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, *, k=1, exclude=frozenset()):
        self.queries.append((query, exclude))
        key = sum(map(ord, query)) % 19
        name = str(key)+'-'+str(len(exclude))
        payload = {'ids': [20, 30+key, 21], 'body_start': 1, 'body_stop': 2,
                   'id': name, 'source': 'test', 'score': 1.}
        return [payload], {'query': query, 'selected_ids': [name], 'seconds': 0.}


def request(ledger, prefix=(), count=12, seed=7, name='test', offset=0):
    return GenerationRequest(ledger.root+tuple(prefix), count, SamplingConfig(eos_token_id=900), seed, name,
                             uniform_offset=offset)


class AnswerNative(Native):
    def sample_batch(self, requests):
        if all(901 in r.prefix for r in requests):
            self.requests.extend(requests)
            return [SequenceSample(tuple(r.prefix), (70, 900), (-1., -1.),
                r.sampling.policy_id, self.model_id, r.request_id, 'eos') for r in requests]
        return super().sample_batch(requests)


class ReadableNative(Native):
    def encode(self, text, **kwargs):
        return (10,) if text == '\n' else tuple(3000+ord(c) for c in text)

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(t-3000) if t >= 3000 else '\n' if t == 10 else
                       '' if t in {900, 901} and skip_special_tokens else 'x ' for t in ids)
