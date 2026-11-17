"""CPU-only decoding and quality metrics for immutable retrieval traces."""
from collections import Counter
import gzip
import json
import random
import re
import statistics

class Decoder:
    def __init__(self, vocabulary):
        assert vocabulary['decoder']['type']=='ByteLevel'
        alphabet=list(range(33,127))+list(range(161,173))+list(range(174,256))
        codes,extra=alphabet.copy(),0
        for byte in range(256):
            if byte not in alphabet:
                alphabet.append(byte);codes.append(256+extra);extra+=1
        self.mapping=dict(zip(map(chr,codes),alphabet,strict=True))
        self.special=set(vocabulary['special_ids'])
        self.vocab=vocabulary['vocab']

    def __call__(self, ids, skip=False):
        encoded=''.join(self.vocab[str(t)] for t in ids if not skip or t not in self.special)
        return bytes(self.mapping[c] for c in encoded).decode('utf-8',errors='replace')


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def grade(answer, target):
    def normalize(text):
        return re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", text.lower())).split()
    a, b = normalize(answer), normalize(target)
    shared = sum((Counter(a) & Counter(b)).values())
    return {"f1": 2*shared/(len(a)+len(b)) if a or b else 1., "exact_match": a == b}


def bootstrap(values):
    values = list(values)
    if not values: return None
    rng = random.Random(20261109)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return {"difference": mean(values), "bootstrap_95": [samples[124], samples[4874]], "n_questions": len(values)}
