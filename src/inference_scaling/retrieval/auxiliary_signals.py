"""Paid candidate-local lookahead with explicit, separate work accounting.

Auxiliary samples never become committed IS tokens. They have independent RNG
streams and retain their exact physical prefixes, including already used evidence.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
from inference_scaling.retrieval.common import _snapshot
from inference_scaling.retrieval.common import token_hash
from inference_scaling.shared.rng import SeedStream


class AuxiliarySampler:
    def __init__(self, backend, tap, seed, close, *, context_tokens, work_tokens=8192):
        if context_tokens < 1 or work_tokens < 1:
            raise ValueError('positive auxiliary limits required')
        self.backend, self.tap, self.seed, self.close = backend, tap, seed, tuple(close)
        self.context_tokens, self.work_tokens = context_tokens, work_tokens
        self.calls, self.work, self.cache = [], 0, {}

    def sample(self, physical, branch, *, purpose, count, tokens):
        physical, branch = tuple(physical), tuple(branch)
        if not purpose or count < 1 or tokens < 1:
            raise ValueError('invalid auxiliary request')
        key = (self.backend.model_id, physical, branch, purpose, count, tokens)
        if key in self.cache:
            return [self.calls[i] for i in self.cache[key]]
        if self.work+count*tokens > self.work_tokens:
            return None  # caller must record budget exhaustion, never a low uncertainty score
        if len(physical)+tokens > self.context_tokens:
            raise ValueError('auxiliary context exceeds model limit')
        sampling = SamplingConfig(eos_token_id=self.backend.tokenizer.eos_token_id)
        prefix_hash, branch_hash = token_hash(physical), token_hash(branch)
        jobs = [GenerationRequest(physical, tokens, sampling,
            SeedStream(self.seed).derive('retrieval_aux', prefix_hash, branch_hash, purpose, i),
            f'aux:{len(self.calls)}:{purpose}:{i}', stop_sequences=(self.close,),
            statistic=TokenStatistic(sampling, top_k=5), intermediate_events=False)
            for i in range(count)]
        before, started = _snapshot(self.backend), time.perf_counter()
        samples = self.backend.sample_batch(jobs)
        seconds, after = time.perf_counter()-started, _snapshot(self.backend)
        if len(samples) != len(jobs):
            raise RuntimeError('incomplete auxiliary batch')
        indices = []
        for i, (job, sample) in enumerate(zip(jobs, samples, strict=True)):
            sc, cn = self.tap.take(job, sample)
            signals = getattr(self.tap, 'history', {}).get(job.request_id)
            row = {'aux_id': len(self.calls), 'purpose': purpose, 'branch_ids': list(branch),
                   'branch_sha256': branch_hash, 'prefix_ids': list(physical), 'prefix_sha256': prefix_hash,
                   'request': asdict(job), 'token_ids': list(sample.token_ids),
                   'lp': list(sample.token_logprobs), 'sc': list(sc), 'cn': list(cn),
                   'finish_reason': sample.finish_reason, 'internal_signals': signals,
                   'batch_index': i, 'batch_size': count, 'batch_seconds': seconds,
                   'batch_cost': {k: after[k]-before[k] for k in before.keys() & after.keys()
                                  if type(before[k]) in (int, float)}}
            indices.append(row['aux_id'])
            self.calls.append(row)
            self.work += len(sample.token_ids)
        # All completed outputs are retained before asserting their validity.
        for job, sample in zip(jobs, samples, strict=True):
            if tuple(sample.prefix) != physical or not 0 < len(sample.token_ids) <= tokens:
                raise RuntimeError('invalid auxiliary prefix or sample length')
        if self.work > self.work_tokens:
            raise RuntimeError('auxiliary backend exceeded budget')
        self.cache[key] = indices
        return [self.calls[i] for i in indices]


def hidden_disagreement(calls, *, tail=4, metric='cosine', ridge=.01):
    """Spread of real independently generated paths, not duplicate prefix states.

These are explicit SeaKR-inspired feature variants, not the paper's exact score.
The final normalized state is projected with a fixed Rademacher matrix upstream.
"""
    import numpy as np
    if len(calls) < 2 or tail < 1 or ridge <= 0:
        raise ValueError('need multiple samples and positive pooling/ridge')
    if metric not in {'cosine', 'variance', 'logvolume'}:
        raise ValueError('unknown disagreement metric')
    if len({c['prefix_sha256'] for c in calls}) != 1:
        raise ValueError('uncertainty samples must have the same physical prefix')
    if len({c['request']['seed'] for c in calls}) != len(calls):
        raise ValueError('uncertainty samples require independent RNG seeds')
    features = []
    for c in calls:
        states = c['internal_signals']
        if states is None or len(states) != len(c['token_ids']):
            raise ValueError('missing aligned hidden states')
        rows = [s['hidden_projection'] for s in states[-tail:]]
        features.append(np.asarray(rows, dtype=np.float64).mean(axis=0))
    matrix = np.stack(features)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('invalid hidden feature matrix')
    paths = len({tuple(c['token_ids']) for c in calls})
    # Batch-layout noise alone is not evidence of epistemic uncertainty.
    if paths == 1:
        value = 0.
    else:
        unit = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        centered = unit-unit.mean(axis=0, keepdims=True)
        if metric == 'cosine':
            pairs = (1-unit@unit.T)[np.triu_indices(len(calls), k=1)]
            value = float(np.clip(pairs, 0., 2.).mean())
        elif metric == 'variance':
            value = float((centered*centered).sum()/len(calls))
        else:
            gram = centered@centered.T/len(calls)
            sign, logdet = np.linalg.slogdet(np.eye(len(calls))+gram/ridge)
            if sign <= 0:
                raise ValueError('invalid regularized hidden Gram matrix')
            value = max(0., float(logdet/len(calls)))
    return {'value': value, 'metric': metric, 'tail': tail, 'ridge': ridge,
            'distinct_paths': paths, 'samples': len(calls), 'feature_dimensions': matrix.shape[1]}


def draft_confidence(call, backend, *, low_probability=.2):
    if not 0 < low_probability <= 1:
        raise ValueError('invalid probability threshold')
    tokens, lp = call['token_ids'], call['lp']
    if not tokens or len(tokens) != len(lp):
        raise ValueError('unaligned draft probabilities')
    cutoff = math.log(low_probability)
    keep = [t for t, p in zip(tokens, lp, strict=True) if p >= cutoff]
    return {'low_fraction': sum(p < cutoff for p in lp)/len(lp),
            'masked_query': backend.decode(tuple(keep), skip_special_tokens=True),
            'retained_token_ids': keep, 'draft_tokens': len(tokens)}
