"""Experiment-local, read-only diagnostics of the native processed distribution.

Four tagged slots travel through vLLM's own per-request logprob alignment. The
original SC row is decoded unchanged after stripping these tags. This adapter
supports top-5 requests only, and deliberately fails closed on other formats.
"""
from __future__ import annotations

from math import isfinite, log
import types

SIGNAL_TAGS = {-2: 'entropy', -3: 'top2_gap', -4: 'p_eos', -5: 'p_think_end'}
DIAGNOSTIC_LOGPROBS = 10  # top 5 + SC + four diagnostics


def install_distribution_signals(sampler, vocabulary_size, eos_id, close_id, hidden_dimensions=0, attention_topk=0):
    import torch
    if getattr(sampler, '_retrieval_signals_installed', False):
        raise RuntimeError('diagnostics already installed')
    original = sampler.gather_logprobs
    attention_dimensions = 2*attention_topk+3 if attention_topk else 0

    def gather(logprobs, num_logprobs, token_ids):
        if num_logprobs != DIAGNOSTIC_LOGPROBS+hidden_dimensions+attention_dimensions:
            if num_logprobs not in (0, 1, 6):
                raise ValueError('unsupported diagnostic worker row width')
            return original(logprobs, num_logprobs, token_ids)
        if logprobs.shape[-1] != vocabulary_size:
            raise RuntimeError('diagnostics require the complete unpadded vocabulary')
        result = original(logprobs, num_logprobs - len(SIGNAL_TAGS)-hidden_dimensions-attention_dimensions, token_ids)
        lp = logprobs.float()
        probabilities = lp.exp()
        entropy = -(probabilities * lp).sum(dim=-1, keepdim=True)
        top2 = lp.topk(2, dim=-1).values.exp()
        values = torch.cat((entropy, top2[:, :1] - top2[:, 1:2],
                            probabilities[:, eos_id:eos_id+1],
                            probabilities[:, close_id:close_id+1]), dim=-1)
        tags = torch.tensor(list(SIGNAL_TAGS), dtype=result.logprob_token_ids.dtype,
                            device=lp.device).expand(lp.shape[0], -1)
        if hidden_dimensions:
            hidden = getattr(sampler, '_retrieval_hidden_projection', None)
            if hidden is None or tuple(hidden.shape) != (lp.shape[0], hidden_dimensions):
                raise RuntimeError('hidden/logit row alignment failure')
            hidden_tags = -1000-torch.arange(hidden_dimensions, dtype=tags.dtype, device=lp.device)
            tags = torch.cat((tags, hidden_tags.expand(lp.shape[0], -1)), dim=-1)
            values = torch.cat((values, hidden), dim=-1)
        if attention_dimensions:
            attention = getattr(sampler, '_retrieval_attention', None)
            if attention is None or tuple(attention.shape) != (lp.shape[0], attention_dimensions):
                raise RuntimeError('attention/logit row alignment failure')
            attention_tags = -2000-torch.arange(attention_dimensions,dtype=tags.dtype,device=lp.device)
            tags = torch.cat((tags,attention_tags.expand(lp.shape[0],-1)),dim=-1)
            values = torch.cat((values,attention),dim=-1)
        # Tags precede top-K so native truncation preserves the private payload.
        return result._replace(
            logprob_token_ids=torch.cat((result.logprob_token_ids[:, :1], tags,
                                         result.logprob_token_ids[:, 1:]), dim=-1),
            logprobs=torch.cat((result.logprobs[:, :1], values, result.logprobs[:, 1:]), dim=-1))

    sampler.gather_logprobs = gather
    sampler._retrieval_signals_installed = True


def unpack_signals(positions, tokens, vocabulary_size, hidden_dimensions=0, attention_topk=0):
    if positions is None or len(positions) != len(tokens):
        raise RuntimeError('diagnostic row length mismatch')
    flat = hasattr(positions, 'start_indices')
    clean, signals = [], []
    for i, token in enumerate(tokens):
        if flat:
            a, b = positions.start_indices[i], positions.end_indices[i]
            row = dict(zip(positions.token_ids[a:b], positions.logprobs[a:b], strict=True))
        else:
            row = {k: float(getattr(v, 'logprob', v)) for k, v in positions[i].items()}
        if not set(SIGNAL_TAGS).issubset(row) or token not in row or -1 not in row:
            raise RuntimeError('missing or misaligned diagnostic tags')
        signal = {name: float(row.pop(tag)) for tag, name in SIGNAL_TAGS.items()}
        if not all(isfinite(v) for v in signal.values()):
            raise RuntimeError('nonfinite internal signal')
        if not -1e-5 <= signal['entropy'] <= log(vocabulary_size)+1e-4:
            raise RuntimeError('invalid entropy')
        if any(not -1e-5 <= signal[k] <= 1+1e-5 for k in ('top2_gap', 'p_eos', 'p_think_end')):
            raise RuntimeError('invalid probability diagnostic')
        if hidden_dimensions:
            hidden = [float(row.pop(-1000-j)) for j in range(hidden_dimensions)]
            if not all(isfinite(v) for v in hidden):
                raise RuntimeError('invalid hidden projection')
            signal['hidden_projection'] = hidden
        if attention_topk:
            values = [float(row.pop(-2000-j)) for j in range(2*attention_topk+3)]
            if not all(isfinite(v) for v in values):
                raise RuntimeError('nonfinite attention diagnostic')
            indices, weights = values[:attention_topk],values[attention_topk:2*attention_topk]
            length,error,normalization = values[-3:]
            if length != int(length) or length < 1 or any(v != int(v) or not 0 <= v < length for v in indices):
                raise RuntimeError('invalid attention token indices')
            if any(not 0 <= v <= 1+1e-5 for v in weights) or sum(weights) > 1+1e-4:
                raise RuntimeError('invalid attention weights')
            if not 0 <= error < .03 or not 0 <= normalization < 1e-4:
                raise RuntimeError('attention reconstruction failed')
            signal['attention'] = {'indices':[int(v) for v in indices], 'weights':weights,
                                   'prefix_tokens':int(length), 'relative_output_error':error,
                                   'normalization_error':normalization, 'layer':63}
        if any(k < -1 for k in row):
            raise RuntimeError('unknown diagnostic slot')
        clean.append(row)
        signals.append(signal)
    return clean, tuple(signals)


class InternalSignals:
    """Install before DualStatistics. Toggle only between completed batches."""
    def __init__(self, backend, hidden_dimensions=0, attention_topk=0):
        self.backend, self.enabled, self.rows = backend, False, {}
        self.params = backend._sampling_params
        self.parse = backend._sample_from_output
        self.pending = set()

        def params(instance, request):
            result = self.params(request)
            if self.enabled and request.statistic is not None:
                if request.statistic.top_k != 5 or result.logprobs != 6:
                    raise ValueError('diagnostics require top-5 plus scalar SC')
                result.logprobs = DIAGNOSTIC_LOGPROBS+hidden_dimensions+(2*attention_topk+3 if attention_topk else 0)
                self.pending.add(request.request_id)
            return result

        def parse(instance, request, output, fused=None):
            if request.request_id in self.pending:
                completion = instance._completion(output)
                clean, signals = unpack_signals(completion.logprobs, completion.token_ids,
                                                len(instance.tokenizer), hidden_dimensions, attention_topk)
                if attention_topk and any(s['attention']['prefix_tokens'] != len(request.prefix)+j
                                          for j,s in enumerate(signals)):
                    raise RuntimeError('attention prefix/request alignment failure')
                completion.logprobs = clean
                self.rows[request.request_id] = (tuple(request.prefix), tuple(completion.token_ids), signals)
                self.pending.remove(request.request_id)
            return self.parse(request, output, fused)

        backend._sampling_params = types.MethodType(params, backend)
        backend._sample_from_output = types.MethodType(parse, backend)

    def take(self, request, sample):
        prefix, tokens, values = self.rows.pop(request.request_id)
        if prefix != tuple(request.prefix) or tokens != tuple(sample.token_ids):
            raise RuntimeError('internal signal prefix/token mismatch')
        return values

    def close(self):
        self.backend._sampling_params = self.params
        self.backend._sample_from_output = self.parse


def select_signal_worker(*, hidden=False, attention=False):
    """Opt-in scoped process entry point; no production source is modified."""
    from inference_scaling.arllm.backends import vllm_backend
    if hidden and attention:
        raise ValueError('validate hidden and attention workers separately first')
    cls = 'AttentionSignalWorker' if attention else 'HiddenSignalWorker' if hidden else 'RetrievalSignalWorker'
    vllm_backend._ASCEND_WORKER = 'inference_scaling.retrieval.internal_signals_worker.'+cls
