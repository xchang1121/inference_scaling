"""Branch-local retrieval inside the unchanged conditional-IS generation backend.

IS sees only model-generated tokens. The native model sees those tokens plus a
deterministic evidence insertion at configured generated-token positions. Thus
the very first generated token belongs to the IS candidate, not to a controller
prefix. No proposal, block, rollout, reward, or selection rule is modified.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

from inference_scaling.arllm.types import SequenceSample, TokenStatistic
from inference_scaling.retrieval.cot_guidance import GuidanceLedger
from inference_scaling.retrieval.literature import encode, unrelated
from inference_scaling.retrieval.stepwise import token_hash


@dataclass(frozen=True)
class OnlineRetrievalConfig:
    positions: tuple[int, ...] = (128, 512)
    query_tail_tokens: int = 256
    packet_tokens: int = 256

    def __post_init__(self):
        if not self.positions or tuple(sorted(set(self.positions))) != self.positions or self.positions[0] < 1:
            raise ValueError('retrieval positions must be positive and strictly increasing')
        if self.query_tail_tokens < 1 or self.packet_tokens < 1:
            raise ValueError('positive retrieval limits required')


def fixed_packet(backend, payload, size):
    """Keep the factual body, padding its tail with newlines to a fixed size."""
    p = dict(payload)
    head = tuple(p['ids'][:p['body_start']])
    tail = tuple(p['ids'][p['body_stop']:])
    width = size-len(head)-len(tail)
    if width < 1:
        raise ValueError('packet limit cannot fit wrapper and body')
    body = tuple(p['ids'][p['body_start']:p['body_stop']])[:width]
    newline = encode(backend, '\n')
    if len(newline) != 1:
        raise ValueError('fixed padding requires one newline token')
    ids = head+body+newline*(width-len(body))+tail
    p.update(ids=list(ids), body_start=len(head), body_stop=len(head)+width,
             sha256=token_hash(ids), original_body_tokens=p['body_stop']-p['body_start'])
    return p


class OnlineISLedger(GuidanceLedger):
    """Expose a generated-token-only policy to the frozen IS implementation.

Evidence is a deterministic function of a branch's generated prefix. Cache keys
contain that entire prefix, so a different candidate cannot inherit its sibling's
retrieval. Reused complete IS trajectories retain the same evidence and scores.
Native calls are segmented exactly at insertion positions, without overshoot or
discarded text. The RNG uniform offset advances only for generated tokens.
"""

    def __init__(self, backend, tap, root, close, seed, cfg, *, question, facts=None,
                 noise=(), mode='closed', retrieval_cfg=OnlineRetrievalConfig(), cap=None):
        if mode not in {'closed', 'relevant', 'noise'}:
            raise ValueError('unknown online retrieval mode')
        if mode != 'closed' and facts is None:
            raise ValueError('online modes require a frozen fact bank')
        if mode == 'noise' and not noise:
            raise ValueError('noise tokens required')
        if len(close) != 1:
            raise ValueError('frozen online IS requires one thinking-close token')
        super().__init__(backend, tap, root, close, seed, cfg, cap=cap)
        self.question, self.facts = question, facts
        self.noise, self.mode, self.retrieval_cfg = tuple(noise), mode, retrieval_cfg
        self.events, self.native_calls, self._event_cache = [], [], {}
        self.materialization_seconds = 0.0

    def materialize(self, generated, *, pending=True):
        """Physical prompt and event IDs determined only by this logical prefix."""
        generated = tuple(generated)
        if self.mode == 'closed':
            return self.root+generated, []
        stop = next((i for i, t in enumerate(generated) if t in {self.eos, self.close[0]}), len(generated))
        physical, event_ids, previous, used = list(self.root), [], 0, set()
        for position in self.retrieval_cfg.positions:
            if position > stop or (not pending and position == len(generated)):
                break
            branch = generated[:position]
            key = token_hash(branch)
            if key not in self._event_cache:
                tail = branch[-self.retrieval_cfg.query_tail_tokens:]
                query = self.question+'\n'+self.backend.decode(tail, skip_special_tokens=True).strip()
                found, retrieval = self.facts.retrieve(query, k=1, exclude=frozenset(used))
                if not found:
                    raise RuntimeError('no unused evidence window returned')
                payload = fixed_packet(self.backend, found[0], self.retrieval_cfg.packet_tokens)
                injected = unrelated(payload, self.noise) if self.mode == 'noise' else tuple(payload['ids'])
                forbidden = {self.eos, self.close[0], self.backend.tokenizer.convert_tokens_to_ids('<think>')}
                if forbidden.intersection(injected):
                    raise ValueError('structural token in evidence')
                event = {'event_id': len(self.events), 'position': position, 'branch_ids': list(branch),
                         'branch_sha256': key, 'query': query, 'packet': payload,
                         'injected_ids': list(injected), 'retrieval': retrieval}
                self._event_cache[key] = event
                self.events.append(event)
            event = self._event_cache[key]
            if event['position'] != position:
                raise RuntimeError('prefix hash collision')
            physical.extend(generated[previous:position])
            physical.extend(event['injected_ids'])
            event_ids.append(event['event_id'])
            used.add(event['packet']['id'])
            previous = position
        physical.extend(generated[previous:])
        return tuple(physical), event_ids

    def sample_batch(self, requests):
        if sum(r.max_new_tokens for r in requests) > self.cap-self.work:
            raise RuntimeError('incomplete candidate pool would exceed search cap')
        states = []
        for request in requests:
            if tuple(request.prefix[:len(self.root)]) != self.root:
                raise RuntimeError('logical root changed inside IS')
            if request.draft is not None or request.log_weight_stop is not None:
                raise ValueError('unsupported draft/rejection request in frozen IS configuration')
            states.append({'request': request, 'existing': tuple(request.prefix[len(self.root):]),
                           'ids': [], 'lp': [], 'sc': [], 'cn': [], 'native_indices': [],
                           'finish': None, 'slots': 0, 'measured': True})
        while any(s['finish'] is None for s in states):
            active, jobs = [], []
            for state in states:
                if state['finish'] is not None:
                    continue
                request = state['request']
                prefix = state['existing']+tuple(state['ids'])
                started = time.perf_counter()
                physical, event_ids = self.materialize(prefix)
                self.materialization_seconds += time.perf_counter()-started
                length = request.max_new_tokens-len(state['ids'])
                if self.mode != 'closed':
                    next_position = next((p for p in self.retrieval_cfg.positions if p > len(prefix)), None)
                    if next_position is not None:
                        length = min(length, next_position-len(prefix))
                if len(physical)+length+len(self.close)+self.cfg.answer_tokens > self.cfg.context_tokens:
                    raise RuntimeError('insufficient context reserved for complete answer')
                job = replace(request, prefix=physical, max_new_tokens=length,
                              request_id=f'{request.request_id}:native:{len(state["native_indices"])}',
                              uniform_offset=request.uniform_offset+len(state['ids']),
                              stop_sequences=(self.close,), intermediate_events=False,
                              statistic=TokenStatistic(request.sampling, top_k=5))
                active.append((state, prefix, event_ids));jobs.append(job)
            samples = self.backend.sample_batch(jobs)
            if len(samples) != len(jobs):
                raise RuntimeError('missing native sample')
            for (state, logical, events), job, sample in zip(active, jobs, samples, strict=True):
                if tuple(sample.prefix) != tuple(job.prefix) or not 0 < len(sample.token_ids) <= job.max_new_tokens:
                    raise RuntimeError('native prefix or token-count mismatch')
                sc, cn = self.tap.take(job, sample)
                if not (len(sc) == len(cn) == len(sample.token_ids)):
                    raise RuntimeError('native statistic alignment failure')
                if sample.finish_reason == 'length' and len(sample.token_ids) != job.max_new_tokens:
                    raise RuntimeError('native generation stopped short without a stop reason')
                state['native_indices'].append(len(self.native_calls))
                self.native_calls.append({'request_id': job.request_id, 'logical_request_id': state['request'].request_id,
                    'logical_prefix_ids': list(self.root+logical), 'prefix_ids': list(job.prefix),
                    'prefix_sha256': token_hash(job.prefix), 'event_ids': events,
                    'seed': job.seed, 'uniform_offset': job.uniform_offset,
                    'max_new_tokens': job.max_new_tokens, 'token_ids': list(sample.token_ids),
                    'lp': list(sample.token_logprobs), 'sc': list(sc), 'cn': list(cn),
                    'finish_reason': sample.finish_reason})
                for key, values in [('ids', sample.token_ids), ('lp', sample.token_logprobs), ('sc', sc), ('cn', cn)]:
                    state[key].extend(values)
                self.work += len(sample.token_ids)
                if sample.forward_token_slots is None:
                    state['measured'] = False
                else:
                    state['slots'] += sample.forward_token_slots
                terminal = sample.finish_reason != 'length' or any(t in {self.eos, self.close[0]} for t in sample.token_ids)
                if terminal or len(state['ids']) == state['request'].max_new_tokens:
                    state['finish'] = sample.finish_reason
        outputs = []
        for state in states:
            request = state['request']
            self.calls.append({'request_id': request.request_id, 'seed': request.seed,
                'uniform_offset': request.uniform_offset, 'prefix_ids': list(request.prefix),
                'prefix_sha256': token_hash(request.prefix), 'max_new_tokens': request.max_new_tokens,
                'token_ids': state['ids'], 'lp': state['lp'], 'sc': state['sc'], 'cn': state['cn'],
                'finish_reason': state['finish'], 'native_indices': state['native_indices']})
            outputs.append(SequenceSample(tuple(request.prefix), tuple(state['ids']), tuple(state['lp']),
                request.sampling.policy_id, self.model_id, request.request_id, state['finish'],
                forward_token_slots=state['slots'] if state['measured'] else None))
        return outputs


def run_online_is(backend, tap, root, close, seed, cfg, *, question,
                  facts=None, noise=(), mode='relevant',
                  retrieval_cfg=OnlineRetrievalConfig()):
    """Run frozen IS from token zero, then read out from the selected context."""
    from inference_scaling.retrieval.cot_guidance import finish_answer, search
    from inference_scaling.shared.rng import SeedStream

    ledger = OnlineISLedger(backend, tap, root, close,
        SeedStream(seed).derive('search'), cfg, question=question, facts=facts,
        noise=noise, mode=mode, retrieval_cfg=retrieval_cfg)
    started = time.perf_counter()
    candidates, audit = search(ledger, 'is_block_sc')
    search_seconds = time.perf_counter()-started
    selected = candidates[0]
    # Final readout cannot initiate an insertion that no thinking token consumed.
    physical, selected_events = ledger.materialize(selected, pending=False)
    started = time.perf_counter()
    answer = finish_answer(backend, root, physical[len(root):], close, cfg,
        SeedStream(seed).derive('answer_suffix', token_hash(selected)))
    return {'root_ids': list(root), 'carried': ledger.carried,
        'candidates': candidates, 'calls': ledger.calls, 'native_calls': ledger.native_calls,
        'events': ledger.events, 'selected_event_ids': selected_events,
        'selected_physical_ids': list(physical), 'presearch_work': 0,
        'search_work': ledger.work,
        'total_generation_tokens': ledger.work+len(answer['answer_ids']),
        'answer': answer, 'audit': audit,
        'stage_seconds': {'search_seconds': search_seconds,
            'materialization_seconds': ledger.materialization_seconds,
            'answer_seconds': time.perf_counter()-started}}
