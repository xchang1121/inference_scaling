"""Experimental retrieval policies around the unchanged IS kernel.

All decisions use the question, generated prefix and already measured token
probabilities. No gold answers, auxiliary LLM or model tool calls are used.
"""
from __future__ import annotations

import math
import re
import time

from inference_scaling.retrieval.literature import encode
from inference_scaling.retrieval.online_is import OnlineISLedger, OnlineRetrievalConfig, fixed_packet
from inference_scaling.retrieval.stepwise import token_hash

VARIANTS = ('resume', 'anchored', 'boundary', 'adaptive', 'bridge', 'early')
FIXED_VARIANTS = {'resume', 'anchored', 'bridge', 'early'}
SIMPLE_QUERY_VARIANTS = {'resume', 'bridge', 'early'}
CHECKPOINTS = tuple(range(64, 161, 32)) + tuple(range(320, 545, 32))
RESUME = ('\nContinue reasoning about the original question. Treat the reference as external evidence, '
          'not your answer. Use relevant facts, ignore irrelevant material, then provide your final answer.\n')
BRIDGE = ('\n\nI should check this external evidence against the original question and correct any '
          'unsupported assumptions before answering. The relevance of this reference to the question is')
UNCERTAIN = re.compile(r'\b(?:unsure|uncertain|unknown|maybe|perhaps|possibly|might|could|not sure|need to (?:check|verify))\b', re.I)


def clean_question(question):
    return question.split('\n\nIn the final answer,')[0].strip()


def boundary(text):
    return bool(re.search(r'(?:[.!?][\"\x27)\]]*|\n)\s*$', text))


def overlaps(left, right):
    """Exclude windows with >50% overlap with an already injected source window."""
    if '@' not in left or '@' not in right:
        return left == right
    a, x = left.rsplit('@', 1); b, y = right.rsplit('@', 1)
    if a != b:
        return False
    lo, hi = map(int, x.split('-')); low, high = map(int, y.split('-'))
    return max(0, min(hi, high)-max(lo, low)) > .5*min(hi-lo, high-low)


def retrieve_anchored(facts, question, tail, used):
    """Fuse question-only and branch-aware ranks; question carries 75% weight."""
    question = clean_question(question)
    queries = (question, question+'\n'+tail)
    lists, logs, payloads, weights = [], [], {}, {}
    for weight, query in zip((.75, .25), queries, strict=True):
        found, log = facts.retrieve(query, k=8, exclude=frozenset(used))
        logs.append(log); lists.append([p['id'] for p in found])
        for rank, payload in enumerate(found, 1):
            key = payload['id']; payloads[key] = payload
            weights[key] = weights.get(key, 0.)+weight/(20+rank)
    ranked = sorted(weights, key=lambda key: (-weights[key], key))
    chosen = next((key for key in ranked if not any(overlaps(key, previous) for previous in used)), None)
    # A narrow corpus may contain no additional nonoverlapping evidence. Skip it.
    return (payloads[chosen] if chosen else None), {
        'queries': list(queries), 'rank_lists': lists, 'weights': weights,
        'selected_ids': [chosen] if chosen else [], 'searches': logs,
        'seconds': sum(log['seconds'] for log in logs)}


class SmartOnlineLedger(OnlineISLedger):
    def __init__(self, *args, variant, **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        positions = CHECKPOINTS if variant in {'boundary', 'adaptive'} else (32, 256) if variant == 'early' else (128, 512)
        kwargs['retrieval_cfg'] = OnlineRetrievalConfig(positions=positions)
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.decisions, self._decision_cache = [], {}
        self.resume_ids = tuple(encode(self.backend, BRIDGE if variant in {'bridge', 'early'} else RESUME))
        forbidden = {self.eos, self.close[0], self.backend.tokenizer.convert_tokens_to_ids('<think>')}
        if forbidden.intersection(self.resume_ids):
            raise ValueError('structural token in continuation instruction')

    def recent_surprisal(self, branch):
        target = self.root+tuple(branch)
        start = max(0, len(branch)-32)
        values = [None]*(len(branch)-start)
        for call in self.native_calls:
            prefix = tuple(call['logical_prefix_ids'])
            if target[:len(prefix)] != prefix:
                continue
            offset = len(prefix)-len(self.root)
            for j, token in enumerate(call['token_ids']):
                at = offset+j
                if at >= len(branch) or branch[at] != token:
                    break
                if at >= start and values[at-start] is None:
                    values[at-start] = call['lp'][j]
        if not values or any(v is None or not math.isfinite(v) for v in values):
            raise RuntimeError('missing prefix probability for adaptive decision')
        return -math.fsum(values)/len(values)

    def materialize(self, generated, *, pending=True):
        generated = tuple(generated)
        stop = next((i for i, t in enumerate(generated) if t in {self.eos, self.close[0]}), len(generated))
        physical, event_ids, previous, used = list(self.root), [], 0, set()
        for position in self.retrieval_cfg.positions:
            if position > stop or (not pending and position == len(generated)):
                break
            branch = generated[:position]; key = token_hash(branch)
            if key not in self._decision_cache:
                text = self.backend.decode(branch[-256:], skip_special_tokens=True)
                first = position <= 160
                already = any(self.events[i]['phase'] == ('early' if first else 'late') for i in event_ids)
                aligned = boundary(text)
                due = self.variant in FIXED_VARIANTS or (aligned or position in {160, 544})
                eligible = not already and due
                surprise = self.recent_surprisal(branch) if self.variant == 'adaptive' and not first and eligible else None
                uncertain = bool(UNCERTAIN.search(text)) or (surprise is not None and surprise >= .8)
                if self.variant == 'adaptive' and not first:
                    eligible = eligible and uncertain
                decision = {'position': position, 'branch_ids': list(branch), 'branch_sha256': key,
                            'phase': 'early' if first else 'late', 'boundary': aligned,
                            'surprisal_last32': surprise, 'uncertainty_cue': bool(UNCERTAIN.search(text)),
                            'eligible': eligible, 'event_id': None}
                if eligible:
                    tail = text.strip()
                    if self.variant in SIMPLE_QUERY_VARIANTS:
                        query = self.question+'\n'+tail
                        found, retrieval = self.facts.retrieve(query, k=1, exclude=frozenset(used))
                        payload = found[0] if found else None
                    else:
                        payload, retrieval = retrieve_anchored(self.facts, self.question, tail, used)
                        query = clean_question(self.question)+'\n'+tail
                    decision['retrieval'] = retrieval
                    if payload is not None:
                        packet = fixed_packet(self.backend, payload, 256)
                        injected = tuple(packet['ids'])+self.resume_ids
                        forbidden = {self.eos, self.close[0], self.backend.tokenizer.convert_tokens_to_ids('<think>')}
                        if forbidden.intersection(injected):
                            raise ValueError('structural token in evidence')
                        event = {'event_id': len(self.events), 'position': position, 'phase': decision['phase'],
                                 'branch_ids': list(branch), 'branch_sha256': key, 'query': query,
                                 'packet': packet, 'injected_ids': list(injected), 'retrieval': retrieval}
                        self.events.append(event); self._event_cache[key] = event
                        decision['event_id'] = event['event_id']
                self.decisions.append(decision); self._decision_cache[key] = decision
            decision = self._decision_cache[key]
            if decision['event_id'] is None:
                continue
            event = self.events[decision['event_id']]
            physical.extend(generated[previous:position]); physical.extend(event['injected_ids'])
            event_ids.append(event['event_id']); used.add(event['packet']['id']); previous = position
        physical.extend(generated[previous:])
        return tuple(physical), event_ids


def run_smart_is(backend, tap, root, close, seed, cfg, *, question, facts, variant):
    from inference_scaling.retrieval.cot_guidance import finish_answer, search
    from inference_scaling.shared.rng import SeedStream
    ledger = SmartOnlineLedger(backend, tap, root, close, SeedStream(seed).derive('search'), cfg,
                               question=question, facts=facts, mode='relevant', variant=variant)
    started = time.perf_counter(); candidates, audit = search(ledger, 'is_block_sc')
    search_seconds = time.perf_counter()-started
    selected = candidates[0]; physical, events = ledger.materialize(selected, pending=False)
    started = time.perf_counter()
    answer = finish_answer(backend, root, physical[len(root):], close, cfg,
                           SeedStream(seed).derive('answer_suffix', token_hash(selected)))
    return {'variant': variant, 'root_ids': list(root), 'carried': ledger.carried, 'candidates': candidates,
            'calls': ledger.calls, 'native_calls': ledger.native_calls, 'events': ledger.events,
            'decisions': ledger.decisions, 'resume_ids': list(ledger.resume_ids),
            'selected_event_ids': events, 'selected_physical_ids': list(physical), 'presearch_work': 0,
            'search_work': ledger.work, 'total_generation_tokens': ledger.work+len(answer['answer_ids']),
            'answer': answer, 'audit': audit,
            'stage_seconds': {'search_seconds': search_seconds, 'materialization_seconds': ledger.materialization_seconds,
                              'answer_seconds': time.perf_counter()-started}}
