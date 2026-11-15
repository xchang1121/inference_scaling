"""Configurable candidate-local retrieval, retaining the frozen IS/search kernel."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import time

from inference_scaling.retrieval.cot_guidance import finish_answer, search
from inference_scaling.retrieval.literature import encode, packet
from inference_scaling.retrieval.online_is import OnlineISLedger, OnlineRetrievalConfig, fixed_packet
from inference_scaling.retrieval.smart_online_is import boundary, clean_question, BRIDGE, RESUME
from inference_scaling.retrieval.stepwise import token_hash, step_char_boundaries
from inference_scaling.shared.rng import SeedStream
from inference_scaling.retrieval.attention_policy import attention_feature, content_piece


@dataclass(frozen=True)
class TuningPolicy:
    trigger: str = 'fixed'
    positions: tuple[int, ...] = (128, 512)
    max_insertions: int = 2
    cooldown: int = 128
    window: int = 32
    threshold: float = .8
    query: str = 'concat'
    query_tail: int = 256
    question_weight: float = .75
    anchor_context: str = 'tail'
    candidates: int = 8
    overlap: float = 1.0
    packet_tokens: int = 256
    format: str = 'original'
    continuation: str = 'none'
    continuation_text: str = ''
    wait_for_boundary: bool = False
    boundary_deadlines: tuple[int, ...] = (160, 544)
    low_probability: float = .2
    statistic_content_only: bool = False
    sc_reference: str = 'previous'
    rule_pattern: str = r'\b(?:unknown|unsure|uncertain|perhaps|maybe|version|error|command|APAR|CVE)\b'
    rule_tail: int = 512
    hybrid_mode: str = 'any'
    hybrid_surprisal: float = 1.5
    hybrid_low_fraction: float = .25
    hybrid_sc_drop: float = .05
    attention_query_tokens: int = 8
    attention_content_only: bool = True
    attention_expand_words: bool = True

    def __post_init__(self):
        if type(self.statistic_content_only) is not bool:
            raise ValueError('statistic_content_only must be boolean')
        if self.statistic_content_only and self.trigger not in {'lowprob', 'surprisal', 'entropy', 'margin'}:
            raise ValueError('content filtering requires a supported window statistic')
        if self.trigger not in {'never','fixed','boundary','surprisal','lowprob','scdrop','rule','hybrid','entropy','margin','risk','dragin'}:
            raise ValueError('unsupported trigger')
        if self.query not in {'concat','question','anchored','uncertain','attention','sentence','sentence_anchored'}:
            raise ValueError('unsupported query')
        if self.anchor_context not in {'tail', 'question_tail'}:
            raise ValueError('unsupported anchored context')
        if self.format not in {'original','plain','quoted','source','document'}:
            raise ValueError('unsupported evidence format')
        if self.continuation not in {'none','resume','bridge','custom'}:
            raise ValueError('unsupported continuation')
        if self.max_insertions < 1 or self.cooldown < 1 or self.window < 1:
            raise ValueError('positive policy budgets required')
        if self.sc_reference not in {'previous', 'initial'} or self.hybrid_mode not in {'any', 'all'}:
            raise ValueError('unknown SC reference or hybrid operator')
        if not 0 < self.low_probability <= 1 or not 0 <= self.question_weight <= 1 or not 0 <= self.overlap <= 1:
            raise ValueError('invalid probability, query weight or overlap')
        if min(self.query_tail, self.candidates, self.rule_tail, self.attention_query_tokens) < 1:
            raise ValueError('positive query and rule budgets required')
        if not all(math.isfinite(v) for v in (self.threshold, self.hybrid_surprisal,
                                             self.hybrid_low_fraction, self.hybrid_sc_drop)):
            raise ValueError('finite thresholds required')
        if not 0 <= self.hybrid_low_fraction <= 1 or not 0 <= self.hybrid_sc_drop < 1:
            raise ValueError('invalid hybrid thresholds')
        re.compile(self.rule_pattern)
        OnlineRetrievalConfig(tuple(self.positions), self.query_tail, self.packet_tokens)


def overlaps(left, right, threshold):
    def parse(value):
        match = re.fullmatch(r'(.*)@(\d+)-(\d+)', value)
        return (match[1], int(match[2]), int(match[3])) if match else (value, 0, 1)
    a, b = parse(left), parse(right)
    if a[0] != b[0]:
        return False
    return max(0, min(a[2], b[2])-max(a[1], b[1]))/max(1, min(a[2]-a[1], b[2]-b[1])) > threshold


def last_complete_step(text):
    """Last complete sentence/paragraph in the observed candidate tail."""
    bounds=[0]+step_char_boundaries(text)
    for left,right in reversed(list(zip(bounds,bounds[1:]))):
        step=text[left:right].strip()
        if step:return step
    return text.strip()


class TuningTap:
    """Keep diagnostic chunks aligned to the same native requests consumed by SC."""
    def __init__(self, dual, signals):
        self.dual, self.signals, self.history = dual, signals, {}

    def take(self, request, sample):
        result = self.dual.take(request, sample)
        if request.request_id in self.signals.rows:
            self.history[request.request_id] = self.signals.take(request, sample)
        return result


class TunedOnlineLedger(OnlineISLedger):
    def __init__(self, *args, policy, **kwargs):
        self.policy = policy
        kwargs['retrieval_cfg'] = OnlineRetrievalConfig(tuple(policy.positions), policy.query_tail, policy.packet_tokens)
        super().__init__(*args, **kwargs)
        self.decisions, self._decisions = [], {}
        text = {'none': '', 'resume': RESUME, 'bridge': BRIDGE, 'custom': policy.continuation_text}[policy.continuation]
        self.continuation_ids = encode(self.backend, text)
        forbidden = {self.eos, self.close[0], self.backend.tokenizer.convert_tokens_to_ids('<think>')}
        if forbidden.intersection(self.continuation_ids):
            raise ValueError('structural token in continuation')

    def recent(self, branch, count):
        target = self.root+tuple(branch)
        start = max(0, len(branch)-count)
        rows = [None]*(len(branch)-start)
        for call in self.native_calls:
            prefix = tuple(call['logical_prefix_ids'])
            if target[:len(prefix)] != prefix:
                continue
            offset = len(prefix)-len(self.root)
            diagnostics = getattr(self.tap, 'history', {}).get(call['request_id'])
            for j, token in enumerate(call['token_ids']):
                at = offset+j
                if at >= len(branch) or branch[at] != token:
                    break
                if at >= start and rows[at-start] is None:
                    rows[at-start] = {'token': token, 'lp': call['lp'][j], 'sc': call['sc'][j],
                                      **(diagnostics[j] if diagnostics is not None else {})}
                    if 'attention' in rows[at-start]:
                        physical = tuple(call['prefix_ids'])+tuple(call['token_ids'][:j])
                        a = rows[at-start]['attention']
                        if a['prefix_tokens'] != len(physical) or any(not 0 <= i < len(physical) for i in a['indices']):
                            raise RuntimeError('attention refers outside the observed physical prefix')
                        rows[at-start].update(physical_index=len(physical),
                            attention_token_ids=[physical[i] for i in a['indices']],attention_prefix_ids=physical)
        if not rows or any(r is None for r in rows):
            raise RuntimeError('missing causal prefix statistics')
        return rows

    def signal(self, branch, text):
        p = self.policy
        if p.trigger == 'never':
            return False, {}
        if p.trigger == 'fixed':
            return True, {}
        if p.trigger == 'boundary':
            return boundary(text) or len(branch) in p.boundary_deadlines, {}
        if p.trigger == 'rule':
            cues = re.findall(p.rule_pattern, text[-p.rule_tail:], re.I)
            return len(cues) >= p.threshold, {'cues': cues}
        need_sc = p.trigger in {'scdrop', 'hybrid'}
        count = (len(branch) if p.sc_reference == 'initial' else 2*p.window) if need_sc else p.window
        history = self.recent(branch, count)
        rows = history[-p.window:]
        if p.trigger == 'dragin':
            feature = attention_feature(rows, self.backend.decode,
                query_tokens=p.attention_query_tokens, content_only=p.attention_content_only,expand_words=p.attention_expand_words)
            return feature['value'] >= p.threshold, feature
        sc_before = history[:p.window] if p.sc_reference == 'initial' else history[-2*p.window:-p.window]
        sc_drop = None
        if need_sc and len(history) >= 2*p.window:
            before = math.fsum(r['sc'] for r in sc_before)/len(sc_before)
            after = math.fsum(r['sc'] for r in rows)/len(rows)
            sc_drop = (before-after)/max(1e-6, abs(before))
        if p.trigger == 'hybrid':
            values = {'surprisal': -math.fsum(r['lp'] for r in rows)/len(rows),
                      'low_fraction': sum(r['lp'] < math.log(p.low_probability) for r in rows)/len(rows),
                      'sc_drop': sc_drop,
                      'rule_match': bool(re.search(p.rule_pattern, text[-p.rule_tail:], re.I))}
            gates = [values['surprisal'] >= p.hybrid_surprisal,
                     values['low_fraction'] >= p.hybrid_low_fraction,
                     sc_drop is not None and sc_drop > p.hybrid_sc_drop, values['rule_match']]
            return (any(gates) if p.hybrid_mode == 'any' else all(gates)), values
        window_tokens = len(rows)
        content_metadata = {}
        if p.statistic_content_only:
            rows = [r for r in rows if content_piece(r['token'], self.backend.decode)]
            content_metadata = {'content_tokens': len(rows), 'content_filter': 'decoded_piece_v1'}
            if not rows:
                return False, {'value': None, 'window_tokens': window_tokens,
                               **content_metadata, 'insufficient_content': True}
        if p.trigger == 'surprisal':
            value = -math.fsum(r['lp'] for r in rows)/len(rows)
        elif p.trigger == 'lowprob':
            value = sum(r['lp'] < math.log(p.low_probability) for r in rows)/len(rows)
        elif p.trigger == 'scdrop':
            if sc_drop is None:
                return False, {'insufficient_history': True}
            value = sc_drop
        elif p.trigger == 'entropy':
            value = math.fsum(r['entropy'] for r in rows)/len(rows)
        elif p.trigger == 'margin':
            value = 1-math.fsum(r['top2_gap'] for r in rows)/len(rows)
        else:
            value = rows[-1]['p_eos']+rows[-1]['p_think_end']
            # Only a feature of the last observed predictive distribution;
            # this is NOT claimed to be the distribution after the last token.
            return value <= p.threshold, {'value': value, 'state_offset': -1}
        return value >= p.threshold, {'value': value, 'window_tokens': window_tokens, **content_metadata}

    def retrieve(self, branch, used):
        p = self.policy
        tail = self.backend.decode(branch[-p.query_tail:], skip_special_tokens=True).strip()
        question = clean_question(self.question)
        if p.query == 'uncertain':
            rows = self.recent(branch, p.query_tail)
            tokens = [r['token'] for r in rows if r['lp'] < math.log(p.low_probability)]
            tail = self.backend.decode(tuple(tokens), skip_special_tokens=True)
        if p.query == 'attention':
            tail = attention_feature(self.recent(branch,p.window),self.backend.decode,
                query_tokens=p.attention_query_tokens,content_only=p.attention_content_only,expand_words=p.attention_expand_words)['query_text']
        query = question if p.query == 'question' else self.question+'\n'+tail
        if p.query in {'sentence','sentence_anchored'}:
            step=last_complete_step(tail)
            query=step if p.query=='sentence' else question+'\n'+step
        if p.query == 'anchored' and p.anchor_context == 'question_tail':
            query = question+'\n'+tail
        if p.query != 'anchored':
            found, log = self.facts.retrieve(query, k=p.candidates, exclude=frozenset(used))
        else:
            left, a = self.facts.retrieve(question, k=p.candidates, exclude=frozenset(used))
            context_query = question+'\n'+tail if p.anchor_context == 'question_tail' else tail or question
            right, b = self.facts.retrieve(context_query, k=p.candidates, exclude=frozenset(used))
            scores, payloads = {}, {}
            for weight, hits in ((p.question_weight, left), (1-p.question_weight, right)):
                for rank, hit in enumerate(hits, 1):
                    scores[hit['id']] = scores.get(hit['id'], 0)+weight/(20+rank)
                    payloads[hit['id']] = hit
            found = [payloads[k] for k in sorted(scores, key=lambda x: (-scores[x], x))]
            log = {'query': query, 'components': [a,b], 'rrf_scores': scores, 'seconds': a['seconds']+b['seconds']}
        found = [h for h in found if not any(overlaps(h['id'], old, p.overlap) for old in used)]
        return (found[0] if found else None), log, query

    def format_packet(self, original):
        p = self.policy
        if p.format == 'original':
            return fixed_packet(self.backend, original, p.packet_tokens)
        body = self.backend.decode(tuple(original['ids'][original['body_start']:original['body_stop']]), skip_special_tokens=False)
        headers = {'plain': ('\n\n','\n\n'), 'quoted': ('\nSource excerpt:\n"','"\n'),
                   'source': ('\nSource: '+original.get('source', original['id'])+'\n','\n'),
                   'document': ('\nDocument: '+original['id']+'\n','\n')}
        header, footer = headers[p.format]
        payload = packet(self.backend, header, body, footer, id=original['id'], source=original.get('source',''))
        return fixed_packet(self.backend, payload, p.packet_tokens)

    def decision_signal(self, branch, text, physical):
        return self.signal(branch, text)

    def materialize(self, generated, *, pending=True):
        generated = tuple(generated)
        stop = next((i for i,t in enumerate(generated) if t in {self.eos,self.close[0]}), len(generated))
        physical, event_ids, previous, used = list(self.root), [], 0, set()
        for position in self.policy.positions:
            if position > stop or (not pending and position == len(generated)):
                break
            branch, p = generated[:position], self.policy
            key = token_hash(branch)
            if key not in self._decisions:
                tail_size = max(p.query_tail, p.rule_tail) if p.trigger in {'rule', 'hybrid'} else p.query_tail
                text = self.backend.decode(branch[-tail_size:], skip_special_tokens=True)
                eligible = len(event_ids) < p.max_insertions and (not event_ids or position-previous >= p.cooldown)
                reason = {}
                if eligible:
                    eligible, reason = self.decision_signal(branch, text, tuple(physical)+generated[previous:position])
                if p.wait_for_boundary:
                    eligible = eligible and (boundary(text) or position in p.boundary_deadlines)
                decision = {'position': position, 'branch_ids': list(branch), 'branch_sha256': key,
                            'triggered': bool(eligible), 'signal': reason, 'event_id': None,
                            'boundary': boundary(text), 'observed_native_calls': len(self.native_calls)}
                if eligible:
                    payload, retrieval, query = self.retrieve(branch, used)
                    decision['retrieval'] = retrieval
                    if payload is not None:
                        payload = self.format_packet(payload)
                        injected = tuple(payload['ids'])+self.continuation_ids
                        forbidden = {self.eos,self.close[0],self.backend.tokenizer.convert_tokens_to_ids('<think>')}
                        if forbidden.intersection(injected):
                            raise ValueError('structural token in injected evidence')
                        event = {'event_id': len(self.events), 'position': position, 'branch_ids': list(branch),
                                 'branch_sha256': key, 'query': query, 'packet': payload,
                                 'injected_ids': list(injected), 'retrieval': retrieval}
                        self.events.append(event)
                        decision['event_id'] = event['event_id']
                self.decisions.append(decision)
                self._decisions[key] = decision
            decision = self._decisions[key]
            if decision['event_id'] is not None:
                event = self.events[decision['event_id']]
                physical.extend(generated[previous:position])
                physical.extend(event['injected_ids'])
                event_ids.append(event['event_id'])
                used.add(event['packet']['id'])
                previous = position
        physical.extend(generated[previous:])
        return tuple(physical), event_ids


def run_tuned_is(backend, tap, root, close, seed, cfg, *, question, facts, policy, auxiliary_policy=None, gap_policy=None, example_policy=None, probe_policy=None, alignment_policy=None, document_vectors=None):
    cls, extra = TunedOnlineLedger, {}
    if sum(v is not None for v in (auxiliary_policy, gap_policy, example_policy, probe_policy, alignment_policy)) > 1:
        raise ValueError('choose one paid auxiliary controller')
    if auxiliary_policy is not None:
        from inference_scaling.retrieval.auxiliary_online_is import AuxiliaryOnlineLedger
        cls, extra = AuxiliaryOnlineLedger, {'auxiliary_policy': auxiliary_policy}
    if gap_policy is not None:
        from inference_scaling.retrieval.gap_online_is import GapOnlineLedger
        cls, extra = GapOnlineLedger, {'gap_policy':gap_policy}
    if example_policy is not None:
        from inference_scaling.retrieval.example_online_is import ExampleOnlineLedger
        cls, extra = ExampleOnlineLedger, {'example_policy':example_policy}
    if probe_policy is not None:
        from inference_scaling.retrieval.probe_online_is import ProbeOnlineLedger
        cls, extra = ProbeOnlineLedger, {'probe_policy':probe_policy}
    if alignment_policy is not None:
        from inference_scaling.retrieval.aligned_online_is import AlignedOnlineLedger
        cls, extra = AlignedOnlineLedger, {'alignment_policy':alignment_policy,'document_vectors':document_vectors}
    ledger = cls(backend, tap, root, close, SeedStream(seed).derive('search'), cfg,
                 question=question, facts=facts, mode='relevant', policy=policy, **extra)
    started = time.perf_counter()
    candidates, audit = search(ledger, 'is_block_sc')
    search_seconds = time.perf_counter()-started
    selected = candidates[0]
    physical, events = ledger.materialize(selected, pending=False)
    started = time.perf_counter()
    answer = finish_answer(backend, root, physical[len(root):], close, cfg,
                           SeedStream(seed).derive('answer_suffix', token_hash(selected)))
    for call in ledger.native_calls:
        if call['request_id'] in getattr(tap, 'history', {}):
            call['internal_signals'] = tap.history[call['request_id']]
    auxiliary = ledger.auxiliary_result() if extra else {}
    return {'policy': asdict(policy), 'root_ids': list(root), 'carried': ledger.carried,
            'candidates': candidates, 'calls': ledger.calls, 'native_calls': ledger.native_calls,
            'events': ledger.events, 'decisions': ledger.decisions, 'selected_event_ids': events,
            'selected_physical_ids': list(physical), 'presearch_work': 0, 'search_work': ledger.work,
            'total_generation_tokens': ledger.work+len(answer['answer_ids'])+auxiliary.get('auxiliary_work',0),
            **auxiliary, 'answer': answer, 'audit': audit,
            'stage_seconds': {'search_seconds': search_seconds, 'materialization_seconds': ledger.materialization_seconds,
                              'answer_seconds': time.perf_counter()-started, 'static_retrieval_seconds': 0.}}
