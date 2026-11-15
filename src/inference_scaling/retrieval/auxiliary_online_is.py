"""Candidate-local paid lookahead controllers around the unchanged IS kernel."""
from dataclasses import dataclass, asdict
import math

from inference_scaling.retrieval.auxiliary_signals import AuxiliarySampler, draft_confidence
from inference_scaling.retrieval.auxiliary_capture import layer_block, policy_disagreement
from inference_scaling.retrieval.tuned_online_is import TunedOnlineLedger, overlaps
from inference_scaling.retrieval.stepwise import token_hash


@dataclass(frozen=True)
class AuxiliaryPolicy:
    method: str = 'flare'
    draft_tokens: int = 64
    samples: int = 3
    low_probability: float = .2
    threshold: float = .1
    hidden_tail: int = 4
    metric: str = 'cosine'
    ridge: float = .01
    work_tokens: int = 8192
    evidence_candidates: int = 0
    evidence_tokens: int = 32
    evidence_ceiling: float = 2.0
    capture_layer: int | str | None = None

    def __post_init__(self):
        if self.method not in {'flare','seakr'} or self.metric not in {'cosine','variance','logvolume'}:
            raise ValueError('unknown auxiliary method or metric')
        if min(self.draft_tokens,self.samples,self.hidden_tail,self.work_tokens) < 1:
            raise ValueError('positive auxiliary budgets required')
        if self.method == 'seakr' and self.samples < 2:
            raise ValueError('representation disagreement needs independent samples')
        if not 0 < self.low_probability <= 1 or not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError('invalid auxiliary threshold')
        if not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError('positive finite ridge required')
        if self.evidence_candidates < 0 or self.evidence_tokens < 1 or not math.isfinite(self.evidence_ceiling) or self.evidence_ceiling < 0:
            raise ValueError('invalid evidence trial parameters')
        if self.evidence_candidates and self.method != 'seakr':
            raise ValueError('hidden evidence trials require SeaKR')
        if self.capture_layer is not None:
            layer_block(self.capture_layer)
            if self.method != 'seakr':
                raise ValueError('captured auxiliary layers require SeaKR')


def auxiliary_feature(policy, calls, backend):
    if policy.method == 'flare':
        feature = draft_confidence(calls[0], backend, low_probability=policy.low_probability)
        return feature['low_fraction'] >= policy.threshold, feature
    feature = policy_disagreement(policy, calls)
    return feature['distinct_paths'] > 1 and feature['value'] >= policy.threshold, feature


class AuxiliaryOnlineLedger(TunedOnlineLedger):
    def __init__(self, *args, auxiliary_policy, **kwargs):
        super().__init__(*args, **kwargs)
        if self.policy.trigger != 'fixed' or self.policy.query not in {'concat','question'}:
            raise ValueError('auxiliary gate requires fixed checkpoints and concat/question query')
        self.auxiliary_policy = auxiliary_policy
        self.auxiliary = AuxiliarySampler(self.backend, self.tap, self.seed, self.close,
            context_tokens=self.cfg.context_tokens, work_tokens=auxiliary_policy.work_tokens)
        self.lookaheads = {}
        self.physical_prefixes = {}

    def decision_signal(self, branch, text, physical):
        p = self.auxiliary_policy
        self.physical_prefixes[token_hash(branch)] = physical
        start = len(self.auxiliary.calls)
        calls = self.auxiliary.sample(physical, branch, purpose=p.method,
            count=1 if p.method == 'flare' else p.samples, tokens=p.draft_tokens)
        reason = {'auxiliary_start':start, 'auxiliary_stop':len(self.auxiliary.calls),
                  'budget_exhausted': calls is None}
        if calls is None:
            return False, reason
        eligible, feature = auxiliary_feature(p, calls, self.backend)
        reason.update(aux_ids=[c['aux_id'] for c in calls], feature=feature)
        self.lookaheads[token_hash(branch)] = feature
        return eligible, reason

    def retrieve(self, branch, used):
        if self.auxiliary_policy.evidence_candidates:
            return self.retrieve_by_hidden_trials(branch, used)
        if self.auxiliary_policy.method != 'flare' or self.policy.query == 'question':
            return super().retrieve(branch, used)
        query = self.question+'\n'+self.lookaheads[token_hash(branch)]['masked_query'].strip()
        found, log = self.facts.retrieve(query, k=self.policy.candidates, exclude=frozenset(used))
        found = [h for h in found if not any(overlaps(h['id'], old, self.policy.overlap) for old in used)]
        return (found[0] if found else None), log, query

    def retrieve_by_hidden_trials(self, branch, used):
        p = self.auxiliary_policy
        tail = self.backend.decode(branch[-self.policy.query_tail:],skip_special_tokens=True).strip()
        from inference_scaling.retrieval.smart_online_is import clean_question
        query = clean_question(self.question) if self.policy.query == 'question' else self.question+'\n'+tail
        found, log = self.facts.retrieve(query,k=self.policy.candidates,exclude=frozenset(used))
        found = [h for h in found if not any(overlaps(h['id'],old,self.policy.overlap) for old in used)][:p.evidence_candidates]
        trials = []
        start = len(self.auxiliary.calls)
        # Reserve the complete pool: never favor an earlier candidate because
        # later candidates did not fit the auxiliary budget.
        exhausted = self.auxiliary.work+len(found)*p.samples*p.evidence_tokens > p.work_tokens
        if not exhausted:
            for candidate in found:
                payload = self.format_packet(candidate)
                injected = tuple(payload['ids'])+self.continuation_ids
                physical = self.physical_prefixes[token_hash(branch)]+injected
                calls = self.auxiliary.sample(physical,branch,purpose='seakr_evidence:'+candidate['id'],
                    count=p.samples,tokens=p.evidence_tokens)
                if calls is None:
                    raise RuntimeError('reserved evidence budget unexpectedly exhausted')
                feature = policy_disagreement(p, calls)
                trials.append({'id':candidate['id'],'packet':payload,'injected_ids':list(injected),
                               'aux_ids':[c['aux_id'] for c in calls],'feature':feature})
        accepted = [t for t in trials if t['feature']['value'] <= p.evidence_ceiling]
        chosen = min(accepted,key=lambda t:(t['feature']['value'],t['id']))['id'] if accepted else None
        log = {**log,'evidence_trials':{'auxiliary_start':start,'auxiliary_stop':len(self.auxiliary.calls),
            'candidate_ids':[h['id'] for h in found],'budget_exhausted':exhausted,
            'trials':trials,'chosen_id':chosen}}
        return next((h for h in found if h['id'] == chosen),None),log,query

    def auxiliary_result(self):
        policy = asdict(self.auxiliary_policy)
        captured = policy['capture_layer'] is not None
        if not captured:
            policy.pop('capture_layer')
        return {'auxiliary_policy':policy, 'auxiliary_calls':self.auxiliary.calls,
                'auxiliary_work':self.auxiliary.work,
                'adaptation':'candidate-local paid lookahead, discarded drafts never committed to IS; '
                    + ('explicit real captured-layer hidden features; ' if captured else 'final-normalized hidden features; ')
                    + 'equal-sized evidence sample pools; '
                    'uncertainty ranking is not a correctness oracle or paper reproduction'}
