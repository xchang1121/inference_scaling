"""Candidate-local ThinkRetrieve adaptation; paid interim answers, whole examples."""
from dataclasses import asdict, dataclass

from inference_scaling.retrieval.auxiliary_signals import AuxiliarySampler
from inference_scaling.retrieval.online_is import OnlineISLedger
from inference_scaling.retrieval.common import token_hash


@dataclass(frozen=True)
class ExamplePolicy:
    answer_tokens: int = 128
    work_tokens: int = 8192
    require_complete_answer: bool = False
    pad_to_budget: bool = True

    def __post_init__(self):
        if min(self.answer_tokens, self.work_tokens) < 1:
            raise ValueError('positive example retrieval budgets required')


class ExampleOnlineLedger(OnlineISLedger):
    def __init__(self, *args, example_policy, **kwargs):
        super().__init__(*args, **kwargs)
        if self.policy.trigger != 'fixed' or self.policy.query != 'concat' or self.policy.format != 'original':
            raise ValueError('example retrieval requires fixed checkpoints, joint QA and intact template')
        self.example_policy = example_policy
        self.auxiliary = AuxiliarySampler(self.backend, self.tap, self.seed, self.close,
            context_tokens=self.cfg.context_tokens, work_tokens=example_policy.work_tokens)
        self.interims = {}

    def decision_signal(self, branch, text, physical):
        p = self.example_policy
        start = len(self.auxiliary.calls)
        calls = self.auxiliary.sample(tuple(physical)+self.close, branch,
            purpose='thinkretrieve_answer', count=1, tokens=p.answer_tokens)
        reason = {'auxiliary_start':start, 'auxiliary_stop':len(self.auxiliary.calls),
                  'budget_exhausted':calls is None}
        if calls is None:
            return False, reason
        call = calls[0]
        ids = tuple(t for t in call['token_ids'] if t not in {self.eos, self.close[0]})
        interim = self.backend.decode(ids, skip_special_tokens=True).strip()
        complete = call['finish_reason'] in {'eos','eos_token','stop'} or call['token_ids'][-1] == self.eos
        reason.update(aux_ids=[call['aux_id']], interim=interim, complete=complete)
        self.interims[token_hash(branch)] = interim
        return bool(interim) and (complete or not p.require_complete_answer), reason

    def retrieve(self, branch, used):
        interim = self.interims[token_hash(branch)]
        found, log = self.facts.retrieve(self.question, interim, k=self.policy.candidates, exclude=frozenset(used))
        # A solution is either retained in full or rejected, never truncated.
        eligible = [p for p in found if len(p['ids']) <= self.policy.packet_tokens]
        log = {**log, 'complete_packet_ids':[p['id'] for p in eligible],
               'oversized_ids':[p['id'] for p in found if len(p['ids']) > self.policy.packet_tokens],
               'candidate_packets':found}
        return (eligible[0] if eligible else None), log, log['query']

    def format_packet(self, original):
        if len(original['ids']) > self.policy.packet_tokens:
            raise ValueError('worked solution must remain complete')
        if not self.example_policy.pad_to_budget:
            return {**original, 'original_packet_tokens':len(original['ids'])}
        newline = tuple(self.backend.encode('\n', add_special_tokens=False))
        if len(newline) != 1:
            raise ValueError('example padding needs a single newline token')
        ids = list(original['ids'])+list(newline)*(self.policy.packet_tokens-len(original['ids']))
        return {**original, 'ids':ids, 'sha256':token_hash(ids),
                'original_packet_tokens':len(original['ids'])}

    def auxiliary_result(self):
        return {'example_policy':asdict(self.example_policy), 'auxiliary_calls':self.auxiliary.calls,
                'auxiliary_work':self.auxiliary.work,
                'adaptation':'candidate-local paid interim answer after an auxiliary thought close; '
                             'joint QA E5 retrieval; whole worked examples; no pre-IS thought or committed rollback'}
