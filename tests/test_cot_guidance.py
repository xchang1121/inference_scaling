from dataclasses import replace
from types import SimpleNamespace
import pytest

from inference_scaling.arllm.types import SequenceSample
from inference_scaling.retrieval.cot_guidance import (
    GuidanceConfig, GuidanceLedger, checkpoint, completed_boundary, finish_answer, finish_answers, search, trajectory_score,
)
from inference_scaling.retrieval.ablation import static_prompt


class Fake:
    model_id = 'fake'
    tokenizer = SimpleNamespace(eos_token_id=99)
    def __init__(self): self.requests = []
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)
    def sample_batch(self, requests):
        self.requests.extend(requests)
        return [SequenceSample(tuple(r.prefix),(65,66,90),(-1.,-2.,-3.),r.sampling.policy_id,self.model_id,r.request_id, 'stop') for r in requests]


class Tap:
    def take(self, request, sample): return (1.,2.,3.), (4.,5.,6.)


def test_same_packet_only_changes_position_and_retains_prefix():
    base, packet, prior = (1,9,2,9,3), (7,8), (4,5)
    assert static_prompt(base,packet,9,()) == (1,9,2,7,8,9,3)
    assert base+prior+packet == (1,9,2,9,3,4,5,7,8)


def test_step_boundary_excludes_partial_sentence_and_version():
    backend=Fake()
    ids=tuple(map(ord,'Version 2.4 supports it. Next incomplete'))
    n=completed_boundary(backend,ids,10)
    assert backend.decode(ids[:n]) == 'Version 2.4 supports it.'
    assert completed_boundary(backend,tuple(map(ord,'Nothing finished')),3) is None


def test_dual_scores_exclude_close_and_include_carried_reasoning():
    cfg=replace(GuidanceConfig(),thinking_tokens=10,candidates=4)
    ledger=GuidanceLedger(Fake(),Tap(),(1,),(90,),1,cfg,carried={'sc':[9.],'cn':[8.],'lp':[-4.]})
    candidates,audit=search(ledger,'bon')
    assert len(candidates)==4 and ledger.work==12
    assert ledger.statistics(candidates[0])=={'sc':[9.,1.,2.],'cn':[8.,4.,5.],'lp':[-4.,-1.,-2.]}
    assert audit['scores']['sc']==(4.,)*4
    assert audit['selection']['bon_sc']==0
    assert len(ledger.backend.requests)==4  # All BoN selectors share one pool.


def test_is_selection_kernels_use_complete_funded_pools():
    for method in ('is_sir_sc','is_met_sc','is_block_sc','is_block_consilience'):
        cfg=replace(GuidanceConfig(),thinking_tokens=10,block_tokens=3,search_tokens=200)
        ledger=GuidanceLedger(Fake(),Tap(),(1,),(90,),1,cfg)
        candidates,audit=search(ledger,method)
        assert candidates and audit['steps']
        assert ledger.work<=cfg.search_tokens


def test_rejects_unfunded_pool_and_changed_root():
    ledger=GuidanceLedger(Fake(),Tap(),(1,),(90,),1,GuidanceConfig(),cap=4)
    with pytest.raises(RuntimeError,match='cap'):
        ledger.sample_batch([ledger.request(3,0),ledger.request(3,1)])
    with pytest.raises(RuntimeError,match='root'):
        ledger.sample_batch([replace(ledger.request(3,0),prefix=(2,))])


def test_answer_continues_after_length_stop_without_closing_it_again():
    class Answer(Fake):
        def sample_batch(self,requests):
            r,=requests
            self.requests.append(r)
            ids=(65,66) if len(self.requests)==1 else (67,99)
            return [SequenceSample(r.prefix,ids,(-1.,)*2,r.sampling.policy_id,self.model_id,r.request_id,'length' if len(self.requests)==1 else 'eos')]
    raw=Answer()
    result=finish_answer(raw,(1,),(2,90),(90,),replace(GuidanceConfig(),answer_chunk=2,answer_tokens=8),1)
    assert not result['truncated'] and not result['forced_thinking_end']
    assert raw.requests[1].prefix==(1,2,90,65,66)


def test_length_limit_is_reported_as_failure_not_natural_eos():
    raw=Fake()
    # Fake reports stop; an answer EOS in thinking must never be repaired into a normal answer.
    result=finish_answer(raw,(1,),(65,99),(90,),GuidanceConfig(),1)
    assert result['eos_inside_thinking'] and result['empty_answer'] and not raw.requests
    assert trajectory_score([],[],[],'sc')==0


def test_batched_answers_keep_individual_prefixes_and_stop_independently():
    class Answer(Fake):
        def __init__(self): super().__init__();self.batches=[]
        def sample_batch(self,requests):
            self.batches.append(len(requests))
            result=[]
            for r in requests:
                self.requests.append(r)
                final=':answer:0:' in r.request_id or r.request_id.endswith(':1')
                ids=(65,99) if final else (65,66)
                result.append(SequenceSample(r.prefix,ids,(-1.,)*2,r.sampling.policy_id,self.model_id,r.request_id,'eos' if final else 'length'))
            return result
    raw=Answer()
    results=finish_answers(raw,(1,),[(2,90),(3,90)],(90,),replace(GuidanceConfig(),answer_chunk=2,answer_tokens=8),[1,2])
    assert raw.batches==[2,1]
    assert [r['answer_ids'] for r in results]==[[65,99],[65,66,65,99]]
    assert raw.requests[-1].prefix==(1,3,90,65,66)
    assert not any(r['truncated'] for r in results)


def test_both_placements_use_identical_checkpoint_stopping_policy():
    class Text(Fake):
        def sample_batch(self,requests):
            self.requests.extend(requests)
            ids=tuple(map(ord,'A complete sentence. More unfinished words'))
            return [SequenceSample(r.prefix,ids,(-1.,)*len(ids),r.sampling.policy_id,self.model_id,r.request_id,'length') for r in requests]
    class Stats:
        def take(self,request,sample):return (2.,)*len(sample.token_ids),(3.,)*len(sample.token_ids)
    cfg=replace(GuidanceConfig(),probe_minimum=10)
    raw=Text()
    left=checkpoint(raw,Stats(),(1,7,8),(90,),1,cfg)
    right=checkpoint(raw,Stats(),(1,),(90,),1,cfg)
    assert left['reason_ids']==right['reason_ids']==list(map(ord,'A complete sentence.'))
    assert left['work']==right['work']
    assert left['probe_root_ids']==[1,7,8] and right['probe_root_ids']==[1]
