from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference_scaling.arllm.types import SequenceSample
from inference_scaling.retrieval.cot_guidance import GuidanceConfig, GuidanceLedger
from inference_scaling.retrieval.literature import (
    EXAMPLE_FOOTER, EXAMPLE_HEADER, FACT_FOOTER, FACT_HEADER, LiteratureConfig,
    LiteratureController, packet, unrelated,
)


class Fake:
    model_id='fake'
    def __init__(self,eos=False):
        self.requests=[];self.eos=eos
        self.tokenizer=SimpleNamespace(eos_token_id=10000,
            convert_tokens_to_ids=lambda s:{'</think>':10001,'<think>':10003,'<|im_end|>':10002}[s])
    def encode(self,text,**kwargs):return tuple(map(ord,text))
    def decode(self,ids,**kwargs):return ''.join(chr(t) for t in ids if t<10000)
    def sample_batch(self,requests):
        self.requests.extend(requests)
        result=[]
        for r in requests:
            ids=(10000,) if self.eos else tuple(map(ord,'Use the supported procedure.'))+(10001,)
            if ':interim:' in r.request_id:ids=tuple(map(ord,'Restart the service.'))+(10000,)
            result.append(SequenceSample(tuple(r.prefix),ids,(-1.,)*len(ids),r.sampling.policy_id,
                self.model_id,r.request_id,'eos' if 10000 in ids else 'stop'))
        return result


class Tap:
    def take(self,request,sample):return (2.,)*len(sample.token_ids),(3.,)*len(sample.token_ids)


class Bank:
    def __init__(self,backend,example):self.backend=backend;self.example=example;self.queries=[]
    def retrieve(self,question,interim='',*,k=1,exclude=frozenset()):
        self.queries.append((question,interim))
        head,tail=(EXAMPLE_HEADER,EXAMPLE_FOOTER) if self.example else (FACT_HEADER,FACT_FOOTER)
        rows=[packet(self.backend,head,'Source-supported steps: check and restart.',tail,id=str(i))
              for i in range(5) if str(i) not in exclude][:k]
        return rows,{'query':question+interim,'selected_ids':[p['id'] for p in rows],'seconds':.01}


def controller(eos=False):
    b=Fake(eos)
    ctrl=LiteratureController(b,Tap(),Bank(b,False),Bank(b,True),(10001,),tuple(map(ord,'unrelated garden prose ')),
        GuidanceConfig(),replace(LiteratureConfig(),thinkretrieve_interval=128))
    return b,ctrl


BASE=(1,10002,2,10002,3)


def test_tr_intercepts_thought_close_keeps_prefix_and_pays_interim_cost():
    b,c=controller()
    h=c.thinkretrieve(BASE,'How to fix?',1)
    assert len(h['packets'])==2
    assert h['root_ids'][:len(BASE)]==BASE
    assert 10001 not in h['root_ids'] and 10000 not in h['root_ids']
    assert h['controller_work']==sum(len(x['token_ids']) for x in h['controller_calls']+h['interim_calls'])
    assert len(h['carried']['sc'])==len(h['retained_reason_ids'])
    assert all(a=='Restart the service.' for q,a in c.examples.queries)
    assert [e['trigger'] for e in h['events'] if e['kind']=='thought']==['natural_close']*2
    first,second=h['controller_calls']
    assert second['prefix_ids']==list(BASE)+first['token_ids'][:-1]+h['injected_ids'][0]
    assert len(b.requests)==4


def test_paired_noise_replaces_entire_example_and_matches_each_token_count():
    b,c=controller()
    real=c.thinkretrieve(BASE,'How to fix?',1)
    noise=c.thinkretrieve(BASE,'How to fix?',1,mode='unrelated',reference=real['packets'])
    for actual,ref,p in zip(noise['injected_ids'],real['injected_ids'],real['packets'],strict=True):
        assert len(actual)==len(ref) and actual!=ref
        assert actual[:p['body_start']]==ref[:p['body_start']]
        assert actual[p['body_stop']:]==ref[p['body_stop']:]
        assert 'Source-supported' not in b.decode(actual)
    assert len(noise['carried']['sc'])==len(real['carried']['sc'])


def test_eos_is_terminal_instead_of_implicitly_reopened_including_noise():
    b,c=controller(True)
    real=c.thinkretrieve(BASE,'question',1)
    noise=c.thinkretrieve(BASE,'question',1,mode='unrelated',reference=real['packets'])
    assert real['terminal_eos'] and noise['terminal_eos']
    assert not real['packets'] and not noise['packets'] and not c.examples.queries
    assert real['controller_work']==noise['controller_work']==1
    assert len(b.requests)==2


def test_rethink_control_has_no_retrieval_or_intermediate_answer():
    b,c=controller()
    h=c.thinkretrieve(BASE,'question',1,mode='rethink')
    assert not c.examples.queries and not h['packets'] and not h['interim_calls']
    assert len(h['injected_ids'])==2 and len(b.requests)==2


def test_ircot_uses_last_sentence_then_discards_agent_cot_for_fresh_reader():
    b,c=controller()
    h=c.ircot(BASE,'question',1)
    assert len(h['packets'])==3
    assert c.facts.queries==[('question',''),('Use the supported procedure.',''),('Use the supported procedure.','')]
    assert 'Use the supported procedure.' not in b.decode(h['root_ids'])
    assert h['carried']=={'sc':[],'cn':[],'lp':[]}
    assert h['controller_work']==sum(len(x['token_ids']) for x in h['controller_calls'])
    n=c.ircot(BASE,'question',1,noise=True)
    assert list(map(len,n['injected_ids']))==list(map(len,h['injected_ids']))
    assert n['controller_calls']==h['controller_calls']
    assert 'Source-supported' not in b.decode(n['root_ids'])


def test_inserted_example_tokens_never_enter_sc_reward():
    b,c=controller()
    h=c.thinkretrieve(BASE,'question',1)
    ledger=GuidanceLedger(b,Tap(),h['root_ids'],c.close,1,GuidanceConfig(),carried=h['carried'])
    sample,=ledger.sample_batch([ledger.request(128,0)])
    stats=ledger.statistics(sample.token_ids)
    assert len(stats['sc'])==len(h['retained_reason_ids'])+len(sample.token_ids)-1
    assert ledger.scores([sample.token_ids],'sc')==(2.,)


def test_noise_rejects_empty_pool():
    b=Fake();p=packet(b,'H','body','F')
    with pytest.raises(ValueError):unrelated(p,())
