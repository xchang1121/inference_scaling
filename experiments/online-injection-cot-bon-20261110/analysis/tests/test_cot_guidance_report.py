import hashlib
import json
from pathlib import Path

import pytest

from inference_scaling.retrieval.cot_guidance import GuidanceConfig, GuidanceLedger, search
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.arllm.types import SequenceSample
from dataclasses import asdict, replace
from types import SimpleNamespace


def test_offline_report_recomputes_shared_pool_and_rejects_tampering(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from summarize_cot_guidance import summarize
    close,eos=248069,248046
    cfg=replace(GuidanceConfig(),candidates=2,thinking_tokens=10)
    class Backend:
        model_id='toy'
        tokenizer=SimpleNamespace(eos_token_id=eos)
        def sample_batch(self, requests):
            return [SequenceSample(r.prefix,(65,66,close),(-1.,-2.,-3.),r.sampling.policy_id,self.model_id,r.request_id,'stop') for r in requests]
    class Tap:
        def take(self, request, sample):return (1.,2.,3.),(4.,5.,6.)
    base=[1,eos,2,eos,3]
    ledger=GuidanceLedger(Backend(),Tap(),base,(close,),1,cfg)
    candidates,audit=search(ledger,'bon')
    answer={'answer_text':'A','answer_ids':[65,eos],'truncated':False,'empty_answer':False,'forced_thinking_end':False,'eos_inside_thinking':False}
    answers=[{**answer,'calls':[{'prefix_sha256':token_hash(tuple(base)+candidate),'token_ids':[65,eos]}]} for candidate in candidates]
    snapshot=tmp_path/'snapshot';folder=snapshot/'a0';folder.mkdir(parents=True)
    prepared={'cases':[{'id':'toy','prompt_ids':base}],'noise_ids':[]}
    prepared_path=folder/'prepared-smoke.json';prepared_path.write_text(json.dumps(prepared))
    (folder/'vocabulary-smoke.json').write_text(json.dumps({'decoder':{'type':'ByteLevel'},'special_ids':[eos],'vocab':{'65':'A',str(eos):'<end>'}}))
    rows=[{'kind':'metadata','config':asdict(cfg),'cases':['toy'],'seeds':[1],'contexts':['static_relevant'],'methods':['bon'],
           'prepared_sha256':hashlib.sha256(prepared_path.read_bytes()).hexdigest()},
        {'kind':'evidence','question_id':'toy','packet':None},
        {'kind':'probe','question_id':'toy','seed':1,'reason_ids':[],'calls':[],'statistics':{'sc':[],'cn':[],'lp':[]}},
        {'kind':'measurement','unit_id':'toy','question_id':'toy','seed':1,'context':'static_relevant','method':'bon',
         'injected':False,'payload_ids':[],'root_ids':base,'root_sha256':token_hash(base),'carried_reason_ids':[],
         'search_work':ledger.work,'probe_work':0,'evidence_work':0,'calls':ledger.calls,'candidates':candidates,'answers':answers,'audit':audit,'seconds':1},
        {'kind':'completed','units':1}]
    log=folder/'smoke.jsonl'
    log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    gold=tmp_path/'gold.jsonl';gold.write_text(json.dumps({'id':'toy','answer':'A'})+'\n')
    summarize(snapshot,tmp_path/'report',gold,'smoke')
    result=json.loads((tmp_path/'report/summary.json').read_text())
    assert result['audit']['passed'] and result['selected_answers']==4
    rows[3]['audit']['scores']['sc']=[99.,99.]
    log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(RuntimeError,match='sc'):
        summarize(snapshot,tmp_path/'tampered',gold,'smoke')
