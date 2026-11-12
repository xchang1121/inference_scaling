"""Exercise real controller/search code and reject corrupted recorded rewards."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_literature_retrieval import ARMS, run_arm
from summarize_literature_retrieval import audit_row, summarize
from test_literature_retrieval import Bank, Fake, Tap
from inference_scaling.arllm.types import SequenceSample
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.literature import LiteratureController


class NativeFake(Fake):
    def __init__(self):
        super().__init__()
        self.tokenizer=SimpleNamespace(eos_token_id=248046,
            convert_tokens_to_ids=lambda s:{'</think>':248069,'<think>':248068,'<|im_end|>':248046}[s])
    def sample_batch(self,requests):
        self.requests.extend(requests)
        result=[]
        for r in requests:
            text='Check the exact version and configuration carefully before making the requested change.'
            ids=tuple(map(ord,text))+(248069,)
            if ':interim:' in r.request_id or ':answer:' in r.request_id:
                ids=tuple(map(ord,'Restart the configured service.'))+(248046,)
            result.append(SequenceSample(tuple(r.prefix),ids,(-1.,)*len(ids),r.sampling.policy_id,
                self.model_id,r.request_id,'eos' if 248046 in ids else 'stop'))
        return result


def make_row(arm,reference=None,qid='fixture'):
    b=NativeFake();cfg=GuidanceConfig()
    c=LiteratureController(b,Tap(),Bank(b,False),Bank(b,True),(248069,),tuple(map(ord,'gardening noise ')),cfg)
    row=run_arm(b,Tap(),c,{'id':qid,'question':'Question','prompt_ids':[1,248046,2,248046,3]},20261112,arm,cfg,reference)
    row.update(unit_id=qid+':20261112:'+arm,worker='test',seconds=sum(row['stage_seconds'].values())+.1,
        cost={'generated_tokens':row['total_generation_tokens']})
    return b,c,json.loads(json.dumps(row))


def test_every_arm_roundtrips_cost_prefix_and_selection_audit():
    reference=None
    for arm in ARMS:
        b,c,row=make_row(arm,reference)
        if arm=='thinkretrieve':reference=row['handoff']['packets']
        failed=[]
        result=audit_row(row,GuidanceConfig(),lambda ids,skip=False,b=b:b.decode(ids),c.noise,
            lambda ok,name,failed=failed:failed.append(name) if not ok else None)
        assert not failed,failed
        assert result['generation_tokens']==row['cost']['generated_tokens']


def test_report_rejects_tampered_sc_and_intermediate_cost():
    b,c,row=make_row('closed_is')
    changed=deepcopy(row)
    changed['audit']['steps'][0]['candidates'][0]['rollouts'][0]['reward']+=1
    failed=[]
    audit_row(changed,GuidanceConfig(),lambda ids,skip=False:b.decode(ids),c.noise,
        lambda ok,name:failed.append(name) if not ok else None)
    assert any('SC_reward_replay' in n for n in failed)
    b,c,row=make_row('thinkretrieve')
    row['presearch_work']-=1
    failed=[]
    audit_row(row,GuidanceConfig(),lambda ids,skip=False:b.decode(ids),c.noise,
        lambda ok,name:failed.append(name) if not ok else None)
    assert any('presearch_cost' in n for n in failed)


def test_complete_report_contains_all_paired_arms_and_latency(tmp_path):
    import hashlib
    snapshot=tmp_path/'snapshot';folder=snapshot/'test';folder.mkdir(parents=True)
    prepared={'cases':[{'id':q,'prompt_ids':[1,248046,2,248046,3]} for q in ['q1','q2']],
        'noise_ids':list(map(ord,'gardening noise '))}
    pp=folder/'prepared-smoke.json';pp.write_text(json.dumps(prepared))
    rows=[{'kind':'metadata','config':GuidanceConfig().__dict__,'source':{},'seeds':[20261112],
        'cases':['q1','q2'],'arms':list(ARMS),'prepared_sha256':hashlib.sha256(pp.read_bytes()).hexdigest()}]
    for q in ['q1','q2']:
        reference=None
        for arm in ARMS:
            _,_,r=make_row(arm,reference,q)
            if arm=='thinkretrieve':reference=r['handoff']['packets']
            rows.append({'kind':'measurement',**r})
    rows.append({'kind':'completed','units':22})
    (folder/'smoke.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    alphabet=list(range(33,127))+list(range(161,173))+list(range(174,256));codes=alphabet.copy();extra=0
    for byte in range(256):
        if byte not in alphabet:alphabet.append(byte);codes.append(256+extra);extra+=1
    mapping=dict(zip(alphabet,map(chr,codes),strict=True))
    vocab={str(i):mapping[i] for i in range(128)}
    vocab.update({'248046':'<|im_end|>','248069':'</think>','248068':'<think>'})
    (folder/'vocabulary-smoke.json').write_text(json.dumps({'decoder':{'type':'ByteLevel'},
        'special_ids':[248046,248069,248068],'vocab':vocab}),encoding='utf-8')
    gold=tmp_path/'gold.jsonl';gold.write_text(''.join(json.dumps({'id':q,'answer':'Restart the configured service.'})+'\n' for q in ['q1','q2']))
    frozen=tmp_path/'frozen.json';frozen.write_text('{"files":{}}')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'smoke_ids':['q1','q2']}))
    out=tmp_path/'report'
    summarize(snapshot,out,gold,'smoke',frozen,manifest)
    result=json.loads((out/'summary.json').read_text())
    assert result['audit']['passed'] and result['measurements']==22
    assert len(result['summary'])==11
    assert all(r['f1']==100 and r['mean_generation_tokens']>0 and r['geomean_paired_latency_ratio']>0 for r in result['summary'])
    assert not result['audit']['noise_length_mismatches']
