"""Offline paired F1, full trajectory selection audit, and stop diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

from summarize_flare_gap import bootstrap, grade
from inference_scaling.retrieval.ablation import static_prompt
from inference_scaling.retrieval.cot_guidance import GuidanceConfig, GuidanceLedger
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation, select, select_metropolized, weigh

EOS, CLOSE = 248046, 248069


class Decoder:
    def __init__(self, vocabulary):
        assert vocabulary['decoder']['type']=='ByteLevel'
        alphabet=list(range(33,127))+list(range(161,173))+list(range(174,256))
        codes,extra=alphabet.copy(),0
        for byte in range(256):
            if byte not in alphabet:
                alphabet.append(byte);codes.append(256+extra);extra+=1
        self.mapping=dict(zip(map(chr,codes),alphabet,strict=True))
        self.special=set(vocabulary['special_ids'])
        self.vocab=vocabulary['vocab']

    def __call__(self, ids, skip=False):
        encoded=''.join(self.vocab[str(t)] for t in ids if not skip or t not in self.special)
        return bytes(self.mapping[c] for c in encoded).decode('utf-8',errors='replace')


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def summarize(snapshot, output, gold_path, stage):
    output.mkdir(parents=True,exist_ok=True)
    gold = {x['id']:x['answer'] for x in read(gold_path)}
    selected, checks, raw_counts, physical = [], Counter(), Counter(), set()
    failures, packets = [], {}
    def check(ok, name):
        checks['total']+=1
        if not ok: failures.append(name)
    for folder in sorted(snapshot.iterdir()):
        if not folder.is_dir(): continue
        path=folder/(stage+'.jsonl.gz')
        if not path.exists():path=folder/(stage+'.jsonl')
        if not path.exists():continue
        rows=read(path)
        meta=next(r for r in rows if r['kind']=='metadata')
        cfg=GuidanceConfig(**meta['config'])
        done=[r for r in rows if r['kind']=='completed']
        expected=len(meta['cases'])*len(meta['seeds'])*len(meta['contexts'])*len(meta['methods'])
        measurements=[r for r in rows if r['kind']=='measurement']
        check(len(done)==1 and done[0]['units']==expected==len(measurements),folder.name+':complete')
        check(not any(r['kind']=='failed' for r in rows),folder.name+':no_failure')
        prepared=json.loads((folder/('prepared-'+stage+'.json')).read_text(encoding='utf-8'))
        decode=Decoder(json.loads((folder/('vocabulary-'+stage+'.json')).read_text(encoding='utf-8')))
        check(hashlib.sha256((folder/('prepared-'+stage+'.json')).read_bytes()).hexdigest()==meta['prepared_sha256'],folder.name+':prepared_hash')
        base={c['id']:tuple(c['prompt_ids']) for c in prepared['cases']}
        evidence={r['question_id']:r for r in rows if r['kind']=='evidence'}
        probes={(r['question_id'],r['seed'],r.get('probe_root_sha256')):r for r in rows if r['kind']=='probe'}
        for probe in probes.values():
            for c in probe['calls']:
                raw_counts['probe_calls']+=1
                raw_counts['probe_thinking_eos']+=EOS in c['token_ids']
                if probe.get('evidence_in_probe') and c['prefix_ids']==probe['probe_root_ids']:
                    raw_counts['injected_root_calls']+=1
                    raw_counts['immediate_injected_eos']+=c['token_ids'][:1]==[EOS]
                    raw_counts['immediate_injected_close']+=c['token_ids'][:1]==[CLOSE]
        packets.update({q:bool(e['packet']) for q,e in evidence.items()})
        for row in measurements:
            uid=row['unit_id']+':'+row['context']
            q,s=row['question_id'],row['seed']
            probe=probes[q,s,row.get('probe_root_sha256')]
            packet=evidence[q]['packet']
            inline=row['context'].startswith('inline')
            payload=list(packet['ids']) if row['injected'] else []
            if payload and row['context'].endswith('unrelated'):
                for i,b in enumerate(packet['bodies']):
                    size=b['stop']-b['start']
                    payload[b['start']:b['stop']]=prepared['noise_ids'][i*512:i*512+size]
            check(payload==row['payload_ids'],uid+':identical_packet')
            reason=probe['reason_ids']
            expected_probe_root=base[q] if inline or not payload else static_prompt(base[q],payload,EOS,())
            expected_root=(base[q]+tuple(reason)+tuple(payload) if inline else
                expected_probe_root+tuple(reason))
            if meta.get('schema')==3:
                check(expected_probe_root==tuple(probe['probe_root_ids']),uid+':probe_position')
                check(row['probe_work']==probe['work'],uid+':matched_probe_cost')
            check(expected_root==tuple(row['root_ids']),uid+':placement_preserves_reason')
            check(reason==row['carried_reason_ids'],uid+':no_reset')
            check(token_hash(row['root_ids'])==row['root_sha256'],uid+':root_hash')
            check(sum(len(c['token_ids']) for c in row['calls'])==row['search_work'],uid+':work')
            check(row['search_work']+row['probe_work']<=cfg.search_tokens,uid+':budget')
            seed=SeedStream(SeedStream(s).derive('question',q)).derive('search')
            backend=SimpleNamespace(model_id='offline',tokenizer=SimpleNamespace(eos_token_id=EOS))
            ledger=GuidanceLedger(backend,None,row['root_ids'],(CLOSE,),seed,cfg,carried=probe['statistics'])
            ledger.calls=row['calls']
            for ci,c in enumerate(row['calls']):
                tag=uid+f':call{ci}'
                check(c['prefix_ids'][:len(row['root_ids'])]==row['root_ids'],tag+':immutable_root')
                check(token_hash(c['prefix_ids'])==c['prefix_sha256'],tag+':hash')
                check(len(c['token_ids'])<=c['max_new_tokens'],tag+':length')
                check(all(len(c[k])==len(c['token_ids']) and all(math.isfinite(v) for v in c[k]) for k in ('sc','cn','lp')),tag+':stats')
                check(EOS not in c['token_ids'] or c['token_ids'][-1]==EOS,tag+':EOS_last')
                check(CLOSE not in c['token_ids'] or c['token_ids'][-1]==CLOSE,tag+':close_last')
            audit=row['audit']
            if 'scores' in audit:
                for kind, stored in audit['scores'].items():
                    computed=ledger.scores(row['candidates'],kind)
                    check(all(math.isclose(x,y,abs_tol=1e-10) for x,y in zip(computed,stored,strict=True)),uid+':'+kind)
                    chosen=max(range(len(computed)),key=lambda i:(computed[i],-i))
                    check(audit['selection']['bon_'+kind]==chosen,uid+':selection_'+kind)
                check(len(row['candidates'])==cfg.candidates,uid+':complete_bon')
            kept=()
            for si,record in enumerate(audit.get('steps',[])):
                fixed=kept[:record['generated_length_before']]
                remaining=audit['config']['total_length']-len(fixed)
                width=min(audit['config']['block_size'],remaining)
                carried=int(bool(kept))
                bound=(audit['config']['candidate_count']-carried)*(width+audit['config']['rollout_count']*(remaining-width))+carried*(audit['config']['rollout_count']-1)*(remaining-width)
                check(bound==record['reserved'],uid+f':pool{si}:reservation_formula')
                check(sum(len(c['token_ids']) for c in row['calls'] if c['request_id'].startswith(f'conditional-is:step:{si}:'))==record['work'],uid+f':pool{si}:actual_work')
                blocks,leaves=[],[]
                reward='consilience' if row['method'].endswith('consilience') else 'sc'
                for candidate in record['candidates']:
                    block=tuple(candidate['token_ids']);group=[]
                    for leaf in candidate['rollouts']:
                        suffix=tuple(leaf['token_ids'])
                        score,=ledger.scores([fixed+block+suffix],reward)
                        check(math.isclose(score,leaf['reward'],abs_tol=1e-10),uid+f':pool{si}:reward')
                        check(math.isclose(score/audit['config']['reward_temperature'],leaf['log_weight'],abs_tol=1e-10),uid+f':pool{si}:weight')
                        group.append(RolloutEvaluation(suffix,leaf['reward'],leaf['log_weight']))
                    blocks.append(block);leaves.append(group)
                candidates=weigh(blocks,leaves)
                kernel=audit['config']['selection']
                mi,ki=(select(candidates,SeedStream(seed),'conditional_is',si) if kernel=='sir' else
                    select_metropolized(candidates,SeedStream(seed),'conditional_is',si,group_blocks=kernel=='block_metropolized'))
                check(mi==record['selected_index'] and ki==record['completion_index'],uid+f':pool{si}:selected')
                check(all(math.isclose(c.log_weight,r['log_weight'],abs_tol=1e-10) for c,r in zip(candidates,record['candidates'],strict=True)),uid+f':pool{si}:group_weights')
                check(record['work']<=record['reserved'],uid+f':pool{si}:reserved')
                kept=fixed+candidates[mi].token_ids+candidates[mi].rollouts[ki].token_ids
            if audit.get('steps'):check(kept==tuple(row['candidates'][0]),uid+':final_IS_sequence')
            for i,answer in enumerate(row['answers']):
                check(decode(answer['answer_ids'],True).strip()==answer['answer_text'],uid+f':answer{i}:decoded_text')
                check(not answer['truncated'],uid+f':answer{i}:not_truncated')
                check([t for c in answer['calls'] for t in c['token_ids']]==answer['answer_ids'],uid+f':answer{i}:ids')
                suffix=tuple(row['candidates'][i])
                answer_root=tuple(row['root_ids'])+suffix+(() if suffix[-1:]==(CLOSE,) else (CLOSE,))
                if audit.get('natural_end_before_branching'):
                    answer_root=expected_probe_root+tuple(reason)
                for c in answer['calls']:
                    check(token_hash(answer_root)==c['prefix_sha256'],uid+f':answer{i}:exact_prefix')
                    answer_root+=tuple(c['token_ids'])
                check(len(answer['answer_ids'])<=cfg.answer_tokens,uid+f':answer{i}:cap')
            if row['unit_id'] not in physical:
                physical.add(row['unit_id'])
                for c in row['calls']:
                    raw_counts['search_calls']+=1
                    raw_counts['thinking_eos']+=EOS in c['token_ids']
                    if inline and c['prefix_ids']==row['root_ids'] and row['injected']:
                        raw_counts['injected_root_calls']+=1
                        raw_counts['immediate_injected_eos']+=c['token_ids'][:1]==[EOS]
                        raw_counts['immediate_injected_close']+=c['token_ids'][:1]==[CLOSE]
                for answer in row['answers']:
                    raw_counts['candidate_answers']+=1
                    raw_counts['candidate_empty']+=answer['empty_answer']
                    raw_counts['candidate_truncated']+=answer['truncated']
            for method,i in audit['selection'].items():
                answer=row['answers'][i]
                selected.append({'question_id':q,'seed':s,'context':row['context'],'method':method,
                    'f1':100*grade(answer['answer_text'],gold[q])['f1'],
                    'packet_available':bool(packet),'injected':row['injected'],'answer_text':answer['answer_text'],
                    'answer_tokens':len(answer['answer_ids']),'empty':answer['empty_answer'],
                    'forced_thinking_end':answer['forced_thinking_end'],'eos_inside_thinking':answer['eos_inside_thinking'],
                    'search_tokens':(len(row['candidates'][0]) if method=='single' else row['search_work'])+row['probe_work'],
                    'physical_shared_search_tokens':row['search_work']+row['probe_work'],'evidence_tokens':row['evidence_work'],
                    'all_answer_tokens':sum(len(x['answer_ids']) for x in row['answers']),
                    'seconds':row['seconds'],'selected_index':i,'unit_id':row['unit_id']})
    check(bool(selected),'nonempty_results')
    check(len({(r['question_id'],r['seed'],r['context'],r['method']) for r in selected})==len(selected),'unique_conditions')
    audit_out={'passed':not failures,'checks':checks['total'],'failures':failures,'physical_search_units':len(physical),'raw':dict(raw_counts)}
    (output/'audit.json').write_text(json.dumps(audit_out,indent=2)+'\n',encoding='utf-8')
    if failures:raise RuntimeError(str(failures[:15]))
    groups=defaultdict(list)
    for row in selected:groups[row['context'],row['method']].append(row)
    summary=[]
    for (context,method),rows in groups.items():
        summary.append({'context':context,'method':method,'n':len(rows),'f1':statistics.fmean(r['f1'] for r in rows),
            'packet_present_f1':statistics.fmean(r['f1'] for r in rows if r['packet_available']) if any(r['packet_available'] for r in rows) else None,
            'mean_search_tokens':statistics.fmean(r['search_tokens'] for r in rows),
            'mean_answer_tokens':statistics.fmean(r['answer_tokens'] for r in rows),
            'empty':sum(r['empty'] for r in rows),'forced_thinking_end':sum(r['forced_thinking_end'] for r in rows)})
    summary.sort(key=lambda r:(r['context'],-r['f1']))
    qmeans={}
    for key,rows in groups.items():
        byq=defaultdict(list)
        for r in rows:byq[r['question_id']].append(r['f1'])
        qmeans[key]={q:statistics.fmean(v) for q,v in byq.items()}
    contrasts=[]
    for method in sorted({r['method'] for r in selected}):
        for left,right in [('inline_relevant','static_relevant'),('inline_relevant','inline_unrelated'),('static_relevant','static_unrelated')]:
            a,b=qmeans.get((left,method),{}),qmeans.get((right,method),{})
            if a and a.keys()==b.keys():contrasts.append({'method':method,'left':left,'right':right,**bootstrap(a[q]-b[q] for q in sorted(a))})
    for context in sorted({r['context'] for r in selected}):
        b=qmeans.get((context,'bon_sc'),{})
        for method in sorted({r['method'] for r in selected}-{'bon_sc'}):
            a=qmeans.get((context,method),{})
            if a and a.keys()==b.keys():contrasts.append({'method':context,'left':method,'right':'bon_sc',**bootstrap(a[q]-b[q] for q in sorted(a))})
    result={'stage':stage,'questions':len(packets),'accepted_packets':sum(packets.values()),'selected_answers':len(selected),
        'summary':summary,'contrasts':contrasts,'audit':audit_out}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (output/'answers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    with (output/'scores.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    text=['# CoT guidance: paired F1 comparison','',f"{len(packets)} questions; {sum(packets.values())} accepted evidence packets; {len(selected)} selected-answer views.",
        '', 'F1 is lexical overlap with the reference answer. BoN selectors share the same candidate pool; single is candidate 0 with its logical token cost, not a separately timed run. All eight BoN answers were generated for selector/oracle diagnostics; selected readout cost and total physical readout cost are separate columns in scores.csv.',
        '', '| Context | Method | F1 | Search tokens | Answer tokens | Empty | Forced thinking close |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"| {r['context']} | {r['method']} | {r['f1']:.3f} | {r['mean_search_tokens']:.0f} | {r['mean_answer_tokens']:.0f} | {r['empty']} | {r['forced_thinking_end']} |")
    text+=['','Paired differences below average seeds within each question; bootstrap intervals are descriptive and unadjusted.','']
    for r in contrasts:text.append(f"- {r['method']}, {r['left']} − {r['right']}: {r['difference']:+.3f} F1 points, 95% CI [{r['bootstrap_95'][0]:+.3f}, {r['bootstrap_95'][1]:+.3f}].")
    text+=['','Raw stop diagnostics: `'+json.dumps(dict(raw_counts),sort_keys=True)+'`.',
        '',f"Offline audit: {checks['total']} checks passed. Same search cap does not imply equal realized FLOPs; complete-pool IS and BoN use different amounts of work. Final answer readout and evidence extraction are additional measured costs."]
    (output/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('stage','questions','accepted_packets','selected_answers','audit')}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','formal'],required=True)
    a=p.parse_args()
    summarize(a.snapshot,a.output,a.gold,a.stage)
