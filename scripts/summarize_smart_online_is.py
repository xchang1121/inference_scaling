"""Paired exploratory metrics for eight frozen retrieval conditions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

from audit_smart_online_is import audit_snapshot
from summarize_cot_guidance import Decoder, read
from summarize_flare_gap import bootstrap, grade
from inference_scaling.retrieval.smart_online_is import boundary

ARMS = ('static_facts', 'online_is', 'resume', 'anchored', 'boundary', 'adaptive', 'bridge', 'early')
LABELS = {'static_facts': '静态事实 + IS', 'online_is': '原在线注入', 'resume': '在线 + 继续推理提示',
          'anchored': '问题约束双查询融合', 'boundary': '句子边界优先注入', 'adaptive': '不确定性门控注入',
          'bridge': '第一人称未完成句续接', 'early': '提前至32/256的第一人称续接'}
PAIRS = [(a, ref) for a in ARMS[2:] for ref in ('online_is', 'static_facts')]+[
    ('anchored', 'resume'), ('boundary', 'anchored'), ('adaptive', 'boundary'), ('bridge', 'resume'), ('early', 'bridge')]


def aggregate(records):
    idx = {(r['question_id'], r['seed'], r['arm']): r for r in records}
    qs = sorted({r['question_id'] for r in records}); seeds = sorted({r['seed'] for r in records})
    assert len(idx) == len(records) == len(qs)*len(seeds)*len(ARMS)
    assert set(idx) == {(q,s,a) for q in qs for s in seeds for a in ARMS}
    arms=[]
    for arm in ARMS:
        group=[r for r in records if r['arm']==arm]
        qf1=[st.fmean(idx[q,s,arm]['f1'] for s in seeds) for q in qs]
        item={'arm':arm,'n':len(group),'f1':st.fmean(qf1),'f1_question_bootstrap_95':bootstrap(qf1)['bootstrap_95']}
        for key in ('generated_tokens','search_tokens','answer_tokens','thought_tokens','seconds','retrieval_seconds',
                    'selected_injected_tokens','retrieval_searches','native_prefill_tokens','native_calls','selected_injections'):
            item['mean_'+key]=st.fmean(r[key] for r in group)
        for key in ('empty','truncated','thinking_eos','forced_thinking_end','termination_within2_after_injection'):
            item[key]=sum(r[key] for r in group)
        for key in ('gate_evaluations','gate_skips','selected_boundary_injections','eligible_no_evidence','reference_header_echo_calls'):
            item[key]=sum(r.get(key,0) for r in group)
        arms.append(item)
    for r in arms:
        r['tokens_relative_static']=r['mean_generated_tokens']/arms[0]['mean_generated_tokens']
        r['latency_relative_static']=r['mean_seconds']/arms[0]['mean_seconds']
    pairs=[]
    for left,right in PAIRS:
        deltas=[st.fmean(idx[q,s,left]['f1']-idx[q,s,right]['f1'] for s in seeds) for q in qs]
        pairs.append({'left':left,'right':right,**bootstrap(deltas),
                      'question_deltas':dict(zip(qs,deltas,strict=True)),
                      'wins_ties_losses':[sum(d>0 for d in deltas),sum(d==0 for d in deltas),sum(d<0 for d in deltas)]})
    return {'questions':len(qs),'seeds':seeds,'measurements':len(records),'arms':arms,'paired_f1':pairs}


def summarize(snapshot, output, gold_path, frozen_path, protocol_path):
    output.mkdir(parents=True,exist_ok=True)
    gold={r['id']:r['answer'] for r in read(gold_path)}
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    records=[]; audits=[]; setup=[]; identities=[]; failures=[]; traces=[]
    for folder in sorted(snapshot.iterdir()):
        if not folder.is_dir() or not (folder/'formal.jsonl').exists():continue
        audits.append(audit_snapshot(folder,'formal',frozen_path,output/('audit-'+folder.name+'.json')))
        rows=read(folder/'formal.jsonl');meta=next(r for r in rows if r['kind']=='metadata')
        engine=next(r for r in rows if r['kind']=='engine')
        decode=Decoder(json.loads((folder/'vocabulary-formal.json').read_text(encoding='utf-8')))
        cases={c['id']:c for c in json.loads((folder/'prepared-formal.json').read_text(encoding='utf-8'))['cases']}
        assert tuple(meta['arms'])==ARMS and meta['seeds']==protocol['seeds']
        identities.append({'worker':folder.name,'source':meta['source'],'settings_sha256':meta['settings_sha256'],
                           'raw_sha256':hashlib.sha256((folder/'formal.jsonl').read_bytes()).hexdigest()})
        start=json.loads((folder/'formal-start.json').read_text());finish=json.loads((folder/'formal-finish.json').read_text())
        setup.append({'worker':folder.name,'job_seconds':finish['finished']-start['started'],
                      'container_stopped':not finish['state']['Running'],'index_seconds':engine['fact_build_seconds']})
        for r in rows:
            if r['kind']!='measurement':continue
            a=r['answer'];graded=grade(a['answer_text'],gold[r['question_id']])
            selected=[r['events'][i] for i in r['selected_event_ids']]
            delta=len(r['candidates'][0])-selected[-1]['position'] if selected else None
            searches=sum(len(d['retrieval'].get('searches',[d['retrieval']])) for d in r.get('decisions',[]) if 'retrieval' in d)
            if 'decisions' not in r:searches=len(r['events'])+len(r['static_retrievals'])
            rec={'question_id':r['question_id'],'seed':r['seed'],'arm':r['arm'],'worker':folder.name,
                 'f1':100*graded['f1'],'exact_match':graded['exact_match'],'answer_text':a['answer_text'],
                 'generated_tokens':r['total_generation_tokens'],'search_tokens':r['search_work'],
                 'answer_tokens':len(a['answer_ids']),'thought_tokens':len(r['candidates'][0]),'seconds':r['seconds'],
                 'retrieval_seconds':r['stage_seconds']['materialization_seconds']+r['stage_seconds'].get('static_retrieval_seconds',0.),
                 'selected_injected_tokens':sum(len(e['injected_ids']) for e in selected)+len(r['static_injected_ids']),
                 'selected_injections':len(selected)+len(r['static_packets']), 'retrieval_searches':searches,
                 'native_prefill_tokens':r['cost'].get('prefill_tokens',0),'native_calls':len(r['native_calls']),
                 'empty':a['empty_answer'],'truncated':a['truncated'],'thinking_eos':a['eos_inside_thinking'],
                 'forced_thinking_end':a['forced_thinking_end'],
                 'termination_within2_after_injection':delta is not None and delta<=2,
                 'tokens_after_last_injection':delta,
                 'gate_evaluations':sum(d['surprisal_last32'] is not None for d in r.get('decisions',[])),
                 'gate_skips':sum(d['surprisal_last32'] is not None and not d['eligible'] for d in r.get('decisions',[])),
                 'selected_boundary_injections':sum(boundary(decode(e['branch_ids'],True)) for e in selected),
                 'eligible_no_evidence':sum(d['eligible'] and d['event_id'] is None for d in r.get('decisions',[])),
                 'reference_header_echo_calls':sum('[Retrieved reference; source data]' in decode(c['token_ids']) for c in r['native_calls']),
                 'selected_positions':[e['position'] for e in selected],
                 'selected_documents':[p['id'] for p in r['static_packets']]+[e['packet']['id'] for e in selected]}
            records.append(rec)
            traces.append({'unit_id':r['unit_id'],'question':cases[r['question_id']]['question'],
                           'root_context':decode(r['root_ids']),
                           'selected_physical_thinking':decode(r['selected_physical_ids'][len(r['root_ids']):]),
                           'answer':a['answer_text'],'selected_positions':rec['selected_positions'],
                           'selected_documents':rec['selected_documents'],
                           'f1':rec['f1'],'empty':rec['empty'],
                           'IS_selection':[{'before':s['generated_length_before'],'selected_index':s['selected_index'],
                                            'completion_index':s['completion_index'],
                                            'candidate_log_weights':[c['log_weight'] for c in s['candidates']],
                                            'rollout_rewards':[[t['reward'] for t in c['rollouts']] for c in s['candidates']]}
                                           for s in r['audit']['steps']]})
            if a['empty_answer']:
                failures.append({'unit_id':r['unit_id'],'worker':folder.name,'last_injection_position':selected[-1]['position'] if selected else None,
                                 'thought_tokens':len(r['candidates'][0]),'thought_tail':decode(r['candidates'][0][-96:]),
                                 'answer_tokens':a['answer_ids'],'decoded_answer_with_specials':decode(a['answer_ids']),
                                 'eos_inside_thinking':a['eos_inside_thinking']})
    assert {r['question_id'] for r in records}==set(protocol['cases'])
    assert all(len({r['worker'] for r in records if r['question_id']==q})==1 for q in protocol['cases'])
    assert all(i['source']==identities[0]['source'] and i['settings_sha256']==identities[0]['settings_sha256'] for i in identities)
    result=aggregate(records)
    result.update(audit={'passed':all(a['passed'] for a in audits),'checks':sum(a['checks'] for a in audits)},
                  identities=identities,setup=setup,total_generated_tokens=sum(r['generated_tokens'] for r in records),
                  total_pipeline_hours=sum(r['seconds'] for r in records)/3600,
                  total_device_hours=sum(s['job_seconds'] for s in setup)/3600)
    def write(name,data):
        (output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    write('summary.json',result);write('records.json',records);write('empty-answers.json',failures)
    write('selected-traces.json',traces)
    with (output/'scores.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=['# 更智能在线检索：探索性配对结果','',
           '同一固定 16 题 DEV 集、2 个种子、8 组，共 256 次全新原生运行。每道题所有组在同一卡执行。',
           '原在线与静态基线在本轮重新运行；F1 为词重叠指标（0–100）。', '',
           '| 方法 | F1 | 平均总生成 token | 平均耗时（秒） | 空答案 | 截断 | 注入后 ≤2 token 结束思考 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['arms']:
        lines.append(f'| {LABELS[r["arm"]]} | {r["f1"]:.3f} | {r["mean_generated_tokens"]:.1f} | {r["mean_seconds"]:.2f} | {r["empty"]} | {r["truncated"]} | {r["termination_within2_after_injection"]} |')
    lines+=['','## 配对 F1 差异','','| 比较 | 差值 | 95% 问题级 bootstrap 区间 |','|---|---:|---:|']
    for p in result['paired_f1']:
        lo,hi=p['bootstrap_95']
        lines.append(f'| {LABELS[p["left"]]} − {LABELS[p["right"]]} | {p["difference"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
    lines+=['','## 逐题 F1（两个种子平均）','','| 题目 | '+' | '.join(LABELS[a] for a in ARMS)+' |',
            '|---|'+'---:|'*len(ARMS)]
    for q in sorted(protocol['cases']):
        lines.append('| '+q+' | '+' | '.join(f'{st.fmean(r["f1"] for r in records if r["question_id"]==q and r["arm"]==a):.2f}' for a in ARMS)+' |')
    lines+=['','## 限制与计数口径','',
            '- 这轮是根据此前逐题结果设计的探索实验，开发集已被查看，不能当作独立测试集；未对多重比较调整区间。',
            '- 原 IS 内核、SC 评分、M=3、K=2、block=512、温度0.25及生成预算不变；没有固定预生成前缀。',
            '- 六种方案都是无工具调用的外部控制器。继续推理提示增加输入 token；边界策略仍有超时强制插入，不能保证每次都位于完整句末。',
            '- 边界方案改变原生分段及 offset 派生随机种子。不同策略之间不共享完全相同的连续采样流，观察到的变化不能全部归因于语义位置。',
            '- 不确定性门控使用最近32 token的平均惊讶度≥0.8 nats或最近256 token的固定不确定措辞，只决定后一次检索。阈值在正式评测前固定。',
            '- 不重试空答案、不抑制 EOS、不修补读出；所有运行纳入均值。总生成量包括被丢弃候选、rollout及答案；检索与读出均计入单次延迟。',
            '- 查询融合、去重、提示和位置是递进组合，不能由该表分别估计每个改动的独立因果效应。',
            f'- 完整审计 {result["audit"]["checks"]} 项通过；生成 {result["total_generated_tokens"]:,} token，累计测量 {result["total_pipeline_hours"]:.3f} 小时，设备 {result["total_device_hours"]:.3f} 小时。', '']
    (output/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot','output','gold','frozen','protocol'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=summarize(a.snapshot,a.output,a.gold,a.frozen,a.protocol)
    print(json.dumps({k:r[k] for k in ('measurements','questions','audit','total_generated_tokens')}))
