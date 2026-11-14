from pathlib import Path
import json
import statistics as st
import hashlib
import re

import argparse
p=argparse.ArgumentParser(description='Export paired static/online baseline traces for diagnosis.')
p.add_argument('--package',type=Path,required=True)
p.add_argument('--snapshot',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
base=a.package
out=a.output
out.mkdir(parents=True,exist_ok=True)
folder=a.snapshot
from summarize_cot_guidance import Decoder, read
decode = Decoder(json.loads((folder/'vocabulary-formal.json').read_text(encoding='utf-8')))
rows = [r for r in read(folder/'formal.jsonl') if r['kind']=='measurement']
indexed = {(r['question_id'],r['seed'],r['arm']):r for r in rows}
scores = {(r['question_id'],r['seed'],r['arm']):r for r in json.loads((base/'reports/records.json').read_text(encoding='utf-8'))}
gold = {r['id']:r['answer'] for r in read(base/'inputs/gold.jsonl')}
questions = {r['id']:r['question'] for r in read(base/'inputs/questions.jsonl')}
summary=[]
for q in sorted(questions):
    lines=[q, questions[q], '\nGOLD\n'+gold[q]]
    for seed in (20261112,20261113):
        for arm in ('static_facts','online_is'):
            r=indexed[q,seed,arm]; s=scores[q,seed,arm]
            selected=r['candidates'][0]
            events=[r['events'][i] for i in r['selected_event_ids']]
            packets=r['static_packets']+[e['packet'] for e in events]
            lines += ['\n'+'='*80, f'{seed} {arm} F1={s["f1"]:.4f} thinking={len(selected)} answer={s["answer_tokens"]} empty={s["empty"]}',
                'SELECTED PHYSICAL THINKING\n'+decode(r['selected_physical_ids'][len(r['root_ids']):]),
                'ANSWER\n'+r['answer']['answer_text'],
                'ANSWER TOKEN TAIL '+repr(decode(r['answer']['answer_ids'][-12:])),
                'INJECTED PACKETS\n'+'\n'.join(p['id']+'\n'+decode(p['ids']) for p in packets)]
            for e in events:
                lines += [f'EVENT {e["event_id"]} AT {e["position"]}\nQUERY\n'+e['query']]
            for i,step in enumerate(r['audit']['steps']):
                small={k:v for k,v in step.items() if k!='candidates'}
                small['candidates']=[{'length':len(c['token_ids']), 'tail':decode(c['token_ids'][-18:]),'log_weight':c['log_weight'],
                    'rollouts':[{'length':len(t['token_ids']), 'tail':decode(t['token_ids'][-20:]),'reward':t['reward']} for t in c['rollouts']]} for c in step['candidates']]
                lines += [f'STEP {i}\n'+json.dumps(small,ensure_ascii=False)]
            summary.append({'q':q,'seed':seed,'arm':arm,'f1':s['f1'],'thought_tokens':len(selected),'answer_tokens':s['answer_tokens'],
                'empty':s['empty'],'packets':[p['id'] for p in packets], 'selected_events':len(events),
                'generated_reference_headers':decode(selected).count('[Retrieved reference; source data]'),
                'eos_inside_thinking':s['thinking_eos'],'selected_tail':decode(selected[-30:])})
    (out/(q+'.txt')).write_text('\n'.join(lines),encoding='utf-8')
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')

paired=[]
for q in sorted(questions):
    static=st.mean(r['f1'] for r in summary if r['q']==q and r['arm']=='static_facts')
    online=st.mean(r['f1'] for r in summary if r['q']==q and r['arm']=='online_is')
    paired.append({'question':q,'static_f1':static,'online_f1':online,'delta':online-static})
deltas=[]
for r in summary:
    if r['arm']=='online_is':
        s=next(s for s in summary if s['q']==r['q'] and s['seed']==r['seed'] and s['arm']=='static_facts')
        deltas.append((r['f1']-s['f1'],r['empty']))
online_rows=[r for r in rows if r['arm']=='online_is']
selected_events=[r['events'][i] for r in online_rows for i in r['selected_event_ids']]
boundary_count=sum(bool(re.search(r'(?:[.!?][\"\x27)\]]*|\n)\s*$',decode(e['branch_ids'],True))) for e in selected_events)
near=[]
for r in online_rows:
    events=[r['events'][i] for i in r['selected_event_ids']]
    delta=len(r['candidates'][0])-events[-1]['position'] if events else None
    if delta is not None and delta<=2:
        near.append({'unit_id':r['unit_id'],'tokens_after_injection':delta,'empty':r['answer']['empty_answer']})
overall=st.mean(d for d,_ in deltas)
empty_contribution=sum(d for d,empty in deltas if empty)/len(deltas)
echoes=[]
for row in online_rows:
    for call in row['native_calls']:
        text=decode(call['token_ids'])
        at=text.find('[Retrieved reference; source data]')
        if at>=0:
            echoes.append({'unit_id':row['unit_id'],'request_id':call['request_id'],
                           'generated_tokens':len(call['token_ids']),'excerpt':text[max(0,at-80):at+900]})
(out/'generated-reference-echoes.json').write_text(json.dumps(echoes,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
diagnosis={'baseline_manifest_sha256':hashlib.sha256((base/'MANIFEST.json').read_bytes()).hexdigest(),
           'paired_questions':paired,'overall_online_minus_static':overall,
           'empty_pair_contribution_to_overall':empty_contribution,
           'empty_fraction_of_net_gap':empty_contribution/overall,
           'nonempty_29_pairs_mean_difference':st.mean(d for d,empty in deltas if not empty),
           'selected_injections_at_sentence_or_newline_end':boundary_count,
           'selected_injections':len(selected_events),'termination_within2_after_injection':near,
           'native_calls_generating_reference_headers':len(echoes)}
(out/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines=['# 原在线注入落后静态注入：逐题与轨迹诊断','',
       '本报告只分析上一轮冻结的 192 次运行，不修改或补跑其中任何答案。完整物理上下文、证据、最终答案、IS候选分数和选择见每题 .txt。',
       '观察到的轨迹可以定位故障路径，但无法单独证明提示格式、切分随机流或检索内容的因果贡献。','',
       '## 逐题结果（两个种子平均）','','| 题目及轨迹 | 静态 F1 | 在线 F1 | 在线−静态 |','|---|---:|---:|---:|']
for r in sorted(paired,key=lambda r:r['delta']):
    q=r['question'];lines.append(f'| [{q}]({q}.txt) | {r["static_f1"]:.3f} | {r["online_f1"]:.3f} | {r["delta"]:+.3f} |')
lines += ['', '13/16 道题在线 F1 较低；最大的负向差异来自 Q183、Q292、Q025。',
          f'整体差值 {overall:.3f}；3 个空答案配对贡献 {empty_contribution:.3f}，占净差距 {100*empty_contribution/overall:.1f}%。其余 29 对的平均差值仍为 {diagnosis["nonempty_29_pairs_mean_difference"]:.3f}。',
          '该分解仅用于诊断，正式指标仍保留全部空答案。不能把这29对当作修复空答案后的反事实效果。','',
          '## 三个空答案的精确路径','',
          '| 运行 | 生成思考数 | 最后注入 | 注入后输出 | 答案阶段 | 同配对静态 F1 |',
          '|---|---:|---:|---|---|---:|',
          '| Q292 / 20261112 | 514 | 512 | 换行 + EOS（token 248046），共2 token | 因思考已有EOS，不发起答案请求 | 16.018 |',
          '| Q244 / 20261113 | 129 | 128 | EOS（token 248046），1 token | 因思考已有EOS，不发起答案请求 | 8.247 |',
          '| Q218 / 20261113 | 129 | 128 | `</think>`，1 token | 发起请求，仅输出换行和EOS，共2 token | 7.407 |','',
          'Q244 的证据包含相关 crm_resource 命令，Q218 检索到了相关 APAR IT15252；因此它们不能解释为“未检索到资料”。Q292 则同时存在检索跑偏和终止异常。',
          '这三次都没有耗尽预算或发生截断，也不是统计脚本丢失答案。finish_answer 在思考中检测到 EOS 就保留空答案；Q218 的答案解码后确实为空白。',
          'SC 会保留这种退化路径：Q244 空候选 SC=15.623，高于两个正常候选的池 log-weight/4≈15.315、14.847；Q218 短候选 SC=17.503，高于正常长候选约15.101；Q292 后续步骤保留EOS路径 SC=16.755，两个可正常结束的替代路径为16.541和16.639。',
          '这不是审计发现的选择实现偏离：原IS选择已被重放核验。现有SC只评价已生成思考的统计特征，EOS不进入该均值，也没有最终答案完整性/正确性约束。','',
          '## 主要故障机制','',
          f'1. **任意token位置切断语句，证据结束后缺少继续推理的引导。** 选中路径的{len(selected_events)}次注入中，仅{boundary_count}次发生在句末或换行处；其余往往插在半个句子或列表项中。{len(near)}/32条选中路径在最后一次注入后不超过2个token就结束思考，其中3条最终为空。Q244在“CRM’s C”之后被打断，Q218在列表编号“1”后被打断。此现象提示需要改变拼接协议，但还不能排除分段采样随机流的影响。',
          f'此外，有{len(echoes)}段原生候选输出自己生成了 `[Retrieved reference; source data]` 标记及后续伪引用格式。这里统计的是模型新生成的token，不包含外部插入的证据。选中路径没有这些标记，但候选池存在格式续写污染；样例见 generated-reference-echoes.json。',
          '2. **未核实的早期猜测污染查询，也会继续支配后续答案。** Q183/seed12先猜Micro Focus，首包检索到了Micro Focus COBOL编译资料；seed13先认定SolarWinds，即使两次收到IBM的正确指南资料，后续推理和最终答案仍指向SolarWinds。静态组从开始便看到IBM支持指南及PDF链接。',
          '3. **找对文档不等于获得了答案窗口。** Q025静态第二包是192–377窗口，包含具体keytool命令；在线第二包是96–288窗口，重复错误日志并在命令前截止。在线推理还把ITCAM扩展成错误的产品名称，答案更长且混入GUI/CLI猜测。',
          '4. **有效证据量不同。** 静态固定注入2包共512输入token；在线16条路径只有1包、16条有2包，平均384token。它同时改变检索查询、时机、消息角色和实际曝光量，不是单独的“位置消融”。',
          '5. **词F1受答案冗长度影响。** 在线平均答案669token，静态536token；Q183静态约337–365token，在线442–677token。F1偏好与短参考答案重叠的简短输出，不能直接当语义正确率。Q158在线较高F1主要来自重复参数定义，它还断言引号必需，而参考段本身不支持此断言；不能把该分数上涨直接视作事实正确性改进。','',
          '## 值得保留的在线能力与下一轮测试','',
          'Q218在线检索能找出MQ版本回退的相关APAR，而原静态查询命中了GPFS文档；这说明思考扩展查询也有价值。问题约束检索应保留受控的动态查询，而不是完全退回固定查询。',
          '后续方案检验继续推理提示、原问题占优的双查询融合和窗口去重、句末优先注入、后一次检索的不确定性门控，以及未完成第一人称句续接和提前供证。原IS参数、SC评分和EOS处理保留，避免靠补答掩盖失败。',
          '本诊断与方案均使用了已有DEV数据，因此下一轮仍应标注为开发集探索；若要得出泛化结论，需要另外冻结未查看的测试集。','']
(out/'BASELINE_DIAGNOSIS_ZH.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
print(json.dumps({'questions':len(paired),'empty_pair_contribution':empty_contribution,'near_injection_terminations':len(near)}))
