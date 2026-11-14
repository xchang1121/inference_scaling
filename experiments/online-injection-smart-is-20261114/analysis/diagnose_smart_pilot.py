"""Verify the controlled TRAIN comparison of inline continuation formats."""
from pathlib import Path
import argparse
import json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--snapshot',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();records=[]
for worker in ('a0','a1','a3'):
    rows=[json.loads(line) for line in (a.snapshot/worker/'smoke.jsonl').read_text(encoding='utf-8').splitlines()]
    idx={(r['question_id'],r['arm']):r for r in rows if r['kind']=='measurement'}
    old=idx['TRAIN_Q545','online_is'];new=idx['TRAIN_Q545','resume']
    first=lambda r:[c for c in r['native_calls'] if len(c['logical_prefix_ids'])==len(r['root_ids'])]
    before_old,before_new=first(old),first(new)
    assert len(before_old)==len(before_new)==3
    step=new['audit']['steps'][0]
    for i,(left,right) in enumerate(zip(before_old,before_new,strict=True)):
        assert left['seed']==right['seed'] and left['uniform_offset']==right['uniform_offset']==0
        assert left['token_ids']==right['token_ids'] and left['prefix_ids']==right['prefix_ids']
        old_event=next(e for e in old['events'] if e['branch_ids']==left['token_ids'])
        new_event=next(e for e in new['events'] if e['branch_ids']==right['token_ids'])
        assert old_event['packet']==new_event['packet']
        old_after=next(c for c in old['native_calls'] if c['logical_request_id']==left['logical_request_id'] and c['uniform_offset']==128)
        new_after=next(c for c in new['native_calls'] if c['logical_request_id']==right['logical_request_id'] and c['uniform_offset']==128)
        assert old_after['seed']==new_after['seed'] and old_after['max_new_tokens']==new_after['max_new_tokens']
        assert new_after['prefix_ids']==old_after['prefix_ids']+new['resume_ids']
        records.append({'worker':worker,'question_id':'TRAIN_Q545','seed':20261112,'candidate':i,
                        'same_first128_generated_tokens':True,'same_evidence':True,'same_continuation_rng_and_limit':True,
                        'only_extra_input_tokens':len(new['resume_ids']),
                        'old_continuation_first4':old_after['token_ids'][:4],
                        'resume_continuation_tokens':new_after['token_ids'],
                        'resume_finish_reason':new_after['finish_reason'],
                        'SC_reward':step['candidates'][i]['rollouts'][0]['reward'],
                        'selected':step['selected_index']==i,
                        'selected_empty_answer':new['answer']['empty_answer']})
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'controlled-format-check.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8',newline='\n')
lines=['# TRAIN上的命令式提示反例','',
       '六组TRAIN试验共36次，未计算F1。TRAIN_Q545/20261112的原在线与resume比较，在A0/A1/A3三卡复现相同的终止模式。',
       '三个候选各自的前128个生成token、源证据、续写seed、uniform offset和长度上限完全相同。resume的唯一额外上下文是34个token的命令式继续推理提示。',
       '原在线三个候选都继续生成普通文本；resume两个候选立刻输出思考结束标记，一个立刻EOS。IS最终选择了EOS候选。', '',
       '| 卡 | 候选 | 提示后生成token | SC | 被选中 |','|---|---:|---|---:|---|']
for r in records:
    lines.append(f'| {r["worker"]} | {r["candidate"]} | {r["resume_continuation_tokens"]} | {r["SC_reward"]:.4f} | {r["selected"]} |')
lines+=['','token 248069为思考结束，248046为EOS。两条思考正常结束的候选没有单独读出答案，因此不能推断其最终答案一定非空。',
        '这是单题、单种子的格式对照，三卡重复不等于三个独立问题。它支持“注入后的提示形态会显著改变终止行为”，不支持“加继续推理提示总能改善”或泛化效果的结论。',
        '据此在正式DEV生成前增加bridge（未完成的第一人称句）与early（同格式提前到32/256）两组，保留原六组；全部八组随后重新做完整性验证。','']
(a.output/'PILOT_FORMAT_ZH.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
print(json.dumps({'controlled_pairs':len(records),'workers':3,'question_seed_units':1,'verified':True}))
