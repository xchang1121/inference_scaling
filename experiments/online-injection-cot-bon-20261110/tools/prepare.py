from pathlib import Path
import hashlib
import json

base = Path(__file__).resolve().parent
old = base.parent/'online_injection_grounded_is_20261110'
raw_path = base.parent/'online_injection_20261109/techqa_raw/train.json'
raw = json.loads(raw_path.read_text(encoding='utf-8'))
byid = {x['id']: x for x in raw}
exclude = set(json.loads((old/'data/excluded-groups.json').read_text()))
prior = [json.loads(s) for s in (old/'data/gold.jsonl').read_text(encoding='utf-8').splitlines()]
for row in prior:
    exclude.update(row['document_ids'])
ranked = sorted((x for x in raw if x['id'].startswith('DEV_') and not x['is_impossible'] and x['answer'].strip()),
    key=lambda x: hashlib.sha256(('cot-bon-20261110:'+x['id']).encode()).hexdigest())
chosen, used = [], set(exclude)
for row in ranked:
    groups = {c['filename'] for c in row['contexts']}
    if groups & used:
        continue
    chosen.append(row['id'])
    used.update(groups)
    if len(chosen) == 16:
        break
assert len(chosen) == 16
smoke = ['TRAIN_Q545', 'TRAIN_Q559']
assert all(q in byid for q in smoke)
data = base/'data'
data.mkdir(exist_ok=True)
def write(name, rows):
    p = data/name
    assert not p.exists(), p
    p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows), encoding='utf-8')
def question(q, cohort):
    return {'id':q,'question':byid[q]['question']+'\n\nIn the final answer, directly address the question and include the necessary technical details. Avoid repeating the analysis.',
        'dataset':'techqa','category':'technical_qa','cohort':cohort}
write('questions.jsonl',[question(q,'fresh') for q in chosen])
write('smoke-questions.jsonl',[question(q,'smoke') for q in smoke])
write('gold.jsonl',[{'id':q,'answer':byid[q]['answer'],'document_ids':[c['filename'] for c in byid[q]['contexts']]} for q in chosen+smoke])
(data/'manifest.json').write_text(json.dumps({'fresh_ids':chosen,'smoke_ids':smoke,'excluded_document_groups':sorted(exclude),
    'selection':'first 16 SHA256 cot-bon-20261110:id; answerable DEV; disjoint gold document groups from earlier selection exclusions and previous formal/smoke gold groups',
    'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),'seeds':[20261112,20261113],
    'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}},indent=2)+'\n',encoding='utf-8')
settings = json.loads((old/'settings.json').read_text())
settings['ar']['engine']['context_window'] = 32768
settings['ar']['engine']['vllm']['max_num_seqs'] = 4
settings['ar']['max_new_tokens'] = 24576
(base/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
print(json.dumps({'fresh':chosen,'smoke':smoke,'context':32768}))
