"""Freeze the previous 16-question cohort and a source-supported TRAIN example bank."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import shutil

base = Path(__file__).resolve().parent
old = base.parent/'online_injection_cot_bon_20261110'
data = base/'data'
data.mkdir(exist_ok=True)
for name in ('questions.jsonl','smoke-questions.jsonl','gold.jsonl'):
    assert not (data/name).exists(), 'inputs already frozen'
    shutil.copyfile(old/'data'/name, data/name)
shutil.copyfile(old/'settings.json', base/'settings.json')
manifest = json.loads((old/'data/manifest.json').read_text())
raw_path = base.parent/'online_injection_20261109/techqa_raw/train.json'
raw = json.loads(raw_path.read_text(encoding='utf-8'))
held_ids = set(manifest['fresh_ids']+manifest['smoke_ids'])
held = [r for r in raw if r['id'] in held_ids]
groups = {c['filename'] for r in held for c in r['contexts']}
normalize = lambda s: ' '.join(s.split()).casefold()
words = lambda s: set(re.findall(r'\w+', s.casefold()))
held_words = [words(r['question']) for r in held]
bank, dropped = [], Counter()
for row in raw:
    if not row['id'].startswith('TRAIN_') or row['is_impossible'] or not row['answer'].strip():
        dropped['not_answerable_train'] += 1; continue
    ids = {c['filename'] for c in row['contexts']}
    if ids & groups:
        dropped['shared_held_document'] += 1; continue
    qw = words(row['question'])
    similarity = max(len(qw & h)/max(1,len(qw | h)) for h in held_words)
    if similarity >= .65:
        dropped['near_duplicate_question'] += 1; continue
    answer = normalize(row['answer'])
    support = [c['filename'] for c in row['contexts'] if answer in normalize(c['text'])]
    if not support:
        dropped['not_verbatim_supported'] += 1; continue
    # Keep complete native troubleshooting solutions, never synthesize hidden CoT.
    if len(row['answer'].split()) < 25:
        dropped['solution_too_short'] += 1; continue
    if not re.search(r'\b(?:steps?|run|select|click|configure|check|install|restart|command|procedure|set|enable|disable|edit|remove)\b',row['answer'],re.I):
        dropped['not_procedural'] += 1; continue
    bank.append({'id':row['id'],'question':row['question'].strip(),'solution':row['answer'].strip(),
        'source_documents':sorted(ids),'support_documents':support,'max_held_question_jaccard':similarity})
assert len(bank) >= 50, len(bank)
(data/'examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in bank),encoding='utf-8')
result = {**manifest,'selection':'Exact cohort and seeds from online-injection-cot-bon-20261110; no reselection',
    'bank':{'entries_before_token_filter':len(bank),'rejected':dict(dropped),
        'held_document_groups':sorted(groups),'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'runtime_filter':'complete question+solution body <= 1024 model tokens; no truncation'},
    'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.iterdir())}}
(data/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'cases':len(manifest['fresh_ids']),'bank':len(bank),'rejected':dict(dropped)}))
