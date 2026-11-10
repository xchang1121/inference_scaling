"""Collect completed formal jobs, evaluate, and package a replayable result."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--skip-collect',action='store_true');a=p.parse_args()
base=Path(__file__).resolve().parent
repo=base.parent/'online_injection_20261109/source'
dest=repo/'experiments/online-injection-cot-bon-20261110'
assert not dest.exists(),'preserve any existing package'
if not a.skip_collect:
    subprocess.run([sys.executable,'-X','utf8',str(base/'collect.py'),'formal'],check=True)
env=os.environ.copy()
env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTHONPATH']=os.pathsep.join([str(base.parent/'online_injection_20261109/devdeps'),str(repo/'src'),str(repo/'scripts'),env.get('PYTHONPATH','')])
for stage in ['smoke','formal']:
    subprocess.run([sys.executable,'-X','utf8',str(repo/'scripts/summarize_cot_guidance.py'),
        '--snapshot',str(base/'snapshot'/stage),'--output',str(base/('reports-'+stage)),
        '--gold',str(base/'data/gold.jsonl'),'--stage',stage],env=env,check=True)
summary=json.loads((base/'reports-formal/summary.json').read_text(encoding='utf-8'))
assert summary['questions']==16 and summary['selected_answers']==1024 and summary['audit']['passed']
assert summary['audit']['raw']['candidate_truncated']==0
manifest=json.loads((base/'data/manifest.json').read_text(encoding='utf-8'))
answers=[json.loads(s) for s in (base/'reports-formal/answers.jsonl').read_text(encoding='utf-8').splitlines()]
assert {r['question_id'] for r in answers}==set(manifest['fresh_ids'])
assert {r['seed'] for r in answers}==set(manifest['seeds'])
for worker in ['a0','a5']:
    folder=base/'snapshot/formal'/worker
    finish=json.loads((folder/'formal-finish.json').read_text())
    assert finish['exit']==0 and not finish['state']['Running']
for name,digest in manifest['files'].items():assert hashlib.sha256((base/'data'/name).read_bytes()).hexdigest()==digest,name
dest.mkdir(parents=True)
def copy(src,relative):
    target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
for name in ['questions.jsonl','smoke-questions.jsonl','gold.jsonl','manifest.json']:
    copy(base/'data'/name,'inputs/'+name)
copy(base/'settings.json','inputs/settings.json')
copy(base/'formal-approval.json','inputs/formal-approval.json')
copy(base/'source-frozen.tgz','generation-source.tar.gz')
copy(repo/'docs/reports/COT_GUIDANCE_PROTOCOL.md','PROTOCOL.md')
for stage in ['smoke','formal']:
    for worker in ['a0','a5']:
        copy(base/'snapshot'/stage/(worker+'.tar.gz'),'raw/'+stage+'/'+worker+'.tar.gz')
    for f in (base/('reports-'+stage)).iterdir():
        if f.is_file():copy(f,'reports/'+stage+'/'+f.name)
for version,folder in [('v1','smoke-logprob-cap-v1'),('v2','smoke-unmatched-checkpoint-v2')]:
    copy(base/('source-frozen-'+version+'.tgz'),'retired/'+version+'/generation-source.tar.gz')
    for worker in ['a0','a5']:
        copy(base/'snapshot'/folder/(worker+'.tar.gz'),'retired/'+version+'/'+worker+'.tar.gz')
for name in ['summarize_cot_guidance.py','summarize_flare_gap.py']:
    copy(repo/'scripts'/name,'analysis/'+name)
for name in ['test_cot_guidance.py','test_cot_guidance_report.py']:
    copy(repo/'tests'/name,'analysis/tests/'+name)
for name in ['replay.py','prepare.py','deploy.py','launch.py','collect.py','approve.py','monitor_remote.py','watch.py','finalize.py']:
    copy(base/name,'tools/'+name)
readme='''# Short evidence in CoT: BoN and conditional IS

Start with [formal results](reports/formal/REPORT.md),
[all selected answers](reports/formal/answers.jsonl), and
[paired BoN budget curves](reports/formal/budget-curves.json).
The full protocol is [PROTOCOL.md](PROTOCOL.md).

16 fresh TechQA questions × 2 seeds × 4 evidence conditions × 8 selector views.
BoN selectors share one pool of 8 candidates. Both static and inline evidence
conditions branch after the same complete-step checkpoint policy; existing
reasoning is preserved. Gold is available only to offline evaluation.

Generation source and settings are frozen. Thinking horizon: 8192 tokens;
final-answer cap: 16384; context: 32768; generated search cap: 65536 including
the checkpoint. Final answers continue in chunks until EOS. Any answer cap
failure would invalidate completion. Actual cost varies between algorithms;
the raw traces and budget curves expose that variation.

Raw archives contain full sampled token IDs, SC/top-5 statistics, exact roots,
candidate pools, selection records, final answers, runtime metadata and a
tokenizer vocabulary subset. The `retired` directory preserves TRAIN-only
pre-formal attempts; those are excluded from the formal results.

Offline replay needs Python and NumPy, without a model or network:

```text
python tools/replay.py --package . --workdir /absolute/path/to/new-replay-folder
```

Replay verifies file hashes and rebuilds both reports byte-for-byte from raw
tokens, probabilities and local gold. `OFFLINE-VERIFICATION.json` records the
verified rebuild. Both task-owned native containers stopped after completion.
'''
(dest/'README.md').write_text(readme,encoding='utf-8')
def manifest_write():
    value={'schema':1,'files':{p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}}
    (dest/'MANIFEST.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
manifest_write()
replay=base/'offline-replay-final'
subprocess.run([sys.executable,'-X','utf8',str(dest/'tools/replay.py'),'--package',str(dest),'--workdir',str(replay)],env=env,check=True)
copy(replay/'OFFLINE-VERIFICATION.json','OFFLINE-VERIFICATION.json')
manifest_write()
print(json.dumps({'package':str(dest),'formal_audit':summary['audit'],'generation_source_sha256':hashlib.sha256((dest/'generation-source.tar.gz').read_bytes()).hexdigest()}),flush=True)
