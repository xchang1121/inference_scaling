"""Replay the completed native smoke with only the frozen source and audit files."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

base=Path(__file__).resolve().parent
repo=base.parent/'online_injection_20261109/source'
work=base/'offline-smoke-v2'
assert not work.exists()
work.mkdir()
with tarfile.open(base/'source-frozen.tgz') as tar:
    tar.extractall(work/'source',filter='data')
with tarfile.open(base/'snapshot/smoke/a3.tar.gz') as tar:
    tar.extractall(work/'snapshot/a3',filter='data')
(work/'analysis').mkdir()
for name in ['summarize_literature_retrieval.py','summarize_cot_guidance.py','summarize_flare_gap.py']:
    shutil.copyfile(repo/'scripts'/name,work/'analysis'/name)
env=dict(os.environ)
env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTHONPATH']=os.pathsep.join(map(str,[work/'source/src',work/'analysis',repo.parent/'devdeps']))
subprocess.run([sys.executable,'-X','utf8',str(work/'analysis/summarize_literature_retrieval.py'),
    '--snapshot',str(work/'snapshot'),'--output',str(work/'rebuilt'),'--gold',str(base/'data/gold.jsonl'),
    '--stage','smoke','--frozen',str(base/'FROZEN_IS.json'),'--manifest',str(base/'data/manifest.json')],env=env,check=True)
verified=[]
for path in sorted((work/'rebuilt').iterdir()):
    assert path.read_bytes()==(base/'reports-smoke'/path.name).read_bytes(),path.name
    verified.append(path.name)
receipt={'passed':True,'byte_identical':verified,'generation_sha256':hashlib.sha256((base/'source-frozen.tgz').read_bytes()).hexdigest()}
(work/'VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
