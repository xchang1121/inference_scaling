from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

base = Path(__file__).resolve().parent
repo = base.parent / 'online_injection_20261109/source'
for name, digest in json.loads((base / 'FROZEN_IS.json').read_text())['files'].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
source = base / 'source-frozen.tgz'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'f4689d610481f2c20c6f992276506837a135f86700922c94cf84f0306f3075b5'
shutil.copy2(source, base / 'source-frozen-v1.tgz')
for src, dst in [('snapshot/smoke', 'snapshot/smoke-template-v1'), ('reports-smoke', 'reports-smoke-template-v1')]:
    shutil.copytree(base / src, base / dst, dirs_exist_ok=True)
script = r'''
from pathlib import Path
import fcntl,hashlib,json,shutil,subprocess
root=Path('/data/disk/changxiangyu/agentic_scaling')
dep=root/'deployment/ascend/online-injection-literature-20261112'
out=root/'results/online-injection-literature-20261112-a3'
assert out.resolve()==Path('/data/disk/changxiangyu/agentic_scaling/results/online-injection-literature-20261112-a3')
assert not (dep/'FORMAL-FROZEN.json').exists()
info=json.loads(subprocess.check_output(['docker','inspect','agentic-scaling-changxiangyu-literature-a3'],text=True))[0]
assert not info['State']['Running']
assert (out/'diagnostic.exit').read_text().strip()=='0'
with (out/'job.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    archive=out/'attempts/smoke-template-v1'
    archive.mkdir(parents=True,exist_ok=True)
    for name in ['source-frozen.tgz','SOURCE-FROZEN.json']:
        dest=archive/name
        if dest.exists():assert dest.read_bytes()==(dep/name).read_bytes()
        else:shutil.copy2(dep/name,dest)
    moved=[]
    for p in list(out.iterdir()):
        if p.is_file() and (p.name.startswith('smoke') or p.name=='vocabulary-smoke.json'):
            assert p.resolve().parent==out.resolve()
            dest=archive/p.name
            assert not dest.exists(),str(dest)
            shutil.move(p,dest);moved.append(p.name)
    print(json.dumps({'preserved':str(archive),'moved':moved,'source_sha256':hashlib.sha256((archive/'source-frozen.tgz').read_bytes()).hexdigest()}))
'''
(base / 'preserve-v1-remote.py').write_text(script, encoding='utf-8')
subprocess.run([sys.executable, '-X', 'utf8', str(base.parent/'online_injection_scaling_20261109/remote.py'), 'a', '--file', str(base/'preserve-v1-remote.py'), '--timeout', '90'], check=True)
print('Local V1 artifacts preserved; all 17 IS/reward hashes unchanged.')
