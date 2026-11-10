"""Rebuild both reports from frozen raw tokens, without model or network access."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile

p=argparse.ArgumentParser()
p.add_argument('--package',type=Path,required=True)
p.add_argument('--workdir',type=Path,required=True)
a=p.parse_args()
package=a.package.resolve();work=a.workdir.resolve()
assert not work.exists(),'choose a fresh replay directory'
manifest=json.loads((package/'MANIFEST.json').read_text(encoding='utf-8'))
for name,expected in manifest['files'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected,name
work.mkdir(parents=True)
with tarfile.open(package/'generation-source.tar.gz') as tar:
    tar.extractall(work/'source',filter='data')
env=os.environ.copy()
env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTHONPATH']=os.pathsep.join([str(work/'source/src'),str(package/'analysis'),env.get('PYTHONPATH','')])
compared=[]
for stage in ['smoke','formal']:
    snapshot=work/'snapshot'/stage
    for worker in ['a0','a5']:
        dest=snapshot/worker;dest.mkdir(parents=True)
        with tarfile.open(package/'raw'/stage/(worker+'.tar.gz')) as tar:tar.extractall(dest,filter='data')
    output=work/'rebuilt'/stage
    subprocess.run([sys.executable,'-X','utf8',str(package/'analysis/summarize_cot_guidance.py'),
        '--snapshot',str(snapshot),'--output',str(output),'--gold',str(package/'inputs/gold.jsonl'),'--stage',stage],env=env,check=True)
    for f in sorted(output.iterdir()):
        original=package/'reports'/stage/f.name
        assert original.read_bytes()==f.read_bytes(),str(original)
        compared.append(stage+'/'+f.name)
receipt={'passed':True,'verified_package_files':len(manifest['files']),'byte_identical_reports':compared,
    'generation_sha256':hashlib.sha256((package/'generation-source.tar.gz').read_bytes()).hexdigest(),
    'raw_archives':{p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (package/'raw').rglob('*.tar.gz')}}
(work/'OFFLINE-VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
