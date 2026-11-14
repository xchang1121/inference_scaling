"""Verify frozen package and byte-reproduce native audits, metrics and plots."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--destination',type=Path,required=True)
p.add_argument('--plot-python',type=Path)
p.add_argument('--baseline-package',type=Path)
a=p.parse_args();base=Path(__file__).resolve().parent
manifest=json.loads((base/'MANIFEST.json').read_text())
for n,h in manifest['files'].items():
    assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h,n
a.destination.mkdir(parents=True,exist_ok=False)
source=a.destination/'source'
with tarfile.open(base/'source-frozen.tgz') as t:t.extractall(source,filter='data')
for f in (base/'analysis').glob('*.py'):shutil.copyfile(f,source/'scripts'/f.name)
for stage in ('smoke','formal','smoke-pilot-v1'):
    for f in (base/'raw'/stage).glob('a*.tar.gz'):
        with tarfile.open(f) as t:t.extractall(a.destination/'snapshot'/stage/f.name.removesuffix('.tar.gz'),filter='data')
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join([str(source/'src'),str(source/'scripts'),env.get('PYTHONPATH','')])
audits=[];compared=[]
for worker in ('a0','a1','a3'):
    path=a.destination/('audit-smoke-'+worker+'.json')
    subprocess.run([sys.executable,str(source/'scripts/audit_smart_online_is.py'),
        '--folder',str(a.destination/'snapshot/smoke'/worker),'--frozen',str(base/'inputs/FROZEN_IS.json'),
        '--output',str(path)],env=env,check=True)
    assert path.read_bytes()==(base/path.name).read_bytes(),path.name
    audits.append(json.loads(path.read_text()));compared.append(path.name)
combined={'passed':all(x['passed'] for x in audits),'quality_scored':False,
          'units':sum(x['units'] for x in audits),'checks':sum(x['checks'] for x in audits),'workers':audits}
assert (json.dumps(combined,indent=2)+'\n').encode()==(base/'audit-smoke.json').read_bytes()
compared.append('audit-smoke.json')
pilot_audits=a.destination/'pilot-audits';pilot_audits.mkdir()
for worker in ('a0','a1','a3'):
    path=pilot_audits/('audit-smoke-'+worker+'.json')
    subprocess.run([sys.executable,str(source/'scripts/audit_smart_online_is.py'),
        '--folder',str(a.destination/'snapshot/smoke-pilot-v1'/worker),'--frozen',str(base/'inputs/FROZEN_IS.json'),
        '--output',str(path)],env=env,check=True)
    assert path.read_bytes()==(base/'raw/smoke-pilot-v1'/path.name).read_bytes(),path.name
    compared.append('pilot/'+path.name)
subprocess.run([sys.executable,str(source/'scripts/diagnose_smart_pilot.py'),
    '--snapshot',str(a.destination/'snapshot/smoke-pilot-v1'),'--output',str(a.destination/'pilot-analysis')],env=env,check=True)
for f in (base/'pilot-analysis').iterdir():
    assert (a.destination/'pilot-analysis'/f.name).read_bytes()==f.read_bytes(),f.name
    compared.append('pilot-analysis/'+f.name)
report=a.destination/'reports'
subprocess.run([sys.executable,str(source/'scripts/summarize_smart_online_is.py'),
    '--snapshot',str(a.destination/'snapshot/formal'),'--output',str(report),
    '--gold',str(base/'inputs/gold.jsonl'),'--frozen',str(base/'inputs/FROZEN_IS.json'),
    '--protocol',str(base/'inputs/PROTOCOL.json')],env=env,check=True)
for f in (base/'reports').iterdir():
    assert (report/f.name).read_bytes()==f.read_bytes(),f.name
    compared.append('reports/'+f.name)
if a.plot_python:
    penv=dict(os.environ);penv.pop('PYTHONPATH',None)
    subprocess.run([str(a.plot_python),str(source/'scripts/plot_smart_online_is.py'),
        '--summary',str(report/'summary.json'),'--output',str(a.destination/'figures')],env=penv,check=True)
    for f in (base/'figures').iterdir():
        assert (a.destination/'figures'/f.name).read_bytes()==f.read_bytes(),f.name
        compared.append('figures/'+f.name)
if a.baseline_package:
    ref=json.loads((base/'inputs/baseline-reference.json').read_text())
    assert hashlib.sha256((a.baseline_package/'MANIFEST.json').read_bytes()).hexdigest()==ref['manifest_sha256']
    baseline_manifest=json.loads((a.baseline_package/'MANIFEST.json').read_text())
    for name,h in baseline_manifest['files'].items():
        assert hashlib.sha256((a.baseline_package/name).read_bytes()).hexdigest()==h,name
    raw=a.baseline_package/'raw/formal/a3.tar.gz'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==baseline_manifest['files']['raw/formal/a3.tar.gz']
    with tarfile.open(raw) as t:t.extractall(a.destination/'baseline-snapshot',filter='data')
    subprocess.run([sys.executable,str(source/'scripts/diagnose_online_is_failures.py'),
        '--package',str(a.baseline_package),'--snapshot',str(a.destination/'baseline-snapshot'),
        '--output',str(a.destination/'baseline-analysis')],env=env,check=True)
    for f in (base/'baseline-analysis').iterdir():
        assert (a.destination/'baseline-analysis'/f.name).read_bytes()==f.read_bytes(),f.name
        compared.append('baseline-analysis/'+f.name)
print(json.dumps({'verified':True,'manifest_files':len(manifest['files']),'byte_identical':compared}))
