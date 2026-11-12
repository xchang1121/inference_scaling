"""Verify and replay complete native reports without a model or network."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile

p=argparse.ArgumentParser()
p.add_argument('--package',type=Path,required=True)
p.add_argument('--workdir',type=Path,required=True)
p.add_argument('--figures',action='store_true',help='also rebuild and byte-check the figures')
p.add_argument('--plot-python',type=Path,default=Path(sys.executable))
a=p.parse_args()
package=a.package.resolve();work=a.workdir.resolve()
assert not work.exists(),'choose a new replay directory'
manifest=json.loads((package/'MANIFEST.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==digest,name
for name,digest in json.loads((package/'inputs/compressed-input-hashes.json').read_text()).items():
    with gzip.open(package/'inputs'/(name+'.gz'),'rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
work.mkdir(parents=True)
with tarfile.open(package/'generation-source.tar.gz') as tar:tar.extractall(work/'source',filter='data')
for name,digest in json.loads((package/'inputs/FROZEN_IS.json').read_text())['files'].items():
    assert hashlib.sha256((work/'source'/name).read_bytes()).hexdigest()==digest,name
env=dict(os.environ)
env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTHONPATH']=os.pathsep.join([str(work/'source/src'),str(package/'analysis'),env.get('PYTHONPATH','')])
compared=[]
for stage in ['smoke','formal']:
    snapshot=work/'snapshot'/stage
    for archive in sorted((package/'raw'/stage).glob('*.tar.gz')):
        worker=archive.name.removesuffix('.tar.gz');dest=snapshot/worker;dest.mkdir(parents=True)
        with tarfile.open(archive) as tar:tar.extractall(dest,filter='data')
    output=work/'rebuilt'/stage
    subprocess.run([sys.executable,'-X','utf8',str(package/'analysis/summarize_literature_retrieval.py'),
        '--snapshot',str(snapshot),'--output',str(output),'--gold',str(package/'inputs/gold.jsonl'),
        '--stage',stage,'--frozen',str(package/'inputs/FROZEN_IS.json'),
        '--manifest',str(package/'inputs/manifest.json')],env=env,check=True)
    if stage=='formal':
        subprocess.run([sys.executable,'-X','utf8',str(package/'tools/interpret.py'),
            '--summary',str(output/'summary.json'),'--output',str(output/'RESULTS_ZH.md')],env=env,check=True)
    for f in sorted(output.iterdir()):
        assert f.read_bytes()==(package/'reports'/stage/f.name).read_bytes(),stage+'/'+f.name
        compared.append(stage+'/'+f.name)
subprocess.run([sys.executable,'-X','utf8',str(package/'analysis/diagnose_literature_empty_answers.py'),
    '--snapshot',str(work/'snapshot/formal'),'--summary',str(work/'rebuilt/formal/summary.json'),
    '--answers',str(work/'rebuilt/formal/answers.jsonl'),'--output',str(work/'diagnostics')],check=True)
diagnostic_comparisons=[]
for f in sorted((work/'diagnostics').iterdir()):
    assert f.read_bytes()==(package/'diagnostics'/f.name).read_bytes(),'diagnostics/'+f.name
    diagnostic_comparisons.append('diagnostics/'+f.name)
figure_comparisons=[]
if a.figures:
    plot_env=dict(os.environ);plot_env.pop('PYTHONPATH',None)
    subprocess.run([str(a.plot_python),str(package/'analysis/plot_literature_retrieval.py'),
        '--summary',str(work/'rebuilt/formal/summary.json'),
        '--answers',str(work/'rebuilt/formal/answers.jsonl'),
        '--output',str(work/'visuals')],env=plot_env,check=True)
    for f in sorted((work/'visuals').iterdir()):
        assert f.read_bytes()==(package/'visuals'/f.name).read_bytes(),'visuals/'+f.name
        figure_comparisons.append('visuals/'+f.name)
receipt={'passed':True,'verified_package_files':len(manifest['files']),'byte_identical_reports':compared,
    'byte_identical_figures':figure_comparisons,
    'byte_identical_diagnostics':diagnostic_comparisons,
    'generation_sha256':hashlib.sha256((package/'generation-source.tar.gz').read_bytes()).hexdigest(),
    'raw_archives':{p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((package/'raw').rglob('*.tar.gz'))}}
(work/'OFFLINE-VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
