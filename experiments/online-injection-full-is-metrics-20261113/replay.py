"""Verify package hashes and byte-reproduce native audits and metric reports."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--destination', type=Path, required=True)
p.add_argument('--plot-python', type=Path)
a = p.parse_args()
base = Path(__file__).resolve().parent
manifest = json.loads((base/'MANIFEST.json').read_text())
for n, h in manifest['files'].items():
    assert hashlib.sha256((base/n).read_bytes()).hexdigest() == h, n
a.destination.mkdir(parents=True, exist_ok=False)
source = a.destination/'source'
with tarfile.open(base/'source-frozen.tgz') as archive:
    archive.extractall(source, filter='data')
for file in (base/'analysis').glob('*.py'):
    shutil.copyfile(file, source/'scripts'/file.name)
for stage in ('smoke', 'formal'):
    for file in (base/'raw'/stage).glob('*.tar.gz'):
        with tarfile.open(file) as archive:
            archive.extractall(a.destination/'snapshot'/stage/file.name.removesuffix('.tar.gz'), filter='data')
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join([str(source/'src'), str(source/'scripts'), env.get('PYTHONPATH', '')])
smoke = a.destination/'audit-smoke.json'
subprocess.run([sys.executable, str(source/'scripts/audit_online_is.py'),
    '--folder', str(a.destination/'snapshot/smoke/a3'), '--frozen', str(base/'inputs/FROZEN_IS.json'),
    '--output', str(smoke)], env=env, check=True)
assert smoke.read_bytes() == (base/'audit-smoke.json').read_bytes()
report = a.destination/'reports'
subprocess.run([sys.executable, str(source/'scripts/summarize_online_is.py'),
    '--snapshot', str(a.destination/'snapshot/formal'), '--output', str(report),
    '--gold', str(base/'inputs/gold.jsonl'), '--frozen', str(base/'inputs/FROZEN_IS.json'),
    '--protocol', str(base/'inputs/PROTOCOL.json')], env=env, check=True)
compared = []
for file in (base/'reports').iterdir():
    assert (report/file.name).read_bytes() == file.read_bytes(), file.name
    compared.append(file.name)
plots = []
if a.plot_python:
    plot_env = dict(os.environ);plot_env.pop('PYTHONPATH', None)
    subprocess.run([str(a.plot_python), str(source/'scripts/plot_online_is.py'),
        '--summary', str(report/'summary.json'), '--output', str(a.destination/'figures')], env=plot_env, check=True)
    for file in (base/'figures').iterdir():
        assert (a.destination/'figures'/file.name).read_bytes() == file.read_bytes(), file.name
        plots.append(file.name)
print(json.dumps({'verified': True, 'manifest_files': len(manifest['files']),
                  'byte_identical': ['audit-smoke.json']+compared+plots}))
