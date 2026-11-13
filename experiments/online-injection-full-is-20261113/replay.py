"""Reproduce the native smoke audit from the archived source and raw records."""
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
a = p.parse_args()
base = Path(__file__).resolve().parent
for name, digest in json.loads((base/'MANIFEST.json').read_text())['files'].items():
    assert hashlib.sha256((base/name).read_bytes()).hexdigest() == digest, name
a.destination.mkdir(parents=True, exist_ok=False)
source = a.destination/'source'
with tarfile.open(base/'source-frozen.tgz') as archive:
    archive.extractall(source, filter='data')
for file in (base/'analysis').glob('*.py'):
    shutil.copyfile(file, source/'scripts'/file.name)
snapshot = a.destination/'snapshot'
with tarfile.open(base/'raw/smoke-a3.tar.gz') as archive:
    archive.extractall(snapshot, filter='data')
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join([str(source/'src'), str(source/'scripts'), env.get('PYTHONPATH', '')])
out = a.destination/'audit-smoke.json'
subprocess.run([sys.executable, str(source/'scripts/audit_online_is.py'),
    '--folder', str(snapshot), '--frozen', str(base/'inputs/FROZEN_IS.json'),
    '--output', str(out)], env=env, check=True)
assert out.read_bytes() == (base/'audit-smoke.json').read_bytes(), 'audit replay bytes differ'
print(json.dumps({'verified': True, 'manifest_files': len(json.loads((base/'MANIFEST.json').read_text())['files']),
                  'native_audit_byte_identical': True}))
