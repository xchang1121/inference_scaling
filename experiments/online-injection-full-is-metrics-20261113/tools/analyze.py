from pathlib import Path
import argparse
import os
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('stage', choices=['smoke', 'formal'])
a = p.parse_args()
base = Path(__file__).resolve().parent
repo = base.parent/'online_injection_20261109/source'
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join([str(repo/'src'), str(repo/'scripts'), str(base.parent/'online_injection_20261109/devdeps')])
if a.stage == 'smoke':
    cmd = [sys.executable, str(repo/'scripts/audit_online_is.py'), '--folder', str(base/'snapshot/smoke/a3'),
        '--frozen', str(base/'FROZEN_IS.json'), '--output', str(base/'audit-smoke.json')]
else:
    cmd = [sys.executable, str(repo/'scripts/summarize_online_is.py'), '--snapshot', str(base/'snapshot/formal'),
        '--output', str(base/'reports-formal'), '--gold', str(base/'data/gold.jsonl'),
        '--frozen', str(base/'FROZEN_IS.json'), '--protocol', str(base/'PROTOCOL.json')]
subprocess.run(cmd, env=env, check=True)
