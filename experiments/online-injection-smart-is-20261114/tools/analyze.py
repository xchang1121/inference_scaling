from pathlib import Path
import argparse
import json
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
    audits = []
    for worker in ('a0', 'a1', 'a3'):
        destination = base/('audit-smoke-'+worker+'.json')
        subprocess.run([sys.executable, str(repo/'scripts/audit_smart_online_is.py'), '--folder', str(base/'snapshot/smoke'/worker),
            '--frozen', str(base/'FROZEN_IS.json'), '--output', str(destination)], env=env, check=True)
        audits.append(json.loads(destination.read_text()))
    combined = {'passed': all(x['passed'] for x in audits), 'quality_scored': False,
        'units': sum(x['units'] for x in audits), 'checks': sum(x['checks'] for x in audits), 'workers': audits}
    (base/'audit-smoke.json').write_text(json.dumps(combined, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(combined)); raise SystemExit(0)
else:
    cmd = [sys.executable, str(repo/'scripts/summarize_smart_online_is.py'), '--snapshot', str(base/'snapshot/formal'),
        '--output', str(base/'reports-formal'), '--gold', str(base/'data/gold.jsonl'),
        '--frozen', str(base/'FROZEN_IS.json'), '--protocol', str(base/'PROTOCOL.json')]
subprocess.run(cmd, env=env, check=True)
