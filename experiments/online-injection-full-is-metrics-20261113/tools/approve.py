"""Freeze formal execution only after the complete ungraded TRAIN audit passes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

base = Path(__file__).resolve().parent
audit = json.loads((base/'audit-smoke.json').read_text())
assert audit['passed'] and audit['units'] == 12 and not audit['quality_scored']
raw = base/'snapshot/smoke/a3/smoke.jsonl'
rows = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()]
meta = next(r for r in rows if r['kind'] == 'metadata')
protocol = json.loads((base/'PROTOCOL.json').read_text())
assert meta['arms'] == protocol['arms'] and set(meta['cases']) == set(protocol['smoke_cases'])
payload = {'protocol': protocol, 'smoke_audit': audit,
    'smoke_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
    'generation_archive_sha256': hashlib.sha256((base/'source-frozen.tgz').read_bytes()).hexdigest(),
    'source': meta['source'], 'settings_sha256': meta['settings_sha256']}
remote = 'payload='+repr(payload)+'\n'+r'''
from pathlib import Path
import hashlib,json,subprocess,time
root=Path('/data/disk/changxiangyu/agentic_scaling');name='online-injection-full-is-metrics-20261113'
dep=root/'deployment/ascend'/name;out=root/('results/'+name+'-a3')
assert not (dep/'FORMAL-FROZEN.json').exists()
assert not (out/'formal.jsonl').exists()
assert (out/'smoke.exit').read_text().strip()=='0'
assert hashlib.sha256((out/'smoke.jsonl').read_bytes()).hexdigest()==payload['smoke_sha256']
assert hashlib.sha256((dep/'source-frozen.tgz').read_bytes()).hexdigest()==payload['generation_archive_sha256']
assert hashlib.sha256((dep/'settings.json').read_bytes()).hexdigest()==payload['settings_sha256']
for n,h in json.loads((dep/'SOURCE-FROZEN.json').read_text())['hashes'].items():
    assert hashlib.sha256((dep/n).read_bytes()).hexdigest()==h,n
for n,h in payload['source'].items():
    assert hashlib.sha256((dep/'source'/n).read_bytes()).hexdigest()==h,n
assert not json.loads(subprocess.check_output(['docker','inspect','agentic-scaling-changxiangyu-full-is-metrics-a3'],text=True))[0]['State']['Running']
payload['registered']=time.time()
(dep/'FORMAL-FROZEN.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'approved':True,'registered':payload['registered'],'archive_sha256':payload['generation_archive_sha256'],'units':192}))
'''
script = base/'approve-remote.py';script.write_text(remote, encoding='utf-8')
result = subprocess.run([sys.executable, '-X', 'utf8', str(base.parent/'online_injection_scaling_20261109/remote.py'),
    'a', '--file', str(script), '--timeout', '90'], capture_output=True)
assert result.returncode == 0, result.stderr.decode()
(base/'formal-approval.json').write_bytes(result.stdout)
print(result.stdout.decode())
