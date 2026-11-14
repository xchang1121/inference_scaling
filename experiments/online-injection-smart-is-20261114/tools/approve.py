"""Freeze the formal protocol after all ungraded native TRAIN audits pass."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

base=Path(__file__).resolve().parent
audit=json.loads((base/'audit-smoke.json').read_text())
protocol=json.loads((base/'PROTOCOL.json').read_text())
assert audit['passed'] and audit['units']==48 and not audit['quality_scored']
hashes={}; source=None; settings=None
for worker in protocol['workers']:
    raw=base/'snapshot/smoke'/worker/'smoke.jsonl'
    rows=[json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()]
    meta=next(r for r in rows if r['kind']=='metadata')
    assert meta['arms']==protocol['arms'] and set(meta['cases'])==set(protocol['smoke_cases'])
    assert source is None or source==meta['source']
    assert settings is None or settings==meta['settings_sha256']
    source,settings=meta['source'],meta['settings_sha256']
    hashes[worker]=hashlib.sha256(raw.read_bytes()).hexdigest()
original=json.loads((base/'ORIGINAL_ONLINE.json').read_text())
assert source[original['path']]==original['sha256']
payload={'protocol':protocol,'smoke_audit':audit,'smoke_hashes':hashes,
         'generation_archive_sha256':hashlib.sha256((base/'source-frozen.tgz').read_bytes()).hexdigest(),
         'source':source,'settings_sha256':settings,'original_online':original}
remote='payload='+repr(payload)+'\n'+r'''
from pathlib import Path
import hashlib,json,subprocess,time
root=Path('/data/disk/changxiangyu/agentic_scaling');name='online-injection-smart-is-20261114'
dep=root/'deployment/ascend'/name
assert not (dep/'FORMAL-FROZEN.json').exists()
assert hashlib.sha256((dep/'source-frozen.tgz').read_bytes()).hexdigest()==payload['generation_archive_sha256']
assert hashlib.sha256((dep/'settings.json').read_bytes()).hexdigest()==payload['settings_sha256']
for n,h in json.loads((dep/'SOURCE-FROZEN.json').read_text())['hashes'].items():
    assert hashlib.sha256((dep/n).read_bytes()).hexdigest()==h,n
for n,h in payload['source'].items():
    assert hashlib.sha256((dep/'source'/n).read_bytes()).hexdigest()==h,n
for worker,h in payload['smoke_hashes'].items():
    out=root/('results/'+name+'-'+worker)
    assert not (out/'formal.jsonl').exists()
    assert (out/'smoke.exit').read_text().strip()=='0'
    assert hashlib.sha256((out/'smoke.jsonl').read_bytes()).hexdigest()==h
    assert not json.loads(subprocess.check_output(['docker','inspect','agentic-scaling-changxiangyu-smart-is-'+worker],text=True))[0]['State']['Running']
payload['registered']=time.time()
(dep/'FORMAL-FROZEN.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'approved':True,'registered':payload['registered'],'archive_sha256':payload['generation_archive_sha256'],'units':256}))
'''
script=base/'approve-remote.py';script.write_text(remote,encoding='utf-8')
r=subprocess.run([sys.executable,'-X','utf8',str(base.parent/'online_injection_scaling_20261109/remote.py'),
                  'a','--file',str(script),'--timeout','90'],capture_output=True)
assert r.returncode==0,r.stderr.decode()
(base/'formal-approval.json').write_bytes(r.stdout);print(r.stdout.decode())
