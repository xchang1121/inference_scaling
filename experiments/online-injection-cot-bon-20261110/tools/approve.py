from pathlib import Path
import hashlib
import json
import subprocess
import sys

base=Path(__file__).resolve().parent
summary=json.loads((base/'reports-smoke/summary.json').read_text())
assert summary['audit']['passed'] and summary['questions']==2 and summary['selected_answers']==64
assert summary['accepted_packets']==2 and summary['audit']['raw']['candidate_truncated']==0
expected={}
for worker in ['a0','a5']:
    rows=[json.loads(s) for s in (base/'snapshot/smoke'/worker/'smoke.jsonl').read_text(encoding='utf-8').splitlines()]
    meta=next(r for r in rows if r['kind']=='metadata')
    assert meta['schema']==3
    assert meta['config']['candidates']==8 and meta['config']['answer_tokens']==16384
    expected[worker]={'source':meta['source'],'settings_sha256':meta['settings_sha256'],
        'smoke_sha256':hashlib.sha256((base/'snapshot/smoke'/worker/'smoke.jsonl').read_bytes()).hexdigest()}
payload={'smoke_audit':summary['audit'],'expected':expected,'generation_archive_sha256':hashlib.sha256((base/'source-frozen.tgz').read_bytes()).hexdigest(),
    'manifest':json.loads((base/'data/manifest.json').read_text())}
script='payload='+repr(payload)+'\n'+r'''
from pathlib import Path
import hashlib,json,subprocess,time
root=Path('/data/disk/changxiangyu/agentic_scaling');name='online-injection-cot-bon-20261110'
dep=root/'deployment/ascend'/name
assert not (dep/'FORMAL-FROZEN.json').exists()
assert hashlib.sha256((dep/'source-frozen.tgz').read_bytes()).hexdigest()==payload['generation_archive_sha256']
frozen=json.loads((dep/'SOURCE-FROZEN.json').read_text())
for n,h in frozen['hashes'].items():assert hashlib.sha256((dep/n).read_bytes()).hexdigest()==h,n
for worker,expect in payload['expected'].items():
    out=root/('results/'+name+'-'+worker)
    assert (out/'smoke.exit').read_text().strip()=='0'
    assert not (out/'formal.jsonl').exists()
    assert hashlib.sha256((out/'smoke.jsonl').read_bytes()).hexdigest()==expect['smoke_sha256']
    for n,h in expect['source'].items():assert hashlib.sha256((dep/'source'/n).read_bytes()).hexdigest()==h,n
    assert hashlib.sha256((dep/'settings.json').read_bytes()).hexdigest()==expect['settings_sha256']
    assert not json.loads(subprocess.check_output(['docker','inspect','agentic-scaling-changxiangyu-cot-bon-'+worker],text=True))[0]['State']['Running']
payload['registered']=time.time()
(dep/'FORMAL-FROZEN.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'approved':True,'registered':payload['registered'],'generation_archive_sha256':payload['generation_archive_sha256']}))
'''
p=base/'approve-remote.py';p.write_text(script,encoding='utf-8')
r=subprocess.run([sys.executable,'-X','utf8',str(base.parent/'online_injection_scaling_20261109/remote.py'),'a','--file',str(p),'--timeout','90'],capture_output=True)
assert r.returncode==0,r.stderr.decode()
(base/'formal-approval.json').write_bytes(r.stdout)
print(r.stdout.decode())
