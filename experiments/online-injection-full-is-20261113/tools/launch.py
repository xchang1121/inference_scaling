from pathlib import Path
import argparse
import json
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('stage',choices=['smoke','formal'])
p.add_argument('--workers',nargs='+',default=['a3'])
p.add_argument('--via-jump',action='store_true')
a=p.parse_args()
base=Path(__file__).resolve().parent
manifest=json.loads((base/'data/manifest.json').read_text())
job=r'''
from pathlib import Path
import fcntl,hashlib,json,subprocess,time
root=Path('/data/disk/changxiangyu/agentic_scaling');name='online-injection-full-is-20261113'
dep=root/'deployment/ascend'/name;out=root/('results/'+name+'-'+worker)
c='agentic-scaling-changxiangyu-full-is-'+worker
status=1
with (out/'job.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        frozen=json.loads((dep/'SOURCE-FROZEN.json').read_text())
        for n,h in frozen['hashes'].items():assert hashlib.sha256((dep/n).read_bytes()).hexdigest()==h,n
        assert not (out/(stage+'.jsonl')).exists()
        baseline=json.loads((dep/'FROZEN_IS.json').read_text())
        for n,h in baseline['files'].items():assert hashlib.sha256((dep/'source'/n).read_bytes()).hexdigest()==h,n
        if stage=='formal':assert (dep/'FORMAL-FROZEN.json').exists(),'smoke approval required'
        inventory=subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)
        (out/(stage+'-npu-before.txt')).write_text(inventory)
        assert f'No running processes found in NPU {worker[1:]}' in inventory
        for n in subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).split():
            x=json.loads(subprocess.check_output(['docker','inspect',n],text=True))[0]
            assert not any(d['PathOnHost']=='/dev/davinci'+worker[1:] for d in x['HostConfig']['Devices'] or []),n
        (out/(stage+'-start.json')).write_text(json.dumps({'worker':worker,'stage':stage,'cases':cases,'seeds':seeds,'started':time.time(),'source':frozen})+'\n')
        subprocess.run(['docker','start',c],check=True,capture_output=True)
        cmd=['docker','exec',c,'bash','/deployment/'+name+'/env.sh','python3','/deployment/'+name+'/source/scripts/benchmark_online_is.py',
            '--settings','/deployment/'+name+'/settings.json','--prepared','/results/prepared-'+stage+'.json',
            '--corpus','/deployment/'+name+'/data/corpus.jsonl','--output','/results/'+stage+'.jsonl',
            '--case-ids',*cases,'--seeds',*map(str,seeds),'--worker',worker]
        with (out/(stage+'.log')).open('wb') as log:
            status=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
    finally:
        subprocess.run(['docker','stop',c],capture_output=True,timeout=60)
        info=json.loads(subprocess.check_output(['docker','inspect',c],text=True))[0]
        (out/(stage+'-finish.json')).write_text(json.dumps({'exit':status,'finished':time.time(),'state':info['State']})+'\n')
        (out/(stage+'.exit')).write_text(str(status)+'\n')
'''
for i,worker in enumerate(a.workers):
    cases=(manifest['smoke_ids'] if a.stage=='smoke' else manifest['fresh_ids'][i::len(a.workers)])
    seeds=[20261112] if a.stage=='smoke' else manifest['seeds']
    body='worker='+repr(worker)+'\nstage='+repr(a.stage)+'\ncases='+repr(cases)+'\nseeds='+repr(seeds)+'\n'+job
    remote='job='+repr(body)+'\nworker='+repr(worker)+'\nstage='+repr(a.stage)+'\n'+r'''
from pathlib import Path
import json,subprocess
root=Path('/data/disk/changxiangyu/agentic_scaling');name='online-injection-full-is-20261113'
dep=root/'deployment/ascend'/name;out=root/('results/'+name+'-'+worker)
p=out/(stage+'-launch.py')
if p.exists():
    assert p.read_text()==job,'different launch specification already exists'
    pid=int((out/(stage+'.pid')).read_text())
    assert (out/(stage+'.exit')).exists() or str(p).encode() in Path('/proc/'+str(pid)+'/cmdline').read_bytes()
    print(json.dumps({'worker':worker,'stage':stage,'pid':pid,'already_launched':True}));raise SystemExit(0)
p.write_text(job)
with (out/(stage+'-launch.log')).open('wb') as log:
    proc=subprocess.Popen(['python3',str(p)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/(stage+'.pid')).write_text(str(proc.pid)+'\n')
print(json.dumps({'worker':worker,'stage':stage,'pid':proc.pid}))
'''
    script=base/(a.stage+'-launch-'+worker+'.py');script.write_text(remote,encoding='utf-8')
    r=subprocess.run([sys.executable,'-X','utf8',str(base.parent/'online_injection_scaling_20261109/remote.py'),worker[0],*(['--via-jump'] if a.via_jump else []),'--file',str(script),'--timeout','75'],capture_output=True)
    (base/(a.stage+'-launch-'+worker+'.log')).write_bytes(r.stdout+r.stderr)
    print(r.stdout.decode(),r.stderr.decode(),flush=True)
    assert r.returncode==0
