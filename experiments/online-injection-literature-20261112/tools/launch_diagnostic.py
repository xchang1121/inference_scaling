from pathlib import Path
import base64
import subprocess
import sys

base=Path(__file__).resolve().parent
source=base64.b64encode((base/'native_diagnostic.py').read_bytes()).decode()
license_text=(base/'ThinkRetrieve/thinkretrieve/LICENSE').read_text()
job=r'''
from pathlib import Path
import fcntl,hashlib,json,subprocess,time
root=Path('/data/disk/changxiangyu/agentic_scaling');name='online-injection-literature-20261112'
dep=root/'deployment/ascend'/name;out=root/('results/'+name+'-a3')
c='agentic-scaling-changxiangyu-literature-a3';status=1
with (out/'job.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  assert (out/'smoke.exit').read_text().strip()=='0'
  assert not (out/'diagnostic.jsonl').exists() and not (out/'formal.jsonl').exists()
  frozen=json.loads((dep/'SOURCE-FROZEN.json').read_text())
  for n,h in frozen['hashes'].items():assert hashlib.sha256((dep/n).read_bytes()).hexdigest()==h,n
  assert 'No running processes found in NPU 3' in subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)
  for n in subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).split():
   x=json.loads(subprocess.check_output(['docker','inspect',n],text=True))[0]
   assert not any(d['PathOnHost']=='/dev/davinci3' for d in x['HostConfig']['Devices'] or []),n
  (out/'diagnostic-start.json').write_text(json.dumps({'started':time.time(),'source':frozen})+'\n')
  subprocess.run(['docker','start',c],check=True,capture_output=True)
  cmd=['docker','exec',c,'bash','/deployment/'+name+'/env.sh','python3','/deployment/'+name+'/diagnostics/native_diagnostic.py']
  with (out/'diagnostic.log').open('wb') as log:status=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 finally:
  subprocess.run(['docker','stop',c],capture_output=True,timeout=60)
  info=json.loads(subprocess.check_output(['docker','inspect',c],text=True))[0]
  (out/'diagnostic-finish.json').write_text(json.dumps({'finished':time.time(),'exit':status,'state':info['State']})+'\n')
  (out/'diagnostic.exit').write_text(str(status)+'\n')
'''
remote='source='+repr(source)+'\nlicense_text='+repr(license_text)+'\njob='+repr(job)+'\n'+r'''
from pathlib import Path
import base64,json,subprocess
root=Path('/data/disk/changxiangyu/agentic_scaling');name='online-injection-literature-20261112'
dep=root/'deployment/ascend'/name;out=root/('results/'+name+'-a3')
d=dep/'diagnostics';d.mkdir(exist_ok=True)
p=out/'diagnostic-launch.py'
assert not p.exists(),'preserve prior diagnostic'
(d/'native_diagnostic.py').write_bytes(base64.b64decode(source));(d/'THINKRETRIEVE-LICENSE').write_text(license_text)
p.write_text(job)
with (out/'diagnostic-launch.log').open('wb') as log:
 proc=subprocess.Popen(['python3',str(p)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'diagnostic.pid').write_text(str(proc.pid)+'\n')
print(json.dumps({'pid':proc.pid,'stage':'diagnostic'}))
'''
p=base/'diagnostic-launch-remote.py';p.write_text(remote,encoding='utf-8')
r=subprocess.run([sys.executable,'-X','utf8',str(base.parent/'online_injection_scaling_20261109/remote.py'),
    'a','--via-jump','--file',str(p),'--timeout','90'],capture_output=True)
(base/'diagnostic-launch.log').write_bytes(r.stdout+r.stderr)
print(r.stdout.decode(),r.stderr.decode())
raise SystemExit(r.returncode)
