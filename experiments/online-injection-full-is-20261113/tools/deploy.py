from pathlib import Path
import base64
import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile

base=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--replica',choices=['a'],default='a')
p.add_argument('--refresh-source',action='store_true')
a=p.parse_args()
repo=base.parent/'online_injection_20261109/source'
name='online-injection-full-is-20261113'
paths=['src','settings','pyproject.toml','scripts/benchmark_online_is.py','scripts/benchmark_flare_gap.py',
       'scripts/benchmark_is_gpu.py','tests/test_online_is.py','docs/reports/ONLINE_IS_PROTOCOL.md','docs/third_party/THINKRETRIEVE-LICENSE.txt']
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
    for rel in paths:
        p=repo/rel
        if p.is_file(): tar.add(p,arcname=rel)
        else:
            for f in p.rglob('*'):
                if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':
                    tar.add(f,arcname=f.relative_to(repo).as_posix())
source=buf.getvalue()
if (base/'source-frozen.tgz').exists() and not a.refresh_source:
    source=(base/'source-frozen.tgz').read_bytes()
else:
    (base/'source-frozen.tgz').write_bytes(source)
files={'source-frozen.tgz':source,'settings.json':(base/'settings.json').read_bytes(),
       'FROZEN_IS.json':(base/'FROZEN_IS.json').read_bytes(),
       'unrelated-prose.txt':(base.parent/'online_injection_scaling_20261109/unrelated-prose.txt').read_bytes(),
       **{'data/'+n:(base/'data'/n).read_bytes() for n in ('questions.jsonl','smoke-questions.jsonl','examples.jsonl')}}
payload={n:base64.b64encode(b).decode() for n,b in files.items()}
env='''#!/usr/bin/env bash
set -e
export USER=changxiangyu LOGNAME=changxiangyu
export HF_HOME=/cache/huggingface XDG_CACHE_HOME=/results/cache
export TORCHINDUCTOR_CACHE_DIR=/results/cache/torchinductor TRITON_CACHE_DIR=/results/cache/triton
export ASCEND_PROCESS_LOG_PATH=/results/cache/ascend-log
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export PYTHONPATH="/deployment/online-injection-full-is-20261113/source/src:/deployment/online-injection-full-is-20261113/source/scripts:${PYTHONPATH:-}"
exec "$@"
'''
remote=r'''
from pathlib import Path
import base64,hashlib,json,os,subprocess,tarfile,time,shutil
root=Path('/data/disk/changxiangyu/agentic_scaling')
name='online-injection-full-is-20261113'
dep=root/'deployment/ascend'/name
dep.mkdir(exist_ok=True)
assert not (dep/'FORMAL-FROZEN.json').exists()
for n,s in payload.items():
    p=dep/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(base64.b64decode(s))
with tarfile.open(dep/'source-frozen.tgz') as t:t.extractall(dep/'source',filter='data')
(dep/'env.sh').write_text(env)
shutil.copyfile(root/'deployment/ascend/online-injection-grounded-is-20261110/data/corpus.jsonl',dep/'data/corpus.jsonl')
old=json.loads(subprocess.check_output(['docker','inspect','agentic-scaling-changxiangyu-grounded-is-'+('a0' if replica=='a' else 'b4')],text=True))[0]
devices=[3]
allnames=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).split()
busy=set()
for n in allnames:
    info=json.loads(subprocess.check_output(['docker','inspect',n],text=True))[0]
    for dev in info['HostConfig']['Devices'] or []:
        if dev['PathOnHost'] in ['/dev/davinci'+str(d) for d in devices] and not n.startswith('agentic-scaling-changxiangyu-full-is-'):
            busy.add(dev['PathOnHost'])
assert not busy,sorted(busy)
for device in devices:
    worker=replica+str(device)
    c='agentic-scaling-changxiangyu-full-is-'+worker
    out=root/('results/'+name+'-'+worker)
    out.mkdir(exist_ok=True);(out/'cache').mkdir(exist_ok=True)
    exists=subprocess.run(['docker','inspect',c],capture_output=True).returncode==0
    if not exists:
        assert f'No running processes found in NPU {device}' in subprocess.check_output(['/usr/local/bin/npu-smi','info'],text=True)
        cmd=['docker','create','--name',c,'--user',old['Config']['User'],'--group-add','1001','--shm-size','34359738368',
            '--label','inference_scaling.owner=changxiangyu','--label','inference_scaling.task='+name,
            '-e','OMP_NUM_THREADS=4','-e','VLLM_WORKER_MULTIPROC_METHOD=spawn']
        for dp in [f'/dev/davinci{device}','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:cmd+=['--device',dp+':'+dp]
        for m in old['Mounts']:
            src=str(out) if m['Destination']=='/results' else m['Source']
            cmd+=['--mount','type=bind,source='+src+',target='+m['Destination']+('' if m['RW'] else ',readonly')]
        cmd += [old['Image'],'sleep','infinity']
        subprocess.run(cmd,check=True,capture_output=True)
    else:
        assert not json.loads(subprocess.check_output(['docker','inspect',c],text=True))[0]['State']['Running']
    subprocess.run(['docker','start',c],check=True,capture_output=True)
    uid,gid=map(int,old['Config']['User'].split(':'))
    code='from pathlib import Path;import os\nfor n in ["/ascend","/.config"]:\n p=Path(n);p.mkdir(exist_ok=True);os.chown(p,'+str(uid)+','+str(gid)+')'
    subprocess.run(['docker','exec','--user','0:0',c,'python3','-c',code],check=True,capture_output=True)
    for stage,question in [('smoke','smoke-questions.jsonl'),('formal','questions.jsonl')]:
        if (out/('prepared-'+stage+'.json')).exists():continue
        cmd=['docker','exec',c,'bash','/deployment/'+name+'/env.sh','python3','/deployment/'+name+'/source/scripts/benchmark_online_is.py',
            '--settings','/deployment/'+name+'/settings.json','--corpus','/deployment/'+name+'/data/corpus.jsonl',
            '--questions','/deployment/'+name+'/data/'+question,'--noise','/deployment/'+name+'/unrelated-prose.txt',
            '--prepared','/results/prepared-'+stage+'.json','--prepare-only']
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
        (out/('prepare-'+stage+'.log')).write_text(r.stdout+r.stderr)
        assert r.returncode==0,(r.stdout+r.stderr)[-2000:]
    subprocess.run(['docker','stop',c],check=True,capture_output=True,timeout=45)
    print(json.dumps({'worker':worker,'prepared':True,'image':old['Image']}),flush=True)
# No dense encoder is needed for this BM25-only integration.
frozen={'created':time.time(),'hashes':{p.relative_to(dep).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dep.rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.relative_to(dep).parts[0] in {'source','data','encoder'} or p.relative_to(dep).as_posix() in {'source-frozen.tgz','settings.json','unrelated-prose.txt','env.sh','FROZEN_IS.json'})}}
(dep/'SOURCE-FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
print(json.dumps({'source_sha256':hashlib.sha256((dep/'source-frozen.tgz').read_bytes()).hexdigest(),'files':len(frozen['hashes'])}))
'''
script='payload='+repr(payload)+'\nenv='+repr(env)+'\nreplica='+repr(a.replica)+'\n'+remote
p=base/('deploy-'+a.replica+'-remote.py');p.write_text(script,encoding='utf-8')
r=subprocess.run([sys.executable,'-X','utf8',str(base.parent/'online_injection_scaling_20261109/remote.py'),a.replica,'--file',str(p),'--timeout','600'],capture_output=True,timeout=620)
(base/('deploy-'+a.replica+'.log')).write_bytes(r.stdout+r.stderr)
print(r.stdout.decode());print(r.stderr.decode());raise SystemExit(r.returncode)
