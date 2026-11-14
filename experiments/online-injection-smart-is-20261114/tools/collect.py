from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile

p=argparse.ArgumentParser()
p.add_argument('stage',choices=['smoke','formal','diagnostic'])
p.add_argument('--allow-failed',action='store_true')
p.add_argument('--via-jump',action='store_true')
a=p.parse_args()
base=Path(__file__).resolve().parent
remote_tool=base.parent/'online_injection_scaling_20261109/remote.py'
script='stage='+repr(a.stage)+'\nallow_failed='+repr(a.allow_failed)+'\n'+r'''
from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path('/data/disk/changxiangyu/agentic_scaling');name='online-injection-smart-is-20261114'
dep=root/'deployment/ascend'/name
result=[]
for worker in ['a0','a1','a3']:
    out=root/('results/'+name+'-'+worker)
    assert (out/(stage+'.exit')).exists(),str(out)
    assert allow_failed or (out/(stage+'.exit')).read_text().strip()=='0'
    info=json.loads(subprocess.check_output(['docker','inspect','agentic-scaling-changxiangyu-smart-is-'+worker],text=True))[0]
    assert not info['State']['Running']
    tokenizer=json.loads((root/'runtime/models/Qwen3.8-27B/tokenizer.json').read_text())
    vocabulary={v:k for k,v in tokenizer['model']['vocab'].items()}
    vocabulary.update({x['id']:x['content'] for x in tokenizer['added_tokens']})
    seen=set()
    def visit(x):
        if type(x) is int and x in vocabulary:seen.add(x)
        elif isinstance(x,list):
            for v in x:visit(v)
        elif isinstance(x,dict):
            for v in x.values():visit(v)
    if (out/(stage+'.jsonl')).exists():
        for line in (out/(stage+'.jsonl')).open():visit(json.loads(line))
    vpath=out/('vocabulary-'+stage+'.json')
    vpath.write_text(json.dumps({'decoder':tokenizer['decoder'],'special_ids':[x['id'] for x in tokenizer['added_tokens'] if x.get('special')],
        'vocab':{str(i):vocabulary[i] for i in sorted(seen)},'tokenizer_sha256':hashlib.sha256((root/'runtime/models/Qwen3.8-27B/tokenizer.json').read_bytes()).hexdigest()},ensure_ascii=False)+'\n')
    archive=dep/(stage+'-snapshot-'+worker+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for f in out.iterdir():
            if f.is_file() and (f.name.startswith(stage) or f.name in {'prepared-'+stage+'.json','vocabulary-'+stage+'.json'}):
                tar.add(f,arcname=f.name)
    result.append({'worker':worker,'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
print(json.dumps(result))
'''
r=subprocess.run([sys.executable,'-X','utf8',str(remote_tool),'a',*(['--via-jump'] if a.via_jump else []),'--timeout','180'],input=script.encode(),capture_output=True,timeout=200)
assert r.returncode==0,r.stderr.decode()
manifest=json.loads(r.stdout)
interface=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',
    'Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias WLAN | Select-Object IPAddress,InterfaceIndex | ConvertTo-Json -Compress'],text=True))
config=base/'ssh-transfer-config'
socket=base.parent/'online_injection_grounded_is_20261110/ssh_socket_transfer.py'
config.write_text(f'Host 159.138.54.61\n    ProxyCommand {Path(sys.executable).as_posix()} {socket.as_posix()} --interface-index {interface["InterfaceIndex"]} --source {interface["IPAddress"]} %h %p\nHost *\n    Include C:/Users/singm/.ssh/config\n')
jump=f'C:/Windows/System32/OpenSSH/ssh.exe -F {config.as_posix()} -o BindAddress={interface["IPAddress"]} -o BatchMode=yes -o IdentityAgent=none -o ConnectTimeout=20 -W %h:%p 159.138.54.61'
snapshot=base/'snapshot'/a.stage
snapshot.mkdir(exist_ok=True,parents=True)
def download(item):
    path=snapshot/(item['worker']+'.tar.gz')
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        r=subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-o','BindAddress='+interface['IPAddress'],
            '-o','ProxyCommand='+jump,'-o','BatchMode=yes','-o','IdentityAgent=none',
            '-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3',
            '27.106.124.198:'+item['path'],str(path)],capture_output=True,timeout=600)
        assert r.returncode==0,r.stderr.decode(errors='replace')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    dest=snapshot/item['worker'];dest.mkdir(exist_ok=True)
    with tarfile.open(path) as tar:tar.extractall(dest,filter='data')
    return {'worker':item['worker'],'bytes':item['bytes'],'verified':True}
with ThreadPoolExecutor(max_workers=2) as pool:
    for result in pool.map(download,manifest):print(json.dumps(result),flush=True)
(snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
