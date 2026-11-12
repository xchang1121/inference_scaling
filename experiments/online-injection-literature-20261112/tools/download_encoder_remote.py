from pathlib import Path
import subprocess,json
base=Path('/data/disk/changxiangyu/agentic_scaling/deployment/ascend/online-injection-literature-20261112')
base.mkdir(exist_ok=True)
script=r'''from pathlib import Path
import urllib.request,hashlib,json,time,os
base=Path(__file__).resolve().parent
out=base/'encoder';out.mkdir(exist_ok=True)
revision='f52bf8ec8c7124536f0efb74aca902b2995e5bcd'
names=['config.json','model.safetensors','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt']
for name in names:
 p=out/name
 if not p.exists():
  tmp=p.with_suffix(p.suffix+'.partial')
  urllib.request.urlretrieve('https://huggingface.co/intfloat/e5-base-v2/resolve/'+revision+'/'+name,tmp)
  os.replace(tmp,p)
 print(json.dumps({'file':name,'size':p.stat().st_size}),flush=True)
(out/'MANIFEST.json').write_text(json.dumps({'model':'intfloat/e5-base-v2','revision':revision,'files':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in names}},indent=2))
print('COMPLETE',flush=True)
'''
p=base/'download_encoder.py'
if not p.exists():
 p.write_text(script)
 with (base/'download-encoder.log').open('wb') as f:
  proc=subprocess.Popen(['python3',str(p)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (base/'download-encoder.pid').write_text(str(proc.pid))
print((base/'download-encoder.pid').read_text())
