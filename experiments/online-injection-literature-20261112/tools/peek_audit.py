"""Read-only structural audit of completed records; never grades partial output."""
from pathlib import Path
import argparse
import base64
import gzip
import json
import subprocess
import sys

base=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--stage',choices=['smoke','formal'],default='smoke')
a=p.parse_args()
remote=r'''
from pathlib import Path
import base64,gzip,json
root=Path('/data/disk/changxiangyu/agentic_scaling')
out=root/'results/online-injection-literature-20261112-a3'
rows=[]
for line in (out/(stage+'.jsonl')).open():
 try:rows.append(json.loads(line))
 except ValueError:break
tokenizer=json.loads((root/'runtime/models/Qwen3.8-27B/tokenizer.json').read_text())
vocab={v:k for k,v in tokenizer['model']['vocab'].items()}
vocab.update({x['id']:x['content'] for x in tokenizer['added_tokens']})
seen=set()
def visit(x):
 if type(x) is int and x in vocab:seen.add(x)
 elif isinstance(x,list):
  for v in x:visit(v)
 elif isinstance(x,dict):
  for v in x.values():visit(v)
visit(rows)
v={'decoder':tokenizer['decoder'],'special_ids':[x['id'] for x in tokenizer['added_tokens'] if x.get('special')],
 'vocab':{str(i):vocab[i] for i in seen}}
payload={'rows':rows,'vocabulary':v,'prepared':json.loads((out/('prepared-'+stage+'.json')).read_text())}
print(base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode())
'''
remote='stage='+repr(a.stage)+'\n'+remote
r=subprocess.run([sys.executable,'-X','utf8',str(base.parent/'online_injection_scaling_20261109/remote.py'),
    'a','--via-jump','--timeout','90'],input=remote.encode(),capture_output=True,timeout=110)
assert r.returncode==0,r.stderr.decode(errors='replace')
compressed=base64.b64decode(r.stdout)
(base/(a.stage+'-partial-audit-input.json.gz')).write_bytes(compressed)
data=json.loads(gzip.decompress(compressed))
repo=base.parent/'online_injection_20261109/source'
sys.path[:0]=list(map(str,[repo/'src',repo/'scripts',repo.parent/'devdeps']))
from summarize_cot_guidance import Decoder
from summarize_literature_retrieval import audit_row
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.literature import unrelated
decode=Decoder(data['vocabulary'])
checks=0;failed=[];units=[]
def check(ok,name):
    global checks
    checks+=1
    if not ok:failed.append(name)
for row in data['rows']:
    if row['kind']!='measurement':continue
    try:
        audit_row(row,GuidanceConfig(),decode,data['prepared']['noise_ids'],check)
        units.append(row['unit_id'])
    except Exception as e:failed.append(row['unit_id']+':'+repr(e))
measures={(r['question_id'],r['seed'],r['arm']):r for r in data['rows'] if r['kind']=='measurement'}
for (q,s,arm),row in measures.items():
    if arm!='thinkretrieve_noise' or (q,s,'thinkretrieve') not in measures:continue
    reference=measures[q,s,'thinkretrieve']['handoff']['packets'];h=row['handoff']
    check(len(h['injected_ids'])==len(reference) or h['terminal_eos'],f'{q}:{s}:noise_count_or_EOS')
    for i,ids in enumerate(h['injected_ids']):
        check(i<len(reference) and ids==list(unrelated(reference[i],data['prepared']['noise_ids'])),
              f'{q}:{s}:exact_noise_payload_{i}')
report={'passed':not failed,'checks':checks,'units':units,'failures':failed,'graded':False}
(base/(a.stage+'-partial-audit.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(bool(failed))
