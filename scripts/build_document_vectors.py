"""Frozen E5 document vectors for supervised hidden-state alignment."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('corpus','encoder','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=('cpu','npu'),default='cpu')
    a=p.parse_args();assert a.limit>=0 and 1<=a.threads<=4
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    encoder=json.loads((a.encoder/'MANIFEST.json').read_text())
    assert encoder['revision']=='f52bf8ec8c7124536f0efb74aca902b2995e5bcd'
    assert all(digest(a.encoder/name)==value for name,value in encoder['files'].items())
    docs=sorted(map(json.loads,a.corpus.read_text().splitlines()),key=lambda d:d['id'])
    assert all(set(d)<= {'id','title','text','source'} for d in docs)
    assert len({d['id'] for d in docs})==len(docs)
    if a.limit:docs=docs[:a.limit]
    spec={'encoder':encoder,'corpus_sha256':digest(a.corpus),'documents':len(docs),'limit':a.limit,
        'template':'passage: {title}\\n{text}','max_encoder_tokens':512,'pooling':'masked token mean then L2',
        'device':a.device,'threads':a.threads,'batch_size':8,'dtype':'float32','chunk_documents':64,
        'attention':'eager'}
    a.output.mkdir(parents=True,exist_ok=True);spec_path=a.output/'spec.json'
    if spec_path.exists():assert json.loads(spec_path.read_text())==spec
    else:spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    completed=a.output/'complete.json'
    if completed.exists():
        manifest=json.loads(completed.read_text())
        assert all(digest(a.output/name)==value for name,value in manifest['files'].items())
        print(json.dumps({'already_complete':True,'documents':len(docs)}));return
    started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(a.encoder,local_files_only=True)
    model=AutoModel.from_pretrained(a.encoder,local_files_only=True,attn_implementation='eager').to(a.device).eval()
    assert all(v.device.type==a.device for v in model.parameters())
    chunks=[]
    for begin in range(0,len(docs),64):
        group=docs[begin:begin+64];path=a.output/f'chunk-{begin:06d}.npy';proof=path.with_suffix('.json')
        if path.exists() and proof.exists():
            record=json.loads(proof.read_text());assert record['ids']==[d['id'] for d in group] and record['sha256']==digest(path)
        else:
            vectors=[];truncated=0
            with torch.inference_mode():
                for offset in range(0,len(group),8):
                    texts=['passage: '+d['title']+'\n'+d['text'] for d in group[offset:offset+8]]
                    truncated+=sum(len(tokenizer.encode(t,add_special_tokens=True,truncation=False))>512 for t in texts)
                    batch=tokenizer(texts,max_length=512,padding=True,truncation=True,return_tensors='pt').to(a.device)
                    hidden=model(**batch).last_hidden_state
                    masked=hidden.masked_fill(~batch['attention_mask'][...,None].bool(),0.)
                    pooled=masked.sum(1)/batch['attention_mask'].sum(1)[...,None]
                    vectors.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy().astype(np.float32))
            values=np.concatenate(vectors,axis=0)
            assert values.shape==(len(group),768) and np.isfinite(values).all()
            assert np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5)
            partial=path.with_suffix('.partial')
            with partial.open('wb') as f:np.save(f,values)
            os.replace(partial,path)
            record={'ids':[d['id'] for d in group],'sha256':digest(path),'truncated_documents':truncated}
            proof.write_text(json.dumps(record,indent=2)+'\n')
        chunks.append(path)
        print(json.dumps({'completed_documents':min(begin+64,len(docs)),'expected_documents':len(docs),
                          'seconds':time.perf_counter()-started}),flush=True)
    matrix=np.concatenate([np.load(path,allow_pickle=False) for path in chunks],axis=0)
    target=a.output/'vectors.npy'
    with target.open('wb') as f:np.save(f,matrix)
    with (a.output/'head.npy').open('wb') as f:np.save(f,matrix[:2])
    (a.output/'ids.json').write_text(json.dumps([d['id'] for d in docs])+'\n')
    manifest={'spec':spec,'files':{name:digest(a.output/name) for name in ('vectors.npy','ids.json','spec.json','head.npy')},
        'seconds':time.perf_counter()-started,'truncated_documents':sum(json.loads(p.with_suffix('.json').read_text())['truncated_documents'] for p in chunks),
        'norm_max_error':float(np.max(np.abs(np.linalg.norm(matrix,axis=1)-1)))}
    completed.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'complete':True,'documents':len(docs),'seconds':manifest['seconds']}),flush=True)


if __name__=='__main__':main()
