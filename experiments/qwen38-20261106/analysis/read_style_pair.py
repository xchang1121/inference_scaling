"""Read one masked pair losslessly and record a human semantic-style annotation."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.stdin.reconfigure(encoding='utf-8')
root=Path(__file__).parent/'coding_bench_20261104/style_review'
packets=json.loads((root/'packets.json').read_text(encoding='utf-8'))
parser=argparse.ArgumentParser()
parser.add_argument('action',choices=['read','record','status'])
parser.add_argument('--index',type=int)
args=parser.parse_args()
path=root/'labels.jsonl'
labels=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
done={r['problem_id'] for r in labels}
if args.action=='status':
    print(json.dumps({'reviewed':len(done),'remaining':len(packets)-len(done)}))
elif args.action=='read':
    packet=packets[args.index] if args.index is not None else next(p for p in packets if p['problem_id'] not in done)
    print('PROBLEM',packet['problem_id'],'STRATUM',packet['stratum'])
    print(packet['prompt'])
    previous=''
    for c in packet['cases']:
        print('\nSIDE',c['side'],'REVIEW_ID',c['review_id'],'CHARACTERS',len(c['thinking_text']))
        common=len(os.path.commonprefix([previous,c['thinking_text']]))
        if common:
            print(f'[Exact first {common} characters are identical to side A above; remainder follows.]')
        print(c['thinking_text'][common:])
        previous=c['thinking_text']
elif args.action=='record':
    payload=json.load(sys.stdin)
    rows=payload if isinstance(payload,list) else [payload]
    for row in rows:
        packet=next(p for p in packets if p['problem_id']==row['problem_id'])
        assert row['problem_id'] not in done
        done.add(row['problem_id'])
        assert len(row['cases'])==2
        for source,label in zip(packet['cases'],row['cases']):
            assert source['review_id']==label['review_id'] and label['note']
            for k in ('exploration','revisiting','walkthrough','implementation','formalization','closure'):
                assert label[k] in (0,1,2) and (k!='formalization' or label[k] in (0,1))
            assert label['quote'] and label['quote'] in source['thinking_text']
            start=source['thinking_text'].index(label['quote'])
            label.update(quote_start=start,quote_end=start+len(label['quote']))
        assert row['comparison']
    with path.open('a',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'recorded':len(rows),'reviewed':len(done)}))
