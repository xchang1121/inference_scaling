"""Decode saved evidence after a completed cohort; never used by generation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--gold',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    raw = args.records.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if not any(r.get('kind') == 'complete' for r in rows):
        raise ValueError('only inspect a completed cohort offline')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    gold = {r['id']:r for line in args.gold.read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))}
    normalize = lambda text:re.sub(r'\s+',' ',text).strip().casefold()
    outputs = []
    for r in rows:
        if r.get('kind') != 'measurement':
            continue
        ids = [t for s in r['segments'] if s['kind'] in {'evidence','prompt_evidence'} for t in s['token_ids']]
        text = tokenizer.decode(ids,skip_special_tokens=False)
        answer = gold[r['question_id']]['answer']
        outputs.append({'question_id':r['question_id'],'arm':r['arm'],
            'trajectory_sha256':r['trajectory_sha256'],'evidence_tokens':len(ids),'evidence_text':text,
            'gold_answer_is_normalized_substring':normalize(answer) in normalize(text) if isinstance(answer,str) else None,
            'source_document_ids':[h['id'] for e in r['events'] if e.get('injected') for h in e['hits']]})
    args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs),encoding='utf-8')
    metadata = {'count':len(outputs),'records_sha256':hashlib.sha256(raw).hexdigest(),
        'tokenizer_sha256':hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Lexical exposure diagnostic only; substring absence does not prove absence of the fact.'}
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
