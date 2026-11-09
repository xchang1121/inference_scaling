"""One fixed, exploratory BM25 chunking diagnostic; no model generation."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source/src'))
from inference_scaling.retrieval import Document, Retriever
from inference_scaling.retrieval.corpus import build_query


def normalized(text):
    return ' '.join(re.findall(r'\w+',text.casefold()))


def main():
    corpus = BASE/'data/techqa/corpus.jsonl'
    target = BASE/'chunking-diagnostic.json'
    if target.exists():
        raise ValueError('preserve the first fixed diagnostic; do not tune on these questions')
    protocol = {'chunk_words':256,'overlap_words':64,'retriever':'bm25','top_k':[1,2,3,5],
        'questions':8,'generated_answers':0,'corpus_sha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Exploratory after the pilot questions were seen; not an independent test and not mixed into the thirteen primary arms.'}
    (BASE/'chunking-diagnostic-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    at = time.perf_counter()
    docs, parents, spans = [], {}, {}
    for line in corpus.read_text(encoding='utf-8').splitlines():
        d = json.loads(line)
        words = list(re.finditer(r'\S+',d['text']))
        if not words:
            continue
        start = 0
        while True:
            end = min(start+256,len(words))
            cid = f"{d['id']}::words-{start}-{end}"
            docs.append(Document(cid,d['title'],d['text'][words[start].start():words[end-1].end()],d['source']))
            parents[cid] = d['id']
            spans[cid] = {'start_word':start,'end_word':end,'start_char':words[start].start(),'end_char':words[end-1].end()}
            if end == len(words):
                break
            start += 192
    index = Retriever(docs,kind='bm25')
    build_seconds = time.perf_counter()-at
    questions = [json.loads(s) for s in (BASE/'data/techqa/questions.jsonl').read_text(encoding='utf-8').splitlines()]
    gold = {g['id']:g for s in (BASE/'data/techqa/gold.jsonl').read_text(encoding='utf-8').splitlines() if (g:=json.loads(s))}
    results, selected = [], {}
    for q in questions:
        query = build_query(q['question'],'',kind='question')
        hits = index.search(query,k=5)
        answer = normalized(gold[q['id']]['answer'])
        expected = set(gold[q['id']]['document_ids'])
        row = {'question_id':q['id'],'query':query,'hits':[
            {'id':h.document.id,'parent':parents[h.document.id],'score':h.score,**spans[h.document.id],
             'gold_parent':parents[h.document.id] in expected,
             'full_normalized_gold_answer_in_chunk':answer in normalized(h.document.text)} for h in hits]}
        for k in (1,2,3,5):
            row[f'parent_hit@{k}'] = any(h['gold_parent'] for h in row['hits'][:k])
            row[f'gold_text_in_chunk@{k}'] = any(h['full_normalized_gold_answer_in_chunk'] for h in row['hits'][:k])
        results.append(row)
        selected.update({h.document.id:h.document.__dict__ for h in hits})
    summary = {metric:sum(r[metric] for r in results)/len(results)
               for k in (1,2,3,5) for metric in (f'parent_hit@{k}',f'gold_text_in_chunk@{k}')}
    result = {'protocol':protocol,'index_chunks':len(docs),'index_fingerprint':index.fingerprint,
              'index_build_seconds':build_seconds,'summary':summary,'questions':results}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (BASE/'chunking-retrieved-documents.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected.values()),encoding='utf-8')
    print(json.dumps({'index_chunks':len(docs),'build_seconds':build_seconds,'summary':summary}),flush=True)


if __name__ == '__main__':
    main()
