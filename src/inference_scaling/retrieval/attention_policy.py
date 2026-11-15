"""Causal sparse-attention DRAGIN adaptation, without rewriting committed IS.

Scores use only observed later rows within the configured prefix window. Missing
top-K attention weights are zero lower bounds, never imputed full attention.
Content filtering is explicitly on decoded token pieces rather than POS-tagged words.
"""
import math
import re

STOPWORDS = frozenset('a an the and or but if then else of to in on at by for from with '
                     'as is are was were be been being it its this that these those '
                     'i we you they he she have has had do does did not can could '
                     'would should will may might'.split())


def content_piece(token, decode):
    text = decode((token,), skip_special_tokens=True).strip().casefold()
    return bool(re.search(r'\w', text)) and text not in STOPWORDS and not (text.startswith('<') and text.endswith('>'))


def query_text(prefix, selected, decode, expand_words):
    if prefix is not None and expand_words:
        text=decode(prefix,skip_special_tokens=True)
        spans=set()
        for index,_,_ in selected:
            left_text=decode(prefix[:index],skip_special_tokens=True)
            right_text=decode(prefix[:index+1],skip_special_tokens=True)
            # Byte-fragment decoding can change the previous character. Only
            # use stable boundaries, never fabricate an offset into another word.
            if not text.startswith(left_text) or not text.startswith(right_text):continue
            left,right=len(left_text),len(right_text)
            while left<right and text[left].isspace():left+=1
            while right>left and text[right-1].isspace():right-=1
            while left>0 and re.match(r'[\w./:-]',text[left-1]):left-=1
            while right<len(text) and re.match(r'[\w./:-]',text[right]):right+=1
            if left<right:spans.add((left,right))
        merged=[]
        for left,right in sorted(spans):
            if merged and left<=merged[-1][1]:merged[-1]=(merged[-1][0],max(right,merged[-1][1]))
            else:merged.append((left,right))
        return ' '.join(text[left:right] for left,right in merged)
    groups=[]
    for index,_,token in selected:
        if groups and index==groups[-1][-1][0]+1:groups[-1].append((index,token))
        else:groups.append([(index,token)])
    return ' '.join(decode(tuple(t for _,t in g),skip_special_tokens=True).strip() for g in groups)


def attention_feature(rows, decode, *, query_tokens=8, content_only=True, expand_words=True):
    if not rows or query_tokens < 1:
        raise ValueError('nonempty causal rows and positive query budget required')
    scores, influences = [], []
    for i, row in enumerate(rows):
        a = row['attention']
        if len(a['indices']) != len(a['weights']) or len(row['attention_token_ids']) != len(a['indices']):
            raise ValueError('unaligned attention query tokens')
        if row['physical_index'] != a['prefix_tokens']:
            raise ValueError('attention must precede the sampled token')
        influence = max((weight for later in rows[i+1:]
                         for index, weight in zip(later['attention']['indices'], later['attention']['weights'], strict=True)
                         if index == row['physical_index']), default=0.)
        keep = not content_only or content_piece(row['token'], decode)
        value = row['entropy']*influence if keep else 0.
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid attention need score')
        scores.append(value)
        influences.append(influence)
    chosen = max(range(len(rows)), key=lambda i:(scores[i], rows[i]['entropy'], -i))
    row = rows[chosen]
    ranked = sorted(zip(row['attention']['indices'], row['attention']['weights'],
                        row['attention_token_ids'], strict=True),key=lambda x:(-x[1],x[0]))
    ranked = [r for r in ranked if not content_only or content_piece(r[2], decode)][:query_tokens]
    selected = sorted(ranked)
    return {'value':scores[chosen], 'uncertain_physical_index':row['physical_index'],
            'uncertain_token_id':row['token'], 'entropy':row['entropy'],
            'observed_max_influence':influences[chosen],
            'query_indices':[r[0] for r in selected], 'query_token_ids':[r[2] for r in selected],
            'query_text':query_text(row.get('attention_prefix_ids'),selected,decode,expand_words),
            'window_tokens':len(rows), 'attention_is_sparse_lower_bound':True,
            'state_offset':-1}
