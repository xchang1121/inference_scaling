"""Print complete pending cases, sharing only the repeated problem statement."""
import argparse
import difflib
import json
import os
import re
from pathlib import Path

import coding_study as cs

parser = argparse.ArgumentParser()
parser.add_argument('--characters', type=int, default=30000)
parser.add_argument('--problems', type=int, default=5)
parser.add_argument('--shared-prefix', action='store_true', help='Reference exact prefixes already read for the same problem.')
parser.add_argument('--shared-paragraphs', action='store_true', help='Reference exact paragraphs already read for the same problem; emit every novel character.')
parser.add_argument('--shared-spans', action='store_true', help='Losslessly reference exact spans of at least 400 characters in previously read same-problem thoughts.')
parser.add_argument('--shared-code', action='store_true', help='Reference complete identical code already read for the same problem.')
args = parser.parse_args()
done = cs.labels()
all_cases = cs.read_jsonl(cs.OUT/'review-cases.jsonl')
reviewed_problem_ids = {c['problem_id'] for c in all_cases if c['case_id'] in done}
pending = [c for c in all_cases if c['case_id'] not in done]
known = {}
paragraphs = {}
def paragraph_spans(value):
    return [m for m in re.finditer(r'.+?(?:\n\n|\Z)', value, re.S)]
def render_spans(case, prefix, previous_cases):
    value = case['thinking_text'][prefix:]
    matches = list(re.finditer(r'\s+|\S+', value))
    tokens = [m[0] for m in matches]
    spans = []
    for old in previous_cases:
        previous = list(re.finditer(r'\s+|\S+', old['thinking_text']))
        matcher = difflib.SequenceMatcher(None, [m[0] for m in previous], tokens, autojunk=True)
        for block in matcher.get_matching_blocks():
            if not block.size:
                continue
            start, end = matches[block.b].start(), matches[block.b+block.size-1].end()
            old_start, old_end = previous[block.a].start(), previous[block.a+block.size-1].end()
            if end - start >= 400:
                assert value[start:end] == old['thinking_text'][old_start:old_end]
                spans.append((start,end,old['case_id'],old_start))
    rendered, reconstructed, cursor = [], [], 0
    while cursor < len(value):
        options = [s for s in spans if s[0] <= cursor and s[1]-cursor >= 400]
        if options:
            start,end,cid,old_start = max(options,key=lambda s:s[1])
            offset = old_start + cursor - start
            rendered.append('\n[EXACT SPAN: same as CASE '+cid+' thinking characters ['+str(offset)+':'+str(offset+end-cursor)+'].]\n')
            reconstructed.append(next(c for c in previous_cases if c['case_id']==cid)['thinking_text'][offset:offset+end-cursor])
            cursor = end
        else:
            end = min([s[0] for s in spans if s[0] > cursor] or [len(value)])
            rendered.append(value[cursor:end])
            reconstructed.append(value[cursor:end])
            cursor = end
    assert ''.join(reconstructed) == value
    return ''.join(rendered)
for c in all_cases:
    if c['case_id'] in done:
        known.setdefault(c['problem_id'], []).append(c)
        for m in paragraph_spans(c['thinking_text']):
            if len(m[0]) >= 200:
                paragraphs.setdefault((c['problem_id'], m[0]), (c['case_id'], m.start(), m.end()))
chosen, seen, size = [], set(), 0
for case in pending:
    is_new = case['problem_id'] not in seen
    prefix, reference = 0, None
    if args.shared_prefix:
        for previous in known.get(case['problem_id'], []):
            length = len(os.path.commonprefix([previous['thinking_text'], case['thinking_text']]))
            length = case['thinking_text'].rfind('\n\n', 0, length) + 2
            if length > max(prefix, 1000):
                prefix, reference = length, previous['case_id']
    case = {**case, 'read_prefix_characters': prefix, 'read_prefix_reference': reference}
    additions, rendered = {}, []
    for m in paragraph_spans(case['thinking_text'][prefix:]):
        key = (case['problem_id'], m[0])
        old = additions.get(key, paragraphs.get(key)) if args.shared_paragraphs and len(m[0]) >= 200 else None
        if old:
            rendered.append('[EXACT PARAGRAPH: same as CASE '+old[0]+' thinking characters ['+str(old[1])+':'+str(old[2])+'].]\n\n')
        else:
            rendered.append(m[0])
        if len(m[0]) >= 200:
            additions.setdefault(key, (case['case_id'], prefix+m.start(), prefix+m.end()))
    rendered = ''.join(rendered)
    if args.shared_spans:
        alternative = render_spans(case, prefix, known.get(case['problem_id'], []))
        if len(alternative) < len(rendered):
            rendered = alternative
    case['read_rendered_thinking'] = rendered
    code_reference = next((old['case_id'] for old in known.get(case['problem_id'], [])
                           if old['submitted_code'] == case['submitted_code']), None) if args.shared_code else None
    case['read_code_reference'] = code_reference
    extra = len(rendered) + (120 if code_reference else len(case['submitted_code'])) + (len(case['prompt']) if is_new and case['problem_id'] not in reviewed_problem_ids else 0) + 250
    if chosen and (size + extra > args.characters or is_new and len(seen) >= args.problems):
        break
    chosen.append(case)
    seen.add(case['problem_id'])
    size += extra
    known.setdefault(case['problem_id'], []).append(case)
    paragraphs.update({k:v for k,v in additions.items() if k not in paragraphs})
(cs.ROOT/'review-batch.json').write_text(json.dumps(chosen,ensure_ascii=False),encoding='utf-8')
print('BATCH',len(chosen),'cases',len(seen),'problems',size,'characters')
seen = set()
for case in chosen:
    if case['problem_id'] not in seen:
        print('\nPROBLEM',case['problem_id'])
        if case['problem_id'] in reviewed_problem_ids:
            print('[Same problem statement as previously reviewed cases for this problem.]')
        else:
            print(case['prompt'])
        seen.add(case['problem_id'])
    print('\nCASE',case['case_id'])
    if case['read_prefix_reference']:
        print('EXACT SHARED PREFIX: characters [0:'+str(case['read_prefix_characters'])+'] match previously read CASE '+case['read_prefix_reference']+'. Continuation follows:')
    print(case['read_rendered_thinking'])
    print('\nSUBMITTED CODE\n'+('[EXACT COMPLETE CODE: same as CASE '+case['read_code_reference']+'.]' if case['read_code_reference'] else case['submitted_code']))
    print('END CASE',case['case_id'])
print('END BATCH',','.join(c['case_id'] for c in chosen))
