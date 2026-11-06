"""Print complete selected thinking for an explicitly named, unblinded comparison."""
import argparse
import os
import coding_study as cs

parser = argparse.ArgumentParser()
parser.add_argument('problem_id')
parser.add_argument('modes', nargs='+', choices=cs.MODES)
args = parser.parse_args()
_, problems, records, _, _ = cs.load()
print('PROBLEM', args.problem_id)
print(problems[args.problem_id]['prompt'])
seen = []
for mode in args.modes:
    source = records[args.problem_id, mode]['thinking_text']
    print('\nMODE', mode, 'CHARACTERS', len(source))
    if seen:
        previous, count = max(((m, len(os.path.commonprefix([t, source]))) for m, t in seen), key=lambda x:x[1])
        if count:
            print(f'[Exact first {count} characters match {previous} above; remainder follows.]')
    else:
        count = 0
    print(source[count:])
    seen.append((mode, source))
