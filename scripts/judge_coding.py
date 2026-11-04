"""Offline CPU judges using pinned upstream implementations, without feedback to generation."""
from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import time
import zlib
from pathlib import Path

from coding_common import digest, read_jsonl


def read_live(path):
    data = Path(path).read_bytes() if Path(path).exists() else b''
    # The only tolerated incomplete JSON is an in-flight final append.
    lines = data.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n')]


def lcb_sample(problem):
    public = json.loads(problem['public_test_cases'])
    try:
        private = json.loads(problem['private_test_cases'])
    except json.JSONDecodeError:
        # This is the upstream pinned dataset format, never a model-supplied pickle.
        private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(problem['private_test_cases']))))
    tests = public + private
    if json.loads(problem['metadata']).get('func_name'):
        for test in tests:
            [json.loads(line) for line in test['input'].split('\n')]
            json.loads(test['output'])
    return {'input_output': json.dumps({'inputs': [x['input'] for x in tests],
                                        'outputs': [x['output'] for x in tests],
                                        'fn_name': json.loads(problem['metadata']).get('func_name')})}


def grade_lcb(sample, code):
    from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations_by_problem
    results, metadata = evaluate_generations_by_problem(([code], sample, False, 6))
    values, detail = results[0], metadata[0]
    if detail.get('error_code') == -5:
        return {'status': 'judge_error', 'correct': None, 'metadata': detail}
    return {'status': 'graded', 'correct': bool(values) and all(v > 0 for v in values),
            'test_results': values, 'metadata': detail,
            'total_tests': len(json.loads(sample['input_output'])['inputs']), 'timeout_per_test': 6}


def grade_he(problem, code, cache):
    from evalplus.eval import PASS, untrusted_check
    from evalplus.gen.util import trusted_exec
    key = digest(problem)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (key + '.pickle')
    if path.exists():
        expected = pickle.loads(path.read_bytes())
    else:
        expected = {name: trusted_exec(problem['prompt'] + problem['canonical_solution'],
                                       problem[name + '_input'], problem['entry_point'], record_time=True)
                    for name in ('base', 'plus')}
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(pickle.dumps(expected))
        temporary.replace(path)
    results = {}
    for name in ('base', 'plus'):
        outputs, seconds = expected[name]
        status, details = untrusted_check('humaneval', code, problem[name + '_input'], problem['entry_point'],
                                          expected=outputs, atol=problem['atol'], ref_time=seconds,
                                          fast_check=False, min_time_limit=4.0, gt_time_limit_factor=4.0)
        results[name] = {'status': status, 'test_results': [bool(x) for x in details],
                         'total_tests': len(problem[name + '_input'])}
    return {'status': 'graded', 'correct': all(results[n]['status'] == PASS for n in ('base', 'plus')),
            'base_correct': results['base']['status'] == PASS, **results,
            'min_time_limit': 4.0, 'gt_time_limit_factor': 4.0, 'fast_check': False}


def self_test(data, cache):
    # Positive and negative controls exercise both official API forms and all HE+ inputs.
    problem = data['humanevalplus']['HumanEval/0']
    good = grade_he(problem, problem['prompt'] + problem['canonical_solution'], cache)
    bad = grade_he(problem, 'def has_close_elements(numbers, threshold):\n    return False\n', cache)
    stdin = {'input_output': json.dumps({'inputs': ['2 3\n', '5 -8\n'], 'outputs': ['5\n', '-3\n'], 'fn_name': None})}
    func = {'input_output': json.dumps({'inputs': ['[1,2]', '[5]'], 'outputs': ['3', '5'], 'fn_name': 'sumArray'})}
    checks = {'humaneval_canonical': good['correct'], 'humaneval_wrong_rejected': not bad['correct'],
              'lcb_stdin_good': grade_lcb(stdin, 'a,b=map(int,input().split());print(a+b)')['correct'],
              'lcb_stdin_wrong_rejected': not grade_lcb(stdin, 'print(0)')['correct'],
              'lcb_function_good': grade_lcb(func, 'class Solution:\n    def sumArray(self, a): return sum(a)')['correct'],
              'lcb_function_wrong_rejected': not grade_lcb(func, 'class Solution:\n    def sumArray(self, a): return 0')['correct']}
    sample_counts = {key: len(json.loads(lcb_sample(value)['input_output'])['inputs'])
                     for key, value in data['livecodebench'].items()}
    assert all(checks.values()) and all(sample_counts.values()), checks
    return {'checks': checks, 'lcb_parsed': len(sample_counts),
            'lcb_test_count_range': [min(sample_counts.values()), max(sample_counts.values())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inputs', nargs='*', type=Path, default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding='utf-8'))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    public = {p['id']: p for p in manifest['problems']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / 'groundtruth-cache'
    if args.self_test:
        result = self_test(data, cache)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result), flush=True)
        return
    import fcntl
    with args.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = read_jsonl(args.output)
        identity = digest({'data': digest(data), 'manifest': digest(manifest),
                           'evalplus': '26d6d00bb1fd0fa37f39c99d5290da67891d1c5e',
                           'livecodebench': '28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24',
                           'script': Path(__file__).read_text(encoding='utf-8')})
        if records and records[0].get('identity') != identity:
            raise ValueError('judge resume identity mismatch')
        done = {(r['problem_id'], r['mode'], r['submission_sha256']) for r in records if r.get('kind') == 'judgment'}
        reused = {(r['problem_id'], r['submission_sha256']): r['judgment'] for r in records
                  if r.get('kind') == 'judgment' and r['judgment']['status'] == 'graded'}
        with args.output.open('a', encoding='utf-8') as sink:
            def write(row):
                sink.write(json.dumps(row, ensure_ascii=False) + '\n')
                sink.flush()
                os.fsync(sink.fileno())
            if not records:
                write({'kind': 'metadata', 'identity': identity, 'judge': 'pinned upstream CPU evaluators',
                       'manifest_digest': digest(manifest), 'data_digest': digest(data)})
            while True:
                all_complete = bool(args.inputs)
                for path in args.inputs:
                    rows = read_live(path)
                    all_complete &= any(r.get('kind') == 'generation_complete' for r in rows)
                    for row in rows:
                        if row.get('kind') != 'measurement' or row.get('status') != 'complete':
                            continue
                        key = row['problem_id'], row['mode'], row['submission_sha256']
                        if key in done:
                            continue
                        if digest(row['submitted_code']) != row['submission_sha256']:
                            raise ValueError('submission checksum mismatch')
                        meta = public[row['problem_id']]
                        cached_key = row['problem_id'], row['submission_sha256']
                        started = time.perf_counter()
                        if cached_key in reused:
                            result = reused[cached_key]
                        elif not row['submitted_code']:
                            result = {'status': 'graded', 'correct': False, 'reason': 'no_final_fenced_submission'}
                        elif meta['dataset'] == 'humanevalplus':
                            result = grade_he(data['humanevalplus'][meta['task_id']], row['submitted_code'], cache)
                        else:
                            result = grade_lcb(lcb_sample(data['livecodebench'][meta['task_id']]), row['submitted_code'])
                        write({'kind': 'judgment', 'problem_id': row['problem_id'], 'mode': row['mode'],
                               'submission_sha256': row['submission_sha256'], 'judgment': result,
                               'seconds': time.perf_counter() - started, 'reused_identical_code': cached_key in reused})
                        done.add(key)
                        if result['status'] != 'graded':
                            raise RuntimeError(f'judge infrastructure error: {key}: {result}')
                        reused[cached_key] = result
                        print(json.dumps({'problem_id': row['problem_id'], 'mode': row['mode'],
                                          'correct': result['correct'], 'graded': len(done)}), flush=True)
                if all_complete or not args.follow:
                    break
                time.sleep(30)


if __name__ == '__main__':
    main()
