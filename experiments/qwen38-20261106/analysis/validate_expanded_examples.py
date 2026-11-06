"""Reproduce only the concrete witnesses used by the detailed report."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import coding_study as cs


def validate(cases, records):
    frozen_paths = [cs.ROOT / 'snapshots' / replica / folder / filename
                    for replica in ('a', 'b')
                    for folder, filename in [('generation', f'{replica}.jsonl'),
                                             ('evaluation', f'{replica}-judgments.jsonl')]]
    before = {str(p.relative_to(cs.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    checks = []

    def check(name, actual, expected, **details):
        assert actual == expected, (name, actual, expected)
        checks.append({'name': name, 'actual': actual, 'expected': expected, **details})

    def call(cid, target, args):
        program = ('from typing import List\nimport json,sys\n' + cases[cid]['submitted_code']
                   + '\n_arguments=json.loads(sys.stdin.read())\n'
                   + f'print(json.dumps({target}(*_arguments)))\n')
        completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', program],
                                   input=json.dumps(args), text=True, encoding='utf-8',
                                   capture_output=True, timeout=10)
        if completed.returncode:
            return {'error': completed.stderr.strip().splitlines()[-1]}
        return json.loads(completed.stdout)

    def stdio(cid, data, wrap=False):
        code = cases[cid]['submitted_code']
        if wrap:
            code = 'def code():\n' + '\n'.join('    ' + s for s in code.splitlines()) + '\ncode()\n'
        completed = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], input=data,
                                   text=True, encoding='utf-8', capture_output=True, timeout=10)
        if completed.returncode:
            return {'error': completed.stderr.strip().splitlines()[-1]}
        return completed.stdout.strip()

    sample = '5 3\n2 4 1 3 3\n2 5\n5 2\n5 3\n'
    check('01_C_incomplete_submission', stdio('5064c85c8239689fcc75', sample, wrap=True),
          {'error': "NameError: name 'root' is not defined"}, input=sample)
    check('01_SC_public_example', stdio('d05f2eb2cc1efda64fa1', sample), '2\n1\n2', input=sample)

    @lru_cache(None)
    def pizza_optimum(remaining, day):
        if not remaining:
            return 0
        return max(group[-1 if day % 2 else -2]
                   + pizza_optimum(tuple(x for x in remaining if x not in group), day + 1)
                   for group in combinations(remaining, 4))

    check('02_exhaustive_optimum', pizza_optimum(tuple(range(1, 13)), 1), 32)
    check('02_C_counterexample', call('3db6e3d8e61b2957adf6', 'Solution().maxWeight', [list(range(1, 13))]), 32)
    check('02_SC_counterexample', call('eaf6f9fbbe5fd0bc397d', 'Solution().maxWeight', [list(range(1, 13))]), 31)

    vals = [1, 11, -1, -11, -12]
    check('03_Base_public_example', call('d010e4e1149deb826268', 'order_by_points', [vals]), [1, -1, 11, -11, -12])
    check('03_BoN_public_example', call('381be7956f422da17538', 'order_by_points', [vals]), [-1, -11, 1, -12, 11])

    vals = [10, 10, 3, 7, 6]
    expected = sum((sum(vals[:i]) - sum(vals[i:])) % 2 == 0 for i in range(1, len(vals)))
    check('04_direct_partition_count', expected, 4)
    for cid in ('522147770130ef3737ec', 'b5fa890d166725201216'):
        check('04_' + cid, call(cid, 'Solution().countPartitions', [vals]), expected)

    groups, elements = [8, 4, 3, 2, 4], [4, 2]
    expected = [next((j for j, e in enumerate(elements) if g % e == 0), -1) for g in groups]
    for cid in ('737c671191e54701b73a', 'b6689e2b24a0e86ef7d5'):
        check('05_' + cid, call(cid, 'Solution().assignElements', [groups, elements]), expected)

    args = [[4, 2, 1, 5, 3], [10, 20, 30, 40, 50], 2]
    result = call('3b67bd32e7ad7a72a64e', 'Solution().findMaxSum', args)
    assert isinstance(result, dict) and 'AttributeError' in result['error'] and 'heapop' in result['error']
    checks.append({'name': '06_C_API_typo', 'actual': result, 'expected': 'AttributeError for heapq.heapop'})
    check('06_SC_public_example', call('4909fe4435fc4c3f8f62', 'Solution().findMaxSum', args), [80, 30, 0, 80, 50])

    nums, queries = [7, 1, 2, 7, 3, 4, 5, 6, 8, 7], [0]
    expected = [min(min(abs(i - j), len(nums) - abs(i - j)) for j in range(len(nums))
                    if i != j and nums[j] == nums[i]) for i in queries]
    check('07_exhaustive_distance', expected, [1], nums=nums, queries=queries)
    check('07_met_k2', call('86217df34921541dfbda', 'Solution().solveQueries', [nums, queries]), [3])
    check('07_fast_is', call('c81caf648a042532f711', 'Solution().solveQueries', [nums, queries]), [1])

    dice_input = '3\n2 1 2\n1 3\n2 1 2\n'
    check('08_C_state_mutation', float(stdio('b9b0c5cc051f21637bcb', dice_input)), 0.0, input=dice_input)
    check('08_SC_state_preserved', float(stdio('f35599a7e0885bd605e3', dice_input)), 0.5, input=dice_input)
    tol_row = records['lcb/abc392_d', 'is-self-certainty']['judgment']
    meta = tol_row['metadata']
    tokens = list(map(int, meta['inputs'].split()))
    n, cursor, dice = tokens[0], 1, []
    for _ in range(n):
        k = tokens[cursor]
        dice.append(tokens[cursor + 1:cursor + 1 + k])
        cursor += k + 1
    optimum = max(Fraction(sum(Counter(x)[v] * Counter(y)[v] for v in set(x) & set(y)), len(x) * len(y))
                  for x, y in combinations(dice, 2))
    assert optimum == Fraction(1, 13)
    pred, expected_text = meta['output'].strip(), meta['expected'].strip()
    assert pred == '0.076923076923077' and expected_text == '0.076923076923'
    for cid in ('b9b0c5cc051f21637bcb', 'f35599a7e0885bd605e3'):
        check('08_frozen_float_input_' + cid, stdio(cid, meta['inputs']), pred)
    true_decimal = Decimal(optimum.numerator) / Decimal(optimum.denominator)
    pred_error = abs(Decimal(pred) - true_decimal)
    reference_error = abs(Decimal(expected_text) - true_decimal)
    assert pred_error < Decimal('1e-8') and reference_error < Decimal('1e-8')
    assert Decimal(pred) != Decimal(expected_text)
    grader = cs.ROOT / 'vendor/LiveCodeBench/lcb_runner/evaluation/testing_util.py'
    grader_source = grader.read_text(encoding='utf-8')
    assert 'if decimal_prediction_line == decimal_gtout_line:' in grader_source
    tolerance = {
        'problem_id': 'lcb/abc392_d', 'input': meta['inputs'], 'true_probability': str(optimum),
        'observed_output': pred, 'reference_output': expected_text,
        'absolute_error_of_output': str(pred_error), 'absolute_error_of_reference': str(reference_error),
        'problem_tolerance': '1e-8', 'within_problem_tolerance': True,
        'frozen_judge_correct': tol_row['correct'], 'frozen_error_message': meta['error_message'],
        'grader_path': str(grader.relative_to(cs.ROOT)),
        'grader_sha256': hashlib.sha256(grader.read_bytes()).hexdigest(),
        'grader_comparison': 'decimal_prediction_line == decimal_gtout_line',
        'interpretation': 'Exact decimal equality rejects distinct representations within the stated tolerance.',
    }

    def subsequence_optimum(nums, k, limit):
        products = []
        for mask in range(1, 1 << len(nums)):
            sub = [x for i, x in enumerate(nums) if mask & (1 << i)]
            if sum(x if i % 2 == 0 else -x for i, x in enumerate(sub)) == k and math.prod(sub) <= limit:
                products.append(math.prod(sub))
        return max(products, default=-1)

    for nums, k, limit, answer in [([0], 0, 1, 0), ([10, 0], 10, 1, 0), ([3, 2, 1, 3], 4, 8, 6)]:
        check('09_exhaustive_' + str(nums), subsequence_optimum(nums, k, limit), answer,
              nums=nums, k=k, limit=limit)
    check('09_Base_empty_subsequence', call('33047554383b3cc4b3a7', 'Solution().maxProduct', [[0], 0, 1]), 1)
    check('09_Base_lost_smaller_product', call('33047554383b3cc4b3a7', 'Solution().maxProduct', [[3, 2, 1, 3], 4, 8]), -1)
    check('09_block_kept_smaller_product', call('a66262b492a64f368c33', 'Solution().maxProduct', [[3, 2, 1, 3], 4, 8]), 6)
    check('09_block_zero', call('a66262b492a64f368c33', 'Solution().maxProduct', [[0], 0, 1]), 0)
    check('09_block_zero_recovery', call('a66262b492a64f368c33', 'Solution().maxProduct', [[10, 0], 10, 1]), -1)
    row = records['lcb/3777', 'block-met-k2']
    assert row['judgment']['correct'] and row['judgment']['total_tests'] == 43
    assert row['thinking_text'] == records['lcb/3777', 'block-met-k2-self-certainty']['thinking_text']
    assert row['submitted_code'] == records['lcb/3777', 'block-met-k2-self-certainty']['submitted_code']

    def jump_possible(n, a, b, intervals):
        reachable = {1}
        for position in range(1, n + 1):
            if position in reachable:
                for dest in range(position + a, min(n, position + b) + 1):
                    if not any(left <= dest <= right for left, right in intervals):
                        reachable.add(dest)
        return n in reachable

    check('10_SC_counterexample_oracle', jump_possible(2, 2, 3, []), False)
    check('10_SC_actual', stdio('8365280d2bcd3e04608f', '2 0 2 3\n'), 'Yes')
    check('10_C_zero_step_oracle', jump_possible(3, 1, 2, [(2, 2)]), True)
    check('10_C_actual', stdio('b9d08bc11b68a11e90de', '3 1 1 2\n2 2\n'), 'No')

    sequence = [0, 0, 2, 0]
    for n in range(4, 7):
        sequence.append(sum(sequence[n - 4:n]))
    check('11_fib4_oracle', sequence[6], 8)
    check('11_fast_SC', call('2d08d33a41e0d57c34e4', 'fib4', [6]), 6)
    check('11_met_k1_SC', call('8482a026a8c999fc404e', 'fib4', [6]), 8)

    check('12_Base_tab', call('82b01251f220ee233da5', 'split_words', ['a\tb']), 1)
    check('12_fast_C_tab', call('85115e5cbc606d0d3adf', 'split_words', ['a\tb']), ['a', 'b'])
    for mode in ('base', 'is'):
        assert records['HumanEval/125', mode]['judgment']['correct']

    for cid in ('dfddce6f0e269c7bc341', '312775e042c982c108c5'):
        check('13_' + cid, call(cid, 'do_algebra', [['+', '*', '-'], [2, 3, 4, 5]]), 9)
    check('14_BoN_negative_digit', call('da237fe7ba135d10b36c', 'multiply', [14, -12]), 32)
    check('14_met_k1_negative_digit', call('b994d3b80115eaa2d5db', 'multiply', [14, -12]), 8)
    numbers = [-10, 4, 6, 1000, 10, 20]
    check('15_median_oracle', statistics.median(numbers), 8.0)
    for cid in ('3f0ec062125a71ff7413', '68487bab726a484bd3fd'):
        check('15_' + cid, call(cid, 'median', [numbers]), 8.0)

    after = {str(p.relative_to(cs.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    assert before == after
    return {'witness_checks_passed': len(checks), 'checks': checks,
            'frozen_generation_and_judgment_hashes': after,
            'grading_tolerance_finding': tolerance,
            'verification_scope': 'Only the examples cited in the narrative; no new model generation or aggregate regrading.'}
