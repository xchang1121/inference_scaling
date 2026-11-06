"""Build paired reports and blinded thought-review packets from frozen snapshots."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent / 'coding_bench_20261104'
OUT = ROOT.parents[1] / 'outputs/qwen38-coding-20261104'
sys.path[:0] = [str(ROOT / 'source/src'), str(ROOT / 'source/scripts')]
from coding_common import MODES, digest, read_jsonl
from inference_scaling.shared.rng import SeedStream

NAMES = {'base': 'Base', 'is': '快速 IS + C', 'sir-k1': 'SIR K=1 + C', 'met-k1': '逐条 K=1 + C',
         'sir-k2': 'SIR K=2 + C', 'met-k2': '逐条 K=2 + C', 'block-met-k2': '按块 K=2 + C',
         'bon-consilience': 'Best-of-4 + C', 'is-self-certainty': '快速 IS + SC',
         'met-k1-self-certainty': '逐条 K=1 + SC', 'met-k2-self-certainty': '逐条 K=2 + SC',
         'block-met-k2-self-certainty': '按块 K=2 + SC'}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    path.with_suffix(path.suffix + '.tmp').write_text(text, encoding='utf-8')
    path.with_suffix(path.suffix + '.tmp').replace(path)


def load():
    manifest = json.loads((ROOT / 'frozen/public-manifest.json').read_text(encoding='utf-8'))
    problems = {p['id']: p for p in manifest['problems']}
    records, status, attempts = {}, {}, []
    for replica in ('a', 'b'):
        folder = ROOT / 'snapshots' / replica
        snapshot = folder / 'snapshot.json'
        if not snapshot.exists():
            status[replica] = {'state': 'not_collected'}
            continue
        snap = json.loads(snapshot.read_text(encoding='utf-8'))
        assert not snap['changed_frozen_files'], snap['changed_frozen_files']
        rows = read_jsonl(folder / f'generation/{replica}.jsonl')
        grades = {(r['problem_id'], r['mode'], r['submission_sha256']): r['judgment']
                  for r in read_jsonl(folder / f'evaluation/{replica}-judgments.jsonl') if r.get('kind') == 'judgment'}
        starts = Counter((r['problem_id'], r['mode']) for r in rows if r.get('kind') == 'attempt_start')
        ended = Counter((r['problem_id'], r['mode']) for r in rows if r.get('kind') == 'measurement')
        state = json.loads((folder / f'{replica}-queue.json').read_text())
        state.update(snapshot_time=snap['time'], frozen_verified=True,
                     generation_exit=(folder / f'{replica}-generation.exit').read_text().strip()
                     if (folder / f'{replica}-generation.exit').exists() else None,
                     judge_exit=(folder / f'{replica}-judge.exit').read_text().strip()
                     if (folder / f'{replica}-judge.exit').exists() else None,
                     open_attempts=sum((starts - ended).values()))
        status[replica] = state
        for line_number, row in enumerate(rows, 1):
            if row.get('kind') != 'measurement':
                continue
            if row['status'] != 'complete':
                attempts.append({'replica': replica, **row})
                continue
            key = row['problem_id'], row['mode']
            assert key not in records, f'duplicate measurement: {key}'
            assert row['mode'] in MODES and row['problem_id'] in problems
            assert problems[row['problem_id']]['global_index'] % 2 == (0 if replica == 'a' else 1)
            assert row['seed'] == SeedStream(20260808).derive('paired-benchmark', row['problem_id'], 0)
            assert digest(row['submitted_code']) == row['submission_sha256']
            records[key] = {**row, 'replica': replica, 'source_line': line_number,
                            'source_file': str(folder / f'generation/{replica}.jsonl'),
                            'judgment': grades.get((*key, row['submission_sha256']))}
    return manifest, problems, records, status, attempts


def wilson(correct, count):
    if not count:
        return None
    p, z = correct / count, 1.959963984540054
    center = (p + z*z/(2*count))/(1+z*z/count)
    radius = z * math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [center-radius, center+radius]


def native_flops(record):
    return record.get('graph_execution', {}).get('estimated_dense_padded_flops')


def paired(a, b):
    n = len(a)
    delta = np.array([int(y['judgment']['correct']) - int(x['judgment']['correct']) for x, y in zip(a, b)])
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    rng = np.random.default_rng(20261104)
    strata = {replica: np.array([i for i, r in enumerate(a) if r['replica'] == replica]) for replica in ('a', 'b')}
    indices = np.concatenate([rng.choice(ix, size=(5000, len(ix)), replace=True)
                              for ix in strata.values() if len(ix)], axis=1)
    latency = np.array([x['seconds'] for x in a]), np.array([x['seconds'] for x in b])
    ci = lambda values: [float(x) for x in np.quantile(values, [.025, .975])]
    flops_a, flops_b = [native_flops(x) for x in a], [native_flops(x) for x in b]
    result = {'n': n, 'wins': wins, 'losses': losses, 'ties': n-wins-losses,
              'accuracy_difference': float(delta.mean()), 'difference_bootstrap_95': ci(delta[indices].mean(axis=1)),
              'wall_ratio': sum(latency[1])/sum(latency[0]),
              'wall_ratio_bootstrap_95': ci(latency[1][indices].sum(axis=1)/latency[0][indices].sum(axis=1)),
              'mcnemar_exact_p': min(1., 2 * sum(math.comb(wins+losses, i) for i in range(min(wins, losses)+1)) / 2**(wins+losses))
                                  if wins+losses else 1.0}
    if all(x is not None and x > 0 for x in flops_a + flops_b):
        fa, fb = np.array(flops_a), np.array(flops_b)
        result.update(flops_ratio=sum(flops_b)/sum(flops_a),
                      flops_ratio_bootstrap_95=ci(fb[indices].sum(axis=1)/fa[indices].sum(axis=1)))
    return result


def review_packets(problems, records, *, ready=False):
    # User requested reading and example selection only after all judging finishes.
    if not ready:
        key_path = OUT / 'review-key.json'
        return (read_jsonl(OUT / 'review-cases.jsonl'),
                json.loads(key_path.read_text(encoding='utf-8')) if key_path.exists() else {})
    complete = [pid for pid in problems if all((pid, m) in records and records[pid,m]['judgment'] is not None
                                               and records[pid,m]['judgment']['status'] == 'graded' for m in MODES)]
    assert len(complete) == len(problems)
    cases, key = {}, {}
    for pid in complete:
        for mode in MODES:
            row = records[pid, mode]
            identity = digest([pid, row['thinking_text'], row['submitted_code']])[:20]
            cases.setdefault(identity, {'case_id': identity, 'problem_id': pid, 'prompt': problems[pid]['prompt'],
                                         'cohort': 'full',
                                         'thinking_text': row['thinking_text'], 'submitted_code': row['submitted_code']})
            key.setdefault(identity, []).append({'problem_id': pid, 'mode': mode, 'replica': row['replica'],
                                                 'source_file': row['source_file'], 'source_line': row['source_line'],
                                                 'submission_sha256': row['submission_sha256'],
                                                 'correct': row['judgment'].get('correct')})
    ordered = sorted(cases.values(), key=lambda x: (problems[x['problem_id']]['global_index'], x['case_id']))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'review-cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in ordered), encoding='utf-8')
    save(OUT / 'review-key.json', key)
    return ordered, key


def labels():
    result = {}
    for row in read_jsonl(OUT / 'review-labels.jsonl'):
        result[row['case_id']] = row
    return result


def prepare():
    manifest, problems, records, status, attempts = load()
    review_ready = (len(records) == len(problems) * len(MODES)
                    and all(s.get('generation_exit') == '0' and s.get('judge_exit') == '0' for s in status.values())
                    and all(r['judgment'] is not None and r['judgment']['status'] == 'graded' for r in records.values()))
    cases, key = review_packets(problems, records, ready=review_ready)
    reviewed = labels()
    trace_rows = []
    for case in cases:
        for ref in key[case['case_id']]:
            row = records[ref['problem_id'], ref['mode']]
            trace = row['trace']
            steps = trace.get('steps', [])
            trace_rows.append({'case_id': case['case_id'], **ref,
                               'thinking_characters': len(row['thinking_text']),
                               'selection_steps': len(steps),
                               'reported_rollout_evaluations': trace.get('rollout_evaluations'),
                               'new_rollout_evaluations': sum(s.get('rollout_evaluations_performed', 0) for s in steps),
                               'step_details': [{k: s.get(k) for k in ('prefix_length', 'selected_index',
                                                'completion_index', 'retained_candidate',
                                                'rollout_evaluations_performed', 'sampling_plan')}
                                                for s in steps]})
    save(OUT / 'review-traces.json', trace_rows)
    checks = []
    for row in records.values():
        full_pool = row['mode'].removesuffix('-self-certainty') in ('sir-k1', 'sir-k2', 'met-k1', 'met-k2', 'block-met-k2')
        counters = row['counters']
        checks.append({'problem_id': row['problem_id'], 'mode': row['mode'],
                       'seed_and_shard_match': True, 'native_cost_recorded': native_flops(row) is not None,
                       'no_rescoring_forward': counters.get('score_forward_token_slots', 0) == 0,
                       'no_preemption': counters.get('preemptions', 0) == 0,
                       'no_fallback': not row['fallbacks'],
                       'full_pool_no_cancel': not full_pool or counters.get('cancelled_requests', 0) == 0,
                       'full_pool_wait': not full_pool or row['algorithm_config'].get('selection_ready') is False})
    datasets = {}
    for dataset in ('humanevalplus', 'livecodebench'):
        pids = [pid for pid, p in problems.items() if p['dataset'] == dataset]
        common = [pid for pid in pids if all((pid,m) in records and records[pid,m]['judgment'] is not None
                                             and records[pid,m]['judgment']['status'] == 'graded' for m in MODES)]
        methods = []
        for mode in MODES:
            available = [r for (pid,m),r in records.items() if m == mode and pid in pids]
            graded = [r for r in available if r['judgment'] and r['judgment']['status'] == 'graded']
            rows = [records[pid,mode] for pid in common]
            correct = sum(r['judgment']['correct'] for r in rows)
            cost = [native_flops(r) for r in rows]
            methods.append({'mode': mode, 'generated': len(available), 'graded': len(graded),
                             'common_n': len(common), 'correct': correct,
                             'accuracy': correct/len(common) if common else None, 'wilson_95': wilson(correct,len(common)),
                             'mean_seconds': statistics.fmean(r['seconds'] for r in rows) if rows else None,
                             'pflops': sum(cost)/1e15 if rows and all(x is not None for x in cost) else None,
                             'mean_thinking_characters': statistics.fmean(len(r['thinking_text']) for r in rows) if rows else None,
                             'length_exhausted': sum(r['length_exhausted'] for r in rows),
                             'incomplete_thinking': sum(r['thinking_status'] != 'complete' for r in rows),
                             'empty_submission': sum(not r['submitted_code'] for r in rows)})
        comparisons = []
        if common:
            for left, right in itertools.combinations(MODES, 2):
                comparisons.append({'reference': left, 'alternative': right,
                                    **paired([records[pid,left] for pid in common], [records[pid,right] for pid in common])})
        datasets[dataset] = {'expected_problems': len(pids), 'paired_complete_problems': len(common),
                             'common_problem_ids': common, 'methods': methods, 'comparisons': comparisons}
    total = len(problems) * len(MODES)
    generation_complete = len(records) == total and all(s.get('generation_exit') == '0' for s in status.values())
    grading_complete = generation_complete and all(s.get('judge_exit') == '0' for s in status.values()) and all(
        d['paired_complete_problems'] == d['expected_problems'] for d in datasets.values())
    selected_path = OUT / 'selected-excerpts.json'
    selected = json.loads(selected_path.read_text(encoding='utf-8')) if selected_path.exists() else []
    narrative_path = OUT / 'THOUGHT_OBSERVATIONS.md'
    findings_path = OUT / 'FINDINGS.md'
    excerpts_ready = (bool(selected) and len(selected) == len(set(selected))
                      and all(c in reviewed for c in selected)
                      and narrative_path.exists() and bool(narrative_path.read_text(encoding='utf-8').strip())
                      and findings_path.exists() and bool(findings_path.read_text(encoding='utf-8').strip()))
    summary = {'status': 'complete' if grading_complete and excerpts_ready and all(c['case_id'] in reviewed for c in cases) else 'in_progress',
               'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'expected_measurements': total,
               'generated_measurements': len(records), 'judged_measurements': sum(r['judgment'] is not None for r in records.values()),
               'generation_complete': generation_complete, 'grading_complete': grading_complete,
               'typical_excerpts_ready': excerpts_ready,
               'failed_attempts': len(attempts), 'incomplete_attempts': sum(s.get('open_attempts',0) for s in status.values()),
               'saved_measurements_cost_checked': bool(checks) and not attempts and all(all(v for k,v in c.items() if isinstance(v,bool)) for c in checks),
               'servers': status, 'datasets': datasets,
               'review': {'deferred_until_all_graded': not review_ready, 'scope': 'full_dataset_after_completion',
                          'unique_cases': len(cases), 'reviewed': sum(c['case_id'] in reviewed for c in cases),
                          'remaining': sum(c['case_id'] not in reviewed for c in cases)}}
    save(OUT / 'summary.json', summary)
    save(OUT / 'alignment-checks.json', checks)
    save(OUT / 'failed-attempts.json', attempts)
    report(manifest, problems, summary, cases, key, reviewed)
    print(json.dumps({'generated': len(records), 'judged': summary['judged_measurements'],
                      'review': summary['review'], 'state': {k:v['state'] for k,v in status.items()}}, ensure_ascii=False))


def percent(value):
    return f'{value*100:.1f}%' if value is not None else '—'


def report(manifest, problems, summary, cases, key, reviewed):
    title = '# Qwen3.8-27B 十二方法编程评测'
    if summary['status'] != 'complete':
        message = ('全部题目已完成生成和判题，正在通读思维链并选择典型案例。'
                   if summary['grading_complete'] else
                   '评测正在运行。待全部题目结束后，再通读思维链、挑选少量典型案例并生成最终报告。')
        (OUT / 'REPORT.md').write_text(title + '\n\n' + message +
            '\n\n最终报告保留一张核心结果表、主要结论及少量对照摘录。\n', encoding='utf-8')
        return
    if (OUT / 'style-summary.json').exists():
        # The current report answers the algorithm-level style question; the old
        # Keep quantitative results in the main report and detailed case comparisons in the appendix.
        from style_coding_report import build
        build()
        return
    lcb = {r['mode']: r for r in summary['datasets']['livecodebench']['methods']}
    he = {r['mode']: r for r in summary['datasets']['humanevalplus']['methods']}
    base_flops = lcb['base']['pflops']
    lines = [title, '', (OUT / 'FINDINGS.md').read_text(encoding='utf-8').strip(), '',
             '主评测为固定的 LiveCodeBench 100 题（25 易、30 中、45 难），HumanEval+ 164 题用于验证；'
             '每题每方法按固定种子运行一次，提交一份代码。'
             'C 为 Consilience，SC 为 Self-Certainty。表中耗时和计算量均来自 LiveCodeBench。', '',
             '| 方法 | LiveCodeBench 通过 | HumanEval+ 通过 | 平均耗时（秒/题） | 计算量（Base=1） |',
             '|---|---:|---:|---:|---:|']
    for mode in MODES:
        a, b = lcb[mode], he[mode]
        seconds = f"{a['mean_seconds']:.1f}" if a['mean_seconds'] is not None else '—'
        ratio = f"{a['pflops']/base_flops:.2f}" if a['pflops'] is not None and base_flops else '—'
        lines.append(f"| {NAMES[mode]} | {a['correct']}/{a['common_n']} | {b['correct']}/{b['common_n']} | {seconds} | {ratio} |")
    lines += ['', '## 典型思维链对照', '',
              (OUT / 'THOUGHT_OBSERVATIONS.md').read_text(encoding='utf-8').strip(), '',
              '本次 LiveCodeBench 使用 2025 年 1—4 月 test6 批次的固定子集，并非完整榜单。'
              '同题共用种子，隐藏测试只用于最终判题。计算量为按实际执行图估算的 FLOPs，包含全部内部候选；耗时受共享负载影响。'
              '本次未核查训练数据污染，思维链个例也不代表方法的必然倾向。', '',
              '详细统计与证据保存在 [summary.json](summary.json)、[review-labels.jsonl](review-labels.jsonl) '
              '及 [review-key.json](review-key.json)；原文摘录及反例核验见 '
              '[selected-evidence.json](selected-evidence.json) 与 [report-validation.json](report-validation.json)，'
              '实验版本见 [FROZEN.json](FROZEN.json)。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','batch','record','select'])
    parser.add_argument('--problems', type=int, default=1)
    args = parser.parse_args()
    if args.action in ('batch', 'record', 'select'):
        summary_path = OUT / 'summary.json'
        if not summary_path.exists() or not json.loads(summary_path.read_text(encoding='utf-8'))['grading_complete']:
            print(json.dumps({'review_deferred': True, 'reason': 'Wait until all generation and grading finish.'}))
            return
    if args.action == 'prepare':
        prepare()
    elif args.action == 'batch':
        done = labels()
        pending = [c for c in read_jsonl(OUT / 'review-cases.jsonl') if c['case_id'] not in done]
        ids = list(dict.fromkeys(c['problem_id'] for c in pending))[:args.problems]
        for c in pending:
            if c['problem_id'] in ids:
                print(json.dumps(c, ensure_ascii=False))
    elif args.action == 'select':
        chosen = json.load(sys.stdin)
        available = labels()
        assert isinstance(chosen,list) and chosen and len(chosen) == len(set(chosen)) and all(x in available for x in chosen)
        save(OUT/'selected-excerpts.json',chosen)
        print(json.dumps({'selected':len(chosen)}))
    else:
        cases = {c['case_id']: c for c in read_jsonl(OUT / 'review-cases.jsonl')}
        entries = json.load(sys.stdin)
        for label in entries:
            case = cases[label['case_id']]
            assert label['assessment'] in ('valid','local_error','critical_error','unclear')
            assert label['summary'] and isinstance(label['features'], list)
            for feature in label['features']:
                quote = feature['quote']
                start = case['thinking_text'].find(quote)
                assert quote and start >= 0 and len(quote) <= 1000
                feature.update(start=start,end=start+len(quote))
            label['reviewed_at_utc'] = datetime.now(timezone.utc).isoformat()
        with (OUT / 'review-labels.jsonl').open('a',encoding='utf-8') as sink:
            for label in entries:
                sink.write(json.dumps(label,ensure_ascii=False)+'\n')
        print(json.dumps({'recorded':len(entries)}))


if __name__ == '__main__':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    main()
