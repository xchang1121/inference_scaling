"""Plot audited, complete formal results; never read partial native output."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics

ARMS = ('direct', 'closed_is', 'static_facts', 'static_facts_noise', 'ircot',
        'ircot_noise', 'static_examples', 'static_examples_noise', 'thinkretrieve',
        'thinkretrieve_noise', 'rethink')
LABELS = ('Direct generation', 'No-injection IS', 'Static facts + IS',
          'Static facts: noise', 'IRCoT + IS', 'IRCoT: noise',
          'Static examples + IS', 'Static examples: noise', 'ThinkRetrieve + IS',
          'ThinkRetrieve: noise', 'Rethink only + IS')
SHORT = ('Direct', 'No-inj. IS', 'Facts', 'Facts noise', 'IRCoT', 'IRCoT noise',
         'Examples', 'Example noise', 'ThinkRetrieve', 'TR noise', 'Rethink')


def validate(summary, records):
    """Require the entire cohort and cross-check every plotted aggregate."""
    assert summary['stage'] == 'formal' and summary['audit']['passed']
    assert summary['measurements'] == len(records) == 352
    questions = sorted({r['question_id'] for r in records})
    seeds = sorted({r['seed'] for r in records})
    assert len(questions) == summary['questions'] == 16 and len(seeds) == 2
    assert seeds == sorted(summary['seeds'])
    keyed = {(r['question_id'], r['seed'], r['arm']): r for r in records}
    assert len(keyed) == len(records), 'duplicate measurement'
    assert set(keyed) == {(q, s, a) for q in questions for s in seeds for a in ARMS}
    means = {r['arm']: r for r in summary['summary']}
    assert set(means) == set(ARMS)
    for arm in ARMS:
        rows = [r for r in records if r['arm'] == arm]
        mean = means[arm]
        assert mean['n'] == len(rows) == 32
        for metric in ('f1', 'generation_tokens', 'seconds', 'controller_tokens',
                       'probe_tokens', 'search_tokens', 'answer_tokens'):
            field = metric if metric == 'f1' else 'mean_' + metric
            value = statistics.fmean(r[metric] for r in rows)
            assert math.isclose(mean[field], value, rel_tol=1e-10, abs_tol=1e-10), field
        for field in ('empty', 'truncated', 'thinking_eos'):
            assert mean[field] == sum(r[field] for r in rows), field
        assert math.isclose(mean['mean_generation_tokens'], sum(mean['mean_' + k]
            for k in ('controller_tokens', 'probe_tokens', 'search_tokens', 'answer_tokens')))
        assert math.isclose(mean['latency_ratio_of_means'],
                            mean['mean_seconds'] / means['direct']['mean_seconds'])
    pairs = []
    for question in questions:
        baseline = statistics.fmean(keyed[question, seed, 'closed_is']['f1'] for seed in seeds)
        for arm in ARMS:
            rows = [keyed[question, seed, arm] for seed in seeds]
            score = statistics.fmean(r['f1'] for r in rows)
            pairs.append({'question_id': question, 'arm': arm, 'mean_f1': score,
                          'difference_vs_closed_is': score - baseline,
                          'empty_of_two': sum(r['empty'] for r in rows),
                          'thinking_eos_of_two': sum(r['thinking_eos'] for r in rows),
                          'truncated_of_two': sum(r['truncated'] for r in rows)})
    return means, questions, pairs


def render(summary, records, output, *, title_prefix=''):
    means, questions, pairs = validate(summary, records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.hashsalt': 'literature-frozen-is-20261112', 'savefig.facecolor': 'white'})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, stem):
        fig.savefig(output / (stem + '.png'), dpi=180, metadata={'Software': 'Matplotlib'})
        fig.savefig(output / (stem + '.svg'), metadata={'Date': None, 'Creator': 'Matplotlib'})
        plt.close(fig)

    y = np.arange(len(ARMS))
    colors = ['#6b7280' if arm in ('direct', 'closed_is', 'rethink') else
              '#c5ccd3' if arm.endswith('_noise') else '#186b89' for arm in ARMS]
    fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True,
                             gridspec_kw={'width_ratios': [1.15, 1.6, 1.15, 1]})
    fig.subplots_adjust(left=.19, right=.98, top=.82, bottom=.22, wspace=.24)
    fig.suptitle(title_prefix + 'IRCoT / ThinkRetrieve with frozen IS', x=.19, ha='left',
                 y=.96, fontsize=21, fontweight='bold')
    fig.text(.19, .90, '16 questions × 2 seeds per arm · complete cohort, including empty answers',
             color='#4b5563', fontsize=11)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e5e7eb', linewidth=.7)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    axes[0].set_yticks(y, LABELS)
    axes[0].invert_yaxis()
    f1 = [means[a]['f1'] for a in ARMS]
    bars = axes[0].barh(y, f1, color=colors, height=.61)
    axes[0].bar_label(bars, labels=[f'{v:.1f}' for v in f1], padding=3, fontsize=9)
    axes[0].set_xlim(0, 110)
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].set_title('Mean F1', loc='left', pad=20, fontweight='bold')
    axes[0].set_xlabel('0–100; empty = 0')

    left = np.zeros(len(ARMS))
    components = [('controller_tokens', 'Controller / interim', '#e4a336'),
                  ('probe_tokens', 'Checkpoint', '#a7d2c8'),
                  ('search_tokens', 'IS / direct thinking', '#29718a'),
                  ('answer_tokens', 'Final answer', '#727eab')]
    for field, label, color in components:
        values = np.array([means[a]['mean_' + field] / 1000 for a in ARMS])
        axes[1].barh(y, values, left=left, height=.61, label=label, color=color)
        left += values
    axes[1].set_xlim(0, max(left) * 1.17)
    for i, value in enumerate(left):
        axes[1].text(value + max(left) * .025, i, f'{value:.1f}', va='center', fontsize=9)
    axes[1].set_title('Mean generation cost', loc='left', pad=20, fontweight='bold')
    axes[1].set_xlabel('Thousands of generated tokens')

    latency = [means[a]['latency_ratio_of_means'] for a in ARMS]
    bars = axes[2].barh(y, latency, color=colors, height=.61)
    axes[2].bar_label(bars, labels=[f'{v:.2f}×' for v in latency], padding=3, fontsize=9)
    axes[2].axvline(1, color='#374151', linestyle=':', linewidth=1)
    axes[2].set_xlim(0, max(latency) * 1.3)
    axes[2].set_title('Mean latency / direct', loc='left', pad=20, fontweight='bold')
    axes[2].set_xlabel('Ratio of means; direct = 1')

    empties = [means[a]['empty'] for a in ARMS]
    bars = axes[3].barh(y, empties, color='#cf6d55', height=.61)
    axes[3].bar_label(bars, labels=[str(v) for v in empties], padding=3, fontsize=9)
    eos_counts = np.array([means[a]['thinking_eos'] for a in ARMS])
    axes[3].plot(eos_counts[eos_counts > 0], y[eos_counts > 0], 'D', color='#3d4149',
                 markersize=4, label='EOS during thinking')
    axes[3].set_xlim(0, 35)
    axes[3].set_xticks([0, 8, 16, 24, 32])
    axes[3].set_title('Empty final answers', loc='left', pad=20, fontweight='bold')
    axes[3].set_xlabel('Count / 32 runs')
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in components],
               loc='lower left', bbox_to_anchor=(.19, .115), ncol=4, frameon=False, fontsize=9)
    fig.text(.19, .09, 'Dark diamonds: EOS during thinking (overlaps with empty answers). '
             f"Final truncations: {sum(means[a]['truncated'] for a in ARMS)}.", fontsize=9)
    fig.text(.19, .055, 'Costs include discarded candidates and interim answers. '
             'Latency includes retrieval; excludes loading, index construction and warm-up.', fontsize=9)
    fig.text(.19, .025, 'Shorter runtime caused by an empty answer is not evidence of better reasoning efficiency.',
             fontsize=9, color='#92452f')
    save(fig, 'overview')

    compare = [a for a in ARMS if a != 'closed_is']
    lookup = {(p['question_id'], p['arm']): p for p in pairs}
    values = np.array([[lookup[q, a]['difference_vs_closed_is'] for a in compare] for q in questions])
    fig, ax = plt.subplots(figsize=(15, 11))
    fig.subplots_adjust(left=.12, right=.88, top=.82, bottom=.17)
    fig.suptitle(title_prefix + 'Where do the F1 differences come from?', x=.12, ha='left',
                 y=.96, fontsize=21, fontweight='bold')
    fig.text(.12, .91, 'Each cell: mean of two seeds minus no-injection IS on the same question.', fontsize=11)
    matrix = ax.imshow(values, cmap='RdBu', vmin=-100, vmax=100, aspect='auto')
    row_labels = [q + (' †' if lookup[q, 'closed_is']['empty_of_two'] else '') for q in questions]
    ax.set_yticks(range(len(questions)), row_labels)
    ax.set_xticks(range(len(compare)), [SHORT[ARMS.index(a)] for a in compare], rotation=35, ha='right')
    ax.set_xticks(np.arange(-.5, len(compare), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(questions), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='both', length=0)
    for i, q in enumerate(questions):
        for j, arm in enumerate(compare):
            value = values[i, j]
            marker = '*' if lookup[q, arm]['empty_of_two'] else ''
            ax.text(j, i, f'{value:+.1f}{marker}', ha='center', va='center', fontsize=9,
                    color='white' if abs(value) > 53 else '#1f2937')
    bar = fig.colorbar(matrix, ax=ax, fraction=.025, pad=.025)
    bar.set_label('F1 points relative to no-injection IS')
    fig.text(.12, .07, '* Compared arm has an empty answer; † no-injection IS baseline has an empty answer. '
             'All empty answers stay in the means.', fontsize=10)
    fig.text(.12, .035, 'Blue = higher F1; red = lower F1. This is a descriptive per-question view, '
             'not a significance test.', fontsize=10)
    save(fig, 'question_differences')
    with (output / 'question_differences.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    return {'questions': len(questions), 'measurements': len(records), 'plots': 2,
            'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    records = [json.loads(line) for line in args.answers.read_text(encoding='utf-8').splitlines()]
    receipt = render(summary, records, args.output)
    receipt['input_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (args.summary, args.answers)}
    (args.output / 'PLOT-VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n',
                                                     encoding='utf-8', newline='\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
