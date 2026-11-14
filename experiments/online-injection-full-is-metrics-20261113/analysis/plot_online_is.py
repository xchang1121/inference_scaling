"""Deterministic three-panel metric figure from the audited paired results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def plot(summary_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['audit']['passed'] and summary['measurements'] == 192 and summary['questions'] == 16
    rows = summary['arms']
    assert all(r['n'] == 32 for r in rows)
    labels = ['Direct', 'IS, no retrieval', 'Static facts + IS', 'Static noise + IS', 'Online facts + IS', 'Online noise + IS']
    arms = ['direct', 'closed_is', 'static_facts', 'static_facts_noise', 'online_is', 'online_is_noise']
    assert [r['arm'] for r in rows] == arms
    for r in rows:
        assert math.isclose(r['tokens_relative_direct'], r['mean_generated_tokens']/rows[0]['mean_generated_tokens'])
        assert math.isclose(r['latency_relative_direct'], r['mean_seconds']/rows[0]['mean_seconds'])
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'online-is-metrics-20261113'})
    colors = ['#94a3b8', '#475569', '#0d9488', '#99d5cf', '#2563eb', '#9dbcf4']
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.2), sharey=True)
    y = np.arange(6)
    metrics = [('f1', 'Answer F1 (0–100)'), ('tokens_relative_direct', 'Generated tokens / direct'),
               ('latency_relative_direct', 'Pipeline latency / direct')]
    for ax, (key, title) in zip(axes, metrics, strict=True):
        values = [r[key] for r in rows]
        bars = ax.barh(y, values, color=colors, height=.62)
        for i in (3, 5):
            bars[i].set_hatch('///');bars[i].set_edgecolor('#64748b')
        if key == 'f1':
            errors = [[r['f1']-r['f1_question_bootstrap_95'][0] for r in rows],
                      [r['f1_question_bootstrap_95'][1]-r['f1'] for r in rows]]
            ax.errorbar(values, y, xerr=errors, fmt='none', ecolor='#334155', capsize=3, lw=1)
            xmax = max(r['f1_question_bootstrap_95'][1] for r in rows)
        else:
            xmax = max(values)
            ax.axvline(1, color='#475569', ls=':', lw=1)
        ax.set_xlim(0, xmax*1.27)
        for i, value in enumerate(values):
            xpos = rows[i]['f1_question_bootstrap_95'][1] if key == 'f1' else value
            ax.text(xpos+xmax*.025, i, f'{value:.2f}'+('×' if key != 'f1' else ''), va='center', fontsize=9)
        ax.set_title(title, fontweight='bold', pad=14)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle('SYNTHETIC LAYOUT PREVIEW — NOT EXPERIMENT RESULTS' if summary.get('synthetic') else
                 'IS from the first thought token: online retrieval evaluation', fontsize=15, fontweight='bold', y=.98)
    fig.text(.5, .04, ('Synthetic values used only to check layout\n' if summary.get('synthetic') else
             '16 fixed DEV questions × 2 seeds · 192 fresh runs · identical IS settings\n')+
             'F1 intervals: question bootstrap. Cost includes all candidates/rollouts; latency includes retrieval.',
             ha='center', fontsize=10, color='#475569')
    fig.subplots_adjust(left=.16, right=.98, top=.85, bottom=.19, wspace=.25)
    fig.savefig(output/'overview.png', dpi=180, metadata={'Software': 'online-is-metrics'})
    fig.savefig(output/'overview.svg', metadata={'Date': None, 'Creator': 'online-is-metrics'})
    plt.close(fig)
    fields = ['arm', 'f1', 'mean_generated_tokens', 'tokens_relative_direct', 'mean_seconds', 'latency_relative_direct', 'empty', 'truncated']
    with (output/'overview-data.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore');writer.writeheader();writer.writerows(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();plot(a.summary, a.output)
