"""Render saved descriptive aggregates; never launches model inference.
Usage: python render_charts.py --font /path/to/CJK-font.ttf --output /path/to/output
Requires matplotlib and numpy; use the original pinned environment for exact appearance.
"""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.patches import Patch
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/msyh.ttc'))
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
if not args.font.is_file():parser.error('Provide an installed Chinese font with --font')
fontManager.addfont(str(args.font))
plt.rcParams.update({'font.family':FontProperties(fname=str(args.font)).get_name(),
 'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none'})
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)

STAGES = [('search_seconds', '思考 / IS 搜索（含评分、候选）', '#4477AA'),
          ('retrieval_seconds', '检索与上下文拼接', '#DD9933'),
          ('answer_seconds', '最终答案生成', '#32977E'),
          ('other_seconds', '其他控制开销', '#A5ABB5')]

def plot(rows, name, subtitle, base, cross_run=False):
    n = len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 3.5 + .7*n), sharey=True,
                             gridspec_kw={'width_ratios': [1.0, 1.52, .82]})
    fig.subplots_adjust(left=.16, right=.975, top=.71, bottom=.15, wspace=.21)
    y = np.arange(n)
    colors = ['#8796A8' if r['arm']=='direct' else '#32977E' if r['arm']=='static_facts' else '#4477AA' for r in rows]
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E6E9ED', linewidth=.7)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#BCC3CA')
        ax.tick_params(axis='y', length=0)
    ax = axes[0]
    vals = np.array([r['f1'] for r in rows])
    errors = np.array([[r['f1']-r['f1_ci_low'] for r in rows], [r['f1_ci_high']-r['f1'] for r in rows]])
    ax.barh(y, vals, color=colors, height=.53)
    ax.errorbar(vals, y, xerr=errors, fmt='none', ecolor='#263B50', capsize=3, linewidth=1.25)
    for i, r in enumerate(rows):
        ax.text(r['f1_ci_high']+.35, i, f"{r['f1']:.2f}", va='center', fontsize=11)
    ax.set_yticks(y, [f"{r['label']}\n空答案 {r['empty']}/32" for r in rows], fontsize=11)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_title('F1（0–100）\n横线：问题级 95% bootstrap 区间', loc='left', fontsize=13, pad=19)
    ax.invert_yaxis()

    ax = axes[1]
    left = np.zeros(n)
    for key, _, color in STAGES:
        values = np.array([r[key] for r in rows])
        ax.barh(y, values, left=left, height=.53, color=color)
        if key in ('search_seconds', 'answer_seconds'):
            for i, v in enumerate(values):
                ax.text(left[i]+v/2, i, f'{v:.1f}', ha='center', va='center', color='white', fontsize=10)
        left += values
    for i, r in enumerate(rows):
        ax.text(r['total_seconds']+2, i, f"{r['total_seconds']:.1f}s", va='center', fontsize=11)
        if r['retrieval_seconds'] > .01:
            ax.text(r['search_seconds']+r['retrieval_seconds']/2, i+.38,
                    f"检索/拼接 {r['retrieval_seconds']:.2f}s", ha='center', va='center', fontsize=9, color='#855100')
    ax.set_xlim(0, 190 if cross_run else 148)
    ax.set_title('平均单次时延拆分（秒）\n所有候选、丢弃分支均计入', loc='left', fontsize=13, pad=19)
    ax.set_xlabel('墙钟时间；各色段相加等于总时延')

    ax = axes[2]
    ratios = [r['total_seconds']/base for r in rows]
    ax.barh(y, ratios, height=.53, color=colors)
    ax.axvline(1, color='#5A6877', linestyle='--', linewidth=1.2)
    for i, value in enumerate(ratios):
        ax.text(value+.04, i, f'{value:.2f}×', va='center', fontsize=11)
    ax.set_xlim(0, 2.6 if cross_run else 2.05)
    ax.set_title('总时延 / Base\n'+('跨轮参考，非同轮配对' if cross_run else '同轮自由生成 = 1.00×'),
                 loc='left', fontsize=13, pad=19)
    ax.set_xlabel('各组平均时延之比')
    fig.suptitle('Training-free RAG + IS：质量与时延'+(' · 动态策略扩展' if cross_run else ' · 基础对照'),
                 fontsize=20, fontweight='bold', x=.16, ha='left', y=.96)
    fig.text(.16, .90, subtitle, fontsize=11, color='#46576A')
    fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in STAGES],
               loc='upper left', bbox_to_anchor=(.155,.865), ncol=4, frameon=False, fontsize=10)
    note = (f'Base = {base:.2f}s；'+('本轮未重测 Base，采用 11/13 同题、同种子的自由生成均值。' if cross_run else '同轮 16 道 DEV × 2 种子 × 6 组，共 192 次运行。'))
    fig.text(.16, .075, note, fontsize=10, color='#46576A')
    fig.text(.16, .035, '已查看开发集，非独立测试；F1 ≠ 正确率。冷前缀缓存；不含模型加载、索引初始化、预热。阶段分解不是独立组件因果增量。',
             fontsize=10, color='#46576A')
    for extension in ('png','svg','pdf'):
        fig.savefig(OUT/f'{name}.{extension}', dpi=170, facecolor='white')
    plt.close(fig)

data=json.loads((Path(__file__).resolve().parent/'chart-data.json').read_text(encoding='utf-8'))
core=[r for r in data['rows'] if r['study']=='2026-11-13 core']
smart=[r for r in data['rows'] if r['study']=='2026-11-14 smart']
base=data['base_seconds']
assert len(core)==6 and len(smart)==8
plot(core,'core-f1-stage-latency','Qwen3.8-27B · 原 IS 参数不变 · 事实与等长无关文本对照',base)
plot(smart,'dynamic-f1-stage-latency','同一 16 道 DEV × 2 种子 × 8 组，共 256 次运行 · 与基础对照分轮展示',base,True)
