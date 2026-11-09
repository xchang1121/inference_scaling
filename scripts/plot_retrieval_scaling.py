"""Deterministic standalone figures for the frozen retrieval/search pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.summary.read_text(encoding="utf-8"))
    rows = {(r["cohort"], r["arm"]): r for r in report["summary"]}
    methods = ("single", "sc_bon", "block_met_sc")
    labels = ("Single long", "SC-BoN (up to 8)", "Block-Met IS (K=2)")
    views = ("closed.none", "static.relevant", "static.unrelated", "dynamic.relevant", "dynamic.unrelated")
    names = ("Closed", "Static\nreal", "Static\nnoise", "Dynamic\nreal", "Dynamic\nnoise")
    colors = ("#9aa7b2", "#248a9e", "#bb6b35")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "svg.hashsalt": "retrieval-scaling-20261109", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), layout="constrained")
    for ri, cohort in enumerate(("heldout", "regression")):
        for ci, (metric, title) in enumerate((("f1_percent", "Mean answer token F1 (%)"), ("native_pflops", "Mean native graph estimate (PFLOPs)"))):
            ax = axes[ri, ci]
            for mi, (method, label, color) in enumerate(zip(methods, labels, colors, strict=True)):
                values = [rows[(cohort, view+"/"+method)][metric] for view in views]
                bars = ax.bar(np.arange(len(views))+(mi-1)*.25, values, .23, color=color, label=label)
                if ci == 0:
                    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
            ax.set_xticks(np.arange(len(views)), names)
            ax.set_title(("Fresh, n=4" if cohort == "heldout" else "Regression, n=4")+" | "+title, loc="left", fontsize=12)
            ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True); ax.margins(y=.22)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Evidence retrieval + inference scaling: feasibility pilot", fontsize=18)
    fig.supxlabel("8192 search/aux generated-token cap + 512 answer tokens; one seed; realized compute differs.", fontsize=11)
    a.output.mkdir(parents=True, exist_ok=True)
    png, svg = a.output/"retrieval-scaling.png", a.output/"retrieval-scaling.svg"
    fig.savefig(png, dpi=160, metadata={"Software": "matplotlib"})
    fig.savefig(svg, metadata={"Date": None, "Creator": "matplotlib"})
    plt.close(fig)
    manifest = {"summary_sha256": hashlib.sha256(a.summary.read_bytes()).hexdigest(),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (png, svg)}}
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
