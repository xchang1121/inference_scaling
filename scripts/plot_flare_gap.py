"""Standalone research figure from the audited summary, without model access."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    methods = ["closed", "static", "legacy_sc", "step_sc", "flare_gap", "flare_gap_rebuild"]
    labels = ["Closed", "Static RAG", "SC inline", "SC step\nrebuild", "FLARE step\ninline", "FLARE step\nrebuild"]
    cohorts = [c for c in ("heldout", "regression") if c in data["cohorts"]]
    fig, axes = plt.subplots(len(cohorts), 3, figsize=(15, 4*len(cohorts)), squeeze=False, layout="constrained")
    x = np.arange(len(methods)); width = .36
    for ri, cohort in enumerate(cohorts):
        groups = {g["arm"]: g for g in data["cohorts"][cohort]["groups"]}
        n = len(data["cohorts"][cohort]["complete_questions"])
        for ci, (metric, title, ymax) in enumerate([
            ("f1", "Generative token overlap F1 (%)", None),
            ("empty_answer", "Empty final answers (count)", n),
            ("eos_inside_thinking", "EOS inside thinking (count)", n)]):
            ax = axes[ri, ci]
            for content, offset, color in (("relevant", -width/2, "#2166AC"), ("unrelated", width/2, "#D87824")):
                values = []
                for method in methods:
                    arm = "closed.none" if method == "closed" else method+"."+content
                    value = groups[arm][metric]
                    values.append(100*value if metric == "f1" and value is not None else value or 0)
                ax.bar(x+offset, values, width, label=content.capitalize(), color=color, edgecolor="white")
                if metric != "f1":
                    for xpos, value in zip(x+offset, values):
                        if value: ax.text(xpos, value+.08, str(int(value)), ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, labels)
            ax.set_title(f"{cohort.capitalize()} (n={n}) — {title}", loc="left", fontsize=10, weight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
            if ymax is not None: ax.set_ylim(0, ymax+1)
            if ri == 0 and ci == 0: ax.legend(frameon=False)
    fig.suptitle("Step-boundary retrieval: fixed 1024-token work cap (previews and checkers charged)", fontsize=14, weight="bold")
    fig.savefig(args.output/"comparison.png", dpi=180)
    fig.savefig(args.output/"comparison.svg")
    plt.close(fig)
    print(json.dumps({"figures": ["comparison.png", "comparison.svg"], "source": str(args.summary)}))


if __name__ == "__main__": main()
