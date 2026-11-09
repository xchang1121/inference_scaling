"""Export figures from a completed, audited online-injection pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohorts",nargs="+",choices=("techqa","api"),default=["techqa","api"])
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8"))
    if not result["all_saved_measurements_pass_audit"] or any(
            result["datasets"][key]["generated"] != result["datasets"][key]["expected"]
            or not result["datasets"][key]["complete_marker"] or result["exit_codes"][key] != "0"
            for key in args.cohorts):
        raise ValueError("only completed, audited measurements can be plotted")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "svg.fonttype":"none", "savefig.dpi":180})
    titles = {"techqa":"TechQA: answer token F1","api":"Official rules: exact JSON accuracy"}
    cohorts = [(key,titles[key]) for key in args.cohorts]
    width = 7.5*len(cohorts)

    def save(fig, name):
        for extension in ("png","svg"):
            fig.savefig(args.output/f"{name}.{extension}", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1,len(cohorts),figsize=(width,6.2),squeeze=False)
    axes = axes.ravel()
    for ax,(key,title) in zip(axes,cohorts,strict=True):
        cohort = result["datasets"][key]
        cs = cohort["comparisons"]
        points = np.array([c["difference"]*100 for c in cs])
        low = np.array([c["bootstrap_95"][0]*100 for c in cs])
        high = np.array([c["bootstrap_95"][1]*100 for c in cs])
        y = np.arange(len(cs))
        ax.errorbar(points,y,xerr=np.vstack([points-low,high-points]),fmt="o",
                    color="#226a9f",ecolor="#91b4cf",capsize=3,markersize=5)
        ax.axvline(0,color="#777777",lw=1,ls="--")
        ax.set_yticks(y,[c["alternative"] for c in cs])
        ax.invert_yaxis()
        ax.set_title(title+f" (n={len(cohort['common_questions'])})",loc="left",weight="bold")
        ax.set_xlabel("Paired change from closed book (percentage points)")
        ax.grid(axis="x",alpha=.15)
    fig.suptitle("Quality differences and 95% paired bootstrap intervals",weight="bold",y=1.02)
    fig.text(.5,-.02,"5,000 question-level resamples; no multiple-comparison correction. Small pilot, not a leaderboard.",ha="center",color="#555555")
    fig.tight_layout(w_pad=4)
    save(fig,"quality-paired")

    fig, axes = plt.subplots(1,len(cohorts),figsize=(width,6.5),squeeze=False)
    axes = axes.ravel()
    for ax,(key,title) in zip(axes,cohorts,strict=True):
        methods = result["datasets"][key]["methods"]
        baseline = next(m for m in methods if m["arm"] == "closed-book")
        y = np.arange(len(methods))
        ax.barh(y-.17,[m["mean_seconds"]/baseline["mean_seconds"] for m in methods],height=.31,color="#226a9f",label="End-to-end time")
        ax.barh(y+.17,[m["pflops"]/baseline["pflops"] for m in methods],height=.31,color="#d68b38",label="Native graph cost estimate")
        ax.axvline(1,color="#777777",lw=1,ls="--")
        ax.set_yticks(y,[m["arm"] for m in methods])
        ax.invert_yaxis()
        ax.set_xlabel("Ratio to closed book")
        ax.set_title(title,loc="left",weight="bold")
        ax.grid(axis="x",alpha=.15)
    axes[0].legend(loc="lower right",frameon=False)
    fig.suptitle("Latency and computation on the same physical NPU",weight="bold",y=1.02)
    fig.text(.5,-.02,"Graph cost = 2 x parameter count x executed padded token positions; not a hardware FLOPs counter.",ha="center",color="#555555")
    fig.tight_layout(w_pad=4)
    save(fig,"cost-ratios")

    fig, axes = plt.subplots(1,len(cohorts),figsize=(width,6.5),squeeze=False)
    axes = axes.ravel()
    for ax,(key,title) in zip(axes,cohorts,strict=True):
        cohort = result["datasets"][key]
        rows = [r for line in (args.snapshot/f"{key}.jsonl").read_text(encoding="utf-8").splitlines()
                if (r:=json.loads(line)).get("kind") == "measurement" and r["question_id"] in cohort["common_questions"]]
        methods = cohort["methods"]
        for i,method in enumerate(methods):
            selected = [r for r in rows if r["arm"] == method["arm"]]
            pairs = [(r,next((e for e in r["events"] if e.get("injected")),None)) for r in selected]
            injected = [(r,e) for r,e in pairs if e is not None]
            if injected:
                before = np.mean([min(r["thinking_tokens"],e["actual_generated_offset"]) for r,e in injected])
                after = np.mean([max(0,r["thinking_tokens"]-e["actual_generated_offset"]) for r,e in injected])
                ax.barh(i,before,height=.6,color="#91b4cf")
                ax.barh(i,after,left=before,height=.6,color="#d68b38")
            else:
                ax.barh(i,np.mean([r["thinking_tokens"] for r in selected]),height=.6,color="#dddddd")
            ax.text(264,i,f"{len(injected)}/{len(selected)}",va="center",color="#555555",fontsize=8)
        ax.set_yticks(np.arange(len(methods)),[m["arm"] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0,292)
        ax.set_xticks([0,64,128,192,256])
        ax.set_xlabel("Mean sampled thinking tokens (injected subset)")
        ax.set_title(title,loc="left",weight="bold")
        ax.grid(axis="x",alpha=.15)
    fig.legend(handles=[Patch(color="#91b4cf",label="Before first injection"),
        Patch(color="#d68b38",label="After first injection"),Patch(color="#dddddd",label="No injection")],
        loc="lower center",bbox_to_anchor=(.5,-.08),ncol=3,frameon=False)
    fig.suptitle("Where evidence arrives and whether thinking continues",weight="bold",y=1.02)
    fig.text(.5,-.13,"Right labels: injected / total questions. Evidence and forced closing tokens are excluded.",ha="center",color="#555555")
    fig.tight_layout(w_pad=4)
    save(fig,"thinking-position")
    metadata = {"matplotlib":matplotlib.__version__,"numpy":np.__version__,"cohorts":args.cohorts,
                "source_summary_status":result["status"],
                "summary_sha256":hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in args.output.iterdir() if p.suffix in {".png",".svg"}}}
    (args.output/"figures-manifest.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
