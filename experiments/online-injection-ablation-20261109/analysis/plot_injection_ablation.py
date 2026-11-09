"""Render completed cohorts of the matched-content injection follow-up."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from summarize_injection_ablation import bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--grades",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--budgets",nargs="+",type=int,default=[256,1024,4096])
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8"))
    if not result["all_saved_measurements_pass_audit"] or any(
        result["groups"][str(b)]["generated"] != result["groups"][str(b)]["expected"]
        or len(result["groups"][str(b)]["common_questions"]) != 8 for b in args.budgets):
        raise ValueError("only completed, audited budget cohorts can be plotted")
    grades = [json.loads(line) for line in args.grades.read_text(encoding="utf-8").splitlines()]
    indexed = {(r["budget"],r["question_id"],r["arm"]):r for r in grades}
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"svg.fonttype":"none","svg.hashsalt":"injection-ablation-20261109","savefig.dpi":180})
    width = 8.2*len(args.budgets)
    blue,orange,grey = "#2674a6","#ce8935","#d0d3d5"

    def save(fig,name):
        for ext in ("png","svg"):
            metadata={"Date":None} if ext == "svg" else None
            fig.savefig(args.output/f"{name}.{ext}",bbox_inches="tight",facecolor="white",metadata=metadata)
        plt.close(fig)

    fig,axes = plt.subplots(1,len(args.budgets),figsize=(width,7.0),squeeze=False)
    for ax,budget in zip(axes.ravel(),args.budgets,strict=True):
        methods = [m for m in result["groups"][str(budget)]["methods"] if m["arm"] != "closed"]
        for i,m in enumerate(methods):
            count = m["injected_count"]
            value = m["immediate_end_given_injected"]
            color = orange if "unrelated" in m["arm"] else grey if "wrapper-only" in m["arm"] else blue
            if value is not None:ax.barh(i,value*100,height=.65,color=color)
            ax.text(103,i,f"{m['immediate_end_count']}/{count}" if count else "no injection",va="center",fontsize=8)
        ax.set_yticks(range(len(methods)),[m["arm"] for m in methods]);ax.invert_yaxis()
        ax.set_xlim(0,126);ax.set_xticks([0,25,50,75,100]);ax.grid(axis="x",alpha=.15)
        ax.set_title(f"Thinking budget {budget:,} (same 8 questions)",loc="left",weight="bold")
        ax.set_xlabel("First model token after evidence is </think> (%)")
    fig.suptitle("Immediate thinking termination with matched-length controls",weight="bold",y=1.02)
    fig.legend(handles=[Patch(color=blue,label="Relevant body"),Patch(color=orange,label="Unrelated body"),
        Patch(color=grey,label="Wrapper and source labels only")],loc="lower center",bbox_to_anchor=(.5,-.055),ncol=3,frameon=False)
    fig.text(.5,-.10,"Right labels: immediate ends / injected cases. Static RAG uses the first assistant token. Small mechanism pilot.",ha="center",color="#555555")
    fig.tight_layout(w_pad=4);save(fig,"immediate-end")

    stop_colors=["#2674a6","#d0d3d5","#c45b48","#8a79a7"]
    stop_labels=["Natural </think>","Forced at thinking cap","EOS inside thinking","Other stop"]
    fig,axes = plt.subplots(1,len(args.budgets),figsize=(width,7.4),squeeze=False)
    for ax,budget in zip(axes.ravel(),args.budgets,strict=True):
        methods=result["groups"][str(budget)]["methods"]
        for i,m in enumerate(methods):
            counts=[m["natural_end_count"],m["forced_end_count"],m["eos_before_thinking_end_count"]]
            counts.append(m["n"]-sum(counts));left=0
            for count,color in zip(counts,stop_colors,strict=True):
                if count:ax.barh(i,count,left=left,height=.65,color=color)
                left+=count
            ax.text(8.2,i,f"{m['empty_answers']}/8 empty",va="center",fontsize=8)
        ax.set_xlim(0,10);ax.set_xticks(range(0,9,2));ax.grid(axis="x",alpha=.15)
        ax.set_yticks(range(len(methods)),[m["arm"] for m in methods]);ax.invert_yaxis()
        ax.set_title(f"Thinking budget {budget:,}",loc="left",weight="bold")
        ax.set_xlabel("Questions by thinking-segment outcome")
    fig.suptitle("Avoiding immediate </think> can still produce an empty answer",weight="bold",y=1.02)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(stop_colors,stop_labels,strict=True)],
               loc="lower center",bbox_to_anchor=(.5,-.055),ncol=4,frameon=False)
    fig.text(.5,-.10,"EOS before </think> terminates generation without a final answer. Empty answers also include EOS just after </think>.",ha="center",color="#555555")
    fig.tight_layout(w_pad=4);save(fig,"stop-outcomes")

    longest = max((m["mean_injection_offset"]+m["mean_post_injection_thinking"] if m["injected_count"] else m["mean_thinking_tokens"])
                  for b in args.budgets for m in result["groups"][str(b)]["methods"])
    thinking_axis = max(256,math.ceil(longest/256)*256)
    fig,axes = plt.subplots(1,len(args.budgets),figsize=(width,7.4),squeeze=False)
    for ax,budget in zip(axes.ravel(),args.budgets,strict=True):
        group = result["groups"][str(budget)];methods=group["methods"]
        for i,m in enumerate(methods):
            rows = [indexed[budget,q,m["arm"]] for q in group["common_questions"]]
            injected = [r for r in rows if r["injected"]]
            if injected:
                before = np.mean([min(r["thinking_tokens"],r["injection_offset"]) for r in injected])
                after = np.mean([r["post_injection_thinking"] for r in injected])
                ax.barh(i,before,height=.65,color="#91b4cf")
                ax.barh(i,after,left=before,height=.65,color=orange)
            else:ax.barh(i,np.mean([r["thinking_tokens"] for r in rows]),height=.65,color=grey)
            ax.text(thinking_axis*1.035,i,f"{len(injected)}/{len(rows)}",va="center",fontsize=8)
        ax.set_xlim(0,thinking_axis*1.2);ax.set_xticks([thinking_axis*v/4 for v in range(5)])
        ax.set_yticks(range(len(methods)),[m["arm"] for m in methods]);ax.invert_yaxis();ax.grid(axis="x",alpha=.15)
        ax.set_title(f"Thinking budget {budget:,}",loc="left",weight="bold")
        ax.set_xlabel("Sampled thinking tokens (injected subset when present)")
    fig.suptitle("How much thinking remains after the intervention",weight="bold",y=1.02)
    fig.legend(handles=[Patch(color="#91b4cf",label="Before injection"),Patch(color=orange,label="After injection"),
        Patch(color=grey,label="No injection")],loc="lower center",bbox_to_anchor=(.5,-.055),ncol=3,frameon=False)
    fig.text(.5,-.10,"Shared token axis across budgets. Right labels: injected / total. Forced endings are distinguished in the report.",ha="center",color="#555555")
    fig.tight_layout(w_pad=4);save(fig,"thinking-length")

    fig,axes = plt.subplots(1,len(args.budgets),figsize=(width,7.0),squeeze=False)
    for ax,budget in zip(axes.ravel(),args.budgets,strict=True):
        group = result["groups"][str(budget)];methods=[m for m in group["methods"] if m["arm"] != "closed"]
        changes=[]
        for m in methods:
            changes.append(bootstrap([indexed[budget,q,m["arm"]]["f1"]-indexed[budget,q,"closed"]["f1"] for q in group["common_questions"]]))
        points=np.array([v["difference"]*100 for v in changes]);low=np.array([v["bootstrap_95"][0]*100 for v in changes]);high=np.array([v["bootstrap_95"][1]*100 for v in changes])
        ax.errorbar(points,range(len(methods)),xerr=np.vstack([points-low,high-points]),fmt="o",capsize=3,color=blue,ecolor="#91b4cf")
        ax.axvline(0,color="#777777",ls="--",lw=1)
        ax.set_yticks(range(len(methods)),[m["arm"] for m in methods]);ax.invert_yaxis();ax.grid(axis="x",alpha=.15)
        base=next(m for m in group["methods"] if m["arm"]=="closed")
        ax.set_title(f"Budget {budget:,}; closed F1 {base['score']*100:.1f}%",loc="left",weight="bold")
        ax.set_xlabel("Paired answer F1 change from closed book (pp)")
    fig.suptitle("Answer quality under the same thinking and answer caps",weight="bold",y=1.02)
    fig.text(.5,-.025,"8 question-level pairs; 5,000 bootstrap resamples; no multiple-comparison correction. Budgets repeat the same questions.",ha="center",color="#555555")
    fig.tight_layout(w_pad=4);save(fig,"quality")
    metadata={"matplotlib":matplotlib.__version__,"numpy":np.__version__,"budgets":args.budgets,
        "summary_sha256":hashlib.sha256(args.summary.read_bytes()).hexdigest(),"grades_sha256":hashlib.sha256(args.grades.read_bytes()).hexdigest(),
        "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.suffix in {".png",".svg"}}}
    (args.output/"figures-manifest.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
