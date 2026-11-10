"""Publication-style static summary of the frozen fresh-question comparisons."""
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cohort",choices=["fresh","regression","smoke"],default="fresh")
    a=p.parse_args()
    data=json.loads(a.summary.read_text(encoding="utf-8"))
    rows={r["arm"]:r for r in data["summary"] if r["cohort"] == a.cohort}
    if not rows:raise ValueError("the selected cohort has no results")
    arms=["static_legacy_sc","dynamic_legacy_sc","static_anchor_sc","dynamic_anchor_sc","static_verified_is","dynamic_verified_is","static_verified_noise","dynamic_verified_noise"]
    labels=["Static\nlegacy SC","Dynamic\nlegacy SC","Static\nanchored SC","Dynamic\nanchored SC","Static\nverified IS","Dynamic\nverified IS","Static\nnoise IS","Dynamic\nnoise IS"]
    colors=["#9badc2","#54708f","#91baa3","#4b8864","#e8b66b","#b76b19","#d1d1d1","#969696"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.hashsalt":"grounded-is-20261110","axes.spines.top":False,"axes.spines.right":False})
    fig=plt.figure(figsize=(14,9))
    axes=[fig.add_axes([.06,.59,.42,.29]),fig.add_axes([.56,.59,.42,.29]),fig.add_axes([.32,.15,.63,.25])]
    x=np.arange(len(arms))
    for ax,field,title in [(axes[0],"f1","Lexical answer F1 (%)"),(axes[1],"native_pflops","Native graph estimate (PFLOPs / answer)")]:
        values=[rows[k][field]*(100 if field=="f1" else 1) for k in arms]
        bars=ax.bar(x,values,color=colors,width=.76)
        for bar in bars[-2:]:bar.set_hatch("///");bar.set_edgecolor("#666")
        ax.set_xticks(x,labels,rotation=38,ha="right")
        ax.set_title(title,fontsize=11,pad=13)
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        ax.set_ylim(0,max(values)*1.2 if max(values) else 1)
        for xi,v in zip(x,values,strict=True):ax.text(xi,v+max(values)*.02,f"{v:.2f}",ha="center",fontsize=9)
    pairs=[("dynamic_verified_is","dynamic_legacy_sc","Verified dynamic − old dynamic"),
           ("dynamic_verified_is","dynamic_anchor_sc","Verified dynamic − anchored SC"),
           ("dynamic_verified_is","static_verified_is","Verified dynamic − verified static"),
           ("dynamic_verified_is","static_legacy_sc","Verified dynamic − old static"),
           ("dynamic_verified_is","dynamic_verified_noise","Verified dynamic − matched noise"),
           ("dynamic_anchor_sc","dynamic_legacy_sc","Anchored dynamic − old dynamic")]
    contrasts={(r["left"],r["right"]):r for r in data["contrasts"] if r["cohort"] == a.cohort}
    ax=axes[2]
    for i,(left,right,label) in enumerate(pairs):
        row=contrasts[left,right];v=100*row["difference"];low,high=[100*t for t in row["bootstrap_95"]]
        ax.errorbar(v,len(pairs)-1-i,xerr=[[v-low],[high-v]],fmt="o",color="#914a11",capsize=4)
    ax.set_yticks(range(len(pairs)),[x[2] for x in pairs][::-1],fontsize=9)
    ax.axvline(0,color="#555",linewidth=1,linestyle="--")
    ax.grid(axis="x",alpha=.2)
    ax.set_xlabel("F1 difference (percentage points)")
    ax.set_title("Paired question bootstrap: 95% intervals",fontsize=11,pad=13)
    n=rows["dynamic_verified_is"]["questions"]
    seeds=len({s for meta in data["metadata"] for s in meta["seeds"]})
    cohort="TRAIN smoke fixtures" if a.cohort=="smoke" else a.cohort+" questions"
    seed_label="seed" if seeds==1 else "seeds"
    fig.suptitle(f"Grounded retrieval + IS · {n} {cohort} · {seeds} {seed_label}",fontsize=17,y=.96)
    fig.text(.06,.03,"Same 8,192 work-token cap; realized compute differs. Verified IS pays candidate answers and checks inside the cap.\nHatched noise controls keep source wrappers and exact body length. F1 measures overlap, not factual accuracy.\nBootstrap averages seeds within each question; intervals are descriptive and unadjusted.",fontsize=9,color="#444")
    a.output.mkdir(parents=True,exist_ok=True)
    files=[]
    for extension in ["png","svg"]:
        path=a.output/("grounded-retrieval-is."+extension)
        fig.savefig(path,dpi=160,metadata={"Date":None} if extension=="svg" else {"Software":"Matplotlib"})
        blob=path.read_bytes();files.append({"path":path.name,"bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest()})
    plt.close(fig)
    (a.output/"manifest.json").write_text(json.dumps(files,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(files))


if __name__ == "__main__":main()
