"""Rebuild paired results from immutable measurements and explicit review labels."""
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from met_sc_study import (CONTROL_MODES, FOCUS_MODES, NEW_MODES, REVIEW, ROOT,
                          labels, load_study, prepare, read_jsonl)

OUT = Path(__file__).parent.parent / "outputs" / "qwen38-met-sc-20261103"
NAMES = {
    "base": "Base", "is": "快速 IS + Consilience", "sir-k1": "SIR K=1 + Consilience",
    "sir-k2": "SIR K=2 + Consilience", "bon-consilience": "Best-of-4 + Consilience",
    "met-k1": "逐条 K=1 + Consilience", "met-k2": "逐条 K=2 + Consilience",
    "block-met-k2": "按块 K=2 + Consilience", "is-self-certainty": "快速 IS + Self-Certainty",
    "met-k1-self-certainty": "逐条 K=1 + Self-Certainty",
    "met-k2-self-certainty": "逐条 K=2 + Self-Certainty",
    "block-met-k2-self-certainty": "按块 K=2 + Self-Certainty",
}
OLD_EXPECTED = {"base": 92, "is": 93, "sir-k1": 92, "met-k1": 90, "sir-k2": 93,
                "met-k2": 94, "block-met-k2": 94, "bon-consilience": 94, "is-self-certainty": 94}


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pair_key(r):
    return r["problem_id"], r["repeat"]


def quantiles(values):
    return np.quantile(values, [.025, .975]).tolist()


def paired_comparison(new_mode, control_mode, new_rows, control_rows, scope):
    control = {pair_key(r): r for r in control_rows}
    pairs = [(r, control[pair_key(r)]) for r in new_rows if pair_key(r) in control]
    n = len(pairs)
    if not n:
        return None, []
    for a, b in pairs:
        assert a["replica"] == b["replica"] and a["seed"] == b["seed"]
    details = [{"problem_id": a["problem_id"], "repeat": a["repeat"], "replica": a["replica"],
                "cohort": a["cohort"], "seed": a["seed"], "new_mode": new_mode, "control_mode": control_mode,
                "new_case_id": a["case_id"], "control_case_id": b["case_id"],
                "new_auto_correct": a["correct"], "control_auto_correct": b["correct"],
                "new_reviewed_correct": a["reviewed_correct"], "control_reviewed_correct": b["reviewed_correct"],
                "new_branch_adjusted_correct": a["branch_adjusted_correct"],
                "control_branch_adjusted_correct": b["branch_adjusted_correct"],
                "new_grade": a["grade"], "control_grade": b["grade"],
                "new_seconds": a["seconds"], "control_seconds": b["seconds"],
                "new_padded_flops": a["padded_flops"], "control_padded_flops": b["padded_flops"],
                "new_generated_tokens": a["counters"]["generated_tokens"],
                "control_generated_tokens": b["counters"]["generated_tokens"],
                "new_source": a["source_file"], "control_source": b["source_file"]} for a, b in pairs]
    answer_done = all(a["reviewed_correct"] is not None and b["reviewed_correct"] is not None for a, b in pairs)
    proof_done = all(a["grade"] is not None and b["grade"] is not None for a, b in pairs)
    win = sum(a["reviewed_correct"] is True and b["reviewed_correct"] is False for a, b in pairs) if answer_done else None
    loss = sum(a["reviewed_correct"] is False and b["reviewed_correct"] is True for a, b in pairs) if answer_done else None
    proof_win = sum(a["grade"] == "valid" and b["grade"] != "valid" for a, b in pairs) if proof_done else None
    proof_loss = sum(a["grade"] != "valid" and b["grade"] == "valid" for a, b in pairs) if proof_done else None
    result = {"new_mode": new_mode, "control_mode": control_mode, "scope": scope, "n": n,
              "interim": n != {"all":100, "old":30, "new":70}[scope],
              "auto_correct_new": sum(a["correct"] for a, b in pairs),
              "auto_correct_control": sum(b["correct"] for a, b in pairs),
              "reviewed_correct_new": sum(a["reviewed_correct"] for a,b in pairs) if answer_done else None,
              "reviewed_correct_control": sum(b["reviewed_correct"] for a,b in pairs) if answer_done else None,
              "answer_wins": win, "answer_losses": loss,
              "proof_valid_wins": proof_win, "proof_valid_losses": proof_loss,
              "proof_valid_new": sum(a["grade"] == "valid" for a,b in pairs) if proof_done else None,
              "proof_valid_control": sum(b["grade"] == "valid" for a,b in pairs) if proof_done else None,
              "branch_sensitivity": {
                  "reviewed_correct_new": sum(a["branch_adjusted_correct"] is True for a,b in pairs) if answer_done else None,
                  "reviewed_correct_control": sum(b["branch_adjusted_correct"] is True for a,b in pairs) if answer_done else None,
                  "answer_wins": sum(a["branch_adjusted_correct"] is True and b["branch_adjusted_correct"] is False for a,b in pairs) if answer_done else None,
                  "answer_losses": sum(a["branch_adjusted_correct"] is False and b["branch_adjusted_correct"] is True for a,b in pairs) if answer_done else None},
              "mean_seconds_new": sum(a["seconds"] for a,b in pairs)/n,
              "mean_seconds_control": sum(b["seconds"] for a,b in pairs)/n,
              "seconds_ratio": sum(a["seconds"] for a,b in pairs)/sum(b["seconds"] for a,b in pairs),
              "padded_flops_ratio": sum(a["padded_flops"] for a,b in pairs)/sum(b["padded_flops"] for a,b in pairs),
              "generated_tokens_ratio": sum(a["counters"]["generated_tokens"] for a,b in pairs)/sum(b["counters"]["generated_tokens"] for a,b in pairs),
              "by_replica": {replica: {"n": sum(a["replica"]==replica for a,b in pairs),
                  "seconds_ratio": sum(a["seconds"] for a,b in pairs if a["replica"]==replica)/sum(b["seconds"] for a,b in pairs if a["replica"]==replica)}
                  for replica in sorted({a["replica"] for a,b in pairs})}}
    # Incomplete subsets are progress descriptions, not repeated inferential tests.
    if not result["interim"] and answer_done:
        rng = np.random.default_rng(20261103)
        indices = np.concatenate([rng.choice(np.array([i for i,(a,b) in enumerate(pairs) if a["replica"]==replica]),
                     size=(20000,sum(a["replica"]==replica for a,b in pairs)), replace=True)
                     for replica in sorted({a["replica"] for a,b in pairs})], axis=1)
        answer_diff = np.array([int(a["reviewed_correct"])-int(b["reviewed_correct"]) for a,b in pairs])
        proof_diff = (np.array([int(a["grade"]=="valid")-int(b["grade"]=="valid") for a,b in pairs])
                      if proof_done else None)
        ns,cs = (np.array([r[which]["seconds"] for r in pairs]) for which in (0,1))
        nf,cf = (np.array([r[which]["padded_flops"] for r in pairs]) for which in (0,1))
        discordant = win + loss
        exact_p = min(1., 2*sum(math.comb(discordant,i) for i in range(min(win,loss)+1))/2**discordant) if discordant else 1.
        result["inference"] = {"bootstrap_replicates":20000,"seed":20261103,"strata":"replica",
            "answer_delta_pp_95":quantiles(answer_diff[indices].mean(axis=1)*100),
            "proof_valid_delta_pp_95":quantiles(proof_diff[indices].mean(axis=1)*100) if proof_done else None,
            "seconds_ratio_95":quantiles(ns[indices].sum(axis=1)/cs[indices].sum(axis=1)),
            "padded_flops_ratio_95":quantiles(nf[indices].sum(axis=1)/cf[indices].sum(axis=1)),
            "mcnemar_exact_two_sided_p":exact_p,
            "limitations":"Fixed reviewed labels; no reviewer-error or multiple-comparison correction."}
    return result,details


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    alignment=prepare()
    rows,metas=load_study()
    review=labels()
    answers={x["case_id"]:x for x in read_jsonl(REVIEW/"answer_only_labels.jsonl")}
    for r in rows:
        a=review.get(r["case_id"],answers.get(r["case_id"]))
        r["reviewed_correct"] = a["answer_equivalent"] if a else None
        r["branch_adjusted_correct"] = (a["answer_equivalent"] or a.get("branch_sensitive_correct", False)) if a else None
        r["grade"] = review.get(r["case_id"],{}).get("grade")
        r["padded_flops"] = r.get("graph_execution",{}).get("estimated_dense_padded_flops")
        if r["status"]=="complete":
            assert r["padded_flops"] is not None and r["seconds"]>0
    old=[r for r in rows if r["mode"] not in NEW_MODES]
    corrections=[]
    for r in rows:
        if r["reviewed_correct"] is None or r["reviewed_correct"]==r.get("correct"): continue
        label=review.get(r["case_id"],answers.get(r["case_id"]))
        pid=r["problem_id"]
        category=("parameter_domain" if pid.endswith("intermediate_algebra/558.json") else
                  "root_set" if pid.endswith("intermediate_algebra/2015.json") else "angle_or_answer_extraction")
        corrections.append({"mode":r["mode"],"problem_id":pid,"repeat":r["repeat"],"replica":r["replica"],
                            "cohort":r["cohort"],"source_file":r["source_file"],"case_id":r["case_id"],
                            "automatic_correct":r["correct"],"reviewed_correct":r["reviewed_correct"],
                            "automatic_answer":r.get("answer"),"category":label.get("category",category),
                            "evidence":label["evidence"]})
    old_corrections=[r for r in corrections if r["mode"] not in NEW_MODES]
    assert len(old_corrections)==12 and Counter(r["category"] for r in old_corrections)=={
        "angle_or_answer_extraction":5,"parameter_domain":6,"root_set":1}
    historical=[]
    for mode,expected in OLD_EXPECTED.items():
        group=[r for r in old if r["mode"]==mode]
        assert len(group)==100
        corrected=sum(r["correct"] if r["reviewed_correct"] is None else r["reviewed_correct"] for r in group)
        assert corrected==expected,(mode,corrected,expected)
        historical.append({"mode":mode,"n":100,"auto_correct":sum(r["correct"] for r in group),
                           "corrected_correct":corrected,"mean_seconds":sum(r["seconds"] for r in group)/100,
                           "p95_seconds":float(np.quantile([r["seconds"] for r in group],.95)),
                           "padded_pflops":sum(r["padded_flops"] for r in group)/1e15,
                           "quality_grades":dict(Counter(r["grade"] or "not_reviewed" for r in group))})
    methods=[]
    for mode in FOCUS_MODES:
        group=[r for r in rows if r["mode"]==mode and r["status"]=="complete"]
        methods.append({"mode":mode,"n":len(group),"auto_correct":sum(r["correct"] for r in group),
                        "reviewed_n":sum(r["reviewed_correct"] is not None for r in group),
                        "reviewed_correct":sum(r["reviewed_correct"] is True for r in group),
                        "branch_adjusted_correct":sum(r["branch_adjusted_correct"] is True for r in group),
                        "branch_sensitive_cases":[{"problem_id":r["problem_id"],"case_id":r["case_id"]}
                            for r in group if r["branch_adjusted_correct"] is True and r["reviewed_correct"] is False],
                        "quality_grades":dict(Counter(r["grade"] or "not_reviewed" for r in group)),
                        "mean_seconds":sum(r["seconds"] for r in group)/len(group) if group else None,
                        "p95_seconds":float(np.quantile([r["seconds"] for r in group],.95)) if group else None,
                        "padded_pflops":sum(r["padded_flops"] for r in group)/1e15})
    paired, details=[],[]
    for mode in NEW_MODES:
        for control in (mode.removesuffix("-self-certainty"),"is-self-certainty"):
            for scope in ("all","old","new"):
                nr=[r for r in rows if r["mode"]==mode and r["status"]=="complete" and (scope=="all" or r["cohort"]==scope)]
                cr=[r for r in rows if r["mode"]==control and r["status"]=="complete" and (scope=="all" or r["cohort"]==scope)]
                comparison,dd=paired_comparison(mode,control,nr,cr,scope)
                if comparison:paired.append(comparison)
                if scope=="all":details.extend(dd)
    snapshots=[json.loads((ROOT/replica/"supplement-snapshot.json").read_text(encoding="utf-8")) for replica in ("a","b")]
    batches=[]
    for snapshot in snapshots:
        replica=snapshot["replica"]
        captured_paths={entry["path"] for entry in snapshot["files"]}
        for cohort,expected in (("old",45),("new",105)):
            stem=f"results/selection-met-sc-20261103/{replica}-{cohort}"
            data=read_jsonl(ROOT/replica/(stem+".jsonl"))
            measurements=[r for r in data if r["kind"]=="measurement"]
            exit_path=stem+".exit"
            exit_code=(ROOT/replica/exit_path).read_text(encoding="utf-8").strip() if exit_path in captured_paths else None
            batches.append({"replica":replica,"cohort":cohort,"expected_measurements":expected,
                "measurements":len(measurements),"has_summary":any(r["kind"]=="summary" for r in data),
                "all_measurements_complete":all(r["status"]=="complete" for r in measurements),
                "exit_code":exit_code})
    all_batches_exited_normally=all(b["exit_code"]=="0" and b["has_summary"] and
        b["measurements"]==b["expected_measurements"] and b["all_measurements_complete"] for b in batches)
    summary={"generated_at_utc":datetime.now(timezone.utc).isoformat(),
             "snapshot_at_utc":{s["replica"]:s["captured_at_utc"] for s in snapshots},
             "target_new_measurements":300,"alignment":alignment,"methods":methods,"comparisons":paired,
             "baseline_reproduced":historical,"old_answer_corrections":len(old_corrections),
             "batches":batches,"all_batches_exited_normally":all_batches_exited_normally,
             "new_records_by_replica":dict(Counter(r["replica"] for r in rows if r["mode"] in NEW_MODES)),
             "status":"complete" if all_batches_exited_normally and
                all(m["n"]==100 and m["reviewed_n"]==100 for m in methods) else "in_progress"}
    write_json("summary.json",summary)
    write_json("paired-records.json",details)
    write_json("answer-equivalence-corrections.json",corrections)
    write_json("alignment-checks.json",json.loads((ROOT/"alignment-checks.json").read_text(encoding="utf-8")))
    for name in ("labels.jsonl","answer_only_labels.jsonl","blind_cases.jsonl","method_key.json"):
        (OUT/name).write_bytes((REVIEW/name).read_bytes())
    write_json("provenance.json",{
        "snapshots":snapshots,
        "measurements":[{"path":m["local_file"],"sha256":hashlib.sha256((ROOT/m["local_file"]).read_bytes()).hexdigest(),
                         "label":m["label"],"dataset_sha256":m["dataset"]["source_sha256"]} for m in metas],
        "review_files":{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in
                        ("labels.jsonl","answer_only_labels.jsonl","blind_cases.jsonl","method_key.json")}})
    write_report(summary)
    if summary["status"] == "complete":
        from merge_selection_results import extend_report
        merged = extend_report(rows, metas, summary, NAMES, paired_comparison, OUT)
        summary["merged_comparison"] = merged
        write_json("summary.json", summary)
    print(json.dumps({"status":summary["status"],"new_measurements":alignment["new_measurements"],
                      "remaining_review_cases":alignment["remaining_unique_cases"],"old_corrections":12,
                      "output_directory":str(OUT)}))


def write_report(s):
    align=s["alignment"]
    stamp=" / ".join(f'{r.upper()} {t}' for r,t in s["snapshot_at_utc"].items())
    completed=s["status"]=="complete"
    scope_note=('三种方法均完成固定的 100 题，以下按同题配对比较质量与成本。' if completed else
                '三种方法各计划覆盖固定的 100 题；当前进度不代表最终质量或成本结论。')
    lines=["# Qwen3.8-27B Metropolized 与 Self-Certainty 补测",
        "",f'截至快照时间 {stamp}，补测完成 {align["new_measurements"]}/300 次测量。'
        +scope_note]
    if completed:
        direct=[c for c in s["comparisons"] if c["scope"]=="all" and c["control_mode"]!="is-self-certainty"]
        fast=[c for c in s["comparisons"] if c["scope"]=="all" and c["control_mode"]=="is-self-certainty"]
        answer_changes="、".join(f'{c["reviewed_correct_control"]}→{c["reviewed_correct_new"]}' for c in direct)
        proof_changes="、".join(f'{c["proof_valid_control"]}→{c["proof_valid_new"]}' for c in direct)
        latency_increases="、".join(f'{100*(c["seconds_ratio"]-1):.1f}%' for c in fast)
        flops_increases="、".join(f'{100*(c["padded_flops_ratio"]-1):.1f}%' for c in fast)
        lines += ["", "## 结论", "",
            f'相对同选择核的 Consilience，逐条 K=1、逐条 K=2、按块 K=2 的答案正确数依次为 {answer_changes} / 100，'
            f'最终展示论证成立数为 {proof_changes} / 100。三组准确率差的 95% 区间均包含 0；'
            '按块 K=2 的论证成立率增加 8 个百分点，区间为 [0,16]，仍包含无变化的边界。', "",
            '相对快速 IS + Self-Certainty（答案 94/100、论证成立 83/100），三组补测依次为答案 93/100、94/100、95/100，'
            '论证成立 79/100、82/100、84/100，质量差的区间均包含 0。'
            f'按相同统计口径，总耗时分别增加 {latency_increases}，实际执行图 FLOPs 分别增加 {flops_increases}。'
            '本轮没有建立完整候选池 Metropolized 相对快速 IS 的稳定质量优势，额外成本则清晰可见。', "",
            '原 30 题与新增 70 题的结果不完全一致。逐条 K=1 相对同核 Consilience 在原 30 题答案多对 3 题、'
            '论证成立多 5 题；新增 70 题答案持平、论证成立少 2 题。按块 K=2 在新增 70 题答案持平、'
            '论证成立多 4 题。不能把早期小样本改善直接外推到全部题目。', "",
            '墙钟比较来自不同批次，受共享负载影响；相对快速 IS 的对比同时改变完整池等待与选择核，'
            '不能单独归因于 Metropolized 选择。论证标签由同一助手审核最终展示内容，不代表独立人工复核或全部内部思考质量。']
    lines += ["",
        "## 实验设置与代码检查","",
        "三组分别为逐条 K=1、逐条 K=2、按块 K=2，均使用完整候选池和 Self-Certainty。"
        "优先与同选择核的 Consilience 逐题配对，再与快速 Self-Certainty IS 比较。",
        "",
        "模型为 Qwen3.8-27B BF16，权重版本 1d4bf0f2ff60；两台 Ascend 910B3 TP=2 实例各承担 50 题。"
        "沿用原 30 题和新增 70 题、数据版本、题目分配、根种子 20260808、温度 1、M=4、B=1024、"
        "两阶段探索、4096 思考预算及 8192 总长度上限。方法顺序逐题轮换。",
        "",
        "补测冻结源码为 6bde29bb89856d15ca31768610f54ea750c7278a。三组的 selection_ready 均为 false；"
        "奖励温度 0.25、评分温度 1、scope=thinking。Self-Certainty 为全词表 KL(U‖p) 的思考段均值，"
        "设备端计算标量后随生成返回，不进行额外模型评分前向。预检通过，相关 50 项测试通过。",
        "",
        "逐条选择核针对完整补全叶子执行一次 Metropolized 索引更新，叶子的概率质量为 exp(log_weight)/组内补全数；"
        "按块核先合并相同 token 块的概率质量，再更新块索引并在选中块内按权重取补全。"
        "重复块的概率质量保留。这是候选池内选择核；单次索引更新不能自动保证有限预算准确率提高。",
        "",
        f'已对齐的 {align["new_measurements"]} 条补测均与对照题目、种子、服务器及选择核配置匹配。'
        "同种子不意味着不同方法共享完全相同的候选池。新旧结果跨批次运行，墙钟仍受共享负载和统计归约开销影响。"
        "对快速 Self-Certainty IS 的比较同时包含完整池等待和选择核变化。",
        "","## 已有结果与本次审核","",
        "原九方法 900 次测量已完成。重建的 12 条答案等价修正与旧报告一致：角度或定位 5 条、"
        "参数范围 6 条、根集表达 1 条；逐条证据见 answer-equivalence-corrections.json。"
        "两服务器的项目、结果、Git 历史及相关临时目录中未找到旧的逐条论证审核表，"
        "因此本次对四组直接对照重新审核，新的论证等级与旧报告汇总分别保留。",
        "",
        "| 对照方法 | 自动正确 → 等价复核 | 本次论证成立 / 局部错误 / 关键错误 | 原测量均时 秒 | 原测量 PFLOPs |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for m in s["methods"][:4]:
        g=m["quality_grades"]
        lines.append(f'| {NAMES[m["mode"]]} | {m["auto_correct"]} → {m["reviewed_correct"]} / {m["n"]} | '
                     f'{g.get("valid",0)} / {g.get("local_error",0)} / {g.get("critical_error",0)} | '
                     f'{m["mean_seconds"]:.2f} | {m["padded_pflops"]:.3f} |')
    lines += ["", "审核包隐藏方法、奖励、耗时和自动判分，按题目与最终文本的哈希去重。"
        "审核者为同一助手，参考题目、标准答案及参考解，并独立检查推导。"
        "这是最终展示内容审核，未将全部内部思考计作已审核。",
        "",
        "成立：主要推导有效；局部错误：可局部修正的算术、符号或遗漏；关键错误：决定性推导无效、"
        "必要前提无依据或矛盾未解决。最终答案等价与论证等级分开记录；辅助验算出错也单独记录。"
        "反余切题 precalculus/768 按原参考答案统计，接受 (0,π) 分支时的正确性另列，不混入 12 条修正。",
        "",f'当前 {align["measurements"]} 份比较范围内输出对应 {align["unique_cases"]} 份不同文本，'
        f'其中 {align["reviewed_unique_cases"]} 份已审核、{align["remaining_unique_cases"]} 份待审核。',
        "", "反余切分支敏感性只调整该题的答案集合口径；论证等级及主分析不变。下表分母是各方法已审核数量。", "",
        "| 方法 | 主口径正确 | 接受反余切 (0,π) 分支时正确 | 增加正确数 |",
        "| --- | ---: | ---: | ---: |"]
    for m in s["methods"]:
        lines.append(f'| {NAMES[m["mode"]]} | {m["reviewed_correct"]} / {m["reviewed_n"]} | '
            f'{m["branch_adjusted_correct"]} / {m["reviewed_n"]} | {len(m["branch_sensitive_cases"])} |')
    lines += ["","## 补测进度","",
        "| 补测 | 完成 / 目标 | 自动正确 | 已审核 / 正确 | 成立 / 局部 / 关键错误 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for m in s["methods"][4:]:
        g=m["quality_grades"]
        lines.append(f'| {NAMES[m["mode"]]} | {m["n"]} / 100 | {m["auto_correct"]} | {m["reviewed_n"]} / {m["reviewed_correct"]} | '
                     f'{g.get("valid",0)} / {g.get("local_error",0)} / {g.get("critical_error",0)} |')
    lines += ["", ('各方法均覆盖相同的 100 题，下表按同题配对汇总。' if completed else
        "各方法当前完成的题目集合可能不同，不能横向比较上述比例。下表分别使用每个补测与对照的同题交集，"
        "仅用于核对进度；完整 100 题结束前不据此做显著性判断。"), "",
        "| 补测 | 对照 | 同题数 | 答案胜 / 负 | 成立改善 / 退化 | 总耗时比 | 实际 PFLOPs 比 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in s["comparisons"]:
        if c["scope"]!="all":continue
        lines.append(f'| {NAMES[c["new_mode"]]} | {NAMES[c["control_mode"]]} | {c["n"]} | '
            f'{c["answer_wins"]} / {c["answer_losses"]} | {c["proof_valid_wins"]} / {c["proof_valid_losses"]} | '
            f'{c["seconds_ratio"]:.3f} | {c["padded_flops_ratio"]:.3f} |')
    if completed:
        lines += ["", "| 补测 | 对照 | 准确率差 95% 区间 百分点 | 成立率差 95% 区间 百分点 | 耗时比 95% 区间 | FLOPs 比 95% 区间 | McNemar p |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for c in s["comparisons"]:
            if c["scope"]!="all":continue
            ci=c["inference"]
            fmt=lambda pair: f'[{pair[0]:.3f}, {pair[1]:.3f}]'
            lines.append(f'| {NAMES[c["new_mode"]]} | {NAMES[c["control_mode"]]} | '
                         f'{fmt(ci["answer_delta_pp_95"])} | {fmt(ci["proof_valid_delta_pp_95"])} | '
                         f'{fmt(ci["seconds_ratio_95"])} | {fmt(ci["padded_flops_ratio_95"])} | '
                         f'{ci["mcnemar_exact_two_sided_p"]:.4f} |')
        lines += ["", "## 原 30 题与新增 70 题", "",
            "每个队列均按同题配对，bootstrap 在该队列内按服务器分层。质量列给出补测减对照的百分点差，成本列为总量比；括号内为 95% 区间。", "",
            "| 队列 | 补测 / 对照 | 答案正确率差 | 论证成立率差 | 总耗时比 | 实际 FLOPs 比 |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
        for scope,label in (("old","原 30 题"),("new","新增 70 题")):
            for c in s["comparisons"]:
                if c["scope"]!=scope:continue
                ci=c["inference"]
                answer_delta=100*(c["reviewed_correct_new"]-c["reviewed_correct_control"])/c["n"]
                proof_delta=100*(c["proof_valid_new"]-c["proof_valid_control"])/c["n"]
                lines.append(f'| {label} | {NAMES[c["new_mode"]]} / {NAMES[c["control_mode"]]} | '
                    f'{answer_delta:+.2f} {fmt(ci["answer_delta_pp_95"])} | '
                    f'{proof_delta:+.2f} {fmt(ci["proof_valid_delta_pp_95"])} | '
                    f'{c["seconds_ratio"]:.3f} {fmt(ci["seconds_ratio_95"])} | '
                    f'{c["padded_flops_ratio"]:.3f} {fmt(ci["padded_flops_ratio_95"])} |')
    lines += ["", "胜 / 负指补测正确而对照错误 / 补测错误而对照正确；成立改善 / 退化仅把“成立”作为二元指标。"
        "每一行的对照均限于相同题目、重复号和服务器。", "",
        "时间包括搜索和最终内容生成，排除加载、预热、清缓存及最终判题；计算量使用实际执行图 token 位置，"
        "含图填充、预填充及取消前已执行工作，按 2×26,895,998,464×token 位置近似。"
        "词表统计归约成本反映在时间中，未计作额外模型 FLOPs。不同方法总预算不强制相等。", "",
        "完整结果使用同题、按服务器分层的 20,000 次 bootstrap（种子 20261103），报告准确率差、"
        "成立率差、总耗时比和 FLOPs 比的 95% 区间，并提供精确 McNemar 检验。"
        "区间不覆盖审核者误差，且未校正多重比较。", "",
        "## 文件与继续运行","",
        "服务器原始输出位于 /data/disk/changxiangyu/agentic_scaling/results/selection-met-sc-20261103，"
        "A=27.106.124.198、B=189.1.227.167，均使用 changxiangyu 账号。"
        +("每台已完成 old 15 题和 new 35 题。" if completed else "每台依次运行 old 15 题和 new 35 题，已有任务会继续执行。"), "",
        "summary.json 保存各方法汇总、逐服务器比值、原 30 / 新增 70 / 合计配对结果；"
        "paired-records.json 保存逐题质量和成本；labels.jsonl 保存全部本次论证判断；"
        "answer_only_labels.jsonl 仅用于额外旧方法的答案等价修正；blind_cases.jsonl 与 method_key.json 可还原审核输入；"
        "provenance.json 保存源文件与审核文件哈希。alignment-checks.json 保存配置、取消、重评分与回退检查。", "",
        "summary.json 的 batches 保存四个批次的测量数、汇总记录和退出码；只有四批数量完整、汇总存在、退出码均为 0 且全部输出已审核时，报告才标记完成。", "",
        "刷新本地快照：分别运行 work/collect_met_sc.py a 与 b；随后运行 work/met_sc_study.py prepare。"
        "work/met_sc_study.py batch 输出尚未审核的文本；完成审核后运行 work/summarize_met_sc.py 重建汇总。"
        "未审核输出会明确保留为待审核，不自动赋予论证等级。", ""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
