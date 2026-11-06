"""Join the fixed-question studies without inventing unavailable proof labels."""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from met_sc_study import NEW_MODES, ROOT

ORDER = (
    "base", "is", "sir-k1", "met-k1", "sir-k2", "met-k2", "block-met-k2",
    "bon-consilience", "is-self-certainty", "met-k1-self-certainty",
    "met-k2-self-certainty", "block-met-k2-self-certainty",
)
HISTORICAL_NAMES = dict(zip((
    "Base", "快速 IS，K=1", "完整池 SIR，K=1", "逐条 Metropolized，K=1",
    "完整池 SIR，K=2", "逐条 Metropolized，K=2", "按块 Metropolized，K=2",
    "Best-of-4 + Consilience", "IS + Self-Certainty",
), ORDER[:9]))
BRANCH_PID = "test/precalculus/768.json"
BRANCH_ANSWER = "{-3 + 2*sqrt(2), -3 - 1*2*sqrt(2), 2*sqrt(2) + 3, -1*2*sqrt(2) + 3}"
GRADES = ("valid", "local_error", "critical_error")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(out, name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [100 * (center - half), 100 * (center + half)]


def historical_grades(text):
    section = text.split("## 最终内容的论证质量", 1)[1].split("## 时间与计算量", 1)[0]
    result = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        fields = [x.strip() for x in line.strip("|").split("|")]
        if fields[0] not in HISTORICAL_NAMES:
            continue
        new, all_rows = ([int(x.strip()) for x in field.split("/")] for field in fields[1:3])
        assert len(new) == len(all_rows) == 3 and sum(new) == 70 and sum(all_rows) == 100
        result[HISTORICAL_NAMES[fields[0]]] = {
            "all": dict(zip(GRADES, all_rows)), "new": dict(zip(GRADES, new)),
            "old": dict(zip(GRADES, [a - b for a, b in zip(all_rows, new)])),
            "source": "historical-selection-report.md",
            "individual_labels_available": False,
        }
    assert set(result) == set(ORDER[:9])
    return result


def common_generation(metadata):
    s = metadata["settings"]
    ar = s["ar"]
    ds = s["datasets"]["math500"]
    return {
        "seed": s["run"]["seed"], "draws": s["run"]["draws"],
        "model": ar["model"], "thinking": ar["thinking"],
        "max_new_tokens": ar["max_new_tokens"], "sampling": ar["sampling"],
        "prompt": ar["prompt"], "output": ar["output"],
        "dataset_sha256": metadata["dataset"]["source_sha256"],
        "dataset_source": ds["source"], "dataset_prompt": ds["prompt_template"],
        "selection_seed": ds["selection"]["seed"],
        "minimum_level": ds["selection"]["minimum_level"],
        "excluded_ids": ds["selection"]["excluded_ids"],
        "dtype": ar["engine"]["dtype"], "device": ar["engine"]["device"],
        "tensor_parallel_size": ar["engine"]["vllm"]["tensor_parallel_size"],
        "parameter_count": metadata["parameter_count"],
    }


def aggregate(rows):
    n = len(rows)
    correct = sum(r["reviewed_correct"] for r in rows)
    grades = dict(Counter(r["grade"] or "not_reviewed" for r in rows))
    return {
        "n": n, "auto_correct": sum(r["correct"] for r in rows),
        "corrected_correct": correct, "accuracy_wilson_95_pct": wilson(correct, n),
        "branch_adjusted_correct": sum(r["branch_adjusted_correct"] for r in rows),
        "answer_sources": dict(Counter(r["answer_source"] for r in rows)),
        "mean_seconds": sum(r["seconds"] for r in rows) / n,
        "p95_seconds": float(np.quantile([r["seconds"] for r in rows], .95)),
        "total_seconds": sum(r["seconds"] for r in rows),
        "padded_pflops": sum(r["padded_flops"] for r in rows) / 1e15,
        "current_proof_reviewed_n": sum(r["grade"] is not None for r in rows),
        "current_proof_grades": grades,
        "current_proof_complete": "not_reviewed" not in grades,
    }


def ci(values, digits=1):
    return f'[{values[0]:.{digits}f}, {values[1]:.{digits}f}]'


def grade_string(grades):
    return " / ".join(str(grades.get(k, 0)) for k in GRADES)


def extend_report(raw_rows, metas, supplement, names, paired_comparison, out):
    historical_path = ROOT / "a/source/docs/reports/QWEN38_IS_SELECTION.md"
    historical_text = historical_path.read_text(encoding="utf-8")
    hist = historical_grades(historical_text)
    (out / "historical-selection-report.md").write_bytes(historical_path.read_bytes())
    original_provenance = json.loads((ROOT / "provenance-before-merge.json").read_text(encoding="utf-8"))
    for entry in original_provenance["measurements"]:
        assert sha(ROOT / entry["path"]) == entry["sha256"], entry["path"]

    baseline = common_generation(metas[0])
    assert all(common_generation(m) == baseline for m in metas)
    rows = []
    for source in raw_rows:
        r = dict(source)
        assert r["status"] == "complete" and isinstance(r["correct"], bool)
        if r["reviewed_correct"] is None:
            r["reviewed_correct"] = r["correct"]
            r["answer_source"] = "original_automatic_judgment_with_retained_correction_policy"
        else:
            r["answer_source"] = "current_final_text_review" if r["grade"] else "explicit_answer_equivalence_review"
        if r["problem_id"] == BRANCH_PID:
            assert r["answer"] == BRANCH_ANSWER and not r["reviewed_correct"]
            r["branch_adjusted_correct"] = True
            r["branch_sensitivity_source"] = "four_root_answer_set_checked; historical_branch_policy_retained"
        else:
            r["branch_adjusted_correct"] = r["reviewed_correct"]
        rows.append(r)
    assert len(rows) == 1200 and set(r["mode"] for r in rows) == set(ORDER)
    base = {(r["problem_id"], r["repeat"]): r for r in rows if r["mode"] == "base"}
    checks = []
    methods = []
    for mode in ORDER:
        group = [r for r in rows if r["mode"] == mode]
        assert len(group) == len(base) == 100
        assert set((r["problem_id"], r["repeat"]) for r in group) == set(base)
        assert Counter(r["replica"] for r in group) == {"a": 50, "b": 50}
        assert Counter(r["cohort"] for r in group) == {"old": 30, "new": 70}
        for r in group:
            b = base[(r["problem_id"], r["repeat"])]
            assert (r["seed"], r["replica"], r["cohort"]) == (b["seed"], b["replica"], b["cohort"])
        cohorts = {scope: aggregate([r for r in group if scope == "all" or r["cohort"] == scope])
                   for scope in ("all", "old", "new")}
        methods.append({"mode": mode, "name": names[mode], "cohorts": cohorts,
                        "historical_proof_summary": hist.get(mode)})
        checks.append({"mode": mode, "same_100_question_repeat_keys": True, "same_per_question_seed": True,
                       "same_server_and_cohort": True, "same_common_generation_settings": True})
    by_mode = {m["mode"]: m for m in methods}
    for old in supplement["baseline_reproduced"]:
        assert by_mode[old["mode"]]["cohorts"]["all"]["corrected_correct"] == old["corrected_correct"]
    for m in supplement["methods"]:
        merged = by_mode[m["mode"]]["cohorts"]["all"]
        assert merged["corrected_correct"] == m["reviewed_correct"]
        assert merged["current_proof_grades"] == m["quality_grades"]

    comparisons = []
    for i, mode in enumerate(ORDER):
        for control in ORDER[:i]:
            for scope in ("all", "old", "new"):
                nr = [r for r in rows if r["mode"] == mode and (scope == "all" or r["cohort"] == scope)]
                cr = [r for r in rows if r["mode"] == control and (scope == "all" or r["cohort"] == scope)]
                result, _ = paired_comparison(mode, control, nr, cr, scope)
                assert result and not result["interim"]
                result["answer_label_policy"] = "original judgments plus documented equivalence corrections"
                result["proof_pairing_policy"] = "available only when every paired output has a current individual label"
                comparisons.append(result)
    assert len(comparisons) == 198
    lookup = {(c["new_mode"], c["control_mode"], c["scope"]): c for c in comparisons}
    for old in supplement["comparisons"]:
        current = lookup[(old["new_mode"], old["control_mode"], old["scope"])]
        assert all(current[k] == v for k, v in old.items()), (old["new_mode"], old["control_mode"], old["scope"])

    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
              "measurement_count": 1200, "method_count": 12, "question_count": 100,
              "method_order": ORDER, "methods": methods, "comparisons": comparisons,
              "alignment_checks": checks, "common_generation_settings": baseline,
              "inference": {"pairwise_comparisons": 66, "scopes": ["all", "old", "new"],
                            "bootstrap_replicates": 20000, "seed": 20261103, "strata": "replica",
                            "multiple_comparison_correction": False,
                            "scope": "fixed questions, one generation seed; no reviewer-error uncertainty"},
              "proof_audit": {"complete_current_methods": [m["mode"] for m in methods
                             if m["cohorts"]["all"]["current_proof_complete"]],
                             "current_unique_labels": supplement["alignment"]["reviewed_unique_cases"],
                             "label_matched_outputs": sum(r["grade"] is not None for r in rows),
                             "historical_summaries_have_individual_labels": False},
              "limitations": ["Cross-batch wall time depends on shared load and scalar statistics overhead.",
                  "Best-of-4 wall time also includes CPU grading of its four candidates; it does not affect selection.",
                  "Same per-sequence length caps, not an equal total token/FLOPs budget.",
                  "The seven complete current proof audits cannot be ranked against historical aggregate grades.",
                  "Other old answers retain automatic judgments plus documented corrections; not all were newly re-reviewed."]}
    dump(out, "all-methods-summary.json", report)
    records = [{k: r[k] for k in (
        "mode", "problem_id", "repeat", "replica", "cohort", "seed", "case_id", "source_file",
        "status", "correct", "reviewed_correct", "answer_source", "branch_adjusted_correct", "grade",
        "seconds", "padded_flops", "algorithm_config",
    )} for r in rows]
    dump(out, "all-method-records.json", records)
    dump(out, "all-methods-provenance.json", {
        "measurements": original_provenance["measurements"],
        "review_files": {name: sha(out / name) for name in ("labels.jsonl", "answer_only_labels.jsonl")},
        "historical_report": {"source": str(historical_path), "copy": "historical-selection-report.md",
                              "sha256": sha(historical_path)},
        "branch_sensitivity": {"problem_id": BRANCH_PID, "answer": BRANCH_ANSWER,
                               "all_12_answer_sets_equal": True, "correct_increase_per_method": 1,
                               "evidence": "For arccot in (0,pi), the equation reduces to 2|x|/(1+x^2)=1/3, giving the four displayed roots."},
        "generators": {name: sha(ROOT.parent / name) for name in ("summarize_met_sc.py", "merge_selection_results.py")},
        "outputs": {name: sha(out / name) for name in ("all-methods-summary.json", "all-method-records.json")},
    })

    lines = ["# Qwen3.8-27B：十二种方法合并对照", "",
        "原九种方法的 900 次测量与三种 Metropolized + Self-Certainty 补测的 300 次测量已合并，"
        "共 12 种方法、同一固定 100 题、1200 次测量。原 30 题与新增 70 题分别保留；"
        "每题的种子、服务器、数据版本与公共生成设置核对一致。合并使用已有测量，没有重新运行模型。", "",
        "## 合并结论", "",
        "答案正确数最高为按块 K=2 + Self-Certainty 的 95/100；Base 为 92/100，快速 Self-Certainty IS 和 Best-of-4 均为 94/100。"
        "所有方法相对 Base 的准确率差 95% 配对区间均包含 0，当前样本没有确立稳定准确率优势。", "",
        "Base 的平均耗时和计算量最低。快速 Self-Certainty IS 的均时为 87.71 秒；Best-of-4 为 91.12 秒，"
        "但累计计算量更少（51.629 对 57.904 PFLOPs）。按块 K=2 + Self-Certainty 为 105.29 秒、101.441 PFLOPs，"
        "相对快速 Self-Certainty IS 仅净多对 1 题，耗时约增加 20%，FLOPs 约增加 75%。", "",
        "论证质量保留两轮审核：历史汇总中 Best-of-4 为 85/100 成立；本次有完整逐条证据的七组中，"
        "按块 K=2 + Self-Certainty 为 84/100，快速 Self-Certainty IS 为 83/100。"
        "这两个审核版本不能混成十二方法的统一论证排名。", "",
        "## 十二方法总表", "",
        "C = Consilience；SC = Self-Certainty。答案采用原判加已保存的等价修正；"
        "旧九方法的 12 条修正均已重建。未新增复核标签的旧输出保留原判，不计作本次重新审核。"
        "PFLOPs 是对应 100 题累计量；耗时为每题均值。", "",
        "| 方法 | 原 30 题正确 | 新增 70 题正确 | 合计正确 / 100 | 均时 秒 | P95 秒 | 累计 PFLOPs | 耗时 / Base | FLOPs / Base |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    base_all = by_mode["base"]["cohorts"]["all"]
    short = lambda mode: names[mode].replace("Consilience", "C").replace("Self-Certainty", "SC")
    for m in methods:
        c = m["cohorts"]
        a = c["all"]
        lines.append(f'| {short(m["mode"])} | {c["old"]["corrected_correct"]} / 30 | {c["new"]["corrected_correct"]} / 70 | '
            f'{a["corrected_correct"]} | {a["mean_seconds"]:.2f} | {a["p95_seconds"]:.1f} | {a["padded_pflops"]:.3f} | '
            f'{a["total_seconds"]/base_all["total_seconds"]:.3f} | {a["padded_pflops"]/base_all["padded_pflops"]:.3f} |')
    lines += ["", "## 全部方法相对 Base 的配对结果", "",
        "胜 / 负为该方法正确且 Base 错误 / 该方法错误且 Base 正确。答案差以百分点计，成本比为总量比。"
        "括号内为同题配对、按服务器分层的 20,000 次 bootstrap 95% 区间。", "",
        "| 方法 | 答案胜 / 负 | 答案差及 95% 区间 | 耗时比及 95% 区间 | FLOPs 比及 95% 区间 | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode in ORDER[1:]:
        c = lookup[(mode, "base", "all")]
        q = c["inference"]
        assert q["answer_delta_pp_95"][0] <= 0 <= q["answer_delta_pp_95"][1]
        lines.append(f'| {short(mode)} | {c["answer_wins"]} / {c["answer_losses"]} | '
            f'{c["reviewed_correct_new"]-c["reviewed_correct_control"]:+d} {ci(q["answer_delta_pp_95"])} | '
            f'{c["seconds_ratio"]:.3f} {ci(q["seconds_ratio_95"],3)} | '
            f'{c["padded_flops_ratio"]:.3f} {ci(q["padded_flops_ratio_95"],3)} | {q["mcnemar_exact_two_sided_p"]:.4f} |')
    lines += ["", "## 方法之间的关键对照", "",
        "补测与同选择核 C 的对照主要考察奖励变化；补测与快速 SC 的对照同时改变等待完整池及选择核。"
        "下表均为 100 题；其余任意两方法以及原 30 / 新增 70 题的结果见 all-methods-summary.json。", "",
        "| 方法 / 对照 | 答案胜 / 负 | 答案差及 95% 区间 | 总耗时比 | FLOPs 比 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    selected = [("bon-consilience", "is"), ("is-self-certainty", "is"),
                ("is-self-certainty", "bon-consilience"), ("met-k1", "sir-k1"),
                ("met-k2", "sir-k2"), ("block-met-k2", "sir-k2")]
    selected += [(mode, control) for mode in NEW_MODES
                 for control in (mode.removesuffix("-self-certainty"), "is-self-certainty", "bon-consilience")]
    for mode, control in selected:
        c = lookup[(mode, control, "all")]
        lines.append(f'| {short(mode)} / {short(control)} | {c["answer_wins"]} / {c["answer_losses"]} | '
            f'{c["reviewed_correct_new"]-c["reviewed_correct_control"]:+d} {ci(c["inference"]["answer_delta_pp_95"])} | '
            f'{c["seconds_ratio"]:.3f} | {c["padded_flops_ratio"]:.3f} |')
    lines += ["", "## 论证质量：保留审核版本", "",
        "以下数字顺序均为成立 / 局部错误 / 关键错误。历史列来自原报告汇总，原始逐条审核表未找到；"
        "本次列只在当前逐条标签完整覆盖该方法 100 题时显示。四组旧直接对照的历史和本次数字不同，"
        "反映审核判断变化，不能解释为模型输出改善。", "",
        "| 方法 | 历史审核汇总 / 100 | 本次完整审核 / 100 | 本次标签覆盖 |",
        "| --- | ---: | ---: | ---: |"]
    for m in methods:
        a = m["cohorts"]["all"]
        historical = grade_string(hist[m["mode"]]["all"]) if m["mode"] in hist else "—"
        current = grade_string(a["current_proof_grades"]) if a["current_proof_complete"] else "未完整复核"
        lines.append(f'| {short(m["mode"])} | {historical} | {current} | {a["current_proof_reviewed_n"]} / 100 |')
    lines += ["", "快速 C IS、SIR K=1、SIR K=2 的部分输出与已审核文本完全一致，可按题目与最终文本哈希复用标签。"
        "这些匹配子集并非随机样本，不用其成立比例估计整组论证质量。Base 与 Best-of-4 本次没有完整论证审核。"
        "两轮审核均由同一助手进行，仅覆盖最终展示的推导；不代表独立人工审核或全部内部思考质量。", "",
        "## 原 30 题与新增 70 题的成本", "",
        "| 方法 | 原 30：均时 秒 | 原 30：PFLOPs | 新增 70：均时 秒 | 新增 70：PFLOPs |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for m in methods:
        a, b = m["cohorts"]["old"], m["cohorts"]["new"]
        lines.append(f'| {short(m["mode"])} | {a["mean_seconds"]:.2f} | {a["padded_pflops"]:.3f} | '
                     f'{b["mean_seconds"]:.2f} | {b["padded_pflops"]:.3f} |')
    lines += ["", "## 统一口径与限制", "",
        "模型、题目、逐题种子、分配服务器、温度、惩罚及逐序列长度上限一致。原七方法代码为 39dfa0a，"
        "Best-of-4 与快速 SC 为 150b1ba，三组补测为 6bde29b。后两批启用全词表 SC 标量统计，"
        "统计归约耗时进入墙钟。各方法实际配置和源码哈希保存在合并数据及原始元数据中。", "",
        "计时包括搜索和最终内容生成，通常排除加载、预热、清缓存、最终判题及写盘。"
        "Best-of-4 的现有入口还在计时内对四个候选做 CPU 判题，判题结果不参与选择；该开销保留。"
        "跨批次的共享负载仍会影响墙钟，因此耗时差不能全部归因于选择核或奖励。", "",
        "计算量使用 2 × 26,895,998,464 × 实际执行 token 位置数的稠密近似，包含预填充、执行图填充和取消前已执行工作。"
        "全词表统计归约未计作额外模型 FLOPs。各方法总 token/FLOPs 预算不相等；本报告不能回答固定总计算预算下谁更优。", "",
        "反余切题 precalculus/768 保留参考解的两正根口径。若接受 arccot 值域 (0,π)，"
        "十二方法的四根答案均增加 1 题正确，方法之间的准确率差不变；该敏感性不混入等价修正。", "",
        "合并保存 66 对方法 × 3 个队列，共 198 组配对结果。未校正多重比较，区间不覆盖审核者误差或换种子的波动；"
        "只在两方法全部配对输出都有本次论证标签时计算论证质量差。缺失标签保留为空。", "",
        "## 合并数据与复现", "",
        "all-methods-summary.json 包含十二方法汇总、历史及当前论证统计、原 30 / 新增 70 / 合计的所有两两比较与置信区间；"
        "all-method-records.json 保存 1200 条可配对的质量、来源和成本记录；all-methods-provenance.json 保存原始数据、审核与历史报告哈希。"
        "historical-selection-report.md 是旧报告原样快照，末尾的补测待完成说明属于历史状态。", "",
        "运行 work/summarize_met_sc.py 可重建补测明细及合并结果；它会核验原九方法正确数及原有 18 组补测配对结果保持一致。"
        "两台服务器上的合并报告与证据继续保存在 results/selection-met-sc-20261103/audit-codex-20261103/。", "",
        "## 三种补测的详细报告", ""]
    previous = (out / "REPORT.md").read_text(encoding="utf-8").splitlines()
    assert previous[0] == "# Qwen3.8-27B Metropolized 与 Self-Certainty 补测"
    lines.extend(("#" + line if line.startswith("## ") else line) for line in previous[1:])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return {"methods": 12, "measurements": 1200, "questions": 100, "pairwise_comparisons": 198,
            "summary_file": "all-methods-summary.json", "records_file": "all-method-records.json",
            "provenance_file": "all-methods-provenance.json", "current_full_proof_audits": 7}
