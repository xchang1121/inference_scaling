"""Audit and summarize the no-effort-prompt, length-matched injection study."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

from benchmark_injection import grade
from inference_scaling.retrieval.ablation import arms


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def bootstrap(values):
    values = np.asarray(values,dtype=float)
    if not len(values):return None
    rng = np.random.default_rng(20261109)
    ix = rng.integers(0,len(values),size=(5000,len(values)))
    return {"difference":float(values.mean()),"bootstrap_95":[float(v) for v in np.quantile(values[ix].mean(axis=1),[.025,.975])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--gold",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    state = json.loads((args.snapshot/"state.json").read_text(encoding="utf-8"))
    assert not state["changed_files"]
    prepared = json.loads((args.snapshot/"prepared.json").read_text(encoding="utf-8"))
    cases = {c["id"]:c for c in prepared["cases"]}
    gold = {r["id"]:r for line in args.gold.read_text(encoding="utf-8").splitlines() if (r:=json.loads(line))}
    assert set(gold) == set(cases)
    path = args.snapshot/"generation.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    measurements = [r for r in rows if r.get("kind") == "measurement"]
    checks,graded,by_key = [],[],{}
    for r in measurements:
        key = r["budget"],r["question_id"],r["arm"]
        assert key not in by_key,key
        case = cases[r["question_id"]]
        variant = case["variants"][r["content_condition"]+"."+r["wrapper_condition"]]
        injections = [e for e in r["events"] if e.get("injected")]
        evidence = [t for s in r["segments"] if s["kind"] in {"evidence","prompt_evidence"} for t in s["token_ids"]]
        reconstructed = r["prompt_ids"].copy();prefix_ok = True
        for s in r["segments"]:
            if s["kind"] == "prompt_evidence":continue
            prefix_ok &= s["context_offset"] == len(reconstructed)
            reconstructed += s["token_ids"]
        thinking = [(t,lp) for s in r["segments"] if s["kind"] == "generated" and s["phase"] == "thinking"
                    for t,lp in zip(s["token_ids"],s["token_logprobs"],strict=True)]
        natural = next((i for i,(t,_) in enumerate(thinking) if t == 248069),None)
        if natural is not None:thinking = thinking[:natural]
        thinking_mean = mean(lp for _,lp in thinking)
        observed = r["confidence"]["thinking_mean_logprob"]
        sc = [v for s in r["segments"] if s["kind"] == "generated" and s["phase"] == "thinking"
              for v in (s.get("token_statistics") or [])]
        if natural is not None:sc = sc[:natural]
        sc_mean = mean(v for v in sc if v is not None and math.isfinite(v))
        observed_sc = r["confidence"]["thinking_mean_self_certainty"]
        answer_count = sum(len(s["token_ids"]) for s in r["segments"] if s["kind"] == "generated" and s["phase"] == "answer")
        answer_reopens = sum(s["token_ids"].count(248068) for s in r["segments"] if s["kind"] == "generated" and s["phase"] == "answer")
        expected_prompt = variant["static_ids"] if r["config"]["position"] == "user" else case["prompt_ids"]
        audit = {"budget":key[0],"question_id":key[1],"arm":key[2],
            "exact_prefix_reconstruction":prefix_ok and reconstructed == r["context_ids"],
            "trajectory_hash_matches":hashlib.sha256(json.dumps(r["context_ids"]).encode()).hexdigest() == r["trajectory_sha256"],
            "actual_prompt_matches_no_effort_preparation":r["prompt_ids"] == expected_prompt and "Reasoning effort is set to" not in case["prompt_text"],
            "exact_prepared_payload":evidence == variant["ids"] if injections else not evidence,
            "payload_length_matched":r["prepared_payload_tokens"] == len(variant["ids"]) == len(case["variants"]["relevant.boxed"]["ids"]),
            "thinking_excludes_external_tokens":len(thinking) == r["thinking_tokens"] == r["confidence"]["sampled_thinking_tokens"],
            "thinking_budget_respected":r["thinking_tokens"] <= r["budget"] == r["config"]["thinking_tokens"],
            "fixed_answer_cap":answer_count == r["answer_generated_tokens"] and answer_count <= 512 == r["config"]["answer_tokens"],
            "confidence_recomputed":thinking_mean == observed or (thinking_mean is not None and observed is not None and abs(thinking_mean-observed) < 1e-7),
            "self_certainty_recomputed":sc_mean == observed_sc or (sc_mean is not None and observed_sc is not None and abs(sc_mean-observed_sc) < 1e-7),
            "thinking_statistics_length":len(sc) == r["thinking_tokens"],
            "external_probability_scope":r["confidence"]["includes_injected_tokens"] is False and all(
                not ({"token_logprobs","token_statistics"} & set(s)) for s in r["segments"] if s["kind"] != "generated"),
            "generated_token_count":sum(len(s["token_ids"]) for s in r["segments"] if s["kind"] == "generated") == r["generated_tokens"],
            "context_budget_respected":len(r["context_ids"]) <= r["config"]["context_tokens"],
            "native_at_least_logical":r["graph_execution"]["forward_token_slots"] >= r["cost"]["generation_forward_token_slots"],
            "native_graph_cost_present":r["graph_execution"]["estimated_dense_padded_flops"] > 0,
            "no_rescoring":r["cost"].get("score_forward_token_slots",0) == 0,
            "no_preemption_or_cancellation":r["cost"].get("preemptions",0) == r["cost"].get("cancelled_requests",0) == 0}
        checks.append(audit)
        result = grade(r["answer_text"],gold[r["question_id"]],r["dataset"])
        forced = any(e["kind"] == "forced_thinking_end" for e in r["events"])
        thinking_eos = r["finish_reason"] == "eos" and natural is None and not forced
        offset = injections[0]["actual_generated_offset"] if injections else None
        first = []
        if injections:
            if r["config"]["position"] == "user":
                first = next((s["token_ids"] for s in r["segments"] if s["kind"] == "generated"),[])
            else:
                at = next(i for i,s in enumerate(r["segments"]) if s["kind"] == "evidence")
                first = next((s["token_ids"] for s in r["segments"][at+1:] if s["kind"] == "generated"),[])
        immediate = bool(first) and first[0] == 248069 if injections else None
        value = {**r,"score":result["f1"],"exact_match":result["exact_match"],"injected":bool(injections),
                 "injection_offset":offset,"forced_end":forced,"natural_end":natural is not None,
                 "eos_before_thinking_end":thinking_eos,
                 "answer_thinking_reopens":answer_reopens,
                 "post_injection_thinking":max(0,r["thinking_tokens"]-offset) if injections else None,
                 "immediate_end_recomputed":immediate,"injection_at_budget_boundary":offset is not None and offset >= r["budget"]}
        by_key[key] = value
        graded.append({"budget":key[0],"question_id":key[1],"arm":key[2],**result,
            "thinking_tokens":r["thinking_tokens"],"injected":bool(injections),"injection_offset":offset,
            "post_injection_thinking":value["post_injection_thinking"],"immediate_end":immediate,
            "forced_end":forced,"natural_end":natural is not None,"eos_before_thinking_end":thinking_eos,
            "answer_thinking_reopens":answer_reopens,"prefix_coupling":r["prefix_coupling"]})
    groups = {}
    contrasts = [("static-relevant","inline32-relevant"),("static-relevant","scdrop-relevant"),
        ("static-relevant","rule-relevant"),("static-relevant","static-unrelated"),
        ("start-relevant","start-unrelated"),("inline32-relevant","inline32-unrelated"),
        ("inline32-relevant","inline32-neutral-relevant"),("inline32-unrelated","inline32-neutral-unrelated"),
        ("inline32-relevant","inline32-wrapper-only"),("scdrop-relevant","scdrop-unrelated"),
        ("rule-relevant","rule-unrelated")]
    for budget in state["frozen"]["budgets"]:
        methods = list(arms(budget))
        common = sorted(q for q in cases if all((budget,q,a) in by_key for a in methods))
        summaries = []
        for arm in methods:
            chosen = [by_key[budget,q,arm] for q in common]
            injected = [r for r in chosen if r["injected"]]
            cache = sum(r["cost"].get("shared_prefill_tokens_saved",0) for r in chosen)
            fresh = sum(r["cost"].get("prefill_tokens",0) for r in chosen)
            summaries.append({"arm":arm,"generated":sum(b==budget and a==arm for b,q,a in by_key),"n":len(chosen),
                "score":mean(r["score"] for r in chosen),"exact_match":mean(r["exact_match"] for r in chosen),
                "mean_seconds":mean(r["seconds"] for r in chosen),"mean_thinking_tokens":mean(r["thinking_tokens"] for r in chosen),
                "mean_answer_tokens":mean(r["answer_generated_tokens"] for r in chosen),
                "injected_count":len(injected),"mean_injection_offset":mean(r["injection_offset"] for r in injected),
                "mean_post_injection_thinking":mean(r["post_injection_thinking"] for r in injected),
                "immediate_end_count":sum(r["immediate_end_recomputed"] for r in injected),
                "immediate_end_given_injected":mean(r["immediate_end_recomputed"] for r in injected),
                "forced_end_count":sum(r["forced_end"] for r in chosen),"natural_end_count":sum(r["natural_end"] for r in chosen),
                "eos_before_thinking_end_count":sum(r["eos_before_thinking_end"] for r in chosen),
                "injection_at_budget_count":sum(r["injection_at_budget_boundary"] for r in chosen),
                "empty_answers":sum(not r["answer_text"].strip() for r in chosen),
                "answer_budget_exhausted":sum(r["finish_reason"] == "answer_budget" for r in chosen),
                "answers_with_reopened_thinking":sum(r["answer_thinking_reopens"] > 0 for r in chosen),
                "finish_reasons":{reason:sum(r["finish_reason"] == reason for r in chosen)
                                  for reason in sorted({r["finish_reason"] for r in chosen})},
                "maximum_context_tokens":max((len(r["context_ids"]) for r in chosen),default=None),
                "prefix_coupling_failures":sum(r["prefix_coupling"] is not None and not r["prefix_coupling"]["matches_baseline_before_injection"] for r in chosen),
                "native_pflops":sum(r["graph_execution"]["estimated_dense_padded_flops"] for r in chosen)/1e15 if chosen else None,
                "logical_pflops":sum(r["cost"]["estimated_dense_forward_flops"] for r in chosen)/1e15 if chosen else None,
                "cache_hit_fraction":cache/(cache+fresh) if cache+fresh else None})
        pairs = []
        for left,right in contrasts:
            qs = sorted(q for q in cases if (budget,q,left) in by_key and (budget,q,right) in by_key)
            a,b = [by_key[budget,q,left] for q in qs],[by_key[budget,q,right] for q in qs]
            both = [(x,y) for x,y in zip(a,b,strict=True) if x["injected"] and y["injected"]]
            checkpoint = [(x,y) for x,y in both if x["config"]["position"] != "user" and y["config"]["position"] != "user"
                          and x["injection_offset"] == y["injection_offset"] and x["prefix_coupling"]["matches_baseline_before_injection"]
                          and y["prefix_coupling"]["matches_baseline_before_injection"]]
            pairs.append({"reference":left,"alternative":right,"n":len(qs),"quality_difference":bootstrap([y["score"]-x["score"] for x,y in zip(a,b,strict=True)]),
                "both_injected":len(both),"same_checkpoint_prefix_pairs":len(checkpoint),
                "same_checkpoint_question_ids":[x["question_id"] for x,y in checkpoint],
                "immediate_end_difference_same_checkpoint":bootstrap([int(y["immediate_end_recomputed"])-int(x["immediate_end_recomputed"]) for x,y in checkpoint]),
                "post_thinking_difference_same_checkpoint":bootstrap([y["post_injection_thinking"]-x["post_injection_thinking"] for x,y in checkpoint]),
                "immediate_end_difference_both_injected":bootstrap([int(y["immediate_end_recomputed"])-int(x["immediate_end_recomputed"]) for x,y in both]),
                "mean_post_thinking_difference_both_injected":mean(y["post_injection_thinking"]-x["post_injection_thinking"] for x,y in both)})
        selected = [r for (b,q,a),r in by_key.items() if b==budget]
        groups[str(budget)] = {"generated":len(selected),"expected":len(cases)*len(methods),"common_questions":common,
                              "methods":summaries,"contrasts":pairs}
    exit_path = args.snapshot/"generation.exit"
    code = exit_path.read_text().strip() if exit_path.exists() else None
    expected = state["frozen"]["expected"]
    complete = len(measurements) == expected and any(r.get("kind") == "complete" for r in rows) and code == "0"
    cross_budget = []
    budgets = sorted(state["frozen"]["budgets"])
    for i,low in enumerate(budgets):
        for high in budgets[i+1:]:
            for q in cases:
                for arm in arms(low):
                    if (low,q,arm) not in by_key or (high,q,arm) not in by_key:continue
                    a,b = by_key[low,q,arm],by_key[high,q,arm]
                    at = [t for s in a["segments"] if s["kind"] == "generated" for t in s["token_ids"]]
                    bt = [t for s in b["segments"] if s["kind"] == "generated" for t in s["token_ids"]]
                    count = min(low,len(at),len(bt))
                    ap = [s["phase"] for s in a["segments"] if s["kind"] == "generated" for t in s["token_ids"]]
                    bp = [s["phase"] for s in b["segments"] if s["kind"] == "generated" for t in s["token_ids"]]
                    mismatch = next((j for j in range(count) if at[j] != bt[j]),None)
                    ar = [t for s in a["segments"] if s["kind"] == "generated" and s["phase"] == "thinking" for t in s["token_ids"]]
                    br = [t for s in b["segments"] if s["kind"] == "generated" and s["phase"] == "thinking" for t in s["token_ids"]]
                    thinking_count = min(low,len(ar),len(br))
                    cross_budget.append({"lower_budget":low,"higher_budget":high,"question_id":q,"arm":arm,
                        "checked_generated_tokens":count,"same_initial_prompt":a["prompt_ids"] == b["prompt_ids"],
                        "matches_before_lower_cap":at[:count] == bt[:count],
                        "first_mismatch_generated_offset":mismatch,
                        "first_mismatch_lower_phase":ap[mismatch] if mismatch is not None else None,
                        "first_mismatch_higher_phase":bp[mismatch] if mismatch is not None else None,
                        "checked_thinking_tokens_including_structural_end":thinking_count,
                        "thinking_prefix_matches_before_lower_cap":ar[:thinking_count] == br[:thinking_count],
                        "neither_forced_thinking_end":not a["forced_end"] and not b["forced_end"],
                        "identical_complete_context":a["context_ids"] == b["context_ids"]})
    result = {"status":"complete" if complete else "in_progress","generation_exit":code,"generated":len(measurements),
        "expected":expected,"frozen":state["frozen"],"snapshot_time":state["time"],"groups":groups,"cross_budget_checks":cross_budget,
        "failed_attempts":[r for r in rows if r.get("kind") == "failed"],
        "prefix_coupling_failures_all_saved":[{"budget":r["budget"],"question_id":r["question_id"],"arm":r["arm"],"coupling":r["prefix_coupling"]}
            for r in measurements if r["prefix_coupling"] is not None and not r["prefix_coupling"]["matches_baseline_before_injection"]],
        "all_saved_measurements_pass_audit":bool(checks) and all(all(v for v in c.values() if isinstance(v,bool)) for c in checks),
        "analysis_runtime":{"python":sys.version,"numpy":np.__version__,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "grading_script_sha256":hashlib.sha256(Path(sys.modules[grade.__module__].__file__).read_bytes()).hexdigest(),
                            "gold_sha256":hashlib.sha256(args.gold.read_bytes()).hexdigest()}}
    for name,value in (("summary.json",result),("audit-checks.json",checks)):
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (args.output/"grades.jsonl").write_text(''.join(json.dumps(r,ensure_ascii=False)+"\n" for r in graded),encoding="utf-8")
    report(args.output,result)
    print(json.dumps({"status":result["status"],"generated":len(measurements),"audited":result["all_saved_measurements_pass_audit"],
                      "complete_questions":{k:len(v["common_questions"]) for k,v in groups.items()}}))


def report(out,result):
    stamp = datetime.fromtimestamp(result["snapshot_time"],timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    lines = ["# 去除 effort 提示后的思考预算与注入格式对照","",f"快照 {stamp}；状态 {result['status']}；已生成 {result['generated']}/{result['expected']}。","",
        "实际模型输入完全不含 low、xhigh 或其他 effort 指令；thinking 已打开，预算由控制器独立限制。所有条件另有 512-token 最终回答上限。模型、完整文档集及八道 TechQA 问题与首轮相同。","",
        "所有正式计算统一使用 198 物理 3 号单卡。relevant 标签指 BM25 检索资料，并不保证包含 gold 答案。相关资料固定为同一组问题检索结果；无关对照只替换正文，来源标题和标签保留。去包装条件将外层 header/footer 换为等 token 数的换行；仅包装条件将正文换为等长度换行。五种变体的 token 数逐题完全一致，静态 RAG 插入同一原始资料 token IDs。","",
        "各位置条件独立生成完整轨迹，开始前清空 APC，内部允许 continuation 复用。动态 SC/规则每 32 token 检查一次，最早 64 token；未触发的结果保留。生成源码、输入、阈值和三档预算在正式测量前冻结，未按结果调整。"]
    def f(v,digits=1):return "—" if v is None else f"{v:.{digits}f}"
    for budget,g in result["groups"].items():
        lines += ["",f"## 思考预算 {budget}","",f"已生成 {g['generated']}/{g['expected']}；十四方法共同完成 {len(g['common_questions'])}/8 题。","",
            "| 方法 | F1 | 思考 token | 回答 token | 注入数 | 注入偏移 | 资料后思考 | 立即结束数 | 强制结束数 | thinking 内 EOS | 空回答 | 平均秒 | 总原生 PFLOPs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m in g["methods"]:
            lines.append(f"| {m['arm']} | {f(m['score']*100 if m['score'] is not None else None)}% | {f(m['mean_thinking_tokens'])} | {f(m['mean_answer_tokens'])} | {m['injected_count']}/{m['n']} | {f(m['mean_injection_offset'])} | {f(m['mean_post_injection_thinking'])} | {m['immediate_end_count']}/{m['injected_count']} | {m['forced_end_count']} | {m['eos_before_thinking_end_count']} | {m['empty_answers']} | {f(m['mean_seconds'],2)} | {f(m['native_pflops'],3)} |")
        lines += ["", "立即结束指资料之后的第一个模型生成 token 为 thinking 结束标记；静态条件取第一个 assistant 生成 token。资料后思考按实际注入子集计算。动态信号在预算边界才触发时，随后强制结束不是自然提前停止，须结合强制结束和边界注入计数解释。", "",
            "| 对照 → 替代 | 配对题数 | F1 差（百分点） | 95% 区间 | 两侧均注入 | 同 checkpoint 前缀配对 | 立即结束率差（百分点） |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for p in g["contrasts"]:
            q=p["quality_difference"];e=p["immediate_end_difference_both_injected"]
            ci="—" if q is None else f"[{q['bootstrap_95'][0]*100:.1f}, {q['bootstrap_95'][1]*100:.1f}]"
            lines.append(f"| {p['reference']} → {p['alternative']} | {p['n']} | {f(q['difference']*100 if q else None)} | {ci} | {p['both_injected']} | {p['same_checkpoint_prefix_pairs']} | {f(e['difference']*100 if e else None)} |")
    lines += ["", "## 核对与解释边界", "",
        f"当前已保存测量的原始前缀、实际无 effort prompt、资料 token 对齐、独立回答预算、概率来源与计算计量核对：{result['all_saved_measurements_pass_audit']}。生成退出码：{result['generation_exit']}。", "",
        "相关/无关对照保留了来源标题，所以比较的是正文贡献。去包装对照引入等长换行，不能据此排除所有其他格式效应。SC/规则未触发与预算边界触发单独保留；原始生成前缀不匹配的样本不能作为严格同前缀因果证据。", "",
        "立即结束指结构性的 thinking 段结束，不等于模型此后没有进行任何推理。回答段重新生成 thinking 起始标记的次数单独保存在统计中。每题每条件仅一条随机轨迹，题级区间不覆盖跨种子方差。", "",
        "模型也可能在生成 </think> 之前直接输出 EOS，整条轨迹结束且无最终回答。thinking 内 EOS 单独统计；立即 </think> 比例降低不能单独解释成恢复了有效推理。", "",
        "预算是上限，不强制模型生成相应长度。三档预算使用相同 prompt 和问题种子，因此在较小上限生效前，轨迹本应相同；跨预算前缀与完整上下文一致性另保存在 cross_budget_checks，不把重复的立即结束现象当作独立重复证据。", "",
        "F1 是词项重叠，不是语义正确率；首轮看过的这八题用于机制复核，不是新的独立泛化测试。配对区间使用题级 5,000 次 bootstrap，未校正多重比较，三档预算的重复不能作为独立新题合并。", "",
        "与首轮还存在独立回答上限和统一 SC 统计采集的差别，因此跨轮的质量或耗时变化不作为仅由 low 提示造成的单因素效应；主要证据来自本轮冻结矩阵内的配对。", "",
        "耗时包含本方法完整前缀生成、追加和续写；检索/资料预处理、模型加载及编译不计入单题耗时。PFLOPs 是原生图含 padding 的 2×参数量×执行位置估算，不是硬件计数器。外部资料不占 thinking budget，但占上下文和 prefill 成本。"]
    cross = result["cross_budget_checks"]
    unforced = [c for c in cross if c["neither_forced_thinking_end"]]
    failures = sum(not c["matches_before_lower_cap"] or not c["same_initial_prompt"] for c in cross)
    thinking_failures = sum(not c["thinking_prefix_matches_before_lower_cap"] for c in cross)
    changed = sum(not c["identical_complete_context"] for c in unforced)
    prefix_failures = len(result["prefix_coupling_failures_all_saved"])
    lines += ["", f"当前已保存测量的注入前闭卷前缀不一致数：{prefix_failures}；跨预算已核对 {len(cross)} 对，较小上限生效前的 prompt/生成前缀不一致 {failures} 对，其中 thinking 前缀不一致 {thinking_failures} 对。两侧均未触发思考上限强制结束的 {len(unforced)} 对中，完整上下文不同 {changed} 对。首次分叉的位置与阶段逐对保存在 summary.json，不把轨迹重复作为独立实验。"]
    narrative=out/"OBSERVATIONS.md"
    if narrative.exists():lines += ["",narrative.read_text(encoding="utf-8").strip()]
    if (out/"figures/immediate-end.png").exists():
        lines += ["", "## 图示", "", "![立即结束思考](figures/immediate-end.png)", "", "![停止类型与空回答](figures/stop-outcomes.png)", "", "![资料前后思考长度](figures/thinking-length.png)", "", "![回答质量](figures/quality.png)"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
