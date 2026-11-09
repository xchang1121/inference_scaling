"""Audit snapshot provenance and summarize paired online-injection results."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from benchmark_injection import arms, grade, read_jsonl


def paired(base, alternative, key):
    a, b = np.array([r[key] for r in base]), np.array([r[key] for r in alternative])
    rng = np.random.default_rng(20261109)
    ix = rng.integers(0, len(a), size=(5000, len(a)))
    delta = b-a
    return {"difference":float(delta.mean()),
            "bootstrap_95":[float(v) for v in np.quantile(delta[ix].mean(axis=1),[.025,.975])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = json.loads((args.snapshot/"state.json").read_text(encoding="utf-8"))
    assert not state["changed_files"]
    summary, all_grades, checks = {}, [], []
    for name in ("api", "techqa"):
        manifest = json.loads((args.data/name/"manifest.json").read_text(encoding="utf-8"))
        gold = {r["id"]:r for r in read_jsonl(args.data/name/"gold.jsonl")}
        path = args.snapshot/f"{name}.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        measurements = [r for r in rows if r["kind"] == "measurement"]
        metadata = next((r for r in rows if r["kind"] == "metadata"), {})
        modes = metadata.get("arms", list(arms()))
        by_key = {}
        for row in measurements:
            key = row["question_id"], row["arm"]
            assert key not in by_key, key
            result = grade(row["answer_text"], gold[row["question_id"]], row["dataset"])
            row.update(score=float(result["correct"] if "correct" in result else result["f1"]),
                       exact_match=float(result.get("correct",result.get("exact_match",False))),
                       parseable=result.get("parseable",True))
            injections = [e for e in row["events"] if e.get("injected")]
            row["post_injection_thinking_tokens"] = max(0, row["thinking_tokens"]-injections[0]["actual_generated_offset"]) if injections else None
            row["evidence_doc_hit"] = any(h["id"] in gold[row["question_id"]]["document_ids"]
                                          for event in injections for h in event["hits"])
            all_grades.append({"dataset":name,"question_id":key[0],"arm":key[1], **result})
            by_key[key] = row
            reconstructed = tuple(row["prompt_ids"])
            prefix_ok = True
            for segment in row["segments"]:
                if segment["kind"] == "prompt_evidence":
                    continue
                prefix_ok &= segment["context_offset"] == len(reconstructed)
                reconstructed += tuple(segment["token_ids"])
            prefix_ok &= reconstructed == tuple(row["context_ids"])
            model_tokens = sum(len(s["token_ids"]) for s in row["segments"] if s["kind"] == "generated")
            thinking_lp = [v for s in row["segments"] if s["kind"] == "generated" and s["phase"] == "thinking"
                           for v in s["token_logprobs"]][:row["thinking_tokens"]]
            expected_lp = statistics.fmean(thinking_lp) if thinking_lp else None
            observed_lp = row["confidence"]["thinking_mean_logprob"]
            probability_scope_ok = all(not ({"token_logprobs","token_statistics"} & set(s))
                                       for s in row["segments"] if s["kind"] != "generated")
            probability_scope_ok &= row["confidence"]["sampled_thinking_tokens"] == len(thinking_lp) == row["thinking_tokens"]
            probability_scope_ok &= expected_lp == observed_lp or (
                expected_lp is not None and observed_lp is not None and abs(expected_lp-observed_lp) < 1e-7)
            checks.append({"dataset":name,"question_id":key[0],"arm":key[1],
                "exact_prefix_reconstruction":prefix_ok, "generated_tokens_match":model_tokens == row["generated_tokens"],
                "trajectory_hash_matches":hashlib.sha256(json.dumps(row["context_ids"]).encode()).hexdigest() == row["trajectory_sha256"],
                "confidence_excludes_injections":row["confidence"]["includes_injected_tokens"] is False,
                "confidence_recomputed_from_generated_thinking":probability_scope_ok,
                "generated_probability_lengths_match":all(len(s["token_ids"]) == len(s["token_logprobs"])
                    for s in row["segments"] if s["kind"] == "generated"),
                "native_graph_cost_available":row.get("graph_execution",{}).get("estimated_dense_padded_flops",0) > 0,
                "native_at_least_logical_forward":row.get("graph_execution",{}).get("forward_token_slots",-1) >= row["cost"].get("generation_forward_token_slots",0),
                "no_rescoring_forward":row["cost"].get("score_forward_token_slots",0) == 0,
                "no_preemption":row["cost"].get("preemptions",0) == 0,
                "no_cancelled_request":row["cost"].get("cancelled_requests",0) == 0})
        common = sorted(qid for qid in gold if all((qid, mode) in by_key for mode in modes))
        methods = []
        for mode in modes:
            chosen = [by_key[qid,mode] for qid in common]
            native = [r.get("graph_execution",{}).get("estimated_dense_padded_flops") for r in chosen]
            cached = sum(r["cost"].get("shared_prefill_tokens_saved",0) for r in chosen)
            fresh = sum(r["cost"].get("prefill_tokens",0) for r in chosen)
            methods.append({"arm":mode,"generated":sum(key[1]==mode for key in by_key),"n":len(chosen),
                "score":statistics.fmean(r["score"] for r in chosen) if chosen else None,
                "exact_match":statistics.fmean(r["exact_match"] for r in chosen) if chosen else None,
                "mean_seconds":statistics.fmean(r["seconds"] for r in chosen) if chosen else None,
                "mean_retrieval_seconds":statistics.fmean(r["retrieval_seconds"] for r in chosen) if chosen else None,
                "mean_generation_tokens":statistics.fmean(r["generated_tokens"] for r in chosen) if chosen else None,
                "mean_thinking_tokens":statistics.fmean(r["thinking_tokens"] for r in chosen) if chosen else None,
                "mean_new_prefill_tokens":fresh/len(chosen) if chosen else None,
                "mean_cached_prefill_tokens":cached/len(chosen) if chosen else None,
                "mean_evidence_tokens":statistics.fmean(r["injected_tokens"] for r in chosen) if chosen else None,
                "injection_rate":statistics.fmean(r["injected_tokens"] > 0 for r in chosen) if chosen else None,
                "evidence_doc_hit_rate":statistics.fmean(r["evidence_doc_hit"] for r in chosen) if chosen else None,
                "evidence_doc_hit_given_injected":statistics.fmean(r["evidence_doc_hit"] for r in chosen if r["injected_tokens"] > 0)
                    if any(r["injected_tokens"] > 0 for r in chosen) else None,
                "parseable_rate":statistics.fmean(r["parseable"] for r in chosen) if chosen else None,
                "correct_count":sum(r["exact_match"] for r in chosen),
                "mean_post_injection_thinking_tokens":statistics.fmean(
                    r["post_injection_thinking_tokens"] for r in chosen if r["post_injection_thinking_tokens"] is not None)
                    if any(r["post_injection_thinking_tokens"] is not None for r in chosen) else None,
                "empty_answers":sum(not r["answer_text"].strip() for r in chosen),
                "length_exhausted":sum(r["finish_reason"] == "max_tokens" for r in chosen),
                "pflops":sum(native)/1e15 if chosen and all(v is not None for v in native) else None,
                "logical_pflops":sum(r["cost"]["estimated_dense_forward_flops"] for r in chosen)/1e15 if chosen else None,
                "cache_hit_fraction":cached/(cached+fresh) if cached+fresh else None})
        comparisons, coupling = [], []
        if common:
            baseline = [by_key[qid,"closed-book"] for qid in common]
            for mode in modes:
                if mode == "closed-book":
                    continue
                chosen = [by_key[qid,mode] for qid in common]
                ratio = sum(r["seconds"] for r in chosen)/sum(r["seconds"] for r in baseline)
                comparisons.append({"reference":"closed-book", "alternative":mode,
                    "n":len(common), **paired(baseline,chosen,"score"), "wall_ratio":ratio})
            def sampled_tokens(row):
                return [t for segment in row["segments"] if segment["kind"] == "generated" for t in segment["token_ids"]]
            for qid in common:
                base = by_key[qid,"closed-book"]
                original = sampled_tokens(base)
                for mode in modes:
                    if mode in {"closed-book","static-user-bm25","start-bm25"}:
                        continue
                    row = by_key[qid,mode]
                    tokens = sampled_tokens(row)
                    shared = next((i for i,(a,b) in enumerate(zip(original,tokens)) if a != b),min(len(original),len(tokens)))
                    injections = [e for e in row["events"] if e.get("injected")]
                    intervention = injections[0]["actual_generated_offset"] if injections else min(len(original),len(tokens))
                    # Sentence waiting changes request seeds/cutting before the
                    # actual evidence append; treat that as the first intervention.
                    waiting = [e["generated_offset"] for e in row["events"] if e.get("position") == "sentence"]
                    cutoff = min([intervention,*waiting])
                    coupling.append({"question_id":qid,"arm":mode,"expected_prefix_tokens":cutoff,
                        "common_sampled_prefix_tokens":shared,"same_prompt":row["prompt_ids"] == base["prompt_ids"],
                        "matches_before_intervention":shared >= cutoff,
                        "identical_complete_context":row["context_ids"] == base["context_ids"]})
        indexes = [r for r in rows if r["kind"] == "index"]
        reconciliation = [{"question_id":r["question_id"],"arm":r["arm"],
            "logical_slots":r["cost"]["generation_forward_token_slots"],
            "native_slots":r["graph_execution"]["forward_token_slots"],
            "extra_native_slots":r["graph_execution"]["forward_token_slots"]-r["cost"]["generation_forward_token_slots"],
            "forced_thinking_end":any(e["kind"] == "forced_thinking_end" for e in r["events"])} for r in measurements]
        summary[name] = {"manifest":manifest,"generated":len(measurements),"expected":len(gold)*len(modes),
            "complete_marker":any(r["kind"] == "complete" for r in rows),"common_questions":common,
            "failed_attempts":[r for r in rows if r["kind"] == "failed"],"methods":methods,
            "comparisons":comparisons,"coupling_checks":coupling,"cost_reconciliation":reconciliation,"index_builds":indexes}
    retrieval = {}
    for name, path in (("api", args.data/"api/retrieval.jsonl"), ("techqa",args.snapshot/"techqa-retrieval.jsonl")):
        if path.exists():
            rs = read_jsonl(path)
            retrieval[name] = {kind:{f"hit@{k}":statistics.fmean(r[f"hit@{k}"] for r in rs if r["retriever"]==kind)
                                     for k in (1,2,3,5)} for kind in ("bm25","char","hybrid")}
    exits = {name:(args.snapshot/(name+".exit")).read_text().strip()
             if (args.snapshot/(name+".exit")).exists() else None for name in ("generation","techqa","api")}
    extra_exits = {name:(args.snapshot/(name+".exit")).read_text().strip()
                   if (args.snapshot/(name+".exit")).exists() else None
                   for name in ("target-cache-validation","cpu-cache-reference","post-validation")}
    cache_path = args.snapshot/"target-cache-validation.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else None
    validation = {"status":"pending" if extra_exits["post-validation"] is None else
                  "passed" if extra_exits["post-validation"] == "0" and cache and cache["passed"]
                  and extra_exits["target-cache-validation"] == "0"
                  and extra_exits["cpu-cache-reference"] == "0" else "failed",
                  "exit_codes":extra_exits,"target_cache":None,"conditional_cache":None}
    if cache:
        validation["target_cache"] = {k:cache.get(k) for k in ("passed","source_question","source_arm",
            "parent_sha256","seed","sample_tokens","absolute_logprob_tolerance","exception")}
        validation["target_cache"]["result_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        validation["target_cache"]["branches"] = [{"branch":b["branch"],"passed":b["passed"],
            "tokens_equal":b["tokens_equal"],"max_abs_logprob_difference":b["max_abs_logprob_difference"],
            "cached_prefill_tokens":b["cached"]["cached_prefill_tokens"],
            "fresh_run_cached_tokens":b["fresh"]["cached_prefill_tokens"]} for b in cache.get("branches",[])]
    conditional_name = ("conditional-cache-diagnostic-v2" if (args.snapshot/"conditional-cache-protocol-v2.json").exists()
                        else "conditional-cache-diagnostic")
    conditional_path = args.snapshot/(conditional_name+".json")
    conditional_exit = args.snapshot/(conditional_name+".exit")
    if conditional_path.exists():
        diagnostic = json.loads(conditional_path.read_text())
        code = conditional_exit.read_text().strip() if conditional_exit.exists() else None
        comparisons = diagnostic.get("checks",[])
        differences = [c["max_abs_logprob_difference"] for c in comparisons if c["tokens_equal"]]
        validation["conditional_cache"] = {
            "status":"pending" if code is None else "passed" if code == "0" and diagnostic["conditional_checks_passed"] else "failed",
            "exit_code":code,"checks":len(comparisons),"passed_checks":sum(c["passed"] for c in comparisons),
            "max_abs_logprob_difference_on_equal_tokens":max(differences,default=None),
            "all_cached_prefills_positive":bool(comparisons) and all(c["cached"]["cached_prefill_tokens"] > 0 for c in comparisons),
            "all_fresh_prefills_uncached":bool(comparisons) and all(c["fresh"]["cached_prefill_tokens"] == 0 for c in comparisons),
            "result_sha256":hashlib.sha256(conditional_path.read_bytes()).hexdigest(),
            "original_validation_passed":diagnostic["original_validation_passed"]}
    finished = all(s["generated"] == s["expected"] and s["complete_marker"] and not s["failed_attempts"] for s in summary.values()) and all(v == "0" for v in exits.values())
    result = {"status":"complete" if finished else "in_progress", "frozen":state["frozen"],
              "analysis_runtime":{"python":sys.version,"numpy":np.__version__,
                  "summarizer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "snapshot_time":state["time"],"datasets":summary,"retrieval":retrieval,"exit_codes":exits,"validation":validation,
              "all_saved_measurements_pass_audit":bool(checks) and all(all(v for v in c.values() if isinstance(v,bool)) for c in checks)}
    (args.output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (args.output/"audit-checks.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    (args.output/"grades.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in all_grades),encoding="utf-8")
    report(args.output, result)
    print(json.dumps({"status":result["status"],"generated":{k:v["generated"] for k,v in summary.items()},
                      "paired_questions":{k:len(v["common_questions"]) for k,v in summary.items()},
                      "audited":result["all_saved_measurements_pass_audit"]}))


def report(out, result):
    stamp = datetime.fromtimestamp(result["snapshot_time"],timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    lines = ["# 技术文档在线注入：十三方法 pilot", "", f"快照：{stamp}；正式生成状态：{result['status']}；状态复用验证：{result['validation']['status']}。", "",
        "本实验由外部控制器自动检索和追加资料，模型没有工具定义或检索调用。对照保持同题、同一组生成/推理预算和预定阈值。源码冻结提交：`"+result["frozen"]["source_commit"]+"`。", "",
        "数据与方法的完整设计见仓库 docs/online_injection.md。规则题为八份官方文档衍生的 16 题人工诊断，TechQA 为 NVIDIA 发布版的生成式小样本评测；二者分别统计，不作为官方榜单成绩。原虚构合同题只保留作机制自测。", "",
        "运行资源：所有正式测量按用户选定使用 27.106.124.198 的物理 3 号单卡，TP=1 / vLLM 0.23。两个数据集在同一引擎中顺序执行。同一数据集的所有方法保持相同运行时；质量与成本在各自组内配对比较。其他卡上的旧输出已归档为预运行，不计入本报告。", "",
        ("质量与成本比较使用十三种方法均已完成的全部共同题；正式队列已结束。" if result["status"] == "complete" else
         "质量比较仅使用十三种方法都完成的共同题。尚未完成的输出不按错误计入；完整队列仍在运行时，此报告会更新。")]
    for name,title in (("api","Official Document Rules Diagnostic"),("techqa","TechQA-RAG-Eval DEV")):
        s = result["datasets"][name]
        lines += ["",f"## {title}","",f"已生成 {s['generated']}/{s['expected']} 份；共同完成 {len(s['common_questions'])}/{s['manifest']['count']} 题。", "",
            "| 方法 | 已生成 | 共同题数 | 严格正确率 / 回答 F1 | 平均秒 | 平均生成 token | 实际注入率 | PFLOPs | 缓存命中 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        def fmt(value, percent=False):
            return "—" if value is None else f"{value*100:.1f}%" if percent else f"{value:.3f}"
        for m in s["methods"]:
            lines.append(f"| {m['arm']} | {m['generated']} | {m['n']} | {fmt(m['score'],True)} | {fmt(m['mean_seconds'])} | {fmt(m['mean_generation_tokens'])} | {fmt(m['injection_rate'],True)} | {fmt(m['pflops'])} | {fmt(m['cache_hit_fraction'],True)} |")
        lines += ["", "| 方法 | 决策 JSON 可提取率 | EM | 空回答 | 长度耗尽 | 注入后推理 token | 注入文档命中率 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for m in s["methods"]:
            parse = fmt(m["parseable_rate"],True) if name == "api" else "不适用"
            lines.append(f"| {m['arm']} | {parse} | {fmt(m['exact_match'],True)} | {m['empty_answers']} | {m['length_exhausted']} | {fmt(m['mean_post_injection_thinking_tokens'])} | {fmt(m['evidence_doc_hit_rate'],True)} |")
        if s["comparisons"]:
            lines += ["", "| 相对闭卷 | 分数差（百分点） | 95% 配对区间 | 耗时比 |", "|---|---:|---:|---:|"]
            for c in s["comparisons"]:
                lo,hi=c["bootstrap_95"]
                lines.append(f"| {c['alternative']} | {100*c['difference']:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] | {c['wall_ratio']:.3f} |")
        if name in result["retrieval"]:
            lines += ["", "问题文本检索（不使用生成轨迹）的文档命中率：", "", "| 检索器 | Hit@1 | Hit@2 | Hit@3 | Hit@5 |", "|---|---:|---:|---:|---:|"]
            for kind, metrics in result["retrieval"][name].items():
                lines.append("| "+kind+" | "+" | ".join(f"{metrics[f'hit@{k}']*100:.1f}%" for k in (1,2,3,5))+" |")
        lines += ["", f"生成失败记录：{len(s['failed_attempts'])}。索引构建记录：{json.dumps(s['index_builds'],ensure_ascii=False)}。"]
        extra = [r["extra_native_slots"] for r in s["cost_reconciliation"]]
        if extra:
            lines += ["", f"原生执行图相对逻辑估算多出的前向位置：合计 {sum(extra)}，单条范围 [{min(extra)}, {max(extra)}]；逐条对应关系见 cost_reconciliation。PFLOPs 表格采用原生执行图计数。"]
    lines += ["", "## 解释范围与成本口径", "",
        "规则题评分提取输出中最后一个不含嵌套的 JSON 对象，要求字段恰为 action 和 wait_seconds，类型分别为字符串与整数；两项取值都与 gold 相同才得分。它允许对象前后的其他文字，不等同于整段输出仅为 JSON。可提取率同时要求字段与类型满足约束。action 是题目给定的决策标签，wait_seconds 是剩余强制等待秒数，无计时要求时取 0。八条短文档的检索难度低，且没有隐藏新事实，闭卷可能已掌握规则。TechQA 使用规范化回答 token F1；EM 与逐样本判分在 summary.json / grades.jsonl。这里只抽取可回答 DEV 样本，未覆盖不可回答题；原数据集可能出现在模型训练资料中，不能声称无污染。", "",
        "配对 bootstrap 以题目为单位重采样 5,000 次，未作多重比较校正。小样本区间退化不代表方法稳定等价；阈值没有在本 pilot 上优化。", "",
        "端到端耗时含检索、格式化、追加与续写，排除模型加载、图编译、索引构建及离线判题。PFLOPs 为原生执行图含填充的 2×参数量×token 位置估算，不是硬件 FLOPs 实测。逻辑前向、检索时间、证据长度、空回答、截断和缓存命中均保留在 summary.json。注入文档命中率按 gold 文档 ID 计算；证据截断后未必包含答案原文，因此它不是答案片段召回率。", "",
        "每组方法起始缓存重置，方法内部允许前缀复用。模型为混合 attention/recurrent 架构，短前缀可能无法形成可复用块。没有缓存命中时，中途续写会重算前缀，不能宣称已获得 KV 加速。", "",
        "外部证据和强制控制标记不计入模型生成置信度；可追溯片段与原始 token IDs 保留在测量中。不同 chunk 切分的随机流不保证逐 token 一致，句界位置对照包含请求切分差异。", "",
        f"当前已保存测量的前缀、概率来源、原生图成本、无额外评分、无抢占与无取消核对：{result['all_saved_measurements_pass_audit']}。", "",
        "## 状态复用的独立核对", "",
        "该诊断在全部正式生成结束后单独运行，未计入上表的质量、延迟与计算成本。两份固定资料追加到同一记录前缀，比较缓存兄弟分支与逐次清空缓存后的重算；每支最多 8 token，BF16 log probability 绝对容差 0.05 nats。另有无下载的小型因果模型 CPU 状态隔离测试。", "",
        f"验证状态：{result['validation']['status']}；退出码：{result['validation']['exit_codes']}。"]
    narrative = out/"OBSERVATIONS.md"
    if result["validation"]["target_cache"]:
        value = result["validation"]["target_cache"]
        lines += ["", "缓存核对的逐分支结果：", "", "| 分支 | token 一致 | 最大 log probability 差 | 缓存分支复用 token | 重算分支复用 token |", "|---|---|---:|---:|---:|"]
        for b in value["branches"]:
            difference = b['max_abs_logprob_difference'] if b['tokens_equal'] else "路径不同，未比较"
            lines.append(f"| {b['branch']} | {b['tokens_equal']} | {difference} | {b['cached_prefill_tokens']} | {b['fresh_run_cached_tokens']} |")
    if result["validation"]["status"] == "failed":
        lines += ["", "目标模型的同种子续写复现要求没有通过，不能将本次结果作为缓存分支一致性的证明。采样路径不同本身不能区分随机流变化与 KV/循环状态差异。原失败记录保留，未更换题目、资料或种子。"]
    conditional = result["validation"]["conditional_cache"]
    if conditional:
        lines += ["", f"事后固定历史的条件状态核对：{conditional['status']}，{conditional['passed_checks']}/{conditional['checks']} 个位置通过；相同贪心 token 的最大 log probability 差为 {conditional['max_abs_logprob_difference_on_equal_tokens']}。",
                  "该诊断沿用原始共同前缀、两份资料和首条完整八 token 记录历史，每个固定位置单独预测一个贪心 token；缓存与重算使用相同条件历史。它不替代原来的多 token 随机续写测试，也不把原失败改记为通过。"]
    lines += ["", "## 来源与复核", "",
        "[NVIDIA 数据卡](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval)、[TechQA 原论文](https://aclanthology.org/2020.acl-main.117/)、[vLLM Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)。规则题八份官方来源、原始网页 SHA256 见 summary.json 的 api.manifest.sources。原始结果、冻结配置与源码哈希保存在服务器 results/online-injection-20261109-a3 和 deployment/ascend/online-injection-20261109-a3。", "", f"退出状态：{result['exit_codes']}。"]
    if (out/"figures/quality-paired.png").exists():
        lines += ["", "## 图示", "", "![配对质量差与区间](figures/quality-paired.png)", "",
                  "![延迟与估算计算量](figures/cost-ratios.png)", "", "![注入前后推理长度](figures/thinking-position.png)"]
    if narrative.exists():
        lines += ["", narrative.read_text(encoding="utf-8").strip()]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
