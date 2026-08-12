"""Aggregate repeated rollout-infrastructure runs into JSON, Markdown, and SVG."""

from __future__ import annotations

import argparse
import json
import statistics
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def _load(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _mean_std(values: Sequence[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    return {
        "mean": statistics.mean(numbers),
        "sample_std": statistics.stdev(numbers) if len(numbers) >= 2 else 0.0,
        "runs": len(numbers),
    }


def _arm_metrics(reports: Sequence[dict[str, Any]], section: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for arm in report[section]:
            grouped[str(arm["name"])].append(arm)
    result: dict[str, Any] = {}
    for name, arms in grouped.items():
        values: dict[str, list[float]] = defaultdict(list)
        for arm in arms:
            online = arm["online"]
            model = online["main_model"]
            telemetry = online["telemetry"]
            quality = online.get("quality", online)
            values["online_wall_seconds"].append(telemetry["wall_seconds"])
            values["online_pflops"].append(
                model["estimated_dense_forward_flops"] / 1e15
            )
            values["online_forward_token_slots"].append(
                model["generation_forward_token_slots"]
            )
            values["output_tokens_per_second"].append(
                quality["output_tokens_per_second"]
            )
            values["cache_build_seconds"].append(
                arm["cache_build"]["telemetry"]["wall_seconds"]
            )
            values["background_drain_seconds"].append(
                arm.get("background_drain", {})
                .get("telemetry", {})
                .get("wall_seconds", 0.0)
            )
            values["online_plus_drain_seconds"].append(
                telemetry["wall_seconds"]
                + arm.get("background_drain", {})
                .get("telemetry", {})
                .get("wall_seconds", 0.0)
            )
            for field in (
                "mean_gpu_utilization_percent",
                "maximum_gpu_memory_used_mib",
                "mean_gpu_power_watts",
            ):
                value = telemetry.get(field)
                if value is not None:
                    values[field].append(value)
            if "exact_token_trace_match_fraction_vs_baseline" in online:
                value = online["exact_token_trace_match_fraction_vs_baseline"]
                if value is not None:
                    values["trace_match_fraction"].append(value)
            proposed = model.get(
                "draft_tokens_proposed", model.get("native_draft_tokens", 0)
            )
            accepted = model.get(
                "draft_tokens_accepted", model.get("native_accepted_draft_tokens", 0)
            )
            values["draft_tokens"].append(proposed)
            values["accepted_draft_tokens"].append(accepted)
            values["draft_acceptance_fraction"].append(
                accepted / proposed if proposed else 0.0
            )
            if section == "algorithm_arms":
                diagnostics = online.get("diagnostics", [])
                for field in (
                    "pilot_rollouts",
                    "evaluation_rollouts",
                    "fresh_rollouts",
                    "reused_rollouts",
                    "reward_tail_seconds",
                ):
                    values[field].append(
                        sum(float(item.get(field, 0.0)) for item in diagnostics)
                    )
                run_ahead = arm.get("background_drain", {}).get("run_ahead")
                if run_ahead:
                    values["run_ahead_tokens"].append(
                        float(run_ahead["completed_tokens"])
                    )
                    values["critical_wait_seconds"].append(
                        float(run_ahead["critical_wait_seconds"])
                    )
        result[name] = {field: _mean_std(items) for field, items in values.items()}
    return result


def _paired_ratios(
    reports: Sequence[dict[str, Any]],
    section: str,
    numerator: str,
    denominator: str,
) -> dict[str, Any]:
    ratios: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        arms = {arm["name"]: arm for arm in report[section]}
        left = arms[numerator]
        right = arms[denominator]
        ratios["wall_time_factor"].append(
            left["online"]["telemetry"]["wall_seconds"]
            / right["online"]["telemetry"]["wall_seconds"]
        )
        ratios["main_model_flops_factor"].append(
            left["online"]["main_model"]["estimated_dense_forward_flops"]
            / right["online"]["main_model"]["estimated_dense_forward_flops"]
        )
    return {name: _mean_std(values) for name, values in ratios.items()}


def _validate_reports(
    decode_reports: Sequence[dict[str, Any]],
    algorithm_reports: Sequence[dict[str, Any]],
) -> None:
    if not decode_reports or not algorithm_reports:
        raise ValueError("at least one decode and algorithm report is required")
    if len(decode_reports) != len(algorithm_reports):
        raise ValueError("decode and algorithm repetitions must have equal length")
    expected = {
        "decode_arms": {
            "baseline",
            "history_tree_static",
            "history_tree_load_aware",
        },
        "algorithm_arms": {
            "conditional_fixed",
            "progressive",
            "progressive_streaming_runahead",
            "smc_no_reuse",
            "smc_reuse",
        },
    }
    all_reports = [*decode_reports, *algorithm_reports]
    reference = deepcopy(all_reports[0]["setting"])
    reference.pop("seed", None)
    gpu = all_reports[0]["machine"].get("gpu")
    for report in all_reports:
        setting = deepcopy(report["setting"])
        setting.pop("seed", None)
        if setting != reference:
            raise ValueError("all repetitions must use the same non-seed setting")
        if report["machine"].get("gpu") != gpu:
            raise ValueError("all repetitions must use the same GPU")
    for reports, section in (
        (decode_reports, "decode_arms"),
        (algorithm_reports, "algorithm_arms"),
    ):
        for report in reports:
            names = [str(arm["name"]) for arm in report[section]]
            if len(names) != len(set(names)) or set(names) != expected[section]:
                raise ValueError(f"{section} is incomplete or contains duplicate arms")
    decode_seeds = {int(report["setting"]["seed"]) for report in decode_reports}
    algorithm_seeds = {
        int(report["setting"]["seed"]) for report in algorithm_reports
    }
    if decode_seeds != algorithm_seeds or len(decode_seeds) != len(decode_reports):
        raise ValueError("decode and algorithm reports need the same unique seeds")


def build_summary(
    decode_reports: Sequence[dict[str, Any]],
    algorithm_reports: Sequence[dict[str, Any]],
    *,
    decode_paths: Sequence[Path] = (),
    algorithm_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    _validate_reports(decode_reports, algorithm_reports)
    return {
        "schema_version": 1,
        "runs": len(decode_reports),
        "machine": decode_reports[0]["machine"],
        "setting": decode_reports[0]["setting"],
        "decode": _arm_metrics(decode_reports, "decode_arms"),
        "algorithm": _arm_metrics(algorithm_reports, "algorithm_arms"),
        "comparisons": {
            "static_tree_over_baseline": _paired_ratios(
                decode_reports,
                "decode_arms",
                "history_tree_static",
                "baseline",
            ),
            "load_aware_tree_over_baseline": _paired_ratios(
                decode_reports,
                "decode_arms",
                "history_tree_load_aware",
                "baseline",
            ),
            "progressive_over_fixed": _paired_ratios(
                algorithm_reports,
                "algorithm_arms",
                "progressive",
                "conditional_fixed",
            ),
            "streaming_runahead_over_progressive": _paired_ratios(
                algorithm_reports,
                "algorithm_arms",
                "progressive_streaming_runahead",
                "progressive",
            ),
            "smc_reuse_over_no_reuse": _paired_ratios(
                algorithm_reports,
                "algorithm_arms",
                "smc_reuse",
                "smc_no_reuse",
            ),
        },
        "source_files": {
            "decode": [str(path) for path in decode_paths],
            "algorithm": [str(path) for path in algorithm_paths],
        },
    }


def _fmt(stats: dict[str, Any], digits: int = 3) -> str:
    return f"{stats['mean']:.{digits}f} ± {stats['sample_std']:.{digits}f}"


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _svg(summary: dict[str, Any], path: Path) -> None:
    decode_order = ["baseline", "history_tree_static", "history_tree_load_aware"]
    algorithm_order = [
        "conditional_fixed",
        "progressive",
        "progressive_streaming_runahead",
        "smc_no_reuse",
        "smc_reuse",
    ]
    labels = {
        "baseline": "无草稿",
        "history_tree_static": "静态草稿",
        "history_tree_load_aware": "负载感知",
        "conditional_fixed": "固定条件 IS",
        "progressive": "Pilot/Eval",
        "progressive_streaming_runahead": "流式+预生成",
        "smc_no_reuse": "SMC 不复用",
        "smc_reuse": "SMC 复用",
    }
    width, height = 1480, 650
    margin_x, top = 90, 105
    panel_width, chart_height = 610, 390
    gap = 105
    colors = ["#64748b", "#ef8354", "#2a9d8f", "#7c3aed", "#2563eb"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfdff"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#14213d}.title{font-size:24px;font-weight:700}.sub{font-size:15px;fill:#52617a}.axis{font-size:13px;fill:#52617a}.value{font-size:13px;font-weight:600}</style>',
        '<text x="90" y="45" class="title">RTX 3090 rollout 基础设施消融</text>',
        '<text x="90" y="73" class="sub">柱高为 3 个独立 seed 的平均在线墙钟时间；误差线为样本标准差</text>',
    ]

    def panel(x0: int, order: list[str], metrics: dict[str, Any], title: str) -> None:
        maximum = max(metrics[name]["online_wall_seconds"]["mean"] for name in order) * 1.18
        parts.append(f'<text x="{x0}" y="{top}" class="title">{_escape(title)}</text>')
        y0 = top + chart_height
        parts.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x0 + panel_width}" y2="{y0}" stroke="#94a3b8"/>'
        )
        for tick in range(5):
            value = maximum * tick / 4
            y = y0 - chart_height * tick / 4
            parts.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" y2="{y:.1f}" stroke="#dbe4ef" stroke-dasharray="4 5"/>'
            )
            parts.append(
                f'<text x="{x0 - 12}" y="{y + 5:.1f}" text-anchor="end" class="axis">{value:.1f}s</text>'
            )
        slot = panel_width / len(order)
        bar_width = min(78, slot * 0.56)
        for index, name in enumerate(order):
            stats = metrics[name]["online_wall_seconds"]
            center = x0 + slot * (index + 0.5)
            bar_height = chart_height * stats["mean"] / maximum
            y = y0 - bar_height
            parts.append(
                f'<rect x="{center - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="5" fill="{colors[index % len(colors)]}" opacity="0.9"/>'
            )
            error = chart_height * stats["sample_std"] / maximum
            parts.append(
                f'<line x1="{center:.1f}" y1="{y - error:.1f}" x2="{center:.1f}" y2="{y + error:.1f}" stroke="#14213d" stroke-width="2"/>'
            )
            parts.append(
                f'<line x1="{center - 8:.1f}" y1="{y - error:.1f}" x2="{center + 8:.1f}" y2="{y - error:.1f}" stroke="#14213d" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{center:.1f}" y="{y - error - 9:.1f}" text-anchor="middle" class="value">{stats["mean"]:.2f}s</text>'
            )
            parts.append(
                f'<text x="{center:.1f}" y="{y0 + 28}" text-anchor="middle" class="axis">{_escape(labels[name])}</text>'
            )

    panel(margin_x, decode_order, summary["decode"], "解码层")
    panel(margin_x + panel_width + gap, algorithm_order, summary["algorithm"], "算法层")
    parts.extend(
        [
            '<text x="90" y="610" class="sub">注意：静态草稿的分母是无草稿；SMC 复用的直接分母是 SMC 不复用。cache build 与后台 drain 未并入柱高。</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _markdown(summary: dict[str, Any], svg_path: Path) -> str:
    decode = summary["decode"]
    algorithm = summary["algorithm"]
    comparisons = summary["comparisons"]
    lines = [
        "# RTX 3090 rollout 基础设施消融",
        "",
        "## 实验设置",
        "",
        "该实验只回答基础设施成本问题，不把单道题的准确率当作方法效果结论。三次运行使用固定的 Qwen2.5-1.5B-Instruct、BF16、公开且校验过哈希的 GSM8K test 第 1311 题、64 token 上限、16 token block，以及相同请求级随机数构造；仅改变被消融的基础设施。解码层依次提交 active batch 4、2、1，模拟 rollout 尾部逐渐变稀。算法层使用 3 个候选、每候选 2 条总 rollout 预算；SMC 使用 3 个粒子、每粒子 2 个分支。",
        "",
        "主模型计算量按 `2 × 参数量 × 实际 target forward token slots` 估算；它覆盖 prefill、decode、评分以及被拒绝草稿的 target 验证，未包含 attention 的长度二次项、CPU token tree、采样和奖励解析。cache build、在线路径和后台 drain 分开报告。奖励使用标准答案 verifier 仅为稳定诊断算法关系，不能视为可部署 setting。",
        "",
        f"![RTX 3090 rollout 基础设施消融]({svg_path.as_posix()})",
        "",
        "## 解码层结果",
        "",
        "| 路径 | 在线墙钟时间（s） | 输出 token/s | 主模型 PFLOPs | cache build（s） | 草稿接受率 | 相对无草稿墙钟 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    decode_labels = {
        "baseline": "无草稿",
        "history_tree_static": "历史树，始终草稿",
        "history_tree_load_aware": "历史树，负载感知",
    }
    baseline_wall = decode["baseline"]["online_wall_seconds"]["mean"]
    for name in ("baseline", "history_tree_static", "history_tree_load_aware"):
        item = decode[name]
        acceptance = item["draft_acceptance_fraction"]["mean"]
        lines.append(
            "| "
            + decode_labels[name]
            + f" | {_fmt(item['online_wall_seconds'])}"
            + f" | {_fmt(item['output_tokens_per_second'], 1)}"
            + f" | {_fmt(item['online_pflops'], 5)}"
            + f" | {_fmt(item['cache_build_seconds'])}"
            + f" | {acceptance:.1%}"
            + f" | {item['online_wall_seconds']['mean'] / baseline_wall:.3f}× |"
        )
    lines.extend(
        [
            "",
            "静态草稿相对无草稿更慢，因为低接受率会让 target 一次验证多个随后丢弃的 token；这条对照说明不能把 speculative decoding 默认当作加速。KV 裁剪避免拒绝后重新 prefill，而负载策略在 batch 4 和 2 保留普通批处理、只在 batch 1 长尾启用草稿。其平均在线时间基本回到无草稿基线，同时只增加很少的 target FLOPs。BF16 下单请求验证与批量基线的数值路径不同，因此 token trace 不要求逐条完全相等；理论分布不因草稿改变，FP32 有限状态测试另外验证固定随机流的一致性。",
            "",
            "## 算法层结果",
            "",
            "| 路径 | 在线墙钟时间（s） | 在线主模型 PFLOPs | cache build（s） | 后台 drain（s） | fresh / reused rollout |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    algorithm_labels = {
        "conditional_fixed": "固定 rollout 条件 IS",
        "progressive": "pilot/evaluation 分离",
        "progressive_streaming_runahead": "流式奖励 + run-ahead",
        "smc_no_reuse": "SMC forest，不复用",
        "smc_reuse": "SMC forest，复用",
    }
    for name in (
        "conditional_fixed",
        "progressive",
        "progressive_streaming_runahead",
        "smc_no_reuse",
        "smc_reuse",
    ):
        item = algorithm[name]
        fresh = item.get("fresh_rollouts", {"mean": 0})["mean"]
        reused = item.get("reused_rollouts", {"mean": 0})["mean"]
        lines.append(
            "| "
            + algorithm_labels[name]
            + f" | {_fmt(item['online_wall_seconds'])}"
            + f" | {_fmt(item['online_pflops'], 5)}"
            + f" | {_fmt(item['cache_build_seconds'])}"
            + f" | {_fmt(item['background_drain_seconds'])}"
            + f" | {fresh:.1f} / {reused:.1f} |"
        )
    smc = comparisons["smc_reuse_over_no_reuse"]
    progressive = comparisons["progressive_over_fixed"]
    runahead = comparisons["streaming_runahead_over_progressive"]
    lines.extend(
        [
            "",
            f"pilot/evaluation 分离相对一次性固定 rollout 的在线墙钟因子为 `{progressive['wall_time_factor']['mean']:.3f}×`，FLOPs 因子为 `{progressive['main_model_flops_factor']['mean']:.3f}×`。两阶段必须先完成 pilot 再冻结预算，因此在候选成本近似相同、奖励解析很便宜的这个小 workload 上没有速度收益；它保留的价值是避免自适应预算直接读取 evaluation 值，并在异质 proposal、变长 rollout 或 replay 成本差异明显时重新分配预算。",
            "",
            f"流式奖励 + run-ahead 相对纯 progressive 的在线墙钟因子为 `{runahead['wall_time_factor']['mean']:.3f}×`。本实验的正则数值 verifier 几乎没有 CPU 尾部，后台工作没有足够空泡可隐藏；其 drain 已单列，因此不能把预生成 token 写成免费。默认配置保持 run-ahead 关闭，只有实测 reward/KV 空泡足够大时才开启。",
            "",
            f"SMC rollout forest 的复用版相对不复用版墙钟因子为 `{smc['wall_time_factor']['mean']:.3f}×`，主模型 FLOPs 因子为 `{smc['main_model_flops_factor']['mean']:.3f}×`。这是直接同算法、同粒子和分支 setting 的加速比较；复用来自上一层 lookahead 中与所选子 block 匹配的条件后缀，缺少的粒子仍用 fresh base rollout 补齐。",
            "",
            "## vLLM 复现实验状态",
            "",
            "同一入口支持 `--backend vllm`：常驻 `AsyncLLM` 使用原生 global suffix tree，并从 vLLM 原生计数器读取 drafted/accepted token，把被拒绝的验证 token 加回主模型 FLOPs。active-batch 动态表只在专门的 load-aware arm 中显式启用；算法层默认使用静态 suffix，避免把 vLLM 0.25 的实验性动态调度开销混入算法比较。当前 RTX 3090 位于 Windows 主机，未安装 WSL；vLLM 不原生支持 Windows，因此这里没有伪造 vLLM 数值。安装 WSL2/Linux 环境后可用下方命令生成同 schema 的结果，再交给同一汇总器：",
            "",
            "```bash",
            "export PYTHONPATH=src",
            "python experiments/benchmark_rollout_infra.py \\",
            "  --output results/infra/rtx3090_vllm.json \\",
            "  --backend vllm --dtype bfloat16 --section all",
            "```",
            "",
            "## 结论边界",
            "",
            "- 历史序列进入 draft tree 不等于再次进入统计 estimator；base 会验证每个草稿 token。",
            "- pilot 只冻结 evaluation 预算，pilot reward 不进入最终条件能量均值。",
            "- 墙钟更短不一定意味着 FLOPs 更少：SMC 的大 batch 就可能以更高并行度换取较低墙钟。",
            "- 单题三 seed 的结果用于 infra 消融，不支持质量排序；方法质量仍应读取完整 GSM8K 对照实验。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode", type=Path, nargs="+", required=True)
    parser.add_argument("--algorithm", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    decode_reports = _load(args.decode)
    algorithm_reports = _load(args.algorithm)
    summary = build_summary(
        decode_reports,
        algorithm_reports,
        decode_paths=args.decode,
        algorithm_paths=args.algorithm,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _svg(summary, args.output_svg)
    # Store a repository-relative chart path in Markdown.
    try:
        relative_svg = args.output_svg.relative_to(args.output_markdown.parent)
    except ValueError:
        relative_svg = Path("../assets") / args.output_svg.name
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(
        _markdown(summary, relative_svg), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
