"""Aggregate the RTX 3090 IS/MH rollout-reuse ablations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Sequence


EXPECTED_ARMS = {
    "broker_arms": {
        "discard_partial_rollouts",
        "resume_partial_rollouts",
    },
    "streaming_is_arms": {
        "wait_batch_then_verify_cheap",
        "stream_completion_into_is_cheap",
        "wait_batch_then_verify_delayed",
        "stream_completion_into_is_delayed",
    },
    "stochastic_draft_arms": {
        "no_history_draft",
        "deterministic_history_draft",
        "stochastic_history_draft_exact",
    },
    "mh_prefetch_arms": {
        "ordinary_mh_cheap_reward",
        "proposal_tree_prefetch_cheap_reward",
        "ordinary_mh_delayed_reward",
        "proposal_tree_prefetch_delayed_reward",
    },
    "mh_delayed_acceptance_arms": {
        "ordinary_mh_expensive_reward",
        "delayed_acceptance_exact",
    },
    "mh_replay_proposal_arms": {
        "base_suffix_proposal",
        "frozen_replay_mixture_proposal",
    },
}


def _load(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _mean_std(values: Sequence[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    return {
        "mean": statistics.mean(numbers),
        "sample_std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "runs": len(numbers),
    }


def _arms(report: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    return {str(arm["name"]): arm for arm in report[section]}


def _wall(arm: dict[str, Any]) -> float:
    return float(arm["online"]["telemetry"]["wall_seconds"])


def _flops(arm: dict[str, Any]) -> float:
    return float(arm["online"]["main_model"]["estimated_dense_forward_flops"])


def _cache_wall(arm: dict[str, Any]) -> float:
    return float(arm.get("cache_build", {}).get("telemetry", {}).get("wall_seconds", 0.0))


def _cache_flops(arm: dict[str, Any]) -> float:
    return float(
        arm.get("cache_build", {})
        .get("main_model", {})
        .get("estimated_dense_forward_flops", 0.0)
    )


def _ratio(
    reports: Sequence[dict[str, Any]],
    section: str,
    numerator: str,
    denominator: str,
    metric: Callable[[dict[str, Any]], float],
) -> dict[str, float | int]:
    values = []
    for report in reports:
        arms = _arms(report, section)
        values.append(metric(arms[numerator]) / metric(arms[denominator]))
    return _mean_std(values)


def _paired(
    reports: Sequence[dict[str, Any]],
    section: str,
    numerator: str,
    denominator: str,
    *,
    extra: dict[str, Callable[[dict[str, Any]], float]] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Callable[[dict[str, Any]], float]] = {
        "wall_time_factor": _wall,
        "main_model_flops_factor": _flops,
    }
    metrics.update(extra or {})
    return {
        name: _ratio(reports, section, numerator, denominator, metric)
        for name, metric in metrics.items()
    }


def _aggregate_arms(
    reports: Sequence[dict[str, Any]], section: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for arm in report[section]:
            grouped[str(arm["name"])].append(arm)
    output: dict[str, Any] = {}
    for name, arms in grouped.items():
        values: dict[str, list[float]] = defaultdict(list)
        for arm in arms:
            online = arm["online"]
            model = online["main_model"]
            telemetry = online["telemetry"]
            values["online_wall_seconds"].append(float(telemetry["wall_seconds"]))
            values["online_pflops"].append(
                float(model["estimated_dense_forward_flops"]) / 1e15
            )
            values["generated_tokens"].append(float(model.get("generated_tokens", 0)))
            values["scored_tokens"].append(float(model.get("scored_tokens", 0)))
            values["prefill_tokens"].append(float(model.get("prefill_tokens", 0)))
            values["cache_build_seconds"].append(_cache_wall(arm))
            values["cache_build_pflops"].append(_cache_flops(arm) / 1e15)
            for field in (
                "mean_gpu_utilization_percent",
                "maximum_gpu_memory_used_mib",
                "mean_gpu_power_watts",
            ):
                if telemetry.get(field) is not None:
                    values[field].append(float(telemetry[field]))
            if "first_estimator_update_seconds" in online:
                values["first_estimator_update_seconds"].append(
                    float(online["first_estimator_update_seconds"])
                )
            draft = online.get("draft_cache", {})
            proposed = float(draft.get("proposed_tokens", 0))
            accepted = float(draft.get("accepted_tokens", 0))
            if proposed:
                values["draft_acceptance_fraction"].append(accepted / proposed)
            for field in (
                "exact_reward_evaluations",
                "surrogate_reward_evaluations",
                "early_rejections",
                "acceptance_rate",
            ):
                if field in online:
                    values[field].append(float(online[field]))
            if "proposal_sources" in online:
                total = sum(float(value) for value in online["proposal_sources"].values())
                values["history_proposal_fraction"].append(
                    float(online["proposal_sources"].get("history", 0)) / total
                    if total
                    else 0.0
                )
        output[name] = {
            field: _mean_std(numbers) for field, numbers in values.items()
        }
    return output


def _validate(reports: Sequence[dict[str, Any]]) -> None:
    if not reports:
        raise ValueError("at least one IS/MH infra report is required")
    reference = deepcopy(reports[0]["setting"])
    reference.pop("seed", None)
    gpu = reports[0]["machine"].get("gpu")
    seeds: set[int] = set()
    for report in reports:
        setting = deepcopy(report["setting"])
        seed = int(setting.pop("seed"))
        if seed in seeds:
            raise ValueError("IS/MH infra reports need unique seeds")
        seeds.add(seed)
        if setting != reference:
            raise ValueError("all IS/MH infra runs must use the same non-seed setting")
        if report["machine"].get("gpu") != gpu:
            raise ValueError("all IS/MH infra runs must use the same GPU")
        for section, expected in EXPECTED_ARMS.items():
            names = [str(arm["name"]) for arm in report[section]]
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError(f"{section} is incomplete or contains duplicate arms")
    for report in reports:
        for section in ("broker_arms", "streaming_is_arms"):
            if any(not arm["online"]["is_estimator"]["complete"] for arm in report[section]):
                raise ValueError("a frozen IS design did not receive every declared sample")
        broker = _arms(report, "broker_arms")
        if any(arm["online"]["records"] != 8 for arm in broker.values()):
            raise ValueError("broker comparison changed the completed rollout count")
        expected_completion_tokens = 20 * int(report["setting"]["chunk_tokens"])
        if any(
            arm["online"]["completion_tokens"] != expected_completion_tokens
            for arm in broker.values()
        ):
            raise ValueError("broker comparison changed the useful completion workload")


def build_summary(
    reports: Sequence[dict[str, Any]], *, paths: Sequence[Path] = ()
) -> dict[str, Any]:
    _validate(reports)
    replay_online = _paired(
        reports,
        "mh_replay_proposal_arms",
        "frozen_replay_mixture_proposal",
        "base_suffix_proposal",
    )
    replay_online["cold_wall_time_factor"] = _mean_std(
        [
            (
                _cache_wall(
                    _arms(report, "mh_replay_proposal_arms")[
                        "frozen_replay_mixture_proposal"
                    ]
                )
                + _wall(
                    _arms(report, "mh_replay_proposal_arms")[
                        "frozen_replay_mixture_proposal"
                    ]
                )
            )
            / _wall(_arms(report, "mh_replay_proposal_arms")["base_suffix_proposal"])
            for report in reports
        ]
    )
    replay_online["cold_main_model_flops_factor"] = _mean_std(
        [
            (
                _cache_flops(
                    _arms(report, "mh_replay_proposal_arms")[
                        "frozen_replay_mixture_proposal"
                    ]
                )
                + _flops(
                    _arms(report, "mh_replay_proposal_arms")[
                        "frozen_replay_mixture_proposal"
                    ]
                )
            )
            / _flops(_arms(report, "mh_replay_proposal_arms")["base_suffix_proposal"])
            for report in reports
        ]
    )
    setting = deepcopy(reports[0]["setting"])
    setting.pop("seed", None)
    return {
        "schema_version": 1,
        "runs": len(reports),
        "machine": reports[0]["machine"],
        "setting": setting,
        "seeds": [int(report["setting"]["seed"]) for report in reports],
        "arms": {
            section: _aggregate_arms(reports, section)
            for section in EXPECTED_ARMS
        },
        "comparisons": {
            "partial_resume_over_discard": _paired(
                reports,
                "broker_arms",
                "resume_partial_rollouts",
                "discard_partial_rollouts",
                extra={
                    "generated_token_factor": lambda arm: float(
                        arm["online"]["main_model"]["generated_tokens"]
                    ),
                    "prefill_token_factor": lambda arm: float(
                        arm["online"]["main_model"]["prefill_tokens"]
                    ),
                },
            ),
            "streaming_cheap_over_wait": _paired(
                reports,
                "streaming_is_arms",
                "stream_completion_into_is_cheap",
                "wait_batch_then_verify_cheap",
                extra={
                    "first_update_time_factor": lambda arm: float(
                        arm["online"]["first_estimator_update_seconds"]
                    )
                },
            ),
            "streaming_delayed_over_wait": _paired(
                reports,
                "streaming_is_arms",
                "stream_completion_into_is_delayed",
                "wait_batch_then_verify_delayed",
                extra={
                    "first_update_time_factor": lambda arm: float(
                        arm["online"]["first_estimator_update_seconds"]
                    )
                },
            ),
            "deterministic_draft_over_no_draft": _paired(
                reports,
                "stochastic_draft_arms",
                "deterministic_history_draft",
                "no_history_draft",
            ),
            "stochastic_draft_over_no_draft": _paired(
                reports,
                "stochastic_draft_arms",
                "stochastic_history_draft_exact",
                "no_history_draft",
            ),
            "prefetch_cheap_over_ordinary": _paired(
                reports,
                "mh_prefetch_arms",
                "proposal_tree_prefetch_cheap_reward",
                "ordinary_mh_cheap_reward",
            ),
            "prefetch_delayed_over_ordinary": _paired(
                reports,
                "mh_prefetch_arms",
                "proposal_tree_prefetch_delayed_reward",
                "ordinary_mh_delayed_reward",
            ),
            "delayed_acceptance_over_ordinary": _paired(
                reports,
                "mh_delayed_acceptance_arms",
                "delayed_acceptance_exact",
                "ordinary_mh_expensive_reward",
                extra={
                    "exact_reward_call_factor": lambda arm: float(
                        arm["online"]["exact_reward_evaluations"]
                    )
                },
            ),
            "replay_proposal_over_base": replay_online,
        },
        "source_files": [str(path) for path in paths],
    }


def _svg(summary: dict[str, Any], path: Path) -> None:
    comparisons = summary["comparisons"]
    rows = [
        ("部分 rollout 续跑", "partial_resume_over_discard"),
        ("流式 IS（0.2 s verifier）", "streaming_delayed_over_wait"),
        ("随机历史草稿", "stochastic_draft_over_no_draft"),
        ("MH 预取（便宜奖励）", "prefetch_cheap_over_ordinary"),
        ("MH 预取（0.2 s 奖励）", "prefetch_delayed_over_ordinary"),
        ("延迟接受", "delayed_acceptance_over_ordinary"),
        ("replay proposal（在线）", "replay_proposal_over_base"),
    ]
    width, height = 1500, 720
    left, top, row_height = 330, 125, 70
    panel_width, gap = 430, 120
    maximum = 3.6
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfdff"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#14213d}.title{font-size:25px;font-weight:700}.sub{font-size:15px;fill:#52617a}.label{font-size:15px}.value{font-size:14px;font-weight:700}.axis{font-size:13px;fill:#64748b}</style>',
        '<text x="70" y="46" class="title">IS / MH rollout 复用：墙钟与主模型 FLOPs 不能混为一谈</text>',
        '<text x="70" y="75" class="sub">3 个独立 seed 的成对因子均值；分母是每行对应的未优化路径，虚线为 1.0×</text>',
        f'<text x="{left + panel_width / 2}" y="105" text-anchor="middle" class="title">墙钟因子</text>',
        f'<text x="{left + panel_width + gap + panel_width / 2}" y="105" text-anchor="middle" class="title">主模型 FLOPs 因子</text>',
    ]
    for panel in range(2):
        x0 = left + panel * (panel_width + gap)
        one = x0 + panel_width / maximum
        parts.append(
            f'<line x1="{one:.1f}" y1="{top - 18}" x2="{one:.1f}" y2="{top + len(rows) * row_height - 18}" stroke="#ef8354" stroke-width="2" stroke-dasharray="6 5"/>'
        )
        for tick in (0, 1, 2, 3):
            x = x0 + panel_width * tick / maximum
            parts.append(f'<text x="{x:.1f}" y="{top + len(rows) * row_height + 12}" text-anchor="middle" class="axis">{tick}×</text>')
    for index, (label, key) in enumerate(rows):
        y = top + index * row_height
        parts.append(f'<text x="{left - 22}" y="{y + 17}" text-anchor="end" class="label">{label}</text>')
        for panel, field in enumerate(("wall_time_factor", "main_model_flops_factor")):
            value = float(comparisons[key][field]["mean"])
            x0 = left + panel * (panel_width + gap)
            bar = min(value, maximum) / maximum * panel_width
            color = (
                "#2a9d8f"
                if value < 0.98
                else "#d65a4a"
                if value > 1.02
                else "#64748b"
            )
            parts.append(f'<rect x="{x0}" y="{y}" width="{bar:.1f}" height="28" rx="5" fill="{color}" opacity="0.88"/>')
            parts.append(f'<text x="{x0 + bar + 9:.1f}" y="{y + 20}" class="value">{value:.3f}×</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    summary = build_summary(_load(args.inputs), paths=args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.svg is not None:
        _svg(summary, args.svg)


if __name__ == "__main__":
    main()
