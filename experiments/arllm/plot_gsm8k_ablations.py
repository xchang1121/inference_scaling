"""从对齐消融汇总生成质量—预算 SVG 图。

脚本只依赖 Python 标准库，直接读取机器可读汇总，避免图表与实验记录分离。
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "gsm8k_3090"
    / "gsm8k_3090_aligned_ablations_validated.json"
)
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "gsm8k_3090_aligned_ablations.svg"

WIDTH = 1280
HEIGHT = 1000
PANEL_LEFTS = (82, 700)
PANEL_WIDTH = 500
PLOT_HEIGHT = 270
ROW_TOPS = (150, 600)


@dataclass(frozen=True)
class Point:
    x: float
    accuracy: float
    interval: tuple[float, float]
    label: str


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    shape: str
    points: tuple[Point, ...]


SERIES_STYLE = {
    "base": ("Base", "#4b5563", "circle"),
    "beam": ("Beam-8", "#2563eb", "square"),
    "best_of_n": ("自一致性-8", "#0891b2", "diamond"),
    "conditional_is": ("标准条件 IS", "#15803d", "circle"),
    "conditional_is_small_proposal": (
        "0.5B rollout proposal 条件 IS",
        "#7e22ce",
        "diamond",
    ),
    "mh": ("幂分布 MH", "#ea580c", "triangle"),
    "rl_greedy": ("GRPO 参数 + 贪心解码", "#be123c", "square"),
}


def _row(group: list[dict[str, Any]], tag: str, method: str | None = None) -> dict[str, Any]:
    matches = [
        row
        for row in group
        if row["tag"] == tag and (method is None or row["method"] == method)
    ]
    if len(matches) != 1:
        raise ValueError(f"期望唯一结果：tag={tag!r}, method={method!r}，实际 {len(matches)} 条")
    return matches[0]


def _point(row: dict[str, Any], x: float, label: str) -> Point:
    interval = row["accuracy_wilson_95"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{row['tag']}: accuracy_wilson_95 必须含两个端点")
    return Point(
        x=x,
        accuracy=float(row["accuracy"]),
        interval=(float(interval[0]), float(interval[1])),
        label=label,
    )


def _total_pflops(row: dict[str, Any]) -> float:
    return (
        float(row["estimated_dense_flops_per_example"])
        * int(row["examples"])
        / 1e15
    )


def _styled(method: str, points: Iterable[Point]) -> Series:
    label, color, shape = SERIES_STYLE[method]
    return Series(label, color, shape, tuple(points))


def load_series(payload: dict[str, Any]) -> tuple[
    tuple[Series, ...], tuple[Series, ...], tuple[Series, ...], tuple[Series, ...]
]:
    groups = payload["groups"]
    curve = groups["quality_compute_curve"]
    candidate_series = []
    candidate_specs = {
        "conditional_is": (
            (3, "budget-conditional_is-m3-k3"),
            (5, "budget-conditional_is-m5-k3"),
            (8, "conditional-reference"),
            (10, "budget-conditional_is-m10-k3"),
        ),
        "conditional_is_small_proposal": (
            (3, "budget-conditional_is_small_proposal-m3-k3"),
            (5, "budget-conditional_is_small_proposal-m5-k3"),
            (8, "conditional-small-proposal-reference"),
            (10, "budget-conditional_is_small_proposal-m10-k3"),
        ),
    }
    for method, specs in candidate_specs.items():
        points = []
        for budget, tag in specs:
            row = _row(curve, tag, method)
            points.append(_point(row, _total_pflops(row), f"M={budget}"))
        candidate_series.append(_styled(method, points))

    guidance = groups["guidance_steps"]
    guidance_points = []
    for steps, tag in (
        (2, "guidance-steps-2"),
        (4, "conditional-reference"),
        (8, "guidance-steps-8"),
        (16, "guidance-steps-16"),
    ):
        row = _row(guidance, tag, "conditional_is")
        guidance_points.append(_point(row, _total_pflops(row), f"S={steps}"))

    power = groups["power_sampling"]
    mh_points = []
    for updates in (1, 2, 5, 10):
        row = _row(power, f"steps-{updates}", "mh")
        mh_points.append(_point(row, _total_pflops(row), f"U={updates}"))

    length = groups["generation_length"]
    length_series = []
    for method in (
        "base",
        "beam",
        "best_of_n",
        "conditional_is",
        "conditional_is_small_proposal",
        "rl_greedy",
    ):
        points = []
        for max_tokens in (128, 256, 512):
            row = _row(length, f"length-{max_tokens}", method)
            points.append(_point(row, float(max_tokens), str(max_tokens)))
        length_series.append(_styled(method, points))

    return (
        tuple(candidate_series),
        (_styled("conditional_is", guidance_points),),
        (_styled("mh", mh_points),),
        tuple(length_series),
    )


def _marker(shape: str, color: str, x: float, y: float) -> str:
    common = f'fill="{color}" stroke="#ffffff" stroke-width="1.4"'
    if shape == "square":
        return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" {common}/>'
    if shape == "diamond":
        coords = (
            f"{x:.1f},{y - 7:.1f} {x + 7:.1f},{y:.1f} "
            f"{x:.1f},{y + 7:.1f} {x - 7:.1f},{y:.1f}"
        )
        return f'<polygon points="{coords}" {common}/>'
    if shape == "triangle":
        coords = (
            f"{x:.1f},{y - 7:.1f} {x + 7:.1f},{y + 6:.1f} "
            f"{x - 7:.1f},{y + 6:.1f}"
        )
        return f'<polygon points="{coords}" {common}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" {common}/>'


def _y(value: float, top: int) -> float:
    return top + PLOT_HEIGHT - value * PLOT_HEIGHT


def _log_domain(series: tuple[Series, ...]) -> tuple[float, float]:
    values = [point.x for item in series for point in item.points]
    low = min(values)
    high = max(values)
    return low / 1.3, high * 1.3


def _x_log(value: float, left: int, domain: tuple[float, float]) -> float:
    low, high = domain
    fraction = (math.log10(value) - math.log10(low)) / (
        math.log10(high) - math.log10(low)
    )
    return left + fraction * PANEL_WIDTH


def _base_panel(left: int, top: int, title: str, subtitle: str) -> list[str]:
    right = left + PANEL_WIDTH
    title_y = top - 78
    lines = [
        f'<text class="panel-title" x="{left}" y="{title_y}">{html.escape(title)}</text>',
        f'<text class="panel-subtitle" x="{left}" y="{title_y + 21}">{html.escape(subtitle)}</text>',
        f'<rect class="frame" x="{left}" y="{top}" width="{PANEL_WIDTH}" height="{PLOT_HEIGHT}"/>',
    ]
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = _y(value, top)
        lines.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value * 100:.0f}%</text>',
            ]
        )
    return lines


def _legend(series: tuple[Series, ...], left: int, y: int, columns: int = 3) -> list[str]:
    lines = []
    column_width = PANEL_WIDTH / columns
    for index, item in enumerate(series):
        row = index // columns
        column = index % columns
        x = left + column * column_width
        marker_x = x + 7
        marker_y = y + row * 21
        lines.extend(
            [
                _marker(item.shape, item.color, marker_x, marker_y),
                f'<text class="legend" x="{marker_x + 12:.1f}" y="{marker_y + 4:.1f}">{html.escape(item.label)}</text>',
            ]
        )
    return lines


def _log_panel(
    series: tuple[Series, ...],
    left: int,
    top: int,
    title: str,
    subtitle: str,
    *,
    point_labels: bool = True,
) -> list[str]:
    lines = _base_panel(left, top, title, subtitle)
    bottom = top + PLOT_HEIGHT
    domain = _log_domain(series)
    for tick in (0.1, 0.2, 0.5, 1.0, 2.0):
        if not domain[0] <= tick <= domain[1]:
            continue
        x = _x_log(tick, left, domain)
        lines.extend(
            [
                f'<line class="grid vertical" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>',
                f'<text class="tick" x="{x:.1f}" y="{bottom + 21}" text-anchor="middle">{tick:g}</text>',
            ]
        )
    if len(series) > 1:
        lines.extend(_legend(series, left, top - 32, columns=2))
    for series_index, item in enumerate(series):
        coords = " ".join(
            f"{_x_log(point.x, left, domain):.1f},{_y(point.accuracy, top):.1f}"
            for point in item.points
        )
        lines.append(f'<polyline class="series" points="{coords}" stroke="{item.color}"/>')
        for point in item.points:
            x = _x_log(point.x, left, domain)
            y = _y(point.accuracy, top)
            y_low = _y(point.interval[0], top)
            y_high = _y(point.interval[1], top)
            accessible = (
                f"{item.label}，{point.label}，准确率 {point.accuracy * 100:.1f}%，"
                f"95% 区间 {point.interval[0] * 100:.1f}%–{point.interval[1] * 100:.1f}%，"
                f"8 题合计 {point.x:.4f} PFLOPs"
            )
            lines.extend(
                [
                    f'<g aria-label="{html.escape(accessible)}">',
                    f'<line class="error" x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="{item.color}"/>',
                    f'<line class="error" x1="{x - 4:.1f}" y1="{y_high:.1f}" x2="{x + 4:.1f}" y2="{y_high:.1f}" stroke="{item.color}"/>',
                    f'<line class="error" x1="{x - 4:.1f}" y1="{y_low:.1f}" x2="{x + 4:.1f}" y2="{y_low:.1f}" stroke="{item.color}"/>',
                    _marker(item.shape, item.color, x, y),
                ]
            )
            if point_labels:
                dy = -10 if series_index == 0 else 19
                lines.append(
                    f'<text class="point-label" x="{x:.1f}" y="{y + dy:.1f}" text-anchor="middle">{html.escape(point.label)}</text>'
                )
            lines.append("</g>")
    lines.append(
        f'<text class="axis-title" x="{left + PANEL_WIDTH / 2:.1f}" y="{bottom + 52}" text-anchor="middle">8 题合计估算稠密前向计算量（PFLOPs，对数轴）</text>'
    )
    return lines


def _length_panel(series: tuple[Series, ...], left: int, top: int) -> list[str]:
    lines = _base_panel(
        left,
        top,
        "生成长度预算",
        "相同 8 道题；每个长度点重新运行全部方法",
    )
    bottom = top + PLOT_HEIGHT
    inset = 48
    x_positions = {
        value: left + inset + index * (PANEL_WIDTH - 2 * inset) / 2
        for index, value in enumerate((128.0, 256.0, 512.0))
    }
    lines.extend(_legend(series, left, top - 34, columns=3))
    for value, x in x_positions.items():
        lines.extend(
            [
                f'<line class="grid vertical" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>',
                f'<text class="tick" x="{x:.1f}" y="{bottom + 21}" text-anchor="middle">{int(value)}</text>',
            ]
        )
    for item in series:
        coords = " ".join(
            f"{x_positions[point.x]:.1f},{_y(point.accuracy, top):.1f}"
            for point in item.points
        )
        lines.append(f'<polyline class="series" points="{coords}" stroke="{item.color}"/>')
        for point in item.points:
            x = x_positions[point.x]
            y = _y(point.accuracy, top)
            y_low = _y(point.interval[0], top)
            y_high = _y(point.interval[1], top)
            accessible = (
                f"{item.label}，最大生成 {int(point.x)} token，准确率 "
                f"{point.accuracy * 100:.1f}%"
            )
            lines.extend(
                [
                    f'<g aria-label="{html.escape(accessible)}">',
                    f'<line class="error light" x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="{item.color}"/>',
                    _marker(item.shape, item.color, x, y),
                    "</g>",
                ]
            )
    lines.append(
        f'<text class="axis-title" x="{left + PANEL_WIDTH / 2:.1f}" y="{bottom + 52}" text-anchor="middle">最大新生成 token 数</text>'
    )
    return lines


def render(payload: dict[str, Any]) -> str:
    candidate, guidance, mh_steps, length = load_series(payload)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">GSM8K 单卡消融：预算、质量与计算量</title>',
        '<desc id="desc">四个面板展示候选数、引导阶段、MH 每阶段更新次数和最大生成长度对准确率及估算计算量的影响。每个点使用相同的八道固定题，误差线为 Wilson 95% 区间。</desc>',
        "<style>",
        "text { font-family: 'Noto Sans CJK SC', 'Microsoft YaHei', Arial, sans-serif; fill: #172033; }",
        ".title { font-size: 25px; font-weight: 600; }",
        ".panel-title { font-size: 18px; font-weight: 600; }",
        ".panel-subtitle { font-size: 13px; fill: #536078; }",
        ".legend { font-size: 11px; font-weight: 600; }",
        ".frame { fill: #ffffff; stroke: #aab4c4; stroke-width: 1; }",
        ".grid { stroke: #dbe1ea; stroke-width: 1; }",
        ".grid.vertical { stroke-dasharray: 3 4; }",
        ".tick { font-size: 12px; fill: #536078; }",
        ".axis-title { font-size: 13px; font-weight: 600; }",
        ".series { fill: none; stroke-width: 2.1; }",
        ".error { stroke-width: 1.3; opacity: 0.58; }",
        ".error.light { opacity: 0.25; }",
        ".point-label { font-size: 11px; font-weight: 600; paint-order: stroke; stroke: #ffffff; stroke-width: 3px; }",
        ".note { font-size: 13px; fill: #536078; }",
        "</style>",
        '<text class="title" x="82" y="43">GSM8K 质量—预算消融（8 题）</text>',
    ]
    lines.extend(
        _log_panel(
            candidate,
            PANEL_LEFTS[0],
            ROW_TOPS[0],
            "候选数 M（每个候选 3 条 rollout）",
            "标准 rollout 与 0.5B off-policy rollout 的直接比较",
        )
    )
    lines.extend(
        _log_panel(
            guidance,
            PANEL_LEFTS[1],
            ROW_TOPS[0],
            "标准条件 IS 的引导阶段数 S",
            "S=4/8/16 均为 6/8，计算量继续增加",
        )
    )
    lines.extend(
        _log_panel(
            mh_steps,
            PANEL_LEFTS[0],
            ROW_TOPS[1],
            "MH 每个长度阶段的更新次数 U",
            "更多更新提高本轮点估计，但后续每题增益成本更高",
        )
    )
    lines.extend(_length_panel(length, PANEL_LEFTS[1], ROW_TOPS[1]))
    lines.extend(
        [
            '<text class="axis-title" x="22" y="510" text-anchor="middle" transform="rotate(-90 22 510)">单次生成准确率</text>',
            '<text class="note" x="82" y="980">所有点均为同一组 8 道固定题；竖线为 Wilson 95% 区间，区间重叠较大。FLOPs 使用 2 × 参数量 × 实际 forward token slots 的线性主导项估算。</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(payload), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
