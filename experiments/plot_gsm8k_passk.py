"""Render the combined GSM8K pass@k report as an SVG figure."""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "gsm8k_3090_aligned_passk_comparison_validated.json"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "gsm8k_3090_aligned_passk.svg"

WIDTH = 1280
HEIGHT = 720
PLOT_TOP = 158
PLOT_BOTTOM = 580
PANEL_WIDTH = 500
PANEL_LEFTS = (82, 700)
Y_MIN = 0.2
Y_MAX = 1.0
KS = (1, 2, 4, 8)


@dataclass(frozen=True)
class SeriesStyle:
    label: str
    color: str
    shape: str


STYLES = {
    "base": SeriesStyle("Base", "#4b5563", "circle"),
    "mh": SeriesStyle("幂分布 MH", "#ea580c", "triangle"),
    "rl_sample": SeriesStyle("GRPO", "#be123c", "square"),
    "conditional_is": SeriesStyle("标准条件 IS", "#15803d", "circle"),
    "conditional_is_small_proposal": SeriesStyle(
        "小 proposal（截断）", "#7e22ce", "diamond"
    ),
    "conditional_is_small_proposal_unclipped": SeriesStyle(
        "小 proposal（不截断）", "#2563eb", "triangle"
    ),
}

PANELS = (
    (
        "采样分布锐化与训练后策略",
        "Base、幂分布 MH 与 GRPO",
        ("base", "mh", "rl_sample"),
    ),
    (
        "条件重要性采样",
        "候选均来自 base；只改变 rollout proposal 与权重截断",
        (
            "conditional_is",
            "conditional_is_small_proposal",
            "conditional_is_small_proposal_unclipped",
        ),
    ),
)


def _x(k: int, panel_left: int) -> float:
    inset = 38
    return panel_left + inset + math.log2(k) / 3 * (PANEL_WIDTH - 2 * inset)


def _y(value: float) -> float:
    return PLOT_BOTTOM - (value - Y_MIN) / (Y_MAX - Y_MIN) * (
        PLOT_BOTTOM - PLOT_TOP
    )


def _marker(shape: str, color: str, x: float, y: float) -> str:
    common = f'fill="{color}" stroke="#ffffff" stroke-width="1.5"'
    if shape == "square":
        return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" {common}/>'
    if shape == "diamond":
        points = (
            f"{x:.1f},{y - 7:.1f} {x + 7:.1f},{y:.1f} "
            f"{x:.1f},{y + 7:.1f} {x - 7:.1f},{y:.1f}"
        )
        return f'<polygon points="{points}" {common}/>'
    if shape == "triangle":
        points = (
            f"{x:.1f},{y - 7:.1f} {x + 7:.1f},{y + 6:.1f} "
            f"{x - 7:.1f},{y + 6:.1f}"
        )
        return f'<polygon points="{points}" {common}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" {common}/>'


def _series_values(summary: dict[str, Any]) -> list[tuple[int, float, float, float]]:
    values = []
    for k in KS:
        key = str(k)
        estimate = float(summary["estimated_pass_at_k"][key])
        interval = summary["estimated_pass_at_k_problem_bootstrap_95"][key]
        values.append((k, estimate, float(interval[0]), float(interval[1])))
    return values


def _panel(
    payload: dict[str, Any],
    panel_left: int,
    title: str,
    subtitle: str,
    methods: tuple[str, ...],
) -> list[str]:
    right = panel_left + PANEL_WIDTH
    lines = [
        f'<text class="panel-title" x="{panel_left}" y="82">{html.escape(title)}</text>',
        f'<text class="panel-subtitle" x="{panel_left}" y="104">{html.escape(subtitle)}</text>',
        f'<rect class="frame" x="{panel_left}" y="{PLOT_TOP}" width="{PANEL_WIDTH}" height="{PLOT_BOTTOM - PLOT_TOP}"/>',
    ]

    legend_x = panel_left
    for method in methods:
        style = STYLES[method]
        marker_x = legend_x + 7
        lines.append(_marker(style.shape, style.color, marker_x, 130))
        lines.append(
            f'<text class="legend" x="{marker_x + 12}" y="134">{html.escape(style.label)}</text>'
        )
        legend_x += 155 if len(style.label) <= 9 else 180

    for tick in (0.2, 0.4, 0.6, 0.8, 1.0):
        y = _y(tick)
        lines.extend(
            [
                f'<line class="grid" x1="{panel_left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                f'<text class="tick" x="{panel_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick * 100:.0f}%</text>',
            ]
        )
    for k in KS:
        x = _x(k, panel_left)
        lines.extend(
            [
                f'<line class="grid vertical" x1="{x:.1f}" y1="{PLOT_TOP}" x2="{x:.1f}" y2="{PLOT_BOTTOM}"/>',
                f'<text class="tick" x="{x:.1f}" y="{PLOT_BOTTOM + 23}" text-anchor="middle">{k}</text>',
            ]
        )

    for method in methods:
        if method not in payload["methods"]:
            raise ValueError(f"combined pass@k report is missing {method}")
        style = STYLES[method]
        values = _series_values(payload["methods"][method])
        points = " ".join(
            f"{_x(k, panel_left):.1f},{_y(estimate):.1f}"
            for k, estimate, _, _ in values
        )
        lines.append(
            f'<polyline class="series" points="{points}" stroke="{style.color}"/>'
        )
        for k, estimate, low, high in values:
            x = _x(k, panel_left)
            y = _y(estimate)
            y_low = _y(low)
            y_high = _y(high)
            accessible = (
                f"{style.label}, pass@{k} {estimate * 100:.1f}%, "
                f"95% 区间 {low * 100:.1f}%–{high * 100:.1f}%"
            )
            lines.extend(
                [
                    f'<g aria-label="{html.escape(accessible)}">',
                    f'<line class="error" x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="{style.color}"/>',
                    f'<line class="error" x1="{x - 4:.1f}" y1="{y_high:.1f}" x2="{x + 4:.1f}" y2="{y_high:.1f}" stroke="{style.color}"/>',
                    f'<line class="error" x1="{x - 4:.1f}" y1="{y_low:.1f}" x2="{x + 4:.1f}" y2="{y_low:.1f}" stroke="{style.color}"/>',
                    _marker(style.shape, style.color, x, y),
                    "</g>",
                ]
            )

    lines.append(
        f'<text class="axis-title" x="{panel_left + PANEL_WIDTH / 2:.1f}" y="{PLOT_BOTTOM + 55}" text-anchor="middle">一次评测允许的独立采样数 k</text>'
    )
    return lines


def render(payload: dict[str, Any]) -> str:
    if int(payload["draws_per_problem"]) != 8:
        raise ValueError("this figure expects eight draws per problem")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">GSM8K 多次采样 pass@k</title>',
        '<desc id="desc">两个面板分别比较 Base、幂分布 MH、GRPO，以及三种条件重要性采样。点为 32 道题上的 pass@k，误差线为题目 bootstrap 95% 区间。</desc>',
        "<style>",
        "text { font-family: 'Noto Sans CJK SC', 'Microsoft YaHei', Arial, sans-serif; fill: #172033; }",
        ".title { font-size: 25px; font-weight: 600; }",
        ".panel-title { font-size: 18px; font-weight: 600; }",
        ".panel-subtitle { font-size: 13px; fill: #536078; }",
        ".legend { font-size: 12px; font-weight: 600; }",
        ".frame { fill: #ffffff; stroke: #aab4c4; stroke-width: 1; }",
        ".grid { stroke: #dbe1ea; stroke-width: 1; }",
        ".grid.vertical { stroke-dasharray: 3 4; }",
        ".tick { font-size: 12px; fill: #536078; }",
        ".axis-title { font-size: 13px; font-weight: 600; }",
        ".series { fill: none; stroke-width: 2.2; }",
        ".error { stroke-width: 1.3; opacity: 0.62; }",
        ".note { font-size: 13px; fill: #536078; }",
        "</style>",
        '<text class="title" x="82" y="43">多次采样：pass@k 与题目级不确定性</text>',
    ]
    for panel_left, (title, subtitle, methods) in zip(
        PANEL_LEFTS, PANELS, strict=True
    ):
        lines.extend(_panel(payload, panel_left, title, subtitle, methods))
    y_mid = (PLOT_TOP + PLOT_BOTTOM) / 2
    lines.extend(
        [
            f'<text class="axis-title" x="22" y="{y_mid:.1f}" text-anchor="middle" transform="rotate(-90 22 {y_mid:.1f})">pass@k</text>',
            '<text class="note" x="82" y="681">每种方法：32 道固定题 × 8 个独立 draw；竖线为题目级 bootstrap 95% 区间。只估计 k≤8。</text>',
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
