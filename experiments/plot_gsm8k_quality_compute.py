"""从正式汇总 JSON 生成 GSM8K 准确率—计算量 SVG。

脚本只依赖 Python 标准库，输出可直接嵌入 Markdown。运行方式：

    python experiments/plot_gsm8k_quality_compute.py
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "gsm8k_3090"
    / "gsm8k_3090_aligned_comparison_validated.json"
)
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "gsm8k_3090_aligned_quality_compute.svg"

WIDTH = 1280
HEIGHT = 720
PLOT_TOP = 126
PLOT_BOTTOM = 574
PANEL_LEFTS = (82, 700)
PANEL_WIDTH = 500
X_MIN = 0.02
X_MAX = 3.2
Y_MIN = 0.2
Y_MAX = 0.95


@dataclass(frozen=True)
class Point:
    method: str
    label: str
    accuracy: float
    interval: tuple[float, float]
    pflops: float
    color: str
    shape: str = "circle"
    dx: int = 10
    dy: int = -9
    anchor: str = "start"


STYLE = {
    "base": ("Base", "#4b5563", "circle", 10, 18, "start"),
    "beam": ("Beam-8", "#2563eb", "square", 10, -12, "start"),
    "best_of_n": ("Best-of-8", "#0891b2", "diamond", -10, -11, "end"),
    "mh": ("幂分布 MH", "#ea580c", "triangle", -10, 19, "end"),
    "conditional_is": ("标准条件 IS", "#15803d", "circle", -10, -12, "end"),
    "conditional_is_small_proposal": (
        "0.5B proposal 条件 IS",
        "#7e22ce",
        "diamond",
        -10,
        20,
        "end",
    ),
    "rl_sample": ("GRPO 随机采样", "#be123c", "triangle", 10, -10, "start"),
    "rl_greedy": ("GRPO 贪心", "#db2777", "square", 10, -10, "start"),
    "verifier_mh": ("verifier-MH", "#ea580c", "triangle", -10, -12, "end"),
    "verifier_conditional_is": (
        "标准 verifier-IS",
        "#15803d",
        "circle",
        -10,
        20,
        "end",
    ),
    "verifier_conditional_is_small_proposal": (
        "0.5B proposal verifier-IS",
        "#7e22ce",
        "diamond",
        -10,
        20,
        "end",
    ),
}


def _point(row: dict[str, object], *, matched: bool = False) -> Point:
    method = str(row["method"])
    label, color, shape, dx, dy, anchor = STYLE[method]
    raw_interval = row["accuracy_wilson_95"]
    if not isinstance(raw_interval, list) or len(raw_interval) != 2:
        raise ValueError(f"{method}: accuracy_wilson_95 必须含两个端点")
    pflops = (
        float(row["estimated_dense_forward_flops"]) / 1e15
        if matched
        else float(row["estimated_dense_forward_petaflops"])
    )
    return Point(
        method=method,
        label=label,
        accuracy=float(row["accuracy"]),
        interval=(float(raw_interval[0]), float(raw_interval[1])),
        pflops=pflops,
        color=color,
        shape=shape,
        dx=dx,
        dy=dy,
        anchor=anchor,
    )


def load_points(path: Path) -> tuple[list[Point], list[Point]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    main = [_point(row) for row in payload["table"]]
    oracle = [_point(row, matched=True) for row in payload["matched_target_comparison"]["table"]]

    rl_row = next(row for row in payload["table"] if row["method"] == "rl_sample")
    oracle.append(_point(rl_row))
    return main, oracle


def _x(value: float, panel_left: int) -> float:
    fraction = (math.log10(value) - math.log10(X_MIN)) / (
        math.log10(X_MAX) - math.log10(X_MIN)
    )
    return panel_left + fraction * PANEL_WIDTH


def _y(value: float) -> float:
    fraction = (value - Y_MIN) / (Y_MAX - Y_MIN)
    return PLOT_BOTTOM - fraction * (PLOT_BOTTOM - PLOT_TOP)


def _marker(point: Point, x: float, y: float) -> str:
    common = f'fill="{point.color}" stroke="#ffffff" stroke-width="1.5"'
    if point.shape == "square":
        return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" {common}/>'
    if point.shape == "diamond":
        coords = f"{x:.1f},{y - 7:.1f} {x + 7:.1f},{y:.1f} {x:.1f},{y + 7:.1f} {x - 7:.1f},{y:.1f}"
        return f'<polygon points="{coords}" {common}/>'
    if point.shape == "triangle":
        coords = f"{x:.1f},{y - 7:.1f} {x + 7:.1f},{y + 6:.1f} {x - 7:.1f},{y + 6:.1f}"
        return f'<polygon points="{coords}" {common}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" {common}/>'


def _panel(points: Iterable[Point], panel_left: int, title: str, subtitle: str) -> list[str]:
    right = panel_left + PANEL_WIDTH
    lines = [
        f'<text class="panel-title" x="{panel_left}" y="88">{html.escape(title)}</text>',
        f'<text class="panel-subtitle" x="{panel_left}" y="109">{html.escape(subtitle)}</text>',
        f'<rect class="frame" x="{panel_left}" y="{PLOT_TOP}" width="{PANEL_WIDTH}" height="{PLOT_BOTTOM - PLOT_TOP}"/>',
    ]

    for tick in (0.2, 0.4, 0.6, 0.8):
        y = _y(tick)
        lines.extend(
            [
                f'<line class="grid" x1="{panel_left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                f'<text class="tick" x="{panel_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick * 100:.0f}%</text>',
            ]
        )

    for tick in (0.03, 0.1, 0.3, 1.0, 3.0):
        x = _x(tick, panel_left)
        label = f"{tick:g}"
        lines.extend(
            [
                f'<line class="grid vertical" x1="{x:.1f}" y1="{PLOT_TOP}" x2="{x:.1f}" y2="{PLOT_BOTTOM}"/>',
                f'<text class="tick" x="{x:.1f}" y="{PLOT_BOTTOM + 23}" text-anchor="middle">{label}</text>',
            ]
        )

    for point in points:
        x = _x(point.pflops, panel_left)
        y = _y(point.accuracy)
        low = _y(point.interval[0])
        high = _y(point.interval[1])
        label = f"{point.label}  {point.accuracy * 100:.1f}%"
        lines.extend(
            [
                f'<g aria-label="{html.escape(label)}, {point.pflops:.4f} PFLOPs">',
                f'<line class="error" x1="{x:.1f}" y1="{high:.1f}" x2="{x:.1f}" y2="{low:.1f}" stroke="{point.color}"/>',
                f'<line class="error" x1="{x - 4:.1f}" y1="{high:.1f}" x2="{x + 4:.1f}" y2="{high:.1f}" stroke="{point.color}"/>',
                f'<line class="error" x1="{x - 4:.1f}" y1="{low:.1f}" x2="{x + 4:.1f}" y2="{low:.1f}" stroke="{point.color}"/>',
                _marker(point, x, y),
                f'<text class="point-label" x="{x + point.dx:.1f}" y="{y + point.dy:.1f}" text-anchor="{point.anchor}">{html.escape(label)}</text>',
                "</g>",
            ]
        )

    axis_mid = panel_left + PANEL_WIDTH / 2
    lines.append(
        f'<text class="axis-title" x="{axis_mid:.1f}" y="{PLOT_BOTTOM + 55}" text-anchor="middle">估算稠密前向计算量（PFLOPs，对数轴）</text>'
    )
    return lines


def render(main: list[Point], oracle: list[Point]) -> str:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">GSM8K 单次回答准确率与推理计算量</title>',
        '<desc id="desc">左图比较可部署或不读取标准答案的方法；右图比较读取标准答案的共享目标诊断。点为 32 道题准确率，误差线为 Wilson 95% 区间，横轴为对数尺度 PFLOPs。</desc>',
        "<style>",
        "text { font-family: 'Noto Sans CJK SC', 'Microsoft YaHei', Arial, sans-serif; fill: #172033; }",
        ".title { font-size: 25px; font-weight: 600; }",
        ".panel-title { font-size: 18px; font-weight: 600; }",
        ".panel-subtitle { font-size: 13px; fill: #536078; }",
        ".frame { fill: #ffffff; stroke: #aab4c4; stroke-width: 1; }",
        ".grid { stroke: #dbe1ea; stroke-width: 1; }",
        ".grid.vertical { stroke-dasharray: 3 4; }",
        ".tick { font-size: 12px; fill: #536078; }",
        ".axis-title { font-size: 13px; font-weight: 600; }",
        ".point-label { font-size: 12px; font-weight: 600; paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; }",
        ".error { stroke-width: 1.5; opacity: 0.72; }",
        ".note { font-size: 13px; fill: #536078; }",
        "</style>",
        '<text id="main-title" class="title" x="82" y="43">质量—计算量：同一 32 题上的当前结果</text>',
    ]
    lines.extend(
        _panel(
            main,
            PANEL_LEFTS[0],
            "可部署与自一致性比较",
            "不读取测试集标准答案；不同方法的目标并不完全相同",
        )
    )
    lines.extend(
        _panel(
            oracle,
            PANEL_LEFTS[1],
            "共享精确奖励目标诊断",
            "verifier 方法读取标准答案，仅用于检验算法关系",
        )
    )
    y_mid = (PLOT_TOP + PLOT_BOTTOM) / 2
    lines.extend(
        [
            f'<text class="axis-title" x="22" y="{y_mid:.1f}" text-anchor="middle" transform="rotate(-90 22 {y_mid:.1f})">单次回答准确率</text>',
            '<text class="note" x="82" y="681">点：准确率；竖线：Wilson 95% 区间。FLOPs = 2 × 模型参数量 × 实际 forward token slots；不等同于墙钟时间。</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_points, oracle_points = load_points(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(main_points, oracle_points), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
