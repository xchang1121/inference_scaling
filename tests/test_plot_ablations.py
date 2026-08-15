from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.plot_gsm8k_ablations import load_series, render


ROOT = Path(__file__).resolve().parents[1]


def test_ablation_svg_loads_formal_report_and_is_valid_xml() -> None:
    payload = json.loads(
        (
            ROOT
            / "results"
            / "gsm8k_3090"
            / "gsm8k_3090_aligned_ablations_validated.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    candidate, guidance, mh_steps, length = load_series(payload)
    assert [len(series.points) for series in candidate] == [4, 4]
    assert len(guidance[0].points) == 4
    assert len(mh_steps[0].points) == 4
    assert [len(series.points) for series in length] == [3] * 6

    svg = render(payload)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    for label in ("M=3", "S=16", "U=10", "GRPO 参数 + 贪心解码"):
        assert label in svg
    assert svg.count('class="series"') == 10
