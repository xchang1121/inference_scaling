from __future__ import annotations

import xml.etree.ElementTree as ET

from experiments.plot_gsm8k_passk import KS, STYLES, render


def _summary(offset: float) -> dict:
    estimates = {
        str(k): min(0.98, 0.4 + offset + 0.06 * index)
        for index, k in enumerate(KS)
    }
    return {
        "estimated_pass_at_k": estimates,
        "estimated_pass_at_k_problem_bootstrap_95": {
            key: [max(0.0, value - 0.1), min(1.0, value + 0.1)]
            for key, value in estimates.items()
        },
    }


def test_passk_svg_contains_all_methods_and_valid_xml() -> None:
    payload = {
        "draws_per_problem": 8,
        "methods": {
            method: _summary(index * 0.03)
            for index, method in enumerate(STYLES)
        },
    }
    svg = render(payload)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    for style in STYLES.values():
        assert style.label in svg
    assert svg.count('class="series"') == 6
