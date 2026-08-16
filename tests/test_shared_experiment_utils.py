from __future__ import annotations

import pytest

from experiments.shared.config_overrides import (
    apply_config_overrides,
    parse_override_value,
)
from experiments.shared.statistics import (
    estimated_pass_at_k,
    jensen_shannon_bits,
    probability_distribution,
    total_variation_distance,
    wilson_interval,
)


def test_dotted_config_overrides_are_typed_isolated_and_validated() -> None:
    source = {"generation": {"temperature": 1.0}, "method": {"clip": 10.0}}
    result = apply_config_overrides(
        source,
        ("generation.temperature=0.7", "method.clip=none"),
    )

    assert result == {"generation": {"temperature": 0.7}, "method": {"clip": None}}
    assert source["generation"]["temperature"] == 1.0
    with pytest.raises(KeyError, match="unknown config override"):
        apply_config_overrides(source, ("generation.temperatur=0.7",))
    with pytest.raises(ValueError, match="must contain"):
        apply_config_overrides(source, ("generation.temperature",))


def test_override_parser_accepts_toml_values_and_plain_strings() -> None:
    assert parse_override_value("8") == 8
    assert parse_override_value("[1, 2]") == [1, 2]
    assert parse_override_value("random") == "random"


def test_shared_probability_estimators_have_known_values() -> None:
    assert estimated_pass_at_k(1, 4, 1) == 0.25
    assert estimated_pass_at_k(1, 4, 2) == 0.5
    assert estimated_pass_at_k(1, 4, 4) == 1.0
    interval = wilson_interval(5, 10)
    assert interval[0] < 0.5 < interval[1]

    left = probability_distribution({"a": 2})
    right = probability_distribution({"b": 2})
    assert total_variation_distance(left, right) == 1.0
    assert jensen_shannon_bits(left, right) == 1.0

