import pytest

from inference_scaling.arllm.algorithms.config import ConditionalISConfig, PowerMHConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import SequenceSample


def test_sampling_config_identifies_actual_policy() -> None:
    config = SamplingConfig(temperature=0.7, top_p=0.9, top_k=20, eos_token_id=2)
    assert config.policy_id == "temperature=0.7;top_p=0.9;top_k=20;eos=2"


def test_policy_id_preserves_distinct_float_values() -> None:
    assert (
        SamplingConfig(temperature=1.0000001).policy_id
        != SamplingConfig(temperature=1.0000002).policy_id
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SamplingConfig(temperature=0),
        lambda: SamplingConfig(top_p=1.1),
        lambda: PowerMHConfig(total_length=4, block_size=8),
        lambda: PowerMHConfig(suffix_schedule="unknown"),
        lambda: SamplingConfig(temperature=float("nan")),
        lambda: SamplingConfig(top_p=float("inf")),
        lambda: ConditionalISConfig(reward_temperature=float("inf")),
        lambda: ConditionalISConfig(block_size=8, total_length=4),
    ],
)
def test_invalid_configs_fail_early(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_sampled_token_logprob_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        SequenceSample((), (1,), (float("nan"),), "policy", "model", "request")
