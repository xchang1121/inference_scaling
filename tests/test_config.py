import pytest

from inference_scaling.config import BaseReplayConfig, DynamicISConfig, MHConfig, SamplingConfig


def test_sampling_config_identifies_actual_policy() -> None:
    config = SamplingConfig(temperature=0.7, top_p=0.9, top_k=20, eos_token_id=2)
    assert config.policy_id == "temperature=0.7;top_p=0.9;top_k=20;eos=2"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SamplingConfig(temperature=0),
        lambda: SamplingConfig(top_p=1.1),
        lambda: MHConfig(total_length=4, block_size=8),
        lambda: BaseReplayConfig(fresh_rollouts=0),
        lambda: DynamicISConfig(auxiliary_mixture=1.0),
    ],
)
def test_invalid_configs_fail_early(factory) -> None:
    with pytest.raises(ValueError):
        factory()
