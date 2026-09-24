from experiments.shared.methods import (
    AR_ASYNC_METHODS,
    AR_METHODS,
    AR_DEFAULT_METHODS,
    AR_PAIRED_METHODS,
    DLLM_METHODS,
    METHOD_REGISTRY,
    METHOD_SPECS,
    method_spec,
    methods_for,
)


def test_method_registry_has_unique_family_scoped_names():
    assert len(METHOD_REGISTRY) == len(METHOD_SPECS)
    assert AR_METHODS == methods_for("arllm", "quality")
    assert set(AR_ASYNC_METHODS) <= set(AR_METHODS)
    assert "iterated_conditional_is" in AR_METHODS
    assert "iterated_conditional_is" not in AR_DEFAULT_METHODS
    assert "iterated_conditional_is" not in AR_PAIRED_METHODS
    assert set(DLLM_METHODS) == {
        spec.name for spec in METHOD_SPECS if spec.family == "dllm"
    }


def test_method_requirements_are_explicit():
    assert method_spec(
        "arllm", "conditional_is_small_proposal"
    ).requires_proposal
    assert method_spec("arllm", "rl_sample").requires_adapter
    assert method_spec(
        "dllm", "conditional_is_reduced_layer_proposal"
    ).requires_proposal
    assert method_spec("dllm", "vrpo_sample").requires_adapter


def test_archived_methods_are_selected_only_by_name():
    from experiments.arllm.gsm8k_reproduction import METHODS
    from experiments.shared.methods import AR_ARCHIVED_METHODS

    assert {"block_conditional_is", "verifier_block_conditional_is"} <= set(AR_ARCHIVED_METHODS)
    for name in ("block_conditional_is", "verifier_block_conditional_is"):
        assert method_spec("arllm", name).components == frozenset()
    assert method_spec("arllm", "conditional_is_small_proposal").archived
    assert not method_spec("arllm", "conditional_is").archived
    assert set(AR_ARCHIVED_METHODS) <= set(METHODS)
