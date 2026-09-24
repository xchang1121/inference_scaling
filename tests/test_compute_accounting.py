from inference_scaling.app.records import pass_at_k
from inference_scaling.shared.compute import (
    estimate_grpo_compute,
    estimate_grpo_compute_from_logs,
)


def test_grpo_compute_is_split_into_observed_forward_equivalents() -> None:
    compute = estimate_grpo_compute(
        model_sequence_tokens=1_000,
        generated_completions=10,
        total_parameters=100,
        trainable_parameters=5,
        optimizer_steps=2,
        gradient_checkpointing=True,
        reference_scoring=True,
    )

    assert compute.rollout_generation_forward_token_slots == 990
    assert compute.reference_scoring_forward_token_slots == 1_000
    assert compute.policy_forward_backward_equivalent_token_slots == 3_000
    assert compute.total_forward_equivalent_token_slots == 4_990
    assert compute.estimated_dense_model_flops == 998_000
    assert compute.estimated_optimizer_flops == 100
    assert compute.estimated_total_flops == 998_100


def test_grpo_logs_reconstruct_padded_generation_and_training_slots() -> None:
    compute = estimate_grpo_compute_from_logs(
        log_history=[
            {
                "step": 1,
                "num_tokens": 40,
                "completions/mean_length": 5,
                "completions/max_length": 6,
            },
            {
                "step": 2,
                "num_tokens": 90,
                "completions/mean_length": 6,
                "completions/max_length": 8,
            },
        ],
        sequences_per_optimizer_step=4,
        generated_completions=8,
        total_parameters=100,
        trainable_parameters=5,
        optimizer_steps=2,
        gradient_checkpointing=True,
        reference_scoring=True,
    )

    assert compute.rollout_generation_forward_token_slots == 94
    assert compute.reference_scoring_forward_token_slots == 102
    assert compute.policy_forward_backward_equivalent_token_slots == 306
    assert compute.total_forward_equivalent_token_slots == 502
    assert compute.estimated_dense_model_flops == 100_400
    assert compute.estimated_optimizer_flops == 100
    assert compute.estimated_total_flops == 100_500
    assert compute.accounting_basis.startswith("padded forward token slots")


def test_standard_pass_at_k_estimator() -> None:
    assert pass_at_k(correct=1, draws=4, k=1) == 0.25
    assert pass_at_k(correct=1, draws=4, k=2) == 0.5
    assert pass_at_k(correct=1, draws=4, k=4) == 1.0
