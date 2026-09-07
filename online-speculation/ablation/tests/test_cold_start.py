from dataclasses import replace
import copy

import pytest
import torch

from blockspec.parallel import DualViewConfig, DualViewDecoder
from blockspec.parallel.fitting import FitConfig
from blockspec.parallel.generation import Generation
from blockspec_ablation import cold_start as cold


def model():
    torch.manual_seed(442)
    return DualViewDecoder(DualViewConfig(vocab_size=17, hidden_size=16, intermediate_size=24,
                                          num_hidden_layers=2, num_attention_heads=2,
                                          num_key_value_heads=1, head_dim=8)).eval().requires_grad_(False)


def fit():
    return FitConfig(steps=8, warmup_steps=0, sequence_length=12, anchors_per_sequence=2, learning_rate=.001)


def test_time_credit_charges_setup_and_repays_estimation_debt():
    budget = cold.TimeBudget(.01)
    budget.charge("setup", .2)
    budget.delivered(25.)
    assert budget.credit == pytest.approx(.05)
    assert not budget.admits(.051)
    assert budget.admits(.049)
    budget.charge("training", .08)
    assert budget.summary()["debt_seconds"] == pytest.approx(.03)
    assert not budget.admits(.001)
    budget.delivered(20.)
    assert budget.admits(.16)
    assert budget.summary()["extra_over_service"] < .01


@pytest.mark.parametrize("fraction", [0, -.01, float("nan"), float("inf"), .2])
def test_invalid_budget(fraction):
    with pytest.raises(ValueError):
        cold.TimeBudget(fraction)


def test_clean_replay_owns_records_and_samples_complete_blocks():
    replay = cold.CleanReplay(2, 12, 4, 731)
    tokens = torch.arange(15)[None]
    replay.append(tokens)
    tokens.zero_()
    replay.append(torch.tensor([[3, 5, 7]]))
    assert len(replay.records) == 1
    assert int(replay.records[0].max()) == 14
    for sequence, anchors in replay.batch(8, 3):
        assert sequence.shape == (1, 12)
        assert (sequence[:, 1:] - sequence[:, :-1] == 1).all()
        assert ((anchors >= 0) & (anchors <= 8)).all()
    replay.append(torch.full((1, 4), 8))
    replay.append(torch.full((1, 6), 9))
    assert len(replay.records) == 2 and int(replay.records[0].max()) == 8


def test_cold_copy_and_serving_isolation_with_candidate_rollback():
    decoder = model()
    with torch.no_grad():
        for layer in decoder.layers:
            layer.attention.draft.q.weight.add_(1.)
    learner = cold.FullBlockLearner(decoder, fit())
    for layer in decoder.layers:
        for name, p in layer.attention.draft.named_parameters():
            ar = dict(layer.attention.ar.named_parameters())[name]
            assert torch.equal(p, ar) and p.data_ptr() != ar.data_ptr()
    original = {name: p.clone() for name, p in decoder.named_parameters()}
    learner.step([(torch.tensor([[2, 3, 5, 7, 4, 6, 8, 4, 5, 8, 2, 3]]), torch.tensor([[1, 6]]))])
    assert all(torch.equal(p, original[name]) for name, p in decoder.named_parameters())
    assert any(not torch.equal(p, original[name]) for name, p in learner.master.items())
    with pytest.raises(RuntimeError, match="probe failure"), learner.candidate():
        assert learner.serving_step == 1
        assert all(torch.equal(p, learner.master[name]) for name, p in learner.execution.items())
        raise RuntimeError("probe failure")
    assert learner.serving_step == 0
    assert all(torch.equal(p, original[name]) for name, p in decoder.named_parameters())
    learner.publish()
    assert learner.serving_step == 1
    learner.check()
    assert all(torch.equal(p, original[name]) for name, p in decoder.named_parameters()
               if ".attention.draft." not in name)


def test_frozen_ar_changes_are_detected():
    learner = cold.FullBlockLearner(model(), fit())
    with torch.no_grad():
        learner.model.layers[0].attention.ar.q.weight.add_(.01)
    with pytest.raises(RuntimeError, match="remain fixed"):
        learner.check()


def test_identical_replay_has_identical_cold_and_offline_updates():
    first = cold.FullBlockLearner(model(), fit())
    second = cold.FullBlockLearner(model(), fit())
    replay = cold.CleanReplay(8, 12, 4, 732)
    transcript = []
    for i in range(4):
        replay.append((torch.arange(16)[None] + i) % 15 + 2)
        batches = replay.batch(2, 2)
        transcript.append(copy.deepcopy(batches))
        first.step(batches)
    for batches in transcript:
        second.step(batches)
    assert all(torch.equal(p, second.master[name]) for name, p in first.master.items())
    assert first.steps == second.steps == 4


def test_service_gate_reserves_budget_and_keeps_cold_drafts_out_of_serving(monkeypatch):
    ticks = [0.]
    monkeypatch.setattr(cold.time, "perf_counter", lambda: ticks[0])
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(), cold.ServiceConfig(probe_every=1, probe_tokens=2),
                                    gate_prompts=[prompt], retain_batches=True)
    used = []

    def deliver(prompt, tokens, *, speculative, seed):
        used.append(speculative)
        ticks[0] += 10.
        return Generation(tokens=[4, 5, 6, 7], seconds=10.)

    monkeypatch.setattr(service, "_generate", deliver)
    service.step_estimate = .01
    _, record = service.serve(prompt, 4, seed=1)
    assert service.learner.steps == 1 and service.next_probe == 1
    assert record["mode"] == "ar" and used == [False]
    assert len(service.transcript) == 1
    # A due trial accumulates credit rather than spending it on more updates.
    service.serve(prompt, 4, seed=2)
    assert service.learner.steps == 1 and used == [False, False]
    service.budget.delivered(100000.)
    service.tokens += 40000
    monkeypatch.setattr(service, "_probe", lambda: {"passed": True, "ratio": 2., "step": 1})
    service.serve(prompt, 4, seed=3)
    assert service.speculating and service.learner.serving_step == 1
    service.serve(prompt, 4, seed=4)
    assert used[-1] is True
    assert service.learner.steps == 2
    released_version = service.learner.serving_step
    monkeypatch.setattr(service, "_probe", lambda: {"passed": False, "fallback": True, "ratio": .9, "step": 2})
    service.serve(prompt, 4, seed=5)
    assert not service.speculating and service.learner.serving_step == released_version


def test_configuration_requires_batch_one():
    with pytest.raises(ValueError, match="batch-one"):
        cold.FullBlockLearner(model(), replace(fit(), batch_size=2))


@pytest.mark.parametrize("published", [False, True])
def test_resume_preserves_master_moments_replay_rng_and_serving_version(published):
    first = cold.ColdStartService(model(), fit())
    first.replay.append(torch.tensor([[2, 3, 5, 7, 4, 6, 8, 4, 5, 8, 2, 3, 9]]))
    first.replay.append(torch.tensor([[9, 3, 2, 8, 4, 5, 7, 4, 5, 2, 9, 3]]))
    for _ in range(2):
        first.learner.step(first.replay.batch(1, 2))
    if published:
        first.learner.publish()
        first.speculating = True
    first.budget.delivered(100.)
    first.tokens, first.requests = 200, 3
    state = first.state_dict()
    second = cold.ColdStartService(model(), fit())
    second.load_state_dict(state)
    assert second.speculating == published
    assert second.learner.serving_step == (2 if published else 0)
    assert second.budget.spent > first.budget.spent
    assert (second.tokens, second.requests) == (200, 3)
    for _ in range(2):
        a, b = first.replay.batch(1, 2), second.replay.batch(1, 2)
        assert all(torch.equal(x, y) for pair_a, pair_b in zip(a, b, strict=True)
                   for x, y in zip(pair_a, pair_b, strict=True))
        assert first.learner.step(a) == second.learner.step(b)
    for name, p in first.learner.master.items():
        assert torch.equal(p, second.learner.master[name])
        a, b = first.learner.optimizer.state[p], second.learner.optimizer.state[second.learner.master[name]]
        assert all(torch.equal(value, b[key]) for key, value in a.items())
    for name, p in first.learner.execution.items():
        assert torch.equal(p, second.learner.execution[name])


def test_resume_rejects_a_different_frozen_base_and_corrupt_weights():
    service = cold.ColdStartService(model(), fit())
    state = service.state_dict()
    different = model()
    with torch.no_grad():
        different.embedding.weight.add_(.01)
    with pytest.raises(ValueError, match="frozen base"):
        cold.ColdStartService(different, fit()).load_state_dict(state)
    next(iter(state["master"].values())).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="finite learning"):
        service.load_state_dict(state)


def test_deterministic_update_restores_serving_execution_even_on_exception():
    previous = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match="interrupted"), cold.update_determinism(not previous):
        assert torch.are_deterministic_algorithms_enabled() != previous
        raise RuntimeError("interrupted")
    assert torch.are_deterministic_algorithms_enabled() == previous


def test_publication_trials_compare_with_ar_after_speculation_is_enabled(monkeypatch):
    prompt = torch.tensor([[2, 3, 5]])
    service = cold.ColdStartService(model(), fit(), gate_prompts=[prompt])
    service.learner.steps, service.learner.serving_step, service.speculating = 2, 1, True

    def deliver(prompt, tokens, *, speculative, seed):
        tps = (40. if service.learner.serving_step == 2 else 30.) if speculative else 60.
        return Generation(tokens=[4] * 8, seconds=8 / tps)

    monkeypatch.setattr(service, "_generate", deliver)
    result = service._probe()
    assert result["ratio"] > 1.1 and result["candidate_over_ar"] < 1.
    assert result["fallback"] and not result["passed"]
