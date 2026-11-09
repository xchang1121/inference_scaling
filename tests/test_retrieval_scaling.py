"""Budget and context contracts for the retrieval/search combination."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.types import SequenceSample
from inference_scaling.retrieval.scaled import (
    ScalingConfig, SearchLedger, pool_upper_bound, search_suffix,
)


class Engine:
    tokenizer = SimpleNamespace(eos_token_id=3)
    model_id = "test"

    def __init__(self, mode="full"):
        self.mode = mode

    def sample_batch(self, jobs):
        result = []
        for job in jobs:
            if self.mode == "short" and len(job.prefix) > 1:
                ids, finish = (8, 4), "stop"
            elif self.mode == "eos":
                ids, finish = (3,), "eos"
            else:
                ids, finish = (8,)*job.max_new_tokens, "length"
            result.append(SequenceSample(job.prefix, ids, (-1.,)*len(ids),
                job.sampling.policy_id, "test", job.request_id, finish_reason=finish,
                token_statistics=tuple(float(i+1) for i in range(len(ids)))))
        return result


def ledger(cap=8192, mode="full"):
    return SearchLedger(Engine(mode), (1,), (4,), cap, 77, 16384)


def test_reservation_matches_full_nonterminal_trees():
    cfg = ConditionalISConfig(4, 2, 256, 1024, .25, True)
    assert pool_upper_bound(cfg, 0, False) == 7168
    assert pool_upper_bound(cfg, 256, True) == 4352
    book = ledger()
    _, audit = search_suffix(book, "block_met_sc", replace(ScalingConfig(), context_tokens=16384))
    assert book.work == 7168
    assert len(audit["steps"]) == 1
    assert audit["stop"] == "whole_pool_budget"


def test_short_rollouts_can_fund_another_complete_pool():
    book = ledger(mode="short")
    selected, audit = search_suffix(book, "block_met_sc", replace(ScalingConfig(), context_tokens=16384))
    assert len(audit["steps"]) == 2
    assert audit["stop"] == "terminal"
    assert selected[-1] == 4
    assert all(s["actual_work"] <= s["reserved_work"] for s in audit["steps"])


def test_auxiliary_work_reduces_bon_count():
    book = ledger(cap=8000)
    _, audit = search_suffix(book, "sc_bon", replace(ScalingConfig(), context_tokens=16384))
    assert audit["candidate_count"] == 7
    assert book.work == 7*1024


def test_reward_excludes_frozen_prefix_and_terminal_marker():
    book = ledger(mode="short")
    # A block then a continuation; each score is native to its exact context.
    sample, = book.sample_batch([book.request(2, 0)])
    tail, = book.sample_batch([book.request(2, 1, sample.token_ids)])
    assert book.reward((1,), [sample.token_ids+tail.token_ids]) == ((1+2+1)/3,)
    with pytest.raises(RuntimeError, match="root differs"):
        book.reward((2,), [sample.token_ids])
    with pytest.raises(RuntimeError, match="uncached"):
        book.reward((1,), [(9,)])


def test_eos_is_kept_as_a_real_candidate_not_filtered():
    book = ledger(mode="eos")
    selected, audit = search_suffix(book, "sc_bon", replace(ScalingConfig(), context_tokens=16384))
    assert selected == (3,)
    assert audit["scores"] == (0.,)*8
    assert book.work == 8


def test_generation_rejects_unfunded_or_changed_context():
    book = ledger(cap=8)
    with pytest.raises(RuntimeError, match="unfunded"):
        book.sample_batch([book.request(9, 0)])
    with pytest.raises(RuntimeError, match="context changed"):
        book.sample_batch([replace(book.request(1, 0), prefix=(2,))])


def test_fresh_evidence_epoch_cannot_reuse_old_statistics():
    old = ledger()
    sample, = old.sample_batch([old.request(2, 0)])
    new = SearchLedger(old.backend, (2,), (4,), 8192, 77, 16384)
    with pytest.raises(RuntimeError, match="uncached"):
        new.reward((2,), [sample.token_ids])
