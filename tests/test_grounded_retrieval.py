"""Grounded retrieval/search budget, isolation and failure behavior contracts."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from inference_scaling.arllm.types import SequenceSample
from inference_scaling.retrieval.grounded import (
    AnchoredController, AnchoredEvidence, GroundedConfig, VerifiedLedger,
    grounded_condition, parse_evidence, parse_verdict, verified_search, verdict_reward,
)
from inference_scaling.retrieval.stepwise import StepConfig, StepState
from inference_scaling.retrieval.corpus import Document, Retriever
from inference_scaling.shared.model.output import ThinkingFormat
from test_stepwise_injection import BASE, BODY, Engine as TextEngine


def evidence_output(**changes):
    return json.dumps({"use": True, "same_scope": True, "covers_gap": True,
        "keep_prefix": False, "index": 0, "quote": BODY, **changes})


def test_evidence_requires_scope_decision_and_exact_case_sensitive_quote():
    sources = [{"text": BODY}]
    assert parse_evidence(evidence_output(), sources)[0]["use"]
    for changes in ({"same_scope": False}, {"covers_gap": False}, {"quote": BODY.upper()},
                    {"index": True}, {"keep_prefix": "false"}, {"quote": "irrelevant words that are not in source"}):
        assert parse_evidence(evidence_output(**changes), sources)[0] is None


def test_verdict_positive_coverage_requires_literal_source_support():
    base = {"coverage": 2, "unsupported": False, "scope_error": False,
            "contradiction": False, "quote": "twenty seven seconds"}
    assert parse_verdict(json.dumps(base), [BODY])[0] == base
    for change in ({"quote": "thirty seven seconds"}, {"coverage": True}, {"unsupported": 0}):
        assert parse_verdict(json.dumps({**base, **change}), [BODY])[0] is None
    assert verdict_reward(base, 0) > verdict_reward({**base, "scope_error": True}, 1000)
    assert verdict_reward(None, 100) < 0


def test_question_retrieval_survives_wrong_draft_product_and_includes_solution_tail():
    raw = TextEngine()
    docs = [Document("right", "Mercury retry failure", "Mercury retry failure. "*30+BODY),
            Document("wrong", "Venus file menu", "Venus file menu "*30)]
    ev = AnchoredEvidence(raw, Retriever(docs, kind="bm25"), (120,)*2048)
    pool = ev.candidates("Mercury retry failure", "Venus file menu")
    assert "right" in pool["parent_ids"]
    assert any(BODY in s["text"] for s in pool["sources"])
    assert len(pool["sources"]) <= 8


def test_unsafe_confirmed_prefix_is_removed_only_after_valid_evidence():
    cfg = replace(StepConfig(), work_tokens=8192, context_tokens=16384, evidence_tokens=400)
    for keep in (False, True):
        raw = TextEngine(auxiliary=[evidence_output(keep_prefix=keep)])
        ev = AnchoredEvidence(raw, Retriever([Document("d", "Mercury", BODY)], kind="bm25"), (120,)*2048, cfg)
        ctrl = AnchoredController(raw, ev, ThinkingFormat((126,), (94,)),
            lambda system, user, codec=raw: codec.encode(system+user), cfg=cfg, grounded_cfg=replace(GroundedConfig(), extract_tokens=512))
        state = StepState(BASE, 1, "Mercury timeout?", reason=list(raw.encode("Venus is the product.")))
        packet = ctrl.extract(state)
        assert packet
        assert bool(state.reason) == keep
        assert state.work == state.auxiliary == len(evidence_output(keep_prefix=keep))
        real, noise = ev.variant(packet, "relevant"), ev.variant(packet, "unrelated")
        assert len(real) == len(noise) and real != noise


class VerifiedEngine:
    model_id = "test"
    tokenizer = SimpleNamespace(eos_token_id=3)

    def __init__(self, mode="short"):
        self.mode = mode

    @staticmethod
    def decode(ids, skip_special_tokens=False):
        if ids == (91,):
            return '{"coverage":2,"unsupported":false,"scope_error":false,"contradiction":false,"quote":"twenty seven seconds"}'
        return "twenty seven seconds" if ids else ""

    def sample_batch(self, jobs):
        out = []
        for job in jobs:
            if job.request_id.startswith("leaf_answer:"):
                ids, reason = ((21, 3), "eos") if self.mode == "short" else ((21,)*job.max_new_tokens, "length")
            elif job.request_id.startswith("leaf_verify:"):
                ids, reason = ((91,), "eos") if self.mode == "short" else ((91,)*job.max_new_tokens, "length")
            elif self.mode == "eos":
                ids, reason = (3,), "eos"
            else:
                ids, reason = ((8, 4), "stop") if self.mode == "short" else ((8,)*job.max_new_tokens, "length")
            out.append(SequenceSample(job.prefix, ids, (-1.,)*len(ids), job.sampling.policy_id,
                self.model_id, job.request_id, finish_reason=reason,
                token_statistics=(12.,)*len(ids) if job.statistic else None))
        return out


def book(mode="short", cap=8192):
    raw = VerifiedEngine(mode)
    ctrl = SimpleNamespace(auxiliary_prompt=lambda system, user: (90,))
    return VerifiedLedger(raw, (1,), (4,), cap, 7, 16384, controller=ctrl,
        question="What timeout?", evidence_ids=(20,), cfg=replace(GroundedConfig(), context_tokens=16384))


def test_complete_pool_budget_includes_answers_and_verification():
    ledger = book("full", cap=8192-576)
    selected, audit = verified_search(ledger, replace(GroundedConfig(), context_tokens=16384), True)
    # Identical mock suffixes share one deterministic cached verification.
    assert audit["steps"][0]["reserved_work"] == 7488
    assert ledger.work <= audit["steps"][0]["reserved_work"]
    assert ledger.work <= 8192-576
    assert selected in ledger.answers
    assert sum(len(c["token_ids"]) for c in ledger.calls) == ledger.work


def test_cannot_launch_an_unfunded_verified_pool():
    ledger = book(cap=7487)
    with pytest.raises(RuntimeError, match="initial complete grounded"):
        verified_search(ledger, GroundedConfig(), True)
    assert ledger.calls == []


def test_verified_reward_cache_is_epoch_local_and_selected_answer_is_reused():
    ledger = book()
    sample, = ledger.sample_batch([ledger.request(10, 0)])
    first = ledger.reward(ledger.root, [sample.token_ids])
    charged = ledger.work
    assert ledger.reward(ledger.root, [sample.token_ids]) == first and ledger.work == charged
    assert ledger.answers[sample.token_ids]["answer_ids"] == [21, 3]
    other = book()
    with pytest.raises(RuntimeError, match="uncached"):
        other.reward(other.root, [sample.token_ids])
    with pytest.raises(RuntimeError, match="root differs"):
        ledger.reward((2,), [sample.token_ids])


def test_empty_or_invalid_verification_is_preserved_with_pessimistic_reward():
    ledger = book("full")
    sample, = ledger.sample_batch([ledger.request(1, 0)])
    assert ledger.reward(ledger.root, [sample.token_ids])[0] < 0
    assert ledger.verdicts[0]["error"] == "invalid_json"


def test_verified_condition_does_not_generate_a_new_unverified_answer():
    ledger = book()
    raw = ledger.backend
    ctrl = SimpleNamespace(backend=raw, fmt=ThinkingFormat((4,), (5,)), eos=3,
        auxiliary_prompt=lambda system, user: (90,), prefix=lambda state: tuple(state.base)+tuple(state.reason),
        evidence=None, answer=lambda state: pytest.fail("selected answer must be reused"))
    common = StepState((1,), 7, "Timeout?")
    state, audit = grounded_condition(ctrl, common, None, "relevant", True, replace(GroundedConfig(), context_tokens=16384))
    assert state.answer == [21, 3] and state.finish == "eos"
    assert audit["answer_prepaid_in_work"]
    assert state.work == sum(len(c["token_ids"]) for c in state.calls)
    assert common.calls == [] and common.work == 0


def test_source_title_cannot_substitute_for_missing_body_evidence():
    ledger = book()
    ledger.evidence_bodies = ["The unrelated gardener watered the roses."]
    sample, = ledger.sample_batch([ledger.request(10, 0)])
    assert ledger.reward(ledger.root, [sample.token_ids])[0] < 0
    assert ledger.verdicts[0]["error"] == "unsubstantiated_positive_score"


def test_verified_search_keeps_thinking_eos_without_answer_or_verifier():
    ledger = book("eos")
    selected, audit = verified_search(ledger, GroundedConfig(), True)
    assert selected == (3,) and ledger.work == 3
    assert ledger.answers[selected]["answer_ids"] == []
    assert not ledger.verdicts
    assert all(c["purpose"] == "search" for c in ledger.calls)
    assert audit["stop"] == "terminal"
