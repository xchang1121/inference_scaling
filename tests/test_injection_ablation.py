"""Behavioral invariants for token-matched controls and separate answer limits."""
from types import SimpleNamespace

import pytest

from inference_scaling.retrieval import Document, Hit, InjectionConfig, OnlineInjectionController
from inference_scaling.retrieval.ablation import ReferenceVariant, arms, reference_tokens, static_prompt
from inference_scaling.shared.model.output import ThinkingFormat
from test_retrieval_injection import EngineDouble, encode


class Codec:
    encode = staticmethod(encode)


def test_content_and_wrapper_controls_preserve_exact_token_lengths():
    raw = Codec()
    hits = [Hit(Document("d1","A source","Important technical facts. "*40),1),
            Hit(Document("d2","Another source","Other relevant facts. "*40),.5)]
    noise = encode("Unrelated garden and room descriptions. "*50)
    values = {}
    for content,wrapper in (("relevant","boxed"),("unrelated","boxed"),("blank","boxed"),
                            ("relevant","neutral"),("unrelated","neutral")):
        values[content,wrapper] = reference_tokens(ReferenceVariant(raw,hits,noise,content=content,wrapper=wrapper),hits)
    assert len({len(v) for v in values.values()}) == 1
    assert values["relevant","boxed"] != values["unrelated","boxed"]
    for key,tokens in values.items():
        text = "".join(map(chr,tokens))
        assert "Source d1: A source" in text and "Source d2: Another source" in text
        assert ("External reference" in text) == (key[1] == "boxed")
    assert "Important technical facts" in "".join(map(chr,values["relevant","neutral"]))
    assert "Important technical facts" not in "".join(map(chr,values["unrelated","boxed"]))


def test_static_insertion_keeps_evidence_ids_and_message_boundaries():
    base = (10,11,99,12,13,99,14,15)
    result = static_prompt(base,(71,72,73),99,(20,21))
    assert result == (10,11,99,12,13,20,21,71,72,73,99,14,15)
    with pytest.raises(ValueError):
        static_prompt((1,99,2),(7,),99,(20,))


def test_answer_cap_is_independent_of_unused_thinking_budget():
    for budget in (8,32):
        engine = EngineDouble();engine.reason = "A."
        cfg = InjectionConfig(position="none",max_retrievals=0,thinking_tokens=budget,
                              max_new_tokens=budget+64,answer_tokens=2,chunk_tokens=8)
        out = OnlineInjectionController(engine,SimpleNamespace(kind="bm25",fingerprint="fixed"),cfg,
            thinking_format=ThinkingFormat((ord("~"),),(ord("^"),)),prompt=lambda q,e:encode(q+"^")).run("Q",seed=1,request_id="cap")
        assert out["thinking_tokens"] == 2
        assert out["answer_text"] == "DO"
        assert out["answer_generated_tokens"] == 2
        assert out["finish_reason"] == "answer_budget"
        assert not any(e["kind"] == "forced_thinking_end" for e in out["events"])


def test_followup_matrix_has_fixed_answer_and_evidence_limits():
    for budget in (256,1024,4096):
        matrix = arms(budget)
        assert len(matrix) == 14
        for _,_,cfg in matrix.values():
            assert cfg.thinking_tokens == budget
            assert cfg.answer_tokens == 512
            assert cfg.max_new_tokens == budget+512
            assert cfg.evidence_tokens == cfg.total_evidence_tokens == 1024
            assert cfg.query == "question" and cfg.self_certainty
    with pytest.raises(ValueError):
        InjectionConfig(answer_tokens=0)
