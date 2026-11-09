"""Controller contract tests; engine doubles are not model-quality evidence."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

from inference_scaling.arllm.types import SequenceSample
from inference_scaling.retrieval.stepwise import (
    StepConfig, StepController, StepState, WindowEvidence, first_step_tokens,
    parse_plan, parse_support, step_char_boundaries,
)
from inference_scaling.retrieval.corpus import Document, Retriever
from inference_scaling.shared.model.output import ThinkingFormat


class Engine:
    tokenizer = SimpleNamespace(eos_token_id=3, convert_tokens_to_ids=lambda token: 4)
    encode = staticmethod(lambda text, **kwargs: tuple(map(ord, text)))

    def __init__(self, main=(), auxiliary=()):
        self.main = list(main)
        self.auxiliary = list(auxiliary)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(t) for t in ids if not skip_special_tokens or t not in {3, 4})

    def sample_batch(self, requests):
        out = []
        for request in requests:
            queue = self.auxiliary if request.sampling.top_k == 1 else self.main
            value = queue.pop(0)
            ids = tuple(map(ord, value))
            if len(ids) > request.max_new_tokens:
                queue.insert(0, value[request.max_new_tokens:])
                ids = ids[:request.max_new_tokens]
            out.append(SequenceSample(request.prefix, ids, (-3.,)*len(ids), request.sampling.policy_id,
                "double", request.request_id, finish_reason="eos" if 3 in ids else "stop" if 126 in ids else "length",
                token_statistics=(20.,)*len(ids) if request.statistic else None))
        return out


BODY = "Mercury uses a timeout of twenty seven seconds for retries."
PACKET = {"ids": tuple(map(ord, BODY)), "bodies": [{"text": BODY, "start": 0, "stop": len(BODY)}]}


class Evidence:
    fetch = staticmethod(lambda *args: PACKET)
    variant = staticmethod(lambda packet, content: packet["ids"] if content == "relevant" else (ord("z"),)*len(packet["ids"]))


def controller(engine, cfg=StepConfig()):
    return StepController(engine, Evidence(), ThinkingFormat((126,), (94,)),
                          lambda system, user: engine.encode(system+user), cfg=cfg)


BASE = (ord("S"), 4, ord("Q"), 4, 94)


class StepwiseTests(unittest.TestCase):
    def test_boundaries_ignore_versions_lists_and_code(self):
        text = "1. Version 1.2, e.g. v2.3, is selected. Next"
        self.assertEqual(step_char_boundaries(text), [text.index("selected.")+len("selected.")])
        self.assertEqual(step_char_boundaries("```\nfoo.\n\nbar!\n```"), [])
        self.assertEqual(step_char_boundaries("Use `x. y!` safely"), [])
        self.assertEqual(step_char_boundaries("A line\ncontinues"), [])
        self.assertTrue(step_char_boundaries("A complete paragraph\n\nNext"))

    def test_boundary_preserves_original_ids_and_rejects_crossing_token(self):
        engine = Engine()
        ids = engine.encode("First fact. Next")
        self.assertEqual(first_step_tokens(engine, ids), len("First fact."))
        fused = SimpleNamespace(decode=lambda ids, **kw: {(): "", (1,): "First fact. Next"}[tuple(ids)])
        self.assertIsNone(first_step_tokens(fused, (1,)))
        self.assertIsNone(first_step_tokens(engine, engine.encode("Still unfinished")))

    def test_plan_schema_strict_boolean_and_no_markdown_retry(self):
        for text in ('{"need":"false"}', '```json\n{"need":false}\n```', '{"need":true}', 'null'):
            self.assertIsNone(parse_plan(text)[0])
        self.assertEqual(parse_plan('{"need":false}'), ({"need": False}, None))
        self.assertTrue(parse_plan('{"need":true,"fact":"timeout","query":"Mercury timeout"}')[0]["need"])

    def test_support_requires_verbatim_exposed_body(self):
        valid = {"use": True, "index": 0, "quote": "uses a timeout of twenty seven seconds"}
        self.assertTrue(parse_support(json.dumps(valid), PACKET)[0]["use"])
        for update in ({"index": True}, {"quote": "uses a timeout of forty seconds"}, {"use": "true"}, {"quote": "Mercury"}):
            self.assertIsNone(parse_support(json.dumps({**valid, **update}), PACKET)[0])

    def test_paid_preview_auxiliary_and_exact_shared_checkpoint(self):
        text = "Confirmed context is available. A missing Mercury timeout must be checked.~"
        plan = '{"need":true,"fact":"Mercury timeout","query":"Mercury retry timeout"}'
        support = '{"use":true,"index":0,"quote":"uses a timeout of twenty seven seconds"}'
        engine = Engine([text, "~", "OK\x03", "~", "\x03"], [plan, support])
        ctrl = controller(engine, replace(StepConfig(), minimum_tokens=8, chunk_tokens=1024))
        common, packet = ctrl.stem("Mercury timeout?", BASE, 1, "flare_gap")
        self.assertIsNotNone(packet)
        self.assertEqual(engine.decode(common.reason), "Confirmed context is available.")
        self.assertEqual(common.work, len(text)+len(plan)+len(support))
        self.assertEqual(common.auxiliary, len(plan)+len(support))
        self.assertEqual(common.discarded, len(text)-len(common.reason))
        real = ctrl.branch(common, packet, "relevant", "flare_gap_rebuild")
        noise = ctrl.branch(common, packet, "unrelated", "flare_gap_rebuild")
        self.assertEqual(ctrl.result(real)["answer_text"], "OK")
        self.assertTrue(ctrl.result(noise)["empty_answer"])
        self.assertTrue(ctrl.result(noise)["immediate_thinking_end"])
        self.assertEqual(len(real.payload), len(noise.payload))
        self.assertEqual(real.reason[:len(common.reason)], common.reason)
        self.assertEqual(real.work, common.work+1)
        self.assertEqual(len(common.calls), 3)
        self.assertEqual(len(real.calls), 2)
        self.assertEqual(real.calls[0]["prefix_ids"][-len(common.reason):], common.reason)

    def test_invalid_or_unsupported_check_keeps_draft_and_charges_cost(self):
        text = "Confirmed context is available. A missing Mercury timeout must be checked.~"
        for aux in (["broken"], ['{"need":true,"fact":"timeout","query":"Mercury timeout"}', '{"use":false}']):
            engine = Engine([text, "OK\x03"], aux)
            ctrl = controller(engine, replace(StepConfig(), minimum_tokens=8, chunk_tokens=1024, planner_checks=1))
            state, packet = ctrl.stem("Mercury?", BASE, 1, "flare_gap")
            self.assertIsNone(packet)
            self.assertEqual(state.discarded, 0)
            self.assertEqual(engine.decode(state.reason), text[:-1])
            self.assertEqual(state.work, len(text)+sum(map(len, aux)))

    def test_eos_is_never_reopened_and_empty_answer_is_preserved(self):
        engine = Engine(["Unfinished\x03"])
        ctrl = controller(engine)
        state, packet = ctrl.stem("Q", BASE, 1, "closed")
        row = ctrl.result(state)
        self.assertTrue(row["eos_inside_thinking"])
        self.assertTrue(row["empty_answer"])
        self.assertFalse(row["forced_thinking_end"])
        self.assertEqual(state.work, len("Unfinished\x03"))
        self.assertEqual(len(state.calls), 1)

    def test_short_remaining_budget_skips_aux_and_enforces_cap(self):
        engine = Engine(["First step is finished. Second step finished. More text without ending", "OK\x03"])
        cfg = replace(StepConfig(), work_tokens=64, minimum_tokens=8, chunk_tokens=64)
        ctrl = controller(engine, cfg)
        # The main stream reaches the cap; remove its unused double-only tail.
        engine.main = [engine.main[0][:64], "OK\x03"]
        state, _ = ctrl.stem("Q", BASE, 1, "flare_gap")
        self.assertEqual(state.work, 64)
        self.assertEqual(state.auxiliary, 0)
        self.assertTrue(state.forced)
        self.assertEqual(ctrl.result(state)["answer_text"], "OK")

    def test_inline_payload_remains_at_original_offset(self):
        ctrl = controller(Engine())
        state = StepState(BASE, 1, "Q", reason=list(map(ord, "beforeAFTER")), payload=(777,),
                          placement="assistant_inline", inline_offset=6)
        self.assertEqual(ctrl.prefix(state), BASE+tuple(map(ord, "before"))+(777,)+tuple(map(ord, "AFTER")))

    def test_flare_placement_pair_shares_exact_checkpoint_and_generation_work(self):
        engine = Engine(["~", "A\x03", "~", "A\x03"])
        ctrl = controller(engine)
        common = StepState(BASE, 3, "Q", reason=list(map(ord, "Confirmed fact.")), work=15)
        inline = ctrl.branch(common, PACKET, "relevant", "flare_gap")
        rebuild = ctrl.branch(common, PACKET, "relevant", "flare_gap_rebuild")
        self.assertEqual(inline.placement, "assistant_inline")
        self.assertEqual(rebuild.placement, "user_reference_rebuild_at_step_boundary")
        self.assertEqual(inline.payload, rebuild.payload)
        self.assertEqual(inline.reason, rebuild.reason)
        self.assertEqual(inline.work, rebuild.work)
        self.assertEqual(inline.calls[0]["seed"], rebuild.calls[0]["seed"])
        self.assertNotEqual(inline.calls[0]["prefix_ids"], rebuild.calls[0]["prefix_ids"])

    def test_window_packet_noise_preserves_labels_and_length(self):
        engine = Engine()
        index = Retriever([Document("m", "Mercury", BODY*30)], kind="bm25")
        evidence = WindowEvidence(engine, index, (ord("z"),)*2000, replace(StepConfig(), evidence_tokens=256))
        packet = evidence.fetch("Mercury timeout")
        real = evidence.variant(packet, "relevant"); noise = evidence.variant(packet, "unrelated")
        self.assertLessEqual(len(real), 256)
        self.assertEqual(len(real), len(noise))
        positions = {i for body in packet["bodies"] for i in range(body["start"], body["stop"])}
        self.assertTrue(positions)
        self.assertTrue(all(a == b for i, (a, b) in enumerate(zip(real, noise)) if i not in positions))


if __name__ == "__main__": unittest.main()
