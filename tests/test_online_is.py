from dataclasses import replace

import pytest

from inference_scaling.retrieval.cot_guidance import GuidanceConfig, search
from inference_scaling.retrieval.online_is import OnlineISLedger, TuningPolicy, run_online_is
from retrieval_fixtures import Native, Facts, request, AnswerNative


def make(stop=9, eos=False):
    backend, facts = Native(stop, eos), Facts()
    cfg = replace(GuidanceConfig(), thinking_tokens=12, search_tokens=1024,
                  block_tokens=4, context_tokens=256, answer_tokens=16)
    ledger = OnlineISLedger(backend, backend, (1, 2), (901,), 17, cfg, question='Q', facts=facts,
                            policy=TuningPolicy(positions=(3, 6), query_tail=4, packet_tokens=8, cooldown=1))
    return backend, facts, ledger


def test_rejects_unfunded_pool_and_changed_root_before_native_generation():
    backend, _, ledger = make()
    ledger.cap = 4
    with pytest.raises(RuntimeError, match='cap'):
        ledger.sample_batch([request(ledger, count=3), request(ledger, count=3, name='second')])
    with pytest.raises(RuntimeError, match='root'):
        ledger.sample_batch([replace(request(ledger, count=3), prefix=(2,))])
    assert not backend.requests


def test_injection_is_inside_candidate_and_not_counted_as_model_output():
    b, facts, ledger = make()
    sample, = ledger.sample_batch([request(ledger)])
    assert b.requests[0].prefix == ledger.root
    assert [len(r.prefix) for r in b.requests] == [2, 2+3+8, 2+6+16]
    assert len(sample.token_ids) == 10 and all(t >= 100 for t in sample.token_ids)
    assert ledger.work == 10 and len(ledger.events) == len(facts.queries) == 2
    assert [r.uniform_offset for r in b.requests] == [0, 3, 6]
    assert len(ledger.statistics(sample.token_ids)['sc']) == 9  # close token excluded as before
    assert ledger.carried == {'sc': [], 'cn': [], 'lp': []}


def test_request_partition_and_retained_prefix_replay_are_identical():
    _, _, whole = make()
    one, = whole.sample_batch([request(whole)])
    _, facts, split = make()
    head, = split.sample_batch([request(split, count=4, name='head')])
    tail, = split.sample_batch([request(split, head.token_ids, 8, name='tail', offset=4)])
    assert head.token_ids+tail.token_ids == one.token_ids
    assert split.statistics(one.token_ids) == whole.statistics(one.token_ids)
    physical, events = split.materialize(one.token_ids)
    assert (physical, events) == whole.materialize(one.token_ids)
    assert len(facts.queries) == 2  # repeated prefix must not reretrieve or alter evidence


def test_sibling_candidate_cannot_inherit_other_candidates_evidence():
    _, _, ledger = make()
    a, b = ledger.sample_batch([request(ledger, seed=1, name='a'), request(ledger, seed=2, name='b')])
    assert a.token_ids[:3] != b.token_ids[:3]
    _, ea = ledger.materialize(a.token_ids)
    _, eb = ledger.materialize(b.token_ids)
    assert set(ea).isdisjoint(eb)
    for tokens, events in [(a.token_ids, ea), (b.token_ids, eb)]:
        for eid in events:
            event = ledger.events[eid]
            assert tuple(event['branch_ids']) == tokens[:event['position']]


def test_frozen_is_starts_with_empty_state_and_can_select_first_generated_block():
    b, _, ledger = make()
    candidates, audit = search(ledger)
    assert audit['steps'][0]['generated_length_before'] == 0
    assert ledger.root == (1, 2) and not ledger.carried['sc']
    first = audit['steps'][0]
    assert len(first['candidates']) == 3
    assert all(len(c['token_ids']) == 4 for c in first['candidates'])
    assert len({tuple(c['token_ids']) for c in first['candidates']}) > 1
    assert all(r.prefix == ledger.root for r in b.requests[:3])
    assert audit['config']['rollout_count'] == 2
    assert audit['config']['reward_temperature'] == .25
    assert len(candidates[0]) <= 12 and candidates[0][-1] == 901


@pytest.mark.parametrize('eos', [False, True])
def test_actual_termination_before_trigger_is_never_suppressed(eos):
    _, facts, ledger = make(stop=1, eos=eos)
    sample, = ledger.sample_batch([request(ledger)])
    assert len(sample.token_ids) == 2 and sample.token_ids[-1] == (900 if eos else 901)
    assert not facts.queries and not ledger.events and ledger.work == 2


def test_no_insertion_after_close_at_trigger_position():
    _, facts, ledger = make(stop=2)
    sample, = ledger.sample_batch([request(ledger)])
    assert sample.token_ids[-1] == 901 and len(sample.token_ids) == 3
    physical, events = ledger.materialize(sample.token_ids)
    assert physical == ledger.root+sample.token_ids and not events and not facts.queries


def test_readout_does_not_trigger_unused_end_of_budget_insertion():
    _, facts, ledger = make()
    prefix = (101, 102, 103)
    physical, events = ledger.materialize(prefix, pending=False)
    assert physical == ledger.root+prefix and not events and not facts.queries


def test_entrypoint_runs_is_at_zero_and_reads_answer_with_selected_evidence():
    _, facts, template = make()
    backend = AnswerNative()
    result = run_online_is(backend, backend, template.root, template.close, 17, template.cfg,
        question='Q', facts=facts, policy=template.policy)
    assert result['presearch_work'] == 0 and result['root_ids'] == list(template.root)
    assert result['audit']['steps'][0]['generated_length_before'] == 0
    assert result['selected_event_ids']
    assert backend.requests[-1].prefix == tuple(result['selected_physical_ids'])
    assert len(result['selected_physical_ids']) == len(template.root)+len(result['candidates'][0])+16
    assert result['total_generation_tokens'] == sum(len(c['token_ids']) for c in result['native_calls'])+2
    assert result['answer']['answer_ids'] == [70, 900]
