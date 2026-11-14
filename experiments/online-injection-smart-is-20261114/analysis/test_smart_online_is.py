from dataclasses import replace
import copy
import json
from pathlib import Path
import sys

import pytest

from test_online_is import Native, Facts, request
from inference_scaling.retrieval.cot_guidance import GuidanceConfig, search
from inference_scaling.retrieval.smart_online_is import (
    SmartOnlineLedger, boundary, clean_question, overlaps, retrieve_anchored,
)


def make(variant, stop=600, eos=False):
    b, f = Native(stop=stop, eos=eos), Facts()
    cfg = replace(GuidanceConfig(), thinking_tokens=640, answer_tokens=1024, context_tokens=8192)
    ledger = SmartOnlineLedger(b, b, (1, 2), (901,), 17, cfg, question='Original issue', facts=f,
                               variant=variant, mode='relevant')
    return b, f, ledger


@pytest.mark.parametrize('variant', ['resume', 'anchored', 'boundary', 'adaptive', 'bridge', 'early'])
def test_each_variant_preserves_tokens_branch_context_and_native_eos(variant):
    b, _, ledger = make(variant)
    sample, = ledger.sample_batch([request(ledger, count=640)])
    assert len(sample.token_ids) == 601 and sample.token_ids[-1] == 901
    assert ledger.work == 601 and len(ledger.statistics(sample.token_ids)['sc']) == 600
    physical, events = ledger.materialize(sample.token_ids, pending=False)
    assert len(physical) == 2+601+sum(len(ledger.events[i]['injected_ids']) for i in events)
    assert len(events) <= 2
    assert b.requests[0].prefix == (1, 2)
    for call in ledger.native_calls:
        expected, ids = ledger.materialize(call['logical_prefix_ids'][2:])
        assert tuple(call['prefix_ids']) == expected and call['event_ids'] == ids
    for e in ledger.events:
        assert len(e['packet']['ids']) == 256
        assert e['injected_ids'][256:] == list(ledger.resume_ids)


def test_short_terminal_prefix_never_retrieves_or_retries():
    _, f, ledger = make('adaptive', stop=40, eos=True)
    sample, = ledger.sample_batch([request(ledger, count=640)])
    assert sample.token_ids[-1] == 900 and ledger.work == 41
    assert not ledger.events and not f.queries


def test_adaptive_uses_only_measured_prefix_probabilities():
    _, _, ledger = make('adaptive')
    # Native supplies logprob -1, so uncertainty gate opens on late fallback.
    sample, = ledger.sample_batch([request(ledger, count=640)])
    _, ids = ledger.materialize(sample.token_ids)
    assert [ledger.events[i]['position'] for i in ids] == [160, 544]
    late = [d for d in ledger.decisions if d['surprisal_last32'] is not None]
    assert all(d['surprisal_last32'] == 1. for d in late)


def test_adaptive_skips_late_evidence_when_prefix_is_confident():
    class LowSurprisalNative(Native):
        def sample_batch(self, requests):
            return [replace(s, token_logprobs=(-.01,)*len(s.token_ids))
                    for s in super().sample_batch(requests)]
    b = LowSurprisalNative(stop=600)
    cfg = replace(GuidanceConfig(), thinking_tokens=640, answer_tokens=1024, context_tokens=8192)
    ledger = SmartOnlineLedger(b, b, (1, 2), (901,), 17, cfg, question='Q', facts=Facts(),
                               variant='adaptive', mode='relevant')
    sample, = ledger.sample_batch([request(ledger, count=640)])
    _, ids = ledger.materialize(sample.token_ids)
    assert [ledger.events[i]['position'] for i in ids] == [160]
    assert any(d['phase'] == 'late' and d['surprisal_last32'] == .01 and not d['eligible']
               for d in ledger.decisions)


def test_boundary_policy_uses_observed_boundaries_before_deadlines():
    class BoundaryNative(Native):
        def decode(self, ids, **kwargs):
            return super().decode(ids, **kwargs)+('\n' if len(ids) in {96, 256} else '')
    b = BoundaryNative(stop=600)
    cfg = replace(GuidanceConfig(), thinking_tokens=640, answer_tokens=1024, context_tokens=8192)
    ledger = SmartOnlineLedger(b, b, (1, 2), (901,), 17, cfg, question='Q', facts=Facts(),
                               variant='boundary', mode='relevant')
    sample, = ledger.sample_batch([request(ledger, count=640)])
    _, ids = ledger.materialize(sample.token_ids)
    assert [ledger.events[i]['position'] for i in ids] == [96, 320]


def test_sibling_isolation_and_final_replay_do_not_retrieve_again():
    _, facts, ledger = make('anchored')
    samples = ledger.sample_batch([request(ledger, count=640, seed=1, name='a'),
                                   request(ledger, count=640, seed=2, name='b')])
    count = len(facts.queries)
    paths = [ledger.materialize(s.token_ids)[1] for s in samples]
    assert set(paths[0]).isdisjoint(paths[1])
    assert len(facts.queries) == count


def test_original_is_configuration_and_first_block_stay_unchanged():
    _, _, ledger = make('resume', stop=140)
    _, audit = search(ledger, 'is_block_sc')
    assert audit['steps'][0]['generated_length_before'] == 0
    assert audit['config']['candidate_count'] == 3
    assert audit['config']['rollout_count'] == 2
    assert audit['config']['block_size'] == 512
    assert audit['config']['reward_temperature'] == .25


def test_question_anchor_boundary_and_overlap_rules():
    assert clean_question('problem\n\nIn the final answer, please answer') == 'problem'
    assert boundary('Complete sentence.\n') and boundary('Complete step\n')
    assert not boundary('unfinished wor')
    assert overlaps('x@0-192', 'x@64-256')
    assert not overlaps('x@0-192', 'x@96-288')
    assert not overlaps('x@0-192', 'y@0-192')


def test_rank_fusion_prioritizes_question_without_dropping_branch_query():
    f = Facts()
    payload, log = retrieve_anchored(f, 'Original issue', 'a speculative different topic', set())
    assert len(log['searches']) == 2
    assert log['queries'][0] == 'Original issue'
    assert 'speculative' in log['queries'][1]
    assert payload['id'] == log['rank_lists'][0][0]


class ReadableNative(Native):
    def encode(self, text, **kwargs):
        return (10,) if text == '\n' else tuple(3000+ord(c) for c in text)

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(t-3000) if t >= 3000 else '\n' if t == 10 else
                       '' if t in {900, 901} and skip_special_tokens else 'x ' for t in ids)


def audited_row(variant):
    from inference_scaling.retrieval.smart_online_is import run_smart_is
    b, f = ReadableNative(stop=180, eos=True), Facts()
    cfg = replace(GuidanceConfig(), thinking_tokens=640, answer_tokens=1024, context_tokens=8192)
    result = run_smart_is(b, b, (1, 2), (901,), 17, cfg, question='Q', facts=f, variant=variant)
    row = {'unit_id': 'synthetic', 'arm': variant, 'question': 'Q', 'question_id': 'synthetic',
           'seed': 1, 'base_ids': [1, 2], 'static_packets': [], 'static_injected_ids': [],
           'seconds': 100., 'cost': {'generated_tokens': result['total_generation_tokens']}, **result}
    return b, cfg, row


@pytest.mark.parametrize('variant', ['resume', 'anchored', 'boundary', 'adaptive', 'bridge', 'early'])
def test_policy_audit_replays_physical_calls_and_detects_tampered_branch(variant):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_smart_online_is import audit_smart_row
    from inference_scaling.retrieval.online_is import OnlineRetrievalConfig
    b, cfg, row = audited_row(variant)
    # ReplayLedger derives search seeds from the row question/seed; use exactly
    # the same derivation for the fixture rather than bypassing its RNG audit.
    from inference_scaling.shared.rng import SeedStream
    from inference_scaling.retrieval.smart_online_is import run_smart_is
    b = ReadableNative(stop=180, eos=True)
    result = run_smart_is(b, b, (1, 2), (901,), SeedStream(1).derive('question', 'synthetic'), cfg,
                          question='Q', facts=Facts(), variant=variant)
    row.update(result); row['cost']['generated_tokens'] = result['total_generation_tokens']
    row = json.loads(json.dumps(row))
    def audit(target):
        failures = []
        audit_smart_row(target, cfg, OnlineRetrievalConfig(),
                        lambda ids, skip=False: b.decode(ids, skip_special_tokens=skip), (), 900, 901,
                        lambda ok, label: failures.append(label) if not ok else None)
        return failures
    assert not audit(row)
    broken = copy.deepcopy(row)
    broken['events'][0]['query'] += ' leaked answer'
    assert any('event_query' in f for f in audit(broken))
    broken = copy.deepcopy(row)
    broken['decisions'][0]['eligible'] = not broken['decisions'][0]['eligible']
    assert any('trigger_policy' in f for f in audit(broken))


def test_metric_pairing_keeps_empty_runs_and_resamples_questions():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from summarize_smart_online_is import ARMS, aggregate
    records = []
    for qi, q in enumerate(('q1', 'q2')):
        for seed in (1, 2):
            for ai, arm in enumerate(ARMS):
                item = {'question_id': q, 'seed': seed, 'arm': arm, 'f1': qi*10+ai,
                        'empty': qi == 0 and arm == 'static_facts', 'truncated': False,
                        'thinking_eos': False, 'forced_thinking_end': False,
                        'termination_within2_after_injection': False}
                for key in ('generated_tokens','search_tokens','answer_tokens','thought_tokens','seconds','retrieval_seconds',
                            'selected_injected_tokens','retrieval_searches','native_prefill_tokens','native_calls','selected_injections'):
                    item[key] = 10+ai
                records.append(item)
    result = aggregate(records)
    assert result['measurements'] == 32 and result['questions'] == 2
    assert result['arms'][0]['empty'] == 2 and result['arms'][0]['f1'] == 5.
    pair = result['paired_f1'][0]
    assert pair['n_questions'] == 2 and pair['difference'] == 1. and pair['bootstrap_95'] == [1., 1.]
    with pytest.raises(AssertionError):
        aggregate(records+[records[0]])


@pytest.mark.parametrize('arm', ['static_facts', 'online_is', 'resume', 'anchored', 'boundary', 'adaptive', 'bridge', 'early'])
def test_benchmark_entry_accepts_the_complete_native_runner_signature(arm):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from benchmark_smart_online_is import run_arm
    from inference_scaling.retrieval.online_is import OnlineRetrievalConfig
    b = ReadableNative(stop=180, eos=True)
    cfg = replace(GuidanceConfig(), thinking_tokens=640, answer_tokens=1024, context_tokens=8192)
    # Static formatting requires two message-end IDs in the prompt.
    case = {'id':'synthetic', 'question':'Q', 'prompt_ids':[1,900,2,900,3]}
    result = run_arm(b, b, Facts(), case, 1, arm, cfg, (901,), (40,41), OnlineRetrievalConfig())
    assert result['arm'] == arm and result['answer']['empty_answer']
    assert result['total_generation_tokens'] == result['search_work']
