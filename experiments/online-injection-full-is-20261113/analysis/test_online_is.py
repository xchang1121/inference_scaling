from dataclasses import replace
from types import SimpleNamespace

import pytest

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, SequenceSample
from inference_scaling.retrieval.cot_guidance import GuidanceConfig, GuidanceLedger, search
from inference_scaling.retrieval.online_is import OnlineISLedger, OnlineRetrievalConfig, run_online_is


class Native:
    model_id = 'test-online'
    tokenizer = SimpleNamespace(eos_token_id=900, convert_tokens_to_ids=lambda s: 902 if s == '<think>' else 901)

    def __init__(self, stop=9, eos=False):
        self.stop, self.eos = stop, eos
        self.requests, self.stats = [], {}

    def encode(self, text, **kwargs):
        return (10,) if text == '\n' else tuple(30+ord(c) % 20 for c in text)

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))

    def sample_batch(self, requests):
        self.requests.extend(requests)
        samples = []
        for r in requests:
            physical, ids, sc = list(r.prefix), [], []
            for i in range(r.max_new_tokens):
                count = sum(100 <= t < 200 for t in physical)
                if count >= self.stop:
                    token = 900 if self.eos else 901
                else:
                    # Changing injected context changes both sampling and reward.
                    token = 100+(r.seed+r.uniform_offset+i+sum(physical)) % 73
                sc.append(1+sum(physical) % 17/10)
                physical.append(token);ids.append(token)
                if token in {900, 901}:
                    break
            sample = SequenceSample(tuple(r.prefix), tuple(ids), (-1.,)*len(ids), r.sampling.policy_id,
                                    self.model_id, r.request_id, 'eos' if ids[-1] == 900 else 'stop' if ids[-1] == 901 else 'length')
            self.stats[r.request_id] = tuple(sc), (3.,)*len(ids)
            samples.append(sample)
        return samples

    def take(self, request, sample):
        return self.stats.pop(request.request_id)


class Facts:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, *, k=1, exclude=frozenset()):
        self.queries.append((query, exclude))
        key = sum(map(ord, query)) % 19
        name = str(key)+'-'+str(len(exclude))
        payload = {'ids': [20, 30+key, 21], 'body_start': 1, 'body_stop': 2,
                   'id': name, 'source': 'test', 'score': 1.}
        return [payload], {'query': query, 'selected_ids': [name], 'seconds': 0.}


def make(mode='relevant', stop=9, eos=False):
    backend, facts = Native(stop, eos), Facts()
    cfg = replace(GuidanceConfig(), thinking_tokens=12, search_tokens=1024,
                  block_tokens=4, context_tokens=256, answer_tokens=16)
    ledger = OnlineISLedger(backend, backend, (1, 2), (901,), 17, cfg, question='Q', facts=facts,
                            noise=(40, 41), mode=mode,
                            retrieval_cfg=OnlineRetrievalConfig((3, 6), 4, 8))
    return backend, facts, ledger


def request(ledger, prefix=(), count=12, seed=7, name='test', offset=0):
    return GenerationRequest(ledger.root+tuple(prefix), count, SamplingConfig(eos_token_id=900), seed, name,
                             uniform_offset=offset)


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
    candidates, audit = search(ledger, 'is_block_sc')
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


def test_disabled_retrieval_is_exactly_the_original_ledger_and_is_algorithm():
    backend, facts, wrapped = make('closed')
    plain_backend = Native()
    plain = GuidanceLedger(plain_backend, plain_backend, wrapped.root, wrapped.close, wrapped.seed, wrapped.cfg)
    actual, actual_audit = search(wrapped, 'is_block_sc')
    expected, expected_audit = search(plain, 'is_block_sc')
    assert actual == expected and actual_audit == expected_audit
    assert wrapped.work == plain.work and not facts.queries and not wrapped.events
    assert [(r.prefix, r.seed, r.uniform_offset, r.max_new_tokens) for r in backend.requests] == [
        (r.prefix, r.seed, r.uniform_offset, r.max_new_tokens) for r in plain_backend.requests]


@pytest.mark.parametrize('eos', [False, True])
def test_actual_termination_before_trigger_is_never_suppressed(eos):
    _, facts, ledger = make(stop=1, eos=eos)
    sample, = ledger.sample_batch([request(ledger)])
    assert len(sample.token_ids) == 2 and sample.token_ids[-1] == (900 if eos else 901)
    assert not facts.queries and not ledger.events and ledger.work == 2


def test_noise_is_same_size_per_insertion_and_contains_no_fact_body():
    _, _, real = make()
    _, _, noise = make('noise')
    prefix = (101, 102, 103)
    _, re = real.materialize(prefix)
    _, ne = noise.materialize(prefix)
    r, n = real.events[re[0]], noise.events[ne[0]]
    assert r['query'] == n['query'] and r['packet'] == n['packet']
    assert len(r['injected_ids']) == len(n['injected_ids']) == 8
    assert n['injected_ids'] == [20, 40, 41, 40, 41, 40, 41, 21]


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


class AnswerNative(Native):
    def sample_batch(self, requests):
        if all(901 in r.prefix for r in requests):
            self.requests.extend(requests)
            return [SequenceSample(tuple(r.prefix), (70, 900), (-1., -1.),
                r.sampling.policy_id, self.model_id, r.request_id, 'eos') for r in requests]
        return super().sample_batch(requests)


def test_entrypoint_runs_is_at_zero_and_reads_answer_with_selected_evidence():
    _, facts, template = make()
    backend = AnswerNative()
    result = run_online_is(backend, backend, template.root, template.close, 17, template.cfg,
        question='Q', facts=facts, retrieval_cfg=template.retrieval_cfg)
    assert result['presearch_work'] == 0 and result['root_ids'] == list(template.root)
    assert result['audit']['steps'][0]['generated_length_before'] == 0
    assert result['selected_event_ids']
    assert backend.requests[-1].prefix == tuple(result['selected_physical_ids'])
    assert len(result['selected_physical_ids']) == len(template.root)+len(result['candidates'][0])+16
    assert result['total_generation_tokens'] == sum(len(c['token_ids']) for c in result['native_calls'])+2
    assert result['answer']['answer_ids'] == [70, 900]


def test_benchmark_entrypoint_has_no_checkpoint_or_controller_generation(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from scripts.benchmark_online_is import run_arm
    _, facts, template = make()
    backend = AnswerNative()
    case = {'id': 'test', 'question': 'Q', 'prompt_ids': list(template.root)}
    result = run_arm(backend, backend, facts, case, 7, 'closed_is', template.cfg,
        template.close, (), template.retrieval_cfg)
    assert result['presearch_work'] == 0 and result['events'] == []
    assert result['calls'][0]['request_id'].startswith('conditional-is:step:0:')
    assert result['calls'][0]['prefix_ids'] == case['prompt_ids']


@pytest.mark.parametrize('arm', ['direct', 'closed_is', 'online_is', 'online_is_noise'])
def test_full_offline_replay_and_evidence_audit(arm, monkeypatch):
    from pathlib import Path
    import json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from benchmark_online_is import run_arm
    from audit_online_is import audit_row
    _, facts, template = make()
    backend = AnswerNative()
    case = {'id': 'test', 'question': 'Q', 'prompt_ids': list(template.root)}
    row = run_arm(backend, backend, facts, case, 7, arm, template.cfg,
        template.close, (40, 41), template.retrieval_cfg)
    row.update(unit_id='test', question='Q', seconds=100.,
        cost={'generated_tokens': row['total_generation_tokens']})
    row = json.loads(json.dumps(row))
    failures = []
    audit_row(row, template.cfg, template.retrieval_cfg, lambda ids, skip=False: backend.decode(ids),
        (40, 41), 900, 901, lambda ok, label: failures.append(label) if not ok else None)
    assert not failures
    if arm.startswith('online'):
        row['native_calls'][0]['uniform_offset'] += 1
        row['selected_physical_ids'][0] += 1
        row['events'][0]['query'] = 'unrelated branch'
        row['audit']['steps'][0]['selected_index'] = 999
        audit_row(row, template.cfg, template.retrieval_cfg, lambda ids, skip=False: backend.decode(ids),
            (40, 41), 900, 901, lambda ok, label: failures.append(label) if not ok else None)
        assert any('native_rng_offsets' in x for x in failures)
        assert any('selected_physical_context' in x for x in failures)
        assert any('branch_query' in x for x in failures)
        assert any('unchanged_IS_full_replay' in x for x in failures)
