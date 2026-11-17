from dataclasses import replace

import pytest

from retrieval_fixtures import Native, Facts, request
from inference_scaling.retrieval.cot_guidance import GuidanceConfig, search
from inference_scaling.retrieval.online_is import TuningPolicy, OnlineISLedger


def make(policy, backend=None):
    backend = backend or Native(stop=600)
    cfg = replace(GuidanceConfig(), thinking_tokens=640, answer_tokens=1024, context_tokens=8192)
    ledger = OnlineISLedger(backend, backend, (1,2), (901,), 17, cfg,
                            question='Original issue', facts=Facts(), policy=policy)
    return backend, ledger


@pytest.mark.parametrize(('policy', 'expected'), [
    (TuningPolicy(), '82fc589ff0ca3c1ca27865b28ce90098ff850620699841289310617c96195384'),
    (TuningPolicy(query='anchored', anchor_context='question_tail', overlap=.5, continuation='resume'),
     '1784a3ebb648c60d686993efd5020e5d503e6035103794c45169d7504af37c65'),
    (TuningPolicy(trigger='never'), 'd84ef419aa37ef1890a0969a52219561b924c01038cb10ed1ae39f98dee50e30'),
    (TuningPolicy(trigger='surprisal', threshold=.5),
     'e48f9c6bb82a35cd5d122f3615635718e04c97bbfe51a2dd61346719f9acb693'),
])
def test_unified_ledger_matches_pre_refactor_trajectories(policy, expected):
    """Fingerprints captured from 0063f45 before consolidating the controllers."""
    import hashlib
    import json
    _, ledger = make(policy)
    sample, = ledger.sample_batch([request(ledger, count=640)])
    trajectory = [list(sample.token_ids), list(sample.token_logprobs), ledger.native_calls,
                  ledger.events, ledger.decisions, ledger.materialize(sample.token_ids),
                  ledger.statistics(sample.token_ids)]
    actual = hashlib.sha256(json.dumps(trajectory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert actual == expected


@pytest.mark.parametrize(('trigger','threshold','expected'),
    [('surprisal',.5,2),('surprisal',1.5,0),('lowprob',.1,0),('scdrop',.1,0)])
def test_causal_statistic_gates(trigger, threshold, expected):
    _, ledger = make(TuningPolicy(trigger=trigger, threshold=threshold))
    sample, = ledger.sample_batch([request(ledger, count=640)])
    assert len(ledger.materialize(sample.token_ids)[1]) == expected
    assert ledger.work == 601


def test_native_distribution_rows_drive_gate_and_stay_branch_local():
    class Signalled(Native):
        def __init__(self):
            super().__init__(stop=600)
            self.history = {}
        def take(self, request, sample):
            result = super().take(request, sample)
            self.history[request.request_id] = tuple({'entropy': .7, 'top2_gap': .1,
                                                      'p_eos': .01, 'p_think_end': .02}
                                                     for _ in sample.token_ids)
            return result
    _, ledger = make(TuningPolicy(trigger='entropy', threshold=.5), Signalled())
    samples = ledger.sample_batch([request(ledger, count=640, seed=1, name='a'),
                                   request(ledger, count=640, seed=2, name='b')])
    paths = [set(ledger.materialize(s.token_ids)[1]) for s in samples]
    assert len(paths[0]) == len(paths[1]) == 2
    assert paths[0].isdisjoint(paths[1])
    assert all(d['signal']['value'] == pytest.approx(.7) for d in ledger.decisions)


def test_missing_internal_distribution_never_silently_uses_logprob():
    _, ledger = make(TuningPolicy(trigger='entropy'))
    with pytest.raises(KeyError, match='entropy'):
        ledger.sample_batch([request(ledger, count=640)])


def test_initial_and_previous_sc_references_detect_different_changes():
    from types import SimpleNamespace
    # Earlier decline has stabilized; only the initial reference still flags it.
    rows = [{'sc': s, 'lp': -.1} for s in [10.]*32+[8.]*64]
    for reference, expected in [('initial', True), ('previous', False)]:
        proxy = SimpleNamespace(policy=TuningPolicy(trigger='scdrop', sc_reference=reference,
                                                    threshold=.1),
                                recent=lambda branch, count: rows[-count:])
        result, _ = OnlineISLedger.signal(proxy, tuple(range(96)), '')
        assert result == expected


def test_hybrid_combines_sc_decline_with_other_gates():
    from types import SimpleNamespace
    p = TuningPolicy(trigger='hybrid', sc_reference='initial', low_probability=.2,
                     rule_tail=800)
    lp, sc = [-.1]*96, [10.]*32+[8.]*64
    proxy = SimpleNamespace(policy=p, recent=lambda branch,count:
                            [{'sc':s,'lp':v} for s,v in zip(sc[-count:],lp[-count:])])
    got, _ = OnlineISLedger.signal(proxy, tuple(range(96)), '')
    assert got is True
    proxy.policy = replace(p, hybrid_mode='all')
    assert OnlineISLedger.signal(proxy, tuple(range(96)), '')[0] is False


def test_never_control_preserves_checkpoint_splitting_without_retrieving():
    _, ledger = make(TuningPolicy(trigger='never', positions=(32,64,128,512)))
    sample, = ledger.sample_batch([request(ledger, count=640)])
    assert not ledger.events
    assert [c['max_new_tokens'] for c in ledger.native_calls] == [32,32,64,384,128]
    assert len(sample.token_ids) == 601


def test_frozen_is_and_token_zero_are_retained():
    backend, ledger = make(TuningPolicy(trigger='surprisal'), Native(stop=140))
    _, audit = search(ledger)
    assert backend.requests[0].prefix == (1,2)
    assert audit['config']['candidate_count'] == 3
    assert audit['config']['rollout_count'] == 2
    assert audit['config']['block_size'] == 512
    assert audit['config']['reward_temperature'] == .25


def test_eos_does_not_trigger_posthumous_evidence():
    _, ledger = make(TuningPolicy(positions=(64,128)), Native(stop=63, eos=True))
    sample, = ledger.sample_batch([request(ledger, count=640)])
    assert sample.token_ids[-1] == 900
    assert not ledger.events


@pytest.mark.parametrize('policy', [TuningPolicy(), TuningPolicy(trigger='surprisal', threshold=.5),
                                   TuningPolicy(continuation='bridge'), TuningPolicy(query='anchored'),
                                   TuningPolicy(query='anchored', anchor_context='question_tail'),
                                   TuningPolicy(format='document'),
                                   TuningPolicy(trigger='lowprob', statistic_content_only=True, threshold=0),
                                   TuningPolicy(query='sentence'),TuningPolicy(query='sentence_anchored')])
def test_offline_audit_replays_gates_and_detects_tampering(policy):
    import copy
    import json
    from pathlib import Path
    import sys
    from retrieval_fixtures import ReadableNative
    from inference_scaling.retrieval.online_is import run_online_is
    from inference_scaling.shared.rng import SeedStream
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_tuned_online_is import audit_tuned_row
    backend = ReadableNative(stop=180, eos=True)
    cfg = replace(GuidanceConfig(), thinking_tokens=640, answer_tokens=1024, context_tokens=8192)
    result = run_online_is(backend, backend, (1,2), (901,), SeedStream(1).derive('question','synthetic'),
                          cfg, question='Q', facts=Facts(), policy=policy)
    row = {'unit_id':'synthetic', 'arm':'tuned_online', 'question':'Q', 'question_id':'synthetic',
           'seed':1, 'base_ids':[1,2], 'static_packets':[], 'static_injected_ids':[],
           'cost': {'generated_tokens': result['total_generation_tokens']}, **result}
    row = json.loads(json.dumps(row))
    def audit(target):
        failures = []
        audit_tuned_row(target,cfg,lambda ids,skip=False:backend.decode(ids,skip_special_tokens=skip),
                        900,901,lambda ok,label:failures.append(label) if not ok else None)
        return failures
    assert not audit(row)
    if policy.statistic_content_only:
        broken = copy.deepcopy(row)
        broken['decisions'][0]['signal']['content_tokens'] += 1
        assert any('gate_replay' in x for x in audit(broken))
    broken = copy.deepcopy(row)
    broken['decisions'][0]['triggered'] = not broken['decisions'][0]['triggered']
    assert any('gate_replay' in x for x in audit(broken))
    broken = copy.deepcopy(row)
    broken['events'][0]['injected_ids'][0] += 1
    assert any('packet_prefix' in x for x in audit(broken))


@pytest.mark.parametrize(('trigger', 'expected'), [('lowprob', .5), ('surprisal', 1.05),
                                                  ('entropy', 2.), ('margin', .6)])
def test_content_statistic_excludes_stopwords_and_punctuation_from_denominator(trigger, expected):
    from types import SimpleNamespace
    pieces = {1: ' the', 2: ',', 3: ' APAR', 4: ' 64'}
    rows = [{'token': i, 'lp': lp, 'entropy': ent, 'top2_gap': gap}
            for i, lp, ent, gap in [(1,-9.,9.,0.), (2,-8.,8.,0.), (3,-2.,3.,.2), (4,-.1,1.,.6)]]
    proxy = SimpleNamespace(policy=TuningPolicy(trigger=trigger, threshold=.4, statistic_content_only=True),
                            recent=lambda branch,count:rows,
                            backend=SimpleNamespace(decode=lambda ids,**kw: ''.join(pieces[i] for i in ids)))
    gate, signal = OnlineISLedger.signal(proxy, (1,2,3,4), '')
    assert gate and signal['value'] == pytest.approx(expected)
    assert signal['window_tokens'] == 4 and signal['content_tokens'] == 2
    rows[:] = rows[:2]
    gate, signal = OnlineISLedger.signal(proxy, (1,2), '')
    assert gate is False and signal['value'] is None and signal['insufficient_content']
    assert signal['window_tokens'] == 2 and signal['content_tokens'] == 0


@pytest.mark.parametrize('trigger', ['fixed', 'hybrid', 'scdrop', 'dragin', 'risk'])
def test_content_filter_rejects_unimplemented_statistics(trigger):
    with pytest.raises(ValueError, match='supported window statistic'):
        TuningPolicy(trigger=trigger, statistic_content_only=True)


@pytest.mark.parametrize('value', [0, 1, 'false', None])
def test_content_filter_requires_boolean(value):
    with pytest.raises(ValueError, match='boolean'):
        TuningPolicy(statistic_content_only=value)
