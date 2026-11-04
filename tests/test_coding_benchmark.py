"""Experiment guards: private data, deterministic submissions, paired modes and resumes."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from coding_common import MODES, PublicCodingDataset, completed_keys, extract_code
from benchmark_is_gpu import select_mode
from types import SimpleNamespace


def test_final_only_extraction_matches_upstream():
    assert extract_code('Explanation\n```python\nfirst\n```\n```python\nlast\n```') == 'last'
    assert extract_code('def f(): return 1') == ''
    assert extract_code('```python\nunclosed') == ''


def test_private_fields_are_refused(tmp_path):
    path = tmp_path / 'manifest.json'
    row = {'id': 'he/0', 'prompt': 'public', 'global_index': 0}
    path.write_text(json.dumps({'problems': [{**row, 'canonical_solution': 'secret'}]}))
    with pytest.raises(ValueError, match='unknown fields'):
        PublicCodingDataset(path)
    path.write_text(json.dumps({'problems': [row, {**row, 'id': 'he/1', 'global_index': 1}]}))
    a, b = PublicCodingDataset(path, shard=0, shards=2), PublicCodingDataset(path, shard=1, shards=2)
    assert {p.id for p in a.problems} == {'he/0'}
    assert {p.id for p in b.problems} == {'he/1'}
    assert not a.oracle and a.problems[0].answer == ''


def test_resume_cannot_mix_changed_or_duplicate_measurements():
    row = {'kind': 'measurement', 'status': 'complete', 'problem_id': 'he/0', 'mode': 'base'}
    rows = [{'kind': 'metadata', 'identity': 'frozen'}, row]
    assert completed_keys(rows, 'frozen') == {('he/0', 'base')}
    with pytest.raises(ValueError, match='identity'):
        completed_keys(rows, 'changed')
    with pytest.raises(ValueError, match='duplicate'):
        completed_keys(rows + [row], 'frozen')
    assert completed_keys(rows + [{**row, 'status': 'failed'}], 'frozen') == {('he/0', 'base')}


def test_twelve_modes_keep_kernel_and_reward_pairs_aligned():
    settings = {'ar': {'algorithms': {'sample': {}, 'best_of_n': {'samples': 4},
                'is': {'candidate_count': 4, 'rollout_count': 1, 'selection_ready': True,
                       'selection': 'sir', 'exploration_schedule': 'two_stage'}}},
                'rewards': {'consilience': {'scope': 'thinking'}, 'self_certainty': {'scope': 'thinking'}}}
    family = SimpleNamespace(ar=settings['ar'])
    configurations = {}
    for mode in MODES:
        select_mode(family, mode, settings, study='selection')
        configurations[mode] = family.config.copy()
        assert family.choices.reward != 'verifier'
    for mode in ('is', 'met-k1', 'met-k2', 'block-met-k2'):
        assert configurations[mode] == configurations[mode + '-self-certainty']
    assert configurations['met-k2']['rollout_count'] == 2
    assert configurations['block-met-k2']['selection'] == 'block_metropolized'
    assert configurations['bon-consilience']['samples'] == 4
