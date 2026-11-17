import json
from types import SimpleNamespace

import pytest

from retrieval_fixtures import Native
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.retrieval.native_journal import NativeJournal


def test_journal_preserves_successes_before_a_later_failure(tmp_path):
    backend = Native()
    original = backend.sample_batch
    count = 0
    def failing(requests):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('synthetic second call failure')
        return original(requests)
    backend.sample_batch = failing
    path = tmp_path/'native.jsonl'
    journal = NativeJournal(backend, SimpleNamespace(rows={}), path)
    request = GenerationRequest((1,2), 4, SamplingConfig(eos_token_id=900), 7, 'test')
    result = backend.sample_batch([request])
    with pytest.raises(RuntimeError, match='synthetic'):
        backend.sample_batch([request])
    journal.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['kind'] for r in rows] == ['native_start','native_done','native_start','native_failed']
    assert rows[1]['samples'][0]['token_ids'] == list(result[0].token_ids)
    assert backend.sample_batch is failing
