from types import SimpleNamespace
import json

import pytest

from inference_scaling.retrieval.common import token_hash
from inference_scaling.retrieval.corpus import Document, Retriever
from inference_scaling.retrieval.evidence import EvidenceConfig, FactBank, FactRanking


class TextBackend:
    tokenizer = SimpleNamespace(eos_token_id=100000,
                                convert_tokens_to_ids=lambda s: 100001 if s == '<think>' else 100002)

    def encode(self, text, **kwargs):
        return tuple(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


def test_retrievers_and_stable_exclusion():
    docs = [Document('v2', 'Mercury API v2', 'Mercury API v2 retry timeout lease: 27 seconds. '*8),
            Document('v1', 'Mercury API v1', 'Mercury v1 pagination cursor lasts 8 seconds. '*8),
            Document('noise', 'Other system', 'An unrelated archive of historical notes.')]
    for kind in ('bm25', 'char', 'hybrid'):
        retriever = Retriever(docs, kind=kind)
        assert retriever.search('Mercury API v2 retry timeout', k=1)[0].document.id == 'v2'
        assert 'v2' not in [h.document.id for h in retriever.search('Mercury', exclude=frozenset({'v2'}))]
        assert retriever.fingerprint == Retriever(list(reversed(docs)), kind=kind).fingerprint
        assert retriever.search('') == []


def test_corpus_refuses_qa_annotations(tmp_path):
    path = tmp_path/'corpus.jsonl'
    path.write_text(json.dumps({'id': 'a', 'title': 't', 'text': 'x', 'answer': 'secret'}), encoding='utf8')
    with pytest.raises(ValueError, match='document fields'):
        Retriever.from_jsonl(path)


@pytest.mark.parametrize('ranking', [None, FactRanking(), FactRanking(window_kind='char'),
                                   FactRanking(window_kind='hybrid', max_windows_per_parent=1)])
def test_shared_window_retrieval_keeps_exclusion_budget_and_source_identity(ranking):
    backend = TextBackend()
    docs = [Document('fix', 'Database fix', 'Restart database after changing cache limit. '*8, 'manual'),
            Document('guide', 'Database guide', 'Inspect database cache limit before restarting. '*8, 'guide')]
    cfg = EvidenceConfig(fact_tokens=112, window_tokens=64, window_stride=32)
    bank = FactBank(backend, Retriever(docs), cfg, ranking=ranking)
    first, log = bank.retrieve('database cache limit', k=2)
    assert first and log['selected_ids'] == [p['id'] for p in first]
    assert len({p['id'] for p in first}) == len(first)
    for packet in first:
        assert len(packet['ids']) <= cfg.fact_tokens
        assert packet['sha256'] == token_hash(packet['ids'])
        assert 0 < packet['body_start'] < packet['body_stop'] < len(packet['ids'])
        assert packet['source'] in {'manual', 'guide'}
    second, _ = bank.retrieve('database cache limit', k=2, exclude=frozenset(log['selected_ids']))
    assert not set(log['selected_ids']).intersection(p['id'] for p in second)
    if ranking and ranking.max_windows_per_parent == 1:
        assert len({p['id'].split('@')[0] for p in first}) == len(first)
        assert not second
