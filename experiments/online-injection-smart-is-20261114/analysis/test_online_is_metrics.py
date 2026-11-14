from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_online_is import ARMS, aggregate


def fixture():
    rows = []
    for qi, q in enumerate(['q1', 'q2']):
        for seed in [11, 12]:
            for i, arm in enumerate(ARMS):
                rows.append({'question_id': q, 'seed': seed, 'arm': arm,
                    'f1': 10*qi+seed+(4 if arm == 'online_is' else 0),
                    'exact_match': False, 'generated_tokens': 100*(i+1),
                    'search_tokens': 80*(i+1), 'answer_tokens': 20*(i+1),
                    'seconds': (qi+1)*(i+1), 'retrieval_seconds': .1,
                    'selected_injected_tokens': 256 if i > 1 else 0,
                    'retrieval_calls': int(i > 1), 'native_prefill_tokens': 100,
                    'empty': False, 'truncated': q == 'q1' and seed == 11 and arm == 'online_is',
                    'thinking_eos': False, 'forced_thinking_end': False})
    return rows


def test_metrics_preserve_pairs_denominators_and_truncations():
    summary = aggregate(fixture())
    assert summary['questions'] == 2 and summary['measurements'] == 24
    online = next(r for r in summary['arms'] if r['arm'] == 'online_is')
    assert online['f1'] == 20.5 and online['truncated'] == 1
    assert online['tokens_relative_direct'] == online['latency_relative_direct'] == 5
    assert online['paired_latency_geomean_relative_direct'] == pytest.approx(5)
    pair = summary['paired_f1'][0]
    assert pair['difference'] == 4 and pair['bootstrap_95'] == [4, 4]
    assert pair['n_questions'] == 2 and pair['wins_ties_losses'] == [2, 0, 0]


@pytest.mark.parametrize('alter', ['duplicate', 'missing'])
def test_reject_incomplete_or_duplicate_metric_cohort(alter):
    rows = fixture()
    if alter == 'duplicate':
        rows[-1] = rows[0]
    else:
        rows.pop()
    with pytest.raises(AssertionError):
        aggregate(rows)
