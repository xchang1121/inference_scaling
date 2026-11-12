"""Synthetic-only checks for paired visual summaries; no model or DEV outputs."""
from copy import deepcopy
from pathlib import Path
import statistics
import sys
import unittest

analysis_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(analysis_root / 'scripts' if (analysis_root / 'scripts').is_dir() else analysis_root))
from plot_literature_retrieval import ARMS, validate


def fixture():
    records = []
    for question in range(16):
        for seed in range(2):
            for index, arm in enumerate(ARMS):
                empty = arm.endswith('_noise') and question % 5 == 0 and seed == 0
                components = {'controller_tokens': 40 * index + question,
                              'probe_tokens': 64 if index else 0,
                              'search_tokens': 800 + 700 * index + 20 * question,
                              'answer_tokens': 0 if empty else 120 + question}
                total = sum(components.values())
                records.append({'question_id': f'SYNTHETIC_Q{question:02}', 'seed': seed,
                    'arm': arm, 'f1': 0 if empty else 15 + 2 * question + 2 * index + seed,
                    'generation_tokens': total, 'seconds': total / 100 + (question + seed) / 10 + 1,
                    'empty': empty, 'truncated': False, 'thinking_eos': empty, **components})
    means = []
    for arm in ARMS:
        rows = [r for r in records if r['arm'] == arm]
        means.append({'arm': arm, 'n': len(rows), 'f1': statistics.fmean(r['f1'] for r in rows),
            **{'mean_' + metric: statistics.fmean(r[metric] for r in rows) for metric in
               ('generation_tokens', 'seconds', 'controller_tokens', 'probe_tokens',
                'search_tokens', 'answer_tokens')},
            **{metric: sum(r[metric] for r in rows) for metric in ('empty', 'truncated', 'thinking_eos')}})
    for mean in means:
        mean['latency_ratio_of_means'] = mean['mean_seconds'] / means[0]['mean_seconds']
    summary = {'stage': 'formal', 'audit': {'passed': True}, 'measurements': len(records),
               'questions': 16, 'seeds': [0, 1], 'summary': means}
    return summary, records


class PlotsTest(unittest.TestCase):
    def test_paired_means_include_empty_answers_and_match_question(self):
        summary, records = fixture()
        _, questions, pairs = validate(summary, records)
        self.assertEqual(len(questions), 16)
        keyed = {(row['question_id'], row['arm']): row for row in pairs}
        self.assertTrue(all(keyed[q, 'closed_is']['difference_vs_closed_is'] == 0 for q in questions))
        self.assertEqual(keyed['SYNTHETIC_Q01', 'ircot']['difference_vs_closed_is'], 6)
        row = keyed['SYNTHETIC_Q00', 'ircot_noise']
        self.assertEqual(row['empty_of_two'], 1)
        self.assertEqual(row['mean_f1'], 13)
        self.assertEqual(row['difference_vs_closed_is'], -4.5)

    def test_incomplete_or_inconsistent_plot_data_is_rejected(self):
        summary, records = fixture()
        with self.assertRaises(AssertionError):
            validate(summary, records[:-1])
        bad = deepcopy(records)
        bad[-1] = deepcopy(bad[0])
        with self.assertRaises(AssertionError):
            validate(summary, bad)
        bad = deepcopy(summary)
        bad['summary'][0]['mean_generation_tokens'] += 1
        with self.assertRaises(AssertionError):
            validate(bad, records)
        bad = deepcopy(summary)
        bad['summary'][2]['latency_ratio_of_means'] *= 2
        with self.assertRaises(AssertionError):
            validate(bad, records)


if __name__ == '__main__':
    unittest.main()
