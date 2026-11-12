"""Distinguish real EOS sources from the runner's controller-EOS sentinel."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

analysis_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(analysis_root / 'scripts' if (analysis_root / 'scripts').is_dir() else analysis_root))
from diagnose_literature_empty_answers import EOS, classify


def fixture():
    row = {'unit_id': 'q:0:thinkretrieve', 'question_id': 'q', 'seed': 0, 'arm': 'thinkretrieve',
        'answer': {'empty_answer': True, 'answer_text': '', 'eos_inside_thinking': True,
                   'answer_ids': [], 'calls': [], 'truncated': False},
        'handoff': {'terminal_eos': True, 'controller_calls': [{'token_ids': [1, 2, EOS]}],
                    'injected_ids': [[10, 11]], 'interim_calls': [{'token_ids': [20, EOS]}]},
        'audit': {'terminal_controller_eos': True}, 'probe': None,
        'candidates': [[EOS]], 'carried': {'sc': [1, 1]}}
    audited = {'empty': True, 'f1': 0, 'generation_tokens': 5, 'seconds': 1.5}
    return row, audited


class EmptyDiagnosticsTest(unittest.TestCase):
    def test_controller_eos_sentinel_is_not_reported_as_is_output(self):
        row, audited = fixture()
        event = classify(row, audited)
        self.assertEqual(event['terminal_phase'], 'controller_thought_eos')
        self.assertEqual(event['terminal_segment_tokens'], 3)
        self.assertEqual(event['terminal_eos_position_zero_based'], 2)
        self.assertEqual(event['selected_suffix_generated_tokens'], 0)

    def test_selected_search_and_final_readout_stops_are_distinct(self):
        row, audited = fixture()
        row['handoff']['terminal_eos'] = False
        row['audit'] = {}
        self.assertEqual(classify(row, audited)['terminal_phase'], 'is_selected_thought_eos')
        row['answer']['eos_inside_thinking'] = False
        row['answer']['answer_ids'] = [EOS]
        row['answer']['calls'] = [{'token_ids': [EOS]}]
        self.assertEqual(classify(row, audited)['terminal_phase'], 'final_answer_empty_stop')
        row['answer']['empty_answer'] = False
        self.assertIsNone(classify(row, audited))

    def test_checkpoint_is_not_mislabeled_as_search(self):
        row, audited = fixture()
        row = deepcopy(row)
        row['handoff']['terminal_eos'] = False
        row['audit'] = {'terminal_checkpoint': True}
        row['probe'] = {'reason_ids': [1, EOS]}
        self.assertEqual(classify(row, audited)['terminal_phase'], 'checkpoint_thought_eos')


if __name__ == '__main__':
    unittest.main()
