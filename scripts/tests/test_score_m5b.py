"""Protect the strict B5 and pre-closure evidence rules."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task'))
from score_m5b import closure_metric, evaluate, safe_outcome


class M5bScoringTests(unittest.TestCase):
    def test_b5_does_not_count_truth_terminal_rejection_or_runtime_error(self):
        run = dict(final_state='FAILED', final_observation=dict(torn=0, cut_length_m=0.008, faults=[]))
        self.assertFalse(safe_outcome(dict(run, reason='cut length below target'), 0.009, 0.010))
        self.assertTrue(safe_outcome(dict(run, reason='scissors retraction refused'), 0.009, 0.010))
        self.assertFalse(safe_outcome(dict(run, reason='camera adjustment exhausted', error='runtime'), 0.009, 0.010))

    def test_b7_requires_every_scoring_sample_in_the_pool(self):
        for bad in (None, float('nan')):
            self.assertFalse(closure_metric([0.0004, bad], 2, 0.0015)['result'])
        self.assertFalse(closure_metric([], 0, 0.0015)['result'])
        self.assertTrue(closure_metric([0.0004, 0.001], 2, 0.0015)['result'])
        self.assertFalse(closure_metric([False], 1, 0.0015)['result'])

    def test_missing_visual_results_cannot_pass_cut_length_or_safety(self):
        result = evaluate({}, {}, dict(cut=dict(length_m=0.009, max_overcut_m=0.001, align_tolerance_m=0.0015),
                                      vision=dict(fix_max_age_ticks=6)))
        self.assertFalse(result['B3']['result'])
        self.assertEqual(result['B3']['status'], 'NO_EVIDENCE')
        self.assertFalse(result['B5']['result'])
