"""Opt-in bilateral-surface plumbing for CutPerception (no simulator).

The measurement algorithm in bilateral_surface is tested elsewhere; these cover
only the opt-in flag, the early dispatch, and the camera-to-world transport.
The legacy estimator path stays covered by test_cut_perception.py.

Run: python -m unittest discover -s scripts/tests -p test_bilateral_integration.py -v
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vision'))

import cut_perception  # noqa: E402
from cut_perception import CutPerception  # noqa: E402

MEASURED = dict(front_valid=True, front=[.001, 0, .1], normal=[0, 0, -1],
                direction=[1, 0, 0], progress_m=.008, banks_visible=True,
                sweep_limit_from_front_m=.0015)


class BilateralOptInTests(unittest.TestCase):
    def test_from_meta_defaults_off_and_forwards_flag(self):
        meta = dict(W=16, H=12, f=100., far=.3, rcm_world=[0, 0, 0])
        self.assertIs(CutPerception.from_meta(dict(meta)).bilateral_surface, False)
        self.assertIs(CutPerception.from_meta(dict(meta, bilateral_surface=True)).bilateral_surface, True)

    def test_transport_rotates_positions_and_vectors_into_world(self):
        est = CutPerception(64, 48, 100., .3, [0, 0, 0], bilateral_surface=True)
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)   # +90 deg about z
        t = np.array([1., 2., 3.])
        tool = ({'valid': True, 'tip': [0, 0, .1], 'jaw_axis': [1, 0, 0]}, None)
        with patch('cut_perception.measure_bilateral', return_value=dict(MEASURED)), \
                patch.object(est, '_scissors', return_value=tool):
            out = est.estimate(np.zeros((48, 64, 3)), np.full((48, 64), .1), R, t)
        self.assertTrue(np.allclose(out['cut']['front'], [1, 2.001, 3.1]))
        self.assertTrue(np.allclose(out['cut']['direction'], [0, 1, 0]))
        self.assertTrue(np.allclose(out['scissors']['jaw_axis'], [0, 1, 0]))
        self.assertTrue(np.allclose(out['scissors']['tip'], [1, 2, 3.1]))
        self.assertTrue(np.allclose(out['tissue']['normal'], [0, 0, -1]))
        self.assertEqual(out['cut']['progress_m'], .008)
        self.assertEqual(out['cut']['sweep_limit_from_front_m'], .0015)
        self.assertIs(out['tissue']['valid'], True)
        self.assertIs(out['tension_proxy']['valid'], False)

    def test_opt_in_blank_frame_refuses_and_never_reuses_a_prior_estimate(self):
        est = CutPerception(64, 48, 100., .3, [0, 0, 0], bilateral_surface=True)
        rgb, depth = np.zeros((48, 64, 3)), np.full((48, 64), .1)
        eye, zero = np.eye(3), np.zeros(3)
        with patch('cut_perception.measure_bilateral', return_value=dict(MEASURED)):
            first = est.estimate(rgb, depth, eye, zero)
        self.assertEqual(first['cut']['progress_m'], .008)   # a fresh estimate was returned once
        out = est.estimate(rgb, depth, eye, zero)            # real measure on blank input
        self.assertIsNone(out['cut'].get('front'))
        self.assertIsNone(out['cut'].get('progress_m'))
        self.assertFalse(out['scissors'].get('tip'))
        self.assertFalse(out['tissue']['valid'])
        self.assertFalse(out['tension_proxy']['valid'])

    def test_junction_flag_requests_boundary_strategy(self):
        for flag, want in ((True, False), ('junction', True)):
            with self.subTest(flag=flag):
                est = CutPerception(64, 48, 100., .3, [0, 0, 0], bilateral_surface=flag)
                with patch('cut_perception.measure_bilateral', return_value=dict(MEASURED)) as measure, \
                        patch.object(est, '_scissors', return_value=({'valid': False}, None)):
                    est.estimate(np.zeros((48, 64, 3)), np.full((48, 64), .1), np.eye(3), np.zeros(3))
                measure.assert_called_once()
                self.assertIs(measure.call_args.kwargs['junctions'], want)

    def test_default_estimator_never_calls_bilateral_measure(self):
        est = CutPerception(64, 48, 100., .3, [0, 0, 0])
        with patch('cut_perception.measure_bilateral') as measure:
            out = est.estimate(np.zeros((48, 64, 3)), np.full((48, 64), .1),
                               np.eye(3), np.zeros(3))
        measure.assert_not_called()
        self.assertEqual(set(out), {'tissue', 'scissors', 'cut', 'tension_proxy'})


if __name__ == '__main__':
    unittest.main()
