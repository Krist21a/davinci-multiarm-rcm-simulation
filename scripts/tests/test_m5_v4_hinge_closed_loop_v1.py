"""Negative controls for row-10 inverse, optical permits, and recovery evidence."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision','analysis')]
import m5_v4_hinge_controller_v1 as ctl
from m5_v4_hinge_work_v1 import inverse, margin
from score_m5_v4_hinge_closed_loop_v1 import exercise


class HingeLoopTests(unittest.TestCase):
    def test_no_inverse_for_missing_or_rank_one_connected_bank(self):
        p = np.array([0.,0.,.14])
        z = np.full((21,21),.14)
        m = np.zeros_like(z,bool)
        m[10] = True
        self.assertIsNone(inverse(p,z,m,0.,900))
        m[:] = True
        np.testing.assert_allclose(inverse(p,z,m,0.,900),p)
        self.assertAlmostEqual(margin(.0005),.00022)
        with self.assertRaises(ValueError):
            margin(.001)

    def test_one_reader_is_not_a_withdrawal_permit(self):
        f = SimpleNamespace(rgb=None,depth=None,far=.3)
        with patch.object(ctl,'frontier',return_value=dict(verdict='TARGET',prefix_mm=9)), \
             patch.object(ctl,'full_review',return_value=dict(verdict='UNKNOWN',prefix_mm=None)):
            self.assertEqual(ctl.receipt(f,7,0.),('UNKNOWN',None))

    def test_injected_slip_not_recovery_without_new_optical_work(self):
        b = dict(closed=True,q=[0]*6,held={'psm3':3})
        event = dict(bites=2,before=b,after=dict(b,held={'psm3':0}),recovered=b,tick=8,state='RECOVER_GRASP')
        reviews = {'r':dict(prefix=5,closures=2,shutter=False,tick=9)}
        r = dict(event=event,captures={'w':dict(closures=2,tick=8)})
        pred = {'w':dict(actual=True)}
        self.assertFalse(exercise('slip_regrasp',r,pred,reviews))
        r['captures']['w']['tick'] = 10
        self.assertTrue(exercise('slip_regrasp',r,pred,reviews))
