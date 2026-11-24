"""Coverage for evidence that must not be credited to the wrong edge or cut."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task'))
from score_ecm_view_study import closure_checks, paired_columns


class ViewEvidenceTests(unittest.TestCase):
    def test_disjoint_edges_and_wrong_depth_do_not_establish_paired_column(self):
        rgb = np.zeros((40, 40, 3))
        depth = np.ones((40, 40))
        f = 20.0
        nodes = np.array([[[(20 - u) / f, (20 - v) / f, 1.0] for u in (9, 28)] for v in (9, 28)])
        near_left, far_right, far_left = np.s_[6:13, 6:13], np.s_[25:32, 25:32], np.s_[25:32, 6:13]
        rgb[near_left] = rgb[far_right] = [220, 100, 140]
        frame = dict(rgb=rgb, depth=depth, cam_R=np.eye(3), cam_t=np.zeros(3), nodes=nodes)
        meta = dict(NU=2, NV=2, f=f)
        spec = dict(pixel_radius=2, depth_tolerance_m=0.0015, paired_strip_columns=[0, 1])
        self.assertEqual(paired_columns(frame, meta, spec), [])
        rgb[far_left] = [220, 100, 140]
        depth[far_left] = 0.9
        self.assertEqual(paired_columns(frame, meta, spec), [])
        depth[far_left] = 1.0
        self.assertEqual(paired_columns(frame, meta, spec), [0])

    def test_memory_or_late_reacquisition_cannot_cover_multiple_closures(self):
        close = dict(event='closure', fix_age=1, scissors_valid=True)
        regain = dict(event='post_closure_reacquired')
        events = [dict(close, tick=2), dict(regain, tick=3, mode='memory'), dict(close, tick=4),
                  dict(regain, tick=5, mode='bank_slit'), dict(close, tick=6, scissors_valid=False)]
        checks = closure_checks(events, 6)
        self.assertEqual(checks['missing_reacquisition'], [2, 6])
        self.assertEqual(checks['invalid_closures'], [6])
        self.assertEqual([p['closure_tick'] for p in checks['paired']], [4])
