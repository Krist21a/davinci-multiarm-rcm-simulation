"""Partial damage, optical refusal diagnostics and FK-independent placement."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/task'),str(ROOT/'scripts/vision')]
from m5_v3_controller_v7 import Refused, World
from m5_v3_frozen_fixture import place_sheet
from seam_damage_v2 import DamageBaseline, exact_endpoint, within_envelope
from workspace import NU, NV


class OuterSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        edges = set()
        for j in range(NV):
            for i in range(NU):
                a = j*NU+i
                if i+1<NU:
                    edges.add((a,a+1))
                if j+1<NV:
                    edges.add((a,a+NU))
                    if i+1<NU:
                        edges.add((a,a+NU+1))
                        if i!=11:
                            edges.add((a+1,a+NU))
        cls.uncut = np.asarray([(*edge,1) for edge in sorted(edges)])

    def test_single_distal_edge_refuses_even_if_full_row_metric_unchanged(self):
        cut = self.uncut.copy()
        a = 10*NU+12
        cut[(cut[:,0]==a)&(cut[:,1]==a+NU),2] = 0
        state = DamageBaseline(self.uncut).score(cut[::-1])
        self.assertEqual((state['prefix_rows'],state['farthest_row_end']), (0,0))
        self.assertEqual(state['faces'][10],(False,True))
        self.assertEqual((state['farthest_partial_row_end'],state['newly_dead_edges']),(11,1))
        self.assertFalse(within_envelope(state,10))
        self.assertFalse(exact_endpoint(dict(state,prefix_rows=10,farthest_row_end=10),10))
        with self.assertRaises(ValueError):
            DamageBaseline(cut)

    def test_off_axis_non_face_edge_and_topology_mismatch(self):
        cut = self.uncut.copy()
        cut[(cut[:,0]==11*NU+1)&(cut[:,1]==12*NU),2] = 0
        baseline = DamageBaseline(self.uncut)
        self.assertEqual(baseline.score(cut)['farthest_partial_row_end'],12)
        cut[-1] = cut[0]
        with self.assertRaises(ValueError):
            baseline.score(cut)

    def test_fresh_optical_refusal_distinct_from_stale_chart(self):
        w = World.__new__(World)
        w.tick,w.epoch,w.sigma,w.goal_mm = 15,2,0.,9
        w.capture = Mock(return_value=SimpleNamespace(rgb=None,depth=None,focal=1,far=1,tick=15))
        banks = [{8:1,9:1},{8:1,9:1,10:1}]
        with patch('m5_v3_controller_v7.perception._solve_work',return_value=(None,banks,())):
            with self.assertRaisesRegex(Refused,r'optical work refused.*L \[10\]'):
                w.current(w.work(True,7))
        with self.assertRaisesRegex(Refused,'stale working geometry'):
            w.current(SimpleNamespace(tick=15,epoch=1))

    def test_placement_requires_no_fk_or_frame(self):
        spec = json.loads((ROOT/'config/task/m5_v3_frozen_fixture_v1.json').read_bytes())
        w = SimpleNamespace(canonical_seam=True,roi_proxy=None,primary='psm1',assist='psm3',
                            pretension=.08,sheet_cfg={'guide_band_m':.0001},steps=Mock(),call=Mock(return_value=[]))
        place_sheet(w,spec)
        self.assertEqual(w.call.call_args_list[0].args[1]['origin'],spec['sheet_origin'])
        self.assertEqual(w.grasp_points['psm1'].tolist(),spec['grasp_points']['psm1'])
        w.steps.assert_called_once_with(30)
