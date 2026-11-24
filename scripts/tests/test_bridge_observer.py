"""A missing material bridge is only a stop with two labelled neighbours and deep RGB-D."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from bridge_observer import COLORS, measure
from bridge_fracture import FRACTURE_COLORS, measure as fracture_measure


class BridgeObserverTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((160, 200, 3), [225, 77, 112], dtype=np.uint8)
        self.z = np.full((160, 200), .135, dtype=float)
        self.estimate = dict(
            scissors=dict(valid=True), tissue=dict(valid=True),
            cut=dict(front_valid=True, mode='slit', slit_px=30,
                     front=[.0027, .00135, .135]))
        for i, color in enumerate(COLORS):
            self.rgb[78:83, 78+i*7:83+i*7] = np.rint(color).astype(np.uint8)

    def probe(self):
        return measure(self.rgb, self.z, np.eye(3), np.zeros(3), 100., self.estimate)

    def fracture(self):
        return fracture_measure(self.rgb, self.z, np.eye(3), np.zeros(3), 100., self.estimate)

    def test_seven_live_bonds_exclude_the_terminal_cut(self):
        self.assertEqual(self.probe()['status'], 'NOT_STOP')

    def test_one_identified_severed_bridge_with_depth_gap(self):
        self.rgb[78:83, 99:104] = [70, 45, 38]
        self.z[78:83, 99:104] = .21
        result = self.probe()
        self.assertEqual(result['status'], 'STOP_CANDIDATE')
        self.assertIn(11, result['absent_lanes'])

    def test_occlusion_or_unidentified_outer_bank_is_not_a_stop(self):
        self.rgb[78:83, 99:104] = [20, 20, 20]
        self.z[78:83, 99:104] = .12
        self.assertEqual(self.probe()['status'], 'UNKNOWN')
        self.rgb[78:83, 78:83] = [70, 45, 38]
        self.z[78:83, 78:83] = .21
        self.assertEqual(self.probe()['status'], 'UNKNOWN')
        self.estimate['scissors']['valid'] = False
        self.assertEqual(self.probe()['status'], 'UNKNOWN')

    def test_missing_intact_bridge_with_deep_background_is_not_a_fracture(self):
        self.rgb[78:83, 99:104] = [70, 45, 38]
        self.z[78:83, 99:104] = .21
        self.assertEqual(self.probe()['status'], 'STOP_CANDIDATE')
        self.assertEqual(self.fracture()['status'], 'UNKNOWN')

    def test_two_exposed_physical_flanks_require_direct_gap(self):
        self.rgb[78:83, 99:104] = [70, 45, 38]
        self.z[78:83, 99:104] = .21
        self.rgb[78:83, 95:98] = np.rint(FRACTURE_COLORS[0]).astype(np.uint8)
        self.rgb[78:83, 105:108] = np.rint(FRACTURE_COLORS[1]).astype(np.uint8)
        self.assertEqual(self.fracture()['status'], 'STOP_CANDIDATE')
        self.z[78:83, 99:104] = .135
        self.assertEqual(self.fracture()['status'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
