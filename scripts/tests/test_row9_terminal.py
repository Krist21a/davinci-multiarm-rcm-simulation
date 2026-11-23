"""Real RGB-D positive and fail-closed counterexamples for the passive row-9 probe."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from row9_terminal import measure


class Row9TerminalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT / 'validation/task/m5_live_active_look_v1'
        report = json.loads((folder / 'nominal.json').read_bytes())
        cls.estimate = {row['tick']: row['estimate'] for row in report['perception']}
        cls.folder = folder / 'nominal_hd'

    def observe(self, tick, *, edit=None, depth_bad=False):
        with np.load(self.folder / f'{tick:03d}.npz') as image:
            rgb = image['rgb']
            depth = image['depth'].copy()
            if depth_bad:
                depth[:] = np.nan
            estimate = copy.deepcopy(self.estimate[tick])
            if edit:
                edit(estimate)
            return measure(rgb, depth, image['cam_R'], image['cam_t'],
                           914.0147243149534, estimate)

    def test_actual_exposures_do_not_derive_a_stop_from_cut_count(self):
        for tick in (31, 32):
            result = self.observe(tick)
            self.assertEqual(result['status'], 'STOP_CANDIDATE')
            self.assertTrue(all(bank['pairs'] >= 3 for bank in result['banks']))
        # Both later exposures still have structural length 9 mm; a rendered
        # slit alone must not make them an unconditional terminal signal.
        self.assertNotEqual(self.observe(38)['status'], 'STOP_CANDIDATE')

    def test_shutter_stale_front_and_absent_depth_fail_closed(self):
        self.assertEqual(self.observe(31, edit=lambda e: e['cut'].update(mode='memory'))['status'], 'UNKNOWN')
        self.assertEqual(self.observe(31, edit=lambda e: e['scissors'].update(valid=False))['status'], 'UNKNOWN')
        self.assertEqual(self.observe(31, depth_bad=True)['status'], 'UNKNOWN')
