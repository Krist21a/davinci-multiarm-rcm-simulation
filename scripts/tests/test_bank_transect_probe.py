"""Scoring-only geometry: a folded sheet is not an affine material ruler."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import probe_bank_transects as probe
from check_terminal_band_contact import contacts


class TransectTests(unittest.TestCase):
    def test_contact_on_both_signs_is_not_two_material_banks(self):
        gap = np.array([[True, False, True]])
        terminal = np.array([[False, True, False]])
        cue = contacts(gap, np.array([[0., 1., 2.]]), terminal)
        self.assertTrue(cue['bilateral'])
        self.assertEqual(cue['positive_uv'], cue['negative_uv'])
        self.assertEqual(cue['gap_touching_both_uv'], [])

    def test_requires_adjacent_rows_same_cell_and_both_identified_ends(self):
        sides = np.r_[np.linspace(-.007, -.002, 8), np.linspace(.002, .007, 8)]
        rows = [np.column_stack((sides, k+.1*sides)) for k in probe.MC.KNOTS_M]
        self.assertAlmostEqual(probe.read_banks(rows, 0., .005, 'slit')['apparent_progress_mm'], 5.)
        disagree = [q.copy() for q in rows]
        for q in disagree:
            q[q[:, 0] > 0, 1] += .001
        self.assertNotIn('apparent_progress_mm', probe.read_banks(disagree, 0., .005, 'slit'))
        rows[2] = rows[2][sides > 0]
        self.assertNotIn('apparent_progress_mm', probe.read_banks(rows, 0., .005, 'slit'))
        rows[-1] = rows[-1][sides < 0]
        self.assertNotIn('apparent_progress_mm', probe.read_banks(rows, 0., 0., 'pre_cut'))

    def test_surface_pixel_uses_perspective_and_rejects_occlusion_and_dead_faces(self):
        # Pixel (3,3) has projected barycentrics (1/2,1/4,1/4)
        # in a triangle with depths (.1,.1,.2) and rows (0,0,1).
        uv = np.array([[2,2], [6,2], [1,6], [2,6]])
        z = np.array([.1, .1, .2, .2])
        xyz = np.column_stack(((4-uv)*z[:, None]/10, z))
        frame = dict(nodes=xyz.reshape(2, 2, 3), tissue_alive=np.array([[0,1,1],[1,3,1],[0,3,1]]),
                     cam_t=np.zeros(3), cam_R=np.eye(3), depth=np.full((8,8), .3))
        lab = np.full((8,8), -1)
        lab[3,3] = 0
        frame['depth'][3,3] = 1/(.75/.1+.25/.2)
        self.assertAlmostEqual(probe.oracle(frame, [3,3], 10, lab)[0]['material_mm'], 1/7)
        frame['depth'][3,3] -= .01
        self.assertFalse(probe.oracle(frame, [3,3], 10, lab))
        frame['depth'][3,3] += .01
        frame['tissue_alive'][1,2] = 0
        self.assertFalse(probe.oracle(frame, [3,3], 10, lab))
