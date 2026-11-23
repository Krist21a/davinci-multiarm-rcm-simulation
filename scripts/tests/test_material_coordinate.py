"""Synthetic bank evidence for the coded PBD sheet chart."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vision'))
import material_coordinate as mc  # noqa: E402


class MaterialCoordinateTests(unittest.TestCase):
    def scan(self, cover=None, skew=0.0, crop=False, allow_partial=False):
        s, w = np.meshgrid(np.arange(-10, 110) * 0.00013, np.arange(-40, 41) * 0.00021)
        boundaries = np.array([0., 0.0016, 0.0041, 0.0078, 0.0104])
        q = s - skew * (w > 0)
        rgb = np.zeros((*s.shape, 3))
        depth = np.full(s.shape, 0.22)
        mask = (q >= boundaries[0]) & (q < boundaries[-1])
        for a, b, color in zip(boundaries, boundaries[1:], mc.COLORS):
            sel = (q >= a) & (q < b)
            rgb[sel] = np.asarray(color) * 255
            depth[sel] = 0.10
        if cover is not None:
            sel = cover(q, w)
            rgb[sel] = 80
            depth[sel] = 0.093
            mask[sel] = False
        sl = np.s_[:, 12:] if crop else np.s_[:, :]
        return mc.measure(rgb[sl], depth[sl], mask[sl], s[sl], w[sl], 0.0,
                          allow_partial=allow_partial)

    def test_local_nonuniform_stretch_is_preserved(self):
        chart = self.scan()
        self.assertTrue(chart['valid'], chart)
        np.testing.assert_allclose(chart['projected_knots_m'], [0, .0016, .0041, .0078, .0104], atol=.00013)
        self.assertAlmostEqual(np.interp(.0078, chart['projected_knots_m'], chart['material_knots_m']), .009, delta=.00015)

    def test_either_end_hidden_on_one_bank_fails(self):
        for cut in (lambda s: s < 0.0009, lambda s: s > 0.0096):
            chart = self.scan(cover=lambda s, w: cut(s) & (w < 0))
            self.assertFalse(chart['valid'], chart)
        self.assertFalse(self.scan(crop=True)['valid'])

    def test_missing_internal_transition_or_incompatible_banks_fails(self):
        self.assertFalse(self.scan(cover=lambda s, w: np.abs(s - .0041) < .0004)['valid'])
        self.assertFalse(self.scan(skew=.0018)['valid'])

    def test_sweep_uses_the_earlier_bank(self):
        chart = self.scan(skew=.0004)
        self.assertTrue(chart['valid'], chart)
        self.assertTrue(np.all(np.asarray(chart['guard_knots_m']) < chart['projected_knots_m']))

    def test_partial_frame_only_maps_a_supported_local_cell(self):
        chart = self.scan(cover=lambda s, w: (np.abs(s - .0078) < .0004) & (w < 0),
                          allow_partial=True)
        self.assertTrue(chart['valid'], chart)
        self.assertFalse(chart['complete'])
        m, x, guard = (chart[k] for k in ('material_knots_m', 'projected_knots_m', 'guard_knots_m'))
        np.testing.assert_array_equal(m, mc.KNOTS_M[[0, 1, 2, 4]])
        result, cell = mc.locate(m, x, (x[1] + x[2]) / 2)
        self.assertAlmostEqual(result, .0045)
        self.assertEqual(cell, m[1:3])
        self.assertIsNone(mc.locate(m, x, .007))
        self.assertEqual(mc.locate(m, x, x[2]), (m[2], [m[2], m[2]]))
        self.assertIsNone(mc.project(m, guard, .010))
        self.assertEqual(mc.project(m, guard, .010, stop_at_supported_row=True), guard[2])
        self.assertIsNone(mc.project(m, guard, .014, stop_at_supported_row=True))

    def test_partial_never_accepts_missing_end_or_contradictory_banks(self):
        self.assertFalse(self.scan(cover=lambda s, w: (s > .0096) & (w < 0),
                                   allow_partial=True)['valid'])
        self.assertFalse(self.scan(crop=True, allow_partial=True)['valid'])
        self.assertFalse(self.scan(skew=.0018, allow_partial=True)['valid'])
