"""Offline tests for scripts/vision/ruler_readout.py on a synthetic printed sheet.

The fixture rasterises the protocol inks directly: 13 material rows of six pixel rows each
starting at Y0, a left bank at columns 20:49 and a right bank at 49:79 on an otherwise black
image. Sheet ink sits at .105 m over a .15 m background; an actual opening is columns 47:52
returned to background depth for the cut rows only.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vision'))

import ruler_readout as rr  # noqa: E402
import connected_ruler  # noqa: E402

Y0, ROW_PX = 10, 6                  # first ink row and raster rows per material millimetre
LEFT, RIGHT = (20, 49), (49, 79)    # pixel column ranges of the two printed banks
SHEET_Z, BACK_Z = .105, .15


def ruler(length, gap_depth=BACK_Z):
    """13-row printed ruler; `length` material millimetres opened from row 0."""
    rgb = np.zeros((100, 100, 3), np.uint8)
    depth = np.full((100, 100), BACK_Z)
    for j in range(13):
        r, k = 32 + 48 * (j % 4), 35 + 46 * (j // 4)
        rows = slice(Y0 + ROW_PX * j, Y0 + ROW_PX * (j + 1))
        rgb[rows, LEFT[0]:LEFT[1]] = (r, 235, k)
        rgb[rows, RIGHT[0]:RIGHT[1]] = (r, k, 235)
        depth[rows, LEFT[0]:RIGHT[1]] = SHEET_Z
    if length:
        rgb[Y0:Y0 + ROW_PX * length, 47:52] = 0
        depth[Y0:Y0 + ROW_PX * length, 47:52] = gap_depth
    return rgb, depth


class ClassifyTests(unittest.TestCase):
    def test_connected_count_is_cells_not_open_boundaries(self):
        for length, expected in ((0, 'UNKNOWN'), (8, 'BELOW'), (9, 'STOP'),
                                 (10, 'STOP'), (11, 'UNKNOWN')):
            with self.subTest(length=length):
                r = connected_ruler.classify(*ruler(length))
                self.assertEqual(r['decision'], expected)
                if expected != 'UNKNOWN':
                    self.assertEqual(r['open_boundary'], length)

    def test_local_candidate_does_not_claim_an_occluded_prefix(self):
        im, z = ruler(9)
        im[:Y0+4*ROW_PX], z[:Y0+4*ROW_PX] = 90, .02
        self.assertEqual(connected_ruler.local_front(im, z)['prefix_mm'], 9)
        self.assertEqual(connected_ruler.classify(im, z)['decision'], 'UNKNOWN')

    def test_intact_bridge_or_foreground_is_not_a_local_endpoint(self):
        plain, sheet = ruler(0)
        for j, cover in ((4, False), (8, True)):
            im, z = ruler(9)
            row = slice(Y0+j*ROW_PX, Y0+(j+1)*ROW_PX)
            im[row], z[row] = (90, .02) if cover else (plain[row], sheet[row])
            self.assertIsNone(connected_ruler.local_front(im, z)['prefix_mm'])

    def test_opening_below_nine_is_below(self):
        self.assertEqual(rr.classify(*ruler(8))['decision'], 'BELOW')

    def test_opening_at_nine_and_ten_is_stop(self):
        for length in (9, 10):
            with self.subTest(length=length):
                self.assertEqual(rr.classify(*ruler(length))['decision'], 'STOP')

    def test_uncut_is_not_stop(self):
        self.assertNotEqual(rr.classify(*ruler(0))['decision'], 'STOP')

    def test_foreground_in_the_gap_is_unknown(self):
        # Something unprinted nearer than the sheet fills the crack: no opening evidence.
        self.assertEqual(rr.classify(*ruler(9, gap_depth=.095))['decision'], 'UNKNOWN')

    def test_covered_terminal_bank_strip_is_unknown(self):
        rgb, depth = ruler(9)
        rgb[Y0 + 72:Y0 + 78, RIGHT[0]:RIGHT[1]] = (219, 107, 122)   # unprinted occluder
        depth[Y0 + 72:Y0 + 78, RIGHT[0]:RIGHT[1]] = .08
        self.assertEqual(rr.classify(rgb, depth)['decision'], 'UNKNOWN')

    def test_cropped_far_end_is_unknown(self):
        rgb, depth = ruler(9)
        self.assertEqual(rr.classify(rgb[:85], depth[:85])['decision'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
