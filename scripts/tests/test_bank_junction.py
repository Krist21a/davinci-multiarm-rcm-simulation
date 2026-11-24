"""Correspondence regressions for bank_junction.boundary (no simulator).

The fixture is a synthetic two-ordinal junction (inks 6 and 7 meeting at row 21,
both banks ending at column 24) so the tests pin the correspondence bug, not any
scored real capture.

Run: python -m unittest discover -s scripts/tests -p test_bank_junction.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vision'))

from bank_junction import boundary  # noqa: E402


def fixture():
    lab = np.where(np.indices((42, 48))[0] < 21, 6, 7).astype(int)
    lab[:, 24:] = -1
    z = np.where(lab < 0, .15, .1)
    pp = np.array([24., 21.])
    toward = np.array([1., 0.])
    return lab, z, pp, toward


class BankJunctionTests(unittest.TestCase):
    def detect(self, lab, z, pp, toward):
        return boundary(lab, z, 7, 0, toward, pp, 1000., .3)

    def test_clean_interior_junction_ends_at_band_exit(self):
        lab, z, pp, toward = fixture()
        r = self.detect(lab, z, pp, toward)
        self.assertIsNotNone(r)
        self.assertTrue(np.allclose(r[0], [24, 21], atol=.25))
        self.assertTrue(np.allclose(r[1], [0, 0, .1], atol=1e-4))

    def test_adjacent_occluder_does_not_terminate_inboard(self):
        lab, z, pp, toward = fixture()
        z[:, 24:27] = .07                      # 3-column occluder at the band ends
        self.assertIsNone(self.detect(lab, z, pp, toward))

    def test_equal_depth_needs_the_correct_opposite_ordinal(self):
        lab, z, pp, toward = fixture()
        z[:, 24:] = .1                         # equal material depth beyond the ends
        lab[:, 24:] = 13+4                     # wrong opposite-bank ordinal
        self.assertIsNone(self.detect(lab, z, pp, toward))
        lab[:, 24:] = 13+7                     # correct opposite-bank ordinal
        self.assertIsNotNone(self.detect(lab, z, pp, toward))


if __name__ == '__main__':
    unittest.main()
