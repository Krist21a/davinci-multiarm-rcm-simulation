"""Focused regressions for the v2 coded-cell reader on rendered RGB-D inputs.

Every case feeds ONLY rendered RGB and depth plus camera intrinsics to classify;
the oracle uv is consumed by assertions, never by the reader. No simulator.

Run: python -m unittest discover -s scripts/tests -p test_coded_cell_v2.py -v
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
with patch.object(sys, 'path', [str(ROOT / 'scripts/vision'), str(ROOT / 'scripts/analysis'),
                                *sys.path]):
    from cell_certificate import classify
    from cell_v2_controls import contains, render


class CodedCellV2Tests(unittest.TestCase):
    def test_plain_nine_mm_stops_with_witnessed_support(self):
        im, z, f, pp, uv = render(9)
        r = classify(im, z, focal=f, principal=pp)
        self.assertEqual((r['local_mm'], r['prefix'], r['decision']), (9, 9, 'STOP'))
        self.assertTrue(any(q['box_pixels'] == 0 and q['blocks'] > 0
                            for row in r['faces'][:9] for q in row))
        self.assertTrue(all(contains(c['polygon'], uv[i])
                            for i, c in enumerate(r['corners']) if c))

    def test_hidden_entry_refuses_prefix_but_far_cap_still_stops(self):
        im, z, f, pp, uv = render(9, 'hide_entry')
        r = classify(im, z, focal=f, principal=pp)
        self.assertEqual((r['local_mm'], r['prefix'], r['decision']), (9, None, 'UNKNOWN'))
        im, z, f, pp, uv = render(9, 'far_cap')
        r = classify(im, z, focal=f, principal=pp)
        self.assertEqual((r['local_mm'], r['prefix'], r['decision']), (9, 9, 'STOP'))

    def test_speckles_cover_and_visible_bridge(self):
        for kind, want in (('deep_speck', (9, 9, 'STOP')),
                           ('near_speck', (None, None, 'UNKNOWN')),
                           ('target_cover', (None, None, 'UNKNOWN')),
                           ('bridge', (None, None, 'UNKNOWN'))):
            with self.subTest(kind=kind):
                im, z, f, pp, uv = render(9, kind)
                r = classify(im, z, focal=f, principal=pp)
                self.assertEqual((r['local_mm'], r['prefix'], r['decision']), want)

    def test_wrong_sized_sensor_input_is_not_a_prefix(self):
        im, z = np.zeros((5, 6, 3), np.uint8), np.zeros((6, 5))
        r = classify(im, z, focal=100., principal=np.zeros(2))
        self.assertEqual(r['decision'], 'UNKNOWN')
        self.assertIsNone(r['prefix'])


if __name__ == '__main__':
    unittest.main()
