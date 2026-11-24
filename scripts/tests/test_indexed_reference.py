"""Indexed-print observer tests: palette round-trip, world transport, sensor refusal.

Run: python -m unittest discover -s scripts/tests -p test_indexed_reference.py -v
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vision'))

from indexed_cut_perception import IndexedCutPerception  # noqa: E402
from indexed_reference import decode, measure  # noqa: E402


def ink(j, k):
    """The immutable Lua address formula, computed independently here."""
    return (42+13*j, 50+21*k, 42+25*((3*j+5*k) % 8))


class PaletteTests(unittest.TestCase):
    def test_all_104_codes_round_trip_and_checksum_corruption_is_unknown(self):
        im = np.zeros((13*2, 8*2, 3), np.uint8)
        for j in range(13):
            for k in range(8):
                im[2*j:2*j+2, 2*k:2*k+2] = ink(j, k)
        lab = decode(im)
        for j in range(13):
            for k in range(8):
                self.assertEqual(lab[2*j, 2*k], 8*j+k, f'code ({j},{k}) did not round-trip')
        self.assertEqual(int((lab >= 0).sum()), 104*4)
        # +7 on the checksum channel invalidates every one of the 104 codes.
        corrupt = im.copy()
        corrupt[..., 2] = (corrupt[..., 2].astype(int)+7).astype(np.uint8)
        self.assertEqual(int((decode(corrupt) >= 0).sum()), 0)
        z = np.full(im.shape[:2], .1)
        self.assertEqual(measure(corrupt, z, im.shape[1], im.shape[0], 1000., .3)['decision'], 'UNKNOWN')


class IndexedObserverTests(unittest.TestCase):
    def test_stations_and_tool_transport_to_world_without_certifying_a_cut(self):
        est = IndexedCutPerception(64, 48, 100., .3, [0, 0, 0])
        reference = dict(stations=[dict(row=7, column=9, camera_m=[.001, 0, .1])], local_mm=8)
        tool = ({'valid': True, 'tip': [0, 0, .1], 'jaw_axis': [1, 0, 0]}, None)
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)   # +90 deg about z
        t = np.array([1., 2., 3.])
        with patch('indexed_cut_perception.measure', return_value=reference), \
                patch.object(est, '_scissors', return_value=tool):
            out = est.estimate(np.zeros((48, 64, 3), np.uint8), np.full((48, 64), .1), R, t)
        station = out['material_reference']['stations'][0]
        self.assertTrue(np.allclose(station['world_m'], [1, 2.001, 3.1]))
        self.assertEqual(station['camera_m'], [.001, 0, .1])
        self.assertTrue(np.allclose(out['scissors']['tip'], [1, 2, 3.1]))
        self.assertTrue(np.allclose(out['scissors']['jaw_axis'], [0, 1, 0]))
        # A local material reference is never a certified control cut or progress.
        self.assertEqual(out['material_reference']['local_mm'], 8)
        self.assertFalse(out['cut']['valid'])
        self.assertIsNone(out['cut']['front'])
        self.assertIsNone(out['cut']['progress_m'])
        self.assertFalse(out['tension_proxy']['valid'])

    def test_wrong_shape_sensor_refuses_front_without_casting_or_caching(self):
        est = IndexedCutPerception(64, 48, 100., .3, [0, 0, 0])
        eye, zero = np.eye(3), np.zeros(3)
        for im, z in ((np.zeros((48, 64, 3), np.uint8), np.full((46, 64), .1)),   # depth shape
                      (np.zeros((48, 64, 3)), np.full((48, 64), .1))):            # float RGB: no cast
            with self.subTest(shapes=(im.shape, z.shape, im.dtype.str)):
                with patch.object(est, '_scissors') as scissors:
                    out = est.estimate(im, z, eye, zero)
                scissors.assert_not_called()
                self.assertIsNone(out['cut']['front'])
                self.assertFalse(out['cut']['valid'])
                self.assertIsNone(out['cut']['progress_m'])
                self.assertEqual(out['material_reference']['stations'], [])
                self.assertEqual(out['material_reference']['decision'], 'UNKNOWN')
                self.assertFalse(out['material_reference']['motion_authorized'])
                self.assertFalse(out['scissors']['valid'])
                self.assertFalse(out['tissue']['valid'])


if __name__ == '__main__':
    unittest.main()
