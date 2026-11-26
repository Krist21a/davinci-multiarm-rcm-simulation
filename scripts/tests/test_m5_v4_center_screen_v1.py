"""Direct row-10 identity, contact and depth refusal tests without archive labels."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/vision'))
from m5_v4_center_screen_v1 import measure, witnesses


def read(rgb,z,axis=(0.,0.,1.)):
    return measure(witnesses(rgb),z,focal=100.,far=.3,sigma=0.,axis=np.asarray(axis))


class CenterScreenTests(unittest.TestCase):
    def test_cap_bound_uses_actual_material_extent(self):
        a = np.zeros((20,20,3),np.uint8)
        z = np.full(a.shape[:2],.14)
        a[8:10,4:6] = [14,164,147]
        a[8:10,14:16] = [14,164,205]
        q = read(a,z)
        self.assertEqual(q['route'],'caps')
        self.assertLess(q['interval_m'][0],.1397)
        self.assertGreater(q['interval_m'][1],.1403)
        z[8,4] = np.nan
        self.assertIsNone(read(a,z)['interval_m'])

    def test_missing_node_not_filled_by_wrong_row_or_detached_ink(self):
        a = np.zeros((20,20,3),np.uint8)
        z = np.full(a.shape[:2],.14)
        a[8:10,4:6] = [14,164,147]
        a[8:10,14:16] = [14,150,205]
        self.assertIsNone(read(a,z)['interval_m'])
        a[8:10,14:16] = [14,164,205]
        a[1,1] = [14,164,205]
        self.assertIsNone(read(a,z)['interval_m'])

    def test_face_bound_requires_correct_two_faces_and_depth_compatible_contact(self):
        a = np.zeros((20,20,3),np.uint8)
        z = np.full(a.shape[:2],.14)
        a[4:8,4:12] = [248,150,205]
        a[8:12,4:12] = [248,164,147]
        self.assertEqual(read(a,z)['route'],'faces')
        z[8:12,4:12] += .005
        self.assertIsNone(read(a,z)['interval_m'])
        z[:] = .14
        a[8:12,4:12] = [248,164,205]
        self.assertIsNone(read(a,z)['interval_m'])
