"""Analytic tests independent of the blind images and archived tissue scores."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision','analysis')]
from score_m5_v4_plane_blind_v2 import COLORS, F, H, W, forward, old, parity_score
from transfer_m5_v4_plane_blind_v2 import optical_inverse


class BlindPlaneTests(unittest.TestCase):
    def test_diagonal_max_not_sum_and_normal_scaling(self):
        z = np.array([124.,142.])
        n = np.array([[.5,-.5,1.],[0.,0.,1.]])
        d = np.array([123.,139.])
        expected = 1./(1./z-np.array([.25/(F*123),0.]))
        np.testing.assert_allclose(forward(z,n,d),expected)
        np.testing.assert_allclose(forward(z,-3*n,-3*d),expected)

    def test_endpoint_color_identity_not_just_presence_of_crossing(self):
        verts = old.grid()+[.00057,.00031,.138]
        yy,xx = np.indices((H,W))
        i = np.floor(((W*.5-.5-xx)*.138/F-.00057)/.004+4).astype(int)
        j = np.floor(((H*.5-.5-yy)*.138/F-.00031)/.004+3).astype(int)
        board = (i>=0)&(i<8)&(j>=0)&(j<6)
        rgb = np.zeros((H,W,3),np.uint8)
        rgb[board] = COLORS[((i+j)%2)[board]]
        ok = parity_score(rgb,verts,np.zeros(2))
        self.assertTrue(ok['passed'])
        rgb[board] = COLORS[((i+j+1)%2)[board]]
        bad = parity_score(rgb,verts,np.zeros(2))
        self.assertFalse(bad['passed'])
        self.assertEqual(ok['observed']['n'],bad['observed']['n'])

    def test_optical_inverse_on_affine_reciprocal_depth_and_missing_support(self):
        yy,xx = np.indices((15,15))
        u = 7.2+(yy-7)*.0008+(xx-7)*.0006
        measured = 1./(u-.0004)
        p = np.array([0.,0.,measured[7,7]])
        ok = np.ones(u.shape,bool)
        np.testing.assert_allclose(optical_inverse(p,measured,ok,0.),[0.,0.,1/7.2],atol=1e-12)
        ok[:] = False
        ok[7,:] = True
        self.assertIsNone(optical_inverse(p,measured,ok,0.))
