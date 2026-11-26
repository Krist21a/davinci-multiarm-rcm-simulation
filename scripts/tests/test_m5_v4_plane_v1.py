"""Check the plane audit using known shifts without opening any recorded exposures."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision','analysis')]
from score_m5_v4_plane_v1 import bracket, depth, fit_depth


class PlaneTests(unittest.TestCase):
    def test_opposite_slopes_separate_range_bias_from_ray_shift(self):
        rc = np.array([[400+i,610+j] for i in range(0,37,6) for j in range(0,32,7)])
        data = {}
        target = [.43,-.28,.0012,.023]
        for i,n in enumerate([[0.,0.,1.],[.8,0.,.6],[-.8,0.,.6],[0.,.8,.6],[0.,-.8,.6],[.4,-.4,.8]]):
            d = (100.+20*i)*n[2]
            data[i] = dict(rc=rc,n=n,d=d,z=depth(rc,n,d,target))
        result = fit_depth(data,'joint')
        self.assertTrue(result['passes'])
        np.testing.assert_allclose(result['params'],target,atol=1e-6,rtol=0)
        self.assertFalse(fit_depth(data,'native')['passes'])

    def test_rgb_signed_edge_brackets_do_not_use_z(self):
        edges = (np.array([[1.,0.],[0.,-1.]]),np.array([.2,-.6]),np.array([.4,-.3]))
        np.testing.assert_allclose(bracket(dict(edges=edges),[.25,.35]),[0.,0.])
        np.testing.assert_allclose(bracket(dict(edges=edges),[0.,0.]),[.2,.3])
