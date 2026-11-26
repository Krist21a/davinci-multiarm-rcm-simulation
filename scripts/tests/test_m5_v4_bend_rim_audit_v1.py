"""Analytic checks on the offline sheet-only intersection and station replay."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
from audit_m5_v4_bend_rim_v1 import Sheet, projected, rays, tied_pairs


class RimAuditTests(unittest.TestCase):
    def test_frontmost_live_face_beats_hidden_face_and_dead_face_does_not(self):
        j,i = np.indices((14,24))
        nodes = np.stack((10+i,10+j,np.ones_like(i)),axis=-1).astype(float)
        edges = {}
        for r in range(13):
            for c in range(23):
                a = 24*r+c
                for u,v in ((a,a+1),(a,a+24),(a,a+25),(a+1,a+25),(a+24,a+25)):
                    edges[u,v] = 0
        for c,depth in ((0,1.),(3,2.)):
            idx = (c,c+1,c+25)
            nodes.reshape(-1,3)[list(idx)] = depth*np.array([[0,0,1],[1,0,1],[1,1,1]])
            for k,v in enumerate(idx):
                edges[tuple(sorted((v,idx[k-1])))] = 1

        def hit():
            return Sheet(nodes,[(u,v,a) for (u,v),a in edges.items()]).hits([[.5,.25,1],[2,2,1]])

        for c,d in ((0,1.),(3,2.)):
            z,ids = hit()
            self.assertEqual(z[0],d)
            self.assertEqual(ids[0].tolist(),[0,c,0])
            self.assertTrue(np.isinf(z[1]))
            self.assertEqual(ids[1].tolist(),[-1,-1,-1])
            edges[c,c+1] = 0

    def test_subpixel_convention_and_all_depth_compatible_ties(self):
        uv = np.array([[24.5,783.],[711.,87.25]])
        p = rays(uv,(960,1280),850.)*np.array([[.2],[.13]])
        np.testing.assert_allclose(projected(p,(960,1280),850.),uv,rtol=0,atol=1e-12)
        a = np.array([[0.,0.,.1],[0.,1.,.2],[2.,2.,.101]])
        b = np.array([[1.,0.,.1],[2.,1.,.101],[0.,1.,.1]])
        self.assertEqual(tied_pairs(a,b,.001).tolist(),[[0,0],[0,2],[2,1]])
