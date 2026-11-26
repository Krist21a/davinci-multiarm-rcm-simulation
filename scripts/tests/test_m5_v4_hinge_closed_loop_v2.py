"""Analytic controls for the two opt-in hinge-only estimation changes."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/vision'))
from m5_v4_hinge_marker_v2 import residual, sphere
from m5_v4_hinge_work_v2 import fit, plane_intercept


class HingeV2Tests(unittest.TestCase):
    def test_grazing_miss_uses_radial_distance_and_finite_hit_floor(self):
        rays = np.array([[0.,0.,1.]])
        r,z = .0012,.145
        for extra in (0.,1e-10,.0001):
            hit,miss = residual(np.array([r+extra,0,z]),rays,np.array([z]),r)
            self.assertAlmostEqual(hit,-1e-7,places=13)
            self.assertAlmostEqual(miss,extra,places=13)

    def test_exact_front_sphere_recovers_pose_without_back_surface(self):
        ctr = np.array([.0003,.0004,.14])
        rays = np.array([[x,y,1.] for x in np.linspace(-.003,.005,9) for y in np.linspace(-.002,.006,9)])
        rays /= np.linalg.norm(rays,axis=1)[:,None]
        foot = rays@ctr
        disc = .0012**2-((foot[:,None]*rays-ctr)**2).sum(1)
        pts = rays*(foot-np.sqrt(disc))[:,None]
        result,info = sphere(pts,.0012,ctr+[.0001,-.0001,.0001])
        np.testing.assert_allclose(result,ctr,rtol=0,atol=1e-8)
        self.assertLessEqual(info['nfev'],60)
        self.assertIsNone(sphere(pts[:3],.0012,ctr)[0])

    def test_planar_intercept_exact_on_asymmetric_patch_and_rank_veto(self):
        coords = np.array([(y,x) for y in range(-2,3) for x in (-2,0,1,2) if (x,y)!=(2,2)])
        z = 1./(7.1+coords@[.002,-.003])
        self.assertAlmostEqual(plane_intercept(coords,z,0.),1./7.1,places=13)
        self.assertIsNone(plane_intercept(np.column_stack((coords[:,1],coords[:,1])),z,0.))

    def test_corner_fit_cannot_borrow_unconnected_third_pixel(self):
        z = np.full((25,25),.14)
        label,valid,pink = np.zeros_like(z,int),np.ones_like(z,bool),np.ones_like(z,bool)
        ctx = label,z,valid,pink,np.array([1.,0.]),.001,0.,900.
        a,b,w = np.array([12,12]),np.array([12,13]),np.array([13,12])
        pairs = [(a+i,b+i) for i in ([-2,0],[-1,0],[0,0],[1,0])]
        self.assertIsNotNone(fit(ctx,a,b,(a+w)*.5,0,w,pairs))
        valid[tuple(w)] = False
        self.assertIsNone(fit(ctx,a,b,(a+w)*.5,0,w,pairs))
