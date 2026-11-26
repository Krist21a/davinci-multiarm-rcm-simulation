"""Negative controls for v3: no replacement, no late rescue, same ray objective."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/vision'))
import m5_v4_hinge_work_v3 as work
from cut_perception import _ray_sphere_centre
from m5_v4_hinge_marker_v3 import sphere


class HingeV3Tests(unittest.TestCase):
    def test_accepted_chart_is_exact_same_object(self):
        witness = object()
        with patch.object(work.primary,'chart',return_value=witness), patch.object(work.planar,'chart') as fallback:
            self.assertIs(work.chart(None,None,1.,1.,0.,0),witness)
            fallback.assert_not_called()

    def test_no_late_fallback_even_if_planar_would_accept(self):
        witness = object()
        with patch.object(work.primary,'chart',return_value=None), patch.object(work.planar,'chart',return_value=witness) as f:
            self.assertIsNone(work.chart(None,None,1.,1.,0.,6))
            f.assert_not_called()
            args = (object(),object(),1.,.3,.0005,5,9)
            self.assertIs(work.chart(*args),witness)
            f.assert_called_once_with(*args)

    def test_original_converged_sphere_is_bit_identical(self):
        c = np.array([.0003,.0004,.14])
        rays = np.array([[x,y,1.] for x in np.linspace(-.003,.005,9) for y in np.linspace(-.002,.006,9)])
        rays /= np.linalg.norm(rays,axis=1)[:,None]
        foot = rays@c
        disc = .0012**2-((foot[:,None]*rays-c)**2).sum(1)
        pts = rays*(foot-np.sqrt(disc))[:,None]
        start = c+[.0001,-.0001,.0001]
        old = _ray_sphere_centre(pts,.0012,start)
        result,info = sphere(pts,.0012,start)
        self.assertIsNotNone(old)
        self.assertLessEqual(info['nfev'],60)
        np.testing.assert_array_equal(result,old)
