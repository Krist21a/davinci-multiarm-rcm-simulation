"""Offline geometry-core tests for scripts/analysis/prostate_access_wrist.py (no simulator)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
sys.path.insert(0, str(ROOT / 'scripts' / 'build'))

import prostate_access_feasibility as pa  # noqa: E402
import prostate_access_wrist as pw  # noqa: E402


def sphere_occ(centre, radius):
    return pa.Occluder([trimesh.creation.icosphere(subdivisions=3, radius=radius)
                        .apply_translation(np.asarray(centre, float))])


def far_occ():
    """Occluder that can never block the test geometry."""
    return sphere_occ((0.0, 0.0, 1.0), 0.002)


def verbatim_straight(occ, ports, target):
    """The original analysis's reach test, verbatim."""
    ports = np.asarray(ports, float)
    L = np.linalg.norm(np.asarray(target) - ports, axis=1)
    cand = (L >= 0.06) & (L <= 0.22)
    out = np.zeros(len(ports), bool)
    if cand.any():
        out[cand] = occ.clear(ports[cand], np.repeat(np.asarray(target)[None], int(cand.sum()), axis=0),
                              0.006, skip_start=0.02, skip_end=0.012, step=0.001)
    return out


class ReachPolyTests(unittest.TestCase):
    def test_theta_zero_equals_verbatim_straight_model(self):
        # Polyline at theta=0, Ld=12 mm must equal the original straight-shaft reach elementwise,
        # including the tip-zone centre-ray block (last 12 mm is radius-excluded but ray-checked).
        ports = np.array([[0.0, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.5]])
        target = np.array([0.1, 0.0, 0.0])
        occ = sphere_occ((0.05, 0.0, 0.0), 0.006)          # blocks the x axis, not the offset port
        recs = pw.reach_poly(occ, ports, target, Ld=0.012, theta_max=0.0)
        straight = verbatim_straight(occ, ports, target)
        for i in range(len(ports)):
            self.assertEqual(recs[i] is not None, bool(straight[i]), f'port {i}')
        self.assertFalse(straight[0])                       # axis blocked
        self.assertTrue(straight[1])                        # offset port clear
        self.assertIsNone(recs[2])                          # out of range -> never sampled

    def test_tip_zone_ray_blocks_theta_zero(self):
        # A sliver in the last 12 mm before the target is outside the radius-sampled arc but inside
        # the centre-ray span: the original blocks it and so must the polyline at theta = 0.
        ports = np.array([[0.0, 0.0, 0.0], [0.0, 0.05, 0.0]])
        target = np.array([0.1, 0.0, 0.0])
        occ = sphere_occ((0.094, 0.0, 0.0), 0.003)         # inside the tip exclusion zone
        recs = pw.reach_poly(occ, ports, target, Ld=0.012, theta_max=0.0)
        straight = verbatim_straight(occ, ports, target)
        for i in range(len(ports)):
            self.assertEqual(recs[i] is not None, bool(straight[i]), f'port {i}')
        self.assertFalse(straight[0])                       # ray hits the sliver -> blocked

    def test_blocked_straight_cleared_by_bend(self):
        # Sphere 8 mm centred on the port->target line: the straight shaft is blocked, every
        # nominal bend <= 28.6 deg keeps the shaft within ~10.7 mm of the centre (< 14 mm needed),
        # while a 53 deg nominal bend passes ~14.4 mm off the centre and clears on both segments.
        port = np.zeros((1, 3))
        target = np.array([0.08, 0.0, 0.0])
        occ = sphere_occ((0.04, 0.0, 0.0), 0.008)
        straight = pw.reach_poly(occ, port, target, Ld=0.03, theta_max=0.0)
        self.assertIsNone(straight[0])
        too_stiff = pw.reach_poly(occ, port, target, Ld=0.03, theta_max=0.5)
        self.assertIsNone(too_stiff[0])                     # no sampled direction dodges the sphere
        bent = pw.reach_poly(occ, port, target, Ld=0.03, theta_max=1.4)
        self.assertIsNotNone(bent[0])
        self.assertGreater(bent[0]['bend_rad'], 0.0)
        self.assertLessEqual(bent[0]['bend_rad'], 1.4 + 1e-9)

    def test_range_applies_to_polyline_arc(self):
        # 55 mm port-target distance: unreachable straight (arc = 55 mm < 60 mm) but reachable with
        # a large bend because the arc port->W->t length lands inside [60, 220] mm.
        port = np.zeros((1, 3))
        target = np.array([0.055, 0.0, 0.0])
        occ = far_occ()
        self.assertIsNone(pw.reach_poly(occ, port, target, Ld=0.022, theta_max=0.0)[0])
        rec = pw.reach_poly(occ, port, target, Ld=0.022, theta_max=1.4)[0]
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec['arc_m'], rec['port_w_m'] + 0.022, places=9)
        self.assertTrue(0.06 - 1e-9 <= rec['arc_m'] <= 0.22 + 1e-9)
        self.assertGreater(rec['bend_rad'], 0.0)

    def test_bend_never_exceeds_limit(self):
        # Every accepted record must satisfy the actual bend <= theta_max (sampled theta is nominal
        # about the port->target line; the verified bend is between the real shaft direction
        # port->W and the distal direction).
        port = np.zeros((1, 3))
        target = np.array([0.1, 0.0, 0.0])
        occ = far_occ()
        for th in (0.0, 0.5, 1.0, 1.4):
            for rec in pw.reach_poly(occ, port, target, Ld=0.022, theta_max=th):
                if rec is not None:
                    self.assertLessEqual(rec['bend_rad'], th + 1e-9)


if __name__ == '__main__':
    unittest.main()
