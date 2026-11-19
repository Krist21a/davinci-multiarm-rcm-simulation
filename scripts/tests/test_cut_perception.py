"""Offline tests for scripts/vision/cut_perception.py on synthetic pinhole renders.

The renderer draws the tissue rectangle, an optional slit and the two marker spheres with
the same camera model the estimator assumes (x = -(u - W/2) z/f, forward z). Camera = world
(R = I, t = 0), so estimated quantities compare directly to the constructed truth.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vision'))

import cut_perception as cp  # noqa: E402

W, H, F, FAR = 200, 120, 150.0, 0.30

# Scene layout (camera frame = world frame for the tests).
P0 = np.array([0.005, -0.002, 0.100])          # near-edge midpoint
NORMAL = np.array([0.10, -0.05, -1.0])
NORMAL = NORMAL / np.linalg.norm(NORMAL)
E_V = np.array([1.0, 0.30, 0.0])
E_V = E_V - (E_V @ NORMAL) * NORMAL
E_V = E_V / np.linalg.norm(E_V)                # cut direction, in-plane
E_U = np.cross(NORMAL, E_V)                    # across the cut, in-plane
L_V, L_U = 0.030, 0.023                        # sheet extents along / across the cut

# The shaft approaches along the sheet (in-plane component ~e_v) so the approach-matched
# principal axis is the cut direction. Markers sit on the housing near the wrist, in view.
AXIS = np.array([0.40, 0.15, 0.90])
AXIS = AXIS / np.linalg.norm(AXIS)
TCP = np.array([0.002, 0.000, 0.095])
LAT = np.array([0.2, 1.0, 0.0])
LAT = LAT - (LAT @ AXIS) * AXIS
LAT = LAT / np.linalg.norm(LAT)
YAW = TCP - 0.025 * AXIS
RCM = TCP - 0.080 * AXIS
TCP_LEN = 0.025
# Yaw-frame offsets in the estimator's convention: (x_lat, y_shaft_toward_tcp, z_lat).
OFF_D = np.array([0.0, -0.018, 0.005])
OFF_P = np.array([0.0, -0.033, 0.005])
M_D = YAW + OFF_D[1] * AXIS + OFF_D[2] * LAT
M_P = YAW + OFF_P[1] * AXIS + OFF_P[2] * LAT
MR = cp.MARKER_RADIUS

SLIT_U0, SLIT_W = 0.0015, 0.0018               # cut line lateral offset, slit width
FRONT_SV = 0.010
FRONT_TRUE = P0 + FRONT_SV * E_V + SLIT_U0 * E_U


def render(slit_sv=None, markers=('d', 'p'), nan_frac=0.0, tissue=True):
    """Pinhole render -> (rgb float HxWx3, depth HxW). Depth is z along the unnormalised ray."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([-(u - W / 2) / F, -(v - H / 2) / F, np.ones((H, W))], axis=-1)
    rgb = np.full((H, W, 3), 40.0)
    depth = np.full((H, W), FAR * 0.999)
    if tissue:
        denom = d @ NORMAL
        s = (P0 @ NORMAL) / denom
        hit = s > 0
        q = d * s[..., None] - P0
        sv, su = q @ E_V, q @ E_U
        inside = hit & (sv >= 0) & (sv <= L_V) & (np.abs(su) <= L_U / 2)
        if slit_sv is not None:
            inside &= ~((sv <= slit_sv) & (np.abs(su - SLIT_U0) <= SLIT_W / 2))
        sel = inside & (s < depth)
        depth[sel] = s[sel]
        rgb[sel] = np.array(cp.TISSUE_RGB) * 255
    dn = d / np.linalg.norm(d, axis=-1, keepdims=True)
    for centre, color, tag in ((M_D, cp.MARKER_DISTAL_RGB, 'd'), (M_P, cp.MARKER_PROX_RGB, 'p')):
        if tag not in markers:
            continue
        oc = centre
        b = dn @ oc
        disc = b ** 2 - (oc @ oc - MR ** 2)
        hitm = (disc > 0) & (b > 0)
        t = np.where(hitm, b - np.sqrt(np.maximum(disc, 0)), np.inf)
        sel = hitm & (t * dn[..., 2] < depth)
        depth[sel] = t[sel] * dn[sel, 2]
        rgb[sel] = np.array(color) * 255
    if nan_frac:
        rng = np.random.default_rng(3)
        depth[rng.random(depth.shape) < nan_frac] = np.nan
    return rgb, depth


def estimator():
    return cp.CutPerception(W, H, F, FAR, RCM, tcp_len=TCP_LEN,
                            marker_offsets=(OFF_D, OFF_P))


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.asarray(a) @ np.asarray(b) /
                      (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


class GeometryTests(unittest.TestCase):
    def test_pre_cut_start_direction_and_tip(self):
        rgb, depth = render()
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3), post_grasp=True)
        cut, scis = est['cut'], est['scissors']
        self.assertTrue(est['tissue']['valid'], est['tissue'])
        self.assertTrue(cut['valid'], cut)
        self.assertEqual(cut['mode'], 'pre_cut')
        self.assertLess(np.linalg.norm(np.array(cut['start']) - P0), 1.5e-3)
        self.assertLess(angle(cut['direction'], E_V), 3.0)
        self.assertLess(np.linalg.norm(np.array(cut['front']) - P0), 1.5e-3)   # front = start pre-cut
        self.assertTrue(scis['valid'], scis)
        self.assertLess(np.linalg.norm(np.array(scis['tip']) - TCP), 0.6e-3)
        self.assertLess(angle(scis['axis'], AXIS), 2.0)

    def test_slit_front_and_direction(self):
        rgb, depth = render(slit_sv=FRONT_SV)
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        cut = est['cut']
        self.assertTrue(cut['valid'], cut)
        self.assertEqual(cut['mode'], 'slit')
        self.assertLess(np.linalg.norm(np.array(cut['front']) - FRONT_TRUE), 1.5e-3)
        self.assertLess(angle(cut['direction'], E_V), 3.0)

    def test_depth_noise_increases_tip_error(self):
        errs = []
        for sigma in (0.0, 0.001, 0.003):
            run = []
            for seed in range(5):
                rgb, depth = render()
                if sigma:
                    depth = depth + np.random.default_rng(seed).normal(0, sigma, depth.shape)
                est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
                if est['scissors']['valid']:
                    run.append(np.linalg.norm(np.array(est['scissors']['tip']) - TCP))
            errs.append(np.mean(run))
        self.assertLessEqual(errs[0], errs[1])
        self.assertLessEqual(errs[1], errs[2])
        self.assertLess(errs[0], 0.6e-3)


class ValidityTests(unittest.TestCase):
    def test_missing_marker_is_invalid_not_exception(self):
        rgb, depth = render(markers=('p',))
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        self.assertFalse(est['scissors']['valid'])
        self.assertIn('distal', est['scissors']['reason'])
        self.assertFalse(est['cut']['valid'])           # no marker -> cannot pick the near side

    def test_no_tissue_is_invalid_not_exception(self):
        rgb, depth = render(tissue=False)
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        self.assertFalse(est['tissue']['valid'])
        self.assertFalse(est['cut']['valid'])

    def test_nan_depth_handled(self):
        rgb, depth = render(nan_frac=0.4)
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        self.assertIsInstance(est, dict)                # NaNs must not raise
        self.assertIn('valid', est['tissue'])

    def test_strain_proxy_after_grasp(self):
        rgb, depth = render()
        p = estimator()
        est = p.estimate(rgb, depth, np.eye(3), np.zeros(3), post_grasp=True)
        self.assertIsNotNone(est['tension_proxy']['strain_proxy'])
        self.assertAlmostEqual(est['tension_proxy']['strain_proxy'], 0.0, places=3)


if __name__ == '__main__':
    unittest.main()
