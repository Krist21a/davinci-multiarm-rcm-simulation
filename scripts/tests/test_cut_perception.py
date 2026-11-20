"""Offline tests for scripts/vision/cut_perception.py on synthetic pinhole renders.

The renderer draws the tissue strip (wider across the cut than along it, like the real
24 mm x 14 mm sheet), an optional slit and the three marker spheres placed from a known
yaw pose with the same camera model the estimator assumes (x = -(u - W/2) z/f, forward
z). Camera = world (R = I, t = 0), so estimated quantities compare to the constructed
truth directly. tcp_local / jaw_axis_local use the real articulated-scene calibration.
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
E_V = E_V / np.linalg.norm(E_V)                # cut direction, in-plane (strip length)
E_U = np.cross(NORMAL, E_V)                    # across the cut, in-plane (strip WIDTH)
L_V, L_U = 0.012, 0.024                        # along / across the cut (width is the long axis)

# Yaw pose: the shaft +y axis aims from the housing toward a point just past the near
# edge, so the near edge is the strip end closest to the tip — as in the real scene.
TCP_LOCAL = np.array([0.000042, 0.013052, 0.001585])   # measured: rigid in the yaw frame
JAW_LOCAL = cp.unit(np.array([0.05, -0.02, -1.0]))
YAW_Y = cp.unit(np.array([0.15, 0.85, -0.50]))         # shaft axis, yaw-frame +y
YAW_X = cp.unit(np.cross(YAW_Y, np.array([0.0, 0.0, 1.0])))
YAW_Z = np.cross(YAW_X, YAW_Y)
R_YAW = np.stack([YAW_X, YAW_Y, YAW_Z], axis=1)
TIP_TRUE = P0 + 0.001 * E_V + 0.003 * NORMAL           # tip ~3 mm above the near edge
T_YAW = TIP_TRUE - R_YAW @ TCP_LOCAL
JAW_TRUE = cp.unit(R_YAW @ JAW_LOCAL)
M_D = R_YAW @ np.asarray(cp.MARKER_DISTAL_OFF) + T_YAW
M_P = R_YAW @ np.asarray(cp.MARKER_PROX_OFF) + T_YAW
M_T = R_YAW @ np.asarray(cp.MARKER_THIRD_OFF) + T_YAW
MR = cp.MARKER_RADIUS
RCM = T_YAW - 0.080 * YAW_Y

SLIT_U0, SLIT_W = 0.0, 0.0025                  # slit on the cut line (the band the
                                               # estimator scans), wide enough to give
                                               # >=2 behind-plane px per 0.5 mm bin
FRONT_SV = 0.007
FRONT_TRUE = P0 + FRONT_SV * E_V + SLIT_U0 * E_U


def render(slit_sv=None, markers=('d', 'p', 't'), nan_frac=0.0, tissue=True,
           occlude_sv=None):
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
        if occlude_sv is not None:
            # Foreground occluder: covers the band ahead of sv in front of the plane.
            occ = hit & (sv > occlude_sv) & (sv < occlude_sv + 0.004) & (np.abs(su) < 0.004)
            depth[occ] = s[occ] - 0.010
            rgb[occ] = (120.0, 120.0, 120.0)
    dn = d / np.linalg.norm(d, axis=-1, keepdims=True)
    for centre, color, tag in ((M_D, cp.MARKER_DISTAL_RGB, 'd'),
                               (M_P, cp.MARKER_PROX_RGB, 'p'),
                               (M_T, cp.MARKER_THIRD_RGB, 't')):
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
    return cp.CutPerception(W, H, F, FAR, RCM,
                            marker_offsets=cp.MARKER_OFFS, marker_colours=cp.MARKER_COLOURS,
                            tcp_local=TCP_LOCAL, jaw_axis_local=JAW_LOCAL)


def meta_dict():
    return dict(estimator_args=dict(W=W, H=H, f=F, far=FAR, rcm_world=RCM.tolist(),
                                    marker_radius=MR, marker_offsets=[list(o) for o in cp.MARKER_OFFS],
                                    marker_colours=[list(c) for c in cp.MARKER_COLOURS],
                                    tcp_local=TCP_LOCAL.tolist(), jaw_axis_local=JAW_LOCAL.tolist()))


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.asarray(a) @ np.asarray(b) /
                      (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))


class GeometryTests(unittest.TestCase):
    def test_kabsch_recovers_tip_and_jaw_with_lateral_offset(self):
        """3 non-collinear markers: full yaw pose, TCP off-axis to <1e-6 m."""
        rgb, depth = render()
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        scis = est['scissors']
        self.assertTrue(scis['valid'], scis)
        self.assertLess(scis['fit_rms_m'], cp.FIT_RMS_MAX_M)
        self.assertLess(np.linalg.norm(np.array(scis['tip']) - TIP_TRUE), 1e-6)
        self.assertLess(angle(scis['jaw_axis'], JAW_TRUE), 0.01)
        self.assertLess(angle(scis['axis'], YAW_Y), 0.01)

    def test_from_meta_matches_constructor(self):
        rgb, depth = render()
        a = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        b = cp.CutPerception.from_meta(meta_dict()).estimate(rgb, depth, np.eye(3), np.zeros(3))
        self.assertEqual(a['scissors']['valid'], b['scissors']['valid'])
        np.testing.assert_allclose(a['scissors']['tip'], b['scissors']['tip'], atol=1e-12)
        self.assertEqual(a['cut']['front_valid'], b['cut']['front_valid'])

    def test_pre_cut_start_direction_and_front(self):
        rgb, depth = render()
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3), post_grasp=True)
        cut = est['cut']
        self.assertTrue(est['tissue']['valid'], est['tissue'])
        self.assertTrue(cut['valid'], cut)
        self.assertTrue(cut['front_valid'], cut)
        self.assertEqual(cut['mode'], 'pre_cut')
        self.assertLess(np.linalg.norm(np.array(cut['start']) - P0), 1.5e-3)
        self.assertLess(angle(cut['direction'], E_V), 3.0)
        self.assertLess(np.linalg.norm(np.array(cut['front']) - P0), 1.5e-3)
        self.assertIn('front_uv', cut)
        self.assertIn('cut_site_offset', cut)

    def test_slit_front_is_the_apex(self):
        rgb, depth = render(slit_sv=FRONT_SV)
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        cut = est['cut']
        self.assertTrue(cut['valid'], cut)
        self.assertTrue(cut['front_valid'], cut)
        self.assertEqual(cut['mode'], 'slit')
        self.assertLess(np.linalg.norm(np.array(cut['front']) - FRONT_TRUE), 1.5e-3)
        self.assertLess(angle(cut['direction'], E_V), 3.0)

    def test_occluded_apex_fails_closed(self):
        """Slit continuing under a foreground cover beyond the apex -> invalid."""
        rgb, depth = render(slit_sv=FRONT_SV + 0.002, occlude_sv=FRONT_SV)
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        cut = est['cut']
        self.assertFalse(cut['front_valid'], cut)
        self.assertNotEqual(cut.get('reason'), '')

    def test_pre_cut_covered_lookahead_fails_closed(self):
        """Foreground over the near-edge look-ahead: the crack start is unprovable."""
        rgb, depth = render(occlude_sv=0.0)
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3), post_grasp=True)
        cut = est['cut']
        self.assertFalse(cut['front_valid'], cut)
        self.assertNotEqual(cut.get('reason'), '')

    def test_front_never_reads_the_tip(self):
        """Anti-circularity: after the near edge is chosen the front is tissue-only —
        the same slit frame with an absent / on-sheet / dwelling / distant scissors
        must give bit-identical front fields."""
        rgb0, depth0 = render()                      # establish the near edge
        rgb_s, depth_s = render(slit_sv=FRONT_SV)
        fields = ('front_valid', 'front', 'mode', 'slit_opening_m')

        def front_seq(scis_ret):
            est = estimator()
            est.estimate(rgb0, depth0, np.eye(3), np.zeros(3))   # pick the near edge
            assert est._cut_dir is not None
            est._scissors = lambda rgb, depth, _r=scis_ret: _r
            seq = []
            for _ in range(3):
                cut = est.estimate(rgb_s, depth_s, np.eye(3),
                                   np.zeros(3))['cut']
                seq.append(tuple(tuple(cut[k]) if isinstance(cut[k], list)
                                 else cut[k] for k in fields))
            return seq

        absent = (dict(valid=False, reason='patched'), None)

        def scis(tip):
            d = dict(valid=True, tip=np.asarray(tip).tolist(), tip_uv=(100.0, 60.0))
            return (d, d)

        base = front_seq(absent)
        for variant in (scis(FRONT_TRUE + 0.002 * E_V),      # on-sheet ahead of apex
                        scis(P0 + 0.004 * E_V),              # dwelling at one spot
                        scis(P0 + 0.05 * E_U + 0.05 * NORMAL)):   # far off-sheet
            self.assertEqual(base, front_seq(variant))

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
                    run.append(np.linalg.norm(np.array(est['scissors']['tip']) - TIP_TRUE))
            errs.append(np.mean(run) if run else np.inf)
        self.assertLessEqual(errs[0], errs[1])
        self.assertLessEqual(errs[1], errs[2])
        self.assertLess(errs[0], 1e-5)


class ValidityTests(unittest.TestCase):
    def test_fewer_than_three_markers_invalid(self):
        rgb, depth = render(markers=('d', 'p'))
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        self.assertFalse(est['scissors']['valid'])
        self.assertIn('third', est['scissors']['reason'])
        self.assertFalse(est['cut']['valid'])           # no tip -> cannot pick the near edge

    def test_no_sheet_is_invalid(self):
        rgb, depth = render(tissue=False)
        est = estimator().estimate(rgb, depth, np.eye(3), np.zeros(3))
        self.assertFalse(est['tissue']['valid'])
        self.assertFalse(est['cut']['valid'])
        self.assertFalse(est['cut']['front_valid'])

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


class DirectionRefitTests(unittest.TestCase):
    """_refit_direction: the anchored, low-passed slit refit (m5a_amendment_v4)."""
    ANCHOR = np.array([1.0, 0.0, 0.0])

    def rotated(self, deg, axis=(0.0, 0.0, 1.0)):
        axis = axis / np.linalg.norm(axis)
        a = np.radians(deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K) @ self.ANCHOR

    def test_rejects_candidate_10_deg_off_anchor(self):
        self.assertIsNone(cp._refit_direction(self.ANCHOR, self.ANCHOR, self.rotated(10.0)))

    def test_accepts_3_deg_and_returns_the_blend(self):
        d = self.rotated(3.0)
        out = cp._refit_direction(self.ANCHOR, self.ANCHOR, d)
        np.testing.assert_allclose(out, cp.unit(cp.DIR_SMOOTH * self.ANCHOR + (1 - cp.DIR_SMOOTH) * d),
                                   atol=1e-12)

    def test_ratcheted_candidates_never_leave_the_anchor_cone(self):
        # Each candidate rotates 4 deg further from the PREVIOUS accepted direction; the
        # old prev-relative 20 deg clamp ratcheted ~0.95 rad, the anchor cone cannot.
        prev = self.ANCHOR.copy()
        ang = 0.0
        for k in range(1, 21):
            cand = self.rotated(min(4.0 * k, 90.0))
            out = cp._refit_direction(self.ANCHOR, prev, cand)
            if out is not None:
                prev = out
            ang = np.degrees(np.arccos(np.clip(prev @ self.ANCHOR, -1, 1)))
            self.assertLessEqual(ang, np.degrees(cp.DIR_UPDATE_RAD) + 1e-9)
        # By construction candidates past ~5 deg are all rejected, so the estimate stays
        # inside the cone; the old ratchet would have walked to ~80 deg.
        self.assertLessEqual(ang, np.degrees(cp.DIR_UPDATE_RAD) + 1e-9)


if __name__ == '__main__':
    unittest.main()
