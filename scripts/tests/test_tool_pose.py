"""Unit tests for scripts/task/tool_pose.py (DLS IK) against a synthetic analytic arm.

The synthetic FK mirrors the scene semantics: a fixed RCM point, shaft direction
yaw/pitch like control()'s d vector, insertion along it, then an intrinsic
roll -> pitch -> yaw wrist chain and a tip offset along the tool axis. Jaw pointing
axis = tcp z (column 2), jaw opening axis = tcp y (column 1) — matching solve()
defaults axis_col=2, yaw_col=1.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task'))

import tool_pose  # noqa: E402

RCM = np.array([0.0, 0.0, 0.10])
AXIS_HOME = np.array([0.0, 0.0, 1.0])
TX, TY = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
INS_HOME, DIST_HOME, TIP = 0.04, 0.10, 0.006
LIMITS = (np.array([-5.0, -5.0, 0.0, -1.0, -1.0, -1.0]),
          np.array([5.0, 5.0, 0.05, 1.0, 1.0, 1.0]))


def unit(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def rot(axis, angle):
    a = unit(axis)
    c, s = math.cos(angle), math.sin(angle)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + np.outer(a, a) * (1 - c) + K * s


def fk(q):
    yaw, pitch, ins, roll, wp, wy = np.asarray(q, dtype=float)
    d = unit(AXIS_HOME + math.tan(math.radians(yaw)) * TX + math.tan(math.radians(pitch)) * TY)
    p_shaft = RCM - d * (DIST_HOME - (ins - INS_HOME))
    # Minimal rotation carrying +z onto d (the controller's constrain_roll=False frame).
    k = np.cross([0, 0, 1], d)
    s = np.linalg.norm(k)
    base = rot(k / s, math.asin(s)) if s > 1e-12 else np.eye(3)
    R = base @ rot([0, 0, 1], roll) @ rot([0, 1, 0], wp) @ rot([1, 0, 0], wy)
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = p_shaft + R @ np.array([0, 0, TIP])
    return m


def pose_of(q):
    m = fk(q)
    return m[:3, 3], m[:3, 2], m[:3, 1]


class ToolPoseSolveTests(unittest.TestCase):
    def test_converges_to_reachable_target_with_orientation(self):
        q_true = np.array([2.0, -1.5, 0.042, 0.25, -0.1, 0.3])
        p, a, y = pose_of(q_true)
        r = tool_pose.solve(fk, q_true + np.array([0.4, 0.3, -0.002, -0.15, 0.05, -0.1]),
                            LIMITS, p, a, y)
        self.assertTrue(r['converged'], r)
        rp, ra, ry = pose_of(r['q'])
        np.testing.assert_allclose(rp, p, atol=2e-4)
        np.testing.assert_allclose(ra, a, atol=0.02)
        np.testing.assert_allclose(ry, y, atol=0.02)
        self.assertLess(r['pos_err_m'], 2e-4)

    def test_position_only_converges(self):
        q_true = np.array([-1.0, 1.0, 0.038, 0.0, 0.0, 0.0])
        p, _, _ = pose_of(q_true)
        r = tool_pose.solve(fk, np.array([0.0, 0.0, 0.04, 0.0, 0.0, 0.0]), LIMITS, p)
        self.assertTrue(r['converged'], r)
        self.assertLess(r['pos_err_m'], 2e-4)
        self.assertIsNone(r['axis_err_rad'])
        self.assertIsNone(r['yaw_axis_err_rad'])

    def test_respects_limits_and_reports_clamped_axis(self):
        # A point far to +x needs |yaw| beyond the 5 deg limit; the solver must stop at
        # the margin-shrunk bound instead of commanding the limit edge.
        p, _, _ = pose_of(np.array([9.0, 0.0, 0.04, 0.0, 0.0, 0.0]))
        r = tool_pose.solve(fk, np.zeros(6), LIMITS, p)
        self.assertLessEqual(r['q'][0], LIMITS[1][0] - 0.5 + 1e-6)
        self.assertIn('yaw_deg', r['at_limit'])
        for j in range(6):
            self.assertGreaterEqual(r['q'][j], LIMITS[0][j])
            self.assertLessEqual(r['q'][j], LIMITS[1][j])

    def test_unreachable_target_reports_non_convergence(self):
        p, _, _ = pose_of(np.zeros(6))
        r = tool_pose.solve(fk, np.zeros(6), LIMITS, p + np.array([0.5, 0.0, 0.0]))
        self.assertFalse(r['converged'])
        self.assertGreater(r['pos_err_m'], 0.01)

    def test_position_dominates_conflicting_orientation(self):
        # Same position, but ask for a jaw axis the wrist limits cannot express here:
        # the solve should keep position accuracy while honestly reporting the axis
        # residual instead of drifting away to satisfy orientation.
        q_true = np.array([0.0, 0.0, 0.04, 0.0, 0.0, 0.0])
        p, _, y = pose_of(q_true)
        r = tool_pose.solve(fk, q_true, LIMITS, p, np.array([1.0, 0.0, 0.0]), y)
        self.assertLess(r['pos_err_m'], 2e-3)
        self.assertIsNotNone(r['axis_err_rad'])

    def test_yaw_axis_is_sign_free(self):
        # The opening axis may point either way (blades are mirror images): a target of
        # -y must converge as well as +y.
        q_true = np.array([1.0, 0.5, 0.042, 0.2, 0.1, -0.15])
        p, _, y = pose_of(q_true)
        for target in (y, -y):
            r = tool_pose.solve(fk, q_true + np.array([0.2, -0.1, 0.001, 0.05, -0.03, 0.02]),
                                LIMITS, p, None, target)
            self.assertTrue(r['converged'], (target, r))


if __name__ == '__main__':
    unittest.main()
