"""Damped-least-squares IK for the 6-DOF tool pose under RCM.

q = [yaw_deg, pitch_deg, ins_m, roll, wpitch, wyaw]: the shaft command exactly as
setRCMTarget/setToolPose take it (yaw/pitch in degrees, insertion in metres) plus the
three logical wrist angles in radians from getWristState. fk(q) returns the TCP pose as a
4x4 (or 3x4) matrix; for the live arm it wraps Lua toolPoseFK, for tests any analytic map.

Orientation terms: the TCP z column is the jaw pointing axis (the closed blade direction;
jaws are mounted along the shaft distal axis). The opening/yaw axis is the TCP column the
grip hinges rotate about — measured on the verified scene (see SimWorld) and passed in as
yaw_col; its sign is free because the two blades are mirror images, so the nearer of
+-target_yaw_axis is used.
"""
import math

import numpy as np

Q_NAMES = ('yaw_deg', 'pitch_deg', 'ins_m', 'roll', 'wpitch', 'wyaw')
# Solver works in scaled coordinates: degrees -> radians, metres/radians unchanged.
_SCALE = np.array([math.pi / 180, math.pi / 180, 1.0, 1.0, 1.0, 1.0])
# Per-iteration trust region, in scaled coordinates (rad, m, rad).
_STEP_MAX = np.array([0.10, 0.10, 0.01, 0.15, 0.15, 0.15])
# Limit margins (same units as q): the live controller refuses boundary commands, so the
# solver clamps to slightly shrunk bounds instead of landing exactly on a limit.
_MARGIN = np.array([0.5, 0.5, 0.001, 0.02, 0.02, 0.02])
_FD_EPS = np.array([1e-5, 1e-5, 1e-6, 1e-5, 1e-5, 1e-5])
_DAMPING = 1e-3


def _pose(fk, q):
    m = np.asarray(fk(np.asarray(q, dtype=float)), dtype=float)
    if m.shape == (3, 4):
        m = np.vstack([m, [0, 0, 0, 1]])
    return m


def _residual(m, target_p, target_axis, target_yaw_axis, w_pos, w_rot, axis_col, yaw_col):
    """Position residual (m) then axis residuals (m via w_rot); returns (e, errs)."""
    R, p = m[:3, :3], m[:3, 3]
    e = [w_pos * (p - target_p)]
    pos_err = float(np.linalg.norm(p - target_p))
    axis_err = yaw_err = None
    if target_axis is not None:
        a = R[:, axis_col]
        axis_err = float(math.acos(max(-1.0, min(1.0, float(a @ target_axis)))))
        e.append(w_rot * np.cross(a, target_axis))
    if target_yaw_axis is not None:
        y = R[:, yaw_col]
        # Blades are mirror images: +-target are equally good; take the nearer side.
        sign = 1.0 if float(y @ target_yaw_axis) >= 0 else -1.0
        yaw_err = float(math.acos(max(-1.0, min(1.0, abs(float(y @ target_yaw_axis))))))
        e.append(w_rot * np.cross(y, sign * target_yaw_axis))
    return np.concatenate(e), pos_err, axis_err, yaw_err


def solve(fk, q0, limits, target_p, target_axis=None, target_yaw_axis=None,
          w_pos=1.0, w_rot=0.02, iters=30, tol_p=2e-4, tol_axis=0.02,
          axis_col=2, yaw_col=1):
    """DLS with central finite differences, margin-shrunk limits and a per-axis trust region.

    limits: (lo, hi) arrays of 6 in q units (deg, deg, m, rad, rad, rad).
    Returns dict(q, pos_err_m, axis_err_rad, yaw_axis_err_rad, iterations, converged,
    at_limit)."""
    lo = np.maximum(np.asarray(limits[0], dtype=float), -np.inf) + _MARGIN
    hi = np.minimum(np.asarray(limits[1], dtype=float), np.inf) - _MARGIN
    target_p = np.asarray(target_p, dtype=float)
    if target_axis is not None:
        target_axis = np.asarray(target_axis, dtype=float) / np.linalg.norm(target_axis)
    if target_yaw_axis is not None:
        target_yaw_axis = np.asarray(target_yaw_axis, dtype=float) / np.linalg.norm(target_yaw_axis)

    q = np.clip(np.asarray(q0, dtype=float), lo, hi)
    z = q * _SCALE
    zlo, zhi = lo * _SCALE, hi * _SCALE
    e, pos_err, axis_err, yaw_err = _residual(_pose(fk, q), target_p, target_axis,
                                            target_yaw_axis, w_pos, w_rot, axis_col, yaw_col)
    used = 0
    for it in range(iters):
        used = it + 1
        J = np.empty((e.size, 6))
        for j in range(6):
            dz = np.zeros(6)
            dz[j] = _FD_EPS[j]
            dp = _pose(fk, (z + dz) / _SCALE)
            dm = _pose(fk, (z - dz) / _SCALE)
            ep = _residual(dp, target_p, target_axis, target_yaw_axis, w_pos, w_rot,
                           axis_col, yaw_col)[0]
            em = _residual(dm, target_p, target_axis, target_yaw_axis, w_pos, w_rot,
                           axis_col, yaw_col)[0]
            J[:, j] = (ep - em) / (2 * _FD_EPS[j])
        # Damped least squares step in scaled coordinates.
        A = J.T @ J + _DAMPING ** 2 * np.eye(6)
        try:
            dz = np.linalg.solve(A, J.T @ e)
        except np.linalg.LinAlgError:
            break
        dz = np.clip(dz, -_STEP_MAX, _STEP_MAX)
        z_new = np.clip(z - dz, zlo, zhi)
        e_new, pos_err, axis_err, yaw_err = _residual(_pose(fk, z_new / _SCALE), target_p,
                                                    target_axis, target_yaw_axis,
                                                    w_pos, w_rot, axis_col, yaw_col)
        z = z_new
        q = z / _SCALE
        e = e_new
        if (pos_err < tol_p and (axis_err is None or axis_err < tol_axis)
                and (yaw_err is None or yaw_err < tol_axis)):
            break
    at_limit = [Q_NAMES[j] for j in range(6) if q[j] <= lo[j] + 1e-9 or q[j] >= hi[j] - 1e-9]
    converged = (pos_err < tol_p and (axis_err is None or axis_err < tol_axis)
                 and (yaw_err is None or yaw_err < tol_axis))
    return dict(q=q, pos_err_m=float(pos_err),
                axis_err_rad=None if axis_err is None else float(axis_err),
                yaw_axis_err_rad=None if yaw_err is None else float(yaw_err),
                iterations=used, converged=bool(converged), at_limit=at_limit)
