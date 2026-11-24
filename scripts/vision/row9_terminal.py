"""Passive same-exposure row-9/slit proximity probe, not a structural endpoint oracle."""
import numpy as np

import cut_perception
import material_coordinate as material

LIMIT_M = .00035
BANK_M = .001
MIN_BANK_PAIRS = 3
MIN_SLIT_PX = 6


def measure(rgb, depth, R, t, f, estimate):
    cut = estimate.get('cut') or {}
    tissue = estimate.get('tissue') or {}
    scissors = estimate.get('scissors') or {}
    base = dict(status='UNKNOWN', banks=None, slit_px=cut.get('slit_px'))
    if (not scissors.get('valid') or not tissue.get('valid') or not cut.get('front_valid')
            or cut.get('mode') not in ('slit', 'bank_slit')
            or (cut.get('slit_px') or 0) < MIN_SLIT_PX):
        return dict(base, reason='no current direct slit and visible scissors')
    R, t = np.asarray(R), np.asarray(t)
    front = R.T @ (np.asarray(cut['front']) - t)
    center = R.T @ (np.asarray(tissue['centroid']) - t)
    axis = R.T @ np.asarray(cut['direction'])
    normal = R.T @ np.asarray(tissue['normal'])
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (front, center, axis, normal)):
        return dict(base, reason='invalid 3D geometry')
    lateral = np.cross(normal, axis)
    if np.linalg.norm(lateral) < 1e-8:
        return dict(base, reason='degenerate plane')
    lateral /= np.linalg.norm(lateral)
    along_front = float((front - center) @ axis)
    line_lat = float((front - center) @ lateral)
    rgb, depth = np.asarray(rgb, dtype=float), np.asarray(depth, dtype=float)
    H, W = depth.shape
    if rgb.shape != (H, W, 3):
        return dict(base, reason='invalid exposure shape')
    camera = cut_perception.CutPerception(W, H, float(f), .3, [0, 0, 0])
    along, across, _ = camera._plane_coords(center, normal, axis, lateral)
    mask = cut_perception._sheet_mask(rgb)
    label = np.full((H, W), -1, dtype=np.int8)
    for i, color in enumerate(material.COLORS):
        matches = np.linalg.norm(rgb - np.asarray(color)*255, axis=2) <= material.COLOR_TOL
        label[mask & matches] = i
    samples = []
    for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        ascending = along[first] < along[second]
        lo = np.where(ascending, label[first], label[second])
        hi = np.where(ascending, label[second], label[first])
        z0 = np.where(ascending, depth[first], depth[second])
        z1 = np.where(ascending, depth[second], depth[first])
        valid = ((lo == 2) & (hi == 3) & np.isfinite(z0) & np.isfinite(z1)
                 & (z0 > 0) & (z1 > 0) & (z0 < .3) & (z1 < .3)
                 & (np.abs(z0 - z1) <= material.DEPTH_EVIDENCE_M))
        position = (along[first] + along[second]) / 2
        side = (across[first] + across[second]) / 2 - line_lat
        samples.append(np.column_stack((position[valid], side[valid])))
    samples = np.concatenate(samples)
    banks = []
    for sign in (-1, 1):
        side = sign * samples[:, 1]
        q = samples[(side > 0) & (side <= BANK_M), 0]
        banks.append(dict(pairs=int(len(q)),
                          offset_mm=None if len(q) == 0 else float(np.median(q-along_front)*1000)))
    if any(b['pairs'] < MIN_BANK_PAIRS for b in banks):
        return dict(base, banks=banks, reason='row 9 not resolved at both actual slit banks')
    good = all(abs(b['offset_mm']) <= LIMIT_M*1000 for b in banks)
    return dict(base, banks=banks, status='STOP_CANDIDATE' if good else 'NOT_STOP',
                reason='' if good else 'observed row 9 is not coincident with slit')
