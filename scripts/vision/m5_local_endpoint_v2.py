"""Local-depth bilateral row-9 reading. This module issues no motion permits."""
import numpy as np

import cut_perception as cp
import material_coordinate as mc
import row9_terminal as prior


def _surface_at_apex(mask, z, dx, dy, apex_z):
    """Lift the apex ray from a nearby same-bank inverse-depth fit."""
    yy, xx = np.nonzero(mask)
    if len(xx) < 12:
        return None
    x, y, depth = dx[yy, xx], dy[yy, xx], z[yy, xx]
    if min(np.ptp(x), np.ptp(y)) < 2 or np.min(np.hypot(x, y)) > 5:
        return None
    design = np.column_stack((x, y, np.ones(len(xx))))
    beta, _, rank, _ = np.linalg.lstsq(design, 1 / depth, rcond=None)
    model = design @ beta
    if rank != 3 or beta[2] <= 0 or np.any(model <= 0):
        return None
    lift = 1 / beta[2]
    if abs(lift - apex_z) > mc.DEPTH_EVIDENCE_M or np.max(abs(1 / model - depth)) > .0005:
        return None
    return float(lift)


def measure(rgb, depth, R, t, focal, estimate):
    est = estimate or {}
    cut, tissue = est.get('cut') or {}, est.get('tissue') or {}
    unknown = dict(status='UNKNOWN', reason='no direct slit, tissue or optical tool')
    if (not (est.get('scissors') or {}).get('valid') or not tissue.get('valid')
            or not cut.get('front_valid') or cut.get('mode') not in ('slit', 'bank_slit')
            or (cut.get('slit_px') or 0) < prior.MIN_SLIT_PX):
        return unknown
    R, t = np.asarray(R), np.asarray(t)
    front, c = (R.T @ (np.asarray(p)-t) for p in (cut['front'], tissue['centroid']))
    axis, normal = (R.T @ np.asarray(x) for x in (cut['direction'], tissue['normal']))
    lateral = np.cross(normal, axis)
    im, z = np.asarray(rgb, dtype=float), np.asarray(depth, dtype=float)
    if (z.ndim != 2 or im.shape != z.shape+(3,) or not np.isfinite(focal) or focal <= 0
            or not np.isfinite((front, c, axis, normal)).all() or not 0 < front[2] < .3
            or np.linalg.norm(lateral) < 1e-8):
        return dict(unknown, reason='invalid optical geometry')
    lateral /= np.linalg.norm(lateral)
    h, w = z.shape
    camera = cp.CutPerception(w, h, focal, .3, [0, 0, 0])
    u, v = camera._project(front)
    if not (8 <= u < w-8 and 8 <= v < h-8):
        return dict(unknown, reason='apex outside usable image')
    mask = cp._sheet_mask(im)
    label = np.full(z.shape, -1, dtype=np.int8)
    for i, color in enumerate(mc.COLORS):
        label[mask & (np.linalg.norm(im - 255*np.asarray(color), axis=2) <= mc.COLOR_TOL)] = i
    along, across, _ = camera._plane_coords(c, normal, axis, lateral)
    along -= (front-c) @ axis
    across -= (front-c) @ lateral
    y, x = np.indices(z.shape)
    dx, dy = x-u, y-v
    measured = np.isfinite(z) & (z > 0) & (z < .3)
    patch = (measured & (label >= 0) & (along >= .00025) & (along <= .0015)
             & (np.abs(across) <= .0015) & (np.hypot(dx, dy) <= 16))
    lifts = [_surface_at_apex(patch & (side*across > 0), z, dx, dy, front[2])
             for side in (-1, 1)]
    if any(d is None for d in lifts) or abs(lifts[0]-lifts[1]) > .0005:
        return dict(unknown, reason='two local apex surfaces not supported or disagree', depths_m=lifts)
    edge_points = []
    for dy, dx in ((0, 1), (1, 0)):
        a, b = np.s_[:h-dy, :w-dx], np.s_[dy:, dx:]
        asc = along[a] < along[b]
        hit = (measured[a] & measured[b] & (np.where(asc, label[a], label[b]) == 2)
               & (np.where(asc, label[b], label[a]) == 3) & (abs(z[a]-z[b]) <= mc.DEPTH_EVIDENCE_M))
        yy, xx = np.nonzero(hit)
        za, zb = z[a][hit], z[b][hit]
        total = za+zb
        edge_points.append(.5*np.column_stack((((w/2-xx)*total-dx*zb)/focal,
                                               ((h/2-yy)*total-dy*zb)/focal, total)))
    points = np.concatenate(edge_points)
    signs = (-1, 1)
    banks = []
    ray = front / front[2]
    for side, lift in zip(signs, lifts, strict=True):
        delta = points - ray*lift
        distance = side*(delta @ lateral)
        near = (distance > 0) & (distance <= prior.BANK_M)
        banks.append(dict(pairs=int(near.sum()),
                          offset_m=float(np.median(delta[near] @ axis)) if near.any() else None))
    result = dict(unknown, reason='missing direct bilateral near-lip support', banks=banks, depths_m=lifts)
    if any(bank['pairs'] < prior.MIN_BANK_PAIRS for bank in banks):
        return result
    offsets = [bank['offset_m'] for bank in banks]
    if all(abs(x) <= prior.LIMIT_M for x in offsets):
        return dict(result, status='ROW9_MATCH', reason='')
    if all(x > prior.LIMIT_M for x in offsets):
        return dict(result, status='BELOW', reason='')
    return dict(result, reason='inconsistent endpoints or beyond row 9')
