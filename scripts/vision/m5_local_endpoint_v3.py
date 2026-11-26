"""Read-only near-lip and distal-plane geometry from actual RGB-D."""
import numpy as np

import cut_perception as cp
import material_coordinate as mc
from m5_local_endpoint_v2 import prior


def _lift(mask, z, x, y, original, sigma):
    px, py, depth = x[mask], y[mask], z[mask]
    if len(px) < 12:
        return None, 'insufficient same-bank pixels'
    if min(np.ptp(px), np.ptp(py)) < 2 or np.min(np.hypot(px, py)) > 5:
        return None, 'insufficient nearby spatial support'
    design = np.column_stack((px, py, np.ones(len(px))))
    coef, _, rank, _ = np.linalg.lstsq(design, 1/depth, rcond=None)
    modeled = design @ coef
    if rank != 3 or coef[-1] <= 0 or np.any(modeled <= 0):
        return None, 'degenerate local surface'
    fitted = 1/coef[-1]
    if abs(fitted-original) > mc.DEPTH_EVIDENCE_M:
        return None, 'local and original apex depths disagree'
    if np.max(abs(1/modeled-depth)) > .0005+3*sigma:
        return None, 'surface residual exceeds calibrated noise bound'
    return float(fitted), ''


def _distal_plane(z, label, x, y, along, across, axis, normal, f, h, w, front, sigma):
    ok = (label >= 2) & np.isfinite(z) & (z > 0) & (z < .3)
    if ok.sum() < 24:
        return dict(reason='missing distal strip')
    rays = np.column_stack(((w/2-x[ok])/f, (h/2-y[ok])/f, np.ones(ok.sum())))
    points = rays*z[ok, None]
    center = points.mean(axis=0)
    _, spread, axes = np.linalg.svd(points-center, full_matrices=False)
    n = axes[-1]
    a = axis-(axis@n)*n
    if spread[1] <= .001 or np.linalg.norm(a) < 1e-6:
        return dict(reason='distal strip fit is degenerate')
    if np.quantile(abs((points-center) @ n), .9) > .0005+3*sigma:
        return dict(reason='distal strip not locally planar')
    a /= np.linalg.norm(a)
    bins = {6: [], 9: [], 13: []}
    for dy, dx in ((0, 1), (1, 0)):
        l, r = np.s_[:h-dy, :w-dx], np.s_[dy:, dx:]
        ascend = along[l] < along[r]
        low, high = np.where(ascend, label[l], label[r]), np.where(ascend, label[r], label[l])
        z0, z1 = np.where(ascend, z[l], z[r]), np.where(ascend, z[r], z[l])
        m0, m1 = np.where(ascend, ok[l], ok[r]), np.where(ascend, ok[r], ok[l])
        side = (across[l]+across[r])/2
        base = (np.isfinite(z0) & np.isfinite(z1) & (z0 > 0) & (z1 > 0)
                & (abs(side) >= mc.BANK_INNER_M) & (abs(side) <= mc.BANK_OUTER_M))
        for row, evidence in (
            (6, (low == 1) & (high == 2) & (abs(z1-z0) <= mc.DEPTH_EVIDENCE_M)),
            (9, (low == 2) & (high == 3) & (abs(z1-z0) <= mc.DEPTH_EVIDENCE_M)),
            (13, (low == 3) & (high == -1) & m0 & ~m1 & (z1-z0 > mc.DEPTH_EVIDENCE_M)),
        ):
            hit = base & evidence
            iy, ix = np.nonzero(hit)
            ray = np.column_stack(((w/2-ix-.5*dx)/f, (h/2-iy-.5*dy)/f, np.ones(len(ix))))
            denom = ray@n
            good = abs(denom) > 1e-6
            local = ray[good]*(float(center@n)/denom[good])[:, None]
            flat = .5*(along[l]+along[r])[hit][good]
            bins[row].append(np.column_stack((side[hit][good], flat, (local-front)@a)))
    rows = {}
    for row, samples in bins.items():
        data = np.concatenate(samples)
        banks = [data[side*data[:, 0] > 0, 1:] for side in (-1, 1)]
        if min(map(len, banks)) < mc.MIN_PAIRS:
            rows[row] = dict(pairs=list(map(len, banks)))
            continue
        flat, local = np.mean([np.median(b, axis=0) for b in banks], axis=0)
        rows[row] = dict(pairs=list(map(len, banks)), global_mm=float(flat*1000),
                         distal_mm=float(local*1000))
    spans = {}
    for first, last in ((6, 9), (9, 13)):
        if all('distal_mm' in rows[j] for j in (first, last)):
            spans[f'{first}_{last}'] = {k: rows[last][k]-rows[first][k]
                                        for k in ('global_mm', 'distal_mm')}
    return dict(rows=rows, spans=spans, normal_difference_deg=float(np.degrees(
        np.arccos(np.clip(abs(n@normal), 0., 1.)))))


def measure(rgb, depth, R, t, focal, estimate, *, sigma):
    e = estimate or {}
    cut, tissue = e.get('cut') or {}, e.get('tissue') or {}
    result = dict(status='UNKNOWN', reason='no direct slit, sheet or observed tool')
    if (not (e.get('scissors') or {}).get('valid') or not tissue.get('valid')
            or not cut.get('front_valid') or cut.get('mode') not in ('slit', 'bank_slit')
            or (cut.get('slit_px') or 0) < prior.MIN_SLIT_PX):
        return result
    im, z, R, t = (np.asarray(v, dtype=float) for v in (rgb, depth, R, t))
    if (z.ndim != 2 or im.shape != z.shape+(3,) or R.shape != (3, 3) or t.shape != (3,)
            or not np.isfinite([focal, sigma]).all() or focal <= 0 or sigma < 0):
        return dict(result, reason='unsupported calibration')
    front, center = (R.T @ (np.asarray(p)-t) for p in (cut['front'], tissue['centroid']))
    axis, normal = (R.T @ np.asarray(p) for p in (cut['direction'], tissue['normal']))
    lateral = np.cross(normal, axis)
    if (not np.isfinite([front, center, axis, normal]).all() or not 0 < front[2] < .3
            or np.linalg.norm(lateral) < 1e-8):
        return dict(result, reason='unusable view geometry')
    lateral /= np.linalg.norm(lateral)
    h, w = z.shape
    camera = cp.CutPerception(w, h, focal, .3, [0, 0, 0])
    u, v = camera._project(front)
    if not (8 <= u < w-8 and 8 <= v < h-8):
        return dict(result, reason='apex outside usable view')
    along, across, _ = camera._plane_coords(center, normal, axis, lateral)
    along -= (front-center) @ axis
    across -= (front-center) @ lateral
    y, x = np.indices(z.shape)
    pink = cp._sheet_mask(im)
    valid = np.isfinite(z) & (z > 0) & (z < .3)
    label = np.full(z.shape, -1, np.int8)
    for j, color in enumerate(mc.COLORS):
        label[pink & (np.linalg.norm(im-255*np.asarray(color), axis=-1) <= mc.COLOR_TOL)] = j
    result['distal'] = _distal_plane(z, label, x, y, along, across, axis, normal,
                                     focal, h, w, front, sigma)
    dx, dy = x-u, y-v
    patch = (pink & valid & (along >= .00025) & (along <= .0025)
             & (abs(across) <= .0025) & (np.hypot(dx, dy) <= 24))
    lifts, reasons = zip(*[_lift(patch & (s*across > 0), z, dx, dy, front[2], sigma)
                           for s in (-1, 1)])
    result.update(depths_m=lifts, lift_reasons=reasons, sigma_m=sigma)
    if None in lifts or abs(lifts[0]-lifts[1]) > .0005:
        return dict(result, reason='two local apex surfaces not supported or disagree')
    edges = []
    for dy, dx in ((0, 1), (1, 0)):
        l, r = np.s_[:h-dy, :w-dx], np.s_[dy:, dx:]
        asc = along[l] < along[r]
        hit = (valid[l] & valid[r] & (np.where(asc, label[l], label[r]) == 2)
               & (np.where(asc, label[r], label[l]) == 3) & (abs(z[l]-z[r]) <= mc.DEPTH_EVIDENCE_M))
        iy, ix = np.nonzero(hit)
        z1 = z[r][hit]
        total = z[l][hit]+z1
        edges.append(.5*np.column_stack((((w/2-ix)*total-dx*z1)/focal,
                                         ((h/2-iy)*total-dy*z1)/focal, total)))
    points = np.concatenate(edges)
    banks = []
    for s, lift in zip((-1, 1), lifts, strict=True):
        delta = points-(front/front[2])*lift
        space = s*(delta@lateral)
        near = (space > 0) & (space <= prior.BANK_M)
        banks.append(dict(pairs=int(near.sum()), offset_m=float(np.median(delta[near]@axis))
                          if near.any() else None))
    result['banks'] = banks
    if any(b['pairs'] < prior.MIN_BANK_PAIRS for b in banks):
        return dict(result, reason='missing direct bilateral near-lip support')
    offsets = [b['offset_m'] for b in banks]
    if all(abs(o) <= prior.LIMIT_M for o in offsets):
        return dict(result, status='ROW9_MATCH', reason='')
    if all(o > prior.LIMIT_M for o in offsets):
        return dict(result, status='BELOW', reason='')
    return dict(result, reason='inconsistent endpoints or beyond row 9')
