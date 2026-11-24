"""Require two visible exposed fracture surfaces; missing intact ink cannot authorize STOP."""
import numpy as np

from bridge_observer import (COLOR_TOL, DEEP_M, MAX_FRONT_PX, MIN_PATCH_PX,
                             measure as intact_reading)
from cut_perception import _sheet_mask

FRACTURE_COLORS = np.asarray(((.12, .94, .17), (.09, .21, .97))) * 255
MIN_FRACTURE_PX = 3
MIN_GAP_PX = 2
MAX_PAIR_PX = 12


def measure(rgb, depth, R, t, f, estimate):
    original = intact_reading(rgb, depth, R, t, f, estimate)
    result = dict(original, status='UNKNOWN', fracture_px=[0, 0], fracture_deep_px=0)
    if original['reason'] == 'no current direct slit with optical scissors':
        return result
    rgb = np.asarray(rgb, float)
    depth = np.asarray(depth, float)
    H, W = depth.shape
    sheet = _sheet_mask(rgb)
    ys, xs = np.nonzero(sheet)
    if len(xs) < 40:
        return result
    roi = np.zeros((H, W), dtype=bool)
    roi[max(0, ys.min()-12):min(H, ys.max()+13),
        max(0, xs.min()-12):min(W, xs.max()+13)] = True
    masks = [np.sum((rgb-color)**2, axis=2) <= COLOR_TOL**2
             for color in FRACTURE_COLORS]
    masks = [mask & roi & np.isfinite(depth) & (depth > 0) & (depth < .3) for mask in masks]
    result['fracture_px'] = [int(mask.sum()) for mask in masks]
    if any(n < MIN_FRACTURE_PX for n in result['fracture_px']):
        if original['status'] == 'NOT_STOP':
            result.update(status='NOT_STOP', reason='all physical row-8 ligaments visible')
        else:
            result['reason'] = 'no positive paired fracture faces'
        return result
    cut = (estimate.get('cut') or {})
    front = np.asarray(R).T @ (np.asarray(cut['front'])-np.asarray(t))
    if front.shape != (3,) or not np.isfinite(front).all() or front[2] <= 0:
        return result
    front_uv = np.asarray([W/2-f*front[0]/front[2], H/2-f*front[1]/front[2]])
    left_yx, right_yx = [np.argwhere(mask) for mask in masks]
    for ly, lx in left_yx:
        matches = right_yx[np.linalg.norm(right_yx - [ly, lx], axis=1) <= MAX_PAIR_PX]
        for ry, rx in matches:
            center = np.asarray([(lx+rx)/2, (ly+ry)/2])
            if np.linalg.norm(center-front_uv) > MAX_FRONT_PX:
                continue
            px, py = np.rint(center).astype(int)
            if not (3<=px<W-3 and 3<=py<H-3):
                continue
            z = depth[py-2:py+3,px-2:px+3]
            threshold = max(depth[ly,lx], depth[ry,rx])+DEEP_M
            deep = int((np.isfinite(z) & (z>threshold)).sum())
            result['fracture_deep_px'] = max(result['fracture_deep_px'], deep)
            if deep >= MIN_GAP_PX:
                result.update(status='STOP_CANDIDATE',
                              reason='paired physical fracture faces and deep gap')
                return result
    result['reason'] = 'fracture colors present but not paired across a deep direct slit'
    return result
