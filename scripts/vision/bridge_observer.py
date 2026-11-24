"""Image-only observation of physical row-8 ligaments on the synthetic sheet."""
import numpy as np

from cut_perception import _sheet_mask

COLORS = np.asarray(((.18, .82, .25), (.27, .72, .43), (.19, .57, .72),
                     (.15, .36, .88), (.47, .76, .32), (.61, .65, .28),
                     (.34, .73, .64))) * 255
COLOR_TOL = 9.
MIN_PATCH_PX = 5
MIN_DEEP_PX = 3
DEEP_M = .002
MAX_FRONT_PX = 14


def measure(rgb, depth, R, t, f, estimate):
    cut = estimate.get('cut') or {}
    tissue = estimate.get('tissue') or {}
    scissors = estimate.get('scissors') or {}
    base = dict(status='UNKNOWN', reason='', patch_px=[], absent_lanes=[], deep_px={})
    if (not scissors.get('valid') or not tissue.get('valid') or not cut.get('front_valid')
            or cut.get('mode') not in ('slit', 'bank_slit')
            or (cut.get('slit_px') or 0) < 6):
        return dict(base, reason='no current direct slit with optical scissors')
    rgb, depth = np.asarray(rgb, float), np.asarray(depth, float)
    if depth.ndim != 2 or rgb.shape != depth.shape + (3,):
        return dict(base, reason='bad RGB-D exposure')
    H, W = depth.shape
    mask = _sheet_mask(rgb)
    y, x = np.nonzero(mask)
    if len(x) < 40:
        return dict(base, reason='no registered sheet')
    roi = np.zeros((H, W), dtype=bool)
    roi[max(0, y.min()-12):min(H, y.max()+13),
        max(0, x.min()-12):min(W, x.max()+13)] = True
    patches, centers, depths = [], [], []
    for color in COLORS:
        found = (np.sum((rgb - color)**2, axis=2) <= COLOR_TOL**2) & roi
        found &= np.isfinite(depth) & (depth > 0) & (depth < .3)
        values = np.argwhere(found)
        patches.append(int(len(values)))
        centers.append(None if len(values) < MIN_PATCH_PX else
                       np.median(values[:, ::-1], axis=0).tolist())
        depths.append(None if len(values) < MIN_PATCH_PX else
                      float(np.median(depth[found])))
    base['patch_px'] = patches
    base['absent_lanes'] = [i+8 for i, n in enumerate(patches) if n < MIN_PATCH_PX]
    if min(patches) >= MIN_PATCH_PX:
        return dict(base, status='NOT_STOP', reason='all physical row-8 ligaments visible')
    R, t = np.asarray(R), np.asarray(t)
    front = R.T @ (np.asarray(cut['front'])-t)
    if (front.shape != (3,) or not np.isfinite(front).all() or front[2] <= 0):
        return dict(base, reason='bad camera-space front')
    front_uv = np.asarray([W/2 - f*front[0]/front[2], H/2 - f*front[1]/front[2]])
    if not np.isfinite(front_uv).all():
        return dict(base, reason='front projects off camera')
    # A missing material patch is not proof of a cut. Two visible immediate
    # neighbours locate its slot and its depth must show through the sheet.
    for i in range(1, len(COLORS)-1):
        if patches[i] >= MIN_PATCH_PX or centers[i-1] is None or centers[i+1] is None:
            continue
        projected = (np.asarray(centers[i-1]) + np.asarray(centers[i+1])) / 2
        if np.linalg.norm(projected - front_uv) > MAX_FRONT_PX:
            continue
        px, py = np.rint(projected).astype(int)
        if not (3 <= px < W-3 and 3 <= py < H-3):
            continue
        observed = depth[py-2:py+3, px-2:px+3]
        good = np.isfinite(observed) & (observed > max(depths[i-1], depths[i+1]) + DEEP_M)
        deep = int(good.sum())
        base['deep_px'][str(i+8)] = deep
        if deep >= MIN_DEEP_PX:
            return dict(base, status='STOP_CANDIDATE',
                        reason='identified severed bridge with deep gap at direct slit')
    return dict(base, reason='one or more ligament slots missing or not depth-confirmed')
