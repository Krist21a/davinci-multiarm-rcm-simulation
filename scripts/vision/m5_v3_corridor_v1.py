"""Offline-only two-dimensional corridor witnesses on the guided coded sheet."""
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from cell_certificate_v1 import block, decode
from m5_guided_bank_corridor_v1 import bank_patch
from ruler_readout import inks

FOOTPRINT = np.ones((4, 4), dtype=bool)


def hull(points, shape):
    mask = np.zeros(shape, dtype=bool)
    try:
        eq = ConvexHull(points).equations
    except QhullError:
        return mask
    y, x = points.min(0)
    Y, X = points.max(0) + 1
    yy, xx = np.mgrid[y:Y, x:X]
    mask[y:Y, x:X] = ((yy[..., None]*eq[:, 0] + xx[..., None]*eq[:, 1] + eq[:, 2]) <= 1e-8).all(-1)
    return mask


def grow(seed, allowed, edges, steps=None):
    """Four-neighbor reachability without crossing a depth discontinuity."""
    sy, sx = edges
    reached = seed & allowed
    for _ in range(steps if steps is not None else reached.size):
        next_mask = reached.copy()
        next_mask[1:, :] |= reached[:-1, :] & sy
        next_mask[:-1, :] |= reached[1:, :] & sy
        next_mask[:, 1:] |= reached[:, :-1] & sx
        next_mask[:, :-1] |= reached[:, 1:] & sx
        next_mask &= allowed
        if np.array_equal(reached, next_mask):
            break
        reached = next_mask
    return reached


def spanning_deep(deep, shores):
    labels, _ = ndimage.label(deep)
    candidates = set(np.unique(labels[ndimage.binary_erosion(deep, structure=FOOTPRINT)])) - {0}
    for shore in shores:
        candidates &= set(np.unique(labels[ndimage.binary_dilation(shore)]))
    return np.isin(labels, sorted(candidates))


def read_row(j, bank, cap, face, z, good, sigma, edges, face_seen):
    entry = dict(row=j, status='UNKNOWN')
    pts = [bank_patch((bank == j+13*side) & good) for side in range(2)]
    if any(not len(p) for p in pts):
        return dict(entry, reason='missing or ambiguous bank patch'), None
    anchors = np.concatenate(pts)
    rng = z[tuple(anchors.T)]
    lo, hi = float(rng.min()), float(rng.max())
    tol = .001+3*sigma
    domain = hull(anchors, z.shape)
    if not domain.any():
        return dict(entry, reason='degenerate row domain'), None
    blank = (bank < 0) & (cap < 0) & (face < 0)
    deep = domain & good & blank & (z > hi+.002+3*sigma)
    near = good & (z >= lo-tol) & (z <= hi+tol)
    shores = []
    for side, p in enumerate(pts):
        seed = np.zeros(z.shape, dtype=bool)
        seed[tuple(p.T)] = True
        allowed = (np.isin(bank, 13*side + np.arange(max(0, j-1), min(12, j+1)+1))
                   | np.isin(cap, (2*j+side, 2*(j+1)+side)))
        shores.append(grow(seed, near & allowed, edges, 12) & domain)
    faces = [domain & near & (face == 2*j+k) for k in range(2)]
    shallow = shores[0] | shores[1] | faces[0] | faces[1]
    left, right = (grow(s, shallow, edges) for s in shores)
    bridge = left & right
    linked_faces = [bool(block(m & (left | right)).any()) for m in faces]
    both_in_bridge = all(block(m & bridge).any() for m in faces)
    unresolved = int(np.count_nonzero(domain & ~(deep | shallow)))
    through = spanning_deep(deep, shores)
    status, reason = 'UNKNOWN', 'no positive local certificate'
    if both_in_bridge:
        status, reason = 'INTACT', ''
    elif unresolved:
        reason = 'unexplained samples inside row domain'
    elif any(linked_faces) and bool(ndimage.binary_erosion(deep, structure=FOOTPRINT).any()) and not bridge.any():
        status, reason = 'PARTIAL', ''
    elif face_seen:
        reason = 'observed central face vetoes OPEN'
    elif bridge.any():
        reason = 'shallow cross-bank bridge vetoes OPEN'
    elif through.any():
        status, reason = 'OPEN', ''
    return dict(entry, status=status, reason=reason, unexplained_pixels=unresolved,
                through_pixels=int(through.sum()), face_patches=linked_faces,
                bridge=bool(bridge.any()), bank_depth_m=[lo, hi]), (anchors, through, hi)


def measure(rgb, depth, *, far, sigma):
    result = dict(verdict='UNKNOWN', prefix_mm=None, rows=[], reason='invalid RGB-D')
    if (depth.ndim != 2 or rgb.shape != depth.shape+(3,) or rgb.dtype != np.uint8
            or depth.dtype.kind != 'f' or not (math.isfinite(far) and far > 0)
            or not (math.isfinite(sigma) and sigma >= 0)):
        return result
    bank = inks(rgb)
    yy, xx = np.where(bank >= 0)
    if not len(yy):
        return dict(result, reason='no bank ink')
    cap, face = decode(rgb)
    face_seen = np.bincount(face[face >= 0], minlength=26).reshape(13, 2).any(axis=1)
    roi = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
    bank, cap, face, z = bank[roi], cap[roi], face[roi], depth[roi]
    good = np.isfinite(z) & (z > 0) & (z < far*.999)
    tol = .001 + 3*sigma
    safe_z = np.where(good, z, 0.)
    edges = (good[1:] & good[:-1] & (np.abs(safe_z[1:]-safe_z[:-1]) <= tol),
             good[:, 1:] & good[:, :-1] & (np.abs(safe_z[:, 1:]-safe_z[:, :-1]) <= tol))
    pairs = [read_row(j, bank, cap, face, z, good, sigma, edges, face_seen[j])
             for j in range(13)]
    rows, proof = zip(*pairs, strict=True)
    result['rows'] = list(rows)
    states = [r['status'] for r in rows]
    n = next((j for j, s in enumerate(states) if s != 'OPEN'), 13)
    if not 0 < n <= 10:
        return dict(result, reason='no eligible open prefix from row zero')
    if states[n] not in ('INTACT', 'PARTIAL') or states[n+1] != 'INTACT':
        return dict(result, reason='missing positive terminal boundary')
    if any(s in ('OPEN', 'PARTIAL') for s in states[n+2:]):
        return dict(result, reason='later observed opening')
    eligible = good & (bank < 0) & (cap < 0) & (face < 0)
    path = np.logical_or.reduce([p[1] for p in proof[:n]])
    for (a, _, d), (b, _, e) in zip(proof[:n], proof[1:n]):
        path |= hull(np.concatenate((a, b)), z.shape) & eligible & (z > max(d, e)+.002+3*sigma)
    core, _ = ndimage.label(ndimage.binary_erosion(path, structure=FOOTPRINT))
    shared = set(np.unique(core[proof[0][1]])) - {0}
    for _, mask, _ in proof[1:n]:
        shared &= set(np.unique(core[mask]))
    if not shared:
        return dict(result, reason='no shared minimum-width through-depth across rows')
    return dict(result, verdict='BELOW' if n < 9 else 'TARGET', prefix_mm=n, reason='')
