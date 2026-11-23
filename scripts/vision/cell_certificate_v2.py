"""Local coded-face observations and a separate connected-prefix readout. No task control."""
import itertools

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from cell_certificate_v1 import OPEN, PRESENT, UNKNOWN, block, corners, decode, patch
from ruler_readout import inks


def _cross(v, p):
    return v[0]*p[..., 1] - v[1]*p[..., 0]


def _clip(poly, n, b):
    """Clip a convex polygon to n dot x <= b."""
    result = []
    for a, c in zip(poly, np.roll(poly, -1, axis=0)):
        da, dc = a @ n-b, c @ n-b
        if da <= 0:
            result.append(a)
        if (da < 0 < dc) or (dc < 0 < da):
            result.append(a + (da/(da-dc))*(c-a))
    return np.asarray(result, dtype=float).reshape(-1, 2)


def _tighten(poly, mask, bank, z, row, side, dz, exterior, opposite):
    """Continue the directly observed transverse ink edge into its attached cap.

    Each four-neighbour ink contact brackets an image crossing to half a pixel.
    Use a short straight run attached to this cap, not a line through bank
    centroids. No qualifying run means keep the original physical uncertainty.
    """
    end = row in (0, 13)
    if end:
        if not patch(opposite):
            return poly, 'cap'
        own = np.median(np.argwhere(mask)[:, ::-1], axis=0)+.5
        across = own - (np.median(np.argwhere(opposite)[:, ::-1], axis=0)+.5)
        if np.linalg.norm(across) <= 1:
            return poly, 'cap'
        across /= np.linalg.norm(across)
    lo, hi = np.argwhere(mask).min(0), np.argwhere(mask).max(0)
    y0, x0 = np.maximum(lo-16, 0)
    roi = np.s_[y0:hi[0]+17, x0:hi[1]+17]
    m, b, d = mask[roi], bank[roi], z[roi]
    before, after = 13*side+row-1, 13*side+row
    distance = ndimage.distance_transform_edt(~m)
    points, distances = [], []
    for dy, dx in ((0, 1), (1, 0)):
        p = np.s_[:b.shape[0]-dy, :b.shape[1]-dx]
        q = np.s_[dy:, dx:]
        near = np.minimum(distance[p], distance[q])
        if end:
            # At the physical ends the second row does not exist. Require the
            # same bank's outward rim against measured deep background, never
            # the inner longitudinal lip of a slit or a foreground shutter.
            ink, ext = (b == (after if row == 0 else before)), exterior[roi]
            hit = (ink[p] & ext[q] & (d[q] > d[p]+.002+2*dz)) | (ink[q] & ext[p] & (d[p] > d[q]+.002+2*dz))
        else:
            hit = (((b[p] == before) & (b[q] == after)) | ((b[p] == after) & (b[q] == before)))
            hit &= abs(d[p]-d[q]) <= 2*dz
        hit &= (near <= 13) & np.isfinite(d[p]) & np.isfinite(d[q]) & (d[p] > 0) & (d[q] > 0)
        y, x = np.where(hit)
        pairs = np.column_stack((x+x0+.5+.5*dx, y+y0+.5+.5*dy))
        keep = (pairs-own) @ across > 0 if end else np.ones(len(pairs), bool)
        points.extend(pairs[keep])
        distances.extend(near[hit][keep])
    if len(points) < 6 or min(distances) > np.sqrt(2):
        return poly, 'cap'
    pts = np.asarray(points)
    anchor = pts[np.argmin(distances)]
    if np.max(np.linalg.norm(pts[np.asarray(distances) <= np.sqrt(2)]-anchor, axis=1)) > 3:
        return poly, 'cap'
    pts = pts[np.linalg.norm(pts-anchor, axis=1) <= 12]
    graph = np.sum((pts[:, None]-pts)**2, axis=2) <= 2
    reached = np.linalg.norm(pts-anchor, axis=1) < .01
    while True:
        extended = graph[reached].any(axis=0)
        if np.array_equal(extended, reached):
            break
        reached = extended
    if not reached.all():
        return poly, 'cap'
    outer = pts[np.argmax(np.linalg.norm(pts-anchor, axis=1))]
    span = np.linalg.norm(outer-anchor)
    if span < 6:
        return poly, 'cap'
    axis = (outer-anchor)/span
    if end and axis @ across < .8:
        return poly, 'cap'
    normal = np.array([-axis[1], axis[0]])
    run = np.sort((pts-anchor) @ axis)
    if (np.max(abs((pts-anchor) @ normal)) > 1 or run[0] < -2
            or np.max(np.diff(run), initial=0) > 2):
        return poly, 'cap'
    # The two endpoints of the fitted segment each have a 0.5 px crossing
    # interval. sin(angle error) <= 1/span; widen for extrapolation to the
    # farthest feasible cap point rather than treating this line as exact.
    reach = np.max(np.linalg.norm(poly-anchor, axis=1))
    width = .5 + (reach+.5)/np.sqrt(span*span-1)
    strip = _clip(_clip(poly, normal, anchor @ normal+width), -normal, -anchor @ normal+width)
    return (strip, 'end-edge' if end else 'ink-edge') if len(strip) >= 3 else (None, 'edge conflicts with cap')


def _interior(polys, xy, *, footprint):
    centres = np.array([p.mean(axis=0) for p in polys])
    orient = np.sign(np.linalg.det(centres[1:]-centres[0]))
    valid = np.full(xy.shape[:2], bool(orient))
    for k in range(3):
        for a, b in itertools.product(polys[k], polys[(k+1) % 3]):
            v = b-a
            bound = .5*abs(v).sum()+1e-8 if footprint else -1e-8
            valid &= orient*_cross(v, xy-a) >= bound
        if not valid.any():
            break
    return valid


def _crop_codes(rgb):
    # The certificate occupies only a small fraction of an HD image. Decode
    # in strips first; do not allocate full-frame floating point point clouds.
    bounds = []
    for y0 in range(0, rgb.shape[0], 96):
        cap, face = decode(rgb[y0:y0+96])
        y, x = np.where((cap >= 0) | (face >= 0))
        if len(x):
            bounds.append([x.min(), y.min()+y0, x.max(), y.max()+y0])
    if bounds:
        b = np.array(bounds)
        x, y = np.maximum(0, b[:, :2].min(0)-16)
        X, Y = b[:, 2:].max(0)+17
        return np.s_[y:Y, x:X]
    return None


def classify(rgb, depth, *, focal, principal, sigma=0.):
    """Return local_mm separately from decision/prefix; neither is an action permit."""
    rgb, z, pp = np.asarray(rgb), np.asarray(depth), np.asarray(principal)
    empty = dict(decision=UNKNOWN, prefix=None, local_mm=None)
    if (z.ndim != 2 or rgb.shape != z.shape+(3,) or pp.shape != (2,)
            or not np.isfinite([*pp, focal, sigma]).all() or focal <= 0 or sigma < 0):
        return dict(empty, reason='invalid sensor calibration')
    crop = _crop_codes(rgb)
    if crop is None:
        return dict(empty, reason='no coded material')
    origin = np.array([crop[1].start, crop[0].start])
    z, rgb = z[crop], rgb[crop]
    cap, face = decode(rgb)
    bank = inks(rgb)
    depth_ok = np.isfinite(z) & (z > 0)
    blank = (cap < 0) & (face < 0) & (bank < 0) & depth_ok
    yy, xx = np.indices(z.shape)
    xy = np.stack((xx+.5, yy+.5), axis=-1)
    dz = .00025+3*sigma
    nodes, report = [], []
    for k in range(28):
        mask = cap == k
        base = corners(mask, z, xy, pp-origin, focal, dz)
        node = None
        if base is not None:
            center, radius, zz = base
            box = center+radius*np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
            poly, method = _tighten(box, mask, bank, z, k//2, k%2, dz, blank, cap == (k^1))
            if poly is not None:
                node = (poly, box, zz)
        nodes.append(node)
        report.append(None if node is None else dict(polygon=(poly+origin).tolist(), source=method))
    union = np.zeros(z.shape, bool)
    states, metrics, supports = [], [], []
    for j in range(13):
        k = 2*j
        pair, info = [], []
        for h, ids in enumerate(([k,k+1,k+3], [k,k+3,k+2])):
            ns = [nodes[i] for i in ids]
            code = face == k+h
            record = dict(box_pixels=0, interior_pixels=0, blocks=0)
            support = (np.array([], int), np.array([], int))
            if any(n is None for n in ns):
                # A coherent patch of unique face ink is positive presence,
                # not a way of inferring OPEN when a corner is missing.
                state = PRESENT if patch(code & depth_ok) else UNKNOWN
            else:
                polys, boxes, zz = zip(*ns)
                corners_uv = np.concatenate(boxes)
                x, y = np.maximum(np.floor(corners_uv.min(0)).astype(int)-1, 0)
                X, Y = np.minimum(np.ceil(corners_uv.max(0)).astype(int)+2, z.shape[::-1])
                roi = np.s_[y:Y, x:X]
                pts, depth = xy[roi], z[roi]
                region = _interior(polys, pts, footprint=True)
                old_region = _interior(boxes, pts, footprint=True)
                deep = blank[roi] & (depth > np.max(zz)+dz+.002)
                far = block(region & deep)
                fy, fx = np.where(far)
                support = (fy+y, fx+x)
                # Restrict possible contact to the convex hull of candidate
                # vertices; an unrelated pixel elsewhere cannot veto this face.
                try:
                    eq = ConvexHull(np.concatenate(polys)).equations
                    possible = np.all(pts @ eq[:, :2].T + eq[:, 2] <= .71, axis=2)
                except QhullError:
                    possible = np.zeros(depth.shape, bool)
                near = code[roi] & possible & depth_ok[roi] & (depth >= np.min(zz)-dz) & (depth <= np.max(zz)+dz)
                seen, through = patch(near), bool(len(fy))
                state = PRESENT if seen and not through else OPEN if through and not near.any() else UNKNOWN
                if state == OPEN:
                    nominal = [p.mean(0, keepdims=True) for p in polys]
                    union[roi] |= deep & _interior(nominal, pts, footprint=False)
                record.update(box_pixels=int(old_region.sum()), interior_pixels=int(region.sum()), blocks=len(fy))
            supports.append(support)
            pair.append(state)
            info.append(record)
        states.append(pair)
        metrics.append(info)

    cc, _ = ndimage.label(union)
    def connected(first, last):
        common = None
        for s in supports[2*first:2*last]:
            parts = set(np.unique(cc[s]))-{0}
            common = parts if common is None else common & parts
        return bool(common)

    found = [n for n in range(1, 11) if states[n-1] == [OPEN, OPEN]
             and UNKNOWN not in states[n] and PRESENT in states[n] and states[n+1] == [PRESENT, PRESENT]
             and not any(PRESENT in p for p in states[:n]) and not any(OPEN in p for p in states[n+1:])
             and connected(n-1, n)]
    local = found[0] if len(found) == 1 else None
    prefix = local if (local and all(p == [OPEN, OPEN] for p in states[:local]) and connected(0, local)) else None
    decision = UNKNOWN if prefix is None else 'STOP' if prefix >= 9 else 'BELOW'
    return dict(decision=decision, prefix=prefix, local_mm=local, states=states, faces=metrics, corners=report,
                reason='' if prefix is not None else 'prefix not proven' if local else 'no unique supported local front')
