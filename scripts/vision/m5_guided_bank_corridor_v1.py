"""Development-only connected corridor reader; RGB-D and ink identities only."""
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree

from cell_certificate_v1 import decode
from ruler_readout import inks


def bank_patch(mask):
    labels, n = ndimage.label(mask)
    if not n:
        return np.empty((0, 2), dtype=int)
    area = np.bincount(labels.ravel(), minlength=n+1)[1:]
    i = int(np.argmax(area))
    largest = area[i]
    if largest < 8 or np.count_nonzero(2*area >= largest) != 1:
        return np.empty((0, 2), dtype=int)
    return np.argwhere(labels == i+1)


def endpoint_depth(points, p, depth):
    close = points[np.max(abs(points-p), axis=1) <= 2]
    return None if len(close) < 3 else float(np.median(depth[tuple(close.T)]))


def read_row(j, bank, cap, face, depth, good, offset, sigma):
    rec = dict(row=j, status='UNKNOWN')
    left, right = [bank_patch((bank == k) & good) for k in (j, j+13)]
    if not len(left) or not len(right):
        return dict(rec, reason='missing or ambiguous bank patch'), None
    dist, nearest = cKDTree(right).query(left)
    i = int(np.argmin(dist))
    a, b = left[i], right[nearest[i]]
    lo, hi = endpoint_depth(left, a, depth), endpoint_depth(right, b, depth)
    rec.update(left_uv=(a+offset)[::-1].tolist(), right_uv=(b+offset)[::-1].tolist())
    if lo is None or hi is None:
        return dict(rec, reason='insufficient flank depth'), None
    count = int(np.max(abs(a-b)))+1
    line = np.rint(np.linspace(a, b, count)).astype(int)[3:-3]
    if len(line) < 4:
        return dict(rec, reason='insufficient trimmed chord'), None
    pos = tuple(line.T)
    z = depth[pos]
    valid = good[pos]
    blank = (bank[pos] < 0) & (cap[pos] < 0) & (face[pos] < 0)
    deep = valid & blank & (z > max(lo,hi)+.002+3*sigma)
    bound = .001+3*sigma
    intact = valid & np.isin(face[pos], (2*j,2*j+1)) & (z >= min(lo,hi)-bound) & (z <= max(lo,hi)+bound)
    status = 'OPEN' if deep.all() else 'INTACT' if intact.all() else 'UNKNOWN'
    return dict(rec, status=status, reason='' if status != 'UNKNOWN' else 'chord unresolved',
                samples=len(line), deep=int(deep.sum()), intact=int(intact.sum()),
                invalid=int((~valid).sum()), flank_depth_m=[lo,hi]), (a,b,line,max(lo,hi))


def connected(rows, blank, face, depth, good, sigma):
    support = np.zeros(depth.shape, dtype=bool)
    for a,b,line,ceil in rows:
        pos = tuple(line.T)
        support[pos] |= blank[pos] & good[pos] & (depth[pos] > ceil+.002+3*sigma)
    for (a,b,_,z0), (c,d,_,z1) in zip(rows,rows[1:]):
        corners = np.array([a,b,c,d])
        try:
            eq = ConvexHull(corners).equations
        except QhullError:
            return 'degenerate adjacent chord hull'
        y,x = corners.min(0)
        Y,X = corners.max(0)+1
        roi = np.s_[y:Y,x:X]
        pos = np.moveaxis(np.mgrid[y:Y,x:X], 0, -1)
        inside = (pos @ eq[:,:2].T + eq[:,2] <= 1e-8).all(-1)
        if (inside & (face[roi]>=0)).any():
            return 'central-face ink inside open corridor'
        support[roi] |= inside & blank[roi] & good[roi] & (depth[roi] > max(z0,z1)+.002+3*sigma)
    lab, _ = ndimage.label(support)
    common = None
    for _,_,line,_ in rows:
        on = set(np.unique(lab[tuple(line.T)]))-{0}
        common = on if common is None else common & on
    return '' if common else 'no shared through-depth component'


def measure(rgb, depth, *, far, sigma):
    result = dict(verdict='UNKNOWN', prefix_mm=None, rows=[], reason='invalid RGB-D')
    if (depth.ndim != 2 or rgb.shape != depth.shape+(3,) or rgb.dtype != np.uint8
            or depth.dtype.kind != 'f' or not (math.isfinite(far) and far > 0)
            or not (math.isfinite(sigma) and sigma >= 0)):
        return result
    bank = inks(rgb)
    y,x = np.where(bank>=0)
    if not len(y):
        return dict(result, reason='no bank ink')
    offset = np.array([y.min(),x.min()])
    roi = np.s_[y.min():y.max()+1,x.min():x.max()+1]
    cap, face = decode(rgb[roi])
    bank, depth = bank[roi], depth[roi]
    good = np.isfinite(depth) & (depth>0) & (depth<far*.999)
    evidence = []
    for j in range(13):
        row, samples = read_row(j,bank,cap,face,depth,good,offset,sigma)
        result['rows'].append(row)
        evidence.append(samples)
    states = [e['status'] for e in result['rows']]
    n = next((i for i, s in enumerate(states) if s != 'OPEN'), 13)
    if not 0 < n <= 10:
        return dict(result, reason='no eligible open prefix from row zero')
    if states[n:n+2] != ['INTACT','INTACT'] or 'OPEN' in states[n+2:]:
        return dict(result, reason='missing immediate intact boundary or later opening')
    blank = (bank<0) & (cap<0) & (face<0)
    why = connected(evidence[:n],blank,face,depth,good,sigma)
    if why:
        return dict(result, reason=why)
    return dict(result, verdict='BELOW' if n < 9 else 'TARGET', prefix_mm=n, reason='')
