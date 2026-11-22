"""Read only the immutable left/right inks and depth of an observed opening."""
import numpy as np
from scipy import ndimage


def inks(rgb):
    rgb = np.asarray(rgb, dtype=np.int16)
    red, green, blue = np.moveaxis(rgb, 2, 0)
    out = np.full(red.shape, -1, dtype=np.int8)
    for side, (high, low) in enumerate(((green, blue), (blue, green))):
        i = np.rint((red-32)/48).astype(int)
        k = np.rint((low-35)/46).astype(int)
        j = i+4*k
        match = ((abs(high-235) <= 6) & (abs(red-32-48*i) <= 6) & (abs(low-35-46*k) <= 6)
                 & (i >= 0) & (i < 4) & (k >= 0) & (j <= 12))
        out[match] = side*13+j[match]
    return out


def observe_ruler(rgb, depth):
    label = inks(rgb)
    areas = []
    for j in range(26):
        regions, n = ndimage.label(label == j, np.ones((3, 3), int))
        if not n:
            return dict(valid=False, reason='missing row or end', label=j)
        counts = np.bincount(regions.ravel())[1:]
        substantial = counts >= 4
        if not np.any(substantial):
            return dict(valid=False, reason='insufficient row ink', label=j)
        substantial = np.r_[False, substantial]
        areas.append(np.argwhere(substantial[regions])[:, ::-1])
    # Each step of a transect is bracketed by independently decoded material
    # pixels. No projective extrapolation from a row several millimetres away.
    med = np.array([np.median(a, axis=0) for a in areas]).reshape(2, 13, 2)
    along = np.mean(med[:, -1]-med[:, 0], axis=0)
    along /= max(1e-9, np.linalg.norm(along))
    if np.any(np.diff(med @ along, axis=1) <= 0):
        return dict(valid=False, reason='unordered printed ink')
    h, w = depth.shape
    for side in (0, 1):
        for end, sign in ((0, -1), (12, 1)):
            code = side*13+end
            uv = areas[code]
            adjacent = np.zeros(len(uv), bool)
            exterior = np.zeros(len(uv), bool)
            for off in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                xy = uv+off
                x, y = xy.T
                ok = (x > 0) & (x < w-1) & (y > 0) & (y < h-1)
                i = np.flatnonzero(ok)
                x, y = x[i], y[i]
                adjacent[i] |= label[y, x] == code-sign
                if np.dot(off, along)*sign > .3:
                    z = depth[uv[i, 1], uv[i, 0]]
                    exterior[i] |= (label[y, x] < 0) & (depth[y, x] > z+.002)
            if adjacent.sum() < 2 or exterior.sum() < 2:
                return dict(valid=False, reason='unverified visible end and adjacent cell', label=code)
    lat = np.array([-along[1], along[0]])
    if np.median((med[1]-med[0]) @ lat) < 0:
        lat = -lat
    # In each material cell only transects touching the *same* ordinal on both
    # flanks count. Foreground, tool, other ink and unknown depth are not gaps.
    rows = []
    for j in range(13):
        left, right = areas[j], areas[j+13]
        c = med[:, j].mean(axis=0)
        # Follow local row direction using the observed left and right codes.
        a = med[:, min(j+1, 12)].mean(axis=0)-med[:, max(j-1, 0)].mean(axis=0)
        a /= max(1e-9, np.linalg.norm(a))
        b = np.array([-a[1], a[0]])
        if b @ lat < 0:
            b *= -1
        la, ra = (left-c) @ a, (right-c) @ a
        lb, rb = (left-c) @ b, (right-c) @ b
        scans = []
        for pos in np.arange(np.ceil(max(la.min(), ra.min())), min(la.max(), ra.max())+.001):
            l, r = np.flatnonzero(abs(la-pos) <= .7), np.flatnonzero(abs(ra-pos) <= .7)
            if not len(l) or not len(r):
                continue
            l, r = left[l[np.argmax(lb[l])]], right[r[np.argmin(rb[r])]]
            mid = np.rint(np.linspace(l, r, int(np.ceil(np.linalg.norm(l-r)))+1)).astype(int)
            mid = np.unique(mid, axis=0)
            x, y = mid.T
            z0 = max(depth[l[1], l[0]], depth[r[1], r[0]])
            if not np.isfinite(z0) or z0 <= 0:
                continue
            marks, z = label[y, x], depth[y, x]
            other = (marks != j) & (marks != j+13)
            if not other.any():
                scans.append(('joined', float(pos), l.tolist(), r.tolist()))
            elif (np.all(marks[other] < 0) and np.all(np.isfinite(z[other]))
                  and np.all(z[other] > z0+.002)):
                scans.append(('open', float(pos), l.tolist(), r.tolist()))
        rows.append(dict(row=j, open=sum(s[0] == 'open' for s in scans),
                         joined=sum(s[0] == 'joined' for s in scans), scans=scans))
    return dict(valid=True, rows=rows)


def crossings(rgb, depth):
    """Read adjacent ordinal boundaries themselves, not a global plane fit."""
    lab = inks(rgb)
    nz = np.argwhere(lab >= 0)
    if not nz.size:
        return []
    lo = np.maximum(nz.min(axis=0)-2, 0)
    hi = np.minimum(nz.max(axis=0)+3, lab.shape)
    sy, sx = (slice(int(a), int(b)) for a, b in zip(lo, hi))
    lab, z = lab[sy, sx], depth[sy, sx]
    out = []
    for j in range(1, 13):
        tips, mids = [], []
        for side in (0, 1):
            low, high = lab == side*13+j-1, lab == side*13+j
            boundary = np.argwhere(high & ndimage.binary_dilation(low, np.ones((3, 3))))
            if len(boundary) < 4:
                break
            tips.append(boundary)
            mids.append(np.median(boundary, axis=0))
        if len(tips) != 2:
            out.append(dict(j=j, state='unknown'))
            continue
        across = mids[1]-mids[0]
        tips = [q[np.argmax(s*(q @ across))] for s, q in zip((1, -1), tips)]
        # Sample between the innermost visible ink endpoints from different banks.
        # Ignore only the two registered local row codes on each end, never other
        # material colours or foreground as possible evidence of an opening.
        a, b = tips
        n = int(np.ceil(np.linalg.norm(b-a)))+1
        y, x = np.unique(np.rint(np.linspace(a, b, n)).astype(int), axis=0).T
        local = np.isin(lab[y, x], [j-1, j, j+12, j+13])
        gap = ~local
        flank_z = max(z[tuple(a)], z[tuple(b)])
        clear = (lab[y, x] < 0) & np.isfinite(z[y, x]) & (z[y, x] > flank_z+.002)
        state = 'open' if gap.any() and np.all(clear[gap]) else 'joined' if local.all() else 'unknown'
        out.append(dict(j=j, state=state, open_pixels=int(clear.sum()), ends_yx=[(v+lo).tolist() for v in tips]))
    return out


def classify(rgb, depth):
    """Threshold observation only: BELOW never by itself clears a blade movement."""
    scan = observe_ruler(rgb, depth)
    if not scan['valid']:
        return dict(decision='UNKNOWN', reason=scan['reason'])
    rows = scan['rows']
    states = [r['state'] for r in crossings(rgb, depth)]
    opened = [r['open'] >= 2 for r in rows]
    joined = [r['joined'] >= 2 for r in rows]
    n = next((k for k, v in enumerate(opened) if not v), len(opened))
    result = dict(open_rows=[i for i, v in enumerate(opened) if v],
                  joined_rows=[i for i, v in enumerate(joined) if v], crossings=states)
    if not (0 < n < len(opened)-1 and not any(opened[n:]) and joined[n] and joined[n+1]):
        return dict(result, decision='UNKNOWN', reason='no continuous opening with intact tissue ahead')
    if not (all(s == 'open' for s in states[:n-1]) and states[n-1] == 'joined'
            and all(s != 'open' for s in states[n:])):
        return dict(result, decision='UNKNOWN', reason='ordinal transects do not bracket a unique front')
    return dict(result, decision='STOP' if n >= 9 else 'BELOW', last_open_cell=n-1)
