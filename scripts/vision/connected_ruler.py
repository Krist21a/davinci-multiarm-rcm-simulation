"""RGB-D development observations; neither a local front nor a prefix authorizes a bite."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from ruler_readout import crossings, inks, observe_ruler


def classify(rgb, depth):
    unknown = dict(decision='UNKNOWN')
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if depth.ndim != 2 or rgb.shape != depth.shape+(3,):
        return dict(unknown, reason='invalid sensor shape')
    label = inks(rgb)
    y, x = np.where(label >= 0)
    if not len(y):
        return dict(unknown, reason='no material codes')
    roi = (slice(max(0,y.min()-8),y.max()+9), slice(max(0,x.min()-8),x.max()+9))
    rgb, depth, label = rgb[roi], depth[roi], label[roi]
    scan = observe_ruler(rgb, depth)
    if not scan['valid']:
        return dict(unknown, reason=scan['reason'])
    states = [r['state'] for r in crossings(rgb, depth)]
    if len(states) != 12:
        return dict(unknown, reason='missing ordinal transects')
    visible = np.zeros(label.shape, bool)
    edges = []
    for j in range(13):
        sides = [label == code for code in (j, j+13)]
        xy = np.argwhere(sides[0] | sides[1])[:, ::-1]
        try:
            hull = ConvexHull(xy)
        except QhullError:
            return dict(unknown, reason='degenerate ordinal window')
        window = Image.new('1', label.shape[::-1])
        ImageDraw.Draw(window).polygon([tuple(p) for p in xy[hull.vertices]], fill=1)
        banks = [depth[side] for side in sides]
        if any(not np.isfinite(z).all() or (z <= 0).any() for z in banks):
            return dict(unknown, reason='invalid flank depth')
        z0 = max(np.quantile(z, .9) for z in banks)
        gap = (label < 0) & np.asarray(window) & np.isfinite(depth) & (depth > z0+.002)
        edges.append([gap & ndimage.binary_dilation(s, np.ones((3, 3), bool)) for s in sides])
        visible |= gap
    comps, count = ndimage.label(visible)  # four-connected; diagonal or hidden pixels cannot bridge
    contact = np.zeros((count+1, 13, 2), dtype=int)
    for j, sides in enumerate(edges):
        for s, side in enumerate(sides):
            contact[:,j,s] = np.bincount(comps[side], minlength=count+1)
    # states[i] samples the high side of boundary i+1. The first joined
    # boundary is the first intact row, not the number of open boundaries.
    n = next((j for j, state in enumerate(states, 1) if state != 'open'), 13)
    extents = [next((j for j, pair in enumerate(row) if min(pair) < 2), 13) for row in contact[1:]]
    best = int(np.argmax(extents))+1 if extents else 0
    criteria = dict(crossings=states, bilateral_prefix_rows=max(extents, default=0),
                    both_bank_pixels=contact[best].tolist())
    if not (1 <= n <= 10 and states[n-1] == 'joined' and 'open' not in states[n:]
            and all(row['open'] >= 2 and not row['joined'] for row in scan['rows'][:n])
            and all(row['joined'] >= 2 for row in scan['rows'][n:n+2])
            and not any(row['open'] for row in scan['rows'][n:])):
        return dict(unknown, **criteria, reason='no uninterrupted crossed ordinals with intact tissue ahead')
    if not any(k >= n for k in extents):
        return dict(unknown, **criteria, reason='no single observed slit touches both banks of preceding rows')
    return dict(criteria, decision='STOP' if n >= 9 else 'BELOW', open_boundary=n)


def _local_row(label, depth, j):
    """Inspect both material boundaries on this row's ink, not on the next row."""
    sides = [label == k for k in (j, j+13)]
    hood = np.ones((3, 3), bool)
    ends = []
    for delta in (-1, 1):
        clouds = [np.argwhere(s & ndimage.binary_dilation(label == k+delta, hood))
                  for s, k in zip(sides, (j, j+13))]
        if any(len(q) < 4 for q in clouds):
            return 'UNKNOWN'
        across = np.median(clouds[1], axis=0) - np.median(clouds[0], axis=0)
        if not across.any():
            return 'UNKNOWN'
        ends.append([q[np.argmax(sign*(q @ across))] for q, sign in zip(clouds, (1, -1))])
    lines, anchors = [], []
    for a, b in ends:
        y, x = np.unique(np.rint(np.linspace(a, b, int(np.ceil(np.linalg.norm(b-a)))+1))
                         .astype(int), axis=0).T
        anchors.extend((depth[tuple(a)], depth[tuple(b)]))
        lines.append((y, x))
    if not np.isfinite(anchors).all() or min(anchors) <= 0:
        return 'UNKNOWN'
    flanks = sides[0] | sides[1]
    zmax, zmin = max(anchors), min(anchors)
    if all(flanks[y, x].all() and np.all(np.isfinite(depth[y, x]) & (depth[y, x] > 0)
           & (depth[y, x] >= zmin-.002) & (depth[y, x] <= zmax+.002)) for y, x in lines):
        return 'INTACT'
    if np.any(sides[0] & ndimage.binary_dilation(sides[1], hood)):
        return 'UNKNOWN'
    # Confine the connecting path to the quadrilateral between the four
    # observed endpoints. Exterior background must not join the transects.
    quad = np.array([ends[0][0], ends[0][1], ends[1][1], ends[1][0]])
    if len(np.unique(quad, axis=0)) != 4:
        return 'UNKNOWN'
    window = Image.new('1', label.shape[::-1])
    ImageDraw.Draw(window).polygon([tuple(p[::-1]) for p in quad], fill=1)
    deep = (np.asarray(window) & (label < 0) & np.isfinite(depth) & (depth > zmax+.002))
    components, _ = ndimage.label(deep)
    shared = None
    for y, x in lines:
        gap = ~flanks[y, x]
        if gap.sum() < 2 or not deep[y[gap], x[gap]].all():
            return 'UNKNOWN'
        ids = np.unique(components[y[gap], x[gap]])
        if len(ids) != 1 or not ids[0] or (shared is not None and ids[0] != shared):
            return 'UNKNOWN'
        shared = ids[0]
    return 'OPEN'


def local_front(rgb, depth):
    """Local ordinal candidate only. No extrapolation or assertion about hidden prior rows."""
    rgb, z = np.asarray(rgb), np.asarray(depth)
    if z.ndim != 2 or rgb.shape != z.shape + (3,):
        return dict(prefix_mm=None, reason='invalid sensor shape')
    label = inks(rgb)
    ix = np.argwhere(label >= 0)
    if not len(ix):
        return dict(prefix_mm=None, reason='no material codes')
    lo, hi = ix.min(0), ix.max(0)+1
    roi = np.s_[lo[0]:hi[0], lo[1]:hi[1]]
    # Row 0 and row 12 lack one neighbouring ordinal. No silhouette estimate.
    states = ['UNKNOWN', *(_local_row(label[roi], z[roi], j) for j in range(1, 12)), 'UNKNOWN']
    fronts = [j for j in range(2, 12) if states[j-1:j+1] == ['OPEN', 'INTACT']]
    n = fronts[0] if len(fronts) == 1 else None
    if n is None or n > 10 or 'INTACT' in states[:n] or 'OPEN' in states[n:]:
        return dict(prefix_mm=None, rows=states, reason='no unique supported local front in budget')
    return dict(prefix_mm=n, rows=states, reason='')
