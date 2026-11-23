"""Local RGB-D geometry of the immutable 1 mm sided surface print. Camera coordinates, metres."""
import numpy as np
from scipy import ndimage

from ruler_readout import inks


_HOOD = np.ones((3, 3), bool)
_GAP = .002
_PITCH = .001
_LIMIT_ROW = 10
_Z_ALLOWANCE = .00025 + 3*.0005


def _unit(v):
    n = np.linalg.norm(v)
    return v/n if np.isfinite(n) and n > 1e-10 else None


def _ray(uv, principal, focal):
    return np.r_[(principal-uv)/focal, 1.]


def _bank_plane(label, z, codes, end, pp, f, far, depth):
    """Fit inverse range in one local material neighbourhood; never fit across a slit."""
    yy, xx = np.where(np.isin(label,codes) & np.isfinite(z) & (z > 0) & (z < far*.999))
    uv = np.column_stack([xx+.5, yy+.5])-end
    near = np.linalg.norm(uv, axis=1) <= .0015*f/depth
    uv, zz = uv[near], z[yy[near], xx[near]]
    if len(zz)<6:
        return None
    X = np.column_stack([uv, np.ones(len(zz))])
    keep = np.ones(len(zz), bool)
    for _ in range(2):
        if keep.sum()<6 or not np.any(np.linalg.svd(uv[keep]-uv[keep].mean(0),compute_uv=False)[-1:] > 1):
            return None
        slope, _, rank, _ = np.linalg.lstsq(X[keep],1/zz[keep],rcond=None)
        if rank!=3 or slope[2]<=0:
            return None
        pred = X @ slope
        if np.min(pred)<=0:
            return None
        keep = abs(zz-1/pred) <= _Z_ALLOWANCE
    if keep.sum()<.8*len(zz):
        return None
    # 1/Z = a*(u-u0)+b*(v-v0)+c implies N.P=1 for this camera convention.
    a,b,c = slope
    n = _unit(np.array([-f*a,-f*b,c+a*(pp[0]-end[0])+b*(pp[1]-end[1])]))
    if n is None or abs(1/c-depth) > _GAP:
        return None
    return 1/c, -n


def _boundary(label, z, j, side, towards, pp, f, far):
    """Interpolate across actual pairs of neighbouring ordinal inks on ONE bank."""
    a, b = 13*side+j-1, 13*side+j
    samples, zs = [], []
    for dy, dx in ((0, 1), (1, 0)):
        p, q = np.s_[:z.shape[0]-dy, :z.shape[1]-dx], np.s_[dy:, dx:]
        contact = ((label[p] == a) & (label[q] == b)) | ((label[p] == b) & (label[q] == a))
        contact &= ((z[p] > 0) & (z[q] > 0) & (z[p] < far*.999) & (z[q] < far*.999)
                    & (abs(z[p]-z[q]) <= _GAP))
        yy, xx = np.where(contact)
        samples.extend(np.column_stack((xx+.5+dx*.5, yy+.5+dy*.5)))
        # Linear interpolation of inverse optical depth in image space.
        zs.extend(2/(1/z[p][contact]+1/z[q][contact]))
    if len(samples) < 4:
        return None
    uv, zz = np.asarray(samples), np.asarray(zs)
    center = np.median(uv, axis=0)
    _, sv, axes = np.linalg.svd(uv-center, full_matrices=False)
    if sv[0] < 2 or sv[1] >= .5*sv[0]:
        return None
    axis = axes[0] if axes[0] @ towards > 0 else -axes[0]
    s = (uv-center) @ axis
    near = (s >= s.max()-5) & (abs((uv-center) @ axes[1]) <= 1.5)
    if near.sum() < 4 or np.max(np.diff(np.sort(s[near])), initial=0) > 2:
        return None
    end = uv[np.argmax(s)]
    t = s[near]-s.max()
    # The end is not a depth return from the slit. Fit local bank range on
    # the observed short segment, not a plane shared by the opposite banks.
    design = np.column_stack([t, np.ones(len(t))])
    fit, _, rank, _ = np.linalg.lstsq(design, 1/zz[near], rcond=None)
    if rank < 2 or fit[1] <= 0:
        return None
    depth = 1/fit[1]
    if not 0 < depth < far*.999 or np.max(abs(zz[near]-depth)) > .003:
        return None
    sheet = _bank_plane(label,z,[a,b],end,pp,f,far,depth)
    if sheet is None:
        return None
    return end, _ray(end, pp, f)*sheet[0], sheet[1]


def _triangle(uv, xyz, code, label, z, far):
    """Use the same pixel-centre barycentric test for support and connectivity."""
    x0, y0 = np.maximum(0, np.floor(uv.min(0)).astype(int))
    x1, y1 = np.minimum(z.shape[::-1], np.ceil(uv.max(0)).astype(int)+1)
    shape = (y1-y0, x1-x0)
    edges = (uv[1:]-uv[0]).T
    if abs(np.linalg.det(edges)) < 1e-8 or min(shape) <= 0:
        return '?', None
    yy, xx = np.indices(shape)
    xy = np.stack((xx+x0+.5, yy+y0+.5), axis=-1)
    ab = (xy-uv[0]) @ np.linalg.inv(edges).T
    w = np.dstack((1-ab.sum(axis=2), ab))
    inside = (w > 1e-7).all(axis=2)  # a shared diagonal is not proof of either face
    roi = np.s_[y0:y1, x0:x1]
    d, lab = z[roi], label[roi]
    expected = np.zeros(shape)
    expected[inside] = 1/(w[inside] @ (1/xyz[:, 2]))
    finite = inside & (d > 0) & (d < far*.999) & np.isfinite(d)
    deep = finite & (lab < 0) & (d > expected+_GAP)
    surface = finite & (abs(d-expected) < _GAP) & (lab == code)
    def component(mask):
        cc, count = ndimage.label(mask)
        if not count:
            return np.zeros(shape, bool)
        sizes = np.bincount(cc.ravel())[1:]
        return np.r_[False, sizes >= 2][cc]
    gap, ink = component(deep), component(surface)
    # Foreground, an adjacent material label, or a near-depth return inside an
    # otherwise deep face is not evidence for separation. Do not bridge it.
    if gap.any() and not (inside & ~deep).any():
        y, x = np.where(gap)
        return 'gap', (y+y0, x+x0)
    if ink.any() and not deep.any():
        return 'ink', None
    return '?', None


def _lip(label, z, row, side, knots, support, pp, focal, far):
    """Independently map a measured near-side lip back into its own ordinal interval."""
    lo, hi = knots[row][1][side], knots[row+1][1][side]
    uv = knots[row][0][side]
    axis = hi-lo
    scale = np.dot(axis, axis)
    if scale <= 1e-12:
        return None
    a = knots[row+1][0][side]-uv
    span = np.linalg.norm(a)
    if span < 1:
        return None
    yy, xx = np.where((label == 13*side+row) & ndimage.binary_dilation(support, _HOOD)
                      & (z > 0) & (z < far*.999))
    if not len(xx):
        return None
    pix = np.column_stack([xx+.5, yy+.5])
    # Choose the most distal contact using only the observed aperture on this
    # row. Project its surface ray to the *local bank segment*, not the far wall.
    frac = (pix-uv) @ a/(span*span)
    j = np.argmax(frac)
    t = np.clip(frac[j], 0., 1.)
    dep = 1/((1-t)/lo[2]+t/hi[2])
    if abs(z[yy[j], xx[j]]-dep) > _GAP:
        return None
    point = _ray(uv+t*a, pp, focal)*dep
    u = (point-lo) @ axis/scale
    # The last material pixel's footprint, not its centre, may touch the knot.
    if not (.5 <= u <= 1.+1e-8 and abs(1-frac[j])*span <= np.sqrt(2)+.5):
        return None
    return point, (row+u)*_PITCH


def _entry(label, z, towards, pp, f, far):
    """Both banks must meet at every printed row, and the free rim must see background."""
    for j in range(13):
        left, right = label == j, label == j+13
        touch = left & ndimage.binary_dilation(right, _HOOD)
        if np.count_nonzero(touch & (z > 0) & (z < far*.999)) < 2:
            return None
    along = np.median(np.argwhere(label == 1), axis=0)[::-1]-np.median(np.argwhere(label == 0), axis=0)[::-1]
    across = _unit(towards)
    if across is None:
        return None
    along = _unit(along-across*(along @ across))
    if along is None:
        return None
    rims = []
    for side in (0, 1):
        yy, xx = np.where(label == 13*side)
        points = []
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            if np.array([dx, dy]) @ along >= -.25:
                continue
            y, x = yy+dy, xx+dx
            ok = (x >= 0) & (y >= 0) & (x < z.shape[1]) & (y < z.shape[0])
            y, x, oy, ox = y[ok], x[ok], yy[ok], xx[ok]
            d, bg = z[oy, ox], z[y, x]
            hit = (label[y,x] < 0) & (d > 0) & (bg < far*.999) & (bg > d+_GAP)
            points.extend(zip(ox[hit]+.5+.5*dx, oy[hit]+.5+.5*dy, d[hit]))
        if len(points) < 4:
            return None
        rims.append(np.array(points))
    # The central material columns are symmetric about the middle of this
    # nine-cell-wide print. Only accept an observed continuous exterior rim.
    rim = np.vstack(rims)
    s = rim[:, :2] @ across
    if np.max(np.diff(np.sort(s)), initial=0) > 2:
        return None
    mid = (s.min()+s.max())*.5
    # At the first row the printed diagonal meets the rim at a vertex, not
    # at the centre of the 1-mm cell. The midpoint of the two *outer* rim
    # ends gives the material centre on an intact, fully visible 9-cell print.
    left,right = rim[s.argmin()],rim[s.argmax()]
    if not (any(np.array_equal(left,r) for r in rims[0]) and any(np.array_equal(right,r) for r in rims[1])):
        return None
    xy = .5*(left[:2]+right[:2])
    k = np.argmin(abs(s-mid))
    dep = rim[k,2]
    code = 0 if np.linalg.norm(xy-rims[0][:,:2],axis=1).min() < np.linalg.norm(xy-rims[1][:,:2],axis=1).min() else 13
    plane = _bank_plane(label,z,[code],xy,pp,f,far,dep)
    return _ray(xy,pp,f)*plane[0] if plane is not None else None


def measure(rgb, depth, width, height, focal, far, *, boundary=_boundary, record_knots=False):
    """Local bilateral surface measure. `endpoints` (record_knots only) are diagnostics in
    camera/pixel coordinates, even when the caller transports the result to world space."""
    z = np.asarray(depth)
    miss = dict(valid=False, front_valid=False, progress_m=None, banks_visible=False,
                sweep_limit_from_front_m=None, mode='bilateral_unknown')
    if (z.shape != (height, width) or rgb.shape != (*z.shape, 3)
            or not np.isfinite([focal, far]).all() or min(focal, far) <= 0):
        return dict(miss, reason='invalid sensor')
    label = inks(rgb)
    ys, xs = np.where(label >= 0)
    if not len(xs):
        return dict(miss, reason='no sided print')
    y, x = max(0, ys.min()-2), max(0, xs.min()-2)
    roi = np.s_[y:ys.max()+3, x:xs.max()+3]
    label, z = label[roi], z[roi]
    pp = np.array([width*.5-x, height*.5-y])
    knots = {}
    across = {}
    endpoints = {}
    for j in range(1, 13):
        banks = [np.argwhere((label == s*13+j) | (label == s*13+j-1))[:, ::-1] for s in (0, 1)]
        if min(map(len, banks)) < 4:
            continue
        delta = np.median(banks[1], axis=0)-np.median(banks[0], axis=0)
        ends = [boundary(label, z, j, s, (1-2*s)*delta, pp, focal, far) for s in (0, 1)]
        if record_knots:
            endpoints[str(j)] = [None if e is None else dict(uv=(e[0]+[x,y]).tolist(), camera=e[1].tolist()) for e in ends]
        if all(v is not None for v in ends):
            knots[j] = tuple(np.stack(a) for a in zip(*ends))
            across[j] = delta
    states, supports = [['?', '?'] for _ in range(13)], {}
    for j in range(1, 12):
        if j not in knots or j+1 not in knots:
            continue
        uv, xyz = (np.vstack((w, v)) for w, v in zip(knots[j][:2], knots[j+1][:2]))
        for h, ids in enumerate(([0,1,3], [0,3,2])):
            state, pixels = _triangle(uv[ids], xyz[ids], j+(13 if h == 0 else 0), label, z, far)
            states[j][h] = state
            if pixels is not None:
                supports[j,h] = pixels
    report = dict(miss, states=states, supported_knots=sorted(knots), banks_visible=bool(knots))
    if record_knots: report['endpoints'] = endpoints
    found = []
    for j in range(2, 12):
        if (states[j-1] != ['gap','gap'] or 'ink' not in states[j] or '?' in states[j]
                or j+1 >= len(states) or states[j+1] != ['ink','ink']
                or any('ink' in q for q in states[:j]) or any('gap' in q for q in states[j+1:])):
            continue
        # Two independent deep samples, one in each triangle, must belong to
        # one connected aperture. Permit shared-edge pixels only if actually deep.
        points = np.vstack([np.column_stack(supports[j-1,h]) for h in (0,1)])
        y0,x0 = points.min(0)
        y1,x1 = points.max(0)+1
        sl = np.s_[y0:y1,x0:x1]
        d = z[sl]
        back = max(knots[k][1][:,2].max() for k in (j-1,j))
        # Restrict any joining pixel to one neighbour of an actual interior witness.
        witness = np.zeros(d.shape, bool)
        witness[points[:,0]-y0,points[:,1]-x0] = True
        links = (ndimage.binary_dilation(witness) & (label[sl] < 0) & (d < far*.999) & (d > back+_GAP))
        cc, _ = ndimage.label(links)
        parts = [set(cc[sy-y0,sx-x0])-{0} for sy,sx in (supports[j-1,0],supports[j-1,1])]
        if parts[0] & parts[1]:
            found.append(j)
    entry = None
    if not found and 1 in knots and all(p == ['ink','ink'] for p in states[1:12]):
        entry = _entry(label, z, across[1], pp, focal, far)
    if entry is None and (len(found) != 1 or found[0] > _LIMIT_ROW):
        return dict(report, reason='no unique supported in-budget transition')
    n = found[0] if entry is None else 0
    # The material-coordinate reference is the last *fully* separated row,
    # not the farthest dark pixel of the next, potentially half-open triangle.
    bank_m = [0., 0.]
    if n:
        aperture = np.zeros(label.shape, bool)
        for half in (0,1):
            aperture[supports[n-1,half]] = True
        lips = [_lip(label, z, n-1, s, knots, aperture, pp, focal, far) for s in (0,1)]
        if any(v is None for v in lips):
            return dict(report, reason='terminal gap lacks bilateral registered lip contact', candidate_row=n)
        front = np.mean([p for p, _ in lips], axis=0)
        bank_m = [float(m) for _, m in lips]
    else:
        front = entry
    end = min(n+3, 12)
    if any(k not in knots for k in range(n+1,end+1)):
        return dict(report, reason='incomplete forward surface geometry')
    ahead = knots[end][1]
    normals = knots[n or 1][2]
    normal = _unit(normals.mean(0))
    if normal is None or normals[0] @ normals[1] <= 0:
        return dict(report, reason='no local surface normal')
    chord = ahead.mean(0)-front
    direction = _unit(chord-normal*(chord @ normal))
    if direction is None:
        return dict(report, reason='no local tangent')
    if normal @ front > 0:
        normal = -normal
    guard = None
    if all(k in knots for k in range(max(n,1), _LIMIT_ROW+1)):
        track = np.array([(knots[k][1]-front) @ direction for k in range(max(n,1), _LIMIT_ROW+1)])
        if np.all(np.diff(track, axis=0) > 0):
            bound = knots[_LIMIT_ROW][1]
            ray_scale = np.abs((bound/bound[:, 2:]) @ direction)
            # Lower endpoint from the nearer bank, with range and pixel-footprint reserve.
            guard = float(np.min(track[-1]-_Z_ALLOWANCE*ray_scale-np.sqrt(2)*bound[:,2]/focal))
    return dict(report, valid=True, front_valid=True, mode='bilateral_slit' if n else 'bilateral_entry',
                bank_progress_m=bank_m,
                progress_m=float(n*_PITCH), front=front.tolist(), direction=direction.tolist(),
                normal=normal.tolist(), sweep_limit_from_front_m=guard, reason='')
