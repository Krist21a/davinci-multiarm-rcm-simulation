"""Denoised outer-bank work chart; no change to the narrow reader."""
import numpy as np

from m5_v3_perception_v4 import PINK, WorkChart, _good, _stations, inks
from m5_v3_perception_v4 import intact_preflight as intact_preflight


def _fit_rim_depth(label, depth, valid, pink, a, b, across, tol, sigma):
    center = (a+b)*.5
    lo = np.maximum(np.floor(center).astype(int)-5,0)
    hi = np.minimum(np.ceil(center).astype(int)+6,depth.shape)
    window = np.s_[lo[0]:hi[0],lo[1]:hi[1]]
    z = depth[window].astype(float)
    coords = np.moveaxis(np.indices(z.shape),0,-1)+lo-center
    along = np.array([-across[1],across[0]])
    allowed = (valid[window] & ((label[window]==label[tuple(a)]) | pink[window])
               & (((coords@across)/5.)**2+((coords@along)/2.)**2<=1.))
    start = tuple(a-lo)
    end = tuple(b-lo)
    reached = {start}
    stack = [start]
    while stack:
        y,x = stack.pop()
        for i,j in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            p = (i,j)
            if (0<=i<z.shape[0] and 0<=j<z.shape[1] and allowed[p]
                    and p not in reached and abs(z[p]-z[y,x])<=tol):
                reached.add(p)
                stack.append(p)
    if end not in reached or len(reached)<8:
        return None
    indices = tuple(np.asarray(sorted(reached)).T)
    zz = z[indices]
    design = np.column_stack((np.ones(len(zz)),coords[indices]))
    # The intercept weights evaluate the local plane on the actual half-pixel ray.
    pinv = np.linalg.pinv(design)
    weights = pinv[0]
    if not np.allclose(weights@design,[1.,0.,0.],rtol=0,atol=1e-10):
        return None
    leverage = float(weights@weights)
    if leverage>.25:
        return None
    inv = 1./zz
    mean = float(weights@(inv-sigma**2*inv**3))
    if not np.isfinite(mean) or mean<=0:
        return None
    fitted = 1./mean-leverage*sigma**2*mean
    return fitted if fitted>0 else None


def _direction(pairs, n):
    rows, centers = [], []
    for j in range(n+1,min(13,n+6)):
        if j not in pairs or not .0065<=np.linalg.norm(pairs[j][1]-pairs[j][0])<=.0115:
            break
        rows.append(j)
        centers.append(pairs[j].mean(0))
    if len(rows)<3:
        return None
    offsets = np.asarray(rows)-np.mean(rows)
    slope = offsets@np.asarray(centers)/(offsets@offsets)
    length = np.linalg.norm(slope)
    if not .00025<=length<=.0015:
        return None
    return slope/length, rows


def _rim(label, depth, valid, pink, side, across, tol, sigma):
    points = np.argwhere(valid & (label//13==side) & (label>=0))
    rows = [[] for _ in range(13)]
    for step in ((0,1),(0,-1),(1,0),(-1,0)):
        off = np.asarray(step)
        if float(off@across)*(1 if side else -1)<.25:
            continue
        out = points+off
        ok = ((out>=0)&(out<depth.shape)).all(1)
        a,b = points[ok],out[ok]
        i,k = tuple(a.T),tuple(b.T)
        keep = pink[k] & (abs(depth[i]-depth[k])<=tol)
        a,b = a[keep],b[keep]
        for p,q in zip(a,b):
            z = _fit_rim_depth(label,depth,valid,pink,p,q,across,tol,sigma)
            if z is not None:
                j = label[tuple(p)]%13
                rows[j].append((*((p+q)*.5),z))
    return [np.asarray(p).reshape(-1,3) for p in rows]


def work(rgb, depth, *, focal, far, sigma, prefix_mm, goal_mm):
    if (type(prefix_mm) is not int or goal_mm not in (9,10) or not 0<=prefix_mm<goal_mm
            or depth.ndim!=2 or rgb.dtype!=np.uint8 or rgb.shape!=depth.shape+(3,)
            or not np.isfinite([focal,far,sigma]).all() or min(focal,far)<=0 or sigma<0):
        return None
    n = prefix_mm
    label = inks(rgb)
    valid = _good(depth,far)
    pink = valid & (rgb==PINK).all(-1)
    pix = [np.argwhere(valid & (label>=0) & (label//13==s)) for s in range(2)]
    if not all(len(p) for p in pix):
        return None
    across = pix[1].mean(0)-pix[0].mean(0)
    if np.linalg.norm(across)<1e-6:
        return None
    across /= np.linalg.norm(across)
    tol = .001+3*sigma
    sides = [_stations(_rim(label,depth,valid,pink,s,across,tol,sigma),depth.shape,focal,tol) for s in range(2)]
    pairs = {j:np.stack((sides[0][j],sides[1][j])) for j in sides[0].keys() & sides[1].keys()}
    if not {n+1,n+2}<=pairs.keys():
        return None
    first,second = pairs[n+1].mean(0),pairs[n+2].mean(0)
    step = np.linalg.norm(second-first)
    if not .00025<=step<=.0015:
        return None
    axis = _direction(pairs,n)
    if axis is None:
        return None
    direction,fit_rows = axis
    checked = {}
    for j,p in pairs.items():
        chord = p[1]-p[0]
        if .0065<=np.linalg.norm(chord)<=.0115 and abs(chord@direction)<=.00075:
            checked[j] = p
    if not set(fit_rows)<=checked.keys():
        return None
    opening = (checked[n+1][1]-checked[n+1][0]+checked[n+2][1]-checked[n+2][0])*.5
    opening -= (opening@direction)*direction
    if np.linalg.norm(opening)<.0065:
        return None
    if n<=5:
        bound = max((j for j in checked if n+2<=j<=8), default=None)
        if bound is None:
            return None
        aim_row = limit_row = bound
    elif {goal_mm,10}<=checked.keys():
        aim_row,limit_row = goal_mm,10
    else:
        return None
    axial = float((checked[n].mean(0)-first)@direction) if n in checked else -step
    aim = float(np.min((checked[aim_row]-first)@direction))
    limit = float(np.min((checked[limit_row]-first)@direction))
    if not -.0015<=axial<=-.00025 or not axial<aim<=limit:
        return None
    return WorkChart(n,first+axial*direction,direction,opening/np.linalg.norm(opening),
                     first+aim*direction,first+limit*direction)
