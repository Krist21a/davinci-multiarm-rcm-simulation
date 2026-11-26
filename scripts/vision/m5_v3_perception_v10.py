"""Anisotropic rim depth with a shared three-pixel corner witness."""
import numpy as np

from m5_v3_perception_v8 import PINK, WorkChart, _direction, _good, _rim, _stations, inks
from m5_v3_perception_v8 import intact_preflight as intact_preflight


def _fit_corner_depth(label, depth, valid, pink, a, b, w, center, side, across, tol, sigma, pairs):
    r_across,r_along = 10.5,2.35
    along = np.array([-across[1],across[0]])
    extent = np.hypot(r_across*across,r_along*along)
    lo = np.maximum(np.ceil(center-extent).astype(int),0)
    hi = np.minimum(np.floor(center+extent).astype(int)+1,depth.shape)
    window = np.s_[lo[0]:hi[0],lo[1]:hi[1]]
    z = depth[window].astype(float)
    basis = np.column_stack((across/r_across,along/r_along))
    coords = (np.moveaxis(np.indices(z.shape),0,-1)+lo-center)@basis
    codes = label[window]
    allowed = (valid[window] & (pink[window] | ((codes>=0) & (codes//13==side)))
               & (np.sum(coords**2,axis=-1)<=1.))
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
    if tuple(w-lo) not in reached:
        return None
    if sum(tuple(p-lo) in reached and tuple(q-lo) in reached for p,q in pairs)<4:
        return None
    indices = tuple(np.asarray(sorted(reached)).T)
    zz = z[indices]
    y,x = coords[indices].T
    design = np.column_stack((np.ones(len(zz)),y,x,x*x))
    # Minimum-norm weights reproduce the polynomial intercept on the witnessed ray.
    target = np.array([1.,0.,0.,0.])
    weights,_,rank,_ = np.linalg.lstsq(design.T,target,rcond=None)
    if rank!=len(target) or not np.allclose(weights@design,target,rtol=0,atol=1e-10):
        return None
    if weights@weights>.125:
        return None
    inv = 1./zz
    mean = float(weights@(inv-sigma**2*inv**3))
    if not np.isfinite(mean) or mean<=0:
        return None
    variance = sigma**2*((weights**2)@(inv**4))
    fitted = 1./mean-variance/mean**3
    return fitted if np.isfinite(fitted) and fitted>0 else None


def _fallback_station(label,depth,valid,pink,side,across,tol,sigma,focal,j):
    if not 1<=j<=12:
        return None
    prev = np.argwhere(valid & (label==side*13+j-1))
    pairs = []
    for step in ((0,1),(0,-1),(1,0),(-1,0)):
        out = prev+step
        ok = ((out>=0)&(out<depth.shape)).all(1)
        a,b = prev[ok],out[ok]
        i,k = tuple(a.T),tuple(b.T)
        keep = (valid[k] & (label[k]==side*13+j) & (abs(depth[i]-depth[k])<=tol))
        pairs.extend(zip(a[keep],b[keep]))
    if len(pairs)<4:
        return None
    outward = across*(1 if side else -1)
    candidates = []
    for a,b in pairs:
        center = (a+b)*.5
        lo = np.maximum(a-2,0)
        hi = np.minimum(a+3,depth.shape)
        near = np.argwhere(pink[lo[0]:hi[0],lo[1]:hi[1]])+lo
        disp = near-center
        z = depth[tuple(near.T)]
        shared = ((np.max(abs(near-a),axis=1)==1) & (np.max(abs(near-b),axis=1)==1))
        keep = (shared & (np.sum(disp**2,axis=1)<=4.) & (disp@outward>=.25)
                & (abs(z-depth[tuple(a)])<=tol) & (abs(z-depth[tuple(b)])<=tol))
        for w in near[keep]:
            candidates.append((float(center@outward),a,b,w))
    if not candidates:
        return None
    outer = max(p[0] for p in candidates)
    xyz = []
    for score,a,b,w in candidates:
        if score!=outer:
            continue
        p,q = (a+w)*.5,(b+w)*.5
        za = _fit_corner_depth(label,depth,valid,pink,a,b,w,p,side,across,tol,sigma,pairs)
        zb = _fit_corner_depth(label,depth,valid,pink,a,b,w,q,side,across,tol,sigma,pairs)
        if za is None or zb is None or abs(za-zb)>tol:
            return None
        center = (p+q)*.5
        z = 2./(1./za+1./zb)
        rc = (np.array(depth.shape)*.5-center-.5)*z/focal
        xyz.append((rc[1],rc[0],z))
    points = np.asarray(xyz)
    median = np.median(points,axis=0)
    return median if np.max(np.linalg.norm(points-median,axis=1))<=.00035 else None


def _observe(rgb, depth, *, focal, far, sigma, prefix_mm, goal_mm):
    if (type(prefix_mm) is not int or goal_mm not in (9,10) or not 0<=prefix_mm<goal_mm
            or depth.ndim!=2 or rgb.dtype!=np.uint8 or rgb.shape!=depth.shape+(3,)
            or not np.isfinite([focal,far,sigma]).all() or min(focal,far)<=0 or sigma<0):
        return None
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
    return sides,(label,depth,valid,pink,across,tol,sigma,focal)


def _chart(sides,n,goal_mm):
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
    direction,checked = axis
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
    chart = WorkChart(n,first+axial*direction,direction,opening/np.linalg.norm(opening),
                      first+aim*direction,first+limit*direction)
    return chart,limit_row,checked[limit_row],checked


def _solve_work(rgb,depth,*,focal,far,sigma,prefix_mm,goal_mm):
    observed = _observe(rgb,depth,focal=focal,far=far,sigma=sigma,
                        prefix_mm=prefix_mm,goal_mm=goal_mm)
    if observed is None:
        return None,None,()
    sides,ctx = observed
    n = prefix_mm
    chart = _chart(sides,n,goal_mm)
    if chart is not None or n<6:
        return chart,sides,()
    required = {n+1,n+2,goal_mm,10}
    if all(required<=points.keys() for points in sides):
        return None,sides,()
    recovered = []
    label,z,valid,pink,across,tol,sigma,focal = ctx
    for side,points in enumerate(sides):
        for j in range(1,13):
            if j in points:
                continue
            p = _fallback_station(label,z,valid,pink,side,across,tol,sigma,focal,j)
            if p is not None:
                points[j] = p
                recovered.append((side,j))
    return _chart(sides,n,goal_mm),sides,tuple(recovered)


def _work_evidence(rgb,depth,*,focal,far,sigma,prefix_mm,goal_mm):
    solution,_,_ = _solve_work(rgb,depth,focal=focal,far=far,sigma=sigma,
                               prefix_mm=prefix_mm,goal_mm=goal_mm)
    return None if solution is None else solution[:3]


def work(rgb, depth, *, focal, far, sigma, prefix_mm, goal_mm):
    evidence = _work_evidence(rgb,depth,focal=focal,far=far,sigma=sigma,
                              prefix_mm=prefix_mm,goal_mm=goal_mm)
    return None if evidence is None else evidence[0]
