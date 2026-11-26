"""Planar perspective depth on v11's rim and corner supports (hinge work optics only)."""
import numpy as np

import m5_v3_perception_v11 as v11
from m5_v4_hinge_work_v1 import inverse, margin


def plane_intercept(coords,z,sigma):
    design = np.column_stack((np.ones(len(z)),coords))
    target = [1.,0.,0.]
    weights,_,rank,_ = np.linalg.lstsq(design.T,target,rcond=None)
    if (rank!=3 or not np.allclose(weights@design,target,rtol=0,atol=1e-10) or weights@weights>.125):
        return None
    inv = 1./z
    mean = float(weights@(inv-sigma**2*inv**3))
    if not np.isfinite(mean) or mean<=0:
        return None
    variance = sigma**2*((weights**2)@(inv**4))
    fitted = 1./mean-variance/mean**3
    return fitted if np.isfinite(fitted) and fitted>0 else None


def fit(ctx,a,b,center,side,corner=None,pairs=None):
    label,depth,valid,pink,across,tol,sigma,_ = ctx
    along = np.array([-across[1],across[0]])
    extent = np.hypot(10.5*across,2.35*along)
    lo = np.maximum(np.ceil(center-extent).astype(int),0)
    hi = np.minimum(np.floor(center+extent).astype(int)+1,depth.shape)
    window = np.s_[lo[0]:hi[0],lo[1]:hi[1]]
    z = depth[window].astype(float)
    coords = (np.moveaxis(np.indices(z.shape),0,-1)+lo-center)@np.column_stack((across/10.5,along/2.35))
    codes = label[window]
    allowed = valid[window] & (pink[window]|((codes>=0)&(codes//13==side))) & ((coords**2).sum(-1)<=1.)
    reached,stack = {tuple(a-lo)},[tuple(a-lo)]
    while stack:
        y,x = stack.pop()
        for i,j in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if (0<=i<z.shape[0] and 0<=j<z.shape[1] and allowed[i,j] and (i,j) not in reached
                    and abs(z[i,j]-z[y,x])<=tol):
                reached.add((i,j))
                stack.append((i,j))
    if tuple(b-lo) not in reached or len(reached)<8:
        return None
    if corner is not None and (tuple(corner-lo) not in reached or
            sum(tuple(p-lo) in reached and tuple(q-lo) in reached for p,q in pairs)<4):
        return None
    idx = tuple(np.array(sorted(reached)).T)
    return plane_intercept(coords[idx],z[idx],sigma)


def rim(ctx,side):
    label,depth,valid,pink,across,tol,_,f = ctx
    points = np.argwhere(valid&(label>=0)&(label//13==side))
    rows = [[] for _ in range(13)]
    for step in ((0,1),(0,-1),(1,0),(-1,0)):
        off = np.asarray(step)
        if float(off@across)*(1 if side else -1)<.25:
            continue
        out = points+off
        ok = ((out>=0)&(out<depth.shape)).all(1)
        a,b = points[ok],out[ok]
        i,k = tuple(a.T),tuple(b.T)
        keep = pink[k]&(abs(depth[i]-depth[k])<=tol)
        for p,q in zip(a[keep],b[keep],strict=True):
            center = (p+q)*.5
            z = fit(ctx,p,q,center,side)
            if z is not None:
                rows[label[tuple(p)]%13].append((*center,z))
    return v11._stations([np.asarray(row).reshape(-1,3) for row in rows],depth.shape,f,tol)


def fallback(ctx,side,j):
    label,depth,valid,pink,across,tol,_,f = ctx
    prev = np.argwhere(valid&(label==side*13+j-1))
    pairs4,pairs8 = [],[]
    for step in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
        out = prev+step
        ok = ((out>=0)&(out<depth.shape)).all(1)
        a,b = prev[ok],out[ok]
        i,k = tuple(a.T),tuple(b.T)
        keep = valid[k]&(label[k]==side*13+j)&(abs(depth[i]-depth[k])<=tol)
        adjacent = list(zip(a[keep],b[keep],strict=True))
        pairs8.extend(adjacent)
        if 0 in step:
            pairs4.extend(adjacent)
    if len(pairs4)<4:
        return None
    outward = across*(1 if side else -1)
    candidates = []
    for a,b in pairs8:
        center = (a+b)*.5
        lo,hi = np.maximum(a-2,0),np.minimum(a+3,depth.shape)
        near = np.argwhere(pink[lo[0]:hi[0],lo[1]:hi[1]])+lo
        disp = near-center
        z = depth[tuple(near.T)]
        keep = ((np.max(abs(near-a),axis=1)==1)&(np.max(abs(near-b),axis=1)==1)
                &((disp**2).sum(1)<=4.)&(disp@outward>=.25)
                &(abs(z-depth[tuple(a)])<=tol)&(abs(z-depth[tuple(b)])<=tol))
        candidates.extend((float(center@outward),a,b,w) for w in near[keep])
    if not candidates:
        return None
    outer = max(p[0] for p in candidates)
    xyz = []
    for score,a,b,w in candidates:
        if score!=outer:
            continue
        p,q = (a+w)*.5,(b+w)*.5
        za,zb = (fit(ctx,a,b,center,side,w,pairs4) for center in (p,q))
        if za is None or zb is None or abs(za-zb)>tol:
            return None
        z = 2./(1./za+1./zb)
        rc = (np.array(depth.shape)*.5-(p+q)*.5-.5)*z/f
        xyz.append((rc[1],rc[0],z))
    median = np.median(xyz,axis=0)
    return median if np.max(np.linalg.norm(xyz-median,axis=1))<=.00035 else None


def chart(rgb,z,f,far,sigma,prefix,goal=9):
    margin(sigma)
    if (z.shape!=(960,1280) or not np.isfinite([f,far]).all() or far<=0 or abs(f-640/np.tan(np.deg2rad(35)))>1e-8
            or rgb.dtype!=np.uint8 or rgb.shape!=(*z.shape,3) or type(prefix) is not int or goal not in (9,10)
            or not 0<=prefix<goal):
        return None
    label,good = v11.inks(rgb),v11._good(z,far)
    pink = good&(rgb==v11.PINK).all(-1)
    pix = [np.argwhere(good&(label>=0)&(label//13==s)) for s in (0,1)]
    if not all(len(p) for p in pix):
        return None
    across = pix[1].mean(0)-pix[0].mean(0)
    if np.linalg.norm(across)<1e-6:
        return None
    across /= np.linalg.norm(across)
    ctx = label,z,good,pink,across,.001+3*sigma,sigma,f
    sides = [rim(ctx,s) for s in (0,1)]
    sol = v11._chart(sides,prefix,goal)
    if sol is None and prefix>=6 and any(not {prefix+1,prefix+2,goal,10}<=p.keys() for p in sides):
        for s,points in enumerate(sides):
            for j in range(1,13):
                if j not in points:
                    p = fallback(ctx,s,j)
                    if p is not None:
                        points[j] = p
        sol = v11._chart(sides,prefix,goal)
    if sol is None:
        return None
    g,row,pair,_ = sol
    if prefix<6:
        return g if row<=8 else None
    if row!=10:
        return None
    fixed = [inverse(p,z,good&(pink|((label>=0)&(label//13==s))),sigma,f) for s,p in enumerate(pair)]
    if any(p is None for p in fixed):
        return None
    delta = np.min(np.stack(fixed)@g.direction)-np.min(pair@g.direction)
    return v11.WorkChart(g.prefix_mm,g.front,g.direction,g.opening,g.aim,g.limit+delta*g.direction)
