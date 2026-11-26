"""Same-frame outer-bank working limit on the transverse hinge coupon only."""
import numpy as np

import m5_v3_perception_v11 as v11


def margin(sigma):
    if sigma not in (0.,.0005):
        raise ValueError('unregistered hinge noise')
    return .000070+.3*sigma


def inverse(point,z,allowed,sigma,f):
    center = np.rint(np.array(z.shape)*.5-.5-f*point[1::-1]/point[2]).astype(int)
    y,x = center
    if not (3<=y<z.shape[0]-3 and 3<=x<z.shape[1]-3):
        return None
    roi = np.s_[y-3:y+4,x-3:x+4]
    mask,depth = allowed[roi],z[roi].astype(float)
    if not mask[3,3]:
        return None
    seen,pending = {(3,3)},[(3,3)]
    while pending:
        i,j = pending.pop()
        for p in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
            if (0<=p[0]<7 and 0<=p[1]<7 and p not in seen and mask[p] and
                    abs(depth[p]-depth[i,j])<=.001+3*sigma):
                seen.add(p)
                pending.append(p)
    if len(seen)<8:
        return None
    ij = np.array(sorted(seen))
    A = np.column_stack((np.ones(len(ij)),ij-3))
    c,_,rank,_ = np.linalg.lstsq(A,1./depth[tuple(ij.T)],rcond=None)
    if rank!=3:
        return None
    return point/(1.+point[2]*.5*np.max(abs(c[1:])))


def chart(rgb,z,f,far,sigma,prefix,goal=9):
    margin(sigma)
    if z.shape!=(960,1280) or abs(f-640/np.tan(np.deg2rad(35)))>1e-8:
        return None
    sol,_,_ = v11._solve_work(rgb,z,focal=f,far=far,sigma=sigma,prefix_mm=prefix,goal_mm=goal)
    if sol is None:
        return None
    g,row,pair,_ = sol
    if prefix<6:
        return g if row<=8 else None
    if row!=10:
        return None
    label,good,pink = v11.inks(rgb),v11._good(z,far),(rgb==v11.PINK).all(-1)
    corrected = [inverse(p,z,good&(pink|((label>=0)&(label//13==s))),sigma,f) for s,p in enumerate(pair)]
    if any(p is None for p in corrected):
        return None
    delta = np.min(np.stack(corrected)@g.direction)-np.min(pair@g.direction)
    return v11.WorkChart(g.prefix_mm,g.front,g.direction,g.opening,g.aim,g.limit+delta*g.direction)
