"""Locate a material row at a measured lip/ink junction, never at a free line end."""
import numpy as np
from scipy.spatial import cKDTree

from bilateral_surface import _GAP, _bank_plane, _ray, _unit


def _contacts(label, z, code, other, far):
    """Pixel-edge midpoints oriented from near material to the requested third region."""
    yy, xx = np.where((label == code) & np.isfinite(z) & (z > 0) & (z < far*.999))
    points, ranges = [], []
    for dx, dy in ((0,-1), (0,1), (-1,0), (1,0)):
        x, y = xx+dx, yy+dy
        inside = (y >= 0) & (x >= 0) & (y < z.shape[0]) & (x < z.shape[1])
        x,y,px,py = x[inside],y[inside],xx[inside],yy[inside]
        d,q = z[py,px],z[y,x]
        good = np.isfinite(q) & (q > 0) & (q < far*.999)
        if other is None:
            good &= (label[y,x] < 0) & (q-d > _GAP)
        else:
            good &= (label[y,x] == other) & (abs(d-q) <= _GAP)
        points.extend(np.column_stack((px[good]+.5+dx*.5, py[good]+.5+dy*.5)))
        ranges.extend(d[good])
    return np.array(points).reshape(-1,2), np.array(ranges)


def _line(points, q, span):
    """Local fit, so a bend several material columns away cannot rotate this edge."""
    p = points[np.linalg.norm(points-q, axis=1) <= span]
    if len(p)<3:
        return None
    mid = p.mean(0)
    _,s,axes = np.linalg.svd(p-mid, full_matrices=False)
    if s[0]<1.3 or s[1]>=s[0]*.5 or np.max(abs((p-mid) @ axes[1]))>1:
        return None
    return mid,axes[0],p


def _cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def boundary(label, z, j, side, towards, pp, f, far):
    """Return (pixel intersection, 3D bank point, local normal), or abstain."""
    a,b = side*13+j-1,side*13+j
    # On the printed central diagonal, the third ink at the left vertex is
    # right[j], and at the right vertex it is left[j-1]. They are not interchangeable.
    intact = j-1 if side else 13+j
    edge,depth = _contacts(label,z,a,b,far)
    inward = _unit(towards)
    if len(edge)<3 or inward is None:
        return None
    pitch = f*.001/np.median(depth)
    span = max(3., min(9., pitch))
    candidates=[]
    for other in (None,intact):
        rims = [_contacts(label,z,k,other,far)[0] for k in (a,b)]
        for rim in rims:
            if len(rim)<3:
                continue
            near,idx = cKDTree(rim).query(edge)
            for ei in np.flatnonzero(near<=np.sqrt(2)+.5):
                approx = .5*(edge[ei]+rim[idx[ei]])
                band,lip = _line(edge,approx,span),_line(rim,approx,span)
                if band is None or lip is None:
                    continue
                c,d,points = band
                m,l,rp = lip
                det = _cross(d,l)
                if abs(det)<.2 or abs(d @ inward)<.2:
                    continue
                if d @ inward<0:
                    d,det = -d,-det
                knot = c+d*(_cross(m-c,l)/det)
                # The band must end HERE on its inboard side. A parallel edge,
                # outboard sheet border or interrupted run has no corner.
                if (np.linalg.norm(knot-approx)>np.sqrt(2)+.5
                        or np.max((points-knot) @ d)>1.5
                        or np.max((knot-points) @ d)<1.5
                        or min(np.linalg.norm(rp-knot,axis=1))>np.sqrt(2)+.5):
                    continue
                candidates.append(knot)
    if not candidates:
        return None
    candidates=np.array(candidates)
    knot=np.median(candidates,axis=0)
    if np.max(np.linalg.norm(candidates-knot,axis=1))>np.sqrt(2):
        return None
    local=depth[np.linalg.norm(edge-knot,axis=1)<=span]
    plane=_bank_plane(label,z,[a,b],knot,pp,f,far,np.median(local))
    if plane is None:
        return None
    return knot,_ray(knot,pp,f)*plane[0],plane[1]
