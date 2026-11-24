"""Passive row/column stations and local slit evidence on the indexed surface print."""
from collections import defaultdict

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree

from bilateral_surface import _bank_plane, _ray


def decode(rgb):
    """Invert the immutable face ink, including its checksum channel."""
    r,g,b=np.moveaxis(rgb.astype(np.int16),2,0)
    j=np.rint((r-42)/13).astype(np.int16)
    k=np.rint((g-50)/21).astype(np.int16)
    ok=((0<=j)&(j<13)&(0<=k)&(k<8)&(abs(r-42-13*j)<=3)&(abs(g-50-21*k)<=3)
        &(abs(b-42-25*((3*j+5*k)%8))<=3))
    return np.where(ok,8*j+k,-1)


def _contacts(lab,z,far):
    valid=(lab>=0)&np.isfinite(z)&(z>0)&(z<far*.999)
    groups=defaultdict(list)
    for dy,dx in ((0,1),(1,0)):
        p,q=np.s_[:lab.shape[0]-dy,:lab.shape[1]-dx],np.s_[dy:,dx:]
        good=valid[p]&valid[q]&(lab[p]!=lab[q])&(abs(z[p]-z[q])<=.002)
        yy,xx=np.where(good)
        for y,x in zip(yy,xx):
            key=tuple(sorted((lab[y,x],lab[y+dy,x+dx])))
            groups[key].append((x+.5+dx/2,y+.5+dy/2))
    return {k:np.array(uv) for k,uv in groups.items()}


def _station(lab,z,contacts,j,k,pp,f,far):
    # Two distinct transverse edges and two distinct longitudinal edges must
    # terminate in the same pixel neighbourhood. An exposed lip cannot replace one.
    a=(j-1)*8+k
    codes=[a,a+1,a+8,a+9]
    edges=[contacts.get((codes[i],codes[t])) for i,t in ((0,1),(2,3),(0,2),(1,3))]
    if any(e is None for e in edges):
        return None
    anchors=min(edges,key=len)
    chosen=[e[cKDTree(e).query(anchors)[1]] for e in edges]
    clusters=np.stack(chosen,axis=1)
    centers=clusters.mean(1)
    radius=np.max(np.linalg.norm(clusters-centers[:,None],axis=2),axis=1)
    good=radius<=1.6
    if not good.any():
        return None
    uv=centers[radius.argmin()]
    if np.max(np.linalg.norm(centers[good]-uv,axis=1))>2:
        return None
    axes=[]
    for group in (edges[:2],edges[2:]):
        pts=np.vstack(group)
        pts=pts[np.linalg.norm(pts-uv,axis=1)<=3.5]
        if len(pts)<3:
            return None
        m=pts.mean(0)
        _,s,v=np.linalg.svd(pts-m,full_matrices=False)
        if s[0]<1 or s[1]>=.6*s[0]:
            return None
        axes.append((m,v[1]))
    normals=np.stack([v for _,v in axes])
    if abs(np.linalg.det(normals))<.2:
        return None
    p=np.linalg.solve(normals,[m@v for m,v in axes])
    if np.linalg.norm(p-uv)>1 or max(min(np.linalg.norm(e-p,axis=1)) for e in edges)>2:
        return None
    ys,xs=np.where(np.isin(lab,codes)&(z>0)&(z<far*.999))
    near=np.hypot(xs+.5-p[0],ys+.5-p[1])<=2.5
    if set(lab[ys[near],xs[near]])!=set(codes):
        return None
    plane=_bank_plane(lab,z,codes,p,pp,f,far,np.median(z[ys[near],xs[near]]))
    if plane is None:
        return None
    return p,_ray(p,pp,f)*plane[0]


def _patch(mask,n):
    regions,num=ndimage.label(mask)
    if not num:
        return np.zeros_like(mask)
    counts=np.bincount(regions.ravel(),minlength=num+1)
    return mask & (counts[regions]>=n)


def _aperture(lab,z,j,stations,far):
    """Positive connected deep passage touching BOTH indexed banks of this row."""
    ref=[stations.get((t,k)) for t in (j,j+1) for k in (9,14)]
    if any(p is None for p in ref):
        return 'no bilateral bracket'
    sides=[np.isin(lab,[8*j+k for k in lanes]) for lanes in ((1,2),(5,6))]
    sides=[_patch(s & (z>0) & (z<far*.999),2) for s in sides]
    if not all(s.any() for s in sides):
        return 'missing bank'
    xy=np.vstack([np.stack([p[0] for p in ref]),*(np.argwhere(s)[:,::-1]+.5 for s in sides)])
    try:
        hull=ConvexHull(xy)
    except QhullError:
        return 'degenerate bracket'
    x0,y0=np.maximum(0,np.floor(xy.min(0)).astype(int))
    x1,y1=np.minimum(lab.shape[::-1],np.ceil(xy.max(0)).astype(int)+1)
    roi=np.s_[y0:y1,x0:x1]
    y,x=np.indices((y1-y0,x1-x0))
    p=np.stack([x+x0+.5,y+y0+.5],axis=-1)
    inside=np.all(p @ hull.equations[:,:2].T+hull.equations[:,2]<=1e-7,axis=2)
    ceil=max(station[1][2] for station in ref)
    # Use a conservative local upper bound on flank depth. Neither unknown
    # range nor an opaque nearer tool can serve as a positive through ray.
    ceil=max(ceil,*(float(np.max(z[s])) for s in sides))
    ink,d=lab[roi],z[roi]
    valid=np.isfinite(d)&(d>0)&(d<far*.999)
    deep=inside&(ink<0)&valid&(d>ceil+.002)
    if (inside&(~valid|((ink<0)&~deep))).any():
        return 'unresolved or covered corridor'
    if np.any(inside&np.isin(ink,[8*j+3,8*j+4])):
        return 'residual central ink'
    regions,nc=ndimage.label(deep)
    for i in range(1,nc+1):
        on=regions==i
        if on.sum()<4:
            continue
        rim=ndimage.binary_dilation(on)
        if all(np.count_nonzero(rim&s[roi])>=2 for s in sides):
            return 'through'
    return 'no connected through-depth passage'


def measure(rgb,depth,width,height,focal,far):
    """RGB-D + intrinsics only. Locations are camera metres and full-frame pixel coordinates."""
    report=dict(stations=[],central_faces=[],local_mm=None,prefix_mm=None,front_xyz=None,
                decision='UNKNOWN',motion_authorized=False,reason='invalid RGB-D')
    im,z=np.asarray(rgb),np.asarray(depth)
    if (z.shape!=(height,width) or im.shape!=(*z.shape,3) or im.dtype!=np.uint8
            or not np.isfinite([focal,far]).all() or min(focal,far)<=0 or z.dtype.kind!='f'):
        return report
    lab=decode(im)
    # Work in the print's bounding box. Do not select an ROI by tool, phase or truth.
    yy,xx=np.where(lab>=0)
    if not len(xx):
        return dict(report,reason='no indexed material')
    ox,oy=xx.min(),yy.min()
    crop=np.s_[oy:yy.max()+1,ox:xx.max()+1]
    lab,z=lab[crop],z[crop]
    pp=np.array([width/2-ox,height/2-oy])
    contacts=_contacts(lab,z,far)
    stations={}
    for j in range(1,13):
        for k,col in ((0,9),(6,14)):
            pair=_station(lab,z,contacts,j,k,pp,focal,far)
            if pair is not None:
                stations[j,col]=pair
                report['stations'].append(dict(row=j,column=col,uv=(pair[0]+[ox,oy]).tolist(),
                                               camera_m=pair[1].tolist()))
    # There is no ink equivalence between a surviving outboard cell and either
    # central triangle. One suspicious central pixel vetoes OPEN, but cannot
    # prove PRESENT; require a connected patch adjacent to the indexed sheet.
    positive=set()
    suspect=set()
    valid=(z>0)&(z<far*.999)
    for j in range(13):
        for half,k in ((0,4),(1,3)):
            mask=lab==8*j+k
            if not mask.any():
                continue
            suspect.add(j)
            patch=_patch(mask&valid,2)
            near_depth=ndimage.maximum_filter(np.where(patch,z,0.),size=3)
            neighbours=(ndimage.binary_dilation(patch)&np.isin(lab,[8*j+i for i in (1,2,5,6,7-k)])
                        &valid&(abs(z-near_depth)<.002))
            if patch.any() and neighbours.any():
                positive.add((j,half))
                report['central_faces'].append(dict(row=j,half=half,pixels=int(patch.sum())))
    # A directly seen surviving central triangle gives only an upper bound:
    # without an open aperture behind it, the local endpoint stays unknown.
    report['first_present_row']=min((j for j,_ in positive),default=None)
    openings={}
    for j in range(1,12):
        if j not in suspect:
            openings[j]=_aperture(lab,z,j,stations,far)
    report['apertures']={str(k):v for k,v in openings.items()}
    n=report['first_present_row']
    if (n is not None and 2<=n<=10 and openings.get(n-1)=='through'
            and not any(j<n for j in suspect) and any(j==n+1 for j,_ in positive)
            and not any(v=='through' and j>=n for j,v in openings.items())):
        report.update(local_mm=n,reason='through passage followed by indexed central material')
    else:
        report['reason']='no uniquely supported local transition'
    return report
