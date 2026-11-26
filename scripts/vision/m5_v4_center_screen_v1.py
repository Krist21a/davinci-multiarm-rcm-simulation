"""Development-only axial interval from central material ink in one RGB-D exposure."""
import numpy as np
from scipy import ndimage

from cell_certificate_v1 import CAP_RADIUS_M, PITCH_M, decode

KEYS = ('cap_left','cap_right','face_before','face_after')
FACE_RADIUS = np.sqrt(1.5)*1.5*PITCH_M


def witnesses(rgb):
    cap,face = decode(rgb)
    raw = (cap==20,cap==21,face==19,face==20)
    pts = [np.argwhere(m) for m in raw]
    report = {}
    for name,rc in zip(KEYS,pts,strict=True):
        n,largest = 0,0
        if len(rc):
            lo,hi = rc.min(0),rc.max(0)
            mask = np.zeros(hi-lo+1,bool)
            mask[tuple((rc-lo).T)] = True
            labels,n = ndimage.label(mask)
            largest = int(np.bincount(labels[mask]).max())
        report[name] = dict(raw=len(rc),components=n,largest=largest)
    ids = {tuple(p):i for i,p in enumerate(pts[3].tolist())}
    contacts = [(i,ids[q]) for i,(y,x) in enumerate(pts[2].tolist())
                for q in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if q in ids]
    return pts,np.asarray(contacts,dtype=int).reshape(-1,2),report


def axial_box(rc,z,shape,focal,axis,dz,radius):
    # The projected scalar is affine in pixel row/column for fixed depth.
    y,x = ((np.array(shape)*.5-.5-rc)/focal).T
    coeff = axis[0]*x+axis[1]*y+axis[2]
    half = .5*(abs(axis[0])+abs(axis[1]))/focal
    lows = np.minimum((coeff-half)*(z-dz),(coeff-half)*(z+dz))-radius
    highs = np.maximum((coeff+half)*(z-dz),(coeff+half)*(z+dz))+radius
    return np.array([np.max(lows),np.min(highs)])


def measure(evidence,depth,*,focal,far,sigma,axis):
    pts,pairs,stats = evidence
    dz = .00025+3*sigma
    zs = [depth[tuple(rc.T)].astype(float) for rc in pts]
    good = [np.isfinite(z)&(z>dz)&(z<far*.999) for z in zs]
    counts = [int(v.sum()) for v in good]
    patch = [stats[k]['components']==1 and c==stats[k]['raw'] and c>=4 for k,c in zip(KEYS,counts,strict=True)]
    a,b = pairs.T
    sel = good[2][a]&good[3][b]&(np.abs(zs[2][a]-zs[3][b])<=.001+3*sigma)
    touch = [len(np.unique(i[sel])) for i in (a,b)]
    cap_ok = patch[0] and patch[1]
    face_ok = patch[2] and patch[3] and min(touch)>=4
    out = dict(valid_counts=counts,eligible_patches=patch,contacts=touch,cap_witness=cap_ok,
               face_witness=face_ok,route=None,interval_m=None,reason='no bilateral caps or witnessed central edge')
    if not (cap_ok or face_ok):
        return out
    if axis is None:
        return dict(out,reason='missing optical direction')
    if not np.isclose(np.linalg.norm(axis),1.,rtol=0,atol=1e-8):
        raise ValueError('invalid unit axis')
    routes = []
    for offset,radius,active in ((0,CAP_RADIUS_M,cap_ok),(2,FACE_RADIUS,face_ok)):
        if not active:
            continue
        boxes = np.array([axial_box(pts[k],zs[k],depth.shape,focal,axis,dz,radius) for k in (offset,offset+1)])
        if (boxes[:,0]>boxes[:,1]).any():
            return dict(out,reason='inconsistent pixel intervals')
        routes.append(boxes.mean(0) if offset==0 else [boxes[:,0].max(),boxes[:,1].min()])
    low,high = np.array(routes).T
    if low.max()>high.min():
        return dict(out,reason='contradictory central material intervals')
    return dict(out,reason='',route='both' if cap_ok and face_ok else ('caps' if cap_ok else 'faces'),
                interval_m=[float(low.max()),float(high.min())])
