"""Development RGB-D certificate for optically coded cells. Not imported by task control."""
import itertools

import numpy as np
from scipy import ndimage

from ruler_readout import inks

UNKNOWN = 'UNKNOWN'
PRESENT = 'PRESENT'
OPEN = 'OPEN'
PITCH_M = .001
# Any surviving host triangle still has all three of its actual (strain-checked) edges.
CAP_RADIUS_M = .15 * 1.5 * np.sqrt(2) * PITCH_M
BOX = np.array(list(itertools.product((-1, 1), repeat=2)))


def decode(rgb):
    image = np.asarray(rgb, dtype=np.int16)
    r,g,b = np.moveaxis(image, 2, 0)
    row = np.rint((g-24)/14).astype(int)
    side = np.rint((b-147)/58).astype(int)
    ok = ((row >= 0) & (row <= 13) & (side >= 0) & (side <= 1)
          & (np.abs(g-24-14*row) <= 3) & (np.abs(b-147-58*side) <= 3))
    vertex = np.where(ok & (abs(r-14)<=3), 2*row+side, -1)
    face = np.where(ok & (row<13) & (abs(r-248)<=3), 2*row+side, -1)
    return vertex, face


def block(mask):
    if min(mask.shape, default=0) < 2:
        return np.zeros((0,0),bool)
    return mask[:-1,:-1]&mask[1:,:-1]&mask[:-1,1:]&mask[1:,1:]


def patch(mask):
    cc,n = ndimage.label(mask)
    return bool(n and np.bincount(cc[mask]).max() >= 4)


def projected_interior(center, radius, xy):
    """Require the ENTIRE pixel to be inside for any choice of the three corner boxes."""
    area = np.linalg.det(center[1:]-center[0])
    if not np.isfinite(area) or abs(area)<1e-8:
        return np.zeros(xy.shape[:-1],bool)
    ok = np.ones(xy.shape[:-1],bool)
    for k in range(3):
        a = np.unique(center[k]+BOX*radius[k],axis=0)
        b = np.unique(center[(k+1)%3]+BOX*radius[(k+1)%3],axis=0)
        for p,q in itertools.product(a,b):
            v = q-p
            signed = np.sign(area)*(v[0]*(xy[...,1]-p[1])-v[1]*(xy[...,0]-p[0]))
            ok &= signed >= .5*np.sum(abs(v))+1e-8
    return ok


def verdict(states, linked):
    n = next((j for j,pair in enumerate(states) if tuple(pair)!=(OPEN,OPEN)),len(states))
    if n==0 or n>10 or len(states)!=13 or not linked:
        return UNKNOWN,None
    front = states[n]
    if UNKNOWN in front or PRESENT not in front or any(tuple(p)!=(PRESENT,PRESENT) for p in states[n+1:]):
        return UNKNOWN,None
    return ('STOP' if n>=9 else 'BELOW'),n


def corners(mask, z, xy, principal, focal, dz):
    if not patch(mask):
        return None
    pixels, values = xy[mask],z[mask]
    if not np.isfinite(values).all() or np.min(values)<=CAP_RADIUS_M+dz:
        return None
    zrange = np.array([values.max()-CAP_RADIUS_M-dz,values.min()+CAP_RADIUS_M+dz])
    if zrange[1]<zrange[0]:
        return None
    # Each cap pixel comes from material within CAP_RADIUS_M of the node, not from its centroid.
    extent = (.5+CAP_RADIUS_M*focal*np.sqrt(1+((abs(pixels-principal)+.5)/focal)**2)
              /(values[:,None]-CAP_RADIUS_M-dz))
    lo,hi=(pixels-extent).max(0),(pixels+extent).min(0)
    if (hi<lo).any():
        return None
    return (lo+hi)/2,(hi-lo)/2,zrange


def classify(rgb, depth, *, focal, principal, sigma=0.):
    rgb,z = np.asarray(rgb),np.asarray(depth)
    stop=lambda why,**kw: dict(decision=UNKNOWN,reason=why,**kw)
    if (z.ndim!=2 or rgb.shape!=z.shape+(3,) or not np.isfinite([*principal,focal,sigma]).all()
            or focal<=0 or sigma<0):
        return stop('invalid sensor calibration')
    # Crop by optical codes only; geometry and noise calibration remain in the full image coordinates.
    cap,face=decode(rgb)
    y,x=np.where((cap>=0)|(face>=0))
    if not len(y):
        return stop('no coded material')
    box=np.s_[max(0,y.min()-3):y.max()+4,max(0,x.min()-3):x.max()+4]
    z,cap,face,rgb=z[box],cap[box],face[box],rgb[box]
    yy,xx=np.indices(z.shape)
    xy=np.stack((xx+.5,yy+.5),axis=-1)
    origin=np.array([box[1].start,box[0].start])
    dz=.00025+3*sigma
    nodes=[corners(cap==k,z,xy,np.asarray(principal)-origin,focal,dz) for k in range(28)]
    missing=[k for k,p in enumerate(nodes) if p is None]
    if missing:
        return stop('unbounded material corners',missing=missing)
    center,radius,zrange=map(np.asarray,zip(*nodes))
    axis=center[-2:].mean(0)-center[:2].mean(0)
    if np.any(np.diff((center@axis).reshape(14,2),axis=0)<=0):
        return stop('unordered bank corners')
    blank=(cap<0)&(face<0)&(inks(rgb)<0)&np.isfinite(z)
    union=np.zeros(z.shape,bool)
    supports=[]
    states=[]
    for j in range(13):
        k=2*j
        pair=[]
        for h,ids in enumerate(([k,k+1,k+3],[k,k+3,k+2])):
            c,r,zz=center[ids],radius[ids],zrange[ids]
            region=projected_interior(c,r,xy)
            deep=blank&(z>np.max(zz[:,1])+dz+.002)
            # A joining path is constrained to the nominal material corridor; empty pixels outside
            # the sheet and an opaque tool do not connect two certified blocks.
            orient=np.sign(np.linalg.det(c[1:]-c[0]))
            hull=np.ones(z.shape,bool) if orient else np.zeros(z.shape,bool)
            for a,b in zip(c,np.roll(c,-1,axis=0)):
                v=b-a
                hull &= orient*(v[0]*(xy[...,1]-a[1])-v[1]*(xy[...,0]-a[0]))>=-1e-8
            union |= deep&hull
            far=block(region&deep)
            lo=(c-r).min(0)-.5
            hi=(c+r).max(0)+.5
            near=(face==k+h)&((xy>=lo)&(xy<=hi)).all(axis=2)&(z>=zz[:,0].min()-dz)&(z<=zz[:,1].max()+dz)
            here=patch(near)
            code=UNKNOWN if here==bool(far.any()) else PRESENT if here else OPEN
            if not here and np.any(face==k+h):
                code=UNKNOWN  # unresolved speck of this face cannot be declared absent
            pair.append(code)
            supports.append(np.where(far))
        states.append(pair)
    comps,_=ndimage.label(union)
    prefix=next((j for j,p in enumerate(states) if p!=[OPEN,OPEN]),13)
    same=None
    for y,x in supports[:2*prefix]:
        ids=set(np.unique(comps[y,x]))-{0}
        same=ids if same is None else same&ids
    decision,n=verdict(states,bool(same))
    return dict(decision=decision,prefix=n,states=states,connected=bool(same),
                reason='' if decision!=UNKNOWN else 'incomplete or contradictory face evidence')
