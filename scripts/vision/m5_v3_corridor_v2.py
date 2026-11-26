"""Offline-only ordered inner-bank RGB-D strip certificates."""
import math

import numpy as np
from scipy import ndimage

from m5_v3_corridor_v1 import FOOTPRINT, bank_patch, block, decode, grow, inks, spanning_deep


def side_stations(masks, side):
    pts = [np.argwhere(m) for m in masks]
    ids = [j for j,p in enumerate(pts) if len(p)]
    result = [None]*14
    if len(ids)<2:
        return result
    start,end = pts[ids[0]].mean(0),pts[ids[-1]].mean(0)
    fwd = end-start
    norm = np.linalg.norm(fwd)
    if norm<1e-9:
        return result
    fwd /= norm
    across = np.array([-fwd[1],fwd[0]])
    if across[1]<0:
        across = -across
    inward = across if side==0 else -across
    for j in range(1,13):
        if not (len(pts[j-1]) and len(pts[j])):
            continue
        # Find the inboard end of the contact between two independently decoded rows.
        for steps in (1,2,3,4,5):
            stencil = np.ones((3,3),bool)
            a = np.argwhere(masks[j-1]&ndimage.binary_dilation(masks[j],stencil,iterations=steps))
            b = np.argwhere(masks[j]&ndimage.binary_dilation(masks[j-1],stencil,iterations=steps))
            if len(a) and len(b):
                ends = [p[(p@inward)>=(p@inward).max()-2].mean(0) for p in (a,b)]
                result[j] = (ends[0]+ends[1])*.5
                break
    for j,band,sign in ((0,0,-1),(13,12,1)):
        if not len(pts[band]):
            continue
        p = pts[band]
        near = p[(p@inward)>=(p@inward).max()-12]
        s = sign*(near@fwd)
        edge = near[s>=s.max()-1]
        result[j] = edge.mean(0)
    return result


def strips(banks, shape):
    left,right = (side_stations(banks[s],s) for s in range(2))
    stops = []
    for a,b in zip(left,right,strict=True):
        if a is None or b is None or b[1]<=a[1] or np.linalg.norm(b-a)<1e-9:
            stops.append(None)
            continue
        # The pad allows real shallow contact at the curved lip; it adds no depth.
        d = 12*(b-a)/np.linalg.norm(b-a)
        stops.append((a-d,b+d))
    row_owner = np.full(shape,-1,np.int8)
    rejected = set()
    xy = np.moveaxis(np.indices(shape,dtype=float),0,-1)
    cross = lambda v,q:v[0]*q[...,1]-v[1]*q[...,0]
    for j,(lower,upper) in enumerate(zip(stops,stops[1:])):
        if lower is None or upper is None:
            continue
        a,b = lower
        c,d = upper
        # Polygon side edges are line segments between inboard row junctions,
        # not a hull over bank material. Upper crossing is half-open.
        corners = (a,b,d,c)
        seg = [(p,q-p) for p,q in zip(corners,corners[1:]+corners[:1],strict=True)]
        turns = [cross(v,seg[(k+1)%4][1]) for k,(_,v) in enumerate(seg)]
        sign = np.sign(turns[0])
        if sign==0 or any(sign*t<=1e-6 for t in turns):
            continue
        inside = np.ones(shape,bool)
        for k,(p,v) in enumerate(seg):
            value = sign*cross(v,xy-p)
            inside &= value>1e-8 if k==2 else value>=-1e-8
        assigned = row_owner[inside]
        if (assigned>=0).any():
            rejected.update(np.unique(assigned[assigned>=0]).tolist())
            rejected.add(j)
        row_owner[inside] = j
    for j in rejected:
        row_owner[row_owner==j] = -1
    return row_owner


def measure(rgb, depth, *, far, sigma):
    result = dict(verdict='UNKNOWN',prefix_mm=None,rows=[],reason='invalid RGB-D')
    if (depth.ndim!=2 or rgb.shape!=depth.shape+(3,) or rgb.dtype!=np.uint8 or depth.dtype.kind!='f'
            or not (math.isfinite(sigma) and sigma>=0 and math.isfinite(far) and far>0)):
        return result
    bank = inks(rgb)
    yy,xx = np.where(bank>=0)
    if not len(yy):
        return dict(result,reason='no bank ink')
    cap,face = decode(rgb)
    seen = np.bincount(face[face>=0],minlength=26).reshape(13,2).any(1)
    roi = np.s_[yy.min():yy.max()+1,xx.min():xx.max()+1]
    bank,cap,face,z = bank[roi],cap[roi],face[roi],depth[roi]
    good = np.isfinite(z)&(z>0)&(z<far*.999)
    tol = .001+3*sigma
    safe = np.where(good,z,0.)
    edges = (good[1:]&good[:-1]&(np.abs(safe[1:]-safe[:-1])<=tol),
             good[:,1:]&good[:,:-1]&(np.abs(safe[:,1:]-safe[:,:-1])<=tol))
    banks = []
    anchors = []
    for s in range(2):
        by_row = []
        anchor = np.zeros(z.shape,bool)
        for j in range(13):
            m = np.zeros(z.shape,bool)
            p = bank_patch((bank==j+13*s)&good)
            m[tuple(p.T)] = True
            by_row.append(m)
            anchor |= m
        banks.append(by_row)
        anchors.append(anchor)
    owners = strips(banks,z.shape)
    # Reassign only measured coded-face pixels on the rasterized shared fringe.
    nominal = owners.copy()
    for j in range(13):
        own = nominal==j
        if own.any():
            fringe = ndimage.distance_transform_edt(~own)<=2.
            transfer = (good & fringe & (np.abs(nominal-j)==1) & (nominal>=0)
                        & ((face==2*j)|(face==2*j+1)))
            owners[transfer] = j
    # A decoded neighboring stamp is explainable only when depth-attached on its own side.
    attached = []
    for s in range(2):
        cap_proof = np.zeros(z.shape,bool)
        for j in range(14):
            stamp = (cap==2*j+s)&good
            if stamp.any():
                seed = np.logical_or.reduce(banks[s][max(0,j-1):min(13,j+1)])
                cap_proof |= stamp&grow(seed,seed|stamp,edges,12)
        attached.append(cap_proof)
    blank = (bank<0)&(cap<0)&(face<0)
    witnesses = []
    for j in range(13):
        row = dict(row=j,status='UNKNOWN',reason='missing row strip or bilateral support')
        result['rows'].append(row)
        dom = owners==j
        current = [banks[s][j] for s in range(2)]
        if not dom.any() or not all(m.any() for m in current):
            witnesses.append(np.zeros(z.shape,bool))
            continue
        depths = z[current[0]|current[1]]
        near = good&(z>=depths.min()-tol)&(z<=depths.max()+tol)
        deep = good&dom&blank&(z>depths.max()+.002+3*sigma)
        shores = [dom&near&(current[s]|(attached[s]&np.isin(cap,(2*j+s,2*(j+1)+s)))) for s in range(2)]
        other = np.zeros(z.shape,bool)
        for s in range(2):
            other |= (anchors[s]&~current[s])|(attached[s]&~np.isin(cap,(2*j+s,2*(j+1)+s)))
        faces = [dom&near&(face==2*j+s) for s in range(2)]
        shallow = shores[0]|shores[1]|faces[0]|faces[1]
        left,right = (grow(m,shallow,edges) for m in shores)
        bridge = left&right
        linked_faces = [bool(block(m&(left|right)).any()) for m in faces]
        intact = all(block(m&bridge).any() for m in faces)
        unresolved = int(np.count_nonzero(dom&~(deep|shallow|other)))
        through = spanning_deep(deep,shores)
        row.update(reason='no positive local certificate',unexplained_pixels=unresolved,
                   through_pixels=int(through.sum()),face_patches=linked_faces,bridge=bool(bridge.any()))
        witnesses.append(through)
        if intact:
            row.update(status='INTACT',reason='')
        elif unresolved:
            row.update(reason='unexplained samples inside row strip')
        elif any(linked_faces) and ndimage.binary_erosion(deep,structure=FOOTPRINT).any() and not bridge.any():
            row.update(status='PARTIAL',reason='')
        elif seen[j]:
            row.update(reason='observed central face vetoes OPEN')
        elif bridge.any():
            row.update(reason='shallow cross-bank bridge vetoes OPEN')
        elif through.any():
            row.update(status='OPEN',reason='')
    states = [r['status'] for r in result['rows']]
    n = next((j for j,s in enumerate(states) if s!='OPEN'),13)
    if not 0<n<=10:
        return dict(result,reason='no eligible open prefix from row zero')
    if states[n] not in ('INTACT','PARTIAL') or states[n+1]!='INTACT':
        return dict(result,reason='missing positive terminal boundary')
    if any(s in ('OPEN','PARTIAL') for s in states[n+2:]):
        return dict(result,reason='later observed opening')
    core,_ = ndimage.label(ndimage.binary_erosion(np.logical_or.reduce(witnesses[:n]),structure=FOOTPRINT))
    shared = set(np.unique(core[witnesses[0]]))-{0}
    for m in witnesses[1:n]:
        shared &= set(np.unique(core[m]))
    if not shared:
        return dict(result,reason='no shared minimum-width through-depth across rows')
    return dict(result,verdict='BELOW' if n<9 else 'TARGET',prefix_mm=n,reason='')
