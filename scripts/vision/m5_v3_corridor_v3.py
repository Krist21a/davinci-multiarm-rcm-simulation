"""Offline strip certificates with depth-connected raster banks."""
import math

import numpy as np
from scipy import ndimage

from m5_v3_corridor_v1 import FOOTPRINT, bank_patch, block, decode, grow, inks, spanning_deep
from m5_v3_corridor_v2 import side_stations


def cross(v, q):
    return v[0]*q[..., 1]-v[1]*q[..., 0]


def triangles(corners):
    for ids in (((0, 1, 2), (0, 2, 3)), ((0, 1, 3), (1, 2, 3))):
        pair = [[corners[k] for k in t] for t in ids]
        areas = [cross(b-a, c-a) for a,b,c in pair]
        if min(abs(a) for a in areas)>1e-6 and np.sign(areas[0])==np.sign(areas[1]):
            return pair,np.sign(areas[0])
    return None


def strips(banks, shape):
    left,right = (side_stations(banks[s],s) for s in range(2))
    stops = []
    for a,b in zip(left,right,strict=True):
        if a is None or b is None or b[1]<=a[1] or np.linalg.norm(b-a)<1e-9:
            stops.append(None)
            continue
        pad = 12*(b-a)/np.linalg.norm(b-a)
        stops.append((a-pad,b+pad))
    owner = np.full(shape,-1,np.int8)
    rejected = set()
    xy = np.moveaxis(np.indices(shape,dtype=float),0,-1)
    for j,(start,end) in enumerate(zip(stops,stops[1:])):
        if start is None or end is None:
            continue
        a,b = start
        c,d = end
        result = triangles((a,b,d,c))
        if result is None:
            continue
        pair,sign = result
        area = np.zeros(shape,bool)
        for tri in pair:
            m = np.ones(shape,bool)
            for u,v in zip(tri,tri[1:]+tri[:1],strict=True):
                m &= sign*cross(v-u,xy-u)>=-1e-8
            area |= m
        delta = xy-c
        t = delta@(d-c)
        upper = (abs(cross(d-c,delta))<=1e-8)&(t>=-1e-8)&(t<=(d-c)@(d-c)+1e-8)
        area &= ~upper
        assigned = owner[area]
        if (assigned>=0).any():
            rejected.update(np.unique(assigned[assigned>=0]).tolist())
            rejected.add(j)
        owner[area]=j
    for j in rejected:
        owner[owner==j]=-1
    return owner


def edge_graph(good, z, tol, diagonal=False):
    safe = np.where(good,z,0.)

    def joins(a,b):
        return good[a]&good[b]&(np.abs(safe[a]-safe[b])<=tol)

    edges = (joins(np.s_[1:,:],np.s_[:-1,:]),joins(np.s_[:,1:],np.s_[:,:-1]))
    if diagonal:
        edges += (joins(np.s_[1:,1:],np.s_[:-1,:-1]),joins(np.s_[1:,:-1],np.s_[:-1,1:]))
    return edges


def grow_bank(seed, eligible, edges):
    vertical,horizontal,falling,rising = edges
    seen = seed&eligible
    for _ in range(12):
        nxt = seen.copy()
        nxt[1:,:] |= seen[:-1,:]&vertical
        nxt[:-1,:] |= seen[1:,:]&vertical
        nxt[:,1:] |= seen[:,:-1]&horizontal
        nxt[:,:-1] |= seen[:,1:]&horizontal
        nxt[1:,1:] |= seen[:-1,:-1]&falling
        nxt[:-1,:-1] |= seen[1:,1:]&falling
        nxt[1:,:-1] |= seen[:-1,1:]&rising
        nxt[:-1,1:] |= seen[1:,:-1]&rising
        nxt &= eligible
        if np.array_equal(nxt,seen):
            break
        seen=nxt
    return seen


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
    edges = edge_graph(good,z,tol,diagonal=True)
    banks,attached = [],[]
    for s in range(2):
        side,proofs = [],[]
        for j in range(13):
            anchor = np.zeros(z.shape,bool)
            p = bank_patch((bank==j+13*s)&good)
            anchor[tuple(p.T)]=True
            side.append(anchor)
            codes = np.isin(bank,13*s+np.arange(max(0,j-1),min(12,j+1)+1))
            endpoints = np.isin(cap,(2*j+s,2*(j+1)+s))
            proofs.append(grow_bank(anchor,good&(codes|endpoints),edges))
        banks.append(side)
        attached.append(np.logical_or.reduce(proofs))
    owners = strips(banks,z.shape)
    original = owners.copy()
    for j in range(13):
        own = original==j
        if own.any():
            fringe = ndimage.distance_transform_edt(~own)<=2.
            transfer = (good&fringe&(np.abs(original-j)==1)&(original>=0)
                        &((face==2*j)|(face==2*j+1)))
            owners[transfer]=j
    blank = (bank<0)&(cap<0)&(face<0)
    witnesses,domains = [],[]
    for j in range(13):
        row = dict(row=j,status='UNKNOWN',reason='missing row strip or bilateral support')
        result['rows'].append(row)
        dom = owners==j
        domains.append(dom)
        current = [banks[s][j] for s in range(2)]
        if not dom.any() or not all(m.any() for m in current):
            witnesses.append(np.zeros(z.shape,bool))
            continue
        depths = z[current[0]|current[1]]
        near = good&(z>=depths.min()-tol)&(z<=depths.max()+tol)
        deep = good&dom&blank&(z>depths.max()+.002+3*sigma)
        shores = [dom&near&attached[s]&((bank==j+13*s)|np.isin(cap,(2*j+s,2*(j+1)+s)))
                  for s in range(2)]
        other = (attached[0]|attached[1])&~(shores[0]|shores[1])&((bank>=0)|(cap>=0))
        # Never exempt current-row pixels from their measured flank envelope.
        other &= ~((bank==j)|(bank==j+13)|np.isin(cap,(2*j,2*j+1,2*j+2,2*j+3)))
        faces = [dom&near&(face==2*j+s) for s in range(2)]
        shallow = shores[0]|shores[1]|faces[0]|faces[1]
        left,right = (grow(m,shallow,edges[:2]) for m in shores)
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
    domain = np.logical_or.reduce(domains[:n])
    a,b = (domain&side for side in attached)
    if (grow(a,a|b,edges[:2])&b).any():
        return dict(result,reason='shallow cross-bank bridge in open prefix')
    core,_ = ndimage.label(ndimage.binary_erosion(np.logical_or.reduce(witnesses[:n]),structure=FOOTPRINT))
    shared = set(np.unique(core[witnesses[0]]))-{0}
    for m in witnesses[1:n]:
        shared &= set(np.unique(core[m]))
    if not shared:
        return dict(result,reason='no shared minimum-width through-depth across rows')
    return dict(result,verdict='BELOW' if n<9 else 'TARGET',prefix_mm=n,reason='')
