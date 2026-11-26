"""Inner-bank strip certificates with width measured on the connected opening."""
import math

import numpy as np
from scipy import ndimage

from m5_v3_corridor_v1 import FOOTPRINT, bank_patch, block, decode, grow, inks
from m5_v3_corridor_v3 import edge_graph, grow_bank, strips


def read_strip(j, dom, bank, cap, face, z, good, sigma, edges, seen, banks, attached):
    row = dict(row=j,status='UNKNOWN',reason='missing row strip or bilateral support')
    current = [banks[s][j] for s in range(2)]
    unset = np.zeros(z.shape,bool)
    if not dom.any() or not all(m.any() for m in current):
        return row,unset
    tol = .001+3*sigma
    depths = z[current[0]|current[1]]
    near = good&(z>=depths.min()-tol)&(z<=depths.max()+tol)
    blank = (bank<0)&(cap<0)&(face<0)
    deep = good&dom&blank&(z>depths.max()+.002+3*sigma)
    row_codes = [(bank==j+13*s)|np.isin(cap,(2*j+s,2*(j+1)+s)) for s in range(2)]
    shores = [dom&near&attached[s]&row_codes[s] for s in range(2)]
    other = (attached[0]|attached[1])&~(row_codes[0]|row_codes[1])
    faces = [dom&near&(face==2*j+s) for s in range(2)]
    shallow = shores[0]|shores[1]|faces[0]|faces[1]
    left,right = (grow(m,shallow,edges[:2]) for m in shores)
    bridge = left&right
    linked = [bool(block(m&(left|right)).any()) for m in faces]
    intact = all(block(m&bridge).any() for m in faces)
    unresolved = int(np.count_nonzero(dom&~(deep|shallow|other)))
    components,n_components = ndimage.label(deep)
    shared = set(range(1,n_components+1))
    for shore in shores:
        shared &= set(np.unique(components[ndimage.binary_dilation(shore)]))
    through = np.isin(components,sorted(shared))
    row.update(reason='no positive local certificate',unexplained_pixels=unresolved,
               through_pixels=int(through.sum()),face_patches=linked,bridge=bool(bridge.any()))
    if intact:
        row.update(status='INTACT',reason='')
    elif unresolved:
        row.update(reason='unexplained samples inside row strip')
    elif any(linked) and ndimage.binary_erosion(deep,structure=FOOTPRINT).any() and not bridge.any():
        row.update(status='PARTIAL',reason='')
    elif seen:
        row.update(reason='observed central face vetoes OPEN')
    elif bridge.any():
        row.update(reason='shallow cross-bank bridge vetoes OPEN')
    elif through.any():
        row.update(status='OPEN',reason='')
    return row,through


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
    edges = edge_graph(good,z,.001+3*sigma,diagonal=True)
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
    evidence = [read_strip(j,owners==j,bank,cap,face,z,good,sigma,edges,seen[j],banks,attached)
                for j in range(13)]
    rows,witnesses = zip(*evidence,strict=True)
    result['rows'] = list(rows)
    states = [r['status'] for r in rows]
    n = next((j for j,s in enumerate(states) if s!='OPEN'),13)
    if not 0<n<=10:
        return dict(result,reason='no eligible open prefix from row zero')
    if states[n] not in ('INTACT','PARTIAL') or states[n+1]!='INTACT':
        return dict(result,reason='missing positive terminal boundary')
    if any(s in ('OPEN','PARTIAL') for s in states[n+2:]):
        return dict(result,reason='later observed opening')
    domain = (owners>=0)&(owners<n)
    a,b = (domain&side for side in attached)
    if (grow(a,a|b,edges[:2])&b).any():
        return dict(result,reason='shallow cross-bank bridge in open prefix')
    # Erode the opening itself, not each projected strip independently.
    core,_ = ndimage.label(ndimage.binary_erosion(np.logical_or.reduce(witnesses[:n]),structure=FOOTPRINT))
    shared = set(np.unique(core[witnesses[0]]))-{0}
    for m in witnesses[1:n]:
        shared &= set(np.unique(core[m]))
    if not shared:
        return dict(result,reason='no shared minimum-width through-depth across rows')
    return dict(result,verdict='BELOW' if n<9 else 'TARGET',prefix_mm=n,reason='')
