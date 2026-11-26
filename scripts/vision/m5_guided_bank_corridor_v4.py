"""Versioned row reader with bounded, depth-connected shore lips."""
from collections import deque
import math

import numpy as np
from scipy import ndimage

from cell_certificate_v1 import decode
from m5_guided_bank_corridor_v1 import bank_patch, connected
from m5_guided_bank_corridor_v2 import both_present, read_row
from ruler_readout import inks

PIXEL_STEPS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def shore_path(start, target, allowed, z, tol):
    if not allowed[start]:
        return None
    previous = {start: None}
    queue = deque([(start, 0)])
    while queue:
        p, steps = queue.popleft()
        if p in target:
            path = []
            while p is not None:
                path.append(p)
                p = previous[p]
            return path[::-1]
        if steps == 12:
            continue
        for dy, dx in PIXEL_STEPS:
            q = p[0]+dy, p[1]+dx
            if (q not in previous and 0 <= q[0] < z.shape[0] and 0 <= q[1] < z.shape[1]
                    and allowed[q] and abs(z[p]-z[q]) <= tol):
                previous[q] = p
                queue.append((q, steps+1))
    return None


def recover_open(row, proof, bank, cap, face, z, good, offset, sigma):
    a, b, line, ceil = proof
    j = row['row']
    p = tuple(line.T)
    if len(line) < 9 or (face[p] >= 0).any() or not good[p].all():
        return row, proof
    tol = .001+3*sigma
    deep_mask = good & (bank<0) & (cap<0) & (face<0) & (z>ceil+.002+3*sigma)
    deep = deep_mask[p]
    core = np.flatnonzero(deep)
    if len(core)<4 or deep.all() or core[-1]-core[0]+1!=len(core):
        return row, proof
    left,right = int(core[0]),int(len(line)-core[-1]-1)
    if max(left,right)>min(len(line)//4,12):
        return row, proof
    paths = []
    for k in np.flatnonzero(~deep):
        side = 0 if k<left else 1
        start = tuple(line[k])
        # Clip the graph to the largest possible reach of the bounded search.
        low = np.maximum(np.array(start)-12,0)
        high = np.minimum(np.array(start)+13,z.shape)
        roi = np.s_[low[0]:high[0],low[1]:high[1]]
        code = 13*side
        seed_patch = bank_patch((bank==code+j)&good)
        target = {tuple(p-low) for p in seed_patch if (p>=low).all() and (p<high).all()}
        bank_ok = np.isin(bank[roi], code+np.arange(max(0,j-1),min(12,j+1)+1))
        cap_ok = np.isin(cap[roi], (2*j+side,2*(j+1)+side))
        dz = z[roi]
        allowed = (good[roi] & (face[roi]<0) & (bank_ok|cap_ok)
                   & (abs(dz-row['flank_depth_m'][side]) <= tol))
        path = shore_path(tuple(line[k]-low),target,allowed,dz,tol)
        if path is None:
            return row, proof
        paths.append(dict(sample_index=int(k),side=side,uv=(np.asarray(path)+low+offset)[:,::-1].tolist()))
    y,x = np.maximum(line.min(axis=0)-2,0)
    Y,X = np.minimum(line.max(axis=0)+3,z.shape)
    seeds = np.zeros((Y-y,X-x),dtype=bool)
    seeds[tuple((line-[y,x]).T)] = True
    strip = ndimage.binary_dilation(seeds,iterations=2)
    labels, _ = ndimage.label(strip&deep_mask[y:Y,x:X])
    ids = np.unique(labels[tuple((line[deep]-[y,x]).T)])
    if len(ids) != 1 or ids[0] == 0:
        return row, proof
    revised = dict(row,status='OPEN',reason='',shore_samples=len(paths),unresolved=0,shore_paths=paths)
    return revised,(a,b,line[deep],ceil)


def measure(rgb, depth, *, far, sigma):
    result = dict(verdict='UNKNOWN',prefix_mm=None,rows=[],reason='invalid RGB-D')
    if (depth.ndim!=2 or rgb.shape!=depth.shape+(3,) or rgb.dtype!=np.uint8
            or depth.dtype.kind!='f' or not (math.isfinite(far) and far>0)
            or not (math.isfinite(sigma) and sigma>=0)):
        return result
    bank = inks(rgb)
    y,x = np.where(bank>=0)
    if not len(y):
        return dict(result,reason='no bank ink')
    offset = np.array([y.min(),x.min()])
    roi = np.s_[y.min():y.max()+1,x.min():x.max()+1]
    cap,face = decode(rgb[roi])
    bank,z = bank[roi],depth[roi]
    good = np.isfinite(z)&(z>0)&(z<far*.999)
    evidence = []
    for j in range(13):
        row,proof = read_row(j,bank,cap,face,z,good,offset,sigma)
        if row['status']=='UNKNOWN' and proof is not None:
            row,proof = recover_open(row,proof,bank,cap,face,z,good,offset,sigma)
        result['rows'].append(row)
        evidence.append(proof)
    states = [r['status'] for r in result['rows']]
    n = next((j for j,s in enumerate(states) if s!='OPEN'),13)
    if not 0<n<=10:
        return dict(result,reason='no eligible open prefix from row zero')
    if states[n] not in ('INTACT','PARTIAL') or states[n+1]!='INTACT':
        return dict(result,reason='missing positive terminal boundary')
    if any(s in ('OPEN','PARTIAL') for s in states[n+2:]):
        return dict(result,reason='later observed opening')
    if not both_present(n+1,result['rows'],evidence,face,z,good,sigma):
        return dict(result,reason='next row lacks both observed face codes')
    why = connected(evidence[:n],(bank<0)&(cap<0)&(face<0),face,z,good,sigma)
    if why:
        return dict(result,reason=why)
    return dict(result,verdict='BELOW' if n<9 else 'TARGET',prefix_mm=n,reason='')
