"""Development-only RGB-D corridor with depth-attached bank ends."""
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree

from cell_certificate_v1 import decode
from m5_guided_bank_corridor_v1 import bank_patch, connected, endpoint_depth
from ruler_readout import inks


def inward(line, side, j, flank, tolerance, bank, cap, depth, good):
    last = 0
    previous = float(depth[tuple(line[0])])
    if abs(previous-flank) > tolerance:
        return None
    for i, pixel in enumerate(map(tuple,line[1:]),1):
        if not good[pixel] or (bank[pixel] != 13*side+j
                               and cap[pixel] not in (2*j+side,2*(j+1)+side)):
            break
        current = float(depth[pixel])
        if abs(current-previous) > tolerance or abs(current-flank) > tolerance:
            break
        previous, last = current, i
    return last


def read_row(j, bank, cap, face, z, good, offset, sigma):
    ent = dict(row=j, status='UNKNOWN')
    left,right = (bank_patch((bank==k)&good) for k in (j,13+j))
    if not len(left) or not len(right):
        return dict(ent,reason='missing or ambiguous bank patch'),None
    distance,nearest = cKDTree(right).query(left)
    i = int(np.argmin(distance))
    a,b = left[i],right[nearest[i]]
    lo,hi = endpoint_depth(left,a,z),endpoint_depth(right,b,z)
    if lo is None or hi is None:
        return dict(ent,reason='insufficient flank depth'),None
    line = np.rint(np.linspace(a,b,int(np.max(abs(a-b)))+1)).astype(int)
    if len(line) < 2:
        return dict(ent,reason='degenerate chord'),None
    tol = .001+3*sigma
    nl = inward(line,0,j,lo,tol,bank,cap,z,good)
    nr = inward(line[::-1],1,j,hi,tol,bank,cap,z,good)
    if nl is None or nr is None:
        return dict(ent,reason='flank anchor not supported in depth'),None
    lips = line[nl],line[len(line)-1-nr]
    ent.update(left_uv=(lips[0]+offset)[::-1].tolist(),right_uv=(lips[1]+offset)[::-1].tolist(),
               extension_px=[nl,nr])
    line = line[max(3,nl+1):len(line)-max(3,nr+1)]
    if len(line)<4:
        return dict(ent,reason='insufficient inner chord'),None
    p = tuple(line.T)
    zz,code,valid = z[p],face[p],good[p]
    blank = (bank[p]<0)&(cap[p]<0)&(code<0)
    deep = valid&blank&(zz>max(lo,hi)+.002+3*sigma)
    center = valid&np.isin(code,(2*j,2*j+1))&(zz>=min(lo,hi)-tol)&(zz<=max(lo,hi)+tol)
    middle = np.flatnonzero(center)
    if deep.all():
        status = 'OPEN'
    elif center.all():
        status = 'INTACT'
    elif (len(middle)>=4 and deep.sum()>=4 and (deep | center).all()
          and middle[-1]-middle[0]+1==len(middle) and np.unique(code[center]).size==1):
        status = 'PARTIAL'
    else:
        status = 'UNKNOWN'
    return dict(ent,status=status,reason='' if status!='UNKNOWN' else 'inner chord unresolved',
                samples=len(line),deep=int(deep.sum()),center=int(center.sum()),
                unresolved=int((~(deep|center)).sum()),invalid=int((~valid).sum()),
                flank_depth_m=[lo,hi]),(*lips,line,max(lo,hi))


def both_present(j, records, evidence, face, depth, good, sigma):
    neighbors = evidence[j-1:j+2]
    if any(e is None for e in neighbors):
        return False
    corners = np.asarray([p for a,b,_,_ in neighbors for p in (a,b)])
    try:
        eq = ConvexHull(corners).equations
    except QhullError:
        return False
    y,x = corners.min(0)
    Y,X = corners.max(0)+1
    roi = np.s_[y:Y,x:X]
    pos = np.moveaxis(np.mgrid[y:Y,x:X],0,-1)
    inside = (pos @ eq[:,:2].T+eq[:,2] <= 1e-8).all(-1)
    z = depth[roi]
    lo,hi = sorted(records[j]['flank_depth_m'])
    tol = .001+3*sigma
    measured = inside&good[roi]&(z>=lo-tol)&(z<=hi+tol)
    for k in (2*j,2*j+1):
        patch = measured&(face[roi]==k)
        if not (patch[:-1,:-1]&patch[1:,:-1]&patch[:-1,1:]&patch[1:,1:]).any():
            return False
    return True


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
