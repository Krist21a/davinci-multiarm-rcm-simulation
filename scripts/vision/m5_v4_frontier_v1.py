"""Development-only incremental review: carry ordinals, never image geometry."""
import numpy as np
from scipy import ndimage

from m5_v3_corridor_v5 import FOOTPRINT, _bridge8, bank_patch, decode, edge_graph, grow_bank, inks
from m5_v3_corridor_v5 import read_strip, strips
from m5_v3_perception_v4 import intact_preflight


def measure(rgb, z, *, far, sigma, prefix):
    result = dict(verdict='UNKNOWN',prefix_mm=None,prior=prefix,rows=[],reason='invalid input')
    if (type(prefix) is not int or not 0<=prefix<=9 or z.ndim!=2 or z.dtype.kind!='f'
            or rgb.shape!=(*z.shape,3) or rgb.dtype!=np.uint8 or not np.isfinite(far) or far<=0
            or sigma not in (0.,.0005)):
        return result
    bank = inks(rgb)
    y,x = np.nonzero(bank>=0)
    if not len(y):
        return dict(result,reason='no bank ink')
    cap,face = decode(rgb)
    seen = np.bincount(face[face>=0],minlength=26).reshape(13,2).any(1)
    roi = np.s_[y.min():y.max()+1,x.min():x.max()+1]
    bank,cap,face,z = bank[roi],cap[roi],face[roi],z[roi]
    good = np.isfinite(z)&(z>0)&(z<far*.999)
    edges = edge_graph(good,z,.001+3*sigma,diagonal=True)
    banks,attached = [],[]
    for s in range(2):
        side,proofs = [],[]
        for j in range(13):
            mask = np.zeros(z.shape,bool)
            pts = bank_patch(good&(bank==j+13*s))
            mask[tuple(pts.T)] = True
            side.append(mask)
            codes = np.isin(bank,13*s+np.arange(max(0,j-1),min(12,j+1)+1))
            proofs.append(grow_bank(mask,good&(codes|np.isin(cap,(2*j+s,2*(j+1)+s))),edges))
        banks.append(side)
        attached.append(np.logical_or.reduce(proofs))
    original = strips(banks,z.shape)
    owners = original.copy()
    for j in range(13):
        own = original==j
        if own.any():
            fringe = ndimage.distance_transform_edt(~own)<=2.
            owners[good&fringe&(np.abs(original-j)==1)&(original>=0)&((face==2*j)|(face==2*j+1))] = j
    rows,masks = zip(*(read_strip(j,owners,bank,cap,face,z,good,sigma,edges,seen[j],banks,attached)
                       for j in range(13)),strict=True)
    result['rows'] = list(rows)
    start = max(0,prefix-1)
    if seen[:prefix].any() or any(r['status'] in ('INTACT','PARTIAL') or r.get('bridge') for r in rows[:prefix]):
        return dict(result,reason='observed contradiction to receipt')
    n = next((j for j in range(start,13) if rows[j]['status']!='OPEN'),13)
    if not prefix<n<=10:
        return dict(result,reason='missing overlap or fresh complete new row')
    if rows[n]['status'] not in (('INTACT','PARTIAL') if n<9 else ('INTACT',)) or rows[n+1]['status']!='INTACT':
        return dict(result,reason='missing positive terminal boundary')
    if any(r['status'] in ('OPEN','PARTIAL') for r in rows[n+2:]):
        return dict(result,reason='later opening')
    domain = (owners>=0)&(owners<n)
    if _bridge8(domain&attached[0],domain&attached[1],edges):
        return dict(result,reason='shallow cross-bank bridge')
    active = masks[start:n]
    core,_ = ndimage.label(ndimage.binary_erosion(np.logical_or.reduce(active),structure=FOOTPRINT))
    if not set.intersection(*(set(np.unique(core[m]))-{0} for m in active)):
        return dict(result,reason='no common minimum-width core across overlap and new rows')
    return dict(result,verdict='TARGET' if n>=9 else 'BELOW',prefix_mm=n,reason='')


class Reviews:
    """One immutable specimen/run identity; a lost sequence requires a new preflight."""
    def __init__(self, identity):
        self.identity = identity
        self.prefix,self.closures,self.tick = None,0,-1
        self.pending,self.terminal,self.spent = False,False,False

    def observe(self, identity, tick, closures, closed, rgb, z, *, far, sigma=0., alternate=False):
        if (identity!=self.identity or type(tick) is not int or tick<=self.tick or closed is not True
                or self.terminal or alternate!=self.pending or (alternate and self.spent)):
            raise ValueError('unbound, stale or unscheduled review')
        if type(closures) is not int or closures!=(0 if self.prefix is None else self.closures+1):
            raise ValueError('missing or repeated closure')
        self.tick = tick
        if self.prefix is None:
            if alternate or not intact_preflight(rgb,z,far=far)['passed']:
                raise ValueError('missing current uncut entrance certificate')
            self.prefix = 0
            return dict(verdict='UNCUT',prefix_mm=0)
        result = measure(rgb,z,far=far,sigma=sigma,prefix=self.prefix)
        if alternate:
            self.spent = True
        if result['verdict']=='UNKNOWN':
            self.pending = True
            if self.spent:
                self.terminal = True
        else:
            self.prefix,self.closures = result['prefix_mm'],closures
            self.pending,self.terminal = False,result['verdict']=='TARGET'
        return result
