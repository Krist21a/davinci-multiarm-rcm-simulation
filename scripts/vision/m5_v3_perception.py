"""Stateless material-row chart and terminal certificate from the same print."""
from dataclasses import dataclass

import numpy as np

from bilateral_surface import _bank_plane, _ray
from m5_v3_corridor_v1 import bank_patch, inks
from m5_v3_corridor_v2 import side_stations
from m5_v3_corridor_v4 import measure as review


@dataclass(frozen=True)
class WorkChart:
    prefix_mm: int
    front: np.ndarray
    direction: np.ndarray
    opening: np.ndarray
    aim: np.ndarray
    limit: np.ndarray


def _station(label, z, seeds, j, side, f, far):
    point = seeds[j]
    if point is None:
        return None
    codes = [side*13+k for k in (j-1,j) if 0<=k<13]
    y,x = np.where(np.isin(label,codes)&np.isfinite(z)&(z>0)&(z<far*.999))
    coords = np.column_stack((y,x))-point
    near = (coords*coords).sum(1)<=9.
    if near.sum()<6 or not all(np.any(label[y[near],x[near]]==c) for c in codes):
        return None
    uv = point[::-1]+.5
    pp = np.array(z.shape[::-1],dtype=float)/2.
    fit = _bank_plane(label,z,codes,uv,pp,f,far,float(np.median(z[y[near],x[near]])))
    if fit is None:
        return None
    return _ray(uv,pp,f)*fit[0]


def work(rgb, depth, *, focal, far, sigma, virgin, goal_mm):
    """Camera coordinates. Never use a slit median, temporal prior or empty-bin hop."""
    if goal_mm not in (9,10) or not np.isfinite(focal) or focal<=0:
        return None
    r = review(rgb,depth,far=far,sigma=sigma)
    s = [row['status'] for row in r['rows']]
    if virgin:
        if len(s)!=13 or s[:3]!=['INTACT','INTACT','INTACT'] or any(a in ('OPEN','PARTIAL') for a in s):
            return None
        n = 0
    else:
        n = r['prefix_mm']
        if r['verdict'] not in ('BELOW','TARGET') or type(n) is not int or n>=goal_mm:
            return None
        if n>8 or s[n:n+3]!=['INTACT','INTACT','INTACT']:
            return None
    label = inks(rgb)
    yy,xx = np.where(label>=0)
    if not len(yy):
        return None
    y0,x0 = yy.min(),xx.min()
    crop = np.s_[y0:yy.max()+1,x0:xx.max()+1]
    b,d = label[crop],depth[crop]
    good = np.isfinite(d)&(d>0)&(d<far*.999)
    stations = []
    for side in range(2):
        masks = []
        for j in range(13):
            mask = np.zeros(d.shape,bool)
            p = bank_patch((b==side*13+j)&good)
            mask[tuple(p.T)] = True
            masks.append(mask)
        stations.append([None if q is None else q+[y0,x0] for q in side_stations(masks,side)])
    left,right = [],[]
    for j in range(n,11):
        p,q = (_station(label,depth,stations[side],j,side,focal,far) for side in range(2))
        if p is None or q is None:
            return None
        left.append(p)
        right.append(q)
    left,right = np.array(left),np.array(right)
    mid = (left+right)*.5
    d = mid[2]-mid[1]
    if not .00025<=np.linalg.norm(d)<=.0015:
        return None
    direction = d/np.linalg.norm(d)
    chords = right[1:3]-left[1:3]
    if ((np.linalg.norm(chords,axis=1)<.0004).any() or
            (np.linalg.norm(chords,axis=1)>.003).any() or (abs(chords@direction)>.00075).any()):
        return None
    opening = chords.mean(0)
    opening -= float(opening@direction)*direction
    if np.linalg.norm(opening)<.0004:
        return None
    spacing = np.diff(mid@direction)
    if not np.all((spacing>=.00025)&(spacing<=.0015)):
        return None
    axial = (mid-mid[1])@direction
    nearer = np.minimum((left-mid[1])@direction,(right-mid[1])@direction)
    if (np.diff(nearer)<=0.).any() or nearer[goal_mm-n]<=axial[0]:
        return None
    front = mid[1]+axial[0]*direction
    aim = mid[1]+nearer[goal_mm-n]*direction
    limit = mid[1]+nearer[10-n]*direction
    return WorkChart(n,front,direction,opening/np.linalg.norm(opening),aim,limit)
