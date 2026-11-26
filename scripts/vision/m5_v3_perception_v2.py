"""Stateless material-row chart and terminal certificate from the same print."""
import numpy as np

from m5_v3_perception import WorkChart, _station, bank_patch, inks, side_stations, review


def work(rgb, depth, *, focal, far, sigma, prefix_mm, goal_mm):
    if (type(prefix_mm) is not int or goal_mm not in (9,10) or not 0<=prefix_mm<goal_mm
            or depth.ndim!=2 or rgb.dtype!=np.uint8 or rgb.shape!=depth.shape+(3,)
            or not np.isfinite([focal,far,sigma]).all() or min(focal,far)<=0 or sigma<0):
        return None
    n = prefix_mm
    label = inks(rgb)
    yy,xx = np.where(label>=0)
    if not len(yy):
        return None
    y0,x0 = yy.min(),xx.min()
    crop = np.s_[y0:yy.max()+1,x0:xx.max()+1]
    b,z = label[crop],depth[crop]
    good = np.isfinite(z)&(z>0)&(z<far*.999)
    stations = []
    for side in range(2):
        masks = []
        for j in range(13):
            mask = np.zeros(z.shape,bool)
            p = bank_patch((b==side*13+j)&good)
            mask[tuple(p.T)] = True
            masks.append(mask)
        stations.append([None if q is None else q+[y0,x0] for q in side_stations(masks,side)])
    pairs = {}
    for j in sorted({n,n+1,n+2,goal_mm,10}):
        a,b = (_station(label,depth,stations[s],j,s,focal,far) for s in range(2))
        if a is not None and b is not None:
            pairs[j] = np.array([a,b])
    if not {n+1,n+2,goal_mm,10}<=pairs.keys():
        return None
    first,second = pairs[n+1].mean(0),pairs[n+2].mean(0)
    step = np.linalg.norm(second-first)
    if not .00025<=step<=.0015:
        return None
    direction = (second-first)/step
    chords = [p[1]-p[0] for j,p in pairs.items() if j!=n]
    if any(not .0004<=np.linalg.norm(c)<=.003 or abs(c@direction)>.00075 for c in chords):
        return None
    opening = np.mean([p[1]-p[0] for p in (pairs[n+1],pairs[n+2])],axis=0)
    opening -= (opening@direction)*direction
    if np.linalg.norm(opening)<.0004:
        return None
    # If the boundary itself is occluded, only the approach gets a one-row estimate.
    axial = float((pairs[n].mean(0)-first)@direction) if n in pairs else -step
    aim = float(np.min((pairs[goal_mm]-first)@direction))
    limit = float(np.min((pairs[10]-first)@direction))
    if not -.0015<=axial<=-.00025 or not axial<aim<=limit:
        return None
    return WorkChart(n,first+axial*direction,direction,opening/np.linalg.norm(opening),
                     first+aim*direction,first+limit*direction)
