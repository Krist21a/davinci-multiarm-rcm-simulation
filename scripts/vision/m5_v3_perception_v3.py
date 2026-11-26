"""Work-frame outer bank coordinates; narrow completion still uses the sealed v4 reader."""
import numpy as np

from m5_v3_perception import WorkChart, inks, review

PINK = np.array([219, 107, 122], dtype=np.uint8)


def _good(depth, far):
    return np.isfinite(depth) & (depth>0) & (depth<far*.999)


def _rim(label, depth, valid, pink, side, across, tol):
    points = np.argwhere(valid & (label//13==side) & (label>=0))
    rows = [[] for _ in range(13)]
    for step in ((0,1),(0,-1),(1,0),(-1,0)):
        off = np.asarray(step)
        if float(off@across)*(1 if side else -1)<.25:
            continue
        out = points+off
        ok = ((out>=0)&(out<depth.shape)).all(1)
        a,b = points[ok],out[ok]
        i,k = tuple(a.T),tuple(b.T)
        keep = pink[k] & (abs(depth[i]-depth[k])<=tol)
        a,b = a[keep],b[keep]
        i,k = tuple(a.T),tuple(b.T)
        if not len(a):
            continue
        z = 2./(1./depth[i]+1./depth[k])
        edge = np.column_stack(((a+b)*.5,z))
        for j in np.unique(label[i]%13):
            rows[j].extend(edge[label[i]%13==j])
    return [np.asarray(p).reshape(-1,3) for p in rows]


def _stations(rim, depth_shape, focal, tol):
    cols = {}
    if len(rim[0]) and len(rim[-1]):
        axis = np.median(rim[-1][:,:2],axis=0)-np.median(rim[0][:,:2],axis=0)
        if np.linalg.norm(axis)>1e-6:
            rank = rim[0][:,:2]@axis
            cols[0] = rim[0][rank==rank.min()]
    for j in range(1,13):
        a,b = rim[j-1],rim[j]
        if not len(a) or not len(b):
            continue
        d2 = np.sum((a[:,None,:2]-b[None,:,:2])**2,axis=-1)
        d2[abs(a[:,None,2]-b[None,:,2])>tol] = np.inf
        i,k = np.where((d2==d2.min()) & (d2<=5.))
        if len(i):
            cols[j] = np.column_stack(((a[i,:2]+b[k,:2])*.5,2./(1./a[i,2]+1./b[k,2])))
    result = {}
    for j,p in cols.items():
        z = p[:,2]
        rc = (np.array(depth_shape)*.5-p[:,:2]-.5)*z[:,None]/focal
        xyz = np.column_stack((rc[:,1],rc[:,0],z))
        median = np.median(xyz,axis=0)
        if np.max(np.linalg.norm(xyz-median,axis=1))<=.00035:
            result[j] = median
    return result


def work(rgb, depth, *, focal, far, sigma, prefix_mm, goal_mm):
    if (type(prefix_mm) is not int or goal_mm not in (9,10) or not 0<=prefix_mm<goal_mm
            or depth.ndim!=2 or rgb.dtype!=np.uint8 or rgb.shape!=depth.shape+(3,)
            or not np.isfinite([focal,far,sigma]).all() or min(focal,far)<=0 or sigma<0):
        return None
    n = prefix_mm
    label = inks(rgb)
    valid = _good(depth,far)
    pink = valid & (rgb==PINK).all(-1)
    pix = [np.argwhere(valid & (label>=0) & (label//13==s)) for s in range(2)]
    if not all(len(p) for p in pix):
        return None
    across = pix[1].mean(0)-pix[0].mean(0)
    if np.linalg.norm(across)<1e-6:
        return None
    across /= np.linalg.norm(across)
    tol = .001+3*sigma
    sides = [_stations(_rim(label,depth,valid,pink,s,across,tol),depth.shape,focal,tol) for s in range(2)]
    pairs = {j:np.stack((sides[0][j],sides[1][j])) for j in sides[0].keys() & sides[1].keys()}
    if not {n+1,n+2}<=pairs.keys():
        return None
    first,second = pairs[n+1].mean(0),pairs[n+2].mean(0)
    step = np.linalg.norm(second-first)
    if not .00025<=step<=.0015:
        return None
    direction = (second-first)/step
    checked = {}
    for j,p in pairs.items():
        chord = p[1]-p[0]
        if .0065<=np.linalg.norm(chord)<=.0115 and abs(chord@direction)<=.00075:
            checked[j] = p
    if not {n+1,n+2}<=checked.keys():
        return None
    opening = (checked[n+1][1]-checked[n+1][0]+checked[n+2][1]-checked[n+2][0])*.5
    opening -= (opening@direction)*direction
    if np.linalg.norm(opening)<.0065:
        return None
    if n<=5:
        bound = max((j for j in checked if n+2<=j<=8), default=None)
        if bound is None:
            return None
        aim_row = limit_row = bound
    elif {goal_mm,10}<=checked.keys():
        aim_row,limit_row = goal_mm,10
    else:
        return None
    axial = float((checked[n].mean(0)-first)@direction) if n in checked else -step
    aim = float(np.min((checked[aim_row]-first)@direction))
    limit = float(np.min((checked[limit_row]-first)@direction))
    if not -.0015<=axial<=-.00025 or not axial<aim<=limit:
        return None
    return WorkChart(n,first+axial*direction,direction,opening/np.linalg.norm(opening),
                     first+aim*direction,first+limit*direction)


def intact_preflight(rgb, depth, *, far):
    """Uncut entrance-clear check only, never used for post-closure verdicts."""
    bank = inks(rgb)
    valid = _good(depth,far)
    counts = np.bincount(bank[valid & (bank>=0)],minlength=26)
    optical = review(rgb,depth,far=far,sigma=0.)
    prefix = [r['status'] for r in optical['rows'][:9]]
    return dict(passed=bool((counts>=8).all() and prefix==['INTACT']*9),
                bank_px=counts.tolist(),entrance=prefix,optical=optical)
