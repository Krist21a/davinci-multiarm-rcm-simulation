"""Fit on rigid development poses, freeze, then measure independent held-out poses."""
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision')]
from audit_m5_v4_bend_rim_v1 import PARENT, Sheet, projected, rays
from capture_m5_v4_plane_v1 import AUDIT, COLORS, F, H, OUT, W, check, grid, load, sha256, write

MODELS = dict(native=(),ray=(0,1),affine=(2,3),joint=(0,1,2,3))
BOUNDS = np.array([2.,2.,.02,5.])


def checked(path,h):
    check(sha256(path)==h,'seal mismatch: '+str(path))
    return path


def depth(rc,n,d,params):
    dr,dc,gain,b = params
    xy = (([H*.5-.5,W*.5-.5]-np.asarray(rc)-[dr,dc])/F)[:,::-1]
    return (1+gain)*d/(xy[:,0]*n[0]+xy[:,1]*n[1]+n[2])+b


def edge_constraints(rgb,vertices,inv):
    yy,xx = np.nonzero(rgb>=0)
    lo = np.maximum([0,0],[yy.min()-1,xx.min()-1])
    crop = rgb[lo[0]:yy.max()+2,lo[1]:xx.max()+2]
    projected_grid = projected(vertices,(H,W),F).reshape(7,9,2)
    start = np.vstack((projected_grid[1:6,0],projected_grid[0,1:8]))
    end = np.vstack((projected_grid[1:6,-1],projected_grid[-1,1:8]))
    v = end-start
    norm = np.column_stack((-v[:,1],v[:,0]))/np.linalg.norm(v,axis=1)[:,None]
    N,low,high = [],[],[]
    for step in ((1,0),(0,1)):
        dy,dx = step
        c = crop[:crop.shape[0]-dy,:crop.shape[1]-dx]
        b = crop[dy:,dx:]
        rc = np.transpose(np.nonzero((c>=0)&(b>=0)&(c!=b)))+lo
        ends = rc+step
        def inside(x):
            local = np.column_stack((x,np.ones(len(x))))@inv.T
            xy = local[:,:2]/local[:,2,None]
            return (np.abs(xy)<=[.014,.010]).all(1)
        keep = inside(rc)&inside(ends)
        rc,ends = rc[keep],ends[keep]
        mid = (rc+ends)*.5
        k = np.abs(((mid[:,None]-start)*norm).sum(2)).argmin(1)
        n = norm[k]
        p,q = np.sum((start[k]-rc)*n,axis=1),np.sum((start[k]-ends)*n,axis=1)
        N.extend(n)
        low.extend(np.minimum(p,q))
        high.extend(np.maximum(p,q))
    return np.asarray(N),np.array(low),np.array(high)


def sample(path,pose):
    with np.load(path,allow_pickle=False) as f:
        rgb,z,verts = f['rgb'],f['depth'],f['vertices_camera']
        check(z.shape==(H,W) and rgb.shape==(H,W,3) and f['focal']==F and np.array_equal(f['pose'],pose),
              'unexpected optics or pose')
        check(np.max(np.linalg.norm(f['mesh_vertices_camera']-verts[f['mesh_indices']],axis=1))<2e-6,
              'two checker colors not coincident')
    g = grid()
    mean = verts.mean(0)
    U,_,V = np.linalg.svd(g.T@(verts-mean))
    fix = np.diag([1,1,np.linalg.det(U@V)])
    A = U@fix@V
    check(np.max(np.linalg.norm(g@A+mean-verts,axis=1))<2e-6,'nonrigid board')
    n = A[2]
    d = 1000*(n@mean)
    cells = np.full(z.shape,-1,dtype=np.int8)
    for k,color in enumerate(COLORS):
        cells[np.max(np.abs(rgb.astype(np.int16)-color),axis=2)<=2] = k
    ok = (ndimage.distance_transform_edt(cells==0)>3)|(ndimage.distance_transform_edt(cells==1)>3)
    rc = np.transpose(np.nonzero(ok[::5,::5]))*5
    measured = z[tuple(rc.T)].astype(float)*1000
    check(len(rc)>=64 and np.isfinite(measured).all() and (measured>0).all(),'insufficient valid target depths')
    # Pixel projection of a board point is homogeneous in its local x,y,1 coordinates.
    K = np.array([[0,-F,H*.5-.5],[-F,0,W*.5-.5],[0,0,1]])
    inv = np.linalg.inv(K@np.stack((A[0],A[1],mean)).T)
    edges = edge_constraints(cells,verts,inv)
    check(len(edges[0])>=40,'insufficient measured color edges')
    return dict(rc=rc,n=n,d=d,z=measured,edges=edges)


def stats(res):
    return dict(samples=len(res),median_mm=float(np.median(res)),p95_abs_mm=float(np.percentile(abs(res),95)),
                max_abs_mm=float(np.max(abs(res))))


def depth_scores(data,params):
    return {k:stats(v['z']-depth(v['rc'],v['n'],v['d'],params)) for k,v in data.items()}


def fit_depth(data,model):
    indices = list(MODELS[model])
    def unpack(x):
        p = np.zeros(4)
        p[indices] = x
        return p
    def fun(x):
        p = unpack(x)
        return np.concatenate([(v['z']-depth(v['rc'],v['n'],v['d'],p))/np.sqrt(len(v['z'])) for v in data.values()])
    if indices:
        bound = BOUNDS[indices]
        fit = least_squares(fun,np.zeros(len(indices)),bounds=(-bound,bound),xtol=1e-13,ftol=1e-13,gtol=1e-13)
        check(fit.success,'nonconvergent dev fit')
        p = unpack(fit.x)
    else:
        p = np.zeros(4)
    scores = depth_scores(data,p)
    return dict(params=p.tolist(),poses=scores,passes=all(v['p95_abs_mm']<=.010 for v in scores.values())
                and not np.any(np.isclose(abs(p),BOUNDS,rtol=0,atol=1e-6)))


def bracket(v,shift):
    n,lo,hi = v['edges']
    x = n@shift
    return np.maximum(lo-x,0)+np.maximum(x-hi,0)


def rgb_scores(data,p):
    return {k:dict(samples=len(v['edges'][0]),bracketed=float(np.mean(bracket(v,p)<=.02))) for k,v in data.items()}


def transfer(params):
    dec = load(AUDIT/'decision.json')
    lock = load(checked(AUDIT/'registration.json',dec['registration_sha256']))
    support = load(checked(AUDIT/'support.json',dec['support_sha256']))
    dr,dc,gain,b = params
    rows = {}
    for rec in lock['records']:
        folder = PARENT/rec['case']
        h,name = rec['h'],rec['name']
        r = load(folder/'run.json')
        with (np.load(checked(folder/r['captures'][name]['file'],h)) as f,
              np.load(checked(folder/r['scoring_sidecars'][name]['file'],rec['files'][1])) as s,
              np.load(checked(folder/r['review_sidecars'][rec['review']]['file'],rec['files'][2])) as e):
            sheet = Sheet((s['nodes_world']-f['cam_t'])@f['cam_R'],e['edges'])
            banks = []
            for bank in support[h]:
                rc = np.array([x['rc'] for x in bank['pixels']])
                z = f['depth'][tuple(rc.T)].astype(float)*1000
                raw,_ = sheet.hits(rays(rc,(H,W),F))
                shifted,_ = sheet.hits(rays(rc+[dr,dc],(H,W),F))
                check(np.isfinite(raw).all() and np.isfinite(shifted).all(),'unscorable transfer ray')
                banks.append(dict(native=stats(z-1000*raw),model=stats(z-(1+gain)*1000*shifted-b)))
            rows[h] = dict(case=rec['case'],name=name,banks=banks)
    return rows


def run():
    lock = load(OUT/'registration.json')
    for k,h in lock['sources'].items():
        checked(ROOT/k,h)
    cap = load(OUT/'capture.json')
    checked(OUT/'registration.json',cap['registration_sha256'])
    check(cap['port_free'],'capture cleanup failed')
    def read(split):
        poses,files = lock['spec'][split],cap['frames'][split]
        check(len(files)==len(poses),'missing poses')
        out = {}
        for i,pose in enumerate(poses):
            name = f'{split}_{i:02d}'
            m = files[name]
            out[name] = sample(checked(OUT/m['file'],m['sha256']),pose)
        return out
    dev = read('development')
    models = {k:fit_depth(dev,k) for k in MODELS}
    chosen = next((k for k,v in models.items() if v['passes']),None)
    rgb = least_squares(lambda p:np.r_[np.concatenate([bracket(v,p)/np.sqrt(len(v['edges'][0]))
                        for v in dev.values()]),1e-6*p],[0.,0.],bounds=(-2.,2.),gtol=1e-13)
    check(rgb.success,'RGB fit failed')
    rgb_dev = rgb_scores(dev,rgb.x)
    seal = OUT/'development.json'
    write(seal,dict(models=models,chosen=chosen,rgb_shift=rgb.x.tolist(),rgb_dev=rgb_dev,
                    capture_sha256=sha256(OUT/'capture.json')))
    frozen = load(seal)
    held = read('holdout')
    scores = {k:depth_scores(held,m['params']) for k,m in frozen['models'].items()}
    cs = rgb_scores(held,frozen['rgb_shift'])
    rgb_ok = all(v['bracketed']>=.98 for v in [*rgb_dev.values(),*cs.values()]) and np.max(abs(rgb.x))<2-1e-6
    depth_ok = chosen is not None and all(v['p95_abs_mm']<=.010 for v in scores[chosen].values())
    tr = transfer(frozen['models'][chosen]['params']) if rgb_ok and depth_ok else None
    write(OUT/'decision.json',dict(development_sha256=sha256(seal),registration_sha256=cap['registration_sha256'],
          holdout=scores,rgb_holdout=cs,rgb_passed=bool(rgb_ok),depth_passed=bool(depth_ok),chosen=chosen,
          transfer=tr,authorizes_motion=False))
    print('chosen',chosen,'frozen models',[(k,v['params']) for k,v in models.items()],'RGB',rgb.x,'holdout gates',
          rgb_ok,depth_ok,'transfer frames',len(tr or {}))


if __name__=='__main__':
    run()
