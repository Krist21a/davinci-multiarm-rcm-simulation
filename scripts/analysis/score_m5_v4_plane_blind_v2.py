"""Score the fixed half-slope law and RGB parity independently, without fitting."""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision')]
import score_m5_v4_plane_v1 as prev
from capture_m5_v4_plane_blind_v2 import OUT, check, checked, load, old, sha256, write
from score_m5_v4_plane_v1 import COLORS, F, H, W, projected


def forward(z,n,d,f=F):
    grad = np.max(np.abs(np.asarray(n)[...,:2]),axis=-1)/(f*np.abs(d))
    u = 1./z-.5*grad
    check(np.isfinite(u).all() and (u>0).all(),'invalid forward geometry')
    return 1./u


def rate(good,take):
    n = int(np.count_nonzero(take))
    k = int(np.count_nonzero(take & good))
    return dict(n=n,good=k,fraction=k/n if n else None)


def parity_score(rgb,verts,legacy_shift):
    pts = projected(verts,(H,W),F).reshape(7,9,2)
    yy,xx = np.mgrid[max(0,int(pts[:,:,0].min())-1):min(H,int(pts[:,:,0].max())+2),
                      max(0,int(pts[:,:,1].min())-1):min(W,int(pts[:,:,1].max())+2)]
    rc = np.stack((yy,xx),axis=-1)
    colors = rgb[yy,xx].astype(int)
    obs = np.full(yy.shape,-1,dtype=int)
    for i,c in enumerate(COLORS):
        obs[np.max(abs(colors-c),axis=-1)<=2] = i
    # Read-back plane in board cell coordinates (i,j), independent of observed colors/depth.
    j,i = np.indices((7,9))
    A = np.linalg.lstsq(np.column_stack((i.ravel(),j.ravel(),np.ones(i.size))),verts,rcond=None)[0].T
    K = np.array([[0,-F,H*.5-.5],[-F,0,W*.5-.5],[0,0,1]])
    hom = np.dstack((rc,np.ones(yy.shape)))@np.linalg.inv(K@A).T
    ij = hom[:,:,:2]/hom[:,:,2,None]
    interior = ((ij>=[.5,.5])&(ij<=[7.5,5.5])).all(2)
    i,j = np.clip(np.floor(ij).astype(int),[0,0],[7,5]).transpose(2,0,1)
    expected = (i+j)%2
    corners = [pts[j,i],pts[j,i+1],pts[j+1,i+1],pts[j+1,i]]
    dist = np.full(yy.shape,np.inf)
    for a,b in zip(corners,corners[1:]+corners[:1],strict=True):
        v = b-a
        k = np.clip(((rc-a)*v).sum(2)/(v*v).sum(2),0.,1.)
        dist = np.minimum(dist,np.linalg.norm(rc-a-k[:,:,None]*v,axis=2))
    correct = (obs>=0)&((expected==obs)|(dist<=.02))
    junctions = cKDTree(pts[1:-1,1:-1].reshape(-1,2))
    starts = np.vstack((pts[1:6,0],pts[0,1:8]))
    v = np.vstack((pts[1:6,-1],pts[-1,1:8]))-starts
    normals = np.column_stack((-v[:,1],v[:,0]))/np.linalg.norm(v,axis=1)[:,None]
    records = []
    for step in ((0,1),(1,0)):
        dy,dx = step
        a,b = np.s_[:yy.shape[0]-dy,:yy.shape[1]-dx],np.s_[dy:,dx:]
        take = interior[a]&interior[b]
        x,y = rc[a][take],rc[b][take]
        mid = (x+y)*.5
        near = junctions.query(mid)[0]<=1.5
        observed = (obs[a][take]>=0)&(obs[b][take]>=0)&(obs[a][take]!=obs[b][take])
        predicted = expected[a][take]!=expected[b][take]
        missing = (obs[a][take]<0)|(obs[b][take]<0)
        good = correct[a][take]&correct[b][take]
        ids = np.abs(((mid[:,None]-starts)*normals).sum(2)).argmin(1)
        p = ((x+legacy_shift-starts[ids])*normals[ids]).sum(1)
        q = ((y+legacy_shift-starts[ids])*normals[ids]).sum(1)
        legacy = (np.minimum(p,q)<=.02)&(np.maximum(p,q)>=-.02)
        records.append(np.column_stack((near,observed,predicted,missing,good,legacy)))
    near,obs,pred,missing,good,legacy = np.concatenate(records).T
    union = obs|pred|missing
    out = dict(observed=rate(good,obs),union=rate(good,union),all_pairs=rate(good,np.ones_like(obs)),
               near=rate(good,near&union),away=rate(good,~near&union),undecoded_pairs=int(missing.sum()),
               legacy=rate(legacy,obs),legacy_rejected_but_both_colors_match=int((obs&~legacy&good).sum()),
               legacy_reject_near=int((obs&~legacy&near).sum()))
    out['passed'] = out['observed']['n']>=40 and all(out[k]['fraction']>=.98 for k in ('all_pairs','observed','union'))
    return out


def run():
    lock = load(OUT/'registration.json')
    for k,h in lock['sources'].items():
        checked(ROOT/k,h)
    cap = load(OUT/'capture.json')
    checked(OUT/'registration.json',cap['registration_sha256'])
    poses = lock['spec']['blind']
    check(cap['stationary'] and cap['port_free'] and cap['error'] is None and len(cap['frames'])==len(poses),
          'incomplete acquisition')
    rows = {}
    shift = np.array(load(old.OUT/'development.json')['rgb_shift'])
    for i,pose in enumerate(poses):
        name = f'blind_{i:02d}'
        m = cap['frames'][name]
        path = checked(OUT/m['file'],m['sha256'])
        v = prev.sample(path,pose)
        z = prev.depth(v['rc'],v['n'],v['d'],[0.]*4)
        model = forward(z,v['n'],v['d'])
        with np.load(path,allow_pickle=False) as f:
            rgb = parity_score(f['rgb'],f['vertices_camera'],shift)
        lag = prev.rgb_scores({name:v},shift)[name]
        check(lag['samples']==rgb['legacy']['n'] and abs(lag['bracketed']-rgb['legacy']['fraction'])<1e-12,
              'lost or changed legacy transition population')
        rows[name] = dict(native=prev.stats(v['z']-z),model=prev.stats(v['z']-model),rgb=rgb)
    depth_ok = all(r['model']['p95_abs_mm']<=.010 for r in rows.values())
    rgb_ok = all(r['rgb']['passed'] for r in rows.values())
    write(OUT/'blind_decision.json',dict(rows=rows,depth_passed=depth_ok,rgb_passed=rgb_ok,
          transfer_eligible=rgb_ok and depth_ok,capture_sha256=sha256(OUT/'capture.json'),
          registration_sha256=cap['registration_sha256'],authorizes_motion=False))
    print('blind gates: depth',depth_ok,'RGB',rgb_ok,flush=True)


if __name__=='__main__':
    run()
