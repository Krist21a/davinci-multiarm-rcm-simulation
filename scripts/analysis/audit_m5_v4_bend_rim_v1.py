"""Offline ray/surface audit of the frozen row-10 estimator, never a motion input."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/d) for d in ('vision','task')]
import m5_v3_perception_v8 as rim
import m5_v3_perception_v11 as reader
from run_m5_v4_bend_terminal_v1 import OUT as PARENT, check, load, sha256, verify, write

ID = 'm5_v4_bend_rim_audit_v1'
PLAN = ROOT/f'config/task/{ID}.json'
OUT = ROOT/f'validation/task/{ID}'


def rays(rc,shape,focal):
    rc = np.asarray(rc)
    xy = ((np.array(shape)*.5-.5-rc)/focal)[...,::-1]
    return np.concatenate((xy,np.ones((*xy.shape[:-1],1))),axis=-1)


class Sheet:
    """First intersection in ideal camera geometry; does not model the rest of the scene."""
    def __init__(self,nodes,edges):
        alive = {tuple(sorted((a,b))):bit for a,b,bit in np.reshape(edges,(-1,3))}
        triangles,ids = [],[]
        for j in range(13):
            for i in range(23):
                a = 24*j+i
                for half,vertices in enumerate(((a,a+1,a+25),(a,a+25,a+24))):
                    if all(alive[tuple(sorted((v,vertices[k-1])))] for k,v in enumerate(vertices)):
                        ids.append((j,i,half))
                        triangles.append(nodes.reshape(-1,3)[list(vertices)])
        self.ids = np.asarray(ids)
        tri = np.asarray(triangles)
        self.a = tri[:,0]
        self.u,self.v = tri[:,1]-self.a,tri[:,2]-self.a
        self.normal = np.cross(self.u,self.v)
        self.n2 = (self.normal**2).sum(1)
        self.offset = (self.normal*self.a).sum(1)

    def hits(self,r):
        direction = np.atleast_2d(r)
        denom = direction@self.normal.T
        z = np.divide(self.offset,denom,out=np.zeros_like(denom),where=np.abs(denom)>1e-17)
        rel = z[:,:,None]*direction[:,None,:]-self.a
        nz = np.maximum(self.n2,1e-40)
        s = (np.cross(rel,self.v)*self.normal).sum(-1)/nz
        t = (np.cross(self.u,rel)*self.normal).sum(-1)/nz
        ok = (z>0)&(self.n2>1e-32)&(s>=-1e-9)&(t>=-1e-9)&(s+t<=1+1e-9)
        opt = np.where(ok,z,np.inf)
        k = opt.argmin(1)
        depth = opt[np.arange(len(k)),k]
        return depth,np.where(np.isfinite(depth[:,None]),self.ids[k],-1)


def tied_pairs(a,b,tol):
    check(len(a)>0 and len(b)>0,'missing native rim evidence')
    d2 = ((a[:,None,:2]-b[None,:,:2])**2).sum(-1)
    d2[np.abs(a[:,None,2]-b[None,:,2])>tol] = np.inf
    i,k = np.where((d2==d2.min())&(d2<=5.))
    check(len(i)>0,'missing row pair')
    return np.column_stack((i,k))


def fit_support(center,code,ctx,native_z):
    label,z,valid,pink,across,tol,sigma,_ = ctx
    check(sigma==0,'not native depth')
    ends = np.array([np.floor(center),np.ceil(center)],dtype=int)
    if label[tuple(ends[0])]!=code:
        ends = ends[::-1]
    a,b = ends
    check(label[tuple(a)]==code and pink[tuple(b)] and np.abs(a-b).sum()==1,'bad rim endpoints')
    along = np.array([-across[1],across[0]])
    extent = np.hypot(10.5*across,2.35*along)
    lo = np.maximum(0,np.ceil(center-extent).astype(int))
    hi = np.minimum(z.shape,np.floor(center+extent).astype(int)+1)
    roi = np.s_[lo[0]:hi[0],lo[1]:hi[1]]
    coords = (np.moveaxis(np.indices(z[roi].shape),0,-1)+lo-center)@np.column_stack((across/10.5,along/2.35))
    allowed = valid[roi]&(pink[roi]|((label[roi]>=0)&(label[roi]//13==code//13)))&((coords**2).sum(-1)<=1.)
    zz = z[roi].astype(float)
    start = tuple(a-lo)
    visited,queue = {start},[start]
    while queue:
        y,x = queue.pop()
        for p in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if (0<=p[0]<zz.shape[0] and 0<=p[1]<zz.shape[1] and allowed[p] and p not in visited
                    and abs(zz[p]-zz[y,x])<=tol):
                visited.add(p)
                queue.append(p)
    check(tuple(b-lo) in visited,'disconnected fit')
    ij = np.array(sorted(visited))
    idx = tuple(ij.T)
    u,v = coords[idx].T
    design = np.column_stack((np.ones(len(u)),u,v,v*v))
    w = np.linalg.lstsq(design.T,[1.,0.,0.,0.],rcond=None)[0]
    check(np.allclose(w@design,[1.,0.,0.,0.],atol=1e-10,rtol=0) and w@w<=.125
          and abs(1./(w@(1./zz[idx]))-native_z)<1e-13,'fit replay differs')
    return ij+lo,w,ends


def projected(p,shape,f):
    return np.array(shape)*.5-.5-f*p[...,1::-1]/p[...,2,None]


def station(side,ctx,sheet,nodes,pair,axis):
    label,z,valid,pink,across,tol,sigma,f = ctx
    a,b = rim._rim(label,z,valid,pink,side,across,tol,sigma)[9:11]
    pairs = tied_pairs(a,b,tol)
    center = (a[pairs[:,0],:2]+b[pairs[:,1],:2])*.5
    lines = rays(center,z.shape,f)
    outer = nodes[10,(7,16)[side]]
    truth = nodes[10,11:13].mean(0)
    fit_records,fit_ids,depths = [],{},[{}, {}, {}]
    pixel_set = {}
    max_residual = 0.
    misses = []
    for row,edges,indices in ((9,a,pairs[:,0]),(10,b,pairs[:,1])):
        for k in np.unique(indices):
            edge = edges[k]
            ij,w,ends = fit_support(edge[:2],13*side+row,ctx,edge[2])
            support,ids = sheet.hits(rays(ij,z.shape,f))
            middle,mid_id = sheet.hits(rays(edge[:2],z.shape,f))
            model_labels = np.where((ids[:,1]<7)|(ids[:,1]>15),-1,ids[:,0]+13*(ids[:,1]>11))
            obs = np.where(pink[tuple(ij.T)],-1,label[tuple(ij.T)])
            observed_z = z[tuple(ij.T)].astype(float)
            pixels = []
            for rc,wt,face,code,model,raw,ideal in zip(ij,w,ids,obs,model_labels,observed_z,support,strict=True):
                key = tuple(rc.tolist())
                finite = bool(np.isfinite(ideal))
                residual = 1000*(raw-ideal) if finite else None
                expected = None if not finite or face[1] in (10,11,12) else int(model)
                if finite:
                    max_residual = max(max_residual,abs(residual))
                pixel_set[key] = dict(rc=key,observed_bank=int(code),face=face,model_bank=expected,
                                      measured_z_m=raw,sheet_z_m=ideal if finite else None,
                                      residual_mm=residual)
                pixels.append(dict(rc=key,weight=float(wt)))
            idx = (row,k)
            depths[0][idx] = middle[0]
            depths[1][idx] = 1./(w@(1./support)) if np.isfinite(support).all() else np.inf
            depths[2][idx] = edge[2]
            if not np.isfinite(support).all() or not np.isfinite(middle[0]):
                misses.append((row,int(k)))
            fit_ids[idx] = len(fit_records)
            fit_records.append(dict(row=row,ends=ends,center=edge[:2],pixels=pixels,center_face=mid_id[0],
                                    native_z_m=edge[2],unique_support=len(w),squared_weight_norm=float(w@w)))
    combined,ids = sheet.hits(lines)
    stages = [outer,np.median(lines*combined[:,None],axis=0) if np.isfinite(combined).all() else None]
    for d in depths:
        d1,d2 = np.array([(d[9,i],d[10,k]) for i,k in pairs]).T
        stages.append(np.median(lines*(2./(1./d1+1./d2))[:,None],axis=0)
                      if np.isfinite([d1,d2]).all() else None)
    check(np.allclose(stages[-1],pair[side],atol=1e-12,rtol=0),'station median differs')
    terms = {key:float(1000*(v-u)@axis) if v is not None and u is not None else None
             for key,u,v in zip(('sampled_ray_on_sheet','edge_centers_and_pairing','fit_on_ideal_sheet',
                                 'measured_minus_sheet'),stages[:-1],stages[1:],strict=True)}
    parts = dict(rim_error_mm=float(1000*(pair[side]-outer)@axis),proxy_mm=float(1000*(outer-truth)@axis),
                 true_rim_rc=projected(outer,z.shape,f),station_rc=projected(pair[side],z.shape,f),
                 centroid_error_px=float(np.linalg.norm(projected(pair[side],z.shape,f)-projected(outer,z.shape,f))),
                 terms_mm=terms,pairs=len(pairs),fits=len(fit_records),unique_pixels=len(pixel_set),
                 max_measured_sheet_residual_mm=max_residual,
                 color_disagreements=sum(p['model_bank'] is not None and p['model_bank']!=p['observed_bank']
                                         for p in pixel_set.values()),
                 support_misses=sum(p['residual_mm'] is None for p in pixel_set.values()),
                 ambiguous_color=sum(p['model_bank'] is None for p in pixel_set.values()),
                 support_rows=dict(Counter(int(p['face'][0]) for p in pixel_set.values())),
                 fit_misses=misses,pair_ray_misses=int(np.count_nonzero(~np.isfinite(combined))))
    if all(v is not None for v in terms.values()):
        check(abs(sum(terms.values())-parts['rim_error_mm'])<1e-8,'axial terms do not close')
    return parts,dict(fits=fit_records,pixels=list(pixel_set.values()),pairs=[
        dict(a=fit_ids[9,i],b=fit_ids[10,k],rc=c,face=face) for (i,k),c,face in zip(pairs,center,ids,strict=True)])


def seal():
    plan = load(PLAN)
    prior = load(verify(PARENT/'decision.json',plan['decision_sha256']))
    attrib = load(verify(PARENT/'clean_attribution.json',plan['attribution_sha256']))
    check(attrib['decision_sha256']==plan['decision_sha256'] and len(attrib['rows'])==6,'wrong frame population')
    reg = load(verify(PARENT/'registration.json',prior['registration_sha256']))
    for name,pin in reg['sources'].items():
        verify(ROOT/name,pin)
    records = []
    for h,ref in attrib['rows'].items():
        case,name = ref['case'],ref['name']
        folder = PARENT/case
        r = load(verify(folder/'run.json',prior['run_hashes'][case]))
        m = r['captures'][name]
        prev = max((n for n in r['review_sidecars'] if r['captures'][n]['tick']<m['tick']),
                   key=lambda n:r['captures'][n]['tick'])
        q = r['captures'][prev]
        check(m['sha256']==h and m['closures']==q['closures'] and r['receipts'][prev]['prefix_mm'] in (7,8,9),
              'missing same-closure topology reference')
        fs = [m,r['scoring_sidecars'][name],r['review_sidecars'][prev]]
        for rec in fs:
            verify(folder/rec['file'],rec['sha256'])
        records.append(dict(case=case,name=name,h=h,review=prev,prefix=r['receipts'][prev]['prefix_mm'],
                            files=[x['sha256'] for x in fs]))
    inputs = [PLAN,Path(__file__).resolve(),ROOT/'scripts/tests/test_m5_v4_bend_rim_audit_v1.py',
              ROOT/'scripts/learn/capture_m5_temporal_coded.py',ROOT/'src/tissue/tissue.lua',
              ROOT/'src/tissue/cell_certificate_print.lua']
    return dict(plan=plan,records=records,sources={k.relative_to(ROOT).as_posix():sha256(k) for k in inputs})


def run():
    lock = seal()
    OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',lock)
    summary,details = {},{}
    prior = load(PARENT/'clean_attribution.json')['rows']
    for rec in lock['records']:
        h,case,name,n,review = (rec[k] for k in ('h','case','name','prefix','review'))
        folder = PARENT/case
        r = load(folder/'run.json')
        m = r['captures'][name]
        with (np.load(verify(folder/m['file'],rec['files'][0]),allow_pickle=False) as f,
              np.load(verify(folder/r['scoring_sidecars'][name]['file'],rec['files'][1]),allow_pickle=False) as s,
              np.load(verify(folder/r['review_sidecars'][review]['file'],rec['files'][2]),allow_pickle=False) as e):
            z,rgb = f['depth'],f['rgb']
            args = dict(focal=f['focal'].item(),far=f['far'].item(),sigma=0.,prefix_mm=n,goal_mm=10 if n==9 else 9)
            (g,row,pair,_),_,recover = reader._solve_work(rgb,z,**args)
            _,ctx = reader._observe(rgb,z,**args)
            side = int(np.argmin(pair@g.direction))
            check(not recover and row==10 and side==prior[h]['side'],'not the original main path')
            check(s['frame_sha256'].item()==h and s['tick'].item()==m['tick'] and np.array_equal(f['q'],s['q'])
                  and s['closed'].item()==m['closed'] and e['frame_sha256'].item()==r['captures'][review]['sha256']
                  and e['closures'].item()==m['closures'] and e['stamp'].item()<=s['simulation_time_s'].item(),
                  'unbound scoring data')
            nodes = (s['nodes_world']-f['cam_t'])@f['cam_R']
            sheet = Sheet(nodes,e['edges'])
            stats,pixels = zip(*(station(i,ctx,sheet,nodes,pair,g.direction) for i in (0,1)),strict=True)
            tip = 1000*((s['tcp_world']-f['cam_t'])@f['cam_R']-m['marker']['tip'])@g.direction
            v = stats[side]
            optimism = v['rim_error_mm']+v['proxy_mm']+tip
            check(abs(optimism-prior[h]['total_mm'])<1e-8,'original optimism mismatch')
            summary[h] = dict(case=case,name=name,side=side,tip_mm=tip,optimism_mm=optimism,banks=stats)
            details[h] = pixels
    write(OUT/'support.json',details)
    write(OUT/'decision.json',dict(registration_sha256=sha256(OUT/'registration.json'),
          support_sha256=sha256(OUT/'support.json'),frames=summary,scope='diagnostic',authorizes_motion=False))
    for x in summary.values():
        b = x['banks'][x['side']]
        print(x['case'],x['name'],b['terms_mm'],'residual max mm',b['max_measured_sheet_residual_mm'])


if __name__=='__main__':
    run()
