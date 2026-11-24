"""Optimistic visibility: real candidate camera origins, no tools, sample only living faces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_sheet_observability import surface

ROOT = Path(__file__).resolve().parents[2]


def inspect(frame, R, t):
    rows = surface(frame['tissue_alive'])
    xyz = ((frame['nodes'].reshape(-1,3)-t) @ R)[rows[:,1:]]
    xy = xyz[:,:,:2]/xyz[:,:,2:]
    a,b = xy[:,1]-xy[:,0], xy[:,2]-xy[:,0]
    cross = a[:,0]*b[:,1]-b[:,0]*a[:,1]
    codes = []
    for row, v0, v1, v2 in rows:
        i = v0%24
        s = int(i>11 or (i==11 and v1==v0+1))
        codes.append(s*13+row if 7 <= i <= 15 else -1)
    codes=np.array(codes)
    # Oversampled z-buffer, twice the linear resolution of native RGB-D. This
    # ignores all other scene objects and does not treat a missing pixel as proof
    # of a cut. Overlap is handled by perspective-correct reciprocal depth.
    angle=np.deg2rad(70)
    stats=[]
    for scale in (1,2,4):
        W,H=640*scale,480*scale
        f=W/2/np.tan(angle/2)
        px=np.array([W/2,H/2])-f*xy
        lo=np.maximum(np.floor(px.min(axis=(0,1))).astype(int)-2,0)
        hi=np.minimum(np.ceil(px.max(axis=(0,1))).astype(int)+2,[W,H])
        w,h=hi-lo
        z=np.full((h,w),np.inf)
        label=np.full((h,w),-1)
        for k,q in enumerate(px-lo):
            if abs(cross[k])<1e-14:
                continue
            lx,ly=np.maximum(np.floor(q.min(axis=0)).astype(int),0)
            hx,hy=np.minimum(np.ceil(q.max(axis=0)).astype(int)+1,[w,h])
            if hx<=lx or hy<=ly:
                continue
            y,x=np.mgrid[ly:hy,lx:hx]
            dx,dy=(q[0,0]-x-.5)/f,(q[0,1]-y-.5)/f
            u=(dx*b[k,1]-dy*b[k,0])/cross[k]
            v=(a[k,0]*dy-a[k,1]*dx)/cross[k]
            dep=1/((1-u-v)/xyz[k,0,2]+u/xyz[k,1,2]+v/xyz[k,2,2])
            ok=(u>=0)&(v>=0)&(u+v<=1)&(dep<z[y,x])
            z[y[ok],x[ok]]=dep[ok]
            label[y[ok],x[ok]]=codes[k]
        counts=np.bincount((label+1).ravel(), minlength=27)[1:]
        stats.append(dict(scale=scale,left=counts[:13].tolist(),right=counts[13:].tolist()))
    return stats


def main(out):
    d=ROOT/'validation/task'
    fp=ROOT/'validation/perception/frames/m5b_v1_10_marked_truth/tick_016.npz'
    hs=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    study=json.loads((ROOT/'config/task/m5b_ecm_view_study_v1.json').read_bytes())
    result=[]
    with np.load(fp) as frame:
        if frame['cut_length']!=.010:
            raise ValueError('wrong reference')
        for i,target in enumerate(study['candidates_deg_deg_m']):
            p=d/f'm5b_ecm_view_study_v1_c{i:02}.json'
            r=json.loads(p.read_bytes())
            read=r.get('initial_view_readback')
            if not read or not np.allclose(read['target'],target,atol=1e-6,rtol=0):
                continue
            result.append(dict(target=target,source=hs(p),
                               visible=inspect(frame,np.array(read['applied_R']),np.array(read['applied_t']))))
    with out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(dict(reference_sha256=hs(fp),code_sha256=hs(Path(__file__)),candidates=result,
                       scope='Scoring-only sheet raster. No instruments/occluders, single historical frozen 10 mm state, not live evidence or a proof over all poses.'),f,indent=2)
        f.write('\n')
    for v in result:
        print(v['target'], [(c['scale'],min(min(c['left'][7:11]),min(c['right'][7:11]))) for c in v['visible']])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    main(p.parse_args().out)
