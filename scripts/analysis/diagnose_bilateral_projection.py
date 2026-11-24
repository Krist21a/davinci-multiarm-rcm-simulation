"""Post-hoc attribution only: replace the optical knots, not the measured depth or the reader."""
import argparse
from pathlib import Path

import numpy as np

from replay_bilateral_3d import ROOT, digest, load, write
from bilateral_surface import _boundary, _triangle
from ruler_readout import inks


def main(out):
    decision=load(out/'decision.json')
    if decision['passed']:
        raise ValueError('this attribution is for a failed development replay')
    src=ROOT/decision['criteria']['source']
    plan=dict(decision=digest(out/'decision.json'), source=digest(Path(__file__)),
              scope='Post-hoc only. All three actual terminal closed look-backs, native depth. For both ends of the last fully separated cell compare optical boundary to the SAME material node (central columns). Report error in pixels and mm. Evaluate the exact same triangle predicate with RGB-D knots and separately with scorer-only projected node positions/depths. No new cut readings, no controller input, no selection or threshold change.')
    write(out/'projection_plan.json',plan)
    cal=load(src/'a/calibration.json')
    f,far=cal['f'],cal['far']
    records=[]
    for c in ('a','b','c'):
        folder=src/c
        o=load(folder/'oracle.json')
        row=next(r for r in o['frames'] if r['phase']=='look')
        expected=next(r for r in decision['rows'] if r['case']==c and r['phase']=='look')
        hd,st=(folder/k/(row['key']+'.npz') for k in ('hd','structure'))
        assert digest(hd)==expected['sha256']==row['hd_sha256'] and digest(st)==row['structure_sha256']
        with np.load(hd) as data, np.load(st) as truth:
            lab,z=inks(data['rgb']),data['depth']
            xyz=(truth['nodes']-o['initial_view']['applied_t']) @ np.array(o['initial_view']['applied_R'])
        ys,xs=np.where(lab>=0)
        y,x=max(0,ys.min()-2),max(0,xs.min()-2)
        roi=np.s_[y:ys.max()+3,x:xs.max()+3]
        lab,z=lab[roi],z[roi]
        pp=np.array([cal['W']*.5-x,cal['H']*.5-y])
        mm=expected['mm']
        measured, ideal, offsets=[],[],[]
        for j in (mm-1,mm):
            clouds=[np.argwhere((lab==13*s+j) | (lab==13*s+j-1))[:,::-1] for s in (0,1)]
            delta=np.median(clouds[1],axis=0)-np.median(clouds[0],axis=0)
            for s in (0,1):
                q=xyz[j,11+s]
                uv=pp-f*q[:2]/q[2]
                v=_boundary(lab,z,j,s,(1-2*s)*delta,pp,f,far)
                ideal.append((uv,q))
                measured.append(v)
                offsets.append(None if v is None else dict(row=j,side=s,
                               pixel_error=float(np.linalg.norm(uv-v[0])),
                               mm_error=float(np.linalg.norm(q-v[1])*1000)))
        faces={}
        for kind,ends in (('optical',measured),('oracle_only',ideal)):
            states=[]
            for half,idx in enumerate(((0,1,3),(0,3,2))):
                tri=[ends[i] for i in idx]
                state='?'
                if all(v is not None for v in tri):
                    uv,q=(np.stack([a[i] for a in tri]) for i in (0,1))
                    state,_=_triangle(uv,q,mm-1+(13 if half==0 else 0),lab,z,far)
                states.append(state)
            faces[kind]=states
        records.append(dict(mm=mm,phase='look',sha256=digest(hd),boundary_errors=offsets,faces=faces))
    write(out/'projection_diagnostic.json',dict(plan_sha256=digest(out/'projection_plan.json'),rows=records))
    for r in records:
        print(r['mm'],r['faces'],[(round(v['pixel_error'],2),round(v['mm_error'],2))
                                  if v else None for v in r['boundary_errors']])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
