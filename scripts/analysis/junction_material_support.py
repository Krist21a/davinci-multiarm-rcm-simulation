"""Scorer only: is a supposed colour junction actually part of the rendered sheet?"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from replay_bilateral_3d import ROOT, digest, load, write
from seam_topology import material_state
from workspace import NU, NV


def facets(triples):
    edges={tuple(sorted((int(a),int(b)))):live for a,b,live in triples}
    alive={}
    for j in range(NV-1):
        for i in range(NU-1):
            a=j*NU+i
            for h,tri in enumerate(((a,a+1,a+NU+1),(a,a+NU+1,a+NU))):
                if all(edges[tuple(sorted(e))] for e in combinations(tri,2)):
                    side=int(i>11 or (i==11 and h==0))
                    alive[j,i,h]=(tri,side*13+j if 7<=i<=15 else -1)
    return alive


def evidence(alive,j,side):
    col=11+side
    node=j*NU+col
    take={key:v for key,v in alive.items() if node in v[0]}
    codes=[13*side+j-1,13*side+j]
    shared=[]
    for ta,ca in take.values():
        for tb,cb in take.values():
            if [ca,cb]==codes:
                edge=set(ta)&set(tb)
                if len(edge)==2:
                    shared.append(sorted(edge))
    return dict(row=j,side=side,col=col,live_incident_faces=[list(k) for k in take],
                live_incident_codes=sorted({c for _,c in take.values()}),
                has_both_inks=all(k in [c for _,c in take.values()] for k in codes),
                live_same_bank_ab_edges=shared)


def run(out):
    result=load(out/'m5_decision.json')
    reg=load(out/'registration.json')
    spec=reg['criteria']
    src=ROOT/spec['source']
    lua=ROOT/'src/tissue/tissue.lua'
    if digest(lua)!=load(src/'registration.json')['sources']['src/tissue/tissue.lua']:
        raise ValueError('renderer mismatch')
    plan=dict(decision_sha256=digest(out/'m5_decision.json'),script_sha256=digest(Path(__file__)),
              renderer_sha256=digest(lua),scope='Post-scoring attribution only, raw work AND look 8/9/10. Reconstruct every surviving face with the exact three-spring and fixed-ink rendering rules; validate central faces against saved table and prefix against material_state. For all four terminal nodes count incident live faces, expected colours and common edges. Also compare every non-null reported optical point to reference and its immediate same-row OUTBOARD neighbour. Lack of shared material edge means the assumed colour boundary does not reach this physical node; existence does not prove visibility. Predictions and control unchanged.')
    write(out/'material_support_plan.json',plan)
    predictions=load(out/'predictions.json')['images']
    all_rows=[]
    for row in result['terminal_corners']:
        c,mm,sha=row['case'],row['mm'],row['sha256']
        folder=src/c
        if digest(folder/'oracle.json')!=spec['oracles'][c]:
            raise ValueError('wrong mechanical trace')
        o=load(folder/'oracle.json')
        fr=next(f for f in o['frames'] if f['hd_sha256']==sha and f['phase']==row['phase'])
        path=folder/'structure'/(fr['key']+'.npz')
        if digest(path)!=fr['structure_sha256']:
            raise ValueError('wrong structure')
        with np.load(path,allow_pickle=False) as st:
            state=material_state(st['edges'])
            alive=facets(st['edges'])
            center=[[(j,11,h) in alive for h in (0,1)] for j in range(13)]
            if state['prefix_rows']!=mm or not np.array_equal(st['faces'],center):
                raise ValueError('cross-check of rendered faces failed')
            xyz=(st['nodes']-o['initial_view']['applied_t']) @ np.array(o['initial_view']['applied_R'])
        cut=predictions[sha][0]['cut']
        points=[]
        for j in (mm-1,mm):
            for side,e in enumerate(cut.get('endpoints',{}).get(str(j),[None,None])):
                p=evidence(alive,j,side)
                col=p['col']
                if e:
                    p['raw_error_mm']=float(1000*np.linalg.norm(e['camera']-xyz[j,col]))
                    p['to_outboard_neighbour_mm']=float(1000*np.linalg.norm(e['camera']-xyz[j,col+2*side-1]))
                points.append(p)
        all_rows.append(dict(case=c,phase=row['phase'],mm=mm,sha256=sha,points=points))
    write(out/'material_support.json',dict(plan_sha256=digest(out/'material_support_plan.json'),rows=all_rows))
    for r in all_rows:
        if r['phase']=='look':
            print(r['mm'],[(p['row'],p['side'],len(p['live_incident_faces']),len(p['live_same_bank_ab_edges']),
                  round(p.get('to_outboard_neighbour_mm',-1),3)) for p in r['points']])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    run(ap.parse_args().out)
