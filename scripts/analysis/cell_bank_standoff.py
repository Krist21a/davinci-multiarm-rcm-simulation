"""Find intact candidate print hosts; measure, never assume, transfer to the inner edge."""
import argparse
import json
from pathlib import Path

import numpy as np

from capture_surface_seam import ROOT, sha256, write_new
from contact_veto_pilot import OUT as DEV
from score_cell_certificate import OUT


def host_counts(links,column,side):
    cell=column-1 if side=='left' else column
    count=np.zeros(14,int)
    for row in range(13):
        a=24*row+cell
        for tri in ((a,a+1,a+25),(a,a+25,a+24)):
            if all(links[tuple(sorted((tri[i-1],tri[i])))] for i in range(3)):
                for k in tri:
                    if k%24==column:
                        count[k//24]+=1
    return count


def main(folder):
    folder.mkdir(parents=True,exist_ok=False)
    plan=dict(scope='Secondary structural design screen after a failed camera test. Not perception or confirmation.',
        population='All five verified cutting trajectories, every closed working frame and each loaded/park frame; '
                   'plus both negative-control frames. Count unique acquired frames, not independent seeds.',
        prerequisite='For each of the 14 node rows, at least one outward-side incident triangle of the chosen '
                     'column must exist on EVERY stored structure; near-half and far-half candidates independent. '
                     'Test 0..4 one-column outward shifts from columns 11 and 12 (original +/-0.5 mm). '
                     'Nearest passing offset wins. Survival is necessary, not optical visibility.',
        transfer='For each positive-offset candidate audit the tempting rest-vector transfer. Predict each real inner '
                 'node from its candidate same-row outer node plus the exact rest offset measured on the fixed last '
                 'row. Report the worst 3D error and material-advance component on the 18 actual working frames; '
                 'derive the reference advance direction from ready TCP to centre of fixed last row (positive v). '
                 'Do not apply this estimate to the cutter even if an individual frame is close.',
        source_sha256=sha256(DEV/'decision.json'),camera_decision_sha256=sha256(OUT/'zoom/decision.json'),
        script_sha256=sha256(Path(__file__)))
    write_new(folder/'criteria.json',plan)
    dec=json.loads((DEV/'decision.json').read_bytes())
    details=[]
    for run in dec['runs']:
        src=DEV/'slide'/run['case']
        if not run['mechanics_ok'] or sha256(src/'oracle.json')!=run['oracle_sha256']:
            raise ValueError('unqualified trajectory')
        o=json.loads((src/'oracle.json').read_bytes())
        ready=np.array(o['ready_tcp'])[:,3]
        for fr in o['frames']:
            p=src/'structure'/(fr['key']+'.npz')
            if sha256(p)!=fr['structure_sha256']:
                raise ValueError('unmatched structure')
            with np.load(p) as data:
                raw=data['edges']
                e={tuple(sorted((a,b))):bool(alive) for a,b,alive in raw.tolist()}
                w=data['nodes']
                if len(e)!=len(raw) or w.shape!=(14,24,3) or not np.isfinite(w).all():
                    raise ValueError('invalid material state')
                v=w[-1].mean(axis=0)-ready
                if not np.isclose(np.linalg.norm(v),.015,atol=1e-6,rtol=0):
                    raise ValueError('not the registered flat pitch and ready setup')
                v=v/np.linalg.norm(v)
                for side,inner,sign in (('left',11,-1),('right',12,1)):
                    for off in range(5):
                        col=inner+sign*off
                        hosts=host_counts(e,col,side)
                        rest=w[-1,inner]-w[-1,col]
                        if not np.isclose(np.linalg.norm(rest),off*.001,atol=1e-8,rtol=0):
                            raise ValueError('reference row moved')
                        err=w[:,col]+rest-w[:,inner]
                        details.append(dict(case=run['case'],phase=fr['phase'],key=fr['key'],actual_mm=fr['prefix_mm'],
                            side=side,column=col,outward_steps=off,missing_rows=np.where(hosts==0)[0],
                            worst_3d_mm=float(np.linalg.norm(err,axis=1).max()*1000),
                            worst_advance_mm=float(np.abs(err@v).max()*1000)))
    result=[]
    for side in ('left','right'):
        candidates=[]
        for off in range(5):
            group=[x for x in details if x['side']==side and x['outward_steps']==off]
            work=[x for x in group if x['phase']=='work']
            candidates.append(dict(offset_steps=off,rest_distance_from_seam_mm=.5+off,
                sample_count=len(group),work_count=len(work),column=group[0]['column'],
                all_hosts=all(len(x['missing_rows'])==0 for x in group),
                samples_missing_hosts=sum(bool(len(x['missing_rows'])) for x in group),
                work_max_3d_mm=max(x['worst_3d_mm'] for x in work),
                work_max_advance_mm=max(x['worst_advance_mm'] for x in work)))
        result.append(dict(side=side,candidates=candidates,nearest=next((x for x in candidates if x['all_hosts']),None)))
    write_new(folder/'decision.json',dict(criteria_sha256=sha256(folder/'criteria.json'),results=result,details=details,
              qualified_visual_controller=False))
    print([(r['side'],r['nearest']) for r in result])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=OUT/'host_standoff')
    main(ap.parse_args().out)
