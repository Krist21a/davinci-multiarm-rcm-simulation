"""Aggregate sealed per-case decisions; describe optical refusals without changing any gates."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision')]
from cut_perception import _ray_sphere_centre
import m5_v3_perception_v11 as v11
from run_m5_v4_hinge_closed_loop_v1 import OUT, check, load, lock, marker_reader, sensor, sha256, verify, write
from score_m5_v4_hinge_closed_loop_v1 import model


def stopped_frame(folder,r):
    if r['state']=='DONE':
        return None
    name,m = max(r['captures'].items(),key=lambda kv:kv[1]['tick'])
    if 'work_prefix' not in m:
        return None
    with np.load(verify(folder/m['file'],m['sha256'])) as f:
        rgb,z = sensor(f['rgb'],f['depth'],m['tick'],m['sigma'])
        focal,far = f['focal'].item(),f['far'].item()
    n = m['work_prefix']
    sol,sides,_ = v11._solve_work(rgb,z,focal=focal,far=far,sigma=m['sigma'],prefix_mm=n,goal_mm=9)
    result = dict(frame=name,closed=m['closed'],prefix=n,v11_chart=sol is not None)
    if sides is not None and sol is None:
        wide = {j:np.stack([s[j] for s in sides]) for j in sides[0].keys() & sides[1].keys()
                if .0065<=np.linalg.norm(sides[1][j]-sides[0][j])<=.0115}
        js = sorted(wide)
        if len(js)>=3 and {n+1,n+2}<=wide.keys():
            off = np.array(js)-np.mean(js)
            axis = off@np.stack([wide[j].mean(0) for j in js])
            axis /= np.linalg.norm(axis)
            result.update(axis_accepted=v11._direction(wide,n) is not None,first_fit_skew_mm={
                j:1000*abs((wide[j][1]-wide[j][0])@axis) for j in (n+1,n+2)})
    reader = marker_reader(model(),focal,far)
    details = {}
    for key,color in zip(('distal','proximal','third'),reader.marker_colours,strict=True):
        c,_,why,pts = reader._marker(rgb.astype(float),z,color)
        vals = dict(initial_refusal=why,pixels=0 if pts is None else len(pts))
        if c is not None:
            def inspect(*args,**kw):
                opt = least_squares(*args,**kw)
                vals.update(status=opt.status,nfev=opt.nfev,rank=np.linalg.matrix_rank(opt.jac))
                return opt
            with patch('scipy.optimize.least_squares',inspect):
                vals['refined'] = _ray_sphere_centre(pts,reader.marker_radius,c) is not None
        details[key] = vals
    return dict(result,markers=details)


def run():
    spec = lock()
    cases = {}
    nominal = load(OUT/'nominal/decision.json')
    normal_run = load(OUT/'nominal/run.json')
    for case in spec['plan']['cases']:
        folder = OUT/case
        d = load(folder/'decision.json')
        r = load(verify(folder/'run.json',d['run_sha256']))
        verify(OUT/'registration.json',r['registration_sha256'])
        verify(folder/'optical_replay.json',d['replay_sha256'])
        before = [n for n in d['nodes_sha256'] if r['captures'][n]['tick']<=9]
        matches = []
        for name in before:
            m = r['captures'][name]
            if r['moves'][:m['moves']]==normal_run['moves'][:m['moves']]:
                matches.append(d['nodes_sha256'][name]==nominal['nodes_sha256'][name])
        cases[case] = dict(decision_sha256=sha256(folder/'decision.json'),passed=d['passed'],state=d['state'],
             closures=d['closures'],closed=d['closed'],ready=d['ready'],prefix=d['final']['prefix_rows'],
             full=d['final']['farthest_row_end'],partial=d['final']['farthest_partial_row_end'],
             gates=d['gates'],reason=d['reason'],sealed_refusal_diagnostic=stopped_frame(folder,r),
             paired_pre_event_work_nodes=dict(compared=len(matches),identical=sum(matches)))
    check(not any(not (OUT/c/'decision.json').exists() for c in spec['plan']['cases']),'incomplete matrix')
    count = sum(c['passed'] for c in cases.values())
    write(OUT/'matrix.json',dict(registration_sha256=sha256(OUT/'registration.json'),
          offline_gate_sha256=sha256(OUT/'offline_gate.json'),completed=len(cases),passed=count,
          all_passed=count==len(cases),cases=cases))
    print('matrix:',count,'/',len(cases))


if __name__=='__main__':
    run()
