"""Grid-search yaw and pitch together, using the sealed distal calibration."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from screen_m5_v3_distal_workspace_v1 import DATA, ROOT, RAY, checked, descriptor, digest
from screen_m5_v3_distal_workspace_v1 import geometry, load, reconstruct, write
from screen_m5_v3_yaw_v1 import angles

ID = 'm5_v3_distal_workspace_v2'
SPEC = ROOT/f'config/task/{ID}.json'
PRIOR = DATA/'m5_v3_distal_workspace_v1'


def screen(out):
    plan = load(SPEC)
    lock = checked(PRIOR/'registration.json',plan['distal_v1_registration_sha256'])
    ref = checked(PRIOR/'decision.json',plan['distal_v1_decision_sha256'])
    reg,cal = load(lock),load(ref)
    if cal['registration_sha256']!=digest(lock) or not cal['model_checks_pass']:
        raise ValueError('unqualified distal calibration')
    for p,h in reg['sources'].items():
        checked(ROOT/p,h)
    ray_file = checked(RAY/'decision.json',reg['protocol']['ray_decision_sha256'])
    ray_reg = checked(RAY/'registration.json',reg['protocol']['ray_registration_sha256'])
    ray,anchor = load(ray_file),load(ray_reg)
    for p,h in anchor['sources'].items():
        checked(ROOT/p,h)
    rcm,basis = np.array(anchor['rcm_world_m']),np.array(anchor['basis_world'])
    source = Path(__file__).resolve()
    wrist_id = reg['protocol']['tool']
    if wrist_id!=descriptor(wrist_id)['model'] or not set(ray['views'])=={'original','plus3'}:
        raise ValueError('unexpected prior registration')
    wrist = geometry(wrist_id)
    lo,hi = np.array(reg['soft_low']),np.array(reg['soft_high'])
    intercept = cal['pitch_pivot_intercept_m']
    centers = np.array([v['rows_world_m'] for v in ray['views'].values()])
    if centers.shape!=(2,11,3):
        raise ValueError('unexpected sample set')
    reference = centers[:,5,:].mean(0)
    axial = float((reference-rcm)@basis[0])
    base_angles = np.deg2rad(angles(reference,rcm,basis))
    params = plan['thresholds']
    yaw_window = params['centered_yaw_deg']
    min_margin = np.array([params['minimum_soft_yaw_headroom_deg'],
                           params['minimum_soft_pitch_headroom_deg']])
    if list(min_margin)!=[1.5,1.5] or axial<=0.:
        raise ValueError('invalid two-axis requirements')
    perturb = np.vstack((np.zeros((1,3)),.0005*basis,-.0005*basis))
    views = {}
    for name,v in ray['views'].items():
        row = np.asarray(v['rows_world_m'])
        d = row[-1]-row[0]
        d /= np.linalg.norm(d)
        targets = np.concatenate((row,[p-s*d for p in (row[0],row[-1]) for s in (.001,.003)]))
        rotations = [np.array(s['tcp'])[:,:3] for s in reg['samples']
                     if s['view']==name and s['phase']=='calibration']
        if len(rotations)!=3:
            raise ValueError('missing blade rotations')
        views[name] = targets,rotations
    out.mkdir(exist_ok=False)
    path = out/'registration.json'
    write(path,dict(protocol=plan,sources={p.relative_to(ROOT).as_posix():digest(p) for p in
               (SPEC,source,lock,ref,ray_file,ray_reg,ROOT/'scripts/analysis/screen_m5_v3_distal_workspace_v1.py')},
               reference_m=reference.tolist(),rcm_world_m=rcm.tolist(),basis_world=basis.tolist(),
               prior_shaft_angles_deg=np.rad2deg(base_angles).tolist(),soft_low=lo.tolist(),soft_high=hi.tolist()))
    results = []
    for dy,dp in product(plan['delta_yaw_deg'],plan['delta_pitch_deg']):
        coefficients = axial*(np.tan(base_angles+np.deg2rad([dy,dp]))-np.tan(base_angles))
        shift = coefficients@basis[1:]
        by_view = {}
        for name,(targets,rotations) in views.items():
            q = np.array([reconstruct(np.column_stack((R,p+shift+noise)),wrist,rcm,basis,intercept)
                          for p,noise,R in product(targets,perturb,rotations)])
            qlo,qhi = q.min(0),q.max(0)
            room = np.minimum(qlo-lo,hi-qhi)
            accepted = bool((room>=0).all() and (room[:2]>=min_margin).all()
                            and yaw_window[0]<=qlo[0]<=qhi[0]<=yaw_window[1]
                            and cal['roll_soft_span_rad']>=2*np.pi)
            by_view[name] = dict(samples=len(q),min_q=qlo.tolist(),max_q=qhi.tolist(),
                                 minimum_soft_headroom=room.tolist(),passes=accepted)
        results.append(dict(delta_yaw_deg=dy,delta_pitch_deg=dp,translation_m=shift.tolist(),
                            translation_mm=float(np.linalg.norm(shift)*1000),
                            by_view=by_view,passes=all(v['passes'] for v in by_view.values()) and dp>0.))
    eligible = [o for o in results if o['passes']]
    selected = min(eligible,key=lambda x:(x['translation_mm'],x['delta_yaw_deg'],x['delta_pitch_deg'])) if eligible else None
    write(out/'decision.json',dict(registration_sha256=digest(path),evaluated=len(results),
                                   passing_grid_points=len(eligible),selected=selected,results=results,
                                   verified_proximal_ik=False,verified_collision_free_path=False,
                                   authorizes_motion=False))
    print('evaluated',len(results),'passing',len(eligible),'selected',selected,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    screen(a.out)
