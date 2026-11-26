"""Fit a static CAD/shaft alignment, then check held-back archive poses."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from screen_m5_v3_distal_workspace_v1 import DATA, ROOT, RAY, checked, digest, geometry, load, pitch_frame, write
from m5_v3_rigid_fk import CutterFK, rot, shaft_basis

PLAN = ROOT/'config/task/m5_v3_readonly_v1.json'


def calibrate(out):
    plan = load(PLAN)
    prior = DATA/'m5_v3_distal_workspace_v1'
    a = checked(prior/'registration.json',plan['distal_registration_sha256'])
    b = checked(prior/'decision.json',plan['distal_decision_sha256'])
    reg,decision = load(a),load(b)
    if not decision['model_checks_pass'] or decision['registration_sha256']!=digest(a):
        raise ValueError('unqualified archive calibration')
    for name,h in reg['sources'].items():
        checked(ROOT/name,h)
    ray = checked(RAY/'registration.json',reg['protocol']['ray_registration_sha256'])
    ref = load(ray)
    basis,rcm = map(np.array,(ref['basis_world'],ref['rcm_world_m']))
    g = geometry(reg['protocol']['tool'])
    rolls = []
    for e in reg['samples']:
        if e['phase']=='calibration':
            q = np.array(e['command'])
            p,_ = pitch_frame(np.array(e['tcp']),q[5],g)
            t = shaft_basis(basis,q).T @ p @ rot(2,-q[4]) @ rot(0,-q[3])
            rolls.append(float(np.arctan2(t[2,1],t[1,1])))
    if len(rolls)!=6:
        raise ValueError('missing archive calibration')
    payload = dict(rcm=rcm.tolist(),basis=basis.tolist(),
                   insertion_intercept=decision['pitch_pivot_intercept_m'],
                   neutral_twist=float(np.median(rolls)),pitch_to_yaw=g['T_PY'].tolist(),
                   tip_R=g['T_Y_tip'][:3,:3].tolist(),tip_Y=g['tip_Y'].tolist(),hinge_Y=g['hinge_Y'].tolist())
    fk = CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in payload.items()})
    records = []
    for e in reg['samples']:
        if e['phase']=='validation':
            m = np.array(e['tcp'])
            p = fk.pose(e['command'])
            records.append(dict(view=e['view'],tick=e['tick'],
                                position_mm=float(1000*np.linalg.norm(p[:,3]-m[:,3])),
                                orientation_deg=float(np.rad2deg(Rotation.from_matrix(p[:,:3].T@m[:,:3]).magnitude()))))
    qualified = len(records)==6 and all(r['position_mm']<=.20 and r['orientation_deg']<=.20 for r in records)
    if out.exists():
        raise FileExistsError(out)
    write(out,dict(qualified=qualified,calibration=payload,validation=records,authorizes_motion=False,
                   sources={p.relative_to(ROOT).as_posix():digest(p) for p in
                            (PLAN,a,b,ray,Path(__file__).resolve(),ROOT/'scripts/task/m5_v3_rigid_fk.py')}))
    print('qualified',qualified,'maximum residuals',max(r['position_mm'] for r in records),
          max(r['orientation_deg'] for r in records),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    calibrate(p.parse_args().out)
