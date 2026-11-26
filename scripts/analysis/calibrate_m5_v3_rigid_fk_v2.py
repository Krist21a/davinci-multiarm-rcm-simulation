"""Fit the passive twist exclusively on the sealed legacy FK development split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from m5_v3_rigid_fk_v2 import CutterFK, rot, shaft_basis


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'config/task/m5_v3_rigid_fk_v2.json'
ARCHIVE = ROOT/'validation/task/m5_v3_distal_workspace_v1/registration.json'
V1 = ROOT/'validation/task/m5_v3_rigid_fk_v1.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def observed_twist(entry,rigid,neutral):
    q = np.asarray(entry['command'],dtype=float)
    tcp_R = np.asarray(entry['tcp'],dtype=float)[:,:3]
    pitch = tcp_R@rigid['tip_R'].T@rot(2,-q[5])@rigid['pitch_to_yaw'][:3,:3].T
    residual = shaft_basis(rigid['basis'],q).T@pitch@rot(2,-q[4])@rot(0,-q[3])
    angle = np.arctan2(residual[2,1],residual[1,1])
    return neutral+np.arctan2(np.sin(angle-neutral),np.cos(angle-neutral))


def calibrate():
    plan = load(PLAN)
    archive,old = load(ARCHIVE),load(V1)
    sources = {}
    for pins in (plan['sources'],archive['sources'],old['sources']):
        for name,pin in pins.items():
            if digest(ROOT/name)!=pin:
                raise ValueError('source drift: '+name)
            sources[name] = pin
    for path in (PLAN,Path(__file__).resolve(),ROOT/'scripts/task/m5_v3_rigid_fk_v2.py'):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    fit = [e for e in archive['samples'] if e['phase']=='calibration']
    check = [e for e in archive['samples'] if e['phase']=='validation']
    if not old['qualified'] or len(fit)!=6 or len(check)!=6 or len({e['frame'] for e in fit+check})!=12:
        raise ValueError('unqualified or non-disjoint legacy FK archive')
    rigid = {k:np.asarray(v) if isinstance(v,list) else v for k,v in old['calibration'].items()}
    neutral = rigid.pop('neutral_twist')
    X = np.asarray([[1.,*e['command'][:2]] for e in fit])
    y = np.asarray([observed_twist(e,rigid,neutral) for e in fit])
    coeff,_,rank,_ = np.linalg.lstsq(X,y,rcond=None)
    if rank!=3 or not np.isfinite(coeff).all():
        raise ValueError('rank-deficient or invalid passive twist fit')
    fk = CutterFK(**rigid,twist_rad=coeff)

    def entry(e):
        return {k:e[k] for k in ('phase','view','tick','command_tick','frame','sha256')}

    validation = []
    for e in check:
        truth = np.asarray(e['tcp'])
        pose = fk.pose(e['command'])
        validation.append(dict(entry(e), position_mm=float(1000*np.linalg.norm(pose[:,3]-truth[:,3])),
                               orientation_deg=float(np.rad2deg(
                                   Rotation.from_matrix(pose[:,:3].T@truth[:,:3]).magnitude()))))
    qualified = all(r['position_mm']<=plan['position_max_mm'] and r['orientation_deg']<=plan['orientation_max_deg']
                    for r in validation)
    payload = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in rigid.items()}
    return dict(qualified=qualified,authorizes_motion=False,plan=plan,sources=sources,
                calibration=dict(payload,twist_rad=coeff.tolist()),fit_rank=int(rank),
                development=[dict(entry(e),observed_twist_rad=float(t),fit_twist_rad=float(p))
                             for e,t,p in zip(fit,y,X@coeff,strict=True)],
                validation=validation,constant_twist_validation=old['validation'],
                max_position_mm=max(r['position_mm'] for r in validation),
                max_orientation_deg=max(r['orientation_deg'] for r in validation))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    data = calibrate()
    with args.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write('\n')
    print(data['qualified'],data['calibration']['twist_rad'],data['max_position_mm'],data['max_orientation_deg'])
