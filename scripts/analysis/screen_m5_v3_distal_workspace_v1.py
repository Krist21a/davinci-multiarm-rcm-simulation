"""Reconstruct orientation-preserving distal geometry; no proximal IK or collision checking."""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from replay_cell_v2 import DATA, ROOT, digest, load, write
from screen_m5_v3_yaw_v1 import ARCHIVE, angles, checked
from tool_pose import _MARGIN

sys.path.insert(0,str(ROOT/'scripts/build'))
from instruments import CONFIG, XACRO, descriptor, geometry, parse_stage

ID = 'm5_v3_distal_workspace_v1'
SPEC = ROOT/f'config/task/{ID}.json'
RAY = DATA/'m5_v3_yaw_screen_v1'


def rz(w):
    return Rotation.from_rotvec([0.,0.,w]).as_matrix()


def pitch_frame(tcp, w, wrist):
    Y = tcp[:,:3]@wrist['T_Y_tip'][:3,:3].T
    T = wrist['T_PY']
    P = Y@rz(-w)@T[:3,:3].T
    pivot = tcp[:,3] - Y@wrist['tip_Y'] - P@T[:3,3]
    return P,pivot


def reconstruct(tcp, wrist, rcm, basis, intercept):
    Y = tcp[:,:3]@wrist['T_Y_tip'][:3,:3].T
    ray = Y.T@(tcp[:,3]-rcm)-wrist['tip_Y']
    # This tool's pitch axis is +X in the neutral yaw frame; the
    # pitch-to-yaw translation is orthogonal to it. Un-yaw the RCM ray.
    wy = np.arctan2(ray[0],ray[1])
    P,pivot = pitch_frame(tcp,wy,wrist)
    shaft = pivot-rcm
    radius = np.linalg.norm(shaft)
    if abs(shaft@P[:,2])>1e-8:
        raise ValueError('pitch pivot does not lie in the shaft plane')
    wp = np.arctan2(-shaft@P[:,1],shaft@P[:,0])
    return np.r_[angles(pivot,rcm,basis),radius-intercept,wp,wy]


def screen(out, frames):
    plan = load(SPEC)
    ray_file = checked(RAY/'decision.json',plan['ray_decision_sha256'])
    seal = checked(RAY/'registration.json',plan['ray_registration_sha256'])
    ray,reg = load(ray_file),load(seal)
    if ray['registration_sha256']!=digest(seal):
        raise ValueError('unmatched yaw screen')
    for p,h in reg['sources'].items():
        checked(ROOT/p,h)
    wrist_model = plan['tool']
    desc = descriptor(wrist_model)
    stages = desc['stages']
    assets = [CONFIG/'instruments'/f'{wrist_model}.json']
    for part,stage in stages.items():
        model = XACRO/part/f'{part}_{stage}.urdf.xacro'
        _,visuals = parse_stage(model)
        assets.extend((model,*(v['file'] for group in visuals.values() for v in group)))
    wrist = geometry(wrist_model)
    T = wrist['T_PY']
    if (not np.allclose(T[:3,:3].T@[0,0,1],[1,0,0],atol=1e-12)
            or abs(T[2,3])>1e-12):
        raise ValueError('wrist geometry is not the registered orthogonal chain')
    rcm,basis = np.array(reg['rcm_world_m']),np.array(reg['basis_world'])
    arm, = (a for a in load(ROOT/reg['protocol']['rcm']) if a['arm']=='psm2')
    shaft,k = arm['limits'],desc['kinematics']
    hard = np.array([[-shaft['yaw_deg'],shaft['yaw_deg']],[-shaft['pitch_deg'],shaft['pitch_deg']],
                     [shaft['insertion_min'],shaft['insertion_max']],
                     k['wrist_pitch_limits_rad'],k['wrist_yaw_limits_rad']])
    margin = _MARGIN[[0,1,2,4,5]]
    lo,hi = hard[:,0]+margin,hard[:,1]-margin
    roll_span = float(np.ptp(k['roll_limits_rad'])-2*_MARGIN[3])
    sources = {p.relative_to(ROOT).as_posix():digest(p) for p in
               (SPEC,Path(__file__).resolve(),ray_file,seal,ROOT/'scripts/build/instruments.py',
                ROOT/'scripts/build/articulation.py',ROOT/'scripts/build/simtools.py',*assets)}
    capture = load(ARCHIVE/'decision.json')
    samples = []
    for view in reg['protocol']['views']:
        run = load(ARCHIVE/view/'nominal.json')
        commands = [e for e in run['events'] if e['event']=='guided_v21_set_tool_pose' and e['accepted']]
        for phase,prefix in (('calibration','closure'),('validation','advance')):
            for i in range(1,4):
                tag = capture['views'][view]['frames'][prefix+str(i)]
                p = checked(frames/ARCHIVE.name/view/'nominal'/tag['file'],tag['sha256'])
                with np.load(p,allow_pickle=False) as f:
                    assert f['tick'].item()==tag['tick']
                    tcp = f['tcp']
                if not np.allclose(tcp[:,:3].T@tcp[:,:3],np.eye(3),atol=1e-6):
                    raise ValueError('invalid TCP rotation')
                e = max((e for e in commands if e['tick']<tag['tick']),key=lambda e:e['tick'])
                if (phase=='calibration' and (e['state']!='CUT' or e['tick']!=tag['tick']-1)
                        or phase=='validation' and (e['state']!='ALIGN' or not e['closed_before']
                                                    or not e['closed_after'])):
                    raise ValueError('unmatched readback and command')
                samples.append(dict(phase=phase,view=view,tick=tag['tick'],command_tick=e['tick'],
                                    command=e['q'],tcp=tcp.tolist(),sha256=tag['sha256'],
                                    frame=p.relative_to(frames).as_posix()))
    out.mkdir(exist_ok=False)
    lock = out/'registration.json'
    write(lock,dict(protocol=plan,sources=sources,samples=samples,limit_order=[
        'yaw_deg','pitch_deg','ins_m','wrist_pitch_rad','wrist_yaw_rad'],
        soft_low=lo.tolist(),soft_high=hi.tolist()))

    distances = []
    for s in samples:
        if s['phase']=='calibration':
            q = np.asarray(s['command'])
            _,pivot = pitch_frame(np.asarray(s['tcp']),q[5],wrist)
            distances.append(float(np.linalg.norm(pivot-rcm)-q[2]))
    intercept = float(np.median(distances))
    tol = plan['checks']
    bounds = np.array([tol['shaft_deg']]*2+[tol['insertion_m']]+[tol['wrist_rad']]*2)
    validation = []
    for s in samples:
        if s['phase']=='validation':
            q = reconstruct(np.asarray(s['tcp']),wrist,rcm,basis,intercept)
            error = q-np.asarray(s['command'])[[0,1,2,4,5]]
            validation.append(dict(view=s['view'],tick=s['tick'],
                                   reconstruction_minus_command=error.tolist(),
                                   passed=bool((abs(error)<=bounds).all())))
    model_checks_pass = len(validation)==6 and all(v['passed'] for v in validation)
    verdict = dict(registration_sha256=digest(lock),model_checks_pass=model_checks_pass,
                   pitch_pivot_intercept_m=intercept,pitch_pivot_samples_m=distances,
                   validation=validation,roll_soft_span_rad=roll_span,
                   verified_proximal_ik=False,verified_collision_free_path=False,
                   verified_roll_readback_at_candidate=False,authorizes_motion=False)
    if not model_checks_pass:
        write(out/'decision.json',verdict)
        print('Model rejected by matched readbacks.')
        return
    perturb = np.vstack((np.zeros((1,3)),.0005*basis,-.0005*basis))
    result = {}
    for name,v in ray['views'].items():
        row = np.asarray(v['rows_world_m'])
        d = row[-1]-row[0]
        d /= np.linalg.norm(d)
        points = np.concatenate((row,[p-s*d for p in (row[0],row[-1]) for s in (.001,.003)]))
        R = [np.array(s['tcp'])[:,:3] for s in samples if s['view']==name and s['phase']=='calibration']
        shifts = []
        for shift in v['shifts']:
            translation = np.array(shift['translation_world_m'])
            candidates = np.array([reconstruct(np.column_stack((rot,p+translation+e)),wrist,rcm,basis,intercept)
                                   for p in points for e in perturb for rot in R])
            min_q,max_q = candidates.min(0),candidates.max(0)
            soft_room = np.minimum(min_q-lo,hi-max_q)
            within_limits = bool((soft_room>=0).all() and roll_span>=2*np.pi)
            centered = bool(within_limits and -2.5<=min_q[0]<=max_q[0]<=1.)
            shifts.append(dict(shift_mm=shift['shift_mm'],samples=len(candidates),
                               min_q=min_q.tolist(),max_q=max_q.tolist(),
                               minimum_soft_headroom=soft_room.tolist(),
                               all_modeled_limits=within_limits,centered_candidate=centered))
        result[name] = shifts
    common = [m['shift_mm'] for m,n in zip(*result.values(),strict=True)
              if m['shift_mm']==n['shift_mm'] and m['centered_candidate'] and n['centered_candidate']]
    verdict.update(views=result,common_centered_candidates_mm=common)
    write(out/'decision.json',verdict)
    print('Model readback checks:',model_checks_pass,'common centered candidates:',common,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    screen(a.out,a.frames)
