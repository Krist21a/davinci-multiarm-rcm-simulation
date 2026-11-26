"""Gate the outer-bank development reader on the four sealed uncut exposures."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_m5_v3_optics_attribution_v1 import CAPTURE, ROOT, digest, load, verified
from m5_v3_controller_v2 import World, Work
from m5_v3_perception_v3 import WorkChart, intact_preflight, work
from m5_v3_rigid_fk import CutterFK


PLAN = ROOT/'config/task/m5_v3_outer_bank_v1.json'
CAL = ROOT/'validation/task/m5_v3_rigid_fk_v1.json'


def points(f, a):
    return f['cam_R']@np.asarray(a)+f['cam_t']


def measure(f, marker, model, goal):
    if not marker.get('valid'):
        return dict(passed=False, reason='captured marker fit invalid')
    g = work(f['rgb'], f['depth'], focal=f['focal'].item(), far=f['far'].item(),
             sigma=0., prefix_mm=0, goal_mm=goal)
    if g is None:
        return dict(passed=False, reason='no current chart')
    R = f['cam_R']
    xyz = WorkChart(0, points(f,g.front), R@g.direction, R@g.opening, points(f,g.aim), points(f,g.limit))
    m = Work(xyz, points(f,marker['tip']), R@marker['jaw_axis'], R@marker['opening_axis'], -1, 0)
    room = World(None,goal,model).sweep_room(m)
    return dict(passed=bool(room>=.0002), sweep_room_mm=room*1000, front=xyz.front.tolist(),
                direction=xyz.direction.tolist(), limit=xyz.limit.tolist(), aim=xyz.aim.tolist())


def replay():
    plan = load(PLAN)
    dec = verified(CAPTURE/'decision.json', plan['capture_decision_sha256'])
    reg = verified(CAPTURE/'registration.json', plan['capture_registration_sha256'])
    saved = load(dec)
    sources = load(reg)['sources']
    if saved['registration_sha256']!=digest(reg) or saved['failure'] or saved['cleanup_error']:
        raise ValueError('incomplete sealed acquisition')
    verified(CAL,sources[CAL.relative_to(ROOT).as_posix()])
    c = load(CAL)
    if not c['qualified']:
        raise ValueError('unqualified rigid calibration')
    model = CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in c['calibration'].items()})
    files = [PLAN,CAL,dec,reg,Path(__file__).resolve(),
             ROOT/'scripts/vision/m5_v3_perception_v3.py',
             ROOT/'scripts/vision/m5_v3_corridor_v4.py',
             ROOT/'scripts/task/m5_v3_controller_v2.py', ROOT/'scripts/task/m5_v3_rigid_fk.py']
    narrow, wide, agreement = {}, {goal:{} for goal in (9,10)}, {}
    for name, m in saved['captures'].items():
        path = verified(CAPTURE/m['file'],m['sha256'])
        files.append(path)
        with np.load(path,allow_pickle=False) as f:
            shape = (960,1280) if name.startswith('work') else (1920,2560)
            fov = 70 if name.startswith('work') else 20
            if f['depth'].shape!=shape or m['fov_deg']!=fov or m['closed']!=(name!='work_open'):
                raise ValueError('capture does not match registered optical phase')
            if fov==70:
                for goal,a in wide.items():
                    a[name] = measure(f,m['marker'],model,goal)
            else:
                narrow[name] = intact_preflight(f['rgb'],f['depth'],far=f['far'].item())
    for goal, frames in wide.items():
        if set(frames)!={'work_closed','work_open'} or set(narrow)!={'narrow_before_backoff','narrow_entry_clear'}:
            raise ValueError('incomplete comparisons')
        if not all(r['passed'] for r in frames.values()):
            agreement[goal] = dict(passed=False)
            continue
        x,y = frames['work_closed'],frames['work_open']
        distance = float(np.linalg.norm(np.asarray(x['front'])-y['front']))
        angle = float(np.rad2deg(np.arccos(np.clip(np.dot(x['direction'],y['direction']),-1,1))))
        agreement[goal] = dict(passed=distance<.0004 and angle<5., front_delta_mm=1000*distance,
                               angle_delta_deg=angle)
    result = bool(all(a['passed'] for a in agreement.values()) and narrow['narrow_entry_clear']['passed'])
    return dict(passed=result, scope='development positives only', authorizes_production_motion=False,
                goal_work=wide, goal_agreement=agreement, narrow=narrow,
                sources={p.relative_to(ROOT).as_posix():digest(p) for p in files})


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    r = replay()
    with a.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(r,f,indent=2,allow_nan=False)
        f.write('\n')
    print('offline gate:',r['passed'])
