"""Registered offline comparison of v4 and v5 on sealed work exposures."""
import argparse
from pathlib import Path

import numpy as np

from calibrate_m5_v3_sensor_margin_v1 import CAL, ROOT, digest, load, verified, write
from cut_perception import CutPerception
from m5_v3_controller_v2 import Work, World
import m5_v3_perception_v4 as v4
import m5_v3_perception_v5 as v5
from m5_v3_rigid_fk import CutterFK
from replay_m5_v3_outer_bank_v1 import CAPTURE, intact_preflight


PLAN = ROOT/'config/task/m5_v3_work_denoise_v1.json'
PRIOR = ROOT/'validation/task/m5_v3_sensor_margin_calibration_v1/registration.json'
OUTER = ROOT/'validation/task/m5_v3_outer_bank_v1.json'


def registration(plan):
    sources = {}
    for pins in (plan['sources'],load(PRIOR)['sources'],load(OUTER)['sources']):
        for name,pin in pins.items():
            verified(ROOT/name,pin)
            sources[name] = pin
    for path in (PLAN,Path(__file__).resolve(),Path(v5.__file__).resolve()):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    prior = load(PRIOR)
    frames = prior['frames']
    if (len(frames)!=15 or sum(r['prefix_mm']>=6 for r in frames)!=3
            or not load(OUTER)['passed']):
        raise ValueError('missing sealed work or outer gate evidence')
    return dict(plan=plan,noise_plan=prior['plan'],sources=sources,frames=frames)


def axis_angle(a,b):
    return float(np.rad2deg(np.arccos(np.clip(a@b,-1,1))))


def project(vec,axis):
    return float(1000*(vec@axis))


def measure_frame(rec, noise, model):
    with np.load(ROOT/rec['path'],allow_pickle=False) as f:
        rgb,z = f['rgb'],f['depth']
        focal,far = f['focal'].item(),f['far'].item()
    if z.shape!=(960,1280) or not np.isclose(focal,640/np.tan(np.deg2rad(35)),rtol=0,atol=1e-6):
        raise ValueError('unexpected work optics')
    reader = CutPerception(1280,960,focal,far,model.rcm,tcp_local=model.tip_Y,
                           jaw_axis_local=model.tip_R[:,2],opening_axis_local=model.tip_R[:,0],
                           ray_marker_fit=True)
    rgb_float = rgb.astype(float)
    n = rec['prefix_mm']

    def chart(version, depth, sigma):
        return version.work(rgb,depth,focal=focal,far=far,sigma=sigma,prefix_mm=n,goal_mm=10)

    old,new = chart(v4,z,0.),chart(v5,z,0.)
    mark,_ = reader._scissors(rgb_float,z)
    saved = rec['marker']
    if bool(mark['valid'])!=bool(saved['valid']) or (mark['valid'] and any(
            not np.allclose(mark[k],saved[k],rtol=0,atol=1e-12)
            for k in ('tip','jaw_axis','opening_axis'))):
        raise ValueError('clean ray-marker replay differs from sealed fit: '+rec['path'])
    base = dict(v4_chart_valid=old is not None,v5_chart_valid=new is not None,marker_valid=mark['valid'])
    tip = np.asarray(mark['tip']) if mark['valid'] else None
    if tip is not None and new is not None:
        base['v5_room_mm'] = project(new.limit-tip,new.direction)
    if tip is not None and old is not None and new is not None:
        base['v4_room_mm'] = project(old.limit-tip,old.direction)
        base['v5_minus_v4_room_mm'] = base['v5_room_mm']-base['v4_room_mm']
        base['axis_delta_from_v4_deg'] = axis_angle(new.direction,old.direction)
        base['front_delta_from_v4_mm'] = float(1000*np.linalg.norm(new.front-old.front))

    sigma = noise['depth_sigmas_m'][1]
    draws = []
    for part in ('development','check'):
        for seed in noise[part+'_seeds']:
            rng = np.random.default_rng(np.random.SeedSequence([seed,rec['case_id'],rec['tick']]))
            zz = z+rng.normal(0.,sigma,z.shape)
            g = chart(v5,zz,sigma)
            noisy,_ = reader._scissors(rgb_float,zz)
            row = dict(part=part,seed=seed,chart_valid=g is not None,marker_valid=noisy['valid'])
            if g is not None and noisy['valid'] and 'v5_room_mm' in base:
                noisy_tip = np.asarray(noisy['tip'])
                delta = g.limit-noisy_tip
                row.update(axis_delta_deg=axis_angle(g.direction,new.direction),
                           delta_visual_room_mm=project(delta,g.direction)-base['v5_room_mm'],
                           delta_limit_on_raw_axis_mm=project(g.limit-new.limit,new.direction),
                           delta_tip_on_raw_axis_mm=project(tip-noisy_tip,new.direction),
                           delta_axis_mm=project(delta,g.direction-new.direction))
            elif not noisy['valid']:
                row['marker_reason'] = noisy['reason']
            draws.append(row)
    return dict(**{k:v for k,v in rec.items() if k!='marker'},clean=base,draws=draws)


def outer_gate(model, plan):
    capture = load(CAPTURE/'decision.json')
    if set(capture['captures'])!={'work_closed','work_open','narrow_before_backoff','narrow_entry_clear'}:
        raise ValueError('unexpected sealed outer gate exposures')
    result = dict(narrow={},goals={})
    for key in ('narrow_before_backoff','narrow_entry_clear'):
        m = capture['captures'][key]
        path = verified(CAPTURE/m['file'],m['sha256'])
        with np.load(path,allow_pickle=False) as f:
            if f['depth'].shape!=(1920,2560) or m['fov_deg']!=20 or not m['closed']:
                raise ValueError('unexpected sealed narrow optics')
            result['narrow'][key] = intact_preflight(f['rgb'],f['depth'],far=f['far'].item())['passed']
    for goal in (9,10):
        stats, front, axes = {}, [], []
        for key in ('work_closed','work_open'):
            m = capture['captures'][key]
            path = verified(CAPTURE/m['file'],m['sha256'])
            with np.load(path,allow_pickle=False) as f:
                R,t = f['cam_R'],f['cam_t']
                if f['depth'].shape!=(960,1280) or m['fov_deg']!=70 or m['closed']!=(key=='work_closed'):
                    raise ValueError('unexpected sealed work optics')
                g = v5.work(f['rgb'],f['depth'],focal=f['focal'].item(),far=f['far'].item(),
                            sigma=0.,prefix_mm=0,goal_mm=goal)
            marker = m['marker']
            if g is None or not marker['valid']:
                stats[key] = dict(passed=False)
                continue
            xyz = v5.WorkChart(0,R@g.front+t,R@g.direction,R@g.opening,R@g.aim+t,R@g.limit+t)
            pose = Work(xyz,R@marker['tip']+t,R@marker['jaw_axis'],R@marker['opening_axis'],-1,0)
            room = World(None,goal,model).sweep_room(pose)
            stats[key] = dict(passed=bool(room>=.0002),sweep_room_mm=float(1000*room))
            front.append(xyz.front)
            axes.append(xyz.direction)
        if len(front)==2:
            distance = float(1000*np.linalg.norm(front[0]-front[1]))
            angle = axis_angle(*axes)
            result['goals'][goal] = dict(work=stats,front_delta_mm=distance,axis_delta_deg=angle,
                legacy_gate=all(s['passed'] for s in stats.values()) and distance<.40 and angle<5.,
                tighter_front_gate=distance<plan['clean_front_closed_open_delta_mm_lt'])
        else:
            result['goals'][goal] = dict(work=stats,legacy_gate=False,tighter_front_gate=False)
    return result


def summarize(rows, part=None):
    draws = [d for r in rows for d in r['draws'] if part is None or d['part']==part]
    return dict(draws=len(draws),chart_valid=sum(d['chart_valid'] for d in draws),
                marker_valid=sum(d['marker_valid'] for d in draws),
                paired_valid=sum(d['chart_valid'] and d['marker_valid'] for d in draws),
                scored=sum('delta_visual_room_mm' in d for d in draws),
                max_axis_delta_deg=max((d['axis_delta_deg'] for d in draws if 'axis_delta_deg' in d),
                                       default=None),
                max_optimism_delta_mm=max((d['delta_visual_room_mm'] for d in draws
                                          if 'delta_visual_room_mm' in d),default=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    plan = load(PLAN)
    reg = registration(plan)
    args.out.mkdir(exist_ok=False)
    seal = args.out/'registration.json'
    write(seal,reg)
    cal = load(CAL)
    if not cal['qualified']:
        raise ValueError('unqualified rigid marker calibration')
    model = CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in cal['calibration'].items()})
    rows = []
    for rec in reg['frames']:
        rows.append(measure_frame(rec,reg['noise_plan'],model))
        print('measured',rec['path'],flush=True)
    measurements = args.out/'measurements.json'
    write(measurements,rows)
    rows = load(measurements)
    late_rows = [r for r in rows if r['prefix_mm']>=6]
    late = summarize(late_rows)
    if len(rows)!=15 or summarize(rows)['draws']!=360 or late['draws']!=72:
        raise ValueError('registered draw count mismatch')
    early = [r for r in rows if r['prefix_mm']<=5]
    bias = [abs(r['clean']['v5_minus_v4_room_mm']) for r in rows if 'v5_minus_v4_room_mm' in r['clean']]
    outer = outer_gate(model,plan)
    gates = dict(clean=len(bias)==15 and max(bias)<plan['clean_abs_room_delta_mm_lt'],
                 legacy_outer=outer['narrow']['narrow_entry_clear']
                 and all(g['legacy_gate'] for g in outer['goals'].values()),
                 tighter_outer=all(g['tighter_front_gate'] for g in outer['goals'].values()),
                 late_pairs=late['paired_valid']>=plan['late_pair_min_count'],
                 late_axis=late['scored']==late['paired_valid']>0
                 and late['max_axis_delta_deg']<plan['late_noisy_axis_delta_deg_lt'],
                 late_optimism=late['scored']==late['paired_valid']>0
                 and late['max_optimism_delta_mm']<plan['late_noisy_optimism_delta_mm_lt'])
    result = dict(passed=all(gates.values()),authorizes_motion=False,gates=gates,
                  clean_pairs=len(bias),clean_max_abs_room_delta_mm=max(bias,default=None),outer=outer,
                  early={p:summarize(early,p) for p in ('development','check')},
                  late={**late,**{p:summarize(late_rows,p) for p in ('development','check')}},
                  registration_sha256=digest(seal),measurements_sha256=digest(measurements))
    write(args.out/'decision.json',result)
    print('offline gate:',result['passed'],gates,flush=True)


if __name__=='__main__':
    main()
