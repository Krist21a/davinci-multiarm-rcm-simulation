"""Paired v4/v7 work replay: separate fixed-axis depth bias from axis stability."""
import argparse
from pathlib import Path

import numpy as np

from calibrate_m5_v3_sensor_margin_v1 import CAL, ROOT, digest, load, verified, write
from cut_perception import CutPerception
from m5_v3_controller_v2 import Work, World
import m5_v3_perception_v4 as v4
import m5_v3_perception_v7 as v7
from m5_v3_rigid_fk import CutterFK
from replay_m5_v3_outer_bank_v1 import CAPTURE, intact_preflight
from replay_m5_v3_work_denoise_v1 import OUTER, PRIOR, axis_angle, project, summarize


PLAN = ROOT/'config/task/m5_v3_work_denoise_v3.json'


def registration(plan):
    sources = {}
    for pins in (plan['sources'],load(PRIOR)['sources'],load(OUTER)['sources']):
        for name,pin in pins.items():
            verified(ROOT/name,pin)
            sources[name] = pin
    for path in (PLAN,Path(__file__).resolve(),Path(v7.__file__).resolve()):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    prior = load(PRIOR)
    frames = prior['frames']
    if len(frames)!=15 or sum(r['prefix_mm']>=6 for r in frames)!=3 or not load(OUTER)['passed']:
        raise ValueError('missing sealed work or outer gate evidence')
    return dict(plan=plan,noise_plan=prior['plan'],sources=sources,frames=frames)


def old_limit_points(rgb,z,focal,far,n,chart):
    label = v4.inks(rgb)
    valid = v4._good(z,far)
    pink = valid & (rgb==v4.PINK).all(-1)
    pix = [np.argwhere(valid & (label>=0) & (label//13==side)) for side in (0,1)]
    across = pix[1].mean(0)-pix[0].mean(0)
    across /= np.linalg.norm(across)
    sides = [v4._stations(v4._rim(label,z,valid,pink,side,across,.001),z.shape,focal,.001)
             for side in (0,1)]
    checked = {}
    for j in sides[0].keys() & sides[1].keys():
        a,b = sides[0][j],sides[1][j]
        if .0065<=np.linalg.norm(b-a)<=.0115 and abs((b-a)@chart.direction)<=.00075:
            checked[j] = np.stack((a,b))
    row = max(j for j in checked if n+2<=j<=8) if n<=5 else 10
    points = checked[row]
    if not np.isclose(np.min(points@chart.direction),chart.limit@chart.direction,rtol=0,atol=1e-12):
        raise ValueError('v4 reconstructed bound disagrees with original work output')
    return row,points


def measure_frame(rec,noise,model):
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
    opts = dict(focal=focal,far=far,prefix_mm=n,goal_mm=10)
    old = v4.work(rgb,z,sigma=0.,**opts)
    evidence = v7._work_evidence(rgb,z,sigma=0.,**opts)
    new = evidence[0] if evidence is not None else None
    marker,_ = reader._scissors(rgb_float,z)
    saved = rec['marker']
    if bool(marker['valid'])!=bool(saved['valid']) or (marker['valid'] and any(
            not np.allclose(marker[k],saved[k],rtol=0,atol=1e-12)
            for k in ('tip','jaw_axis','opening_axis'))):
        raise ValueError('clean ray-marker replay differs from sealed fit: '+rec['path'])
    clean = dict(v4_chart_valid=old is not None,v7_chart_valid=new is not None,marker_valid=marker['valid'])
    tip = np.asarray(marker['tip']) if marker['valid'] else None
    if tip is not None and new is not None:
        clean['v7_room_mm'] = project(new.limit-tip,new.direction)
    if tip is not None and old is not None and new is not None:
        v4_row,v4_points = old_limit_points(rgb,z,focal,far,n,old)
        _,v7_row,v7_points = evidence
        clean.update(v4_limit_row=v4_row,v7_limit_row=v7_row,v4_limit_points_m=v4_points.tolist(),
                     v7_limit_points_m=v7_points.tolist(),v4_direction=old.direction.tolist(),
                     v4_room_mm=project(old.limit-tip,old.direction),
                     axis_delta_from_v4_deg=axis_angle(new.direction,old.direction))
        if v4_row==v7_row:
            before = 1000*float(np.min((v4_points-tip)@old.direction))
            after = 1000*float(np.min((v7_points-tip)@old.direction))
            clean['fixed_v4_axis_limit_delta_mm'] = after-before
    sigma = noise['depth_sigmas_m'][1]
    draws = []
    for part in ('development','check'):
        for seed in noise[part+'_seeds']:
            rng = np.random.default_rng(np.random.SeedSequence([seed,rec['case_id'],rec['tick']]))
            zz = z+rng.normal(0.,sigma,z.shape)
            g = v7.work(rgb,zz,sigma=sigma,**opts)
            mark,_ = reader._scissors(rgb_float,zz)
            row = dict(part=part,seed=seed,chart_valid=g is not None,marker_valid=mark['valid'])
            if g is not None and mark['valid'] and 'v7_room_mm' in clean:
                noisy_tip = np.asarray(mark['tip'])
                delta = g.limit-noisy_tip
                row.update(axis_delta_deg=axis_angle(g.direction,new.direction),
                           delta_visual_room_mm=project(delta,g.direction)-clean['v7_room_mm'],
                           delta_limit_on_raw_axis_mm=project(g.limit-new.limit,new.direction),
                           delta_tip_on_raw_axis_mm=project(tip-noisy_tip,new.direction),
                           delta_axis_mm=project(delta,g.direction-new.direction))
            elif not mark['valid']:
                row['marker_reason'] = mark['reason']
            draws.append(row)
    return dict(**{k:v for k,v in rec.items() if k!='marker'},clean=clean,draws=draws)


def outer_gate(model,plan):
    captures = load(CAPTURE/'decision.json')['captures']
    if set(captures)!={'work_closed','work_open','narrow_before_backoff','narrow_entry_clear'}:
        raise ValueError('unexpected sealed outer gate exposures')
    result = dict(narrow={},goals={})
    for key in ('narrow_before_backoff','narrow_entry_clear'):
        m = captures[key]
        with np.load(verified(CAPTURE/m['file'],m['sha256']),allow_pickle=False) as f:
            if f['depth'].shape!=(1920,2560) or m['fov_deg']!=20 or not m['closed']:
                raise ValueError('unexpected sealed narrow optics')
            result['narrow'][key] = intact_preflight(f['rgb'],f['depth'],far=f['far'].item())['passed']
    for goal in (9,10):
        stats,front,axes = {},[],[]
        for key in ('work_closed','work_open'):
            m = captures[key]
            with np.load(verified(CAPTURE/m['file'],m['sha256']),allow_pickle=False) as f:
                R,t = f['cam_R'],f['cam_t']
                if f['depth'].shape!=(960,1280) or m['fov_deg']!=70 or m['closed']!=(key=='work_closed'):
                    raise ValueError('unexpected sealed work optics')
                g = v7.work(f['rgb'],f['depth'],focal=f['focal'].item(),far=f['far'].item(),
                            sigma=0.,prefix_mm=0,goal_mm=goal)
            mark = m['marker']
            if g is None or not mark['valid']:
                stats[key] = dict(passed=False)
                continue
            xyz = v7.WorkChart(0,R@g.front+t,R@g.direction,R@g.opening,R@g.aim+t,R@g.limit+t)
            pose = Work(xyz,R@mark['tip']+t,R@mark['jaw_axis'],R@mark['opening_axis'],-1,0)
            room = World(None,goal,model).sweep_room(pose)
            stats[key] = dict(passed=bool(room>=.0002),sweep_room_mm=float(1000*room))
            front.append(xyz.front)
            axes.append(xyz.direction)
        if len(front)==2:
            distance = float(1000*np.linalg.norm(front[0]-front[1]))
            angle = axis_angle(*axes)
            result['goals'][goal] = dict(work=stats,front_delta_mm=distance,axis_delta_deg=angle,
                legacy_gate=all(s['passed'] for s in stats.values()) and distance<.40 and angle<5.,
                tighter_front_gate=distance<plan['clean_front_closed_open_delta_mm_lt'],
                tighter_axis_gate=angle<plan['clean_axis_closed_open_delta_deg_lt'])
        else:
            result['goals'][goal] = dict(work=stats,legacy_gate=False,tighter_front_gate=False,
                                         tighter_axis_gate=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    plan = load(PLAN)
    registered = registration(plan)
    args.out.mkdir(exist_ok=False)
    seal = args.out/'registration.json'
    write(seal,registered)
    cal = load(CAL)
    if not cal['qualified']:
        raise ValueError('unqualified rigid marker calibration')
    model = CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in cal['calibration'].items()})
    rows = []
    for rec in registered['frames']:
        rows.append(measure_frame(rec,registered['noise_plan'],model))
        print('measured',rec['path'],flush=True)
    measurements = args.out/'measurements.json'
    write(measurements,rows)
    rows = load(measurements)
    early_rows = [r for r in rows if r['prefix_mm']<=5]
    late_rows = [r for r in rows if r['prefix_mm']>=6]
    late = summarize(late_rows)
    if len(rows)!=15 or summarize(rows)['draws']!=360 or late['draws']!=72:
        raise ValueError('registered draw count mismatch')
    field = 'fixed_v4_axis_limit_delta_mm'
    bias = [abs(r['clean'][field]) for r in rows if field in r['clean']]
    outer = outer_gate(model,plan)
    goals = outer['goals'].values()
    gates = dict(clean_fixed_axis=len(bias)==15 and max(bias)<plan['clean_abs_fixed_axis_limit_delta_mm_lt'],
                 legacy_outer=outer['narrow']['narrow_entry_clear'] and all(g['legacy_gate'] for g in goals),
                 tighter_outer_front=all(g['tighter_front_gate'] for g in goals),
                 tighter_outer_axis=all(g['tighter_axis_gate'] for g in goals),
                 late_pairs=late['paired_valid']>=plan['late_pair_min_count'],
                 late_axis=late['scored']==late['paired_valid']>0
                 and late['max_axis_delta_deg']<plan['late_noisy_axis_delta_deg_lt'],
                 late_optimism=late['scored']==late['paired_valid']>0
                 and late['max_optimism_delta_mm']<plan['late_noisy_optimism_delta_mm_lt'])
    result = dict(passed=all(gates.values()),authorizes_motion=False,gates=gates,
                  clean_pairs=len(bias),clean_max_abs_fixed_axis_limit_delta_mm=max(bias,default=None),
                  outer=outer,early={p:summarize(early_rows,p) for p in ('development','check')},
                  late={**late,**{p:summarize(late_rows,p) for p in ('development','check')}},
                  registration_sha256=digest(seal),measurements_sha256=digest(measurements))
    write(args.out/'decision.json',result)
    print('offline gate:',result['passed'],gates,flush=True)


if __name__=='__main__':
    main()
