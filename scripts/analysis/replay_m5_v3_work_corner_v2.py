"""Sealed main-path, forced-corner and stopped-frame screens for v10; no simulator."""
import argparse
import json
from pathlib import Path

import numpy as np

import calibrate_m5_v3_sensor_margin_v2 as calibration
from calibrate_m5_v3_sensor_margin_v1 import CAL, ROOT, digest, load, verified, write
import m5_v3_perception_v10 as v10
from m5_v3_rigid_fk import CutterFK
import replay_m5_v3_work_denoise_v4 as denoise
from replay_m5_v3_work_denoise_v1 import project


PLAN = ROOT/'config/task/m5_v3_work_corner_v2.json'
WORK = ROOT/'validation/task/m5_v3_work_denoise_v4'
TRUTH = ROOT/'validation/task/m5_v3_sensor_margin_calibration_v2'
MARGIN = ROOT/'validation/task/m5_v3_sensor_margin_calibration_v3'
STOP = ROOT/'validation/task/m5_v3_positive_outer_v4/nominal'
VARIANTS = {'left':(0,), 'right':(1,), 'bilateral':(0,1)}


def seal_inputs(plan):
    work_decision,margin = load(WORK/'decision.json'),load(MARGIN/'decision.json')
    previous = load(TRUTH/'decision.json')
    for d,base in ((work_decision,WORK),(margin,MARGIN)):
        if len(d['gates'])!=7 or not all(d['gates'].values()):
            raise ValueError('unqualified predecessor gates: '+str(base))
        verified(base/'registration.json',d['registration_sha256'])
        verified(base/'measurements.json',d['measurements_sha256'])
    if margin['sensor_margin']!=plan['sensor_margin'] or not margin['qualified']:
        raise ValueError('frozen margin mismatch')
    sources = {}
    for pins in (plan['sources'],load(WORK/'registration.json')['sources'],load(MARGIN/'registration.json')['sources']):
        for name,pin in pins.items():
            verified(ROOT/name,pin)
            sources[name] = pin
    paths = [WORK/'measurements.json',MARGIN/'measurements.json']
    paths += [verified(TRUTH/(name+'.json'),previous[name+'_sha256']) for name in ('fit','check')]
    fit = load(TRUTH/'fit.json')
    paths += [verified(TRUTH/(name+'.json'),fit[name+'_sha256']) for name in ('clean','development')]
    paths += [PLAN,Path(__file__).resolve(),Path(v10.__file__).resolve()]
    sources.update({p.relative_to(ROOT).as_posix():digest(p) for p in paths})
    return sources


def main_path(model,samples,noise,cal_plan):
    old = load(WORK/'registration.json')
    denoise.v8 = v10
    expected = load(WORK/'measurements.json')
    diffs = []
    seen = 0
    for rec,prev in zip(old['frames'],expected,strict=True):
        actual = denoise.measure_frame(rec,old['noise_plan'],model)
        if actual!=prev:
            if actual['clean']!=prev['clean']:
                diffs.append(dict(frame=rec['path'],part='clean'))
            for a,b in zip(actual['draws'],prev['draws'],strict=True):
                if a!=b:
                    diffs.append(dict(frame=rec['path'],part=b['part'],seed=b['seed']))
        seen += len(actual['draws'])
    outer = json.loads(json.dumps(denoise.outer_gate(model,old['plan'])))
    outer_same = outer==load(WORK/'decision.json')['outer']
    if len(expected)!=15 or seen!=360:
        raise ValueError('wrong denoise population')
    print('denoise identity:',not diffs and outer_same,flush=True)
    calibration.reader = v10
    measured = {'clean':[calibration.measure(x,x['z'],0.) for x in samples]}
    for split in ('development','check'):
        measured[split] = calibration.replay(samples,cal_plan,noise,split)
    cal_diffs = []
    for part,rows in measured.items():
        if len(rows)!=(15 if part=='clean' else 180):
            raise ValueError('wrong calibration population')
        for a,b in zip(rows,load(TRUTH/(part+'.json')),strict=True):
            if a!=b:
                cal_diffs.append(dict(part=part,frame=b['frame'],seed=b['seed']))
    print('calibration identity:',not cal_diffs,flush=True)
    return dict(passed=not diffs and outer_same and not cal_diffs,
                denoise_frames=len(expected),denoise_draws=seen,outer_identical=outer_same,
                denoise_differences=diffs,calibration_frames=len(measured['clean']),
                calibration_draws=len(measured['development'])+len(measured['check']),
                calibration_differences=cal_diffs)


def corner(ctx,side,j,pink_override=None):
    label,depth,valid,pink,across,tol,sigma,focal = ctx
    return v10._fallback_station(label,depth,valid,pink if pink_override is None else pink_override,
                                side,across,tol,sigma,focal,j)


def occluded_pink(ctx,side,j):
    label,depth,valid,pink,_,tol,_,_ = ctx
    masked = pink.copy()
    for a in np.argwhere(valid & (label==13*side+j-1)):
        for step in ((-1,0),(1,0),(0,-1),(0,1)):
            b = a+step
            if ((b<0)|(b>=depth.shape)).any() or not valid[tuple(b)] or label[tuple(b)]!=13*side+j:
                continue
            if abs(depth[tuple(a)]-depth[tuple(b)])>tol:
                continue
            box = np.s_[max(0,a[0]-2):a[0]+3,max(0,a[1]-2):a[1]+3]
            coords = np.argwhere(masked[box])+np.maximum(a-2,0)
            near = np.sum((coords-(a+b)*.5)**2,axis=1)<=4.
            masked[tuple(coords[near].T)] = False
    return masked,int(np.count_nonzero(pink & ~masked))


def forced_draw(s,z,sigma,seed,part):
    rec = s['rec']
    n = rec['prefix_mm']
    obs = v10._observe(s['rgb'],z,focal=s['focal'],far=s['far'],sigma=sigma,prefix_mm=n,goal_mm=10)
    if obs is None:
        raise ValueError('missing shared observations on sealed frame')
    sides,ctx = obs
    mark,_ = s['markers']._scissors(s['rgb_float'],z)
    points = {side:corner(ctx,side,10) for side in (0,1)}
    rows = []
    for variant,force in VARIANTS.items():
        stations = [dict(bank) for bank in sides]
        for side in force:
            stations[side].pop(10,None)
            if points[side] is not None:
                stations[side][10] = points[side]
        solution = v10._chart(stations,n,10)
        r = dict(frame=rec['frame'].name,prefix_mm=n,closed=rec['closed'],part=part,seed=seed,
                 sigma_m=sigma,variant=variant,accepted_sides=[s for s in force if points[s] is not None],
                 chart_valid=solution is not None,marker_valid=bool(mark['valid']))
        if solution is not None and mark['valid']:
            g,limit_row,_,_ = solution
            if limit_row!=10:
                raise ValueError('forced draw lost the required row-10 bound')
            direction = s['R']@g.direction
            lim,tip = s['R']@g.limit+s['t'],s['R']@mark['tip']+s['t']
            observed = lim-tip
            reference = s['row10']-s['tcp']
            r['optimism_mm'] = project(observed-reference,direction)
        rows.append(r)
    return rows


def forced_path(samples,noise,cal_plan,margin):
    late = [x for x in samples if x['rec']['prefix_mm']>=6]
    if len(late)!=5:
        raise ValueError('wrong late cohort')
    rows,controls = [],[]
    for s in late:
        rec = s['rec']
        z = s['z']
        options = dict(focal=s['focal'],far=s['far'],sigma=0.,prefix_mm=rec['prefix_mm'],goal_mm=10)
        _,ctx = v10._observe(s['rgb'],z,**options)
        for side in (0,1):
            masked,removed = occluded_pink(ctx,side,10)
            controls.append(dict(frame=rec['frame'].name,side=side,removed=removed,
                                 qualified_before=corner(ctx,side,10) is not None,
                                 refused_after=corner(ctx,side,10,masked) is None))
        rows.extend(forced_draw(s,z,0.,None,'clean'))
        sigma = noise['depth_sigmas_m'][1]
        for part in ('development','check'):
            for seed in noise[part+'_seeds']:
                rng = np.random.default_rng(np.random.SeedSequence([seed,cal_plan['noise_case_id'],rec['tick']]))
                rows.extend(forced_draw(s,z+rng.normal(0.,sigma,z.shape),sigma,seed,part))
        print('forced paths:',rec['frame'].name,flush=True)
    results = {}
    for name in VARIANTS:
        parts = {}
        for part,count,minimum in (('clean',5,5),('development',60,57),('check',60,57)):
            all_rows = [r for r in rows if (r['variant'],r['part'])==(name,part)]
            paired = [r for r in all_rows if 'optimism_mm' in r]
            if len(all_rows)!=count:
                raise ValueError('wrong forced draw count')
            cap = 1000*(margin['c0_m']+margin['k']*all_rows[0]['sigma_m'])
            failures = [r for r in paired if not r['optimism_mm']<=cap]
            parts[part] = dict(draws=count,paired=len(paired),chart_valid=sum(r['chart_valid'] for r in all_rows),
                               max_optimism_mm=max((r['optimism_mm'] for r in paired),default=None),
                               margin_mm=cap,violations=len(failures),
                               worst=max(failures,key=lambda r:r['optimism_mm'],default=None),
                               passed=len(paired)>=minimum and not failures)
        results[name] = parts
    occlusion = all(r['removed']>0 and r['qualified_before'] and r['refused_after'] for r in controls)
    passed = occlusion and all(p['passed'] for parts in results.values() for p in parts.values())
    return dict(passed=passed,variants=results,occlusion_passed=occlusion,occlusion_controls=controls),rows


def stopped_frame(model):
    capture = load(STOP/'run.json')['captures']['015_work']
    path = verified(STOP/capture['file'],capture['sha256'])
    with np.load(path,allow_pickle=False) as f:
        rgb,z,focal,far = f['rgb'],f['depth'],f['focal'].item(),f['far'].item()
    mark,_ = calibration.CutPerception(1280,960,focal,far,model.rcm,tcp_local=model.tip_Y,
        jaw_axis_local=model.tip_R[:,2],opening_axis_local=model.tip_R[:,0],ray_marker_fit=True
        )._scissors(rgb.astype(float),z)
    goals = {}
    for goal in (9,10):
        solution,sides,recovered = v10._solve_work(rgb,z,focal=focal,far=far,sigma=0.,prefix_mm=7,goal_mm=goal)
        _,row,_,checked = solution if solution is not None else (None,None,None,{})
        complete = sides is not None and all(set(b)==set(range(13)) for b in sides)
        goals[goal] = dict(side_rows=[sorted(b) for b in sides] if sides is not None else [],
                           recovered=recovered,checked=sorted(checked),limit_row=row,
                           passed=complete and sorted(checked)==list(range(13)) and row==10)
    return dict(passed=bool(mark['valid']) and all(g['passed'] for g in goals.values()),
                marker_valid=bool(mark['valid']),goals=goals,frame_sha256=capture['sha256'])


def run(out):
    plan = load(PLAN)
    sources = seal_inputs(plan)
    out.mkdir(exist_ok=False)
    seal = out/'registration.json'
    write(seal,dict(plan=plan,sources=sources))
    cal_plan,noise = load(calibration.PLAN),load(calibration.NOISE)
    records,_,probe = calibration.registered_frames(TRUTH,cal_plan)
    model = CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in load(CAL)['calibration'].items()})
    samples = [calibration.load_sample(r,model,probe['final_q']) for r in records]
    recovery = stopped_frame(model)
    print('stopped-frame geometry:',recovery,flush=True)
    forced,rows = forced_path(samples,noise,cal_plan,plan['sensor_margin'])
    draw_path = out/'forced_draws.json'
    write(draw_path,rows)
    print('forced calibration:',forced['passed'],flush=True)
    main = main_path(model,samples,noise,cal_plan)
    gates = dict(main_path=main['passed'],forced_path=forced['passed'],stopped_frame=recovery['passed'])
    result = dict(passed=all(gates.values()),authorizes_motion=False,gates=gates,main_path=main,
                  forced_path=forced,stopped_frame=recovery,registration_sha256=digest(seal),
                  forced_draws_sha256=digest(draw_path))
    write(out/'decision.json',result)
    print('offline gates:',gates,flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',required=True,type=Path)
    run(cli.parse_args().out)
