"""Single-exposure direct seam metrology; an offline development screen, not a controller."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision')]
import m5_v3_perception_v11 as work
from audit_m5_v4_bend_rim_v1 import OUT as AUDIT, PARENT, Sheet, projected, rays
from cut_perception import CutPerception
from m5_v3_controller_v7 import sensor_margin, sweep_projection
from m5_v4_center_screen_v1 import KEYS, measure, witnesses
from run_m5_v4_bend_terminal_v1 import NOISE, base, check, load, sha256, verify, write
from score_m5_v4_bend_terminal_v1 import margin_gate

ID = 'm5_v4_center_screen_v1'
PLAN = ROOT/f'config/task/{ID}.json'
OUT = ROOT/f'validation/task/{ID}'
SUITE = ROOT/f'validation/safety/regressions_{ID}.json'
PLANE = ROOT/'validation/task/m5_v4_plane_blind_v2'


def register():
    plan = load(PLAN)
    prior = load(verify(PARENT/'decision.json',plan['terminal_sha256']))
    audit = load(verify(AUDIT/'decision.json',plan['rim_audit_sha256']))
    metric = load(verify(PLANE/'decision.json',plan['plane_transfer_sha256']))['transfer']
    check(metric['metric_passed'] and not metric['margin']['qualified'],'unexpected plane predecessor')
    pop = load(verify(AUDIT/'registration.json',audit['registration_sha256']))
    sources = dict(load(verify(PARENT/'registration.json',prior['registration_sha256']))['sources'])
    sources.update(pop['sources'])
    suite = load(SUITE)
    check(suite['success'] and not suite['skipped'] and suite['tests_run']>=503,'offline suite refused')
    for p in (PLAN,Path(__file__).resolve(),ROOT/'scripts/vision/m5_v4_center_screen_v1.py',
              ROOT/'scripts/tests/test_m5_v4_center_screen_v1.py',ROOT/'scripts/analysis/audit_m5_v4_bend_rim_v1.py',
              ROOT/'scripts/vision/cell_certificate_v1.py',ROOT/'scripts/analysis/score_m5_v4_bend_terminal_v1.py'):
        k = p.relative_to(ROOT).as_posix()
        sources[k] = suite['source_sha256'][k]
    for p in (PARENT/'decision.json',AUDIT/'decision.json',AUDIT/'registration.json',SUITE,PLANE/'decision.json',
              verify(PARENT/'optical_predictions.json',prior['optical_sha256'])):
        sources[p.relative_to(ROOT).as_posix()] = sha256(p)
    cases = {}
    for case,h in prior['run_hashes'].items():
        p = verify(PARENT/case/'run.json',h)
        sources[p.relative_to(ROOT).as_posix()] = h
        r = load(p)
        cases[case] = [m['sha256'] for name,m in r['captures'].items()
                       if name in r['scoring_sidecars'] and m['closures']>=3]
    pop = pop['records']
    check(len(pop)==6 and all(len(v)==4 for v in cases.values()) and
          {r['h'] for r in pop}==set.union(*(set(v) for v in cases.values())),'wrong late population')
    for k,h in sources.items():
        verify(ROOT/k,h)
    OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',dict(plan=plan,sources=sources,records=pop,cases=cases))


def lock():
    reg = load(OUT/'registration.json')
    for k,h in reg['sources'].items():
        verify(ROOT/k,h)
    return reg


def model():
    return base.CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v
                             for k,v in load(base.FK)['calibration'].items()})


def predict():
    reg = lock()
    fk = model()
    noise = load(NOISE)
    baseline = load(PARENT/'optical_predictions.json')['predictions']
    frames = {}
    for rec in reg['records']:
        case,name,h,n = (rec[k] for k in ('case','name','h','prefix'))
        run = load(PARENT/case/'run.json')
        m = run['captures'][name]
        with np.load(verify(PARENT/case/m['file'],h),allow_pickle=False) as f:
            rgb,z,rot,focal,far = f['rgb'],f['depth'],f['cam_R'],f['focal'].item(),f['far'].item()
        tick = m['tick']
        check(z.shape==(960,1280) and np.isclose(focal,640/np.tan(np.deg2rad(35)),atol=1e-6,rtol=0)
              and m['fov_deg']==70,'changed optics')
        ref = baseline[case]['work'][name]
        evidence = witnesses(rgb)
        marker = CutPerception(1280,960,focal,far,fk.rcm,tcp_local=fk.tip_Y,jaw_axis_local=fk.tip_R[:,2],
                               opening_axis_local=fk.tip_R[:,0],ray_marker_fit=True)
        floating = rgb.astype(float)
        draws = []
        stage = 'pre_advance' if tick==run['fork']['tick'] else ('pre_close' if not m['closed'] else 'observe')
        for (split,seed),orig in zip([('clean',None)]+[(k,s) for k in ('development','check')
                                    for s in noise[k+'_seeds']],ref,strict=True):
            sigma = 0. if seed is None else .0005
            depth = z if seed is None else z+np.random.default_rng(np.random.SeedSequence(
                [seed,2,tick])).normal(0.,sigma,z.shape)
            sol = work.work(rgb,depth,focal=focal,far=far,sigma=sigma,prefix_mm=n,goal_mm=10 if n==9 else 9)
            axis = None if sol is None else sol.direction
            mark,_ = marker._scissors(floating,depth)
            check((split,seed,n,m['closed'],axis is not None,bool(mark['valid']))==
                  tuple(orig[k] for k in ('split','seed','prefix','closed','chart','marker')),'reference replay differs')
            if axis is not None and mark['valid']:
                check(np.allclose(rot@axis,orig['axis'],atol=1e-12,rtol=0),'wrong optical axis')
            central = measure(evidence,depth,focal=focal,far=far,sigma=sigma,axis=axis)
            d = dict(central,split=split,seed=seed,closed=m['closed'],prefix=n,axis=None if axis is None else axis.tolist(),
                     marker=bool(mark['valid']),marker_reason=mark.get('reason',''),stage=stage)
            if d['interval_m'] is not None and mark['valid']:
                tip,jaw,hinge = (np.asarray(mark[k]) for k in ('tip','jaw_axis','opening_axis'))
                room = d['interval_m'][0]-tip@axis
                R = np.column_stack((hinge,np.cross(jaw,hinge),jaw))
                check(np.allclose(R.T@R,np.eye(3),rtol=0,atol=1e-6) and np.linalg.det(R)>0,'invalid marker pose')
                sweep = max(sweep_projection(R@a,R@b,np.zeros(3),axis) for a,b in fk.blade_local())
                clearance = room-sweep-.0003-sensor_margin(sigma)
                d.update(row=10,tip=tip.tolist(),room_mm=1000*room,sweep_clearance_mm=1000*clearance,
                         advance_after_reserve_mm=1000*(clearance-.00015))
            draws.append(d)
        frames[h] = dict(case=case,name=name,tick=tick,ink=evidence[2],draws=draws)
    write(OUT/'optical_predictions.json',dict(registration_sha256=sha256(OUT/'registration.json'),frames=frames))


def raster_score(nodes,edges,focal,shape,observed):
    """Expected native pixel centers of each exact patch vs sheet-only first hits and RGB."""
    sheet = Sheet(nodes,edges)
    ids = {tuple(face) for face in sheet.ids}
    pieces = {k:[] for k in KEYS}
    for j in (9,10):
        for i in (10,11,12):
            for half,idx in enumerate((((j,i),(j,i+1),(j+1,i+1)),((j,i),(j+1,i+1),(j+1,i)))):
                face = (j,i,half)
                if face not in ids:
                    continue
                tri = nodes[tuple(np.array(idx).T)]
                if i==11:
                    if (j,half) in ((9,1),(10,0)):
                        pieces[KEYS[2+(j==10)]].append((tri,face))
                else:
                    for k,(r,c) in enumerate(idx):
                        if r==10 and c in (11,12):
                            pieces[KEYS[c-11]].append((tri[k]+.15*(tri-tri[k]),face))
    result = {}
    for k,rc in zip(KEYS,observed,strict=True):
        potential,visible = set(),set()
        for tri,face in pieces[k]:
            corners = projected(tri,shape,focal)
            lo = np.maximum(np.floor(corners.min(0)).astype(int),0)
            hi = np.minimum(np.ceil(corners.max(0)).astype(int)+1,shape)
            if np.any(lo>=hi):
                continue
            A = (corners[1:]-corners[0]).T
            if abs(np.linalg.det(A))<1e-12:
                continue
            pix = np.moveaxis(np.indices(hi-lo),0,-1).reshape(-1,2)+lo
            uv = (pix-corners[0])@np.linalg.inv(A).T
            hit = pix[(uv>=-1e-9).all(1)&(uv.sum(1)<=1+1e-9)]
            if len(hit):
                _,first = sheet.hits(rays(hit,shape,focal))
                potential.update(map(tuple,hit.tolist()))
                visible.update(map(tuple,hit[(first==face).all(1)].tolist()))
        seen = set(map(tuple,rc.tolist()))
        result[k] = dict(projected_pixels=len(potential),sheet_visible=len(visible),observed=len(seen),
                         sheet_and_observed=len(visible&seen),observed_not_predicted=len(seen-visible))
    return result


def summarize(rows):
    strata = {(r['prefix'],r['closed']) for r in rows}
    coverage = all({(r['prefix'],r['closed']) for r in rows if r['split']==s and 'row' in r}==strata
                   for s in ('clean','development','check'))
    margin = margin_gate(rows,coverage)
    for split,m in margin['parts'].items():
        pop = [r for r in rows if r['split']==split]
        m['marker_valid'] = sum(r['marker'] for r in pop)
        m['axis_valid'] = sum(r['axis'] is not None for r in pop)
        m['central_witness'] = sum(r['cap_witness'] or r['face_witness'] for r in pop)
        m['central_interval'] = sum(r['interval_m'] is not None for r in pop)
        m['reasons'] = dict(Counter(r['reason'] or ('missing marker' if not r['marker'] else 'accepted') for r in pop))
        m['routes'] = dict(Counter(r['route'] for r in pop if r['route']))
        m['min_optimism_mm'] = min((r['optimism_mm'] for r in pop if 'row' in r),default=None)
    intervals = [r for r in rows if r['interval_m'] is not None]
    axes = [r['axis_error_deg'] for r in rows if r['axis'] is not None]
    needed = [r for r in rows if r['stage']!='observe' and r['split']=='clean']
    useful = bool(needed) and all('row' in r for r in needed) and all(
        r['advance_after_reserve_mm']>=.2 if r['stage']=='pre_advance' else r['sweep_clearance_mm']>=0 for r in needed)
    return dict(margin=margin,interval_containment=all(r['contains'] for r in intervals) if intervals else None,
                max_axis_deg=max(axes,default=None),native_action_room=useful,
                utility=[{k:r.get(k) for k in ('name','case','stage','advance_after_reserve_mm','sweep_clearance_mm')}
                         for r in needed])


def score():
    reg = lock()
    pp = OUT/'optical_predictions.json'
    optical = load(pp)
    verify(OUT/'registration.json',optical['registration_sha256'])
    fk = model()
    scored,frames = [],{}
    for rec in reg['records']:
        case,name,h,review = (rec[k] for k in ('case','name','h','review'))
        folder = PARENT/case
        r = load(folder/'run.json')
        m = r['captures'][name]
        pred = optical['frames'][h]
        with (np.load(verify(folder/m['file'],h),allow_pickle=False) as f,
              np.load(verify(folder/r['scoring_sidecars'][name]['file'],rec['files'][1]),allow_pickle=False) as s,
              np.load(verify(folder/r['review_sidecars'][review]['file'],rec['files'][2]),allow_pickle=False) as e):
            check(s['frame_sha256'].item()==h and m['tick']==s['tick'].item()==pred['tick'] and
                  s['closed'].item()==m['closed'] and np.array_equal(s['q'],f['q']) and
                  e['frame_sha256'].item()==r['captures'][review]['sha256'] and e['closures'].item()==m['closures']
                  and e['stamp'].item()<=s['simulation_time_s'].item(),'unbound evaluation reference')
            cam = f['cam_R']
            nodes = (s['nodes_world']-f['cam_t'])@cam
            tcp = (s['tcp_world']-f['cam_t'])@cam
            seam = nodes[:,11:13].mean(1)
            true_axis = (np.arange(13)-6)@seam[:13]
            true_axis /= np.linalg.norm(true_axis)
            err = float(1000*np.linalg.norm(fk.pose(s['q'])[:,3]-s['tcp_world']))
            raster = raster_score(nodes,e['edges'],f['focal'].item(),f['depth'].shape,witnesses(f['rgb'])[0])
            frames[h] = dict(case=case,name=name,ink=pred['ink'],raster=raster,fk_mm=err)
            for d in pred['draws']:
                row = dict(d,frame_sha256=h,name=name,case=case)
                if row['axis'] is not None:
                    axis = np.asarray(row['axis'])
                    row['axis_error_deg'] = float(np.degrees(np.arccos(np.clip(axis@true_axis,-1,1))))
                    if row['interval_m'] is not None:
                        lo,hi = row['interval_m']
                        c = seam[10]@axis
                        row['contains'] = bool(lo<=c<=hi)
                        row['interval_width_mm'] = 1000*(hi-lo)
                        if 'row' in row:
                            row['optimism_mm'] = row['room_mm']-1000*((seam[10]-tcp)@axis)
                scored.append(row)
    check(len(scored)==150 and len({(r['frame_sha256'],r['seed']) for r in scored})==150,'missing/duplicate draws')
    by = {k:summarize([r for r in scored if r['frame_sha256'] in hashes]) for k,hashes in reg['cases'].items()}
    summary = summarize(scored)
    gates = dict(coverage_and_room=all(s['margin']['qualified'] for s in [summary,*by.values()]),
                 containment=summary['interval_containment'] is True,axis=summary['max_axis_deg'] is not None
                 and summary['max_axis_deg']<5,rigid_fk=all(v['fk_mm']<=.010 for v in frames.values()),
                 native_action_room=all(s['native_action_room'] for s in by.values()))
    mp = OUT/'measurements.json'
    write(mp,scored)
    write(OUT/'decision.json',dict(passed=all(gates.values()),gates=gates,summary=summary,cases=by,frames=frames,
          registration_sha256=sha256(OUT/'registration.json'),optical_sha256=sha256(pp),measurements_sha256=sha256(mp),
          authorizes_motion=False))
    print('gates',gates,'parts',summary['margin']['parts'],flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('register','predict','score'))
    {'register':register,'predict':predict,'score':score}[p.parse_args().stage]()
