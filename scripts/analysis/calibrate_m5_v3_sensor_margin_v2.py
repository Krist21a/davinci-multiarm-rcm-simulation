"""Fit the v8 margin on tagged same-exposure references; evaluate held-out seeds once."""
import argparse
import math
from pathlib import Path

import numpy as np

from calibrate_m5_v3_sensor_margin_v1 import CAL, ROOT, digest, load, verified, write
from cut_perception import CutPerception
import m5_v3_perception_v8 as reader
from m5_v3_rigid_fk import CutterFK
from replay_m5_v3_work_denoise_v1 import axis_angle, project
from score_m5_v3_outer_positive_v3 import examine, verdict
from workspace import NU, NV


PLAN = ROOT/'config/task/m5_v3_sensor_margin_calibration_v2.json'
NOISE = ROOT/'config/task/m5_v3_sensor_margin_calibration_v1.json'
BASE = ROOT/'config/task/m5_v3_positive_outer_v3.json'


def registered_frames(batch,plan):
    reg = batch/'registration.json'
    raw_path = batch/'run.json'
    seal,raw = load(reg),load(raw_path)
    if seal['calibration_plan']!=plan or digest(reg)!=raw['registration_sha256']:
        raise ValueError('calibration registration mismatch')
    for pins in (plan['sources'],seal['sources']):
        for name,pin in pins.items():
            verified(ROOT/name,pin)
    for p in (PLAN,Path(__file__).resolve(),Path(reader.__file__).resolve()):
        if seal['sources'].get(p.relative_to(ROOT).as_posix())!=digest(p):
            raise ValueError('scorer or candidate not preregistered')
    if raw['state']!='DONE' or any(raw.get(k) for k in ('failure','scoring_error','cleanup_error')):
        raise ValueError('reference collector refused: '+str(raw.get('failure') or raw.get('cleanup_error')
                                                              or raw.get('scoring_error')))
    optical,_ = examine(batch,'nominal',load(BASE))
    receipts = optical['receipts']
    if [[x['verdict'],x['prefix_mm']] for x in receipts]!=plan['required_receipts']:
        raise ValueError('wrong independently replayed reference sequence')
    start,end = raw['nominal_end'],raw['probe_end']
    original = dict(raw,**start,moves=raw['moves'][:raw['nominal_move_count']])
    if not (verdict(optical,original)['passed'] and start['closures']==end['closures']==4
            and end['closed'] is False and np.allclose(end['final_q'],raw['entrance_q'],rtol=0,atol=1e-6)
            and start['constraints_sha256']==end['constraints_sha256']
            and start['final_structure']==end['final_structure']
            and end['final_tissue_stats']['torn']==0 and end['rcm']
            and all(v['failures']==0 and v['max_drift_m']<=.0005 and v['collision_blocks']==0 for v in end['rcm'])):
        raise ValueError('nominal ready, probe clearance, material identity or safety failed')
    extra_moves = raw['moves'][raw['nominal_move_count']:]
    if (len(extra_moves)!=1 or not extra_moves[0]['closed'] or not extra_moves[0]['accepted']
            or not np.allclose(extra_moves[0]['q'],raw['entrance_q'],rtol=0,atol=1e-6)):
        raise ValueError('terminal probe did not follow the sole closed entrance-clear waypoint')
    views = {name:m for name,m in raw['captures'].items() if m['fov_deg']==70}
    log = raw['truth_reads_by_purpose']
    if (set(views)!=set(raw['scoring_sidecars']) or 'untagged' in log
            or log['scoring']['nodes']!=len(views) or log['scoring']['tcp']!=len(views)):
        raise ValueError('scoring sidecars and purpose counts do not match exposures')
    out = []
    for name,m in sorted(views.items(),key=lambda kv:kv[1]['tick']):
        prev = [x['prefix_mm'] for x in receipts if x['tick']<m['tick']]
        s = raw['scoring_sidecars'][name]
        if s['file']!='scoring/'+name+'.npz':
            raise ValueError('reference sidecar name mismatch')
        out.append(dict(frame=verified(batch/m['file'],m['sha256']),
                        scoring=verified(batch/s['file'],s['sha256']),
                        prefix_mm=prev[-1] if prev else 0,**{k:m[k] for k in ('tick','closed','marker','sha256')}))
    expected = {(n,b) for n in plan['prefixes'] for b in (True,False)}
    terminal = [r for r in out if r['prefix_mm']==9]
    if {(r['prefix_mm'],r['closed']) for r in out}!=expected or [r['closed'] for r in terminal]!=[True,False]:
        raise ValueError('incomplete prefix/jaw coverage')
    return out,start,end


def load_sample(rec,model,entrance_q):
    with np.load(rec['frame'],allow_pickle=False) as f, np.load(rec['scoring'],allow_pickle=False) as s:
        if (s['frame_sha256'].item()!=rec['sha256'] or s['tick'].item()!=rec['tick']
                or s['closed'].item()!=rec['closed'] or not np.array_equal(s['q'],f['q'])):
            raise ValueError('unpaired scoring sidecar: '+rec['frame'].name)
        rgb,z,q = f['rgb'],f['depth'],f['q']
        R,t = f['cam_R'],f['cam_t']
        focal,far = f['focal'].item(),f['far'].item()
        nodes,tcp,stamp = s['nodes_world'],s['tcp_world'],s['simulation_time_s'].item()
    if (z.shape!=(960,1280) or rgb.shape!=z.shape+(3,) or nodes.shape!=(NV,NU,3) or tcp.shape!=(3,)
            or not np.isfinite(nodes).all() or not np.isfinite(tcp).all() or not np.isfinite(stamp)
            or not np.isclose(focal,640/np.tan(np.deg2rad(35)),rtol=0,atol=1e-6)
            or R.shape!=(3,3) or not np.allclose(R.T@R,np.eye(3),rtol=0,atol=1e-6)
            or np.linalg.det(R)<=0 or not np.isfinite(t).all()
            or (rec['prefix_mm']==9 and not np.allclose(q,entrance_q,rtol=0,atol=1e-6))):
        raise ValueError('invalid same-state RGB-D geometry: '+rec['frame'].name)
    seam = .5*(nodes[:,11]+nodes[:,12])
    slope = (np.arange(13)-6)@seam[:13]
    terminal = seam[10]-seam[9]
    if min(np.linalg.norm(slope),np.linalg.norm(terminal))<=1e-9:
        raise ValueError('degenerate exact seam reference')
    return dict(rec=rec,rgb=rgb,rgb_float=rgb.astype(float),z=z,focal=focal,far=far,R=R,t=t,stamp=stamp,
                tcp=tcp,row10=seam[10],axis=slope/np.linalg.norm(slope),end_axis=terminal/np.linalg.norm(terminal),
                fk_mm=float(1000*np.linalg.norm(model.pose(q)[:,3]-tcp)),
                markers=CutPerception(1280,960,focal,far,model.rcm,tcp_local=model.tip_Y,
                    jaw_axis_local=model.tip_R[:,2],opening_axis_local=model.tip_R[:,0],ray_marker_fit=True))


def measure(s,z,sigma,seed=None):
    rec = s['rec']
    chart = reader._work_evidence(s['rgb'],z,focal=s['focal'],far=s['far'],sigma=sigma,
                                  prefix_mm=rec['prefix_mm'],goal_mm=10)
    mark,_ = s['markers']._scissors(s['rgb_float'],z)
    if seed is None:
        saved = rec['marker']
        if bool(mark['valid'])!=bool(saved['valid']) or (mark['valid'] and any(
                not np.allclose(mark[k],saved[k],rtol=0,atol=1e-12) for k in ('tip','jaw_axis','opening_axis'))):
            raise ValueError('live marker replay disagrees: '+rec['frame'].name)
    result = dict(frame=rec['frame'].name,closed=rec['closed'],prefix_mm=rec['prefix_mm'],
                  seed=seed,chart_valid=chart is not None,marker_valid=bool(mark['valid']))
    if chart is None or not mark['valid']:
        if not mark['valid']:
            result['marker_reason'] = mark['reason']
        return result
    g,limit_row,_ = chart
    if rec['prefix_mm']>=6 and limit_row!=10:
        raise ValueError('non-row-10 late chart')
    d = s['R']@g.direction
    lim,tip = s['R']@g.limit+s['t'],s['R']@mark['tip']+s['t']
    observed = lim-tip
    reference = s['row10']-s['tcp']
    error = observed-reference
    return dict(result,limit_row=limit_row,vision_room_mm=project(observed,d),
                truth_room_mm=project(reference,d),optimism_mm=project(error,d),
                difference_on_true_axis_mm=project(error,s['axis']),
                absolute_axis_deg=axis_angle(d,s['axis']),terminal_chord_deg=axis_angle(d,s['end_axis']),
                fk_tcp_error_mm=s['fk_mm'])


def replay(samples,spec,noise,split):
    sigma = noise['depth_sigmas_m'][1]
    rows = []
    for x in samples:
        for seed in noise[split+'_seeds']:
            rng = np.random.default_rng(np.random.SeedSequence([seed,spec['noise_case_id'],x['rec']['tick']]))
            rows.append(measure(x,x['z']+rng.normal(0.,sigma,x['z'].shape),sigma,seed))
    return rows


def late_scores(rows):
    late = [r for r in rows if r['prefix_mm']>=6]
    return late,[r for r in late if 'optimism_mm' in r]


def summarize(rows):
    good = [r for r in rows if 'optimism_mm' in r]
    return dict(draws=len(rows),pairs=len(good),max_axis_deg=max((r['absolute_axis_deg'] for r in good),default=None),
                min_optimism_mm=min((r['optimism_mm'] for r in good),default=None),
                max_optimism_mm=max((r['optimism_mm'] for r in good),default=None))


def calibrate(batch,plan):
    frames,nominal,probe = registered_frames(batch,plan)
    cal = load(CAL)
    if not cal['qualified']:
        raise ValueError('unsealed FK sanity reference')
    model = CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in cal['calibration'].items()})
    samples = [load_sample(r,model,probe['final_q']) for r in frames]
    stamps = [r['stamp'] for r in samples]
    if stamps!=sorted(stamps) or stamps[-1]<=stamps[-2]:
        raise ValueError('nonmonotonic reference times')
    noise = load(NOISE)
    if noise['depth_sigmas_m']!=[0.,.0005] or set(noise['development_seeds']) & set(noise['check_seeds']):
        raise ValueError('invalid frozen seed split')
    clean = [measure(r,r['z'],0.) for r in samples]
    dev = replay(samples,plan,noise,'development')
    raw_path,dev_path = batch/'clean.json',batch/'development.json'
    write(raw_path,clean)
    write(dev_path,dev)
    _,raw_scored = late_scores(clean)
    dev_all,dev_scored = late_scores(dev)

    def coverage(all_rows,scored):
        return (bool(all_rows) and len(scored)>=math.ceil(plan['late_pair_min_fraction_each_split']*len(all_rows))
                and {(r['prefix_mm'],r['closed']) for r in scored}=={(7,True),(7,False),(9,True),(9,False)})

    clean_ok = all('optimism_mm' in r for r in clean) and bool(raw_scored)
    dev_ok = coverage(dev_all,dev_scored)
    fit = batch/'fit.json'
    c0=k=None
    if clean_ok and dev_ok:
        c0 = math.ceil(100*max(0.,max(r['optimism_mm'] for r in raw_scored)))/100
        k = math.ceil(10*max(0.,max(r['optimism_mm'] for r in dev_scored)-c0)/.5)/10
    write(fit,dict(c0_mm=c0,k=k,clean_sha256=digest(raw_path),development_sha256=digest(dev_path)))
    check = replay(samples,plan,noise,'check')
    check_path = batch/'check.json'
    write(check_path,check)
    check_all,check_scored = late_scores(check)
    frozen = load(fit)
    c0,k = frozen['c0_mm'],frozen['k']
    envelope = None if c0 is None else c0+k*.5
    max_fk = max(x['fk_mm'] for x in samples)
    axes = [r['absolute_axis_deg'] for r in clean+dev+check if 'absolute_axis_deg' in r]
    gates = dict(clean_coverage=clean_ok,development_coverage=dev_ok,
                 check_coverage=coverage(check_all,check_scored),
                 exact_tcp_fk=max_fk<=plan['fk_tcp_max_error_mm'],
                 absolute_axis=bool(axes) and max(axes)<plan['absolute_global_axis_deg_lt'],
                 smaller_clean_envelope=c0 is not None and c0<plan['legacy_margin_mm'],
                 held_out_envelope=envelope is not None and bool(check_scored)
                 and all(r['optimism_mm']<=envelope for r in check_scored))
    passed = all(gates.values())
    return dict(qualified=passed,authorizes_motion=False,gates=gates,frames=len(frames),
                nominal_closed_ready=nominal['closed'] and nominal['ready'],closures=nominal['closures'],
                terminal_structure_unchanged=nominal['constraints_sha256']==probe['constraints_sha256'],
                max_fk_mm=max_fk,max_absolute_axis_deg=max(axes,default=None),
                fit_candidate=dict(c0_mm=c0,k=k,at_0p5mm_mm=envelope),
                sensor_margin=dict(c0_m=c0*.001,k=k) if passed else None,
                clean=summarize(clean),late_clean=summarize([r for r in clean if r['prefix_mm']>=6]),
                late_development=summarize(dev_all),late_check=summarize(check_all),
                fit_sha256=digest(fit),check_sha256=digest(check_path))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    args = parser.parse_args()
    batch = args.batch
    if (batch/'decision.json').exists():
        raise FileExistsError('calibration already scored')
    data = dict(registration_sha256=digest(batch/'registration.json'),run_sha256=digest(batch/'run.json'))
    try:
        data.update(calibrate(batch,load(PLAN)))
    except ValueError as exc:
        data.update(qualified=False,authorizes_motion=False,sensor_margin=None,failure=str(exc))
    write(batch/'decision.json',data)
    print(data,flush=True)
