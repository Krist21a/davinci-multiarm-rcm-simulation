"""Verify matched trajectories and inspect the original optical gate without changing it."""
import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from replay_cell_v2 import ROOT, digest, load, write
import cut_perception as cp
from tool_pose import _MARGIN

ID = 'm5_guided_disturbance_dry_diagnostic_v1'
TRACE = 'm5_guided_disturbance_fault_trace_v1'
REF = ROOT/'validation/task'/TRACE
SPEC = ROOT/f'config/task/{ID}.json'
EXTRA = {'fault_trace_dry_contrast','fault_trace_estimator_input'}


def events(r,name):
    return [e for e in r['events'] if e['event']==name]


def verify(folder,name,sha):
    f = folder/name
    if f.name!=name or digest(f)!=sha:
        raise ValueError(f'archive identity mismatch: {f}')
    return f


def verify_trajectory(out,frames,case,reg_sha):
    original = load(REF/(case+'.json'))
    r = load(out/(case+'.json'))
    if (r['registration_sha256']!=reg_sha or r['error'] or r['cleanup_error']
            or any(r[k]!=original[k] for k in (
                'config','observations','log','transitions','ticks','controller_sha256',
                'truth_purposes','actual_closed','actual_ready','final_state','reason',
                'attempts_camera','structure_sha256'))):
        raise ValueError(f'{case}: original trajectory changed')
    events_clean = [e for e in r['events'] if e['event'] not in EXTRA
                    and not (e['event']=='fault_trace_fk_refused' and not e['in_solve'])]
    if events_clean!=original['events']:
        raise ValueError(f'{case}: native call trace changed')
    for folder in (REF,out):
        verify(folder,r['structure_file'],r['structure_sha256'])
    for e in events_clean:
        if e['event'] in ('fault_trace_working_frame','guided_prefix_review','retained_coded_exposure'):
            for id_ in (TRACE,ID):
                verify(frames/(id_+'_'+case),e['file'],e.get('sha256') or e['frame_sha256'])
    return r


def dry_summary(r,plan,case,diagnosis):
    tick = plan['dry_ticks'][case]
    traces = events(r,'fault_trace_dry_contrast')
    orig = diagnosis['noise']['recorded_candidates'] if case=='depth_noise_0p5' else diagnosis['slip']['clear_solves']
    expected = [a for a in orig if a['tick']==tick]
    if len(traces)!=len(expected) or len(traces)!=(3 if case=='depth_noise_0p5' else 1):
        raise ValueError('unexpected dry contrast target count')
    rows = []
    for e,old in zip(traces,expected,strict=True):
        if (e['tick']!=tick or (e['target_m'],e['original_q'],e['native'])!=(
                old['target_m'],old['seed_q'],old['result'])
                or [v['name'] for v in e['variants']]!=plan['contrasts']):
            raise ValueError('dry contrast did not use the original target')
        q0 = np.array(e['original_q'])
        q_inward = q0.copy()
        q_inward[0] = np.sign(q0[0])*max(0,abs(q0[0])-1)
        if not all(np.array_equal(v['seed_q'],q) for v,q in zip(e['variants'],(q_inward,q0),strict=True)):
            raise ValueError('dry seed changed')
        for v in e['variants']:
            res = v['result']
            v['meets_original_residual_limits'] = bool(res and res['converged']
                and res['pos_err_m']<.0002 and res['axis_err_rad']<.02 and res['yaw_axis_err_rad']<.02
                and np.all(np.array(res['q'])>=np.array(e['lo'])+_MARGIN-1e-12)
                and np.all(np.array(res['q'])<=np.array(e['hi'])-_MARGIN+1e-12))
        rows.append(e)
    return dict(tick=tick,targets=rows,
                passing_by_variant={name:sum(v['meets_original_residual_limits']
                    for e in rows for v in e['variants'] if v['name']==name) for name in plan['contrasts']})


def check_window(s,band,occ,b_start,edges,sa_px,mask,depth,pz,far):
    ahead = band & (sa_px>=s) & (sa_px<s+cp.AHEAD_WINDOW_M+.0005)
    seen = ahead & np.isfinite(depth) & (depth<far*.999)
    n,visible,covered,sheet = (int(x.sum()) for x in (
        ahead,seen,seen&(depth<pz-cp.COVER_DEPTH_M),ahead&mask))
    b0 = max(b_start,np.digitize(s,edges)-1)
    b1 = min(len(occ),np.digitize(s+cp.AHEAD_WINDOW_M+.0005,edges))
    gaps = (b0+np.flatnonzero(occ[b0:b1])).tolist()
    checks = dict(nonempty=n>0,seen_at_least_half=2*visible>=n,covered_at_most_half=2*covered<=visible,
                  enough_sheet=sheet>=max(cp.MIN_AHEAD_PX,visible//2+1),no_occupied_gap_bin=not gaps)
    return dict(start_m=float(s),window_pixels=n,seen_pixels=visible,foreground_pixels=covered,
                sheet_pixels=sheet,required_sheet_pixels=max(cp.MIN_AHEAD_PX,visible//2+1),
                occupied_gap_bins=gaps,checks=checks)


def optical_frame(saved,working,observed):
    with np.load(saved,allow_pickle=False) as a,np.load(working,allow_pickle=False) as w:
        if (not np.array_equal(a['rgb'],w['rgb']) or not np.array_equal(a['depth'],w['native_depth'])
                or a['rgb'].shape!=(960,1280,3)):
            raise ValueError('saved estimator inputs do not match the original native RGB-D')
        state = json.loads(a['state'].item())
        est = object.__new__(cp.CutPerception)
        if 'estimate' in state:
            raise ValueError('unexpected estimator snapshot')
        est.__dict__.update(state)
        for k in a['array_keys'].tolist():
            setattr(est,k,np.asarray(state[k],dtype=float))
        original = cp._provable_ahead
        windows,details = [],{}
        code = cp.CutPerception._cut.__code__

        def observe_window(*args):
            result = original(*args)
            row = check_window(*args)
            if result!=all(row['checks'].values()):
                raise ValueError('ahead-check trace disagrees with original')
            frame = sys._getframe(1)
            row.update(call_site=frame.f_code.co_name,passed=bool(result))
            windows.append(row)
            return result

        def local(frame,event,value):
            if event=='return':
                s = frame.f_locals
                res = s.get('res') or {}
                details.update(near_pixel_count=s.get('evidence'),minimum_near_pixels=cp.MIN_NEAR_PX,
                               slit_run_mm=float(res['run_len']*1000) if 'run_len' in res else None,
                               slit_min_mm=cp.SLIT_MIN_LEN_M*1000,
                               candidates=[[kind,float(pos),float(lat)] for kind,pos,lat in s.get('cands',[])],
                               rejected_distal_offsets_m=[float(x) for x in s.get('distal',[])])
            return local

        def enter(frame,event,value):
            if event=='call' and frame.f_code is code:
                frame.f_trace_lines = False
                return local
            return None

        prior_trace = sys.gettrace()
        try:
            cp._provable_ahead = observe_window
            sys.settrace(enter)
            r = est.estimate(a['rgb'],a['depth'],a['R'],a['t'],post_grasp=a['post_grasp'].item())
        finally:
            sys.settrace(prior_trace)
            cp._provable_ahead = original
    cut,scis = r['cut'],r['scissors']
    chart = cut.get('material_chart') or {}
    expected = dict(front_valid=bool(cut.get('front_valid')),reason=cut.get('reason'),mode=cut.get('mode'),
                    slit_px=cut.get('slit_px'),progress_m=cut.get('progress_m'),chart_valid=bool(chart.get('valid')),
                    chart_reason=chart.get('reason'),scissors_valid=bool(scis.get('valid')),
                    scissors_reason=scis.get('reason'))
    if any(v!=observed[k] for k,v in expected.items()):
        raise ValueError('offline estimator does not reproduce original online decisions')
    return dict(tick=observed['tick'],original_estimate=expected,**details,
                windows=windows,matched_estimator_input_sha256=digest(saved))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True)
    a = p.parse_args()
    plan,reg = load(SPEC),a.out/'registration.json'
    lock = load(reg)
    if (lock['policy']!=ID or any(digest(ROOT/k)!=v for k,v in lock['sources'].items())
            or digest(REF/'diagnosis.json')!=plan['trace_diagnosis_sha256']
            or digest(REF/'registration.json')!=plan['trace_registration_sha256']):
        raise ValueError('unregistered diagnostic')
    d = load(REF/'diagnosis.json')
    reports = {c:verify_trajectory(a.out,a.frames,c,digest(reg)) for c in plan['dry_ticks']}
    dry = {c:dry_summary(r,plan,c,d) for c,r in reports.items()}
    slip = reports['slip_regrasp']
    frame_events = events(slip,'fault_trace_estimator_input')
    if [e['tick'] for e in frame_events]!=plan['optical_ticks']:
        raise ValueError('unexpected optical snapshot schedule')
    estimates = {e['tick']:e for e in events(slip,'fault_trace_estimate')}
    wide = {e['tick']:e for e in events(slip,'fault_trace_working_frame')}
    optics = [optical_frame(verify(a.frames/(ID+'_slip_regrasp'),e['file'],e['sha256']),
                verify(a.frames/(TRACE+'_slip_regrasp'),wide[e['tick']]['file'],wide[e['tick']]['sha256']),
                estimates[e['tick']]) for e in frame_events]
    result = dict(registration_sha256=digest(reg),scorer_sha256=digest(Path(__file__)),matched_trajectories=True,
                  dry_ik=dry,optical_windows=optics,authorizes_motion=False,changes_prior_acceptance=False)
    write(a.out/'diagnosis.json',result)
    print({c:v['passing_by_variant'] for c,v in dry.items()})
    for r in optics:
        print(r)


if __name__=='__main__':
    main()
