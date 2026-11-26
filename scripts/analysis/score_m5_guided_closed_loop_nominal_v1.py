"""Seal a nominal's endpoint pixels, then qualify its trace and structure."""
import argparse
from pathlib import Path

import numpy as np

from replay_cell_v2 import ROOT, digest, load, material_state, write
from replay_m5_guided_reverse_v1 import events
from m5_guided_bank_corridor_v3 import measure

SPEC = ROOT/'config/task/m5_guided_closed_loop_nominal_v1.json'
PURPOSES = {'grasp_signal_and_safety_monitor','tension_until_m6','safety_monitor',
            'scoring','retractor_planning'}


def registration(folder):
    r = folder/'registration.json'
    plan, reg = load(SPEC),load(r)
    if reg['policy']!=plan['id'] or any(digest(ROOT/k)!=v for k,v in (plan['pins']|reg['sources']).items()):
        raise ValueError('nominal registration changed')
    return dict(policy_sha256=digest(SPEC),registration_sha256=digest(r),
                scorer_sha256=digest(Path(__file__)),run_sha256=digest(folder/'nominal.json'))


def predict(folder,frames):
    reg = registration(folder)
    raw = load(folder/'nominal.json')
    if raw['registration_sha256']!=reg['registration_sha256']:
        raise ValueError('run not from the registered nominal')
    readings, poses = [],[]
    for v in events(raw,'nominal_live_optical_review'):
        file = frames/v['file']
        if file.name!=v['file'] or digest(file)!=v['frame_sha256']:
            raise ValueError('current-frame archive changed')
        with np.load(file,allow_pickle=False) as f:
            rgb,z = f['rgb'],f['depth']
            if (rgb.shape!=(1920,2560,3) or rgb.dtype!=np.uint8 or z.shape!=rgb.shape[:2]
                    or not np.array_equal(f['measured'],z)
                    or abs(f['hfov_rad'].item()-np.deg2rad(20))>1e-6):
                raise ValueError('unexpected nominal review optics')
            check = measure(rgb,z,far=f['far'].item(),sigma=0.)
            poses.append([f[k] for k in ('cam_R','cam_t','hfov_rad','focal','far')])
        readings.append(dict(tick=v['tick'],phase=v['phase'],sha256=v['frame_sha256'],
                             same_as_live=all(check[k]==v[k] for k in ('verdict','prefix_mm')),
                             optical=check))
    unchanged = len(poses)==2 and all(np.array_equal(a,b) for a,b in zip(*poses,strict=True))
    write(folder/'predictions.json',dict(registration=reg,same_camera=unchanged,readings=readings))


def score(folder):
    sealed = folder/'predictions.json'
    raw, pred = load(folder/'nominal.json'),load(sealed)
    if pred['registration']!=registration(folder):
        raise ValueError('unsealed run or predictions')
    plan = load(SPEC)
    low,high = plan['target_m']
    obs = raw['observations']
    last = obs[-1] if obs else {}
    peak = max((v['cut_length_m'] for v in obs),default=None)
    cuts,anchors,reverse,reviews,reacq,park,reuse,moves = (
        events(raw,n) for n in ('closure','reverse_waypoint_recorded','closed_reverse_segment',
                               'nominal_live_optical_review','post_closure_reacquired',
                               'nominal_guarded_withdrawal','nominal_terminal_ecm_reuse',
                               'diagnostic_final_advance_request'))
    structural = None
    if raw.get('structure_file'):
        structure = folder/'terminal_structure.npz'
        if raw['structure_file']!=structure.name or digest(structure)!=raw['structure_sha256']:
            raise ValueError('unsealed terminal structure')
        with np.load(structure,allow_pickle=False) as f:
            s = material_state(f['edges'])
        structural = {k:s[k] for k in ('prefix_rows','farthest_row_end')}
    stop = next((v for v in reviews if v['phase']=='withdraw' and v['optically_permitted']),None)
    held = [v['cut_length_m'] for v in obs if stop is not None and v['tick']>=stop['tick']]
    marks = {v['tick']:v for v in obs}
    below = reviews[0] if reviews else None
    phases = [v['phase'] for v in reviews]
    issue = [v for v in events(raw,'advance_reserve') if moves and v['tick']==moves[0]['tick']]
    checks = {
        'runtime_ok': not raw['error'] and not raw['cleanup_error'],
        'terminal_done_closed_ready': raw.get('final_state')=='DONE'
                                      and raw.get('actual_closed') is True and raw.get('actual_ready') is True,
        'strict_truth_purposes': bool(raw.get('strict_truth') and raw.get('truth_purposes')
                                      and set(raw['truth_purposes'])<=PURPOSES
                                      and raw['truth_purposes']['scoring'].get('constraints')==1),
        'bounded_structure': bool(structural and obs and obs[0]['cut_length_m']==0.
                                  and structural['farthest_row_end']<=10 and peak<=high+1e-12
                                  and low-1e-12<=last['cut_length_m']<=high+1e-12
                                  and abs(last['cut_length_m']-structural['prefix_rows']*.001)<=1e-12),
        'safety_clean': bool(obs and all(not v['torn'] and not v['faults'] for v in obs)
                             and not events(raw,'perception_error')),
        'guarded_closures_and_reacquisitions': bool(
            len(cuts)==len(reacq)==3 and all(v['direct_front'] and v['scissors_valid'] and v['fix_age']==0
            and v['tick']<reacq[i]['tick']<(cuts[i+1]['tick'] if i<2 else (stop or {}).get('tick',-1))
            for i,v in enumerate(cuts))),
        'fresh_below_then_fresh_target': bool(
            phases==['continue','withdraw'] and stop and len(cuts)==3 and structural
            and below['optically_permitted'] and below['verdict']=='BELOW' and below['prefix_mm']==7
            and cuts[1]['tick']<below['tick']<cuts[-1]['tick']<stop['tick']
            and abs(marks[below['tick']]['cut_length_m']-.007)<=1e-12
            and stop['verdict']=='TARGET' and stop['prefix_mm']==structural['prefix_rows']
            and stop['prefix_mm'] in (9,10) and held and len(set(held))==1
            and len(held)>=2 and last['tick']>stop['tick']),
        'same_frame_replay_and_fixed_ecm': bool(
            pred['same_camera'] and len(pred['readings'])==len(reviews)==2 and
            all(a['same_as_live'] and a['tick']==v['tick'] and a['sha256']==v['frame_sha256']
                for a,v in zip(pred['readings'],reviews,strict=True))
            and len(reuse)==1 and cuts and reuse[0]['tick']>cuts[-1]['tick']
            and reuse[0]['attempts']==raw.get('attempts_camera')<=raw['config']['budgets']['camera_adjust_attempts']),
        'single_third_advance_with_native_reserve': bool(
            len(moves)==len(issue)==1 and moves[0]['fixed_cap_m']==plan['third_advance_cap_m']
            and issue[0]['reserve_m']==.00015 and .0002<=issue[0]['commanded_m']<=moves[0]['fixed_cap_m']
            and issue[0]['commanded_m']<=issue[0]['native_budget_m']-.00015+1e-12),
        'guarded_closed_reverse': bool(
            1<=len(reverse)<=4 and len(anchors)==len(reverse)+1
            and all(a['accepted'] and a['commanded'] and np.allclose(a['q'],b['q'],atol=1e-6,rtol=0)
                    for a,b in zip(reverse,reversed(anchors[:-1]),strict=True))),
        'one_current_permit_withdrawal': bool(
            stop and len(park)==1 and park[0]['tick']==stop['tick'] and park[0]['attempt']==1
            and park[0]['accepted'] and not park[0]['ready_before'] and park[0]['ready_after']
            and park[0]['closed_after']),
    }
    result = dict(predictions_sha256=digest(sealed),passed=all(checks.values()),checks=checks,
                  final_state=raw.get('final_state'),reason=raw['error'] or raw.get('reason'),
                  terminal_structure=structural,peak_m=peak,final_m=last.get('cut_length_m'),
                  closures=[v['tick'] for v in cuts],reacquired=[v['tick'] for v in reacq],
                  actual_closed=raw.get('actual_closed'),actual_ready=raw.get('actual_ready'),
                  optical=reviews,withdrawal=park,perturbations_tested=False)
    write(folder/'decision.json',result)
    print(result,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','score'))
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--frames',type=Path)
    a = p.parse_args()
    if a.stage=='predict':
        if a.frames is None:
            p.error('--frames required')
        predict(a.folder,a.frames)
    else:
        score(a.folder)


if __name__=='__main__':
    main()
