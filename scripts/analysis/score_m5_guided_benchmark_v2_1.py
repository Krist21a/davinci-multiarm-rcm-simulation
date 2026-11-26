"""Seal the v2.1 optical replay before joining independent structural scores."""
import argparse
from pathlib import Path

import numpy as np

from replay_cell_v2 import ROOT, digest, load, material_state, write
from score_m5_guided_benchmark_v2_disturbances_v1 import coverage, events, measure, PURPOSES

ID = 'm5_guided_benchmark_v2_1'
SPEC = ROOT/f'config/task/{ID}.json'
POSITIVES = {'nominal':9,'exact_10mm':10}


def registration(out):
    p = out/'registration.json'
    r = load(p)
    if r['policy']!=ID or any(digest(ROOT/f)!=h for f,h in r['sources'].items()):
        raise ValueError('unregistered v2.1 sources')
    return dict(registration_sha256=digest(p),scorer_sha256=digest(Path(__file__)))


def source(out,case,reg):
    f = out/(case+'.json')
    r = load(f)
    k = 'nominal' if case=='exact_10mm' else case
    expected = load(ROOT/'config/task/m5_live_same_bite_v1.json')['cases'][k]
    perc,cut = r['config']['perception'],r['config']['cut']
    target = .010 if case=='exact_10mm' else .009
    if (r['case']!=case or r['registration_sha256']!=reg['registration_sha256']
            or [r['scenario'],perc['depth_noise_m'],perc['seed']]!=expected
            or cut['control_source']!='vision' or cut['length_m']!=target
            or abs(cut['length_m']+cut['max_overcut_m']-.01)>1e-12):
        raise ValueError('unregistered case inputs')
    return r,f


def predict(out,frames):
    reg = registration(out)
    cases = {}
    for case in load(SPEC)['case_order']:
        if not (out/(case+'.json')).is_file():
            cases[case] = None
            continue
        r,f = source(out,case,reg)
        opt = []
        sigma,seed = (r['config']['perception'][k] for k in ('depth_noise_m','seed'))
        for e in events(r,'guided_v21_optical'):
            p = frames/case/e['file']
            if p.name!=e['file'] or digest(p)!=e['frame_sha256'] or sigma!=e['sigma']:
                raise ValueError('unsealed measured exposure')
            with np.load(p,allow_pickle=False) as z:
                rgb,raw,measured = z['rgb'],z['depth'],z['measured']
                if (rgb.shape!=(1920,2560,3) or rgb.dtype!=np.uint8
                        or raw.shape!=rgb.shape[:2] or measured.shape!=raw.shape
                        or abs(float(z['hfov_rad'])-np.deg2rad(20))>1e-6):
                    raise ValueError('unregistered narrow camera')
                if sigma:
                    raw = raw+np.random.default_rng(np.random.SeedSequence([seed,e['tick'],513])).normal(
                        0,sigma,raw.shape)
                if not np.array_equal(raw,measured):
                    raise ValueError('measured depth does not match the registered noise')
                ans = measure(rgb,measured,far=z['far'].item(),sigma=sigma)
            opt.append(dict(tick=e['tick'],verdict=ans['verdict'],prefix_mm=ans['prefix_mm'],reason=ans['reason'],
                            same_as_live=all(e[k]==ans[k] for k in ('verdict','prefix_mm')),
                            frame_sha256=e['frame_sha256']))
        cases[case] = dict(report_sha256=digest(f),optics=opt)
    write(out/'predictions.json',dict(registration=reg,cases=cases))


def load_structure(out,r,label):
    m = r['structures'].get(label)
    if m is None:
        return None,None
    p = out/m['file']
    if p.name!=m['file'] or digest(p)!=m['sha256']:
        raise ValueError('unsealed structural sample')
    with np.load(p,allow_pickle=False) as z:
        edges = z['edges']
    state = material_state(edges)
    return {k:state[k] for k in ('prefix_rows','farthest_row_end')},edges


def grade(out,case,r,pred):
    def get(n):
        return events(r,n)

    obs = r['observations']
    by_tick = {o['tick']:o for o in obs}
    cuts,live,park = (get(k) for k in ('closure','guided_v21_optical','guided_v2_withdrawal'))
    stop = live[-1] if live and live[-1]['action']=='withdraw' else None
    optics = pred['optics']
    state,edges = load_structure(out,r,'final')
    pre_state,pre_edges = load_structure(out,r,'pre_withdraw')
    exercised,injections = (True,[]) if case in POSITIVES else coverage(case,r)
    real = POSITIVES.get(case)
    allowed = (9,10) if real is None else (real,)
    paired = {}
    motions = get('guided_v21_set_tool_pose')
    for a in get('guided_v21_active_set'):
        if a['converged']:
            for m in motions:
                if m['tick']==a['tick'] and np.array_equal(m['q'],a['q']):
                    paired[m['tick']] = m
    tag_ik = list(paired.values())
    reacq = get('post_closure_reacquired')
    posts = [c['tick'] for c in cuts]
    growth = (bool(live and cuts and stop) and live[0]['previous_prefix_mm'] is None
              and all(e['previous_prefix_mm']==prior['prefix_mm'] and e['prefix_mm']>prior['prefix_mm']
                      for prior,e in zip(live,live[1:]))
              and all(e['action']=='resume' for e in live[:-1])
              and all(any(t<e['tick']<(live[i+1]['tick'] if i+1<len(live) else obs[-1]['tick'])
                          and (i==0 or t>live[i-1]['tick']) for t in posts)
                      for i,e in enumerate(live)))
    closure_evidence = (bool(cuts and stop) and all(
        c['scissors_valid'] and c['direct_front'] and c['fix_age']==0
        and (i==len(cuts)-1 and stop['tick']>c['tick']
             or i<len(cuts)-1 and any(c['tick']<s['tick']<cuts[i+1]['tick'] for s in reacq))
        for i,c in enumerate(cuts)))
    same_lengths = all(e['verdict']=='UNKNOWN' or (
        type(e['prefix_mm']) is int and e['tick'] in by_tick
        and abs(by_tick[e['tick']]['cut_length_m']-.001*e['prefix_mm'])<1e-12) for e in live)
    caps = get('guided_visual_cap')
    checks = dict(
        disturbance_exercised=exercised,
        fault_tear_overcut_error_free=bool(obs and not r['error'] and not r['cleanup_error']
            and not r['scoring_errors'] and not get('perception_error')
            and all(not o['faults'] and not o['torn'] and 0<=o['cut_length_m']<=.010+1e-12 for o in obs)),
        strict_truth=bool(r.get('strict_truth') and set(r['truth_purposes'])<=PURPOSES|{'event_injection'}),
        caps_and_camera_budget=bool(r.get('attempts_camera',4)<=r['config']['budgets']['camera_adjust_attempts']==3
            and all(0<e['limited_m']<=e['cap_m']<=.002 for e in caps)),
        closure_evidence=bool(closure_evidence),
        increasing_narrow_prefix=bool(growth),
        sealed_live_measured_optics=bool(stop and len(optics)==len(live) and same_lengths
                                        and all(e['same_as_live'] for e in optics)),
        exact_endpoint_and_no_withdrawal_growth=bool(stop and state and pre_state==state
            and obs[0]['cut_length_m']==0 and state['prefix_rows']==stop['prefix_mm']
            and state['prefix_rows'] in allowed and state['farthest_row_end']<=10
            and r['structures']['pre_withdraw']['tick']==stop['tick'] and np.array_equal(edges,pre_edges)),
        one_verified_closed_withdrawal=bool(stop and len(park)==1 and park[0]['tick']==stop['tick']
            and park[0]['accepted'] and not park[0]['ready_before'] and park[0]['ready_after']
            and park[0]['closed_after'] and r.get('actual_closed') and r.get('actual_ready')
            and r.get('final_state')=='DONE' and obs[-1]['tick']>stop['tick']),
    )
    return dict(case=case,tested=True,passed=all(checks.values()),checks=checks,structure=state,injections=injections,
                state=r.get('final_state'),reason=r['error'] or r.get('reason'),closed=r.get('actual_closed'),
                ready=r.get('actual_ready'),closures=posts,camera_attempts=r.get('attempts_camera'),
                reviews=live,withdrawal=park,active_set_guarded_commands=tag_ik)


def score(out):
    path = out/'predictions.json'
    p,reg = load(path),registration(out)
    if p['registration']!=reg or tuple(p['cases'])!=tuple(load(SPEC)['case_order']):
        raise ValueError('predictions use another registration')
    cases = []
    for case,pred in p['cases'].items():
        if pred is None:
            cases.append(dict(case=case,tested=False,passed=False))
            continue
        r,f = source(out,case,reg)
        if digest(f)!=pred['report_sha256']:
            raise ValueError('run report changed after prediction')
        cases.append(grade(out,case,r,pred))
    write(out/'decision.json',dict(predictions_sha256=digest(path),registration=reg,cases=cases,
        passed=all(r['passed'] for r in cases),positive_cases_passed=all(r['passed'] for r in cases[:2]),
        disturbance_passes=sum(r['passed'] for r in cases[2:]),replaces_historical_acceptance=False))
    for case in cases:
        print({k:case.get(k) for k in ('case','tested','passed','state','reason','structure','closed','ready',
                                      'closures','camera_attempts')},flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','score'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path)
    a = p.parse_args()
    if a.stage=='predict':
        if a.frames is None:
            p.error('--frames is required for predict')
        predict(a.out,a.frames)
    else:
        score(a.out)


if __name__=='__main__':
    main()
