"""Seal measured optical predictions, then independently grade the disturbance matrix."""
import argparse
from pathlib import Path

import numpy as np

from replay_cell_v2 import ROOT, digest, load, material_state, write
from replay_m5_guided_reverse_v1 import events
from score_m5_guided_closed_loop_nominal_v1 import PURPOSES
from m5_guided_bank_corridor_v3 import measure

ID = 'm5_guided_benchmark_v2_disturbances_v1'
SPEC = ROOT/f'config/task/{ID}.json'


def registration(out):
    p = out/'registration.json'
    reg,plan = load(p),load(SPEC)
    if reg['policy']!=ID or any(digest(ROOT/f)!=h for f,h in (reg['sources']|plan['pins']).items()):
        raise ValueError('unregistered disturbance matrix')
    return dict(registration_sha256=digest(p),scorer_sha256=digest(Path(__file__)))


def predict(out,frames):
    reg = registration(out)
    cases = {}
    for case,(scenario,sigma,seed) in load(SPEC)['cases'].items():
        file = out/(case+'.json')
        if not file.is_file():
            cases[case] = None
            continue
        r = load(file)
        c = r['config']['perception']
        if (r['registration_sha256']!=reg['registration_sha256'] or r['case']!=case
                or (r['scenario'],c['depth_noise_m'],c['seed'])!=(scenario,sigma,seed)):
            raise ValueError('unregistered scenario or input')
        evidence = frames/(ID+'_'+case)
        optics = []
        for e in events(r,'guided_v2_optical'):
            f = evidence/e['file']
            if f.name!=e['file'] or digest(f)!=e['frame_sha256'] or e['sigma']!=sigma:
                raise ValueError('archived optical input changed')
            with np.load(f,allow_pickle=False) as a:
                rgb,z,raw = (a[k] for k in ('rgb','measured','depth'))
                if (rgb.shape!=(1920,2560,3) or rgb.dtype!=np.uint8 or z.shape!=rgb.shape[:2]
                        or raw.shape!=z.shape or abs(a['hfov_rad'].item()-np.deg2rad(20))>1e-6):
                    raise ValueError('unregistered certificate optics')
                expected = raw if not sigma else raw+np.random.default_rng(
                    np.random.SeedSequence([seed,e['tick'],513])).normal(0,sigma,z.shape)
                if not np.array_equal(expected,z):
                    raise ValueError('registered measured noise not present')
                answer = measure(rgb,z,far=a['far'].item(),sigma=sigma)
            optics.append(dict(tick=e['tick'],file=f.name,sha256=e['frame_sha256'],sigma=sigma,
                               verdict=answer['verdict'],prefix_mm=answer['prefix_mm'],reason=answer['reason'],
                               same_as_live=all(answer[k]==e[k] for k in ('verdict','prefix_mm'))))
        cases[case] = dict(report_sha256=digest(file),optics=optics)
    write(out/'predictions.json',dict(registration=reg,cases=cases))


def coverage(case,r):
    by_tick = {o['tick']:o for o in r['observations']}
    roles = r['config']['roles']
    if case=='depth_noise_0p5':
        return bool(by_tick and r['config']['perception']['depth_noise_m']==.0005),[]
    if case=='occlusion_transient':
        e = events(r,'occlude')
        return len(e)==1 and e[0]['tick']==6 and not e[0]['persistent'],e
    primary = roles['retract_primary']
    if case=='slip_regrasp':
        e = events(r,'slip')
        lost = by_tick.get(13,{}).get('grasped',{}).get(primary,0)
        prior = by_tick.get(12,{}).get('grasped',{}).get(primary,0)
        return bool(len(e)==1 and e[0]['tick']==13 and e[0]['arm']==primary
                    and 13 in by_tick and prior>=r['config']['grasp']['min_nodes'] and lost==0),e
    e = events(r,'relax')
    return bool({x['arm'] for x in e}=={primary,roles['retract_assist']} and len(e)==2
                and all(x['tick']==11 and x['accepted'] and x['relax_m']==.0005 for x in e)),e


def score(out):
    path = out/'predictions.json'
    p = load(path)
    if registration(out)!=p['registration'] or set(p['cases'])!=set(load(SPEC)['cases']):
        raise ValueError('unsealed scenario predictions')
    rows = []
    for case,pred in p['cases'].items():
        if pred is None:
            rows.append(dict(case=case,tested=False,passed=False))
            continue
        file = out/(case+'.json')
        if digest(file)!=pred['report_sha256']:
            raise ValueError('sealed scenario report changed')
        r = load(file)
        get = lambda n: events(r,n)
        obs = r['observations']
        by_tick = {o['tick']:o for o in obs}
        cuts,reacq,park,live = (get(n) for n in ('closure','post_closure_reacquired',
                                                 'guided_v2_withdrawal','guided_v2_optical'))
        stop = live[-1] if live and live[-1]['verdict']=='TARGET' else None
        held = [o['cut_length_m'] for o in obs if stop and o['tick']>=stop['tick']]
        state = None
        if r.get('structure_file'):
            name = case+'_structure.npz'
            f = out/name
            if name!=r['structure_file'] or digest(f)!=r['structure_sha256']:
                raise ValueError('unsealed structural sample')
            with np.load(f,allow_pickle=False) as z:
                a = material_state(z['edges'])
            state = {k:a[k] for k in ('prefix_rows','farthest_row_end')}
        exercised,injections = coverage(case,r)
        fault_free = bool(obs and not r['error'] and not r['cleanup_error'] and not get('perception_error')
                          and all(not o['faults'] and not o['torn'] and 0<=o['cut_length_m']<=.010+1e-12
                                  for o in obs))
        correct = bool(live and len(live)==len(pred['optics'])
                       and all(a['same_as_live'] and a['sha256']==e['frame_sha256'] and a['tick']==e['tick']
                               for a,e in zip(pred['optics'],live,strict=True))
                       and all(e['verdict']=='UNKNOWN' or (
                           type(e['prefix_mm']) is int and e['verdict']==('BELOW' if e['prefix_mm']<9 else 'TARGET')
                           and 1<=e['prefix_mm']<=10
                           and abs(by_tick[e['tick']]['cut_length_m']-.001*e['prefix_mm'])<1e-12)
                               for e in live))
        checks = dict(
            disturbance_exercised=exercised,
            fault_tear_overcut_error_free=fault_free,
            strict_truth=bool(r.get('strict_truth') and set(r['truth_purposes'])<=PURPOSES|{'event_injection'}),
            all_closures_current_and_reacquired=bool(stop and cuts and len(cuts)==len(reacq)
                and all(c['direct_front'] and c['scissors_valid'] and c['fix_age']==0
                    and c['tick']<a['tick']<(cuts[i+1]['tick'] if i+1<len(cuts) else stop['tick'])
                    for i,(c,a) in enumerate(zip(cuts,reacq,strict=True)))),
            live_measured_optics_correct=bool(correct and stop and stop['prefix_mm'] in (9,10)),
            independent_endpoint_no_growth=bool(state and stop and obs and obs[0]['cut_length_m']==0
                and state['prefix_rows']==stop['prefix_mm'] and state['farthest_row_end']<=10
                and len(held)>=2 and len(set(held))==1 and abs(held[0]-.001*state['prefix_rows'])<1e-12),
            one_closed_ready_withdrawal=bool(stop and len(park)==1 and park[0]['tick']==stop['tick']
                and park[0]['accepted'] and not park[0]['ready_before'] and park[0]['ready_after']
                and park[0]['closed_after'] and r.get('actual_closed') and r.get('actual_ready')
                and r.get('final_state')=='DONE' and obs and obs[-1]['tick']>stop['tick']),
            original_camera_budget=bool(r.get('attempts_camera',4)<=r['config']['budgets']['camera_adjust_attempts']==3),
        )
        rows.append(dict(case=case,tested=True,passed=all(checks.values()),checks=checks,injections=injections,
                         final_state=r.get('final_state'),reason=r['error'] or r.get('reason'),
                         structure=state,final_m=obs[-1]['cut_length_m'] if obs else None,
                         actual_closed=r.get('actual_closed'),actual_ready=r.get('actual_ready'),
                         closures=[e['tick'] for e in cuts],reacquired=[e['tick'] for e in reacq],
                         camera_attempts=r.get('attempts_camera'),optics=pred['optics'],withdrawal=park))
    result = dict(passed=all(a['passed'] for a in rows),predictions_sha256=digest(path),
                  tested=sum(a['tested'] for a in rows),qualified=sum(a['passed'] for a in rows),cases=rows)
    write(out/'decision.json',result)
    for row in rows:
        print({k:row.get(k) for k in ('case','passed','final_state','reason','final_m','actual_closed',
                                     'actual_ready','closures','camera_attempts','optics')},flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','score'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path)
    a = p.parse_args()
    if a.stage=='predict':
        if a.frames is None:
            p.error('--frames required')
        predict(a.out,a.frames)
    else:
        score(a.out)


if __name__=='__main__':
    main()
