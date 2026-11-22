"""Independently read the five full-length development runs; never count a recovery park as DONE."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_identity(r, case, recipe, policy, registered_hash, run_ids):
    scenario, noise, seed = recipe
    perception = r['config']['perception']
    if (r.get('registration_sha256') != registered_hash or r.get('policy') != policy
            or r.get('case') != case or r.get('scenario') != scenario
            or perception.get('depth_noise_m') != noise or perception.get('seed') != seed
            or r['config']['cut']['control_source'] != 'vision'):
        raise ValueError(f'{case}: run identity does not match registration')
    try:
        uid = UUID(r['run_id']).hex
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{case}: invalid run ID') from exc
    if uid in run_ids:
        raise ValueError(f'{case}: duplicate run ID')
    run_ids.add(uid)


def summarize(folder):
    reg=folder/'registration.json'
    registration=json.loads(reg.read_bytes())
    if registration.get('schema')!=2 or registration.get('policy') not in ('same_bite','active_look'):
        raise ValueError('unbound historical registration; retain archived scores, do not rescore')
    policy=registration['policy']
    plan=registration['protocol']
    registered_hash=digest(reg)
    run_ids=set()
    cases={}
    for case in plan['cases']:
        file=folder/(case+'.json')
        r=json.loads(file.read_bytes())
        check_identity(r,case,plan['cases'][case],policy,registered_hash,run_ids)
        raw=r['observations']
        if len({s['tick'] for s in raw})!=len(raw):
            raise ValueError('repeated observation tick')
        src={s['tick']:s for s in raw}
        bites=[e for e in r['events'] if e['event']=='closure']
        seen=[e for e in r['events'] if e['event']=='post_closure_reacquired']
        fresh={s['tick']:s['estimate'] for s in r['perception']}
        issues=[]
        seq=[]
        for i,b in enumerate(bites):
            t=b['tick']
            n=bites[i+1]['tick'] if i+1<len(bites) else max(src)+1
            reacq=[e['tick'] for e in seen if t<e['tick']<n]
            if len(reacq)!=1:
                issues.append(f'bite {i+1} has {len(reacq)} fresh reacquisitions')
            for v in [t,*reacq]:
                est=fresh[v]
                if not (est['scissors'].get('valid') and est['cut'].get('front_valid') and
                        est['cut'].get('mode') in (('pre_cut','slit','bank_slit') if v==t and i==0
                                                   else ('slit','bank_slit'))):
                    issues.append(f'no direct tool + front at {v}')
                if v!=t and (not (est['cut'].get('material_chart') or {}).get('valid')
                             or est['cut'].get('progress_m') is None):
                    issues.append(f'fresh slit not mapped at {v}')
            seq.append(dict(closure=t,chart=b['material_source'],fix_age=b['fix_age'],reacquired=reacq,
                            before_mm=src[t]['cut_length_m']*1000,after_mm=src[t+1]['cut_length_m']*1000))
        if len(seen)!=sum(len(s['reacquired']) for s in seq):
            issues.append('unmatched reacquisition')
        # A completed endpoint originates from a same-frame slit and not just
        # a finite accepted structural length or a TCP already at ready.
        exits=[e for e in r['log'] if e.get('to')=='DONE']
        if len(exits)!=1 or not bites:
            issues.append('no full visual completion')
        else:
            direct=[q for q in r['perception'] if q['tick']>bites[-1]['tick'] and
                    q['tick']<=exits[0]['world_tick'] and q['estimate']['cut'].get('mode') in ('slit','bank_slit')
                    and q['estimate']['scissors'].get('valid') and q['estimate']['cut'].get('front_valid')
                    and (q['estimate']['cut'].get('progress_m') or 0)>=r['config']['cut']['length_m']]
            if not direct:
                issues.append('no optical endpoint after last closure')
        if r.get('error') or r.get('cleanup_error') or any(e['event']=='perception_error' for e in r['events']):
            issues.append('run error')
        if not r['strict_truth'] or r['truth_purposes'].get('untagged'):
            issues.append('truth isolation')
        qs=np.asarray(r['ready_q']) if r.get('ready_q') is not None else np.zeros(0)
        q=np.asarray(r['actual_q']) if r.get('actual_q') is not None else np.zeros(0)
        same_q=bool(q.shape==qs.shape==(6,) and np.allclose(q,qs,atol=1e-6,rtol=0))
        if not (r['actual_ready'] and r['actual_closed'] and same_q):
            issues.append('not closed at measured ready pose and command')
        tf=r['final']['cut_length_m']
        peak=max(s['cut_length_m'] for s in raw)
        if raw[0]['cut_length_m']!=0 or not .009-1e-12<=tf<=peak<=.010+1e-12:
            issues.append('structural length')
        if any(s['faults'] or s['torn'] for s in raw):
            issues.append('tear or fault')
        post=[s['cut_length_m'] for s in raw if seen and s['tick']>=seen[-1]['tick']]
        if not post or len(set(post))!=1:
            issues.append('no verified constant post-reacquisition length to park')
        cases[case]=dict(report_sha256=digest(file),state=r['final_state'],reason=r['reason'],final_mm=tf*1000,
                         peak_mm=peak*1000,tear=max(s['torn'] for s in raw),faults=sorted({v for s in raw for v in s['faults']}),
                         actual_ready=r['actual_ready'],actual_closed=r['actual_closed'],same_q=same_q,
                         strict=r['strict_truth'],bites=seq,issues=issues,passed=r['accepted'] and not issues)
    return dict(policy=policy,registration_sha256=registered_hash,scorer_sha256=digest(Path(__file__)),cases=cases,
                passed=all(c['passed'] for c in cases.values()),signed_gates_modified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder',required=True,type=Path)
    folder=ap.parse_args().folder
    result=summarize(folder)
    with (folder/'five_scenarios.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    for k,c in result['cases'].items():
        print(k,c['state'],round(c['final_mm'],3),len(c['bites']),c['actual_ready'],c['actual_closed'],c['passed'])
