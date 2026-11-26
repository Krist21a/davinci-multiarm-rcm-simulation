"""Independently replay corridor v5, then score the two registered development runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_m5_v3_optics_attribution_v1 import ROOT, digest, load, verified
from m5_v3_corridor_v5 import measure
from m5_v3_perception_v4 import intact_preflight
from score_m5_v3_outer_positive_v1 import verdict


PLAN = ROOT/'config/task/m5_v3_positive_outer_v3.json'


def examine(folder, case, plan):
    goal = plan['cases'][case]
    run, reg = folder/'run.json', folder/'registration.json'
    raw, seal = load(run), load(reg)
    if (digest(reg)!=raw['registration_sha256'] or raw['run_id']!=seal['run_id']
            or seal['plan']!=plan or seal['case']!=case or raw['case']!=case
            or seal['goal_mm']!=goal or raw['goal_mm']!=goal):
        raise ValueError('mismatched live registration')
    for path,pin in seal['sources'].items():
        verified(ROOT/path,pin)
    receipts, initial, previous = [], False, 0
    ticks = []
    for m in sorted(raw['captures'].values(),key=lambda a:a['tick']):
        frame = verified(folder/m['file'],m['sha256'])
        ticks.append(m['tick'])
        if m['fov_deg']!=20:
            continue
        with np.load(frame,allow_pickle=False) as f:
            if (not m['closed'] or f['depth'].shape!=(1920,2560)
                    or not np.allclose(f['q'],raw['entrance_q'],rtol=0,atol=1e-6)
                    or not np.allclose(f['q'],m['q'],rtol=0,atol=1e-6)):
                raise ValueError('not the recorded closed entrance-clear view')
            args = f['rgb'],f['depth']
            far = f['far'].item()
            if m['closures']==0:
                r = intact_preflight(*args,far=far)
                if r!=m['uncut'] or initial or receipts:
                    raise ValueError('unexpected uncut check')
                initial = r['passed']
                continue
            r = measure(*args,far=far,sigma=0.)
        if r!=m['optical']:
            raise ValueError('independent optical replay differed')
        v,n = r['verdict'],r['prefix_mm']
        growing = type(n) is int and previous<n<=goal and v==('BELOW' if n<9 else 'TARGET')
        if growing:
            previous = n
        receipts.append(dict(tick=m['tick'],closure=m['closures'],frame=frame.name,
                             verdict=v,prefix_mm=n,positive=growing,sha256=m['sha256'],
                             reason=r['reason'],
                             rows=[(s['row'],s['status']) for s in r['rows']],
                             recovered_rows=[s['row'] for s in r['rows']
                                             if s.get('witness_method')=='adjacent_strips']))
    if ticks!=list(range(1,len(ticks)+1)):
        raise ValueError('missing or duplicate exposure')
    optical = dict(case=case,goal_mm=goal,initial_passed=initial,receipts=receipts,
                   run_sha256=digest(run),registration_sha256=digest(reg))
    return optical,raw


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    a = p.parse_args()
    plan = load(PLAN)
    data = {k:examine(a.batch/k,k,plan) for k in plan['cases']}
    if len({r['run_id'] for _,r in data.values()})!=2:
        raise ValueError('expected two fresh registered runs')
    cases = {}
    for k,(o,r) in data.items():
        summary = verdict(o,r)
        sequence = [(x['verdict'],x['prefix_mm']) for x in summary['receipts']]
        required = [('BELOW',3),('BELOW',5),('BELOW',7),('TARGET',9)]
        if k=='exact_10mm':
            required.append(('TARGET',10))
        summary['required_sequence_passed'] = sequence==required
        summary['passed'] &= summary['required_sequence_passed']
        cases[k] = summary
    result = dict(plan_sha256=digest(PLAN),scorer_sha256=digest(Path(__file__)),cases=cases,
                  passed=all(v['passed'] for v in cases.values()),authorizes_disturbances=False)
    with (a.batch/'decision.json').open('x',encoding='utf-8',newline='\n') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n')
    print({k:(v['state'],v['passed'],v['reason']) for k,v in cases.items()})
