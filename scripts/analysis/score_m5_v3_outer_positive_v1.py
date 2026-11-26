"""Blind-to-structure optical reread followed by independent final-state scoring."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_m5_v3_optics_attribution_v1 import ROOT, digest, load, verified
from m5_v3_perception_v3 import review


PLAN = ROOT/'config/task/m5_v3_positive_outer_v1.json'


def examine(path, case, plan):
    goal = plan['cases'][case]
    run_file, seal_file = path/'run.json', path/'registration.json'
    r, seal = load(run_file), load(seal_file)
    if (digest(seal_file)!=r['registration_sha256'] or seal['case']!=case or r['case']!=case
            or seal['plan']!=plan or r['run_id']!=seal['run_id'] or r['goal_mm']!=goal):
        raise ValueError('mismatched run registration')
    for name,pin in seal['sources'].items():
        verified(ROOT/name,pin)
    receipts = []
    previous = 0
    for name, row in sorted(r['captures'].items(), key=lambda item:item[1]['tick']):
        frame = verified(path/row['file'],row['sha256'])
        if not row.get('closures'):
            continue
        with np.load(frame,allow_pickle=False) as f:
            if (not row['closed'] or row['fov_deg']!=20 or f['depth'].shape!=(1920,2560)
                    or not np.allclose(f['q'],r['entrance_q'],rtol=0,atol=1e-6)
                    or not np.allclose(f['q'],row['q'],rtol=0,atol=1e-6)):
                raise ValueError('certificate was not taken at the registered closed entrance')
            opt = review(f['rgb'],f['depth'],far=f['far'].item(),sigma=0.)
        if opt!=row['optical']:
            raise ValueError('optical receipt could not be replayed')
        v,n = opt['verdict'],opt['prefix_mm']
        growing = type(n) is int and previous<n<=goal
        valid = growing and v==('TARGET' if n>=9 else 'BELOW')
        if valid:
            previous = n
        receipts.append(dict(tick=row['tick'],closure=row['closures'],frame=name,
                             verdict=v,prefix_mm=n,positive=bool(valid),sha256=row['sha256']))
    return dict(case=case, goal_mm=goal, initial_passed=any(
                    x.get('uncut',{}).get('passed',False) for x in r['captures'].values()),
                receipts=receipts, run_sha256=digest(run_file), registration_sha256=digest(seal_file)),r


def verdict(optical, r):
    receipts,goal = optical['receipts'],optical['goal_mm']
    structure,tissue = r.get('final_structure') or {},r.get('final_tissue_stats') or {}
    closures = r.get('closures',0)
    last_move = r['moves'][-1] if r['moves'] else {}
    ready = bool(r.get('closed') and r.get('ready') and last_move.get('closed')
                 and last_move.get('accepted') and np.allclose(r['final_q'],r['ready_q'],atol=1e-6,rtol=0))
    real_stop = bool(closures and len(receipts)==closures and
                     [x['closure'] for x in receipts]==list(range(1,closures+1)) and
                     all(x['positive'] for x in receipts) and receipts[-1]['verdict']=='TARGET' and
                     receipts[-1]['prefix_mm']==goal)
    no_faults = bool(r.get('rcm') and all(v['failures']==0 and v['max_drift_m']<=.0005
                     and v['collision_blocks']==0 for v in r['rcm']) and tissue.get('torn')==0)
    structure_ok = structure.get('prefix_rows')==structure.get('farthest_row_end')==goal
    passed = (r['state']=='DONE' and not r['failure'] and not r['cleanup_error'] and not r.get('scoring_error')
              and optical['initial_passed'] and real_stop and ready and no_faults and structure_ok)
    return dict(**optical, passed=bool(passed), state=r['state'], reason=r['failure'], closures=closures,
                prefix_rows=structure.get('prefix_rows'), farthest_row_end=structure.get('farthest_row_end'),
                closed=r.get('closed'), ready_verified=ready, no_faults=no_faults)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    a = p.parse_args()
    plan = load(PLAN)
    data = {k:examine(a.batch/k,k,plan) for k in plan['cases']}
    if len({r['run_id'] for _,r in data.values()})!=2:
        raise ValueError('not two independent fresh runs')
    decisions = {k:verdict(*v) for k,v in data.items()}
    result = dict(plan_sha256=digest(PLAN), scorer_sha256=digest(Path(__file__)),cases=decisions,
                  passed=all(v['passed'] for v in decisions.values()), authorizes_disturbances=False)
    with (a.batch/'decision.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print({k:(v['state'],v['passed'],v['reason']) for k,v in decisions.items()})
