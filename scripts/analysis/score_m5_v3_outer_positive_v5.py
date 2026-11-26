"""Replay v5 optical receipts, uncut-to-final damage and same-exposure references."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_m5_v3_optics_attribution_v1 import ROOT, digest, load, verified
from cut_perception import CutPerception
import m5_v3_perception_v11 as reader
from score_m5_v3_outer_positive_v3 import examine, verdict
from seam_damage_v2 import DamageBaseline, exact_endpoint


PLAN = ROOT/'config/task/m5_v3_positive_outer_v5.json'
REQUIRED = [('BELOW',3),('BELOW',5),('BELOW',7),('TARGET',9)]
REFERENCE = ROOT/'validation/task/m5_v3_sensor_margin_calibration_v2'


def score_work(folder,run,optical,plan):
    work = {k:m for k,m in run['captures'].items() if m['fov_deg']==70}
    tags = run['truth_reads_by_purpose'].get('scoring',{})
    if (set(work)!=set(run['scoring_sidecars']) or not work
            or any(tags.get(k)!=len(work) for k in ('nodes','tcp'))):
        raise ValueError('missing same-exposure scoring reference')
    cal = load(ROOT/'validation/task/m5_v3_rigid_fk_v2.json')['calibration']
    refs = plan['fixture_geometry']['sources']
    first,checks = {},[]
    for name,m in sorted(work.items(),key=lambda item:item[1]['tick']):
        ref = run['scoring_sidecars'][name]
        if ref['file']!='scoring/'+name+'.npz':
            raise ValueError('wrong sidecar name')
        with (np.load(verified(folder/m['file'],m['sha256']),allow_pickle=False) as f,
              np.load(verified(folder/ref['file'],ref['sha256']),allow_pickle=False) as s):
            nodes,tcp = s['nodes_world'],s['tcp_world']
            if (s['tick'].item()!=m['tick'] or s['closed'].item()!=m['closed']
                    or s['frame_sha256'].item()!=m['sha256'] or not np.array_equal(s['q'],f['q'])
                    or nodes.shape!=(14,24,3) or tcp.shape!=(3,) or not np.isfinite(nodes).all()
                    or not np.isfinite(tcp).all() or not np.isfinite(s['simulation_time_s'].item())):
                raise ValueError('unbound or invalid reference: '+name)
            if m['tick']==1:
                for data,path,keys in ((f,REFERENCE/'001_work.npz',('rgb','depth','q','cam_R','cam_t','focal','far')),
                                        (s,REFERENCE/'scoring/001_work.npz',('nodes_world',))):
                    with np.load(verified(path,refs[path.relative_to(ROOT).as_posix()])) as old:
                        first.update({k:np.array_equal(data[k],old[k]) for k in keys})
            prev = [x['prefix_mm'] for x in optical['receipts'] if x['positive'] and x['tick']<m['tick']]
            prefix = prev[-1] if prev else 0
            focal,far = f['focal'].item(),f['far'].item()
            rgb,z = f['rgb'],f['depth']
            sol,_,recovered = reader._solve_work(rgb,z,focal=focal,far=far,sigma=0.,
                                                 prefix_mm=prefix,goal_mm=run['goal_mm'])
            if recovered:
                rec = dict(frame=name,prefix_mm=prefix,recovered=recovered,used=sol is not None)
                if sol is not None:
                    g,row,_,_ = sol
                    mark,_ = CutPerception(1280,960,focal,far,np.array(cal['rcm']),
                        tcp_local=np.array(cal['tip_Y']),jaw_axis_local=np.array(cal['tip_R'])[:,2],
                        opening_axis_local=np.array(cal['tip_R'])[:,0],ray_marker_fit=True)._scissors(rgb.astype(float),z)
                    if row!=10 or not mark['valid'] or not np.allclose(mark['tip'],m['marker']['tip'],atol=1e-12,rtol=0):
                        raise ValueError('unscorable natural corner: '+name)
                    R = f['cam_R']
                    rec['optimism_mm'] = float(1000*(R@(g.limit-mark['tip'])-(nodes[10,11:13].mean(0)-tcp))@(R@g.direction))
                checks.append(rec)
    return dict(first_frame=first,first_frame_identical=bool(first) and all(first.values()),
                paired_sidecars=len(work),natural_fallback=checks,
                fallback_envelope_passed=all(r['optimism_mm']<=0. for r in checks if r['used']))


def score(folder,case,plan):
    optical,run = examine(folder,case,plan)
    seal = load(folder/'registration.json')
    for path in (PLAN,Path(__file__).resolve(),ROOT/plan['controller']):
        verified(path,seal['sources'][path.relative_to(ROOT).as_posix()])
    truth = run.get('truth_reads_by_purpose',{})
    if 'untagged' in truth:
        raise ValueError('untagged truth in live trace')
    structure = edges_sha = None
    if run.get('final_constraints_sha256'):
        edges_sha = run['final_constraints_sha256']
        uncut = load(verified(folder/'uncut_constraints.json',run['uncut_constraints_sha256']))
        structure = DamageBaseline(uncut).score(load(verified(folder/'final_constraints.json',edges_sha)))
        recorded = json.loads(json.dumps(structure))
        if recorded!=run.get('final_structure'):
            raise ValueError('independent material replay differed')
    goal = plan['cases'][case]
    result = verdict(optical,dict(run,final_structure=structure))
    sequence = [(x['verdict'],x['prefix_mm']) for x in optical['receipts']]
    expected = REQUIRED+[('TARGET',10)] if goal==10 else REQUIRED
    faces = None if structure is None else list(structure['faces'][goal-1])
    accepted_moves = bool(run['moves']) and all(m['accepted'] for m in run['moves'])
    evidence = score_work(folder,run,optical,plan) if run['captures'] and 'truth_reads_by_purpose' in run else {}
    result.update(required_sequence_passed=sequence==expected,
                  work_audit=evidence,
                  farthest_partial_row_end=None if structure is None else structure['farthest_partial_row_end'],
                  closure_count_passed=run.get('closures')==len(expected),
                  both_terminal_faces_cut=faces==[False,False],terminal_faces_intact=faces,
                  final_constraints_sha256=edges_sha,all_moves_accepted=accepted_moves,
                  failure=run.get('failure'),cleanup_error=run.get('cleanup_error'),
                  scoring_error=run.get('scoring_error'))
    result['passed'] &= (result['required_sequence_passed'] and result['closure_count_passed']
                         and result['both_terminal_faces_cut'] and accepted_moves
                         and structure is not None and exact_endpoint(structure,goal)
                         and evidence.get('first_frame_identical',False) and evidence.get('fallback_envelope_passed',False))
    return result,run['run_id']


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    args = parser.parse_args()
    plan = load(PLAN)
    cases = {}
    ids = set()
    for case in plan['cases']:
        cases[case],run_id = score(args.batch/case,case,plan)
        ids.add(run_id)
    if len(ids)!=2:
        raise ValueError('expected two independently started cases')
    result = dict(plan_sha256=digest(PLAN),scorer_sha256=digest(Path(__file__)),
                  cases=cases,passed=all(r['passed'] for r in cases.values()),authorizes_disturbances=False)
    with (args.batch/'decision.json').open('x',encoding='utf-8',newline='\n') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n')
    print({case:(r['state'],r['passed'],r['reason']) for case,r in cases.items()},flush=True)
