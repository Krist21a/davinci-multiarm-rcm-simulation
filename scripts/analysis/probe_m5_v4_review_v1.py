"""Register and score a once-only, no-permit alternate-view observation experiment."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/s) for s in ('task','vision')]
from m5_v4_frontier_v1 import Reviews, inks, measure
from preflight_m5_v3_fixture_v1 import load, write
from run_cut_task import sha256
from seam_damage_v2 import DamageBaseline
from seam_topology import material_state


ID = 'm5_v4_review_probe_v1'
DATA = ROOT/'validation/task'
OUT = DATA/ID
PLAN = ROOT/f'config/task/{ID}.json'
BENCH = ROOT/'config/task/m5_v4_benchmark_v1.json'
REG = ROOT/'validation/safety/regressions_m5_v4_review_probe_v1_sealed.json'
PARENT = DATA/'m5_v3_positive_outer_v5/nominal'
LATERAL = DATA/'m5_guided_slip_lateral_capture_v1'
OLD = DATA/'m5_v3_positive_outer_v3/exact_10mm'
RUNNER = ROOT/'scripts/task/capture_m5_v4_review_probe_v1.py'


def check(ok,reason):
    if not ok:
        raise ValueError(reason)


def checked(f,pin):
    check(sha256(f)==pin,'changed input: '+str(f))
    return f


def inputs(folder):
    decision = load(folder.parent/'decision.json')['cases'][folder.name]
    run = load(checked(folder/'run.json',decision['run_sha256']))
    reg = load(checked(folder/'registration.json',run['registration_sha256']))
    check(reg['run_id']==run['run_id'] and run['case']==folder.name,'historical identity')
    return run


def validate_manifest():
    seal = OUT/'registration.json'
    lock = load(seal)
    for name,h in lock['sources'].items():
        checked(ROOT/name,h)
    check(lock['plan']==load(PLAN) and lock['benchmark']==load(BENCH),'changed experiment')
    return lock


def exposure(folder,m):
    with np.load(checked(folder/m['file'],m['sha256']),allow_pickle=False) as f:
        check(f['depth'].shape==(1920,2560) and np.isclose(f['focal'].item(),1280/np.tan(np.deg2rad(10)),
              rtol=0,atol=1e-6),'unregistered review optics')
        return f['rgb'],f['depth'],f['far'].item()


def sequence(folder,run):
    ledger = Reviews(run['run_id'])
    reads = {}
    for name,m in sorted(run['captures'].items(),key=lambda kv:kv[1]['tick']):
        if m['fov_deg']!=20 or m['closures']>4:
            continue
        rgb,z,far = exposure(folder,m)
        reads[name] = ledger.observe(run['run_id'],m['tick'],m['closures'],m['closed'],rgb,z,far=far,
                                     alternate=name=='019_alternate')
    return reads


def framing(primary,alternative):
    with np.load(primary,allow_pickle=False) as f,np.load(alternative,allow_pickle=False) as g:
        bank = inks(f['rgb'])
        z = f['depth']
        h,w = z.shape
        src_f = f['focal'].item()
        dst_f = g['focal'].item()
        counts = {}
        for s in range(2):
            for j in range(6,11):
                y,x = np.where((bank==j+13*s)&np.isfinite(z)&(z>0)&(z<f['far'].item()*.999))
                d = z[y,x]
                xyz = np.column_stack(((w*.5-x-.5)*d/src_f,(h*.5-y-.5)*d/src_f,d))
                other = (xyz@f['cam_R'].T+f['cam_t']-g['cam_t'])@g['cam_R']
                x2,y2 = (np.array([w,h])*.5-.5-dst_f*other[:,:2]/other[:,2:]).T
                counts[f'{s}:{j}'] = int(((other[:,2]>0)&(other[:,2]<g['far'].item()*.999)
                                           &(x2>=0)&(x2<w)&(y2>=0)&(y2<h)).sum())
    return counts


def register(archive):
    plan,bench = load(PLAN),load(BENCH)
    pins = dict(bench['pins'],**plan['pins'])
    for name,h in pins.items():
        checked(ROOT/name,h)
    regression = load(REG)
    check(regression['success'] and regression['tests_run']>=489,'missing full offline suite')
    for p in (PLAN,BENCH,RUNNER,Path(__file__).resolve(),ROOT/'scripts/vision/m5_v4_frontier_v1.py',
              ROOT/'scripts/tests/test_m5_v4_frontier_v1.py'):
        name = p.relative_to(ROOT).as_posix()
        checked(p,regression['source_sha256'][name])
        pins[name] = sha256(p)
    pins[REG.relative_to(ROOT).as_posix()] = sha256(REG)
    for folder in (OLD,PARENT):
        run = inputs(folder)
        reg = load(folder/'registration.json')
        for name,h in reg['sources'].items():
            checked(ROOT/name,h)
            pins[name] = h
        for name in ('registration.json','run.json'):
            pins[(folder/name).relative_to(ROOT).as_posix()] = sha256(folder/name)
    lateral = load(LATERAL/'decision.json')
    r = load(checked(LATERAL/'slip_regrasp.json',lateral['identity']['run_sha256']))
    event, = [e for e in r['events'] if e['event']=='slip_optical_pair' and e['phase']=='lateral']
    alt = checked(archive/event['narrow'],event['narrow_sha256'])
    check(lateral['checks']['sole_guarded_camera_attempt'] and
          lateral['attempts'][0]['target']==plan['alternative'],'alternate pose not achieved historically')
    struct = checked(LATERAL/event['structure'],event['structure_sha256'])
    OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',dict(plan=plan,benchmark=bench,sources=pins,lateral=event))
    baseline = inputs(PARENT)
    old = inputs(OLD)
    hist = {tag:sequence(folder,run) for tag,folder,run in [('old',OLD,old),('current',PARENT,baseline)]}
    rgb,z,far = exposure(OLD,old['captures']['022_review'])
    partial = {str(n):measure(rgb,z,far=far,sigma=0.,prefix=n) for n in (7,9)}
    with np.load(alt,allow_pickle=False) as f:
        check(np.array_equal(f['depth'],f['measured']),'unexpected noise')
        rgb,z,far = f['rgb'],f['measured'],f['far'].item()
        negatives = [measure(rgb,z,far=far,sigma=0.,prefix=n) for n in range(8)]
    primary = baseline['captures']['018_review']
    counts = framing(checked(PARENT/primary['file'],primary['sha256']),alt)
    predictions = OUT/'offline_predictions.json'
    write(predictions,dict(registration_sha256=sha256(OUT/'registration.json'),sequences=hist,
                           old_partial=partial,lateral_negative=negatives,framing=counts))
    with np.load(struct,allow_pickle=False) as f:
        lbl = material_state(f['edges'])
    def ordinals(case):
        return [(r['verdict'],r['prefix_mm']) for r in hist[case].values()]
    common = [('UNCUT',0),('BELOW',3),('BELOW',5),('BELOW',7)]
    gates = dict(prior=ordinals('old')==common+[('TARGET',9)],
                 primary=ordinals('current')==common+[('UNKNOWN',None)],
                 old_halfcut=old['final_structure']['faces'][9].count(True)==1
                 and all(x['rows'][9]['status']=='PARTIAL' and x['verdict']=='UNKNOWN' for x in partial.values()),
                 same_angle_negative=lbl['prefix_rows']==8 and sum(lbl['faces'][8])==1 and all(
                     (x['verdict'],x['prefix_mm']) in (('UNKNOWN',None),('BELOW',8)) for x in negatives),
                 field_of_view=min(counts.values())>=8)
    out = dict(gates=gates,passed=all(gates.values()),predictions_sha256=sha256(predictions),
               registration_sha256=sha256(OUT/'registration.json'),authorizes_endpoint_motion=False,
               negative_positive_discrimination=negatives[7]['verdict']=='BELOW' and
                   negatives[7]['prefix_mm']==8 and negatives[7]['rows'][8]['status']=='PARTIAL')
    write(OUT/'offline_gate.json',out)
    print('offline gates:',gates,'lateral negative:',negatives[7]['verdict'],flush=True)
    return out['passed']


def score():
    validate_manifest()
    case = OUT/'capture'
    run = load(case/'run.json')
    gates = load(OUT/'offline_gate.json')
    checked(OUT/'registration.json',run['registration_sha256'])
    checked(OUT/'offline_gate.json',run['gate_sha256'])
    checked(OUT/'offline_predictions.json',gates['predictions_sha256'])
    reads = sequence(case,run)
    check(reads==run['online_reviews'],'offline review differs from acquisition')
    pred = OUT/'live_predictions.json'
    write(pred,dict(run_sha256=sha256(case/'run.json'),registration_sha256=run['registration_sha256'],reads=reads))
    before,after = (run.get(k,{}) for k in ('before','after'))
    b = DamageBaseline(load(checked(case/'uncut.json',run['uncut_sha256']))) if before else None
    structures = {}
    for k,s in (('before',before),('after',after)):
        if s:
            structures[k] = b.score(load(checked(case/(k+'_edges.json'),s['constraints_sha256'])))
    alt = reads.get('019_alternate',{})
    states = [r['status'] for r in alt.get('rows',[])]
    ref = inputs(PARENT)
    match = all(run['captures'].get(k,{}).get('sha256')==m['sha256'] for k,m in ref['captures'].items())
    measured_pose = False
    if alt:
        primary,second = (run['captures'][name] for name in ('018_review','019_alternate'))
        with (np.load(case/primary['file']) as f,np.load(case/second['file']) as g):
            measured_pose = (np.array_equal(f['q'],g['q']) and np.array_equal(f['q'],ref['entrance_q'])
                             and not np.allclose(f['cam_R'],g['cam_R'],rtol=0,atol=1e-6))
    work = {k:v for k,v in run['captures'].items() if v['fov_deg']==70}
    paired = set(work)==set(run['scoring_sidecars'])
    for k,v in work.items():
        if k not in run['scoring_sidecars']:
            paired = False
            continue
        ref = run['scoring_sidecars'][k]
        with np.load(checked(case/ref['file'],ref['sha256']),allow_pickle=False) as f:
            paired &= (f['frame_sha256'].item()==v['sha256'] and f['tick'].item()==v['tick']
                       and f['nodes_world'].shape==(14,24,3) and np.isfinite(f['nodes_world']).all())
    checks = dict(trace_match=match and len(run['captures'])==19,
                  scoring_isolated=bool(paired and len(work)==13 and 'untagged' not in run.get('truth_reads',{})
                      and run.get('truth_reads',{}).get('scoring',{}).get('nodes')==len(work)),
                  no_fault=bool(run.get('rcm')) and all(arm['failures']==0 and arm['collision_blocks']==0
                      and arm['max_drift_m']<=.0005 for arm in run['rcm']),
                  acquisition=not run['error'] and not run['cleanup_error'] and run['attempts']==1 and
                  bool(after) and before['closed'] and after['closed'] and before['closures']==after['closures']==4
                  and run.get('closed') and not run.get('ready') and run.get('closures')==4
                  and before['moves']==after['moves'] and np.array_equal(before['q'],after['q'])
                  and np.allclose(after['ecm'],load(PLAN)['alternative'],atol=1e-6,rtol=0) and measured_pose,
                  unchanged_structure=bool(after) and before['constraints_sha256']==after['constraints_sha256']
                  and all(s['prefix_rows']==s['farthest_row_end']==s['farthest_partial_row_end']==9
                          for s in structures.values()),
                  recognition=alt.get('verdict')=='TARGET' and alt['prefix_mm']==9
                  and states[6:11]==['OPEN','OPEN','OPEN','INTACT','INTACT'],
                  negative_discrimination=gates['negative_positive_discrimination'])
    result = dict(checks=checks,passed=all(checks.values()),predictions_sha256=sha256(pred),
                  registration_sha256=run['registration_sha256'],structures=structures,
                  authorizes_motion=False,disturbances_tested=False)
    write(OUT/'decision.json',result)
    print('development decision:',checks,flush=True)
    return result['passed']


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','score'))
    p.add_argument('--archive',type=Path)
    a = p.parse_args()
    if a.phase=='register' and a.archive is None:
        p.error('register needs --archive for the pinned lateral capture')
    sys.exit(0 if (register(a.archive) if a.phase=='register' else score()) else 1)
