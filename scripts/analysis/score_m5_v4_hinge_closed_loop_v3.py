"""Independent v3 optical replay, endpoint/withdrawal scoring and matrix report."""
import argparse
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision')]
from m5_v4_hinge_controller_v1 import receipt
from m5_v4_hinge_marker_v3 import read, reader_for
from m5_v4_hinge_work_v3 import observe, margin
from run_m5_v4_hinge_closed_loop_v3 import OUT, PLAN, base, check, load, lock, old, sha256, term, verify, write
from score_m5_v4_hinge_closed_loop_v1 import draw, exercise, model, summarize_margin


def predict(r,folder,fk):
    sigma = .0005 if r['case']=='depth_noise_0p5' else 0.
    archived = set(load(old.INV/'optical_predictions.json'))
    expected,work,reviews = None,{},{}
    for name,m in sorted(r['captures'].items(),key=lambda kv:kv[1]['tick']):
        with np.load(verify(folder/m['file'],m['sha256'])) as f:
            rgb,raw,far,focal,R,t = (f[k] for k in ('rgb','depth','far','focal','cam_R','cam_t'))
            far,focal = far.item(),focal.item()
            _,z = old.sensor(rgb,raw,m['tick'],sigma)
            check(sigma==m['sigma'] and (not sigma or (hashlib.sha256(z.tobytes()).hexdigest()==m['sensor_depth_sha256']
                  and np.isclose((z-raw).std(),sigma,atol=0.,rtol=.01))),'noise replay mismatch')
            if name in r['receipts']:
                rec = r['receipts'][name]
                check(rec['prior']==expected and m['closed'] and np.allclose(f['q'],r['entrance_q'],atol=1e-6,rtol=0),
                      'unbound closed entrance')
                val,n = receipt(SimpleNamespace(rgb=rgb,depth=z,far=far),expected,sigma)
                check((val,n)==(rec['verdict'],rec['prefix']),'review mismatch')
                reviews[name] = dict(verdict=val,prefix=n,tick=m['tick'],closures=m['closures'],shutter=m['shutter'])
                if n is not None:
                    expected = n
                continue
            if name not in r['scoring_sidecars'] or 'work_prefix' not in m:
                continue
            n = m['work_prefix']
            check(n==(expected or 0),'stale work ordinal')
            reader = reader_for(fk,focal,far)
            def estimate(zz,sig):
                mk = read(reader,rgb,zz)
                g,route = observe(rgb,zz,focal,far,sig,n)
                if not mk['valid'] or g is None:
                    return mk,None,route
                return mk,dict(displacement=(R@(g.limit-mk['tip'])).tolist(),axis=(R@g.direction).tolist()),route
            mk,actual,route = estimate(z,sigma)
            saved = m['work_estimate']
            check(mk==m['v3_marker'] and route==m['chart_route'] and (n<6 or route=='primary') and
                  (actual is None)==(saved is None) and (not actual or all(np.allclose(actual[k],saved[k],
                  rtol=0,atol=1e-12) for k in actual)),'v3 work mismatch')
            draws = []
            if r['case']=='nominal' and n>=6 and m['sha256'] not in archived:
                for seed in [None,*load(term.NOISE)['check_seeds']]:
                    zz,s = draw(raw,seed,m['tick'])
                    _,est,_ = estimate(zz,s)
                    draws.append(dict(seed=seed,sigma=s,estimate=est))
            work[name] = dict(actual=actual,tip=(R@mk['tip']+t).tolist() if mk['valid'] else None,draws=draws,late=n>=6)
    return work,reviews


def score(case):
    lock()
    folder = OUT/case
    r = load(folder/'run.json')
    verify(OUT/'registration.json',r['registration_sha256'])
    if 'final_sha256' not in r:
        write(folder/'decision.json',dict(passed=False,mechanically_safe=False,run_sha256=sha256(folder/'run.json'),
              reason=r['error'] or 'incomplete episode'))
        return False
    fk = model()
    work,reviews = predict(r,folder,fk)
    pp = folder/'optical_replay.json'
    write(pp,dict(work=work,reviews=reviews))
    baseline = base.DamageBaseline(load(verify(folder/'uncut.json',r['uncut_sha256'])))
    edges = load(verify(folder/'final.json',r['final_sha256']))
    final = baseline.score(edges)
    bounds,unchanged,agree,target,withdraw = True,False,True,None,False
    for name,rec in reviews.items():
        m,s = r['captures'][name],r['review_sidecars'][name]
        with np.load(verify(folder/s['file'],s['sha256'])) as f:
            check(f['frame_sha256'].item()==m['sha256'] and f['tick'].item()==m['tick'] and
                  f['closures'].item()==m['closures'] and f['closed'].item() and np.array_equal(f['q'],m['q']),
                  'unbound edge reference')
            actual = baseline.score(f['edges'])
            agree &= rec['prefix'] in (None,actual['prefix_rows'])
            bounds &= base.within_envelope(actual,9)
            if rec['verdict']=='TARGET' and not m['alternate']:
                target = name
                unchanged = np.array_equal(f['edges'],edges)
                moves = r['moves'][m['moves']:]
                withdraw = (len(moves)==1 and moves[0]['closed'] and moves[0]['accepted'] and r['closed']
                            and r['ready'] and m['closures']==r['closures'] and not any(final['faces'][rec['prefix']-1])
                            and all(np.allclose(q,r['ready_q'],atol=1e-6,rtol=0) for q in (moves[0]['q'],r['q'])))
    held,late,tips,nodes_trace = [],[],[],{}
    for name,p in work.items():
        m,s = r['captures'][name],r['scoring_sidecars'][name]
        with np.load(verify(folder/s['file'],s['sha256'])) as side, np.load(verify(folder/m['file'],m['sha256'])) as f:
            check(side['frame_sha256'].item()==m['sha256'] and side['tick'].item()==m['tick'] and
                  side['closed'].item()==m['closed'] and np.array_equal(side['q'],f['q']),'unbound node/TCP reference')
            tcp,nodes = side['tcp_world'],side['nodes_world']
            check(np.linalg.norm(fk.pose(f['q'])[:,3]-tcp)<=.000010,'rigid FK mismatch')
            nodes_trace[name] = hashlib.sha256(nodes.tobytes()).hexdigest()
            truth = nodes[10,11:13].mean(0)-tcp
            if p['tip'] is not None:
                tips.append(float(1000*np.linalg.norm(p['tip']-tcp)))
            for row in p['draws']+([dict(live=True,estimate=p['actual'],sigma=m['sigma'])] if p['late'] else []):
                est = row['estimate']
                d = dict(name=name,sigma=row['sigma'],seed=row.get('seed'))
                if est is not None:
                    d['optimism_mm'] = float(1000*(est['displacement']-truth)@est['axis'])
                (late if row.get('live') else held).append(d)
    parts = summarize_margin(held)
    safe = (r['port_free'] and not r['error'] and bounds and base.within_envelope(final,9) and
            r['tissue']['torn']==0 and r['closures']<=5 and all(m['accepted'] for m in r['moves']) and
            'untagged' not in r['truth'] and bool(r['rcm']) and all(v['failures']==v['collision_blocks']==0 and
            v['max_drift_m']<=.0005 for v in r['rcm']))
    checks = dict(mechanics=safe,withdrawal=bool(target and unchanged and agree and withdraw and r['state']=='DONE'),
                  actual_margin=all(d['optimism_mm']<=1000*margin(d['sigma']) for d in late if 'optimism_mm' in d),
                  new_margin=case!='nominal' or (bool(held) and all(d['passed'] for d in parts.values())),
                  disturbance=exercise(case,r,work,reviews),nominal=case!='nominal' or (r['closures']==4 and all(
                  final[k]==9 for k in ('farthest_partial_row_end','farthest_row_end','prefix_rows'))))
    v1 = load(old.OUT/case/'run.json')
    ref = load(old.OUT/case/'decision.json')['nodes_sha256']
    identical = {}
    for name,h in nodes_trace.items():
        m = r['captures'][name]
        if m['tick']>9 or name not in ref or r['moves'][:m['moves']]!=v1['moves'][:v1['captures'][name]['moves']]:
            continue
        identical[name] = h==ref[name]
    checks['common_trace'] = bool(identical) and all(identical.values())
    old_frames = {m['sha256'] for m in v1['captures'].values()}
    repeat = sorted({d['name'] for d in held if r['captures'][d['name']]['sha256'] in old_frames})
    write(folder/'decision.json',dict(run_sha256=sha256(folder/'run.json'),replay_sha256=sha256(pp),gates=checks,
          v1_identical_node_checks=identical,heldout_frames_already_in_v1=repeat,
          passed=all(checks.values()),mechanically_safe=safe,final=final,state=r['state'],reason=r['reason'],
          closures=r['closures'],closed=r['closed'],ready=r['ready'],target=target,reviews=reviews,unchanged=unchanged,
          new_margin=parts,heldout=held,actual_margin=late,max_tip_error_mm=max(tips,default=None),nodes_sha256=nodes_trace))
    print(case,r['state'],r['reason'],checks,flush=True)
    return all(checks.values())


def matrix():
    cases = lock()['plan']['cases']
    summary = {}
    for case in cases:
        folder = OUT/case
        if not (folder/'decision.json').exists():
            summary[case] = dict(attempted=False,passed=False)
            continue
        d = load(folder/'decision.json')
        verify(folder/'run.json',d['run_sha256'])
        summary[case] = dict(attempted=True,decision_sha256=sha256(folder/'decision.json'),
                             **{k:d.get(k) for k in ('passed','state','closures','reason','target','closed','ready')})
    write(OUT/'matrix.json',dict(registration_sha256=sha256(OUT/'registration.json'),cases=summary,
          all_passed=all(c['passed'] for c in summary.values()),offline_gate_sha256=sha256(OUT/'offline_gate.json')))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=load(PLAN)['cases'])
    a = p.parse_args()
    if a.case:
        score(a.case)
    else:
        matrix()
