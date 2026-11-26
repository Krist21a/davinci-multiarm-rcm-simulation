"""Replay the frozen hinge controller's optical inputs before joining structural labels."""
import argparse
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision')]
from m5_v4_hinge_controller_v1 import receipt
from m5_v4_hinge_work_v1 import chart, margin
from run_m5_v4_hinge_closed_loop_v1 import INV, OUT, PLAN, base, check, load, lock, marker_reader, sensor, sha256, term, verify, write


def model():
    return base.CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in load(base.FK)['calibration'].items()})


def work_estimate(rgb,z,f,far,R,fk,n,sigma):
    c = chart(rgb,z,f,far,sigma,n,10 if n==9 else 9)
    mk,_ = marker_reader(fk,f,far)._scissors(rgb.astype(float),z)
    if c is None or not mk['valid']:
        return None
    return dict(axis=(R@c.direction).tolist(),displacement=(R@(c.limit-mk['tip'])).tolist())


def draw(z,seed,tick):
    if seed is None:
        return z,0.
    return z+np.random.default_rng(np.random.SeedSequence([seed,2,tick])).normal(0.,.0005,z.shape),.0005


def offline():
    lock()
    source = load(INV/'optical_predictions.json')
    labels = load(INV/'transfer_measurements.json')
    bound = {'clean':.07,'development':.22,'check':.22}
    fk = model()
    ref = load(term.OUT/'decision.json')
    runs = {k:load(verify(term.OUT/k/'run.json',v)) for k,v in ref['run_hashes'].items()}
    clean,matched = [],0
    for h,draws in source.items():
        case,m = next((k,m) for k,r in runs.items() for m in r['captures'].values() if m['sha256']==h)
        with np.load(verify(term.OUT/case/m['file'],h)) as f:
            for row in draws:
                z,s = draw(f['depth'],row['seed'],row['tick'])
                est = work_estimate(f['rgb'],z,f['focal'].item(),f['far'].item(),f['cam_R'],fk,row['prefix'],s)
                check((est is None)==('row' not in row) and (not est or all(
                      np.allclose(est[k],row[k],rtol=0,atol=1e-12) for k in est)),'online versus archive inverse drift')
                matched += 1
                if row['seed'] is None:
                    clean.append(h)
    parts = {}
    for s,b in bound.items():
        rows = [r for r in labels if r['split']==s]
        valid = [r['optimism_mm'] for r in rows if 'optimism_mm' in r]
        parts[s] = dict(total=len(rows),valid=len(valid),maximum_mm=max(valid),bound_mm=b,
                        passed=len(valid)>=.95*len(rows) and max(valid)<=b)
    ok = len(clean)==6 and matched==len(labels)==150 and all(p['passed'] for p in parts.values())
    write(OUT/'offline_gate.json',dict(registration_sha256=sha256(OUT/'registration.json'),matched=matched,
          parts=parts,passed=ok))
    check(ok,'archive hinge envelope failed')
    print('archived work replay:',matched,'split gate',parts,flush=True)


def summarize_margin(rows):
    groups = {}
    for name,sigma in [('native',0.),('check',.0005)]:
        part = [r for r in rows if r['sigma']==sigma]
        vals = [r['optimism_mm'] for r in part if 'optimism_mm' in r]
        b = 1000*margin(sigma)
        groups[name] = dict(total=len(part),valid=len(vals),bound_mm=b,maximum_mm=max(vals,default=None),
                            passed=bool(vals) and len(vals)>=.95*len(part) and max(vals)<=b)
    return groups


def score(case):
    lock()
    folder = OUT/case
    r = load(folder/'run.json')
    verify(OUT/'registration.json',r['registration_sha256'])
    if 'final_sha256' not in r:
        write(folder/'decision.json',dict(passed=False,mechanically_safe=False,error=r['error'] or 'incomplete episode',
              run_sha256=sha256(folder/'run.json')))
        return False
    fk = model()
    sigma = .0005 if case=='depth_noise_0p5' else 0.
    seen = set(load(INV/'optical_predictions.json'))
    seeds = load(term.NOISE)['check_seeds']
    pred,reviews,prior = {},{},None
    for name,m in sorted(r['captures'].items(),key=lambda pair:pair[1]['tick']):
        with np.load(verify(folder/m['file'],m['sha256'])) as f:
            rgb,z = sensor(f['rgb'],f['depth'],m['tick'],sigma)
            far = f['far'].item()
            check(m['sigma']==sigma,'changed live noise')
            if sigma:
                check(hashlib.sha256(z.tobytes()).hexdigest()==m['sensor_depth_sha256'] and
                      np.isclose((z-f['depth']).std(),sigma,rtol=.01,atol=0.),'noise did not reach actual sensor')
            if name in r['receipts']:
                saved = r['receipts'][name]
                check(prior==saved['prior'] and m['closed'] and np.allclose(f['q'],r['entrance_q'],rtol=0,atol=1e-6),
                      'stale/displaced review')
                v,n = receipt(SimpleNamespace(rgb=rgb,depth=z,far=far),prior,sigma)
                check((v,n)==(saved['verdict'],saved['prefix']),'review replay differs')
                reviews[name] = dict(verdict=v,prefix=n,tick=m['tick'],closures=m['closures'],shutter=m['shutter'])
                if n is not None:
                    prior = n
            elif name in r['scoring_sidecars'] and 'work_prefix' in m:
                n = m['work_prefix']
                check(n==(prior or 0),'invalid working ordinal')
                def read(depth,s):
                    return work_estimate(rgb,depth,f['focal'].item(),far,f['cam_R'],fk,n,s)
                actual = read(z,sigma)
                saved = m['work_estimate']
                check((actual is None)==(saved is None) and (actual is None or all(np.allclose(
                      actual[k],saved[k],atol=1e-12,rtol=0) for k in actual)),'online work replay mismatch')
                draws = []
                if case=='nominal' and n>=6 and m['sha256'] not in seen:
                    for seed in [None,*seeds]:
                        zz,s = draw(f['depth'],seed,m['tick'])
                        draws.append(dict(seed=seed,sigma=s,estimate=read(zz,s)))
                pred[name] = dict(actual=actual,draws=draws,late=n>=6,hash=m['sha256'])
    pp = folder/'optical_replay.json'
    write(pp,dict(work=pred,reviews=reviews))
    damage = base.DamageBaseline(load(verify(folder/'uncut.json',r['uncut_sha256'])))
    final_edges = load(verify(folder/'final.json',r['final_sha256']))
    final = damage.score(final_edges)
    target = None
    agreement,unchanged = True,False
    for name,opt in reviews.items():
        s = r['review_sidecars'][name]
        m = r['captures'][name]
        with np.load(verify(folder/s['file'],s['sha256'])) as f:
            check(f['frame_sha256'].item()==m['sha256'] and f['closures'].item()==m['closures'] and
                  f['tick'].item()==m['tick'] and f['closed'].item() and np.array_equal(f['q'],m['q']),'unbound review')
            state = damage.score(f['edges'])
            agreement &= base.within_envelope(state,9) and (opt['prefix'] is None or opt['prefix']==state['prefix_rows'])
            if opt['verdict']=='TARGET' and not m['alternate']:
                target = name
                unchanged = np.array_equal(f['edges'],final_edges)
    rows,new,margins,trace = [],[],[],{}
    for name,p in pred.items():
        m,s = r['captures'][name],r['scoring_sidecars'][name]
        with np.load(verify(folder/s['file'],s['sha256'])) as f, np.load(verify(folder/m['file'],m['sha256'])) as image:
            check(f['frame_sha256'].item()==m['sha256'] and f['tick'].item()==m['tick'] and
                  f['closed'].item()==m['closed'] and np.array_equal(image['q'],f['q']),'unbound work')
            nodes,tcp = f['nodes_world'],f['tcp_world']
            trace[name] = hashlib.sha256(nodes.tobytes()).hexdigest()
            check(np.linalg.norm(fk.pose(f['q'])[:,3]-tcp)<=.000010,'FK v2 discrepancy')
            truth = nodes[10,11:13].mean(0)-tcp
            if p['draws']:
                new.append(name)
            for d in p['draws']+([dict(estimate=p['actual'],sigma=sigma,live=True)] if p['late'] else []):
                est = d['estimate']
                row = dict(name=name,sigma=d['sigma'],seed=d.get('seed'))
                if est:
                    row['optimism_mm'] = float(1000*(np.asarray(est['displacement'])-truth)@est['axis'])
                (margins if d.get('live') else rows).append(row)
    parts = summarize_margin(rows)
    live_room = all(x.get('optimism_mm',0.)<=1000*margin(sigma) for x in margins)
    t = r['truth']
    safe = (not r['error'] and r['port_free'] and base.within_envelope(final,9) and r['tissue']['torn']==0 and
            r['closures']<=5 and bool(r['rcm']) and all(v['failures']==v['collision_blocks']==0 and v['max_drift_m']<=.0005
            for v in r['rcm']) and all(m['accepted'] for m in r['moves']) and 'untagged' not in t)
    withdrawal = False
    if target:
        m = r['captures'][target]
        tail = r['moves'][m['moves']:]
        withdrawal = (unchanged and len(tail)==1 and tail[0]['closed'] and tail[0]['accepted'] and r['ready'] and
                      r['closed'] and np.allclose(r['q'],r['ready_q'],rtol=0,atol=1e-6) and
                      np.allclose(tail[0]['q'],r['ready_q'],rtol=0,atol=1e-6) and
                      m['closures']==r['closures'] and not any(final['faces'][final['prefix_rows']-1]))
    event_ok = exercise(case,r,pred,reviews)
    gates = dict(mechanics=safe,verification=bool(withdrawal and agreement and target and r['state']=='DONE'),
                 envelope=live_room and (case!='nominal' or (bool(new) and all(s['passed'] for s in parts.values()))),
                 disturbance=event_ok,nominal_signature=case!='nominal' or (r['closures']==4 and
                 all(final[k]==9 for k in ('prefix_rows','farthest_row_end','farthest_partial_row_end'))))
    write(folder/'decision.json',dict(passed=all(gates.values()),gates=gates,mechanically_safe=safe,
          run_sha256=sha256(folder/'run.json'),replay_sha256=sha256(pp),state=r['state'],reason=r['reason'],reviews=reviews,
          target=target,closures=r['closures'],ready=r['ready'],closed=r['closed'],final=final,withdrawal_unchanged=unchanged,
          new_nominal_frames=new,new_nominal_margin=parts,actual_late_margin=margins,heldout_rows=rows,nodes_sha256=trace))
    print(case,r['state'],r['reason'],'closure count',r['closures'],'final',final['prefix_rows'],gates,flush=True)
    return all(gates.values())


def exercise(case,r,pred,reviews):
    if case=='nominal':
        return not r['event']
    if case=='depth_noise_0p5':
        return bool(pred) and all(m['sigma']==.0005 for m in r['captures'].values())
    e = r['event']
    if not e or e.get('bites')!=2 or 'after' not in e:
        return False
    before,after = e['before'],e['after']
    if not before['closed'] or not after['closed'] or before['q']!=after['q']:
        return False
    observed = any(d['prefix'] and d['closures']==2 and not d['shutter'] and d['tick']>e['tick'] for d in reviews.values())
    t = min((d['tick'] for d in reviews.values() if d['prefix'] and d['closures']==2 and not d['shutter']),default=400)
    fresh = observed and any(p['actual'] and r['captures'][n]['tick']>t and r['captures'][n]['closures']==2
                             for n,p in pred.items())
    if case=='occlusion_transient':
        return fresh and any(d['shutter'] and d['verdict']=='UNKNOWN' for d in reviews.values()) and any(
               s['state']=='OCCLUDED' for s in r['trace'])
    end = e.get('recovered')
    if not end or end['q']!=before['q'] or not end['closed']:
        return False
    if case=='relax':
        return fresh and e['state']=='RECOVER_TENSION' and after['strain']<before['strain'] and all(
            not np.allclose(before['grippers'][a],after['grippers'][a],rtol=0,atol=1e-6) and
            np.allclose(before['grippers'][a],end['grippers'][a],rtol=0,atol=1e-6) for a in before['grippers'])
    return fresh and e['state']=='RECOVER_GRASP' and before['held']['psm3']>0 and after['held']['psm3']==0 and end['held']['psm3']>0


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=load(PLAN)['cases'])
    args = p.parse_args()
    if args.case:
        score(args.case)
    else:
        offline()
