"""Seal independent optical estimates before joining any same-exposure scoring sidecars."""
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/d) for d in ('task','vision')]
from cut_perception import CutPerception
from m5_v3_corridor_v5 import measure as full_review
from m5_v3_perception_v4 import intact_preflight
from m5_v3_perception_v11 import _work_evidence
from m5_v4_frontier_v1 import Reviews
from run_m5_v4_bend_terminal_v1 import NOISE, OUT, REF, base, check, load, sha256, verify, write


def predict_frame(f,m,prefix,model,noise):
    rgb,z,R,t = f['rgb'],f['depth'],f['cam_R'],f['cam_t']
    focal,far = f['focal'].item(),f['far'].item()
    check(z.shape==(960,1280) and np.isclose(focal,640/np.tan(np.deg2rad(35)),atol=1e-6,rtol=0),
          'invalid wide optics')
    marker = CutPerception(1280,960,focal,far,model.rcm,tcp_local=model.tip_Y,jaw_axis_local=model.tip_R[:,2],
                           opening_axis_local=model.tip_R[:,0],ray_marker_fit=True)
    floating = rgb.astype(float)
    draws = []
    for split,seeds in [('clean',[None])]+[(s,noise[s+'_seeds']) for s in ('development','check') if prefix>=6]:
        for seed in seeds:
            sigma = .0 if seed is None else .0005
            measured = z
            if seed is not None:
                rng = np.random.default_rng(np.random.SeedSequence([seed,2,m['tick']]))
                measured = z+rng.normal(0.,sigma,z.shape)
            sol = _work_evidence(rgb,measured,focal=focal,far=far,sigma=sigma,prefix_mm=prefix,
                                 goal_mm=10 if prefix==9 else 9)
            mark,_ = marker._scissors(floating,measured)
            if seed is None:
                saved = m['marker']
                check(mark['valid']==saved['valid'] and (not mark['valid'] or all(np.allclose(mark[k],saved[k],
                      rtol=0,atol=1e-12) for k in ('tip','jaw_axis','opening_axis'))),'marker differs')
            d = dict(split=split,seed=seed,prefix=prefix,closed=m['closed'],marker=bool(mark['valid']),chart=sol is not None)
            if sol is not None and mark['valid']:
                g,row,_ = sol
                d.update(row=row,axis=(R@g.direction).tolist(),displacement=(R@(g.limit-mark['tip'])).tolist())
            draws.append(d)
    return draws


def predict(folder,r,model,cache):
    reviews,work = {},{}
    ledger = Reviews(r['run_id'])
    work_names,review_names = set(),set()
    noise = load(NOISE)
    for name,m in sorted(r['captures'].items(),key=lambda v:v[1]['tick']):
        with np.load(verify(folder/m['file'],m['sha256']),allow_pickle=False) as f:
            if m['fov_deg']==70:
                work_names.add(name)
                prefix = ledger.prefix or 0
                key = (m['sha256'],m['tick'],prefix,m['closed'])
                if key not in cache:
                    cache[key] = predict_frame(f,m,prefix,model,noise)
                work[name] = cache[key]
            else:
                review_names.add(name)
                check(m['fov_deg']==20 and f['depth'].shape==(1920,2560) and np.isclose(f['focal'].item(),
                      1280/np.tan(np.deg2rad(10)),atol=1e-6,rtol=0) and m['closed']
                      and np.allclose(f['q'],r['entrance_q'],rtol=0,atol=1e-6),'invalid review')
                if not m['closures'] and not m['uncut']['passed']:
                    check(not intact_preflight(f['rgb'],f['depth'],far=f['far'].item())['passed'],'preflight differs')
                    continue
                rec = ledger.observe(r['run_id'],m['tick'],m['closures'],m['closed'],f['rgb'],f['depth'],far=f['far'].item())
                check(rec==r['receipts'][name],'independent incremental replay differs')
                if m['closures']:
                    check(full_review(f['rgb'],f['depth'],far=f['far'].item(),sigma=0.)==m['optical'],
                          'independent full-prefix replay differs')
                reviews[name] = rec
    check(work_names==set(r['scoring_sidecars']) and review_names==set(r['review_sidecars']),'unpaired exposures')
    return dict(reviews=reviews,work=work)


def margin_gate(rows,coverage):
    late = [x for x in rows if x['prefix']>=6]
    summary = {}
    for part in ('clean','development','check'):
        trials = [x for x in late if x['split']==part]
        valid = [x for x in trials if 'optimism_mm' in x and x['row']==10]
        bound = .0 if part=='clean' else .1000
        summary[part] = dict(total=len(trials),valid=len(valid),max_optimism_mm=max(
            (x['optimism_mm'] for x in valid),default=None),bound_mm=bound,passed=bool(trials)
            and (len(valid)==len(trials) if part=='clean' else len(valid)>=math.ceil(.95*len(trials)))
            and all(x['optimism_mm']<=bound for x in valid))
    return dict(qualified=coverage and all(x['passed'] for x in summary.values()),coverage=coverage,parts=summary)


def grade_case(folder,r,opt,model):
    snapshots,draws,refs = {},[],{}
    baseline = base.DamageBaseline(load(verify(folder/'uncut.json',r['uncut_sha256'])))
    guards = (r['port_free'] and not r['error'] and 'untagged' not in r['truth'] and r['closures']<=4
              and not any(not v['accepted'] for v in r['moves']) and r['tissue']['torn']==0 and r['rcm']
              and all(v['failures']==v['collision_blocks']==0 and v['max_drift_m']<=.0005 for v in r['rcm']))
    final_edges = load(verify(folder/'final_edges.json',r['final_sha256']))
    final = baseline.score(final_edges)
    work_count = len(opt['work'])
    check(all(r['truth'].get('scoring',{}).get(k)==work_count for k in ('nodes','tcp')),'purpose tally')
    previous_stamp = -math.inf
    identity = True
    for name,m in sorted(r['captures'].items(),key=lambda v:v[1]['tick']):
        wide = name in opt['work']
        sf = r['scoring_sidecars' if wide else 'review_sidecars'][name]
        check(sf['file']==f'scoring/{name}.npz','unbound file')
        with (np.load(verify(folder/sf['file'],sf['sha256']),allow_pickle=False) as s,
              np.load(verify(folder/m['file'],m['sha256']),allow_pickle=False) as f):
            stamp = s['simulation_time_s' if wide else 'stamp'].item()
            check(stamp>=previous_stamp and np.isfinite(stamp) and s['tick'].item()==m['tick'] and
                  s['frame_sha256'].item()==m['sha256'] and s['closed'].item()==m['closed']
                  and np.array_equal(s['q'],f['q']),'invalid scoring binding')
            previous_stamp = stamp
            if not wide:
                check(s['closures'].item()==m['closures'] and np.array_equal(s['ecm'],m['ecm']),'review mismatch')
                snapshots[name] = baseline.score(s['edges'])
                if m['closures']==4:
                    check(np.array_equal(s['edges'],final_edges),'damage changed after terminal look')
                continue
            nodes,tcp = s['nodes_world'],s['tcp_world']
            check(nodes.shape==(14,24,3) and tcp.shape==(3,) and np.isfinite(nodes).all()
                  and np.isfinite(tcp).all(),'invalid reference')
            if m['tick']==1:
                with np.load(REF/'scoring/work.npz') as old:
                    identity = (np.array_equal(old['nodes_world'],nodes) and
                                m['sha256']==load(REF/'run.json')['captures']['work']['sha256'])
            seam = nodes[:,11:13].mean(1)
            axis = (np.arange(13)-6)@seam[:13]
            axis /= np.linalg.norm(axis)
            fk = 1000*np.linalg.norm(model.pose(f['q'])[:,3]-tcp)
            refs[name] = dict(fk_mm=float(fk),sidecar_sha256=sf['sha256'])
            for d in opt['work'][name]:
                row = dict(**d,frame_sha256=m['sha256'],tick=m['tick'])
                if 'axis' in d:
                    delta = np.asarray(d['displacement'])-(seam[10]-tcp)
                    row.update(optimism_mm=float(1000*delta@d['axis']),
                               axis_deg=float(np.rad2deg(np.arccos(np.clip(axis@d['axis'],-1,1)))))
                draws.append(row)
    known_correct = all(x['prefix_mm']==snapshots[n]['prefix_rows'] for n,x in opt['reviews'].items()
                        if x['prefix_mm'] is not None)
    last = next((n for n in snapshots if r['captures'][n]['closures']==4),None)
    last_read = opt['reviews'].get(last,{})
    last_state = snapshots.get(last,{})
    status = [x['status'] for x in last_read.get('rows',[])]
    contrast = (last is not None and last_state==final and r['closed'] and not r['ready'] and
                (last_read.get('verdict'),last_read.get('prefix_mm'))==
                (('TARGET',9) if r['case']=='full' else ('BELOW',8)))
    if contrast:
        contrast = (base.within_envelope(final,9) and
                    (final['prefix_rows']==final['farthest_partial_row_end']==final['farthest_row_end']==9
                     and status[7:11]==['OPEN','OPEN','INTACT','INTACT'] if r['case']=='full' else
                     final['prefix_rows']==8 and sum(final['faces'][8])==1 and final['farthest_partial_row_end']==9
                     and status[8:10]==['PARTIAL','INTACT']))
    n = 9 if r['case']=='full' else 8
    coverage = all({(7,True),(7,False),(n,True)}<={(x['prefix'],x['closed']) for x in draws
                      if x['split']==part and x.get('row')==10} for part in ('clean','development','check'))
    max_fk = max((s['fk_mm'] for s in refs.values()),default=None)
    max_axis = max((s['axis_deg'] for s in draws if 'axis_deg' in s),default=None)
    cal = margin_gate(draws,coverage)
    cal['qualified'] &= bool(max_fk is not None and max_fk<=.010 and max_axis is not None and max_axis<5.
                             and all('axis' in row for row in draws if row['split']=='clean'))
    summary = dict(safe=bool(guards and base.within_envelope(final,9)),identity=identity,refusal=r['refusal'],closures=r['closures'],
                   known_correct=known_correct,contrast=contrast,terminal_row8_status=status[8] if status else None,
                   final_structure=final,review_structure=snapshots,max_fk_mm=max_fk,max_axis_deg=max_axis,margin=cal)
    return summary,draws


def run():
    lock = load(OUT/'registration.json')
    for name,pin in lock['sources'].items():
        verify(ROOT/name,pin)
    cal = load(base.FK)['calibration']
    model = base.CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in cal.items()})
    runs,hashes,preds,cache = {},{},{},{}
    for case in lock['plan']['order']:
        path = OUT/case/'run.json'
        if not path.exists():
            continue
        r = runs[case] = load(path)
        hashes[case] = sha256(path)
        check(r['case']==case and r['registration_sha256']==sha256(OUT/'registration.json'),'wrong acquisition')
        if not r['error']:
            preds[case] = predict(path.parent,r,model,cache)
    pp = OUT/'optical_predictions.json'
    write(pp,dict(registration_sha256=sha256(OUT/'registration.json'),run_hashes=hashes,predictions=preds))
    cases,unique = {},{}
    for case,pred in preds.items():
        cases[case],rows = grade_case(OUT/case,runs[case],pred,model)
        for x in rows:
            k = (x['frame_sha256'],x['tick'],x['prefix'],x['seed'],x['closed'])
            if k in unique:
                check(unique[k]==x,'duplicated image has different scoring reference')
            unique[k] = x
    paired = False
    if len(cases)==2 and all('fork' in r for r in runs.values()):
        a,b = runs.values()
        ta,tb = a['fork'],b['fork']
        def common(r,t):
            return sorted((m['tick'],m['sha256']) for m in r['captures'].values() if m['tick']<=t)
        paired = (ta['tick']==tb['tick'] and ta['moves']==tb['moves'] and
                  common(a,ta['tick'])==common(b,tb['tick']) and a['moves'][:ta['moves']]==b['moves'][:tb['moves']])
    dp = OUT/'measurements.json'
    write(dp,list(unique.values()))
    margin = margin_gate(unique.values(),len(cases)==2 and all(v['margin']['qualified'] for v in cases.values()))
    checks = dict(paired=paired,independent_contrast=len(cases)==2 and all(v['contrast'] for v in cases.values()),
                  observed_safety=bool(cases) and len(cases)==len(runs) and all(v['safe'] and v['identity'] and
                  v['known_correct'] for v in cases.values()),margin=margin['qualified'])
    write(OUT/'decision.json',dict(passed=all(checks.values()),checks=checks,cases=cases,unique_margin=margin,
          errors={k:r['error'] for k,r in runs.items()},missing=sorted(set(lock['plan']['order'])-runs.keys()),
          authorizes_motion=False,registration_sha256=sha256(OUT/'registration.json'),run_hashes=hashes,
          optical_sha256=sha256(pp),measurements_sha256=sha256(dp)))
    print('terminal controls',checks,'late margin',margin,flush=True)
    return all(checks.values())


if __name__=='__main__':
    sys.exit(0 if run() else 1)
