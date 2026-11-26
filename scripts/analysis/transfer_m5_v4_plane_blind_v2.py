"""Conditional archive-only metric transfer and optical work-distance replay."""
import numpy as np

from score_m5_v4_plane_blind_v2 import F, H, OUT, ROOT, W, check, checked, forward, load, old, projected, sha256, write
from audit_m5_v4_bend_rim_v1 import PARENT, Sheet, rays
from cut_perception import CutPerception
import m5_v3_perception_v11 as reader
from run_m5_v4_bend_terminal_v1 import NOISE, base
from score_m5_v4_bend_terminal_v1 import margin_gate
from score_m5_v4_plane_v1 import stats


def optical_inverse(point,depth,allowed,sigma):
    center = np.rint(projected(point,depth.shape,F)).astype(int)
    y,x = center
    if not (3<=y<depth.shape[0]-3 and 3<=x<depth.shape[1]-3):
        return None
    box = np.s_[y-3:y+4,x-3:x+4]
    mask,z = allowed[box],depth[box].astype(float)
    if not mask[3,3]:
        return None
    seen,pending = {(3,3)},[(3,3)]
    while pending:
        i,j = pending.pop()
        for p in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
            if (0<=p[0]<7 and 0<=p[1]<7 and p not in seen and mask[p] and
                    abs(z[p]-z[i,j])<=.001+3*sigma):
                seen.add(p)
                pending.append(p)
    if len(seen)<8:
        return None
    ij = np.array(sorted(seen))
    A = np.column_stack((np.ones(len(ij)),ij-3))
    c,_,rank,_ = np.linalg.lstsq(A,1./z[tuple(ij.T)],rcond=None)
    if rank!=3:
        return None
    return point/(1.+point[2]*.5*np.max(abs(c[1:])))


def optical_draws(rec,run,pinned):
    h,name,n = rec['h'],rec['name'],rec['prefix']
    m = run['captures'][name]
    seeds = load(NOISE)
    population = [('clean',None)]+[(k,s) for k in ('development','check') for s in seeds[k+'_seeds']]
    cal = load(base.FK)['calibration']
    with np.load(checked(PARENT/rec['case']/m['file'],h),allow_pickle=False) as f:
        rgb,z,focal,far,R = f['rgb'],f['depth'],f['focal'].item(),f['far'].item(),f['cam_R']
        check(z.shape==(H,W) and abs(focal-F)<1e-8,'wrong work camera')
        marker = CutPerception(W,H,focal,far,np.asarray(cal['rcm']),tcp_local=np.asarray(cal['tip_Y']),
            jaw_axis_local=np.asarray(cal['tip_R'])[:,2],opening_axis_local=np.asarray(cal['tip_R'])[:,0],ray_marker_fit=True)
        floating,label = rgb.astype(float),reader.inks(rgb)
        pink = (rgb==reader.PINK).all(-1)
        masks = [pink|((label>=0)&(label//13==s)) for s in (0,1)]
        rows = []
        for (split,seed),orig in zip(population,pinned,strict=True):
            check((orig['split'],orig['seed'],orig['prefix'],orig['closed'])==(split,seed,n,m['closed']),
                  'unexpected archive draw')
            sigma = .0 if seed is None else .0005
            measured = z if seed is None else z+np.random.default_rng(np.random.SeedSequence(
                [seed,2,m['tick']])).normal(0.,sigma,z.shape)
            sol,_,_ = reader._solve_work(rgb,measured,focal=F,far=far,sigma=sigma,prefix_mm=n,goal_mm=10 if n==9 else 9)
            mark,_ = marker._scissors(floating,measured)
            check(orig['chart']==(sol is not None) and orig['marker']==bool(mark['valid']),'baseline refusal drift')
            row = dict(frame_sha256=h,tick=m['tick'],split=split,seed=seed,prefix=n,closed=m['closed'])
            if sol is not None and mark['valid']:
                g,limit,points,_ = sol
                diff,d = R@(g.limit-mark['tip']),R@g.direction
                check(limit==orig['row']==10 and np.allclose(d,orig['axis'],atol=1e-12,rtol=0) and
                      np.allclose(diff,orig['displacement'],atol=1e-12,rtol=0),'baseline geometry drift')
                good = reader._good(measured,far)
                fixed = [optical_inverse(p,measured,good&mask,sigma) for p,mask in zip(points,masks,strict=True)]
                if all(p is not None for p in fixed):
                    delta = np.min(np.stack(fixed)@g.direction)-np.min(points@g.direction)
                    row.update(axis=d.tolist(),row=limit,displacement=(diff+delta*d).tolist(),delta_mm=1000*delta)
            rows.append(row)
    return rows


def transfer():
    audit = load(old.AUDIT/'decision.json')
    lock = load(checked(old.AUDIT/'registration.json',audit['registration_sha256']))
    support = load(checked(old.AUDIT/'support.json',audit['support_sha256']))
    prior = load(checked(PARENT/'decision.json',lock['plan']['decision_sha256']))
    baseline = load(checked(PARENT/'optical_predictions.json',prior['optical_sha256']))['predictions']
    refs = {(r['frame_sha256'],r['seed']):r for r in load(checked(PARENT/'measurements.json',prior['measurements_sha256']))
            if r['prefix']>=6}
    records = lock['records']
    optical,runs = {},{}
    for rec in records:
        case = rec['case']
        runs[case] = load(checked(PARENT/case/'run.json',prior['run_hashes'][case]))
        optical[rec['h']] = optical_draws(rec,runs[case],baseline[case]['work'][rec['name']])
    path = OUT/'optical_predictions.json'
    write(path,optical)
    metric,rows = {},[]
    for rec in records:
        case,name,h,prev = (rec[k] for k in ('case','name','h','review'))
        r = runs[case]
        folder = PARENT/case
        with (np.load(checked(folder/r['captures'][name]['file'],h)) as f,
              np.load(checked(folder/r['scoring_sidecars'][name]['file'],rec['files'][1])) as s,
              np.load(checked(folder/r['review_sidecars'][prev]['file'],rec['files'][2])) as e):
            check(s['frame_sha256'].item()==h and np.array_equal(s['q'],f['q']) and
                  e['closures'].item()==r['captures'][name]['closures'],'unbound references')
            nodes = (s['nodes_world']-f['cam_t'])@f['cam_R']
            sheet = Sheet(nodes,e['edges'])
            index = {tuple(key):k for k,key in enumerate(sheet.ids)}
            metric[h] = []
            for bank in support[h]:
                pixels = bank['pixels']
                rc = np.array([p['rc'] for p in pixels])
                exact,faces = sheet.hits(rays(rc,(H,W),F))
                check(np.array_equal(faces,[p['face'] for p in pixels]),'changed first hits')
                ids = np.array([index[tuple(face)] for face in faces])
                z = f['depth'][tuple(rc.T)].astype(float)*1000.
                metric[h].append(dict(native=stats(z-1000*exact),
                                 model=stats(z-forward(1000*exact,sheet.normal[ids],1000*sheet.offset[ids]))))
            truth = s['nodes_world'][10,11:13].mean(0)-s['tcp_world']
            for pred in optical[h]:
                row = dict(pred)
                if 'row' in row:
                    e_mm = 1000*(np.asarray(row['displacement'])-truth)@row['axis']
                    check(abs(e_mm-row['delta_mm']-refs[h,row['seed']]['optimism_mm'])<1e-8,'score delta not closed')
                    row['optimism_mm'] = e_mm
                rows.append(row)
    check(sum(b['model']['samples'] for banks in metric.values() for b in banks)==1102 and
          len(rows)==len(refs)==150,'missing archived population')
    metric_pass = all(b['model']['p95_abs_mm']<=.010 for banks in metric.values() for b in banks)
    strata = {(r['prefix'],r['closed']) for r in rows}
    coverage = all({(r['prefix'],r['closed']) for r in rows if r['split']==part and 'row' in r}==strata
                   for part in ('clean','development','check'))
    score_path = OUT/'transfer_measurements.json'
    write(score_path,rows)
    return dict(metric=metric,metric_passed=metric_pass,margin=margin_gate(rows,coverage),
                predictions_sha256=sha256(path),measurements_sha256=sha256(score_path))


def run():
    seal = OUT/'blind_decision.json'
    blind = load(seal)
    lock = load(checked(OUT/'registration.json',blind['registration_sha256']))
    for name,pin in lock['sources'].items():
        checked(ROOT/name,pin)
    checked(OUT/'capture.json',blind['capture_sha256'])
    ok = blind['depth_passed'] and blind['rgb_passed']
    check(ok==blind['transfer_eligible'],'inconsistent blind gates')
    result = transfer() if ok else None
    write(OUT/'decision.json',dict(blind_sha256=sha256(seal),transfer=result,authorizes_motion=False))
    print('archive transfer:',None if result is None else (result['metric_passed'],result['margin']),flush=True)


if __name__=='__main__':
    run()
