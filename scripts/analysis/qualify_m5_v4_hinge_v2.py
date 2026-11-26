"""One-shot optical replay of both v2 changes, then join SHA/tick-bound truth for scoring."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision')]
from m5_v4_hinge_marker_v2 import read, reader_for
from m5_v4_hinge_work_v2 import chart, margin
from run_m5_v4_hinge_closed_loop_v2 import OUT, World, check, load, lock, old, sha256, term, v7, verify, work_pose, write
from score_m5_v4_hinge_closed_loop_v1 import draw, model


def population():
    opt = load(old.INV/'optical_predictions.json')
    prior = load(term.OUT/'decision.json')
    cases = {case:load(verify(term.OUT/case/'run.json',pin)) for case,pin in prior['run_hashes'].items()}
    for h,draws in opt.items():
        case,name = next((k,n) for k,r in cases.items() for n,m in r['captures'].items() if m['sha256']==h)
        yield term.OUT/case,name,cases[case],[(d['split'],d['seed'],d['prefix']) for d in draws],'calibration'
    for case in ('nominal','depth_noise_0p5','relax','slip_regrasp'):
        folder = old.OUT/case
        d = load(folder/'decision.json')
        r = load(verify(folder/'run.json',d['run_sha256']))
        for name,m in r['captures'].items():
            if 'work_prefix' not in m:
                continue
            selected = {'relax':'018_work','slip_regrasp':'013_work'}.get(case)
            if selected and name!=selected:
                continue
            n = m['work_prefix']
            yield folder,name,r,[('actual',1 if m['sigma'] else None,n)],case
            if case=='nominal' and name in d['new_nominal_frames']:
                yield folder,name,r,[('clean',None,n)]+[('check',s,n) for s in load(term.NOISE)['check_seeds']],'new_late'


def legacy(reader,rgb,z):
    fits = []
    def record(*a,**kw):
        opt = least_squares(*a,**kw)
        fits.append(dict(nfev=opt.nfev,status=opt.status,rank=int(np.linalg.matrix_rank(opt.jac))))
        return opt
    with patch('scipy.optimize.least_squares',record):
        mk,_ = reader._scissors(rgb.astype(float),z)
    return dict(valid=mk['valid'],fits=fits,fit_rms_m=mk.get('fit_rms_m'),tip=mk.get('tip'))


def optical(pop,fk):
    output = []
    for folder,name,r,draws,group in pop:
        m = r['captures'][name]
        with np.load(verify(folder/m['file'],m['sha256'])) as f:
            rgb,z,focal,far,R,t = (f[k] for k in ('rgb','depth','focal','far','cam_R','cam_t'))
            focal,far = focal.item(),far.item()
        reader = reader_for(fk,focal,far)
        for split,seed,n in draws:
            zz,s = draw(z,seed,m['tick'])
            mk = read(reader,rgb,zz)
            before = legacy(reader,rgb,zz)
            g = chart(rgb,zz,focal,far,s,n,10 if n==9 else 9)
            d = dict(group=group,name=name,case=r['case'],split=split,seed=seed,sigma=s,prefix=n,
                     closed=m['closed'],tick=m['tick'],h=m['sha256'],mark=mk,old_marker=before,chart=g is not None)
            if mk['valid']:
                d['tip'] = (R@mk['tip']+t).tolist()
            if before['valid']:
                d['old_tip'] = (R@before['tip']+t).tolist()
            if g is not None and mk['valid']:
                frame = SimpleNamespace(R=R,t=t,tip=np.array(d['tip']),jaw=R@mk['jaw_axis'],hinge=R@mk['opening_axis'],
                                        tick=m['tick'])
                p = work_pose(frame,g,0)
                d.update(axis=p.chart.direction.tolist(),displacement=(p.chart.limit-p.tip).tolist())
                if group=='depth_noise_0p5' and name=='005_work':
                    w = World(None,9,fk,sigma=s)
                    w.tick = p.tick
                    d.update(aligned=bool(v7.aligned(p,.0015)),sweep_room_mm=1000*w.sweep_room(p))
            if split=='actual':
                d['old_pair'] = m['work_estimate'] is not None
            output.append(d)
    return output


def summarize(rows):
    ans = {}
    for (group,split),size in {( 'calibration','clean'):6,('calibration','development'):72,
                              ('calibration','check'):72,('new_late','clean'):3,('new_late','check'):36}.items():
        pop = [d for d in rows if (d['group'],d['split'])==(group,split)]
        valid = [d for d in pop if 'optimism_mm' in d]
        check(len(pop)==size and len({(d['h'],d['seed']) for d in pop})==size,'changed replay population')
        bound = 1000*margin(pop[0]['sigma'])
        max_e = max((d['optimism_mm'] for d in valid),default=None)
        coverage = {(d['prefix'],d['closed']) for d in pop}=={(d['prefix'],d['closed']) for d in valid}
        ans[group+'/'+split] = dict(total=size,valid=len(valid),max_optimism_mm=max_e,bound_mm=bound,
            over_bound=sum(d['optimism_mm']>bound for d in valid),passed=coverage and len(valid)>=.95*size and
            max_e is not None and max_e<=bound)
    return ans


def run():
    lock()
    fk = model()
    pop = list(population())
    pp = OUT/'archive_predictions.json'
    write(pp,optical(pop,fk))
    refs = {(group,name,r['captures'][name]['sha256']):(folder,r) for folder,name,r,_,group in pop}
    rows = load(pp)
    for (group,name,h),(folder,r) in refs.items():
        side,m = r['scoring_sidecars'][name],r['captures'][name]
        with np.load(verify(folder/side['file'],side['sha256'])) as f, np.load(verify(folder/m['file'],h)) as img:
            check(h==f['frame_sha256'].item() and m['tick']==f['tick'].item() and m['closed']==f['closed'].item()
                  and np.array_equal(f['q'],img['q']),'unbound scorer reference')
            tcp,nodes = f['tcp_world'],f['nodes_world']
            seam = nodes[:,11:13].mean(1)
            direction = (np.arange(13)-6)@seam[:13]
            direction /= np.linalg.norm(direction)
            fk_err = 1000*np.linalg.norm(fk.pose(f['q'])[:,3]-tcp)
            for row in rows:
                if (row['group'],row['name'],row['h'])!=(group,name,h):
                    continue
                row['fk_mm'] = fk_err
                for tip in ('tip','old_tip'):
                    if tip in row:
                        row[tip+'_error_mm'] = 1000*np.linalg.norm(row[tip]-tcp)
                if 'axis' in row:
                    row['axis_deg'] = float(np.degrees(np.arccos(np.clip(direction@row['axis'],-1,1))))
                    if row['prefix']>=6:
                        row['optimism_mm'] = float(1000*(np.asarray(row['displacement'])-(seam[10]-tcp))@row['axis'])
    parts = summarize(rows)
    failed_frames = [d for d in rows if d['group'] in ('relax','slip_regrasp') or
                     (d['group']=='depth_noise_0p5' and d['name']=='005_work')]
    tips = [d['tip_error_mm'] for d in rows if 'tip_error_mm' in d]
    native = [d['tip_error_mm'] for d in rows if d['group']=='nominal' and 'tip_error_mm' in d]
    gates = dict(margin=all(v['passed'] for v in parts.values()),fk=max(d['fk_mm'] for d in rows)<=.010,
                 late_axis=max(d['axis_deg'] for d in rows if 'optimism_mm' in d)<5.,
                 failed_controls=len(failed_frames)==3 and all('axis' in d and (d['group']!='depth_noise_0p5'
                     or (d['aligned'] and d['sweep_room_mm']>=-1e-9)) for d in failed_frames),
                 actual_coverage=all(not d.get('old_pair') or 'axis' in d for d in rows),
                 tip_accuracy=max(tips)<2. and np.median(native)<.5)
    path = OUT/'archive_measurements.json'
    write(path,rows)
    report = dict(passed=all(gates.values()),gates=gates,parts=parts,controls=failed_frames,
                  max_tip_mm=max(tips),median_native_nominal_tip_mm=np.median(native),
                  registration_sha256=sha256(OUT/'registration.json'),predictions_sha256=sha256(pp),
                  measurements_sha256=sha256(path))
    write(OUT/'offline_gate.json',report)
    print('offline v2:',gates,parts,flush=True)
    return report['passed']


if __name__=='__main__':
    sys.exit(0 if run() else 1)
