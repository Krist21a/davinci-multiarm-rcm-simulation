"""Same six qualification gates as v2, adding exact v1 equivalence checks before scoring."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/k) for k in ('task','vision')]
from m5_v4_hinge_marker_v3 import read, reader_for
from m5_v4_hinge_work_v3 import margin, observe, primary
from qualify_m5_v4_hinge_v2 import population, summarize
from run_m5_v4_hinge_closed_loop_v3 import OUT, SUITE, World, check, load, lock, sha256, v2, v7, verify, work_pose, write
from score_m5_v4_hinge_closed_loop_v1 import draw, model


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
            before,_ = reader._scissors(rgb.astype(float),zz)
            goal = 10 if n==9 else 9
            g,route = observe(rgb,zz,focal,far,s,n,goal)
            original = primary.chart(rgb,zz,focal,far,s,n,goal)
            marker_same = not before['valid'] or all(mk.get(k)==before[k] for k in
                          ('valid','tip','jaw_axis','opening_axis','fit_rms_m','original_rms_m'))
            chart_same = original is None or (route=='primary' and g is not None and
                         all(np.array_equal(getattr(original,k),getattr(g,k)) for k in vars(original)))
            d = dict(group=group,name=name,case=r['case'],split=split,seed=seed,sigma=s,prefix=n,
                     closed=m['closed'],tick=m['tick'],h=m['sha256'],mark=mk,old_marker=before,chart=g is not None,
                     original_chart=original is not None,route=route,marker_same=marker_same,chart_same=chart_same)
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


def run():
    reg = lock()
    tests = load(SUITE)
    check(tests['success'] and not tests['skipped'] and tests['tests_run']>=513,'offline regressions required')
    for k,h in reg['sources'].items():
        if k in tests['source_sha256']:
            check(tests['source_sha256'][k]==h,'regression source drift: '+k)
    fk = model()
    pop = list(population())
    pp = OUT/'archive_predictions.json'
    write(pp,optical(pop,fk))
    refs = {(group,name,r['captures'][name]['sha256']):(folder,r) for folder,name,r,_,group in pop}
    rows = load(pp)
    key = lambda r:tuple(r[k] for k in ('h','group','name','case','split','seed','prefix','closed','tick'))
    historical = load(v2.OUT/'archive_predictions.json')
    check(list(map(key,rows))==list(map(key,historical)),'population differs from v2')
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
    preservation = all(d['marker_same'] and d['chart_same'] and (d['prefix']<6 or d['route']=='primary') for d in rows)
    report = dict(passed=all(gates.values()) and preservation,gates=gates,parts=parts,controls=failed_frames,
                  original_estimates_preserved=preservation,suite_sha256=sha256(SUITE),
                  availability_98p6=all(v['valid']/v['total']>=.986 for v in parts.values()),
                  max_tip_mm=max(tips),median_native_nominal_tip_mm=np.median(native),
                  registration_sha256=sha256(OUT/'registration.json'),predictions_sha256=sha256(pp),
                  measurements_sha256=sha256(path))
    write(OUT/'offline_gate.json',report)
    print('offline v3:',report['passed'],gates,parts,flush=True)
    return report['passed']


if __name__=='__main__':
    sys.exit(0 if run() else 1)
