"""Post-decision 2x2 archive comparison. It cannot select a controller or authorize a run."""
import numpy as np

from qualify_m5_v4_hinge_v2 import OUT, check, draw, load, lock, population, sha256, verify, write
from m5_v4_hinge_work_v1 import chart as quad_chart
from m5_v4_hinge_work_v2 import chart as plane_chart


def run():
    lock()
    decision = load(OUT/'offline_gate.json')
    check(not decision['passed'],'only diagnose a sealed failure')
    rows = load(verify(OUT/'archive_measurements.json',decision['measurements_sha256']))
    ref = {(d['group'],d['h'],d['seed']):d for d in rows if d['group'] in ('calibration','new_late')}
    measured = []
    for folder,name,r,draws,group in population():
        if group not in ('calibration','new_late'):
            continue
        m,sf = r['captures'][name],r['scoring_sidecars'][name]
        with np.load(verify(folder/m['file'],m['sha256'])) as f, np.load(verify(folder/sf['file'],sf['sha256'])) as side:
            check(m['sha256']==side['frame_sha256'].item() and np.array_equal(f['q'],side['q']),'sidecar mismatch')
            z,rgb,R,t,far,focal = (f[k] for k in ('depth','rgb','cam_R','cam_t','far','focal'))
            truth = side['nodes_world'][10,11:13].mean(0)-side['tcp_world']
        for split,seed,n in draws:
            d = ref[group,m['sha256'],seed]
            zz,s = draw(z,seed,m['tick'])
            vals = {}
            for work,fn in (('quad',quad_chart),('plane',plane_chart)):
                g = fn(rgb,zz,focal.item(),far.item(),s,n,10 if n==9 else 9)
                for marker,key in (('v1','old_tip'),('v2','tip')):
                    err = None if g is None or key not in d else float(1000*(R@g.limit+t-d[key]-truth)@(R@g.direction))
                    vals[work+'_'+marker] = err
            check((vals['plane_v2'] is None)==('optimism_mm' not in d) and (vals['plane_v2'] is None or
                  abs(vals['plane_v2']-d['optimism_mm'])<1e-9),'failed to reproduce frozen decision')
            measured.append(dict(group=group,case=r['case'],name=name,split=split,seed=seed,values=vals))
    summary = {}
    for group,split in sorted({(r['group'],r['split']) for r in measured}):
        take = [r for r in measured if (r['group'],r['split'])==(group,split)]
        part = {}
        for key in ('quad_v1','quad_v2','plane_v1','plane_v2'):
            vals = [r['values'][key] for r in take if r['values'][key] is not None]
            part[key] = dict(total=len(take),valid=len(vals),maximum_mm=max(vals,default=None),
                             over_bound=sum(e>(.07 if split=='clean' else .22) for e in vals))
        summary[group+'/'+split] = part
    check(len(measured)==len(ref)==189,'missing late population')
    write(OUT/'posthoc_factorial.json',dict(offline_gate_sha256=sha256(OUT/'offline_gate.json'),
          source_sha256=sha256(__file__),rows=measured,summary=summary,authorizes_motion=False))
    for k,v in summary.items():
        print(k,[(key,s['valid'],round(s['maximum_mm'],5),s['over_bound']) for key,s in v.items()])


if __name__=='__main__':
    run()
