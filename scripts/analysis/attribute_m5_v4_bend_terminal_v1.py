"""Post-hoc clean main-path distance decomposition. No fit, threshold or permit."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/d) for d in ('task','vision')]
from m5_v3_perception_v11 import _solve_work
from run_m5_v4_bend_terminal_v1 import OUT, check, load, sha256, verify, write


def run():
    dec = load(OUT/'decision.json')
    pred = load(verify(OUT/'optical_predictions.json',dec['optical_sha256']))['predictions']
    draws = load(verify(OUT/'measurements.json',dec['measurements_sha256']))
    expected = {x['frame_sha256']:x['optimism_mm'] for x in draws if x['prefix']>=6 and x['split']=='clean'}
    rows = {}
    for case,obs in pred.items():
        folder = OUT/case
        r = load(verify(folder/'run.json',dec['run_hashes'][case]))
        for name,reads in obs['work'].items():
            m = r['captures'][name]
            h,n = m['sha256'],reads[0]['prefix']
            if n<6 or h in rows:
                continue
            sc = r['scoring_sidecars'][name]
            with (np.load(verify(folder/m['file'],h),allow_pickle=False) as f,
                  np.load(verify(folder/sc['file'],sc['sha256']),allow_pickle=False) as s):
                (g,row,pair,_),_,recovered = _solve_work(f['rgb'],f['depth'],focal=f['focal'].item(),
                    far=f['far'].item(),sigma=0.,prefix_mm=n,goal_mm=10 if n==9 else 9)
                check(row==10 and s['frame_sha256'].item()==h,'not a bound late sample')
                R,t = f['cam_R'],f['cam_t']
                d = R@g.direction
                side = int(np.argmin(pair@g.direction))
                # Bank print covers cells 7..15 inclusive, so its outer nodes are 7/16.
                outer = s['nodes_world'][row,(7,16)[side]]
                point = pair[side]
                true_z = (R.T@(outer-t))[2]
                same_ray = R@(point*true_z/point[2])+t
                tip = R@np.asarray(m['marker']['tip'])+t
                parts = dict(ray=same_ray-outer,depth=R@point+t-same_ray,
                             outer_to_center=outer-s['nodes_world'][row,11:13].mean(0),tip=s['tcp_world']-tip)
                mm = {k:float(1000*v@d) for k,v in parts.items()}
                check(abs(sum(mm.values())-expected[h])<1e-8,'decomposition does not close')
                rows[h] = dict(case=case,name=name,side=side,recovered=recovered,mm=mm,total_mm=expected[h])
    write(OUT/'clean_attribution.json',dict(decision_sha256=sha256(OUT/'decision.json'),
          script_sha256=sha256(Path(__file__)),rows=rows,post_hoc=True,authorizes_motion=False))
    for row in rows.values():
        print(row['case'],row['name'],{k:round(v,5) for k,v in row['mm'].items()})


if __name__=='__main__':
    run()
