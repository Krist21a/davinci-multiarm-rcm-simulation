"""Read RGB-D predictions first; compare bound A/B physics sidecars independently."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/d) for d in ('task','vision')]
from run_m5_v4_bend_pair_v1 import OUT, REF, PHASES, base, verify
from m5_v3_corridor_v5 import bank_patch, inks, measure
from m5_v3_perception_v11 import _solve_work


def file(folder,name,h):
    p = folder/name
    base.check(base.sha256(p)==h,'changed input: '+str(p))
    return p


def predict(folder,r):
    result = {}
    for key in PHASES:
        m = r['captures'][key]
        with np.load(file(folder,m['file'],m['sha256']),allow_pickle=False) as f:
            im,z,far = f['rgb'],f['depth'],f['far'].item()
            if key=='work':
                sol,sides,_ = _solve_work(im,z,focal=f['focal'].item(),far=far,sigma=0.,prefix_mm=0,goal_mm=9)
                result[key] = dict(chart=sol is not None,marker=m['marker']['valid'],
                                   stations=[sorted(side) for side in sides] if sides else [[],[]])
            else:
                decoded = inks(im)
                good = np.isfinite(z)&(z>0)&(z<far*.999)
                areas = [[len(bank_patch(good&(decoded==j+13*s))) for j in range(6,11)] for s in range(2)]
                rows = measure(im,z,far=far,sigma=0.)['rows']
                result[key] = dict(areas=areas,weakest=int(np.min(areas)),
                                   intact=[row['row'] for row in rows[6:11] if row['status']=='INTACT'])
    return result


def run():
    lock = base.load(OUT/'registration.json')
    verify(lock)
    names = lock['plan']['order']
    runs = {}
    for k in names:
        r = OUT/k/'run.json'
        if r.is_file():
            runs[k] = base.load(r)
            base.check(runs[k]['registration_sha256']==base.sha256(OUT/'registration.json')
                       and runs[k]['variant']==k,'unregistered capture')
    seals = {k:base.sha256(OUT/k/'run.json') for k in runs}
    if len(runs)!=2 or any(r['error'] or r['cleanup_error'] for r in runs.values()):
        errors = {k:(runs[k]['error'] or runs[k]['cleanup_error']) if k in runs else 'not run' for k in names}
        base.write(OUT/'decision.json',dict(passed=False,reason='incomplete acquisition',errors=errors,
                   run_hashes=seals,registration_sha256=base.sha256(OUT/'registration.json'),authorizes_cutting=False))
        return False
    pred = {k:predict(OUT/k,r) for k,r in runs.items()}
    pp = OUT/'optical_predictions.json'
    base.write(pp,dict(run_hashes=seals,predictions=pred))
    geom = base.load(base.GEOMETRY)
    normal = np.cross(geom['u'],geom['v'])
    cfg = base.load(base.BASE)
    checks = dict(paired=True,stock_identity=True,mechanics=True,truth_isolated=True)
    bends,mechanics,frames,common_edges = {},{},{},None
    old = base.load(REF/'run.json')
    for k,r in runs.items():
        bends[k],mechanics[k] = {},{}
        checks['truth_isolated'] &= 'untagged' not in r['truth_reads'] and r['truth_reads'].get('scoring',{}).get('nodes')==3
        for tick,phase in enumerate(PHASES,1):
            s,m = r['states'][phase],r['captures'][phase]
            with (np.load(file(OUT/k,m['file'],m['sha256']),allow_pickle=False) as f,
                  np.load(file(OUT/k,'scoring/'+phase+'.npz',s['side_sha256']),allow_pickle=False) as sc):
                nodes,edges = sc['nodes_world'],sc['edges']
                base.check(sc['frame_sha256'].item()==m['sha256'] and sc['tick'].item()==m['tick']==tick
                           and nodes.shape==(14,24,3) and np.isfinite(nodes).all() and np.array_equal(sc['q'],f['q']),
                           'unbound reference')
                base.DamageBaseline(edges)
                if common_edges is None:
                    common_edges = edges.copy()
                t = s['tissue']
                checks['mechanics'] &= (edges.size==3*1219 and np.array_equal(edges,common_edges)
                    and m['closed'] and s['closed'] and s['ready'] and np.array_equal(f['q'],geom['ready_q'])
                    and t['alive']==t['constraints']==1219 and t['cut']==t['torn']==t['seam_releases']==0
                    and cfg['tension']['low_strain']<=s['strain']<=cfg['tension']['high_strain']
                    and all(t['grasped'].get(arm,0)>=cfg['grasp']['min_nodes'] for arm in ('psm1','psm3'))
                    and all(v['failures']==0 and v['collision_blocks']==0 and v['max_drift_m']<=.0005 for v in s['rcm']))
                h = (nodes-np.array(geom['sheet_origin']))@normal*1000
                amp = (2*h[6:12,11:13]-h[5:11,11:13]-h[7:13,11:13])*.25
                bends[k][phase] = dict(rms_mm=float(np.sqrt(np.mean(amp**2))),
                                       column_rms_mm=np.sqrt(np.mean(amp**2,axis=0)).tolist())
                mechanics[k][phase] = dict(grasped=t['grasped'],strain=s['strain'],max_strain=t['max_strain'])
                fields = [sc['stamp'],f['q'],f['cam_t'],f['cam_R'],f['far'],f['focal'],f['depth'].shape]
                if k==names[0]:
                    frames[phase] = fields
                    if phase=='work':
                        with (np.load(REF/old['captures']['001_work']['file']) as ref,
                              np.load(REF/old['scoring_sidecars']['001_work']['file']) as rs):
                            checks['stock_identity'] &= (all(np.array_equal(f[n],ref[n]) for n in ref.files)
                                                         and np.array_equal(nodes,rs['nodes_world']))
                else:
                    checks['paired'] &= (s['targets']==runs[names[0]]['states'][phase]['targets']
                        and all(np.array_equal(x,y) for x,y in zip(fields,frames[phase],strict=True)))
    (a,b) = names
    x,y = (bends[k]['work']['rms_mm'] for k in names)
    pa,pb = (pred[k]['primary']['weakest'] for k in names)
    checks.update(fold_reduction=x>0 and y<=.75*x,
                  improved_primary_area=pb>=max(1.10*pa,pa+1),
                  retained_intact=all(set(pred[a][p]['intact'])<=set(pred[b][p]['intact']) for p in PHASES[1:]),
                  retained_work=all(pred[k]['work']['chart'] and pred[k]['work']['marker'] for k in names)
                  and all(set(x)<=set(y) for x,y in zip(pred[a]['work']['stations'],pred[b]['work']['stations'])))
    base.write(OUT/'decision.json',dict(passed=all(checks.values()),checks=checks,bending=bends,mechanics=mechanics,
               run_hashes=seals,predictions_sha256=base.sha256(pp),registration_sha256=base.sha256(OUT/'registration.json'),
               authorizes_cutting=False))
    print('checks',checks,'fold RMS mm',x,y,'primary weakest patch px',pa,pb)
    return all(checks.values())


if __name__=='__main__':
    sys.exit(0 if run() else 1)
