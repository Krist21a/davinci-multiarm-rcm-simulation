"""Attribute the slip entry veto using the exact frozen estimator inputs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from replay_cell_v2 import ROOT, digest, load, write
import cut_perception as cp
import score_m5_guided_disturbance_dry_diagnostic_v1 as prior_score
from tool_pose import _MARGIN

ID = 'm5_guided_short_gap_attribution_v1'
SPEC = ROOT/f'config/task/{ID}.json'
PRIOR = 'm5_guided_disturbance_dry_diagnostic_v1'
REF = ROOT/'validation/task'/PRIOR


def snapshot(path):
    with np.load(path,allow_pickle=False) as f:
        data = {k:f[k] for k in ('rgb','depth','R','t')}
        state = json.loads(f['state'].item())
        for k in f['array_keys'].tolist():
            state[k] = np.asarray(state[k],dtype=float)
        data.update(state=state,post_grasp=f['post_grasp'].item())
    return data


def replay(snap,rgb=None,depth=None):
    if rgb is None:
        rgb = snap['rgb']
    if depth is None:
        depth = snap['depth']
    estimator = object.__new__(cp.CutPerception)
    estimator.__dict__.update(snap['state'])
    original = cp._provable_ahead
    windows = []

    def audit(s,band,occ,b_start,edges,sa_px,mask,depth,pz,far):
        original_result = original(s,band,occ,b_start,edges,sa_px,mask,depth,pz,far)
        if sys._getframe(1).f_code.co_name=='provable':
            row = prior_score.check_window(s,band,occ,b_start,edges,sa_px,mask,depth,pz,far)
            if original_result!=all(row['checks'].values()):
                raise ValueError('window audit disagrees with frozen predicate')
            bins = np.digitize(sa_px,edges)-1
            behind = band&~mask&((depth>pz+cp.GAP_DEPTH_M)|~np.isfinite(depth)|(depth>=far*.999))
            windows.append(dict(row=row,bins=bins,behind=behind,mask=mask,band=band,pz=pz,far=far,edges=edges))
        return original_result

    try:
        cp._provable_ahead = audit
        r = estimator.estimate(rgb,depth,snap['R'],snap['t'],post_grasp=snap['post_grasp'])
    finally:
        cp._provable_ahead = original
    cut,tool = r['cut'],r['scissors']
    chart = cut.get('material_chart') or {}
    estimate = dict(front_valid=bool(cut.get('front_valid')),reason=cut.get('reason'),mode=cut.get('mode'),
                    slit_px=cut.get('slit_px'),progress_m=cut.get('progress_m'),chart_valid=bool(chart.get('valid')),
                    chart_reason=chart.get('reason'),scissors_valid=bool(tool.get('valid')),
                    scissors_reason=tool.get('reason'))
    return estimate,windows


def ray(frame,scan,v,u):
    z,pz = float(frame['depth'][v,u]),float(scan['pz'][v,u])
    sheet,band = bool(scan['mask'][v,u]),bool(scan['band'][v,u])
    idx = int(scan['bins'][v,u])
    return dict(rgb=frame['rgb'][v,u].tolist(),measured_mm=z*1000,plane_mm=pz*1000,
                behind_plane_mm=(z-pz)*1000,finite=bool(np.isfinite(z)),far_background=bool(z>=scan['far']*.999),
                sheet_mask=sheet,in_scan_band=band,bin=idx,counts_as_gap_pixel=bool(scan['behind'][v,u]))


def native_ik_slack(dry):
    rows = []
    for case,rec in dry.items():
        for i,e in enumerate(rec['targets']):
            shrunk_lo = np.asarray(e['lo'])+_MARGIN
            shrunk_hi = np.asarray(e['hi'])-_MARGIN
            for v in [dict(name='native',result=e['native']),*e['variants']]:
                r = v['result']
                if r is None:
                    rows.append(dict(case=case,target=i+1,variant=v['name'],result='unavailable'))
                    continue
                q = r.get('q')
                yaw_margin = None if q is None else float(min(q[0]-shrunk_lo[0],shrunk_hi[0]-q[0]))
                rows.append(dict(case=case,tick=rec['tick'],target=i+1,variant=v['name'],
                                 original_solver_pass=r['converged'],at_limit=r['at_limit'],
                                 position_slack_um=(.0002-r['pos_err_m'])*1e6,
                                 jaw_axis_slack_rad=.02-r['axis_err_rad'],
                                 hinge_axis_slack_rad=.02-r['yaw_axis_err_rad'],
                                 yaw_slack_deg=yaw_margin))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True)
    args = p.parse_args()
    plan = load(SPEC)
    locked,decision = REF/'registration.json',REF/'diagnosis.json'
    orig,diag = load(locked),load(decision)
    if (digest(locked)!=plan['prior_registration_sha256'] or digest(decision)!=plan['prior_diagnosis_sha256']
            or not diag['matched_trajectories'] or diag['registration_sha256']!=digest(locked)
            or digest(Path(prior_score.__file__))!=diag['scorer_sha256']
            or any(digest(ROOT/k)!=v for k,v in orig['sources'].items())):
        raise ValueError('unregistered source')
    paths = {n:args.frames/(PRIOR+'_slip_regrasp')/f'{n:03d}_estimator_input.npz' for n in plan['ticks']}
    source_rows = {r['tick']:r for r in diag['optical_windows']}
    if (set(paths)!=set(source_rows)
            or any(digest(p)!=source_rows[n]['matched_estimator_input_sha256'] for n,p in paths.items())):
        raise ValueError('archived estimator input changed')
    args.out.mkdir(exist_ok=False)
    registration = dict(plan=plan,plan_sha256=digest(SPEC),
                        code_sha256=digest(Path(__file__)),sources=orig['sources'],
                        prior_scorer_sha256=diag['scorer_sha256'],
                        frames={str(n):digest(f) for n,f in paths.items()})
    write(args.out/'registration.json',registration)
    snaps,scans,frames = {},{},[]
    for n,path in paths.items():
        snaps[n] = snapshot(path)
        result,windows = replay(snaps[n])
        src = source_rows[n]
        previous_window = [w for w in src['windows'] if w['call_site']=='provable']
        if (result!=src['original_estimate'] or len(windows)!=1 or len(previous_window)!=1
                or windows[0]['row']!={k:v for k,v in previous_window[0].items() if k not in ('call_site','passed')}):
            raise ValueError('original pre-cut evidence not reproduced')
        scans[n] = windows[0]
        frames.append(dict(tick=n,estimate=result,window=windows[0]['row']))
    a,b,c = (plan[k] for k in ('pre_veto_tick','veto_tick','confirmation_tick'))
    if not all(np.array_equal(snaps[a][k],snaps[b][k]) for k in ('R','t')):
        raise ValueError('paired camera pose changed')
    if any(snaps[a]['state'][k]!=snaps[b]['state'][k] for k in ('W','H','f','far')):
        raise ValueError('paired camera projection changed')
    scan = scans[b]
    veto = scan['row']['occupied_gap_bins']
    v,u = np.nonzero(scan['behind'] & np.isin(scan['bins'],veto))
    if not veto or len(u)!=source_rows[b]['original_estimate']['slit_px']:
        raise ValueError('insufficient coverage of occupied look-ahead bins')
    rays = []
    for y,x in zip(v,u,strict=True):
        before,after,confirm = (ray(snaps[n],scans[n],y,x) for n in (a,b,c))
        zm0,zm1,pz0,pz1 = (r[k] for r,k in (
            (before,'measured_mm'),(after,'measured_mm'),(before,'plane_mm'),(after,'plane_mm')))
        rays.append(dict(uv=[int(x),int(y)],before=before,veto=after,confirmation=confirm,
                         measured_change_um=(zm1-zm0)*1000,plane_change_um=(pz1-pz0)*1000,
                         prior_depth_minus_current_plane_mm=zm0-pz1,
                         current_depth_minus_prior_plane_mm=zm1-pz0))
    controls = []
    for label,im,z in (('prior_rgb',snaps[a]['rgb'],None),('prior_depth',None,snaps[a]['depth']),
                       ('prior_rgb_and_depth',snaps[a]['rgb'],snaps[a]['depth'])):
        result,windows = replay(snaps[b],rgb=im,depth=z)
        controls.append(dict(input_swap=label,estimate=result,windows=[w['row'] for w in windows]))
    result = dict(registration_sha256=digest(args.out/'registration.json'),
                  native_frames=frames,veto_rays=rays,offline_input_swaps=controls,
                  rgb_pixels_changed=int(np.any(snaps[a]['rgb']!=snaps[b]['rgb'],axis=2).sum()),
                  depth_pixels_changed=int((snaps[a]['depth']!=snaps[b]['depth']).sum()),
                  ik_slack=native_ik_slack(diag['dry_ik']),
                  authorizes_motion=False,physical_identity_established=False)
    write(args.out/'diagnosis.json',result)
    print('Veto rays:',len(rays),'offline swaps:',[(x['input_swap'],x['estimate']['front_valid']) for x in controls])


if __name__=='__main__':
    main()
