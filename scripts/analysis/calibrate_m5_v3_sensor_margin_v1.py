"""Sealed offline audit of sweep-distance error terms, not a motion calibration."""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from audit_m5_v3_optics_attribution_v1 import ROOT, digest, load, verified
from cell_certificate_v1 import decode
from cut_perception import CutPerception
from m5_v3_perception_v4 import work
from m5_v3_rigid_fk import CutterFK
from replay_cell_v2 import write


PLAN = ROOT/'config/task/m5_v3_sensor_margin_calibration_v1.json'
DATA = ROOT/'validation/task'
READ = DATA/'m5_v3_readonly_v4'
OUTER = DATA/'m5_v3_positive_outer_v2'
CAL = DATA/'m5_v3_rigid_fk_v1.json'


def register(spec):
    sources = {}

    def check(path, pin):
        sources[path.relative_to(ROOT).as_posix()] = digest(verified(path,pin))
        return path

    for name,pin in spec['sources'].items():
        check(ROOT/name,pin)
    ro,outer = load(READ/'decision.json'),load(OUTER/'decision.json')
    check(READ/'registration.json',ro['registration_sha256'])
    case = outer['cases']['nominal']
    raw = load(check(OUTER/'nominal/run.json',case['run_sha256']))
    check(OUTER/'nominal/registration.json',case['registration_sha256'])
    if raw['registration_sha256']!=case['registration_sha256'] or ro['failure'] or ro['cleanup_error']:
        raise ValueError('source run provenance mismatch')
    manifest = []
    for case_id,folder,report,receipts in ((0,READ,ro,[]),(1,OUTER/'nominal',raw,case['receipts'])):
        for m in sorted(report['captures'].values(),key=lambda e:e['tick']):
            if m['fov_deg']!=70:
                continue
            f = check(folder/m['file'],m['sha256'])
            prior = [r['prefix_mm'] for r in receipts if r['tick']<m['tick'] and r['positive']]
            prefix = prior[-1] if prior else 0
            manifest.append(dict(path=f.relative_to(ROOT).as_posix(),case_id=case_id,
                                 prefix_mm=prefix,tick=m['tick'],closed=m['closed'],marker=m['marker']))
    if len(manifest)!=15 or {m['prefix_mm'] for m in manifest}!={0,3,5,7}:
        raise ValueError('missing registered work images')
    for p in (PLAN,Path(__file__).resolve(),*(ROOT/n for n in (
            'scripts/task/m5_v3_rigid_fk.py','scripts/task/capture_m5_v3_readonly_v4.py',
            'scripts/vision/cell_certificate_v1.py','scripts/vision/cut_perception.py',
            'scripts/vision/m5_v3_perception_v4.py','scripts/vision/ruler_readout.py'))):
        sources[p.relative_to(ROOT).as_posix()] = digest(p)
    return dict(plan=spec,sources=sources,frames=manifest)


def support(rgb,z,far):
    caps,faces = decode(rgb)
    good = np.isfinite(z)&(z>0)&(z<far*.999)
    before,after = [(faces==c)&good for c in (19,20)]
    # Adjacent face samples indicate possible sight of the transverse edge,
    # but without both endpoint locations they do not locate its midpoint.
    return dict(row10_caps_px=[int(((caps==c)&good).sum()) for c in (20,21)],
                transition_faces_px=[int(a.sum()) for a in (before,after)],
                after_px_touching_before=int((after&ndimage.binary_dilation(before)).sum()))


def fit(estimator, rgb, rgb_float, z, focal, far, n, sigma):
    chart = work(rgb,z,focal=focal,far=far,sigma=sigma,prefix_mm=n,goal_mm=10)
    marker,_ = estimator._scissors(rgb_float,z)
    return chart,marker


def one_frame(rec, spec, model):
    with np.load(ROOT/rec['path'],allow_pickle=False) as f:
        rgb,z = f['rgb'],f['depth']
        R,t,q = (f[k] for k in ('cam_R','cam_t','q'))
        focal,far = (f[k].item() for k in ('focal','far'))
        has_midpoint_label = any(k in f.files for k in ('nodes','nodes_world','tissue_nodes','row10_ref'))
    if z.shape!=(960,1280) or not np.isclose(focal,640/np.tan(np.deg2rad(35)),rtol=0,atol=1e-6):
        raise ValueError('unexpected work optics')
    # Do not quietly treat a differently registered sidecar as calibrated truth.
    if has_midpoint_label:
        raise ValueError('unexpected reference payload; reference provenance not registered')
    estimator = CutPerception(1280,960,focal,far,model.rcm,tcp_local=model.tip_Y,
                              jaw_axis_local=model.tip_R[:,2],opening_axis_local=model.tip_R[:,0],
                              ray_marker_fit=True)
    rgb_float = rgb.astype(float)
    tip_fk = R.T@(model.pose(q)[:,3]-t)
    n = rec['prefix_mm']
    project = lambda xy,d: float(1000*(xy@d))
    g,m = fit(estimator,rgb,rgb_float,z,focal,far,n,0.)
    saved = rec['marker']
    if bool(m['valid'])!=bool(saved['valid']) or (m['valid'] and any(
            not np.allclose(m[k],saved[k],rtol=0,atol=1e-12) for k in ('tip','jaw_axis','opening_axis'))):
        raise ValueError('clean ray-marker replay differs from sealed fit: '+rec['path'])
    diagnostics = dict(**support(rgb,z,far),same_state_row10_reference=False)
    base = dict(chart_valid=g is not None,marker_valid=m['valid'])
    base_delta = None
    if g is not None and m['valid']:
        tip = np.asarray(m['tip'])
        base_delta = g.limit-tip
        base.update(visual_room_mm=project(base_delta,g.direction),
                    tip_optimism_mm=project(tip_fk-tip,g.direction),
                    tip_error_norm_mm=float(1000*np.linalg.norm(tip_fk-tip)))
    samples = []
    for part in ('development','check'):
        for seed in spec[part+'_seeds']:
            rng = np.random.default_rng(np.random.SeedSequence([seed,rec['case_id'],rec['tick']]))
            sigma = spec['depth_sigmas_m'][1]
            zz = z+rng.normal(0.,sigma,z.shape)
            h,mark = fit(estimator,rgb,rgb_float,zz,focal,far,n,sigma)
            row = dict(part=part,seed=seed,chart_valid=h is not None,marker_valid=mark['valid'])
            if h is not None and mark['valid']:
                tip = np.asarray(mark['tip'])
                noisy_delta = h.limit-tip
                room = project(noisy_delta,h.direction)
                row.update(visual_room_mm=room,tip_optimism_mm=project(tip_fk-tip,h.direction))
                if base_delta is not None:
                    row.update(delta_visual_room_mm=room-base['visual_room_mm'],
                               delta_on_raw_axis_mm=project(noisy_delta-base_delta,g.direction),
                               axis_delta_deg=float(np.rad2deg(np.arccos(np.clip(h.direction@g.direction,-1,1)))))
            elif not mark['valid']:
                row['marker_reason'] = mark['reason']
            samples.append(row)
    return dict(**{k:v for k,v in rec.items() if k!='marker'},reference=diagnostics,raw=base,noisy=samples)


def maxima(rows, field):
    values = [r[field] for r in rows if field in r]
    return dict(samples=len(values),max_mm=max(values,default=None),min_mm=min(values,default=None))


def score(rows):
    out = {}
    for key,predicate in (('conservative_early',lambda n:n<=5),('direct_row10',lambda n:n>=6)):
        sel = [r for r in rows if predicate(r['prefix_mm'])]
        clean = [r['raw'] for r in sel]
        row = dict(frames=len(sel),open_frames=sum(not r['closed'] for r in sel),
                   same_state_row10_references=sum(r['reference']['same_state_row10_reference'] for r in sel),
                   zero_sigma_tip=maxima(clean,'tip_optimism_mm'),
                   zero_sigma_tip_error_norm=maxima(clean,'tip_error_norm_mm'))
        for part in ('development','check'):
            noisy = [s for r in sel for s in r['noisy'] if s['part']==part]
            row[part] = dict(draws=len(noisy),chart_valid=sum(s['chart_valid'] for s in noisy),
                             marker_valid=sum(s['marker_valid'] for s in noisy),
                             paired_valid=sum(s['chart_valid'] and s['marker_valid'] for s in noisy),
                             delta_visual_room=maxima(noisy,'delta_visual_room_mm'),
                             delta_on_raw_axis=maxima(noisy,'delta_on_raw_axis_mm'),
                             tip_optimism=maxima(noisy,'tip_optimism_mm'))
        out[key] = row
    return out


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    spec = load(PLAN)
    reg = register(spec)
    args.out.mkdir(exist_ok=False)
    seal = args.out/'registration.json'
    write(seal,reg)
    data = load(CAL)
    if not data['qualified']:
        raise ValueError('FK diagnosis requires qualified joint calibration')
    model = CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in data['calibration'].items()})
    rows = []
    for rec in reg['frames']:
        rows.append(one_frame(rec,spec,model))
        print('measured',rec['path'],flush=True)
    measurements = args.out/'measurements.json'
    write(measurements,rows)
    summary = score(load(measurements))
    result = dict(qualified=False,authorizes_motion=False,sensor_margin=None,
                  reason='no independently bounded same-state loaded row-10 midpoint in sealed work data',
                  legacy_margin_m=.00025,summary=summary,registration_sha256=digest(seal),
                  measurements_sha256=digest(measurements))
    write(args.out/'decision.json',result)
    print(summary,flush=True)
