"""M5a offline perception evaluation on recorded shadow-mode frames.

Replays CutPerception over validation/perception/frames/<dataset>/tick_*.npz (written by
SimWorld._dump_frame) and scores it against the pre-registered protocol
config/perception/m5a_gates.json. The estimator only sees rgb, depth and the (possibly
perturbed) camera pose, exactly as in SimWorld._perception_tick; the truth fields of the
dump are used for scoring only.

Degradations are re-applied offline with the same formulas as SimWorld._perception_tick
(seeded per dataset/seed), so a noise sweep needs no simulator.

Usage: replay_perception.py [--out validation/perception/m5a_eval.json] [--datasets nominal ...]
       [--degradations none ...] [--label baseline]
"""
import argparse
import hashlib
import io
import json
import math
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'vision'))
sys.path.insert(0, str(ROOT / 'scripts' / 'task'))
import cut_perception as cp  # noqa: E402
from visibility import depth_visible  # noqa: E402

FRAMES = ROOT / 'validation' / 'perception' / 'frames'
GATES = ROOT / 'config' / 'perception' / 'm5a_gates.json'
AMENDMENT = GATES.with_name('m5a_amendment_v5.json')
V6 = GATES.with_name('m5a_amendment_v6.json')
SCENE = ROOT / 'scenes' / 'davinci_si_articulated_verified.ttt'


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def load_dataset(name, capture_spec=None):
    d = FRAMES / name
    meta_bytes = (d / 'meta.json').read_bytes()
    meta = json.loads(meta_bytes)
    files = sorted(d.glob('tick_*.npz'), key=lambda f: int(f.stem.split('_')[-1]))
    h = hashlib.sha256()
    frames = []
    for fn in files:
        raw = fn.read_bytes()
        h.update(raw)
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            if int(z['tick']) != int(fn.stem.split('_')[-1]):
                raise ValueError(f'{fn.name}: stored tick does not match filename')
            frames.append({k: z[k] for k in z.files})
    ticks = [int(fr['tick']) for fr in frames]
    if not ticks or ticks != list(range(1, len(ticks) + 1)):
        raise ValueError('missing frames or non-contiguous ticks')
    sources = dict(frames=len(frames), sha256=h.hexdigest(),
                   meta_sha256=hashlib.sha256(meta_bytes).hexdigest(), run_name=meta.get('run_name'),
                   scene_sha256=meta.get('scene_sha256'))
    manifest = d / 'capture.json'
    if capture_spec is not None or meta.get('run_id') is not None:
        capture_bytes = manifest.read_bytes()
        capture = json.loads(capture_bytes)
        if not meta.get('run_id') or capture.get('run_id') != meta['run_id'] or capture.get('frame_count') != len(frames):
            raise ValueError('capture manifest does not match metadata and frame count')
        sources.update(run_id=meta['run_id'], capture_sha256=hashlib.sha256(capture_bytes).hexdigest())
        if capture_spec is not None:
            verify_initial_view(meta, capture, frames, capture_spec)
            sources['initial_view_readback'] = capture['initial_view_readback']
    return meta, frames, sources


def verify_registered_capture(meta, spec):
    perc = meta.get('perception_cfg') or {}
    pose_error = perc.get('camera_pose_error') or {}
    if (meta.get('scenario') != spec['scenario'] or meta.get('initial_view') != spec.get('initial_view')
            or meta.get('initial_view_insertion_m') is not None
            or meta.get('mode') != 'full' or meta.get('align_mode') != spec.get('align_mode', 'tool_pose')
            or meta.get('control_source') != 'truth' or perc.get('mode') != 'shadow'
            or not perc.get('markers', True) or perc.get('depth_noise_m', 0)
            or pose_error.get('rot_deg', 0) or pose_error.get('trans_m', 0)):
        raise ValueError('capture parameters differ from the registered run')
    with SCENE.open('rb') as stream:
        if meta.get('scene_sha256') != hashlib.file_digest(stream, 'sha256').hexdigest():
            raise ValueError('capture scene differs from the registered scene')


def verify_initial_view(meta, capture, frames, spec):
    verify_registered_capture(meta, spec)
    expected = spec['initial_view']
    view = capture.get('initial_view_readback') or {}
    home_R, applied_R = (np.asarray(view.get(k), dtype=float) for k in ('home_R', 'applied_R'))
    if (home_R.shape != (3, 3) or applied_R.shape != (3, 3)
            or not np.isfinite([home_R, applied_R]).all()
            or np.allclose(home_R, applied_R, atol=1e-8, rtol=0)):
        raise ValueError('missing or unchanged post-home camera rotation')
    sheet = [fr for fr in frames if 'nodes' in fr]
    if not sheet or len(sheet) == len(frames):
        raise ValueError('need both pre-sheet and sheet frames')

    def target_matches(target):
        return len(target) == 3 and np.allclose(target[:2], expected, atol=1e-6, rtol=0)

    if not target_matches(view.get('target', [])) or any(
            not target_matches(fr.get('camera_target', []))
            or not np.allclose(fr['cam_R'], applied_R, atol=1e-6, rtol=0) for fr in sheet):
        raise ValueError('registered view not maintained on the sheet frames')


def effective_protocol(base, amendment):
    replacements = amendment['holdout_replacements']
    holdouts = [replacements.get(ds, {}).get('dataset', ds) for ds in base['holdout_datasets']]
    gates = []
    for gate in base['gates']:
        if gate['dataset'] in replacements:
            spec = replacements[gate['dataset']]
            gate = dict(gate, dataset=spec['dataset'], id=gate['id'].replace('H3_', 'H5_', 1),
                        replaces=gate['id'])
        gates.append(gate)
    return dict(base, gates=gates, holdout_datasets=holdouts,
                holdout_capture={ds: base['holdout_capture'][ds] for ds in holdouts
                                 if ds in base['holdout_capture']}
                | {spec['dataset']: spec for spec in replacements.values()})


def make_estimator(meta):
    if hasattr(cp.CutPerception, 'from_meta'):
        return cp.CutPerception.from_meta(meta)
    a = meta['estimator_args']
    return cp.CutPerception(a['W'], a['H'], a['f'], a['far'], np.asarray(a['rcm_world'], float),
                            tcp_len=a['tcp_len'])


def truth_of(fr, meta):
    NU, NV, Hs = meta['NU'], meta['NV'], meta['spacing_m']
    W, H, f = meta['W'], meta['H'], meta['f']
    R, t = fr['cam_R'], fr['cam_t']
    tcp = fr['tcp_M']
    out = dict(tip=tcp[:3, 3], jaw_axis=unit(tcp[:3, meta['tool_axis_col']]), sheet='nodes' in fr)

    def visible(p):
        pc = R.T @ (np.asarray(p) - t)
        if pc[2] <= 0:
            return False
        uv = (W / 2 - f * pc[0] / pc[2], H / 2 - f * pc[1] / pc[2])
        if not (2 <= uv[0] < W - 2 and 2 <= uv[1] < H - 2):
            return False
        return depth_visible(fr['depth'].astype(float), uv, pc[2])

    out['tip_visible'] = visible(out['tip'])
    if out['sheet']:
        n = fr['nodes']
        m = 0.5 * (n[:, NU // 2 - 1] + n[:, NU // 2])
        j0 = min(int(round(float(fr['cut_length']) / Hs)), NV - 4)
        out.update(front=np.asarray(fr['cut_front'], float), front_visible=visible(fr['cut_front']),
                   dir_live=unit(m[j0 + 3] - m[j0]), dir_place=unit(fr['sheet_v']),
                   cut_length=float(fr['cut_length']))
    return out


def degrade(fr, deg, rng):
    R_true, t_true = fr['cam_R'], fr['cam_t']
    R_est = cp.rotvec(unit(rng.normal(size=3)) * math.radians(deg['rot_deg'])) @ R_true
    t_est = t_true + unit(rng.normal(size=3)) * deg['trans_m']
    depth = fr['depth'].astype(float)
    if deg['depth_noise_m'] > 0:
        depth = depth + rng.normal(0.0, deg['depth_noise_m'], depth.shape)
    return depth, R_est, t_est


def score_frame(fr, meta, estimator, depth, R_est, t_est):
    post_grasp = 'grasped_arms' in fr and len(fr['grasped_arms']) > 0
    est = estimator.estimate(fr['rgb'].astype(float), depth, R_est, t_est, post_grasp=post_grasp)
    tr = truth_of(fr, meta)
    cut, sc = est.get('cut') or {}, est.get('scissors') or {}
    r = dict(tick=int(fr['tick']), sheet=tr['sheet'], tip_visible=tr['tip_visible'],
             tip_valid=bool(sc.get('valid')), cut_valid=bool(cut.get('valid')),
             front_valid=bool(cut.get('front_valid')),
             cut_reason=cut.get('reason'), tip_reason=sc.get('reason'))
    if sc.get('tip') is not None:
        r['tip_err_m'] = float(np.linalg.norm(np.asarray(sc['tip']) - tr['tip']))
    if sc.get('jaw_axis') is not None:
        r['jaw_axis_err_rad'] = float(math.acos(np.clip(unit(sc['jaw_axis']) @ tr['jaw_axis'], -1, 1)))
    if tr['sheet']:
        r['front_visible'] = tr['front_visible']
        r['cut_length_m'] = tr['cut_length']
        if cut.get('front') is not None:
            r['front_err_m'] = float(np.linalg.norm(np.asarray(cut['front']) - tr['front']))
        if cut.get('progress_m') is not None:
            r['progress_m'] = cut['progress_m']
        if cut.get('direction') is not None:
            d = unit(cut['direction'])
            r['dir_live_err_rad'] = float(math.acos(np.clip(d @ tr['dir_live'], -1, 1)))
            r['dir_place_err_rad'] = float(math.acos(np.clip(d @ tr['dir_place'], -1, 1)))
        if cut.get('front') is not None and sc.get('tip') is not None:
            rel_e = np.asarray(cut['front']) - np.asarray(sc['tip'])
            r['relative_err_m'] = float(np.linalg.norm(rel_e - (tr['front'] - tr['tip'])))
    return r


def replay(meta, frames, deg, seed, probe_rows=None):
    """One sequential pass (the estimator is stateful) -> per-tick records."""
    est_obj = make_estimator(meta)
    rng = np.random.default_rng(seed)
    rows = []
    for fr in frames:
        depth, R_est, t_est = degrade(fr, deg, rng)
        if probe_rows is not None and 'probe_rgb' in fr and 'probe_depth' in fr:
            probe = dict(fr, rgb=fr['probe_rgb'], depth=fr['probe_depth'])
            probe_rows.append(score_frame(probe, meta, deepcopy(est_obj), probe['depth'].astype(float),
                                          R_est, t_est))
        rows.append(score_frame(fr, meta, est_obj, depth, R_est, t_est))
    return rows


def supplement_occlusion(results, original, probe_metrics, minimum):
    result = results.get(original, {}).get('none', {}).get('metrics')
    if result is None:
        return
    gate_metrics = result.setdefault('occlusion_gate_metrics', {})
    for target in ('front', 'tip'):
        n, wrong = f'occluded_{target}_ticks', f'{target}_vbw_on_occluded'
        gate_metrics[wrong] = (result[wrong] + probe_metrics[wrong]
                               if probe_metrics[n] >= minimum else None)


def _stats(v):
    v = sorted(v)
    if not v:
        return dict(n=0)
    q = lambda p: float(v[min(len(v) - 1, int(math.ceil(p / 100 * len(v))) - 1)])  # noqa: E731
    return dict(n=len(v), median=float(np.median(v)), p95=q(95), max=float(v[-1]))


def metrics(rows, wrong):
    def rate(num, den):
        return (num / den) if den else None

    sheet = [r for r in rows if r['sheet']]
    tipv = [r for r in rows if r['tip_valid']]
    frv = [r for r in sheet if r['front_valid']]
    both = [r for r in sheet if r['front_valid'] and r['tip_valid'] and 'relative_err_m' in r]
    tip_vis = [r for r in sheet if r['tip_visible']]
    fr_vis = [r for r in sheet if r['front_visible']]
    occl_f = [r for r in sheet if not r['front_visible']]
    occl_t = [r for r in sheet if not r['tip_visible']]
    m = dict(
        ticks=len(rows), sheet_ticks=len(sheet),
        pre_sheet_ticks=len(rows) - len(sheet),
        tip_err=_stats([r['tip_err_m'] for r in tipv]),
        tip_err_median_m=_stats([r['tip_err_m'] for r in tipv]).get('median'),
        tip_vbw_rate=rate(sum(r['tip_err_m'] > wrong for r in tipv), len(tipv)),
        tip_availability=rate(sum(r['tip_valid'] for r in tip_vis), len(tip_vis)),
        jaw_axis_err=_stats([r['jaw_axis_err_rad'] for r in tipv if 'jaw_axis_err_rad' in r]),
        front_err=_stats([r['front_err_m'] for r in frv]),
        front_vbw_rate=rate(sum(r['front_err_m'] > wrong for r in frv), len(frv)),
        front_availability=rate(sum(r['front_valid'] for r in fr_vis), len(fr_vis)),
        direction_live_err=_stats([r['dir_live_err_rad'] for r in sheet if r['cut_valid'] and 'dir_live_err_rad' in r]),
        direction_live_err_median_rad=_stats([r['dir_live_err_rad'] for r in sheet
                                              if r['cut_valid'] and 'dir_live_err_rad' in r]).get('median'),
        direction_place_err=_stats([r['dir_place_err_rad'] for r in sheet if r['cut_valid'] and 'dir_place_err_rad' in r]),
        relative_err=_stats([r['relative_err_m'] for r in both]),
        relative_vbw_rate=rate(sum(r['relative_err_m'] > wrong for r in both), len(both)),
        pre_sheet_false_positives=sum(1 for r in rows if not r['sheet'] and (r['cut_valid'] or r['front_valid'])),
        occluded_front_ticks=len(occl_f), occluded_tip_ticks=len(occl_t),
        front_vbw_on_occluded=sum(1 for r in occl_f if r['front_valid'] and r.get('front_err_m', 0) > wrong),
        tip_vbw_on_occluded=sum(1 for r in occl_t if r['tip_valid'] and r.get('tip_err_m', 0) > wrong),
        front_valid_on_occluded=sum(1 for r in occl_f if r['front_valid']),
    )
    return m


def check(gate, results, dev_datasets):
    relevant_counts = {'pre_sheet_false_positives': 'pre_sheet_ticks',
                       'front_vbw_on_occluded': 'occluded_front_ticks',
                       'tip_vbw_on_occluded': 'occluded_tip_ticks'}

    def value(ds, deg):
        metrics = results.get(ds, {}).get(deg, {}).get('metrics') or {}
        name = gate['metric']
        if name in metrics.get('occlusion_gate_metrics', {}):
            return metrics['occlusion_gate_metrics'][name]
        if name in relevant_counts and not metrics.get(relevant_counts[name]):
            return None
        val = metrics.get(name)
        return val if val is not None and math.isfinite(val) else None

    if gate['dataset'] == 'all':
        vals = [value(ds, gate['degradation']) for ds in dev_datasets]
        val = None if not vals or any(v is None for v in vals) else (sum(vals) if gate['op'] == '==' else max(vals))
    else:
        val = value(gate['dataset'], gate['degradation'])
    if val is None:
        return None, 'FAILED (no evidence)'
    ok = dict(zip(('<', '<=', '>', '>=', '=='), (val < gate['value'], val <= gate['value'], val > gate['value'],
                                                  val >= gate['value'], val == gate['value'])))[gate['op']]
    return val, 'PASSED' if ok else 'FAILED'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True, help='new output path; never replaces an existing report')
    ap.add_argument('--datasets', nargs='*')
    ap.add_argument('--degradations', nargs='*')
    ap.add_argument('--label', default='')
    ap.add_argument('--keep-rows', action='store_true', help='store per-tick records in the report')
    ap.add_argument('--holdout', action='store_true',
                    help='also replay the held-out datasets (only once the estimator is frozen)')
    a = ap.parse_args()
    if a.out.exists():
        ap.error(f'output already exists: {a.out}')
    amendment = json.loads(AMENDMENT.read_text(encoding='utf-8'))
    v6 = json.loads(V6.read_bytes())
    protocol = effective_protocol(json.loads(GATES.read_text(encoding='utf-8')), amendment)
    captures = {spec['dataset']: spec for spec in amendment['holdout_replacements'].values()}
    wrong = protocol['wrong_threshold_m']
    datasets = list(a.datasets or protocol['datasets'])
    if a.holdout:
        datasets += [d for d in protocol.get('holdout_datasets', []) if d not in datasets]
    elif any(d in protocol.get('holdout_datasets', []) for d in datasets):
        ap.error('held-out datasets need --holdout')
    degs = a.degradations or list(protocol['degradations'])
    results, sources, dataset_errors = {}, {}, {}
    for ds in datasets:
        try:
            meta, frames, sources[ds] = load_dataset(ds, captures.get(ds))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            dataset_errors[ds] = f'{type(exc).__name__}: {exc}'
            print(ds, dataset_errors[ds])
            continue
        results[ds] = {}
        for dname in degs:
            if dname != 'none' and ds != 'nominal':
                continue                          # sweeps run on nominal only (protocol)
            deg = protocol['degradations'][dname]
            rows = []
            for seed in deg['seeds']:
                rows += replay(meta, frames, deg, seed)
            entry = dict(metrics=metrics(rows, wrong), seeds=deg['seeds'])
            if a.keep_rows:
                entry['rows'] = rows
            results[ds][dname] = entry
            m = entry['metrics']
            print(f"{ds:20s} {dname:22s} tip med {m['tip_err_median_m']} vbw {m['tip_vbw_rate']} | "
                  f"front med {m['front_err'].get('median')} vbw {m['front_vbw_rate']} avail {m['front_availability']} | "
                  f"dir {m['direction_live_err_median_rad']} | rel vbw {m['relative_vbw_rate']}")
    spec = v6['occlusion_probes']
    probe_details = {}
    for original, companion in spec['datasets'].items():
        if original not in datasets:
            continue
        name = companion['dataset']
        try:
            capture_spec = companion if 'initial_view' in companion else None
            meta, frames, sources[name] = load_dataset(name, capture_spec)
            if capture_spec is None:
                verify_registered_capture(meta, companion)
            expected = dict(companion, **{k: spec[k] for k in ('ticks', 'distance_m', 'radius_m')})
            if meta.get('occlusion_probe') != expected or not sources[name].get('run_id'):
                raise ValueError('probe manifest does not match protocol')
            ticks = [int(f['tick']) for f in frames if 'probe_rgb' in f and 'probe_depth' in f]
            if ticks != spec['ticks']:
                raise ValueError(f'probe ticks differ from protocol: {ticks}')
            probe_rows = []
            replay(meta, frames, protocol['degradations']['none'], 0, probe_rows=probe_rows)
            probe_metrics = metrics(probe_rows, wrong)
            probe_details[original] = dict(dataset=name, metrics=probe_metrics, rows=probe_rows)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            dataset_errors[name] = f'{type(exc).__name__}: {exc}'
            probe_metrics = metrics([], wrong)
        supplement_occlusion(results, original, probe_metrics, spec['minimum_hidden_frames_per_target'])
    gates = []
    for g in protocol['gates']:
        if g.get('holdout') and not a.holdout:
            gates.append(dict(g, measured=None, status='NOT_RUN'))
            continue
        val, status = check(g, results, protocol['datasets'])
        gates.append(dict(g, measured=val, status=status))
        print(f"{g['id']:32s} {status:22s} measured={val} {g['op']} {g['value']}")
    src = ['scripts/vision/cut_perception.py', 'scripts/vision/replay_perception.py', 'config/perception/m5a_gates.json']
    report = dict(run_id=uuid.uuid4().hex, label=a.label, created_at=datetime.now(timezone.utc).isoformat(),
                   protocol=protocol, protocol_sha256=hashlib.sha256(GATES.read_bytes()).hexdigest(),
                   previous_amendment=amendment, amendment=v6,
                   previous_amendment_sha256=hashlib.sha256(AMENDMENT.read_bytes()).hexdigest(),
                   amendment_sha256=hashlib.sha256(V6.read_bytes()).hexdigest(),
                   source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:16] for s in src},
                   datasets=sources, dataset_errors=dataset_errors, results=results, probes=probe_details, gates=gates,
                   holdout_evaluated=a.holdout,
                   all_gates_passed=not dataset_errors
                   and all(g['status'] == 'PASSED' for g in gates if g['status'] != 'NOT_RUN'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(report, indent=1) + '\n')
    print('report:', a.out, 'all gates passed:', report['all_gates_passed'])
    return 0 if report['all_gates_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
