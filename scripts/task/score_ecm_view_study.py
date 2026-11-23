"""Score the registered ECM shadow captures and the two visual trials."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from score_m5b import ROOT, TASK, clean_run, cut_length, done, no_tear, safe_outcome

sys.path.insert(0, str(ROOT / 'scripts' / 'vision'))
from replay_perception import cp, load_dataset  # noqa: E402

PROTOCOL = ROOT / 'config' / 'task' / 'm5b_ecm_view_study_v1.json'
OPTICAL_DIAGNOSTIC = PROTOCOL.with_name('m5b_ecm_view_optical_diagnostic_v1.json')


def matched(value, expected):
    a = np.asarray(value, dtype=float)
    b = np.asarray(expected, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=1e-6, rtol=0))


def paired_columns(frame, meta, qualification):
    nodes = frame['nodes'].reshape(meta['NV'], meta['NU'], 3)
    mask = cp._sheet_mask(frame['rgb'].astype(float))
    depth = frame['depth']
    radius = qualification['pixel_radius']
    tolerance = qualification['depth_tolerance_m']
    H, W = mask.shape
    f = meta['f']

    def visible(point):
        x, y, z = frame['cam_R'].T @ (point - frame['cam_t'])
        if not np.isfinite([x, y, z]).all() or z <= 0:
            return False
        u, v = round(W / 2 - f * x / z), round(H / 2 - f * y / z)
        if not (0 <= u < W and 0 <= v < H):
            return False
        region = np.s_[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
        return bool(np.any(mask[region] & (np.abs(depth[region] - z) <= tolerance)))

    return [col for col in qualification['paired_strip_columns']
            if visible(nodes[0, col]) and visible(nodes[-1, col])]


def frame_evidence(f, meta, shadow, qualification):
    cut, scissors = shadow['cut'], shadow['scissors']
    columns = paired_columns(f, meta, qualification)
    progress = cut.get('progress_m')
    supported = (len(columns) >= qualification['minimum_paired_columns']
                 and cut.get('front_valid') is True and scissors.get('valid') is True
                 and isinstance(progress, (int, float)) and math.isfinite(progress) and progress >= 0)
    nodes = f['nodes'].reshape(meta['NV'], meta['NU'], 3)
    lengths = np.linalg.norm(nodes[-1, columns] - nodes[0, columns], axis=1)
    return dict(tick=int(f['tick']), paired_columns=columns, supported=bool(supported),
                visible_chord_lengths_m=lengths.tolist(),
                progress_m=progress, front_valid=cut.get('front_valid'), scissors_valid=scissors.get('valid'),
                front_mode=cut.get('mode'))


def read_run(name):
    path = TASK / (name + '.json')
    raw = path.read_bytes()
    entry = dict(report=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw), entry


def candidate(name, pose, protocol, scene_hash):
    out = dict(name=name, pose=pose, qualified=False, eligible=False, windows={}, counts={})
    try:
        run, provenance = read_run(name)
        out.update(provenance, state=run.get('final_state'),
                   reason=run.get('error') or run.get('cleanup_error') or run.get('reason'))
        if not done(run) or not no_tear(run) or not run.get('trace') or run['final_observation'].get('faults') != []:
            return out
        meta, frames, sources = load_dataset(name)
        readback = run['initial_view_readback']
        perc = meta['perception_cfg']
        if (meta.get('run_id') != run.get('run_id') or meta['scene_sha256'] != scene_hash
                or run['scene_sha256'] != scene_hash or meta['scenario'] != 'nominal'
                or meta['mode'] != 'full' or meta['align_mode'] != 'tool_pose' or meta['control_source'] != 'truth'
                or perc['mode'] != 'shadow' or not perc['markers'] or perc['depth_noise_m'] != 0
                or any(perc['camera_pose_error'][key] for key in ('rot_deg', 'trans_m'))
                or not matched(meta['initial_view'], pose[:2]) or meta['initial_view_insertion_m'] != pose[2]
                or not matched(readback['target'], pose)):
            raise ValueError('capture metadata or initial ECM readback differs from protocol')
        out['capture'] = sources
        frames = [f for f in frames if 'nodes' in f]
        if not frames or not all(matched(f.get('camera_target'), pose)
                                 and matched(f['cam_R'], readback['applied_R'])
                                 and matched(f['cam_t'], readback['applied_t']) for f in frames):
            raise ValueError('missing sheet frames or ECM pose changed during capture')
        out['eligible'] = True
        windows = dict(pre_grasp=[], two_arms_uncut=[], post_cut=[])
        shadow = {p['tick']: p['estimate'] for p in run['perception']}
        seen_grasp = False
        for f in frames:
            holds = len(f['grasped_arms'])
            seen_grasp |= holds > 0
            if float(f['cut_length']) > 0:
                phase = 'post_cut'
            elif holds == 2:
                phase = 'two_arms_uncut'
            elif not seen_grasp:
                phase = 'pre_grasp'
            else:
                continue
            windows[phase].append(frame_evidence(f, meta, shadow[int(f['tick'])], protocol['qualification']))
        out['windows'] = windows
        out['counts'] = {phase: dict(supported=sum(row['supported'] for row in rows), frames=len(rows),
                                    both_ends_visible=sum(len(row['paired_columns']) >=
                                                          protocol['qualification']['minimum_paired_columns']
                                                          for row in rows))
                         for phase, rows in windows.items()}
        out['qualified'] = all(out['counts'][phase]['supported'] > 0 for phase in ('pre_grasp', 'two_arms_uncut'))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out.update(error=f'{type(exc).__name__}: {exc}', eligible=False, qualified=False)
    return out


def select(candidates):
    def rank(c):
        def fraction(phase):
            row = c['counts'][phase]
            return row['supported'] / max(1, row['frames'])
        pre = fraction('pre_grasp')
        loaded = fraction('two_arms_uncut')
        return min(pre, loaded), loaded

    eligible = [c for c in candidates if c['eligible'] and c['counts']['two_arms_uncut']['supported'] > 0]
    pool = [c for c in eligible if c['qualified']] or eligible
    return max(pool, key=rank) if pool else None


def select_optical(candidates):
    eligible = [c for c in candidates if c['eligible'] and len(c['counts']) == 3 and
                all(row['frames'] > 0 and row['both_ends_visible'] == row['frames']
                    for row in c['counts'].values())]
    return max(eligible, key=lambda c: c['counts']['post_cut']['supported'] / c['counts']['post_cut']['frames'],
               default=None)


def closure_checks(events, max_age):
    pending = None
    closure_rows, pairs, missing, invalid = [], [], [], []
    for event in events:
        if event['event'] == 'closure':
            tick = event['tick']
            closure_rows.append(event)
            if pending is not None:
                missing.append(pending)
            pending = tick
            age = event.get('fix_age')
            if type(age) is not int or not 0 <= age <= max_age or event.get('scissors_valid') is not True:
                invalid.append(tick)
        if (event['event'] == 'post_closure_reacquired' and pending is not None
                and event['tick'] > pending and event['mode'] in ('slit', 'bank_slit')):
            pairs.append(dict(closure_tick=pending, reacquisition=event))
            pending = None
    if pending is not None:
        missing.append(pending)
    return dict(closures=closure_rows, paired=pairs, missing_reacquisition=missing, invalid_closures=invalid)


def trial(name, pose, depth_noise, protocol, scene_hash):
    run, out = read_run(name)
    cfg = run['config']
    perc = cfg['perception']
    limit = cfg['cut']['length_m'] + cfg['cut']['max_overcut_m']
    obs = run.get('final_observation') or {}
    checks = closure_checks(run.get('event_log', []), cfg['vision']['fix_max_age_ticks'])
    trace = [row.get('cut_length_m') for row in run.get('trace', [])]
    peak = max(trace) if trace and all(isinstance(v, (float, int)) and math.isfinite(v) for v in trace) else None
    matching_run = (run.get('scenario') == protocol['visual_trials']['scenario'] and run.get('mode') == 'full'
                    and run.get('control_source') == 'vision' and run.get('scene_sha256') == scene_hash
                    and cfg['cut']['align_mode'] == 'tool_pose' and perc['seed'] == protocol['visual_trials']['seed']
                    and perc['depth_noise_m'] == depth_noise and matched(run.get('initial_view'), pose[:2])
                    and run.get('initial_view_insertion_m') == pose[2]
                    and matched((run.get('initial_view_readback') or {}).get('target'), pose)
                    and run.get('protocol_amendment') == json.loads((ROOT / protocol['controller_protocol']).read_bytes()))
    completed = (done(run) and run.get('reason', '').endswith('scissors retracted') and no_tear(run)
                 and obs.get('faults') == [] and 'untagged' not in run.get('truth_reads_by_purpose', {})
                 and safe_outcome(run, cfg['cut']['length_m'], limit))
    return dict(out, depth_noise_m=depth_noise, state=run.get('final_state'), reason=run.get('reason'),
                error=run.get('error') or run.get('cleanup_error'), matching_run=matching_run,
                cut_length_m=cut_length(run), peak_m=peak, torn=obs.get('torn'), camera_moves=run.get('camera_moves'),
                endpoint_trial_pass=bool(matching_run and completed and peak is not None and peak <= limit + 1e-12
                                         and checks['closures'] and not checks['missing_reacquisition']
                                         and not checks['invalid_closures']),
                strict_B5=clean_run(run) and safe_outcome(run, cfg['cut']['length_m'], limit), **checks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, help='existing candidate summary; omit for selection stage')
    p.add_argument('--optical-diagnostic', action='store_true',
                   help='report follow-up optical diagnostics after failed qualification')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    raw_protocol = PROTOCOL.read_bytes()
    protocol = json.loads(raw_protocol)
    protocol_hash = hashlib.sha256(raw_protocol).hexdigest()
    scene_hash = hashlib.sha256((ROOT / protocol['scene']).read_bytes()).hexdigest()
    result = dict(protocol=protocol, protocol_sha256=protocol_hash, scene_sha256=scene_hash)
    if args.selection:
        raw = args.selection.read_bytes()
        selection = json.loads(raw)
        if selection['protocol_sha256'] != protocol_hash or selection['scene_sha256'] != scene_hash:
            p.error('the selection is for another protocol/scene')
        selected = selection['selected']
        run_prefix = protocol['id']
        if args.optical_diagnostic:
            opt_raw = OPTICAL_DIAGNOSTIC.read_bytes()
            optical = json.loads(opt_raw)
            if selected or args.selection.resolve() != (ROOT / optical['selection']).resolve():
                p.error('the optical diagnostic requires the registered failed selection')
            best = select_optical(selection['candidates'])
            if best is not None:
                selected = {k: best[k] for k in ('name', 'pose', 'qualified', 'counts')}
            run_prefix = optical['id']
            result.update(optical_protocol=optical, optical_protocol_sha256=hashlib.sha256(opt_raw).hexdigest())
        if not selected:
            p.error('the selection is absent')
        runs = [trial(f"{run_prefix}_{label}", selected['pose'], noise, protocol, scene_hash)
                for label, noise in zip(('V_nominal', 'V_noise_depth_0p5'), protocol['visual_trials']['depth_noise_m'])]
        result.update(selection_sha256=hashlib.sha256(raw).hexdigest(), selected=selected, trials=runs,
                      study_pass=selected['qualified'] and all(run['endpoint_trial_pass'] for run in runs))
    else:
        if args.optical_diagnostic:
            p.error('--optical-diagnostic requires --selection')
        candidates = [candidate(f"{protocol['id']}_c{i:02d}", pose, protocol, scene_hash)
                      for i, pose in enumerate(protocol['candidates_deg_deg_m'])]
        best = select(candidates)
        selected = {k: best[k] for k in ('name', 'pose', 'qualified', 'counts')} if best else None
        result.update(candidates=candidates, selected=selected)
    with args.out.open('x', encoding='utf-8', newline='\n') as dst:
        json.dump(result, dst, indent=2, allow_nan=False)
        dst.write('\n')
    print(json.dumps(result['selected']))
    return 0 if result.get('study_pass', bool(result['selected'] and result['selected']['qualified'])) else 1


if __name__ == '__main__':
    raise SystemExit(main())
