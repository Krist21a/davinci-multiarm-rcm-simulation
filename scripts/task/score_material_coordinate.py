"""Score material-chart diagnostics; simulator nodes are scoring-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from score_ecm_view_study import ROOT, closure_checks, load_dataset, matched, paired_columns, read_run
from score_m5b import clean_run, cut_length, done, safe_outcome
from material_coordinate import KNOTS_M, ROWS, project


POSE = (4.5, 0.0, 0.080)
SCENE = ROOT / 'scenes/davinci_si_articulated_verified.ttt'
REFERENCE = 'm5b_ecm_view_study_v1_c02'


def chart_error(frame, estimate, limit_m):
    cut = estimate['cut']
    chart = cut['material_chart']
    start = np.asarray(cut['material_start'])
    along = np.asarray(cut['direction'])
    lateral = np.cross(estimate['tissue']['normal'], along)
    knots = chart['material_knots_m']
    if not np.isin(knots, KNOTS_M).all():
        raise ValueError('unexpected material row labels')
    truth = []
    # Ground-truth rows are available here for scoring only, including hidden ones.
    for j in ROWS:
        d = frame['nodes'][j] - start
        w = d @ lateral
        order = np.argsort(w)
        if not w[order[0]] <= 0 <= w[order[-1]]:
            raise ValueError('inferred cut line misses a registered row')
        truth.append(float(np.interp(0, w[order], (d @ along)[order])))
    visible_truth = np.asarray(truth)[np.searchsorted(KNOTS_M, knots)]
    true_at_limit = float(np.interp(limit_m, KNOTS_M, truth))
    limit_s = project(knots, chart['guard_knots_m'], limit_m, stop_at_supported_row=True)
    return dict(error_m=(chart['projected_knots_m'] - visible_truth).tolist(),
                guard_minus_true_boundary_m=(chart['guard_knots_m'] - visible_truth).tolist(),
                guard_minus_true_limit_m=limit_s - true_at_limit,
                true_projected_span_m=truth[-1] - truth[0])


def capture_metrics(run, provenance, prefix, local, front_recovery=False):
    meta, frames, files = load_dataset(prefix + 'truth')
    if (files.get('run_id') != run['run_id'] or files['scene_sha256'] != run['scene_sha256']
            or meta['estimator_args'].get('material_pattern') is not True
            or meta['estimator_args'].get('local_material_chart', False) is not local
            or meta['estimator_args'].get('front_recovery', False) is not front_recovery):
        raise ValueError('frame capture is not from the registered truth run')
    recorded = {row['tick']: row['estimate'] for row in run['perception']}
    if set(recorded) != {int(f['tick']) for f in frames}:
        raise ValueError('incomplete frame/estimate evidence')
    camera = run['initial_view_readback']
    qual = json.loads((ROOT / 'config/task/m5b_ecm_view_study_v1.json').read_bytes())['qualification']
    cfg = run['config']['cut']
    limit = cfg['length_m'] + cfg['max_overcut_m']

    samples = []
    for frame in frames:
        if 'nodes' not in frame:
            continue
        if (not matched(frame.get('camera_target'), POSE) or not matched(frame['cam_R'], camera['applied_R'])
                or not matched(frame['cam_t'], camera['applied_t'])):
            raise ValueError('truth capture moved away from the registered fixed camera pose')
        tick = int(frame['tick'])
        chart = recorded[tick]['cut'].get('material_chart') or {}
        columns = paired_columns(frame, meta, qual)
        row = dict(tick=tick, paired_end_columns=columns, banks_verified=chart.get('valid', False),
                   verified_rows_m=chart.get('material_knots_m'), complete_chart=chart.get('complete', chart.get('valid', False)),
                   boundary_pairs=chart.get('boundary_pairs'), reason=chart.get('reason'),
                   cut_length_m=float(frame['cut_length']), held_arms=len(frame['grasped_arms']))
        if row['banks_verified']:
            row.update(chart_error(frame, recorded[tick], limit))
        samples.append(row)
    errors = [abs(x) for row in samples for x in row.get('error_m', ())]
    guard_errors = [row['guard_minus_true_limit_m'] for row in samples if row['banks_verified']]
    groups = [s for s in samples if s['held_arms'] == 2 and s['cut_length_m'] == 0]
    return dict(provenance, capture=files, sheet_frames=len(samples),
                marked_chart_frames=sum(s['banks_verified'] for s in samples),
                partial_chart_frames=sum(s['banks_verified'] and not s['complete_chart'] for s in samples),
                loaded_uncut_chart_frames=sum(s['banks_verified'] for s in groups),
                loaded_uncut_frames=len(groups), max_abs_boundary_error_m=max(errors, default=None),
                max_guard_minus_true_limit_m=max(guard_errors, default=None),
                samples=samples)


def check_run(run, protocol_hash, scene_hash, source, noise, seed, local, front_recovery, predictive, exclude_marker,
              recovery_view_after_cut=False, ray_fit=False):
    perc = run['config']['perception']
    pose = (run.get('initial_view_readback') or {}).get('target')
    if (not clean_run(run) or run['protocol_amendment_sha256'] != protocol_hash
            or run['scene_sha256'] != scene_hash or not matched(pose, POSE) or run['scenario'] != 'nominal'
            or run['mode'] != 'full' or run['control_source'] != source
            or run['config']['cut']['align_mode'] != 'tool_pose' or perc.get('material_pattern') is not True
            or perc.get('local_material_chart', False) is not local
            or perc.get('front_recovery', False) is not front_recovery
            or perc.get('predictive_view', False) is not predictive
            or perc.get('exclude_tcp_marker', False) is not exclude_marker
            or perc.get('recovery_view_after_cut', False) is not recovery_view_after_cut
            or perc.get('ray_marker_fit', False) is not ray_fit
            or perc.get('depth_noise_m') != noise or perc.get('seed') != seed or not run['trace']
            or (source == 'vision' and 'untagged' in run.get('truth_reads_by_purpose', {}))):
        raise ValueError('run does not match the registered development protocol')
    if exclude_marker:
        marker = run['tip_marker_optics']
        if (marker['alias'] != 'psm2_tool_tip_visual' or marker['layer_after'] != 0
                or marker['properties_after'] != (marker['properties_before'] & ~marker['render_bit'])):
            raise ValueError('missing verified TCP marker optics')


def verify_current_capture(run, name):
    """Bind a current-workspace score to the runner, effective inputs and actual frames."""
    sources = run['source_sha256']
    required = {'scripts/task/' + f + '.py' for f in
                ('cut_fsm', 'sim_world', 'workspace', 'ground_truth', 'visibility', 'tool_pose', 'cutter_view',
                 'run_cut_task')} | {'scripts/vision/cut_perception.py', 'scripts/vision/material_coordinate.py',
                                     'config/task/cut_task.json'}
    if set(sources) != required or any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h
                                       for p, h in sources.items()):
        raise ValueError('current runner source mismatch')
    fixed = json.loads((ROOT / 'config/task/cut_task.json').read_bytes())
    fixed['cut']['control_source'] = run['control_source']
    for k in fixed:
        if k != 'perception' and fixed[k] != run['config'][k]:
            raise ValueError('task or safety configuration mismatch')
    perc = run['config']['perception']
    if (perc['dump_dir'] != f'validation/perception/frames/{name}' or perc['mode'] != 'shadow'
            or perc['camera_pose_error'] != fixed['perception']['camera_pose_error'] or perc['markers'] is not True):
        raise ValueError('capture optics mismatch')
    meta, frames, files = load_dataset(name)
    ticks = [int(f['tick']) for f in frames]
    if (not files.get('run_id') or files['run_id'] != run['run_id'] or meta['run_id'] != run['run_id']
            or meta['perception_cfg'] != perc or files['scene_sha256'] != run['scene_sha256']
            or meta['estimator_args'].get('ray_marker_fit', False) is not perc.get('ray_marker_fit', False)
            or ticks != list(range(1, len(ticks) + 1)) or ticks != [r['tick'] for r in run['trace']]
            or ticks != [r['tick'] for r in run['perception']] or not ticks):
        raise ValueError('incomplete capture or mismatched identity')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--version', choices=('1.5', '1.6', '1.7', '1.8', '1.9', '1.9.1', '1.10'), default='1.9')
    args = parser.parse_args()
    version = args.version.replace('.', '_')
    prefix = f'm5b_v{version}_marked_'
    local = args.version != '1.5'
    front_recovery = args.version not in ('1.5', '1.6')
    predictive = front_recovery and args.version != '1.7'
    exclude_marker = predictive and args.version != '1.8'
    recovery = args.version in ('1.9.1', '1.10')
    ray_fit = args.version == '1.10'
    raw = (ROOT / 'config/task' / f'm5b_protocol_v{version}.json').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    with SCENE.open('rb') as f:
        scene_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    truth, provenance = read_run(prefix + 'truth')
    reference, ref_source = read_run(REFERENCE)
    check_run(truth, sha, scene_sha, 'truth', 0., 0, local, front_recovery, predictive, exclude_marker, recovery, ray_fit)
    if recovery:
        verify_current_capture(truth, prefix + 'truth')
    capture = capture_metrics(truth, provenance, prefix, local, front_recovery)
    capture['trace_matches_unmarked_same_view'] = truth['trace'] == reference['trace']
    capture['unmarked_reference'] = ref_source
    trials = []
    for label, noise, seed in [('V_nominal', 0., 0), ('V_noise_depth_0p5', .0005, 1)]:
        r, sources = read_run(prefix + label)
        check_run(r, sha, scene_sha, 'vision', noise, seed, local, front_recovery, predictive, exclude_marker, recovery, ray_fit)
        if recovery:
            if r['controller_sha256'] != truth['controller_sha256'] or r['run_id'] == truth['run_id']:
                raise ValueError('controller or run identity mismatch')
            sources['capture'] = verify_current_capture(r, prefix + label)
        checks = closure_checks(r['event_log'], 0)
        cfg = r['config']['cut']
        low, high = cfg['length_m'], cfg['length_m'] + cfg['max_overcut_m']
        safety = safe_outcome(r, low, high)
        cuts = [x['estimate']['cut'] for x in r['perception']]
        charts = [x['material_chart'] for x in cuts if (x.get('material_chart') or {}).get('valid')]
        length = {x['tick']: x['cut_length_m'] for x in r['trace']}
        trials.append(dict(sources, **checks, depth_noise_m=noise, strict_B5=safety, observations=len(cuts),
                           closure_growth_m=[length[e['tick'] + 1] - length[e['tick']] for e in checks['closures']],
                           view_predictions=[x for x in r['event_log'] if x['event'] == 'predicted_view'],
                           optical_retries=[dict(tick=x['world_tick'], reason=x['reason']) for x in r['log']
                                            if x['reason'].startswith('post-advance optical retry:')],
                           peak_cut_m=max(x['cut_length_m'] for x in r['trace']),
                           cut_length_m=cut_length(r), torn=r['final_observation']['torn'],
                           state=r['final_state'], reason=r['reason'],
                           valid_chart_frames=len(charts),
                           partial_chart_frames=sum(not x.get('complete', True) for x in charts),
                           mapped_front_frames=sum(x.get('progress_m') is not None for x in cuts),
                           endpoint_pass=(done(r) and safety and bool(checks['closures'])
                                          and not checks['invalid_closures'] and not checks['missing_reacquisition'])))
    report = dict(protocol=json.loads(raw), protocol_sha256=sha, scene_sha256=scene_sha,
                  scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  marked_truth=capture, trials=trials)
    contents = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with NamedTemporaryFile(dir=args.out.parent, mode='w', encoding='utf-8', newline='\n', delete=False) as f:
        tmp = Path(f.name)
    try:
        tmp.write_text(contents, encoding='utf-8', newline='\n')
        os.link(tmp, args.out)  # Publish complete contents without replacing an existing report.
    finally:
        tmp.unlink()
    print(f"chart: {capture['marked_chart_frames']}/{capture['sheet_frames']} sheet frames; "
          f"truth trace identity: {capture['trace_matches_unmarked_same_view']}")
    for t in trials:
        print(t['depth_noise_m'], t['state'], t['reason'], len(t['closures']), len(t['paired']))


if __name__ == '__main__':
    main()
