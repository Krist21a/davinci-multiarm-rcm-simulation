"""Score one version of the preregistered M5b matrix without modifying its reports."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'validation' / 'task'
BASE = ROOT / 'config' / 'task' / 'm5b_protocol.json'
AMENDMENT = BASE.with_name('m5b_protocol_v1_4.json')
REFERENCE = TASK / 'm5a_trace_check.json'
MAIN = ('V_nominal', 'V_occlusion_transient', 'V_slip_regrasp', 'V_relax')
NOISE = ('V_noise_depth_0p5', 'V_noise_depth_1p0', 'V_cam_trans_1mm_fixed', 'V_cam_rot_1deg_fixed')
TRUTH = 'T_nominal_identity'


def clean_run(run):
    return (run.get('final_state') in ('DONE', 'FAILED')
            and not run.get('error') and not run.get('cleanup_error'))


def done(run):
    return clean_run(run) and run['final_state'] == 'DONE' and run.get('success') is True


def cut_length(run):
    value = (run.get('final_observation') or {}).get('cut_length_m')
    return value if finite_number(value) else None


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def no_tear(run):
    torn = (run.get('final_observation') or {}).get('torn')
    return clean_run(run) and type(torn) is int and torn == 0


def safe_outcome(run, low, high):
    length = cut_length(run)
    if (not no_tear(run) or length is None or not 0 <= length <= high + 1e-12
            or (run.get('final_observation') or {}).get('faults') != []):
        return False
    if any(not finite_number(row.get('cut_length_m')) or not 0 <= row['cut_length_m'] <= high + 1e-12
           for row in (run.get('trace') or [])):
        return False
    if done(run):
        return length >= low - 1e-12
    if run.get('final_state') != 'FAILED':
        return False
    reason = run.get('reason', '')
    return (reason.startswith('alignment not reached (')
            or reason in {'cut site not found', 'exposure not reached', 'no camera pose left',
                          'homing failed', 'scissors cannot open', 'scissors cannot close',
                          'scissors retraction refused', 'clear-and-look action refused',
                          'total tick budget exhausted', 'clear-and-look exhausted', 'camera adjustment exhausted',
                          'retraction adjustment exhausted', 'relocalization exhausted'}
            or reason.startswith(('scissors alignment refused:', 'scissors advance refused',
                                  'scissors closure refused:'))
            or reason.endswith(' retraction adjustment refused'))


def closure_metric(samples, count, tolerance):
    finite = [x for x in samples if finite_number(x) and x >= 0]
    median = statistics.median(finite) if finite else None
    return dict(result=len(finite) == count and median is not None and median < tolerance,
                expected_samples=count, valid_samples=len(finite), median_m=median)


def evaluate(runs, reference, cfg):
    expected = (*MAIN, *NOISE, 'V_cam_trans_1mm_per_tick', 'V_relax_large')
    vision = {key: runs.get(key, {}) for key in expected}
    low = cfg['cut']['length_m']
    high = low + cfg['cut']['max_overcut_m']
    ages = cfg['vision']['fix_max_age_ticks']
    closures = {}
    for key, run in vision.items():
        events = run.get('event_log')
        closures[key] = [e for e in events if e.get('event') == 'closure'] if isinstance(events, list) else []
    lengths = {key: cut_length(run) for key, run in vision.items() if done(run)}
    b5 = {key: safe_outcome(run, low, high) for key, run in vision.items()}
    grasp = {key: no_tear(run) for key, run in vision.items()}
    invalid = [(key, event.get('tick')) for key, events in closures.items() for event in events
               if not (type(event.get('fix_age')) is int and 0 <= event['fix_age'] <= ages
                       and event.get('scissors_valid') is True)]
    pool = [e.get('lateral_truth_m') for key in ('V_nominal', *NOISE) for e in closures[key]]
    truth = runs.get(TRUTH, {})
    main_done = [key for key in MAIN if done(vision[key])]
    noise_done = [key for key in NOISE if done(vision[key])]
    truth_errors = {key: [event for event in (run.get('event_log') or [])
                          if 'TruthReadInControl' in event.get('error', '')]
                    for key, run in vision.items()}
    b7 = closure_metric(pool, len(pool), cfg['cut']['align_tolerance_m'])
    b7['result'] &= all(clean_run(vision[key]) for key in ('V_nominal', *NOISE))
    return {
        'B0': dict(result=done(truth) and bool(reference.get('trace'))
                   and truth.get('trace') == reference['trace'], ticks=len(truth.get('trace') or [])),
        'B1': dict(result=all(clean_run(run) and bool(run.get('truth_reads_by_purpose'))
                              and 'untagged' not in run['truth_reads_by_purpose'] and not truth_errors[key]
                              for key, run in vision.items()), exceptions=truth_errors),
        'B2': dict(result=len(main_done) == len(MAIN), done=main_done),
        'B3': dict(result=bool(lengths) and all(v is not None and low - 1e-12 <= v <= high + 1e-12
                                              for v in lengths.values()), lengths_m=lengths,
                   status='SCORED' if lengths else 'NO_EVIDENCE'),
        'B4': dict(result=all(grasp.values()), per_run=grasp),
        'B5': dict(result=all(b5.values()), per_run=b5),
        'B6': dict(result=all(clean_run(r) and isinstance(r.get('event_log'), list) for r in vision.values())
                   and not invalid, closures=sum(map(len, closures.values())), violations=invalid),
        'B7': b7,
        'B8': dict(result=len(noise_done) >= 3 and all(b5[key] for key in NOISE), done=noise_done),
    }


def matching_trial(key, run, cfg, scene_hash):
    if (not isinstance(run.get('run_id'), str) or not run['run_id']
            or run.get('scene_sha256') != scene_hash or run.get('mode') != 'full'
            or run.get('initial_view') is not None or run.get('initial_view_insertion_m') is not None):
        return False
    vision = key != TRUTH
    source = 'vision' if vision else 'truth'
    actual = run.get('config') or {}
    if (run.get('control_source') != source or actual.get('cut') != dict(cfg['cut'], control_source=source)
            or any(actual.get(section) != cfg[section]
                   for section in ('budgets', 'tension', 'grasp', 'view', 'vision', 'roles', 'events'))):
        return False
    scenario = key[2:].lower() if key in (*MAIN, 'V_relax_large') else 'nominal'
    if run.get('scenario') != scenario:
        return False
    perc = actual.get('perception') or {}
    pose = perc.get('camera_pose_error') or {}
    return (perc.get('markers') is True
            and perc.get('marker3_off') is None
            and perc.get('depth_noise_m') == {'V_noise_depth_0p5': 0.0005, 'V_noise_depth_1p0': 0.001}.get(key, 0)
            and pose.get('trans_m') == (0.001 if key.startswith('V_cam_trans') else 0)
            and pose.get('rot_deg') == (1 if key == 'V_cam_rot_1deg_fixed' else 0)
            and pose.get('mode', 'per_tick') == ('fixed' if key.endswith('_fixed') else 'per_tick')
            and (not key.startswith(('V_noise', 'V_cam')) or perc.get('seed') == 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix')
    p.add_argument('--amendment', type=Path, default=AMENDMENT, help='registered protocol version to score')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    amendment_raw = args.amendment.read_bytes()
    base, ref = (json.loads(f.read_bytes()) for f in (BASE, REFERENCE))
    amendment = json.loads(amendment_raw)
    amendment_hash = hashlib.sha256(amendment_raw).hexdigest()
    prefix = args.prefix or f"m5b_v{str(amendment['protocol_version']).replace('.', '_')}_"
    cfg_path = BASE.parent / 'cut_task.json'
    cfg_raw = cfg_path.read_bytes()
    cfg = json.loads(cfg_raw)
    scene_path = ROOT / 'scenes' / 'davinci_si_articulated_verified.ttt'
    with scene_path.open('rb') as scene:
        scene_hash = hashlib.file_digest(scene, 'sha256').hexdigest()
    reports, runs = {}, {}
    for item in base['runs']:
        key = item['id']
        path = TASK / f'{prefix}{key}.json'
        if not path.exists():
            reports[key] = dict(status='NOT_RUN', path=path.relative_to(ROOT).as_posix())
            continue
        raw = path.read_bytes()
        runs[key] = r = json.loads(raw)
        if key != TRUTH and (r.get('protocol_amendment') != amendment
                             or r.get('protocol_amendment_sha256') != amendment_hash):
            p.error(f'{path.name}: does not match the requested protocol')
        if not matching_trial(key, r, cfg, scene_hash):
            p.error(f'{path.name}: run arguments, task config or scene differ from the registered trial')
        events = r.get('event_log') or []
        reports[key] = dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
                            state=r.get('final_state'), reason=r.get('reason'),
                            final_observation=r.get('final_observation'), report_only=item.get('report_only', False),
                            closures=sum(e.get('event') == 'closure' for e in events),
                            reacquisitions=[e for e in events if e.get('event') == 'post_closure_reacquired'])
    gates = evaluate(runs, ref, cfg)
    previous = json.loads((ROOT / amendment['previous_amendment']).read_bytes())
    report = dict(protocol=base, previous_amendment=previous, amendment=amendment,
                  scope='Development diagnostics, not acceptance',
                  amendment_sha256=amendment_hash, config_sha256=hashlib.sha256(cfg_raw).hexdigest(),
                  scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scene_sha256=scene_hash,
                  reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(), reports=reports, criteria=gates)
    with args.out.open('x', encoding='utf-8', newline='\n') as dst:
        json.dump(report, dst, indent=2, allow_nan=False)
        dst.write('\n')
    print(' '.join(f'{key}={"PASS" if v["result"] else "FAIL"}' for key, v in gates.items()))
    return 0 if all(v['result'] for v in gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
