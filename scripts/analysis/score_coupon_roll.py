"""Seal image-only coupon-roll observations before joining structural run labels."""
from bisect import bisect_left
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/task'), str(ROOT / 'scripts/vision')]
from bridge_fracture import measure
from bridge_observer import COLORS, COLOR_TOL
from cut_perception import _sheet_mask
from run_coupon_roll import ACTIVE, OUTPUT, PLAN, POLICY, angle_folder, registered
from run_same_bite import SPEC, put, sha256

EPS = 1e-12


def load(path):
    return json.loads(path.read_bytes())


def ink_counts(rgb, depth):
    ys, xs = np.nonzero(_sheet_mask(rgb.astype(float)))
    if len(xs) < 40:
        return None
    crop = np.s_[max(0, ys.min()-12):ys.max()+13, max(0, xs.min()-12):xs.max()+13]
    colors = rgb[crop].astype(float)
    d = depth[crop]
    valid = np.isfinite(d) & (0 < d) & (d < .3)
    return [int((valid & (np.sum((colors-c)**2, axis=-1) <= COLOR_TOL**2)).sum())
            for c in COLORS]


def predict(plan):
    frozen = dict(protocol_sha256=sha256(PLAN), scorer_sha256=sha256(Path(__file__)),
                  reader_sha256=plan['reader_sha256'], trials=[])
    seen_ids = set()
    for angle in plan['angles_deg']:
        folder = angle_folder(angle)
        for case in plan['cases']:
            src = folder / f'{case}_optical.json'
            result = dict(roll_deg=angle, case=case)
            frozen['trials'].append(result)
            if not src.is_file():
                result['missing'] = True
                continue
            optical = load(src)
            run_id = optical['run_id']
            reg = folder / 'registration.json'
            registration = load(reg)
            if (optical['case'] != case or optical['roll_deg'] != angle or run_id in seen_ids
                    or optical['plan_sha256'] != frozen['protocol_sha256']
                    or optical['registration_sha256'] != sha256(reg)
                    or registration['policy'] != POLICY or registration['protocol'] != load(SPEC)
                    or any(sha256(ROOT / name) != value
                           for name, value in registration['sources'].items())
                    or not {PLAN.relative_to(ROOT).as_posix(), ACTIVE.relative_to(ROOT).as_posix(),
                            Path(__file__).resolve().relative_to(ROOT).as_posix(),
                            *plan['reader_sha256']} <= registration['sources'].keys()):
                raise ValueError(f'incorrect optical identity for {angle}/{case}')
            seen_ids.add(run_id)
            result.update(run_id=run_id, registration_sha256=sha256(reg),
                          optical_sha256=sha256(src), frames=[])
            seen_ticks = set()
            for entry in optical['exposures']:
                tick = entry['tick']
                if tick in seen_ticks:
                    raise ValueError(f'duplicate optical tick: {angle}/{case}/{tick}')
                seen_ticks.add(tick)
                image = folder / f'{case}_frames/{tick:03d}.npz'
                noisy = image.with_name(f'{tick:03d}_depth.npy')
                with np.load(image) as frame:
                    d = np.load(noisy)
                    rgb, R, t, f = (frame[k] for k in ('rgb', 'cam_R', 'cam_t', 'f'))
                    estimate = entry['estimate']
                    reading = measure(rgb, d, R, t, float(f), estimate)
                    result['frames'].append(dict(
                        tick=tick, image_sha256=sha256(image), depth_sha256=sha256(noisy),
                        camera=[*rgb.shape[:2], float(f), *R.ravel(), *t],
                        ink_px=ink_counts(rgb, d), scissors_valid=bool(estimate['scissors']['valid']),
                        **reading))
    put(OUTPUT / 'predictions.json', frozen)


def score_trial(trial, plan, parent):
    angle, case = trial['roll_deg'], trial['case']
    src = angle_folder(angle) / f'{case}.json'
    result = dict(roll_deg=angle, case=case, passed=False)
    if not src.is_file():
        return dict(result, error='missing task report')
    run = load(src)
    result.update(report_sha256=sha256(src), final_state=run.get('final_state'),
                  reason=run.get('error') or run.get('reason'))
    if trial.get('missing') or run.get('error') or run.get('cleanup_error'):
        return dict(result, error=run.get('error') or run.get('cleanup_error') or 'no optical record')
    scenario, sigma, seed = parent['cases'][case]
    cfg = run['config']
    if (run['policy'] != POLICY or run['registration_sha256'] != trial['registration_sha256']
            or run['run_id'] != trial['run_id'] or run['case'] != case or run['scenario'] != scenario
            or cfg['sheet'] != dict(plan['sheet'], roll_deg=angle)
            or cfg['perception']['depth_noise_m'] != sigma or cfg['perception']['seed'] != seed
            or cfg['cut']['control_source'] != 'vision'):
        raise ValueError(f'incorrect structural run identity: {angle}/{case}')
    frames = {row['tick']: row for row in trial['frames']}
    trace = run['observations']
    truth = {row['tick']: row for row in trace}
    if not trace or len(trace) != len(truth) or set(frames) != set(truth):
        raise ValueError(f'incomplete or mismatched exposures: {angle}/{case}')
    closures = [e['tick'] for e in run['events'] if e['event'] == 'closure']
    reacq = [e['tick'] for e in run['events'] if e['event'] == 'post_closure_reacquired']

    def during_reacquired_bite(tick):
        i = bisect_left(closures, tick) - 1
        return i if i >= 0 and i < len(reacq) and closures[i] < reacq[i] <= tick else None

    alignment = next((e['world_tick'] for e in run['log']
                      if e.get('frm') == 'EXPOSE' and e.get('to') == 'ALIGN'), None)
    at_align = truth.get(alignment, {})
    grasped = at_align.get('grasped', {})
    both = (bool(at_align) and abs(at_align['cut_length_m']) <= EPS
            and all(grasped.get(cfg['roles'][key], 0) >= cfg['grasp']['min_nodes']
                    for key in ('retract_primary', 'retract_assist')))
    unsafe = [t for t, row in truth.items()
              if row['torn'] or row['faults'] or not 0 <= row['cut_length_m'] <= .010 + EPS]
    false_stops = [t for t, row in sorted(frames.items()) if row['status'] == 'STOP_CANDIDATE'
                   and truth[t]['cut_length_m'] < .009 - EPS]
    goal, wanted = (.009, 'STOP_CANDIDATE') if case == 'nominal' else (.008, 'NOT_STOP')
    at_goal = [t for t in sorted(frames) if abs(truth[t]['cut_length_m']-goal) <= EPS]
    qualified = [t for t in at_goal
                 if frames[t]['status'] == wanted and during_reacquired_bite(t) is not None]
    pairs = [[a, b] for a, b in zip(qualified, qualified[1:])
             if b == a+1 and during_reacquired_bite(a) == during_reacquired_bite(b)
             and np.allclose(frames[a]['camera'], frames[b]['camera'], rtol=0, atol=1e-12)]
    strict = (run['strict_truth'] is True and 'untagged' not in run['truth_purposes']
              and not any(e['event'] == 'perception_error' for e in run['events']))
    direct = run['checks']['direct']
    result.update(
        start_mm=trace[0]['cut_length_m'] * 1000,
        final_mm=run['final']['cut_length_m'] * 1000,
        peak_mm=max(row['cut_length_m'] for row in trace) * 1000,
        exposed_align_tick=alignment, both_grasps=bool(both), grasped=grasped,
        exposed_ink_px=frames[alignment]['ink_px'] if alignment in frames else None,
        closures=closures, reacquired=reacq, direct_closures=direct, strict_truth=bool(strict),
        unsafe_ticks=unsafe, false_stop_ticks=false_stops,
        actual_closed=run['actual_closed'], actual_ready=run['actual_ready'],
        target_frames=[dict(tick=t, status=frames[t]['status'], ink_px=frames[t]['ink_px'],
                            scissors_valid=frames[t]['scissors_valid']) for t in at_goal],
        qualified_pairs=pairs,
        passed=bool(strict and direct and both and abs(trace[0]['cut_length_m']) <= EPS
                    and not unsafe and not false_stops and pairs))
    return result


def score(plan):
    frozen = OUTPUT / 'predictions.json'
    p = load(frozen)
    if (p['protocol_sha256'] != sha256(PLAN) or p['scorer_sha256'] != sha256(Path(__file__))
            or p['reader_sha256'] != plan['reader_sha256']):
        raise ValueError('predictions are not bound to the current protocol')
    expected = [(a, c) for a in plan['angles_deg'] for c in plan['cases']]
    if [(r['roll_deg'], r['case']) for r in p['trials']] != expected:
        raise ValueError('sealed predictions lack registered trial identities')
    parent = load(SPEC)
    trials = [score_trial(row, plan, parent) for row in p['trials']]
    qualified = [a for a in plan['angles_deg']
                 if all(row['passed'] for row in trials if row['roll_deg'] == a)]
    outcome = dict(protocol_sha256=sha256(PLAN), predictions_sha256=sha256(frozen),
                   trials=trials, passed_roll_deg=qualified, screen_passed=bool(qualified))
    put(OUTPUT / 'decision.json', outcome)
    print('Qualified roll angles:', qualified)


if __name__ == '__main__':
    protocol = registered()
    predict(protocol)
    score(protocol)
