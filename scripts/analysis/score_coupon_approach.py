"""Seal RGB-D evidence before separately scoring the guarded coupon approach."""
import argparse
from bisect import bisect_left
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task'))
from run_coupon_approach import OUTPUT, PLAN, REPO, folder, registered
from run_same_bite import SPEC, put, sha256
from score_coupon_roll import ink_counts, measure

EPS = 1e-12


def load(path):
    return json.loads(path.read_bytes())


def predict(phase, trials):
    rows = []
    for angle, case in trials:
        dest = folder(phase, angle)
        path = dest / f'{case}_optical.json'
        entry = dict(angle=angle, case=case)
        rows.append(entry)
        if not path.is_file():
            continue
        optical = load(path)
        reg = dest / 'registration.json'
        contents = load(reg)
        if (tuple(optical[k] for k in ('phase', 'roll_deg', 'case')) != (phase, angle, case)
                or optical['registration_sha256'] != sha256(reg)
                or contents['policy'] != f'coupon_approach_{phase}'
                or contents['sources'].get(PLAN.relative_to(REPO).as_posix()) != sha256(PLAN)
                or any(sha256(REPO / p) != h for p, h in contents['sources'].items())):
            raise ValueError(f'invalid optical registration: {phase}/{angle}/{case}')
        entry.update(optical_sha256=sha256(path), run_id=optical['run_id'],
                     registration_sha256=sha256(reg), frames=[])
        for x in optical['exposures']:
            tick, est = x['tick'], x['estimate']
            image = dest / f'{case}_frames/{tick:03d}.npz'
            noisy = image.with_name(f'{tick:03d}_depth.npy')
            cut = est.get('cut') or {}
            chart = cut.get('material_chart') or {}
            with np.load(image) as data:
                d = np.load(noisy)
                rgb, R, t, f = (data[k] for k in ('rgb', 'cam_R', 'cam_t', 'f'))
                row = dict(tick=tick, image_sha256=sha256(image), noisy_sha256=sha256(noisy),
                           mode=cut.get('mode'), front_valid=bool(cut.get('front_valid')),
                           scissors_valid=bool((est.get('scissors') or {}).get('valid')),
                           tissue_valid=bool((est.get('tissue') or {}).get('valid')),
                           material_9=bool(chart.get('valid') and any(
                               abs(k-.009) <= EPS for k in chart.get('material_knots_m', []))),
                           boundary_pairs=chart.get('boundary_pairs'), ink_px=ink_counts(rgb, d),
                           camera=[*rgb.shape[:2], float(f), *R.ravel(), *t])
                if phase == 'full':
                    row['reading'] = measure(rgb, d, R, t, float(f), est)
            entry['frames'].append(row)
    dest = OUTPUT / phase / 'predictions.json'
    put(dest, dict(protocol_sha256=sha256(PLAN), phase=phase, trials=rows))
    return dest


def same_camera(a, b):
    return np.allclose(a['camera'], b['camera'], rtol=0, atol=EPS)


def score_trial(phase, entry, roll, cases):
    angle, case = entry['angle'], entry['case']
    path = folder(phase, angle) / f'{case}.json'
    result = dict(angle=angle, case=case, passed=False)
    if not path.is_file():
        return dict(result, error='missing task report')
    task = load(path)
    result.update(report_sha256=sha256(path), final_state=task.get('final_state'),
                  reason=task.get('error') or task.get('reason'))
    if task.get('error') or task.get('cleanup_error') or 'frames' not in entry:
        return dict(result, error=task.get('error') or task.get('cleanup_error') or 'missing exposures')
    cfg = task['config']
    scenario, sigma, seed = cases[case]
    if (task['case'] != case or task['scenario'] != scenario or task['run_id'] != entry['run_id']
            or task['registration_sha256'] != entry['registration_sha256']
            or task['policy'] != f'coupon_approach_{phase}'
            or cfg['sheet'] != dict(roll['sheet'], roll_deg=angle)
            or (cfg['perception']['depth_noise_m'], cfg['perception']['seed']) != (sigma, seed)
            or cfg['cut']['control_source'] != 'vision'):
        raise ValueError(f'incorrect structural run: {phase}/{angle}/{case}')
    trace = task['observations']
    obs = {o['tick']: o for o in trace}
    frames = {f['tick']: f for f in entry['frames']}
    if len(trace) != len(obs) or len(frames) != len(entry['frames']) or set(obs) != set(frames):
        raise ValueError('optical and structural samples do not match one-to-one')
    events = task['events']
    unsafe = [o['tick'] for o in trace if o['torn'] or o['faults']
              or not 0 <= o['cut_length_m'] <= .010 + EPS]
    strict = (task['strict_truth'] is True and 'untagged' not in task['truth_purposes']
              and not any(e['event'] == 'perception_error' for e in events))
    checks = task['checks']
    result.update(strict_truth=bool(strict), start_mm=checks['start_m']*1000,
                  peak_mm=checks['peak_m']*1000, final_mm=task['final']['cut_length_m']*1000,
                  unsafe_ticks=unsafe, closures=checks['closures'], reacquired=checks['reacquired'],
                  direct_closures=checks['direct'], coverage_sequence=checks['sequence'],
                  actual_closed=task['actual_closed'], actual_ready=task['actual_ready'])
    roles = [cfg['roles'][key] for key in ('retract_primary', 'retract_assist')]

    def held(o):
        return all(o['grasped'].get(a, 0) >= cfg['grasp']['min_nodes'] for a in roles)

    start = next((x['world_tick'] for x in task['log']
                  if x.get('frm') == 'EXPOSE' and x.get('to') == 'ALIGN'
                  and held(obs[x['world_tick']])), None)
    result.update(dual_held_align_tick=start, loaded_ink_px=None if start is None else frames[start]['ink_px'])
    if phase == 'short':
        checkpoint = task['dry_checkpoint']
        advance = None if checkpoint is None else checkpoint['advance']
        cutoff = max(obs) if advance is None else advance['tick']
        mat = [t for t in sorted(frames)
               if start is not None and start <= t <= cutoff and frames[t]['material_9']
               and frames[t]['scissors_valid'] and frames[t]['tissue_valid'] and held(obs[t])
               and cfg['tension']['low_strain'] <= obs[t]['tension'] <= cfg['tension']['high_strain']]
        image = frames[checkpoint['tick']] if checkpoint else {}
        current = obs[checkpoint['tick']] if checkpoint else None
        zero_damage = (not checks['closures'] and task['tissue'].get('cut') == 0
                       and all(abs(o['cut_length_m']) <= EPS for o in trace))
        reached = bool(checkpoint and advance and advance['accepted'] and advance['forward_m'] > EPS
                       and advance['tick'] < checkpoint['tick'] == checkpoint['fix_tick']
                       and checkpoint['closure_aligned'] and not checkpoint['awaiting_new_slit']
                       and checkpoint['remaining_sweep_m'] is not None
                       and checkpoint['remaining_sweep_m'] >= -EPS
                       and len(checkpoint['jaw_angles']) == 2
                       and all(abs(q) > 1e-6 for q in checkpoint['jaw_angles'])
                       and not task['actual_closed'] and held(current)
                       and current['cut_site_visible'] and current['scissors_visible']
                       and image['front_valid'] and image['scissors_valid'] and image['tissue_valid']
                       and image['mode'] == 'pre_cut')
        result.update(checkpoint=checkpoint, reached_guard=bool(checkpoint), verified_approach=reached,
                      material_9_ticks=mat, zero_damage=zero_damage,
                      advance_attempts=[e for e in events if e['event'] == 'probe_advance'],
                      passed=bool(reached and mat and zero_damage and strict and not unsafe))
    else:
        closures, reacquired = checks['closures'], checks['reacquired']

        def observed_bite(tick):
            i = bisect_left(closures, tick) - 1
            return i if i >= 0 and i < len(reacquired) and closures[i] < reacquired[i] <= tick else None

        goal, wanted = (9, 'STOP_CANDIDATE') if case == 'nominal' else (8, 'NOT_STOP')
        at_goal = [t for t in sorted(frames) if abs(obs[t]['cut_length_m'] - goal/1000) <= EPS]
        qualified = [t for t in at_goal if observed_bite(t) is not None
                     and frames[t]['reading']['status'] == wanted]
        pairs = [[a, b] for a, b in zip(qualified, qualified[1:])
                 if b == a+1 and observed_bite(a) == observed_bite(b)
                 and same_camera(frames[a], frames[b])]
        false = [t for t in sorted(frames) if obs[t]['cut_length_m'] < .009-EPS
                 and frames[t]['reading']['status'] == 'STOP_CANDIDATE']
        result.update(actual_target_ticks=at_goal, optical_pairs=pairs, false_stops=false,
                      passed=bool(strict and start is not None and checks['direct'] and pairs
                                  and abs(checks['start_m']) <= EPS and not unsafe and not false))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('short', 'full'))
    phase = parser.parse_args().phase
    plan, roll = registered()
    angles = (plan['angles_deg'] if phase == 'short' else
              [load(OUTPUT / 'short' / 'decision.json')['selected_roll_deg']])
    dest = OUTPUT / phase
    launch = load(dest / 'launch.json')
    trials = [(a, c) for a in angles for c in plan['cases']]
    if (load(dest / 'protocol.json') != plan or launch['protocol_sha256'] != sha256(PLAN)
            or [(x['angle'], x['case']) for x in launch['trials']] != trials):
        raise ValueError('launcher did not attempt every planned case')
    frozen = predict(phase, trials)
    data = load(frozen)
    ids = [x['run_id'] for x in data['trials'] if 'run_id' in x]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate run id')
    cases = load(SPEC)['cases']
    out = dict(protocol_sha256=sha256(PLAN), predictions_sha256=sha256(frozen),
               trials=[score_trial(phase, x, roll, cases) for x in data['trials']])
    eligible = [a for a in angles if all(x['passed'] for x in out['trials'] if x['angle'] == a)]
    if phase == 'short':
        shortlisted = sorted((a for a in eligible if a), key=lambda a: (abs(a), -a))
        out.update(control_passed=0 in eligible, nonzero_passing=shortlisted,
                   selected_roll_deg=shortlisted[0] if 0 in eligible and shortlisted else None)
    else:
        out.update(selected_roll_deg=angles[0], optical_screen_passed=bool(eligible))
    put(dest / 'decision.json', out)
    print(f'{phase}: passing angles {eligible}; selected {out["selected_roll_deg"]}')


if __name__ == '__main__':
    main()
