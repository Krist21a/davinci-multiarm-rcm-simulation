"""Independently score fresh observations, structural increments and ready-pose readback."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from run_two_bite import BASE, PLAN
from simtools import REPO, sha256

TASK = REPO / 'validation/task'


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def optical(est, modes):
    c = est['cut']
    return (est['scissors'].get('valid') is True and c.get('front_valid') is True and c.get('mode') in modes
            and (c.get('material_chart') or {}).get('valid') is True and number(c.get('progress_m'))
            and isinstance(c.get('progress_cell_m'), list) and len(c['progress_cell_m']) == 2)


def score(r, protocol, *, injected_front_read=False):
    case = r['case']
    cfg = json.loads(BASE.read_bytes())
    cfg['cut']['control_source'] = 'vision'
    cfg['perception'].update(protocol['cases'][case], mode='shadow', material_pattern=True, local_material_chart=True,
                             front_recovery=True, predictive_view=True, exclude_tcp_marker=True,
                             recovery_view_after_cut=True, ray_marker_fit=True)
    if (r['protocol'] != protocol or r['protocol_sha256'] != sha256(PLAN) or cfg != r['config']
            or r['scene_sha256'] != protocol['scene_sha256'] or sha256(REPO / protocol['scene']) != r['scene_sha256']
            or sha256(BASE) != protocol['base_config_sha256'] or r['base_config_sha256'] != sha256(BASE)
            or r['error'] or r['cleanup_error'] or r['controller_sha256'] != protocol['controller_sha256']
            or not np.allclose(r['applied_view']['target'], protocol['initial_ecm'], atol=1e-6, rtol=0)
            or set(r['roles']) != set(cfg['roles'].values()) or not all(v['ok'] is True for v in r['roles'].values())
            or not r['source_sha256'] or any(sha256(REPO / k) != h for k, h in r['source_sha256'].items())):
        raise ValueError('invalid run provenance or configuration')
    rows = r['samples']
    ticks = list(range(1, len(rows)+1))
    allowed = {'scoring', 'safety_monitor', 'retractor_planning', 'tension_until_m6', 'grasp_signal_and_safety_monitor'}
    if injected_front_read:
        if r['truth_purposes'].get('event_injection') != dict(cut_front=1, nodes=1, cut_length=1):
            raise ValueError('unexpected truth access by the occluder harness')
        allowed.add('event_injection')
    if (not ticks or [x['tick'] for x in rows] != ticks or [x['tick'] for x in r['perception']] != ticks
            or r['strict_truth'] is not True or not r['truth_purposes'] or
            set(r['truth_purposes']) - allowed):
        raise ValueError('missing observations or invalid truth boundary')
    obs = {x['tick']: x for x in rows}
    est = {x['tick']: x['estimate'] for x in r['perception']}
    events = r['events']
    bites = [e for e in events if e['event'] == 'closure']
    seen = [e for e in events if e['event'] == 'post_closure_reacquired']
    changes = {i: obs[i+1]['cut_length_m']-obs[i]['cut_length_m'] for i in ticks[:-1]}
    all_lengths = [x['cut_length_m'] for x in rows]
    limit = cfg['cut']['length_m'] + cfg['cut']['max_overcut_m']
    safe = (all(number(v) and 0 <= v <= limit+1e-12 for v in all_lengths)
            and all(x['torn'] == 0 and x['faults'] == [] and number(x['tension']) and number(x['max_strain'])
                    for x in rows) and not any(e['event'] == 'perception_error' for e in events))
    pairs = []
    for i, (b, a) in enumerate(zip(bites, seen)):
        c, t = b['tick'], a['tick']
        valid = (c < t < (bites[i+1]['tick'] if i+1 < len(bites) else ticks[-1])
                 and b.get('fix_age') == 0 and b['scissors_valid'] is True
                 and optical(est[c], ('slit', 'bank_slit') if i else ('pre_cut', 'slit', 'bank_slit'))
                 and optical(est[t], ('slit', 'bank_slit'))
                 and obs[c]['cut_site_visible'] and obs[c]['scissors_visible']
                 and number(obs[c]['cut_site_offset']) and obs[c]['cut_site_offset'] <= cfg['view']['center_margin']
                 and all(obs[c]['grasped'].get(arm, 0) >= cfg['grasp']['min_nodes'] for arm in ('psm1', 'psm3'))
                 and a['mode'] == est[t]['cut']['mode'] and a['progress_m'] == est[t]['cut']['progress_m']
                 and a['progress_m'] - est[c]['cut']['progress_m'] >= .00025-1e-12
                 and changes[c] >= .001-1e-12)
        pairs.append(dict(closure=c, reacquisition=t, growth_mm=changes[c]*1000, passed=valid))
    peak, final = max(all_lengths), all_lengths[-1]
    out = dict(state=r['final_state'], reason=r['reason'], peak_mm=peak*1000, final_mm=final*1000,
               closures=len(bites), pairs=pairs, safe=safe, camera_moves=r['camera_moves'],
               torn=max(x['torn'] for x in rows), faults=sorted({f for x in rows for f in x['faults']}),
               clear_view_entries=sum(x.get('to') == 'CLEAR_VIEW' for x in r['log']))
    if case == 'blocked':
        shutters = [e['tick'] for e in events if e['event'] == 'shutter_after_advance']
        hidden = [x['tick'] for x in rows if x['shutter']]
        blocked = (len(shutters) == 1 and shutters[0] in obs and optical(est[shutters[0]], ('pre_cut',))
                   and hidden == ticks[shutters[0]:] and bool(hidden)
                   and all(not obs[t]['cut_site_visible'] and not obs[t]['scissors_visible']
                           and not est[t]['scissors'].get('valid') and not est[t]['cut'].get('front_valid')
                           for t in hidden))
        passed = (safe and blocked and not bites and not seen and peak == 0 and r['bites'] == 0
                  and r['park'] is None and r['final_state'] == 'FAILED'
                  and r['reason'] in ('no camera pose left', 'camera adjustment exhausted', 'clear-and-look exhausted'))
        return dict(out, shutter_after_advance=shutters, hidden_frames=len(hidden),
                    expected_refusal_passed=passed, task_completed=False)
    p = r['park']
    park_ok = False
    if p and pairs:
        ready, after = (np.asarray(p[k], float) for k in ('ready_tcp', 'after_tcp'))
        pos = float(np.linalg.norm(after[:, 3]-ready[:, 3]))
        angle = float(np.rad2deg(np.arccos(np.clip((np.trace(after[:, :3].T @ ready[:, :3])-1)/2, -1, 1))))
        t = p['tick']
        strand = cfg['tension']
        park_ok = (pairs[-1]['reacquisition'] <= t < ticks[-1] and optical(est[t], ('slit', 'bank_slit'))
                   and p['accepted'] is True and p['closed_after'] is True
                   and all(strand['low_strain_after_cut_start'] <= obs[i]['tension'] <= strand['high_strain']
                           for i in range(t, ticks[-1]+1))
                   and pos <= protocol['park_position_tolerance_m'] and angle <= protocol['park_rotation_tolerance_deg']
                   and all(obs[i]['grasped'].get(a, 0) >= cfg['grasp']['min_nodes']
                           for i in range(t, ticks[-1]+1) for a in ('psm1','psm3')))
        out.update(park_tick=t, park_error_mm=pos*1000, park_error_deg=angle)
    passed = (safe and park_ok and len(pairs) == len(bites) == len(seen) == r['bites'] == 2
              and all(p['passed'] for p in pairs) and rows[0]['cut_length_m'] == 0
              and set(t for t, d in changes.items() if abs(d) > 1e-12) == {b['tick'] for b in bites}
              and r['final_state'] == 'DONE' and not any(x['shutter'] for x in rows))
    return dict(out, park_verified=park_ok, task_completed=passed)


def main(out):
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads(PLAN.read_bytes())
    results, ids = {}, set()
    for case in protocol['cases']:
        path = TASK / f'two_bite_v1_{case}.json'
        r = json.loads(path.read_bytes())
        if r['case'] != case or not r['run_id'] or r['run_id'] in ids:
            raise ValueError('mismatched or duplicate trial')
        ids.add(r['run_id'])
        results[case] = dict(score(r, protocol), report_sha256=sha256(path), run_id=r['run_id'])
    truth, ref = (TASK / name for name in ('two_bite_v1_truth.json','m5b_v1_10_marked_truth.json'))
    t, historic = (json.loads(p.read_bytes()) for p in (truth, ref))
    matched = (t['control_source'] == 'truth' and t['final_state'] == 'DONE' and t['success'] is True
               and t['controller_sha256'] == protocol['controller_sha256'] and t['scene_sha256'] == protocol['scene_sha256']
               and t['trace'] == historic['trace'] and len(t['trace']) == 17
               and t['final_observation']['cut_length_m'] == .010 and t['final_observation']['faults'] == []
               and t['final_observation']['torn'] == 0)
    passed = matched and all(results[c]['task_completed'] for c in ('nominal','noise')) and results['blocked']['expected_refusal_passed']
    report = dict(protocol=protocol, scorer_sha256=sha256(Path(__file__)), cases=results, development_passed=passed,
                  truth=dict(identity=matched, ticks=len(t['trace']), report_sha256=sha256(truth), reference_sha256=sha256(ref)))
    with out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(results, indent=2), 'truth identity:', matched)
    return int(not passed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    raise SystemExit(main(ap.parse_args().out))
