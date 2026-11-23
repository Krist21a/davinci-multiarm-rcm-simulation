"""Sealed optical-only 8 mm HD negative before structural scoring; no control permit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from row9_terminal import measure


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, record):
    with path.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    folder = parser.parse_args().folder
    plan_file = ROOT / 'config/task/m5_row9_hd_dev_v1_1.json'
    optical_file = folder / 'hd_optical.json'
    observer = json.loads(optical_file.read_bytes())
    if observer['plan_sha256'] != sha(plan_file):
        raise ValueError('not the registered optical probe')
    readings = []
    for row in observer['exposures']:
        tick = row['tick']
        raw_file = folder / f'hd_exposures/{tick:03d}.npz'
        with np.load(raw_file) as frame:
            result = measure(frame['rgb'], frame['observed_depth'], frame['cam_R'],
                             frame['cam_t'], frame['f'], row['estimate'])
        readings.append(dict(tick=tick, rgbd_sha256=sha(raw_file), **result))
    predictions = dict(plan_sha256=sha(plan_file), optical_sha256=sha(optical_file),
                       observer_sha256=sha(ROOT / 'scripts/vision/row9_terminal.py'),
                       run_id=observer['run_id'], readings=readings)
    put(folder / 'hd_predictions.json', predictions)

    # The full live report and its hidden structure are accessed only here.
    run_file = folder / 'depth_noise_0p5.json'
    run = json.loads(run_file.read_bytes())
    previous = json.loads((ROOT / 'validation/task/m5_live_active_look_v1/depth_noise_0p5.json').read_bytes())
    if run['run_id'] != observer['run_id'] or run['registration_sha256'] != observer['registration_sha256']:
        raise ValueError('unmatched run identity')
    actions = lambda a: [(e['tick'], e['event']) for e in a['events']
                         if e['event'] in ('closure', 'post_closure_reacquired')]
    mechanical_identity = (actions(run) == actions(previous)
                           and [(s['tick'], s['cut_length_m'], s['faults'], s['torn'])
                                for s in run['observations']]
                           == [(s['tick'], s['cut_length_m'], s['faults'], s['torn'])
                               for s in previous['observations']])
    length = {s['tick']: s['cut_length_m'] for s in run['observations']}
    eight = [q for q in readings if .0075 <= length[q['tick']] < .009 - 1e-12]
    bad = [q['tick'] for q in readings if q['status'] == 'STOP_CANDIDATE'
           and length[q['tick']] < .009 - 1e-12]
    original = json.loads((ROOT / 'validation/task/m5_row9_contact_dev_v1_1_capture/row9_decision.json').read_bytes())
    nominal = original['cases']['nominal']
    passed = (mechanical_identity and not bad and
              sum(q['status'] == 'NOT_STOP' for q in eight) >= 2
              and any(b == a+1 for a, b in zip(nominal['positive_9_ticks'],
                                                 nominal['positive_9_ticks'][1:])))
    decision = dict(plan_sha256=sha(plan_file), predictions_sha256=sha(folder/'hd_predictions.json'),
                    report_sha256=sha(run_file), mechanically_identical=mechanical_identity,
                    stopped_8_ticks=bad, visible_not_stop_8_ticks=[q['tick'] for q in eight
                                                                    if q['status'] == 'NOT_STOP'],
                    unknown_8_ticks=[q['tick'] for q in eight if q['status'] == 'UNKNOWN'],
                    nominal_positive_9_ticks=nominal['positive_9_ticks'], optical_screen_passed=bool(passed))
    put(folder / 'hd_decision.json', decision)
    print('same-frame HD 8/9 screen:', decision)
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
