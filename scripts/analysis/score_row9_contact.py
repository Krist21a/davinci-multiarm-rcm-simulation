"""Seal image-only predictions before opening independent structural labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from row9_terminal import measure


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def put(file, data):
    with file.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def predict(folder, cases):
    # Only optical-only records and raw RGB-D are opened before the predictions exist.
    plan = ROOT / 'config/task/m5_row9_contact_dev_v1_1.json'
    predictions = dict(plan_sha256=digest(plan), observer_sha256=digest(ROOT / 'scripts/vision/row9_terminal.py'),
                       cases={})
    for case in cases:
        record = folder / f'{case}_optical.json'
        optical = json.loads(record.read_bytes())
        entries = []
        for row in optical['exposure']:
            tick = row['tick']
            image = folder / f'{case}_frames/{tick:03d}.npz'
            noisy = folder / f'{case}_frames/{tick:03d}_depth.npy'
            with np.load(image) as frame:
                with noisy.open('rb') as stream:
                    depth = np.load(stream)
                decision = measure(frame['rgb'], depth, frame['cam_R'], frame['cam_t'],
                                   frame['f'], row['estimate'])
            entries.append(dict(tick=tick, rgbd_sha256=digest(image), depth_sha256=digest(noisy),
                                **decision))
        predictions['cases'][case] = dict(optical_sha256=digest(record), run_id=optical['run_id'],
                                         registration_sha256=optical['registration_sha256'],
                                         entries=entries)
    put(folder / 'row9_predictions.json', predictions)
    return predictions


def score(folder, predictions):
    # This stage is forbidden until predictions are sealed.
    seal = folder / 'row9_predictions.json'
    if not seal.is_file() or json.loads(seal.read_bytes()) != predictions:
        raise ValueError('predictions were not sealed')
    outcome = dict(plan_sha256=predictions['plan_sha256'],
                   predictions_sha256=digest(seal), cases={})
    for case, optical in predictions['cases'].items():
        run_path = folder / f'{case}.json'
        run = json.loads(run_path.read_bytes())
        if run['case'] != case or run['run_id'] != optical['run_id'] or (
                run['registration_sha256'] != optical['registration_sha256']):
            raise ValueError('run identity mismatch')
        original = json.loads((ROOT / f'validation/task/m5_live_active_look_v1/{case}.json').read_bytes())
        stable = ([(e['tick'], e['event']) for e in run['events'] if e['event'] in ('closure','post_closure_reacquired')]
                  == [(e['tick'], e['event']) for e in original['events']
                      if e['event'] in ('closure','post_closure_reacquired')]
                  and [(s['tick'], s['cut_length_m']) for s in run['observations']]
                  == [(s['tick'], s['cut_length_m']) for s in original['observations']])
        truth = {s['tick']: s['cut_length_m'] for s in run['observations']}
        final_closure = [e['tick'] for e in run['events'] if e['event'] == 'closure']
        positive = [x['tick'] for x in optical['entries'] if x['status'] == 'STOP_CANDIDATE'
                    and truth[x['tick']] >= .009 - 1e-12
                    and len(final_closure) >= 3 and x['tick'] > final_closure[2]]
        wrong = [x['tick'] for x in optical['entries'] if x['status'] == 'STOP_CANDIDATE'
                 and truth[x['tick']] < .009 - 1e-12]
        eight = [x for x in optical['entries'] if .0075 <= truth[x['tick']] < .009 - 1e-12]
        outcome['cases'][case] = dict(report_sha256=digest(run_path), trace_identical=stable,
                                     false_stop_ticks=wrong, positive_9_ticks=positive,
                                     observed_8_ticks=[x['tick'] for x in eight if x['status'] != 'UNKNOWN'],
                                     eight_unknown_ticks=[x['tick'] for x in eight if x['status'] == 'UNKNOWN'],
                                     actual_mm=run['final']['cut_length_m']*1000)
    nominal = outcome['cases']['nominal']
    noise = outcome['cases']['depth_noise_0p5']
    outcome['screen_passed'] = bool(all(v['trace_identical'] and not v['false_stop_ticks']
                                        for v in outcome['cases'].values())
                                    and len(noise['observed_8_ticks']) >= 2
                                    and any(b == a + 1 for a, b in zip(nominal['positive_9_ticks'],
                                                                        nominal['positive_9_ticks'][1:])))
    put(folder / 'row9_decision.json', outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    folder = parser.parse_args().folder
    plan = json.loads((ROOT / 'config/task/m5_row9_contact_dev_v1_1.json').read_bytes())
    decision = score(folder, predict(folder, plan['cases']))
    print('row9 optical screen:', decision['screen_passed'], decision['cases'])
    return int(not decision['screen_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
