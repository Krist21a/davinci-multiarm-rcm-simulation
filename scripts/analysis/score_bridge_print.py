"""Seal RGB-D bridge observations before reading independent structural length."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from bridge_observer import measure as measure_intact
from bridge_fracture import measure as measure_fracture

PLAN = ROOT / 'config/task/m5_visible_fracture_dev_v4.json'


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def put(file, obj):
    with file.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(obj, out, indent=2, allow_nan=False)
        out.write('\n')


def predict(folder, protocol):
    plan = json.loads(protocol.read_bytes())
    observer = 'bridge_fracture.py' if plan['sheet'].get('bridge_fracture_faces') else 'bridge_observer.py'
    measure = measure_fracture if observer == 'bridge_fracture.py' else measure_intact
    result = dict(protocol_sha256=sha(protocol),
                  observer_sha256=sha(ROOT / 'scripts/vision' / observer),
                  cases={})
    for case in plan['cases']:
        source = folder / (case + '_optical.json')
        optical = json.loads(source.read_bytes())
        if optical['case'] != case or optical['plan_sha256'] != sha(protocol):
            raise ValueError('incorrect optical record')
        frames = []
        for entry in optical['exposures']:
            tick = entry['tick']
            raw = folder / f'{case}_frames/{tick:03d}.npz'
            noisy = folder / f'{case}_frames/{tick:03d}_depth.npy'
            with np.load(raw) as frame, noisy.open('rb') as file:
                observed = np.load(file)
                reading = measure(frame['rgb'], observed, frame['cam_R'],
                                  frame['cam_t'], float(frame['f']), entry['estimate'])
            frames.append(dict(tick=tick, image_sha256=sha(raw), noise_sha256=sha(noisy),
                               **reading))
        result['cases'][case] = dict(optical_sha256=sha(source),
                                     registration_sha256=optical['registration_sha256'],
                                     run_id=optical['run_id'], frames=frames)
    put(folder / 'bridge_predictions.json', result)
    return result


def score(folder, predictions, protocol):
    frozen = folder / 'bridge_predictions.json'
    if json.loads(frozen.read_bytes()) != predictions:
        raise ValueError('predictions were not frozen')
    plan = json.loads(protocol.read_bytes())
    decision = dict(protocol_sha256=sha(protocol), predictions_sha256=sha(frozen), cases={})
    for case, predicted in predictions['cases'].items():
        report_file = folder / (case + '.json')
        run = json.loads(report_file.read_bytes())
        if (run['case'] != case or run['run_id'] != predicted['run_id']
                or run['registration_sha256'] != predicted['registration_sha256']
                or run['config'].get('sheet') != plan['sheet']
                or run['config']['cut']['control_source'] != 'vision'
                or not run['strict_truth']):
            raise ValueError('incorrect structural run identity or truth isolation')
        truth = {s['tick']: s for s in run['observations']}
        if set(p['tick'] for p in predicted['frames']) != set(truth):
            raise ValueError('missing or misplaced optical exposure')
        closings = [e['tick'] for e in run['events'] if e['event'] == 'closure']
        start = closings[2] if len(closings) >= 3 else float('inf')
        stop = closings[3] if len(closings) >= 4 else float('inf')
        goal = .009 if case == 'nominal' else .008
        wanted = 'STOP_CANDIDATE' if case == 'nominal' else 'NOT_STOP'
        matching = [p for p in predicted['frames'] if abs(truth[p['tick']]['cut_length_m']-goal)<1e-12]
        qualified = [p['tick'] for p in matching if p['status'] == wanted and
                     start < p['tick'] < stop]
        false_stops = [p['tick'] for p in predicted['frames']
                       if p['status'] == 'STOP_CANDIDATE'
                       and truth[p['tick']]['cut_length_m'] < .009 - 1e-12]
        unsafe = [s['tick'] for s in run['observations']
                  if s['torn'] or s['faults'] or s['cut_length_m'] > .010 + 1e-12]
        decision['cases'][case] = dict(report_sha256=sha(report_file),
                                       final_state=run.get('final_state'),
                                       actual_mm=run['final']['cut_length_m']*1000,
                                       start_mm=run['observations'][0]['cut_length_m']*1000,
                                       closures=closings, qualified=qualified,
                                       false_stop_ticks=false_stops, unsafe_ticks=unsafe,
                                       unknown_at_goal=[p['tick'] for p in matching
                                                        if p['status'] == 'UNKNOWN'],
                                       strictly_isolated=run['strict_truth'])
    nominal, noise = decision['cases']['nominal'], decision['cases']['depth_noise_0p5']
    adjacent = lambda ticks: any(b == a+1 for a,b in zip(ticks,ticks[1:]))
    decision['screen_passed'] = bool(
        all(not row['false_stop_ticks'] and not row['unsafe_ticks'] and row['start_mm'] == 0
            for row in (nominal, noise)) and adjacent(nominal['qualified']) and
        adjacent(noise['qualified']))
    put(folder / 'bridge_decision.json', decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', required=True, type=Path)
    parser.add_argument('--protocol', choices=('v4', 'v5'), default='v4')
    args = parser.parse_args()
    folder = args.folder
    protocol = PLAN if args.protocol == 'v4' else PLAN.with_name('m5_visible_fracture_dev_v5.json')
    decision = score(folder, predict(folder, protocol), protocol)
    print('row-8 physical-bridge optical screen:', decision)
    return int(not decision['screen_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
