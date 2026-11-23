"""Seal passive-HD bridge readings before opening the structural run report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from bridge_observer import measure

PLAN = ROOT / 'config/task/m5_bridge_hd_negative_v1.json'


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def write(file, obj):
    with file.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', type=Path, required=True)
    folder = ap.parse_args().folder
    optical_file = folder / 'hd_optical.json'
    optical = json.loads(optical_file.read_bytes())
    if optical['plan_sha256'] != sha(PLAN):
        raise ValueError('unregistered HD negative')
    readings = []
    for row in optical['exposures']:
        tick = row['tick']
        image = folder / f'hd_exposures/{tick:03d}.npz'
        with np.load(image) as frame:
            reading = measure(frame['rgb'], frame['observed_depth'],
                              frame['cam_R'], frame['cam_t'], float(frame['f']), row['estimate'])
        readings.append(dict(tick=tick, rgbd_sha256=sha(image), **reading))
    sealed = dict(protocol_sha256=sha(PLAN), observer_sha256=sha(ROOT / 'scripts/vision/bridge_observer.py'),
                  optical_sha256=sha(optical_file), run_id=optical['run_id'], readings=readings)
    write(folder / 'hd_bridge_predictions.json', sealed)

    # Only the independent structural scorer determines the true length here.
    run_file = folder / 'depth_noise_0p5.json'
    report = json.loads(run_file.read_bytes())
    old = json.loads((ROOT/'validation/task/m5_visible_fracture_dev_v4/depth_noise_0p5.json').read_bytes())
    if (report['run_id'] != optical['run_id'] or
            report['registration_sha256'] != optical['registration_sha256'] or
            report['config'].get('sheet') != {'bridge_print': True} or
            not report['strict_truth']):
        raise ValueError('HD/structural identity mismatch')
    events = lambda r: [(e['tick'], e['event']) for e in r['events']
                        if e['event'] in ('closure','post_closure_reacquired')]
    trace = lambda r: [(s['tick'],s['cut_length_m'],s['torn'],s['faults']) for s in r['observations']]
    same = events(report)==events(old) and trace(report)==trace(old)
    truth = {s['tick']: s for s in report['observations']}
    closures = [e['tick'] for e in report['events'] if e['event']=='closure']
    after = closures[2] if len(closures)>2 else float('inf')
    before = closures[3] if len(closures)>3 else float('inf')
    eight = [r for r in readings if abs(truth[r['tick']]['cut_length_m']-.008)<1e-12
             and after<r['tick']<before]
    negative = [r['tick'] for r in eight if r['status']=='NOT_STOP']
    false_stop = [r['tick'] for r in readings if r['status']=='STOP_CANDIDATE'
                  and truth[r['tick']]['cut_length_m'] < .009-1e-12]
    baseline = json.loads((ROOT/'validation/task/m5_visible_fracture_dev_v4/bridge_decision.json').read_bytes())
    positive = baseline['cases']['nominal']['qualified']
    adjacent = lambda xs:any(b==a+1 for a,b in zip(xs,xs[1:]))
    passed = bool(same and not false_stop and adjacent(negative) and adjacent(positive))
    outcome = dict(predictions_sha256=sha(folder/'hd_bridge_predictions.json'),
                   report_sha256=sha(run_file), trace_identical=same,
                   true_8_not_stop_ticks=negative, true_8_unknown_ticks=[r['tick'] for r in eight
                                                                         if r['status']=='UNKNOWN'],
                   false_stop_ticks=false_stop, nominal_9_stop_ticks=positive, screen_passed=passed)
    write(folder/'hd_bridge_decision.json', outcome)
    print('passive HD bridge negative:', outcome)
    return int(not passed)


if __name__=='__main__':
    raise SystemExit(main())
