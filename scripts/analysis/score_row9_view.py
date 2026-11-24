"""Freeze both HD predictions before opening the separate 8 mm structural label."""
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


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    path = parser.parse_args().folder
    plan = ROOT / 'config/task/m5_row9_view_dev_v1.json'
    optical = json.loads((path/'optical.json').read_bytes())
    if optical['plan_sha256'] != sha(plan) or len(optical['entries']) != 2:
        raise ValueError('unregistered or incomplete optical capture')
    candidates = []
    for row in optical['entries']:
        file = path / f"view_{row['exposure']}.npz"
        with np.load(file) as frame:
            reading = measure(frame['rgb'], frame['observed_depth'],
                              frame['cam_R'], frame['cam_t'], frame['f'], row['estimate'])
        candidates.append(dict(exposure=row['exposure'], image_sha256=sha(file), **reading))
    output = dict(plan_sha256=sha(plan), observer_sha256=sha(ROOT/'scripts/vision/row9_terminal.py'),
                  optical_sha256=sha(path/'optical.json'), run_id=optical['run_id'],
                  readings=candidates)
    write(path/'predictions.json', output)

    label = json.loads((path/'labels.json').read_bytes())
    original = json.loads((ROOT/'validation/task/m5_live_active_look_v1/depth_noise_0p5.json').read_bytes())
    positive = json.loads((ROOT/'validation/task/m5_row9_contact_dev_v1_1_capture/row9_decision.json').read_bytes())
    baseline_closures = [e['tick'] for e in original['events'] if e['event']=='closure'][:3]
    baseline_reacq = [e['tick'] for e in original['events']
                      if e['event']=='post_closure_reacquired' and e['tick']<=baseline_closures[-1]]
    same_prefix = (optical['prefix']['closure_ticks'] == baseline_closures
                   and optical['prefix']['reacquired_ticks'] == baseline_reacq
                   and optical['prefix']['controller_sha256'] == original['controller_sha256'])
    valid = (label['run_id']==optical['run_id'] and label['strict'] and
             abs(label['structural_length_m']-.008)<1e-12 and
             all(row['status']=='NOT_STOP' for row in candidates) and
             len(positive['cases']['nominal']['positive_9_ticks'])>=2 and same_prefix)
    decision = dict(predictions_sha256=sha(path/'predictions.json'), labels_sha256=sha(path/'labels.json'),
                    structural_mm=label['structural_length_m']*1000, same_prefix=same_prefix,
                    readings=[c['status'] for c in candidates], screen_passed=bool(valid))
    write(path/'decision.json', decision)
    print('paired 8/9 same-pose optical screen:', decision)
    return int(not valid)


if __name__ == '__main__':
    raise SystemExit(main())
