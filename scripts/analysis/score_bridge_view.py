"""Seal same-pose physical-bridge predictions before reading the 8 mm label."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from bridge_observer import measure

PLAN = ROOT / 'config/task/m5_bridge_view_negative_v1.json'


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def write(file, obj):
    with file.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(obj, out, indent=2, allow_nan=False)
        out.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    folder = parser.parse_args().folder
    optical_file = folder / 'optical.json'
    optical = json.loads(optical_file.read_bytes())
    if optical['plan_sha256'] != sha(PLAN) or len(optical['entries']) != 2:
        raise ValueError('wrong optical checkpoint')
    predictions = []
    for entry in optical['entries']:
        image = folder / f"view_{entry['exposure']}.npz"
        with np.load(image) as frame:
            reading = measure(frame['rgb'], frame['observed_depth'],
                              frame['cam_R'], frame['cam_t'], frame['f'], entry['estimate'])
        predictions.append(dict(exposure=entry['exposure'], rgbd_sha256=sha(image), **reading))
    frozen = dict(protocol_sha256=sha(PLAN), observer_sha256=sha(ROOT/'scripts/vision/bridge_observer.py'),
                  optical_sha256=sha(optical_file), run_id=optical['run_id'], readings=predictions)
    write(folder/'predictions.json', frozen)

    label_file = folder/'labels.json'
    labels = json.loads(label_file.read_bytes())
    prior = json.loads((ROOT/'validation/task/m5_visible_fracture_dev_v4/depth_noise_0p5.json').read_bytes())
    positive = json.loads((ROOT/'validation/task/m5_visible_fracture_dev_v4/bridge_decision.json').read_bytes())
    old_closures = [e['tick'] for e in prior['events'] if e['event']=='closure'][:3]
    old_reacquired = [e['tick'] for e in prior['events']
                      if e['event']=='post_closure_reacquired' and e['tick']<=old_closures[-1]]
    prefix = optical['prefix']
    same = (prefix['closure_ticks']==old_closures and prefix['reacquired_ticks']==old_reacquired
            and prefix['controller_sha256']==prior['controller_sha256'])
    goal = json.loads(PLAN.read_bytes())['view']
    view = np.allclose(labels['camera_target'],goal,rtol=0,atol=1e-6) and (
        np.allclose(labels['actual_camera'],goal,rtol=0,atol=1e-6))
    positive_ticks = positive['cases']['nominal']['qualified']
    passed = bool(labels['run_id']==optical['run_id'] and labels['strict']
                  and abs(labels['structural_length_m']-.008)<1e-12
                  and view and same and all(r['status']=='NOT_STOP' for r in predictions)
                  and any(b==a+1 for a,b in zip(positive_ticks,positive_ticks[1:])))
    result = dict(predictions_sha256=sha(folder/'predictions.json'),
                  labels_sha256=sha(label_file), actual_mm=labels['structural_length_m']*1000,
                  view_landed=bool(view), mechanical_prefix_identical=same,
                  readings=[r['status'] for r in predictions], nominal9_positive=positive_ticks,
                  screen_passed=passed)
    write(folder/'decision.json', result)
    print('bridge same-pose 8/9 screen:', result)
    return int(not passed)


if __name__=='__main__':
    raise SystemExit(main())
