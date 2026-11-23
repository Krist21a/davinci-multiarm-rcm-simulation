"""Freeze all legal-view RGB-D estimates before joining the structural 8 mm label."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from bridge_fracture import measure

PLAN = ROOT/'config/task/m5_bridge_fracture_view_sweep_v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', type=Path, required=True)
    folder = ap.parse_args().folder
    plan = json.loads(PLAN.read_bytes())
    optical_file = folder/'optical.json'
    optical = json.loads(optical_file.read_bytes())
    if optical['plan_sha256'] != sha(PLAN):
        raise ValueError('unregistered read-only view sweep')
    readings = []
    for exposure in optical['exposures']:
        source = folder/exposure['file']
        with np.load(source) as frame:
            result = measure(frame['rgb'],frame['observed_depth'],frame['cam_R'],
                             frame['cam_t'],float(frame['f']),exposure['estimate'])
        readings.append(dict(pose=exposure['pose'], exposure=exposure['exposure'],
                             image_sha256=sha(source), readback=exposure['readback'], **result))
    frozen = dict(protocol_sha256=sha(PLAN),
                  observer_sha256=sha(ROOT/'scripts/vision/bridge_fracture.py'),
                  optical_sha256=sha(optical_file), run_id=optical['run_id'],
                  refused=optical['refused'], readings=readings)
    put(folder/'predictions.json', frozen)

    labels_file = folder/'labels.json'
    labels = json.loads(labels_file.read_bytes())
    baseline = json.loads((ROOT/'validation/task/m5_visible_fracture_dev_v5/depth_noise_0p5.json').read_bytes())
    positive = json.loads((ROOT/'validation/task/m5_visible_fracture_dev_v5/bridge_decision.json').read_bytes())
    closures = [e['tick'] for e in baseline['events'] if e['event']=='closure'][:3]
    prefix = optical['prefix']
    same = prefix['closure_ticks']==closures and prefix['controller_sha256']==baseline['controller_sha256']
    qualified = []
    for i, target in enumerate(plan['poses']):
        subset = [x for x in readings if x['pose']==i]
        if (len(subset)==2 and all(x['status']=='NOT_STOP' for x in subset)
                and all(np.allclose(x['readback'],target,rtol=0,atol=1e-6) for x in subset)):
            qualified.append(i)
    false_stops = [(r['pose'],r['exposure']) for r in readings if r['status']=='STOP_CANDIDATE']
    positive_ticks = positive['cases']['nominal']['qualified']
    adjacent = any(b==a+1 for a,b in zip(positive_ticks,positive_ticks[1:]))
    passed = bool(labels['run_id']==optical['run_id'] and labels['strict'] and same
                  and abs(labels['structural_length_m']-.008)<1e-12
                  and not labels['tissue']['torn'] and qualified and not false_stops and adjacent)
    decision = dict(predictions_sha256=sha(folder/'predictions.json'),
                    labels_sha256=sha(labels_file), actual_mm=labels['structural_length_m']*1000,
                    pre_checkpoint_identical=same, refused_views=optical['refused'],
                    negative_qualified_poses=qualified, false_stop_frames=false_stops,
                    positive_nominal_ticks=positive_ticks, optical_screen_passed=passed)
    put(folder/'decision.json', decision)
    print('frature-visible view sweep:', decision)
    return int(not passed)


if __name__=='__main__':
    raise SystemExit(main())
