"""Predict with just ordered RGB/depth arrays, then open the independent structural key."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/vision'))
import ruler_readout

PLAN = ROOT/'config/task/ruler_endpoint_v1.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate(split, out, lock):
    cfg = json.loads(PLAN.read_bytes())
    reader = sha(Path(ruler_readout.__file__))
    if sha(PLAN) != lock['protocol_sha256'] or reader != lock['reader_sha256'] or sha(Path(__file__)) != lock['scorer_sha256']:
        raise ValueError('unregistered detector/scorer/protocol')
    caps = cfg['sets'][split+'_caps_mm']
    folders = [ROOT/'validation/task'/f'ruler_{split[:3]}_c{cap}_g{g}' for cap in caps for g in (6,7,8,9)]
    predictions = {}
    for p in sorted((p for folder in folders for p in folder.glob('*.npz')), key=lambda p: p.name):
        with np.load(p, allow_pickle=False) as f:
            if set(f.files) != {'rgb', 'depth'}:
                raise ValueError('blinding violation')
            rgb, depth = f['rgb'], f['depth']
        rng = np.random.default_rng(int(sha(p)[:12], 16))
        res = []
        for d in (depth, depth+rng.normal(0., .0005, depth.shape)):
            res.append(ruler_readout.classify(rgb, d))
        if p.stem in predictions:
            raise ValueError('duplicate token')
        predictions[p.stem] = dict(sha256=sha(p), raw=res[0], noise=res[1])
    shutter = ruler_readout.classify(np.full((480,640,3), 96, np.uint8), np.full((480,640), .020))
    seen, covered, good = set(), set(), shutter['decision'] == 'UNKNOWN'
    rows, problems, ids, poses = [], [], set(), {}
    for k, folder in enumerate(folders):
        p = folder/'oracle.json'
        r = json.loads(p.read_bytes())
        if (r['goal_mm'] != (6,7,8,9)[k%4] or r['cap_mm'] != caps[k//4] or r['run_id'] in ids
                or r['scene_sha256'] != cfg['scene_sha256'] or sha(ROOT/cfg['scene']) != r['scene_sha256']
                or r['protocol_sha256'] != sha(PLAN) or r['base_sha256'] != cfg['base_sha256']
                or r['base_controller_sha256'] != cfg['base_controller_sha256']
                or not r['controller_sha256'] or any(sha(ROOT/t) != h for t,h in r['source_sha256'].items())
                or r['config']['cut']['control_source'] != 'truth'):
            raise ValueError('invalid collection identity')
        ids.add(r['run_id'])
        if r['error'] or r.get('cleanup_error') or r.get('final_state') != 'DONE':
            good = False
            problems.append([folder.name, r.get('error') or r.get('reason')])
        if sorted(f['phase'] for f in r['frames']) != ['loaded', 'park', 'work']:
            good = False
            problems.append([folder.name, 'incomplete phases'])
        for f in r['frames']:
            key = f['key']
            if key in seen or (pred := predictions[key])['sha256'] != f['sha256']:
                raise ValueError('mismatched optical prediction')
            seen.add(key)
            length, phase = int(round(f['length_m']*1000)), f['phase']
            if abs(f['length_m']*1000-length) > 1e-6 or length > 10:
                raise ValueError('unexpected structural label')
            decisions = [pred[n]['decision'] for n in ('raw','noise')]
            target = 'BELOW' if length < 9 else 'STOP'
            # At a loaded pre-cut there is no slit to see; BELOW cannot authorize
            # a closure, but a STOP is an outright false positive.
            if phase == 'park' and length >= 7:
                covered.add(length)
                ok = decisions == [target, target]
                poses.setdefault(length, set()).add(f['nodes_sha256'])
            elif phase == 'loaded':
                ok = length == 0 and 'STOP' not in decisions
            else:
                ok = all(x in (target, 'UNKNOWN') for x in decisions)
            good &= ok
            rows.append(dict(run=folder.name, sample=key, sha256=f['sha256'], phase=phase, length_mm=length,
                             raw=pred['raw'], noise=pred['noise'], passed=ok))
    if seen != set(predictions):
        raise ValueError('unlabelled optical prediction')
    passed = good and set(range(7,11)) <= covered
    with out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(dict(split=split, lock=lock, passed=passed, coverage=sorted(covered), shutter=shutter,
                       issues=problems, rows=rows, distinct_park_states={k:len(v) for k,v in poses.items()},
                       source_runs={d.name:sha(d/'oracle.json') for d in folders}), f, indent=2, allow_nan=False)
        f.write('\n')
    print(split, 'gate', passed, 'labels', sorted(covered), 'unaccepted', sum(not r['passed'] for r in rows))
    return int(not passed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', required=True, choices=('development', 'confirmation'))
    ap.add_argument('--lock', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    raise SystemExit(evaluate(args.split, args.out, json.loads(args.lock.read_bytes())))
