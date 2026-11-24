"""Score registered two-stage support tests, locking RGB-D predictions before joining labels."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import ndimage

from capture_seam_rail import OUT, PLAN, ROOT, SOURCES, sha256, write_new
import score_surface_seam as parent
import ruler_readout as rr


def sensor(path, scale):
    with np.load(path, allow_pickle=False) as f:
        if set(f.files) != {'rgb', 'depth'}:
            raise ValueError('unblinded frame')
        rgb, z = f['rgb'], f['depth']
    if (rgb.shape != (scale*480, scale*640, 3) or rgb.dtype != np.uint8 or
            z.shape != rgb.shape[:2] or not np.isfinite(z).all() or z.min() <= 0):
        raise ValueError('invalid RGB-D readback')
    seed = int(hashlib.sha256(z.tobytes()).hexdigest()[:16], 16)
    noisy = z + np.random.default_rng(seed).normal(0, .0005, z.shape)
    native, perturbed = rr.classify(rgb, z), rr.classify(rgb, noisy)
    lab = rr.inks(rgb)
    ys, xs = np.where(lab >= 0)
    # Restrict label/adjacency counting to the actual print, without resampling.
    if len(ys):
        lab = lab[max(0, ys.min()-2):ys.max()+3, max(0, xs.min()-2):xs.max()+3]
    kernel = np.ones((3, 3), bool)
    areas, boundaries = [], []
    for side in (0, 13):
        comps, adjacency = [], []
        for j in range(7, 11):
            mask = lab == side+j
            components, n = ndimage.label(mask, kernel)
            comps.append(int(np.bincount(components[mask]).max()) if n else 0)
            if j > 7:
                adjacency.append(int(np.count_nonzero(mask & ndimage.binary_dilation(lab == side+j-1, kernel))))
        areas.append(comps)
        boundaries.append(adjacency)
    cross = [rr.crossings(rgb, dep) for dep in (z, noisy)]
    terminal = [[x['state'] for x in v if x['j'] in (8, 9, 10)] for v in cross]
    ok = (min(map(min, areas)) >= 4 and min(map(min, boundaries)) >= 4
          and all(len(states) == 3 and 'open' not in states for states in terminal))
    return dict(sensor_sha256=sha256(path), native=native, noisy=perturbed, patches_7_10=areas,
                adjacent_8_10=boundaries, terminal_crossings=terminal, uncut_ink_gate=ok)


def score(stage, folder):
    spec = json.loads(PLAN.read_bytes())
    hashes = {s: sha256(ROOT/s) for s in SOURCES}
    if hashes['scripts/vision/ruler_readout.py'] != spec['reader_sha256']:
        raise ValueError('reader drift')
    preflight = stage == 'preflight'
    if preflight:
        rails, entries = list(spec['columns']), [spec['development_runs'][0]]
    else:
        select = json.loads((folder/'selection.json').read_bytes())
        if select['spec_sha256'] != sha256(PLAN) or select['source_sha256'] != hashes or not select['selected']:
            raise ValueError('working data not authorized')
        rails, entries = [select['selected']['rail']], spec['development_runs']
    dest = folder/('selection.json' if preflight else 'decision.json')
    blind = folder/(stage+'_predictions.json')
    if dest.exists() or blind.exists():
        raise FileExistsError('never replace frozen evidence')
    paths = [p for rail in rails for e in entries for kind in ('sensor', 'high')
             for p in (folder/rail/e['id']/kind).glob('*.npz')]
    pred = {1: {}, 4: {}}
    for p in sorted(paths, key=lambda p: p.stem):
        scale = 1 if p.parent.name == 'sensor' else 4
        if p.stem in pred[scale]:
            raise ValueError('duplicate token')
        pred[scale][p.stem] = sensor(p, scale)
    shutter = rr.classify(np.full((480, 640, 3), 96, np.uint8), np.full((480, 640), .020))
    write_new(blind, dict(predictions=pred, protocol_sha256=sha256(PLAN), source_sha256=hashes, shutter=shutter))
    runs, passing, seen, high_seen, controllers = [], [], set(), set(), {}
    for rail in rails:
        for e in entries:
            src = folder/rail/e['id']/'oracle.json'
            r = json.loads(src.read_bytes())
            if r['rail'] != rail:
                raise ValueError('unexpected support')
            if r.get('final_state') == 'DONE':
                left = spec['columns'][rail]
                if r['pins'] != [24*j+i for j in range(6, 13) for i in (left, 23-left)]:
                    raise ValueError('support readback mismatch')
            if r.get('controller_sha256'):
                controllers.setdefault(rail, set()).add(r['controller_sha256'])
            with patch.object(parent, 'OUTPUT', folder/rail), patch.object(parent, 'PLAN', PLAN):
                result = parent.run_score(e, pred[1], seen, spec, hashes)
            result['rail'] = rail
            for f in r['frames']:
                token = f['key']
                if 'high_sha256' in f:
                    if f['high_sha256'] != pred[4][token]['sensor_sha256']:
                        raise ValueError('high exposure mismatch')
                    high_seen.add(token)
            if preflight:
                for scale in (1, 4):
                    observations = [pred[scale].get(f['key']) for f in r['frames']]
                    if (result['mechanics_ok'] and len(observations) == 2 and shutter['decision'] == 'UNKNOWN'
                            and all(x and x['uncut_ink_gate'] and
                                    all(x[k]['decision'] == 'UNKNOWN' for k in ('native', 'noisy')) for x in observations)):
                        passing.append((scale, rails.index(rail), rail))
            else:
                scale = select['selected']['scale']
                if r['scale'] != (None if e.get('pull_only') else scale):
                    raise ValueError('acquisition scale drift')
                for f in result['rows']:
                    x = pred[scale][f['key']]
                    decisions = [x[k]['decision'] for k in ('native', 'noisy')]
                    mm, phase = f['actual_mm'], f['phase']
                    wanted = 'STOP' if mm >= 9 else 'BELOW'
                    if phase in ('loaded', 'pulled'):
                        ok = decisions == ['UNKNOWN']*2
                    elif phase == 'work' and mm in (8, 9):
                        ok = decisions == [wanted]*2
                    else:
                        ok = all(d in (wanted, 'UNKNOWN') for d in decisions)
                    f.update(selected_decisions=decisions, selected_optical_ok=ok)
            runs.append(result)
    if (seen != set(pred[1]) or high_seen != set(pred[4]) or any(len(s) != 1 for s in controllers.values())
            or len({r['run_id'] for r in runs}) != len(runs)):
        raise ValueError('inconsistent trajectories')
    record = dict(spec_sha256=sha256(PLAN), source_sha256=hashes, blind_sha256=sha256(blind), runs=runs,
                  confirmation_run=False, visual_control=False)
    if preflight:
        best = min(passing, default=None)
        record['selected'] = dict(rail=best[2], scale=best[0]) if best else None
        record['passing'] = [dict(rail=r, scale=s) for s, _, r in sorted(passing)]
        passed = bool(best)
    else:
        cover = {mm: sorted({r['case'] for r in runs if r['mechanics_ok'] for f in r['rows']
                             if f['phase'] == 'work' and f['actual_mm'] == mm}) for mm in (8, 9)}
        good = all(r['mechanics_ok'] and all(f['selected_optical_ok'] for f in r['rows']) for r in runs)
        record.update(coverage=cover, selected=select['selected'], selection_sha256=sha256(folder/'selection.json'))
        passed = good and any(a != b for a in cover[8] for b in cover[9]) and shutter['decision'] == 'UNKNOWN'
    write_new(dest, dict(passed=passed, **record))
    print(stage, passed, record['selected'])
    return int(not passed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', required=True, choices=('preflight', 'work'))
    ap.add_argument('--out-root', type=Path, default=OUT)
    opts = ap.parse_args()
    raise SystemExit(score(opts.stage, opts.out_root))
