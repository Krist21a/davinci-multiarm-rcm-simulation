"""Blind image predictions first; only then join the mechanical oracle for the pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/vision'))
from ruler_readout import classify, crossings, inks
from capture_surface_seam import OUTPUT, PLAN, SOURCES, write_new
from seam_topology import material_state
from simtools import sha256


def blind_images():
    predictions = {}
    for p in sorted(OUTPUT.glob('*/sensor/*.npz'), key=lambda q: q.stem):
        with np.load(p, allow_pickle=False) as f:
            if set(f.files) != {'rgb', 'depth'} or p.stem in predictions:
                raise ValueError('sensor file not uniquely blinded')
            rgb, z = f['rgb'], f['depth']
        if rgb.shape != (480, 640, 3) or z.shape != rgb.shape[:2] or not np.isfinite(z).all():
            raise ValueError('invalid native RGB-D')
        seed = int(hashlib.sha256(z.tobytes()).hexdigest()[:16], 16)
        noisy = z + np.random.default_rng(seed).normal(0, .0005, z.shape)
        ink = inks(rgb)
        counts = np.bincount(ink.ravel().astype(int)+1, minlength=27)[1:].reshape(2, 13)
        predictions[p.stem] = dict(sensor_sha256=sha256(p), native=classify(rgb, z), noisy=classify(rgb, noisy),
                                   ink=counts.tolist(), crossing_states=[x['state'] for x in crossings(rgb, z)])
    shutter = classify(np.full((480, 640, 3), 96, np.uint8), np.full((480, 640), .02))
    return predictions, shutter


def expected_edges():
    pairs = set()
    for j in range(13):
        for i in range(23):
            a = 24*j+i
            pairs.update(((a, a+1), (a, a+24), (a, a+25), (a+1, a+25), (a+24, a+25)))
            if i != 11:
                pairs.add((a+1, a+24))
    return pairs


def park_errors(ready, after):
    a, b = np.asarray(ready), np.asarray(after)
    if a.shape != (3, 4) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError('invalid TCP readback')
    angle = math.degrees(math.acos(np.clip((np.trace(a[:, :3].T@b[:, :3])-1)/2, -1., 1.)))
    return float(np.linalg.norm(a[:, 3]-b[:, 3])*1000), angle


def run_score(entry, pred, seen, contract, hashes):
    folder = OUTPUT/entry['id']
    path = folder/'oracle.json'
    r = json.loads(path.read_bytes())
    parent = json.loads((ROOT/contract['parent']).read_bytes())
    base = json.loads((ROOT/'config/task/cut_task.json').read_bytes())
    cfg = r['cfg']
    base['cut'].update(length_m=entry['goal_mm']/1000, advance_m=entry['cap_mm']/1000, control_source='truth')
    base['perception'].update(exclude_tcp_marker=True)
    if (r['spec_sha256'] != sha256(PLAN) or r['entry'] != entry or cfg != base or
            r['source_sha256'] != hashes or r['scene_sha256'] != parent['scene_sha256']
            or r['base_sha256'] != parent['base_sha256']):
        raise ValueError('invalid capture identity '+entry['id'])
    issues = []
    if r.get('error') or r.get('cleanup_error') or r.get('final_state') != 'DONE':
        issues.append(r.get('error') or r.get('cleanup_error') or r.get('reason') or 'no terminal state')
    if (r.get('base_controller_sha256') != parent['base_controller_sha256'] or
            not np.allclose((r.get('view') or {}).get('target', [1e3]*3), parent['view'], rtol=0, atol=1e-6)):
        issues.append('base controller or registered camera not verified')
    limit = entry['goal_mm']+1
    obs = r['observations']
    if not obs or any(x['torn'] or x['faults'] or not 0 <= x['cut_length_m']*1000 <= limit+1e-8 for x in obs):
        issues.append('unsafe observation / missing observation history')
    rows, prev, last_work, park, expected = [], set(), 0, None, expected_edges()
    hold = cfg['grasp']['min_nodes']
    for f in r['frames']:
        key = f['key']
        p, st = folder/'sensor'/(key+'.npz'), folder/'structure'/(key+'.npz')
        if (key in seen or pred[key]['sensor_sha256'] != f['sensor_sha256'] or
                sha256(p) != f['sensor_sha256'] or sha256(st) != f['structure_sha256']):
            raise ValueError('sensor/structure manifest mismatch')
        seen.add(key)
        with np.load(st, allow_pickle=False) as data:
            edges, faces, nodes = data['edges'], data['faces'], data['nodes']
        E = {tuple(sorted((a, b))): flag for a, b, flag in edges.tolist()}
        dead = {pair for pair, flag in E.items() if not flag}
        state = material_state(edges)
        mm = state['prefix_rows']
        if (E.keys() != expected or not np.array_equal(state['faces'], faces) or mm != f['prefix_mm']
                or abs(mm-f['obs']['cut_length_m']*1000) > 1e-8 or state['farthest_row_end'] != f['farthest_mm']
                or nodes.shape != (14, 24, 3) or not np.isfinite(nodes).all()
                or not prev <= dead or len(dead) != f['direct_cuts']+f['obs']['torn']+f['releases']):
            raise ValueError('material evidence does not match independent reconstruction')
        prev = dead
        o = f['obs']
        if (o['faults'] or o['torn'] or state['farthest_row_end'] > limit+1e-8 or
                min(o['grasped'].get(k, 0) for k in ('psm1', 'psm3')) < hold):
            issues.append(f"unsafe {f['phase']} frame at tick {f['tick']}")
        decisions = [pred[key][k]['decision'] for k in ('native', 'noisy')]
        phase = f['phase']
        if phase in ('loaded', 'pulled'):
            if mm or dead or f['bites']:
                issues.append('negative control already damaged')
            correct = decisions == ['UNKNOWN']*2
        else:
            target = 'BELOW' if mm < 9 else 'STOP'
            correct = (decisions == [target]*2 if phase == 'work' and mm in (8, 9)
                       else all(d in (target, 'UNKNOWN') for d in decisions))
        if phase == 'work':
            if mm <= last_work or not f['closed'] or not o['scissors_visible']:
                issues.append(f"invalid bite at tick {f['tick']}")
            last_work = mm
        if phase == 'park':
            pos, angle = park_errors(r['ready_tcp'], f['tcp'])
            park = dict(error_mm=pos, error_deg=angle, closed=f['closed'])
            if not (last_work == mm and entry['goal_mm'] <= mm <= limit and pos <= .5 and angle <= 1 and f['closed']):
                issues.append('park or terminal length not verified')
        rows.append(dict(key=key, phase=phase, tick=f['tick'], actual_mm=mm, farthest_mm=f['farthest_mm'],
                         decisions=decisions, optical_ok=correct, sha256=f['sensor_sha256']))
    phases = [x['phase'] for x in rows]
    hits = r.get('events') or []
    if entry.get('pull_only'):
        if (phases != ['loaded', 'pulled'] or hits or not all(r['pull_motion_m'].get(a, 0) > 0 for a in ('psm1', 'psm3'))):
            issues.append('no valid paired no-cut pull')
    else:
        if (not hits or phases != ['loaded']+['work']*len(hits)+['park'] or
                any(h['to_angle'] != 0 or h['from_angle'] <= 0 or h['arm'] != 'psm2' for h in hits) or
                [f['bites'] for f in r['frames'] if f['phase'] == 'work'] != list(range(1, len(hits)+1))):
            issues.append('missing phases or closure sequence')
        if not r.get('roles') or any(not v['ok'] for v in r['roles'].values()):
            issues.append('unverified instrument roles')
    return dict(case=entry['id'], run_id=r['run_id'], controller=r.get('controller_sha256'),
                state=r.get('final_state'), reason=r.get('reason'), mechanics_ok=not issues, issues=issues,
                oracle_sha256=sha256(path), rows=rows, park=park, pull_motion_m=r['pull_motion_m'])


def illustration(runs, predictions, outdir):
    selected = []
    for label in ('pulled', 8, 9):
        candidates = [(run, row) for run in runs for row in run['rows'] if
                      (row['phase'] == 'pulled' if label == 'pulled' else
                       (row['phase'] == 'work' and row['actual_mm'] == label))]
        candidates.sort(key=lambda pair: not pair[0]['mechanics_ok'])
        if candidates:
            selected.append(candidates[0])
    if not selected:
        return
    out = Image.new('RGB', (350*len(selected), 260), 'white')
    draw = ImageDraw.Draw(out)
    for n, (run, row) in enumerate(selected):
        with np.load(OUTPUT/run['case']/'sensor'/(row['key']+'.npz')) as f:
            rgb = f['rgb']
        y, x = np.where(inks(rgb) >= 0)
        if len(x):
            box = (max(0, int(x.min()-6)), max(0, int(y.min()-6)),
                   min(640, int(x.max()+7)), min(480, int(y.max()+7)))
            crop = Image.fromarray(rgb).crop(box)
            factor = min(340/crop.width, 206/crop.height)
            out.paste(crop.resize((max(1, int(crop.width*factor)), max(1, int(crop.height*factor))),
                                  Image.Resampling.NEAREST), (n*350+4, 48))
        draw.text((n*350+4, 4), f"{row['phase']} | {row['actual_mm']} mm | {row['decisions'][0]}", fill='black')
        draw.text((n*350+4, 24), 'case '+run['case']+' | '+str(predictions[row['key']]['native'].get('reason', '')), fill='black')
    with (outdir/'working_pair.png').open('xb') as stream:
        out.save(stream, format='PNG')


def main(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    if any((outdir/f).exists() for f in ('blind_predictions.json', 'decision.json', 'working_pair.png')):
        raise FileExistsError('use a fresh --out-dir for replay')
    protocol = json.loads(PLAN.read_bytes())
    hashes = {p: sha256(ROOT/p) for p in SOURCES}
    if hashes['scripts/vision/ruler_readout.py'] != json.loads((ROOT/'config/task/ruler_readout_dev_lock_v1.json').read_bytes())['reader_sha256']:
        raise ValueError('reader was not frozen')
    pred, shutter = blind_images()
    raw = outdir/'blind_predictions.json'
    write_new(raw, dict(predictions=pred, source_sha256=hashes, shutter=shutter, protocol_sha256=sha256(PLAN)))
    seen = set()
    runs = [run_score(e, pred, seen, protocol, hashes) for e in protocol['development_runs']]
    if seen != set(pred) or len({r['run_id'] for r in runs}) != len(runs):
        raise ValueError('missing or duplicate run/image')
    controllers = {r['controller'] for r in runs}
    if len(controllers-{None}) > 1:
        raise ValueError('mixed staged controllers')
    coverage = {mm: sorted({r['case'] for r in runs if r['mechanics_ok'] for f in r['rows']
                            if f['phase'] == 'work' and f['actual_mm'] == mm}) for mm in (8, 9)}
    independent = any(a != b for a in coverage[8] for b in coverage[9])
    optical = all(f['optical_ok'] for r in runs for f in r['rows'])
    gate = dict(all_mechanical=all(r['mechanics_ok'] for r in runs), actual_working_pair=independent,
                all_optical=optical, shutter_unknown=shutter['decision'] == 'UNKNOWN')
    write_new(outdir/'decision.json', dict(protocol=protocol, protocol_sha256=sha256(PLAN),
              prediction_sha256=sha256(raw), gates=gate, passed=all(gate.values()), coverage=coverage, runs=runs,
              confirmation_run=False, visual_control=False))
    illustration(runs, pred, outdir)
    print(gate, 'working trajectories', coverage)
    return int(not all(gate.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUTPUT)
    raise SystemExit(main(parser.parse_args().out_dir))
