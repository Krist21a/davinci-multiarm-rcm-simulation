"""Paired marker-only replay. Geometry labels are read after both fits, for scoring."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from replay_perception import cp, load_dataset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(entry):
    path = ROOT / entry['report']
    run = json.loads(path.read_bytes())
    meta, frames, files = load_dataset(path.stem)
    if (sha(path) != entry['sha256'] or files != entry['capture'] or files['run_id'] != run['run_id']
            or files['scene_sha256'] != sha(ROOT / run['scene']) or meta['perception_cfg'] != run['config']['perception']
            or len(frames) != len(run['perception']) or not run['source_sha256']):
        raise ValueError('historical capture provenance mismatch')
    if any(meta['perception_cfg']['camera_pose_error'][k] for k in ('rot_deg', 'trans_m')):
        raise ValueError('expected an unperturbed camera')
    baseline, candidate = (cp.CutPerception.from_meta(meta) for _ in range(2))
    candidate.ray_marker_fit = True
    rng = np.random.default_rng(meta['perception_cfg']['seed'])
    closures = {c['tick'] for c in entry['closures']}
    paired, at_closure, before, after, promoted = [], [], 0, 0, 0
    noise = meta['perception_cfg']['depth_noise_m']
    for fr, row in zip(frames, run['perception']):
        rng.normal(size=6)
        depth = fr['depth'].astype(float)
        if noise:
            depth += rng.normal(0, noise, depth.shape)
        R, t = fr['cam_R'], fr['cam_t']
        rgb = fr['rgb'].astype(float)
        old, new = (e._scissors(rgb, depth)[0] for e in (baseline, candidate))
        logged = row['estimate']['scissors']
        if (int(fr['tick']) != row['tick'] or logged['valid'] != old['valid'] or
                (old['valid'] and not np.allclose(R @ old['tip'] + t, logged['tip'], atol=1e-12, rtol=0))):
            raise ValueError('original fit does not reproduce logged pose')
        before += int(old['valid'])
        after += int(new['valid'])
        promoted += int(new['valid'] and not old['valid'])
        if old['valid'] and new['valid']:
            tcp = fr['tcp_M']
            errors = []
            for obs in [old, new]:
                error = np.linalg.norm(R @ obs['tip'] + t - tcp[:3, 3]) * 1000
                dot = (R @ obs['jaw_axis']) @ tcp[:3, meta['tool_axis_col']]
                errors.append([float(error), float(np.rad2deg(np.arccos(np.clip(dot, -1, 1))))])
            paired.append(errors)
            if row['tick'] in closures:
                at_closure.append(dict(tick=row['tick'], tip_mm=[x[0] for x in errors]))
    if not paired:
        raise ValueError('no common valid marker poses')
    a = np.array(paired)
    return dict(report_sha256=entry['sha256'], capture=files, run_id=run['run_id'], noise_m=noise,
                before=before, after=after, common_valid=len(a), promoted=promoted, at_prior_closures=at_closure,
                tip_mm=np.percentile(a[:, :, 0], [50, 95, 100], axis=0).tolist(),
                jaw_deg=np.percentile(a[:, :, 1], [50, 95, 100], axis=0).tolist())


def main(out):
    previous = ROOT / 'validation/task/m5b_v1_9_1_verified_summary.json'
    spec = ROOT / 'config/task/m5b_protocol_v1_10.json'
    if out.exists():
        raise FileExistsError(out)
    rows = [compare(x) for x in json.loads(previous.read_bytes())['trials']]
    nominal, noisy = rows
    if [x['noise_m'] for x in rows] != [0., .0005]:
        raise ValueError('wrong pair')
    passed = (all(not x['promoted'] and x['after'] >= .8*x['before'] for x in rows)
              and all(new < old for old, new in noisy['tip_mm'][:2]) and nominal['tip_mm'][0][1] < 1.)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(schema='errors: [median,p95,max] x [original,ray]; mm or degrees',
                       previous_sha256=sha(previous), protocol_sha256=sha(spec), scorer_sha256=sha(Path(__file__)),
                       estimator_sha256=sha(Path(cp.__file__)), trials=rows, live_development_allowed=passed),
                  stream, indent=2, allow_nan=False)
        stream.write('\n')
    for x in rows:
        print(x['noise_m'], x['before'], x['after'], x['tip_mm'])
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    raise SystemExit(main(parser.parse_args().out))
