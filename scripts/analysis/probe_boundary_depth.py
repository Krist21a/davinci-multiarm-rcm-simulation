"""Test optical boundary localization before changing any live endpoint/sweep calculation."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audit_entry_skew import ROOT, cp, load_dataset, sha

MC = cp.material_coordinate
PLAN = ROOT / 'config/task/m5b_boundary_depth_probe.json'


def local_z(label, depth, x, y, code):
    x0, y0 = max(0, int(np.ceil(x-5))), max(0, int(np.ceil(y-5)))
    sl = np.s_[y0:int(y+5)+1, x0:int(x+5)+1]
    patch_depth = depth[sl]
    v, u = np.nonzero((label[sl] == code) & np.isfinite(patch_depth) & (patch_depth > 0))
    z = patch_depth[v, u]
    X = np.column_stack((u + x0 - x, v + y0 - y, np.ones(len(v))))
    keep = np.linalg.norm(X[:, :2], axis=1) <= 5
    X, z = X[keep], z[keep]
    if len(z) < 4:
        return None
    beta, _, rank, _ = np.linalg.lstsq(X, 1/z, rcond=None)
    if rank < 3 or beta[-1] <= 0 or np.any(X @ beta <= 0):
        return None
    if np.percentile(abs(1/(X @ beta) - z), 90) > MC.DEPTH_EVIDENCE_M:
        return None
    return float(1/beta[-1])


def measure(args, front, f, direction, cam_R):
    rgb, depth, mask, along, lateral, centre = args
    h, w = mask.shape
    label = np.full(mask.shape, -1, dtype=np.int8)
    for i, col in enumerate(MC.COLORS):
        label[mask & (np.linalg.norm(rgb - 255*np.array(col), axis=2) <= MC.COLOR_TOL)] = i
    axis = cam_R.T @ direction

    def along_at(x, y, z):
        return float(np.array([(w/2-x)/f, (h/2-y)/f, 1.]) @ axis * z)

    rows = [[[], []] for _ in MC.ROWS]
    for left, right, du, dv in [(np.s_[:, :-1], np.s_[:, 1:], 1, 0), (np.s_[:-1, :], np.s_[1:, :], 0, 1)]:
        low = along[left] < along[right]
        l, r = np.where(low, label[left], label[right]), np.where(low, label[right], label[left])
        z0, z1 = np.where(low, depth[left], depth[right]), np.where(low, depth[right], depth[left])
        a, b = np.where(low, mask[left], mask[right]), np.where(low, mask[right], mask[left])
        side = .5*(lateral[left]+lateral[right])-centre
        bank = (np.abs(side) >= MC.BANK_INNER_M) & (np.abs(side) <= MC.BANK_OUTER_M)
        bank &= np.isfinite(z0) & np.isfinite(z1) & (z0 > 0) & (z1 > 0)
        hits = [(r == 0) & ~a & (z0-z1 > MC.DEPTH_EVIDENCE_M)]
        hits += [(l == k-1) & (r == k) & (abs(z1-z0) <= MC.DEPTH_EVIDENCE_M) for k in range(1,4)]
        hits += [(l == 3) & ~b & (z1-z0 > MC.DEPTH_EVIDENCE_M)]
        for j, hit in enumerate(hits):
            for v, u in np.argwhere(hit & bank):
                x, y = u+du/2, v+dv/2
                depths = [local_z(label, depth, x, y, k) for k in (j-1, j) if 0 <= k < len(MC.COLORS)]
                if None in depths or np.ptp(depths) > MC.DEPTH_EVIDENCE_M:
                    return dict(reason='unresolved observed transition', row=MC.ROWS[j])
                rows[j][int(side[v, u] > 0)].append(along_at(x, y, np.mean(depths)))
    indices = [j for j, banks in enumerate(rows) if min(map(len, banks)) >= MC.MIN_PAIRS]
    if not indices or indices[0] != 0 or indices[-1] != len(MC.ROWS)-1:
        return dict(reason='unverified endpoints')
    q = np.array([[np.percentile(bank, [10, 50, 90]) for bank in rows[j]] for j in indices])
    lo, med, hi = q[:, :, 0].min(axis=1), q[:, :, 1].mean(axis=1), q[:, :, 2].max(axis=1)
    report = dict(rows=MC.KNOTS_M[indices].tolist(), skew_mm=((hi-lo)*1e3).tolist())
    if (np.any(hi-lo > MC.MAX_BANK_SKEW_M) or np.any(lo[1:] <= hi[:-1])
            or any(np.any(np.diff(arr) < MC.MIN_ROW_SPAN_M) for arr in (lo, med))):
        return dict(report, reason='inconsistent measured boundaries')
    report.update(knots=med.tolist(), guard=lo.tolist())
    x, y = front['front_uv']
    zs = [z for k in range(len(MC.COLORS)) if (z := local_z(label, depth, x, y, k)) is not None]
    if not zs or np.ptp(zs) > MC.DEPTH_EVIDENCE_M:
        return dict(report, reason='unresolved apex surface')
    s = along_at(x, y, np.median(zs))
    loc = MC.locate(MC.KNOTS_M[indices], med, s)
    return dict(report, progress_m=None if loc is None else loc[0],
                reason='unsupported apex cell' if loc is None else '')


def replay(source):
    path = ROOT / source['report']
    meta, frames, files = load_dataset(path.stem)
    run = json.loads(path.read_bytes())
    if (files != source['capture'] or sha(path) != source['sha256'] or files['run_id'] != run['run_id']
            or files['scene_sha256'] != sha(ROOT / run['scene']) or meta['perception_cfg'] != run['config']['perception']
            or any(sha(ROOT / k) != v for k, v in run['source_sha256'].items())):
        raise ValueError('frame provenance mismatch')
    rng = np.random.default_rng(meta['perception_cfg']['seed'])
    noise = meta['perception_cfg']['depth_noise_m']
    e = cp.CutPerception.from_meta(meta)
    last, samples = [], []
    original = MC.measure
    def capture(*args, **kwargs):
        last[:] = args
        return original(*args, **kwargs)
    for frame, record, trace in zip(frames, run['perception'], run['trace'], strict=True):
        rng.normal(size=6)
        depth = frame['depth'].astype(float)
        if noise:
            depth += rng.normal(0, noise, depth.shape)
        with patch.object(MC, 'measure', capture):
            out = e.estimate(frame['rgb'], depth, frame['cam_R'], frame['cam_t'],
                             'grasped_arms' in frame and bool(len(frame['grasped_arms'])))
        c = out['cut']
        if json.loads(json.dumps(c, default=lambda x: x.tolist())) != record['estimate']['cut']:
            raise ValueError('non-identical replay')
        if not c.get('front_valid') or c.get('mode') not in ('slit', 'bank_slit', 'pre_cut'):
            continue
        d = np.asarray(c['direction'])
        test = measure(last, c, e.f, d, frame['cam_R'])
        if 'knots' in test:
            lateral = cp.unit(np.cross(out['tissue']['normal'], d))
            origin = np.asarray(c['start'])
            true_s = []
            for j in MC.ROWS:
                nodes = frame['nodes'][j]
                b, s = (nodes-origin) @ lateral, (nodes-frame['cam_t']) @ d
                order = np.argsort(b)
                if not b.min() <= 0 <= b.max():
                    raise ValueError('chart outside model for scoring')
                true_s.append(float(np.interp(0, b[order], s[order])))
            indices = np.searchsorted(MC.KNOTS_M, test['rows'])
            test['knot_error_mm'] = (1000*(test['knots']-np.array(true_s)[indices])).tolist()
            test['guard_error_mm'] = (1000*(test['guard']-np.array(true_s)[indices])).tolist()
        samples.append(dict(tick=trace['tick'], truth_m=trace['cut_length_m'],
                            planar_progress_m=c.get('progress_m'), **test))
    return dict(name=path.stem, source=source['sha256'], frames=files, samples=samples)


def main(out):
    if out.exists():
        raise FileExistsError(out)
    source = ROOT / 'validation/task/m5b_v1_10_verified_summary.json'
    summary = json.loads(source.read_bytes())
    trials = [replay(s) for s in [summary['marked_truth'], *summary['trials']]]
    ok = True
    for trial, checkpoints in zip(trials, [(16,17), (26,), (34,)]):
        at = {v['tick']:v for v in trial['samples']}
        for t in checkpoints:
            s = at[t]
            x = s.get('progress_m')
            ok &= x is not None and ((.009 <= x <= .010) if s['truth_m'] == .01 else abs(x-.005) <= .001)
            print(trial['name'], t, x, s['reason'])
        ok &= not any(v['truth_m'] < .009 and (v.get('progress_m') or 0) >= .009 for v in trial['samples'])
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(plan_sha256=sha(PLAN), script_sha256=sha(Path(__file__)), source_sha256=sha(source),
                       required_checkpoint_checks_passed=bool(ok), trials=trials), stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
