"""Independent two-bank apparent material readout; the mesh oracle is scoring only."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audit_entry_skew import ROOT, cp, load_dataset, sha
from probe_boundary_depth import measure as measure_depth

MC = cp.material_coordinate
PLAN = ROOT / 'config/task/m5b_bank_transect_probe.json'


def labels(rgb, mask):
    lab = np.full(mask.shape, -1, dtype=np.int8)
    for k, col in enumerate(MC.COLORS):
        lab[mask & (np.linalg.norm(rgb-255*np.array(col), axis=2) <= MC.COLOR_TOL)] = k
    return lab


def votes(args):
    rgb, depth, mask, s, b, centre = args
    lab = labels(rgb, mask)
    rows = [[] for _ in MC.ROWS]
    for left, right in [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])]:
        low = s[left] < s[right]
        l, r = np.where(low, lab[left], lab[right]), np.where(low, lab[right], lab[left])
        z, zz = np.where(low, depth[left], depth[right]), np.where(low, depth[right], depth[left])
        a, aa = np.where(low, mask[left], mask[right]), np.where(low, mask[right], mask[left])
        x, y = (s[left]+s[right])/2, (b[left]+b[right])/2-centre
        keep = (np.abs(y) >= MC.BANK_INNER_M) & (np.abs(y) <= MC.BANK_OUTER_M)
        keep &= np.isfinite(x) & np.isfinite(z) & np.isfinite(zz) & (z > 0) & (zz > 0)
        sel = [(r == 0) & ~a & (z-zz > MC.DEPTH_EVIDENCE_M)]
        sel += [(l == k-1) & (r == k) & (abs(z-zz) <= MC.DEPTH_EVIDENCE_M) for k in range(1, len(MC.COLORS))]
        sel += [(l == len(MC.COLORS)-1) & ~aa & (zz-z > MC.DEPTH_EVIDENCE_M)]
        for row, selected in zip(rows, sel):
            hit = keep & selected
            row.append(np.column_stack((y[hit], x[hit])))
    return [np.concatenate(row) for row in rows]


def read_banks(rows, lateral, along, mode):
    knots, crossings = [], []
    for material, pts in zip(MC.KNOTS_M, rows, strict=True):
        banks = [pts[sign*pts[:, 0] > 0] for sign in (-1, 1)]
        if min(map(len, banks)) < MC.MIN_PAIRS:
            continue
        pair = []
        for q in banks:
            fit, _, rank, _ = np.linalg.lstsq(np.column_stack((q[:, 0]-lateral, np.ones(len(q)))), q[:, 1], rcond=None)
            if rank < 2 or not np.isfinite(fit).all():
                return dict(reason='unresolved row orientation')
            pair.append(fit[1])
        knots.append(material)
        crossings.append(pair)
    if not knots or knots[0] != MC.KNOTS_M[0] or knots[-1] != MC.KNOTS_M[-1]:
        return dict(reason='unverified endpoints')
    domains = np.array(crossings).T
    if np.any(np.diff(domains, axis=1) < MC.MIN_ROW_SPAN_M):
        return dict(reason='unordered row intersections')
    if mode == 'pre_cut':
        return dict(reason='', bank_progress_mm=[0., 0.], apparent_progress_mm=0.)
    values = [MC.locate(knots, domain, along) for domain in domains]
    if None in values:
        return dict(reason='unsupported apex cell')
    pair = [1000*x[0] for x in values]
    cells = [x[1] for x in values]
    if cells[0] != cells[1] or np.ptp(pair) > cp.GAP_BIN_M*1000:
        return dict(reason='banks disagree at apex', bank_progress_mm=pair, cells_m=cells)
    return dict(reason='', bank_progress_mm=pair, cells_m=cells, apparent_progress_mm=float(np.mean(pair)))


def rendered_faces(frame):
    n = frame['nodes']
    nv, nu = n.shape[:2]
    alive = {tuple(sorted((a, b))) for a, b, flag in frame['tissue_alive'] if flag}
    faces = []
    for row in range(nv-1):
        for k in range(row*nu, (row+1)*nu-1):
            for face in ((k, k+1, k+nu+1), (k, k+nu+1, k+nu)):
                if all(tuple(sorted((face[i-1], face[i]))) in alive for i in range(3)):
                    faces.append(face)
    return np.asarray(faces, dtype=int).reshape(-1, 3)


def oracle(frame, front_uv, f, lab):
    faces = rendered_faces(frame)
    nu = frame['nodes'].shape[1]
    xyz = ((frame['nodes'].reshape(-1, 3)-frame['cam_t']) @ frame['cam_R'])[faces]
    if not len(faces) or np.any(xyz[..., 2] <= 0):
        return []
    h, w = lab.shape
    uv = np.array([w/2, h/2])-f*xyz[..., :2]/xyz[..., 2:]
    a, b = uv[:, 1]-uv[:, 0], uv[:, 2]-uv[:, 0]
    det = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    cx, cy = np.rint(front_uv).astype(int)
    samples = []
    for y in range(max(0, cy-1), min(h, cy+2)):
        for x in range(max(0, cx-1), min(w, cx+2)):
            if lab[y, x] < 0:
                continue
            offset = [x, y]-uv[:, 0]
            den = np.where(abs(det) > 1e-10, det, np.nan)
            u = (offset[:, 0]*b[:, 1]-offset[:, 1]*b[:, 0])/den
            v = (a[:, 0]*offset[:, 1]-a[:, 1]*offset[:, 0])/den
            hits = np.flatnonzero((u >= 0) & (v >= 0) & (u+v <= 1))
            if not len(hits):
                continue
            weights = np.column_stack((1-u[hits]-v[hits], u[hits], v[hits])) / xyz[hits, :, 2]
            z = 1/weights.sum(axis=1)
            i = int(np.argmin(z))
            rows = faces[hits[i]]//nu
            band = np.searchsorted(MC.ROWS[1:], rows.min(), side='right')
            if band != lab[y, x] or not abs(z[i]-frame['depth'][y, x]) <= MC.DEPTH_EVIDENCE_M:
                continue
            samples.append(dict(uv=[x, y], material_mm=float((weights[i] @ rows)*z[i])))
    return samples


def replay(source):
    path = ROOT / source['report']
    run = json.loads(path.read_bytes())
    meta, frames, files = load_dataset(path.stem)
    if (sha(path) != source['sha256'] or files != source['capture'] or files['run_id'] != run['run_id']
            or meta['perception_cfg'] != run['config']['perception'] or files['scene_sha256'] != sha(ROOT/run['scene'])
            or meta['spacing_m'] != .001 or meta['NV'] != MC.ROWS[-1]+1
            or any(meta['perception_cfg']['camera_pose_error'][k] for k in ('trans_m', 'rot_deg'))
            or any(sha(ROOT/k) != h for k, h in run['source_sha256'].items())):
        raise ValueError('unmatched development capture')
    rng = np.random.default_rng(meta['perception_cfg']['seed'])
    e = cp.CutPerception.from_meta(meta)
    original = MC.measure
    last, samples = [], []
    def tap(*args, **kwargs):
        last[:] = args
        return original(*args, **kwargs)
    for frame, ref, trace in zip(frames, run['perception'], run['trace'], strict=True):
        rng.normal(size=6)
        depth = frame['depth'].astype(float)
        noise = meta['perception_cfg']['depth_noise_m']
        if noise:
            depth += rng.normal(0, noise, depth.shape)
        last.clear()
        with patch.object(MC, 'measure', tap):
            out = e.estimate(frame['rgb'], depth, frame['cam_R'], frame['cam_t'],
                             'grasped_arms' in frame and bool(len(frame['grasped_arms'])))
        c = out['cut']
        if (int(frame['tick']) != ref['tick'] or ref['tick'] != trace['tick'] or
                json.loads(json.dumps(c, default=lambda a: a.tolist())) != ref['estimate']['cut']):
            raise ValueError('non-identical reference replay')
        entry = dict(tick=ref['tick'], truth_mm=1000*trace['cut_length_m'], mode=c.get('mode'), guarded_mm=None)
        samples.append(entry)
        if not c.get('front_valid') or c.get('mode') not in ('slit','bank_slit','pre_cut'):
            continue
        rows = votes(last)
        counts = [[int(sum(s*q[:,0] > 0)) for s in (-1,1)] for q in rows]
        if counts != c['material_chart']['boundary_pairs']:
            raise ValueError('unmatched bank votes')
        a = np.array(c['direction'])
        b = cp.unit(np.cross(out['tissue']['normal'], a))
        p = np.array(c['front'])-out['tissue']['centroid']
        read = read_banks(rows, p @ b-last[-1], p @ a, c['mode'])
        physical = measure_depth(last, c, e.f, a, frame['cam_R'])
        entry.update(readout=read, physical=physical)
        if not read['reason'] and 'guard' in physical and not physical['reason']:
            entry['guarded_mm'] = read['apparent_progress_mm']
        # Only after both RGB-D decisions, intersect the recorded mesh for scoring.
        entry['surface_pixels'] = oracle(frame, c['front_uv'], e.f, labels(last[0], last[2]))
    return dict(name=path.stem, capture=files, report_sha256=sha(path), samples=samples)


def main(out):
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(PLAN.read_bytes())
    summary = ROOT / 'validation/task/m5b_v1_10_verified_summary.json'
    refs = json.loads(summary.read_bytes())
    trials = [replay(s) for s in (refs['marked_truth'], *refs['trials'])]
    decisions = []
    for t, (checkpoints, truth, lo, hi) in zip(
            trials, [((16, 17), 10, 9, 10), ((26,), 5, 4, 6), ((34,), 5, 4, 6)], strict=True):
        for tick in checkpoints:
            s = next(x for x in t['samples'] if x['tick'] == tick)
            value = s['guarded_mm']
            if s['truth_mm'] != truth:
                raise ValueError('unexpected reference length at checkpoint')
            decisions.append(dict(name=t['name'], tick=tick, expected_mm=[lo, hi],
                                  passed=value is not None and lo <= value <= hi))
            print(t['name'], tick, s.get('readout'), 'guarded', value,
                  'mesh samples', [round(v['material_mm'],2) for v in s.get('surface_pixels',[])])
        t['premature_stops'] = [s['tick'] for s in t['samples'] if s['truth_mm'] < 9 and (s['guarded_mm'] or 0) >= 9]
    ok = all(row['passed'] for row in decisions) and not any(t['premature_stops'] for t in trials)
    with out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(dict(protocol=plan, plan_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
                       depth_probe_sha256=sha(ROOT/'scripts/analysis/probe_boundary_depth.py'),
                       source_sha256=sha(summary), checkpoint_gate=bool(ok), checkpoints=decisions, trials=trials),
                  f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    main(ap.parse_args().out)
