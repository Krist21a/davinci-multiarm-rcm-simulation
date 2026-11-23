"""Decompose row-zero apparent spread at failed chart checkpoints (scoring only)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from replay_perception import cp, load_dataset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spread(s, b):
    banks = [s[(b*k >= cp.material_coordinate.BANK_INNER_M) & (b*k <= cp.material_coordinate.BANK_OUTER_M)]
             for k in (-1, 1)]
    if min(map(len, banks)) < cp.material_coordinate.MIN_PAIRS:
        raise ValueError('insufficient evaluation samples')
    q = np.array([np.percentile(bank, [10, 50, 90]) for bank in banks]) * 1000
    return dict(samples=list(map(len, banks)), quantiles_mm=q.tolist(),
                spread_mm=float(q[:, 2].max() - q[:, 0].min()))


def measure_entry(frame, estimate, recorded):
    rgb, depth, mask, along, lateral, centre = recorded
    tissue, cut = estimate['tissue'], estimate['cut']
    c, n, a = map(np.asarray, (tissue['centroid'], tissue['normal'], cut['direction']))
    b = cp.unit(np.cross(n, a))
    camera = frame['cam_t']
    # Score the same continuous material edge both before and after global plane projection.
    row = frame['nodes'][0]
    t = (np.arange(100) + .5)/100
    points = (row[:-1, None] + t[None, :, None] * np.diff(row, axis=0)[:, None]).reshape(-1, 3)
    rays = points-camera
    plane = camera + rays * (((c-camera) @ n)/(rays @ n))[:, None]
    side = (plane-c) @ b - centre
    physical = spread((points-c) @ a, side)
    ideal = spread((plane-c) @ a, side)

    pink = mask & (np.linalg.norm(rgb - 255*np.array(cp.material_coordinate.COLORS[0]), axis=-1)
                   <= cp.material_coordinate.COLOR_TOL)
    hits = []
    for left, right in [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])]:
        low = along[left] < along[right]
        zz0, zz1 = np.where(low, depth[left], depth[right]), np.where(low, depth[right], depth[left])
        keep = (np.where(low, pink[right], pink[left]) & ~np.where(low, mask[left], mask[right])
                & (zz0 - zz1 > cp.material_coordinate.DEPTH_EVIDENCE_M)
                & np.isfinite(zz0) & np.isfinite(zz1) & (zz0 > 0) & (zz1 > 0))
        hits.append(np.stack([((along[left]+along[right])/2)[keep], ((lateral[left]+lateral[right])/2-centre)[keep]], axis=1))
    optical = spread(*np.concatenate(hits).T)
    if optical['samples'] != cut['material_chart']['boundary_pairs'][0]:
        raise ValueError('row zero samples do not reproduce the live measurement')
    return dict(model_edge_physical=physical, same_edge_on_fitted_plane=ideal, rgbd_transitions=optical)


def inspect(name, tick, source):
    ref = ROOT / 'validation/task' / (name + '.json')
    run = json.loads(ref.read_bytes())
    meta, frames, files = load_dataset(name)
    if (source['sha256'] != sha(ref) or files != source['capture'] or run['run_id'] != files['run_id']
            or meta['perception_cfg'] != run['config']['perception'] or files['scene_sha256'] != sha(ROOT / run['scene'])
            or any(sha(ROOT / k) != v for k, v in run['source_sha256'].items())):
        raise ValueError('checkpoint provenance mismatch')
    e = cp.CutPerception.from_meta(meta)
    rng = np.random.default_rng(meta['perception_cfg']['seed'])
    measure = cp.material_coordinate.measure
    last = []
    def tap(*args, **kwargs):
        last[:] = args
        return measure(*args, **kwargs)
    for frame in frames[:tick]:
        depth = frame['depth'].astype(float)
        noise = meta['perception_cfg']['depth_noise_m']
        rng.normal(size=6)
        if noise:
            depth += rng.normal(0, noise, depth.shape)
        with patch.object(cp.material_coordinate, 'measure', tap):
            observation = e.estimate(frame['rgb'], depth, frame['cam_R'], frame['cam_t'],
                                     'grasped_arms' in frame and bool(len(frame['grasped_arms'])))
    c = observation['cut']
    if (int(frame['tick']) != tick or json.loads(json.dumps(c, default=lambda x: x.tolist())) !=
            run['perception'][tick-1]['estimate']['cut'] or c['material_chart']['valid']):
        raise ValueError('checkpoint is not an exact failed chart replay')
    return dict(name=name, tick=tick, report_sha256=sha(ref), capture=files,
                cut_length_m=run['trace'][tick-1]['cut_length_m'], **measure_entry(frame, observation, last))


def main(out):
    if out.exists():
        raise FileExistsError(out)
    summary = ROOT / 'validation/task/m5b_v1_10_verified_summary.json'
    refs = json.loads(summary.read_bytes())
    rows = [inspect('m5b_v1_10_marked_'+label, t, s) for label, t, s in
            [('truth', 16, refs['marked_truth']), ('V_nominal', 26, refs['trials'][0]),
             ('V_noise_depth_0p5', 34, refs['trials'][1])]]
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(scope='Scoring only. Physical and ideal-plane spreads use identical dense samples of row zero '
                             'within the plane-projected bank windows. The actual RGB-D transitions additionally '
                             'reflect rasterization, edge selection and visibility; the residual is not pixel noise alone.',
                       summary_sha256=sha(summary), scorer_sha256=sha(Path(__file__)), rows=rows),
                  stream, indent=2, allow_nan=False)
        stream.write('\n')
    for r in rows:
        print(r['name'], r['tick'], [round(r[k]['spread_mm'], 3) for k in
              ('model_edge_physical', 'same_edge_on_fitted_plane', 'rgbd_transitions')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
