"""Reject/inspect a local-surface hypothesis offline, before allowing it to drive a blade.

This is a development counterfactual, not a new acceptance test or controller.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
import cut_perception as cp


class SurfaceExperiment(cp.CutPerception):
    def _cut(self, tissue, mask, plane, pts, scis, depth, rgb):
        self.support, self.raw_depth = mask, depth
        return super()._cut(tissue, mask, plane, pts, scis, depth, rgb)

    def _plane_coords(self, c, normal, forward, across):
        s, b, z = super()._plane_coords(c, normal, forward, across)
        y, x = np.nonzero(self.support)
        region = np.s_[max(y.min()-12, 0):y.max()+13, max(x.min()-12, 0):x.max()+13]
        mask = self.support[region]
        # Normalized convolution of only observed sheet residuals. Keep the
        # original raw depth for slit, coverage and silhouette evidence tests.
        weights = gaussian_filter(mask.astype(float), 2, mode='constant')
        weighted = gaussian_filter(np.where(mask, self.raw_depth[region] - z[region], 0.), 2, mode='constant')
        dz = np.divide(weighted, weights, out=np.zeros_like(weighted), where=weights > .15)
        s[region] += dz * (s[region] + c @ forward) / z[region]
        b[region] += dz * (b[region] + c @ across) / z[region]
        z[region] += dz
        return s, b, z


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(out):
    if out.exists():
        raise FileExistsError(out)
    result = dict(scope='Offline only; truth labels score the estimate after RGB-D inference.',
                  experiment_sha256=sha(Path(__file__)), trials=[])
    for case, tick in [('truth', 16), ('V_nominal', 25), ('V_noise_depth_0p5', 20)]:
        name = 'm5b_v1_9_1_marked_' + case
        path = ROOT / 'validation/task' / (name + '.json')
        run = json.loads(path.read_bytes())
        folder = ROOT / run['config']['perception']['dump_dir']
        meta = json.loads((folder / 'meta.json').read_bytes())
        capture = json.loads((folder / 'capture.json').read_bytes())
        if (meta['run_id'] != capture['run_id'] or meta['run_id'] != run['run_id']
                or meta['perception_cfg'] != run['config']['perception']
                or capture['frame_count'] != len(run['trace'])
                or sha(ROOT / run['scene']) != run['scene_sha256']
                or any(sha(ROOT / k) != h for k, h in run['source_sha256'].items())
                or sha(ROOT / 'config/task/m5b_protocol_v1_9_1.json') != run['protocol_amendment_sha256']):
            raise ValueError('reference provenance mismatch')
        baseline, experiment = (klass.from_meta(meta) for klass in (cp.CutPerception, SurfaceExperiment))
        rng = np.random.default_rng(meta['perception_cfg']['seed'])
        noise = meta['perception_cfg']['depth_noise_m']
        prefix = hashlib.sha256()
        for t in range(1, tick+1):
            frame = folder / f'tick_{t:03d}.npz'
            prefix.update(bytes.fromhex(sha(frame)))
            with np.load(frame, allow_pickle=False) as fr:
                depth = fr['depth'].astype(float)
                rng.normal(size=6)  # the two zero-magnitude per-tick camera perturbations
                if noise:
                    depth += rng.normal(0, noise, depth.shape)
                args = fr['rgb'], depth, fr['cam_R'], fr['cam_t'], 'grasped_arms' in fr and bool(len(fr['grasped_arms']))
                ref, trial = (e.estimate(*args)['cut'] for e in (baseline, experiment))
            if json.loads(json.dumps(ref, default=lambda a: a.tolist())) != run['perception'][t-1]['estimate']['cut']:
                raise ValueError(f'{name}/{t}: non-identical reference perception')
        truth = run['trace'][tick-1]['cut_length_m']
        result['trials'].append(dict(case=case, tick=tick, truth_m=truth, report_sha256=sha(path),
                                     prefix_sha256=prefix.hexdigest(), reference=ref, experiment=trial))
        print(case, tick, truth, ref.get('progress_m'), trial.get('progress_m'), trial.get('mode'))
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False, default=lambda a: a.tolist())
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
