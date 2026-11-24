"""Post-hoc single-shape knockout on the two original images; no motion or closure."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import probe_workpose_pairs as p
from score_workpose_pairs import regions


class ShapeProbe(p.ProbeWorld):
    def observe(self, **kwargs):
        if self.tick != self.spec['tick']:
            return super().observe(**kwargs)
        prior_est, prior_rng = copy.deepcopy((self._perc_estimator, self.perc_rng))
        p.SimWorld.observe(self, **kwargs)
        rgb, depth, time = np.asarray(self.image), self.depth.copy(), self.sim.getSimulationTime()
        with np.load(self.original_dir / 'after.npz') as z:
            if not np.array_equal(rgb, z['rgb']) or not np.array_equal(depth, z['depth']):
                raise ValueError('post-action image does not reproduce the original')
            _, _, ahead, near = regions(z)
        r, g, b = rgb.astype(float).transpose(2, 0, 1)
        green = (g > r + 50) & (g > b + 50)
        sim = self.sim
        shapes = {k: v for k, v in self.by.items() if k.startswith(self.cutter + '_')}
        shapes.update({f'housing_marker_{i}': h for i, h in enumerate(self.markers)})
        records = []
        for name, h in sorted(shapes.items()):
            if sim.getObjectType(h) != sim.object_shape_type:
                continue
            layer = sim.getObjectInt32Param(h, sim.objintparam_visibility_layer)
            if not layer:
                continue
            prop = sim.getObjectSpecialProperty(h)
            try:
                sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
                sim.setObjectSpecialProperty(h, prop & ~sim.objectspecialproperty_renderable)
                self.render()
                diff = self.depth - depth > .00001
                row = dict(name=name, changed_front=int(diff[near].sum()), changed_ahead=int(diff[ahead].sum()),
                           changed_green_front=int((green & diff)[near].sum()))
                if row['changed_front'] or row['changed_ahead']:
                    # Score this counterfactual with the pre-observation estimator and identical noise.
                    real_est, real_rng = self._perc_estimator, self.perc_rng
                    try:
                        self._perc_estimator, self.perc_rng = copy.deepcopy((prior_est, prior_rng))
                        out = self._estimate()
                    finally:
                        self._perc_estimator, self.perc_rng = real_est, real_rng
                    row.update(mode=out['cut'].get('mode'), progress_m=out['cut'].get('progress_m'),
                               reason=out['cut'].get('reason'), scissors_valid=out['scissors'].get('valid'))
                records.append(row)
            finally:
                sim.setObjectSpecialProperty(h, prop)
                sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, layer)
        self.render()
        if (sim.getSimulationTime() != time or not np.array_equal(self.depth, depth)
                or not np.array_equal(rgb, np.asarray(self.image))):
            raise ValueError('failed to restore optical baseline')
        p.write_json(self.original_dir / 'shape_attribution.json', dict(
            scope='Scoring-only single shape removal; not a controller observation or an authorized closure.',
            phase=self.phase, script_sha256=p.sha256(Path(__file__)), after_sha256=p.sha256(self.original_dir/'after.npz'),
            shapes=records))
        print(self.phase, [x for x in records if x['changed_front'] or x['changed_ahead']], flush=True)
        raise p.Sampled()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    p.add_arguments(ap)
    ap.add_argument('--case-dir', type=Path, required=True)
    args = ap.parse_args()
    case = args.case_dir
    if (case / 'shape_attribution.json').exists():
        raise FileExistsError(case)
    src = json.loads((case / 'result.json').read_bytes())
    phase = src['phase']
    spec = json.loads(p.PROTOCOL.read_bytes())['cases'][phase]
    ref = json.loads((p.ROOT / 'validation/task' / (spec['run'] + '.json')).read_bytes())
    args.scene, args.scenario = p.INSTRUMENT_SCENE, 'nominal'
    args.initial_view, args.initial_view_insertion_m = ref['initial_view'], ref['initial_view_insertion_m']
    client, sim = p.connect(args)
    try:
        w = ShapeProbe(client, sim, args, ref['config'], mode='full')
        w.phase, w.spec, w.reference, w.distance = phase, spec, ref, spec['distances_m'][0]
        w.original, w.result, w.folder, w.original_dir = True, {}, args.stage_dir, case
        w.pair_state = dict(src['before'], target=[tuple(x) for x in src['before']['target']])
        try:
            p.CutTaskFSM(w, ref['config']).run()
        except p.Sampled:
            pass
        if not (case / 'shape_attribution.json').is_file():
            raise ValueError('checkpoint refused or not reached')
    finally:
        p.stop_simulation(client, sim)
        client.socket.close(linger=0)
        client.context.term()
