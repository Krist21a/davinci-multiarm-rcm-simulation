"""Replay one registered M5b action per branch; stop before any subsequent actuation."""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/task'))
from run_cut_task import CutTaskFSM, INSTRUMENT_SCENE, SimWorld, add_arguments, connect, sha256, stop_simulation, write_json

PROTOCOL = ROOT / 'config/task/m5b_workpose_pair_v1.json'


class Sampled(Exception):
    """Stop immediately after observation or refusal, before the FSM resumes."""


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def snapshot(w, path, hide=False):
    """Passive logging only. The hidden-tool counterfactual must not advance dynamics."""
    sim = w.sim
    w.render()
    rgb, depth = np.asarray(w.image), w.depth.copy()
    start_time = sim.getSimulationTime()
    R, t = w.camera()
    with w.truth.purpose('scoring'):
        nodes = w.truth.nodes()
        front = w.truth.cut_front()
    record = dict(rgb=rgb, depth=depth, cam_R=R, cam_t=t, focal=w.f, nodes=nodes,
                  truth_front=front, truth_direction=w.perc_log[-1]['truth']['direction'],
                  tcp=np.asarray(sim.getObjectMatrix(w.by[w.cutter + '_tcp'], -1)).reshape(3, 4))
    state = dict(time=start_time, nodes=array_sha(nodes), rgb=array_sha(rgb), depth=array_sha(depth),
                 camera=np.column_stack((R, t)).tolist(), tcp=record['tcp'].tolist(),
                 target=[(r['target'], r.get('grip')) for r in w.call('getRCMStats')],
                 cutter_q=w._tool_q(w.cutter)[0].tolist())
    if hide:
        handles = sorted(set([h for name, h in w.by.items() if name.startswith(w.cutter + '_')] + w.markers))
        old = []
        try:
            for h in handles:
                if sim.getObjectType(h) == sim.object_shape_type:
                    layer = sim.getObjectInt32Param(h, sim.objintparam_visibility_layer)
                    properties = sim.getObjectSpecialProperty(h)
                    old.append((h, layer, properties))
                    sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
                    sim.setObjectSpecialProperty(h, properties & ~sim.objectspecialproperty_renderable)
            w.render()
            record.update(no_cutter_rgb=np.asarray(w.image), no_cutter_depth=w.depth.copy())
        finally:
            for h, layer, properties in old:
                sim.setObjectSpecialProperty(h, properties)
                sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, layer)
            w.render()
        if (sim.getSimulationTime() != start_time or not np.array_equal(rgb, np.asarray(w.image))
                or not np.array_equal(depth, w.depth) or np.array_equal(rgb, record['no_cutter_rgb'])):
            raise ValueError('counterfactual render did not isolate and restore cutter visibility')
    np.savez_compressed(path, **record)
    return state


class ProbeWorld(SimWorld):
    def observe(self, **kwargs):
        obs = super().observe(**kwargs)
        tick = self.tick
        if tick <= self.spec['tick'] + 1:
            with self.truth.purpose('scoring'):
                node_hash = array_sha(np.asarray(self.call('tissueNodes'), dtype=float))[:16]
            trace = dict(tick=tick, tension=obs.tension, cut_length_m=obs.cut_length_m,
                         nodes_sha256=node_hash, protected_mm=self.last_protected)
            if tick <= self.spec['tick'] or self.original:
                logged = json.loads(json.dumps(self.perc_log[-1], default=lambda x: x.tolist()))
                if self.reference['trace'][tick - 1] != trace or logged != self.reference['perception'][tick - 1]:
                    raise ValueError(f'checkpoint trajectory or perception differs at tick {tick}')
        if tick == self.spec['tick'] + 1:
            if not self.result.get('motion_accepted'):
                raise ValueError('next observation without the registered accepted action')
            sc = self.closure_aligned()
            geom = self._vision_alignment_geometry()
            self.result.update(after_observation=asdict(obs), after_cut=self._last_est['cut'],
                               after_scissors=self._last_est['scissors'], closure_aligned=sc,
                               sweep_budget_m=self.endpoint_advance_budget(),
                               along_tip_minus_front_m=None if geom is None else float((geom[1] - geom[0]) @ geom[2]),
                               after=snapshot(self, self.folder / 'after.npz', hide=True))
            raise Sampled()
        return obs

    def close_scissors(self):
        if self.tick >= self.spec['tick']:
            raise RuntimeError('probe cannot close scissors at or after the checkpoint')
        return super().close_scissors()

    def _visible_tool_candidate(self, phase, current, targets, axis, hinge):
        if self.tick != self.spec['tick']:
            return super()._visible_tool_candidate(phase, current, targets, axis, hinge)
        if phase != self.phase:
            raise ValueError('unexpected checkpoint action')
        self.result['before'] = snapshot(self, self.folder / 'before.npz')
        self.result['before_cut'] = self._last_est['cut']
        if self.pair_state is not None and self.result['before'] != self.pair_state:
            raise ValueError('action-start image, tissue or arm state differs from paired original')
        targets = [(d, p) for d, p in targets if abs(d - self.distance) < 1e-12]
        if len(targets) != 1:
            raise ValueError('registered distance not among eligible action targets')
        res = super()._visible_tool_candidate(phase, current, targets, axis, hinge)
        self.result['prediction'] = self.event_log[-1]
        if res is None:
            self.result['refusal'] = 'candidate refused before motion'
            raise Sampled()
        return res

    def _vision_tool_motion(self, res, n):
        ok = super()._vision_tool_motion(res, n)
        if self.tick == self.spec['tick']:
            self.result['motion_accepted'] = ok
            if not ok:
                self.result['refusal'] = 'guarded command refused'
                raise Sampled()
        return ok


def run(args):
    cfg = json.loads(PROTOCOL.read_bytes())
    spec = cfg['cases'][args.phase]
    p = ROOT / 'validation/task' / (spec['run'] + '.json')
    ref = json.loads(p.read_bytes())
    if sha256(p) != spec['sha256'] or sha256(args.scene) != ref['scene_sha256']:
        raise ValueError('reference run or scene mismatch')
    for file, expect in ref['source_sha256'].items():
        if sha256(ROOT / file) != expect:
            raise ValueError(f'v1.8 source changed: {file}')
    args.initial_view = ref['initial_view']
    args.initial_view_insertion_m = ref['initial_view_insertion_m']
    args.scenario = 'nominal'
    args.out.mkdir(parents=True, exist_ok=False)
    pair_state = None
    client, sim = connect(args)
    try:
        for i, distance in enumerate(spec['distances_m']):
            folder = args.out / f'{i:02d}'
            folder.mkdir()
            try:
                w = ProbeWorld(client, sim, args, ref['config'], mode='full')
                if w.controller_sha256 != ref['controller_sha256']:
                    raise ValueError('injected controller hash mismatch')
                w.spec, w.phase, w.reference = spec, args.phase, ref
                w.folder, w.distance, w.original, w.pair_state = folder, distance, i == 0, pair_state
                w.result = dict(phase=args.phase, distance_m=distance, tick=spec['tick'],
                                protocol_sha256=sha256(PROTOCOL), probe_sha256=sha256(Path(__file__)),
                                reference_sha256=sha256(p), controller_sha256=w.controller_sha256,
                                scene_sha256=ref['scene_sha256'], motion_accepted=False)
                try:
                    fsm = CutTaskFSM(w, ref['config'])
                    fsm.run()
                    raise ValueError('task terminated without a paired sample')
                except Sampled:
                    pass
                closures = [e['tick'] for e in w.event_log if e['event'] == 'closure']
                if len(closures) != spec['prior_closures'] or 'untagged' in w.truth.reads_by_purpose:
                    raise ValueError('unexpected closure or truth access')
                pair_state = w.result['before']
                w.result.update(closure_ticks=closures, truth_reads_by_purpose=w.truth.reads_by_purpose,
                                files={x.name: sha256(x) for x in folder.glob('*.npz')})
                write_json(folder / 'result.json', w.result)
                print(args.phase, distance, 'accepted', w.result['motion_accepted'],
                      'next progress', w.result.get('after_observation', {}).get('cut_progress_m'), flush=True)
            finally:
                stop_simulation(client, sim)
    finally:
        client.socket.close(linger=0)
        client.context.term()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--phase', choices=('align', 'advance'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.set_defaults(scene=INSTRUMENT_SCENE)
    run(parser.parse_args())
