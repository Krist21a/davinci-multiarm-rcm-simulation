"""Acquire separate sensor and structural records on the development-only surface seam."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/task'))
from cut_fsm import CutTaskFSM, S
from seam_topology import material_state
from surface_seam_world import SurfaceSeamWorld
from simtools import add_arguments, connect, matrix_to_np, sha256, stop_simulation

PLAN = ROOT / 'config/task/surface_seam_pilot_v1.json'
OUTPUT = ROOT / 'validation/task/surface_seam_pilot_v1'
SOURCES = (
    'scripts/analysis/capture_surface_seam.py', 'scripts/analysis/score_surface_seam.py',
    'scripts/task/surface_seam_world.py', 'scripts/task/seam_topology.py',
    'scripts/vision/ruler_readout.py', 'src/tissue/ruler_print.lua', 'src/tissue/surface_seam.lua',
    'scripts/task/cut_fsm.py', 'scripts/task/sim_world.py', 'scripts/task/ground_truth.py',
    'scripts/task/workspace.py', 'src/tissue/tissue.lua', 'scripts/build/simtools.py')


def write_new(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, default=lambda obj: np.asarray(obj).tolist(), allow_nan=False)
        f.write('\n')


class Capture(CutTaskFSM):
    def __init__(self, world, cfg, folder, report, pull_only):
        super().__init__(world, cfg)
        self.folder, self.record, self.pull_only = folder, report, pull_only
        self.bites = 0
        self.ready = None

    def tcp(self, arm):
        return matrix_to_np(self.w.sim.getObjectMatrix(self.w.by[arm + '_tcp'], -1))

    def on_init(self, obs):
        super().on_init(obs)
        if self.state == S.LOCALIZE:
            self.ready = self.tcp(self.cutter)
            self.record['ready_tcp'] = self.ready.tolist()

    def save(self, phase, obs):
        w, key = self.w, uuid.uuid4().hex
        rgb, depth = np.asarray(w.image), np.asarray(w.depth, dtype='<f4')
        if not (rgb.shape == (480, 640, 3) and depth.shape == rgb.shape[:2]):
            raise ValueError('unexpected native camera resolution')
        sensor = self.folder / 'sensor' / (key+'.npz')
        structure = self.folder / 'structure' / (key+'.npz')
        with w.truth.purpose('scoring'):
            edges = np.asarray(w.call('tissueConstraints')).reshape(-1, 3)
            faces = np.asarray(w.call('surfaceSeamFaces')).reshape(13, 2)
            nodes = w.truth.nodes()
            st = w.truth.tissue_stats()
        state = material_state(edges)
        if not np.array_equal(state['faces'], faces) or abs(state['prefix_rows'] - obs.cut_length_m*1000) > 1e-8:
            raise RuntimeError('mechanical/surface oracle mismatch')
        np.savez_compressed(sensor, rgb=rgb, depth=depth)
        np.savez_compressed(structure, edges=edges, faces=faces, nodes=nodes)
        self.record['frames'].append(dict(key=key, phase=phase, tick=w.tick, sensor_sha256=sha256(sensor),
            structure_sha256=sha256(structure), prefix_mm=state['prefix_rows'], farthest_mm=state['farthest_row_end'],
            bites=self.bites, direct_cuts=st['cut'], releases=w.call('surfaceSeamReleases'),
            obs=asdict(obs), closed=w.jaws_closed(), tcp=self.tcp(self.cutter).tolist()))

    def supervise(self, obs):
        if self.w.sheet_origin is not None:
            hits = self.w.call('tissueStats').get('cut_events') or []
            self.record['observations'].append(dict(tick=self.w.tick, state=self.state.name, **asdict(obs)))
            if len(hits) != self.bites:
                if len(hits) != self.bites + 1 or self.pull_only:
                    raise RuntimeError('unexpected cut event sequence')
                self.bites = len(hits)
                self.save('work', obs)  # before supervision can issue any recovery action
        return super().supervise(obs)

    def on_align(self, obs):
        if not self.record['frames']:
            if not all(obs.grasped.get(a, 0) >= self.cfg['grasp']['min_nodes'] for a in (self.primary, self.assist)):
                return self.go(S.FAILED, 'both physical grasps required', obs)
            self.save('loaded', obs)
        if not self.pull_only:
            return super().on_align(obs)
        # A genuine negative control: record measured outward motion, not just a tension reading.
        for arm in (self.primary, self.assist):
            p = self.tcp(arm)[:, 3].copy()
            side = np.sign((self.w.grasped_points[arm] - self.w.sheet_origin) @ self.w.u)
            if not self.w.retract(arm, self.cfg['cut']['retract_step_m']):
                return self.go(S.FAILED, arm+' pull refused', obs)
            obs = self.w.observe()
            self.record['pull_motion_m'][arm] = float(side * (self.tcp(arm)[:, 3] - p) @ self.w.u)
            if (self.supervise(obs) or not self.grasps_ok(obs) or not self.tension_ok(obs)
                    or self.record['pull_motion_m'][arm] <= 0):
                return self.go(S.FAILED, arm+' pull did not pass motion/safety checks', obs)
        self.save('pulled', obs)
        self.w.hold()
        self.go(S.DONE, 'traction-only capture complete; not a cutting task completion', obs)

    def _complete(self, obs):
        super()._complete(obs)
        if self.state == S.DONE:
            final = self.w.observe(inject_events=False)
            self.save('park', final)


def acquire(args):
    protocol = json.loads(PLAN.read_bytes())
    entry = next(item for item in protocol['development_runs'] if item['id'] == args.case)
    parent = ROOT / protocol['parent']
    base = ROOT/'config/task/cut_task.json'
    setup = json.loads(parent.read_bytes())
    args.scene = ROOT / setup['scene']
    hashes = {p: sha256(ROOT/p) for p in SOURCES}
    lock = json.loads((ROOT/'config/task/ruler_readout_dev_lock_v1.json').read_bytes())
    if (sha256(parent) != protocol['parent_sha256'] or sha256(base) != setup['base_sha256']
            or sha256(args.scene) != setup['scene_sha256']
            or hashes['scripts/vision/ruler_readout.py'] != lock['reader_sha256']):
        raise ValueError('acquisition inputs differ from the registration')
    cfg = json.loads(base.read_bytes())
    cfg['cut'].update(control_source='truth', length_m=entry['goal_mm']/1000, advance_m=entry['cap_mm']/1000)
    cfg['perception'].update(exclude_tcp_marker=True)
    args.initial_view, args.initial_view_insertion_m = setup['view'][:2], setup['view'][2]
    folder = OUTPUT/entry['id']
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'sensor').mkdir()
    (folder/'structure').mkdir()
    report = dict(run_id=uuid.uuid4().hex, entry=entry, scope='truth-controlled specimen acquisition',
                  scene_sha256=sha256(args.scene), base_sha256=sha256(base), spec_sha256=sha256(PLAN),
                  source_sha256=hashes, cfg=cfg, frames=[], observations=[], pull_motion_m={})
    client = sim = fsm = None
    try:
        client, sim = connect(args)
        w = SurfaceSeamWorld(client, sim, args, cfg, mode='full')
        report.update(controller_sha256=w.controller_sha256, base_controller_sha256=w.base_controller_sha256)
        fsm = Capture(w, cfg, folder, report, entry.get('pull_only', False))
        fsm.run()
        report.update(view=w.initial_view_readback, events=w.call('tissueStats').get('cut_events') or [],
                      roles=w.roles_report, marker=w.tip_marker_optics)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if fsm is not None:
            report.update(fsm.report())
        if sim is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                report['cleanup_error'] = str(exc)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        write_new(folder/'oracle.json', report)
    print(entry['id'], report.get('final_state'), report.get('error') or report.get('reason'), flush=True)
    return int(bool(report.get('error') or report.get('cleanup_error')) or report.get('final_state') != 'DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--case', required=True, choices=('pull', 'a', 'b', 'c', 'd', 'e'))
    raise SystemExit(acquire(parser.parse_args()))
