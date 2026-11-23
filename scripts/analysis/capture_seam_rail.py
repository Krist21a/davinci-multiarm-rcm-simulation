"""Guarded specimen acquisition. The high-resolution exposure never enters the FSM."""
import argparse
import json
from pathlib import Path
import uuid

import numpy as np

from capture_surface_seam import Capture, ROOT, SOURCES as BASE_SOURCES, S, write_new
from seam_rail_world import SeamRailWorld
from simtools import add_arguments, connect, sha256, stop_simulation

PLAN = ROOT/'config/task/seam_rail_v1.json'
OUT = ROOT/'validation/task/seam_rail_v1'
SOURCES = (*BASE_SOURCES, 'scripts/analysis/capture_seam_rail.py', 'scripts/analysis/score_seam_rail.py',
           'scripts/task/seam_rail_world.py', 'src/tissue/seam_rail.lua')


def high_exposure(w, sensor_path):
    sim = w.sim
    rgb, z = np.asarray(w.image).copy(), w.depth.copy()
    size = list(sim.getVisionSensorRes(w.cam))
    tm = sim.getSimulationTime()
    tcp = sim.getObjectMatrix(w.by[w.cutter+'_tcp'], -1)
    with w.truth.purpose('scoring'):
        nodes = w.truth.nodes()
    if size != [640, 480]:
        raise ValueError('capture must start at the native resolution')
    hi = [4*k for k in size]
    try:
        sim.setIntArray2Property(w.cam, 'resolution', hi)
        if list(sim.getVisionSensorRes(w.cam)) != hi:
            raise ValueError('high sensor resolution not applied')
        sim.handleVisionSensor(w.cam)
        raw, sr = sim.getVisionSensorImg(w.cam)
        rawz, sz = sim.getVisionSensorDepth(w.cam, 1)
        if list(sr) != hi or list(sz) != hi:
            raise ValueError('wrong high-resolution readback')
        im = np.frombuffer(bytes(raw), np.uint8).reshape(hi[1], hi[0], 3)[::-1]
        dep = np.frombuffer(bytes(rawz), '<f4').reshape(hi[1], hi[0])[::-1]
        np.savez_compressed(sensor_path, rgb=im, depth=dep)
    finally:
        sim.setIntArray2Property(w.cam, 'resolution', size)
        w.render()
    with w.truth.purpose('scoring'):
        restored = np.array_equal(w.truth.nodes(), nodes)
    if not (restored and sim.getSimulationTime() == tm and sim.getObjectMatrix(w.by[w.cutter+'_tcp'], -1) == tcp
            and list(sim.getVisionSensorRes(w.cam)) == size and np.array_equal(w.depth, z)
            and np.array_equal(w.image, rgb)):
        raise RuntimeError('frozen exposure did not restore native state')


class RailCapture(Capture):
    def save(self, phase, obs):
        super().save(phase, obs)
        frame = self.record['frames'][-1]
        if self.pull_only or self.record['scale'] == 4:
            path = self.folder/'high'/(frame['key']+'.npz')
            high_exposure(self.w, path)
            frame['high_sha256'] = sha256(path)


def acquire(args):
    cfgspec = json.loads(PLAN.read_bytes())
    p = ROOT/cfgspec['parent']
    seam = json.loads((ROOT/cfgspec['seam_parent']).read_bytes())
    setup = json.loads(p.read_bytes())
    base = ROOT/'config/task/cut_task.json'
    cfg = json.loads(base.read_bytes())
    args.scene = ROOT/setup['scene']
    hashes = {s: sha256(ROOT/s) for s in SOURCES}
    if (sha256(p) != seam['parent_sha256'] or sha256(args.scene) != setup['scene_sha256']
            or sha256(base) != setup['base_sha256'] or sha256(ROOT/cfgspec['seam_parent']) != cfgspec['seam_parent_sha256']
            or hashes['scripts/vision/ruler_readout.py'] != cfgspec['reader_sha256']):
        raise ValueError('pilot input drift')
    entry = next(x for x in cfgspec['development_runs'] if x['id'] == args.case)
    scale = None
    if args.case != 'pull':
        select = json.loads((args.out_root/'selection.json').read_bytes())
        chosen = select['selected']
        if (not chosen or chosen['rail'] != args.rail or select['spec_sha256'] != sha256(PLAN)
                or select['source_sha256'] != hashes):
            raise ValueError('working trials require a passing frozen preflight')
        scale = chosen['scale']
    folder = args.out_root/args.rail/args.case
    folder.mkdir(parents=True, exist_ok=False)
    for child in ('sensor', 'high', 'structure'):
        (folder/child).mkdir()
    cfg['cut'].update(control_source='truth', length_m=entry['goal_mm']/1000, advance_m=entry['cap_mm']/1000)
    cfg['perception'].update(exclude_tcp_marker=True)
    args.initial_view, args.initial_view_insertion_m = setup['view'][:2], setup['view'][2]
    r = dict(run_id=uuid.uuid4().hex, rail=args.rail, scale=scale, entry=entry, spec_sha256=sha256(PLAN),
             source_sha256=hashes, scene_sha256=sha256(args.scene), base_sha256=sha256(base),
             cfg=cfg, frames=[], observations=[], pull_motion_m={})
    client = sim = fsm = None
    try:
        client, sim = connect(args)
        w = SeamRailWorld(client, sim, args, cfg, mode='full')
        r.update(controller_sha256=w.controller_sha256, base_controller_sha256=w.base_controller_sha256)
        fsm = RailCapture(w, cfg, folder, r, bool(entry.get('pull_only')))
        fsm.run()
        r.update(view=w.initial_view_readback, events=w.call('tissueStats').get('cut_events') or [],
                 roles=w.roles_report, pins=w.call('seamRailPins'), marker=w.tip_marker_optics)
        if fsm.state == S.FAILED and w.sheet_origin is not None:
            # Failure evidence, never eligible to substitute for a loaded/working sample.
            w.render()
            np.savez_compressed(folder/'failure.npz', rgb=np.asarray(w.image), depth=w.depth.astype('<f4'))
    except Exception as exc:
        r['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if fsm is not None:
            r.update(fsm.report())
        if sim is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                r['cleanup_error'] = str(exc)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        write_new(folder/'oracle.json', r)
    print(args.rail, args.case, r.get('final_state'), r.get('error') or r.get('reason'), flush=True)
    return int(bool(r.get('error') or r.get('cleanup_error')) or r.get('final_state') != 'DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--rail', required=True, choices=('outer', 'inner'))
    parser.add_argument('--case', required=True, choices=('pull', 'a', 'b', 'c', 'd', 'e'))
    parser.add_argument('--out-root', type=Path, default=OUT)
    raise SystemExit(acquire(parser.parse_args()))
