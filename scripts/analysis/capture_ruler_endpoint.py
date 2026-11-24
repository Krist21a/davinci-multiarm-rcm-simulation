"""Capture strictly separated RGB-D and structural labels; never a vision controller."""
import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/task'))
from cut_fsm import CutTaskFSM, S
from ruler_world import RulerWorld
from simtools import add_arguments, connect, sha256, stop_simulation

PLAN = ROOT / 'config/task/ruler_endpoint_v1.json'
BASE = ROOT / 'config/task/cut_task.json'


class CaptureFSM(CutTaskFSM):
    def save(self, phase, obs):
        w, folder = self.w, self.folder
        rgb, depth = np.asarray(w.image).copy(), w.depth.astype(np.float32)
        key = uuid.uuid4().hex
        path = folder / (key+'.npz')
        np.savez_compressed(path, rgb=rgb, depth=depth)
        with w.truth.purpose('scoring'):
            length = w.truth.cut_length()
            nodes = w.truth.nodes()
        if length != obs.cut_length_m or obs.faults or obs.torn:
            raise RuntimeError('unsafe or mismatched capture')
        self.frames.append(dict(key=key, phase=phase, tick=w.tick, length_m=length, sha256=sha256(path),
                                nodes_sha256=hashlib.sha256(nodes.tobytes()).hexdigest(),
                                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                                depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest()))

    def on_align(self, obs):
        if not self.frames and obs.cut_length_m == 0 and self.grasps_ok(obs) and len(self.retractors) == 2:
            self.save('loaded', obs)
        super().on_align(obs)

    def _complete(self, obs):
        self.save('work', obs)
        super()._complete(obs)
        if self.state == S.DONE:
            self.save('park', self.w.observe(inject_events=False))


def main(args):
    protocol = json.loads(PLAN.read_bytes())
    args.scene = ROOT / protocol['scene']
    if sha256(BASE) != protocol['base_sha256'] or sha256(args.scene) != protocol['scene_sha256']:
        raise ValueError('registered base inputs differ')
    cfg = json.loads(BASE.read_bytes())
    if (args.goal not in (6, 7, 8, 9) or args.cap not in sum(protocol['sets'].values(), [])):
        raise ValueError('unregistered acquisition goal or cap')
    cfg['cut'].update(length_m=args.goal*.001, advance_m=args.cap*.001, control_source='truth')
    cfg['perception'].update(exclude_tcp_marker=True)
    args.initial_view, args.initial_view_insertion_m = protocol['view'][:2], protocol['view'][2]
    args.out.mkdir(exist_ok=False)
    js = dict(run_id=uuid.uuid4().hex, goal_mm=args.goal, cap_mm=args.cap, protocol_sha256=sha256(PLAN),
              config=cfg, scene_sha256=sha256(args.scene), base_sha256=sha256(BASE), error=None,
              frames=[], source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in
              (Path(__file__).resolve(), ROOT/'scripts/task/ruler_world.py', ROOT/'src/tissue/ruler_print.lua',
               ROOT/'scripts/task/sim_world.py', ROOT/'scripts/task/cut_fsm.py', ROOT/'scripts/task/ground_truth.py')})
    client = sim = None
    try:
        client, sim = connect(args)
        w = RulerWorld(client, sim, args, cfg, mode='full')
        js.update(controller_sha256=w.controller_sha256, base_controller_sha256=w.base_controller_sha256,
                  marker_optics=w.tip_marker_optics)
        fsm = CaptureFSM(w, cfg)
        fsm.folder, fsm.frames = args.out, js['frames']
        fsm.run()
        js.update(fsm.report(), view=w.initial_view_readback, closures=w.call('tissueStats').get('cut_events'),
                  roles=w.roles_report)
    except Exception as exc:
        js['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                js['cleanup_error'] = str(exc)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        with (args.out/'oracle.json').open('x', encoding='utf-8', newline='\n') as f:
            json.dump(js, f, indent=2, default=lambda v: np.asarray(v).tolist(), allow_nan=False)
            f.write('\n')
    print('acquisition:', js.get('final_state'), js['error'] or js.get('reason'))
    return int(bool(js['error'] or js.get('cleanup_error')) or js.get('final_state') != 'DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--goal', required=True, type=int)
    parser.add_argument('--cap', required=True, type=float)
    parser.add_argument('--out', required=True, type=Path)
    raise SystemExit(main(parser.parse_args()))
