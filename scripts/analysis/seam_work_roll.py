"""Single closed-jaw alternate orientation at a reproduced working checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np

from capture_seam_rail import high_exposure
from capture_surface_seam import Capture, ROOT, add_arguments, connect, sha256, stop_simulation, write_new
from contact_veto_pilot import OUT as DEV, verify_lock
from seam_slide_world import SeamSlideWorld
import tool_pose

SPEC = ROOT/'config/task/seam_work_roll_v1.json'
OUT = ROOT/'validation/task/seam_work_roll_v1'


class Finished(Exception):
    pass


class Probe(Capture):
    def save(self, phase, obs):
        self.record['frames'].append(phase)
        if phase != 'work' or self.w.tick != self.f['tick']:
            return
        w, f = self.w, self.f
        idx = f['key']+'.npz'
        sensor, struct = self.src/'sensor'/idx, self.src/'structure'/idx
        if sha256(sensor) != f['sensor_sha256'] or sha256(struct) != f['structure_sha256']:
            raise ValueError('invalid reference')
        with np.load(sensor) as im, np.load(struct) as data:
            if (not np.array_equal(im['rgb'], w.image) or not np.array_equal(im['depth'], w.depth.astype('<f4'))
                    or not np.array_equal(data['nodes'], w.truth.nodes())):
                raise ValueError('checkpoint mismatch')
            edges = data['edges']
        pose = self.tcp(self.cutter)
        if (not self.w.jaws_closed() or not np.array_equal(pose, f['tcp'])
                or not np.array_equal(edges, np.asarray(w.call('tissueConstraints')).reshape(-1, 3))):
            raise ValueError('mechanical checkpoint mismatch')
        self.record.update(matched=True, before_tick=w.tick, before_tcp=pose.tolist(), before_obs=vars(obs))
        point = pose[:, 3]
        jaw, hinge = pose[:, w.TOOL_AXIS_COL], -pose[:, w.TOOL_YAW_COL]
        state = w._tool_q(self.cutter)
        if state is None:
            self.record['reason'] = 'unavailable joint state'
            raise Finished()
        q0, ws = state
        lo, hi = w._tool_limits(self.cutter, ws)
        seed = q0.copy()
        seed[3] = np.clip(q0[3] + (np.pi if hi[3]-q0[3] >= q0[3]-lo[3] else -np.pi), lo[3], hi[3])
        fk = w._tool_fk(self.cutter)
        self.record.update(q0=q0.tolist(), seed=seed.tolist())
        try:
            motion = tool_pose.solve(fk, seed, (lo, hi), point, jaw, hinge, axis_col=w.TOOL_AXIS_COL,
                                     yaw_col=w.TOOL_YAW_COL)
        except RuntimeError as exc:
            self.record['reason'] = f'alternate-seed FK guard refused: {exc}'
            raise Finished() from exc
        if motion is not None:
            self.record['solution'] = motion
        if motion is None or not motion['converged']:
            self.record['reason'] = 'guarded IK refused or did not converge'
            raise Finished()
        # The generic solver treats hinge +/- as equivalent. A no-op is not the proposed intervention.
        check = w._tool_fk(self.cutter)(motion['q'])
        cosine = np.cos(np.deg2rad(5))
        if check[:, w.TOOL_YAW_COL]@hinge < cosine:
            self.record['reason'] = 'solver retained original physical blade orientation'
            raise Finished()
        intervals = np.array([5, 5, .0005, *([np.deg2rad(5)]*3)])
        steps = max(1, int(np.ceil(np.max(abs(motion['q']-q0)/intervals))))
        for q in np.linspace(q0, motion['q'], steps+1)[1:]:
            try:
                trial = fk(q)
            except RuntimeError as exc:
                self.record['reason'] = f'intermediate FK guard refused: {exc}'
                raise Finished() from exc
            if np.linalg.norm(trial[:, 3]-point) > .0005:
                self.record['reason'] = 'closed pivot path exceeds 0.5 mm tip excursion'
                raise Finished()
        before_hits = w.call('tissueStats').get('cut_events') or []
        if not w.tool_command(self.cutter, motion['q'], 10):
            self.record['reason'] = 'guarded tool command refused'
            raise Finished()
        after = w.observe()
        actual = self.tcp(self.cutter)
        post = w.call('tissueStats')
        safe = (w.jaws_closed() and not after.torn and not after.faults and self.tension_ok(after)
                and all(after.grasped.get(a, 0) >= self.cfg['grasp']['min_nodes'] for a in (self.primary, self.assist))
                and post.get('cut_events') == before_hits and after.cut_length_m == obs.cut_length_m
                and np.array_equal(edges, np.asarray(w.call('tissueConstraints')).reshape(-1, 3)))
        qualifies = bool(safe and np.linalg.norm(actual[:, 3]-point) <= .0005 and
                         actual[:, w.TOOL_AXIS_COL]@jaw >= cosine and actual[:, w.TOOL_YAW_COL]@hinge >= cosine)
        self.record.update(after_tcp=actual, after_obs=vars(after), pose_qualified=qualifies)
        if qualifies:
            np.savez_compressed(self.folder/'native.npz', rgb=np.asarray(w.image), depth=w.depth.astype('<f4'))
            high_exposure(w, self.folder/'high.npz')
            self.record['high_sha256'] = sha256(self.folder/'high.npz')
        raise Finished()


def run(args):
    verify_lock()
    spec = json.loads(SPEC.read_bytes())
    decision = DEV/'decision.json'
    src = DEV/'slide'/args.case
    r = json.loads((src/'oracle.json').read_bytes())
    q = next(x for x in json.loads(decision.read_bytes())['runs'] if x['case'] == args.case)
    f = [f for f in r['frames'] if f['phase'] == 'work'][-1]
    if (sha256(decision) != spec['decision_sha256'] or not q['mechanics_ok'] or
            sha256(src/'oracle.json') != q['oracle_sha256'] or f['prefix_mm'] != spec['cases'][args.case]):
        raise ValueError('unregistered checkpoint')
    args.scene = ROOT/'scenes/davinci_si_articulated_verified.ttt'
    args.initial_view, args.initial_view_insertion_m = r['view']['target'][:2], r['view']['target'][2]
    folder = OUT/args.case
    folder.mkdir(parents=True, exist_ok=False)
    js = dict(protocol_sha256=sha256(SPEC), source_sha256=sha256(Path(__file__)), ref_sha256=sha256(src/'oracle.json'),
              frames=[], observations=[], matched=False, pose_qualified=False)
    client = sim = None
    try:
        client, sim = connect(args)
        w = SeamSlideWorld(client, sim, args, r['cfg'], mode='full')
        if w.controller_sha256 != r['controller_sha256'] or sha256(args.scene) != r['scene_sha256']:
            raise ValueError('controller/scene mismatch')
        task = Probe(w, r['cfg'], folder, js, False)
        task.src, task.f = src, f
        try:
            task.run()
        except Finished:
            pass
    except Exception as exc:
        js['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            stop_simulation(client, sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        write_new(folder/'probe.json', js)
    print(args.case, js.get('error') or js.get('reason') or js['pose_qualified'])
    return int(bool(js.get('error')) or not js['matched'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--case', required=True, choices=('a','b'))
    raise SystemExit(run(parser.parse_args()))
