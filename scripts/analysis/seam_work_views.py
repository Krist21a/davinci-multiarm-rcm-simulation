"""Actual guarded cameras on a matched working pose. No tool hiding, stepping or recutting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_surface_seam import Capture, ROOT, add_arguments, connect, sha256, stop_simulation, write_new
from contact_veto_pilot import OUT as DEVELOPMENT, verify_lock
from ruler_readout import crossings, inks
from seam_slide_world import SeamSlideWorld
from verified_ruler import classify

PLAN = ROOT/'config/task/seam_work_views_v1.json'
OUT = ROOT/'validation/task/seam_work_views_v1'


class Recorded(Exception):
    pass


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def capture_high(w, p):
    sim = w.sim
    try:
        sim.setIntArray2Property(w.cam, 'resolution', [2560, 1920])
        sim.handleVisionSensor(w.cam)
        raw, res = sim.getVisionSensorImg(w.cam)
        buf, sz = sim.getVisionSensorDepth(w.cam, 1)
        if list(res) != [2560, 1920] or list(sz) != list(res):
            raise ValueError('real high-resolution image not acquired')
        rgb = np.frombuffer(bytes(raw), np.uint8).reshape(1920, 2560, 3)[::-1]
        depth = np.frombuffer(bytes(buf), '<f4').reshape(1920, 2560)[::-1]
        y, x = np.where(inks(rgb) >= 0)
        if len(x) and 8 <= min(x.min(), y.min()) and x.max()+8 < 2560 and y.max()+8 < 1920:
            x0, y0, x1, y1 = x.min()-8, y.min()-8, x.max()+9, y.max()+9
        else:
            x0, y0, x1, y1 = 0, 0, 2560, 1920
        roi = (slice(y0, y1), slice(x0, x1))
        seed = int(digest(depth)[:16], 16)
        noisy = depth+np.random.default_rng(seed).normal(0., .0005, depth.shape)
        pred = [classify(rgb, z) for z in (depth, noisy)]
        np.savez_compressed(p, rgb=rgb[roi], depth=depth[roi], noisy=noisy[roi])
        return dict(rgb_sha256=digest(rgb), depth_sha256=digest(depth), box=[x0, y0, x1, y1],
                    roi_sha256=sha256(p), predictions=pred,
                    transects=[crossings(rgb[roi], z[roi]) for z in (depth, noisy)])
    finally:
        sim.setIntArray2Property(w.cam, 'resolution', [640, 480])
        w.render()


class Views(Capture):
    def save(self, phase, obs):
        self.record['frames'].append(phase)
        f = self.ref_frame
        if phase != 'work' or self.w.tick != f['tick']:
            return
        w, sim = self.w, self.w.sim
        with np.load(self.src/'sensor'/(f['key']+'.npz')) as original:
            rgb, z = original['rgb'], original['depth']
        with np.load(self.src/'structure'/(f['key']+'.npz')) as original:
            nodes, edges = original['nodes'], original['edges']
        poses = {a:self.tcp(a).copy() for a in ('psm1', 'psm2', 'psm3')}
        tm = sim.getSimulationTime()
        home = w.target_of(w.view)
        home_rt = w.camera()

        def matched():
            return (sim.getSimulationTime() == tm and w.jaws_closed()
                    and all(np.array_equal(self.tcp(a), q) for a, q in poses.items())
                    and np.array_equal(nodes, w.truth.nodes())
                    and np.array_equal(edges, np.asarray(w.call('tissueConstraints')).reshape(-1, 3)))

        if (not matched() or obs.cut_length_m != f['obs']['cut_length_m'] or not np.array_equal(rgb, w.image)
                or not np.array_equal(z, w.depth.astype('<f4')) or not np.array_equal(poses['psm2'], f['tcp'])):
            raise ValueError('working checkpoint not reproduced')
        for i, target in enumerate(self.targets):
            if i < self.first:
                continue
            rec = dict(index=i, target=target)
            self.record['views'].append(rec)
            if not np.allclose(home, target, atol=1e-6, rtol=0):
                rec['accepted'] = w.command(w.view, *target, 0)
            else:
                rec['accepted'] = True
            if rec['accepted']:
                rec['actual'] = w.target_of(w.view)
                if not np.allclose(target, rec['actual'], atol=1e-6, rtol=0) or not matched():
                    raise ValueError('camera target/physics mismatch')
                rec.update(capture_high(w, self.folder/f'view_{i:02}.npz'))
                if np.allclose(home, target, atol=1e-6, rtol=0):
                    with np.load(self.src/'high'/(f['key']+'.npz')) as high:
                        if digest(high['rgb']) != rec['rgb_sha256'] or digest(high['depth']) != rec['depth_sha256']:
                            raise ValueError('working high-resolution checkpoint differs')
                elif not w.command(w.view, *home, 0):
                    raise ValueError('camera restoration refused')
            w.render()
            rec['restoration_checks'] = dict(
                physics=bool(matched()), target=bool(np.allclose(w.target_of(w.view), home, atol=1e-6, rtol=0)),
                camera=all(np.allclose(x,y,atol=1e-8,rtol=0) for x,y in zip(w.camera(),home_rt)),
                rgb=bool(np.array_equal(w.image, rgb)), depth=bool(np.array_equal(w.depth.astype('<f4'), z)))
            if not all(rec['restoration_checks'].values()):
                raise ValueError('failed to restore original work observation')
            rec['restored'] = True
        raise Recorded()


def run(args):
    verify_lock()
    spec = json.loads(PLAN.read_bytes())
    table = ROOT/spec['view_table']
    decision = DEVELOPMENT/'decision.json'
    if sha256(table) != spec['view_table_sha256'] or sha256(decision) != spec['development_decision_sha256']:
        raise ValueError('unregistered inputs')
    src = DEVELOPMENT/'slide'/args.case
    r = json.loads((src/'oracle.json').read_bytes())
    score = next(x for x in json.loads(decision.read_bytes())['runs'] if x['case'] == args.case)
    if not score['mechanics_ok'] or sha256(src/'oracle.json') != score['oracle_sha256']:
        raise ValueError('invalid working reference')
    ref = [x for x in r['frames'] if x['phase'] == 'work'][-1]
    for kind, field in (('sensor','sensor_sha256'), ('structure','structure_sha256'), ('high','high_sha256')):
        if sha256(src/kind/(ref['key']+'.npz')) != ref[field]:
            raise ValueError('changed reference')
    args.scene = ROOT/'scenes/davinci_si_articulated_verified.ttt'
    args.initial_view, args.initial_view_insertion_m = r['view']['target'][:2], r['view']['target'][2]
    folder = args.out_root/args.case
    folder.mkdir(parents=True, exist_ok=False)
    js = dict(plan_sha256=sha256(PLAN), code_sha256=sha256(Path(__file__)), ref_sha256=sha256(src/'oracle.json'),
              frames=[], views=[], observations=[], matched=False)
    client = sim = None
    try:
        client, sim = connect(args)
        w = SeamSlideWorld(client, sim, args, r['cfg'], mode='full')
        if w.controller_sha256 != r['controller_sha256'] or sha256(args.scene) != r['scene_sha256']:
            raise ValueError('controller or scene differs')
        fsm = Views(w, r['cfg'], folder, js, False)
        fsm.src, fsm.ref_frame = src, ref
        fsm.first = args.first_index
        fsm.targets = json.loads(table.read_bytes())['candidates_deg_deg_m']
        try:
            fsm.run()
        except Recorded:
            js['matched'] = True
        js['state_at_probe'] = fsm.state.name
    except Exception as exc:
        js['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            stop_simulation(client, sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        write_new(folder/'views.json', js)
    print(args.case, js.get('error') or js['matched'])
    return int(not js['matched'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out-root', type=Path, default=OUT)
    ap.add_argument('--first-index', type=int, choices=range(13), default=0)
    ap.add_argument('--case', choices=('a', 'b'), required=True)
    raise SystemExit(run(ap.parse_args()))
