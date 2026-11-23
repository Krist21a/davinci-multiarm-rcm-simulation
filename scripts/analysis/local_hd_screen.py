"""Capture or read development evidence. No prediction is returned to the cutting FSM."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np

from capture_surface_seam import Capture, ROOT, S, add_arguments, connect, sha256, stop_simulation, write_new
from connected_ruler import classify as connected_prefix, local_front
from cut_perception import CutPerception
from seam_topology import material_state
from sim_world import SimWorld

SPEC = ROOT / 'config/task/m5_local_hd_dev_v1.json'
BASE = ROOT / 'config/task/cut_task.json'
SOURCES = [Path(__file__), ROOT/'scripts/analysis/capture_surface_seam.py',
           ROOT/'scripts/vision/connected_ruler.py', ROOT/'scripts/vision/ruler_readout.py',
           ROOT/'scripts/vision/cut_perception.py', ROOT/'scripts/task/sim_world.py',
           ROOT/'scripts/task/cut_fsm.py', ROOT/'scripts/task/ground_truth.py',
           ROOT/'scripts/task/seam_topology.py', ROOT/'scripts/task/cutter_view.py',
           ROOT/'scripts/task/workspace.py', ROOT/'scripts/task/tool_pose.py',
           ROOT/'scripts/task/visibility.py', ROOT/'scripts/vision/material_coordinate.py',
           ROOT/'scripts/build/simtools.py', ROOT/'src/endoscope/rcm_controller.lua',
           ROOT/'src/instruments/articulation.lua', ROOT/'src/instruments/capability_guard.lua',
           ROOT/'src/tissue/tissue.lua', BASE]


def registration(folder):
    data = dict(criteria=json.loads(SPEC.read_bytes()), spec_sha256=sha256(SPEC),
                sources={p.relative_to(ROOT).as_posix(): sha256(p) for p in SOURCES})
    path = folder/'registration.json'
    if not path.exists():
        write_new(path, data)
    if json.loads(path.read_bytes()) != data:
        raise ValueError('source or criteria changed; use a new development directory')
    return data


def expose(w, path, optics):
    sim = w.sim
    native, size = [w.W, w.H], optics['size']
    def signature():
        with w.truth.purpose('scoring'):
            return (sim.getSimulationTime(), w.truth.nodes().tobytes(),
                    tuple(w.call('tissueConstraints')), tuple(sim.getObjectMatrix(w.by[w.cutter+'_tcp'], -1)))
    before = signature()
    rgb0, z0 = np.asarray(w.image).copy(), w.depth.copy()
    angle = np.rad2deg(sim.getObjectFloatParam(w.cam, sim.visionfloatparam_perspective_angle))
    if (not np.isclose(angle, optics['horizontal_fov_deg'], rtol=0, atol=.0001)
            or list(sim.getVisionSensorRes(w.cam)) != native):
        raise ValueError('unexpected native optics')
    try:
        sim.setIntArray2Property(w.cam, 'resolution', size)
        sim.handleVisionSensor(w.cam)
        raw, sr = sim.getVisionSensorImg(w.cam)
        buf, sz = sim.getVisionSensorDepth(w.cam, 1)
        if list(sr) != size or list(sz) != size or signature() != before:
            raise ValueError('wrong acquisition dimensions or moving checkpoint')
        width, height = size
        im = np.frombuffer(bytes(raw), np.uint8).reshape(height, width, 3)[::-1]
        z = np.frombuffer(bytes(buf), '<f4').reshape(height, width)[::-1]
        with path.open('xb') as f:
            np.savez_compressed(f, rgb=im, depth=z)
    finally:
        sim.setIntArray2Property(w.cam, 'resolution', native)
        w.render()
    if (signature() != before or not np.array_equal(w.image, rgb0) or not np.array_equal(w.depth, z0)
            or list(sim.getVisionSensorRes(w.cam)) != native):
        raise ValueError('sensor restore or frozen checkpoint mismatch')


class DevelopmentCapture(Capture):
    def save(self, phase, obs):
        super().save(phase, obs)
        row = self.record['frames'][-1]
        dst = self.folder/'hd'/(row['key']+'.npz')
        expose(self.w, dst, self.record['optics'])
        row['hd_sha256'] = sha256(dst)

    def _complete(self, obs):
        w = self.w
        if not w.jaws_closed():
            return self.go(S.FAILED, 'look-back requires measured closed jaws', obs)
        with w.truth.purpose('scoring'):
            events = w.truth.tissue_stats()['cut_events']
            edges = list(w.call('tissueConstraints'))
        before = self.tcp(self.cutter)
        if not w.clear_view(.003):
            return self.go(S.FAILED, 'guarded look-back refused', obs)
        seen = w.observe(inject_events=False)
        after = self.tcp(self.cutter)
        with w.truth.purpose('scoring'):
            changed = edges != list(w.call('tissueConstraints')) or events != w.truth.tissue_stats()['cut_events']
        residual = np.linalg.norm(after[:, 3]-before[:, 3]+.003*before[:, w.TOOL_AXIS_COL])
        gap = np.linalg.norm(after[:, 3]-self.ready[:, 3])
        self.record['look'] = dict(residual_m=residual, distance_to_ready_m=gap)
        if (changed or seen.faults or seen.torn or not w.jaws_closed() or not self.grasps_ok(seen)
                or not self.tension_ok(seen) or abs(seen.cut_length_m-obs.cut_length_m) > 1e-12
                or not residual <= .0005 or not gap > .0005):
            return self.go(S.FAILED, 'unqualified look-back', seen)
        self.save('look', seen)
        super()._complete(seen)


def collect(args, folder, lock):
    spec = lock['criteria']
    run = next(r for r in spec['runs'] if r['name'] == args.case)
    dst = folder/args.case
    dst.mkdir(exist_ok=False)
    for part in ('sensor', 'structure', 'hd'):
        (dst/part).mkdir()
    cfg = json.loads(BASE.read_bytes())
    cfg['sheet'] = spec['sheet']
    cfg['cut'].update(control_source='truth', length_m=run['goal_mm']*.001, advance_m=.003)
    cfg['perception'].update(exclude_tcp_marker=True, ray_marker_fit=True)
    args.scene = ROOT/spec['scene']
    if sha256(args.scene) != spec['scene_sha256']:
        raise ValueError('wrong scene')
    args.initial_view, args.initial_view_insertion_m = spec['initial_view'][:2], spec['initial_view'][2]
    report = dict(run_id=uuid.uuid4().hex, config=cfg, registration_sha256=sha256(folder/'registration.json'),
                  scope='truth-controlled development acquisition', frames=[], observations=[],
                  pull_motion_m={}, optics=spec['optics'])
    client = sim = fsm = None
    try:
        client, sim = connect(args)
        w = SimWorld(client, sim, args, cfg, mode='full')
        report['controller_sha256'] = w.controller_sha256
        if len(w.markers) != 3:
            raise ValueError('three staged physical markers required')
        # Only rigid instrument calibration and camera intrinsics, never runtime geometry.
        cal = w._dump_meta()['estimator_args']
        width, height = spec['optics']['size']
        cal.update(W=width, H=height, f=width/2/np.tan(np.deg2rad(spec['optics']['horizontal_fov_deg']/2)))
        write_new(dst/'calibration.json', cal)
        report['calibration_sha256'] = sha256(dst/'calibration.json')
        fsm = DevelopmentCapture(w, cfg, dst, report, run.get('pull_only', False))
        fsm.run()
        report.update(roles=w.roles_report, initial_view=w.initial_view_readback,
                      at_ready=w.at_ready_pose(), closed=w.jaws_closed())
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if fsm is not None:
            report.update(fsm.report())
        try:
            if sim is not None:
                stop_simulation(client, sim)
        except Exception as exc:
            report['cleanup_error'] = str(exc)
        finally:
            if client is not None:
                client.socket.close(linger=0)
                client.context.term()
            write_new(dst/'oracle.json', report)
    print(args.case, report.get('final_state'), report.get('error', report.get('reason')), flush=True)
    return int(report.get('final_state') != 'DONE' or bool(report.get('error') or report.get('cleanup_error')))


def depths(z, sigma):
    yield z
    seed = int(hashlib.sha256(z.tobytes()).hexdigest()[:16], 16)
    yield z + np.random.default_rng(seed).normal(0, sigma, z.shape)


def synthetic(sigma):
    # Rasterized sided material, no truth coordinates enter either classifier.
    def image(n):
        im = np.zeros((100, 100, 3), np.uint8)
        z = np.full(im.shape[:2], .15)
        for j in range(13):
            rows = slice(10+j*6, 16+j*6)
            r, k = 32+48*(j % 4), 35+46*(j//4)
            im[rows, 20:49], im[rows, 49:79] = (r, 235, k), (r, k, 235)
            z[rows, 20:79] = .105
        im[10:10+n*6, 46:52], z[10:10+n*6, 46:52] = 0, .15
        return im, z

    records = []
    for case in (0, 3, 6, 8, 9, 10, 11, 'bridge', 'half_left', 'half_right', 'foreground', 'invalid', 'shutter'):
        im, z = image(case if isinstance(case, int) else 9)
        intact, sheet = image(0)
        expect = case if isinstance(case, int) and 0 < case <= 10 else None
        if case == 'bridge':
            im[28:34], z[28:34] = intact[28:34], sheet[28:34]
        if case in ('half_left', 'half_right'):
            for i in range(6):
                xs = slice(46, 46+i+1) if case == 'half_left' else slice(46+i, 52)
                im[58+i, xs], z[58+i, xs] = intact[58+i, xs], .105
        if case == 'foreground':
            im[58:64, 46:52], z[58:64, 46:52] = 90, .02
        if case == 'invalid':
            z[58:64] = np.nan
        if case == 'shutter':
            im[:], z[:] = 90, .02
        local, full = [], []
        for d in depths(z, sigma):
            local.append(local_front(im, d)['prefix_mm'])
            full.append(connected_prefix(im, d)['decision'])
        want = 'UNKNOWN' if expect is None else 'STOP' if expect >= 9 else 'BELOW'
        records.append(dict(case=case, local=local, full=full,
                            passed=(all(v in (None, 8) for v in local) if case in ('half_left', 'half_right')
                                    else local == [expect]*2) and
                            (all(v in ('UNKNOWN', 'BELOW') for v in full) if case in ('half_left', 'half_right')
                             else full == [want]*2)))
    return records


def score(folder, lock):
    spec = lock['criteria']
    sigma = spec['optics']['depth_noise_m']
    controls = synthetic(sigma)
    pred = {}
    cases = [r['name'] for r in spec['runs']]
    for case in cases:
        dst = folder/case
        if not (dst/'calibration.json').exists():  # failure is counted in the join, not substituted
            continue
        est = CutPerception(**json.loads((dst/'calibration.json').read_bytes()))
        for p in sorted((dst/'hd').glob('*.npz')):
            digest = sha256(p)
            with np.load(p, allow_pickle=False) as d:
                if set(d.files) != {'rgb', 'depth'}:
                    raise ValueError('RGB-D only')
                im, z = d['rgb'], d['depth']
            if im.shape != (960, 1280, 3) or z.shape != im.shape[:2] or im.dtype != np.uint8:
                raise ValueError('not an actual registered-size image')
            key = (case, p.stem)
            values = []
            for depth in depths(z, sigma):
                sc, _ = est._scissors(im, depth)
                values.append(dict(local=local_front(im, depth), scissors=sc))
            pred[key] = dict(key=p.stem, case=case, hd_sha256=digest, values=values)
    write_new(folder/'predictions.json', dict(registration_sha256=sha256(folder/'registration.json'),
              synthetic=controls, images=list(pred.values())))
    cover, used, rows, issues = set(), set(), [], []
    for case in cases:
        dst = folder/case
        if not (dst/'oracle.json').exists():
            issues.append(f'{case}: missing acquisition')
            continue
        o = json.loads((dst/'oracle.json').read_bytes())
        phases = [r['phase'] for r in o['frames']]
        if (o.get('final_state') != 'DONE' or o.get('error') or o.get('cleanup_error')
                or o['registration_sha256'] != sha256(folder/'registration.json')
                or o.get('calibration_sha256') != (sha256(dst/'calibration.json')
                                                   if (dst/'calibration.json').is_file() else None)
                or set(o.get('roles') or {}) != set(o['config']['roles'].values())
                or not all(r['ok'] for r in o.get('roles', {}).values())
                or not np.allclose((o.get('initial_view') or {}).get('target', [np.nan]*3),
                                   spec['initial_view'], atol=1e-6, rtol=0)
                or (case == 'pull' and (phases != ['loaded', 'pulled'] or
                    len(o['pull_motion_m']) != 2 or not all(v > 0 for v in o['pull_motion_m'].values())))
                or (case != 'pull' and (not o.get('at_ready') or not o.get('closed')
                    or phases[:1] != ['loaded'] or phases[-3:] != ['work', 'look', 'park']))):
            issues.append(f'{case}: missing or failed mechanical/pose evidence')
        by_bite = {}
        for f in o['frames']:
            k = (case, f['key'])
            if k not in pred or 'hd_sha256' not in f:
                issues.append(f"{case}/{f['phase']}: incomplete exposure")
                continue
            v = pred[k]
            if k in used or v['hd_sha256'] != f['hd_sha256'] or sha256(
                    dst/'structure'/(f['key']+'.npz')) != f['structure_sha256']:
                raise ValueError('mismatched or duplicated scored image')
            used.add(k)
            with np.load(dst/'structure'/(f['key']+'.npz'), allow_pickle=False) as truth:
                state = material_state(truth['edges'])
                if (not np.array_equal(truth['faces'], state['faces']) or
                        (state['prefix_rows'], state['farthest_row_end']) != (f['prefix_mm'], f['farthest_mm'])):
                    raise ValueError('inconsistent structural witness')
            mm, phase = state['prefix_rows'], f['phase']
            read = [x['local']['prefix_mm'] for x in v['values']]
            markers = [x['scissors']['valid'] for x in v['values']]
            if case == 'pull' and mm != 0 or f['farthest_mm'] > 10 or f['obs']['torn'] or f['obs']['faults']:
                issues.append(f'{case}/{phase}: mechanical safety')
            if any(n is not None and n != mm for n in read):
                issues.append(f'{case}/{phase}: false local ordinal')
            ok = all(markers) and read == [mm, mm]
            rows.append(dict(case=case, phase=phase, bite=f['bites'], mm=mm,
                             local_mm=read, scissors=markers, joint_ok=ok))
            if phase in ('work', 'look'):
                if not f['closed']:
                    issues.append(f'{case}/{phase}: open jaws')
                by_bite.setdefault((f['bites'], mm), []).append(ok)
                cover.add(mm)
        for (bite, mm), outcomes in by_bite.items():
            if mm in (3, 6, 8, 9, 10) and not any(outcomes):
                issues.append(f'{case}/bite{bite}: no joint observation at {mm} mm')
    if not {3, 6, 8, 9, 10} <= cover:
        issues.append('missing length coverage')
    if not all(r['passed'] for r in controls):
        issues.append('synthetic controls failed')
    if used != set(pred):
        issues.append('unmatched sensor image')
    result = dict(criteria=spec, registration_sha256=sha256(folder/'registration.json'),
                  predictions_sha256=sha256(folder/'predictions.json'), coverage_mm=sorted(cover),
                  issues=issues, rows=rows, passed=not issues, control_enabled=False, confirmation_used=False)
    write_new(folder/'decision.json', result)
    print('local HD development:', result['passed'], issues)
    return int(bool(issues))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--case', choices=['pull', 'a', 'b', 'c'])
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    if not options.case and not (options.out/'registration.json').is_file():
        parser.error('no registered acquisition to score')
    frozen = registration(options.out)
    raise SystemExit(collect(options, options.out, frozen) if options.case else score(options.out, frozen))
