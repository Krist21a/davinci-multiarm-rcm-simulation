"""Collect RGB-D + truth-label episodes for the registered learned-perception study.

Protocol: config/learn/learned_perception_v1.json (binding; its pinned scene and base-config
sha256 are verified at startup). One headless CoppeliaSim instance per worker (own RPC port,
settings-profile suffix, ASCII stage dir); episodes run sequentially in-process exactly like
run_cut_task.py does: connect -> SimWorld -> CutTaskFSM.run -> final observe -> stop_simulation.
LearnWorld overrides only _perception_tick so the shadow estimator never runs; every label read
is tagged GroundTruth.purpose('dataset_label'). run_cut_task.py, sim_world.py, cut_fsm.py,
configs and scenes are not modified.

Auxiliary views are an approximation of reachable views for training only: the absolute
(yaw, pitch, insertion) target is drawn uniformly inside the registered ECM envelope and the
recorded delta = target - live ECM target readback is applied to the current camera frame as
rotations about the camera's own y/x axes pivoting on the ECM RCM point, plus a translation
along the rotated optical axis. They never feed control; the registered proof requires
identical nodes_sha256 traces with and without aux renders (phaseA_identity_*).

Frames go to {data_root}/learned_perception_v1/{tag}/ep{index:05d}/; per-episode manifests are
copied to validation/task/learned_perception_v1/{tag}/ and the run summary to
validation/task/learned_perception_v1/{tag}.json. 'holdout' is refused: it requires a sealed
model-freeze record that does not exist yet.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LEARN = Path(__file__).resolve().parent
TASK = LEARN.parent / 'task'
sys.path.insert(0, str(LEARN))
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK.parent / 'build'))
from throughput_probe import launch, sha256, shutdown, write_json  # noqa: E402
from run_cut_task import resolve_perception_flags, scenario_events  # noqa: E402
from cut_fsm import CutTaskFSM, S, completion_issues  # noqa: E402
from sim_world import SimWorld  # noqa: E402
from simtools import REPO, connect, matrix_to_np, stop_simulation  # noqa: E402

PROTOCOL = REPO / 'config' / 'learn' / 'learned_perception_v1.json'
SCENE = REPO / 'scenes' / 'davinci_si_articulated_verified.ttt'
BASE_CFG = REPO / 'config' / 'task' / 'cut_task.json'
HD_RESOLUTION = (1280, 960)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


def _norm(weights):
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def protocol():
    return json.loads(PROTOCOL.read_bytes())


def episode_params(split, index, proto=None):
    """Deterministic per-episode draw, in the registered order."""
    proto = proto or protocol()
    r = proto['data_generation']['episode_randomisation']
    seed = int.from_bytes(hashlib.sha256(f'learned_perception_v1/{split}/{index}'.encode())
                          .digest()[:8], 'big')
    rng = np.random.default_rng(seed)
    target, scen = r['target_length_m'], r['scenario']
    env = r['ecm_initial_view_deg_deg_m']
    return dict(
        index=index, split=split, seed=seed,
        advance_m=float(rng.choice(r['advance_m']['choices'])),
        target_length_m=float(rng.choice(target['choices'], p=_norm(target['weights']))),
        scenario=str(rng.choice(scen['choices'], p=_norm(scen['weights']))),
        yaw_deg=float(rng.uniform(*env['yaw'])),
        pitch_deg=float(rng.uniform(*env['pitch'])),
        insertion_m=float(rng.uniform(*env['insertion'])),
        aux_seed=int(rng.integers(2 ** 63)))


def aux_targets(aux_seed, tick, k, envelope):
    """K absolute (yaw_deg, pitch_deg, insertion_m) targets, uniform inside the envelope."""
    rng = np.random.default_rng([aux_seed, tick])
    return np.array([[rng.uniform(*envelope['yaw']), rng.uniform(*envelope['pitch']),
                      rng.uniform(*envelope['insertion'])] for _ in range(k)])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def aux_pose(cam_R, cam_t, rcm_world, delta):
    """Training-only approximation of a reachable ECM view (see module docstring).

    delta = (dyaw_deg, dpitch_deg, dins_m): rotate the camera's own frame by dyaw about its y
    axis then dpitch about its x axis, pivoting on the ECM RCM world point, then translate
    dins along the rotated optical axis (+z column, the direction depth is measured along)."""
    cam_R, cam_t, rcm = np.asarray(cam_R), np.asarray(cam_t), np.asarray(rcm_world, dtype=float)
    dyaw, dpitch, dins = delta
    W = cam_R @ _ry(math.radians(dyaw)) @ _rx(math.radians(dpitch)) @ cam_R.T
    R2 = W @ cam_R
    t2 = rcm + W @ (cam_t - rcm) + dins * R2[:, 2]
    return R2, t2


def atomic_npz(path_noext, arrays):
    tmp = path_noext.with_name(path_noext.name + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path_noext.with_name(path_noext.name + '.npz'))


class LearnWorld(SimWorld):
    """SimWorld whose _perception_tick writes dataset frames; the shadow estimator never runs.

    capture = dict(out_dir=Path, aux_views=int, hd_twin=bool, aux_seed=int, envelope=dict).
    The HD twin and the aux sensor are separate render-only vision sensors with the ECM
    camera's optics; the ECM sensor, its pose and its image are never moved or replaced."""

    def __init__(self, *a, capture=None, **kw):
        self._capture = dict(capture or {})
        self._sensors = {}
        self._twin = None
        self._aux_sensor = None
        super().__init__(*a, **kw)
        if self._capture.get('hd_twin'):
            self._twin = self._spawn_sensor(HD_RESOLUTION, 'hd_twin')
        if self._capture.get('aux_views', 0) > 0:
            self._aux_sensor = self._spawn_sensor((self.W, self.H), 'aux')

    def _spawn_sensor(self, resolution, name):
        """Render-only vision sensor with the ECM camera's optics; verified type + resolution."""
        sim = self.sim
        handle = None
        try:
            out = sim.copyPasteObjects([self.cam], 0)   # the sensor itself, without children
            handle = out[0] if isinstance(out, (list, tuple)) else out
            if sim.getObjectType(handle) != sim.object_visionsensor_type:
                sim.removeObjects([handle])
                handle = None
        except Exception:
            handle = None
        if handle is not None:
            self._sensors[name] = 'copyPasteObjects'
        else:
            c = self.endo['camera']
            handle = sim.createVisionSensor(
                1 + 2, [int(resolution[0]), int(resolution[1]), 0, 0],
                [c['near_m'], c['far_m'], math.radians(c['view_angle_deg']),
                 0.001, 0, 0, 0.95, 0.95, 0.95, 0, 0])
            self._sensors[name] = 'createVisionSensor'
            for p in ('rendering_attributes', 'disabled_light_components',
                      'entity_to_render', 'render_mode'):
                param = getattr(sim, f'visionintparam_{p}', None)
                if param is not None:
                    try:
                        sim.setObjectInt32Param(handle, param,
                                                sim.getObjectInt32Param(self.cam, param))
                    except Exception:
                        pass
        sim.setObjectInt32Param(handle, sim.visionintparam_resolution_x, int(resolution[0]))
        sim.setObjectInt32Param(handle, sim.visionintparam_resolution_y, int(resolution[1]))
        if list(sim.getVisionSensorRes(handle)) != list(resolution):
            raise RuntimeError(f'{name} sensor resolution {resolution} did not apply')
        if sim.getObjectType(handle) != sim.object_visionsensor_type:
            raise RuntimeError(f'{name} handle is not a vision sensor (no shape allowed)')
        sim.setObjectParent(handle, -1, True)   # world-absolute pose is set before every render
        return handle

    @staticmethod
    def _pose_sensor(sim, handle, R, t):
        sim.setObjectMatrix(handle, -1, np.column_stack([R, t]).flatten().tolist())

    def _snap(self, handle, W, H):
        """handleVisionSensor + readback on a separate sensor; self.image/depth untouched."""
        sim = self.sim
        sim.handleVisionSensor(handle)
        raw, res = sim.getVisionSensorImg(handle)
        d, _ = sim.getVisionSensorDepth(handle, 1)
        if list(res) != [W, H]:
            raise RuntimeError(f'sensor returned resolution {list(res)}, expected {[W, H]}')
        rgb = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(H, W, 3)[::-1].copy()
        depth = np.array(sim.unpackFloatTable(d), dtype=np.float32).reshape(H, W)[::-1]
        return rgb, depth

    def _intrinsics(self, W, H):
        c = self.endo['camera']
        f = (W / 2) / math.tan(math.radians(c['view_angle_deg']) / 2)
        return np.asarray([W, H, f, self.far], dtype=np.float64)

    def _labels(self):
        """Truth-label payload of the main frame; same quantities as SimWorld._dump_frame."""
        sim, lab = self.sim, {}

        def mat4(h):
            M = np.eye(4)
            M[:3] = matrix_to_np(sim.getObjectMatrix(h, -1))
            return M

        lab['label_tcp_M'] = mat4(self.by[f'{self.cutter}_tcp'])
        lab['label_yaw_M'] = (mat4(self.by[f'{self.cutter}_dvrk_yaw_frame'])
                              if f'{self.cutter}_dvrk_yaw_frame' in self.by
                              else np.full((4, 4), np.nan))
        lab['label_marker_pos'] = (np.array([sim.getObjectPosition(m, -1) for m in self.markers],
                                            dtype=float) if self.markers else np.zeros((0, 3)))
        lab['label_events'] = np.asarray(json.dumps(self.events))
        try:
            stats = self.call('getRCMStats')
            lab['label_grip'] = np.float64(stats[self.index[self.cutter] - 1].get('grip', np.nan))
            lab['label_camera_target'] = np.asarray(
                stats[self.index[self.view] - 1]['target'], dtype=float)
        except Exception:
            lab.setdefault('label_grip', np.float64(np.nan))
        if self.sheet_origin is not None:
            arms = sorted(self.grasped_points)
            lab.update(
                label_nodes=self.truth.nodes(),
                label_cut_length=np.float64(self.cut_length()),
                label_cut_front=np.asarray(self.cut_front()),
                label_tension=np.float64(self.cut_line_tension()),
                label_tissue_alive=np.array(self.call('tissueConstraints'),
                                            dtype=np.int64).reshape(-1, 3),
                label_grasped_arms=np.asarray(arms),
                label_grasped_pts=(np.array([self.grasped_points[a] for a in arms], dtype=float)
                                   if arms else np.zeros((0, 3))))
        return lab

    def _perception_tick(self):
        """Dataset frame writer (registered replacement for the shadow-estimator call site)."""
        out = Path(self._capture['out_dir'])
        tick = self.tick
        cam_R, cam_t = self.camera()
        with self.truth.purpose('dataset_label'):
            labels = self._labels()
        frame = dict(rgb=np.asarray(self.image),
                     depth=np.asarray(self.depth, dtype=np.float32),
                     cam_R=cam_R, cam_t=cam_t, intrinsics=self._intrinsics(self.W, self.H),
                     tick=np.int64(tick))
        frame.update(labels)
        atomic_npz(out / f'tick_{tick:03d}', frame)
        if self._twin is not None:
            self._hd_tick(out, tick, cam_R, cam_t)
        if self._aux_sensor is not None:
            self._aux_tick(out, tick, cam_R, cam_t)

    def _hd_tick(self, out, tick, cam_R, cam_t):
        W2, H2 = HD_RESOLUTION

        def render():
            self._pose_sensor(self.sim, self._twin, cam_R, cam_t)
            return self._snap(self._twin, W2, H2)
        rgb, depth = self._time('render_hd', render)
        atomic_npz(out / f'hd_{tick:03d}', dict(
            rgb=rgb, depth=depth, cam_R=cam_R, cam_t=cam_t,
            intrinsics=self._intrinsics(W2, H2)))

    def _aux_tick(self, out, tick, cam_R, cam_t):
        targets = aux_targets(self._capture['aux_seed'], tick,
                              self._capture['aux_views'], self._capture['envelope'])
        rcm = np.asarray(self.arm_cfg[self.view]['rcm_world'], dtype=float)
        view = np.asarray(self.target_of(self.view), dtype=float)   # live ECM yaw/pitch/ins
        for k, target in enumerate(targets):
            delta = target - view
            R2, t2 = aux_pose(cam_R, cam_t, rcm, delta)

            def render():
                self._pose_sensor(self.sim, self._aux_sensor, R2, t2)
                return self._snap(self._aux_sensor, self.W, self.H)
            rgb, depth = self._time('render_aux', render)
            atomic_npz(out / f'aux_{tick:03d}_{k}', dict(
                rgb=rgb, depth=depth, cam_R=R2, cam_t=t2,
                intrinsics=self._intrinsics(self.W, self.H), delta=delta))


def source_hashes(proto):
    return {
        'scripts/learn/collect.py': sha256(LEARN / 'collect.py'),
        'scripts/task/sim_world.py': sha256(TASK / 'sim_world.py'),
        'scripts/task/cut_fsm.py': sha256(TASK / 'cut_fsm.py'),
        'scripts/task/ground_truth.py': sha256(TASK / 'ground_truth.py'),
        'scripts/task/run_cut_task.py': sha256(TASK / 'run_cut_task.py'),
        'config/task/cut_task.json': sha256(BASE_CFG),
        'config/learn/learned_perception_v1.json': sha256(PROTOCOL),
        'scenes/davinci_si_articulated_verified.ttt': sha256(SCENE),
        'protocol_reference_scene_sha256': proto['references']['scene_sha256'],
        'protocol_reference_base_config_sha256': proto['references']['base_config_sha256'],
    }


def run_episode(port, stage, proto, base_cfg, split, index, tag, opts):
    """One unchanged-definition episode; refused/failed/error are recorded, never retried."""
    p = episode_params(split, index, proto)
    env = proto['data_generation']['episode_randomisation']['ecm_initial_view_deg_deg_m']
    ep_dir = Path(opts.data_dir) / f'ep{index:05d}'
    manifest_copy = Path(opts.manifest_dir) / f'ep{index:05d}.json'
    res = dict(index=index, split=split, seed=p['seed'], tag=tag,
               sampled={k: p[k] for k in ('advance_m', 'target_length_m', 'scenario',
                                          'yaw_deg', 'pitch_deg', 'insertion_m')},
               aux_seed=p['aux_seed'], aux_views=opts.aux_views, hd_twin=opts.hd_twin,
               status='error', final_state=None, reason=None, error=None, issues=None,
               cut_length_m=None, ticks=None, trace=[], events=None, files=[],
               timings={}, wall_s=None, truth_reads_by_purpose={}, sensors={},
               initial_view_readback=None, source_sha256=opts.source_sha256,
               git_head=opts.git_head)
    if ep_dir.exists() or manifest_copy.exists():
        res['error'] = f'output path already exists, refusing to reuse: {ep_dir}'
        if not manifest_copy.exists():
            manifest_copy.parent.mkdir(parents=True, exist_ok=True)
            write_json(manifest_copy, _jsonable(res))
        return res
    ep_dir.mkdir(parents=True)
    manifest_copy.parent.mkdir(parents=True, exist_ok=True)

    cfg = deepcopy(base_cfg)
    cfg['cut'].update(advance_m=p['advance_m'], length_m=p['target_length_m'],
                      control_source='truth')
    cfg['perception'].update(mode='shadow', markers=True, material_pattern=True,
                             local_material_chart=True, front_recovery=True,
                             predictive_view=True, exclude_tcp_marker=True,
                             recovery_view_after_cut=True, ray_marker_fit=True)
    cfg['perception'].pop('dump_dir', None)      # this collector writes its own frames
    resolve_perception_flags(cfg)
    events = scenario_events(cfg, p['scenario'])
    res['events'] = events
    res['perception_cfg'] = dict(cfg['perception'])
    args = SimpleNamespace(
        host='localhost', port=port, timeout=180.0, stage_dir=stage,
        scene=SCENE, scenario=p['scenario'], mode='full',
        initial_view=[p['yaw_deg'], p['pitch_deg']],
        initial_view_insertion_m=p['insertion_m'],
        run_id=uuid.uuid4().hex,
        report_name=f'learned_perception_v1/{tag}/ep{index:05d}',
        occlusion_probe=None)
    res['run_id'] = args.run_id
    trace = res['trace']
    client = sim = world = fsm = None
    t0 = time.perf_counter()
    try:
        client, sim = connect(args)
        world = LearnWorld(client, sim, args, cfg, mode='full', events=events,
                           capture=dict(out_dir=ep_dir, aux_views=opts.aux_views,
                                        hd_twin=opts.hd_twin, aux_seed=p['aux_seed'],
                                        envelope=env))
        fsm = CutTaskFSM(world, cfg)
        observe = world.observe

        def traced_observe(**kwargs):
            obs = observe(**kwargs)
            trace.append(dict(tick=world.tick, tension=obs.tension,
                              cut_length_m=obs.cut_length_m,
                              nodes_sha256=hashlib.sha256(
                                  np.asarray(world.call('tissueNodes'), dtype=float)
                                  .tobytes()).hexdigest()[:16]))
            return obs
        world.observe = traced_observe
        fsm.run()
        final = world.observe(inject_events=False)
        issues = completion_issues(final, cfg)
        rep = fsm.report()
        res.update(final_state=rep['final_state'], reason=rep['reason'],
                   ticks=world.tick, cut_length_m=final.cut_length_m, issues=issues)
        if any(e.get('event') == 'perception_error' for e in world.event_log):
            res['status'], res['error'] = 'error', 'frame capture failed (perception_error)'
        elif fsm.state == S.DONE and not issues:
            res['status'] = 'done'
        else:
            res['status'] = 'failed'
    except Exception as exc:
        res['error'] = f'{type(exc).__name__}: {exc}'
        res['status'] = 'refused' if 'initial view' in str(exc) else 'error'
        if fsm is not None:
            res['final_state'] = fsm.state.name
    finally:
        wall = time.perf_counter() - t0
        if sim is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                res['cleanup_error'] = f'{type(exc).__name__}: {exc}'
        if client is not None:
            try:
                client.socket.close(linger=0)
                client.context.term()
            except Exception:
                pass
    if world is not None:
        res['truth_reads_by_purpose'] = world.truth.reads_by_purpose
        res['timings'] = world.timing_summary()
        res['sensors'] = dict(world._sensors)
        res['initial_view_readback'] = world.initial_view_readback
        mean = lambda k: (res['timings'].get(k) or {}).get('mean_s')
        res['mean_render_ms'] = {'hd': None if mean('render_hd') is None else round(mean('render_hd') * 1e3, 2),
                                 'aux': None if mean('render_aux') is None else round(mean('render_aux') * 1e3, 2)}
    res['wall_s'] = round(wall, 3)
    files = []
    for f in sorted(ep_dir.iterdir()):
        if f.name == 'manifest.json' or f.name.endswith('.tmp.npz'):
            continue
        files.append(dict(name=f.name, sha256=sha256(f), bytes=f.stat().st_size))
    res['files'] = files
    kinds = {'main': ('tick_',), 'hd': ('hd_',), 'aux': ('aux_',)}
    res['bytes_by_kind'] = {k: sum(f['bytes'] for f in files if f['name'].startswith(p_))
                            for k, p_ in kinds.items()}
    res['frames_by_kind'] = {k: sum(1 for f in files if f['name'].startswith(p_))
                             for k, p_ in kinds.items()}
    write_json(ep_dir / 'manifest.json', _jsonable(res))
    write_json(manifest_copy, _jsonable(res))
    return res


def worker(k, opts, proto, base_cfg, indices, results):
    port = opts.base_port + k
    stage = Path(tempfile.gettempdir()) / f'{opts.tag}_w{k}'
    stage.mkdir(parents=True, exist_ok=True)
    logs = Path(opts.manifest_dir) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    rec = dict(worker=k, port=port, indices=indices, t_start=time.monotonic(),
               sim_startup_s=None, episodes=[])
    proc = log = None
    try:
        proc, log, rec['sim_startup_s'] = launch(port, f'_{opts.tag}_w{k}',
                                               logs / f'sim_w{k}.log')
        for index in indices:
            ep = run_episode(port, stage, proto, base_cfg, opts.split, index,
                             opts.tag, opts)
            rec['episodes'].append(ep)
            print(f"w{k} ep{index:05d} {ep['status']} {ep['final_state']} "
                  f"{ep['wall_s']}s {ep['error'] or ''}", flush=True)
    except Exception as exc:
        rec['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        rec['t_end'] = time.monotonic()
        shutdown(proc, log)
    results.append(rec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', default='dev', choices=['dev', 'holdout'])
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--base-port', type=int, default=23950)
    p.add_argument('--data-root', default='F:/davinci_m5_learn')
    p.add_argument('--tag', default='dev')
    p.add_argument('--aux-views', type=int, default=4)
    p.add_argument('--hd-twin', dest='hd_twin', action='store_true', default=True)
    p.add_argument('--no-hd-twin', dest='hd_twin', action='store_false')
    p.add_argument('--no-aux', action='store_true',
                   help='auxiliary renders off entirely: aux_views=0 and no HD twin')
    args = p.parse_args()
    if args.split == 'holdout':
        raise SystemExit(
            "split 'holdout' is closed: holdout_H1 episodes require a sealed model-freeze "
            "record (learned_perception_v1.splits.holdout_H1) that does not exist yet")
    if args.no_aux:
        args.aux_views, args.hd_twin = 0, False
    proto = protocol()
    refs = proto['references']
    for path, pinned, name in ((SCENE, refs['scene_sha256'], 'scene'),
                               (BASE_CFG, refs['base_config_sha256'], 'base config')):
        if sha256(path) != pinned:
            raise SystemExit(f'registered {name} hash mismatch: {path} (protocol is binding)')
    if args.count < 1 or args.workers < 1 or args.start < 0:
        raise SystemExit('--count/--workers must be >= 1 and --start >= 0')

    data_dir = Path(args.data_root) / 'learned_perception_v1' / args.tag
    manifest_dir = REPO / 'validation' / 'task' / 'learned_perception_v1' / args.tag
    summary_path = manifest_dir.parent / f'{args.tag}.json'
    for existing in (data_dir, manifest_dir, summary_path):
        if existing.exists():
            raise SystemExit(f'refusing to reuse {args.tag}: {existing} exists')
    manifest_dir.mkdir(parents=True)
    opts = SimpleNamespace(split=args.split, tag=args.tag, base_port=args.base_port,
                           aux_views=args.aux_views, hd_twin=args.hd_twin,
                           data_dir=data_dir, manifest_dir=manifest_dir,
                           source_sha256=source_hashes(proto),
                           git_head=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                                   capture_output=True, text=True).stdout.strip())
    indices = list(range(args.start, args.start + args.count))
    summary = dict(
        purpose='Phase-A collection run for the registered learned-perception study '
                '(config/learn/learned_perception_v1.json). Dataset generation only.',
        protocol_sha256=sha256(PROTOCOL), tag=args.tag, split=args.split,
        indices=indices, aux_views=args.aux_views, hd_twin=args.hd_twin,
        base_port=args.base_port, data_root=args.data_root,
        aux_view_model=('approximation for training only: absolute (yaw,pitch,insertion) '
                        'uniform in the registered ECM envelope; delta applied to the live '
                        'camera frame about the ECM RCM point; never control input'),
        source_sha256=opts.source_sha256, git_head=opts.git_head, workers=[])
    write_json(summary_path, _jsonable(summary))

    results, threads = [], []
    for k in range(min(args.workers, len(indices))):
        mine = indices[k::args.workers]
        t = threading.Thread(target=worker, args=(k, opts, proto, base_cfg_load(), mine, results))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    episodes = [e for rec in results for e in rec['episodes']]
    wall = (max(r['t_end'] for r in results) - min(r['t_start'] for r in results)) if results else 0
    counts = {}
    for e in episodes:
        counts[e['status']] = counts.get(e['status'], 0) + 1
    frames = sum(sum(e['frames_by_kind'].values()) for e in episodes)
    nbytes = sum(f['bytes'] for e in episodes for f in e['files'])
    for rec in results:
        rec.pop('t_start'), rec.pop('t_end')
    summary.update(counts_by_status=counts, episodes=len(episodes), frames=frames,
                   bytes=nbytes, wall_s=round(wall, 2),
                   episodes_per_hour=round(len(episodes) / wall * 3600, 1) if wall else None,
                   frames_per_hour=round(frames / wall * 3600) if wall else None,
                   workers=sorted(results, key=lambda r: r['worker']))
    write_json(summary_path, _jsonable(summary))
    print(json.dumps(dict(counts=counts, frames=frames, bytes=nbytes, wall_s=round(wall, 2))),
          flush=True)
    print('report:', summary_path.relative_to(REPO))
    return 0


def base_cfg_load():
    return json.loads(BASE_CFG.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main())
