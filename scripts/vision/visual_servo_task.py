"""Endoscope-guided pick-and-place with PSM2: position-based visual servoing on the simulated depth buffer.

Perception uses only the endoscope RGB image, its depth buffer and the camera intrinsics:
colour masks find the block, the target ring and a marker on the instrument wrist; their 3-D
positions come from back-projecting the depth of the masked pixels. The camera pose comes from
the ECM kinematics and can be perturbed (--calib-rot-deg/--calib-trans-m) to test robustness.
Ground truth from the simulator is used only for logging and the final success check.

Control: the error between goal and controlled point is measured in the camera frame, rotated
into the (estimated) world frame and integrated into a commanded grasp-point position, which is
converted to PSM2 yaw/pitch/insertion through the fixed-RCM geometry and sent to the scene's own
guarded RCM controller (IK + mesh-collision rollback).

The scene is loaded from a staged copy and never saved.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'build'))
import ecm_pairs  # noqa: E402
from simtools import (ENDO_CONFIG, ENDO_VALIDATION, ENDOSCOPE_SCENE, add_arguments, aliases, connect,  # noqa: E402
                      load_scene, matrix_to_np, read_json, relative, sha256, stop_simulation, write_json)


def unit(v):
    return v / np.linalg.norm(v)


def rotvec(v):
    th = np.linalg.norm(v)
    if th < 1e-15:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


class Endoscope:
    """Image/depth acquisition, colour segmentation and back-projection in the camera frame."""

    def __init__(self, sim, handle, cfg, rng, depth_noise):
        self.sim, self.h, self.rng, self.depth_noise = sim, handle, rng, depth_noise
        self.W, self.H = cfg['resolution']
        self.f = (self.W / 2) / math.tan(math.radians(cfg['view_angle_deg']) / 2)
        self.far = cfg['far_m']

    def grab(self):
        s = self.sim
        s.handleVisionSensor(self.h)
        raw, _ = s.getVisionSensorImg(self.h)
        d, _ = s.getVisionSensorDepth(self.h, 1)
        self.rgb = np.frombuffer(raw, np.uint8).reshape(self.H, self.W, 3)[::-1].astype(float)
        depth = np.array(s.unpackFloatTable(d), dtype=float).reshape(self.H, self.W)[::-1]
        if self.depth_noise > 0:
            depth = depth + self.rng.normal(0, self.depth_noise, depth.shape)
        self.depth = depth

    def mask(self, rgb01, tol=60.0):
        return np.linalg.norm(self.rgb - 255.0 * np.asarray(rgb01), axis=2) < tol

    def locate(self, m, sphere_radius=None, min_px=12):
        """3-D centre (camera frame) of the surface seen in mask m.

        With `sphere_radius` the centre is a least-squares fit of a sphere of that radius to the
        back-projected surface points (valid for partial views); otherwise the point mean is used.
        """
        v, u = np.nonzero(m)
        if len(u) < min_px:
            return None
        z = self.depth[v, u]
        keep = (z < self.far * 0.999) & (np.abs(z - np.median(z)) < 0.01)
        if keep.sum() < min_px:
            return None
        u, v, z = u[keep], v[keep], z[keep]
        pts = np.stack([(self.W / 2 - u) * z / self.f, (self.H / 2 - v) * z / self.f, z], axis=1)
        c = pts.mean(axis=0)
        if sphere_radius:
            c = c + sphere_radius * unit(c)
            for _ in range(15):                        # Gauss-Newton on |p_i - c| - r
                d = pts - c
                n = np.linalg.norm(d, axis=1)
                J = -d / n[:, None]
                dc = np.linalg.lstsq(J, -(n - sphere_radius), rcond=None)[0]
                c = c + dc
                if np.linalg.norm(dc) < 1e-7:
                    break
        return dict(p=c, px=int(len(u)), uv=[float(u.mean()), float(v.mean())])

    def locate_ring(self, m, radius, tube, min_px=30):
        """Centre of a thin ring of known radius: plane fit (SVD) + least-squares circle fit in that plane.

        A plain point mean is biased by perspective (nearer ring parts cover more pixels) and by partial
        occlusion; the circle fit is not. The visible tube surface lies ~`tube` in front of the centre line."""
        v, u = np.nonzero(m)
        if len(u) < min_px:
            return None
        z = self.depth[v, u]
        keep = (z < self.far * 0.999) & (np.abs(z - np.median(z)) < 0.01)
        u, v, z = u[keep], v[keep], z[keep]
        if len(u) < min_px:
            return None
        pts = np.stack([(self.W / 2 - u) * z / self.f, (self.H / 2 - v) * z / self.f, z], axis=1)
        c0 = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - c0, full_matrices=False)
        e1, e2 = Vt[0], Vt[1]
        q = np.stack([(pts - c0) @ e1, (pts - c0) @ e2], axis=1)
        c2 = np.zeros(2)
        for _ in range(20):                            # Gauss-Newton on |q_i - c| - R
            d = q - c2
            n = np.linalg.norm(d, axis=1)
            J = -d / n[:, None]
            dc = np.linalg.lstsq(J, -(n - radius), rcond=None)[0]
            c2 = c2 + dc
            if np.linalg.norm(dc) < 1e-8:
                break
        c = c0 + c2[0] * e1 + c2[1] * e2
        return dict(p=c + tube * unit(c), px=int(len(u)), uv=[float(u.mean()), float(v.mean())])

    def project(self, pc):
        return [self.W / 2 - self.f * pc[0] / pc[2], self.H / 2 - self.f * pc[1] / pc[2]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--depth-noise-m', type=float, default=0.0, help='Gaussian depth noise per pixel [m]')
    p.add_argument('--calib-rot-deg', type=float, default=0.0, help='camera rotation calibration error [deg]')
    p.add_argument('--calib-trans-m', type=float, default=0.0, help='camera position calibration error [m]')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--gain', type=float, default=0.6)
    p.add_argument('--max-step', type=float, default=0.002, help='max commanded grasp-point step per cycle [m]')
    p.add_argument('--tol', type=float, default=0.0008, help='visual convergence tolerance [m]')
    p.add_argument('--max-iter', type=int, default=80, help='max servo cycles per phase')
    p.add_argument('--success-tol', type=float, default=0.002, help='final block-to-target ground-truth tolerance [m]')
    p.add_argument('--name', default='nominal', help='result name: validation/endoscope/visual_task_<name>.json')
    p.add_argument('--frames', action='store_true', help='save annotated key frames (PNG) and an MP4 if imageio-ffmpeg is installed')
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    endo = read_json(ENDO_CONFIG / 'endoscope.json')
    task = read_json(ENDO_CONFIG / 'task.json')
    rcm = read_json(ENDO_CONFIG / 'rcm_config.json')
    pairs = read_json(ENDO_CONFIG / 'collision_pairs.json')
    arm = task['arm']
    index = next(i for i, a in enumerate(rcm, 1) if a['arm'] == arm)
    cfg = rcm[index - 1]
    r_w = np.array(cfg['rcm_world'])
    a0, tx, ty = (np.array(cfg[k]) for k in ('axis_home', 'transverse_x', 'transverse_y'))
    go, app = task['grasp_offset_m'], task['approach_offset_m']
    marker_to_grasp = task['tip_marker']['offset_from_tcp_m'] - go
    # Axial distance from the RCM point to the grasp point at the home insertion.
    tcp_len = 0.4498
    base = tcp_len - go - cfg['rcm_distance_home']

    def command_for(Pw):
        d = unit(Pw - r_w)
        yaw = math.degrees(math.atan2(d @ tx, d @ a0))
        pitch = math.degrees(math.atan2(d @ ty, d @ a0))
        ins = cfg['insertion_home'] + np.linalg.norm(Pw - r_w) - base
        return yaw, pitch, ins

    def point_for(yaw, pitch, ins):
        d = unit(a0 + math.tan(math.radians(yaw)) * tx + math.tan(math.radians(pitch)) * ty)
        return r_w + d * (base + ins - cfg['insertion_home'])

    client, sim = connect(args)
    load_scene(client, sim, args, ENDOSCOPE_SCENE, f'visual_task_{args.name}.ttt')
    by = aliases(sim)
    ctrl = by['RCMController']
    cam = Endoscope(sim, by[endo['camera']['alias']], endo['camera'], rng, args.depth_noise_m)
    block_h = by[task['block']['alias']]
    target_true = np.array(task['target']['center_world'])
    tcp_h, j8_h = by[arm + '_tcp'], by[arm + '_joint8']

    # Camera pose: ECM kinematics (true) with an optional calibration error.
    M = matrix_to_np(sim.getObjectMatrix(cam.h, -1))
    R_true, t_true = M[:, :3], M[:, 3]
    R_err = rotvec(unit(rng.normal(size=3)) * math.radians(args.calib_rot_deg))
    R_est = R_err @ R_true
    t_est = t_true + unit(rng.normal(size=3)) * args.calib_trans_m
    r_cam = R_est.T @ (r_w - t_est)           # RCM point as the (mis)calibrated camera sees it

    def true_cam(pw):
        return R_true.T @ (np.asarray(pw) - t_true)

    labels = pairs['labels']
    hpairs = [[by[a], by[b]] for a, b in labels]
    allowed = {a['tool_shape']: (np.array(a['rcm_world']), a['port_radius_m']) for a in rcm}
    probe = ecm_pairs.create_probe(sim)

    def unexpected_contacts():
        bad = []
        for i, pts in ecm_pairs.scan(sim, probe, hpairs):
            a, b = labels[i]
            if b == 'Patient_Skin' and a in allowed and len(pts) and \
                    np.linalg.norm(pts - allowed[a][0], axis=1).max() <= allowed[a][1]:
                continue
            bad.append(labels[i])
        return bad

    sim.setStepping(True)
    sim.setBoolParam(sim.boolparam_realtime_simulation, False)
    sim.startSimulation()
    sim.step()
    sim.callScriptFunction('homeRCM', ctrl)

    frames, log, phases = [], [], []
    state = dict(holding=False, contacts=[], rejections=0, max_rcm_error=0.0)

    def truth():
        tcp = np.array(sim.getObjectPosition(tcp_h, -1))
        a = matrix_to_np(sim.getObjectMatrix(j8_h, -1))[:, 2]
        # The ring is a mesh shape whose frame sits at the world origin, so its true centre comes from task.json.
        return dict(grasp=tcp - go * a, block=np.array(sim.getObjectPosition(block_h, -1)),
                    target=target_true, rcm_err=float(np.linalg.norm(np.cross(r_w - tcp, a))))

    def perceive():
        cam.grab()
        blk = cam.locate(cam.mask(task['block']['color']), sphere_radius=task['block']['size_m'] / 2)
        tgt = cam.locate_ring(cam.mask(task['target']['color']), task['target']['radius_m'], task['target']['tube_m'])
        mk = cam.locate(cam.mask(task['tip_marker']['color']), sphere_radius=task['tip_marker']['radius_m'])
        grasp = None
        if mk:
            axis = unit(mk['p'] - r_cam)
            grasp = dict(p=mk['p'] + marker_to_grasp * axis, axis=axis)
        return dict(block=blk, target=tgt, marker=mk, grasp=grasp)

    def snapshot(label, obs):
        if not args.frames:
            return
        from PIL import Image, ImageDraw
        im = Image.fromarray(cam.rgb.astype(np.uint8))
        dr = ImageDraw.Draw(im)
        for key, col in [('block', (0, 255, 0)), ('target', (60, 120, 255)), ('grasp', (255, 0, 255))]:
            o = obs.get(key)
            if o:
                u, v = cam.project(o['p'])
                dr.line([u - 8, v, u + 8, v], fill=col, width=2)
                dr.line([u, v - 8, u, v + 8], fill=col, width=2)
        dr.text((8, 8), f'{args.name} | {label}', fill=(255, 255, 255))
        frames.append((label, im))

    def servo(name, goal_fn, ctrl_key):
        """Drive the observed controlled point (grasp point or held block) to goal_fn(obs)."""
        yaw, pitch, ins = sim.callScriptFunction('getRCMStats', ctrl)[index - 1]['target']
        T_cmd = point_for(yaw, pitch, ins)
        hits, lost, rejected, converged = 0, 0, 0, False
        for it in range(args.max_iter):
            sim.step()
            obs = perceive()
            c, g = obs.get(ctrl_key), goal_fn(obs)
            if c is None or g is None:
                lost += 1
                if lost > 5:
                    break
                continue
            e_cam = g - c['p']
            gt = truth()
            true_ctrl = gt['block'] if ctrl_key == 'block' else gt['grasp']
            state['max_rcm_error'] = max(state['max_rcm_error'], gt['rcm_err'])
            entry = dict(phase=name, it=it, t=sim.getSimulationTime(), visual_error_m=float(np.linalg.norm(e_cam)),
                         ctrl_estimation_error_m=float(np.linalg.norm(c['p'] - true_cam(true_ctrl))))
            if obs['block']:
                entry['block_estimation_error_m'] = float(np.linalg.norm(obs['block']['p'] - true_cam(gt['block'])))
            if obs['target']:
                entry['target_estimation_error_m'] = float(np.linalg.norm(obs['target']['p'] - true_cam(gt['target'])))
            bad = unexpected_contacts()
            if bad:
                state['contacts'].append(dict(phase=name, it=it, contacts=bad))
            if np.linalg.norm(e_cam) < args.tol:
                hits += 1
                log.append(entry)
                if hits >= 2:
                    converged = True
                    break
                continue
            hits = 0
            step = args.gain * (R_est @ e_cam)
            n = np.linalg.norm(step)
            if n > args.max_step:
                step *= args.max_step / n
            cmd = command_for(T_cmd + step)
            ok = sim.callScriptFunction('setRCMTarget', ctrl, index, *map(float, cmd))
            entry.update(command=list(map(float, cmd)), accepted=bool(ok))
            log.append(entry)
            if ok:
                T_cmd = T_cmd + step
                rejected = 0
            else:
                entry['rejected_by'] = sim.callScriptFunction('getRCMStats', ctrl)[index - 1]['last_collision'] or 'limits/IK'
                state['rejections'] += 1
                rejected += 1
                if rejected >= 3:
                    break
            if args.frames and it % 2 == 0:
                snapshot(name, obs)
        gt = truth()
        true_ctrl = gt['block'] if ctrl_key == 'block' else gt['grasp']
        obs = perceive()
        g = goal_fn(obs)
        res = dict(phase=name, converged=converged, iterations=it + 1, lost_frames=lost,
                   final_visual_error_m=float(np.linalg.norm(g - obs[ctrl_key]['p'])) if (g is not None and obs.get(ctrl_key)) else None,
                   final_true_error_m=float(np.linalg.norm(R_true @ g + t_true - true_ctrl)) if g is not None else None)
        phases.append(res)
        snapshot(name + ' done', obs)
        print(res, flush=True)
        return converged

    t0 = time.time()
    ok = True
    # Bring the instrument into view (open loop, as a surgeon would): retract along the docked axis, then
    # swing to the planned approach pose; open the jaws. Everything after this is vision-driven.
    for ready in task['planning']['command_sequence_checked'][:2]:
        ok &= bool(sim.callScriptFunction('setRCMTarget', ctrl, index, *map(float, ready)))
    ok &= bool(sim.callScriptFunction('setGrip', ctrl, index, task['grip_open_rad']))
    sim.step()
    first = perceive()
    snapshot('start', first)
    if not (first['block'] and first['target'] and first['grasp']):
        ok = False
        print('initial detection failed', {k: v is not None for k, v in first.items()}, flush=True)

    # Bead and target are static and the endoscope holds still, so their positions are measured once, on the
    # first frame where nothing covers them; the jaws and the carried bead would otherwise occlude them.
    bead0 = first['block']['p'] if first.get('block') else None
    target0 = first['target']['p'] if first.get('target') else None
    measured = dict(bead_initial_estimation_error_m=float(np.linalg.norm(bead0 - true_cam(truth()['block']))) if bead0 is not None else None,
                    target_estimation_error_m=float(np.linalg.norm(target0 - true_cam(target_true))) if target0 is not None else None)

    def along_axis(point, offset):
        def goal(obs):
            if obs.get('grasp') is None:
                return None
            return point - offset * obs['grasp']['axis']
        return goal

    grasp_result = None
    if ok:
        ok = servo('pre-grasp', along_axis(bead0, app), 'grasp') and servo('grasp', along_axis(bead0, 0.0), 'grasp')
    if ok:
        gt = truth()
        grasp_result = dict(true_grasp_point_to_block_m=float(np.linalg.norm(gt['grasp'] - gt['block'])))
        ok = bool(sim.callScriptFunction('setGrip', ctrl, index, task['grip_hold_rad'])) and \
            bool(sim.callScriptFunction('grasp', ctrl, index, task['block']['alias']))
        grasp_result['grasped'] = ok
    if ok:
        lift = perceive()
        lift_goal = lift['block']['p'] - app * lift['grasp']['axis']
        ok = servo('lift', lambda obs: lift_goal, 'block') and servo('pre-place', along_axis(target0, app), 'block') \
            and servo('place', along_axis(target0, 0.0), 'block')
    released = False
    if ok:
        released = bool(sim.callScriptFunction('release', ctrl, index)) and \
            bool(sim.callScriptFunction('setGrip', ctrl, index, task['grip_open_rad']))
        after = perceive()
        if after['grasp']:
            back = after['grasp']['p'] - app * after['grasp']['axis']
            servo('retract', lambda obs: back, 'grasp')
    gt = truth()
    final_error = float(np.linalg.norm(gt['block'] - gt['target']))
    sim.callScriptFunction('setGrip', ctrl, index, 0.0)
    stats = sim.callScriptFunction('getRCMStats', ctrl)
    sim.removeObjects([probe])
    stop_simulation(client, sim)
    sim.setStepping(False)
    sim.setBoolParam(sim.boolparam_realtime_simulation, True)

    est = [e['ctrl_estimation_error_m'] for e in log]
    blk = [e['block_estimation_error_m'] for e in log if 'block_estimation_error_m' in e]
    tgt = [e['target_estimation_error_m'] for e in log if 'target_estimation_error_m' in e]
    success = bool(ok and released and final_error <= args.success_tol and not state['contacts'])
    result = {
        'scene': relative(ENDOSCOPE_SCENE), 'scene_sha256': sha256(ENDOSCOPE_SCENE),
        'settings': {k: getattr(args, k) for k in ['depth_noise_m', 'calib_rot_deg', 'calib_trans_m', 'seed', 'gain',
                                                    'max_step', 'tol', 'max_iter', 'success_tol']},
        'success': success,
        'final_block_to_target_m': final_error,
        'grasp': grasp_result, 'released': released, 'static_object_measurement': measured,
        'phases': phases,
        'servo_cycles': len(log),
        'commands_rejected': state['rejections'],
        'unexpected_contacts': state['contacts'],
        'max_psm_rcm_error_m': state['max_rcm_error'],
        'controller_stats': stats[index - 1],
        'estimation_error_m': {
            'controlled_point_median': float(np.median(est)) if est else None, 'controlled_point_max': float(np.max(est)) if est else None,
            'block_median': float(np.median(blk)) if blk else None, 'target_median': float(np.median(tgt)) if tgt else None},
        'wall_time_s': round(time.time() - t0, 1),
        'log': log,
        'notes': 'Kinematic grasp (object parented to the jaw tip frame); no gravity, contact forces or tissue response. '
                 'The instrument is first brought into view with one open-loop command; every later motion is vision-driven.',
    }
    out = ENDO_VALIDATION / f'visual_task_{args.name}.json'
    write_json(out, result)
    if args.frames and frames:
        fdir = ENDO_VALIDATION / f'visual_task_{args.name}_frames'
        fdir.mkdir(parents=True, exist_ok=True)
        keys = [(lbl, im) for lbl, im in frames if lbl in ('start',) or lbl.endswith(' done')]
        for k, (lbl, im) in enumerate(keys):
            im.save(fdir / f'{k:02d}_{lbl.replace(" ", "_")}.png')
        try:
            import imageio_ffmpeg
            w = imageio_ffmpeg.write_frames(str(ENDO_VALIDATION / f'visual_task_{args.name}.mp4'), (cam.W, cam.H), fps=10,
                                            codec='libx264', quality=7)
            w.send(None)
            for _, im in frames:
                w.send(im.tobytes())
            w.close()
        except ImportError:
            pass
    print('success', success, 'final block-target error mm', round(final_error * 1000, 3), 'wrote', relative(out), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
