"""Choose pick/place positions for the visual-servo task in the visible cavity near the PSM2 tip.

Uses the built endoscope scene (staged copy, never saved) and its own RCM controller:
  * renders the endoscope depth buffer with the task arm hidden (anatomy and other arms only);
  * sweeps PSM2 RCM targets on a grid; a pose counts only if the controller accepts it (IK + collision
    guard), the jaws can open there, the grasp point is unoccluded in the endoscope image and has
    clearance from every rendered surface;
  * picks a block position A and a target position B whose full pick-and-place command sequence is
    accepted by the controller, and writes config/endoscope/task.json.
"""
import argparse
import itertools
import math
from pathlib import Path

import numpy as np

import ecm_pairs
from simtools import (ENDO_CONFIG, ENDOSCOPE_SCENE, add_arguments, aliases, connect, load_scene, matrix_to_np,
                      np_to_matrix, read_json, relative, sha256, stop_simulation, write_json)

TASK_DEFAULTS = {
    'arm': 'psm2',
    'grasp_offset_m': 0.006,       # grasp point between the jaws, behind the tcp (jaw tip)
    'approach_offset_m': 0.008,    # retract distance along the shaft before/after grasp and release
    'grip_open_rad': 0.5,
    'grip_hold_rad': 0.2,
    # A 5 mm bead: a sphere of known radius lets vision fit its centre from any partial view.
    'block': {'alias': 'TaskBead', 'shape': 'sphere', 'size_m': 0.005, 'color': [0.05, 0.95, 0.1]},
    'target': {'alias': 'TaskTarget', 'radius_m': 0.005, 'tube_m': 0.0007, 'color': [0.1, 0.35, 1.0]},
    # Sphere centred on the wrist pivot; the wrist/shaft radius there is 4.2 mm, so the sphere must be larger to show.
    'tip_marker': {'alias': 'psm2_TipMarker', 'radius_m': 0.006, 'offset_from_tcp_m': 0.0197, 'color': [1.0, 0.1, 0.9]},
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--grid-deg', type=float, default=1.25)
    p.add_argument('--span-deg', type=float, default=5.0)
    p.add_argument('--clearance', type=float, default=0.007, help='min distance of the grasp point from rendered surfaces [m]')
    p.add_argument('--separation', type=float, nargs=2, default=[0.012, 0.022], help='pick-place distance range [m]')
    p.add_argument('--dump', type=Path, help='write per-pose diagnostics to this JSON file')
    p.add_argument('--limit-margin-deg', type=float, default=1.25, help='keep planned yaw/pitch this far inside the limits')
    p.add_argument('--insertions', type=float, nargs='+', default=[0.010, 0.014, 0.018, 0.022, 0.026, 0.030],
                   help='grasp insertions to try [m]; the approach pose retracts by approach_offset_m')
    args = p.parse_args()

    endo = read_json(ENDO_CONFIG / 'endoscope.json')
    rcm = read_json(ENDO_CONFIG / 'rcm_config.json')
    task = dict(TASK_DEFAULTS)
    arm = task['arm']
    index = next(i for i, a in enumerate(rcm, 1) if a['arm'] == arm)
    cfg = rcm[index - 1]
    L = cfg['limits']
    cam_cfg = endo['camera']

    client, sim = connect(args)
    load_scene(client, sim, args, ENDOSCOPE_SCENE, 'task_plan_stage.ttt')
    by = aliases(sim)
    ctrl, cam = by['RCMController'], by[cam_cfg['alias']]
    W, H = cam_cfg['resolution']
    f = (W / 2) / math.tan(math.radians(cam_cfg['view_angle_deg']) / 2)
    sim.setStepping(True)
    sim.startSimulation()
    sim.step()
    sim.callScriptFunction('homeRCM', ctrl)
    M = matrix_to_np(sim.getObjectMatrix(cam, -1))

    # Anatomy-and-other-arms depth: hide the task arm and any existing task objects.
    hidden = []
    for n, h in by.items():
        if sim.getObjectType(h) == sim.sceneobject_shape and (n.startswith(arm + '_') or n.startswith('Task')):
            hidden.append((h, sim.getObjectInt32Param(h, sim.objintparam_visibility_layer)))
            sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
    sim.handleVisionSensor(cam)
    d, _ = sim.getVisionSensorDepth(cam, 1)
    depth = np.array(sim.unpackFloatTable(d), dtype=float).reshape(H, W)[::-1]   # top-down rows
    for h, layer in hidden:
        sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, layer)
    vv, uu = np.mgrid[0:H, 0:W]
    cloud = np.stack([(W / 2 - uu) * depth / f, (H / 2 - vv) * depth / f, depth], axis=-1)   # camera frame
    far = depth >= cam_cfg['far_m'] * 0.999

    def to_cam(pw):
        return M[:, :3].T @ (np.asarray(pw) - M[:, 3])

    def visibility(pw, radius):
        pc = to_cam(pw)
        if pc[2] <= 0:
            return None
        u, v = W / 2 - f * pc[0] / pc[2], H / 2 - f * pc[1] / pc[2]
        if not (20 <= u < W - 20 and 20 <= v < H - 20):
            return None
        rp = int(math.ceil((radius + 0.003) * f / pc[2]))
        ui, vi = int(round(u)), int(round(v))
        patch = depth[vi - rp:vi + rp + 1, ui - rp:ui + rp + 1]
        front_margin = float(patch.min() - pc[2])          # >0: nothing in front of the object's centre plane
        w = int(math.ceil(0.02 * f / pc[2]))
        win = cloud[max(0, vi - w):vi + w + 1, max(0, ui - w):ui + w + 1][~far[max(0, vi - w):vi + w + 1, max(0, ui - w):ui + w + 1]]
        clear = float(np.min(np.linalg.norm(win - pc, axis=1))) if len(win) else 1.0
        return dict(u=u, v=v, z=float(pc[2]), front_margin=front_margin, clearance=clear)

    def pose_after(yaw, pitch, ins):
        ok = sim.callScriptFunction('setRCMTarget', ctrl, index, yaw, pitch, ins)
        if not ok:
            return None
        tcp = np.array(sim.getObjectPosition(by[arm + '_tcp'], -1))
        J = matrix_to_np(sim.getObjectMatrix(by[arm + '_joint8'], -1))
        return dict(grasp=tcp - task['grasp_offset_m'] * J[:, 2], frame=J[:, :3])

    span, step = args.span_deg, args.grid_deg
    angles = np.arange(-span, span + 1e-9, step)
    feasible, samples = [], []
    counts = dict(tried=0, rejected_by_controller=0, jaws_blocked=0, out_of_frame=0, occluded=0, low_clearance=0)
    for ins_g in args.insertions:
        ins_a = ins_g - task['approach_offset_m']
        if ins_a < L['insertion_min'] or ins_g > L['insertion_max']:
            continue
        for yaw, pitch in itertools.product(angles, angles):
            # Keep a margin to the manual limits: vision-driven corrections must have room around the plan.
            if abs(yaw) > L['yaw_deg'] - args.limit_margin_deg or abs(pitch) > L['pitch_deg'] - args.limit_margin_deg:
                continue
            counts['tried'] += 1
            app = pose_after(yaw, pitch, ins_a)
            g = pose_after(yaw, pitch, ins_g) if app else None
            if not g:
                counts['rejected_by_controller'] += 1
                sim.callScriptFunction('homeRCM', ctrl)
                continue
            jaws = sim.callScriptFunction('setGrip', ctrl, index, task['grip_open_rad'])
            sim.callScriptFunction('setGrip', ctrl, index, 0.0)
            vis = visibility(g['grasp'], math.sqrt(3) * task['block']['size_m'] / 2)
            counts['jaws_blocked'] += not jaws
            counts['out_of_frame'] += vis is None
            if vis:
                counts['occluded'] += vis['front_margin'] <= 0.004
                counts['low_clearance'] += vis['clearance'] < args.clearance
                samples.append([float(yaw), float(pitch), float(ins_g), round(vis['u']), round(vis['v']),
                                round(vis['front_margin'], 4), round(vis['clearance'], 4)])
            if jaws and vis and vis['front_margin'] > 0.004 and vis['clearance'] >= args.clearance:
                feasible.append(dict(yaw=float(yaw), pitch=float(pitch), ins=float(ins_g), ins_approach=float(ins_a),
                                     grasp=g['grasp'], frame=g['frame'], vis=vis))
            sim.callScriptFunction('setRCMTarget', ctrl, index, float(yaw), float(pitch), ins_a)
        sim.callScriptFunction('homeRCM', ctrl)
    print('feasible grasp poses', len(feasible), counts, flush=True)
    if args.dump:
        write_json(args.dump, {'counts': counts, 'samples_yaw_pitch_ins_u_v_front_clear': samples})
    if len(feasible) < 2:
        stop_simulation(client, sim)
        raise SystemExit('Not enough feasible visible grasp poses; widen the PSM limits or relax --clearance.')

    # Rank pick/place pairs: separation in range, both well inside the image, large clearance.
    def score(a, b):
        centre = max(abs(a['vis']['u'] - W / 2), abs(b['vis']['u'] - W / 2)) / W + \
                 max(abs(a['vis']['v'] - H / 2), abs(b['vis']['v'] - H / 2)) / H
        return min(a['vis']['clearance'], b['vis']['clearance']) - 0.01 * centre

    pairs = [(a, b) for a, b in itertools.permutations(feasible, 2)
             if args.separation[0] <= np.linalg.norm(a['grasp'] - b['grasp']) <= args.separation[1]]
    pairs.sort(key=lambda ab: -score(*ab))

    probe = ecm_pairs.create_probe(sim)
    anatomy = [h for n, h in by.items() if n.startswith('Patient_') and sim.getObjectType(h) == sim.sceneobject_shape]
    chosen = None
    for a, b in pairs[:40]:
        # First retract along the docked axis, so the shaft never sweeps through the block on the way into view.
        seq = [(0.0, 0.0, a['ins_approach']),
               (a['yaw'], a['pitch'], a['ins_approach']), (a['yaw'], a['pitch'], a['ins']), (a['yaw'], a['pitch'], a['ins_approach']),
               (b['yaw'], b['pitch'], b['ins_approach']), (b['yaw'], b['pitch'], b['ins']), (b['yaw'], b['pitch'], b['ins_approach'])]
        # A temporary block at A: it must not touch anatomy, nor the PSM2 shaft at home and on the way to the
        # grasp (the shaft shapes are the ones the runtime guard checks against the block).
        kind = sim.primitiveshape_spheroid if task['block'].get('shape') == 'sphere' else sim.primitiveshape_cuboid
        cube = sim.createPrimitiveShape(kind, [task['block']['size_m']] * 3, 0)
        sim.setObjectMatrix(cube, -1, np_to_matrix(a['frame'], a['grasp']))
        shaft_pairs = [[cube, by[arm + '_tool_main_respondable_visual']], [cube, by[arm + '_rot_ref_respondable_visual']]]
        clear = not ecm_pairs.scan(sim, probe, [[cube, h] for h in anatomy])
        sim.callScriptFunction('homeRCM', ctrl)
        clear = clear and not ecm_pairs.scan(sim, probe, shaft_pairs)
        prev = (0.0, 0.0, cfg['insertion_home'])
        for k, t in enumerate(seq):
            if not clear:
                break
            # Sub-steps before the grasp so a sweep through the block is caught; afterwards the block is carried.
            parts = 6 if k < 3 else 1
            for j in range(1, parts + 1):
                sub = [prev[i] + j / parts * (t[i] - prev[i]) for i in range(3)]
                if not sim.callScriptFunction('setRCMTarget', ctrl, index, *map(float, sub)):
                    clear = False
                    break
                if k < 3 and ecm_pairs.scan(sim, probe, shaft_pairs):
                    clear = False
                    break
            prev = t
        sim.removeObjects([cube])
        if not clear:
            continue
        chosen = (a, b, seq)
        break
    sim.removeObjects([probe])
    sim.callScriptFunction('homeRCM', ctrl)
    stop_simulation(client, sim)
    sim.setStepping(False)
    if chosen is None:
        raise SystemExit('No pick/place pair passed the command-sequence and block-clearance checks.')

    a, b, seq = chosen
    nrm = M[:, 3] - b['grasp']
    task['block'] = {**task['block'], 'center_world': a['grasp'].tolist(), 'frame_world': a['frame'].tolist()}
    task['target'] = {**task['target'], 'center_world': b['grasp'].tolist(), 'normal_world': (nrm / np.linalg.norm(nrm)).tolist()}
    task['planning'] = {
        '_note': 'Generated by scripts/build/plan_visual_task.py. Objects are static kinematic fixtures; no gravity or contact physics.',
        'scene_sha256_at_planning': sha256(ENDOSCOPE_SCENE),
        'pick_command_deg_deg_m': [a['yaw'], a['pitch'], a['ins']],
        'place_command_deg_deg_m': [b['yaw'], b['pitch'], b['ins']],
        'pick_visibility': a['vis'], 'place_visibility': b['vis'],
        'separation_m': float(np.linalg.norm(a['grasp'] - b['grasp'])),
        'feasible_grasp_poses': len(feasible), 'grid_deg': step, 'span_deg': span,
        'clearance_threshold_m': args.clearance,
        'command_sequence_checked': [list(map(float, t)) for t in seq],
    }
    write_json(ENDO_CONFIG / 'task.json', task)
    print('wrote', relative(ENDO_CONFIG / 'task.json'), 'separation mm', round(task['planning']['separation_m'] * 1000, 1),
          'pick px', round(a['vis']['u']), round(a['vis']['v']), 'place px', round(b['vis']['u']), round(b['vis']['v']), flush=True)


if __name__ == '__main__':
    main()
