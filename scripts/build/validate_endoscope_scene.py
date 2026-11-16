"""Headless regression check of scenes/davinci_endoscope_task.ttt.

Loads a staged copy of the scene (never saved) and checks, with the scene's own scripts:
  * embedded controller/panel text == generated config + src/endoscope/*.lua;
  * one full 24 s PSM demo cycle (481 steps) with the endoscope holding still:
    solver failures, collision rollbacks, RCM residual (controller and independent axis measure),
    independent mesh-collision scans every 10 steps;
  * ECM manual targets at the corners of its range, each followed by an independent collision scan;
  * an out-of-range ECM target is rejected without moving joints;
  * endoscope image and depth: PSM instruments and the prostate are visible, depth values are sane.
Writes validation/endoscope/scene_validation.json and endoscope_view.png.
"""
import argparse
import itertools
import json
import math

import numpy as np

import ecm_pairs
from simtools import (ENDO_CONFIG, ENDO_VALIDATION, ENDOSCOPE_SCENE, add_arguments, aliases, connect,
                      controller_source, load_scene, matrix_to_np, panel_data, panel_source, read_json, relative,
                      sha256, stop_simulation, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--steps', type=int, default=481)
    p.add_argument('--scan-every', type=int, default=10)
    args = p.parse_args()

    endo = read_json(ENDO_CONFIG / 'endoscope.json')
    rcm = read_json(ENDO_CONFIG / 'rcm_config.json')
    pairs = read_json(ENDO_CONFIG / 'collision_pairs.json')
    client, sim = connect(args)
    load_scene(client, sim, args, ENDOSCOPE_SCENE, 'endoscope_validation_stage.ttt')
    by = aliases(sim)
    result = {'scene': relative(ENDOSCOPE_SCENE), 'scene_sha256': sha256(ENDOSCOPE_SCENE)}

    # 1. Embedded scripts match the generated sources.
    def code(alias):
        return sim.getStringProperty(by[alias], 'code').replace('\r\n', '\n')
    result['embedded_controller_matches_source'] = code('RCMController') == controller_source(rcm, pairs['labels'])
    result['embedded_panel_matches_source'] = code('RCMControlPanel') == panel_source(panel_data(endo, rcm))

    labels = pairs['labels']
    hpairs = [[by[a], by[b]] for a, b in labels]
    allowed = {a['tool_shape']: (np.array(a['rcm_world']), a['port_radius_m']) for a in rcm}
    probe = ecm_pairs.create_probe(sim)

    def scan():
        bad, crossings = [], {}
        for i, pts in ecm_pairs.scan(sim, probe, hpairs):
            a, b = labels[i]
            if b == 'Patient_Skin' and a in allowed and len(pts):
                r = float(np.linalg.norm(pts - allowed[a][0], axis=1).max())
                if r <= allowed[a][1]:
                    crossings[a] = max(crossings.get(a, 0.0), r)
                    continue
            bad.append(labels[i])
        return bad, crossings

    def independent_residuals():
        out = []
        for a in rcm:
            M = matrix_to_np(sim.getObjectMatrix(by[a['shaft_frame']], -1))
            out.append(float(np.linalg.norm(np.cross(np.array(a['rcm_world']) - M[:, 3], M[:, 2]))))
        return out

    ecm_joints = [by[j['name']] for j in rcm[-1]['joints']] + [by[rcm[-1]['insertion_joint']]]
    ecm_home = [sim.getJointPosition(h) for h in ecm_joints]
    ctrl = by['RCMController']
    fixed = {n: np.array(sim.getObjectMatrix(by[n], -1)) for n in
             ['BodyParts3D_Male', 'TaskSurface'] + [a['arm'] + '_RCM_reference' for a in rcm]}

    # 2. One full demo cycle.
    sim.setStepping(True)
    sim.setBoolParam(sim.boolparam_realtime_simulation, False)
    sim.startSimulation()
    demo = dict(steps=args.steps, max_rcm_error_m=[0.0] * len(rcm), independent_max_rcm_error_m=[0.0] * len(rcm),
                solver_failures=None, collision_blocks=None, unexpected_events=[], scans=0,
                max_port_crossing_radius_m={}, ecm_max_joint_change=0.0, fixed_frame_max_change=0.0)
    for k in range(args.steps):
        sim.step()
        stats = sim.callScriptFunction('getRCMStats', ctrl)
        for i, a in enumerate(stats):
            demo['max_rcm_error_m'][i] = max(demo['max_rcm_error_m'][i], a['drift_m'])
        if k % args.scan_every == 0 or k == args.steps - 1:
            demo['scans'] += 1
            bad, crossings = scan()
            if bad:
                demo['unexpected_events'].append({'time': sim.getSimulationTime(), 'contacts': bad})
            for s_, r in crossings.items():
                demo['max_port_crossing_radius_m'][s_] = max(demo['max_port_crossing_radius_m'].get(s_, 0.0), r)
            for i, r in enumerate(independent_residuals()):
                demo['independent_max_rcm_error_m'][i] = max(demo['independent_max_rcm_error_m'][i], r)
            q = [sim.getJointPosition(h) for h in ecm_joints]
            demo['ecm_max_joint_change'] = max(demo['ecm_max_joint_change'], float(np.max(np.abs(np.array(q) - ecm_home))))
            for n, m in fixed.items():
                demo['fixed_frame_max_change'] = max(demo['fixed_frame_max_change'],
                                                     float(np.max(np.abs(np.array(sim.getObjectMatrix(by[n], -1)) - m))))
    demo['solver_failures'] = [a['failures'] for a in stats]
    demo['collision_blocks'] = [a['collision_blocks'] for a in stats]
    demo['simulation_time_s'] = sim.getSimulationTime()
    result['psm_demo_cycle'] = demo

    # 3. ECM manual targets.
    ecm_index = len(rcm)
    L = rcm[-1]['limits']
    sim.callScriptFunction('homeRCM', ctrl)
    manual = []
    for yaw, pitch, ins in itertools.product([-L['yaw_deg'], L['yaw_deg']], [-L['pitch_deg'], L['pitch_deg']],
                                             [L['insertion_min'], L['insertion_max']]):
        ok = sim.callScriptFunction('setRCMTarget', ctrl, ecm_index, yaw, pitch, ins)
        bad, _ = scan()
        st = sim.callScriptFunction('getRCMStats', ctrl)[-1]
        manual.append(dict(target=[yaw, pitch, ins], accepted=ok, unexpected_contacts=bad, drift_m=st['drift_m'],
                           independent_rcm_error_m=independent_residuals()[-1]))
        sim.callScriptFunction('homeRCM', ctrl)
    result['ecm_manual_corners'] = manual
    before = [sim.getJointPosition(h) for h in ecm_joints]
    rejected = not sim.callScriptFunction('setRCMTarget', ctrl, ecm_index, L['yaw_deg'] * 1.5, 0, rcm[-1]['insertion_home'])
    after = [sim.getJointPosition(h) for h in ecm_joints]
    result['ecm_out_of_range'] = dict(rejected=rejected, joint_change=float(np.max(np.abs(np.array(after) - before))))

    # 4. Endoscope image and depth at the home pose.
    sim.callScriptFunction('homeRCM', ctrl)
    cam = by[endo['camera']['alias']]
    sim.handleVisionSensor(cam)
    raw, res = sim.getVisionSensorImg(cam)
    depth_raw, _ = sim.getVisionSensorDepth(cam, 1)
    depth = np.array(sim.unpackFloatTable(depth_raw), dtype=float).reshape(res[1], res[0])
    img = np.frombuffer(raw, dtype=np.uint8).reshape(res[1], res[0], 3)[::-1]
    try:
        from PIL import Image
        ENDO_VALIDATION.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(ENDO_VALIDATION / 'endoscope_view.png')
    except ImportError:
        pass
    # Visibility of each instrument tip and of the look target, from the known camera model.
    M = matrix_to_np(sim.getObjectMatrix(cam, -1))
    f = (res[0] / 2) / math.tan(math.radians(endo['camera']['view_angle_deg']) / 2)

    def project(pw):
        pc = M[:, :3].T @ (np.asarray(pw) - M[:, 3])
        if pc[2] <= 0:
            return None
        return [res[0] / 2 - f * pc[0] / pc[2], res[1] / 2 - f * pc[1] / pc[2], float(pc[2])]
    plan = read_json(ENDO_CONFIG / 'ecm_plan.json')
    vis = {}
    for name in ['psm1_tcp', 'psm2_tcp', 'psm3_tcp']:
        vis[name] = project(sim.getObjectPosition(by[name], -1))
    vis['look_target'] = project(plan['look_target_world'])
    inside = {k: bool(v is not None and 0 <= v[0] < res[0] and 0 <= v[1] < res[1]) for k, v in vis.items()}

    def depth_at(uv, window=2):
        # depth rows are bottom-up (same as the raw image); projections are top-down
        u, v = int(round(uv[0])), int(round(uv[1]))
        r = res[1] - 1 - v
        patch = depth[max(0, r - window):r + window + 1, max(0, u - window):u + window + 1]
        return float(patch.max()) if patch.size else None

    # A point is unoccluded when the rendered surface at its pixel is not in front of it.
    tip_visibility = {}
    for name in ['psm1_tcp', 'psm2_tcp', 'psm3_tcp']:
        v = vis[name]
        if inside[name]:
            d = depth_at(v)
            tip_visibility[name] = dict(depth_at_pixel_m=d, distance_along_axis_m=v[2], unoccluded=bool(d >= v[2] - 0.006))
        else:
            tip_visibility[name] = dict(unoccluded=False)
    pv = np.array(sim.getShapeMesh(by['Patient_Prostate'])[0]).reshape(-1, 3)
    Mp = matrix_to_np(sim.getObjectMatrix(by['Patient_Prostate'], -1))
    seen = in_frame = 0
    for pw in (pv @ Mp[:, :3].T + Mp[:, 3])[::max(1, len(pv) // 400)]:
        pr = project(pw)
        if pr is None or not (0 <= pr[0] < res[0] and 0 <= pr[1] < res[1]):
            continue
        in_frame += 1
        seen += depth_at(pr, 0) >= pr[2] - 0.003
    result['endoscope_image'] = dict(
        resolution=res, depth_min_m=float(depth.min()), depth_median_m=float(np.median(depth)),
        depth_far_fraction=float(np.mean(depth >= endo['camera']['far_m'] * 0.999)),
        mean_rgb=img.reshape(-1, 3).mean(axis=0).tolist(), projections_px_depth=vis, inside_image=inside,
        instrument_tip_visibility=tip_visibility,
        prostate_vertices_in_frame=in_frame, prostate_vertices_unoccluded=int(seen),
        note='Pinhole model with the horizontal view angle; projections are top-down pixel rows. '
             'Visibility compares the depth buffer with each point\'s distance along the optical axis.')

    sim.removeObjects([probe])
    stop_simulation(client, sim)
    sim.setStepping(False)
    sim.setBoolParam(sim.boolparam_realtime_simulation, True)

    checks = {
        'scripts_match_sources': result['embedded_controller_matches_source'] and result['embedded_panel_matches_source'],
        'demo_no_failures': sum(demo['solver_failures']) == 0 and sum(demo['collision_blocks']) == 0,
        'demo_no_unexpected_contacts': not demo['unexpected_events'],
        'demo_rcm_error_below_1um': max(demo['independent_max_rcm_error_m']) < 1e-6,
        'ecm_held_during_psm_demo': demo['ecm_max_joint_change'] < 1e-12,
        'fixed_frames_unchanged': demo['fixed_frame_max_change'] == 0.0,
        'ecm_corners_accepted_and_clear': all(m['accepted'] and not m['unexpected_contacts'] and m['independent_rcm_error_m'] < 1e-6 for m in manual),
        'ecm_out_of_range_rejected': rejected and result['ecm_out_of_range']['joint_change'] == 0.0,
        'instrument_tips_and_target_in_frame': all(inside.values()),
    }
    # Reported, not gated: static anatomy (no bladder take-down, no pneumoperitoneum) can hide tips and prostate.
    result['visibility_summary'] = {
        'unoccluded_tips': [k for k, v in tip_visibility.items() if v['unoccluded']],
        'prostate_unoccluded_fraction': (seen / in_frame) if in_frame else 0.0,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    result['checks'] = checks
    result['passed'] = all(checks.values())
    write_json(ENDO_VALIDATION / 'scene_validation.json', result)
    print(json.dumps(checks, indent=2), '\npassed', result['passed'], flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
