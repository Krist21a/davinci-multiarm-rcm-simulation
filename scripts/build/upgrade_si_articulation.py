"""Upgrade the existing Si scene to explicit wrist yaw and save a NEW verified scene.

Use a dedicated stopped simulator. The original scene and its historical manifests are
never overwritten. Checks cover kinematic mapping and small collision-guarded motions,
not all joint-space trajectories or calibrated hardware accuracy.
"""
import argparse
import hashlib
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import instrument_guard
import instruments as inst
from articulation import migrate_loaded_scene
from simtools import (INSTRUMENT_SCENE, REPO, ENDO_CONFIG, add_arguments, aliases, connect,
                      load_scene, matrix_to_np, panel_data, panel_source, read_json,
                      relative, save_scene, sha256, stop_simulation, write_json)


def rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def check_geometry(sim, by, arm):
    """Independent FK checks in stopped scene, restoring every directly varied joint."""
    tcp, yaw = by[arm + '_tcp'], by[arm + '_wrist_yaw_joint']
    j8, j9, j10, j11 = [by[f'{arm}_joint{n}'] for n in (8, 9, 10, 11)]
    handles = [j8, j9, yaw, j10, j11]
    initial = [sim.getJointPosition(h) for h in handles]
    tests = {}
    try:
        for h, label in [(j8, 'roll'), (j9, 'pitch'), (yaw, 'yaw')]:
            q = sim.getJointPosition(h)
            M = matrix_to_np(sim.getObjectMatrix(h, -1))
            p = np.array(sim.getObjectPosition(tcp, -1))
            before = matrix_to_np(sim.getObjectMatrix(tcp, -1))[:, :3]
            delta = 0.07
            sim.setJointPosition(h, q + delta)
            actual = np.array(sim.getObjectPosition(tcp, -1))
            actual_R = matrix_to_np(sim.getObjectMatrix(tcp, -1))[:, :3]
            R = rotation(M[:, 2], delta)
            tests[label] = dict(position_error_m=float(np.linalg.norm(actual - (M[:, 3] + R @ (p - M[:, 3])))),
                                rotation_error=float(np.linalg.norm(actual_R - R @ before)))
            sim.setJointPosition(h, q)
        # Neutral TCP belongs to yaw, not either moving leaf. Grip does not add yaw.
        p = np.array(sim.getObjectPosition(tcp, -1))
        sim.setJointPosition(j10, .2)
        sim.setJointPosition(j11, .2)
        tests['grip_tcp_change_m'] = float(np.linalg.norm(np.array(sim.getObjectPosition(tcp, -1)) - p))
        tests['opposed_leaf_axes_error'] = float(np.linalg.norm(
            matrix_to_np(sim.getObjectMatrix(j10, -1))[:, 2] + matrix_to_np(sim.getObjectMatrix(j11, -1))[:, 2]))
        Y = matrix_to_np(sim.getObjectMatrix(by[arm + '_dvrk_yaw_frame'], -1))
        tcp_R = matrix_to_np(sim.getObjectMatrix(tcp, -1))[:, :3]
        tests['tcp_dvrk_orientation_error'] = float(np.linalg.norm(Y[:, :3].T @ tcp_R - inst.rpy_matrix(0, math.pi / 2, math.pi / 2)))
        tests['tcp_parent_is_yaw_frame'] = sim.getObjectParent(tcp) == by[arm + '_dvrk_yaw_frame']
        expected_parent = by.get(arm + '_jaw_hinge_slide', by[arm + '_dvrk_yaw_frame'])
        tests['leaf_parents_correct'] = all(sim.getObjectParent(h) == expected_parent for h in (j10, j11))
        tests['geometric_grip_bounds'] = [list(sim.getJointInterval(h)[1]) for h in (j10, j11)]
        tests['passed'] = (all(tests[a]['position_error_m'] < 1e-7 and tests[a]['rotation_error'] < 1e-6
                               for a in ('roll', 'pitch', 'yaw')) and tests['grip_tcp_change_m'] < 1e-9
                           and tests['opposed_leaf_axes_error'] < 1e-5 and tests['tcp_dvrk_orientation_error'] < 1e-7
                           and tests['tcp_parent_is_yaw_frame'] and tests['leaf_parents_correct'])
        return tests
    finally:
        for h, q in zip(handles, initial):
            sim.setJointPosition(h, q)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--out-scene', type=Path, default=REPO / 'scenes' / 'davinci_si_articulated.ttt')
    p.add_argument('--report', type=Path, default=REPO / 'validation' / 'safety' / 'articulation.json')
    p.add_argument('--launch-simulator', type=Path)
    args = p.parse_args()
    if args.out_scene.resolve() == INSTRUMENT_SCENE.resolve() or args.out_scene.exists():
        raise SystemExit('refusing to overwrite an existing scene; choose a new --out-scene')
    process = client = sim = logfile = None
    report = dict(success=False, source_scene=relative(INSTRUMENT_SCENE), source_scene_sha256=sha256(INSTRUMENT_SCENE))
    try:
        if args.launch_simulator:
            import socket
            with socket.socket() as probe:
                probe.settimeout(.2)
                if probe.connect_ex((args.host, args.port)) == 0:
                    raise RuntimeError('dedicated simulator port already in use')
            args.report.parent.mkdir(parents=True, exist_ok=True)
            logfile = args.report.with_suffix('.sim.log').open('w', encoding='utf-8')
            env = dict(os.environ, COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX='_davinci_safety_review')
            process = subprocess.Popen([str(args.launch_simulator), '-h', f'-GzmqRemoteApi.rpcPort={args.port}'],
                                       stdin=subprocess.PIPE, stdout=logfile, stderr=subprocess.STDOUT, env=env)
            time.sleep(5)
        client, sim = connect(args)
        load_scene(client, sim, args, INSTRUMENT_SCENE, 'si_articulation_stage.ttt')
        by = aliases(sim)
        source = migrate_loaded_scene(sim, by['RCMController'])
        by = aliases(sim)
        rcm = read_json(REPO / 'config' / 'instruments' / 'rcm_config.json')
        sim.setStringProperty(by['RCMControlPanel'], 'code', panel_source(panel_data(read_json(ENDO_CONFIG / 'endoscope.json'), rcm)))
        psms = ['psm1', 'psm2', 'psm3']
        report['geometry'] = {arm: check_geometry(sim, by, arm) for arm in psms}
        ok_roles, roles = instrument_guard.check(sim, by, rcm, inst.roles())
        report['identified'] = {a: dict(model=r['identified'], ok=r['ok']) for a, r in roles.items()}
        if not ok_roles or not all(g['passed'] for g in report['geometry'].values()):
            raise RuntimeError('geometry or instrument identification gate failed')
        sim.setStepping(True)
        sim.startSimulation()
        sim.step()
        ctrl = by['RCMController']
        sim.callScriptFunction('hold', ctrl)
        report['runtime'] = {}
        for i, arm in enumerate(psms, 1):
            state = sim.callScriptFunction('getWristState', ctrl, i)
            q = [state[name] for name in ('roll', 'pitch', 'yaw')]
            checks = []
            shaft = matrix_to_np(sim.getObjectMatrix(by[arm + '_joint8'], -1))
            rcm_point = np.array(next(a['rcm_world'] for a in rcm if a['arm'] == arm))
            for axis in range(3):
                target = q.copy()
                target[axis] += .03
                accepted = sim.callScriptFunction('setWrist', ctrl, i, *target)
                actual = sim.callScriptFunction('getWristState', ctrl, i)
                restored = sim.callScriptFunction('setWrist', ctrl, i, *q)
                checks.append(dict(axis=axis, accepted=bool(accepted), restored=bool(restored), actual=actual))
            combo = [q[0] + .04, q[1] - .03, q[2] + .04]
            accepted = bool(sim.callScriptFunction('setWrist', ctrl, i, *combo))
            opened = bool(sim.callScriptFunction('setGrip', ctrl, i, .2))
            combined = sim.callScriptFunction('getWristState', ctrl, i)
            current_shaft = matrix_to_np(sim.getObjectMatrix(by[arm + '_joint8'], -1))
            residual = float(np.linalg.norm(np.cross(rcm_point - current_shaft[:, 3], current_shaft[:, 2])))
            closed = bool(sim.callScriptFunction('setGrip', ctrl, i, 0.0))
            restored = bool(sim.callScriptFunction('setWrist', ctrl, i, *q))
            combo_ok = accepted and opened and closed and restored and abs(combined['jaw'] - .4) < 1e-7
            combo_ok &= all(abs(combined[k] - t) < 1e-7 for k, t in zip(('roll', 'pitch', 'yaw'), combo))
            report['runtime'][arm] = dict(available=state['available'], checks=checks,
                                          combined_passed=bool(combo_ok), combined_state=combined,
                                          independent_rcm_error_m=residual,
                                          shaft_axis_change=float(np.linalg.norm(current_shaft[:, 2] - shaft[:, 2])))
        if not all(e['available'] and e['combined_passed'] and e['independent_rcm_error_m'] < 1e-6 and
                   e['shaft_axis_change'] < 1e-7 and all(c['accepted'] and c['restored'] for c in e['checks'])
                   for e in report['runtime'].values()):
            raise RuntimeError('small guarded wrist motion gate failed; inspect contacts instead of disabling them')
        stop_simulation(client, sim)
        save_scene(sim, args, args.out_scene)
        report.update(success=True, output_scene=relative(args.out_scene), output_sha256=sha256(args.out_scene),
                      controller_sha256=hashlib.sha256(source.encode()).hexdigest())
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                report.update(success=False, cleanup_error=str(exc))
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdin.close()
        if logfile:
            logfile.close()
    report['source_sha256'] = {relative(f): sha256(f) for f in
                               [Path(__file__).resolve(), REPO / 'scripts' / 'build' / 'articulation.py',
                                REPO / 'src' / 'instruments' / 'articulation.lua']}
    write_json(args.report, report)
    print('articulation:', report['success'], report.get('error', report.get('output_scene')), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())