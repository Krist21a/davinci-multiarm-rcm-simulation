"""Plan the endoscope arm (ECM) docking through the planned camera port.

Offline URDF kinematics places a 30-degree oblique scope through PlannedCameraPort with
the PSM approach targets on the optical axis. The null space of that task, together with
the null space of the co-docked PSM arms (their instruments keep the exact same pose), is
searched for a posture without unexpected mesh intersections. Collision queries run in a
CoppeliaSim instance holding a staged copy of the base scene that is never saved. The
chosen posture must also be collision-free at the corners of the ECM motion range.

Output: config/endoscope/ecm_plan.json
"""
import argparse
import itertools
import math
import time
import xml.etree.ElementTree as ET

import numpy as np

import ecm_pairs
from kinematics import (Chain, PoseTask, angle_gap, frames, link_vertices, rodrigues, scope_geometry,
                        unit, xy_azimuth)
from simtools import (BASE_SCENE, CONFIG, ENDO_CONFIG, URDF, add_arguments, aliases, connect, load_scene,
                      matrix_to_np, read_json, relative, sha256, stop_simulation, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--only', help='restrict the search to one geometry, e.g. 0.14,-60 (diagnostics)')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    endo = read_json(ENDO_CONFIG / 'endoscope.json')
    ecm, search = endo['ecm'], endo['ecm']['search']
    plan = read_json(CONFIG / 'registered_plan.json')
    psm_cfg = read_json(CONFIG / 'rcm_config.json')
    home = read_json(CONFIG / 'home_joint_positions.json')
    old_pairs = read_json(CONFIG / 'collision_pairs.json')
    limits = ecm['joint_limits']
    lift, rcm_joints, ins_joint = ecm['lift_joint'], ecm['rcm_joints'], ecm['insertion_joint']
    dock_joints = [lift] + rcm_joints
    ins_home = ecm['insertion_home']

    # ECM docking task: scope tip point + shaft axis.
    ecm_chain = Chain(ecm['scope_link'])
    ins_axis = next(j['axis'] for j in ecm_chain.joints if j['name'] == ins_joint)
    verts = link_vertices(ecm['scope_link'])
    proj = verts @ ins_axis
    tip_local = verts[proj > proj.max() - 1e-4].mean(axis=0)
    ecm_task = PoseTask(ecm_chain, dock_joints, {ins_joint: ins_home}, limits, tip_local, [ins_axis])

    # Co-docked PSMs: the instrument link keeps its full pose, only the arm null space moves.
    co = []
    urdf_limits = {}
    for j in ET.parse(URDF).getroot().findall('joint'):
        lim = j.find('limit')
        if lim is not None:
            urdf_limits[j.get('name')] = [float(lim.get('lower')), float(lim.get('upper'))]
    for arm in endo['co_dock']['arms']:
        cfg = next(a for a in psm_cfg if a['arm'] == arm)
        names = [arm + '_lift_joint'] + [j['name'] for j in cfg['joints']] + [arm + '_joint8']
        lims = {n: plan['joint_limit_revisions'].get(n, urdf_limits[n]) for n in names}
        chain = Chain(arm + '_rot_ref_respondable')
        task = PoseTask(chain, names, {arm + '_joint7': cfg['insertion_home']}, lims, np.zeros(3), [[0, 0, 1], [1, 0, 0]])
        x0 = np.array([home.get(n, 0.0) for n in names])
        p0, (z0, x0v) = task.evaluate(x0)
        co.append(dict(arm=arm, task=task, x0=x0, des=(p0, [z0, x0v]), port=np.array(cfg['rcm_world']),
                       tool=arm + '_tool_main_respondable_visual', radius=0.016, cfg=cfg, names=names, lims=lims,
                       chain=chain))

    port = np.array(plan['camera_port_world'])
    target = np.mean([a['target'] for a in plan['arms']], axis=0)
    anterior = np.array(plan['rotation'])[:, 2]
    theta = math.radians(ecm['oblique_deg'])

    psm_out = [xy_azimuth(-np.array(a['axis_home'])) for a in psm_cfg]
    geometries = []
    for distance in ecm['view_distance_candidates_m']:
        for azimuth in ecm['bend_azimuth_candidates_deg']:
            g = scope_geometry(port, target, -anterior, theta, distance, azimuth)
            if g:
                g.update(distance=distance, azimuth=azimuth,
                         gap_deg=min(angle_gap(xy_azimuth(-g['axis']), b) for b in psm_out))
                geometries.append(g)
    geometries.sort(key=lambda g: -g['gap_deg'])
    if args.only:
        d_, a_ = (float(v) for v in args.only.split(','))
        geometries = [g for g in geometries if abs(g['distance'] - d_) < 1e-9 and abs(g['azimuth'] - a_) < 1e-9]

    # Collision scene (staged copy, never saved).
    client, sim = connect(args)
    load_scene(client, sim, args, BASE_SCENE, 'ecm_plan_stage.ttt')
    by = aliases(sim)
    for n in dock_joints + [ins_joint]:
        sim.setJointInterval(by[n], False, [limits[n][0], limits[n][1] - limits[n][0]])
    shapes = [n for n, h in by.items() if sim.getObjectType(h) == sim.sceneobject_shape]
    labels, categories = ecm_pairs.ecm_pair_candidates(shapes)
    probe = ecm_pairs.create_probe(sim)
    all_joints = dock_joints + [ins_joint] + [n for c in co for n in c['task'].vars]
    handles = [by[n] for n in all_joints]

    def set_q(xe, ins, xcs):
        values = list(map(float, xe)) + [float(ins)] + [float(v) for xc in xcs for v in xc]
        sim.callScriptFunction('setJoints', probe, handles, values)

    x_co0 = [c['x0'] for c in co]
    set_q(np.zeros(len(dock_joints)), 0.0, x_co0)
    zero_hits = ecm_pairs.scan(sim, probe, [[by[a], by[b]] for a, b in labels])
    designed = [labels[i] for i, _ in zero_hits if categories[i] == 'ecm_same_arm']
    pair_set = {frozenset(l) for l in labels if l not in designed}
    for l in old_pairs['labels']:
        if any(l[0].startswith(c['arm'] + '_') or l[1].startswith(c['arm'] + '_') for c in co):
            pair_set.add(frozenset(l))
    check_labels = [sorted(s, key=lambda n: (not n.startswith(('ecm_', 'psm')), n)) for s in pair_set]
    check_labels.sort()
    hpairs = [[by[a], by[b]] for a, b in check_labels]
    allowed = [(ecm['scope_shape'], port, ecm['port_radius_m'])] + [(c['tool'], c['port'], c['radius']) for c in co]
    print(f'pairs checked {len(check_labels)}; designed ECM contacts excluded {designed}', flush=True)

    vis = by[ecm['scope_shape']]
    mv = np.array(sim.getShapeMesh(vis)[0]).reshape(-1, 3)

    def scene_tip(axis):
        M = matrix_to_np(sim.getObjectMatrix(vis, -1))
        wv = mv @ M[:, :3].T + M[:, 3]
        pv = wv @ axis
        return wv[pv > pv.max() - 1e-4].mean(axis=0)
    T_zero = ecm_chain.fk({})                       # scene was just set to the CAD zero pose
    t0, a0 = (T_zero @ np.append(tip_local, 1))[:3], T_zero[:3, :3] @ ins_axis
    tip_consistency = float(np.linalg.norm(scene_tip(a0) - t0))
    if tip_consistency > 1e-5:
        raise SystemExit(f'Scene scope mesh does not match the URDF scope tip ({tip_consistency} m).')

    def evaluate(xe, xcs, ins=ins_home):
        set_q(xe, ins, xcs)
        return evaluate_current()

    def evaluate_current(ecm_only=False):
        """Unexpected contacts at the current joint state, and cost = intersection length + 2 mm per pair.

        ecm_only: count only pairs with an ECM shape (used for co-docked PSM motion, whose own contacts
        with anatomy are left to the runtime collision guard)."""
        bad, crossing, length = [], None, 0.0
        for i, pts in ecm_pairs.scan(sim, probe, hpairs):
            pair = check_labels[i]
            if ecm_only and not any(n.startswith('ecm_') for n in pair):
                continue
            ok = False
            for shape, prt, rad in allowed:
                if pair == [shape, 'Patient_Skin'] and len(pts):
                    r = float(np.linalg.norm(pts - prt, axis=1).max())
                    ok = r <= rad
                    if ok and shape == ecm['scope_shape']:
                        crossing = r
            if ok:
                continue
            bad.append(pair)
            n = len(pts) // 2
            if n:
                length += float(np.linalg.norm(pts[0:2 * n:2] - pts[1:2 * n:2], axis=1).sum())
        return bad, crossing, length + 0.002 * len(bad)

    def solve_co(xcs_seed):
        out = []
        for c, seed in zip(co, xcs_seed):
            x, r = c['task'].solve(seed, *c['des'])
            if r > 1e-8:
                return None
            out.append(x)
        return out

    motion = ecm['motion']
    corners = ([(y, pc, di) for y in (-1, 1) for pc in (-1, 1) for di in (-1, 1)]
               + [(y, 0, 0) for y in (-1, 1)] + [(0, pc, 0) for pc in (-1, 1)] + [(0, 0, di) for di in (-1, 1)])

    stress_corners = [(y, pc, di) for y in (-1, 1) for pc in (-1, 1) for di in (-1, 1)]

    co_motion = endo['co_dock'].get('motion_check')

    def co_corner_check(xe, xcs, stop_at_failure=True, step_deg=1.5):
        """Co-docked PSM RCM motion corners (shaft point + axis, like the runtime controller), ECM at home."""
        results, all_ok = [], True
        if not co_motion:
            return True, results
        for ci, c in enumerate(co):
            cfg = c['cfg']
            x_home = dict(zip(c['names'], xcs[ci]))
            rcm_names = [j['name'] for j in cfg['joints']]
            fixed = {n: v for n, v in x_home.items() if n not in rcm_names}
            r_w, a0 = np.array(cfg['rcm_world']), np.array(cfg['axis_home'])
            tx, ty = np.array(cfg['transverse_x']), np.array(cfg['transverse_y'])
            for yw, pc, ins in itertools.product((-1, 1), (-1, 1), co_motion['insertions']):
                yaw, pitch = yw * co_motion['yaw_deg'], pc * co_motion['pitch_deg']
                steps = max(1, math.ceil(abs(yaw) / step_deg), math.ceil(abs(pitch) / step_deg))
                xr, r = np.array([x_home[n] for n in rcm_names]), 0.0
                for k in range(1, steps + 1):
                    f = k / steps
                    d = unit(a0 + math.tan(math.radians(f * yaw)) * tx + math.tan(math.radians(f * pitch)) * ty)
                    ins_k = cfg['insertion_home'] + f * (ins - cfg['insertion_home'])
                    task = PoseTask(c['chain'], rcm_names, {**fixed, c['arm'] + '_joint7': ins_k}, c['lims'],
                                    np.zeros(3), [[0, 0, 1]])
                    xr, r = task.solve(xr, r_w - d * (cfg['rcm_distance_home'] - (ins_k - cfg['insertion_home'])), [d], max_nfev=60)
                    if r > 1e-7:
                        break
                q_all = [x_home[n] if n not in rcm_names else float(xr[rcm_names.index(n)]) for n in c['names']]
                xcs_k = [list(x) for x in xcs]
                xcs_k[ci] = q_all
                # The insertion joint is not a co-dock variable; set it through the probe afterwards.
                set_q(xe, ins_home, xcs_k)
                sim.callScriptFunction('setJoints', probe, [by[c['arm'] + '_joint7']], [float(ins)])
                bad, _, cost = evaluate_current(ecm_only=True)
                sim.callScriptFunction('setJoints', probe, [by[c['arm'] + '_joint7']], [float(cfg['insertion_home'])])
                results.append(dict(arm=c['arm'], target=[yaw, pitch, ins], ik_residual=r, collisions=bad,
                                    length=max(0.0, cost - 0.002 * len(bad))))
                # An unreachable corner (IK residual) is anatomy/limit-bound, not the ECM's doing: record, don't fail.
                if bad:
                    all_ok = False
                    if stop_at_failure:
                        return False, results
        return all_ok, results

    def corner_cost(xe, xcs, g):
        """Search objective: IK failures + intersection length over the ECM diagonal corners and the
        co-docked PSM motion corners (coarse 1.5 deg paths)."""
        _, results = corner_check(xe, xcs, g, stress_corners, step_deg=1.5, stop_at_failure=False)
        _, co_results = co_corner_check(xe, xcs, stop_at_failure=False)
        cost = 0.0
        for r in results:
            cost += (1.0 if r['ik_residual'] > 1e-7 else 0.0) + r['length'] + 0.002 * len(r['collisions'])
        for r in co_results:
            cost += r['length'] + 0.002 * len(r['collisions'])
        return cost, results + co_results

    def corner_check(xe, xcs, g, corner_list=None, step_deg=.5, stop_at_failure=True):
        """IK at the ECM motion-range corners with the runtime controller's task (tip + full tip frame)."""
        corner_list = corners if corner_list is None else corner_list
        q0 = ecm_task.q(xe)
        T0 = ecm_chain.fk(q0)
        tip0 = (T0 @ np.append(tip_local, 1))[:3]
        a_home = T0[:3, :3] @ ins_axis
        tipF, _ = frames(a_home, g['view'], anterior)
        R0 = T0[:3, :3]
        D_home = float(np.dot(port - tip0, a_home))
        rlim = {n: limits[n] for n in rcm_joints}
        results = []
        all_ok = True
        for yw, pc, di in corner_list:
            yaw, pitch, dins = yw * motion['yaw_deg'], pc * motion['pitch_deg'], di * motion['insertion_delta_m']
            # Follow the path from home (controller steps: 0.5 deg / 5 mm), seeding each IK step from the last.
            steps = max(1, math.ceil(abs(yaw) / step_deg), math.ceil(abs(pitch) / step_deg), math.ceil(abs(dins) / .005))
            xr, r = xe[1:].copy(), 0.0
            for k in range(1, steps + 1):
                f = k / steps
                d = unit(a_home + math.tan(math.radians(f * yaw)) * tipF[:, 0] + math.tan(math.radians(f * pitch)) * tipF[:, 1])
                Rd = rodrigues(a_home, d) @ tipF
                task = PoseTask(ecm_chain, rcm_joints, {lift: xe[0], ins_joint: ins_home + f * dins}, rlim, tip_local,
                                [R0.T @ tipF[:, 2], R0.T @ tipF[:, 0]])
                xr, r = task.solve(xr, port - d * (D_home - f * dins), [Rd[:, 2], Rd[:, 0]], max_nfev=60)
                if r > 1e-7:
                    break
            bad, _, cost = evaluate(np.r_[xe[0], xr], xcs, ins_home + dins)
            results.append(dict(target=[yaw, pitch, dins], ik_residual=r, collisions=bad,
                                length=max(0.0, cost - 0.002 * len(bad))))
            if r > 1e-7 or bad:
                all_ok = False
                if stop_at_failure:
                    return False, results
        return all_ok, results

    chosen, report = None, []
    sigmas = search['local_sigmas_rad']
    for g in geometries:
        pool = []
        for k in range(search['random_ik_per_geometry']):
            seed = np.zeros(len(dock_joints)) if k == 0 else rng.uniform(np.maximum(ecm_task.lo, -1.6), np.minimum(ecm_task.hi, 1.6))
            xe, r = ecm_task.solve(seed, g['tip'], [g['axis']])
            if r < 1e-8:
                bad, crossing, cost = evaluate(xe, x_co0)
                pool.append([cost, xe, x_co0, bad, crossing])
        pool.sort(key=lambda row: row[0])
        best = None
        for cost, xe, xcs, bad, crossing in pool[:search['local_starts']]:
            for it in range(search['local_iterations']):
                if not bad and ecm_task.margin(xe, 1) >= search['min_joint_margin']:
                    break
                sigma = sigmas[it % len(sigmas)]
                te, r = ecm_task.solve(xe + rng.normal(0, sigma, len(xe)), g['tip'], [g['axis']])
                if r > 1e-8:
                    continue
                tcs = xcs
                if co and rng.random() < search['co_dock_move_probability']:
                    tcs = solve_co([x + rng.normal(0, sigma, len(x)) for x in xcs]) or xcs
                tb, tc, tcost = evaluate(te, tcs)
                # Minimise contacts first; once contact-free, push the ECM joints away from their limits.
                if tcost < cost or (tcost == cost == 0 and ecm_task.margin(te, 1) > ecm_task.margin(xe, 1)):
                    cost, xe, xcs, bad, crossing = tcost, te, tcs, tb, tc
            if best is None or cost < best['cost']:
                best = dict(cost=cost, bad=bad, margin=ecm_task.margin(xe, 1))
            if bad or ecm_task.margin(xe, 1) < search['min_joint_margin']:
                continue
            # Stage 2: keep the home pose clear and within margin, and clear the motion-range corners.
            ccost, _ = corner_cost(xe, xcs, g)
            for it in range(search['corner_iterations']):
                if ccost == 0:
                    break
                sigma = search['corner_sigmas_rad'][it % len(search['corner_sigmas_rad'])]
                te, r = ecm_task.solve(xe + rng.normal(0, sigma, len(xe)), g['tip'], [g['axis']])
                if r > 1e-8 or ecm_task.margin(te, 1) < search['min_joint_margin']:
                    continue
                tcs = xcs
                if co and rng.random() < search['co_dock_move_probability']:
                    tcs = solve_co([x + rng.normal(0, sigma, len(x)) for x in xcs]) or xcs
                tb, tc, _ = evaluate(te, tcs)
                if tb:
                    continue
                tcc, _ = corner_cost(te, tcs, g)
                if tcc < ccost:
                    ccost, xe, xcs, crossing = tcc, te, tcs, tc
            best['corner_cost'] = round(ccost, 5)
            print(f'   stage 2: corner cost {ccost:.5f} t={time.time() - t_start:.0f}s', flush=True)
            if args.verbose and ccost > 0:
                _, cres = corner_cost(xe, xcs, g)
                for r_ in cres:
                    print(f"     corner {r_['target']} ik={r_['ik_residual']:.1e} len={r_['length']:.3f} {r_['collisions']}", flush=True)
            if ccost > 0:
                continue
            ok, corner_results = corner_check(xe, xcs, g)
            co_ok, co_results = co_corner_check(xe, xcs, step_deg=.5) if ok else (False, [])
            if ok and co_ok:
                chosen = dict(g=g, xe=xe, xcs=xcs, crossing=crossing, corners=corner_results, co_corners=co_results)
                break
            best['corner_failure'] = (corner_results if not ok else co_results)[-1]
        row = dict(view_distance_m=g['distance'], bend_azimuth_deg=g['azimuth'], gap_deg=round(g['gap_deg'], 1),
                   ik_solutions=len(pool), best_cost=None if best is None else round(best['cost'], 5),
                   best_contacts=None if best is None else best['bad'],
                   corner_failure=None if best is None else best.get('corner_failure'))
        report.append(row)
        print(f"D={g['distance']:.2f} az={g['azimuth']:+.0f} gap={g['gap_deg']:.0f} sols={len(pool)} cost={row['best_cost']} "
              f"contacts={None if best is None else len(best['bad'])} margin={None if best is None else round(best['margin'], 3)} corner_fail={row['corner_failure'] is not None} "
              f"t={time.time() - t_start:.0f}s", flush=True)
        if row['corner_failure']:
            print('   first failing corner:', row['corner_failure'], flush=True)
        if chosen:
            break

    if chosen is None:
        sim.removeObjects([probe])
        write_json(ENDO_CONFIG / 'ecm_plan_rejected.json', report)
        raise SystemExit('No collision-free ECM docking; see config/endoscope/ecm_plan_rejected.json')

    g, xe, xcs = chosen['g'], chosen['xe'], chosen['xcs']
    q = {**ecm_task.q(xe)}
    T0 = ecm_chain.fk(q)
    tip0 = (T0 @ np.append(tip_local, 1))[:3]
    a_home = T0[:3, :3] @ ins_axis
    set_q(xe, ins_home, xcs)
    fk_err = float(np.linalg.norm(scene_tip(a_home) - tip0))
    co_err = []
    co_q = {}
    for c, x in zip(co, xcs):
        co_q.update(dict(zip(c['task'].vars, map(float, x))))
        j8 = by[c['arm'] + '_joint8']
        M = matrix_to_np(sim.getObjectMatrix(j8, -1))
        cfg = next(a for a in psm_cfg if a['arm'] == c['arm'])
        co_err.append(float(max(np.linalg.norm(M[:, 3] - cfg['shaft_home']), np.linalg.norm(M[:, 2] - cfg['axis_home']))))
    sim.removeObjects([probe])
    stop_simulation(client, sim)
    if fk_err > 1e-6 or max(co_err, default=0) > 1e-6:
        raise SystemExit(f'Scene FK disagrees with URDF FK: ECM {fk_err} m, co-docked {co_err}')
    tipF, camF = frames(a_home, g['view'], anterior)
    # Rank of the runtime task (tip velocity + angular velocity) with respect to the RCM joints.
    axes = {n: None for n in rcm_joints}
    ecm_chain.fk(q, axes)
    cols = [np.r_[np.cross(axes[n][1], tip0 - axes[n][0]), axes[n][1]] for n in rcm_joints]
    sv = np.linalg.svd(np.array(cols).T, compute_uv=False)

    out = {
        '_note': 'Generated by scripts/build/plan_ecm_docking.py. Exploratory simulation docking, not a clinical setup.',
        'inputs': {relative(p_): sha256(p_) for p_ in [ENDO_CONFIG / 'endoscope.json', CONFIG / 'registered_plan.json',
                                                     CONFIG / 'rcm_config.json', CONFIG / 'home_joint_positions.json',
                                                     CONFIG / 'collision_pairs.json', URDF, BASE_SCENE]},
        'camera_port_world': port.tolist(),
        'look_target_world': target.tolist(),
        'look_target_basis': 'mean of the three PSM approach targets in config/registered_plan.json',
        'oblique_deg': ecm['oblique_deg'],
        'view_distance_m': g['distance'],
        'bend_azimuth_deg': g['azimuth'],
        'scope_depth_past_port_m': g['depth'],
        'shaft_tilt_from_port_target_line_deg': g['alpha_deg'],
        'ecm_joint_positions': q,
        'co_dock_joint_positions': co_q,
        'co_dock_basis': 'Null-space re-posture; instrument link pose kept, joint8 compensates the shaft-frame roll.',
        'co_dock_instrument_error_m': co_err,
        'ecm_joint_limits': limits,
        'ecm_joint_margin_normalised': ecm_task.margin(xe, 1),
        'scope_tip_local': tip_local.tolist(),
        'tip_world': tip0.tolist(),
        'tip_frame_world': tipF.tolist(),
        'camera_frame_world': camF.tolist(),
        'shaft_axis_world': a_home.tolist(),
        'view_axis_world': g['view'].tolist(),
        'scene_fk_error_m': fk_err,
        'tip_consistency_urdf_vs_scene_m': tip_consistency,
        'task_singular_values': sv.tolist(),
        'task_rank': int(np.sum(sv > 1e-6)),
        'home_port_crossing_radius_m': chosen['crossing'],
        'designed_zero_pose_contacts_excluded': designed,
        'motion_range_checked': motion,
        'motion_corner_results': chosen['corners'],
        'co_dock_motion_checked': co_motion,
        'co_dock_motion_corner_results': chosen['co_corners'],
        'search_args': {'seed': args.seed, 'only': args.only},
        'search_report': report,
        'search_seconds': round(time.time() - t_start, 1),
    }
    write_json(ENDO_CONFIG / 'ecm_plan.json', out)
    print('wrote', relative(ENDO_CONFIG / 'ecm_plan.json'), 'task rank', out['task_rank'], flush=True)


if __name__ == '__main__':
    main()
