"""Re-plan the PSM1 / PSM3 skin ports so both retraction arms reach the cutting workspace.

With the original ports PSM1 needs yaw 7.3 deg / pitch -6.2 deg to reach its grasp point (limits +-1 / +-0.75 deg)
and both tips sit behind the pubic bones. PSM2 (scissors) and the ECM keep their ports.

1. Workspace: PSM2 at its ready pose defines the tissue sheet (scripts/task/workspace.py); the grasp points
   are on its two side edges (PSM1 on the -u side, PSM3 on the +u side).
2. Candidate ports: skin vertices (models/anatomy/corrected_skin.npz) on a 6 mm grid, above the workspace,
   at a port-to-grasp distance that keeps the insertion joint at its home value. Each candidate must
     - keep >= 60 mm to every other port (PSM2, ECM and the other new port),
     - meet the PSM2 shaft at 25..80 deg (instruments neither parallel nor head-on),
     - leave the endoscope's lines of sight to the cut site and the scissors tip (>= 7 mm), and
     - give a straight instrument path from the port to the grasp point that clears the bones and organs
       (a 6 mm radius cylinder, checked as a mesh collision in CoppeliaSim).
   Candidates are ranked by distance from the original port plus half the depth difference to 127 mm.
3. Arm posture: URDF IK puts the instrument shaft (joint8 frame) on the port-to-grasp line with the insertion
   joint at home and the tool tip 4 mm short of the grasp point; a random null-space search removes every
   unexpected mesh contact of that arm (same pairs as the runtime guard). The posture must also stay
   contact-free over the task motion: approach and retraction along the shaft, +-2 deg corrections.

Output: config/instruments/port_plan.json (applied by install_instruments.py).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import ecm_pairs
from kinematics import Chain, PoseTask, unit
from simtools import (CONFIG, ENDO_CONFIG, INSTRUMENT_SCENE, REPO, URDF, add_arguments, aliases, connect, load_scene,
                      matrix_to_np, np_to_matrix, read_json, relative, sha256, stop_simulation, write_json)

sys.path.insert(0, str(REPO / 'scripts' / 'task'))
import workspace  # noqa: E402

OUT = CONFIG / 'instruments' / 'port_plan.json'
ARMS = ['psm1', 'psm3']
GRID_M = 0.006
DEPTH_RANGE_M = (0.09, 0.17)          # port to grasp point; original ports: 127 mm tip depth at insertion home
MIN_PORT_SPACING_M = 0.06
ANGLE_TO_PSM2_DEG = (25.0, 80.0)
SIGHT_CLEARANCE_M = 0.007
PATH_RADIUS_M = 0.006                 # 4.2 mm shaft + 1.8 mm margin
APPROACH_GAP_M = 0.004                # home tool tip this far short of the grasp point
# Task motion: approach (+5 mm) and retraction (-10 mm) along the shaft, +-2 deg corrections at the grasp depth.
MOTION = dict(axial_m=(-0.010, APPROACH_GAP_M + 0.001), lateral_deg=2.0)
NEW_LIMITS = dict(yaw_deg=5.0, pitch_deg=5.0, insertion_min=0.025, insertion_max=0.055)
CANDIDATES_PER_ARM = 40
NOMINAL_DEPTH_M = 0.127              # tip depth below the port at insertion home with the original ports
MIN_JOINT_MARGIN = 0.08               # normalised distance of the RCM joints from their limits at home


def seg_seg_distance(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-18 else 0.0
    t = (b * s + f) / e
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + s * d1 - (p2 + t * d2)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--random-ik', type=int, default=25)
    p.add_argument('--local-iterations', type=int, default=120)
    args = p.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rcm = read_json(ENDO_CONFIG / 'rcm_config.json')
    task = read_json(ENDO_CONFIG / 'task.json')
    pairs = read_json(ENDO_CONFIG / 'collision_pairs.json')['labels']
    plan0 = read_json(CONFIG / 'registered_plan.json')
    cfg = {a['arm']: a for a in rcm}
    index = {a['arm']: i for i, a in enumerate(rcm, 1)}

    client, sim = connect(args)
    load_scene(client, sim, args, INSTRUMENT_SCENE, 'port_plan_stage.ttt')
    by = aliases(sim)
    ctrl = by['RCMController']

    # 1. Workspace from PSM2 at its ready pose (the simulation restores the joints when it stops).
    sim.setStepping(True)
    sim.startSimulation()
    sim.step()
    sim.callScriptFunction('homeRCM', ctrl)
    for _ in range(20):
        sim.step()
    for ready in task['planning']['command_sequence_checked'][:2]:
        sim.callScriptFunction('setRCMTarget', ctrl, index['psm2'], *map(float, ready))
    for _ in range(30):
        sim.step()
    j9, j10 = (matrix_to_np(sim.getObjectMatrix(by[f'psm2_joint{n}'], -1)) for n in (9, 10))
    tcp2 = np.array(sim.getObjectPosition(by['psm2_tcp'], -1))
    W = workspace.layout(j10[:, 3], unit(j10[:, 3] - j9[:, 3]), j10[:, 2], tcp2)
    camM = matrix_to_np(sim.getObjectMatrix(by['EndoscopeCamera'], -1))
    cam = camM[:, 3]
    psm2_axis = matrix_to_np(sim.getObjectMatrix(by['psm2_joint8'], -1))[:, 2]
    tcp_len = {}
    for a in ARMS:
        j8 = matrix_to_np(sim.getObjectMatrix(by[f'{a}_joint8'], -1))
        tcp_len[a] = float(np.dot(np.array(sim.getObjectPosition(by[f'{a}_tcp'], -1)) - j8[:, 3], j8[:, 2]))
    sim.callScriptFunction('setAuto', ctrl, False)
    from simtools import stop_simulation as _stop
    _stop(client, sim)
    sim.setStepping(False)
    by = aliases(sim)
    print('workspace cut start', np.round(W['cut_start'], 4), 'grasp', {a: np.round(g, 4).tolist() for a, g in W['grasp_points'].items()}, flush=True)

    # 2. Candidate ports on the skin.
    z = np.load(REPO / 'models' / 'anatomy' / 'corrected_skin.npz')
    Ms = matrix_to_np(sim.getObjectMatrix(by['Patient_Skin'], -1))
    skin = z['vertices'] @ Ms[:, :3].T + Ms[:, 3]
    keys = np.round(skin / GRID_M).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    skin = skin[np.sort(first)]
    anterior = np.array(plan0['rotation'])[:, 2]
    fixed_ports = {a: np.array(cfg[a]['rcm_world']) for a in ('psm2', 'ecm')}
    anatomy = [n for n in by if n.startswith('Patient_') and n != 'Patient_Skin']
    others = ['psm2_tool_main_respondable_visual', 'ecm_insertion_visual']
    cyl = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [2 * PATH_RADIUS_M, 2 * PATH_RADIUS_M, 0.1], 0)
    probe = ecm_pairs.create_probe(sim)

    def path_clear(P, G):
        L = float(np.linalg.norm(G - P))
        a = (G - P) / L
        start, end = P + 0.012 * a, G - 0.004 * a           # leave the skin crossing and the jaws out
        length = float(np.linalg.norm(end - start))
        # Rebuild the cylinder for this length (primitive sizes cannot be changed in place).
        nonlocal cyl
        sim.removeObjects([cyl])
        cyl = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [2 * PATH_RADIUS_M, 2 * PATH_RADIUS_M, length], 0)
        x = unit(np.cross(a, [0, 0, 1.0]) if abs(a[2]) < .9 else np.cross(a, [1.0, 0, 0]))
        sim.setObjectMatrix(cyl, -1, np_to_matrix(np.column_stack([x, np.cross(a, x), a]), 0.5 * (start + end)))
        hits = ecm_pairs.scan(sim, probe, [[cyl, by[n]] for n in anatomy + others])
        return [(anatomy + others)[i] for i, _ in hits]

    sight = [(cam, W['cut_start']), (cam, tcp2)]
    candidates = {}
    for arm in ARMS:
        G = W['grasp_points'][arm]
        side = -1 if arm == 'psm1' else 1
        rows = []
        for P in skin:
            L = float(np.linalg.norm(G - P))
            if not DEPTH_RANGE_M[0] <= L <= DEPTH_RANGE_M[1]:
                continue
            if np.dot(P - G, anterior) < 0.04 or side * np.dot(P - W['cut_start'], W['u']) < 0.02:
                continue
            if min(np.linalg.norm(P - q) for q in fixed_ports.values()) < MIN_PORT_SPACING_M:
                continue
            a = (G - P) / L
            ang = math.degrees(math.acos(np.clip(np.dot(a, psm2_axis), -1, 1)))
            if not ANGLE_TO_PSM2_DEG[0] <= ang <= ANGLE_TO_PSM2_DEG[1]:
                continue
            end = G - 0.015 * a
            if min(seg_seg_distance(P, end, s, t) for s, t in sight) < SIGHT_CLEARANCE_M:
                continue
            rows.append(dict(port=P, axis=a, depth=L, angle_to_psm2_deg=ang,
                             move_m=float(np.linalg.norm(P - np.array(cfg[arm]['rcm_world'])))))
        # Small posture change first; deep paths (long lever arm inside the pelvis) count against a port.
        rows.sort(key=lambda r: r['move_m'] + 0.5 * abs(r['depth'] - NOMINAL_DEPTH_M))
        clear = []
        for r in rows:
            hits = path_clear(r['port'], G)
            if not hits:
                clear.append(r)
            if len(clear) >= CANDIDATES_PER_ARM:
                break
        candidates[arm] = clear
        print(f'{arm}: {len(rows)} geometric candidates, {len(clear)} with a clear path (checked in order of change)', flush=True)
    sim.removeObjects([cyl])

    # 3. Arm postures.
    urdf_limits = {}
    import xml.etree.ElementTree as ET
    for j in ET.parse(URDF).getroot().findall('joint'):
        lim = j.find('limit')
        if lim is not None:
            urdf_limits[j.get('name')] = [float(lim.get('lower')), float(lim.get('upper'))]
    by = aliases(sim)
    chosen = {}
    report = {}
    for arm in ARMS:
        c = cfg[arm]
        rcm_names = [j['name'] for j in c['joints']]
        names = [arm + '_lift_joint'] + rcm_names
        lims = {n: plan0['joint_limit_revisions'].get(n, urdf_limits[n]) for n in names}
        for j in c['joints']:
            lims[j['name']] = j['limits']
        chain = Chain(arm + '_rot_ref_respondable')
        fixed = {arm + '_joint7': c['insertion_home'], arm + '_joint8': sim.getJointPosition(by[arm + '_joint8'])}
        taskk = PoseTask(chain, names, fixed, lims, np.zeros(3), [[0, 0, 1]])
        handles = [by[n] for n in names]
        x_now = np.array([sim.getJointPosition(h) for h in handles])
        p_fk, (z_fk,) = taskk.evaluate(x_now)
        M8 = matrix_to_np(sim.getObjectMatrix(by[arm + '_joint8'], -1))
        fk_err = max(float(np.linalg.norm(p_fk - M8[:, 3])), float(np.linalg.norm(z_fk - M8[:, 2])))
        if fk_err > 1e-6:
            raise SystemExit(f'{arm}: URDF FK disagrees with the scene by {fk_err}')
        mine = [l for l in pairs if l[0].startswith(arm + '_') or l[1].startswith(arm + '_')]
        hp = [[by[a], by[b]] for a, b in mine]
        tool = c['tool_shape']

        def contacts(port):
            bad, length = [], 0.0
            for i, pts in ecm_pairs.scan(sim, probe, hp):
                pair = mine[i]
                if pair == [tool, 'Patient_Skin'] and len(pts) and np.linalg.norm(pts - port, axis=1).max() <= c['port_radius_m']:
                    continue
                bad.append(pair)
                n = len(pts) // 2
                if n:
                    length += float(np.linalg.norm(pts[0:2 * n:2] - pts[1:2 * n:2], axis=1).sum())
            return bad, length + 0.002 * len(bad)

        def set_x(x):
            sim.callScriptFunction('setJoints', probe, handles, [float(v) for v in x])

        def motion_check(x_home, port, axis):
            """RCM IK along the planned motion (shaft through the port), contacts of this arm at each end pose."""
            rcm_dist = tcp_len[arm] - (np.linalg.norm(W['grasp_points'][arm] - port) - APPROACH_GAP_M)
            tx = unit(np.cross(axis, [0, 0, 1.0]))
            ty = np.cross(axis, tx)
            home_q = dict(zip(names, x_home))
            fixed_m = {**fixed, arm + '_lift_joint': home_q[arm + '_lift_joint']}
            out = []
            moves = [(0.0, 0.0, d) for d in MOTION['axial_m']] + [(s * MOTION['lateral_deg'], q * MOTION['lateral_deg'], 0.0)
                                                                 for s in (-1, 1) for q in (-1, 1)]
            for yaw, pitch, dins in moves:
                # Small steps from home, seeding each IK from the last (as the runtime controller does).
                steps = 12
                xr = np.array([home_q[n] for n in rcm_names])
                r = 0.0
                for k in range(1, steps + 1):
                    f = k / steps
                    dk = unit(axis + math.tan(math.radians(f * yaw)) * tx + math.tan(math.radians(f * pitch)) * ty)
                    t2 = PoseTask(chain, rcm_names, {**fixed_m, arm + '_joint7': c['insertion_home'] + f * dins},
                                  lims, np.zeros(3), [[0, 0, 1]])
                    xr, r = t2.solve(xr, port - dk * (rcm_dist - f * dins), [dk], max_nfev=200)
                    if r > 1e-6:
                        break
                if r > 1e-6:
                    out.append(dict(yaw=yaw, pitch=pitch, dins=dins, ik_residual=r))
                    return False, out
                set_x(np.r_[home_q[arm + '_lift_joint'], xr])
                sim.callScriptFunction('setJoints', probe, [by[arm + '_joint7']], [float(c['insertion_home'] + dins)])
                bad, _ = contacts(port)
                sim.callScriptFunction('setJoints', probe, [by[arm + '_joint7']], [float(c['insertion_home'])])
                out.append(dict(yaw=yaw, pitch=pitch, dins=dins, contacts=bad))
                if bad:
                    set_x(x_home)
                    return False, out
            set_x(x_home)
            return True, out

        tried = []
        for cand in candidates[arm]:
            port, axis = cand['port'], cand['axis']
            tip_home = W['grasp_points'][arm] - APPROACH_GAP_M * axis
            p8 = tip_home - tcp_len[arm] * axis
            pool = []
            for k in range(args.random_ik):
                seed = x_now if k == 0 else rng.uniform(np.maximum(taskk.lo, -1.6), np.minimum(taskk.hi, 1.6))
                x, r = taskk.solve(seed, p8, [axis])
                if r < 1e-8:
                    set_x(x)
                    bad, cost = contacts(port)
                    pool.append([cost, x, bad])
            pool.sort(key=lambda row: row[0])
            ok = False
            motion_failure = None
            for cost, x, bad in pool[:3]:
                for it in range(args.local_iterations):
                    # Contacts first; once contact-free, move the joints away from their limits (motion needs room).
                    if not bad and taskk.margin(x, 1) >= MIN_JOINT_MARGIN:
                        break
                    tx_, r = taskk.solve(x + rng.normal(0, (0.05, 0.15, 0.4)[it % 3], len(x)), p8, [axis])
                    if r > 1e-8:
                        continue
                    set_x(tx_)
                    tb, tcost = contacts(port)
                    if tcost < cost or (tcost == cost == 0 and taskk.margin(tx_, 1) > taskk.margin(x, 1)):
                        cost, x, bad = tcost, tx_, tb
                if bad:
                    continue
                set_x(x)
                mok, mres = motion_check(x, port, axis)
                motion_failure = None if mok else mres[-1]
                if mok:
                    ok = True
                    chosen[arm] = dict(port=port, axis=axis, x=x, names=names, tip_home=tip_home, p8=p8,
                                       candidate=cand, motion=mres, margin=taskk.margin(x, 1))
                    break
            tried.append(dict(port=cand['port'].tolist(), move_m=cand['move_m'], ik_solutions=len(pool),
                              best_contacts=pool[0][2] if pool else None, motion_failure=motion_failure, accepted=ok))
            print(f"{arm}: port moved {cand['move_m'] * 1000:.0f} mm, depth {cand['depth'] * 1000:.0f} mm, "
                  f"{len(pool)} IK, accepted {ok} {motion_failure or ''}  t={time.time() - t0:.0f}s", flush=True)
            if ok:
                set_x(chosen[arm]['x'])       # keep it: the next arm is planned against this posture
                break
        report[arm] = tried
        if arm not in chosen:
            sim.removeObjects([probe])
            stop_simulation(client, sim)
            write_json(CONFIG / 'instruments' / 'port_plan_rejected.json', dict(candidates=report))
            raise SystemExit(f'no port found for {arm}; see config/instruments/port_plan_rejected.json')

    # Both arms together: no contact between the two re-planned arms.
    both = [l for l in pairs if any(l[0].startswith(a + '_') for a in ARMS) and any(l[1].startswith(a + '_') for a in ARMS)
            and l[0][:4] != l[1][:4]]
    cross = [both[i] for i, _ in ecm_pairs.scan(sim, probe, [[by[a], by[b]] for a, b in both])]
    sim.removeObjects([probe])
    stop_simulation(client, sim)
    if cross:
        raise SystemExit(f'the two re-planned arms touch: {cross}')
    spacing = float(np.linalg.norm(chosen['psm1']['port'] - chosen['psm3']['port']))

    write_json(OUT, {
        '_note': 'Generated by scripts/build/plan_psm_ports.py. Exploratory simulation ports, not a clinical port placement.',
        'inputs': {relative(f): sha256(f) for f in (INSTRUMENT_SCENE, ENDO_CONFIG / 'rcm_config.json', ENDO_CONFIG / 'task.json',
                                                     ENDO_CONFIG / 'collision_pairs.json', URDF,
                                                     REPO / 'models' / 'anatomy' / 'corrected_skin.npz')},
        'workspace': {k: (v.tolist() if hasattr(v, 'tolist') else {a: g.tolist() for a, g in v.items()}) for k, v in W.items()},
        'rules': dict(grid_m=GRID_M, depth_range_m=DEPTH_RANGE_M, min_port_spacing_m=MIN_PORT_SPACING_M,
                      angle_to_psm2_deg=ANGLE_TO_PSM2_DEG, sight_clearance_m=SIGHT_CLEARANCE_M, path_radius_m=PATH_RADIUS_M,
                      approach_gap_m=APPROACH_GAP_M, motion_checked=MOTION),
        'new_limits': NEW_LIMITS,
        'psm1_psm3_port_spacing_m': spacing,
        'arms': {a: dict(port_world=v['port'].tolist(), old_port_world=cfg[a]['rcm_world'],
                         port_moved_m=v['candidate']['move_m'], axis_world=v['axis'].tolist(),
                         grasp_point_world=W['grasp_points'][a].tolist(), tip_home_world=v['tip_home'].tolist(),
                         shaft_frame_world=v['p8'].tolist(), depth_port_to_grasp_m=v['candidate']['depth'],
                         angle_to_psm2_deg=v['candidate']['angle_to_psm2_deg'],
                         joint_positions=dict(zip(v['names'], map(float, v['x']))), joint_margin_normalised=v['margin'],
                         motion_results=v['motion']) for a, v in chosen.items()},
        'search': report,
        'search_seconds': round(time.time() - t0, 1),
    })
    print('wrote', relative(OUT), {a: round(v['candidate']['move_m'] * 1000) for a, v in chosen.items()}, 'mm moved', flush=True)


if __name__ == '__main__':
    main()
