"""Build scenes/davinci_si_instruments.ttt: the endoscope scene with da Vinci Si instruments per config/instruments/roles.json.

Steps (in a CoppeliaSim instance reserved for the build; scenes/davinci_endoscope_task.ttt is not modified):
  1. load a staged copy of the endoscope scene;
  2. capture the original needle-driver geometry as models/meshes/instrument/legacy_lnd_reference.npz and
     measure how well the dVRK Si Large Needle Driver (420006) fits the scene frames (registration check);
  3. per PSM: wrist and jaw meshes from dvrk_model, jaw joints moved to the instrument's yaw axis / hinge,
     tool-tip frame moved to the instrument's tool tip, wrist and jaw limits from the descriptor;
  4. identify every PSM instrument from the new geometry (instrument_guard) and refuse to save unless it is
     the requested model and the roles in roles.json are satisfied;
  5. collision gate at the home pose with jaws closed and open (same pairs as the endoscope scene);
  6. controller = generated data + src/endoscope/rcm_controller.lua + src/instruments/capability_guard.lua;
  7. save the scene, reload it, run the controller (demo cycle, grasp refusal) and write
     validation/instruments/install_manifest.json.
"""
import argparse
import time

import numpy as np

import build_endoscope_scene as bes
import ecm_pairs
import instrument_guard as guard
import instruments as inst
from simtools import (ENDO_CONFIG, ENDO_SRC, ENDOSCOPE_SCENE, INSTRUMENT_SCENE, INSTRUMENT_SRC, INSTRUMENT_VALIDATION, TISSUE_SRC,
                      REPO, add_arguments, aliases, connect, load_scene, lua, np_to_matrix, read_json, read_text,
                      relative, save_scene, sha256, stop_simulation)

LEGACY_NPZ = REPO / 'models' / 'meshes' / 'instrument' / 'legacy_lnd_reference.npz'
PORT_PLAN = REPO / 'config' / 'instruments' / 'port_plan.json'
RCM_OUT = REPO / 'config' / 'instruments' / 'rcm_config.json'
JAW_OPEN_PER_JOINT = 0.5           # rad on each jaw joint = 1.0 rad jaw opening (the panel's "Open jaws", setGrip(0.5))
ROLL_STEP_RAD = np.radians(5.0)
INSTRUMENT_PARTS = ('rot_ref_respondable_visual', 'wrist_ref_respondable_visual', 'yaw_ref1_respondable_visual',
                    'yaw_ref2_respondable_visual', 'tool_tip_visual')


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return inst.transform(R.T, -R.T @ t)


def shape_mesh_world(sim, h):
    V, F = sim.getShapeMesh(h)[:2]
    V = np.asarray(V, dtype=float).reshape(-1, 3)
    return inst.apply(guard.world(sim, h), V), np.asarray(F, dtype=np.int64).reshape(-1, 3)


def capture_legacy(sim, by, arm):
    """Original instrument geometry of `arm` in the dVRK frames (P for the wrist, H = Y for the jaws)."""
    T_WP, _ = guard.wrist_frame(sim, by, arm)
    T_PY = inst.geometry('420006')['T_PY']
    T_WY = T_WP @ T_PY
    parts = {}
    for part, n, T in (('wrist', 9, T_WP), ('jaw_plus', 10, T_WY), ('jaw_minus', 11, T_WY)):
        meshes = [shape_mesh_world(sim, h) for h in guard.shapes_under(sim, by[f'{arm}_joint{n}'])]
        V, F = inst.merge(meshes)
        parts[part] = (inst.apply(inv(T), V), F)
    tip = inst.apply(inv(T_WY), np.array(sim.getObjectPosition(by[f'{arm}_tcp'], -1))[None])[0]
    return dict(T_PY=T_PY, hinge_Y=np.zeros(3), tip_Y=tip, **parts)


def copy_appearance(sim, src, dst):
    for comp in (sim.colorcomponent_ambient_diffuse, sim.colorcomponent_specular, sim.colorcomponent_emission):
        res, rgb = sim.getShapeColor(src, None, comp)
        if res > 0:
            sim.setShapeColor(dst, None, comp, rgb)
    for p in (sim.shapeintparam_static, sim.shapeintparam_respondable):
        sim.setObjectInt32Param(dst, p, sim.getObjectInt32Param(src, p))
    sim.setObjectInt32Param(dst, sim.objintparam_visibility_layer,
                            sim.getObjectInt32Param(src, sim.objintparam_visibility_layer))
    try:
        sim.setObjectSpecialProperty(dst, sim.getObjectSpecialProperty(src))
    except Exception:
        pass


def replace_shape(sim, old, Vw, F):
    """New mesh shape with the old shape's frame, parent, children, appearance and alias."""
    pose = sim.getObjectPose(old, -1)
    parent = sim.getObjectParent(old)
    alias = sim.getObjectAlias(old)
    new = sim.createMeshShape(0, 0.35, Vw.reshape(-1).tolist(), F.reshape(-1).astype(int).tolist())
    sim.relocateShapeFrame(new, pose)
    sim.setObjectParent(new, parent, True)
    copy_appearance(sim, old, new)
    children = [c for c in (sim.getObjectChild(old, i) for i in range(64)) if c != -1]
    for c in children:
        sim.setObjectParent(c, new, True)
    sim.removeObjects([old])
    sim.setObjectAlias(new, alias)
    return new


def install(sim, by, arm, model):
    import trimesh
    g, d = inst.geometry(model), inst.descriptor(model)
    k = d['kinematics']
    T_WP, _ = guard.wrist_frame(sim, by, arm)
    T_WY = T_WP @ g['T_PY']
    T_WH = T_WY @ inst.transform(t=g['hinge_Y'])

    def interval(name, lim):
        sim.setJointInterval(by[f'{arm}_{name}'], False, [lim[0], lim[1] - lim[0]])
    interval('joint8', k['roll_limits_rad'])
    interval('joint9', k['wrist_pitch_limits_rad'])
    jaw = [k['jaw_limits_rad'][0] / 2, k['jaw_limits_rad'][1] / 2]
    interval('joint10', jaw)
    interval('joint11', jaw)

    for j in ('joint10', 'joint11'):
        sim.setObjectPosition(by[f'{arm}_{j}'], -1, T_WH[:3, 3].tolist())
    tip = inst.apply(T_WY, g['tip_Y'][None])[0]
    for n in ('tcp', 'tool_tip_fixed_joint'):
        sim.setObjectPosition(by[f'{arm}_{n}'], -1, tip.tolist())

    for base, part, T in (('wrist_ref', 'wrist', T_WP), ('yaw_ref1', 'jaw_plus', T_WH), ('yaw_ref2', 'jaw_minus', T_WH)):
        V, F = g[part]
        Vw = inst.apply(T, V)
        hull = trimesh.Trimesh(Vw, F).convex_hull
        replace_shape(sim, by[f'{arm}_{base}_respondable_respondable'], np.asarray(hull.vertices), np.asarray(hull.faces))
        by.update(aliases(sim))
        replace_shape(sim, by[f'{arm}_{base}_respondable_visual'], Vw, F)
        by.update(aliases(sim))

    slide = None
    if g['hinge_slope_m_per_rad']:
        # Moving jaw hinge (ProGrasp): a prismatic slide along the distal axis carries both jaw joints.
        # setGrip drives it with the dvrk_model law (capability_guard.lua); 0 = closed jaws.
        h = sim.createJoint(sim.joint_prismatic, sim.jointmode_kinematic, 0, [0.002, 0.0006])
        slide = f'{arm}_jaw_hinge_slide'
        sim.setObjectAlias(h, slide)
        z, x = T_WY[:3, 1], T_WY[:3, 2]          # slide along the distal axis; x along the jaw axis
        sim.setObjectMatrix(h, -1, np_to_matrix(np.column_stack([x, np.cross(z, x), z]), T_WH[:3, 3]))
        sim.setObjectParent(h, by[f'{arm}_wrist_ref_respondable_respondable'], True)
        travel = g['hinge_slope_m_per_rad'] * g['hinge_jaw_range_rad'][1]
        sim.setJointInterval(h, False, [min(travel, 0.0) - 1e-4, abs(travel) + 2e-4])
        sim.setJointPosition(h, 0.0)
        for j in ('joint10', 'joint11'):
            sim.setObjectParent(by[f'{arm}_{j}'], h, True)
        by.update(aliases(sim))
    return dict(model=model, yaw_axis_world=T_WY[:3, 3], hinge_world=T_WH[:3, 3], tool_tip_world=tip,
                tool_tip_dvrk_world=inst.apply(T_WY, g['tip_dvrk_Y'][None])[0],
                tool_tip_minus_dvrk_m=float(np.linalg.norm(g['tip_Y'] - g['tip_dvrk_Y'])),
                jaw_joint_limits_rad=jaw, hinge_slide=slide, hinge_slope_m_per_rad=g['hinge_slope_m_per_rad'],
                hinge_jaw_range_rad=g['hinge_jaw_range_rad'])


def set_grip(sim, by, arm, q, info):
    """Same jaw kinematics as setGrip in capability_guard.lua (both jaw joints = q, hinge slide from the law)."""
    for n in (10, 11):
        sim.setJointPosition(by[f'{arm}_joint{n}'], q)
    if info.get('hinge_slide'):
        lo, hi = info['hinge_jaw_range_rad']
        sim.setJointPosition(by[info['hinge_slide']], info['hinge_slope_m_per_rad'] * min(max(2 * q, lo), hi))


def choose_roll(sim, by, arm, info, labels, allowed, open_q):
    """Instrument roll (joint8) closest to the current one at which the jaws open without contact at home.

    Roll turns the instrument about its own shaft: the shaft, the RCM point and the tool tip stay where they are,
    only the plane in which the jaws open changes."""
    j8 = by[f'{arm}_joint8']
    q0 = sim.getJointPosition(j8)
    lo, hi = inst.descriptor(info['model'])['kinematics']['roll_limits_rad']
    mine = [l for l in labels if any(n.startswith(arm + '_') and n[len(arm) + 1:] in INSTRUMENT_PARTS for n in l)]
    probe = ecm_pairs.create_probe(sim)
    tried = []
    try:
        for k in range(0, 73):
            q = q0 + (k + 1) // 2 * ROLL_STEP_RAD * (1 if k % 2 else -1)
            if not lo <= q <= hi:
                continue
            sim.setJointPosition(j8, q)
            results = []
            for grip in (0.0, open_q):
                set_grip(sim, by, arm, grip, info)
                bad, _ = bes.classify(sim, probe, mine, [[by[a], by[b]] for a, b in mine], allowed)
                results.append(bad)
            set_grip(sim, by, arm, 0.0, info)
            tried.append(dict(roll_rad=q, closed=results[0], open=results[1]))
            if not results[0] and not results[1]:
                return dict(roll_rad=q, roll_change_deg=float(np.degrees(q - q0)), candidates_tried=len(tried),
                            pairs_checked=len(mine), first_blocked=tried[0] if len(tried) > 1 else None)
        sim.setJointPosition(j8, q0)
        raise SystemExit(f'{arm}: no instrument roll opens the jaws without contact at home: {tried[:3]}')
    finally:
        sim.removeObjects([probe])


def apply_port_plan(sim, by, rcm, plan):
    """Set the planned arm postures, move the RCM markers and rings, and return the updated RCM configuration."""
    out = []
    for c in rcm:
        arm = c['arm']
        if arm not in plan['arms']:
            out.append(c)
            continue
        a = plan['arms'][arm]
        for n, v in a['joint_positions'].items():
            sim.setJointPosition(by[n], v)
        M = guard.world(sim, by[c['shaft_frame']])
        err = max(float(np.linalg.norm(M[:3, 3] - a['shaft_frame_world'])), float(np.linalg.norm(M[:3, 2] - a['axis_world'])))
        if err > 1e-6:
            raise SystemExit(f'{arm}: planned shaft pose not reproduced in the scene ({err})')
        port = np.array(a['port_world'])
        for suffix in ('_RCM_reference', '_RCM_projection'):
            sim.setObjectPosition(by[arm + suffix], -1, port.tolist())
        sim.removeObjects([by[arm + '_RCM_fixed_ring']])
        bes.ring(sim, arm + '_RCM_fixed_ring', port, M[:3, 0], M[:3, 1], M[:3, 2], .008)
        e = dict(c)
        e.update(rcm_world=port.tolist(), axis_home=M[:3, 2].tolist(), transverse_x=M[:3, 0].tolist(),
                 transverse_y=M[:3, 1].tolist(), shaft_home=M[:3, 3].tolist(),
                 rcm_distance_home=float(np.dot(port - M[:3, 3], M[:3, 2])), limits=dict(plan['new_limits']),
                 port_source='config/instruments/port_plan.json')
        out.append(e)
    return out


def tcp_length(sim, by, arm):
    """Axial distance from the instrument roll frame (joint8) to the tool tip; replaces the old 0.4498 constant."""
    j8 = guard.world(sim, by[f'{arm}_joint8'])
    tcp = np.array(sim.getObjectPosition(by[f'{arm}_tcp'], -1))
    return float(np.dot(tcp - j8[:3, 3], j8[:3, 2]))


def jaw_tips(sim, by, arm, model):
    """World positions of the distal-most points of the two jaws (along the distal direction)."""
    T_WP, _ = guard.wrist_frame(sim, by, arm)
    distal = T_WP[:3, 0]
    out = []
    for n in (10, 11):
        P = np.vstack([shape_mesh_world(sim, h)[0] for h in guard.shapes_under(sim, by[f'{arm}_joint{n}'])])
        out.append(P[np.argmax(P @ distal)])
    return out


def set_jaws(sim, by, arms, q):
    for a in arms:
        for n in (10, 11):
            sim.setJointPosition(by[f'{a}_joint{n}'], q)


def gate(sim, by, labels, allowed):
    probe = ecm_pairs.create_probe(sim)
    bad, crossings = bes.classify(sim, probe, labels, [[by[a], by[b]] for a, b in labels], allowed)
    sim.removeObjects([probe])
    return bad, crossings


def runtime_check(client, sim, args):
    """Reload the saved scene, run one 24 s demo cycle and try a grasp with every PSM."""
    load_scene(client, sim, args, INSTRUMENT_SCENE, 'instrument_runtime_check.ttt')
    by = aliases(sim)
    ctrl = by['RCMController']
    sim.setStepping(True)
    sim.setBoolParam(sim.boolparam_realtime_simulation, False)
    sim.startSimulation()
    sim.step()
    sim.callScriptFunction('homeRCM', ctrl)
    dt = sim.getSimulationTimeStep()
    for _ in range(int(round(24.0 / dt)) + 1):
        sim.step()
    stats = sim.callScriptFunction('getRCMStats', ctrl)
    sim.callScriptFunction('homeRCM', ctrl)
    for _ in range(10):
        sim.step()
    grasps = {}
    for i, s in enumerate(stats, 1):
        if s['arm'] == 'ecm':
            continue
        info = sim.callScriptFunction('getInstrument', ctrl, i)
        ok = bool(sim.callScriptFunction('grasp', ctrl, i, 'TaskBead'))
        if ok:
            sim.callScriptFunction('release', ctrl, i)
        opened = bool(sim.callScriptFunction('setGrip', ctrl, i, JAW_OPEN_PER_JOINT))
        slide = by.get(f"{s['arm']}_jaw_hinge_slide")
        slide_open = sim.getJointPosition(slide) if slide is not None else None
        closed = bool(sim.callScriptFunction('setGrip', ctrl, i, 0.0))
        grasps[s['arm']] = dict(model=info['model'], grasp_accepted=ok, open_accepted=opened, close_accepted=closed,
                                hinge_slide_open_m=slide_open,
                                hinge_slide_closed_m=sim.getJointPosition(slide) if slide is not None else None,
                                capability_refusals=sim.callScriptFunction('getInstrument', ctrl, i)['capability_refusals'])
    stop_simulation(client, sim)
    return dict(steps=int(round(24.0 / dt)) + 1, time_step_s=dt,
                arms={s['arm']: dict(max_drift_m=s['max_drift_m'], failures=s['failures'],
                                     collision_blocks=s['collision_blocks']) for s in stats}, grasp=grasps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    args = p.parse_args()
    role_cfg = inst.roles()
    rcm = read_json(ENDO_CONFIG / 'rcm_config.json')
    pairs = read_json(ENDO_CONFIG / 'collision_pairs.json')
    labels = pairs['labels']
    allowed = {a['tool_shape']: (np.array(a['rcm_world']), a['port_radius_m']) for a in rcm}
    psms = [a for a in role_cfg['arms'] if a != 'ecm']

    client, sim = connect(args)
    load_scene(client, sim, args, ENDOSCOPE_SCENE, 'instrument_build_stage.ttt')
    by = aliases(sim)
    set_jaws(sim, by, psms, 0.0)

    # 2. Original geometry and frame registration check.
    inst.save_reference(LEGACY_NPZ, capture_legacy(sim, by, 'psm2'))
    guard._GEOMETRY.clear()
    before = {a: guard.identify_arm(sim, by, a) for a in psms}
    tcp_before = {a: tcp_length(sim, by, a) for a in psms}

    # 2b. Re-planned ports (config/instruments/port_plan.json, scripts/build/plan_psm_ports.py).
    port_plan = read_json(PORT_PLAN) if PORT_PLAN.exists() else None
    if port_plan:
        rcm = apply_port_plan(sim, by, rcm, port_plan)
        by = aliases(sim)
        allowed = {a['tool_shape']: (np.array(a['rcm_world']), a['port_radius_m']) for a in rcm}

    # 3. Install.
    installed = {a: install(sim, by, a, role_cfg['arms'][a]['instrument']) for a in psms}
    by = aliases(sim)
    # 3b. Instrument roll at which each PSM can open its jaws at home.
    rolls = {a: choose_roll(sim, by, a, installed[a], labels, allowed, JAW_OPEN_PER_JOINT) for a in psms}

    # 4. Identification and role check from geometry only.
    ok, roles_report = guard.check(sim, by, rcm, role_cfg)
    wrong = {a: (role_cfg['arms'][a]['instrument'], roles_report[a]['identified']) for a in psms
             if roles_report[a]['identified'] != role_cfg['arms'][a]['instrument']}
    if wrong or not ok:
        raise SystemExit(f'instrument check failed, scene not saved: wrong={wrong} roles_ok={ok}')

    # 5. Collision gate, jaws closed and open; jaw-opening geometry.
    closed_tips = {a: jaw_tips(sim, by, a, installed[a]['model']) for a in psms}
    bad_closed, crossings = gate(sim, by, labels, allowed)
    for a in psms:
        set_grip(sim, by, a, JAW_OPEN_PER_JOINT, installed[a])
    open_tips = {a: jaw_tips(sim, by, a, installed[a]['model']) for a in psms}
    bad_open, _ = gate(sim, by, labels, allowed)
    for a in psms:
        set_grip(sim, by, a, 0.0, installed[a])
    if bad_closed or bad_open:
        raise SystemExit(f'Unexpected contacts at the home pose, scene not saved: closed={bad_closed} open={bad_open}')
    opening = {a: dict(closed_tip_gap_m=float(np.linalg.norm(closed_tips[a][0] - closed_tips[a][1])),
                       open_tip_gap_m=float(np.linalg.norm(open_tips[a][0] - open_tips[a][1]))) for a in psms}
    if any(o['open_tip_gap_m'] <= o['closed_tip_gap_m'] + 1e-3 for o in opening.values()):
        raise SystemExit(f'jaws do not open with positive joint angles: {opening}')

    # 6. Controller with the capability guard; INSTRUMENTS comes from the geometry identification.
    table = {}
    for a in psms:
        e = dict(model=roles_report[a]['identified'], name=inst.descriptor(roles_report[a]['identified'])['name'],
                 capabilities=roles_report[a]['capabilities'])
        if installed[a]['hinge_slide']:
            e.update(hinge_slide=installed[a]['hinge_slide'], hinge_slope_m_per_rad=installed[a]['hinge_slope_m_per_rad'],
                     hinge_jaw_range_rad=installed[a]['hinge_jaw_range_rad'])
        table[a] = e
    table['ecm'] = dict(model='endoscope_camera', name='30 deg endoscope camera', capabilities=roles_report['ecm']['capabilities'])
    controller = ('CONFIG=' + lua(rcm) + '\nCOLLISION_PAIRS=' + lua(labels) + '\nINSTRUMENTS=' + lua(table) + '\n'
                  + read_text(ENDO_SRC / 'rcm_controller.lua') + '\n' + read_text(INSTRUMENT_SRC / 'capability_guard.lua')
                  + '\n' + read_text(TISSUE_SRC / 'tissue.lua'))
    sim.removeObjects([by['RCMController']])
    h = sim.createScript(sim.scripttype_simulation, controller, 0, 'lua')
    sim.setObjectAlias(h, 'RCMController')
    time.sleep(.3)
    tcp_after = {a: tcp_length(sim, by, a) for a in psms}
    save_scene(sim, args, INSTRUMENT_SCENE)
    from simtools import write_json as _wj
    _wj(RCM_OUT, rcm)          # the controller CONFIG of this scene; scripts/task reads it

    # 7. Runtime check on the saved file.
    runtime = runtime_check(client, sim, args)
    ok_runtime = all(s['failures'] == 0 and s['collision_blocks'] == 0 for s in runtime['arms'].values()) and \
        all(g['grasp_accepted'] == ('grasp' in table[a]['capabilities']) and g['open_accepted'] and g['close_accepted']
            for a, g in runtime['grasp'].items())

    def clean(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    inputs = [ENDOSCOPE_SCENE, ENDO_CONFIG / 'rcm_config.json', ENDO_CONFIG / 'collision_pairs.json',
              inst.INSTRUMENT_CONFIG / 'roles.json', ENDO_SRC / 'rcm_controller.lua', INSTRUMENT_SRC / 'capability_guard.lua', TISSUE_SRC / 'tissue.lua',
              inst.DVRK / 'SOURCE.json'] + [inst.INSTRUMENT_CONFIG / f'{m}.json' for m in guard.MODELS] + ([PORT_PLAN] if port_plan else [])
    from simtools import write_json
    write_json(INSTRUMENT_VALIDATION / 'install_manifest.json', {
        'built_at_unix': int(time.time()),
        'coppeliasim_version': sim.getInt32Param(sim.intparam_program_full_version),
        'inputs': {relative(f): sha256(f) for f in inputs},
        'outputs': {relative(f): sha256(f) for f in (INSTRUMENT_SCENE, LEGACY_NPZ, RCM_OUT)},
        'ports': {c['arm']: dict(rcm_world=c['rcm_world'], limits=c['limits'], source=c.get('port_source', 'config/endoscope/rcm_config.json'))
                  for c in rcm},
        'registration_before_install': {a: dict(closest=b['closest'], scores_m=b['scores_m'], frame=b['frame'])
                                        for a, b in before.items()},
        'installed': {a: clean(v) for a, v in installed.items()},
        'identification_after_install': {a: dict(identified=roles_report[a]['identified'],
                                                 closest_worst_part_m=roles_report[a]['identification']['closest_worst_part_m'],
                                                 margin_to_second_m=roles_report[a]['identification']['margin_to_second_m'])
                                         for a in psms},
        'roles': {a: dict(role=e['role'], requires=e['requires'], capabilities=e['capabilities'], ok=e['ok'])
                  for a, e in roles_report.items()},
        'tcp_length_m': {'before': tcp_before, 'after': tcp_after},
        'instrument_roll': rolls,
        'jaw_opening': opening, 'jaw_open_per_joint_rad': JAW_OPEN_PER_JOINT,
        'home_unexpected_contacts': {'closed': bad_closed, 'open': bad_open},
        'home_port_crossing_radius_m': crossings,
        'collision_pairs': len(labels),
        'runtime_check': runtime, 'runtime_ok': ok_runtime,
    })
    print('saved', relative(INSTRUMENT_SCENE), 'roles ok', ok, 'runtime ok', ok_runtime, flush=True)
    print('tcp length before', tcp_before, 'after', tcp_after)
    print('opening', opening)
    print('runtime', runtime)
    if not ok_runtime:
        raise SystemExit('runtime check failed (see install_manifest.json)')


if __name__ == '__main__':
    main()
