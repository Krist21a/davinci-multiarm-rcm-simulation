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
from simtools import (ENDO_CONFIG, ENDO_SRC, ENDOSCOPE_SCENE, INSTRUMENT_SCENE, INSTRUMENT_SRC, INSTRUMENT_VALIDATION,
                      REPO, add_arguments, aliases, connect, load_scene, lua, matrix_to_np, read_json, read_text,
                      relative, save_scene, sha256, stop_simulation)

LEGACY_NPZ = REPO / 'models' / 'meshes' / 'instrument' / 'legacy_lnd_reference.npz'
JAW_OPEN_PER_JOINT = 0.25          # rad on each jaw joint = 0.5 rad jaw opening (the panel's "Open jaws")


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
    return dict(model=model, yaw_axis_world=T_WY[:3, 3], hinge_world=T_WH[:3, 3], tool_tip_world=tip,
                jaw_joint_limits_rad=jaw)


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
    sim.callScriptFunction('hold', ctrl)
    grasps = {}
    for i, s in enumerate(stats, 1):
        if s['arm'] == 'ecm':
            continue
        info = sim.callScriptFunction('getInstrument', ctrl, i)
        ok = bool(sim.callScriptFunction('grasp', ctrl, i, 'TaskBead'))
        if ok:
            sim.callScriptFunction('release', ctrl, i)
        grasps[s['arm']] = dict(model=info['model'], grasp_accepted=ok,
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

    # 3. Install.
    installed = {a: install(sim, by, a, role_cfg['arms'][a]['instrument']) for a in psms}
    by = aliases(sim)

    # 4. Identification and role check from geometry only.
    ok, roles_report = guard.check(sim, by, rcm, role_cfg)
    wrong = {a: (role_cfg['arms'][a]['instrument'], roles_report[a]['identified']) for a in psms
             if roles_report[a]['identified'] != role_cfg['arms'][a]['instrument']}
    if wrong or not ok:
        raise SystemExit(f'instrument check failed, scene not saved: wrong={wrong} roles_ok={ok}')

    # 5. Collision gate, jaws closed and open; jaw-opening geometry.
    closed_tips = {a: jaw_tips(sim, by, a, installed[a]['model']) for a in psms}
    bad_closed, crossings = gate(sim, by, labels, allowed)
    set_jaws(sim, by, psms, JAW_OPEN_PER_JOINT)
    open_tips = {a: jaw_tips(sim, by, a, installed[a]['model']) for a in psms}
    bad_open, _ = gate(sim, by, labels, allowed)
    set_jaws(sim, by, psms, 0.0)
    # Closed jaws must be contact-free at home. Open jaws are only reported: at runtime setGrip already refuses
    # (and rolls back) an opening that would collide, so an arm that cannot open at home must move first.
    if bad_closed:
        raise SystemExit(f'Unexpected contacts at the home pose (jaws closed), scene not saved: {bad_closed}')
    opening = {a: dict(closed_tip_gap_m=float(np.linalg.norm(closed_tips[a][0] - closed_tips[a][1])),
                       open_tip_gap_m=float(np.linalg.norm(open_tips[a][0] - open_tips[a][1]))) for a in psms}
    if any(o['open_tip_gap_m'] <= o['closed_tip_gap_m'] + 1e-3 for o in opening.values()):
        raise SystemExit(f'jaws do not open with positive joint angles: {opening}')

    # 6. Controller with the capability guard; INSTRUMENTS comes from the geometry identification.
    table = {a: dict(model=roles_report[a]['identified'], name=inst.descriptor(roles_report[a]['identified'])['name'],
                     capabilities=roles_report[a]['capabilities']) for a in psms}
    table['ecm'] = dict(model='endoscope_camera', name='30 deg endoscope camera', capabilities=roles_report['ecm']['capabilities'])
    controller = ('CONFIG=' + lua(rcm) + '\nCOLLISION_PAIRS=' + lua(labels) + '\nINSTRUMENTS=' + lua(table) + '\n'
                  + read_text(ENDO_SRC / 'rcm_controller.lua') + '\n' + read_text(INSTRUMENT_SRC / 'capability_guard.lua'))
    sim.removeObjects([by['RCMController']])
    h = sim.createScript(sim.scripttype_simulation, controller, 0, 'lua')
    sim.setObjectAlias(h, 'RCMController')
    time.sleep(.3)
    tcp_after = {a: tcp_length(sim, by, a) for a in psms}
    save_scene(sim, args, INSTRUMENT_SCENE)

    # 7. Runtime check on the saved file.
    runtime = runtime_check(client, sim, args)
    ok_runtime = all(s['failures'] == 0 and s['collision_blocks'] == 0 for s in runtime['arms'].values()) and \
        all(g['grasp_accepted'] == ('grasp' in table[a]['capabilities']) for a, g in runtime['grasp'].items())

    def clean(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    inputs = [ENDOSCOPE_SCENE, ENDO_CONFIG / 'rcm_config.json', ENDO_CONFIG / 'collision_pairs.json',
              inst.INSTRUMENT_CONFIG / 'roles.json', ENDO_SRC / 'rcm_controller.lua', INSTRUMENT_SRC / 'capability_guard.lua',
              inst.DVRK / 'SOURCE.json'] + [inst.INSTRUMENT_CONFIG / f'{m}.json' for m in guard.MODELS]
    from simtools import write_json
    write_json(INSTRUMENT_VALIDATION / 'install_manifest.json', {
        'built_at_unix': int(time.time()),
        'coppeliasim_version': sim.getInt32Param(sim.intparam_program_full_version),
        'inputs': {relative(f): sha256(f) for f in inputs},
        'outputs': {relative(f): sha256(f) for f in (INSTRUMENT_SCENE, LEGACY_NPZ)},
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
