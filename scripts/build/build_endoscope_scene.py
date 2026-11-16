"""Build scenes/davinci_endoscope_task.ttt from the base scene, src/endoscope and config/endoscope.

Steps (all in a CoppeliaSim instance reserved for the build; the base scene file is not modified):
  1. load a staged copy of scenes/davinci_foot_docking_rcm.ttt;
  2. apply the planned ECM docking and the PSM2 null-space re-posture (config/endoscope/ecm_plan.json);
  3. add the scope tip frame, ECM RCM markers and the 30-degree endoscope vision sensor;
  4. generate config/endoscope/rcm_config.json and collision_pairs.json;
  5. replace the controller and panel scripts with generated text (data + src/endoscope/*.lua);
  6. refuse to save if the home pose has any unexpected mesh intersection;
  7. save the scene and validation/endoscope/build_manifest.json.
"""
import argparse
import math
import time

import numpy as np

import ecm_pairs
from simtools import (BASE_SCENE, CONFIG, ENDO_CONFIG, ENDO_SRC, ENDO_VALIDATION, ENDOSCOPE_SCENE, add_arguments,
                      aliases, connect, controller_source, load_scene, matrix_to_np, np_to_matrix, panel_data,
                      panel_source, read_json, relative, save_scene, sha256, stop_simulation, write_json)


def ring(sim, alias, port, x, y, axis, radius, tube=.0006):
    verts, faces = [], []
    for i in range(48):
        a = 2 * math.pi * i / 48
        for j in range(8):
            b = 2 * math.pi * j / 8
            rr = radius + tube * math.cos(b)
            verts.extend((port + rr * math.cos(a) * x + rr * math.sin(a) * y + tube * math.sin(b) * axis).tolist())
    for i in range(48):
        for j in range(8):
            a, b = i * 8 + j, ((i + 1) % 48) * 8 + j
            c, d = ((i + 1) % 48) * 8 + (j + 1) % 8, i * 8 + (j + 1) % 8
            faces.extend([a, b, c, a, c, d])
    h = sim.createMeshShape(0, .5, verts, faces)
    sim.setObjectAlias(h, alias)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [1, .55, .04])
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
    return h


def scope_tip_world(sim, shape, axis):
    v = np.array(sim.getShapeMesh(shape)[0]).reshape(-1, 3)
    M = matrix_to_np(sim.getObjectMatrix(shape, -1))
    w = v @ M[:, :3].T + M[:, 3]
    p = w @ axis
    return w[p > p.max() - 1e-4].mean(axis=0)


def flat_color(sim, h, rgb):
    """Unlit colour so colour segmentation does not depend on scene lighting."""
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0, 0, 0])
    sim.setShapeColor(h, None, sim.colorcomponent_specular, [0, 0, 0])
    sim.setShapeColor(h, None, sim.colorcomponent_emission, list(rgb))
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)


def add_task_objects(sim, by, task):
    """Static task fixtures for the visual-servo task (kinematic objects, no contact physics)."""
    b = task['block']
    kind = sim.primitiveshape_spheroid if b.get('shape') == 'sphere' else sim.primitiveshape_cuboid
    h = sim.createPrimitiveShape(kind, [b['size_m']] * 3, 0)
    sim.setObjectAlias(h, b['alias'])
    R = np.array(b['frame_world'])
    sim.setObjectMatrix(h, -1, np_to_matrix(R, np.array(b['center_world'])))
    flat_color(sim, h, b['color'])
    t = task['target']
    c, n = np.array(t['center_world']), np.array(t['normal_world'])
    x = np.cross(n, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    h = ring(sim, t['alias'], c, x, np.cross(n, x), n, t['radius_m'], tube=t['tube_m'])
    flat_color(sim, h, t['color'])
    m = task['tip_marker']
    h = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [2 * m['radius_m']] * 3, 0)
    sim.setObjectAlias(h, m['alias'])
    tcp = matrix_to_np(sim.getObjectMatrix(by[task['arm'] + '_tcp'], -1))
    shaft = matrix_to_np(sim.getObjectMatrix(by[task['arm'] + '_joint8'], -1))[:, 2]
    sim.setObjectPosition(h, -1, (tcp[:, 3] - m['offset_from_tcp_m'] * shaft).tolist())
    sim.setObjectParent(h, sim.getObjectParent(by[task['arm'] + '_joint9']), True)
    flat_color(sim, h, m['color'])


def task_pairs(shapes, task):
    """The block must not touch anatomy, tables, other arms, or the task arm's shaft (jaws excluded)."""
    arm = task['arm']
    block = task['block']['alias']
    out = []
    for n in sorted(shapes):
        if n.startswith('Patient_') or n == 'TaskSurface' or n.startswith('LegSupport_'):
            out.append([block, n])
        elif n.endswith('_visual') and (n.startswith(('psm', 'ecm_')) and not n.startswith(arm + '_')):
            out.append([n, block])
        elif n in (arm + '_tool_main_respondable_visual', arm + '_rot_ref_respondable_visual'):
            out.append([n, block])
    return out


def classify(sim, probe, labels, hpairs, allowed):
    """Unexpected contacts; `allowed` maps tool shape -> (port, radius) for its own skin crossing."""
    bad, crossings = [], {}
    for i, pts in ecm_pairs.scan(sim, probe, hpairs):
        a, b = labels[i]
        if b == 'Patient_Skin' and a in allowed and len(pts):
            port, radius = allowed[a]
            r = float(np.linalg.norm(pts - port, axis=1).max())
            if r <= radius:
                crossings[a] = r
                continue
        bad.append(labels[i])
    return bad, crossings


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    args = p.parse_args()

    endo = read_json(ENDO_CONFIG / 'endoscope.json')
    plan = read_json(ENDO_CONFIG / 'ecm_plan.json')
    psm_cfg = read_json(CONFIG / 'rcm_config.json')
    old_pairs = read_json(CONFIG / 'collision_pairs.json')
    ecm = endo['ecm']
    stale = [k for k, v in plan['inputs'].items() if not k.endswith('endoscope.json') and sha256(CONFIG.parent / k) != v]
    if stale:
        raise SystemExit(f'ecm_plan.json is stale (inputs changed: {stale}); rerun plan_ecm_docking.py')

    client, sim = connect(args)
    load_scene(client, sim, args, BASE_SCENE, 'endoscope_build_stage.ttt')
    by = aliases(sim)

    # 1. ECM limits and docking, PSM2 null-space re-posture.
    for n, (lo, hi) in plan['ecm_joint_limits'].items():
        sim.setJointInterval(by[n], False, [lo, hi - lo])
    for n, v in {**plan['ecm_joint_positions'], **plan['co_dock_joint_positions']}.items():
        sim.setJointPosition(by[n], v)
    tip_world = np.array(plan['tip_world'])
    tipF = np.array(plan['tip_frame_world'])
    camF = np.array(plan['camera_frame_world'])
    scope = by[ecm['scope_shape']]
    err = float(np.linalg.norm(scope_tip_world(sim, scope, tipF[:, 2]) - tip_world))
    if err > 1e-6:
        raise SystemExit(f'Scene scope tip differs from the plan by {err} m')
    for c in psm_cfg:
        M = matrix_to_np(sim.getObjectMatrix(by[c['arm'] + '_joint8'], -1))
        if np.linalg.norm(M[:, 3] - c['shaft_home']) > 1e-6 or np.linalg.norm(M[:, 2] - c['axis_home']) > 1e-6:
            raise SystemExit(f"{c['arm']} instrument shaft moved by the re-posture")

    # 2. Scope tip frame, endoscope camera and ECM RCM markers.
    tip = sim.createDummy(.004)
    sim.setObjectAlias(tip, 'ecm_scope_tip')
    sim.setObjectMatrix(tip, -1, np_to_matrix(tipF, tip_world))
    sim.setObjectParent(tip, by['ecm_insertion_respondable'], True)
    cam = endo['camera']
    sensor = sim.createVisionSensor(1 + 2 + 4 + 128, [cam['resolution'][0], cam['resolution'][1], 0, 0],
                                    [cam['near_m'], cam['far_m'], math.radians(cam['view_angle_deg']), .004, 0, 0,
                                     .05, .03, .03, 0, 0])
    sim.setObjectAlias(sensor, cam['alias'])
    sim.setObjectMatrix(sensor, -1, np_to_matrix(camF, tip_world))
    sim.setObjectParent(sensor, tip, True)
    port = np.array(plan['camera_port_world'])
    for suffix in ['_RCM_reference', '_RCM_projection']:
        h = sim.createDummy(.006)
        sim.setObjectAlias(h, 'ecm' + suffix)
        sim.setObjectPosition(h, -1, port.tolist())
    ring(sim, 'ecm_RCM_fixed_ring', port, tipF[:, 0], tipF[:, 1], tipF[:, 2], .014)
    task_file = ENDO_CONFIG / 'task.json'
    task = read_json(task_file) if task_file.exists() else None
    if task:
        add_task_objects(sim, by, task)
    by = aliases(sim)

    # 3. Generated controller configuration (PSM entries keep the base-scene RCM data).
    pm = endo['psm']
    rcm_config = []
    for c in psm_cfg:
        e = dict(c)
        arm = c['arm']
        e.update(insertion_joint=arm + '_joint7', shaft_frame=arm + '_joint8',
                 projection=arm + '_RCM_projection', tool_shape=arm + '_tool_main_respondable_visual',
                 port_radius_m=pm['port_radius_m'], constrain_roll=False, auto_demo=True,
                 limits={**pm['motion'], **pm.get('limit_overrides', {}).get(arm, {})}, demo=dict(pm['demo']),
                 grip_joints=[arm + '_' + j for j in pm['grip_joints']], tcp_frame=arm + '_' + pm['tcp_frame'])
        rcm_config.append(e)
    M = matrix_to_np(sim.getObjectMatrix(tip, -1))
    ih, mo = ecm['insertion_home'], ecm['motion']
    rcm_config.append(dict(
        arm='ecm', joints=[dict(name=n, type='revolute', limits=plan['ecm_joint_limits'][n]) for n in ecm['rcm_joints']],
        rcm_world=port.tolist(), axis_home=M[:, 2].tolist(), transverse_x=M[:, 0].tolist(), transverse_y=M[:, 1].tolist(),
        shaft_home=M[:, 3].tolist(), insertion_home=ih, rcm_distance_home=float(np.dot(port - M[:, 3], M[:, 2])),
        insertion_joint=ecm['insertion_joint'], shaft_frame='ecm_scope_tip', projection='ecm_RCM_projection',
        tool_shape=ecm['scope_shape'], port_radius_m=ecm['port_radius_m'], constrain_roll=True, auto_demo=False,
        demo=dict(yaw_deg=0.0, pitch_deg=0.0, insertion_amp_m=0.0),
        limits=dict(yaw_deg=mo['yaw_deg'], pitch_deg=mo['pitch_deg'], insertion_min=ih - mo['insertion_delta_m'],
                    insertion_max=ih + mo['insertion_delta_m'])))

    # 4. Collision pairs: base-scene pairs plus every pair involving a moving ECM shape.
    shapes = [n for n, h in by.items() if sim.getObjectType(h) == sim.sceneobject_shape]
    new_labels, new_cats = ecm_pairs.ecm_pair_candidates(shapes)
    designed = plan['designed_zero_pose_contacts_excluded']
    seen, labels, categories = set(), [], []
    extra = [(l, 'task_block') for l in task_pairs(shapes, task)] if task else []
    for l, c in list(zip(old_pairs['labels'], old_pairs['categories'])) + list(zip(new_labels, new_cats)) + extra:
        key = frozenset(l)
        if key in seen or l in designed:
            continue
        seen.add(key)
        labels.append(l)
        categories.append(c)
    missing = sorted({n for l in labels for n in l if n not in by})
    if missing:
        raise SystemExit(f'collision pairs reference missing shapes: {missing}')

    # 5. Home-pose collision gate.
    probe = ecm_pairs.create_probe(sim)
    allowed = {a['tool_shape']: (np.array(a['rcm_world']), a['port_radius_m']) for a in rcm_config}
    bad, crossings = classify(sim, probe, labels, [[by[a], by[b]] for a, b in labels], allowed)
    sim.removeObjects([probe])
    if bad:
        raise SystemExit(f'Unexpected contacts at the home pose, scene not saved: {bad}')

    # 6. Scripts: generated data + committed Lua sources.
    controller = controller_source(rcm_config, labels)
    panel = panel_source(panel_data(endo, rcm_config))
    sim.removeObjects([by['RCMController'], by['RCMControlPanel']])
    h = sim.createScript(sim.scripttype_simulation, controller, 0, 'lua')
    sim.setObjectAlias(h, 'RCMController')
    h = sim.createScript(sim.scripttype_customization, panel, 0, 'lua')
    sim.setObjectAlias(h, 'RCMControlPanel')
    time.sleep(.3)

    write_json(ENDO_CONFIG / 'rcm_config.json', rcm_config)
    write_json(ENDO_CONFIG / 'collision_pairs.json', {
        '_note': 'Generated by scripts/build/build_endoscope_scene.py: base-scene pairs plus all pairs with a moving ECM shape.',
        'labels': labels, 'categories': categories, 'excluded_designed_contacts': designed})
    save_scene(sim, args, ENDOSCOPE_SCENE)
    stop_simulation(client, sim)

    counts = {}
    for c in categories:
        counts[c] = counts.get(c, 0) + 1
    runtime = {a['arm']: sum(1 for l in labels if l[0].startswith(a['arm'] + '_') or l[1].startswith(a['arm'] + '_'))
               for a in rcm_config}
    inputs = [BASE_SCENE, ENDO_CONFIG / 'endoscope.json', ENDO_CONFIG / 'ecm_plan.json', CONFIG / 'rcm_config.json',
              CONFIG / 'collision_pairs.json', ENDO_SRC / 'rcm_controller.lua', ENDO_SRC / 'rcm_panel.lua']
    if task:
        inputs.append(task_file)
    outputs = [ENDOSCOPE_SCENE, ENDO_CONFIG / 'rcm_config.json', ENDO_CONFIG / 'collision_pairs.json']
    write_json(ENDO_VALIDATION / 'build_manifest.json', {
        'built_at_unix': int(time.time()),
        'coppeliasim_version': sim.getInt32Param(sim.intparam_program_full_version),
        'inputs': {relative(f): sha256(f) for f in inputs},
        'outputs': {relative(f): sha256(f) for f in outputs},
        'scope_tip_plan_vs_scene_m': err,
        'collision_pairs': len(labels), 'pairs_by_category': counts, 'runtime_pairs_per_arm': runtime,
        'home_unexpected_contacts': bad, 'home_port_crossing_radius_m': crossings,
    })
    print('saved', relative(ENDOSCOPE_SCENE), 'pairs', len(labels), 'runtime pairs', runtime, flush=True)


if __name__ == '__main__':
    main()
