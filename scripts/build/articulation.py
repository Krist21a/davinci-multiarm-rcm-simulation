"""Migrate a stopped Si instrument scene to explicit roll/pitch/yaw + dependent jaw leaves.

This is geometry-model kinematics, not actuator/cable dynamics or manufacturer calibration.
Existing joint8/9 remain roll/pitch. New wrist_yaw is upstream of both jaws, the ProGrasp
hinge slide and the fixed TCP. setGrip retains its historical half-opening-angle convention.
"""
import math

import numpy as np

import instruments as inst
from simtools import aliases, lua, matrix_to_np, read_text, ENDO_SRC, INSTRUMENT_SRC, TISSUE_SRC


def jaw_coordinates(yaw, opening):
    """Equivalent leaf angles about a common +yaw axis (not two extra independent DOFs)."""
    if not all(math.isfinite(v) for v in (yaw, opening)):
        raise ValueError('yaw/opening must be finite radians')
    return yaw + opening / 2, yaw - opening / 2


def append_yaw_collision_pairs(labels, arms):
    """Split a formerly merged housing without dropping its external collision coverage."""
    result = [list(pair) for pair in labels]
    seen = {tuple(sorted(pair)) for pair in result}
    split = {arm + '_wrist_ref_respondable_visual': arm + '_wrist_yaw_visual' for arm in arms}
    for left, right in labels:
        for a in [left] + ([split[left]] if left in split else []):
            for b in [right] + ([split[right]] if right in split else []):
                key = tuple(sorted((a, b)))
                if key not in seen and a != b:
                    result.append([a, b])
                    seen.add(key)
    # New distal DOFs need non-adjacent local checks, not only the old external pairs.
    # Adjacent pitch/yaw hinge and yaw/jaw mounting contacts remain deliberately excluded.
    for arm in arms:
        distal = [arm + '_wrist_yaw_visual', arm + '_yaw_ref1_respondable_visual',
                  arm + '_yaw_ref2_respondable_visual', arm + '_tool_tip_visual']
        proximal = [arm + '_tool_main_respondable_visual', arm + '_rot_ref_respondable_visual']
        for a in distal:
            for b in proximal:
                key = tuple(sorted((a, b)))
                if key not in seen:
                    result.append([a, b]);seen.add(key)
    return result


def _dummy(sim, name, parent, matrix):
    handle = sim.createDummy(0.001)
    sim.setObjectAlias(handle, name)
    sim.setObjectMatrix(handle, -1, np.asarray(matrix)[:3].reshape(-1).tolist())
    sim.setObjectParent(handle, parent, True)
    return handle


def install_wrist_yaw(sim, by, arm, model):
    """Install only at neutral pitch/jaws, preserving zero-pose geometry and the roll home.

    Refuse repeat installation so callers cannot accidentally add another independent yaw.
    The returned metadata is stored with the geometry-identified INSTRUMENTS table.
    """
    import instrument_guard as guard
    from install_instruments import replace_shape, copy_appearance
    import trimesh

    name = arm + '_wrist_yaw_joint'
    if name in by:
        raise ValueError(f'{arm}: wrist yaw already exists')
    if any(abs(sim.getJointPosition(by[f'{arm}_joint{n}'])) > 1e-7 for n in (9, 10, 11)):
        raise ValueError(f'{arm}: migration requires neutral pitch and closed jaws')
    g, desc = inst.geometry(model), inst.descriptor(model)
    if 'pitch_mesh' not in g:
        raise ValueError(f'{arm}: no articulated dVRK geometry for {model}')
    P, _ = guard.wrist_frame(sim, by, arm)
    Y = P @ g['T_PY']
    k = desc['kinematics']
    # Negative hardware jaw requests are used for torque/closure control, not a
    # geometric licence for blades to pass through each other. ProGrasp's hinge
    # law is only specified on 0..60 deg by the pinned geometric source.
    jaw_max = min(k['jaw_limits_rad'][1], g['hinge_jaw_range_rad'][1]) if g['hinge_jaw_range_rad'] else k['jaw_limits_rad'][1]
    for n in (10, 11):
        sim.setJointInterval(by[f'{arm}_joint{n}'], False, [0.0, jaw_max / 2])
    # Keep the old pitch body and its child hierarchy, replacing only its merged mesh.
    V, F = g['pitch_mesh']
    Vw = inst.apply(P, V)
    hull = trimesh.Trimesh(Vw, F).convex_hull
    for suffix, verts, faces in [('respondable', np.asarray(hull.vertices), np.asarray(hull.faces)),
                                  ('visual', Vw, F)]:
        replace_shape(sim, by[f'{arm}_wrist_ref_respondable_{suffix}'], verts, faces)
        by.update(aliases(sim))
    pitch_body = by[f'{arm}_wrist_ref_respondable_respondable']
    pitch_frame = _dummy(sim, arm + '_dvrk_pitch_frame', pitch_body, P)

    yaw_joint = sim.createJoint(sim.joint_revolute, sim.jointmode_kinematic, 0, [0.002, 0.0006])
    sim.setObjectAlias(yaw_joint, name)
    sim.setObjectMatrix(yaw_joint, -1, Y[:3].reshape(-1).tolist())
    sim.setObjectParent(yaw_joint, pitch_body, True)
    lo, hi = k['wrist_yaw_limits_rad']
    sim.setJointInterval(yaw_joint, False, [lo, hi - lo])
    sim.setJointPosition(yaw_joint, 0.0)
    yaw_link = _dummy(sim, arm + '_dvrk_yaw_frame', yaw_joint, Y)
    V, F = g['yaw_mesh']
    Vw = inst.apply(Y, V)
    housing = sim.createMeshShape(0, 0.35, Vw.reshape(-1).tolist(), F.reshape(-1).astype(int).tolist())
    sim.setObjectAlias(housing, arm + '_wrist_yaw_visual')
    copy_appearance(sim, by[f'{arm}_wrist_ref_respondable_visual'], housing)
    sim.setObjectParent(housing, yaw_link, True)
    slide = by.get(arm + '_jaw_hinge_slide')
    if slide is not None:
        sim.setObjectParent(slide, yaw_link, True)
    else:
        for n in (10, 11):
            sim.setObjectParent(by[f'{arm}_joint{n}'], yaw_link, True)
    # The reference TCP is the closed-jaw centre, not the position of one open leaf.
    for suffix in ('tool_tip_fixed_joint', 'tcp'):
        sim.setObjectParent(by[f'{arm}_{suffix}'], yaw_link, True)
    T_tcp = Y @ inst.transform(g['T_Y_tip'][:3, :3], g['tip_Y'])
    sim.setObjectMatrix(by[arm + '_tcp'], -1, T_tcp[:3].reshape(-1).tolist())
    _dummy(sim, arm + '_dvrk_tool_tip', yaw_link, Y @ g['T_Y_tip'])
    pitch_axis = matrix_to_np(sim.getObjectMatrix(by[f'{arm}_joint9'], -1))[:, 2]
    pitch_sign = 1 if np.dot(pitch_axis, P[:3, 2]) >= 0 else -1
    return dict(version=1, roll_joint=arm + '_joint8', pitch_joint=arm + '_joint9', yaw_joint=name,
                roll_sign=1, pitch_sign=pitch_sign, yaw_sign=1,
                pitch_frame=arm + '_dvrk_pitch_frame', yaw_frame=arm + '_dvrk_yaw_frame',
                limits=dict(roll=k['roll_limits_rad'], pitch=k['wrist_pitch_limits_rad'], yaw=k['wrist_yaw_limits_rad']),
                geometric_jaw_limits_rad=[0.0, jaw_max],
                tcp_semantics='closed-jaw geometric position, dVRK tool_tip orientation; separate dvrk_tool_tip uses exact source offset')


def migrate_loaded_scene(sim, ctrl):
    """Upgrade known embedded Si data plus geometry in memory; never save a .ttt here.

    Called by refresh_controller while stopped. Generated prefix values are obtained
    using a temporary Lua customization script, not by parsing/evaluating Lua in Python.
    """
    embedded = sim.getStringProperty(ctrl, 'code').replace('\r\n', '\n')
    marker = '-- Software RCM controller for the endoscope task scene'
    head, found, _ = embedded.partition(marker)
    if not found or '\nINSTRUMENTS=' not in head:
        raise ValueError('explicit articulation requires a known Si instrument scene')
    import ecm_pairs
    probe = sim.createScript(sim.scripttype_customization,
                             head + '\nfunction sysCall_init() end\nfunction data() return {CONFIG,COLLISION_PAIRS,INSTRUMENTS} end', 0, 'lua')
    try:
        ecm_pairs.wait_initialized(sim, probe)
        rcm, labels, table = sim.callScriptFunction('data', probe)
    finally:
        sim.removeObjects([probe])
    by = aliases(sim)
    arms = [c['arm'] for c in rcm if c['arm'] != 'ecm']
    for arm in arms:
        info = table[arm]
        if info.get('articulation'):
            if info['articulation']['yaw_joint'] not in by:
                raise ValueError(f'{arm}: articulation metadata has no matching joint')
            continue
        info['articulation'] = install_wrist_yaw(sim, by, arm, info['model'])
        by = aliases(sim)
    labels = append_yaw_collision_pairs(labels, arms)
    source = ('CONFIG=' + lua(rcm) + '\nCOLLISION_PAIRS=' + lua(labels) + '\nINSTRUMENTS=' + lua(table) + '\n'
              + read_text(ENDO_SRC / 'rcm_controller.lua') + '\n' + read_text(INSTRUMENT_SRC / 'capability_guard.lua')
              + '\n' + read_text(INSTRUMENT_SRC / 'articulation.lua') + '\n' + read_text(TISSUE_SRC / 'tissue.lua'))
    sim.setStringProperty(ctrl, 'code', source)
    if sim.getStringProperty(ctrl, 'code').replace('\r\n', '\n') != source:
        raise RuntimeError('articulated controller read-back mismatch')
    return source