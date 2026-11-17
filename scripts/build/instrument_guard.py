"""Identify the instrument on each PSM from scene geometry and check it against the arm's role.

The guard never reads a model number or role label from the scene. For each PSM it
  1. rebuilds the dVRK wrist frame P from the scene joints (pitch axis joint9, jaw axes joint10/joint11),
  2. places every candidate instrument of config/instruments/ in that frame, and
  3. compares the candidate wrist and jaw surfaces with the meshes hanging from joint9/joint10/joint11
     (symmetric mean nearest-point distance).
The closest candidate within tolerance is the instrument on that arm. A role in roles.json only lists the
capabilities it needs; the arm passes only when the identified instrument has all of them. An 'instrument'
entry in roles.json that disagrees with the geometry is reported as a mismatch and fails the arm.

CLI: loads a scene in a dedicated CoppeliaSim instance and writes a JSON report.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import instruments as inst
from simtools import (ENDO_CONFIG, REPO, add_arguments, aliases, connect, load_scene, matrix_to_np, read_json,
                      relative, stop_simulation, write_json)

MODELS = ['420006', '420179', '420172', '420093', 'legacy_lnd']
TOLERANCE_M = 0.0005          # mean surface distance accepted as "same instrument"
_GEOMETRY = {}


def candidate(model):
    if model not in _GEOMETRY:
        _GEOMETRY[model] = inst.geometry(model)
    return _GEOMETRY[model]


def unit(v):
    return v / np.linalg.norm(v)


def world(sim, h):
    M = np.eye(4)
    M[:3, :] = matrix_to_np(sim.getObjectMatrix(h, -1))
    return M


def shape_points(sim, h):
    """Vertices plus triangle centroids of a shape, in world coordinates."""
    V, F = sim.getShapeMesh(h)[:2]
    V = np.asarray(V, dtype=float).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    P = np.vstack([V, V[F].mean(axis=1)])
    return inst.apply(world(sim, h), P)


def mesh_points(T, mesh):
    V, F = mesh
    return inst.apply(T, np.vstack([V, V[F].mean(axis=1)]))


def surface_distance(a, b):
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def shapes_under(sim, joint):
    """Visible shapes rigidly attached to a joint (its child link), excluding deeper joints."""
    out, stack = [], [sim.getObjectChild(joint, i) for i in range(64) if sim.getObjectChild(joint, i) != -1]
    while stack:
        h = stack.pop()
        t = sim.getObjectType(h)
        if t in (sim.sceneobject_joint, sim.sceneobject_forcesensor):     # next link, or the tool-tip marker
            continue
        if t == sim.sceneobject_shape and sim.getObjectInt32Param(h, sim.objintparam_visibility_layer) & 0xff:
            out.append(h)
        stack.extend(c for c in (sim.getObjectChild(h, i) for i in range(64)) if c != -1)
    return out


def wrist_frame(sim, by, arm):
    """dVRK wrist_pitch_link frame P (world) from the scene joints; also returns the frame residuals."""
    j9, j10, j11 = (world(sim, by[f'{arm}_joint{n}']) for n in (9, 10, 11))
    x = unit(j10[:3, 3] - j9[:3, 3])
    z9, z10, z11 = j9[:3, 2], j10[:3, 2], j11[:3, 2]
    s = 1.0 if np.dot(np.cross(z9, x), z10) > 0 else -1.0
    z = s * z9
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = j9[:3, 3]
    res = dict(pitch_axis_dot_distal=float(abs(np.dot(z9, x))), jaw_axis_error=float(np.linalg.norm(y - z10)),
               jaw_axes_antiparallel_error=float(np.linalg.norm(z10 + z11)),
               jaw_joint_offset_m=float(np.linalg.norm(j10[:3, 3] - j9[:3, 3])),
               jaw_joints_coincident_m=float(np.linalg.norm(j10[:3, 3] - j11[:3, 3])))
    return T, res


def expected(model, T_WP):
    g = candidate(model)
    T_WY = T_WP @ g['T_PY']
    T_WH = T_WY @ inst.transform(t=g['hinge_Y'])
    return dict(wrist=mesh_points(T_WP, g['wrist']), jaw_plus=mesh_points(T_WH, g['jaw_plus']),
                jaw_minus=mesh_points(T_WH, g['jaw_minus']), T_WY=T_WY, T_WH=T_WH)


def identify_arm(sim, by, arm):
    T_WP, frame_res = wrist_frame(sim, by, arm)
    scene = {}
    for part, n in (('wrist', 9), ('jaw_plus', 10), ('jaw_minus', 11)):
        hs = shapes_under(sim, by[f'{arm}_joint{n}'])
        scene[part] = np.vstack([shape_points(sim, h) for h in hs]) if hs else np.zeros((0, 3))
    scores = {}
    for m in MODELS:
        e = expected(m, T_WP)
        scores[m] = {p: float(surface_distance(scene[p], e[p])) for p in ('wrist', 'jaw_plus', 'jaw_minus')}
    mean = {m: float(np.mean(list(s.values()))) for m, s in scores.items()}
    best = min(mean, key=mean.get)
    worst_part = max(scores[best].values())
    second = sorted(mean.values())[1]
    model = best if worst_part <= TOLERANCE_M else None
    return dict(model=model, closest=best, closest_worst_part_m=worst_part, margin_to_second_m=second - mean[best],
                scores_m=scores, frame=frame_res)


def ecm_has_camera(sim, by, rcm):
    ecm = next((a for a in rcm if a['arm'] == 'ecm'), None)
    if ecm is None:
        return False
    tip = by.get(ecm['shaft_frame'])
    stack = [tip] if tip is not None else []
    while stack:
        h = stack.pop()
        if sim.getObjectType(h) == sim.sceneobject_visionsensor:
            return True
        stack.extend(c for c in (sim.getObjectChild(h, i) for i in range(64)) if c != -1)
    return False


def check(sim, by, rcm, role_cfg):
    """Per-arm identification and role check. Returns (ok, report)."""
    report, ok = {}, True
    for arm, spec in role_cfg['arms'].items():
        need = role_cfg['roles'][spec['role']]['requires']
        if arm == 'ecm':
            caps = ['view'] if ecm_has_camera(sim, by, rcm) else []
            entry = dict(role=spec['role'], requires=need, identified='endoscope_camera' if caps else None,
                         capabilities=caps)
        else:
            ident = identify_arm(sim, by, arm)
            caps = inst.descriptor(ident['model'])['capabilities'] if ident['model'] else []
            entry = dict(role=spec['role'], requires=need, identified=ident['model'], capabilities=caps,
                         identification=ident)
            if spec.get('instrument') and spec['instrument'] != ident['model']:
                entry['declared_instrument_mismatch'] = dict(declared=spec['instrument'], geometry=ident['model'])
        entry['missing'] = [c for c in need if c not in caps]
        entry['ok'] = not entry['missing'] and 'declared_instrument_mismatch' not in entry
        ok &= entry['ok']
        report[arm] = entry
    return ok, report


def require(sim, by, rcm, arm, capabilities):
    """Raise SystemExit unless the instrument identified on `arm` has every capability listed."""
    ident = identify_arm(sim, by, arm)
    caps = inst.descriptor(ident['model'])['capabilities'] if ident['model'] else []
    missing = [c for c in capabilities if c not in caps]
    if missing:
        raise SystemExit(f'{arm}: instrument {ident["model"] or "unidentified"} lacks {missing}; refusing the task')
    return ident


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--rcm-config', type=Path, default=ENDO_CONFIG / 'rcm_config.json')
    p.add_argument('--roles', type=Path, default=inst.INSTRUMENT_CONFIG / 'roles.json')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    client, sim = connect(args)
    load_scene(client, sim, args, args.scene.resolve(), 'instrument_guard.ttt')
    by = aliases(sim)
    ok, report = check(sim, by, read_json(args.rcm_config), read_json(args.roles))
    stop_simulation(client, sim)
    write_json(args.out, dict(scene=relative(args.scene), roles=relative(args.roles), ok=ok,
                              tolerance_m=TOLERANCE_M, arms=report))
    for arm, e in report.items():
        print(arm, e['role'], 'identified', e['identified'], 'missing', e['missing'],
              'mismatch' if 'declared_instrument_mismatch' in e else '', 'OK' if e['ok'] else 'REFUSED')
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 3)


if __name__ == '__main__':
    main()
