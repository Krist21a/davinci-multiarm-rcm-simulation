"""Instrument descriptors (config/instruments/) and dVRK instrument geometry in dVRK link frames.

Frames (all at wrist pitch = wrist yaw = jaw = 0):
  P  dvrk wrist_pitch_link: origin on the pitch axis, z = pitch axis, x = distal (towards the yaw axis).
  Y  dvrk wrist_yaw_link:   origin on the yaw axis,   z = yaw/jaw axis, y = distal.
  H  jaw hinge frame: Y shifted along y by the jaw hinge offset (non-zero only for the ProGrasp tip).
The scene uses two jaw joints per arm (joint10, joint11) that both open with a positive angle. A dVRK
jaw link that opens by rotating about +z_Y belongs to the joint whose axis is +z_Y, the other to the
joint whose axis is -z_Y; `jaw_plus` / `jaw_minus` below follow that rule.
"""
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from simtools import CONFIG, MESHES, read_json

INSTRUMENT_CONFIG = CONFIG / 'instruments'
DVRK = MESHES / 'instrument' / 'dvrk'
XACRO = DVRK / 'urdf' / 'common' / 'instruments'
CAPABILITIES = ['grasp', 'grasp_tissue', 'retract_tissue', 'cut', 'monopolar_energy', 'bipolar_energy',
                'blunt_dissection', 'needle_drive', 'view']


def descriptor(model):
    d = read_json(INSTRUMENT_CONFIG / f'{model}.json')
    unknown = set(d['capabilities']) - set(CAPABILITIES)
    if unknown:
        raise ValueError(f'{model}: unknown capabilities {sorted(unknown)}')
    return d


def roles():
    return read_json(INSTRUMENT_CONFIG / 'roles.json')


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _num(text):
    text = text.replace('${PI_2}', repr(math.pi / 2)).replace('${PI}', repr(math.pi))
    return [float(v) for v in text.split()]


def _origin(el):
    o = el.find('origin')
    if o is None:
        return np.eye(4)
    return transform(rpy_matrix(*_num(o.get('rpy', '0 0 0'))), _num(o.get('xyz', '0 0 0')))


def _strip(name):
    return name.replace('${prefix}', '')


def parse_stage(path):
    """Joints and visual meshes of one dvrk_model instrument stage xacro."""
    root = ET.parse(path).getroot()
    joints, visuals = {}, {}
    for j in root.iter('joint'):
        mimic = j.find('mimic')
        limit = j.find('limit')
        axis = j.find('axis')
        joints[j.get('name')] = dict(
            type=j.get('type'), parent=_strip(j.find('parent').get('link')), child=_strip(j.find('child').get('link')),
            origin=_origin(j), axis=np.array(_num(axis.get('xyz'))) if axis is not None else None,
            mimic=None if mimic is None else dict(joint=mimic.get('joint'), multiplier=float(mimic.get('multiplier', 1)),
                                                  offset=float(mimic.get('offset', 0))),
            limit=None if limit is None else (float(limit.get('lower')), float(limit.get('upper'))))
    for link in root.iter('link'):
        for v in link.findall('visual'):
            mesh = v.find('geometry/mesh')
            if mesh is None:
                continue
            scale = _num(mesh.get('scale', '1 1 1'))
            rel = mesh.get('filename').replace('package://dvrk_model/', '')
            visuals.setdefault(_strip(link.get('name')), []).append(dict(file=DVRK / rel, origin=_origin(v), scale=scale))
    return joints, visuals


def load_obj(path):
    import trimesh
    m = trimesh.load(path, force='mesh', process=True)
    return np.asarray(m.vertices, dtype=float), np.asarray(m.faces, dtype=np.int64)


def mesh_in_link(visual):
    V, F = load_obj(visual['file'])
    V = V * np.asarray(visual['scale'])
    T = visual['origin']
    return V @ T[:3, :3].T + T[:3, 3], F


def merge(meshes):
    Vs, Fs, n = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + n)
        n += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def apply(T, V):
    return V @ T[:3, :3].T + T[:3, 3]


def decimate(V, F, faces):
    import trimesh
    m = trimesh.Trimesh(V, F, process=True)
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    return np.asarray(m.vertices, dtype=float), np.asarray(m.faces, dtype=np.int64)


def save_reference(path, g):
    np.savez_compressed(path, wrist_V=g['wrist'][0], wrist_F=g['wrist'][1], jaw_plus_V=g['jaw_plus'][0],
                        jaw_plus_F=g['jaw_plus'][1], jaw_minus_V=g['jaw_minus'][0], jaw_minus_F=g['jaw_minus'][1],
                        T_PY=g['T_PY'], hinge_Y=g['hinge_Y'], tip_Y=g['tip_Y'])


def geometry(model, faces_per_part=2500):
    """Wrist (pitch body + yaw clevis) in P, the two jaws in H, and the frame offsets, from dvrk_model."""
    d = descriptor(model)
    if 'reference_npz' in d:          # geometry captured from the repository's own instrument meshes
        z = np.load(CONFIG.parent / d['reference_npz'])
        return dict(model=model, wrist=(z['wrist_V'], z['wrist_F']), jaw_plus=(z['jaw_plus_V'], z['jaw_plus_F']),
                    jaw_minus=(z['jaw_minus_V'], z['jaw_minus_F']), T_PY=z['T_PY'], hinge_Y=z['hinge_Y'],
                    tip_Y=z['tip_Y'], tip_dvrk_Y=z['tip_Y'], hinge_slope_m_per_rad=0.0, hinge_jaw_range_rad=None)
    st = d['stages']
    pj, pv = parse_stage(XACRO / 'wrist_pitch' / f"wrist_pitch_{st['wrist_pitch']}.urdf.xacro")
    yj, yv = parse_stage(XACRO / 'wrist_yaw' / f"wrist_yaw_{st['wrist_yaw']}.urdf.xacro")
    tj, tv = parse_stage(XACRO / 'tip' / f"tip_{st['tip']}.urdf.xacro")
    T_PY = yj['wrist_yaw']['origin']
    A = float(np.linalg.norm(T_PY[:3, 3]))
    if abs(A - d['kinematics']['pitch_to_yaw_m']) > 1e-6:
        raise ValueError(f'{model}: dvrk_model yaw offset {A} differs from the descriptor')
    pitch_mesh = merge([mesh_in_link(v) for v in pv.get('wrist_pitch_link', [])])
    yaw_mesh = merge([mesh_in_link(v) for v in yv.get('wrist_yaw_link', [])])
    wrist = [pitch_mesh, (apply(T_PY, yaw_mesh[0]), yaw_mesh[1])]
    wrist = decimate(*merge(wrist), faces_per_part)

    # Jaw hinge: jaw links hang from wrist_yaw_link directly or from a prismatic jaw_axis_link (ProGrasp).
    hinge = np.zeros(3)
    hinge_slope, hinge_range = 0.0, None
    jaws = {}
    for name, j in tj.items():
        if j['mimic'] is None or j['mimic']['joint'] != 'jaw':
            continue
        if j['type'] == 'prismatic':
            hinge = j['origin'][:3, 3] + j['axis'] * j['mimic']['offset']      # closed jaw (jaw = 0)
            hinge_slope = j['mimic']['multiplier']                              # m of hinge travel per rad of jaw
            hinge_range = list(tj['jaw']['limit'])
            continue
        if j['child'] not in tv:
            continue
        if not np.allclose(j['origin'], np.eye(4)):
            raise ValueError(f'{model}: jaw joint {name} has a non-identity origin')
        sign = int(np.sign(j['mimic']['multiplier'] * j['axis'][2]))
        mesh = decimate(*merge([mesh_in_link(v) for v in tv[j['child']]]), faces_per_part)
        jaws['jaw_plus' if sign > 0 else 'jaw_minus'] = mesh
    if set(jaws) != {'jaw_plus', 'jaw_minus'}:
        raise ValueError(f'{model}: expected one opening and one closing jaw, got {sorted(jaws)}')
    k = d['kinematics']
    if 'jaw_pivot_Y_m' in k:
        # Measured pivot (the pin hole shared by both jaws) replaces the dvrk_model hinge; the jaw meshes keep
        # their place in the wrist and are re-expressed about the new pivot.
        pivot = np.array(k['jaw_pivot_Y_m'], dtype=float)
        jaws = {n: (V + hinge - pivot, F) for n, (V, F) in jaws.items()}
        hinge = pivot
    if abs(float(np.linalg.norm(hinge)) - k['jaw_hinge_offset_m']) > 1e-6:
        raise ValueError(f'{model}: hinge offset {hinge} differs from the descriptor')
    tip_dvrk = tj['tool_tip']['origin'][:3, 3]
    return dict(model=model, wrist=wrist, pitch_mesh=decimate(*pitch_mesh, faces_per_part),
                yaw_mesh=decimate(*yaw_mesh, faces_per_part),
                T_PY=T_PY, hinge_Y=hinge, tip_Y=jaw_tip(jaws, hinge), tip_dvrk_Y=tip_dvrk,
                T_Y_tip=tj['tool_tip']['origin'],
                hinge_slope_m_per_rad=hinge_slope, hinge_jaw_range_rad=hinge_range, **jaws)


def jaw_tip(jaws, hinge, band=3e-4):
    """Geometric tool tip (in Y): centre of the distal-most 0.3 mm of the closed jaws, on the jaw centre plane."""
    P = np.vstack([V for V, _ in jaws.values()]) + hinge
    sel = P[P[:, 1] >= P[:, 1].max() - band]
    return np.array([sel[:, 0].mean(), P[:, 1].max(), sel[:, 2].mean()])


def fingerprint(V, F, quantum=1e-5):
    """Order-independent hash of a mesh given in its link frame, quantised to 10 um."""
    q = np.round(np.asarray(V, dtype=float).reshape(-1, 3) / quantum).astype(np.int64)
    tris = sorted(tuple(sorted(map(tuple, t.tolist()))) for t in q[np.asarray(F, dtype=np.int64).reshape(-1, 3)])
    return hashlib.sha256(np.array(tris, dtype=np.int64).tobytes()).hexdigest()
