"""Surgical access for scenes/davinci_si_instruments.ttt: pneumoperitoneum and cannulas (trocars).

Pneumoperitoneum
  The anterior abdominal skin over the ports is lifted into a dome and a 30 mm abdominal wall is added under it
  (inner surface = peritoneum), so that each RCM point ends up inside the wall, 15 mm below the skin. The lift is
      L(x) = w(x) * max(0, b(x) + sum_j c_j exp(-|x - p_j|^2 / 2 sigma^2))
  in the plane across the anterior direction: b is a dome over an ellipse around the four ports, w a window that is
  1 in its inner part and falls to 0 at its edge (cos^2), and the c_j are solved so the skin above every port sits
  exactly `rcm_depth_below_skin_m` above its RCM point. The arms and RCM points do not move.
  The skin and the inner wall become one compound shape, Patient_Skin, so every collision pair and the port
  allowance of the runtime guard apply to both surfaces.

Cannulas
  Each arm gets a cannula on the link that carries its insertion joint. With the software RCM the RCM point is
  fixed in that link's frame, so the cannula's RCM mark stays on the RCM point through every motion.
  PSMs: jhu-dvrk/dvrk-cannulas Si-cannula-8mm.stl. ECM: the same OpenSCAD profile revolved with a bore that
  fits the scope. Cannulas collide with anatomy, tables, other arms and other cannulas (not with the wall they
  pass through).
"""
import math

import numpy as np

from simtools import CONFIG, REPO, matrix_to_np, np_to_matrix, read_json

ACCESS = CONFIG / 'instruments' / 'access.json'
SKIN_NPZ = REPO / 'models' / 'anatomy' / 'corrected_skin.npz'
INSUFFLATED_NPZ = REPO / 'models' / 'anatomy' / 'insufflated_wall.npz'


def unit(v):
    return v / np.linalg.norm(v)


def world(sim, h):
    M = np.eye(4)
    M[:3, :] = matrix_to_np(sim.getObjectMatrix(h, -1))
    return M


def apply(T, V):
    return V @ T[:3, :3].T + T[:3, 3]


def frame_from_z(z):
    x = unit(np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0]))
    return np.column_stack([x, np.cross(z, x), z])


def pose7(R, t):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()
    return list(map(float, t)) + list(map(float, q))


# -- pneumoperitoneum -------------------------------------------------------------------------------------------
def lift_field(skin_w, normals, ports, anterior, p):
    """Lift per skin vertex (m) and the window, from the port positions; see the module docstring."""
    e1 = unit(np.cross(anterior, [1.0, 0, 0]) if abs(anterior[0]) < 0.9 else np.cross(anterior, [0, 1.0, 0]))
    e2 = np.cross(anterior, e1)
    B = np.column_stack([e1, e2])
    q = ports @ B
    c = q.mean(axis=0)
    _, _, Vt = np.linalg.svd(q - c)
    axes = Vt                                            # principal directions of the port layout
    ext = np.abs((q - c) @ axes.T).max(axis=0)
    semi = ext + p['region_margin_m']

    def rho(x2):
        return np.sqrt(((((x2 - c) @ axes.T) / semi) ** 2).sum(axis=1))

    f = p['region_flat_fraction']

    def window(r):
        t = np.clip((r - f) / (1 - f), 0, 1)
        w = np.where(r <= f, 1.0, np.cos(0.5 * math.pi * t) ** 2)
        return np.where(r >= 1, 0.0, w)

    def dome(r):
        return p['dome_height_m'] * np.clip(1 - r ** 2, 0, None)

    sig = p['rbf_sigma_m']

    def phi(x2):
        d2 = ((x2[:, None, :] - q[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-d2 / (2 * sig * sig))

    # Skin height at each port: front-facing skin vertices within 5 mm (across the anterior axis).
    front = normals @ anterior > 0.2
    xs = skin_w @ B
    hs = skin_w @ anterior
    target = []
    old_offset = []
    for P, qp in zip(ports, q):
        # Skin over the port: vertices within 8 mm across the anterior axis and within 30 mm of the RCM height
        # (the back of the body is further away); the highest of them is the outer surface.
        near = (np.linalg.norm(xs - qp, axis=1) < 0.008) & (np.abs(hs - P @ anterior) < 0.03)
        if not near.any():
            raise SystemExit(f'no skin above port {P}')
        s = float(hs[near].max() - P @ anterior)               # RCM this far below the original skin
        old_offset.append(s)
        target.append(p['rcm_depth_below_skin_m'] - s)
    rp = rho(q)
    wp, bp = window(rp), dome(rp)
    coef = np.linalg.solve(phi(q), np.array(target) / wp - bp)
    r_all = rho(xs)
    w_all = window(r_all) * front
    L = w_all * np.clip(dome(r_all) + phi(xs) @ coef, 0, None)
    return dict(L=L, w=w_all, front=front, old_offset=old_offset, port_lift=target, rho_ports=rp.tolist(),
                semi_axes_m=semi.tolist(), center2d=c.tolist(), coef=coef.tolist())


def build_pneumoperitoneum(sim, by, rcm, anterior, probe_scan):
    """Replace Patient_Skin by the lifted skin + inner abdominal wall compound. Returns a report."""
    import trimesh
    p = read_json(ACCESS)['pneumoperitoneum']
    z = np.load(SKIN_NPZ)
    old = by['Patient_Skin']
    Ms = world(sim, old)
    parent = sim.getObjectParent(old)
    Vw = apply(Ms, z['vertices'])
    F = z['faces']
    tm = trimesh.Trimesh(Vw, F, process=False)
    tm.fix_normals()
    normals = np.asarray(tm.vertex_normals)
    if (normals @ anterior)[np.argmax(Vw @ anterior)] < 0:          # make the normals point out of the body
        normals = -normals
    ports = np.array([a['rcm_world'] for a in rcm])
    field = lift_field(Vw, normals, ports, anterior, p)
    outer = Vw + field['L'][:, None] * anterior
    # Wall thickness is measured along the local surface normal of the lifted skin (an offset along the anterior
    # axis would make the wall much thicker where the lower abdomen slopes, which is where the ports are).
    n_out = np.asarray(trimesh.Trimesh(outer, F, process=False).vertex_normals)
    n_out = np.where(((n_out * normals).sum(axis=1) < 0)[:, None], -n_out, n_out)
    T = p['wall_thickness_m'] * field['w']
    inner_V = outer - T[:, None] * n_out
    # Insufflation lifts the wall away from the organs; it must not push the peritoneum into the original body
    # space, where the instrument tips work. The inner surface stays at least this far up (along the anterior
    # axis) from `peritoneum_floor_below_original_skin_m` under the original skin.
    floor = Vw - p['peritoneum_floor_below_original_skin_m'] * anterior
    below = (inner_V - floor) @ anterior < 0
    inner_V[below] = floor[below]
    keep = (field['w'][F] > 1e-3).all(axis=1)
    used = np.unique(F[keep])
    remap = -np.ones(len(Vw), dtype=np.int64)
    remap[used] = np.arange(len(used))
    inner_F = remap[F[keep]][:, ::-1]
    inner_V = inner_V[used]

    # Inner wall must not cut into organs or bones.
    hi = sim.createMeshShape(0, math.radians(30), inner_V.reshape(-1).tolist(), inner_F.reshape(-1).astype(int).tolist())
    organs = [n for n in by if n.startswith('Patient_') and n != 'Patient_Skin']
    hits = [organs[i] for i, _ in probe_scan([[hi, by[n]] for n in organs])]
    if hits:
        sim.removeObjects([hi])
        raise SystemExit(f'inner abdominal wall intersects {hits}')

    # Crossing depths along each shaft: outer skin above, inner wall below the RCM.
    m_out = trimesh.Trimesh(outer, F, process=False)
    m_in = trimesh.Trimesh(inner_V, inner_F, process=False)
    depths = {}
    for a in rcm:
        r, ax = np.array(a['rcm_world']), np.array(a['axis_home'])
        up = m_out.ray.intersects_location([r], [-ax])[0]
        dn = m_in.ray.intersects_location([r], [ax])[0]
        depths[a['arm']] = dict(rcm_to_skin_along_shaft_m=float(np.linalg.norm(up - r, axis=1).min()) if len(up) else None,
                                rcm_to_peritoneum_along_shaft_m=float(np.linalg.norm(dn - r, axis=1).min()) if len(dn) else None)

    res, col = sim.getShapeColor(old, None, sim.colorcomponent_ambient_diffuse)
    ho = sim.createMeshShape(0, math.radians(30), outer.reshape(-1).tolist(), F.reshape(-1).astype(int).tolist())
    sim.setShapeColor(ho, None, sim.colorcomponent_ambient_diffuse, col)
    sim.setShapeColor(hi, None, sim.colorcomponent_ambient_diffuse, list(p['inner_wall_rgb']))
    for h in (ho, hi):
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
    sim.setObjectAlias(hi, 'Patient_AbdominalWall_Inner')
    sim.removeObjects([old])
    comp = sim.groupShapes([ho, hi], False)
    sim.setObjectAlias(comp, 'Patient_Skin')
    sim.setObjectParent(comp, parent, True)
    sim.setObjectInt32Param(comp, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(comp, sim.shapeintparam_respondable, 0)
    Mp = np.linalg.inv(world(sim, parent))
    np.savez_compressed(INSUFFLATED_NPZ, outer_vertices_parent=apply(Mp, outer), faces=F,
                        inner_vertices_parent=apply(Mp, inner_V), inner_faces=inner_F)
    return dict(params=p, port_lift_m=dict(zip([a['arm'] for a in rcm], field['port_lift'])),
                rcm_below_original_skin_m=dict(zip([a['arm'] for a in rcm], field['old_offset'])),
                port_rho=dict(zip([a['arm'] for a in rcm], field['rho_ports'])),
                max_lift_m=float(field['L'].max()), lifted_vertices=int((field['L'] > 1e-4).sum()),
                inner_wall_faces=int(len(inner_F)), crossing=depths, region_semi_axes_m=field['semi_axes_m'],
                inner_wall_organ_contacts=hits)


# -- cannulas --------------------------------------------------------------------------------------------------
def revolve(profile, n=64):
    """Closed surface of revolution about z from a closed (r, z) polygon."""
    P = np.asarray(profile, float)
    m = len(P)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    V = np.array([[r * math.cos(a), r * math.sin(a), zz] for a in ang for r, zz in P])
    F = []
    for i in range(n):
        j = (i + 1) % n
        for k in range(m):
            k2 = (k + 1) % m
            a, b, c, d = i * m + k, j * m + k, j * m + k2, i * m + k2
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F)


def ecm_cannula_mesh(c, bore_r):
    t = c['ecm_from_scad']
    ri = bore_r + t['bore_clearance_m']
    ro = ri + t['wall_m']
    rm, rt = t['mount_d_m'] / 2, t['top_d_m'] / 2
    hm, ht, L = t['mount_h_m'], t['top_h_m'], t['tube_length_m']
    # (r, z) outline, tube top at z = 0, counter-clockwise
    prof = [(ri, -L), (ro, -L), (ro, 0.0), (rm, 0.0), (rm, hm), (rt, hm), (rt, hm + ht), (ri, hm + ht)]
    return revolve(prof)


def psm_cannula_mesh(c, faces):
    import trimesh
    m = trimesh.load(REPO / c['psm_mesh'])
    m.apply_scale(0.001)
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def scope_radius(sim, by, ecm_cfg):
    tip = world(sim, by[ecm_cfg['shaft_frame']])
    h = by[ecm_cfg['tool_shape']]
    V = apply(world(sim, h), np.asarray(sim.getShapeMesh(h)[0], float).reshape(-1, 3))
    rel = V - tip[:3, 3]
    along = rel @ tip[:3, 2]
    sel = (along < -0.01) & (along > -0.12)                  # scope tube behind the tip
    radial = np.linalg.norm(rel[sel] - np.outer(along[sel], tip[:3, 2]), axis=1)
    return float(np.percentile(radial, 95))


def add_cannulas(sim, by, rcm, ring_fn):
    """One cannula per arm, RCM mark on the RCM point. Returns {arm: info}."""
    c = read_json(ACCESS)['cannula']
    psm_mesh = psm_cannula_mesh(c, c['faces'])
    out = {}
    for a in rcm:
        arm = a['arm']
        if arm == 'ecm':
            bore = scope_radius(sim, by, a)
            V, F = ecm_cannula_mesh(c, bore)
            kind = f"revolved Si profile, bore {2 * (bore + c['ecm_from_scad']['bore_clearance_m']) * 1000:.1f} mm"
        else:
            V, F = psm_mesh
            kind = c['psm_mesh']
        zc = -np.array(a['axis_home'])
        R = frame_from_z(zc)
        origin = np.array(a['rcm_world']) + c['rcm_below_tube_top_m'] * zc
        Vw = V @ R.T + origin
        h = sim.createMeshShape(0, math.radians(25), Vw.reshape(-1).tolist(), F.reshape(-1).astype(int).tolist())
        sim.relocateShapeFrame(h, pose7(R, origin))
        sim.setObjectAlias(h, f'{arm}_cannula')
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0.80, 0.82, 0.85])
        sim.setShapeColor(h, None, sim.colorcomponent_specular, [0.4, 0.4, 0.4])
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
        parent = sim.getObjectParent(by[a['insertion_joint']])
        sim.setObjectParent(h, parent, True)
        # Dark RCM band on the tube (visual only, not in any collision pair).
        tube_r = float(np.percentile(np.linalg.norm(V[V[:, 2] < -0.05][:, :2], axis=1), 99))
        band = ring_fn(sim, f'{arm}_cannula_rcm_band', np.array(a['rcm_world']), R[:, 0], R[:, 1], R[:, 2],
                       tube_r + 0.0003, c['rcm_band']['width_m'] / 2)
        sim.setShapeColor(band, None, sim.colorcomponent_ambient_diffuse, [0.08, 0.08, 0.08])
        sim.setObjectParent(band, h, True)
        out[arm] = dict(shape=f'{arm}_cannula', parent=sim.getObjectAlias(parent), mesh=kind,
                        faces=int(len(F)), tube_outer_radius_m=tube_r,
                        length_below_rcm_m=float(-V[:, 2].min() - c['rcm_below_tube_top_m']),
                        length_above_rcm_m=float(V[:, 2].max() + c['rcm_below_tube_top_m']))
    return out


def cannula_pairs(shape_names, arms):
    """Collision pairs of the cannulas: anatomy (not the wall they cross), tables, other arms, other cannulas."""
    names = sorted(set(shape_names))
    anatomy = [n for n in names if n.startswith('Patient_') and n != 'Patient_Skin']
    tables = [n for n in names if n == 'TaskSurface' or n.startswith('LegSupport_')]
    labels, cats = [], []
    for i, arm in enumerate(arms):
        c = f'{arm}_cannula'
        for n in anatomy:
            labels.append([c, n]); cats.append('cannula_anatomy')
        for n in tables:
            labels.append([c, n]); cats.append('cannula_table')
        for n in names:
            if n.endswith('_visual') and n.startswith(('psm', 'ecm_')) and not n.startswith(arm + '_'):
                labels.append([c, n]); cats.append('cannula_cross_arm')
        for other in arms[i + 1:]:
            labels.append([c, f'{other}_cannula']); cats.append('cannula_cannula')
    return labels, cats
