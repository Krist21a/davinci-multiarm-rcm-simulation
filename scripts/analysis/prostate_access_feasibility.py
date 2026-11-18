"""Stage-2 feasibility: can the endoscope see, and can three instruments reach, the tissue beside the prostate?

Read-only analysis. The scene is loaded from a staged copy only to read its meshes and poses (cached in
--cache); it is never saved.

Targets
  Points 2 mm outside the prostate surface on its lateral and anterolateral sides (where the endopelvic fascia
  is incised), one per 3 mm cell.
Occluders
  Bones (hip bones, sacrum), rectum, prostate, bladder, and from BodyParts3D (CC BY 4.0) parts the scene does not
  have yet: seminal vesicles, urethra, levator ani (pubococcygeus, iliococcygeus), tendinous arch of levator ani,
  obturator internus, external iliac arteries and veins. Two bladder cases: in place, and removed (an optimistic
  bound for a bladder that has been taken down).
Camera
  From the present ECM port: scope tip 50-170 mm past the port, tilted up to 45 deg from the port-target line.
  A target is visible when, for some tip, the scope path (8 mm radius) and the line of sight are clear, the
  target is 25-150 mm away and within 65 deg of the scope axis (30 deg oblique + 35 deg half view angle).
Instruments
  Candidate ports on the insufflated skin (10 mm grid, anterior side). A port reaches a target when the target
  lies 60-220 mm from it and the straight shaft (6 mm radius incl. margin; first 20 mm in the wall and last
  12 mm at the jaws excluded) is clear. A target is triangulable when three such ports exist that are >= 60 mm
  apart, >= 60 mm from the ECM port and whose shafts meet at 20-100 deg.

Writes validation/feasibility/prostate_access.json and docs/feasibility_prostate_access.png.
"""
import argparse
import itertools
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'build'))
from simtools import INSTRUMENT_SCENE, REPO, add_arguments, aliases, connect, load_scene, matrix_to_np, read_json, write_json  # noqa: E402

BP3D_ZIP = Path(r'D:\仿真环境\assets\bodyparts3d_v4\isa_BP3D_4.0_obj_99.zip')
ELEMENTS = Path(r'D:\仿真环境\assets\bodyparts3d_v4\isa_element_parts.txt')
NEW_PARTS = {          # FMA concept -> name (all element files of the concept are loaded)
    'FMA19387': 'right seminal vesicle', 'FMA19388': 'left seminal vesicle', 'FMA19667': 'urethra',
    'FMA45854': 'right pubococcygeus', 'FMA45855': 'left pubococcygeus', 'FMA45858': 'right iliococcygeus',
    'FMA45859': 'left iliococcygeus', 'FMA46442': 'tendinous arch of levator ani',
    'FMA22324': 'right obturator internus', 'FMA22325': 'left obturator internus',
    'FMA18806': 'right external iliac artery', 'FMA18807': 'left external iliac artery',
    'FMA18885': 'right external iliac vein', 'FMA18886': 'left external iliac vein',
}
SCENE_PARTS = ['Patient_RightHipBone', 'Patient_LeftHipBone', 'Patient_Sacrum', 'Patient_Rectum', 'Patient_Prostate',
               'Patient_Bladder']
R0 = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0.]])      # BodyParts3D -> body frame (scripts/authoring)
T0 = np.array([.78, 0, .4652476])
OUT = REPO / 'validation' / 'feasibility' / 'prostate_access.json'
FIG = REPO / 'docs' / 'feasibility_prostate_access.png'


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def read_obj(data):
    v, f = [], []
    for line in data.decode('latin-1').splitlines():
        a = line.split()
        if not a:
            continue
        if a[0] == 'v':
            v.append([float(x) for x in a[1:4]])
        elif a[0] == 'f':
            ids = [int(x.split('/')[0]) - 1 for x in a[1:]]
            f += [[ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1)]
    return np.array(v) * 0.001, np.array(f)


def bp3d_to_world(v, B, shift):
    return (v @ R0.T + T0) @ B.T + shift


def load_new_parts(B, shift):
    rows = [l.split('\t') for l in ELEMENTS.read_text(encoding='utf-8').splitlines()[1:]]
    files = {}
    for c, _, fj in rows:
        if c in NEW_PARTS:
            files.setdefault(c, []).append(fj.strip())
    meshes = {}
    with zipfile.ZipFile(BP3D_ZIP) as z:
        names = {Path(n).stem: n for n in z.namelist() if n.endswith('.obj')}
        for c, fjs in files.items():
            parts = []
            for fj in fjs:
                if fj in names:
                    V, F = read_obj(z.read(names[fj]))
                    parts.append(trimesh.Trimesh(bp3d_to_world(V, B, shift), F, process=False))
            if parts:
                meshes[NEW_PARTS[c]] = dict(mesh=trimesh.util.concatenate(parts), files=fjs)
    return meshes


def read_scene(args, cache):
    """Scene meshes, camera pose and skin parent frame; cached so the analysis can be re-run without the simulator."""
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        return z['data'].item()
    client, sim = connect(args)
    load_scene(client, sim, args, INSTRUMENT_SCENE, 'feasibility.ttt')
    by = aliases(sim)
    data = {}
    for n in SCENE_PARTS:
        h = by[n]
        V, F = sim.getShapeMesh(h)[:2]
        M = matrix_to_np(sim.getObjectMatrix(h, -1))
        data[n] = (np.asarray(V, float).reshape(-1, 3) @ M[:, :3].T + M[:, 3], np.asarray(F, np.int64).reshape(-1, 3))
    data['camera'] = matrix_to_np(sim.getObjectMatrix(by['EndoscopeCamera'], -1))
    data['skin_parent'] = matrix_to_np(sim.getObjectMatrix(sim.getObjectParent(by['Patient_Skin']), -1))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, data=np.array(data, dtype=object))
    return data


class Occluder:
    """Batched clearance tests against a set of meshes (ray for the centre line, sampled surface for the radius)."""

    def __init__(self, meshes, spacing=0.0015, ray_faces=60000):
        mesh = trimesh.util.concatenate(meshes)
        n = int(mesh.area / spacing ** 2)
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=1)
        self.tree = cKDTree(np.vstack([pts, mesh.vertices]))
        self.ray_mesh = mesh.simplify_quadric_decimation(face_count=ray_faces) if len(mesh.faces) > ray_faces else mesh
        self.mesh = mesh

    def clear(self, A, Bp, radius, skip_start=0.0, skip_end=0.0, step=0.002):
        A, Bp = np.atleast_2d(A), np.atleast_2d(Bp)
        d = Bp - A
        L = np.linalg.norm(d, axis=1)
        u = d / L[:, None]
        ok = np.ones(len(A), bool)
        locs, idx, _ = self.ray_mesh.ray.intersects_location(A + u * 1e-4, u, multiple_hits=False)
        if len(idx):
            hit = np.linalg.norm(locs - A[idx], axis=1) < L[idx] - 1e-4
            ok[idx[hit]] = False
        if radius > 0:
            idx = np.nonzero(ok)[0]
            counts = np.maximum(np.floor((L[idx] - skip_end - skip_start) / step).astype(int), 0)
            keep = counts > 0
            idx, counts = idx[keep], counts[keep]
            if len(idx):
                seg = np.repeat(np.arange(len(idx)), counts)
                s = skip_start + step * (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
                P = A[idx][seg] + s[:, None] * u[idx][seg]
                dist, _ = self.tree.query(P, distance_upper_bound=radius)
                blocked = np.zeros(len(idx), bool)
                np.logical_or.at(blocked, seg, np.isfinite(dist))
                ok[idx[blocked]] = False
        return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--target-spacing-m', type=float, default=0.003)
    p.add_argument('--port-grid-m', type=float, default=0.010)
    p.add_argument('--cache', type=Path, default=None, help='npz cache of the scene data (re-used when present)')
    args = p.parse_args()
    t_start = time.time()
    plan = read_json(REPO / 'config' / 'registered_plan.json')
    B, shift = np.array(plan['rotation']), np.array(plan['shift'])
    anterior = B[:, 2]
    rcm = read_json(REPO / 'config' / 'instruments' / 'rcm_config.json')
    ecm_port = np.array(next(a for a in rcm if a['arm'] == 'ecm')['rcm_world'])

    data = read_scene(args, args.cache)
    scene = {n: trimesh.Trimesh(*data[n], process=False) for n in SCENE_PARTS}
    cam, Mp = data['camera'], data['skin_parent']

    # Transform check: the scene prostate against BodyParts3D FJ3139 through the same transform.
    with zipfile.ZipFile(BP3D_ZIP) as z:
        Vp, _ = read_obj(z.read(next(n for n in z.namelist() if n.endswith('FJ3139.obj'))))
    transform_error = float(cKDTree(scene['Patient_Prostate'].vertices).query(bp3d_to_world(Vp, B, shift))[0].max())
    if transform_error > 1e-4:
        raise SystemExit(f'BodyParts3D -> world transform does not reproduce the scene prostate ({transform_error} m)')
    new = load_new_parts(B, shift)
    print('parts loaded', {k: len(v['mesh'].faces) for k, v in new.items()}, f't={time.time() - t_start:.0f}s', flush=True)

    # Targets: lateral / anterolateral prostate surface, 2 mm outside.
    pro = scene['Patient_Prostate']
    c = pro.vertices.mean(axis=0)
    lr = B @ R0 @ np.array([1.0, 0, 0])                 # BodyParts3D x axis (patient left-right)
    lr = unit(lr - np.dot(lr, anterior) * anterior)
    sup = np.cross(lr, anterior)
    pts, fid = trimesh.sample.sample_surface(pro, 40000, seed=2)
    keys = np.round(pts / args.target_spacing_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    pts, fid = pts[first], fid[first]
    nrm = pro.face_normals[fid]
    nrm = nrm * np.sign(np.mean(((pts - c) * nrm).sum(axis=1)))
    sel = (np.abs(nrm @ lr) > 0.55) & (nrm @ anterior > -0.35)
    targets = pts[sel] + 0.002 * nrm[sel]
    side = np.where((targets - c) @ lr > 0, '+x', '-x')
    angle = np.degrees(np.arctan2((targets - c) @ lr, (targets - c) @ anterior))   # 0 = anterior, +-90 = lateral
    level = (targets - c) @ sup

    # Candidate ports on the insufflated skin, anterior side.
    w = np.load(REPO / 'models' / 'anatomy' / 'insufflated_wall.npz')
    skin = w['outer_vertices_parent'] @ Mp[:, :3].T + Mp[:, 3]
    keys = np.round(skin / args.port_grid_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    skin = skin[np.sort(first)]
    ports = skin[((skin - c) @ anterior > 0.05) & (np.linalg.norm(skin - ecm_port, axis=1) >= 0.06)]
    print(f'{len(targets)} targets, {len(ports)} candidate ports', flush=True)

    base = [scene[n] for n in SCENE_PARTS if n != 'Patient_Bladder']
    others = [m['mesh'] for m in new.values()]
    cases = {'bladder_in_place': base + others + [scene['Patient_Bladder']], 'bladder_removed': base + others}

    # Scope candidates (shared by every target): tip positions from the ECM port.
    results = {}
    for case, meshes in cases.items():
        occ = Occluder(meshes)
        vis = np.zeros(len(targets), bool)
        reach = np.zeros(len(targets), int)
        trio_out = [None] * len(targets)
        for ti, t in enumerate(targets):
            to_t = unit(t - ecm_port)
            k1 = unit(np.cross(to_t, [0, 0, 1.0]))
            k2 = np.cross(to_t, k1)
            tips, axes = [], []
            for depth in (0.05, 0.08, 0.11, 0.14, 0.17):
                for tilt in (0.0, 15.0, 30.0, 45.0):
                    for az in (range(0, 360, 45) if tilt else [0]):
                        a = math.radians(az)
                        d = unit(to_t * math.cos(math.radians(tilt)) + math.sin(math.radians(tilt)) *
                                 (k1 * math.cos(a) + k2 * math.sin(a)))
                        tip = ecm_port + depth * d
                        view = t - tip
                        dist = np.linalg.norm(view)
                        if 0.025 <= dist <= 0.15 and np.degrees(np.arccos(np.clip(view @ d / dist, -1, 1))) <= 65:
                            tips.append(tip)
                            axes.append(d)
            if tips:
                tips = np.array(tips)
                sight = occ.clear(tips, np.repeat(t[None], len(tips), axis=0), 0.0)
                if sight.any():
                    path = occ.clear(np.repeat(ecm_port[None], sight.sum(), axis=0), tips[sight], 0.008, skip_start=0.03)
                    vis[ti] = bool(path.any())
            L = np.linalg.norm(t - ports, axis=1)
            cand = ports[(L >= 0.06) & (L <= 0.22)]
            if len(cand):
                ok = occ.clear(cand, np.repeat(t[None], len(cand), axis=0), 0.006, skip_start=0.02, skip_end=0.012)
                good = cand[ok]
                reach[ti] = len(good)
                if len(good) >= 3:
                    dirs = unit(t - good)
                    rng = np.random.default_rng(ti)
                    for _ in range(400):
                        trio = rng.choice(len(good), 3, replace=False)
                        if all(np.linalg.norm(good[i] - good[j]) >= 0.06 and
                               20 <= math.degrees(math.acos(np.clip(dirs[i] @ dirs[j], -1, 1))) <= 100
                               for i, j in itertools.combinations(trio, 2)):
                            trio_out[ti] = good[trio].tolist()
                            break
            if ti % 20 == 0:
                print(f'  {case} target {ti}/{len(targets)} t={time.time() - t_start:.0f}s', flush=True)
        tri = np.array([x is not None for x in trio_out])
        feasible = vis & tri
        summary = {s: dict(targets=int((side == s).sum()), visible=int((vis & (side == s)).sum()),
                           reachable_by_3=int((tri & (side == s)).sum()), feasible=int((feasible & (side == s)).sum()))
                   for s in ('+x', '-x')}
        best = None
        if feasible.any():
            k = int(np.argmax(np.where(feasible, reach, -1)))
            best = dict(target=targets[k].tolist(), side=str(side[k]), angle_from_anterior_deg=float(angle[k]),
                        level_from_centre_m=float(level[k]), ports=trio_out[k], ports_reaching=int(reach[k]))
        results[case] = dict(summary=summary, visible=vis.tolist(), triangulable=tri.tolist(),
                             ports_reaching=reach.tolist(), feasible=feasible.tolist(), best=best)
        print(case, summary, 'best', best and {k: v for k, v in best.items() if k != 'ports'}, flush=True)

    write_json(OUT, dict(
        _note='Generated by scripts/analysis/prostate_access_feasibility.py. Read-only geometric feasibility; '
              'no deformation, no gravity shift of organs; bladder removal is an optimistic bound.',
        transform_check_max_error_m=transform_error,
        new_parts={k: v['files'] for k, v in new.items()},
        axes=dict(left_right=lr.tolist(), anterior=anterior.tolist(), superior=sup.tolist(), prostate_centre=c.tolist()),
        targets=dict(points=targets.tolist(), side=side.tolist(), angle_from_anterior_deg=angle.tolist(),
                     level_from_centre_m=level.tolist()),
        ecm_port=ecm_port.tolist(), candidate_ports=int(len(ports)), cases=results,
        seconds=round(time.time() - t_start, 1)))
    render(cam, scene, new, pro, targets, results, base)
    print('wrote', OUT, FIG, f't={time.time() - t_start:.0f}s')


def render(cam, scene, new, pro, targets, results, base):
    """Painter's-algorithm view from the scene's endoscope camera, prostate coloured by result, per case."""
    W, H = 640, 480
    f = (W / 2) / math.tan(math.radians(35))
    R, tcam = cam[:, :3], cam[:, 3]
    tiles = []
    for case, res in results.items():
        im = Image.new('RGB', (W, H + 26), 'white')
        d = ImageDraw.Draw(im)
        ctx = [(m, (205, 200, 185)) for m in base if m is not pro] + [(m['mesh'], (190, 120, 110)) for m in new.values()]
        if case == 'bladder_in_place':
            ctx.append((scene['Patient_Bladder'], (230, 200, 90)))
        tris, cols = [], []
        for m, col in ctx:
            tris.append(m.vertices[m.faces])
            cols += [col] * len(m.faces)
        fe, vv = np.array(res['feasible']), np.array(res['visible'])
        dist, k = cKDTree(targets).query(pro.triangles_center)
        for di, ki in zip(dist, k):
            cols.append((40, 170, 60) if di < 0.004 and fe[ki] else (240, 150, 40) if di < 0.004 and vv[ki]
                        else (200, 60, 60) if di < 0.004 else (150, 150, 165))
        tris.append(pro.vertices[pro.faces])
        T = np.vstack(tris)
        q = (T - tcam) @ R
        z = q[:, :, 2]
        good = (z > 0.005).all(axis=1)
        uv = np.stack([W / 2 - f * q[:, :, 0] / np.maximum(z, 1e-6), H / 2 - f * q[:, :, 1] / np.maximum(z, 1e-6)], axis=2)
        for i in np.argsort(-z.mean(axis=1)):
            if good[i] and np.abs(uv[i]).max() < 4 * W:
                d.polygon([tuple(x) for x in uv[i]], fill=cols[i])
        d.rectangle([0, H, W, H + 26], fill='white')
        s = res['summary']
        d.text((6, H + 6), f"{case}: feasible +x {s['+x']['feasible']}/{s['+x']['targets']}, -x {s['-x']['feasible']}/"
                           f"{s['-x']['targets']}  green feasible / orange visible only / red not visible", fill='black')
        tiles.append(im)
    sheet = Image.new('RGB', (W * len(tiles), H + 26), 'white')
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * W, 0))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(FIG)


if __name__ == '__main__':
    main()
