"""M3a: retracted-bladder feasibility sweep + port replanning (analysis only, no scene build).

Modelling assumption (user decision): the bladder stays in the scene as an occluder and protected
structure, RIGIDLY displaced to a retracted position — it is never deleted. No deformation; the
urethra/bladder-neck continuity is broken by the rigid displacement (recorded as a limitation).

Protocol (written into the report BEFORE any evaluation — do not change after seeing results):
  * retraction direction d_r = normalise(cranial + anterior) in the patient frame. The frame is
    derived exactly as in prostate_access_wrist.py: anterior = registered_plan B[:,2] (the skin
    registration's outward axis), lr = B @ R0 @ +x re-orthogonalised, cranial = lr x anterior,
    sign-checked so the bladder centroid is cranial-anterior of the prostate centroid.
  * magnitudes 0, 5, ..., 60 mm, translation only, no rotation.
  * admissibility per magnitude: penetration of the displaced bladder into each hip bone, the
    sacrum and the insufflated abdominal wall <= 2 mm (max penetration depth reported per
    structure; signed distance for the watertight bones, oriented-normal depth for the wall
    shell). Distances to prostate and urethra reported for information.
  * per admissible magnitude: the same wrist-aware evaluation as prostate_access_wrist.py
    (same targets/ports/camera/margins/polyline model), conservative (a) and optimistic (b)
    separately.
  * selection: the SMALLEST admissible magnitude with >= 1 target triangulable under (a).
  * port plan at the selected magnitude: t* = the triangulable target maximising the minimum
    per-role reaching-port count (tie -> nearest lateral mid-gland, min |(t-c).lr|). One port per
    role on the same skin grid, keeping each arm on its current side; trio must satisfy the
    original spacing/meeting-angle rules; score = min over roles of clearance margin along the
    polyline (m) - 0.0005 * sum over pairs |meet_deg - 60|. ECM port unchanged.
  * realisability: plan rules from config/instruments/port_plan.json (depth range, spacing,
    side) — the setup-joint IK itself is plan_psm_ports' job (M3b); this report gives the
    geometric verdict.
"""
import argparse
import itertools
import math
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS.parent / 'build'))
sys.path.insert(0, str(ANALYSIS))

from simtools import REPO, add_arguments, relative, sha256, write_json  # noqa: E402
import prostate_access_feasibility as pa  # noqa: E402
import prostate_access_wrist as paw  # noqa: E402

OUT = REPO / 'validation' / 'feasibility' / 'bladder_retraction.json'
FIG = REPO / 'docs' / 'feasibility_bladder_retraction.png'
CACHE = REPO / 'validation' / 'feasibility' / 'prostate_access_wrist_cache.npz'
INSUFFLATED = REPO / 'models' / 'anatomy' / 'insufflated_wall.npz'

MAG_MM = list(range(0, 61, 5))
PEN_LIMIT_M = 0.002
TOP_K = 12                      # per-role margin shortlist for the trio search
ANGLE_WEIGHT_M_PER_DEG = 0.0005 / 60.0
unit = pa.unit


def penetration_depth(mesh, verts):
    """Max depth (m) that `verts` penetrate `mesh`; 0 if none.

    Watertight meshes: `contains` decides inside; depth = nearest-surface distance of the
    contained verts (this trimesh's signed_distance sign convention is unreliable).
    Open shells (the abdominal wall dome): see shell_penetration.
    """
    if mesh.is_watertight:
        inside = mesh.contains(verts)
        if not inside.any():
            return 0.0
        _, dist, _ = mesh.nearest.on_surface(verts[inside])
        return float(dist.max())
    return None


def shell_penetration(shell, verts, interior_point):
    """Penetration through an open shell (the insufflated wall's inner surface is a local dome,
    not a closed boundary). A vertex is 'outside' iff the ray from it toward `interior_point`
    hits the wall before reaching the interior; penetration depth is then the hit distance.
    Verts outside the dome's footprint (rays missing the wall) are not penetrating."""
    d = interior_point - verts
    L = np.linalg.norm(d, axis=1)
    u = d / np.maximum(L, 1e-12)[:, None]
    locs, ridx, _ = shell.ray.intersects_location(verts, u, multiple_hits=True)
    if not len(ridx):
        return 0.0
    hit = np.linalg.norm(locs - verts[ridx], axis=1)
    hit = hit[hit < L[ridx] - 1e-6]
    return float(hit.max()) if len(hit) else 0.0


def surface_distance(mesh, verts):
    """Min distance between `verts` and `mesh` surface (samples + vertices)."""
    pts = np.vstack([mesh.vertices, trimesh.sample.sample_surface(mesh, 4000, seed=3)[0]])
    return float(cKDTree(pts).query(verts)[0].min())


def protocol():
    return dict(
        modelling_assumption=('bladder stays in the scene as occluder + protected structure, '
                              'rigidly displaced to a retracted position; never deleted; no '
                              'deformation; urethra/bladder-neck continuity broken (limitation).'),
        retraction_direction='d_r = normalise(cranial + anterior), patient frame as in '
                             'prostate_access_wrist (anterior = plan B[:,2], cranial = lr x anterior '
                             'with sign check: bladder centroid cranial-anterior of prostate)',
        magnitudes_mm=MAG_MM, rotation='none (translation only)',
        admissibility=dict(penetration_limit_m=PEN_LIMIT_M,
                           structures=['Patient_RightHipBone', 'Patient_LeftHipBone',
                                       'Patient_Sacrum', 'insufflated_inner_wall'],
                           wall_note='inner wall is an open dome: a vertex is outside iff its '
                                     'ray to the interior reference hits the wall first'),
        evaluation='identical to prostate_access_wrist: targets, ports grid, camera visibility, '
                   'margins, polyline model; (a) conservative and (b) optimistic separate',
        selection='smallest admissible magnitude with >=1 triangulable target under (a)',
        port_plan=dict(target_choice='max min-role reaching count; tie -> min |(t-c).lr|',
                       trio_rules='>=60 mm spacing pairwise + >=60 mm from ECM; meeting 20-100 deg',
                       side_rule='each role stays on its current arm side (sign of (port-c).lr)',
                       score='min_role margin_m - %.5f * sum_pairs |meet_deg - 60|' % (ANGLE_WEIGHT_M_PER_DEG * 60),
                       top_k_per_role=TOP_K,
                       realisability='port_plan.json rules check; setup IK deferred to M3b build'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--bp3d-zip', type=Path, default=pa.BP3D_ZIP)
    p.add_argument('--elements', type=Path, default=pa.ELEMENTS)
    p.add_argument('--insufflated', type=Path, default=INSUFFLATED)
    p.add_argument('--baseline', type=Path, default=paw.BASELINE)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--fig', type=Path, default=FIG)
    args = p.parse_args()
    t_start = time.time()
    pa.ELEMENTS, pa.BP3D_ZIP = args.elements, args.bp3d_zip

    plan = pa.read_json(REPO / 'config' / 'registered_plan.json')
    B, shift = np.array(plan['rotation']), np.array(plan['shift'])
    anterior = B[:, 2]
    rcm = pa.read_json(REPO / 'config' / 'instruments' / 'rcm_config.json')
    ecm_port = np.array(next(a for a in rcm if a['arm'] == 'ecm')['rcm_world'])
    current_ports = {a['arm']: np.array(a['rcm_world']) for a in rcm if a['arm'] in paw.ARMS}
    roles_cfg = pa.read_json(REPO / 'config' / 'task' / 'cut_task.json')['roles']
    tools = paw.load_tools(roles_cfg)
    port_plan = pa.read_json(REPO / 'config' / 'instruments' / 'port_plan.json')

    payload = paw.read_inputs(args)
    data, wrist = payload['scene'], payload['wrist']
    for role in paw.ROLES:
        tools[role]['ld_m'] = wrist[tools[role]['arm']]['ld_m']
    scene = {n: trimesh.Trimesh(*data[n], process=False) for n in pa.SCENE_PARTS}
    try:
        import embreex  # noqa: F401
        ray_engine = 'ray_pyembree'
    except ImportError:
        ray_engine = 'trimesh default (embreex not installed)'

    new = pa.load_new_parts(B, shift)
    pro, bladder = scene['Patient_Prostate'], scene['Patient_Bladder']
    c = pro.vertices.mean(axis=0)
    lr = unit(B @ pa.R0 @ np.array([1.0, 0, 0]))
    lr = unit(lr - np.dot(lr, anterior) * anterior)
    sup = np.cross(lr, anterior)
    bc, pc = bladder.vertices.mean(axis=0), c
    sign = np.sign((bc - pc) @ sup) or 1.0
    sup = sup * sign                    # cranial points from prostate toward the bladder
    frame_check = dict(bladder_minus_prostate_dot_sup_m=float((bc - pc) @ sup),
                       bladder_minus_prostate_dot_anterior_m=float((bc - pc) @ anterior))
    d_r = unit(sup + anterior)

    w = np.load(args.insufflated)
    Mp = data['skin_parent']
    skin = w['outer_vertices_parent'] @ Mp[:, :3].T + Mp[:, 3]
    inner_wall = trimesh.Trimesh(w['inner_vertices_parent'] @ Mp[:, :3].T + Mp[:, 3],
                                 w['inner_faces'], process=False)
    keys = np.round(skin / 0.010).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    skin = skin[np.sort(first)]
    ports_pre = skin[((skin - c) @ anterior > 0.05)]
    ports = ports_pre[np.linalg.norm(ports_pre - ecm_port, axis=1) >= paw.PORT_SEP]

    baseline = pa.read_json(args.baseline)
    targets = np.array(baseline['targets']['points'], float)
    side = np.array(baseline['targets']['side'])

    base = [scene[n] for n in pa.SCENE_PARTS if n != 'Patient_Bladder']
    others = [m['mesh'] for m in new.values()]
    urethra = next(m['mesh'] for k, m in new.items() if 'urethra' in k.lower()) if \
        any('urethra' in k.lower() for k in new) else None
    bones = {n: scene[n] for n in ('Patient_RightHipBone', 'Patient_LeftHipBone', 'Patient_Sacrum')}

    proto = protocol()
    proto['frame_derivation'] = dict(anterior=anterior.tolist(), lr=lr.tolist(),
                                     cranial=sup.tolist(), d_r=d_r.tolist(), **frame_check)
    report = dict(run_id=uuid.uuid4().hex, protocol=proto,
                  source_sha256={relative(x): sha256(x) for x in
                                 [Path(__file__), ANALYSIS / 'prostate_access_wrist.py',
                                  ANALYSIS / 'prostate_access_feasibility.py',
                                  REPO / 'config' / 'task' / 'cut_task.json',
                                  REPO / 'config' / 'instruments' / 'rcm_config.json',
                                  REPO / 'config' / 'instruments' / 'port_plan.json']},
                  params=dict(ray_engine=ray_engine, tools=tools,
                              candidate_ports=int(len(ports)), ecm_port=ecm_port.tolist(),
                              current_ports={k: v.tolist() for k, v in current_ports.items()}),
                  magnitudes=[], selection=None, port_plan=None, limitations=[
                      'rigid displacement: no bladder deformation; urethra/bladder-neck '
                      'continuity is broken geometrically',
                      'abdominal-wall penetration measured vs the 30 mm inner shell dome '
                      '(open surface; verts outside its footprint do not count)',
                      'exploratory geometric feasibility only; no clinical claim'],
                  seconds=None)

    for mm in MAG_MM:
        m = mm / 1000.0
        bverts = bladder.vertices + m * d_r
        pen = {n: penetration_depth(bones[n], bverts) for n in bones}
        pen['insufflated_inner_wall'] = shell_penetration(inner_wall, bverts, pc)
        admissible = all(v <= PEN_LIMIT_M + 1e-9 for v in pen.values())
        row = dict(magnitude_mm=mm, penetration_m=pen, admissible=bool(admissible),
                   dist_prostate_m=float(surface_distance(pro, bverts)),
                   dist_urethra_m=(float(surface_distance(urethra, bverts)) if urethra is not None else None))
        if admissible:
            disp = trimesh.Trimesh(bverts, bladder.faces, process=False)
            occ = pa.Occluder(base + others + [disp])
            res = paw.evaluate_case(occ, targets, ports, ecm_port, tools, wrist, current_ports, t_start)
            row['visible'] = res['vis'].tolist()
            for v in ('conservative', 'optimistic'):
                var = res['variants'][v]
                row[v] = dict(reaching_ports_per_role={r: var['role_reach'][r].tolist() for r in paw.ROLES},
                              triangulable=var['triangulable'].tolist(),
                              feasible=(res['vis'] & var['triangulable']).tolist(),
                              trios=var['trios'])
            row['n_triangulable_a'] = int(res['variants']['conservative']['triangulable'].sum())
        report['magnitudes'].append(row)
        print(f'{mm} mm admissible={admissible} pen_max={max(pen.values()) * 1000:.2f} mm '
              f't={time.time() - t_start:.0f}s', flush=True)

    # ---- pre-registered selection ----------------------------------------------------------
    selected = next((r for r in report['magnitudes']
                     if r['admissible'] and r.get('conservative') and
                     any(r['conservative']['triangulable'])), None)
    report['selection'] = dict(magnitude_mm=selected['magnitude_mm'] if selected else None,
                               rule='smallest admissible magnitude with >=1 triangulable target under (a)')
    if selected is None:
        report['seconds'] = round(time.time() - t_start, 1)
        write_json(args.out, report)
        print('INFEASIBLE: no admissible magnitude triangulable under (a); stopping.')
        render_fig(report, args.fig)
        return

    # ---- port replanning -----------------------------------------------------------------
    # At the pre-registered magnitude first; if no side-respecting trio exists at any of its
    # triangulable targets, continue through subsequent admissible magnitudes with tri > 0 in
    # ascending order ("next-best trio" per the brief) and record the deviation.
    arm_sides = {role: np.sign((current_ports[tools[role]['arm']] - c) @ lr) or 1.0 for role in paw.ROLES}

    def margin_of(r, ti):
        """Clearance margin (m) of one accepted polyline: min sample distance to the occluder's
        surface KDTree minus the 6 mm shaft radius. Same sampling convention as clear_poly."""
        P = np.asarray(r['port']); W = np.asarray(r['wrist'])
        s = unit(W - P); L1 = np.linalg.norm(W - P)
        n1 = max(int((L1 - paw.WALL_SKIP) / paw.STEP), 0)
        S1 = P + paw.WALL_SKIP + s * np.arange(n1)[:, None] * paw.STEP if n1 else np.empty((0, 3))
        d2 = unit(targets[ti] - W); L2 = np.linalg.norm(targets[ti] - W)
        n2 = max(int((L2 - paw.TIP_SKIP) / paw.STEP), 0)
        S2 = W + d2 * np.arange(n2)[:, None] * paw.STEP if n2 else np.empty((0, 3))
        S = np.vstack([S1, S2]) if len(S1) + len(S2) else np.array([W])
        return float(occ.tree.query(S)[0].min()) - paw.SHAFT_RADIUS

    def find_trio(row, occ_):
        """Search row's triangulable targets (min-role desc, lateral-centre tie-break) for a
        side-respecting trio. Returns (ti, combo, score, margins, attempted) or None."""
        nonlocal occ
        occ = occ_
        con = row['conservative']
        tri_idx = [i for i, ok in enumerate(con['triangulable']) if ok]
        min_role = {ti: min(con['reaching_ports_per_role'][r][ti] for r in paw.ROLES) for ti in tri_idx}
        order = sorted(tri_idx, key=lambda ti: (-min_role[ti], abs((targets[ti] - c) @ lr)))
        attempted = []
        for ti in order:
            recs_by_role, margins, side_filtered = {}, {}, {}
            for role in paw.ROLES:
                recs = paw.reach_poly(occ, ports, targets[ti], tools[role]['ld_m'],
                                      tools[role]['theta_conservative_rad'])
                kept = [x for x in enumerate(recs)
                        if x[1] is not None
                        and np.sign((ports[x[0]] - c) @ lr) == arm_sides[role]
                        and np.linalg.norm(ports[x[0]] - ecm_port) >= paw.PORT_SEP - 1e-12]
                side_filtered[role] = len(kept)
                scored = sorted(((i, r, margin_of(r, ti)) for i, r in kept),
                                key=lambda x: -x[2])
                recs_by_role[role] = scored[:TOP_K]
                margins[role] = [round(x[2], 5) for x in scored[:TOP_K]]
            attempted.append(dict(target=int(ti), same_side_reaching=side_filtered))
            best_t, best_t_score = None, -np.inf
            for combo in itertools.product(*(recs_by_role[r] for r in paw.ROLES)):
                pset = [np.asarray(x[1]['port']) for x in combo]
                if len(set(x[0] for x in combo)) < 3:
                    continue
                sdirs = [unit(np.asarray(x[1]['wrist']) - np.asarray(x[1]['port'])) for x in combo]
                meets = [math.degrees(math.acos(np.clip(sdirs[a] @ sdirs[b], -1, 1)))
                         for a, b in itertools.combinations(range(3), 2)]
                if not all(np.linalg.norm(pset[a] - pset[b]) >= paw.PORT_SEP - 1e-12 and
                           paw.MEET_DEG[0] <= meets[k] <= paw.MEET_DEG[1]
                           for k, (a, b) in enumerate(itertools.combinations(range(3), 2))):
                    continue
                score = min(x[2] for x in combo) - ANGLE_WEIGHT_M_PER_DEG * 60 * \
                    sum(abs(x - 60.0) for x in meets) / 60.0
                if score > best_t_score:
                    best_t, best_t_score = combo, score
            attempted[-1]['trio_found'] = best_t is not None
            if best_t is not None:
                return ti, best_t, best_t_score, margins, attempted
        return None, attempted

    m = selected['magnitude_mm'] / 1000.0
    occ = pa.Occluder(base + others + [trimesh.Trimesh(bladder.vertices + m * d_r,
                                                       bladder.faces, process=False)])
    con = selected['conservative']
    tri_idx = [i for i, ok in enumerate(con['triangulable']) if ok]
    min_role = {ti: min(con['reaching_ports_per_role'][r][ti] for r in paw.ROLES) for ti in tri_idx}
    best_min = max(min_role.values())
    cand_t = [ti for ti in tri_idx if min_role[ti] == best_min]
    t_star = min(cand_t, key=lambda ti: abs((targets[ti] - c) @ lr))
    report['selection']['t_star'] = dict(index=int(t_star), point=targets[t_star].tolist(),
                                         side=str(side[t_star]), min_role_ports=int(best_min),
                                         lateral_mid_distance_m=float(abs((targets[t_star] - c) @ lr)))

    found, used_mm, attempted_all = None, None, []
    sel_idx = report['magnitudes'].index(selected)
    for row in report['magnitudes'][sel_idx:]:
        if not (row['admissible'] and row.get('n_triangulable_a')):
            continue
        mm = row['magnitude_mm']
        o = pa.Occluder(base + others + [trimesh.Trimesh(bladder.vertices + (mm / 1000.0) * d_r,
                                                         bladder.faces, process=False)])
        found = find_trio(row, o)
        attempted_all.append(dict(magnitude_mm=mm,
                                  targets=found[4] if found[0] is not None else found[1]))
        if found[0] is not None:
            used_mm = mm
            found = found[:4]
            break
        found = None
    rules = port_plan['rules']
    plan_out = dict(t_star=report['selection']['t_star'],
                    used_magnitude_mm=used_mm, used_target=int(found[0]) if found else None,
                    attempted=attempted_all,
                    ecm_port=ecm_port.tolist(),
                    arm_side_signs={r: int(arm_sides[r]) for r in paw.ROLES},
                    margins_top_m=found[3] if found else {},
                    trio=None, realisability=None)
    if found:
        chosen, best, best_score = found[0], found[1], found[2]
    if best is not None:
        trio_ports = [np.asarray(x[1]['port']) for x in best]
        sdirs = [unit(np.asarray(x[1]['wrist']) - np.asarray(x[1]['port'])) for x in best]
        meets = [math.degrees(math.acos(np.clip(sdirs[a] @ sdirs[b], -1, 1)))
                 for a, b in itertools.combinations(range(3), 2)]
        depth_lo, depth_hi = rules['depth_range_m']
        reach_t = [float(np.linalg.norm(p - targets[chosen])) for p in trio_ports]
        verdict = dict(
            port_target_distance_m=[round(x, 4) for x in reach_t],
            depth_range_required=list(rules['depth_range_m']),
            in_depth_range=[bool(depth_lo <= x <= depth_hi) for x in reach_t],
            spacing_ok=bool(all(np.linalg.norm(trio_ports[a] - trio_ports[b]) >= rules['min_port_spacing_m']
                                for a, b in itertools.combinations(range(3), 2))),
            ecm_clear_ok=bool(all(np.linalg.norm(p - ecm_port) >= rules['min_port_spacing_m']
                                  for p in trio_ports)),
            note=('ports come from the same skin grid the current realised ports used; the '
                  'setup-joint IK is plan_psm_ports work (M3b). depth_range/spacing rules are '
                  'the planner criteria recorded in port_plan.json.'))
        verdict['nominally_realisable'] = bool(all(verdict['in_depth_range']) and
                                               verdict['spacing_ok'] and verdict['ecm_clear_ok'])
        plan_out['trio'] = [dict(role=paw.ROLES[i], arm=tools[paw.ROLES[i]]['arm'],
                                 port=best[i][1]['port'], wrist=best[i][1]['wrist'],
                                 bend_rad=best[i][1]['bend_rad'], margin_m=round(best[i][2], 5))
                            for i in range(3)]
        plan_out['meeting_angles_deg'] = [round(x, 1) for x in meets]
        plan_out['score'] = float(best_score)
        plan_out['realisability'] = verdict
    report['port_plan'] = plan_out
    report['seconds'] = round(time.time() - t_start, 1)
    write_json(args.out, report)
    render_fig(report, args.fig)
    print('wrote', args.out, args.fig, f"t={time.time() - t_start:.0f}s")


def render_fig(report, out):
    from PIL import Image, ImageDraw
    W, H, PL, PT = 900, 420, 60, 30
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    rows = report['magnitudes']
    xs = [r['magnitude_mm'] for r in rows]
    pen = [max(r['penetration_m'].values()) * 1000 for r in rows]
    tri_a = [int(sum(r.get('conservative', {}).get('triangulable', []))) for r in rows]
    tri_b = [int(sum(r.get('optimistic', {}).get('triangulable', []))) for r in rows]
    px = lambda x: PL + (W - PL - 20) * x / max(xs)
    ypen = lambda y: PT + 150 - 140 * min(y, 8) / 8
    ytri = lambda y: PT + 340 - 130 * y / 8
    d.text((10, 8), 'Bladder retraction sweep: penetration (top, mm) vs triangulable count (bottom)')
    for r, x, y in zip(rows, xs, pen):
        d.ellipse([px(x) - 3, ypen(y) - 3, px(x) + 3, ypen(y) + 3],
                  fill=(40, 140, 60) if r['admissible'] else (220, 40, 40))
    d.line([(px(a), ypen(b)) for a, b in zip(xs, pen)], fill=(150, 150, 150))
    d.line([PL, ypen(2), W - 20, ypen(2)], fill=(220, 40, 40), width=2)
    d.text((PL - 52, ypen(2) - 6), '2mm limit', fill=(220, 40, 40))
    for yv, col, lab in ((tri_a, (30, 100, 230), 'conservative (a)'),
                         (tri_b, (240, 150, 40), 'optimistic (b)')):
        d.line([(px(a), ytri(b)) for a, b in zip(xs, yv)], fill=col, width=2)
        for x, y in zip(xs, yv):
            d.ellipse([px(x) - 3, ytri(y) - 3, px(x) + 3, ytri(y) + 3], fill=col)
    d.text((PL + 10, PT + 330), 'a (blue) / b (orange)', fill='black')
    sel = (report.get('selection') or {}).get('magnitude_mm')
    if sel is not None:
        d.line([px(sel), PT, px(sel), PT + 340], fill=(120, 60, 200), width=2)
        d.text((px(sel) + 4, PT + 4), f'selected {sel} mm', fill=(120, 60, 200))
    for x in xs[::2]:
        d.text((px(x) - 8, PT + 352), f'{x}mm', fill=(90, 90, 90))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


if __name__ == '__main__':
    main()
