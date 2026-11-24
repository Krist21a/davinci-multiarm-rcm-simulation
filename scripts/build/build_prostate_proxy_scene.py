"""M3b: build scenes/davinci_si_prostate_proxy.ttt from davinci_si_articulated_verified.ttt.

Staged copy only — the source scene is never modified or re-saved.

  1. Patient_Bladder (and children) displaced rigidly 30 mm along d_r from
     validation/feasibility/bladder_retraction.json (M3a plan). Recorded in the scene config.
  2. psm1/psm2/psm3 re-docked to the M3a port plan: URDF PoseTask IK puts each arm's joint8 shaft
     frame on the port->aim line with the insertion joint at home (same mechanism as
     plan_psm_ports.py / install_instruments.apply_port_plan). Contacts of the moved arm are
     checked with the runtime collision pairs; a port that cannot be realised stops the build.
  3. ROI proxy sheet (validation/anatomy/roi_proxy.json + config/scenes/<stem>/roi_proxy.json):
     curved 24x14 sheet offset +2 mm from the prostate surface around t*.
  4. Scene-keyed config dir config/scenes/davinci_si_prostate_proxy/ (rcm_config, collision
     pairs, roi_proxy, anatomy transform). The controller's embedded CONFIG is rewritten to the
     new ports; COLLISION_PAIRS keeps the existing labels (same object names); INSTRUMENTS kept.

Aims: psm2 -> the proxy cut start (near-edge midpoint, scissors approach), psm1/psm3 -> the
retraction grasp nodes (middle row, columns 1 and NU-2), ECM keeps its port (view offset to t*
recorded; SimWorld commands the scope at runtime).
"""
import argparse
import math
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

BUILD = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILD))

import ecm_pairs  # noqa: E402
import build_endoscope_scene as bes  # noqa: E402
from kinematics import Chain, PoseTask, unit  # noqa: E402
from simtools import (ARTICULATED_SCENE, CONFIG, REPO, URDF, add_arguments, aliases, connect,
                      load_scene, matrix_to_np, read_json, relative, save_scene, sha256,
                      stop_simulation, write_json, lua)  # noqa: E402

M3A = REPO / 'validation' / 'feasibility' / 'bladder_retraction.json'
ROI_OUT = REPO / 'validation' / 'anatomy' / 'roi_proxy.json'
PROXY_SCENE = REPO / 'scenes' / 'davinci_si_prostate_proxy.ttt'
SCENE_CFG = CONFIG / 'scenes' / 'davinci_si_prostate_proxy'
REPORT = REPO / 'validation' / 'build' / 'prostate_proxy_scene.json'
APPROACH_GAP_M = 0.004
OFFSET_M = 0.002
MIN_JOINT_MARGIN = 0.08             # plan_psm_ports: normalised RCM-joint margin at home
NU, NV, H = 24, 14, 0.001           # same grid as the flat sheet (scripts/task/workspace.py)


def shape_world(sim, h):
    V, F = sim.getShapeMesh(h)[:2]
    M = matrix_to_np(sim.getObjectMatrix(h, -1))
    return np.asarray(V, float).reshape(-1, 3) @ M[:, :3].T + M[:, 3], \
        np.asarray(F, np.int64).reshape(-1, 3)


def build_roi(prostate_v, prostate_f, bladder_c0, t_star, psm2_port):
    """The construction rule, exactly as specified:
    c = t* (2 mm outside the prostate surface); n = outward surface normal at the nearest
    surface point (area-weighted face normals within 3 mm); v = prostate longitudinal
    (base->apex PCA axis, sign via the ORIGINAL bladder side = base) projected on the tangent
    plane; u = n x v. Nodes follow the +2 mm offset surface: flat tangent-plane grid points
    projected onto the prostate surface then offset by the local normal. The near edge (row 0)
    faces the side the scissors approach from along v, decided by psm2's port.
    """
    import trimesh
    pro = trimesh.Trimesh(prostate_v, prostate_f, process=False)
    near, dist, tid = pro.nearest.on_surface(np.atleast_2d(t_star))
    c_surf, c_nrm = near[0], pro.face_normals[tid[0]]
    # n: area-weighted mean of face normals within 3 mm of the nearest surface point
    fid = np.nonzero(np.linalg.norm(pro.triangles_center - c_surf, axis=1) < 0.003)[0]
    w = pro.area_faces[fid]
    n = unit((pro.face_normals[fid] * w[:, None]).sum(axis=0))
    if n @ c_nrm < 0:
        n = -n
    # v: major PCA axis of the prostate vertices = longitudinal; sign base->apex where the
    # ORIGINAL (pre-retraction) bladder centroid marks the base side.
    _, _, Vt = np.linalg.svd(prostate_v - prostate_v.mean(axis=0), full_matrices=False)
    v = Vt[0]
    v = unit(v - (v @ n) * n)
    if v @ (pro.vertices.mean(axis=0) - bladder_c0) < 0:
        v = -v                                    # apex->base flip if needed: keep v -> apex far side
    # Near edge faces the scissors approach: psm2's port side along v.
    if (psm2_port - t_star) @ v > 0:
        v = -v
    u = unit(np.cross(n, v))
    # Flat grid in the tangent plane at the c surface point, then re-project to +2 mm offset.
    i = np.arange(NU) - (NU - 1) / 2.0
    j = np.arange(NV)
    c_on_line = c_surf - ((NV - 1) / 2.0) * H * v        # row (NV-1)/2 passes through c's foot point
    grid = c_on_line[None, None] + i[None, :, None] * H * u[None, None] + \
        j[:, None, None] * H * v[None, None]
    flat = grid.reshape(-1, 3)
    near2, _, tid2 = pro.nearest.on_surface(flat)
    nodes = (near2 + OFFSET_M * pro.face_normals[tid2]).reshape(NV, NU, 3)
    cut_start = nodes[0, (NU - 1) // 2]
    grasp = {'psm1': nodes[NV // 2, 1], 'psm3': nodes[NV // 2, NU - 2]}
    return dict(c=t_star, c_surface=c_surf, n=n, v=v, u=u, nodes=nodes,
                cut_start=cut_start, grasp=grasp)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--random-ik', type=int, default=25)
    p.add_argument('--local-iterations', type=int, default=120)
    args = p.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    m3a = read_json(M3A)
    d_r = np.array(m3a['protocol']['frame_derivation']['d_r'])
    used_mm = m3a['port_plan']['used_magnitude_mm']
    trio = {t['role']: t for t in m3a['port_plan']['trio']}
    roles = read_json(CONFIG / 'task' / 'cut_task.json')['roles']
    arm_port = {roles[r]: np.array(trio[r]['port']) for r in trio}
    arm_wrist = {roles[r]: np.array(trio[r]['wrist']) for r in trio}
    t_star = np.array(m3a['port_plan']['t_star']['point'])
    rcm = read_json(CONFIG / 'instruments' / 'rcm_config.json')
    pairs = read_json(CONFIG / 'instruments' / 'collision_pairs.json')['labels']
    plan0 = read_json(CONFIG / 'registered_plan.json')
    cfg = {a['arm']: a for a in rcm}

    report = dict(run_id=uuid.uuid4().hex, source_scene=str(relative(ARTICULATED_SCENE)),
                  source_scene_sha256=sha256(ARTICULATED_SCENE), m3a_report=relative(M3A),
                  retraction=dict(magnitude_mm=used_mm, d_r=d_r.tolist()),
                  roi=None, arms={}, unrealisable=None, seconds=None)
    client, sim = connect(args)
    load_scene(client, sim, args, ARTICULATED_SCENE, 'prostate_proxy_build.ttt')
    by = aliases(sim)

    # --- 1. bladder rigid retraction ----------------------------------------------------------
    bc0 = None
    for name in by:
        if 'Bladder' in name:
            if name == 'Patient_Bladder':
                pv, pf = shape_world(sim, by[name])
                bc0 = pv.mean(axis=0)
    moved = []
    for name in by:
        if 'Bladder' in name:
            sim.setObjectPosition(by[name], -1,
                                  (np.array(sim.getObjectPosition(by[name], -1)) + (used_mm / 1000.0) * d_r).tolist())
            moved.append(name)
    report['retraction']['moved_objects'] = moved

    # --- 2. ROI proxy -------------------------------------------------------------------------
    pv, pf = shape_world(sim, by['Patient_Prostate'])
    roi = build_roi(pv, pf, bc0, t_star, arm_port[roles['cut']])
    nodes_flat = roi['nodes'].reshape(-1, 3)
    import hashlib
    roi_hash = hashlib.sha256(np.round(nodes_flat, 9).tobytes()).hexdigest()
    roi_doc = dict(benchmark_kind='synthetic_proxy', validated=False,
                   label='not endopelvic fascia ground truth; not prostatic or Denonvilliers fascia',
                   inputs={'prostate_mesh_world_sha256': hashlib.sha256(np.round(pv, 9).tobytes()).hexdigest(),
                           'prostate_faces_sha256': hashlib.sha256(pf.tobytes()).hexdigest(),
                           't_star_m': t_star.tolist(), 'offset_m': OFFSET_M,
                           'retraction': report['retraction']},
                   rule=dict(centre='t* (2 mm outside the prostate surface)',
                             normal='area-weighted face normals within 3 mm of the nearest surface point',
                             cut_direction='prostate major PCA axis projected on the tangent plane; '
                                           'sign base->apex (original bladder side = base), then flipped '
                                           'if needed so the near edge faces the psm2 port',
                             grid=f'{NU}x{NV}, H={H} m; flat tangent grid re-projected to +{OFFSET_M*1000:.0f} mm '
                                  'offset surface; row 0 = near edge (scissors side)',
                             attachment='same as the flat full-mode sheet: far row + side columns from row 3'),
                   centre_m=roi['c'].tolist(), normal=roi['n'].tolist(),
                   v=roi['v'].tolist(), u=roi['u'].tolist(),
                   cut_start_m=roi['cut_start'].tolist(),
                   grasp_m={k: v.tolist() for k, v in roi['grasp'].items()},
                   nodes_m=[np.round(x, 9).tolist() for x in nodes_flat],
                   nodes_sha256=roi_hash)
    write_json(ROI_OUT, roi_doc)
    report['roi'] = dict(centre_m=roi_doc['centre_m'], normal=roi['n'].tolist(), v=roi['v'].tolist(),
                         u=roi['u'].tolist(), cut_start_m=roi['cut_start'].tolist(),
                         nodes_sha256=roi_hash, out=relative(ROI_OUT))

    # --- 3. re-dock the PSMs -------------------------------------------------------------------
    # The M3a plan is a bent polyline port -> W -> t* (wrist point W, distal bend bend_rad):
    # dock the SHAFT along port->W (the clearance-checked corridor). Ld = wrist-pitch-axis to
    # TCP, measured on the straight tool; tip_home = W + Ld*axis puts the wrist axis on W.
    urdf_root = ET.parse(URDF).getroot()
    urdf_limits = {j.get('name'): [float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))]
                   for j in urdf_root.findall('joint') if j.find('limit') is not None}

    def distal_links(arm):
        """Scene visual aliases for the links strictly distal of <arm>_joint9 (the wrist pitch
        axis) -- the unchecked distal segment of the M3a polyline model."""
        children = {}
        for j in urdf_root.findall('joint'):
            children.setdefault(j.find('parent').get('link'), []).append(j.find('child').get('link'))
        start = next(j.find('child').get('link') for j in urdf_root.findall('joint')
                     if j.get('name') == arm + '_joint9')
        out, stack = set(), [start]
        while stack:
            link = stack.pop()
            for ch in children.get(link, []):
                out.add(ch)
                stack.append(ch)
        aliases_d = set()
        for a in by:
            base = a[:-len('_visual')] if a.endswith('_visual') else a
            if base == start or any(base == l or base.startswith(l + '_') for l in out):
                aliases_d.add(a)
        return aliases_d
    # Dock small movers first: each accepted posture is kept, so the arm with the largest port
    # displacement is checked last, against neighbours already at their new poses.
    updates = {}
    dock_order = sorted(arm_port, key=lambda a: np.linalg.norm(arm_port[a] - np.array(cfg[a]['rcm_world'])))
    for arm in dock_order:
        c = cfg[arm]
        port, W_pt = arm_port[arm], arm_wrist[arm]
        axis = unit(W_pt - port)
        M8 = matrix_to_np(sim.getObjectMatrix(by[arm + '_joint8'], -1))
        tcp_len = float(np.dot(np.array(sim.getObjectPosition(by[f'{arm}_tcp'], -1)) - M8[:, 3], M8[:, 2]))
        ld = float(np.dot(np.array(sim.getObjectPosition(by[f'{arm}_tcp'], -1)) -
                          np.array(sim.getObjectPosition(by[arm + '_joint9'], -1)), M8[:, 2]))
        tip_home = W_pt + ld * axis
        p8 = tip_home - tcp_len * axis
        G = tip_home                           # task-motion reference (straight-tool tip)
        names = [arm + '_lift_joint'] + [j['name'] for j in c['joints']]
        lims = {n: plan0['joint_limit_revisions'].get(n, urdf_limits[n]) for n in names}
        for j in c['joints']:
            lims[j['name']] = j['limits']
        fixed = {arm + '_joint7': c['insertion_home'],
                 arm + '_joint8': sim.getJointPosition(by[arm + '_joint8'])}
        handles = [by[n] for n in names]
        x_now = np.array([sim.getJointPosition(h) for h in handles])
        chain = Chain(arm + '_rot_ref_respondable')
        taskk = PoseTask(chain, names, fixed, lims, np.zeros(3), [[0, 0, 1]])
        rcm_names = [j['name'] for j in c['joints']]
        probe = ecm_pairs.create_probe(sim)
        mine = [l for l in pairs if l[0].startswith(arm + '_') or l[1].startswith(arm + '_')]
        hp = [[by[a], by[b]] for a, b in mine]
        tool = c['tool_shape']

        s_w = float(np.dot(W_pt - port, axis))
        distal_contacts = []

        dl = distal_links(arm)

        def contacts():
            """Contacts of this arm. Forgiven: the shaft crossing its own skin port, and contacts
            on a distal (wrist->TCP) link with all points beyond W (the M3a plan leaves the bent
            distal segment unchecked, documented model optimism). Anything else fails."""
            bad, length = [], 0.0
            distal_contacts.clear()
            for i, pts in ecm_pairs.scan(sim, probe, hp):
                pair = mine[i]
                if pair == [tool, 'Patient_Skin'] and len(pts) and \
                        np.linalg.norm(pts - port, axis=1).max() <= c['port_radius_m']:
                    continue            # the shaft legitimately crosses its own port
                flat = pts.reshape(-1, 3)
                if pair[0] in dl and len(flat) and \
                        np.all(np.dot(flat - port, axis) > s_w + 0.002):
                    distal_contacts.append(pair)    # unchecked distal segment (documented)
                    continue
                bad.append(pair)
                n = len(pts) // 2
                if n:
                    length += float(np.linalg.norm(pts[0:2 * n:2] - pts[1:2 * n:2], axis=1).sum())
            return bad, length + 0.002 * len(bad)

        def set_x(x):
            sim.callScriptFunction('setJoints', probe, handles, [float(v) for v in x])

        def motion_check(x_home, p8_used):
            """Contact-free over the task motion, same scheme as plan_psm_ports.motion_check:
            shaft through the port, axial approach/retraction, +-2 deg corrections at aim depth."""
            rcm_dist = float(np.dot(port - p8_used, axis))   # port->joint8 distance outside
            tx = unit(np.cross(axis, [0, 0, 1.0]))
            ty = np.cross(axis, tx)
            home_q = dict(zip(names, x_home))
            fixed_m = {**fixed, arm + '_lift_joint': home_q[arm + '_lift_joint']}
            out = []
            axial = (-0.010, 0.016 if arm == roles['cut'] else APPROACH_GAP_M + 0.001)
            moves = [(0.0, 0.0, d) for d in axial] + \
                    [(s * 2.0, q * 2.0, 0.0) for s in (-1, 1) for q in (-1, 1)]
            for yaw, pitch, dins in moves:
                steps = 12
                xr = np.array([home_q[n] for n in rcm_names])
                r = 0.0
                for k in range(1, steps + 1):
                    f = k / steps
                    dk = unit(axis + math.tan(math.radians(f * yaw)) * tx +
                              math.tan(math.radians(f * pitch)) * ty)
                    t2 = PoseTask(chain, rcm_names, {**fixed_m, arm + '_joint7': c['insertion_home'] + f * dins},
                                  lims, np.zeros(3), [[0, 0, 1]])
                    xr, r = t2.solve(xr, port - dk * (rcm_dist - f * dins), [dk], max_nfev=200)
                    if r > 1e-6:
                        break
                if r > 1e-6:
                    out.append(dict(yaw=yaw, pitch=pitch, dins=dins, ik_residual=r))
                    set_x(x_home)
                    return False, out
                set_x(np.r_[home_q[arm + '_lift_joint'], xr])
                sim.callScriptFunction('setJoints', probe, [by[arm + '_joint7']],
                                       [float(c['insertion_home'] + dins)])
                bad, _ = contacts()
                sim.callScriptFunction('setJoints', probe, [by[arm + '_joint7']], [float(c['insertion_home'])])
                out.append(dict(yaw=yaw, pitch=pitch, dins=dins, contacts=bad))
                if bad:
                    set_x(x_home)
                    return False, out
            set_x(x_home)
            return True, out

        def search(p8_):
            """IK pool + local null-space search for a joint8 target on the port->W line.
            Returns (contact-free posture or None, pool for diagnostics)."""
            pool = []
            for k in range(args.random_ik):
                seed = x_now if k == 0 else rng.uniform(np.maximum(taskk.lo, -1.6),
                                                      np.minimum(taskk.hi, 1.6))
                x, r = taskk.solve(seed, p8_, [axis])
                if r < 1e-8:
                    set_x(x)
                    bad, cost = contacts()
                    pool.append([cost, x, bad])
            pool.sort(key=lambda row: row[0])
            for cost, x, bad in pool[:3]:
                for it in range(args.local_iterations):
                    if not bad and taskk.margin(x, 1) >= MIN_JOINT_MARGIN:
                        break
                    tx_, r = taskk.solve(x + rng.normal(0, (0.05, 0.15, 0.4)[it % 3], len(x)),
                                         p8_, [axis])
                    if r > 1e-8:
                        continue
                    set_x(tx_)
                    tb, tcost = contacts()
                    if tcost < cost or (tcost == cost == 0 and
                                        taskk.margin(tx_, 1) > taskk.margin(x, 1)):
                        cost, x, bad = tcost, tx_, tb
                if not bad:
                    return x, pool
            return None, pool

        # The M3a trio's wrist points converge ~12 mm apart, so the straight tools may
        # interpenetrate at their tips when all arms are docked. If so, dock with the wrist
        # `setback` mm short of W on the SAME port->W corridor (the FSM's insertion joint brings
        # it to W during the task; bounded by joint7 headroom so the planned depth stays
        # reachable). The port realisation is unchanged.
        j7_headroom = (plan0['joint_limit_revisions'].get(arm + '_joint7', urdf_limits[arm + '_joint7'])[1]
                       - c['insertion_home'])
        setbacks = [0.0] + [d for d in np.arange(0.005, 0.031, 0.005) if d <= j7_headroom]
        accepted = None
        motion_failure = None
        motion_ok = motion_res = p8_final = None
        setback_m = 0.0
        first_pool = None
        for sb in setbacks:
            p8_sb = p8 - sb * axis
            x_, pool = search(p8_sb)
            if first_pool is None:
                first_pool = pool
            if x_ is None:
                continue
            set_x(x_)
            # Advisory only (recorded, not gating): the +-2 deg motion clearance is
            # plan_psm_ports' port-SELECTION criterion. The M3a trio is fixed; its thin margins
            # (0.16-1.3 mm) were sized for the static polyline, so a lateral-correction graze is
            # reported as a realisability caveat instead of rejecting the dock.
            mok, mres = motion_check(x_, p8_sb)
            accepted, motion_ok, motion_res = x_, mok, mres
            motion_failure = None if mok else mres[-1]
            p8_final, setback_m = p8_sb, sb
            if sb:
                report.setdefault('setbacks', {})[arm] = sb
            break
        pool = first_pool or []
        if accepted is None and any('ecm_' in n for pair in (pool[0][2] if pool else []) for n in pair):
            # The only blocker is an ECM setup link: the ECM keeps its port and axis, but its
            # redundant elbow posture may be re-posed to clear the PSM carriage.
            def repose_ecm():
                ce = cfg['ecm']
                enames = ['ecm_lift_joint'] + [j['name'] for j in ce['joints']]
                elims = {n: plan0['joint_limit_revisions'].get(n, urdf_limits[n]) for n in enames}
                for j in ce['joints']:
                    elims[j['name']] = j['limits']
                efixed = {'ecm_insertion_joint': ce['insertion_home']}
                etask = PoseTask(Chain('ecm_insertion'), enames, efixed, elims,
                                 np.zeros(3), [[0, 0, 1]])
                eh = [by[n] for n in enames]
                ep8, (eax,) = etask.evaluate(np.array([sim.getJointPosition(h) for h in eh]))
                epairs = [l for l in pairs if l[0].startswith('ecm_') or l[1].startswith('ecm_')]
                eport = np.array(ce['rcm_world'])
                for k in range(200):
                    seed = rng.uniform(np.maximum(etask.lo, -1.6), np.minimum(etask.hi, 1.6))
                    x_, r_ = etask.solve(seed, ep8, [eax])
                    if r_ >= 1e-8:
                        continue
                    sim.callScriptFunction('setJoints', probe, eh, [float(v) for v in x_])
                    bad_e = []
                    for i_, pts_ in ecm_pairs.scan(sim, probe, [[by[a], by[b]] for a, b in epairs]):
                        if epairs[i_] == [ce['tool_shape'], 'Patient_Skin'] and len(pts_) and \
                                np.linalg.norm(pts_ - eport, axis=1).max() <= ce['port_radius_m']:
                            continue
                        bad_e.append(epairs[i_])
                    if not bad_e:
                        return True
                return False

            for cost, x, bad in pool[:5]:
                if not bad or not all(any('ecm_' in n for n in pair) for pair in bad):
                    continue
                if taskk.margin(x, 1) < MIN_JOINT_MARGIN:
                    continue
                set_x(x)
                if repose_ecm():
                    report['ecm_reposed'] = True
                    set_x(x)
                    mok, mres = motion_check(x, p8)
                    accepted, motion_ok, motion_res = x, mok, mres
                    motion_failure = None if mok else mres[-1]
                    p8_final = p8
                    break
        try:
            sim.removeScript(probe)
        except Exception:
            pass
        if accepted is None:
            report['unrealisable'] = dict(arm=arm, port=port.tolist(), aim=G.tolist(),
                                          ik_solutions=len(pool),
                                          best_contacts=pool[0][2] if pool else None,
                                          motion_failure=motion_failure,
                                          reason='no contact-free posture on the port->aim line')
            write_json(REPORT, report)
            raise SystemExit(f'{arm}: port not realisable; see {relative(REPORT)}')
        set_x(accepted)                    # keep the docked posture for the next arm's contact scan
        contacts()                         # refresh distal_contacts for the accepted posture
        M8 = matrix_to_np(sim.getObjectMatrix(by[f'{arm}_joint8'], -1))
        err = max(float(np.linalg.norm(M8[:, 3] - p8_final)), float(np.linalg.norm(M8[:, 2] - axis)))
        e = dict(c)
        e.update(rcm_world=port.tolist(), axis_home=M8[:, 2].tolist(), transverse_x=M8[:, 0].tolist(),
                 transverse_y=M8[:, 1].tolist(), shaft_home=M8[:, 3].tolist(),
                 rcm_distance_home=float(np.dot(port - M8[:, 3], M8[:, 2])),
                 port_source='validation/feasibility/bladder_retraction.json (M3a plan)')
        updates[arm] = e
        for suffix in ('_RCM_reference', '_RCM_projection'):
            sim.setObjectPosition(by[arm + suffix], -1, port.tolist())
        if arm + '_RCM_fixed_ring' in by:
            sim.removeObjects([by[arm + '_RCM_fixed_ring']])
        bes.ring(sim, arm + '_RCM_fixed_ring', port, M8[:, 0], M8[:, 1], M8[:, 2], 0.008)
        report['arms'][arm] = dict(port=port.tolist(), wrist_point=W_pt.tolist(),
                                 bend_rad=trio[[r for r, a in roles.items() if a == arm][0]]['bend_rad'],
                                 plan_margin_m=trio[[r for r, a in roles.items() if a == arm][0]]['margin_m'],
                                 ld_m=ld, setback_m=setback_m, fk_err_m=err,
                                 distal_zone_contacts=list(distal_contacts),
                                 joint_positions=dict(zip(names, map(float, accepted))),
                                 joint_margin_normalised=float(taskk.margin(accepted, 1)),
                                 motion_ok=motion_ok, motion_failure=motion_failure,
                                 motion=motion_res,
                                 rcm_distance_home=e['rcm_distance_home'])

    new_rcm = [updates.get(c['arm'], c) for c in rcm]

    # Final check: with all three arms docked, scan every pair again (a later arm may now reach an
    # earlier arm's links).
    probe = ecm_pairs.create_probe(sim)
    full = [l for l in pairs if any(l[0].startswith(a + '_') for a in arm_port) or
            any(l[1].startswith(a + '_') for a in arm_port)]
    tool2port = {e['tool_shape']: (arm_port[e['arm']], e['port_radius_m'])
                 for e in new_rcm if e['arm'] in arm_port}
    bad_final = []
    for i, pts in ecm_pairs.scan(sim, probe, [[by[a], by[b]] for a, b in full]):
        pair = full[i]
        if pair[1] == 'Patient_Skin' and pair[0] in tool2port and len(pts) and \
                np.linalg.norm(pts - tool2port[pair[0]][0], axis=1).max() <= tool2port[pair[0]][1]:
            continue                        # shaft crossing its own port
        bad_final.append(pair)
    try:
        sim.removeScript(probe)
    except Exception:
        pass
    report['final_contacts'] = bad_final
    if bad_final:
        report['unrealisable'] = dict(arm='all', reason='docked postures leave residual contacts',
                                      contacts=bad_final)
        write_json(REPORT, report)
        raise SystemExit(f'docked arms have residual contacts; see {relative(REPORT)}')

    # ECM view offset to t* (the scope is commanded at runtime; this only records geometry)
    ecm = cfg['ecm']
    ecm_port = np.array(ecm['rcm_world'])
    view_dir = unit(t_star - ecm_port)
    report['ecm'] = dict(port=ecm_port.tolist(),
                         view_axis_to_t_star_deg=float(math.degrees(math.acos(
                             np.clip(np.dot(np.array(ecm['axis_home']), view_dir), -1, 1)))))

    # --- 4. controller embedded CONFIG refresh ------------------------------------------------
    old = sim.getStringProperty(by['RCMController'], 'code')
    marker = '-- Software RCM controller for the endoscope task scene'
    head, found, tail = old.partition(marker)
    lines = head.rstrip('\n').split('\n')
    inst_line = next((l for l in lines if l.startswith('INSTRUMENTS=')), None)
    source = ('CONFIG=' + lua(new_rcm) + '\nCOLLISION_PAIRS=' + lua(pairs) + '\n' +
              (inst_line + '\n' if inst_line else '') + marker + tail)
    sim.setStringProperty(by['RCMController'], 'code', source)

    # --- 5. scene-keyed config + save -----------------------------------------------------------
    write_json(SCENE_CFG / 'rcm_config.json', new_rcm)
    write_json(SCENE_CFG / 'collision_pairs.json',
               {'_note': 'same labels as config/instruments/collision_pairs.json (same object names)',
                'labels': pairs})
    write_json(SCENE_CFG / 'roi_proxy.json', roi_doc)
    write_json(SCENE_CFG / 'anatomy_transform.json',
               {'Patient_Bladder': dict(translate_m=(used_mm / 1000.0) * d_r.tolist(),
                                        source='validation/feasibility/bladder_retraction.json',
                                        magnitude_mm=used_mm)})
    save_scene(sim, args, PROXY_SCENE)
    stop_simulation(client, sim)
    report['seconds'] = round(time.time() - t0, 1)
    report['scene'] = str(relative(PROXY_SCENE))
    report['scene_sha256'] = sha256(PROXY_SCENE)
    report['config_dir'] = str(relative(SCENE_CFG))
    write_json(REPORT, report)
    print('wrote', relative(PROXY_SCENE), relative(REPORT), f"t={time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
