"""M3a protocol v2: re-select the port trio with tool-tool corridor and motion-robustness rules.

v1 (bladder_retraction.json, kept untouched) lacked tool-vs-tool and motion-robustness criteria;
its selected trio was found UNREALISABLE in M3b: the three port->W corridors pass within
0.65-1.96 mm of each other, so the straight tool bodies interpenetrate mid-corridor.

Protocol v2 (written into the report BEFORE evaluation, under 'protocol_v2'):

* Same inputs as v1: same targets, port grid, camera model, polyline model, margins, side rule.
* Trio admissibility = all v1 rules PLUS:
  - tool-tool: centreline distance between every pair of PSM port->W segments >= 12 mm
    (2 x the 6 mm instrument radius used for anatomy), measured inside the body
    (beyond 20 mm from the ports); |W_i - W_j| >= 12 mm; each distal W->t segment vs the other
    arms' port->W segments >= 12 mm except the distal's last 12 mm before t (the contact zone).
    Each PSM corridor vs the ECM scope corridor (ECM port -> current scope tip, 8 mm radius
    as in the camera model): >= 6 + 8 = 14 mm.
  - motion robustness: each arm's polyline stays clear of anatomy (same 6 mm radius, same
    sampling: first 20 mm and last 12 mm of arc exempt) when its shaft is pivoted +-2 deg about
    the port in the four corner directions, with the W->t bend re-solved to keep the tip at t;
    a corner fails if the re-solved bend exceeds the conservative theta_max (a).
* Selection: magnitudes 25, 30, ..., 60 mm; at each, search trios over all triangulable targets
  in v1 priority order; pick the smallest magnitude with >= 1 admissible trio, then the trio
  maximising score = min over all checks of (clearance - required) and the three polylines'
  anatomy margins. If no magnitude <= 60 mm works: report infeasible and stop.

Output: appends 'protocol_v2', 'v2' (per-magnitude table) and 'port_plan_v2' to
validation/feasibility/bladder_retraction.json. v1 keys are not modified.
"""
import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))
import prostate_access_feasibility as pa  # noqa: E402
import prostate_access_wrist as paw  # noqa: E402
from simtools import CONFIG, REPO, add_arguments, read_json, relative, sha256, write_json  # noqa: E402

unit = pa.unit

CACHE = REPO / 'validation' / 'feasibility' / 'prostate_access_wrist_cache.npz'
INSUFFLATED = REPO / 'models' / 'anatomy' / 'insufflated_wall.npz'
OUT = REPO / 'validation' / 'feasibility' / 'bladder_retraction.json'

MAG_MM = range(25, 65, 5)
TOOL_SEP_M = 0.012                  # 2 x 6 mm instrument radius, inside the body
INSIDE_SKIP_M = 0.020               # corridor clearance measured beyond 20 mm from the ports
CONTACT_ZONE_M = 0.012              # last 12 mm of the distal segment exempt (the contact zone)
ECM_RADIUS_M = 0.008                # scope corridor radius from the camera model
ECM_SEP_M = paw.SHAFT_RADIUS + ECM_RADIUS_M    # >= 14 mm
MOTION_DEG = 2.0                    # +-2 deg pivot about the port, 4 corners (plan_psm_ports)
TOP_K = 12


def seg_samples(P, Q, skip_start=0.0, skip_end=0.0, step=paw.STEP):
    """Sample points along segment P->Q, skipping the ends."""
    d = Q - P
    L = float(np.linalg.norm(d))
    n = max(int((L - skip_start - skip_end) / step), 0)
    if not n:
        return np.empty((0, 3))
    return P + (d / L) * (skip_start + np.arange(n)[:, None] * step)


def seg_seg_min(A1, A2, B1, B2):
    """Min distance between two segments."""
    d1, d2, r = A2 - A1, B2 - B1, A1 - B2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-18 else 0.0
    t = (b * s + f) / e
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((A1 + s * d1) - (B2 + t * d2)))


def corridor_clearance(combo, t, ecm_seg, tip_skip=CONTACT_ZONE_M):
    """Slack values for the v2 tool-tool checks of one trio at target t.
    Returns list of (name, clearance_m, required_m) or the first violated check."""
    corrs = []      # inside-body corridor segments per arm
    checks = []
    for x in combo:
        P = np.asarray(x[1]['port']); W = np.asarray(x[1]['wrist'])
        S = seg_samples(P, W, skip_start=INSIDE_SKIP_M)
        corrs.append((P, W, S))
    for i, j in itertools.combinations(range(3), 2):
        Pi, Wi, Si = corrs[i]
        Pj, Wj, Sj = corrs[j]
        if len(Si) and len(Sj):
            d = float(np.linalg.norm(Si[:, None, :] - Sj[None, :, :], axis=2).min())
        else:
            d = seg_seg_min(Pi, Wi, Pj, Wj)
        checks.append((f'corridor_{i}_{j}', d, TOOL_SEP_M))
        checks.append((f'wrist_{i}_{j}', float(np.linalg.norm(Wi - Wj)), TOOL_SEP_M))
        # distal_i vs corridor_j and distal_j vs corridor_i, last 12 mm of the distal exempt
        for a, b in ((i, j), (j, i)):
            Pa, Wa, _ = corrs[a]
            Pb, Wb, Sb = corrs[b]
            Sd = seg_samples(Wa, t, skip_end=tip_skip)
            if len(Sd) and len(Sb):
                d = float(np.linalg.norm(Sd[:, None, :] - Sb[None, :, :], axis=2).min())
            else:
                d = seg_seg_min(Wa, t, Pb, Wb)
            checks.append((f'distal{a}_shaft{b}', d, TOOL_SEP_M))
    for i, (P, W, S) in enumerate(corrs):
        d = float(np.linalg.norm(S[:, None, :] - ecm_seg[None, :, :], axis=2).min()) \
            if len(S) and len(ecm_seg) else seg_seg_min(P, W, *ecm_seg[[0, -1]])
        checks.append((f'corridor{i}_ecm', d, ECM_SEP_M))
    return checks


def motion_ok(occ, P, W, t, theta_max, radius=paw.SHAFT_RADIUS):
    """v2 motion robustness: shaft pivoted +-2 deg about the port in 4 corners; W moves with it,
    the W->t bend is re-solved to keep the tip at t; corner fails if bend' > theta_max (a) or
    the re-solved polyline is within `radius` of anatomy (same sampling as reach_poly)."""
    axis = unit(W - P)
    L1 = float(np.linalg.norm(W - P))
    tx = unit(np.cross(axis, [0, 0, 1.0]))
    ty = np.cross(axis, tx)
    out = []
    for sgn in (-1, 1):
        for q in (-1, 1):
            dk = unit(axis + math.tan(math.radians(MOTION_DEG)) * (sgn * tx + q * ty))
            Wp = P + dk * L1
            dd = t - Wp
            L2 = float(np.linalg.norm(dd))
            if L2 < 1e-9:
                out.append(dict(corner=(sgn, q), fail='degenerate'))
                return False, out
            d2 = dd / L2
            bend = float(math.acos(np.clip(dk @ d2, -1, 1)))
            if bend > theta_max + 1e-9:
                out.append(dict(corner=(sgn, q), fail='bend', bend_rad=round(bend, 4)))
                return False, out
            S1 = seg_samples(P, Wp, skip_start=paw.WALL_SKIP)
            S2 = seg_samples(Wp, t, skip_end=paw.TIP_SKIP)
            S = np.vstack([x for x in (S1, S2) if len(x)]) if len(S1) + len(S2) else np.array([Wp])
            d = float(occ.tree.query(S)[0].min())
            if d < radius:
                out.append(dict(corner=(sgn, q), fail='anatomy', clearance_mm=round(d * 1000, 2)))
                return False, out
            out.append(dict(corner=(sgn, q), ok=True, clearance_mm=round(d * 1000, 2),
                            bend_rad=round(bend, 4)))
    return True, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(p)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--bp3d-zip', type=Path, default=pa.BP3D_ZIP)
    p.add_argument('--elements', type=Path, default=pa.ELEMENTS)
    p.add_argument('--insufflated', type=Path, default=INSUFFLATED)
    p.add_argument('--baseline', type=Path, default=paw.BASELINE)
    p.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args()
    t_start = time.time()
    pa.ELEMENTS, pa.BP3D_ZIP = args.elements, args.bp3d_zip
    sys.path.insert(0, str(REPO / 'scripts' / 'build'))

    v1 = read_json(args.out)
    v1.pop('v2', None)
    v1.pop('port_plan_v2', None)
    proto_v2 = v1.get('protocol_v2') or dict(reason='v1 lacked tool-tool and motion-robustness criteria; its trio was '
                           'found unrealisable in M3b (corridors 0.65-1.96 mm apart)',
                    tool_sep_m=TOOL_SEP_M, inside_skip_m=INSIDE_SKIP_M,
                    contact_zone_m=CONTACT_ZONE_M, ecm_sep_m=ECM_SEP_M,
                    motion_deg=MOTION_DEG,
                    trio_rules='v1 spacing/side/meeting rules PLUS tool-tool corridor, wrist '
                               'separation, distal-vs-shaft and ECM-corridor checks, PLUS +-2 deg '
                               'motion robustness with re-solved bend within theta_max(a)',
                    score='min over all checks of (clearance_m - required_m) and the three '
                          'polylines anatomy margins',
                    selection='smallest magnitude 25..60 mm with >=1 admissible trio, then max score; '
                              'targets scanned in v1 priority order (min-role desc, lateral tie-break)')

    plan = pa.read_json(REPO / 'config' / 'registered_plan.json')
    B = np.array(plan['rotation'])
    anterior = B[:, 2]
    rcm = pa.read_json(CONFIG / 'instruments' / 'rcm_config.json')
    ecm_port = np.array(next(a for a in rcm if a['arm'] == 'ecm')['rcm_world'])
    current_ports = {a['arm']: np.array(a['rcm_world']) for a in rcm if a['arm'] in paw.ARMS}
    roles_cfg = pa.read_json(CONFIG / 'task' / 'cut_task.json')['roles']
    tools = paw.load_tools(roles_cfg)
    payload = paw.read_inputs(args)
    data, wrist = payload['scene'], payload['wrist']
    for role in paw.ROLES:
        tools[role]['ld_m'] = wrist[tools[role]['arm']]['ld_m']
    scene = {n: trimesh.Trimesh(*data[n], process=False) for n in pa.SCENE_PARTS}
    new = pa.load_new_parts(B, np.array(plan['shift']))
    pro, bladder = scene['Patient_Prostate'], scene['Patient_Bladder']
    c = pro.vertices.mean(axis=0)
    cam = np.asarray(data['camera'])
    ecm_seg = seg_samples(ecm_port, cam[:, 3], skip_start=INSIDE_SKIP_M)
    d_r = np.array(v1['protocol']['frame_derivation']['d_r'])
    lr = unit(B @ pa.R0 @ np.array([1.0, 0, 0]))
    lr = unit(lr - np.dot(lr, anterior) * anterior)

    w = np.load(args.insufflated)
    Mp = data['skin_parent']
    skin = w['outer_vertices_parent'] @ Mp[:, :3].T + Mp[:, 3]
    keys = np.round(skin / 0.010).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    skin = skin[np.sort(first)]
    ports = skin[((skin - c) @ anterior > 0.05)]
    ports = ports[np.linalg.norm(ports - ecm_port, axis=1) >= paw.PORT_SEP]

    baseline = pa.read_json(args.baseline)
    targets = np.array(baseline['targets']['points'], float)
    side = np.array(baseline['targets']['side'])
    base = [scene[n] for n in pa.SCENE_PARTS if n != 'Patient_Bladder']
    others = [m['mesh'] for m in new.values()]
    arm_sides = {role: np.sign((current_ports[tools[role]['arm']] - c) @ lr) or 1.0
                 for role in paw.ROLES}
    v1_rows = {r['magnitude_mm']: r for r in v1['magnitudes']}

    def margin_of(P, W, t, occ):
        S = np.vstack([x for x in (seg_samples(P, W, skip_start=paw.WALL_SKIP),
                                   seg_samples(W, t, skip_end=paw.TIP_SKIP)) if len(x)])
        return float(occ.tree.query(S)[0].min()) - paw.SHAFT_RADIUS if len(S) else -np.inf

    def pair_slack(x, y, t):
        """v2 pairwise slack between rec x (arm i) and rec y (arm j): -inf on a v1 pair-rule
        violation, else min over corridor, wrist and both distal-vs-shaft checks."""
        Pi, Wi = np.asarray(x[1]['port']), np.asarray(x[1]['wrist'])
        Pj, Wj = np.asarray(y[1]['port']), np.asarray(y[1]['wrist'])
        if np.linalg.norm(Pi - Pj) < paw.PORT_SEP - 1e-12:
            return -np.inf
        si, sj = unit(Wi - Pi), unit(Wj - Pj)
        ang = math.degrees(math.acos(np.clip(si @ sj, -1, 1)))
        if not (paw.MEET_DEG[0] <= ang <= paw.MEET_DEG[1]):
            return -np.inf
        Si = seg_samples(Pi, Wi, skip_start=INSIDE_SKIP_M)
        Sj = seg_samples(Pj, Wj, skip_start=INSIDE_SKIP_M)
        Sdi = seg_samples(Wi, t, skip_end=CONTACT_ZONE_M)
        Sdj = seg_samples(Wj, t, skip_end=CONTACT_ZONE_M)
        s = [float(np.linalg.norm(Si[:, None, :] - Sj[None, :, :], axis=2).min()) - TOOL_SEP_M,
             float(np.linalg.norm(Wi - Wj)) - TOOL_SEP_M]
        s.append(float(np.linalg.norm(Sdi[:, None, :] - Sj[None, :, :], axis=2).min())
                 - TOOL_SEP_M if len(Sdi) else np.inf)
        s.append(float(np.linalg.norm(Sdj[:, None, :] - Si[None, :, :], axis=2).min())
                 - TOOL_SEP_M if len(Sdj) else np.inf)
        return min(s)

    def search(mm, occ):
        """Exhaustive v2 search at one magnitude over all triangulable targets. All v2
        constraints are unary or pairwise, so slack matrices M_ij[a,b] are computed once per
        target and the trio score max is enumerated via the best-a per (b,c); motion robustness
        gates the candidates in descending score order."""
        row = v1_rows[mm]
        con = row['conservative']
        tri_idx = [i for i, ok in enumerate(con['triangulable']) if ok]
        attempted = []
        best_overall = None           # (score, ti, trio, detail)
        for ti in tri_idx:
            t = targets[ti]
            lists = []                # per role: (port_index, rec, margin)
            for role in paw.ROLES:
                recs = paw.reach_poly(occ, ports, t, tools[role]['ld_m'],
                                      tools[role]['theta_conservative_rad'])
                kept = []
                for x in enumerate(recs):
                    if x[1] is None:
                        continue
                    P, W = np.asarray(x[1]['port']), np.asarray(x[1]['wrist'])
                    if np.sign((P - c) @ lr) != arm_sides[role]:
                        continue
                    if np.linalg.norm(P - ecm_port) < paw.PORT_SEP - 1e-12:
                        continue
                    S = seg_samples(P, W, skip_start=INSIDE_SKIP_M)
                    d = float(np.linalg.norm(S[:, None, :] - ecm_seg[None, :, :], axis=2).min()) \
                        if len(S) else seg_seg_min(P, W, ecm_seg[0], ecm_seg[-1])
                    if d < ECM_SEP_M - 1e-12:
                        continue
                    kept.append((x[0], x[1], margin_of(P, W, t, occ)))
                lists.append(kept)
            att = dict(target=int(ti), sizes=[len(x) for x in lists],
                       corridor_feasible=0, motion_fail=0, best_corridor_slack_m=None,
                       admissible=0, best_score_m=None)
            M = {}
            for (i, j) in itertools.combinations(range(3), 2):
                m = np.full((len(lists[i]), len(lists[j])), -np.inf)
                for a, x in enumerate(lists[i]):
                    for b, y in enumerate(lists[j]):
                        m[a, b] = pair_slack(x, y, t)
                M[(i, j)] = m
            if all(len(x) for x in lists):
                att['best_corridor_slack_m'] = float(
                    max(M[(i, j)].max() for i, j in M)) if M else None
                cands = []
                for b in range(len(lists[1])):
                    for cix in range(len(lists[2])):
                        s12 = M[(1, 2)][b, cix]
                        if not np.isfinite(s12):
                            continue
                        col = np.minimum(M[(0, 1)][:, b], M[(0, 2)][:, cix])
                        a = int(np.argmax(col))
                        s = min(col[a], s12, lists[0][a][2], lists[1][b][2], lists[2][cix][2])
                        if np.isfinite(s) and s >= 0:
                            cands.append((s, a, b, cix))
                cands.sort(key=lambda z: -z[0])
                att['corridor_feasible'] = len(cands)
                for s, a, b, cix in cands:
                    trio = (lists[0][a], lists[1][b], lists[2][cix])
                    motion = [motion_ok(occ, np.asarray(x[1]['port']), np.asarray(x[1]['wrist']),
                                        t, tools[paw.ROLES[i]]['theta_conservative_rad'])
                              for i, x in enumerate(trio)]
                    if not all(ok for ok, _ in motion):
                        att['motion_fail'] += 1
                        continue
                    att['admissible'] = 1
                    att['best_score_m'] = s
                    if best_overall is None or s > best_overall[0]:
                        sdirs = [unit(np.asarray(x[1]['wrist']) - np.asarray(x[1]['port']))
                                 for x in trio]
                        meets = [math.degrees(math.acos(np.clip(sdirs[p] @ sdirs[q], -1, 1)))
                                 for p, q in itertools.combinations(range(3), 2)]
                        best_overall = (s, ti, trio, dict(
                            margins_m=[x[2] for x in trio],
                            meeting_deg=[round(x, 1) for x in meets],
                            motion=[[o for _, o in mo] for mo in motion]))
                    break
            attempted.append(att)
        return best_overall, attempted

    table = []
    found, used_mm = None, None
    for mm in MAG_MM:
        row = v1_rows.get(mm)
        if row is None or not (row['admissible'] and row.get('n_triangulable_a')):
            table.append(dict(magnitude_mm=mm, skipped='not admissible/triangulable under v1'))
            continue
        occ = pa.Occluder(base + others + [trimesh.Trimesh(
            bladder.vertices + (mm / 1000.0) * d_r, bladder.faces, process=False)])
        best, attempted = search(mm, occ)
        table.append(dict(magnitude_mm=mm,
                          admissible_targets=[a['target'] for a in attempted if a['admissible']],
                          best_score_m=round(best[0], 5) if best else None,
                          attempted=attempted))
        if best is not None:
            found, used_mm = best, mm
            break
        print(f'{mm} mm: no v2-admissible trio t={time.time() - t_start:.0f}s', flush=True)

    v1['protocol_v2'] = proto_v2
    v1['v2'] = dict(table=table, selected_magnitude_mm=used_mm, seconds=round(time.time() - t_start, 1),
                    search='exhaustive (all v2 constraints unary or pairwise; trio max via slack '
                           'matrices, motion-gated in descending score order)',
                    source_sha256={relative(x): sha256(x) for x in [Path(__file__)]})
    if found is None:
        v1['port_plan_v2'] = None
        write_json(args.out, v1)
        print('INFEASIBLE under v2: no admissible trio at 25..60 mm')
        return
    score, ti, trio, detail = found
    combo = [(x[0], x[1]) for x in trio]
    checks = corridor_clearance(combo, targets[ti], ecm_seg)
    plan_out = dict(t_star=dict(index=int(ti), point=targets[ti].tolist(), side=str(side[ti])),
                    used_magnitude_mm=used_mm, used_target=int(ti),
                    protocol='protocol_v2',
                    ecm_port=ecm_port.tolist(), ecm_scope_tip=cam[:, 3].tolist(),
                    arm_side_signs={r: int(arm_sides[r]) for r in paw.ROLES},
                    trio=[dict(role=paw.ROLES[i], arm=tools[paw.ROLES[i]]['arm'],
                               port=combo[i][1]['port'], wrist=combo[i][1]['wrist'],
                               bend_rad=combo[i][1]['bend_rad'],
                               margin_m=detail['margins_m'][i]) for i in range(3)],
                    meeting_angles_deg=detail['meeting_deg'],
                    tool_tool_checks={n: dict(clearance_m=round(d, 5), required_m=req)
                                      for n, d, req in checks},
                    score_m=float(score), motion=detail['motion'])
    v1['port_plan_v2'] = plan_out
    write_json(args.out, v1)
    print('v2 plan:', used_mm, 'mm target', ti, 'score', round(score * 1000, 2), 'mm slack,',
          f"t={time.time() - t_start:.0f}s", flush=True)


if __name__ == '__main__':
    main()
