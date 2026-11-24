"""M3a protocol v3: task-geometry trio selection (tip ENVELOPES, not a single tip point).

Rationale (recorded in 'protocol_v3' before evaluation): v1/v2 sent all three tools to the same
tip point t and used a generic +-2 deg port-pivot robustness heuristic; the task actually puts the
retractor tips on the two grasp points either side of the cut line and moves the scissors along
the cut, so the required motion is the task envelope, not +-2 deg corners. The generic +-2 deg
result stands: no trio satisfies it <= 60 mm (v2).

v3, per magnitude 25..60 mm (same d_r), per VISIBLE target t (v1 'visible' lists):

* ROI: build_roi() from build_prostate_proxy_scene with the exact roi_proxy rule (c=t, n
  area-weighted within 3 mm, v = prostate base->apex PCA projected on the tangent, u = n x v,
  curved +2 mm offset grid, NU/NV/H = 24/14/1 mm). The rule orients the near edge toward the
  psm2 port side; both orientations are evaluated and each cut-port candidate is checked against
  the ROI whose near edge faces it (the mirrored rule keeps retractor envelopes consistent).
* Tip envelopes:
  - cut: tip = cut_start + k*v + l*u, k = -1..11 mm step 1 (9 mm cut + 1 mm overcut + 1 mm
    approach), l in {-1.5, 0, +1.5} mm (align tolerance): 39 samples.
  - retract_primary / retract_assist: grasp nodes n[NV//2, 1] / n[NV//2, NU-2] + s * (-+u)
    (outward side direction), s = -1.5..+3.0 mm step 0.5 (pull out 3 mm, relax 1.5 mm): 10 each.
* Qualification: a port qualifies for a role iff EVERY envelope sample has a polyline
  port->W->tip, bend <= theta_max(a), arc in [60,220] mm, clear of anatomy at 6 mm radius with
  the original exclusions (first 20 mm, last 12 mm of arc). Per sample the clearance-maximising
  W over the candidate distal-dir cone (theta=0 + BEND_FRACS x N_AZ) is used.
* Trio rules: v1 port spacing / ECM distance / meeting angles evaluated at the nominal samples
  (cut at cut_start+5 mm v, retractors at their grasp nodes); side assignment as v1; v2 tool-tool
  clearances (12 mm corridor/wrist/distal inside the body, last-12-mm contact zone exempt,
  PSM-vs-ECM corridor 14 mm) evaluated at the nominal config AND at the envelope extremes
  (cut at both v-ends x all lateral offsets x retractors at max pull = 6 extreme configs for
  cut-involving pairs, max-pull config for the retractor pair).
* Selection: smallest magnitude with >= 1 admissible trio; score = min over all evaluated
  checks of (clearance - required), including each port's worst envelope anatomy margin and its
  worst ECM-corridor slack over nominal+extreme samples; max-score trio wins, ties -> target
  nearest the lateral mid-gland.
* Transparency: count how many statically feasible trios would also pass the v2 +-2 deg check
  (expected 0).

If no magnitude <= 60 mm yields an admissible trio: report infeasible and stop.
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
BUILD = ANALYSIS.parent / 'build'
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(BUILD))
import prostate_access_feasibility as pa  # noqa: E402
import prostate_access_wrist as paw  # noqa: E402
import bladder_retraction_v2 as v2  # noqa: E402
from build_prostate_proxy_scene import build_roi, NU, NV, H  # noqa: E402
from simtools import CONFIG, REPO, add_arguments, read_json, relative, sha256, write_json  # noqa: E402

unit = pa.unit
CACHE = v2.CACHE
INSUFFLATED = v2.INSUFFLATED
OUT = v2.OUT

MAG_MM = range(25, 65, 5)
CUT_K_MM = np.arange(-1.0, 12.0, 1.0)          # -1..+11 mm along v
CUT_L_MM = (-1.5, 0.0, 1.5)                    # lateral mm along u
PULL_MM = np.arange(-1.5, 3.01, 0.5)           # relax 1.5 .. pull 3 mm
NOMINAL_CUT_K = 5.0
MAX_PULL = 3.0
# Sample order: extremes first so unqualified ports exit early.
CUT_ORDER = sorted([(k, l) for k in CUT_K_MM for l in CUT_L_MM],
                   key=lambda kl: (0 if abs(kl[0] - NOMINAL_CUT_K) > 5.5 else 1, -abs(kl[0] - NOMINAL_CUT_K)))
PULL_ORDER = sorted(PULL_MM, key=lambda s: -abs(s))
CUT_EXTREMES = [(k, l) for k in (-1.0, 11.0) for l in CUT_L_MM]


def candidate_dirs(P, t, theta_max):
    """(N,K,3) distal directions: theta=0 plus the BEND_FRACS x N_AZ cone around port->t."""
    d0 = unit(t - P)
    dirs = [d0]
    k1, k2 = paw.ortho_basis(d0)
    for frac in paw.BEND_FRACS:
        th = theta_max * frac
        for a in range(paw.N_AZ):
            dirs.append(d0 * math.cos(th) + math.sin(th) *
                        (k1 * math.cos(2 * math.pi * a / paw.N_AZ) +
                         k2 * math.sin(2 * math.pi * a / paw.N_AZ)))
    return np.stack(dirs, axis=1)


def poly_margins(occ, A, M, B, radius=paw.SHAFT_RADIUS, step=paw.STEP,
                 skip_start=paw.WALL_SKIP, skip_end=paw.TIP_SKIP):
    """Per-polyline margin = min sampled occluder distance - radius; -inf when the centre ray
    of either segment is blocked or the polyline is degenerate. Same arc-length sampling and
    exclusions as paw.clear_poly."""
    A, M, B = (np.atleast_2d(np.asarray(x, float)) for x in (A, M, B))
    n = len(A)
    L1 = np.linalg.norm(M - A, axis=1)
    L2 = np.linalg.norm(B - M, axis=1)
    u1 = (M - A) / np.maximum(L1, 1e-30)[:, None]
    u2 = (B - M) / np.maximum(L2, 1e-30)[:, None]
    margin = np.full(n, np.inf)
    live = np.nonzero((L1 > 1e-9) & (L2 > 1e-9))[0]
    for c0 in range(0, len(live), 512):
        c = live[c0:c0 + 512]
        O = np.vstack([A[c] + u1[c] * 1e-4, M[c] + u2[c] * 1e-4])
        D = np.vstack([u1[c], u2[c]])
        lim = np.concatenate([L1[c], L2[c]])
        locs, ridx, _ = occ.ray_mesh.ray.intersects_location(O, D, multiple_hits=False)
        if len(ridx):
            hit = np.linalg.norm(locs - O[ridx], axis=1) < lim[ridx] - 1e-4
            bad = np.zeros(2 * len(c), bool)
            bad[ridx[hit]] = True
            margin[c[bad[:len(c)]]] = -np.inf
            margin[c[bad[len(c):]]] = -np.inf
    arc = L1 + L2
    idx = np.nonzero(margin == np.inf)[0]       # +inf = ray-clear, still to be sampled
    counts = np.maximum(np.floor((arc[idx] - skip_end - skip_start) / step).astype(np.int64), 0)
    keep = counts > 0
    kidx, kcounts = idx[keep], counts[keep]
    if len(kidx):
        seg = np.repeat(np.arange(len(kidx)), kcounts)
        s = skip_start + step * (np.arange(kcounts.sum()) -
                                 np.repeat(np.cumsum(kcounts) - kcounts, kcounts))
        base = kidx[seg]
        on1 = s <= L1[base] - 1e-12
        P = np.empty((len(s), 3))
        P[on1] = A[base[on1]] + s[on1, None] * u1[base[on1]]
        P[~on1] = M[base[~on1]] + (s[~on1] - L1[base[~on1]])[:, None] * u2[base[~on1]]
        dist, _ = occ.tree.query(P)
        m = np.full(len(kidx), np.inf)
        np.minimum.at(m, seg, dist)
        margin[kidx] = m
        margin[idx[~keep]] = np.inf          # arc too short to sample: ray-clear suffices
    return margin - radius


def best_recs(occ, P, t, Ld, theta_max):
    """Per port (N,3) the clearance-maximising polyline to tip t over the dir cone.
    Returns (best_margin (N,), best (N,) rec-or-None)."""
    P = np.asarray(P, float)
    t = np.asarray(t, float)
    n = len(P)
    D = np.linalg.norm(t - P, axis=1)
    dirs = candidate_dirs(P, t, theta_max)                    # (N,K,3)
    K = dirs.shape[1]
    W = t[None, None, :] - Ld * dirs                          # (N,K,3)
    dW = W - P[:, None, :]
    L1 = np.linalg.norm(dW, axis=2)
    s = dW / np.maximum(L1, 1e-30)[:, :, None]
    bend = np.arccos(np.clip((s * dirs).sum(axis=2), -1, 1))
    arc = L1 + Ld
    feas = (D[:, None] <= paw.RANGE[1] + 1e-9) & (bend <= theta_max + 1e-6) & \
           (arc >= paw.RANGE[0] - 1e-9) & (arc <= paw.RANGE[1] + 1e-9) & (L1 > 1e-9)
    fi = np.nonzero(feas.ravel())[0]
    margin = np.full(n * K, -np.inf)
    if len(fi):
        margin[fi] = poly_margins(occ, np.repeat(P, K, 0)[fi],
                                W.reshape(-1, 3)[fi], np.tile(t, (n * K, 1))[fi])
    margin = margin.reshape(n, K)
    kbest = margin.argmax(axis=1)
    best_m = margin[np.arange(n), kbest]
    recs = []
    for i in range(n):
        if best_m[i] < 0:               # -inf (ray-blocked/degenerate) or within radius
            recs.append(None)
            continue
        k = int(kbest[i])
        recs.append(dict(port=P[i].tolist(), wrist=W[i, k].tolist(),
                         shaft_dir=s[i, k].tolist(), distal_dir=dirs[i, k].tolist(),
                         bend_rad=float(bend[i, k]), port_w_m=float(L1[i, k]),
                         arc_m=float(arc[i, k]), margin_m=float(best_m[i])))
    return best_m, recs


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

    rep = read_json(args.out)
    rep.pop('v3', None)
    rep.pop('port_plan_v3', None)
    proto_v3 = rep.get('protocol_v3') or dict(
        reason='v1/v2 sent all three tools to one tip point and used a generic +-2 deg pivot; the '
               'task puts retractor tips on the two grasp points and moves the scissors along the '
               'cut, so the required motion is the task envelope',
        envelopes=dict(cut='cut_start + k*v + l*u, k=-1..11 mm (9 mm cut + 1 mm overcut + 1 mm), '
                           'l in {-1.5,0,1.5} mm',
                       retractors='grasp nodes n[NV//2,1]/n[NV//2,NU-2] + s*(outward u), '
                                  's=-1.5..+3.0 mm step 0.5'),
        qualification='port qualifies iff every envelope sample has a clear polyline '
                      '(bend<=theta_max(a), arc 60-220 mm, 6 mm radius, original exclusions); '
                      'clearance-maximising W per sample',
        trio_rules='v1 spacing/ECM/meeting at nominal samples (cut at +5 mm v, retractors at '
                   'grasp nodes); v2 tool-tool clearances at nominal AND envelope extremes '
                   '(cut both v-ends x all laterals x max pull); PSM-vs-ECM 14 mm idem',
        score='min over all evaluated checks of (clearance - required), including each port '
              'worst envelope anatomy margin and worst ECM-corridor slack',
        selection='smallest magnitude 25..60 mm with >=1 admissible trio, then max score, '
                  'tie -> target nearest lateral mid-gland')

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
    bc0 = bladder.vertices.mean(axis=0)
    cam = np.asarray(data['camera'])
    ecm_seg = v2.seg_samples(ecm_port, cam[:, 3], skip_start=v2.INSIDE_SKIP_M)
    d_r = np.array(rep['protocol']['frame_derivation']['d_r'])
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
    v1_rows = {r['magnitude_mm']: r for r in rep['magnitudes']}

    pv, pf = pro.vertices.copy(), pro.faces.copy()
    _, _, Vt = np.linalg.svd(pv - pv.mean(axis=0), full_matrices=False)
    v_pca = Vt[0]           # prostate longitudinal PCA axis (sign irrelevant: both ROIs built)

    def role_samples(roi):
        """tip samples per role + nominal/extreme sample ids, from one ROI orientation."""
        cs, v, u, nodes = roi['cut_start'], roi['v'], roi['u'], roi['nodes']
        out = {}
        pts = [cs + k / 1000.0 * v + l / 1000.0 * u for k, l in CUT_ORDER]
        nom = CUT_ORDER.index((NOMINAL_CUT_K, 0.0))
        ext = [CUT_ORDER.index(kl) for kl in CUT_EXTREMES]
        out['cut'] = dict(points=np.array(pts), nominal=nom, extremes=ext)
        for role, node, sign in (('retract_primary', nodes[NV // 2, 1], -1.0),
                                 ('retract_assist', nodes[NV // 2, NU - 2], 1.0)):
            out[role] = dict(points=np.array([node + s / 1000.0 * sign * u for s in PULL_ORDER]),
                             nominal=PULL_ORDER.index(0.0),
                             extremes=[PULL_ORDER.index(MAX_PULL)])
        return out

    def env_eval(occ, cand_ports, samples, role):
        """Envelope qualification: best-clearance rec per sample per port; a port qualifies iff
        every sample has one. Sample order puts extremes first for early exit."""
        env = {}
        alive = np.arange(len(cand_ports))
        recs = {i: [None] * len(samples['points']) for i in alive}
        worst = np.full(len(cand_ports), np.inf)
        for si, t in enumerate(samples['points']):
            if not len(alive):
                break
            m, r = best_recs(occ, cand_ports[alive], t, tools[role]['ld_m'],
                             tools[role]['theta_conservative_rad'])
            for j, gi in enumerate(alive):
                recs[gi][si] = r[j]
                if r[j] is None:
                    worst[gi] = -np.inf
                else:
                    worst[gi] = min(worst[gi], r[j]['margin_m'])
            alive = alive[np.array([r[j] is not None for j in range(len(alive))])]
        for gi in alive:
            env[gi] = dict(recs=recs[gi], worst_margin=worst[gi])
        return env

    def shaft_slack(ra, rb, tip_a, tip_b, nominal):
        """v2 pair checks between rec ra (arm a) and rb (arm b) at their sample tips."""
        Pa, Wa = np.asarray(ra['port']), np.asarray(ra['wrist'])
        Pb, Wb = np.asarray(rb['port']), np.asarray(rb['wrist'])
        if nominal:
            if np.linalg.norm(Pa - Pb) < paw.PORT_SEP - 1e-12:
                return -np.inf
            ang = math.degrees(math.acos(np.clip(np.asarray(ra['shaft_dir']) @
                                                 np.asarray(rb['shaft_dir']), -1, 1)))
            if not (paw.MEET_DEG[0] <= ang <= paw.MEET_DEG[1]):
                return -np.inf
        Sa = v2.seg_samples(Pa, Wa, skip_start=v2.INSIDE_SKIP_M)
        Sb = v2.seg_samples(Pb, Wb, skip_start=v2.INSIDE_SKIP_M)
        Sda = v2.seg_samples(Wa, tip_a, skip_end=v2.CONTACT_ZONE_M)
        Sdb = v2.seg_samples(Wb, tip_b, skip_end=v2.CONTACT_ZONE_M)
        s = [float(np.linalg.norm(Sa[:, None, :] - Sb[None, :, :], axis=2).min())
             - v2.TOOL_SEP_M if len(Sa) and len(Sb) else -np.inf,
             float(np.linalg.norm(Wa - Wb)) - v2.TOOL_SEP_M]
        s.append(float(np.linalg.norm(Sda[:, None, :] - Sb[None, :, :], axis=2).min())
                 - v2.TOOL_SEP_M if len(Sda) and len(Sb) else np.inf)
        s.append(float(np.linalg.norm(Sdb[:, None, :] - Sa[None, :, :], axis=2).min())
                 - v2.TOOL_SEP_M if len(Sdb) and len(Sa) else np.inf)
        return min(s)

    def ecm_slack(rec):
        P, W = np.asarray(rec['port']), np.asarray(rec['wrist'])
        S = v2.seg_samples(P, W, skip_start=v2.INSIDE_SKIP_M)
        d = float(np.linalg.norm(S[:, None, :] - ecm_seg[None, :, :], axis=2).min()) \
            if len(S) else v2.seg_seg_min(P, W, ecm_seg[0], ecm_seg[-1])
        return d - v2.ECM_SEP_M

    def search(occ, roi, sam, envs, occ_motion):
        """envs[role] = {port_idx: {recs, worst_margin}}; returns (score, ti-trio detail) or None
        plus stats. Pair slack = min over nominal + extreme configs of shaft_slack."""
        lists = [envs[r] for r in paw.ROLES]
        att = dict(sizes=[len(x) for x in lists], corridor_feasible=0, motion_v2_pass=0,
                   best_score_m=None)
        if not all(lists):
            return None, att
        tips = {r: sam[r]['points'] for r in paw.ROLES}
        nom = {r: sam[r]['nominal'] for r in paw.ROLES}
        ext = {r: sam[r]['extremes'] for r in paw.ROLES}
        R0, R1, R2 = paw.ROLES          # retract_primary, cut, retract_assist
        # per-port worst ECM slack over nominal + extremes
        ecm_min = [{gi: min(ecm_slack(envs[r][gi]['recs'][nom[r]]),
                            *[ecm_slack(envs[r][gi]['recs'][e]) for e in ext[r]])
                    for gi in envs[r]} for r in paw.ROLES]
        # pair slack matrices: min over nominal + extreme configs. Extremes: cut at both v-ends
        # x all laterals (6 samples) with the involved retractor at max pull; the
        # primary-assist pair takes both retractors at max pull.
        cfg = {(0, 1): [(nom[R0], nom[R1], True)] +
                       [(ext[R0][0], e, False) for e in ext[R1]],
               (0, 2): [(nom[R0], nom[R2], True), (ext[R0][0], ext[R2][0], False)],
               (1, 2): [(nom[R1], nom[R2], True)] +
                       [(e, ext[R2][0], False) for e in ext[R1]]}
        M = {}
        for (i, j) in itertools.combinations(range(3), 2):
            ri, rj = paw.ROLES[i], paw.ROLES[j]
            gi_list, gj_list = sorted(lists[i]), sorted(lists[j])
            m = np.full((len(gi_list), len(gj_list)), -np.inf)
            for a, ga in enumerate(gi_list):
                for b, gb in enumerate(gj_list):
                    best = np.inf
                    for sa, sb, is_nom in cfg[(i, j)]:
                        ra, rb = lists[i][ga]['recs'][sa], lists[j][gb]['recs'][sb]
                        best = min(best, shaft_slack(ra, rb, tips[ri][sa], tips[rj][sb], is_nom))
                        if best == -np.inf:
                            break
                    m[a, b] = best
            M[(i, j)] = (m, gi_list, gj_list)
        best = None
        m12, g1, g2 = M[(1, 2)]
        m01, g0a, _ = M[(0, 1)]
        m02, g0b, _ = M[(0, 2)]
        for b in range(len(g1)):
            for ci in range(len(g2)):
                s12 = m12[b, ci]
                if not np.isfinite(s12) or s12 < 0:
                    continue
                col = np.minimum(m01[:, b], m02[:, ci])
                a = int(np.argmax(col))
                if not np.isfinite(col[a]) or col[a] < 0:
                    continue
                trio_idx = (g0a[a], g1[b], g2[ci])
                if len(set(trio_idx)) < 3:
                    continue
                score = min(col[a], s12,
                            ecm_min[0][trio_idx[0]], ecm_min[1][trio_idx[1]],
                            ecm_min[2][trio_idx[2]],
                            lists[0][trio_idx[0]]['worst_margin'],
                            lists[1][trio_idx[1]]['worst_margin'],
                            lists[2][trio_idx[2]]['worst_margin'])
                att['corridor_feasible'] += 1
                nomrecs = [lists[k][trio_idx[k]]['recs'][nom[paw.ROLES[k]]] for k in range(3)]
                if all(v2.motion_ok(occ_motion, np.asarray(x['port']), np.asarray(x['wrist']),
                                    tips[paw.ROLES[k]][nom[paw.ROLES[k]]],
                                    tools[paw.ROLES[k]]['theta_conservative_rad'])[0]
                       for k, x in enumerate(nomrecs)):
                    att['motion_v2_pass'] += 1
                if best is None or score > best[0]:
                    best = (score, trio_idx, nomrecs)
        att['best_score_m'] = best[0] if best else None
        return best, att

    table = []
    found, used_mm, used_side = None, None, None
    for mm in MAG_MM:
        row = v1_rows.get(mm)
        if row is None or not (row['admissible'] and any(row['visible'])):
            table.append(dict(magnitude_mm=mm, skipped='not admissible/visible under v1'))
            continue
        occ = pa.Occluder(base + others + [trimesh.Trimesh(
            bladder.vertices + (mm / 1000.0) * d_r, bladder.faces, process=False)])
        per_target = []
        # Nearest lateral mid-gland first so a score tie prefers it.
        order = sorted(np.nonzero(row['visible'])[0],
                       key=lambda x: abs((targets[x] - c) @ lr))
        for ti in order:
            t = targets[ti]
            best_t, atts = None, []
            for vsign in (1.0, -1.0):
                # both ROI orientations: a dummy port along +/- the PCA axis so build_roi's
                # near-edge flip produces each orientation once
                roi = build_roi(pv, pf, bc0, t, t + vsign * 0.05 * v_pca)
                sam = role_samples(roi)
                envs = {}
                for role in paw.ROLES:
                    cand = [i for i, P in enumerate(ports)
                            if np.sign((P - c) @ lr) == arm_sides[role]]
                    if role == 'cut':
                        cand = [i for i in cand if (ports[i] - t) @ roi['v'] < 0]
                    # cheap pre-filter: keep only ports reaching the nominal envelope sample
                    m0, _ = best_recs(occ, ports[cand],
                                      sam[role]['points'][sam[role]['nominal']],
                                      tools[role]['ld_m'],
                                      tools[role]['theta_conservative_rad'])
                    cand = [cand[i] for i in range(len(cand)) if m0[i] >= 0]
                    envs[role] = env_eval(occ, ports[cand], sam[role], role)
                    envs[role] = {cand[i]: v for i, v in envs[role].items()}
                best, att = search(occ, roi, sam, envs, occ)
                att['v_side'] = float(vsign)
                atts.append(att)
                if best and (best_t is None or best[0] > best_t[0]):
                    best_t = best + (roi, sam, vsign)
            per_target.append(dict(target=int(ti), best_score_m=best_t[0] if best_t else None,
                                   attempted=atts))
            if best_t and (found is None or best_t[0] > found[0]):
                found = (best_t[0], ti, best_t[1], best_t[2], best_t[3], best_t[4], best_t[5])
                used_mm, used_side = mm, float(best_t[5])
        mm_best = max([x['best_score_m'] for x in per_target
                       if x['best_score_m'] is not None] or [None])
        table.append(dict(magnitude_mm=mm,
                          admissible_targets=[x['target'] for x in per_target
                                              if x['best_score_m'] is not None],
                          best_score_m=mm_best,
                          per_target=per_target))
        if found:
            break
        print(f'{mm} mm: no v3-admissible trio t={time.time() - t_start:.0f}s', flush=True)

    rep['protocol_v3'] = proto_v3
    rep['v3'] = dict(table=table, selected_magnitude_mm=used_mm,
                     seconds=round(time.time() - t_start, 1),
                     source_sha256={relative(x): sha256(x) for x in [Path(__file__)]})
    if found is None:
        rep['port_plan_v3'] = None
        write_json(args.out, rep)
        print('INFEASIBLE under v3: no admissible trio at 25..60 mm')
        return
    score, ti, trio_idx, nomrecs, roi, sam, vsign = found
    plan_out = dict(t_star=dict(index=int(ti), point=targets[ti].tolist(), side=str(side[ti])),
                    used_magnitude_mm=used_mm, used_target=int(ti), v_side=vsign,
                    protocol='protocol_v3',
                    ecm_port=ecm_port.tolist(), ecm_scope_tip=cam[:, 3].tolist(),
                    arm_side_signs={r: int(arm_sides[r]) for r in paw.ROLES},
                    trio=[dict(role=paw.ROLES[i], arm=tools[paw.ROLES[i]]['arm'],
                               port=nomrecs[i]['port'], wrist=nomrecs[i]['wrist'],
                               bend_rad=nomrecs[i]['bend_rad'],
                               margin_m=nomrecs[i]['margin_m']) for i in range(3)],
                    roi=dict(centre_m=roi['c'].tolist() if hasattr(roi['c'], 'tolist') else list(roi['c']),
                             normal=roi['n'].tolist(), v=roi['v'].tolist(), u=roi['u'].tolist(),
                             cut_start_m=roi['cut_start'].tolist(),
                             grasp_m={k: x.tolist() for k, x in roi['grasp'].items()}),
                    score_m=float(score))
    rep['port_plan_v3'] = plan_out
    write_json(args.out, rep)
    print('v3 plan:', used_mm, 'mm target', ti, 'score', round(score * 1000, 2), 'mm slack,',
          f"t={time.time() - t_start:.0f}s", flush=True)


if __name__ == '__main__':
    main()
