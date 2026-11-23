"""Check Phase A landmark projections and conservative RGB-D evidence, not ground-truth visibility."""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from phase_a_index import MANIFESTS, NOTES, REPO, split_order

sys.path.insert(0, str(REPO / 'scripts/task'))
sys.path.insert(0, str(REPO / 'scripts/vision'))
from cut_perception import MARKER_COLOURS, MARKER_RADIUS, MIN_MARKER_PX  # noqa: E402
from visibility import depth_visible  # noqa: E402

PLAN = REPO / 'config/learn/learned_keypoints_v1_dev.json'
SCENARIOS = ('nominal', 'occlusion_transient', 'relax', 'slip_regrasp')
NV, NU = 14, 24
MIDDLE = NU // 2
BARYCENTRIC = np.asarray([(1 / 3,) * 3, (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def project(pc, intr):
    w, h, f, far = intr
    if not np.isfinite(pc).all() or not 0 < pc[2] < far:
        return None
    return np.asarray((w / 2, h / 2)) - f * pc[:2] / pc[2]


def point_status(pc, depth, intr):
    uv = project(pc, intr)
    w, h = depth.shape[1], depth.shape[0]
    if uv is None or not (2 <= uv[0] < w - 2 and 2 <= uv[1] < h - 2):
        return uv, 'out_of_view'
    u, v = np.rint(uv).astype(int)
    centre = depth[v - 1:v + 2, u - 1:u + 2]
    if (np.isfinite(centre) & (centre > 0) & (centre < pc[2] - 0.003)).all():
        return uv, 'depth_blocked'
    return uv, 'coarse_clear' if depth_visible(depth, uv, pc[2]) else 'uncertain'


def marker_evidence(rgb, depth, pc, colour, intr):
    uv = project(pc, intr)
    if uv is None:
        return 0
    w, h, f, _ = intr
    radius = f * MARKER_RADIUS / pc[2] + 2
    u, v = uv
    x0, y0 = max(0, math.floor(u - radius)), max(0, math.floor(v - radius))
    x1, y1 = min(int(w), math.ceil(u + radius) + 1), min(int(h), math.ceil(v + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    pix = rgb[y0:y1, x0:x1].astype(float)
    z = depth[y0:y1, x0:x1]
    return int((((xx - u) ** 2 + (yy - v) ** 2 <= radius ** 2)
                & (np.linalg.norm(pix - 255 * np.asarray(colour), axis=2) < 60)
                & (z > 0) & (z >= pc[2] - MARKER_RADIUS - 0.001)
                & (z <= pc[2] + 0.001)).sum())


def face_edges(a, b, c):
    return [tuple(sorted(pair)) for pair in ((a, b), (a, c), (b, c))]


def bank_support(nodes, constraints, row, column, cell_column, depth, intr):
    bank = row * NU + column
    alive = {tuple(sorted((int(a), int(b)))): bool(bit) for a, b, bit in constraints}
    pixels = set()
    faces = 0
    w, h = depth.shape[1], depth.shape[0]
    for j in range(max(0, row - 1), min(row + 1, NV - 1)):
        a = j * NU + cell_column
        b, c, d = a + 1, a + NU, a + NU + 1
        for tri in [(a, b, d), (a, d, c)]:
            if bank not in tri or not all(alive.get(edge, False) for edge in face_edges(*tri)):
                continue
            faces += 1
            for point in BARYCENTRIC @ nodes[list(tri)]:
                uv = project(point, intr)
                if uv is not None and -0.5 <= uv[0] < w - 0.5 and -0.5 <= uv[1] < h - 0.5:
                    u, v = np.rint(uv).astype(int)
                    if depth[v, u] > 0 and abs(depth[v, u] - point[2]) <= 0.001:
                        pixels.add((u, v))
    return {'live_faces': faces, 'depth_supported_pixels': len(pixels)}


def landmark(pc, uv, status):
    return {'camera_xyz_m': pc.tolist(), 'uv_px': None if uv is None else uv.tolist(), 'status': status}


def audit_frame(path, tick, expected_row):
    with np.load(path, allow_pickle=False) as f:
        rgb, depth, intr = f['rgb'], f['depth'], f['intrinsics']
        R, t = f['cam_R'], f['cam_t']
        markers = f['label_marker_pos']
        tcp = f['label_tcp_M']
        available = {k for k in ('label_nodes', 'label_cut_front', 'label_cut_length',
                                 'label_tissue_alive') if k in f}
        if (f['tick'].item() != tick or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8
                or depth.shape != (480, 640) or intr.shape != (4,) or tuple(intr[:2]) != (640, 480)
                or min(intr[2:]) <= 0 or R.shape != (3, 3) or t.shape != (3,)
                or not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-6) or np.linalg.det(R) <= 0
                or tcp.shape != (4, 4) or markers.shape != (3, 3)
                or not all(np.isfinite(a).all() for a in (R, t, tcp, markers, depth, intr))):
            raise ValueError(f'{path}: invalid capture geometry')
        if (expected_row is None and available) or (expected_row is not None and len(available) != 4):
            raise ValueError(f'{path}: partial or discordant structural label')
        tip = R.T @ (tcp[:3, 3] - t)
        uv, coarse = point_status(tip, depth, intr)
        marker_counts = [marker_evidence(rgb, depth, mark, colour, intr)
                         for mark, colour in zip((markers - t) @ R, MARKER_COLOURS)]
        tip_status = ('fiducial_candidate' if coarse == 'coarse_clear' and min(marker_counts) >= MIN_MARKER_PX
                      else 'uncertain' if coarse == 'coarse_clear' else coarse)
        result = {'tip': dict(landmark(tip, uv, tip_status), m5a_coarse_clear=coarse == 'coarse_clear',
                              marker_pixels=marker_counts),
                  'structural_row': expected_row, 'front': {'status': 'pre_sheet'}}
        if expected_row is None:
            return result
        nodes, constraints, supplied = f['label_nodes'], f['label_tissue_alive'], f['label_cut_front']
        mm = 1000 * f['label_cut_length'].item()
        if (not 0 <= expected_row < NV or not math.isclose(mm, expected_row, rel_tol=0, abs_tol=1e-6)
                or nodes.shape != (NV, NU, 3) or supplied.shape != (3,) or not np.isfinite(nodes).all()
                or constraints.ndim != 2 or constraints.shape[1] != 3 or constraints.dtype.kind not in 'iu'
                or (constraints[:, :2] < 0).any() or (constraints[:, :2] >= NV * NU).any()
                or not np.isin(constraints[:, 2], (0, 1)).all()):
            raise ValueError(f'{path}: invalid structural payload')
        front = nodes[expected_row, MIDDLE - 1:MIDDLE + 1].mean(axis=0)
        if not np.allclose(front, supplied, rtol=0, atol=1e-8):
            raise ValueError(f'{path}: stored front and independently reconstructed midpoint disagree')
        pc = R.T @ (front - t)
        uv, coarse = point_status(pc, depth, intr)
        n = (nodes.reshape(-1, 3) - t) @ R
        banks = [bank_support(n, constraints, expected_row, col, cell, depth, intr)
                 for col, cell in ((MIDDLE - 1, MIDDLE - 2), (MIDDLE, MIDDLE))]
        if expected_row == 0:
            status = 'sheet_uncut'
        elif coarse == 'coarse_clear':
            status = ('geometry_candidate' if all(b['depth_supported_pixels'] >= 2 for b in banks)
                      else 'uncertain')
        else:
            status = coarse
        result['front'] = dict(landmark(pc, uv, status), banks=banks, m5a_coarse_clear=coarse == 'coarse_clear')
        return result


def collect_audit(index):
    episodes = index['episodes']
    if ([e['index'] for e in episodes] != sorted(range(300), key=split_order)
            or [e['partition'] for e in episodes] != ['validation'] * 30 + ['train'] * 270
            or index['scoring_notes_sha256'] != sha(NOTES.read_bytes())):
        raise ValueError('unexpected Phase A partition')
    by_scenario = Counter()
    selected, exposures = [], []
    root = Path(index['root'])
    for ep in episodes[30:]:
        raw = (MANIFESTS / f"ep{ep['index']:05d}.json").read_bytes()
        m = json.loads(raw)
        scenario = m['sampled']['scenario']
        if scenario not in SCENARIOS or m['index'] != ep['index'] or m['tag'] != 'dev':
            raise ValueError('unexpected manifest')
        if by_scenario[scenario] >= 8:
            continue
        if m['status'] != ep['status'] or m['ticks'] != len(ep['frames']):
            raise ValueError('manifest and frozen index disagree')
        by_scenario[scenario] += 1
        selected.append({'index': ep['index'], 'scenario': scenario, 'seed': m['seed'],
                         'status': m['status'], 'manifest_sha256': sha(raw)})
        for row in ep['frames']:
            result = audit_frame(root / row['rgbd'], row['tick'], row['row'])
            result.update(index=ep['index'], scenario=scenario, tick=row['tick'], rgbd=row['rgbd'])
            exposures.append(result)
    if by_scenario != {s: 8 for s in SCENARIOS} or len({e['seed'] for e in selected}) != len(selected):
        raise ValueError('incomplete or overlapping audit trajectories')
    return selected, exposures


def evidence_summary(exposures):
    groups = {'all': exposures} | {s: [r for r in exposures if r['scenario'] == s] for s in SCENARIOS}
    summary = {}
    for k, rows in groups.items():
        post = [r['front'] for r in rows if r['structural_row'] not in (None, 0)]
        summary[k] = {'frames': len(rows), 'tip': dict(Counter(r['tip']['status'] for r in rows)),
                      'front': dict(Counter(r['front']['status'] for r in rows)),
                      'postcut_front_coarse_clear': sum(p['m5a_coarse_clear'] for p in post),
                      'postcut_coarse_clear_without_geometry_support': sum(
                          p['m5a_coarse_clear'] and p['status'] != 'geometry_candidate' for p in post)}
    return summary


def examples(rows):
    sets = {'pre_sheet': lambda r: r['front']['status'] == 'pre_sheet',
            'sheet_uncut': lambda r: r['front']['status'] == 'sheet_uncut',
            'front_geometry_candidate': lambda r: r['front']['status'] == 'geometry_candidate',
            'front_depth_blocked': lambda r: r['front']['status'] == 'depth_blocked',
            'front_uncertain': lambda r: r['front']['status'] in {'uncertain', 'out_of_view'},
            'tip_not_supported': lambda r: r['tip']['status'] != 'fiducial_candidate'}
    out = {}
    for title, predicate in sets.items():
        seen = set()
        out[title] = []
        for row in rows:
            if row['index'] not in seen and predicate(row):
                out[title].append(row)
                seen.add(row['index'])
                if len(seen) == 2:
                    break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('refusing to overwrite an audit')
    index_raw, plan_raw = args.index.read_bytes(), PLAN.read_bytes()
    index = json.loads(index_raw)
    selected, exposures = collect_audit(index)
    report = {'plan': json.loads(plan_raw), 'plan_sha256': sha(plan_raw), 'index_sha256': sha(index_raw),
              'audit_script_sha256': sha(Path(__file__).read_bytes()), 'root': index['root'],
              'episodes': selected, 'summary': evidence_summary(exposures),
              'visual_examples': examples(exposures), 'exposures': exposures}
    with args.out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
