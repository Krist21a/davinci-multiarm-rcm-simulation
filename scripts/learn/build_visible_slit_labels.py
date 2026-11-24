"""Cache native-HD inputs and conservative weak aperture masks for Phase A."""
import argparse
import io
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy import ndimage

from tip_data import MANIFESTS, REPO, frozen_rows, sha, write_json

PLAN = REPO / 'config/learn/visible_slit_weak_v1.json'
SIZE = 384
X0, Y0 = 448, 288
CROP = np.s_[Y0:Y0 + SIZE, X0:X0 + SIZE]
NV, NU = 14, 24
UNKNOWN = 255
HOOD = np.ones((3, 3), bool)


def strip_faces(row, col):
    a = row * NU + col
    b, c, d = a + 1, a + NU, a + NU + 1
    return ((a, b, d), (a, d, c))


STRIPS = tuple(tuple(strip_faces(j, i) for i in (10, 11, 12)) for j in range(NV - 1))
EDGES = {tri: tuple(tuple(sorted((tri[i], tri[(i + 1) % 3]))) for i in range(3))
         for strip in STRIPS for cell in strip for tri in cell}


def pixel_hits(uv, z):
    """Return crop indices and perspective-correct depth strictly inside the triangle."""
    lo = np.maximum(0, np.ceil(uv.min(axis=0) - .5).astype(int))
    hi = np.minimum(SIZE, np.floor(uv.max(axis=0) - .5).astype(int) + 1)
    if (hi <= lo).any():
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    e1, e2 = uv[1] - uv[0], uv[2] - uv[0]
    det = e1[0] * e2[1] - e1[1] * e2[0]
    if abs(det) < 1e-9:
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    yy, xx = np.ogrid[lo[1]:hi[1], lo[0]:hi[0]]
    dx, dy = xx + .5 - uv[0, 0], yy + .5 - uv[0, 1]
    b = (dx * e2[1] - dy * e2[0]) / det
    c = (e1[0] * dy - e1[1] * dx) / det
    a = 1 - b - c
    hit = (a > 1e-7) & (b > 1e-7) & (c > 1e-7)
    y, x = np.nonzero(hit)
    return y + lo[1], x + lo[0], 1 / (a[hit] / z[0] + b[hit] / z[1] + c[hit] / z[2])


def weak_mask(rgb, depth, intr, camera_nodes, constraints):
    if camera_nodes is None:
        return np.zeros((SIZE, SIZE), np.uint8), 'pre_sheet'
    if (camera_nodes.shape != (NV * NU, 3) or not np.isfinite(camera_nodes).all()
            or constraints.ndim != 2 or constraints.shape[1] != 3 or constraints.dtype.kind not in 'iu'
            or not np.isin(constraints[:, 2], (0, 1)).all()
            or (constraints[:, :2] < 0).any() or (constraints[:, :2] >= NV * NU).any()):
        raise ValueError('invalid render geometry')
    alive = {tuple(sorted((a, b))): bool(k) for a, b, k in constraints}
    if len(alive) != len(constraints):
        raise ValueError('duplicate captured edge')
    live = {tri: all(alive[e] for e in edges) for tri, edges in EDGES.items()}
    if all(live[tri] for _, central, _ in STRIPS for tri in central):
        return np.zeros((SIZE, SIZE), np.uint8), 'intact_central_strip'
    w, h, f, far = intr
    z = camera_nodes[:, 2]
    if not ((0 < z) & (z < far)).all():
        return np.full((SIZE, SIZE), UNKNOWN, np.uint8), 'projection_undefined'
    uv = np.asarray((w / 2 - X0, h / 2 - Y0)) - f * camera_nodes[:, :2] / z[:, None]
    r, g, b = rgb.astype(np.int16).transpose(2, 0, 1)
    pink = (b > g + 5) & (r > b + 30)
    lane = np.zeros_like(pink)
    live_lane = np.zeros_like(pink)
    supported_live = np.zeros_like(pink)
    positive = np.zeros_like(pink)

    def hits(tri):
        ix = np.asarray(tri)
        return pixel_hits(uv[ix], z[ix])

    def bank_support(cell):
        observed = set()
        for tri in cell:
            if not live[tri]:
                continue
            y, x, zz = hits(tri)
            good = pink[y, x] & (np.abs(depth[y, x] - zz) <= .001)
            observed.update(zip(y[good], x[good]))
        return len(observed) >= 2

    for left, central, right in STRIPS:
        both_banks = not all(live[tri] for tri in central) and bank_support(left) and bank_support(right)
        for tri in central:
            y, x, zz = hits(tri)
            lane[y, x] = True
            d = depth[y, x]
            if live[tri]:
                live_lane[y, x] = True
                good = pink[y, x] & (np.abs(d - zz) <= .001)
                supported_live[y[good], x[good]] = True
            elif both_banks:
                good = ~pink[y, x] & (d > zz + .002)
                positive[y[good], x[good]] = True
    if not lane.any():
        return np.full((SIZE, SIZE), UNKNOWN, np.uint8), 'projection_undefined'
    cc, n = ndimage.label(positive & ~live_lane, structure=HOOD)
    kept = np.bincount(cc.ravel(), minlength=n+1) >= 2
    kept[0] = False
    mask = np.full((SIZE, SIZE), UNKNOWN, np.uint8)
    mask[~ndimage.binary_dilation(lane, structure=HOOD, iterations=2) | supported_live] = 0
    mask[kept[cc]] = 1
    return mask, 'supported_aperture' if kept.any() else 'ambiguous_cut'


def open_pinned(path, digest):
    raw = path.read_bytes()
    if sha(raw) != digest:
        raise ValueError(f'{path}: input digest mismatch')
    return np.load(io.BytesIO(raw), allow_pickle=False)


def prepare(root, row, hd_sha):
    path = root / row['rgbd']
    with (open_pinned(path, row['sha256']) as main,
          open_pinned(path.with_name(path.name.replace('tick_', 'hd_', 1)), hd_sha) as hi):
        dl, dh = main['intrinsics'], hi['intrinsics']
        R, t = main['cam_R'], main['cam_t']
        rgb, depth = hi['rgb'], hi['depth']
        if (main['tick'].item() != row['tick'] or dl.shape != (4,) or dh.shape != (4,)
                or tuple(dl[:2]) != (640, 480) or not np.isfinite(dh).all() or min(dh[2:]) <= 0
                or not np.array_equal(dh, dl * (2, 2, 2, 1))
                or rgb.shape != (960, 1280, 3) or rgb.dtype != np.uint8
                or depth.shape != (960, 1280) or R.shape != (3, 3) or t.shape != (3,)
                or not np.isfinite(t).all() or not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-6)
                or np.linalg.det(R) <= 0 or not np.array_equal(R, hi['cam_R'])
                or not np.array_equal(t, hi['cam_t']) or not np.isfinite(depth).all()):
            raise ValueError(f'{path}: malformed or misaligned HD pair')
        if ('label_nodes' in main) != ('label_tissue_alive' in main):
            raise ValueError(f'{path}: partial geometry payload')
        mesh, edges = None, None
        if 'label_nodes' in main:
            nodes = main['label_nodes']
            if nodes.shape != (NV, NU, 3):
                raise ValueError(f'{path}: unexpected mesh')
            mesh = (nodes.reshape(-1, 3) - t) @ R
            edges = main['label_tissue_alive']
    rgb, depth = rgb[CROP], depth[CROP]
    # Invalid positive/negative metric depths are not visibility evidence.
    usable = (depth > 0) & (depth <= np.float32(dh[3]))
    target, kind = weak_mask(rgb, np.where(usable, depth, np.nan), dh, mesh, edges)
    pixels = np.empty((4, SIZE, SIZE), np.float16)
    pixels[:3] = rgb.transpose(2, 0, 1) / np.float32(255)
    pixels[3] = np.clip(depth, 0, dh[3]) / dh[3]
    return pixels, target, kind


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root, rows = frozen_rows(args.index)
    hd = {}
    for episode in {row['episode'] for row in rows}:
        manifest = json.loads((MANIFESTS / f'ep{episode:05d}.json').read_bytes())
        hd[episode] = {f['name']: f['sha256'] for f in manifest['files'] if f['name'].startswith('hd_')}
    plan_raw = PLAN.read_bytes()
    args.out.mkdir()
    reg = args.out / 'label_registration.json'
    write_json(reg, {
        'protocol': json.loads(plan_raw), 'protocol_sha256': sha(plan_raw),
        'index_sha256': sha(args.index.read_bytes()), 'source_sha256': sha(Path(__file__).read_bytes()),
        'parent_data_source_sha256': sha((REPO / 'scripts/learn/tip_data.py').read_bytes()),
        'renderer_source_sha256': sha((REPO / 'src/tissue/tissue.lua').read_bytes()),
    })
    images = open_memmap(args.out / 'inputs.npy', mode='w+', dtype=np.float16, shape=(len(rows), 4, SIZE, SIZE))
    masks = open_memmap(args.out / 'masks.npy', mode='w+', dtype=np.uint8, shape=(len(rows), SIZE, SIZE))
    start = time.perf_counter()

    def job(index_and_row):
        i, row = index_and_row
        digest = hd[row['episode']][f"hd_{row['tick']:03d}.npz"]
        x, y, kind = prepare(root, row, digest)
        images[i] = x
        masks[i] = y
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        return dict(row, cache_index=i, hd_sha256=digest, mask_kind=kind,
                    positives=n1, negatives=n0, unknown=SIZE * SIZE - n0 - n1)

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(job, enumerate(rows)), 1):
            records.append(row)
            if i % 1000 == 0:
                print(f'{i}/{len(rows)} masks', flush=True)
    images.flush()
    masks.flush()
    summary = {}
    for partition in ('train', 'validation'):
        subset = [r for r in records if r['partition'] == partition]
        summary[partition] = {
            'episodes': len({r['episode'] for r in subset}), 'frames': len(subset),
            'positive_episodes': len({r['episode'] for r in subset if r['positives']}),
            'positive_frames': sum(r['positives'] > 0 for r in subset),
            'positives': sum(r['positives'] for r in subset),
            'negatives': sum(r['negatives'] for r in subset),
            'unknown': sum(r['unknown'] for r in subset),
            'kinds': Counter(r['mask_kind'] for r in subset),
        }
    write_json(args.out / 'manifest.json', {
        'registration_sha256': sha(reg.read_bytes()), 'crop_uvxy': [X0, Y0, SIZE, SIZE],
        'inputs_dtype': str(images.dtype), 'masks_dtype': str(masks.dtype),
        'summary': summary, 'rows': records, 'wall_s': time.perf_counter() - start,
    })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
