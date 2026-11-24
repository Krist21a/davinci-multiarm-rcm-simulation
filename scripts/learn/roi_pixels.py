"""Mechanical ROI pixel cache for config/learn/learned_perception_v1_roi_dev.json.

Preprocessing only -- no training, scoring, labels or statistics. For each Phase A main
exposure listed by the pilot index, the existing ray-refined CutPerception._scissors
marker fit locates the projected tool tip on the clean 640 RGB-D (RGB fed 0..255,
matching _mask/estimate conventions). A 192x192 crop centered at the rounded tip is
taken from the 640 frame and, identically centered at twice the pixel, a 384x384 crop
from the paired HD frame average-pooled by 2. Unlocalizable or out-of-bounds records
stay zero in fresh memory maps under one shared validity flag. layout.json is written
only after every row is processed and the arrays are flushed.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'config' / 'learn' / 'learned_perception_v1_roi_dev.json'
EXPECTED_FRAMES = 5841
sys.path.insert(0, str(REPO / 'scripts' / 'vision'))
from cut_perception import CutPerception  # noqa: E402


def open_pixels(path, width):
    with np.load(path, allow_pickle=False) as z:
        rgb, d, intr = z['rgb'], z['depth'], z['intrinsics']
    h = width * 3 // 4
    if (rgb.shape != (h, width, 3) or rgb.dtype != np.uint8 or d.shape != (h, width)
            or intr.shape != (4,) or tuple(intr[:2]) != (width, h)
            or not np.isfinite(d).all() or not np.isfinite(intr).all() or min(intr[2:]) <= 0):
        raise ValueError(f'{path}: invalid native RGB-D')
    return rgb, d, intr


def locate(rgb, depth, intr, tcp, half):
    w, h, f, far = intr
    optics = CutPerception(int(w), int(h), f, far, (0, 0, 0), tcp_local=tcp, ray_marker_fit=True)
    est, _ = optics._scissors(rgb.astype(np.float32), depth)
    if not est['valid'] or not np.isfinite(est['tip_uv']).all() or not 0 < est['tip'][2] < far:
        return None
    u, v = (round(x) for x in est['tip_uv'])
    if not (half <= u <= w - half and half <= v <= h - half):
        return None
    return u, v


def crop(rgb, depth, far, uv, half, scale):
    u, v = (scale * x for x in uv)
    side = 2 * half
    window = np.s_[v - scale * half:v + scale * half, u - scale * half:u + scale * half]
    out = np.empty((4, side * scale, side * scale), np.float32)
    out[:3] = rgb[window].transpose(2, 0, 1) / np.float32(255)
    out[3] = np.clip(depth[window], 0, far)
    if scale == 2:
        out = out.reshape(4, side, 2, side, 2).mean(axis=(2, 4))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True, help='Phase A pilot index JSON')
    parser.add_argument('--out', type=Path, required=True, help='fresh cache directory on the SSD')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to reuse an existing cache directory')

    index_bytes = args.index.read_bytes()
    plan_bytes = PLAN.read_bytes()
    index = json.loads(index_bytes)
    plan = json.loads(plan_bytes)
    frames = [f['rgbd'] for ep in index['episodes'] for f in ep['frames']]
    if len(frames) != EXPECTED_FRAMES or len(set(frames)) != EXPECTED_FRAMES:
        raise RuntimeError(f'expected {EXPECTED_FRAMES} unique rgbd exposures, got '
                           f'{len(frames)} rows / {len(set(frames))} unique')
    root = Path(index['root'])
    tcp = np.asarray(plan['locator']['tcp_local_m'], dtype=float)
    half = int(plan['locator']['half_width_640_px'])
    n = len(frames)

    args.out.mkdir(parents=True)
    side = 2 * half
    roi640 = open_memmap(args.out / 'roi640.npy', mode='w+', dtype=np.float32,
                         shape=(n, 4, side, side))
    roi1280 = open_memmap(args.out / 'roi1280.npy', mode='w+', dtype=np.float32,
                          shape=(n, 4, side, side))
    valid = np.zeros(n, dtype=bool)

    t0 = time.perf_counter()
    for i, name in enumerate(frames):
        p = root / name
        if not p.is_file():
            raise FileNotFoundError(f'missing source exposure {p}')
        rgb, d, intr = open_pixels(p, 640)
        uv = locate(rgb, d, intr, tcp, half)
        if uv is not None:
            hp = p.with_name(p.name.replace('tick_', 'hd_', 1))
            hrgb, hd, hintr = open_pixels(hp, 1280)
            roi640[i] = crop(rgb, d, float(intr[3]), uv, half, 1)
            roi1280[i] = crop(hrgb, hd, float(hintr[3]), uv, half, 2)
            valid[i] = True
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f'{i + 1}/{n} frames, {int(valid.sum())} valid, '
                  f'{time.perf_counter() - t0:.1f}s', flush=True)

    roi640.flush()
    roi1280.flush()
    layout = {'frames': frames, 'valid': valid.tolist(),
              'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
              'plan_sha256': hashlib.sha256(plan_bytes).hexdigest()}
    with (args.out / 'layout.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(layout, stream, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps({'frames': n, 'valid': int(valid.sum()),
                      'wall_s': round(time.perf_counter() - t0, 2),
                      'cache': str(args.out)}))


if __name__ == '__main__':
    main()
