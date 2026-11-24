"""Native 384x384 HD ROI cache for config/learn/learned_perception_v1_native_hd_dev.json.

Preprocessing only. Reuses the Phase A ROI cache's provenance and validity: for every
previously-valid frame the clean 640 tip is re-located (must succeed again -- any miss is
an error, not a re-exclusion), and the paired HD frame is cropped at twice the integer
center with twice the half width, kept at native resolution (no pooling). Invalid rows
stay zero in the fresh mmap. layout.json is written last after the flush, carrying the
previous layout's sha256 for provenance.
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
sys.path.insert(0, str(REPO / 'scripts' / 'learn'))
import roi_pixels  # noqa: E402
from roi_pixels import locate, open_pixels  # noqa: E402

PLAN = REPO / 'config' / 'learn' / 'learned_perception_v1_native_hd_dev.json'
ROI_PLAN = roi_pixels.PLAN
EXPECTED_FRAMES = 5841


def native_crop(path, uv, half):
    rgb, depth, intr = open_pixels(
        path.with_name(path.name.replace('tick_', 'hd_', 1)), 1280)
    u, v = (2 * x for x in uv)
    h = 2 * half
    region = np.s_[v - h:v + h, u - h:u + h]
    result = np.empty((4, 2 * h, 2 * h), np.float32)
    result[:3] = rgb[region].transpose(2, 0, 1) / np.float32(255)
    result[3] = np.clip(depth[region], 0, intr[3])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True, help='Phase A pilot index JSON')
    parser.add_argument('--previous-cache', type=Path, required=True,
                        help='phase_a_roi_cache_v1 directory (layout.json provenance)')
    parser.add_argument('--out', type=Path, required=True, help='fresh cache directory on the SSD')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to reuse an existing cache directory')

    index_bytes = args.index.read_bytes()
    plan_bytes = PLAN.read_bytes()
    roi_plan_bytes = ROI_PLAN.read_bytes()
    prev_bytes = (args.previous_cache / 'layout.json').read_bytes()
    index = json.loads(index_bytes)
    prev = json.loads(prev_bytes)
    if prev['index_sha256'] != hashlib.sha256(index_bytes).hexdigest():
        raise RuntimeError('previous cache was built from a different index')
    if prev['plan_sha256'] != hashlib.sha256(roi_plan_bytes).hexdigest():
        raise RuntimeError('previous cache was built from a different ROI plan')

    frames = [f['rgbd'] for ep in index['episodes'] for f in ep['frames']]
    if (len(frames) != EXPECTED_FRAMES or len(set(frames)) != EXPECTED_FRAMES
            or frames != prev['frames']):
        raise RuntimeError('frame list does not exactly match the previous cache')
    valid = np.asarray(prev['valid'], dtype=bool)
    if valid.shape != (EXPECTED_FRAMES,):
        raise RuntimeError('previous valid mask has the wrong length')

    root = Path(index['root'])
    plan = json.loads(plan_bytes)
    roi_plan = json.loads(roi_plan_bytes)
    tcp = np.asarray(roi_plan['locator']['tcp_local_m'], dtype=float)
    half = int(roi_plan['locator']['half_width_640_px'])
    n = len(frames)

    args.out.mkdir(parents=True)
    native = open_memmap(args.out / 'native384.npy', mode='w+', dtype=np.float32,
                         shape=(n, 4, 4 * half, 4 * half))
    done = 0
    t0 = time.perf_counter()
    for i, name in enumerate(frames):
        if not valid[i]:
            continue
        p = root / name
        if not p.is_file():
            raise FileNotFoundError(f'missing source exposure {p}')
        rgb, d, intr = open_pixels(p, 640)
        uv = locate(rgb, d, intr, tcp, half)
        if uv is None:
            raise RuntimeError(f'{name}: previously-valid frame failed to re-localize')
        native[i] = native_crop(p, uv, half)
        done += 1
        if done % 500 == 0 or done == int(valid.sum()):
            print(f'{done}/{int(valid.sum())} valid rows, '
                  f'{time.perf_counter() - t0:.1f}s', flush=True)

    native.flush()
    first = int(np.argmax(valid))
    others = np.nonzero(valid)[0]
    check_idx = [first, int(others[len(others) // 2])]
    roi1280 = np.load(args.previous_cache / 'roi1280.npy', mmap_mode='r')
    for i in check_idx:
        pooled = native[i].reshape(4, 2 * half, 2, 2 * half, 2).mean(axis=(2, 4))
        diff = float(np.abs(pooled - roi1280[i]).max())
        if diff > 1e-7:
            raise RuntimeError(f'native 2x-pool disagrees with roi1280 row {i}: {diff}')
        print(f'row {i}: pooled-native vs roi1280 max |diff| = {diff:.3e}', flush=True)

    layout = {'frames': frames, 'valid': valid.tolist(),
              'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
              'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
              'previous_layout_sha256': hashlib.sha256(prev_bytes).hexdigest()}
    with (args.out / 'layout.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(layout, stream, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps({'frames': n, 'valid': int(valid.sum()),
                      'wall_s': round(time.perf_counter() - t0, 2),
                      'cache': str(args.out)}))


if __name__ == '__main__':
    main()
