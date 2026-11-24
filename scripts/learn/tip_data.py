"""Input provenance and offline scorer labels for the registered tip study."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from audit_keypoints import (MIN_MARKER_PX, MARKER_COLOURS, SCENARIOS,
                             marker_evidence, point_status)
from phase_a_index import MANIFESTS, NOTES, REPO, split_order

PLAN = REPO / 'config/learn/learned_tip_v1.json'
LOCK = REPO / 'config/learn/learned_tip_v1_episodes.json'
CALIBRATION = REPO / 'config/learn/learned_perception_v1_roi_dev.json'
POSITIVE = 'fiducial_candidate'
NEGATIVES = ('depth_blocked', 'out_of_view')
UNKNOWN = 'uncertain'
VARIANTS = ('clean', 'noise_0p5mm')
TIMING = ('pre_sheet', 'loaded_uncut', 'cut')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def frozen_rows(index_path):
    raw = index_path.read_bytes()
    index, plan, lock = (json.loads(x) for x in (raw, PLAN.read_bytes(), LOCK.read_bytes()))
    lock_sha = lock['phase_a_index_sha256']
    if (sha(raw) != lock_sha or lock_sha != plan['parents']['phase_a_index_sha256']
            or index['scoring_notes_sha256'] != sha(NOTES.read_bytes())):
        raise ValueError('Phase A index/scoring-notes hash mismatch')
    splits = lock['splits']
    locked = [r for s in ('validation', 'train', 'independent_evaluation') for r in splits[s]]
    if ([len(splits[s]) for s in ('validation', 'train', 'independent_evaluation')] != [30, 270, 64]
            or len({r['index'] for r in locked}) != 364
            or len({int(r['seed']) for r in locked}) != 364
            or [r['index'] for r in locked[:300]] != sorted(range(300), key=split_order)):
        raise ValueError('unexpected episode lock')
    episodes = index['episodes']
    if len(episodes) != 300:
        raise ValueError('unexpected Phase A episode count')
    rows = []
    for i, (ep, ref) in enumerate(zip(episodes, locked[:300], strict=True)):
        eid = ref['index']
        name = f'ep{eid:05d}'
        manifest_raw = (MANIFESTS / f'{name}.json').read_bytes()
        m = json.loads(manifest_raw)
        partition = 'validation' if i < 30 else 'train'
        if (ep['index'] != eid or ep['partition'] != partition
                or m['index'] != eid or str(m['seed']) != ref['seed']
                or m['tag'] != 'dev' or m['split'] != 'dev'
                or m['sampled']['scenario'] not in SCENARIOS
                or m['status'] != ep['status'] or m['status'] not in ('done', 'failed')
                or len(ep['frames']) != m['ticks'] or m['ticks'] <= 0):
            raise ValueError(f'{name}: index, episode lock and manifest disagree')
        files = {f['name']: f for f in m['files']}
        for tick, frame in enumerate(ep['frames'], 1):
            fname = f'tick_{tick:03d}.npz'
            if frame['rgbd'] != f'{name}/{fname}' or frame['tick'] != tick:
                raise ValueError(f'{name}: unexpected RGB-D frame at tick {tick}')
            label = frame['row']
            rows.append({
                'episode': eid, 'seed': ref['seed'], 'partition': partition,
                'episode_status': m['status'], 'scenario': m['sampled']['scenario'],
                'timing': (TIMING[0] if label is None else TIMING[1] if label == 0 else TIMING[2]),
                'tick': tick, 'rgbd': frame['rgbd'], 'sha256': files[fname]['sha256'],
                'manifest_sha256': sha(manifest_raw),
            })
    if len(rows) != 5841:
        raise ValueError('unexpected Phase A main-frame count')
    return Path(index['root']), rows


def _open_capture(root, row):
    path = root / row['rgbd']
    raw = path.read_bytes()
    if sha(raw) != row['sha256']:
        raise ValueError(f'{path}: capture digest mismatch')
    return np.load(io.BytesIO(raw), allow_pickle=False)


def _pixels(cap):
    rgb, depth, intr = cap['rgb'], cap['depth'], cap['intrinsics']
    if (rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8
            or depth.shape != (480, 640) or not np.isfinite(depth).all()
            or intr.shape != (4,) or tuple(intr[:2]) != (640, 480)
            or not np.isfinite(intr).all() or min(intr[2:]) <= 0):
        raise ValueError('corrupt RGB-D capture')
    return rgb, depth, intr


def read_label(root, row):
    with _open_capture(root, row) as cap:
        rgb, depth, intr = _pixels(cap)
        R, t, tcp, marks = (cap[k] for k in ('cam_R', 'cam_t', 'label_tcp_M', 'label_marker_pos'))
        if (cap['tick'].item() != row['tick'] or R.shape != (3, 3) or t.shape != (3,)
                or tcp.shape != (4, 4) or marks.shape != (3, 3)
                or not all(np.isfinite(a).all() for a in (R, t, tcp, marks))
                or not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-6)
                or np.linalg.det(R) <= 0):
            raise ValueError(f"{row['rgbd']}: invalid label geometry")
    tip = R.T @ (tcp[:3, 3] - t)
    uv, status = point_status(tip, depth, intr)
    if status == 'coarse_clear':
        counts = [marker_evidence(rgb, depth, pc, color, intr)
                  for pc, color in zip((marks - t) @ R, MARKER_COLOURS, strict=True)]
        status = POSITIVE if min(counts) >= MIN_MARKER_PX else UNKNOWN
    return dict(status=status, xyz_m=tip.tolist(), uv_px=None if uv is None else uv.tolist(),
                intrinsics=intr.tolist())


def read_pixels(root, row):
    with _open_capture(root, row) as cap:
        return _pixels(cap)


def input_depth(depth, far, variant, row):
    if variant == 'clean':
        return np.clip(depth, 0, far).astype(np.float32)
    if variant != 'noise_0p5mm':
        raise ValueError(f'unknown depth stream {variant}')
    digest = hashlib.sha256(f"learned_tip_v1/noise/{row['seed']}/{row['tick']}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], 'big')))
    return np.clip(depth.astype(np.float64) + rng.normal(0., 0.0005, depth.shape), 0., far).astype(np.float32)


def model_input(rgb, depth):
    x = np.empty((4, 480, 640), np.float32)
    x[:3] = rgb.transpose(2, 0, 1) / np.float32(255)
    x[3] = depth
    return x
