"""Registered adjacent closure exposures for the offline M5 pair reader."""
import json
import math
from pathlib import Path

import numpy as np

from build_visible_slit_labels import CROP, SIZE, open_pinned
from tip_data import MANIFESTS, REPO, frozen_rows, sha

PLAN = REPO / 'config/learn/m5_close_pair_v1.json'
MATCH = REPO / 'config/learn/learned_perception_v1_native_hd_dev.json'


def category(row):
    if row is None or not 0 <= row <= 13:
        raise ValueError('post-closure exposure needs a structural row')
    return 0 if row < 9 else 1 if row <= 10 else 2


def endpoints(episode, frames, advance_um):
    first = {}
    for f in frames:
        if f['row'] in (9, 10):
            first.setdefault('in', f['tick'])
            first.setdefault(str(f['row']), f['tick'])
    return {'episode': episode, 'first': first, 'advance_um': advance_um}


def pair(episode, a, b, cap, pose_a, pose_b):
    ra, ta, intr_a = pose_a
    rb, tb, intr_b = pose_b
    valid = (np.array_equal(intr_a, intr_b) and np.max(np.abs(ra - rb)) <= 1e-6
             and np.linalg.norm(ta - tb) <= 1e-6)
    if b['tick'] != a['tick'] + 1 or b['closure_proxy'] != a['closure_proxy'] + 1:
        raise ValueError(f'{episode}: nonadjacent closure pair')
    return {
        'episode': episode, 'before': a['pixel_index'], 'current': b['pixel_index'],
        'before_tick': a['tick'], 'tick': b['tick'], 'row': b['row'], 'before_row': a['row'],
        'class': category(b['row']), 'closure_count': b['closure_proxy'], 'advance_um': cap,
        'far': float(intr_b[3]), 'eligible': bool(valid),
    }


def geometry(npz):
    R, t, intr = npz['cam_R'], npz['cam_t'], npz['intrinsics']
    if (R.shape != (3, 3) or t.shape != (3,) or intr.shape != (4,)
            or tuple(intr[:2]) != (640, 480) or min(intr[2:]) <= 0
            or not all(np.isfinite(a).all() for a in (R, t, intr))
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0) or np.linalg.det(R) <= 0):
        raise ValueError('invalid camera geometry')
    return R, t, intr


def frozen_phase(data, index_path, plan):
    index = json.loads(index_path.read_bytes())
    root, rows = frozen_rows(index_path)
    cache = data / plan['source']['phase_a_slit_cache']
    mraw = (cache / 'manifest.json').read_bytes()
    if sha(mraw) != plan['source']['phase_a_slit_manifest_sha256']:
        raise ValueError('unexpected slit pixel-cache manifest')
    manifest = json.loads(mraw)
    if manifest['registration_sha256'] != sha((cache / 'label_registration.json').read_bytes()):
        raise ValueError('bad slit cache provenance')
    indexed = {r['rgbd']: r for r in rows}
    lookup = {}
    for i, (recorded, locked) in enumerate(zip(manifest['rows'], rows, strict=True)):
        if {k: recorded[k] for k in locked} != locked or recorded['cache_index'] != i:
            raise ValueError('disagreement in Phase A cache join')
        lookup[locked['rgbd']] = i
    pixels = np.load(cache / 'inputs.npy', mmap_mode='r', allow_pickle=False)
    if pixels.shape != (len(rows), 4, SIZE, SIZE) or pixels.dtype != np.float16:
        raise ValueError('bad slit cache shape or dtype')
    partitions = {name: {'pairs': [], 'endpoints': []} for name in ('train', 'validation')}
    for ep in index['episodes']:
        part = partitions[ep['partition']]
        frames = ep['frames']
        part['endpoints'].append(endpoints(ep['index'], frames, ep['advance_um']))
        for a, b in zip(frames, frames[1:]):
            delta = b['closure_proxy'] - a['closure_proxy']
            if not delta:
                continue
            if delta != 1:
                raise ValueError('invalid Phase A closure proxy')
            records = []
            for f in (a, b):
                with open_pinned(root / f['rgbd'], indexed[f['rgbd']]['sha256']) as cap:
                    if cap['tick'].item() != f['tick']:
                        raise ValueError('Phase A tick disagreement')
                    records.append(geometry(cap))
            part['pairs'].append(pair(
                ep['index'], dict(a, pixel_index=lookup[a['rgbd']]), dict(b, pixel_index=lookup[b['rgbd']]),
                ep['advance_um'], *records))
    if [len(partitions[s]['pairs']) for s in ('train', 'validation')] != [807, 84]:
        raise ValueError('unexpected Phase A closure sample count')
    return pixels, partitions


def frame_label(frame):
    if ('label_nodes' in frame) != ('label_cut_length' in frame):
        raise ValueError('partial structural label')
    if 'label_nodes' not in frame:
        return None
    m = float(frame['label_cut_length'].item()) * 1000
    if not math.isfinite(m) or not 0 <= round(m) <= 13 or abs(m - round(m)) > 1e-6:
        raise ValueError('bad structural row')
    return round(m)


def matched(data):
    raw = MATCH.read_bytes()
    spec = json.loads(raw)['cohort']
    tag = spec['tag']
    root = data / tag
    indices = list(range(312, 336))
    if sorted(p.name for p in root.glob('ep*') if p.is_dir()) != [f'ep{i:05d}' for i in indices]:
        raise ValueError('wrong matched development capture roster')
    output = {'pairs': [], 'endpoints': [], 'manifests': {}}
    pixels = []
    for eid in indices:
        name = f'ep{eid:05d}'
        path = root / name
        manifest_raw = (path / 'manifest.json').read_bytes()
        m = json.loads(manifest_raw)
        pos = (eid - 312) % 4
        cap = round(m['sampled']['advance_m'] * 1e6)
        n = m['ticks']
        if (manifest_raw != (MANIFESTS.parent / tag / f'{name}.json').read_bytes()
                or m['index'] != eid or m['split'] != 'dev' or m['tag'] != tag
                or m['status'] not in ('done', 'failed') or m.get('error') or m.get('cleanup_error')
                or not m['hd_twin'] or m['aux_views'] or m['sampled']['scenario'] != 'nominal'
                or cap != spec['advance_um'][pos] or round(m['sampled']['target_length_m'] * 1e6) != spec['target_um'][pos]
                or m['source_sha256']['native_hd_plan_sha256'] != sha(raw)
                or not n or len(m['trace']) != n):
            raise ValueError(f'{name}: invalid matched cohort manifest')
        output['manifests'][name] = sha(manifest_raw)
        commands = m['closure_commands']
        if any(type(c['accepted']) is not bool or type(c['after_tick']) is not int
               or not 1 <= c['after_tick'] < n for c in commands):
            raise ValueError(f'{name}: malformed closure log')
        accepted = {c['after_tick'] for c in commands if c['accepted']}
        if len(accepted) != sum(c['accepted'] for c in commands):
            raise ValueError(f'{name}: repeated acceptance tick')
        wanted = accepted | {t + 1 for t in accepted}
        files = {f['name']: f['sha256'] for f in m['files']}
        frames, poses = [], {}
        count, opened = 0, False
        for tick in range(1, n + 1):
            with open_pinned(path / f'tick_{tick:03d}.npz', files[f'tick_{tick:03d}.npz']) as main:
                row = frame_label(main)
                g = float(main['label_grip'].item())
                if math.isclose(g, .5, abs_tol=1e-6):
                    now_open = True
                elif math.isclose(g, 0, abs_tol=1e-6):
                    now_open = False
                else:
                    raise ValueError(f'{name}: invalid commanded grip at {tick}')
                count += int(opened and not now_open)
                opened = now_open
                trace = m['trace'][tick-1]
                if (count != sum(t < tick for t in accepted)
                        or main['label_accepted_closures'].item() != count or main['tick'].item() != tick
                        or trace['tick'] != tick or abs(trace['cut_length_m']*1000 - (row or 0)) > 1e-6):
                    raise ValueError(f'{name}: mismatched direct/proxy closure count or structural row')
                frame = {'tick': tick, 'row': row, 'closure_proxy': count}
                if tick in wanted:
                    poses[tick] = geometry(main)
                    frame['pixel_index'] = len(pixels)
                    pixels.append(load_hd(path, tick, files, poses[tick]))
            frames.append(frame)
        output['endpoints'].append(endpoints(eid, frames, cap))
        for t in sorted(accepted):
            output['pairs'].append(pair(eid, frames[t-1], frames[t], cap, poses[t], poses[t+1]))
    if len(output['pairs']) != 84:
        raise ValueError('unexpected matched cohort pair count')
    return pixels, output


def load_hd(path, tick, files, expected):
    name = f'hd_{tick:03d}.npz'
    R, t, low = expected
    with open_pinned(path / name, files[name]) as frame:
        intr, rgb, depth = frame['intrinsics'], frame['rgb'], frame['depth']
        if (not np.array_equal(intr, low * (2, 2, 2, 1))
                or not np.array_equal(frame['cam_R'], R) or not np.array_equal(frame['cam_t'], t)
                or rgb.shape != (960, 1280, 3) or rgb.dtype != np.uint8
                or depth.shape != (960, 1280) or not np.isfinite(depth).all()):
            raise ValueError(f'{path/name}: bad HD pair')
    x = np.empty((4, SIZE, SIZE), dtype=np.float16)
    x[:3] = rgb[CROP].transpose(2, 0, 1) / np.float32(255)
    x[3] = np.clip(depth[CROP], 0, intr[3]) / intr[3]
    return x
