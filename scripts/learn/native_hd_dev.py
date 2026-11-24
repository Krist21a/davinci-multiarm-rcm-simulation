"""Train a matched Phase A crop pair and apply old-dev cutoffs to fresh dev trajectories."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from curve_dev import FixedUpdates, diagnose, sha
from native_hd_pixels import native_crop
from phase_a_index import read_frame, split_order
from roi_dev import flatten, labels, scores
from roi_pixels import PLAN as ROI_PLAN, crop, locate, open_pixels
from train_pilot import NOTES, REPO, frame_losses

PLAN = REPO / 'config/learn/learned_perception_v1_native_hd_dev.json'


class CachedFrames(torch.utils.data.Dataset):
    def __init__(self, path, indices, truth):
        self.path, self.indices, self.truth = path, indices, truth
        self.cache = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.cache is None:
            self.cache = np.load(self.path, mmap_mode='r', allow_pickle=False)
        return torch.from_numpy(self.cache[self.indices[i]].copy()), int(self.truth[i])


def old_dev(index_path, old_cache, native_cache, plan_raw):
    raw = index_path.read_bytes()
    index = json.loads(raw)
    episodes = index['episodes']
    order = sorted(range(300), key=split_order)
    if ([e['index'] for e in episodes] != order
            or [e['partition'] for e in episodes] != ['validation'] * 30 + ['train'] * 270
            or index['scoring_notes_sha256'] != sha(NOTES.read_bytes())):
        raise ValueError('not the frozen Phase A index')
    val, train = flatten(episodes[:30]), flatten(episodes[30:])
    names = [frame['rgbd'] for frame, _ in val + train]
    old_raw = (old_cache / 'layout.json').read_bytes()
    native_raw = (native_cache / 'layout.json').read_bytes()
    a, b = json.loads(old_raw), json.loads(native_raw)
    if (len(names) != 5841 or len(set(names)) != len(names)
            or a['frames'] != names or b['frames'] != names
            or a['index_sha256'] != sha(raw) or b['index_sha256'] != sha(raw)
            or a['plan_sha256'] != sha(ROI_PLAN.read_bytes()) or b['plan_sha256'] != sha(plan_raw)
            or b['previous_layout_sha256'] != sha(old_raw)
            or a['valid'] != b['valid'] or len(a['valid']) != len(names)
            or not all(type(v) is bool for v in a['valid'])):
        raise ValueError('Phase A cache pixel order or provenance mismatch')
    valid = np.asarray(a['valid'])
    paths = {'roi640': old_cache / 'roi640.npy', 'native384': native_cache / 'native384.npy'}
    for arm, size in [('roi640', 192), ('native384', 384)]:
        pixels = np.load(paths[arm], mmap_mode='r', allow_pickle=False)
        if pixels.shape != (len(names), 4, size, size) or pixels.dtype != np.float32:
            raise ValueError(f'{arm}: bad cache shape/dtype')
    train_eligible = np.flatnonzero(valid[len(val):])
    val_eligible = valid[:len(val)]
    datasets = {arm: (CachedFrames(path, train_eligible + len(val), labels(train)[train_eligible]),
                      CachedFrames(path, np.arange(len(val)), labels(val)))
                for arm, path in paths.items()}
    provenance = {'index_sha256': sha(raw), 'old_layout_sha256': sha(old_raw),
                  'native_layout_sha256': sha(native_raw), 'old_validation_ids': order[:30],
                  'eligible_train': len(train_eligible), 'eligible_old_val': int(val_eligible.sum())}
    return datasets, val_eligible, provenance


def loader(dataset, plan, train=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=plan['training']['batch_size'], shuffle=train, drop_last=train,
        num_workers=2, persistent_workers=True, pin_memory=True,
        generator=torch.Generator().manual_seed(plan['training']['seed']) if train else None)


def fit(dataset, plan):
    spec = plan['training']
    torch.manual_seed(spec['seed'])
    model = resnet18(weights=None, num_classes=14)
    model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    total = torch.zeros((), device='cuda', dtype=torch.float64)
    draws = 0
    for x, y in FixedUpdates(loader(dataset, plan, train=True), spec['updates_per_arm']):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss = frame_losses(logits, y).mean()
        loss.backward()
        optimizer.step()
        total += loss.detach().double() * len(y)
        draws += len(y)
    mean = total.item() / draws
    if draws != spec['updates_per_arm'] * spec['batch_size'] or not math.isfinite(mean):
        raise ValueError('invalid train draw count or mean loss')
    return model.eval(), {'draws': draws, 'mean_loss': mean}


def predict(model, data):
    truth, probabilities = [], []
    with torch.inference_mode():
        for x, y in data:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x.cuda(non_blocking=True))
            probabilities.append(logits.float().softmax(1).cpu().numpy())
            truth.extend(y.tolist())
    p = np.concatenate(probabilities)
    if len(truth) != len(p) or not np.isfinite(p).all():
        raise ValueError('non-finite or incomplete predictions')
    return np.asarray(truth), p


def cohort_frames(root, plan, plan_sha):
    c = plan['cohort']
    conditions = c['condition_names']
    root = root / c['tag']
    expected = {f'ep{i:05d}' for i in range(c['start'], c['start'] + c['blocks'] * len(conditions))}
    if {p.name for p in root.glob('ep*') if p.is_dir()} != expected:
        raise ValueError('unexpected cohort directory population')
    frames, episodes, anchors = [], [], {}
    for name in sorted(expected):
        folder = root / name
        raw = (folder / 'manifest.json').read_bytes()
        m = json.loads(raw)
        i = int(name[2:])
        block, condition = divmod(i - c['start'], len(conditions))
        sample = m['sampled']
        if (m['index'] != i or m['split'] != 'dev' or m['tag'] != c['tag']
                or m['source_sha256']['native_hd_plan_sha256'] != plan_sha
                or not m['hd_twin'] or m['aux_views'] != 0 or sample['scenario'] != c['scenario']
                or round(sample['advance_m'] * 1e6) != c['advance_um'][condition]
                or round(sample['target_length_m'] * 1e6) != c['target_um'][condition]):
            raise ValueError(f'{name}: unexpected cohort provenance')
        pose = tuple(sample[key] for key in ('yaw_deg', 'pitch_deg', 'insertion_m'))
        if anchors.setdefault(block, pose) != pose:
            raise ValueError(f'{name}: cohort block not initial-target matched')
        summary = {'index': i, 'block': block, 'condition': conditions[condition],
                   'status': m['status'], 'reason': m['reason'], 'error': m['error'],
                   'final_length_m': m['cut_length_m'], 'initial_target': pose,
                   'initial_view_readback': m['initial_view_readback'], 'manifest_sha256': sha(raw)}
        episodes.append(summary)
        if m['status'] not in {'done', 'failed'} or m.get('cleanup_error'):
            continue
        n = m['ticks']
        if len(list(folder.glob('tick_*.npz'))) != n or len(list(folder.glob('hd_*.npz'))) != n:
            raise ValueError(f'{name}: missing or extra images')
        seen_rows = set()
        for t in range(1, n + 1):
            path = folder / f'tick_{t:03d}.npz'
            tick, _, row = read_frame(path)
            if tick != t or t == 1 and row is not None:
                raise ValueError(f'{path}: unexpected tick/label')
            with np.load(path, allow_pickle=False) as frame:
                R, xyz = frame['cam_R'], frame['cam_t']
                if R.shape != (3, 3) or xyz.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(xyz).all():
                    raise ValueError(f'{path}: invalid scored camera pose')
            frames.append({'path': path, 'row': -1 if row is None else row, 'tick': t, 'index': i,
                           'block': block, 'first_near_row': row in (8, 9, 10, 11, 12) and row not in seen_rows,
                           'R': R, 't': xyz})
            seen_rows.add(row)
    if not frames:
        raise ValueError('cohort has no completed-capture trajectories')
    return frames, episodes


def pose_audit(frames, blocks):
    result = []
    for i in range(blocks):
        use = [f for f in frames if f['block'] == i]
        if not use:
            result.append({'block': i, 'frames': 0, 'matched': False})
            continue
        # Compare observed camera extrinsics against one block reference, not target readback.
        max_rotation = float(np.abs(np.stack([f['R'] for f in use]) - use[0]['R']).max())
        max_translation = float(np.linalg.norm(np.stack([f['t'] for f in use]) - use[0]['t'], axis=1).max())
        result.append({'block': i, 'frames': len(use), 'max_rotation_matrix_entry_delta': max_rotation,
                       'max_translation_m': max_translation,
                       'matched': max_rotation <= 1e-6 and max_translation <= 1e-6})
    return result


def fresh_inference(models, frames, batch_size):
    optical = json.loads(PLAN.read_bytes())
    locator = json.loads(ROI_PLAN.read_bytes())['locator']
    half, tcp = locator['half_width_640_px'], locator['tcp_local_m']
    if batch_size != optical['training']['batch_size']:
        raise ValueError('unexpected batch size')
    values = {arm: np.zeros((len(frames), 14), dtype=np.float32) for arm in models}
    valid = np.zeros(len(frames), dtype=bool)
    for start in range(0, len(frames), batch_size):
        crops = {arm: [] for arm in models}
        present = []
        for i in range(start, min(start + batch_size, len(frames))):
            path = frames[i]['path']
            rgb, depth, intr = open_pixels(path, 640)
            uv = locate(rgb, depth, intr, tcp, half)
            if uv is None:
                continue
            crops['roi640'].append(crop(rgb, depth, float(intr[3]), uv, half, 1))
            crops['native384'].append(native_crop(path, uv, half))
            present.append(i)
        if not present:
            continue
        with torch.inference_mode():
            for arm, model in models.items():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(torch.from_numpy(np.stack(crops[arm])).cuda())
                p = logits.float().softmax(1).cpu().numpy()
                if not np.isfinite(p).all():
                    raise ValueError(f'{arm}: non-finite fresh probabilities')
                values[arm][present] = p
        valid[present] = True
    return values, valid


def fresh_scores(frames, p, valid, floor, blocks):
    y = np.asarray([f['row'] for f in frames])
    row, confidence = p.argmax(1), p.max(1)
    keep = valid if floor is None else valid & (confidence > floor)
    first = np.asarray([f['first_near_row'] for f in frames])
    ix = np.asarray([f['block'] for f in frames])
    return {'eligible_exposures': int(valid.sum()), 'applied_old_dev_floor': floor,
            'all_no_confidence_cut': diagnose(y, row, valid),
            'all_locked': diagnose(y, row, keep),
            'first_near_rows_locked': diagnose(y[first], row[first], keep[first]),
            'by_block_locked': {str(i): diagnose(y[ix == i], row[ix == i], keep[ix == i])
                                for i in range(blocks)}}


def save(path, report):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['index', 'old-cache', 'native-cache', 'cohort-parent', 'out']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    locked = args.out.with_name(args.out.stem + '_old_dev_locked.json')
    if args.out.exists() or locked.exists():
        parser.error('refusing to replace existing result/old-dev lock')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 CUDA required')
    plan_raw = PLAN.read_bytes()
    plan = json.loads(plan_raw)
    ds, valid, provenance = old_dev(args.index, args.old_cache, args.native_cache, plan_raw)
    provenance.update(plan_sha256=sha(plan_raw), script_sha256=sha(Path(__file__).read_bytes()),
                      torch=torch.__version__, device=torch.cuda.get_device_name())
    models, arms = {}, {}
    for arm in plan['training']['arms']:
        start = time.perf_counter()
        models[arm], training = fit(ds[arm][0], plan)
        y, p = predict(models[arm], loader(ds[arm][1], plan))
        if not np.array_equal(y, ds[arm][1].truth):
            raise ValueError('old validation exposure order changed')
        arms[arm] = {'training': training, 'seconds': time.perf_counter() - start,
                     'old_development': scores(y, p, valid, plan['validation']['confidence_cutoffs'])}
        print(f"{arm} old dev complete ({arms[arm]['seconds']:.1f}s)", flush=True)
    save(locked, {'plan': plan, 'provenance': provenance, 'arms': arms,
                  'scope': 'Thresholds saved before any new cohort frames are loaded; diagnostic only'})
    frames, episodes = cohort_frames(args.cohort_parent, plan, provenance['plan_sha256'])
    audit = pose_audit(frames, plan['cohort']['blocks'])
    p, valid = fresh_inference(models, frames, plan['training']['batch_size'])
    for arm in models:
        floor = arms[arm]['old_development']['max_false_endpoint_confidence']
        arms[arm]['fresh_development'] = fresh_scores(frames, p[arm], valid, floor, len(audit))
    save(args.out, {'plan': plan, 'provenance': provenance, 'episodes': episodes,
                    'pose_audit': audit, 'fresh_exposures': len(frames), 'arms': arms})
    print(f'Finished; {len(frames)} fresh development exposures', flush=True)


if __name__ == '__main__':
    main()
