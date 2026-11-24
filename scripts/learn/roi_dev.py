"""Matched-input development comparison, with no access to H1 or H2."""
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from curve_dev import FixedUpdates, diagnose, sha
from phase_a_index import split_order
from train_pilot import Frames, NOTES, REPO, frame_losses

PLAN = REPO / 'config/learn/learned_perception_v1_roi_dev.json'


def flatten(episodes):
    return [(frame, ep['advance_um']) for ep in episodes for frame in ep['frames']]


def labels(items):
    return np.array([-1 if frame['row'] is None else frame['row'] for frame, _ in items])


def metadata_baseline(train, validation, valid):
    pair, count, overall = defaultdict(Counter), defaultdict(Counter), Counter()
    for frame, advance in train:
        row, closed = frame['row'], frame['closure_proxy']
        if row is not None:
            pair[closed, advance][row] += 1
            count[closed][row] += 1
            overall[row] += 1
    mode = lambda histogram: min(histogram, key=lambda x: (-histogram[x], x))
    pred = np.array([mode(pair.get((frame['closure_proxy'], cap))
                          or count.get(frame['closure_proxy']) or overall)
                     for frame, cap in validation])
    y = labels(validation)
    nondefault = np.array([advance != 3000 for _, advance in validation])
    return {'all': diagnose(y, pred, np.ones(y.shape, dtype=bool)),
            'common': diagnose(y, pred, valid),
            'nondefault': diagnose(y[nondefault], pred[nondefault], np.ones(nondefault.sum(), bool)),
            'nondefault_common': diagnose(y[nondefault], pred[nondefault], valid[nondefault])}


def arrays(path):
    with np.load(path, allow_pickle=False) as frame:
        return frame['rgb'].astype(np.float32) / 255, frame['depth']


def differences(rgb1, d1, rgb2, d2):
    return {'rgb_equal': bool(np.array_equal(rgb1, rgb2)),
            'depth_equal': bool(np.array_equal(d1, d2)),
            'rgb_mean_abs_255': float(np.abs(rgb1 - rgb2).mean() * 255),
            'depth_mean_abs_mm': float(np.abs(d1 - d2).mean() * 1000)}


def image_pairs(root, episodes, lookup, valid, crops):
    pairs = []
    for ep in episodes:
        earliest = {}
        for frame in ep['frames']:
            if frame['row'] in (8, 9):
                earliest.setdefault(frame['row'], frame)
        if len(earliest) != 2:
            continue
        a, b = (earliest[k] for k in (8, 9))
        i, j = lookup[a['rgbd']], lookup[b['rgbd']]
        record = {'episode': ep['index'], 'partition': ep['partition'], 'tick_8': a['tick'],
                  'tick_9': b['tick'], 'full640': differences(*arrays(root / a['rgbd']),
                                                              *arrays(root / b['rgbd']))}
        if valid[i] and valid[j]:
            for arm, crop in crops.items():
                record[arm] = differences(crop[i, :3], crop[i, 3], crop[j, :3], crop[j, 3])
        pairs.append(record)
    return pairs


class ArmFrames(torch.utils.data.Dataset):
    def __init__(self, root, items, crop_path, lookup):
        self.paths = Frames(root, [{'frames': [frame for frame, _ in items]}])
        self.indices = [lookup[frame['rgbd']] for frame, _ in items]
        self.crop_path = crop_path
        self.cache = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.crop_path is None:
            return self.paths[i]
        if self.cache is None:
            self.cache = np.load(self.crop_path, mmap_mode='r', allow_pickle=False)
        return torch.from_numpy(self.cache[self.indices[i]].copy()), self.paths.rows[i][1]


def scores(y, p, eligible, cutoffs):
    row, conf = p.argmax(1), p.max(1)
    false = eligible & np.isin(row, (9, 10)) & ~np.isin(y, (9, 10))
    largest = float(conf[false].max()) if false.any() else None
    return {'max_false_endpoint_confidence': largest,
            'fixed': {str(t): diagnose(y, row, eligible & (conf >= t)) for t in cutoffs},
            'in_sample_zero_false_reference': diagnose(
                y, row, eligible if largest is None else eligible & (conf > largest))}


def train_arm(root, arm, train, validation, lookup, valid, cache, plan):
    start = time.perf_counter()
    s = plan['training']
    torch.manual_seed(s['seed'])
    crop = None if arm == 'full640' else cache / f'{arm}.npy'
    datasets = [ArmFrames(root, items, crop, lookup) for items in (train, validation)]
    generator = torch.Generator().manual_seed(s['seed'])
    make = lambda ds, shuffle: torch.utils.data.DataLoader(
        ds, batch_size=s['batch_size'], shuffle=shuffle, drop_last=shuffle,
        num_workers=2, pin_memory=True, persistent_workers=True,
        generator=generator if shuffle else None)
    train_loader, val_loader = make(datasets[0], True), make(datasets[1], False)
    model = resnet18(weights=None, num_classes=14)
    model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    sums = torch.zeros(2, device='cuda', dtype=torch.float64)
    counts = torch.zeros(2, device='cuda', dtype=torch.int64)
    for x, y in FixedUpdates(train_loader, s['updates_per_arm']):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss = frame_losses(logits, y)
        loss.mean().backward()
        optimizer.step()
        sheet = y >= 0
        loss = loss.detach().double()
        sums += torch.stack(((loss * sheet).sum(), (loss * ~sheet).sum()))
        counts += torch.stack((sheet.sum(), (~sheet).sum()))
    totals, draws = sums.tolist(), counts.tolist()
    if not all(map(math.isfinite, totals)) or sum(draws) != s['batch_size'] * s['updates_per_arm']:
        raise ValueError(f'{arm}: bad training count or non-finite loss')
    model.eval()
    proba, seen = [], []
    with torch.inference_mode():
        for x, y in val_loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x.cuda(non_blocking=True))
            proba.append(logits.float().softmax(1).cpu().numpy())
            seen.extend(y.tolist())
    p, truth = np.concatenate(proba), labels(validation)
    if not np.isfinite(p).all() or seen != truth.tolist():
        raise ValueError(f'{arm}: invalid predictions or frame order')
    cutoffs = plan['validation']['confidence_cutoffs']
    out = {'seconds': time.perf_counter() - start, 'draws_sheet_pre_sheet': draws,
           'loss_sheet_pre_sheet': [total / n if n else None for total, n in zip(totals, draws)],
           'common': scores(truth, p, valid, cutoffs)}
    if arm == 'full640':
        out['unrestricted'] = scores(truth, p, np.ones(valid.shape, bool), cutoffs)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to replace a previous report')
    plan_raw, index_raw = PLAN.read_bytes(), args.index.read_bytes()
    plan, index = json.loads(plan_raw), json.loads(index_raw)
    episodes = index['episodes']
    order = sorted(range(300), key=split_order)
    if ([e['index'] for e in episodes] != order
            or [e['partition'] for e in episodes] != ['validation'] * 30 + ['train'] * 270
            or index['scoring_notes_sha256'] != sha(NOTES.read_bytes())):
        raise ValueError('not the frozen Phase A index')
    plan_sha, index_sha = sha(plan_raw), sha(index_raw)
    layout_raw = (args.cache / 'layout.json').read_bytes()
    layout = json.loads(layout_raw)
    names = [frame['rgbd'] for frame, _ in flatten(episodes)]
    if (layout['plan_sha256'] != plan_sha or layout['index_sha256'] != index_sha
            or layout['frames'] != names or len(set(names)) != 5841
            or len(layout['valid']) != len(names) or not all(type(v) is bool for v in layout['valid'])):
        raise ValueError('cache does not match plan and Phase A exposures')
    root = Path(index['root'])
    lookup, valid = {f: i for i, f in enumerate(names)}, np.array(layout['valid'])
    crops = {arm: np.load(args.cache / f'{arm}.npy', mmap_mode='r', allow_pickle=False)
             for arm in plan['arms'] if arm != 'full640'}
    if any(a.shape != (5841, 4, 192, 192) or a.dtype != np.float32 for a in crops.values()):
        raise ValueError('unexpected cache array shape or dtype')
    val, train = flatten(episodes[:30]), flatten(episodes[30:])
    valid_val = valid[:len(val)]
    matched_train = [item for item in train if valid[lookup[item[0]['rgbd']]]]
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 CUDA required')
    report = {'plan': plan, 'plan_sha256': plan_sha, 'index_sha256': index_sha,
              'layout_sha256': sha(layout_raw), 'script_sha256': sha(Path(__file__).read_bytes()),
              'pixel_script_sha256': sha(Path(__file__).with_name('roi_pixels.py').read_bytes()),
              'validation_episode_ids': order[:30],
              'matched_train_frames': len(matched_train), 'all_train_frames': len(train),
              'matched_validation_frames': int(valid_val.sum()), 'all_validation_frames': len(val),
              'metadata_proxy_baseline': metadata_baseline(train, val, valid_val),
              'same_trajectory_pairs': image_pairs(root, episodes, lookup, valid, crops),
              'arms': {}, 'status': 'running'}
    with args.out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=2)
    print(f"Baseline and {len(report['same_trajectory_pairs'])} image pairs written", flush=True)
    for arm in plan['arms']:
        report['arms'][arm] = train_arm(root, arm, matched_train, val, lookup, valid_val, args.cache, plan)
        if arm == plan['arms'][-1]:
            report['status'] = 'completed'
        tmp = args.out.with_suffix('.tmp')
        with tmp.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(report, f, indent=2)
            f.write('\n')
        tmp.replace(args.out)
        print(f"{arm}: {report['arms'][arm]['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
