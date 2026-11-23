"""Run the predeclared Phase A length-only validation curve; no holdout access."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from train_pilot import Frames, NOTES, REPO, run_epoch

PLAN = REPO / 'config/learn/learned_perception_v1_curve_dev.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class FixedUpdates:
    def __init__(self, loader, count):
        self.loader, self.count = loader, count

    def __iter__(self):
        batches = iter(self.loader)
        for _ in range(self.count):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(self.loader)
                batch = next(batches)
            yield batch


def predict(model, loader, sigma, seed, expected_rows):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    rows, scores = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to('cuda', non_blocking=True)
            if sigma:
                depth = x[:, 3:4]
                noise = torch.randn(depth.shape, device='cuda', dtype=depth.dtype, generator=generator)
                depth.add_(noise, alpha=sigma).clamp_(0.0, 0.3)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
            scores.append(logits.float().softmax(1).cpu().numpy())
            rows.extend(y.tolist())
    if rows != expected_rows:
        raise ValueError('validation exposures are missing or out of order')
    p = np.concatenate(scores)
    if not np.isfinite(p).all():
        raise ValueError('non-finite model probabilities')
    return p


def rate(a, b):
    return None if not b else a / b


def diagnose(y, row, keep):
    endpoint = keep & np.isin(row, (9, 10))
    true_endpoint = np.isin(y, (9, 10))
    pre, sheet = y < 0, y >= 0
    eight_nine = np.isin(y, (8, 9))
    correct = keep & (row == y)
    false = endpoint & ~true_endpoint
    by_row = {str(k): {'frames': int((y == k).sum()),
                       'numeric': int(((y == k) & keep).sum()),
                       'exact': int(((y == k) & correct).sum()),
                       'endpoint_calls': int(((y == k) & endpoint).sum())}
              for k in sorted(set(y.tolist()) - {-1})}
    n_endpoint = int(true_endpoint.sum())
    hits = int((true_endpoint & endpoint).sum())
    n_valid = int((sheet & keep).sum())
    wrong = int((sheet & keep & ~correct).sum())
    n_8_9 = int((eight_nine & keep).sum())
    right_8_9 = int((eight_nine & correct).sum())
    return {'false_endpoints': int(false.sum()),
            'endpoint_hits': hits, 'true_endpoints': n_endpoint, 'endpoint_recall': rate(hits, n_endpoint),
            'wrong': wrong, 'numeric_sheet_frames': n_valid, 'sheet_frames': int(sheet.sum()),
            'valid_but_wrong': rate(wrong, n_valid),
            'exact_8_9': right_8_9, 'numeric_8_9': n_8_9, 'accuracy_8_9': rate(right_8_9, n_8_9),
            'pre_sheet': {'frames': int(pre.sum()), 'abstain': int((pre & ~keep).sum()),
                          'numeric_non_endpoint': int((pre & keep & ~endpoint).sum()),
                          'false_endpoints': int((pre & endpoint).sum())},
            'by_true_mm': by_row}


def score(y, probabilities, cutoffs):
    by_variant = {key: (p.argmax(1), p.max(1).astype(float)) for key, p in probabilities.items()}
    wrong_endpoint_scores = [conf[np.isin(row, (9, 10)) & ~np.isin(y, (9, 10))]
                             for row, conf in by_variant.values()]
    observed = np.concatenate(wrong_endpoint_scores)
    floor = float(observed.max()) if observed.size else None
    return {'largest_false_endpoint_confidence_either_depth': floor,
            'variants': {key: {
                'fixed_cutoffs': {str(t): diagnose(y, row, conf >= t) for t in cutoffs},
                'in_sample_zero_false_reference': diagnose(
                    y, row, np.ones(y.shape, dtype=bool) if floor is None else conf > floor)}
                for key, (row, conf) in by_variant.items()}}


def train_arm(root, training, validation, plan):
    spec, validate = plan['training'], plan['validation']
    seed, batch = spec['seed'], spec['batch_size']
    torch.manual_seed(seed)
    train = torch.utils.data.DataLoader(
        Frames(root, training), batch_size=batch, drop_last=True, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=2, pin_memory=True,
        persistent_workers=True)
    val = torch.utils.data.DataLoader(Frames(root, validation), batch_size=batch, shuffle=False,
                                      num_workers=2, pin_memory=True, persistent_workers=True)
    model = resnet18(weights=None, num_classes=14)
    model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    train_stats = run_epoch(model, FixedUpdates(train, spec['updates_per_arm']), optimizer)
    if train_stats['sheet_frames'] + train_stats['pre_sheet_frames'] != batch * spec['updates_per_arm']:
        raise ValueError('training draw count does not match the plan')
    y = [-1 if f['row'] is None else f['row'] for e in validation for f in e['frames']]
    predictions = {str(sigma): predict(model, val, sigma, validate['noise_seed'], y)
                   for sigma in validate['depth_sigmas_m']}
    results = score(np.asarray(y), predictions, validate['confidence_cutoffs'])
    return {'episode_ids': [e['index'] for e in training], 'unique_train_frames': len(train.dataset),
            'training': train_stats, 'scores': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix('.tmp').exists():
        parser.error('refusing to reuse output')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with bf16 required')

    plan_raw, index_raw = PLAN.read_bytes(), args.index.read_bytes()
    plan, index = json.loads(plan_raw), json.loads(index_raw)
    notes_sha = sha(NOTES.read_bytes())
    if notes_sha != index['scoring_notes_sha256']:
        raise ValueError('scoring notes changed since the Phase A index was sealed')
    training = [e for e in index['episodes'] if e['partition'] == 'train']
    validation = [e for e in index['episodes'] if e['partition'] == 'validation']
    ids = [e['index'] for e in training + validation]
    if len(training) != 270 or len(validation) != 30 or set(ids) != set(range(300)):
        raise ValueError('not the frozen Phase A partition')
    if [e['index'] for e in training[:68]] != index['training_smoke_episodes']:
        raise ValueError('training prefix does not match the frozen order')

    report = {'plan': plan, 'plan_sha256': sha(plan_raw), 'index_sha256': sha(index_raw),
              'scoring_notes_sha256': notes_sha, 'script_sha256': sha(Path(__file__).read_bytes()),
              'loader_sha256': sha(Path(__file__).with_name('train_pilot.py').read_bytes()),
              'validation_episode_ids': [e['index'] for e in validation],
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'status': 'running', 'arms': {}}
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    temp = args.out.with_suffix('.tmp')
    for n in plan['arms']:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = train_arm(Path(index['root']), training[:n], validation, plan)
        result['seconds'] = time.perf_counter() - start
        if not math.isfinite(result['seconds']):
            raise ValueError('invalid duration')
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['arms'][str(n)] = result
        if n == plan['arms'][-1]:
            report['status'] = 'completed'
        temp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
        os.replace(temp, args.out)
        print(f'{n} episodes: {result["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
