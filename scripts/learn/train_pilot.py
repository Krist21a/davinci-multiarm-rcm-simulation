"""One-epoch engineering CUDA pilot over the Phase A pilot index (labels only train the loss; no predictions are emitted)."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

REPO = Path(__file__).resolve().parents[2]
NOTES = REPO / 'config/learn/learned_perception_v1_scoring_notes.json'


class Frames(torch.utils.data.Dataset):
    def __init__(self, root, episodes):
        self.root = root
        self.rows = [(f['rgbd'], -1 if f['row'] is None else f['row'])
                     for e in episodes for f in e['frames']]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, row = self.rows[index]
        with np.load(self.root / name, allow_pickle=False) as data:
            rgb, depth, intrinsics = data['rgb'], data['depth'], data['intrinsics']
        if (rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8
                or depth.shape != (480, 640) or not np.isfinite(depth).all()
                or intrinsics.shape != (4,) or not np.isfinite(intrinsics).all()
                or tuple(intrinsics[:2]) != (640, 480) or intrinsics[3] <= 0):
            raise ValueError(f'{name}: invalid 640 RGB-D or intrinsics')
        x = np.empty((4, 480, 640), dtype=np.float32)
        x[:3] = rgb.transpose(2, 0, 1) / np.float32(255)
        x[3] = np.clip(depth, 0, intrinsics[3])
        return torch.from_numpy(x), row


def frame_losses(logits, rows):
    logits = logits.float()
    hard = F.cross_entropy(logits, rows.clamp_min(0), reduction='none')
    uniform = -F.log_softmax(logits, dim=1).mean(dim=1)
    return torch.where(rows >= 0, hard, uniform)


def run_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    totals = torch.zeros(2, device='cuda', dtype=torch.float64)
    counts = torch.zeros(2, device='cuda', dtype=torch.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for x, rows in loader:
            x = x.to('cuda', non_blocking=True)
            rows = rows.to('cuda', non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
            losses = frame_losses(logits, rows)
            if optimizer is not None:
                losses.mean().backward()
                optimizer.step()
            valid = rows >= 0
            vals = losses.detach().double()
            totals += torch.stack(((vals * valid).sum(), (vals * ~valid).sum()))
            counts += torch.stack((valid.sum(), (~valid).sum()))
    n_sheet, n_pre = counts.tolist()
    sheet, pre = totals.tolist()
    if min(n_sheet, n_pre) == 0 or not all(map(math.isfinite, (sheet, pre))):
        raise RuntimeError('invalid or empty loss bucket')
    return {'sheet_frames': n_sheet, 'pre_sheet_frames': n_pre,
            'sheet_loss': sheet / n_sheet, 'pre_sheet_loss': pre / n_pre,
            'mean_loss': (sheet + pre) / (n_sheet + n_pre)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True, help='Phase A pilot index JSON')
    parser.add_argument('--out', type=Path, required=True, help='fresh output report JSON path')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to replace an existing report')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA bf16 support required')

    index_bytes = args.index.read_bytes()
    index = json.loads(index_bytes)
    notes_sha = hashlib.sha256(NOTES.read_bytes()).hexdigest()
    if notes_sha != index['scoring_notes_sha256']:
        raise RuntimeError('scoring notes sha256 does not match the index record')

    root = Path(index['root'])
    smoke = set(index['training_smoke_episodes'])
    train_eps = [e for e in index['episodes']
                 if e['partition'] == 'train' and e['index'] in smoke]
    val_eps = [e for e in index['episodes'] if e['partition'] == 'validation']
    train_idx = [e['index'] for e in train_eps]
    val_idx = [e['index'] for e in val_eps]
    if (len(train_idx) != len(set(train_idx)) or len(val_idx) != len(set(val_idx))
            or set(train_idx) & set(val_idx) or len(train_idx) != 68 or len(val_idx) != 30):
        raise RuntimeError('expected 68 unique train and 30 unique disjoint validation episodes')

    torch.manual_seed(20261123)
    generator = torch.Generator()
    generator.manual_seed(20261123)
    train_loader = torch.utils.data.DataLoader(
        Frames(root, train_eps), batch_size=8, shuffle=True,
        num_workers=2, pin_memory=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(
        Frames(root, val_eps), batch_size=8, shuffle=False,
        num_workers=2, pin_memory=True)

    model = resnet18(weights=None, num_classes=14)
    model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train = run_epoch(model, train_loader, optimizer)
    validation = run_epoch(model, val_loader)
    wall_s = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()

    report = {'scope': 'one-epoch engineering pilot only; no threshold predictions, holdout, '
                       'H2, formal scoring or checkpoint',
              'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
              'scoring_notes_sha256': notes_sha,
              'train_episodes': len(train_eps), 'validation_episodes': len(val_eps),
              'train': train, 'validation': validation,
              'wall_s': wall_s, 'cuda_peak_allocated_bytes': peak,
              'torch_version': torch.__version__,
              'device_name': torch.cuda.get_device_name(0)}
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
    print(f'report: {args.out}')


if __name__ == '__main__':
    main()
