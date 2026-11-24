"""Train and score the frozen eight-epoch tip-only Phase A development candidate."""
import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18

from tip_data import (PLAN, POSITIVE, REPO, UNKNOWN, VARIANTS, input_depth,
                      model_input, read_pixels, sha, write_json)
from tip_score import threshold_and_score

SEED = 20261123


class TipNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        self.encoder = nn.Sequential(nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False),
                                     base.bn1, base.relu, base.maxpool, base.layer1, base.layer2)
        self.position = nn.Conv2d(128, 4, 1)
        self.visibility = nn.Linear(128, 1)

    def forward(self, x):
        features = self.encoder(x)
        return self.position(features), self.visibility(features.mean(dim=(2, 3))).squeeze(1)


class Captures(torch.utils.data.Dataset):
    def __init__(self, root, rows, variant, training=False):
        self.root, self.rows, self.variant, self.training = root, rows, variant, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        rgb, depth, intr = read_pixels(self.root, row)
        x = model_input(rgb, input_depth(depth, intr[3], self.variant, row))
        if not self.training:
            return x, intr, i
        label = row['label']
        if label['status'] != POSITIVE:
            # No coordinate label is used for a negative.
            return x, False, np.zeros(3, np.float64), float(intr[3])
        target = np.asarray([*label['uv_px'], label['xyz_m'][2]], np.float64)
        return x, True, target, float(intr[3])


def cell_coordinates(uv):
    # Midpoints 7.5, 15.5, ... belong to the smaller row/column.
    ij = (torch.ceil((uv + .5) / 8) - 1).long()
    return ij[:, 0].clamp(0, 79), ij[:, 1].clamp(0, 59)


def uv_at(col, row, regression):
    centre = torch.stack((col, row), dim=1).float() * 8 + 3.5
    return centre + 4 * torch.tanh(regression)


def loss_for(maps, logits, positive, target, far):
    maps, logits = maps.float(), logits.float()
    visible = F.binary_cross_entropy_with_logits(logits, positive.float())
    if not positive.any():
        return visible
    maps, target, far = maps[positive], target[positive], far[positive].float()
    col, row = cell_coordinates(target[:, :2])
    ix = torch.arange(len(maps), device=maps.device)
    regressions = maps[ix, 1:, row, col]
    pos = F.cross_entropy(maps[:, 0].flatten(1), row * 80 + col)
    pos = pos + F.smooth_l1_loss(uv_at(col, row, regressions[:, :2]), target[:, :2].float())
    pos = pos + F.smooth_l1_loss(1000 * far * regressions[:, 2].sigmoid(), 1000 * target[:, 2].float())
    return visible + pos


def decode(maps, logits, intrinsics):
    maps, logits = maps.float(), logits.float()
    heat = maps[:, 0].flatten(1)
    cell = heat.argmax(dim=1)
    col, row = cell % 80, cell // 80
    ix = torch.arange(len(maps), device=maps.device)
    regression = maps[ix, 1:, row, col]
    uv = uv_at(col, row, regression[:, :2])
    confidence = logits.sigmoid()
    depth_fraction = regression[:, 2].sigmoid()
    finite = heat.isfinite().all(dim=1) & regression.isfinite().all(dim=1)
    if not logits.isfinite().all().item():
        raise ValueError('non-finite visibility score')
    result = []
    for (u, v), frac, score, valid, (w, h, f, far) in zip(
            uv.tolist(), depth_fraction.tolist(), confidence.tolist(), finite.tolist(),
            intrinsics.tolist(), strict=True):
        z = frac * far
        valid = valid and 0 <= u < w and 0 <= v < h and 0 < z < far
        result.append({'confidence': score, 'uv_px': [u, v] if valid else None,
                       'xyz_m': [(w / 2 - u) * z / f, (h / 2 - v) * z / f, z] if valid else None})
    return result


def verified_inputs(run):
    baseline = json.loads((run / 'baseline_report.json').read_bytes())
    reg = (run / 'registration.json').read_bytes()
    labels = (run / 'labels.json').read_bytes()
    if (baseline['registration_sha256'] != sha(reg) or baseline['labels_sha256'] != sha(labels)
            or baseline['protocol_sha256'] != sha(PLAN.read_bytes())
            or baseline['raw_predictions_sha256'] != sha((run / 'baseline_predictions.json').read_bytes())):
        raise ValueError('baseline or label digest mismatch')
    for name, digest in json.loads(reg)['source_sha256'].items():
        if sha((REPO / name).read_bytes()) != digest:
            raise ValueError(f'baseline source changed since replay: {name}')
    data = json.loads(labels)
    rows = data['rows']
    train = [r for r in rows if r['partition'] == 'train' and r['label']['status'] != UNKNOWN]
    val = [r for r in rows if r['partition'] == 'validation']
    if (len({r['episode'] for r in rows if r['partition'] == 'train'}) != 270
            or len({r['episode'] for r in val}) != 30 or len(rows) != 5841):
        raise ValueError('unexpected training or validation partition')
    return Path(data['root']), train, val, baseline


def train_epochs(net, loader):
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0003, weight_decay=.01)
    history = []
    for epoch in range(1, 9):
        start = time.perf_counter()
        net.train()
        total, n = 0., 0
        for x, positives, targets, far in loader:
            x, positives, targets, far = (t.cuda(non_blocking=True) for t in (x, positives, targets, far))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                maps, logits = net(x)
            loss = loss_for(maps, logits, positives, targets, far)
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError(f'non-finite loss in epoch {epoch}')
            loss.backward()
            optimizer.step()
            n += len(x)
            total += len(x) * value
        if n != len(loader.dataset):
            raise RuntimeError('training frame count mismatch')
        info = {'epoch': epoch, 'frames': n, 'batches': len(loader),
                'mean_batch_loss_weighted_by_batch_size': total / n,
                'wall_s': time.perf_counter() - start}
        history.append(info)
        print(json.dumps(info), flush=True)
    return history


def predict(net, root, rows):
    net.eval()
    result = [{'rgbd': r['rgbd']} for r in rows]
    with torch.inference_mode():
        for variant in VARIANTS:
            loader = torch.utils.data.DataLoader(Captures(root, rows, variant), batch_size=8,
                                                 num_workers=2, pin_memory=True, shuffle=False)
            seen = []
            for x, intr, ids in loader:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    maps, logits = net(x.cuda(non_blocking=True))
                batch = decode(maps, logits, intr)
                seen.extend(ids.tolist())
                for i, prediction in zip(ids.tolist(), batch, strict=True):
                    result[i][variant] = prediction
            if seen != list(range(len(rows))):
                raise RuntimeError('missing or reordered validation predictions')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='directory with completed baseline')
    parser.add_argument('--out', type=Path, required=True, help='fresh training directory')
    args = parser.parse_args()
    root, train, val, baseline = verified_inputs(args.run)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('registered recipe requires CUDA BF16')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    shuffle = torch.Generator().manual_seed(SEED)
    recipe = {
        'protocol': json.loads(PLAN.read_bytes()),
        'protocol_sha256': sha(PLAN.read_bytes()),
        'baseline_report_sha256': sha((args.run / 'baseline_report.json').read_bytes()),
        'labels_sha256': baseline['labels_sha256'],
        'training_source_sha256': sha(Path(__file__).read_bytes()),
        'train_frames_used': len(train), 'validation_frames': len(val),
        'frame_counts_by_label': baseline['frame_counts'],
        'seed': SEED, 'device': torch.cuda.get_device_name(0),
        'cuda_driver': subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0],
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'torchvision': torchvision.__version__, 'numpy': np.__version__,
                     'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
    }
    args.out.mkdir()
    registration = args.out / 'training_registration.json'
    write_json(registration, recipe)
    train_loader = torch.utils.data.DataLoader(
        Captures(root, train, 'clean', training=True), batch_size=8, shuffle=True, generator=shuffle,
        num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False)
    net = TipNet().cuda()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = train_epochs(net, train_loader)
    if not all(t.isfinite().all().item() for t in net.state_dict().values()):
        raise RuntimeError('non-finite checkpoint')
    checkpoint = args.out / 'tip_epoch08.pt'
    with checkpoint.open('xb') as f:
        torch.save(net.state_dict(), f)
    del train_loader
    raw = predict(net, root, val)
    predictions = args.out / 'model_predictions.json'
    write_json(predictions, raw)
    result = threshold_and_score(val, raw)
    report = {
        'scope': 'Tip-only Phase A development; no independent evaluation and no controller',
        'training_registration_sha256': sha(registration.read_bytes()),
        'checkpoint_sha256': sha(checkpoint.read_bytes()),
        'raw_predictions_sha256': sha(predictions.read_bytes()),
        'training_epochs': history, 'wall_s': time.perf_counter() - start,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        **result,
    }
    write_json(args.out / 'model_report.json', report)
    print(json.dumps({k: report[k] for k in ('threshold_candidate', 'operating_point_defined',
                                            'checks', 'development_pass')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
