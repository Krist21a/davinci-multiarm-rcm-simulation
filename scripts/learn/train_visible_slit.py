"""Fit the registered visible-aperture segmenter on the Phase A weak masks."""
import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from build_visible_slit_labels import PLAN, SIZE
from tip_data import REPO, sha, write_json

SEED = 20261126


class Patches(torch.utils.data.Dataset):
    def __init__(self, root, rows):
        self.root = root
        self.indices = [r['cache_index'] for r in rows]
        self.frames = self.labels = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.frames is None:
            self.frames = np.load(self.root / 'inputs.npy', mmap_mode='r', allow_pickle=False)
            self.labels = np.load(self.root / 'masks.npy', mmap_mode='r', allow_pickle=False)
        k = self.indices[i]
        return (torch.from_numpy(self.frames[k].astype(np.float32)),
                torch.from_numpy(self.labels[k].copy()), k)


def block(nin, nout):
    return nn.Sequential(nn.Conv2d(nin, nout, 3, padding=1),
                         nn.GroupNorm(4, nout), nn.SiLU(),
                         nn.Conv2d(nout, nout, 3, padding=1),
                         nn.GroupNorm(4, nout), nn.SiLU())


def up(x, skip):
    return torch.cat((x.repeat_interleave(2, -2).repeat_interleave(2, -1), skip), dim=1)


class SlitUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.high = block(4, 16)
        self.middle = block(16, 32)
        self.low = block(32, 64)
        self.decode_middle = block(64 + 32, 32)
        self.decode_high = block(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        top = self.high(x)
        middle = self.middle(F.avg_pool2d(top, 2))
        low = self.low(F.avg_pool2d(middle, 2))
        return self.head(self.decode_high(up(self.decode_middle(up(low, middle)), top)))[:, 0]


def masked_loss(logits, target):
    logits = logits.float()
    pos, neg = target == 1, target == 0
    npix, nnix = pos.sum(), neg.sum()
    p = torch.where(pos, F.softplus(-logits), 0.).sum() / npix.clamp_min(1)
    n = torch.where(neg, F.softplus(logits), 0.).sum() / nnix.clamp_min(1)
    terms = (npix > 0).float() + (nnix > 0).float()
    return (p + n) / terms.clamp_min(1)


def checked_data(cache):
    mraw = (cache / 'manifest.json').read_bytes()
    manifest = json.loads(mraw)
    lraw = (cache / 'label_registration.json').read_bytes()
    labels = json.loads(lraw)
    if (manifest['registration_sha256'] != sha(lraw) or labels['protocol_sha256'] != sha(PLAN.read_bytes())
            or labels['source_sha256'] != sha((REPO / 'scripts/learn/build_visible_slit_labels.py').read_bytes())
            or labels['parent_data_source_sha256'] != sha((REPO / 'scripts/learn/tip_data.py').read_bytes())
            or labels['renderer_source_sha256'] != sha((REPO / 'src/tissue/tissue.lua').read_bytes())):
        raise ValueError('weak label cache registration mismatch')
    rows = manifest['rows']
    train = [r for r in rows if r['partition'] == 'train']
    validation = [r for r in rows if r['partition'] == 'validation']
    tr, va = ({r['episode'] for r in part} for part in (train, validation))
    if (len(tr) != 270 or len(va) != 30 or tr & va or len(rows) != len(train) + len(validation)
            or len(rows) != 5841 or len({r['rgbd'] for r in rows}) != len(rows)
            or [r['cache_index'] for r in rows] != list(range(len(rows)))):
        raise ValueError('unexpected frozen Phase A cache split')
    x = np.load(cache / 'inputs.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(cache / 'masks.npy', mmap_mode='r', allow_pickle=False)
    if (x.shape != (len(rows), 4, SIZE, SIZE) or x.dtype != np.float16
            or y.shape != (len(rows), SIZE, SIZE) or y.dtype != np.uint8):
        raise ValueError('unexpected image/mask shape or dtype')
    if len({r['episode'] for r in train if r['positives']}) < 20 or len(
            {r['episode'] for r in validation if r['positives']}) < 3:
        raise ValueError('insufficient labelled aperture support; do not fit')
    eligible = [r for r in train if r['positives'] or r['negatives']]
    return manifest, sha(mraw), eligible, validation


def train_model(model, data, rows):
    pos = sum(r['positives'] > 0 for r in rows)
    neg = len(rows) - pos
    if min(pos, neg) == 0:
        raise ValueError('training sampler needs both frame categories')
    gen = torch.Generator().manual_seed(SEED)
    weights = [1 / (pos if r['positives'] else neg) for r in rows]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(rows), replacement=True, generator=gen)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=8, drop_last=False,
                                         num_workers=2, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    history = []
    for epoch in range(1, 7):
        model.train()
        seen, steps, total = 0, 0, 0.
        start = time.perf_counter()
        for x, y, _ in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x)
            loss = masked_loss(pred, y)
            amount = loss.item()
            if not math.isfinite(amount):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            opt.step()
            seen += len(x)
            steps += 1
            total += amount
        if seen != len(rows):
            raise ValueError('training frame draw count mismatch')
        record = dict(epoch=epoch, frame_draws=seen, mean_batch_loss=total / steps,
                      wall_s=time.perf_counter() - start)
        history.append(record)
        print(json.dumps(record), flush=True)
    return history


def example_indices(rows):
    result, used = {}, set()
    for row in rows:
        if row['positives'] and row['episode'] not in used:
            result[row['cache_index']] = row['rgbd']
            used.add(row['episode'])
            if len(result) == 6:
                break
    return result


def render_examples(path, examples):
    if not examples:
        return
    width, height = 216, 192
    sheet = Image.new('RGB', (3 * width, 28 + (height + 22) * len(examples)), (20, 27, 27))
    text = ImageDraw.Draw(sheet)
    for i, col in enumerate(('ECM RGB (native detail)', 'Weak positive pixels', 'Network pixels > 0.5')):
        text.text((i * width + 8, 7), col, fill='white')
    for i, (key, x, y, pred) in enumerate(examples):
        v, u = np.nonzero(y == 1)
        v0 = int(np.clip(round(v.mean()) - 48, 0, SIZE - 96))
        u0 = int(np.clip(round(u.mean()) - 48, 0, SIZE - 96))
        region = np.s_[v0:v0 + 96, u0:u0 + 96]
        rgb = np.rint(255 * x[:3].transpose(1, 2, 0)).astype(np.uint8)[region]
        parts = [(rgb, None, None), (rgb, y[region] == 1, (72, 245, 197)),
                 (rgb, pred[region], (255, 170, 50))]
        yt = 28 + i * (height + 22)
        text.text((8, yt + 3), key, fill='white')
        for j, (pixels, mask, color) in enumerate(parts):
            view = pixels.copy()
            if mask is not None:
                view[mask] = color
            panel = Image.fromarray(view).resize((height, height), Image.Resampling.NEAREST)
            sheet.paste(panel, (j * width + 8, yt + 22))
    with path.open('xb') as f:
        sheet.save(f, format='PNG')


def score_validation(model, cache, rows, preview_path):
    data = Patches(cache, rows)
    loader = torch.utils.data.DataLoader(data, batch_size=8, num_workers=2, pin_memory=True)
    model.eval()
    ids_wanted = example_indices(rows)
    examples, ids_seen = [], []
    counts = torch.zeros(8, dtype=torch.int64, device='cuda')
    with torch.inference_mode():
        for x, y_cpu, ids in loader:
            y = y_cpu.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x.cuda(non_blocking=True))
            if not logits.isfinite().all().item():
                raise RuntimeError('non-finite validation logits')
            pred, pos, neg = logits > 0, y == 1, y == 0
            fp = pred & neg
            without_positive = ~pos.flatten(1).any(dim=1)
            counts += torch.stack(((pred & pos).sum(), fp.sum(), (~pred & pos).sum(),
                                   (neg & ~pred).sum(), (pred & ~(pos | neg)).sum(),
                                   without_positive.sum(),
                                   (fp.flatten(1).any(dim=1) & without_positive).sum(),
                                   (fp.flatten(1).sum(dim=1) * without_positive).sum()))
            for i, k in enumerate(ids.tolist()):
                ids_seen.append(k)
                if k in ids_wanted:
                    examples.append((ids_wanted[k], x[i].numpy(), y_cpu[i].numpy(), pred[i].cpu().numpy()))
    if ids_seen != [r['cache_index'] for r in rows]:
        raise ValueError('incomplete development validation')
    tp, fp, fn, tn, unknown, no_positive, no_pos_fp_frames, no_pos_fp_pixels = counts.tolist()
    if tp + fn != sum(r['positives'] for r in rows) or fp + tn != sum(r['negatives'] for r in rows):
        raise ValueError('mask count mismatch')
    render_examples(preview_path, examples)

    def ratio(n, d):
        return n / d if d else None

    return dict(tp=tp, fp=fp, fn=fn, tn=tn, predicted_unknown_pixels=unknown,
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn), iou=ratio(tp, tp + fp + fn),
                frames_without_positive_label=no_positive, no_positive_frames_with_known_fp=no_pos_fp_frames,
                known_fp_pixels_in_no_positive_frames=no_pos_fp_pixels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    manifest, digest, train, validation = checked_data(a.cache)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA BF16 support required')
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    a.out.mkdir()
    reg = a.out / 'training_registration.json'
    write_json(reg, {
        'protocol': json.loads(PLAN.read_bytes()), 'protocol_sha256': sha(PLAN.read_bytes()),
        'trainer_sha256': sha(Path(__file__).read_bytes()), 'label_manifest_sha256': digest,
        'eligible_training_frames': len(train), 'split_support': manifest['summary'], 'seed': SEED,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                     'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0)},
    })
    start = time.perf_counter()
    net = SlitUNet().cuda()
    torch.cuda.reset_peak_memory_stats()
    history = train_model(net, Patches(a.cache, train), train)
    if not all(t.isfinite().all().item() for t in net.state_dict().values()):
        raise ValueError('non-finite weights')
    checkpoint = a.out / 'slit_epoch06.pt'
    with checkpoint.open('xb') as f:
        torch.save(net.state_dict(), f)
    preview = a.out / 'development_examples.png'
    result = score_validation(net, a.cache, validation, preview)
    write_json(a.out / 'report.json', {
        'scope': 'Phase A weak-pixel localization only; no independent or controller evaluation',
        'registration_sha256': sha(reg.read_bytes()), 'weak_label_manifest_sha256': digest,
        'checkpoint_sha256': sha(checkpoint.read_bytes()), 'preview_sha256': sha(preview.read_bytes()),
        'epochs': history, 'development_validation_at_0p5': result,
        'wall_s': time.perf_counter() - start, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    })
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
