"""Transfer frozen slit features to a paired post-closure M5 terminal candidate."""
import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from m5_close_pair_data import MATCH, PLAN, frozen_phase, matched
from tip_data import REPO, sha, write_json
from train_visible_slit import SlitUNet, up

SEED = 20261127
ARMS = ('single', 'paired')
DEPTHS = ('clean', 'noise_0p5mm')
CHANNELS = 117


def feature_map(net, x):
    hi = net.high(x)
    mid = net.middle(F.avg_pool2d(hi, 2))
    low = net.low(F.avg_pool2d(mid, 2))
    slit = net.head(net.decode_high(up(net.decode_middle(up(low, mid)), hi)))
    maps = [F.avg_pool2d(a.float(), factor) for a, factor in ((x, 8), (hi, 8), (mid, 4), (low, 2))]
    return torch.cat((*maps, F.max_pool2d(slit.float(), 8)), dim=1)


def noisy_crop(x, identity):
    episode, tick, far = identity
    name = f'm5_close_pair_v1/noise/{episode}/{tick}'
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'big')
    rng = np.random.Generator(np.random.PCG64(seed))
    x[3] = np.clip(x[3] + rng.normal(0., .0005 / far, x[3].shape), 0., 1.)


def extract(net, pixels, pairs):
    exposures = {}
    for r in pairs:
        if not r['eligible']:
            continue
        for key, tick in (('before', 'before_tick'), ('current', 'tick')):
            k = r[key]
            identity = (r['episode'], r[tick], r['far'])
            if exposures.setdefault(k, identity) != identity:
                raise ValueError('conflicting optical frame identity')
    keys = sorted(exposures)
    lookup = {k: i for i, k in enumerate(keys)}
    bank = np.empty((len(DEPTHS), len(keys), CHANNELS, 48, 48), dtype=np.float16)
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(keys), 8):
            indices = keys[start:start+8]
            current = np.stack([pixels[i] for i in indices]).astype(np.float32)
            if not np.isfinite(current).all() or np.any((current < 0) | (current > 1)):
                raise ValueError('invalid input RGB-D pixels')
            for stream in range(len(DEPTHS)):
                x = current.copy()
                if stream:
                    for row, i in zip(x, indices, strict=True):
                        noisy_crop(row, exposures[i])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    features = feature_map(net, torch.from_numpy(x).cuda())
                result = features.half().cpu().numpy()
                if not np.isfinite(result).all():
                    raise ValueError('invalid frozen image features')
                bank[stream, start:start+len(indices)] = result
            if start % 400 == 0 or start + len(indices) == len(keys):
                print(f'features {start + len(indices)}/{len(keys)} exposures', flush=True)
    return bank, lookup


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 * CHANNELS, 32, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GroupNorm(4, 48), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GroupNorm(4, 64), nn.SiLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 6 * 6, 96), nn.SiLU(), nn.Linear(96, 3))

    def forward(self, current, previous=None):
        if previous is None:
            previous = delta = torch.zeros_like(current)
        else:
            delta = current - previous
        return self.net(torch.cat((current, previous, delta), dim=1))


def batches(rows):
    return [(d, r) for d in range(len(DEPTHS)) for r in rows if r['eligible']]


def tensors(arm, bank, lookup, examples):
    stream = [d for d, _ in examples]
    now = [lookup[r['current']] for _, r in examples]
    current = torch.from_numpy(bank[stream, now]).cuda().to(torch.bfloat16)
    previous = None
    if arm == 'paired':
        pre = [lookup[r['before']] for _, r in examples]
        previous = torch.from_numpy(bank[stream, pre]).cuda().to(torch.bfloat16)
    return current, previous


def fit(arm, bank, lookup, rows):
    torch.manual_seed(SEED)
    order = torch.Generator().manual_seed(SEED)
    model = Head().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    examples = batches(rows)
    support = Counter(r['class'] for _, r in examples)
    if set(support) != {0, 1, 2}:
        raise ValueError('need training support for all three classes')
    w = torch.tensor([len(examples) / (3 * support[k]) for k in range(3)], dtype=torch.float32, device='cuda')
    history = []
    for epoch in range(1, 9):
        start = time.perf_counter()
        total = 0.
        ids = torch.randperm(len(examples), generator=order).tolist()
        for i in range(0, len(ids), 16):
            use = [examples[j] for j in ids[i:i+16]]
            curr, prev = tensors(arm, bank, lookup, use)
            target = torch.tensor([r['class'] for _, r in use], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scores = model(curr, prev)
            loss = F.cross_entropy(scores.float(), target, weight=w, reduction='sum') / len(use)
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError('non-finite loss')
            loss.backward()
            optimizer.step()
            total += value * len(use)
        summary = {'arm': arm, 'epoch': epoch, 'examples': len(ids), 'loss': total / len(ids),
                   'wall_s': time.perf_counter()-start}
        history.append(summary)
        print(json.dumps(summary), flush=True)
    return model, history


def infer(arm, model, bank, lookup, part):
    result = [{v: None for v in DEPTHS} for _ in part['pairs']]
    rows = [(i, r) for i, r in enumerate(part['pairs']) if r['eligible']]
    model.eval()
    with torch.inference_mode():
        for d, v in enumerate(DEPTHS):
            for i in range(0, len(rows), 16):
                batch = rows[i:i+16]
                curr, prev = tensors(arm, bank, lookup, [(d, r) for _, r in batch])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    probs = model(curr, prev).float().softmax(1)
                if not probs.isfinite().all().item():
                    raise ValueError('non-finite terminal score')
                for (k, _), p in zip(batch, probs.tolist(), strict=True):
                    result[k][v] = p
    return result


def calibration(part, raw):
    first = {(e['episode'], e['first']['in']) for e in part['endpoints'] if 'in' in e['first']}
    pairs = list(zip(part['pairs'], raw, strict=True))
    for v in DEPTHS:
        if (not any(r['class'] != 1 and p[v] is not None for r, p in pairs)
                or not any((r['episode'], r['tick']) in first and p[v] is not None for r, p in pairs)):
            raise ValueError('terminal threshold lacks negative or first-IN calibration evidence')
    return max(p[v][1] for r, p in pairs if r['class'] != 1 for v in DEPTHS if p[v] is not None)


def thresholded(probs, tau):
    return probs is not None and int(np.argmax(probs)) == 1 and probs[1] > tau


def lookup_metadata(rows):
    by_pair, by_close, overall = defaultdict(Counter), defaultdict(Counter), Counter()
    for r in rows:
        key = (r['closure_count'], r['advance_um'])
        by_pair[key][r['row']] += 1
        by_close[key[0]][r['row']] += 1
        overall[r['row']] += 1

    def mode(counter):
        return min(counter, key=lambda k: (-counter[k], k))

    return {'pair': {f'{c}/{a}': mode(v) for (c, a), v in by_pair.items()},
            'closure': {str(c): mode(v) for c, v in by_close.items()}, 'fallback': mode(overall)}


def metadata_ordinal(table, r):
    c, a = str(r['closure_count']), r['advance_um']
    return table['pair'].get(f'{c}/{a}', table['closure'].get(c, table['fallback']))


def score(part, decisions):
    series = list(zip(part['pairs'], decisions, strict=True))
    known = {(r['episode'], r['tick']): (r, y) for r, y in series}
    if len(known) != len(series):
        raise ValueError('duplicate scored exposure')
    first = {}
    for key in ('in', '9', '10', 'in_nondefault'):
        eligible = []
        for ep in part['endpoints']:
            tick = ep['first'].get('in' if key == 'in_nondefault' else key)
            if tick is None:
                continue
            found = known.get((ep['episode'], tick))
            if key == 'in_nondefault' and (ep['advance_um'] == 3000):
                continue
            eligible.append(bool(found and found[1]))
        first[key] = {'detected': sum(eligible), 'total': len(eligible)}
    fp = sum(y and r['class'] != 1 for r, y in series)
    return {
        'pairs': len(series), 'eligible_pairs': sum(r['eligible'] for r, _ in series),
        'false_terminal_count': fp, 'first': first,
        'by_exact_row': {str(k): {'total': sum(r['row'] == k for r, _ in series),
                                  'terminal_candidates': sum(r['row'] == k and y for r, y in series)}
                         for k in sorted({r['row'] for r, _ in series})},
    }


def summarize(part, raw_by_arm, taus, meta):
    return {
        **{arm: {v: score(part, [thresholded(p[v], taus[arm]) for p in raw_by_arm[arm]]) for v in DEPTHS}
           for arm in ARMS},
        'metadata': score(part, [metadata_ordinal(meta, r) in (9, 10) for r in part['pairs']]),
    }


def progression(result):
    checks = {}
    for v in DEPTHS:
        single = result['single'][v]['first']['in']
        paired = result['paired'][v]['first']['in']
        checks[v] = {
            'zero_false_terminal': result['paired'][v]['false_terminal_count'] == 0,
            'first_recall_ge_half': paired['total'] > 0 and 2 * paired['detected'] >= paired['total'],
            'strictly_better_than_single': paired['total'] == single['total']
                                          and paired['detected'] > single['detected'],
        }
    return {'checks': checks, 'pass_development_progression': all(all(g.values()) for g in checks.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(PLAN.read_bytes())
    path = a.data_root / plan['source']['frozen_slit_checkpoint']
    if sha(path.read_bytes()) != plan['source']['frozen_slit_checkpoint_sha256']:
        raise ValueError('wrong frozen slit checkpoint')
    index = a.data_root / 'phase_a_pilot_index_v1.json'
    a.out.mkdir()
    reg = a.out / 'registration.json'
    write_json(reg, {
        'protocol': plan, 'index_sha256': sha(index.read_bytes()),
        'matched_protocol_sha256': sha(MATCH.read_bytes()),
        'source_sha256': {name: sha((REPO / 'scripts/learn' / name).read_bytes()) for name in (
            'm5_close_pair_data.py', 'train_m5_close_pair.py', 'train_visible_slit.py',
            'build_visible_slit_labels.py', 'tip_data.py')},
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'seed': SEED,
    })
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    start = time.perf_counter()
    pixels, phase = frozen_phase(a.data_root, index, plan)
    training, validation = phase['train'], phase['validation']
    phase_path = a.out / 'phase_pairs.json'
    write_json(phase_path, phase)
    meta = lookup_metadata(training['pairs'])
    encoder = SlitUNet().cuda()
    encoder.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    bank, reverse = extract(encoder, pixels, training['pairs'] + validation['pairs'])
    model, history, weights, cal, taus = {}, {}, {}, {}, {}
    for arm in ARMS:
        model[arm], history[arm] = fit(arm, bank, reverse, training['pairs'])
        if not all(t.isfinite().all().item() for t in model[arm].state_dict().values()):
            raise ValueError('non-finite trained state')
        w = a.out / f'{arm}_epoch08.pt'
        with w.open('xb') as file:
            torch.save(model[arm].state_dict(), file)
        weights[arm] = sha(w.read_bytes())
        cal[arm] = infer(arm, model[arm], bank, reverse, validation)
        taus[arm] = calibration(validation, cal[arm])
    del bank, pixels
    seal = a.out / 'locked_operating_points.json'
    write_json(seal, {'thresholds': taus, 'checkpoint_sha256': weights, 'metadata': meta,
                      'phase_pairs_sha256': sha(phase_path.read_bytes())})
    write_json(a.out / 'validation_raw.json', cal)
    print(f'locked thresholds {taus}', flush=True)
    other_images, other = matched(a.data_root)
    new_bank, new_ids = extract(encoder, other_images, other['pairs'])
    raw = {arm: infer(arm, model[arm], new_bank, new_ids, other) for arm in ARMS}
    saved = a.out / 'matched_replay.json'
    write_json(saved, {'population': other, 'predictions': raw})
    metrics = summarize(other, raw, taus, meta)
    result = progression(metrics)
    report = {
        'scope': 'Post-closure development only, no H1, no controller authorization',
        'registration_sha256': sha(reg.read_bytes()), 'operating_points_sha256': sha(seal.read_bytes()),
        'matched_replay_sha256': sha(saved.read_bytes()), 'wall_s': time.perf_counter()-start,
        'training_history': history, 'thresholds': taus, 'validation': summarize(validation, cal, taus, meta),
        'matched_development': metrics, **result,
    }
    write_json(a.out / 'report.json', report)
    print(json.dumps({'first_IN': {arm: {v: metrics[arm][v]['first']['in'] for v in DEPTHS}
                                  for arm in ARMS}, **result}, indent=2), flush=True)


if __name__ == '__main__':
    main()
