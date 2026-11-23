"""Freeze one development reader, then score work and one guarded look without retuning."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from curve_dev import diagnose, sha
from native_hd_dev import PLAN as NATIVE_PLAN, fit, loader, old_dev, predict, save
from native_hd_pixels import native_crop
from phase_a_index import read_frame
from roi_dev import scores
from roi_pixels import PLAN as ROI_PLAN, locate, open_pixels
from train_pilot import REPO

PLAN = REPO / 'config/learn/learned_perception_v1_closed_look_dev.json'


def freeze(args, plan, plan_sha):
    if args.lock.exists() or args.weights.exists():
        raise ValueError('refusing to overwrite old development lock or weights')
    native_raw = NATIVE_PLAN.read_bytes()
    native = json.loads(native_raw)
    datasets, valid, provenance = old_dev(args.index, args.old_cache, args.native_cache, native_raw)
    train, val = datasets['native384']
    model, report = fit(train, native)
    y, p = predict(model, loader(val, native))
    if not np.array_equal(y, val.truth):
        raise ValueError('old development exposure order changed')
    old = scores(y, p, valid, native['validation']['confidence_cutoffs'])
    with args.weights.open('xb') as stream:
        torch.save(model.state_dict(), stream)
    save(args.lock, {'scope': 'Closed look development reader; no H1 or candidate freeze',
                     'plan': plan, 'plan_sha256': plan_sha, 'parent_training_sha256': sha(native_raw),
                     'weights_sha256': sha(args.weights.read_bytes()), 'data': provenance,
                     'engine_sha256': sha(Path(__file__).read_bytes()),
                     'training': report, 'old_development': old})
    print(f"Old development reader locked at {old['max_false_endpoint_confidence']}", flush=True)


def gather(root, plan, plan_sha):
    cfg = plan['cohort']
    names = cfg['condition_names']
    anchors = {}
    episodes, images = [], []
    for i in range(cfg['start'], cfg['start'] + len(names) * cfg['blocks']):
        block, cond = divmod(i - cfg['start'], len(names))
        folder = root / cfg['tag'] / f'ep{i:05d}'
        raw = (folder / 'manifest.json').read_bytes()
        m = json.loads(raw)
        p = m['sampled']
        if (m['tag'] != cfg['tag'] or m['split'] != 'dev' or m['index'] != i
                or m['source_sha256']['closed_look_plan_sha256'] != plan_sha
                or m['aux_views'] != 0 or not m['hd_twin'] or p['scenario'] != cfg['scenario']
                or round(p['advance_m'] * 1e6) != cfg['advance_um'][cond]
                or round(p['target_length_m'] * 1e6) != cfg['target_um'][cond]):
            raise ValueError(f'{folder}: cohort manifest mismatch')
        anchor = tuple(p[k] for k in ('yaw_deg', 'pitch_deg', 'insertion_m'))
        if anchors.setdefault(block, anchor) != anchor:
            raise ValueError(f'{folder}: block camera target mismatch')
        row = {'index': i, 'block': block, 'condition': names[cond], 'status': m['status'],
               'reason': m['reason'], 'error': m['error'], 'pair_qualified': False,
               'overcut_work_negative': False, 'manifest_sha256': sha(raw)}
        episodes.append(row)
        if m['status'] not in {'done', 'failed'} or m.get('cleanup_error'):
            row['exclusion'] = 'capture or cleanup failure'
            continue
        trace = m['trace']
        if [t['tick'] for t in trace] != list(range(1, m['ticks'] + 1)):
            raise ValueError(f'{folder}: incomplete ordered trace')
        probes = [e for e in m['world_event_log'] if e['event'] == 'closed_look_probe']
        if len(probes) > 1:
            raise ValueError(f'{folder}: repeated probe')
        if probes:
            e = probes[0]
            work, look = e['work_tick'], e['look_tick']
            row['mechanics'] = e
            phase_ticks = {'work': work}
            if look is not None:
                if look != work + 1:
                    raise ValueError(f'{folder}: look not immediately paired')
                phase_ticks['look'] = look
            length = trace[work - 1]['cut_length_m']
            first = all(abs(t['cut_length_m'] - length) > 1e-12 for t in trace[:work - 1])
            no_more_closures = (bool(m['closure_commands'])
                                and m['closure_commands'][-1]['accepted']
                                and m['closure_commands'][-1]['after_tick'] < work)
            guards_ok = (m['status'] == 'done' and not m['issues'] and e['qualified']
                         and e['structure_signature_unchanged'] and e['grasp_counts_unchanged']
                         and e['length_delta_m'] <= 1e-12 and e['residual_m'] <= 0.0005
                         and e['distance_to_ready_m'] > 0.0005 and e['rotation_entry_delta'] <= 1e-6
                         and e['camera_translation_m'] <= 1e-6)
            row.update(pair_qualified=bool(first and no_more_closures and guards_ok and cond < 3
                                           and round(length * 1000) == (8, 9, 10)[cond]),
                       first_of_exact_terminal_row=first, no_further_closure=no_more_closures)
        else:
            outside = next((t['tick'] for t in trace if t['cut_length_m'] > 0.010 + 1e-12), None)
            if (cond == 3 and outside is not None and m['status'] == 'failed'
                    and m['reason'].startswith('cut exceeds allowed length')):
                row['overcut_work_negative'] = True
                phase_ticks = {'work': outside}
            else:
                row['exclusion'] = 'no registered work/look or overcut work observation'
                continue
        for phase, tick in phase_ticks.items():
            path = folder / f'tick_{tick:03d}.npz'
            captured, opened, mm = read_frame(path)
            if (captured != tick or mm is None
                    or abs(trace[tick - 1]['cut_length_m'] - mm / 1000) > 1e-12):
                raise ValueError(f'{path}: trace and capture disagree')
            if row['pair_qualified'] and opened:
                raise ValueError(f'{path}: supposedly closed paired exposure is open')
            images.append({'index': i, 'phase': phase, 'tick': tick, 'true_mm': mm, 'path': path,
                           'pair_qualified': row['pair_qualified'],
                           'overcut_work_negative': row['overcut_work_negative']})
    return episodes, images


def infer(images, model, floor):
    geometry = json.loads(ROI_PLAN.read_bytes())['locator']
    tcp, half = geometry['tcp_local_m'], geometry['half_width_640_px']
    for entry in images:
        path = entry.pop('path')
        im, depth, intr = open_pixels(path, 640)
        center = locate(im, depth, intr, tcp, half)
        entry.update(locator_valid=center is not None, argmax_mm=None, confidence=None, locked_mm=None)
        if center is None:
            continue
        pixels = native_crop(path, center, half)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(torch.from_numpy(pixels[None]).cuda())
        probs = logits.float().softmax(1)
        value, pred = probs[0].max(0)
        score, mm = float(value.item()), int(pred.item())
        if not np.isfinite(score):
            raise ValueError(f'{path}: invalid probability')
        entry.update(argmax_mm=mm, confidence=score,
                     locked_mm=mm if floor is None or score > floor else None)


def stats(images, field):
    y = np.asarray([im['true_mm'] for im in images], dtype=int)
    p = np.asarray([-1 if im[field] is None else im[field] for im in images], dtype=int)
    return diagnose(y, p, p >= 0)


def result(episodes, images):
    phases = {phase: {field: stats([x for x in images if x['phase'] == phase and x['pair_qualified']], field)
                      for field in ('argmax_mm', 'locked_mm')} for phase in ('work', 'look')}
    negatives = stats([x for x in images if x['overcut_work_negative']], 'locked_mm')
    grouped = {}
    for x in images:
        if x['pair_qualified']:
            grouped.setdefault(x['index'], {})[x['phase']] = x
    gains = losses = 0
    for g in grouped.values():
        w, l = g['work'], g['look']
        if w['true_mm'] != l['true_mm']:
            raise ValueError('mismatched truth in a qualified pair')
        if w['true_mm'] in (9, 10):
            work, look = (im['locked_mm'] in (9, 10) for im in (w, l))
            gains += int(look and not work)
            losses += int(work and not look)
    counts = Counter(g['work']['true_mm'] for g in grouped.values())
    look = phases['look']['locked_mm']
    false = stats(images, 'locked_mm')['false_endpoints']
    criteria = {'all_nine_mechanical_pairs': counts == {8: 3, 9: 3, 10: 3},
                'all_three_overcut_negatives': negatives['sheet_frames'] == 3,
                'zero_false_endpoints_any_checked_frame': not false,
                'paired_improvement_without_losses': gains > 0 and losses == 0,
                'at_least_three_of_six_endpoints': look['endpoint_hits'] >= 3 and look['true_endpoints'] == 6,
                'detects_both_terminal_rows': all(
                    look['by_true_mm'].get(str(r), {}).get('endpoint_calls', 0) > 0 for r in (9, 10))}
    return {'episodes': episodes, 'exposures': images, 'qualified_pairs_by_true_mm': dict(counts),
            'paired_phases': phases, 'overcut_work_negatives_locked': negatives,
            'true_endpoint_gains': gains, 'true_endpoint_losses': losses, 'false_endpoints': false,
            'development_progression_checks': criteria, 'proceed_to_larger_development_qualification': all(criteria.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    a, b = sub.add_parser('freeze'), sub.add_parser('score')
    for p, options in [(a, ('weights', 'lock', 'index', 'old-cache', 'native-cache')),
                       (b, ('weights', 'lock', 'cohort-parent', 'out'))]:
        for option in options:
            p.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 CUDA required')
    raw = PLAN.read_bytes()
    plan, plan_sha = json.loads(raw), sha(raw)
    if args.mode == 'freeze':
        freeze(args, plan, plan_sha)
        return
    if args.out.exists():
        parser.error('refusing to replace an existing result')
    lock_raw = args.lock.read_bytes()
    lock = json.loads(lock_raw)
    if (lock['plan_sha256'] != plan_sha or lock['engine_sha256'] != sha(Path(__file__).read_bytes())
            or lock['weights_sha256'] != sha(args.weights.read_bytes())):
        raise ValueError('locked recipe or weights mismatch')
    model = resnet18(weights=None, num_classes=14)
    model.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    episodes, images = gather(args.cohort_parent, plan, plan_sha)
    floor = lock['old_development']['max_false_endpoint_confidence']
    infer(images, model.cuda().eval(), floor)
    report = result(episodes, images)
    report.update(plan=plan, plan_sha256=plan_sha, lock_sha256=sha(lock_raw), locked_floor=floor)
    save(args.out, report)
    print(json.dumps(report['development_progression_checks']))


if __name__ == '__main__':
    main()
