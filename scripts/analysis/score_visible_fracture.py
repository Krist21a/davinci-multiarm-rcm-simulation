"""Seal image-only predictions on the fracture-visible sheet before reading structural reports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/vision'))
from row9_terminal import measure

PLAN = ROOT / 'config/task/m5_visible_fracture_dev_v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def predict(folder, protocol):
    plan = json.loads(protocol.read_bytes())
    result = dict(protocol_sha256=sha(protocol),
                  observer_sha256=sha(ROOT / 'scripts/vision/row9_terminal.py'),
                  sources={}, cases={})
    for case in plan['cases']:
        optical_file = folder / f'{case}_optical.json'
        optical = json.loads(optical_file.read_bytes())
        if optical['case'] != case or optical['plan_sha256'] != sha(protocol):
            raise ValueError('wrong optical record')
        entries = []
        for row in optical['exposures']:
            tick = row['tick']
            raw = folder / f'{case}_frames/{tick:03d}.npz'
            noisy = folder / f'{case}_frames/{tick:03d}_depth.npy'
            with np.load(raw) as frame, noisy.open('rb') as stream:
                depth = np.load(stream)
                result_frame = measure(frame['rgb'], depth, frame['cam_R'],
                                       frame['cam_t'], frame['f'], row['estimate'])
            entries.append(dict(tick=tick, frame_sha256=sha(raw),
                                depth_sha256=sha(noisy), **result_frame))
        result['sources'][case] = sha(optical_file)
        result['cases'][case] = dict(run_id=optical['run_id'],
                                     registration_sha256=optical['registration_sha256'],
                                     entries=entries)
    put(folder / 'predictions.json', result)
    return result


def score(folder, predictions, protocol):
    sealed = folder / 'predictions.json'
    if json.loads(sealed.read_bytes()) != predictions:
        raise ValueError('optical predictions were not sealed')
    plan = json.loads(protocol.read_bytes())
    decision = dict(protocol_sha256=predictions['protocol_sha256'],
                    predictions_sha256=sha(sealed), cases={})
    for case, predicted in predictions['cases'].items():
        report_file = folder / f'{case}.json'
        report = json.loads(report_file.read_bytes())
        if (report['case'] != case or report['run_id'] != predicted['run_id']
                or report['registration_sha256'] != predicted['registration_sha256']):
            raise ValueError('run identity differs from image predictions')
        cfg = report['config']
        sheet_cfg = {k: plan[k] for k in ('visual_gap_m', 'visual_gap_style',
                                        'visual_gap_start_row') if k in plan}
        if (cfg.get('sheet') != sheet_cfg or not report['strict_truth']
                or cfg['cut']['control_source'] != 'vision'):
            raise ValueError('wrong model or disabled isolation')
        truth = {row['tick']: row for row in report['observations']}
        closings = [e['tick'] for e in report['events'] if e['event'] == 'closure']
        labels = {8: [], 9: []}
        mistakes, invalid_safety = [], []
        for p in predicted['entries']:
            row = truth.get(p['tick'])
            if row is None:
                raise ValueError('unmatched optical and structural tick')
            length = row['cut_length_m']
            if row['torn'] or row['faults'] or length > .010 + 1e-12:
                invalid_safety.append(p['tick'])
            if p['status'] == 'STOP_CANDIDATE' and length < .009 - 1e-12:
                mistakes.append(p['tick'])
            if abs(length - .008) < 1e-12:
                labels[8].append(p)
            if abs(length - .009) < 1e-12:
                labels[9].append(p)
        candidate = (labels[8] if case == 'depth_noise_0p5' else labels[9])
        expected = 'NOT_STOP' if case == 'depth_noise_0p5' else 'STOP_CANDIDATE'
        first = closings[2] if len(closings) >= 3 else float('inf')
        last = closings[3] if len(closings) >= 4 else float('inf')
        qualified = [p['tick'] for p in candidate if p['status'] == expected
                     and first < p['tick'] < last]
        decision['cases'][case] = dict(report_sha256=sha(report_file), final_state=report.get('final_state'),
                                       actual_mm=report['final']['cut_length_m']*1000,
                                       closure_ticks=closings, qualified_ticks=qualified,
                                       below_target_false_stops=mistakes, unsafe_ticks=invalid_safety,
                                       unknown_at_target=[p['tick'] for p in candidate
                                                          if p['status'] == 'UNKNOWN'],
                                       strict=report['strict_truth'], accepted=report.get('accepted'))
    nominal, noisy = decision['cases']['nominal'], decision['cases']['depth_noise_0p5']
    decision['screen_passed'] = bool(
        all(not row['below_target_false_stops'] and not row['unsafe_ticks']
            for row in (nominal, noisy)) and
        any(b == a + 1 for a, b in zip(nominal['qualified_ticks'],
                                      nominal['qualified_ticks'][1:])) and
        any(b == a + 1 for a, b in zip(noisy['qualified_ticks'],
                                      noisy['qualified_ticks'][1:])))
    put(folder / 'decision.json', decision)
    return decision


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', required=True, type=Path)
    ap.add_argument('--protocol', choices=('v1', 'v2', 'v3'), default='v1')
    args = ap.parse_args()
    folder = args.folder
    protocol = PLAN.with_name(f'm5_visible_fracture_dev_{args.protocol}.json')
    decision = score(folder, predict(folder, protocol), protocol)
    print('rendered fracture optical screen:', decision)
    return int(not decision['screen_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
