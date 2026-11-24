"""Compare the Phase A snapshot proxy against direct command logs on dev indices 300-311."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phase_a_index import NOTES, read_frame

INDICES = range(300, 312)
TAG = 'dev_closure_audit_v1'
RECOVERY = {'OCCLUDED', 'CAMERA_ADJUST', 'RETRACT_ADJUST', 'RELOCALIZE', 'RESUME', 'CLEAR_VIEW'}


def check_episode(directory):
    raw = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest['status'] not in {'done', 'failed'} or manifest.get('cleanup_error'):
        raise ValueError(f'{directory}: capture or cleanup failed')
    n = manifest['ticks']
    commands = manifest['closure_commands']
    if n < 1 or any(type(e['accepted']) is not bool or type(e['after_tick']) is not int
                    or not 1 <= e['after_tick'] < n for e in commands):
        raise ValueError(f'{directory}: invalid closure command log')
    if len(list(directory.glob('tick_*.npz'))) != n or len(manifest['trace']) != n:
        raise ValueError(f'{directory}: missing or extra observations')
    prev_open, proxy = False, 0
    disagreements = []
    for tick in range(1, n + 1):
        path = directory / f'tick_{tick:03d}.npz'
        actual, opened, row = read_frame(path)
        if actual != tick or tick == 1 and row is not None:
            raise ValueError(f'{path}: unexpected audit exposure')
        proxy += int(prev_open and not opened)
        prev_open = opened
        direct = sum(e['accepted'] and e['after_tick'] < tick for e in commands)
        with np.load(path, allow_pickle=False) as frame:
            count = frame['label_accepted_closures']
            if count.shape != () or count.dtype.kind not in {'i', 'u'}:
                raise ValueError(f'{path}: invalid explicit count format')
            recorded = int(count.item())
        if direct != proxy or direct != recorded:
            disagreements.append({'tick': tick, 'command_log': direct, 'frame_count': recorded,
                                  'grip_proxy': proxy})
    return {'index': manifest['index'], 'scenario': manifest['sampled']['scenario'],
            'status': manifest['status'], 'reason': manifest['reason'], 'frames': n,
            'accepted': sum(e['accepted'] for e in commands),
            'refused': sum(not e['accepted'] for e in commands),
            'recovery_states': sorted({e.get('to') for e in manifest['fsm_log']} & RECOVERY),
            'frame_disagreements': disagreements, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='development audit tag on the SSD')
    parser.add_argument('--out', type=Path, required=True, help='fresh audit comparison JSON')
    args = parser.parse_args()
    expected = {f'ep{i:05d}' for i in INDICES}
    present = {p.name for p in args.root.glob('ep*') if p.is_dir()}
    if args.root.name != TAG or present != expected or args.out.exists():
        parser.error('expected the complete new development audit tag and fresh output')
    episodes = [check_episode(args.root / path) for path in sorted(expected)]
    if [e['index'] for e in episodes] != list(INDICES):
        raise ValueError('manifest and directory indices disagree')
    report = {'scope': 'Development instrumentation only; H1 isolated',
              'scoring_notes_sha256': hashlib.sha256(NOTES.read_bytes()).hexdigest(),
              'episodes': episodes, 'frames': sum(e['frames'] for e in episodes),
              'accepted_closures': sum(e['accepted'] for e in episodes),
              'refused_closures': sum(e['refused'] for e in episodes),
              'recovery_episodes': sum(bool(e['recovery_states']) for e in episodes),
              'mismatched_frames': sum(len(e['frame_disagreements']) for e in episodes)}
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}))
    return int(report['mismatched_frames'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
