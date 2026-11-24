"""Validate Phase A frame joins and write a compact, metadata-only pilot index."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
NOTES = REPO / 'config/learn/learned_perception_v1_scoring_notes.json'
MANIFESTS = REPO / 'validation/task/learned_perception_v1/dev'


def read_frame(path):
    with np.load(path, allow_pickle=False) as frame:
        has_nodes = 'label_nodes' in frame
        has_length = 'label_cut_length' in frame
        if has_nodes != has_length:
            raise ValueError(f'{path}: partial structural label')
        row = None
        if has_length:
            mm = float(frame['label_cut_length'].item()) * 1000
            row = round(mm) if math.isfinite(mm) else -1
            if not 0 <= row <= 13 or not math.isclose(mm, row, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f'{path}: invalid structural length {mm} mm')
            nodes = frame['label_nodes']
            if nodes.shape != (14, 24, 3) or not np.isfinite(nodes).all():
                raise ValueError(f'{path}: invalid structural nodes')
        grip = float(frame['label_grip'].item())
        if math.isclose(grip, 0.5, rel_tol=0, abs_tol=1e-6):
            opened = True
        elif math.isclose(grip, 0.0, rel_tol=0, abs_tol=1e-6):
            opened = False
        else:
            raise ValueError(f'{path}: missing or off-command grip {grip}')
        return int(frame['tick'].item()), opened, row


def episode(root, index, partition):
    name = f'ep{index:05d}'
    manifest = json.loads((MANIFESTS / f'{name}.json').read_bytes())
    if manifest['index'] != index or manifest['split'] != 'dev' or manifest['tag'] != 'dev':
        raise ValueError(f'{name}: not a Phase A development manifest')
    directory = root / name
    main = sorted(directory.glob('tick_*.npz'))
    hd = {p.name for p in directory.glob('hd_*.npz')}
    n = manifest['ticks']
    if not n or len(main) != n or len(hd) != n or len(manifest['trace']) != n:
        raise ValueError(f'{name}: incomplete frame, HD or trace sequence')
    files = {f['name']: f for f in manifest['files']}
    count, previous_open, seen_sheet = 0, False, False
    rows = []
    for tick, path in enumerate(main, 1):
        actual_tick, opened, row = read_frame(path)
        twin = f'hd_{tick:03d}.npz'
        if path.name != f'tick_{tick:03d}.npz' or actual_tick != tick or twin not in hd:
            raise ValueError(f'{name}: mismatched tick {tick}')
        for fname in (path.name, twin):
            if (directory / fname).stat().st_size != files[fname]['bytes']:
                raise ValueError(f'{name}: frame length disagrees with manifest: {fname}')
        trace = manifest['trace'][tick - 1]
        if trace['tick'] != tick or not math.isclose(
                trace['cut_length_m'] * 1000, 0 if row is None else row,
                rel_tol=0, abs_tol=1e-6):
            raise ValueError(f'{name}: frame and trace disagree at {tick}')
        if (tick == 1 and row is not None) or (seen_sheet and row is None):
            raise ValueError(f'{name}: unexpected sheet lifecycle at {tick}')
        count += int(previous_open and not opened)
        previous_open = opened
        seen_sheet |= row is not None
        rows.append({'tick': tick, 'rgbd': f'{name}/{path.name}', 'row': row, 'closure_proxy': count})
    return {'index': index, 'partition': partition, 'status': manifest['status'],
            'advance_um': round(manifest['sampled']['advance_m'] * 1_000_000), 'frames': rows}


def split_order(index):
    return hashlib.sha256(f'learned_perception_v1/phase_a_partition/{index}'.encode()).digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Phase A dev directory on the SSD')
    parser.add_argument('--out', type=Path, required=True, help='fresh output JSON path')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to replace an existing index')
    order = sorted(range(300), key=split_order)
    eps = [episode(args.root, index, 'validation' if rank < 30 else 'train')
           for rank, index in enumerate(order)]
    counts = Counter('pre_sheet' if f['row'] is None else str(f['row'])
                     for ep in eps for f in ep['frames'])
    report = {'scope': 'Phase A only; closure proxy not yet validated against a command log',
              'scoring_notes_sha256': hashlib.sha256(NOTES.read_bytes()).hexdigest(),
              'root': str(args.root.resolve()), 'training_smoke_episodes': order[30:98],
              'exposures_by_label': dict(counts),
              'episodes_by_final_closure_count': dict(Counter(e['frames'][-1]['closure_proxy'] for e in eps)),
              'episodes': eps}
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps(dict(episodes=len(eps), main_frames=sum(counts.values()),
                          labels=dict(counts), closure_counts=report['episodes_by_final_closure_count'])))


if __name__ == '__main__':
    main()
