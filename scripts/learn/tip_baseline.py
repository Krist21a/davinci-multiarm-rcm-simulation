"""Materialize Phase A tip scorer labels and report the raw three-marker baseline."""
import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from tip_data import (CALIBRATION, LOCK, PLAN, REPO, VARIANTS, frozen_rows, input_depth,
                      read_label, read_pixels, sha, write_json)
from cut_perception import CutPerception
from tip_score import summarize


def baseline(root, row, tcp):
    rgb, depth, intr = read_pixels(root, row)
    w, h, f, far = intr
    rgb = rgb.astype(np.float32)
    preds = {'rgbd': row['rgbd']}
    for variant in VARIANTS:
        reader = CutPerception(int(w), int(h), f, far, (0, 0, 0),
                               tcp_local=tcp, ray_marker_fit=True)
        est, _ = reader._scissors(rgb, input_depth(depth, far, variant, row))
        preds[variant] = dict(est, xyz_m=est.get('tip'), uv_px=est.get('tip_uv'))
    return preds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root, rows = frozen_rows(args.index)
    sources = (
        PLAN, LOCK, CALIBRATION, Path(__file__), REPO / 'scripts/learn/tip_data.py',
        REPO / 'scripts/learn/tip_score.py', REPO / 'scripts/learn/audit_keypoints.py',
        REPO / 'scripts/vision/cut_perception.py', REPO / 'scripts/task/visibility.py',
        REPO / 'scripts/learn/phase_a_index.py',
    )
    registration = {
        'protocol': json.loads(PLAN.read_bytes()),
        'index_sha256': sha(args.index.read_bytes()),
        'source_sha256': {str(path.relative_to(REPO)).replace('\\', '/'): sha(path.read_bytes())
                          for path in sources},
    }
    args.out.mkdir()
    write_json(args.out / 'registration.json', registration)
    start = time.perf_counter()
    for row in rows:
        row['label'] = read_label(root, row)
    labels = args.out / 'labels.json'
    write_json(labels, {'root': str(root), 'rows': rows})
    val = [r for r in rows if r['partition'] == 'validation']
    tcp = json.loads(CALIBRATION.read_bytes())['locator']['tcp_local_m']
    predictions = [baseline(root, r, tcp) for r in val]
    raw = args.out / 'baseline_predictions.json'
    write_json(raw, predictions)
    metrics = summarize(val, predictions)
    report = {
        'scope': 'Unfiltered three-marker baseline; Phase A development validation; no controller',
        'protocol_sha256': sha(PLAN.read_bytes()),
        'registration_sha256': sha((args.out / 'registration.json').read_bytes()),
        'labels_sha256': sha(labels.read_bytes()),
        'raw_predictions_sha256': sha(raw.read_bytes()),
        'frame_counts': {part: dict(Counter(r['label']['status'] for r in rows if r['partition'] == part))
                         for part in ('train', 'validation')},
        'summary': metrics,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'wall_s': time.perf_counter() - start,
    }
    write_json(args.out / 'baseline_report.json', report)
    print(json.dumps({v: m['all'] for v, m in metrics.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
