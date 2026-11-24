"""Compare marker-only, fixed depth rejection, and a learned depth-only gate offline."""
import argparse
import json
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from tip_data import (CALIBRATION, NEGATIVES, POSITIVE, REPO, UNKNOWN, VARIANTS,
                      input_depth, read_pixels, sha, write_json)
from cut_perception import CutPerception
from tip_score import summarize

PLAN = REPO / 'config/learn/tip_marker_gate_v1_dev.json'
WORKERS = 4


def patch_features(depth, intr, estimate):
    if not estimate['valid']:
        return None
    w, h, _, far = intr
    u, v = estimate['uv_px']
    z = estimate['xyz_m'][2]
    if not (0 <= u < w and 0 <= v < h and 0 < z < far):
        return None
    x, y = round(u), round(v)
    if not (4 <= x < w - 4 and 4 <= y < h - 4):
        return None
    patch = depth[y - 4:y + 5, x - 4:x + 5].astype(np.float64)
    if not ((patch > 0) & (patch <= np.float32(far))).all():
        return None
    relative = 1000 * (patch - z)
    feats = []
    for r in (1, 2, 4):
        window = relative[4 - r:5 + r, 4 - r:5 + r]
        feats.extend((np.clip(np.median(window), -20, 20), np.mean(window < -3)))
    feats.append(np.clip(relative[3:6, 3:6].max(), -20, 20))
    return [float(v) for v in feats]


def train_episode(task):
    root, rows, tcp = task
    out = []
    for row in rows:
        rgb, depth, intr = read_pixels(root, row)
        w, h, f, far = intr
        rgb = rgb.astype(np.float32)
        features = {}
        for variant in VARIANTS:
            measured = input_depth(depth, far, variant, row)
            reader = CutPerception(int(w), int(h), f, far, (0, 0, 0),
                                   tcp_local=tcp, ray_marker_fit=True)
            est, _ = reader._scissors(rgb, measured)
            est.update(uv_px=est.get('tip_uv'), xyz_m=est.get('tip'))
            features[variant] = patch_features(measured, intr, est)
        out.append({'rgbd': row['rgbd'], 'status': row['label']['status'], **features})
    return out


def fit(training):
    buckets = {v: Counter() for v in VARIANTS}
    data, target = [], []
    for row in training:
        for variant in VARIANTS:
            x = row[variant]
            if x is None:
                continue
            buckets[variant][row['status']] += 1
            data.append(x)
            target.append(row['status'] == POSITIVE)
    y = np.asarray(target, dtype=bool)
    if not y.any() or y.all():
        raise ValueError('need both classes in the training patches')
    data = np.asarray(data, np.float64)
    mean, scale = data.mean(axis=0), data.std(axis=0)
    scale[scale == 0.] = 1.
    standardized = (data - mean) / scale
    features = np.column_stack((np.ones(len(data)), standardized))
    weights = np.where(y, .5 / y.sum(), .5 / (~y).sum())

    def objective(params):
        logits = features @ params
        loss = np.dot(weights, np.logaddexp(0, np.where(y, -logits, logits)))
        grad = features.T @ (weights * (expit(logits) - y))
        grad[1:] += .01 * params[1:]
        return loss + .005 * (params[1:] @ params[1:]), grad

    fitted = minimize(objective, np.zeros(features.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-13})
    if not fitted.success or not np.isfinite(fitted.x).all() or not np.isfinite(fitted.fun):
        raise RuntimeError(f'logistic fit failed: {fitted.message}')
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'coef': fitted.x[1:].tolist(),
            'intercept': float(fitted.x[0]), 'training_patches_by_label': buckets,
            'objective': float(fitted.fun), 'iterations': int(fitted.nit)}


def validation_features(root, rows, baseline, model):
    mean, scale, coef = (np.asarray(model[k]) for k in ('mean', 'scale', 'coef'))
    results = []
    for row, marker in zip(rows, baseline, strict=True):
        if marker['rgbd'] != row['rgbd']:
            raise ValueError('validation marker and capture ordering differ')
        _, depth, intr = read_pixels(root, row)
        per_variant = {}
        for variant in VARIANTS:
            measured = input_depth(depth, intr[3], variant, row)
            features = patch_features(measured, intr, marker[variant])
            score = None if features is None else float(expit(model['intercept']
                                                              + ((features - mean) / scale) @ coef))
            if score is not None and not np.isfinite(score):
                raise ValueError('non-finite visibility score')
            per_variant[variant] = {'features': features, 'score': score}
        results.append({'rgbd': row['rgbd'], **per_variant})
    return results


def suppress(marker, keep):
    return {'xyz_m': marker['xyz_m'] if keep else None, 'uv_px': marker['uv_px'] if keep else None}


def pair_predictions(rows, marker, features):
    negative_scores = {v: [] for v in VARIANTS}
    blocked_scores = Counter()
    for row, measured in zip(rows, features, strict=True):
        status = row['label']['status']
        if status in NEGATIVES:
            for v in VARIANTS:
                score = measured[v]['score']
                if score is not None:
                    negative_scores[v].append(score)
                    blocked_scores[v] += status == 'depth_blocked'
    tau = max((s for vals in negative_scores.values() for s in vals), default=0.)
    rule, learned, paired = [], [], {v: {} for v in VARIANTS}
    for row, base, measured in zip(rows, marker, features, strict=True):
        status = row['label']['status']
        fixed_row, learned_row = {'rgbd': row['rgbd']}, {'rgbd': row['rgbd']}
        for v in VARIANTS:
            data = measured[v]
            fixed_keeps = data['features'] is not None and data['features'][-1] >= -3
            learned_keeps = data['score'] is not None and data['score'] > tau
            fixed_row[v] = suppress(base[v], fixed_keeps)
            learned_row[v] = suppress(base[v], learned_keeps)
            key = 'both' if fixed_keeps and learned_keeps else (
                'fixed_only' if fixed_keeps else 'learned_only' if learned_keeps else 'neither')
            paired[v].setdefault(status, Counter())[key] += 1
        rule.append(fixed_row)
        learned.append(learned_row)
    return rule, learned, paired, tau, {v: blocked_scores[v] for v in VARIANTS}


def gates(summary, scored_blocked=None):
    checks = {}
    for v in VARIANTS:
        m = summary[v]['all']
        checks[v] = {
            'operating_point_defined': all((m['support_by_label']['depth_blocked'] > 0,
                                            m['accepted_positives'] > 0,
                                            scored_blocked is None or scored_blocked[v] > 0)),
            'coverage_ge_90pct': m['positive_coverage'] is not None and m['positive_coverage'] >= .9,
            'median_lt_0p5mm': m['median_xyz_error_mm'] is not None and m['median_xyz_error_mm'] < .5,
            'gt2mm_fraction_lt_5pct': (m['accepted_gt2mm_fraction'] is not None
                                      and m['accepted_gt2mm_fraction'] < .05),
            'no_output_on_explicit_negative': m['accepted_blocked'] + m['accepted_out_of_view'] == 0,
        }
    return {'checks': checks, 'passes': all(all(c.values()) for c in checks.values())}


def checked_inputs(run, plan):
    if run.name != plan['inputs']['phase_a_run']:
        raise ValueError('wrong v1 run')
    loaded = {}
    for name, digest in plan['inputs']['artifacts_sha256'].items():
        raw = (run / name).read_bytes()
        if sha(raw) != digest:
            raise ValueError(f'frozen v1 artifact changed: {name}')
        loaded[name] = json.loads(raw)
    for name, digest in loaded['registration.json']['source_sha256'].items():
        if sha((REPO / name).read_bytes()) != digest:
            raise ValueError(f'v1 source changed: {name}')
    labels = loaded['labels.json']
    tr = [r for r in labels['rows'] if r['partition'] == 'train']
    va = [r for r in labels['rows'] if r['partition'] == 'validation']
    if (len({r['episode'] for r in tr}) != 270 or len({r['episode'] for r in va}) != 30
            or (len(tr), len(va)) != (5273, 568)):
        raise ValueError('unexpected frozen split')
    return (Path(labels['root']), tr, va, loaded['baseline_predictions.json'],
            loaded['baseline_report.json']['summary'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    plan_raw = PLAN.read_bytes()
    plan = json.loads(plan_raw)
    root, train, val, baseline, baseline_metrics = checked_inputs(args.v1_run, plan)
    args.out.mkdir()
    registration = args.out / 'registration.json'
    write_json(registration, {
        'protocol': plan, 'protocol_sha256': sha(plan_raw),
        'entrypoint_sha256': sha(Path(__file__).read_bytes()), 'workers': WORKERS,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
    })
    tcp = json.loads(CALIBRATION.read_bytes())['locator']['tcp_local_m']
    tasks = [(root, list(ep), tcp) for _, ep in groupby(
        (r for r in train if r['label']['status'] != UNKNOWN), key=lambda r: r['episode'])]
    start = time.perf_counter()
    train_features = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for n, features in enumerate(pool.map(train_episode, tasks), 1):
            train_features.extend(features)
            if n % 50 == 0 or n == len(tasks):
                print(f'{n}/{len(tasks)} training episodes featurized', flush=True)
    if len(train_features) != sum(r['label']['status'] != UNKNOWN for r in train):
        raise ValueError('incomplete training feature extraction')
    train_path = args.out / 'train_features.json'
    write_json(train_path, train_features)
    model = fit(train_features)
    fit_path = args.out / 'logistic_fit.json'
    write_json(fit_path, model)
    print(f"Logistic fit converged in {model['iterations']} iterations", flush=True)
    measured = validation_features(root, val, baseline, model)
    rule, learned, paired, tau, scored_blocked = pair_predictions(val, baseline, measured)
    paired_path = args.out / 'validation_predictions.json'
    write_json(paired_path, {'features': measured, 'depth_rule': rule, 'learned_gate': learned})
    rule_metrics, learned_metrics = summarize(val, rule), summarize(val, learned)
    rule_gates, learned_gates = gates(rule_metrics), gates(learned_metrics, scored_blocked)
    differences = {v: learned_metrics[v]['all']['accepted_positives']
                   - rule_metrics[v]['all']['accepted_positives'] for v in VARIANTS}
    benefit = learned_gates['passes'] and (
        not rule_gates['passes'] or (min(differences.values()) >= 0 and max(differences.values()) > 0))
    report = {
        'scope': 'Paired Phase A development/calibration only, no independent evaluation or control',
        'registration_sha256': sha(registration.read_bytes()),
        'train_features_sha256': sha(train_path.read_bytes()), 'fit_sha256': sha(fit_path.read_bytes()),
        'paired_predictions_sha256': sha(paired_path.read_bytes()),
        'threshold_candidate': tau, 'scored_blocked_by_variant': scored_blocked,
        'baseline': baseline_metrics, 'depth_rule': rule_metrics, 'learned_gate': learned_metrics,
        'gates': {'depth_rule': rule_gates, 'learned_gate': learned_gates},
        'paired_decisions_by_label': paired, 'learned_minus_fixed_positive_count': differences,
        'learned_development_benefit': benefit,
        'wall_s': time.perf_counter() - start,
    }
    write_json(args.out / 'report.json', report)
    print(json.dumps({'gates': report['gates'], 'learned_development_benefit': benefit,
                      'learned_minus_fixed_positive_count': differences}, indent=2), flush=True)


if __name__ == '__main__':
    main()
