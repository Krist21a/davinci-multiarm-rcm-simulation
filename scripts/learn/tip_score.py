"""Scoring for the frozen offline tip protocol; no model fitting or inference."""
import numpy as np

from tip_data import NEGATIVES, POSITIVE, SCENARIOS, TIMING, UNKNOWN, VARIANTS

RATE_COLUMNS = {
    'positive_coverage': (1, 0),
    'accepted_gt2mm_fraction': (2, 1),
    'blocked_accept_fraction': (4, 3),
    'out_of_view_accept_fraction': (6, 5),
    'unknown_accept_fraction': (8, 7),
}


def _observe(label, prediction):
    status = label['status']
    accepted = prediction['xyz_m'] is not None
    error = None
    if accepted and status == POSITIVE:
        error = (float(np.linalg.norm(np.subtract(prediction['uv_px'], label['uv_px']))),
                 float(1000 * np.linalg.norm(np.subtract(prediction['xyz_m'], label['xyz_m']))))
    return status, error, (
        int(status == POSITIVE), int(status == POSITIVE and accepted), int(error is not None and error[1] > 2),
        int(status == 'depth_blocked'), int(status == 'depth_blocked' and accepted),
        int(status == 'out_of_view'), int(status == 'out_of_view' and accepted),
        int(status == UNKNOWN), int(status == UNKNOWN and accepted),
    )


def _rate(n, d):
    return n / d if d else None


def _summary(observations):
    values = [o[2] for o in observations]
    counts = np.asarray(values, int).sum(axis=0).tolist() if values else [0] * 9
    errors = [o[1] for o in observations if o[1] is not None]
    quantiles = np.percentile(errors, [50, 95], axis=0).tolist() if errors else [[None] * 2] * 2
    return {
        'frames': len(observations),
        'support_by_label': dict(zip((POSITIVE, *NEGATIVES, UNKNOWN),
                                     (counts[0], counts[3], counts[5], counts[7]), strict=True)),
        'accepted_positives': counts[1],
        'accepted_positives_gt2mm': counts[2],
        'accepted_blocked': counts[4],
        'accepted_out_of_view': counts[6],
        'accepted_unknown': counts[8],
        'median_uv_error_px': quantiles[0][0], 'p95_uv_error_px': quantiles[1][0],
        'median_xyz_error_mm': quantiles[0][1], 'p95_xyz_error_mm': quantiles[1][1],
        **{name: _rate(counts[n], counts[d]) for name, (n, d) in RATE_COLUMNS.items()},
    }


def _bootstrap(by_episode):
    matrix = np.asarray([np.asarray(by_episode[k], int).sum(axis=0) for k in sorted(by_episode)])
    rng = np.random.Generator(np.random.PCG64(20261125))
    indices = rng.integers(0, len(matrix), size=(2000, len(matrix)))
    samples = matrix[indices].sum(axis=1)
    intervals = {}
    for name, (n, d) in RATE_COLUMNS.items():
        ok = samples[:, d] > 0
        intervals[name] = {
            'percentile_95': np.percentile(samples[ok, n] / samples[ok, d], [2.5, 97.5]).tolist()
            if ok.any() else None,
            'undefined_draws': int((~ok).sum()),
        }
    return intervals


def summarize(rows, raw):
    if len(rows) != len(raw) or not rows:
        raise ValueError('predictions and scorer rows must align')
    summary = {}
    for variant in VARIANTS:
        observations, episodes = [], {}
        for row, pred in zip(rows, raw, strict=True):
            if pred['rgbd'] != row['rgbd']:
                raise ValueError('prediction row mismatch')
            obs = _observe(row['label'], pred[variant])
            observations.append(obs)
            episodes.setdefault(row['episode'], []).append(obs[2])
        summary[variant] = {
            'all': _summary(observations),
            'scenario': {s: _summary([o for r, o in zip(rows, observations) if r['scenario'] == s])
                         for s in SCENARIOS},
            'timing': {s: _summary([o for r, o in zip(rows, observations) if r['timing'] == s])
                       for s in TIMING},
            'episode_bootstrap_2000': _bootstrap(episodes),
        }
    return summary


def threshold_and_score(rows, raw):
    scores = [p[v]['confidence'] for r, p in zip(rows, raw, strict=True)
              if r['label']['status'] in NEGATIVES for v in VARIANTS]
    tau = max((s for s in scores if s is not None), default=0.)
    filtered = []
    for p in raw:
        kept = {'rgbd': p['rgbd']}
        for v in VARIANTS:
            pred = p[v]
            kept[v] = pred if pred['confidence'] > tau else dict(pred, xyz_m=None, uv_px=None)
        filtered.append(kept)
    summary = summarize(rows, filtered)
    checks = {}
    for variant, parts in summary.items():
        m = parts['all']
        checks[variant] = {
            'operating_point_defined': (m['support_by_label']['depth_blocked'] > 0
                                        and m['accepted_positives'] > 0),
            'coverage_ge_90pct': m['positive_coverage'] is not None and m['positive_coverage'] >= .90,
            'median_lt_0p5mm': m['median_xyz_error_mm'] is not None and m['median_xyz_error_mm'] < .5,
            'gt2mm_fraction_lt_5pct': (m['accepted_gt2mm_fraction'] is not None
                                      and m['accepted_gt2mm_fraction'] < .05),
            'no_output_on_explicit_negative': m['accepted_blocked'] + m['accepted_out_of_view'] == 0,
        }
    return {
        'threshold_candidate': tau,
        'operating_point_defined': all(v['operating_point_defined'] for v in checks.values()),
        'checks': checks,
        'development_pass': all(all(v.values()) for v in checks.values()),
        'summary': summary,
    }
