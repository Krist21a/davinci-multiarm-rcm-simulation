"""M4 shadow-perception live evaluation driver.

Runs run_cut_task.py --mode full with cfg['perception'].mode='shadow' across scenarios and a
one-factor-at-a-time degradation sweep on nominal:
  depth_noise_m   in {0, 0.0005, 0.001, 0.002}
  camera trans    in {0, 1, 2} mm
  camera rot      in {0, 1, 2} deg
Noisy settings run 3 seeds; deterministic settings run once.

Aggregates validation/task/perc_eval_*.json into:
  validation/perception/shadow_eval.json  (run_id, source hashes, per-config stats)
  docs/perception_degradation.png         (error-vs-degradation plot)

Key safety metric reported per config: the rate of "valid but wrong" estimates
(valid=True while error > 2 mm) — estimates must fail invalid, not lie.
"""
import argparse
import json
import math
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / '.venv' / 'Scripts' / 'python.exe'
RUNNER = ROOT / 'scripts' / 'task' / 'run_cut_task.py'
OUT = ROOT / 'validation' / 'task'
REPORT = ROOT / 'validation' / 'perception' / 'shadow_eval.json'
FIG = ROOT / 'docs' / 'perception_degradation.png'

SCENARIOS = ['nominal', 'occlusion_transient', 'slip_regrasp']
SWEEP = [
    ('depth_noise_m', [0.0, 0.0005, 0.001, 0.002], 'depth_noise_m'),
    ('trans_m', [0.0, 0.001, 0.002], 'camera_pose_error.trans_m'),
    ('rot_deg', [0.0, 1.0, 2.0], 'camera_pose_error.rot_deg'),
]
NOISY_KEYS = {'depth_noise_m', 'camera_pose_error.trans_m', 'camera_pose_error.rot_deg'}
SEEDS = [1, 2, 3]
WRONG_MM = 0.002


def run(mode, scenario, perc, name):
    args = [str(PY), str(RUNNER), '--mode', 'full', '--scenario', scenario,
            '--perception', json.dumps(perc), '--name', name]
    r = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    path = OUT / f'{name}.json'
    if not path.exists():
        return None, f'no report (rc={r.returncode}): {r.stdout[-400:]} {r.stderr[-400:]}'
    return json.loads(path.read_text(encoding='utf-8')), None


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    return float(v[min(len(v) - 1, int(math.ceil(q / 100 * len(v))) - 1)])


def summarize(reports):
    """Aggregate per-tick estimates across reports -> per-field stats + validity rates."""
    fields = {'front_pos_err_m': [], 'tip_pos_err_m': [], 'axis_err_rad': [],
              'direction_err_rad': [], 'front_uv_err_px': []}
    counts = {'ticks': 0, 'tissue_valid': 0, 'cut_valid': 0, 'front_valid': 0,
              'scissors_valid': 0, 'tension_valid': 0}
    wrong = {'ticks_eval': 0, 'front_valid_wrong': 0, 'tip_valid_wrong': 0}
    tension_pairs = []
    for rep in reports:
        for e in rep.get('perception') or []:
            counts['ticks'] += 1
            est = e.get('estimate') or {}
            err = e.get('errors') or {}
            truth = e.get('truth') or {}
            if (est.get('tissue') or {}).get('valid'):
                counts['tissue_valid'] += 1
            if (est.get('cut') or {}).get('valid'):
                counts['cut_valid'] += 1
            if (est.get('cut') or {}).get('front_valid'):
                counts['front_valid'] += 1
            if (est.get('scissors') or {}).get('valid'):
                counts['scissors_valid'] += 1
            tp = est.get('tension_proxy') or {}
            if tp.get('valid'):
                counts['tension_valid'] += 1
            for k in fields:
                if err.get(k) is not None:
                    fields[k].append(float(err[k]))
            # Valid-but-wrong: estimate claims valid yet truth disagrees beyond 2 mm.
            wrong['ticks_eval'] += 1
            if (est.get('cut') or {}).get('front_valid') and err.get('front_pos_err_m') is not None:
                if err['front_pos_err_m'] > WRONG_MM:
                    wrong['front_valid_wrong'] += 1
            if (est.get('scissors') or {}).get('valid') and err.get('tip_pos_err_m') is not None:
                if err['tip_pos_err_m'] > WRONG_MM:
                    wrong['tip_valid_wrong'] += 1
            sp = tp.get('strain_proxy')
            if sp is not None and truth.get('tension') is not None:
                tension_pairs.append((float(sp), float(truth['tension'])))
    stats = {}
    for k, v in fields.items():
        if v:
            stats[k] = dict(n=len(v), median=pct(v, 50), p95=pct(v, 95), max=max(v))
        else:
            stats[k] = dict(n=0)
    corr = None
    if len(tension_pairs) > 10:
        a = np.array(tension_pairs)
        if a[:, 0].std() > 1e-9 and a[:, 1].std() > 1e-9:
            corr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    valid_rates = {k: (v / counts['ticks'] if counts['ticks'] else 0)
                   for k, v in counts.items() if k != 'ticks'}
    return dict(ticks=counts['ticks'], valid_rates=valid_rates, errors=stats,
                valid_but_wrong={k: v for k, v in wrong.items() if k != 'ticks_eval'},
                tension_proxy_correlation=corr)


def plot(rows, out):
    """PIL line chart: front-pos error (median/p95/max) vs each degraded factor."""
    from PIL import Image, ImageDraw
    PW, PH, TILE_W, TILE_H = 1380, 460, 460, 460
    PAD_L, PAD_R, PAD_T, PAD_B = 70, 16, 42, 52
    img = Image.new('RGB', (PW, PH), 'white')
    dr = ImageDraw.Draw(img)
    dr.text((PW // 2 - 140, 8), 'Shadow perception degradation (nominal, full)', fill='black')
    series = [('median', (30, 130, 240)), ('p95', (240, 140, 30)), ('max', (220, 40, 40))]
    for ti, (factor, _, label) in enumerate(SWEEP):
        ox, oy = ti * TILE_W, 0
        pts = [r for r in rows if r.get('factor') == factor and
               (r['stats']['errors'].get('front_pos_err_m') or {}).get('median')]
        if not pts:
            continue
        xs = [r['value'] for r in pts]
        ys = {k: [((r['stats']['errors']['front_pos_err_m'].get(k) or
                    r['stats']['errors']['front_pos_err_m']['median']) * 1000) for r in pts]
              for k, _ in series}
        x0, x1 = min(xs), max(xs)
        y1 = max(max(v) for v in ys.values())
        y1 = max(y1, 0.5)
        def px(val, lo, hi, a, b):
            return a + (b - a) * ((val - lo) / (hi - lo) if hi > lo else 0.5)
        gx0, gx1 = ox + PAD_L, ox + TILE_W - PAD_R
        gy0, gy1 = oy + PAD_T, oy + TILE_H - PAD_B
        dr.rectangle([gx0, gy0, gx1, gy1], outline=(200, 200, 200))
        for gy in range(5):
            yy = gy1 - (gy1 - gy0) * gy / 4
            dr.line([gx0, yy, gx1, yy], fill=(230, 230, 230))
            dr.text((ox + 6, yy - 6), f'{y1 * gy / 4:.1f}', fill=(90, 90, 90))
        dr.text((gx0, oy + 24), f'front pos err (mm) vs {label}', fill='black')
        for k, color in series:
            seq = sorted(zip(xs, ys[k]))
            xy = [(px(x, x0, x1, gx0, gx1), px(y, 0, y1, gy1, gy0)) for x, y in seq]
            dr.line(xy, fill=color, width=2)
            for X, Y in xy:
                dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], fill=color)
            dr.text((gx0 + 8 + 60 * [k for k, _ in series].index(k), gy1 + 10), k, fill=color)
        dr.text((gx0, gy1 + 28), label, fill=(90, 90, 90))
    img.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-scenarios', action='store_true')
    ap.add_argument('--skip-sweep', action='store_true')
    a = ap.parse_args()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    started = datetime.now(timezone.utc).isoformat()
    rows = []

    if not a.skip_scenarios:
        for sc in SCENARIOS:
            name = f'perc_eval_{sc}'
            perc = dict(mode='shadow', markers=True, seed=1)
            rep, err = run('full', sc, perc, name)
            if rep is None:
                rows.append(dict(kind='scenario', scenario=sc, error=err))
                print(sc, 'RUN FAIL', err)
                continue
            rows.append(dict(kind='scenario', scenario=sc, name=name,
                             success=rep.get('success'),
                             state=rep.get('final_state'),
                             stats=summarize([rep])))
            print(sc, rep.get('final_state'), 'done')

    if not a.skip_sweep:
        for factor, values, key in SWEEP:
            for v in values:
                seeds = SEEDS if v != 0 else [SEEDS[0]]
                reps = []
                for seed in seeds:
                    perc = dict(mode='shadow', markers=True, seed=seed,
                                depth_noise_m=0.0,
                                camera_pose_error=dict(trans_m=0.0, rot_deg=0.0))
                    if key == 'depth_noise_m':
                        perc['depth_noise_m'] = v
                    elif key == 'camera_pose_error.trans_m':
                        perc['camera_pose_error']['trans_m'] = v
                    else:
                        perc['camera_pose_error']['rot_deg'] = v
                    name = f'perc_eval_{factor}_{str(v).replace(".", "p")}_s{seed}'
                    rep, err = run('full', 'nominal', perc, name)
                    if rep is None:
                        print(f'{factor}={v} seed{seed} FAIL', err)
                        continue
                    reps.append(rep)
                    print(f'{factor}={v} seed{seed}', rep.get('final_state'))
                if reps:
                    rows.append(dict(kind='sweep', factor=factor, key=key, value=v,
                                     runs=len(reps),
                                     success=all(r.get('success') for r in reps),
                                     stats=summarize(reps)))

    sources = ['scripts/vision/cut_perception.py', 'scripts/vision/run_perception_eval.py',
               'scripts/task/sim_world.py', 'scripts/task/run_cut_task.py',
               'config/task/cut_task.json']
    import hashlib
    hashes = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:16] for s in sources}
    REPORT.write_text(json.dumps(dict(run_id=run_id, started_at=started,
                                      finished_at=datetime.now(timezone.utc).isoformat(),
                                      source_sha256=hashes, results=rows),
                                 indent=2), encoding='utf-8')
    try:
        plot([r for r in rows if r.get('kind') == 'sweep'], FIG)
    except Exception as e:
        print('figure skipped:', e)
    print('report:', REPORT)


if __name__ == '__main__':
    main()
