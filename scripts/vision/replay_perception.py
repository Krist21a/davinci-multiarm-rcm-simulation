"""M5a offline perception evaluation on recorded shadow-mode frames.

Replays CutPerception over validation/perception/frames/<dataset>/tick_*.npz (written by
SimWorld._dump_frame) and scores it against the pre-registered protocol
config/perception/m5a_gates.json. The estimator only sees rgb, depth and the (possibly
perturbed) camera pose, exactly as in SimWorld._perception_tick; the truth fields of the
dump are used for scoring only.

Degradations are re-applied offline with the same formulas as SimWorld._perception_tick
(seeded per dataset/seed), so a noise sweep needs no simulator.

Usage: replay_perception.py [--out validation/perception/m5a_eval.json] [--datasets nominal ...]
       [--degradations none ...] [--label baseline]
"""
import argparse
import glob
import hashlib
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'vision'))
sys.path.insert(0, str(ROOT / 'scripts' / 'task'))
import cut_perception as cp  # noqa: E402
from visibility import depth_visible  # noqa: E402

FRAMES = ROOT / 'validation' / 'perception' / 'frames'
GATES = ROOT / 'config' / 'perception' / 'm5a_gates.json'


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def load_dataset(name):
    d = FRAMES / name
    meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
    files = sorted(glob.glob(str(d / 'tick_*.npz')))
    h = hashlib.sha256()
    frames = []
    for fn in files:
        h.update(Path(fn).read_bytes())
        z = np.load(fn)
        frames.append({k: z[k] for k in z.files})
    return meta, frames, h.hexdigest()


def make_estimator(meta):
    if hasattr(cp.CutPerception, 'from_meta'):
        return cp.CutPerception.from_meta(meta)
    a = meta['estimator_args']
    return cp.CutPerception(a['W'], a['H'], a['f'], a['far'], np.asarray(a['rcm_world'], float),
                            tcp_len=a['tcp_len'])


def truth_of(fr, meta):
    NU, NV, Hs = meta['NU'], meta['NV'], meta['spacing_m']
    W, H, f = meta['W'], meta['H'], meta['f']
    R, t = fr['cam_R'], fr['cam_t']
    tcp = fr['tcp_M']
    out = dict(tip=tcp[:3, 3], jaw_axis=unit(tcp[:3, meta['tool_axis_col']]), sheet='nodes' in fr)

    def visible(p):
        pc = R.T @ (np.asarray(p) - t)
        if pc[2] <= 0:
            return False
        uv = (W / 2 - f * pc[0] / pc[2], H / 2 - f * pc[1] / pc[2])
        if not (2 <= uv[0] < W - 2 and 2 <= uv[1] < H - 2):
            return False
        return depth_visible(fr['depth'].astype(float), uv, pc[2])

    out['tip_visible'] = visible(out['tip'])
    if out['sheet']:
        n = fr['nodes']
        m = 0.5 * (n[:, NU // 2 - 1] + n[:, NU // 2])
        j0 = min(int(round(float(fr['cut_length']) / Hs)), NV - 4)
        out.update(front=np.asarray(fr['cut_front'], float), front_visible=visible(fr['cut_front']),
                   dir_live=unit(m[j0 + 3] - m[j0]), dir_place=unit(fr['sheet_v']),
                   cut_length=float(fr['cut_length']))
    return out


def degrade(fr, deg, rng):
    R_true, t_true = fr['cam_R'], fr['cam_t']
    R_est = cp.rotvec(unit(rng.normal(size=3)) * math.radians(deg['rot_deg'])) @ R_true
    t_est = t_true + unit(rng.normal(size=3)) * deg['trans_m']
    depth = fr['depth'].astype(float)
    if deg['depth_noise_m'] > 0:
        depth = depth + rng.normal(0.0, deg['depth_noise_m'], depth.shape)
    return depth, R_est, t_est


def replay(meta, frames, deg, seed):
    """One sequential pass (the estimator is stateful) -> per-tick records."""
    est_obj = make_estimator(meta)
    rng = np.random.default_rng(seed)
    rows = []
    for fr in frames:
        depth, R_est, t_est = degrade(fr, deg, rng)
        post_grasp = 'grasped_arms' in fr and len(fr['grasped_arms']) > 0
        est = est_obj.estimate(fr['rgb'].astype(float), depth, R_est, t_est, post_grasp=post_grasp)
        tr = truth_of(fr, meta)
        cut, sc = est.get('cut') or {}, est.get('scissors') or {}
        r = dict(tick=int(fr['tick']), sheet=tr['sheet'], tip_visible=tr['tip_visible'],
                 tip_valid=bool(sc.get('valid')), cut_valid=bool(cut.get('valid')),
                 front_valid=bool(cut.get('front_valid')),
                 cut_reason=cut.get('reason'), tip_reason=sc.get('reason'))
        if sc.get('tip') is not None:
            r['tip_err_m'] = float(np.linalg.norm(np.asarray(sc['tip']) - tr['tip']))
        if sc.get('jaw_axis') is not None:
            r['jaw_axis_err_rad'] = float(math.acos(np.clip(unit(sc['jaw_axis']) @ tr['jaw_axis'], -1, 1)))
        if tr['sheet']:
            r['front_visible'] = tr['front_visible']
            r['cut_length_m'] = tr['cut_length']
            if cut.get('front') is not None:
                r['front_err_m'] = float(np.linalg.norm(np.asarray(cut['front']) - tr['front']))
            if cut.get('direction') is not None:
                d = unit(cut['direction'])
                r['dir_live_err_rad'] = float(math.acos(np.clip(d @ tr['dir_live'], -1, 1)))
                r['dir_place_err_rad'] = float(math.acos(np.clip(d @ tr['dir_place'], -1, 1)))
            if cut.get('front') is not None and sc.get('tip') is not None:
                rel_e = np.asarray(cut['front']) - np.asarray(sc['tip'])
                r['relative_err_m'] = float(np.linalg.norm(rel_e - (tr['front'] - tr['tip'])))
        rows.append(r)
    return rows


def _stats(v):
    v = sorted(v)
    if not v:
        return dict(n=0)
    q = lambda p: float(v[min(len(v) - 1, int(math.ceil(p / 100 * len(v))) - 1)])  # noqa: E731
    return dict(n=len(v), median=float(np.median(v)), p95=q(95), max=float(v[-1]))


def metrics(rows, wrong):
    def rate(num, den):
        return (num / den) if den else None

    sheet = [r for r in rows if r['sheet']]
    tipv = [r for r in rows if r['tip_valid']]
    frv = [r for r in sheet if r['front_valid']]
    both = [r for r in sheet if r['front_valid'] and r['tip_valid'] and 'relative_err_m' in r]
    tip_vis = [r for r in sheet if r['tip_visible']]
    fr_vis = [r for r in sheet if r['front_visible']]
    occl_f = [r for r in sheet if not r['front_visible']]
    occl_t = [r for r in sheet if not r['tip_visible']]
    m = dict(
        ticks=len(rows), sheet_ticks=len(sheet),
        tip_err=_stats([r['tip_err_m'] for r in tipv]),
        tip_err_median_m=_stats([r['tip_err_m'] for r in tipv]).get('median'),
        tip_vbw_rate=rate(sum(r['tip_err_m'] > wrong for r in tipv), len(tipv)),
        tip_availability=rate(sum(r['tip_valid'] for r in tip_vis), len(tip_vis)),
        jaw_axis_err=_stats([r['jaw_axis_err_rad'] for r in tipv if 'jaw_axis_err_rad' in r]),
        front_err=_stats([r['front_err_m'] for r in frv]),
        front_vbw_rate=rate(sum(r['front_err_m'] > wrong for r in frv), len(frv)),
        front_availability=rate(sum(r['front_valid'] for r in fr_vis), len(fr_vis)),
        direction_live_err=_stats([r['dir_live_err_rad'] for r in sheet if r['cut_valid'] and 'dir_live_err_rad' in r]),
        direction_live_err_median_rad=_stats([r['dir_live_err_rad'] for r in sheet
                                              if r['cut_valid'] and 'dir_live_err_rad' in r]).get('median'),
        direction_place_err=_stats([r['dir_place_err_rad'] for r in sheet if r['cut_valid'] and 'dir_place_err_rad' in r]),
        relative_err=_stats([r['relative_err_m'] for r in both]),
        relative_vbw_rate=rate(sum(r['relative_err_m'] > wrong for r in both), len(both)),
        pre_sheet_false_positives=sum(1 for r in rows if not r['sheet'] and (r['cut_valid'] or r['front_valid'])),
        occluded_front_ticks=len(occl_f), occluded_tip_ticks=len(occl_t),
        front_vbw_on_occluded=sum(1 for r in occl_f if r['front_valid'] and r.get('front_err_m', 0) > wrong),
        tip_vbw_on_occluded=sum(1 for r in occl_t if r['tip_valid'] and r.get('tip_err_m', 0) > wrong),
        front_valid_on_occluded=sum(1 for r in occl_f if r['front_valid']),
    )
    return m


def check(gate, results):
    if gate['dataset'] == 'all':
        vals = [results[ds]['none']['metrics'].get(gate['metric']) for ds in results if 'none' in results[ds]]
        val = None if not vals or any(v is None for v in vals) else (sum(vals) if gate['op'] == '==' else max(vals))
    else:
        val = (results.get(gate['dataset'], {}).get(gate['degradation'], {}).get('metrics') or {}).get(gate['metric'])
    if val is None:
        return None, 'FAILED (no evidence)'
    ok = dict(zip(('<', '<=', '>', '>=', '=='), (val < gate['value'], val <= gate['value'], val > gate['value'],
                                                  val >= gate['value'], val == gate['value'])))[gate['op']]
    return val, 'PASSED' if ok else 'FAILED'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'validation' / 'perception' / 'm5a_eval.json')
    ap.add_argument('--datasets', nargs='*')
    ap.add_argument('--degradations', nargs='*')
    ap.add_argument('--label', default='')
    ap.add_argument('--keep-rows', action='store_true', help='store per-tick records in the report')
    a = ap.parse_args()
    protocol = json.loads(GATES.read_text(encoding='utf-8'))
    wrong = protocol['wrong_threshold_m']
    datasets = a.datasets or protocol['datasets']
    degs = a.degradations or list(protocol['degradations'])
    results, sources = {}, {}
    for ds in datasets:
        if not (FRAMES / ds / 'meta.json').exists():
            print(ds, 'missing frames')
            continue
        meta, frames, sha = load_dataset(ds)
        sources[ds] = dict(frames=len(frames), sha256=sha, run_name=meta.get('run_name'),
                           scene_sha256=meta.get('scene_sha256'))
        results[ds] = {}
        for dname in degs:
            if dname != 'none' and ds != 'nominal':
                continue                          # sweeps run on nominal only (protocol)
            deg = protocol['degradations'][dname]
            rows = []
            for seed in deg['seeds']:
                rows += replay(meta, frames, deg, seed)
            entry = dict(metrics=metrics(rows, wrong), seeds=deg['seeds'])
            if a.keep_rows:
                entry['rows'] = rows
            results[ds][dname] = entry
            m = entry['metrics']
            print(f"{ds:20s} {dname:22s} tip med {m['tip_err_median_m']} vbw {m['tip_vbw_rate']} | "
                  f"front med {m['front_err'].get('median')} vbw {m['front_vbw_rate']} avail {m['front_availability']} | "
                  f"dir {m['direction_live_err_median_rad']} | rel vbw {m['relative_vbw_rate']}")
    gates = []
    for g in protocol['gates']:
        val, status = check(g, results)
        gates.append(dict(g, measured=val, status=status))
        print(f"{g['id']:32s} {status:22s} measured={val} {g['op']} {g['value']}")
    src = ['scripts/vision/cut_perception.py', 'scripts/vision/replay_perception.py', 'config/perception/m5a_gates.json']
    report = dict(run_id=uuid.uuid4().hex, label=a.label, created_at=datetime.now(timezone.utc).isoformat(),
                  protocol=protocol, protocol_sha256=hashlib.sha256(GATES.read_bytes()).hexdigest(),
                  source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:16] for s in src},
                  datasets=sources, results=results, gates=gates,
                  all_gates_passed=all(g['status'] == 'PASSED' for g in gates))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print('report:', a.out, 'all gates passed:', report['all_gates_passed'])


if __name__ == '__main__':
    main()
