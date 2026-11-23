"""Post-hoc v1.8 prediction/next-observation pairs; never used for control or acceptance."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/task'))
from score_ecm_view_study import cp, read_run  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_stencil(estimate, frame, focal):
    cut, tissue = estimate['cut'], estimate['tissue']
    direction = np.asarray(cut['direction'])
    lateral = np.cross(tissue['normal'], direction)
    lateral /= np.linalg.norm(lateral)
    points = np.array([cut['front'] + a * direction + b * lateral
                       for a in np.linspace(0, cp.AHEAD_WINDOW_M + .0005, 5)
                       for b in np.linspace(-cp.GAP_BAND_M, cp.GAP_BAND_M, 5)])
    cam = (points - frame['cam_t']) @ frame['cam_R']
    depth = frame['depth']
    h, w = depth.shape
    if not np.isfinite(cam).all() or np.any(cam[:, 2] <= 0):
        raise ValueError('invalid projected stencil')
    uv = np.rint([w / 2, h / 2] - focal * cam[:, :2] / cam[:, 2:3]).astype(int)
    u, v = uv.T
    if np.any((u < 0) | (u >= w) | (v < 0) | (v >= h)):
        raise ValueError('stencil outside recorded camera image')
    raw_depth = depth[v, u]
    sheet = cp._sheet_mask(frame['rgb'].astype(float))[v, u]
    foreground = np.isfinite(raw_depth) & (raw_depth > 0) & (raw_depth < cam[:, 2] - cp.COVER_DEPTH_M)
    return dict(sheet_sites=int(sheet.sum()), foreground_sites=int(foreground.sum()),
                front_sheet_sites=int(sheet[:5].sum()), distinct_pixels=len(np.unique(uv, axis=0)))


def audit(label, old_label):
    run, source = read_run('m5b_v1_8_marked_' + label)
    baseline, baseline_source = read_run('m5b_v1_7_marked_' + label)
    name = 'm5b_front_diag_v1_6_' + old_label
    capture, capture_source = read_run(name)
    folder = ROOT / 'validation/perception/frames' / name
    meta = json.loads((folder / 'meta.json').read_bytes())
    manifest = json.loads((folder / 'capture.json').read_bytes())
    if (meta['run_id'] != capture['run_id'] or manifest['run_id'] != capture['run_id']
            or meta['scene_sha256'] != run['scene_sha256']
            or capture['initial_view_readback'] != run['initial_view_readback']
            or any(run['config']['perception']['camera_pose_error'][k] != 0 for k in ('trans_m', 'rot_deg'))):
        raise ValueError('camera/scene/capture mismatch')
    prefix = 0
    for curr, old, curr_trace, old_trace in zip(run['perception'], capture['perception'],
                                                run['trace'], capture['trace']):
        if curr != old or curr_trace != old_trace:
            break
        prefix += 1
    es = {x['tick']: x['estimate'] for x in run['perception']}
    pairs = []
    for event in run['event_log']:
        if event['event'] != 'predicted_view' or event['selected_m'] is None:
            continue
        tick = event['tick']
        after = es[tick + 1]
        cut = after['cut']
        direct = bool(cut.get('front_valid') and cut.get('mode') in ('pre_cut', 'slit', 'bank_slit')
                      and cut.get('progress_m') is not None and after['scissors'].get('valid'))
        row = dict(tick=tick, phase=event['phase'], chosen=event['candidates'][-1], direct_next_frame=direct,
                   next_reason=cut.get('reason') or cut.get('mode'))
        if tick + 1 <= prefix:
            path = folder / f'tick_{tick+1:03d}.npz'
            with np.load(path, allow_pickle=False) as frame:
                if int(frame['tick']) != tick + 1:
                    raise ValueError('incorrect raw frame')
                row['raw_stencil'] = dict(observed_stencil(es[tick], frame, meta['f']),
                                          frame_sha256=digest(path))
        pairs.append(row)
    return dict(run=source, reference=baseline_source, raw_capture=capture_source,
                raw_prefix_ticks=prefix, meta_sha256=digest(folder / 'meta.json'),
                manifest_sha256=digest(folder / 'capture.json'), same_trace_as_v17=run['trace'] == baseline['trace'],
                selected=len(pairs), next_frame_direct=sum(p['direct_next_frame'] for p in pairs),
                phases=dict(Counter(p['phase'] for p in pairs)), pairs=pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope='Post-hoc development diagnostic. A capsule-clear ray is not proof of sheet visibility. '
                        'Raw samples project the pre-motion stencil into the next image; not all v1.8 ticks '
                        'have matching archived images. Foreground depth is unperturbed and not tool-specific.',
                  script_sha256=digest(Path(__file__)),
                  trials=[audit('V_nominal', 'nominal'), audit('V_noise_depth_0p5', 'noise')])
    text = json.dumps(result, indent=2, allow_nan=False)
    with args.out.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text + '\n')
