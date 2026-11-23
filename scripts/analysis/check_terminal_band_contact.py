"""Necessary optical cue only. No material assignment to background and no control."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_entry_skew import ROOT, cp, load_dataset, sha
from probe_bank_transects import labels

PLAN = ROOT / 'config/task/m5b_terminal_band_contact.json'


class ScanCapture(cp.CutPerception):
    def _gap_scan(self, mask, sa_px, sb_px, pz, depth, line_lat, s_near, s_far):
        out = super()._gap_scan(mask, sa_px, sb_px, pz, depth, line_lat, s_near, s_far)
        if getattr(self, '_scan_axis', None) is not sa_px:
            self._scan_axis = sa_px
            self._gap = np.zeros_like(mask)
        self._gap |= out['slit_sel'] & np.isfinite(depth) & (depth > pz + cp.GAP_DEPTH_M)
        self._lateral, self._sheet = sb_px, mask
        return out


def contacts(gap, lateral, terminal):
    h, w = gap.shape
    negative, positive, bridges = set(), set(), set()
    for y, x in np.argwhere(gap):
        sides = set()
        for v in range(max(0, y-1), min(h, y+2)):
            for u in range(max(0, x-1), min(w, x+2)):
                if not terminal[v, u]:
                    continue
                sign = np.sign(lateral[v, u]-lateral[y, x])
                if sign:
                    (positive if sign > 0 else negative).add((int(u), int(v)))
                    sides.add(sign)
        if len(sides) == 2:
            bridges.add((int(x), int(y)))
    return dict(gap_pixels=int(gap.sum()), negative_uv=sorted(negative), positive_uv=sorted(positive),
                gap_touching_both_uv=sorted(bridges), bilateral=bool(negative and positive))


def replay(source):
    path = ROOT / source['report']
    run = json.loads(path.read_bytes())
    meta, frames, files = load_dataset(path.stem)
    if (sha(path) != source['sha256'] or files != source['capture'] or run['run_id'] != files['run_id']
            or files['scene_sha256'] != sha(ROOT / run['scene']) or meta['perception_cfg'] != run['config']['perception']
            or any(meta['perception_cfg']['camera_pose_error'][k] for k in ('rot_deg', 'trans_m'))
            or any(sha(ROOT / k) != v for k, v in run['source_sha256'].items())):
        raise ValueError('invalid capture identity or unsupported perturbation')
    est = ScanCapture.from_meta(meta)
    rng = np.random.default_rng(meta['perception_cfg']['seed'])
    samples = []
    for frame, ref, trace in zip(frames, run['perception'], run['trace'], strict=True):
        rng.normal(size=6)
        depth = frame['depth'].astype(float)
        sigma = meta['perception_cfg']['depth_noise_m']
        if sigma:
            depth += rng.normal(0, sigma, depth.shape)
        est._scan_axis = None
        cut = est.estimate(frame['rgb'], depth, frame['cam_R'], frame['cam_t'],
                           'grasped_arms' in frame and bool(len(frame['grasped_arms'])))['cut']
        if (int(frame['tick']) != ref['tick'] or ref['tick'] != trace['tick'] or
                json.loads(json.dumps(cut, default=lambda a: a.tolist())) != ref['estimate']['cut']):
            raise ValueError('reference replay diverged')
        cue = None
        if cut.get('front_valid') and cut['mode'] in ('slit', 'bank_slit'):
            if est._scan_axis is None:
                raise ValueError('direct front lacks a current-frame scan')
            code = labels(frame['rgb'].astype(float), est._sheet)
            cue = contacts(est._gap, est._lateral, code == len(cp.material_coordinate.COLORS)-1)
        samples.append(dict(tick=ref['tick'], mode=cut.get('mode'), cue=cue, truth_mm=1000*trace['cut_length_m']))
    early = [s for s in samples if s['truth_mm'] < 9 and s['cue'] is not None]
    return dict(name=path.stem, capture=files, report_sha256=sha(path), samples=samples,
                direct_subtarget_frames=len(early), false_positives=[s['tick'] for s in early if s['cue']['bilateral']])


def main(out):
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(PLAN.read_bytes())
    source = ROOT / 'validation/task/m5b_v1_10_verified_summary.json'
    refs = json.loads(source.read_bytes())
    trials = [replay(s) for s in (refs['marked_truth'], *refs['trials'])]
    endpoints = []
    for t in [16, 17]:
        s = next(s for s in trials[0]['samples'] if s['tick'] == t)
        if s['truth_mm'] != 10:
            raise ValueError('not the registered withdrawn checkpoint')
        endpoints.append(dict(tick=t, bilateral=bool(s['cue'] and s['cue']['bilateral'])))
        print(t, s['cue'])
    passed = all(e['bilateral'] for e in endpoints) and not any(t['false_positives'] for t in trials)
    with out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(dict(protocol=plan, plan_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
                       labels_module_sha256=sha(ROOT / 'scripts/analysis/probe_bank_transects.py'),
                       source_sha256=sha(source), endpoint_checks=endpoints, optical_screen_passed=passed,
                       trials=trials), f, indent=2, allow_nan=False)
        f.write('\n')
    for t in trials:
        print(t['name'], t['direct_subtarget_frames'], 'early positives:', t['false_positives'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
