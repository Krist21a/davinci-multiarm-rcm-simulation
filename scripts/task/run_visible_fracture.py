"""Collect optical 8/9 development exposures on a render-only fracture-visible synthetic sheet."""
import argparse
import json
from pathlib import Path

import run_same_bite as parent
from live_active_look import ActiveLookFSM
from run_cut_task import REPO, add_arguments, sha256
from run_row9_capture import PassiveFramesWorld

SPEC = REPO / 'config/task/m5_visible_fracture_dev_v1.json'
ACTIVE = REPO / 'config/task/m5_live_active_look_v1.json'
V2 = REPO / 'config/task/m5_visible_fracture_dev_v2.json'
V3 = REPO / 'config/task/m5_visible_fracture_dev_v3.json'
V4 = REPO / 'config/task/m5_visible_fracture_dev_v4.json'
V5 = REPO / 'config/task/m5_visible_fracture_dev_v5.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--case', choices=('nominal', 'depth_noise_0p5'), required=True)
    ap.add_argument('--protocol', choices=('v1', 'v2', 'v3', 'v4', 'v5'), default='v1')
    args = ap.parse_args()
    protocol = {'v1': SPEC, 'v2': V2, 'v3': V3, 'v4': V4, 'v5': V5}[args.protocol]
    spec = json.loads(protocol.read_bytes())
    if sha256(REPO / spec['parent']) != spec['parent_sha256'] or (
            sha256(REPO / spec['scene']) != spec['scene_sha256']):
        raise ValueError('registered scene or parent policy changed')
    if args.case not in spec['cases']:
        ap.error('unregistered case')
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    args.capture = args.out / (args.case + '_hd')
    args.all_capture = args.out / (args.case + '_frames')
    if any(p.exists() for p in (args.capture, args.all_capture,
                                 args.out / (args.case + '.json'))):
        raise FileExistsError('new optical capture required')
    args.capture.mkdir(parents=True)
    args.all_capture.mkdir()
    status = parent.run(args, world_class=PassiveFramesWorld, fsm_class=ActiveLookFSM,
                        active_spec=ACTIVE, policy=('bridge_print_development' if args.protocol in ('v4','v5')
                                                    else 'visible_fracture_development'),
                        extra_sources=(protocol, Path(__file__), Path(__file__).with_name('run_row9_capture.py')),
                        sheet_cfg=(spec['sheet'] if 'sheet' in spec else
                                   {k: spec[k] for k in ('visual_gap_m', 'visual_gap_style',
                                                        'visual_gap_start_row') if k in spec}))
    report = json.loads((args.out / (args.case + '.json')).read_bytes())
    if 'perception' not in report:
        raise RuntimeError(f"{args.case}: optical acquisition failed: {report.get('error')}")
    optical = dict(run_id=report['run_id'], case=args.case, registration_sha256=report['registration_sha256'],
                   source_sha256=sha256(Path(__file__)), plan_sha256=sha256(protocol),
                   exposures=[dict(tick=p['tick'], estimate=p['estimate']) for p in report['perception']])
    parent.put(args.out / (args.case + '_optical.json'), optical)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
