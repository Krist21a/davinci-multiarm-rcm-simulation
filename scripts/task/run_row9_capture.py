"""Passive RGB-D capture on the unchanged full visual controller, including its native 640p frames."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import run_same_bite as parent
from live_active_look import ActiveLookFSM, ActiveLookWorld
from run_cut_task import REPO, add_arguments, sha256

PLAN = REPO / 'config/task/m5_row9_contact_dev_v1_1.json'
ACTIVE = REPO / 'config/task/m5_live_active_look_v1.json'


class PassiveFramesWorld(ActiveLookWorld):
    def _estimate(self):
        if (self.perc_cfg.get('camera_pose_error') or {}).get('mode', 'per_tick') != 'per_tick':
            raise ValueError('capture supports only per-tick calibration RNG')
        exposure = np.asarray(self.image)
        raw_depth = np.asarray(self.depth, dtype='<f4')
        R, t = self.camera()
        clone = np.random.default_rng()
        clone.bit_generator.state = copy.deepcopy(self.perc_rng.bit_generator.state)
        estimate = super()._estimate()
        with (self.args.all_capture / f'{self.tick:03d}.npz').open('xb') as file:
            np.savez_compressed(file, rgb=exposure, depth=raw_depth,
                                cam_R=R, cam_t=t, f=np.float64(self.f))
        clone.normal(size=3)
        clone.normal(size=3)
        sigma = self.perc_cfg.get('depth_noise_m', 0.)
        measured = self.depth if sigma == 0 else self.depth + clone.normal(0, sigma, self.depth.shape)
        with (self.args.all_capture / f'{self.tick:03d}_depth.npy').open('xb') as file:
            np.save(file, np.asarray(measured, dtype='<f4'))
        return estimate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--case', choices=('nominal', 'depth_noise_0p5'), required=True)
    args = ap.parse_args()
    plan = json.loads(PLAN.read_bytes())
    if args.case not in plan['cases']:
        ap.error('case is not preregistered')
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    args.capture = args.out / (args.case + '_hd')
    args.all_capture = args.out / (args.case + '_frames')
    if any(p.exists() for p in (args.capture, args.all_capture, args.out / (args.case + '.json'))):
        raise FileExistsError('new capture directory required')
    args.capture.mkdir(parents=True)
    args.all_capture.mkdir()
    status = parent.run(args, world_class=PassiveFramesWorld, fsm_class=ActiveLookFSM,
                        active_spec=ACTIVE, policy='row9_capture', extra_sources=(PLAN, Path(__file__)))
    report = json.loads((args.out / (args.case + '.json')).read_bytes())
    if 'perception' not in report:
        raise RuntimeError(f"{args.case}: acquisition failed: {report.get('error')}")
    optical = dict(run_id=report['run_id'], case=args.case, registration_sha256=report['registration_sha256'],
                   source_sha256=sha256(Path(__file__)), plan_sha256=sha256(PLAN),
                   exposure=[dict(tick=p['tick'], estimate=p['estimate']) for p in report['perception']])
    parent.put(args.out / (args.case + '_optical.json'), optical)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
