"""One nominal review of the optical endpoint before any further grip command."""
import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

import run_same_bite as parent
from cut_fsm import CutTaskFSM, S
from m5_guarded_post_bite_v2 import RoutinePostBiteFSM
from run_cut_task import REPO, add_arguments, sha256
from run_m5_sweep_reserve_v3 import ACTIVE, PARENT, SPEC as RESERVE, ReservingWorld

sys.path.insert(0, str(REPO/'scripts/learn'))
from tip_marker_gate_dev import patch_features
from m5_local_endpoint_v3 import measure

SPEC = REPO/'config/task/m5_endpoint_review_v4.json'


class ReviewWorld(ReservingWorld):
    review_armed = False

    def direct_progress(self, obs):
        estimate = self._last_est or {}
        scissors, cut = estimate.get('scissors') or {}, estimate.get('cut') or {}
        progress = obs.cut_progress_m
        if (progress is None or not math.isfinite(progress) or not obs.cut_site_visible
                or not obs.scissors_visible or not scissors.get('valid') or not cut.get('front_valid')
                or cut.get('mode') not in ('slit', 'bank_slit')
                or not (cut.get('material_chart') or {}).get('valid')
                or not self._fix_valid() or self._fix['tick'] != self.tick):
            return None
        return progress

    def record_post_bite(self, obs):
        super().record_post_bite(obs)
        progress = self.direct_progress(obs)
        if progress is not None and progress >= .006:
            self.review_armed = True
            self.event_log.append(dict(event='endpoint_review_armed', tick=self.tick, progress_m=progress))

    def read_review(self, obs):
        R, t = self.camera()
        im, depth = np.asarray(self.image), self.depth
        progress = self.direct_progress(obs)
        estimate = self._last_est or {}
        local = measure(im, depth, R, t, self.f, estimate, sigma=0.)
        tip = estimate.get('scissors') or {}
        marker = False
        if tip.get('valid'):
            patch = patch_features(depth, (self.W, self.H, self.f, self.far), dict(
                valid=True, uv_px=tip.get('tip_uv'), xyz_m=R.T @ (np.asarray(tip['tip'])-t)))
            marker = patch is not None and patch[-1] >= -3.
        path = self.args.evidence_out/f'review_{self.tick:03d}.npz'
        with path.open('xb') as file:
            np.savez_compressed(file, rgb=im, depth=depth.astype('<f4'), cam_R=R, cam_t=t,
                                focal=self.f, far=self.far)
        gate = bool(progress is not None and .009 <= progress <= .010 and marker
                    and local['status'] == 'ROW9_MATCH')
        self.event_log.append(dict(event='endpoint_review_verdict', tick=self.tick, progress_m=progress,
                                   marker_usable=bool(marker), local=local, optical_gate=gate,
                                   file=path.name, frame_sha256=sha256(path)))
        return gate


class ReviewFSM(RoutinePostBiteFSM):
    review_consumed = False
    verified_tick = None

    def _complete(self, obs):
        if self.w.tick == self.verified_tick:
            return super()._complete(obs)
        if self.w.review_armed and not self.review_consumed:
            return self.go(S.ALIGN, 'armed closed optical review precedes withdrawal', obs)
        return self.go(S.FAILED, 'no guarded optical endpoint review', obs)

    def supervise(self, obs):
        if not self.w.review_armed or self.review_consumed or self.state != S.ALIGN:
            return super().supervise(obs)
        self.review_consumed = True
        tick = self.w.tick
        def mechanically_blocked(sample):
            return CutTaskFSM.supervise(self, replace(sample, cut_site_visible=True,
                                                       scissors_visible=True, cut_site_offset=0.0))
        if mechanically_blocked(obs):
            return self._refuse(obs, 'interlock before ECM review')
        if (not self.w.jaws_closed() or self.attempts[S.CAMERA_ADJUST]
                >= self.cfg['budgets']['camera_adjust_attempts']):
            return self._refuse(obs, 'jaws or original ECM budget preclude review')
        self.attempts[S.CAMERA_ADJUST] += 1
        if not self.w.look_from(self.w.args.review_target):
            return self._refuse(obs, 'guarded ECM review move refused')
        fresh = self.w.observe()
        if mechanically_blocked(fresh):
            return self._refuse(fresh, 'interlock after ECM review')
        optical = self.w.read_review(fresh)
        if (self.w.tick <= tick or not self.w.jaws_closed() or not self.centred(fresh) or not optical):
            return self._refuse(fresh, 'endpoint review abstained; no further closure')
        self.verified_tick = self.w.tick
        self.w.event_log.append(dict(event='optical_withdrawal_trigger', tick=self.w.tick))
        self._complete(fresh)
        return True

    def _refuse(self, obs, reason):
        if self.state != S.FAILED:
            self.go(S.FAILED, reason, obs)
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--evidence-out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(SPEC.read_bytes())
    reserve = json.loads(RESERVE.read_bytes())
    args.reserve_m = reserve['advance_reserve_m']
    args.review_target = plan['view']['target']
    if (args.reserve_m != .00015 or REPO/plan['parent'] != RESERVE
            or plan['view'] != dict(target=[0, 4.5, .088], size=[1280, 960], hfov_deg=70)):
        raise ValueError('unregistered endpoint review')
    args.case = 'nominal'
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    args.capture = None
    args.evidence_out.mkdir(exist_ok=False)
    return parent.run(args, world_class=ReviewWorld, fsm_class=ReviewFSM, active_spec=ACTIVE,
                      policy='endpoint_review_v4', extra_sources=(
                          SPEC, RESERVE, PARENT, Path(__file__).resolve(),
                          *(Path(__file__).resolve().with_name(n) for n in (
                              'run_m5_sweep_reserve_v3.py', 'run_m5_guarded_post_bite_v2.py',
                              'm5_guarded_post_bite_v2.py')),
                          *(REPO/'scripts/vision'/n for n in ('m5_local_endpoint_v2.py',
                                                             'm5_local_endpoint_v3.py', 'row9_terminal.py')),
                          REPO/'scripts/learn/tip_marker_gate_dev.py'))


if __name__ == '__main__':
    raise SystemExit(main())
