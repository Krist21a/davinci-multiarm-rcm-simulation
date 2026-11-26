"""Acquire depth and retained inks from the same guarded medium-view exposure."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import run_same_bite as parent
from run_cut_task import REPO, add_arguments, sha256
from run_m5_endpoint_review_v4 import ReviewWorld, ReviewFSM, SPEC as REVIEW, ACTIVE, PARENT, RESERVE

SPEC = REPO/'config/task/m5_retained_print_v7.json'
PRINT = REPO/'src/tissue/retained_print_v7.lua'
NATIVE = 'd403b81dd20b1d0cba44300c2f6a52163d63bd3c06a4f8e8a2ac51a143469ca2'


class RetainedWorld(ReviewWorld):
    def start(self):
        if self.controller_sha256 != NATIVE:
            raise ValueError('unregistered native controller')
        s = self.sim
        source = s.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n')+'\n'+PRINT.read_text(encoding='utf-8')
        s.setStringProperty(self.ctrl, 'code', source)
        if s.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n') != source:
            raise ValueError('retained print staging mismatch')
        self.controller_sha256 = hashlib.sha256(source.encode()).hexdigest()
        super().start()

    def snapshot_print(self, phase):
        s, cam = self.sim, self.cam
        p = s.visionfloatparam_perspective_angle
        angle = s.getObjectFloatParam(cam, p)
        base = [self.W, self.H]
        if base != [1280,960] or list(s.getVisionSensorRes(cam)) != base or abs(angle-math.radians(70)) > 1e-6:
            raise ValueError('unexpected native approach intrinsics')
        if phase != 'seed' and not self.jaws_closed():
            raise ValueError('coded observation requires physically closed jaws')
        pose = self.camera()
        stamp = s.getSimulationTime()
        rgb0, z0 = np.array(self.image), self.depth.copy()
        res = [2560,1920]
        w, h = res
        try:
            s.setObjectFloatParam(cam, p, math.radians(45))
            s.setIntArray2Property(cam,'resolution',res)
            if not self.call('drawRetainedPrintV7', True):
                raise ValueError('retained print rendering failed')
            s.handleVisionSensor(cam)
            rgb, size_rgb = s.getVisionSensorImg(cam)
            depth, size_z = s.getVisionSensorDepth(cam,1)
            actual = s.getObjectFloatParam(cam,p)
            if list(size_rgb) != res or list(size_z) != res or abs(actual-math.pi/4) > 1e-6:
                raise ValueError('medium exposure intrinsics mismatch')
            file = self.args.evidence_out/f'{self.tick:03d}_{phase}.npz'
            with file.open('xb') as f:
                np.savez_compressed(f,rgb=np.frombuffer(bytes(rgb),np.uint8).reshape(h,w,3)[::-1],
                                    depth=np.frombuffer(bytes(depth),'<f4').reshape(h,w)[::-1],
                                    cam_R=pose[0],cam_t=pose[1],focal=w/2/math.tan(actual/2),
                                    far=self.far,hfov_rad=actual,tcp_local=self._tcp_local)
        finally:
            s.setObjectFloatParam(cam,p,angle)
            s.setIntArray2Property(cam,'resolution',base)
            if not self.call('drawRetainedPrintV7',False):
                raise ValueError('native unprinted rendering restore refused')
            self.render()
        restored = self.camera()
        if (s.getSimulationTime() != stamp or not all(np.array_equal(a,b) for a,b in zip(pose,restored,strict=True))
                or list(s.getVisionSensorRes(cam)) != base
                or abs(s.getObjectFloatParam(cam,p)-angle) > 1e-6
                or not np.array_equal(np.asarray(self.image),rgb0) or not np.array_equal(self.depth,z0)):
            raise ValueError('native observation did not restore byte-identically')
        closures = [e['tick'] for e in self.event_log if e['event']=='closure']
        self.event_log.append(dict(event='retained_coded_exposure',tick=self.tick,phase=phase,
                                   sim_time=stamp, preceding_closure_tick=closures[-1] if closures else None,
                                   file=file.name,frame_sha256=sha256(file)))

    def record_post_bite(self, obs):
        super().record_post_bite(obs)
        self.snapshot_print('post_bite')

    def read_review(self, obs):
        self.snapshot_print('review')
        return False


class RetainedFSM(ReviewFSM):
    seeded = False

    def on_align(self, obs):
        if not self.seeded and not self.cut_started:
            self.w.snapshot_print('seed')
            self.seeded = True
        super().on_align(obs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--evidence-out',type=Path,required=True)
    args = ap.parse_args()
    plan = json.loads(SPEC.read_bytes())
    if REPO/plan['parent'] != REVIEW or plan['native_review'] != dict(size=[2560,1920],hfov_deg=45):
        raise ValueError('unregistered v7 review')
    args.reserve_m = json.loads(RESERVE.read_bytes())['advance_reserve_m']
    args.review_target = json.loads(REVIEW.read_bytes())['view']['target']
    args.case = 'nominal'
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    args.capture = None
    args.evidence_out.mkdir(exist_ok=False)
    return parent.run(args,world_class=RetainedWorld,fsm_class=RetainedFSM,active_spec=ACTIVE,
                      policy='retained_print_v7',extra_sources=(SPEC,PRINT,REVIEW,RESERVE,PARENT,
                          Path(__file__).resolve(),*(Path(__file__).resolve().with_name(n) for n in (
                              'run_m5_endpoint_review_v4.py','run_m5_sweep_reserve_v3.py',
                              'run_m5_guarded_post_bite_v2.py','m5_guarded_post_bite_v2.py')),
                          REPO/'scripts/vision/m5_local_endpoint_v2.py',
                          REPO/'scripts/vision/m5_local_endpoint_v3.py',REPO/'scripts/vision/row9_terminal.py',
                          REPO/'scripts/learn/tip_marker_gate_dev.py'))


if __name__=='__main__':
    raise SystemExit(main())
