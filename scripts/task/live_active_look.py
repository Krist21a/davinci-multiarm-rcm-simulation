"""Closed optical recovery before parking; bounded real ECM movement on the conventional sheet."""
import numpy as np

from cut_fsm import CutTaskFSM, S
from same_bite_world import SameBiteWorld


class ActiveLookWorld(SameBiteWorld):
    def use_recovery_optics(self):
        if self.W==1280 and self.H==960:
            return
        if self.W!=640 or self.H!=480:
            raise ValueError('unexpected native calibration')
        self.sim.setIntArray2Property(self.cam,'resolution',[1280,960])
        if list(self.sim.getVisionSensorRes(self.cam))!=[1280,960]:
            raise ValueError('HD optics refused')
        self.W,self.H,self.f=1280,960,2*self.f
        # Only intrinsics change. Optical direction and slit history are metric.
        if self._perc_estimator:
            self._perc_estimator.W,self._perc_estimator.H,self._perc_estimator.f=self.W,self.H,self.f
        self._perc_meta=None
        self._anchor_stamp=None
        self._invalidate_fix()

    def look_from(self,target):
        ok=self.command(self.view,*target,25)
        if ok:
            self._invalidate_fix()
            self.camera_moves+=1
            if not np.allclose(self.target_of(self.view),target,rtol=0,atol=1e-6):
                raise ValueError('guarded ECM pose did not land')
            self.use_recovery_optics()
        self.event_log.append(dict(event='closed_active_look',tick=self.tick,target=target,accepted=ok))
        return ok

    def advance_scissors(self,d):
        sigma=self.perc_cfg.get('depth_noise_m',0.)
        if sigma:
            budget=self.endpoint_advance_budget()
            if budget is None or not np.isfinite(budget) or budget-sigma<.0002:
                return False
            d=min(d,budget-sigma)
        return super().advance_scissors(d)

    def render(self):
        super().render()
        if self.W==1280 and self.H==960 and self.args.capture is not None:
            p=self.args.capture/f'{self.tick:03d}.npz'
            if not p.exists():
                with p.open('xb') as f:
                    np.savez_compressed(f,rgb=np.asarray(self.image),depth=np.asarray(self.depth,dtype='<f4'),
                                        cam_R=self.camera()[0],cam_t=self.camera()[1])


class ActiveLookFSM(CutTaskFSM):
    def on_clear_view(self,obs):
        if self.visible(obs) or not obs.scissors_visible or not obs.tissue_visible:
            return super().on_clear_view(obs)
        if not self.cut_started and self.w.jaws_closed() and self.return_to in (S.EXPOSE,S.GRASP):
            if self.ticks_in_state<=self.cfg['view']['occlusion_wait_ticks']:
                # At setup no cutter closure has occurred. Discard a false initial
                # slit prior, never discard the history of an actual closure.
                self.w._perc_estimator=None
                self.w._invalidate_fix()
                self.w.hold()
                return
            return self.go(S.CAMERA_ADJUST,'setup view not restored while holding',obs)
        attempt=self.attempts[S.CLEAR_VIEW]+1
        if (self.w.jaws_closed() and self.cut_started and 2<=attempt<=self.cfg['budgets']['clear_view_attempts']
                and self.attempts[S.CAMERA_ADJUST]<self.cfg['budgets']['camera_adjust_attempts']):
            self.attempts[S.CLEAR_VIEW]=attempt
            idx=self.attempts[S.CAMERA_ADJUST]
            self.attempts[S.CAMERA_ADJUST]+=1
            self.w.look_from(self.w.args.targets[idx])
            return
        return super().on_clear_view(obs)
