"""Integrated vision-triggered review without changing the frozen v2 path."""
import math

import numpy as np

from cut_fsm import S
from m5_guided_controller_v2 import GuidedFSMV2, GuidedWorldV2, measured_review, remaining_m
from m5_guided_ahead_window_v1 import AheadWindowPerception
from run_cut_task import sha256
import tool_pose_active_set


def review_action(verdict,n,previous,target_m,cap_m):
    if type(n) is not int or not 1<=n<=10 or (previous is not None and n<=previous):
        return 'stop',None
    if verdict=='TARGET' and previous is not None and n in (9,10) and n>=round(target_m*1000):
        return 'withdraw',None
    rem = remaining_m(verdict,n,target_m)
    return ('resume',min(rem,cap_m)) if rem is not None else ('stop',None)


class GuidedWorldV21(GuidedWorldV2):
    relocalized = False

    def _estimate(self):
        if self._perc_estimator is None:
            if self._perc_meta is None:
                self._perc_meta = self._dump_meta()
            self._perc_estimator = AheadWindowPerception.from_meta(self._perc_meta)
        return super()._estimate()

    def _tool_solve(self,arm,target_p,target_axis=None,target_yaw=None):
        state = self._tool_q(arm)
        if state is None:
            return None
        q0,ws = state
        try:
            r = tool_pose_active_set.solve(self._tool_fk(arm),q0,self._tool_limits(arm,ws),target_p,
                                           target_axis,target_yaw,self.TOOL_AXIS_COL,self.TOOL_YAW_COL)
        except RuntimeError:
            return None
        except (ValueError,np.linalg.LinAlgError):
            return None
        r['q0'] = q0
        if r.get('active_set'):
            self.event_log.append(dict(event='guided_v21_active_set',tick=self.tick,arm=arm,converged=r['converged'],
                                       q=r['q'].tolist(),target_m=np.asarray(target_p).tolist(),
                                       position_error_m=r['pos_err_m'],axis_error_rad=r['axis_err_rad'],
                                       hinge_error_rad=r['yaw_axis_err_rad'],at_limit=r['at_limit']))
        return r

    def advance_scissors(self,requested):
        if self.relocalized:
            requested = min(requested,self.args.review_advance_cap_m)
        return super().advance_scissors(requested)


class GuidedFSMV21(GuidedFSMV2):
    previous_prefix = None

    def on_camera_adjust(self,obs):
        s = S.CAMERA_ADJUST
        if (self.attempts[s]>=self.cfg['budgets']['camera_adjust_attempts']
                and not (self.visible(obs) and self.centred(obs) and self.ticks_in_state>1)):
            return self.go(S.FAILED,'camera adjustment exhausted',obs)
        return super().on_camera_adjust(obs)

    def on_relocalize(self,obs):
        result = super().on_relocalize(obs)
        if self.state==S.RESUME:
            self.w.relocalized = True
        return result

    def _aligned(self):
        g = self.w._vision_alignment_geometry()
        return (g is not None and g[-1] and max(g[-2],abs(float((g[0]-g[1])@g[2])-.001))
                <=self.cfg['cut']['align_tolerance_m'])

    def on_align(self,obs):
        w = self.w
        progress = self._progress(obs)
        if progress is None or not math.isfinite(progress) or progress<0:
            return self.go(S.FAILED,'scissors alignment refused: invalid visual progress',obs)
        if progress>=self.cfg['cut']['length_m']:
            return self._complete(obs)
        if self._aligned():
            return self.go(S.CUT,'blades aligned on current vision; open only before advance',obs)
        err = w.align_scissors(self.target.cut_start,progress)
        if w.view_blocked:
            return self._blocked_candidate(obs,'alignment')
        if self.ticks_in_state>=self.cfg['budgets']['align_ticks']:
            return self.go(S.FAILED,f'alignment not reached ({err})',obs)

    def on_cut(self,obs):
        w = self.w
        progress = self._progress(obs)
        if progress is None or not math.isfinite(progress) or progress<0:
            return self.go(S.FAILED,'scissors advance refused: invalid visual progress',obs)
        if progress>=self.cfg['cut']['length_m']:
            return self._complete(obs)
        if not self._aligned():
            return self.go(S.ALIGN,'refresh alignment before opening',obs)
        if not w.open_scissors():
            return self.go(S.FAILED,'scissors cannot open',obs)
        obs = w.observe()
        if self.supervise(obs):
            return
        progress = self._progress(obs)
        if progress is None or not math.isfinite(progress) or progress<0 or not self._aligned():
            return self.go(S.FAILED,'scissors advance refused: fresh open alignment',obs)
        if progress>=self.cfg['cut']['length_m']:
            return self._complete(obs)
        travel = w.endpoint_travel(self.cfg['cut']['length_m']-progress)
        budget = w.endpoint_advance_budget()
        cap = self.cfg['cut']['advance_m']
        if travel is None or budget is None or min(travel,cap,budget)<.0002:
            return self.go(S.FAILED,'scissors advance refused: visual endpoint bound',obs)
        if not w.advance_scissors(min(travel,cap,budget)):
            if w.view_blocked:
                return self._blocked_candidate(obs,'advance')
            return self.go(S.FAILED,'scissors advance refused (limits or collision)',obs)
        fresh = w.observe()
        if self.front_recovery:
            fresh = self._retry_advanced_view(fresh)
        if self.supervise(fresh):
            return
        progress = self._progress(fresh)
        if progress is None or not math.isfinite(progress) or progress<0:
            return self.go(S.FAILED,'scissors closure refused: invalid visual progress',fresh)
        if progress>=self.cfg['cut']['length_m']:
            return self._complete(fresh)
        if not w.closure_aligned():
            return self.go(S.FAILED,'scissors closure refused: fresh alignment out of tolerance',fresh)
        budget = w.endpoint_advance_budget()
        if budget is None or budget< -1e-12:
            return self.go(S.FAILED,'scissors closure refused: visual endpoint bound',fresh)
        if not w.close_scissors():
            return self.go(S.FAILED,'scissors cannot close',fresh)
        self.cut_started = True

    def _review(self,obs,first):
        w = self.w
        self._closed(obs)
        if not first and (not w.trail_valid or w.resume_index is None or not w.at_waypoint(w.trail[-1])
                          or not 0<w.resume_index<len(w.trail)):
            raise ValueError('no complete recorded reverse path')
        obs = self._camera(obs)
        camera = w.camera()
        if first:
            obs = self._entry(obs)
        else:
            obs = self._path(obs,reversed(w.trail[:-1]),'clear',camera)
        if not w.at_waypoint(w.trail[0]) or not all(np.array_equal(a,b) for a,b in zip(camera,w.camera(),strict=True)):
            raise ValueError('closed review did not reach its matched pose')
        self._closed(obs)
        tick = w.tick
        w.read_review(obs)
        e = w.event_log[-1]
        if e.get('event')!='guided_prefix_review' or e['tick']!=tick or w.tick!=tick or not e['physically_closed']:
            raise ValueError('missing current measured exposure')
        p = w.args.evidence_out/e['file']
        sigma = w.perc_cfg['depth_noise_m']
        if p.name!=e['file'] or sha256(p)!=e['frame_sha256'] or sigma!=e['noise_sigma_m'] or not w.jaws_closed():
            raise ValueError('measured exposure identity changed')
        r = measured_review(p,sigma)
        v,n = r['verdict'],r['prefix_mm']
        action,cap = review_action(v,n,self.previous_prefix,self.cfg['cut']['length_m'],w.args.review_advance_cap_m)
        w.event_log.append(dict(event='guided_v21_optical',tick=tick,verdict=v,prefix_mm=n,previous_prefix_mm=self.previous_prefix,
                                action=action,sigma=sigma,file=p.name,frame_sha256=e['frame_sha256']))
        if action=='stop':
            raise ValueError(f'optical verdict {v}({n}) is not a fresh, growing permit')
        self.previous_prefix = n
        if action=='withdraw':
            w.withdrawal_tick = tick
            return self._complete(obs)
        fresh = self._path(obs,w.trail[1:w.resume_index+1],'resume',camera)
        if (w.awaiting_new_slit or not self.visible(fresh) or not self.centred(fresh)
                or w.direct_progress(fresh) is None):
            raise ValueError('closed resume lacks fresh direct wide alignment evidence')
        w.trail = w.trail[:w.resume_index+1]
        w.cap,w.advanced_once = cap,False
        w.event_log.append(dict(event='guided_v2_resume',tick=w.tick,visual_prefix_mm=n,cap_m=cap))
        return self.go(S.ALIGN,'visual BELOW; closed near-cut return verified',fresh)
