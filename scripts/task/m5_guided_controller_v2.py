"""Guided workflow driven by fresh visual reviews, not bite ordinals."""
from dataclasses import replace
import math

import numpy as np

from capture_m5_guided_reverse_v1 import pose_match
from cut_fsm import CutTaskFSM, S
from m5_guarded_post_bite_v2 import RoutinePostBiteFSM
from m5_guided_bank_corridor_v3 import measure
from run_m5_guided_benchmark_v1 import GuidedWorld
from run_cut_task import SimWorld, sha256
from simtools import matrix_to_np


def remaining_m(verdict, prefix, target):
    if verdict != 'BELOW' or type(prefix) is not int or not 1 <= prefix <= 8:
        return None
    rem = target - prefix / 1000
    return rem if math.isfinite(rem) and rem >= .0002 else None


def measured_review(path, sigma):
    with np.load(path, allow_pickle=False) as f:
        return measure(f['rgb'], f['measured'], far=f['far'].item(), sigma=sigma)


def camera_action(actual, target, attempts, limit):
    q = np.asarray(actual, dtype=float)
    if q.shape != (3,) or not np.isfinite(q).all():
        return 'invalid'
    if np.allclose(q, target, atol=1e-6, rtol=0):
        return 'reuse'
    return 'move' if attempts < limit else 'exhausted'


class GuidedWorldV2(GuidedWorld):
    trail = None
    trail_valid = True
    replaying = False
    resume_index = None
    cap = None
    advanced_once = False
    withdrawal_tick = None
    withdrawal_attempted = False

    def waypoint(self):
        state = self._tool_q(self.cutter)
        tcp = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter+'_tcp'], -1))
        if state is None or state[0].shape != (6,) or not np.isfinite(state[0]).all() or not np.isfinite(tcp).all():
            raise ValueError('invalid cutter waypoint')
        return state[0].copy(), tcp

    def at_waypoint(self, point):
        q,tcp = self.waypoint()
        return np.allclose(q,point[0],atol=1e-6,rtol=0) and pose_match(tcp,point[1])

    def tool_command(self, arm, q, n=15):
        ok = super().tool_command(arm,q,n)
        if ok and arm==self.cutter and self.trail is not None and not self.replaying:
            point = self.waypoint()
            if not np.allclose(point[0],self.trail[-1][0],atol=1e-6,rtol=0):
                if len(self.trail)>=self.args.max_waypoints:
                    self.trail_valid = False
                else:
                    self.trail.append(point)
        return ok

    def advance_scissors(self, requested):
        if self.trail is None:
            return super().advance_scissors(requested)
        if (self.cap is None or self.advanced_once or not self.trail_valid
                or not self.at_waypoint(self.trail[-1])):
            self.view_blocked = False
            return False
        self.resume_index = len(self.trail)-1
        limited = min(requested,self.cap)
        ok = super().advance_scissors(limited)
        self.advanced_once = bool(ok)
        self.event_log.append(dict(event='guided_visual_cap',tick=self.tick,cap_m=self.cap,
                                   limited_m=limited,accepted=bool(ok)))
        return ok

    def close_scissors(self):
        if self.trail is not None and (not self.advanced_once or not self.trail_valid
                                       or not self.at_waypoint(self.trail[-1])):
            return False
        ok = super().close_scissors()
        if ok and self.trail is not None:
            self.cap = None
            self.advanced_once = False
        return ok

    def retract_scissors(self):
        if self.withdrawal_attempted or self.withdrawal_tick!=self.tick or not self.jaws_closed():
            return False
        self.withdrawal_tick = None
        self.withdrawal_attempted = True
        before = self.at_ready_pose()
        self.replaying = True
        try:
            ok = SimWorld.retract_scissors(self)
        finally:
            self.replaying = False
        closed,ready = self.jaws_closed(),self.at_ready_pose()
        self.event_log.append(dict(event='guided_v2_withdrawal',tick=self.tick,accepted=bool(ok),
                                   ready_before=before,ready_after=ready,closed_after=closed))
        return bool(ok and closed and ready)


class GuidedFSMV2(RoutinePostBiteFSM):
    def _complete(self,obs):
        if self.w.withdrawal_tick==self.w.tick and self.w.jaws_closed():
            return CutTaskFSM._complete(self,obs)
        if self.w.trail is None and self.w.review_armed and self.w.jaws_closed():
            return self.go(S.ALIGN,'closed optical review precedes withdrawal',obs)
        return self.go(S.FAILED,'no current guided withdrawal permit',obs)

    def supervise(self,obs):
        w = self.w
        first = w.trail is None and w.review_armed and self.state==S.ALIGN
        again = w.trail is not None and w.post_bite_pending and self.state==S.CUT
        if not (first or again):
            return super().supervise(obs)
        w.review_armed = w.post_bite_pending = False
        try:
            self._review(obs,first)
        except (ValueError,KeyError,OSError) as err:
            self.go(S.FAILED,'guided closed review rejected: '+str(err),obs)
        return True

    def _closed(self,obs):
        if not self.w.jaws_closed() or CutTaskFSM.supervise(self,replace(obs,cut_site_visible=True,
                                                                       scissors_visible=True,cut_site_offset=0.)):
            raise ValueError('closed mechanical interlock')

    def _fresh(self,tick):
        obs = self.w.observe()
        if self.w.tick<=tick:
            raise ValueError('stale observation')
        self._closed(obs)
        return obs

    def _camera(self,obs):
        w = self.w
        s = S.CAMERA_ADJUST
        tick = w.tick
        target = w.args.review_target
        action = camera_action(w.target_of(w.view),target,self.attempts[s],
                               self.cfg['budgets']['camera_adjust_attempts'])
        w.event_log.append(dict(event='guided_v2_camera',tick=w.tick,action=action,attempts=self.attempts[s]))
        if action not in ('move','reuse'):
            raise ValueError('ECM review '+action)
        if action=='move':
            self.attempts[s] += 1
            if not w.look_from(target):
                raise ValueError('guarded ECM review refused')
        if camera_action(w.target_of(w.view),target,0,0)!='reuse':
            raise ValueError('ECM review readback missed')
        return self._fresh(tick)

    def _entry(self,obs):
        w = self.w
        if w.direct_progress(obs) is None:
            raise ValueError('initial entry clearance needs fresh wide evidence')
        est = w._last_est
        entry,direction,tip = (np.asarray(p,dtype=float) for p in (
            est['cut'].get('material_start'),est['cut'].get('direction'),est['scissors'].get('tip')))
        if (any(p.shape!=(3,) or not np.isfinite(p).all() for p in (entry,direction,tip))
                or not math.isclose(float(np.linalg.norm(direction)),1.,abs_tol=1e-6)):
            raise ValueError('unbounded visual entry geometry')
        near = w.waypoint()
        axis = near[1][:,w.TOOL_AXIS_COL]
        projection = float(axis@direction)
        along = float((tip-entry)@direction)
        retreat = (along+.0025)/projection if projection>=.8 else math.nan
        if not 0<retreat<=.008:
            raise ValueError('entry clearance travel outside bound')
        tick = w.tick
        ok = w.clear_view(retreat,prefer_park=False)
        anchor = w.waypoint()
        delta = anchor[1][:,3]-near[1][:,3]
        err = float(np.linalg.norm(delta+retreat*axis))
        accepted = bool(ok and err<=.0005 and along+float(delta@direction)<=-.002 and w.jaws_closed())
        w.event_log.append(dict(event='guided_v2_entry',tick=tick,accepted=accepted,retreat_m=retreat,residual_m=err))
        if not accepted:
            raise ValueError('guarded entry clearance refused')
        w.trail,w.resume_index = [anchor,near],1
        return self._fresh(tick)

    def _path(self,obs,points,phase,camera):
        w = self.w
        w.replaying = True
        try:
            for point in points:
                self._closed(obs)
                tick = w.tick
                ok = bool(w.tool_command(w.cutter,point[0],15) and w.jaws_closed() and w.at_waypoint(point))
                w.event_log.append(dict(event='guided_v2_segment',tick=tick,phase=phase,accepted=ok))
                if not ok:
                    raise ValueError('guarded '+phase+' segment refused')
                obs = self._fresh(tick)
                if not all(np.array_equal(a,b) for a,b in zip(camera,w.camera(),strict=True)):
                    raise ValueError('ECM changed during '+phase)
            return obs
        finally:
            w.replaying = False

    def _review(self,obs,first):
        w = self.w
        self._closed(obs)
        if not first and w.awaiting_new_slit:
            raise ValueError('post-closure slit was not freshly reacquired')
        obs = self._camera(obs)
        camera = w.camera()
        if first:
            obs = self._entry(obs)
        else:
            if (not w.trail_valid or w.resume_index is None or not w.at_waypoint(w.trail[-1])
                    or not 0<w.resume_index<len(w.trail)):
                raise ValueError('no complete recorded reverse path')
            obs = self._path(obs,reversed(w.trail[:-1]),'clear',camera)
        if not w.at_waypoint(w.trail[0]) or not all(np.array_equal(a,b) for a,b in zip(camera,w.camera(),strict=True)):
            raise ValueError('closed review did not reach its matched pose')
        self._closed(obs)
        w.read_review(obs)
        e = w.event_log[-1]
        if e.get('event')!='guided_prefix_review' or e['tick']!=w.tick or not e['physically_closed']:
            raise ValueError('missing current measured exposure')
        p = w.args.evidence_out/e['file']
        sigma = w.perc_cfg['depth_noise_m']
        if p.name!=e['file'] or sha256(p)!=e['frame_sha256'] or sigma!=e['noise_sigma_m']:
            raise ValueError('measured exposure identity changed')
        r = measured_review(p,sigma)
        verdict,n = r['verdict'],r['prefix_mm']
        w.event_log.append(dict(event='guided_v2_optical',tick=w.tick,verdict=verdict,prefix_mm=n,
                                sigma=sigma,file=p.name,frame_sha256=e['frame_sha256']))
        if verdict=='TARGET' and type(n) is int and n in (9,10):
            w.withdrawal_tick = w.tick
            return self._complete(obs)
        cap = remaining_m(verdict,n,self.cfg['cut']['length_m'])
        if cap is None:
            raise ValueError('optical verdict '+verdict+' is not a permit')
        next_obs = self._path(obs,w.trail[1:w.resume_index+1],'resume',camera)
        if not self.visible(next_obs) or not self.centred(next_obs) or w.direct_progress(next_obs) is None:
            raise ValueError('closed resume lacks fresh direct wide alignment evidence')
        w.trail = w.trail[:w.resume_index+1]
        w.cap = cap
        w.advanced_once = False
        w.event_log.append(dict(event='guided_v2_resume',tick=w.tick,visual_prefix_mm=n,cap_m=cap))
        return self.go(S.ALIGN,'visual BELOW; closed near-cut return verified',next_obs)
