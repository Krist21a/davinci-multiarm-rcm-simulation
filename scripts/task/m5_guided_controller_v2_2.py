"""Precision-separated ALIGN and closed retreat before the first camera switch."""
from collections import deque
import math

import numpy as np

from cut_fsm import CutTaskFSM, S
from m5_guided_controller_v2_1 import GuidedFSMV21, GuidedWorldV21, review_action
from m5_guided_bank_corridor_v4 import measure
from run_cut_task import sha256
from simtools import matrix_to_np

ALIGN_M = .0004
NOISY_CROSS_M = .00028


class AlignmentSamples:
    def __init__(self):
        self.rows = deque(maxlen=3)
        self.epoch = None
        self.tick = None

    def clear(self):
        self.rows.clear()
        self.tick = self.epoch = None

    def add(self,epoch,tick,g):
        if g is None:
            self.clear()
            return None
        if self.epoch!=epoch or self.tick is None or tick not in (self.tick,self.tick+1):
            self.clear()
        if tick!=self.tick:
            self.rows.append(g)
        self.epoch,self.tick = epoch,tick
        if len(self.rows)<3:
            return None
        front,tip,direction,normal,jaw,hinge = np.median(np.asarray(self.rows),axis=0)
        def unit(x):
            return x/np.linalg.norm(x)
        if min(np.linalg.norm(x) for x in (direction,normal,jaw,hinge))<1e-6:
            return None
        direction,jaw,hinge = map(unit,(direction,jaw,hinge))
        lateral = np.cross(normal,direction)
        if np.linalg.norm(lateral)<1e-6:
            return None
        return front,tip,direction,unit(lateral),jaw,hinge


class GuidedWorldV22(GuidedWorldV21):
    _alignment_epoch = 0
    _samples = None
    _open_readback = None

    def call(self,name,*args):
        if name in ('setRCMTarget','setToolPose','setGrip','tissueRelease','tissueGrasp'):
            self._alignment_epoch += 1
            if self._samples is not None:
                self._samples.clear()
            if name=='setGrip':
                self._open_readback = None
        return super().call(name,*args)

    def _grip_positions(self):
        joints = self.arm_cfg[self.cutter]['grip_joints']
        return np.array([self.sim.getJointPosition(self.by[j]) for j in joints])

    def open_scissors(self):
        q = self._grip_positions()
        if self._open_readback is not None and np.allclose(q,self._open_readback,rtol=0,atol=1e-6):
            return True
        if not super().open_scissors():
            return False
        q = self._grip_positions()
        if not np.isfinite(q).all() or np.any(np.abs(q)<=1e-6):
            return False
        self._open_readback = q
        return True

    def precision_geometry(self):
        raw = self._vision_alignment_geometry()
        if not self.perc_cfg['depth_noise_m']:
            return raw
        if self._samples is None:
            self._samples = AlignmentSamples()
        vector = None
        if raw is not None:
            scis = self._last_est['scissors']
            vector = np.asarray([*raw[:3],self._fix['normal'],scis['jaw_axis'],scis['opening_axis']],dtype=float)
            if vector.shape!=(6,3) or not np.isfinite(vector).all():
                vector = None
        q = self._samples.add((self._alignment_epoch,self._fix_epoch),self.tick,vector)
        if q is None:
            return None
        front,tip,axis,lateral,jaw,hinge = q
        if lateral@self.u<0:
            lateral = -lateral
        cos_tol = math.cos(math.radians(self.cfg['cut']['vision_axis_tolerance_deg']))
        cross = float(np.linalg.norm(np.cross(front-tip,jaw)))
        return front,tip,axis,lateral,cross,bool(jaw@axis>=cos_tol and abs(hinge@lateral)>=cos_tol)

    def align_precisely(self,g):
        self.view_blocked = False
        front,tip,axis,lateral,_,_ = g
        handle = self.by[self.cutter+'_tcp']
        tcp = matrix_to_np(self.sim.getObjectMatrix(handle,-1))
        target = tcp[:,3]+(front-.001*axis)-tip
        r = self._visible_tool_candidate('align',tcp,[(d,target-(d-.001)*axis) for d in (.001,.0015,.002)],
                                          axis,lateral)
        if not self._vision_tool_motion(r,10):
            return False
        end = np.asarray(self.sim.getObjectPosition(handle,-1))
        self.advanced += float((end-tcp[:,3])@self.v)
        return True


class GuidedFSMV22(GuidedFSMV21):
    def _precise(self,g):
        cross = NOISY_CROSS_M if self.w.perc_cfg['depth_noise_m'] else ALIGN_M
        return (g is not None and g[-1] and g[-2]<=cross
                and abs(float((g[0]-g[1])@g[2])-.001)<=ALIGN_M)

    def on_align(self,obs):
        p = self._progress(obs)
        if p is None or not math.isfinite(p) or p<0:
            return self.go(S.FAILED,'scissors alignment refused: invalid visual progress',obs)
        if p>=self.cfg['cut']['length_m']:
            return self._complete(obs)
        w = self.w
        g = w.precision_geometry()
        if self._precise(g):
            w.event_log.append(dict(event='guided_v22_precision',tick=w.tick,cross_m=g[-2],
                                    gap_m=float((g[0]-g[1])@g[2])))
            return self.go(S.CUT,'precise alignment on current vision',obs)
        if self.ticks_in_state>=self.cfg['budgets']['align_ticks']:
            return self.go(S.FAILED,'precision alignment not reached',obs)
        if g is not None:
            w.align_precisely(g)
            if w.view_blocked:
                return self._blocked_candidate(obs,'alignment')

    def on_cut(self,obs):
        if not self._precise(self.w.precision_geometry()):
            return self.go(S.ALIGN,'recheck precise alignment before opening',obs)
        return CutTaskFSM.on_cut(self,obs)

    def _review(self,obs,first):
        w = self.w
        self._closed(obs)
        if not first and (not w.trail_valid or w.resume_index is None or not w.at_waypoint(w.trail[-1])
                          or not 0<w.resume_index<len(w.trail)):
            raise ValueError('no complete recorded reverse path')
        if first:
            obs = self._entry(obs)
        obs = self._camera(obs)
        camera = w.camera()
        if not first:
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
        with np.load(p,allow_pickle=False) as frame:
            r = measure(frame['rgb'],frame['measured'],far=frame['far'].item(),sigma=sigma)
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
