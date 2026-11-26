"""Pre-cut estimate reset and near-waypoint closed resume for the v2.4 sequence."""
import numpy as np

from cut_fsm import S
from m5_guided_controller_v2_1 import review_action
from m5_guided_controller_v2_2 import GuidedFSMV22, GuidedWorldV22
from m5_guided_bank_corridor_v4 import measure
from run_cut_task import sha256


class GuidedWorldV24(GuidedWorldV22):
    _has_cut = False

    def _reset_precut(self, reason):
        if self._has_cut or getattr(self._perc_estimator, '_slit_seen', False):
            return
        had_estimate = self._perc_estimator is not None
        self._perc_estimator = self._last_est = None
        self._anchor_stamp = self._transfer_tick = None
        self._invalidate_fix()
        if self._fix is not None:  # Pre-cut disturbance commands also invalidate the unloaded fix.
            self._fix_epoch += 1
            self._fix = None
        if had_estimate:
            self.event_log.append(dict(event='guided_v24_precut_reset', tick=self.tick, reason=reason))

    def call(self, name, *args):
        result = super().call(name, *args)
        if (name == 'tissueRelease' or name == 'tissueGrasp' and result
                or name in ('setRCMTarget', 'setToolPose', 'setGrip') and result
                and args[0] != self.index[self.cutter]):
            self._reset_precut(name)
        return result

    def close_scissors(self):
        cut = (self._last_est or {}).get('cut') or {}
        if (not self._fix_valid() or self._fix['tick'] != self.tick or not cut.get('front_valid')
                or cut.get('mode') not in ('pre_cut', 'slit', 'bank_slit')
                or not (cut.get('material_chart') or {}).get('valid')):
            return False
        ok = super().close_scissors()
        if ok:
            self._has_cut = True
        return ok


class GuidedFSMV24(GuidedFSMV22):
    def on_clear_view(self, obs):
        # Keep the parent's setup wait/action budget, but guard its estimator reset too.
        if (not self.visible(obs) and obs.scissors_visible and obs.tissue_visible
                and not self.cut_started and self.w.jaws_closed() and self.return_to in (S.EXPOSE, S.GRASP)):
            if self.ticks_in_state <= self.cfg['view']['occlusion_wait_ticks']:
                self.w._reset_precut('setup_clear_view')
                return self.w.hold()
            return self.go(S.CAMERA_ADJUST, 'setup view not restored while holding', obs)
        return super().on_clear_view(obs)

    def _resume_below(self, obs, camera, prefix, cap):
        w = self.w
        # The first near observation was recorded after the normal closed back-off.
        near = w.trail[1]
        fresh = self._path(obs, [near], 'resume', camera)
        cut = (w._last_est or {}).get('cut') or {}
        if (w.awaiting_new_slit or not self.visible(fresh) or not self.centred(fresh) or not w._fix_valid()
                or not cut.get('front_valid') or cut.get('mode') not in ('slit', 'bank_slit')):
            raise ValueError('closed resume lacks fresh direct wide alignment evidence')
        source = 'current' if w._fix['tick'] == w.tick else 'confirmed_same_bite'
        w.trail = w.trail[:2]
        w.resume_index = 1
        w.cap, w.advanced_once = cap, False
        w.event_log.append(dict(event='guided_v2_resume', tick=w.tick, visual_prefix_mm=prefix, cap_m=cap,
                                waypoint_index=1, material_source=source))
        return self.go(S.ALIGN, 'visual BELOW; closed near-cut return verified', fresh)

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
        return self._resume_below(obs,camera,n,cap)
