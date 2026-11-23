"""Two accepted bites, re-observation and guarded withdrawal; not a length endpoint."""
import math
from dataclasses import asdict

import numpy as np

from cut_fsm import CutTaskFSM, S
from sim_world import LEGACY_TISSUE, SimWorld, matrix_to_np


class TwoBiteWorld(SimWorld):
    # Locked collector: the historical protocol pins the frozen legacy controller.
    TISSUE_SOURCE = LEGACY_TISSUE

    def __init__(self, *args, protocol, blocked=False, **kwargs):
        kwargs.setdefault('tissue_source', self.TISSUE_SOURCE)
        self.protocol, self.blocked = protocol, blocked
        self.bites, self.shutter = 0, False
        self.samples, self.park = [], None
        super().__init__(*args, **kwargs)
        if not (self.vision and self.truth.strict and self.local_material_chart and self.align_mode == 'tool_pose'):
            raise ValueError('two-bite task requires the marked visual tool-pose path')

    def _tcp_pose(self):
        return matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))

    def home(self):
        if not super().home():
            return False
        self.ready_q = self._tool_q(self.cutter)[0].copy()
        self.ready_tcp = self._tcp_pose().copy()
        return bool(np.isfinite(self.ready_q).all() and np.isfinite(self.ready_tcp).all())

    def observe(self, **kwargs):
        obs = super().observe(**kwargs)
        self.samples.append(dict(tick=self.tick, shutter=self.shutter, **asdict(obs)))
        return obs

    def render(self):
        super().render()
        if self.shutter:
            self.image.paste((96, 96, 96), (0, 0, self.W, self.H))
            self.depth = np.full_like(self.depth, .020)

    def advance_scissors(self, d):
        if self.bites >= 2 or not super().advance_scissors(d):
            return False
        if self.blocked and not self.shutter and d > 0:
            self.shutter = True
            self.event_log.append(dict(event='shutter_after_advance', tick=self.tick))
        return True

    def direct_fix(self, modes):
        est = self._last_est or {}
        cut = est.get('cut') or {}
        return (self._fix_valid() and self._fix['tick'] == self.tick and not self.awaiting_new_slit
                and cut.get('mode') in modes and cut.get('front_valid') is True
                and (cut.get('material_chart') or {}).get('valid') is True
                and (est.get('scissors') or {}).get('valid') is True)

    def close_scissors(self):
        modes = ('slit', 'bank_slit') if self.bites else ('pre_cut', 'slit', 'bank_slit')
        if self.bites >= 2 or not self.direct_fix(modes) or not super().close_scissors():
            return False
        self.bites += 1
        return True

    def clear_view(self, d, prefer_park=False):
        return super().clear_view(d, prefer_park=prefer_park and self.bites < 2)

    def retract_scissors(self):
        # Recovery may back off to see the second cut, but cannot claim or perform
        # the final park before reacquisition (including the takeover recovery).
        return self.bites < 2 and super().retract_scissors()

    def finish_park(self):
        if self.bites != 2 or not self.direct_fix(('slit', 'bank_slit')) or not self.jaws_closed():
            return False
        before = self._tcp_pose()
        accepted = self.tool_command(self.cutter, self.ready_q, 20)
        after = self._tcp_pose()
        delta = after[:, :3].T @ self.ready_tcp[:, :3]
        pos = float(np.linalg.norm(after[:, 3]-self.ready_tcp[:, 3]))
        angle = math.degrees(math.acos(np.clip((np.trace(delta)-1)/2, -1, 1)))
        self.park = dict(tick=self.tick, accepted=accepted, ready_tcp=self.ready_tcp.tolist(),
                         before_tcp=before.tolist(), after_tcp=after.tolist(), ready_q=self.ready_q.tolist(),
                         position_error_m=pos, rotation_error_deg=angle, closed_after=self.jaws_closed())
        return (accepted and self.park['closed_after'] and pos <= self.protocol['park_position_tolerance_m']
                and angle <= self.protocol['park_rotation_tolerance_deg'])


class TwoBiteFSM(CutTaskFSM):
    def tension_ok(self, obs):
        band = self.cfg['tension']
        low = band['low_strain_after_cut_start'] if self.cut_started else band['low_strain']
        return low <= obs.tension <= band['high_strain']

    def _complete(self, obs):
        self.go(S.FAILED, 'visual length bound reached before two-bite withdrawal', obs)

    def supervise(self, obs):
        if super().supervise(obs):
            return True
        if self.w.bites < 2:
            return False
        if not self.w.direct_fix(('slit', 'bank_slit')):
            if self.state not in (S.OCCLUDED, S.CLEAR_VIEW, S.CAMERA_ADJUST, S.RESUME):
                self.go(S.CLEAR_VIEW, 'two bites; require fresh slit for withdrawal', obs)
                return True
            return False
        if not (self.visible(obs) and self.centred(obs) and self._held(obs) and self.tension_ok(obs)):
            self.go(S.FAILED, 'two-bite withdrawal interlock refused', obs)
        elif not self.w.finish_park():
            self.go(S.FAILED, 'two-bite park command or pose readback refused', obs)
        else:
            final = self.w.observe(inject_events=False)
            if (final.faults or final.torn or not self._held(final) or not self.tension_ok(final)
                    or not math.isfinite(final.max_strain) or not 0 <= final.cut_length_m <=
                    self.cfg['cut']['length_m'] + self.cfg['cut']['max_overcut_m'] + 1e-12):
                self.go(S.FAILED, 'two-bite post-withdrawal safety check failed', final)
            else:
                self.go(S.DONE, 'two bites reacquired; closed scissors at verified ready pose', final)
        return True

    def _held(self, obs):
        return all(obs.grasped.get(a, 0) >= self.cfg['grasp']['min_nodes'] for a in (self.primary, self.assist))
