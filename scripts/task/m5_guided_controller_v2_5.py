"""Single-use stationary ALIGN permit and guarded closed wrist-view rollback."""
from dataclasses import replace

from cut_fsm import CutTaskFSM, S
from m5_guided_controller_v2_4 import GuidedFSMV24, GuidedWorldV24


class GuidedWorldV25(GuidedWorldV24):
    pending_view_check = None

    def align_precisely(self, g):
        origin = self.waypoint()
        trail_size = len(self.trail) if self.trail is not None else None
        advanced = self.advanced
        ok = super().align_precisely(g)
        if ok:
            self.pending_view_check = (self.tick + 1, self._alignment_epoch, origin, trail_size, advanced)
        return ok

    def rollback_alignment(self, origin, trail_size, advanced):
        self.replaying = True
        try:
            ok = (self.jaws_closed() and self.tool_command(self.cutter, origin[0], 10)
                  and self.jaws_closed() and self.at_waypoint(origin))
        finally:
            self.replaying = False
        if ok:
            self.advanced = advanced
            if trail_size is not None:
                del self.trail[trail_size:]
        self.event_log.append(dict(event='guided_v25_alignment_rollback', tick=self.tick, accepted=bool(ok)))
        return bool(ok)


class GuidedFSMV25(GuidedFSMV24):
    align_permit = None

    def on_align(self, obs):
        self.align_permit = None
        if not self.w.jaws_closed():
            return self.go(S.FAILED, 'precision alignment requires closed jaws', obs)
        result = super().on_align(obs)
        if self.state == S.CUT:
            self.align_permit = (self.w.tick + 1, self.w._alignment_epoch)
        return result

    def on_cut(self, obs):
        permit, self.align_permit = self.align_permit, None
        if permit != (self.w.tick, self.w._alignment_epoch) or not self.w.jaws_closed():
            return self.go(S.ALIGN, 'stationary alignment permit no longer current', obs)
        return CutTaskFSM.on_cut(self, obs)

    def supervise(self, obs):
        w = self.w
        check, w.pending_view_check = w.pending_view_check, None
        if (self.state == S.ALIGN and check is not None and check[:2] == (w.tick, w._alignment_epoch)
                and w.occluder in (None, -1) and w.jaws_closed() and not obs.scissors_visible
                and ((w._last_est or {}).get('scissors') or {}).get('valid') is False):
            # Mask only view checks while authorizing a closed, guarded reverse move.
            if CutTaskFSM.supervise(self, replace(obs, scissors_visible=True, cut_site_visible=True,
                                                 cut_site_offset=0.)):
                return True
            if not w.rollback_alignment(*check[2:]):
                self.go(S.FAILED, 'guarded alignment rollback refused', obs)
            elif self.ticks_in_state >= self.cfg['budgets']['align_ticks']:
                self.go(S.FAILED, 'precision alignment exhausted after rollback', obs)
            return True
        return super().supervise(obs)
