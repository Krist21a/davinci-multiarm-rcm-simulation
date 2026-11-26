"""Opt-in routine closed-jaw post-bite motion, separate from exceptional recovery."""
from dataclasses import replace

import numpy as np

from cut_fsm import S
from live_active_look import ActiveLookFSM, ActiveLookWorld
from simtools import matrix_to_np


class RoutinePostBiteFSM(ActiveLookFSM):
    def supervise(self, obs):
        if self.state != S.CUT or not getattr(self.w, 'post_bite_pending', False):
            return super().supervise(obs)
        self.w.post_bite_pending = False
        # The cut site is expected to be covered until the cutter backs away. Only
        # mask the view checks here; always run the existing mechanical supervisor.
        def mechanical(sample):
            return super(RoutinePostBiteFSM, self).supervise(replace(
                sample, cut_site_visible=True, scissors_visible=True, cut_site_offset=0.0))

        if mechanical(obs):
            return True
        if (self.w.align_mode != 'tool_pose' or not self.w.jaws_closed()
                or self.cfg['cut']['clear_back_m'] != .003):
            self.go(S.FAILED, 'routine post-bite look requires registered closed-jaw back-off', obs)
            return True
        tick = self.w.tick
        s = self.w.sim
        handle = self.w.by[self.w.cutter + '_tcp']
        before = matrix_to_np(s.getObjectMatrix(handle, -1))
        ok = self.w.clear_view(.003, prefer_park=False)
        after = matrix_to_np(s.getObjectMatrix(handle, -1))
        target = before[:, 3] - .003 * before[:, self.w.TOOL_AXIS_COL]
        err = float(np.linalg.norm(after[:, 3] - target))
        accepted = bool(ok and np.isfinite(err) and err <= .0005)
        self.w.event_log.append(dict(event='routine_post_bite_backoff', tick=tick, accepted=accepted,
                                     commanded=bool(ok), residual_m=err if np.isfinite(err) else None))
        if not accepted:
            self.go(S.FAILED, 'guarded routine back-off refused', obs)
            return True
        fresh = self.w.observe()
        if mechanical(fresh):
            return True
        if self.w.tick <= tick or not self.w.jaws_closed():
            self.go(S.FAILED, 'routine post-bite observation stale or jaws open', fresh)
            return True
        record = getattr(self.w, 'record_post_bite', None)
        if record is not None:
            record(fresh)
        if super().supervise(fresh):
            return True
        progress = self._progress(fresh)
        if progress is not None and progress >= self.cfg['cut']['length_m']:
            self._complete(fresh)
        else:
            self.go(S.ALIGN, 'fresh post-bite look; re-align before advancing', fresh)
        return True


class NativeHDWorld(ActiveLookWorld):
    post_bite_pending = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_recovery_optics()

    def close_scissors(self):
        if self.post_bite_pending:
            return False
        accepted = super().close_scissors()
        self.post_bite_pending = bool(accepted)
        return accepted
