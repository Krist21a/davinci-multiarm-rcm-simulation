"""Find the current slit by reversing one guarded, recorded closed segment."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_m5_guided_10mm_exact_v1 as exact
from m5_guided_controller_v2 import camera_action
from run_cut_task import REPO, sha256

SPEC = REPO/'config/task/m5_guided_10mm_closed_reacquire_v1.json'
EXACT = REPO/'validation/task/m5_guided_10mm_exact_v1'


def segment_length(start,goal,axis_col,limit):
    delta = goal[:,3]-start[:,3]
    distance = float(np.linalg.norm(delta))
    if not (0<distance<=limit and np.isfinite(delta).all() and float(delta@start[:,axis_col])<0):
        raise ValueError('closed slit recovery is not a bounded retreat')
    return distance


class ReacquireFSM(exact.ExactFSM):
    def _review(self,obs,first):
        w = self.w
        if not first and w.awaiting_new_slit:
            self._closed(obs)
            if (w.closures!=w.args.maximum_closures or not obs.scissors_visible or not obs.tissue_visible
                    or not w.trail_valid or w.resume_index is None or len(w.trail)<3
                    or w.resume_index!=len(w.trail)-2 or not w.at_waypoint(w.trail[-1])
                    or camera_action(w.target_of(w.view),w.args.review_target,0,0)!='reuse'):
                raise ValueError('closed slit recovery interlock')
            distance = segment_length(w.trail[-1][1],w.trail[-2][1],w.TOOL_AXIS_COL,
                                      w.args.control_plan['recovery_max_backoff_m'])
            tick = w.tick
            obs = self._path(obs,w.trail[-2:-1],'slit_reacquire',w.camera())
            ok = bool(w.tick>tick and not w.awaiting_new_slit and w.direct_progress(obs) is not None)
            w.event_log.append(dict(event='exact_closed_slit_recovery',tick=w.tick,start_tick=tick,
                                    distance_m=distance,accepted=ok))
            if not ok:
                raise ValueError('guarded reverse did not freshly reacquire the slit')
            w.trail = w.trail[:-1]
        return super()._review(obs,first)


def main():
    plan = json.loads(SPEC.read_bytes())
    dec,reg = EXACT/'decision.json',EXACT/'registration.json'
    if (sha256(dec)!=plan['exact_decision_sha256'] or sha256(reg)!=plan['exact_registration_sha256']
            or any(sha256(REPO/k)!=v for k,v in json.loads(reg.read_bytes())['sources'].items())
            or plan['recovery_max_backoff_m']!=.003):
        raise ValueError('unregistered closed slit recovery contrast')
    spec,fsm,parent = exact.SPEC,exact.ExactFSM,exact.v2.parent

    def register(out,policy,files):
        return parent.register(out,policy,(*files,Path(__file__).resolve(),spec,dec,reg))

    try:
        exact.SPEC,exact.ExactFSM = SPEC,ReacquireFSM
        exact.v2.parent = SimpleNamespace(register=register,put=parent.put)
        return exact.main()
    finally:
        exact.SPEC,exact.ExactFSM,exact.v2.parent = spec,fsm,parent


if __name__=='__main__':
    raise SystemExit(main())
