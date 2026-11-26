"""Acquisition only at three poses along the exact run's recorded closed reverse."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_m5_guided_10mm_exact_v1 as exact
import run_m5_guided_10mm_closed_reacquire_v1 as prior
from m5_guided_controller_v2 import camera_action
from run_cut_task import REPO, S, sha256

SPEC = REPO/'config/task/m5_guided_10mm_optical_capture_v1.json'
REF = REPO/'validation/task/m5_guided_10mm_closed_reacquire_v1'


class CaptureFSM(exact.ExactFSM):
    def _complete(self,obs):
        return self.go(S.FAILED,'acquisition never authorizes terminal withdrawal',obs)

    def _pair(self,obs,phase):
        w = self.w
        self._closed(obs)
        s = w.sim
        cam = w.cam
        stamp = s.getSimulationTime()
        rgb = np.asarray(w.image)
        z = w.depth
        R,t = w.camera()
        angle = s.getObjectFloatParam(cam,s.visionfloatparam_perspective_angle)
        if rgb.shape!=(960,1280,3) or z.shape!=(960,1280) or abs(angle-math.radians(70))>1e-6:
            raise ValueError('changed working optics')
        wide = w.args.evidence_out/f'{w.tick:03d}_wide.npz'
        with wide.open('xb') as f:
            np.savez_compressed(f,rgb=rgb,depth=z,cam_R=R,cam_t=t,focal=w.f,far=w.far,hfov_rad=angle)
        est = w._last_est or {}
        cut = est.get('cut') or {}
        chart = cut.get('material_chart') or {}
        wide_state = dict(direct=w.direct_progress(obs),mode=cut.get('mode'),
                          front_valid=bool(cut.get('front_valid')),chart_valid=bool(chart.get('valid')),
                          cut_visible=obs.cut_site_visible,scissors_visible=obs.scissors_visible,
                          awaiting_new_slit=bool(w.awaiting_new_slit))
        w.read_review(obs)
        e = w.event_log[-1]
        if (e.get('event')!='guided_prefix_review' or e['tick']!=w.tick or not e['physically_closed']
                or s.getSimulationTime()!=stamp or not w.jaws_closed()):
            raise ValueError('pair not taken in one held state')
        with w.truth.purpose('scoring'):
            edges = w.truth._read('constraints',lambda:w.call('tissueConstraints'))
            nodes = w.truth.nodes()
        score = w.args.out/(phase+'_structure.npz')
        with score.open('xb') as f:
            np.savez_compressed(f,edges=np.asarray(edges),nodes=np.asarray(nodes))
        w.event_log.append(dict(event='ten_mm_optical_pair',tick=w.tick,phase=phase,sim_time=stamp,
                                wide=wide.name,wide_sha256=sha256(wide),narrow=e['file'],
                                narrow_sha256=e['frame_sha256'],structure=score.name,
                                structure_sha256=sha256(score),wide_state=wide_state))

    def _review(self,obs,first):
        if first:
            return super()._review(obs,first)
        w = self.w
        self._closed(obs)
        if (w.closures!=3 or not w.trail_valid or not 3<=len(w.trail)<=w.args.max_waypoints
                or w.resume_index!=len(w.trail)-2 or not w.at_waypoint(w.trail[-1])
                or camera_action(w.target_of(w.view),w.args.review_target,0,0)!='reuse'):
            raise ValueError('acquisition lacks the exact recorded reverse start')
        prior.segment_length(w.trail[-1][1],w.trail[-2][1],w.TOOL_AXIS_COL,
                             w.args.control_plan['recovery_max_backoff_m'])
        camera = w.camera()
        self._pair(obs,'closed')
        obs = self._path(obs,w.trail[-2:-1],'optical_reverse_first',camera)
        self._pair(obs,'backoff')
        obs = self._path(obs,reversed(w.trail[:-2]),'optical_reverse_rest',camera)
        if not w.at_waypoint(w.trail[0]):
            raise ValueError('optical entrance anchor not reached')
        self._pair(obs,'entrance')
        return self.go(S.FAILED,'optical capture complete; held closed',obs)


def main():
    plan = json.loads(SPEC.read_bytes())
    dec,reg = REF/'recovery_gate.json',REF/'registration.json'
    if (sha256(dec)!=plan['reverse_gate_sha256'] or sha256(reg)!=plan['reverse_registration_sha256']
            or any(sha256(REPO/k)!=v for k,v in json.loads(reg.read_bytes())['sources'].items())
            or plan['recovery_max_backoff_m']!=.003):
        raise ValueError('unregistered 10 mm optical capture sources')
    spec,fsm,parent = exact.SPEC,exact.ExactFSM,exact.v2.parent

    def register(out,policy,files):
        return parent.register(out,policy,(*files,Path(__file__).resolve(),spec,dec,reg,
               Path(prior.__file__).resolve()))

    try:
        exact.SPEC,exact.ExactFSM = SPEC,CaptureFSM
        exact.v2.parent = SimpleNamespace(register=register,put=parent.put)
        return exact.main()
    finally:
        exact.SPEC,exact.ExactFSM,exact.v2.parent = spec,fsm,parent


if __name__=='__main__':
    raise SystemExit(main())
