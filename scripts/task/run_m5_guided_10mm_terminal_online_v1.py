"""Exact terminal-only 10 mm control: no wide slit can be used to resume cutting."""
import json
from pathlib import Path
from types import SimpleNamespace

import capture_m5_guided_10mm_optical_v1 as optical
import run_m5_guided_10mm_exact_v1 as exact
from m5_guided_controller_v2 import camera_action, measured_review
from run_cut_task import REPO, sha256

SPEC = REPO/'config/task/m5_guided_10mm_terminal_online_v1.json'
CAPTURE = REPO/'validation/task/m5_guided_10mm_optical_capture_v1'


def terminal_permit(verdict,prefix):
    return verdict=='TARGET' and type(prefix) is int and prefix==10


class TerminalFSM(optical.CaptureFSM):
    def _review(self,obs,first):
        if first:
            return super()._review(obs,first)
        w = self.w
        self._closed(obs)
        if (w.closures!=3 or not w.trail_valid or not 3<=len(w.trail)<=w.args.max_waypoints
                or w.resume_index!=len(w.trail)-2 or not w.at_waypoint(w.trail[-1])
                or camera_action(w.target_of(w.view),w.args.review_target,0,0)!='reuse'):
            raise ValueError('terminal-only recorded reverse interlock')
        optical.prior.segment_length(w.trail[-1][1],w.trail[-2][1],w.TOOL_AXIS_COL,
                                     w.args.control_plan['recovery_max_backoff_m'])
        camera = w.camera()
        self._pair(obs,'closed')
        obs = self._path(obs,w.trail[-2:-1],'optical_reverse_first',camera)
        self._pair(obs,'backoff')
        obs = self._path(obs,reversed(w.trail[:-2]),'optical_reverse_rest',camera)
        if not w.at_waypoint(w.trail[0]):
            raise ValueError('terminal entry anchor was not reached')
        self._pair(obs,'entrance')
        self._closed(obs)
        cert,pair = w.event_log[-2:]
        if (pair.get('event')!='ten_mm_optical_pair' or pair.get('phase')!='entrance'
                or pair.get('tick')!=w.tick or cert.get('event')!='guided_prefix_review'
                or cert.get('tick')!=w.tick or not cert.get('physically_closed') or not w.jaws_closed()
                or w.sim.getSimulationTime()!=pair['sim_time']
                or camera_action(w.target_of(w.view),w.args.review_target,0,0)!='reuse'):
            raise ValueError('terminal-only review is not current and physically closed')
        file = w.args.evidence_out/cert['file']
        sigma = w.perc_cfg['depth_noise_m']
        if (file.name!=cert['file'] or cert['file']!=pair['narrow'] or sigma!=cert['noise_sigma_m']
                or cert['frame_sha256']!=pair['narrow_sha256'] or sha256(file)!=cert['frame_sha256']):
            raise ValueError('terminal-only frame identity changed')
        r = measured_review(file,sigma)
        v,n = r['verdict'],r['prefix_mm']
        w.event_log.append(dict(event='guided_v2_optical',tick=w.tick,verdict=v,prefix_mm=n,
                                sigma=sigma,file=file.name,frame_sha256=cert['frame_sha256']))
        if not terminal_permit(v,n):
            raise ValueError('terminal-only optical verdict refused: '+v)
        w.withdrawal_tick = w.tick
        return exact_parent_complete(self,obs)


exact_parent_complete = exact.ExactFSM._complete


def main():
    plan = json.loads(SPEC.read_bytes())
    dec,reg = CAPTURE/'decision.json',CAPTURE/'registration.json'
    if (sha256(dec)!=plan['optics_decision_sha256'] or sha256(reg)!=plan['optics_registration_sha256']
            or not json.loads(dec.read_bytes())['entrance_optical_10_mm_pass']
            or any(sha256(REPO/k)!=v for k,v in json.loads(reg.read_bytes())['sources'].items())):
        raise ValueError('unregistered or unqualified 10 mm optics')
    spec,fsm,parent = optical.SPEC,optical.CaptureFSM,exact.v2.parent

    def register(out,policy,files):
        return parent.register(out,policy,(*files,Path(__file__).resolve(),spec,dec,reg,
                                          Path(optical.__file__).resolve()))

    try:
        optical.SPEC,optical.CaptureFSM = SPEC,TerminalFSM
        exact.v2.parent = SimpleNamespace(register=register,put=parent.put)
        return optical.main()
    finally:
        optical.SPEC,optical.CaptureFSM,exact.v2.parent = spec,fsm,parent


if __name__=='__main__':
    raise SystemExit(main())
