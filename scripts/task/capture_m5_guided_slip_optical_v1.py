"""Read-only endpoint optics after a closed reverse of the matched slip trajectory."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_m5_guided_slip_terminal_trace_v1 as trace
from m5_guided_controller_v2 import camera_action
from run_m5_guided_10mm_closed_reacquire_v1 import segment_length
from run_cut_task import REPO, S, sha256

SPEC = REPO/'config/task/m5_guided_slip_optical_capture_v1.json'


class CaptureFSM(trace.prior.base.GuidedFSMV2):
    def _pair(self,obs,phase,camera):
        w = self.w
        self._closed(obs)
        stamp = w.sim.getSimulationTime()
        rgb,z = np.asarray(w.image),w.depth
        angle = w.sim.getObjectFloatParam(w.cam,w.sim.visionfloatparam_perspective_angle)
        if (rgb.shape!=(960,1280,3) or z.shape!=(960,1280) or abs(angle-math.radians(70))>1e-6
                or not all(np.array_equal(a,b) for a,b in zip(camera,w.camera(),strict=True))):
            raise ValueError('working view does not match the held camera')
        wide = w.args.evidence_out/f'{w.tick:03d}_wide.npz'
        with wide.open('xb') as f:
            np.savez_compressed(f,rgb=rgb,depth=z,cam_R=camera[0],cam_t=camera[1],hfov_rad=angle,
                                focal=w.f,far=w.far)
        w.read_review(obs)
        cert = w.event_log[-1]
        if (cert.get('event')!='guided_prefix_review' or cert['tick']!=w.tick or not cert['physically_closed']
                or w.sim.getSimulationTime()!=stamp or not w.jaws_closed()):
            raise ValueError('optical pair not captured in the same closed state')
        with w.truth.purpose('scoring'):
            edges = w.truth._read('constraints',lambda:w.call('tissueConstraints'))
        file = w.args.out/(phase+'_structure.npz')
        with file.open('xb') as f:
            np.savez_compressed(f,edges=np.asarray(edges))
        w.event_log.append(dict(event='slip_optical_pair',phase=phase,tick=w.tick,sim_time=stamp,
                                wide=wide.name,wide_sha256=sha256(wide),narrow=cert['file'],
                                narrow_sha256=cert['frame_sha256'],structure=file.name,structure_sha256=sha256(file)))

    def _review(self,obs,first):
        if first:
            return super()._review(obs,first)
        w = self.w
        self._closed(obs)
        plan = w.args.slip_optics
        if (sum(e['event']=='closure' for e in w.event_log)!=plan['closures'] or not w.trail_valid
                or not w.awaiting_new_slit or not obs.scissors_visible or not obs.tissue_visible
                or not 3<=len(w.trail)<=w.args.max_waypoints or w.resume_index!=len(w.trail)-2
                or not w.at_waypoint(w.trail[-1]) or w.at_ready_pose()
                or camera_action(w.target_of(w.view),w.args.review_target,0,0)!='reuse'):
            raise ValueError('slip acquisition lacks the matched closed trail')
        dist = segment_length(w.trail[-1][1],w.trail[-2][1],w.TOOL_AXIS_COL,plan['max_first_backoff_m'])
        camera = w.camera()
        w.event_log.append(dict(event='slip_optical_checkpoint',tick=w.tick,waypoints=len(w.trail),
                                first_reverse_m=dist,camera_attempts=self.attempts[S.CAMERA_ADJUST]))
        self._pair(obs,'closed',camera)
        obs = self._path(obs,reversed(w.trail[:-1]),'slip_optical_reverse',camera)
        if not w.at_waypoint(w.trail[0]) or w.at_ready_pose():
            raise ValueError('not at the recorded entrance anchor')
        self._pair(obs,'entrance',camera)
        return self.go(S.FAILED,'slip optical capture complete; held closed without a permit',obs)


def main():
    plan = json.loads(SPEC.read_bytes())
    if (any(sha256(REPO/k)!=v for k,v in plan['pins'].items()) or plan['closures']!=3
            or plan['max_first_backoff_m']!=.003 or plan['phases']!=['closed','entrance']):
        raise ValueError('unregistered slip optical acquisition')
    base = trace.prior.base
    old_run,old_fsm,parent = base.run,base.GuidedFSMV2,base.parent

    def register(out,policy,files):
        return parent.register(out,plan['id'],(*files,Path(__file__).resolve(),SPEC,*(REPO/k for k in plan['pins'])))

    def run(a):
        a.slip_optics = plan
        return old_run(a)

    try:
        base.run,base.GuidedFSMV2 = run,CaptureFSM
        base.parent = SimpleNamespace(register=register,put=parent.put)
        return trace.main()
    finally:
        base.run,base.GuidedFSMV2,base.parent = old_run,old_fsm,parent


if __name__=='__main__':
    raise SystemExit(main())
