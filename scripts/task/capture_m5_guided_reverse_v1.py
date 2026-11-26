"""Record and reverse the guarded third-bite path for optical acquisition only."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import uuid

import numpy as np

import run_same_bite as parent
from capture_m5_guided_terminal_controls_v1 import CaptureFSM, CaptureWorld, SPEC as PRIOR
from run_m5_guided_benchmark_v1 import ACTIVE, CERT, PRINT, SHEET
from run_cut_task import REPO, S, add_arguments, connect, sha256, stop_simulation
from cut_fsm import CutTaskFSM
from simtools import matrix_to_np

SPEC = REPO/'config/task/m5_guided_reverse_capture_v1.json'


def pose_match(a,b):
    if (a.shape!=(3,4) or b.shape!=(3,4) or not np.isfinite([a,b]).all()
            or np.linalg.norm(a[:,3]-b[:,3])>.0005):
        return False
    angle = math.acos(float(np.clip((np.trace(a[:,:3].T@b[:,:3])-1)/2,-1.,1.)))
    return angle<=math.radians(1)


class ReverseWorld(CaptureWorld):
    recorded = ()

    def waypoint(self):
        state = self._tool_q(self.cutter)
        tcp = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter+'_tcp'],-1))
        if state is None or state[0].shape!=(6,) or not np.isfinite(state[0]).all() or not np.isfinite(tcp).all():
            raise ValueError('nonfinite cutter waypoint')
        return state[0].copy(),tcp

    def note_waypoint(self):
        q,tcp = self.waypoint()
        if not self.recorded or not np.allclose(q,self.recorded[-1][0],atol=1e-6,rtol=0):
            self.recorded += ((q,tcp),)
            self.event_log.append(dict(event='reverse_waypoint_recorded',tick=self.tick,q=q.tolist(),tcp=tcp.tolist()))

    def tool_command(self, arm, q, n=15):
        ok = super().tool_command(arm,q,n)
        if ok and arm==self.cutter and self.diagnostic_armed and self.bites==2:
            self.note_waypoint()
        return ok


class ReverseFSM(CaptureFSM):
    def _entry_clearance_capture(self, obs):
        result = super()._entry_clearance_capture(obs)
        if self.state==S.ALIGN and self.w.diagnostic_armed and self.w.bites==2:
            self.w.note_waypoint()
        return result

    def _mechanical(self, obs):
        return CutTaskFSM.supervise(self,replace(obs,cut_site_visible=True,
                                                 scissors_visible=True,cut_site_offset=0.0))

    def supervise(self, obs):
        w = self.w
        if w.bites!=3 or not w.post_bite_pending or self.state!=S.CUT:
            return super().supervise(obs)
        w.post_bite_pending = False
        self._reverse(obs)
        return True

    def _reverse(self, obs):
        w = self.w
        if self._mechanical(obs) or not w.jaws_closed() or not 2<=len(w.recorded)<=5:
            return self.go(S.FAILED,'closed reverse mechanical or path interlock',obs)
        actual = np.asarray(w.target_of(w.view),dtype=float)
        if actual.shape!=(3,) or not np.allclose(actual,w.args.review_target,atol=1e-6,rtol=0):
            return self.go(S.FAILED,'closed reverse requires unchanged review ECM',obs)
        q,tcp = w.waypoint()
        expected,target = w.recorded[-1]
        if not np.allclose(q,expected,atol=1e-6,rtol=0) or not pose_match(tcp,target):
            return self.go(S.FAILED,'closed reverse start does not match forward waypoint',obs)
        with w.truth.purpose('scoring'):
            before_edges = w.truth._read('constraints',lambda:w.call('tissueConstraints'))
        w.read_review(obs)
        before = w.event_log[-1]
        for step,(command,target) in enumerate(reversed(w.recorded[:-1]),1):
            tick = w.tick
            if not w.jaws_closed():
                return self.go(S.FAILED,'jaws open before reverse segment',obs)
            commanded = w.tool_command(w.cutter,command,15)
            q,tcp = w.waypoint()
            ok = bool(commanded and w.jaws_closed() and pose_match(tcp,target)
                      and np.allclose(q,command,atol=1e-6,rtol=0))
            w.event_log.append(dict(event='closed_reverse_segment',tick=tick,step=step,
                                    commanded=bool(commanded),accepted=ok,q=q.tolist(),
                                    residual_m=float(np.linalg.norm(tcp[:,3]-target[:,3]))))
            if not ok:
                return self.go(S.FAILED,'guarded reverse segment refused or missed',obs)
            obs = w.observe()
            current = np.asarray(w.target_of(w.view),dtype=float)
            if (w.tick<=tick or self._mechanical(obs) or not w.jaws_closed() or current.shape!=(3,)
                    or not np.allclose(current,actual,atol=1e-6,rtol=0)):
                return self.go(S.FAILED,'closed reverse segment lost safety interlock',obs)
        w.read_review(obs)
        last = w.event_log[-1]
        with w.truth.purpose('scoring'):
            after_edges = w.truth._read('constraints',lambda:w.call('tissueConstraints'))
        structure = w.args.out/(w.args.case+'_structure.npz')
        with structure.open('xb') as f:
            np.savez_compressed(f,before_edges=np.asarray(before_edges),after_edges=np.asarray(after_edges))
        w.event_log.append(dict(event='diagnostic_final_capture',tick=w.tick,pre_file=before['file'],
                                pre_sha256=before['frame_sha256'],file=last['file'],sha256=last['frame_sha256'],
                                structure_file=structure.name,structure_sha256=sha256(structure)))
        return self.go(S.FAILED,'closed reverse capture complete; held closed',obs)


def run(a):
    plan,prior = (json.loads(p.read_bytes()) for p in (SPEC,PRIOR))
    if (any(sha256(REPO/k)!=sha for k,sha in (plan['pins']|prior['pins']).items())
            or any(sha256(a.clearance_reference/k)!=sha for k,sha in prior['reference_frames'].items())
            or any(prior['cases'][k]!=v for k,v in plan['cases'].items())):
        raise ValueError('reverse path source pin changed')
    task = Path(__file__).resolve().parent
    files = (SPEC,PRIOR,Path(__file__).resolve(),ACTIVE,PRINT,CERT,
             *(task/k for k in ('capture_m5_guided_terminal_controls_v1.py','run_m5_guided_benchmark_v1.py',
                'run_m5_retained_print_v7.py','run_m5_endpoint_review_v4.py','run_m5_sweep_reserve_v3.py',
                'seam_topology.py','m5_guarded_post_bite_v2.py','run_m5_guarded_post_bite_v2.py','live_active_look.py')),
             REPO/'scripts/learn/capture_m5_temporal_coded.py',REPO/'scripts/vision/m5_guided_prefix_v1.py')
    base = parent.register(a.out,plan['id'],files)
    path = a.out/(a.case+'.json')
    if path.exists():
        raise FileExistsError(path)
    a.scene,a.scenario,a.run_id = REPO/base['scene'],'nominal',uuid.uuid4().hex
    a.initial_view,a.initial_view_insertion_m = base['initial_view'][:2],base['initial_view'][2]
    a.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    a.review_target = json.loads((REPO/'config/task/m5_guided_benchmark_v1.json').read_bytes())['review']['target']
    a.cap_m,a.reserve_m,a.capture = plan['cases'][a.case],.00015,None
    a.evidence_out.mkdir(exist_ok=False)
    cfg = json.loads((REPO/base['base']).read_bytes())
    cfg['sheet'] = SHEET.copy()
    cfg['cut']['control_source'] = 'vision'
    cfg['perception'].update(mode='shadow',markers=True,depth_noise_m=0.,seed=0,material_pattern=True,
                             local_material_chart=True,front_recovery=True,predictive_view=True,
                             exclude_tcp_marker=True,recovery_view_after_cut=True,ray_marker_fit=True)
    r = dict(case=a.case,run_id=a.run_id,cap_m=a.cap_m,config=cfg,
             registration_sha256=sha256(a.out/'registration.json'),observations=[],error=None,cleanup_error=None)
    c=s=w=None
    try:
        c,s = connect(a)
        w = ReverseWorld(c,s,a,cfg,mode='full')
        if not w.canonical_seam or not w.truth.strict or not w.vision:
            raise ValueError('wrong reverse guided world')
        fsm = ReverseFSM(w,cfg)
        observe = w.observe
        def sample(**kwargs):
            obs = observe(**kwargs)
            r['observations'].append(dict(tick=w.tick,**asdict(obs)))
            return obs
        w.observe = sample
        fsm.run()
        r.update(fsm.report())
    except Exception as err:
        r['error'] = f'{type(err).__name__}: {err}'
    finally:
        try:
            if w is not None:
                r.update(events=w.event_log,actual_closed=w.jaws_closed(),actual_ready=w.at_ready_pose(),
                         truth_purposes=w.truth.reads_by_purpose,controller_sha256=w.controller_sha256)
        except Exception as err:
            r['error'] = r['error'] or f'{type(err).__name__}: {err}'
        if s is not None:
            try:
                stop_simulation(c,s)
            except Exception as err:
                r['cleanup_error'] = str(err)
        if c is not None:
            c.socket.close(linger=0)
            c.context.term()
        parent.put(path,r)
    print(a.case,r.get('final_state'),r['error'] or r.get('reason'),flush=True)
    return int(bool(r['error'] or r['cleanup_error']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    for flag in ('out','evidence-out','clearance-reference'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--case',choices=('short','mid'),required=True)
    return run(p.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
