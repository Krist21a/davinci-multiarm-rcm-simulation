"""Optical-only capture on fixed diagnostic bites under the guided working guards."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import uuid

import numpy as np

import run_same_bite as parent
from run_m5_guided_benchmark_v1 import ACTIVE, CERT, PRINT, SHEET, GuidedFSM, GuidedWorld
from run_cut_task import REPO, S, add_arguments, connect, sha256, stop_simulation
from cut_fsm import CutTaskFSM
from simtools import matrix_to_np


SPEC = REPO/'config/task/m5_guided_terminal_controls_v1.json'


class CaptureWorld(GuidedWorld):
    bites = 0
    diagnostic_armed = False
    final_advance_attempted = False

    def advance_scissors(self, requested):
        if self.bites < 2:
            return super().advance_scissors(requested)
        if self.bites != 2 or not self.diagnostic_armed or self.final_advance_attempted:
            self.view_blocked = False
            return False
        self.final_advance_attempted = True
        self.event_log.append(dict(event='diagnostic_final_advance_request', tick=self.tick,
                                   requested_m=requested, fixed_cap_m=self.args.cap_m))
        return super().advance_scissors(min(requested,self.args.cap_m))

    def close_scissors(self):
        if self.bites >= 3 or (self.bites == 2 and not (self.diagnostic_armed and self.final_advance_attempted)):
            return False
        ok = super().close_scissors()
        if ok:
            self.bites += 1
        return ok

    def record_post_bite(self, obs):
        super().record_post_bite(obs)
        if self.bites == 3:
            self.review_armed = True

    def retract_scissors(self):
        return False


class CaptureFSM(GuidedFSM):
    def _complete(self, obs):
        if self.w.bites == 3 and self.w.review_armed and self.w.jaws_closed():
            return self.go(S.ALIGN,'final diagnostic bite needs closed optical capture',obs)
        return self.go(S.FAILED,'diagnostic never authorizes task completion',obs)

    def _entry_clearance_capture(self, obs):
        w = self.w
        initial = w.bites == 2 and not w.diagnostic_armed
        final = w.bites == 3 and w.diagnostic_armed
        if not (initial or final) or w.direct_progress(obs) is None or not w.jaws_closed():
            return self.go(S.FAILED,'diagnostic capture needs fresh wide front and closed jaws',obs)
        if final:
            with w.truth.purpose('scoring'):
                pre_edges = w.truth._read('constraints',lambda:w.call('tissueConstraints'))
        w.read_review(obs)
        before = w.event_log[-1]
        if initial:
            self._require_reference(before,'039_certificate.npz')
        est = w._last_est
        entry,direction,tip = (np.asarray(p,dtype=float) for p in (
            est['cut'].get('material_start'),est['cut'].get('direction'),est['scissors'].get('tip')))
        if (any(p.shape!=(3,) or not np.isfinite(p).all() for p in (entry,direction,tip))
                or not math.isclose(float(np.linalg.norm(direction)),1.,abs_tol=1e-6)):
            return self.go(S.FAILED,'diagnostic clearance lacks bounded optical geometry',obs)
        handle = w.by[w.cutter+'_tcp']
        start = matrix_to_np(w.sim.getObjectMatrix(handle,-1))
        axis = start[:,w.TOOL_AXIS_COL]
        projection = float(axis@direction)
        tip_s = float((tip-entry)@direction)
        retreat = (tip_s+.0025)/projection if projection>=.8 else math.nan
        if not 0<retreat<=.008:
            return self.go(S.FAILED,'diagnostic entry clearance exceeds travel bound',obs)
        tick = w.tick
        ok = w.clear_view(retreat,prefer_park=False)
        delta = matrix_to_np(w.sim.getObjectMatrix(handle,-1))[:,3]-start[:,3]
        residual = float(np.linalg.norm(delta+retreat*axis))
        margin = tip_s+float(delta@direction)
        accepted = bool(ok and residual<=.0005 and margin<=-.002 and w.jaws_closed())
        w.event_log.append(dict(event='diagnostic_entry_clearance',tick=tick,accepted=accepted,
                                requested_m=retreat,residual_m=residual,estimated_tip_after_m=margin))
        if not accepted:
            return self.go(S.FAILED,'diagnostic entry clearance refused',obs)
        after = w.observe()
        blocked = CutTaskFSM.supervise(self,replace(after,cut_site_visible=True,
                                                   scissors_visible=True,cut_site_offset=0.0))
        if blocked or w.tick<=tick or not w.jaws_closed():
            return self.go(S.FAILED,'diagnostic clearance failed fresh closed interlock',after)
        w.read_review(after)
        last = w.event_log[-1]
        if initial:
            self._require_reference(last,'040_certificate.npz')
            if w.direct_progress(after) is None or not self.visible(after) or not self.centred(after):
                return self.go(S.FAILED,'matched entry lacks fresh wide view for alignment',after)
            w.diagnostic_armed = True
            w.event_log.append(dict(event='diagnostic_final_bite_armed',tick=w.tick,matched=True))
            return self.go(S.ALIGN,'registered diagnostic bite; native guards still required',after)
        with w.truth.purpose('scoring'):
            post_edges = w.truth._read('constraints',lambda:w.call('tissueConstraints'))
        structure = w.args.out/(w.args.case+'_structure.npz')
        with structure.open('xb') as f:
            np.savez_compressed(f, before_edges=np.asarray(pre_edges), after_edges=np.asarray(post_edges))
        w.event_log.append(dict(event='diagnostic_final_capture',tick=w.tick,pre_file=before['file'],
                                pre_sha256=before['frame_sha256'],file=last['file'],
                                sha256=last['frame_sha256'],structure_file=structure.name,
                                structure_sha256=sha256(structure)))
        return self.go(S.FAILED,'diagnostic acquisition complete; held closed',after)

    def _require_reference(self,event,name):
        a = self.w.args
        with np.load(a.evidence_out/event['file'],allow_pickle=False) as now, np.load(
                a.clearance_reference/name,allow_pickle=False) as ref:
            if set(now.files)!=set(ref.files) or any(not np.array_equal(now[k],ref[k]) for k in now.files):
                raise ValueError('registered closed optical reference did not reproduce: '+name)


def run(a):
    plan = json.loads(SPEC.read_bytes())
    if (any(sha256(REPO/k)!=sha for k,sha in plan['pins'].items())
            or any(sha256(a.clearance_reference/k)!=sha for k,sha in plan['reference_frames'].items())):
        raise ValueError('terminal acquisition evidence pin changed')
    task = Path(__file__).resolve().parent
    files = (SPEC,Path(__file__).resolve(),ACTIVE,PRINT,CERT,
             *(task/k for k in ('run_m5_guided_benchmark_v1.py','run_m5_retained_print_v7.py',
                'run_m5_endpoint_review_v4.py','run_m5_sweep_reserve_v3.py','seam_topology.py',
                'm5_guarded_post_bite_v2.py','run_m5_guarded_post_bite_v2.py','live_active_look.py')),
             REPO/'scripts/learn/capture_m5_temporal_coded.py',REPO/'scripts/vision/m5_guided_prefix_v1.py')
    base = parent.register(a.out,plan['id'],files)
    rpath = a.out/(a.case+'.json')
    if rpath.exists():
        raise FileExistsError(rpath)
    a.initial_view,a.initial_view_insertion_m = base['initial_view'][:2],base['initial_view'][2]
    a.scene,a.scenario,a.run_id = REPO/base['scene'],'nominal',uuid.uuid4().hex
    a.cap_m,a.reserve_m,a.capture = plan['cases'][a.case],.00015,None
    a.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    a.review_target = json.loads((REPO/'config/task/m5_guided_benchmark_v1.json').read_bytes())['review']['target']
    a.evidence_out.mkdir(exist_ok=False)
    cfg = json.loads((REPO/base['base']).read_bytes())
    cfg['sheet'] = SHEET.copy()
    cfg['cut']['control_source'] = 'vision'
    cfg['perception'].update(mode='shadow',markers=True,depth_noise_m=0.,seed=0,material_pattern=True,
                             local_material_chart=True,front_recovery=True,predictive_view=True,
                             exclude_tcp_marker=True,recovery_view_after_cut=True,ray_marker_fit=True)
    r = dict(case=a.case,run_id=a.run_id,config=cfg,cap_m=a.cap_m,observations=[],
             registration_sha256=sha256(a.out/'registration.json'),error=None,cleanup_error=None)
    c=s=w=fsm=None
    try:
        c,s = connect(a)
        w = CaptureWorld(c,s,a,cfg,mode='full')
        if not w.canonical_seam or not w.truth.strict or not w.vision:
            raise ValueError('wrong guided world')
        fsm = CaptureFSM(w,cfg)
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
        parent.put(rpath,r)
    print(a.case,r.get('final_state'),r['error'] or r.get('reason'),flush=True)
    return int(bool(r['error'] or r['cleanup_error']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evidence-out',type=Path,required=True)
    p.add_argument('--clearance-reference',type=Path,required=True)
    p.add_argument('--case',required=True,choices=('short','mid','full'))
    return run(p.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
