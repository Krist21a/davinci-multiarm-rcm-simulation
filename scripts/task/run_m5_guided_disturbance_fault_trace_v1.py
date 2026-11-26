"""Observe the existing disturbance runner's calls without altering control."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_m5_guided_benchmark_v2_disturbances_v1 as base
from m5_guided_controller_v2 import GuidedWorldV2
from run_cut_task import REPO, add_arguments, sha256

SPEC = REPO/'config/task/m5_guided_disturbance_fault_trace_v1.json'
REF = REPO/'validation/task/m5_guided_benchmark_v2_disturbances_v1'
CASES = ('depth_noise_0p5','slip_regrasp')


class TraceWorld(GuidedWorldV2):
    trace_solve = False

    def trace(self,name,**fields):
        low,high = self.args.trace_window
        if low<=self.tick<=high:
            self.event_log.append(dict(event='fault_trace_'+name,tick=self.tick,**fields))

    def call(self,name,*args):
        r = super().call(name,*args)
        low,high = self.args.trace_window
        if low<=self.tick<=high and args and args[0]==self.index[self.cutter]:
            if name=='toolPoseFK' and (not isinstance(r,dict) or r.get('ok') is not True
                                       or r.get('shaft_ok') is not True):
                info = {k:r.get(k) for k in ('ok','shaft_ok','reason')} if isinstance(r,dict) else None
                self.trace('fk_refused',in_solve=self.trace_solve,q=list(args[1:]),result=info)
            if name=='setToolPose':
                self.trace('tool_command',q=list(args[1:]),accepted=bool(r))
        return r

    def _tool_q(self,arm):
        r = super()._tool_q(arm)
        if self.trace_solve and arm==self.cutter:
            self.trace('seed',valid=r is not None,q=None if r is None else r[0].tolist())
        return r

    def _tool_solve(self,arm,target_p,target_axis=None,target_yaw=None):
        if arm!=self.cutter:
            return super()._tool_solve(arm,target_p,target_axis,target_yaw)
        self.trace_solve = True
        try:
            r = super()._tool_solve(arm,target_p,target_axis,target_yaw)
        finally:
            self.trace_solve = False
        keys = ('converged','pos_err_m','axis_err_rad','yaw_axis_err_rad','iterations','at_limit')
        self.trace('solve',target_m=np.asarray(target_p).tolist(),
                   result=None if r is None else {k:r[k] for k in keys})
        return r

    def clear_view(self,d,prefer_park=False):
        r = super().clear_view(d,prefer_park=prefer_park)
        self.trace('clear',distance_m=d,prefer_park=prefer_park,accepted=bool(r))
        return r

    def _estimate(self):
        est = super()._estimate()
        low,high = self.args.trace_window
        if not low<=self.tick<=high:
            return est
        cut = (est or {}).get('cut') or {}
        scis = (est or {}).get('scissors') or {}
        chart = cut.get('material_chart') or {}
        self.trace('estimate',front_valid=bool(cut.get('front_valid')),reason=cut.get('reason'),
                   mode=cut.get('mode'),slit_px=cut.get('slit_px'),progress_m=cut.get('progress_m'),
                   chart_valid=bool(chart.get('valid')),chart_reason=chart.get('reason'),
                   scissors_valid=bool(scis.get('valid')),scissors_reason=scis.get('reason'))
        if self.tick in self.args.trace_frames:
            p = self.args.evidence_out/f'{self.tick:03d}_fault_working.npz'
            with p.open('xb') as f:
                np.savez_compressed(f,rgb=np.asarray(self.image),native_depth=self.depth)
            self.trace('working_frame',file=p.name,sha256=sha256(p))
        return est


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--case',choices=CASES,required=True)
    for k in ('out','evidence-out'):
        p.add_argument('--'+k,type=Path,required=True)
    a = p.parse_args()
    plan = json.loads(SPEC.read_bytes())
    r = REF/'registration.json'
    inputs = tuple(REF/(k+'.json') for k in plan['pins'])
    if (any(sha256(f)!=plan['pins'][f.stem] for f in inputs)
            or any(sha256(REPO/k)!=v for k,v in json.loads(r.read_bytes())['sources'].items())):
        raise ValueError('diagnostic prerequisite changed')
    a.trace_window = plan['windows'][a.case]
    a.trace_frames = plan['working_frame_ticks'][a.case]
    parent,world = base.parent,base.GuidedWorldV2

    def register(out,policy,files):
        return parent.register(out,plan['id'],(*files,Path(__file__).resolve(),SPEC,r,*inputs))

    try:
        base.parent,base.GuidedWorldV2 = SimpleNamespace(register=register,put=parent.put),TraceWorld
        return base.run(a)
    finally:
        base.parent,base.GuidedWorldV2 = parent,world


if __name__=='__main__':
    raise SystemExit(main())
