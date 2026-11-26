"""Snapshot exact perception state and test original targets with no candidate motion."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_m5_guided_disturbance_fault_trace_v1 as trace
import tool_pose
from run_cut_task import REPO, sha256

SPEC = REPO/'config/task/m5_guided_disturbance_dry_diagnostic_v1.json'
REF = REPO/'validation/task/m5_guided_disturbance_fault_trace_v1'


class DryWorld(trace.TraceWorld):
    def _tool_solve(self,arm,target_p,target_axis=None,target_yaw=None):
        native = super()._tool_solve(arm,target_p,target_axis,target_yaw)
        if (self.tick!=self.args.dry_plan['dry_ticks'][self.args.case] or arm!=self.cutter
                or native is None or native['converged']):
            return native
        q,ws = self._tool_q(arm)
        stamp = self.sim.getSimulationTime()
        closed = self.jaws_closed()
        if not np.array_equal(q,native['q0']):
            raise ValueError('native seed changed')
        lo,hi = self._tool_limits(arm,ws)
        inward = q.copy()
        inward[0] = np.sign(q[0])*max(0.,abs(q[0])-1.)
        variants = zip(self.args.dry_plan['contrasts'],(inward,q),(.02,.01),strict=True)
        rows = []
        for name,seed,w_rot in variants:
            try:
                result = tool_pose.solve(self._tool_fk(arm),seed,(lo,hi),np.asarray(target_p),
                                         target_axis=target_axis,target_yaw_axis=target_yaw,w_rot=w_rot,
                                         axis_col=self.TOOL_AXIS_COL,yaw_col=self.TOOL_YAW_COL)
                result['q'] = result['q'].tolist()
                rows.append(dict(name=name,seed_q=seed.tolist(),result=result,fk_error=None))
            except RuntimeError as err:
                rows.append(dict(name=name,seed_q=seed.tolist(),result=None,fk_error=str(err)))
        if (not np.array_equal(q,self._tool_q(arm)[0]) or self.sim.getSimulationTime()!=stamp
                or self.jaws_closed()!=closed):
            raise ValueError('dry solver changed the world')
        original = {k:native[k] for k in ('converged','pos_err_m','axis_err_rad','yaw_axis_err_rad','iterations','at_limit')}
        self.trace('dry_contrast',target_m=np.asarray(target_p).tolist(),native=original,
                   original_q=q.tolist(),lo=lo.tolist(),hi=hi.tolist(),variants=rows)
        return native

    def _estimate(self):
        if self.args.case!='slip_regrasp' or self.tick not in self.args.dry_plan['optical_ticks']:
            return super()._estimate()
        est = self._perc_estimator
        if est is None or 'estimate' in vars(est):
            raise ValueError('working estimator state unavailable')
        arrays = [k for k,v in vars(est).items() if isinstance(v,np.ndarray)]
        state = json.dumps({k:v.tolist() if k in arrays else v for k,v in vars(est).items()},allow_nan=False)
        estimate = est.estimate
        p = self.args.evidence_out/f'{self.tick:03d}_estimator_input.npz'

        def capture(rgb,depth,R,t,post_grasp=False):
            with p.open('xb') as f:
                np.savez_compressed(f,rgb=rgb.astype(np.uint8),depth=depth,R=R,t=t,post_grasp=post_grasp,
                                    state=state,array_keys=np.asarray(arrays))
            self.trace('estimator_input',file=p.name,sha256=sha256(p))
            return estimate(rgb,depth,R,t,post_grasp=post_grasp)

        est.estimate = capture
        try:
            return super()._estimate()
        finally:
            del est.estimate


def main():
    plan = json.loads(SPEC.read_bytes())
    reg,diag = REF/'registration.json',REF/'diagnosis.json'
    if (sha256(reg)!=plan['trace_registration_sha256'] or sha256(diag)!=plan['trace_diagnosis_sha256']
            or not json.loads(diag.read_bytes())['control_trace_matched']
            or any(sha256(REPO/k)!=v for k,v in json.loads(reg.read_bytes())['sources'].items())
            or plan['contrasts']!=['yaw_seed_inward_1deg','orientation_weight_half']):
        raise ValueError('unregistered dry contrast')
    tm,world,parent = trace.base.run,trace.TraceWorld,trace.base.parent
    spec = trace.SPEC

    def register(out,policy,files):
        return parent.register(out,plan['id'],(*files,Path(__file__).resolve(),SPEC,reg,diag,
                                                Path(trace.__file__).resolve()))

    def run(a):
        a.dry_plan = plan
        return tm(a)

    try:
        trace.TraceWorld,trace.base.run = DryWorld,run
        trace.base.parent = SimpleNamespace(register=register,put=parent.put)
        return trace.main()
    finally:
        trace.TraceWorld,trace.base.run,trace.base.parent = world,tm,parent


if __name__=='__main__':
    raise SystemExit(main())
