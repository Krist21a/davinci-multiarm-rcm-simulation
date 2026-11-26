"""Exact-target control; keep the original 10 mm upper safety envelope."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import run_m5_guided_benchmark_v2 as v2
from m5_guided_controller_v2 import GuidedWorldV2
from run_cut_task import REPO, S, add_arguments, sha256
from run_m5_guided_10mm_control_v1 import ControlFSM, ControlWorld, V2_OUT

SPEC = REPO/'config/task/m5_guided_10mm_exact_v1.json'
PRIOR = REPO/'validation/task/m5_guided_10mm_control_v1/decision.json'


def exact_settings(cut, plan):
    limit = plan['absolute_limit_m']
    if (cut['length_m']!=.009 or cut['max_overcut_m']!=.001 or cut['advance_m']!=.003
            or abs(cut['length_m']+cut['max_overcut_m']-limit)>1e-12
            or plan['exact_control_target_m']!=limit or plan['exact_control_overcut_m']!=0.0
            or limit!=.010):
        raise ValueError('exact control must preserve the absolute native sweep limit')
    return dict(cut,length_m=limit,max_overcut_m=0.0)


class ExactWorld(ControlWorld):
    def __init__(self,client,sim,args,cfg,**kwargs):
        cfg['cut'] = exact_settings(cfg['cut'],args.control_plan)
        super().__init__(client,sim,args,cfg,**kwargs)

    def advance_scissors(self,requested):
        if self.closures>=self.args.maximum_closures:
            return False
        return GuidedWorldV2.advance_scissors(self,requested)


class ExactFSM(ControlFSM):
    def _complete(self,obs):
        w = self.w
        if w.withdrawal_tick==w.tick:
            e = w.event_log[-1]
            if (e.get('event')!='guided_v2_optical' or e.get('tick')!=w.tick
                    or e.get('verdict')!='TARGET' or e.get('prefix_mm')!=10 or w.closures!=3):
                w.withdrawal_tick = None
                return self.go(S.FAILED,'exact control requires current TARGET(10) after three closures',obs)
        return super()._complete(obs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    for flag in ('out','evidence-out'):
        p.add_argument('--'+flag,type=Path,required=True)
    a = p.parse_args()
    plan = json.loads(SPEC.read_bytes())
    dec,reg = V2_OUT/'decision.json',V2_OUT/'registration.json'
    if (sha256(dec)!=plan['nominal_v2_decision_sha256'] or not json.loads(dec.read_bytes())['passed']
            or sha256(reg)!=plan['nominal_v2_registration_sha256'] or sha256(PRIOR)!=plan['prior_control_sha256']
            or any(sha256(REPO/k)!=v for k,v in json.loads(reg.read_bytes())['sources'].items())
            or plan['maximum_closures']!=3 or plan['ordinary_advance_cap_m']!=.003):
        raise ValueError('unregistered exact control sources')
    a.control_plan,a.maximum_closures = plan,plan['maximum_closures']
    spec,parent,world,fsm = v2.SPEC,v2.parent,v2.GuidedWorldV2,v2.GuidedFSMV2

    def register(out,policy,files):
        return parent.register(out,policy,(*files,Path(__file__).resolve(),spec,dec,reg,PRIOR,
               Path(__file__).with_name('run_m5_guided_10mm_control_v1.py')))

    try:
        v2.SPEC,v2.parent = SPEC,SimpleNamespace(register=register,put=parent.put)
        v2.GuidedWorldV2,v2.GuidedFSMV2 = ExactWorld,ExactFSM
        return v2.run(a)
    finally:
        v2.SPEC,v2.parent,v2.GuidedWorldV2,v2.GuidedFSMV2 = spec,parent,world,fsm


if __name__=='__main__':
    raise SystemExit(main())
