"""One isolated three-closure 10 mm candidate with a smaller second advance."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import run_m5_guided_benchmark_v2 as v2
from m5_guided_controller_v2 import GuidedFSMV2, GuidedWorldV2
from run_cut_task import REPO, add_arguments, sha256

SPEC = REPO/'config/task/m5_guided_10mm_control_v1.json'
V2_OUT = REPO/'validation/task/m5_guided_benchmark_v2'


def limit_second(requested, closures, cap):
    return min(requested,cap) if closures==1 else requested


class ControlWorld(GuidedWorldV2):
    closures = 0

    def advance_scissors(self,requested):
        if self.closures>=self.args.maximum_closures:
            return False
        limited = limit_second(requested,self.closures,self.args.second_advance_cap_m)
        if self.closures==1:
            self.event_log.append(dict(event='control_second_advance_cap',tick=self.tick,
                                       requested_m=requested,limited_m=limited))
        return super().advance_scissors(limited)

    def open_scissors(self):
        return self.closures<self.args.maximum_closures and super().open_scissors()

    def close_scissors(self):
        if self.closures>=self.args.maximum_closures or not super().close_scissors():
            return False
        self.closures += 1
        return True


class ControlFSM(GuidedFSMV2):
    def _path(self,obs,points,phase,camera):
        if phase=='resume' and self.w.closures>=self.w.args.maximum_closures:
            raise ValueError('10 mm control stops closed after its third closure')
        return super()._path(obs,points,phase,camera)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    for flag in ('out','evidence-out'):
        p.add_argument('--'+flag,type=Path,required=True)
    a = p.parse_args()
    plan = json.loads(SPEC.read_bytes())
    decision,reg = V2_OUT/'decision.json',V2_OUT/'registration.json'
    if (sha256(decision)!=plan['nominal_v2_decision_sha256'] or not json.loads(decision.read_bytes())['passed']
            or sha256(reg)!=plan['nominal_v2_registration_sha256']
            or any(sha256(REPO/k)!=v for k,v in json.loads(reg.read_bytes())['sources'].items())
            or plan['second_advance_cap_m']!=.0025 or plan['maximum_closures']!=3):
        raise ValueError('unregistered 10 mm control sources or intervention')
    a.second_advance_cap_m,a.maximum_closures = plan['second_advance_cap_m'],plan['maximum_closures']
    old_spec,parent,world,fsm = v2.SPEC,v2.parent,v2.GuidedWorldV2,v2.GuidedFSMV2

    def register(out, policy, files):
        return parent.register(out,policy,(*files,Path(__file__).resolve(),old_spec,reg,decision))

    try:
        v2.SPEC,v2.parent = SPEC,SimpleNamespace(register=register,put=parent.put)
        v2.GuidedWorldV2,v2.GuidedFSMV2 = ControlWorld,ControlFSM
        return v2.run(a)
    finally:
        v2.SPEC,v2.parent,v2.GuidedWorldV2,v2.GuidedFSMV2 = old_spec,parent,world,fsm


if __name__=='__main__':
    raise SystemExit(main())
