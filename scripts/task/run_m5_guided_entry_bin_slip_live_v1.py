"""Run only the registered slip case with the opt-in entry-bin reader."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import run_m5_guided_benchmark_v2_disturbances_v1 as base
from m5_guided_entry_bin_v1 import EntryBinPerception
from run_cut_task import REPO, add_arguments, sha256

SPEC = REPO/'config/task/m5_guided_entry_bin_slip_live_v1.json'
READER = REPO/'validation/task/m5_guided_entry_bin_v1'


class EntryBinWorld(base.GuidedWorldV2):
    def _estimate(self):
        if self._perc_estimator is None:
            if self._perc_meta is None:
                self._perc_meta = self._dump_meta()
            self._perc_estimator = EntryBinPerception.from_meta(self._perc_meta)
            self.event_log.append(dict(event='entry_bin_estimator_created',tick=self.tick))
        return super()._estimate()

    def advance_scissors(self,requested):
        tick = self.tick
        ok = super().advance_scissors(requested)
        self.event_log.append(dict(event='entry_bin_advance',tick=tick,requested_m=requested,accepted=bool(ok)))
        return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    for k in ('out','evidence-out'):
        p.add_argument('--'+k,type=Path,required=True)
    a = p.parse_args()
    plan = json.loads(SPEC.read_bytes())
    a.case = plan['case']
    reg = READER/'registration.json'
    if (a.case!='slip_regrasp' or any(sha256(REPO/k)!=v for k,v in plan['pins'].items())
            or any(sha256(REPO/k)!=v for k,v in json.loads(reg.read_bytes())['sources'].items())
            or not json.loads((READER/'decision.json').read_bytes())['archived_replay_passed']):
        raise ValueError('unregistered entry bin slip probe')
    parent,world = base.parent,base.GuidedWorldV2

    def register(out,policy,files):
        return parent.register(out,plan['id'],(*files,Path(__file__).resolve(),SPEC,*(REPO/k for k in plan['pins']),
                 Path(base.__file__).resolve(),REPO/'scripts/vision/m5_guided_entry_bin_v1.py'))

    try:
        base.parent,base.GuidedWorldV2 = SimpleNamespace(register=register,put=parent.put),EntryBinWorld
        return base.run(a)
    finally:
        base.parent,base.GuidedWorldV2 = parent,world


if __name__=='__main__':
    raise SystemExit(main())
