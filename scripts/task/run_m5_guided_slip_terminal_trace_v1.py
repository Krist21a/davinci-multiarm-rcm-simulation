"""Record the already-rendered wide images around the failed slip terminal review."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_m5_guided_entry_bin_slip_live_v1 as prior
from run_cut_task import REPO, sha256

SPEC = REPO/'config/task/m5_guided_slip_terminal_trace_v1.json'
REF = REPO/'validation/task/m5_guided_entry_bin_slip_live_v1'


class TerminalTraceWorld(prior.EntryBinWorld):
    def _estimate(self):
        if self.tick not in self.args.terminal_ticks:
            return super()._estimate()
        est = self._perc_estimator
        if est is None or 'estimate' in vars(est):
            raise ValueError('working estimator unavailable')
        arrays = [k for k,v in vars(est).items() if isinstance(v,np.ndarray)]
        state = json.dumps({k:v.tolist() if k in arrays else v for k,v in vars(est).items()},allow_nan=False)
        estimate = est.estimate
        f = self.args.evidence_out/f'{self.tick:03d}_estimator_input.npz'

        def capture(rgb,depth,R,t,post_grasp=False):
            with f.open('xb') as fp:
                np.savez_compressed(fp,rgb=rgb.astype(np.uint8),depth=depth,R=R,t=t,post_grasp=post_grasp,
                                    state=state,array_keys=np.asarray(arrays))
            return estimate(rgb,depth,R,t,post_grasp=post_grasp)

        est.estimate = capture
        try:
            result = super()._estimate()
        finally:
            del est.estimate
        self.event_log.append(dict(event='terminal_trace_estimate',tick=self.tick,file=f.name,
                                   sha256=sha256(f),estimate=result))
        return result

    def _update_fix(self,est):
        before = self.awaiting_new_slit
        result = super()._update_fix(est)
        if self.tick in self.args.terminal_ticks:
            self.event_log.append(dict(event='terminal_trace_fix',tick=self.tick,awaiting_before=before,
                                       awaiting_after=self.awaiting_new_slit))
        return result


def main():
    plan = json.loads(SPEC.read_bytes())
    reg = REF/'registration.json'
    if (any(sha256(REPO/k)!=v for k,v in plan['pins'].items())
            or any(sha256(REPO/k)!=v for k,v in json.loads(reg.read_bytes())['sources'].items())):
        raise ValueError('unregistered terminal trace')
    old_run,old_world,parent = prior.base.run,prior.EntryBinWorld,prior.base.parent

    def register(out,policy,files):
        return parent.register(out,plan['id'],(*files,Path(__file__).resolve(),SPEC,*(REPO/k for k in plan['pins'])))

    def run(a):
        a.terminal_ticks = plan['ticks']
        return old_run(a)

    try:
        prior.base.run,prior.EntryBinWorld = run,TerminalTraceWorld
        prior.base.parent = SimpleNamespace(register=register,put=parent.put)
        return prior.main()
    finally:
        prior.base.run,prior.EntryBinWorld,prior.base.parent = old_run,old_world,parent


if __name__=='__main__':
    raise SystemExit(main())
