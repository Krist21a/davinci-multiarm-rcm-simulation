"""One nominal diagnostic with an advance reserve; preserve native closure authority."""
import argparse
import json
import math
from pathlib import Path

import run_same_bite as parent
from m5_guarded_post_bite_v2 import RoutinePostBiteFSM
from run_cut_task import REPO, add_arguments
from run_m5_guarded_post_bite_v2 import ACTIVE, SPEC as PARENT, DevelopmentWorld

SPEC = REPO/'config/task/m5_sweep_reserve_v3.json'


class ReservingWorld(DevelopmentWorld):
    def endpoint_advance_budget(self):
        budget = super().endpoint_advance_budget()
        self.event_log.append(dict(event='native_sweep_budget', tick=self.tick,
                                   budget_m=budget if budget is not None and math.isfinite(budget) else None))
        return budget

    def advance_scissors(self, requested):
        budget = self.endpoint_advance_budget()
        reserve = self.args.reserve_m + self.perc_cfg.get('depth_noise_m', 0.)
        limited = (min(requested, budget-reserve) if budget is not None
                   and all(math.isfinite(x) for x in (requested, budget, reserve)) else None)
        self.event_log.append(dict(event='advance_reserve', tick=self.tick, reserve_m=reserve,
                                   native_budget_m=budget if budget is not None and math.isfinite(budget) else None,
                                   requested_m=requested if math.isfinite(requested) else None,
                                   commanded_m=limited if limited is not None and limited >= .0002 else None))
        if limited is None or limited < .0002:
            self.view_blocked = False
            return False
        return super().advance_scissors(limited)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(SPEC.read_bytes())
    args.reserve_m = plan['advance_reserve_m']
    if args.reserve_m != .00015 or REPO/plan['parent'] != PARENT:
        raise ValueError('unregistered reserve or parent policy')
    args.case = 'nominal'
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    args.capture = None
    return parent.run(args, world_class=ReservingWorld, fsm_class=RoutinePostBiteFSM,
                      active_spec=ACTIVE, policy='sweep_reserve_v3', extra_sources=(
                          SPEC, PARENT, Path(__file__).resolve(),
                          Path(__file__).resolve().with_name('run_m5_guarded_post_bite_v2.py'),
                          Path(__file__).resolve().with_name('m5_guarded_post_bite_v2.py'),
                          REPO/'scripts/vision/m5_local_endpoint_v2.py'))


if __name__ == '__main__':
    raise SystemExit(main())
