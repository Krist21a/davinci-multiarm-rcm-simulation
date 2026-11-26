"""Execute the opt-in nominal HD and routine post-bite development controller."""
import argparse
import json
from pathlib import Path
import sys

import run_same_bite as parent
from run_cut_task import REPO, add_arguments
from m5_guarded_post_bite_v2 import NativeHDWorld, RoutinePostBiteFSM

sys.path.insert(0, str(REPO / 'scripts/vision'))
from m5_local_endpoint_v2 import measure as local_endpoint

SPEC = REPO / 'config/task/m5_guarded_post_bite_v2.json'
ACTIVE = REPO / 'config/task/m5_live_active_look_v1.json'


class DevelopmentWorld(NativeHDWorld):
    def record_post_bite(self, obs):
        R, t = self.camera()
        result = local_endpoint(self.image, self.depth, R, t, self.f, self._last_est)
        self.event_log.append(dict(event='local_endpoint_shadow', tick=self.tick, **result))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.case = 'nominal'
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    args.capture = None
    return parent.run(args, world_class=DevelopmentWorld, fsm_class=RoutinePostBiteFSM,
                      active_spec=ACTIVE, policy='guarded_post_bite_v2', extra_sources=(
                          SPEC, Path(__file__).resolve(), Path(__file__).resolve().with_name('m5_guarded_post_bite_v2.py'),
                          REPO/'scripts/vision/m5_local_endpoint_v2.py'))


if __name__ == '__main__':
    raise SystemExit(main())
