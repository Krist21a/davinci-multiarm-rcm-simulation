"""Native-HD development cohort for config/learn/learned_perception_v1_native_hd_dev.json.

Thin wrapper over collect.py: for the CLI duration it replaces episode_params with the
registered cohort mapping (each block of 4 consecutive dev indices shares the block
anchor's sampled ECM target; per-index seed/aux_seed kept; condition advance/target and
the nominal scenario come from the plan) and extends source_hashes with the plan and this
wrapper's sha256. Everything else -- sampling order, world, FSM, manifests -- is unchanged.
Aux views are 0; the native HD twin stays enabled.
"""
import json
import sys
from pathlib import Path

import collect

PLAN = collect.REPO / 'config' / 'learn' / 'learned_perception_v1_native_hd_dev.json'


def main():
    plan = json.loads(PLAN.read_bytes())
    c = plan['cohort']
    n = len(c['condition_names'])
    first = c['start']
    parent_params = collect.episode_params
    parent_hashes = collect.source_hashes

    def cohort_params(split, index, proto=None):
        if split != 'dev' or not first <= index < first + c['blocks'] * n:
            raise ValueError('unexpected split/index in development cohort')
        block, condition = divmod(index - first, n)
        params = parent_params(split, index, proto)
        anchor = parent_params(split, first + n * block, proto)
        params.update({k: anchor[k] for k in ('yaw_deg', 'pitch_deg', 'insertion_m')})
        params.update(advance_m=c['advance_um'][condition] / 1e6,
                      target_length_m=c['target_um'][condition] / 1e6,
                      scenario=c['scenario'])
        return params

    collect.episode_params = cohort_params
    collect.source_hashes = lambda proto: dict(
        parent_hashes(proto),
        native_hd_plan_sha256=collect.sha256(PLAN),
        native_hd_wrapper_sha256=collect.sha256(Path(__file__)))
    argv = sys.argv
    try:
        sys.argv = [argv[0], '--split', 'dev', '--start', str(first),
                    '--count', str(c['blocks'] * n), '--workers', '4',
                    '--tag', c['tag'], '--aux-views', '0']
        return collect.main()
    finally:
        collect.episode_params = parent_params
        collect.source_hashes = parent_hashes
        sys.argv = argv


if __name__ == '__main__':
    raise SystemExit(main())
