"""Execute the versioned active-look development policy with the unchanged full-length scorer."""
import argparse
import json
from pathlib import Path

import run_same_bite as parent
from live_active_look import ActiveLookFSM, ActiveLookWorld
from run_cut_task import REPO, add_arguments

SPEC=REPO/'config/task/m5_live_active_look_v1.json'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--case',required=True)
    args=ap.parse_args()
    protocol=json.loads(SPEC.read_bytes())
    args.targets=protocol['recovery_targets']
    if args.case not in json.loads(parent.SPEC.read_bytes())['cases']:
        ap.error('unknown case')
    args.capture=args.out/(args.case+'_hd')
    args.capture.mkdir(parents=True,exist_ok=False)
    return parent.run(args,world_class=ActiveLookWorld,fsm_class=ActiveLookFSM,active_spec=SPEC)


if __name__=='__main__':
    raise SystemExit(main())
