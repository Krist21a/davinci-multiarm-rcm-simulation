"""Preregistered mechanical screening of the angled, printed development fixture."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import capture_ruler_endpoint as acquisition
from simtools import add_arguments, sha256

PILOT = acquisition.ROOT/'config/task/ruler_fixture_pilot_v1.json'


def run(args):
    from ruler_fixture import FixtureWorld

    if args.angle not in json.loads(PILOT.read_bytes())['tilt_deg'] or args.cap != 3 or args.goal not in (8,9):
        raise ValueError('unregistered placement study')
    with patch.object(acquisition, 'RulerWorld', FixtureWorld):
        rc = acquisition.main(args)
    oracle = args.out/'oracle.json'
    rec = json.loads(oracle.read_bytes())
    report = dict(angle_deg=args.angle, oracle_sha256=sha256(oracle), pilot_sha256=sha256(PILOT),
                  code_sha256=sha256(Path(__file__)), fixture_sha256=sha256(acquisition.ROOT/'scripts/task/ruler_fixture.py'),
                  state=rec.get('final_state'), reason=rec.get('error') or rec.get('reason'), eligible=False)
    held = any(set(e.get('obs', {}).get('grasped', {})) >= {'psm1','psm3'}
               and min(e['obs']['grasped'][a] for a in ('psm1','psm3')) >= rec['config']['grasp']['min_nodes']
               for e in rec.get('log', []) if e.get('to')=='ALIGN' and e.get('obs'))
    report['both_arms_at_align'] = held
    report['eligible'] = rc == 0 and held and [f['phase'] for f in rec['frames']] == ['loaded','work','park']
    with (args.out/'fixture.json').open('x', encoding='utf-8',newline='\n') as f:
        json.dump(report,f,indent=2)
        f.write('\n')
    return int(not report['eligible'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--angle',type=int,required=True)
    ap.add_argument('--goal',type=int,required=True)
    ap.add_argument('--cap',type=float,default=3)
    ap.add_argument('--out',type=Path,required=True)
    raise SystemExit(run(ap.parse_args()))
