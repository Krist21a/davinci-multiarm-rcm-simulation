"""Paired pilot score; clean stops remain task failures and missing trials invalidate pairing."""
import argparse
import json
from pathlib import Path

from run_two_bite_pair import PAIR, digest
from score_two_bite import PLAN, TASK, score, sha256


def main(out):
    if out.exists():
        raise FileExistsError(out)
    plan, pair = (json.loads(p.read_bytes()) for p in (PLAN, PAIR))
    baseline = TASK / 'two_bite_v1_nominal.json'
    base = json.loads(baseline.read_bytes())
    checkpoint = None
    ids, table = set(), []
    for disturbance in pair['disturbances']:
        for strategy in pair['strategies']:
            path = TASK / f'two_bite_pair_v1_{disturbance}_{strategy}.json'
            r = json.loads(path.read_bytes())
            if (r['pair_protocol'] != pair or r['pair_sha256'] != sha256(PAIR) or r['case'] != 'nominal'
                    or r['strategy'] != strategy or r['disturbance'] != disturbance or not r['run_id']
                    or r['run_id'] in ids or r['baseline_sha256'] != sha256(baseline)):
                raise ValueError('not the registered paired run')
            ids.add(r['run_id'])
            row = score(r, plan, injected_front_read=disturbance == 'view')
            cp = r['checkpoint']
            tick = cp['tick']
            if (not cp['nodes_sha256'] or not cp['pose_sha256'] or cp['width_m'] is None
                    or cp['perception_prefix_sha256'] != digest(r['perception'][:tick])
                    or r['perception'][:tick] != base['perception'][:tick]
                    or r['samples'][:tick] != base['samples'][:tick]
                    or (checkpoint is not None and cp != checkpoint)):
                raise ValueError('unmatched first-bite checkpoint')
            checkpoint = cp
            es = r['events']
            injection = [e for e in es if e['event'] == 'pair_injection']
            if len(injection) != 1 or injection[0]['tick'] != tick+1 or injection[0]['kind'] != disturbance:
                raise ValueError('missing/mismatched disturbance')
            if disturbance == 'view':
                if not any(e['tick'] == tick+1 and e['event'] == 'occlude' and e['persistent'] for e in es):
                    raise ValueError('occluder not injected')
            else:
                relaxed = [e for e in es if e['tick'] == tick+1 and e['event'] == 'relax' and e['accepted']
                           and e['relax_m'] == r['config']['events']['relax_m']]
                if sorted(e['arm'] for e in relaxed) != ['psm1', 'psm3']:
                    raise ValueError('creep not applied to both held banks')
            active = [e for e in es if e['event'] == 'optical_traction_request']
            commands = [e for e in es if e['tick'] > tick and e['event'] == 'retract' and e['accepted']
                        and not (disturbance == 'creep' and e['tick'] == tick+1 and e['step'] < 0)]
            if strategy not in ('camera', 'both') and r['camera_moves'] != 0:
                raise ValueError('fixed camera moved')
            if strategy not in ('traction', 'both') and (active or commands):
                raise ValueError('fixed traction changed')
            for e in commands:
                if not any(a['tick'] == e['tick'] and a['step_m'] == e['step'] for a in active):
                    raise ValueError('unattributed assistance command')
            for a in active:
                measured = r['perception'][a['tick']-1]['estimate']['tension_proxy']
                cap = r['config']['cut']['retract_step_m']
                if (not measured['valid'] or measured['width_m'] != a['width_m']
                        or cp['width_m'] != a['reference_m']
                        or max(-cap, min(cap, (cp['width_m']-a['width_m'])/2)) != a['step_m']):
                    raise ValueError('optical traction request disagrees with current RGB-D')
            closures = [e['tick'] for e in es if e['event'] == 'closure']
            table.append(dict(row, disturbance=disturbance, strategy=strategy, report_sha256=sha256(path),
                              observations_to_second_after_checkpoint=closures[1]-tick if len(closures) == 2 else None,
                              optical_traction_requests=len(active), accepted_assist_commands=len(commands),
                              terminal_observation=r['samples'][-1]['tick']))
    with out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(dict(protocol=pair, common_checkpoint=checkpoint, scorer_sha256=sha256(Path(__file__)),
                       bite_scorer_sha256=sha256(Path(__file__).with_name('score_two_bite.py')), trials=table,
                       all_pairs_matched=True, all_safe=all(x['safe'] for x in table)),
                  f, indent=2, allow_nan=False)
        f.write('\n')
    for t in table:
        print(t['disturbance'], t['strategy'], t['state'], t['final_mm'], t['camera_moves'],
              t['accepted_assist_commands'], t['reason'])
    return int(not all(x['safe'] for x in table))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    raise SystemExit(main(parser.parse_args().out))
