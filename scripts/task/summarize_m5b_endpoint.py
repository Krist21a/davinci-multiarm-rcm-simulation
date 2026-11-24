"""Summarize the four registered M5b endpoint diagnostics, not full gate acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from score_m5b import BASE, MAIN, NOISE, REFERENCE, ROOT, TASK, TRUTH, clean_run, closure_metric, cut_length, safe_outcome

RUNS = (TRUTH, 'V_nominal', 'V_occlusion_transient', 'V_noise_depth_0p5')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol = BASE.with_name('m5b_protocol_v1_3_1.json')
    amendment = json.loads(protocol.read_bytes())
    preregistered = json.loads((ROOT / amendment['previous_amendment']).read_bytes())
    cfg = json.loads((BASE.parent / 'cut_task.json').read_bytes())
    target = cfg['cut']['length_m']
    limit = target + cfg['cut']['max_overcut_m']
    max_age = cfg['vision']['fix_max_age_ticks']
    result = {}
    lateral = []
    for key in RUNS:
        path = TASK / f'm5b_v1_3_1_{key}.json'
        raw = path.read_bytes()
        run = json.loads(raw)
        entry = result[key] = dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
                                   clean_run=clean_run(run), state=run.get('final_state'), reason=run.get('reason'),
                                   cut_length_m=cut_length(run), torn=(run.get('final_observation') or {}).get('torn'))
        if key == TRUTH:
            trace = run.get('trace')
            entry['B0'] = bool(trace and trace == json.loads(REFERENCE.read_bytes())['trace'])
            continue
        if run.get('protocol_amendment') != amendment:
            p.error(f'{path.name}: wrong embedded protocol amendment')
        events = run.get('event_log', [])
        closures = [e for e in events if e['event'] == 'closure']
        recovered = [e for e in events if e['event'] == 'post_closure_reacquired']
        entry.update(closures=len(closures), reacquisitions=recovered,
                     strict_B5=safe_outcome(run, target, limit),
                     invalid_closure_ticks=[e['tick'] for e in closures
                                            if not (isinstance(e.get('fix_age'), int) and 0 <= e['fix_age'] <= max_age
                                                    and e.get('scissors_valid') is True)])
        if key in ('V_nominal', *NOISE):
            lateral.extend(e.get('lateral_truth_m') for e in closures)
    b7 = closure_metric(lateral, len(lateral), cfg['cut']['align_tolerance_m'])
    if b7['median_m'] is not None and not math.isfinite(b7['median_m']):
        p.error('nonfinite B7 median')
    report = dict(amendment=amendment, protocol=preregistered, scope='Partial development diagnostics only',
                  protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(), runs=result,
                  unrun=[item['id'] for item in json.loads(BASE.read_bytes())['runs'] if item['id'] not in RUNS],
                  main_done_observed=[k for k in MAIN if result.get(k, {}).get('state') == 'DONE'],
                  noise_done_observed=[k for k in NOISE if result.get(k, {}).get('state') == 'DONE'],
                  B7_diagnostic_samples=b7)
    with args.out.open('x', encoding='utf-8', newline='\n') as dst:
        json.dump(report, dst, indent=2, allow_nan=False)
        dst.write('\n')


if __name__ == '__main__':
    main()
