"""Independent optical and structural grading; intermediate cuts are diagnostic."""
from pathlib import Path

import score_m5_guided_benchmark_v2_1 as base
from score_m5_guided_benchmark_v2_2 import bind, main as cli
from m5_guided_bank_corridor_v4 import measure
from replay_cell_v2 import ROOT, digest, load, write

ID = 'm5_guided_benchmark_v2_4'
SPEC = ROOT/f'config/task/{ID}.json'
# A positive key skips disturbance coverage; None selects the 9-10 mm interval.
POSITIVES = {'nominal': None, 'exact_10mm': 10}
registration = bind(base.registration, ID=ID, __file__=__file__)
predict = bind(base.predict, SPEC=SPEC, registration=registration, measure=measure)
grade = bind(base.grade, POSITIVES=POSITIVES)


def score(out, file, positive_gate):
    pred, reg = load(file), registration(out)
    order = tuple(load(SPEC)['case_order'])
    if pred['registration'] != reg or tuple(pred['cases']) != order:
        raise ValueError('predictions use another registration')
    if positive_gate and any(pred['cases'][k] is not None for k in order[2:]):
        raise ValueError('positives must be scored before disturbances')
    rows, reports = [], {}
    for case, p in pred['cases'].items():
        if p is None:
            rows.append(dict(case=case, tested=False, passed=False))
            continue
        r, f = base.source(out, case, reg)
        reports[case] = digest(f)
        if reports[case] != p['report_sha256']:
            raise ValueError('run report changed after prediction')
        row = grade(out, case, r, p)
        row['structural_prefix_after_closures_m'] = [
            next((o['cut_length_m'] for o in r['observations'] if o['tick'] > t), None)
            for t in row['closures']
        ]
        rows.append(row)
    positives_passed = all(r['passed'] for r in rows[:2])
    write(out/('positive_gate.json' if positive_gate else 'decision.json'), dict(
        predictions_sha256=digest(file), registration=reg, report_sha256=reports, cases=rows,
        passed=positives_passed if positive_gate else all(r['passed'] for r in rows),
        positive_cases_passed=positives_passed, disturbance_passes=sum(r['passed'] for r in rows[2:]),
        replaces_historical_acceptance=False))
    for r in rows:
        print(r['case'], r['passed'], r.get('state'), r.get('reason'),
              r.get('structural_prefix_after_closures_m'), r.get('structure'), flush=True)


main = bind(cli, __doc__=__doc__, SPEC=SPEC, registration=registration, predict=predict, score=score)

if __name__ == '__main__':
    main()
