"""Independent v4 optical replay and structural grading for the v2.2 gates."""
import argparse
from pathlib import Path
from types import FunctionType

import score_m5_guided_benchmark_v2_1 as base
from m5_guided_bank_corridor_v4 import measure
from replay_cell_v2 import ROOT, digest, load, write

ID = 'm5_guided_benchmark_v2_2'
SPEC = ROOT/f'config/task/{ID}.json'


def bind(fn,**changes):
    return FunctionType(fn.__code__,dict(fn.__globals__,**changes),fn.__name__)


registration = bind(base.registration,ID=ID,__file__=__file__)
predict = bind(base.predict,SPEC=SPEC,registration=registration,measure=measure)


def score(out,file,positive_gate):
    pred = load(file)
    reg = registration(out)
    order = tuple(load(SPEC)['case_order'])
    if pred['registration']!=reg or tuple(pred['cases'])!=order:
        raise ValueError('predictions use another registration')
    if positive_gate and any(pred['cases'][k] is not None for k in order[2:]):
        raise ValueError('positives must be scored before disturbances')
    rows = []
    reports = {}
    for case,p in pred['cases'].items():
        if p is None:
            rows.append(dict(case=case,tested=False,passed=False))
            continue
        r,f = base.source(out,case,reg)
        reports[case] = digest(f)
        if reports[case]!=p['report_sha256']:
            raise ValueError('run report changed after prediction')
        row = base.grade(out,case,r,p)
        obs = r['observations']
        after = [next((s['cut_length_m'] for s in obs if s['tick']>t),None) for t in row['closures']]
        row['structural_prefix_after_closures_m'] = after
        if case in base.POSITIVES:
            row['checks']['first_two_bites_at_registered_prefixes'] = (
                len(after)>=2 and after[0]==.003 and after[1] in (.006,.007))
        row['passed'] = all(row['checks'].values())
        rows.append(row)
    d = out/('positive_gate.json' if positive_gate else 'decision.json')
    ps = all(r['passed'] for r in rows[:2])
    write(d,dict(predictions_sha256=digest(file),registration=reg,report_sha256=reports,cases=rows,
                 passed=ps if positive_gate else all(r['passed'] for r in rows),
                 positive_cases_passed=ps,disturbance_passes=sum(r['passed'] for r in rows[2:]),
                 replaces_historical_acceptance=False))
    for r in rows:
        print(r['case'],r['passed'],r.get('state'),r.get('reason'),r.get('structural_prefix_after_closures_m'),
              r.get('structure'),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','score'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path)
    p.add_argument('--positives',action='store_true')
    a = p.parse_args()
    file = a.out/('positive_predictions.json' if a.positives else 'predictions.json')
    if a.stage=='predict':
        if a.frames is None:
            p.error('--frames required for predict')
        if a.positives:
            # Set the destination before writing; never rename or overwrite historical files.
            def save(path,obj):
                write(file,obj)
            bind(base.predict,SPEC=SPEC,registration=registration,measure=measure,write=save)(a.out,a.frames)
        else:
            predict(a.out,a.frames)
    else:
        score(a.out,file,a.positives)


if __name__=='__main__':
    main()
