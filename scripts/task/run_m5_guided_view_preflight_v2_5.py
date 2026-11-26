"""Compare three registered initial ECM poses in development-only positive runs."""
import argparse
import json
from pathlib import Path

import run_m5_guided_benchmark_v2_5 as candidate
from run_cut_task import REPO, add_arguments, sha256

PLAN = REPO/'config/task/m5_guided_view_preflight_v2_5.json'
PARENT = REPO/'config/task/m5_guided_benchmark_v2_4.json'
BASELINE = REPO/'validation/task/m5_guided_benchmark_v2_4/registration.json'


def instrument(w):
    advance, observe = w.advance_scissors, w.observe
    pending = None

    def advance_with_probe(requested):
        nonlocal pending
        fix = w._fix
        before = fix['progress'] if w._fix_valid() else None
        tick = w.tick
        accepted = advance(requested)
        if accepted:
            pending = (tick, before)
        return accepted

    def observe_with_probe(**kwargs):
        nonlocal pending
        obs = observe(**kwargs)
        if pending is not None:
            tick, before = pending
            pending = None
            est = w._last_est or {}
            cut = est.get('cut') or {}
            w.event_log.append(dict(event='v25_dev_first_post_advance', tick=w.tick, advance_tick=tick,
                                    before_progress_m=before, after_progress_m=obs.cut_progress_m,
                                    scissors_valid=obs.scissors_visible, mode=cut.get('mode'),
                                    chart_valid=bool((cut.get('material_chart') or {}).get('valid')),
                                    fix_age=None if w._fix is None else w.tick-w._fix['tick'],
                                    fix_valid=w._fix_valid()))
        return obs

    w.advance_scissors, w.observe = advance_with_probe, observe_with_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--view', required=True, choices=('original','plus3','review'))
    parser.add_argument('--case', required=True, choices=candidate.ORDER[:2])
    for key in ('out','evidence-out'):
        parser.add_argument('--'+key, type=Path, required=True)
    a = parser.parse_args()
    spec = json.loads(PLAN.read_bytes())
    plan = json.loads(PARENT.read_bytes())
    baseline = json.loads(BASELINE.read_bytes())
    if baseline['policy']!='m5_guided_benchmark_v2_4' or any(
            sha256(REPO/k)!=h for k,h in baseline['sources'].items()):
        raise ValueError('the frozen v2.4 parent differs')
    a.out, a.evidence_out = a.out/a.view, a.evidence_out/a.view
    plan['initial_view'] = spec['views'][a.view]
    files = (PLAN, BASELINE, *(REPO/k for k in baseline['sources']),
             Path(__file__).resolve(), Path(candidate.__file__).resolve(),
             REPO/'scripts/task/m5_guided_controller_v2_5.py',
             REPO/'scripts/analysis/audit_m5_guided_view_preflight_v2_5.py')
    base = candidate.parent.register(a.out, spec['id']+'/'+a.view, files)
    reg = sha256(a.out/'registration.json')
    if a.case=='exact_10mm':
        prior = json.loads((a.out/'nominal.json').read_bytes())
        if prior['registration_sha256']!=reg or prior['error'] or prior['cleanup_error']:
            raise ValueError('prior development case did not complete under this registration')
    return candidate.run_case(a, plan, base, reg, instrument=instrument)


if __name__=='__main__':
    raise SystemExit(main())
