"""Grade the development screen without making the reports eligible for formal scoring."""
import argparse
from pathlib import Path

import score_m5_guided_benchmark_v2_4 as gate
from replay_cell_v2 import ROOT, digest, load, write

PLAN = ROOT/'config/task/m5_guided_view_preflight_v2_5.json'


def audit(out, frames):
    plan = load(PLAN)
    views = {}
    for view, pose in plan['views'].items():
        folder = out/view
        reg_file = folder/'registration.json'
        reg = load(reg_file)
        if (reg['policy']!=plan['id']+'/'+view
                or any(digest(ROOT/k)!=h for k,h in reg['sources'].items())):
            raise ValueError('invalid preflight registration: '+view)
        identity = dict(registration_sha256=digest(reg_file), scorer_sha256=digest(Path(__file__)))
        registration = lambda directory: identity
        gate.bind(gate.predict, registration=registration)(folder, frames/view)
        pred_path = folder/'predictions.json'
        pred = load(pred_path)
        cases = {}
        for case in plan['cases']:
            report, path = gate.base.source(folder, case, identity)
            p = pred['cases'][case]
            if (not p or p['report_sha256']!=digest(path) or report['initial_view']!=pose
                    or report['error'] or report['cleanup_error']):
                raise ValueError('invalid preflight report: '+view+'/'+case)
            events = report['events']
            closures = [e for e in events if e['event']=='closure']
            precision = [e for e in events if e['event']=='guided_v22_precision']
            lateral = closures[0]['lateral_truth_m'] if closures else None
            fallback = [e for e in events if e['event']=='guided_v21_active_set']
            alignment = [e for e in events if e['event']=='predicted_view' and e.get('phase')=='align']
            probes = [e for e in events if e['event']=='v25_dev_first_post_advance']
            if closures:
                terminal = next((e for e in reversed(probes) if e['tick']<=closures[-1]['tick']), None)
            else:
                terminal = None
            terminal_current = bool(terminal and terminal['tick']==closures[-1]['tick']
                                    and terminal['scissors_valid'] and terminal['chart_valid']
                                    and terminal['fix_valid'] and terminal['fix_age']==0)
            row = gate.grade(folder, case, report, p)
            row.update(precision=precision, orientation_solves=fallback, predicted_alignment=alignment,
                       closure_evidence=closures, post_advance_probes=probes, optical_replay=p['optics'],
                       first_closure_lateral_m=lateral, final_closure_first_frame_current=terminal_current,
                       qualifies=bool(row['passed'] and terminal_current and lateral is not None
                                      and lateral<plan['first_closure_lateral_limit_m']),
                       report_sha256=digest(path))
            cases[case] = row
        views[view] = dict(pose=pose, cases=cases, predictions_sha256=digest(pred_path),
                           qualifies=all(row['qualifies'] for row in cases.values()))
    chosen = next((v['pose'] for v in views.values() if v['qualifies']), None)
    write(out/'decision.json', dict(plan_sha256=digest(PLAN), development_only=True,
                                   selected_view=chosen, views=views))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    args = parser.parse_args()
    audit(args.out, args.frames)
