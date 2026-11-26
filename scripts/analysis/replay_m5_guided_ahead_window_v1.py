"""Gate the ahead-window candidate on the matched slip terminal snapshots."""
import argparse
from pathlib import Path

import audit_m5_guided_slip_terminal_trace_v1 as trace
from m5_guided_ahead_window_v1 import AheadWindowPerception
from replay_cell_v2 import ROOT, digest, load, write

ID = 'm5_guided_ahead_window_v1'
REF = ROOT/'validation/task'/trace.ID
SPEC = ROOT/f'config/task/{ID}.json'


def replay(reader,snap):
    estimator = object.__new__(reader)
    estimator.__dict__.update(snap['state'])
    return estimator.estimate(snap['rgb'],snap['depth'],snap['R'],snap['t'],post_grasp=snap['post_grasp'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True)
    a = p.parse_args()
    plan = load(SPEC)
    if any(digest(ROOT/k)!=v for k,v in plan['pins'].items()):
        raise ValueError('unregistered parent evidence')
    events,_,sha = trace.matching_run(REF,a.frames)
    paths = {e['tick']:trace.verify(a.frames/trace.ID,e['file'],e['sha256']) for e in events}
    if set(paths)!={28,29,30}:
        raise ValueError('unexpected working frame ticks')
    files = (SPEC,Path(__file__),Path(trace.__file__),ROOT/'scripts/vision/m5_guided_ahead_window_v1.py',
             ROOT/'scripts/tests/test_m5_guided_ahead_window_v1.py')
    a.out.mkdir(exist_ok=False)
    lock = a.out/'registration.json'
    write(lock,dict(plan=plan,source_run_registration_sha256=sha,
                    sources={f.relative_to(ROOT).as_posix():digest(f) for f in files},
                    frames={str(t):digest(f) for t,f in paths.items()}))
    rows = []
    for event in events:
        tick = event['tick']
        snap = trace.snapshot(paths[tick])
        old = replay(trace.EntryBinPerception,snap)
        if old!=event['estimate']:
            raise ValueError(f'original frame {tick} did not reproduce')
        new = replay(AheadWindowPerception,snap)
        cut = new['cut']
        progress,cell = cut.get('progress_m'),cut.get('progress_cell_m')
        chart = cut.get('material_chart') or {}
        direct = bool(cut.get('front_valid') and cut.get('mode') in ('slit','bank_slit')
                      and chart.get('valid') and new['scissors'].get('valid')
                      and progress is not None and cell is not None and cell[0]<=progress<=cell[1])
        rows.append(dict(tick=tick,full_estimate_unchanged=old==new,legacy=old,candidate=new,
                         valid_direct_chart_front=direct,passed=(old==new if tick<30 else direct)))
    physical = load(trace.REF/'decision.json')['structure']
    final = rows[-1]['candidate']['cut']
    length = final.get('progress_m')
    result = dict(registration_sha256=digest(lock),archived_replay_passed=all(r['passed'] for r in rows),
                  frames=rows,terminal_working_chart_progress_m=length,
                  terminal_working_chart_cell_m=final.get('progress_cell_m'),
                  offline_structure=physical,
                  working_chart_minus_structural_prefix_m=None if length is None else length-.001*physical['prefix_rows'],
                  authorizes_motion=False,changes_prior_acceptance=False)
    write(a.out/'decision.json',result)
    print([(r['tick'],r['full_estimate_unchanged'],r['valid_direct_chart_front'],r['passed']) for r in rows])
    print('Working chart versus structure:',length,physical)


if __name__=='__main__':
    main()
