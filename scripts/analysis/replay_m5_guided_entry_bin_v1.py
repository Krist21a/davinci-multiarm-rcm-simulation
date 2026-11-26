"""Compare the isolated entry-index candidate on hash-matched working RGB-D."""
import argparse
from pathlib import Path

import numpy as np

from audit_m5_guided_short_gap_attribution_v1 import PRIOR, cp, snapshot
from m5_guided_entry_bin_v1 import EntryBinPerception
from replay_cell_v2 import ROOT, digest, load, write

ID = 'm5_guided_entry_bin_v1'
SPEC = ROOT/f'config/task/{ID}.json'
REF = ROOT/'validation/task/m5_guided_short_gap_attribution_v1'


def run(reader,snap):
    est = object.__new__(reader)
    est.__dict__.update(snap['state'])
    original = est._gap_scan
    scans = []

    def observed(*args):
        r = original(*args)
        scans.append(dict(start_bin=r['b_start'],occupied=np.flatnonzero(r['occ']).tolist(),
                          run_len_m=r['run_len'],slit_pixels=int(r['slit_sel'].sum())))
        return r

    est._gap_scan = observed
    r = est.estimate(snap['rgb'],snap['depth'],snap['R'],snap['t'],post_grasp=snap['post_grasp'])
    cut,tool = r['cut'],r['scissors']
    chart = cut.get('material_chart') or {}
    compact = dict(front_valid=bool(cut.get('front_valid')),reason=cut.get('reason'),mode=cut.get('mode'),
                   slit_px=cut.get('slit_px'),progress_m=cut.get('progress_m'),chart_valid=bool(chart.get('valid')),
                   chart_reason=chart.get('reason'),scissors_valid=bool(tool.get('valid')),
                   scissors_reason=tool.get('reason'))
    return r,compact,scans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True)
    a = p.parse_args()
    plan,ref = load(SPEC),load(REF/'registration.json')
    if any(digest(ROOT/k)!=v for k,v in plan['pins'].items()):
        raise ValueError('unregistered parent source')
    data = a.frames/(PRIOR+'_slip_regrasp')
    ticks = ref['plan']['ticks']
    paths = {t:data/f'{t:03d}_estimator_input.npz' for t in ticks}
    if any(digest(f)!=ref['frames'][str(t)] for t,f in paths.items()):
        raise ValueError('unsealed working frames')
    files = (SPEC,Path(__file__),ROOT/'scripts/vision/m5_guided_entry_bin_v1.py',
             ROOT/'scripts/tests/test_m5_guided_entry_bin_v1.py',
             ROOT/'scripts/analysis/audit_m5_guided_short_gap_attribution_v1.py')
    a.out.mkdir(exist_ok=False)
    registration = a.out/'registration.json'
    write(registration,dict(plan=plan,frames=ref['frames'],
                            sources={f.relative_to(ROOT).as_posix():digest(f) for f in files}))
    expected = {r['tick']:r['estimate'] for r in load(REF/'diagnosis.json')['native_frames']}
    boundary = {r['tick']:r for r in load(REF/'bin_boundary.json')['rows']}
    rows = []
    for t,f in paths.items():
        snap = snapshot(f)
        old,legacy,s0 = run(cp.CutPerception,snap)
        if legacy!=expected[t] or len(s0)!=1 or (s0[0]['start_bin'],s0[0]['occupied'])!=(
                boundary[t]['native_first_bin'],boundary[t]['occupied_bins']):
            raise ValueError('legacy replay does not match sealed source')
        new,candidate,s1 = run(EntryBinPerception,snap)
        equal = old==new
        occupancy = len(s1)==1 and s1[0]['occupied']==s0[0]['occupied']
        corrected = bool(candidate['front_valid'] and candidate['chart_valid'] and candidate['scissors_valid']
                         and candidate['mode']=='pre_cut' and candidate['progress_m']==0.
                         and candidate['slit_px']==0 and s1 and s1[0]['start_bin']==4)
        passes = bool(occupancy and corrected and (equal if legacy['front_valid'] else t in (31,32)))
        rows.append(dict(tick=t,legacy=legacy,candidate=candidate,legacy_scan=s0,candidate_scan=s1,
                         full_estimate_unchanged=equal,occupancy_unchanged=occupancy,passed=passes))
    report = dict(registration_sha256=digest(registration),archived_replay_passed=all(r['passed'] for r in rows),
                  rows=rows,synthetic_evidence='separate unittest and regression reports',
                  authorizes_motion=False,disturbance_acceptance_changed=False)
    write(a.out/'decision.json',report)
    print('Archived replay:',report['archived_replay_passed'],
          [(r['tick'],r['legacy']['front_valid'],r['candidate']['front_valid']) for r in rows])


if __name__=='__main__':
    main()
