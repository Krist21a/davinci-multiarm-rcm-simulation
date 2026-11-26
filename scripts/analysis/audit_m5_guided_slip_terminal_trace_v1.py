"""Replay frozen terminal working images and attribute the slit rejection."""
import argparse
import sys
from pathlib import Path

import numpy as np

from audit_m5_guided_short_gap_attribution_v1 import cp, snapshot
from m5_guided_entry_bin_v1 import EntryBinPerception
from replay_cell_v2 import ROOT, digest, load, write
from score_m5_guided_disturbance_dry_diagnostic_v1 import check_window

ID = 'm5_guided_slip_terminal_trace_v1'
PRIOR = 'm5_guided_entry_bin_slip_live_v1'
REF = ROOT/'validation/task'/PRIOR
SPEC = ROOT/f'config/task/{ID}.json'


def verify(folder,name,sha):
    f = folder/name
    if f.name!=name or digest(f)!=sha:
        raise ValueError(f'archive mismatch: {f}')
    return f


def matching_run(out,frames):
    p = load(SPEC)
    lock = out/'registration.json'
    reg = load(lock)
    prior = load(REF/'slip_regrasp.json')
    r = load(out/'slip_regrasp.json')
    if (reg['policy']!=ID or digest(lock)!=r['registration_sha256']
            or any(digest(ROOT/k)!=v for k,v in (reg['sources']|p['pins']).items())
            or r['error'] or r['cleanup_error']):
        raise ValueError('unregistered or incomplete replay')
    keys = ('case','scenario','config','log','observations','transitions','ticks','final_state',
            'reason','actual_closed','actual_ready','structure_file','structure_sha256',
            'attempts_camera','strict_truth','truth_purposes','controller_sha256')
    clean = [e for e in r['events'] if not e['event'].startswith('terminal_trace_')]
    if any(r[k]!=prior[k] for k in keys) or clean!=prior['events']:
        raise ValueError('original trajectory did not match')
    for folder in (out,REF):
        verify(folder,r['structure_file'],r['structure_sha256'])
    for e in clean:
        if e['event'] in ('retained_coded_exposure','guided_prefix_review'):
            for run in (ID,PRIOR):
                verify(frames/run,e['file'],e['frame_sha256'])
    ests,fixes = ([e for e in r['events'] if e['event']=='terminal_trace_'+name] for name in ('estimate','fix'))
    if any([e['tick'] for e in group]!=p['ticks'] for group in (ests,fixes)):
        raise ValueError('checkpoint schedule changed')
    return ests,fixes,r['registration_sha256']


def inspect(estimator_input,expected):
    snap = snapshot(estimator_input)
    estimator = object.__new__(EntryBinPerception)
    estimator.__dict__.update(snap['state'])
    original_gap,original_prove = estimator._gap_scan,cp._provable_ahead
    scans,windows,cut_details = [],[],{}

    def gap(mask,sa,sb,pz,depth,lat,near,far):
        r = original_gap(mask,sa,sb,pz,depth,lat,near,far)
        scans.append(dict(lateral_mm=lat*1000,run_end_mm=None if r['run_end'] is None else r['run_end']*1000,
                          run_mm=r['run_len']*1000,slit_pixels=int(r['slit_sel'].sum()),reach_far=r['reach_far'],
                          recovery=r['recovery'],occupied_bins=np.flatnonzero(r['occ']).tolist()))
        return r

    def prove(*args):
        passed = original_prove(*args)
        row = check_window(*args)
        if passed!=all(row['checks'].values()):
            raise ValueError('forward-window audit disagrees with native predicate')
        windows.append(dict(**row,site=sys._getframe(1).f_code.co_name,passed=bool(passed)))
        return passed

    def local(frame,event,_value):
        if event=='return':
            values = frame.f_locals
            for key in ('cands','ok'):
                cut_details[key] = [dict(source=src,along_m=float(s),lateral_m=float(b))
                                    for src,s,b in values.get(key,[])]
            cut_details['distal_m'] = [float(x) for x in values.get('distal',[])]
            cut_details['main_slit_measured'] = bool(values.get('slit_meas'))
        return local

    def enter(frame,event,_value):
        if event=='call' and frame.f_code is cp.CutPerception._cut.__code__:
            frame.f_trace_lines = False
            return local
        return None

    prev = sys.gettrace()
    try:
        estimator._gap_scan,cp._provable_ahead = gap,prove
        sys.settrace(enter)
        result = estimator.estimate(snap['rgb'],snap['depth'],snap['R'],snap['t'],post_grasp=snap['post_grasp'])
    finally:
        sys.settrace(prev)
        cp._provable_ahead = original_prove
    if result!=expected or not cut_details:
        raise ValueError('offline estimator does not match recorded estimation')
    cut = result['cut']
    return dict(camera_shape=list(snap['depth'].shape),cut_reason=cut.get('reason'),cut_mode=cut.get('mode'),
                cutoff_front_valid=cut.get('front_valid'),progress_m=cut.get('progress_m'),
                chart_valid=(cut.get('material_chart') or {}).get('valid'),
                chart_reason=(cut.get('material_chart') or {}).get('reason'),
                tool_valid=result['scissors'].get('valid'),scans=scans,windows=windows,**cut_details)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True)
    a = p.parse_args()
    est,fix,sha = matching_run(a.out,a.frames)
    rows = [dict(tick=e['tick'],frame_sha256=e['sha256'],awaiting_before=f['awaiting_before'],
                 awaiting_after=f['awaiting_after'],
                 **inspect(verify(a.frames/ID,e['file'],e['sha256']),e['estimate']))
            for e,f in zip(est,fix,strict=True)]
    write(a.out/'diagnosis.json',dict(registration_sha256=sha,scorer_sha256=digest(Path(__file__)),
                                     original_run_matched=True,rows=rows,authorizes_motion=False))
    print(rows[-1])


if __name__=='__main__':
    main()
