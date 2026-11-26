"""Seal pre/post reverse RGB-D readings, then join the independent constraints."""
import argparse
from pathlib import Path

import numpy as np

from replay_cell_v2 import ROOT, depths, digest, load, material_state, write
from m5_guided_bank_corridor_v2 import measure


SPEC = ROOT/'config/task/m5_guided_reverse_capture_v1.json'
PARENT = ROOT/'validation/task/m5_guided_terminal_controls_v1'
READER = tuple(ROOT/'scripts/vision'/name for name in ('m5_guided_bank_corridor_v2.py',
               'm5_guided_bank_corridor_v1.py','cell_certificate_v1.py','ruler_readout.py'))
SOURCES = (Path(__file__).resolve(),*READER,
           ROOT/'scripts/analysis/cell_v2_controls.py',ROOT/'scripts/task/seam_topology.py')


def events(r, name):
    return [e for e in r.get('events',[]) if e['event']==name]


def registration(folder):
    p = folder/'registration.json'
    plan = load(SPEC)
    prior = load(ROOT/'config/task/m5_guided_terminal_controls_v1.json')
    if (load(p)['policy']!=plan['id']
            or any(digest(ROOT/k)!=v for k,v in
                   (plan['pins']|prior['pins']|load(p)['sources']).items())):
        raise ValueError('registered acquisition or reader chain changed')
    return dict(policy_sha256=digest(SPEC),registration_sha256=digest(p),
                sources={f.relative_to(ROOT).as_posix():digest(f) for f in SOURCES})


def predict(folder, frames):
    predictions = {}
    reg = registration(folder)
    for case in load(SPEC)['cases']:
        path = folder/(case+'.json')
        if not path.exists():
            predictions[case] = dict(report_sha256=None,optics={})
            continue
        r = load(path)
        if r['registration_sha256']!=reg['registration_sha256'] or r['case']!=case:
            raise ValueError('unregistered case')
        capture = events(r,'diagnostic_final_capture')
        result = dict(report_sha256=digest(path),optics={})
        predictions[case] = result
        if not capture:
            continue
        capture, = capture
        for kind,key,sha_key in (('before','pre_file','pre_sha256'),('after','file','sha256')):
            f = frames/('m5_guided_reverse_capture_v1_'+case)/capture[key]
            if capture[key]!=f.name or digest(f)!=capture[sha_key]:
                raise ValueError('sensor pin changed')
            with np.load(f,allow_pickle=False) as x:
                z = x['depth']
                if z.shape!=(1920,2560) or abs(x['hfov_rad'].item()-np.deg2rad(20))>1e-6:
                    raise ValueError('unexpected sensor optics')
                rgb,far = x['rgb'],x['far'].item()
                result['optics'][kind] = dict(sha256=capture[sha_key],variants=[
                    dict(sigma=s,**measure(rgb,d,far=far,sigma=s)) for s,d in depths(z)])
                pose = [x[k] for k in ('cam_R','cam_t','hfov_rad','focal','far')]
            if kind=='before':
                original = pose
            elif not all(np.array_equal(a,b) for a,b in zip(pose,original,strict=True)):
                raise ValueError('before and after camera poses differ')
    write(folder/'predictions.json',dict(registration=reg,cases=predictions))


def score(folder):
    prediction_file = folder/'predictions.json'
    p = load(prediction_file)
    if p['registration']!=registration(folder) or set(p['cases'])!=set(load(SPEC)['cases']):
        raise ValueError('unsealed prediction')
    rows = []
    for case,record in p['cases'].items():
        f = folder/(case+'.json')
        if record['report_sha256'] is None:
            rows.append(dict(case=case,captured=False,qualified=False,reason='missing run'))
            continue
        if digest(f)!=record['report_sha256']:
            raise ValueError('raw report changed')
        r,baseline = load(f),load(PARENT/(case+'.json'))
        capture,moves,waypoints,closures = (events(r,n) for n in (
            'diagnostic_final_capture','closed_reverse_segment','reverse_waypoint_recorded','closure'))
        last = r['observations'][-1] if r['observations'] else {}
        row = dict(case=case,captured=bool(capture),qualified=False,reason=r.get('error') or r.get('reason'),
                   last_m=last.get('cut_length_m'),closed=r.get('actual_closed'),ready=r.get('actual_ready'),
                   closures=[e['tick'] for e in closures],reverse_segments=moves)
        rows.append(row)
        if not capture:
            continue
        event, = capture
        structure = folder/(case+'_structure.npz')
        if structure.name!=event['structure_file'] or digest(structure)!=event['structure_sha256']:
            raise ValueError('structure pin changed')
        with np.load(structure,allow_pickle=False) as f:
            before,after = f['before_edges'],f['after_edges']
        same = np.array_equal(before,after)
        s = material_state(after)
        n = s['prefix_rows']
        expected_mm = 8 if case=='short' else 9
        native, = events(r,'diagnostic_final_advance_request')
        original_request, = events(baseline,'diagnostic_final_advance_request')
        specs = ('tick','requested_m','fixed_cap_m')
        arms = events(r,'diagnostic_final_bite_armed')
        reviews = [v for v in events(r,'guided_prefix_review') if v['tick']>native['tick']]
        guard = bool(not r['error'] and not r['cleanup_error'] and r['actual_closed'] and not r['actual_ready']
                     and r['run_id']!=baseline['run_id'] and n==expected_mm
                     and same and s['farthest_row_end']<=10 and abs(last['cut_length_m']-n*.001)<=1e-12
                     and all(not v['faults'] and not v['torn'] and v['cut_length_m']<=.010+1e-12
                             for v in r['observations'])
                     and len(arms)==1 and arms[0]['matched']
                     and all(native[k]==original_request[k] for k in specs)
                     and len(closures)==3 and all(e['direct_front'] and e['scissors_valid']
                                                 for e in closures)
                     and [e['tick'] for e in closures]==[e['tick'] for e in events(baseline,'closure')]
                     and len(reviews)==2 and all(e['physically_closed'] for e in reviews)
                     and 1<=len(moves)<=4 and len(waypoints)==len(moves)+1
                     and all(e['accepted'] and np.allclose(e['q'],w['q'],atol=1e-6,rtol=0)
                             for e,w in zip(moves,reversed(waypoints[:-1]),strict=True)))
        optics = record['optics']
        if optics['before']['sha256']!=event['pre_sha256'] or optics['after']['sha256']!=event['sha256']:
            raise ValueError('optical join changed')
        target = 'BELOW' if expected_mm==8 else 'TARGET'
        summaries = {label:[dict(sigma=v['sigma'],verdict=v['verdict'],prefix_mm=v['prefix_mm'],
                                  why=v['reason']) for v in x['variants']] for label,x in optics.items()}
        qualified = guard and all((v['verdict'],v['prefix_mm'])==(target,expected_mm)
                                  for v in summaries['after'])
        row.update(qualified=qualified,mechanics_verified=guard,actual_mm=n,
                   farthest_row_end_mm=s['farthest_row_end'],structure_unchanged=bool(same),optical=summaries)
    coverage = sorted({r['actual_mm'] for r in rows if r['qualified']})
    result = dict(predictions_sha256=digest(prediction_file),qualified_coverage_mm=coverage,
                  passed=coverage==[8,9],ten_mm_tested=False,authorizes_motion=False,rows=rows)
    write(folder/'decision.json',result)
    print(result,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','score'))
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--frames',type=Path)
    a = p.parse_args()
    if a.stage=='predict':
        if a.frames is None:
            p.error('--frames required')
        predict(a.folder,a.frames)
    else:
        score(a.folder)


if __name__=='__main__':
    main()
