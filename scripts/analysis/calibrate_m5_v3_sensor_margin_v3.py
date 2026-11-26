"""Recheck the sealed FK gate without refitting or replaying optical measurements."""
import argparse
from pathlib import Path

import numpy as np

from calibrate_m5_v3_sensor_margin_v1 import ROOT, digest, load, verified, write
from m5_v3_rigid_fk_v2 import CutterFK


PLAN = ROOT/'config/task/m5_v3_sensor_margin_calibration_v3.json'
PREV = ROOT/'config/task/m5_v3_sensor_margin_calibration_v2.json'
BATCH = ROOT/'validation/task/m5_v3_sensor_margin_calibration_v2'
FK = ROOT/'validation/task/m5_v3_rigid_fk_v2.json'
GATES = {'clean_coverage','development_coverage','check_coverage','exact_tcp_fk',
         'absolute_axis','smaller_clean_envelope','held_out_envelope'}


def recheck(out):
    plan,prev = load(PLAN),load(PREV)
    sources = dict(plan['sources'])
    for name,pin in sources.items():
        verified(ROOT/name,pin)
    for key in ('fk_tcp_max_error_mm','absolute_global_axis_deg_lt','late_pair_min_fraction_each_split',
                'legacy_margin_mm'):
        if plan[key]!=prev[key]:
            raise ValueError('changed gate: '+key)
    prior,reg,run,fk = (load(p) for p in (BATCH/'decision.json',BATCH/'registration.json',BATCH/'run.json',FK))
    if (set(prior['gates'])!=GATES or reg['calibration_plan']!=prev or not fk['qualified']
            or prior['qualified'] or run['state']!='DONE' or prior['frames']!=15
            or {k for k,v in prior['gates'].items() if not v}!={'exact_tcp_fk'}):
        raise ValueError('unexpected predecessor or unqualified FK')
    for pins in (reg['sources'],fk['sources']):
        for name,pin in pins.items():
            verified(ROOT/name,pin)
            sources[name] = pin
    for path in (PLAN,Path(__file__).resolve()):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    for name,key in (('run.json','run_sha256'),('registration.json','registration_sha256'),
                     ('fit.json','fit_sha256'),('check.json','check_sha256')):
        sources[(BATCH/name).relative_to(ROOT).as_posix()] = digest(verified(BATCH/name,prior[key]))
    fit = load(BATCH/'fit.json')
    for name,key in (('clean.json','clean_sha256'),('development.json','development_sha256')):
        sources[(BATCH/name).relative_to(ROOT).as_posix()] = digest(verified(BATCH/name,fit[key]))
    margin = dict(c0_m=.001*fit['c0_mm'],k=fit['k'])
    if margin!=plan['frozen_fit'] or prior['fit_candidate']!=dict(
            c0_mm=fit['c0_mm'],k=fit['k'],at_0p5mm_mm=fit['c0_mm']+.5*fit['k']):
        raise ValueError('changed frozen fit')
    frames = {n:m for n,m in run['captures'].items() if m['fov_deg']==70}
    refs = run['scoring_sidecars']
    clean = {r['frame']:r for r in load(BATCH/'clean.json')}
    if (len(frames)!=15 or set(frames)!=set(refs) or set(clean)!={m['file'] for m in frames.values()}
            or run['truth_reads_by_purpose']['scoring']['tcp']!=15 or 'untagged' in run['truth_reads_by_purpose']):
        raise ValueError('changed scoring coverage')
    for r in [*frames.values(),*refs.values()]:
        p = verified(BATCH/r['file'],r['sha256'])
        sources[p.relative_to(ROOT).as_posix()] = r['sha256']
    out.mkdir(exist_ok=False)
    seal = out/'registration.json'
    write(seal,dict(plan=plan,sources=sources))

    model = CutterFK(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in fk['calibration'].items()})
    rows = []
    stamps = []
    for name,m in sorted(frames.items(),key=lambda kv:kv[1]['tick']):
        with np.load(BATCH/m['file'],allow_pickle=False) as frame, np.load(
                BATCH/refs[name]['file'],allow_pickle=False) as ref:
            q = frame['q']
            tcp = ref['tcp_world']
            stamp = ref['simulation_time_s'].item()
            if (ref['frame_sha256'].item()!=m['sha256'] or ref['tick'].item()!=m['tick']
                    or ref['closed'].item()!=m['closed'] or not np.array_equal(q,ref['q'])
                    or tcp.shape!=(3,) or not np.isfinite(tcp).all() or not np.isfinite(stamp)):
                raise ValueError('sidecar binding failed: '+name)
        prev_row = clean[m['file']]
        stamps.append(stamp)
        rows.append(dict(frame=m['file'],prefix_mm=prev_row['prefix_mm'],closed=m['closed'],
                         previous_fk_mm=prev_row['fk_tcp_error_mm'],
                         fk_mm=float(1000*np.linalg.norm(model.pose(q)[:,3]-tcp))))
    if stamps!=sorted(stamps) or stamps[-1]<=stamps[-2]:
        raise ValueError('nonmonotonic reference time')
    measurements = out/'measurements.json'
    write(measurements,rows)
    max_fk = max(r['fk_mm'] for r in rows)
    gates = dict(prior['gates'],exact_tcp_fk=max_fk<=plan['fk_tcp_max_error_mm'])
    passed = all(gates.values())
    result = dict(prior,qualified=passed,gates=gates,max_fk_mm=max_fk,
                  previous_max_fk_mm=prior['max_fk_mm'],sensor_margin=margin if passed else None,
                  capture_registration_sha256=prior['registration_sha256'],
                  registration_sha256=digest(seal),measurements_sha256=digest(measurements))
    write(out/'decision.json',result)
    print('qualified:',passed,'max FK mm:',max_fk,'issued margin:',result['sensor_margin'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    recheck(parser.parse_args().out)
