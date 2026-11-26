"""Compare v3/v4 on every sealed guided certificate before joining labels."""
import argparse
from pathlib import Path

import numpy as np

from replay_cell_v2 import ROOT, digest, load, write
from m5_guided_bank_corridor_v3 import measure as v3
from m5_guided_bank_corridor_v4 import measure as v4

ID = 'm5_guided_bank_corridor_v4'
SPEC = ROOT/f'config/task/{ID}.json'


def collect(frames):
    records = []
    for archive,pin in load(SPEC)['archives'].items():
        folder = ROOT/'validation/task'/archive
        if digest(folder/'registration.json')!=pin:
            raise ValueError(f'changed registration: {archive}')
        runs = 0
        for p in sorted(folder.glob('*.json')):
            r = load(p)
            if 'events' not in r or 'observations' not in r:
                continue
            if r['registration_sha256']!=pin:
                raise ValueError(f'unregistered report: {p}')
            runs += 1
            case = p.stem
            d = (f'{archive}_{case}' if archive in ('m5_guided_benchmark_v2',
                 'm5_guided_benchmark_v2_disturbances_v1') else
                 f'{archive}/{case}' if archive.endswith('v2_1') else archive)
            for e in r['events']:
                if e['event']!='guided_prefix_review':
                    continue
                f = frames/d/e['file']
                if f.name!=e['file'] or digest(f)!=e['frame_sha256']:
                    raise ValueError(f'changed frame: {f}')
                records.append(dict(report=p.relative_to(ROOT).as_posix(),report_sha256=digest(p),
                                    frame=f.relative_to(frames).as_posix(),frame_sha256=digest(f),tick=e['tick'],
                                    sigma=e['noise_sigma_m'],seed=r['config']['perception']['seed']))
        if runs not in (1,4,6):
            raise ValueError(f'incomplete archive: {archive}')
    if len(records)!=16:
        raise ValueError(f'incomplete certificate corpus: {len(records)}')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    rows = collect(a.frames)
    a.out.mkdir(exist_ok=False)
    sources = (SPEC,Path(__file__),*(ROOT/'scripts/vision'/n for n in (
        'm5_guided_bank_corridor_v1.py','m5_guided_bank_corridor_v2.py','m5_guided_bank_corridor_v3.py',
        'm5_guided_bank_corridor_v4.py','cell_certificate_v1.py','ruler_readout.py')))
    lock = a.out/'registration.json'
    write(lock,dict(sources={p.relative_to(ROOT).as_posix():digest(p) for p in sources},frames=rows))
    predictions = []
    for r in rows:
        with np.load(a.frames/r['frame'],allow_pickle=False) as f:
            z,sigma = f['depth'],r['sigma']
            if sigma:
                z = z+np.random.default_rng(np.random.SeedSequence([r['seed'],r['tick'],513])).normal(0,sigma,z.shape)
            if not np.array_equal(z,f['measured']):
                raise ValueError('measured noise identity changed')
            forecasts = [reader(f['rgb'],f['measured'],far=f['far'].item(),sigma=sigma) for reader in (v3,v4)]
        predictions.append(dict(r,readers=[{k:f[k] for k in ('verdict','prefix_mm','reason','rows')} for f in forecasts]))
    pred = a.out/'predictions.json'
    write(pred,dict(registration_sha256=digest(lock),frames=predictions))
    checks = []
    for r in predictions:
        old,new = [(e['verdict'],e['prefix_mm']) for e in r['readers']]
        src = ROOT/r['report']
        if digest(src)!=r['report_sha256']:
            raise ValueError('changed structural label source')
        label = next(o['cut_length_m'] for o in load(src)['observations'] if o['tick']==r['tick'])
        allowed = (new[0]=='UNKNOWN' or (new[1]*.001==label and 1<=new[1]<=10
                   and new[0]==('BELOW' if new[1]<9 else 'TARGET')))
        ok = (old[0]=='UNKNOWN' or new==old) and allowed
        checks.append(dict(frame=r['frame'],label_m=label,v3=list(old),v4=list(new),passed=ok))
        print(r['frame'],old,'->',new,label,ok)
    passed = all(c['passed'] for c in checks)
    write(a.out/'decision.json',dict(registration_sha256=digest(lock),predictions_sha256=digest(pred),
                                     passed=passed,frames=checks,authorizes_motion=False))
    return int(not passed)


if __name__=='__main__':
    raise SystemExit(main())
