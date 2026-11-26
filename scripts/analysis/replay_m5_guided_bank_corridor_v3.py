"""Seal the 2D shore diagnostic before joining independent cut labels."""
import argparse
from pathlib import Path

import numpy as np

from replay_cell_v2 import DATA, ROOT, depths, digest, load, write
from m5_guided_bank_corridor_v3 import measure, recover_open

SPEC = ROOT/'config/task/m5_guided_bank_corridor_v3.json'
OLD = DATA/'m5_guided_bank_corridor_v2'
PAIR = DATA/'m5_guided_reverse_capture_v1'
SOURCES = (SPEC, Path(__file__).resolve(),
           *(ROOT/'scripts/analysis'/f for f in ('replay_cell_v2.py','cell_v2_controls.py')),
           *(ROOT/'scripts/vision'/f for f in ('m5_guided_bank_corridor_v3.py',
               'm5_guided_bank_corridor_v2.py','m5_guided_bank_corridor_v1.py','cell_certificate_v1.py',
               'ruler_readout.py')))


def controls():
    cases = ('attached','detached','depth_jump','wrong_side','unknown_foreground',
             'invalid_depth','core_island','central_bridge')
    out = []
    for case in cases:
        z = np.full((17,32),.23,np.float32)
        bank,cap,face = (np.full(z.shape,-1,np.int8) for _ in range(3))
        bank[:,:7],bank[:,25:] = 6,19
        z[:,:7] = z[:,25:] = .13
        peak = 14 if case=='core_island' else 8
        cap[8,peak],z[8,peak] = 14,.13
        if case!='detached':
            bank[7,7:peak+1],z[7,7:peak+1] = 7,.13
        if case=='depth_jump':
            z[8,peak] = .121
        if case=='wrong_side':
            cap[8,peak] = 15
        if case=='unknown_foreground':
            cap[8,peak] = -1
        if case=='invalid_depth':
            z[8,peak] = np.nan
        if case=='central_bridge':
            cap[8,peak],face[8,peak] = -1,12
        line = np.column_stack((np.full(18,8),np.arange(7,25)))
        for sigma,depth in depths(z):
            row = dict(row=6,status='UNKNOWN',flank_depth_m=[.13,.13])
            good = np.isfinite(depth)
            proof = (np.array([8,6]), np.array([8,25]), line, .13)
            revised, samples = recover_open(row,proof,bank,cap,face,depth,good,np.zeros(2,int),sigma)
            passed = (revised['status']=='OPEN') == (case=='attached')
            if case=='attached':
                passed &= (len(samples[2])==len(line)-1 and not np.any((samples[2]==[8,8]).all(axis=1))
                           and len(revised['shore_paths'][0]['uv'])==4)
            out.append(dict(case=case,sigma=sigma,status=revised['status'],passed=bool(passed)))
    return out


def registration():
    plan = load(SPEC)
    if any(digest(ROOT/p)!=sha for p,sha in plan['prediction_pins'].items()):
        raise ValueError('archived optical manifest changed')
    images = load(OLD/'registration.json')['images'].copy()
    for case,record in load(PAIR/'predictions.json')['cases'].items():
        for phase,f in (('before','045_certificate.npz'),('after','047_certificate.npz')):
            images[f'reverse/{case}/{f}'] = record['optics'][phase]['sha256']
    if len(images)!=21:
        raise ValueError('unexpected optical coverage')
    return dict(protocol=plan,sources={p.relative_to(ROOT).as_posix():digest(p) for p in SOURCES},images=images)


def predict(out, frames):
    reg = registration()
    out.mkdir(exist_ok=False)
    write(out/'registration.json',reg)
    tests = controls()
    write(out/'synthetic.json',tests)
    if not all(t['passed'] for t in tests):
        raise ValueError('synthetic gate failed; no exposure evaluated')
    optical = {}
    for key,sha in reg['images'].items():
        parts = key.split('/')
        if parts[0]=='guided':
            file = frames/'m5_guided_entry_clearance_v1_nominal'/parts[-1]
        elif parts[0]=='reverse':
            file = frames/('m5_guided_reverse_capture_v1_'+parts[1])/parts[-1]
        else:
            file = DATA/key
        if digest(file)!=sha:
            raise ValueError('archived exposure changed')
        if sha in optical:
            continue
        guided = parts[0] in ('guided','reverse')
        with np.load(file,allow_pickle=False) as f:
            z = f['depth']
            if z.shape!=(1920,2560) or (guided and abs(f['hfov_rad'].item()-np.deg2rad(20))>1e-6):
                raise ValueError('unexpected sensor optics')
            im,far = f['rgb'],f['far'].item() if guided else .3
            optical[sha] = [dict(sigma=sigma,**measure(im,d,far=far,sigma=sigma)) for sigma,d in depths(z)]
    write(out/'predictions.json',dict(registration_sha256=digest(out/'registration.json'),results=optical))


def score(out):
    reg,pred = (load(out/f) for f in ('registration.json','predictions.json'))
    plan = reg['protocol']
    if (reg!=registration() or pred['registration_sha256']!=digest(out/'registration.json')
            or set(reg['images'].values())!=set(pred['results'])
            or any(digest(ROOT/p)!=sha for p,sha in plan['scoring_pins'].items())):
        raise ValueError('unsealed replay or structure label drift')
    rows,seen = [],set()

    def join(key, phase, sigma, mm, required):
        if (key,sigma) in seen:
            raise ValueError('duplicate structural label')
        seen.add((key,sigma))
        result, = (r for r in pred['results'][reg['images'][key]] if r['sigma']==sigma)
        reported = result['verdict'],result['prefix_mm']
        correct = ('BELOW',mm) if 0<mm<9 else ('TARGET',mm) if mm in (9,10) else ('UNKNOWN',None)
        expected = correct if required and phase in ('cleared','after') else ('UNKNOWN',None) if required else None
        rows.append(dict(path=key,phase=phase,sigma=sigma,actual_mm=mm,verdict=reported[0],
                         prefix_mm=reported[1],reason=result['reason'],required=expected,
                         meets_gate=expected is None or reported==expected,
                         wrong_known=reported[0]!='UNKNOWN' and reported!=correct,
                         shore_rows=[r['row'] for r in result['rows'] if r.get('shore_samples')]))

    for old in load(OLD/'decision.json')['rows']:
        k,phase,sigma,mm = (old[x] for x in ('path','phase','sigma','actual_mm'))
        join(k,phase,sigma,mm,k.startswith('guided/'))
    for old in load(PAIR/'decision.json')['rows']:
        if not (old['mechanics_verified'] and old['structure_unchanged']):
            raise ValueError('unmatched paired cut labels')
        for phase,f in (('before','045_certificate.npz'),('after','047_certificate.npz')):
            for sigma in (0.,.0005):
                join(f"reverse/{old['case']}/{f}",phase,sigma,old['actual_mm'],True)
    if seen!={(f,sigma) for f in reg['images'] for sigma in (0.,.0005)}:
        raise ValueError('incomplete optical/structure join')
    info = dict(predictions_sha256=digest(out/'predictions.json'),records=len(reg['images']),
                independent_image_digests=len(pred['results']),variants=len(rows),
                wrong_known=sum(r['wrong_known'] for r in rows),
                development_gate=all(r['meets_gate'] and not r['wrong_known'] for r in rows),
                ten_mm_guided_tested=False,independent_optical_confirmation=False,
                authorizes_motion=False,rows=rows)
    write(out/'decision.json',info)
    print({k:v for k,v in info.items() if k!='rows'},flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','score'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path)
    a = p.parse_args()
    if a.stage=='predict':
        if a.frames is None:
            p.error('--frames required')
        predict(a.out,a.frames)
    else:
        score(a.out)


if __name__=='__main__':
    main()
