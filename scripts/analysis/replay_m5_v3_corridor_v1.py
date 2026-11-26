"""Seal optical predictions, then join the historical structural prefixes."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from replay_cell_v2 import DATA, ROOT, digest, load, write
from m5_guided_bank_corridor_v4 import measure as legacy
from m5_v3_corridor_v1 import measure

ID = 'm5_v3_corridor_screen_v1'
PLAN = ROOT/f'config/task/{ID}.json'
LEGACY = DATA/'m5_guided_bank_corridor_v4/registration.json'
NEW = 'm5_guided_benchmark_v2_6_dev'
SOURCES = (PLAN, Path(__file__).resolve(), LEGACY, ROOT/'scripts/analysis/replay_cell_v2.py',
           *(ROOT/'scripts/vision'/f for f in ('m5_v3_corridor_v1.py',
             'm5_guided_bank_corridor_v1.py', 'm5_guided_bank_corridor_v2.py',
             'm5_guided_bank_corridor_v4.py', 'cell_certificate_v1.py', 'ruler_readout.py')))


def checked(path, pin):
    if digest(path) != pin:
        raise ValueError('changed seal: '+str(path))


def manifest(plan):
    checked(LEGACY, plan['legacy_manifest'])
    old = load(LEGACY)
    for name, pin in old['sources'].items():
        checked(ROOT/name, pin)
    certs = {r['frame']: dict(r) for r in old['frames']}
    if len(certs) != 16:
        raise ValueError('incomplete legacy manifest')
    passive = []
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in SOURCES}
    for archive, (pin, runs, certificates, post_bite) in plan['archives'].items():
        d = DATA/archive
        reg = d/'registration.json'
        checked(reg, pin)
        sources[reg.relative_to(ROOT).as_posix()] = pin
        tally = Counter()
        for path in sorted(d.glob('*.json')):
            report = load(path)
            if 'events' not in report:
                continue
            tally['runs'] += 1
            if report['registration_sha256'] != pin or report['error'] or report['cleanup_error']:
                raise ValueError('not a sealed run: '+str(path))
            report_name, report_hash = path.relative_to(ROOT).as_posix(), digest(path)
            sources[report_name] = report_hash
            for e in report['events']:
                event = e['event']
                if event not in ('guided_prefix_review', 'retained_coded_exposure'):
                    continue
                if Path(e['file']).name != e['file']:
                    raise ValueError('unexpected frame name')
                tally[event] += 1
                r = dict(report=report_name, report_sha256=report_hash,
                         frame=f'{archive}/{path.stem}/{e["file"]}',
                         frame_sha256=e['frame_sha256'], tick=e['tick'])
                if event == 'retained_coded_exposure':
                    passive.append(r)
                    continue
                if not e['physically_closed']:
                    raise ValueError('certificate was not closed-jaw')
                r.update(sigma=e['noise_sigma_m'], seed=report['config']['perception']['seed'])
                if certs.setdefault(r['frame'], r) != r:
                    raise ValueError('conflicting certificate identity')
        if tuple(tally[k] for k in ('runs','guided_prefix_review','retained_coded_exposure')) != (
                runs, certificates, post_bite):
            raise ValueError(f'incomplete archive: {archive}: {tally}')
    return dict(protocol=plan, sources=sources, certificates=list(certs.values()),
                passive_other_print=passive)


def coded_sheet(prefix, stride=10):
    rgb = np.zeros((13*stride, 48, 3), dtype=np.uint8)
    z = np.full(rgb.shape[:2], .23, dtype=float)
    for row in range(13):
        y = slice(row*stride, row*stride+10)
        red, other = 32+48*(row%4), 35+46*(row//4)
        rgb[y, :8], rgb[y, 40:] = (red,235,other), (red,other,235)
        z[y, :8], z[y, 40:] = .13, .13
        if row >= prefix:
            rgb[y, 8:24], rgb[y, 24:40] = (248,24+14*row,147), (248,24+14*row,205)
            z[y, 8:40] = .13
    return rgb, z


def controls():
    tests = []
    for name, prefix, expected, expected_row4 in (
            ('below',6,('BELOW',6),'OPEN'),
            ('target9',9,('TARGET',9),'OPEN'),
            ('target10',10,('TARGET',10),'OPEN'),
            ('attached_cap',6,('BELOW',6),'OPEN'),
            ('detached_cap',6,('UNKNOWN',None),'UNKNOWN'),
            ('cap_depth_jump',6,('UNKNOWN',None),'UNKNOWN'),
            ('one_sided_face',6,('UNKNOWN',None),'PARTIAL'),
            ('shallow_bridge',6,('UNKNOWN',None),'INTACT'),
            ('unknown_foreground',6,('UNKNOWN',None),'UNKNOWN'),
            ('invalid_depth',6,('UNKNOWN',None),'UNKNOWN'),
            ('missing_bank',6,('UNKNOWN',None),'UNKNOWN'),
            ('thin_interrow_neck',6,('UNKNOWN',None),'OPEN'),
            ('missing_terminal_face',6,('UNKNOWN',None),'OPEN'),
            ('partial_terminal',6,('BELOW',6),'OPEN')):
        stride = 14 if name == 'thin_interrow_neck' else 10
        rgb, z = coded_sheet(prefix,stride)
        y = 4*stride+4
        if name in ('attached_cap','detached_cap','cap_depth_jump'):
            x = 39 if name != 'detached_cap' else 34
            rgb[y,x], z[y,x] = (14,24+14*4,205), .124 if name == 'cap_depth_jump' else .13
        elif name in ('one_sided_face','shallow_bridge'):
            x = 40 if name == 'shallow_bridge' else 18
            rgb[y:y+2,8:18] = (248,24+14*4,147)
            rgb[y:y+2,18:x] = (248,24+14*4,205)
            z[y:y+2,8:x] = .13
        elif name == 'unknown_foreground':
            z[y,25] = .13
        elif name == 'invalid_depth':
            z[y,25] = np.nan
        elif name == 'missing_bank':
            rgb[40:50,40:] = 0
        elif name == 'thin_interrow_neck':
            z[10:14] = np.nan
            z[10:14,22:25] = .23
        elif name == 'missing_terminal_face':
            rgb[60:70,24:40] = 0
        elif name == 'partial_terminal':
            rgb[60:70,24:40] = 0
            z[60:70,24:40] = .23
        r = measure(rgb,z,far=.3,sigma=0.)
        actual = (r['verdict'],r['prefix_mm'])
        row4 = r['rows'][4]['status']
        tests.append(dict(name=name,expected=expected,actual=actual,row4=row4,
                          expected_row4=expected_row4,reason=r['reason'],
                          passed=actual==expected and row4==expected_row4))
    return tests


def predict(out, frames):
    reg = manifest(load(PLAN))
    for r in reg['certificates']+reg['passive_other_print']:
        checked(frames/r['frame'],r['frame_sha256'])
    out.mkdir(exist_ok=False)
    write(out/'registration.json',reg)
    control = controls()
    write(out/'synthetic.json',control)
    if not all(c['passed'] for c in control):
        raise ValueError('synthetic gate failed before any real frame')
    optical = {}
    cases = []
    for r in reg['certificates']:
        key = r['frame_sha256']
        with np.load(frames/r['frame'],allow_pickle=False) as f:
            sigma = r['sigma']
            z = f['depth']
            if sigma:
                seed = np.random.SeedSequence([r['seed'],r['tick'],513])
                z = z+np.random.default_rng(seed).normal(0,sigma,z.shape)
            if (z.shape != (1920,2560) or not np.array_equal(z,f['measured'])
                    or not np.isclose(f['hfov_rad'].item(),np.deg2rad(20),rtol=0,atol=1e-6)):
                raise ValueError('unregistered measured optics: '+r['frame'])
            if key not in optical:
                kwargs = dict(far=f['far'].item(),sigma=sigma)
                a,b = (reader(f['rgb'],z,**kwargs) for reader in (legacy,measure))
                optical[key] = dict(sigma=sigma,legacy=[a['verdict'],a['prefix_mm']],
                                    candidate=b)
            if optical[key]['sigma'] != sigma:
                raise ValueError('same frame under conflicting noise settings')
        cases.append(dict(r,reader_key=key))
    write(out/'predictions.json',dict(registration_sha256=digest(out/'registration.json'),
                                     synthetic_sha256=digest(out/'synthetic.json'),
                                     cases=cases,results=optical))


def score(out):
    reg = load(out/'registration.json')
    if reg != manifest(load(PLAN)):
        raise ValueError('registration drift')
    pred_path = out/'predictions.json'
    pred = load(pred_path)
    checked(out/'registration.json',pred['registration_sha256'])
    checked(out/'synthetic.json',pred['synthetic_sha256'])
    if pred['cases'] != [dict(r,reader_key=r['frame_sha256']) for r in reg['certificates']]:
        raise ValueError('optical coverage differs from the sealed manifest')
    reports = {}
    rows = []
    for r in pred['cases']:
        name = r['report']
        if name not in reports:
            checked(ROOT/name,r['report_sha256'])
            reports[name] = load(ROOT/name)
        at, = (o for o in reports[name]['observations'] if o['tick']==r['tick'])
        mm = round(at['cut_length_m']*1000)
        if not np.isclose(at['cut_length_m'],mm*.001,rtol=0,atol=1e-12):
            raise ValueError('non-row structural label')
        model = pred['results'][r['reader_key']]
        p = model['candidate']
        actual = [p['verdict'],p['prefix_mm']]
        correct = ['BELOW' if mm<9 else 'TARGET',mm] if 0<mm<=10 else ['UNKNOWN',None]
        old = model['legacy']
        rows.append(dict(frame=r['frame'],tick=r['tick'],frame_sha256=r['frame_sha256'],
                         old=old,new=actual,label_mm=mm,reason=p['reason'],
                         states=[row['status'] for row in p['rows']],
                         wrong_known=actual[0]!='UNKNOWN' and actual!=correct,
                         lost_legacy=old[0]!='UNKNOWN' and actual!=old,
                         recovered=old[0]=='UNKNOWN' and actual==correct and actual[0]!='UNKNOWN'))
    target = [r for r in rows if r['frame'].startswith(NEW+'/')]
    recover_six = len(target)==2 and all(r['new']==['BELOW',6] and r['label_mm']==6 for r in target)
    summary = dict(
        certificate_records=len(rows),distinct_image_files=len(pred['results']),
        passive_other_print_verified=len(reg['passive_other_print']),
        legacy_known=sum(r['old'][0]!='UNKNOWN' for r in rows),
        lost_legacy=sum(r['lost_legacy'] for r in rows),wrong_known=sum(r['wrong_known'] for r in rows),
        recovered=sum(r['recovered'] for r in rows),v26_recovered=recover_six)
    qualified = not summary['lost_legacy'] and not summary['wrong_known'] and recover_six
    write(out/'decision.json',dict(registration_sha256=digest(out/'registration.json'),
                                   predictions_sha256=digest(pred_path), summary=summary,
                                   qualified=qualified,authorizes_motion=False,rows=rows))
    print(summary, 'qualified', qualified, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','score'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path)
    a = p.parse_args()
    if a.stage == 'predict':
        if a.frames is None:
            p.error('predict requires --frames')
        predict(a.out,a.frames)
    else:
        score(a.out)


if __name__=='__main__':
    main()
