"""Seal the strip reader's RGB-D predictions before joining the fixed legacy labels."""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from replay_m5_v3_corridor_v1 import DATA, ROOT, NEW, checked, coded_sheet, digest, load, write
from m5_v3_corridor_v2 import measure

ID = 'm5_v3_corridor_screen_v2'
PLAN = ROOT/f'config/task/{ID}.json'
PRIOR = DATA/'m5_v3_corridor_screen_v1'


def manifest():
    plan = load(PLAN)
    p = PRIOR/'registration.json'
    checked(p,plan['predecessor_manifest_sha256'])
    pred = load(p)
    for key,pin in pred['sources'].items():
        checked(ROOT/key,pin)
    paths = (PLAN,Path(__file__).resolve(),p,
             ROOT/'scripts/vision/m5_v3_corridor_v2.py',
             ROOT/'scripts/analysis/replay_m5_v3_corridor_v1.py')
    cases = pred['certificates']
    if len(cases)!=32 or len({r['frame_sha256'] for r in cases})!=16:
        raise ValueError('missing registered exposures')
    return dict(protocol=plan,sources={x.relative_to(ROOT).as_posix():digest(x) for x in paths},
                certificates=cases)


def tilt_skew(rgb,z):
    th = np.deg2rad(30)
    rotation = np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])
    # y moves forty pixels between the inner lips (x=7 and x=40) before rotation.
    shear = np.array([[1.,40/33],[0.,1.]])
    matrix = np.linalg.inv(rotation@shear)
    offset = -matrix@np.array([25.,120.])
    warp = lambda x, fill: ndimage.affine_transform(x,matrix,offset,output_shape=(270,250),
                                                   order=0,cval=fill,prefilter=False)
    return np.stack([warp(rgb[:,:,c],0) for c in range(3)],axis=2),warp(z,.23)


def controls(reader):
    tests = []
    for name,prefix,expected in (
        ('below',6,('BELOW',6)),('target9',9,('TARGET',9)),('target10',10,('TARGET',10)),
        ('attached_cap',6,('BELOW',6)),('detached_cap',6,('UNKNOWN',None)),
        ('detached_neighbor_cap',6,('UNKNOWN',None)),('detached_neighbor_bank',6,('UNKNOWN',None)),
        ('cap_depth_jump',6,('UNKNOWN',None)),('one_sided_face',6,('UNKNOWN',None)),
        ('shallow_bridge',6,('UNKNOWN',None)),('unknown_foreground',6,('UNKNOWN',None)),
        ('invalid_depth',6,('UNKNOWN',None)),('missing_bank',6,('UNKNOWN',None)),
        ('thin_interrow_neck',6,('UNKNOWN',None)),('missing_terminal_face',6,('UNKNOWN',None)),
        ('partial_terminal',6,('BELOW',6)),('tilt30_skew40',6,('BELOW',6)),
        ('tilt30_skew40_half_face',6,('UNKNOWN',None)),
        ('tilt30_skew40_invalid',6,('UNKNOWN',None))):
        stride = 14 if name=='thin_interrow_neck' else 10
        rgb,z = coded_sheet(prefix,stride)
        y = 4*stride+4
        if name in ('attached_cap','detached_cap','detached_neighbor_cap','cap_depth_jump'):
            x = 39 if name in ('attached_cap','cap_depth_jump') else 34
            rgb[y,x] = (14,24+14*(6 if name=='detached_neighbor_cap' else 4),205)
            z[y,x] = .124 if name=='cap_depth_jump' else .13
        elif name=='detached_neighbor_bank':
            rgb[y:y+3,24:27],z[y:y+3,24:27] = (128,81,235),.13
        elif name in ('one_sided_face','shallow_bridge','tilt30_skew40_half_face'):
            x = 40 if name=='shallow_bridge' else 18
            rgb[y:y+3,8:18] = (248,24+14*4,147)
            rgb[y:y+3,18:x] = (248,24+14*4,205)
            z[y:y+3,8:x] = .13
        elif name=='unknown_foreground':
            z[y,25] = .13
        elif name in ('invalid_depth','tilt30_skew40_invalid'):
            z[y:y+2,25:27] = np.nan
        elif name=='missing_bank':
            rgb[40:50,40:] = 0
        elif name=='thin_interrow_neck':
            z[10:14] = np.nan
            z[10:14,22:25] = .23
        elif name in ('missing_terminal_face','partial_terminal'):
            rgb[60:70,24:40] = 0
            if name=='partial_terminal':
                z[60:70,24:40] = .23
        if name.startswith('tilt'):
            rgb,z = tilt_skew(rgb,z)
        val = reader(rgb,z,far=.3,sigma=0.)
        actual = (val['verdict'],val['prefix_mm'])
        tests.append(dict(name=name,expected=expected,actual=actual,reason=val['reason'],
                          rows=val['rows'],passed=actual==expected))
    return tests


def predict(out,frames):
    reg = manifest()
    out.mkdir(exist_ok=False)
    lock = out/'registration.json'
    write(lock,reg)
    samples = controls(measure)
    tof = out/'synthetic.json'
    write(tof,samples)
    if not all(x['passed'] for x in samples):
        print('Rejected by synthetic gate:',[x['name'] for x in samples if not x['passed']])
        return
    results = {}
    for r in reg['certificates']:
        key = r['frame_sha256']
        p = frames/r['frame']
        checked(p,key)
        with np.load(p,allow_pickle=False) as f:
            z = f['depth']
            sigma = r['sigma']
            if sigma:
                seed = np.random.SeedSequence([r['seed'],r['tick'],513])
                z = z+np.random.default_rng(seed).normal(0,sigma,z.shape)
            if (z.shape!=(1920,2560) or not np.array_equal(z,f['measured'])
                    or abs(f['hfov_rad'].item()-np.deg2rad(20))>1e-6):
                raise ValueError('measured certificate mismatch: '+r['frame'])
            if key not in results:
                results[key] = dict(sigma=sigma,result=measure(f['rgb'],z,far=f['far'].item(),sigma=sigma))
            if sigma!=results[key]['sigma']:
                raise ValueError('different noise for duplicate file')
    write(out/'predictions.json',dict(registration_sha256=digest(lock),synthetic_sha256=digest(tof),
                                     results=results))


def score(out):
    lock = out/'registration.json'
    reg = load(lock)
    if reg!=manifest():
        raise ValueError('changed replay registration')
    pred_path = out/'predictions.json'
    pred = load(pred_path)
    checked(lock,pred['registration_sha256'])
    synthetic = out/'synthetic.json'
    checked(synthetic,pred['synthetic_sha256'])
    if not all(t['passed'] for t in load(synthetic)):
        raise ValueError('unqualified synthetic control')
    prior = PRIOR/'decision.json'
    checked(prior,reg['protocol']['predecessor_decision_sha256'])
    labels = load(prior)
    if labels['registration_sha256']!=reg['protocol']['predecessor_manifest_sha256']:
        raise ValueError('changed structural reference')
    old = {r['frame']:r for r in labels['rows']}
    rows = []
    for r in reg['certificates']:
        label = old[r['frame']]
        key = r['frame_sha256']
        if key!=label['frame_sha256'] or r['tick']!=label['tick']:
            raise ValueError('misaligned legacy optical/structure evidence')
        v = pred['results'][key]['result']
        new = [v['verdict'],v['prefix_mm']]
        old_verdict = label['old']
        mm = label['label_mm']
        expected = ['BELOW' if mm<9 else 'TARGET',mm] if 0<mm<=10 else ['UNKNOWN',None]
        rows.append(dict(frame=r['frame'],sha256=key,old=old_verdict,new=new,structure_mm=mm,
                         reason=v['reason'],states=[q['status'] for q in v['rows']],
                         wrong_known=new[0]!='UNKNOWN' and new!=expected,
                         lost_legacy=old_verdict[0]!='UNKNOWN' and new!=old_verdict))
    targets = [r for r in rows if r['frame'].startswith(NEW+'/')]
    recovered = len(targets)==2 and all(r['new']==['BELOW',6] and r['structure_mm']==6 for r in targets)
    old_known = sum(r['old'][0]!='UNKNOWN' for r in rows)
    if old_known!=17:
        raise ValueError('unexpected legacy coverage')
    summary = dict(records=len(rows),distinct_files=len(pred['results']),legacy_known=old_known,
                   wrong_known=sum(r['wrong_known'] for r in rows),lost_legacy=sum(r['lost_legacy'] for r in rows),
                   v26_recovered=recovered)
    passed = not summary['wrong_known'] and not summary['lost_legacy'] and recovered
    write(out/'decision.json',dict(registration_sha256=digest(lock),predictions_sha256=digest(pred_path),
                                   labels_sha256=digest(prior),summary=summary,qualified=passed,
                                   authorizes_motion=False,rows=rows))
    print(summary,'qualified',passed,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','score'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=Path)
    a = p.parse_args()
    if a.stage=='predict':
        if a.frames is None:
            p.error('predict requires --frames')
        predict(a.out,a.frames)
    else:
        score(a.out)
