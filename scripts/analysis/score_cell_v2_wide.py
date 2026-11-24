"""Separate same-exposure joint tool/cell assessment on the native wide-angle coded print."""
import argparse
from pathlib import Path

import numpy as np

from replay_cell_v2 import ROOT, classify, contains, depths, digest, load, material_state, write
from cell_certificate_v1 import decode, patch
from cut_perception import CutPerception

CASES = {'zero':0, 'three':3, 'six':6, 'eight':8, 'nine':9, 'ten':10}


def main(out):
    regfile = out/'registration.json'
    reg = load(regfile)
    if any(digest(ROOT/p)!=expected for p,expected in reg['sources'].items()):
        raise ValueError('registered software changed')
    sealed = {}
    for case in CASES:
        dst = out/case
        if not (dst/'calibration.json').is_file():
            continue
        cal = load(dst/'calibration.json')
        tool = CutPerception(**cal)
        if (cal['W'],cal['H']) != (1280,960) or not np.isclose(cal['f'],640/np.tan(np.deg2rad(35.))):
            raise ValueError('wrong optics')
        kw = dict(focal=cal['f'],principal=[640,480])
        for path in sorted((dst/'hd').glob('*.npz')):
            key = digest(path)
            if key in sealed:
                continue
            with np.load(path,allow_pickle=False) as data:
                if set(data.files)!={'rgb','depth'}:
                    raise ValueError('not a sensor-only exposure')
                im,z = data['rgb'],data['depth']
            if im.shape!=(960,1280,3) or z.shape!=(960,1280) or im.dtype!=np.uint8:
                raise ValueError('wrong exposure dimensions')
            cap,_ = decode(im)
            variants = []
            for sigma,d in depths(z):
                cell = classify(im,d,**kw,sigma=sigma)
                scissors,_ = tool._scissors(im,d)
                variants.append(dict(cell=cell, scissors=scissors))
            sealed[key] = dict(variants=variants, cap_ink_with_4px_support=sum(patch(cap==k) for k in range(28)))
    shutter = [classify(np.full((96,128,3),90,np.uint8),z,focal=640/np.tan(np.deg2rad(35.)),
                        principal=[64,48],sigma=s) for s,z in depths(np.full((96,128),.03))]
    write(out/'predictions.json',dict(registration_sha256=digest(regfile), shutter=shutter, sensors=sealed))
    rows, used, issues = [],set(),[]
    if any(v['decision']!='UNKNOWN' or v['local_mm'] is not None for v in shutter):
        issues.append('false shutter read')
    for case, mm in CASES.items():
        dst = out/case
        if not (dst/'oracle.json').exists():
            issues.append(case+': missing acquisition')
            continue
        o = load(dst/'oracle.json')
        phases = [f['phase'] for f in o['frames']]
        if (not o['qualified'] or o.get('error') or o.get('cleanup_error')
                or o['registration_sha256']!=digest(regfile) or o['reference_sha256']!=reg['references'][case]
                or not (dst/'calibration.json').is_file() or o.get('calibration_sha256')!=digest(dst/'calibration.json')
                or (mm and (phases!=['work','look'] or not o.get('closed_ready_return')))
                or (not mm and (phases!=['loaded','pulled'] or len(o['pull_motion_m'])!=2
                    or not all(x>0 for x in o['pull_motion_m'].values())))):
            issues.append(case+': unqualified checkpoint')
        qualified = []
        for frame in o['frames']:
            sha = frame.get('hd_sha256')
            if not sha or sha not in sealed:
                issues.append(case+': incomplete exposure')
                continue
            p=dst/'structure'/(frame['key']+'.npz')
            if digest(p)!=frame['structure_sha256'] or digest(dst/'hd'/(frame['key']+'.npz'))!=sha:
                raise ValueError('inconsistent join')
            used.add(sha)
            with np.load(p,allow_pickle=False) as s:
                oracle=material_state(s['edges'])
                if (oracle['prefix_rows']!=mm or mm!=frame['prefix_mm'] or oracle['farthest_row_end']>10
                        or not np.array_equal(oracle['faces'],s['faces']) or frame['obs']['torn']
                        or frame['obs']['faults'] or not frame['closed']
                        or (frame['phase']!='look' and not frame.get('matched_key'))):
                    issues.append(case+': structural or pose mismatch')
                R,t=map(np.array,frame['camera'])
                xyz=((s['nodes']-t)@R)[:,11:13].reshape(28,3)
            uv=np.array([640,480])-640/np.tan(np.deg2rad(35.))*xyz[:,:2]/xyz[:,2:]
            record=sealed[sha]
            vs=[]
            for v in record['variants']:
                cell=v['cell']
                escaped=[i for i,c in enumerate(cell.get('corners',[])) if c and not contains(c['polygon'],uv[i])]
                wrong=[(j,h) for j,pair in enumerate(cell.get('states',[])) for h,st in enumerate(pair)
                       if (st=='OPEN' and oracle['faces'][j][h]) or (st=='PRESENT' and not oracle['faces'][j][h])]
                if escaped or wrong or any(n is not None and (not mm or n!=mm) for n in (cell['local_mm'],cell['prefix'])):
                    issues.append(case+'/'+frame['phase']+': incorrect or unbounded evidence')
                vs.append(dict(local=cell['local_mm'],prefix=cell['prefix'],decision=cell['decision'],
                               tool=v['scissors']['valid'],bounded=sum(bool(c) for c in cell.get('corners',[])),
                               escaped=escaped,wrong_faces=wrong))
            qualified.append(all(v['local']==mm and v['tool'] for v in vs))
            rows.append(dict(case=case,phase=frame['phase'],mm=mm,sensor_sha256=sha,
                             caps_with_patch=record['cap_ink_with_4px_support'],variants=vs))
        if mm and not any(qualified):
            issues.append(f'{case}: no joint work/look observation at {mm} mm')
    if used!=set(sealed):
        issues.append('unmatched sensor')
    write(out/'decision.json',dict(criteria=reg['criteria'],registration_sha256=digest(regfile),
                                   predictions_sha256=digest(out/'predictions.json'),rows=rows,issues=issues,
                                   passed=not issues,confirmation=False,control_enabled=False))
    print('Step 2 wide coded-print development:',not issues,issues)
    return not issues


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    raise SystemExit(0 if main(ap.parse_args().out) else 1)
