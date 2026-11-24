"""Evaluate only sensor files first; use actual topology afterwards to explain unavailable marks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_cell_certificate import PRINT,READER
from capture_surface_seam import ROOT, sha256, write_new
from cell_certificate_v1 import classify, decode, patch
from contact_veto_pilot import OUT as DEV
from score_cell_certificate import OUT, SPEC, noises


def blind(path):
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!={'rgb','depth'} or data['rgb'].shape!=(1920,2560,3):
            raise ValueError('not a registered sensor exposure')
        rgb,z=data['rgb'],data['depth']
    if z.shape!=rgb.shape[:2] or not np.isfinite(z).all() or z.min()<=0:
        raise ValueError('invalid depth')
    # Locate the material codes without using state or structure. Keep full-camera intrinsics.
    y0,x0=rgb.shape[:2]
    y1=x1=0
    for top in range(0,len(rgb),64):
        cap,face=decode(rgb[top:top+64])
        y,x=np.where((cap>=0)|(face>=0))
        if len(y):
            y0,x0=min(y0,int(y.min())+top),min(x0,int(x.min()))
            y1,x1=max(y1,int(y.max())+top+1),max(x1,int(x.max())+1)
    if y1==0:
        y0=x0=0
        y1=x1=1
    y0,x0=max(0,y0-8),max(0,x0-8)
    roi=np.s_[y0:y1+8,x0:x1+8]
    cap,face=decode(rgb[roi])
    pp=np.array([1280-x0,960-y0])
    focal=1280/np.tan(np.deg2rad(15))
    return dict(sensor_sha256=sha256(path),corners=[dict(pixels=int((cap==k).sum()),patch=patch(cap==k)) for k in range(28)],
                faces=[dict(pixels=int((face==k).sum()),patch=patch(face==k)) for k in range(26)],
                predictions=[classify(rgb[roi],dep[roi],focal=focal,principal=pp,sigma=sig) for sig,dep in noises(z)])


def hosts(edges):
    bonds={tuple(sorted((a,b))):v for a,b,v in edges.tolist()}
    available=[]
    for row in range(14):
        for i in (11,12):
            k=24*row+i
            candidates=[]
            for j in (row-1,row):
                if 0<=j<13:
                    a=24*j+(10 if i==11 else 12)
                    candidates.extend(q for q in ((a,a+1,a+25),(a,a+25,a+24)) if k in q)
            alive=[all(bonds[tuple(sorted((q[h],q[(h+1)%3])))] for h in range(3)) for q in candidates]
            available.append(sum(alive))
    return available


def main(folder):
    regs=json.loads((OUT/'registration.json').read_bytes())
    pred={}
    paths=sorted(folder.glob('*/sensor/*.npz'),key=lambda p:p.stem)
    if len(paths)!=6:
        raise ValueError('missing two complete three-phase acquisitions')
    for path in paths:
        if path.stem in pred:
            raise ValueError('duplicate sensor token')
        pred[path.stem]=blind(path)
    write_new(folder/'blind_predictions.json',dict(spec_sha256=sha256(SPEC),reader_sha256=sha256(READER),predictions=pred))
    rows,used=[],set()
    for name in ('a','b'):
        oracle=json.loads((folder/name/'oracle.json').read_bytes())
        source=DEV/'slide'/name/'oracle.json'
        if (not oracle['complete'] or oracle.get('error') or oracle['print_sha256']!=sha256(PRINT)
                or not sha256(READER)==regs['reader_sha256']==oracle['reader_sha256']
                or not sha256(SPEC)==regs['spec_sha256']==oracle['spec_sha256']
                or sha256(source)!=oracle['ref_sha256'] or [x['phase'] for x in oracle['exposures']]!=['loaded','work','backoff']):
            raise ValueError('unmatched scientific record')
        frames=json.loads(source.read_bytes())['frames']
        for x in oracle['exposures']:
            f=frames[0] if x['phase']=='loaded' else [f for f in frames if f['phase']=='work'][-1]
            st=source.parent/'structure'/(f['key']+'.npz')
            if sha256(st)!=f['structure_sha256'] or not np.isclose(x['calibrated_angle_rad'],np.deg2rad(30.),rtol=0,atol=1e-6):
                raise ValueError('unmatched material or optics')
            with np.load(st) as raw:
                edges=raw['edges']
                if x['unchanged_state']['edges'] != hashlib.sha256(np.ascontiguousarray(edges).tobytes()).hexdigest():
                    raise ValueError('unmatched edge table')
                support=hosts(edges)
            token=x['key']
            d=pred[token]
            if token in used or d['sensor_sha256']!=x['sensor_sha256'] or not x['fov_restored'] or not np.isclose(x['mm'],f['prefix_mm']):
                raise ValueError('unverified frame')
            used.add(token)
            impossible=[k for k,n in enumerate(support) if not n]
            if any(d['corners'][k]['pixels'] for k in impossible):
                raise ValueError('measured corner ink without host material: check colour attribution')
            rows.append(dict(case=name,phase=x['phase'],mm=f['prefix_mm'],key=token,hostless=impossible,
                             available_but_no_patch=[k for k,n in enumerate(support) if n and not d['corners'][k]['patch']],
                             surviving_hosts=support,decisions=[p['decision'] for p in d['predictions']],
                             target_faces=d['faces'][16:18]))
    if used!=set(pred):
        raise ValueError('partial join')
    pair=all(q['decisions']==[('STOP' if q['mm']>=9 else 'BELOW')]*2 for q in rows if q['phase']=='work')
    write_new(folder/'decision.json',dict(criteria=regs['criteria'],blind_sha256=sha256(folder/'blind_predictions.json'),
              scorer_sha256=sha256(Path(__file__)),rows=rows,working_pair=pair,full_development_validated=False,
              confirmation=False,visual_control=False))
    for q in rows:
        print(q['case'],q['phase'],q['decisions'],'hostless',q['hostless'],'intact but no patch',q['available_but_no_patch'],
              'cell 8',q['target_faces'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder',type=Path,default=OUT/'zoom')
    main(ap.parse_args().folder)
