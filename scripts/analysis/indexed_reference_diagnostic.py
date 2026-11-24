"""Explain the sealed indexed-print replay without making a new progress prediction."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

from score_indexed_reference import ROOT, decode, digest, load, variants, write
from indexed_reference import _patch


def passage(obs,lab,z,j,far):
    pts={(s['row'],s['column']):s for s in obs['stations']}
    keys=[(r,k) for r in (j,j+1) for k in (9,14)]
    if not set(keys)<=pts.keys():
        return dict(reason='missing station bracket'),None
    marks=[_patch(np.isin(lab,[8*j+k for k in group])&(z>0)&(z<far*.999),2)
           for group in ((1,2),(5,6))]
    if not all(m.any() for m in marks):
        return dict(reason='missing flank'),None
    uv=np.array([pts[k]['uv'] for k in keys])
    cloud=np.vstack([uv,*(np.argwhere(m)[:,::-1]+.5 for m in marks)])
    hull=ConvexHull(cloud)
    x0,y0=np.floor(cloud.min(0)).astype(int)
    x1,y1=np.ceil(cloud.max(0)).astype(int)+1
    yy,xx=np.indices((y1-y0,x1-x0))
    xy=np.stack([xx+x0+.5,yy+y0+.5],axis=-1)
    inside=np.all(xy @ hull.equations[:,:2].T+hull.equations[:,2]<=1e-7,axis=2)
    roi=np.s_[y0:y1,x0:x1]
    back=max(*(pts[k]['camera_m'][2] for k in keys),*(float(z[m].max()) for m in marks))
    d=z[roi]
    deep=inside&(lab[roi]<0)&(d>back+.002)&(d<far*.999)
    cc,n=ndimage.label(deep)
    comps=[]
    for i in range(1,n+1):
        c=cc==i
        touch=ndimage.binary_dilation(c)
        comps.append(dict(area=int(c.sum()),flank_contacts=[int(np.count_nonzero(touch&m[roi])) for m in marks]))
    return dict(connected_patches=comps),uv


def main(out):
    dec=load(out/'decision.json')
    sealed=load(out/'predictions.json')
    if dec['predictions_sha256']!=digest(out/'predictions.json'):
        raise ValueError('not the scored predictions')
    result=dict(scope='Post-hoc explanation only, no estimator updates or changed acceptance.',
                decision_sha256=digest(out/'decision.json'),script_sha256=digest(Path(__file__)),rows=[])
    panels=[]
    for r in dec['rows']:
        if r['phase']!='look':
            continue
        sha=r['sha256']
        rel=next(k for k,v in sealed['files'].items() if v==sha and k.split('/')[0]==r['case'])
        path=out/rel
        if digest(path)!=sha:
            raise ValueError('HD frame changed')
        parent=ROOT/dec['protocol']['parent']/r['case']
        old=next(fr for fr in load(parent/'oracle.json')['frames'] if fr['phase']=='look')
        prior=parent/'hd'/(old['key']+'.npz')
        if digest(prior)!=old['hd_sha256']:
            raise ValueError('unmatched parent exposure')
        with np.load(path) as a,np.load(prior) as b:
            im,z=a['rgb'],a['depth']
            dz=abs(z-b['depth'])*1000
        lab=decode(im)
        row=dict(mm=r['mm'],sha256=sha,max_depth_delta_mm=float(dz.max()),
                 depth_pixels_above_half_mm=int((dz>.5).sum()),variants=[])
        for i,dep in enumerate(variants(z)):
            ref=sealed['images'][sha][i]['reference']
            j=r['mm']-1
            support,uv=passage(ref,lab,dep,j,.3)
            row['variants'].append(dict(reference_first_present=ref['first_present_row'],
                reported_local_mm=ref['local_mm'],reader_reason=ref['apertures'].get(str(j)),**support))
            if i or uv is None or r['mm'] not in (8,9):
                continue
            x,y=np.floor(uv.mean(0)).astype(int)
            x0,y0=x-34,y-28
            panel=Image.new('RGB',(544,490),'white')
            crop=Image.fromarray(im[y0:y0+56,x0:x0+68]).resize((544,448),Image.Resampling.NEAREST)
            panel.paste(crop)
            draw=ImageDraw.Draw(panel)
            for k,q in enumerate(uv):
                u,v=(q-[x0,y0])*8
                draw.ellipse((u-5,v-5,u+5,v+5),outline='white',width=2)
                draw.text((u+6,v-12),f'{j+k//2}/{(9,14)[k%2]}',fill='white',stroke_width=1,stroke_fill='black')
            draw.text((5,452),f"Actual {r['mm']} mm | closed look | native RGB, 8x pixel display",fill='black')
            draw.text((5,469),f"Optical row/column stations (white); local reading: {ref['local_mm']}",fill='black')
            panels.append(panel)
        result['rows'].append(row)
    write(out/'passage_diagnostic.json',result)
    if len(panels)==2:
        image=Image.new('RGB',(1088,490))
        for i,panel in enumerate(panels):
            image.paste(panel,(i*544,0))
        with (out/'terminal_stations.png').open('xb') as fp:
            image.save(fp,format='PNG')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True,type=Path)
    main(ap.parse_args().out)
