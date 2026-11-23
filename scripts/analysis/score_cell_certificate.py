"""Synthetic reader checks and independent optimistic physical screen. Never drives a tool."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_surface_seam import ROOT, sha256, write_new
from contact_veto_pilot import OUT as DEV
from cell_certificate_v1 import OPEN, PRESENT, UNKNOWN, block, classify, projected_interior, verdict

SPEC=ROOT/'config/task/seam_cell_certificate_v1.json'
OUT=ROOT/'validation/task/seam_cell_certificate_v1'


def pattern(n, *, partial=None, bridge=None, covered=None):
    # Ideal pinhole image of 1 mm cells on a frontoparallel calibration target at 100 mm.
    pitch=48
    h,w=14*pitch,4*pitch
    yy,xx=np.indices((h,w))+.5
    x,y=(xx-1.5*pitch)/pitch,(yy-.5*pitch)/pitch
    rgb=np.zeros((h,w,3),np.uint8)
    z=np.full((h,w),.14)
    for j in range(13):
        row=(y>=j)&(y<j+1)
        for s,bank in enumerate(((x>=-.6)&(x<0),(x>=1)&(x<1.6))):
            rgb[row&bank]=(32+48*(j%4),235 if s==0 else 35+46*(j//4),35+46*(j//4) if s==0 else 235)
            z[row&bank]=.1
        for hidx,tri in enumerate((x>=y-j,x<y-j)):
            mask=row&(x>=0)&(x<1)&tri
            if (j>=n and (j,hidx)!=partial) or (j,hidx)==bridge:
                rgb[mask]=(248,24+14*j,147+58*hidx)
                z[mask]=.1
            if covered==(j,hidx):
                rgb[mask]=100
                z[mask]=.09
    for j in range(14):
        for side in (0,1):
            d=(x-side)*(1 if side else -1)
            patch=(d>=0)&(d+abs(y-j)<.15)
            rgb[patch]=(14,24+14*j,147+58*side)
            z[patch]=.1
    return rgb,z,4800.,[w/2,h/2]


def noises(z):
    seed=int(hashlib.sha256(z.tobytes()).hexdigest()[:16],16)
    yield 0., z
    yield .0005,z+np.random.default_rng(seed).normal(0.,.0005,z.shape)


def controls():
    entries=[('uncut',0,{},UNKNOWN),('eight',8,{},'BELOW'),('half_a',8,dict(partial=(8,0)),'BELOW'),
             ('half_b',8,dict(partial=(8,1)),'BELOW'),('nine',9,{},'STOP'),('ten',10,{},'STOP'),
             ('earlier_bridge_a',10,dict(bridge=(3,0)),UNKNOWN),('earlier_bridge_b',10,dict(bridge=(6,1)),UNKNOWN),
             ('hidden_ninth',9,dict(covered=(8,0)),UNKNOWN),('shutter',9,{},UNKNOWN)]
    out=[]
    for name,n,kw,want in entries:
        rgb,z,f,pp=pattern(n,**kw)
        if name=='shutter':
            rgb[:]=70
            z[:]=.02
        predictions=[classify(rgb,depth,focal=f,principal=pp,sigma=sig) for sig,depth in noises(z)]
        out.append(dict(case=name,expected=want,predictions=predictions,passed=all(q['decision']==want for q in predictions)))
    return out


def ceiling(depth, xyz, faces, focal, radius, sigma):
    h,w=depth.shape
    if np.any(xyz[:,2]<=0):
        raise ValueError('behind camera')
    pix=np.array([w/2,h/2])-focal*xyz[:,:2]/xyz[:,2:]
    rows=[]
    for j in range(13):
        a=24*j+11
        record=[]
        for side,ids in enumerate(([a,a+1,a+25],[a,a+25,a+24])):
            if faces[j,side]:
                record.append(dict(state=PRESENT,privileged=True))
                continue
            q=pix[ids]
            lo=np.maximum(np.floor(q.min(0)).astype(int),[0,0])
            hi=np.minimum(np.ceil(q.max(0)).astype(int)+1,[w,h])
            y,x=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
            valid=projected_interior(q,np.full((3,2),radius),np.stack((x+.5,y+.5),axis=-1))
            far=valid & (depth[y,x]>xyz[ids,2].max()+.002+.00025+3*sigma)
            count=int(block(far).sum())
            record.append(dict(state=OPEN if count else UNKNOWN,pixels=int(far.sum()),blocks=count))
        rows.append(record)
    decision,n=verdict([[face['state'] for face in row] for row in rows],linked=True)
    return dict(rows=rows,decision=decision,n=n)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(SPEC.read_bytes())
    write_new(out/'registration.json',dict(criteria=protocol,spec_sha256=sha256(SPEC),source_sha256=sha256(Path(__file__)),
              reader_sha256=sha256(ROOT/'scripts/vision/cell_certificate_v1.py')))
    synthetic=controls()
    write_new(out/'synthetic.json',dict(tests=synthetic,all_pass=all(t['passed'] for t in synthetic)))
    orig=json.loads((DEV/'decision.json').read_bytes())
    reports=[]
    for run in orig['runs']:
        folder=DEV/'slide'/run['case']
        if not run['mechanics_ok'] or run['oracle_sha256']!=sha256(folder/'oracle.json'):
            raise ValueError('unverified physical run')
        oracle=json.loads((folder/'oracle.json').read_bytes())
        if 'CAMERA_ADJUST' in oracle['visited'] or not np.allclose(oracle['view']['target'],[4.5,0,.08],atol=1e-6,rtol=0):
            raise ValueError('camera changed')
        for ref in oracle['frames']:
            key=ref['key']+'.npz'
            struct,high=folder/'structure'/key,folder/'high'/key
            if sha256(struct)!=ref['structure_sha256'] or sha256(high)!=ref['high_sha256']:
                raise ValueError('broken evidence pair')
            with np.load(struct) as st,np.load(high) as data:
                z=data['depth']
                if z.shape!=(1920,2560):
                    raise ValueError('expected real 4x ECM image')
                xyz=(st['nodes'].reshape(-1,3)-oracle['view']['applied_t'])@np.array(oracle['view']['applied_R'])
                checks=[dict(radius_px=radius,sigma_m=sigma,**ceiling(depth,xyz,st['faces'],1280/np.tan(np.deg2rad(35)),radius,sigma))
                        for sigma,depth in noises(z) for radius in (0,1)]
            reports.append(dict(case=run['case'],phase=ref['phase'],mm=ref['prefix_mm'],key=ref['key'],checks=checks))
    write_new(out/'ceiling.json',dict(scope='uses privileged coordinates and presence; not a reader',
              source_sha256=sha256(DEV/'decision.json'),reports=reports))
    print('synthetic',[(t['case'],[p['decision'] for p in t['predictions']]) for t in synthetic])
    print('ideal working pair',[(r['case'],r['mm'],r['checks'][0]['decision']) for r in reports if r['phase']=='work' and r['mm']>=8])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=OUT)
    main(ap.parse_args().out)
