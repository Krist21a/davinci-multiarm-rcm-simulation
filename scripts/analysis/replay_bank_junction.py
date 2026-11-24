"""Registered development screen for the opt-in CutPerception junction correspondence."""
import argparse
from pathlib import Path
from zipfile import ZipFile
import hashlib

import numpy as np

import replay_bilateral_3d as base
from bank_junction import _contacts
from cut_perception import CutPerception
from ruler_readout import inks

ROOT = base.ROOT
SPEC = ROOT/'config/task/m5_junction_hd_v1.json'
PREVIOUS = ROOT/'validation/task/m5_bilateral_3d_hd_dev_v1'
SCENARIOS = ['nominal','depth_noise_0p5','occlusion_transient','slip_regrasp','relax']


def register(out):
    amendment=base.load(SPEC)
    parent=base.load(ROOT/amendment['inherits'])
    reg=base.load(PREVIOUS/'registration.json')
    if parent!=reg['criteria'] or amendment['live_acceptance']['scenarios']!=SCENARIOS:
        raise ValueError('parent protocol or scenario mismatch')
    source=ROOT/parent['source']
    if (base.digest(source/'decision.json')!=parent['source_decision_sha256']
            or base.digest(source/'registration.json')!=parent['source_registration_sha256']):
        raise ValueError('source development identity')
    cal=source/'a/calibration.json'
    if base.digest(cal)!=parent['calibration_sha256']:
        raise ValueError('rigid calibration mismatch')
    paths={p.relative_to(source).as_posix():base.digest(p) for c in parent['oracles']
           for p in (source/c/'hd').glob('*.npz')}
    if paths!=reg['exposures']:
        raise ValueError('exposures differ from original registration')
    archive=ROOT/'scripts/vision/frozen_bilateral_hd_v1.zip'
    with ZipFile(archive) as z:
        for member in ('scripts/vision/bilateral_surface.py','scripts/vision/cut_perception.py'):
            if hashlib.sha256(z.read(member)).hexdigest()!=reg['sources'][member]:
                raise ValueError('v1 source snapshot mismatch')
    sources=[*base.SOURCES,SPEC,Path(__file__),ROOT/'scripts/vision/bank_junction.py',archive]
    data=dict(criteria={**parent,'junction_amendment':amendment},exposures=paths,
              parent_registration_sha256=base.digest(PREVIOUS/'registration.json'),
              sources={p.relative_to(ROOT).as_posix():base.digest(p) for p in sources})
    out.mkdir(parents=True,exist_ok=False)
    base.write(out/'registration.json',data)
    return source,base.load(cal),data


def predict(im,z,cal):
    return [dict(cut=g['cut'],tool=g['scissors']) for d in base.variants(z)
            for g in [CutPerception(**cal,bilateral_surface='junction').estimate(im,d,np.eye(3),np.zeros(3))]]


def controls():
    rows=[]
    for tilt in (0,32):
        for n,fault,expect in ([(k,'',k if k<=10 else None) for k in (0,3,6,8,9,10,11)]
                               +[(9,k,None) for k in ('bridge','invalid','foreground','hidden','shutter')]
                               +[(9,k,8) for k in ('half0','half1')]):
            im,z,cal,point,v,normal=base.fixture(n,tilt,fault)
            pair=[]
            for val in predict(im,z,cal):
                c=val['cut']
                metric=base.compare(c,expect,point+(expect-n)*.001*v,v,normal) if expect is not None else None
                ok=(not c['front_valid'] if expect is None else bool(metric and metric['geometry_ok'])
                    or (fault.startswith('half') and not c['front_valid']))
                pair.append(dict(cut=c,errors=metric,passed=ok))
            rows.append(dict(nominal=n,tilt=tilt,case=fault or 'plain',variants=pair))
    return rows


def seal(out,source,cal,reg):
    if (cal['W'],cal['H'])!=(1280,960) or not np.isclose(cal['f'],640/np.tan(np.deg2rad(35))):
        raise ValueError('wrong HD intrinsics')
    images={}
    for path,sha in reg['exposures'].items():
        if sha in images:
            continue
        with np.load(source/path,allow_pickle=False) as d:
            if set(d.files)!={'rgb','depth'}:
                raise ValueError('not a pure sensor image')
            im,z=d['rgb'],d['depth']
        if im.shape!=(960,1280,3) or im.dtype!=np.uint8 or z.shape!=im.shape[:2]:
            raise ValueError('not native HD')
        images[sha]=predict(im,z,cal)
    base.write(out/'predictions.json',dict(registration_sha256=base.digest(out/'registration.json'),images=images))


def nearest(points,ref):
    return float(np.min(np.linalg.norm(points-ref,axis=1))) if len(points) else None


def attribute(labels, depth, j, side, ref, f, far):
    # Independent, scorer-only quantities: how far is the proposed reference
    # vertex from any actual colour transition, even without a reader fit?
    a,b=13*side+j-1,13*side+j
    uv=np.array([depth.shape[1]/2,depth.shape[0]/2])-f*ref[:2]/ref[2]
    x,y=np.floor(uv).astype(int)
    px=float(depth[y,x]) if 0<=x<depth.shape[1] and 0<=y<depth.shape[0] else None
    return dict(uv=uv.tolist(),z_reference_m=float(ref[2]),z_pixel_m=px,
                pixel_depth_residual_mm=None if px is None else (px-ref[2])*1000,
                to_band_edge_px=nearest(_contacts(labels,depth,a,b,far)[0],uv),
                to_open_lip_px=[nearest(_contacts(labels,depth,k,None,far)[0],uv) for k in (a,b)])


def score_corners(out,source,cal,decision):
    pred=base.load(out/'predictions.json')['images']
    rows=[]
    for c in ('a','b','c'):
        folder=source/c
        o=base.load(folder/'oracle.json')
        R,t=map(np.array,(o['initial_view']['applied_R'],o['initial_view']['applied_t']))
        for fr in o['frames']:
            if fr['phase'] not in ('work','look') or fr['prefix_mm'] not in (8,9,10):
                continue
            sha,mm=fr['hd_sha256'],fr['prefix_mm']
            phys=next(r for r in decision['rows'] if (r['case'],r['phase'],r['mm'],r['sha256'])
                      ==(c,fr['phase'],mm,sha))
            sp,hp=(folder/k/(fr['key']+'.npz') for k in ('structure','hd'))
            if base.digest(sp)!=fr['structure_sha256'] or base.digest(hp)!=sha:
                raise ValueError('correspondence score identity')
            with np.load(sp,allow_pickle=False) as st, np.load(hp,allow_pickle=False) as hd:
                xyz=(st['nodes']-t) @ R
                label,raw=inks(hd['rgb']),hd['depth']
            values=[]
            for val,session,z in zip(pred[sha],phys['variants'],base.variants(raw),strict=True):
                cut=val['cut']
                points=[]
                for j in (mm-1,mm):
                    pair=cut.get('endpoints',{}).get(str(j),[None,None])
                    for side,e in enumerate(pair):
                        q=xyz[j,11+side]
                        diag=attribute(label,z,j,side,q,cal['f'],cal['far'])
                        points.append(dict(row=j,side=side,diagnostic=diag,
                            reference_next_row_span_px=float(cal['f']*np.linalg.norm(
                                xyz[j+1,11+side,:2]/xyz[j+1,11+side,2]-q[:2]/q[2])),
                            error_mm=float(1000*np.linalg.norm(e['camera']-q)) if e else None,
                            error_px=float(np.linalg.norm(np.array(e['uv'])-diag['uv'])) if e else None))
                faces=cut['states'][mm-1] if 'states' in cut else ['?','?']
                good=bool(session['tool_ok'] and faces==['gap','gap'] and
                          all(p['error_mm'] is not None and p['error_mm']<.5 for p in points))
                values.append(dict(points=points,faces=faces,tool_ok=session['tool_ok'],corner_gate=good))
            rows.append(dict(case=c,phase=fr['phase'],mm=mm,sha256=sha,variants=values))
    look=[r for r in rows if r['phase']=='look' and r['mm'] in (8,9)]
    if sorted(r['mm'] for r in look)!=[8,9]:
        raise ValueError('incomplete terminal look pair')
    return rows,all(v['corner_gate'] for r in look for v in r['variants'])


def run(out):
    source,cal,reg=register(out)
    base.write(out/'synthetic.json',controls())
    seal(out,source,cal,reg)
    physical=base.score(out)
    dec=base.load(out/'decision.json')
    rows,minimum=score_corners(out,source,cal,dec)
    result=dict(protocol=reg['criteria']['junction_amendment'],parent_decision_sha256=base.digest(out/'decision.json'),
                sealed_predictions_sha256=base.digest(out/'predictions.json'),terminal_corners=rows,
                minimum_8_9_passed=minimum,inherited_gate_passed=physical,development_passed=minimum and physical,
                scenarios={k:'not_run; independent validation required' if minimum and physical else
                           'not_run; failed optical development gate' for k in SCENARIOS},
                confirmation_used=False,control_enabled=False)
    base.write(out/'m5_decision.json',result)
    print('8/9 corner gate:',minimum,'full development gate:',physical)
    return result['development_passed']


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True,type=Path)
    raise SystemExit(0 if run(ap.parse_args().out) else 1)
