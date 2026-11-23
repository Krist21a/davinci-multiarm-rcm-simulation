"""Calibrated real 20-degree acquisition for the planar-rim development coupon."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from capture_surface_seam import Capture, ROOT, add_arguments, sha256, write_new
from cell_certificate_v1 import classify, decode, patch as ink_patch
from score_cell_certificate import noises
import planar_rim_pilot as rim

SPEC=ROOT/'config/task/seam_rim_zoom_v1.json'
OUT=ROOT/'validation/task/seam_rim_zoom_v1'
ANGLE=20.
PREV=rim.previous


class Capture20(PREV.ReportCapture):
    def save(self,phase,obs):
        Capture.save(self,phase,obs)
        w,sim=self.w,self.w.sim
        rec=self.record['frames'][-1]
        param=sim.visionfloatparam_perspective_angle
        def state():
            with w.truth.purpose('scoring'):
                return (sim.getSimulationTime(),w.truth.nodes().tobytes(),np.asarray(w.call('tissueConstraints')).tobytes(),
                        tuple(self.tcp(a).tobytes() for a in ('psm1','psm2','psm3')),w.jaws_closed())
        before=state()
        rgb0,z0=np.array(w.image),w.depth.copy()
        old=sim.getObjectFloatParam(w.cam,param)
        if not np.isclose(old,np.deg2rad(70),atol=1e-6,rtol=0):
            raise ValueError('unexpected native optics')
        tagged=False
        try:
            sim.setIntArray2Property(w.cam,'resolution',[2560,1920])
            sim.setObjectFloatParam(w.cam,param,float(np.deg2rad(ANGLE)))
            angle=sim.getObjectFloatParam(w.cam,param)
            if not np.isclose(angle,np.deg2rad(ANGLE),atol=1e-6,rtol=0):
                raise ValueError('incorrect zoom state')
            tagged=bool(w.call('setCellCertificatePrint',True))
            if not tagged:
                raise ValueError('print refused')
            sim.handleVisionSensor(w.cam)
            raw,wh=sim.getVisionSensorImg(w.cam)
            buf,sz=sim.getVisionSensorDepth(w.cam,1)
            if list(wh)!=[2560,1920] or list(sz)!=list(wh) or before!=state():
                raise ValueError('unexpected readback or changed physics')
            path=self.folder/'high'/(rec['key']+'.npz')
            np.savez_compressed(path,rgb=np.frombuffer(bytes(raw),np.uint8).reshape(1920,2560,3)[::-1],
                depth=np.frombuffer(bytes(buf),'<f4').reshape(1920,2560)[::-1])
        finally:
            if tagged:
                w.call('setCellCertificatePrint',False)
            sim.setObjectFloatParam(w.cam,param,old)
            sim.setIntArray2Property(w.cam,'resolution',[640,480])
            w.render()
        if (not np.array_equal(w.image,rgb0) or not np.array_equal(w.depth,z0) or state()!=before
                or not np.isclose(sim.getObjectFloatParam(w.cam,param),old,atol=1e-6,rtol=0)):
            raise ValueError('original state was not restored')
        rec.update(high_sha256=sha256(path),restored=True,zoom_angle=angle)
        if phase in ('loaded','pulled'):
            self.record['pull_direction_basis']='transverse distance divided by Lua rest length'


def read20(path):
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!={'rgb','depth'} or data['rgb'].shape!=(1920,2560,3):
            raise ValueError('not registered RGB-D')
        rgb,z=data['rgb'],data['depth']
    if z.shape!=rgb.shape[:2] or np.any(~np.isfinite(z)) or z.min()<=0:
        raise ValueError('invalid depth')
    # Bound the memory footprint of colour decoding on the full-resolution image.
    bounds=[]
    for top in range(0,1920,64):
        cap,face=decode(rgb[top:top+64])
        y,x=np.where((cap>=0)|(face>=0))
        if len(y):
            bounds.append((int(x.min()),int(y.min())+top,int(x.max()),int(y.max())+top))
    x,y=(max(0,min(b[k] for b in bounds)-8) for k in (0,1)) if bounds else (0,0)
    roi=np.s_[y:max((b[3]+9 for b in bounds),default=1),x:max((b[2]+9 for b in bounds),default=1)]
    im=rgb[roi]
    cap,face=decode(im)
    return dict(sensor_sha256=sha256(path),corners=[dict(patch=ink_patch(cap==k)) for k in range(28)],
        faces=[dict(patch=ink_patch(face==k)) for k in range(26)],predictions=[
            classify(im,dep[roi],focal=1280/np.tan(np.deg2rad(ANGLE/2)),principal=[1280-x,960-y],sigma=sigma)
            for sigma,dep in noises(z)])


def evaluate(folder,stage,contract,hashes,lock):
    pre=stage=='preflight'
    entries=contract['development_runs'][:1] if pre else contract['development_runs']
    preds={}
    for e in entries:
        for path in sorted((folder/'asym'/e['id']/'high').glob('*.npz')):
            if path.stem in preds:
                raise ValueError('duplicate frame')
            preds[path.stem]=read20(path)
    output=folder/(stage+'_predictions.json')
    write_new(output,dict(criteria=contract,spec_sha256=sha256(lock),predictions=preds))
    reports,seen=[],set()
    for e in entries:
        src=folder/'asym'/e['id']
        o=json.loads((src/'oracle.json').read_bytes())
        native={}
        for f in o['frames']:
            v=preds[f['key']]
            if not f['restored'] or not np.isclose(f['zoom_angle'],np.deg2rad(ANGLE),atol=1e-6,rtol=0) or f['high_sha256']!=v['sensor_sha256']:
                raise ValueError('invalid optics')
            native[f['key']]=dict(sensor_sha256=f['sensor_sha256'],native=v['predictions'][0],noisy=v['predictions'][1])
        with patch.multiple(PREV.asym.parent,OUTPUT=folder/'asym',PLAN=lock):
            q=PREV.asym.parent.run_score(e,native,seen,contract,hashes)
        if (o.get('pins')!=[24*j+i for j in range(13) for i in (*range(6,11),*range(13,18))]
                or not o.get('roles') or not all(r['ok'] for r in o['roles'].values())
                or len(o.get('measured_cutline_rest_m',[]))!=14
                or not np.allclose(o['measured_cutline_rest_m'],.00276,atol=1e-12,rtol=0)):
            q['issues'].append('missing fixture, rest, or role readback')
        for f in q['rows']:
            with np.load(src/'structure'/(f['key']+'.npz')) as data:
                far=data['nodes'][-1]
                span=far[-1]-far[0]
                pos=(far[[11,12]]-(far[-1]+far[0])/2)@(span/np.linalg.norm(span))
            if not np.allclose([np.linalg.norm(span),*pos],[.023,-.0025,.0005],rtol=0,atol=1e-8):
                q['issues'].append('specimen geometry mismatch')
            v=preds[f['key']]
            f['missing_corners']=[k for k,it in enumerate(v['corners']) if not it['patch']]
            f['missing_face_patches']=[k for k,it in enumerate(v['faces']) if not it['patch']]
            if f['phase'] in ('loaded','pulled'):
                f['optical_ok'] &= all(it.get('states')==[['PRESENT','PRESENT']]*13 for it in v['predictions'])
            elif f['phase']=='work' and f['actual_mm']>=9:
                f['optical_ok'] &= f['decisions']==['STOP']*2
        q['mechanics_ok']=not q['issues']
        reports.append(q)
    if seen!=set(preds) or len({q['run_id'] for q in reports})!=len(reports):
        raise ValueError('incomplete or duplicate evidence')
    ok=all(q['mechanics_ok'] and all(f['optical_ok'] for f in q['rows']) for q in reports)
    if not pre:
        cover={n:{q['case'] for q in reports for f in q['rows'] if f['phase']=='work' and f['actual_mm']==n} for n in (8,9,10)}
        ok &= bool(cover[10]) and any(a!=b for a in cover[8] for b in cover[9])
    write_new(folder/('selection.json' if pre else 'decision.json'),dict(criteria=contract,source_sha256=hashes,
              spec_sha256=sha256(lock),passed=ok,selected=dict(rail='asym',scale=4) if ok else None,
              reports=reports,predictions_sha256=sha256(output),visual_control=False))
    print('passed',ok,[(q['case'],q['issues'],[(f['phase'],f['actual_mm'],f['decisions'],
                f['missing_corners']) for f in q['rows']]) for q in reports])
    return int(not ok)


def main(args):
    amend=json.loads(SPEC.read_bytes())
    baseline=ROOT/amend['baseline']
    old=json.loads(baseline.read_bytes())
    contract=dict(**{k:v for k,v in old.items() if k!='id'},id=amend['id'],zoom_amendment=amend,
                  zoom_amendment_sha256=sha256(SPEC),rim_registration_sha256=sha256(baseline))
    lock=args.out_root/'registration.json'
    if args.stage=='collect' and args.case=='pull' and not lock.exists():
        args.out_root.mkdir(parents=True,exist_ok=True)
        write_new(lock,contract)
    if json.loads(lock.read_bytes())!=contract:
        raise ValueError('registration drift')
    sources=(*PREV.SOURCES,'scripts/analysis/planar_rim_pilot.py','config/task/seam_asymmetric_rest_v1.json','scripts/analysis/rim_zoom_pilot.py')
    hashes={p:sha256(ROOT/p) for p in sources}
    if hashes['scripts/vision/cell_certificate_v1.py']!=contract['certificate_sha256'] or hashes['src/tissue/cell_certificate_print.lua']!=contract['print_sha256']:
        raise ValueError('certificate or print changed')
    if args.stage=='collect':
        if not args.case:
            raise ValueError('missing case')
        args.rail='asym'
        with patch.object(PREV.asym,'asymmetric_source',rim.staged),patch.multiple(PREV.acquisition,PLAN=lock,
                SOURCES=sources,SeamRailWorld=PREV.World,RailCapture=Capture20):
            return PREV.acquisition.acquire(args)
    return evaluate(args.out_root,args.stage,contract,hashes,lock)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--stage',choices=('collect','preflight','work'),required=True)
    ap.add_argument('--case',choices=('pull','a','b','c','d','e'))
    ap.add_argument('--out-root',type=Path,default=OUT)
    raise SystemExit(main(ap.parse_args()))
