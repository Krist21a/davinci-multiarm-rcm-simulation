"""Independent wider specimen preflight; no cuts unless the loaded RGB-D passes."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import capture_seam_rail as acquisition
from capture_surface_seam import Capture, ROOT, add_arguments, sha256, write_new
from cell_certificate_v1 import PRESENT
from score_cell_camera import blind
import score_surface_seam as parent
from seam_wide_cell_world import WideCellWorld

SPEC=ROOT/'config/task/seam_wide_cell_v1.json'
OUT=ROOT/'validation/task/seam_wide_cell_v1'
SOURCES=(*acquisition.SOURCES, 'scripts/analysis/wide_cell_pilot.py','scripts/analysis/score_cell_camera.py',
         'scripts/analysis/score_cell_certificate.py','scripts/vision/cell_certificate_v1.py','src/tissue/cell_certificate_print.lua',
         'scripts/task/seam_wide_cell_world.py','scripts/task/seam_slide_world.py','src/tissue/seam_slide.lua',
         'src/tissue/seam_slide_create.lua')


class CaptureWide(Capture):
    def save(self,phase,obs):
        super().save(phase,obs)
        w,sim=self.w,self.w.sim
        fr=self.record['frames'][-1]
        im0,z0=np.asarray(w.image).copy(),w.depth.copy()
        def state():
            with w.truth.purpose('scoring'):
                return (sim.getSimulationTime(),w.truth.nodes().tobytes(),np.asarray(w.call('tissueConstraints')).tobytes(),
                        tuple(self.tcp(a).tobytes() for a in ('psm1','psm2','psm3')),w.jaws_closed())
        before=state()
        param=sim.visionfloatparam_perspective_angle
        old=sim.getObjectFloatParam(w.cam,param)
        if not np.isclose(old,np.deg2rad(70),rtol=0,atol=1e-6):
            raise ValueError('native lens not restored')
        tagged=False
        try:
            sim.setObjectFloatParam(w.cam,param,float(np.deg2rad(30)))
            sim.setIntArray2Property(w.cam,'resolution',[2560,1920])
            angle=sim.getObjectFloatParam(w.cam,param)
            if not np.isclose(angle,np.deg2rad(30),rtol=0,atol=1e-6):
                raise ValueError('camera angle readback failed')
            tagged=bool(w.call('setCellCertificatePrint',True))
            if not tagged:
                raise ValueError('print rendering refused')
            sim.handleVisionSensor(w.cam)
            raw,size=sim.getVisionSensorImg(w.cam)
            buf,sz=sim.getVisionSensorDepth(w.cam,1)
            if list(size)!=[2560,1920] or list(sz)!=list(size) or before!=state():
                raise ValueError('unmatched camera/physical capture')
            dest=self.folder/'high'/(fr['key']+'.npz')
            np.savez_compressed(dest,rgb=np.frombuffer(bytes(raw),np.uint8).reshape(1920,2560,3)[::-1],
                                depth=np.frombuffer(bytes(buf),'<f4').reshape(1920,2560)[::-1])
        finally:
            if tagged:
                w.call('setCellCertificatePrint',False)
            sim.setIntArray2Property(w.cam,'resolution',[640,480])
            sim.setObjectFloatParam(w.cam,param,old)
            w.render()
        if not np.array_equal(w.image,im0) or not np.array_equal(w.depth,z0) or before!=state():
            raise ValueError('physical or optical state did not restore')
        fr.update(high_sha256=sha256(dest),zoom_angle=angle,restored=True)


def evaluate(args,contract,hashes):
    pre=args.stage=='preflight'
    entries=contract['development_runs'][:1] if pre else contract['development_runs']
    folder=args.out_root
    raw=folder/(args.stage+'_predictions.json')
    decision=folder/('selection.json' if pre else 'decision.json')
    if raw.exists() or decision.exists():
        raise FileExistsError('frozen outputs cannot be replaced')
    predictions={}
    for e in entries:
        for p in sorted((folder/'wide'/e['id']/'high').glob('*.npz')):
            if p.stem in predictions:
                raise ValueError('duplicate token')
            predictions[p.stem]=blind(p)
    write_new(raw,dict(criteria=contract,spec_sha256=sha256(SPEC),source_sha256=hashes,predictions=predictions))
    seen=set()
    reports=[]
    for e in entries:
        src=folder/'wide'/e['id']
        oracle=json.loads((src/'oracle.json').read_bytes())
        manifest={}
        for fr in oracle['frames']:
            token=fr['key']
            v=predictions[token]
            if not fr['restored'] or fr['high_sha256']!=v['sensor_sha256']:
                raise ValueError('unverified capture')
            manifest[token]=dict(sensor_sha256=fr['sensor_sha256'],native=v['predictions'][0],noisy=v['predictions'][1])
        with patch.multiple(parent,PLAN=SPEC,OUTPUT=folder/'wide'):
            q=parent.run_score(e,manifest,seen,contract,hashes)
        issues=q['issues']
        ids=[24*j+i for j in range(13) for i in (*range(6,11),*range(13,18))]
        if oracle.get('pins')!=ids:
            issues.append('guide readback mismatch')
        for f in q['rows']:
            v=predictions[f['key']]
            with np.load(src/'structure'/(f['key']+'.npz')) as st:
                far=st['nodes'][-1]
                if (not np.isclose(np.linalg.norm(far[0]-far[-1]),.023,rtol=0,atol=1e-8)
                        or not np.isclose(np.linalg.norm(far[11]-far[12]),.005,rtol=0,atol=1e-8)):
                    issues.append('wide specimen dimensions not verified')
            outputs=v['predictions']
            if f['phase'] in ('loaded','pulled'):
                f['all_present']=all(x.get('states')==[[PRESENT,PRESENT]]*13 for x in outputs)
                f['optical_ok'] &= f['all_present']
            elif f['phase']=='work' and f['actual_mm']>=9:
                f['optical_ok'] &= f['decisions']==['STOP','STOP']
            f['corners_no_patch']=[k for k,c in enumerate(v['corners']) if not c['patch']]
            f['faces_no_patch']=[k for k,c in enumerate(v['faces']) if not c['patch']]
        q['mechanics_ok']=not issues
        reports.append(q)
    if set(predictions)!=seen or len({q['run_id'] for q in reports})!=len(reports):
        raise ValueError('unjoined records')
    good=all(q['mechanics_ok'] and all(f['optical_ok'] for f in q['rows']) for q in reports)
    if not pre:
        lengths={m:{q['case'] for q in reports for f in q['rows'] if f['phase']=='work' and f['actual_mm']==m} for m in (8,9,10)}
        good &= bool(lengths[10]) and any(x!=y for x in lengths[8] for y in lengths[9])
    write_new(decision,dict(criteria=contract,runs=reports,passed=good,selected=dict(rail='wide',scale=4) if good else None,
              spec_sha256=sha256(SPEC),source_sha256=hashes,predictions_sha256=sha256(raw),visual_control=False))
    print('pass',good,[(q['case'],q['mechanics_ok'],q['issues'],[(f['phase'],f['corners_no_patch'],f['faces_no_patch'])
                for f in q['rows']]) for q in reports])
    return int(not good)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--case',choices=('pull','a','b','c','d','e'))
    ap.add_argument('--stage',choices=('collect','preflight','work'),required=True)
    ap.add_argument('--out-root',type=Path,default=OUT)
    args=ap.parse_args()
    spec=json.loads(SPEC.read_bytes())
    fingerprints={s:sha256(ROOT/s) for s in SOURCES}
    if (fingerprints['scripts/vision/cell_certificate_v1.py']!=spec['certificate_sha256'] or
            fingerprints['src/tissue/cell_certificate_print.lua']!=spec['print_sha256']):
        raise ValueError('reader/print mismatch')
    if args.stage=='collect':
        if not args.case:
            ap.error('collection requires --case')
        args.rail='wide'
        with patch.multiple(acquisition,PLAN=SPEC,SOURCES=SOURCES,SeamRailWorld=WideCellWorld,RailCapture=CaptureWide):
            raise SystemExit(acquisition.acquire(args))
    raise SystemExit(evaluate(args,spec,fingerprints))
