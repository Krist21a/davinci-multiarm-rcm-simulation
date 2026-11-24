"""Non-actuating print/optics probe on exactly replayed slide work and look-back states."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import uuid

import numpy as np

from capture_surface_seam import ROOT, add_arguments, connect, sha256, stop_simulation, write_new
from contact_veto_pilot import OUT as DEV, verify_lock
from score_cell_certificate import OUT, SPEC
from seam_face_visibility import CLEAR, FaceCapture, Finished, digest
import seam_slide_world as slide

PRINT=ROOT/'src/tissue/cell_certificate_print.lua'
READER=ROOT/'scripts/vision/cell_certificate_v1.py'


class Optical(FaceCapture):
    def exposure(self, phase, refpath):
        w,sim=self.w,self.w.sim
        def physical():
            return dict(time=sim.getSimulationTime(), nodes=digest(w.truth.nodes()),
                        edges=digest(np.asarray(w.call('tissueConstraints'))),
                        psms=[digest(self.tcp(a)) for a in ('psm1','psm2','psm3')],
                        camera=[digest(v) for v in w.camera()], grip=w.jaws_closed())

        def acquire():
            sim.handleVisionSensor(w.cam)
            raw,size=sim.getVisionSensorImg(w.cam)
            z,sz=sim.getVisionSensorDepth(w.cam,1)
            if list(size)!=[2560,1920] or list(sz)!=list(size):
                raise ValueError('sensor resolution mismatch')
            return np.frombuffer(bytes(raw),np.uint8).reshape(1920,2560,3)[::-1],np.frombuffer(bytes(z),'<f4').reshape(1920,2560)[::-1]

        param=sim.visionfloatparam_perspective_angle
        old=sim.getObjectFloatParam(w.cam,param)
        target=np.deg2rad(30.)
        if not np.isclose(old,np.deg2rad(70.),atol=1e-6,rtol=0):
            raise ValueError('unexpected reference optics')
        before=physical()
        native=[digest(np.asarray(w.image)),digest(w.depth)]
        key=uuid.uuid4().hex
        path=self.folder/'sensor'/(key+'.npz')
        tagged=False
        sim.setIntArray2Property(w.cam,'resolution',[2560,1920])
        try:
            with np.load(refpath) as ref:
                clean=acquire()
                if not all(np.array_equal(a,ref[k]) for k,a in zip(('rgb','depth'),clean)):
                    raise ValueError('original full image mismatch: '+phase)
            sim.setObjectFloatParam(w.cam,param,float(target))
            angle=sim.getObjectFloatParam(w.cam,param)
            if not np.isclose(angle,target,atol=1e-6,rtol=0):
                raise ValueError('30-degree optics readback failed')
            unprinted,z0=acquire()
            tagged=bool(w.call('setCellCertificatePrint',True))
            if not tagged:
                raise ValueError('passive print rejected')
            rgb,z=acquire()
            if before!=physical():
                raise ValueError('physical state changed')
            np.savez_compressed(path,rgb=rgb,depth=z)
            if not w.call('setCellCertificatePrint',False):
                raise ValueError('passive print did not clear')
            tagged=False
            clean_zoom=acquire()
            if not np.array_equal(unprinted,clean_zoom[0]) or not np.array_equal(z0,clean_zoom[1]):
                raise ValueError('30-degree sensor did not restore')
            sim.setObjectFloatParam(w.cam,param,old)
            restored=acquire()
            if not all(np.array_equal(a,b) for a,b in zip(clean,restored)):
                raise ValueError('70-degree high resolution did not restore')
        finally:
            if tagged:
                w.call('setCellCertificatePrint',False)
            sim.setObjectFloatParam(w.cam,param,old)
            sim.setIntArray2Property(w.cam,'resolution',[640,480])
            w.render()
        if native!=[digest(np.asarray(w.image)),digest(w.depth)] or before!=physical():
            raise ValueError('native sensor or physical state did not restore')
        self.record['exposures'].append(dict(key=key,phase=phase,sensor_sha256=sha256(path),
            matched_high_sha256=sha256(refpath),calibrated_angle_rad=angle,fov_restored=True,unchanged_state=before,
            depth_rebuild_max_m=float(np.abs(z-z0).max()),mm=float(w.truth.cut_length()*1000)))


def main(args):
    verify_lock()
    registration=json.loads((OUT/'registration.json').read_bytes())
    if registration['spec_sha256']!=sha256(SPEC) or registration['reader_sha256']!=sha256(READER):
        raise ValueError('registration drift')
    src=DEV/'slide'/args.case
    baseline=json.loads((DEV/'decision.json').read_bytes())
    run=next(r for r in baseline['runs'] if r['case']==args.case)
    clear=json.loads((CLEAR/args.case/'probe.json').read_bytes())
    if (not run['mechanics_ok'] or run['oracle_sha256']!=sha256(src/'oracle.json') or not clear['pose_qualified']
            or sha256(CLEAR/args.case/'high.npz')!=clear['high_sha256']):
        raise ValueError('unqualified source')
    ref=json.loads((src/'oracle.json').read_bytes())
    args.scene=ROOT/'scenes/davinci_si_articulated_verified.ttt'
    args.initial_view,args.initial_view_insertion_m=ref['view']['target'][:2],ref['view']['target'][2]
    folder=args.out_root/args.case
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'sensor').mkdir()
    js=dict(criteria=registration['criteria'],spec_sha256=sha256(SPEC),reader_sha256=sha256(READER),
            collector_sha256=sha256(Path(__file__)),print_sha256=sha256(PRINT),ref_sha256=sha256(src/'oracle.json'),
            phases=['loaded','work','backoff'],frames=[],observations=[],exposures=[],complete=False)
    client=sim=None
    build=slide.staged_source
    def staged(parent):
        orig=build(parent)
        if hashlib.sha256(orig.encode()).hexdigest()!=ref['controller_sha256']:
            raise ValueError('controller reference mismatch')
        return orig+'\n'+PRINT.read_text(encoding='utf-8')
    try:
        client,sim=connect(args)
        if sha256(args.scene)!=ref['scene_sha256']:
            raise ValueError('scene mismatch')
        with patch.object(slide,'staged_source',staged):
            w=slide.SeamSlideWorld(client,sim,args,ref['cfg'],mode='full')
        js['controller_sha256']=w.controller_sha256
        task=Optical(w,ref['cfg'],folder,js,False)
        task.src,task.srcframes=src,ref['frames']
        task.last=[f for f in ref['frames'] if f['phase']=='work'][-1]
        task.clear_ref,task.clear_path=clear,CLEAR/args.case/'high.npz'
        try:
            task.run()
        except Finished:
            js['complete']=True
    except Exception as exc:
        js['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            stop_simulation(client,sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        write_new(folder/'oracle.json',js)
    print(args.case,js.get('error') or js['complete'])
    return int(not js['complete'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--case',choices=('a','b'),required=True)
    ap.add_argument('--out-root',type=Path,default=OUT/'zoom')
    raise SystemExit(main(ap.parse_args()))
