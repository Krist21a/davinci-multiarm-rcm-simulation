"""Actual fixed-view A/B optics of two material faces; read-only with respect to motion control."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import ndimage

from capture_surface_seam import Capture, ROOT, add_arguments, connect, sha256, stop_simulation, write_new
from contact_veto_pilot import OUT as DEV, verify_lock
import seam_slide_world as slide

SPEC = ROOT/'config/task/seam_face_visibility_v1.json'
OUT = ROOT/'validation/task/seam_face_visibility_v1'
CLEAR = ROOT/'validation/task/seam_clear_observation_v1'
LUA = '''
local faceProbeSaved=nil
function setBenchFacePrint(on)
    local k=2*(8*23+11)+1
    if on then
        assert(not faceProbeSaved)
        faceProbeSaved={T.tri[k][7],T.tri[k+1][7],#T.colors}
        T.colors[#T.colors+1]={245/255,17/255,184/255};T.tri[k][7]=#T.colors
        T.colors[#T.colors+1]={245/255,188/255,17/255};T.tri[k+1][7]=#T.colors
    else
        assert(faceProbeSaved)
        T.tri[k][7],T.tri[k+1][7]=faceProbeSaved[1],faceProbeSaved[2]
        while #T.colors>faceProbeSaved[3] do table.remove(T.colors) end
        faceProbeSaved=nil
    end
    drawPrinted()
    return true
end
'''


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Finished(Exception):
    pass


class FaceCapture(Capture):
    def exposure(self, phase, refpath):
        w, sim = self.w, self.w.sim
        p = self.folder
        native, native_z = np.array(w.image), w.depth.copy()

        def state():
            return (sim.getSimulationTime(), digest(w.truth.nodes()), digest(np.asarray(w.call('tissueConstraints'))),
                    tuple(digest(self.tcp(a)) for a in ('psm1','psm2','psm3')),
                    tuple(digest(v) for v in w.camera()), w.jaws_closed())

        def high():
            sim.handleVisionSensor(w.cam)
            raw, wh = sim.getVisionSensorImg(w.cam)
            buf, whz = sim.getVisionSensorDepth(w.cam, 1)
            if list(wh) != [2560,1920] or list(whz) != list(wh):
                raise ValueError('high-resolution readback mismatch')
            return (np.frombuffer(bytes(raw), np.uint8).reshape(1920,2560,3)[::-1],
                    np.frombuffer(bytes(buf), '<f4').reshape(1920,2560)[::-1])

        baseline = state()
        sim.setIntArray2Property(w.cam, 'resolution', [2560,1920])
        printed = False
        try:
            im, z = high()
            with np.load(refpath, allow_pickle=False) as ref:
                if not np.array_equal(im, ref['rgb']) or not np.array_equal(z, ref['depth']):
                    raise ValueError('unaltered high exposure did not reproduce '+phase)
            printed = bool(w.call('setBenchFacePrint', True))
            if not printed:
                raise ValueError('probe print refused')
            target, dep = high()
            if state() != baseline:
                raise ValueError('non-optical state changed')
            if not w.call('setBenchFacePrint', False):
                raise ValueError('could not remove passive print')
            printed = False
            restored, restored_z = high()
            if not np.array_equal(restored, im) or not np.array_equal(restored_z, z) or state() != baseline:
                raise ValueError('high exposure did not exactly restore')
        finally:
            if printed:
                w.call('setBenchFacePrint', False)
            sim.setIntArray2Property(w.cam, 'resolution', [640,480])
            w.render()
        if (not np.array_equal(w.image, native) or not np.array_equal(w.depth, native_z)
                or state() != baseline):
            raise ValueError('native exposure did not exactly restore')
        changed = np.any(im != target, axis=2)
        y, x = np.where(changed)
        roi = (slice(max(0,y.min()-8),y.max()+9),slice(max(0,x.min()-8),x.max()+9)) if len(y) else np.s_[:1,:1]
        file = p/(phase+'.npz')
        np.savez_compressed(file, original=im[roi], printed=target[roi], original_depth=z[roi], printed_depth=dep[roi])
        counts = []
        for color in self.colors:
            mask = changed[roi] & (dep[roi]==z[roi]) & (np.abs(target[roi].astype(int)-color).max(axis=2)<=3)
            cc, n = ndimage.label(mask,np.ones((3,3),bool))
            counts.append(dict(pixels=int(mask.sum()), largest=int(np.bincount(cc[mask]).max()) if n else 0))
        self.record['exposures'].append(dict(phase=phase, restored=True, time=baseline[0], reference_sha256=sha256(refpath),
            original_rgb_sha256=digest(im), printed_rgb_sha256=digest(target), depth_changed=int((dep!=z).sum()),
            changed_rgb_pixels=int(changed.sum()), inks=counts, crop_sha256=sha256(file),
            faces=w.call('surfaceSeamFaces')[16:18]))

    def save(self, phase, obs):
        self.record['frames'].append(phase)
        if phase == 'loaded':
            f = self.srcframes[0]
        elif phase == 'work' and self.w.tick == self.last['tick']:
            f = self.last
        else:
            return
        token = f['key']+'.npz'
        if f['phase'] != phase or self.w.tick != f['tick']:
            raise ValueError('frame sequence did not match')
        for kind in ('sensor','structure','high'):
            if sha256(self.src/kind/token) != f[kind+'_sha256']:
                raise ValueError('changed source data')
        with np.load(self.src/'sensor'/token) as raw, np.load(self.src/'structure'/token) as st:
            edges = np.asarray(st['edges'])
            if (not np.array_equal(raw['rgb'],self.w.image) or not np.array_equal(raw['depth'],self.w.depth.astype('<f4'))
                    or not np.array_equal(st['nodes'],self.w.truth.nodes()) or not np.array_equal(f['tcp'],self.tcp(self.cutter))
                    or not np.array_equal(edges,np.asarray(self.w.call('tissueConstraints')).reshape(-1,3))):
                raise ValueError('unmatched loaded/working state')
        self.exposure(phase, self.src/'high'/token)
        if phase == 'loaded':
            return
        if not self.w.jaws_closed() or not self.w.clear_view(.003):
            raise ValueError('paired backoff not accepted')
        after = self.w.observe()
        if (after.faults or after.torn or not self.tension_ok(after) or not self.grasps_ok(after)
                or not self.w.jaws_closed() or not np.array_equal(edges,np.asarray(self.w.call('tissueConstraints')).reshape(-1,3))
                or not np.array_equal(self.tcp(self.cutter), self.clear_ref['after_tcp'])):
            raise ValueError('backoff not reproduced')
        self.exposure('backoff', self.clear_path)
        raise Finished()


def main(args):
    verify_lock()
    spec = json.loads(SPEC.read_bytes())
    decision, src = DEV/'decision.json', DEV/'slide'/args.case
    case = next(r for r in json.loads(decision.read_bytes())['runs'] if r['case']==args.case)
    clear = json.loads((CLEAR/args.case/'probe.json').read_bytes())
    if (sha256(decision) != spec['source_decision_sha256'] or not case['mechanics_ok'] or
            sha256(src/'oracle.json') != case['oracle_sha256'] or not clear['pose_qualified'] or
            sha256(CLEAR/args.case/'high.npz') != clear['high_sha256']):
        raise ValueError('unqualified source')
    r = json.loads((src/'oracle.json').read_bytes())
    args.scene = ROOT/'scenes/davinci_si_articulated_verified.ttt'
    args.initial_view, args.initial_view_insertion_m = r['view']['target'][:2], r['view']['target'][2]
    folder = OUT/args.case
    folder.mkdir(parents=True, exist_ok=False)
    js = dict(criteria=spec, plan_sha256=sha256(SPEC), code_sha256=sha256(Path(__file__)), frames=[], observations=[],
              exposures=[], complete=False, ref_sha256=sha256(src/'oracle.json'), visual_control=False)
    original = slide.staged_source

    def add_probe(parent):
        old = original(parent)
        if hashlib.sha256(old.encode()).hexdigest()!=r['controller_sha256']:
            raise ValueError('unregistered underlying controller')
        return old+'\n'+LUA

    client = sim = None
    try:
        client, sim = connect(args)
        if sha256(args.scene) != r['scene_sha256']:
            raise ValueError('wrong base scene')
        with patch.object(slide, 'staged_source', add_probe):
            w = slide.SeamSlideWorld(client,sim,args,r['cfg'],mode='full')
        js['probe_controller_sha256'] = w.controller_sha256
        task = FaceCapture(w,r['cfg'],folder,js,False)
        task.colors = np.array(spec['ink'])
        task.src, task.srcframes = src, r['frames']
        task.last = [f for f in r['frames'] if f['phase']=='work'][-1]
        task.clear_ref, task.clear_path = clear, CLEAR/args.case/'high.npz'
        try:
            task.run()
        except Finished:
            js['complete'] = True
    except Exception as exc:
        js['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            stop_simulation(client,sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        write_new(folder/'probe.json',js)
    print(args.case, js.get('error') or js['complete'])
    return int(not js['complete'])


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--case', choices=('a','b'), required=True)
    raise SystemExit(main(ap.parse_args()))
