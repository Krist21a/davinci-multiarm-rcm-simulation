"""Registered real sensor-resolution comparison at a matched parked checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_ruler_endpoint import BASE, PLAN, ROOT, CaptureFSM, RulerWorld, S
from simtools import add_arguments, connect, sha256, stop_simulation

OPTICS = ROOT/'config/task/ruler_endpoint_optics_v1.json'


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def capture(args):
    spec = json.loads(OPTICS.read_bytes())
    p = ROOT/f'validation/task/ruler_dev_c3_g{args.goal}/oracle.json'
    ref = json.loads(p.read_bytes())
    source = next(f for f in ref['frames'] if f['phase'] == 'park')
    setup = json.loads(PLAN.read_bytes())
    args.scene = ROOT/setup['scene']
    if (ref['error'] or ref.get('cleanup_error') or ref['final_state'] != 'DONE' or sha256(BASE) != ref['base_sha256']
            or sha256(args.scene) != ref['scene_sha256'] or sha256(PLAN) != ref['protocol_sha256']
            or any(sha256(ROOT/path) != h for path,h in ref['source_sha256'].items())):
        raise ValueError('invalid development reference')
    args.initial_view, args.initial_view_insertion_m = setup['view'][:2], setup['view'][2]
    target = next(x['target'] for x in spec['candidates'] if x['name'] == args.view)
    args.out.mkdir(exist_ok=False)
    report = dict(source_sha256=sha256(p),spec_sha256=sha256(OPTICS),code_sha256=sha256(Path(__file__)),
                  target=target,frames=[],error=None)
    client = sim = None
    try:
        client,sim = connect(args)
        w = RulerWorld(client,sim,args,ref['config'],mode='full')
        fsm = CaptureFSM(w,ref['config'])
        fsm.frames = []
        fsm.save = lambda phase, obs: fsm.frames.append(phase)
        fsm.run()
        if (fsm.state != S.DONE or w.tick != source['tick'] or w.controller_sha256 != ref['controller_sha256']
                or digest(np.asarray(w.image)) != source['rgb_sha256']
                or digest(w.depth.astype(np.float32)) != source['depth_sha256']
                or digest(w.truth.nodes()) != source['nodes_sha256'] or w.truth.cut_length() != source['length_m']):
            raise ValueError('reference park not exactly reproduced')
        time = sim.getSimulationTime()
        tcp = w.sim.getObjectMatrix(w.by[w.cutter+'_tcp'],-1)
        if args.view != 'native_pose' and not w.command(w.view,*target,0):
            raise ValueError('camera guard refused')
        report['target_readback'] = w.target_of(w.view)
        if not np.allclose(report['target_readback'],target,atol=1e-6,rtol=0):
            raise ValueError('camera target not reached')
        for scale in (1,4):
            W,H = 640*scale,480*scale
            if scale > 1:
                sim.setIntArray2Property(w.cam,'resolution',[W,H])
            if list(sim.getVisionSensorRes(w.cam)) != [W,H]:
                raise ValueError('sensor resolution readback failed')
            sim.handleVisionSensor(w.cam)
            raw,res = sim.getVisionSensorImg(w.cam)
            buf,dres = sim.getVisionSensorDepth(w.cam,1)
            if list(res) != [W,H] or list(dres) != [W,H]:
                raise ValueError('unexpected RGB-D dimensions')
            rgb=np.frombuffer(bytes(raw),np.uint8).reshape(H,W,3)[::-1].copy()
            depth=np.frombuffer(bytes(buf),dtype='<f4').reshape(H,W)[::-1].copy()
            if (sim.getSimulationTime()!=time or sim.getObjectMatrix(w.by[w.cutter+'_tcp'],-1)!=tcp
                    or digest(w.truth.nodes())!=source['nodes_sha256']):
                raise ValueError('physical state changed')
            if args.view=='native_pose' and scale==1 and (digest(rgb)!=source['rgb_sha256']
                                                          or digest(depth)!=source['depth_sha256']):
                raise ValueError('native optics did not reproduce')
            dst=args.out/f'scale{scale}.npz'
            np.savez_compressed(dst,rgb=rgb,depth=depth)
            report['frames'].append(dict(path=dst.name,shape=[H,W],sha256=sha256(dst)))
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            stop_simulation(client,sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        with (args.out/'capture.json').open('x',encoding='utf-8',newline='\n') as f:
            json.dump(report,f,indent=2)
            f.write('\n')
    print(args.goal,args.view,report['error'] or 'captured')
    return int(bool(report['error']))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--goal',required=True,type=int,choices=(8,9))
    ap.add_argument('--view',required=True,choices=('native_pose','alternative'))
    ap.add_argument('--out',required=True,type=Path)
    raise SystemExit(capture(ap.parse_args()))
