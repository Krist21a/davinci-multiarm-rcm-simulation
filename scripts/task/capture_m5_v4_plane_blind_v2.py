"""Once-only blind board exposures in a disposable, never-started scene."""
import argparse
from pathlib import Path
import tempfile

import numpy as np

import capture_m5_v4_plane_v1 as old
from capture_m5_v4_plane_v1 import REPO, check, load, sha256, write

ID = 'm5_v4_plane_blind_v2'
PLAN = REPO/f'config/task/{ID}.json'
OUT = REPO/f'validation/task/{ID}'
REG = REPO/f'validation/safety/regressions_{ID}.json'
TERMINAL = REPO/'validation/task/m5_v4_bend_terminal_v1'


def checked(path,digest):
    check(sha256(path)==digest,'source drift: '+str(path))
    return path


def register():
    spec = load(PLAN)
    d = load(checked(old.OUT/'decision.json',spec['prior_decision']))
    hypoth = load(checked(old.OUT/'posthoc_dev_slope.json',spec['prior_hypothesis']))
    check(hypoth['decision_sha256']==spec['prior_decision'] and d['chosen'] is None,'unexpected predecessor')
    parent = load(checked(old.OUT/'registration.json',d['registration_sha256']))
    sources = dict(parent['sources'])
    past = parent['spec']['development']+parent['spec']['holdout']
    check(len(spec['blind'])==len({tuple(p) for p in spec['blind']})==12 and not any(
          p in past for p in spec['blind']),'reused blind pose')
    tests = load(REG)
    check(tests['success'] and not tests['skipped'] and tests['tests_run']>=500,'missing offline gate')
    for p in (PLAN,Path(__file__).resolve(),REPO/'scripts/analysis/score_m5_v4_plane_blind_v2.py',
              REPO/'scripts/analysis/transfer_m5_v4_plane_blind_v2.py',REPO/'scripts/tests/test_m5_v4_plane_blind_v2.py'):
        name = p.relative_to(REPO).as_posix()
        sources[name] = tests['source_sha256'][name]
    audit_reg = load(old.AUDIT/'registration.json')
    tdpath = TERMINAL/'decision.json'
    td = load(checked(tdpath,audit_reg['plan']['decision_sha256']))
    tlock = load(checked(TERMINAL/'registration.json',td['registration_sha256']))
    sources.update(tlock['sources'])
    for p in (REG,old.OUT/'registration.json',old.OUT/'decision.json',old.OUT/'posthoc_dev_slope.json',
              checked(old.OUT/'development.json',d['development_sha256']),tdpath,
              checked(TERMINAL/'optical_predictions.json',td['optical_sha256']),
              checked(TERMINAL/'measurements.json',td['measurements_sha256'])):
        sources[p.relative_to(REPO).as_posix()] = sha256(p)
    for k,h in sources.items():
        checked(REPO/k,h)
    OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',dict(spec=spec,sources=sources,authorizes_motion=False))
    return spec


def capture(args,spec,report):
    with old.dedicated(args):
        client,s = old.connect(args)
        cam = None
        try:
            check(s.getSimulationState()==s.simulation_stopped,'not a stopped instance')
            with tempfile.TemporaryDirectory(dir=args.stage_dir) as tmp:
                args.stage_dir = Path(tmp)
                old.load_scene(client,s,args,old.ARTICULATED_SCENE,'plane_blind.ttt')
            original = old.aliases(s)['EndoscopeCamera']
            before,stamp = s.getObjectMatrix(original,-1),s.getSimulationTime()
            cam, = s.copyPasteObjects([original],0)
            check(cam!=original,'sensor copy failed')
            s.setObjectParent(cam,-1,True)
            s.setObjectMatrix(cam,-1,[1,0,0,0,0,1,0,0,0,0,1,3])
            angle = s.visionfloatparam_perspective_angle
            s.setObjectFloatParam(cam,angle,float(np.deg2rad(70)))
            s.setIntArray2Property(cam,'resolution',[old.W,old.H])
            clips = (s.visionfloatparam_near_clipping,s.visionfloatparam_far_clipping)
            report['clipping'] = [s.getObjectFloatParam(cam,k) for k in clips]
            check(report['clipping']==[s.getObjectFloatParam(original,k) for k in clips] and
                  np.allclose(report['clipping'],[.002,.3],atol=1e-6,rtol=0) and
                  list(s.getVisionSensorRes(cam))==[old.W,old.H] and
                  abs(s.getObjectFloatParam(cam,angle)-np.deg2rad(70))<1e-6,'unregistered optics')
            for i,pose in enumerate(spec['blind']):
                name = f'blind_{i:02d}'
                report['frames'][name] = old.acquire(s,cam,OUT,name,pose)
                check(s.getSimulationState()==s.simulation_stopped and s.getSimulationTime()==stamp and
                      s.getObjectMatrix(original,-1)==before,'original scene or time changed')
            report['stationary'] = True
        finally:
            try:
                if cam is not None:
                    s.removeObjects([cam])
            finally:
                client.socket.close(linger=0)
                client.context.term()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    old.add_arguments(parser)
    parser.set_defaults(host='127.0.0.1',port=24763)
    parser.add_argument('--launch-simulator',type=Path,required=True)
    args = parser.parse_args()
    check(not old.listening() and args.launch_simulator.is_file(),'dedicated simulator unavailable')
    spec = register()
    args.out = OUT
    report = dict(registration_sha256=sha256(OUT/'registration.json'),frames={},stationary=False,error=None)
    try:
        capture(args,spec,report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['port_free'] = not old.listening()
        write(OUT/'capture.json',report)
    print('blind exposures:',len(report['frames']),'port free:',report['port_free'],flush=True)
