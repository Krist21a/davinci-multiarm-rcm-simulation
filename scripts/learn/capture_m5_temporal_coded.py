"""Record coded RGB-D around every closure on the unchanged nominal reference."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tip_data import REPO, write_json
import run_cut_task as task

PLAN = REPO / 'config/learn/m5_temporal_coded_capture_v1.json'
PRINT = REPO / 'src/tissue/cell_certificate_print.lua'
BANKS = '''
function drawTemporalMaterial()
    assert(T and T.nu==24 and T.nv==14 and not T.seamMode and not T.p.visualGap and not T.bridgeRefs)
    local original, bankIds, colors = T.colors, {}, {{0.86,0.42,0.48}}
    for side=0,1 do
        for j=0,12 do
            local r,k=32+48*(j%4),35+46*math.floor(j/4)
            colors[#colors+1]={r/255,(side==0 and 235 or k)/255,(side==0 and k or 235)/255}
        end
    end
    for idx,tr in ipairs(T.tri) do
        bankIds[idx]=tr[7]
        local i,j=math.floor((idx-1)/2)%23,math.floor((idx-1)/46)
        local side=(i>11 or (i==11 and idx%2==1)) and 1 or 0
        tr[7]=(i>=7 and i<=15) and (2+side*13+j) or 1
    end
    T.colors=colors
    local ok,result=pcall(setCellCertificatePrint,true)
    T.colors=original
    for idx,tr in ipairs(T.tri) do tr[7]=bankIds[idx] end
    if not ok then error(result) end
    return result
end
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    task.add_arguments(p)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(PLAN.read_bytes())
    base = REPO / 'config/task/cut_task.json'
    scene = task.INSTRUMENT_SCENE
    if (task.sha256(scene) != plan['scene_sha256']
            or task.sha256(base) != plan['base_config_sha256']):
        p.error('registered scene or base config changed')
    args.out.mkdir(exist_ok=False)
    frames = args.out / 'rgbd'
    frames.mkdir()
    reg = dict(protocol=plan, sources={task.relative(path): task.sha256(path) for path in (
        PLAN, PRINT, Path(__file__).resolve(), Path(task.__file__).resolve(),
        task.HERE / 'sim_world.py', task.HERE / 'cut_fsm.py',
        REPO / 'src/tissue/tissue.lua', REPO / 'config/endoscope/endoscope.json',
    )})
    lock = args.out / 'registration.json'
    write_json(lock, reg)
    manifest = dict(registration_sha256=task.sha256(lock), seed=None, attempts=[], calibration=None)

    class CaptureWorld(task.SimWorld):
        pending = None

        def start(self):
            if self.controller_sha256 != plan['native_controller_sha256']:
                raise ValueError('unexpected native controller')
            sim = self.sim
            base_source = sim.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n')
            source = base_source + '\nlocal drawPrinted=render\n' + PRINT.read_text(encoding='utf-8') + BANKS
            sim.setStringProperty(self.ctrl, 'code', source)
            if sim.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n') != source:
                raise ValueError('staged passive print verification failed')
            self.controller_sha256 = hashlib.sha256(source.encode()).hexdigest()
            super().start()

        def snapshot(self, name):
            s = self.sim
            camera = self.cam
            angle_param = s.visionfloatparam_perspective_angle
            angle = s.getObjectFloatParam(camera, angle_param)
            size = [self.W, self.H]
            if list(s.getVisionSensorRes(camera)) != size or abs(angle - math.radians(70)) > 1e-6:
                raise ValueError('unexpected native optics')
            rgb0, z0 = np.asarray(self.image).copy(), self.depth.copy()
            t0 = s.getSimulationTime()
            R, t = self.camera()
            row = dict(tick=self.tick, sim_time=t0, views={})
            try:
                if not self.call('drawTemporalMaterial'):
                    raise ValueError('passive print refused')
                for field, optic in plan['optics'].items():
                    w, h = res = optic['size']
                    a = math.radians(optic['hfov_deg'])
                    s.setObjectFloatParam(camera, angle_param, a)
                    s.setIntArray2Property(camera, 'resolution', res)
                    s.handleVisionSensor(camera)
                    rgb, sr = s.getVisionSensorImg(camera)
                    depth, dr = s.getVisionSensorDepth(camera, 1)
                    actual = s.getObjectFloatParam(camera, angle_param)
                    if list(sr) != res or list(dr) != res or abs(actual - a) > 1e-6:
                        raise ValueError('capture optics readback failed')
                    file = frames / f'{name}_{field}.npz'
                    with file.open('xb') as f:
                        np.savez_compressed(f, rgb=np.frombuffer(bytes(rgb), np.uint8).reshape(h, w, 3)[::-1],
                                            depth=np.frombuffer(bytes(depth), '<f4').reshape(h, w)[::-1],
                                            cam_R=R, cam_t=t, focal=w/2/math.tan(actual/2),
                                            far=self.far, angle_rad=actual)
                    row['views'][field] = dict(file=file.name, sha256=task.sha256(file))
            finally:
                s.setObjectFloatParam(camera, angle_param, angle)
                s.setIntArray2Property(camera, 'resolution', size)
                if not self.call('setCellCertificatePrint', False):
                    raise ValueError('passive print restore refused')
                self.render()
            R1, t1 = self.camera()
            if (s.getSimulationTime() != t0 or not np.array_equal(R1, R) or not np.array_equal(t1, t)
                    or list(s.getVisionSensorRes(camera)) != size
                    or abs(s.getObjectFloatParam(camera, angle_param) - angle) > 1e-6
                    or not np.array_equal(self.image, rgb0) or not np.array_equal(self.depth, z0)):
                raise ValueError('still frame or original RGB-D not restored')
            return row

        def close_scissors(self):
            if self.pending is not None or manifest['seed'] is None:
                raise ValueError('unpaired closure or lost loaded observation')
            seq = len(manifest['attempts']) + 1
            row = dict(attempt=seq, accepted=None, before=None, after=None)
            manifest['attempts'].append(row)
            row['before'] = self.snapshot(f'cut_{seq:02d}_before')
            ok = super().close_scissors()
            row['accepted'] = bool(ok)
            if ok:
                self.pending = row
            return ok

        def observe(self, **kw):
            obs = super().observe(**kw)
            if self.pending is not None:
                row, self.pending = self.pending, None
                if self.tick != row['before']['tick'] + 1 or not self.jaws_closed():
                    raise ValueError('post-closure frame not adjacent and closed')
                row['after'] = self.snapshot(f"cut_{row['attempt']:02d}_after")
            return obs

    class CaptureFSM(task.CutTaskFSM):
        def on_align(self, obs):
            if manifest['seed'] is None:
                arms = self.primary, self.assist
                if not all(obs.grasped.get(a, 0) >= self.cfg['grasp']['min_nodes'] for a in arms):
                    raise ValueError('loaded seed requires both grips')
                manifest['calibration'] = dict(tcp_local=self.w._tcp_local.tolist())
                manifest['seed'] = self.w.snapshot('loaded')
            super().on_align(obs)

    argv = ['--mode', 'full', '--scenario', 'nominal', '--control-source', 'truth',
            '--align-mode', 'tool_pose', '--trace', '--scene', str(scene), '--name', 'nominal',
            '--host', args.host, '--port', str(args.port), '--stage-dir', str(args.stage_dir),
            '--timeout', str(args.timeout)]
    try:
        with (patch.object(task, 'SimWorld', CaptureWorld), patch.object(task, 'CutTaskFSM', CaptureFSM),
              patch.object(task, 'OUT', args.out), patch.object(sys, 'argv', [task.__file__, *argv])):
            rc = task.main()
    finally:
        nominal = args.out / 'nominal.json'
        manifest['nominal_sha256'] = task.sha256(nominal) if nominal.exists() else None
        write_json(args.out / 'capture.json', manifest)
    print('capture:', args.out, flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
