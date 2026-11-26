"""Sealed RGB-D acquisition on the translated, loaded, never-cut coupon."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import uuid

import numpy as np

from run_cut_task import REPO, add_arguments, connect, sha256, stop_simulation
from preflight_m5_v3_fixture_v1 import BASE, Fixture, load, write
from run_m5_guided_benchmark_v1 import BANKS, CERT, NATIVE
from sim_world import SimWorld, cut_perception as markers
from m5_v3_controller_v2 import RGBD, World
from m5_v3_rigid_fk import CutterFK
from m5_v3_perception_v2 import inks, review
from cell_certificate_v1 import decode

ID = 'm5_v3_readonly_v1'
SPEC = REPO/f'config/task/{ID}.json'
FIXTURE = REPO/'validation/task/m5_v3_fixture_preflight_v1/decision.json'
CAL = REPO/'validation/task/m5_v3_rigid_fk_v1.json'
CAL_PIN = '38b921173221d6c9dd1f2781d3e3ac09ac49fb6ae6452363afba62f91a6b5672'
GRID = REPO/'validation/task/m5_v3_distal_workspace_v2/decision.json'


class Acquisition(Fixture):
    def start(self):
        if self.controller_sha256!=NATIVE:
            raise ValueError('unexpected stock controller')
        s = self.sim
        base = s.getStringProperty(self.ctrl,'code').replace('\r\n','\n')
        src = base+'\nlocal drawPrinted=render\n'+CERT.read_text(encoding='utf-8')+BANKS.replace(
            'not T.seamMode','T.seamMode')
        s.setStringProperty(self.ctrl,'code',src)
        if s.getStringProperty(self.ctrl,'code').replace('\r\n','\n')!=src:
            raise ValueError('print staging mismatch')
        self.controller_sha256 = hashlib.sha256(src.encode()).hexdigest()
        SimWorld.start(self)

    def frame(self, arm):
        if arm!=self.cutter:
            return super().frame(arm)
        m = self.args.model.pose(self._tool_q(arm)[0])
        p,_ = self.args.model.blade_local()[0]
        yaw = m[:,:3]@self.args.model.tip_R.T
        return dict(pivot=m[:,3]+m[:,:3]@p,x=yaw[:,1],axis=yaw[:,2],tcp=m[:,3])

    def _create_markers(self):
        yaw = self.by[self.cutter+'_dvrk_yaw_frame']
        model = self.args.model
        self._tcp_local = model.tip_Y
        self._jaw_axis_local = model.tip_R[:,2]
        self._opening_axis_local = model.tip_R[:,0]
        self._marker_tcp_len = float(model.tip_Y[1])
        self._marker_offsets_used = [list(v) for v in (markers.MARKER_DISTAL_OFF,markers.MARKER_PROX_OFF,
                                                      markers.MARKER_THIRD_OFF)]
        for color,off in zip((markers.MARKER_DISTAL_RGB,markers.MARKER_PROX_RGB,markers.MARKER_THIRD_RGB),
                             self._marker_offsets_used,strict=True):
            s = self.sim
            h = s.createPrimitiveShape(s.primitiveshape_spheroid,[2*markers.MARKER_RADIUS]*3,0)
            s.setShapeColor(h,None,s.colorcomponent_ambient_diffuse,list(color))
            s.setShapeColor(h,None,s.colorcomponent_emission,list(color))
            s.setObjectInt32Param(h,s.shapeintparam_static,1)
            s.setObjectInt32Param(h,s.shapeintparam_respondable,0)
            s.setObjectParent(h,yaw,True)
            s.setObjectPosition(h,yaw,off)
            self.markers.append(h)


def check(ok, why):
    if not ok:
        raise ValueError(why)


class PassivePort:
    def __init__(self,w,a,p,r,edges):
        self.w,self.a,self.p,self.r = w,a,p,r
        self.phase,self.tick = '',0
        self.edges = edges

    def closed(self):
        return self.w.jaws_closed()

    def safe(self):
        w = self.w
        rs = w.call('getRCMStats')
        with w.truth.purpose('grasp_signal_and_safety_monitor'):
            stats = w.truth.tissue_stats()
        with w.truth.purpose('tension_until_m6'):
            tension = w.cut_line_tension()
        return bool(stats['exists'] and stats['torn']==stats['cut']==0
                    and all(v['failures']==0 and v['max_drift_m']<=.0005 for v in rs)
                    and all((stats.get('grasped') or {}).get(arm,0)>=w.cfg['grasp']['min_nodes']
                            for arm in (w.primary,w.assist))
                    and w.cfg['tension']['low_strain']<=tension<=w.cfg['tension']['high_strain']
                    and np.array_equal(self.edges,np.asarray(w.call('tissueConstraints'))))

    def capture(self,narrow):
        self.tick += 1
        w,p,s = self.w,self.p,self.w.sim
        mode = 'review' if narrow else 'work'
        size = p['views'][mode+'_size']
        angle = np.deg2rad(p['views'][mode+'_deg'])
        W,H = size
        cam = w.cam
        param = s.visionfloatparam_perspective_angle
        before = s.getSimulationTime(),self.closed(),w.target_of(w.view),self.q().copy()
        old_size,old_angle = list(s.getVisionSensorRes(cam)),s.getObjectFloatParam(cam,param)
        try:
            s.setObjectFloatParam(cam,param,float(angle))
            s.setIntArray2Property(cam,'resolution',size)
            check(w.call('drawTemporalMaterial') is True,'coded print refused')
            s.handleVisionSensor(cam)
            im,a = s.getVisionSensorImg(cam)
            raw,b = s.getVisionSensorDepth(cam,1)
            check(list(a)==list(b)==size and abs(s.getObjectFloatParam(cam,param)-angle)<=1e-6,
                    'wrong image optics')
            rgb = np.frombuffer(bytes(im),np.uint8).reshape(H,W,3)[::-1]
            z = np.frombuffer(bytes(raw),'<f4').reshape(H,W)[::-1]
            R,t = w.camera()
        finally:
            s.setObjectFloatParam(cam,param,old_angle)
            s.setIntArray2Property(cam,'resolution',old_size)
            check(w.call('setCellCertificatePrint',False) is True,'print restore refused')
        check(before[:3]==(s.getSimulationTime(),self.closed(),w.target_of(w.view))
                and np.array_equal(before[3],self.q()),'exposure changed motion state')
        f = W/2/np.tan(angle/2)
        path = self.a.out/(self.phase+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream,rgb=rgb,depth=z,focal=f,far=w.far,cam_R=R,cam_t=t,q=self.q())
        self.r['captures'][self.phase]=dict(file=path.name,sha256=sha256(path),closed=self.closed(),
                                            tick=self.tick,fov_deg=p['views'][mode+'_deg'])
        fit = {}
        if not narrow:
            meta = w._dump_meta()
            meta['estimator_args'].update(W=W,H=H,f=f)
            fit, _ = markers.CutPerception.from_meta(meta)._scissors(rgb.astype(float),z)
            self.r['captures'][self.phase]['marker']=fit
        pose = [None]*3
        if fit.get('valid') and all(k in fit for k in ('tip','jaw_axis','opening_axis')):
            pose = [R@np.asarray(fit[k])+(t if k=='tip' else 0) for k in ('tip','jaw_axis','opening_axis')]
        return RGBD(self.tick,rgb,z,f,w.far,R,t,*pose)

    def q(self):
        return self.w._tool_q(self.w.cutter)[0]


def run(a):
    p = load(SPEC)
    check(sha256(FIXTURE)==p['fixture_decision_sha256'] and load(FIXTURE)['passed'],'fixture seal')
    check(sha256(CAL)==CAL_PIN,'calibration seal')
    cal = load(CAL)
    check(cal['qualified'],'calibration qualification')
    for name,pin in cal['sources'].items():
        check(sha256(REPO/name)==pin,'calibration source drift')
    a.model = CutterFK(**{k:np.array(v) if isinstance(v,list) else v for k,v in cal['calibration'].items()})
    saved = load(FIXTURE)
    f_reg = FIXTURE.with_name('registration.json')
    check(sha256(f_reg)==saved['registration_sha256'],'fixture registration seal')
    check(sha256(GRID)==load(f_reg)['plan']['workspace_decision_sha256'],'candidate seal')
    c = load(GRID)['selected']
    a.fixture_translation_m = c['translation_m']
    a.scene = REPO/load(f_reg)['plan']['scene']
    a.run_id,a.scenario = uuid.uuid4().hex,'nominal'
    cfg = load(BASE)
    cfg['sheet'] = load(f_reg)['plan']['fixture']
    cfg['perception'].update(mode='off',markers=True,exclude_tcp_marker=True,ray_marker_fit=True)
    q = {e['mm']:e['q'] for e in saved['steps'] if e['stage']=='ik' and e['converged']}
    a.out.mkdir(exist_ok=False)
    files = [SPEC,FIXTURE,f_reg,CAL,GRID,BASE,a.scene,CERT,Path(__file__).resolve(),
             *(REPO/name for name in ('scripts/task/m5_v3_controller_v2.py',
                'scripts/task/m5_v3_rigid_fk.py','scripts/task/preflight_m5_v3_fixture_v1.py',
                'scripts/task/sim_world.py','scripts/vision/m5_v3_corridor_v4.py',
                'scripts/vision/m5_v3_perception_v2.py','scripts/vision/m5_v3_perception.py',
                'scripts/learn/capture_m5_temporal_coded.py','scripts/vision/cut_perception.py'))]
    lock = a.out/'registration.json'
    write(lock,dict(plan=p,sources={k.relative_to(REPO).as_posix():sha256(k) for k in files}))
    r = dict(passed=False,authorizes_motion=False,captures={},work={},narrow={},failure=None,cleanup_error=None,
             registration_sha256=sha256(lock))
    client=sim=w=None
    try:
        client,sim = connect(a)
        w = Acquisition(client,sim,a,cfg,mode='full')
        w.truth.strict = True
        check(w.check_roles()[0] and w.call('homeRCM') is True,'role/home refused')
        w.steps(20)
        for ready in w.task['planning']['command_sequence_checked'][:2]:
            check(w.call('setRCMTarget',w.index[w.cutter],*ready) is True,'ready motion refused')
        w.steps(30)
        check(w.command(w.view,*p['views']['ecm_work'],25) and w.jaws_closed(),'work ECM or jaws refused')
        w.place_sheet()
        edges = np.asarray(w.call('tissueConstraints'))
        for arm in (w.primary,w.assist):
            ok,why = w.grasp(arm,w.grasp_points[arm])
            check(ok,arm+' grasp: '+why)
        for arm in (w.primary,w.assist):
            check(w.retract(arm,load(f_reg)['plan']['pull_per_arm_m']),arm+' loading refused')
        port = PassivePort(w,a,p,r,edges)
        world = World(port,9,a.model)

        def move(name,closed):
            mm = p['guarded_waypoints_mm'][name]
            check(port.safe() and port.closed()==closed,'pre-move monitor refused')
            check(w.tool_command(w.cutter,q[mm]) and np.allclose(port.q(),q[mm],atol=1e-6,rtol=0)
                    and port.safe() and port.closed()==closed,'guarded/readback motion refused: '+name)

        def work(name,closed):
            port.phase = name
            try:
                g = world.work(closed,0)
                margin = None if g is None else world.sweep_room(g)
                r['work'][name]=dict(valid=g is not None,chart=None if g is None else asdict(g.chart),
                                     sweep_room_m=margin,pass_work=margin is not None and margin>=.0002)
                return g
            except (ValueError,RuntimeError) as e:
                r['work'][name]=dict(valid=False,reason=str(e),pass_work=False)
                return None

        def narrow(name):
            port.phase = name
            f = world.capture(True,True)
            bank,(_cap,face) = inks(f.rgb),decode(f.rgb)
            good = np.isfinite(f.depth)&(f.depth>0)&(f.depth<f.far*.999)
            banks = np.bincount(bank[good&(bank>=0)],minlength=26)
            faces = np.bincount(face[good&(face>=0)],minlength=26)
            optical = review(f.rgb,f.depth,far=f.far,sigma=0.)
            r['narrow'][name]=dict(bank_px=banks.tolist(),face_px=faces.tolist(),optical=optical,
                                  pass_narrow=bool((banks>=8).all() and (faces>=4).all() and
                                                   optical['rows'] and optical['rows'][0]['status']=='INTACT'))

        move('entry_clear',True)
        move('align',True)
        closed_chart = work('work_closed',True)
        check(w.command(w.view,*p['views']['ecm_review'],25),'review ECM refused')
        narrow('narrow_before_backoff')
        move('entry_clear',True)
        narrow('narrow_entry_clear')
        check(w.command(w.view,*p['views']['ecm_work'],25),'work ECM return refused')
        move('align',True)
        check(w.call('setGrip',w.index[w.cutter],.5) is True,'opening refused')
        w.steps(3)
        move('advanced',False)
        opened_chart = work('work_open',False)
        if closed_chart is not None and opened_chart is not None:
            x,y = closed_chart.chart,opened_chart.chart
            front = float(np.linalg.norm(x.front-y.front))
            angle = float(np.rad2deg(np.arccos(np.clip(x.direction@y.direction,-1,1))))
            r['agreement']=dict(front_m=front,angle_deg=angle,passes=front<=.0004 and angle<=5.)
        check(port.safe(),'final no-cut or grasp monitor refused')
        r['passed'] = bool(r.get('agreement',{}).get('passes') and all(v['pass_work'] for v in r['work'].values())
                           and r['narrow']['narrow_entry_clear']['pass_narrow'])
        w.hold()
    except Exception as e:
        r['passed']=False
        r['failure']=f'{type(e).__name__}: {e}'
        if w:
            w.hold()
    finally:
        if sim:
            try:
                check(stop_simulation(client,sim) is not False,'simulator stop refused')
            except Exception as e:
                r['cleanup_error']=str(e)
                r['passed']=False
        write(a.out/'decision.json',r)
    print('passed',r['passed'],'failure',r['failure'],'cleanup error',r['cleanup_error'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
