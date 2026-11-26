"""Dedicated fresh scenes for the frozen hinge-coupon benchmark."""
import argparse
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from cut_perception import CutPerception
from m5_v4_hinge_controller_v1 import FSM, World
import run_m5_v4_bend_terminal_v1 as term
from run_m5_v4_bend_terminal_v1 import base, check, load, sha256, verify, write

ROOT = base.REPO
ID = 'm5_v4_hinge_closed_loop_v1'
PLAN = ROOT/f'config/task/{ID}.json'
OUT = ROOT/f'validation/task/{ID}'
REG = ROOT/f'validation/safety/regressions_{ID}.json'
INV = ROOT/'validation/task/m5_v4_plane_blind_v2'
ALT = [-4.5,4.5,.080]


def lock():
    r = load(OUT/'registration.json')
    for name,pin in r['sources'].items():
        verify(ROOT/name,pin)
    check(r['plan']==load(PLAN),'plan drift')
    return r


def register():
    plan = load(PLAN)
    ref = load(verify(term.OUT/'decision.json',plan['parent_terminal']))
    inv = load(verify(INV/'decision.json',plan['parent_inverse']))
    check(inv['transfer']['metric_passed'] and ref['checks']['independent_contrast'],'missing hinge screens')
    sources = dict(load(verify(term.OUT/'registration.json',ref['registration_sha256']))['sources'])
    suite = load(REG)
    check(suite['success'] and not suite['skipped'] and suite['tests_run']>=506,'missing regressions')
    for p in (PLAN,Path(__file__).resolve(),ROOT/'scripts/task/m5_v4_hinge_controller_v1.py',
              ROOT/'scripts/vision/m5_v4_hinge_work_v1.py',ROOT/f'scripts/analysis/score_{ID}.py',
              ROOT/f'scripts/tests/test_{ID}.py',ROOT/'scripts/vision/m5_v4_frontier_v1.py'):
        k = p.relative_to(ROOT).as_posix()
        sources[k] = suite['source_sha256'][k]
    for p in (REG,INV/'decision.json',term.OUT/'decision.json',term.OUT/'registration.json',
              term.REF/'run.json',verify(INV/'transfer_measurements.json',inv['transfer']['measurements_sha256']),
              verify(INV/'optical_predictions.json',inv['transfer']['predictions_sha256'])):
        sources[p.relative_to(ROOT).as_posix()] = sha256(p)
    for k,h in sources.items():
        verify(ROOT/k,h)
    OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',dict(plan=plan,sources=sources))


def marker_reader(model,f,far):
    return CutPerception(1280,960,f,far,model.rcm,tcp_local=model.tip_Y,jaw_axis_local=model.tip_R[:,2],
                         opening_axis_local=model.tip_R[:,0],ray_marker_fit=True)


def sensor(rgb,z,tick,sigma):
    if sigma:
        rng = np.random.default_rng(np.random.SeedSequence([1,2,tick]))
        return rgb,z+rng.normal(0.,sigma,z.shape)
    return rgb,z


class Port(term.Port):
    def __init__(self,w,a,plan,r,ready,baseline,rest,loaded):
        super().__init__(w,a,plan,r,9,ready,baseline)
        self.rest,self.loaded = rest,loaded
        self.case = r['case']
        self.sigma = .0005 if self.case=='depth_noise_0p5' else 0.
        self.injected=self.waited=self.alt=False
        self.shutter = None

    def trace(self,state,prefix,bites,epoch):
        self.report['trace'].append(dict(state=state,prefix=prefix,bites=bites,epoch=epoch,tick=self.optics.tick))

    def entrance(self,q):
        self.report['entrance_q'] = q.tolist()

    def work_prefix(self,prefix):
        self.report['captures'][self.optics.phase]['work_prefix'] = prefix

    def work_estimate(self,value):
        self.report['captures'][self.optics.phase]['work_estimate'] = None if value is None else dict(
            axis=value[0].tolist(),displacement=value[1].tolist())

    def log_receipt(self,prior,val,n):
        self.report['receipts'][self.optics.phase] = dict(prior=prior,verdict=val,prefix=n,alternate=self.alt)

    def move(self,q):
        if np.allclose(q,self.ready_q,atol=1e-6,rtol=0):
            rec = self.report['receipts'].get(self.optics.phase,{})
            check(rec.get('verdict')=='TARGET' and not self.alt and self.stationary(),'no fresh primary withdrawal permit')
        return base.LivePort.move(self,q)

    def stationary(self):
        return self.closed() and np.allclose(self.q(),self.report['entrance_q'],atol=1e-6,rtol=0)

    def set_open(self):
        return self.verb(.5)

    def cut(self):
        return self.verb(0.)

    def verb(self,value):
        check(self.closures<5 and self.safe() and self.closed()==(value!=0.),'grip/closure budget')
        self.w.armed_verb = value
        try:
            return base.LivePort.set_open(self) if value else base.LivePort.cut(self)
        finally:
            self.w.armed_verb = None

    def capture(self,narrow):
        f = super().capture(narrow)
        rgb,z = sensor(f.rgb,f.depth,f.tick,self.sigma)
        self.report['captures'][self.optics.phase].update(shutter=self.shutter is not None,sigma=self.sigma,alternate=self.alt)
        if self.sigma:
            self.report['captures'][self.optics.phase]['sensor_depth_sha256'] = hashlib.sha256(z.tobytes()).hexdigest()
        if narrow:
            check(self.stationary(),'review outside closed entrance')
        if not self.sigma:
            return f
        pose = [None]*3
        if not narrow:
            mk,_ = marker_reader(self.optics.a.model,f.focal,f.far)._scissors(rgb.astype(float),z)
            if mk['valid']:
                pose = [f.R@np.asarray(mk[k])+(f.t if k=='tip' else 0) for k in ('tip','jaw_axis','opening_axis')]
        return replace(f,rgb=rgb,depth=z,tip=pose[0],jaw=pose[1],hinge=pose[2])

    def alternative(self):
        check(not self.alt and self.stationary() and self.safe(),'unsafe or repeated alternate')
        self.alt = True
        self.views = dict(self.views,ecm_review=ALT)

    def wait_available(self):
        return not self.waited

    def wait_closed(self):
        check(not self.waited and self.safe() and self.stationary(),'wait refused')
        self.waited = True
        self.w.steps(3)
        if self.shutter is not None:
            self.w.sim.removeObjects([self.shutter])
            self.shutter = None
        check(self.safe() and self.stationary(),'unsafe wait')

    def signals(self):
        w = self.w
        with w.truth.purpose('grasp_signal_and_safety_monitor'):
            t = w.truth.tissue_stats()
        with w.truth.purpose('tension_until_m6'):
            strain = w.cut_line_tension()
        with w.truth.purpose('safety_monitor'):
            damage = self.baseline.score(w.call('tissueConstraints'))
        # Unlike cutting, restoring support may begin with low tension or a lost
        # assist grasp. No other mechanical or structural veto is suspended.
        check(self.stationary() and t['exists'] and not t['torn'] and base.within_envelope(damage,9) and
              all(v['failures']==v['collision_blocks']==0 and v['max_drift_m']<=.0005 for v in w.call('getRCMStats'))
              and np.isfinite(strain) and strain<=w.cfg['tension']['high_strain'],'recovery safety veto')
        held = {k:t['grasped'].get(k,0) for k in self.loaded}
        check(held[w.primary]>=w.cfg['grasp']['min_nodes'],'unavailable primary support')
        return dict(held=held,strain=strain,q=self.q().tolist(),closed=self.closed(),
                    grippers={arm:w.target_of(arm) for arm in self.loaded})

    def at_review(self,bites):
        check(self.stationary() and self.closures==bites and bites==1+sum(
            d['verdict']=='BELOW' for d in self.report['receipts'].values()),'unbound closure for review')
        if self.injected or bites!=2 or self.case in ('nominal','depth_noise_0p5'):
            return
        check(self.safe() and self.stationary(),'event outside safe hold')
        self.injected = True
        e = self.report['event']
        e.update(case=self.case,bites=bites,before=self.signals(),tick=self.optics.tick)
        if self.case=='occlusion_transient':
            self._fixed_view('review')
            sim = self.w.sim
            self.shutter = sim.createPrimitiveShape(sim.primitiveshape_cuboid,[.020,.020,.001],0)
            sim.setObjectInt32Param(self.shutter,sim.shapeintparam_static,1)
            sim.setObjectInt32Param(self.shutter,sim.shapeintparam_respondable,0)
            sim.setShapeColor(self.shutter,None,sim.colorcomponent_ambient_diffuse,[.5,.5,.5])
            sim.setObjectMatrix(self.shutter,self.w.cam,[1,0,0,0,0,1,0,0,0,0,1,.01])
        elif self.case=='slip_regrasp':
            self.w.call('tissueRelease',self.w.index[self.w.assist])
            self.w.steps(3)
        elif self.case=='relax':
            for arm in self.loaded:
                self.signals()
                delta = self.loaded[arm]-self.rest[arm]
                check(self.w.command(arm,*(self.loaded[arm]-2.5*delta),10),'relax injection refused')
        e['after'] = self.signals()

    def hold_status(self):
        s = self.signals()
        w = self.w
        if s['held'][w.assist]<w.cfg['grasp']['min_nodes']:
            return 'RECOVER_GRASP'
        if any(not np.allclose(s['grippers'][k],v,atol=1e-6,rtol=0) for k,v in self.loaded.items()):
            return 'RECOVER_TENSION'
        return 'READ'

    def recover(self,state):
        w,e = self.w,self.report['event']
        check('recovered' not in e,'recovery budget exhausted')
        before = self.signals()
        if state=='RECOVER_GRASP':
            check(before['held'][w.assist]<w.cfg['grasp']['min_nodes'] and
                  np.allclose(before['grippers'][w.assist],self.loaded[w.assist],rtol=0,atol=1e-6),'no registered slip')
            i = w.index[w.assist]
            for value in (.5,0.):
                self.signals()
                check(w.call('setGrip',i,value) is True,'in-place assist grasp refused')
                w.steps(3)
            check(w.call('tissueGrasp',i)>=w.cfg['grasp']['min_nodes'],'empty in-place regrasp')
            w.steps(10)
        else:
            check(before['held'][w.assist]>=w.cfg['grasp']['min_nodes'],'assist unavailable')
            for _ in range(4):
                for arm,q in self.loaded.items():
                    sig = self.signals()
                    check(sig['held'][w.assist]>=w.cfg['grasp']['min_nodes'],'assist lost during pull')
                    delta = q-self.rest[arm]
                    current = np.asarray(sig['grippers'][arm])
                    # Scale the original joint increment, preserving the registered pull direction.
                    rem = float((q-current)@delta/(delta@delta))
                    check(-1e-5<=rem<=2.501 and np.allclose(q-current,rem*delta,atol=1e-6,rtol=0),'unregistered gripper error')
                    if rem>1e-5:
                        check(w.command(arm,*(current+min(1.,rem)*delta),10),'reload refused')
                if self.hold_status()=='READ':
                    break
        check(self.hold_status()=='READ' and self.safe(),'support not recovered')
        e.update(recovered=self.signals(),recovery_tick=self.optics.tick,state=state)


def episode(w,a,cfg,saved,r):
    check(w.controller_sha256==load(term.REF/'run.json')['controller_sha256'] and w.truth.strict,'incorrect hinge stage')
    check(w.check_roles()[0] and w.call('homeRCM') is True,'roles/home failed')
    w.steps(20)
    for q in w.task['planning']['command_sequence_checked'][:2]:
        check(w.call('setRCMTarget',w.index[w.cutter],*q) is True,'ready approach refused')
    w.steps(30)
    check(w.command(w.view,*cfg['views']['ecm_work'],25) and w.jaws_closed(),'initial view/jaws')
    ready = w._tool_q(w.cutter)[0].copy()
    check(np.array_equal(ready,cfg['fixture_geometry']['ready_q']),'ready mismatch')
    r['ready_q'] = ready.tolist()
    base.place_sheet(w,cfg['fixture_geometry'])
    with w.truth.purpose('safety_monitor'):
        uncut = w.call('tissueConstraints')
    write(a.out/'uncut.json',uncut)
    r['uncut_sha256'] = sha256(a.out/'uncut.json')
    arms = (w.primary,w.assist)
    for arm in arms:
        ok,why = w.grasp(arm,w.grasp_points[arm])
        check(ok,'fixture grasp: '+why)
    rest = {arm:np.asarray(w.target_of(arm)) for arm in arms}
    for arm in arms:
        check(w.retract(arm,saved['pull_per_arm_m']),'registered loading refused')
    w.steps(30)
    loaded = {arm:np.asarray(w.target_of(arm)) for arm in arms}
    port = Port(w,a,cfg,r,ready,base.DamageBaseline(uncut),rest,loaded)
    fsm = FSM(World(port,9,a.model,sigma=port.sigma))
    try:
        fsm.run()
    finally:
        w.hold()
        with w.truth.purpose('scoring'):
            write(a.out/'final.json',w.call('tissueConstraints'))
            r['tissue'] = w.truth.tissue_stats()
        r.update(state=fsm.state,reason=fsm.reason,closures=port.closures,closed=port.closed(),ready=port.ready(),
                 q=port.q().tolist(),final_sha256=sha256(a.out/'final.json'),truth=w.truth.reads_by_purpose,
                 rcm=w.call('getRCMStats'),controller_sha256=w.controller_sha256)


def collect(a,case):
    spec = lock()['plan']
    gate = load(OUT/'offline_gate.json')
    check(gate['passed'] and gate['registration_sha256']==sha256(OUT/'registration.json'),'offline gate refused')
    order = spec['cases']
    if case!='nominal':
        for prev in order[:order.index(case)]:
            score = load(OUT/prev/'decision.json')
            verify(OUT/prev/'run.json',score['run_sha256'])
            check(score['passed'] if prev=='nominal' else score['mechanically_safe'],'batch prerequisite failed')
    a.case,a.variant,a.out = 'nominal','transverse_hinge_v1',OUT/case
    cfg,task,saved,_ = base.prepare(a)
    check(not base.listening() and a.launch_simulator.is_file(),'port or executable unavailable')
    a.out.mkdir(exist_ok=False)
    (a.out/'scoring').mkdir()
    r = dict(case=case,run_id=a.run_id,registration_sha256=sha256(OUT/'registration.json'),error=None,
             trace=[],moves=[],captures={},receipts={},scoring_sidecars={},review_sidecars={},event={})
    try:
        with base.dedicated(a):
            client,sim = base.connect(a)
            try:
                w = term.Acquisition(client,sim,a,task,mode='full')
                w.truth.strict = True
                episode(w,a,cfg,saved,r)
            finally:
                base.stop_simulation(client,sim)
    except Exception as exc:
        r['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        r['port_free'] = not base.listening()
        write(a.out/'run.json',r)
    print(case,r.get('state'),r.get('reason'),r['error'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base.add_arguments(parser)
    parser.set_defaults(host='127.0.0.1',port=24763)
    parser.add_argument('--case',choices=load(PLAN)['cases'])
    parser.add_argument('--launch-simulator',type=Path)
    args = parser.parse_args()
    if args.case:
        collect(args,args.case)
    else:
        register()
