"""One fresh nominal v3 outer-bank scene; no event injection or camera sweep."""
import argparse
from pathlib import Path
import uuid

import numpy as np

from capture_m5_v3_readonly_v4 import Acquisition, PassivePort, BASE, CAL, CAL_PIN, FIXTURE, GRID, REPO
from capture_m5_v3_readonly_v4 import add_arguments, check, connect, load, sha256, stop_simulation, write
from m5_v3_controller_v4 import World, FSM
from m5_v3_perception_v4 import intact_preflight, review
from m5_v3_rigid_fk import CutterFK
from seam_topology import material_state

PLAN = REPO/'config/task/m5_v3_positive_outer_v2.json'
OFFLINE = REPO/'validation/task/m5_v3_outer_bank_v1.json'


class LivePort:
    def __init__(self,w,a,plan,report,goal,ready):
        self.w,self.report,self.goal,self.ready_q = w,report,goal,ready.copy()
        self.views = plan['views']
        self.optics = PassivePort(w,a,plan,report,None)
        self.closures = 0

    def q(self):
        return self.optics.q()

    def grip(self):
        w = self.w
        names = w.arm_cfg[w.cutter]['grip_joints']
        check(len(names)==2,'missing blade joints')
        q = np.array([w.sim.getJointPosition(w.by[n]) for n in names])
        check(np.isfinite(q).all(),'invalid blade readback')
        return q

    def closed(self):
        return bool(np.allclose(self.grip(),0.,atol=1e-6,rtol=0))

    def safe(self):
        w = self.w
        stats = w.call('getRCMStats')
        with w.truth.purpose('grasp_signal_and_safety_monitor'):
            tissue = w.truth.tissue_stats()
        with w.truth.purpose('tension_until_m6'):
            tension = w.cut_line_tension()
        with w.truth.purpose('safety_monitor'):
            edges = material_state(w.call('tissueConstraints'))
        floor = w.cfg['tension']['low_strain_after_cut_start' if self.closures else 'low_strain']
        opened = bool(np.allclose(self.grip(),.5,atol=1e-6,rtol=0))
        check(self.closed() or opened,'indeterminate jaw state')
        check(tissue['exists'] and tissue['torn']==0,'missing or torn coupon')
        check(all(v['failures']==0 and v['max_drift_m']<=.0005 and v['collision_blocks']==0 for v in stats),
              'RCM or collision fault')
        check(all((tissue.get('grasped') or {}).get(arm,0)>=w.cfg['grasp']['min_nodes']
                  for arm in (w.primary,w.assist)),'lost grasp')
        check(floor<=tension<=w.cfg['tension']['high_strain'],'strain outside safety band')
        check(edges['farthest_row_end']<=min(10,self.goal+1),'safety row envelope exceeded')
        return True

    def limits(self):
        w = self.w
        q,ws = w._tool_q(w.cutter)
        return w._tool_limits(w.cutter,ws)

    def move(self,q):
        ok = self.w.tool_command(self.w.cutter,q)
        self.report['moves'].append(dict(q=np.asarray(q).tolist(),accepted=bool(ok),closed=self.closed()))
        return bool(ok)

    def ready(self):
        return self.closed() and bool(np.allclose(self.q(),self.ready_q,atol=1e-6,rtol=0))

    def camera(self,target):
        return False  # no unregistered adjustment views

    def _fixed_view(self,key):
        target = self.views['ecm_'+key]
        w = self.w
        if not np.allclose(w.target_of(w.view),target,atol=1e-6,rtol=0):
            check(self.closed() and self.safe(),'ECM move requires safe closed hold')
            check(w.command(w.view,*target,25),'fixed ECM move refused')
            check(self.closed() and self.safe() and np.allclose(w.target_of(w.view),target,atol=1e-6,rtol=0),
                  'fixed ECM readback refused')

    def capture(self,narrow):
        self._fixed_view('review' if narrow else 'work')
        name = f'{self.optics.tick+1:03d}_{"review" if narrow else "work"}'
        self.optics.phase = name
        f = self.optics.capture(narrow)
        if narrow:
            self.report['captures'][name]['q'] = self.q().tolist()
            self.report['captures'][name]['closures'] = self.closures
            if self.closures:
                self.report['captures'][name]['optical'] = review(f.rgb,f.depth,far=f.far,sigma=0.)
            else:
                self.report['captures'][name]['uncut'] = intact_preflight(f.rgb,f.depth,far=f.far)
        return f

    def set_open(self):
        w = self.w
        ok = w.call('setGrip',w.index[w.cutter],.5) is True
        if ok:
            w.steps(3)
        return ok and bool(np.allclose(self.grip(),.5,atol=1e-6,rtol=0))

    def cut(self):
        w = self.w
        check(self.safe() and not self.closed(),'closure precondition refused')
        ok = w.call('setGrip',w.index[w.cutter],0.) is True
        if ok:
            self.closures += 1
            w.steps(8)
        return ok and self.closed()

    def hold(self):
        self.w.hold()


def prepare(a):
    plan = load(PLAN)
    check(sha256(OFFLINE)==plan['outer_replay_sha256'],'unregistered outer replay')
    gate = load(OFFLINE)
    check(gate['passed'],'offline gate did not pass')
    for name,pin in gate['sources'].items():
        check(sha256(REPO/name)==pin,'offline source drift: '+name)
    check(sha256(FIXTURE)==plan['fixture_decision_sha256'] and load(FIXTURE)['passed'],'fixture seal')
    check(sha256(CAL)==CAL_PIN and load(CAL)['qualified'],'calibration seal')
    a.model = CutterFK(**{k:np.array(v) if isinstance(v,list) else v for k,v in load(CAL)['calibration'].items()})
    f_reg = FIXTURE.with_name('registration.json')
    check(sha256(f_reg)==load(FIXTURE)['registration_sha256'],'fixture registration seal')
    saved = load(f_reg)['plan']
    check(sha256(GRID)==saved['workspace_decision_sha256'],'candidate seal')
    a.fixture_translation_m = load(GRID)['selected']['translation_m']
    a.scene = REPO/saved['scene']
    a.run_id,a.scenario = uuid.uuid4().hex,'nominal'
    cfg = load(BASE)
    cfg['sheet'] = saved['fixture']
    cfg['perception'].update(mode='off',markers=True,exclude_tcp_marker=True,ray_marker_fit=True)
    cfg['cut']['length_m'] = plan['cases'][a.case]*.001
    return plan,cfg,saved,f_reg


def run(a):
    plan,cfg,saved,f_reg = prepare(a)
    goal = plan['cases'][a.case]
    files = (PLAN,OFFLINE,BASE,FIXTURE,f_reg,CAL,GRID,a.scene,Path(__file__).resolve(),
             *(REPO/n for n in ('scripts/task/m5_v3_controller_v4.py',
                                'scripts/task/capture_m5_v3_readonly_v4.py',
                                'scripts/task/sim_world.py','scripts/task/seam_topology.py',
                                'scripts/vision/m5_v3_perception_v4.py','scripts/vision/m5_v3_corridor_v4.py',
                                'src/tissue/cell_certificate_print.lua')))
    a.out.mkdir(exist_ok=False)
    seal = a.out/'registration.json'
    write(seal,dict(plan=plan,case=a.case,goal_mm=goal,run_id=a.run_id,
                    sources={f.relative_to(REPO).as_posix():sha256(f) for f in files}))
    r = dict(registration_sha256=sha256(seal),case=a.case,goal_mm=goal,run_id=a.run_id,
             captures={},moves=[],steps=[],failure=None,cleanup_error=None,state=None,closed=None,ready=None)
    client=sim=w=port=fsm=None
    try:
        client,sim = connect(a)
        w = Acquisition(client,sim,a,cfg,mode='full')
        w.truth.strict = True
        check(w.check_roles()[0] and w.call('homeRCM') is True,'role/home refused')
        w.steps(20)
        for q in w.task['planning']['command_sequence_checked'][:2]:
            check(w.call('setRCMTarget',w.index[w.cutter],*q) is True,'ready motion refused')
        w.steps(30)
        check(w.command(w.view,*plan['views']['ecm_work'],25) and w.jaws_closed(),'initial fixed view or jaws refused')
        ready = w._tool_q(w.cutter)[0].copy()
        w.place_sheet()
        for arm in (w.primary,w.assist):
            ok,why = w.grasp(arm,w.grasp_points[arm])
            check(ok,arm+' grasp: '+why)
        for arm in (w.primary,w.assist):
            check(w.retract(arm,saved['pull_per_arm_m']),arm+' loading refused')
        w.steps(30)
        port = LivePort(w,a,plan,r,goal,ready)
        world = World(port,goal,a.model)
        fsm = FSM(world)
        r['ready_q'] = ready.tolist()
        for n in range(400):
            state = fsm.step()
            r['steps'].append(dict(index=n,state=state,reason=fsm.reason,prefix_mm=fsm.prefix,bites=fsm.bites,
                                   last_capture=world.tick,closed=port.closed()))
            if n==0 and state=='ALIGN' and len(fsm.trail)==2:
                r['entrance_q'] = fsm.trail[1].tolist()
                world.capture(True,True)
                name = port.optics.phase
                check(r['captures'][name]['uncut']['passed'],'live uncut entrance-clear gate failed')
            if state in ('DONE','FAILED'):
                break
        r['state'] = fsm.state
        r['failure'] = fsm.reason
        if r['state'] not in ('DONE','FAILED'):
            r['failure'] = '400-step budget exhausted'
            r['state'] = 'FAILED'
        w.hold()
    except Exception as e:
        r['failure']=f'{type(e).__name__}: {e}'
        r['state']='FAILED'
        if w:
            w.hold()
    finally:
        if port:
            try:
                r.update(closed=port.closed(),ready=port.ready(),closures=port.closures,final_q=port.q().tolist())
                with w.truth.purpose('scoring'):
                    r['final_structure'] = material_state(w.call('tissueConstraints'))
                    r['final_tissue_stats'] = w.truth.tissue_stats()
                r['rcm'] = w.call('getRCMStats')
                r['truth_reads_by_purpose'] = w.truth.reads_by_purpose
            except Exception as e:
                r['scoring_error']=f'{type(e).__name__}: {e}'
        if sim:
            try:
                stop_simulation(client,sim)
            except Exception as e:
                r['cleanup_error']=str(e)
        write(a.out/'run.json',r)
    print(a.case,r['state'],r['failure'],flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--case',choices=('nominal','exact_10mm'),required=True)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
