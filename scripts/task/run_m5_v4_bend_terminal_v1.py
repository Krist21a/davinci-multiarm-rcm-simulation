"""Once-only optical acquisition of naturally cut hinge-coupon terminal controls."""
import argparse
from pathlib import Path

import numpy as np

import m5_v3_controller_v5 as conservative
import run_m5_v4_bend_pair_v1 as bend
from m5_v4_frontier_v1 import Reviews
from run_m5_v3_outer_positive_v5 import REPO, World, FSM, LivePort, check, load, sha256, write

base = bend.base
ID = 'm5_v4_bend_terminal_v1'
PLAN = REPO/f'config/task/{ID}.json'
OUT = REPO/f'validation/task/{ID}'
SUITE = REPO/f'validation/safety/regressions_{ID}.json'
SCORER = REPO/f'scripts/analysis/score_{ID}.py'
NOISE = REPO/'config/task/m5_v3_sensor_margin_calibration_v1.json'
REF = bend.OUT/'transverse_hinge_v1'


def verify(path,pin):
    check(sha256(path)==pin,'input drift: '+str(path))
    return path


def register():
    plan = load(PLAN)
    dec = load(verify(bend.OUT/'decision.json',plan['parent_decision_sha256']))
    check(dec['passed'] and all(dec['checks'].values()),'hinge uncut screen failed')
    parent = load(verify(bend.OUT/'registration.json',dec['registration_sha256']))
    bend.verify(parent)
    old = load(verify(REF/'run.json',dec['run_hashes'][REF.name]))
    sources = dict(parent['sources'])
    reg = load(SUITE)
    check(reg['success'] and not reg['skipped'] and reg['tests_run']>=493,'missing offline suite')
    for p in (PLAN,Path(__file__).resolve(),SCORER,REPO/'scripts/tests/test_m5_v4_bend_terminal_v1.py',
              REPO/'scripts/analysis/calibrate_m5_v3_sensor_margin_v2.py',NOISE,
              REPO/'scripts/vision/m5_v4_frontier_v1.py'):
        key = p.relative_to(REPO).as_posix()
        verify(p,reg['source_sha256'][key])
        sources[key] = sha256(p)
    for p in (SUITE,bend.OUT/'decision.json',bend.OUT/'registration.json',REF/'run.json',
              verify(REF/'work.npz',old['captures']['work']['sha256']),
              verify(REF/'scoring/work.npz',old['states']['work']['side_sha256'])):
        sources[p.relative_to(REPO).as_posix()] = sha256(p)
    OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',dict(plan=plan,sources=sources))


class Acquisition(bend.Coupon):
    def call(self,name,*args):
        # The only cutter grip verbs come from the guarded port, within its four-bite budget.
        if name=='setGrip' and args[0]==self.index[self.cutter]:
            check(getattr(self,'armed_verb',None)==args[1],'unscheduled cutter command')
            self.armed_verb = None
            return bend.SimWorld.call(self,name,*args)
        return super().call(name,*args)


class Port(LivePort):
    def set_open(self):
        check(self.closures<4 and self.closed(),'not a pre-bite opening')
        self.w.armed_verb = .5
        try:
            return super().set_open()
        finally:
            self.w.armed_verb = None

    def cut(self):
        check(self.closures<4,'fifth closure forbidden')
        self.w.armed_verb = 0.
        try:
            return super().cut()
        finally:
            self.w.armed_verb = None

    def move(self,q):
        check(not np.allclose(q,self.ready_q,rtol=0,atol=1e-6),'no final withdrawal')
        return super().move(q)

    def capture(self,narrow):
        before_view = self.w.target_of(self.w.view)
        stamp_before = self.w.sim.getSimulationTime()
        f = super().capture(narrow)
        name = self.optics.phase
        m = self.report['captures'][name]
        m.update(ecm=self.w.target_of(self.w.view),moves=len(self.report['moves']),closures=self.closures)
        if narrow:
            check(self.closed() and self.w.truth.strict,'review requires closed strict state')
            stamp = self.w.sim.getSimulationTime()
            if np.array_equal(before_view,m['ecm']):
                check(stamp_before==stamp,'motion during fixed-pose exposure')
            d = self.optics.a.out/'scoring'/(name+'.npz')
            with self.w.truth.purpose('scoring'):
                edges = self.w.call('tissueConstraints')
            check(stamp==self.w.sim.getSimulationTime(),'unbound narrow reference')
            with d.open('xb') as stream:
                np.savez_compressed(stream,edges=edges,q=self.q(),ecm=m['ecm'],closed=True,tick=f.tick,
                                    stamp=stamp,frame_sha256=m['sha256'],closures=self.closures)
            self.report['review_sidecars'][name] = dict(file='scoring/'+d.name,sha256=sha256(d))
        return f


class CollectionWorld(World):
    def __init__(self,port,model,cap):
        super().__init__(port,9,model,sigma=0.)
        self.cap = cap

    def sweep_room(self,p):
        room = conservative.World.sweep_room(self,p)  # fixed .250 mm, never the candidate envelope
        if p.chart.prefix_mm==7 and self.io.closed():
            self.io.report['fork'] = dict(tick=p.tick,moves=len(self.io.report['moves']),cap_m=self.cap)
            return min(room,self.cap+.00015)  # the FSM subtracts reserve after this bound
        return room


def run_live(w,a,plan,cfg_plan,saved,r):
    check(w.controller_sha256==load(REF/'run.json')['controller_sha256'],'changed staged hinge')
    check(w.check_roles()[0] and w.call('homeRCM') is True,'role/home refused')
    w.steps(20)
    for q in w.task['planning']['command_sequence_checked'][:2]:
        check(w.call('setRCMTarget',w.index[w.cutter],*q) is True,'ready refused')
    w.steps(30)
    check(w.command(w.view,*cfg_plan['views']['ecm_work'],25) and w.jaws_closed(),'initial ECM or jaws refused')
    ready = w._tool_q(w.cutter)[0].copy()
    check(np.array_equal(ready,cfg_plan['fixture_geometry']['ready_q']),'ready differs')
    base.place_sheet(w,cfg_plan['fixture_geometry'])
    with w.truth.purpose('safety_monitor'):
        uncut = w.call('tissueConstraints')
        baseline = base.DamageBaseline(uncut)
    write(a.out/'uncut.json',uncut)
    r['uncut_sha256'] = sha256(a.out/'uncut.json')
    for arm in (w.primary,w.assist):
        ok,why = w.grasp(arm,w.grasp_points[arm])
        check(ok,arm+' grasp: '+why)
    for arm in (w.primary,w.assist):
        check(w.retract(arm,saved['pull_per_arm_m']),arm+' loading refused')
    w.steps(30)
    port = Port(w,a,cfg_plan,r,9,ready,baseline)
    world = CollectionWorld(port,a.model,plan['last_advance_cap_m'][r['case']])
    fsm,ledger = FSM(world),Reviews(r['run_id'])
    try:
        for i in range(400):
            if fsm.state=='ALIGN':
                state = fsm.step()
                if state=='FAILED':
                    r['refusal'] = fsm.reason
                    break
                if i!=0:
                    continue
                check(state=='ALIGN' and len(fsm.trail)==2,'entrance not registered')
                check(r['captures']['001_work']['sha256']==load(REF/'run.json')['captures']['work']['sha256'],
                      'different hinge initial work frame')
                r['entrance_q'] = fsm.trail[1].tolist()
            else:
                check(fsm.state=='REVIEW' and 1<=fsm.bites<=4 and len(fsm.trail)>=4 and
                      np.allclose(port.q(),fsm.trail[-1],rtol=0,atol=1e-6),'unexpected review path')
                for q in reversed(fsm.trail[1:-1]):
                    world.move(q,True)
                fsm.trail = fsm.trail[:2]
            check(np.allclose(port.q(),r['entrance_q'],rtol=0,atol=1e-6),'wrong entrance pose')
            f = world.capture(True,True)
            name = port.optics.phase
            m = r['captures'][name]
            if fsm.bites==0 and not m['uncut']['passed']:
                r['refusal'] = 'uncut preflight refused'
                break
            rec = ledger.observe(r['run_id'],f.tick,port.closures,port.closed(),f.rgb,f.depth,far=f.far)
            r['receipts'][name] = rec
            if not fsm.bites:
                continue
            val,n = rec['verdict'],rec['prefix_mm']
            if val=='UNKNOWN' or (val,n)!=(m['optical']['verdict'],m['optical']['prefix_mm']):
                r['refusal'] = 'missing or disagreeing optical review'
                break
            if fsm.bites==4:
                if n in (8,9):
                    world.capture(False,True)
                break
            if (val,n)!=('BELOW',(3,5,7)[fsm.bites-1]):
                r['refusal'] = 'unregistered optical common path'
                break
            fsm.prefix,fsm.align_count,fsm.state = n,0,'ALIGN'
        else:
            r['refusal'] = 'step budget exhausted'
    finally:
        with w.truth.purpose('scoring'):
            write(a.out/'final_edges.json',w.call('tissueConstraints'))
            r['tissue'] = w.truth.tissue_stats()
        r.update(final_sha256=sha256(a.out/'final_edges.json'),rcm=w.call('getRCMStats'),closed=port.closed(),
                 ready=port.ready(),closures=port.closures,q=port.q().tolist(),truth=w.truth.reads_by_purpose)


def collect(a,case):
    a.case,a.variant,a.out = 'nominal','transverse_hinge_v1',OUT/case
    lock = load(OUT/'registration.json')
    for name,pin in lock['sources'].items():
        verify(REPO/name,pin)
    cfg_plan,cfg,saved,_ = base.prepare(a)
    check(not base.listening() and a.launch_simulator.is_file(),'simulator or port unavailable')
    a.out.mkdir(exist_ok=False)
    (a.out/'scoring').mkdir()
    r = dict(case=case,run_id=a.run_id,registration_sha256=sha256(OUT/'registration.json'),
             captures={},moves=[],scoring_sidecars={},review_sidecars={},receipts={},refusal=None,error=None)
    try:
        with base.dedicated(a):
            client,sim = base.connect(a)
            try:
                w = Acquisition(client,sim,a,cfg,mode='full')
                w.truth.strict = True
                try:
                    run_live(w,a,lock['plan'],cfg_plan,saved,r)
                finally:
                    w.hold()
            finally:
                base.stop_simulation(client,sim)
    except Exception as exc:
        r['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        r['port_free'] = not base.listening()
        write(a.out/'run.json',r)
    print(case,'closures',r.get('closures'),'stop',r['error'] or r['refusal'],flush=True)
    return (r['port_free'] and not r['error'] and r.get('closures')==4 and r.get('closed') and not r['ready']
            and r['tissue']['torn']==0 and all(v['failures']==v['collision_blocks']==0 and v['max_drift_m']<=.0005
                                               for v in r['rcm']))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base.add_arguments(parser)
    parser.set_defaults(host='127.0.0.1',port=24763)
    parser.add_argument('--launch-simulator',required=True,type=Path)
    args = parser.parse_args()
    register()
    for case in load(PLAN)['order']:
        if not collect(args,case):
            break
