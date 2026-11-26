"""Reproduce the primary look, then spend at most one closed ECM observation."""
import argparse
from pathlib import Path
import sys

import numpy as np

import run_m5_v3_outer_positive_v5 as base
from m5_v4_frontier_v1 import Reviews

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from probe_m5_v4_review_v1 import OUT, PARENT, PLAN, check, checked, inputs, validate_manifest


class CapturePort(base.LivePort):
    def __init__(self,*args,reference):
        super().__init__(*args)
        self.reference = reference
        self.locked = False
        self.reviews = Reviews(self.report['run_id'])

    def set_open(self):
        check(not self.locked and self.closures<4,'no further opening')
        return super().set_open()

    def cut(self):
        check(not self.locked and self.closures<4,'no further closure')
        return super().cut()

    def move(self,q):
        check(not self.locked and not np.allclose(q,self.ready_q,atol=1e-6,rtol=0),
              'probe forbids terminal withdrawal and post-look motion')
        return super().move(q)

    def receipt(self,f,alternate):
        key = self.optics.phase
        self.report['online_reviews'][key] = self.reviews.observe(
            self.report['run_id'],f.tick,self.closures,self.closed(),f.rgb,f.depth,far=f.far,alternate=alternate)

    def capture(self,narrow):
        check(not self.locked,'unexpected exposure after primary fourth look')
        f = super().capture(narrow)
        key = self.optics.phase
        check(key in self.reference and self.report['captures'][key]['sha256']==self.reference[key]['sha256'],
              'optical trajectory differs from archive at '+key)
        if narrow:
            self.receipt(f,False)
            if self.closures==4:
                self.locked = True
        return f

    def alternate(self,target):
        r = self.report
        check(self.locked and self.reviews.pending and self.reviews.prefix==7 and
              self.closures==4 and self.optics.tick==18 and not r['attempts'],'unscheduled alternate')
        check(self.safe() and self.closed(),'unsafe primary hold')
        q = self.q()
        check(np.allclose(q,r['entrance_q'],atol=1e-6,rtol=0),'not at closed entrance waypoint')
        r['attempts'] += 1
        check(self.w.command(self.w.view,*target,25),'guarded alternative ECM move refused')
        check(self.safe() and self.closed() and np.array_equal(q,self.q()) and
              np.allclose(self.w.target_of(self.w.view),target,rtol=0,atol=1e-6),'alternate readback mismatch')
        self.optics.phase = '019_alternate'
        f = self.optics.capture(True)  # passive; do not restore the primary ECM pose
        check(self.safe() and self.closed() and np.array_equal(q,self.q()),'unstable alternate exposure')
        r['captures'][self.optics.phase].update(q=self.q().tolist(),closures=self.closures)
        self.receipt(f,True)


def snapshot(port,key,out):
    w = port.w
    check(w.truth.strict,'unsealed scoring boundary')
    with w.truth.purpose('scoring'):
        path = out/(key+'_edges.json')
        base.write(path,w.call('tissueConstraints'))
    return dict(constraints_sha256=base.sha256(path),ecm=w.target_of(w.view),closed=port.closed(),
                closures=port.closures,q=port.q().tolist(),moves=len(port.report['moves']))


def run(a):
    lock = validate_manifest()
    check(checked(PLAN,lock['sources'][PLAN.relative_to(base.REPO).as_posix()])==PLAN,'protocol mismatch')
    gate = base.load(OUT/'offline_gate.json')
    checked(OUT/'registration.json',gate['registration_sha256'])
    checked(OUT/'offline_predictions.json',gate['predictions_sha256'])
    check(gate['passed'] and all(gate['gates'].values()),'offline gate failed')
    ref = inputs(PARENT)
    a.out,a.case = OUT/'capture','nominal'
    plan,cfg,saved,_ = base.prepare(a)
    check(not base.listening() and a.launch_simulator.is_file(),'simulator or port unavailable')
    a.out.mkdir(exist_ok=False)
    (a.out/'scoring').mkdir()
    r = dict(run_id=a.run_id,registration_sha256=gate['registration_sha256'],
             gate_sha256=base.sha256(OUT/'offline_gate.json'),captures={},online_reviews={},scoring_sidecars={},
             moves=[],attempts=0,error=None,cleanup_error=None,authorizes_motion=False)
    client=sim=w=port=None
    try:
        with base.dedicated(a):
            try:
                client,sim = base.connect(a)
                w = base.Acquisition(client,sim,a,cfg,mode='full')
                w.truth.strict = True
                check(w.check_roles()[0] and w.call('homeRCM') is True,'role/home refused')
                w.steps(20)
                for q in w.task['planning']['command_sequence_checked'][:2]:
                    check(w.call('setRCMTarget',w.index[w.cutter],*q) is True,'ready motion refused')
                w.steps(30)
                check(w.command(w.view,*plan['views']['ecm_work'],25) and w.jaws_closed(),'initial fixed view refused')
                ready = w._tool_q(w.cutter)[0].copy()
                check(np.array_equal(ready,plan['fixture_geometry']['ready_q']),'different ready pose')
                base.place_sheet(w,plan['fixture_geometry'])
                with w.truth.purpose('safety_monitor'):
                    edges = w.call('tissueConstraints')
                    baseline = base.DamageBaseline(edges)
                    base.write(a.out/'uncut.json',edges)
                    r['uncut_sha256'] = base.sha256(a.out/'uncut.json')
                for arm in (w.primary,w.assist):
                    ok,why = w.grasp(arm,w.grasp_points[arm])
                    check(ok,arm+' grasp: '+why)
                for arm in (w.primary,w.assist):
                    check(w.retract(arm,saved['pull_per_arm_m']),arm+' loading refused')
                w.steps(30)
                port = CapturePort(w,a,plan,r,9,ready,baseline,reference=ref['captures'])
                world = base.World(port,9,a.model)
                fsm = base.FSM(world)
                for idx in range(400):
                    state = fsm.step()
                    check(state not in ('DONE','FAILED'),fsm.reason or 'unexpected terminal transition')
                    if idx==0:
                        check(state=='ALIGN' and len(fsm.trail)==2,'missing entrance')
                        r['entrance_q'] = fsm.trail[1].tolist()
                        world.capture(True,True)
                    if state=='REVIEW' and fsm.bites==4:
                        break
                else:
                    raise ValueError('approach budget exhausted')
                check(fsm.prefix==7 and len(fsm.trail)>=4 and
                      np.allclose(port.q(),fsm.trail[-1],rtol=0,atol=1e-6),'invalid review trail')
                for q in reversed(fsm.trail[1:-1]):
                    world.move(q,True)
                world.capture(True,True)
                r['before'] = snapshot(port,'before',a.out)
                port.alternate(lock['plan']['alternative'])
                r['after'] = snapshot(port,'after',a.out)
            finally:
                if w:
                    try:
                        w.hold()
                        if port:
                            r.update(closed=port.closed(),ready=port.ready(),closures=port.closures,
                                     rcm=w.call('getRCMStats'),truth_reads=w.truth.reads_by_purpose)
                    finally:
                        check(base.stop_simulation(client,sim) is not False,'simulation stop refused')
                elif sim:
                    check(base.stop_simulation(client,sim) is not False,'simulation stop refused')
    except Exception as e:
        r['error'] = f'{type(e).__name__}: {e}'
    finally:
        if base.listening():
            r['cleanup_error'] = 'dedicated listener did not close'
        base.write(a.out/'run.json',r)
    print('acquisition:',r['error'] or 'complete','alternate attempts:',r['attempts'],flush=True)
    return not r['error'] and not r['cleanup_error']


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    base.add_arguments(p)
    p.set_defaults(host='127.0.0.1',port=24763)
    p.add_argument('--launch-simulator',type=Path,required=True)
    sys.exit(0 if run(p.parse_args()) else 1)
