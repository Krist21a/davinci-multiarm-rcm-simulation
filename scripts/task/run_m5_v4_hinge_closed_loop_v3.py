"""Frozen v3; one fresh scene per case, using the original hinge mechanics and safety FSM."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import m5_v3_controller_v7 as v7
from m5_v4_hinge_marker_v3 import read, reader_for
from m5_v4_hinge_work_v3 import observe
import run_m5_v4_hinge_closed_loop_v1 as old
import run_m5_v4_hinge_closed_loop_v2 as v2
from run_m5_v4_hinge_closed_loop_v1 import ROOT, base, check, load, sha256, term, verify, write

ID = 'm5_v4_hinge_closed_loop_v3'
PLAN = ROOT/f'config/task/{ID}.json'
OUT = ROOT/f'validation/task/{ID}'
SUITE = ROOT/f'validation/safety/regressions_{ID}.json'


def register():
    spec = load(PLAN)
    matrix = load(verify(v2.OUT/'matrix.json',spec['v2_matrix_sha256']))
    parent = load(verify(v2.OUT/'registration.json',spec['v2_registration_sha256']))
    negative = load(verify(v2.OUT/'offline_gate.json',spec['v2_offline_sha256']))
    check(not negative['passed'] and not any(c['attempted'] for c in matrix['cases'].values())
          and negative['registration_sha256']==matrix['registration_sha256']==spec['v2_registration_sha256']
          and spec['cases']==parent['plan']['cases'],'wrong v2 predecessor')
    sources = dict(parent['sources'])
    files = [PLAN,Path(__file__).resolve(),ROOT/'scripts/vision/m5_v4_hinge_work_v3.py',
             ROOT/'scripts/vision/m5_v4_hinge_marker_v3.py',ROOT/f'scripts/tests/test_{ID}.py',
             ROOT/f'scripts/analysis/score_{ID}.py',ROOT/'scripts/analysis/qualify_m5_v4_hinge_v3.py',
             v2.OUT/'registration.json',v2.OUT/'matrix.json',v2.OUT/'offline_gate.json',
             verify(v2.OUT/'archive_measurements.json',negative['measurements_sha256']),
             verify(v2.OUT/'archive_predictions.json',negative['predictions_sha256']),v2.OUT/'posthoc_factorial.json']
    for p in files:
        sources[p.relative_to(ROOT).as_posix()] = sha256(p)
    for name,h in sources.items():
        verify(ROOT/name,h)
    OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',dict(plan=spec,inherited=parent['inherited'],sources=sources))


def lock():
    reg = load(OUT/'registration.json')
    for name,h in reg['sources'].items():
        verify(ROOT/name,h)
    check(reg['plan']==load(PLAN),'v3 plan drift')
    return reg


def work_pose(f,g,epoch):
    R,t = np.asarray(f.R),v7.vector(f.t,3)
    v7.require(np.allclose(R.T@R,np.eye(3),atol=1e-6) and np.linalg.det(R)>0,'invalid camera matrix')
    world_chart = v7.perception.WorkChart(g.prefix_mm,R@g.front+t,v7.unit(R@g.direction),v7.unit(R@g.opening),
                                          R@g.aim+t,R@g.limit+t)
    return v7.Work(world_chart,v7.vector(f.tip,3),v7.unit(f.jaw),v7.unit(f.hinge),f.tick,epoch)


class World(old.World):
    def work(self,closed,prefix_mm):
        f = self.capture(False,closed)
        self.io.work_prefix(prefix_mm)
        g,route = observe(f.rgb,f.depth,f.focal,f.far,self.sigma,prefix_mm)
        self.io.report['captures'][self.io.optics.phase]['chart_route'] = route
        if g is None or f.tip is None:
            self.io.work_estimate(None)
            self.work_refusal = f'work tick {f.tick}: '+('missing chart/inverse support' if g is None else 'missing marker')
            return None
        p = work_pose(f,g,self.epoch)
        self.io.work_estimate((p.chart.direction,p.chart.limit-p.tip))
        return p


class Port(old.Port):
    def capture(self,narrow):
        # Capture and noise are unchanged; do not mix tool poses across exposures.
        f = super().capture(narrow)
        if narrow:
            return f
        mk = read(reader_for(self.optics.a.model,f.focal,f.far),f.rgb,f.depth)
        self.report['captures'][self.optics.phase]['v3_marker'] = mk
        pose = [None]*3
        if mk['valid']:
            pose = [f.R@np.asarray(mk[k])+(f.t if k=='tip' else 0.) for k in ('tip','jaw_axis','opening_axis')]
        return replace(f,tip=pose[0],jaw=pose[1],hinge=pose[2])


def episode(w,a,cfg,saved,r):
    check(w.controller_sha256==load(term.REF/'run.json')['controller_sha256'] and w.truth.strict,'wrong hinge stage')
    check(w.check_roles()[0] and w.call('homeRCM') is True,'role/home check')
    w.steps(20)
    for q in w.task['planning']['command_sequence_checked'][:2]:
        check(w.call('setRCMTarget',w.index[w.cutter],*q) is True,'ready approach refused')
    w.steps(30)
    check(w.command(w.view,*cfg['views']['ecm_work'],25) and w.jaws_closed(),'initial view or jaws')
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
        check(w.retract(arm,saved['pull_per_arm_m']),'loading refused')
    w.steps(30)
    loaded = {arm:np.asarray(w.target_of(arm)) for arm in arms}
    port = Port(w,a,cfg,r,ready,base.DamageBaseline(uncut),rest,loaded)
    fsm = old.FSM(World(port,9,a.model,sigma=port.sigma))
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
    cases = lock()['plan']['cases']
    gate = load(OUT/'offline_gate.json')
    check(gate['passed'] and gate['registration_sha256']==sha256(OUT/'registration.json'),'offline gate refused')
    verify(SUITE,gate['suite_sha256'])
    verify(OUT/'archive_predictions.json',gate['predictions_sha256'])
    verify(OUT/'archive_measurements.json',gate['measurements_sha256'])
    for prev in cases[:cases.index(case)]:
        score = load(OUT/prev/'decision.json')
        verify(OUT/prev/'run.json',score['run_sha256'])
        check(score['passed'] if prev=='nominal' else score['mechanically_safe'],'batch interlock')
    a.case,a.variant,a.out = 'nominal','transverse_hinge_v1',OUT/case
    cfg,task,saved,_ = base.prepare(a)
    check(not base.listening() and a.launch_simulator.is_file(),'dedicated port/executable unavailable')
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
    except Exception as e:
        r['error'] = f'{type(e).__name__}: {e}'
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
