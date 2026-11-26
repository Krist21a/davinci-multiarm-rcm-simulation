"""Registered uncut A/B physics probe; no cutter FSM or cutter jaw commands."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

import run_m5_v3_outer_positive_v5 as base
from capture_m5_v3_readonly_v4 import BANKS, CERT, NATIVE, SimWorld
from m5_v4_bend_staging import HINGE, stage

ID = 'm5_v4_bend_pair_v1'
ROOT = base.REPO
PLAN = ROOT/f'config/task/{ID}.json'
OUT = ROOT/f'validation/task/{ID}'
REG = ROOT/f'validation/safety/regressions_{ID}.json'
REF = ROOT/'validation/task/m5_v3_positive_outer_v5/nominal'
SCORER = ROOT/'scripts/analysis/score_m5_v4_bend_pair_v1.py'
TEST = ROOT/'scripts/tests/test_m5_v4_bend_pair_v1.py'
PHASES = ('work','primary','alternate')


class Coupon(base.Acquisition):
    def start(self):
        if self.args.variant=='stock':
            return super().start()
        base.check(self.args.variant=='transverse_hinge_v1' and self.controller_sha256==NATIVE,
                   'unregistered coupon or controller')
        sim = self.sim
        source = stage(sim.getStringProperty(self.ctrl,'code').replace('\r\n','\n'))
        source += '\nlocal drawPrinted=render\n'+CERT.read_text(encoding='utf8')+BANKS.replace(
            'not T.seamMode','T.seamMode')
        sim.setStringProperty(self.ctrl,'code',source)
        base.check(sim.getStringProperty(self.ctrl,'code').replace('\r\n','\n')==source,'stage mismatch')
        self.controller_sha256 = hashlib.sha256(source.encode()).hexdigest()
        SimWorld.start(self)

    def call(self,name,*args):
        if name=='setGrip' and args[0]==self.index[self.cutter]:
            raise ValueError('no cutter grip commands in uncut probe')
        return super().call(name,*args)


def verify(lock):
    for path,pin in lock['sources'].items():
        base.check(base.sha256(ROOT/path)==pin,'changed source: '+path)
    base.check(lock['plan']==base.load(PLAN),'changed protocol')


def register():
    old = base.load(REF/'registration.json')
    past = base.load(REF/'run.json')
    base.check(base.sha256(REF/'registration.json')==past['registration_sha256'] and
               base.sha256(REF/'run.json')==base.load(REF.parent/'decision.json')['cases']['nominal']['run_sha256'],
               'reference registration or run drift')
    sources = dict(old['sources'])
    suite = base.load(REG)
    base.check(suite['success'] and not suite['skipped'] and suite['tests_run']>=491,'offline gates failed')
    for path in (PLAN,HINGE,ROOT/'scripts/task/m5_v4_bend_staging.py',SCORER,Path(__file__).resolve(),TEST):
        name = path.relative_to(ROOT).as_posix()
        sources[name] = suite['source_sha256'][name]
    for path in (REG,REF/'run.json',REF/'registration.json',REF.parent/'decision.json'):
        sources[path.relative_to(ROOT).as_posix()] = base.sha256(path)
    for rec in (past['captures']['001_work'],past['scoring_sidecars']['001_work']):
        sources[(REF/rec['file']).relative_to(ROOT).as_posix()] = rec['sha256']
    lock = dict(plan=base.load(PLAN),sources=sources)
    verify(lock)
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'registration.json',lock)


def capture(w,a,plan,saved,r):
    base.check(w.check_roles()[0] and w.call('homeRCM') is True,'role/home refused')
    w.steps(20)
    for q in w.task['planning']['command_sequence_checked'][:2]:
        base.check(w.call('setRCMTarget',w.index[w.cutter],*q) is True,'ready motion refused')
    w.steps(30)
    work = plan['views']['ecm_work']
    base.check(w.command(w.view,*work,25) and w.jaws_closed(),'initial view/closure refused')
    ready = np.asarray(plan['fixture_geometry']['ready_q'])
    base.check(np.array_equal(w._tool_q(w.cutter)[0],ready),'initial ready mismatch')
    base.place_sheet(w,plan['fixture_geometry'])
    with w.truth.purpose('safety_monitor'):
        initial = w.call('tissueConstraints')
        baseline = base.DamageBaseline(initial)
    for arm in (w.primary,w.assist):
        ok,why = w.grasp(arm,w.grasp_points[arm])
        base.check(ok,arm+' grasp: '+why)
    for arm in (w.primary,w.assist):
        base.check(w.retract(arm,saved['pull_per_arm_m']),arm+' loading refused')
    w.steps(30)
    port = base.LivePort(w,a,plan,r,9,ready,baseline)
    for phase,target in zip(PHASES,(work,plan['views']['ecm_review'],[-4.5,4.5,.080]),strict=True):
        base.check(port.safe() and port.ready(),'unsafe or non-ready before '+phase)
        if phase!='work':
            base.check(w.command(w.view,*target,25),'closed ECM command refused')
        base.check(port.safe() and port.ready() and np.allclose(w.target_of(w.view),target,atol=1e-6,rtol=0),
                   'view/grasp/mechanics refused')
        stamp = w.sim.getSimulationTime()
        port.optics.phase = phase
        f = port.optics.capture(phase!='work')
        with w.truth.purpose('scoring'):
            nodes,edges = w.truth.nodes(),w.call('tissueConstraints')
            tissue = w.truth.tissue_stats()
        with w.truth.purpose('tension_until_m6'):
            strain = w.cut_line_tension()
        base.check(np.array_equal(initial,edges) and tissue['cut']==tissue['torn']==tissue['seam_releases']==0,
                   'uncut structural veto')
        base.check(stamp==w.sim.getSimulationTime() and port.ready(),'unpaired exposure')
        side = a.out/'scoring'/(phase+'.npz')
        with side.open('xb') as stream:
            np.savez_compressed(stream,nodes_world=nodes,edges=np.asarray(edges),tick=f.tick,stamp=stamp,
                                q=port.q(),frame_sha256=r['captures'][phase]['sha256'])
        r['states'][phase] = dict(side_sha256=base.sha256(side),closed=port.closed(),ready=port.ready(),
                                  tissue=tissue,strain=strain,rcm=w.call('getRCMStats'),
                                  targets={k:w.target_of(k) for k in (w.primary,w.assist,w.cutter,w.view)})
    r['truth_reads'] = w.truth.reads_by_purpose


def run_case(a,variant):
    verify(base.load(OUT/'registration.json'))
    a.case,a.variant,a.out = 'nominal',variant,OUT/variant
    plan,cfg,saved,_ = base.prepare(a)
    base.check(not base.listening() and a.launch_simulator.is_file(),'dedicated RPC unavailable')
    a.out.mkdir(exist_ok=False)
    (a.out/'scoring').mkdir()
    r = dict(variant=variant,run_id=a.run_id,registration_sha256=base.sha256(OUT/'registration.json'),
             error=None,cleanup_error=None,captures={},states={})
    try:
        with base.dedicated(a):
            client,sim = base.connect(a)
            try:
                w = Coupon(client,sim,a,cfg,mode='full')
                w.truth.strict = True
                r['controller_sha256'] = w.controller_sha256
                try:
                    capture(w,a,plan,saved,r)
                finally:
                    w.hold()
            finally:
                base.stop_simulation(client,sim)
    except Exception as e:
        r['error'] = f'{type(e).__name__}: {e}'
    finally:
        if base.listening():
            r['cleanup_error'] = 'dedicated listener survived'
        base.write(a.out/'run.json',r)
    print(variant,r['error'] or 'captured',flush=True)
    return not r['error'] and not r['cleanup_error']


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    base.add_arguments(p)
    p.set_defaults(host='127.0.0.1',port=24763)
    p.add_argument('--launch-simulator',type=Path,required=True)
    args = p.parse_args()
    register()
    for variant in base.load(PLAN)['order']:
        if not run_case(args,variant):
            raise SystemExit(1)
