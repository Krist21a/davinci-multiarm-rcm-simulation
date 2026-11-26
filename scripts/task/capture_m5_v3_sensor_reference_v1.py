"""One scoring-reference acquisition on the frozen nominal v3 controller."""
import argparse
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

import run_m5_v3_outer_positive_v3 as base


PLAN = base.REPO/'config/task/m5_v3_sensor_margin_calibration_v2.json'
OFFLINE = base.REPO/'validation/task/m5_v3_work_denoise_v4/decision.json'
NOMINAL = base.REPO/'validation/task/m5_v3_positive_outer_v3/nominal/registration.json'
SCORER = base.REPO/'scripts/analysis/calibrate_m5_v3_sensor_margin_v2.py'


class ReferencePort(base.LivePort):
    def cut(self):
        base.check(self.closures<4,'reference collection forbids a fifth closure')
        return super().cut()

    def capture(self,narrow):
        if narrow:
            return super().capture(True)
        self._fixed_view('work')
        w = self.w
        base.check(w.truth.strict is True,'unsealed truth boundary')
        stamp,q,closed = w.sim.getSimulationTime(),self.q().copy(),self.closed()
        view = w.target_of(w.view)
        f = super().capture(False)
        with w.truth.purpose('scoring'):
            nodes = w.truth.nodes()
            tcp = w.truth.tcp(w.cutter)
        base.check(w.sim.getSimulationTime()==stamp and np.array_equal(q,self.q())
                   and closed==self.closed() and view==w.target_of(w.view),'reference not from the exposure state')
        name = self.optics.phase
        frame = self.report['captures'][name]
        path = self.optics.a.out/'scoring'/(name+'.npz')
        with path.open('xb') as out:
            np.savez_compressed(out,nodes_world=nodes,tcp_world=tcp,q=q,tick=f.tick,
                                simulation_time_s=stamp,closed=closed,frame_sha256=frame['sha256'])
        self.report['scoring_sidecars'][name] = dict(file='scoring/'+path.name,sha256=base.sha256(path))
        return f


def snapshot(w,port):
    with w.truth.purpose('scoring'):
        edges = np.asarray(w.call('tissueConstraints'),dtype='<i8').reshape(-1,3)
        structure = base.material_state(edges)
        tissue = w.truth.tissue_stats()
    return dict(closed=port.closed(),ready=port.ready(),closures=port.closures,final_q=port.q().tolist(),
                final_structure=structure,final_tissue_stats=tissue,rcm=w.call('getRCMStats'),
                constraints_sha256=hashlib.sha256(edges.tobytes()).hexdigest())


def checked_inputs(a):
    spec = base.load(PLAN)
    sources = {}
    for pins in (spec['sources'],base.load(OFFLINE.with_name('registration.json'))['sources'],
                 base.load(NOMINAL)['sources']):
        for name,pin in pins.items():
            base.check(base.sha256(base.REPO/name)==pin,'source drift: '+name)
            sources[name] = pin
    offline = base.load(OFFLINE)
    base.check(offline['passed'] and len(offline['gates'])==7 and all(offline['gates'].values())
               and base.sha256(OFFLINE.with_name('registration.json'))==offline['registration_sha256'],
               'v8 offline gate unavailable')
    base.check(base.sha256(OFFLINE.with_name('measurements.json'))==offline['measurements_sha256'],
               'v8 measurements seal mismatch')
    a.case = 'nominal'
    past,cfg,fixture,_ = base.prepare(a)
    base.check(past==base.load(NOMINAL)['plan'] and past['cases']['nominal']==9,'baseline nomination mismatch')
    for path in (PLAN,Path(__file__).resolve(),SCORER,OFFLINE.with_name('measurements.json')):
        sources[path.relative_to(base.REPO).as_posix()] = base.sha256(path)
    return past,spec,cfg,fixture,sources


def listen(a):
    with socket.socket() as probe:
        probe.settimeout(.2)
        return probe.connect_ex((a.host,a.port))==0


def run(a):
    base.check(a.host=='127.0.0.1' and a.port==24763,'dedicated local listener required')
    base.check(a.launch_simulator.is_file() and not listen(a),'executable missing or dedicated listener busy')
    past,spec,cfg,saved,sources = checked_inputs(a)
    a.out.mkdir(exist_ok=False)
    (a.out/'scoring').mkdir()
    seal = a.out/'registration.json'
    base.write(seal,dict(plan=past,calibration_plan=spec,case=a.case,goal_mm=9,run_id=a.run_id,sources=sources))
    r = dict(registration_sha256=base.sha256(seal),case=a.case,goal_mm=9,run_id=a.run_id,
             captures={},scoring_sidecars={},moves=[],steps=[],failure=None,cleanup_error=None,state=None)
    client=sim=w=port=process=None
    try:
        env = os.environ.copy()
        env['COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX'] = '_davinci_safety_review'
        with (a.out/'simulator.log').open('xb') as log:
            process = subprocess.Popen([str(a.launch_simulator),'-h',f'-GzmqRemoteApi.rpcPort={a.port}'],
                                       stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env)
            deadline = time.monotonic()+45
            while time.monotonic()<deadline:
                base.check(process.poll() is None,'dedicated simulator exited before RPC')
                if listen(a):
                    break
                time.sleep(.5)
            else:
                raise TimeoutError('dedicated RPC did not open')
        client,sim = base.connect(a)
        w = base.Acquisition(client,sim,a,cfg,mode='full')
        w.truth.strict = True
        base.check(w.check_roles()[0] and w.call('homeRCM') is True,'role/home refused')
        w.steps(20)
        for q in w.task['planning']['command_sequence_checked'][:2]:
            base.check(w.call('setRCMTarget',w.index[w.cutter],*q) is True,'ready motion refused')
        w.steps(30)
        base.check(w.command(w.view,*past['views']['ecm_work'],25) and w.jaws_closed(),
                   'initial fixed view or jaws refused')
        ready = w._tool_q(w.cutter)[0].copy()
        w.place_sheet()
        for arm in (w.primary,w.assist):
            ok,why = w.grasp(arm,w.grasp_points[arm])
            base.check(ok,arm+' grasp: '+why)
        for arm in (w.primary,w.assist):
            base.check(w.retract(arm,saved['pull_per_arm_m']),arm+' loading refused')
        w.steps(30)
        port = ReferencePort(w,a,past,r,9,ready)
        world = base.World(port,9,a.model)
        fsm = base.FSM(world)
        r['ready_q'] = ready.tolist()
        for n in range(400):
            state = fsm.step()
            r['steps'].append(dict(index=n,state=state,reason=fsm.reason,prefix_mm=fsm.prefix,bites=fsm.bites,
                                   last_capture=world.tick,closed=port.closed()))
            if n==0 and state=='ALIGN' and len(fsm.trail)==2:
                r['entrance_q'] = fsm.trail[1].tolist()
                world.capture(True,True)
                base.check(r['captures'][port.optics.phase]['uncut']['passed'],'live uncut entrance-clear gate failed')
            if state in ('DONE','FAILED'):
                break
        base.check(fsm.state=='DONE' and port.closures==4 and port.ready(),fsm.reason or 'nominal did not return ready')
        receipts = [f['optical'] for f in r['captures'].values() if 'optical' in f]
        base.check([[f['verdict'],f['prefix_mm']] for f in receipts]==spec['required_receipts'],
                   'nominal optical sequence differs')
        r['nominal_end'] = snapshot(w,port)
        r['nominal_move_count'] = len(r['moves'])
        world.move(fsm.trail[1],True)
        world.capture(False,True)
        world.open()
        world.capture(False,False)
        r['state'] = 'DONE'
    except Exception as exc:
        r['state'],r['failure'] = 'FAILED',f'{type(exc).__name__}: {exc}'
    finally:
        if w is not None:
            try:
                w.hold()
                if port is not None:
                    r['probe_end'] = snapshot(w,port)
                    r['truth_reads_by_purpose'] = w.truth.reads_by_purpose
            except Exception as exc:
                r['scoring_error'] = f'{type(exc).__name__}: {exc}'
        if sim is not None:
            try:
                base.stop_simulation(client,sim)
            except Exception as exc:
                r['cleanup_error'] = str(exc)
        if process is not None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process.stdin.close()
                base.check(not listen(a),'dedicated RPC did not close')
            except Exception as exc:
                r['cleanup_error'] = (r['cleanup_error'] or '')+str(exc)
        base.write(a.out/'run.json',r)
    print('reference collection:',r['state'],r['failure'],r['cleanup_error'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    base.add_arguments(parser)
    parser.set_defaults(host='127.0.0.1',port=24763)
    parser.add_argument('--launch-simulator',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
