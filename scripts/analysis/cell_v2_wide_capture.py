"""Replay one registered mechanical checkpoint and branch once for actual wide RGB-D."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from capture_surface_seam import Capture, ROOT, S, add_arguments, connect, sha256, stop_simulation, write_new
from local_hd_screen import expose
from planar_rim_pilot import staged
from replay_cell_v2 import DATA, NEAR, SOURCES, ZOOM, digest, load
from rest_cell_pilot import READBACK, World
from sim_world import SimWorld

PLAN = ROOT/'config/task/cell_certificate_v2_hd.json'
REFS = {'zero':(ZOOM,'pull',0), 'three':(NEAR,'near',1), 'six':(NEAR,'near',2),
        'eight':(NEAR,'near',3), 'nine':(ZOOM,'b',3), 'ten':(ZOOM,'c',3)}
OPTICS = dict(size=[1280,960], horizontal_fov_deg=70)


def register(out):
    sources = list(SOURCES)+[Path(__file__), Path(__file__).with_name('score_cell_v2_wide.py'),
                             Path(__file__).with_name('local_hd_screen.py'), PLAN, READBACK]
    for relative in ('scripts/task/sim_world.py','scripts/task/cut_fsm.py','scripts/task/ground_truth.py',
                     'scripts/build/simtools.py','scripts/vision/cut_perception.py','scripts/analysis/capture_surface_seam.py',
                     'scripts/analysis/planar_rim_pilot.py','scripts/analysis/rest_cell_pilot.py',
                     'src/tissue/cell_certificate_print.lua','src/tissue/legacy/tissue_v1.lua'):
        sources.append(ROOT/relative)
    data = dict(criteria=load(PLAN), sources={p.relative_to(ROOT).as_posix():digest(p) for p in sources},
                references={k:digest(DATA/s/'asym'/c/'oracle.json') for k,(s,c,_) in REFS.items()},
                narrow_replay=digest(DATA/'cell_certificate_v2_dev/decision.json'))
    sealed = out/'registration.json'
    if sealed.exists():
        if load(sealed) != data:
            raise ValueError('source drift; do not overwrite or resume this registration')
    else:
        write_new(sealed, data)
    return digest(sealed)


class MatchedWorld(World):
    """Use the already registered specimen bytes, not a mutable monkey-patch chain."""
    def start(self):
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            raise ValueError('requires stopped dedicated scene')
        base = sim.getStringProperty(self.ctrl, 'code').replace('\r\n','\n')
        parent = '\n'.join([base, *( (ROOT/'src/tissue'/name).read_text(encoding='utf-8')
                                    for name in ('ruler_print.lua','surface_seam.lua'))])
        code = staged(parent)+'\n'+READBACK.read_text(encoding='utf-8')
        self.controller_sha256 = hashlib.sha256(code.encode()).hexdigest()
        if self.controller_sha256 != '4af2af2da33785b5110a6d0201a3403ef5496dc14d0625fdf706fa81b3435cf5':
            raise ValueError('not the registered physical specimen')
        sim.setStringProperty(self.ctrl, 'code', code)
        if sim.getStringProperty(self.ctrl, 'code').replace('\r\n','\n') != code:
            raise ValueError('staged controller readback mismatch')
        SimWorld.start(self)


class Recorded(Exception):
    """A measurement fork is not a completed cutting task."""


class Probe(Capture):
    def save(self, phase, obs):
        if self.pull_only:
            self.take(phase, obs, True)
            return
        if phase == 'park':
            raise ValueError('missed registered working checkpoint')
        if phase != 'work' or self.bites != self.bite:
            return
        w = self.w
        self.take('work', obs, True)
        edges = list(w.call('tissueConstraints'))
        events = w.call('tissueStats')['cut_events']
        before = self.tcp(self.cutter)
        if not w.jaws_closed() or not w.clear_view(.003):
            raise ValueError('closed back-off refused')
        seen = w.observe(inject_events=False)
        after = self.tcp(self.cutter)
        residual = float(np.linalg.norm(after[:,3]-before[:,3]+.003*before[:,w.TOOL_AXIS_COL]))
        clearance = float(np.linalg.norm(after[:,3]-self.ready[:,3]))
        self.record.update(residual_m=residual, clearance_m=clearance)
        if (not residual <= .0005 or not clearance > .0005 or not self.grasps_ok(seen) or
                not self.tension_ok(seen) or seen.torn or seen.faults or not w.jaws_closed()
                or seen.cut_length_m != obs.cut_length_m or list(w.call('tissueConstraints')) != edges
                or w.call('tissueStats')['cut_events'] != events):
            raise ValueError('unqualified observation fork')
        self.take('look', seen, False)
        if not w.retract_scissors():
            raise ValueError('guarded return refused')
        final = w.observe(inject_events=False)
        if (not w.at_ready_pose() or not w.jaws_closed() or not self.grasps_ok(final) or final.torn
                or final.faults or not self.tension_ok(final) or list(w.call('tissueConstraints')) != edges
                or final.cut_length_m != obs.cut_length_m or w.call('tissueStats')['cut_events'] != events):
            raise ValueError('probe did not return closed and undamaged')
        self.record['closed_ready_return'] = True
        raise Recorded()

    def take(self, phase, obs, match):
        w = self.w
        super().save(phase, obs)
        r = self.record['frames'][-1]
        token = r['key']+'.npz'
        if match:
            f = next(x for x in self.ref['frames'] if x['phase'] == phase and x['bites'] == self.bites)
            if (f['tick'] != r['tick'] or not np.array_equal(f['tcp'],r['tcp'])
                    or not np.array_equal(w.camera()[0], self.ref['view']['applied_R'])
                    or not np.array_equal(w.camera()[1], self.ref['view']['applied_t'])):
                raise ValueError('reference pose/tick mismatch')
            for kind in ('sensor','structure'):
                p = self.source/kind/(f['key']+'.npz')
                if digest(p) != f[kind+'_sha256']:
                    raise ValueError('reference identity mismatch')
                with np.load(p) as old, np.load(self.folder/kind/token) as new:
                    if set(old.files)!=set(new.files) or any(not np.array_equal(old[k],new[k]) for k in old.files):
                        raise ValueError('checkpoint mismatch: '+kind)
            r['matched_key'] = f['key']
        r['camera'] = [v.tolist() for v in w.camera()]
        rgb, depth, time = np.asarray(w.image).copy(), w.depth.copy(), w.sim.getSimulationTime()
        with w.truth.purpose('scoring'):
            nodes = w.truth.nodes().copy()
            bonds = list(w.call('tissueConstraints'))
        try:
            if not w.call('setCellCertificatePrint', True):
                raise ValueError('registered print refused')
            w.render()
            expose(w, self.folder/'hd'/token, OPTICS)
        finally:
            if not w.call('setCellCertificatePrint', False):
                raise ValueError('original print restore refused')
            w.render()
        with w.truth.purpose('scoring'):
            if (not np.array_equal(w.image,rgb) or not np.array_equal(w.depth,depth)
                    or w.sim.getSimulationTime()!=time or not np.array_equal(nodes,w.truth.nodes())
                    or list(w.call('tissueConstraints'))!=bonds or not np.array_equal(r['tcp'],self.tcp(self.cutter))):
                raise ValueError('frozen print exposure altered checkpoint')
        r['hd_sha256'] = digest(self.folder/'hd'/token)


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lock = register(out)
    study, case, bite = REFS[args.case]
    refdir = DATA/study/'asym'/case
    ref = load(refdir/'oracle.json')
    parent = load(DATA/study/'decision.json')
    validated = parent['near'] if study == NEAR else next(r for r in parent['reports'] if r['case']==case)
    if not validated['mechanics_ok'] or digest(refdir/'oracle.json') != validated['oracle_sha256']:
        raise ValueError('not a validated mechanical reference')
    args.scene = ROOT/'scenes/davinci_si_articulated_verified.ttt'
    if sha256(args.scene) != ref['scene_sha256']:
        raise ValueError('scene mismatch')
    args.initial_view, args.initial_view_insertion_m = ref['view']['target'][:2], ref['view']['target'][2]
    dst = out/args.case
    dst.mkdir(exist_ok=False)
    for k in ('hd','sensor','structure'):
        (dst/k).mkdir()
    report = dict(registration_sha256=lock, reference_sha256=digest(refdir/'oracle.json'), frames=[],
                  observations=[],pull_motion_m={}, qualified=False)
    client = sim = None
    try:
        client, sim = connect(args)
        w = MatchedWorld(client,sim,args,ref['cfg'],mode='full')
        if len(w.markers)!=3 or w.controller_sha256!=ref['controller_sha256']:
            raise ValueError('fixture or instrument calibration mismatch')
        cal = w._dump_meta()['estimator_args']
        cal.update(W=1280,H=960,f=640/np.tan(np.deg2rad(35.)),ray_marker_fit=True)
        write_new(dst/'calibration.json',cal)
        report.update(calibration_sha256=digest(dst/'calibration.json'), controller_sha256=w.controller_sha256)
        fsm = Probe(w,ref['cfg'],dst,report,bite==0)
        fsm.ref, fsm.bite, fsm.source = ref, bite, refdir
        try:
            fsm.run()
            if not (bite==0 and fsm.state==S.DONE and [f['phase'] for f in report['frames']]==['loaded','pulled']):
                raise ValueError('probe incomplete: '+fsm.reason)
        except Recorded:
            if not bite or not report.get('closed_ready_return'):
                raise
        report['qualified'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if sim is not None:
                stop_simulation(client,sim)
        except Exception as exc:
            report['cleanup_error'] = str(exc)
        finally:
            if client is not None:
                client.socket.close(linger=0)
                client.context.term()
            write_new(dst/'oracle.json',report)
    print(args.case,report['qualified'],report.get('error',report.get('cleanup_error','')),flush=True)
    return int(not report['qualified'] or 'cleanup_error' in report)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--case', choices=REFS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
