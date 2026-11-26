"""Opt-in nominal guided benchmark v2; current RGB-D is the only endpoint permit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import numpy as np

import run_same_bite as parent
from m5_guided_controller_v2 import GuidedFSMV2, GuidedWorldV2
from run_m5_guided_benchmark_v1 import ACTIVE, CERT, PRINT, SHEET, SPEC as BENCHMARK
from run_cut_task import REPO, add_arguments, connect, scenario_events, sha256, stop_simulation, S

SPEC = REPO/'config/task/m5_guided_benchmark_v2_disturbances_v1.json'
READER = REPO/'validation/task/m5_guided_bank_corridor_v3/registration.json'
NOMINAL = REPO/'validation/task/m5_guided_benchmark_v2'
TEN = REPO/'validation/task/m5_guided_10mm_terminal_online_v1'
CASE_ORDER = ('depth_noise_0p5','occlusion_transient','slip_regrasp','relax')


def run(a):
    plan = json.loads(SPEC.read_bytes())
    n,t = NOMINAL/'registration.json',TEN/'registration.json'
    if (any(sha256(REPO/k)!=v for k,v in plan['pins'].items())
            or any(sha256(REPO/k)!=v for f in (n,t) for k,v in json.loads(f.read_bytes())['sources'].items())
            or not all(json.loads((d/'decision.json').read_bytes())['passed'] for d in (NOMINAL,TEN))
            or tuple(plan['cases'])!=CASE_ORDER):
        raise ValueError('unregistered disturbance matrix or prerequisite')
    readers = json.loads(READER.read_bytes())['sources']
    task = Path(__file__).resolve().parent
    files = (SPEC,Path(__file__).resolve(),BENCHMARK,ACTIVE,PRINT,CERT,n,t,
             NOMINAL/'decision.json',TEN/'decision.json',READER,
             *(REPO/p for p in readers),
             *(task/n for n in ('m5_guided_controller_v2.py','capture_m5_guided_reverse_v1.py',
                 'capture_m5_guided_terminal_controls_v1.py',
                 'run_m5_guided_benchmark_v1.py','run_m5_retained_print_v7.py','run_m5_endpoint_review_v4.py',
                 'run_m5_sweep_reserve_v3.py','m5_guarded_post_bite_v2.py','live_active_look.py','seam_topology.py')),
             REPO/'scripts/learn/capture_m5_temporal_coded.py',REPO/'scripts/vision/m5_guided_prefix_v1.py')
    base = parent.register(a.out,plan['id'],files)
    scenario,sigma,seed = plan['cases'][a.case]
    if plan['cases'][a.case]!=base['cases'][a.case]:
        raise ValueError('registered scenario inputs differ')
    dest = a.out/(a.case+'.json')
    if dest.exists():
        raise FileExistsError(dest)
    a.initial_view,a.initial_view_insertion_m = base['initial_view'][:2],base['initial_view'][2]
    a.scene,a.scenario,a.run_id = REPO/base['scene'],scenario,uuid.uuid4().hex
    a.review_target = json.loads(BENCHMARK.read_bytes())['review']['target']
    a.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    a.max_waypoints,a.reserve_m,a.capture = 8,.00015,None
    cfg = json.loads((REPO/base['base']).read_bytes())
    if cfg['cut']['length_m']!=.009 or cfg['cut']['max_overcut_m']!=.001:
        raise ValueError('registered cut envelope changed')
    cfg['sheet'] = SHEET.copy()
    cfg['cut']['control_source'] = 'vision'
    cfg['perception'].update(mode='shadow',markers=True,depth_noise_m=sigma,seed=seed,material_pattern=True,
                             local_material_chart=True,front_recovery=True,predictive_view=True,
                             exclude_tcp_marker=True,recovery_view_after_cut=True,ray_marker_fit=True)
    a.evidence_out.mkdir(exist_ok=False)
    r = dict(run_id=a.run_id,case=a.case,scenario=scenario,config=cfg,registration_sha256=sha256(a.out/'registration.json'),
             observations=[],error=None,cleanup_error=None)
    c=s=w=None
    try:
        c,s = connect(a)
        w = GuidedWorldV2(c,s,a,cfg,mode='full',events=scenario_events(cfg,scenario))
        if not w.canonical_seam or not w.truth.strict or not w.vision:
            raise ValueError('wrong nominal guided world')
        fsm = GuidedFSMV2(w,cfg)
        observe = w.observe
        def sample(**kwargs):
            obs = observe(**kwargs)
            r['observations'].append(dict(tick=w.tick,**asdict(obs)))
            return obs
        w.observe = sample
        fsm.run()
        r.update(fsm.report(),strict_truth=w.truth.strict,attempts_camera=fsm.attempts[S.CAMERA_ADJUST])
        if w.sheet_origin is not None:
            with w.truth.purpose('scoring'):
                edges = w.truth._read('constraints',lambda:w.call('tissueConstraints'))
            structure = a.out/(a.case+'_structure.npz')
            with structure.open('xb') as fp:
                np.savez_compressed(fp,edges=np.asarray(edges))
            r.update(structure_file=structure.name,structure_sha256=sha256(structure))
    except Exception as err:
        r['error'] = f'{type(err).__name__}: {err}'
    finally:
        try:
            if w is not None:
                r.update(events=w.event_log,actual_closed=w.jaws_closed(),actual_ready=w.at_ready_pose(),
                         truth_purposes=w.truth.reads_by_purpose,controller_sha256=w.controller_sha256)
        except Exception as err:
            r['error'] = r['error'] or f'{type(err).__name__}: {err}'
        if s is not None:
            try:
                stop_simulation(c,s)
            except Exception as err:
                r['cleanup_error'] = str(err)
        if c is not None:
            c.socket.close(linger=0)
            c.context.term()
        parent.put(dest,r)
    print(r.get('final_state'),r['error'] or r.get('reason'),flush=True)
    return int(bool(r['error'] or r['cleanup_error']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--case',choices=CASE_ORDER,required=True)
    for flag in ('out','evidence-out'):
        p.add_argument('--'+flag,type=Path,required=True)
    return run(p.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
