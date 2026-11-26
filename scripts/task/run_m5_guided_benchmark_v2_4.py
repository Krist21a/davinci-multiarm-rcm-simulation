"""Run one case in the frozen guided v2.4 integration sequence."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import numpy as np

import run_same_bite as parent
from m5_guided_controller_v2_4 import GuidedFSMV24, GuidedWorldV24
from run_cut_task import REPO, S, add_arguments, connect, scenario_events, sha256, stop_simulation
from run_m5_guided_benchmark_v1 import ACTIVE, CERT, PRINT, SHEET, SPEC as FIXTURE

SPEC = REPO/'config/task/m5_guided_benchmark_v2_4.json'
READER = REPO/'validation/task/m5_guided_bank_corridor_v3/registration.json'
REPLAY = REPO/'validation/task/m5_guided_bank_corridor_v4'
DIAG = REPO/'validation/task/m5_guided_benchmark_v2_1/integration_diagnostic.json'
REGRESSIONS = REPO/'validation/safety/regressions_m5_guided_benchmark_v2_4.json'
SCORER = REPO/'scripts/analysis/score_m5_guided_benchmark_v2_4.py'
ORDER = ('nominal','exact_10mm','depth_noise_0p5','occlusion_transient','slip_regrasp','relax')


def run(a):
    plan = json.loads(SPEC.read_bytes())
    readers = json.loads(READER.read_bytes())['sources']
    if (tuple(plan['case_order'])!=ORDER or plan['id']!='m5_guided_benchmark_v2_4'
            or sha256(READER)!=plan['reader_registration_sha256']
            or sha256(parent.SPEC)!=plan['base_protocol_sha256']
            or sha256(FIXTURE)!=plan['fixture_protocol_sha256']
            or any(sha256(REPO/k)!=v for k,v in readers.items())
            or (plan['max_recorded_waypoints'],plan['review_advance_cap_m'],plan['absolute_limit_m'])!=(8,.002,.010)):
        raise ValueError('unregistered integrated benchmark')
    replay,decision = REPLAY/'registration.json',REPLAY/'decision.json'
    if (sha256(replay)!=plan['reader_v4_replay_registration_sha256']
            or sha256(decision)!=plan['reader_v4_replay_decision_sha256']
            or sha256(DIAG)!=plan['v2_1_diagnostic_sha256']
            or not json.loads(decision.read_bytes())['passed']
            or any(sha256(REPO/k)!=v for k,v in json.loads(replay.read_bytes())['sources'].items())):
        raise ValueError('v4 offline qualification is missing or changed')
    checked = json.loads(REGRESSIONS.read_bytes())
    if not checked['success'] or checked['failures'] or checked['errors'] or checked['skipped']:
        raise ValueError('v2.4 requires passing full offline regressions')
    task = Path(__file__).resolve().parent
    files = (SPEC,Path(__file__).resolve(),FIXTURE,ACTIVE,PRINT,CERT,READER,replay,decision,DIAG,SCORER,
             REGRESSIONS,REPO/'scripts/analysis/score_m5_guided_benchmark_v2_1.py',
             REPO/'scripts/analysis/score_m5_guided_benchmark_v2_2.py',*(REPO/k for k in readers),
             *(task/n for n in ('tool_pose_active_set.py','m5_guided_controller_v2_4.py',
                 'm5_guided_controller_v2_2.py','m5_guided_controller_v2_1.py','m5_guided_controller_v2.py',
                 'capture_m5_guided_reverse_v1.py',
                 'capture_m5_guided_terminal_controls_v1.py','run_m5_guided_benchmark_v1.py',
                 'run_m5_retained_print_v7.py','run_m5_endpoint_review_v4.py','run_m5_sweep_reserve_v3.py',
                 'm5_guarded_post_bite_v2.py','live_active_look.py','seam_topology.py')),
             *(REPO/'scripts/vision'/k for k in ('m5_guided_prefix_v1.py','m5_guided_entry_bin_v1.py',
                                                 'm5_guided_ahead_window_v1.py','m5_guided_bank_corridor_v4.py')),
             REPO/'scripts/learn/capture_m5_temporal_coded.py',REPO/'src/instruments/articulation.lua',
             REPO/'config/rcm_config.json')
    for p in files:
        name = p.relative_to(REPO).as_posix()
        if name.startswith(('scripts/','src/','config/')) and checked['source_sha256'].get(name)!=sha256(p):
            raise ValueError('source changed since regressions: '+name)
    base = parent.register(a.out,plan['id'],files)
    if not (base['initial_view']==[4.5,0.,.08]==plan['initial_view']):
        raise ValueError('unregistered initial work view')
    reg = sha256(a.out/'registration.json')
    if ORDER.index(a.case)>=2:
        gate = json.loads((a.out/'positive_gate.json').read_bytes())
        if (not gate['passed'] or not gate['positive_cases_passed']
                or gate['registration']['registration_sha256']!=reg
                or gate['registration']['scorer_sha256']!=sha256(SCORER)
                or gate['predictions_sha256']!=sha256(a.out/'positive_predictions.json')
                or tuple(gate['report_sha256'])!=ORDER[:2]
                or any(sha256(a.out/(case+'.json'))!=h for case,h in gate['report_sha256'].items())):
            raise ValueError('both independently verified positives are required before any disturbance')
    for case in ORDER[:ORDER.index(a.case)]:
        prior = json.loads((a.out/(case+'.json')).read_bytes())
        if prior['registration_sha256']!=reg or prior['error'] or prior['cleanup_error']:
            raise ValueError('prior case did not complete under this registration')
    dest = a.out/(a.case+'.json')
    if dest.exists():
        raise FileExistsError(dest)
    exact = a.case=='exact_10mm'
    scenario,sigma,seed = base['cases']['nominal' if exact else a.case]
    cfg = json.loads((REPO/base['base']).read_bytes())
    cut = cfg['cut']
    if (cut['length_m'],cut['max_overcut_m'],cut['advance_m'])!=(.009,.001,.003):
        raise ValueError('native cut envelope changed')
    if exact:
        cut.update(length_m=.010,max_overcut_m=0.)
    cut['control_source'] = 'vision'
    cfg['sheet'] = SHEET.copy()
    cfg['perception'].update(mode='shadow',markers=True,depth_noise_m=sigma,seed=seed,material_pattern=True,
                             local_material_chart=True,front_recovery=True,predictive_view=True,
                             exclude_tcp_marker=True,recovery_view_after_cut=True,ray_marker_fit=True)
    a.initial_view,a.initial_view_insertion_m = base['initial_view'][:2],base['initial_view'][2]
    a.scene,a.scenario,a.run_id = REPO/base['scene'],scenario,uuid.uuid4().hex
    a.review_target = json.loads(FIXTURE.read_bytes())['review']['target']
    a.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    a.max_waypoints,a.reserve_m,a.capture = plan['max_recorded_waypoints'],.00015,None
    a.review_advance_cap_m = plan['review_advance_cap_m']
    a.evidence_out = a.evidence_out/a.case
    a.evidence_out.mkdir(exist_ok=False,parents=True)
    r = dict(run_id=a.run_id,case=a.case,scenario=scenario,config=cfg,registration_sha256=reg,
             observations=[],structures={},scoring_errors=[],error=None,cleanup_error=None)
    c=s=w=None

    def snapshot(label):
        try:
            with w.truth.purpose('scoring'):
                edges = w.truth._read('constraints',lambda:w.call('tissueConstraints'))
            path = a.out/f'{a.case}_{label}.npz'
            with path.open('xb') as fp:
                np.savez_compressed(fp,edges=np.asarray(edges))
            r['structures'][label] = dict(file=path.name,sha256=sha256(path),tick=w.tick)
        except Exception as err:
            r['scoring_errors'].append(f'{label}: {type(err).__name__}: {err}')

    try:
        c,s = connect(a)
        w = GuidedWorldV24(c,s,a,cfg,mode='full',events=scenario_events(cfg,scenario))
        if not w.canonical_seam or not w.truth.strict or not w.vision:
            raise ValueError('wrong integrated guided world')
        fsm = GuidedFSMV24(w,cfg)
        observe,retract,tool_command = w.observe,w.retract_scissors,w.tool_command

        def sample(**kwargs):
            obs = observe(**kwargs)
            r['observations'].append(dict(tick=w.tick,**asdict(obs)))
            return obs

        def capture_retraction():
            if w.withdrawal_tick==w.tick and w.jaws_closed():
                snapshot('pre_withdraw')
            return retract()

        def record_tool(arm,q,n=15):
            if arm!=w.cutter:
                return tool_command(arm,q,n)
            closed = w.jaws_closed()
            tick = w.tick
            ok = tool_command(arm,q,n)
            w.event_log.append(dict(event='guided_v21_set_tool_pose',tick=tick,state=fsm.state.name,
                                    q=np.asarray(q).tolist(),accepted=bool(ok),
                                    closed_before=closed,closed_after=w.jaws_closed()))
            return ok

        w.observe,w.retract_scissors,w.tool_command = sample,capture_retraction,record_tool
        fsm.run()
        r.update(fsm.report(),strict_truth=w.truth.strict,attempts_camera=fsm.attempts[S.CAMERA_ADJUST])
        if w.sheet_origin is not None:
            snapshot('final')
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
    print(a.case,r.get('final_state'),r['error'] or r.get('reason'),flush=True)
    return int(bool(r['error'] or r['cleanup_error']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--case',choices=ORDER,required=True)
    for k in ('out','evidence-out'):
        p.add_argument('--'+k,type=Path,required=True)
    return run(p.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
