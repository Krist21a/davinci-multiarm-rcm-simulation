"""From-zero full-length visual development trials. No indexed ruler or truth fallback."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import numpy as np

from run_cut_task import CutTaskFSM, REPO, S, add_arguments, completion_issues, connect, scenario_events, sha256, stop_simulation
from same_bite_world import SameBiteWorld

SPEC=REPO/'config/task/m5_live_same_bite_v1.json'


def put(path,data):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,indent=2,allow_nan=False,default=lambda x:np.asarray(x).tolist())
        f.write('\n')


def register(out, policy, extra_sources=()):
    plan=json.loads(SPEC.read_bytes())
    if sha256(REPO/plan['base'])!=plan['base_sha256'] or sha256(REPO/plan['scene'])!=plan['scene_sha256']:
        raise ValueError('scene/config differs')
    files=[Path(__file__),Path(__file__).with_name('same_bite_world.py'),SPEC,
           *(REPO/'scripts/task'/f for f in ('cut_fsm.py','sim_world.py','ground_truth.py','run_cut_task.py',
                                             'workspace.py','cutter_view.py','tool_pose.py','visibility.py')),
           *(REPO/'scripts/vision'/f for f in ('cut_perception.py','material_coordinate.py')),
           REPO/'src/tissue/tissue.lua',REPO/'src/endoscope/rcm_controller.lua']
    info=dict(schema=2,policy=policy,protocol=plan,
              sources={p.relative_to(REPO).as_posix():sha256(p) for p in (*files,*extra_sources)})
    out.mkdir(exist_ok=True,parents=True)
    if not (out/'registration.json').exists():
        put(out/'registration.json',info)
    if json.loads((out/'registration.json').read_bytes())!=info:
        raise ValueError('source change; use fresh directory')
    return plan


def run(args, *, world_class=SameBiteWorld, fsm_class=CutTaskFSM, active_spec=None,
        policy=None, extra_sources=(), sheet_cfg=None):
    policy=policy or ('same_bite' if active_spec is None else 'active_look')
    extra=() if active_spec is None else (active_spec, Path(__file__).with_name('run_active_look.py'),
                                          Path(__file__).with_name('live_active_look.py'))
    plan=register(args.out, policy, (*extra, *extra_sources))
    scenario,noise,seed=plan['cases'][args.case]
    dest=args.out/(args.case+'.json')
    if dest.exists():
        raise FileExistsError(dest)
    cfg=json.loads((REPO/plan['base']).read_bytes())
    cfg['cut']['control_source']='vision'
    if sheet_cfg is not None:
        cfg['sheet']=dict(sheet_cfg)
    cfg['perception'].update(mode='shadow',markers=True,depth_noise_m=noise,seed=seed,material_pattern=True,
                             local_material_chart=True,front_recovery=True,predictive_view=True,
                             exclude_tcp_marker=True,recovery_view_after_cut=True,ray_marker_fit=True)
    args.initial_view,args.initial_view_insertion_m=plan['initial_view'][:2],plan['initial_view'][2]
    args.scenario,args.scene=scenario,REPO/plan['scene']
    args.run_id=uuid.uuid4().hex
    report=dict(registration_sha256=sha256(args.out/'registration.json'),policy=policy,scenario=scenario,
                run_id=args.run_id,case=args.case,
                scope=plan['scope'],config=cfg,accepted=False,error=None,cleanup_error=None,observations=[])
    client=sim=w=fsm=None
    try:
        client,sim=connect(args)
        w=world_class(client,sim,args,cfg,mode='full',events=scenario_events(cfg,scenario))
        if not w.vision or not w.truth.strict or w.canonical_seam:
            raise ValueError('not conventional strict visual controller')
        fsm=fsm_class(w,cfg)
        observe=w.observe
        def sample(**kwargs):
            obs=observe(**kwargs)
            report['observations'].append(dict(tick=w.tick,**asdict(obs)))
            return obs
        w.observe=sample
        fsm.run()
        final=w.observe(inject_events=False)
        issues=completion_issues(final,cfg)
        with w.truth.purpose('scoring'):
            stats=w.truth.tissue_stats()
        closed,ready=w.jaws_closed(),w.at_ready_pose()
        q=w._tool_q(w.cutter)
        report.update(fsm.report(),final=asdict(final),strict_truth=w.truth.strict,truth_purposes=w.truth.reads_by_purpose,
                      actual_ready=ready,actual_closed=closed,ready_q=w._ready_q,
                      actual_q=None if q is None else q[0],tissue=stats,events=w.event_log,
                      perception=w.perc_log,controller_sha256=w.controller_sha256,
                      optical_observer=getattr(w, 'hd_optical', None))
        closures=[e['tick'] for e in w.event_log if e['event']=='closure']
        reacq=[e['tick'] for e in w.event_log if e['event']=='post_closure_reacquired']
        complete_sequence=bool(closures and len(closures)==len(reacq) and all(
            tick<seen<(closures[i+1] if i+1<len(closures) else w.tick+1) for i,(tick,seen) in enumerate(zip(closures,reacq))))
        marks=[e for e in w.event_log if e['event']=='closure']
        direct=all(e.get('direct_front') and e.get('scissors_valid') and e.get('fix_age') is not None
                   and e['fix_age']<=w._fix_max_age for e in marks)
        samples=report['observations']
        first=samples[0]['cut_length_m']
        peak=max(s['cut_length_m'] for s in samples)
        after=[s['cut_length_m'] for s in samples if reacq and s['tick']>=reacq[-1]]
        strict_ok=not any(e['event']=='perception_error' for e in w.event_log)
        ok=(fsm.state==S.DONE and not issues and first==0 and .009<=final.cut_length_m<=.010
            and peak<=.010+1e-12 and not any(s['torn'] or s['faults'] for s in samples) and direct
            and complete_sequence and len(set(after))==1 and closed and ready and strict_ok)
        report.update(accepted=bool(ok),checks=dict(issues=issues,closures=closures,reacquired=reacq,
                      sequence=complete_sequence,direct=direct,start_m=first,peak_m=peak,
                      over_target_m=max(0,peak-.009),over_limit_m=max(0,peak-.010)))
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            try:
                stop_simulation(client,sim)
            except Exception as exc:
                report.update(cleanup_error=str(exc),accepted=False)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        put(dest,report)
    print(args.case,report.get('final_state'),report['error'] or report.get('reason'),report.get('checks'),flush=True)
    return int(not report['accepted'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--case',required=True)
    raise SystemExit(run(ap.parse_args()))
