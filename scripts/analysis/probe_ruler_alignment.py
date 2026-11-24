"""Read-only, restored FK queries at the first unresolved fixture alignment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import capture_ruler_endpoint as acquisition
from ruler_entry_mount import EntryMountWorld
from sim_world import tool_pose
from simtools import add_arguments, connect, sha256, stop_simulation

SPEC=acquisition.ROOT/'config/task/ruler_align_probe_v1.json'
MOUNT=acquisition.ROOT/'config/task/ruler_entry_mount_v1.json'


def main(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    parent=json.loads(acquisition.PLAN.read_bytes())
    args.scene=acquisition.ROOT/parent['scene']
    args.initial_view,args.initial_view_insertion_m=parent['view'][:2],parent['view'][2]
    args.pivot_local=json.loads(MOUNT.read_bytes())['pivot_local_uvw_m']
    cfg=json.loads(acquisition.BASE.read_bytes())
    cfg['cut'].update(length_m=.008,control_source='truth')
    cfg['perception'].update(exclude_tcp_marker=True)
    result=dict(candidates=[],spec_sha256=sha256(SPEC),code_sha256=sha256(Path(__file__)),error=None)
    if sha256(args.scene)!=parent['scene_sha256'] or sha256(acquisition.BASE)!=parent['base_sha256']:
        raise ValueError('wrong scene or configuration')
    client=sim=None
    try:
        client,sim=connect(args)
        w=EntryMountWorld(client,sim,args,cfg,mode='full')
        fsm=acquisition.CutTaskFSM(w,cfg)
        while fsm.state not in (acquisition.S.DONE,acquisition.S.FAILED,acquisition.S.ALIGN):
            fsm.tick()
        if fsm.state != acquisition.S.ALIGN:
            raise ValueError('no ALIGN')
        obs=w.observe()
        if fsm.supervise(obs) or not w.open_scissors():
            raise ValueError('first alignment refused by supervisor')
        w.render()
        q0,ws=w._tool_q(w.cutter)
        def frozen():
            s=w.call('getRCMStats')[w.index[w.cutter]-1]
            return dict(time=sim.getSimulationTime(),tcp=sim.getObjectMatrix(w.by[w.cutter+'_tcp'],-1),
                        q=w._tool_q(w.cutter)[0].tolist(),grip=s['grip'],failures=s['failures'],cut=w.cut_length(),
                        torn=w.call('tissueStats')['torn'],nodes=hashlib.sha256(w.truth.nodes().tobytes()).hexdigest())
        before=frozen()
        limits=w._tool_limits(w.cutter,ws)
        target=w.cut_front()-.001*w.v
        result.update(tick=w.tick,baseline=before,target=target.tolist(),limits=[a.tolist() for a in limits],
                      controller_sha256=w.controller_sha256)
        for index,deg in [(4,0)]+[(i,d) for i in (4,5) for d in (-22.5,22.5,-45.,45.)]:
            guess=q0.copy()
            guess[index]+=np.radians(deg)
            row=dict(seed_dof=index,offset_deg=deg)
            original=w._tool_fk(w.cutter)
            def fk(q):
                try:
                    return original(q)
                except RuntimeError as err:
                    row.update(rejected_q=q.tolist(),fk_error=str(err))
                    raise
            try:
                sol=tool_pose.solve(fk,guess,limits,target,w.v,w.u,axis_col=w.TOOL_AXIS_COL,yaw_col=w.TOOL_YAW_COL)
                row.update(converged=sol['converged'],position_error_m=sol['pos_err_m'],ang_error_rad=sol['axis_err_rad'],
                           at_limit=sol['at_limit'],q=sol['q'].tolist())
            except RuntimeError:
                if 'fk_error' not in row:
                    raise
                row['converged']=False
            result['candidates'].append(row)
            after=frozen()
            if (any(after[k]!=before[k] for k in ('time','grip','cut','torn','nodes','failures'))
                    or any(not np.allclose(after[k],before[k],atol=1e-9,rtol=0) for k in ('q','tcp'))):
                raise ValueError('FK state did not restore')
        result['restored']=True
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            stop_simulation(client,sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        with args.out.open('x',encoding='utf-8',newline='\n') as f:
            json.dump(result,f,indent=2)
            f.write('\n')
    print(result['error'] or 'probe complete')
    return int(bool(result['error']))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out',required=True,type=Path)
    raise SystemExit(main(ap.parse_args()))
