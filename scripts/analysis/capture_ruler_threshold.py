"""Supplementary optical development samples. Requested lengths are not structural labels."""
import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import capture_ruler_endpoint as acquisition
from simtools import add_arguments, connect, sha256, stop_simulation

SPEC = acquisition.ROOT/'config/task/ruler_threshold_sample_v1.json'


def main(args):
    goal = args.material_goal_mm
    if goal not in (8.25, 8.5, 8.75):
        raise ValueError('unregistered supplementary collection goal')
    parent = json.loads(acquisition.PLAN.read_bytes())
    args.scene = acquisition.ROOT/parent['scene']
    if sha256(args.scene) != parent['scene_sha256'] or sha256(acquisition.BASE) != parent['base_sha256']:
        raise ValueError('wrong scene or base')
    cfg = json.loads(acquisition.BASE.read_bytes())
    cfg['cut'].update(length_m=goal*.001, control_source='truth', advance_m=.003)
    cfg['perception'].update(exclude_tcp_marker=True)
    args.initial_view, args.initial_view_insertion_m = parent['view'][:2], parent['view'][2]
    args.out.mkdir(exist_ok=False)
    r = dict(run_id=uuid.uuid4().hex, effective_goal_mm=goal, cfg=cfg, frames=[], error=None,
             spec_sha256=sha256(SPEC), script_sha256=sha256(Path(__file__)), scene_sha256=sha256(args.scene),
             collector_sha256=sha256(Path(acquisition.__file__)), parent_sha256=sha256(acquisition.PLAN))
    client = sim = None
    try:
        client,sim = connect(args)
        w=acquisition.RulerWorld(client,sim,args,cfg,mode='full')
        fsm=acquisition.CaptureFSM(w,cfg)
        fsm.folder,fsm.frames=args.out,r['frames']
        fsm.run()
        r.update(fsm.report(),controller_sha256=w.controller_sha256,view=w.initial_view_readback)
    except Exception as exc:
        r['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            stop_simulation(client,sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
        with (args.out/'supplement.json').open('x',encoding='utf-8',newline='\n') as f:
            json.dump(r,f,indent=2,allow_nan=False,default=lambda v: np.asarray(v).tolist())
            f.write('\n')
    print(goal,r.get('final_state'),r['error'] or r.get('reason'))
    return int(bool(r['error']) or r.get('final_state')!='DONE')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--material-goal-mm',type=float,required=True)
    ap.add_argument('--out',type=Path,required=True)
    raise SystemExit(main(ap.parse_args()))
