"""Run a registered two-bite development case on a dedicated CoppeliaSim port."""
import argparse
import json
import uuid
from pathlib import Path

import numpy as np

from two_bite import TwoBiteFSM, TwoBiteWorld
from simtools import REPO, add_arguments, connect, relative, sha256, stop_simulation

PLAN = REPO / 'config/task/two_bite_v1.json'
BASE = REPO / 'config/task/cut_task.json'


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = json.loads(PLAN.read_bytes())
    cfg = json.loads(BASE.read_bytes())
    case = protocol['cases'][args.case]
    args.scene = REPO / protocol['scene']
    if sha256(BASE) != protocol['base_config_sha256'] or sha256(args.scene) != protocol['scene_sha256']:
        raise ValueError('registered scene or safety configuration differs')
    cfg['cut']['control_source'] = 'vision'
    cfg['perception'].update(case, mode='shadow', material_pattern=True, local_material_chart=True,
                             front_recovery=True, predictive_view=True, exclude_tcp_marker=True,
                             recovery_view_after_cut=True, ray_marker_fit=True)
    args.initial_view = protocol['initial_ecm'][:2]
    args.initial_view_insertion_m = protocol['initial_ecm'][2]
    args.scenario = args.case
    args.run_id = uuid.uuid4().hex
    scripts = Path(__file__).resolve().parents[1]
    sources = [scripts / 'task' / (n+'.py') for n in
               ('run_two_bite', 'two_bite', 'cut_fsm', 'sim_world', 'ground_truth', 'workspace',
                'cutter_view', 'tool_pose', 'visibility')]
    sources += [scripts / 'vision' / (n+'.py') for n in ('cut_perception', 'material_coordinate')]
    report = dict(protocol=protocol, protocol_sha256=sha256(PLAN), run_id=args.run_id, case=args.case,
                  config=cfg, source_sha256={relative(p): sha256(p) for p in sources},
                  base_config_sha256=sha256(BASE), scene_sha256=sha256(args.scene), error=None, cleanup_error=None)
    client = sim = world = fsm = None
    try:
        client, sim = connect(args)
        world = TwoBiteWorld(client, sim, args, cfg, mode='full', protocol=protocol, blocked=args.case == 'blocked')
        if world.controller_sha256 != protocol['controller_sha256']:
            raise ValueError('unexpected staged controller')
        fsm = TwoBiteFSM(world, cfg)
        fsm.run()
        world.observe(inject_events=False)
        report.update(fsm.report(), samples=world.samples, perception=world.perc_log, events=world.event_log,
                      park=world.park, bites=world.bites, camera_moves=world.camera_moves,
                      applied_view=world.initial_view_readback, controller_sha256=world.controller_sha256,
                      strict_truth=world.truth.strict, truth_purposes=world.truth.reads_by_purpose,
                      roles=world.roles_report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                report['cleanup_error'] = f'{type(exc).__name__}: {exc}'
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
    data = json.dumps(report, indent=2, allow_nan=False, default=lambda x: np.asarray(x).tolist())+'\n'
    with args.out.open('x', encoding='utf-8', newline='\n') as f:
        f.write(data)
    print(args.case, report.get('final_state'), report['error'] or report['cleanup_error'] or report.get('reason'))
    return int(report.get('final_state') != 'DONE' or bool(report['error'] or report['cleanup_error']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--case', choices=['nominal', 'noise', 'blocked'], required=True)
    parser.add_argument('--out', required=True, type=Path)
    raise SystemExit(run(parser.parse_args()))
