"""Live tool-pose accuracy benchmark on the verified articulated scene.

Drives the FSM to ALIGN (scissors staged at the cut site), then commands a grid of
~30 tool-pose targets through IK (tool_pose.solve) + setToolPose and measures the
achieved tcp against each commanded target. Reports position/orientation errors,
RCM drift, FK-vs-measured consistency and refusals.

Run: python scripts/tests/tool_pose_accuracy.py --port 23000
"""
import argparse
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'task'))
sys.path.insert(0, str(REPO / 'scripts' / 'build'))

from cut_fsm import CutTaskFSM, S  # noqa: E402
from sim_world import SimWorld  # noqa: E402
from simtools import add_arguments, connect, matrix_to_np, relative, sha256, stop_simulation, write_json  # noqa: E402


def axis_err(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(a) @ np.asarray(b))))))


def main():
    p = argparse.ArgumentParser()
    add_arguments(p)
    p.add_argument('--scene', type=Path, default=REPO / 'scenes' / 'davinci_si_articulated_verified.ttt')
    args = p.parse_args()
    cfg = json.loads((REPO / 'config' / 'task' / 'cut_task.json').read_text(encoding='utf-8'))
    run_id = uuid.uuid4().hex
    report = dict(run_id=run_id, scene='scenes/davinci_si_articulated_verified.ttt',
                  source_sha256={relative(x): sha256(x) for x in
                                 [REPO / 'scripts' / 'task' / 'sim_world.py', REPO / 'scripts' / 'task' / 'tool_pose.py',
                                  REPO / 'src' / 'instruments' / 'articulation.lua',
                                  REPO / 'src' / 'endoscope' / 'rcm_controller.lua',
                                  REPO / 'config' / 'task' / 'cut_task.json']},
                  samples=[], summary={})
    client = sim = world = None
    try:
        client, sim = connect(args)
        world = SimWorld(client, sim, args, cfg, mode='full', events={})
        report['controller_sha256'] = world.controller_sha256
        fsm = CutTaskFSM(world, cfg)
        while fsm.state != S.ALIGN and fsm.state not in (S.DONE, S.FAILED):
            fsm.tick()
        if fsm.state != S.ALIGN:
            raise RuntimeError(f'could not stage scissors: {fsm.state.name}')

        arm = world.cutter
        i = world.index[arm]
        front = world.cut_front()
        v, u = world.v, world.u
        n = np.cross(u, v) / np.linalg.norm(np.cross(u, v))     # sheet normal
        tcp_h = world.by[arm + '_tcp']

        # 6 positions (lateral x along-v, mm) x 5 jaw-axis tilts about the opening axis u.
        du_mm, dv_mm, tilts = (-4, 0, 4), (-4, 4), (0, -15, 15, -30, 30)
        for du in du_mm:
            for dv in dv_mm:
                for tilt in tilts:
                    target_axis = v * math.cos(math.radians(tilt)) + n * math.sin(math.radians(tilt))
                    target_p = front - 0.001 * v + du * 1e-3 * u + dv * 1e-3 * v
                    sample = dict(du_mm=du, dv_mm=dv, tilt_deg=tilt,
                                  target_p=[float(x) for x in target_p],
                                  target_axis=[float(x) for x in target_axis])
                    res = world._tool_solve(arm, target_p, target_axis, u)
                    if res is None or not res['converged']:
                        sample['refused'] = 'ik_not_converged' if res else 'fk_error'
                        report['samples'].append(sample)
                        continue
                    fk_pred = world._tool_fk(arm)(res['q'])
                    ok = world.tool_command(arm, res['q'], 10)
                    if not ok:
                        sample.update(refused='setToolPose', ik_pos_err_m=res['pos_err_m'])
                        report['samples'].append(sample)
                        continue
                    m = matrix_to_np(sim.getObjectMatrix(tcp_h, -1))
                    pos_err = float(np.linalg.norm(m[:3, 3] - target_p))
                    stats = world.call('getRCMStats')[i - 1]
                    sample.update(refused=None, pos_err_m=pos_err,
                                  axis_err_deg=axis_err(m[:3, 2], target_axis),
                                  yaw_axis_err_deg=min(axis_err(m[:3, 0], u), axis_err(m[:3, 0], -u)),
                                  fk_pos_err_m=float(np.linalg.norm(np.asarray(fk_pred)[:3, 3] - m[:3, 3])),
                                  ik_pos_err_m=res['pos_err_m'], ik_iterations=res['iterations'],
                                  rcm_drift_m=float(stats['drift_m']), at_limit=res['at_limit'])
                    report['samples'].append(sample)
                    print(f"du{du:+d} dv{dv:+d} t{tilt:+d}: pos {pos_err * 1000:.2f} mm  "
                          f"axis {sample['axis_err_deg']:.2f} deg  fk {sample['fk_pos_err_m'] * 1000:.2f} mm")
        done = [s for s in report['samples'] if s.get('refused') is None]
        for key in ('pos_err_m', 'axis_err_deg', 'yaw_axis_err_deg', 'fk_pos_err_m'):
            vals = np.array([s[key] for s in done]) if done else np.array([float('nan')])
            report['summary'][key] = dict(median=float(np.median(vals)), p95=float(np.percentile(vals, 95)),
                                          max=float(np.max(vals)), n=int(vals.size))
        report['summary']['max_rcm_drift_m'] = float(max((s['rcm_drift_m'] for s in done), default=0))
        report['summary']['refusals'] = len(report['samples']) - len(done)
        print(json.dumps(report['summary'], indent=1))
    finally:
        if client is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                report['cleanup_error'] = str(exc)
        out = REPO / 'validation' / 'safety' / 'tool_pose_accuracy.json'
        write_json(out, report)
        print('wrote', out)


if __name__ == '__main__':
    main()
