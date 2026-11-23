"""Run current cutting code on a dedicated simulator; never replace historical reports.

The input scene is staged and not saved. Results go to validation/safety by default.
This reports FAIL honestly if the stricter safety checks invalidate the old demo.
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'task'))
sys.path.insert(0, str(REPO / 'scripts' / 'build'))
from cut_fsm import CutTaskFSM, S, completion_issues  # noqa: E402
from sim_world import SimWorld  # noqa: E402
from simtools import ARTICULATED_SCENE as INSTRUMENT_SCENE  # noqa: E402
from simtools import add_arguments, connect, read_json, relative, sha256, stop_simulation, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--out', type=Path, default=REPO / 'validation' / 'safety' / 'live_cut.json')
    parser.add_argument('--mode', choices=['cut_only', 'full'], default='cut_only')
    parser.add_argument('--scene', type=Path, default=INSTRUMENT_SCENE)
    parser.add_argument('--scenario', choices=['nominal', 'occlusion_transient', 'occlusion_persistent'], default='nominal')
    parser.add_argument('--launch-simulator', type=Path, help='start and clean up a dedicated local CoppeliaSim executable')
    args = parser.parse_args()
    cfg = read_json(REPO / 'config' / 'task' / 'cut_task.json')
    client = sim = process = sim_log = None
    world = fsm = None
    start = time.time()
    result = dict(mode=args.mode, scenario=args.scenario, success=False)
    try:
        if args.launch_simulator:
            if args.host not in ('localhost', '127.0.0.1'):
                raise ValueError('--launch-simulator requires a local host')
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex((args.host, args.port)) == 0:
                    raise RuntimeError('requested dedicated port is already in use')
            args.out.parent.mkdir(parents=True, exist_ok=True)
            sim_log = args.out.with_suffix('.sim.log').open('w', encoding='utf-8')
            env = os.environ.copy()
            env['COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX'] = '_davinci_safety_review'
            process = subprocess.Popen([str(args.launch_simulator), '-h', f'-GzmqRemoteApi.rpcPort={args.port}'],
                                       stdin=subprocess.PIPE, stdout=sim_log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'simulator exited during startup: {process.returncode}')
                with socket.socket() as probe:
                    probe.settimeout(0.2)
                    if probe.connect_ex((args.host, args.port)) == 0:
                        break
                time.sleep(0.5)
            else:
                raise TimeoutError('simulator did not open the dedicated API port')
        client, sim = connect(args)
        world = SimWorld(client, sim, args, cfg, mode=args.mode,
                         event=None if args.scenario == 'nominal' else args.scenario)
        fsm = CutTaskFSM(world, cfg, cut_only=args.mode == 'cut_only')
        fsm.run()
        obs = world.observe(inject_events=False)
        issues = completion_issues(obs, cfg)
        result.update(fsm.report(), controller_sha256=world.controller_sha256,
                      success=fsm.state == S.DONE and not issues, acceptance_issues=issues,
                      final_observation=vars(obs), controller_stats=world.call('getRCMStats'))
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}')
        if fsm is not None:
            result.update(fsm.report())
    finally:
        if sim is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                result['cleanup_error'] = str(exc)
                result['success'] = False
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdin.close()
        if sim_log is not None:
            sim_log.close()
    result['seconds'] = round(time.time() - start, 3)
    sources = [REPO / 'scripts' / 'task' / name for name in ('cut_fsm.py', 'sim_world.py', 'visibility.py', 'workspace.py')]
    sources += [REPO / 'src' / 'endoscope' / 'rcm_controller.lua', REPO / 'src' / 'tissue' / 'tissue.lua',
                REPO / 'src' / 'instruments' / 'capability_guard.lua', REPO / 'config' / 'task' / 'cut_task.json',
                args.scene]
    result['source_sha256'] = {relative(p): sha256(p) for p in sources}
    write_json(args.out, result)
    print(result.get('final_state', 'ERROR'), result.get('reason', result.get('error')), flush=True)
    print('success:', result['success'], 'report:', args.out, flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())