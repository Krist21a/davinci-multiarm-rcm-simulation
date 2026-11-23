"""Measure RGB-D + truth-label frame collection throughput for the learned-perception study.

Launches N dedicated headless CoppeliaSim instances (one per worker: own RPC port, own
settings profile, own ASCII stage dir) and runs R unchanged `run_cut_task.py` episodes per
worker (full / nominal / truth control / shadow perception with frame dump, --trace).
Frames are written under --data-root (outside git); per-run reports go to the fresh
directory validation/task/<tag>/. Scenes, configs and historical reports are not touched.

The measurement protocol is written into the output JSON before any episode runs. This is
a throughput/infrastructure measurement only: no perception or task claim is made.
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / 'scripts' / 'task' / 'run_cut_task.py'
SIM_EXE = Path(r'C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe')
SOURCES = ['scripts/task/run_cut_task.py', 'scripts/task/sim_world.py', 'scripts/task/cut_fsm.py',
           'config/task/cut_task.json', 'src/tissue/tissue.lua', 'src/endoscope/rcm_controller.lua',
           'scenes/davinci_si_articulated_verified.ttt']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    tmp.replace(path)


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(('127.0.0.1', port)) == 0


def launch(port, suffix, log_path):
    if port_open(port):
        raise RuntimeError(f'port {port} already in use')
    env = os.environ.copy()
    env['COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX'] = suffix
    log = log_path.open('w', encoding='utf-8')
    proc = subprocess.Popen([str(SIM_EXE), '-h', f'-GzmqRemoteApi.rpcPort={port}'],
                            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 120:
        if proc.poll() is not None:
            raise RuntimeError(f'simulator on {port} exited during startup: {proc.returncode}')
        if port_open(port):
            return proc, log, time.monotonic() - t0
        time.sleep(0.5)
    raise TimeoutError(f'simulator on {port} did not open its RPC port')


def shutdown(proc, log):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    proc.stdin.close()
    log.close()


def worker(cfg, n, k, ready, results):
    port = cfg['base_port'] + 10 * n + k
    stage = Path(tempfile.gettempdir()) / f"{cfg['tag']}_n{n}_w{k}"
    stage.mkdir(parents=True, exist_ok=True)
    logs = REPO / 'validation' / 'task' / cfg['tag'] / 'logs'
    rec = dict(n=n, worker=k, port=port, runs=[])
    results.append(rec)
    proc = log = None
    try:
        proc, log, rec['sim_startup_s'] = launch(port, f"_{cfg['tag']}_w{k}", logs / f'sim_n{n}_w{k}.log')
    except Exception as exc:
        rec['error'] = f'{type(exc).__name__}: {exc}'
    ready.wait()
    try:
        for r in range(cfg['repeats'] if 'error' not in rec else 0):
            name = f'n{n}_w{k}_r{r}'
            dump = Path(cfg['data_root']) / cfg['tag'] / name
            cmd = [sys.executable, str(RUNNER), '--mode', 'full', '--scenario', cfg['scenario'],
                   '--control-source', 'truth', '--trace', '--port', str(port), '--stage-dir', str(stage),
                   '--perception', json.dumps(dict(mode='shadow', dump_dir=dump.as_posix())),
                   '--name', f"{cfg['tag']}/{name}"]
            t0 = time.monotonic()
            with (logs / f'{name}.log').open('w', encoding='utf-8') as out:
                rc = subprocess.run(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT).returncode
            wall = time.monotonic() - t0
            frames = sorted(dump.glob('tick_*.npz')) if dump.exists() else []
            report_path = REPO / 'validation' / 'task' / cfg['tag'] / f'{name}.json'
            report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
            rec['runs'].append(dict(
                name=name, returncode=rc, wall_s=round(wall, 3), success=report.get('success'),
                final_state=report.get('final_state'), error=report.get('error'),
                frames=len(frames), bytes=sum(f.stat().st_size for f in frames),
                capture_manifest=(dump / 'capture.json').exists(),
                trace_nodes_sha256=[t.get('nodes_sha256') for t in report.get('trace', [])],
                t_start=t0, t_end=t0 + wall))
    finally:
        shutdown(proc, log)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default='learn_throughput_v1')
    p.add_argument('--data-root', default='F:/davinci_m5_learn')
    p.add_argument('--concurrency', default='1,2,3', help='comma list of simultaneous instances')
    p.add_argument('--repeats', type=int, default=3, help='episodes per worker per concurrency level')
    p.add_argument('--scenario', default='nominal')
    p.add_argument('--base-port', type=int, default=23800)
    args = p.parse_args()

    res_dir = REPO / 'validation' / 'task' / args.tag
    out_path = REPO / 'validation' / 'task' / f'{args.tag}.json'
    data_dir = Path(args.data_root) / args.tag
    if res_dir.exists() or out_path.exists() or data_dir.exists():
        raise SystemExit(f'refusing to reuse {args.tag}: choose a fresh --tag')
    (res_dir / 'logs').mkdir(parents=True)
    data_dir.mkdir(parents=True)
    levels = [int(v) for v in args.concurrency.split(',')]
    cfg = dict(tag=args.tag, data_root=Path(args.data_root).as_posix(), repeats=args.repeats,
               scenario=args.scenario, base_port=args.base_port)
    git = lambda *a: subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    result = dict(
        purpose='Infrastructure throughput measurement for a proposed learned-perception study. '
                'Not a perception, task or M5 result.',
        protocol=dict(
            episode='run_cut_task.py --mode full --scenario {scenario} --control-source truth --trace, '
                    'perception mode=shadow with per-tick npz dump (rgb, depth, camera, tcp, markers, truth labels)',
            concurrency_levels=levels, repeats_per_worker=args.repeats, **cfg,
            batch_wall='from the moment all instances of a level have opened their RPC port (or failed) '
                       'to the end of the last episode; simulator startup is reported separately',
            frames_per_hour='total dumped frames of a level / batch wall * 3600',
            determinism_check='every successful episode is expected to produce the same nodes_sha256 trace; '
                              'a difference across parallel instances would indicate interference'),
        host=dict(platform=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
                  python=sys.version.split()[0]),
        git=dict(head=git('rev-parse', 'HEAD'), dirty_paths=len(git('status', '--porcelain').splitlines())),
        source_sha256={s: sha256(REPO / s) for s in SOURCES},
        levels=[])
    write_json(out_path, result)

    for n in levels:
        ready, results, threads = threading.Barrier(n), [], []
        for k in range(n):
            t = threading.Thread(target=worker, args=(cfg, n, k, ready, results))
            t.start()
            threads.append(t)
        t_ready = None
        for t in threads:
            t.join()
        runs = [r for w in results for r in w['runs']]
        if runs:
            t_ready = min(r['t_start'] for r in runs)
            wall = max(r['t_end'] for r in runs) - t_ready
            frames = sum(r['frames'] for r in runs)
            ok = [r for r in runs if r['success']]
            level = dict(
                concurrency=n, workers=sorted(results, key=lambda w: w['worker']), episodes=len(runs),
                successful=len(ok), batch_wall_s=round(wall, 2), frames=frames,
                gb=round(sum(r['bytes'] for r in runs) / 1e9, 3),
                frames_per_hour=round(frames / wall * 3600) if wall > 0 else None,
                episodes_per_hour=round(len(runs) / wall * 3600, 1) if wall > 0 else None,
                mean_episode_s=round(sum(r['wall_s'] for r in runs) / len(runs), 2),
                mean_bytes_per_frame=round(sum(r['bytes'] for r in runs) / frames) if frames else None,
                traces_identical=len({tuple(r['trace_nodes_sha256']) for r in ok}) == 1 if ok else None)
        else:
            level = dict(concurrency=n, workers=results, episodes=0)
        for w in level['workers']:
            for r in w['runs']:
                r.pop('t_start'), r.pop('t_end')
        result['levels'].append(level)
        write_json(out_path, result)
        print(json.dumps({k: v for k, v in level.items() if k != 'workers'}), flush=True)
    print('report:', out_path.relative_to(REPO))


if __name__ == '__main__':
    main()
