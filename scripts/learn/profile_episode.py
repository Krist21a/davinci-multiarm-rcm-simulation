"""Profile where one unchanged run_cut_task.py episode spends its wall time.

Launches one dedicated headless CoppeliaSim instance, runs the episode in-process under
cProfile (full / nominal / truth / shadow perception with frame dump, --trace) and writes
the raw profile plus a per-function cumulative-time table to validation/task/<tag>/.
Infrastructure measurement only; no perception or task claim.
"""
import argparse
import cProfile
import io
import json
import pstats
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task'))
from throughput_probe import REPO, launch, shutdown, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', default='learn_profile_v1')
    p.add_argument('--data-root', default='F:/davinci_m5_learn')
    p.add_argument('--port', type=int, default=23900)
    args = p.parse_args()
    res = REPO / 'validation' / 'task' / args.tag
    dump = Path(args.data_root) / args.tag / 'episode'
    if res.exists() or dump.parent.exists():
        raise SystemExit(f'refusing to reuse {args.tag}')
    res.mkdir(parents=True)
    stage = Path(tempfile.gettempdir()) / args.tag
    stage.mkdir(exist_ok=True)
    proc, log, startup = launch(args.port, f'_{args.tag}', res / 'sim.log')
    import run_cut_task
    sys.argv = ['run_cut_task.py', '--mode', 'full', '--scenario', 'nominal', '--control-source', 'truth',
                '--trace', '--port', str(args.port), '--stage-dir', str(stage),
                '--perception', json.dumps(dict(mode='shadow', dump_dir=dump.as_posix())),
                '--name', f'{args.tag}/episode']
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    try:
        rc = prof.runcall(run_cut_task.main)
    finally:
        wall = time.perf_counter() - t0
        shutdown(proc, log)
    prof.dump_stats(str(res / 'episode.prof'))
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats('cumulative').print_stats(80)
    (res / 'cumulative.txt').write_text(buf.getvalue(), encoding='utf-8', newline='\n')
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats('tottime').print_stats(40)
    (res / 'tottime.txt').write_text(buf.getvalue(), encoding='utf-8', newline='\n')
    write_json(res / 'summary.json', dict(
        purpose='Wall-time breakdown of one episode for the learned-perception collection plan.',
        returncode=rc, wall_s=round(wall, 3), sim_startup_s=round(startup, 3),
        frames=len(list(dump.glob('tick_*.npz'))), profile='episode.prof'))
    print('wall', round(wall, 1), 'rc', rc, 'report', res.relative_to(REPO))


if __name__ == '__main__':
    main()
