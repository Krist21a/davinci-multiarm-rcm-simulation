"""Run the visual-servo pick-and-place once nominally and under perturbations; write a summary.

Each run is a separate visual_servo_task.py process on the same CoppeliaSim instance.
Output: validation/endoscope/visual_task_summary.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'build'))
from simtools import ENDO_VALIDATION, write_json  # noqa: E402

RUNS = [('nominal', [], True)] + [
    (f'perturbed_s{s}', ['--calib-rot-deg', '2', '--calib-trans-m', '0.003', '--depth-noise-m', '0.0005', '--seed', str(s)], False)
    for s in range(1, 6)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='localhost')
    p.add_argument('--port', type=int, default=23000)
    args = p.parse_args()
    rows = []
    for name, extra, frames in RUNS:
        cmd = [sys.executable, str(HERE / 'visual_servo_task.py'), '--host', args.host, '--port', str(args.port), '--name', name] + extra
        if frames:
            cmd.append('--frames')
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = ENDO_VALIDATION / f'visual_task_{name}.json'
        r = json.loads(out.read_text(encoding='utf-8')) if out.exists() else {}
        rows.append(dict(name=name, args=extra, exit_code=proc.returncode, success=r.get('success'),
                         final_block_to_target_mm=round(r['final_block_to_target_m'] * 1000, 3) if r else None,
                         servo_cycles=r.get('servo_cycles'), commands_rejected=r.get('commands_rejected'),
                         unexpected_contacts=len(r.get('unexpected_contacts', [])) if r else None,
                         max_psm_rcm_error_m=r.get('max_psm_rcm_error_m')))
        print(rows[-1], flush=True)
        if proc.returncode not in (0, 1):
            print(proc.stderr[-2000:], flush=True)
    write_json(ENDO_VALIDATION / 'visual_task_summary.json', {
        'runs': rows, 'successes': sum(1 for r in rows if r['success']), 'total': len(rows),
        'perturbation': 'camera pose error 2 deg + 3 mm (random direction per seed), depth noise sigma 0.5 mm per pixel'})
    return 0 if all(r['success'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
