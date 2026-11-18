"""Run the visual-servo pick-and-place once nominally and under perturbations; write a summary.

Each run is a separate visual_servo_task.py process on the same CoppeliaSim instance.
Output: validation/endoscope/visual_task_summary.json
"""
import argparse
import json
import math
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'build'))
from simtools import ENDO_VALIDATION, relative, sha256, write_json  # noqa: E402

RUNS = [('nominal', [], True)] + [
    (f'perturbed_s{s}', ['--calib-rot-deg', '2', '--calib-trans-m', '0.003', '--depth-noise-m', '0.0005', '--seed', str(s)], False)
    for s in range(1, 6)]


def finite_nonnegative(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def read_result(path, run_id):
    """Fail closed on stale, partial or malformed reports (including non-finite JSON numbers)."""
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        return {}, f'{type(exc).__name__}: {exc}'
    if not isinstance(result, dict):
        return {}, 'report must be a JSON object'
    if result.get('run_id') != run_id:
        return {}, 'report run_id does not match this invocation'
    if type(result.get('success')) is not bool:
        return {}, 'report success must be a boolean'
    error = result.get('final_block_to_target_m')
    if not finite_nonnegative(error):
        return {}, 'report final_block_to_target_m must be finite and nonnegative'
    for key in ('servo_cycles', 'commands_rejected'):
        if key in result and (type(result[key]) is not int or result[key] < 0):
            return {}, f'report {key} must be a nonnegative integer'
    if not isinstance(result.get('unexpected_contacts', []), list):
        return {}, 'report unexpected_contacts must be a list'
    rcm_error = result.get('max_psm_rcm_error_m')
    if rcm_error is not None and not finite_nonnegative(rcm_error):
        return {}, 'report max_psm_rcm_error_m must be finite and nonnegative'
    return result, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='localhost')
    p.add_argument('--port', type=int, default=23000)
    p.add_argument('--run-timeout', type=float, default=900.0, help='maximum wall time for each child process [s]')
    args = p.parse_args()
    if not math.isfinite(args.run_timeout) or args.run_timeout <= 0:
        p.error('--run-timeout must be finite and positive')
    suite_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    sources = [Path(__file__), HERE / 'visual_servo_task.py', HERE.parent / 'build' / 'simtools.py']
    source_hashes = {relative(path): sha256(path) for path in sources}
    rows = []
    for name, extra, frames in RUNS:
        run_id = uuid.uuid4().hex
        cmd = [sys.executable, str(HERE / 'visual_servo_task.py'), '--host', args.host, '--port', str(args.port),
               '--name', name, '--run-id', run_id] + extra
        if frames:
            cmd.append('--frames')
        exit_code, process_error, stderr, timed_out = None, None, '', False
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=args.run_timeout)
            exit_code, stderr = proc.returncode, proc.stderr or ''
            if exit_code != 0:
                process_error = f'child exited with code {exit_code}'
        except subprocess.TimeoutExpired:
            timed_out = True
            process_error = f'child timed out after {args.run_timeout:g} seconds'
        except OSError as exc:
            process_error = f'{type(exc).__name__}: {exc}'
        out = ENDO_VALIDATION / f'visual_task_{name}.json'
        r, report_error = read_result(out, run_id)
        rows.append(dict(name=name, run_id=run_id, args=extra, exit_code=exit_code,
                         success=exit_code == 0 and report_error is None and r.get('success') is True,
                         report=out.name, process_error=process_error, report_error=report_error, timed_out=timed_out,
                         stderr_tail=stderr[-2000:] if process_error else None,
                         final_block_to_target_mm=round(r['final_block_to_target_m'] * 1000, 3) if r else None,
                         servo_cycles=r.get('servo_cycles'), commands_rejected=r.get('commands_rejected'),
                         unexpected_contacts=len(r.get('unexpected_contacts', [])) if r else None,
                         max_psm_rcm_error_m=r.get('max_psm_rcm_error_m'),
                         controller_sha256=r.get('controller_sha256'), source_sha256=r.get('source_sha256')))
        print(rows[-1], flush=True)
    write_json(ENDO_VALIDATION / 'visual_task_summary.json', {
        'run_id': suite_id, 'started_at_utc': started_at, 'source_sha256': source_hashes,
        'runs': rows, 'successes': sum(1 for r in rows if r['success']), 'total': len(rows),
        'perturbation': 'camera pose error 2 deg + 3 mm (random direction per seed), depth noise sigma 0.5 mm per pixel'})
    return 0 if all(r['success'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
