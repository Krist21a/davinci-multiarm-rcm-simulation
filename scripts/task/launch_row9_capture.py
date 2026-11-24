"""Run two preregistered passive development captures in an owned CoppeliaSim instance."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SIM = Path('C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/coppeliaSim.exe')
PORT = 23876
OUTPUT = ROOT / 'validation/task/m5_row9_contact_dev_v1_1_capture'
STAGE = Path(tempfile.gettempdir()) / 'davinci_row9_contact_dev_v1_1'


def ready():
    with socket.socket() as sock:
        sock.settimeout(.3)
        return sock.connect_ex(('127.0.0.1', PORT)) == 0


def main():
    if ready() or not SIM.is_file() or OUTPUT.exists():
        raise RuntimeError('dedicated port in use, simulator missing, or result directory already exists')
    OUTPUT.mkdir(parents=True)
    STAGE.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX'] = '_davinci_safety_review'
    with (OUTPUT / 'simulator.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([str(SIM), '-h', f'-GzmqRemoteApi.rpcPort={PORT}'],
                                   stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(90):
                if process.poll() is not None:
                    raise RuntimeError(f'owned simulator exited: {process.returncode}')
                if ready():
                    break
                time.sleep(.5)
            else:
                raise TimeoutError('dedicated simulator did not start')
            for case in ('nominal', 'depth_noise_0p5'):
                command = [sys.executable, str(ROOT / 'scripts/task/run_row9_capture.py'),
                           '--case', case, '--out', str(OUTPUT),
                           '--port', str(PORT), '--stage-dir', str(STAGE)]
                result = subprocess.run(command, cwd=ROOT, env=env, check=False)
                print('capture', case, 'return code', result.returncode, flush=True)
                # A real fail-closed optical trial has rc=1 but must have a complete report.
                if not (OUTPUT / f'{case}.json').is_file():
                    raise RuntimeError(f'{case}: no run report')
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == '__main__':
    main()
