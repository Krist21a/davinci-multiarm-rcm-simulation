"""Collect two versioned development runs in an owned local simulator instance."""
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SIM = Path('C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/coppeliaSim.exe')
PORT = 23880


def ready():
    with socket.socket() as sock:
        sock.settimeout(.3)
        return sock.connect_ex(('127.0.0.1', PORT)) == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=('v1', 'v2', 'v3', 'v4', 'v5'), default='v1')
    version = parser.parse_args().protocol
    output = ROOT / f'validation/task/m5_visible_fracture_dev_{version}'
    if ready() or output.exists() or not SIM.is_file():
        raise RuntimeError('dedicated port in use, result already exists or simulator missing')
    env = os.environ.copy()
    env['COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX'] = '_davinci_safety_review'
    stage = Path(tempfile.gettempdir()) / ('davinci_visible_fracture_dev_' + version)
    stage.mkdir(exist_ok=True)
    log_path = stage / f'simulator_{os.getpid()}.log'
    with log_path.open('x', encoding='utf-8') as log:
        proc = subprocess.Popen([str(SIM), '-h', f'-GzmqRemoteApi.rpcPort={PORT}'],
                                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(90):
                if proc.poll() is not None:
                    raise RuntimeError(f'owned simulator exited {proc.returncode}')
                if ready():
                    break
                time.sleep(.5)
            else:
                raise TimeoutError('dedicated simulator did not start')
            for case in ('nominal', 'depth_noise_0p5'):
                command = [sys.executable, str(ROOT / 'scripts/task/run_visible_fracture.py'),
                           '--case', case, '--out', str(output), '--protocol', version,
                           '--port', str(PORT), '--stage-dir', str(stage)]
                result = subprocess.run(command, cwd=ROOT, env=env, check=False)
                print(case, 'exit', result.returncode, flush=True)
                # Fail-closed visual trials return 1; an absent report is a setup error.
                if not (output / f'{case}.json').is_file():
                    raise RuntimeError(f'{case}: no task report')
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            if output.is_dir():
                shutil.copyfile(log_path, output / 'simulator.log')


if __name__ == '__main__':
    main()
