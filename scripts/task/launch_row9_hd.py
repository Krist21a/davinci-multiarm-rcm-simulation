"""Run one preregistered passive-HD development exposure in an owned simulator."""
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
PORT = 23877


def ready():
    with socket.socket() as sock:
        sock.settimeout(.3)
        return sock.connect_ex(('127.0.0.1', PORT)) == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sheet', choices=('conventional','bridge'), default='conventional')
    selected = parser.parse_args().sheet
    output = ROOT / ('validation/task/m5_bridge_hd_negative_v1_capture' if selected == 'bridge'
                     else 'validation/task/m5_row9_hd_dev_v1_1_capture')
    if ready() or not SIM.is_file() or output.exists():
        raise RuntimeError('dedicated port in use, simulator missing, or result directory already exists')
    stage = Path(tempfile.gettempdir()) / ('davinci_bridge_hd_negative_v1' if selected == 'bridge'
                                            else 'davinci_row9_hd_dev_v1_1')
    stage.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX'] = '_davinci_safety_review'
    log_path = stage / f'owned_simulator_{os.getpid()}.log'
    with log_path.open('x', encoding='utf-8') as log:
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
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/task/run_row9_hd.py'),
                                     '--out', str(output), '--port', str(PORT), '--sheet', selected,
                                     '--stage-dir', str(stage)], cwd=ROOT, env=env, check=False)
            print('passive HD run return code', result.returncode, flush=True)
            if not (output / 'hd_optical.json').exists():
                raise RuntimeError('no optical acquisition; inspect the run report')
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            if output.is_dir():
                shutil.copyfile(log_path, output / 'simulator.log')


if __name__ == '__main__':
    main()
