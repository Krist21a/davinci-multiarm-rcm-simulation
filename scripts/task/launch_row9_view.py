"""Launch the single versioned real-ECM negative checkpoint with an owned simulator."""
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
PORT = 23878


def ready():
    with socket.socket() as s:
        s.settimeout(.3)
        return s.connect_ex(('127.0.0.1', PORT))==0


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sheet', choices=('conventional','bridge'), default='conventional')
    selected = parser.parse_args().sheet
    output = ROOT / ('validation/task/m5_bridge_view_negative_v1_capture' if selected == 'bridge'
                     else 'validation/task/m5_row9_view_dev_v1_capture')
    if ready() or output.exists() or not SIM.is_file():
        raise RuntimeError('reserved port in use, result exists or simulator absent')
    env = os.environ.copy()
    env['COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX'] = '_davinci_safety_review'
    stage = Path(tempfile.gettempdir()) / ('davinci_bridge_view_negative_v1' if selected == 'bridge'
                                            else 'davinci_row9_view_dev_v1')
    stage.mkdir(exist_ok=True)
    logfile = stage / f'sim_{os.getpid()}.log'
    with logfile.open('x', encoding='utf-8') as log:
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
                raise TimeoutError('simulator did not open dedicated port')
            command = [sys.executable, str(ROOT/'scripts/task/run_row9_view.py'),
                       '--out', str(output), '--port', str(PORT), '--stage-dir', str(stage),
                       '--sheet', selected]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            if output.is_dir():
                shutil.copyfile(logfile, output/'simulator.log')


if __name__ == '__main__':
    main()
