"""Run the preregistered passive bridge view sweep in an owned simulator."""
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
OUTPUT = ROOT / 'validation/task/m5_bridge_fracture_view_sweep_v1_capture'
PORT = 23881


def ready():
    with socket.socket() as sock:
        sock.settimeout(.3)
        return sock.connect_ex(('127.0.0.1', PORT)) == 0


def main():
    if ready() or OUTPUT.exists() or not SIM.is_file():
        raise RuntimeError('dedicated port in use, output already exists or simulator absent')
    env = os.environ.copy()
    env['COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX'] = '_davinci_safety_review'
    stage = Path(tempfile.gettempdir()) / 'davinci_bridge_fracture_view_sweep_v1'
    stage.mkdir(exist_ok=True)
    log_path = stage / f'owned_{os.getpid()}.log'
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen([str(SIM), '-h', f'-GzmqRemoteApi.rpcPort={PORT}'],
                                   stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(90):
                if process.poll() is not None:
                    raise RuntimeError(f'owned simulator exited {process.returncode}')
                if ready():
                    break
                time.sleep(.5)
            else:
                raise TimeoutError('dedicated simulator did not start')
            subprocess.run([sys.executable, str(ROOT/'scripts/task/run_bridge_sweep.py'),
                            '--out', str(OUTPUT), '--port', str(PORT),
                            '--stage-dir', str(stage)], cwd=ROOT, env=env, check=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            if OUTPUT.is_dir():
                shutil.copyfile(log_path, OUTPUT/'simulator.log')


if __name__ == '__main__':
    main()
