"""Run the registered coupon-roll development captures; no endpoint reader in control."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from live_active_look import ActiveLookFSM
from run_cut_task import REPO, add_arguments, sha256
from run_row9_capture import PassiveFramesWorld
import run_same_bite as parent

PLAN = REPO / 'config/task/m5_coupon_roll_dev_v1.json'
ACTIVE = REPO / 'config/task/m5_live_active_look_v1.json'
OUTPUT = REPO / 'validation/task/m5_coupon_roll_dev_v1'
SIM = Path('C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/coppeliaSim.exe')
PORT = 23883
POLICY = 'coupon_roll_development'


def registered():
    plan = json.loads(PLAN.read_bytes())
    for path, expected in {
            plan['parent']: plan['parent_sha256'], plan['scene']: plan['scene_sha256'],
            **plan['reader_sha256']}.items():
        if sha256(REPO / path) != expected:
            raise ValueError(f'registered input changed: {path}')
    return plan


def angle_folder(angle):
    return OUTPUT / f'roll_{angle:+d}'


def capture(args, plan):
    if args.roll not in plan['angles_deg'] or args.case not in plan['cases']:
        raise ValueError('unregistered angle or case')
    args.out = angle_folder(args.roll)
    args.capture = None
    args.all_capture = args.out / (args.case + '_frames')
    if any(path.exists() for path in (args.all_capture, args.out / (args.case + '.json'),
                                      args.out / (args.case + '_optical.json'))):
        raise FileExistsError('capture exists; never overwrite a run')
    args.all_capture.mkdir(parents=True)
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    sheets = dict(plan['sheet'], roll_deg=args.roll)
    result = parent.run(
        args, world_class=PassiveFramesWorld, fsm_class=ActiveLookFSM, active_spec=ACTIVE,
        policy=POLICY, sheet_cfg=sheets,
        extra_sources=(PLAN, Path(__file__), Path(__file__).with_name('run_row9_capture.py'),
                       REPO / 'scripts/analysis/score_coupon_roll.py',
                       *(REPO / path for path in plan['reader_sha256'])))
    report = json.loads((args.out / (args.case + '.json')).read_bytes())
    if 'perception' in report:
        parent.put(args.out / (args.case + '_optical.json'), dict(
            case=args.case, roll_deg=args.roll, run_id=report['run_id'],
            registration_sha256=report['registration_sha256'], plan_sha256=sha256(PLAN),
            exposures=[dict(tick=row['tick'], estimate=row['estimate']) for row in report['perception']]))
    return result


def ready():
    with socket.socket() as sock:
        sock.settimeout(.3)
        return sock.connect_ex(('127.0.0.1', PORT)) == 0


def launch(plan):
    if OUTPUT.exists() or not SIM.is_file() or ready():
        raise RuntimeError('output exists, simulator missing, or dedicated port in use')
    stage = Path(tempfile.mkdtemp(prefix='davinci_coupon_roll_v1_'))
    env = os.environ.copy()
    env['COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX'] = '_davinci_safety_review'
    logfile = stage / 'simulator.log'
    OUTPUT.mkdir()
    trials = []
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
                raise TimeoutError('dedicated simulator did not start')
            for angle in plan['angles_deg']:
                for case in plan['cases']:
                    cmd = [sys.executable, str(Path(__file__).resolve()), '--roll', str(angle),
                           '--case', case, '--host', '127.0.0.1', '--port', str(PORT),
                           '--stage-dir', str(stage)]
                    code = subprocess.run(cmd, cwd=REPO, env=env, check=False).returncode
                    report = angle_folder(angle) / (case + '.json')
                    trials.append(dict(roll_deg=angle, case=case, exit_code=code,
                                       report_present=report.is_file()))
                    print(f'{angle:+} deg {case}: process exit {code}', flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            shutil.copyfile(logfile, OUTPUT / 'simulator.log')
            parent.put(OUTPUT / 'launch.json', dict(plan_sha256=sha256(PLAN), trials=trials))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--roll', type=int)
    parser.add_argument('--case')
    args = parser.parse_args()
    plan = registered()
    if args.roll is None and args.case is None:
        launch(plan)
        return 0
    if args.roll is None or args.case is None:
        parser.error('--roll and --case must be set together')
    return capture(args, plan)


if __name__ == '__main__':
    raise SystemExit(main())
