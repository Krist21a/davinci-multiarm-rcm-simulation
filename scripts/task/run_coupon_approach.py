"""Drive the unchanged task up to its first authorized closure, without closing."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

from live_active_look import ActiveLookFSM
from run_coupon_roll import ACTIVE, SIM, registered as registered_roll
from run_cut_task import REPO, add_arguments, sha256
from run_row9_capture import PassiveFramesWorld
import run_same_bite as parent

PLAN = REPO / 'config/task/m5_coupon_approach_dev_v1.json'
ROLL_PLAN = REPO / 'config/task/m5_coupon_roll_dev_v1.json'
SCORER = REPO / 'scripts/analysis/score_coupon_approach.py'
OUTPUT = REPO / 'validation/task/m5_coupon_approach_dev_v1'
PORT = 23884


def load(path):
    return json.loads(path.read_bytes())


def registered():
    plan, roll = load(PLAN), registered_roll()
    if sha256(ROLL_PLAN) != plan['parent_sha256'] or sha256(ACTIVE) != plan['active_sha256']:
        raise ValueError('registered parent or active policy changed')
    return plan, roll


def folder(phase, angle):
    return OUTPUT / phase / f'roll_{angle:+d}'


class ApproachReached(Exception):
    pass


class NonCuttingWorld(PassiveFramesWorld):
    approach = None
    last_advance = None

    def call(self, name, *args):
        if (name == 'setGrip' and args[0] == self.index[self.cutter]
                and args[1] != .5):
            raise RuntimeError('short check forbids cutter closing or any non-open command')
        return super().call(name, *args)

    def advance_scissors(self, distance):
        tcp = self.by[self.cutter + '_tcp']
        before = np.asarray(self.sim.getObjectPosition(tcp, -1))
        accepted = super().advance_scissors(distance)
        after = np.asarray(self.sim.getObjectPosition(tcp, -1))
        self.last_advance = dict(tick=self.tick, accepted=bool(accepted), request_m=distance,
                                 forward_m=float((after - before) @ self.v))
        self.event_log.append(dict(event='probe_advance', **self.last_advance))
        return accepted

    def close_scissors(self):
        # This virtual call is reached only after the production FSM's fresh interlock checks.
        joints = self.arm_cfg[self.cutter]['grip_joints']
        jaw_angles = [self.sim.getJointPosition(self.by[name]) for name in joints]
        self.approach = dict(tick=self.tick, advance=self.last_advance, jaw_angles=jaw_angles,
                             awaiting_new_slit=bool(self.awaiting_new_slit),
                             fix_tick=None if self._fix is None else self._fix['tick'],
                             closure_aligned=bool(self.closure_aligned()),
                             remaining_sweep_m=self.endpoint_advance_budget())
        self.hold()
        raise ApproachReached


class ShortFSM(ActiveLookFSM):
    def run(self):
        try:
            return super().run()
        except ApproachReached:
            return self.state

    def report(self):
        return dict(super().report(), dry_checkpoint=self.w.approach)


def capture(args, roll):
    args.out = folder(args.phase, args.angle)
    args.capture = None
    args.all_capture = args.out / f'{args.case}_frames'
    args.all_capture.mkdir(parents=True, exist_ok=False)
    args.targets = load(ACTIVE)['recovery_targets']
    short = args.phase == 'short'
    status = parent.run(
        args, world_class=NonCuttingWorld if short else PassiveFramesWorld,
        fsm_class=ShortFSM if short else ActiveLookFSM, active_spec=ACTIVE,
        policy=f'coupon_approach_{args.phase}', sheet_cfg=dict(roll['sheet'], roll_deg=args.angle),
        extra_sources=(PLAN, ROLL_PLAN, SCORER, Path(__file__),
                       Path(__file__).with_name('run_row9_capture.py'),
                       *(REPO / p for p in roll['reader_sha256'])))
    # A healthy development refusal is a report, not a launch failure.
    task = load(args.out / f'{args.case}.json')
    if 'perception' in task:
        parent.put(args.out / f'{args.case}_optical.json', dict(
            case=args.case, phase=args.phase, roll_deg=args.angle, run_id=task['run_id'],
            registration_sha256=task['registration_sha256'],
            exposures=[dict(tick=p['tick'], estimate=p['estimate']) for p in task['perception']]))
    return status


def ready():
    with socket.socket() as sock:
        sock.settimeout(.3)
        return sock.connect_ex(('127.0.0.1', PORT)) == 0


def launch(phase, angles, cases, plan):
    dest = OUTPUT / phase
    if dest.exists() or ready() or not SIM.is_file():
        raise RuntimeError('output exists, simulator missing, or dedicated port in use')
    stage = Path(tempfile.mkdtemp(prefix=f'davinci_coupon_approach_{phase}_'))
    dest.mkdir(parents=True)
    parent.put(dest / 'protocol.json', plan)
    env = dict(os.environ, COPPELIASIM_USER_SETTINGS_FOLDER_SUFFIX='_davinci_safety_review')
    launched = []
    with (dest / 'simulator.log').open('xb') as log:
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
            for angle in angles:
                for case in cases:
                    cmd = [sys.executable, '-B', str(Path(__file__).resolve()), phase,
                           '--angle', str(angle), '--case', case, '--host', '127.0.0.1',
                           '--port', str(PORT), '--stage-dir', str(stage)]
                    code = subprocess.run(cmd, cwd=REPO, env=env, check=False).returncode
                    launched.append(dict(angle=angle, case=case, exit_code=code))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            parent.put(dest / 'launch.json', dict(protocol_sha256=sha256(PLAN), trials=launched))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('phase', choices=('short', 'full'))
    parser.add_argument('--angle', type=int)
    parser.add_argument('--case')
    args = parser.parse_args()
    plan, roll = registered()
    angles = plan['angles_deg']
    if args.phase == 'full':
        chosen = load(OUTPUT / 'short' / 'decision.json')
        if (chosen['protocol_sha256'] != sha256(PLAN)
                or chosen['predictions_sha256'] != sha256(OUTPUT / 'short' / 'predictions.json')
                or not chosen['control_passed'] or chosen['selected_roll_deg'] not in angles
                or not chosen['selected_roll_deg']):
            raise ValueError('no eligible sealed short-check selection')
        angles = [chosen['selected_roll_deg']]
    if args.angle is None and args.case is None:
        launch(args.phase, angles, plan['cases'], plan)
        return 0
    if args.angle not in angles or args.case not in plan['cases']:
        parser.error('unregistered case or angle')
    return capture(args, roll)


if __name__ == '__main__':
    raise SystemExit(main())
