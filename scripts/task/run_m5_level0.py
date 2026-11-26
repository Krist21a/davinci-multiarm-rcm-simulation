"""Run one instrumented nominal truth-feedback reference on the unchanged M5 scene."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import run_cut_task as task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    task.add_arguments(parser)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    scene = task.INSTRUMENT_SCENE
    cfg = task.REPO / 'config/task/cut_task.json'
    cut = json.loads(cfg.read_bytes())['cut']
    lo, hi = 0.009, 0.010
    eps = 1e-12
    if abs(cut['length_m'] - lo) > eps or abs(cut['length_m'] + cut['max_overcut_m'] - hi) > eps:
        parser.error('Level 0 requires the original 9-10 mm endpoint window')
    argv = ['--mode', 'full', '--scenario', 'nominal', '--control-source', 'truth',
            '--align-mode', 'tool_pose', '--trace', '--scene', str(scene), '--name', 'nominal',
            '--host', args.host, '--port', str(args.port), '--stage-dir', str(args.stage_dir),
            '--timeout', str(args.timeout)]
    reg = {
        'id': 'm5_level0_v1',
        'scope': 'One nominal truth-feedback simulation reference. No vision or signed M5 acceptance.',
        'endpoint_m': [lo, hi],
        'numeric_tolerance_m': eps,
        'criteria': [
            'Original full nominal runner reaches DONE without fault, tear or cleanup failure.',
            'Trace starts at zero, is consecutive and nondecreasing, and never exceeds 10 mm.',
            'At least one accepted cutting closure; all cutting closures accepted and before first IN.',
            'No cutter setGrip call at or after the first IN observation, including during withdrawal.',
            'Exactly one accepted withdrawal starts at first IN, off ready, with physical jaws closed, '
            'and ends at the original ready pose with physical jaws still closed.',
            'Final structural length equals the first IN length; CUT transitions directly to DONE.',
        ],
        'runner_arguments': argv,
        'sha256': {task.relative(p): task.sha256(p) for p in (
            scene, cfg, Path(__file__).resolve(), Path(task.__file__).resolve(),
            task.HERE / 'cut_fsm.py', task.HERE / 'sim_world.py', task.HERE / 'ground_truth.py',
            task.REPO / 'src/tissue/tissue.lua', task.REPO / 'src/endoscope/rcm_controller.lua',
        )},
    }
    args.out.mkdir(exist_ok=False)
    registration = args.out / 'registration.json'
    task.write_json(registration, reg)
    grips, closures, withdrawals = [], [], []

    class AuditedWorld(task.SimWorld):
        def call(self, name, *params):
            result = super().call(name, *params)
            if name == 'setGrip' and params[0] == self.index[self.cutter]:
                grips.append(dict(tick=self.tick, angle_rad=params[1], accepted=bool(result)))
            return result

        def close_scissors(self):
            accepted = super().close_scissors()
            closures.append(dict(tick=self.tick, accepted=bool(accepted)))
            return accepted

        def retract_scissors(self):
            entry = dict(tick=self.tick, closed_before=self.jaws_closed(),
                         ready_before=self.at_ready_pose())
            withdrawals.append(entry)
            accepted = super().retract_scissors()
            entry.update(accepted=bool(accepted), closed_after=self.jaws_closed(),
                         ready_after=self.at_ready_pose())
            return accepted

    with (patch.object(task, 'SimWorld', AuditedWorld), patch.object(task, 'OUT', args.out),
          patch.object(sys, 'argv', [task.__file__, *argv])):
        rc = task.main()
    raw = args.out / 'nominal.json'
    report = json.loads(raw.read_bytes())
    trace = report.get('trace', [])
    lengths = [r['cut_length_m'] for r in trace]
    first = next((r for r in trace if lo - eps <= r['cut_length_m'] <= hi + eps), None)
    final = report.get('final_observation', {})
    tick = first['tick'] if first is not None else None
    after_target = [g for g in grips if tick is not None and g['tick'] >= tick]
    checks = {
        'done_clean': (rc == 0 and report['success'] and report['control_source'] == 'truth'
                       and report['final_state'] == 'DONE' and not report['acceptance_issues']
                       and final.get('torn') == 0 and not final.get('faults')
                       and report.get('tissue', {}).get('torn_events') == 0),
        'bounded_from_zero': (bool(lengths) and abs(lengths[0]) <= eps
                              and [r['tick'] for r in trace] == list(range(1, len(trace) + 1))
                              and all(0 <= length <= hi + eps for length in lengths)
                              and all(a <= b + eps for a, b in zip(lengths, lengths[1:]))),
        'cutting_closures_before_first_in': (tick is not None and bool(closures)
                                             and all(c['accepted'] and c['tick'] < tick for c in closures)),
        'no_cutter_grip_at_or_after_first_in': tick is not None and not after_target,
        'closed_withdrawal_at_first_in': (tick is not None and len(withdrawals) == 1
                                         and withdrawals[0] == dict(
                                             tick=tick, closed_before=True, ready_before=False,
                                             accepted=True, closed_after=True, ready_after=True)),
        'length_retained_and_cut_to_done': (
            first is not None and abs(final.get('cut_length_m', -1) - first['cut_length_m']) <= eps
            and 'CUT->DONE' in report.get('transitions', [])),
    }
    result = dict(protocol=reg, registration_sha256=task.sha256(registration),
                  nominal_sha256=task.sha256(raw), checks=checks, passed=all(checks.values()),
                  strict_visual_m5_passed=False, first_in=first,
                  final_observation=final, trace=trace,
                  cutter_grip_commands=grips, cutting_closures=closures, withdrawals=withdrawals)
    task.write_json(args.out / 'decision.json', result)
    print('Level 0:', 'PASS' if result['passed'] else 'FAIL', flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
