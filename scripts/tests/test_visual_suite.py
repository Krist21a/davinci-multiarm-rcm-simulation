"""Visual-suite regressions: subprocess mocks and temporary reports, never a simulator.

Run: python -m unittest discover -s scripts/tests -p test_visual_suite.py -v
"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

VISION = Path(__file__).resolve().parents[1] / 'vision'
with patch.object(sys, 'path', [str(VISION), *sys.path]):
    import run_visual_task_suite as suite


class VisualSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'visual_task_nominal.json'

    def report(self, run_id, **extra):
        return dict(run_id=run_id, success=True, final_block_to_target_m=0.001,
                    servo_cycles=12, commands_rejected=0, unexpected_contacts=[],
                    max_psm_rcm_error_m=1e-8, **extra)

    def invoke(self, child, runs=None):
        with patch.object(suite, 'ENDO_VALIDATION', self.root), \
                patch.object(suite, 'RUNS', runs or [('nominal', [], False)]), \
                patch.object(suite.subprocess, 'run', side_effect=child) as process, \
                patch.object(sys, 'argv', ['run_visual_task_suite.py', '--run-timeout', '5']), \
                contextlib.redirect_stdout(io.StringIO()):
            code = suite.main()
        result = json.loads((self.root / 'visual_task_summary.json').read_text(encoding='utf-8'))
        return code, result, process

    def write_report(self, data):
        self.output.write_text(json.dumps(data), encoding='utf-8')

    def test_exit_99_cannot_reuse_old_success(self):
        self.write_report(self.report('previous-invocation'))
        old = self.output.read_bytes()
        code, summary, _ = self.invoke(lambda *a, **kw: subprocess.CompletedProcess(a[0], 99, '', 'failed'))
        row = summary['runs'][0]
        self.assertEqual(code, 1)
        self.assertIs(row['success'], False)
        self.assertEqual(summary['successes'], 0)
        self.assertEqual(row['exit_code'], 99)
        self.assertIn('run_id', row['report_error'])
        self.assertEqual(row['stderr_tail'], 'failed')
        self.assertEqual(self.output.read_bytes(), old)

    def test_exit_zero_still_rejects_old_report(self):
        self.write_report(self.report('previous-invocation'))
        code, summary, _ = self.invoke(lambda *a, **kw: subprocess.CompletedProcess(a[0], 0))
        self.assertEqual(code, 1)
        self.assertIs(summary['runs'][0]['success'], False)

    def test_fresh_success_and_traceability(self):
        def child(cmd, **kwargs):
            self.assertEqual(kwargs['timeout'], 5)
            self.assertEqual(cmd[cmd.index('--name') + 1], 'nominal')
            run_id = cmd[cmd.index('--run-id') + 1]
            self.write_report(self.report(run_id, controller_sha256='controller-hash', source_sha256={'task.py': 'hash'}))
            return subprocess.CompletedProcess(cmd, 0)

        code, summary, _ = self.invoke(child)
        row = summary['runs'][0]
        self.assertEqual(code, 0)
        self.assertEqual(summary['successes'], 1)
        self.assertIs(row['success'], True)
        self.assertIsNone(row['report_error'])
        self.assertEqual(row['final_block_to_target_mm'], 1.0)
        self.assertEqual(row['controller_sha256'], 'controller-hash')
        self.assertEqual(row['source_sha256'], {'task.py': 'hash'})
        self.assertNotEqual(summary['run_id'], row['run_id'])
        self.assertTrue(summary['started_at_utc'])
        for digest in summary['source_sha256'].values():
            self.assertRegex(digest, r'^[0-9a-f]{64}$')

    def test_fresh_success_with_nonzero_exit_fails(self):
        def child(cmd, **kwargs):
            self.write_report(self.report(cmd[cmd.index('--run-id') + 1]))
            return subprocess.CompletedProcess(cmd, 99)

        code, summary, _ = self.invoke(child)
        self.assertEqual(code, 1)
        self.assertIsNone(summary['runs'][0]['report_error'])
        self.assertIs(summary['runs'][0]['success'], False)

    def test_missing_and_malformed_reports_fail_without_crashing(self):
        cases = [None, '{', '[]', 'null', 'true', '"success"', '{}', '\ufffd']
        for raw in cases:
            with self.subTest(raw=raw):
                if self.output.exists():
                    self.output.unlink()
                if raw is not None:
                    self.output.write_text(raw, encoding='utf-8')
                code, summary, _ = self.invoke(lambda *a, **kw: subprocess.CompletedProcess(a[0], 0))
                row = summary['runs'][0]
                self.assertEqual(code, 1)
                self.assertIs(row['success'], False)
                self.assertIsNotNone(row['report_error'])
                self.assertIsNone(row['final_block_to_target_mm'])

    def test_invalid_fields_in_current_report_fail(self):
        changes = [{'success': 'true'}, {'success': 1}, {'final_block_to_target_m': None},
                   {'final_block_to_target_m': '0.001'}, {'final_block_to_target_m': float('nan')},
                   {'final_block_to_target_m': float('inf')}, {'final_block_to_target_m': -1},
                   {'final_block_to_target_m': 10 ** 400},
                   {'final_block_to_target_m': True}, {'unexpected_contacts': None},
                   {'servo_cycles': '12'}, {'commands_rejected': -1}, {'max_psm_rcm_error_m': float('nan')}]
        for change in changes:
            with self.subTest(change=change):
                def child(cmd, **kwargs):
                    report = self.report(cmd[cmd.index('--run-id') + 1])
                    report.update(change)
                    self.write_report(report)
                    return subprocess.CompletedProcess(cmd, 0)

                code, summary, _ = self.invoke(child)
                self.assertEqual(code, 1)
                self.assertIs(summary['runs'][0]['success'], False)
                self.assertIsNotNone(summary['runs'][0]['report_error'])

    def test_missing_required_measurement_fails(self):
        def child(cmd, **kwargs):
            self.write_report(dict(run_id=cmd[cmd.index('--run-id') + 1], success=True))
            return subprocess.CompletedProcess(cmd, 0)

        code, summary, _ = self.invoke(child)
        self.assertEqual(code, 1)
        self.assertIn('final_block_to_target_m', summary['runs'][0]['report_error'])

    def test_reported_failure_and_invalid_encoding_fail(self):
        def child(cmd, **kwargs):
            report = self.report(cmd[cmd.index('--run-id') + 1])
            report['success'] = False
            self.write_report(report)
            return subprocess.CompletedProcess(cmd, 0)

        code, summary, _ = self.invoke(child)
        self.assertEqual(code, 1)
        self.assertIs(summary['runs'][0]['success'], False)
        self.output.write_bytes(b'\xff\xfe')
        code, summary, _ = self.invoke(lambda *a, **kw: subprocess.CompletedProcess(a[0], 0))
        self.assertEqual(code, 1)
        self.assertIn('UnicodeDecodeError', summary['runs'][0]['report_error'])

    def test_timeout_or_launch_failure_cannot_reuse_report(self):
        for failure in (subprocess.TimeoutExpired('child', 5), OSError('cannot launch child')):
            with self.subTest(failure=failure):
                self.write_report(self.report('old-run'))
                code, summary, _ = self.invoke(failure)
                row = summary['runs'][0]
                self.assertEqual(code, 1)
                self.assertIs(row['success'], False)
                self.assertIsNone(row['exit_code'])
                self.assertIsNotNone(row['process_error'])
                self.assertEqual(row['timed_out'], isinstance(failure, subprocess.TimeoutExpired))

    def test_run_ids_are_unique_even_for_same_name(self):
        ids = []

        def child(cmd, **kwargs):
            ids.append(cmd[cmd.index('--run-id') + 1])
            self.write_report(self.report(ids[-1]))
            return subprocess.CompletedProcess(cmd, 0)

        code, summary, _ = self.invoke(child, [('nominal', [], False)] * 2)
        self.assertEqual(code, 0)
        self.assertEqual(summary['total'], 2)
        self.assertEqual(len(set(ids)), 2)


class VisualTaskSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Suite/report tests above need only the stdlib; task math uses existing runtime dependencies.
        try:
            with patch.object(sys, 'path', [str(VISION), *sys.path]):
                import visual_servo_task
        except ImportError as exc:
            raise unittest.SkipTest(f'visual task runtime dependencies unavailable: {exc}')
        cls.task_module = visual_servo_task

    def run_task(self, lift_missing=None, retract='ok', home=True):
        task_module = self.task_module
        np = task_module.np
        frames = []

        def frame(block=1.0, marker=1.0):
            return dict(block=None if block is None else dict(p=np.array([0., 0., block])),
                        marker=None if marker is None else dict(p=np.array([0., 0., marker])),
                        target=dict(p=np.array([0., 0., 1.])))

        frames += [frame()]
        frames += [frame(marker=0.9)] * 3                  # pre-grasp: two hits + phase report
        frames += [frame()] * 3                            # grasp
        frames += [frame(block=None if lift_missing == 'block' else 1.0,
                         marker=None if lift_missing == 'grasp' else 1.0)]
        frames += [frame(block=0.9)] * 6                   # lift, pre-place
        frames += [frame()] * 3                            # place
        frames += [frame(marker=None if retract == 'missing_initial' else 1.0)]
        frames += [frame(marker=0.9 if retract == 'ok' else None if retract == 'lost' else 1.0)] * 8

        class Camera:
            h = 4

            def grab(self):
                self.obs = frames.pop(0)

            def mask(self, color):
                return color

            def locate(self, color, **kwargs):
                return self.obs.get(color)

            def locate_ring(self, color, *args):
                return self.obs.get(color)

        config = dict(arm='psm2', rcm_world=[0, 0, 0], axis_home=[0, 0, 1], transverse_x=[1, 0, 0],
                      transverse_y=[0, 1, 0], insertion_home=0, rcm_distance_home=0,
                      tool_shape='tool', port_radius_m=0.01)
        task = dict(arm='psm2', grasp_offset_m=0, approach_offset_m=0.1,
                    tip_marker=dict(offset_from_tcp_m=0, color='marker', radius_m=0.01),
                    block=dict(alias='block', color='block', size_m=0.01),
                    target=dict(center_world=[0, 0, 1], color='target', radius_m=0.01, tube_m=0.001),
                    planning=dict(command_sequence_checked=[[0, 0, 0], [0, 0, 0]]),
                    grip_open_rad=0.5, grip_hold_rad=0.1)
        by = dict(RCMController=1, psm2_joint8=2, psm2_tcp=3, camera=4, block=5)
        sim = Mock()
        sim.getObjectMatrix.return_value = np.eye(4)[:3].flatten().tolist()
        sim.getObjectPosition.return_value = [0., 0., 1.]
        sim.getSimulationTime.return_value = 0.0
        released = False
        events = []

        def controller(name, *args):
            nonlocal released
            if name == 'homeRCM':
                events.append('home')
                return home
            if name == 'getRCMStats':
                return [dict(target=[0, 0, 0], last_collision='test rejection')]
            if name == 'release':
                released = True
            if name == 'setRCMTarget' and released and retract == 'rejected':
                return False
            return True

        sim.callScriptFunction.side_effect = controller
        sim.startSimulation.side_effect = lambda: events.append('start')
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            output = Path(tmp)
            replacements = dict(ENDO_VALIDATION=output,
                                read_json=Mock(side_effect=[{'camera': {'alias': 'camera'}}, task, [config], {'labels': []}]),
                                connect=Mock(return_value=(Mock(), sim)),
                                load_scene=Mock(side_effect=lambda *a: events.append('load')),
                                aliases=Mock(return_value=by),
                                refresh_controller=Mock(side_effect=lambda *a: events.append('refresh') or 'verified-controller'),
                                Endoscope=Mock(return_value=Camera()), stop_simulation=Mock(),
                                relative=Mock(side_effect=lambda path: Path(path).name),
                                sha256=Mock(return_value='file-hash'))
            for name, value in replacements.items():
                stack.enter_context(patch.object(task_module, name, value))
            stack.enter_context(patch.object(task_module.instrument_guard, 'require', return_value={'model': 'test'}))
            stack.enter_context(patch.object(task_module.ecm_pairs, 'create_probe', return_value=6))
            stack.enter_context(patch.object(task_module.ecm_pairs, 'scan', return_value=[]))
            stack.enter_context(patch.object(sys, 'argv', ['visual_servo_task.py', '--run-id', 'test-run', '--max-iter', '6']))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if not home:
                with self.assertRaisesRegex(RuntimeError, 'homeRCM rejected'):
                    task_module.main()
                self.assertFalse((output / 'visual_task_nominal.json').exists())
                self.assertEqual([c.args[0] for c in sim.callScriptFunction.call_args_list], ['homeRCM'])
                result, code = None, None
            else:
                code = task_module.main()
                result = json.loads((output / 'visual_task_nominal.json').read_text(encoding='utf-8'))
            replacements['refresh_controller'].assert_called_once_with(sim, 1)
            replacements['stop_simulation'].assert_called_once()
            sim.setStepping.assert_called_with(False)
        self.assertEqual(events, ['load', 'refresh', 'start', 'home'])
        return code, result

    def test_success_requires_completed_retraction_and_records_provenance(self):
        code, report = self.run_task()
        self.assertEqual(code, 0)
        self.assertIs(report['success'], True)
        self.assertIs(report['retracted'], True)
        self.assertEqual(report['run_id'], 'test-run')
        self.assertEqual(report['controller_sha256'], 'verified-controller')
        self.assertTrue(report['source_sha256'])
        self.assertTrue(report['config_sha256'])

    def test_missing_first_lift_detection_fails_without_exception(self):
        for key in ('block', 'grasp'):
            with self.subTest(key=key):
                code, report = self.run_task(lift_missing=key)
                self.assertEqual(code, 1)
                self.assertIs(report['success'], False)
                self.assertIs(report['retracted'], False)
                self.assertEqual(report['phases'][-1]['reason'], 'initial lift detection failed')

    def test_retract_missing_rejected_or_lost_is_not_success(self):
        for mode in ('missing_initial', 'rejected', 'lost'):
            with self.subTest(mode=mode):
                code, report = self.run_task(retract=mode)
                self.assertEqual(code, 1)
                self.assertIs(report['released'], True)
                self.assertIs(report['retracted'], False)
                self.assertIs(report['success'], False)
                self.assertEqual(report['final_block_to_target_m'], 0.0)
                self.assertEqual(report['phases'][-1]['phase'], 'retract')
                self.assertIs(report['phases'][-1]['converged'], False)

    def test_home_rejection_aborts_before_task_commands(self):
        self.run_task(home=False)


if __name__ == '__main__':
    unittest.main()