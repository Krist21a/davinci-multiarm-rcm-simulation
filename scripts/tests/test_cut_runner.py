"""Cut runner lifecycle regressions; all simulator access is mocked, outputs stay in temporary dirs.

Run: python -m unittest discover -s scripts/tests -p test_cut_runner.py -v
"""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

TASK = Path(__file__).resolve().parents[1] / 'task'
with patch.object(sys, 'path', [str(TASK), *sys.path]):
    import run_cut_task as runner


class CutRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'validation' / 'task'
        self.report_path = self.output / 'sim_cut_only_nominal.json'
        config_path = self.root / 'config' / 'task' / 'cut_task.json'
        config_path.parent.mkdir(parents=True)
        self.cfg = dict(cut=dict(length_m=0.01, max_overcut_m=0.001),
                        tension=dict(tear_is_failure=True, high_strain=0.2))
        config_path.write_text(json.dumps(self.cfg), encoding='utf-8')
        self.scene = self.root / 'scene.ttt'
        self.scene.write_bytes(b'mocked scene, never opened by a simulator')
        self.events = []
        self.client, self.sim = Mock(), Mock()
        self.sim.startSimulation.side_effect = lambda: self.events.append('start')
        self.world = Mock(controller_sha256='actual-injected-controller-sha256', roles_report={'psm2': 'scissors'},
                          camera_moves=2, image=None, W=32, H=24)
        self.world.observe.return_value = SimpleNamespace(cut_length_m=0.01, tension=0.1, max_strain=0.15, torn=0, faults=[])
        self.world.call.return_value = dict(cut=10, torn=0, cut_events=4, torn_events=0)
        self.fsm = Mock(state=runner.S.INIT, log=[])

        def go(new, reason, obs=None):
            self.fsm.log.append(dict(tick=1, frm=self.fsm.state.name, to=new.name, reason=reason))
            self.fsm.state = new

        self.fsm.go.side_effect = go
        self.fsm.report.side_effect = lambda: dict(final_state=self.fsm.state.name, reason='test result', log=self.fsm.log)

    def invoke(self, *, init_error=None, fsm_init_error=None, run_error=None, observe_error=None,
               stop_error=None, state=None, write=None, image_error=None, scene=None):
        state = state or runner.S.DONE

        def world_init(*args, **kwargs):
            self.assertEqual(args[2].scene, scene or self.scene)
            self.sim.startSimulation()
            if init_error:
                raise init_error
            return self.world

        def run():
            self.events.append('run')
            self.fsm.go(state, 'test transition')
            if run_error:
                raise run_error

        def stop(client, sim):
            self.assertIs(client, self.client)
            self.assertIs(sim, self.sim)
            self.events.append('stop')
            if stop_error:
                raise stop_error

        self.fsm.run.side_effect = run
        if observe_error:
            self.world.observe.side_effect = observe_error
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'REPO', self.root))
            stack.enter_context(patch.object(runner, 'OUT', self.output))
            stack.enter_context(patch.object(runner, 'INSTRUMENT_SCENE', self.scene))
            stack.enter_context(patch.object(runner, 'relative', side_effect=lambda p: str(p)))
            stack.enter_context(patch.object(runner, 'connect', return_value=(self.client, self.sim)))
            stack.enter_context(patch.object(runner, 'SimWorld', side_effect=world_init))
            stack.enter_context(patch.object(runner, 'CutTaskFSM', return_value=self.fsm, side_effect=fsm_init_error))
            cleanup = stack.enter_context(patch.object(runner, 'stop_simulation', side_effect=stop))
            argv = ['run_cut_task.py'] + (['--scene', str(scene)] if scene is not None else [])
            stack.enter_context(patch.object(sys, 'argv', argv))
            stdout, stderr = io.StringIO(), io.StringIO()
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            if write is not None:
                stack.enter_context(patch.object(runner, 'write_json', side_effect=write))
            if image_error is not None:
                stack.enter_context(patch.object(runner.Image.Image, 'save', side_effect=image_error))
            code = runner.main()
        cleanup.assert_called_once_with(self.client, self.sim)
        if self.world.observe.called:
            self.world.observe.assert_called_once_with(inject_events=False)
        report = json.loads(self.report_path.read_text(encoding='utf-8')) if self.report_path.exists() else None
        return code, report, stderr.getvalue()

    def assert_failed(self, code, report, error=None):
        self.assertNotEqual(code, 0)
        self.assertIs(report['success'], False)
        self.assertRegex(report['run_id'], r'^[0-9a-f]{32}$')
        self.assertIn('cleanup_error', report)
        if error:
            self.assertIn(error, report['error'])

    def test_world_init_failure_after_start_stops_and_writes_current_failure(self):
        self.output.mkdir(parents=True)
        self.report_path.write_text('{"run_id":"old-run","success":true}', encoding='utf-8')
        code, report, _ = self.invoke(init_error=RuntimeError('init after start failed'))
        self.assert_failed(code, report, 'world initialization: RuntimeError: init after start failed')
        self.assertEqual(self.events, ['start', 'stop'])
        self.assertIsNone(report['controller_sha256'])
        self.fsm.run.assert_not_called()

    def test_fsm_init_failure_stops_without_using_missing_fsm(self):
        code, report, _ = self.invoke(fsm_init_error=RuntimeError('fsm init failed'))
        self.assert_failed(code, report, 'FSM initialization: RuntimeError: fsm init failed')
        self.assertEqual(report['controller_sha256'], self.world.controller_sha256)
        self.assertEqual(self.events, ['start', 'stop'])

    def test_run_failure_is_reported_and_stopped(self):
        code, report, _ = self.invoke(run_error=RuntimeError('run failed'))
        self.assert_failed(code, report, 'FSM run: RuntimeError: run failed')
        self.assertEqual(self.events, ['start', 'run', 'stop'])
        self.world.observe.assert_not_called()

    def test_final_observation_failure_is_reported_and_stopped(self):
        code, report, _ = self.invoke(observe_error=RuntimeError('observe failed'))
        self.assert_failed(code, report, 'final observation: RuntimeError: observe failed')
        self.world.call.assert_not_called()

    def test_final_statistics_failure_is_reported(self):
        self.world.call.side_effect = RuntimeError('statistics failed')
        code, report, _ = self.invoke()
        self.assert_failed(code, report, 'final tissue statistics: RuntimeError: statistics failed')

    def test_cleanup_failure_prevents_done_success(self):
        code, report, _ = self.invoke(stop_error=RuntimeError('stop failed'))
        self.assert_failed(code, report)
        self.assertIsNone(report['error'])
        self.assertEqual(report['cleanup_error'], 'RuntimeError: stop failed')
        self.assertEqual(report['final_state'], 'DONE')

    def test_cleanup_error_does_not_replace_initialization_or_run_error(self):
        for failure in ('init_error', 'run_error'):
            with self.subTest(failure=failure):
                code, report, _ = self.invoke(**{failure: ValueError('primary failure')},
                                              stop_error=RuntimeError('secondary failure'))
                self.assert_failed(code, report, 'ValueError: primary failure')
                self.assertEqual(report['cleanup_error'], 'RuntimeError: secondary failure')

    def test_fsm_failed_returns_one_even_with_acceptable_final_observation(self):
        code, report, _ = self.invoke(state=runner.S.FAILED)
        self.assertEqual(code, 1)
        self.assert_failed(code, report)
        self.assertEqual(report['final_state'], 'FAILED')
        self.assertEqual(report['acceptance_issues'], [])

    def test_done_with_failed_acceptance_returns_one(self):
        self.world.observe.return_value.torn = 1
        code, report, _ = self.invoke()
        self.assert_failed(code, report)
        self.assertEqual(report['final_state'], 'DONE')
        self.assertIn('tissue torn (1 constraints)', report['acceptance_issues'])

    def test_normal_done_preserves_report_and_real_hashes(self):
        code, report, _ = self.invoke()
        self.assertEqual(code, 0)
        self.assertIs(report['success'], True)
        self.assertIsNone(report['error'])
        self.assertIsNone(report['cleanup_error'])
        self.assertEqual(report['controller_sha256'], 'actual-injected-controller-sha256')
        self.assertEqual(report['final_state'], 'DONE')
        self.assertEqual(report['scene_sha256'], hashlib.sha256(self.scene.read_bytes()).hexdigest())
        for path, digest in report['source_sha256'].items():
            self.assertEqual(digest, hashlib.sha256(Path(path).read_bytes()).hexdigest())
        self.assertEqual(report['final_observation']['cut_length_m'], 0.01)
        self.assertEqual(report['tissue']['cut_events'], 4)
        self.assertEqual(report['camera_moves'], 2)
        self.assertTrue((self.output / 'sim_cut_only_nominal' / '00_DONE.png').exists())
        self.assertEqual(list(self.output.glob('*.tmp')), [])

    def test_screenshot_error_cannot_produce_success(self):
        code, report, _ = self.invoke(image_error=OSError('image write failed'))
        self.assert_failed(code, report, 'screenshots: OSError: image write failed')

    def test_selected_scene_is_passed_to_world_and_hashed(self):
        scene = self.root / 'articulated.ttt'
        scene.write_bytes(b'mocked articulated scene')
        code, report, _ = self.invoke(scene=scene)
        self.assertEqual(code, 0)
        self.assertEqual(report['scene'], str(scene.resolve()))
        self.assertEqual(report['scene_sha256'], hashlib.sha256(scene.read_bytes()).hexdigest())
        self.assertNotEqual(report['scene_sha256'], hashlib.sha256(self.scene.read_bytes()).hexdigest())

    def test_fsm_report_error_still_emits_failure(self):
        self.fsm.report.side_effect = RuntimeError('fsm report failed')
        code, report, _ = self.invoke()
        self.assert_failed(code, report, 'FSM report: RuntimeError: fsm report failed')

    def test_report_write_error_retries_with_current_failure_not_success(self):
        real_write = runner.write_json
        writes = []

        def write(path, data):
            writes.append(data.copy())
            if len(writes) == 1:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{partial', encoding='utf-8')
                raise OSError('report write failed')
            real_write(path, data)

        code, report, _ = self.invoke(write=write)
        self.assert_failed(code, report, 'report write: OSError: report write failed')
        self.assertEqual(len(writes), 2)
        self.assertEqual(writes[0]['run_id'], report['run_id'])
        self.assertIn('report_error', report)
        self.assertEqual(list(self.output.glob('*.tmp')), [])

    def test_permanent_storage_failure_returns_failure_and_emits_current_id(self):
        code, report, stderr = self.invoke(write=OSError('disk unavailable'))
        self.assertEqual(code, 1)
        self.assertIsNone(report)
        self.assertIn('cannot publish report', stderr)
        self.assertIn('"success": false', stderr)
        self.assertIn('"run_id":', stderr)

    def test_serialization_failure_does_not_hide_primary_run_error(self):
        self.fsm.report.side_effect = lambda: dict(final_state='FAILED', unencodable=object())
        code, report, _ = self.invoke(run_error=RuntimeError('original run failure'))
        self.assert_failed(code, report, 'FSM run: RuntimeError: original run failure')
        self.assertIn('TypeError', report['report_error'])
        self.assertEqual(list(self.output.glob('*.tmp')), [])

    def test_atomic_write_does_not_replace_old_report_with_partial_json(self):
        self.output.mkdir(parents=True)
        old = '{"run_id":"previous","success":true}'
        self.report_path.write_text(old, encoding='utf-8')
        with patch.object(runner.Path, 'replace', side_effect=OSError('replace failed')):
            code, report, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(self.report_path.read_text(encoding='utf-8'), old)
        self.assertIn('cannot publish report', stderr)
        self.assertEqual(list(self.output.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()