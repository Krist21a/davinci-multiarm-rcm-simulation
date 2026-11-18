"""World-interface regressions: unittest + numpy only, no simulator or global dependency mocks.

Run: python -m unittest discover -s scripts/tests -p test_sim_world_safety.py -v
"""
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

TASK = Path(__file__).resolve().parents[1] / 'task'
with patch.object(sys, 'path', [str(TASK), *sys.path]):
    import sim_world
    import simtools
    from sim_world import SimWorld, TISSUE_RGB
    from visibility import depth_visible


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_load_heavy_simulation_dependencies(self):
        # A separate process avoids sys.modules pollution and proves a direct import works
        # even when optional runtime dependencies are unavailable.
        code = f"""
import importlib.abc
import sys
class RejectRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'PIL', 'trimesh', 'scipy', 'coppeliasim_zmqremoteapi_client', 'instrument_guard'}}:
            raise ImportError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RejectRuntimeImports())
sys.path.insert(0, {str(TASK)!r})
from sim_world import SimWorld
assert SimWorld is not None
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ControllerStartupTests(unittest.TestCase):
    def make_world(self, refresh_error=None):
        sim = Mock()
        events = []
        config = [[{'arm': 'psm2'}], {}, {'camera': {'alias': 'camera', 'resolution': [41, 41],
                                                    'view_angle_deg': 60, 'far_m': 1.0}}]

        def refresh(*args, **kwargs):
            events.append('refresh')
            if refresh_error:
                raise refresh_error
            return 'verified-controller-sha256'

        with patch.object(sim_world, 'read_json', side_effect=config), \
                patch.object(sim_world, 'load_scene', side_effect=lambda *a: events.append('load')), \
                patch.object(sim_world, 'aliases', return_value={'RCMController': 10, 'camera': 11}), \
                patch.object(simtools, 'refresh_controller', side_effect=refresh) as update, \
                patch.object(SimWorld, 'start', side_effect=lambda: events.append('start')) as start:
            if refresh_error:
                with self.assertRaisesRegex(RuntimeError, 'refresh failed'):
                    SimWorld(None, sim, None, {})
                start.assert_not_called()
                world = None
            else:
                world = SimWorld(None, sim, None, {})
            update.assert_called_once_with(sim, 10, tissue=True)
        return world, events

    def test_loaded_controller_is_refreshed_before_start(self):
        world, events = self.make_world()
        self.assertEqual(events, ['load', 'refresh', 'start'])
        self.assertEqual(world.controller_sha256, 'verified-controller-sha256')

    def test_refresh_failure_never_starts_simulation(self):
        _, events = self.make_world(RuntimeError('refresh failed'))
        self.assertEqual(events, ['load', 'refresh'])

    def test_articulated_frame_uses_yaw_frame_not_pitch_to_hinge_chord(self):
        world = SimWorld.__new__(SimWorld)
        world.by = {'psm2_joint9': 9, 'psm2_joint10': 10, 'psm2_tcp': 20, 'psm2_dvrk_yaw_frame': 30}
        matrices = {9: np.eye(4)[:3], 10: np.eye(4)[:3], 30: np.eye(4)[:3]}
        matrices[10] = matrices[10].copy()
        matrices[10][:, 3] = [.01, .002, 0]
        world.sim = Mock()
        world.sim.getObjectMatrix.side_effect = lambda h, _: matrices[h].reshape(-1).tolist()
        world.sim.getObjectPosition.return_value = [.01, .02, 0]
        frame = world.frame('psm2')
        np.testing.assert_allclose(frame['x'], [0, 1, 0])
        np.testing.assert_allclose(frame['axis'], [0, 0, 1])


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.world = SimWorld.__new__(SimWorld)
        self.world.W = self.world.H = 41
        self.world.f = 20.0
        self.world.far = 1.0
        self.world.camera = Mock(return_value=(np.eye(3), np.zeros(3)))
        self.world.depth = np.full((41, 41), 0.1)
        self.world.image = np.full((41, 41, 3), 255 * np.array(TISSUE_RGB))
        self.point = np.array([0.0, 0.0, 0.1])

    def visible(self, **kwargs):
        return self.world.point_visible(self.point, **kwargs)[0]

    def test_clear_point_is_visible_with_correct_offset(self):
        self.assertEqual(self.world.point_visible(self.point), (True, 0.0))

    def test_tissue_colour_cannot_override_nearer_depth(self):
        self.world.depth.fill(0.09)
        self.assertFalse(self.visible())

    def test_occluded_centre_with_clear_surroundings_is_not_visible(self):
        self.world.depth[20, 20] = 0.09
        self.assertFalse(self.visible())

    def test_entire_small_central_window_must_pass_depth_gate(self):
        for row in range(19, 22):
            for col in range(19, 22):
                with self.subTest(row=row, col=col):
                    self.world.depth.fill(0.1)
                    self.world.depth[row, col] = 0.09
                    self.assertFalse(self.visible())

    def test_clear_centre_still_requires_surrounding_support(self):
        self.world.depth.fill(0.09)
        self.world.depth[19:22, 19:22] = 0.1
        self.assertFalse(self.visible())

    def test_default_depth_tolerance_is_not_relaxed(self):
        self.world.depth.fill(self.point[2] - 0.003)
        self.assertTrue(self.visible())
        self.world.depth.fill(self.point[2] - 0.003 - 1e-8)
        self.assertFalse(self.visible())

    def test_invalid_depth_cannot_supply_central_or_surrounding_support(self):
        for invalid in (np.nan, np.inf, -np.inf, 0.0, -0.1):
            with self.subTest(invalid=invalid):
                self.world.depth.fill(0.1)
                self.world.depth[20, 20] = invalid
                self.assertFalse(self.visible())
                self.world.depth.fill(invalid)
                self.world.depth[19:22, 19:22] = 0.1
                self.assertFalse(self.visible())

    def test_no_rgb_dependency_in_depth_gate(self):
        self.world.image = None
        self.assertTrue(self.visible())
        self.world.depth.fill(0.09)
        self.assertFalse(self.visible())

    def test_localize_cannot_bypass_depth_with_tissue_colour(self):
        self.world.render = Mock()
        self.world.sheet_origin = self.point
        self.world.nodes = Mock()
        self.world.depth.fill(0.09)
        self.assertIsNone(self.world.localize())
        self.world.nodes.assert_not_called()

    def test_depth_support_threshold_is_still_enforced(self):
        # 10/21 rows plus the centre are clear: enough for 0.35, not enough for 0.75.
        self.world.depth.fill(0.09)
        self.world.depth[10:20, 10:31] = 0.1
        self.world.depth[19:22, 19:22] = 0.1
        self.assertTrue(self.visible())
        self.assertFalse(self.visible(min_clear=0.75))

    def test_invalid_and_out_of_view_points_fail_closed(self):
        for point in ([0, 0, -0.1], [0, 0, 0], [0, 0, 1.0], [0, 0, 1.1],
                      [np.nan, 0, 0.1], [0, 0, np.inf], [0.2, 0, 0.1]):
            with self.subTest(point=point):
                with np.errstate(invalid='ignore'):
                    self.assertEqual(self.world.point_visible(point), (False, None))

    def test_depth_helper_rejects_invalid_frame_or_parameters(self):
        self.assertFalse(depth_visible(np.ones(41), (20, 20), 0.1))
        for uv in ((0, 0), (40, 40), (np.nan, 20), (20, np.inf)):
            self.assertFalse(depth_visible(self.world.depth, uv, 0.1))
        for kwargs in ({'tol': -1}, {'tol': np.nan}, {'window': 0}, {'window': 1.5},
                       {'min_clear': -0.1}, {'min_clear': np.nan}, {'min_clear': 1.1}):
            self.assertFalse(depth_visible(self.world.depth, (20, 20), 0.1, **kwargs))


def bare_world():
    world = SimWorld.__new__(SimWorld)
    world.index = {'psm1': 1, 'psm2': 2, 'psm3': 3}
    world.call = Mock()
    world.steps = Mock()
    return world


class CommandSafetyTests(unittest.TestCase):
    def test_command_rejection_does_not_step(self):
        for refused in (False, None):
            with self.subTest(refused=refused):
                world = bare_world()
                world.call.return_value = refused
                self.assertIs(world.command('psm2', 1, 2, 0.03, 7), False)
                world.steps.assert_not_called()

    def test_accepted_command_steps_exactly_requested_amount(self):
        world = bare_world()
        world.call.return_value = True
        self.assertIs(world.command('psm2', 1, 2, 0.03, 7), True)
        world.call.assert_called_once_with('setRCMTarget', 2, 1.0, 2.0, 0.03)
        world.steps.assert_called_once_with(7)

    def test_rejected_jaw_motion_does_not_step(self):
        for method in ('open_scissors', 'close_scissors'):
            with self.subTest(method=method):
                world = bare_world()
                world.call.return_value = False
                self.assertIs(getattr(world, method)(), False)
                world.steps.assert_not_called()

    def test_accepted_jaw_motion_keeps_existing_settling_time(self):
        for method, steps in (('open_scissors', 3), ('close_scissors', 8)):
            with self.subTest(method=method):
                world = bare_world()
                world.call.return_value = True
                self.assertIs(getattr(world, method)(), True)
                world.steps.assert_called_once_with(steps)

    def test_advance_only_accounts_for_accepted_motion(self):
        world = bare_world()
        world.advanced = 0.003
        world.target_of = Mock(return_value=(0, 0, 0.04))
        world.call.return_value = False
        self.assertIs(world.advance_scissors(0.003), False)
        self.assertEqual(world.advanced, 0.003)
        world.steps.assert_not_called()
        world.call.return_value = True
        self.assertIs(world.advance_scissors(0.003), True)
        self.assertEqual(world.advanced, 0.006)


class GraspSafetyTests(unittest.TestCase):
    def make_world(self, open_ok=True, close_ok=True, motion_ok=True, nodes=3):
        world = bare_world()
        world.cfg = {'grasp': {'min_nodes': 3}}
        world.arm_cfg = {'psm1': {'limits': {'yaw_deg': 30, 'pitch_deg': 30,
                                          'insertion_min': 0.0, 'insertion_max': 0.1}}}
        world.rcm_command_for = Mock(return_value=(0, 0, 0.04))
        world.target_of = Mock(return_value=(0, 0, 0.04))
        world.by = {'psm1_tcp': 101}
        world.sim = SimpleNamespace(getObjectPosition=Mock(return_value=(0, 0, 0)))
        world.grasped_points = {}
        # Model the pinning side effect so insufficient-node cleanup is tested, not merely its return value.
        world.pinned = 0

        def controller(name, *args):
            if name == 'setGrip':
                return open_ok if args[1] == 0.5 else close_ok
            if name == 'setRCMTarget':
                return motion_ok
            if name == 'tissueGrasp':
                world.pinned = nodes if nodes is not None and nodes > 0 else 0
                return nodes
            if name == 'tissueRelease':
                world.pinned = 0
                return nodes
            raise AssertionError(name)

        world.call.side_effect = controller
        return world

    def test_failed_closure_never_pins_tissue(self):
        for closed in (False, None):
            with self.subTest(closed=closed):
                world = self.make_world(close_ok=closed)
                ok, why = world.grasp('psm1', np.zeros(3))
                self.assertIs(ok, False)
                self.assertIn('close', why)
                self.assertEqual(world.pinned, 0)
                self.assertNotIn(call('tissueGrasp', 1), world.call.call_args_list)

    def test_insufficient_nodes_are_released(self):
        for nodes in (None, -1, 0, 1, 2):
            with self.subTest(nodes=nodes):
                world = self.make_world(nodes=nodes)
                ok, why = world.grasp('psm1', np.zeros(3))
                self.assertIs(ok, False)
                self.assertIn('nodes', why)
                self.assertEqual(world.pinned, 0)
                self.assertEqual(world.call.call_args_list[-2:], [call('tissueGrasp', 1), call('tissueRelease', 1)])

    def test_successful_grasp_retains_nodes(self):
        world = self.make_world()
        self.assertEqual(world.grasp('psm1', np.zeros(3)), (True, ''))
        self.assertEqual(world.pinned, 3)
        self.assertEqual(world.call.call_args_list[-2:], [call('setGrip', 1, 0.0), call('tissueGrasp', 1)])

    def test_grasp_records_the_held_point_and_release_clears_it(self):
        world = self.make_world()
        point = np.array([-0.012, 0.0, 0.05])
        self.assertTrue(world.grasp('psm1', point)[0])
        np.testing.assert_array_equal(world.grasped_points['psm1'], point)
        world.release('psm1')
        self.assertNotIn('psm1', world.grasped_points)

    def test_failed_grasp_leaves_no_stale_point(self):
        for nodes in (0, None):
            with self.subTest(nodes=nodes):
                world = self.make_world(nodes=nodes)
                world.grasped_points['psm1'] = np.array([9.0, 9.0, 9.0])
                self.assertFalse(world.grasp('psm1', np.zeros(3))[0])
                self.assertNotIn('psm1', world.grasped_points)

    def test_opening_or_approach_refusal_stops_before_closure(self):
        for options in ({'open_ok': False}, {'motion_ok': False}):
            with self.subTest(options=options):
                world = self.make_world(**options)
                self.assertIs(world.grasp('psm1', np.zeros(3))[0], False)
                self.assertNotIn(call('setGrip', 1, 0.0), world.call.call_args_list)
                self.assertNotIn(call('tissueGrasp', 1), world.call.call_args_list)
                world.steps.assert_not_called()


class HomeAndAlignmentTests(unittest.TestCase):
    def make_home_world(self, results):
        world = bare_world()
        world.call.side_effect = results
        world.task = {'planning': {'command_sequence_checked': [[0, 1, 0.04], [2, 1, 0.05]]}}
        world.place_sheet = Mock()
        return world

    def test_home_requires_explicit_success(self):
        for result in (False, None):
            world = self.make_home_world([result])
            self.assertIs(world.home(), False)
            world.call.assert_called_once_with('homeRCM')
            world.steps.assert_not_called()
            world.place_sheet.assert_not_called()

    def test_rejected_ready_pose_does_not_create_tissue(self):
        for results in ([True, False], [True, True, False]):
            world = self.make_home_world(results)
            self.assertIs(world.home(), False)
            world.steps.assert_called_once_with(20)
            world.place_sheet.assert_not_called()

    def test_successful_home_creates_tissue(self):
        world = self.make_home_world([True, True, True])
        self.assertIs(world.home(), True)
        self.assertEqual(world.steps.call_args_list, [call(20), call(30)])
        world.place_sheet.assert_called_once_with()

    def test_alignment_rejection_is_not_reported_as_zero_error(self):
        world = bare_world()
        world.frame = Mock(return_value={'tcp': np.array([0, 0, 0.01]), 'pivot': np.zeros(3)})
        world.cut_front = Mock(return_value=np.array([0, 0, 0.02]))
        world.v = np.array([0, 0, 1])
        world.target_of = Mock(return_value=(0, 0, 0.04))
        world.call.return_value = False
        self.assertIsNone(world.align_scissors(None, 0))
        world.steps.assert_not_called()
        world.call.return_value = True
        self.assertEqual(world.align_scissors(None, 0), 0.0)


class RetractionDirectionTests(unittest.TestCase):
    """retract() pulls in the direction of the side the arm actually holds — not its name.

    Handover can put psm3 on the -u (psm1) side; the old arm-name lookup then pulled it the
    wrong way: the sheet edge moved toward the cut line instead of away from it.
    """
    def make_world(self):
        world = bare_world()
        world.sheet_origin = np.zeros(3)
        world.u = np.array([1.0, 0.0, 0.0])               # +u = the "psm3 side" by layout convention
        world.by = {'psm1_tcp': 11, 'psm3_tcp': 13}
        world.sim = SimpleNamespace(getObjectPosition=Mock(return_value=[0.0, 0.0, 0.04]))
        world.arm_cfg = {a: {'limits': {'yaw_deg': 30.0, 'pitch_deg': 30.0,
                                        'insertion_min': 0.0, 'insertion_max': 0.1}}
                         for a in ('psm1', 'psm3')}
        world.target_of = Mock(return_value=(0.0, 0.0, 0.04))
        # Identity mapping: the commanded correction equals the requested tip displacement.
        world.rcm_command_for = Mock(side_effect=lambda arm, point: tuple(np.asarray(point, dtype=float)))
        world.command = Mock(return_value=True)
        world.grasped_points = {}
        return world

    def direction(self, world, arm, point, step=0.0002):
        world.grasped_points[arm] = np.asarray(point, dtype=float)
        ok = world.retract(arm, step)
        return ok, np.asarray(world.command.call_args[0][1:4], dtype=float) if world.command.called else None

    def test_pull_direction_follows_the_held_side_not_the_arm_name(self):
        world = self.make_world()
        ok, cmd = self.direction(world, 'psm3', [-0.012, 0.0, 0.05])   # handover: psm3 on the -u side
        self.assertTrue(ok)
        self.assertLess(cmd[0], 0.0)
        world.command.reset_mock()
        ok, cmd = self.direction(world, 'psm1', [0.012, 0.0, 0.05])    # unusual assignment: psm1 on +u
        self.assertTrue(ok)
        self.assertGreater(cmd[0], 0.0)
        world.command.reset_mock()
        ok, cmd = self.direction(world, 'psm3', [-0.012, 0.0, 0.05], step=-0.0002)  # relax pulls inward
        self.assertTrue(ok)
        self.assertGreater(cmd[0], 0.0)

    def test_retract_refuses_without_a_verified_held_side(self):
        world = self.make_world()
        self.assertFalse(world.retract('psm3', 0.0002))                # never grasped
        world.grasped_points['psm3'] = np.zeros(3)                     # exactly on the cut line
        self.assertFalse(world.retract('psm3', 0.0002))
        world.grasped_points['psm3'] = np.array([np.nan, 0.0, 0.05])   # corrupted record
        self.assertFalse(world.retract('psm3', 0.0002))
        world.command.assert_not_called()

    def test_retract_refuses_before_any_sheet_exists(self):
        world = self.make_world()
        world.sheet_origin = None
        world.grasped_points['psm1'] = np.array([-0.012, 0.0, 0.05])
        self.assertFalse(world.retract('psm1', 0.0002))
        world.command.assert_not_called()


class RetractionSafetyTests(unittest.TestCase):
    def make_world(self, grip=0.0, tissue=True, close_ok=True, motion_ok=True):
        world = bare_world()
        world.advanced = 0.006

        def controller(name, *args):
            if name == 'getRCMStats':
                return [{}, {'target': [1.0, 2.0, 0.04], 'grip': grip}]
            if name == 'tissueStats':
                return {'exists': tissue}
            if name == 'setGrip':
                return close_ok
            if name == 'setRCMTarget':
                return motion_ok
            raise AssertionError(name)

        world.call.side_effect = controller
        return world

    def test_closed_scissors_retract_without_another_closure_and_are_idempotent(self):
        world = self.make_world()
        self.assertIs(world.retract_scissors(), True)
        self.assertEqual(world.advanced, 0)
        self.assertEqual(world.call.call_args_list,
                         [call('getRCMStats'), call('setRCMTarget', 2, 1.0, 2.0, 0.034)])
        world.steps.assert_called_once_with(20)
        world.call.reset_mock()
        world.steps.reset_mock()
        self.assertIs(world.retract_scissors(), True)
        world.call.assert_not_called()
        world.steps.assert_not_called()

    def test_no_advance_requires_neither_closure_nor_motion(self):
        world = self.make_world(grip=0.5)
        world.advanced = 0
        self.assertIs(world.retract_scissors(), True)
        world.call.assert_not_called()
        world.steps.assert_not_called()

    def test_open_scissors_with_tissue_are_not_closed_merely_to_retract(self):
        for tissue in (True, None):
            world = self.make_world(grip=0.5, tissue=tissue)
            self.assertIs(world.retract_scissors(), False)
            self.assertEqual(world.call.call_args_list, [call('getRCMStats'), call('tissueStats')])
            self.assertEqual(world.advanced, 0.006)
            world.steps.assert_not_called()

    def test_failed_closure_without_tissue_stops_before_motion(self):
        for closed in (False, None):
            world = self.make_world(grip=0.5, tissue=False, close_ok=closed)
            self.assertIs(world.retract_scissors(), False)
            self.assertEqual(world.call.call_args_list,
                             [call('getRCMStats'), call('tissueStats'), call('setGrip', 2, 0.0)])
            self.assertEqual(world.advanced, 0.006)
            world.steps.assert_not_called()

    def test_successful_closure_without_tissue_allows_retraction(self):
        world = self.make_world(grip=0.5, tissue=False)
        self.assertIs(world.retract_scissors(), True)
        self.assertEqual(world.advanced, 0)
        self.assertEqual(world.call.call_args_list[-2:],
                         [call('setGrip', 2, 0.0), call('setRCMTarget', 2, 1.0, 2.0, 0.034)])

    def test_rejected_retraction_keeps_advance_for_retry_without_stepping(self):
        world = self.make_world(motion_ok=False)
        self.assertIs(world.retract_scissors(), False)
        self.assertEqual(world.advanced, 0.006)
        world.steps.assert_not_called()

    def test_unknown_grip_is_not_assumed_closed(self):
        for grip in (None, np.nan, np.inf):
            world = self.make_world(grip=grip)
            self.assertIs(world.retract_scissors(), False)
            world.call.assert_called_once_with('getRCMStats')
            world.steps.assert_not_called()

    def test_missing_tissue_status_does_not_allow_closure(self):
        world = self.make_world(grip=0.5)
        world.call.side_effect = [[{}, {'target': [1.0, 2.0, 0.04], 'grip': 0.5}], None]
        self.assertIs(world.retract_scissors(), False)
        self.assertEqual(world.call.call_args_list, [call('getRCMStats'), call('tissueStats')])
        world.steps.assert_not_called()

    def test_invalid_advance_fails_without_commands(self):
        for advance in (-0.001, np.nan, np.inf):
            world = self.make_world()
            world.advanced = advance
            self.assertIs(world.retract_scissors(), False)
            world.call.assert_not_called()
            world.steps.assert_not_called()


if __name__ == '__main__':
    unittest.main()