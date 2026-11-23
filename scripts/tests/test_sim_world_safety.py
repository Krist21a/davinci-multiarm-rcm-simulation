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
    import ground_truth
    import sim_world
    import simtools
    from sim_world import SimWorld, TISSUE_RGB
    from visibility import depth_visible
    from cutter_view import clear_stencil
    from workspace import H, NU, NV


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
            update.assert_called_once_with(sim, 10, tissue=True, tissue_source=None)
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


class CouponRollTests(unittest.TestCase):
    def test_sheet_roll_preserves_entrance_and_cut_axis_and_rotates_both_grasps(self):
        w = SimWorld.__new__(SimWorld)
        w.roi_proxy, w.canonical_seam = None, False
        w.cutter, w.mode, w.pretension, w.tissue_source = 'psm2', 'full', .08, None
        w.frame = Mock(return_value=dict(pivot=np.zeros(3), x=np.array([1., 0., 0.]),
                                         axis=np.array([0., 1., 0.]), tcp=np.array([0., 0., .1])))
        w.steps = Mock()
        w.call = Mock(side_effect=[None, [.00092] * NV, None, [.00092] * NV])
        w.sheet_cfg = {}
        w.place_sheet()
        base = w.call.call_args_list[0].args[1]
        w.sheet_cfg = {'bridge_print': True, 'bridge_fracture_faces': True, 'roll_deg': 16}
        w.place_sheet()
        tilted = w.call.call_args_list[2].args[1]
        angle = np.deg2rad(16.)
        across = np.array([0., np.cos(angle), np.sin(angle)])
        np.testing.assert_array_equal(tilted['origin'], base['origin'])
        np.testing.assert_array_equal(tilted['v'], base['v'])
        self.assertEqual(tilted['spacing'], base['spacing'])
        np.testing.assert_allclose(tilted['u'], across, atol=1e-14)
        mid = w.sheet_origin + .5 * (NV - 1) * H * w.v
        for arm, sign in [('psm1', -1), ('psm3', 1)]:
            np.testing.assert_allclose(w.grasp_points[arm], mid + sign * .5 * (NU - 1) * H * across)

    def test_sheet_roll_refuses_invalid_mounts(self):
        valid = dict(bridge_print=True, bridge_fracture_faces=True)
        for value in (-16, -8., 0, 8, 16):
            self.assertEqual(SimWorld._check_sheet_cfg(dict(valid, roll_deg=value))['roll_deg'], value)
        for value in (True, None, '8', float('inf'), float('nan'), -17, 17):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'roll_deg'):
                SimWorld._check_sheet_cfg(dict(valid, roll_deg=value))
        with self.assertRaisesRegex(ValueError, 'roll_deg'):
            SimWorld._check_sheet_cfg({'roll_deg': 8})


class VisibilityTests(unittest.TestCase):
    def test_transparent_tcp_marker_keeps_collision_and_detection_properties(self):
        w = SimWorld.__new__(SimWorld)
        w.cutter, w.scene_file = 'psm2', Path('scene.ttt')
        w.sim = Mock(object_shape_type=1, objintparam_visibility_layer=10, objectspecialproperty_renderable=32)
        w.by = {'psm2_tool_tip_visual': 7, 'psm2_yaw_ref2_respondable_visual': 8, 'psm2_tcp': 9}
        w.sim.getObjectType.return_value = 1
        w.sim.getObjectSpecialProperty.side_effect = [47, 15]
        w.sim.getObjectInt32Param.side_effect = [1, 0]
        with patch.object(sim_world, 'file_sha256', return_value='other'):
            with self.assertRaises(ValueError):
                w._exclude_tcp_marker()
        w.sim.setObjectSpecialProperty.assert_not_called()
        with patch.object(sim_world, 'file_sha256', return_value='8bff1b9e5af8c98a969d8ba6d6c2ea722c3e8070540439edeb84e27eda81c8d7'):
            self.assertEqual(w._exclude_tcp_marker()['properties_after'], 15)
        w.sim.setObjectSpecialProperty.assert_called_once_with(7, 15)
        w.sim.setObjectInt32Param.assert_called_once_with(7, 10, 0)
        w.sim.removeObjects.assert_not_called()

    def test_capsules_use_bounded_sightlines_not_infinite_rays(self):
        points = np.array([[0., 0., 1.], [.03, 0., 1.]])
        camera = np.zeros(3)
        occluder = (np.array([-.002, 0., .5]), np.array([.002, 0., .5]), .001)
        np.testing.assert_array_equal(clear_stencil(camera, points, [occluder]), [False, True])
        behind = (occluder[0] + points[0], occluder[1] + points[0], occluder[2])
        np.testing.assert_array_equal(clear_stencil(camera, points, [behind]), [True, True])
        R = sim_world.cut_perception.rotvec(np.array([.5, -.3, .1]))
        t = np.array([-.1, .2, .4])
        transformed = (R @ occluder[0] + t, R @ occluder[1] + t, occluder[2])
        np.testing.assert_array_equal(clear_stencil(t, points @ R.T + t, [transformed]), [False, True])

    def setUp(self):
        self.world = SimWorld.__new__(SimWorld)
        self.world.truth = sim_world.GroundTruth(self.world)   # localize() tags its reads
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
        self.world.cut_front = Mock(return_value=self.point)
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
    world.index = {'psm1': 1, 'psm2': 2, 'psm3': 3, 'ecm': 4}
    world.primary, world.assist, world.cutter, world.view = 'psm1', 'psm3', 'psm2', 'ecm'
    world.call = Mock()
    world.steps = Mock()
    world.truth = sim_world.GroundTruth(world)
    world.refused_failures = {}
    world.ever_grasped = set()
    world.events = {}
    world.event_log = []
    world.advanced = 0.0
    world.align_mode = 'shaft'
    world._pre_cut_q = None
    # No homed anchor until a test installs an explicit ready state: a missing anchor
    # must refuse, never pretend a successful park.
    world._ready_q = None
    world._ready_tcp = None
    world.tick = 0
    world.occluder = None
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
        # The solver-failure snapshot reads getRCMStats before the target command.
        self.assertEqual(world.call.call_args_list[0], call('getRCMStats'))
        self.assertEqual(world.call.call_args_list[-1], call('setRCMTarget', 2, 1.0, 2.0, 0.03))
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
            if name == 'getRCMStats':
                return [{'arm': a, 'failures': 0} for a in world.index]
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

    def test_approach_must_land_on_the_point_before_closing(self):
        # Every command converged, yet the tip ended 5 mm off: closing would pin a neighbouring
        # patch and still report success by node count.
        world = self.make_world()
        world.sim.getObjectPosition = Mock(return_value=(0.005, 0.0, 0.0))
        ok, why = world.grasp('psm1', np.zeros(3))
        self.assertIs(ok, False)
        self.assertIn('mm from the grasp point', why)
        self.assertEqual(world.pinned, 0)
        self.assertNotIn(call('setGrip', 1, 0.0), world.call.call_args_list)
        self.assertNotIn(call('tissueGrasp', 1), world.call.call_args_list)

    def test_grasp_records_the_held_point_and_release_clears_it(self):
        world = self.make_world()
        point = np.array([-0.012, 0.0, 0.05])
        world.sim.getObjectPosition = Mock(return_value=point.tolist())   # tip lands on the point
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
        # Real homed-TCP readback: home() must never fabricate the ready anchor.
        world.by = {'psm2_tcp': 12}
        world.sim = SimpleNamespace(
            getObjectMatrix=Mock(return_value=np.eye(4)[:3].reshape(-1).tolist()))
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
        np.testing.assert_array_equal(world._ready_tcp, np.eye(4)[:3])
        self.assertFalse(world._ready_tcp.flags.writeable)
        self.assertIsNone(world._ready_q)      # shaft mode: the joint anchor is unused

    def test_home_without_valid_tcp_anchor_refuses(self):
        for matrix in (None, np.eye(4)[:3].reshape(-1).tolist()[:8], [np.nan] * 12):
            with self.subTest(matrix=matrix):
                world = self.make_home_world([True, True, True])
                world.sim.getObjectMatrix.return_value = matrix
                self.assertIs(world.home(), False)
                self.assertIsNone(world._ready_tcp)

    def test_tool_pose_home_freezes_measured_joint_anchor(self):
        world = self.make_home_world([True, True, True,
                                      {'available': True, 'roll': .1, 'pitch': .2, 'yaw': .3}])
        world.align_mode = 'tool_pose'
        world.target_of = Mock(return_value=(10.0, 20.0, 0.04))
        self.assertIs(world.home(), True)
        self.assertEqual(world._ready_q.shape, (6,))
        np.testing.assert_allclose(world._ready_q, [10.0, 20.0, 0.04, .1, .2, .3])
        self.assertFalse(world._ready_q.flags.writeable)
        # A missing wrist state leaves tool_pose home refused rather than anchorless.
        world2 = self.make_home_world([True, True, True, None])
        world2.align_mode = 'tool_pose'
        world2.target_of = Mock(return_value=(10.0, 20.0, 0.04))
        self.assertIs(world2.home(), False)
        self.assertIsNone(world2._ready_q)

    def test_explicit_view_is_applied_after_homing_before_sheet_placement(self):
        world = self.make_home_world([True, True, True])
        world.args = SimpleNamespace(initial_view=[1.5, -2.5])
        world.arm_cfg = {'ecm': {'insertion_home': 0.02}}
        world.target_of = Mock(return_value=(1.5, -2.5, 0.02))
        world.command = Mock(return_value=True)
        world.camera = Mock(side_effect=[(np.eye(3), np.zeros(3)), (np.eye(3), np.ones(3))])
        ordered = Mock()
        for name in ('call', 'command', 'place_sheet'):
            ordered.attach_mock(getattr(world, name), name)
        self.assertTrue(world.home())
        self.assertEqual(ordered.mock_calls[0], call.call('homeRCM'))
        self.assertEqual(ordered.mock_calls[-2:], [call.command('ecm', 1.5, -2.5, 0.02, 25),
                                                   call.place_sheet()])
        self.assertEqual(world.initial_view_readback['target'], [1.5, -2.5, 0.02])

    def test_initial_view_refuses_mismatched_insert_readback(self):
        world = self.make_home_world([True, True, True])
        world.args = SimpleNamespace(initial_view=[1.0, 2.0], initial_view_insertion_m=0.088)
        world.arm_cfg = {'ecm': {'insertion_home': 0.08}}
        world.target_of = Mock(return_value=(1.0, 2.0, 0.08))
        world.command = Mock(return_value=True)
        world.camera = Mock(return_value=(np.eye(3), np.zeros(3)))
        with self.assertRaisesRegex(RuntimeError, 'disagrees with ECM readback'):
            world.home()
        world.command.assert_called_once_with('ecm', 1.0, 2.0, 0.088, 25)
        world.place_sheet.assert_not_called()

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
        world.cfg = {'grasp': {'min_nodes': 3}}
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
        # The live tissue model reports a hold exactly where the world still has a record.
        world.call.side_effect = lambda name, *a: ({'grasped': {arm: 6 for arm in world.grasped_points}}
                                                   if name == 'tissueStats' else None)
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

    def test_retract_refuses_when_the_live_hold_is_lost(self):
        # After a physical slip the grasped_points record is stale: the tissue model holds nothing.
        world = self.make_world()
        world.grasped_points['psm1'] = np.array([-0.012, 0.0, 0.05])
        for grasped in ({}, {'psm1': 2}):                    # nothing held / below min_nodes
            with self.subTest(grasped=grasped):
                world.call.side_effect = lambda name, *a: {'grasped': grasped} if name == 'tissueStats' else None
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
        # The second getRCMStats is command()'s solver-failure snapshot before the target write.
        self.assertEqual(world.call.call_args_list,
                         [call('getRCMStats'), call('getRCMStats'), call('setRCMTarget', 2, 1.0, 2.0, 0.034)])
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
        self.assertEqual(world.call.call_args_list,
                         [call('getRCMStats'), call('tissueStats'), call('setGrip', 2, 0.0),
                          call('getRCMStats'), call('setRCMTarget', 2, 1.0, 2.0, 0.034)])

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


class RoleConfigurationTests(unittest.TestCase):
    """Arm names come from cfg['roles'], not from literals in sim_world.py."""
    def make_world(self):
        sim = Mock()
        config = [[{'arm': a} for a in ('psm1', 'psm2', 'psm3', 'ecm')], {},
                  {'camera': {'alias': 'camera', 'resolution': [41, 41], 'view_angle_deg': 60, 'far_m': 1.0}}]
        cfg = {'roles': {'retract_primary': 'psm3', 'cut': 'psm1', 'retract_assist': 'psm2', 'view': 'ecm'}}
        with patch.object(sim_world, 'read_json', side_effect=config), \
                patch.object(sim_world, 'load_scene'), \
                patch.object(sim_world, 'aliases', return_value={'RCMController': 10, 'camera': 11}), \
                patch.object(simtools, 'refresh_controller', return_value='sha'), \
                patch.object(SimWorld, 'start'):
            return SimWorld(None, sim, None, cfg)

    def test_scissors_commands_go_to_the_configured_cutter(self):
        world = self.make_world()
        self.assertEqual(world.cutter, 'psm1')
        world.steps = Mock()
        world.call = Mock(return_value=True)
        self.assertTrue(world.open_scissors())
        self.assertTrue(world.close_scissors())
        # psm1 is arm index 1 in the rcm order; a hard-coded psm2 would have sent index 2.
        self.assertEqual(world.call.call_args_list, [call('setGrip', 1, 0.5), call('setGrip', 1, 0.0)])

    def test_camera_motion_goes_to_the_configured_view_arm(self):
        world = self.make_world()
        self.assertEqual(world.view, 'ecm')
        world.steps = Mock()
        world.call = Mock(return_value=True)
        world.arm_cfg['ecm'] = dict(world.arm_cfg['psm1'], insertion_home=0.02)
        world.target_of = Mock(return_value=[0., 0., 0.02])
        self.assertTrue(world.adjust_camera())
        self.assertEqual(world.call.call_args_list[-1], call('setRCMTarget', 4, 3.0, 0.0, 0.02))

    def test_camera_skips_current_pose_and_only_counts_accepted_move(self):
        world = self.make_world()
        world.target_of = Mock(return_value=[3., 0., 0.088])
        world.command = Mock(side_effect=[False, True])
        self.assertFalse(world.adjust_camera())
        self.assertEqual(world.camera_moves, 0)
        self.assertTrue(world.adjust_camera())
        self.assertEqual(world.camera_moves, 1)
        self.assertEqual(world.command.call_args_list, [call('ecm', -3., 0., 0.088, 25),
                                                       call('ecm', 0., 3., 0.088, 25)])


class RefusedCandidateTests(unittest.TestCase):
    """A refused candidate increments the controller failure counter but rolled back: not a fault."""
    def make_world(self, on_set_target):
        world = bare_world()
        failures = {'psm2': 0}

        def controller(name, *args):
            if name == 'getRCMStats':
                return [{'arm': a, 'failures': failures.get(a, 0)} for a in world.index]
            if name == 'setRCMTarget':
                return on_set_target(failures)
            raise AssertionError(name)

        world.call.side_effect = controller
        return world

    def test_refusal_increments_are_accounted_not_faults(self):
        def reject(failures):
            failures['psm2'] += 1          # the controller counts even a rolled-back candidate
            return False
        world = self.make_world(reject)
        self.assertIs(world.command('psm2', 1, 2, 0.03, 7), False)
        self.assertEqual(world.refused_failures['psm2'], 1)
        world.steps.assert_not_called()

    def test_collision_refusal_without_increment_records_zero(self):
        world = self.make_world(lambda failures: False)
        self.assertIs(world.command('psm2', 1, 2, 0.03, 7), False)
        self.assertEqual(world.refused_failures['psm2'], 0)
        world.steps.assert_not_called()

    def test_accepted_command_keeps_its_failure_increment_a_fault(self):
        def accept_but_count(failures):
            failures['psm2'] += 1
            return True
        world = self.make_world(accept_but_count)
        self.assertIs(world.command('psm2', 1, 2, 0.03, 7), True)
        self.assertNotIn('psm2', world.refused_failures)


class WithdrawalAccountingTests(unittest.TestCase):
    def make_world(self, motion_ok=True):
        world = bare_world()
        world.frame = Mock(return_value={'tcp': np.array([0.0, 0.0, 0.01]), 'pivot': np.zeros(3)})
        world.cut_front = Mock(return_value=np.array([0.0, 0.0, 0.02]))
        world.v = np.array([0.0, 0.0, 1.0])
        world.target_of = Mock(return_value=(0.0, 0.0, 0.04))
        state = {'ins': 0.04}

        def controller(name, *args):
            if name == 'getRCMStats':
                return [{}, {'arm': 'psm2', 'failures': 0, 'target': [0.0, 0.0, state['ins']], 'grip': 0.0}]
            if name == 'setRCMTarget':
                state['ins'] = args[3]
                return motion_ok
            raise AssertionError(name)

        world.call.side_effect = controller
        return world

    def test_accepted_alignment_correction_is_withdrawn(self):
        world = self.make_world()
        self.assertEqual(world.align_scissors(None, 0), 0.0)
        self.assertAlmostEqual(world.advanced, 0.009)       # 0.01 approach - 0.001 standoff
        self.assertTrue(world.advance_scissors(0.003))
        self.assertAlmostEqual(world.advanced, 0.012)
        self.assertTrue(world.retract_scissors())
        last = world.call.call_args_list[-1][0]
        self.assertEqual(last[:2], ('setRCMTarget', 2))
        self.assertAlmostEqual(last[4], 0.031)              # 0.043 commanded - (0.009 + 0.003)

    def test_refused_alignment_correction_is_not_counted(self):
        world = self.make_world(motion_ok=False)
        self.assertIsNone(world.align_scissors(None, 0))
        self.assertEqual(world.advanced, 0.0)


class EventInjectionTests(unittest.TestCase):
    def test_slip_releases_tissue_but_keeps_the_stale_record(self):
        world = bare_world()
        world.tick = 5
        world.occluder = None
        world.events = {'slip': {'psm1': 5}}
        world.grasped_points = {'psm1': np.array([-0.012, 0.0, 0.05])}
        world._events()
        world.call.assert_called_once_with('tissueRelease', 1)
        self.assertIn('psm1', world.grasped_points)         # stale on purpose; a live check must catch it

    def test_relax_commands_each_holding_arm_inward(self):
        world = bare_world()
        world.tick = 7
        world.occluder = None
        world.events = {'relax': 7}
        world.cfg = {'events': {'relax_m': 0.0015}}
        world.grasped_points = {'psm1': np.zeros(3), 'psm3': np.zeros(3)}
        world.retract = Mock(side_effect=[True, False])
        world._events()
        self.assertEqual(world.retract.call_args_list, [call('psm1', -0.0015), call('psm3', -0.0015)])
        relaxes = [e for e in world.event_log if e['event'] == 'relax']
        self.assertEqual([e['accepted'] for e in relaxes], [True, False])

    def test_regrasp_failure_is_injected_before_any_motion(self):
        world = bare_world()
        world.cfg = {'grasp': {'min_nodes': 3}}
        world.arm_cfg = {'psm1': {'limits': {'yaw_deg': 30, 'pitch_deg': 30,
                                             'insertion_min': 0.0, 'insertion_max': 0.1}}}
        world.rcm_command_for = Mock(return_value=(0, 0, 0.04))
        world.grasped_points = {}
        world.ever_grasped = {'psm1'}
        world.events = {'regrasp_fails': {'psm1'}}
        ok, why = world.grasp('psm1', np.zeros(3))
        self.assertIs(ok, False)
        self.assertIn('regrasp', why)
        world.call.assert_not_called()


class TakeoverPointTests(unittest.TestCase):
    """takeover_point() hands the lost side to a node the taking-over arm can actually reach."""
    def make_world(self, reachable):
        """reachable: set of (row, col) the arm can command inside its limits; others get a huge insertion."""
        world = bare_world()
        world.sheet_origin = np.zeros(3)
        world.u = np.array([1.0, 0.0, 0.0])               # +u = the "psm3 side" by layout convention
        world.arm_cfg = {'psm3': {'limits': {'yaw_deg': 5.0, 'pitch_deg': 5.0,
                                              'insertion_min': 0.0, 'insertion_max': 0.055}}}
        nodes = np.zeros((NV, NU, 3))
        for j in range(NV):
            for i in range(NU):
                nodes[j, i] = [(i - (NU - 1) / 2) * H, 0.0, j * H]
        world.nodes = Mock(return_value=nodes)

        def command_for(arm, point):
            p = np.asarray(point, dtype=float)
            j = int(round(p[2] / H))
            i = int(round(p[0] / H + (NU - 1) / 2))
            return (0.0, 0.0, 0.04) if (j, i) in reachable else (0.0, 0.0, 0.2)

        world.rcm_command_for = Mock(side_effect=command_for)
        return world, nodes

    def test_nearest_reachable_node_on_the_lost_side_is_taken(self):
        world, nodes = self.make_world(reachable={(7, 2), (7, 4), (9, 1)})
        got = world.takeover_point('psm3', nodes[7, 1].copy())
        np.testing.assert_allclose(got, nodes[7, 2])    # 1 mm away beats n[7,4] (3 mm) and n[9,1] (2 mm)

    def test_nodes_on_the_far_side_are_never_offered(self):
        world, nodes = self.make_world(reachable={(7, 20)})
        self.assertIsNone(world.takeover_point('psm3', nodes[7, 1].copy()))

    def test_column_band_keeps_the_jaws_off_the_cut_and_edge(self):
        # cols 9-11 sit within three columns of the cut line; row 0 and col 0 are off-limits.
        world, nodes = self.make_world(reachable={(7, 9), (7, 11), (0, 3), (7, 0)})
        self.assertIsNone(world.takeover_point('psm3', nodes[7, 1].copy()))

    def test_limit_margin_excludes_boundary_points(self):
        world, nodes = self.make_world(reachable=set())
        # Insertion at the raw max still fails: the candidate must sit 1 mm inside it.
        world.rcm_command_for = Mock(return_value=(0.0, 0.0, 0.0545))
        self.assertIsNone(world.takeover_point('psm3', nodes[7, 1].copy()))
        # Same for angles: 4.6 deg is inside the raw 5 deg but outside the 0.5 deg margin.
        world.rcm_command_for = Mock(return_value=(4.6, 0.0, 0.04))
        self.assertIsNone(world.takeover_point('psm3', nodes[7, 1].copy()))

    def test_no_sheet_or_no_side_means_no_point(self):
        world = bare_world()
        world.sheet_origin = None
        self.assertIsNone(world.takeover_point('psm3', np.zeros(3)))
        world.sheet_origin = np.zeros(3)
        world.u = np.array([1.0, 0.0, 0.0])
        self.assertIsNone(world.takeover_point('psm3', None))
        self.assertIsNone(world.takeover_point('psm3', np.zeros(3)))     # lost point on the cut line


class GroundTruthPurposeTests(unittest.TestCase):
    """reads_by_purpose accounting and the strict untagged-read gate (m5b_protocol.json)."""
    def make_world(self):
        world = bare_world()
        world.sheet_origin = np.zeros(3)
        world.pretension = 0.08
        world.call = Mock(return_value={'grasped': {}})
        return world

    def test_untagged_strict_read_raises(self):
        world = self.make_world()
        world.truth.strict = True
        with self.assertRaises(ground_truth.TruthReadInControl):
            world.truth.tissue_stats()

    def test_tagged_reads_count_per_purpose_and_totals_unchanged(self):
        world = self.make_world()
        world.truth.strict = True
        with world.truth.purpose('grasp_signal_and_safety_monitor'):
            world.truth.tissue_stats()
        with world.truth.purpose('safety_monitor'):
            world.truth.tissue_stats()
        self.assertEqual(world.truth.reads['tissue_stats'], 2)
        self.assertEqual(world.truth.reads_by_purpose,
                         {'grasp_signal_and_safety_monitor': {'tissue_stats': 1},
                          'safety_monitor': {'tissue_stats': 1}})

    def test_nested_purposes_use_the_innermost_tag(self):
        world = self.make_world()
        with world.truth.purpose('outer'):
            world.truth.tissue_stats()
            with world.truth.purpose('inner'):
                world.truth.tissue_stats()
            world.truth.tissue_stats()
        self.assertEqual(world.truth.reads_by_purpose,
                         {'outer': {'tissue_stats': 2}, 'inner': {'tissue_stats': 1}})

    def test_non_strict_untagged_read_lands_in_untagged_without_raising(self):
        world = self.make_world()
        world.truth.tissue_stats()
        self.assertEqual(world.truth.reads['tissue_stats'], 1)
        self.assertEqual(world.truth.reads_by_purpose['untagged']['tissue_stats'], 1)


class VisionModeTests(unittest.TestCase):
    def test_view_selection_skips_occluded_candidate_before_actuation(self):
        world = self.vision_world()
        longer, shorter = np.ones(6), np.zeros(6)
        world._tool_solve = Mock(side_effect=[dict(converged=True, q=longer), dict(converged=True, q=shorter)])
        world._tool_fk = Mock(return_value=lambda q: np.eye(3, 4) * (1 + q[0]))
        world._candidate_view_counts = Mock(side_effect=[(23, 1), (21, 3)])
        world.tool_command = Mock()
        targets = [(0.003, np.array([0., 0., .043])), (0.0015, np.array([0., 0., .0415]))]
        result = world._visible_tool_candidate('advance', np.eye(3, 4), targets, np.eye(3)[2], np.eye(3)[0])
        np.testing.assert_array_equal(result['q'], shorter)
        world.tool_command.assert_not_called()
        self.assertFalse(world.view_blocked)
        self.assertEqual(world.event_log[-1]['selected_m'], .0015)

    """M5b cutter-arm vision control on SimWorld.__new__ stubs (m5b_protocol.json)."""

    EST = dict(scissors={'valid': True, 'tip': [0.0, 0.0, 0.10], 'jaw_axis': [0.0, 0.0, 1.0]},
               tissue={'valid': True, 'normal': [0.0, 1.0, 0.0]},
               cut={'valid': True, 'front_valid': True,
                    'front': [0.0, 0.0, 0.05], 'direction': [0.0, 0.0, 1.0], 'start': [0.0, 0.0, 0.04],
                    'material_start': [0., 0., 0.04], 'progress_m': 0.01, 'material_scale': 1.0})

    def vision_world(self, est=None, strict=True):
        world = bare_world()
        world.vision = True
        world.truth.strict = strict
        world.perc_mode = 'off'
        world.tick = 0
        world.sheet_origin = np.zeros(3)
        world.grasped_points = {}
        world.u = np.array([1.0, 0.0, 0.0])
        world.v = np.array([0.0, 0.0, 1.0])
        world.camera_moves = len(SimWorld.CAMERA_POSES)
        world.pretension = 0.08
        # Measured per-row rests of this stub's nominal H grid (current-controller world).
        world.cutline_rest_m = np.full(NV, H * (1 - world.pretension))
        world.W = world.H = 41
        world.f = 20.0
        world.far = 1.0
        world.camera = Mock(return_value=(np.eye(3), np.zeros(3)))
        world.render = Mock()
        world._protected_tick = Mock()
        world._estimate = Mock(return_value=est)
        world._score_shadow = Mock()
        nodes = []
        for j in range(NV):
            for i in range(NU):
                nodes += [i * H, 0.0, j * H]
        truth_payload = {'tissueStats': {'grasped': {'psm1': 4}, 'torn': 0, 'max_strain': 0.05},
                         'tissueConstraints': [],
                         'tissueNodes': nodes}
        stats = [{'arm': a, 'failures': 0, 'max_drift_m': 0.0} for a in world.index]

        def controller(name, *a):
            if name in truth_payload:
                return truth_payload[name]
            if name == 'getRCMStats':
                return stats
            return True                                      # actuation stubs (setGrip, commands)

        world.call = Mock(side_effect=controller)
        return world

    def test_observe_maps_the_estimate_and_makes_no_untagged_read(self):
        world = self.vision_world(self.EST)
        obs = world.observe()
        self.assertTrue(obs.cut_site_visible)
        self.assertTrue(obs.scissors_visible)
        self.assertTrue(obs.tissue_visible)
        self.assertAlmostEqual(obs.cut_progress_m, 0.01)          # (front - start).direction
        self.assertEqual(obs.grasped, {'psm1': 4})
        self.assertAlmostEqual(obs.tension, 1.0 / 0.92 - 1.0, places=5)
        self.assertEqual(obs.cut_length_m, 0.0)
        self.assertNotIn('untagged', world.truth.reads_by_purpose)
        self.assertEqual(set(world.truth.reads_by_purpose),
                         {'grasp_signal_and_safety_monitor', 'tension_until_m6', 'safety_monitor'})
        self.assertIsNotNone(world._fix)
        self.assertEqual(world._fix['tick'], world.tick)

    def test_observe_without_a_fix_reports_the_front_hidden(self):
        est = dict(scissors={'valid': True}, tissue={'valid': True},
                   cut={'valid': True, 'front_valid': False})
        world = self.vision_world(est)
        obs = world.observe()
        self.assertFalse(obs.cut_site_visible)
        self.assertTrue(obs.scissors_visible)
        self.assertIsNone(obs.cut_progress_m)

    def test_fix_age_and_commanded_invalidation(self):
        world = self.vision_world()
        world.tick = 10
        fix = dict(front=np.zeros(3), direction=np.array([0.0, 0.0, 1.0]),
                   normal=np.array([0.0, 1.0, 0.0]), progress=0.0, tick=10, epoch=0)
        world._fix = dict(fix, tick=3)
        self.assertFalse(world._fix_valid())                      # 7 > fix_max_age 6
        world._fix = dict(fix, tick=4)
        self.assertTrue(world._fix_valid())                       # exactly at the boundary

        world._fix = dict(fix)
        self.assertTrue(world.close_scissors())
        self.assertIsNone(world._fix)                             # closure voids the fix

        world.camera_moves = 0
        world.target_of = Mock(return_value=[0., 0., 0.08])
        for method, args in ((world.retract, ('psm1', 0.0002)),
                             (world.release, ('psm1',)),
                             (world.takeover_point, ('psm3', np.zeros(3))),
                             (world.adjust_camera, ())):
            world._fix = dict(fix)
            method(*args)
            self.assertIsNone(world._fix, method.__name__)

        world.cfg = {'grasp': {'min_nodes': 3}}
        world.arm_cfg = {'psm1': {'limits': {'yaw_deg': 30, 'pitch_deg': 30,
                                             'insertion_min': 0.0, 'insertion_max': 0.1}}}
        world.rcm_command_for = Mock(return_value=(0, 0, 0.04))
        world.target_of = Mock(return_value=(0, 0, 0))
        world.by = {'psm1_tcp': 7}
        world.sim = SimpleNamespace(getObjectPosition=Mock(return_value=(0, 0, 0)))
        world._fix = dict(fix)
        world.grasp('psm1', np.zeros(3))
        self.assertIsNone(world._fix)

    def test_injected_events_do_not_invalidate_the_fix(self):
        world = self.vision_world()
        world.tick = 10
        world._fix = dict(front=np.zeros(3), direction=np.array([0.0, 0.0, 1.0]),
                          normal=np.array([0.0, 1.0, 0.0]), progress=0.0, tick=10, epoch=0)
        world._in_event = True                                    # inside _events()
        world.retract('psm1', 0.0002)
        self.assertIsNotNone(world._fix)
        world._in_event = False
        world.retract('psm1', 0.0002)
        self.assertIsNone(world._fix)

    def test_closure_demands_a_new_tissue_slit_and_material_coordinate(self):
        world = self.vision_world()
        self.assertTrue(world.close_scissors())
        for mode in ('memory', 'pre_cut'):
            world._update_fix(dict(self.EST, cut=dict(self.EST['cut'], mode=mode)))
            self.assertIsNone(world._fix)
        slit = dict(self.EST['cut'], mode='bank_slit')
        world._update_fix(dict(self.EST, cut=dict(slit, progress_m=None)))
        self.assertIsNone(world._fix)
        world._update_fix(dict(self.EST, cut=slit))
        self.assertFalse(world.awaiting_new_slit)
        self.assertEqual(world.event_log[-1]['event'], 'post_closure_reacquired')

    def test_coded_closure_needs_same_tick_chart_and_a_distal_reacquisition(self):
        world = self.vision_world()
        world.material_pattern = True
        chart = dict(valid=True, material_knots_m=sim_world.cut_perception.material_coordinate.KNOTS_M.tolist(),
                     projected_knots_m=[0, .002, .004, .007, .010],
                     guard_knots_m=[-.0002, .0018, .0038, .0068, .0098])
        cut = dict(self.EST['cut'], mode='slit', progress_m=.004, material_chart=chart)
        world._update_fix(dict(self.EST, cut=cut))
        self.assertTrue(world._fix_valid())
        world.tick += 1
        self.assertFalse(world._fix_valid())
        self.assertFalse(world.close_scissors())
        world._update_fix(dict(self.EST, cut=cut))
        self.assertTrue(world.close_scissors())
        world.tick += 1
        for change in (dict(progress_m=.0041), dict(mode='memory', progress_m=.005),
                       dict(progress_m=.005, material_chart={'valid': False})):
            world._update_fix(dict(self.EST, cut=dict(cut, **change)))
            self.assertFalse(world._fix_valid())
            self.assertTrue(world.awaiting_new_slit)
        world._update_fix(dict(self.EST, cut=dict(cut, progress_m=.005)))
        self.assertTrue(world._fix_valid())
        self.assertFalse(world.awaiting_new_slit)

    def test_partial_chart_reacquires_only_in_the_observed_local_cell(self):
        world = self.vision_world()
        world.material_pattern = world.local_material_chart = True
        world.awaiting_new_slit = True
        world._pre_closure_progress = .003
        knots = sim_world.cut_perception.material_coordinate.KNOTS_M[[0, 1, 2, 4]].tolist()
        chart = dict(valid=True, material_knots_m=knots,
                     projected_knots_m=[0., .002, .004, .010],
                     guard_knots_m=[-.0002, .0018, .0038, .0098])
        cut = dict(self.EST['cut'], mode='slit', progress_m=.0045, front=[0., 0., .043],
                   progress_cell_m=knots[1:3], material_chart=chart)
        for change in (dict(front=[0., 0., .048]), dict(progress_cell_m=knots[2:]),
                       dict(mode='memory')):
            world._update_fix(dict(self.EST, cut=dict(cut, **change)))
            self.assertFalse(world._fix_valid())
            self.assertTrue(world.awaiting_new_slit)
        world._update_fix(dict(self.EST, cut=cut))
        self.assertTrue(world._fix_valid())
        self.assertFalse(world.awaiting_new_slit)
        self.assertEqual(world.event_log[-1]['progress_cell_m'], knots[1:3])

    def test_sweep_can_reach_beyond_closed_tcp(self):
        result = sim_world.sweep_projection(np.zeros(3), np.array([0., 0., 1.]),
                                            np.array([1., -1., 0.]), np.array([1., 0., 0.]))
        self.assertGreater(result, 1.0)

    def align_world(self, tip, jaw=None):
        world = self.vision_world()
        world.align_mode = 'tool_pose'
        world.v = np.array([0.0, 0.0, 1.0])
        world.u = np.array([1.0, 0.0, 0.0])
        world.cfg = {'cut': {'align_tolerance_m': 0.0015, 'vision_axis_tolerance_deg': 5.0}}
        world.tick = 1
        world._fix = dict(front=np.asarray(self.EST['cut']['front'], dtype=float),
                          direction=np.array([0.0, 0.0, 1.0]), normal=np.array([0.0, 1.0, 0.0]),
                          progress=0.0, tick=1, epoch=0)
        world._last_est = dict(scissors={'valid': True, 'tip': list(tip),
                                         'jaw_axis': jaw or [0.0, 0.0, 1.0],
                                         'opening_axis': [1., 0., 0.]},
                               tissue={'valid': True})
        tcp0 = np.array([0.0, 0.0, 0.04])
        M = np.eye(3, 4)
        M[:, 3] = tcp0
        world.tcp1 = np.array([0.0, 0.0, 0.049])
        world.sim = SimpleNamespace(
            getObjectMatrix=Mock(return_value=M.reshape(-1).tolist()),
            getObjectPosition=Mock(return_value=world.tcp1.tolist()))
        world.by = {'psm2_tcp': 99}
        world._tool_q = Mock(return_value=(np.zeros(6), {'limits': {'roll': (-1, 1),
                                                                   'pitch': (-1, 1), 'yaw': (-1, 1)}}))
        world._tool_solve = Mock(return_value={'converged': True, 'q': np.ones(6), 'q0': np.zeros(6)})
        world.tool_command = Mock(return_value=True)
        world.target_of = Mock(return_value=(0.0, 0.0, 0.04))
        return world

    def test_align_vision_commands_the_relative_target(self):
        tip = np.array([0.002, 0.0, 0.049])                       # 2 mm lateral, 1 mm short
        world = self.align_world(tip)
        e = world._align_scissors_vision(0.0)
        self.assertAlmostEqual(e, 0.002)                          # max(err_lat 2 mm, err_along 0)
        target, direction, lat = world._tool_solve.call_args[0][1:4]
        front = np.asarray(self.EST['cut']['front'])
        np.testing.assert_allclose(target, np.array([0.0, 0.0, 0.04])
                                   + (front - 0.001 * np.array([0.0, 0.0, 1.0])) - tip)
        np.testing.assert_allclose(direction, [0, 0, 1])
        np.testing.assert_allclose(lat, [1, 0, 0])                # cross(normal,dir), signed to u
        self.assertTrue(world.tool_command.called)
        self.assertIsNotNone(world._pre_cut_q)
        ev = world.event_log[-1]
        self.assertEqual(ev['event'], 'vision_align')
        self.assertTrue(ev['moved'])
        self.assertEqual(ev['fix_age'], 0)

    def test_align_vision_inside_tolerance_does_not_move(self):
        front = np.asarray(self.EST['cut']['front'])
        tip = front - 0.001 * np.array([0.0, 0.0, 1.0])           # exactly on the target
        world = self.align_world(tip)
        self.assertAlmostEqual(world._align_scissors_vision(0.0), 0.0)
        world._tool_solve.assert_not_called()
        self.assertIsNone(world._pre_cut_q)                       # nothing moved yet

    def test_aligned_position_does_not_accept_misdirected_axes(self):
        tip = [0, 0, 0.049]
        for jaw in ([0.5, 0, np.sqrt(0.75)], [0, 0, -1]):
            with self.subTest(jaw=jaw):
                world = self.align_world(tip, jaw)
                self.assertIsNone(world._align_scissors_vision(0.0))
                self.assertTrue(world.tool_command.called)
                self.assertFalse(world.closure_aligned())
        world = self.align_world(tip)
        world._last_est['scissors']['opening_axis'] = [-1., 0., 0.]
        self.assertTrue(world.closure_aligned())                  # either blade-opening-axis sign
        self.assertAlmostEqual(world._align_scissors_vision(0.0), 0.0)
        world._last_est['scissors']['opening_axis'] = [0., 1., 0.]
        self.assertFalse(world.closure_aligned())

    def test_first_advance_after_no_motion_alignment_does_not_skip_withdrawal(self):
        world = self.align_world([0, 0, 0.049])
        self.assertAlmostEqual(world._align_scissors_vision(0), 0)
        self.assertIsNone(world._pre_cut_q)
        world.tool_command.return_value = False
        self.assertFalse(world.advance_scissors(0.003))
        self.assertIsNone(world._pre_cut_q)
        world.tool_command.return_value = True
        world.call = Mock(return_value=[{}, {'grip': 0.0}])
        self.assertTrue(world.advance_scissors(0.003))
        # Explicit ready anchor; the cutter is displaced until the accepted restore lands.
        world._ready_q = np.zeros(6)
        world._ready_tcp = np.eye(3, 4)
        world.at_ready_pose = Mock(side_effect=[False, True])
        self.assertTrue(world.retract_scissors())
        arm, q, steps = world.tool_command.call_args.args
        self.assertEqual((arm, steps), ('psm2', 20))
        np.testing.assert_array_equal(q, np.zeros(6))

    def test_closure_scoring_samples_before_grip_changes_truth(self):
        world = self.align_world([0, 0, 0.049])
        world.cut_front = Mock(return_value=np.array([0.0005, 0, 0.05]))

        def set_grip(*args):
            world.cut_front.return_value = np.array([0.003, 0, 0.05])
            return True

        world.call = Mock(side_effect=set_grip)
        self.assertTrue(world.close_scissors())
        self.assertAlmostEqual(world.event_log[-1]['lateral_truth_m'], 0.0005)
        world.call = Mock(return_value=False)
        self.assertFalse(world.close_scissors())
        self.assertEqual(len(world.event_log), 1)

    def test_align_vision_refuses_without_fresh_tip_or_agreeing_direction(self):
        world = self.align_world([0, 0, 0.049])
        world._last_est = dict(scissors={'valid': False}, tissue={'valid': True})
        self.assertIsNone(world._align_scissors_vision(0.0))
        world._last_est = dict(scissors={'valid': True, 'tip': [0, 0, 0.049],
                                         'jaw_axis': [0, 0, 1]}, tissue={'valid': True})
        world._fix['direction'] = np.array([0.0, 0.0, -1.0])      # disagrees with the sheet v
        self.assertIsNone(world._align_scissors_vision(0.0))
        world._fix['direction'] = np.array([0.0, 0.0, 1.0])
        world._fix['tick'] = -10                                  # age 11 > fix_max_age 6: stale
        self.assertIsNone(world._align_scissors_vision(0.0))

    def test_retract_scissors_vision_parks_with_open_jaws_without_closing(self):
        world = self.vision_world()
        world.align_mode = 'tool_pose'
        world._pre_cut_q = np.zeros(6)
        world._ready_q = np.zeros(6)
        world._ready_tcp = np.eye(3, 4)
        world.advanced = 0.003
        world.call = Mock(return_value=[{}, {'grip': 0.5}])
        world.tool_command = Mock(side_effect=[False, True])
        world.at_ready_pose = Mock(side_effect=[False, False, True])
        self.assertFalse(world.retract_scissors())
        self.assertIsNotNone(world._pre_cut_q)
        self.assertTrue(world.retract_scissors())
        self.assertIsNone(world._pre_cut_q)
        world.call.assert_not_called()                            # no grip read, no setGrip
        self.assertEqual(world.truth.reads, {})                   # strict: any read would raise

    def test_terminal_observation_does_not_inject_events_or_credit_a_new_fix(self):
        world = self.vision_world(self.EST)
        world._events = Mock()
        world.awaiting_new_slit = True
        world._estimate.return_value = dict(self.EST, cut=dict(self.EST['cut'], mode='slit'))
        world.observe(inject_events=False)
        world._events.assert_not_called()
        self.assertTrue(world.awaiting_new_slit)
        self.assertEqual(world.event_log, [])

    def test_endpoint_travel_includes_actual_tip_gap_and_material_scale(self):
        world = self.align_world([0., 0., 0.049])
        world._fix['material_scale'] = 1.25
        self.assertAlmostEqual(world.endpoint_travel(0.0001), 0.00108)
        world._last_est['scissors']['tip'] = [0., 0., 0.051]
        self.assertLess(world.endpoint_travel(0.0001), 0.0)

    def test_sweep_bound_is_invariant_under_shared_optical_rotation(self):
        world = self.align_world([0.001, 0.0, 0.049])
        world._fix.update(start=np.array([0., 0., 0.04]), material_scale=1.05)
        world.cfg['cut'].update(length_m=0.009, max_overcut_m=0.001)
        world.by.update(j1=1, j2=2)
        world.arm_cfg = {'psm2': {'grip_joints': ['j1', 'j2']}}
        hinge = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        tcp = np.array([0., 0., 0.04])
        pivot = np.array([-0.002, 0.001, 0.033])
        poses = {99: np.column_stack((np.eye(3), tcp)), 1: np.column_stack((hinge, pivot)),
                 2: np.column_stack((hinge @ np.diag([-1., 1., -1.]), pivot))}
        world.sim.getObjectMatrix.side_effect = lambda h, rel: poses[h].ravel().tolist()
        bound = world.endpoint_advance_budget()
        self.assertIsNotNone(bound)
        rotation = sim_world.cut_perception.rotvec([0.25, 0.4, 0.1])
        for key in ('start', 'direction'):
            world._fix[key] = rotation @ world._fix[key]
        for key in ('tip', 'opening_axis', 'jaw_axis'):
            world._last_est['scissors'][key] = rotation @ world._last_est['scissors'][key]
        self.assertAlmostEqual(world.endpoint_advance_budget(), bound)
        physical = np.array([0., .002, .005, .008, .012])
        world._fix['material_chart'] = (
            sim_world.cut_perception.material_coordinate.KNOTS_M, physical, physical - .0003)
        self.assertAlmostEqual(world.endpoint_travel(.001), -.001)
        # 10 mm material -> 9 mm projected -> 8.7 mm on the earlier bank.
        self.assertAlmostEqual(world.endpoint_advance_budget() - bound, .0087 - .010 / 1.05)
        # Without row 9, the sweep may not infer its way into the 6-to-13 mm gap.
        k = [0, 1, 2, 4]
        world._fix['material_chart'] = (
            sim_world.cut_perception.material_coordinate.KNOTS_M[k], physical[k], physical[k] - .0003)
        self.assertAlmostEqual(world.endpoint_travel(.001), -.004)
        self.assertAlmostEqual(world.endpoint_advance_budget() - bound, .0047 - .010 / 1.05)

    def test_clear_view_never_returns_forward_to_park(self):
        world = self.align_world([0., 0., 0.049])
        world._pre_cut_q = np.zeros(6)
        world._tool_fk = Mock(return_value=lambda _: np.column_stack((np.eye(3), [0., 0., 0.043])))
        world.retract_scissors = Mock()
        world._move_scissors = Mock(return_value=True)
        self.assertTrue(world.clear_view(0.003, prefer_park=True))
        world.retract_scissors.assert_not_called()
        world._move_scissors.assert_called_once_with(-0.003, 15, record_park=False)

    def test_initial_backoff_does_not_set_a_forward_park_pose(self):
        world = self.align_world([0., 0., 0.049])
        # Ready anchor at the stub's homed TCP: the back-off never moved the pose,
        # so withdrawal is already parked and issues no second motion.
        world._ready_q = np.zeros(6)
        world._ready_tcp = np.eye(3, 4)
        world._ready_tcp[:, 3] = [0., 0., 0.04]
        self.assertTrue(world.clear_view(0.003))
        self.assertIsNone(world._pre_cut_q)
        self.assertTrue(world.retract_scissors())
        self.assertEqual(world.tool_command.call_count, 1)

    def test_shaft_tip_past_front_cannot_align_on_lateral_error_alone(self):
        world = self.align_world([0., 0., 0.051])
        world.align_mode = 'shaft'
        world.by['psm2_joint8'] = 8
        world.command = Mock(return_value=True)
        self.assertGreater(world._align_scissors_vision(0.0), world.cfg['cut']['align_tolerance_m'])
        world.command.assert_called_once()
        arm, yaw, pitch, insertion, steps = world.command.call_args.args
        self.assertEqual((arm, yaw, pitch, steps), ('psm2', 0., 0., 10))
        self.assertAlmostEqual(insertion, 0.038)

    def test_tool_fk_rejects_pose_from_rolled_back_shaft(self):
        world = self.align_world([0., 0., 0.049])
        world.call = Mock(return_value=dict(ok=True, shaft_ok=False, tcp=np.eye(3, 4).ravel()))
        with self.assertRaisesRegex(RuntimeError, 'shaft pose unavailable'):
            world._tool_fk(world.cutter)(np.zeros(6))

    def test_retract_scissors_vision_withdraws_when_closed(self):
        world = self.vision_world()
        world.align_mode = 'tool_pose'
        world._pre_cut_q = np.zeros(6)
        world.advanced = 0.003
        # Explicit homed anchor; real at_ready_pose readback sees the cutter displaced,
        # then landed at the ready TCP after the accepted restore motion.
        world._ready_q = np.zeros(6)
        world._ready_tcp = np.eye(3, 4)
        world._ready_tcp[:, 3] = [0., 0., 0.04]
        world.by = {'psm2_tcp': 99}
        away = world._ready_tcp.copy()
        away[:, 3] += [0., 0., 0.003]
        world.sim = SimpleNamespace(getObjectMatrix=Mock(
            side_effect=[away.reshape(-1).tolist(), world._ready_tcp.reshape(-1).tolist()]))
        world.call = Mock(side_effect=lambda name, *a: [{'arm': 'psm1'},
                                                        {'target': [1.0, 2.0, 0.04], 'grip': 0.0}]
                          if name == 'getRCMStats' else True)
        world.tool_command = Mock(return_value=True)
        self.assertTrue(world.retract_scissors())
        args = world.tool_command.call_args[0]
        self.assertEqual(args[0], 'psm2')
        np.testing.assert_array_equal(args[1], np.zeros(6))       # the immutable homed pose
        self.assertEqual(args[2], 20)
        self.assertEqual(world.advanced, 0.0)


if __name__ == '__main__':
    unittest.main()