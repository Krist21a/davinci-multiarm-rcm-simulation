"""Focused unit tests for the core safety repairs.

Covers the SimWorld ready-pose anchor and physical jaw readback, tool_pose
retraction, FSM endpoint guards (no setGrip merely to park), perception flag
gating/provenance, and the frozen-legacy tissue selection. No simulator.

Run: python -m unittest discover -s scripts/tests -p test_core_repairs.py -v
"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'scripts' / 'task'
for p in (str(TASK), str(ROOT / 'scripts' / 'build'), str(ROOT / 'scripts' / 'tests')):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_cut_task  # noqa: E402
import sim_world  # noqa: E402
import simtools  # noqa: E402
from sim_world import LEGACY_TISSUE, SimWorld  # noqa: E402

from test_sim_world_safety import bare_world  # noqa: E402


def tcp_at(xyz, rot=None):
    m = np.column_stack((np.eye(3) if rot is None else np.asarray(rot, float), np.asarray(xyz, float)))
    return m


class ReadyPoseTests(unittest.TestCase):
    """at_ready_pose: real TCP readback against the immutable homed anchor."""

    def world(self, ready=None, readback=None):
        w = bare_world()
        w.by = {'psm2_tcp': 99}
        w._ready_q = np.zeros(6)
        w._ready_tcp = ready if ready is not None else tcp_at([0., 0., 0.04])
        w.sim = SimpleNamespace(getObjectMatrix=Mock(return_value=readback))
        return w

    def test_matching_pose_is_ready(self):
        ready = tcp_at([0., 0., 0.04])
        w = self.world(ready, ready.reshape(-1).tolist())
        self.assertTrue(w.at_ready_pose())

    def test_translation_beyond_half_mm_is_not_ready(self):
        away = tcp_at([0., 0., 0.0406])
        w = self.world(tcp_at([0., 0., 0.04]), away.reshape(-1).tolist())
        self.assertFalse(w.at_ready_pose())

    def test_rotation_beyond_one_degree_is_not_ready(self):
        a = np.radians(2.0)
        rot = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
        w = self.world(tcp_at([0., 0., 0.04]),
                       tcp_at([0., 0., 0.04], rot).reshape(-1).tolist())
        self.assertFalse(w.at_ready_pose())

    def test_missing_anchor_is_never_ready(self):
        w = self.world()
        w._ready_tcp = None
        self.assertFalse(w.at_ready_pose())

    def test_nonfinite_readback_is_not_ready(self):
        bad = np.full((3, 4), np.nan).reshape(-1).tolist()
        self.assertFalse(self.world(readback=bad).at_ready_pose())

    def test_readback_failure_is_not_ready(self):
        w = self.world()
        w.sim.getObjectMatrix.side_effect = RuntimeError('sim gone')
        self.assertFalse(w.at_ready_pose())


class JawsClosedTests(unittest.TestCase):
    """jaws_closed: physical joint readback, both grip joints within 1e-6."""

    def world(self, positions):
        w = bare_world()
        w.by = {'psm2_j1': 1, 'psm2_j2': 2}
        w.arm_cfg = {'psm2': {'grip_joints': ['psm2_j1', 'psm2_j2']}}
        w.sim = SimpleNamespace(getJointPosition=Mock(side_effect=positions))
        return w

    def test_both_zero_is_closed(self):
        self.assertTrue(self.world([0.0, 0.0]).jaws_closed())

    def test_one_open_joint_is_not_closed(self):
        self.assertFalse(self.world([0.0, 1e-5]).jaws_closed())

    def test_nonfinite_joint_is_not_closed(self):
        self.assertFalse(self.world([0.0, float('nan')]).jaws_closed())

    def test_missing_handles_or_joints_refuse(self):
        w = self.world([0.0, 0.0])
        w.by = {}
        self.assertFalse(w.jaws_closed())
        w.by = {'psm2_j1': 1, 'psm2_j2': 2}
        w.arm_cfg = {'psm2': {'grip_joints': ['psm2_j1']}}
        self.assertFalse(w.jaws_closed())

    def test_readback_exception_is_not_closed(self):
        w = self.world([0.0, 0.0])
        w.sim.getJointPosition.side_effect = RuntimeError('sim gone')
        self.assertFalse(w.jaws_closed())


class ToolPoseRetractionTests(unittest.TestCase):
    """retract_scissors in tool_pose restores the immutable homed pose, jaws preserved."""

    def world(self, vision=False):
        w = bare_world()
        w.vision = vision
        w.align_mode = 'tool_pose'
        w.by = {'psm2_tcp': 99}
        w._ready_q = np.zeros(6)
        w._ready_tcp = tcp_at([0., 0., 0.04])
        w._pre_cut_q = np.full(6, 0.1)
        w.advanced = 0.003
        w.tool_command = Mock(return_value=True)
        w._invalidate_fix = Mock()
        w.sim = SimpleNamespace(getObjectMatrix=Mock())
        w.call = Mock()
        return w

    def test_missing_anchor_refuses(self):
        for anchor in ('_ready_q', '_ready_tcp'):
            with self.subTest(anchor=anchor):
                w = self.world()
                setattr(w, anchor, None)
                self.assertFalse(w.retract_scissors())
                w.tool_command.assert_not_called()

    def test_already_at_ready_needs_no_motion(self):
        w = self.world()
        w.sim.getObjectMatrix.return_value = w._ready_tcp.reshape(-1).tolist()
        self.assertTrue(w.retract_scissors())
        w.tool_command.assert_not_called()
        w.sim.getObjectMatrix.assert_called_once()      # readback only, no extra time
        self.assertEqual(w.advanced, 0.0)               # same bookkeeping as a landed park
        self.assertIsNone(w._pre_cut_q)
        w._invalidate_fix.assert_called_once()

    def test_accepted_motion_lands_at_homed_pose(self):
        w = self.world(vision=True)
        away = w._ready_tcp.copy()
        away[:, 3] += 0.003
        w.sim.getObjectMatrix.side_effect = [away.reshape(-1).tolist(),
                                             w._ready_tcp.reshape(-1).tolist()]
        self.assertTrue(w.retract_scissors())
        args = w.tool_command.call_args.args
        np.testing.assert_array_equal(args[1], np.zeros(6))     # homed q, not _pre_cut_q
        self.assertEqual(w.advanced, 0.0)
        self.assertIsNone(w._pre_cut_q)
        np.testing.assert_array_equal(w._ready_q, np.zeros(6))  # anchor never deleted
        w.call.assert_not_called()                              # no grip read, no setGrip
        self.assertEqual(w.truth.reads, {})                     # no tissue read in strict vision

    def test_refused_motion_keeps_state_for_retry(self):
        w = self.world()
        w.sim.getObjectMatrix.return_value = w._ready_tcp.copy().reshape(-1).tolist()
        w.tool_command.return_value = False
        w.at_ready_pose = Mock(return_value=False)
        self.assertFalse(w.retract_scissors())
        self.assertIsNotNone(w._ready_q)
        self.assertEqual(w.advanced, 0.003)

    def test_motion_not_landing_is_a_refusal(self):
        w = self.world()
        w.at_ready_pose = Mock(side_effect=[False, False])
        self.assertFalse(w.retract_scissors())
        self.assertIsNotNone(w._ready_q)
        self.assertEqual(w.advanced, 0.003)                     # bookkeeping kept for retry

    def test_nonfinite_advance_refuses(self):
        w = self.world()
        w.advanced = float('nan')
        self.assertFalse(w.retract_scissors())
        w.tool_command.assert_not_called()


# Private driver: reusing test_cut_fsm.run would leak these scenarios into its RESULTS
# dict and therefore into the historical fsm_unit_coverage.json.
from cut_fsm import CutTaskFSM, S  # noqa: E402
from fake_world import FakeWorld  # noqa: E402

CUT_CFG = json.loads((ROOT / 'config' / 'task' / 'cut_task.json').read_text(encoding='utf-8'))


def run_fsm(cut_only=False, vision=False, **events):
    w = FakeWorld(cut_only=cut_only, vision=vision, **events)
    # The real FakeWorld does not log closures; the spy records them so a test can
    # prove no extra bite happened merely to park the cutter.
    orig_close = w.close_scissors

    def close_spy():
        w.actions.append(('close_scissors', w.tick))
        return orig_close()
    w.close_scissors = close_spy
    cfg = json.loads(json.dumps(CUT_CFG)) if vision else CUT_CFG
    if vision:
        cfg['cut']['control_source'] = 'vision'
    m = CutTaskFSM(w, cfg, cut_only=cut_only)
    m.run()
    return m, w


class CompletionGuardTests(unittest.TestCase):
    """FSM _complete: an open or misplaced endpoint is FAILED, never a closed park."""

    def test_open_jaws_at_the_endpoint_fail_without_extra_closure(self):
        w = FakeWorld()
        orig_close = w.close_scissors

        def close_spy():
            w.actions.append(('close_scissors', w.tick))
            return orig_close()
        w.close_scissors = close_spy
        # The physical jaw readback reports open at the stop even though the last
        # commanded motion was a cut closure; no setGrip may follow merely to park.
        w.jaws_closed = lambda: False
        m = CutTaskFSM(w, CUT_CFG)
        m.run()
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('jaws open', m.reason)
        self.assertTrue(any(a[0] == 'close_scissors' for a in w.actions))   # real cuts happened
        last_close = max(i for i, a in enumerate(w.actions) if a[0] == 'close_scissors')
        self.assertEqual([a[0] for a in w.actions[last_close + 1:]],
                         ['retract_scissors', 'hold', 'safe_stop'])       # no extra closure to park
        self.assertNotIn('setGrip', {a[0] for a in w.actions})

    def test_unreached_ready_pose_fails(self):
        m, w = run_fsm(ready_broken=True)
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('ready pose', m.reason)

    def test_nominal_recovery_restores_ready(self):
        m, w = run_fsm()
        self.assertEqual(m.state, S.DONE)
        self.assertTrue(w.at_ready_pose())
        self.assertTrue(w.jaws_closed())
        self.assertEqual(w.actions[-1][0], 'retract_scissors')

    def test_truth_recovery_never_closes_to_park(self):
        # Truth mode nominal: actions after the last cutting closure are only the park.
        m, w = run_fsm()
        self.assertNotIn('setGrip', {a[0] for a in w.actions})
        last_close = max(i for i, a in enumerate(w.actions) if a[0] == 'close_scissors')
        self.assertTrue(all(a[0] != 'close_scissors' for a in w.actions[last_close + 1:]))

    def test_open_jaws_shaft_mode_refuses_before_any_motion(self):
        # Shaft-mode recovery could move open blades close-then-park; an open-at-stop
        # endpoint must fail BEFORE any recovery motion is attempted.
        w = FakeWorld()
        w.align_mode = 'shaft'
        w.jaws_closed = lambda: False
        m = CutTaskFSM(w, CUT_CFG)
        m.run()
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('jaws open', m.reason)
        self.assertNotIn('retract_scissors', {a[0] for a in w.actions})
        self.assertNotIn('setGrip', {a[0] for a in w.actions})


class PerceptionFlagTests(unittest.TestCase):
    """run_cut_task flag gating: real prerequisites vs boolean type checks, provenance."""

    def cfg(self, align_mode='tool_pose', control_source='truth', **perc):
        c = {'cut': {'align_mode': align_mode, 'control_source': control_source}}
        if perc:
            c['perception'] = dict(perc)
        return c

    def test_defaults_keep_historical_baseline(self):
        p = run_cut_task.resolve_perception_flags(self.cfg())
        self.assertFalse(any(p[f] for f, _ in run_cut_task.PERCEPTION_FLAG_CHAIN))
        flags, version = run_cut_task.protocol_provenance(p)
        self.assertEqual(version, '1_4')
        self.assertFalse(any(flags.values()))

    def test_exclude_marker_without_predictor_is_custom(self):
        p = run_cut_task.resolve_perception_flags(
            self.cfg(exclude_tcp_marker=True, predictive_view=False, material_pattern=True,
                     local_material_chart=True, front_recovery=True))
        flags, version = run_cut_task.protocol_provenance(p)
        self.assertIsNone(version)                              # non-prefix: custom development
        self.assertTrue(flags['exclude_tcp_marker'] and not flags['predictive_view'])

    def test_ray_fit_without_recovery_is_custom(self):
        p = run_cut_task.resolve_perception_flags(
            self.cfg(control_source='vision', ray_marker_fit=True))
        flags, version = run_cut_task.protocol_provenance(p)
        self.assertIsNone(version)
        self.assertTrue(flags['ray_marker_fit'] and not flags['recovery_view_after_cut'])

    def test_exact_prefix_keeps_registered_version(self):
        p = run_cut_task.resolve_perception_flags(self.cfg(control_source='vision'))
        flags, version = run_cut_task.protocol_provenance(p)
        enabled = [f for f, _ in run_cut_task.PERCEPTION_FLAG_CHAIN if flags[f]]
        chain = [f for f, _ in run_cut_task.PERCEPTION_FLAG_CHAIN]
        self.assertEqual(enabled, chain[:len(enabled)])
        self.assertEqual(version, run_cut_task.PERCEPTION_FLAG_CHAIN[len(enabled) - 1][1])

    def test_mixed_invalid_prerequisites_refused(self):
        with self.assertRaises(ValueError):                     # predictive without tool_pose
            run_cut_task.resolve_perception_flags(
                self.cfg(align_mode='shaft', predictive_view=True))
        with self.assertRaises(ValueError):                     # chart without pattern
            run_cut_task.resolve_perception_flags(
                self.cfg(material_pattern=False, local_material_chart=True))
        with self.assertRaises(ValueError):                     # ray fit without markers
            run_cut_task.resolve_perception_flags(
                self.cfg(markers=False, ray_marker_fit=True))
        for bad in ('material_pattern', 'front_recovery', 'exclude_tcp_marker',
                    'recovery_view_after_cut', 'ray_marker_fit', 'markers'):
            with self.assertRaises(ValueError, msg=bad):
                run_cut_task.resolve_perception_flags(self.cfg(**{bad: 'yes'}))


class LegacyTissueTests(unittest.TestCase):
    """Locked collectors opt into the frozen source; the default path uses the live one."""

    def test_frozen_source_exists_and_differs_from_current(self):
        live = (ROOT / 'src' / 'tissue' / 'tissue.lua').read_bytes()
        frozen = LEGACY_TISSUE.read_bytes()
        self.assertTrue(frozen)
        self.assertNotEqual(live, frozen)                       # live tissue gained seam options
        self.assertNotIn(b'seam_mode', frozen)                  # freeze predates the new options
        self.assertIn(b'seam_mode', live)

    def test_locked_collectors_opt_in_explicitly(self):
        import ruler_world, surface_seam_world, two_bite
        for cls in (ruler_world.RulerWorld, surface_seam_world.SurfaceSeamWorld,
                    two_bite.TwoBiteWorld):
            self.assertEqual(cls.TISSUE_SOURCE, LEGACY_TISSUE, cls.__name__)
        self.assertFalse(hasattr(SimWorld, 'TISSUE_SOURCE'))    # default path never pins

    def test_refreshed_source_selects_explicit_tissue(self):
        embedded = ('CONFIG={}\nCOLLISION_PAIRS={}\nINSTRUMENTS={}\n'
                    '-- Software RCM controller for the endoscope task scene\nold-lua\n')
        frozen = simtools.refreshed_controller_source(embedded, tissue=True,
                                                      tissue_source=LEGACY_TISSUE)
        live = simtools.refreshed_controller_source(embedded, tissue=True)
        self.assertIn(LEGACY_TISSUE.read_text(encoding='utf-8'), frozen)
        self.assertNotEqual(frozen, live)
        self.assertNotIn(b'seam_mode', frozen.encode())         # no new options leak into locked code

    def test_canonical_seam_spec_passes_only_development_fields(self):
        # The flat development sheet forwards exactly seam_mode/guide_band_m/ruler_print.
        w = bare_world()
        w.mode = 'full'
        w.vision = False
        w.canonical_seam = True
        w.roi_proxy = None
        w.sheet_cfg = {'seam_mode': 'surface', 'guide_band_m': 1e-4, 'ruler_print': True,
                       'stiffness': 99}                          # stray mechanics must not pass
        w.pretension = 0.08
        w.frame = Mock(return_value={'pivot': np.zeros(3), 'x': np.array([1., 0., 0.]),
                                     'axis': np.array([0., 0., 1.]), 'tcp': np.zeros(3)})
        w.sheet_origin, w.u, w.v, w.grasp_points = np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]), {}
        w.material_pattern = False
        w.tissue_source = None                         # current controller: readback required

        def controller(name, *a):
            if name == 'tissueTransverseRestLengths':
                return np.full(sim_world.NV, 0.001 * 0.92).tolist()
            return None
        w.call = Mock(side_effect=controller)
        with patch.object(sim_world.workspace, 'layout', return_value=dict(
                v=np.array([0., 1., 0.]), u=np.array([1., 0., 0.]),
                origin=np.zeros(3), grasp_points={})):
            w.place_sheet()
        spec = w.call.call_args_list[0].args[1]
        self.assertEqual(spec['seam_mode'], 'surface')
        self.assertEqual(spec['guide_band_m'], 1e-4)
        self.assertIs(spec['ruler_print'], True)
        self.assertNotIn('stiffness', spec)


class SheetConfigTests(unittest.TestCase):
    """cfg['sheet'] validation: one input, every option checked before the sim sees it."""

    def test_unknown_and_mode_errors_refused(self):
        for sheet in ({'stiffness': 99}, {'seam_mode': 'holes'},
                      {'guide_band_m': 1e-4}, {'ruler_print': True},
                      {'seam_mode': 'surface', 'ruler_print': 'yes'},
                      'surface'):
            with self.assertRaises(ValueError, msg=repr(sheet)):
                SimWorld._check_sheet_cfg(sheet)

    def test_guide_band_must_be_finite_positive_number(self):
        for bad in (True, False, 0.0, -1e-4, float('nan'), float('inf'), '1e-4'):
            with self.assertRaises(ValueError, msg=repr(bad)):
                SimWorld._check_sheet_cfg({'seam_mode': 'surface', 'guide_band_m': bad})
        self.assertEqual(SimWorld._check_sheet_cfg(
            {'seam_mode': 'surface', 'guide_band_m': 1e-4})['guide_band_m'], 1e-4)
        self.assertEqual(SimWorld._check_sheet_cfg(None), {})

    def gate_world(self, **over):
        w = bare_world()
        w.mode = 'full'
        w.vision = False
        w.roi_proxy = None
        w.tissue_source = None
        w.sheet_cfg = {'seam_mode': 'surface'}
        for k, v in over.items():
            setattr(w, k, v)
        return w

    def test_surface_mode_is_refused_not_silently_downgraded(self):
        # Every incompatible combination must raise rather than silently keep going.
        self.assertTrue(self.gate_world()._canonical_gate({}))
        for over in ({'mode': 'cut_only'}, {'vision': True},
                     {'roi_proxy': object()}, {'tissue_source': LEGACY_TISSUE}):
            with self.assertRaises(ValueError, msg=repr(over)):
                self.gate_world(**over)._canonical_gate({})
        with self.assertRaises(ValueError):
            self.gate_world()._canonical_gate({'perception': {'material_pattern': True}})


class ClearViewParkTests(unittest.TestCase):
    """clear_view parks only when the actual _ready_tcp target sits behind the cutter."""

    def world(self, ready, current):
        w = bare_world()
        w.align_mode = 'tool_pose'
        w._ready_tcp = tcp_at(ready)
        w._ready_q = np.zeros(6)
        w.by = {'psm2_tcp': 99}
        w.sim = SimpleNamespace(
            getObjectMatrix=Mock(return_value=tcp_at(current).reshape(-1).tolist()))
        w.retract_scissors = Mock(return_value=True)
        w._move_scissors = Mock(return_value=True)
        return w

    def test_parks_only_on_backward_displacement(self):
        # Ready target behind the current TCP along the jaw axis -> a guarded park is allowed.
        w = self.world(ready=[0., 0., 0.038], current=[0., 0., 0.04])
        self.assertTrue(w.clear_view(0.004, prefer_park=True))
        w.retract_scissors.assert_called_once()
        w._move_scissors.assert_not_called()

    def test_ready_ahead_of_cutter_never_moves_forward_to_park(self):
        # _pre_cut_q may LOOK parked (behind) while the actual _ready_tcp target is ahead;
        # parking on the wrong anchor would drive the closed blades forward.
        w = self.world(ready=[0., 0., 0.045], current=[0., 0., 0.04])
        w._pre_cut_q = np.zeros(6)
        self.assertTrue(w.clear_view(0.004, prefer_park=True))
        w.retract_scissors.assert_not_called()
        w._move_scissors.assert_called_once()


if __name__ == '__main__':
    unittest.main()
