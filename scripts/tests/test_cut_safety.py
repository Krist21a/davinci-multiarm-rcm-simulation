"""Safety regressions independent of CoppeliaSim; run with unittest discovery."""
import copy
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'task'))
from cut_fsm import CutTaskFSM, S, completion_issues  # noqa: E402
from fake_world import FakeWorld  # noqa: E402

CFG = json.loads((REPO / 'config' / 'task' / 'cut_task.json').read_text(encoding='utf-8'))


class LateEventWorld(FakeWorld):
    def __init__(self, event):
        super().__init__()
        self.event = event
        self.grasped = {'psm1': 6, 'psm3': 6}
        self.base_strain = 0.08
        self.aligned = True
        self.closures = 0

    def observe(self):
        obs = super().observe()
        if self.tick == 2:
            if self.event == 'fault':
                obs.faults = ['injected after advance']
            elif self.event == 'tear':
                obs.torn = 1
            elif self.event == 'slip':
                obs.grasped = {}
            elif self.event == 'strain':
                obs.tension = 0.8
            elif self.event == 'hidden':
                obs.cut_site_visible = False
            elif self.event == 'drift':
                obs.cut_site_offset = 0.9
            elif self.event == 'nan':
                obs.tension = float('nan')
        return obs

    def close_scissors(self):
        self.closures += 1
        return super().close_scissors()


class CutSafety(unittest.TestCase):
    def test_last_clear_attempt_view_after_reacquisition_spends_existing_camera_budget(self):
        cfg = copy.deepcopy(CFG)
        cfg['cut']['control_source'] = 'vision'
        cfg['perception']['recovery_view_after_cut'] = True
        for started, spent, expected in [(True, 2, 1), (True, 3, 0), (False, 2, 0)]:
            with self.subTest(started=started, spent=spent):
                w = FakeWorld(cut_only=True, vision=True)
                obs = w.observe()
                obs.cut_site_visible, obs.tissue_visible, obs.scissors_visible = False, True, True
                w.awaiting_new_slit = False
                w.clear_view = w.close_scissors = lambda *args: self.fail('unexpected cutter motion')
                fsm = CutTaskFSM(w, cfg, cut_only=True)
                fsm.cut_started, fsm.state = started, S.CLEAR_VIEW
                fsm.attempts[S.CLEAR_VIEW], fsm.attempts[S.CAMERA_ADJUST] = 2, spent
                fsm.on_clear_view(obs)
                self.assertEqual(fsm.attempts[S.CAMERA_ADJUST], spent + expected)
                self.assertEqual(w.camera_moves, expected)

    def test_endpoint_caps_motion_and_refuses_changed_sweep_before_closure(self):
        cfg = copy.deepcopy(CFG)
        cfg['cut']['control_source'] = 'vision'
        world = FakeWorld(cut_only=True, vision=True, progress_bias_m=0.008)
        world.endpoint_advance_budget = iter((0.0006, -0.0003)).__next__
        world.closure_aligned = lambda: True
        world.close_scissors = lambda: self.fail('closed beyond visual sweep limit')
        fsm = CutTaskFSM(world, cfg, cut_only=True)
        fsm.state = S.CUT
        fsm.tick()
        self.assertAlmostEqual(world.pending, 0.0006)
        self.assertEqual(fsm.state, S.FAILED)
        self.assertIn('closure refused: visual endpoint', fsm.reason)

    def test_occlusion_recovery_resets_the_clear_view_attempts(self):
        world = FakeWorld(cut_only=True, vision=True)
        fsm = CutTaskFSM(world, CFG, cut_only=True)
        fsm.state = S.OCCLUDED
        fsm.attempts[S.CLEAR_VIEW] = CFG['budgets']['clear_view_attempts']
        fsm.tick()
        self.assertEqual(fsm.state, S.RESUME)
        self.assertEqual(fsm.attempts[S.CLEAR_VIEW], 0)

    def test_visual_closure_checks_new_progress_and_alignment_after_advance(self):
        cfg = copy.deepcopy(CFG)
        cfg['cut']['control_source'] = 'vision'
        for progress, aligned, expected in ((0.0095, True, 'retraction refused'),
                                             (None, True, 'invalid visual progress'),
                                             (0.008, False, 'fresh alignment')):
            with self.subTest(progress=progress, aligned=aligned):
                world = FakeWorld(cut_only=True, vision=True, park_fails=True, progress_bias_m=0.008)
                observe = world.observe

                def report():
                    obs = observe()
                    if world.tick == 2:
                        obs.cut_progress_m = progress
                    return obs

                world.observe = report
                world.closure_aligned = lambda: aligned
                world.close_scissors = lambda: self.fail('unexpected closure')
                fsm = CutTaskFSM(world, cfg, cut_only=True)
                fsm.state = S.CUT
                fsm.tick()
                self.assertEqual(fsm.state, S.FAILED)
                self.assertIn(expected, fsm.reason)

    def test_all_interlocks_checked_after_advance(self):
        expected = {'fault': S.FAILED, 'tear': S.FAILED, 'slip': S.RELOCALIZE,
                    'strain': S.RETRACT_ADJUST, 'hidden': S.OCCLUDED,
                    'drift': S.CAMERA_ADJUST, 'nan': S.FAILED}
        for event, state in expected.items():
            with self.subTest(event=event):
                world = LateEventWorld(event)
                fsm = CutTaskFSM(world, CFG)
                fsm.state = S.CUT
                fsm.retractors = ['psm1', 'psm3']
                fsm.tick()
                self.assertEqual(world.closures, 0)
                self.assertEqual(fsm.state, state)
                self.assertEqual(fsm.log[-1]['obs']['faults'], ['injected after advance'] if event == 'fault' else [])

    def test_post_advance_camera_retry_requires_fresh_safe_closure_evidence(self):
        cfg = copy.deepcopy(CFG)
        cfg['cut']['control_source'] = 'vision'
        cfg['perception'] = {'front_recovery': True}
        for scenario, expected in (('restored', S.CUT), ('still_hidden', S.CLEAR_VIEW),
                                   ('torn_after_move', S.FAILED), ('sweep_after_move', S.FAILED),
                                   ('torn_before_move', S.FAILED), ('budget_exhausted', S.CLEAR_VIEW)):
            with self.subTest(scenario=scenario):
                world = FakeWorld(cut_only=True, vision=True)
                act = world.observe
                def observe():
                    obs = act()
                    if world.tick == 2 or (world.tick == 3 and scenario == 'still_hidden'):
                        obs.cut_site_visible = False
                    if ((world.tick == 2 and scenario == 'torn_before_move')
                            or (world.tick == 3 and scenario == 'torn_after_move')):
                        obs.torn = 1
                    return obs

                world.observe = observe
                world.endpoint_advance_budget = lambda: -.001 if scenario == 'sweep_after_move' and world.tick == 3 else .003
                world.closure_aligned = lambda: True
                called = []
                world.close_scissors = lambda: called.append(world.tick) or True
                fsm = CutTaskFSM(world, cfg, cut_only=True)
                fsm.state = S.CUT
                if scenario == 'budget_exhausted':
                    fsm.attempts[S.CAMERA_ADJUST] = cfg['budgets']['camera_adjust_attempts']
                fsm.tick()
                self.assertEqual(fsm.state, expected)
                self.assertEqual(called, [3] if scenario == 'restored' else [])
                self.assertEqual(world.camera_moves, int(scenario not in ('torn_before_move', 'budget_exhausted')))

    def test_predicted_occlusion_never_advances_to_closure(self):
        cfg = copy.deepcopy(CFG)
        cfg['cut']['control_source'] = 'vision'
        for exhausted, expected in ((False, S.CAMERA_ADJUST), (True, S.FAILED)):
            with self.subTest(exhausted=exhausted):
                world = FakeWorld(cut_only=True, vision=True)
                def refuse(_step):
                    world.view_blocked = True
                    return False

                world.advance_scissors = refuse
                world.close_scissors = lambda: self.fail('blocked candidate must not close')
                fsm = CutTaskFSM(world, cfg, cut_only=True)
                fsm.state = S.CUT
                if exhausted:
                    fsm.attempts[S.CAMERA_ADJUST] = cfg['budgets']['camera_adjust_attempts']
                fsm.tick()
                self.assertEqual(fsm.state, expected)
                self.assertEqual(world.tick, 1)

    def test_retraction_refusal_is_not_done(self):
        world = FakeWorld(cut_only=True)
        world.retract_scissors = lambda: False
        fsm = CutTaskFSM(world, CFG, cut_only=True)
        self.assertEqual(fsm.run(), S.FAILED)
        self.assertIn('retraction refused', fsm.reason)

    def test_damage_during_retraction_is_not_done(self):
        world = FakeWorld(cut_only=True)
        def withdraw():
            world.e['tear'] = world.tick + 1
            world.at_ready = True          # an accepted park restores the ready pose
            return True
        world.retract_scissors = withdraw
        fsm = CutTaskFSM(world, CFG, cut_only=True)
        self.assertEqual(fsm.run(), S.FAILED)
        self.assertIn('torn', fsm.reason)

    def test_excess_strain_after_retraction_is_not_done(self):
        world = FakeWorld(cut_only=True)
        def withdraw():
            world.base_strain = 0.30
            world.at_ready = True          # an accepted park restores the ready pose
            return True
        world.retract_scissors = withdraw
        fsm = CutTaskFSM(world, CFG, cut_only=True)
        self.assertEqual(fsm.run(), S.FAILED)
        self.assertIn('strain above', fsm.reason)

    def test_retraction_adjustment_stops_on_first_refusal(self):
        world = FakeWorld()
        calls = []
        def refuse(arm, step):
            calls.append(arm)
            return False
        world.retract = refuse
        fsm = CutTaskFSM(world, CFG)
        fsm.retractors = ['psm1', 'psm3']
        fsm.state = S.EXPOSE
        fsm._adjust_tension(world.observe())
        self.assertEqual(fsm.state, S.FAILED)
        self.assertEqual(calls, ['psm1'])

    def test_last_advance_has_no_one_mm_floor(self):
        world = LateEventWorld(None)
        world.cut = CFG['cut']['length_m'] - 0.0004
        fsm = CutTaskFSM(world, CFG)
        fsm.state = S.CUT
        fsm.tick()
        self.assertAlmostEqual(world.cut, CFG['cut']['length_m'])

    def test_overcut_limit_is_enforced(self):
        world = FakeWorld(cut_only=True)
        world.cut = CFG['cut']['length_m'] + CFG['cut']['max_overcut_m'] + 0.0001
        fsm = CutTaskFSM(world, CFG, cut_only=True)
        fsm.state = S.CUT
        fsm.tick()
        self.assertEqual(fsm.state, S.FAILED)
        self.assertIn('exceeds', fsm.reason)

    def test_terminal_acceptance_rejects_nonfinite_and_short_cut(self):
        world = FakeWorld(cut_only=True)
        obs = world.observe()
        self.assertIn('cut length below target', completion_issues(obs, CFG))
        obs.cut_length_m = float('nan')
        self.assertIn('invalid final tissue observation', completion_issues(obs, CFG))

    def test_zero_overcut_policy_supported(self):
        cfg = copy.deepcopy(CFG)
        cfg['cut']['max_overcut_m'] = 0.0
        world = FakeWorld(cut_only=True)
        obs = world.observe()
        obs.cut_length_m = cfg['cut']['length_m'] + 0.0001
        self.assertIn('cut exceeds allowed length', completion_issues(obs, cfg))


if __name__ == '__main__':
    unittest.main()