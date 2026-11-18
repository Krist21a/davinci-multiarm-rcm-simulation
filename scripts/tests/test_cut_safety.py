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
            return True
        world.retract_scissors = withdraw
        fsm = CutTaskFSM(world, CFG, cut_only=True)
        self.assertEqual(fsm.run(), S.FAILED)
        self.assertIn('torn', fsm.reason)

    def test_excess_strain_after_retraction_is_not_done(self):
        world = FakeWorld(cut_only=True)
        def withdraw():
            world.base_strain = 0.30
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