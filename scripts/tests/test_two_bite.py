"""Two-bite stopping must not be a shortcut to strict endpoint success."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/task'))
from cut_fsm import CutTaskFSM, Observation, S, completion_issues
from run_two_bite_pair import PairFSM
from two_bite import SimWorld, TwoBiteFSM, TwoBiteWorld


class TwoBiteTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'config/task/cut_task.json').read_bytes())
        self.cfg['cut']['control_source'] = 'vision'
        self.cfg['perception']['front_recovery'] = True
        self.world = Mock(bites=2, tick=22)
        self.fsm = TwoBiteFSM(self.world, self.cfg)
        self.fsm.state, self.fsm.cut_started = S.CLEAR_VIEW, True
        self.obs = Observation(True, True, .1, {'psm1': 5, 'psm3': 5}, .07, 0, .005,
                               cut_progress_m=.0054, tissue_visible=True)
        self.world.direct_fix.return_value = True
        self.world.finish_park.return_value = True
        self.world.observe.return_value = self.obs

    def test_separate_task_finishes_only_after_guarded_park_not_at_old_endpoint(self):
        self.assertTrue(self.fsm.supervise(self.obs))
        self.assertEqual(self.fsm.state, S.DONE)
        self.world.finish_park.assert_called_once()
        self.assertIn('cut length below target', completion_issues(self.obs, self.cfg))
        strict = CutTaskFSM(self.world, self.cfg)
        strict.state = S.CLEAR_VIEW
        self.assertFalse(strict.supervise(self.obs))

    def test_no_park_on_memory_or_tear_and_no_done_after_rejected_or_damaging_park(self):
        self.world.direct_fix.return_value = False
        self.assertFalse(self.fsm.supervise(self.obs))
        self.world.finish_park.assert_not_called()
        self.world.direct_fix.return_value = True
        self.world.finish_park.return_value = False
        self.assertTrue(self.fsm.supervise(self.obs))
        self.assertEqual(self.fsm.state, S.FAILED)
        self.fsm.state = S.CLEAR_VIEW
        self.world.finish_park.return_value = True
        self.world.observe.return_value = Observation(False, False, None, {}, .1, 1, .005)
        self.assertTrue(self.fsm.supervise(self.obs))
        self.assertEqual(self.fsm.state, S.FAILED)
        self.assertIn('post-withdrawal', self.fsm.reason)

    def test_hard_cap_at_actuator_and_no_second_bite_on_pre_cut_evidence(self):
        w = TwoBiteWorld.__new__(TwoBiteWorld)
        w.bites, w.tick, w.awaiting_new_slit = 1, 12, False
        w._fix_valid = lambda: True
        w._fix = dict(tick=12)
        cut = dict(mode='pre_cut', front_valid=True, material_chart=dict(valid=True))
        w._last_est = dict(scissors=dict(valid=True), cut=cut)
        with patch.object(SimWorld, 'close_scissors', return_value=True) as close:
            self.assertFalse(w.close_scissors())
            cut['mode'] = 'slit'
            w._fix['tick'] -= 1
            self.assertFalse(w.close_scissors())
            w._fix['tick'] += 1
            self.assertTrue(w.close_scissors())
            self.assertFalse(w.close_scissors())
            self.assertFalse(w.advance_scissors(.001))
            self.assertFalse(w.retract_scissors())
            close.assert_called_once()

    def test_early_visual_endpoint_cannot_be_a_two_bite_success(self):
        self.world.bites = 1
        self.fsm._complete(self.obs)
        self.assertEqual(self.fsm.state, S.FAILED)
        self.world.finish_park.assert_not_called()

    def test_second_bite_optical_traction_capped_and_vetoed_on_conflicting_sign(self):
        self.world.perturbed = True
        self.world.reference_width = .017
        self.world.args.strategy = 'traction'
        self.world.width.return_value = .016
        self.world.event_log = []
        fsm = PairFSM(self.world, self.cfg)
        fsm.retractors = ['psm1', 'psm3']
        fsm.state, fsm.cut_started = S.RETRACT_ADJUST, True
        self.obs.tension = .01
        fsm._adjust_tension(self.obs)
        self.assertEqual([c.args for c in self.world.retract.call_args_list],
                         [('psm1', .0002), ('psm3', .0002)])
        self.world.retract.reset_mock()
        self.world.width.return_value = .018
        fsm._adjust_tension(self.obs)
        self.world.retract.assert_not_called()
        self.assertEqual(fsm.state, S.FAILED)
