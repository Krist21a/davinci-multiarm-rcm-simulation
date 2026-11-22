"""Scenario tests for scripts/task/cut_fsm.py on the kinematic FakeWorld (no CoppeliaSim).

  python -m unittest test_cut_fsm -v          (from scripts/task)
Also writes validation/task/fsm_unit_coverage.json with the states and transitions every scenario reached.
"""
import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cut_fsm import S, CutTaskFSM, Observation  # noqa: E402
from fake_world import FakeWorld  # noqa: E402

REPO = HERE.parents[1]
CFG = json.loads((REPO / 'config' / 'task' / 'cut_task.json').read_text(encoding='utf-8'))
RESULTS = {}


def run(name, cut_only=False, vision=False, **events):
    w = FakeWorld(cut_only=cut_only, vision=vision, **events)
    cfg = json.loads(json.dumps(CFG)) if vision else CFG
    if vision:
        cfg['cut']['control_source'] = 'vision'
    m = CutTaskFSM(w, cfg, cut_only=cut_only)
    m.run()
    RESULTS[name] = dict(events={k: (sorted(v) if isinstance(v, set) else v) for k, v in events.items()}, cut_only=cut_only,
                         vision=vision,
                         **{k: v for k, v in m.report().items() if k != 'log'},
                         path=[e['to'] for e in m.log if 'to' in e])
    return m, w


class Scenarios(unittest.TestCase):
    def path(self, m):
        return [e['to'] for e in m.log if 'to' in e]

    def test_nominal(self):
        m, w = run('nominal')
        self.assertEqual(m.state, S.DONE)
        self.assertEqual(self.path(m)[:5], ['LOCALIZE', 'GRASP', 'EXPOSE', 'ALIGN', 'CUT'])
        self.assertGreaterEqual(w.cut, CFG['cut']['length_m'])
        self.assertEqual(m.retractors, ['psm1', 'psm3'])

    def test_transient_occlusion_during_cut(self):
        m, w = run('transient_occlusion', occlude=(12, 14))
        self.assertEqual(m.state, S.DONE)
        p = self.path(m)
        i = p.index('OCCLUDED')
        self.assertEqual(p[i:i + 3], ['OCCLUDED', 'RESUME', 'ALIGN'])
        self.assertEqual(w.closures_while_hidden, 0)

    def test_persistent_occlusion_needs_camera(self):
        m, w = run('persistent_occlusion', occlude_until_camera=12)
        self.assertEqual(m.state, S.DONE)
        p = self.path(m)
        self.assertIn('CAMERA_ADJUST', p)
        self.assertLess(p.index('OCCLUDED'), p.index('CAMERA_ADJUST'))
        self.assertEqual(w.closures_while_hidden, 0)

    def test_view_drift(self):
        m, w = run('view_drift', drift=11)
        self.assertEqual(m.state, S.DONE)
        self.assertIn('CAMERA_ADJUST', self.path(m))
        self.assertNotIn('OCCLUDED', self.path(m))

    def test_tension_drop(self):
        m, w = run('tension_drop', relax=12)
        self.assertEqual(m.state, S.DONE)
        p = self.path(m)
        self.assertEqual(p[p.index('RETRACT_ADJUST') + 1], 'RESUME')

    def test_slip_regrasp(self):
        m, w = run('slip_regrasp', slip={'psm1': 12})
        self.assertEqual(m.state, S.DONE)
        p = self.path(m)
        i = p.index('RELOCALIZE')
        self.assertEqual(p[i:i + 3], ['RELOCALIZE', 'RESUME', 'EXPOSE'])
        self.assertIsNone(m.handover)

    def test_slip_handover_to_psm3(self):
        m, w = run('slip_handover', slip={'psm1': 9, 'psm3': 9}, regrasp_fails={'psm1'})
        self.assertEqual(m.state, S.DONE)
        self.assertEqual(m.handover['lost'], 'psm1')
        self.assertEqual(m.handover['taken_by'], 'psm3')

    def test_takeover_point_unreachable_fails(self):
        # Nothing on the lost side is inside the taking-over arm's limits: honest FAILED instead of
        # grasping a point on the other side.
        m, w = run('takeover_unreachable', slip={'psm1': 9, 'psm3': 9}, regrasp_fails={'psm1'},
                   takeover_unreachable={'psm3'})
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('no reachable takeover point on the lost side', m.reason)

    def test_handover_records_requested_and_taken_points(self):
        m, w = run('handover_record', slip={'psm1': 9, 'psm3': 9}, regrasp_fails={'psm1'})
        self.assertEqual(m.state, S.DONE)
        self.assertEqual(m.handover['lost'], 'psm1')
        self.assertEqual(m.handover['taken_by'], 'psm3')
        self.assertEqual(m.handover['requested_point'], 'left flap')
        self.assertEqual(m.handover['taken_point'], 'left flap')
        self.assertFalse(m.handover['substituted'])

    def test_cutter_is_parked_before_the_takeover_grasp(self):
        # The scissors parked mid-cut block the taking-over arm's approach across the midline;
        # the FSM must park them first and record it in the handover.
        m, w = run('handover_park', slip={'psm1': 9, 'psm3': 9}, regrasp_fails={'psm1'})
        self.assertEqual(m.state, S.DONE)
        self.assertTrue(m.handover['cutter_parked'])
        grasp = next(i for i, a in enumerate(w.actions)
                     if a[0] == 'grasp' and a[1] == 'psm3' and a[2] == 'left flap')
        self.assertEqual(w.actions[grasp - 1][0], 'retract_scissors')

    def test_cutter_cannot_park_fails_the_handover(self):
        m, w = run('park_fails', slip={'psm1': 9, 'psm3': 9}, regrasp_fails={'psm1'}, park_fails=True)
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('cutter could not be parked', m.reason)
        # No grasp at the lost point was attempted while the cutter still blocked the approach.
        self.assertFalse([a for a in w.actions if a[0] == 'grasp' and a[1] == 'psm3' and a[2] == 'left flap'])

    def test_handover_keeps_the_taken_over_point(self):
        # The assist arm must still hold the primary (-u side) point afterwards: re-grasping its own
        # point from the stale observation would undo the takeover and drop the primary point.
        m, w = run('handover_point', slip={'psm1': 9, 'psm3': 9}, regrasp_fails={'psm1'})
        self.assertEqual(m.state, S.DONE)
        self.assertEqual(m.retractors, ['psm3'])
        self.assertEqual(w.points['psm3'], 'left flap')
        self.assertNotIn('psm1', w.points)
        undo = [a for a in w.actions if a[0] == 'grasp' and a[1] == 'psm3' and a[2] == 'right flap' and a[3] >= m.handover['tick']]
        self.assertEqual(undo, [])

    def test_handover_then_tension_adjust_pulls_the_taking_over_arm(self):
        # psm1 slips and cannot regrasp, so psm3 takes the -u point; a later tension drop during the
        # resumed cut must pull psm3 — the arm holding it — and the task still completes.
        m, w = run('handover_tension', slip={'psm1': 9, 'psm3': 9}, regrasp_fails={'psm1'}, relax=14)
        self.assertEqual(m.state, S.DONE)
        p = self.path(m)
        self.assertLess(p.index('RELOCALIZE'), p.index('RETRACT_ADJUST'))
        pulls = [a for a in w.actions if a[0] == 'retract' and a[3] >= m.handover['tick']]
        self.assertTrue(pulls)
        self.assertEqual({a[1] for a in pulls}, {'psm3'})
        self.assertTrue(all(a[2] > 0 for a in pulls))

    def test_tension_adjust_pulls_the_arm_holding_the_point(self):
        # Post-handover state: psm3 holds the -u point. A low-tension adjustment must pull psm3 —
        # the arm actually holding the point — and pulling an empty-handed arm is refused.
        w = FakeWorld()
        w.grasped, w.points, w.ever_grasped = {'psm3': 6}, {'psm3': 'left flap'}, {'psm1', 'psm3'}
        w.base_strain = 0.0
        m = CutTaskFSM(w, CFG)
        m.state, m.retractors, m.return_to = S.RETRACT_ADJUST, ['psm3'], S.EXPOSE
        m.tick()
        pulls = [a for a in w.actions if a[0] == 'retract']
        self.assertEqual([a[1] for a in pulls], ['psm3'])
        self.assertGreater(pulls[0][2], 0.0)
        self.assertFalse(w.retract('psm1', CFG['cut']['retract_step_m']))

    def test_assist_unreachable_is_not_fatal(self):
        m, w = run('assist_unreachable', unreachable={'psm3'})
        self.assertEqual(m.state, S.DONE)
        self.assertEqual(m.retractors, ['psm1'])

    def test_primary_unreachable_fails(self):
        m, w = run('primary_unreachable', unreachable={'psm1'})
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('psm1 cannot grasp', m.reason)

    def test_role_check_fails(self):
        m, w = run('role_mismatch', roles_ok=False)
        self.assertEqual(m.state, S.FAILED)
        self.assertEqual(self.path(m), ['FAILED'])

    def test_tear_fails(self):
        m, w = run('tear', tear=13)
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('torn', m.reason)
        self.assertEqual(w.actions[-1][0], 'safe_stop')

    def test_collision_fault_fails(self):
        m, w = run('collision_fault', fault=10)
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('fault', m.reason)

    def test_camera_cannot_help(self):
        m, w = run('camera_useless', occlude_until_camera=12, camera_useless=True)
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('camera', m.reason)

    def test_cut_only(self):
        m, w = run('cut_only', cut_only=True)
        self.assertEqual(m.state, S.DONE)
        self.assertNotIn('GRASP', self.path(m))

    def test_cut_only_tension_drop_fails(self):
        m, w = run('cut_only_relax', cut_only=True, relax=8)
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('no retraction arm', m.reason)

    # -- M5b vision-mode scenarios (front fix + CLEAR_VIEW; m5b_protocol.json) ------------------------------------
    def test_vision_nominal_reaches_done(self):
        m, w = run('vision_nominal', vision=True)
        self.assertEqual(m.state, S.DONE)
        self.assertIn('CLEAR_VIEW', self.path(m))      # every closure voids the fix
        self.assertTrue(any(a[0] == 'clear_view' for a in w.actions))
        self.assertEqual(w.closures_while_hidden, 0)

    def test_vision_front_hidden_recovers_via_clear_view(self):
        m, w = run('vision_front_hidden', vision=True, front_hidden={10, 11})
        self.assertEqual(m.state, S.DONE)
        p = self.path(m)
        i = p.index('CLEAR_VIEW')
        self.assertEqual(p[i:i + 2], ['CLEAR_VIEW', 'RESUME'])

    def test_vision_clear_view_exhaustion_fails(self):
        m, w = run('vision_clear_fail', vision=True,
                   front_hidden=set(range(8, 400)), clear_view_useless=True)
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('clear-and-look exhausted', m.reason)
        n = sum(1 for a in w.actions if a[0] in ('clear_view', 'retract_scissors'))
        self.assertLessEqual(n, 3)                     # back-off, withdrawal-or-back-off, look-only

    def test_vision_both_hidden_goes_occluded(self):
        m, w = run('vision_occluded', vision=True, occlude=(10, 12))
        self.assertEqual(m.state, S.DONE)
        p = self.path(m)
        i = p.index('OCCLUDED')
        self.assertEqual(p[i:i + 2], ['OCCLUDED', 'RESUME'])
        self.assertEqual(w.closures_while_hidden, 0)

    def test_vision_progress_drives_completion_not_truth(self):
        # The fix reports done while the true cut is short: completion fires on
        # cut_progress_m and the truth-based acceptance then honestly fails it.
        m, w = run('vision_progress_poison', vision=True, progress_bias_m=0.009)
        self.assertEqual(m.state, S.FAILED)
        self.assertIn('cut length below target', m.reason)
        # Completion ran at ALIGN entry, before any jaw opening.
        self.assertFalse(any(a[0] == 'open_scissors' for a in w.actions))

    # -- M5b v1.1: RESUME/OCCLUDED with a hidden front also goes to CLEAR_VIEW (vision only) ----
    def test_vision_resume_after_tension_uses_clear_view(self):
        # relax during CUT -> RETRACT_ADJUST; the RESUME tick has the front hidden but
        # scissors+tissue visible -> CLEAR_VIEW -> restored -> RESUME -> ALIGN (v1 went
        # OCCLUDED -> CAMERA_ADJUST instead).
        m, w = run('vision_relax_resume_clear', vision=True, relax=14, front_hidden={18})
        self.assertEqual(m.state, S.DONE)
        p = self.path(m)
        i = p.index('RETRACT_ADJUST')
        self.assertEqual(p[i:i + 4], ['RETRACT_ADJUST', 'RESUME', 'CLEAR_VIEW', 'RESUME'])
        self.assertNotIn('CAMERA_ADJUST', p)

    def test_vision_occluded_with_front_hidden_goes_clear_view(self):
        # Occlusion ends but the front stays hidden: scissors+tissue visible -> CLEAR_VIEW.
        m, w = run('vision_occluded_then_hidden', vision=True, occlude=(10, 11), front_hidden={12})
        self.assertEqual(m.state, S.DONE)
        p = self.path(m)
        i = p.index('OCCLUDED')
        self.assertEqual(p[i:i + 3], ['OCCLUDED', 'CLEAR_VIEW', 'RESUME'])

    def test_truth_resume_not_visible_stays_occluded(self):
        # The v1.1 branch is vision-only: truth mode keeps RESUME -> OCCLUDED.
        w = FakeWorld()
        m = CutTaskFSM(w, CFG)
        m.state = S.RESUME
        m.return_to = S.ALIGN
        m.on_resume(Observation(cut_site_visible=False, scissors_visible=False,
                                cut_site_offset=None, grasped={'psm1': 6, 'psm3': 6},
                                tension=0.08, max_strain=0.08, torn=0,
                                cut_length_m=0.0, faults=[]))
        self.assertEqual(m.state, S.OCCLUDED)

    @classmethod
    def tearDownClass(cls):
        visited = set()
        trans = set()
        for r in RESULTS.values():
            visited |= set(r['visited'])
            trans |= set(r['transitions'])
        missing = sorted(s.name for s in S if s.name not in visited)
        # The default writes the historical coverage file; FSM_UNIT_COVERAGE_JSON is an opt-in
        # versioned override used by run_regressions --out so reruns never rewrite history.
        out = Path(os.environ.get('FSM_UNIT_COVERAGE_JSON', REPO / 'validation' / 'task' / 'fsm_unit_coverage.json'))
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(dict(states={s.name: s.value for s in S}, states_reached=sorted(visited),
                                    states_missing=missing, transitions=sorted(trans), scenarios=RESULTS),
                               indent=2, ensure_ascii=False) + '\n')
        assert not missing, f'states never reached: {missing}'


if __name__ == '__main__':
    unittest.main()
