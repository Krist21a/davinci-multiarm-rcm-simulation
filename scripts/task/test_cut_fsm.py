"""Scenario tests for scripts/task/cut_fsm.py on the kinematic FakeWorld (no CoppeliaSim).

  python -m unittest test_cut_fsm -v          (from scripts/task)
Also writes validation/task/fsm_unit_coverage.json with the states and transitions every scenario reached.
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cut_fsm import S, CutTaskFSM  # noqa: E402
from fake_world import FakeWorld  # noqa: E402

REPO = HERE.parents[1]
CFG = json.loads((REPO / 'config' / 'task' / 'cut_task.json').read_text(encoding='utf-8'))
RESULTS = {}


def run(name, cut_only=False, **events):
    w = FakeWorld(cut_only=cut_only, **events)
    m = CutTaskFSM(w, CFG, cut_only=cut_only)
    m.run()
    RESULTS[name] = dict(events={k: (sorted(v) if isinstance(v, set) else v) for k, v in events.items()}, cut_only=cut_only,
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

    @classmethod
    def tearDownClass(cls):
        visited = set()
        trans = set()
        for r in RESULTS.values():
            visited |= set(r['visited'])
            trans |= set(r['transitions'])
        missing = sorted(s.name for s in S if s.name not in visited)
        out = REPO / 'validation' / 'task' / 'fsm_unit_coverage.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(dict(states={s.name: s.value for s in S}, states_reached=sorted(visited),
                                    states_missing=missing, transitions=sorted(trans), scenarios=RESULTS),
                               indent=2, ensure_ascii=False) + '\n')
        assert not missing, f'states never reached: {missing}'


if __name__ == '__main__':
    unittest.main()
