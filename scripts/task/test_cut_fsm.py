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
