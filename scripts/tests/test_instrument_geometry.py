"""Independent wrist/yaw/jaw transform checks against pinned dVRK xacro data."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'build'))
import instruments as inst
from articulation import append_yaw_collision_pairs, jaw_coordinates


class InstrumentGeometry(unittest.TestCase):
    def test_yaw_and_grip_are_two_not_three_independent_coordinates(self):
        for yaw, gap in [(0, 1), (.3, 0), (.4, .8), (-.5, .4)]:
            a, b = jaw_coordinates(yaw, gap)
            self.assertAlmostEqual((a + b) / 2, yaw)
            self.assertAlmostEqual(a - b, gap)
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                jaw_coordinates(value, .1)

    def test_pinned_yaw_offsets_and_axes(self):
        for model in ['420172', '420179', '420093']:
            d = inst.descriptor(model)
            joints, _ = inst.parse_stage(inst.XACRO / 'wrist_yaw' / f"wrist_yaw_{d['stages']['wrist_yaw']}.urdf.xacro")
            yaw = joints['wrist_yaw']
            self.assertEqual(yaw['parent'], 'wrist_pitch_link')
            self.assertEqual(yaw['child'], 'wrist_yaw_link')
            np.testing.assert_allclose(yaw['axis'], [0, 0, 1])
            self.assertAlmostEqual(np.linalg.norm(yaw['origin'][:3, 3]), d['kinematics']['pitch_to_yaw_m'])

    def test_leaf_mimics_are_dependent_half_opening(self):
        for tip in ['172', '179', '093']:
            joints, _ = inst.parse_stage(inst.XACRO / 'tip' / f'tip_{tip}.urdf.xacro')
            leaves = [joints['jaw_1'], joints['jaw_2']]
            self.assertEqual({j['mimic']['joint'] for j in leaves}, {'jaw'})
            self.assertEqual({j['mimic']['multiplier'] for j in leaves}, {-.5, .5})
            self.assertEqual(joints['tool_tip']['parent'], 'wrist_yaw_link')

    def test_prograsp_hinge_is_dependent_not_an_extra_dof(self):
        joints, _ = inst.parse_stage(inst.XACRO / 'tip' / 'tip_093.urdf.xacro')
        hinge = joints['jaw_axis']
        self.assertEqual(hinge['parent'], 'wrist_yaw_link')
        self.assertEqual(hinge['mimic']['joint'], 'jaw')
        c = hinge['mimic']
        self.assertAlmostEqual(c['offset'], .0077)
        self.assertAlmostEqual(c['offset'] + c['multiplier'] * 1.0472, .0055, places=7)
        # Offset hinge must travel with yaw, unlike simply changing the two old jaw angles.
        R = inst.rpy_matrix(0, 0, .3)
        self.assertGreater(np.linalg.norm(R @ np.array([0, .0077, 0]) - [0, .0077, 0]), .001)

    def test_split_housing_preserves_cross_arm_pair_coverage(self):
        a, b = 'psm1_wrist_ref_respondable_visual', 'psm2_wrist_ref_respondable_visual'
        labels = [[a, b], [a, 'Patient_Bladder']]
        pairs = append_yaw_collision_pairs(labels, ['psm1', 'psm2'])
        self.assertIn(['psm1_wrist_yaw_visual', 'psm2_wrist_yaw_visual'], pairs)
        self.assertIn(['psm1_wrist_yaw_visual', 'Patient_Bladder'], pairs)
        self.assertEqual(len(pairs), 22)
        self.assertIn(['psm1_yaw_ref1_respondable_visual', 'psm1_tool_main_respondable_visual'], pairs)
        self.assertEqual(append_yaw_collision_pairs(pairs, ['psm1', 'psm2']), pairs)


if __name__ == '__main__':
    unittest.main()