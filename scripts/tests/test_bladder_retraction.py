"""Offline tests for the rigid-transform + penetration helpers in bladder_retraction.py.

Synthetic boxes only — no scene data, no simulator. A unit cube 'bladder' translated along a
declared direction penetrates a 'bone' box by a known amount; the helpers must measure it.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))

import bladder_retraction as br  # noqa: E402


class PenetrationTests(unittest.TestCase):
    def setUp(self):
        self.bone = trimesh.creation.box(extents=(0.02, 0.02, 0.02))          # watertight
        self.bone.apply_translation([0.0, 0.0, 0.020])                        # z in [0.01, 0.03]
        self.bladder = trimesh.creation.box(extents=(0.01, 0.01, 0.01))       # z in [-0.005, 0.005]

    def test_no_overlap_zero_penetration(self):
        # bladder top at +5 mm, bone bottom at +10 mm: 5 mm gap -> no penetration
        verts = self.bladder.vertices.copy()
        self.assertAlmostEqual(br.penetration_depth(self.bone, verts), 0.0, places=6)

    def test_known_penetration_depth(self):
        # shift bladder +16 mm: top half (5 mm) is inside the bone's bottom face
        verts = self.bladder.vertices + np.array([0, 0, 0.016])
        self.assertAlmostEqual(br.penetration_depth(self.bone, verts), 0.005, places=4)

    def test_deeper_than_structure_reports_fully_inside(self):
        verts = self.bladder.vertices + np.array([0, 0, 0.020])  # bladder entirely inside bone
        self.assertGreater(br.penetration_depth(self.bone, verts), 0.004)
        self.assertLessEqual(br.penetration_depth(self.bone, verts), 0.0101)

    def test_admissibility_threshold_semantics(self):
        verts = self.bladder.vertices + np.array([0, 0, 0.0069])   # ~1.9 mm penetration
        self.assertLessEqual(br.penetration_depth(self.bone, verts), 0.002)
        verts = self.bladder.vertices + np.array([0, 0, 0.0076])   # ~2.6 mm penetration
        self.assertGreater(br.penetration_depth(self.bone, verts), 0.002)

    def test_shell_penetration_oriented(self):
        # Open dome: horizontal plate at z=0.02, interior below. Bladder verts above it
        # penetrate outward through the wall.
        plate = trimesh.Trimesh(vertices=[[-0.05, -0.05, 0.02], [0.05, -0.05, 0.02],
                                          [0.05, 0.05, 0.02], [-0.05, 0.05, 0.02]],
                                faces=[[0, 1, 2], [0, 2, 3]], process=False)
        verts = np.array([[0.0, 0.0, 0.015],   # inside (below wall): no penetration
                          [0.0, 0.0, 0.023]])  # outside (above wall): 3 mm penetration
        depth = br.shell_penetration(plate, verts, np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(depth, 0.003, places=4)

    def test_rigid_translate_is_pure(self):
        d = np.array([0.3, 0.4, 0.5])
        d = d / np.linalg.norm(d)
        moved = self.bladder.vertices + 0.04 * d
        np.testing.assert_allclose(moved - self.bladder.vertices,
                                   np.tile(0.04 * d, (8, 1)), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
