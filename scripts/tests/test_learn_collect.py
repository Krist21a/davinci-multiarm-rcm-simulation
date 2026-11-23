"""Offline checks for scripts/learn/collect.py (registered study learned_perception_v1).

Determinism of the per-episode sampler, the registered envelope on every draw, the aux-view
target sampler, the holdout refusal, atomic npz writes and the input/label_ key contract of
the three frame kinds -- all without a simulator (a mocked world exercises _perception_tick).
"""
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/learn'))
sys.path.insert(0, str(ROOT / 'scripts/task'))
sys.path.insert(0, str(ROOT / 'scripts/build'))
import collect  # noqa: E402

PROTO = collect.protocol()
RAND = PROTO['data_generation']['episode_randomisation']
ENV = RAND['ecm_initial_view_deg_deg_m']


class FakeTruth:
    """Stand-in for GroundTruth: a purpose() context manager plus fixed label reads."""

    def __init__(self):
        self.reads_by_purpose = {}

    @contextlib.contextmanager
    def purpose(self, name):
        self.reads_by_purpose[name] = self.reads_by_purpose.get(name, 0) + 1
        yield

    def nodes(self):
        return np.zeros((14, 9, 3))

    def cut_front(self):
        return np.zeros(3)

    def cut_length(self):
        return 0.005

    def cut_line_tension(self):
        return 0.07


def fake_world(tmp, aux_views=2, hd_twin=True):
    """LearnWorld.__new__ with every attribute _perception_tick touches mocked."""
    w = collect.LearnWorld.__new__(collect.LearnWorld)
    w._capture = dict(out_dir=tmp, aux_views=aux_views, hd_twin=hd_twin,
                      aux_seed=7, envelope=ENV)
    w._sensors = {}
    w._twin = 11 if hd_twin else None
    w._aux_sensor = 22 if aux_views else None
    w.tick, w.W, w.H, w.f, w.far = 5, 640, 480, 441.0, 0.3
    w.image = np.full((w.H, w.W, 3), 9, np.uint8)
    w.depth = np.full((w.H, w.W), 0.1, float)
    w.camera = lambda: (np.eye(3), np.array([0., 0., 0.1]))
    w.endo = {'camera': {'resolution': [640, 480], 'view_angle_deg': 70.0,
                         'near_m': 0.002, 'far_m': 0.3}}
    w.cutter, w.view, w.ctrl = 'psm2', 'ecm', 99
    w.by = {'psm2_tcp': 1, 'psm2_dvrk_yaw_frame': 2}
    w.index = {'psm2': 2, 'ecm': 4}
    w.markers = [10, 11, 12]
    w.grasped_points = {'psm1': np.array([0.1, 0.2, 0.3])}
    w.sheet_origin = np.array([0., 0., 0.])
    w.events = {}
    w.arm_cfg = {'ecm': {'rcm_world': [0., 0., 0.]}}
    w.target_of = lambda arm: [0.0, 0.0, 0.08]
    w.timing = {}
    w.truth = FakeTruth()

    stats = [{'arm': a, 'failures': 0, 'max_drift_m': 0.0,
              'grip': 0.5, 'target': [0.0, 0.0, 0.08]} for a in ('psm1', 'psm2', 'psm3', 'ecm')]

    def script_call(name, ctrl, *a):
        if name == 'getRCMStats':
            return stats
        if name == 'tissueConstraints':
            return [0, 1, 1, 1, 2, 1]
        raise AssertionError(f'unexpected controller call {name}')

    w.sim = Mock()
    w.sim.callScriptFunction = Mock(side_effect=script_call)
    w.sim.getObjectMatrix = Mock(return_value=np.eye(4)[:3].reshape(-1).tolist())
    w.sim.getObjectPosition = Mock(return_value=[0.0, 0.0, 0.0])
    w.sim.setObjectMatrix = Mock()
    w.sim.handleVisionSensor = Mock()
    w.sim.getVisionSensorImg = Mock(side_effect=lambda h: (bytes([7]) * (w.W * w.H * 3)
                                                           if h == w._aux_sensor else
                                                           bytes([5]) * (1280 * 960 * 3),
                                                           [w.W, w.H] if h == w._aux_sensor
                                                           else [1280, 960]))
    w.sim.getVisionSensorDepth = Mock(side_effect=lambda h, f: (
        [0.11] * (w.W * w.H) if h == w._aux_sensor else [0.12] * (1280 * 960), None))
    w.sim.unpackFloatTable = Mock(side_effect=lambda d: d)
    return w


class SamplerTests(unittest.TestCase):
    def test_seed_formula_and_params_frozen_for_indices_0_and_1(self):
        p0 = collect.episode_params('dev', 0, PROTO)
        self.assertEqual(p0['seed'], 16929790619979785480)
        self.assertEqual(p0['advance_m'], 0.003)
        self.assertEqual(p0['target_length_m'], 0.01)
        self.assertEqual(p0['scenario'], 'occlusion_transient')
        self.assertAlmostEqual(p0['yaw_deg'], 0.4067836819426658)
        self.assertAlmostEqual(p0['pitch_deg'], 2.083328217507077)
        self.assertAlmostEqual(p0['insertion_m'], 0.08016791151138158)
        self.assertEqual(p0['aux_seed'], 3309049042068268760)
        p1 = collect.episode_params('dev', 1, PROTO)
        self.assertEqual(p1['seed'], 3796899042450213804)
        self.assertEqual(p1['advance_m'], 0.003)
        self.assertEqual(p1['target_length_m'], 0.007)
        self.assertEqual(p1['scenario'], 'nominal')
        self.assertAlmostEqual(p1['yaw_deg'], 1.9347444874969124)
        self.assertAlmostEqual(p1['pitch_deg'], 0.13254932767755934)
        self.assertAlmostEqual(p1['insertion_m'], 0.08294240975450773)
        self.assertEqual(p1['aux_seed'], 4469568567856707651)
        # Same index re-drawn must be bit-identical.
        self.assertEqual(collect.episode_params('dev', 0, PROTO), p0)

    def test_weights_are_normalised_before_choice(self):
        p = collect._norm([1, 1, 1, 2, 3, 3, 3])
        self.assertAlmostEqual(float(p.sum()), 1.0)
        p = collect._norm([4, 2, 1, 1])
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertTrue((collect._norm([2, 2]) == [0.5, 0.5]).all())

    def test_all_sampled_values_stay_inside_registered_choices_and_envelope(self):
        adv = set(RAND['advance_m']['choices'])
        tgt = set(RAND['target_length_m']['choices'])
        scen = set(RAND['scenario']['choices'])
        for index in range(64):
            p = collect.episode_params('dev', index, PROTO)
            self.assertIn(p['advance_m'], adv, index)
            self.assertIn(p['target_length_m'], tgt, index)
            self.assertIn(p['scenario'], scen, index)
            self.assertGreaterEqual(p['yaw_deg'], ENV['yaw'][0])
            self.assertLessEqual(p['yaw_deg'], ENV['yaw'][1])
            self.assertGreaterEqual(p['pitch_deg'], ENV['pitch'][0])
            self.assertLessEqual(p['pitch_deg'], ENV['pitch'][1])
            self.assertGreaterEqual(p['insertion_m'], ENV['insertion'][0])
            self.assertLessEqual(p['insertion_m'], ENV['insertion'][1])
            self.assertGreaterEqual(p['aux_seed'], 0)

    def test_aux_absolute_targets_inside_envelope_and_deterministic(self):
        for seed in (0, 7, 2**62):
            for tick in (1, 5, 17):
                t = collect.aux_targets(seed, tick, 4, ENV)
                self.assertEqual(t.shape, (4, 3))
                self.assertTrue((t[:, 0] >= ENV['yaw'][0]).all()
                                and (t[:, 0] <= ENV['yaw'][1]).all())
                self.assertTrue((t[:, 1] >= ENV['pitch'][0]).all()
                                and (t[:, 1] <= ENV['pitch'][1]).all())
                self.assertTrue((t[:, 2] >= ENV['insertion'][0]).all()
                                and (t[:, 2] <= ENV['insertion'][1]).all())
        self.assertTrue(np.array_equal(collect.aux_targets(7, 5, 4, ENV),
                                       collect.aux_targets(7, 5, 4, ENV)))
        self.assertFalse(np.array_equal(collect.aux_targets(7, 5, 4, ENV),
                                        collect.aux_targets(7, 6, 4, ENV)))

    def test_aux_pose_zero_delta_is_identity_and_rotation_is_valid(self):
        R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        t = np.array([0.01, -0.02, 0.3])
        rcm = np.array([0., 0., 0.25])
        R2, t2 = collect.aux_pose(R, t, rcm, [0., 0., 0.])
        np.testing.assert_allclose(R2, R, atol=1e-12)
        np.testing.assert_allclose(t2, t, atol=1e-12)
        R3, t3 = collect.aux_pose(R, t, rcm, [3., -2., 0.005])
        np.testing.assert_allclose(R3 @ R3.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R3)), 1.0)
        self.assertTrue(np.isfinite(t3).all())


class CliTests(unittest.TestCase):
    def test_holdout_split_is_refused(self):
        argv = sys.argv
        try:
            sys.argv = ['collect.py', '--split', 'holdout']
            with self.assertRaises(SystemExit) as ctx:
                collect.main()
            self.assertIn('sealed', str(ctx.exception))
        finally:
            sys.argv = argv


class FrameContractTests(unittest.TestCase):
    def test_atomic_npz_write_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as d:
            collect.atomic_npz(Path(d) / 'tick_001', dict(rgb=np.zeros((2, 2, 3), np.uint8)))
            self.assertTrue((Path(d) / 'tick_001.npz').exists())
            self.assertEqual(list(Path(d).glob('*.tmp.npz')), [])

    def test_input_label_key_separation_and_aux_hd_have_no_labels(self):
        with tempfile.TemporaryDirectory() as d:
            w = fake_world(d, aux_views=2, hd_twin=True)
            w._perception_tick()
            names = {f.name for f in Path(d).iterdir()}
            self.assertEqual(names, {'tick_005.npz', 'hd_005.npz',
                                     'aux_005_0.npz', 'aux_005_1.npz'})
            with np.load(Path(d) / 'tick_005.npz') as main:
                inputs = {k for k in main.files if not k.startswith('label_')}
                self.assertEqual(inputs, {'rgb', 'depth', 'cam_R', 'cam_t', 'intrinsics', 'tick'})
                labels = {k for k in main.files if k.startswith('label_')}
                self.assertIn('label_cut_length', labels)
                self.assertIn('label_tissue_alive', labels)
                self.assertIn('label_nodes', labels)
                self.assertIn('label_tcp_M', labels)
                self.assertIn('label_grasped_pts', labels)
                self.assertEqual(main['rgb'].shape, (480, 640, 3))
            with np.load(Path(d) / 'hd_005.npz') as hd:
                self.assertEqual(set(hd.files), {'rgb', 'depth', 'cam_R', 'cam_t', 'intrinsics'})
                self.assertEqual(hd['rgb'].shape, (960, 1280, 3))
            with np.load(Path(d) / 'aux_005_0.npz') as aux:
                self.assertEqual(set(aux.files),
                                 {'rgb', 'depth', 'cam_R', 'cam_t', 'intrinsics', 'delta'})
                self.assertEqual(aux['delta'].shape, (3,))
            self.assertEqual(w.truth.reads_by_purpose.get('dataset_label'), 1)


if __name__ == '__main__':
    unittest.main()
