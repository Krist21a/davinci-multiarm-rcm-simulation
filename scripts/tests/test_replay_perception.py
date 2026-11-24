"""M5a evidence validation without reading registered holdout frames."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vision'))
import replay_perception as rp


class ReplayEvidenceTests(unittest.TestCase):
    def test_probe_evidence_fills_only_occlusion_gates_with_minimum_coverage(self):
        base = rp.metrics([], 0.002)
        results = {'orig': {'none': {'metrics': base}}}
        sample = dict(base, occluded_front_ticks=3, occluded_tip_ticks=2, front_vbw_on_occluded=1)
        rp.supplement_occlusion(results, 'orig', sample, 3)
        self.assertEqual(base['occlusion_gate_metrics'],
                         {'front_vbw_on_occluded': 1, 'tip_vbw_on_occluded': None})
        self.assertEqual(base['occluded_front_ticks'], 0)
        self.assertIsNone(base['front_availability'])

    def test_zero_event_gates_need_samples_and_all_datasets(self):
        gate = dict(dataset='all', degradation='none',
                    metric='pre_sheet_false_positives', op='==', value=0)
        pre_sheet = dict(sheet=False, tip_valid=False, cut_valid=False, front_valid=False)
        results = {'a': {'none': {'metrics': rp.metrics([pre_sheet], 0.002)}}}
        self.assertEqual(rp.check(gate, results, ['a']), (0, 'PASSED'))
        self.assertEqual(rp.check(gate, results, ['a', 'b']), (None, 'FAILED (no evidence)'))
        results['b'] = {'none': {'metrics': rp.metrics([], 0.002)}}
        self.assertEqual(rp.check(gate, results, ['a', 'b']), (None, 'FAILED (no evidence)'))
        gate.update(dataset='a', metric='front_vbw_on_occluded')
        self.assertEqual(rp.check(gate, results, ['a']), (None, 'FAILED (no evidence)'))

    def test_metadata_identity_and_capture_tail_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(rp, 'FRAMES', Path(tmp)):
            d = Path(tmp) / 'test'
            d.mkdir()
            meta, capture = d / 'meta.json', d / 'capture.json'
            meta.write_text(json.dumps({'run_id': 'demo'}), encoding='utf-8')
            capture.write_text(json.dumps({'run_id': 'demo', 'frame_count': 2}), encoding='utf-8')
            np.savez(d / 'tick_001.npz', tick=1)
            with self.assertRaisesRegex(ValueError, 'frame count'):
                rp.load_dataset('test')
            np.savez(d / 'tick_002.npz', tick=2)
            before = rp.load_dataset('test')[2]
            meta.write_text(json.dumps({'run_id': 'demo', 'W': 3}), encoding='utf-8')
            after = rp.load_dataset('test')[2]
            self.assertNotEqual(before['meta_sha256'], after['meta_sha256'])
            self.assertEqual(before['sha256'], after['sha256'])

    def test_view_capture_rejects_return_to_home_with_sheet_present(self):
        spec = dict(scenario='nominal', initial_view=[1.5, -2.5])
        applied_R = rp.cp.rotvec(np.array([0.0, 0.01, 0.0]))
        target = [*spec['initial_view'], 0.1]
        capture = dict(initial_view_readback=dict(home_R=np.eye(3), applied_R=applied_R, target=target))
        frames = [{}, dict(nodes=[], camera_target=target, cam_R=applied_R)]
        meta = dict(spec, mode='full', align_mode='tool_pose', control_source='truth',
                    perception_cfg=dict(mode='shadow'), scene_sha256=hashlib.sha256(b'scene').hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / 'test.ttt'
            scene.write_bytes(b'scene')
            with patch.object(rp, 'SCENE', scene):
                rp.verify_initial_view(meta, capture, frames, spec)
                frames[1]['cam_R'] = np.eye(3)
                with self.assertRaisesRegex(ValueError, 'not maintained'):
                    rp.verify_initial_view(meta, capture, frames, spec)

    def test_default_view_probe_requires_registered_scene_and_unperturbed_capture(self):
        spec = dict(scenario='nominal', align_mode='shaft')
        meta = dict(spec, mode='full', control_source='truth',
                    perception_cfg=dict(mode='shadow'), scene_sha256=hashlib.sha256(b'scene').hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / 'test.ttt'
            scene.write_bytes(b'scene')
            with patch.object(rp, 'SCENE', scene):
                rp.verify_registered_capture(meta, spec)
                with self.assertRaisesRegex(ValueError, 'capture scene'):
                    rp.verify_registered_capture(dict(meta, scene_sha256='wrong'), spec)
                with self.assertRaisesRegex(ValueError, 'capture parameters'):
                    rp.verify_registered_capture(dict(meta, perception_cfg=dict(mode='shadow', depth_noise_m=0.0005)), spec)
