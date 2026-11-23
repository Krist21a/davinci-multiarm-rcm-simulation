import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
sys.path.insert(0, str(ROOT / 'scripts' / 'task'))
from report_live_endpoint import check_identity, summarize
from run_active_look import SPEC
from run_same_bite import register


class ReportIdentityTest(unittest.TestCase):
    def test_policy_and_case_substitutions(self):
        with TemporaryDirectory() as tmp:
            folders = [Path(tmp) / name for name in ('same_bite', 'active_look')]
            hashes = []
            for folder in folders:
                register(folder, folder.name, () if folder == folders[0] else (SPEC,))
                hashes.append(hashlib.sha256((folder / 'registration.json').read_bytes()).hexdigest())
            self.assertNotEqual(*hashes)
            for folder, wrong_case in [(folders[1], True), (folders[0], False)]:
                registration = json.loads((folder / 'registration.json').read_bytes())
                # Supply only an impostor for the first expected case. Refusal must
                # precede reading observations or checking the other four files.
                case = next(iter(registration['protocol']['cases']))
                source = 'depth_noise_0p5' if wrong_case else case
                scenario, noise, seed = registration['protocol']['cases'][source]
                report = dict(case=source, scenario=scenario, policy='active_look',
                              registration_sha256=hashes[1], run_id=uuid4().hex,
                              config=dict(cut=dict(control_source='vision'),
                                          perception=dict(depth_noise_m=noise, seed=seed)))
                (folder / (case + '.json')).write_text(json.dumps(report), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'identity'):
                    summarize(folder)

    def test_duplicate_id_noise_seed_and_scenario(self):
        r = dict(case='noise', scenario='nominal', policy='same_bite', registration_sha256='a',
                 run_id=uuid4().hex, config=dict(cut=dict(control_source='vision'),
                 perception=dict(depth_noise_m=.0005, seed=1)))
        ids = set()
        def check(recipe):
            check_identity(r, 'noise', recipe, 'same_bite', 'a', ids)
        for recipe in [('relax', .0005, 1), ('nominal', 0, 1), ('nominal', .0005, 0)]:
            with self.assertRaisesRegex(ValueError, 'identity'):
                check(recipe)
        check(('nominal', .0005, 1))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            check(('nominal', .0005, 1))
