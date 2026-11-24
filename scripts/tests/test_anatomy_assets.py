"""Offline stdlib tests. Tiny synthetic OBJ fixtures are NOT anatomical assets.

python -B -S -m unittest discover -s scripts/tests -p test_anatomy_assets.py -v
"""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'analysis'))
import audit_anatomy_assets as audit  # noqa: E402

CFG = json.loads(audit.CONFIG.read_text(encoding='utf-8'))
TRIANGLE = b'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'


class AnatomyAssets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'assets with spaces'
        self.root.mkdir()
        self.parts = []
        self.elements = []
        self.meshes = {}
        for s in CFG['structures']:
            if not s['fma']:
                continue
            self.parts.append([s['fma'], 'BP1', s['name']])
            for element in s['expected_elements']:
                self.elements.append([s['fma'], s['name'], element])
                self.meshes[element + '.obj'] = b'# synthetic test fixture\n' + TRIANGLE
        # Shared compound children must not double-count either OBJ.
        self.parts.append(['FMA45855', 'BP2', 'left pubococcygeus'])
        self.elements += [['FMA45855', 'left pubococcygeus', e]
                          for e in ('FJ1457M', 'FJ2545')]
        self.parts.append(['FMA22423', 'BP3', 'tensor fasciae latae'])
        self.elements.append(['FMA22423', 'tensor fasciae latae', 'FJ9999'])
        self.meshes['FJ9999.obj'] = TRIANGLE
        self.write_tables()
        self.write_zip()

    def write_tables(self):
        for name, header, rows in [
            (audit.PARTS, 'concept id\trepresentation id\ten', self.parts),
            (audit.ELEMENTS, 'concept id\tname\telement file id', self.elements),
        ]:
            (self.root / name).write_text(header + '\n' + '\n'.join('\t'.join(r) for r in rows) + '\n',
                                          encoding='utf-8', newline='\n')

    def write_zip(self, extra=()):
        with zipfile.ZipFile(self.root / audit.ARCHIVE, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(self.meshes.items()):
                info = zipfile.ZipInfo('nested/' + name, date_time=(2020, 1, 1, 0, 0, 0))
                archive.writestr(info, data)
            for name, data in extra:
                archive.writestr(name, data)

    def report(self):
        return audit.audit(self.root)

    def test_complete_compounds_and_deduplicated_counts(self):
        report = self.report()
        self.assertEqual(report['audit_status'], 'passed')
        self.assertEqual(report['status'], 'blocked')
        self.assertFalse(report['errors'])
        structures = {s['key']: s for s in report['structures']}
        self.assertEqual(structures['levator_ani']['element_count'], 9)
        self.assertEqual(structures['left_external_iliac_vein']['element_count'], 4)
        self.assertEqual(structures['tendinous_arch_levator_ani']['element_count'], 4)
        self.assertEqual(report['summary']['unique_referenced_objs'], len(self.meshes))
        self.assertEqual(report['summary']['unique_obj_vertices'], 3 * len(self.meshes))
        self.assertEqual(report['summary']['unique_obj_faces'], len(self.meshes))
        self.assertEqual(report['summary']['required_structures_available'], 14)
        self.assertGreater(len(self.elements), len(self.meshes))
        # Identical bytes of different OBJ IDs still denote different assets.
        self.assertEqual(len(report['objects']), len(self.meshes))

    def test_sha256_sources_config_script_and_uncompressed_obj(self):
        report = self.report()
        for name in (audit.PARTS, audit.ELEMENTS, audit.ARCHIVE):
            self.assertEqual(report['sources'][name]['sha256'],
                             hashlib.sha256((self.root / name).read_bytes()).hexdigest())
        self.assertEqual(report['reproducibility']['config_sha256'],
                         hashlib.sha256(audit.CONFIG.read_bytes()).hexdigest())
        self.assertEqual(report['reproducibility']['script_sha256'],
                         hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest())
        for name, data in self.meshes.items():
            self.assertEqual(report['objects'][name]['sha256'], hashlib.sha256(data).hexdigest())

    def test_repeat_report_is_byte_identical_and_sources_unchanged(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        output = Path(self.temp.name) / 'report.json'
        audit.write_report(self.report(), output)
        first = output.read_bytes()
        audit.write_report(self.report(), output)
        self.assertEqual(first, output.read_bytes())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_duplicate_element_rows_not_duplicate_meshes(self):
        self.elements += [self.elements[0]] * 3
        self.write_tables()
        report = self.report()
        self.assertEqual(report['audit_status'], 'passed')
        self.assertEqual(report['summary']['duplicate_element_rows'], 3)
        self.assertEqual(report['summary']['unique_obj_faces'], len(self.meshes))

    def test_missing_compound_element_in_zip_fails(self):
        del self.meshes['FJ3523.obj']
        self.write_zip()
        report = self.report()
        self.assertEqual(report['audit_status'], 'failed')
        self.assertEqual(report['objects']['FJ3523.obj']['status'], 'missing')
        s = next(s for s in report['structures'] if s['key'] == 'left_external_iliac_vein')
        self.assertEqual(s['element_count'], 4)
        self.assertEqual(s['missing_or_invalid_objects'], ['FJ3523.obj'])

    def test_truncated_compound_mapping_not_silently_accepted(self):
        self.elements = [r for r in self.elements if r[2] != 'FJ3524']
        self.write_tables()
        report = self.report()
        s = next(s for s in report['structures'] if s['key'] == 'left_external_iliac_vein')
        self.assertEqual(s['status'], 'element_mismatch')
        self.assertIn('FJ3524.obj', report['unreferenced_archive_objects'])

    def test_known_name_or_id_mismatch_fails_not_replaced_by_search(self):
        for row in self.parts:
            if row[0] == 'FMA19387':
                row[2] = 'left seminal vesicle'
        self.write_tables()
        report = self.report()
        self.assertEqual(report['audit_status'], 'failed')
        s = next(s for s in report['structures'] if s['key'] == 'right_seminal_vesicle')
        self.assertEqual(s['status'], 'name_mismatch')
        self.parts = [r for r in self.parts if r[0] != 'FMA19387']
        self.elements = [r for r in self.elements if r[0] != 'FMA19387']
        self.write_tables()
        report = self.report()
        s = next(s for s in report['structures'] if s['key'] == 'right_seminal_vesicle')
        self.assertEqual(s['status'], 'missing_id')

    def test_missing_fascia_nvb_is_explicit_and_not_generic_fascia(self):
        report = self.report()
        self.assertEqual(set(report['missing_named_structures']),
                         {'endopelvic_fascia', 'prostatic_fascia', 'denonvilliers_fascia', 'nvb'})
        self.assertEqual(report['audit_status'], 'passed')
        gates = {g['id']: g['status'] for g in report['gates']}
        self.assertEqual(gates['source_asset_integrity'], 'passed')
        self.assertEqual(gates['anatomical_roi_and_registration'], 'blocked')

    def test_alias_hits_require_review_not_anatomical_validation(self):
        self.parts += [['FMA999001', 'BP4', "Denonvilliers' fascia"],
                       ['FMA999002', 'BP5', 'right neurovascular bundle of prostate']]
        self.elements += [['FMA999001', "Denonvilliers' fascia", 'FJ9999'],
                          ['FMA999002', 'right neurovascular bundle of prostate', 'FJ9999']]
        self.write_tables()
        structures = {s['key']: s for s in self.report()['structures']}
        self.assertEqual(structures['denonvilliers_fascia']['status'], 'review_required')
        self.assertEqual(structures['nvb']['status'], 'review_required')
        self.assertEqual(structures['nvb']['matched_fma_ids'], ['FMA999002'])
        self.assertEqual(structures['prostatic_fascia']['status'], 'missing_named_asset')

    def test_ambiguous_obj_basename_fails_even_if_bytes_equal(self):
        self.write_zip(extra=[('other/FJ3139.obj', self.meshes['FJ3139.obj'])])
        report = self.report()
        self.assertEqual(report['audit_status'], 'failed')
        self.assertEqual(report['objects']['FJ3139.obj']['status'], 'ambiguous')
        self.assertEqual(len(report['objects']['FJ3139.obj']['members']), 2)

    def test_empty_or_invalid_obj_fails_archive_audit(self):
        self.meshes['FJ3139.obj'] = b'# comment only, not a mesh\n'
        self.write_zip()
        report = self.report()
        self.assertEqual(report['audit_status'], 'failed')
        obj = report['objects']['FJ3139.obj']
        self.assertFalse(obj['nonempty_vf'])
        self.assertEqual(obj['issues'], ['empty_faces', 'empty_vertices'])

    def test_obj_whitespace_slash_negative_indices_and_comments(self):
        data = b'  v\t0 0 0\r\nv 1 0 0\nv 0 1 0\nf -3/1/2 -2//2 -1 # triangle\n'
        obj = audit.inspect_obj(io.BytesIO(data))
        self.assertEqual(obj['status'], 'ok')
        self.assertEqual((obj['vertices'], obj['faces']), (3, 1))
        self.assertEqual(obj['sha256'], hashlib.sha256(data).hexdigest())

    def test_obj_rejects_bad_numeric_and_face_data(self):
        for data in [b'', b'v 0 0 0\n', b'f 1 2 3\n', TRIANGLE.replace(b'v 0 0 0', b'v nan 0 0'),
                     TRIANGLE.replace(b'v 0 0 0', b'v x 0 0'),
                     TRIANGLE.replace(b'f 1 2 3', b'f 1 2'),
                     TRIANGLE.replace(b'f 1 2 3', b'f 0 2 3'),
                     TRIANGLE.replace(b'f 1 2 3', b'f 1 2 4'),
                     TRIANGLE.replace(b'f 1 2 3', b'f -4 -2 -1'),
                     TRIANGLE.replace(b'f 1 2 3', b'f x 2 3')]:
            with self.subTest(data=data):
                self.assertEqual(audit.inspect_obj(io.BytesIO(data))['status'], 'invalid')

    def test_malformed_header_and_row_fail(self):
        path = self.root / audit.PARTS
        original = path.read_text(encoding='utf-8')
        for text in [original.replace('concept id', 'wrong header', 1),
                     original + 'FMA999\tBP4\n', original + 'notFMA\tBP4\tname\n']:
            with self.subTest(text=text[-40:]):
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.report()

    def test_bom_and_crlf_tables_supported(self):
        for name in (audit.PARTS, audit.ELEMENTS):
            path = self.root / name
            text = path.read_text(encoding='utf-8')
            path.write_bytes(b'\xef\xbb\xbf' + text.replace('\n', '\r\n').encode('utf-8'))
        self.assertEqual(self.report()['audit_status'], 'passed')

    def test_duplicate_fma_and_orphan_mapping_fail(self):
        self.parts.append(self.parts[0])
        self.write_tables()
        with self.assertRaisesRegex(ValueError, 'duplicate concept'):
            self.report()
        self.parts.pop()
        self.elements.append(['FMA999999', 'orphan', 'FJ9999'])
        self.write_tables()
        with self.assertRaisesRegex(ValueError, 'absent from'):
            self.report()

    def test_cli_success_and_failure_replace_stale_report(self):
        output = Path(self.temp.name) / 'validation' / 'anatomy_assets.json'
        with mock.patch.object(audit, 'OUTPUT', output), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(['--asset-dir', str(self.root)]), 0)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['audit_status'], 'passed')
            (self.root / audit.ARCHIVE).write_bytes(b'not a ZIP')
            self.assertEqual(audit.main(['--asset-dir', str(self.root)]), 1)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['audit_status'], 'failed')
            self.assertEqual(audit.main(['--asset-dir', str(self.root / 'absent')]), 1)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['status'], 'blocked')

    def test_cli_imports_without_site_packages_from_other_cwd(self):
        result = subprocess.run([sys.executable, '-B', '-S', audit.__file__, '--help'],
                                cwd=self.temp.name, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--asset-dir', result.stdout)


class BenchmarkSpecification(unittest.TestCase):
    def test_no_invented_roi_material_or_clinical_safety(self):
        self.assertEqual(CFG['status'], 'blocked')
        self.assertEqual(CFG['benchmark_kind'], 'synthetic_proxy')
        self.assertFalse(CFG['roi']['validated'])
        self.assertIsNone(CFG['roi']['world_coordinates'])
        self.assertIsNone(CFG['roi']['source_to_world_transform'])
        self.assertTrue(CFG['exposure']['bladder_present'])
        self.assertFalse(CFG['exposure']['remove_bladder'])
        self.assertEqual(CFG['exposure']['initial_condition'], 'preset_already_exposed')
        self.assertEqual(CFG['nvb']['representation'], 'model_risk_region')
        self.assertFalse(CFG['nvb']['clinical_safety_validated'])
        self.assertIsNone(CFG['nvb']['clinical_safe_distance_m'])
        self.assertIsNone(CFG['nvb']['world_coordinates'])
        self.assertEqual(CFG['material']['status'], 'uncalibrated')
        self.assertIsNone(CFG['material']['parameters'])
        self.assertIsNone(CFG['material']['experimental_evidence'])
        self.assertEqual(CFG['proposed_comparisons']['status'], 'not_run')

    def test_pinned_ids_and_compounds_independent_of_fixture(self):
        specs = {s['fma']: s for s in CFG['structures'] if s['fma']}
        expected = {'FMA19387': ['FJ3143'], 'FMA19388': ['FJ3137'],
                    'FMA19667': ['FJ3148'], 'FMA9600': ['FJ3139'], 'FMA15900': ['FJ3149'],
                    'FMA18806': ['FJ3567'], 'FMA18807': ['FJ3466'], 'FMA18885': ['FJ3568'],
                    'FMA18886': ['FJ3484', 'FJ3522', 'FJ3523', 'FJ3524'],
                    'FMA19089': ['FJ1453M', 'FJ1457M', 'FJ1458M', 'FJ2544', 'FJ2545',
                                 'FJ2546', 'FJ2549', 'FJ2550', 'FJ2551']}
        for fma, elements in expected.items():
            self.assertEqual(specs[fma]['expected_elements'], elements)


if __name__ == '__main__':
    unittest.main()