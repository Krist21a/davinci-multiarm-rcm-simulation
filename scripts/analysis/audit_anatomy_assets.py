"""Read-only BodyParts3D source audit (stdlib only; no scene/ROI generation).

Run from any working directory:
    python -B scripts/analysis/audit_anatomy_assets.py --asset-dir <bodyparts3d_v4>

Exit 0 means source integrity passed, NOT that P1 or clinical safety passed.
Exit 1 means an audit/input failure; exit 2 is argparse usage failure.
The deterministic report always keeps benchmark status blocked.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'config' / 'task' / 'fascia_benchmark.json'
OUTPUT = REPO / 'validation' / 'safety' / 'anatomy_assets.json'
PARTS = 'isa_parts_list_e.txt'
ELEMENTS = 'isa_element_parts.txt'
ARCHIVE = 'isa_BP3D_4.0_obj_99.zip'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', value.casefold()).split())


def table_rows(path, header):
    """Do not silently accept a swapped column, malformed row or wrong encoding."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.reader(stream, delimiter='\t')
        if next(rows, None) != header:
            raise ValueError(f'{Path(path).name}: unexpected TSV header')
        for line_number, row in enumerate(rows, 2):
            if not row or not any(cell.strip() for cell in row):
                continue
            row = [cell.strip() for cell in row]
            if len(row) != len(header) or not all(row):
                raise ValueError(f'{Path(path).name}:{line_number}: malformed TSV row')
            if not re.fullmatch(r'FMA\d+', row[0]):
                raise ValueError(f'{Path(path).name}:{line_number}: invalid FMA ID')
            yield row


def read_catalogue(asset_dir):
    catalogue = {}
    part_rows = element_rows = 0
    for fma, representation, name in table_rows(
            asset_dir / PARTS, ['concept id', 'representation id', 'en']):
        part_rows += 1
        if fma in catalogue:
            raise ValueError(f'{PARTS}: duplicate concept {fma}')
        catalogue[fma] = dict(name=name, representation_id=representation,
                              element_names=set(), elements=set(), element_rows=0)
    for fma, name, element in table_rows(
            asset_dir / ELEMENTS, ['concept id', 'name', 'element file id']):
        element_rows += 1
        if not re.fullmatch(r'FJ\d+M?', element):
            raise ValueError(f'{ELEMENTS}: invalid element ID {element}')
        if fma not in catalogue:
            raise ValueError(f'{ELEMENTS}: {fma} absent from {PARTS}')
        item = catalogue[fma]
        item['element_names'].add(name)
        item['elements'].add(element + '.obj')
        item['element_rows'] += 1
    if not catalogue or not element_rows:
        raise ValueError('Empty parts/element catalogue')
    for item in catalogue.values():
        item['element_names'] = sorted(item['element_names'])
        item['elements'] = sorted(item['elements'])
        item['element_count'] = len(item['elements'])
        item['duplicate_element_rows'] = item['element_rows'] - item['element_count']
        item['name_match'] = bool(item['element_names']) and all(
            normalize_name(n) == normalize_name(item['name']) for n in item['element_names'])
    return catalogue, dict(parts_rows=part_rows, element_rows=element_rows)


def inspect_obj(stream):
    """Hash raw uncompressed bytes and check nonempty v/f + basic vertex indices.

    Not a watertightness, self-intersection, manifold, material or clinical check.
    Positive face indices may refer forward; negative indices are relative to the
    vertices already seen. vt/vn are deliberately outside this minimal audit.
    """
    digest = hashlib.sha256()
    vertices = faces = size = 0
    max_positive_index = 0
    issues = set()
    for line in stream:
        digest.update(line)
        size += len(line)
        tokens = line.split(b'#', 1)[0].split()
        if not tokens:
            continue
        if tokens[0] == b'v':
            vertices += 1
            try:
                if len(tokens) < 4 or not all(math.isfinite(float(x)) for x in tokens[1:4]):
                    issues.add('invalid_vertex')
            except (ValueError, OverflowError):
                issues.add('invalid_vertex')
        elif tokens[0] == b'f':
            faces += 1
            if len(tokens) < 4:
                issues.add('face_has_fewer_than_three_vertices')
            for token in tokens[1:]:
                try:
                    index = int(token.split(b'/', 1)[0])
                    if index == 0 or (index < 0 and -index > vertices):
                        issues.add('invalid_vertex_index')
                    max_positive_index = max(max_positive_index, index)
                except ValueError:
                    issues.add('invalid_vertex_index')
    if not vertices:
        issues.add('empty_vertices')
    if not faces:
        issues.add('empty_faces')
    if max_positive_index > vertices:
        issues.add('vertex_index_out_of_range')
    return dict(bytes=size, sha256=digest.hexdigest(), vertices=vertices, faces=faces,
                nonempty_vf=vertices > 0 and faces > 0,
                status='invalid' if issues else 'ok', issues=sorted(issues))


def inspect_archive(path, referenced):
    """One record per OBJ basename; ambiguous ZIP members are never picked silently."""
    objects = {}
    with zipfile.ZipFile(path) as archive:
        members = defaultdict(list)
        for info in archive.infolist():
            if not info.is_dir() and info.filename.lower().endswith('.obj'):
                members[PurePosixPath(info.filename.replace('\\', '/')).name].append(info)
        for name in sorted(set(members) | referenced):
            entries = sorted(members.get(name, []), key=lambda info: info.filename)
            if not entries:
                objects[name] = dict(status='missing', members=[], issues=['missing_zip_member'])
            elif len(entries) > 1:
                objects[name] = dict(status='ambiguous', members=[i.filename for i in entries],
                                     issues=['duplicate_obj_basename'])
            else:
                info = entries[0]
                try:
                    with archive.open(info) as stream:
                        record = inspect_obj(stream)
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError) as exc:
                    record = dict(status='unreadable', issues=[type(exc).__name__])
                record['members'] = [info.filename]
                objects[name] = record
    return objects


def match_structures(specs, catalogue, objects):
    """ID + normalized exact name for known structures; bounded alias search otherwise.

    Alias hits for unpinned fascia/NVB are candidates only, requiring human review.
    Absence means no named asset in these two tables, not anatomical nonexistence.
    """
    results = []
    for spec in specs:
        aliases = sorted({normalize_name(n) for n in spec['names']})
        fma = spec['fma']
        exact_ids, candidate_ids = [], []
        for concept, item in sorted(catalogue.items()):
            names = [normalize_name(n) for n in [item['name']] + item['element_names']]
            if any(n in aliases for n in names):
                exact_ids.append(concept)
            if any(f' {a} ' in f' {n} ' for a in aliases for n in names):
                candidate_ids.append(concept)
        selected_ids = [fma] if fma in catalogue else ([] if fma else candidate_ids)
        elements = sorted({obj for concept in selected_ids for obj in catalogue[concept]['elements']})
        record = dict(key=spec['key'], name=spec['name'], requested_fma=fma,
                      required=spec['required'], searched_names=spec['names'],
                      exact_name_fma_ids=exact_ids, candidate_fma_ids=candidate_ids,
                      matched_fma_ids=selected_ids,
                      matched_names={c: catalogue[c]['name'] for c in selected_ids},
                      elements=elements, element_count=len(elements),
                      missing_or_invalid_objects=[e for e in elements if objects[e]['status'] != 'ok'])
        if fma:
            record['id_match'] = fma in catalogue
            record['name_match'] = fma in exact_ids
            expected = sorted(e + '.obj' for e in spec.get('expected_elements', []))
            record['expected_elements'] = expected
            record['expected_elements_match'] = elements == expected if expected else None
            if fma not in catalogue:
                status = 'missing_id'
            elif fma not in exact_ids or not catalogue[fma]['name_match']:
                status = 'name_mismatch'
            elif expected and elements != expected:
                status = 'element_mismatch'
            elif not elements or record['missing_or_invalid_objects']:
                status = 'invalid_assets'
            else:
                status = 'available'
        else:
            status = 'review_required' if candidate_ids else 'missing_named_asset'
            record['absence_scope'] = 'Named entries/aliases in the supplied tables only; not clinical anatomy.'
            record['expected_missing'] = spec.get('expected_missing', False)
        record['status'] = status
        results.append(record)
    return results


def audit(asset_dir, config_path=CONFIG):
    asset_dir, config_path = Path(asset_dir), Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    sources = {name: dict(bytes=(asset_dir / name).stat().st_size,
                          sha256=sha256_file(asset_dir / name))
               for name in (PARTS, ELEMENTS, ARCHIVE)}
    catalogue, counts = read_catalogue(asset_dir)
    referenced = {obj for item in catalogue.values() for obj in item['elements']}
    objects = inspect_archive(asset_dir / ARCHIVE, referenced)
    structures = match_structures(config['structures'], catalogue, objects)
    errors = []
    for fma, item in sorted(catalogue.items()):
        if not item['elements']:
            errors.append(f'{fma}: no element mapping')
        elif not item['name_match']:
            errors.append(f'{fma}: name mismatch between tables')
    errors.extend(f'{obj}: {item["status"]}' for obj, item in objects.items() if item['status'] != 'ok')
    errors.extend(f'{s["key"]}: {s["status"]}' for s in structures
                  if s['required'] and s['status'] != 'available')
    good = [item for item in objects.values() if item['status'] == 'ok']
    gates = [dict(gate) for gate in config['gates']]
    for gate in gates:
        if gate['id'] == 'source_asset_integrity':
            gate['status'] = 'passed' if not errors else 'failed'
    return dict(
        schema_version=1, status='blocked', audit_status='passed' if not errors else 'failed',
        scope='Source asset integrity only; not ROI/registration, clinical safety or P1/P4 acceptance.',
        reproducibility=dict(algorithm='SHA256', obj_hash_bytes='raw uncompressed member bytes',
                             ordering='sorted IDs/OBJ names; no timestamps or absolute asset paths',
                             deduplication='OBJ basename, not FMA, not byte content',
                             script_sha256=sha256_file(Path(__file__)),
                             config_sha256=sha256_file(config_path)),
        sources=sources,
        summary=dict(**counts, concepts=len(catalogue), unique_referenced_objs=len(referenced),
                     unique_archive_objs=sum(bool(o['members']) for o in objects.values()),
                     audited_unique_objs=len(good),
                     duplicate_element_rows=sum(c['duplicate_element_rows'] for c in catalogue.values()),
                     unique_obj_vertices=sum(o['vertices'] for o in good),
                     unique_obj_faces=sum(o['faces'] for o in good),
                     required_structures_available=sum(s['required'] and s['status'] == 'available'
                                                       for s in structures)),
        concepts=catalogue, objects=objects, structures=structures,
        missing_named_structures=[s['key'] for s in structures if s['status'] == 'missing_named_asset'],
        unreferenced_archive_objects=sorted(set(objects) - referenced),
        errors=errors, gates=gates,
        limitations=[
            'Basic nonempty v/f, finite xyz and vertex-index checks only; no topology validation.',
            'Source hashes identify bytes; they do not authenticate an official release or license.',
            'Fascia/NVB alias matching is not anatomical annotation; candidate hits need review.',
            'No world coordinates, scene registration, deformation or material calibration generated.',
            'Synthetic proxy; preset exposure with bladder retained, not removed.',
            'NVB model risk region is not a clinically safe envelope.',
        ])


def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, required=True,
                        help='Directory with the two isa_*.txt tables and isa_BP3D_4.0_obj_99.zip')
    args = parser.parse_args(argv)
    try:
        report = audit(args.asset_dir)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as exc:
        # Replace a stale success report even when inputs cannot be audited.
        report = dict(schema_version=1, status='blocked', audit_status='failed',
                      errors=[f'{type(exc).__name__}: {exc}'],
                      scope='Audit could not complete; no P1/P4 or safety acceptance.')
    try:
        write_report(report, OUTPUT)
    except OSError as exc:
        print(f'Cannot write audit report: {exc}', file=sys.stderr)
        return 1
    summary = report.get('summary', {})
    print(f"audit={report['audit_status']}; benchmark=blocked; "
          f"OBJ={summary.get('audited_unique_objs', 0)}; errors={len(report['errors'])}")
    print(f"missing named structures: {', '.join(report.get('missing_named_structures', [])) or 'see report'}")
    print(f'wrote {OUTPUT}')
    return 0 if report['audit_status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())