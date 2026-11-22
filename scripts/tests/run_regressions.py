"""Run safety + original FSM regressions and save versioned evidence, without a simulator."""
import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=None,
                        help='versioned results path (default validation/safety/regressions.json); '
                             'also redirects the FSM coverage JSON to a sibling versioned name')
    args = parser.parse_args()
    if args.out is not None:
        # Opt-in versioned output. The coverage file is a sibling of the chosen report,
        # and every write target is refused BEFORE the suite runs: an existing result or
        # coverage path is never overwritten, and a report named regressions.json under
        # validation/task must not clobber the historical coverage file either.
        suffix = (args.out.stem[len('regressions'):] if args.out.stem.startswith('regressions')
                  else '_' + args.out.stem)
        coverage = args.out.resolve().parent / f'fsm_unit_coverage{suffix}.json'
        historical = (ROOT / 'validation' / 'task' / 'fsm_unit_coverage.json').resolve()
        if coverage.resolve() == historical:
            parser.error(f'--out {args.out} would overwrite the historical coverage file')
        for target in (args.out, coverage):
            if target.exists():
                parser.error(f'--out {args.out} would overwrite existing evidence: {target}')
        os.environ['FSM_UNIT_COVERAGE_JSON'] = str(coverage)
    loader = unittest.TestLoader()
    tests = loader.discover(str(ROOT / 'scripts' / 'tests'))
    sys.path.insert(0, str(ROOT / 'scripts' / 'task'))
    # This original test intentionally regenerates its FSM coverage JSON.
    tests.addTests(loader.loadTestsFromModule(importlib.import_module('test_cut_fsm')))
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    python_files = sorted((ROOT / 'scripts').rglob('*.py'))
    json_files = sorted((ROOT / 'config').rglob('*.json'))
    static_errors = []
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except Exception as exc:
            static_errors.append(f'{path}: {exc}')
    for path in json_files:
        try:
            json.loads(path.read_text(encoding='utf-8-sig'))
        except Exception as exc:
            static_errors.append(f'{path}: {exc}')
    sources = python_files + sorted((ROOT / 'src').rglob('*.lua')) + json_files
    packages = {}
    for name in ('numpy', 'scipy', 'pillow', 'trimesh', 'fast-simplification', 'coppeliasim-zmqremoteapi-client', 'pyzmq'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    evidence = dict(success=result.wasSuccessful() and not result.skipped and not static_errors,
                    tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                    skipped=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
                    python=platform.python_version(), packages=packages,
                    static_checks=dict(python_files=len(python_files), config_json_files=len(json_files), errors=static_errors),
                    source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    scope='Offline unit/mocked regressions and real Lua interpreter with sim stubs; live scene results are separate reports.')
    out = args.out or ROOT / 'validation' / 'safety' / 'regressions.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print('regressions:', evidence['success'], result.testsRun, 'tests;', out)
    return 0 if evidence['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())