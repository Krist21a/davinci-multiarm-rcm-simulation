"""Correct the empty Lua tear-log serialization check, without rerunning M5."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    raw, sealed, original = ((args.folder / name).read_bytes()
                             for name in ('nominal.json', 'registration.json', 'decision.json'))
    run, reg, score = map(json.loads, (raw, sealed, original))
    if (reg['id'] != 'm5_level0_v1' or score['protocol'] != reg
            or score['nominal_sha256'] != sha(raw) or score['registration_sha256'] != sha(sealed)
            or score['trace'] != run['trace'] or score['final_observation'] != run['final_observation']):
        raise ValueError('Level 0 evidence does not match its original registration and decision')
    final = run['final_observation']
    tears = run['tissue']['torn_events']
    checks = dict(score['checks'])
    checks['done_clean'] = (
        run['success'] is True and run['control_source'] == 'truth' and run['final_state'] == 'DONE'
        and run['error'] is None and run['cleanup_error'] is None and not run['acceptance_issues']
        and final['torn'] == run['tissue']['torn'] == 0 and not final['faults']
        and type(tears) in (dict, list) and not tears
    )
    corrected = dict(score, checks=checks, passed=all(checks.values()), correction={
        'id': 'm5_level0_score_v1_1',
        'reason': 'CoppeliaSim serializes the empty Lua tear log as {}; torn_events is not a count. '
                  'Only done_clean is recalculated. Missing and nonempty logs do not pass.',
        'original_decision_sha256': sha(original),
        'scorer_sha256': sha(Path(__file__).read_bytes()),
        'no_new_simulation': True,
    })
    with (args.folder / 'decision_v1_1.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(corrected, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print('Level 0 (corrected score):', 'PASS' if corrected['passed'] else 'FAIL')
    return 0 if corrected['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
