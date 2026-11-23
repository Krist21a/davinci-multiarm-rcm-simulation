"""Staged guide development experiment; archived rail and seam protocols remain immutable."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import capture_seam_rail as acquisition
from capture_surface_seam import ROOT, add_arguments, sha256, write_new
from score_seam_rail import sensor
import score_surface_seam as parent
from seam_slide_world import SeamSlideWorld

SPEC = ROOT/'config/task/seam_slide_v1.json'
OUT = ROOT/'validation/task/seam_slide_v1'
SOURCES = (*acquisition.SOURCES, 'scripts/analysis/seam_slide_pilot.py', 'scripts/task/seam_slide_world.py',
           'src/tissue/seam_slide.lua', 'src/tissue/seam_slide_create.lua')


def evaluate(args):
    folder, pre = args.out_root, args.stage == 'preflight'
    spec = json.loads(SPEC.read_bytes())
    hashes = {s: sha256(ROOT/s) for s in SOURCES}
    if hashes['scripts/vision/ruler_readout.py'] != spec['reader_sha256']:
        raise ValueError('reader drift')
    blind = folder/(args.stage+'_predictions.json')
    dest = folder/('selection.json' if pre else 'decision.json')
    if blind.exists() or dest.exists():
        raise FileExistsError('output already locked')
    entries = spec['development_runs'][:1] if pre else spec['development_runs']
    choice = None
    if not pre:
        selection = json.loads((folder/'selection.json').read_bytes())
        if not selection['passed'] or selection['source_sha256'] != hashes or selection['spec_sha256'] != sha256(SPEC):
            raise ValueError('unchecked slide selection')
        choice = selection['selected']['scale']
    preds = {1: {}, 4: {}}
    paths = [p for e in entries for kind in ('sensor', 'high') for p in (folder/'slide'/e['id']/kind).glob('*.npz')]
    for p in sorted(paths, key=lambda p: p.stem):
        scale = 1 if p.parent.name == 'sensor' else 4
        if p.stem in preds[scale]:
            raise ValueError('duplicate sensor token')
        preds[scale][p.stem] = sensor(p, scale)
    write_new(blind, dict(spec_sha256=sha256(SPEC), source_sha256=hashes, predictions=preds))
    results, seen, high, allowed = [], set(), set(), []
    for e in entries:
        r = json.loads((folder/'slide'/e['id']/'oracle.json').read_bytes())
        if (r['rail'] != 'slide' or (r.get('final_state') == 'DONE' and r['pins'] !=
                [24*j+i for j in range(13) for i in (*range(6, 11), *range(13, 18))])):
            raise ValueError('guide readback differs')
        with patch.object(parent, 'OUTPUT', folder/'slide'), patch.object(parent, 'PLAN', SPEC):
            checked = parent.run_score(e, preds[1], seen, spec, hashes)
        for f in r['frames']:
            if 'high_sha256' in f:
                if preds[4][f['key']]['sensor_sha256'] != f['high_sha256']:
                    raise ValueError('high exposure mismatch')
                high.add(f['key'])
        if pre:
            for scale in (1, 4):
                frames = [preds[scale].get(f['key']) for f in r['frames']]
                if (checked['mechanics_ok'] and len(frames) == 2 and all(x and x['uncut_ink_gate']
                        and x['native']['decision'] == x['noisy']['decision'] == 'UNKNOWN' for x in frames)):
                    allowed.append(scale)
        else:
            if r['scale'] != (None if e.get('pull_only') else choice):
                raise ValueError('acquisition scale mismatch')
            for f in checked['rows']:
                x = preds[choice][f['key']]
                dec = [x[k]['decision'] for k in ('native', 'noisy')]
                wanted = 'STOP' if f['actual_mm'] >= 9 else 'BELOW'
                if f['phase'] in ('pulled', 'loaded'):
                    ok = dec == ['UNKNOWN']*2
                elif f['phase'] == 'work' and f['actual_mm'] in (8, 9):
                    ok = dec == [wanted]*2
                else:
                    ok = all(d in (wanted, 'UNKNOWN') for d in dec)
                f.update(selected_decisions=dec, selected_optical_ok=ok)
        results.append(checked)
    if (seen != set(preds[1]) or high != set(preds[4]) or len({r['run_id'] for r in results}) != len(results)
            or len({r['controller'] for r in results if r['controller']}) > 1):
        raise ValueError('incomplete or inconsistent records')
    record = dict(source_sha256=hashes, spec_sha256=sha256(SPEC), blind_sha256=sha256(blind),
                  confirmation_run=False, visual_control=False, runs=results)
    if pre:
        record['selected'] = dict(rail='slide', scale=min(allowed)) if allowed else None
        passed = bool(allowed)
    else:
        cover = {mm: sorted({r['case'] for r in results if r['mechanics_ok'] for f in r['rows']
                            if f['phase'] == 'work' and f['actual_mm'] == mm}) for mm in (8, 9)}
        passed = (all(r['mechanics_ok'] and all(f['selected_optical_ok'] for f in r['rows']) for r in results)
                  and any(a != b for a in cover[8] for b in cover[9]))
        record.update(coverage=cover, selected=dict(rail='slide', scale=choice))
    write_new(dest, dict(passed=passed, **record))
    print(args.stage, passed, record['selected'])
    return int(not passed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out-root', type=Path, default=OUT)
    ap.add_argument('--stage', choices=('collect', 'preflight', 'work'), required=True)
    ap.add_argument('--case', choices=('pull', 'a', 'b', 'c', 'd', 'e'))
    opts = ap.parse_args()
    if opts.stage == 'collect':
        if not opts.case:
            ap.error('collection requires --case')
        opts.rail = 'slide'
        with patch.multiple(acquisition, PLAN=SPEC, SOURCES=SOURCES, SeamRailWorld=SeamSlideWorld):
            raise SystemExit(acquisition.acquire(opts))
    raise SystemExit(evaluate(opts))
