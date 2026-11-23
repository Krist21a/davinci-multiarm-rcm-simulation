"""Sealed development replay of the coded-cell reader. No simulator, controller, or holdout."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for module in ('vision', 'task'):
    sys.path.insert(0, str(ROOT/'scripts'/module))

from cell_certificate import OPEN, PRESENT, UNKNOWN, classify
from cell_v2_controls import contains, controls, depths
from seam_topology import material_state

DATA = ROOT/'validation/task'
PLAN = ROOT/'config/task/cell_certificate_v2_dev.json'
SOURCES = [PLAN, Path(__file__), Path(__file__).with_name('cell_v2_controls.py'),
           *(ROOT/'scripts/vision'/f for f in ('cell_certificate.py', 'cell_certificate_v2.py',
                                               'cell_certificate_v1.py', 'ruler_readout.py')),
           ROOT/'scripts/task/seam_topology.py']
ZOOM, NEAR, LOOK, HD = ('seam_rim_zoom_v1', 'seam_rim_near_v1', 'seam_rim_look_v1', 'm5_local_hd_dev_v1')
GROUPS = ([(ZOOM, 'asym/'+k) for k in ('pull','a','b','c','d','e')] + [(NEAR,'asym/near')]
          + [(LOOK,k) for k in ('near','b','c')] + [(HD,k) for k in ('pull','a','b','c')])


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_bytes())


def write(p, obj):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def exposure_files():
    paths = []
    for study, case in GROUPS:
        folder = DATA/study/case/('hd' if study == HD else 'high')
        items = sorted(folder.glob('*.npz'))
        if not items:
            raise ValueError(f'no exposures: {folder}')
        paths.extend(items)
    return paths


def optics(study):
    w, h, angle = (1280, 960, 70.) if study == HD else (2560, 1920, 20.)
    return w, h, float(.5*w/np.tan(np.deg2rad(angle/2)))


def predict(out):
    out.mkdir(parents=True, exist_ok=False)
    paths = exposure_files()
    reg = dict(criteria=load(PLAN), sources={p.relative_to(ROOT).as_posix():digest(p) for p in SOURCES},
               images={p.relative_to(DATA).as_posix():digest(p) for p in paths},
               parent_decisions={k:digest(DATA/k/'decision.json') for k in (ZOOM, NEAR, LOOK, HD)})
    write(out/'registration.json', reg)
    checks = controls()
    write(out/'synthetic.json', checks)
    if not all(c['passed'] for c in checks):
        raise ValueError('synthetic gate failed; see synthetic.json; no physical replay started')
    results = {}
    # Deduplicate identical exposures, not observations. A replayed checkpoint
    # is accounted for when joined, but is not independent validation.
    for p in paths:
        key = reg['images'][p.relative_to(DATA).as_posix()]
        if key in results:
            continue
        w, h, f = optics(p.relative_to(DATA).parts[0])
        with np.load(p, allow_pickle=False) as data:
            if set(data.files) != {'rgb', 'depth'}:
                raise ValueError('RGB-D only')
            im, z = data['rgb'], data['depth']
        if im.shape != (h, w, 3) or im.dtype != np.uint8 or z.shape != (h,w):
            raise ValueError('incompatible exposure')
        results[key] = [classify(im, depth, focal=f, principal=[w/2,h/2], sigma=s) for s, depth in depths(z)]
    write(out/'predictions.json', dict(registration_sha256=digest(out/'registration.json'), results=results))


def score(out):
    reg, sealed = load(out/'registration.json'), load(out/'predictions.json')
    if (sealed['registration_sha256'] != digest(out/'registration.json')
            or any(digest(ROOT/p) != s for p,s in reg['sources'].items())
            or any(digest(DATA/k/'decision.json') != s for k,s in reg['parent_decisions'].items())):
        raise ValueError('preregistration changed')
    seen, rows = set(), []
    for study, case in GROUPS:
        folder = DATA/study/case
        path = folder/('probe.json' if study == LOOK else 'oracle.json')
        o, parent = load(path), load(DATA/study/'decision.json')
        if study == LOOK:
            record = next(c for c in parent['cases'] if c['case'] == case)
            if digest(path) != record['probe_sha256'] or not (o['qualified'] and o['matched']):
                raise ValueError('unmatched look')
            R, t = map(np.array, o['camera'])
        elif study != HD:
            record = parent['near'] if study == NEAR else next(c for c in parent['reports'] if c['case'] == folder.name)
            if (digest(path) != record['oracle_sha256'] or not record['mechanics_ok']
                    or 'CAMERA_ADJUST' in o['visited']):
                raise ValueError('unqualified original trajectory')
            R, t = (np.array(o['view'][k]) for k in ('applied_R','applied_t'))
        for fr in o['frames']:
            fpath = folder/('hd' if study == HD else 'high')/(fr['key']+'.npz')
            fid = fpath.relative_to(DATA).as_posix()
            sha = fr['hd_sha256' if study == HD else 'high_sha256']
            if fid in seen or reg['images'][fid] != sha or digest(fpath) != sha:
                raise ValueError('sensor join failed')
            seen.add(fid)
            variants = sealed['results'][sha]
            if study == HD:
                rows.append(dict(study=study, case=case, phase=fr['phase'], sensor=sha,
                                 incompatible_print_ok=all(v['decision']==UNKNOWN and v['local_mm'] is None
                                                           and not any(v.get('corners', [])) for v in variants)))
                continue
            w,h,f = optics(study)
            spath = folder/'structure'/(fr['key']+'.npz')
            if (not fr['restored'] or not np.isclose(fr['zoom_angle'], np.deg2rad(20.), atol=1e-6, rtol=0)
                    or digest(spath) != fr['structure_sha256']):
                raise ValueError('optics/structure mismatch')
            with np.load(spath, allow_pickle=False) as st:
                state = material_state(st['edges'])
                faces = np.asarray(state['faces'], dtype=bool)
                if not np.array_equal(faces, st['faces']) or state['prefix_rows'] != fr['prefix_mm']:
                    raise ValueError('unverified structural labels')
                xyz = ((st['nodes']-t)@R)[:,11:13].reshape(28, 3)
                uv = np.array([w/2,h/2])-f*xyz[:, :2]/xyz[:, 2:]
            metrics = []
            for v in variants:
                escaped, wrong, checked, refined = [], [], 0, 0
                for i,c in enumerate(v.get('corners', [])):
                    if c is not None:
                        checked += 1
                        refined += int(c['source'] in ('ink-edge', 'end-edge'))
                        if not contains(c['polygon'], uv[i]):
                            escaped.append(i)
                gains = 0
                for j,pair in enumerate(v.get('states', [])):
                    for half,val in enumerate(pair):
                        if (faces[j,half] and val == OPEN) or (not faces[j,half] and val == PRESENT):
                            wrong.append([j,half,val])
                        m = v['faces'][j][half]
                        gains += int(not faces[j,half] and m['box_pixels']==0 and m['blocks']>0)
                metrics.append(dict(local=v['local_mm'], prefix=v['prefix'], decision=v['decision'],
                                    bounded=checked, refined=refined, escaped=escaped, wrong_faces=wrong,
                                    empty_to_deep_faces=gains))
            rows.append(dict(study=study,case=case,phase=fr['phase'],sensor=sha,structure_sha256=digest(spath),
                             mm=state['prefix_rows'],variants=metrics))
    if seen != set(reg['images']) or set(reg['images'].values()) != set(sealed['results']):
        raise ValueError('incomplete prediction join')
    verified, terminal, invalid = [], [], []
    for r in rows:
        if r['study'] == HD:
            if not r['incompatible_print_ok']:
                invalid.append(r['sensor'])
            continue
        verified.append(r)
        if r['phase'] in ('work','backoff') and r['mm'] in (8,9,10):
            terminal.append(r)
        for v in r['variants']:
            if (v['escaped'] or v['wrong_faces'] or any(n is not None and (r['mm']==0 or n!=r['mm'])
                    for n in (v['local'],v['prefix']))):
                invalid.append(r['sensor'])
    synthetic = load(out/'synthetic.json')
    all_local = all(all(v['local']==r['mm'] for v in r['variants']) for r in terminal)
    coverage = {phase:sorted({r['mm'] for r in terminal if r['phase']==phase}) for phase in ('work','backoff')}
    passed = (all(r['passed'] for r in synthetic) and not invalid and all_local
              and all(lengths==[8,9,10] for lengths in coverage.values()))
    result = dict(criteria=reg['criteria'],registration_sha256=digest(out/'registration.json'),
                  predictions_sha256=digest(out/'predictions.json'),synthetic_sha256=digest(out/'synthetic.json'),
                  synthetic_pass=sum(r['passed'] for r in synthetic),synthetic_total=len(synthetic),
                  observations=len(rows), unique_images=len(sealed['results']),terminal_coverage=coverage,
                  incorrect_or_unbounded=list(sorted(set(invalid))),passed=passed,
                  control_enabled=False,confirmation=False,rows=rows)
    write(out/'decision.json', result)
    print('synthetic', result['synthetic_pass'], '/',len(synthetic),'development gate:',passed, flush=True)
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    predict(args.out)
    raise SystemExit(0 if score(args.out) else 1)
