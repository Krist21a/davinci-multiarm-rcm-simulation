"""Blind component test; keep rendered-working, post-backoff, synthetic and parked evaluations separate."""
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_surface_seam import ROOT, sha256, write_new
from contact_veto_pilot import OUT as DEV
from connected_ruler import classify

SPEC = ROOT/'config/task/seam_connected_evidence_v1.json'
CLEAR = ROOT/'validation/task/seam_clear_observation_v1'
OUT = ROOT/'validation/task/seam_connected_evidence_v1'


def cuts(n):
    im = np.zeros((100, 100, 3), np.uint8)
    z = np.full((100, 100), .15)
    for j in range(13):
        rows = slice(10+6*j, 16+6*j)
        r, k = 32+48*(j % 4), 35+46*(j//4)
        im[rows, 20:49], im[rows, 49:79] = (r,235,k), (r,k,235)
        z[rows, 20:79] = .105
    im[10:10+6*n,47:52] = 0
    z[10:10+6*n,47:52] = .15
    return im, z


def predict(path):
    with np.load(path, allow_pickle=False) as d:
        if set(d.files) != {'rgb','depth'}:
            raise ValueError('image-only input required')
        rgb, z = d['rgb'], d['depth']
    seed = int(hashlib.sha256(z.tobytes()).hexdigest()[:16], 16)
    noisy = z + np.random.default_rng(seed).normal(0, .0005, z.shape)
    return dict(sha256=sha256(path), native=classify(rgb,z), noisy=classify(rgb,noisy))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    contract = json.loads(SPEC.read_bytes())
    write_new(OUT/'registration.json', dict(criteria=contract, protocol_sha256=sha256(SPEC),
              reader_sha256=sha256(ROOT/'scripts/vision/connected_ruler.py'), scorer_sha256=sha256(Path(__file__))))
    tests = {}
    for n, expected in ((0,'UNKNOWN'),(8,'BELOW'),(9,'STOP'),(10,'STOP')):
        im, z = cuts(n)
        found = classify(im,z)['decision']
        tests[f'{n} entire rows separated'] = dict(expected=expected, found=found, ok=found==expected)
    uncut, intact = cuts(0)
    for name in ('filled bridge', 'foreground band', 'whole shutter'):
        im, z = cuts(10)
        if name == 'filled bridge':
            im[34:40], z[34:40] = uncut[34:40], intact[34:40]
        elif name == 'foreground band':
            im[60:70], z[60:70] = 90, .02
        else:
            im[:], z[:] = 90, .02
        found = classify(im,z)['decision']
        tests[name] = dict(expected='UNKNOWN', found=found, ok=found=='UNKNOWN')
    pred = {p.stem:predict(p) for p in sorted(DEV.glob('slide/*/high/*.npz'),key=lambda p:p.stem)}
    clear = {sha256(p):predict(p) for p in sorted(CLEAR.glob('*/high.npz'))}
    write_new(OUT/'predictions.json', dict(work=pred,clear=clear,synthetic=tests))
    orig = json.loads((DEV/'decision.json').read_bytes())
    rows, back, used = [], [], set()
    for run in orig['runs']:
        q = DEV/'slide'/run['case']/'oracle.json'
        if not run['mechanics_ok'] or sha256(q) != run['oracle_sha256']:
            raise ValueError('not a verified mechanical run')
        for f in json.loads(q.read_bytes())['frames']:
            x = pred[f['key']]
            if f['key'] in used or x['sha256'] != f['high_sha256']:
                raise ValueError('image identity mismatch')
            used.add(f['key'])
            mm = f['prefix_mm']
            ds = [x[k]['decision'] for k in ('native','noisy')]
            target = 'UNKNOWN' if f['phase'] in ('loaded','pulled') else ('STOP' if mm>=9 else 'BELOW')
            wrong = any(d not in ('UNKNOWN', target) for d in ds)
            rows.append(dict(case=run['case'],phase=f['phase'],mm=mm,key=f['key'],decisions=ds,
                             correct=ds==[target]*2, wrong=wrong, bilateral_rows=x['native'].get('bilateral_prefix_rows')))
    if used != set(pred):
        raise ValueError('unmatched evidence')
    cdec = json.loads((CLEAR/'decision.json').read_bytes())
    for r in cdec['cases']:
        path = CLEAR/r['case']/'probe.json'
        p = json.loads(path.read_bytes())
        if not p['pose_qualified'] or sha256(path)!=r['probe_sha256']:
            raise ValueError('not a valid non-park observation')
        x = clear[p['high_sha256']]
        back.append(dict(case=r['case'],mm=r['actual_mm'],predictions=[x[k]['decision'] for k in ('native','noisy')],
                         bilateral_rows=x['native'].get('bilateral_prefix_rows'), bank_pixels=x['native'].get('both_bank_pixels')))
    work = [r for r in rows if r['phase']=='work']
    passage = ({8,9,10} <= {r['mm'] for r in work} and all(r['correct'] for r in work if r['mm'] in (8,9,10))
               and not any(r['wrong'] for r in rows) and all(t['ok'] for t in tests.values()))
    write_new(OUT/'decision.json', dict(scope='Development only; not an action source',passed=passage,
              synthetic=tests, raw_noisy_rows=rows, backoff=back, predictions_sha256=sha256(OUT/'predictions.json'),
              work_sha256=sha256(DEV/'decision.json'), clear_sha256=sha256(CLEAR/'decision.json')))
    print('eligible:',passage,'synthetic:',tests,'backoff:',back)
    for r in rows:
        if r['mm']>=8:
            print(r['case'],r['phase'],r['mm'],r['decisions'],r['bilateral_rows'])


if __name__ == '__main__':
    main()
