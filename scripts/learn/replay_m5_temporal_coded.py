"""Join actual closure exposures to the offline coded-material accumulator."""
import argparse
import io
import json
import math
from pathlib import Path

import numpy as np

from m5_temporal_coded import OPEN, OPEN_ROW, PRESENT, UNKNOWN, Exposure, TemporalEvidence, read_exposure
from tip_data import REPO, sha, write_json

PLAN = REPO / 'config/learn/m5_temporal_coded_capture_v1.json'
REFERENCE = REPO / 'validation/task/m5_level0_v1/nominal.json'


def capture_source_matches(name, pin):
    raw = (REPO / name).read_bytes()
    # The original Windows capture hashed this source before Git normalized it to LF.
    return sha(raw) == pin or (name == 'scripts/learn/capture_m5_temporal_coded.py'
                               and b'\r' not in raw and sha(raw.replace(b'\n', b'\r\n')) == pin)


def read_view(root, row, field, optic, tcp):
    saved = row['views'][field]
    name = saved['file']
    if Path(name).name != name:
        raise ValueError('exposure filename must be local to rgbd')
    raw = (root / 'rgbd' / name).read_bytes()
    if sha(raw) != saved['sha256']:
        raise ValueError(f'unsealed exposure: {name}')
    with np.load(io.BytesIO(raw), allow_pickle=False) as frame:
        if set(frame.files) != {'rgb', 'depth', 'cam_R', 'cam_t', 'focal', 'far', 'angle_rad'}:
            raise ValueError(f'non-optical payload: {name}')
        R, t = frame['cam_R'], frame['cam_t']
        f, far, a = (float(frame[k].item()) for k in ('focal', 'far', 'angle_rad'))
        im, z = frame['rgb'], frame['depth']
        w, h = optic['size']
        if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(t).all()
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0) or np.linalg.det(R) <= 0
                or im.shape != (h, w, 3) or im.dtype != np.uint8 or z.shape != (h, w)
                or z.dtype != np.float32 or not math.isclose(far, .3, abs_tol=1e-6)
                or not math.isclose(a, math.radians(optic['hfov_deg']), rel_tol=0, abs_tol=1e-6)
                or not math.isclose(f, (w / 2) / math.tan(a / 2), rel_tol=1e-9)):
            raise ValueError(f'malformed optical exposure: {name}')
        optical = read_exposure(row['tick'], im, z, focal=f, far=far, tcp_local=tcp)
        if optical.optics != field:
            raise ValueError(f'optics disagree: {name}')
    return optical, R, t, f, far


def describe(exposure):
    return {'tick': exposure.tick, 'marker_usable': exposure.marker_usable,
            'faces': exposure.faces, 'local_mm': exposure.local_mm}


def correspond(before, after):
    """Use the optically decoded material row and face identity, never pixel offsets."""
    faces, changed, candidates = [], [], []
    for j, (a, b) in enumerate(zip(before.faces, after.faces)):
        for k, (u, v) in enumerate(zip(a, b, strict=True)):
            if u != UNKNOWN and v != UNKNOWN:
                faces.append([j, k])
            if u == PRESENT and v == OPEN:
                changed.append([j, k])
        if PRESENT in a and b == OPEN_ROW:
            candidates.append(j)
    return dict(matched_face_ids=faces, present_to_open_face_ids=changed,
                observed_row_transitions=candidates)


def predict(data, manifest, optics):
    tcp = (manifest['calibration'] or {}).get('tcp_local')

    def exposures(row):
        if row is None:
            return {}
        if (type(row['tick']) is not int or row['tick'] < 1 or not math.isfinite(row['sim_time'])
                or row['sim_time'] < 0 or set(row['views']) != set(optics)):
            raise ValueError('invalid exposure record')
        return {field: read_view(data, row, field, spec, tcp) for field, spec in optics.items()}

    seed = exposures(manifest['seed'])
    empty = Exposure(0, None, (), None, False)
    ledger = TemporalEvidence(seed['wide'][0] if seed else empty)
    result = dict(seed={k: describe(v[0]) for k, v in seed.items()}, seed_prefix=ledger.prefix, attempts=[])
    closure, last_tick = 0, (manifest['seed'] or {}).get('tick', 0)
    for index, attempt in enumerate(manifest['attempts'], 1):
        pre, post = attempt['before'], attempt['after']
        accepted = attempt['accepted']
        if (attempt['attempt'] != index or (accepted is not None and type(accepted) is not bool)
                or (accepted is not True and post is not None) or (post is not None and pre is None)):
            raise ValueError('invalid capture event')
        a, b = exposures(pre), exposures(post)
        bound = bool(a and b and pre['tick'] >= last_tick and post['tick'] == pre['tick'] + 1
                     and post['sim_time'] > pre['sim_time'])
        views = {}
        for field in optics:
            x, y = a.get(field), b.get(field)
            same = bool(bound and np.max(np.abs(x[1] - y[1])) <= 1e-6
                        and np.linalg.norm(x[2] - y[2]) <= 1e-6 and x[3:] == y[3:])
            views[field] = dict(bound=same, before=describe(x[0]) if x else None,
                                after=describe(y[0]) if y else None,
                                correspondence=correspond(x[0], y[0]) if same else None)
        delta = None
        if accepted is True:
            closure += 1
            valid = views['wide']['bound']
            delta = ledger.record_pair(closure, a['wide'][0] if valid else empty,
                                       b['wide'][0] if valid else empty)
        if post is not None:
            last_tick = post['tick']
        elif pre is not None:
            last_tick = pre['tick']
        result['attempts'].append(dict(attempt=index, accepted=accepted, views=views, delta_rows=delta,
                                       prefix_rows=ledger.prefix, wants_endpoint_look=ledger.wants_endpoint_look))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    plan = json.loads(PLAN.read_bytes())
    capture_raw = (args.data / 'capture.json').read_bytes()
    registration_raw = (args.data / 'registration.json').read_bytes()
    capture, registration = map(json.loads, (capture_raw, registration_raw))
    if (capture['registration_sha256'] != sha(registration_raw) or registration['protocol'] != plan
            or any(not capture_source_matches(name, pin) for name, pin in registration['sources'].items())):
        raise ValueError('capture source or registration mismatch')
    reference_raw = REFERENCE.read_bytes()
    reg = dict(protocol=plan, capture_sha256=sha(capture_raw), acquisition_registration_sha256=sha(registration_raw),
               reference_sha256=sha(reference_raw), sources={
                   name: sha((REPO / name).read_bytes()) for name in (
                       'scripts/learn/replay_m5_temporal_coded.py', 'scripts/learn/m5_temporal_coded.py',
                       'scripts/learn/tip_marker_gate_dev.py', 'scripts/vision/cell_certificate.py',
                       'scripts/vision/cell_certificate_v2.py', 'scripts/vision/cell_certificate_v1.py',
                       'scripts/vision/ruler_readout.py', 'scripts/vision/cut_perception.py',
                   )})
    args.out.mkdir(exist_ok=False)
    write_json(args.out / 'registration.json', reg)
    optical = predict(args.data, capture, plan['optics'])
    digest = sha((args.out / 'registration.json').read_bytes())
    write_json(args.out / 'optical.json', dict(registration_sha256=digest, **optical))

    run_raw = (args.data / 'nominal.json').read_bytes()
    if capture['nominal_sha256'] != sha(run_raw):
        raise ValueError('acquisition reference hash mismatch')
    run, baseline = map(json.loads, (run_raw, reference_raw))
    keys = ('tick', 'cut_length_m', 'nodes_sha256')
    trace = [tuple(r[k] for k in keys) for r in run.get('trace', [])]
    reference = [tuple(r[k] for k in keys) for r in baseline['trace']]
    same_trace = bool(run['success'] and baseline['success'] and trace and trace == reference
                      and run['control_source'] == baseline['control_source'] == 'truth'
                      and run['scene_sha256'] == baseline['scene_sha256'] == plan['scene_sha256'])
    accepted = [a for a in optical['attempts'] if a['accepted'] is True]
    positive_rows = sum(a['delta_rows'] or 0 for a in accepted)
    by_tick = {t: 1000 * m for t, m, _ in trace}
    correspondences = {
        field: [None if not a['views'][field]['bound'] else {
            'matched_faces': len(a['views'][field]['correspondence']['matched_face_ids']),
            'candidate_rows': a['views'][field]['correspondence']['observed_row_transitions'],
        } for a in accepted] for field in plan['optics']
    }
    decision = dict(
        registration_sha256=digest, optical_sha256=sha((args.out / 'optical.json').read_bytes()),
        capture_run_sha256=sha(run_raw), native_trace_unchanged=same_trace, original_trace_observations=len(trace),
        accepted_closures=len(accepted), adjacent_wide_pairs=sum(a['views']['wide']['bound'] for a in accepted),
        adjacent_narrow_pairs=sum(a['views']['narrow']['bound'] for a in accepted),
        optical_seed_prefix=optical['seed_prefix'], positive_accumulated_rows=positive_rows,
        deltas_by_accepted_closure=[a['delta_rows'] for a in accepted],
        optical_correspondence_by_accepted_closure=correspondences,
        structural_mm_by_accepted_closure=[
            [by_tick.get((c[side] or {}).get('tick')) for side in ('before', 'after')]
            for c in capture['attempts'] if c['accepted'] is True
        ],
        controlled_by_optical_result=False, level0_credit=False,
    )
    write_json(args.out / 'decision.json', decision)
    print(json.dumps(decision, allow_nan=False), flush=True)
    return int(not same_trace)


if __name__ == '__main__':
    raise SystemExit(main())
