"""Post-hoc observability of a frozen work pose from the existing guarded ECM table."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import probe_workpose_pairs as p

PLAN = p.ROOT / 'config/task/m5b_v19_optical_ceiling.json'


def check(w, ref, capture, est0, rng0, candidate):
    sim = w.sim
    base = w.target_of(w.view)
    now = sim.getSimulationTime()
    with np.load(capture) as frame:
        if not (np.array_equal(frame['rgb'], np.asarray(w.image)) and np.array_equal(frame['depth'], w.depth)):
            raise ValueError('recorded work pose not reproduced')
    expected = ref['perception'][w.tick-1]['estimate']['cut']
    real_est, real_rng = w._perc_estimator, w.perc_rng
    tcp = sim.getObjectMatrix(w.by[w.cutter + '_tcp'], -1)
    rows = []
    for yaw, pitch in [base[:2] if candidate is None else candidate]:
        original = [yaw, pitch] == list(base[:2])
        row = dict(yaw=yaw, pitch=pitch, original=original)
        if not original and not w.command(w.view, yaw, pitch, base[2], 0):
            rows.append(dict(row, refused=True))
            continue
        try:
            w.render()
            w._perc_estimator, w.perc_rng = copy.deepcopy((est0, rng0))
            out = w._estimate()
            c = out['cut']
            if original and json.loads(json.dumps(c, default=lambda v: v.tolist())) != expected:
                raise ValueError('reference estimate or noise draw mismatch')
            m = c.get('material_chart') or {}
            rows.append(dict(row, refused=False, direct=c.get('front_valid') and c.get('mode') in ('slit', 'bank_slit'),
                             progress_m=c.get('progress_m'), mode=c.get('mode'), chart=m.get('valid'),
                             reason=c.get('reason') or m.get('reason'), scissors=out['scissors'].get('valid')))
        finally:
            w._perc_estimator, w.perc_rng = real_est, real_rng
            if not original and not w.command(w.view, *base, 0):
                raise RuntimeError('guard refused camera restoration; abort comparison')
        if (sim.getSimulationTime() != now or not np.array_equal(tcp, sim.getObjectMatrix(w.by[w.cutter + '_tcp'], -1))
                or not np.allclose(base, w.target_of(w.view), atol=1e-9, rtol=0)):
            raise ValueError('unmatched branch')
    return rows


def run(args):
    case = args.case
    tick = json.loads(PLAN.read_bytes())['cases'][case]
    name = 'm5b_v1_9_1_marked_' + case
    report = p.ROOT / 'validation/task' / (name + '.json')
    ref = json.loads(report.read_bytes())
    if args.out.exists():
        raise FileExistsError(args.out)
    args.scene = p.INSTRUMENT_SCENE
    cfg = copy.deepcopy(ref['config'])
    capture = p.ROOT / cfg['perception'].pop('dump_dir') / f'tick_{tick:03d}.npz'
    args.initial_view, args.initial_view_insertion_m = ref['initial_view'], ref['initial_view_insertion_m']
    args.scenario = 'nominal'
    if (p.sha256(args.scene) != ref['scene_sha256']
            or any(p.sha256(p.ROOT / k) != v for k, v in ref['source_sha256'].items())):
        raise ValueError('replay source mismatch')
    client, sim = p.connect(args)
    def branch(candidate):
        w = p.SimWorld(client, sim, args, cfg, mode='full')
        if w.controller_sha256 != ref['controller_sha256']:
            raise ValueError('controller mismatch')
        observe = w.observe
        rows = []
        def intercept(**kwargs):
            if w.tick + 1 != tick:
                return observe(**kwargs)
            prior, rng = copy.deepcopy((w._perc_estimator, w.perc_rng))
            observe(**kwargs)
            if json.loads(json.dumps(w.perc_log[-1]['estimate']['cut'], default=lambda v: v.tolist())) != ref['perception'][tick-1]['estimate']['cut']:
                raise ValueError('checkpoint estimator history differs')
            rows.extend(check(w, ref, capture, prior, rng, candidate))
            print(case, rows, flush=True)
            raise p.Sampled()
        w.observe = intercept
        try:
            p.CutTaskFSM(w, cfg).run()
        except p.Sampled:
            return rows
        raise ValueError('checkpoint not reached')
    try:
        rows = []
        for candidate in [None, *p.SimWorld.CAMERA_POSES]:
            # Returning the same target via IK need not restore bit-identical optics.
            # Reload the staged scene for every branch instead of chaining moves.
            rows.extend(branch(candidate))
            p.stop_simulation(client, sim)
        p.write_json(args.out, dict(plan_sha256=p.sha256(PLAN), script_sha256=p.sha256(Path(__file__)),
                                    frame_sha256=p.sha256(capture), report_sha256=p.sha256(report),
                                    tick=tick, case=case, rows=rows))
    finally:
        p.stop_simulation(client, sim)
        client.socket.close(linger=0)
        client.context.term()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    p.add_arguments(parser)
    parser.add_argument('--case', required=True, choices=json.loads(PLAN.read_bytes())['cases'])
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
