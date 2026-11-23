"""Guarded ECM view of an 8 mm visual-run checkpoint; observe only, never resume on probe."""
import argparse
import copy
import json
from pathlib import Path
import uuid

import numpy as np

from cut_fsm import TERMINAL
from live_active_look import ActiveLookFSM, ActiveLookWorld
from run_cut_task import REPO, add_arguments, connect, scenario_events, sha256, stop_simulation
from run_same_bite import put, register

PLAN = REPO / 'config/task/m5_row9_view_dev_v1.json'
BRIDGE = REPO / 'config/task/m5_bridge_view_negative_v1.json'
PARENT = REPO / 'config/task/m5_live_same_bite_v1.json'
ACTIVE = REPO / 'config/task/m5_live_active_look_v1.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sheet', choices=('conventional','bridge'), default='conventional')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = PLAN if args.sheet == 'conventional' else BRIDGE
    plan = json.loads(protocol.read_bytes())
    basis = plan.get('reference', plan.get('parent'))
    expected = plan.get('reference_sha256', plan.get('parent_sha256'))
    if sha256(REPO / basis) != expected:
        raise ValueError('row-9 optical rule changed')
    register(args.out, 'bridge_view' if args.sheet == 'bridge' else 'row9_view',
             (protocol, Path(__file__), ACTIVE,
                                   Path(__file__).with_name('live_active_look.py')))
    cfg = json.loads((REPO / json.loads(PARENT.read_bytes())['base']).read_bytes())
    cfg['cut']['control_source'] = 'vision'
    if args.sheet == 'bridge':
        cfg['sheet'] = {'bridge_print': True}
    cfg['perception'].update(mode='shadow', markers=True, depth_noise_m=.0005, seed=1,
                             material_pattern=True, local_material_chart=True, front_recovery=True,
                             predictive_view=True, exclude_tcp_marker=True,
                             recovery_view_after_cut=True, ray_marker_fit=True)
    args.scene = REPO / 'scenes/davinci_si_articulated_verified.ttt'
    args.initial_view = [4.5, 0]
    args.initial_view_insertion_m = .08
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    args.capture = None
    args.run_id = uuid.uuid4().hex
    client = sim = w = None
    try:
        client, sim = connect(args)
        w = ActiveLookWorld(client, sim, args, cfg, mode='full', events=scenario_events(cfg, 'nominal'))
        if not w.vision or not w.truth.strict or w.canonical_seam:
            raise ValueError('not isolated vision control')
        fsm = ActiveLookFSM(w, cfg)
        for _ in range(cfg['budgets']['total_ticks']):
            if fsm.state in TERMINAL:
                raise RuntimeError('FSM ended before the third visual closure')
            fsm.tick()
            if sum(e['event'] == 'closure' for e in w.event_log) == 3:
                break
        else:
            raise RuntimeError('no third closure before original tick budget')
        if not w.jaws_closed():
            raise RuntimeError('closed-blade checkpoint not achieved')
        prefix = dict(closure_ticks=[e['tick'] for e in w.event_log if e['event'] == 'closure'],
                      reacquired_ticks=[e['tick'] for e in w.event_log
                                        if e['event'] == 'post_closure_reacquired'],
                      world_tick=w.tick, controller_sha256=w.controller_sha256)
        target = plan['view']
        if not w.command(w.view, *target, 25):
            raise RuntimeError('guarded ECM target refused')
        w._invalidate_fix()
        w.use_recovery_optics()
        if not np.allclose(w.target_of(w.view), target, rtol=0, atol=1e-6):
            raise RuntimeError('actual ECM pose differs')
        entries = []
        for k in range(2):
            if k:
                w.steps(1)
            w.render()
            clone = np.random.default_rng()
            clone.bit_generator.state = copy.deepcopy(w.perc_rng.bit_generator.state)
            estimate = w._estimate()
            clone.normal(size=3)
            clone.normal(size=3)
            noisy = w.depth + clone.normal(0, .0005, w.depth.shape)
            R, t = w.camera()
            with (args.out / f'view_{k}.npz').open('xb') as file:
                np.savez_compressed(file, rgb=np.asarray(w.image), depth=np.asarray(w.depth, dtype='<f4'),
                                    observed_depth=np.asarray(noisy, dtype='<f4'),
                                    cam_R=R, cam_t=t, f=np.float64(w.f))
            entries.append(dict(exposure=k, estimate=estimate))
        put(args.out / 'optical.json', dict(run_id=args.run_id, plan_sha256=sha256(protocol),
                                           registration_sha256=sha256(args.out/'registration.json'),
                                           prefix=prefix, entries=entries))
        with w.truth.purpose('scoring'):
            length = w.truth.cut_length()
            tissue = w.truth.tissue_stats()
        put(args.out / 'labels.json', dict(run_id=args.run_id, structural_length_m=length, tissue=tissue,
                                          camera_target=target, actual_camera=w.target_of(w.view),
                                          strict=w.truth.strict, read_purposes=w.truth.reads_by_purpose))
        print('passive view captured; structural length (scoring only):', length, flush=True)
    finally:
        if sim is not None:
            stop_simulation(client, sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()


if __name__ == '__main__':
    main()
