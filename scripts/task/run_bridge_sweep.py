"""Passive legal ECM pose sweep at a closed-jaw, real 8 mm noisy-vision checkpoint."""
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

PLAN = REPO / 'config/task/m5_bridge_fracture_view_sweep_v1.json'
ACTIVE = REPO / 'config/task/m5_live_active_look_v1.json'
PARENT = REPO / 'config/task/m5_live_same_bite_v1.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    plan = json.loads(PLAN.read_bytes())
    if sha256(REPO / plan['parent']) != plan['parent_sha256']:
        raise ValueError('versioned positive-fracture specimen changed')
    register(args.out, 'bridge_fracture_pose_sweep', (PLAN, Path(__file__), ACTIVE,
                                                     Path(__file__).with_name('live_active_look.py')))
    cfg = json.loads((REPO / json.loads(PARENT.read_bytes())['base']).read_bytes())
    cfg['cut']['control_source'] = 'vision'
    cfg['sheet'] = {'bridge_print': True, 'bridge_fracture_faces': True}
    cfg['perception'].update(mode='shadow', markers=True, depth_noise_m=.0005, seed=1,
                             material_pattern=True, local_material_chart=True, front_recovery=True,
                             predictive_view=True, exclude_tcp_marker=True,
                             recovery_view_after_cut=True, ray_marker_fit=True)
    args.scene = REPO / 'scenes/davinci_si_articulated_verified.ttt'
    args.initial_view, args.initial_view_insertion_m = [4.5, 0], .08
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    args.capture = None
    args.run_id = uuid.uuid4().hex
    client = sim = w = None
    try:
        client, sim = connect(args)
        w = ActiveLookWorld(client, sim, args, cfg, mode='full', events=scenario_events(cfg, 'nominal'))
        if not w.vision or not w.truth.strict or w.canonical_seam:
            raise ValueError('not the strict conventional visual controller')
        fsm = ActiveLookFSM(w, cfg)
        for _ in range(cfg['budgets']['total_ticks']):
            if fsm.state in TERMINAL:
                raise RuntimeError('FSM terminated before third closure')
            fsm.tick()
            if sum(e['event'] == 'closure' for e in w.event_log) == 3:
                break
        else:
            raise RuntimeError('no third closure within original budget')
        if not w.jaws_closed():
            raise RuntimeError('third closure left scissors open')
        prefix = dict(closure_ticks=[e['tick'] for e in w.event_log if e['event'] == 'closure'],
                      reacquired_ticks=[e['tick'] for e in w.event_log
                                        if e['event'] == 'post_closure_reacquired'],
                      controller_sha256=w.controller_sha256)
        entries, refused = [], []
        for i, target in enumerate(plan['poses']):
            if not w.command(w.view, *target, 25):
                refused.append(dict(pose=i, target=target, reason='guarded ECM command refused'))
                continue
            w._invalidate_fix()
            w.use_recovery_optics()
            if ((w.W, w.H) != (1280, 960) or
                    not np.allclose(w.target_of(w.view), target, rtol=0, atol=1e-6)):
                refused.append(dict(pose=i, target=target, reason='HD view or pose readback refused'))
                continue
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
                file = args.out / f'view_{i}_{k}.npz'
                with file.open('xb') as stream:
                    np.savez_compressed(stream, rgb=np.asarray(w.image),
                                        depth=np.asarray(w.depth, dtype='<f4'),
                                        observed_depth=np.asarray(noisy, dtype='<f4'),
                                        cam_R=R, cam_t=t, f=np.float64(w.f))
                entries.append(dict(pose=i, exposure=k, target=target, readback=w.target_of(w.view),
                                    file=file.name, estimate=estimate))
        put(args.out / 'optical.json', dict(run_id=args.run_id, plan_sha256=sha256(PLAN),
                                           registration_sha256=sha256(args.out/'registration.json'),
                                           prefix=prefix, exposures=entries, refused=refused))
        with w.truth.purpose('scoring'):
            length, tissue = w.truth.cut_length(), w.truth.tissue_stats()
        put(args.out / 'labels.json', dict(run_id=args.run_id, structural_length_m=length,
                                          tissue=tissue, strict=w.truth.strict,
                                          read_purposes=w.truth.reads_by_purpose))
        print('passive sweep captured', len(entries), 'frames; structural length', length,
              'for scoring only', flush=True)
    finally:
        if sim is not None:
            stop_simulation(client, sim)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()


if __name__ == '__main__':
    main()
