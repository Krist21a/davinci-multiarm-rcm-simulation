"""Paired second-bite pilot. Policies start only after a common first reacquisition."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np

from run_two_bite import BASE, PLAN
from simtools import REPO, add_arguments, connect, relative, sha256, stop_simulation
from two_bite import S, TwoBiteFSM, TwoBiteWorld

PAIR = REPO / 'config/task/two_bite_pair_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    default=lambda v: np.asarray(v).tolist()).encode()).hexdigest()


class PairWorld(TwoBiteWorld):
    checkpoint = None
    perturbed = False

    def width(self):
        proxy = (self._last_est or {}).get('tension_proxy') or {}
        v = proxy.get('width_m')
        return v if proxy.get('valid') and type(v) in (int, float) and np.isfinite(v) and v > 0 else None

    def observe(self, **kwargs):
        obs = super().observe(**kwargs)
        if (self.checkpoint is None and self.bites == 1 and self.direct_fix(('slit', 'bank_slit'))
                and any(e['event'] == 'post_closure_reacquired' and e['tick'] == self.tick for e in self.event_log)):
            self.reference_width = self.width()
            with self.truth.purpose('scoring'):
                nodes = self.truth.nodes()
            self.checkpoint = dict(tick=self.tick, width_m=self.reference_width, nodes_sha256=digest(nodes),
                                   pose_sha256=digest([self._tcp_pose(), *self.camera()]),
                                   perception_prefix_sha256=digest(self.perc_log))
        return obs

    def _events(self):
        if self.checkpoint and not self.perturbed:
            self.perturbed = True
            if self.args.disturbance == 'view':
                self.events = {'occlude_until_camera': self.tick}
            else:
                self.events = {'relax': self.tick, 'relax_m': self.cfg['events']['relax_m']}
            self.event_log.append(dict(tick=self.tick, event='pair_injection', kind=self.args.disturbance))
        super()._events()

    def adjust_camera(self):
        if self.perturbed and self.args.strategy not in ('camera', 'both'):
            self.steps(25)
            self.event_log.append(dict(tick=self.tick, event='fixed_camera_hold'))
            return True
        return super().adjust_camera()


class PairFSM(TwoBiteFSM):
    def go(self, new, reason, obs=None):
        if (reason == 'post-advance optical retry: camera moved' and self.w.perturbed
                and self.w.args.strategy not in ('camera', 'both')):
            reason = 'post-advance optical retry: fixed camera held'
        super().go(new, reason, obs)

    def _adjust_tension(self, obs):
        if not self.w.perturbed:
            return super()._adjust_tension(obs)
        if self.w.args.strategy not in ('traction', 'both'):
            self.w.steps(10 * len(self.retractors))
            self.w.event_log.append(dict(tick=self.w.tick, event='fixed_traction_hold'))
            return
        base, width = self.w.reference_width, self.w.width()
        if base is None or width is None:
            return self.go(S.FAILED, 'traction refused: missing optical width', obs)
        cap = self.cfg['cut']['retract_step_m']
        step = float(np.clip((base-width)/2, -cap, cap))
        low = self.cfg['tension']['low_strain_after_cut_start']
        if not ((obs.tension < low and step > 0) or (obs.tension > self.cfg['tension']['high_strain'] and step < 0)):
            return self.go(S.FAILED, 'traction refused: optical direction disagrees with safety band', obs)
        self.w.event_log.append(dict(tick=self.w.tick, event='optical_traction_request', step_m=step,
                                     width_m=width, reference_m=base))
        for arm in self.retractors:
            if not self.w.retract(arm, step):
                return self.go(S.FAILED, f'guard refused optical traction on {arm}', obs)


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    plan, pair = (json.loads(p.read_bytes()) for p in (PLAN, PAIR))
    args.scene = REPO / plan['scene']
    if sha256(BASE) != plan['base_config_sha256'] or sha256(args.scene) != plan['scene_sha256']:
        raise ValueError('configuration or scene mismatch')
    cfg = json.loads(BASE.read_bytes())
    cfg['cut']['control_source'] = 'vision'
    cfg['perception'].update(plan['cases']['nominal'], mode='shadow', material_pattern=True, local_material_chart=True,
                             front_recovery=True, predictive_view=True, exclude_tcp_marker=True,
                             recovery_view_after_cut=True, ray_marker_fit=True)
    args.initial_view, args.initial_view_insertion_m = plan['initial_ecm'][:2], plan['initial_ecm'][2]
    args.scenario, args.run_id = args.disturbance, uuid.uuid4().hex
    # Bind the paired runs to the exact core that passed the two-bite baseline.
    baseline_path = REPO / 'validation/task/two_bite_v1_nominal.json'
    baseline = json.loads(baseline_path.read_bytes())
    sources = dict(baseline['source_sha256'])
    if cfg != baseline['config'] or any(sha256(REPO/k) != v for k, v in sources.items()):
        raise ValueError('baseline inputs have changed')
    sources[relative(Path(__file__))] = sha256(Path(__file__))
    r = dict(case='nominal', pair_protocol=pair, pair_sha256=sha256(PAIR), strategy=args.strategy,
             disturbance=args.disturbance, protocol=plan, protocol_sha256=sha256(PLAN), config=cfg,
             run_id=args.run_id, scene_sha256=sha256(args.scene), base_config_sha256=sha256(BASE),
             source_sha256=sources, baseline_sha256=sha256(baseline_path), error=None, cleanup_error=None)
    client = sim = None
    try:
        client, sim = connect(args)
        w = PairWorld(client, sim, args, cfg, mode='full', protocol=plan)
        if w.controller_sha256 != plan['controller_sha256']:
            raise ValueError('controller mismatch')
        fsm = PairFSM(w, cfg)
        fsm.run()
        w.observe(inject_events=False)
        r.update(fsm.report(), samples=w.samples, perception=w.perc_log, events=w.event_log, park=w.park,
                 bites=w.bites, camera_moves=w.camera_moves, applied_view=w.initial_view_readback,
                 controller_sha256=w.controller_sha256, strict_truth=w.truth.strict,
                 truth_purposes=w.truth.reads_by_purpose, roles=w.roles_report, checkpoint=w.checkpoint)
    except Exception as exc:
        r['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if sim is not None:
            try:
                stop_simulation(client, sim)
            except Exception as exc:
                r['cleanup_error'] = str(exc)
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()
    with args.out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(r, f, indent=2, allow_nan=False, default=lambda x: np.asarray(x).tolist())
        f.write('\n')
    print(args.disturbance, args.strategy, r.get('final_state'), r['error'] or r['cleanup_error'] or r.get('reason'))
    return int(r.get('final_state') != 'DONE' or bool(r['error'] or r['cleanup_error']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    spec = json.loads(PAIR.read_bytes())
    parser.add_argument('--strategy', choices=spec['strategies'], required=True)
    parser.add_argument('--disturbance', choices=spec['disturbances'], required=True)
    parser.add_argument('--out', required=True, type=Path)
    raise SystemExit(run(parser.parse_args()))
