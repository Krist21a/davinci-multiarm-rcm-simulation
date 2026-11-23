"""Optical-only HD exposure at the third closure of the unchanged 640p vision controller."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import run_same_bite as parent
from live_active_look import ActiveLookFSM, ActiveLookWorld
from run_cut_task import REPO, add_arguments, sha256
from sim_world import SimWorld

PLAN = REPO / 'config/task/m5_row9_hd_dev_v1_1.json'
ACTIVE = REPO / 'config/task/m5_live_active_look_v1.json'
BRIDGE = REPO / 'config/task/m5_bridge_hd_negative_v1.json'


class PassiveHDWorld(ActiveLookWorld):
    hd_optical = None

    def _estimate(self):
        # This opt-in observer never supplies an estimate to the world or FSM.
        before = copy.deepcopy(self._perc_estimator)
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(self.perc_rng.bit_generator.state)
        control = super()._estimate()
        if (self.W != 640 or self.H != 480 or
                sum(e['event'] == 'closure' for e in self.event_log) < 3):
            return control
        original = (self.W, self.H, self.f, self.image, self.depth)
        sigma = self.perc_cfg.get('depth_noise_m', 0.)
        if sigma != .0005:
            raise ValueError('not the preregistered noisy trial')
        try:
            self.sim.setIntArray2Property(self.cam, 'resolution', [1280, 960])
            if list(self.sim.getVisionSensorRes(self.cam)) != [1280, 960]:
                raise ValueError('temporary HD optics refused')
            self.W, self.H, self.f = 1280, 960, 2 * self.f
            SimWorld.render(self)
            rgb = np.asarray(self.image)
            depth = np.asarray(self.depth, dtype=float)
            R, t = self.camera()
            if before is None:
                # Only possible after an uncut setup reset; never create a
                # material-history prior from hidden structural labels.
                before = copy.deepcopy(self._perc_estimator)
            before.W, before.H, before.f = self.W, self.H, self.f
            rng.normal(size=3)
            rng.normal(size=3)
            noisy = depth + rng.normal(0, sigma, depth.shape)
            optical = before.estimate(rgb.astype(float), noisy, R, t,
                                      post_grasp=bool(self.grasped_points))
            with (self.args.hd_dir / f'{self.tick:03d}.npz').open('xb') as file:
                np.savez_compressed(file, rgb=rgb, depth=depth.astype('<f4'),
                                    observed_depth=noisy.astype('<f4'), cam_R=R, cam_t=t, f=self.f)
            if self.hd_optical is None:
                self.hd_optical = []
            self.hd_optical.append(dict(tick=self.tick, estimate=optical))
        finally:
            self.sim.setIntArray2Property(self.cam, 'resolution', [640, 480])
            self.W, self.H, self.f, self.image, self.depth = original
            if list(self.sim.getVisionSensorRes(self.cam)) != [640, 480]:
                raise ValueError('failed to restore native optics')
            # The next real observation must start from the native sensor image,
            # not an incidental HD readback left by this passive acquisition.
            SimWorld.render(self)
            if (not np.array_equal(np.asarray(self.image), np.asarray(original[3])) or
                    not np.array_equal(self.depth, original[4])):
                raise ValueError('passive HD exposure changed the native observation')
        return control


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sheet', choices=('conventional', 'bridge'), default='conventional')
    args = ap.parse_args()
    plan = PLAN if args.sheet == 'conventional' else BRIDGE
    if args.sheet == 'bridge':
        registered = json.loads(BRIDGE.read_bytes())
        if sha256(REPO / registered['parent']) != registered['parent_sha256']:
            raise ValueError('physical bridge specimen differs from the registered one')
    args.case = 'depth_noise_0p5'
    args.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    args.hd_dir = args.out / 'hd_exposures'
    args.capture = args.out / 'native_active_hd'
    if args.out.exists():
        raise FileExistsError(args.out)
    args.hd_dir.mkdir(parents=True)
    args.capture.mkdir()
    status = parent.run(args, world_class=PassiveHDWorld, fsm_class=ActiveLookFSM, active_spec=ACTIVE,
                        policy=('bridge_hd_observer' if args.sheet == 'bridge' else 'row9_hd_observer'),
                        extra_sources=(plan, Path(__file__)),
                        sheet_cfg={'bridge_print': True} if args.sheet == 'bridge' else None)
    report = json.loads((args.out / 'depth_noise_0p5.json').read_bytes())
    if 'perception' not in report:
        raise RuntimeError(f"HD acquisition failed: {report.get('error')}")
    # Optical estimates are stored separately. An independent scorer opens
    # structural labels only after sealing predictions from this file.
    parent.put(args.out / 'hd_optical.json', dict(
        registration_sha256=report['registration_sha256'], run_id=report['run_id'],
        plan_sha256=sha256(plan), source_sha256=sha256(Path(__file__)),
        exposures=report.get('optical_observer') or []))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
