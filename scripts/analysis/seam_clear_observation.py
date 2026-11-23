"""Use one existing closed 3 mm observation back-off on each matched work state."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

import seam_work_roll as replay
from capture_seam_rail import high_exposure
from capture_surface_seam import Capture, ROOT, add_arguments, sha256

SPEC = ROOT/'config/task/seam_clear_observation_v1.json'
OUT = ROOT/'validation/task/seam_clear_observation_v1'


class Clear(Capture):
    def save(self, phase, obs):
        self.record['frames'].append(phase)
        f, w = self.f, self.w
        if phase != 'work' or w.tick != f['tick']:
            return
        token = f['key']+'.npz'
        im, st = self.src/'sensor'/token, self.src/'structure'/token
        if sha256(im) != f['sensor_sha256'] or sha256(st) != f['structure_sha256']:
            raise ValueError('reference changed')
        with np.load(im) as raw, np.load(st) as data:
            edges = data['edges']
            if (not np.array_equal(w.image, raw['rgb']) or not np.array_equal(w.depth.astype('<f4'), raw['depth'])
                    or not np.array_equal(w.truth.nodes(), data['nodes'])):
                raise ValueError('working checkpoint mismatch')
        before = self.tcp(self.cutter)
        if not (w.jaws_closed() and np.array_equal(before, f['tcp']) and np.array_equal(
                edges, np.asarray(w.call('tissueConstraints')).reshape(-1, 3)) and obs.cut_length_m == f['obs']['cut_length_m']):
            raise ValueError('mechanical checkpoint mismatch')
        self.record.update(matched=True, clear_script_sha256=sha256(Path(__file__)), before_tcp=before)
        events = w.call('tissueStats').get('cut_events')
        d = w.cfg['cut']['clear_back_m']
        if d != .003 or not w.clear_view(d):
            self.record['reason'] = 'existing closed back-off refused'
            raise replay.Finished()
        after = w.observe()
        at = self.tcp(self.cutter)
        shift = at[:, 3]-before[:, 3]
        travel = float(-shift@before[:, w.TOOL_AXIS_COL])
        residual = float(np.linalg.norm(shift + d*before[:, w.TOOL_AXIS_COL]))
        ready_distance = float(np.linalg.norm(at[:, 3]-self.ready[:, 3]))
        ok = (residual <= .0005 and ready_distance > .0005 and w.jaws_closed() and self.tension_ok(after)
              and not after.faults and not after.torn and after.cut_length_m == obs.cut_length_m
              and all(after.grasped.get(a, 0) >= self.cfg['grasp']['min_nodes'] for a in (self.primary,self.assist))
              and w.call('tissueStats').get('cut_events') == events
              and np.array_equal(edges, np.asarray(w.call('tissueConstraints')).reshape(-1, 3)))
        self.record.update(pose_qualified=bool(ok), after_tcp=at, travel_m=travel, residual_m=residual,
                           ready_distance_m=ready_distance, after_obs=vars(after))
        if ok:
            high_exposure(w, self.folder/'high.npz')
            self.record['high_sha256'] = sha256(self.folder/'high.npz')
        raise replay.Finished()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument('--case', required=True, choices=('a','b'))
    with patch.multiple(replay, Probe=Clear, SPEC=SPEC, OUT=OUT):
        raise SystemExit(replay.run(ap.parse_args()))
