"""Closed-jaw 3 mm look-back development cohort for config/learn/learned_perception_v1_closed_look_dev.json.

Same cohort wiring as collect_native_hd_dev.py (block-anchor ECM view, condition
advance/target, nominal scenario, aux 0, HD twin on) plus a local CutTaskFSM patch:
ClosedLookCapture wraps _complete with the registered guarded 3 mm clear_view back-off
probe -- mechanical preconditions, a before/after structural signature, pose/residual
checks, and a probe record in world.event_log -- then falls through to the parent's
unmodified completion path. Only reached on a truth-controlled CUT completion; the
parent's supervise/overcut path is untouched. All patches restored in finally.
"""
import json
import sys
from pathlib import Path

import numpy as np

import collect

PLAN = collect.REPO / 'config' / 'learn' / 'learned_perception_v1_closed_look_dev.json'
ParentFSM = collect.CutTaskFSM


class ClosedLookCapture(ParentFSM):
    def _complete(self, obs):
        w = self.w
        record = {'event': 'closed_look_probe', 'work_tick': w.tick,
                  'look_tick': None, 'qualified': False, 'reason': None}
        w.event_log.append(record)

        def reject(reason, observed):
            record['reason'] = reason
            return self.go(collect.S.FAILED, reason, observed)

        def signature():
            with w.truth.purpose('scoring'):
                return (w.truth.tissue_stats()['cut_events'], tuple(w.call('tissueConstraints')),
                        w.accepted_closures)

        def tcp():
            return collect.matrix_to_np(w.sim.getObjectMatrix(w.by[w.cutter + '_tcp'], -1))

        if self.vision or self.state is not collect.S.CUT or w.align_mode != 'tool_pose':
            return reject('look-back not a truth-controlled CUT completion', obs)
        if not w.jaws_closed() or obs.faults or obs.torn or not self.grasps_ok(obs) or not self.tension_ok(obs):
            return reject('look-back work frame fails mechanical preconditions', obs)
        if not w.closure_commands or not w.closure_commands[-1]['accepted'] or w.closure_commands[-1]['after_tick'] >= w.tick:
            return reject('look-back work frame lacks preceding accepted closure', obs)
        if w._ready_tcp is None:
            return reject('look-back missing measured ready pose', obs)
        sig = signature()
        before, (r0, t0) = tcp(), w.camera()
        if not w.clear_view(0.003):
            return reject('guarded look-back refused', obs)
        seen = w.observe(inject_events=False)
        record['look_tick'] = w.tick
        after, (r1, t1) = tcp(), w.camera()
        residual = float(np.linalg.norm(after[:, 3] - before[:, 3] + 0.003 * before[:, w.TOOL_AXIS_COL]))
        gap = float(np.linalg.norm(after[:, 3] - w._ready_tcp[:, 3]))
        pose_r = float(np.max(np.abs(r1 - r0)))
        pose_t = float(np.linalg.norm(t1 - t0))
        unchanged = signature() == sig
        length_delta = abs(seen.cut_length_m - obs.cut_length_m)
        grasp_counts_unchanged = seen.grasped == obs.grasped
        record.update(residual_m=residual, distance_to_ready_m=gap, rotation_entry_delta=pose_r,
                      camera_translation_m=pose_t, structure_signature_unchanged=unchanged,
                      length_delta_m=length_delta, grasp_counts_unchanged=grasp_counts_unchanged)
        if (not unchanged or not grasp_counts_unchanged or seen.faults or seen.torn or not w.jaws_closed()
                or not self.grasps_ok(seen) or not self.tension_ok(seen) or not length_delta <= 1e-12
                or not residual <= 0.0005 or not gap > 0.0005 or not pose_r <= 1e-6 or not pose_t <= 1e-6):
            return reject('unqualified guarded look-back', seen)
        record['qualified'] = True
        return super()._complete(seen)


def main():
    plan = json.loads(PLAN.read_bytes())
    c = plan['cohort']
    n = len(c['condition_names'])
    first = c['start']
    parent_params = collect.episode_params
    parent_hashes = collect.source_hashes
    parent_fsm = collect.CutTaskFSM

    def cohort_params(split, index, proto=None):
        if split != 'dev' or not first <= index < first + c['blocks'] * n:
            raise ValueError('unexpected split/index in development cohort')
        block, condition = divmod(index - first, n)
        params = parent_params(split, index, proto)
        anchor = parent_params(split, first + n * block, proto)
        params.update({k: anchor[k] for k in ('yaw_deg', 'pitch_deg', 'insertion_m')})
        params.update(advance_m=c['advance_um'][condition] / 1e6,
                      target_length_m=c['target_um'][condition] / 1e6,
                      scenario=c['scenario'])
        return params

    collect.episode_params = cohort_params
    collect.CutTaskFSM = ClosedLookCapture
    collect.source_hashes = lambda proto: dict(
        parent_hashes(proto),
        closed_look_plan_sha256=collect.sha256(PLAN),
        closed_look_wrapper_sha256=collect.sha256(Path(__file__)))
    argv = sys.argv
    try:
        sys.argv = [argv[0], '--split', 'dev', '--start', str(first),
                    '--count', str(c['blocks'] * n), '--workers', '4',
                    '--tag', c['tag'], '--aux-views', '0']
        return collect.main()
    finally:
        collect.episode_params = parent_params
        collect.CutTaskFSM = parent_fsm
        collect.source_hashes = parent_hashes
        sys.argv = argv


if __name__ == '__main__':
    raise SystemExit(main())
