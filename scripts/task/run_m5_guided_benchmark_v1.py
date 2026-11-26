"""Opt-in guided-sheet visual M5 Benchmark v2; original scene and motion guards."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

import run_same_bite as parent
from run_cut_task import REPO, S, add_arguments, completion_issues, connect, scenario_events, sha256, stop_simulation
from simtools import matrix_to_np
from run_m5_endpoint_review_v4 import ACTIVE, RESERVE, ReviewWorld
from run_m5_retained_print_v7 import NATIVE, PRINT, RetainedWorld
from m5_guarded_post_bite_v2 import RoutinePostBiteFSM
from cut_fsm import CutTaskFSM
from capture_m5_temporal_coded import BANKS
from m5_guided_prefix_v1 import measure
from seam_topology import SeamTruth

SPEC = REPO/'config/task/m5_guided_benchmark_v1.json'
CERT = REPO/'src/tissue/cell_certificate_print.lua'
CLEARANCE = REPO/'config/task/m5_guided_entry_clearance_v1.json'
SHEET = dict(seam_mode='surface', guide_band_m=.0001)


class GuidedWorld(RetainedWorld):
    def _canonical_gate(self, cfg):
        if (self.sheet_cfg != SHEET or not self.vision or self.mode != 'full'
                or self.tissue_source is not None or self.roi_proxy is not None
                or self.scene_file != REPO/'scenes/davinci_si_articulated_verified.ttt'
                or not cfg['perception']['material_pattern'] or cfg['cut']['align_mode'] != 'tool_pose'):
            raise ValueError('unregistered guided vision sheet')
        return True

    def start(self):
        if self.controller_sha256 != NATIVE:
            raise ValueError('unregistered native controller')
        s = self.sim
        base = s.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n')
        old = 'assert(not surface or (spec.nu==24 and spec.nv==14 and not spec.nodes and not spec.material_bands))'
        new = ('assert(not surface or (spec.nu==24 and spec.nv==14 and not spec.nodes and '
               '(not spec.material_bands or spec.guide_band_m==0.0001)))')
        if base.count(old) != 1:
            raise ValueError('unregistered Lua fixture validation')
        source = (base.replace(old, new) + '\n' + PRINT.read_text(encoding='utf-8').replace('not T.seamMode', 'T.seamMode')
                  + '\nlocal drawPrinted=render\n' + CERT.read_text(encoding='utf-8')
                  + BANKS.replace('not T.seamMode', 'T.seamMode'))
        s.setStringProperty(self.ctrl, 'code', source)
        if s.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n') != source:
            raise ValueError('guided sheet stage mismatch')
        self.controller_sha256 = hashlib.sha256(source.encode()).hexdigest()
        super(RetainedWorld, self).start()

    def clear_view(self, d, prefer_park=False):
        return super().clear_view(d, prefer_park=False)

    def record_post_bite(self, obs):
        super().record_post_bite(obs)

    def read_review(self, obs):
        if not self.jaws_closed():
            return 'UNKNOWN'
        s, cam = self.sim, self.cam
        saved_time, camera = s.getSimulationTime(), self.camera()
        saved_rgb, saved_depth = np.asarray(self.image).copy(), self.depth.copy()
        param = s.visionfloatparam_perspective_angle
        original, size = s.getObjectFloatParam(cam, param), list(s.getVisionSensorRes(cam))
        if size != [1280, 960] or abs(original-math.radians(70)) > 1e-6:
            raise ValueError('native guided optics changed')
        w, h = review = [2560, 1920]
        try:
            s.setObjectFloatParam(cam, param, math.radians(20))
            s.setIntArray2Property(cam, 'resolution', review)
            if not self.call('drawTemporalMaterial'):
                raise ValueError('certificate print refused')
            s.handleVisionSensor(cam)
            im, a = s.getVisionSensorImg(cam)
            z, b = s.getVisionSensorDepth(cam, 1)
            angle = s.getObjectFloatParam(cam, param)
            if list(a) != review or list(b) != review or abs(angle-math.radians(20)) > 1e-6:
                raise ValueError('certificate review intrinsics changed')
            rgb = np.frombuffer(bytes(im), np.uint8).reshape(h,w,3)[::-1]
            depth = np.frombuffer(bytes(z), '<f4').reshape(h,w)[::-1]
        finally:
            s.setObjectFloatParam(cam, param, original)
            s.setIntArray2Property(cam, 'resolution', size)
            if not self.call('drawRetainedPrintV7', False):
                raise ValueError('working print restore refused')
            self.render()
        if (s.getSimulationTime() != saved_time or not self.jaws_closed()
                or not all(np.array_equal(x,y) for x,y in zip(camera,self.camera(),strict=True))
                or list(s.getVisionSensorRes(cam)) != size or s.getObjectFloatParam(cam,param) != original
                or not np.array_equal(np.asarray(self.image), saved_rgb) or not np.array_equal(self.depth, saved_depth)):
            raise ValueError('working RGB-D or held state not restored')
        f = w/2/math.tan(angle/2)
        sigma = self.perc_cfg['depth_noise_m']
        measured = depth
        if sigma:
            rng = np.random.default_rng(np.random.SeedSequence([self.perc_cfg['seed'], self.tick, 513]))
            measured = depth+rng.normal(0, sigma, depth.shape)
        path = self.args.evidence_out/f'{self.tick:03d}_certificate.npz'
        with path.open('xb') as target:
            np.savez_compressed(target, rgb=rgb, depth=depth, measured=measured, focal=f, far=self.far,
                                cam_R=camera[0], cam_t=camera[1], hfov_rad=angle)
        result = measure(rgb, measured, focal=f, far=self.far, sigma=sigma)
        self.event_log.append(dict(event='guided_prefix_review', tick=self.tick, physically_closed=True,
                                   noise_sigma_m=sigma, file=path.name, frame_sha256=sha256(path), **result))
        return result['verdict']


class GuidedFSM(RoutinePostBiteFSM):
    verified_tick = None

    def _entry_clearance_capture(self, obs):
        w = self.w
        # Take the matched baseline before any additional mechanical command.
        w.read_review(obs)
        event = w.event_log[-1]
        before_frame = w.args.evidence_out/event['file']
        with np.load(before_frame, allow_pickle=False) as a, np.load(w.args.clearance_reference, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError('closed entry clearance reference did not reproduce')
        if w.direct_progress(obs) is None or not w.jaws_closed():
            return self.go(S.FAILED, 'entry clearance lacks fresh wide evidence or closed jaws', obs)
        est = w._last_est
        entry, direction, tip = (np.asarray(p, dtype=float) for p in (
            est['cut'].get('material_start'), est['cut'].get('direction'), est['scissors'].get('tip')))
        if (any(p.shape != (3,) or not np.isfinite(p).all() for p in (entry, direction, tip))
                or not math.isclose(float(np.linalg.norm(direction)), 1., abs_tol=1e-6)):
            return self.go(S.FAILED, 'entry clearance lacks bounded optical geometry', obs)
        get_tcp = lambda: matrix_to_np(w.sim.getObjectMatrix(w.by[w.cutter + '_tcp'], -1))
        start = get_tcp()
        axis = start[:, w.TOOL_AXIS_COL]
        projection = float(axis @ direction)
        tip_s = float((tip-entry) @ direction)
        retreat = (tip_s+.0025)/projection if projection >= .8 else math.nan
        if not 0 < retreat <= .008:
            return self.go(S.FAILED, 'entry clearance exceeds registered travel bound', obs)
        tick = w.tick
        ok = w.clear_view(retreat, prefer_park=False)
        actual = get_tcp()[:, 3]-start[:, 3]
        residual = float(np.linalg.norm(actual+retreat*axis))
        tip_after = tip_s+float(actual @ direction)
        accepted = bool(ok and residual <= .0005 and tip_after <= -.002 and w.jaws_closed())
        w.event_log.append(dict(event='guided_entry_clearance', tick=tick, accepted=accepted,
                                requested_m=retreat, residual_m=residual, tip_before_m=tip_s,
                                estimated_tip_after_m=tip_after, commanded=bool(ok)))
        if not accepted:
            return self.go(S.FAILED, 'guarded entry clearance refused or missed', obs)
        after = w.observe()
        mechanically_blocked = CutTaskFSM.supervise(self, replace(after, cut_site_visible=True,
                                                                  scissors_visible=True, cut_site_offset=0.0))
        if not mechanically_blocked and w.tick > tick and w.jaws_closed():
            w.read_review(after)
        return self.go(S.FAILED, 'entry-clearance acquisition only; no further closure', after)

    def _complete(self, obs):
        if self.verified_tick == self.w.tick and self.w.jaws_closed():
            return super()._complete(obs)
        if self.w.review_armed and self.w.jaws_closed():
            return self.go(S.ALIGN, 'guided closed optical review precedes withdrawal', obs)
        return self.go(S.FAILED, 'no current guided optical withdrawal permit', obs)

    def supervise(self, obs):
        if not self.w.review_armed or self.state != S.ALIGN:
            return super().supervise(obs)
        self.w.review_armed = False
        tick = self.w.tick

        def mechanical(sample):
            return CutTaskFSM.supervise(self, replace(sample, cut_site_visible=True,
                                                       scissors_visible=True, cut_site_offset=0.0))

        if mechanical(obs):
            return True
        if not self.w.jaws_closed():
            self.go(S.FAILED, 'closed guided review requires physically closed jaws', obs)
            return True
        target = self.w.args.review_target
        actual = np.asarray(self.w.target_of(self.w.view), dtype=float)
        if actual.shape != (3,) or not np.isfinite(actual).all():
            self.go(S.FAILED, 'invalid guided ECM target readback', obs)
            return True
        if np.allclose(actual, target, atol=1e-6, rtol=0):
            self.w.event_log.append(dict(event='guided_review_camera_reuse', tick=tick,
                                         target=target, attempts=self.attempts[S.CAMERA_ADJUST]))
        else:
            if self.attempts[S.CAMERA_ADJUST] >= self.cfg['budgets']['camera_adjust_attempts']:
                self.go(S.FAILED, 'guided ECM review movement budget exhausted', obs)
                return True
            self.attempts[S.CAMERA_ADJUST] += 1
            if not self.w.look_from(target):
                self.go(S.FAILED, 'guarded guided ECM review refused', obs)
                return True
        fresh = self.w.observe()
        if self.w.tick <= tick or mechanical(fresh) or not self.w.jaws_closed():
            if self.state == S.ALIGN:
                self.go(S.FAILED, 'guided review stale or cutter open', fresh)
            return True
        if self.w.args.clearance_reference is not None:
            self._entry_clearance_capture(fresh)
            return True
        result = self.w.read_review(fresh)
        if result == 'TARGET':
            self.verified_tick = self.w.tick
            self.w.event_log.append(dict(event='guided_optical_withdrawal_trigger', tick=self.w.tick))
            self._complete(fresh)
        elif (result == 'BELOW' and self.visible(fresh) and self.centred(fresh)
              and self.w.direct_progress(fresh) is not None):
            self.go(S.ALIGN, 'explicit optical BELOW; fresh wide alignment required', fresh)
        else:
            self.go(S.FAILED, 'guided endpoint abstained; no further closure', fresh)
        return True


def run(a):
    plan = json.loads(SPEC.read_bytes())
    expected = dict(target=[0, 4.5, .088], size=[2560, 1920], hfov_deg=20, arm_progress_m=.006)
    if plan['review'] != expected or plan['working'] != dict(size=[1280,960], hfov_deg=70,
                                                              advance_reserve_m=.00015, closed_backoff_m=.003):
        raise ValueError('unregistered guided optics')
    task = Path(__file__).resolve().parent
    files = (SPEC, PRINT, CERT, ACTIVE, RESERVE, Path(__file__).resolve(),
             *(task/n for n in ('run_m5_retained_print_v7.py','run_m5_endpoint_review_v4.py',
                 'run_m5_sweep_reserve_v3.py','run_m5_guarded_post_bite_v2.py','m5_guarded_post_bite_v2.py',
                 'live_active_look.py','seam_topology.py')),
             *(REPO/'scripts/vision'/n for n in ('m5_guided_prefix_v1.py','cell_certificate_v1.py',
                 'cell_certificate_v2.py','ruler_readout.py','m5_local_endpoint_v2.py')),
             REPO/'scripts/learn/capture_m5_temporal_coded.py')
    policy = 'guided_benchmark_v1'
    if a.clearance_reference is not None:
        check = json.loads(CLEARANCE.read_bytes())
        archived = REPO/'validation/task/m5_guided_benchmark_v1/nominal.json'
        if (a.case != 'nominal' or sha256(a.clearance_reference) != check['reference_frame_sha256']
                or sha256(archived) != check['reference_run_sha256']):
            raise ValueError('unregistered closed entry clearance reference')
        files = (*files, CLEARANCE, archived)
        policy = 'guided_entry_clearance_v1'
    base = parent.register(a.out, policy, files)
    dest, reg = a.out/(a.case+'.json'), a.out/'registration.json'
    reg_hash = sha256(reg)
    if dest.exists():
        raise FileExistsError(dest)
    if a.case != 'nominal':
        nominal = json.loads((a.out/'nominal.json').read_bytes())
        if nominal['registration_sha256'] != reg_hash or not nominal['accepted']:
            raise ValueError('guided nominal has not qualified')
    scenario, sigma, seed = base['cases'][a.case]
    cfg = json.loads((REPO/base['base']).read_bytes())
    cfg['cut']['control_source'] = 'vision'
    cfg['sheet'] = SHEET.copy()
    cfg['perception'].update(mode='shadow',markers=True,depth_noise_m=sigma,seed=seed,material_pattern=True,
                             local_material_chart=True,front_recovery=True,predictive_view=True,
                             exclude_tcp_marker=True,recovery_view_after_cut=True,ray_marker_fit=True)
    a.initial_view, a.initial_view_insertion_m = base['initial_view'][:2], base['initial_view'][2]
    a.scenario, a.scene, a.run_id = scenario, REPO/base['scene'], uuid.uuid4().hex
    a.review_target, a.reserve_m = expected['target'], .00015
    a.targets = json.loads(ACTIVE.read_bytes())['recovery_targets']
    a.capture = None
    a.evidence_out.mkdir(exist_ok=False)
    r = dict(registration_sha256=reg_hash, policy=policy, case=a.case, config=cfg,
             observations=[], error=None, cleanup_error=None, accepted=False)
    c = s = None
    try:
        c, s = connect(a)
        w = GuidedWorld(c,s,a,cfg,mode='full',events=scenario_events(cfg,scenario))
        if not w.canonical_seam or not isinstance(w.truth, SeamTruth) or not w.truth.strict:
            raise ValueError('guided strict seam scorer not installed')
        fsm = GuidedFSM(w,cfg)
        observe = w.observe

        def sample(**kwargs):
            obs = observe(**kwargs)
            r['observations'].append(dict(tick=w.tick,**asdict(obs)))
            return obs

        w.observe = sample
        fsm.run()
        final = w.observe(inject_events=False)
        r.update(fsm.report(), final=asdict(final), events=w.event_log, perception=w.perc_log,
                 actual_closed=w.jaws_closed(), actual_ready=w.at_ready_pose(), strict_truth=w.truth.strict,
                 truth_purposes=w.truth.reads_by_purpose, controller_sha256=w.controller_sha256)
        marks = [e for e in w.event_log if e['event']=='closure']
        seen = [e['tick'] for e in w.event_log if e['event']=='post_closure_reacquired']
        stops = [e['tick'] for e in w.event_log if e['event']=='guided_optical_withdrawal_trigger']
        sequence = bool(marks and len(marks)==len(seen) and all(
            mark['tick'] < tick < (marks[i+1]['tick'] if i+1<len(marks) else w.tick+1)
            for i,(mark,tick) in enumerate(zip(marks,seen))))
        direct = all(e.get('direct_front') and e.get('scissors_valid') and e.get('fix_age') is not None
                     and e['fix_age']<=w._fix_max_age for e in marks)
        lengths = [x['cut_length_m'] for x in r['observations']]
        held = [x['cut_length_m'] for x in r['observations'] if stops and x['tick']>=stops[0]]
        issues = completion_issues(final,cfg)
        r['accepted'] = bool(fsm.state==S.DONE and not issues and len(stops)==1 and len(set(held))==1
            and lengths[0]==0 and .009<=final.cut_length_m<=.010 and max(lengths)<=.010+1e-12
            and not any(x['torn'] or x['faults'] for x in r['observations']) and sequence and direct
            and r['actual_closed'] and r['actual_ready']
            and not any(e['event']=='perception_error' for e in w.event_log))
        r['checks'] = dict(sequence=sequence,direct=direct,issues=issues,peak_m=max(lengths),stop_ticks=stops)
    except Exception as err:
        r['error'] = f'{type(err).__name__}: {err}'
    finally:
        if s is not None:
            try:
                stop_simulation(c,s)
            except Exception as err:
                r.update(cleanup_error=str(err),accepted=False)
        if c is not None:
            c.socket.close(linger=0)
            c.context.term()
        parent.put(dest,r)
    print(a.case,r.get('final_state'),r['error'] or r.get('reason'),r.get('checks'),flush=True)
    return int(not r['accepted'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evidence-out',type=Path,required=True)
    p.add_argument('--case',default='nominal',choices=('nominal','depth_noise_0p5',
                     'occlusion_transient','slip_regrasp','relax'))
    p.add_argument('--clearance-reference', type=Path)
    return run(p.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
