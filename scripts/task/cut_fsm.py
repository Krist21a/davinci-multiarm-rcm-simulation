"""State machine for the prostate-adjacent cutting task: PSM1 retracts, PSM2 cuts, PSM3 assists / takes over,
the ECM holds the view.

The machine only decides; a World (scripts/task/sim_world.py for CoppeliaSim, fake_world.py for tests) senses
and acts. Every tick it reads one Observation, lets the supervisor interrupt the current state, and otherwise
runs one step of the current state's handler.

Supervisor, in priority order (checked in the states listed in WATCHED):
  1. fault or tear                 -> FAILED   (any non-terminal state)
  2. cut site or scissors hidden   -> OCCLUDED (all arms hold first; the scissors never close while hidden)
  3. a retraction grasp lost       -> RELOCALIZE
  4. tissue tension out of band    -> RETRACT_ADJUST
  5. cut site off-centre in image  -> CAMERA_ADJUST
An interrupted CUT always comes back through ALIGN: the blades are re-aligned before the next closure.
Recovery states keep the first interrupted state as the return target, so nested recoveries return there.
"""
import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class S(enum.Enum):
    INIT = '初始化'
    LOCALIZE = '视觉定位'
    GRASP = '建立抓持'
    EXPOSE = '组织暴露'
    ALIGN = '切割对准'
    CUT = '切割执行'
    OCCLUDED = '遮挡暂停'
    RETRACT_ADJUST = '牵拉调整'
    CAMERA_ADJUST = '相机调整'
    RELOCALIZE = '重新定位'
    RESUME = '恢复执行'
    DONE = '完成'
    FAILED = '失败停止'


TERMINAL = {S.DONE, S.FAILED}
NEEDS_VIEW = {S.GRASP, S.EXPOSE, S.ALIGN, S.CUT}
NEEDS_GRASP = {S.EXPOSE, S.ALIGN, S.CUT}
NEEDS_TENSION = {S.ALIGN, S.CUT}
RECOVERY = {S.OCCLUDED, S.RETRACT_ADJUST, S.CAMERA_ADJUST, S.RELOCALIZE, S.RESUME}


@dataclass
class Observation:
    cut_site_visible: bool
    scissors_visible: bool
    cut_site_offset: Optional[float]          # distance of the cut site from the image centre, 1 = image edge
    grasped: Dict[str, int]                   # retraction arm -> grasped tissue nodes
    tension: float                            # median constraint strain of the sheet
    torn: int
    cut_length_m: float
    faults: List[str] = field(default_factory=list)
    max_strain: float = 0.0                   # largest single constraint strain (corners, cut ends)


@dataclass
class Target:
    cut_start: object                         # world-specific (e.g. a 3-D point)
    grasp_points: Dict[str, object]           # retraction arm -> grasp point


class CutTaskFSM:
    def __init__(self, world, cfg, cut_only=False):
        self.w, self.cfg, self.cut_only = world, cfg, cut_only
        r = cfg['roles']
        self.primary, self.assist, self.cutter = r['retract_primary'], r['retract_assist'], r['cut']
        self.retractors = [] if cut_only else [self.primary]       # arms that hold tissue now
        self.state, self.return_to, self.reason = S.INIT, None, ''
        self.target = None
        self.tick_count, self.ticks_in_state = 0, 0
        self.attempts = {S.CAMERA_ADJUST: 0, S.RETRACT_ADJUST: 0, S.RELOCALIZE: 0}
        self.log, self.visited, self.transitions = [], {S.INIT}, set()
        self.handover = None

    # -- bookkeeping ------------------------------------------------------------------------------------------
    def go(self, new, reason, obs=None):
        old = self.state
        if new in RECOVERY - {S.RESUME} and old not in RECOVERY:
            self.return_to = S.ALIGN if old == S.CUT else old
        if new == S.RELOCALIZE and old not in RECOVERY:
            self.return_to = S.EXPOSE                         # re-tension after a new grasp
        if new in (S.OCCLUDED, S.FAILED, S.RETRACT_ADJUST, S.RELOCALIZE, S.CAMERA_ADJUST):
            self.w.hold()
        self.log.append(dict(tick=self.tick_count, frm=old.name, to=new.name, label=new.value, reason=reason,
                             obs=None if obs is None else _summary(obs)))
        self.transitions.add((old.name, new.name))
        self.visited.add(new)
        self.state, self.ticks_in_state = new, 0
        if new == S.FAILED:
            self.reason = reason
            self.w.safe_stop(reason)
        if new == S.DONE:
            self.reason = reason

    def run(self):
        while self.state not in TERMINAL:
            if self.tick_count >= self.cfg['budgets']['total_ticks']:
                self.go(S.FAILED, 'total tick budget exhausted')
                break
            self.tick()
        return self.state

    def tick(self):
        self.tick_count += 1
        self.ticks_in_state += 1
        obs = self.w.observe()
        if self.supervise(obs):
            return
        getattr(self, 'on_' + self.state.name.lower())(obs)

    # -- supervisor -------------------------------------------------------------------------------------------
    def tension_ok(self, obs):
        # Once the cut has started the sheet relaxes on purpose (the cut releases it), so the lower bound drops.
        t = self.cfg['tension']
        low = t.get('low_strain_after_cut_start', t['low_strain']) if obs.cut_length_m > 0 else t['low_strain']
        return low <= obs.tension <= t['high_strain']

    def grasps_ok(self, obs):
        return all(obs.grasped.get(a, 0) >= self.cfg['grasp']['min_nodes'] for a in self.retractors)

    def visible(self, obs):
        return obs.cut_site_visible and obs.scissors_visible

    def centred(self, obs):
        return obs.cut_site_offset is not None and obs.cut_site_offset <= self.cfg['view']['center_margin']

    def supervise(self, obs):
        s = self.state
        if s in TERMINAL:
            return True
        if not all(math.isfinite(v) for v in (obs.tension, obs.max_strain, obs.cut_length_m)) or obs.cut_length_m < 0:
            self.go(S.FAILED, 'invalid tissue observation', obs)
            return True
        if obs.faults:
            self.go(S.FAILED, 'fault: ' + '; '.join(obs.faults), obs)
            return True
        if obs.torn and self.cfg['tension']['tear_is_failure']:
            self.go(S.FAILED, f'tissue torn ({obs.torn} constraints)', obs)
            return True
        limit = self.cfg['cut']['length_m'] + self.cfg['cut'].get('max_overcut_m', 0.0)
        if obs.cut_length_m > limit + 1e-12:
            self.go(S.FAILED, f'cut exceeds allowed length ({obs.cut_length_m * 1000:.2f} mm)', obs)
            return True
        if s in NEEDS_VIEW and not self.visible(obs):
            self.go(S.OCCLUDED, 'cut site or scissors not visible', obs)
            return True
        if s in NEEDS_GRASP and not self.grasps_ok(obs):
            lost = [a for a in self.retractors if obs.grasped.get(a, 0) < self.cfg['grasp']['min_nodes']]
            self.go(S.RELOCALIZE, f'grasp lost by {lost}', obs)
            return True
        if s in NEEDS_TENSION and not self.tension_ok(obs):
            self.go(S.RETRACT_ADJUST, f'strain {obs.tension:.3f} outside band', obs)
            return True
        if s in NEEDS_VIEW and not self.centred(obs):
            self.go(S.CAMERA_ADJUST, f'cut site off-centre ({obs.cut_site_offset})', obs)
            return True
        return False

    # -- task states ------------------------------------------------------------------------------------------
    def on_init(self, obs):
        ok, why = self.w.check_roles()
        if not ok:
            return self.go(S.FAILED, 'role check: ' + why, obs)
        if not self.w.home():
            return self.go(S.FAILED, 'homing failed', obs)
        self.go(S.LOCALIZE, 'roles satisfied, arms at home', obs)

    def on_localize(self, obs):
        self.target = self.w.localize()
        if self.target is not None:
            if self.cut_only:
                self.log.append(dict(tick=self.tick_count, note='GRASP skipped: cut-only mode, the tissue is held by fixed anchors'))
                return self.go(S.EXPOSE, 'cut site localized (cut-only mode)', obs)
            return self.go(S.GRASP, 'cut site and grasp points localized', obs)
        if self.ticks_in_state >= self.cfg['budgets']['localize_ticks']:
            if self.attempts[S.CAMERA_ADJUST] < self.cfg['budgets']['camera_adjust_attempts']:
                return self.go(S.CAMERA_ADJUST, 'cut site not found', obs)
            return self.go(S.FAILED, 'cut site not found', obs)

    def on_grasp(self, obs):
        for arm in list(self.retractors):
            if obs.grasped.get(arm, 0) >= self.cfg['grasp']['min_nodes']:
                continue
            ok, why = self.w.grasp(arm, self.target.grasp_points[arm])
            if not ok:
                return self.go(S.FAILED, f'{arm} cannot grasp: {why}', obs)
        if self.assist not in self.retractors and self.assist in self.target.grasp_points:
            ok, why = self.w.grasp(self.assist, self.target.grasp_points[self.assist])
            if ok:
                self.retractors.append(self.assist)
            else:
                self.log.append(dict(tick=self.tick_count, note=f'{self.assist} assist grasp not established: {why}'))
        self.go(S.EXPOSE, f'grasped by {self.retractors}', obs)

    def on_expose(self, obs):
        if self.tension_ok(obs) and self.centred(obs):
            return self.go(S.ALIGN, f'tissue exposed, strain {obs.tension:.3f}', obs)
        if self.ticks_in_state > self.cfg['budgets']['expose_ticks']:
            return self.go(S.FAILED, 'exposure not reached', obs)
        if not self.tension_ok(obs):
            if not self.retractors:
                return self.go(S.FAILED, f'strain {obs.tension:.3f} outside band and no retraction arm', obs)
            self._adjust_tension(obs)
        elif not self.centred(obs):
            self.go(S.CAMERA_ADJUST, 'cut site off-centre', obs)

    def on_align(self, obs):
        if not self.w.open_scissors():
            return self.go(S.FAILED, 'scissors cannot open', obs)
        err = self.w.align_scissors(self.target.cut_start, obs.cut_length_m)
        if err is not None and err <= self.cfg['cut']['align_tolerance_m']:
            return self.go(S.CUT, f'blades aligned ({err * 1000:.2f} mm)', obs)
        if self.ticks_in_state >= self.cfg['budgets']['align_ticks']:
            self.go(S.FAILED, f'alignment not reached ({err})', obs)

    def on_cut(self, obs):
        if obs.cut_length_m >= self.cfg['cut']['length_m']:
            if not self.w.retract_scissors():
                return self.go(S.FAILED, 'scissors retraction refused', obs)
            # Retraction can fail indirectly (e.g. tissue damage during simulation steps).
            # The withdrawn tip need not remain in view; only the terminal safety checks apply.
            final = self.w.observe()
            issues = completion_issues(final, self.cfg)
            if issues:
                return self.go(S.FAILED, '; '.join(issues), final)
            return self.go(S.DONE, f'cut length {final.cut_length_m * 1000:.1f} mm reached; scissors retracted', final)
        if not self.w.open_scissors():
            return self.go(S.FAILED, 'scissors cannot open', obs)
        # Bound the commanded advance, not the physical cut: blade extent and the tissue grid
        # can still cause overshoot, which is checked separately by the supervisor.
        step = min(self.cfg['cut']['advance_m'], self.cfg['cut']['length_m'] - obs.cut_length_m)
        if not self.w.advance_scissors(step):
            return self.go(S.FAILED, 'scissors advance refused (limits or collision)', obs)
        # Advancing is not instantaneous. Recheck ALL interlocks on the fresh observation,
        # including faults, tears, grasp loss and strain, before the irreversible closure.
        fresh = self.w.observe()
        if self.supervise(fresh):
            return
        if not self.w.close_scissors():
            return self.go(S.FAILED, 'scissors cannot close', fresh)

    # -- recovery states --------------------------------------------------------------------------------------
    def on_occluded(self, obs):
        if self.visible(obs):
            return self.go(S.RESUME, 'view restored', obs)
        if self.ticks_in_state >= self.cfg['view']['occlusion_wait_ticks']:
            self.go(S.CAMERA_ADJUST, 'occlusion persists', obs)

    def on_camera_adjust(self, obs):
        if self.visible(obs) and self.centred(obs) and self.ticks_in_state > 1:
            return self.go(S.RESUME if self.return_to != S.LOCALIZE else S.LOCALIZE, 'view restored by the camera', obs)
        self.attempts[S.CAMERA_ADJUST] += 1
        if self.attempts[S.CAMERA_ADJUST] > self.cfg['budgets']['camera_adjust_attempts']:
            return self.go(S.FAILED, 'camera adjustment exhausted', obs)
        if not self.w.adjust_camera():
            self.go(S.FAILED, 'no camera pose left', obs)

    def on_retract_adjust(self, obs):
        if self.tension_ok(obs):
            return self.go(S.RESUME, f'strain {obs.tension:.3f} back in band', obs)
        if not self.retractors:
            return self.go(S.FAILED, f'strain {obs.tension:.3f} outside band and no retraction arm', obs)
        self.attempts[S.RETRACT_ADJUST] += 1
        if self.attempts[S.RETRACT_ADJUST] > self.cfg['budgets']['retract_adjust_attempts']:
            return self.go(S.FAILED, 'retraction adjustment exhausted', obs)
        self._adjust_tension(obs)

    def on_relocalize(self, obs):
        self.attempts[S.RELOCALIZE] += 1
        if self.attempts[S.RELOCALIZE] > self.cfg['budgets']['relocalize_attempts']:
            return self.go(S.FAILED, 'relocalization exhausted', obs)
        target = self.w.localize()
        if target is None:
            return
        self.target = target
        need = self.cfg['grasp']['min_nodes']
        handled = set()
        for arm in list(self.retractors):
            # `obs` predates this handler: an arm that just completed a takeover still reads 0 in it.
            if arm in handled or arm not in self.retractors or obs.grasped.get(arm, 0) >= need:
                continue
            self.w.release(arm)
            ok, why = self.w.grasp(arm, target.grasp_points[arm])
            if ok:
                continue
            # Handover: the primary grasp point matters most. The assist arm gives up its own point (if it held
            # one) and takes the lost primary point; a lost assist point is simply dropped.
            if arm == self.assist:
                self.retractors.remove(arm)
                self.log.append(dict(tick=self.tick_count, note=f'{arm} assist grasp dropped: {why}'))
                continue
            if self.assist != arm:
                if self.assist in self.retractors:
                    self.w.release(self.assist)
                    self.retractors.remove(self.assist)
                ok2, why2 = self.w.grasp(self.assist, target.grasp_points[arm])
                if ok2:
                    self.retractors[self.retractors.index(arm)] = self.assist
                    # The assist now holds the primary point; its own pending regrasp is skipped so
                    # the takeover is not undone by the stale observation later in this loop.
                    handled.add(self.assist)
                    self.handover = dict(lost=arm, taken_by=self.assist, point=str(target.grasp_points[arm]),
                                         tick=self.tick_count)
                    continue
                why += f'; handover to {self.assist}: {why2}'
            return self.go(S.FAILED, f'{arm} regrasp failed: {why}', obs)
        self.go(S.RESUME, f'regrasped, retractors {self.retractors}', obs)

    def on_resume(self, obs):
        target = self.return_to or S.ALIGN
        if not self.visible(obs):
            return self.go(S.OCCLUDED, 'view lost on resume', obs)
        if target in NEEDS_GRASP and not self.grasps_ok(obs):
            return self.go(S.RELOCALIZE, 'grasp missing on resume', obs)
        if target in NEEDS_TENSION and not self.tension_ok(obs):
            return self.go(S.RETRACT_ADJUST, 'tension out of band on resume', obs)
        self.return_to = None
        self.go(target, 'preconditions met, resuming', obs)

    def on_done(self, obs):
        pass

    def on_failed(self, obs):
        pass

    # -- helpers ----------------------------------------------------------------------------------------------
    def _adjust_tension(self, obs):
        step = self.cfg['cut'].get('retract_step_m', 0.0005)
        sign = 1.0 if obs.tension < self.cfg['tension']['low_strain'] else -1.0
        for arm in self.retractors:
            if not self.w.retract(arm, sign * step):
                self.go(S.FAILED, f'{arm} retraction adjustment refused', obs)
                return

    def report(self):
        return dict(final_state=self.state.name, final_label=self.state.value, reason=self.reason, ticks=self.tick_count,
                    visited=sorted(s.name for s in self.visited), transitions=sorted('%s->%s' % t for t in self.transitions),
                    retractors=self.retractors, handover=self.handover, log=self.log)


def _summary(o):
    return dict(cut_site_visible=o.cut_site_visible, scissors_visible=o.scissors_visible, cut_site_offset=o.cut_site_offset,
                grasped=o.grasped, tension=round(o.tension, 4), max_strain=round(o.max_strain, 4), torn=o.torn, cut_length_m=round(o.cut_length_m, 5),
                faults=o.faults)


def completion_issues(obs, cfg):
    """Terminal acceptance, independent of visibility after a successful withdrawal.

    The caller must also require DONE and a successful retraction. These thresholds
    describe this simulation benchmark, not clinical safety limits.
    """
    issues = list(obs.faults)
    if not all(math.isfinite(v) for v in (obs.cut_length_m, obs.tension, obs.max_strain)):
        issues.append('invalid final tissue observation')
    if obs.torn and cfg['tension']['tear_is_failure']:
        issues.append(f'tissue torn ({obs.torn} constraints)')
    if obs.tension > cfg['tension']['high_strain']:
        issues.append('final strain above upper bound')
    target = cfg['cut']['length_m']
    if obs.cut_length_m < target - 1e-12:
        issues.append('cut length below target')
    if obs.cut_length_m > target + cfg['cut'].get('max_overcut_m', 0.0) + 1e-12:
        issues.append('cut exceeds allowed length')
    return issues
