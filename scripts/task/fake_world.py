"""Kinematic stand-in world for testing scripts/task/cut_fsm.py without CoppeliaSim.

Strain grows linearly with the total retraction of the arms that hold the tissue; each closure of aligned,
open blades cuts `advance` of tissue. Events are injected by tick:
  occlude=(start, end)              view hidden for those ticks (transient)
  occlude_until_camera=start        view hidden from `start` until the camera has moved
  drift=start                       cut site drifts off-centre from `start` until the camera has moved
  slip={'psm1': tick}               that arm loses its grasp at that tick
  regrasp_fails={'psm1'}            that arm cannot grasp again
  takeover_unreachable={'psm3'}     takeover_point() finds nothing for that arm
  park_fails=True                   retract_scissors() refuses: the cutter cannot park
  relax=tick                        tissue relaxes (strain falls) at that tick
  tear=tick, fault=tick             tissue tears / an unexpected collision is reported
  roles_ok=False, unreachable={'psm1'}, camera_poses=n
"""
from cut_fsm import Observation, Target


class FakeWorld:
    def __init__(self, cut_only=False, vision=False, **events):
        self.e = events
        self.cut_only = cut_only
        # Opt-in M5b vision mode: the observation carries the estimator-side fields the
        # FSM reads (front fix -> cut_progress_m/cut_site_visible, tissue_visible). The
        # 'fix' is lost on every closure (like SimWorld) and on front_hidden ticks, and is
        # restored by a successful clear_view back-off.
        self.vision = vision
        # FakeWorld's retract_scissors models the jaw-preserving tool_pose park; tests
        # can set 'shaft' explicitly to exercise the refuse-before-motion endpoint guard.
        self.align_mode = 'tool_pose'
        self.awaiting_new_slit = False
        self.fix_lost = False
        self.tick = 0
        self.retraction = 0.0
        self.base_strain = 0.08 if cut_only else 0.0       # anchors pre-tension the cut-only sheet
        self.k = 60.0                                       # strain per metre of retraction
        self.grasped = {}
        self.points = {}          # arm -> grasp point actually held (handover changes the holder, not the side)
        self.ever_grasped = set()
        self.blades_open = False
        self.aligned = False
        self.cut = 0.0
        self.pending = 0.0
        self.camera_moves = 0
        self.closures_while_hidden = 0
        self.at_ready = True          # cutter starts at the measured ready pose; motion leaves it
        self.actions = []

    # -- sensing ----------------------------------------------------------------------------------------------
    def hidden(self):
        o = self.e.get('occlude')
        if o and o[0] <= self.tick <= o[1]:
            return True
        start = self.e.get('occlude_until_camera')
        return start is not None and self.tick >= start and self.camera_moves == 0

    def observe(self, *, inject_events=True):
        self.tick += 1
        if inject_events:
            for arm, t in self.e.get('slip', {}).items():
                if self.tick == t:
                    self.grasped[arm] = 0
                    self.points.pop(arm, None)
            if self.e.get('relax') == self.tick:
                self.retraction -= 0.0005
                self.base_strain = max(0.0, self.base_strain - 0.07)
        hidden = self.hidden()
        drift = self.e.get('drift')
        offset = 0.6 if drift is not None and self.tick >= drift and self.camera_moves == 0 else 0.1
        strain = max(0.0, self.base_strain + self.k * max(self.retraction, 0.0)) if (self.grasped or self.cut_only) else 0.0
        faults = ['psm2 shaft / Patient_HipBone'] if self.e.get('fault') == self.tick else []
        obs = Observation(cut_site_visible=not hidden, scissors_visible=not hidden, cut_site_offset=None if hidden else offset,
                          grasped={a: n for a, n in self.grasped.items()}, tension=strain, max_strain=strain,
                          torn=1 if self.e.get('tear') is not None and self.tick >= self.e['tear'] else 0,
                          cut_length_m=self.cut, faults=faults)
        if self.vision:
            # The remembered front is lost on hidden/front_hidden ticks and after a
            # closure voids the fix; a back-off (clear_view) restores fix_lost.
            front_known = not hidden and not self.fix_lost and \
                self.tick not in self.e.get('front_hidden', set())
            obs.cut_site_visible = front_known
            obs.tissue_visible = not hidden
            obs.cut_progress_m = None if not front_known else self.cut + self.e.get('progress_bias_m', 0.0)
            obs.cut_site_offset = None if not front_known else offset
        return obs

    # -- acting -----------------------------------------------------------------------------------------------
    def check_roles(self):
        return (True, '') if self.e.get('roles_ok', True) else (False, 'psm2: identified 420006 lacks [cut]')

    def home(self):
        return True

    def hold(self):
        self.actions.append(('hold', self.tick))

    def safe_stop(self, reason):
        self.actions.append(('safe_stop', self.tick, reason))

    def localize(self):
        if self.hidden():
            return None
        return Target(cut_start=(0, 0, 0), grasp_points={'psm1': 'left flap', 'psm3': 'right flap'})

    def grasp(self, arm, point):
        if arm in self.e.get('unreachable', set()):
            return False, 'out of reach'
        if arm in self.ever_grasped and arm in self.e.get('regrasp_fails', set()):
            return False, 'jaws close on nothing'
        self.grasped[arm] = 6
        self.points[arm] = point
        self.ever_grasped.add(arm)
        self.actions.append(('grasp', arm, point, self.tick))
        return True, ''

    def release(self, arm):
        self.grasped.pop(arm, None)
        self.points.pop(arm, None)
        self.actions.append(('release', arm, self.tick))

    def takeover_point(self, arm, lost_point):
        if arm in self.e.get('takeover_unreachable', set()):
            return None
        return lost_point

    def retract(self, arm, step):
        if arm not in self.points:
            return False        # same contract as SimWorld: refuse a pull with no recorded held point
        self.retraction += step / max(1, len(self.grasped))
        self.actions.append(('retract', arm, step, self.tick))
        return True

    def open_scissors(self):
        self.blades_open = True
        self.actions.append(('open_scissors', self.tick))
        return True

    def jaws_closed(self):
        return not self.blades_open

    def at_ready_pose(self):
        """The cutter is at the ready pose; 'ready_broken' keeps the park refused even after a
        reported recovery motion (models a mismatched or unreachable anchor)."""
        return self.at_ready and not self.e.get('ready_broken', False)

    def clear_view(self, d, prefer_park=False):
        """CLEAR_VIEW back-off: a working one restores the remembered front fix."""
        self.actions.append(('clear_view', d, self.tick))
        if self.e.get('clear_view_refused'):
            return False
        self.at_ready = False               # a real back-off leaves the ready pose
        if not self.e.get('clear_view_useless'):
            self.fix_lost = False
        return True

    def align_scissors(self, cut_start, progress):
        self.aligned = True
        self.at_ready = False
        return 0.0005

    def advance_scissors(self, d):
        self.pending = d
        self.at_ready = False
        return True

    def closure_aligned(self):
        return self.aligned

    def endpoint_advance_budget(self):
        return 1.0

    def endpoint_travel(self, remaining):
        return remaining

    def close_scissors(self):
        if self.hidden():
            self.closures_while_hidden += 1
        if self.blades_open and self.aligned:
            self.cut += self.pending
        self.pending = 0.0
        self.blades_open = False
        if self.vision:
            self.fix_lost = True                # like SimWorld: a closure voids the front fix
        return True

    def retract_scissors(self):
        self.actions.append(('retract_scissors', self.tick))
        if self.e.get('park_fails', False):
            return False
        self.at_ready = True              # an accepted park restores the ready pose
        return True

    def adjust_camera(self):
        if self.camera_moves >= self.e.get('camera_poses', 3):
            return False
        self.camera_moves += 1 if not self.e.get('camera_useless') else 0
        self.actions.append(('camera', self.tick))
        return True
