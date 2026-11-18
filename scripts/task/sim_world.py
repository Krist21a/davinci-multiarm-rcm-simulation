"""CoppeliaSim world for scripts/task/cut_fsm.py, on scenes/davinci_si_instruments.ttt.

Sensing
  - endoscope RGB-D (EndoscopeCamera): the known cut front and scissors tip pass a simulation visibility gate
    only when their central 3x3 depth window and enough surrounding pixels have nothing over 3 mm nearer;
    colour never overrides depth. This uses ground-truth projections, NOT visual localization;
    the cut-site offset is its pixel distance from the image centre (1 = image edge);
  - tissue model (src/tissue/tissue.lua): grasped nodes per arm, maximum strain, torn constraints;
  - cut length: rows of the sheet separated along the cut line (node geometry, not vision);
  - faults: RCM solver failures or RCM drift above 0.5 mm.
Acting goes through the scene controller only (setRCMTarget, setGrip, tissueGrasp ...), so every motion is
checked by the guarded RCM controller (limits, IK, mesh collisions).

Modes: 'full' asks the retraction arms to grasp the sheet edges; 'cut_only' leaves the sheet on its fixed
anchors. Arm roles come from cfg['roles'] (retract_primary, cut, retract_assist, view).
Injected events, mirroring fake_world.py semantics (details in SimWorld._events): an occluding sphere between
the endoscope and the cut site, physical grasp slips, regrasp failures, and retractor creep.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'build'))
from cut_fsm import Observation, Target  # noqa: E402
from ground_truth import GroundTruth  # noqa: E402
from simtools import ENDO_CONFIG, INSTRUMENT_SCENE, aliases, load_scene, matrix_to_np, read_json  # noqa: E402
from visibility import depth_visible  # noqa: E402

import workspace  # noqa: E402
from workspace import H, NU, NV, PRETENSION  # noqa: E402

TISSUE_RGB = (0.86, 0.42, 0.48)   # flat colour of the sheet (src/tissue/tissue.lua default)
GRASP_RADIUS_M = 0.0025      # nodes this close to the closed jaw line are held: about half the 5-7 mm jaw width


def unit(v):
    return v / np.linalg.norm(v)


class SimWorld:
    def __init__(self, client, sim, args, cfg, mode='cut_only', events=None,
                 event=None, event_tick=6, event_ticks=3):
        from simtools import refresh_controller

        self.client, self.sim, self.cfg, self.mode = client, sim, cfg, mode
        roles = cfg.get('roles', {})
        self.primary, self.assist = roles.get('retract_primary', 'psm1'), roles.get('retract_assist', 'psm3')
        self.cutter, self.view = roles.get('cut', 'psm2'), roles.get('view', 'ecm')
        # New-style event dict; the legacy single-event arguments map onto it unchanged.
        self.events = dict(events or {})
        if event == 'occlusion_persistent':
            self.events.setdefault('occlude_until_camera', event_tick)
        elif event == 'occlusion_transient':
            self.events.setdefault('occlude', (event_tick, event_tick + event_ticks - 1))
        self.refused_failures = {}       # arm -> solver-failure increments caused by refused candidates
        self.ever_grasped = set()
        self.event_log = []
        self.timing = {}
        rcm_file = ENDO_CONFIG.parent / 'instruments' / 'rcm_config.json'      # written with the scene by install
        self.rcm = read_json(rcm_file if rcm_file.exists() else ENDO_CONFIG / 'rcm_config.json')
        self.task = read_json(ENDO_CONFIG / 'task.json')
        self.endo = read_json(ENDO_CONFIG / 'endoscope.json')
        self.index = {a['arm']: i for i, a in enumerate(self.rcm, 1)}
        self.arm_cfg = {a['arm']: a for a in self.rcm}
        load_scene(client, sim, args, getattr(args, 'scene', INSTRUMENT_SCENE), 'cut_task.ttt')
        self.by = aliases(sim)
        self.ctrl = self.by['RCMController']
        # Refresh only the stopped, loaded scene; never save the .ttt. In particular,
        # homeRCM must execute the current controller's explicit success/failure contract.
        self.controller_sha256 = refresh_controller(sim, self.ctrl, tissue=True)
        self.truth = GroundTruth(self)  # sensing reads; actuation keeps direct access (see ground_truth.py)
        self.cam = self.by[self.endo['camera']['alias']]
        c = self.endo['camera']
        self.W, self.H = c['resolution']
        self.f = (self.W / 2) / math.tan(math.radians(c['view_angle_deg']) / 2)
        self.far = c['far_m']
        self.tick = 0
        self.occluder = None
        self.advanced = 0.0
        self.camera_moves = 0
        self.image = None
        self.sheet_origin = None
        # The grasp point each arm actually holds. Handover can put an arm on either side of the
        # cut line, so retraction direction must come from this record, not from the arm's name.
        self.grasped_points = {}
        # Fixed side anchors pin the sheet at its laid-out width; with 8 % the constraint where the free near edge
        # meets the first anchor tore at 52 % strain before any cut, so the cut-only sheet uses 5 %.
        self.pretension = 0.05 if mode == 'cut_only' else PRETENSION
        # The bead and ring of the earlier pick-and-place task sit right at the cut site; hide them for this task
        # (staged copy only, the scene file is not changed).
        # The 6 mm magenta wrist marker of that task covers the cut site once the scissors advance.
        for n in ('TaskBead', 'TaskTarget', f'{self.cutter}_TipMarker'):
            if n in self.by:
                sim.setObjectInt32Param(self.by[n], sim.objintparam_visibility_layer, 0)
        self.start()

    # -- helpers ----------------------------------------------------------------------------------------------
    def _time(self, name, fn, *a):
        t0 = time.perf_counter()
        try:
            return fn(*a)
        finally:
            self.timing.setdefault(name, []).append(time.perf_counter() - t0)

    def call(self, name, *a):
        return self._time('call:' + name, self.sim.callScriptFunction, name, self.ctrl, *a)

    def steps(self, n):
        for _ in range(n):
            self._time('sim_step', self.sim.step)

    def timing_summary(self):
        out = {}
        for name, times in self.timing.items():
            a = np.asarray(times, dtype=float)
            out[name] = dict(count=int(a.size), total_s=float(a.sum()), mean_s=float(a.mean()),
                             p95_s=float(np.percentile(a, 95)))
        return out

    def frame(self, arm):
        j9, j10 = (matrix_to_np(self.sim.getObjectMatrix(self.by[f'{arm}_joint{n}'], -1)) for n in (9, 10))
        if arm + '_dvrk_yaw_frame' in self.by:
            yaw = matrix_to_np(self.sim.getObjectMatrix(self.by[arm + '_dvrk_yaw_frame'], -1))
            return dict(pivot=j10[:, 3], x=yaw[:, 1], axis=yaw[:, 2],
                        tcp=np.array(self.sim.getObjectPosition(self.by[f'{arm}_tcp'], -1)))
        return dict(pivot=j10[:, 3], x=unit(j10[:, 3] - j9[:, 3]), axis=j10[:, 2],
                    tcp=np.array(self.sim.getObjectPosition(self.by[f'{arm}_tcp'], -1)))

    def target_of(self, arm):
        return self.call('getRCMStats')[self.index[arm] - 1]['target']

    def _solver_failures(self, arm):
        try:
            return self.call('getRCMStats')[self.index[arm] - 1].get('failures', 0)
        except Exception:
            return None

    def command(self, arm, yaw, pitch, ins, n=15):
        """Advance settling time only for an accepted, synchronous controller command.

        A refused candidate rolls back in the controller but still increments its failure counter, so a
        refusal's increments are tracked separately and never reported as arm faults. Increments under an
        accepted command stay faults."""
        before = self._solver_failures(arm)
        ok = bool(self.call('setRCMTarget', self.index[arm], float(yaw), float(pitch), float(ins)))
        if ok:
            self.steps(n)
            return ok
        after = self._solver_failures(arm)
        if before is not None and after is not None:
            self.refused_failures[arm] = self.refused_failures.get(arm, 0) + (after - before)
        return ok

    def rcm_command_for(self, arm, point):
        """yaw, pitch, insertion that would put this arm's tool tip on `point` (fixed-RCM geometry)."""
        c = self.arm_cfg[arm]
        r, a0, tx, ty = (np.array(c[k]) for k in ('rcm_world', 'axis_home', 'transverse_x', 'transverse_y'))
        j8 = matrix_to_np(self.sim.getObjectMatrix(self.by[arm + '_joint8'], -1))
        tcp_len = float(np.dot(np.array(self.sim.getObjectPosition(self.by[arm + '_tcp'], -1)) - j8[:, 3], j8[:, 2]))
        d = unit(point - r)
        yaw = math.degrees(math.atan2(d @ tx, d @ a0))
        pitch = math.degrees(math.atan2(d @ ty, d @ a0))
        ins = c['insertion_home'] + np.linalg.norm(point - r) - (tcp_len - c['rcm_distance_home'])
        return yaw, pitch, ins

    def camera(self):
        M = matrix_to_np(self.sim.getObjectMatrix(self.cam, -1))
        return M[:, :3], M[:, 3]

    def render(self):
        # Offline safety tests need only numpy, not Pillow or the geometry/Remote API stack.
        from PIL import Image

        t0 = time.perf_counter()
        try:
            s = self.sim
            s.handleVisionSensor(self.cam)
            raw, _ = s.getVisionSensorImg(self.cam)
            d, _ = s.getVisionSensorDepth(self.cam, 1)
            self.image = Image.frombytes('RGB', (self.W, self.H), bytes(raw)).transpose(Image.FLIP_TOP_BOTTOM)
            self.depth = np.array(s.unpackFloatTable(d), dtype=float).reshape(self.H, self.W)[::-1]
        finally:
            self.timing.setdefault('render', []).append(time.perf_counter() - t0)

    def project(self, p):
        R, t = self.camera()
        pc = R.T @ (np.asarray(p) - t)
        if pc[2] <= 0:
            return None, pc
        return (self.W / 2 - self.f * pc[0] / pc[2], self.H / 2 - self.f * pc[1] / pc[2]), pc

    def point_visible(self, p, tol=0.003, window=21, min_clear=0.35):
        """Depth visibility of a known simulation point, not an image-derived position estimate."""
        uv, pc = self.project(p)
        if (uv is None or not np.isfinite(pc).all() or not 0 < pc[2] < self.far or
                not (2 <= uv[0] < self.W - 2 and 2 <= uv[1] < self.H - 2)):
            return False, None
        offset = max(abs(uv[0] - self.W / 2) / (self.W / 2), abs(uv[1] - self.H / 2) / (self.H / 2))
        visible = depth_visible(self.depth, uv, pc[2], tol=tol, window=window, min_clear=min_clear)
        return visible, float(offset)

    # Ground-truth sensing reads; the implementations live in GroundTruth (scripts/task/ground_truth.py)
    # and count themselves for the report's truth_reads.
    def cut_front(self):
        """Where the next closure has to cut: on the sheet as it is now, between the two centre columns."""
        return self.truth.cut_front()

    def nodes(self):
        return self.truth.nodes()

    def cut_line_tension(self):
        """Median strain of the constraints across the cut line (columns NU/2-1 and NU/2), still uncut.

        This is the tension the retraction has to create for the cut; the whole-sheet median stays near zero
        when two graspers pull a sheet that is attached along one edge only."""
        return self.truth.cut_line_tension()

    def cut_length(self):
        """Length of the cut from the near edge: rows whose band to the next row is cut through.

        Row j counts when, for some column gap (i, i+1) near the centre line, the horizontal constraint of row j
        and both diagonals of cell (i, j) are cut, so nothing joins the two sides there. Counted from row 0 and
        stopping at the first row that still holds."""
        return self.truth.cut_length()

    # -- setup ------------------------------------------------------------------------------------------------
    def start(self):
        s = self.sim
        s.setStepping(True)
        s.setBoolParam(s.boolparam_realtime_simulation, False)
        s.startSimulation()
        self.steps(1)

    def place_sheet(self):
        f = self.frame(self.cutter)
        L = workspace.layout(f['pivot'], f['x'], f['axis'], f['tcp'])
        self.v, self.u, self.sheet_origin, self.grasp_points = L['v'], L['u'], L['origin'], L['grasp_points']
        self.call('tissueCreate', dict(origin=self.sheet_origin.tolist(), u=self.u.tolist(), v=self.v.tolist(), nu=NU, nv=NV,
                                       spacing=H, pretension=self.pretension, anchor_sides=self.mode == 'cut_only',
                                       anchor_from_row=3, grasp_radius=GRASP_RADIUS_M))
        self.steps(30)

    # -- World interface --------------------------------------------------------------------------------------
    def observe(self):
        self.tick += 1
        self._events()
        self.render()
        if self.sheet_origin is None:
            cut_vis, offset = False, None
        else:
            cut_vis, offset = self.point_visible(self.cut_front())
        tip_vis, _ = self.point_visible(self.truth.tcp(self.cutter))
        st = self.truth.tissue_stats()
        rs = self.call('getRCMStats')
        # A refused candidate increments the controller failure counter even though it rolled back; only
        # failures beyond those recorded at refusal are arm faults. Drift faults are unchanged.
        faults = [f"{r['arm']}: {r['failures'] - self.refused_failures.get(r['arm'], 0)} RCM solver failures"
                  for r in rs if r['failures'] - self.refused_failures.get(r['arm'], 0) > 0]
        faults += [f"{r['arm']}: RCM drift {r['max_drift_m'] * 1000:.2f} mm" for r in rs if r['max_drift_m'] > 5e-4]
        return Observation(cut_site_visible=cut_vis, scissors_visible=tip_vis, cut_site_offset=offset if cut_vis else None,
                           grasped=dict(st.get('grasped') or {}), tension=self.cut_line_tension(), max_strain=float(st.get('max_strain', 0.0)),
                           torn=int(st.get('torn', 0)), cut_length_m=self.cut_length(), faults=faults)

    def check_roles(self):
        import instrument_guard
        import instruments as inst

        ok, report = instrument_guard.check(self.sim, self.by, self.rcm, inst.roles())
        bad = [f"{a}: {e['identified']} missing {e['missing']}" for a, e in report.items() if not e['ok']]
        self.roles_report = {a: dict(identified=e['identified'], ok=e['ok']) for a, e in report.items()}
        return ok, '; '.join(bad)

    def home(self):
        # Legacy controllers returning None cannot establish that homing succeeded.
        if not self.call('homeRCM'):
            return False
        self.steps(20)
        for ready in self.task['planning']['command_sequence_checked'][:2]:
            if not self.call('setRCMTarget', self.index[self.cutter], *map(float, ready)):
                return False
        self.steps(30)
        self.place_sheet()
        return True

    def hold(self):
        self.call('hold')

    def safe_stop(self, reason):
        self.call('hold')

    def localize(self):
        """Simulation-ground-truth targets, gated by depth and a coarse colour-presence check.

        Neither the RGB count nor the projected depth gate estimates a target pose from vision.
        """
        self.render()
        # Gate on the live cut front, not the recorded origin: the sheet deforms under retraction and
        # cutting, so the origin point can float off the rendered surface and fail the depth gate even
        # though the work site is plainly in view.
        vis, _ = self.point_visible(self.cut_front())
        if not vis:
            return None
        # Coarse sheet-colour presence in the image; this cannot override the depth gate above.
        rgb = np.asarray(self.image, dtype=float)
        tissue = np.linalg.norm(rgb - 255 * np.array(TISSUE_RGB), axis=2) < 40
        if tissue.sum() < 30:
            return None
        # Grasp points from the sheet as it is now (it contracts under pre-tension once the sides are free):
        # one node in from each side edge, at mid length. Primary takes the -u side, assist the +u side.
        n = self.nodes()
        j = NV // 2
        self.grasp_points = {self.primary: n[j, 1], self.assist: n[j, NU - 2]}
        return Target(cut_start=self.sheet_origin, grasp_points=self.grasp_points)

    def grasp(self, arm, point):
        yaw, pitch, ins = self.rcm_command_for(arm, np.asarray(point))
        L = self.arm_cfg[arm]['limits']
        if abs(yaw) > L['yaw_deg'] or abs(pitch) > L['pitch_deg'] or not L['insertion_min'] <= ins <= L['insertion_max']:
            return False, (f'needs yaw {yaw:.1f} deg, pitch {pitch:.1f} deg, insertion {ins * 1000:.1f} mm; limits '
                           f"+-{L['yaw_deg']} / +-{L['pitch_deg']} deg, {L['insertion_min'] * 1000:.0f}-{L['insertion_max'] * 1000:.0f} mm")
        i = self.index[arm]
        if arm in self.ever_grasped and arm in self.events.get('regrasp_fails', ()):
            return False, 'injected regrasp failure'      # before any jaw motion, like FakeWorld
        # A grasp attempt opens the jaws, which already voids any earlier hold on the sheet.
        self.grasped_points.pop(arm, None)
        if not self.call('setGrip', i, 0.5):
            return False, 'jaws cannot open'
        if not self.command(arm, yaw, pitch, ins, 30):
            return False, 'approach refused by the guarded controller'
        # The jaw tip is off the shaft axis for curved tips: correct the remaining tip error with relative commands.
        for _ in range(3):
            tip = np.array(self.sim.getObjectPosition(self.by[arm + '_tcp'], -1))
            if np.linalg.norm(tip - point) < 2e-4:
                break
            c = np.array(self.target_of(arm)) + np.array(self.rcm_command_for(arm, point)) - np.array(self.rcm_command_for(arm, tip))
            if not self.command(arm, *c, 30):
                return False, 'approach correction refused by the guarded controller'
        if not self.call('setGrip', i, 0.0):
            return False, 'jaws cannot close'
        n = self.call('tissueGrasp', i)
        if n is None or n < self.cfg['grasp']['min_nodes']:
            self.release(arm)       # tissueGrasp may already have pinned an insufficient subset of nodes
            return False, f'grasped {n} nodes'
        self.grasped_points[arm] = np.asarray(point, dtype=float).copy()
        self.ever_grasped.add(arm)
        return True, ''

    def release(self, arm):
        self.call('tissueRelease', self.index[arm])
        self.grasped_points.pop(arm, None)

    def takeover_point(self, arm, lost_point):
        """Nearest live node to `lost_point` the arm can grasp on the same side of the cut line, or None.

        A handover keeps the side the lost arm held: candidates sit at least three columns in from
        the cut line, off the edge column and row 0, and inside the arm limits with a small margin
        (a point at the boundary would likely be refused by the guarded controller anyway)."""
        if self.sheet_origin is None or lost_point is None:
            return None
        origin = np.asarray(self.sheet_origin, dtype=float)
        lost = np.asarray(lost_point, dtype=float)
        side = float(np.sign(np.dot(lost - origin, self.u)))
        if not math.isfinite(side) or side == 0.0:
            return None        # a point on the cut line gives no side to keep
        # The cut line runs between columns NU//2-1 and NU//2.
        lo, hi = (1, NU // 2 - 4) if side < 0 else (NU // 2 + 3, NU - 2)
        L = self.arm_cfg[arm]['limits']
        nodes = self.nodes()
        best, best_d = None, np.inf
        for j in range(1, NV):
            for i in range(lo, hi + 1):
                node = np.asarray(nodes[j, i], dtype=float)
                if np.sign(np.dot(node - origin, self.u)) != side:
                    continue
                yaw, pitch, ins = self.rcm_command_for(arm, node)
                if (abs(yaw) > L['yaw_deg'] - 0.5 or abs(pitch) > L['pitch_deg'] - 0.5 or
                        not L['insertion_min'] + 1e-3 <= ins <= L['insertion_max'] - 1e-3):
                    continue
                d = float(np.linalg.norm(node - lost))
                if d < best_d:
                    best, best_d = node, d
        return best

    def retract(self, arm, step):
        """Pull (step > 0) or relax the grasped edge outward in the sheet plane, away from the cut line.

        Pulling back along the shaft would lift the sheet out of its plane and load only the few grasped nodes
        (that tore the sheet at 47 % strain in the first run); a lateral pull spreads the tension across it."""
        point = self.grasped_points.get(arm)
        if self.sheet_origin is None or point is None:
            return False        # no verified grasp side: refuse rather than pull in a guessed direction
        # The record can be stale after a physical slip: also require a live hold in the tissue model.
        stats = self.truth.tissue_stats()
        held = (stats.get('grasped') or {}).get(arm, 0) if isinstance(stats, dict) else 0
        if held < self.cfg['grasp']['min_nodes']:
            return False
        # Outward = away from the cut line, on the side this arm actually holds. The arm's name
        # cannot give this: after a handover psm3 can hold the -u point.
        side = float(np.sign(np.dot(np.asarray(point, dtype=float) - np.asarray(self.sheet_origin, dtype=float), self.u)))
        if not math.isfinite(side) or side == 0.0 or not math.isfinite(step):
            return False        # a grasp on the cut line has no unambiguous outward direction
        tip = np.array(self.sim.getObjectPosition(self.by[arm + '_tcp'], -1))
        # Relative command: the jaw tip of a curved instrument is off the shaft axis, so an absolute
        # point command would jump by that offset (1.4 mm for the Maryland tip).
        c0, c1 = np.array(self.rcm_command_for(arm, tip)), np.array(self.rcm_command_for(arm, tip + side * step * self.u))
        yaw, pitch, ins = np.array(self.target_of(arm)) + (c1 - c0)
        L = self.arm_cfg[arm]['limits']
        if abs(yaw) > L['yaw_deg'] or abs(pitch) > L['pitch_deg'] or not L['insertion_min'] <= ins <= L['insertion_max']:
            return False
        ok = self.command(arm, yaw, pitch, ins, 10)
        # `side` is +1/-1 along u for the held point, so step*side is the signed pull direction in u.
        self.event_log.append(dict(tick=self.tick, event='retract', arm=arm, step=float(step), side=float(side),
                                   accepted=bool(ok)))
        return ok

    def open_scissors(self):
        ok = bool(self.call('setGrip', self.index[self.cutter], 0.5))
        if ok:
            self.steps(3)
        return ok

    def close_scissors(self):
        ok = bool(self.call('setGrip', self.index[self.cutter], 0.0))
        if ok:
            self.steps(8)
        return ok

    def align_scissors(self, cut_start, progress):
        """Lateral distance of the cut front from the blade line, or None if approach was refused."""
        f = self.frame(self.cutter)
        p = self.cut_front()
        d = unit(f['tcp'] - f['pivot'])
        lateral = np.linalg.norm(np.cross(p - f['pivot'], d))
        if np.dot(p - f['tcp'], self.v) > 0.0025:            # tip more than 2.5 mm short of the cut front
            yaw, pitch, ins = self.target_of(self.cutter)
            correction = np.dot(p - f['tcp'], self.v) - 0.001
            if not self.command(self.cutter, yaw, pitch, ins + correction, 10):
                return None
            self.advanced += correction       # withdrawal must undo this accepted insertion correction too
        return float(lateral)

    def advance_scissors(self, d):
        yaw, pitch, ins = self.target_of(self.cutter)
        ok = self.command(self.cutter, yaw, pitch, ins + d, 15)
        if ok:
            self.advanced += d
        return ok

    def retract_scissors(self):
        """Undo accepted cutting advances, without an extra bite merely to close for retreat.

        Normal completion leaves the scissors closed. If they are open with tissue present,
        closing could cut again: refuse instead of inventing an unverified escape trajectory.
        """
        if not math.isfinite(self.advanced) or self.advanced < 0:
            return False
        if self.advanced == 0:
            return True
        state = self.call('getRCMStats')[self.index[self.cutter] - 1]
        grip = state.get('grip')
        if grip is None or not math.isfinite(grip):
            return False
        if grip != 0.0:
            # Missing tissue status is not evidence that an additional closure is safe.
            tissue = self.truth.tissue_stats()
            if not isinstance(tissue, dict) or tissue.get('exists') is not False:
                return False
            if not self.call('setGrip', self.index[self.cutter], 0.0):
                return False
        yaw, pitch, ins = state['target']
        if not self.command(self.cutter, yaw, pitch, ins - self.advanced, 20):
            return False
        self.advanced = 0.0       # a second call must not retract the same distance again
        return True

    CAMERA_POSES = [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0), (4.5, 3.0), (-4.5, -3.0)]

    def adjust_camera(self):
        if self.camera_moves >= len(self.CAMERA_POSES):
            return False
        yaw, pitch = self.CAMERA_POSES[self.camera_moves]
        self.camera_moves += 1
        ins = self.arm_cfg[self.view]['insertion_home']
        return self.command(self.view, yaw, pitch, ins, 25)

    # -- injected events --------------------------------------------------------------------------------------
    def _events(self):
        """Event dict semantics, mirroring FakeWorld (scripts/task/fake_world.py):
          occlude=(start, end)              an occluding sphere blocks the view for those ticks
          occlude_until_camera=start        it blocks the view from `start` until the camera has moved
          slip={arm: tick}                  that arm's tissue hold is released at that tick, jaws left
                                            closed; grasped_points stays stale — only a live check sees it
          regrasp_fails={arm,...}           that arm cannot grasp again after its first success
          relax=tick                        retractor creep: every arm holding tissue is commanded
                                            relax_m toward the cut line (e['relax_m'] if given,
                                            else cfg['events']['relax_m'])
        """
        e = self.events
        if not e:
            return
        occlude, until = e.get('occlude'), e.get('occlude_until_camera')
        if self.occluder is None and (until == self.tick or (occlude is not None and self.tick == occlude[0])):
            R, t = self.camera()
            p = self.cut_front()
            ray = unit(p - t)
            near = until is not None
            dist, radius = (0.015, 0.0025) if near else (0.06, 0.004)
            h = self.sim.createPrimitiveShape(self.sim.primitiveshape_spheroid, [2 * radius] * 3, 0)
            self.sim.setObjectAlias(h, 'InjectedOccluder')
            self.sim.setObjectPosition(h, -1, (t + dist * ray).tolist())
            self.sim.setShapeColor(h, None, self.sim.colorcomponent_ambient_diffuse, [0.55, 0.55, 0.5])
            self.sim.setObjectInt32Param(h, self.sim.shapeintparam_static, 1)
            self.sim.setObjectInt32Param(h, self.sim.shapeintparam_respondable, 0)
            self.occluder = h
            self.event_log.append(dict(tick=self.tick, event='occlude', persistent=near))
        if occlude is not None and self.occluder not in (None, -1) and self.tick > occlude[1]:
            self.sim.removeObjects([self.occluder])
            self.occluder = -1
        for arm, at in sorted(e.get('slip', {}).items()):
            if self.tick == at:
                # Physical slip: the hold is gone though the jaws stay closed; the stale record is kept on
                # purpose so only a live-hold check (retract, grasp supervision) can notice.
                self.call('tissueRelease', self.index[arm])
                self.event_log.append(dict(tick=self.tick, event='slip', arm=arm))
        if e.get('relax') == self.tick:
            relax_m = e.get('relax_m', self.cfg.get('events', {}).get('relax_m', 0.0005))
            for arm in list(self.grasped_points):
                ok = self.retract(arm, -relax_m)
                self.event_log.append(dict(tick=self.tick, event='relax', arm=arm, relax_m=relax_m,
                                           accepted=bool(ok)))
