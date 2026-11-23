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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vision'))
from cut_fsm import Observation, Target  # noqa: E402
from ground_truth import GroundTruth  # noqa: E402
from seam_topology import SeamTruth, material_state  # noqa: E402
from simtools import ENDO_CONFIG, INSTRUMENT_SCENE, REPO, TISSUE_SRC, aliases, load_scene, matrix_to_np, read_json  # noqa: E402
from simtools import sha256 as file_sha256, write_json  # noqa: E402
from visibility import depth_visible  # noqa: E402

import cut_perception  # noqa: E402
import cutter_view  # noqa: E402
import tool_pose  # noqa: E402
import workspace  # noqa: E402
from workspace import H, NU, NV, PRETENSION  # noqa: E402

TISSUE_RGB = (0.86, 0.42, 0.48)   # flat colour of the sheet (src/tissue/tissue.lua default)
GRASP_RADIUS_M = 0.0025      # nodes this close to the closed jaw line are held: about half the 5-7 mm jaw width
CUT_RADIUS_M = 0.0003       # src/tissue/tissue.lua default, also used in the visual sweep guard
# Frozen tissue controller basis pinned by the historical captures; locked collectors opt in
# explicitly. The current src/tissue/tissue.lua stays the default for new development runs.
LEGACY_TISSUE = TISSUE_SRC / 'legacy' / 'tissue_v1.lua'


def sweep_projection(pivot, axis, tcp, direction):
    """Farthest projected point of the same sampled sector used by tissueCut."""
    line = tcp - pivot
    axial = axis * (axis @ line)
    angles = np.linspace(0.0, 0.5, 11)
    samples = (pivot + axial + np.cos(angles)[:, None] * (line - axial)
               + np.sin(angles)[:, None] * np.cross(axis, line))
    return max(float(pivot @ direction), float(np.max(samples @ direction)))


def unit(v):
    return v / np.linalg.norm(v)


class SimWorld:
    # Defaults for the M5b vision state so SimWorld.__new__ test stubs behave as truth mode;
    # __init__ always overwrites them.
    vision = False
    canonical_seam = False
    _fix = None
    _fix_epoch = 0
    _fix_max_age = 6
    _last_est = None
    _pose_err_fixed = None
    _in_event = False
    awaiting_new_slit = False

    SHEET_KEYS = frozenset(('seam_mode', 'guide_band_m', 'ruler_print', 'indexed_print',
                            'visual_gap_m', 'visual_gap_style', 'visual_gap_start_row',
                            'bridge_print', 'bridge_fracture_faces', 'roll_deg'))

    @staticmethod
    def _check_sheet_cfg(sheet):
        """One input: cfg['sheet']. Validate every key up front; unknown or ill-typed
        options are refused, never silently ignored."""
        if sheet is None:
            return {}
        if not isinstance(sheet, dict):
            raise ValueError("cfg['sheet'] must be a dict")
        unknown = set(sheet) - SimWorld.SHEET_KEYS
        if unknown:
            raise ValueError(f'unknown sheet option(s): {sorted(unknown)}')
        mode = sheet.get('seam_mode')
        if mode is not None and mode != 'surface':
            raise ValueError(f'unknown sheet seam_mode {mode!r}')
        if ('guide_band_m' in sheet or 'ruler_print' in sheet or 'indexed_print' in sheet) and mode != 'surface':
            raise ValueError("sheet guide_band_m/ruler_print/indexed_print require seam_mode='surface'")
        if 'ruler_print' in sheet and type(sheet['ruler_print']) is not bool:
            raise ValueError('sheet.ruler_print must be a boolean')
        if 'indexed_print' in sheet and type(sheet['indexed_print']) is not bool:
            raise ValueError('sheet.indexed_print must be a boolean')
        gap = sheet.get('visual_gap_m')
        if gap is not None and (type(gap) not in (int, float) or not math.isfinite(gap)
                                or not 0 < gap <= 0.0004 or mode is not None):
            raise ValueError('sheet.visual_gap_m requires the conventional flat sheet and <= 0.4 mm')
        if 'visual_gap_style' in sheet and (sheet['visual_gap_style'] != 'edge'
                                            or gap is None):
            raise ValueError('sheet.visual_gap_style requires visual_gap_m and edge mode')
        if 'visual_gap_start_row' in sheet and (type(sheet['visual_gap_start_row']) is not int
                                               or sheet['visual_gap_start_row'] != 7
                                               or sheet.get('visual_gap_style') != 'edge'):
            raise ValueError('sheet.visual_gap_start_row requires the row-7 edge specimen')
        if 'bridge_print' in sheet and (sheet['bridge_print'] is not True or gap is not None
                                         or mode is not None):
            raise ValueError('sheet.bridge_print requires the flat conventional marked sheet')
        if 'bridge_fracture_faces' in sheet and (
                sheet['bridge_fracture_faces'] is not True or not sheet.get('bridge_print')):
            raise ValueError('sheet.bridge_fracture_faces requires the physical bridge specimen')
        if 'roll_deg' in sheet:
            roll = sheet['roll_deg']
            if (type(roll) not in (float, int) or not -workspace.MAX_ROLL_DEG <= roll <= workspace.MAX_ROLL_DEG
                    or not sheet.get('bridge_fracture_faces')):
                raise ValueError('sheet.roll_deg requires the v5 fracture specimen and a roll within +/-16 deg')
        if sheet.get('indexed_print') and sheet.get('ruler_print'):
            raise ValueError('conflicting print layouts')
        if 'guide_band_m' in sheet:
            g = sheet['guide_band_m']
            if type(g) is bool or not isinstance(g, (int, float)) or not math.isfinite(g) or g <= 0:
                raise ValueError('sheet.guide_band_m must be a finite positive number')
        return dict(sheet)

    def _canonical_gate(self, cfg):
        """True when the requested surface-seam coupon may run; refuse every
        incompatible setting instead of silently falling back."""
        if 'visual_gap_m' in self.sheet_cfg or 'bridge_print' in self.sheet_cfg:
            if (self.tissue_source is not None or self.mode != 'full'
                    or self.roi_proxy is not None
                    or not (cfg.get('perception') or {}).get('material_pattern')):
                raise ValueError("sheet.visual_gap_m requires the current full-mode "
                                 'flat marked conventional sheet')
        if self.sheet_cfg.get('seam_mode') != 'surface':
            return False
        if self.tissue_source is not None:
            raise ValueError("sheet seam_mode='surface' requires the current controller, "
                             'not an explicit tissue_source')
        if (self.mode != 'full' or self.vision or self.roi_proxy is not None
                or (cfg.get('perception') or {}).get('material_pattern')):
            raise ValueError("sheet seam_mode='surface' is a full-mode truth-only "
                             'flat-scene option without material_pattern bands')
        return True

    def __init__(self, client, sim, args, cfg, mode='cut_only', events=None,
                 event=None, event_tick=6, event_ticks=3, tissue_source=None):
        from simtools import refresh_controller

        self.client, self.sim, self.cfg, self.mode = client, sim, cfg, mode
        self.args = args
        # Canonical development sheet options live in cfg['sheet'] (gated below to
        # full truth-only flat runs); locked collectors opt into the frozen controller
        # via tissue_source=LEGACY_TISSUE.
        self.sheet_cfg = self._check_sheet_cfg(cfg.get('sheet'))
        self.tissue_source = tissue_source
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
        # Scene-keyed config: config/scenes/<scene_stem>/ overrides the global instrument config
        # for scenes built with their own port/anatomy data; existing scenes fall back unchanged.
        scene = getattr(args, 'scene', INSTRUMENT_SCENE)
        self.scene_file = Path(scene)
        self.scene_cfg_dir = ENDO_CONFIG.parent / 'scenes' / Path(scene).stem
        rcm_file = self.scene_cfg_dir / 'rcm_config.json'
        if not rcm_file.exists():
            self.scene_cfg_dir = None
            rcm_file = ENDO_CONFIG.parent / 'instruments' / 'rcm_config.json'   # written by install
            if not rcm_file.exists():
                rcm_file = ENDO_CONFIG / 'rcm_config.json'
        self.rcm = read_json(rcm_file)
        self.roi_proxy = None
        self.anatomy_transform = None
        if self.scene_cfg_dir is not None:
            for key, fname in (('roi_proxy', 'roi_proxy.json'), ('anatomy_transform', 'anatomy_transform.json')):
                p = self.scene_cfg_dir / fname
                if p.exists():
                    setattr(self, key, read_json(p))
        self.task = read_json(ENDO_CONFIG / 'task.json')
        self.endo = read_json(ENDO_CONFIG / 'endoscope.json')
        self.index = {a['arm']: i for i, a in enumerate(self.rcm, 1)}
        self.arm_cfg = {a['arm']: a for a in self.rcm}
        load_scene(client, sim, args, getattr(args, 'scene', INSTRUMENT_SCENE), 'cut_task.ttt')
        self.by = aliases(sim)
        self.ctrl = self.by['RCMController']
        # Refresh only the stopped, loaded scene; never save the .ttt. In particular,
        # homeRCM must execute the current controller's explicit success/failure contract.
        self.controller_sha256 = refresh_controller(sim, self.ctrl, tissue=True,
                                                    tissue_source=self.tissue_source)
        # M5b: 'vision' drives the cutter from the cut_perception estimate; every GroundTruth
        # read then needs a declared purpose or TruthReadInControl raises (config/task/m5b_protocol.json).
        self.vision = cfg.get('cut', {}).get('control_source', 'truth') == 'vision'
        # The canonical flat development coupon (surface seam) selects the single seam scorer;
        # every other sheet keeps GroundTruth.
        self.canonical_seam = self._canonical_gate(cfg)
        self.truth = SeamTruth(self) if self.canonical_seam else GroundTruth(self)
        if self.vision:
            self.truth.strict = True
        self.cam = self.by[self.endo['camera']['alias']]
        c = self.endo['camera']
        self.W, self.H = c['resolution']
        self.f = (self.W / 2) / math.tan(math.radians(c['view_angle_deg']) / 2)
        self.far = c['far_m']
        self.tick = 0
        self.occluder = None
        self.advanced = 0.0
        # 'shaft': scissors align/advance along the shaft (setRCMTarget). 'tool_pose': the wrist
        # aims the tcp + jaw axes with a single 6-DOF command (setToolPose). _pre_cut_q records
        # the full command before cutting so withdrawal restores the wrist pose too.
        self.align_mode = cfg.get('cut', {}).get('align_mode', 'shaft')
        self._pre_cut_q = None
        # Immutable homed anchors recorded once at the end of home(); _pre_cut_q remains the
        # temporary motion state for the vision withdrawal bookkeeping. Never recaptured.
        self._ready_q = None
        self._ready_tcp = None
        # Measured centre transverse rest lengths per newly created sheet (canonical metrology,
        # not the nominal grid); None when the controller predates the readback (legacy).
        self.cutline_rest_m = None
        # Canonical surface-seam safety monitor: farthest_row_end bound and the separate
        # orphan-release count (never part of the direct cut count).
        self._seam_refs = None
        self.seam_releases = 0
        self.camera_moves = 0
        self.initial_view_readback = None
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
        self.tip_marker_optics = None
        if cfg.get('perception', {}).get('exclude_tcp_marker'):
            self.tip_marker_optics = self._exclude_tcp_marker()
        self.start()
        # M4 shadow perception: staged markers + a logging-only estimator. The estimate reads only
        # the already-acquired image/depth and is appended to the report; it never feeds control.
        self.perc_cfg = cfg.get('perception') or {}
        self.material_pattern = self.perc_cfg.get('material_pattern', False)
        self.local_material_chart = self.perc_cfg.get('local_material_chart', False)
        self.perc_mode = self.perc_cfg.get('mode', 'off')
        self.perc_rng = np.random.default_rng(self.perc_cfg.get('seed', 0))
        self.perc_log = []
        self.protected_log = []
        self.last_protected = {}
        self._perc_estimator = None
        self._marker_tcp_len = None
        self.markers = []
        # M5a frame dump: shadow mode only, repo-relative perception.dump_dir. Per-tick npz plus
        # one meta.json; every read is logging-only (sheet reads go through GroundTruth and raise
        # truth_reads, which the report shows together with the dump_dir in cfg['perception']).
        self._dump_dir = None
        self._dump_meta_written = False
        self._perc_meta = None
        # M5b vision-mode state: the remembered front fix, this tick's estimate, the
        # fixed-mode camera-pose draw, and the injected-event suppression flag.
        self._fix = None
        self._fix_epoch = 0
        self._last_est = None
        self._pose_err_fixed = None
        self._in_event = False
        self._fix_max_age = cfg.get('vision', {}).get('fix_max_age_ticks', 6)
        if self.perc_mode == 'shadow' and self.perc_cfg.get('dump_dir'):
            self._dump_dir = REPO / self.perc_cfg['dump_dir']
            self._dump_dir.mkdir(parents=True, exist_ok=False)
        if self.perc_cfg.get('markers', True):
            self._create_markers()
        if self.vision and not self.perc_cfg.get('markers', True):
            raise RuntimeError('vision control requires perception markers')

    def _exclude_tcp_marker(self):
        # The imported URDF's transparent TCP sphere is not one of the dVRK jaws.
        if (self.cutter != 'psm2' or file_sha256(self.scene_file) !=
                '8bff1b9e5af8c98a969d8ba6d6c2ea722c3e8070540439edeb84e27eda81c8d7'):
            raise ValueError('unverified scene or cutter for TCP marker exclusion')
        name = 'psm2_tool_tip_visual'
        sim, h = self.sim, self.by[name]
        if sim.getObjectType(h) != sim.object_shape_type:
            raise ValueError('expected the URDF TCP marker shape')
        before = sim.getObjectSpecialProperty(h)
        layer = sim.getObjectInt32Param(h, sim.objintparam_visibility_layer)
        sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
        sim.setObjectSpecialProperty(h, before & ~sim.objectspecialproperty_renderable)
        after = sim.getObjectSpecialProperty(h)
        layer_after = sim.getObjectInt32Param(h, sim.objintparam_visibility_layer)
        if after != before & ~sim.objectspecialproperty_renderable or layer_after != 0:
            raise RuntimeError('TCP marker optics readback mismatch')
        return dict(alias=name, layer_before=layer, layer_after=layer_after, properties_before=before,
                    properties_after=after, render_bit=sim.objectspecialproperty_renderable)

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

    # -- 6-DOF tool-pose control (cut.align_mode == 'tool_pose') --------------------------------
    # TCP frame columns measured on the verified scene: column 2 is the jaw pointing axis
    # (closed blade direction), column 0 is the jaw opening/hinge axis — it equals the grip
    # joints' z (psm2_joint10/11) within 1e-3, and a 0.4 rad wrist-yaw command rotates the
    # other two columns around it. The opening-axis sign is free (mirrored blades).
    TOOL_AXIS_COL, TOOL_YAW_COL = 2, 0

    def _wrist_state(self, arm):
        ws = self.call('getWristState', self.index[arm])
        return ws if isinstance(ws, dict) and ws.get('available') else None

    def _tool_q(self, arm):
        """Current 6-DOF command vector [yaw_deg, pitch_deg, ins_m, roll, wpitch, wyaw]."""
        ws = self._wrist_state(arm)
        if ws is None:
            return None
        yaw, pitch, ins = self.target_of(arm)
        return np.array([yaw, pitch, ins, ws['roll'], ws['pitch'], ws['yaw']], dtype=float), ws

    def _tool_fk(self, arm):
        i = self.index[arm]

        def fk(q):
            r = self.call('toolPoseFK', i, *map(float, q))
            if not isinstance(r, dict) or r.get('ok') is not True or r.get('shaft_ok') is not True:
                reason = r.get('reason') if isinstance(r, dict) else None
                raise RuntimeError(f"toolPoseFK refused: {reason or 'shaft pose unavailable'}")
            m = np.asarray(r.get('tcp'), dtype=float)
            if m.size != 12 or not np.isfinite(m).all():
                raise RuntimeError('toolPoseFK returned an invalid TCP')
            return m.reshape(3, 4)

        return fk

    def _tool_limits(self, arm, ws):
        L = self.arm_cfg[arm]['limits']
        wl = ws['limits']
        lo = np.array([-L['yaw_deg'], -L['pitch_deg'], L['insertion_min'],
                       wl['roll'][0], wl['pitch'][0], wl['yaw'][0]])
        hi = np.array([L['yaw_deg'], L['pitch_deg'], L['insertion_max'],
                       wl['roll'][1], wl['pitch'][1], wl['yaw'][1]])
        return lo, hi

    def tool_command(self, arm, q, n=15):
        """Atomic shaft+wrist command; refused-candidate accounting identical to command()."""
        i = self.index[arm]
        before = self._solver_failures(arm)
        ok = bool(self.call('setToolPose', i, *map(float, q)))
        if ok:
            self.steps(n)
            return ok
        after = self._solver_failures(arm)
        if before is not None and after is not None:
            self.refused_failures[arm] = self.refused_failures.get(arm, 0) + (after - before)
        return ok

    def _tool_solve(self, arm, target_p, target_axis=None, target_yaw=None):
        """IK-solve one tool pose from the current q; returns the tool_pose.solve result."""
        state = self._tool_q(arm)
        if state is None:
            return None
        q0, ws = state
        lo, hi = self._tool_limits(arm, ws)
        try:
            res = tool_pose.solve(self._tool_fk(arm), q0, (lo, hi), np.asarray(target_p),
                                  target_axis=target_axis, target_yaw_axis=target_yaw,
                                  axis_col=self.TOOL_AXIS_COL, yaw_col=self.TOOL_YAW_COL)
        except RuntimeError:
            return None
        res['q0'] = q0
        return res

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

    # -- M4: shadow perception (logged next to truth, never used for control) ----------------------
    def _create_markers(self):
        """Two small colour-marker spheres on the cutter's yaw housing, staged copy only.

        Pure cyan/yellow are not in the scene palette; the spheres are static, non-respondable and
        sit proud of the housing where the endoscope can see them without lying between the camera
        and the cut front (the markers on/off trace check verifies the depth gates do not change).
        """
        yaw_h = self.by.get(f'{self.cutter}_dvrk_yaw_frame')
        if yaw_h is None:
            return
        yaw = matrix_to_np(self.sim.getObjectMatrix(yaw_h, -1))
        tcp_M = matrix_to_np(self.sim.getObjectMatrix(self.by[f'{self.cutter}_tcp'], -1))
        tcp = tcp_M[:, 3]
        # Instrument calibration on the rigid yaw->tcp chain (verified constant to <1e-15 m):
        # the TCP position and jaw pointing axis in yaw-frame coordinates. The third marker is
        # non-collinear so a Kabsch fit of the three marker offsets recovers the full yaw pose.
        self._tcp_local = yaw[:, :3].T @ (tcp - yaw[:, 3])
        self._jaw_axis_local = yaw[:, :3].T @ tcp_M[:, self.TOOL_AXIS_COL]
        self._opening_axis_local = yaw[:, :3].T @ tcp_M[:, self.TOOL_YAW_COL]
        # Yaw-frame y is the shaft axis (frame() maps col1 to 'distal'); the TCP distance
        # along it calibrates the tip offset.
        self._marker_tcp_len = float(self._tcp_local[1])
        third_off = tuple(self.perc_cfg.get('marker3_off', cut_perception.MARKER_THIRD_OFF))
        self._marker_offsets_used = [list(cut_perception.MARKER_DISTAL_OFF),
                                   list(cut_perception.MARKER_PROX_OFF), list(third_off)]
        for color, off in ((cut_perception.MARKER_DISTAL_RGB, cut_perception.MARKER_DISTAL_OFF),
                           (cut_perception.MARKER_PROX_RGB, cut_perception.MARKER_PROX_OFF),
                           (cut_perception.MARKER_THIRD_RGB, third_off)):
            h = self.sim.createPrimitiveShape(self.sim.primitiveshape_spheroid,
                                              [2 * cut_perception.MARKER_RADIUS] * 3, 0)
            self.sim.setShapeColor(h, None, self.sim.colorcomponent_ambient_diffuse, list(color))
            # Emissive so the mask colour is lighting-independent (pure cyan/yellow pixels).
            self.sim.setShapeColor(h, None, self.sim.colorcomponent_emission, list(color))
            self.sim.setObjectInt32Param(h, self.sim.shapeintparam_static, 1)
            self.sim.setObjectInt32Param(h, self.sim.shapeintparam_respondable, 0)
            self.sim.setObjectParent(h, yaw_h, True)
            self.sim.setObjectPosition(h, yaw_h, list(off))
            self.markers.append(h)

    def _estimate(self):
        """One estimator run on this tick's frame. No GroundTruth access: the camera
        pose is proprioception and the injected perturbations come from perc_rng."""
        if self._perc_estimator is None:
            # Live and offline are identical: the estimator is built from the same meta
            # dict the frame dump writes (CutPerception.from_meta).
            if self._perc_meta is None:
                self._perc_meta = self._dump_meta()
            self._perc_estimator = cut_perception.CutPerception.from_meta(self._perc_meta)
        rng = self.perc_rng
        R_true, t_true = self.camera()
        e = self.perc_cfg.get('camera_pose_error') or {}
        if e.get('mode', 'per_tick') == 'fixed':
            # A calibration bias: axis and direction drawn once per run (m5b_protocol.json).
            if self._pose_err_fixed is None:
                self._pose_err_fixed = (unit(rng.normal(size=3)), unit(rng.normal(size=3)))
            axis, tdir = self._pose_err_fixed
        else:
            axis, tdir = unit(rng.normal(size=3)), unit(rng.normal(size=3))
        R_est = cut_perception.rotvec(axis * math.radians(e.get('rot_deg', 0.0))) @ R_true
        t_est = t_true + tdir * e.get('trans_m', 0.0)
        self._view_frame = (R_est @ R_true.T, t_true, t_est)
        noise = self.perc_cfg.get('depth_noise_m', 0.0)
        depth = self.depth if noise <= 0 else self.depth + rng.normal(0.0, noise, self.depth.shape)
        return self._perc_estimator.estimate(np.asarray(self.image, dtype=float), depth,
                                             R_est, t_est, post_grasp=bool(self.grasped_points))

    def _perception_tick(self):
        self._score_shadow(self._estimate())

    def _score_shadow(self, est):
        """Truth scoring of an estimate for the shadow log/dump. Never control input."""
        est = est or {}
        # Scoring only (m5a_gates.json definitions): tip vs the tcp pose, jaw_axis vs the
        # tcp matrix' tool-axis column, front vs GroundTruth.cut_front, direction vs the
        # live centre line m[j0+3]-m[j0], j0 = min(round(cut_length/H), NV-4).
        truth, errors = None, {}
        tcp_M = matrix_to_np(self.sim.getObjectMatrix(self.by[f'{self.cutter}_tcp'], -1))
        tcp, jaw_true = tcp_M[:, 3], unit(tcp_M[:, self.TOOL_AXIS_COL])
        scis = est.get('scissors') or {}
        if scis.get('valid'):
            errors['tip_pos_err_m'] = float(np.linalg.norm(np.asarray(scis['tip']) - tcp))
            if scis.get('jaw_axis') is not None:
                errors['jaw_axis_err_rad'] = float(math.acos(float(np.clip(
                    unit(np.asarray(scis['jaw_axis'])) @ jaw_true, -1, 1))))
        if self.sheet_origin is not None:
            with self.truth.purpose('scoring'):
                n = self.truth.nodes()
                m = 0.5 * (n[:, NU // 2 - 1] + n[:, NU // 2])
                j0 = min(int(round(self.cut_length() / H)), NV - 4)
                truth = dict(cut_front=self.cut_front(), direction=unit(m[j0 + 3] - m[j0]).tolist(),
                             tcp=tcp.tolist(), jaw_axis=jaw_true.tolist(),
                             tension=self.cut_line_tension())
            cut = est.get('cut') or {}
            if cut.get('front_valid') and cut.get('front') is not None:
                errors['front_pos_err_m'] = float(np.linalg.norm(
                    np.asarray(cut['front']) - np.asarray(truth['cut_front'])))
            if cut.get('valid') and cut.get('direction') is not None:
                errors['direction_err_rad'] = float(math.acos(float(np.clip(
                    np.asarray(cut['direction']) @ np.asarray(truth['direction']), -1, 1))))
        self.perc_log.append(dict(tick=self.tick, estimate=est, truth=truth, errors=errors))
        if self._dump_dir is not None:
            with self.truth.purpose('scoring'):
                self._dump_frame()

    # -- M5b: the remembered front fix (vision mode) -------------------------------------------------------------
    def _update_fix(self, est):
        """Refresh the remembered front from a fresh estimate that sees a valid front."""
        cut = est.get('cut') or {}
        tissue = est.get('tissue') or {}
        if not (cut.get('front_valid') and cut.get('front') is not None
                and cut.get('direction') is not None and cut.get('start') is not None):
            return
        if self.awaiting_new_slit and cut.get('mode') not in ('slit', 'bank_slit'):
            return
        front = np.asarray(cut['front'], dtype=float)
        direction = np.asarray(cut['direction'], dtype=float)
        start = np.asarray(cut.get('material_start'), dtype=float)
        normal = np.asarray(tissue.get('normal'), dtype=float)
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (front, direction, start, normal)):
            return
        if not np.any(direction) or not np.any(normal):
            return
        direction = unit(direction)
        progress = cut.get('progress_m')
        scale = cut.get('material_scale')
        if progress is None or not math.isfinite(progress) or progress < 0:
            return
        chart = None
        if getattr(self, 'material_pattern', False):
            m = cut.get('material_chart') or {}
            if not m.get('valid') or cut.get('mode') not in ('pre_cut', 'slit', 'bank_slit'):
                return
            chart = tuple(np.asarray(m.get(k), dtype=float) for k in
                          ('material_knots_m', 'projected_knots_m', 'guard_knots_m'))
            knots, physical, guard = chart
            coords = cut_perception.material_coordinate
            expected = coords.KNOTS_M
            local = getattr(self, 'local_material_chart', False)
            if (knots.ndim != 1 or len(knots) < 2
                    or not np.isin(knots, expected).all()
                    or np.any(np.diff(knots) <= 0) or knots[0] != expected[0] or knots[-1] != expected[-1]
                    or (not local and len(knots) != len(expected))
                    or any(a.shape != knots.shape or not np.isfinite(a).all() or np.any(np.diff(a) <= 0)
                           for a in (physical, guard))
                    or physical[0] != 0 or np.any(guard > physical)
                    or progress > knots[-1]):
                return
            if local:
                supported = coords.locate(knots, physical, float((front - start) @ direction))
                if (supported is None or cut.get('progress_cell_m') != supported[1]
                        or not math.isclose(progress, supported[0], rel_tol=0, abs_tol=1e-10)):
                    return
            if (self.awaiting_new_slit
                    and progress < self._pre_closure_progress + cut_perception.GAP_BIN_M / 2):
                return
        elif scale is None or not math.isfinite(scale) or scale < 1:
            return
        self._fix = dict(front=front, start=start, direction=direction, material_scale=scale,
                         material_chart=chart, normal=normal, progress=progress,
                         tick=self.tick, epoch=self._fix_epoch)
        if self.awaiting_new_slit:
            self.event_log.append(dict(tick=self.tick, event='post_closure_reacquired', mode=cut['mode'],
                                       progress_m=progress, progress_cell_m=cut.get('progress_cell_m')))
            self.awaiting_new_slit = False

    def _fix_valid(self):
        age = 0 if getattr(self, 'material_pattern', False) else self._fix_max_age
        return self._fix is not None and 0 <= self.tick - self._fix['tick'] <= age

    def _invalidate_fix(self):
        """Commanded motion voids the remembered front; injected events must not."""
        if self.vision and not self._in_event:
            if self._fix is not None:
                self._fix_epoch += 1
            self._fix = None

    def _dump_meta(self):
        """Static run metadata for the frame dump; written once (first dumped tick)."""
        rcm = self.arm_cfg[self.cutter]['rcm_world']
        return dict(W=self.W, H=self.H, f=self.f, far=self.far,
                    camera=dict(self.endo['camera']),
                    cutter=self.cutter, rcm_world=list(rcm),
                    marker_tcp_len=self._marker_tcp_len,
                    tcp_local=getattr(self, '_tcp_local', None),
                    jaw_axis_local=getattr(self, '_jaw_axis_local', None),
                    marker_radius=cut_perception.MARKER_RADIUS,
                    marker_offsets=(getattr(self, '_marker_offsets_used', None) or
                                    [list(cut_perception.MARKER_DISTAL_OFF),
                                     list(cut_perception.MARKER_PROX_OFF),
                                     list(cut_perception.MARKER_THIRD_OFF)]),
                    marker_colours=[list(cut_perception.MARKER_DISTAL_RGB),
                                    list(cut_perception.MARKER_PROX_RGB),
                                    list(cut_perception.MARKER_THIRD_RGB)],
                    tissue_rgb=list(cut_perception.TISSUE_RGB),
                    tool_axis_col=self.TOOL_AXIS_COL, tool_yaw_col=self.TOOL_YAW_COL,
                    NU=NU, NV=NV, spacing_m=H, pretension=self.pretension,
                    scene=str(self.scene_file),
                    scene_sha256=file_sha256(self.scene_file) if self.scene_file.exists() else None,
                    scenario=getattr(self.args, 'scenario', None),
                    mode=self.mode, align_mode=self.align_mode,
                    control_source='vision' if self.vision else 'truth',
                    run_id=getattr(self.args, 'run_id', None),
                    run_name=getattr(self.args, 'report_name', None),
                    occlusion_probe=getattr(self.args, 'occlusion_probe', None),
                    initial_view=getattr(self.args, 'initial_view', None),
                    initial_view_insertion_m=getattr(self.args, 'initial_view_insertion_m', None),
                    estimator='cut_perception.CutPerception',
                    estimator_args=dict(W=self.W, H=self.H, f=self.f, far=self.far,
                                        rcm_world=list(rcm), marker_radius=cut_perception.MARKER_RADIUS,
                                        marker_offsets=(getattr(self, '_marker_offsets_used', None) or
                                                        [list(cut_perception.MARKER_DISTAL_OFF),
                                                         list(cut_perception.MARKER_PROX_OFF),
                                                         list(cut_perception.MARKER_THIRD_OFF)]),
                                        marker_colours=[list(cut_perception.MARKER_DISTAL_RGB),
                                                        list(cut_perception.MARKER_PROX_RGB),
                                                        list(cut_perception.MARKER_THIRD_RGB)],
                                        tcp_local=getattr(self, '_tcp_local', None),
                                        jaw_axis_local=getattr(self, '_jaw_axis_local', None),
                                        opening_axis_local=getattr(self, '_opening_axis_local', None),
                                        material_pattern=getattr(self, 'material_pattern', False),
                                        local_material_chart=getattr(self, 'local_material_chart', False),
                                        front_recovery=self.perc_cfg.get('front_recovery', False),
                                        ray_marker_fit=self.perc_cfg.get('ray_marker_fit', False),
                                        tcp_len=self._marker_tcp_len,
                                        sheet_width_m=(NU - 1) * H, sheet_length_m=(NV - 1) * H),
                    perception_cfg=dict(self.perc_cfg))

    def _dump_frame(self):
        """Per-tick npz of raw estimator inputs and full truth for offline estimator work.

        rgb/depth are exactly what the estimator could see (depth BEFORE any injected noise);
        every cutter/sheet value is a true sim quantity. None of it feeds control.
        """
        if not self._dump_meta_written:
            if self._perc_meta is None:
                self._perc_meta = self._dump_meta()
            write_json(self._dump_dir / 'meta.json', self._perc_meta)
            self._dump_meta_written = True
        cam_R, cam_t = self.camera()

        def mat4(h):
            M = np.eye(4)
            M[:3] = matrix_to_np(self.sim.getObjectMatrix(h, -1))
            return M

        out = dict(rgb=np.asarray(self.image),                          # uint8 (H,W,3)
                   depth=np.asarray(self.depth, dtype=np.float32),      # raw (H,W), pre-noise
                   cam_R=cam_R, cam_t=cam_t, tick=np.int64(self.tick),
                   tcp_M=mat4(self.by[f'{self.cutter}_tcp']),
                   yaw_M=(mat4(self.by[f'{self.cutter}_dvrk_yaw_frame'])
                          if f'{self.cutter}_dvrk_yaw_frame' in self.by else np.full((4, 4), np.nan)),
                   marker_pos=(np.array([self.sim.getObjectPosition(m, -1) for m in self.markers],
                                    dtype=float) if self.markers else np.zeros((0, 3))))
        probe = getattr(self.args, 'occlusion_probe', None)
        if probe and self.sheet_origin is not None and self.tick in probe['ticks']:
            out['probe_rgb'], out['probe_depth'] = self._capture_probe(
                cam_t, (self.cut_front(), out['tcp_M'][:3, 3]), probe)
        try:
            stats = self.call('getRCMStats')
            out['grip'] = np.float64(stats[self.index[self.cutter] - 1].get('grip', np.nan))
            out['camera_target'] = np.asarray(stats[self.index[self.view] - 1]['target'], dtype=float)
        except Exception:
            out.setdefault('grip', np.float64(np.nan))
        if self.sheet_origin is not None:
            arms = sorted(self.grasped_points)
            out.update(nodes=self.truth.nodes(),
                       cut_length=np.float64(self.cut_length()),
                       cut_front=np.asarray(self.cut_front()),
                       tension=np.float64(self.cut_line_tension()),
                       sheet_v=np.asarray(self.v), sheet_u=np.asarray(self.u),
                       sheet_origin=np.asarray(self.sheet_origin),
                       tissue_alive=np.array(self.call('tissueConstraints'),
                                             dtype=np.int64).reshape(-1, 3),
                       grasped_arms=np.asarray(arms),
                       grasped_pts=(np.array([self.grasped_points[a] for a in arms], dtype=float)
                                    if arms else np.zeros((0, 3))))
        np.savez_compressed(self._dump_dir / f'tick_{self.tick:03d}.npz', **out)

    def _capture_probe(self, camera, points, spec):
        """Render blockers without changing any control image or stepping dynamics."""
        image, depth = self.image, self.depth
        handles = []
        sim = self.sim
        try:
            for point in points:
                pos = camera + spec['distance_m'] * unit(np.asarray(point) - camera)
                h = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [2 * spec['radius_m']] * 3, 0)
                handles.append(h)
                sim.setObjectPosition(h, -1, pos.tolist())
                sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0.55, 0.55, 0.5])
                sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
                sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
            self.render()
            return np.asarray(self.image), np.asarray(self.depth, dtype=np.float32)
        finally:
            self.image, self.depth = image, depth
            if handles:
                sim.removeObjects(handles)

    def finish_capture(self):
        if self._dump_dir is not None:
            write_json(self._dump_dir / 'capture.json',
                       dict(run_id=self.args.run_id, frame_count=self.tick,
                            initial_view_readback=self.initial_view_readback))

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
        if self.roi_proxy is not None:
            # Synthetic proxy ROI (M3b): curved sheet on the +2 mm prostate offset surface.
            # Same grid and anchoring as the flat sheet; explicit nodes bypass workspace.layout.
            rp = self.roi_proxy
            nodes = np.asarray(rp['nodes_m'], float).reshape(NV, NU, 3)
            self.v, self.u = np.asarray(rp['v'], float), np.asarray(rp['u'], float)
            self.sheet_origin = nodes[0, (NU - 1) // 2]
            self.grasp_points = {k: np.asarray(g, float) for k, g in rp['grasp_m'].items()}
            self.call('tissueCreate', dict(nodes=nodes.reshape(-1, 3).tolist(), nu=NU, nv=NV,
                                           spacing=H, pretension=self.pretension,
                                           anchor_sides=self.mode == 'cut_only',
                                           anchor_from_row=3, grasp_radius=GRASP_RADIUS_M))
        else:
            f = self.frame(self.cutter)
            L = workspace.layout(f['pivot'], f['x'], f['axis'], f['tcp'],
                                 roll_deg=self.sheet_cfg.get('roll_deg', 0.0))
            self.v, self.u, self.sheet_origin, self.grasp_points = \
                L['v'], L['u'], L['origin'], L['grasp_points']
            seam = {}
            if self.canonical_seam:
                seam['seam_mode'] = 'surface'
                if 'guide_band_m' in self.sheet_cfg:
                    seam['guide_band_m'] = float(self.sheet_cfg['guide_band_m'])
                if self.sheet_cfg.get('ruler_print'):
                    seam['ruler_print'] = True
                if self.sheet_cfg.get('indexed_print'):
                    seam['indexed_print'] = True
            elif 'visual_gap_m' in self.sheet_cfg:
                seam['visual_gap_m'] = self.sheet_cfg['visual_gap_m']
                if 'visual_gap_style' in self.sheet_cfg:
                    seam['visual_gap_style'] = self.sheet_cfg['visual_gap_style']
                if 'visual_gap_start_row' in self.sheet_cfg:
                    seam['visual_gap_start_row'] = self.sheet_cfg['visual_gap_start_row']
            if self.sheet_cfg.get('bridge_print'):
                seam['bridge_print'] = True
                if self.sheet_cfg.get('bridge_fracture_faces'):
                    seam['bridge_fracture_faces'] = True
            self.call('tissueCreate', dict(origin=self.sheet_origin.tolist(), u=self.u.tolist(), v=self.v.tolist(),
                                           nu=NU, nv=NV, spacing=H, pretension=self.pretension,
                                           anchor_sides=self.mode == 'cut_only',
                                           anchor_from_row=3, grasp_radius=GRASP_RADIUS_M,
                                           **seam,
                                           **({'material_bands': cut_perception.material_coordinate.bands()}
                                              if getattr(self, 'material_pattern', False) else {})))
        self.steps(30)
        # Actual measured centre transverse rests for every new sheet (the canonical tension
        # basis regardless of the nominal grid). The frozen legacy controller has no
        # readback; locked collectors keep their explicitly uniform fixture basis.
        if self.tissue_source == LEGACY_TISSUE:
            self.cutline_rest_m = None
            return
        rest = np.asarray(self.call('tissueTransverseRestLengths') or [], dtype=float)
        if rest.shape != (NV,) or not np.isfinite(rest).all() or np.any(rest <= 0):
            raise RuntimeError('tissueTransverseRestLengths readback invalid or missing')
        self.cutline_rest_m = rest
        self.cutline_rest_m.setflags(write=False)

    # Protected-structure monitor (M3b): KD-trees of the rigid anatomy meshes in world
    # coordinates, built once. Per tick, the minimum distance from every instrument TCP and
    # sampled wrist->TCP distal segment to each structure. MODEL DISTANCES ONLY -- these are
    # mesh-vertex distances in the synthetic scene, not clinical safety distances. The values
    # are logged for the report and never influence control.
    PROTECTED = ('Patient_Bladder', 'Patient_Rectum', 'Patient_Prostate',
                 'Patient_LeftUreter', 'Patient_RightUreter',
                 'Patient_LeftHipBone', 'Patient_RightHipBone', 'Patient_Sacrum')

    def _protected_trees(self):
        if getattr(self, '_ptrees', None) is not None:
            return self._ptrees
        from scipy.spatial import cKDTree
        self._ptrees = {}
        for name in self.PROTECTED:
            if name not in self.by:
                continue
            V = np.asarray(self.sim.getShapeMesh(self.by[name])[0], float).reshape(-1, 3)
            M = matrix_to_np(self.sim.getObjectMatrix(self.by[name], -1))
            self._ptrees[name] = cKDTree(V @ M[:, :3].T + M[:, 3])
        return self._ptrees

    def _protected_tick(self):
        try:
            trees = self._protected_trees()
            if not trees:
                return
            pts = []
            for a in self.rcm:
                arm = a['arm']
                if f'{arm}_joint9' in self.by and f'{arm}_tcp' in self.by:
                    wrist = np.asarray(self.sim.getObjectPosition(self.by[arm + '_joint9'], -1))
                    tcp = np.asarray(self.sim.getObjectPosition(self.by[arm + '_tcp'], -1))
                    pts += [wrist + t * (tcp - wrist) for t in np.linspace(0, 1, 6)]
                elif a['shaft_frame'] in self.by:
                    pts.append(np.asarray(self.sim.getObjectPosition(self.by[a['shaft_frame']], -1)))
            P = np.asarray(pts)
            self.last_protected = {n.split('_', 1)[1].lower(): float(t.query(P)[0].min())
                                   for n, t in trees.items()}
            self.protected_log.append(dict(tick=self.tick, **self.last_protected))
        except Exception as exc:
            self.event_log.append(dict(tick=self.tick, event='protected_monitor_error',
                                       error=f'{type(exc).__name__}: {exc}'))
            self._ptrees = {}

    def protected_min(self):
        """Per-structure minimum over the run (model distances, not clinical safety)."""
        out = {}
        for row in self.protected_log:
            for k, v in row.items():
                if k != 'tick':
                    out[k] = min(v, out.get(k, v))
        return out

    # -- World interface --------------------------------------------------------------------------------------
    def observe(self, *, inject_events=True):
        self.tick += 1
        if inject_events:
            self._events()
        self.render()
        if self.vision:
            try:
                self._last_est = self._estimate()
                if inject_events and self._last_est is not None:
                    self._update_fix(self._last_est)
            except Exception as exc:
                import traceback
                self.event_log.append(dict(tick=self.tick, event='perception_error',
                                           error=f'{type(exc).__name__}: {exc}',
                                           trace=traceback.format_exc(limit=6)))
                self._last_est = None           # fail closed: every vision field invalid
            if self.perc_mode == 'shadow':
                try:
                    self._score_shadow(self._last_est)
                except Exception as exc:
                    import traceback
                    self.event_log.append(dict(tick=self.tick, event='perception_error',
                                               error=f'{type(exc).__name__}: {exc}',
                                               trace=traceback.format_exc(limit=6)))
        elif self.perc_mode == 'shadow':
            try:
                self._perception_tick()
            except Exception as exc:
                import traceback
                self.event_log.append(dict(tick=self.tick, event='perception_error',
                                           error=f'{type(exc).__name__}: {exc}',
                                           trace=traceback.format_exc(limit=6)))
        progress = tissue_vis = None
        if self.vision:
            est = self._last_est or {}
            tip_vis = bool((est.get('scissors') or {}).get('valid'))
            tissue_vis = bool((est.get('tissue') or {}).get('valid'))
            cut_vis = self._fix_valid()
            if cut_vis:
                progress = float(self._fix['progress'])
                uv, pc = self.project(np.asarray(self._fix['front'], dtype=float))
                if uv is not None and np.isfinite(pc).all() and 0 < pc[2] < self.far:
                    offset = float(max(abs(uv[0] - self.W / 2) / (self.W / 2),
                                       abs(uv[1] - self.H / 2) / (self.H / 2)))
                else:
                    offset = None
            else:
                offset = None
        else:
            if self.sheet_origin is None:
                cut_vis, offset = False, None
            else:
                cut_vis, offset = self.point_visible(self.cut_front())
            tip_vis, _ = self.point_visible(self.truth.tcp(self.cutter))
        self._protected_tick()
        with self.truth.purpose('grasp_signal_and_safety_monitor'):
            st = self.truth.tissue_stats()
        rs = self.call('getRCMStats')
        # A refused candidate increments the controller failure counter even though it rolled back; only
        # failures beyond those recorded at refusal are arm faults. Drift faults are unchanged.
        faults = [f"{r['arm']}: {r['failures'] - self.refused_failures.get(r['arm'], 0)} RCM solver failures"
                  for r in rs if r['failures'] - self.refused_failures.get(r['arm'], 0) > 0]
        faults += [f"{r['arm']}: RCM drift {r['max_drift_m'] * 1000:.2f} mm" for r in rs if r['max_drift_m'] > 5e-4]
        if self.canonical_seam:
            # SurfaceSeamWorld's farthest-row-end monitor, preserved on the canonical coupon.
            self.seam_releases = int(st.get('seam_releases') or 0)
            if st.get('exists') and self.sheet_origin is not None:
                with self.truth.purpose('safety_monitor'):
                    end = material_state(self.call('tissueConstraints'))['farthest_row_end'] * H
                limit = self.cfg['cut']['length_m'] + self.cfg['cut']['max_overcut_m']
                if end > limit + 1e-12:
                    faults.append(f'seam conservative row end {end * 1000:.2f} mm exceeds '
                                  f'{limit * 1000:.2f} mm')
        with self.truth.purpose('tension_until_m6'):
            tension = self.cut_line_tension()
        with self.truth.purpose('safety_monitor'):
            cut_length = self.cut_length()
        return Observation(cut_site_visible=cut_vis, scissors_visible=tip_vis, cut_site_offset=offset if cut_vis else None,
                           grasped=dict(st.get('grasped') or {}), tension=tension, max_strain=float(st.get('max_strain', 0.0)),
                           torn=int(st.get('torn', 0)), cut_length_m=cut_length, faults=faults,
                           cut_progress_m=progress if self.vision else cut_length, tissue_visible=tissue_vis)

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
        iv = getattr(getattr(self, 'args', None), 'initial_view', None)
        if iv is not None:
            home_R, home_t = self.camera()
            ins = getattr(self.args, 'initial_view_insertion_m', None)
            requested = [*iv, self.arm_cfg[self.view]['insertion_home'] if ins is None else ins]
            if not self.command(self.view, *requested, 25):
                raise RuntimeError(f'initial view {requested} unreachable')
            target = list(self.target_of(self.view))
            if not np.allclose(target, requested, rtol=0, atol=1e-6):
                raise RuntimeError(f'initial view {requested} disagrees with ECM readback {target}')
            R, t = self.camera()
            self.initial_view_readback = dict(home_R=home_R, home_t=home_t, applied_R=R, applied_t=t,
                                              target=target)
        self.place_sheet()
        # Immutable homed anchors for the withdrawal/verification contract, recorded once.
        # A failed, missing or non-finite capture means the ready pose is unverifiable:
        # home itself refuses rather than cutting with no anchor.
        try:
            tcp = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
        except Exception:
            tcp = None
        if tcp is None or tcp.shape != (3, 4) or not np.isfinite(tcp).all():
            return False
        self._ready_tcp = tcp.copy()
        self._ready_tcp.setflags(write=False)
        self._ready_q = None
        if self.align_mode == 'tool_pose':
            state = self._tool_q(self.cutter)
            if state is None or np.shape(state[0]) != (6,) or not np.isfinite(state[0]).all():
                return False
            self._ready_q = np.array(state[0], dtype=float, copy=True)
            self._ready_q.setflags(write=False)
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
        with self.truth.purpose('retractor_planning'):      # retractor planning stays truth until M6
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
        self._invalidate_fix()
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
        # The commands may all have converged yet left the tip off the node (controller tolerance):
        # closing here would pin a neighbouring patch and report it as this point.
        tip = np.array(self.sim.getObjectPosition(self.by[arm + '_tcp'], -1))
        err = np.linalg.norm(tip - point)
        if err > 1e-3:
            return False, f'approach ended {err * 1000:.1f} mm from the grasp point'
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
        self._invalidate_fix()
        self.call('tissueRelease', self.index[arm])
        self.grasped_points.pop(arm, None)

    def takeover_point(self, arm, lost_point):
        """Nearest live node to `lost_point` the arm can grasp on the same side of the cut line, or None.

        A handover keeps the side the lost arm held: candidates sit at least three columns in from
        the cut line, off the edge column and row 0, and inside the arm limits with a small margin
        (a point at the boundary would likely be refused by the guarded controller anyway)."""
        self._invalidate_fix()
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
        with self.truth.purpose('retractor_planning'):
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
        self._invalidate_fix()
        point = self.grasped_points.get(arm)
        if self.sheet_origin is None or point is None:
            return False        # no verified grasp side: refuse rather than pull in a guessed direction
        # The record can be stale after a physical slip: also require a live hold in the tissue model.
        with self.truth.purpose('retractor_planning'):
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
        if getattr(self, 'material_pattern', False) and self.vision:
            if not self._fix_valid() or self.awaiting_new_slit:
                return False
            self._pre_closure_progress = self._fix['progress']
        evidence = self._closure_event() if self.vision else None
        ok = bool(self.call('setGrip', self.index[self.cutter], 0.0))
        if ok:
            if self.vision:
                self.event_log.append(evidence)
                self._invalidate_fix()
                self.awaiting_new_slit = True
            self.steps(8)
        return ok

    def _closure_event(self):
        """Sample B6/B7 evidence before the Lua jaw command changes the tissue."""
        est = self._last_est or {}
        scis = est.get('scissors') or {}
        ev = dict(tick=self.tick, event='closure',
                  fix_age=None if self._fix is None else int(self.tick - self._fix['tick']),
                  scissors_valid=bool(scis.get('valid')), lateral_truth_m=None)
        if getattr(self, 'local_material_chart', False):
            chart = (est.get('cut') or {}).get('material_chart') or {}
            ev.update(verified_rows_m=chart.get('material_knots_m'),
                      progress_cell_m=(est.get('cut') or {}).get('progress_cell_m'))
        try:
            tcp_M = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
            jaw = unit(tcp_M[:, self.TOOL_AXIS_COL])
            with self.truth.purpose('scoring'):
                cf = np.asarray(self.cut_front(), dtype=float)
            lateral = float(np.linalg.norm(np.cross(cf - tcp_M[:, 3], jaw)))
            if math.isfinite(lateral) and np.any(jaw):
                ev['lateral_truth_m'] = lateral
        except Exception as exc:
            ev['scoring_error'] = f'{type(exc).__name__}: {exc}'
        return ev

    def jaws_closed(self):
        """Physical grip joints, not the commanded grip: both blade joints within 1e-6 rad of closed."""
        joints = (getattr(self, 'arm_cfg', {}).get(self.cutter) or {}).get('grip_joints', ())
        if len(joints) != 2 or any(j not in getattr(self, 'by', {}) for j in joints):
            return False
        try:
            for name in joints:
                p = self.sim.getJointPosition(self.by[name])
                if not math.isfinite(p) or abs(p) > 1e-6:
                    return False
        except Exception:
            return False
        return True

    def at_ready_pose(self):
        """Cutter TCP at the measured homed pose: 0.5 mm translation, 1 deg rotation.

        A missing anchor is never reported as ready; the readback itself must be finite."""
        ready = self._ready_tcp
        if ready is None:
            return False
        try:
            tcp = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
        except Exception:
            return False
        if tcp.shape != (3, 4) or not np.isfinite(tcp).all():
            return False
        if np.linalg.norm(tcp[:, 3] - ready[:, 3]) > .0005:
            return False
        c = np.clip((np.trace(tcp[:, :3].T @ ready[:, :3]) - 1) / 2, -1.0, 1.0)
        return math.degrees(math.acos(float(c))) <= 1.0

    def clear_view(self, d, prefer_park=False):
        """Back off with unchanged jaws; a park pose must not undo the back-off."""
        if prefer_park and self.align_mode == 'tool_pose' and self._ready_tcp is not None:
            # Park only when the actual withdrawal target is behind the current pose:
            # retract_scissors lands at _ready_tcp, so guard on exactly that anchor.
            try:
                current = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
            except Exception:
                current = None
            if (current is not None and current.shape == (3, 4) and np.isfinite(current).all()
                    and (self._ready_tcp[:, 3] - current[:, 3]) @ current[:, self.TOOL_AXIS_COL] < -1e-6):
                return self.retract_scissors()
        if prefer_park and self.align_mode == 'shaft' and self.advanced > 0:
            return self.retract_scissors()
        return self._move_scissors(-d, 15, record_park=False)

    def align_scissors(self, cut_start, progress):
        """Lateral distance of the cut front from the blade line, or None if approach was refused."""
        if self.vision:
            return self._align_scissors_vision(progress)
        if self.align_mode == 'tool_pose':
            return self._align_scissors_tool(cut_start, progress)
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

    def _align_scissors_tool(self, cut_start, progress):
        """tool_pose alignment: tcp 1 mm short of the front, jaw axis along v, opening axis along u.

        The blade sweep (tissueCut) rotates the pivot->tip line about the grip-joint z; with the
        pointing axis on v and the opening axis on u, the sweep fans lie in the v x sheet-normal
        plane and pinch springs on the cut line."""
        p = self.cut_front()
        target = p - 0.001 * self.v
        state = self._tool_q(self.cutter)
        if state is None:
            return None
        if self._pre_cut_q is None:
            self._pre_cut_q = state[0].copy()      # pre-cut pose: wrist restore target for withdrawal
        tcp0 = np.array(self.sim.getObjectPosition(self.by[self.cutter + '_tcp'], -1))
        res = self._tool_solve(self.cutter, target, self.v, self.u)
        if res is None or not res['converged']:
            self.event_log.append(dict(tick=self.tick, event='tool_align', refused=True,
                                       detail=None if res is None else dict(
                                           pos_err_m=res['pos_err_m'], axis_err_rad=res['axis_err_rad'],
                                           yaw_axis_err_rad=res['yaw_axis_err_rad'], at_limit=res['at_limit'])))
            return None
        if not self.tool_command(self.cutter, res['q'], 10):
            return None
        tcp1 = np.array(self.sim.getObjectPosition(self.by[self.cutter + '_tcp'], -1))
        # Withdrawal bookkeeping counts forward progress along the cut line, not insertion,
        # because the jaw axis (not the shaft) is the advance direction in tool_pose mode.
        self.advanced += float(np.dot(tcp1 - tcp0, self.v))
        self.event_log.append(dict(tick=self.tick, event='tool_align', refused=False,
                                   pos_err_m=res['pos_err_m'], axis_err_rad=res['axis_err_rad'],
                                   yaw_axis_err_rad=res['yaw_axis_err_rad'], iterations=res['iterations'],
                                   at_limit=res['at_limit']))
        m = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
        d = m[:3, self.TOOL_AXIS_COL]
        return float(np.linalg.norm(np.cross(self.cut_front() - m[:3, 3], d)))

    def _align_scissors_vision(self, progress):
        """tool_pose alignment on the remembered front fix and this tick's tip estimate.

        The move is RELATIVE (tcp_fk + (fix.front - 1 mm*dir) - tip_est) so a kinematic
        offset cancels; the pre-move combined error is returned so the next ALIGN tick
        verifies against a fresh estimate. None on missing evidence or refusal."""
        self.view_blocked = False
        geometry = self._vision_alignment_geometry()
        if geometry is None:
            return None
        front, tip, direction, lat, err_lat, axes_ok = geometry
        err_along = abs(float((front - tip) @ direction) - 0.001)
        e = max(err_lat, err_along)
        fix_age = int(self.tick - self._fix['tick'])
        if e <= self.cfg['cut']['align_tolerance_m'] and axes_ok:
            return e                                          # verified; no motion
        moved = False
        if self.align_mode == 'tool_pose':
            tcp_M = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
            tcp0 = tcp_M[:, 3]                                # kinematic tcp: the offset cancels
            target = tcp0 + (front - 0.001 * direction) - tip
            if self._predictive_view():
                res = self._visible_tool_candidate(
                    'align', tcp_M, [(d, target - (d - 0.001) * direction) for d in (0.001, 0.0015, 0.002)],
                    direction, lat)
                if res is None:
                    return None
            else:
                res = self._tool_solve(self.cutter, target, direction, lat)
            if not self._vision_tool_motion(res, 10):
                self.event_log.append(dict(tick=self.tick, event='vision_align', refused=True,
                                           err_lat_m=err_lat, err_along_m=err_along,
                                           moved=False, fix_age=fix_age))
                return None
            tcp1 = np.array(self.sim.getObjectPosition(self.by[self.cutter + '_tcp'], -1))
            self.advanced += float(np.dot(tcp1 - tcp0, self.v))
            moved = True
        else:
            # shaft: only the along correction is commandable; fix/tip_est supply the gap.
            gap = float((front - tip) @ direction)
            if abs(gap - 0.001) > self.cfg['cut']['align_tolerance_m']:
                tcp_M = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
                axis, _ = self._visual_advance_axis(self._last_est['scissors'], tcp_M)
                if axis is None or not float(axis @ direction) > 0:
                    return None
                yaw, pitch, ins = self.target_of(self.cutter)
                correction = (gap - 0.001) / float(axis @ direction)
                if not self.command(self.cutter, yaw, pitch, ins + correction, 10):
                    return None
                self.advanced += correction
                moved = True
        self.event_log.append(dict(tick=self.tick, event='vision_align', refused=False,
                                   err_lat_m=err_lat, err_along_m=err_along, axes_ok=axes_ok,
                                   moved=moved, fix_age=fix_age))
        return e if axes_ok else None                         # verify the corrected axes next tick

    def _vision_alignment_geometry(self):
        est = self._last_est or {}
        scis = est.get('scissors') or {}
        if (not self._fix_valid() or not scis.get('valid')
                or scis.get('tip') is None or scis.get('jaw_axis') is None
                or self._fix['direction'] is None or self._fix['normal'] is None):
            return None
        direction = np.asarray(self._fix['direction'], dtype=float)
        if float(direction @ self.v) <= 0:
            return None                       # fix direction disagrees with the sheet frame
        front = np.asarray(self._fix['front'], dtype=float)
        tip = np.asarray(scis['tip'], dtype=float)
        jaw = np.asarray(scis['jaw_axis'], dtype=float)
        # Opening axis of the cut, signed to agree with the placement frame u.
        lat = np.cross(np.asarray(self._fix['normal'], dtype=float), direction)
        if not np.isfinite([front, tip, jaw, direction, lat]).all() or not np.any(jaw) or not np.any(lat):
            return None
        jaw, lat = unit(jaw), unit(lat)
        if float(lat @ self.u) < 0:
            lat = -lat
        err_lat = float(np.linalg.norm(np.cross(front - tip, jaw)))
        cos_tol = math.cos(math.radians(self.cfg['cut']['vision_axis_tolerance_deg']))
        axes_ok = bool(jaw @ direction >= cos_tol)
        if self.align_mode == 'tool_pose':
            opening = np.asarray(scis.get('opening_axis'), dtype=float)
            # Pointing is directed, but either hinge-axis sign is valid for symmetric blades.
            axes_ok = (axes_ok and opening.shape == (3,) and np.isfinite(opening).all()
                       and np.any(opening) and abs(unit(opening) @ lat) >= cos_tol)
        return front, tip, direction, lat, err_lat, axes_ok

    def closure_aligned(self):
        """Recheck cross-track and axes from the post-advance observation, not the 1 mm entry gap."""
        geometry = self._vision_alignment_geometry()
        return (geometry is not None and geometry[-1]
                and geometry[-2] <= self.cfg['cut']['align_tolerance_m'])

    def _visual_advance_axis(self, scissors, tcp_M):
        """Express the commanded motion (jaw or shaft) in the observed tool frame."""
        jaw = np.asarray(scissors.get('jaw_axis'), dtype=float)
        hinge = np.asarray(scissors.get('opening_axis'), dtype=float)
        if any(a.shape != (3,) or not np.isfinite(a).all() or not np.any(a) for a in (jaw, hinge)):
            return None, None
        jaw = unit(jaw)
        hinge = hinge - (hinge @ jaw) * jaw
        if np.linalg.norm(hinge) < 1e-8:
            return None, None
        hinge = unit(hinge)
        observed = np.column_stack((hinge, np.cross(jaw, hinge), jaw))
        rotation = observed @ tcp_M[:, :3].T
        if self.align_mode == 'tool_pose':
            return jaw, rotation
        shaft = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_joint8'], -1))
        return rotation @ shaft[:, 2], rotation

    def endpoint_travel(self, remaining):
        if not self._fix_valid() or not math.isfinite(remaining) or remaining <= 0:
            return None
        scis = (self._last_est or {}).get('scissors') or {}
        tip = np.asarray(scis.get('tip'), dtype=float)
        if not scis.get('valid') or tip.shape != (3,) or not np.isfinite(tip).all():
            return None
        tcp_M = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
        axis, _ = self._visual_advance_axis(scis, tcp_M)
        direction = self._fix['direction']
        projection = None if axis is None else float(axis @ direction)
        if projection is None or not math.isfinite(projection) or projection <= 0:
            return None
        gap = float((self._fix['front'] - tip) @ direction)
        chart = self._fix.get('material_chart')
        if chart is not None:
            knots, physical, _ = chart
            target = self.cfg['cut']['length_m']
            target_s = cut_perception.material_coordinate.project(knots, physical, target, stop_at_supported_row=True)
            if target_s is None:
                return None
            tip_s = float((tip - self._fix['start']) @ direction)
            return (target_s - tip_s) / projection
        return (gap + remaining / self._fix['material_scale']) / projection

    def endpoint_advance_budget(self):
        """Remaining available motion before the modeled blade sweep reaches the limit."""
        if not self._fix_valid():
            return None
        scale = self._fix.get('material_scale')
        chart = self._fix.get('material_chart')
        scissors = (self._last_est or {}).get('scissors') or {}
        if (not scissors.get('valid') or
                (chart is None and (scale is None or not math.isfinite(scale) or scale < 1))):
            return None
        tip = np.asarray(scissors.get('tip'), dtype=float)
        if tip.shape != (3,) or not np.isfinite(tip).all():
            return None
        start, direction = self._fix['start'], self._fix['direction']
        tcp_M = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
        axis, rotation = self._visual_advance_axis(scissors, tcp_M)
        if axis is None:
            return None
        projection = float(axis @ direction)
        if not math.isfinite(projection) or projection <= 0:
            return None
        joints = self.arm_cfg[self.cutter].get('grip_joints', ())
        if len(joints) != 2 or not all(j in self.by for j in joints):
            return None
        bounds = []
        for joint in joints:
            m = matrix_to_np(self.sim.getObjectMatrix(self.by[joint], -1))
            if not np.isfinite(m).all() or not np.any(m[:, 2]):
                return None
            pivot = rotation @ (m[:, 3] - tcp_M[:, 3]) + tip - start
            bounds.append(sweep_projection(pivot, unit(rotation @ m[:, 2]),
                                           tip - start, direction))
        bound = max(bounds) + CUT_RADIUS_M + cut_perception.GAP_BIN_M / 2
        limit = self.cfg['cut']['length_m'] + self.cfg['cut']['max_overcut_m']
        if not math.isfinite(bound):
            return None
        if chart is not None:
            knots, _, guard = chart
            limit_s = cut_perception.material_coordinate.project(knots, guard, limit, stop_at_supported_row=True)
            return None if limit_s is None else limit_s - bound
        # A subunit projection does not grant additional travel beyond the existing sweep reserve.
        return limit / scale - bound

    def _vision_tool_motion(self, res, n):
        if res is None or not res['converged']:
            return False
        q0 = res['q0']
        if self._pre_cut_q is None and not np.isfinite(q0).all():
            return False
        if not self.tool_command(self.cutter, res['q'], n):
            return False
        if self._pre_cut_q is None:
            self._pre_cut_q = q0.copy()
        return True

    def _predictive_view(self):
        return self.vision and self.align_mode == 'tool_pose' and self.cfg.get('perception', {}).get('predictive_view', False)

    def _candidate_view_counts(self, current, candidate):
        view = getattr(self, '_view_frame', None)
        if not self._fix_valid() or view is None:
            return 0, 0
        optical_R, optical_origin, estimate_origin = view
        front, direction, normal = (self._fix[k] for k in ('front', 'direction', 'normal'))
        lateral = np.cross(normal, direction)
        if not np.isfinite(lateral).all() or not np.any(lateral):
            return 0, 0
        lateral = unit(lateral)
        points = np.array([front + a * direction + b * lateral
                           for a in np.linspace(0, cut_perception.AHEAD_WINDOW_M + .0005, 5)
                           for b in np.linspace(-cut_perception.GAP_BAND_M, cut_perception.GAP_BAND_M, 5)])
        R = candidate[:, :3] @ current[:, :3].T
        joints = self.arm_cfg[self.cutter].get('grip_joints', ())
        if len(joints) != 2 or any(j not in self.by for j in joints):
            return 0, 0
        segments = []
        for j in joints:
            m = matrix_to_np(self.sim.getObjectMatrix(self.by[j], -1))
            pivot = candidate[:, 3] + R @ (m[:, 3] - current[:, 3])
            blade = candidate[:, 3] - pivot
            tip = pivot + cut_perception.rotvec(0.5 * (R @ m[:, 2])) @ blade
            segments.append((pivot, tip, .0006))
        pivot = np.mean([s[0] for s in segments], axis=0)
        segments.append((pivot - .004 * candidate[:, self.TOOL_AXIS_COL], pivot, .0015))
        # The cut plane is already in estimate coordinates; transform the FK proxy once.
        segments = [(estimate_origin + optical_R @ (a - optical_origin),
                     estimate_origin + optical_R @ (b - optical_origin), radius) for a, b, radius in segments]
        return cutter_view.stencil_counts(estimate_origin, points, segments)

    def _visible_tool_candidate(self, phase, current, targets, axis, hinge):
        self.view_blocked = False
        attempts = []
        for distance, target in targets:
            res = self._tool_solve(self.cutter, target, axis, hinge)
            trial = dict(distance_m=distance, clear=None)
            attempts.append(trial)
            if res is None or not res['converged']:
                continue
            try:
                pose = self._tool_fk(self.cutter)(res['q'])
            except RuntimeError:
                continue
            clear, at_front = self._candidate_view_counts(current, pose)
            trial.update(clear=clear, at_front=at_front)
            if clear >= 13 and at_front >= 2:
                self.view_blocked = False
                self.event_log.append(dict(tick=self.tick, event='predicted_view', phase=phase,
                                           selected_m=distance, candidates=attempts))
                return res
            self.view_blocked = True
        self.event_log.append(dict(tick=self.tick, event='predicted_view', phase=phase,
                                   selected_m=None, candidates=attempts))
        return None

    def advance_scissors(self, d):
        self.view_blocked = False
        return self._move_scissors(d, 15)

    def _move_scissors(self, d, n, record_park=True):
        if self.align_mode == 'tool_pose':
            # Advance along the jaw (pointing) axis, holding the current orientation —
            # not along the shaft, which no longer coincides with the cut direction.
            m = matrix_to_np(self.sim.getObjectMatrix(self.by[self.cutter + '_tcp'], -1))
            target = m[:3, 3] + d * m[:3, self.TOOL_AXIS_COL]
            axis, hinge = m[:3, self.TOOL_AXIS_COL], m[:3, self.TOOL_YAW_COL]
            if d > 0 and self._predictive_view():
                targets = [(step, m[:3, 3] + step * axis) for ratio in (1., .75, .5, .25)
                           if (step := d * ratio) >= .0002]
                res = self._visible_tool_candidate('advance', m, targets, axis, hinge)
            else:
                res = self._tool_solve(self.cutter, target, axis, hinge)
            if res is None or not res['converged']:
                return False
            tcp0 = m[:3, 3]
            ok = (self._vision_tool_motion(res, n) if self.vision and record_park
                  else self.tool_command(self.cutter, res['q'], n))
            if ok and (not self.vision or self._pre_cut_q is not None):
                tcp1 = np.array(self.sim.getObjectPosition(self.by[self.cutter + '_tcp'], -1))
                self.advanced += float(np.dot(tcp1 - tcp0, self.v))
            return ok
        yaw, pitch, ins = self.target_of(self.cutter)
        ok = self.command(self.cutter, yaw, pitch, ins + d, n)
        if ok:
            self.advanced += d
        return ok

    def retract_scissors(self):
        """Undo accepted cutting advances, without an extra bite merely to close for retreat.

        Vision restores the recorded pose with jaws unchanged, under the motion guard.
        This model has no tissue contact during motion; closing just to park could cut again.
        """
        if self.align_mode == 'tool_pose':
            # Restore the immutable homed pose; a missing anchor is a refusal, never a
            # pretended park. Jaw-preserving: no setGrip, no tissue-based veto.
            if self._ready_q is None or self._ready_tcp is None:
                return False
            if not math.isfinite(self.advanced):
                return False
            if self.at_ready_pose():
                # Already parked: no extra motion, but the withdrawal bookkeeping and
                # fix invalidation are identical to a landed recovery motion.
                self.advanced = 0.0
                self._pre_cut_q = None
                self._invalidate_fix()
                return True
            # Atomic: the withdrawal also restores the wrist to its homed angles.
            if not self.tool_command(self.cutter, self._ready_q, 20):
                return False
            if not self.at_ready_pose():
                return False                   # accepted motion did not land at the homed pose
            self.advanced = 0.0
            self._pre_cut_q = None
            self._invalidate_fix()
            return True
        if not math.isfinite(self.advanced) or (self.advanced < 0 and not self.vision):
            return False
        if self.advanced <= 0:
            # A vision back-off can already be behind the insertion at the ready pose.
            return True
        state = self.call('getRCMStats')[self.index[self.cutter] - 1]
        grip = state.get('grip')
        if grip is None or not math.isfinite(grip):
            return False
        if grip != 0.0 and not self.vision:
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
        self._invalidate_fix()
        return True

    CAMERA_POSES = [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0), (4.5, 3.0), (-4.5, -3.0)]

    def adjust_camera(self):
        cursor = getattr(self, '_camera_candidate', self.camera_moves)
        if cursor >= len(self.CAMERA_POSES):
            return False
        current = self.target_of(self.view)
        ins = current[2]
        while cursor < len(self.CAMERA_POSES):
            yaw, pitch = self.CAMERA_POSES[cursor]
            cursor += 1
            self._camera_candidate = cursor
            if np.allclose([yaw, pitch], current[:2], atol=1e-6, rtol=0):
                continue
            if not self.command(self.view, yaw, pitch, ins, 25):
                return False
            self._invalidate_fix()
            self.camera_moves += 1
            return True
        return False

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
        self._in_event = True
        try:
            self._inject_events(e)
        finally:
            self._in_event = False

    def _inject_events(self, e):
        occlude, until = e.get('occlude'), e.get('occlude_until_camera')
        if self.occluder is None and (until == self.tick or (occlude is not None and self.tick == occlude[0])):
            R, t = self.camera()
            with self.truth.purpose('event_injection'):       # harness read, not a control input
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
