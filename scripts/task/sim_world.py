"""CoppeliaSim world for scripts/task/cut_fsm.py, on scenes/davinci_si_instruments.ttt.

Sensing
  - endoscope RGB-D (EndoscopeCamera): the cut front and the scissors tip are visible when the rendered depth at
    their projected pixels is not more than 3 mm in front of them (anything nearer is an occluder);
    the cut-site offset is its pixel distance from the image centre (1 = image edge);
  - tissue model (src/tissue/tissue.lua): grasped nodes per arm, maximum strain, torn constraints;
  - cut length: rows of the sheet separated along the cut line (node geometry, not vision);
  - faults: RCM solver failures or RCM drift above 0.5 mm.
Acting goes through the scene controller only (setRCMTarget, setGrip, tissueGrasp ...), so every motion is
checked by the guarded RCM controller (limits, IK, mesh collisions).

Modes: 'full' asks PSM1 and PSM3 to grasp the sheet edges; 'cut_only' leaves the sheet on its fixed anchors.
Injected events: an occluding sphere between the endoscope and the cut site, either for a few ticks
('occlusion_transient') or until the endoscope moves ('occlusion_persistent').
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'build'))
from cut_fsm import Observation, Target  # noqa: E402
import instrument_guard  # noqa: E402
import instruments as inst  # noqa: E402
from simtools import ENDO_CONFIG, INSTRUMENT_SCENE, aliases, load_scene, matrix_to_np, read_json  # noqa: E402

import workspace  # noqa: E402
from workspace import H, NU, NV, PRETENSION  # noqa: E402

TISSUE_RGB = (0.86, 0.42, 0.48)   # flat colour of the sheet (src/tissue/tissue.lua default)
GRASP_RADIUS_M = 0.0025      # nodes this close to the closed jaw line are held: about half the 5-7 mm jaw width


def unit(v):
    return v / np.linalg.norm(v)


class SimWorld:
    def __init__(self, client, sim, args, cfg, mode='cut_only', event=None, event_tick=6, event_ticks=3):
        self.client, self.sim, self.cfg, self.mode = client, sim, cfg, mode
        self.event, self.event_tick, self.event_ticks = event, event_tick, event_ticks
        rcm_file = ENDO_CONFIG.parent / 'instruments' / 'rcm_config.json'      # written with the scene by install
        self.rcm = read_json(rcm_file if rcm_file.exists() else ENDO_CONFIG / 'rcm_config.json')
        self.task = read_json(ENDO_CONFIG / 'task.json')
        self.endo = read_json(ENDO_CONFIG / 'endoscope.json')
        self.index = {a['arm']: i for i, a in enumerate(self.rcm, 1)}
        self.arm_cfg = {a['arm']: a for a in self.rcm}
        load_scene(client, sim, args, INSTRUMENT_SCENE, 'cut_task.ttt')
        self.by = aliases(sim)
        self.ctrl = self.by['RCMController']
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
        # Fixed side anchors pin the sheet at its laid-out width; with 8 % the constraint where the free near edge
        # meets the first anchor tore at 52 % strain before any cut, so the cut-only sheet uses 5 %.
        self.pretension = 0.05 if mode == 'cut_only' else PRETENSION
        # The bead and ring of the earlier pick-and-place task sit right at the cut site; hide them for this task
        # (staged copy only, the scene file is not changed).
        # The 6 mm magenta wrist marker of that task covers the cut site once the scissors advance.
        for n in ('TaskBead', 'TaskTarget', 'psm2_TipMarker'):
            if n in self.by:
                sim.setObjectInt32Param(self.by[n], sim.objintparam_visibility_layer, 0)
        self.start()

    # -- helpers ----------------------------------------------------------------------------------------------
    def call(self, name, *a):
        return self.sim.callScriptFunction(name, self.ctrl, *a)

    def steps(self, n):
        for _ in range(n):
            self.sim.step()

    def frame(self, arm):
        j9, j10 = (matrix_to_np(self.sim.getObjectMatrix(self.by[f'{arm}_joint{n}'], -1)) for n in (9, 10))
        return dict(pivot=j10[:, 3], x=unit(j10[:, 3] - j9[:, 3]), axis=j10[:, 2],
                    tcp=np.array(self.sim.getObjectPosition(self.by[f'{arm}_tcp'], -1)))

    def target_of(self, arm):
        return self.call('getRCMStats')[self.index[arm] - 1]['target']

    def command(self, arm, yaw, pitch, ins, n=15):
        ok = bool(self.call('setRCMTarget', self.index[arm], float(yaw), float(pitch), float(ins)))
        self.steps(n)
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
        s = self.sim
        s.handleVisionSensor(self.cam)
        raw, _ = s.getVisionSensorImg(self.cam)
        d, _ = s.getVisionSensorDepth(self.cam, 1)
        self.image = Image.frombytes('RGB', (self.W, self.H), bytes(raw)).transpose(Image.FLIP_TOP_BOTTOM)
        self.depth = np.array(s.unpackFloatTable(d), dtype=float).reshape(self.H, self.W)[::-1]

    def project(self, p):
        R, t = self.camera()
        pc = R.T @ (np.asarray(p) - t)
        if pc[2] <= 0:
            return None, pc
        return (self.W / 2 - self.f * pc[0] / pc[2], self.H / 2 - self.f * pc[1] / pc[2]), pc

    def point_visible(self, p, tol=0.003, window=21, min_clear=0.35):
        uv, pc = self.project(p)
        if uv is None or not (2 <= uv[0] < self.W - 2 and 2 <= uv[1] < self.H - 2):
            return False, None
        u, v = int(round(uv[0])), int(round(uv[1]))
        r = window // 2
        v0, v1, u0, u1 = max(v - r, 0), min(v + r + 1, self.H), max(u - r, 0), min(u + r + 1, self.W)
        win = self.depth[v0:v1, u0:u1]
        # A window pixel shows the point's surroundings when it is tissue (the sheet being worked on bends and
        # wraps the blades) or nothing nearer than the point. The point is visible when enough of the window
        # does: the scissors cover part of the cut front while cutting, an occluder covers all of it.
        rgb = np.asarray(self.image, dtype=float)[v0:v1, u0:u1]
        tissue = np.linalg.norm(rgb - 255 * np.array(TISSUE_RGB), axis=2) < 40
        clear = tissue | (win >= pc[2] - tol)
        offset = max(abs(uv[0] - self.W / 2) / (self.W / 2), abs(uv[1] - self.H / 2) / (self.H / 2))
        return bool(clear.mean() >= min_clear), float(offset)

    def cut_front(self):
        """Where the next closure has to cut: on the sheet as it is now, between the two centre columns."""
        n = self.nodes()
        j = min(int(round(self.cut_length() / H)), NV - 1)
        return 0.5 * (n[j, NU // 2 - 1] + n[j, NU // 2])

    def nodes(self):
        return np.array(self.call('tissueNodes')).reshape(NV, NU, 3)

    def cut_line_tension(self):
        """Median strain of the constraints across the cut line (columns NU/2-1 and NU/2), still uncut.

        This is the tension the retraction has to create for the cut; the whole-sheet median stays near zero
        when two graspers pull a sheet that is attached along one edge only."""
        if self.sheet_origin is None:
            return 0.0
        n = self.nodes()
        L = np.linalg.norm(n[:, NU // 2] - n[:, NU // 2 - 1], axis=1) / (H * (1 - self.pretension)) - 1
        # Rows already cut, and the two rows just ahead of the cut end where the slit opens, are not tension.
        ahead = int(round(self.cut_length() / H)) + 2
        L = L[ahead:NV - 1]
        return float(np.median(L)) if len(L) else 0.0

    def cut_length(self):
        """Length of the cut from the near edge: rows whose band to the next row is cut through.

        Row j counts when, for some column gap (i, i+1) near the centre line, the horizontal constraint of row j
        and both diagonals of cell (i, j) are cut, so nothing joins the two sides there. Counted from row 0 and
        stopping at the first row that still holds."""
        if self.sheet_origin is None:
            return 0.0
        c = np.array(self.call('tissueConstraints'), dtype=np.int64).reshape(-1, 3)
        alive = {(min(a, b), max(a, b)): bool(f) for a, b, f in c}

        def dead(a, b):
            return not alive.get((min(a, b), max(a, b)), True)

        def nid(i, j):
            return j * NU + i
        rows = 0
        for j in range(NV - 1):
            through = any(dead(nid(i, j), nid(i + 1, j)) and dead(nid(i, j), nid(i + 1, j + 1)) and
                          dead(nid(i + 1, j), nid(i, j + 1)) for i in range(NU // 2 - 4, NU // 2 + 3))
            if not through:
                break
            rows += 1
        return rows * H

    # -- setup ------------------------------------------------------------------------------------------------
    def start(self):
        s = self.sim
        s.setStepping(True)
        s.setBoolParam(s.boolparam_realtime_simulation, False)
        s.startSimulation()
        s.step()

    def place_sheet(self):
        f = self.frame('psm2')
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
        tip_vis, _ = self.point_visible(np.array(self.sim.getObjectPosition(self.by['psm2_tcp'], -1)))
        st = self.call('tissueStats')
        rs = self.call('getRCMStats')
        faults = [f"{r['arm']}: {r['failures']} RCM solver failures" for r in rs if r['failures']]
        faults += [f"{r['arm']}: RCM drift {r['max_drift_m'] * 1000:.2f} mm" for r in rs if r['max_drift_m'] > 5e-4]
        return Observation(cut_site_visible=cut_vis, scissors_visible=tip_vis, cut_site_offset=offset if cut_vis else None,
                           grasped=dict(st.get('grasped') or {}), tension=self.cut_line_tension(), max_strain=float(st.get('max_strain', 0.0)),
                           torn=int(st.get('torn', 0)), cut_length_m=self.cut_length(), faults=faults)

    def check_roles(self):
        ok, report = instrument_guard.check(self.sim, self.by, self.rcm, inst.roles())
        bad = [f"{a}: {e['identified']} missing {e['missing']}" for a, e in report.items() if not e['ok']]
        self.roles_report = {a: dict(identified=e['identified'], ok=e['ok']) for a, e in report.items()}
        return ok, '; '.join(bad)

    def home(self):
        self.call('homeRCM')
        self.steps(20)
        for ready in self.task['planning']['command_sequence_checked'][:2]:
            if not self.call('setRCMTarget', self.index['psm2'], *map(float, ready)):
                return False
        self.steps(30)
        self.place_sheet()
        return True

    def hold(self):
        self.call('hold')

    def safe_stop(self, reason):
        self.call('hold')

    def localize(self):
        self.render()
        vis, _ = self.point_visible(self.sheet_origin)
        if not vis:
            return None
        R, t = self.camera()
        # Tissue must actually be seen: enough pixels of the sheet colour around the planned cut start.
        rgb = np.asarray(self.image, dtype=float)
        tissue = np.linalg.norm(rgb - 255 * np.array(TISSUE_RGB), axis=2) < 40
        if tissue.sum() < 30:
            return None
        # Grasp points from the sheet as it is now (it contracts under pre-tension once the sides are free):
        # one node in from each side edge, at mid length.
        n = self.nodes()
        j = NV // 2
        self.grasp_points = {'psm1': n[j, 1], 'psm3': n[j, NU - 2]}
        return Target(cut_start=self.sheet_origin, grasp_points=self.grasp_points)

    def grasp(self, arm, point):
        yaw, pitch, ins = self.rcm_command_for(arm, np.asarray(point))
        L = self.arm_cfg[arm]['limits']
        if abs(yaw) > L['yaw_deg'] or abs(pitch) > L['pitch_deg'] or not L['insertion_min'] <= ins <= L['insertion_max']:
            return False, (f'needs yaw {yaw:.1f} deg, pitch {pitch:.1f} deg, insertion {ins * 1000:.1f} mm; limits '
                           f"+-{L['yaw_deg']} / +-{L['pitch_deg']} deg, {L['insertion_min'] * 1000:.0f}-{L['insertion_max'] * 1000:.0f} mm")
        i = self.index[arm]
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
        self.call('setGrip', i, 0.0)
        n = self.call('tissueGrasp', i)
        if n is None or n < self.cfg['grasp']['min_nodes']:
            return False, f'grasped {n} nodes'
        return True, ''

    def release(self, arm):
        self.call('tissueRelease', self.index[arm])

    def retract(self, arm, step):
        """Pull (step > 0) or relax the grasped edge outward in the sheet plane, away from the cut line.

        Pulling back along the shaft would lift the sheet out of its plane and load only the few grasped nodes
        (that tore the sheet at 47 % strain in the first run); a lateral pull spreads the tension across it."""
        side = -1.0 if arm == 'psm1' else 1.0
        tip = np.array(self.sim.getObjectPosition(self.by[arm + '_tcp'], -1))
        # Relative command: the jaw tip of a curved instrument is off the shaft axis, so an absolute
        # point command would jump by that offset (1.4 mm for the Maryland tip).
        c0, c1 = np.array(self.rcm_command_for(arm, tip)), np.array(self.rcm_command_for(arm, tip + side * step * self.u))
        yaw, pitch, ins = np.array(self.target_of(arm)) + (c1 - c0)
        L = self.arm_cfg[arm]['limits']
        if abs(yaw) > L['yaw_deg'] or abs(pitch) > L['pitch_deg'] or not L['insertion_min'] <= ins <= L['insertion_max']:
            return False
        return self.command(arm, yaw, pitch, ins, 10)

    def open_scissors(self):
        ok = bool(self.call('setGrip', self.index['psm2'], 0.5))
        self.steps(3)
        return ok

    def close_scissors(self):
        ok = bool(self.call('setGrip', self.index['psm2'], 0.0))
        self.steps(8)
        return ok

    def align_scissors(self, cut_start, progress):
        """Lateral distance of the cut front from the blade line, and how far the blade tip is past it."""
        f = self.frame('psm2')
        p = self.cut_front()
        d = unit(f['tcp'] - f['pivot'])
        lateral = np.linalg.norm(np.cross(p - f['pivot'], d))
        if np.dot(p - f['tcp'], self.v) > 0.0025:            # tip more than 2.5 mm short of the cut front
            yaw, pitch, ins = self.target_of('psm2')
            self.command('psm2', yaw, pitch, ins + np.dot(p - f['tcp'], self.v) - 0.001, 10)
        return float(lateral)

    def advance_scissors(self, d):
        yaw, pitch, ins = self.target_of('psm2')
        ok = self.command('psm2', yaw, pitch, ins + d, 15)
        if ok:
            self.advanced += d
        return ok

    def retract_scissors(self):
        yaw, pitch, ins = self.target_of('psm2')
        self.call('setGrip', self.index['psm2'], 0.0)
        self.command('psm2', yaw, pitch, ins - self.advanced, 20)

    CAMERA_POSES = [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0), (4.5, 3.0), (-4.5, -3.0)]

    def adjust_camera(self):
        if self.camera_moves >= len(self.CAMERA_POSES):
            return False
        yaw, pitch = self.CAMERA_POSES[self.camera_moves]
        self.camera_moves += 1
        ins = self.arm_cfg['ecm']['insertion_home']
        return self.command('ecm', yaw, pitch, ins, 25)

    # -- injected events --------------------------------------------------------------------------------------
    def _events(self):
        if self.event is None or self.tick < self.event_tick:
            return
        if self.occluder is None and self.tick == self.event_tick:
            R, t = self.camera()
            p = self.cut_front()
            ray = unit(p - t)
            near = self.event == 'occlusion_persistent'
            dist, radius = (0.015, 0.0025) if near else (0.06, 0.004)
            h = self.sim.createPrimitiveShape(self.sim.primitiveshape_spheroid, [2 * radius] * 3, 0)
            self.sim.setObjectAlias(h, 'InjectedOccluder')
            self.sim.setObjectPosition(h, -1, (t + dist * ray).tolist())
            self.sim.setShapeColor(h, None, self.sim.colorcomponent_ambient_diffuse, [0.55, 0.55, 0.5])
            self.sim.setObjectInt32Param(h, self.sim.shapeintparam_static, 1)
            self.sim.setObjectInt32Param(h, self.sim.shapeintparam_respondable, 0)
            self.occluder = h
        if self.event == 'occlusion_transient' and self.occluder not in (None, -1) and \
                self.tick >= self.event_tick + self.event_ticks:
            self.sim.removeObjects([self.occluder])
            self.occluder = -1
