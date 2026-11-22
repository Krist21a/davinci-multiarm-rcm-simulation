"""RGB-D perception for the test-sheet cutting task.

Pure numpy + scipy.ndimage: no simulator imports, so the same code runs live inside
SimWorld.observe() and offline in unit tests / replay_perception.py. The estimator
consumes the ALREADY-ACQUIRED endoscope RGB and depth plus an assumed camera pose
(which may carry a seeded calibration error) and returns an estimate for every field
together with a validity flag and a reason. The same estimates drive M5b vision mode.

Fail closed: an uncertain field is invalid with a reason, never a confident wrong value.

Scene model
-----------
* Tissue: chromatic sheet mask (B > G + 5 and R > B + 30 on 0..255 pixels — pink stands
  out against the orange/brown anatomy that flooded the M4 RGB distance mask) filtered
  by a connected-component size floor, then a robust plane fit (iterative residual
  gating) on the masked depth points.
* Scissors: three marker spheres (cyan distal, yellow proximal, magenta lateral) parented
  to the cutter yaw frame at MARKER_*_OFF. A Kabsch fit of the yaw-frame offsets to the
  recovered sphere centres gives the full yaw pose (the third, non-collinear marker fixes
  roll); the TCP and jaw axis come from the instrument calibration tcp_local /
  jaw_axis_local. Fit RMS must be under FIT_RMS_MAX_M or the field is invalid.
* Cut geometry is measured in the sheet plane's coordinates. The long in-plane axis is
  the sheet width; the cut-direction prior is the in-plane perpendicular, pointing from
  the near (free) long edge — the edge closer to the estimated tip at the first tick the
  tip is valid. The choice is remembered. Pre-cut, the front is the near-edge midpoint
  (valid only with tissue evidence there). Once cut, the slit is the run of non-sheet,
  behind-plane pixels inside the sheet outline, in a lateral band around the cut line,
  connected to the near edge; the front is the slit apex. The apex must be followed by
  visible intact tissue — if whatever lies ahead is covered by something in front of the
  plane, or not seen, front_valid is False. A slit run >= SLIT_UPDATE_M refits the cut
  line position and direction from the slit itself (accepted only within DIR_UPDATE_RAD of
  the first established direction, then low-passed; rejected candidates change nothing).
* Tension proxy: robust in-plane tissue width (p95 - p5 across the cut direction) inside
  a middle band, normalised by the first width seen after the grasp frame.

Marker constants are measured on scenes/davinci_si_articulated_verified.ttt; see
SimWorld._create_markers for the staged creation. tcp_local / jaw_axis_local are the
yaw-frame coordinates of the TCP position and the jaw pointing axis (rigid: verified
constant to <1e-15 m on the nominal dump).
"""
import math

import numpy as np

import material_coordinate


def measure_bilateral(*args, junctions=False):
    from bilateral_surface import measure     # lazy: module import must stay light
    if junctions:
        from bank_junction import boundary
        return measure(*args, boundary=boundary, record_knots=True)
    return measure(*args)

TISSUE_RGB = (0.86, 0.42, 0.48)          # sim_world.TISSUE_RGB / tissue.lua sheet colour
MARKER_DISTAL_RGB = (0.0, 1.0, 1.0)      # pure cyan — not present in the scene palette
MARKER_PROX_RGB = (1.0, 1.0, 0.0)        # pure yellow — not present in the scene palette
MARKER_THIRD_RGB = (1.0, 0.0, 1.0)       # pure magenta — not present in the scene palette
MARKER_RADIUS = 0.0012                   # sphere radius, metres

# Yaw-frame offsets: local +y is the SHAFT axis (toward the TCP), x/z are lateral.
# Verified on the articulated scene: SimWorld.frame() maps yaw col1 -> 'distal' (shaft),
# col2 -> the jaw/hinge axis. The -z lateral faces the endoscope in the home/cutting poses
# (probed: x-offsets sit behind the housing silhouette, -z slivers stay visible), so all
# markers ride 5 mm off-axis along -z, proud of the housing. The third marker breaks the
# marker line's collinearity so a Kabsch fit recovers yaw-frame roll (M5a).
MARKER_DISTAL_OFF = (0.0, -0.012, -0.005)  # 12 mm behind the TCP along the shaft
MARKER_PROX_OFF = (0.0, -0.024, -0.005)    # 12 mm further up the housing
MARKER_THIRD_OFF = (0.004, -0.018, -0.005)   # lateral, mid-housing: min 66 px over all nominal ticks
                                           # (probe: -x side hides behind the housing silhouette)
MARKER_OFFS = (MARKER_DISTAL_OFF, MARKER_PROX_OFF, MARKER_THIRD_OFF)
MARKER_COLOURS = (MARKER_DISTAL_RGB, MARKER_PROX_RGB, MARKER_THIRD_RGB)

MIN_TISSUE_PX = 80                       # masked tissue pixels to accept a sheet at all
                                         # (~1000 px uncovered; occluder scraps pass a lower floor)
MIN_SHEET_COMP_PX = 25                   # connected-component floor on the chromatic mask
MIN_MARKER_PX = 6
MIN_SLIT_PX = 8
GAP_BAND_M = 0.0015                      # lateral band around the cut line for the gap scan
GAP_BIN_M = 0.0005                       # along-line bin size of the gap scan
GAP_DEPTH_M = 0.0005                     # depth behind the plane that reads as 'through the slit'
SLIT_MIN_LEN_M = 0.001                   # contiguous run that counts as a slit at all
SLIT_UPDATE_M = 0.003                    # run length that may refit the cut line/direction
RUN_GAP_BINS = 2                         # empty bins tolerated inside a slit run (1 mm)
LAT_SMOOTH = 0.6                         # low-pass on the remembered cut-line lateral:
                                         # the crack line is persistent, single-frame
                                         # lateral measurements jitter ~mm under depth noise
PLANE_GATE_M = 0.00075                   # iterative plane-fit residual gate
NEAR_EVIDENCE_M = 0.0015                 # tissue-evidence radius around the pre-cut front
MIN_NEAR_PX = 2                          # sheet pixels inside it for a valid pre-cut front
AHEAD_WINDOW_M = 0.002                   # look-ahead of the slit apex for intact tissue
MIN_AHEAD_PX = 4                         # sheet pixels required in that look-ahead
COVER_DEPTH_M = 0.002                    # depth this much in front of the plane occludes
FIT_RMS_MAX_M = 0.0003                   # marker Kabsch RMS gate
DIR_UPDATE_RAD = np.radians(5.0)         # slit refit must stay this close to the anchor
                                         # (the first near->far direction); anchor-relative,
                                         # so a bad refit cannot ratchet (m5a_amendment_v4)
DIR_SMOOTH = 0.6                         # low-pass on an accepted slit refit, same as LAT_SMOOTH
WIDTH_BAND = 1.0 / 3.0                   # middle third of the cut-direction extent


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def material_interval(along, lateral, length):
    """Estimate near/far material edges from multiple populated sheet lanes."""
    lo, hi = np.percentile(lateral, [2, 98])
    edges = np.linspace(lo, hi, 6)
    observed = []
    for a, b in zip(edges[:-1], edges[1:]):
        lane = along[(lateral >= a) & (lateral < b)]
        if len(lane) < MIN_SLIT_PX:
            continue
        lane = np.sort(lane)
        # A single sample outside either edge must not qualify an otherwise short lane.
        start, end = lane[1], lane[-2]
        if end - start >= 0.85 * length and max(start - lane[0], lane[-1] - end) <= GAP_BIN_M:
            observed.append((start, end))
    return np.median(observed, axis=0) if len(observed) >= 2 else None


def _refit_direction(anchor, prev, slit_dir):
    """Anchor-cone gated, low-passed slit refit of the cut direction (m5a_amendment_v4).

    None when the candidate disagrees with the first established direction by more than
    DIR_UPDATE_RAD; otherwise prev low-passed toward it. The cone is convex, so the blend
    can never leave it."""
    anchor, prev, slit_dir = unit(anchor), unit(prev), unit(slit_dir)
    if math.acos(float(np.clip(slit_dir @ anchor, -1, 1))) > DIR_UPDATE_RAD:
        return None
    return unit(DIR_SMOOTH * prev + (1.0 - DIR_SMOOTH) * slit_dir)


def rotvec(v):
    """Same Rodrigues rotation used by visual_servo_task.py."""
    v = np.asarray(v, dtype=float)
    a = np.linalg.norm(v)
    if a == 0:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _mask(rgb, color, tol=60.0):
    return np.linalg.norm(rgb - 255.0 * np.asarray(color), axis=2) < tol


def _sheet_mask(rgb):
    """Chromatic tissue mask: the sheet's pink (R>>B>G) against orange/brown anatomy."""
    from scipy import ndimage          # lazy: module import must stay light
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    m = (b > g + 5.0) & (r > b + 30.0)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes[1:] >= MIN_SHEET_COMP_PX
    return keep[lab]


def _back_project(u, v, z, W, H, f):
    return np.stack([(W / 2 - u) * z / f, (H / 2 - v) * z / f, z], axis=1)


def _masked_points(mask, depth, W, H, f, far, subsample=1):
    v, u = np.nonzero(mask)
    if len(u) == 0:
        return None, None, None
    if subsample > 1:
        keep = np.arange(len(u)) % subsample == 0
        u, v = u[keep], v[keep]
    z = depth[v, u]
    good = np.isfinite(z) & (z > 0) & (z < far * 0.999)
    u, v, z = u[good], v[good], z[good]
    if len(u) == 0:
        return None, None, None
    return _back_project(u, v, z, W, H, f), u, v


def _sphere_centre(pts_cam, radius):
    """Centre of a ball of known radius from its visible surface points (Endoscope.locate)."""
    c = pts_cam.mean(axis=0) + radius * unit(pts_cam.mean(axis=0))
    for _ in range(15):
        d = pts_cam - c
        n = np.linalg.norm(d, axis=1)
        J = -d / np.maximum(n, 1e-12)[:, None]
        dc = np.linalg.lstsq(J, -(n - radius), rcond=None)[0]
        c = c + dc
        if np.linalg.norm(dc) < 1e-7:
            break
    return c


def _ray_sphere_centre(points, radius, initial):
    """Fit measured range to the first sphere intersection, not to the back surface."""
    from scipy.optimize import least_squares

    ranges = np.linalg.norm(points, axis=1)
    if (len(ranges) < MIN_MARKER_PX or not np.all(np.isfinite(ranges) & (ranges > radius))
            or not np.isfinite(initial).all() or radius <= 0):
        return None
    rays = points / ranges[:, None]

    def residual(centre):
        foot = rays @ centre
        disc = radius**2 - np.sum((foot[:, None]*rays - centre)**2, axis=1)
        # Outside the apparent silhouette, penalize the missed ray separately.
        return np.r_[foot - np.sqrt(np.maximum(disc, 0)) - ranges, np.sqrt(np.maximum(-disc, 0))]

    fit = least_squares(residual, initial, loss='soft_l1', f_scale=0.0005, max_nfev=60)
    if (not fit.success or not np.isfinite(fit.x).all() or fit.x[2] <= radius
            or np.linalg.matrix_rank(fit.jac) != 3):
        return None
    return fit.x


def _plane_fit(pts):
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, Vt                       # rows: e1, e2 (in-plane principal), e3 (normal)


def _plane_fit_robust(pts, gate=PLANE_GATE_M, iters=3):
    """Iterative residual gating: blades/glare on the mask edge must not tilt the plane."""
    keep = np.ones(len(pts), bool)
    c, Vt = _plane_fit(pts)
    for _ in range(iters):
        c, Vt = _plane_fit(pts[keep])
        d = np.abs((pts - c) @ Vt[2])
        new = d < gate
        if new.sum() < MIN_TISSUE_PX or new.sum() == keep.sum():
            break
        keep = new
    c, Vt = _plane_fit(pts[keep])
    return c, Vt, keep


def _kabsch(offsets, centres):
    """Rigid transform mapping yaw-frame `offsets` onto camera-frame `centres`.

    Returns (R, t, rms). Both inputs are (k,3) with the same correspondence order.
    """
    Oc, Cc = offsets - offsets.mean(0), centres - centres.mean(0)
    U, _, Vt = np.linalg.svd(Oc.T @ Cc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = centres.mean(0) - R @ offsets.mean(0)
    rms = float(np.sqrt(((centres - (offsets @ R.T + t)) ** 2).sum(1).mean()))
    return R, t, rms


def _provable_ahead(s, band, occ, b_start, edges, sa_px, mask, depth, pz, far):
    """The one front-provability test: the look-ahead window [s, s+AHEAD+0.5 mm]
    inside the cut-line band proves no crack past s — mostly seen, at most half
    foreground-covered, at least MIN_AHEAD_PX sheet pixels, and no slit (occ) bins.
    """
    ahead = band & (sa_px >= s) & (sa_px < s + AHEAD_WINDOW_M + 0.0005)
    if int(ahead.sum()) == 0:
        return False
    seen = ahead & np.isfinite(depth) & (depth < far * 0.999)
    if int(seen.sum()) * 2 < int(ahead.sum()):
        return False
    covered = int((seen & (depth < pz - COVER_DEPTH_M)).sum())
    if covered > 0.5 * int(seen.sum()):
        return False
    # Mostly intact sheet: blades lying on the surface are neither 'behind' nor
    # >COVER_DEPTH_M covered, so a pixel count alone cannot see them — the window
    # must be over half sheet to prove the crack stopped before it.
    if int((ahead & mask).sum()) < max(MIN_AHEAD_PX, int(0.5 * int(seen.sum())) + 1):
        return False
    b0 = max(b_start, np.digitize(s, edges) - 1)
    b1 = np.digitize(s + AHEAD_WINDOW_M + 0.0005, edges) - 1
    return not occ[b0:min(len(occ), b1 + 1)].any()


class CutPerception:
    """One estimator instance per run; state: post-grasp width w0, remembered cut prior."""

    def __init__(self, W, H, f, far, rcm_world, marker_radius=MARKER_RADIUS,
                 marker_offsets=MARKER_OFFS, marker_colours=MARKER_COLOURS,
                 tcp_local=None, jaw_axis_local=None, tcp_len=None,
                 sheet_width_m=None, sheet_length_m=None, opening_axis_local=None,
                 material_pattern=False, local_material_chart=False, front_recovery=False, ray_marker_fit=False,
                 bilateral_surface=False):
        self.W, self.H, self.f, self.far = W, H, f, far
        self.rcm_world = np.asarray(rcm_world, dtype=float)
        self.marker_radius = marker_radius
        self.marker_offsets = np.asarray(marker_offsets, dtype=float)
        self.marker_colours = [tuple(c) for c in marker_colours]
        self.tcp_local = None if tcp_local is None else np.asarray(tcp_local, dtype=float)
        self.jaw_axis_local = (None if jaw_axis_local is None
                               else unit(np.asarray(jaw_axis_local, dtype=float)))
        self.opening_axis_local = (None if opening_axis_local is None
                                   else unit(np.asarray(opening_axis_local, dtype=float)))
        self.sheet_width_m = sheet_width_m   # known full strip width (truncation check)
        self.sheet_length_m = sheet_length_m
        self.material_pattern = material_pattern
        self.local_material_chart = local_material_chart
        self.front_recovery = front_recovery
        self.ray_marker_fit = ray_marker_fit
        self.bilateral_surface = bilateral_surface
        self.w0 = None                   # first post-grasp tissue width
        self._cut_dir = None             # remembered near->far cut direction (camera frame)
        self._cut_dir0 = None            # the first established direction: refit anchor
        self._cut_camera_R = None        # camera-to-world rotation of those coordinates
        self._cut_camera_t = None
        self._cut_line_point = None      # remembered point on the cut line (camera frame)
        self._slit_seen = False          # a slit was measured at least once: no pre-cut ever again
        self._front_d = None             # remembered front offset from the near edge (m):
                                         # the crack is permanent and only grows
        self._front_point = None         # remembered apex, used for its lateral position
        self._front_ext = None           # sheet along-extent when the front was updated

    @classmethod
    def from_meta(cls, meta):
        """Build from a frame-dump meta.json dict (SimWorld._dump_meta)."""
        a = meta.get('estimator_args', meta)
        sheet_width_m = a.get('sheet_width_m')
        if sheet_width_m is None and 'NU' in meta and 'spacing_m' in meta:
            sheet_width_m = (meta['NU'] - 1) * meta['spacing_m']
        sheet_length_m = a.get('sheet_length_m')
        if sheet_length_m is None and 'NV' in meta and 'spacing_m' in meta:
            sheet_length_m = (meta['NV'] - 1) * meta['spacing_m']
        return cls(a['W'], a['H'], a['f'], a['far'],
                   np.asarray(a['rcm_world'], dtype=float),
                   marker_radius=a.get('marker_radius', MARKER_RADIUS),
                   marker_offsets=a.get('marker_offsets', MARKER_OFFS),
                   marker_colours=a.get('marker_colours', MARKER_COLOURS),
                   tcp_local=a.get('tcp_local'), jaw_axis_local=a.get('jaw_axis_local'),
                   tcp_len=a.get('tcp_len'), sheet_width_m=sheet_width_m, sheet_length_m=sheet_length_m,
                   opening_axis_local=a.get('opening_axis_local'),
                   material_pattern=a.get('material_pattern', False),
                   local_material_chart=a.get('local_material_chart', False),
                   front_recovery=a.get('front_recovery', False), ray_marker_fit=a.get('ray_marker_fit', False),
                   bilateral_surface=a.get('bilateral_surface', False))

    # -- field estimators -------------------------------------------------------------------------------------
    def _tissue(self, rgb, depth):
        m = _sheet_mask(rgb) & np.isfinite(depth) & (depth < self.far * 0.999)
        if int(m.sum()) < MIN_TISSUE_PX:
            return dict(valid=False, reason=f'tissue pixels {int(m.sum())} < {MIN_TISSUE_PX}'), m, None, None
        pts, _, _ = _masked_points(m, depth, self.W, self.H, self.f, self.far)
        if pts is None or len(pts) < MIN_TISSUE_PX:
            return dict(valid=False, reason='no finite-depth tissue points'), m, None, None
        sub = pts[:: max(1, len(pts) // 4000)]
        c, Vt, _ = _plane_fit_robust(sub)
        normal = Vt[2]
        # The camera sits at the origin; the sheet normal should face it.
        if normal @ c > 0:
            normal = -normal
        return dict(valid=True, reason='', centroid=c.tolist(), normal=normal.tolist(),
                    e1=Vt[0].tolist(), e2=Vt[1].tolist(), n_points=int(len(pts))), m, (c, Vt), pts

    def _marker(self, rgb, depth, color):
        m = _mask(rgb, color)
        if int(m.sum()) < MIN_MARKER_PX:
            return None, None, f'{int(m.sum())} px', None
        pts, u, v = _masked_points(m, depth, self.W, self.H, self.f, self.far)
        if pts is None or len(pts) < MIN_MARKER_PX:
            return None, None, 'no finite-depth marker points', None
        keep = np.abs(pts[:, 2] - np.median(pts[:, 2])) < 0.01
        if keep.sum() < MIN_MARKER_PX:
            return None, None, 'marker depth outliers', None
        samples = pts[keep]
        return _sphere_centre(samples, self.marker_radius), (float(u.mean()), float(v.mean())), '', samples

    def _scissors(self, rgb, depth):
        """Yaw pose from a Kabsch fit of the three marker offsets to the sphere centres."""
        names = ('distal', 'proximal', 'third')
        centres, uvs, missing, samples = [], [], [], []
        for name, col in zip(names, self.marker_colours):
            c, uv, err, points = self._marker(rgb, depth, col)
            if c is None:
                missing.append(f'{name} marker missing: {err}')
            centres.append(c)
            uvs.append(uv)
            samples.append(points)
        if missing:
            return dict(valid=False, reason='; '.join(missing),
                        distal_uv=list(uvs[0]) if uvs[0] else None,
                        proximal_uv=list(uvs[1]) if uvs[1] else None,
                        third_uv=list(uvs[2]) if uvs[2] else None), None
        R, t, rms = _kabsch(self.marker_offsets, np.stack(centres))
        base = dict(distal_uv=list(uvs[0]), proximal_uv=list(uvs[1]), third_uv=list(uvs[2]),
                    fit_rms_m=rms)
        if not math.isfinite(rms) or rms > FIT_RMS_MAX_M:
            return dict(valid=False, reason=f'marker fit rms {rms * 1000:.2f} mm > '
                                            f'{FIT_RMS_MAX_M * 1000:.2f} mm', **base), None
        if self.ray_marker_fit:
            refined = [_ray_sphere_centre(pts, self.marker_radius, c) for pts, c in zip(samples, centres)]
            if any(v is None for v in refined):
                return dict(valid=False, reason='unresolved ray/sphere fit', **base), None
            R, t, rms = _kabsch(self.marker_offsets, np.stack(refined))
            base.update(original_rms_m=base['fit_rms_m'], fit_rms_m=rms)
            if not math.isfinite(rms) or rms > FIT_RMS_MAX_M:
                return dict(valid=False, reason='ray marker rigid-fit RMS out of tolerance', **base), None
        if self.tcp_local is None:
            return dict(valid=False, reason='tcp_local calibration missing', **base), None
        tip = R @ self.tcp_local + t
        jaw = None if self.jaw_axis_local is None else unit(R @ self.jaw_axis_local)
        axis = R[:, 1]                       # yaw-frame +y is the shaft axis
        est = dict(valid=True, reason='', tip=tip.tolist(), axis=axis.tolist(),
                   tip_uv=list(self._project(tip)), **base)
        if jaw is not None:
            est['jaw_axis'] = jaw.tolist()
        if self.opening_axis_local is not None:
            est['opening_axis'] = unit(R @ self.opening_axis_local).tolist()
        info = dict(tip=tip, axis=axis, tip_uv=est['tip_uv'])
        return est, info

    def _gap_scan(self, mask, sa_px, sb_px, pz, depth, line_lat, s_near, s_far):
        """Slit-gap scan: behind-plane pixels inside the cut-line band.

        The slit shows whatever is behind the sheet — far background, invalid depth,
        or surface > GAP_DEPTH_M below the plane — while foreground pixels (blades,
        occluders) never count. The run must start at the near edge (the cut enters
        there), may skip RUN_GAP_BINS empty bins, and stops at the far edge.

        Returns run_end (the apex boundary), run_len, slit_sel, recovery (intact
        sheet seen past the apex) and reach_far.
        """
        edges = np.arange(s_near - 0.001, s_far + 0.002, GAP_BIN_M)
        band = (np.abs(sb_px - line_lat) < GAP_BAND_M) & (sa_px >= edges[0]) & (sa_px < edges[-1])
        behind = (~mask & band & ((depth > pz + GAP_DEPTH_M) | ~np.isfinite(depth) |
                                  (depth >= self.far * 0.999)))
        idx = np.clip(np.digitize(sa_px[band], edges) - 1, 0, len(edges) - 2)
        occ = np.bincount(idx, weights=behind[band].astype(float), minlength=len(edges) - 1) >= 2
        # Edge guard: the first ~1 mm past s_near is the edge transition — background
        # pixels there are the sheet boundary, not slit. The run must start inside the
        # sheet but still within ~2.5 mm of the edge (the cut enters at the edge).
        b_start = int(np.clip(np.digitize(s_near + 0.001, edges) - 1, 0, len(edges) - 2))
        first = np.flatnonzero(occ[b_start:])
        if len(first) == 0 or edges[b_start + int(first[0])] > s_near + 0.0025:
            return dict(run_end=None, run_len=0.0, slit_sel=np.zeros(mask.shape, bool),
                        recovery=False, reach_far=False,
                        band=band, occ=occ, b_start=b_start, edges=edges)
        lo = b_start + int(first[0])
        end_b, gap_b = lo, 0
        for b in range(lo + 1, len(edges) - 1):
            if occ[b]:
                end_b, gap_b = b, 0
            else:
                gap_b += 1
                if gap_b > RUN_GAP_BINS:
                    break
        run_end = float(edges[min(end_b + 1, len(edges) - 1)])
        reach_far = run_end >= s_far - 0.001
        slit_sel = behind & (sa_px >= edges[lo]) & (sa_px < run_end)
        recovery = (not reach_far and
                    _provable_ahead(run_end, band, occ, b_start, edges,
                                    sa_px, mask, depth, pz, self.far))
        return dict(run_end=run_end, run_len=run_end - float(edges[lo]),
                    slit_sel=slit_sel, recovery=recovery, reach_far=reach_far,
                    band=band, occ=occ, b_start=b_start, edges=edges)

    def _cut(self, tissue, mask, plane, pts, scis, depth, rgb):
        """Cut start / front / direction in the fitted sheet plane.

        The sheet is a strip: Vt[0] spans its WIDTH (the lateral axis), Vt[1] its length
        (the cut-direction candidate). The near edge is the strip end closer to the
        estimated tip on the first tick the tip is valid; the choice is remembered.
        """
        if not tissue['valid']:
            return dict(valid=False, front_valid=False, reason='no tissue plane'), None
        c, Vt = plane
        lat_ax, len_ax = Vt[:2]
        normal = np.asarray(tissue['normal'], dtype=float)

        prior = self._cut_dir
        if self._cut_dir is None:
            if scis is None or scis.get('tip') is None:
                return dict(valid=False, front_valid=False,
                            reason='no valid tip to pick the near edge'), None
            s_len = (pts - c) @ len_ax
            lo, hi = np.percentile(s_len, [2, 98])
            tip_uv = scis['tip_uv']
            ends = (np.abs(s_len - lo) < 0.002, np.abs(s_len - hi) < 0.002)
            if not (ends[0].any() and ends[1].any()):
                return dict(valid=False, front_valid=False,
                            reason='degenerate sheet extent'), None
            s_w = (pts - c) @ lat_ax
            d_uv = []
            for end in ends:
                p = c + np.median(s_len[end]) * len_ax + np.median(s_w[end]) * lat_ax
                d_uv.append(np.hypot(*(np.asarray(self._project(p)) - np.asarray(tip_uv))))
            prior = len_ax if int(np.argmin(d_uv)) == 0 else -len_ax

        projected = prior - (prior @ normal) * normal
        if np.linalg.norm(projected) < 1e-8:
            return dict(valid=False, front_valid=False, reason='cut direction normal to fitted plane'), None
        direction = unit(projected)
        lat_ax = unit(np.cross(normal, direction))
        sa = (pts - c) @ direction
        sb = (pts - c) @ lat_ax
        s_near, s_far = np.percentile(sa, [2, 98])
        start_sa = float(np.percentile(sa, 1))
        extent = s_far - s_near

        # Cut-line lateral before any slit: the midpoint of the lateral extent of
        # the ~3 mm of sheet at the near edge. A median is biased when the view
        # truncates the strip; the extent midpoint is not.
        near = sa < s_near + 0.003
        w_src = sb[near] if int(near.sum()) >= 8 else sb
        w_lo, w_hi = np.percentile(w_src, [2, 98])
        near_midpoint = 0.5 * (w_lo + w_hi)
        line_lat = (near_midpoint if self._cut_line_point is None
                    else float((self._cut_line_point - c) @ lat_ax))

        # A visible extent much narrower than the real strip makes the lateral
        # centre unreliable — fail closed on the front (the direction stays valid).
        # min/max over the component-cleaned mask; percentiles would under-read
        # even a fully visible sheet (the mask never covers the outer edge px).
        def truncated(samples):
            return self.sheet_width_m is not None and np.ptp(samples) < 0.85 * self.sheet_width_m

        if truncated(sb):
            return dict(valid=True, direction=direction.tolist(), front_valid=False,
                        mode='slit' if self._slit_seen else 'pre_cut',
                        reason='sheet lateral extent truncated'), None
        if self._cut_dir is None:
            self._cut_dir = self._cut_dir0 = direction

        sa_px, sb_px, pz = self._plane_coords(c, normal, direction, lat_ax)
        chart = None
        if self.material_pattern:
            chart = material_coordinate.measure(rgb, depth, mask, sa_px, sb_px, line_lat,
                                                allow_partial=self.local_material_chart)
            if chart['valid']:
                start_sa = s_near = chart['origin_m']
                s_far = s_near + chart['projected_knots_m'][-1]
                extent = s_far - s_near
        res = self._gap_scan(mask, sa_px, sb_px, pz, depth, line_lat, s_near, s_far)

        # A long enough slit defines the cut line itself: refit within the anchor cone.
        if res['run_len'] >= SLIT_UPDATE_M and int(res['slit_sel'].sum()) >= 6:
            hs_dir, hs_lat = sa_px[res['slit_sel']], sb_px[res['slit_sel']]
            q = np.stack([hs_dir - hs_dir.mean(), hs_lat - hs_lat.mean()], axis=1)
            _, _, Vq = np.linalg.svd(q, full_matrices=False)
            slit_dir = unit(Vq[0, 0] * direction + Vq[0, 1] * lat_ax)
            if slit_dir @ self._cut_dir0 < 0:
                slit_dir = -slit_dir
            new_dir = _refit_direction(self._cut_dir0, direction, slit_dir)
            if new_dir is not None and new_dir @ self._cut_dir0 >= math.cos(DIR_UPDATE_RAD):
                new_lat_ax = unit(np.cross(normal, new_dir))
                new_sb = (pts - c) @ new_lat_ax
                if truncated(new_sb):
                    return dict(valid=True, direction=direction.tolist(), front_valid=False,
                                mode='slit' if self._slit_seen else 'pre_cut',
                                reason='sheet lateral extent truncated'), None
                line_point = (self._cut_line_point if self._cut_line_point is not None
                              else c + s_near * direction + line_lat * lat_ax)
                direction = new_dir
                self._cut_dir = direction
                lat_ax, sb = new_lat_ax, new_sb
                line_lat = float((line_point - c) @ lat_ax)
                sa = (pts - c) @ direction
                s_near, s_far = np.percentile(sa, [2, 98])
                start_sa = float(np.percentile(sa, 1))
                extent = s_far - s_near
                sa_px, sb_px, pz = self._plane_coords(c, normal, direction, lat_ax)
                if self.material_pattern:
                    chart = material_coordinate.measure(rgb, depth, mask, sa_px, sb_px, line_lat,
                                                        allow_partial=self.local_material_chart)
                    if chart['valid']:
                        start_sa = s_near = chart['origin_m']
                        s_far = s_near + chart['projected_knots_m'][-1]
                        extent = s_far - s_near
                res = self._gap_scan(mask, sa_px, sb_px, pz, depth, line_lat, s_near, s_far)

        slit_px_n = int(res['slit_sel'].sum())
        slit_meas = res['run_len'] >= SLIT_MIN_LEN_M and slit_px_n >= 2

        # Once a slit is measured, the crack line is the median lateral of its gap
        # pixels, low-passed — a single frame jitters, the crack line is persistent.
        if slit_meas:
            slit_lat = float(np.median(sb_px[res['slit_sel']]))
            new_lat = slit_lat if self._cut_line_point is None else \
                LAT_SMOOTH * line_lat + (1.0 - LAT_SMOOTH) * slit_lat
            if new_lat != line_lat:
                # The band moved: re-scan around the updated line before scoring —
                # a fraction of a millimetre of band shift changes the look-ahead.
                line_lat = new_lat
                res = self._gap_scan(mask, sa_px, sb_px, pz, depth,
                                     line_lat, s_near, s_far)
                slit_px_n = int(res['slit_sel'].sum())
                slit_meas = res['run_len'] >= SLIT_MIN_LEN_M and slit_px_n >= 2
            if slit_meas:
                self._slit_seen = True
                self._cut_line_point = c + res['run_end'] * direction + line_lat * lat_ax

        start = c + start_sa * direction + line_lat * lat_ax
        cut = dict(valid=True, reason='', direction=direction.tolist(), start=start.tolist())
        if chart is not None:
            cut['material_chart'] = chart
            if chart['valid']:
                cut['material_start'] = start.tolist()
        elif self.sheet_length_m:
            bounds = material_interval(sa, sb, self.sheet_length_m)
            if bounds is not None:
                near_edge, far_edge = bounds
                cut['material_start'] = (c + near_edge * direction + line_lat * lat_ax).tolist()
                cut['material_scale'] = max(1.0, self.sheet_length_m / (far_edge - near_edge))
        meta = dict(lat=lat_ax, direction=direction)

        def provable(s):
            return _provable_ahead(s, res['band'], res['occ'], res['b_start'],
                                   res['edges'], sa_px, mask, depth, pz, self.far)

        # Candidate fronts — slit apex | memory | pre-cut start — each admitted only
        # if its look-ahead proves the crack ends there (see _provable_ahead). Once a
        # slit has ever been measured there is no fall-back to the pre-cut start.
        cands = []                            # (source, along, lateral)
        distal = []
        if slit_meas:
            cands.append(('slit', res['run_end'], line_lat))
        if self._slit_seen:
            # A displaced slit needs the two banks in the same distal bin, not
            # simply missing colour off to one side of the strip.
            for lat in (line_lat - GAP_BAND_M / 2, line_lat + GAP_BAND_M / 2):
                lane = self._gap_scan(mask, sa_px, sb_px, pz, depth, lat, s_near, s_far)
                if lane['run_len'] < SLIT_MIN_LEN_M or lane['reach_far']:
                    continue
                end = lane['run_end']
                dist = sb_px - lat
                bin_mask = mask & (sa_px >= end - GAP_BIN_M) & (sa_px < end)
                if not all(np.any(bin_mask & (side * dist >= GAP_BAND_M)
                                  & (side * dist < 2 * GAP_BAND_M)) for side in (-1, 1)):
                    continue
                distal.append(end)
                if lane['recovery']:
                    cands.append(('bank_slit', end, lat))
        if self._front_d is not None and \
                (self._front_ext is None or abs(extent - self._front_ext) < 0.003):
            memory_lat = float((self._front_point - c) @ lat_ax)
            cands.append(('memory', s_near + self._front_d, memory_lat))
        evidence = int((((sa_px - start_sa) ** 2 +
                         (sb_px - line_lat) ** 2 < NEAR_EVIDENCE_M ** 2) & mask).sum())
        if not self._slit_seen and evidence >= MIN_NEAR_PX:
            cands.append(('pre_cut', start_sa, line_lat))

        ok = [cd for cd in cands if (cd[0] == 'bank_slit' or provable(cd[1]))
              and not any(end > cd[1] + GAP_BIN_M + 1e-12 for end in distal)]
        if not ok:
            if self._slit_seen:
                reason = 'slit seen but no front candidate provable ahead'
            elif res['run_end'] is not None:
                reason = 'sub-minimum slit and no provable pre-cut front'
            else:
                reason = f'no provable pre-cut front ({evidence} px at the near edge)'
            return dict(cut, front_valid=False,
                        mode='slit' if self._slit_seen else 'pre_cut',
                        slit_px=slit_px_n, reason=reason), meta

        src, front_s, front_lat = max(ok, key=lambda kv: kv[1])
        if self.front_recovery and src == 'memory':
            # A sub-bin difference is not evidence that the crack grew beyond this
            # directly observed apex. A longer disagreement keeps the fix invalid.
            direct = [cd for cd in ok if cd[0] in ('slit', 'bank_slit') and front_s - cd[1] <= GAP_BIN_M + 1e-12]
            if direct:
                src, front_s, front_lat = max(direct, key=lambda kv: kv[1])
        # The remembered front advances only from a provable slit apex; a remembered
        # crack cannot shrink (cracks are permanent).
        if src in ('slit', 'bank_slit') and (self._front_d is None or front_s - s_near >= self._front_d):
            self._front_d = front_s - s_near
            self._front_point, self._front_ext = c + front_s * direction + front_lat * lat_ax, extent
        front_s = min(max(front_s, s_near), s_far - 0.0005)
        hs_lat = sb_px[res['slit_sel'] & (sa_px > front_s - 0.0015)]
        slit_open = float(np.percentile(hs_lat, 95) - np.percentile(hs_lat, 5)) \
            if len(hs_lat) >= 2 else 0.0
        return self._front_out(cut, front_s, front_lat, c, direction, lat_ax,
                               mode=src, slit_open=slit_open, slit_px=slit_px_n), meta

    def _front_out(self, cut, s_apex, s_lat, c, direction, lat_ax, mode, slit_open, slit_px):
        front_cam = c + s_apex * direction + s_lat * lat_ax
        uv = self._project(front_cam)
        offset = max(abs(uv[0] - self.W / 2) / (self.W / 2), abs(uv[1] - self.H / 2) / (self.H / 2))
        cut.update(front_valid=True, front=front_cam.tolist(), front_uv=list(uv),
                   cut_site_offset=float(offset), mode=mode,
                   slit_opening_m=float(slit_open), slit_px=int(slit_px))
        chart = cut.get('material_chart')
        if chart and chart['valid'] and mode != 'memory':
            distance = float((front_cam - cut['material_start']) @ direction)
            estimate = material_coordinate.locate(chart['material_knots_m'], chart['projected_knots_m'], distance)
            if estimate is not None:
                cut['progress_m'], cut['progress_cell_m'] = estimate
        elif 'material_scale' in cut:
            cut['progress_m'] = max(0.0, float((front_cam - cut['material_start']) @ direction)
                                    * cut['material_scale'])
        return cut

    def _tension(self, tissue, mask, plane, cut_meta, depth, post_grasp, slit_opening):
        if not tissue['valid'] or cut_meta is None:
            return dict(valid=False, reason='no tissue plane')
        c, Vt = plane
        lat = cut_meta['lat']
        direction = cut_meta['direction']
        pts, _, _ = _masked_points(mask, depth, self.W, self.H, self.f, self.far)
        s_dir = (pts - c) @ direction
        s_lat = (pts - c) @ lat
        lo, hi = np.percentile(s_dir, [2, 98])
        mid = 0.5 * (lo + hi)
        half = WIDTH_BAND * 0.5 * (hi - lo)
        band = np.abs(s_dir - mid) < half
        if band.sum() < 20:
            return dict(valid=False, reason='too few mid-band points')
        width = float(np.percentile(s_lat[band], 95) - np.percentile(s_lat[band], 5))
        if post_grasp and self.w0 is None:
            self.w0 = width
        out = dict(valid=True, reason='', width_m=width, slit_opening_m=float(slit_opening))
        if self.w0:
            out['strain_proxy'] = width / self.w0 - 1.0
        else:
            out['strain_proxy'] = None
            out['reason'] = 'no post-grasp reference width yet'
        return out

    # -- entry point ------------------------------------------------------------------------------------------
    def estimate(self, rgb, depth, R_est, t_est, post_grasp=False):
        """rgb (H,W,3) 0..255, depth (H,W) metres, R_est/t_est the ASSUMED camera pose
        (camera->world). Returns a JSON-safe dict of fields with validity + reasons."""
        rgb = np.asarray(rgb, dtype=float)
        depth = np.asarray(depth, dtype=float)
        if self.bilateral_surface:
            return self._estimate_bilateral(rgb, depth, R_est, t_est)
        if self._cut_camera_R is not None and self._cut_dir is not None:
            current_from_previous = R_est.T @ self._cut_camera_R
            self._cut_dir = unit(current_from_previous @ self._cut_dir)
            self._cut_dir0 = unit(current_from_previous @ self._cut_dir0)
            translation = R_est.T @ (self._cut_camera_t - t_est)
            if self._cut_line_point is not None:
                self._cut_line_point = current_from_previous @ self._cut_line_point + translation
            if self._front_point is not None:
                self._front_point = current_from_previous @ self._front_point + translation
        self._cut_camera_R = R_est.copy()
        self._cut_camera_t = t_est.copy()
        out = dict(tissue={}, scissors={}, cut={}, tension_proxy={})
        tissue, mask, plane, pts = self._tissue(rgb, depth)
        scis_est, scis = self._scissors(rgb, depth)
        cut, cut_meta = self._cut(tissue, mask, plane, pts, scis, depth, rgb)
        slit_open = cut.get('slit_opening_m', 0.0) if cut else 0.0
        tension = self._tension(tissue, mask, plane, cut_meta, depth, post_grasp, slit_open)

        def world(d):
            d = dict(d)
            for k in ('centroid', 'normal', 'e1', 'e2', 'tip', 'axis', 'jaw_axis', 'opening_axis',
                      'front', 'direction', 'start', 'material_start'):
                if k in d and d[k] is not None:
                    v = np.asarray(d[k])
                    d[k] = (R_est @ v + t_est if v.shape == (3,) and k in
                            ('centroid', 'tip', 'front', 'start', 'material_start') else R_est @ v).tolist()
            return d

        out['tissue'] = world(tissue)
        out['scissors'] = world(scis_est)
        out['cut'] = world(cut)
        out['tension_proxy'] = tension
        return out

    def _estimate_bilateral(self, rgb, depth, R, t):
        # Static replay uses an identity camera pose; a caller can request the same geometry in world coordinates.
        cut = measure_bilateral(rgb, depth, self.W, self.H, self.f, self.far,
                                junctions=self.bilateral_surface == 'junction')
        tool, _ = self._scissors(rgb, depth)
        def transport(field, positions, vectors):
            result = dict(field)
            for k in (*positions, *vectors):
                if field.get(k) is not None:
                    v = R @ np.asarray(field[k], dtype=float)
                    result[k] = (v + t if k in positions else v).tolist()
            return result
        cut = transport(cut, ('front',), ('direction', 'normal'))
        tool = transport(tool, ('tip',), ('axis','jaw_axis','opening_axis'))
        tissue = dict(valid=cut.get('banks_visible', False), normal=cut.get('normal'))
        return dict(cut=cut, scissors=tool, tissue=tissue,
                    tension_proxy=dict(valid=False, reason='not evaluated by local endpoint observer'))

    def _project(self, pc):
        return (self.W / 2 - self.f * pc[0] / pc[2], self.H / 2 - self.f * pc[1] / pc[2])

    def _plane_coords(self, c, normal, e_a, e_b):
        """In-plane (s_a, s_b) coordinates of every pixel's ray∩plane, plus the plane depth
        along each ray (z of the intersection — slit pixels sit behind it)."""
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        d = np.stack([-(u - self.W / 2) / self.f, -(v - self.H / 2) / self.f,
                      np.ones((self.H, self.W))], axis=-1)
        denom = d @ normal
        s = np.where(np.abs(denom) > 1e-9, (c @ normal) / np.where(np.abs(denom) > 1e-9, denom, 1), 0)
        pts = d * s[..., None] - c
        return pts @ e_a, pts @ e_b, s
