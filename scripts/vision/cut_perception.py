"""Shadow-mode perception for the tissue-cutting task (M4).

Pure numpy: no simulator imports, so the same code runs live inside SimWorld.observe() and
offline in unit tests. The estimator consumes the ALREADY-ACQUIRED endoscope RGB and depth
plus an assumed camera pose (which may carry a seeded calibration error) and returns an
estimate for every field together with a validity flag and a reason. Nothing here is ever
used for control.

Reused math: the pinhole back-projection, the colour mask, the sphere-radius lift of a
partially seen ball and the rotvec perturbation mirror scripts/vision/visual_servo_task.py
(Endoscope.grab/locate/mask). That class is coupled to the live sim object, so importing it
in a numpy-only module is impractical; the formulas are duplicated here deliberately.

Scene model
-----------
* Tissue: flat-coloured sheet (TISSUE_RGB, same constant as sim_world.py / tissue.lua).
  Points are back-projected and a plane is fitted (centroid, normal, two principal in-plane
  axes). "Near edge" is the sheet end whose image midpoint is closest to the projected
  scissors markers — the side the instrument enters from (documented, deterministic).
* Cut: before any slit is visible, the cut start is the midpoint of the near edge and the
  direction is the in-plane principal axis that best matches the marker-derived approach
  direction (projected onto the sheet plane). Once a slit exists it is found as non-tissue
  pixels enclosed by the closed tissue mask (morphological fill-holes), restricted to a
  band around the nominal cut line; its 3-D geometry is taken on the fitted PLANE (slit
  pixels have no valid tissue depth — the depth buffer shows whatever is behind the gap).
* Scissors: two marker spheres (cyan distal, yellow proximal) parented to the cutter yaw
  frame at MARKER_DISTAL_OFF / MARKER_PROX_OFF, sharing the same lateral direction so the
  line through them is parallel to the shaft axis. The shaft axis line itself is recovered
  with the RCM point as a robot-calibration prior (config geometry, not tissue truth): the
  lateral offset direction is the marker-to-axis rejection at the distal marker. The TCP is
  then distal_marker + along_axis*(L_tcp - distal_z) - r_lat*latdir.
* Tension proxy: robust in-plane tissue width (p95 - p5 of the coordinate across the cut
  line) inside a middle band along the cut direction, normalised by the first width seen
  after the grasp frame (post_grasp flag supplied by the caller). Also the slit opening
  width measured next to the front.

Marker constants are measured on scenes/davinci_si_articulated_verified.ttt; see
SimWorld._create_markers for the staged creation.
"""
import numpy as np

TISSUE_RGB = (0.86, 0.42, 0.48)          # sim_world.TISSUE_RGB / tissue.lua sheet colour
MARKER_DISTAL_RGB = (0.0, 1.0, 1.0)      # pure cyan — not present in the scene palette
MARKER_PROX_RGB = (1.0, 1.0, 0.0)        # pure yellow — not present in the scene palette
MARKER_RADIUS = 0.0012                   # sphere radius, metres

# Yaw-frame offsets: local +y is the SHAFT axis (toward the TCP), x/z are lateral.
# Verified on the articulated scene: SimWorld.frame() maps yaw col1 -> 'distal' (shaft),
# col2 -> the jaw/hinge axis. The -z lateral faces the endoscope in the home/cutting poses
# (probed: x-offsets sit behind the housing silhouette, -z slivers stay visible), so both
# markers ride 5 mm off-axis along -z, proud of the housing.
MARKER_DISTAL_OFF = (0.0, -0.012, -0.005)  # 12 mm behind the TCP along the shaft
MARKER_PROX_OFF = (0.0, -0.024, -0.005)    # 12 mm further up the housing

MIN_TISSUE_PX = 30
MIN_MARKER_PX = 6
MIN_SLIT_PX = 8
SLIT_BAND_M = 0.003                       # lateral band around the nominal cut line
WIDTH_BAND = 1.0 / 3.0                    # middle third of the cut-direction extent


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


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


def _plane_fit(pts):
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, Vt                       # rows: e1, e2 (in-plane principal), e3 (normal)


class CutPerception:
    """One estimator instance per run; the only state is the post-grasp reference width w0."""

    def __init__(self, W, H, f, far, rcm_world, marker_radius=MARKER_RADIUS,
                 marker_offsets=(MARKER_DISTAL_OFF, MARKER_PROX_OFF), tcp_len=None):
        self.W, self.H, self.f, self.far = W, H, f, far
        self.rcm_world = np.asarray(rcm_world, dtype=float)
        self.marker_radius = marker_radius
        self.off_d, self.off_p = (np.asarray(o, dtype=float) for o in marker_offsets)
        self.tcp_len = tcp_len           # yaw-frame origin -> TCP distance along the axis
        self.w0 = None                   # first post-grasp tissue width
        self._cut_dir = None             # remembered pre-cut direction sign

    # -- field estimators -------------------------------------------------------------------------------------
    def _tissue(self, rgb, depth):
        m = _mask(rgb, TISSUE_RGB) & np.isfinite(depth) & (depth < self.far * 0.999)
        if int(m.sum()) < MIN_TISSUE_PX:
            return dict(valid=False, reason=f'tissue pixels {int(m.sum())} < {MIN_TISSUE_PX}'), m, None
        pts, _, _ = _masked_points(m, depth, self.W, self.H, self.f, self.far)
        if pts is None or len(pts) < MIN_TISSUE_PX:
            return dict(valid=False, reason='no finite-depth tissue points'), m, None
        sub = pts[:: max(1, len(pts) // 4000)]
        c, Vt = _plane_fit(sub)
        normal = Vt[2]
        # The camera sits at the origin; the sheet normal should face it.
        if normal @ c > 0:
            normal = -normal
        return dict(valid=True, reason='', centroid=c.tolist(), normal=normal.tolist(),
                    e1=Vt[0].tolist(), e2=Vt[1].tolist(), n_points=int(len(pts))), m, (c, Vt)

    def _marker(self, rgb, depth, color):
        m = _mask(rgb, color)
        if int(m.sum()) < MIN_MARKER_PX:
            return None, None, f'{int(m.sum())} px'
        pts, u, v = _masked_points(m, depth, self.W, self.H, self.f, self.far)
        if pts is None or len(pts) < MIN_MARKER_PX:
            return None, None, 'no finite-depth marker points'
        keep = np.abs(pts[:, 2] - np.median(pts[:, 2])) < 0.01
        if keep.sum() < MIN_MARKER_PX:
            return None, None, 'marker depth outliers'
        return _sphere_centre(pts[keep], self.marker_radius), (float(u.mean()), float(v.mean())), ''

    def _scissors(self, rgb, depth, rcm_cam):
        md, uvd, errd = self._marker(rgb, depth, MARKER_DISTAL_RGB)
        mp, uvp, errp = self._marker(rgb, depth, MARKER_PROX_RGB)
        if md is None:
            return dict(valid=False, reason='distal marker missing: ' + errd), None
        occluded = mp is None
        if not occluded:
            axis = unit(md - mp)         # marker line is parallel to the shaft axis
        else:
            # Proximal marker occluded (housing silhouette): the shaft always passes
            # through the known RCM port, so the marker->RCM line still gives a degraded
            # axis for the approach guess — but the scissors field is INVALID per spec.
            axis = unit(md - rcm_cam)
        lat = md - rcm_cam               # distal marker position relative to the RCM point
        latdir = unit(lat - (lat @ axis) * axis)
        # off[1] is the along-shaft component; (off[0], off[2]) is the lateral offset.
        r_lat = float(np.hypot(self.off_d[0], self.off_d[2]))
        if self.tcp_len is None:
            tip = md                     # no length calibration: distal marker is the best tip proxy
            reason = 'tip at marker (tcp_len unknown)'
        else:
            tip = md + (self.tcp_len - self.off_d[1]) * axis - r_lat * latdir
            reason = ''
        return dict(valid=not occluded,
                    reason=('proximal marker missing: ' + errp) if occluded else reason,
                    tip=tip.tolist(), axis=axis.tolist(),
                    distal_uv=list(uvd) if uvd else None,
                    proximal_uv=list(uvp) if uvp else None,
                    axis_src='markers' if not occluded else 'rcm'), \
            dict(distal_uv=uvd, axis=axis)

    def _cut(self, tissue, mask, plane, scis, depth):
        """Cut start / front / direction.

        Pre-cut: near-edge midpoint + the in-plane principal axis closest to the marker-derived
        approach. Post-cut: the slit is the run of non-tissue pixels along the cut line. It enters
        from the near edge, so it is edge-connected rather than a closed hole — it is found by
        scanning per-bin non-tissue counts along the line in-plane (2+ pixels in 2+ consecutive
        bins). Its 3-D geometry is measured on the fitted plane, because slit pixels carry the
        background's depth, not the sheet's.
        """
        if not tissue['valid']:
            return dict(valid=False, front_valid=False, reason='no tissue plane'), None
        c, Vt = plane
        e1, e2, normal = Vt[0], Vt[1], Vt[2]
        pts, _, _ = _masked_points(mask, depth, self.W, self.H, self.f, self.far)
        if scis is not None:
            approach = scis['axis'] - (scis['axis'] @ normal) * normal
            cand = max((e1, e2, -e1, -e2), key=lambda a: a @ unit(approach))
        elif self._cut_dir is not None:
            cand = np.asarray(self._cut_dir)
        else:
            return dict(valid=False, front_valid=False,
                        reason='no scissors markers to pick the near side'), None
        lat = unit(np.cross(normal, cand))
        s_dir = (pts - c) @ cand
        s_lat = (pts - c) @ lat
        lo, hi = np.percentile(s_dir, [2, 98])
        ends = (np.abs(s_dir - lo) < 0.002, np.abs(s_dir - hi) < 0.002)
        if not (ends[0].any() and ends[1].any()):
            return dict(valid=False, front_valid=False, reason='degenerate sheet extent'), None
        if scis is not None:
            # Near edge = the end whose image midpoint is closest to the distal marker pixel.
            uv_proj = []
            for end in ends:
                p = c + np.median(s_dir[end]) * cand + np.median(s_lat[end]) * lat
                uv_proj.append(self._project(p))
            d_uv = scis['distal_uv']
            near_i = int(np.argmin([np.hypot(uv[0] - d_uv[0], uv[1] - d_uv[1]) for uv in uv_proj]))
        else:
            near_i = 0 if np.median(s_dir[ends[0]]) < np.median(s_dir[ends[1]]) else 1
        direction = cand if near_i == 0 else -cand   # runs from the near edge to the far edge
        near_pts = ends[near_i]
        line_lat = float(np.median(s_lat[near_pts]))
        start = c + np.median(s_dir[near_pts]) * cand + line_lat * lat

        # Plane coordinates of every pixel's ray∩plane (slit pixels have background depth).
        sa, sb, pz = self._plane_coords(c, normal, direction, lat)
        band = np.abs(sb - line_lat) < SLIT_BAND_M
        # A slit shows whatever is BEHIND the sheet (background/far depth); instrument and
        # marker pixels are non-tissue too but sit in front of the plane and must not count.
        behind = ~mask & ((depth > pz + 0.002) | ~np.isfinite(depth) | (depth >= self.far * 0.999))
        # Non-tissue-behind pixel count per bin along the line; the slit is the contiguous
        # non-empty run starting at the near edge (1-bin gaps tolerated), capped at the far edge.
        edges = np.arange(lo - 0.004, hi + 0.004, 0.0005)
        sel = band & (sa >= edges[0]) & (sa < edges[-1])
        idx = np.clip(np.digitize(sa[sel], edges) - 1, 0, len(edges) - 2)
        non = np.bincount(idx, weights=behind[sel].astype(float), minlength=len(edges) - 1)
        # Any in-band non-tissue-behind pixel inside the sheet interior is real slit gap;
        # confirmation comes from the run length and total pixel count, not a per-bin minimum.
        slit_bin = non >= 1
        s_near = np.median(s_dir[near_pts])
        b0 = int(np.clip(np.digitize(s_near, edges) - 1, 0, len(slit_bin) - 1))
        b_hi = int(np.clip(np.digitize(hi, edges), b0 + 1, len(slit_bin)))
        end_b, gap, started = b0, 0, False
        for b in range(b0, b_hi):
            if slit_bin[b]:
                started, gap, end_b = True, 0, b
            elif started:
                gap += 1
                if gap >= 2:
                    break
        slit_len = (edges[min(end_b + 1, len(edges) - 1)] - s_near) if (started and end_b > b0) else 0.0
        slit_sel = sel & behind & (sa >= s_near - 0.001) & (sa < edges[min(end_b + 1, len(edges) - 1)])
        meta = dict(lat=lat, s_lat=line_lat)
        # A run that reaches (or nearly reaches) the far edge is a gap/torn-off region, not a
        # slit with a measurable front — the sheet's far half tilts out of the tissue colour
        # under retraction, so the gap fans to the boundary. Invalid beats wrong here.
        run_end = edges[min(end_b + 1, len(edges) - 1)]
        if slit_len > 0.001 and run_end >= hi - 0.004:
            self._cut_dir = direction
            return dict(valid=True, reason='gap run reaches the far edge: front not measurable',
                        front_valid=False, direction=direction.tolist(), start=start.tolist(),
                        mode='slit', slit_opening_m=0.0, slit_px=int(slit_sel.sum())), meta
        if slit_len > 0.001 and int(slit_sel.sum()) >= MIN_SLIT_PX:
            hs_dir = sa[slit_sel]
            hs_lat = sb[slit_sel]
            # The slit itself is the cut line: its 2-D principal axis is more faithful than
            # the approach guess once cutting has started (the sheet deforms under tension).
            if len(hs_dir) >= 6:
                q = np.stack([hs_dir - hs_dir.mean(), hs_lat - hs_lat.mean()], axis=1)
                _, _, Vq = np.linalg.svd(q, full_matrices=False)
                slit_dir = unit(Vq[0, 0] * direction + Vq[0, 1] * lat)
                if slit_dir @ direction > 0.3:
                    direction = slit_dir
                    sa, sb, pz = self._plane_coords(c, normal, direction, lat)
                    hs_dir = sa[slit_sel]
                    hs_lat = sb[slit_sel]
            order = np.argsort(hs_dir)
            far = order[-max(3, len(order) // 10):]
            front = c + float(np.median(hs_dir[far])) * direction + float(np.median(hs_lat[far])) * lat
            slit_open = float(np.percentile(hs_lat[far], 95) - np.percentile(hs_lat[far], 5))
            self._cut_dir = direction
            return dict(valid=True, reason='', front_valid=True, front=front.tolist(),
                        direction=direction.tolist(), start=start.tolist(), mode='slit',
                        slit_opening_m=slit_open, slit_px=int(slit_sel.sum())), meta
        self._cut_dir = direction
        # Pre-cut: the front is the cut start.
        return dict(valid=True, reason='', front_valid=True, front=start.tolist(),
                    direction=direction.tolist(), start=start.tolist(), mode='pre_cut',
                    slit_opening_m=0.0, slit_px=0), meta

    def _tension(self, tissue, mask, plane, cut_meta, depth, post_grasp, slit_opening):
        if not tissue['valid'] or cut_meta is None:
            return dict(valid=False, reason='no tissue plane')
        c, Vt = plane
        lat = cut_meta['lat']
        direction = np.asarray(self._cut_dir)
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
        out = dict(tissue={}, scissors={}, cut={}, tension_proxy={})
        rcm_cam = R_est.T @ (self.rcm_world - t_est)
        tissue, mask, plane = self._tissue(rgb, depth)
        scis_est, scis = self._scissors(rgb, depth, rcm_cam)
        cut, cut_meta = self._cut(tissue, mask, plane, scis, depth)
        slit_open = cut.get('slit_opening_m', 0.0) if cut else 0.0
        tension = self._tension(tissue, mask, plane, cut_meta, depth, post_grasp, slit_open)

        def world(d):
            d = dict(d)
            for k in ('centroid', 'normal', 'e1', 'e2', 'tip', 'axis', 'front', 'direction', 'start'):
                if k in d and d[k] is not None:
                    v = np.asarray(d[k])
                    d[k] = (R_est @ v + t_est if v.shape == (3,) and k in
                            ('centroid', 'tip', 'front', 'start') else R_est @ v).tolist()
            return d

        out['tissue'] = world(tissue)
        out['scissors'] = world(scis_est)
        out['cut'] = world(cut)
        out['tension_proxy'] = tension
        return out

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
