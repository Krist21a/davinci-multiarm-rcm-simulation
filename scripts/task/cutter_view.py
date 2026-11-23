"""Geometric cutter-only occlusion prediction, not a visibility certificate."""
import numpy as np


def clear_stencil(camera, points, segments):
    """Check finite sightlines against capsule proxies in one coordinate frame."""
    camera, points = np.asarray(camera), np.asarray(points)
    rays = points - camera
    lengths = np.linalg.norm(rays, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        return None
    directions = rays / lengths[:, None]
    clear = np.ones(len(points), dtype=bool)
    for a, b, radius in segments:
        a, b = a - camera, b - camera
        edge = b - a
        edge2 = edge @ edge
        if not np.isfinite([a, b]).all() or edge2 <= 0:
            return None
        # Minimum distance between two bounded segments: interior solution plus
        # the four segment boundaries. A proxy behind the target cannot mask it.
        dot = directions @ edge
        denom = edge2 - dot * dot
        s = directions @ a
        ae = a @ edge
        t = np.divide(dot * s - ae, denom, out=np.zeros_like(s), where=denom > 1e-20)
        d = s + dot * t
        def dist(along, across):
            delta = along[:, None] * directions - a - across[:, None] * edge
            return np.einsum('ij,ij->i', delta, delta)
        best = np.minimum.reduce([
            np.where((denom > 1e-20) & (t >= 0) & (t <= 1) & (d >= 0) & (d <= lengths), dist(d, t), np.inf),
            dist(np.clip(s, 0, lengths), np.zeros_like(s)),
            dist(np.clip(directions @ b, 0, lengths), np.ones_like(s)),
            dist(np.zeros_like(s), np.full_like(s, np.clip(-ae / edge2, 0, 1))),
            dist(lengths, np.clip((lengths * dot - ae) / edge2, 0, 1)),
        ])
        clear &= best > radius * radius
    return clear


def stencil_counts(camera, points, segments):
    clear = clear_stencil(camera, points, segments)
    if clear is None:
        return 0, 0
    return int(clear.sum()), int(clear[:5].sum())
