"""Pure-numpy depth gate for known simulation points; this is NOT visual localization.

Depth is metric camera-axis depth, as returned by getVisionSensorDepth(..., 1).
A clear background means no nearer surface was rendered, not that a target was detected.
"""
import numpy as np


def depth_visible(depth, uv, target_depth, tol=0.003, window=21, min_clear=0.35):
    """Require a clear 3x3 centre AND surrounding depth support, independent of RGB.

    The strict small window deliberately fails closed at depth edges: clear surroundings
    must never conceal an occluded tip/contact point. Invalid depth pixels are not support.
    The 3 mm tolerance and surrounding support threshold retain the existing defaults.
    """
    depth = np.asarray(depth)
    if (depth.ndim != 2 or not np.isfinite(uv).all() or not np.isfinite(target_depth) or
            target_depth <= 0 or not np.isfinite(tol) or tol < 0 or
            not 0 <= min_clear <= 1 or not isinstance(window, (int, np.integer)) or window < 1):
        return False
    u, v = (int(round(x)) for x in uv)
    height, width = depth.shape
    if not (1 <= u < width - 1 and 1 <= v < height - 1):
        return False

    def clear(values):
        return np.isfinite(values) & (values > 0) & (values >= target_depth - tol)

    if not clear(depth[v - 1:v + 2, u - 1:u + 2]).all():
        return False
    r = max(1, window // 2)
    surrounding = depth[max(v - r, 0):min(v + r + 1, height), max(u - r, 0):min(u + r + 1, width)]
    return bool(clear(surrounding).mean() >= min_clear)