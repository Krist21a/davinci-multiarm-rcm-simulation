"""RGB-D chart for flush, row-coded bands on the non-anatomical test sheet."""
import numpy as np


ROWS = (0, 3, 6, 9, 13)
COLORS = tuple((0.88, 0.30, b) for b in (0.44, 0.53, 0.62, 0.71))
KNOTS_M = np.array(ROWS, dtype=float) * 0.001
COLOR_TOL = 9.0                     # RGB units; below half the adjacent colour separation
MIN_PAIRS = 4
BANK_INNER_M, BANK_OUTER_M = 0.002, 0.007
MAX_BANK_SKEW_M = 0.0015
MIN_ROW_SPAN_M = 0.0005             # separately resolvable knots, not a full-span prior
DEPTH_EVIDENCE_M = 0.002


def bands():
    return [dict(end_row=row, color=list(color)) for row, color in zip(ROWS[1:], COLORS)]


def measure(rgb, depth, mask, along, lateral, line_lat, *, allow_partial=False):
    """Find labelled boundaries in image adjacency, not in extrema of the point cloud.

    A silhouette is evidence only if its outside pixel sees behind the sheet. There
    is no assumed min/max length and no way to promote a cropped or unseen edge.
    """
    label = np.full(mask.shape, -1, dtype=np.int8)
    for i, color in enumerate(COLORS):
        match = np.linalg.norm(rgb - 255.0 * np.asarray(color), axis=2) <= COLOR_TOL
        label[mask & match] = i
    edges = [[] for _ in ROWS]
    for left, right in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        low = along[left] < along[right]
        s = (along[left] + along[right]) / 2
        side = (lateral[left] + lateral[right]) / 2 - line_lat
        a, b = np.where(low, label[left], label[right]), np.where(low, label[right], label[left])
        za, zb = np.where(low, depth[left], depth[right]), np.where(low, depth[right], depth[left])
        ma, mb = np.where(low, mask[left], mask[right]), np.where(low, mask[right], mask[left])
        finite = np.isfinite(s) & np.isfinite(side) & np.isfinite(za) & np.isfinite(zb) & (za > 0) & (zb > 0)
        close = np.abs(zb - za) <= DEPTH_EVIDENCE_M
        candidates = [(b == 0) & ~ma & (za - zb > DEPTH_EVIDENCE_M)]
        candidates += [(a == k-1) & (b == k) & close for k in range(1, len(COLORS))]
        candidates += [(a == len(COLORS)-1) & ~mb & (zb - za > DEPTH_EVIDENCE_M)]
        for buf, good in zip(edges, candidates):
            keep = finite & good
            buf.append(np.column_stack((s[keep], side[keep])))

    measured, counts, lower, upper, supported = [], [], [], [], []
    for j, items in enumerate(edges):
        q = np.concatenate(items)
        banks = [q[(sign*q[:, 1] >= BANK_INNER_M) & (sign*q[:, 1] <= BANK_OUTER_M), 0]
                 for sign in (-1, 1)]
        counts.append([len(bank) for bank in banks])
        if min(counts[-1]) < MIN_PAIRS:
            continue
        supported.append(j)
        measured.append(np.mean([np.median(bank) for bank in banks]))
        lower.append(min(np.percentile(bank, 10) for bank in banks))
        upper.append(max(np.percentile(bank, 90) for bank in banks))

    obs = dict(valid=False, boundary_pairs=counts)
    if not supported or supported[0] != 0 or supported[-1] != len(ROWS) - 1:
        return dict(obs, reason='coded endpoints not verified on both banks')
    if not allow_partial and len(supported) != len(ROWS):
        return dict(obs, reason='incomplete coded boundaries on the two banks')
    x, lo, hi = map(np.asarray, (measured, lower, upper))
    if np.any(np.diff(x) < MIN_ROW_SPAN_M) or np.any(np.diff(lo) < MIN_ROW_SPAN_M):
        return dict(obs, reason='coded boundaries not ordered and separated')
    if np.any(hi - lo > MAX_BANK_SKEW_M) or np.any(lo[1:] <= hi[:-1]):
        return dict(obs, reason='coded boundaries disagree across banks')
    return dict(obs, valid=True, reason='', origin_m=float(x[0]), material_knots_m=KNOTS_M[supported].tolist(),
                complete=len(supported) == len(ROWS),
                projected_knots_m=(x - x[0]).tolist(), guard_knots_m=(lo - x[0]).tolist())


def _local_indices(knots, domain, value):
    if not np.isfinite(value):
        return None
    i = int(np.searchsorted(domain, value - 1e-12))
    if i < len(domain) and abs(domain[i] - value) <= 1e-12:
        return i, i
    if i == 0 or i == len(domain):
        return None
    # Missing internal labels cannot be treated as part of a larger linear cell.
    if np.searchsorted(KNOTS_M, knots[i]) - np.searchsorted(KNOTS_M, knots[i - 1]) != 1:
        return None
    return i - 1, i


def locate(knots, projected, distance):
    """Return material progress and the design interval that supports it."""
    pair = _local_indices(knots, projected, distance)
    if pair is None:
        return None
    a, b = pair
    return float(np.interp(distance, projected, knots)), [float(knots[a]), float(knots[b])]


def project(knots, projected, material, *, stop_at_supported_row=False):
    """Map a material threshold, or stop at a verified predecessor if it is in a gap."""
    pair = _local_indices(knots, knots, material)
    if pair is not None:
        return float(np.interp(material, knots, projected))
    if stop_at_supported_row and knots[0] <= material <= knots[-1]:
        return float(projected[np.searchsorted(knots, material) - 1])
    return None
