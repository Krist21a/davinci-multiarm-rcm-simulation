"""Offline per-closure evidence on the passive coded sheet. No actuation API."""
import math
from dataclasses import dataclass

import numpy as np

from tip_marker_gate_dev import patch_features
from cell_certificate import OPEN, PRESENT, UNKNOWN, classify
from cut_perception import CutPerception

ROWS = 13
OPEN_ROW = (OPEN, OPEN)
INTACT_ROW = (PRESENT, PRESENT)


@dataclass(frozen=True)
class Exposure:
    tick: int
    optics: str | None
    faces: tuple[tuple[str, str], ...]
    local_mm: int | None
    marker_usable: bool


def read_exposure(tick, rgb, depth, *, focal, far, tcp_local, sigma=0.):
    """Use the same native RGB-D exposure for coded faces and the fixed marker gate."""
    z, im = np.asarray(depth), np.asarray(rgb)
    tcp = np.asarray(tcp_local, dtype=float)
    empty = Exposure(tick, None, (), None, False)
    if (z.ndim != 2 or im.shape != (*z.shape, 3) or im.dtype != np.uint8
            or z.dtype.kind != 'f' or tcp.shape != (3,) or not np.isfinite(tcp).all()
            or not all(math.isfinite(v) for v in (focal, far, sigma))
            or min(focal, far) <= 0 or sigma < 0):
        return empty
    h, w = z.shape
    angle = 2 * math.atan2(w / 2, focal)
    optics = next((name for name, size, degrees in (
        ('wide', (1280, 960), 70), ('narrow', (2560, 1920), 20))
        if (w, h) == size and abs(angle - math.radians(degrees)) <= 1e-6), None)
    if optics is None:
        return empty
    z = np.where((z > 0) & (z <= np.float32(far)), z, np.nan)
    coded = classify(im, z, focal=focal, principal=(w / 2, h / 2), sigma=sigma)
    faces = tuple(map(tuple, coded.get('states', ())))
    tip = CutPerception(w, h, focal, far, (0, 0, 0), tcp_local=tcp, ray_marker_fit=True)
    marker, _ = tip._scissors(im.astype(np.float32), z)
    marker.update(uv_px=marker.get('tip_uv'), xyz_m=marker.get('tip'))
    features = patch_features(z, (w, h, focal, far), marker)
    return Exposure(tick, optics, faces, coded['local_mm'], features is not None and features[-1] >= -3.)


def _usable(exposure, optics):
    return (exposure.optics == optics and exposure.marker_usable
            and len(exposure.faces) == ROWS
            and all(len(row) == 2 and all(face in (OPEN, PRESENT, UNKNOWN) for face in row)
                    for row in exposure.faces))


def _boundary(faces, n):
    # A far-side UNKNOWN is not an intact boundary; require one more intact row where possible.
    return n == ROWS or (PRESENT in faces[n] and UNKNOWN not in faces[n]
                         and (n + 1 == ROWS or faces[n + 1] == INTACT_ROW)
                         and not any(OPEN in row for row in faces[n + 1:]))


class TemporalEvidence:
    """One capture stream, seeded once; a missing or contradictory bite breaks the chain."""

    def __init__(self, loaded):
        self._prefix = 0 if (_usable(loaded, 'wide')
                             and all(row == INTACT_ROW for row in loaded.faces)) else None
        self._tick = loaded.tick
        self._closure = 0
        self._reviewed = False

    @property
    def prefix(self):
        """Observed coded prefix in material rows, not the hidden structural result."""
        return self._prefix

    def record_pair(self, closure, before, after):
        """Return newly witnessed rows, or None. Closure id and ticks only bind evidence."""
        start, self._prefix = self._prefix, None
        if (start is None or closure != self._closure + 1
                or before.tick < self._tick or after.tick != before.tick + 1
                or not _usable(before, 'wide') or not _usable(after, 'wide')):
            return None
        a, b = before.faces, after.faces
        if (any(PRESENT in row for rows in (a, b) for row in rows[:start])
                or (start and (a[start - 1] != OPEN_ROW or b[start - 1] != OPEN_ROW))
                or not _boundary(a, start)):
            return None
        end = start
        while end < ROWS and b[end] == OPEN_ROW:
            if PRESENT not in a[end]:
                return None
            end += 1
        if not _boundary(b, end):
            return None
        self._prefix, self._tick, self._closure = end, after.tick, closure
        self._reviewed = False
        return end - start

    @property
    def wants_endpoint_look(self):
        return self._prefix in (8, 9, 10) and not self._reviewed

    def review_look(self, after_tick, look, *, jaws_closed, motion_checks_ok):
        """Consume one look per eligible closure; return diagnostic-only evidence."""
        if not self.wants_endpoint_look:
            return 'UNKNOWN'
        self._reviewed = True
        if (after_tick != self._tick or look.tick <= self._tick
                or not jaws_closed or not motion_checks_ok or not _usable(look, 'narrow')
                or look.local_mm != self._prefix or not _boundary(look.faces, self._prefix)
                or look.faces[self._prefix - 1] != OPEN_ROW
                or any(PRESENT in row for row in look.faces[:self._prefix])):
            return 'UNKNOWN'
        return 'BELOW' if self._prefix == 8 else 'TARGET_MATCH'
