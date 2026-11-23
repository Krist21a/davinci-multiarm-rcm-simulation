"""Structural oracle for the single-column surface-seam coupon, never an optical sensor."""
import numpy as np

from ground_truth import GroundTruth
from workspace import H, NU, NV


def material_state(triples):
    c = np.asarray(triples)
    if c.size == 0 or c.size % 3 or c.dtype.kind not in 'iu':
        raise ValueError('invalid constraint table')
    c = c.reshape(-1, 3)
    if (c[:, :2] < 0).any() or (c[:, :2] >= NU * NV).any() or not np.isin(c[:, 2], [0, 1]).all():
        raise ValueError('invalid constraint record')
    edges = {(min(a, b), max(a, b)): bool(live) for a, b, live in c.tolist()}
    if len(edges) != len(c):
        raise ValueError('duplicate constraint')
    center = NU // 2 - 1
    faces, separated, full_rows = [], [], []
    for j in range(NV - 1):
        a = j * NU + center
        b, d, e = a + 1, a + NU, a + NU + 1
        if (b, d) in edges:
            raise ValueError('non-surface diagonal still present')
        pair = (edges[a, b] and edges[b, e] and edges[a, e],
                edges[a, e] and edges[d, e] and edges[a, d])
        faces.append(pair)
        if edges[a, e] != any(pair):
            raise ValueError('unsupported diagonal')
        separated.append(not edges[a, b] and not edges[a, e])
        missing = not any(pair)
        for i in range(center - 3, center + 4):
            if i != center:
                k = j * NU + i
                missing |= not any((edges[k, k + 1], edges[k, k + NU + 1], edges[k + 1, k + NU]))
        if missing:
            full_rows.append(j)
    for j in range(NV):
        a = j * NU + center
        support = (j > 0 and faces[j-1][1]) or (j < NV-1 and faces[j][0])
        if edges[a, a+1] != bool(support):
            raise ValueError('unsupported transverse bond')
    front = next((j for j, pair in enumerate(faces) if any(pair)), NV-1)
    if next((j for j, through in enumerate(separated) if not through), NV-1) != front:
        raise ValueError('surface and structural prefixes disagree')
    return dict(prefix_rows=front, farthest_row_end=max(full_rows, default=-1)+1,
                faces=faces, separated=separated)


class SeamTruth(GroundTruth):
    """cut_length = severed material-row prefix * H; support releases do not extend it."""

    def cut_length(self):
        return self._read('cut_length', lambda: material_state(
            self.world.call('tissueConstraints'))['prefix_rows'] * H
            if self.world.sheet_origin is not None else 0.)
