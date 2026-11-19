"""Ground-truth sensing boundary for SimWorld (scripts/task/sim_world.py).

Every read the FSM could treat as sensor truth goes through this one object so a future
vision pipeline has a single swap surface. Each method counts its calls in .reads,
reported as truth_reads in the run report.

These are SENSING reads only: actuation (setRCMTarget/setGrip/tissueGrasp) and the
control-motion helpers that need the live sim state (rcm_command_for, frame) keep their
direct sim access in sim_world.py.

M5b: every read also carries a purpose tag (reads_by_purpose). In vision-control mode
(cut.control_source == 'vision') strict=True makes an untagged read raise
TruthReadInControl, so a control path can never silently depend on truth.
"""
import contextlib

import numpy as np

from workspace import H, NU, NV


class TruthReadInControl(RuntimeError):
    """A GroundTruth method ran with no declared purpose while strict mode was on."""


class GroundTruth:
    def __init__(self, world):
        self.world = world          # SimWorld; only .call(), .sim, .by and sheet state are read
        self.reads = {}
        self.reads_by_purpose = {}
        self.strict = False         # vision mode: untagged reads raise TruthReadInControl
        self._purposes = []

    @contextlib.contextmanager
    def purpose(self, name):
        """Tag every read inside the block (nested blocks inherit the innermost tag)."""
        self._purposes.append(name)
        try:
            yield
        finally:
            self._purposes.pop()

    def _read(self, name, fn):
        self.reads[name] = self.reads.get(name, 0) + 1
        purpose = self._purposes[-1] if self._purposes else 'untagged'
        if purpose == 'untagged' and self.strict:
            raise TruthReadInControl(name)
        by = self.reads_by_purpose.setdefault(purpose, {})
        by[name] = by.get(name, 0) + 1
        return fn()

    # -- tissue model -----------------------------------------------------------------------------------------
    def tissue_stats(self):
        """tissueStats(): strain quantiles, per-arm grasped node counts, torn/cut counters."""
        return self._read('tissue_stats', lambda: self.world.call('tissueStats'))

    def nodes(self):
        """Sheet node positions as an (NV, NU, 3) array, from tissueNodes()."""
        return self._read('nodes', lambda: np.array(self.world.call('tissueNodes')).reshape(NV, NU, 3))

    def tcp(self, arm):
        """Tool-tip position used for sensing (e.g. the tip visibility gate), not for control."""
        return self._read('tcp', lambda: np.array(
            self.world.sim.getObjectPosition(self.world.by[arm + '_tcp'], -1)))

    # -- derived sheet state ----------------------------------------------------------------------------------
    def cut_front(self):
        """Where the next closure has to cut: on the sheet as it is now, between the two centre columns."""
        def go():
            n = self.nodes()
            j = min(int(round(self.cut_length() / H)), NV - 1)
            return 0.5 * (n[j, NU // 2 - 1] + n[j, NU // 2])
        return self._read('cut_front', go)

    def cut_length(self):
        """Length of the cut from the near edge: rows whose band to the next row is cut through.

        Row j counts when, for some column gap (i, i+1) near the centre line, the horizontal constraint of row j
        and both diagonals of cell (i, j) are cut, so nothing joins the two sides there. Counted from row 0 and
        stopping at the first row that still holds."""
        def go():
            if self.world.sheet_origin is None:
                return 0.0
            c = np.array(self.world.call('tissueConstraints'), dtype=np.int64).reshape(-1, 3)
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
        return self._read('cut_length', go)

    def cut_line_tension(self):
        """Median strain of the constraints across the cut line (columns NU/2-1 and NU/2), still uncut.

        This is the tension the retraction has to create for the cut; the whole-sheet median stays near zero
        when two graspers pull a sheet that is attached along one edge only."""
        def go():
            if self.world.sheet_origin is None:
                return 0.0
            n = self.nodes()
            L = np.linalg.norm(n[:, NU // 2] - n[:, NU // 2 - 1], axis=1) / (H * (1 - self.world.pretension)) - 1
            # Rows already cut, and the two rows just ahead of the cut end where the slit opens, are not tension.
            ahead = int(round(self.cut_length() / H)) + 2
            L = L[ahead:NV - 1]
            return float(np.median(L)) if len(L) else 0.0
        return self._read('cut_line_tension', go)
