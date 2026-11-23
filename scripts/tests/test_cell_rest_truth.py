"""CellRestTruth measured-rest tension + real Lua readback on all three coupons.

Python side drives a stub world whose transverse distances are rest[j]*(1+s_j)
so expected strains and medians are known exactly; cut_length is stubbed to
isolate the window/divisor (SeamTruth.cut_length itself is covered elsewhere).
Lua side exercises cellTransverseRestLengths() on the actual narrow / symmetric
/ asymmetric staged chains.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))

from test_lua_safety import FIXTURE, LEGACY_SOURCES as SOURCES, LuaRuntime  # noqa: E402
from ground_truth import GroundTruth, TruthReadInControl  # noqa: E402
from cell_rest_truth import CellRestTruth  # noqa: E402
from workspace import H, NU, NV  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PARENT = (ROOT / 'src/tissue/ruler_print.lua', ROOT / 'src/tissue/surface_seam.lua')
READBACK = (ROOT / 'src/tissue/cell_rest_readback.lua').read_text(encoding='utf-8')
SPEC = ('{nu=24,nv=14,spacing=0.001,origin={0,0,0},u={1,0,0},v={0,1,0},'
        'pretension=0.08,anchor_sides=false,anchor_from_row=3,grasp_radius=0.0025}')


def _truth(rest, strains, pretension=0.08, cut_length=0.0):
    rest = np.asarray(rest, float)
    row_rest = rest if rest.shape == (NV,) else np.resize(rest, NV)  # positions stay valid when rest is intentionally invalid
    n = np.zeros((NV, NU, 3))
    n[:, NU // 2, 0] = row_rest * (1 + np.asarray(strains, float))
    world = SimpleNamespace(
        sheet_origin=np.zeros(3), pretension=pretension, cutline_rest_m=rest,
        call=lambda name: list(n.ravel()) if name == 'tissueNodes' else None)
    truth = CellRestTruth(world)
    truth._positions = n
    truth.cut_length = lambda: cut_length
    return truth


class CellRestTruthTests(unittest.TestCase):
    def test_heterogeneous_rest_known_strains_and_window(self):
        rest = np.full(NV, 0.001 * 0.92)
        rest[5] = 0.003 * 0.92
        rest[6] = 0.005 * 0.92
        strains = np.arange(NV) * 0.01 - 0.04          # known, some negative
        truth = _truth(rest, strains)
        self.assertEqual(truth.cut_length(), 0.0)
        self.assertAlmostEqual(truth.cut_line_tension(), float(np.median(strains[2:NV - 1])))
        truth.cut_length = lambda: 3 * H               # prefix rows 0-2 cut -> +2 window
        self.assertAlmostEqual(truth.cut_line_tension(), float(np.median(strains[5:NV - 1])))

    def test_excluded_rows_do_not_affect(self):
        rest = np.full(NV, 0.001 * 0.92)
        strains = np.arange(NV) * 0.01
        truth = _truth(rest, strains)
        base = truth.cut_line_tension()
        for j in (0, 1, NV - 1):                       # excluded: prefix window + last row
            truth._positions[j, NU // 2, 0] += 0.9
        self.assertEqual(truth.cut_line_tension(), base)

    def test_uniform_rest_matches_ground_truth(self):
        strains = np.linspace(-0.02, 0.08, NV)
        rest = np.full(NV, H * (1 - 0.08))
        truth = _truth(rest, strains)
        # The uniform nominal basis applies only on the explicitly frozen legacy source.
        world = SimpleNamespace(sheet_origin=np.zeros(3), pretension=0.08,
                                tissue_source=ROOT / 'src' / 'tissue' / 'legacy' / 'tissue_v1.lua',
                                call=lambda name: list(truth._positions.ravel()))
        gt = GroundTruth(world)
        gt.cut_length = lambda: 0.0
        self.assertAlmostEqual(truth.cut_line_tension(), gt.cut_line_tension())
        self.assertAlmostEqual(truth.cut_line_tension(), float(np.median(strains[2:NV - 1])))

    def test_production_ground_truth_uses_measured_heterogeneous_rest(self):
        # Intentionally heterogeneous rests: the scalar nominal basis would give a
        # different median than the measured per-row readback.
        rest = np.full(NV, 0.001)
        rest[5], rest[6] = 0.003, 0.005
        strains = np.arange(NV) * 0.01 - 0.04
        n = np.zeros((NV, NU, 3))
        n[:, NU // 2, 0] = rest * (1 + strains)
        world = SimpleNamespace(sheet_origin=np.zeros(3), pretension=0.08,
                                cutline_rest_m=rest,
                                call=lambda name: list(n.ravel()))
        gt = GroundTruth(world)
        gt.cut_length = lambda: 0.0
        measured = gt.cut_line_tension()
        scalar = float(np.median(rest[2:NV - 1] * (1 + strains[2:NV - 1])
                               / (H * (1 - 0.08)) - 1))
        self.assertAlmostEqual(measured, float(np.median(strains[2:NV - 1])))
        self.assertNotAlmostEqual(measured, scalar)

    def test_missing_rest_is_an_error_not_a_fallback(self):
        n = np.zeros((NV, NU, 3))
        n[:, NU // 2, 0] = np.full(NV, 0.001)
        # Current controller without a readback: refuse, never use the nominal grid.
        world = SimpleNamespace(sheet_origin=np.zeros(3), pretension=0.08,
                                cutline_rest_m=None,
                                call=lambda name: list(n.ravel()))
        gt = GroundTruth(world)
        gt.cut_length = lambda: 0.0
        with self.assertRaises(ValueError):
            gt.cut_line_tension()
        # CellRestTruth keeps its measured-only contract even on the frozen source.
        legacy = SimpleNamespace(sheet_origin=np.zeros(3), pretension=0.08,
                                 tissue_source=ROOT / 'src' / 'tissue' / 'legacy' / 'tissue_v1.lua',
                                 cutline_rest_m=None,
                                 call=lambda name: list(n.ravel()))
        truth = CellRestTruth(legacy)
        truth.cut_length = lambda: 0.0
        with self.assertRaises(ValueError):
            truth.cut_line_tension()

    def test_no_sheet_and_strict_boundary(self):
        rest = np.full(NV, 0.001 * 0.92)
        truth = _truth(rest, np.zeros(NV))
        truth.world.sheet_origin = None
        self.assertEqual(truth.cut_line_tension(), 0.0)
        truth2 = _truth(rest, np.zeros(NV))
        truth2.strict = True
        with self.assertRaises(TruthReadInControl):
            truth2.cut_line_tension()
        with truth2.purpose('probe'):
            self.assertEqual(truth2.cut_line_tension(), 0.0)
        self.assertEqual(truth2.reads_by_purpose['probe']['cut_line_tension'], 1)

    def test_invalid_rest_refused(self):
        for bad in (np.zeros(NV - 1),                                 # wrong shape
                    np.concatenate([[0.0], np.full(NV - 1, 0.001)]),  # nonpositive
                    np.full(NV, np.nan)):                             # non-finite
            truth = _truth(bad, np.zeros(NV))
            with self.assertRaises(ValueError, msg=str(bad)):
                truth.cut_line_tension()


class LuaRestReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = LuaRuntime()
        print(f'Lua safety runtime: {cls.runtime.description}', flush=True)
        cls.parent = '\n'.join(p.read_text(encoding='utf-8')
                               for p in (FIXTURE, *SOURCES, *PARENT))

    def test_real_readback_all_coupons(self):
        from seam_slide_world import staged_source
        from seam_wide_cell_world import wide_source
        from screen_cell_settle import asymmetric_source
        for name, fn, expected in (('narrow', staged_source, 0.001 * 0.92),
                                   ('symmetric', wide_source, 0.005 * 0.92),
                                   ('asymmetric', asymmetric_source, 0.003 * 0.92)):
            with self.subTest(name=name):
                src = fn(self.parent) + '\n' + READBACK
                self.runtime.run(src + '\nsysCall_init(); auto=false\n' + (
                    'tissueCreate(%s)\n'
                    'local r=cellTransverseRestLengths()\n'
                    'eq(#r,14)\n'
                    'for j=1,14 do eq(r[j],%.9g) end\n' % (SPEC, expected)))


if __name__ == '__main__':
    unittest.main()
