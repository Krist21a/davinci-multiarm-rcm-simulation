"""Boundary and preserved vetoes for the opt-in ahead-window predicate."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vision'))
from cut_perception import AHEAD_WINDOW_M, _provable_ahead
from m5_guided_ahead_window_v1 import AheadWindowPerception, _window_provable_ahead


def evaluate(extras,s=0.,depth=.2,b_start=1):
    end = s+AHEAD_WINDOW_M+.0005
    edges = .0005*np.arange(-3,9)
    sa = np.r_[np.linspace(s+.00003,end-.00003,50),extras]
    band = np.ones(sa.shape,dtype=bool)
    owned = np.arange(len(sa))<50
    plane = np.full(sa.shape,.13)
    measured = np.where(owned,.13,depth)
    behind = ~owned & ((measured>plane+.0005)|~np.isfinite(measured))
    bins = np.digitize(sa[behind],edges)-1
    occ = np.bincount(bins,minlength=len(edges)-1)>=2
    args = (s,band,occ,b_start,edges,sa,owned,measured,plane,.5)
    return _provable_ahead(*args),_window_provable_ahead(*args)


class AheadWindowTests(unittest.TestCase):
    def test_exact_upper_boundary_and_wholly_outside_bins(self):
        self.assertEqual(evaluate((.0025,.0026)),(False,True))
        self.assertEqual(evaluate((.0026,.0027)),(False,True))
        self.assertEqual(evaluate((.0001,.0002),s=.00025),(False,True))

    def test_mixed_bin_with_real_or_missing_support_still_vetoes(self):
        for z in (.2,np.nan):
            self.assertEqual(evaluate((.00261,.00270),s=.00015,depth=z),(False,False))
        self.assertEqual(evaluate((.00254,.00280),s=.00015),(False,False))
        self.assertEqual(evaluate((0.,.00001)),(False,False))

    def test_original_entry_exclusion_stays(self):
        self.assertEqual(evaluate((.0001,.0002),b_start=4),(True,True))

    def test_other_material_guards_stay(self):
        s = 0.
        edges = np.array([0.,.001,.002,.003])
        sa = np.linspace(.0001,.0024,20)
        band = np.ones(20,dtype=bool)
        occ = np.zeros(3,dtype=bool)
        depth = np.full(20,.13)
        for mark,measured in ((band,depth-.01),(~band,depth),(band,np.where(sa<.001,depth,np.nan))):
            self.assertFalse(_window_provable_ahead(s,band,occ,0,edges,sa,mark,measured,depth,.5))

    def test_candidate_binds_both_call_sites_without_mutating_legacy(self):
        for name in ('_cut','_gap_scan'):
            method = getattr(AheadWindowPerception,name)
            native = getattr(AheadWindowPerception.__bases__[0],name)
            self.assertIs(method.__code__,native.__code__)
            self.assertIs(method.__globals__['_provable_ahead'],_window_provable_ahead)
            self.assertIs(native.__globals__['_provable_ahead'],_provable_ahead)


if __name__=='__main__':
    unittest.main()
