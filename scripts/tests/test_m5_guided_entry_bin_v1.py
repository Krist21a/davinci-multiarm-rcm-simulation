"""Controls for the isolated deterministic working-view entry guard."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vision'))
from cut_perception import CutPerception, _provable_ahead
from m5_guided_entry_bin_v1 import EntryBinPerception

ORIGINS = (-.005463624403497512,-.0054633044991712005)


def scan(reader,near,columns=(),value=.2):
    est = object.__new__(reader)
    est.far = .5
    sa = np.tile(near-.001+(np.arange(26)+.5)*.0005,(4,1))
    sb = np.zeros_like(sa)
    plane = np.full_like(sa,.13)
    depth = plane.copy()
    mask = np.ones_like(sa,dtype=bool)
    depth[:,columns] = value
    mask[:,columns] = False
    r = est._gap_scan(mask,sa,sb,plane,depth,0.,near,near+.012)
    valid = _provable_ahead(near,r['band'],r['occ'],r['b_start'],r['edges'],sa,mask,depth,plane,est.far)
    return r,valid


class EntryBinTests(unittest.TestCase):
    def test_border_bin_stays_excluded_at_both_origins(self):
        self.assertEqual([scan(CutPerception,n,(3,))[0]['b_start'] for n in ORIGINS],[4,3])
        for n in (*ORIGINS,np.nextafter(ORIGINS[1],-np.inf),np.nextafter(ORIGINS[1],np.inf)):
            r,valid = scan(EntryBinPerception,n,(3,))
            self.assertEqual(r['b_start'],4)
            self.assertTrue(r['occ'][3])
            self.assertIsNone(r['run_end'])
            self.assertTrue(valid)

    def test_first_post_guard_bin_still_vetoes_deep_or_missing_depth(self):
        for z in (.2,np.nan):
            for n in ORIGINS:
                r,valid = scan(EntryBinPerception,n,(4,),z)
                self.assertEqual(r['b_start'],4)
                self.assertTrue(r['occ'][4])
                self.assertEqual(r['slit_sel'].sum(),4)
                self.assertFalse(valid)

    def test_foreground_and_missing_sheet_keep_their_veto(self):
        for z in (.10,.13):
            r,valid = scan(EntryBinPerception,ORIGINS[1],(2,3,4),z)
            self.assertFalse(r['occ'].any())
            self.assertFalse(valid)

    def test_two_bin_true_slit_matches_unambiguous_legacy(self):
        old,old_valid = scan(CutPerception,ORIGINS[0],(4,5))
        new,new_valid = scan(EntryBinPerception,ORIGINS[0],(4,5))
        self.assertAlmostEqual(new['run_len'],.001)
        self.assertFalse(old_valid)
        self.assertFalse(new_valid)
        self.assertEqual(old.keys(),new.keys())
        for k in old:
            np.testing.assert_equal(old[k],new[k])


if __name__=='__main__':
    unittest.main()
