"""Incremental reviews cannot bridge a missing new row or recycle stale receipts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/s) for s in ('analysis','vision','task')]
from replay_m5_v3_corridor_v1 import coded_sheet
from m5_v4_frontier_v1 import Reviews, measure


class FrontierTests(unittest.TestCase):
    def test_only_old_rows_may_be_hidden(self):
        rgb,z = coded_sheet(9)
        z[25:27,24:27] = .13
        self.assertEqual(measure(rgb,z,far=.3,sigma=0.,prefix=7)['prefix_mm'],9)
        for row in (6,7,8):
            bad = z.copy()
            bad[row*10+3:row*10+6,24:27] = .13
            self.assertEqual(measure(rgb,bad,far=.3,sigma=0.,prefix=7)['verdict'],'UNKNOWN')

    def test_terminal_negative_and_contradiction(self):
        rgb,z = coded_sheet(9)
        self.assertIsNone(measure(*coded_sheet(6),far=.3,sigma=0.,prefix=7)['prefix_mm'])
        rgb[90:100,24:40],z[90:100,24:40] = 0,.23
        self.assertIsNone(measure(rgb,z,far=.3,sigma=0.,prefix=7)['prefix_mm'])

    def test_binding_and_alternate_budget(self):
        ledger = Reviews('one')
        def look(t,c,alt=False,identity='one'):
            return ledger.observe(identity,t,c,True,None,None,far=.3,alternate=alt)
        with patch('m5_v4_frontier_v1.intact_preflight',return_value={'passed':True}):
            look(2,0)
        for args in [(2,1),(6,2),(6,1,False,'other'),(6,1,True)]:
            with self.assertRaises(ValueError):
                look(*args)
        with patch('m5_v4_frontier_v1.measure',side_effect=[
                dict(verdict='UNKNOWN',prefix_mm=None),dict(verdict='BELOW',prefix_mm=3),
                dict(verdict='UNKNOWN',prefix_mm=None)]):
            look(6,1)
            with self.assertRaises(ValueError):
                look(7,2)
            look(7,1,True)
            look(10,2)
        with self.assertRaises(ValueError):
            look(11,2,True)
