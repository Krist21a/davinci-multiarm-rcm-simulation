"""A shorter diagnostic bite cannot replace either mechanical or optical evidence."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'/d) for d in ('analysis','task','vision')]
from run_m5_v4_bend_terminal_v1 import CollectionWorld
from score_m5_v4_bend_terminal_v1 import margin_gate


class BendTerminalTests(unittest.TestCase):
    def test_fixed_collection_protection_even_when_shorter_cap_active(self):
        g = SimpleNamespace(limit=np.array([0.,0.,.003]),direction=np.array([0.,0.,1.]),prefix_mm=7)
        p = SimpleNamespace(tick=17,chart=g,tip=np.zeros(3),hinge=np.array([1.,0.,0.]),jaw=g.direction)
        io = SimpleNamespace(report={'moves':[]},closed=lambda:True)
        w = SimpleNamespace(current=lambda _:g,model=SimpleNamespace(blade_local=lambda:[(np.zeros(3),p.hinge)]),
                            io=io,cap=.0015)
        self.assertAlmostEqual(CollectionWorld.sweep_room(w,p)-.00015,.0015)
        g.limit[2] = .0005
        self.assertLess(CollectionWorld.sweep_room(w,p),0)  # no positive allowance from the cap
        io.closed = lambda:False
        self.assertAlmostEqual(CollectionWorld.sweep_room(w,p),-.00005)

    def test_cannot_qualify_an_empty_or_biased_late_population(self):
        self.assertFalse(margin_gate([],True)['qualified'])
        rows = [dict(prefix=7,row=10,optimism_mm=0.,split=k) for k in ('clean','development','check')]
        self.assertFalse(margin_gate(rows,False)['qualified'])
        rows[-1]['optimism_mm'] = .101
        self.assertFalse(margin_gate(rows,True)['qualified'])
