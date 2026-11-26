"""Compose the experimental coupon in the stopped simulation's staging copy only."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HINGE = ROOT/'src/tissue/m5_v4_transverse_hinge_v1.lua'


def stage(source):
    for anchor,addition in [('local function tissueStep()',HINGE.read_text(encoding='ascii')),
                            ('        if guideKeys then','        bendTransverse()')]:
        if source.count(anchor)!=1:
            raise ValueError('unexpected canonical solver: '+anchor)
        source = source.replace(anchor,addition+'\n'+anchor)
    return source
