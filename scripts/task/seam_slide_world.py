"""Opt-in folded-foreground slide-guide seam world (config/task/seam_slide_v1.json)."""
import hashlib
from pathlib import Path

from sim_world import SimWorld
from surface_seam_world import SurfaceSeamWorld
from workspace import NU

ROOT = Path(__file__).resolve().parents[2]

_STEP_DECL = 'local function tissueStep()'
_STEP_TAIL = '        end\n    end\n    local m=0\n    for _,s in ipairs(T.springs) do'
_STEP_TAIL_SLIDE = ('        end\n        projectSlide()\n    end'
                    '\n    local m=0\n    for _,s in ipairs(T.springs) do')


def staged_source(parent):
    """Inject the slide projection into the verified seam parent controller source.

    Each anchor must occur exactly once; raises instead of guessing if the parent
    layout ever drifts from the registered seam chain.
    """
    if parent.count(_STEP_DECL) != 1:
        raise RuntimeError('expected exactly one tissueStep in seam parent source')
    if parent.count(_STEP_TAIL) != 1:
        raise RuntimeError('expected exactly one solve-loop tail in seam parent source')
    slide = (ROOT / 'src/tissue/seam_slide.lua').read_text(encoding='utf-8')
    create = (ROOT / 'src/tissue/seam_slide_create.lua').read_text(encoding='utf-8')
    head, tail = parent.split(_STEP_DECL)
    parent = head + slide + '\n' + _STEP_DECL + tail
    return parent.replace(_STEP_TAIL, _STEP_TAIL_SLIDE) + '\n' + create


class SeamSlideWorld(SurfaceSeamWorld):
    def __init__(self, client, sim, args, cfg, *, mode='full'):
        if mode != 'full' or cfg['cut']['control_source'] != 'truth':
            raise ValueError('seam slide requires full-mode truth acquisition')
        super().__init__(client, sim, args, cfg, mode=mode)

    def start(self):
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError('seam slide requires stopped staged scene')
        base = sim.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n')
        self.base_controller_sha256 = hashlib.sha256(base.encode()).hexdigest()
        if (self.base_controller_sha256 != 'a682b6ad7c99a6dde5d5fee62a28e99490ba384eadf2b9e95767905bd575d52f'
                or self.controller_sha256 != self.base_controller_sha256):
            raise RuntimeError('unexpected base controller')
        parent = (base + '\n' + (ROOT / 'src/tissue/ruler_print.lua').read_text(encoding='utf-8')
                  + '\n' + (ROOT / 'src/tissue/surface_seam.lua').read_text(encoding='utf-8'))
        self.seam_controller_sha256 = hashlib.sha256(parent.encode()).hexdigest()
        if self.seam_controller_sha256 != '321045737747e24151435dfea9c0bbfe6ebd754469e612c3d9ca7378a24ef257':
            raise RuntimeError('unexpected seam parent controller')
        source = staged_source(parent)
        sim.setStringProperty(self.ctrl, 'code', source)
        actual = sim.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n')
        if actual != source:
            raise RuntimeError('seam slide readback differs')
        self.controller_sha256 = hashlib.sha256(actual.encode()).hexdigest()
        SimWorld.start(self)

    def place_sheet(self):
        super().place_sheet()
        expected = [j * NU + i for j in range(13)
                    for i in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
        if self.call('seamRailPins') != expected:
            raise RuntimeError('seam slide guide IDs differ from registration')
