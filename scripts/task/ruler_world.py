"""Opt-in world for the ruler-print acquisition: appends src/tissue/ruler_print.lua to the staged controller."""
import hashlib
from pathlib import Path

from sim_world import LEGACY_TISSUE, SimWorld

ROOT = Path(__file__).resolve().parents[2]


class RulerWorld(SimWorld):
    # Locked collector: the historical hash chain pins the frozen legacy controller.
    TISSUE_SOURCE = LEGACY_TISSUE

    def __init__(self, *a, **kw):
        kw.setdefault('tissue_source', self.TISSUE_SOURCE)
        super().__init__(*a, **kw)

    def start(self):
        sim=self.sim
        if sim.getSimulationState()!=sim.simulation_stopped: raise RuntimeError('ruler requires stopped staged scene')
        base=sim.getStringProperty(self.ctrl,'code').replace('\r\n','\n')
        self.base_controller_sha256=hashlib.sha256(base.encode()).hexdigest()
        if self.base_controller_sha256 != 'a682b6ad7c99a6dde5d5fee62a28e99490ba384eadf2b9e95767905bd575d52f' or self.controller_sha256 != self.base_controller_sha256: raise RuntimeError('unexpected base controller')
        source=base+'\n'+(ROOT/'src/tissue/ruler_print.lua').read_text(encoding='utf-8')
        sim.setStringProperty(self.ctrl,'code',source)
        actual=sim.getStringProperty(self.ctrl,'code').replace('\r\n','\n')
        if actual!=source: raise RuntimeError('ruler readback differs')
        self.controller_sha256=hashlib.sha256(actual.encode()).hexdigest()
        super().start()
