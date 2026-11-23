"""Opt-in paired support-rail seam world for the registered fixture (config/task/seam_rail_v1.json)."""
import hashlib
from pathlib import Path

from sim_world import SimWorld
from surface_seam_world import SurfaceSeamWorld
from workspace import NU

ROOT = Path(__file__).resolve().parents[2]


class SeamRailWorld(SurfaceSeamWorld):
    def __init__(self, client, sim, args, cfg, *, mode='full'):
        if (mode != 'full' or cfg['cut']['control_source'] != 'truth'
                or getattr(args, 'rail', None) not in ('outer', 'inner')):
            raise ValueError('seam rail requires full-mode truth control with --rail outer|inner')
        self.rail_left = {'outer': 8, 'inner': 9}[args.rail]
        super().__init__(client, sim, args, cfg, mode=mode)

    def start(self):
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError('seam rail requires stopped staged scene')
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
        source = (parent + f'\nlocal railLeft={self.rail_left}\n'
                  + (ROOT / 'src/tissue/seam_rail.lua').read_text(encoding='utf-8'))
        sim.setStringProperty(self.ctrl, 'code', source)
        actual = sim.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n')
        if actual != source:
            raise RuntimeError('seam rail readback differs')
        self.controller_sha256 = hashlib.sha256(actual.encode()).hexdigest()
        SimWorld.start(self)

    def place_sheet(self):
        super().place_sheet()
        expected = [j * NU + i for j in range(6, 13) for i in (self.rail_left, 23 - self.rail_left)]
        if self.call('seamRailPins') != expected:
            raise RuntimeError('seam rail pins differ from registration')
