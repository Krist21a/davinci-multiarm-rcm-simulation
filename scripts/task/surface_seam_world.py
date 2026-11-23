"""Opt-in surface-seam coupon world for the registered pilot (config/task/surface_seam_pilot_v1.json)."""
import hashlib
from pathlib import Path

from seam_topology import SeamTruth, material_state
from sim_world import LEGACY_TISSUE, SimWorld
from workspace import H

ROOT = Path(__file__).resolve().parents[2]

__all__ = ['SeamTruth', 'SurfaceSeamWorld']


class SurfaceSeamWorld(SimWorld):
    # Locked collector: the historical hash chain pins the frozen legacy controller.
    TISSUE_SOURCE = LEGACY_TISSUE

    def __init__(self, *a, **kw):
        kw.setdefault('tissue_source', self.TISSUE_SOURCE)
        super().__init__(*a, **kw)
        if self.vision:
            raise ValueError('surface seam is development truth acquisition only')
        self.truth = SeamTruth(self)

    def start(self):
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError('surface seam requires stopped staged scene')
        base = sim.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n')
        self.base_controller_sha256 = hashlib.sha256(base.encode()).hexdigest()
        if (self.base_controller_sha256 != 'a682b6ad7c99a6dde5d5fee62a28e99490ba384eadf2b9e95767905bd575d52f'
                or self.controller_sha256 != self.base_controller_sha256):
            raise RuntimeError('unexpected base controller')
        printed = base + '\n' + (ROOT / 'src/tissue/ruler_print.lua').read_text(encoding='utf-8')
        self.printed_controller_sha256 = hashlib.sha256(printed.encode()).hexdigest()
        if self.printed_controller_sha256 != 'b0cd9fbf7b5ba3733f1f60b3a149be1a8004665a054f84dbf958adeeaad5a868':
            raise RuntimeError('unexpected printed controller')
        source = printed + '\n' + (ROOT / 'src/tissue/surface_seam.lua').read_text(encoding='utf-8')
        sim.setStringProperty(self.ctrl, 'code', source)
        actual = sim.getStringProperty(self.ctrl, 'code').replace('\r\n', '\n')
        if actual != source:
            raise RuntimeError('seam readback differs')
        self.controller_sha256 = hashlib.sha256(actual.encode()).hexdigest()
        super().start()

    def observe(self, *, inject_events=True):
        obs = super().observe(inject_events=inject_events)
        if self.sheet_origin is not None:
            with self.truth.purpose('safety_monitor'):
                end = material_state(self.call('tissueConstraints'))['farthest_row_end'] * H
            limit = self.cfg['cut']['length_m'] + self.cfg['cut']['max_overcut_m']
            if end > limit + 1e-12:
                obs.faults.append(f'seam conservative row end {end * 1000:.2f} mm exceeds '
                                  f'{limit * 1000:.2f} mm')
        return obs
