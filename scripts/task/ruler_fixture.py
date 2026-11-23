"""Opt-in angled mounting fixture for the registered pilot (config/task/ruler_fixture_pilot_v1.json)."""
import numpy as np
from ruler_world import RulerWorld
from sim_world import GRASP_RADIUS_M, workspace
from workspace import NU, NV, H


class FixtureWorld(RulerWorld):
    def place_sheet(self):
        if self.roi_proxy is not None or self.mode != 'full' or self.args.angle not in (45, 60):
            raise ValueError('unsupported ruler fixture')
        f = self.frame(self.cutter)
        layout = workspace.layout(f['pivot'], f['x'], f['axis'], f['tcp'])
        self.u, old_v = layout['u'], layout['v']
        angle = np.radians(self.args.angle)
        self.v = np.cos(angle)*old_v - np.sin(angle)*np.cross(self.u,old_v)
        mid = layout['origin'] + .5*(NV-1)*H*old_v
        self.sheet_origin = mid - .5*(NV-1)*H*self.v
        self.grasp_points = layout['grasp_points']
        self.call('tissueCreate', dict(origin=self.sheet_origin.tolist(),u=self.u.tolist(),v=self.v.tolist(),nu=NU,nv=NV,spacing=H,pretension=self.pretension,anchor_sides=False,anchor_from_row=3,grasp_radius=GRASP_RADIUS_M))
        self.steps(30)
