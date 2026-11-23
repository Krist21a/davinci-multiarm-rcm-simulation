"""Opt-in 45-degree entry mount for the registered pilot (config/task/ruler_entry_mount_v1.json)."""
import numpy as np
from ruler_world import RulerWorld
from sim_world import GRASP_RADIUS_M, workspace
from workspace import NU, NV, H


class EntryMountWorld(RulerWorld):
    def place_sheet(self):
        if self.roi_proxy is not None or self.mode != 'full':
            raise ValueError('entry fixture requires verified full-mode flat sheet')
        f = self.frame(self.cutter)
        layout = workspace.layout(f['pivot'], f['x'], f['axis'], f['tcp'])
        self.u, old_v = layout['u'], layout['v']
        n = np.cross(self.u, old_v)
        self.v = (old_v - n) / np.sqrt(2.)
        n2 = np.cross(self.u, self.v)
        a, b, c = self.args.pivot_local
        pivot = layout['origin'] + a * self.u + b * old_v + c * n
        self.sheet_origin = pivot - a * self.u - b * self.v - c * n2
        mid = self.sheet_origin + .5 * (NV - 1) * H * self.v
        self.grasp_points = {self.primary: mid - .5 * (NU - 1) * H * self.u,
                             self.assist: mid + .5 * (NU - 1) * H * self.u}
        self.call('tissueCreate', dict(origin=self.sheet_origin.tolist(), u=self.u.tolist(),
                                       v=self.v.tolist(), nu=NU, nv=NV, spacing=H,
                                       pretension=self.pretension, anchor_sides=False,
                                       anchor_from_row=3, grasp_radius=GRASP_RADIUS_M))
        self.steps(30)
