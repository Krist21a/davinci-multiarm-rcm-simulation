"""Passive indexed-print view inside CutPerception: a stateless observer only.

Reports the local material reference plus a guarded tool read; it never certifies
a cut front, progress or tension, and holds no state between calls.
"""
import numpy as np
from cut_perception import CutPerception
from indexed_reference import measure


class IndexedCutPerception(CutPerception):
    """Stateless observer only: no remembered front, no certified progress."""

    def estimate(self, rgb, depth, R_est, t_est, post_grasp=False):
        im,z=np.asarray(rgb),np.asarray(depth)
        reference=measure(im,z,self.W,self.H,self.f,self.far)
        tool=dict(valid=False,reason='invalid RGB-D')
        if im.dtype==np.uint8 and im.shape==(self.H,self.W,3) and z.shape==(self.H,self.W) and z.dtype.kind=='f':
            tool,_=self._scissors(im.astype(float),z)
        R,t=np.asarray(R_est),np.asarray(t_est)
        tool=dict(tool)
        for key in ('tip','axis','jaw_axis','opening_axis'):
            if tool.get(key) is not None:
                v=R @ np.asarray(tool[key]); tool[key]=(v+t if key=='tip' else v).tolist()
        for station in reference['stations']:
            station['world_m']=(R @ np.asarray(station['camera_m'])+t).tolist()
        return dict(material_reference=reference,scissors=tool,
                    tissue=dict(valid=bool(reference['stations'])),
                    cut=dict(valid=False,front_valid=False,front=None,progress_m=None,
                             reason='local material reference does not certify a 3D cut front or sweep'),
                    tension_proxy=dict(valid=False,reason='not evaluated'))
