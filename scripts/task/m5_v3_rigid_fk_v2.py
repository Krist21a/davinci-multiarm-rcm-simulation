"""Joint-space distal FK with calibrated passive shaft-twist coupling."""
from dataclasses import dataclass

import numpy as np

from m5_v3_rigid_fk import rot, shaft_basis


@dataclass(frozen=True)
class CutterFK:
    rcm: np.ndarray
    basis: np.ndarray
    insertion_intercept: float
    twist_rad: np.ndarray
    pitch_to_yaw: np.ndarray
    tip_R: np.ndarray
    tip_Y: np.ndarray
    hinge_Y: np.ndarray

    def pose(self, q):
        q = np.asarray(q,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all():
            raise ValueError('invalid joint readback')
        s = shaft_basis(self.basis,q)
        twist = float(self.twist_rad @ np.r_[1.,q[:2]])
        P = s @ rot(0,twist+q[3]) @ rot(2,q[4])
        Y = P @ self.pitch_to_yaw[:3,:3] @ rot(2,q[5])
        pivot = self.rcm+(self.insertion_intercept+q[2])*s[:,0]
        yaw = pivot+P@self.pitch_to_yaw[:3,3]
        return np.column_stack((Y@self.tip_R,yaw+Y@self.tip_Y))

    def blade_local(self):
        p = self.tip_R.T@(self.hinge_Y-self.tip_Y)
        axis = self.tip_R.T@np.array([0.,0.,1.])
        return ((p,axis),(p,-axis))
