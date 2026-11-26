"""psm2 joint-space distal FK and rigid blade calibration; no simulator reads."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def rot(axis, angle):
    v = np.zeros(3)
    v[axis] = angle
    return Rotation.from_rotvec(v).as_matrix()


def shaft_basis(basis, q):
    home, x, _ = basis
    d = home + np.tan(np.deg2rad(q[:2])) @ basis[1:]
    d /= np.linalg.norm(d)
    v = np.cross(home,d)
    s = np.linalg.norm(v)
    axis = Rotation.from_rotvec(v*(np.arctan2(s,home@d)/s)).apply(x) if s>1e-12 else x
    return np.column_stack((d,axis,np.cross(d,axis)))


@dataclass(frozen=True)
class CutterFK:
    rcm: np.ndarray
    basis: np.ndarray
    insertion_intercept: float
    neutral_twist: float
    pitch_to_yaw: np.ndarray
    tip_R: np.ndarray
    tip_Y: np.ndarray
    hinge_Y: np.ndarray

    def pose(self, q):
        q = np.asarray(q,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all():
            raise ValueError('invalid joint readback')
        s = shaft_basis(self.basis,q)
        P = s @ rot(0,self.neutral_twist+q[3]) @ rot(2,q[4])
        Y = P @ self.pitch_to_yaw[:3,:3] @ rot(2,q[5])
        pivot = self.rcm+(self.insertion_intercept+q[2])*s[:,0]
        yaw = pivot+P@self.pitch_to_yaw[:3,3]
        return np.column_stack((Y@self.tip_R,yaw+Y@self.tip_Y))

    def blade_local(self):
        p = self.tip_R.T@(self.hinge_Y-self.tip_Y)
        axis = self.tip_R.T@np.array([0.,0.,1.])
        return ((p,axis),(p,-axis))
