"""URDF serial-chain kinematics and scope geometry used by the ECM docking planner."""
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from simtools import MESHES, URDF


def axis_angle(k, angle):
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def unit(v):
    return v / np.linalg.norm(v)


def rodrigues(a, b):
    """Minimal rotation matrix taking unit vector a onto unit vector b."""
    k = np.cross(a, b)
    s, c = np.linalg.norm(k), float(np.dot(a, b))
    if s < 1e-15:
        return np.eye(3)
    k = k / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


class Chain:
    """Serial URDF chain from the world to one link."""

    def __init__(self, tip_link):
        root = ET.parse(URDF).getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, link = [], tip_link
        while link in by_child:
            j = by_child[link]
            o = j.find('origin')
            T = np.eye(4)
            if o is not None:
                T[:3, 3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
                T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = j.find('axis')
            chain.insert(0, dict(name=j.get('name'), kind=j.get('type'), T=T,
                                 axis=np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else None))
            link = j.find('parent').get('link')
        self.joints = chain
        self.names = [j['name'] for j in chain]

    def fk(self, q, axes=None):
        """Link transform; optionally collects world (point, axis, kind) of the named joints (before their motion)."""
        T = np.eye(4)
        for j in self.joints:
            T = T @ j['T']
            v = q.get(j['name'], 0.0)
            if axes is not None and j['name'] in axes:
                axes[j['name']] = (T[:3, 3].copy(), T[:3, :3] @ j['axis'], j['kind'])
            if j['kind'] == 'revolute':
                D = np.eye(4)
                D[:3, :3] = axis_angle(j['axis'], v)
                T = T @ D
            elif j['kind'] == 'prismatic':
                D = np.eye(4)
                D[:3, 3] = j['axis'] * v
                T = T @ D
        return T


def link_vertices(link):
    """Visual mesh vertices of a URDF link, expressed in that link frame (visual origin applied)."""
    root = ET.parse(URDF).getroot()
    node = next(l for l in root.findall('link') if l.get('name') == link)
    dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    out = []
    for vis in node.findall('visual'):
        mesh = vis.find('.//mesh')
        v = np.fromfile(MESHES.parent / mesh.get('filename'), dtype=dt, offset=84)['v'].reshape(-1, 3).astype(float)
        v = v * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        o = vis.find('origin')
        if o is not None:
            R = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            v = v @ R.T + np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
        out.append(v)
    return np.concatenate(out)


class PoseTask:
    """Least-squares IK of one link point and a set of link-fixed unit vectors.

    `point_local` is a point in the link frame; `vec_local` are link-frame unit vectors whose
    world directions are constrained (one vector: point + axis, two vectors: full pose).
    """

    def __init__(self, chain, variables, fixed, limits, point_local, vec_local):
        self.chain, self.vars, self.fixed = chain, list(variables), dict(fixed)
        self.lo = np.array([limits[n][0] for n in self.vars])
        self.hi = np.array([limits[n][1] for n in self.vars])
        self.p = np.append(point_local, 1.0)
        self.vecs = [np.asarray(v, float) for v in vec_local]

    def q(self, x):
        q = dict(self.fixed)
        q.update(zip(self.vars, x))
        return q

    def evaluate(self, x):
        T = self.chain.fk(self.q(x))
        return (T @ self.p)[:3], [T[:3, :3] @ v for v in self.vecs]

    def solve(self, seed, point_des, vec_des, max_nfev=200):
        def res(x):
            p, vs = self.evaluate(x)
            return np.concatenate([(p - point_des) * 10] + [v - d for v, d in zip(vs, vec_des)])

        def jac(x):
            q = self.q(x)
            axes = {n: None for n in self.vars}
            T = self.chain.fk(q, axes)
            p = (T @ self.p)[:3]
            vs = [T[:3, :3] @ v for v in self.vecs]
            cols = []
            for n in self.vars:
                pj, kj, kind = axes[n]
                if kind == 'prismatic':
                    cols.append(np.r_[kj * 10, np.zeros(3 * len(vs))])
                else:
                    cols.append(np.r_[np.cross(kj, p - pj) * 10, np.concatenate([np.cross(kj, v) for v in vs])])
            return np.array(cols).T
        seed = np.clip(seed, self.lo + 1e-5, self.hi - 1e-5)
        sol = least_squares(res, seed, jac=jac, bounds=(self.lo + 1e-6, self.hi - 1e-6),
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
        r = res(sol.x)
        return sol.x, float(np.linalg.norm(r))

    def margin(self, x, skip=0):
        """Smallest normalised distance to a joint limit, ignoring the first `skip` variables."""
        x, lo, hi = x[skip:], self.lo[skip:], self.hi[skip:]
        return float(np.min(np.minimum(x - lo, hi - x) / (hi - lo)))


def scope_geometry(port, target, inward, theta, distance, azimuth):
    """Shaft axis a, scope tip and view direction v for an oblique scope through the port.

    The target lies on the optical axis at `distance` from the tip; the shaft is tilted away
    from the port-target line toward the body normal, rotated by `azimuth` about that line.
    """
    w = target - port
    L = np.linalg.norm(w)
    wh = w / L
    if distance >= L:
        return None
    s = -distance * math.cos(theta) + math.sqrt(L * L - (distance * math.sin(theta)) ** 2)
    alpha = math.asin(distance * math.sin(theta) / L)
    e1 = unit(inward - np.dot(inward, wh) * wh)
    e = Rotation.from_rotvec(wh * math.radians(azimuth)).apply(e1)
    a = math.cos(alpha) * wh + math.sin(alpha) * e
    tip = port + s * a
    return dict(axis=a, tip=tip, view=unit(target - tip), depth=s, alpha_deg=math.degrees(alpha))


def frames(axis, view, up_hint):
    """Tip frame (z = shaft axis) and camera frame (z = optical axis, y = image up), as 3x3 columns."""
    u = unit(up_hint - np.dot(up_hint, view) * view)
    cam = np.column_stack((unit(np.cross(u, view)), u, view))
    y = unit(u - np.dot(u, axis) * axis)
    return np.column_stack((np.cross(y, axis), y, axis)), cam


def xy_azimuth(v):
    return math.degrees(math.atan2(v[1], v[0]))


def angle_gap(a, b):
    return abs((a - b + 180) % 360 - 180)
