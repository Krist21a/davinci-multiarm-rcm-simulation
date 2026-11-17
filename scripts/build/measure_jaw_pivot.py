"""Locate the jaw pivot of a dvrk_model tip mesh: the pin hole that both jaws share.

Each jaw mesh is sliced perpendicular to the jaw axis (z of the jaw link frame). Interior loops of the
sections are hole candidates; a pivot is a hole whose centre is the same in both jaws (a hole that is
not shared moves apart as the jaws open and cannot hold the pin). Prints the shared holes in the dVRK
jaw link frame. Needs trimesh + shapely.

  python measure_jaw_pivot.py 420179
"""
import argparse

import numpy as np
import trimesh

import instruments as inst


def holes(V, F, slices=25, rmin=0.3e-3, rmax=1.5e-3):
    m = trimesh.Trimesh(V, F, process=True)
    out = []
    for z in np.linspace(V[:, 2].min() + 1e-4, V[:, 2].max() - 1e-4, slices):
        s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if s is None:
            continue
        p, _ = s.to_2D(to_2D=np.eye(4))
        try:
            polys = p.polygons_full
        except ValueError:          # a self-touching section; skip the slice
            continue
        for poly in polys:
            for ring in poly.interiors:
                c = np.array(ring.coords)
                r = float(np.linalg.norm(c - c.mean(0), axis=1).mean())
                if rmin < r < rmax:
                    out.append(np.r_[c.mean(0), z, r])
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model')
    p.add_argument('--match-m', type=float, default=3e-4, help='max centre distance for a shared hole')
    args = p.parse_args()
    d = inst.descriptor(args.model)
    tj, tv = inst.parse_stage(inst.XACRO / 'tip' / f"tip_{d['stages']['tip']}.urdf.xacro")
    found = {}
    for name, j in tj.items():
        if j['mimic'] and j['mimic']['joint'] == 'jaw' and j['child'] in tv:
            found[j['child']] = holes(*inst.merge([inst.mesh_in_link(v) for v in tv[j['child']]]))
    (a, ha), (b, hb) = found.items()
    shared = []
    for h in ha:
        dist = np.linalg.norm(hb[:, :2] - h[:2], axis=1) if len(hb) else np.array([np.inf])
        if dist.min() <= args.match_m:
            shared.append(np.r_[h[:2], hb[dist.argmin(), :2], h[3]])
    shared = np.array(shared)
    print(args.model, a, len(ha), 'holes;', b, len(hb), 'holes')
    if not len(shared):
        print('no shared hole')
        return
    for s in shared:
        print(f'  shared hole {a} ({s[0]*1e3:.3f}, {s[1]*1e3:.3f}) mm  {b} ({s[2]*1e3:.3f}, {s[3]*1e3:.3f}) mm  r {s[4]*1e3:.2f} mm')
    c = np.r_[shared[:, [0, 2]].mean(), shared[:, [1, 3]].mean()]
    print(f'pivot (x, y) in the jaw link frame: ({c[0]*1e3:.3f}, {c[1]*1e3:.3f}) mm')


if __name__ == '__main__':
    main()
