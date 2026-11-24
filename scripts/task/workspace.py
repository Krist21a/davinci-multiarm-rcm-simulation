"""Task workspace shared by the port planner (scripts/build/plan_psm_ports.py) and the CoppeliaSim world.

Everything is derived from the cutting instrument (PSM2) at its ready pose (config/endoscope/task.json):
  v       advance direction = PSM2 shaft direction at the wrist
  u       jaw axis of PSM2 made orthogonal to v (across the cut)
  roll    optional transverse basis rotation about v through the unchanged origin
  origin  near edge of the tissue sheet, 2 mm in front of the scissors tip, on the blade line
Sheet node (i, j) = origin + (i - (NU - 1) / 2) H u + j H v. The retraction grasp points sit on the two side
edges at mid length: PSM1 on the -u side, PSM3 on the +u side, so pulling them back opens the cut.
"""
import numpy as np

H = 0.001
NU, NV = 24, 14              # 23 mm wide: the grasp points sit 11.5 mm either side of the cut line
PRETENSION = 0.08
NEAR_EDGE_GAP_M = 0.002
MAX_ROLL_DEG = 16.0


def unit(v):
    return v / np.linalg.norm(v)


def layout(pivot, distal, jaw_axis, tcp, *, roll_deg=0.0):
    v = unit(distal)
    u = unit(jaw_axis - np.dot(jaw_axis, v) * v)
    if roll_deg:
        angle = np.deg2rad(roll_deg)
        u = np.cos(angle) * u + np.sin(angle) * np.cross(v, u)
    origin = tcp + NEAR_EDGE_GAP_M * v
    half = (NU - 1) / 2 * H
    mid = origin + 0.5 * (NV - 1) * H * v
    return dict(origin=origin, u=u, v=v, cut_start=origin,
                grasp_points={'psm1': mid - half * u, 'psm3': mid + half * u},
                normal=np.cross(u, v))
