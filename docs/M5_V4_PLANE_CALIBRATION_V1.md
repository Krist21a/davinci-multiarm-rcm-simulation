# Rigid planar target: slope-dependent native depth discrepancy

The stopped-scene [calibration protocol](../config/task/m5_v4_plane_calibration_v1.json)
used a passive copy of the actual endoscope in a clear region, with the same
70-degree, 1280x960 RGB and metric-depth readout as the work frames. Each
checkerboard reference came from the actual mesh vertices and transforms read
back from the simulator. No robot commands, specimen or scene save. Nine poses
were used for development; six other poses were graded after sealing the fits.

No preregistered depth model passed. Worst per-pose p95 absolute error (mm):

| Forward depth model | Development | Held out |
| --- | ---: | ---: |
| Native ray | 0.15571 | 0.19570 |
| Fixed row/column shift | 0.17394 | 0.22047 |
| Range gain and intercept | 0.08043 | 0.11917 |
| Both | 0.08039 | 0.11927 |

The gate was 0.010 mm on **every** pose. On front-facing boards native errors
were below 0.001 mm. At 140 mm with +60 or -60 degree single-axis tilt the
median shifted **farther** by 0.129-0.131 mm for both signs and both axes.
This reproduces the magnitude and sign pattern without tissue, without a rim
estimator and away from color transitions. It does not identify the responsible
renderer setting. The independent RGB shift estimate was (-0.0067,-0.0019) px
(row,column). Four development poses failed the 98% edge bracket requirement
(minimum 96.74%); all six held poses passed it (minimum 98.49%). Neither a
depth correction nor an RGB-to-depth registration was qualified. In particular
the registered transfer to the tissue support pixels was **not** executed.

After sealing the [negative decision](../validation/task/m5_v4_plane_calibration_v1/decision.json),
a [development-only check](../validation/task/m5_v4_plane_calibration_v1/posthoc_dev_slope.json)
examined reciprocal depth `u=1/z`, linear in image coordinates for a plane.
On the six tilted development poses, `(u_geom-u_measured)/max(|du/dr|,|du/dc|)`
had per-pose medians 0.5003-0.5019 px. A half-maximum-slope offset predicted
measured z to within 0.0011 mm p95 on these six poses. This is a post-hoc
description, not evidence of a uniform half-pixel ray shift, a proven graphics
mechanism or a validated online inverse.

Next: freeze that forward prediction **before** fresh independent poses,
including diagonal gradients and reversed slopes; separately audit RGB edge
bracketing at grid intersections. If both gates pass, evaluate transfer to
the sealed tissue support. Scoring-plane normals cannot be supplied to a
controller. Even successfully correcting native depth would not establish
that outer-bank stations conservatively bound the central seam.
