# Rigid blind test and hinge work-distance transfer

The [frozen plan](../config/task/m5_v4_plane_blind_v2.json) registered twelve new
poses (two frontal controls and five opposite-tilt pairs, including diagonal
gradients). There was no fit or development split. The copied work sensor and
renderer were unchanged. No robot moved, and the dedicated instance was closed.

## Rigid test

- All 12 poses passed the 0.010 mm per-pose p95 requirement. Across 5,362
  samples the worst pose p95 changed from 0.20604 mm against the geometric
  depth to 0.00122 mm against the frozen forward prediction. This tests how
  the sensor reports depth on a known plane, not an online depth inverse.
- Both observed endpoint colors agreed with the finite projected checker cells
  on 22,182/22,182 observed-or-predicted transition pairs (0.02 px tolerance).
  The closest-infinite-line comparator rejected 174 of the 22,173 observed
  changes; all 174 were within 1.5 px of a projected junction and matched the
  cells' endpoint colors. No junctions or previously counted pairs were dropped.
- See the [per-pose blind decision](../validation/task/m5_v4_plane_blind_v2/blind_decision.json).
  The historical v1 decision remains a failure under its own registered rule.

## Archive transfer

Only after sealing both rigid gates did the runner evaluate the six distinct
hinge work frames. Using first-hit normals **only for offline scoring**, the
frozen forward law accounted for the reported depth of the 1,102 support
observations: worst frame/bank p95 residual 0.00121 mm (native geometric
comparison: 0.17911 mm).

The separate optical calculation used no surface truth: it estimated the local
reciprocal-depth slope on each bank from the same noisy RGB-D, corrected both
station depths on their measured rays, and took the nearer corrected station.
The reference v11 axis, marker and abstentions were replayed before the new
predictions were sealed and subsequently compared with the exact seam and TCP.

| Distinct late samples | Valid | Max optimism before / after (mm) | Limit (mm) |
| --- | ---: | ---: | ---: |
| Native | 6/6 | +0.22540 / +0.06904 | 0.0000 |
| 0.5 mm, development noise | 70/72 | +0.37079 / +0.21419 | 0.1000 |
| 0.5 mm, check noise | 70/72 | +0.33779 / +0.17956 | 0.1000 |

All six native samples remained optimistic. This rejects the work-room
candidate, not the rigid forward prediction. There is no implication that
all remaining noisy error is material deformation: station ray, marker and
noise error remain in the calculation. The independent true outer-node vs
seam counterexample in the earlier rim audit still applies.

**Decision:** close this calibration branch. Do not fit a new bias, change a
threshold, or spend an independent nominal run. Next, screen whether the actual
row-10 central material boundary can be located or conservatively bounded from
one work exposure together with the tool marker, using independent node/TCP
references only for scoring. Missing central evidence must remain an abstention.
No live motion or change to the v4 completion gate is authorized.

[Transfer decision](../validation/task/m5_v4_plane_blind_v2/decision.json) |
[Optical replay scores](../validation/task/m5_v4_plane_blind_v2/transfer_measurements.json) |
[Offline regressions](../validation/safety/regressions_m5_v4_plane_blind_v2.json)
