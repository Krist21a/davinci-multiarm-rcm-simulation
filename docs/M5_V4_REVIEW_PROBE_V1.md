# M5-v4: first closed alternate-view experiment

The new [development task](../config/task/m5_v4_benchmark_v1.json) aims at 9 mm,
accepts an optically verified 9 or 10 mm, and keeps both full and partial damage
at or below 10 mm. Exact 10 remains a separate historical stress test. No signed
gate, historical outcome, physics, print, fixture or default runner changed.

## What was exercised

The [probe registration](../validation/task/m5_v4_review_probe_v1/registration.json)
pins the readers, fixture, runner, reference trajectories and evaluation rules.
The incremental observer kept only an optical ordinal. The last confirmed row,
every new row, the boundary and the intact next row required fresh evidence in
one closed image. The probe had no path for a fifth closure or final withdrawal.

The [offline gate](../validation/task/m5_v4_review_probe_v1/offline_gate.json)
passed: the old complete 9 mm example was retained; the missing new row in the
current primary image was not inherited; neither archived half-cut was promoted.
Projected row 6-10 banks fit in the candidate 20-degree image; this test of
framing made no claim about visibility. The source-bound [offline suite](../validation/safety/regressions_m5_v4_review_probe_v1_sealed.json)
passed 489 tests. The first pre-registration attempt rejected a mistyped fixture
digest before reading test images or creating a registration; the fixture and
all thresholds stayed unchanged.

## One fresh scene, one alternative

The first 18 optical files reproduced the reference trajectory byte for byte.
The only subsequent motion was the guarded ECM change from `[0,4.5,0.088]` to
`[-4.5,4.5,0.080]`. Both exposures were closed at identical cutter joints.
Scoring-only constraints were identical before and after: prefix/full/partial
9/9/9 mm, 41 newly dead edges from the uncut baseline. All 13 work exposures
retained independent scoring sidecars. The probe ended closed at the entrance,
not at ready; it was never a v4 positive run.

| Independent review | Row 6 overlap | New rows 7, 8 | Boundary 9, 10 | Verdict |
| --- | --- | --- | --- | --- |
| Primary, complete 9 mm | OPEN | UNKNOWN, OPEN | INTACT, INTACT | UNKNOWN |
| Alternate, same cut | OPEN | UNKNOWN, OPEN | INTACT, INTACT | UNKNOWN |
| Archived alternate, different 8 mm trajectory | OPEN | OPEN, PARTIAL | UNKNOWN, INTACT | UNKNOWN |

The archived alternate is a real partial cut, not a matched live negative of
the translated fixture. Its partial row is identified correctly, but the next
row lacks an intact certificate. Abstention is not positive discrimination.

Post-hoc application of the existing row-7 contact audit to the two complete-cut
frames gives 91/92 row-owned deep pixels, 10/8 raw left contacts, **0/0**
certified left contacts and **5/4** certified right contacts (primary/alternate).
The shortest four-neighbor distance to the certified left shore went from 23 to
26 pixels. The alternative changed the view, not the missing positive evidence.

[Independent predictions](../validation/task/m5_v4_review_probe_v1/live_predictions.json)
and [decision](../validation/task/m5_v4_review_probe_v1/decision.json): mechanical
acquisition passed, optical recognition and negative discrimination did not.
No second pose, acceptance trial, disturbance, or M6/M7 run is authorized.

## Next decision

Test bending regularization on a separately named coupon against the frozen
fixture, initially under matched loading **without cutting**. Hold the optical
poses fixed. Measure odd/even normal displacement and visibility as well as
tension, grasp and ready/RCM checks; a smoother render alone is not a pass.
Before using that coupon to cut, test that the regularizer does not couple
disconnected faces or silently alter the damage-label semantics. Do not reuse
the old sensor-margin qualification on a changed coupon without rechecking it.
