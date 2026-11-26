# M5-v4 hinge rail: five registered closed-loop cases pass

[Protocol](../config/task/m5_v4_hinge_closed_loop_v3.json) |
[Offline qualification](../validation/task/m5_v4_hinge_closed_loop_v3/offline_gate.json) |
[Fresh-scene matrix](../validation/task/m5_v4_hinge_closed_loop_v3/matrix.json) |
[Final verification](../validation/task/m5_v4_hinge_closed_loop_v3/completion.json)

One frozen controller, one new simulator scene per case, no retries. All five
produced four optical closures, a fresh primary `TARGET(9)`, and a closed
withdrawal to the original ready joints. Independent scoring found prefix,
farthest complete and farthest partial damage all equal to 9 mm, no intact
terminal face, and no edge-table changes during final withdrawal. No tear,
RCM failure, collision block or rejected cutter move. No alternate review
pose was needed. The absolute 10 mm limit was unchanged.

| Case | Evidence specific to this run | Terminal tick |
| --- | --- | ---: |
| nominal | BELOW(3,5,7), then TARGET(9); no fallback | 19 |
| depth_noise_0p5 | Shared measured-depth noise realization; tick 5, prefix 0 was the sole planar fallback | 22 |
| occlusion_transient | Physical shutter: UNKNOWN at tick 10, closed wait, fresh BELOW(5) at tick 11 | 20 |
| relax | Modeled strain 0.11424 -> 0.09999 -> 0.11418; fresh observations; distal marker converged in 68 evaluations on open tick 18 | 19 |
| slip_regrasp | Assist grasp count 10 -> 0 -> 9; fresh observations; proximal marker converged in 115 evaluations on open tick 13 | 19 |

Only the two previously unresolved physical-recovery cases needed more
than 60 evaluations. The tool fit kept the original first-hit and silhouette
residual, initialization, Kabsch, loss and gates. The primary v1 work chart
was used unchanged wherever available. No late fallback was permitted.

## Frozen archive checks

All six inherited gates passed. On the same 209 replay entries as v2, all
203 formerly valid tool fits and 208 formerly valid primary charts retained
their values. The five late populations were:

| Population | Available | Max longitudinal overestimate (mm) | Limit (mm) |
| --- | ---: | ---: | ---: |
| Hinge calibration, clean | 6/6 | 0.06904 | 0.070 |
| Hinge calibration, development noise | 71/72 | 0.21419 | 0.220 |
| Hinge calibration, check noise | 71/72 | 0.17956 | 0.220 |
| V1 late nominal, clean | 3/3 | 0.05162 | 0.070 |
| V1 late nominal, check noise | 36/36 | 0.16465 | 0.220 |

All accepted distances were within the frozen margin. An abstention is not
a zero-error estimate. The three late nominal exposures outside the original
calibration six reproduced the archived v1 exposures exactly: they verify
determinism, not generalization to additional independent poses. Each live
case decision records matched pre-injection node hashes and the final edge,
joint and jaw checks.

This completes the registered **non-anatomical M5-v4 rail matrix** at these
seeds and disturbance magnitudes. It does not retroactively pass v1/v2,
establish a universal noise bound, qualify an alternate camera policy or
complete original M5b, M6 or M7.
