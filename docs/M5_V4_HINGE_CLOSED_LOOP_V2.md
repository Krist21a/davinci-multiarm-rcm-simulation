# Hinge v2: three refusal controls recovered; unchanged room envelope fails

[Preregistration](../config/task/m5_v4_hinge_closed_loop_v2.json) |
[Offline decision](../validation/task/m5_v4_hinge_closed_loop_v2/offline_gate.json) |
[Paired archive results](../validation/task/m5_v4_hinge_closed_loop_v2/archive_measurements.json) |
[Matrix status](../validation/task/m5_v4_hinge_closed_loop_v2/matrix.json)

Only the marker ray residual and reciprocal-depth intercept basis changed.
The historical readers were not edited. The work reader uses the original
v11 chart/axis gate and the same zero-parameter raster inverse as v1. Both
the native outer-rim and three-pixel corner branches use the planar basis.
Tool, chart and axes always use the same draw. No center reader, print
change, margin change, or score-fed control. This is a simulation benchmark.

## Challenged open frames

| Case / work frame | Challenged fit: old -> new evaluations | New Kabsch RMS (mm) | TCP error (mm) |
| --- | --- | ---: | ---: |
| relax / 018 | distal: 60 (no convergence) -> 6 | 0.02543 | 0.15535 |
| slip_regrasp / 013 | proximal: 60 (no convergence) -> 10 | 0.02178 | 0.14816 |
| depth_noise_0p5 / 005 | all-station work axis now accepted; new marker fits 26/22/25 | 0.07035 | 0.25398 |

The noisy tick 5 replay passed open-pose alignment and the unchanged swept
blade check (4.5056 mm room); this was not an actual closure. Across all 14
native nominal work frames the new maximum fit cost was 39 evaluations per
marker, maximum RMS 0.03577 mm, maximum TCP error 0.24900 mm. Across all 4
actual v1 noise work frames these were 31 evaluations, 0.10767 mm and
0.48858 mm respectively. Original and revised per-marker solver outcomes,
RMS and tip errors are in the paired archive results. These summaries are
not bounds on the other seeds.

## Registered independent scores

Positive error means estimated longitudinal distance remaining exceeds the
same-exposure structural midpoint-to-TCP distance on the measured cut axis.

| Population | Available / registered | Max optimism (mm) | Limit (mm) | Over limit |
| --- | ---: | ---: | ---: | ---: |
| Original hinge, clean | 6/6 | 0.07878 | 0.070 | 1 |
| Original hinge, development | 72/72 | 0.40022 | 0.220 | 5 |
| Original hinge, check | 67/72 | 0.19788 | 0.220 | 0 |
| V1 nominal's new late frames, clean | 3/3 | 0.05583 | 0.070 | 0 |
| Same late frames, check | 35/36 | 0.25535 | 0.220 | 1 |

The original check split also missed its 95% availability gate; its five
refusals had work charts but no converged three-marker pose. The new late
population is archived v1 data, not fresh v2 trajectories. Archive replay
and unit regression are not substitutes for a passing live matrix.

## Attribution after sealing the decision

An [offline 2x2 comparison](../validation/task/m5_v4_hinge_closed_loop_v2/posthoc_factorial.json)
replayed both marker choices with both chart choices, without fitting or
selecting a replacement. At the worst combined draw (short / 019_work,
seed 6111) the signed room errors in mm were:

| Outer chart | Original marker | Revised marker |
| --- | ---: | ---: |
| Four terms | +0.21419 | +0.40399 |
| Three terms | +0.21009 | +0.40022 |

Thus the marker change accounts for most of the difference on this particular
draw. Its rigid-fit RMS was only 0.11704 mm but TCP error rose from 0.94039
to 1.42490 mm. The clean failing frame even had *lower* Euclidean tip error
after revision: this does not imply a conservative signed axial clearance.
The plane-only change also had two development violations (max 0.24612 mm)
on the full set. Neither isolated change is qualified as a global replacement.

**Decision:** preserve the negative offline gate and leave all five new
scenes unattempted. No dedicated simulator launched, no v2 cutter motion,
no signed milestone, no change to the v1 nominal/occlusion positives. Any
different policy for recovering refused measurements requires a new protocol.
