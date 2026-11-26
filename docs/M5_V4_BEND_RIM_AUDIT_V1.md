# Hinge coupon: why the wide-view row-10 limit is optimistic

The [registered offline audit](../config/task/m5_v4_bend_rim_audit_v1.json)
replays all six distinct native late work images from the terminal contrast.
No image or threshold was selected on the result. For each bank, it reproduces
the original row-9/10 rim pair, fitted depths, weights, harmonic midpoint and
station median. Only after fixing those does it intersect the same rays with
the live sheet triangles in the recorded camera frame. Nodes are from the
exposure's own sidecar; edge liveness is from the preceding closed review at
the same closure count. No motion code reads this analysis.

## Evidence

Each bank/station used one pair of two fitted edge midpoints. Across six images
the left bank used 551 pixel observations (unique within each frame), the right
another 551. The left windows sample longitudinal rows 9/10 and parts of 8/11,
columns 5-8; right windows sample columns 14-17. All support rays hit the
registered sheet geometry, all material-side/row colors agree, with no central
cap/face ambiguity. This is not evidence of mixed foreground or of a broken
corner fallback. First hits are sheet-only geometric predictions; matching the
print does not independently calibrate the renderer.

On all six frames the left estimate wins the optical minimum. Signed errors
along the *frozen optical axis*, in mm:

| Substitution, holding the selected supports and axis fixed | Range (left) |
| --- | ---: |
| Actual sampled station ray on sheet vs true row-10 rim vertex | +0.0522 to +0.0760 |
| Combine the two true edge-midpoint depths instead of one pair-center depth | -0.00033 to -0.00010 |
| Fit with the original weights on exact sheet depths | -0.00051 to +0.00011 |
| Substitute native measured support depths for exact sheet depths | +0.15549 to +0.15646 |
| True left rim node vs true seam midpoint | +0.02352 to +0.08883 |

The measured support depths are themselves farther than their ideal first hits:
approximately 0.164-0.170 mm on the left, 0.166-0.180 mm on the right (camera-z,
not axial projection). These six views of one setup do not determine whether
the discrepancy comes from depth readout, projection/registration, or rendering.
Replacing only the selected left support depths by ideal depths, leaving the
tip, axis and winner fixed, would **still** give +0.03289 to +0.06906 mm total
optimism. This is a diagnostic substitution, not a corrected controller.

A separate geometry-only check uses the smaller of **both true outer nodes**
against the center midpoint, with exact tip and the same measured axis. Four of
six values are positive; the maximum is +0.04872 mm. Even exact outer nodes
are not a guaranteed zero-intercept safety boundary on this deformed coupon.

## Decision

Do not change the polynomial, tune a pixel offset, or subtract the observed
bias as a constant. First test the RGB-D ray convention and depth readout
against an independent rigid planar reference at several tilts (including
opposite slopes), ranges and image positions with held-out poses. Then either
measure a directly visible central material boundary or independently bound
the outer-to-center deformation. Both issues must be covered before re-testing
native and noisy work margins. No new positive run or safety margin is issued.

[Numerical audit](../validation/task/m5_v4_bend_rim_audit_v1/decision.json) |
[Per-pixel support and signed weights](../validation/task/m5_v4_bend_rim_audit_v1/support.json) |
[Versioned regressions](../validation/safety/regressions_m5_v4_bend_rim_audit_v1.json)
