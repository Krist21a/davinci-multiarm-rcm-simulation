# Hinge coupon: post-cut terminal contrast and work-distance envelope

The registered [two-case screen](../config/task/m5_v4_bend_terminal_v1.json)
passed the closed RGB-D terminal contrast and failed the wide work margin.
Both fresh scenes used the fixed **0.250 mm** collection protection with native
depth, not the margin under test. All scopes are simulation development only.

| Fourth closure | Independent closed primary read | Structural prefix / farthest full / farthest partial | Row 8 faces intact |
| --- | --- | --- | --- |
| Ordinary cap, 2.0 mm | `TARGET(9)` | 9 / 9 / 9 | false, false |
| Reduced cap, 1.5 mm | `BELOW(8)` | 8 / 8 / 9 | false, true |

The initial frames matched the loaded hinge control. Both optical trajectories
and joint commands matched through the pre-advance fork. The common receipts
were `BELOW(3,5,7)`. The true partial state arose from a native sweep, not a
deleted edge, changed reader or structural progress permit. Both cases stopped
closed at the entrance, with no fifth closure or final withdrawal. No tear or
RCM/collision fault was observed. The two terminal controls do not by themselves
constitute a positive or a noise robustness result.

## Frozen wide error gate

Each case retained 14 paired work references. Count shared RGB-D/prefix/tick
observations only once. All required jaw/late-prefix strata were represented.
The empirical upper bounds were violated even at native depth:

| Unique late samples | Valid pairs | Maximum optimism, mm | Bound, mm |
| --- | ---: | ---: | ---: |
| Native | 6/6 | +0.22540 | 0.0000 |
| 0.5 mm development noise | 70/72 | +0.37079 | 0.1000 |
| 0.5 mm check noise | 70/72 | +0.33779 | 0.1000 |

The [decision](../validation/task/m5_v4_bend_terminal_v1/decision.json) records
per-case physical coverage separately from the aggregate qualification flag.
The latter includes the per-case envelope gate, so its `coverage=false` is not
evidence that a physical jaw/ordinal stratum was missing. Maximum FK/TCP norm
was 0.000125 mm; maximum global axis error was 1.192 degrees.

## Post-hoc diagnosis, not a fitted correction

The [same-exposure decomposition](../validation/task/m5_v4_bend_terminal_v1/clean_attribution.json)
compares the selected observed outer-bank station with its true printed rim
node, then with the seam midpoint, along the measured axis. On all six native
late frames the minimum was the **left main-path** station, not a v11 fallback.
At the true rim depth, the estimated pixel ray contributed +0.006 to +0.008 mm;
at that ray, the fitted-depth displacement contributed +0.202 to +0.223 mm.
The real outer-to-center offset added +0.024 to +0.089 mm. Tip error partly
cancelled the optimism. These are signed axial contributions, not raw depth
RMSE or proof that one operation in the depth fit caused the entire discrepancy.

Next: inspect the depth support and station construction on the sealed images,
separately from deformation of the rim relative to the seam. A different
estimator or optical limit requires new, independently registered evidence;
do not subtract a constant derived from these samples. No new margin or
positive/M6/M7 run is authorized.
