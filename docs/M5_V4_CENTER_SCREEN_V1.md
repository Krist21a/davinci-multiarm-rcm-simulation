# Direct central row-10 screen: no measurable clearance at the work pose

The [registered offline screen](../config/task/m5_v4_center_screen_v1.json)
tested the six distinct late hinge work exposures. It required either the
two actual row-10 node caps (columns 11/12) or the two central triangles that
share their transverse edge. The outer chart supplied a direction only, not
an endpoint. Local depth, registration and marker were replayed on the same
full-image noise draw. Predictions were sealed before opening scoring arrays.

| Distinct work draws | Tool marker | Work axis | Direct central witness | Distance available |
| --- | ---: | ---: | ---: | ---: |
| Native | 6/6 | 6/6 | 0/6 | 0/6 |
| 0.5 mm development noise | 70/72 | 72/72 | 0/72 | 0/72 |
| 0.5 mm check noise | 70/72 | 72/72 | 0/72 | 0/72 |

All six real images contained **zero** left cap pixels and **one** right cap
pixel. The face before the edge had 7-8 pixels. The face after had 8, split
into components of 7 and 1. Each face had only 3 distinct depth-compatible
4-neighbor contacts with the other at native depth (4 required per side).
This included the closed pre-advance and both open pre-closure exposures.

Sheet-only first-hit geometry independently reproduced the observed positions
and counts for all four codes in every native frame. The left cap itself did
not contain a projected pixel center, even before first-hit occlusion testing.
Depth noise changes neither the RGB coverage nor that native sampling result.
Abstentions have **no signed distance error**; do not report zero overestimate,
zero usable travel, successful stopping or a safe numerical clearance.

## Density-only diagnostic

After the negative decision was sealed, one separate scoring-only check kept
the same poses, 70-degree FOV and print, and tested a 2560x1920 pixel-center
lattice against the same recorded sheet geometry. Its native counts, component
topology and contacts matched all six records. On the doubled lattice the
left cap covered 3 pixels, the right cap 1-2; the two faces reached 11/11
four-neighbor contacts but at least one qualifying face was fragmented under
the registered single-component rule. Neither route obtained a witness on any
of the six frozen shapes. This is **not** a rendered HD exposure or proof that
all HD observers are impossible.

Do not schedule a cutting run, relax the contact threshold or simply double
work resolution. A next design candidate should make the *actual central
station* and tool marker jointly observable, e.g. a separately named print
with an identifiable transverse mark tied to the node pair. Screen projected
coverage **and** the physical offset/axial interval of any larger mark before
capture: a visible label far from its node can be as misleading as the outer
bank. Mechanics and the absolute wall stay frozen. Independent nominal and
disturbance validation have not started.

[Optical and structural decision](../validation/task/m5_v4_center_screen_v1/decision.json) |
[Geometry-only density comparison](../validation/task/m5_v4_center_density_v1/decision.json) |
[Screen regressions](../validation/safety/regressions_m5_v4_center_screen_v1.json)
