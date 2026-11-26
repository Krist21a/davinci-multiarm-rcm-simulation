# Hinge rail: verified nominal withdrawal; two of five cases pass

[Registered controller and experiment](../config/task/m5_v4_hinge_closed_loop_v1.json)
| [Five-case decisions and refusal diagnostics](../validation/task/m5_v4_hinge_closed_loop_v1/matrix.json)
| [Independent nominal score](../validation/task/m5_v4_hinge_closed_loop_v1/nominal/decision.json)

One fresh scene per case, no retries. The nominal positive is also the nominal
matrix cell. All cases used the same staged hinge, fixture, print, FK v2, v11
outer chart, corrected late row-10 limit, and full v5 plus incremental closed
review. The sensor margin, in millimeters, is `0.070 + 0.3*sigma_mm` only at
sigma 0 and 0.5 on this setup. The 10 mm absolute damage wall did not change.
The two central-read diagnostics remain archived; neither was an input.

The online inverse reproduced all 150 archived draw results including
abstentions; clean / development / old check maxima were 0.06904 / 0.21419 /
0.17956 mm. These known samples were not independent confirmation. The
registered extra closed approach on the fresh nominal path produced three new
late work frames. Their independent maxima were 0.05162 mm (3/3 native) and
0.16465 mm (36/36 check-noise draws), inside 0.070 / 0.220 mm. This is
empirical coverage, not a worst-case sensor guarantee.

| Case | Closures | Optical result / refusal | Final prefix / full / partial (mm) | Closed and ready |
| --- | ---: | --- | --- | --- |
| nominal | 4 | BELOW(3,5,7), TARGET(9) | 9 / 9 / 9 | yes |
| depth_noise_0p5 | 0 | Missing fresh work axis before closure 1 | 0 / 0 / 0 | no |
| occlusion_transient | 4 | Shutter UNKNOWN, fresh BELOW(5), later TARGET(9) | 9 / 9 / 9 | yes |
| relax | 3 | Recovery and fresh BELOW(5,7), then missing tool pose | 7 / 7 / 7 | no |
| slip_regrasp | 2 | Regrasp and fresh BELOW(5), then missing tool pose | 5 / 5 / 5 | no |

Both positives had exactly the same full edge table before and after final
withdrawal, two severed terminal faces, unchanged closed jaw readback and the
original ready joints. All five stayed inside the damage wall with no tears,
RCM failures or collision blocks. All three refusals occurred on a fresh
**open** pre-closure exposure; holding without another jaw command is not
closed withdrawal. No alternate-view recovery success is claimed. Across
the three physical disturbance scenes, all seven common pre-injection work
node hashes matched the nominal controls.

- Noise reached the actual sensor, not just scoring. On tick 5 the three
  markers fit. Both banks supplied all stations, but the first axis fit
  gave row 1 axial skew 0.75562 mm (original limit 0.750 mm). The required
  axis was unavailable. Row-10 range correction was not involved.
- Occlusion used a real, nonrespondable shutter in the camera path. Closed
  exposure 10 was UNKNOWN; after one closed wait and removal, exposure 11
  re-established BELOW(5). No stale permit or alternate view was used.
- Relax lowered modeled strain 0.11424 -> 0.09999, then reload brought it
  back to 0.11418. Slip dropped assist grasp count 10 -> 0, then restored 9.
  Both exercised invalidation, fresh review and subsequent fresh work.
- The later open images still contained the three tool colors, and v11 had
  a chart. The ray/sphere refinement hit its 60-evaluation cap on the distal
  marker (84 pixels, relax) or proximal marker (286 pixels, slip). These
  are fit refusals, not proof that the missing pose would have been accurate.

Next investigate the two archived marker refusals and the separate axis refusal
offline against matched successful exposures; no threshold change or rerun is
part of v1. The five-case M5-v4 acceptance gate is not met. M6 and M7 remain
separate and untested.
