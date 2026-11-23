# M5 row-9 optical endpoint development: not a stopping permit

The conventional synthetic, non-anatomical PBD sheet still has no verified
visual stopping signal. Structural cut length and PBD strain are model
quantities; neither is a control-camera measurement.

The same-exposure candidate in `scripts/vision/row9_terminal.py` requires a
directly observed slit, a valid observed scissors pose and at least three
row-9 ink-transition pairs on each signed material bank within 1 mm of the
slit. Both median row-to-slit offsets must be within 0.35 mm. This was
preregistered in `config/task/m5_row9_contact_dev_v1.json` as a **passive**
test; it cannot command closure, withdrawal or parking.

| Actual development checkpoint | Optics and camera | Image-only outcome |
| --- | --- | --- |
| Nominal 9 mm, active-look ticks 31/32 | 70 degrees, 1280x960, ECM `[0,4.5,0.088]` | `STOP_CANDIDATE` twice; offsets 0.10-0.18 mm on both banks |
| Depth-noise 8 mm, unmodified full visual FSM | 70 degrees, 640x480 | `UNKNOWN` throughout the post-third-closure observations; too few pairs on one bank |
| Depth-noise 8 mm, extra HD exposure at the unchanged ECM pose | 70 degrees, 1280x960 | `UNKNOWN`; row-9 support remains one-sided |
| Depth-noise 8 mm, guarded ECM moved to the nominal 9 mm camera pose | 70 degrees, 1280x960 | `UNKNOWN` in both same-pose exposures: 0 left-bank pairs, 2 right-bank pairs |

The two full FSM replays in
`validation/task/m5_row9_contact_dev_v1_1_capture/row9_decision.json`
have identical closure/reacquisition ticks, structural observations and
controller hash to their archived counterparts. The extra-HD replay in
`validation/task/m5_row9_hd_dev_v1_1_capture/hd_decision.json` has the
same structural observation sequence as its prior noise run. The guarded
same-pose 8 mm checkpoint and its sealed optical predictions are in
`validation/task/m5_row9_view_dev_v1_capture/`. It was intentionally stopped
after the third closure for optical study; it is **not** a full task run.
The image-only predictions were written before opening the structural labels.
The v1 and HD-v1 acquisition launch/setup errors remain in separate
versioned directories; neither produced an optical acceptance result.

In the same ECM view the estimated slit front projects near `(612,465)` at
actual 8 mm and `(614,467)` at actual 9 mm. The scissors and slit alone do
not resolve the missing bank. Treating that missing bank as proof of either
`CONTINUE` or `STOP`, or reducing the three-pair support after seeing this
negative, would fail the registered test. No terminal override was added
to `CutTaskFSM` or `CutPerception`; M5 has **not** passed.

The next upstream change should make the structural row transition
optically distinguishable on surviving material in *both* banks while
keeping the independent structural length scorer. Before connecting any
new reading to control, test a new uncut-to-8 trajectory as a qualified
negative and a distinct 9-10 mm trajectory as a positive at the same
legal work/look optics. Only then repeat all five full strict-vision runs,
including closed-jaw return to the measured ready pose. Do not label a
missing material bank as a negative control.
