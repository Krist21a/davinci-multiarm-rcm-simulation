# M5 visible-fracture development (not acceptance)

This is a versioned, synthetic, non-anatomical sheet experiment. All tests use
the original independent spring-constraint structural score and `strict=True`.
The cutter, graspers, PBD solver, scene, collision limits and closure/reacquisition
guards are unchanged. No image-derived endpoint reader in this document is
connected to the FSM; unsuccessful trials remain unsuccessful.

## Registered interventions and results

| Version | Render-only physical consequence | Actual nominal / noisy length | RGB-D gate |
| --- | --- | --- | --- |
| v1 | Shift each surviving face alongside a lost face 0.4 mm | 3 / 9 mm | FAIL: the apparent first front shifted forward and nominal's second guarded closure refused. No actual 8 mm negative. |
| v2 | Shift only center-column vertices on such faces | 3 / 9 mm | FAIL: same early guarded refusal. |
| v3 | Shift vertices only in static terminal rows 7..12 | 9 / 8 mm | FAIL: the old two-bank terminal reader returned UNKNOWN at both lengths. Mechanical closure traces match the existing controller. |
| v4 | Seven row-8 material ligaments; each persists while any of its three original crossing bonds survives | 9 / 9 mm | FAIL: nominal 9 mm is optically positive, but a distinct 8 mm run at the same legal ECM pose gave a **false STOP** when a still-intact ligament was hidden and three background pixels appeared in its inferred slot. |
| v5 | As v4, plus physically exposed left/right fracture surfaces only when a row-8 cell's original three bonds are all severed | 9 / 9 mm | FAIL: nominal 9 mm gives four consecutive two-sided direct RGB-D STOP candidates, and the old false-STOP 8 mm exposure becomes UNKNOWN. The new actual 8 mm noise run is also UNKNOWN, not a qualifying BELOW. |

V1..v3 decisions are in `validation/task/m5_visible_fracture_dev_v1/`,
`_v2/`, `_v3/`; v4 and v5 source/data/decisions are in the corresponding
`validation/task/m5_visible_fracture_dev_v4/` and `_v5/` directories. These
are separate protocols, not updates to the historical printed-sheet reports.

## Why the negative still blocks control

In v5, the native 640x480 noisy 8 mm work/look images contain no positive
fracture colors, but have only 0..3 identified pixels for each intact
ligament. That is **UNKNOWN**, not a direct observation of all seven surviving
bridges. A passive same-pose 8 mm view in
`validation/task/m5_bridge_view_negative_v1_capture/` proved that the v4
absence-based rule could actually call STOP on intact material. The passive
native-HD negative in `validation/task/m5_bridge_hd_negative_v1_capture/`
preserved the original mechanical trace but also returned UNKNOWN at 8 mm.
The v5 reader correctly refuses the archived 8 mm false-positive exposure;
that safety improvement cannot be substituted for the required supported
BELOW observation.

There is no controller qualification or five-scenario M5 acceptance. No
terminal image certificate authorizes a closure, stop, withdrawal, or DONE.
Do not treat the nominal 9 mm four-frame optical positive as success from
zero; the active run continued to a fourth closure and failed.

A further pre-registered, read-only ECM sweep on the v5 sheet attempted all
seven poses in `config/task/m5_bridge_fracture_view_sweep_v1.json` at actual
8 mm after the same three guarded closures. All 14 images and their refused
or qualified statuses are preserved in
`validation/task/m5_bridge_fracture_view_sweep_v1_capture/`. All ECM commands
landed and the mechanical prefix matched the original v5 noisy run. There
were zero false STOP readings, but **no qualified NOT_STOP**: when a direct slit
and scissors were visible, the two right-side intact bridges were still
missing (0 identifiable pixels) at every pose. The other poses lacked
same-frame direct evidence. The pre-registered gate therefore failed.

The remaining blocker is positive evidence that the row-8 material is still
connected at *real* 8 mm, not the 9 mm fracture signal. A further specimen
would have to expose intact load-bearing material without reading spring
flags directly into a marker or counting absence as intact. Pre-register
that physical observation and a separate 8/9 mm optical screen before
connecting a stop-only reader; then run all five uncut cases with independent
zero-overcut, zero-tear, closed scissors and `_ready_q` checks.
