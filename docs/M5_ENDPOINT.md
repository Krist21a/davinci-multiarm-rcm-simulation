# M5 endpoint: run boundary

The task is to cut a **non-anatomical simulation sheet** to an independently
scored 9-10 mm, then withdraw with the scissors closed. A two-bite action cap, a
truth-generated trajectory, a good-looking terminal image, or a passing offline
suite does not satisfy that task.

| Path | Intended use |
| --- | --- |
| `scripts/task/run_cut_task.py` | Existing FSM and guarded controller. Default: truth. Historical vision estimator remains separate from the new observer. |
| `scripts/analysis/local_hd_screen.py` | Original sided-ink wide-field development screen. No face or corner print in those images. Never feeds a prediction to the FSM. |
| `config/task/m5_local_hd_dev_v1.json` | Frozen cases, optics, noise, control images, mechanical checks, scoring and decision gate. |
| `scripts/vision/connected_ruler.py` | Image-only measurement. `local_front` describes the visible boundary; `classify` additionally checks an observed connected prefix. Neither is a movement permit. |
| `scripts/vision/cell_certificate.py` | Public v2 coded-face reader. `local_mm` is a visible local boundary. `prefix` and `decision` additionally require a connected, positively observed cut from the entry. None is a motion command. |
| `scripts/analysis/replay_cell_v2.py`, `cell_v2_wide_capture.py`, `score_cell_v2_wide.py` | Offline coded-print replay and the separate same-print wide-angle development test. |
| `scripts/analysis/*pilot*.py`, `scripts/task/*seam*world.py`, `cell_certificate_v1.py` and historical `validation/task/` directories | Frozen-reader experiments and evidence. The v1 reader bytes match the archived digest. Historical whole-run manifests still require their exact source versions. |

## Reproduce the development screen

Use the repo venv and a dedicated, stopped CoppeliaSim Edu 4.10 instance on a
reserved localhost RPC port. `--stage-dir` must be ASCII. The collector loads a
staged scene, reads back the actual optics, and never saves the scene. Stop the
dedicated instance after collection. Choose a **new** result directory:

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'validation/task/local_hd_repeat_01'
$stage = Join-Path $env:TEMP 'davinci_hd_repeat_01'
foreach ($case in 'pull','a','b','c') {
    & $py scripts/analysis/local_hd_screen.py --out $out --case $case --port 23761 --stage-dir $stage
}
& $py scripts/analysis/local_hd_screen.py --out $out
```

The result of the last command is the **optical gate**, not the process exit code
of a truth acquisition. Each case preserves the effective configuration, separate
RGB-D and structural records, instrument calibration and motion readbacks.
`predictions.json` is sealed before the scorer opens the per-case structural
labels. A nonzero result leaves the failure evidence in `decision.json`.

## Recorded outcome

- **From-zero full visual FSM, not a coupon replay.** The two versioned
  development runners use the conventional v1.10 marked sheet and strict
  vision control. Neither uses the failed indexed ruler or the two-bite cap.
  [Same-bite trials](../validation/task/m5_live_same_bite_v1/five_scenarios.json):
  nominal progresses **0 -> 3 -> 5 -> 9 mm**. The third closure uses a
  two-tick-old material chart, not a two-tick-old slit: the current exposure
  confirms the direct slit, tool, axes and stationary printed transitions on
  both banks. The final closed return to ready is a recovery, **not** optical
  confirmation of completion. Noise/occlusion/slip/relax stop at 5/0/3/6 mm.
- [Active-look follow-up](../validation/task/m5_live_active_look_v1/five_scenarios.json):
  the unchanged full-length scorer also checks bounded closed-camera recovery,
  native HD and a more conservative noisy advance. Nominal now observes a
  mapped slit after **every** closure. However it reads **7.72 mm at actual
  9 mm** after the third closure, makes a fourth closure without structural
  growth, later reads at most **8.54 mm**, and fails without verified
  terminal withdrawal. Neither an improving slit readout nor a zero-growth
  bite is a stop certificate. [Nominal trace and actual ECM exposure](../validation/task/m5_live_active_look_v1/nominal_live.png).
  Neither development policy is the production default or a signed M5 pass.
  Reproduce on a reserved instance with `scripts/task/run_same_bite.py` or
  `scripts/task/run_active_look.py --case <case> --out <fresh-dir> --port <port>`
  (both runners need all three options), then
  `scripts/analysis/report_live_endpoint.py --folder <same-dir>` after the five
  cases. The full optical stop still needs a material-boundary observation
  that distinguishes a genuine 8 from a genuine 9 on the executing controller.
- [Indexed material reference](../validation/task/m5_indexed_hd_dev_v1/decision.json):
  `sheet.indexed_print` is a new, opt-in static **row and column** print on
  the original faces, not additional geometry. `IndexedCutPerception` measures
  retained station columns **9 and 14**. Its central-face and through-gap
  observations are separate: station identity is not a cut, a centre node, a
  connected prefix or a 3D motion target. The structural scorer is unchanged.
  On the actual 8 mm closed look, row-7 col-11 has **no live face** while both
  retained stations have six incident live faces each. RGB-D gives local **8**
  at raw and 0.5 mm noisy depth. The four target station errors are all below
  0.28 mm in both. At **9 mm** the first positively observed central material
  is on row 9, but local progress is **UNKNOWN** in both variants. Its 27-pixel
  deep component touches three registered left-bank pixels and only one on the
  right. One noisy target station is 0.526 mm from its independently scored node.
  Do **not** round it down or drop the bilateral evidence requirement.
  [Actual station images](../validation/task/m5_indexed_hd_dev_v1/terminal_stations.png)
  and [passage diagnostic](../validation/task/m5_indexed_hd_dev_v1/passage_diagnostic.json).
- All 20 truth-development exposures matched the previous nodes, edges, faces,
  cutter TCP and bite/tick record, **not** byte-identical rendered depth. No
  exposure passes that preregistered identity condition. There are also 23
  out-of-tolerance noisy station observations counting repeated trajectory
  checkpoints. No incorrect numeric local ordinal occurred; only the 8 mm
  look/park had a positive one. **The joint development gate fails; no M5 trial.**
  All 34 synthetic checks and [402 offline regressions](../validation/safety/regressions_m5_indexed_reference_v1.json)
  pass. To reproduce, use the collector invocation above with
  `scripts/analysis/capture_indexed_reference.py` and a fresh output directory,
  then `scripts/analysis/score_indexed_reference.py --out <same-dir>` (nonzero
  on this data). `scripts/analysis/indexed_reference_diagnostic.py --out <same-dir>`
  writes the post-score illustration and contact counts. Historical controller
  sources predating this opt-in are in `src/tissue/legacy/pre_indexed_reference_v1.zip`.
- [8/9 mm lip/colour-junction replay](../validation/task/m5_junction_hd_dev_v1/m5_decision.json):
  the opt-in `CutPerception(..., bilateral_surface='junction')` checks the
  intersection of a same-bank colour boundary with measured through-depth or
  the specific opposite ink of an intact face. Its four corner candidates are
  diagnostics, **not** verified material node IDs. On the original 70-degree
  closed 8/9 mm look-backs neither length passes in both depth variants; the
  full reader abstains on all 20 exposures. All 56 synthetic controls pass.
  No live vision command, independent confirmation, or M5 scenario was run.
  [398 offline regressions](../validation/safety/regressions_m5_junction_hd_v1.json) pass.
- [Material support behind that failure](../validation/task/m5_junction_hd_dev_v1/material_support.json):
  at 8 mm the left row-7 central node has **zero** surviving incident triangles.
  Its apparent colour junction is 1.09 mm from the scored node but only 0.20 mm
  from the next node outboard (raw look). A camera cannot directly observe a
  colour boundary on a vertex that no longer belongs to any rendered triangle.
  [Closure-by-closure trace](../validation/task/m5_junction_hd_dev_v1/support_trace.json):
  this node still has six faces after 6 mm and loses all six on the third bite.
  At 9 mm the four *terminal* nodes have surviving material edges, but optical
  registration/accuracy still fails. [Shorter bites were checked as well](../validation/task/m5_junction_hd_dev_v1/short_bite_check.json):
  after matching the original mechanics/poses, both archived 2 mm advance-cap
  trajectories at actual 8 mm also preserve only three of the four target
  colour boundaries. Neither zoom nor just changing 3 mm advances to 2 mm
  repairs this loss of material identity. The next mechanics screen should check
  that the scored lip remains on the rendered material after each bite; monitor
  this offline, not as a truth input to motion. Alternatively register different,
  visibly supported landmarks and validate against the unchanged structural cut
  definition. Do not discard difficult checkpoints or replace them with parked
  images. Reproduce with `scripts/analysis/replay_bank_junction.py --out <new-dir>`
  (nonzero is expected), then `scripts/analysis/junction_material_support.py --out <same-dir>`.
  The two changed pre-junction modules are preserved at their registered paths
  inside `scripts/vision/frozen_bilateral_hd_v1.zip` for isolated v1 replays.
- [Bilateral 3D CutPerception replay on the original 70-degree sided print](../validation/task/m5_bilateral_3d_hd_dev_v1/decision.json):
  the opt-in `bilateral_surface` path measures the two banks separately in camera
  coordinates and returns a local material row and 3D front, forward direction,
  normal and observed sweep-limit distance when supported. The distance is not
  the controller's blade-sweep computation or permission to move. Predictions
  were sealed before structural scoring. All 56 synthetic variants passed, but
  **none of 20 real simulation exposures** (13 distinct RGB-D archives) produced
  a qualified 3D front at either native or noisy depth. The tool alone qualified
  in 11/12 raw work/look observations and 12/12 noisy ones. Without a front,
  its error and the sweep-limit error cannot be reported as zero.
  [Scorer-only attribution](../validation/task/m5_bilateral_3d_hd_dev_v1/projection_diagnostic.json):
  on the 8/9/10 mm looks, respectively 0/1/0 of the two last-open faces met the
  gap predicate with optical endpoints versus 2/2/0 with exact endpoints.
  At 8 mm one inner boundary estimate was 7.88 pixels (1.16 mm) from the
  corresponding material node. This isolates a registration problem at 8/9 mm;
  changing optics for 10 mm alone is not the next order of work.
  Neither the canonical truth-only gate nor the closure-freshness guard was
  opened. No confirmation or step 3 run. To replay: `scripts/analysis/replay_bilateral_3d.py
  --out <fresh-directory>` in the repo venv; expected nonzero exit on this set.
  `scripts/analysis/diagnose_bilateral_projection.py --out <same-directory>`
  writes the independent attribution once. [394 offline regressions](../validation/safety/regressions_m5_bilateral_3d_hd_v1.json) pass.
- [Coded-cell v2 development replay](../validation/task/cell_certificate_v2_dev/decision.json):
  the 20-degree closed look-back reads **8, 9, 10 mm locally** at raw and noisy
  depth; it does not certify the hidden entry. Raw depth recovered positive
  through-depth blocks on 11, 11 and 10 absent faces, respectively, whose old
  corner boxes yielded empty interiors. Across 20 distinct coded-print images,
  all 490 returned corner regions per depth variant enclosed their scorer-only
  projected node. No incorrect positive face or progress occurred in this set.
  None of the five distinct terminal **work** images gave a local ordinal; the
  registered work-and-look gate therefore failed. See the [paired pictures](../validation/task/cell_certificate_v2_dev/working_vs_look.png).
  The 34 oblique synthetic checks passed, including half-cells, bridge, shutter,
  isolated speck and plausible depth contradiction. The first synthetic failure
  (one-sided entry) remains in `validation/task/cell_v2_synthetic_dev_01/`.
- [Step 2, wide field with the **same coded print**](../validation/task/cell_certificate_v2_hd_dev/decision.json):
  zero loaded/pulled and all five 3/6/8/9/10 mm work/look pairs were acquired.
  Every fork returned with closed jaws. At raw depth the tool fit passed all
  ten cut exposures, but none had a numeric local progress, raw or noisy.
  [Matched optics check](../validation/task/cell_certificate_v2_hd_dev/sampling_diagnostic.json):
  all six 8/9/10 mm work/look pairs matched by nodes, connectivity, native RGB-D,
  camera, tick and TCP between the 20-degree and 70-degree acquisitions.
  At wide HD **no single corner code covered more than one decoded pixel** in
  these six exposures. This print is not qualified for the wide endpoint;
  independently scored motion safety and [390 offline tests](../validation/safety/regressions_cell_certificate_v2.json)
  do not change that. Steps 3 and 4 have not started.
- [Development gate](../validation/task/m5_local_hd_dev_v1/decision.json):
  **failed**. All 13 synthetic controls met their raw/noisy expectations. Three
  truth-controlled trajectories ended at 8, 9 and 10 mm with a verified closed
  park; the pull-only control stayed at zero. No work or look frame produced a
  numeric local front. Of the 12 work/look frames the rigid scissors fit passed
  11 at raw depth and 12 with synthetic noise; that is not evidence that noise
  improves localization.
- [Scorer-only diagnostic](../validation/task/m5_local_hd_dev_v1/terminal_rays.json):
  in the 8 and 9 mm look-backs both halves of the last severed row have genuine
  through-depth samples at *known* projected positions. At 10 mm one half has
  no pixel centre in its projection even after the look. One assumption about
  the proposed HD configuration is therefore false on this setup: the image
  does not resolve every terminal half-cell. These coordinates never go into
  the image reader.
- [Mechanical reference](../validation/task/m5_local_hd_dev_v1/truth_reference_check.json):
  the original truth sheet still reaches 10 mm with identical node digests and
  structural lengths at all 17 observations. [386 offline regressions](../validation/safety/regressions_m5_local_hd_v1.json)
  pass. Neither result is a vision-controlled endpoint.

For a new offline v2 replay use `scripts/analysis/replay_cell_v2.py --out <fresh-directory>`
with the repo venv. To repeat the actual 70-degree coded-print captures, use
`scripts/analysis/cell_v2_wide_capture.py --out <fresh-directory> --case <case>`
on a dedicated RPC port and ASCII staging directory as above, in the cases
`zero, three, six, eight, nine, ten`. Score once using
`scripts/analysis/score_cell_v2_wide.py --out <same-directory>`. It checks the
tool and cut reading on the same exposure. These are development forks of truth
trajectories: their safe return to ready is **not** autonomous task completion.

## What must not change to turn a failure into a pass

- A row label is a material address, not evidence that the row was cut. Require
  actual behind-flank depth and intact material ahead. Missing ink or a cropped
  upstream region proves nothing. A visible local front alone cannot certify that
  no upstream bridge remains.
- A wider field can include more of the instrument. It does **not** guarantee
  that all three markers or the terminal cell are resolved at every work pose.
  Test the rigid tool fit on the same raw/noisy exposure as the cut reading.
- Moving to a remembered ready pose is not proof that closing an open cutter
  there is harmless: an actual closing command invokes the tissue model. Do not
  issue a spare closure merely to report a closed-jaw success.
- Only after development passes, freeze a causal contiguous-prefix reader and
  test it on independent trajectories and adversarial bridge/partial-cell
  controls. Only then run the nominal and 0.5 mm vision-controlled task
  end-to-end. Keep the original grasp, strain, RCM, collision, blade-sweep, tear
  and overcut guards. M6/M7 and signed acceptance remain separate.
