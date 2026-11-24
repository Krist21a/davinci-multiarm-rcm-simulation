# AGENTS.md — project notes for coding agents

CoppeliaSim da Vinci-style multi-arm simulation (3 PSM + ECM, software RCM, EndoWrist articulation,
PBD tissue cutting, task FSM). Research simulation, not a clinical system. Never claim clinical safety,
fascia/NVB preservation, or measured tissue force (PBD strain is a model quantity).

## Environment (Windows)

- Bash/WSL is broken: use PowerShell. `rg` is not installed.
- Python: `.venv\Scripts\python.exe` (Python 3.13.5, built with `C:\Users\Kuo\.local\bin\uv.exe` from
  `requirements-build.txt`). The default system Python has no numpy/pip.
- CoppeliaSim Edu 4.10.0: `C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu`. If a headless RPC
  connection times out with the default profile, the dedicated settings profile and piped stdin in
  `scripts/tests/live_cut_safety_check.py --launch-simulator` worked for the review. Shut down the
  dedicated instance after live runs.
- Line endings: LF (`.gitattributes`). PowerShell writes may produce CRLF; normalise.
- Git identity is configured (Krist21a). Work branch: `system-p0-baseline`. Do not push unless asked.

## Verification

- Offline suite: `.venv\Scripts\python.exe scripts\tests\run_regressions.py` (241 tests at f21e865;
  rewrites `validation/safety/regressions.json`). PowerShell may print exit code 1 because of stderr
  wrapping; trust the `OK` / `regressions: True` line.
- Live task: `scripts/task/run_cut_task.py --mode full --scenario <nominal|occlusion_transient|slip_regrasp|
  slip_handover|relax|relax_large|handover_tension> [--trace] [--align-mode shaft|tool_pose]
  [--perception '<json>'] [--control-source truth|vision] [--initial-view YAW,PITCH]`. Default scene `scenes/davinci_si_articulated_verified.ttt`.
- Determinism: identical command sequences give identical `nodes_sha256` traces (`--trace`); extra reads
  are side-effect free. Use trace identity to prove "no control change".

## Rules the project follows

- Pre-register criteria (write the protocol into the output JSON) before evaluating; never loosen
  thresholds, collision envelopes, joint limits or budgets to make a run pass. Report genuine failures.
- Gate statuses in `config/task/fascia_benchmark.json` are signed by the user, never by the agent.
- Original analyses/reports stay as-is; add new versions alongside.

## Goal

Reproducible, safety-hardened, perception-driven multi-arm cutting benchmark on the existing scene
`scenes/davinci_si_articulated_verified.ttt`: 3 PSM (psm1 retract primary, psm2 cut, psm3 retract
assist) + ECM, software RCM, EndoWrist, PBD tissue sheet, task FSM, then vision-driven cutting (M5),
multi-arm coordination (M6) and a paired evaluation (M7). The tissue is a non-anatomical test sheet.

## Status (2026-11-20)

Done and committed: M1 foundation hardening, M2a 6-DOF tool-pose control (default
`cut.align_mode = tool_pose`), M2b wrist-aware reachability, M4 shadow perception.

**M3 (anatomical placement) is BLOCKED** (user decision: accepted as infeasible) — see
`docs/DEVELOPMENT_ROADMAP_zh.md`, "待解决问题：前列腺邻近定位尚不成立". Never describe results as
prostate-adjacent / fascia / NVB / clinically meaningful. `3a68248` "M3b WIP" (scene-keyed config,
protected-structure monitor, explicit tissue nodes, proxy scene builder) is inactive on the current
scene; untracked `validation/anatomy/`, `validation/build/` are stale (abandoned v1 port plan).

Re-verification after M3b WIP: `8e54b6e` — nominal DONE 10.0 mm, trace identical across repeats.

M5a perception (`scripts/vision/cut_perception.py`, offline replay `scripts/vision/replay_perception.py`,
gates `config/perception/m5a_gates.json` v3, frames in gitignored `validation/perception/frames/`):
- 3-marker Kabsch tip (median 0.31 mm), chromatic sheet mask, tissue-only cut front (the tip only
  picks the near edge once; anti-circularity tests prove the front is tip-independent), fail closed.
- v2 holdout FAILED (occlusion_persistent 38.9 %, relax_large 37.5 % valid-but-wrong) and was burned;
  v3 (`04ea47f`, `e2e8dad`): every wrongness gate passes incl. 4 fresh holdouts; 8 front-availability
  gates fail (0.22–0.72 vs >= 0.8). User accepted those 8 for the v3 estimator only
  (`config/perception/m5a_acceptance.json`, `8ec3193`).
- v4 (`config/perception/m5a_amendment_v4.json`, pre-registered `448bfcd`, code `f21e865`): direction
  slit-refit anchored to the first direction, 5 deg cone, low-pass 0.6 (fixes the closed-loop ratchet
  seen in M5b v1). Dev replay `validation/perception/m5a_eval_dev_v4.json`: all wrongness gates pass,
  the same 4 dev availability gates fail; holdout v4 replay NOT run yet. v3 acceptance does not carry
  over: v4 needs a new user decision.

M5b cutter-arm visual servo (`cut.control_source = vision`; protocol `config/task/m5b_protocol.json`):
- Front fix lifecycle, relative alignment, CLEAR_VIEW recovery, strict purpose-tagged GroundTruth
  (`ground_truth.py`: untagged read raises `TruthReadInControl`). Tension (until M6) and grasp count are
  permitted non-visual signals; tear/overcut are safety monitors only.
- v1 (`e1d5835`, `validation/task/m5b_summary.json`): B0 B1 B3 B4 B6 B7 pass; B2 0/4 DONE and B8 1/4 FAIL;
  B5 is FAIL under the literal reading (V_noise_depth_0p5 ended by a truth acceptance check, not a
  budget/refusal). Causes: direction ratchet (0.13 -> 0.95 rad), late-cut front unprovable, RESUME spec gap.
- v1.1 (`config/task/m5b_protocol_v1_1.json`): vision-mode RESUME/OCCLUDED -> CLEAR_VIEW when scissors and
  tissue are visible but the front is not (code `f21e865`, 241 regression tests green). Live runs NOT done.

## Next work

Indexed optical reference development now uses the retained row/column stations
at columns 9 and 14 (`config/task/m5_indexed_reference_v1.json`), not an inferred
centre vertex. Active entry and reproduction: `docs/M5_ENDPOINT.md`.
`validation/task/m5_indexed_hd_dev_v1/decision.json`: local 8 mm on the closed
look in raw/noisy HD, 9 mm UNKNOWN. All 20 mechanics checkpoints match the sided
print but depth images are not byte-identical. No authorization for closure,
withdrawal or M5--M7. Keep the original structural endpoint and historical data.

Current entry: `docs/M5_ENDPOINT.md` and `config/task/m5_local_hd_dev_v1.json`.
For the passive face/corner print use `scripts/vision/cell_certificate.py` (v2);
historical scorers import the byte-identical `cell_certificate_v1.py`. The v2
development replay is `validation/task/cell_certificate_v2_dev/decision.json`.
20-degree closed look-backs read 8/9/10 mm locally, raw and noisy, but did not
prove a connected prefix. The working shots did not qualify. The matched
70-degree 1280x960 same-print gate (`validation/task/cell_certificate_v2_hd_dev/decision.json`)
includes zero loaded/pulled and actual 3/6/8/9/10 work plus guarded 3-mm look
at every cut length. All five cut checkpoints failed the joint optical gate.
Do not treat `local_mm` as a stop/closure permit, or combine the zoom cut reading
and a marker fit from a different exposure. Still step 2; no independent or M5-M7 run.
Use `scripts/tests/run_regressions.py --out <fresh-versioned-path>` to keep previous
reports and coverage intact. The canonical surface/guide specimen is truth-acquisition
only; the frozen legacy Lua is for hash-pinned historical collectors. Never close
merely to park. The corrected connected reader no longer miscounts nine synthetic
rows, but all 18 historical work frames still abstain (`validation/task/m5_connected_count_replay_v2/decision.json`).
The actual 1280x960, 70-degree local HD screen also FAILED (`validation/task/m5_local_hd_dev_v1/decision.json`).
All 9 work and 3 look-back frames lack a local ordinal, raw and noisy. At the 10 mm
look-back one of the two severed terminal faces projects onto zero pixel centres,
even using exact scorer-only corners. Do not tune away the uncertainty, use parked
or scorer geometry as a control input, or spend confirmation. The previous M5b
result stands. A next optical proposal must show a resolvable local material front
at the actual observation pose, not just more visible ink.

The active objective is the complete visually stopped 9-10 mm task, not a two-bite
limit. `config/task/ruler_endpoint_v1.json` defines an isolated static printed-sheet
experiment. `validation/task/ruler_v1_threshold_development.json`: two parked
development samples per actual 7/8/9/10 mm, all UNKNOWN at native and noisy depth.
`validation/task/ruler_optics_v1_summary.json`: two real ECM poses, two real
resolutions on matched 8/10 mm states, still no bilateral endpoint readout. Do not
use the ruler to authorize closure or withdrawal, or spend independent confirmation
trajectories on a known failing sensor. Do not describe a truth collection as a
vision run. Keep M6/M7 distinct and paused for the strict benchmark.
`validation/task/ruler_endpoint_v1_decision.json` records the fixture screens too:
centre-pivot mounts at 45/60 degrees failed entry localization; entry-preserving 45
degrees restored localization and both grasps but the cutter could not align. Nine
frozen solver seeds all returned the combined shaft IK/collision refusal; this does
not identify a particular collision or rule out other mounts. Plan legal work poses
and optical views together before further controller integration.

The separately named `surface_seam_pilot_v1` coupon changes only the central strip's
load-bearing topology. All five truth collection runs passed mechanical/park checks;
the pulled, never-cut control had zero damage. All 18 post-closure working RGB-D
frames returned UNKNOWN with native and noisy depth, including the actual 8/9 mm pair.
See `validation/task/surface_seam_pilot_v1/decision.json` and `first_hits.json` in
that directory. Row 8 blocks much of the left row-9 ink in the tested view even in
a sheet-only geometric diagnostic. Do not consume confirmation or enable visual
closure. Screen terminal self-occlusion on a loaded, uncut specimen before another
full optical trial; preserve grasp and guarded path feasibility.

Sliding-guide and terminal-rail development is archived in
`validation/task/seam_work_observability_v1/decision.json`. The sliding guide passed
the loaded optics and mechanical screens, not the working 8/9 mm decision. The
locked reader abstained on all 18 post-closure working frames both raw and noisy.
A closed 3 mm look-back exposed distinct local crossings in the two development
states, but is not final withdrawal and does not prove continuous structural length.
A same-frame connected-gap candidate wrongly called a fully separated synthetic
9-row image BELOW; do not import it into a controller. See also the real passive
face-print probes (`seam_face_visibility_v1`): in the 8 mm case a surviving central
face gave a largest depth-identical ink patch of 2 px at work and 24 px after the
back-off. Missing ink is not proof of a cut. Confirmation and M6/M7 remain blocked.

The asymmetric 3-mm central cell uses measured per-row transverse spring rests
(2.76 mm), not the old 1-mm longitudinal pitch, for PBD strain. Keep the old
coupon and its failed records intact. Normal-only rim support and 20-degree optics
qualified intact/traction preflight, not a working endpoint. Five pre-registered
truth runs plus one separate 8-mm development control all passed mechanics. All
20 post-bite images abstained. The matched closed-jaw 3-mm look-back at actual
8/9/10 mm also abstained raw/noisy; each still lacked some entry-side corner codes.
On the 32 severed faces with three bounded corner observations, exact scorer-only
2D locations and measured depth recovered 32 positive blocks, but exact depths
with the image's corner uncertainty recovered zero. This is a localisation-bound
diagnostic, not a sensor verdict. See `validation/task/seam_rim_look_v1/decision.json`
and `validation/task/seam_rim_near_v1/decision.json`. Do not enable control or
consume independent confirmation. These are new specimen results, not old M5b.

The separate two-bite development objective is archived as a baseline. `scripts/task/run_two_bite.py`
and `run_two_bite_pair.py` reuse the existing cut controller, leaving the default strict
endpoint runner untouched. `validation/task/two_bite_v1_verified_summary.json`: nominal
and noise both grew 3 + 2 mm, reacquired twice, and returned to the measured ready pose
with closed jaws. The post-advance shutter produced zero bites. Strict truth retained
the historical 17-tick trace. `validation/task/two_bite_pair_v1_summary.json`: eight
single-seed runs from an identical first-bite checkpoint. Only movable-camera groups
completed the locally occluded second bite. All creep groups stopped; no optical
traction request was exercised. Do not claim a traction benefit or original M5b/M6/M7
acceptance. A future traction comparison must exercise the policy, preserve the safety
band and distinguish a missing visual target from a strain-recovery request.

The review supersedes the pending v4/v1.1 evaluations without changing their historical records.
M5a v5 transports the direction anchor across camera poses and checks replay completeness, capture pose,
and hashes. M5b v1.2 applies explicit ECM poses after homing, preserves the first cutter pose for
withdrawal, checks fresh visual progress and alignment before closure, and samples B7 before actuation.
The new protocols are `config/perception/m5a_amendment_v5.json` and `config/task/m5b_protocol_v1_2.json`.
The v1.2.1 correction (`config/task/m5b_protocol_v1_2_1.json`) treats the opening axis as sign-free,
as the IK already does; jaw pointing remains directed, with the same angular tolerance. Partial
v1.2 diagnostics stay separate. Offline: 252 tests passed (`validation/safety/regressions_m5_review_v1_2_1.json`).

1. `validation/perception/m5a_eval_holdout_v5.json`: two new view-verified captures replace the two v3
   view captures whose angles were reset by homing. The other two v3 holdouts are reused. Across all
   66 gates: 48 pass, eight front-availability failures and ten zero-occlusion-sample failures.
   The two new views have 50.0% and 56.25% front availability, below 80%. No v5 acceptance yet.
2. The 11 v1.2.1 runs are scored in `validation/task/m5b_v1_2_1_summary.json`
   (`scripts/task/score_m5b.py`, strict B5). B0/B1/B6/B7 pass, B2/B4/B5/B8 fail, and B3 has no
   vision DONE sample. Nominal reached 10 mm but exhausted CLEAR_VIEW. Depth noise 0.5 mm reached
   11 mm (allowed max 10 mm), and injected relax_large tore two constraints before cutting.
   These are development diagnostics, not acceptance.
3. M5a v6 occlusion coverage (`config/perception/m5a_amendment_v6.json`): five registered passive
   probes, three co-occluded front/tip frames per scenario, no scored valid-but-wrong target. All ten
   previously unexercised gates are now covered and pass. Use `validation/perception/m5a_eval_holdout_v6_m5b131.json`
   (58/66, eight availability failures); the initial `m5a_eval_holdout_v6.json` was run before the
   last manifest arrived and is not the complete result. Probe NPZ files remain under the ignored
   `validation/perception/frames/probe6_*`.
4. M5b v1.3.1 (`validation/task/m5b_v1_3_1_summary.json`): material progress, a visual/model blade
   sweep bound and a mandatory fresh slit after closure. Truth baseline still matches 17 ticks.
   All three visual diagnostics stopped before a first closure; no overshoot was observed, but
   this run did not exercise near-endpoint success or live post-closure reacquisition.
   Offline: 257 tests at that stage.
   The registered ECM sweep (`validation/task/m5b_ecm_view_study_v1_selection.json`) subsequently
   tested 13 views: nine stayed fixed and showed both ends, none qualified for visual control.
   `+4.5 deg yaw / 0 deg pitch / 80 mm insertion` had both ends in 16/16 sheet frames and five
   paired columns in each. The true loaded chords are 10.04-10.75 mm, shorter than the 11.05 mm
   span prior; this is not merely a framing issue. Optical diagnostics from that pose
   (`validation/task/m5b_ecm_view_optical_diagnostic_v1_summary.json`) failed nominal before
   closure, and failed 0.5 mm depth noise at 8.0 mm after five closures and four reacquisitions
   (no overcut or tear). At tick 74 there is a fresh slit without material scale; at ticks 76-77
   there is scale but only a memory front. Next: a deformation-aware material coordinate based
   on both observed edges. Do not relax gates or reuse memory to bypass a new slit requirement.
   Offline at that stage: 260 tests.
   The v1.4 review correction (`config/task/m5b_protocol_v1_4.json`) fixes guarded motion,
   recovery, estimator coordinates and evidence reporting without relaxing qualification.
   The verified run (`validation/task/m5b_v1_4_verified_summary.json`) matches B0 at 17 ticks;
   default-view nominal and 0.5 mm noise both stop before closure. Eight trials were not run.
   Offline: 274 tests (`validation/safety/regressions_m5_v1_4_verified.json`). Development replay
   (`validation/perception/m5a_eval_dev_m5b_v1_4.json`) has 33 passes, four availability failures,
   and one relative-error failure under rotation; no fresh holdout or ECM qualification was run.
   v1.5 (`config/task/m5b_protocol_v1_5.json`) adds an optical material chart: the same mesh
   renders four flush bands. Each frame must show both endpoints and three internal boundaries
   on both sides of the cut; interpolate in material rows and keep fresh-slit proof separate.
   At the prior diagnostic view it maps 7/7 loaded uncut frames and 11/16 sheet frames overall
   (maximum boundary error 1.41 mm). Marked truth control retains the 17-tick unmarked trace.
   Nominal vision stops at 3.0 mm after one closure without reacquisition; the slit is visible
   when one bank cannot verify row 9. The 0.5 mm noise trial never closes: after advancing it
   repeatedly loses front evidence, even though 283/289 observations have a valid chart.
   See `validation/task/m5b_v1_5_material_coordinate_summary.json` and
   `validation/safety/regressions_m5_v1_5_verified.json` (280 tests). No M5 acceptance.
   v1.6 allows only same-frame adjacent local rows with both ends still verified on both
   banks. A missing limit cell restricts the modeled sweep to the last verified earlier row.
   Nominal reacquires at world tick 14 with row 9 absent (0-3 mm cell, 2.79 mm estimated);
   the following frame has only a memory front, so no second closure. Noise again stops
   before closure (286/289 charts, 198/289 mapped fronts). The same-view truth trace still
   matches all 17 ticks. See `validation/task/m5b_v1_6_material_coordinate_summary.json` and
   `validation/safety/regressions_m5_v1_6.json` (283 tests). No M5 acceptance.
   v1.7 fixes the rank of a current provable slit vs a memory candidate within one bin.
   On replay, nominal world ticks 15-20 no longer substitute memory for visible slit.
   Live nominal reaches ALIGN after reacquisition, then loses the fix after alignment
   motion (3.0 mm, one closure, one reacquisition). After advance in the 0.5 mm noise trial,
   all three guarded camera retries fail to restore same-frame evidence (0 mm).
   Truth still matches all 17 ticks. See `validation/task/m5b_v1_7_material_coordinate_summary.json`
   and `validation/safety/regressions_m5_v1_7.json` (285 tests). No M5 acceptance.
   v1.8 predicts the candidate open cutter's sightline clearance before guarded alignment
   or advance. All 35 predictions selected the first candidate. Only 2/10 nominal and
   1/25 noise predictions were followed immediately by a directly observed, material-mapped
   front and visible scissors. Both node traces match v1.7; nominal stops at 3.0 mm and
   noise never closes. See `validation/task/m5b_v1_8_material_coordinate_summary.json`
   and `validation/task/m5b_v1_8_view_pairs.json`. The latter is a post-hoc diagnostic;
   its raw pixel comparisons use only exact trace/perception prefixes from v1.6 captures.
   `validation/task/m5b_v1_8_verified_truth_identity.json` repeats truth with the final
   runner after correcting its missing-align-mode default; the same 17-tick trace is retained.
   Evaluate an image-based prediction of the actual look-ahead support on captures before
   another controller revision. Do not equate no intersection with a blade proxy to sheet visibility.
   That work-pose diagnostic is now complete (`config/task/m5b_workpose_pair_v1.json`,
   `validation/task/m5b_workpose_pair_v1_capture/`). Six guarded one-action replays, matched
   before motion, no additional closure. None of the three tested alignment setbacks preserved
   direct slit evidence. A 0.75 mm advance restored entry evidence but left the estimated tip
   0.50 mm short. Post-hoc frozen renders at the two original work poses isolate the green
   `psm2_tool_tip_visual`: hiding this shape alone restores a direct slit/entry estimate at
   the same noise draw. Verify its physical vs visualization role before any optical change;
   do not remove it from collision checking. No evidence of second-bite or endpoint success.
   Superseded by v1.9: the foot-docking URDF explicitly calls the sphere a transparent
   `tool_tip_marker`; the dVRK 179 has separate jaw meshes and an empty tool-tip link.
   Exclude only the verified staged marker from rendering, keeping collision geometry and
   non-render flags. Nominal now advances 0 -> 3 -> 5 mm and reacquires twice; 0.5 mm
   noise advances 0 -> 3 mm and reacquires once. Both exhaust CLEAR_VIEW without tear or
   excess over the 10 mm limit. The opt-in v1.9.1 guarded last-attempt camera alternative
   does not improve either outcome. `validation/task/m5b_v1_9_1_verified_summary.json`;
   `validation/safety/regressions_m5_v1_9_1.json` (290 tests). Truth remains 17 ticks / 10 mm.
   Seven independent frozen views at each of three checkpoints: one useful view at nominal
   5 mm; none at noisy 3 mm or withdrawn reference 10 mm. At the latter, all saw a slit
   but no valid material chart. The offline local-surface hypothesis still estimated only
   7.94 mm at 10 mm: do not deploy it. `validation/task/m5b_v19_frozen_label_ambiguity.json`
   demonstrates that the scored second diagonal is not a rendered face edge: identical
   fixed vertices, face indices and bands allow different structural pass/fail labels.
   Not a second reachable trajectory or an impossibility proof for multi-frame perception.
   Do not feed hidden constraints into visual control or silently relabel the endpoint.
   v1.10 opt-in ray/sphere marker fit retains both the old and refined rigid-pose RMS guards.
   No holdouts or default change. Paired 0.5 mm noise tip error median/p95: 0.80/1.49 ->
   0.47/0.82 mm (25/25 poses). Clean replay loses 3/30 poses. Live nominal and noise both
   achieve 3 + 2 mm real cut growth and two current-slit reacquisitions, then stop at 5 mm
   on exhausted CLEAR_VIEW. Truth still matches 17 ticks and 10 mm. See
   `validation/task/m5b_v1_10_verified_summary.json`. On the two 5 mm failures and withdrawn
   10 mm reference, the physical entry row alone is within the 1.5 mm spread gate,
   but projecting its same vertices to the global fitted plane pushes it outside.
   The local per-boundary depth probe also fails its four checkpoints; no controller change.
   See `validation/task/m5b_v1_10_entry_skew.json`, `validation/task/m5b_v1_10_boundary_depth.json`.
   No controller change after that run. Independent apparent row fits at the two 5 mm
   checkpoints read 4.61 / 5.11 mm but fail the physical check. At the withdrawn 10 mm
   reference, both fits underestimate and disagree: 0/4 guarded checkpoints pass
   (`validation/task/m5b_v1_10_verified_bank_transects.json`). Direct contact of the terminal
   colour band with gap pixels is present at both 10 mm frames, absent in 39 subtarget
   direct-slit frames (3-6 mm). The six positive-side contact pixels at each endpoint also
   occur in the negative-side set. This is not proof of two material banks or a stop
   signal; no 7-8 mm negative or independent terminal trajectory was tested. See
   `validation/task/m5b_v1_10_terminal_band_contact.json`. Test bank identity and near-
   threshold negatives independently before considering this cue for any control use.
5. M6 — tension-proportional retraction from the visual tension proxy (replace fixed 0.2 mm steps;
   relax_large may become recoverable — report, don't tune), ECM view following (replace the fixed pose
   table), arbitration via the guarded controller. Only after M5b is resolved or explicitly accepted.
6. M7 — multi-seed paired evaluation (fixed vs active view vs adaptive traction vs both); freeze
   seeds/budgets/success definitions first.

Known live facts: contralateral handover is mechanically impossible in the current port layout
(psm2/psm3 proximal yaw links collide) — honest FAILED. relax_large exhausts the 4x0.2 mm retraction
budget by design.
