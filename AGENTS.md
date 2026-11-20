# AGENTS.md — project notes for coding agents

CoppeliaSim da Vinci-style multi-arm simulation (3 PSM + ECM, software RCM, EndoWrist articulation,
PBD tissue cutting, task FSM). Research simulation, not a clinical system. Never claim clinical safety,
fascia/NVB preservation, or measured tissue force (PBD strain is a model quantity).

## Environment (Windows)

- Bash/WSL is broken: use PowerShell. `rg` is not installed.
- Python: `.venv\Scripts\python.exe` (Python 3.13.5, built with `C:\Users\Kuo\.local\bin\uv.exe` from
  `requirements-build.txt`). The default system Python has no numpy/pip.
- CoppeliaSim Edu 4.1.0: `C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu` (headless `-h` works;
  see `scripts/build/simtools.py` `add_arguments`/`connect`). Shut it down after live runs.
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

1. `replay_perception.py --holdout --out validation/perception/m5a_eval_holdout_v4.json` (2nd use of the
   v3 holdouts — disclose), report every gate, then ask the user to accept/reject v4 named failures.
2. M5b v1.1: the 11 runs of `m5b_protocol.json` with output prefix `m5b_v1_1_`,
   `validation/task/m5b_v1_1_summary.json`, B5 strict reading; v1.1 is a development result (same
   deterministic scenarios as v1). Known open risk: memory-front error grew to ~2 mm late in V_nominal v1.
3. M6 — tension-proportional retraction from the visual tension proxy (replace fixed 0.2 mm steps;
   relax_large may become recoverable — report, don't tune), ECM view following (replace the fixed pose
   table), arbitration via the guarded controller. Only after M5b is resolved or explicitly accepted.
4. M7 — multi-seed paired evaluation (fixed vs active view vs adaptive traction vs both); freeze
   seeds/budgets/success definitions first.

Known live facts: contralateral handover is mechanically impossible in the current port layout
(psm2/psm3 proximal yaw links collide) — honest FAILED. relax_large exhausts the 4x0.2 mm retraction
budget by design.
