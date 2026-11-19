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

- Offline suite: `.venv\Scripts\python.exe scripts\tests\run_regressions.py` (212 tests at 1b8cefa;
  rewrites `validation/safety/regressions.json`). PowerShell may print exit code 1 because of stderr
  wrapping; trust the `OK` / `regressions: True` line.
- Live task: `scripts/task/run_cut_task.py --mode full --scenario <nominal|occlusion_transient|slip_regrasp|
  slip_handover|relax|relax_large|handover_tension> [--trace] [--align-mode shaft|tool_pose]
  [--perception '<json>']`. Default scene `scenes/davinci_si_articulated_verified.ttt`.
- Determinism: identical command sequences give identical `nodes_sha256` traces (`--trace`); extra reads
  are side-effect free. Use trace identity to prove "no control change".

## Rules the project follows

- Pre-register criteria (write the protocol into the output JSON) before evaluating; never loosen
  thresholds, collision envelopes, joint limits or budgets to make a run pass. Report genuine failures.
- Gate statuses in `config/task/fascia_benchmark.json` are signed by the user, never by the agent.
- Original analyses/reports stay as-is; add new versions alongside.

## Status (2026-11-19)

Done and committed: M1 foundation hardening (refused-vs-fault accounting, config roles, full withdrawal,
live slip/regrasp/relax injection, GroundTruth read boundary, timing, trace), M2a 6-DOF tool-pose control
(`toolPoseFK`/`setToolPose`, `scripts/task/tool_pose.py`, default `cut.align_mode = tool_pose`),
M2b wrist-aware reachability, M4 shadow perception (`scripts/vision/cut_perception.py`,
`validation/perception/shadow_eval.json`).

Known live facts: contralateral handover is mechanically impossible in the current port layout
(psm2/psm3 proximal yaw links collide) — honest FAILED. Recovery from relax_large exhausts the
4×0.2 mm retraction budget by design.

**M3 (anatomical placement) is BLOCKED** — see `docs/DEVELOPMENT_ROADMAP_zh.md`, section
"待解决问题：前列腺邻近定位尚不成立". Do not describe results as prostate-adjacent.

Uncommitted / WIP at handoff: see the commit "M3b WIP" (scene-keyed config dir `config/scenes/<stem>/`
with global fallback, protected-structure distance monitor, `tissueCreate` explicit `nodes`,
`scripts/build/build_prostate_proxy_scene.py`). Untracked `validation/anatomy/roi_proxy.json` and
`validation/build/` belong to the abandoned v1 port plan and are stale.

## Next work (user decision: accept anatomical infeasibility, continue on the current scene)

1. Re-verify the old scene live after the M3b WIP changes: nominal `--trace` DONE, same cut/strain as
   `validation/task/sim_full_nominal_tool_pose.json`; run_regressions green.
2. M5a — fix perception before any closed loop (shadow mode only):
   - scissors tip bias ~1.8 mm at zero noise: two markers cannot fix roll; add a third marker or use the
     jaw/hinge geometry so the tip offset is fully determined; target median < 0.5 mm at zero noise;
   - slit front valid-but-wrong on ~94% of ticks (retraction crease extends the slit ~15 mm): detect the
     slit apex (width → 0) and/or depth discontinuity instead of the run end;
   - direction truth baseline: compare against the live cut direction from nodes, not placement-time v;
   - pre-register gates before closing the loop, e.g. nominal valid-but-wrong (> 2 mm) rate < 5% for
     front and tip, and invalid (not wrong) under occlusion.
3. M5b — closed-loop visual servo: alignment/advance/stop from estimates; control path must not call
   GroundTruth (add a test that counts truth reads during control = 0; truth only for logging/scoring);
   fail closed (pause/relocalise/FAILED) on invalid estimates; run nominal, occlusion, slip, noise sweep.
4. M6 — multi-arm coordination: tension-proportional retraction from the visual tension proxy (replace
   fixed 0.2 mm steps; may make relax_large recoverable — report, don't tune to pass); ECM view
   following that keeps cut front and tip centred (replace the fixed pose table); arbitration via the
   guarded controller.
5. M7 — multi-seed paired evaluation of fixed view/fixed traction vs active view vs adaptive traction vs
   both; freeze seeds/budgets/success definitions first.
