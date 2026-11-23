# M5 full-length visual cutting: development checkpoint

These are CoppeliaSim runs on a synthetic, non-anatomical PBD sheet,
not results on anatomical tissue. PBD strain is a model quantity, not
measured tissue force. Neither policy passed M5. M6/M7 are not evaluated.

| From an uncut sheet | Same-bite final mm | Active-look final mm | Active-look closures / direct reacquisitions |
| --- | ---: | ---: | ---: |
| Nominal | 9 | 9 | 4 / 4 |
| 0.5 mm depth noise | 5 | 8 | 3 / 3 |
| Transient occlusion | 0 | 5 | 2 / 2 |
| Slip and regrasp | 3 | 3 | 1 / 1 |
| Relaxation | 6 | 6 | 2 / 1 |

In active-look nominal the third closure reached 9 mm structurally; the next
direct visual reading was 7.72 mm. The fourth closure did not advance the
structural cut, and the run ended without closing and returning to ready.
Reaching the structural interval is not a visual stop. A recovery park is
not endpoint-driven withdrawal.

- [Same-bite, five historical runs](../validation/task/m5_live_same_bite_v1/five_scenarios.json)
- [Active-look, five historical runs](../validation/task/m5_live_active_look_v1/five_scenarios.json)
- [Nominal trace and actual exposure](../validation/task/m5_live_active_look_v1/nominal_live.png)

The original registrations, individual JSONs and summaries are archived
unchanged. They predate policy-bound registrations; **do not use them to test
the new identity checks or as newly verified M5 passes**. Only the raw HD
exposure used in the figure is included here, not all development captures.

## Reproduce

Use CoppeliaSim Edu 4.10, the Python dependencies in `requirements-build.txt`,
and a dedicated simulator listening on port 23873. Use an ASCII staging path.
For example, in PowerShell from the repository root with the venv activated:

```powershell
$out = "validation/task/active_look_$([guid]::NewGuid().ToString('N'))"
foreach ($case in 'nominal','depth_noise_0p5','occlusion_transient','slip_regrasp','relax') {
    python scripts/task/run_active_look.py --case $case --out $out --port 23873 --stage-dir C:/Temp/m5_stage
}
python scripts/analysis/report_live_endpoint.py --folder $out
```

For the same-bite policy use `scripts/task/run_same_bite.py` and a separate
fresh output directory. Each invocation reloads the scene. Exit code 1 from
a live run may be a genuine FSM refusal; inspect the case file's `error`,
`cleanup_error`, state and reason before aggregating. An infrastructure error
is not an optical result. Shut down the dedicated simulator after evaluation.
For an offline check, run `python scripts/tests/run_regressions.py --out` with
a fresh JSON filename.

For new runs, the registration digest covers the selected policy and its
sources. The scorer checks that digest, policy, case, simulator scenario,
noise setting, seed and uniqueness of run IDs before accepting the inputs.
It rejects legacy unbound registrations rather than rewriting them.
