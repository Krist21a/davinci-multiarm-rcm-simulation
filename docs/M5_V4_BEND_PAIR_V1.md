# Uncut loaded coupon: stock vs transverse hinge

The [registered pair](../validation/task/m5_v4_bend_pair_v1/registration.json)
passed the [uncut development screen](../validation/task/m5_v4_bend_pair_v1/decision.json).
One fresh scene per variant, identical fixture, loading commands, joint and
camera settings, exposure times, print and readers. The stock first exposure
and node reference reproduce the archived v5 first frame. Both cutters stayed
closed at ready after setup. No positive, disturbance or cutting run was made.

| Zero-noise metric | Stock | Transverse hinge |
| --- | ---: | ---: |
| Work: alternating normal displacement RMS, mm | 0.126758 | 0.000799 |
| Primary: weakest valid bank patch, rows 6-10, px | 965 | 3362 |
| Alternative: weakest valid bank patch, px | 702 | 2891 |
| Positively intact rows in both narrow views | 6, 8, 10 | 6, 7, 8, 9, 10 |
| Wide main-path stations per side; marker valid | 13/13; yes | 13/13; yes |
| Grasped nodes, primary/assist | 8/10 | 9/10 |
| Loaded work model cut-line strain | 0.07336 | 0.08587 |

The intervention changes a mechanical response, not just the rendering:
matched commands do not imply identical grasped material or local strain.
The effect is established for this pair, not as a universal reduction in
folding or proof that the previously cut row 7 would have certified OPEN.
All 1219 original structural constraints remained alive and identical, the
strain and RCM guards passed, and neither scene was saved.

The separately staged [hinge](../src/tissue/m5_v4_transverse_hinge_v1.lua)
projects the signed volume about each shared transverse edge toward its
flat rest state. It is a numerical regularizer, not a calibrated tissue
property; it cannot relabel a spring or triangle. In [Lua counterexamples](../scripts/tests/test_m5_v4_bend_pair_v1.py)
a half-cut removes the local hinge, a pinned vertex does not move, and a
perturbation on one completely separated bank cannot move the other through
the bending pass. The [versioned suite](../validation/safety/regressions_m5_v4_bend_pair_v1.json)
passed 491 tests; these counterexamples do not validate a live swept cut.

The [RGB-D predictions](../validation/task/m5_v4_bend_pair_v1/optical_predictions.json)
were sealed before joining the scoring sidecars. The next evidence needed is
late-state cut observability and an independent check of the work-distance
error envelope on this candidate, including native and 0.5 mm noisy depth.
Do not use this result to transplant the old margin, enable final withdrawal,
declare M5 complete or proceed to M6/M7.
