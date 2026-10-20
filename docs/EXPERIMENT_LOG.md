# Experiment Log

**Status:** `EXP-0001` through `EXP-0006` completed. No other experiments have
been run.

This file is the authoritative record of experiments. An experiment is logged
only if it can be reproduced from the recorded configuration, dataset version,
split, and code revision.

## How to use this log

1. Copy the template into the **Entries** section, newest first.
2. Assign the next sequential ID: `EXP-0001`, `EXP-0002`, ...
3. Record the exact code revision (`git rev-parse --short HEAD`) and note whether
   the working tree was clean.
4. Link the config path and the dataset/split identifiers; do not paste large
   tables here. Generated outputs belong under `reports/` (Git-ignored).
5. Record negative, failed, and inconclusive runs. Mark them as such.
6. Never edit a closed entry to "improve" a result. Supersede it with a new ID
   and cross-reference the correction.

## Conventions

| Field | Values |
| --- | --- |
| Status | `planned`, `running`, `completed`, `failed`, `inconclusive`, `superseded` |
| Task | `target-validation`, `dataset-generation`, `quality-regression`, `conventional-benchmark`, `corruption-classification`, `severity-classification`, `cnn-comparison`, `robustness-analysis` |
| Split | Frozen split ID plus hash, for example `split-v1` (`<hash>`) |
| Dataset | Dataset version plus manifest hash |
| Metrics | Name, value, unit, and confidence interval where applicable |

## Index

| ID | Date | Task | Status | One-line outcome |
| --- | --- | --- | --- | --- |
| `EXP-0006` | 2026-10-20 | `robustness-analysis` | `completed` | Smooth elastic warp is deterministic, replayable and monotonic on 60/60 sources; gate is CONDITIONAL PASS because the 0.5 px rung has between-source SD 0.0025 below the 0.005 margin |
| `EXP-0005` | 2026-10-20 | `robustness-analysis` | `completed` | Frozen M3 fails zero-shot absolute calibration on M4A test (R² -1.045); validation-selected retraining cuts RMSE from 0.161 to 0.018, so conventional adaptation is sufficient here |
| `EXP-0004` | 2026-10-20 | `robustness-analysis` | `completed` | Occlusion/interference pilot is deterministic, non-degenerate and feature-distinct from blur/noise (gate PASS); no difficulty or solvability claim |
| `EXP-0003` | 2026-10-20 | `conventional-benchmark` | `completed` | A validation-selected random forest reaches test R² 0.986 (RMSE 0.023) versus naive 0.194; held-out-font R² 0.980; CNN not justified for this target |
| `EXP-0002` | 2026-10-20 | `dataset-generation` | `completed` | Two deterministic datasets generated; grouped splits verified; no single-feature triviality, but an eight-feature linear baseline reaches test R² 0.962 |
| `EXP-0001` | 2026-10-20 | `target-validation` | `completed` | SSIM passes in both families; PSNR passes for noise but fails for blur on an ink-coverage confound; MS-SSIM deferred |

## Entries

<!-- Copy the template below. Newest entries first. -->

```markdown
### EXP-XXXX — <short, specific title>

- **Date:**
- **Status:**
- **Task:**
- **Hypothesis / question:**
- **Code revision:** `<sha>` (working tree clean: yes/no)
- **Environment:** Python `<version>`, key library versions, hardware
- **Configuration:** `configs/<file>.yaml` (config hash: `<hash>`)
- **Dataset:** `<dataset-version>` (manifest hash: `<hash>`)
- **Split:** `<split-id>` (hash: `<hash>`), groups: `<n>`, images: `<n>`
- **Seeds:**
- **Runtime:**

**Method.** What was run, and what differed from the previous experiment.

**Metrics.**

| Metric | Value | 95% CI | Notes |
| --- | --- | --- | --- |
| | | | |

**Observations.** What happened, including unexpected behaviour, instability,
or data issues. Facts only; interpretation goes below.

**Conclusion.** What this supports or rules out, and how confident we are.

**Limitations.** Threats specific to this run (small groups, metric saturation,
class imbalance, and so on).

**Next step.** The concrete follow-up, or "none".

**Artifacts.** Paths to Git-ignored reports/figures, if any.
```

---

### EXP-0006 — Geometric-distortion feasibility pilot: replayable and monotonic, conditional pass on the lowest rung

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Before adding a geometric family to the grouped
  robustness protocol, does a bounded smooth elastic warp produce a valid,
  deterministic and replayable SSIM target with an interpretable severity
  response and a feature pattern that is not merely a reparameterisation of
  blur, noise or occlusion/interference? This is a synthetic image-similarity
  pilot, not a human-readability, CAPTCHA-solvability or security study.
- **Code revision:** base `968d85b` plus the M5A implementation, committed as
  `feat: pilot geometric distortion robustness`. The reported run was repeated
  after the commit so generated provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-image
  0.26.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/geometric_distortion_pilot_v0_1_0.toml`; config
  sha256 `00b8c5caad610fb4aaf159bb8c99e329ad217aeff6450e7cd3e566387e9f0a97`;
  pilot generation hash
  `e5901ce89fda1854d606aa41b6258222aa77e9bde061c9082971d7660a924c2a`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates. The geometry was
  fixed before generation: amplitudes `[0.5, 1.0, 2.0, 3.0]` px maximum
  displacement, minimal control `0.25` px, 5x13 coarse control grid, 2.0 px
  Gaussian smoothing, bilinear interpolation, edge border mode.
- **Datasets:** frozen M3 `v0.1.0` used as the clean reference: 60 sources, 540
  rows, manifest sha256
  `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`,
  features sha256
  `d3715e7975cf41c41e49fa2338ca4e961a7a59981318be19149c78e89804d798`,
  split-assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`.
  New pilot `v0.1.0`: 60 sources, 480 rows = 60 clean + 180 controls (60
  identity, 60 zero-field, 60 minimal) + 240 elastic (4 severities x 60);
  manifest sha256
  `c63be1088db591181f23fa306f955b3ba16fe80ffb2000c4a5e3db7656fa3806`,
  sources sha256
  `b565aed1f3d84d7d9c2685e8145868208d77c24b831a1f8c205778e2196a7ed0`,
  features sha256
  `cb82ae60d8749f21d9d5f6dd324cd075e907544aa54bcdb62bbcab988472bb65`,
  image-pixel-hash set
  `33d14063be82a33cd6518b8fd79c4a78838f601b859b2eede56de36b2a33264b`.
  Frozen M4A comparison: manifest
  `d08c1824ba81e3bc40a585b3047eec6c5f9aa2da0b0b9eb126057ab27fdcc92b`,
  features
  `261cfe9a5e2c3910f08361c800f7bcd7252c86d7b56c32006c7970bfb63a1ad9`,
  summary
  `cd9dc97ff3a25c85976b26367da4a5b5f475d8dc8aeba671a55aecb981705ed4`.
- **Split:** frozen source-grouped split, 36/12/12 train/validation/test
  sources; every clean source and all its controls/severities stay in its
  assigned split. All 60 clean references were re-rendered exactly.
- **Seeds:** pilot seed `20261024`; per-source field seed derived from pilot
  seed, `geometric`, and `source_id`; 1000 source-grouped bootstrap replicates.
- **Runtime:** 5.7 s through the CLI, including frozen-asset verification,
  exact clean re-rendering, two independent builds, 420 transformed-image
  replays, analysis, gate and report.

**Method.** Loaded and hash-checked the frozen M3 dataset, M4A pilot and M4B
artifacts; re-rendered all 60 clean sources from
`configs/dataset_v0_1_0.toml` and required exact pixel equality. Generated one
seeded unit-maximum displacement field per source, then every severity and
control directly from the clean reference by scaling that same field. The field
is zero-mean before normalisation, so there is no global translation. Stored the
full deterministic field specification plus an exact SHA-256 of the float64
field; verification regenerates the field and transformed pixels without
consulting the target. Target and features are unchanged: `ssim_target` against
the exact clean reference (PSNR secondary) and the existing eight image-only
features in their fixed order. Controls: identity bypasses resampling;
zero-field passes integer coordinates through the bilinear sampler; minimal
applies 0.25 px of the same field. Two independent builds were compared by
manifest, source table, feature table, targets and decoded image hashes, and the
written dataset was re-verified from disk.

**Target behaviour.**

| Severity | Amplitude (px) | n | SSIM mean (SD) | SSIM range | PSNR mean (dB) | Changed pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5 | 60 | 0.9923 (0.0025) | 0.9868-0.9975 | 31.41 | 0.212 |
| 2 | 1.0 | 60 | 0.9713 (0.0091) | 0.9514-0.9900 | 25.54 | 0.255 |
| 3 | 2.0 | 60 | 0.9070 (0.0260) | 0.8526-0.9625 | 19.95 | 0.290 |
| 4 | 3.0 | 60 | 0.8372 (0.0396) | 0.7571-0.9242 | 17.13 | 0.307 |

Across all 240 elastic rows: SSIM mean 0.9270, SD 0.0653, range
0.7571-0.9975. Every one of the 60 source ladders had Spearman -1.0; all 180
adjacent steps decreased, with 0 tied and 0 increasing. Mean adjacent drops were
-0.0210, -0.0643 and -0.0697 from severity 1 to 4. No elastic row was
near-identity at `SSIM >= 0.9999`, no row was degenerate, and no elastic image
duplicated another or a clean reference. Grouped bootstrap 95% intervals for
mean SSIM were 0.9917-0.9929, 0.9689-0.9735, 0.9010-0.9134 and 0.8273-0.8470 for
severities 1-4. Font and text-length slices were descriptively similar; all
included groups showed the same ordering.

**Controls and interpolation/boundary diagnostics.** Identity and zero-field
controls reproduced all 60 clean images exactly (SSIM 1.0, no pixels changed).
The 0.25 px minimal control had mean SSIM 0.99803 (minimum 0.99662), changed
17.1% of pixels, and touched 2.09% of pixels through the border policy. At
severity 4, mean outside-canvas/border-affected fraction was 0.0261, mean
bilinear-clipped-sample fraction 0.0046, and mean collapsed 8x8 block fraction
0.0426. These are reported as a confound, not subtracted from SSIM: the minimal
control is a direct near-identity resampling reference, while the border and
block diagnostics isolate different failure modes.

**Cross-family comparison.** Frozen blur/noise and M4A thin-line/arc features
were loaded and hash-verified; the elastic feature means were contrasted at the
same severity. At severity 4, mean absolute Cohen's d was 5.21 against
`gaussian_blur`, 6.49 against `gaussian_noise`, 5.67 against `thin_lines` and
4.28 against `arc_strokes`; 4-8 of the 8 features exceeded |d| = 0.8 in each
contrast. Pooled feature/target correlations still mix severity and are not the
honest summary: for example `laplacian_variance` had pooled Spearman 0.616 but
mean within-severity Spearman -0.075, and `edge_density` had pooled -0.095 but
within-severity -0.304. The eight features clearly change under elastic warp,
but the direction of the target relationship is severity-dependent; this is a
feature-pattern result, not proof of semantic distinctness or human difficulty.

**Gate.** The predeclared gate returned **CONDITIONAL PASS**. Reproducibility,
identity/zero-field exactness, overall target spread (SD 0.065, level-mean range
0.155), trend, interpolation/boundary bounds, feature distinctness and
duplicate-freedom passed. The single failed margin was per-level SD at severity
1: 0.0025 against the predeclared 0.005. That rung is close to the interpolation
floor (0.25 px control SSIM 0.998), so its between-source spread is small even
though the overall target is non-degenerate and every source trajectory is
monotonic. The verdict is therefore conditional, not an unqualified PASS, and
the failed criterion is retained in `gate_results.json` and the report.

**Conclusion.** The smooth elastic family is correctly implemented, deterministic,
exactly replayable, non-destructive, and produces an interpretable monotonic
SSIM response across all 60 frozen sources. Its features are strongly separated
from blur, noise and interference families at high severity. The lowest rung is
too close to identity to support a between-source variance claim, so a separate
benchmark should either drop that rung, use a larger minimum amplitude, or treat
the low end as interpolation-dominated. No model was trained and no CNN claim
follows from this pilot.

**Limitations.**

- One synthetic renderer, one 192x64 canvas, three fonts and 60 sources.
- Bilinear interpolation and edge padding necessarily change pixels; the
  minimal control and border fractions quantify but cannot eliminate the
  confound.
- SSIM is a full-reference structural proxy, not readability or solvability.
- The elastic family is one smooth field design; it does not cover rotation,
  perspective, compression or all geometric distortions.
- The lowest severity has low between-source variance; aggregate means at that
  rung should not be over-interpreted.
- Feature contrasts use descriptive Cohen's d with 60 sources and no
  significance claim.

**Next step.** Stop M5A here. Do not start M5B, retrain a model, train a CNN,
build a demo or create the final presentation. If a separate geometric benchmark
is later approved, predeclare a revised low-end ladder or drop the 0.5 px rung,
keep the source-grouped split, carry the field specification and controls, and
report the interpolation confound explicitly.

**Artifacts.** Git-ignored: `data/processed/geometric_distortion_pilot_v0.1.0/`
(480 images, manifest, sources, features, summary) and
`reports/geometric_distortion_pilot/` (`pilot.md`, level/uncertainty/trend/
control/group/feature tables, `feature_effect_sizes.csv`,
`feature_distinctness.csv`, `feature_target_correlations.csv`,
`gate_results.json`, `examples_elastic.png`). Regenerate with
`uv run --extra image captcha-geometric-pilot run --config
configs/geometric_distortion_pilot_v0_1_0.toml`; verify with
`captcha-geometric-pilot verify --config
configs/geometric_distortion_pilot_v0_1_0.toml`.

---

### EXP-0005 — Grouped occlusion/interference benchmark: zero-shot failure, conventional adaptation success

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Does the frozen M3 feature-only model transfer
  zero-shot to the held-out M4A occlusion/interference test sources, and does a
  controlled conventional model trained on M3 plus M4A training-source data
  improve held-out performance without fitting anything on test? This is a
  synthetic image-similarity benchmark, not a human-readability or CAPTCHA
  solvability study.
- **Code revision:** base `fd98fcf` plus the M4B implementation, committed as
  `feat: benchmark interference robustness`. The reported run was repeated
  after the commit so `run_metadata.json` records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; scikit-image 0.26.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/robustness_benchmark_v0_1_0.toml`; config sha256
  `da1b9f95bebcf359dc920498563579fc5b56ec395c092c17689212160c8326f4`;
  protocol hash
  `bf85b329a27020e244651d49d695fb04c2c29e9f3c8bd66d4a3913bf86792d53`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates; selection on
  validation RMSE. The frozen configs were not edited: dataset config sha256
  `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`;
  M3 benchmark config sha256
  `bea99c4b5ec94bc603c381c2d70366a1a0f411c67c94f61c05df64639e92ac57`;
  M4A pilot config sha256
  `2ef025220beb1b17aab4593e862fbbd11d2a6618eb8d3ed43bc532f6e3924917`.
- **Datasets:** frozen M3 `v0.1.0`, 540 rows / 60 sources, manifest sha256
  `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`,
  features sha256
  `d3715e7975cf41c41e49fa2338ca4e961a7a59981318be19149c78e89804d798`;
  M4A pilot `v0.1.0`, 540 rows / 60 sources (60 clean + 480 corrupted),
  manifest sha256
  `d08c1824ba81e3bc40a585b3047eec6c5f9aa2da0b0b9eb126057ab27fdcc92b`,
  features sha256
  `261cfe9a5e2c3910f08361c800f7bcd7252c86d7b56c32006c7970bfb63a1ad9`.
- **Split:** frozen source-grouped split, assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`;
  36/12/12 train/val/test sources for both datasets. All variants and severity
  levels of a source remain in its assigned split.
- **Merged data:** expanded train 612 rows / 36 sources = 324 M3 rows (36 clean
  + 288 blur/noise) + 288 M4A corrupted rows; expanded validation 204 rows /
  12 sources = 108 M3 + 96 M4A corrupted rows. M4A clean rows were excluded
  from expanded train/validation because all 60 are exact duplicates of the M3
  clean references. M4A test evaluation used 108 rows / 12 sources, including
  12 clean-reference rows. No duplicate sample IDs or image hashes in the
  expanded fit frames; all six isolation/leakage checks passed.
- **Model:** frozen M3 primary model, card sha256
  `26d89292a59964846acf3e8bab4b84a2bf41b17863608c620c35d75975ecca73`,
  artifact sha256
  `d727bdd48b1511c2e8e59a072bf43868c5dd997d71e6a884aa505c7b275fa085`.
  Expanded grid: OLS, 3 Ridge alphas, 4 random forests and 4 histogram gradient
  boosting configurations (12 candidates), fixed before test evaluation.
- **Runtime:** 8.07 s through the CLI, including frozen-image verification,
  M3/M4A target replay, 12-candidate fitting, grouped bootstrap, report and
  model persistence.

**Method.** Loaded and hash-checked both datasets and the frozen M3 bundle
through the existing verifier. Verified M3 with the full dataset verifier and
M4A with the pilot verifier (13 checks), then cross-checked that the two source
tables agree on IDs, splits, fonts and text lengths and that the 60 M4A clean
rows are exact image/feature/target duplicates of the M3 clean rows. Used the
eight existing image-only features and the unchanged SSIM target. Experiment A
evaluated the frozen M3 model and an M3-training-mean baseline on M4A test.
Experiment B fitted the 12-candidate grid on expanded train rows, selected on
expanded validation rows only, and evaluated the selected model plus an
expanded-training-mean baseline once on M4A test. No scaler, calibration,
feature-selection or hyperparameter decision used test rows. Source-grouped
bootstrap resampled the 12 test sources, not images.

**Experiment A — frozen M3 zero-shot on M4A test (108 rows / 12 sources).**

| Model | Slice | n | MAE | RMSE | R² | Mean signed residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| frozen M3 | overall | 108 | 0.1295 | 0.1612 | -1.0455 | -0.1294 |
| frozen M3 | clean | 12 | 0.0004 | 0.0012 | n/a | 0.0004 |
| frozen M3 | corrupted | 96 | 0.1456 | 0.1710 | -1.6031 | -0.1456 |
| frozen M3 | `thin_lines` | 48 | 0.1590 | 0.1876 | -1.6062 | -0.1590 |
| frozen M3 | `arc_strokes` | 48 | 0.1322 | 0.1527 | -1.7624 | -0.1322 |
| M3 training mean | overall | 108 | 0.1688 | 0.1948 | -1.9853 | 0.1589 |
| M3 training mean | corrupted | 96 | 0.1516 | 0.1759 | -1.7545 | 0.1404 |

The frozen model was near-perfect on clean references but under-predicted
increasingly with density: MAE 0.0496, 0.0990, 0.1690 and 0.2649 at severities
1-4. Its overall MAE advantage over the M3 training mean is therefore driven by
the clean slice. On corrupted rows the paired source-grouped MAE difference was
-0.0060 with 95% interval [-0.0307, 0.0129], which includes zero; on
`thin_lines` the frozen model was descriptively worse (+0.0141, interval
[-0.0124, 0.0388]). Negative R² means it was worse than a constant test-mean
predictor in squared-error terms. Zero-shot absolute calibration does not
transfer to this interference distribution.

**Experiment B — conventional adaptation on M4A test.**

| Model | Slice | n | MAE | RMSE | R² | Mean signed residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| expanded model | overall | 108 | 0.0131 | 0.0178 | 0.9752 | 0.0047 |
| expanded model | clean | 12 | 0.0024 | 0.0074 | n/a | 0.0024 |
| expanded model | corrupted | 96 | 0.0144 | 0.0187 | 0.9690 | 0.0050 |
| expanded model | `thin_lines` | 48 | 0.0162 | 0.0213 | 0.9664 | 0.0053 |
| expanded model | `arc_strokes` | 48 | 0.0126 | 0.0156 | 0.9712 | 0.0047 |
| expanded training mean | overall | 108 | 0.1261 | 0.1466 | -0.6912 | 0.0937 |
| expanded training mean | corrupted | 96 | 0.1117 | 0.1300 | -0.5044 | 0.0753 |

Validation selected `random_forest_depth_unlimited_leaf_1` (validation RMSE
0.0201; R² 0.9864). On test, the adapted model improved over frozen M3 by
-0.1164 MAE (95% interval [-0.1277, -0.1003]) and -0.1435 RMSE
([-0.1558, -0.1256]); the paired difference excluded zero at every severity and
both variants. The clean slice was slightly worse than frozen M3 (+0.0021 MAE,
interval [-0.0000, 0.0056]), but both were close to perfect and the interval
includes zero. Severity RMSE rose from 0.0143 to 0.0245 and mean signed residual
from +0.0027 to +0.0100, so the high-density end remains the hardest slice.
Test R² is unstable at low severity (0.445 at severity 1), where target variance
is small, despite low MAE.

On the original M3 blur/noise test rows, the expanded model remained close to
the frozen M3 reference: overall MAE/RMSE/R² 0.0158/0.0239/0.9849 versus
0.0157/0.0226/0.9865 for frozen M3. It did not sacrifice the original families
to gain interference performance.

**Target distribution.** Expanded training targets had mean 0.7587 (SD 0.1717,
range 0.3174-1.0); M4A test targets had mean 0.8524 (SD 0.1133, range
0.5532-1.0). M4A corrupted test mean was 0.8340 and expanded corrupted-only
train mean 0.8319; the lower full-expanded mean is caused by the severe M3 blur
and noise rows. The shift is disclosed and was not used to tune anything.

**Conclusion.** The frozen M3 model does not transfer useful absolute SSIM
calibration to M4A interference, despite near-perfect clean-reference
predictions; the failure is a target/covariate shift, not evidence about human
difficulty. A small conventional model trained on M3 plus training-source M4A
corrupted rows, selected on validation sources only, removes most of the error
and generalises descriptively across both variants and all four severities. On
this benchmark, conventional modelling is sufficient; a CNN is not justified by
these results. The 12-source intervals are preliminary and are not significance
claims.

**Limitations.**

- One synthetic renderer, one 192x64 canvas, three fonts and 60 sources.
- Only 12 held-out test sources; all intervals are wide and preliminary.
- M4A severity is nested stroke density, not an independent redraw at each
  level.
- The expanded model is trained on a mixture whose target distribution differs
  from M4A test; the shift is reported but not corrected.
- SSIM is a full-reference structural proxy, not readability or solvability.
- No cross-generator, real-image, human or security evaluation was performed.
- The clean-reference R² is undefined by construction and is not reported as a
  perfect fit.

**Next step.** Close M4B as a conventional-modelling result. Do not start a CNN,
dashboard or deployment on the basis of this benchmark. A future shift study
would need a new predeclared protocol, ideally with independent generators or
human labels, before model capacity is treated as the limiting factor.

**Artifacts.** Git-ignored: `reports/robustness_benchmark_v0_1_0/`
(`benchmark.md`, data accounting, isolation checks, zero-shot/adaptation
metrics, predictions, residuals, candidate validation, target distribution,
uncertainty, `run_metadata.json`) and
`models/robustness_benchmark_v0_1_0/` (`expanded_model.joblib` + card).
Regenerate with `uv run --extra image --extra ml captcha-robustness-benchmark
run --config configs/robustness_benchmark_v0_1_0.toml`; verify with
`captcha-robustness-benchmark verify --config
configs/robustness_benchmark_v0_1_0.toml`.

---

### EXP-0003 — Conventional image-quality models, leakage-safe comparison and held-out-font stress test

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `conventional-benchmark`
- **Hypothesis / question:** On the frozen 540-sample synthetic dataset, do
  conventional regressors on the eight image-only features improve materially
  over a training-mean baseline; does a nonlinear model beat OLS/Ridge on
  held-out sources; how do results vary by corruption family, severity and ink
  coverage; how much of the target is explained by corruption metadata alone;
  and does the selected model survive a genuinely held-out font? This is a
  synthetic image-similarity benchmark, not a human-readability or
  CAPTCHA-solvability study.
- **Code revision:** base `b9d6686` plus the Milestone 3 implementation; committed
  together in `feat: benchmark conventional image quality models`. The reported
  run was repeated after the commit so `run_metadata.json` and both model cards
  record a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; joblib 1.6.0; macOS 26 (arm64). No GPU, internet or Kaggle access was
  required.
- **Configuration:** `configs/benchmark_v0_1_0.toml`; benchmark config hash
  `e93d2660329a4f8b7e6b99739651250e2104ca9a26c76574db6db1e2e6bc8fa5`.
  The frozen dataset configs were not edited:
  `configs/dataset_v0_1_0.toml` sha256
  `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`;
  `configs/pilot_target_validation.toml` sha256
  `6e9bdf961f36bbf70f5806e41b53661e1577bf15e13eb833348632359b886715`.
- **Dataset:** `v0.1.0`; generation hash
  `820cce7a77b5e95761f033b75e8cb2b304b6ad6293f67b5bc27bcf0823a6cef9`;
  manifest sha256 `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`.
- **Split:** frozen source-grouped, font-stratified split; assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`.
  Sources: 36/12/12 train/val/test; samples: 324/108/108. Every source group is
  in exactly one split; no variant crosses a boundary.
- **Seeds:** benchmark seed `20261022`; model seeds derived per candidate from
  `(seed, "model", candidate)` and reduced to 32 bits for scikit-learn;
  bootstrap seeds derived from `(seed, "bootstrap", scope)`.
- **Runtime:** ~5.5 s in process, ~6.2 s wall through the CLI on this machine.
  The run fits 23 learned candidates on train, evaluates them on validation,
  then evaluates four family winners plus the naive baseline on test.

**Method.** Loaded the frozen dataset through `load_dataset`, enforced a 1:1
manifest/feature join, exact feature schema (`sample_id` + eight features in
documented order), finite values, target range, split membership, source-group
disjointness and the recorded split hash. Built a predeclared 23-candidate grid:
OLS (1), Ridge with six alphas (6), random forest with four depths x three leaf
sizes (12), and histogram gradient boosting with two learning rates x two leaf
counts (4). Each candidate was fitted on the training split only; scalers were
inside a pipeline fitted on those rows. Selection used validation RMSE only,
with ties broken by fewer inputs then name. The best candidate in each family
and the overall best candidate were then evaluated once on test. The candidate
table written to disk has no test columns.

Separately fitted diagnostic OLS models for severity+family, font+text length,
and image+metadata. These use generated labels, train-only category levels and
train-only scaling, and are not deployable image-only models. Post-selection
test slices were computed by corruption family, severity, train-quantile
ink-coverage quartile and font. Residuals and grouped bootstrap intervals
(resampling source groups, not variants) were reported. A separate held-out-font
experiment removed Times New Roman before fitting and validation, re-ran the
same grid on Arial + Courier New train/validation sources, selected on their
validation rows only, and evaluated once on all 20 Times New Roman sources (180
samples). The held-out font was chosen a priori because it is the only serif
face in the configuration; no performance information informed the choice.

**Metrics.**

Primary held-out test (108 samples, 12 source groups):

| Model | MAE | RMSE | R² | Notes |
| --- | ---: | ---: | ---: | --- |
| Naive train mean | 0.1664 | 0.1942 | -0.0000 | baseline |
| OLS, 8 features | 0.0302 | 0.0380 | 0.9618 | linear family winner |
| Ridge, alpha 0.01 | 0.0302 | 0.0379 | 0.9619 | ridge family winner |
| Random forest, depth 8, leaf 1 | 0.0157 | 0.0226 | 0.9865 | overall validation-selected |
| Histogram gradient boosting, lr 0.1, 15 leaves | 0.0174 | 0.0256 | 0.9826 | boosting family winner |

Selected-model train/validation/test RMSE: 0.0089 / 0.0196 / 0.0226. The
random forest's validation-selected configuration was nearly tied with unlimited
depth (validation RMSE 0.019635 vs 0.019665), so the exact forest winner is
unstable even though the family-level comparison is clear. HGB overfit strongly:
train RMSE 0.0021, validation 0.0216, test 0.0256.

Metadata diagnostics on test:

| Model | Inputs | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Severity + family | 3 | 0.0458 | 0.0579 | 0.9112 |
| Font + text length | 3 | 0.1660 | 0.1940 | 0.0023 |
| Image + metadata (OLS) | 11 | 0.0202 | 0.0242 | 0.9844 |

Gaps relative to the selected image-only model on test: metadata-only has
+0.0353 RMSE and -0.0752 R²; combined image+metadata has +0.0016 RMSE and
-0.0020 R². The combined diagnostic did not beat the selected random forest.

Test slices for the selected model (all counts in the generated tables):

| Slice | Samples | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Overall | 108 | 0.0157 | 0.0226 | 0.9865 |
| Corrupted only | 96 | 0.0176 | 0.0240 | 0.9804 |
| Blur | 48 | 0.0170 | 0.0255 | 0.9824 |
| Noise | 48 | 0.0182 | 0.0223 | 0.9721 |
| Clean | 12 | 0.0004 | 0.0012 | undefined |
| Coverage Q1 | 38 | 0.0140 | 0.0189 | 0.9882 |
| Coverage Q2 | 20 | 0.0106 | 0.0148 | 0.9902 |
| Coverage Q3 | 22 | 0.0202 | 0.0252 | 0.9630 |
| Coverage Q4 | 28 | 0.0180 | 0.0288 | 0.9456 |

Severity slices 1–4 had RMSE 0.0210 / 0.0312 / 0.0205 / 0.0214. Their R² values
(0.60 / 0.70 / 0.95 / 0.93) are lower at low severity because within-level target
variance is small, not because absolute error is larger; this is why both
absolute error and R² are reported. The clean gap is a constant-target subgroup,
so R² is undefined and is not filled in.

Grouped bootstrap (1,000 source-group replicates):

| Scope | Metric | Estimate | 95% interval | Groups |
| --- | --- | ---: | --- | ---: |
| Primary test | MAE | 0.0157 | [0.0116, 0.0209] | 12 |
| Primary test | RMSE | 0.0226 | [0.0157, 0.0309] | 12 |
| Primary test | R² | 0.9865 | [0.9757, 0.9934] | 12 |
| Held-out font | MAE | 0.0193 | [0.0157, 0.0233] | 20 |
| Held-out font | RMSE | 0.0271 | [0.0211, 0.0325] | 20 |
| Held-out font | R² | 0.9799 | [0.9712, 0.9876] | 20 |

Held-out font (Times New Roman; 20 sources, 180 samples; training fonts Arial +
Courier New, 24 training sources / 216 samples, 8 validation sources / 72
samples):

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Naive two-font train mean | 0.1628 | 0.1910 | -0.0006 |
| Validation-selected HGB, lr 0.05, 31 leaves | 0.0193 | 0.0271 | 0.9799 |
| Random-forest family winner | 0.0184 | 0.0248 | 0.9832 |
| OLS family winner | 0.0326 | 0.0398 | 0.9566 |
| Ridge family winner | 0.0326 | 0.0399 | 0.9564 |

The random-forest family winner was descriptively better on the held-out font
than the HGB configuration selected on the two-font validation split. The result
is reported as-is; the model was not re-selected on the held-out data.

**Observations.** Facts only:

1. **Conventional models beat the mean by a large margin.** Test RMSE fell from
   0.1942 to 0.0226 (88.4% relative), above the predeclared 5% materiality rule.
   This is expected because SSIM varies strongly with image structure, not
   evidence of a novel method.
2. **The feature/target relationship is nonlinear on this split.** The best
   nonlinear test RMSE (random forest, 0.0226) was 40.4% lower than the best
   linear one (ridge, 0.0379; OLS 0.0380). This is descriptive on 108 test
   samples.
3. **The exact forest winner is unstable.** The top two forest configurations
   differed by 0.00003 validation RMSE. The family-level result should be
   preferred over the exact depth/leaf choice.
4. **Tree models overfit.** HGB's training RMSE was 0.0021 versus validation
   0.0216; random forest training was 0.0089 versus validation 0.0196. The
   report exposes train/validation columns instead of reporting only test.
5. **Metadata explains most of the target.** Severity + family alone reached test
   R² 0.9112. Font + text length reached 0.0023, confirming no font/length
   shortcut in this split. Image features still improved on metadata, and the
   combined OLS diagnostic (0.9844) did not beat the image-only random forest
   (0.9865).
6. **Aggregate performance hides family and coverage variation.** Blur RMSE
   0.0255, noise 0.0223; coverage Q4 (highest ink) was hardest at 0.0288. Clean
   samples are near-perfect and make the pooled R² optimistic relative to
   corrupted-only performance (0.9804).
7. **The held-out-font stress test survives.** A model trained without Times New
   Roman reached R² 0.9799 on its 180 samples (selected HGB), and the random
   forest family winner reached 0.9832. This is not broad generalisation: only
   three fonts exist, and the held-out font shares the renderer and corruption
   design.
8. **Model verification passes.** Both persisted bundles passed card, schema,
   target, dataset-hash, split-hash, scikit-learn compatibility, artifact-hash,
   load and 8-feature smoke-test checks.
9. **Reproducibility.** The documented command was repeated; the report CSVs
   (excluding runtime metadata) and model predictions were compared. The
   benchmark is deterministic under the recorded seed; model artifact bytes are
   not claimed to be byte-identical across library versions.

**Conclusion.** On the frozen synthetic SSIM target, a conventional random
forest is materially better than the naive mean and descriptively better than
OLS/Ridge. The result is broadly stable across blur and noise but not uniform
across severity or coverage. Metadata alone explains 91.1% of held-out variance,
so the synthetic design is a strong confound; image-only features still explain
98.6%. The model survives a preliminary held-out-font stress test with R² 0.980.
For this target, a CNN is **not** justified: the conventional forest already
reconstructs almost all held-out variance and captures the nonlinearity, so a
CNN would mainly add cost and opacity. A CNN would be justified only on a harder,
predeclared shift where it beats this frozen feature-only baseline. A high R²
does not make the target scientifically interesting, and none of these numbers
measure human readability or CAPTCHA solvability.

**Limitations.**

- Three fonts, one canvas, two corruption families, 60 synthetic sources. Not a
  cross-generator or real-image evaluation.
- Test has 12 source groups; the grouped bootstrap intervals are wide and
  unstable. The exact forest configuration is nearly tied with alternatives.
- Validation and test share all three fonts in the primary benchmark. The
  held-out-font experiment removes one but leaves only two for training.
- Severity-1/2 R² is low because within-level variance is small; absolute errors
  are not correspondingly large.
- Clean samples have constant SSIM = 1, so R² is undefined there and pooled R²
  is not a corrupted-only estimate.
- Metadata diagnostics use generated labels and are not deployable inputs; the
  combined model is one reference, not a guaranteed upper bound.
- The model cards record scikit-learn's major/minor version; joblib artifacts
  are not safe for untrusted files and only locally generated bundles should be
  loaded.
- Text length is not stratified; validation contains no length-6 sources.

**Next step.** Do not start a CNN on this target. If Milestone 4 is requested,
first define a harder predeclared shift (for example a new corruption family or a
small documented real-image set) and require any CNN to beat the saved random
forest on the same grouped protocol. Otherwise retain the conventional model and
move to robustness analysis (M5).

**Artifacts.** Git-ignored: `reports/benchmark_v0_1_0/` (benchmark markdown,
candidate/comparison tables, predictions, residuals, strata, coverage,
correlations, held-out-font tables, uncertainty, `run_metadata.json`) and
`models/benchmark_v0_1_0/` (`primary_model.joblib` + card,
`heldout_font_model.joblib` + card). The benchmark CLI is
`uv run --extra image --extra ml captcha-benchmark run --config
configs/benchmark_v0_1_0.toml --dataset data/processed/captcha_quality_v0.1.0`;
verification is `captcha-benchmark verify-model`.

---

### EXP-0004 — Occlusion/interference pilot: reproducible and distinct, not a difficulty result

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Before adding a new corruption family to a grouped
  robustness benchmark, do two occlusion/interference variants produce a
  reproducible, non-degenerate SSIM target with an interpretable density
  response, and do their image features differ from the existing blur and noise
  families? This is a synthetic image-similarity pilot, not a human-readability,
  solvability or security study.
- **Code revision:** base `c781938` plus the M4A implementation, committed as
  `feat: pilot occlusion and interference robustness`. The reported run was
  repeated after the commit so `pilot_summary.json` records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; Pillow 12.3.0;
  scikit-image 0.26.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/robustness_pilot_occlusion_v0_1_0.toml`; config
  sha256 `2ef025220beb1b17…`; pilot generation hash
  `dc1a9b1557b3e8e49e86dc4e59beabf1d35d53c9f3a59300399a9f5a4e49e6f0`.
  The frozen configs were not edited: `configs/dataset_v0_1_0.toml` sha256
  `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`;
  `configs/benchmark_v0_1_0.toml` sha256
  `bea99c4b5ec94bc603c381c2d70366a1a0f411c67c94f61c05df64639e92ac57`.
- **Dataset:** frozen `v0.1.0`; generation hash
  `820cce7a77b5e95761f033b75e8cb2b304b6ad6293f67b5bc27bcf0823a6cef9`;
  manifest sha256 `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`.
  Pilot dataset: manifest sha256
  `d08c1824ba81e3bc40a585b3047eec6c5f9aa2da0b0b9eb126057ab27fdcc92b`,
  features sha256
  `261cfe9a5e2c3910f08361c800f7bcd7252c86d7b56c32006c7970bfb63a1ad9`,
  60 clean + 480 corrupted images.
- **Split:** frozen source-grouped, font-stratified split; assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`.
  36/12/12 train/val/test sources; 324/108/108 frozen samples. The pilot
  preserves the source split and adds no pilot sample to the frozen test set.
- **Seeds:** pilot seed `20261023`; per-source/per-variant stroke-pool seeds
  derived from `(seed, "interference", source_id, variant)`; bootstrap seeds
  derived from `(seed, "bootstrap", variant, severity)`. All randomness is
  explicit; no global RNG state.
- **Runtime:** 2.91 s wall through the CLI on this machine, including the
  frozen-reference audit, two full in-memory builds, dataset verification,
  480 provenance replays, bootstrap, report generation and contact sheets.

**Method.** Verified every frozen artifact hash, re-rendered all 60 clean
sources from the frozen dataset config, and compared source IDs, render seeds,
text hashes, split labels and pixels. Generated two variants with 1 px black
strokes on the existing 192x64 canvas: `thin_lines` through an ink pivot with
line angles in [-60, 60]°, and `arc_strokes` with sweep 120-240°, radius
0.20-0.55 of width and aspect 0.50-1.50. Severity levels were 1, 2, 4 and 7
strokes. A shared seven-stroke pool was planned per `(source, variant)`, and
severity `k` used its first `k` strokes; every severity was rendered from the
same clean reference, never from another corrupted image. Geometry, seed,
width, intensity, hashes, changed-pixel fractions and border contact were
recorded. SSIM used the frozen M1 conventions; PSNR was secondary. The pilot
trained no model and did not use the target or model predictions to choose
parameters. The predeclared gate covered reproducibility, non-degeneracy,
trend, feature distinctness from blur/noise, and duplicate freedom.

**Metrics.** Source-grouped bootstrap used 1000 replicates over 60 sources, not
variants.

| Variant | Severity | Strokes | SSIM mean | SD | 95% CI | PSNR mean (dB) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `thin_lines` | 1 | 1 | 0.943 | 0.020 | [0.938, 0.947] | 22.20 |
| `thin_lines` | 2 | 2 | 0.887 | 0.032 | [0.879, 0.895] | 18.84 |
| `thin_lines` | 3 | 4 | 0.785 | 0.041 | [0.775, 0.795] | 15.65 |
| `thin_lines` | 4 | 7 | 0.662 | 0.044 | [0.651, 0.673] | 13.15 |
| `arc_strokes` | 1 | 1 | 0.947 | 0.022 | [0.942, 0.952] | 22.47 |
| `arc_strokes` | 2 | 2 | 0.902 | 0.029 | [0.894, 0.909] | 19.38 |
| `arc_strokes` | 3 | 4 | 0.818 | 0.037 | [0.809, 0.827] | 16.33 |
| `arc_strokes` | 4 | 7 | 0.715 | 0.042 | [0.705, 0.725] | 13.96 |

Across all 480 corrupted samples: SSIM mean 0.832, SD 0.105, min 0.545, max
0.979. The clean baseline is exactly SSIM = 1.0 by construction. All 120
source-variant ladders had Spearman(severity, SSIM) = -1.0; 0/360 adjacent
steps increased and 0 were tied. Level-mean PSNR decreased monotonically with
severity; per-source PSNR exceptions were not part of the gate.

At the highest severity, Cohen's d against the frozen families at the same
severity:

| Variant | Versus | Mean \|d\| | Features with \|d\| ≥ 0.8 |
| --- | --- | ---: | ---: |
| `thin_lines` | gaussian_blur | 8.29 | 8/8 |
| `thin_lines` | gaussian_noise | 4.02 | 7/8 |
| `arc_strokes` | gaussian_blur | 7.06 | 8/8 |
| `arc_strokes` | gaussian_noise | 3.58 | 7/8 |

Predeclared gate:

| Criterion | Threshold | Observed | Verdict |
| --- | --- | --- | --- |
| Reproducibility | Two builds identical; clean re-renders exact; verification passes | identical; 60/60 exact; 13/13 checks | PASS |
| Non-degeneracy | SD ≥ 0.02; level-mean range ≥ 0.10; level SD ≥ 0.005 | SD 0.105; ranges 0.281/0.233; min level SD 0.0197 | PASS |
| Interpretable trend | Median Spearman ≤ -0.80; increasing fraction ≤ 0.15 | -1.000; 0.000 | PASS |
| Feature distinctness | Mean \|d\| ≥ 0.50 and ≥2 features at \|d\| ≥ 0.80 vs both families | 3.58-8.29; 7-8 features | PASS |
| Duplicate-free | Duplicate fraction ≤ 0.0 | 0.000000 | PASS |

**Observations.** The family is non-degenerate at every level and the severity
response is strictly monotone for every source and variant under the nested
design. Marginal SSIM drop per added stroke declines with density (`thin_lines`
-0.057 to -0.041; `arc_strokes` -0.046 to -0.034), so the high end is mildly
saturated but not collapsed. Every corrupted image touches the canvas border by
construction; changed pixels cover 0.7-5.8% of the canvas and 0.6-4.8% of the
clean ink. Fixed black intensity means a stroke that overlaps existing glyph
cores can change fewer pixels, which is a real confound and a potential source
of flat steps. Feature/target associations also differ from blur: for example,
pooled Pearson between Laplacian variance and SSIM was +0.883 for blur but
-0.964 for `thin_lines` and -0.987 for `arc_strokes`; the within-severity signs
are reported in the tables, so the pooled values are not treated as causal.

**Conclusion.** The pilot supports proceeding to a separate grouped robustness
benchmark for this family: it is deterministic, exactly replayable from recorded
geometry, non-degenerate, interpretable in density, and clearly distinct from
blur/noise in the eight-feature space. It does **not** show that occlusion is
objectively harder or more human-solvable; those claims require a different
study. The pilot also does not establish cross-generator or real-image
generalisation.

**Limitations.**

- One synthetic renderer, one 192x64 canvas, three fonts and 60 sources. The
  feature contrast may change with other canvases, fonts or text lengths.
- Density is the only varied parameter. Width, intensity, colour and coverage
  are fixed; a full benchmark should predeclare whether they vary.
- The nested pool makes the severity response a dose of the same early strokes.
  This improves interpretability but is not an independent sample of geometry
  at each severity.
- Border contact is 100% by construction and is not text clipping, but it does
  mean edge effects are not varied.
- Fixed black strokes can disappear into glyph cores; changed-pixel fraction is
  a better manipulation check than nominal stroke count.
- SSIM is a full-reference structural proxy. It is not human readability or
  CAPTCHA solvability.
- No model was trained, so the pilot says nothing about model performance on
  this family.

**Next step.** If M4 proceeds, run a separate grouped robustness benchmark with
a predeclared protocol, keep these samples out of the frozen M3 test set, and
compare any model against the saved feature-only baseline. Do not treat the
pilot's SSIM levels as a difficulty ordering.

**Artifacts.** Git-ignored: `data/processed/occlusion_interference_pilot_v0.1.0/`
(manifest, sources, features, summary, 540 PNGs) and `reports/robustness_pilot/`
(`pilot.md`, level/uncertainty/trend/feature/saturation tables,
`gate_results.json`, contact sheets). Regenerate with
`uv run --extra image captcha-robustness-pilot run --config
configs/robustness_pilot_occlusion_v0_1_0.toml`.

---

### EXP-0002 — Reproducible synthetic dataset, grouped splits and feature diagnostic

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `dataset-generation`
- **Hypothesis / question:** Can the two validated corruption families be turned
  into a deterministic, versioned dataset with stable IDs, one clean reference
  per sample, grouped splits fixed before generation, and typed image-only
  features — and do those features make the SSIM target trivially predictable on
  held-out groups? This is a dataset-validity and leakage experiment, not a
  human-solvability study.
- **Code revision:** base `92ecf33` plus the Milestone 2 implementation; this
  entry and that implementation are committed together in
  `feat: add reproducible dataset and image features`. Working tree: clean at
  run apart from the files being committed.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; Pillow 12.3.0;
  scikit-image 0.26.0; SciPy 1.18.1; macOS 26 (arm64). No GPU, no internet and
  no Kaggle access were required.
- **Configurations:**
  - `configs/dataset_pilot.toml`, sha256
    `64fb26f78cffecdb9dc8543c600a0bae8fc1c0a66e85adb72bcdb96341e73512`;
    generation hash `f9e66dc9ca109a7b1cf8a47e98f19c0214cf723933047583edbda0faec198b1d`.
  - `configs/dataset_v0_1_0.toml`, sha256
    `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`;
    generation hash `820cce7a77b5e95761f033b75e8cb2b304b6ad6293f67b5bc27bcf0823a6cef9`.
  - The M1 config `configs/pilot_target_validation.toml` is unchanged (sha256
    `6e9bdf961f36bbf70f5806e41b53661e1577bf15e13eb833348632359b886715`).
- **Datasets:** `v0.1.0-pilot` (9 sources, 81 samples; manifest sha256
  `2a5984c3f12efa14caf853c171dbe16a31ac8f399b8d741845be181b2f501f44`) and
  `v0.1.0` (60 sources, 540 samples; manifest sha256
  `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`).
- **Split:** derived from the generation hash plus split seed `20261021`;
  assignment sha256 for `v0.1.0` is
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`.
  Groups: 60 sources (36/12/12 train/val/test); samples: 540 (324/108/108).
- **Seeds:** dataset seed `20261020`; split seed `20261021`. Per-source text,
  render and corruption seeds are derived independently from the dataset seed,
  font, source index, family and severity. No global RNG state.
- **Runtime:** pilot generation ~0.6 s; reference generation ~1.2 s in-process
  (1.4 s wall including CLI startup); full verification 0.9 s; diagnostics
  0.2 s. Disk usage: 503 KiB (pilot) and 3.4 MiB (reference).

**Method.** Loaded a strict TOML configuration, planned one clean source per
`(font, index)` with text length drawn from the configured distribution, and
assigned font-stratified grouped splits **before** rendering. Rendered a clean
192x64 grayscale reference per source, then every non-zero severity of both
families (blur radii `[1, 2, 3.5, 5]` px; noise sigma `[6, 12, 20, 32]`), with
an independent seeded noise realisation per variant. Computed SSIM (primary)
and PSNR (secondary) from the exact reference/variant pair. Extracted eight
documented image-only features from every degraded image. Wrote a public
manifest, a separate internal `ground_truth.csv`, `sources.csv`, `features.csv`
and `dataset_summary.json` through a staging directory, then renamed it into
place.

Verification recomputed every target from disk, checked raw-pixel hashes, path
containment, reference↔source matching, conditional PSNR, split isolation and
feature-table integrity. Two additional independent runs were compared
byte-for-byte. Diagnostics fitted only on `train`: a naive mean, three metadata
baselines, one OLS per feature, and an all-feature OLS; one-feature selection
used validation R² and test was reported once. The pilot dataset was run first,
then the larger reference dataset.

**Metrics.**

| Metric | Value | Notes |
| --- | --- | --- |
| Sources / samples | 60 / 540 | 1 clean + 4 blur + 4 noise per source |
| Split sources (train/val/test) | 36 / 12 / 12 | all three fonts in every split (12/4/4) |
| Split samples (train/val/test) | 324 / 108 / 108 | family and severity composition proportional |
| Feature rows x features | 540 x 8 | finite; stable names and order |
| Verification checks | 11 / 11 PASS | includes 540 recomputed target pairs |
| Leakage checks | 4 / 4 PASS | target exclusion, group isolation, disjointness, train-only fitting |
| Best single feature (val/test R²) | `histogram_entropy` 0.680 / 0.644 | selected on validation only |
| All-feature OLS (val/test R²) | 0.968 / 0.962 | standardised on train |
| Metadata severity + family (val/test R²) | 0.924 / 0.911 | oracle label baseline, not a model input |
| Metadata font (val/test R²) | -0.001 / 0.002 | no font confounding in this split |
| Metadata text length (val/test R²) | 0.000 / 0.000 | no length confounding in this split |
| Reproducibility | byte-identical | two independent runs; manifest and features hashes equal |

Mean SSIM by severity on `v0.1.0` (n = 60 per cell; clean = 1.000):

| Family | s1 | s2 | s3 | s4 |
| --- | --- | --- | --- | --- |
| Gaussian blur | 0.897 (0.013) | 0.675 (0.024) | 0.475 (0.034) | 0.411 (0.053) |
| Gaussian noise | 0.871 (0.035) | 0.742 (0.045) | 0.635 (0.043) | 0.540 (0.042) |

The pilot dataset gave the same ordering: blur 0.901/0.672/0.461/0.395,
noise 0.928/0.804/0.680/0.572.

**Observations.** Facts only:

1. **Determinism held end-to-end.** Repeated generation produced identical
   manifests (`a6a2cb8654eb…`), feature tables (`d3715e7975cf…`), source and
   ground-truth tables, and 540 identical image pixel hashes. No timestamps are
   written into any artifact.
2. **Split allocation needed a floor repair.** With three sources per font and
   0.6/0.2/0.2 ratios, plain largest-remainder allocation gives `2/1/0`, silently
   excluding a split from a font. The implemented rule reserves one group per
   split and takes the remainder from the largest split, producing `1/1/1` and
   satisfying the predeclared font-stratification requirement.
3. **Text-length distribution is not balanced per split.** Train has 11/14/11
   sources at lengths 4/5/6, validation 5/7/0, test 2/6/4. This is reported, not
   corrected: only font was stratified by design.
4. **The target is strongly predictable from simple statistics.** One feature
   (`histogram_entropy`) reaches validation/test R² 0.680/0.644, below the
   predeclared 0.70 "strong confound" threshold, so the single-feature
   classification is **no single-feature triviality**. However, the eight-feature
   linear baseline reaches **0.968/0.962**, classified
   `feature_only_near_solved`.
5. **Family-conditional relationships reverse sign.** `laplacian_variance` has
   overall Pearson r = 0.066, but r = 0.883 within blur and r = -0.861 within
   noise. `ink_fraction` is r = -0.930 within blur and r = 0.182 within noise.
   Aggregate correlations are misleading here; the report includes within-family
   and within-severity tables.
6. **8-bit clipping still compresses heavy noise.** At nominal sigma 20
   (severity 3), mean `intensity_std` fell from 75.29 on the clean renders to
   73.14 on the noisy variants (58 of 60 sources), while the Immerkær estimate
   rose from 3.77 to 15.84 on every source. Intensity variance is therefore not a
   monotone noise indicator.
7. **High blur removes all Canny edges for some sources.**
   `edge_density` reaches exactly 0 on strongly blurred images, which is a valid
   but discrete response; the feature table records it rather than hiding it.
8. **The pilot is too small to override the larger result.** With three training
   sources, pilot entropy R² is 0.822/0.801 (`strong_confound`) and the
   all-feature baseline 0.985/0.974. The larger run is the reference for M3.

**Conclusion.** The pipeline produces a valid, versioned, byte-reproducible
synthetic dataset with one clean reference per sample, correct conditional PSNR
semantics, stable IDs, and grouped splits that never cross sources. The target
and features are leak-free by construction and by programmatic checks. The
diagnostic supports a specific decision: the SSIM target is **not** trivially
predictable from any single feature, but it is almost completely predictable
from the eight simple statistics together. A CNN is therefore not justified by
target predictability alone; M3 baselines must beat the feature-only linear
baseline under grouped evaluation, and any claimed gain must not be explained by
the same statistics. These findings concern synthetic image similarity only and
say nothing about human readability or CAPTCHA solvability.

**Limitations.**

- Three fonts, one canvas size, flat backgrounds only, two corruption families,
  and 60 synthetic sources. No cross-generator or cross-font claim is made; a
  held-out font/configuration split is reserved for later.
- The split is a random grouped split with 12 held-out source groups per split;
  R² values are noisy and are diagnostics, not model-performance estimates.
- Text length is sampled from a distribution but not stratified; validation has
  no length-6 sources by chance.
- PSNR remains secondary and is only interpreted for the noise family (M1).
  MS-SSIM is still deferred.
- The noise family is compressed by 8-bit clipping, so nominal sigma is not the
  realised noise standard deviation.
- Features describe the degraded image only; they cannot capture
  reference-relative structure, which is why the all-feature linear baseline
  still leaves ~4% of held-out variance.
- System fonts are recorded by path and logical ID, not by binary content hash.
- Single-machine run; PNG bytes may vary across Pillow/zlib versions, so
  reproducibility is asserted on decoded pixel hashes and CSV contents, not on
  PNG file bytes.

**Next step.** Milestone 3: grouped conventional baselines (naive, ridge/linear,
tree ensemble) trained on `features.csv`, evaluated on the frozen split, with the
feature-only OLS baseline as the floor. Add coverage-stratified reporting and a
held-out-font check. No CNN or dashboard work before that comparison is
recorded.

**Artifacts.** Git-ignored: `data/processed/captcha_quality_v0.1.0-pilot/`,
`data/processed/captcha_quality_v0.1.0/`, `data/processed/repro_a/`,
`data/processed/repro_b/`, `reports/captcha_quality_v0.1.0-pilot/` and
`reports/captcha_quality_v0.1.0/` (diagnostics markdown plus four CSV tables).

---

### EXP-0001 — Reference-metric validation pilot (SSIM, PSNR, blur, noise)

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `target-validation`
- **Hypothesis / question:** Do SSIM and PSNR respond predictably and consistently
  to increasing severity of Gaussian blur and additive Gaussian noise, with
  adjacent severities distinguishable and the response stable across fonts and
  seeds? This is a metric-validation experiment, not a human-solvability study.
- **Code revision:** base `d0e5256` plus the Milestone 1 implementation; this
  entry and that implementation are committed together in
  `feat: validate synthetic image quality targets`. Working tree: clean at run.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; Pillow 12.3.0;
  scikit-image 0.26.0; SciPy 1.18.1; macOS 26 (arm64). OpenCV was installed via
  the `image` extra but is not used by this pilot.
- **Configuration:** `configs/pilot_target_validation.toml`, sha256
  `6e9bdf961f36bbf70f5806e41b53661e1577bf15e13eb833348632359b886715`.
  Thresholds were fixed in the config and in `docs/METHODOLOGY.md` §3.4 **before**
  the recorded run and were not edited afterwards.
- **Dataset:** synthetic, generated in-process; no stored image collection. The
  Git-ignored `reports/pilot/` directory holds tables and contact sheets only.
- **Split:** not applicable (descriptive validation; no model is trained).
- **Seeds:** run seed `20261020`; repeat seeds `[0, 1, 2]`, each an independent
  render-jitter and noise realisation.
- **Runtime:** 0.82 s wall-clock for the full run (324 unique image contents).

**Method.** Rendered 5-character strings (charset `23456789ABCDEFGHJKMNPQRSTUVWXYZ`,
ambiguous glyphs removed) at 192x64 px, using 4x supersampling with LANCZOS
downsampling and seeded per-character vertical jitter. Three system fonts
(Arial, Courier New, Times New Roman) x 4 texts each x 3 repeat seeds = **36 clean
reference images** (12 unique strings). Each clean image was scored against:

- itself (36 clean self-comparisons), and
- each severity of two families, reusing the byte-identical reference:
  Gaussian blur radius `[0, 1, 2, 3.5, 5]` px and additive Gaussian noise sigma
  `[0, 6, 12, 20, 32]` on the 0-255 scale, with an independent noise draw per
  `(trial, severity)`.

That yields **360 corrupted-pair rows** at 5 severity levels (severity 0 is the
identity baseline), plus 36 sanity rows. SSIM used `data_range=255`,
`win_size=11`, Gaussian weights (`sigma=1.5`), no sample-covariance correction.
PSNR used `data_range=255`; identical images return `math.inf` explicitly.
MS-SSIM was not evaluated (see limitations).

**Metric summaries.**

Level means (n = 36 per cell; SD in parentheses):

| Family | Metric | s0 | s1 | s2 | s3 | s4 |
| --- | --- | --- | --- | --- | --- | --- |
| blur | SSIM | 1.000 | 0.913 (0.009) | 0.712 (0.021) | 0.530 (0.025) | 0.469 (0.043) |
| noise | SSIM | 1.000 | 0.918 (0.007) | 0.781 (0.019) | 0.645 (0.027) | 0.529 (0.033) |
| blur | PSNR (dB) | +inf | 20.663 (0.670) | 15.577 (0.428) | 12.963 (0.449) | 11.955 (0.661) |
| noise | PSNR (dB) | +inf | 35.142 (0.070) | 29.219 (0.077) | 24.813 (0.107) | 20.760 (0.079) |

Spearman rho (severity vs. score, n = 180): blur/SSIM **-0.973**, blur/PSNR
**-0.946**, noise/SSIM **-0.983**, noise/PSNR **-0.968**.

Adjacent-level paired differences and effect sizes (same clean reference; `d` is
paired Cohen's d):

| Family / metric | s0->s1 | s1->s2 | s2->s3 | s3->s4 |
| --- | --- | --- | --- | --- |
| blur / SSIM | -0.087 (d=-9.89) | -0.201 (d=-14.89) | -0.183 (d=-5.59) | -0.061 (d=-2.80) |
| noise / SSIM | -0.082 (d=-11.77) | -0.137 (d=-11.28) | -0.136 (d=-13.03) | -0.116 (d=-15.27) |
| blur / PSNR | not evaluable (+inf baseline) | -5.086 dB (d=-15.44) | -2.614 dB (d=-4.16) | -1.008 dB (d=-3.67) |
| noise / PSNR | not evaluable (+inf baseline) | -5.923 dB (d=-57.51) | -4.406 dB (d=-29.27) | -4.053 dB (d=-30.75) |

Trial-level violation rate was **0.000** for every evaluable adjacent pair in both
families and both metrics. Eta squared (severity share of finite-score variance):
blur/SSIM 0.987, noise/SSIM 0.986, blur/PSNR 0.974, noise/PSNR 1.000. Level-mean
ranges: 0.531 SSIM, 0.471 SSIM, 8.708 dB, 14.382 dB respectively.

Per-font rho ranged over [-0.984, -0.979] (SSIM) and [-0.968, -0.964] (PSNR);
per-seed rho over [-0.984, -0.983] (noise/SSIM), [-0.974, -0.972] (blur/SSIM),
and [-0.968, -0.946] (PSNR). Every font and seed group had strictly decreasing
level means.

Strongest within-level correlation with ink coverage (`|r|`, signed value and
level): blur/SSIM 0.874 (-0.874 at s4), noise/SSIM 0.735 (+0.735 at s4),
blur/PSNR **0.975** (-0.975 at s4), noise/PSNR 0.139 (+0.139 at s4).

**Gate results** (predeclared rule, `docs/METHODOLOGY.md` §3.4):

| Family | Metric | Verdict | Failing criterion |
| --- | --- | --- | --- |
| gaussian_blur | ssim | **PASS** | — |
| gaussian_blur | psnr | **FAIL** | `non_degeneracy` (`|r_ink|` 0.975 > 0.95) |
| gaussian_noise | ssim | **PASS** | — |
| gaussian_noise | psnr | **PASS** | — |
| any | ms_ssim | **DEFERRED** | no lightweight implementation available without PyTorch |

All four combinations passed direction, aggregate trend, adjacent separation,
violation rate, cross-font/cross-seed stability, and clean-reference sanity
(36/36 exact SSIM=1.0, PSNR=+inf, byte-identical; severity-0 identity rows exact).

**Observations.** The response is smooth and near-monotone for all four
combinations. No trial-level reversals occurred at these ladder spacings. The
clean baseline is exactly perfect for both metrics. Two behaviours are worth
recording:

1. **PSNR under blur is largely an ink-coverage measure at fixed severity.**
   Within the highest blur level, PSNR correlates -0.975 with the clean image's
   ink fraction: images with more ink (e.g. Arial, 0.150 coverage) lose more
   edge area than sparse ones (Courier New, 0.089). SSIM shows the same direction
   at a lower magnitude (-0.874).
2. **The noise ladder is compressed by 8-bit clipping.** Realised noise standard
   deviations were 4.10, 8.08, 13.39, and 21.31 for nominal sigma 6, 12, 20, 32;
   44-46% of pixels hit a rail (mostly the white background). This is why
   measured PSNR exceeds the naive `20 log10(255/sigma)` values, and it matters
   for interpreting absolute severity.

No threshold was changed after inspecting these results.

**Conclusion.** SSIM is a suitable candidate primary target for both families
under the tested conditions. PSNR is suitable for additive noise but not for
blur as a standalone target, because at fixed blur severity it mostly restates
how much ink the image contains. MS-SSIM remains unevaluated. These findings
concern synthetic image similarity only; they say nothing about human
readability or CAPTCHA solvability.

**Limitations.**

- One canvas size, one text length, three fonts, two corruption families, and
  36 source images. Results may not transfer to other renderer settings.
- SSIM and PSNR are full-reference proxies for degradation, not perceptual or
  human-legibility measures.
- Geometric warps, occlusion, and compression are not tested; PSNR's geometric
  sensitivity is therefore not characterised here.
- PSNR's `+inf` clean level is excluded from finite aggregates by design, so the
  s0->s1 PSNR separation could not be quantified with the paired statistic
  (the level means still show the drop).
- Within-level ink coverage explains a substantial share of variance at high
  severity even for SSIM (0.874 at the strongest blur), so font/coverage
  confounding must be tracked in later modelling.
- MS-SSIM deferred: scikit-image provides no MS-SSIM and the maintained
  implementations require PyTorch, which is not installed in this milestone.
- Single-machine run on macOS with system fonts; font binaries and versions are
  recorded by path, not by content hash.

**Next step.** Proceed to Milestone 2 with SSIM as the primary reference target
and PSNR as a secondary diagnostic (noise-family comparisons only). When the
feature pipeline exists, re-run the full non-degeneracy check including
trivial-feature predictability, and add coverage-stratified reporting to track
the ink confound. Revisit MS-SSIM when PyTorch is installed for the CNN work.

**Artifacts.** Git-ignored: `reports/pilot/summary.md`, `gate_results.json`,
`trials.csv`, `sanity.csv`, `levels.csv`, `separation.csv`, `stability` tables,
`ink.csv`, `variance.csv`, and `examples_gaussian_{blur,noise}.png`.

---

## Milestone 0 note

Milestone 0 ran no experiments; it established tooling and documentation only.
