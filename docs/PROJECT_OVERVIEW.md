# Project Overview

**Status:** Milestones 0–3 complete, plus the M4A occlusion/interference
feasibility pilot, the M4B grouped robustness benchmark, the M5A
geometric-distortion feasibility pilot, the M5B grouped geometric-robustness
benchmark, the M5C retention-aware selection and calibration study and the M5D
retention-aware training-objective study. A reproducible dataset and
image-feature pipeline, a leakage-safe conventional-ML benchmark, source-isolated
zero-shot/adaptation evaluation, a replayable elastic-warp pilot, a grouped
geometric-robustness benchmark, a retention-aware selection study and a
retention-aware training-objective study exist. The CNN comparison, full M5
error taxonomy and dashboard are planned.
**Last updated:** 2026-10-20

## 1. Purpose

Build a reproducible testbed that generates CAPTCHA-style text images, applies
controlled distortions, measures image quality against a clean reference, and
models the relationship between distortion parameters, cheap image features,
and measured degradation.

The project is a measurement and reproducibility exercise, not a claim to
predict human solvability or to rank real CAPTCHA products.

## 2. System pipeline

```text
[1] Configuration
        |
        v
[2] Text rendering  ------------> clean reference image + metadata
        |
        v
[3] Distortion engine  ---------> degraded image + applied transform record
        |                                  |
        |                                  +--> [4] Reference-based quality targets
        v                                  |
[5] Feature extraction <-------------------+
        |
        v
[6] Dataset assembly + manifest + split assignment
        |
        v
[7] Modelling (conventional baselines, then CNN comparison)
        |
        v
[8] Evaluation, robustness slices, error analysis
        |
        v
[9] Dashboard and reports
```

### Stage 1 — Configuration

- **Input:** human-authored TOML under `configs/` (`pilot_target_validation.toml`, `dataset_pilot.toml`, `dataset_v0_1_0.toml`).
- **Output:** a validated, immutable run configuration with a hash.
- **Contents:** Python/package versions, seed, character set, text
  length distribution, fonts, canvas size, colour model, corruption grid
  (family × severity × probability), sample counts, split ratios, metric list.
- **Rule:** no hard-coded experimental parameters in source; configs are
  versioned and referenced by the experiment log.

### Stage 2 — Synthetic text rendering

- **Input:** config + seed.
- **Output:** a clean 8-bit grayscale image plus a generation record
  (`source_id`, font, text length, seed, background level, content hash).
- **Implemented:** Pillow rendering with a small, explicitly resolved font set
  (no silent substitution), a configurable character set with ambiguous glyphs
  removed, per-character vertical jitter, optional spacing/placement jitter,
  and a configurable flat background policy (`white` or one seeded gray level
  per source). Text length is drawn from a configured distribution.
- **Deliberately out of scope:** rotation, texture backgrounds, colours.
- **Purpose:** provides both training images and the **reference** required by
  full-reference quality metrics.

### Stage 3 — Distortion engine

- **Input:** clean image + corruption specification.
- **Output:** degraded image + exact applied parameters (`family`, `severity`,
  numeric parameters, random state).
- **Implemented families:** Gaussian blur and additive Gaussian noise, each
  with an ordered severity ladder. Severity 0 is the identity condition and is
  represented once per source as the shared clean sample rather than duplicated
  per family.
- **M4A/M4B only:** two occlusion/visual-interference variants (thin lines and
  curved arcs) are implemented in `interference.py`, validated in a separate
  M4A pilot dataset, and used unchanged by the M4B grouped benchmark. They are
  deliberately not part of the frozen `v0.1.0` dataset.
- **M5A only:** a bounded smooth elastic warp with a 5x13 control grid, 2.0 px
  Gaussian field smoothing, bilinear resampling and edge padding is implemented
  in `geometric.py`. It is validated by the isolated M5A pilot and used unchanged
  by the M5B grouped benchmark; it is not in the frozen dataset.
- **Deferred for the frozen dataset:** contrast/brightness shifts, geometric
  warps and compression artefacts.
- **Rule:** each sample records its family, severity, numeric parameters, and
  seed so effects are attributable and reproducible.

### Stage 4 — Reference-based quality targets

- **Input:** clean reference and degraded image pair.
- **Output:** scalar quality targets per pair (for example SSIM, MS-SSIM, PSNR).
- **Primary target:** SSIM against the clean rendering of the same text.
  PSNR is a secondary diagnostic for the noise family only. MS-SSIM remains
  deferred because no lightweight implementation is installed.
- **Why reference-based:** it is deterministic, reproducible, and available by
  construction in a synthetic corpus. It is a proxy for image degradation, not
  for human difficulty.
- **Validation gate (M1, completed):** check sensitivity, monotonicity across severity
  levels, and stability across fonts/seeds before scaling the dataset.
- **M1 outcome:** SSIM passed this gate for Gaussian blur and additive noise;
  PSNR passed for noise but not for blur (ink-coverage confound). See
  `docs/EXPERIMENT_LOG.md`, `EXP-0001`.

### Stage 5 — Feature extraction

- **Input:** the degraded image only. The extractor never sees the clean
  reference, the SSIM/PSNR target, or generation metadata.
- **Output:** eight typed, documented features in a fixed order: mean and
  standard deviation of intensity, RMS contrast, Otsu ink coverage, Laplacian
  variance, Canny edge density, 256-bin histogram entropy, and Immerkær's noise
  estimate. Exactly uniform images are handled explicitly (all structural
  features zero).
- **Purpose:** support conventional ML baselines, expose trivial predictors
  before a model is trained, and provide interpretable comparisons against CNN
  embeddings later.
- **Leakage rule:** SSIM, PSNR, severity, corruption type, font, and text length
  are never inputs. See `METHODOLOGY.md` §4.

### Stage 6 — Dataset assembly

- **Output:** versioned image directories, a public CSV manifest, a separate
  internal ground-truth file, a feature table, and a JSON summary. Generation
  is staged and atomically renamed into place.
- **Split rule:** one group per clean source (`source_id`), assigned before
  generation. Font-stratified shuffling with an explicit seed; no source ever
  crosses train/val/test. Ratios, minimum sizes, and feasibility are validated
  before pixels are produced.
- **Metadata policy:** the public manifest records corruption type, severity,
  numeric parameters, and seeds, but **not** raw text (only `text_sha256`).
  Raw text lives in `ground_truth.csv`, which is internal. Corruption
  parameters are not model inputs in the quality regression task.
- **Verification:** a `verify` command recomputes every SSIM/PSNR target from
  the exact reference/variant pair, checks content hashes, path containment,
  reference matching, split isolation, and feature-table integrity.

### Stage 7 — Modelling

**Implemented (M3): conventional image-only regression.** OLS, Ridge, random
forest and histogram gradient boosting predict SSIM from the eight handcrafted
features. Selection is validation-only; the test split is evaluated once for the
selected model and each family winner. A separate held-out-font experiment
removes one font from fitting and validation. Model artifacts carry feature
schema, target, dataset/split hashes and selection metadata, and a verification
command rejects mismatches.

**Implemented (M4B): grouped interference benchmark.** The frozen M3 model is
evaluated zero-shot on the 12 M4A test sources. An expanded conventional model
is fitted on M3 rows plus M4A corrupted rows from training sources only,
selected on validation sources, and evaluated once on test. M4A clean rows are
excluded from fitting because they duplicate the M3 clean references. The same
eight image-only features and unchanged SSIM target are used; no test row
informs preprocessing, calibration or selection. The frozen model reached test
R² -1.045, while the validation-selected random forest reached 0.975.

**Planned (M4): CNN comparison.** A small from-scratch CNN and a documented
transfer-learning variant, under the same frozen split and held-out-font
protocol. The M3 result (test R² 0.986) and the M4B adaptation result (0.975)
are the baselines a CNN must beat; the evidence does not justify a CNN on the
current synthetic target or the M4B interference split.

Both tracks use the same metrics and per-slice breakdowns. Simpler models are
the default; a CNN must justify its cost out-of-sample.

### Stage 8 — Evaluation and robustness analysis

**Implemented for the M3 conventional benchmark:**

- Regression: MAE, RMSE, R², train/validation/test reporting, residual slices by
  family and severity.
- Corruption/coverage slices, with coverage quartiles fixed from training
  quantiles.
- Source-group bootstrap intervals (not variant-level), reported descriptively.
- Held-out-font stress test and metadata-only/combined diagnostics.

**Implemented for M4A (feasibility pilot, not a benchmark):**

- Two isolated occlusion/interference variants with deterministic stroke
  geometry, exact clean-reference re-rendering, geometry replay, grouped
  bootstrap, feature contrast against blur/noise, and a predeclared go/no-go
  gate. No model is trained.

**Implemented for M4B (grouped robustness benchmark):**

- Zero-shot frozen-M3 evaluation, expanded-data fitting with validation-only
  selection, a naive target-mean baseline, source-grouped bootstrap intervals,
  variant/severity/residual slices, target-distribution accounting, and
  model/report persistence with replay verification. Twelve test sources are
  treated as preliminary uncertainty, not significance evidence.

**Implemented for M5A (geometric-distortion feasibility pilot, not a benchmark):**

- One seeded smooth elastic field per source, scaled to a predeclared
  `[0.5, 1.0, 2.0, 3.0]` px maximum-displacement ladder plus identity,
  zero-field and 0.25 px minimal controls. Bilinear resampling, edge padding,
  exact field/pixel replay, source-grouped bootstrap, feature contrasts against
  frozen blur/noise and M4A interference, and a predeclared gate. No model is
  trained. The result is CONDITIONAL PASS because the lowest rung has
  between-source SD below the declared margin.

**Implemented for M5B (grouped geometric-robustness benchmark):**

- Zero-shot evaluation of the frozen M3 model on M5A test sources, controlled
  conventional adaptation on M3 blur/noise plus M5A 1.0–3.0 px training-source
  rows with validation-only selection, a naive expanded-training-mean baseline,
  and original-task retention on the untouched M3 test split. The 0.5 px rung is
  diagnostic only; clean and control rows are excluded from every fit frame.
  Predeclared integrity and leakage checks run before evaluation.

**Implemented for M5C (retention-aware model selection and calibration):**

- Six predeclared candidate policies (frozen M3, the M5B expanded HGB and one
  expanded conventional alternative, each raw and with a single affine
  calibration), a calibration procedure that fits only on out-of-sample
  validation predictions with leave-one-fold-out validation scoring, and a
  predeclared retention-aware selection rule that prices geometric improvement
  against clean-reference and original-task regressions. The expanded
  alternative is justified from the existing M5B validation table, never from
  an M5C test result.

**Implemented for M5D (retention-aware training objectives):**

- Four predeclared candidate training policies that hold the model family,
  hyperparameters, training rows and seed fixed and vary only the per-example
  sample weight: the frozen M5C champion, an unweighted retrained control, a
  severity-aware weighting and a retention-aware weighting. Weights are built
  from training metadata only, normalised to mean one, and never enter the
  feature matrix. A predeclared eligibility-and-score rule with a per-severity
  constraint selects on validation only.

**Planned (not started):**

- Classification baseline: macro-F1, balanced accuracy, confusion matrix,
  ordinal MAE / quadratic weighted kappa for severity.
- Broader error taxonomy and qualitative examples.
- Cross-family or cross-generator generalisation beyond the current split.
- A retention-aware selection or calibration protocol, if the line continues.

### Stage 9 — Dashboard and reporting

- Streamlit app for browsing sample images, distributions, model metrics, and
  error slices.
- Static reports generated under `reports/` (Git-ignored) with a short summary
  committed in the experiment log when a milestone concludes.

## 3. Package layout

```text
src/captcha_quality/
├── __init__.py         # package metadata (implemented)
├── imaging.py          # shared image-array contract (implemented, M1)
├── render.py           # deterministic pilot text renderer (implemented, M1)
├── corruptions.py      # blur + noise severity ladders (implemented, M1)
├── quality_metrics.py  # reference-based SSIM/PSNR (implemented, M1)
├── summarise.py        # pilot statistics (implemented, M1)
├── gate.py             # predeclared acceptance gate + report (implemented, M1)
├── pilot.py            # pilot runner and `captcha-pilot` CLI (implemented, M1)
├── provenance.py       # versions, revisions, hashing, JSON safety    (implemented, M2)
├── config.py           # typed dataset config + generation hash       (implemented, M2)
├── splits.py           # deterministic grouped split assignment       (implemented, M2)
├── dataset.py          # manifest, generation, verification, compare  (implemented, M2)
├── features.py         # typed handcrafted feature extraction         (implemented, M2)
├── diagnostics.py      # grouped correlation/baseline diagnostic      (implemented, M2)
├── cli.py              # `captcha-dataset` subcommands                (implemented, M2)
├── modelling.py        # conventional models, metrics, persistence      (implemented, M3)
├── benchmark.py        # frozen protocol, diagnostics, report/CLI      (implemented, M3)
├── interference.py     # occlusion/interference pilot corruption        (implemented, M4A)
├── robustness_pilot.py # pilot runner, verification, analysis, gate    (implemented, M4A)
├── robustness_benchmark.py # grouped zero-shot/adaptation benchmark, CLI (implemented, M4B)
├── geometric.py        # smooth elastic field, resampling, replay       (implemented, M5A)
├── geometric_pilot.py  # pilot generation, analysis, gate, report, CLI  (implemented, M5A)
├── geometric_benchmark.py # grouped zero-shot/adaptation benchmark, CLI (implemented, M5B)
├── retention_aware.py  # retention-aware selection and calibration, CLI (implemented, M5C)
├── retention_aware_training.py # retention-aware training objectives, CLI (implemented, M5D)
└── analysis.py         # broader slices, statistics, plots             (planned, M5+)
```

`render.py` now serves both the frozen Milestone 1 pilot and the Milestone 2
dataset. Dataset configurations enable ink-centred placement and strict bounds;
the pilot keeps the original defaults so `EXP-0001` remains byte-reproducible.
`benchmark.py` and `modelling.py` implement the conventional M3 stage;
`robustness_benchmark.py` implements M4B using the same feature/model
persistence path. `geometric.py` and `geometric_pilot.py` implement the isolated
M5A elastic-warp pilot, `geometric_benchmark.py` implements M5B on top of it,
`retention_aware.py` implements M5C and `retention_aware_training.py` implements
M5D, all using the same persistence, verification and reporting conventions. No
CNN code exists yet.

## 4. Artifact contracts

| Artifact | Location | Key fields |
| --- | --- | --- |
| Dataset config | `configs/dataset_*.toml` | seed, text-length distribution, fonts, corruption ladders, split ratios, version, output dir |
| Public manifest | `data/processed/<dataset>/manifest.csv` | `sample_id`, `source_id`, `variant_id`, `split_group_id`, `split`, `text_length`, `font_id`, `render_seed`, `variant_seed`, `corruption_type`, `severity_level`, `corruption_parameters`, `image_path`, `reference_path`, `ssim_target`, `psnr_target`, `psnr_infinite`, hashes, `dataset_version`, `generation_config_hash` |
| Sources | `data/processed/<dataset>/sources.csv` | one row per clean source; split, font, background level, clean path/hash |
| Internal ground truth | `data/processed/<dataset>/ground_truth.csv` | `source_id`, `text` (never in the public manifest) |
| Feature table | `data/processed/<dataset>/features.csv` | `sample_id` + the eight documented features |
| Dataset summary | `data/processed/<dataset>/dataset_summary.json` | counts, split summaries, font resolution, config hash, artifact hashes, disk usage |
| Diagnostic artifacts | `reports/<dataset>/` (Git-ignored) | correlations, model diagnostics, confounds, markdown report |
| Benchmark config | `configs/benchmark_v0_1_0.toml` | seed, held-out font, candidate grids, coverage quantiles, bootstrap count, selection rule, output paths |
| Model artifact | `models/benchmark_v0_1_0/` (Git-ignored) | `.joblib` estimator + JSON card: feature names/order, feature-set version, target, dataset generation/split hashes, selection rule, metrics, package/scikit-learn versions, artifact SHA-256 |
| Benchmark artifacts | `reports/benchmark_v0_1_0/` (Git-ignored) | candidate comparison, family test metrics, predictions, residuals, metadata diagnostics, gaps, stratified rows, coverage definition, correlations, held-out-font tables, grouped uncertainty, `run_metadata.json`, Markdown report |
| Pilot config | `configs/robustness_pilot_occlusion_v0_1_0.toml` | seed, frozen dataset hashes, two stroke ladders, predeclared gate thresholds, output paths |
| Pilot dataset | `data/processed/occlusion_interference_pilot_v0.1.0/` (Git-ignored) | separate manifest, sources, features, summary and 540 images; no raw text |
| Pilot reports | `reports/robustness_pilot/` (Git-ignored) | level/uncertainty/trend/feature/saturation tables, gate results, Markdown report, contact sheets |
| Robustness benchmark config | `configs/robustness_benchmark_v0_1_0.toml` | frozen M3/M4A/model hashes, feature schema, seed, 12-candidate grid, selection metric, bootstrap count, output paths |
| Robustness model | `models/robustness_benchmark_v0_1_0/` (Git-ignored) | expanded `.joblib` estimator + JSON card: feature names/order, target, dataset/split hashes, selected candidate, metrics, package versions, artifact SHA-256 |
| Robustness reports | `reports/robustness_benchmark_v0_1_0/` (Git-ignored) | zero-shot/adaptation metrics, predictions, residuals, isolation checks, data accounting, candidate validation, target distribution, uncertainty, `run_metadata.json`, Markdown report |
| Geometric pilot config | `configs/geometric_distortion_pilot_v0_1_0.toml` | seed, frozen M3/M4A hashes, amplitude ladder, field grid/smoothing, interpolation/border mode, gate thresholds, output paths |
| Geometric pilot dataset | `data/processed/geometric_distortion_pilot_v0.1.0/` (Git-ignored) | separate manifest, sources, features, summary, field specifications and 480 images; no raw text |
| Geometric pilot reports | `reports/geometric_distortion_pilot/` (Git-ignored) | level/uncertainty/trend/control/group tables, feature effect/contrast/correlation tables, gate results, Markdown report, contact sheet |
| Geometric benchmark config | `configs/geometric_robustness_benchmark_v0_1_0.toml` | seed, frozen M3/M5A/model hashes, feature schema, primary/diagnostic severity protocol, 12-candidate grid, selection metric, bootstrap count, output paths |
| Geometric benchmark model | `models/geometric_robustness_benchmark_v0_1_0/` (Git-ignored) | expanded `.joblib` estimator + JSON card: feature names/order, target, dataset/split hashes, selected candidate, metrics, package versions, artifact SHA-256 |
| Geometric benchmark reports | `reports/geometric_robustness_benchmark_v0_1_0/` (Git-ignored) | zero-shot/adaptation/retention metrics, predictions, residuals, isolation checks, data accounting, candidate validation, target distribution, uncertainty, `run_metadata.json`, Markdown report |
| Retention-aware config | `configs/retention_aware_selection_v0_1_0.toml` | seed, frozen M5B hashes, feature schema, candidate policies, calibration procedure, retention-aware selection rule and weights, bootstrap count, output paths |
| Retention-aware models | `models/retention_aware_selection_v0_1_0/` (Git-ignored) | expanded random-forest bundle and selected-policy bundle, each with a JSON card: feature names/order, target, dataset/split hashes, selected candidate, calibration parameters, metrics, package versions, artifact SHA-256 |
| Retention-aware reports | `reports/retention_aware_selection_v0_1_0/` (Git-ignored) | validation slice metrics, selection table, test slice metrics, predictions, uncertainty, paired uncertainty, isolation checks, `run_metadata.json`, Markdown report |
| Retention-aware training config | `configs/retention_aware_training_v0_1_0.toml` | seed, frozen M5C hashes, model family and hyperparameters, candidate policies, weighting formulas and normalisation, eligibility tolerances, selection score and weights, bootstrap count, output paths |
| Retention-aware training models | `models/retention_aware_training_v0_1_0/` (Git-ignored) | one bundle per refitted candidate, each with a JSON card: feature names/order, target, weighting scheme and parameters, weight diagnostics, validation and test metrics, package versions, artifact SHA-256 |
| Retention-aware training reports | `reports/retention_aware_training_v0_1_0/` (Git-ignored) | validation slice metrics, selection table, test slice metrics, predictions, uncertainty, paired uncertainty, isolation checks, `run_metadata.json`, Markdown report |
| Metrics | `reports/` + experiment log | model, split, metric, value, CI, slice definition |

Dataset artefacts are Git-ignored; tiny fixtures are generated in tests. The
public manifest is timestamp-free by design, so repeated runs can be compared
byte-for-byte.

## 5. Milestone mapping

| Milestone | Scope |
| --- | --- |
| M0 | Structure, tooling, documentation (complete) |
| M1 | Validate target definition and metric behaviour before scale-up (complete) |
| M2 | Dataset generation, manifest, grouped splits, handcrafted features, leakage diagnostic (complete) |
| M3 | Conventional baselines with grouped evaluation and held-out-font stress test (complete) |
| M4 | Occlusion/interference robustness benchmark; CNN comparison (M4A pilot and M4B benchmark complete; CNN not started) |
| M5 | Geometric-distortion feasibility pilot; grouped geometric-robustness benchmark; retention-aware selection and calibration; retention-aware training objectives; broader robustness analysis and error taxonomy (M5A pilot, M5B benchmark, M5C selection study and M5D training study complete) |
| M6 | Dashboard, reproducibility packaging, final report |

## 6. Key risks and open questions

- **Proxy validity:** reference-based IQA may not track perceived legibility.
  M1 showed the SSIM target is sensitive and non-degenerate under the pilot
  conditions; it still is not a readability measure.
- **Trivial predictability:** M2 found that no single feature makes the target
  trivial on held-out groups, but the eight-feature linear baseline reaches test
  R² 0.962; M3's random forest reaches 0.986. A CNN must beat that frozen
  baseline, and any gain must survive a harder shift.
- **Metadata confounding:** severity + family alone reaches test R² 0.911. The
  synthetic generation design explains most of the target, so high model R² is
  not evidence of perceptual understanding.
- **Font shift:** M3's held-out-font stress test survives (R² 0.980), but only
  three fonts exist and the held-out font shares the renderer/corruption design.
  It is not broad generalisation evidence.
- **Shortcut learning:** CNNs may learn rendering/font cues instead of
  degradation. Grouped splits and font-held-out evaluation are required.
- **Synthetic-to-real gap:** results apply to the generator, not to deployed
  CAPTCHAs. This limitation is stated in every report.
- **Pilot scope:** the M4A gate supports a non-degenerate, feature-distinct,
  replayable family, but does not show that it is harder for humans or that it
  generalises beyond 60 sources, three fonts and one canvas.
- **Geometric interpolation confound:** M5A's bilinear resampling and edge
  padding change pixels even at zero displacement, and the 0.5 px rung is close
  to that floor. The pilot reports a minimal control and border/clipping
  diagnostics; it returns CONDITIONAL PASS and does not support a low-end
  between-source variance claim.
- **Zero-shot shift:** frozen M3 is near-perfect on clean references but fails
  absolute calibration on M4A interference (test R² -1.045, systematic
  under-prediction) and on M5B geometric distortion (test R² -1.186, mean
  residual -0.072). The benchmarks report this as a transfer failure, not as a
  human-difficulty result.
- **Geometric adaptation trade-off:** M5B's adapted model reduces aggregate
  geometric RMSE by 31.8% but its R² stays near zero (-0.017) and it degrades
  clean-reference calibration on the original task (clean-slice MAE 0.0134
  versus 0.0004 for frozen M3). The adapted model is a geometric-robustness
  candidate, not a drop-in replacement. M5C resolves this on the same frames by
  changing the model family: an expanded random forest keeps the original task
  and clean references while improving the geometric slice further, at a small
  clean-reference cost.
- **Reference noise in the retention comparison:** the frozen M3 model was
  itself selected on the M3 validation split, and its clean-reference
  validation slice (RMSE 0.0162) is unusually noisy relative to its
  clean-reference test slice (0.0012). M5C's measured clean regression is
  therefore an upper bound, not a point estimate.
- **Feature representation may be the binding constraint:** the eight
  image-only features are global whole-image statistics that do not describe
  *where* on the canvas a distortion occurs. M5D's negative result is
  consistent with this: within one model family, one representation and one
  training sample, sample weighting moves error between slices rather than
  removing it. This is a hypothesis consistent with the evidence, not an
  isolated cause.
- **Small test support:** M4B, M5B, M5C and M5D have only 12 held-out sources.
  Source-grouped intervals are wide and preliminary; no universal significance
  threshold was introduced after seeing test results.
- **Font licensing:** only redistributable fonts may be used or bundled.

## 7. Implemented today

Milestone 0: packaging metadata, ignore rules, documentation, and configuration
smoke tests.

Milestone 1: a deterministic pilot renderer, two corruption families with
explicit severity ladders, reference SSIM/PSNR metrics, a predeclared gate, a
`captcha-pilot` CLI, and a completed target-validation experiment (`EXP-0001`).
SSIM passed the gate for blur and noise; PSNR passed for noise and failed for
blur because of an ink-coverage confound; MS-SSIM is deferred.

Milestone 2: a typed dataset configuration, deterministic per-source rendering
with multiple text lengths and fonts, a manifest with stable IDs and content
hashes, font-stratified grouped splits, eight documented handcrafted features,
a staging/verification/comparison workflow (`captcha-dataset`), and a grouped
feature/target diagnostic. Two datasets were generated: a 9-source pilot
(81 samples) and a 60-source reference dataset (540 samples). The diagnostic
found no single-feature triviality (best validation/test R² 0.680/0.644) but a
near-complete eight-feature linear baseline (0.968/0.962).

Milestone 3: a frozen conventional benchmark with 23 candidates, validation-only
selection, test evaluated once, metadata diagnostics, corruption/severity/
coverage slices, grouped bootstrap, model persistence/verification, and a
held-out-font stress test. The selected random forest reached test R² 0.9865 /
RMSE 0.0226 versus naive 0.1942; metadata-only severity + family reached 0.9112;
the held-out font reached 0.9799 (selected HGB) and 0.9832 (random-forest family
winner).

Milestone 4A: an isolated occlusion/interference pilot with thin lines and
curved arcs, deterministic stroke geometry, exact clean re-rendering, feature
contrast and a predeclared gate. All five criteria passed; it established a
replayable, non-degenerate, feature-distinct family without making a difficulty
or solvability claim.

Milestone 4B: a grouped robustness benchmark with 12 training-source expansion,
validation-only selection among 12 conventional candidates, frozen-M3 zero-shot
evaluation and source-grouped intervals. Frozen M3 reached test R² -1.045;
the adapted random forest reached MAE/RMSE/R² 0.0131/0.0178/0.9752 versus
0.1261/0.1466/-0.6912 for the expanded training mean. On the original M3 test
rows the adapted model stayed close to frozen M3 (0.0158/0.0239/0.9849 versus
0.0157/0.0226/0.9865). The evidence supports conventional modelling for this
benchmark and does **not** justify a CNN on this split.

Milestone 5A: an isolated smooth elastic-warp pilot with one seeded field per
source, a 0.5–3 px maximum-displacement ladder, identity/zero-field/minimal
controls and exact field replay. Identity and zero-field controls reproduced all
60 clean references exactly; SSIM fell monotonically on 60/60 sources (0/180
adjacent increases), and at severity 4 the features were strongly separated from
blur, noise, thin lines and arcs. The predeclared gate returned CONDITIONAL PASS
because severity 1 (0.5 px) had between-source SD 0.0025 below the 0.005 margin.
M5A trains no model and makes no human-readability claim.

Milestone 5B: a grouped geometric-robustness benchmark on the frozen M3
source split, with the 0.5 px rung held out as a diagnostic only. Frozen M3
failed zero-shot geometric calibration (primary 1.0–3.0 px MAE 0.0777, RMSE
0.0967, R² -1.1858 versus 0.2030/0.2132/-9.6363 for the M3 training-mean
baseline). Controlled adaptation with representative geometric training data and
validation-only selection among 12 conventional candidates reduced aggregate
geometric MAE to 0.0544 and RMSE to 0.0659 (31.8% lower), but left R² near
zero (-0.0165) and was worse at 1.0 px than frozen M3. On the untouched M3
test rows the adapted model stayed close in aggregate (MAE 0.0179, RMSE 0.0243,
R² 0.9843 versus 0.0157/0.0226/0.9865) but lost clean-reference calibration
(clean-slice MAE 0.0134 versus 0.0004). Conventional feature-based ML remains
the benchmark family; the evidence does **not** justify a CNN on this split.

Milestone 5C: a retention-aware selection and calibration study on the same
frozen frames, with six predeclared candidate policies and a predeclared
retention-aware rule that prices geometric improvement against clean-reference
and original-task regressions. The selected expanded random forest cuts primary
geometric RMSE by 39.0% (0.0590 versus the frozen 0.0967, positive R2 0.1855,
improving on 11 of 12 held-out sources) while keeping the original M3 test RMSE
at 0.0220 versus the frozen 0.0226, at a small clean-reference cost (RMSE
0.0023 versus 0.0012). The affine calibration was tested on every base model
and selected for none of them; it reduced the M5B HGB's clean-reference offset
but could not remove it without moving the other slices. The trade-off was
resolved mostly by the model family rather than by the calibration layer.
No dashboard, deployment or final presentation exists yet.

Milestone 5D: a retention-aware training-objective study in which the model
family, hyperparameters, training rows and seed are held fixed and only the
per-example sample weight varies, so any difference is attributable to the
objective. The unweighted retrained control reproduces the M5C champion
bit-for-bit, confirming the isolation. **No weighted candidate was selected.**
The severity-aware scheme is ineligible because it regresses the 1.0 px rung by
36% while its aggregate geometric RMSE barely moves (+0.0005) — exactly the
failure the predeclared per-severity constraint exists to catch. The
retention-aware scheme is eligible but scores worse than the champion
(+0.00618): it buys clean-reference accuracy at the cost of the original task
and the aggregate geometric slice. The M5C champion is retained. The negative
result points at the *representation* rather than the objective: the eight
image-only features are global whole-image statistics that do not describe
*where* a distortion occurs. No dashboard, deployment or final presentation
exists yet.
