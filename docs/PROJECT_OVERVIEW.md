# Project Overview

**Status:** Milestones 0–2 complete. A reproducible dataset and image-feature
pipeline now exists; modelling, CNN comparison, and the dashboard are planned.
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
- **Deferred:** contrast/brightness shifts, geometric warps, occlusion, and
  compression artefacts. They are not added until the current two-family
  analysis is complete.
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

Two tracks under identical splits:

1. **Conventional baselines:** ridge/linear regression and tree ensembles on
   image-quality features; secondary classification/ordinal experiments for
   corruption family and severity.
2. **CNN comparison:** a small CNN trained from scratch and a pretrained
   transfer-learning variant, with matched preprocessing and early stopping.

Both tracks report the same metrics and per-slice breakdowns. Simpler models
are the default; a CNN must justify its cost out-of-sample.

### Stage 8 — Evaluation and robustness analysis

- Regression: MAE, RMSE, R², Spearman correlation, residual slices by family
  and severity.
- Classification baseline: macro-F1, balanced accuracy, confusion matrix,
  ordinal MAE / quadratic weighted kappa for severity.
- Uncertainty: grouped cross-validation and bootstrap confidence intervals.
- Error analysis: systematic failures by font, text length, corruption family,
  and severity; examples retained as qualitative evidence.

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
├── modelling.py        # baseline and CNN training/evaluation         (planned, M3/M4)
└── analysis.py         # slices, statistics, plots                    (planned, M5)
```

`render.py` now serves both the frozen Milestone 1 pilot and the Milestone 2
dataset. Dataset configurations enable ink-centred placement and strict bounds;
the pilot keeps the original defaults so `EXP-0001` remains byte-reproducible.
No modelling or CNN code exists yet.

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
| Model artifact | `models/` (planned) | model type, hyperparameters, training config hash, split hash |
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
| M3 | Conventional baselines with grouped evaluation |
| M4 | CNN comparison |
| M5 | Robustness analysis and error taxonomy |
| M6 | Dashboard, reproducibility packaging, final report |

## 6. Key risks and open questions

- **Proxy validity:** reference-based IQA may not track perceived legibility.
  M1 showed the SSIM target is sensitive and non-degenerate under the pilot
  conditions; it still is not a readability measure.
- **Trivial predictability:** M2 found that no single feature makes the target
  trivial on held-out groups, but the eight-feature linear baseline reaches test
  R² 0.962. Later models must beat that baseline; this shifts emphasis to
  cross-family and held-out-font generalisation.
- **Shortcut learning:** CNNs may learn rendering/font cues instead of
  degradation. Grouped splits and font-held-out evaluation are required.
- **Synthetic-to-real gap:** results apply to the generator, not to deployed
  CAPTCHAs. This limitation is stated in every report.
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
near-complete eight-feature linear baseline (0.968/0.962), so a later CNN must
beat that baseline. No model, CNN, or dashboard exists yet.
