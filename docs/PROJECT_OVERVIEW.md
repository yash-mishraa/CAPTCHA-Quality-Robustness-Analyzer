# Project Overview

**Status:** Milestone 0 — foundation only. This document describes the intended
end-to-end pipeline. Nothing below the "Implemented today" section exists yet.
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

- **Input:** human-authored YAML/TOML under `configs/`.
- **Output:** a validated, immutable run configuration with a hash.
- **Planned contents:** Python/package versions, seed, character set, text
  length distribution, fonts, canvas size, colour model, corruption grid
  (family × severity × probability), sample counts, split ratios, metric list.
- **Rule:** no hard-coded experimental parameters in source; configs are
  versioned and referenced by the experiment log.

### Stage 2 — Synthetic text rendering

- **Input:** config + seed.
- **Output:** a clean RGB/grayscale image plus a generation record
  (`sample_id`, text, font, font size, position, rotation, seed, renderer
  version).
- **Planned v1:** Pillow-based rendering with a small, licence-compatible font
  set, configurable character set (ambiguous characters excluded by default),
  random but seeded placement, optional background texture.
- **Purpose:** provides both training images and the **reference** required by
  full-reference quality metrics.

### Stage 3 — Distortion engine

- **Input:** clean image + corruption specification.
- **Output:** degraded image + exact applied parameters (`family`, `severity`,
  numeric parameters, random state).
- **Planned v1 families:** Gaussian noise, blur, contrast/brightness shift,
  geometric warp/rotation, occlusion/overlay lines, and compression artefacts.
  Each family has an ordered severity ladder rather than a single "hard" mode.
- **Rule:** transformations are composable but recorded sequentially so effects
  are attributable.

### Stage 4 — Reference-based quality targets

- **Input:** clean reference and degraded image pair.
- **Output:** scalar quality targets per pair (for example SSIM, MS-SSIM, PSNR).
- **Candidate primary target:** a defined full-reference metric computed against
  the clean rendering of the same text (see `METHODOLOGY.md`).
- **Why reference-based:** it is deterministic, reproducible, and available by
  construction in a synthetic corpus. It is a proxy for image degradation, not
  for human difficulty.
- **Validation gate (M1):** check sensitivity, monotonicity across severity
  levels, and stability across fonts/seeds before scaling the dataset.
- **M1 outcome:** SSIM passed this gate for Gaussian blur and additive noise;
  PSNR passed for noise but not for blur (ink-coverage confound). See
  `docs/EXPERIMENT_LOG.md`, `EXP-0001`.

### Stage 5 — Feature extraction

- **Input:** image (and optionally the reference).
- **Output:** a row of interpretable features: dimensions, contrast, entropy,
  sharpness/edge energy, colour statistics, noise estimates, frequency summaries,
  and reference-difference summaries.
- **Purpose:** support conventional ML baselines and provide interpretable
  comparisons against CNN embeddings.

### Stage 6 — Dataset assembly

- **Output:** versioned image directories, a manifest (Parquet/CSV), and split
  assignments.
- **Split rule:** group by generation seed/text so near-duplicates never cross
  splits. Splits are frozen and reused by every model.
- **Metadata policy:** generation metadata is retained for analysis and
  slicing, but corruption parameters are **not** model inputs in the quality
  regression task (that would leak the target's cause).

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

## 3. Planned package layout

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
├── config.py           # typed config loading and validation          (planned)
├── dataset.py          # manifest, splits, dataset versioning         (planned)
├── features.py         # handcrafted feature extraction               (planned)
├── modelling.py        # baseline and CNN training/evaluation         (planned)
└── analysis.py         # slices, statistics, plots                    (planned)
```

The Milestone 1 modules are a deliberately narrow pilot, not the Milestone 2
pipeline. In particular, `render.py` implements only the fixed-canvas,
unrotated, grayscale rendering needed to validate metrics; the Milestone 2
renderer will add configuration, backgrounds, text-length distributions, and
manifest output. `modeling` modules do not exist yet.

## 4. Artifact contracts (planned)

| Artifact | Location | Key fields |
| --- | --- | --- |
| Run config | `configs/` | seed, fonts, corruption grid, metrics, split rule |
| Manifest | `data/raw/` | sample_id, seed, text, font, family, severity, params, path, code revision |
| Feature table | `data/processed/` | sample_id, features, targets, split |
| Model artifact | `models/` | model type, hyperparameters, training config hash, split hash |
| Metrics | `reports/` + experiment log | model, split, metric, value, CI, slice definition |

## 5. Milestone mapping

| Milestone | Scope |
| --- | --- |
| M0 | Structure, tooling, documentation (this milestone, complete) |
| M1 | Validate target definition and metric behaviour before scale-up |
| M2 | Renderer, distortions, quality targets, feature extraction, manifest |
| M3 | Conventional baselines with grouped evaluation |
| M4 | CNN comparison |
| M5 | Robustness analysis and error taxonomy |
| M6 | Dashboard, reproducibility packaging, final report |

## 6. Key risks and open questions

- **Proxy validity:** reference-based IQA may not track perceived legibility.
  M1 must show the target is sensitive and non-degenerate before modelling.
- **Trivial predictability:** severity may be recoverable from a few features,
  making the regression task easy. This would be documented, not hidden, and
  would shift emphasis to cross-family generalisation.
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
blur because of an ink-coverage confound; MS-SSIM is deferred. No dataset
pipeline, feature extraction, model, or dashboard exists yet.
