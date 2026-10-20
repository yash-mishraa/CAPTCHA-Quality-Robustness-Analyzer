# CAPTCHA Quality & Robustness Analyzer

A reproducible, end-to-end computer-vision project for generating synthetic
CAPTCHA-style text images, applying controlled visual distortions, measuring
image quality, and modelling how those distortions affect image degradation.

> **Status:** Milestones 0–3 complete — the metric-validation pilot, a
> reproducible dataset/feature pipeline, and a leakage-safe conventional-ML
> benchmark with a held-out-font stress test have been run and logged. The CNN
> comparison and dashboard are **not implemented yet**, and the project does
> **not** predict human CAPTCHA solvability. See
> [Current status](#current-status).

## Overview

CAPTCHA images are intentionally degraded, but "degraded" is rarely measured.
This project builds a controlled testbed in which the degradation is known,
parameterised, and reproducible. Synthetic text images are rendered from known
strings and fonts, passed through one or more configurable corruption families,
and then described by reference-based image-quality metrics and handcrafted
features. Those measurements become the supervised targets for conventional ML
baselines and a CNN comparison.

The long-term goal is a small, well-documented system that answers questions
such as:

- Which distortion families cause the largest measurable loss of image quality?
- Can image-quality metrics be predicted from cheap image features?
- How sensitive are those predictions to corruption type and severity?
- Where do feature-based models and CNNs disagree, and why?

## Problem statement

There is no widely used, open, reproducible benchmark for quantifying how much
a CAPTCHA-style transformation degrades the underlying text image. Practitioners
choose distortion parameters by intuition, and portfolio datasets often report
model accuracy without documenting the data-generation process.

This project addresses the measurement gap by treating image quality and
robustness as supervised learning problems over a **synthetic, fully labelled**
image corpus. Every image carries its generating parameters, so error analysis
can be grouped by corruption family and severity rather than guessed after the
fact.

### Intended users

- ML/CV practitioners who want a documented synthetic-data and evaluation
  workflow.
- Researchers and students studying image-quality assessment or corruption
  robustness.
- Reviewers assessing the reproducibility of a computer-vision portfolio
  project.

## Scope and limitations

**In scope:**

- A deterministic synthetic text-image renderer with recorded metadata
  (**implemented**).
- A configurable distortion engine with two controlled corruption families and
  severity ladders (**implemented**).
- Reference-based image-quality targets (SSIM primary, PSNR secondary for noise)
  and eight documented handcrafted features (**implemented**).
- A versioned dataset with a public manifest, grouped splits, and programmatic
  verification (**implemented**).
- Leakage-aware conventional ML baselines with grouped evaluation and a
  held-out-font stress test (**implemented**).
- Robustness/error analysis grouped by corruption family, severity and ink
  coverage for the conventional benchmark (**implemented**; a broader M5 error
  taxonomy is planned).
- A CNN comparison (**planned**).
- A Streamlit dashboard for exploring results (**planned**).

**Explicitly out of scope / not claimed:**

- **No prediction of human solvability.** The project has no human-labelled
  responses, so it cannot and will not claim to predict whether a person can
  solve a CAPTCHA. Any quality score here is a machine-computed proxy.
- **No evaluation of production CAPTCHAs.** All images are synthetic and
  generated locally; no third-party CAPTCHA services are scraped.
- **No security claims.** This is an image-quality and robustness study, not a
  CAPTCHA-strength benchmark or an attack toolkit.
- **No novelty claim.** The methods use established image-quality metrics and
  standard supervised learning; the contribution is a careful, reproducible
  pipeline and analysis.
- **No deep-learning results.** PyTorch is a planned dependency, not an
  implemented component. The M3 benchmark shows that a conventional random
  forest reconstructs 98.6% of held-out SSIM variance, so a CNN is not justified
  for this synthetic target without a harder shift.

## Architecture

```text
                    configs/ (seeds, fonts, corruption grid)
                                  |
                                  v
  +-------------------+    +-------------------+    +---------------------+
  | Text renderer     | -> | Distortion engine | -> | Degraded image      |
  | (clean reference) |    | (type + severity) |    | + generation record |
  +-------------------+    +-------------------+    +---------------------+
            |                       |                        |
            |                       v                        v
            |             +-------------------+    +---------------------+
            +-----------> | Quality targets   |    | Feature extractor   |
                          | SSIM / MS-SSIM /  |    | sharpness, contrast,|
                          | PSNR vs reference |    | entropy, edge stats |
                          +-------------------+    +---------------------+
                                    |                        |
                                    +-----------+------------+
                                                v
                                   +--------------------------+
                                   | Versioned dataset +      |
                                   | manifest (Parquet/CSV)   |
                                   +--------------------------+
                                                |
                            +-------------------+-------------------+
                            v                                       v
                 +---------------------+                 +---------------------+
                 | Conventional ML     |                 | CNN comparison      |
                 | (scikit-learn)      |                 | (PyTorch)           |
                 +---------------------+                 +---------------------+
                            |                                       |
                            +------------------+--------------------+
                                               v
                                  +--------------------------+
                                  | Evaluation + robustness  |
                                  | analysis + error slices  |
                                  +--------------------------+
                                               |
                                               v
                                  +--------------------------+
                                  | Streamlit dashboard      |
                                  +--------------------------+
```

Stages 1–7 are implemented. Stage 8 is partially implemented for the
conventional benchmark (grouped uncertainty, corruption/coverage slices and a
held-out-font stress test); the full error taxonomy is planned for M5. Stage 9
is planned. Each implemented stage writes artifacts with an explicit schema and
a recorded configuration hash, so a result can be traced back to the exact data
and parameters that produced it. Manifests and summaries are timestamp-free,
which makes repeated runs byte-comparable.

## Technology stack

| Layer | Tools | Status |
| --- | --- | --- |
| Language | Python >= 3.11 | Configured |
| Data handling | NumPy, pandas | Core dependencies |
| Image I/O and rendering | Pillow | Core dependency |
| Image processing | OpenCV, scikit-image | Optional extra `image`; scikit-image provides SSIM/PSNR and the Otsu/Canny features |
| Conventional ML | scikit-learn, joblib | Optional extra `ml`; used by the M3 benchmark |
| Deep learning | PyTorch | Optional extra `deep-learning` (planned use) |
| Dashboard | Streamlit | Optional extra `app` (planned use) |
| Testing | pytest, pytest-cov | Extra `dev` |
| Lint / format | Ruff | Extra `dev` |
| Packaging | hatchling, `pyproject.toml` | Configured |

Dependencies are grouped by milestone so the foundation installs only what it
needs. Large optional stacks (PyTorch, Streamlit) are declared but not installed
during Milestone 0.

## Repository layout

```text
.
├── app/                     # Streamlit dashboard (planned)
├── configs/                 # Versioned experiment/generator configs
├── data/
│   ├── raw/                 # Generated source images (ignored)
│   ├── processed/           # Feature tables and derived datasets (ignored)
│   └── README.md            # Data and artifact policy
├── docs/
│   ├── PROJECT_OVERVIEW.md  # Full intended pipeline
│   ├── METHODOLOGY.md       # Targets, models, leakage, evaluation
│   └── EXPERIMENT_LOG.md    # Structured experiment record
├── models/                  # Trained artifacts (ignored)
├── notebooks/               # Exploratory notebooks
├── reports/                 # Generated reports and figures (ignored)
├── scripts/                 # Thin CLI entry points
├── src/captcha_quality/     # Importable package
├── tests/                   # Test suite
├── .gitignore
├── LICENSE                  # MIT
├── pyproject.toml
└── README.md
```

## Development setup

### Prerequisites

- Git
- Python 3.11 or newer (the foundation was created and verified with CPython
  3.13.16)
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip` + `venv`

### Option A — uv (recommended)

```bash
uv sync --extra dev --extra image --extra ml
uv run --extra dev --extra image --extra ml pytest
uv run --extra dev --extra image --extra ml ruff check .
uv run --extra dev --extra image --extra ml ruff format --check .
```

The `image` extra is required for SSIM/PSNR and the Otsu/Canny features; the
`ml` extra is required for the Milestone 3 benchmark and its tests. Benchmark
tests skip cleanly if scikit-learn is not installed. To reproduce the Milestone
1 target-validation pilot:

```bash
uv run --extra image captcha-pilot --config configs/pilot_target_validation.toml
```

Outputs are written to `reports/pilot/`, which is Git-ignored.

### Option B — standard venv and pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,image,ml]"
pytest
ruff check .
ruff format --check .
```

The virtual environment (`.venv/`) is ignored by Git. No credentials or
environment files are required at this stage.

### Reproducibility conventions

- Pin the supported Python range in `pyproject.toml` (`>=3.11`).
- Every generator and experiment run is driven by a configuration file and a
  recorded seed.
- Datasets, checkpoints, and generated reports stay out of Git; their configs are
  versioned, and manifests are regenerated deterministically. See
  [`data/README.md`](data/README.md).
- Dataset generation writes to a `.partial` staging directory and renames it into
  place only on success. Existing datasets are never overwritten silently.
- The public manifest contains `text_sha256`, not raw text; raw strings live only
  in the internal, Git-ignored `ground_truth.csv`.
- Results are recorded in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)
  with the exact configuration and dataset version used.

## Planned milestones

| Milestone | Deliverable | Status |
| --- | --- | --- |
| M0 | Repository structure, tooling, documentation foundation | **Complete** |
| M1 | Target definition and validation: confirm the reference-based quality target is stable, sensitive, and worth modelling | **Complete** — SSIM passes for blur and noise; PSNR passes for noise, fails for blur |
| M2 | Dataset generation, manifest, grouped splits, handcrafted features, leakage diagnostic | **Complete** — 540-sample dataset; 11/11 verification checks and 4/4 leakage checks pass |
| M3 | Conventional ML baselines with leakage-safe grouped evaluation | **Complete** — 23-candidate search; selected RF test R² 0.986; held-out-font R² 0.980; CNN not justified |
| M4 | CNN comparison (small CNN and transfer learning) under the same splits | Planned |
| M5 | Robustness analysis by corruption family/severity and error taxonomy | Planned |
| M6 | Streamlit dashboard, reproducibility package, final report | Planned |

Milestone 1 was a validation gate, not a modelling milestone. Milestone 2 turns
that validated target into a versioned dataset and asks whether the target is
trivially predictable from cheap features. Milestone 3 compares conventional
models on that frozen dataset and tests whether the result survives a held-out
font. See [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) for all three
recorded experiments.

## Quick start: dataset pipeline

```bash
uv sync --extra dev --extra image

# 1. End-to-end check on the small pilot configuration (9 sources, 81 samples)
uv run --extra image captcha-dataset generate --config configs/dataset_pilot.toml

# 2. Reference dataset (60 sources, 540 samples), then verify and diagnose
uv run --extra image captcha-dataset generate --config configs/dataset_v0_1_0.toml
uv run --extra image captcha-dataset verify   --dataset data/processed/captcha_quality_v0.1.0
uv run --extra image captcha-dataset diagnose --dataset data/processed/captcha_quality_v0.1.0 \
    --config configs/dataset_v0_1_0.toml

# 3. Reproducibility: generate a second copy and compare
uv run --extra image captcha-dataset generate --config configs/dataset_v0_1_0.toml \
    --output-dir data/processed/repro
uv run --extra image captcha-dataset compare \
    --a data/processed/captcha_quality_v0.1.0 --b data/processed/repro
```

`generate` refuses to replace an existing directory unless `--overwrite` is
passed. `verify` returns exit code 1 when any check fails; configuration and
usage errors return 2. `diagnose` writes a markdown report plus four CSV tables
to `reports/<dataset-name>_<version>/` by default.

Outputs:

| Path | Contents |
| --- | --- |
| `manifest.csv` | one row per sample: IDs, split, family, severity, parameters, paths, SSIM/PSNR targets, hashes; no raw text |
| `sources.csv` | one row per clean source with its split and clean-image hash |
| `ground_truth.csv` | internal `source_id,text` table (Git-ignored) |
| `features.csv` | `sample_id` plus the eight documented features |
| `dataset_summary.json` | counts, per-split summaries, font resolution, split policy and assignment hash, artifact hashes, disk usage |
| `images/` | lossless PNGs: `clean/` and one directory per corruption family |

## Quick start: conventional benchmark

```bash
uv sync --extra dev --extra image --extra ml

# 1. Run the frozen benchmark on the reference dataset
uv run --extra image --extra ml captcha-benchmark run \
    --config configs/benchmark_v0_1_0.toml \
    --dataset data/processed/captcha_quality_v0.1.0

# 2. Verify persisted model cards, feature schema and dataset/split hashes
uv run --extra image --extra ml captcha-benchmark verify-model \
    --dataset data/processed/captcha_quality_v0.1.0 \
    --model-dir models/benchmark_v0_1_0
```

The `run` command fits 23 learned candidates on the training sources, selects on
validation only, evaluates each family winner and the overall winner once on
test, runs the metadata diagnostics and stratified analysis, and repeats the
selection without Times New Roman for the held-out-font stress test. It writes
a Markdown report plus CSV/JSON tables to `reports/benchmark_v0_1_0/` and model
artifacts to `models/benchmark_v0_1_0/`; both are Git-ignored. Use
`--output-dir` / `--model-dir` to redirect them.

The command exits 2 on configuration/dataset errors and 1 if model verification
fails. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5 for the frozen
protocol, model grid, metrics, leakage controls and held-out-font design.

## Current status

Implemented:

- Repository structure, MIT license, `.gitignore`, and contributor-facing docs.
- `pyproject.toml` with a `src/` layout, dependency extras, pytest, coverage, and
  Ruff configuration.
- A deterministic pilot text renderer (`render.py`) with explicit font
  resolution, fixed 192x64 canvas, seeded text/jitter, and read-only clean
  references.
- Two controlled corruption families (`corruptions.py`): Gaussian blur and
  additive Gaussian noise, each with an explicit severity ladder and validated
  parameters.
- Reference metrics (`quality_metrics.py`): SSIM and PSNR via scikit-image, with
  documented range/window settings and exact `+inf` PSNR for identical images.
- A validation pilot (`pilot.py`, `summarise.py`, `gate.py`) with a predeclared
  acceptance gate, a `captcha-pilot` CLI, and Git-ignored CSV/JSON/Markdown
  outputs plus example contact sheets.
- The M1 pilot added 104 tests covering rendering determinism, corruption
  behaviour and non-mutation, metric sanity and input validation, gate
  statistics, configuration errors, and an end-to-end pilot run.
- A typed dataset configuration (`config.py`) with strict key validation, explicit
  text-length distributions, font IDs, background policy, corruption ladders,
  split ratios, seeds, dataset version and output directory, plus a stable
  content-defining generation hash.
- Deterministic grouped splits (`splits.py`): one group per clean source,
  font-stratified allocation, explicit failure when infeasible, and no source in
  more than one split.
- Dataset generation, verification and comparison (`dataset.py`): stable IDs,
  clean reference for every sample, conditional PSNR (`null` + `psnr_infinite`),
  raw-pixel hashes, staging/atomic rename, and recomputation of every target from
  the exact pair.
- Eight image-only features (`features.py`) with documented definitions and
  fixed order; uniform-image short-circuits; no NaN/inf; batch extraction;
  SSIM/PSNR/severity/font/length are never inputs.
- A leakage-safe diagnostic (`diagnostics.py`): train-only fits, validation-based
  one-feature selection, held-out test reporting, metadata baselines, and
  family/severity-conditioned correlations.
- A `captcha-dataset` CLI with `generate`, `verify`, `features`, `diagnose` and
  `compare` subcommands.
- A conventional benchmark (`modelling.py`, `benchmark.py`,
  `configs/benchmark_v0_1_0.toml`): 23-candidate OLS/Ridge/random-forest/HGB
  search, validation-only selection, test evaluated once per family winner,
  metadata diagnostics, corruption/severity/coverage slices, grouped bootstrap,
  and a held-out-font stress test.
- Model persistence (`joblib` + JSON cards) with a `verify-model` command that
  rejects mismatched feature schemas, targets, dataset/split hashes, artifact
  hashes and incompatible scikit-learn major/minor versions.
- 230 tests in total, all passing, including byte-identical dataset
  regeneration, tamper detection, split isolation, feature edge cases,
  train-only preprocessing, validation-only selection, grouped bootstrap,
  held-out-font isolation, persistence checks and CLI/report generation.

Milestone 3 findings (synthetic image similarity only — **not** human
readability or solvability):

- The selected random forest (8 features, depth 8, leaf 1) reached test
  MAE/RMSE/R² **0.0157 / 0.0226 / 0.9865**, versus the naive mean's
  0.1664 / 0.1942 / -0.0000. The 88.4% RMSE reduction is material under the
  predeclared 5% rule.
- A nonlinear forest descriptively beat the best linear family winner (Ridge
  alpha 0.01: RMSE 0.0379, R² 0.9619), by 40.4% RMSE. The top two forest
  configurations were nearly tied on validation, so the family result is more
  stable than the exact hyperparameter choice.
- Severity + family metadata alone reached test R² **0.911**; font + text length
  reached **0.002**; image + metadata (OLS) reached 0.984 and did **not** beat
  the image-only forest. The synthetic design explains most of the target, and
  font/length are not shortcuts in this split.
- Test slices: blur RMSE 0.0255 / R² 0.9824; noise 0.0223 / 0.9721; clean 0.0012
  with R² undefined; highest ink-coverage quartile 0.0288. Grouped bootstrap 95%
  RMSE interval [0.0157, 0.0309] over 12 source groups. No significance claim.
- Held-out font: training on Arial + Courier New and evaluating all 20 Times New
  Roman sources (180 samples) gave selected-HGB R² **0.9799** / RMSE 0.0271,
  with the random-forest family winner at 0.0248 / 0.9832, versus naive 0.1910.
  This is a preliminary three-font stress test, not broad generalisation.
- **A CNN is not justified for this target.** The conventional forest already
  reconstructs 98.6% of held-out variance and captures the nonlinearity. A CNN
  would be justified only on a harder predeclared shift where it beats this
  frozen baseline.

Milestone 2 findings (synthetic image similarity only — **not** human readability
or solvability):

- The reference dataset (60 sources, 540 samples) is byte-reproducible across
  independent runs: identical manifest, features, sources, ground truth, and all
  540 image pixel hashes.
- All 11 verification checks and all 4 leakage checks pass. Splits contain
  36/12/12 sources and 324/108/108 samples; every split contains all three fonts.
- The best single feature (`histogram_entropy`, selected on validation) reaches
  validation/test R² 0.680/0.644 — below the predeclared "strong confound"
  threshold, so there is **no single-feature triviality**.
- However, the eight-feature linear baseline reaches validation/test R²
  0.968/0.962. The target is highly predictable from simple statistics, so a CNN
  is not justified by target predictability alone. M3 models must beat this
  baseline under grouped evaluation.
- Font and text length show no confounding (held-out R² ≈ 0). Corruption severity
  and family explain most of the target (metadata-only R² 0.924/0.911), as
  expected, and are reported as oracle baselines rather than model inputs.
- Family-conditional correlations can reverse sign (e.g. Laplacian variance:
  overall r = 0.066, r = 0.883 within blur, r = -0.861 within noise), so the
  diagnostic reports within-family and within-severity rows.

Milestone 1 findings (synthetic image similarity only — **not** human readability
or solvability):

- **SSIM passed** all predeclared criteria for both blur and noise and is the
  candidate primary target.
- **PSNR passed for noise**, but **failed for blur**: at fixed blur severity,
  PSNR correlates 0.975 with the image's ink coverage, so it largely restates how
  much text is present. It is retained only as a secondary diagnostic for the
  noise family.
- **MS-SSIM is deferred**: no maintained implementation is available without
  adding PyTorch solely for one metric.
- Adjacent severities were separated with large paired effects, no trial-level
  reversals occurred, and clean-vs-clean comparisons were exactly perfect.

Not implemented (planned): the CNN comparison, the full M5 error taxonomy,
notebooks, the Streamlit dashboard, deployment, and any human-solvability or
production-CAPTCHA claims. The conventional model's performance is a synthetic
image-similarity result, not a readability estimate.

## License

Released under the [MIT License](LICENSE).
