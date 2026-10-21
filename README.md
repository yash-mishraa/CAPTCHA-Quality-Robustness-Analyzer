# CAPTCHA Quality & Robustness Analyzer

A reproducible, end-to-end computer-vision project for generating synthetic
CAPTCHA-style text images, applying controlled visual distortions, measuring
image quality, and modelling how those distortions affect image degradation.

> **Status:** Milestones 0–3, M4A, M4B, M5A, M5B, M5C, M5D, M6A, M6B, M6C and
> M6D complete — the metric-validation pilot, a reproducible dataset/feature
> pipeline, a leakage-safe conventional-ML benchmark, an occlusion/interference
> feasibility pilot, a grouped occlusion/interference robustness benchmark, a
> geometric-distortion feasibility pilot, a grouped geometric-robustness
> benchmark, a retention-aware selection and calibration study, a retention-aware
> training-objective study, a spatial-aware feature study, a spatial resolution
> study, a local structural feature study, and a multi-scale structure-tensor
> study have been run and logged. The CNN comparison, full M5 error taxonomy and
> dashboard are **not implemented**, and the project does **not** predict human
> CAPTCHA solvability. See [Current status](#current-status).

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
  severity ladders in the frozen dataset, two isolated occlusion/interference
  pilot variants, and an isolated smooth elastic-warp pilot (**implemented**;
  the geometric family is not in the frozen dataset).
- Reference-based image-quality targets (SSIM primary, PSNR secondary for noise)
  and eight documented handcrafted features (**implemented**).
- A versioned dataset with a public manifest, grouped splits, and programmatic
  verification (**implemented**).
- Leakage-aware conventional ML baselines with grouped evaluation and a
  held-out-font stress test (**implemented**).
- Robustness/error analysis grouped by corruption family, severity and ink
  coverage for the conventional benchmark (**implemented**; a broader M5 error
  taxonomy is planned).
- A grouped occlusion/interference robustness benchmark comparing frozen M3,
  controlled conventional retraining and naive baselines on held-out sources
  (**implemented**).
- A grouped geometric-robustness benchmark comparing frozen M3, controlled
  conventional adaptation and naive baselines on held-out geometric sources,
  with original-task retention on the untouched M3 test split (**implemented**).
- A retention-aware selection and calibration study with predeclared candidate
  policies, a validation-only calibration procedure and a predeclared
  retention-aware selection rule (**implemented**).
- A retention-aware training-objective study that holds the model family,
  hyperparameters, training rows and seed fixed and varies only the per-example
  sample weight (**implemented**).
- A spatial-aware feature study adding localized grid features in a separate
  versioned module, with three predeclared ablations and a grid-resolution
  diagnostic (**implemented**).
- A spatial resolution study varying only the grid (2×2, 4×2, 4×4, 8×8) with
  the feature definitions unchanged, plus training-only redundancy and
  overfitting diagnostics (**implemented**).
- A local structural feature study computing gradient-magnitude and
  gradient-orientation features from the corrupted image alone, under a
  structurally enforced no-reference boundary, with three predeclared ablations
  (**implemented**).
- A multi-scale structure-tensor study computing two-scale coherence,
  anisotropy and tensor-energy summaries from the corrupted image alone, under
  the same structurally enforced no-reference boundary, with three predeclared
  ablations (**implemented**).
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

Stages 1–8 are implemented for the conventional benchmark, the M4B grouped
robustness benchmark (grouped uncertainty, corruption/coverage slices, a
held-out-font stress test and source-isolated interference evaluation), the
M5A geometric-distortion feasibility pilot, the M5B grouped
geometric-robustness benchmark, the M5C retention-aware selection study, the
M5D retention-aware training-objective study, the M6A spatial-aware feature
study, the M6B spatial resolution study, the M6C local structural feature study
and the M6D multi-scale structure-tensor study. The full M5 error taxonomy and
Stage 9 dashboard are planned. Each implemented stage
writes artifacts with an explicit schema and a recorded configuration hash, so a
result can be traced back to the exact data and parameters that produced it. Manifests and summaries are timestamp-free,
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
| M4 | Occlusion/interference robustness benchmark; CNN comparison under the same splits | **Partially complete** — M4A pilot PASS; M4B grouped benchmark complete (frozen M3 R² -1.045, retrained R² 0.975); CNN comparison not started |
| M5 | Geometric-distortion feasibility pilot; broader robustness analysis and error taxonomy | **M5A, M5B, M5C, M5D, M6A, M6B, M6C and M6D complete** — M5A pilot is deterministic/replayable, monotonic 60/60, feature-distinct from blur/noise/interference; CONDITIONAL PASS because the 0.5 px rung has between-source SD below the declared margin. M5B grouped benchmark: frozen M3 fails zero-shot geometric calibration (R² -1.186); validation-selected adaptation cuts primary RMSE 31.8% but leaves R² near zero and degrades clean-reference retention. M5C retention-aware study: a slice-aware rule selects an expanded random forest that cuts primary geometric RMSE 39.0% while keeping the original M3 test RMSE at 0.0220 versus 0.0226. M5D training-objective study: no weighted candidate beats the M5C champion. M6A feature study: coarse spatial features do not help, because a 4×2 grid resolves blur ~25× more strongly than a 1–3 px warp. M6B resolution study: finer grids *do* resolve the warp (correlation +0.45 → +0.83) but still produce no demonstrable gain, so the blocker is feature relevance, not resolution. M6C local-structure study: gradient-magnitude and gradient-orientation features do not beat the global eight while preserving retention; the combined candidate is neutral on the geometric aggregate (paired RMSE +0.0044, interval includes zero) but regresses clean and original-task slices, and the local family is strongly redundant with the global one. M6D structure-tensor study: two-scale coherence/anisotropy/energy features also fail; the combined candidate is neutral on the geometric aggregate (paired RMSE +0.0014, interval includes zero) but breaches the clean-reference tolerance, and the new features are strongly redundant with each other and with the global eight. CNN not justified |
| M6 | Streamlit dashboard, reproducibility package, final report | Planned |

Milestone 1 was a validation gate, not a modelling milestone. Milestone 2 turns
that validated target into a versioned dataset and asks whether the target is
trivially predictable from cheap features. Milestone 3 compares conventional
models on that frozen dataset and tests whether the result survives a held-out
font. Milestone 4A adds a separate feasibility pilot for occlusion and visual
interference; M4B uses its frozen source assignment for a grouped zero-shot and
controlled-adaptation benchmark. M5A adds a separate smooth elastic-warp
feasibility pilot without training a model; M5B uses the M5A dataset for a
grouped zero-shot and controlled-adaptation benchmark on the same frozen M3
source split, M5C adds a retention-aware selection and calibration study on
those same frames, M5D adds a retention-aware training-objective study, M6A adds
a spatial-aware feature study, M6B adds a spatial resolution study, M6C adds
a local structural feature study under a strict no-reference boundary, and M6D
adds a multi-scale structure-tensor study under the same boundary. See
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) for all thirteen recorded
experiments.

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

## Quick start: occlusion/interference pilot

This is a feasibility pilot, not a robustness benchmark. It reads the frozen
reference dataset, verifies and re-renders every clean source, writes a separate
pilot dataset, and does not train or retune a model.

```bash
uv sync --extra dev --extra image

uv run --extra image captcha-robustness-pilot run \
    --config configs/robustness_pilot_occlusion_v0_1_0.toml

# Optional: verify the written pilot dataset against its recorded geometry
uv run --extra image captcha-robustness-pilot verify \
    --config configs/robustness_pilot_occlusion_v0_1_0.toml
```

The run writes `data/processed/occlusion_interference_pilot_v0.1.0/` and
`reports/robustness_pilot/` (both Git-ignored). It exits 1 only when the
predeclared feasibility gate returns `FAIL`; dataset/config errors exit 2.
The gate and findings are recorded in `EXP-0004` and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.9.

## Quick start: grouped robustness benchmark

M4B reads the frozen M3 dataset and the M4A pilot dataset, re-verifies both, and
writes only to its own Git-ignored output directories.

```bash
uv sync --extra dev --extra image --extra ml

uv run --extra image --extra ml captcha-robustness-benchmark run \
    --config configs/robustness_benchmark_v0_1_0.toml

uv run --extra image --extra ml captcha-robustness-benchmark verify \
    --config configs/robustness_benchmark_v0_1_0.toml
```

The run evaluates the frozen M3 model zero-shot on the 12 held-out M4A test
sources, fits a 12-candidate conventional grid on M3 plus M4A training-source
rows, selects on validation sources only, and evaluates once on test. It writes
`reports/robustness_benchmark_v0_1_0/` and
`models/robustness_benchmark_v0_1_0/`, both Git-ignored. `verify` replays both
models and checks the recorded hashes. See
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.10.

## Quick start: geometric-distortion pilot

M5A is a feasibility pilot, not a robustness benchmark. It reads and verifies
the frozen M3 dataset, re-renders every clean source, writes a separate pilot
dataset, and trains no model.

```bash
uv sync --extra dev --extra image

uv run --extra image captcha-geometric-pilot run \
    --config configs/geometric_distortion_pilot_v0_1_0.toml

# Optional: verify the written pilot dataset against recorded fields and targets
uv run --extra image captcha-geometric-pilot verify \
    --config configs/geometric_distortion_pilot_v0_1_0.toml
```

The run writes `data/processed/geometric_distortion_pilot_v0.1.0/` and
`reports/geometric_distortion_pilot/` (both Git-ignored). It exits 1 only when
the predeclared feasibility gate returns `FAIL`; dataset/config errors exit 2.
The gate and findings are recorded in `EXP-0006` and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.11.

## Quick start: grouped geometric-robustness benchmark

M5B benchmarks the frozen M3 model against the M5A geometric dataset. It
measures two different things: **zero-shot transfer** (the frozen M3 model,
never refitted, on unseen geometric sources) and **controlled adaptation** (a
conventional model trained on representative geometric rows, selected on
validation only). The primary geometric score uses the 1.0, 2.0 and 3.0 px
rungs; the 0.5 px rung is reported only as a separately labelled diagnostic.

```bash
uv sync --extra dev --extra image --extra ml

uv run --extra image --extra ml captcha-geometric-benchmark run \
    --config configs/geometric_robustness_benchmark_v0_1_0.toml

# Verify the written run: artifact hashes, prediction replay and protocol
uv run --extra image --extra ml captcha-geometric-benchmark verify \
    --config configs/geometric_robustness_benchmark_v0_1_0.toml
```

The run writes `reports/geometric_robustness_benchmark_v0_1_0/` and
`models/geometric_robustness_benchmark_v0_1_0/` (both Git-ignored). It refuses
to evaluate if any predeclared data-integrity or leakage check fails, and it
never fits anything on test-source rows. The results are recorded in
`EXP-0007` and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.12.

## Quick start: retention-aware selection and calibration

M5C asks whether a conventional model-selection or calibration policy can
improve geometric prediction while retaining performance on clean references
and on the original blur/noise task. It compares six predeclared candidate
policies against the frozen M3 model, fits any calibration only on
out-of-sample validation predictions, and selects using a retention-aware rule
declared before any test evaluation.

```bash
uv sync --extra dev --extra image --extra ml

uv run --extra image --extra ml captcha-retention-aware run \
    --config configs/retention_aware_selection_v0_1_0.toml

# Verify the written run: artifact hashes, prediction replay and protocol
uv run --extra image --extra ml captcha-retention-aware verify \
    --config configs/retention_aware_selection_v0_1_0.toml
```

The run writes `reports/retention_aware_selection_v0_1_0/` and
`models/retention_aware_selection_v0_1_0/` (both Git-ignored). The results are
recorded in `EXP-0008` and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.13.

## Quick start: retention-aware training objectives

M5D asks whether a retention-aware *training objective* can improve geometric
prediction without degrading clean-reference and original-task performance. The
model family, hyperparameters, training rows and seed are held fixed across all
candidates, so only the per-example sample weight varies.

```bash
uv sync --extra dev --extra image --extra ml

uv run --extra image --extra ml captcha-retention-training run \
    --config configs/retention_aware_training_v0_1_0.toml

# Verify the written run: artifact hashes, prediction replay and protocol
uv run --extra image --extra ml captcha-retention-training verify \
    --config configs/retention_aware_training_v0_1_0.toml
```

The run writes `reports/retention_aware_training_v0_1_0/` and
`models/retention_aware_training_v0_1_0/` (both Git-ignored). The results are
recorded in `EXP-0009` and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.14.

## Quick start: spatial-aware feature study

M6A asks whether adding spatially localized image features improves geometric
prediction. The frozen eight-feature pipeline is untouched: the localized
features live in a separate, separately versioned module.

```bash
uv sync --extra dev --extra image --extra ml

uv run --extra image --extra ml captcha-spatial-study run \
    --config configs/spatial_feature_study_v0_1_0.toml

# Verify the written run: artifact hashes, prediction replay and protocol
uv run --extra image --extra ml captcha-spatial-study verify \
    --config configs/spatial_feature_study_v0_1_0.toml
```

The run writes `reports/spatial_feature_study_v0_1_0/` and
`models/spatial_feature_study_v0_1_0/` (both Git-ignored). The results are
recorded in `EXP-0010` and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.15.

## Quick start: spatial resolution study

M6B varies only the grid resolution (2×2, 4×2, 4×4, 8×8), reusing the M6A
feature definitions unchanged, to test whether a finer partition captures
geometric information the coarse grid missed.

```bash
uv sync --extra dev --extra image --extra ml

uv run --extra image --extra ml captcha-resolution-study run \
    --config configs/spatial_resolution_study_v0_1_0.toml

# Verify the written run: artifact hashes, prediction replay and protocol
uv run --extra image --extra ml captcha-resolution-study verify \
    --config configs/spatial_resolution_study_v0_1_0.toml
```

The run writes `reports/spatial_resolution_study_v0_1_0/` and
`models/spatial_resolution_study_v0_1_0/` (both Git-ignored). The results are
recorded in `EXP-0011` and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.16.

## Quick start: local structural feature study

M6C asks whether local structural features computed from the corrupted image
alone add information beyond the global eight while preserving clean-reference
and original-task performance. The extractor accepts one corrupted image and
nothing else; the no-reference boundary is checked structurally on every run.

```bash
uv sync --extra dev --extra image --extra ml

uv run --extra image --extra ml captcha-local-structure-study run \
    --config configs/local_structure_study_v0_1_0.toml

# Verify the written run: boundary, artifact hashes, prediction replay and protocol
uv run --extra image --extra ml captcha-local-structure-study verify \
    --config configs/local_structure_study_v0_1_0.toml
```

The run writes `reports/local_structure_study_v0_1_0/` and
`models/local_structure_study_v0_1_0/` (both Git-ignored). The results are
recorded in `EXP-0012` and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.17.

## Quick start: multi-scale structure-tensor study

M6D asks whether compact multi-scale structure-tensor features computed from the
corrupted image alone add information beyond the global eight while preserving
clean-reference and original-task performance. The extractor accepts one
corrupted image and nothing else; the no-reference boundary is checked
structurally on every run.

```bash
uv sync --extra dev --extra image --extra ml

uv run --extra image --extra ml captcha-structure-tensor-study run \
    --config configs/structure_tensor_study_v0_1_0.toml

# Verify the written run: boundary, artifact hashes, prediction replay and protocol
uv run --extra image --extra ml captcha-structure-tensor-study verify \
    --config configs/structure_tensor_study_v0_1_0.toml
```

The run writes `reports/structure_tensor_study_v0_1_0/` and
`models/structure_tensor_study_v0_1_0/` (both Git-ignored). The results are
recorded in `EXP-0013` and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §5.18.

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
- An isolated occlusion/interference pilot (`interference.py`,
  `robustness_pilot.py`, `configs/robustness_pilot_occlusion_v0_1_0.toml`):
  thin lines and curved arcs, nested deterministic stroke pools, exact clean
  re-rendering, geometry replay, grouped bootstrap, feature contrast against
  blur/noise, and a predeclared feasibility gate.
- A grouped robustness benchmark (`robustness_benchmark.py`,
  `configs/robustness_benchmark_v0_1_0.toml`): frozen-M3 zero-shot evaluation,
  expanded-data conventional grid, validation-only selection, source-grouped
  bootstrap, clean-reference handling, data accounting, target-distribution
  comparison and a run/verify CLI.
- An isolated geometric-distortion pilot (`geometric.py`,
  `geometric_pilot.py`, `configs/geometric_distortion_pilot_v0_1_0.toml`):
  seeded smooth elastic fields, bilinear resampling with edge padding, identity/
  zero-field/minimal controls, field replay, grouped uncertainty, feature
  contrasts against blur/noise/interference, and a predeclared gate.
- A grouped geometric-robustness benchmark (`geometric_benchmark.py`,
  `configs/geometric_robustness_benchmark_v0_1_0.toml`): frozen-M3 zero-shot
  evaluation on the M5A test sources, controlled conventional adaptation with
  validation-only selection, original-task retention on the untouched M3 test
  split, the 0.5 px rung held out as a diagnostic, predeclared
  integrity/leakage checks, source-grouped uncertainty and a run/verify CLI.
- A retention-aware selection and calibration study (`retention_aware.py`,
  `configs/retention_aware_selection_v0_1_0.toml`): six predeclared candidate
  policies, a univariate affine calibration fitted only on out-of-sample
  validation predictions with leave-one-fold-out validation scoring, a
  predeclared retention-aware selection rule, per-source consistency analysis
  and a run/verify CLI.
- A retention-aware training-objective study (`retention_aware_training.py`,
  `configs/retention_aware_training_v0_1_0.toml`): four predeclared candidate
  training policies with a fixed model family, metadata-only sample weighting
  normalised to mean one, a predeclared eligibility-and-score rule with a
  per-severity constraint, and a run/verify CLI.
- A spatial-aware feature study (`spatial_features.py`, `spatial_study.py`,
  `configs/spatial_feature_study_v0_1_0.toml`): sixteen localized grid features
  in a separate versioned module, three predeclared ablations with a fixed
  model family and training partition, a grid-resolution diagnostic, and a
  run/verify CLI.
- A spatial resolution study (`spatial_resolution_study.py`,
  `configs/spatial_resolution_study_v0_1_0.toml`): four predeclared grid
  resolutions in two representations with the feature definitions unchanged,
  training-only redundancy and overfitting diagnostics, and a run/verify CLI.
- Model persistence (`joblib` + JSON cards) with a `verify-model` command that
  rejects mismatched feature schemas, targets, dataset/split hashes, artifact
  hashes and incompatible scikit-learn major/minor versions.
- 488 tests in total, all passing, including byte-identical dataset
  regeneration, tamper detection, split isolation, feature edge cases,
  train-only preprocessing, validation-only selection, grouped bootstrap,
  held-out-font isolation, persistence checks, interference determinism and
  geometry replay, pilot tamper detection, M4B source isolation/merge checks,
  frozen-model zero-shot replay, adaptation determinism, elastic-field
  determinism/replay/validation, M5A manifest/feature reproducibility and the
  M5B split/merge contracts, 0.5 px exclusion, feature-leakage guard, metric
  correctness, grouped bootstrap and model-replay checks, plus the M5C
  validation-only selection, calibration fit-data isolation, test immutability,
  source-group isolation, slice-score calculation, feature-schema
  preservation, calibration serialization/reload, prediction replay and
  deterministic-report tests, the M5D weight validity and normalisation,
  target-independence, group-assignment, training isolation, diagnostic
  exclusion, eligibility, tie-breaking, fallback, control-reproduction and
  deterministic-report tests, the M6A feature names/order/dtypes, blank and
  degenerate images, determinism, metadata-leakage, schema immutability,
  split-integrity, diagnostic exclusion, selection and serialization tests, and
  the M6B per-resolution feature counts, ordering, region boundaries, uneven
  dimensions, blank and empty regions, determinism, leakage, selection
  isolation, fallback and deterministic-report tests, and the M6C exact local
  feature schema/order, deterministic extraction, uniform and empty-edge
  degeneracy, finite/valid ranges, a structural no-reference signature and AST
  scan, metadata-leakage, split-integrity, 0.5 px exclusion, validation-only
  selection, champion fallback, model-reload equivalence and deterministic
  tables/report/model-bundle tests, and the M6D exact structure-tensor
  schema/order, two-scale manifest, known-orientation and isotropic-noise
  behaviour, blank/uniform and near-zero-energy degeneracy, finite/valid ranges,
  structural no-reference signature and AST scan, metadata-leakage,
  split-integrity, 0.5 px exclusion, validation-only selection, champion
  fallback, model-reload equivalence and deterministic tables/report/model-bundle
  tests.

Milestone 5A findings (synthetic image similarity only — **not** human
readability or solvability):

- The smooth elastic-warp pilot is deterministic, exactly replayable, and
  non-destructive. Identity and zero-field controls reproduce all 60 clean
  references exactly; the 0.25 px minimal control has mean SSIM 0.998.
- SSIM falls monotonically on **60/60** source trajectories (Spearman -1.0);
  0/180 adjacent steps increase. Mean SSIM is 0.992, 0.971, 0.907 and 0.837 at
  0.5, 1, 2 and 3 px maximum displacement. No elastic output is near-identity
  or degenerate, and no elastic image duplicates another.
- At severity 4 the eight features are strongly separated from blur, noise,
  thin lines and arcs (mean |Cohen's d| 4.28–6.49; 4–8 features above |d| = 0.8
  per contrast). Pooled feature/target correlations mix severity and are
  reported alongside within-severity correlations, not treated as stable.
- Gate: **CONDITIONAL PASS**. The only failed margin is between-source SD at
  0.5 px (0.0025 versus 0.005), which is close to the interpolation floor.
  This is not an unqualified pass and does not justify a separate geometric
  benchmark without revising or dropping the lowest rung.

Milestone 5B findings (synthetic image similarity only — **not** human
readability or solvability):

- **Zero-shot transfer fails on the new corruption family.** On the primary
  1.0–3.0 px geometric test slice (36 rows / 12 sources) the frozen M3 model
  reaches MAE 0.0777, RMSE 0.0967 and R² -1.1858 with a mean residual of
  -0.0717, against MAE 0.2030, RMSE 0.2132 and R² -9.6363 for the M3
  training-mean baseline. It is far better than the constant but badly
  mis-calibrated in absolute terms. On the original M3 blur/noise test rows the
  same model reaches MAE 0.0157 and R² 0.9865, so this is a cross-corruption
  transfer failure, not a general model failure.
- **Controlled adaptation reduces aggregate error but not the calibration
  failure.** A validation-selected histogram gradient boosting model (12
  predeclared candidates) reaches MAE 0.0544, RMSE 0.0659 and R² -0.0165 on the
  same rows; the paired source-grouped RMSE difference is -0.0307 (preliminary
  interval -0.0472 to -0.0155), a 31.8% reduction. R² stays near zero, so the
  aggregate error falls without the model explaining between-source variation at
  fixed severity.
- **The gain is not uniform across severity.** Adapted MAE is worse at 1.0 px
  (0.0520 versus 0.0259 for frozen M3) and better at 2.0 px (0.0363 versus
  0.0662) and 3.0 px (0.0749 versus 0.1409). At the diagnostic 0.5 px rung the
  two models are nearly indistinguishable (0.0170 versus 0.0171).
- **The original task is largely retained in aggregate, but not on clean
  references.** On the untouched M3 test rows the adapted model reaches MAE
  0.0179, RMSE 0.0243 and R² 0.9843 versus 0.0157/0.0226/0.9865 for frozen
  M3. The aggregate paired RMSE difference includes zero (0.0017, interval
  -0.0022 to 0.0042), but the clean-reference slice degrades clearly (MAE
  0.0134 versus 0.0004), so the adapted model is not a drop-in replacement.
- **Decision.** Conventional feature-based ML remains the benchmark family.
  Zero-shot failure with successful adaptation is a distribution-shift result
  that representative training data addresses in this synthetic setting. The
  clean-calibration trade-off and the near-zero R² are unresolved; the next
  step would be a predeclared retention-aware selection or calibration
  protocol, **not** a larger model. A new corruption family does not by itself
  justify a CNN.
- **Uncertainty.** Every interval resamples whole source groups and there are
  only 12 held-out test sources, so all intervals are preliminary and
  descriptive. Interval overlap or non-overlap is not treated as a
  significance test, and no universal threshold was introduced after seeing
  test results.

Milestone 5C findings (synthetic image similarity only — **not** human
readability or solvability):

- **The trade-off was resolved, mostly by the model family.** A retention-aware
  rule over six predeclared candidates selects an expanded random forest that
  cuts primary geometric RMSE by **39.0%** (0.0590 versus the frozen 0.0967)
  and is the only candidate with a positive geometric R² (0.1855). The frozen
  M3 model reaches 0.0967 with R² -1.186.
- **The original task is retained.** On the untouched M3 test rows the selected
  policy reaches MAE 0.0156, RMSE 0.0220 and R² 0.9872, against the frozen
  0.0157 / 0.0226 / 0.9865. The paired source-grouped RMSE difference is
  -0.0006 with a preliminary interval of -0.0024 to 0.0005, and the
  per-source gains and losses nearly cancel.
- **The cost is small but real.** The clean-reference slice degrades from RMSE
  0.0012 to 0.0023 (paired difference 0.0011, interval 0.0006 to 0.0017), and
  the selected model is worse than the frozen model at the mildest included
  rung (1.0 px MAE 0.0570 versus 0.0259).
- **Calibration did not do the work.** The affine calibration was tested on
  every base model and selected for none of them. It reduced the M5B HGB's
  clean-reference offset (mean residual +0.0122 to +0.0038), confirming the
  error is partly a systematic offset, but a single global affine map could not
  remove it without moving the other slices.
- **The improvement is broad-based.** The geometric gain holds on 11 of 12
  held-out sources, and the single best source accounts for 22.9% of the total
  gain.
- **Decision.** Conventional feature-based ML remains the benchmark champion.
  The next question, if the line continues, is whether a retention-aware
  *training* objective rather than a post-hoc selection or calibration policy
  can remove the remaining small clean-reference cost. The evidence does not
  justify a CNN, and a new corruption family does not by itself justify one.
- **Uncertainty.** Every interval resamples whole source groups and there are
  only 12 held-out test sources, so all intervals are preliminary and
  descriptive. Interval overlap or non-overlap is not treated as a
  significance test, and no universal threshold was introduced after seeing
  test results.

Milestone 5D findings (synthetic image similarity only — **not** human
readability or solvability):

- **No weighted candidate beat the champion.** The M5C champion is retained.
  This is reported as a negative result rather than a manufactured winner.
- **The control is bit-identical to the champion.** The unweighted retrained
  control reproduces the frozen M5C bundle's predictions exactly and has the
  same artifact SHA-256, so every difference below is attributable to the
  sample weighting alone.
- **The per-severity constraint earned its keep.** The severity-aware scheme
  improves the severe geometric rung (severity 4 RMSE 0.0537 versus the
  champion's 0.0624) but degrades the mild one (severity 2 RMSE 0.0715 versus
  0.0525, 36% worse) while its *aggregate* geometric RMSE barely moves
  (+0.0005 on test). Selecting on the aggregate alone would have hidden a real
  regression on the 1.0 px rung; the predeclared per-severity constraint is
  what rejects it.
- **The retention-aware scheme trades rather than wins.** It is eligible but
  scores +0.0062 against the champion's 0: it improves the clean slice
  (validation RMSE 0.0018 versus 0.0022) at the cost of the original task
  (+0.0013) and the aggregate geometric slice (+0.0007). On test it reaches
  geometric RMSE 0.0580 and clean RMSE 0.0019, but the original-task RMSE rises
  to 0.0241 against the champion's 0.0233.
- **Why weighting did not help.** Within one model family, one shared feature
  representation and one shared sample of 432 training rows, reweighting moves
  error between slices instead of removing it.
- **The representation is the likely constraint.** The eight image-only
  features are global whole-image statistics that do not describe *where* on
  the canvas a distortion occurs, and a geometric warp moves ink locally while
  leaving many of them nearly unchanged. This is consistent with the near-zero
  between-source R² that M5B and M5C both observed, and reweighting cannot add
  spatial information the representation never exposed. It is a hypothesis
  consistent with the evidence, not an isolated cause.
- **Decision.** Conventional feature-based ML remains the benchmark champion
  and the M5C champion remains the best geometric-robustness policy found so
  far. The evidence does not justify a CNN, and a new corruption family does
  not by itself justify one.
- **Uncertainty.** Every interval resamples whole source groups and there are
  only 12 held-out test sources, so all intervals are preliminary and
  descriptive. Interval overlap or non-overlap is not treated as a
  significance test, and no universal threshold was introduced after seeing
  test results.

Milestone M6A findings (synthetic image similarity only — **not** human
readability or solvability):

- **No spatial ablation was selected.** The frozen M5C champion is retained.
  This is reported as a negative result rather than a manufactured winner.
- **The combined ablation traded rather than won.** Adding the 16 spatial
  features to the 8 global ones improved the aggregate geometric validation
  RMSE by only 0.00056 while regressing the original task by 15.2%, so it is
  ineligible under the predeclared tolerance. On test its geometric RMSE is
  0.0584 against the champion's 0.0590, but its paired interval includes zero,
  while the original-task and M3-retention regressions exclude zero.
- **The spatial-only ablation is far worse everywhere** (test geometric RMSE
  0.0930 against 0.0590), because it discards the global summary that actually
  tracks blur and noise.
- **A grid-resolution diagnostic explains why.** On training rows, a 4×2
  partition of the 192×64 canvas (16×96-pixel cells) resolves blur about **25
  times** more strongly than the elastic family: the mean absolute per-cell
  ink-occupancy change against the same source's clean reference is 0.136 for
  blur and 0.005 for a 1–3 px warp. A warp that small moves ink by a fraction
  of a cell, so a coarse partition averages the effect away and the spatial
  features mostly re-encode how much ink the image has — which the global
  features already state.
- **What this rules in and out.** The negative result is specific to the
  declared resolution: it shows that a *coarse* spatial summary does not
  help, not that localization is useless. The remaining headroom is a
  resolution question, not an objective question.
- **Decision.** Conventional feature-based ML remains the benchmark champion
  and the M5C champion remains the best geometric-robustness policy found so
  far. The evidence does not justify a CNN, and a new corruption family does
  not by itself justify one.
- **Uncertainty.** Every interval resamples whole source groups and there are
  only 12 held-out test sources, so all intervals are preliminary and
  descriptive. Interval overlap or non-overlap is not treated as a
  significance test, and no universal threshold was introduced after seeing
  test results.

Milestone M6B findings (synthetic image similarity only — **not** human
readability or solvability):

- **No spatial candidate qualified.** Every combined candidate breached the
  10% original-task tolerance and every spatial-only candidate breached it
  plus the clean-slice and per-severity tolerances, so the frozen M5C champion
  is retained. This is reported as a negative result rather than a manufactured
  winner.
- **The finer grid does resolve the warp.** On training rows the per-cell
  ink-occupancy change against the same source's clean reference rises from
  0.0042 at 2×2 to 0.0107 at 8×8, and its correlation with the SSIM drop rises
  from **+0.45 to +0.83**, closing most of the gap to blur's +0.89. M6A's
  diagnosis was right and the obvious follow-up still fails.
- **The model cannot use the information.** No resolution produces a
  demonstrable geometric gain — every paired geometric interval includes zero —
  and the aggregate geometric validation RMSE is *best* at the coarsest grid.
  Meanwhile the retention slices degrade monotonically with the feature count,
  because the forest considers every feature at each split and the extra
  spatial features are informative about the geometric family but largely
  uninformative about blur, noise and clean references.
- **Redundancy is not the mechanism.** Mean absolute pairwise correlation
  *falls* with resolution (0.59 at 2×2 to 0.26 at 8×8), and no feature is
  constant or duplicated at any grid, while the overfitting gap grows.
- **Cross-milestone check.** The 4×2 candidates reproduce the M6A 4×2 candidates
  bit-for-bit (max difference 0.0e+00 on every test slice), confirming M6B is a
  strict extension of M6A.
- **What this rules in and out.** The blocker is the **relevance** of the
  spatial features to the other corruption families, not the resolution of the
  partition. The next defensible question is whether a feature specifically
  sensitive to local *displacement*, rather than to local ink quantity, can add
  information without diluting the splits that support blur, noise and clean
  references.
- **Decision.** Conventional feature-based ML remains the benchmark champion
  and the M5C champion remains the best geometric-robustness policy found so
  far. The evidence does not justify a CNN, and a new corruption family does
  not by itself justify one.
- **Uncertainty.** Every interval resamples whole source groups and there are
  only 12 held-out test sources, so all intervals are preliminary and
  descriptive. Interval overlap or non-overlap is not treated as a
  significance test, and no universal threshold was introduced after seeing
  test results.

Milestone M6C findings (synthetic image similarity only — **not** human
readability or solvability):

- **No local-structure candidate was selected.** The frozen M5C champion is
  retained. Both local candidates are ineligible on validation: the combined
  candidate breaches the 50% clean-reference tolerance (validation RMSE 0.00621
  versus 0.00223), and the local-only candidate breaches the 10% original-task
  tolerance (0.02850 versus 0.02007), the 25% severity-3 tolerance (0.03145
  versus 0.02494) and the 25% severity-4 tolerance (0.08451 versus 0.06240).
- **The combined candidate is neutral on the geometric aggregate, not better.**
  Its test geometric RMSE is 0.06338 versus 0.05901 for the champion (paired
  difference +0.00438, preliminary interval [-0.00015, 0.00891]). The local-only
  candidate is 0.06612 (+0.00711, [0.00165, 0.01204]).
- **The geometric effect is mixed across the ladder.** Both candidates improve
  the 1 px rung (RMSE 0.04277 local-only and 0.05355 combined versus 0.06187)
  and regress the 2 px and 3 px rungs; the 1 px paired intervals include zero,
  so the only apparent gain is not clear.
- **Retention clearly degrades.** On test, the clean-reference RMSE is 0.02021
  (local-only) and 0.00931 (combined) versus 0.00228; the original-task RMSE is
  0.03600 and 0.03112 versus 0.02333; the M3 overall RMSE is 0.03461 and 0.02950
  versus 0.02201. The paired clean, original-task and M3 intervals exclude zero
  for both candidates.
- **The local family is largely redundant with the global eight.** The maximum
  absolute pairwise correlation among the local features is 0.9926 (the four
  magnitude statistics duplicate one another), and the maximum absolute
  local-versus-global correlation in the combined candidate is 0.8743.
  Random-forest importance gives the combined candidate a local share of 0.2283,
  with `histogram_entropy` alone at 0.6303 and `grad_mag_std` at 0.1521.
- **The no-reference boundary is structural, not cosmetic.** The extractor
  signature accepts one corrupted image and an optional spec; an AST scan of the
  feature module confirms no target, reference, source, severity, font, split or
  path identifier, no dynamic import and no file I/O. Tests repeat both checks
  independently, and an opaque `sample_id` never changes a feature value.
- **What this rules in and out.** Local gradient strength and orientation, as
  declared here, do not add usable information beyond the global summary while
  preserving retention. The result does not rule out every local
  representation; it specifically weakens the case for whole-image summaries of
  a local operator. A representation explicitly sensitive to local
  *displacement* (for example multi-scale structure-tensor anisotropy or local
  autocorrelation) is the most defensible next handcrafted question.
- **Decision.** Conventional feature-based ML remains the benchmark champion
  and the M5C champion remains the best geometric-robustness policy found so
  far. The evidence does not justify a CNN, and a new corruption family does
  not by itself justify one.
- **Uncertainty.** Every interval resamples whole source groups and there are
  only 12 held-out test sources, so all intervals are preliminary and
  descriptive. Interval overlap or non-overlap is not treated as a
  significance test, and no universal threshold was introduced after seeing
  test results.

Milestone M6D findings (synthetic image similarity only — **not** human
readability or solvability):

- **No structure-tensor candidate was selected.** The frozen M5C champion is
  retained. The combined candidate is ineligible on validation because its
  clean-reference RMSE 0.00516 exceeds the champion's 0.00223 by more than the
  50% tolerance; the structure-only candidate breaches the clean-reference
  tolerance (0.01540 versus 0.00223) and the original-task tolerance (0.03315
  versus 0.02007).
- **The combined candidate is neutral on the geometric aggregate, not better.**
  Its test geometric RMSE is 0.06036 versus 0.05901 for the champion (paired
  difference +0.00135, preliminary interval [-0.00390, 0.00575]). The
  structure-only candidate is 0.06532 (+0.00631, [-0.00019, 0.01181]).
- **The geometric effect is mixed across the ladder.** The combined candidate
  improves the 1 px rung (RMSE 0.04293 versus 0.06187, paired -0.01894
  [-0.03022, -0.00620]), regresses the 2 px rung (0.03848 versus 0.02362,
  +0.01486 [0.00749, 0.02226]) and the 3 px rung (0.08721 versus 0.07784,
  +0.00937 [-0.00268, 0.01907], interval includes zero).
- **Retention is worse.** On test, the combined candidate's clean-reference RMSE
  is 0.00328 versus 0.00228, original-task RMSE 0.02526 versus 0.02333 and M3
  overall RMSE 0.02384 versus 0.02201. The structure-only candidate is worse on
  every aggregate slice: clean 0.01822, original 0.03879 and M3 0.03707, with
  paired intervals that exclude zero.
- **The new family is strongly redundant.** The maximum absolute pairwise
  correlation among the nine new features is 0.9977 (`st_s2_coh_mean` versus
  `st_s2_aniso_mean`); the energy summaries are also nearly interchangeable
  across scales (0.9820-0.9935). The maximum new-versus-global correlation is
  0.9387 (`st_s1_coh_mean` versus `ink_fraction`). Random-forest importance
  gives the combined candidate a structure-tensor share of 0.1362, with
  `histogram_entropy` alone at 0.6292.
- **The no-reference boundary is structural, not cosmetic.** The extractor
  signature accepts one corrupted image and an optional spec; an AST scan of the
  feature module confirms no target, reference, source, severity, font, split or
  path identifier, no dynamic import and no file I/O. Tests repeat both checks
  independently, and an opaque `sample_id` never changes a feature value.
- **What this rules in and out.** Two-scale structure-tensor coherence,
  anisotropy and energy summaries do not add usable information beyond the
  global eight while preserving retention. The M6A-M6D series now covers coarse
  regional summaries, finer regional grids, whole-image gradient summaries and
  multi-scale structure-tensor summaries; none adds a retention-safe geometric
  gain. A learned representation is the remaining distinct family, but it is a
  separate milestone and is not justified by this result alone.
- **Decision.** Conventional feature-based ML remains the benchmark champion
  and the M5C champion remains the best geometric-robustness policy found so
  far. The evidence does not justify a CNN on this result alone.
- **Uncertainty.** Every interval resamples whole source groups and there are
  only 12 held-out test sources, so all intervals are preliminary and
  descriptive. Interval overlap or non-overlap is not treated as a significance
  test, and no universal threshold was introduced after seeing test results.

Milestone 4B findings (synthetic image similarity only — **not** human
readability or solvability):

- **Zero-shot absolute calibration does not transfer.** The frozen M3 model
  scored MAE/RMSE/R² **0.1295 / 0.1612 / -1.0455** on the 12 held-out M4A test
  sources. Clean references remained near-perfect (MAE 0.0004), but corrupted
  rows were systematically under-predicted; on corrupted rows alone the paired
  MAE difference from the M3 training mean was -0.0060 with 95% interval
  [-0.0307, 0.0129], which includes zero.
- **Controlled conventional adaptation worked.** Validation selected
  `random_forest_depth_unlimited_leaf_1` (validation RMSE 0.0201). Test
  MAE/RMSE/R² was **0.0131 / 0.0178 / 0.9752**, an 89.0% RMSE reduction versus
  frozen M3; the paired RMSE interval was [-0.1558, -0.1256]. The expanded
  training-mean baseline scored 0.1261 / 0.1466 / -0.6912 on the same rows.
- **The M3 result was preserved.** On the original M3 test rows the retrained
  model scored 0.0158 / 0.0239 / 0.9849, versus 0.0157 / 0.0226 / 0.9865 for
  frozen M3.
- **The target shift is disclosed.** Expanded training targets had mean 0.7587
  versus 0.8524 on M4A test; the lower training mean comes from severe M3
  blur/noise rows. No test row informed feature selection, scaling,
  hyperparameters or calibration.
- Intervals use only 12 source groups and are preliminary. The result supports
  conventional modelling for this benchmark; it is not evidence that a CNN
  would fail on a harder or human-labelled shift.

Milestone 4A findings (synthetic image similarity only — **not** human
readability or solvability):

- The occlusion/interference pilot passed all five predeclared gate criteria:
  deterministic and exactly replayable, non-degenerate, monotone in density,
  feature-distinct from blur/noise, and duplicate-free.
- 60 sources × (1 clean + 2 variants × 4 severities) = 540 samples. Mean SSIM
  fell from 0.943 to 0.662 for `thin_lines` and 0.947 to 0.715 for
  `arc_strokes`; 0/360 adjacent steps increased.
- The pilot does **not** show the family is harder for humans or more
  solvable/unsolvable. It is evidence for proceeding to a separate grouped
  robustness benchmark, not a benchmark result.

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

Not implemented (planned): the CNN comparison, the broader M5 error taxonomy,
notebooks, the Streamlit dashboard, deployment, and any human-solvability or
production-CAPTCHA claims.
The conventional model's performance is a synthetic image-similarity result, not
a readability estimate.

## License

Released under the [MIT License](LICENSE).
