# CAPTCHA Quality & Robustness Analyzer

A reproducible, end-to-end computer-vision project for generating synthetic
CAPTCHA-style text images, applying controlled visual distortions, measuring
image quality, and modelling how those distortions affect image degradation.

> **Status:** Milestone 0 — repository foundation only. The generator, feature
> extraction, models, and dashboard are **not implemented yet**. See
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

**In scope (planned):**

- A deterministic synthetic text-image renderer with recorded metadata.
- A configurable distortion engine with multiple corruption families and
  severity levels.
- Reference-based image-quality targets (for example SSIM, MS-SSIM, PSNR) and
  feature extraction.
- Leakage-aware conventional ML baselines and a CNN comparison.
- Robustness/error analysis grouped by corruption family and severity.
- A Streamlit dashboard for exploring results.

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
- **No deep-learning results yet.** PyTorch is a planned dependency, not an
  implemented component.

## Planned architecture

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

Each stage is planned to write artifacts with an explicit schema and a recorded
configuration hash, so a result can be traced back to the exact data and
parameters that produced it.

## Technology stack

| Layer | Tools | Status |
| --- | --- | --- |
| Language | Python >= 3.11 | Configured |
| Data handling | NumPy, pandas | Core dependencies |
| Image I/O and rendering | Pillow | Core dependency |
| Image processing | OpenCV, scikit-image | Optional extra `image` (planned use) |
| Conventional ML | scikit-learn | Optional extra `ml` (planned use) |
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
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

### Option B — standard venv and pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
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
- Datasets, checkpoints, and generated reports stay out of Git; their manifests
  and generation configs are versioned. See [`data/README.md`](data/README.md).
- Results are recorded in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)
  with the exact configuration and dataset version used.

## Planned milestones

| Milestone | Deliverable | Status |
| --- | --- | --- |
| M0 | Repository structure, tooling, documentation foundation | **Complete** |
| M1 | Target definition and validation: confirm the reference-based quality target is stable, sensitive, and worth modelling | Planned |
| M2 | Deterministic renderer, distortion engine, manifest schema, feature extraction | Planned |
| M3 | Conventional ML baselines with leakage-safe grouped evaluation | Planned |
| M4 | CNN comparison (small CNN and transfer learning) under the same splits | Planned |
| M5 | Robustness analysis by corruption family/severity and error taxonomy | Planned |
| M6 | Streamlit dashboard, reproducibility package, final report | Planned |

Milestone 1 is a validation gate: if the proposed quality target proves unstable
or trivially predictable from generation metadata, the modelling strategy will
be revised before dataset scale-up.

## Current status

Implemented:

- Repository structure, MIT license, `.gitignore`, and contributor-facing docs.
- `pyproject.toml` with a `src/` layout, dependency extras, pytest, coverage, and
  Ruff configuration.
- Importable `captcha_quality` package containing only `__version__`.
- Configuration smoke tests that verify packaging metadata and layout.

Not implemented (planned): synthetic renderer, distortion engine, image-quality
feature extraction, datasets, ML models, notebooks, dashboard, and any results.

No experiments have been run and no performance numbers are reported.

## License

Released under the [MIT License](LICENSE).
