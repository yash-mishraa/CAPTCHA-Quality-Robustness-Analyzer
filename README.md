# CAPTCHA Quality & Robustness Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://captcha-robustness-analyzer.streamlit.app)
[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/yash-mishraa/CAPTCHA-Quality-Robustness-Analyzer/releases/tag/v1.0.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

> **🌐 Live Research Dashboard:** [captcha-robustness-analyzer.streamlit.app](https://captcha-robustness-analyzer.streamlit.app)  
> Interactive results-only public demo exploring benchmark predictions, distortion profiles, and visual quality error analysis.

A reproducible research repository for measuring how controlled visual
degradations affect **synthetic CAPTCHA-style images**, and for testing whether
a conventional image-quality model can predict that degradation.

> **Release status:** `v1.0.0` research release. Milestones M0–M9C are complete.
> M9C selected **Path A — stop experimentation and package the research**.
> Active experimentation is closed. M9B remains a preregistered but unexecuted
> protocol, and H1, H2 and H3 remain `not_tested`.

The project is deliberately narrower than the name may suggest. It predicts a
reference-based image-similarity target. It does **not** measure human
readability, CAPTCHA solvability, CAPTCHA security or production robustness.

## Research question

Can a small set of deterministic, whole-image features predict the structural
similarity of a degraded synthetic text image to its corresponding clean
reference, and how does that prediction behave when the corruption family,
severity or font changes?

The primary target is **SSIM against the exact source-specific clean reference**.
PSNR is recorded as a diagnostic only. The model receives only eight image
features; it never receives the target, corruption metadata, font identity, text
length, source identifiers or split labels.

## Scope and non-claims

**In scope**

- Synthetic text rendering at 192×64 in 8-bit grayscale PNG.
- Controlled Gaussian blur, additive Gaussian noise, thin-line interference,
  arc-stroke interference and bounded elastic warp.
- Source-grouped train/validation/test evaluation.
- Conventional regression on eight documented image features.
- Zero-shot evaluation of one frozen model on held-out fonts and matched cases.
- Provenance, integrity, lifecycle and preregistration checks.
- Read-only error analysis of the frozen benchmark.

**Out of scope**

- Human readability, legibility, solvability or user-response prediction.
- CAPTCHA security, attack success, production robustness or deployment claims.
- Real CAPTCHA services, captured text or human-subject data.
- Causal claims about fonts, features or corruption mechanisms.
- A new model-fitting or benchmark programme. M9C explicitly closed that work.

## Contributions

| Contribution | What was delivered | Evidence |
| --- | --- | --- |
| Synthetic corpus and baseline | Deterministic renderer, corruption ladder, grouped splits, manifest/feature pipeline and a leakage-safe conventional benchmark | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md), [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) |
| Robustness evaluation | Separate pilots and grouped benchmarks for occlusion/interference and geometric warp; frozen-model zero-shot and controlled adaptation | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md), [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) |
| Feature and retention studies | Retention-aware selection and training objectives, plus four feature-family studies; no added feature family was retained | [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) |
| Cross-font evaluation | Held-out-font benchmark followed by a matched 10-font, 30-case, 300-source benchmark | [`reports/matched_font_benchmark_v0_1_0/report.md`](reports/matched_font_benchmark_v0_1_0/report.md) |
| Provenance and integrity | Independent M8C audit, M8E evidence contract, replay, lifecycle and ground-truth isolation checks | [`docs/BENCHMARK_INTEGRITY.md`](docs/BENCHMARK_INTEGRITY.md), [`reports/matched_font_benchmark_provenance_audit_v0_1_0/report.md`](reports/matched_font_benchmark_provenance_audit_v0_1_0/report.md) |
| Error analysis | Read-only analysis of where interference error concentrates and how the frozen model responds | [`reports/interference_error_analysis_v0_1_0/README.md`](reports/interference_error_analysis_v0_1_0/README.md) |
| Preregistration and closure | Three frozen hypotheses, explicit decision gates and a documented stop decision | [`reports/interference_experiment_protocol_v0_1_0/README.md`](reports/interference_experiment_protocol_v0_1_0/README.md), [`reports/research_prioritization_v0_1_0/README.md`](reports/research_prioritization_v0_1_0/README.md) |

The project's main methodological contribution is not a higher model score. It
is the combination of source-grouped evaluation, a frozen champion, matched
cross-font cases, fail-closed provenance checks and a preregistered stopping
point.

## Architecture and workflow

The repository contains several related runners rather than one monolithic
pipeline. The shared stages are:

```text
configs/*.toml
      |
      v
render.py ────────────────► clean reference PNG
      |                           |
      |                           +──────────────┐
      v                                          v
corruptions.py / interference.py / geometric.py  quality_metrics.py
      |                                          |
      v                                          v
degraded PNG ──► features.py ───────────────► ssim_target (PSNR diagnostic)
      |                                          |
      +──────────────┬───────────────────────────+
                     v
        dataset.py + splits.py
        manifest.csv / sources.csv / features.csv / summary / images
                     |
      ┌──────────────┼─────────────────────────────┐
      v              v                             v
 benchmark.py   robustness / geometric      matched_font_benchmark.py
 (M3, M5C)       benchmarks (M4B, M5B)       (M8C, frozen zero-shot)
      |              |                             |
      +──────────────┴──────────────┬──────────────+
                                    v
                     metrics, tables, reports, predictions
                                    |
                                    v
             provenance + lifecycle + replay + integrity verifiers
             (M8D, M8E, M8C verify, M9B validator)
```

Key points:

- **The clean image is the reference.** Every SSIM/PSNR target is computed
  against the exact clean image for that source.
- **Splits are assigned by clean source.** All variants of a source stay in one
  split. The independent unit is the source group, not the image.
- **Features see only the degraded image.** They never see the clean reference,
  targets or generation metadata.
- **The M5C champion is frozen.** The M7B, M8C and M9A stages do not fit,
  calibrate, tune or replace it.
- **The M8C pipeline is frozen at its historical revision.** The M8C run
  recorded the honest revision `75c8ef4-dirty`; the historical record is not
  rewritten as a clean run.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the component map and
[`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) for the system
boundaries and navigation guide.

## Repository map

```text
configs/                         committed experiment and protocol configs
docs/                            methodology, architecture, log, release record
reports/                         committed final evidence packages
src/captcha_quality/             implementation
tests/                           unit, integration and documentation checks
data/                            generated datasets (Git-ignored)
models/                          generated model artifacts (Git-ignored)
```

Important source modules:

| Module | Responsibility |
| --- | --- |
| `render.py` | Deterministic clean text rendering and reference images |
| `corruptions.py` | Gaussian blur and additive Gaussian noise |
| `interference.py` | Thin-line and arc-stroke interference |
| `geometric.py` | Bounded smooth elastic warp |
| `quality_metrics.py` | SSIM and PSNR against the clean reference |
| `features.py` | The eight documented whole-image features |
| `dataset.py` | Dataset generation, manifests, feature tables and verification |
| `splits.py` | Deterministic source-grouped split assignment |
| `benchmark.py` / `modelling.py` | M3 conventional benchmark, grouped metrics and frozen-model inference |
| `robustness_benchmark.py` / `geometric_benchmark.py` | M4B and M5B grouped robustness benchmarks |
| `retention_aware*.py` | M5C selection/calibration study and M5D training-objective study |
| `spatial_*.py`, `local_structure*.py`, `structure_tensor*.py` | M6A–M6D feature studies |
| `heldout_font_benchmark.py`, `cross_font_analysis.py` | M7B and M7C font evaluation |
| `matched_font_benchmark.py` | M8C matched cross-font execution and replay |
| `benchmark_lifecycle.py`, `benchmark_preflight.py` | M8A/M8B output lifecycle and generator preflight |
| `benchmark_integrity.py` | M8E historical-evidence and future-execution checks |
| `interference_error_analysis.py` | M9A read-only error analysis |
| `interference_experiment_protocol.py` | M9B static protocol validator |

Committed evidence packages under `reports/` are the M8C benchmark, the M8D
provenance audit, the M9A error analysis, the M9B preregistered protocol and the
M9C closure decision. Earlier M3–M7 reports and datasets are generated locally
and are Git-ignored; they are not present in a fresh clone.

## Setup

Python **3.11+** is required. The historical benchmark environment was CPython
**3.13.16 on macOS 27.0.1 (arm64)** with the pinned library versions recorded in
the M8C metadata. Determinism is demonstrated only in that environment.

Using `uv` (recommended):

```bash
uv sync --extra dev --extra image --extra ml
```

Or a standard virtual environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,image,ml]'
```

The `deep-learning` and `app` extras exist in `pyproject.toml`, but they are not
used by the completed research. No CNN or dashboard result is part of this
release.

## Verification and tests

These commands are read-only with respect to the research evidence:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .

uv run python -m captcha_quality.benchmark_integrity verify
uv run python -m captcha_quality.benchmark_lifecycle verify \
  --config configs/benchmark_lifecycle_v0_3_0.toml --state post-execution
uv run python -m captcha_quality.matched_font_benchmark verify \
  --lifecycle-post configs/benchmark_lifecycle_v0_3_0.toml
uv run python -m captcha_quality.interference_experiment_protocol verify

git diff --check
```

The M8E, M8C and M9B checks expect the recorded local evidence workspace,
including the Git-ignored datasets, model artifact and pinned font files. They
fail closed when those inputs are absent. That is a missing-input condition, not
permission to weaken or skip a check. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full boundary.

**Artifact-generating commands are not part of the release check.**
`captcha-dataset generate`, `captcha-benchmark run`,
`captcha-robustness-pilot run`, `captcha-geometric-pilot run`,
`captcha-matched-font-benchmark run` and the study runners write images,
datasets, predictions, models or reports. They were not run for this release.
The M9A runner can write a new report and also was not run.

## Results

All figures below are observed on synthetic data. Intervals are descriptive
source-grouped or matched-case summaries, not significance tests. Do not compare
metrics from different benchmark compositions as if they were a controlled
like-for-like experiment.

| Stage | Evaluation context | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| M3 baseline, selected random forest | 108 test rows, 12 sources, familiar fonts, blur/noise + clean | 0.0157 | 0.0226 | 0.9865 |
| M5C champion, original M3 test slice | 108 rows, 12 sources | 0.0156 | 0.0220 | 0.9872 |
| M5C champion, geometric test slice | 36 rows, 12 sources, 1–3 px warp | 0.0493 | 0.0590 | 0.1855 |
| M5C champion, clean reference | 12 rows, 12 sources | 0.0013 | 0.0023 | n/a |
| M5C champion, original blur/noise | 96 rows, 12 sources | 0.0174 | 0.0233 | 0.9814 |
| M7B held-out fonts, all primary | 1,140 rows, 60 sources, Georgia/Monaco/Impact | 0.063427 | 0.092952 | 0.743537 |
| M7B held-out fonts, original blur/noise | 480 rows, 60 sources | 0.060075 | 0.095923 | 0.761361 |
| M7B held-out fonts, geometric primary | 180 rows, 60 sources | 0.047571 | 0.058839 | 0.265726 |
| M8C matched fonts, all primary | 5,700 rows, 300 sources, 10 fonts, 30 cases | 0.059607 | 0.088610 | 0.732898 |
| M8C matched fonts, original blur/noise | 2,400 rows, 300 sources | 0.032735 | 0.057131 | 0.899708 |
| M8C matched fonts, interference | 2,400 rows, 300 sources | 0.090825 | 0.118525 | −0.430242 |
| M8C matched fonts, thin lines | 1,200 rows, 300 sources | 0.102073 | 0.132114 | −0.575158 |
| M8C matched fonts, arc strokes | 1,200 rows, 300 sources | 0.079578 | 0.103162 | −0.298911 |
| M8C matched fonts, geometric primary | 900 rows, 300 sources | 0.048018 | 0.059687 | 0.162498 |

### What the results show

- **The baseline task is easy on its original synthetic split.** The selected
  random forest reaches R² 0.9865 on 12 held-out source groups. This is a
  within-renderer, within-font-family result, not broad generalisation.
- **Held-out fonts are harder, but the comparison is unmatched.** The M7B
  held-out-font blur/noise RMSE is 0.095923 versus 0.023328 on the recorded
  familiar-font slice. The datasets, fonts, cases and corruption mixtures are
  not matched, so this is a descriptive difference, not a font effect.
- **The matched M8C benchmark does not reproduce a uniform held-out penalty.**
  On 30 matched cases, the recorded all-primary RMSE is 0.092894 for the
  familiar group, 0.091626 for the M7B-held-out group and 0.082831 for the new
  held-out group. Font-level MAE ranges from 0.053974 to 0.077869. These are
  descriptive differences over seven observed held-out identities, not a
  font-population or causal result.
- **Thin-line and arc interference are the weak conditions.** M8C interference
  R² is −0.430242, thin-line R² is −0.575158 and arc R² is −0.298911. On those
  slices the model is worse than a condition-mean predictor in squared-error
  terms. The clean-reference and 0.5 px elastic rows are diagnostic and are
  excluded from primary metrics.
- **The error is broad, not a few outliers.** M9A finds that the top 5% of
  interference source groups account for 10.6% of interference squared error,
  compared with 51.5% for blur and 49.7% for noise. This is exploratory
  evidence from the observed M8C benchmark.
- **The response is compressed at severe interference.** M9A reports
  within-source thin-line slope 0.241 and prediction/target SD ratio 0.320;
  arc-stroke slope 0.337 and ratio 0.397. Large errors at `|residual| >= 0.20`
  occur on 15.17% of thin-line rows and 7.00% of arc rows, rising to 55.3% and
  28.0% at severity 4. These are descriptive, exploratory findings; the cause is
  not identified.

The frozen M5C champion is `expanded_random_forest` with calibration `none`.
Its artifact SHA-256 is
`c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0` and its
model-card SHA-256 is
`9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8`.

## Limitations

- **Synthetic target.** SSIM against a clean reference is an image-similarity
  proxy. It is not human readability, legibility, solvability or security.
- **One renderer and platform.** The benchmark uses a fixed 192×64 grayscale
  renderer on one macOS environment with pinned Apple-bundled fonts. No
  cross-platform, cross-renderer or cross-format result is claimed.
- **Constrained font roster.** Ten purposively selected fonts are not a random
  sample of typefaces. Seven held-out identities, each rendered with 30 matched
  cases, are the central font-generalisation limitation. The 300 M8C sources
  are not 300 independent fonts.
- **Confounding.** Font identity is entangled with glyph shape, font metrics,
  rasterisation, target difficulty and corruption response. M8C does not
  identify a causal font effect.
- **Limited independent groups.** M3 and M5C have 12 test source groups; M7B
  has 60; M8C has 300. Intervals describe those observed groups and are not a
  population estimate.
- **Diagnostic slices.** Clean-reference rows have a constant target and
  undefined R². The 0.5 px elastic rung is near-degenerate and is diagnostic
  only. Neither is part of a primary metric or the primary bootstrap.
- **Provenance boundaries.** M8C recorded the honest revision `75c8ef4-dirty`.
  M8D found six unpinned modules and a historical duplicate input-manifest key;
  these are documented rather than silently repaired. M8E verifies the
  historical evidence and adds a clean-tree guard for future execution.
- **No human or security validation.** There are no human labels, no solver
  data, no live CAPTCHA experiments and no production deployment evidence.
- **No confirmatory follow-up.** M9B is a preregistered design, not a result.
  Its three hypotheses remain `not_tested`.

## Research status

M9C selected **Path A — stop experimentation and package the research**. The
experiment phase is closed. H1 (attenuated response) is the first candidate only
if the project is formally reopened with new independent data and an approved
protocol. H2 (edge/sharpness feature conflation) and H3 (font/glyph difficulty)
are deprioritized. M9B was not executed, and no part of this release authorizes
a new experiment, dataset, model fit, calibration, feature change, benchmark
change or champion replacement.

The final release record is in
[`docs/RELEASE_NOTES_v1.0.0.md`](docs/RELEASE_NOTES_v1.0.0.md) and the closure
evidence is in
[`docs/RELEASE_CHECKLIST_v1.0.0.md`](docs/RELEASE_CHECKLIST_v1.0.0.md).

## Versioning

The Python package version is `1.0.0`; this is the documentation and release
version for the completed research study. Historical datasets, model cards,
protocols and reports keep their original `v0.1.0`, `v0.2.0` or milestone
labels. The release does not re-version historical scientific evidence.

## License

MIT. See [`LICENSE`](LICENSE). Font binaries are not redistributed. The
generated benchmark images and local model artifacts are not tracked in Git.
