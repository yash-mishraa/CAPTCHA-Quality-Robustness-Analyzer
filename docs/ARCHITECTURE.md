# Architecture and component map

**Status:** release architecture for the completed M0–M9C research.

This document describes the repository as implemented. It is not a planned
architecture. There is no single `run-all` command: the project contains a
shared data/feature layer and several separately versioned experiment runners.

## 1. Shared data path

```text
configs/*.toml
     │
     ▼
src/captcha_quality/render.py
     │  deterministic clean PNG + reference
     ├──────────────────────────────────────────────┐
     ▼                                              │
corruptions.py / interference.py / geometric.py     │
     │  degraded PNG                              │
     ▼                                              │
features.py ── eight whole-image features           │
     │                                              │
     │                                              ▼
     │                                  quality_metrics.py
     │                                  SSIM target; PSNR diagnostic
     │                                              │
     └──────────────┬───────────────────────────────┘
                    ▼
             dataset.py + splits.py
   manifest.csv / sources.csv / features.csv /
   dataset_summary.json / images/
```

The clean reference is generated once per source. Every corruption variant of
that source is compared with the same reference. The split is assigned to the
clean source before pixels are generated, so all variants stay in one split.

The public manifest contains no raw answer text. Literal text, when needed for
replay, lives only in the Git-ignored `ground_truth.csv`. The target is
`ssim_target`; PSNR is recorded for diagnostics and is not a model input.

## 2. Modelling and evaluation paths

```text
dataset + features
        │
        ├── benchmark.py ───────────────► M3 baseline, M5C selection
        │                                  (train / validation / test)
        │
        ├── robustness_benchmark.py ────► M4B occlusion/interference
        │                                  (zero-shot + adaptation)
        │
        ├── geometric_benchmark.py ─────► M5B geometric warp
        │                                  (zero-shot + adaptation)
        │
        ├── retention_aware*.py ────────► M5C / M5D retention studies
        │
        ├── spatial_*.py,
        │   local_structure*.py,
        │   structure_tensor*.py ───────► M6A–M6D feature studies
        │
        ├── heldout_font_benchmark.py ──► M7B held-out-font transfer
        │
        └── matched_font_benchmark.py ──► M8C matched 10-font benchmark
                                           (frozen M5C champion, zero-shot)
```

Every modelling path uses the same eight-feature representation and the same
`ssim_target`. The M3 baseline and M5C selection are the only stages that fit
and select the champion. M4B and M5B contain controlled adaptation experiments
for their own frozen designs. M7B, M8C and M9A load or analyse the frozen
champion without fitting, calibrating, selecting or replacing it.

## 3. Verification and provenance layer

```text
M8C dataset + report + predictions + run metadata
        │
        ├── matched_font_benchmark.py verify ──► 6,300-row replay
        │
        ├── benchmark_lifecycle.py verify ─────► pre/post-execution state
        │
        ├── benchmark_integrity.py verify ─────► M8E historical evidence
        │
        ├── interference_error_analysis.py ────► M9A read-only report
        │
        └── interference_experiment_protocol.py verify
                                                 M9B static preregistration
```

The verification layer is intentionally separate from execution:

- **M8C replay** re-decodes the images, recomputes targets and features, replays
  the frozen predictions and compares the committed tables and report.
- **Lifecycle** checks the authorised dataset/report membership and the
  historical planning state.
- **M8E** pins the execution module closure, external packages, input manifest
  and exact ground-truth byte hash without reading the text.
- **M9A** analyses the frozen predictions and writes a scoped exploratory
  report; it does not load the model.
- **M9B** validates a static protocol package; it has no execution command.

## 4. Component responsibilities

| Component | Responsibility | Primary output |
| --- | --- | --- |
| `config.py` | Typed config loading and validation | Validated experiment config |
| `render.py` | Deterministic clean text rendering | Clean reference image |
| `corruptions.py` | Blur and noise operators | Degraded image |
| `interference.py` | Thin-line and arc-stroke operators | Degraded image |
| `geometric.py` | Bounded elastic warp | Degraded image |
| `quality_metrics.py` | SSIM and PSNR | `ssim_target`, `psnr_target` |
| `features.py` | Eight whole-image features | `features.csv` |
| `dataset.py` | Generation, manifests, verification | Dataset directory |
| `splits.py` | Source-grouped split assignment | Split labels and hash |
| `modelling.py` | Estimators, grouped metrics, persistence | Model and metrics |
| `benchmark.py` | M3 frozen benchmark | M3 model/report |
| `robustness_benchmark.py` | M4B benchmark | M4B report |
| `geometric_benchmark.py` | M5B benchmark | M5B report |
| `retention_aware.py` | M5C selection/calibration | M5C champion |
| `retention_aware_training.py` | M5D training objectives | M5D report |
| `spatial_features.py` et al. | M6 feature studies | Study reports |
| `heldout_font_benchmark.py` | M7B zero-shot font evaluation | M7B report |
| `cross_font_analysis.py` | M7C read-only analysis | M7C report |
| `evaluation_plan.py` | M7D evaluation design | Planning report |
| `font_roster_audit.py` | M7E licence/availability audit | Audit report |
| `benchmark_lifecycle.py` | M8A state contract | Lifecycle verification |
| `benchmark_preflight.py` | M8B deterministic cases/preflight | Preflight evidence |
| `matched_font_benchmark.py` | M8C execution and replay | M8C dataset/report |
| `benchmark_integrity.py` | M8E integrity contract | Historical/future checks |
| `interference_error_analysis.py` | M9A error analysis | M9A report |
| `interference_experiment_protocol.py` | M9B static validator | Preregistration checks |
| `research_prioritization_v0_1_0/` | M9C decision | Path A closure memo |

## 5. Artifact and evidence boundaries

- `data/processed/` and `models/` are Git-ignored. They are local evidence, not
  clone contents.
- Most earlier reports (`reports/benchmark_v0_1_0/`,
  `reports/heldout_font_benchmark_v0_1_0/`, and similar M3–M7 outputs) are
  Git-ignored and absent from a fresh clone.
- The M8C benchmark, M8D audit, M9A analysis, M9B protocol and M9C decision
  packages are committed deliberately because they are the final evidence for
  the release.
- The M8C `ground_truth.csv` is never tracked. Its exact hash and parsed
  structure can be checked without exposing its text.
- Font binaries are never tracked. The benchmark uses pinned local system
  fonts under an evaluation-only workflow.

## 6. Execution boundaries

| Command family | Writes artifacts? | Release status |
| --- | --- | --- |
| `captcha-dataset generate` | Yes: dataset, images, features | Not run |
| `captcha-benchmark run` | Yes: models and report | Not run |
| `captcha-*-pilot run` | Yes: pilot data and report | Not run |
| `captcha-*-benchmark run` | Yes: benchmark data/report | Not run |
| `captcha-*-study run` | Yes: study data/report | Not run |
| `captcha-matched-font-benchmark run` | Yes: 6,300 images and report | Not run |
| `captcha-interference-error-analysis run` | Yes: M9A report | Not run |
| `captcha-*-benchmark verify`, `captcha-benchmark-integrity verify`, `captcha-benchmark-lifecycle verify` | No research output | Allowed |
| `captcha-interference-experiment-protocol verify` | No | Allowed |
| `pytest`, `ruff`, `git diff --check` | No research output | Allowed |

The full reproducibility boundary is documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
