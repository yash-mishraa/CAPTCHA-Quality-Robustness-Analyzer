# Project Overview

**Status:** M0–M9C complete. `v1.0.0` research release prepared. M9C selected
**Path A — stop experimentation and package the research**. M9B remains
preregistered and unexecuted; H1, H2 and H3 remain `not_tested`.

**Last updated:** 2026-10-22

## 1. Purpose

This project asks a deliberately narrow measurement question:

> Can a conventional model predict the structural similarity of a degraded
> synthetic CAPTCHA-style text image to its exact clean reference from
> whole-image features alone, and how does that prediction change under
> different corruption families, severities and fonts?

The target is **SSIM against the corresponding clean synthetic reference**.
PSNR is recorded as a diagnostic. A model that predicts SSIM is not a model of
human readability, CAPTCHA solvability, security or production robustness.

The work is a reproducible measurement and robustness study. It is not a new
CAPTCHA-strength benchmark and does not evaluate real CAPTCHA services.

## 2. Problem formulation

For each clean source image, the pipeline generates controlled corruptions and
computes:

- `ssim_target`: SSIM between the degraded image and its clean reference.
- `psnr_target`: PSNR, diagnostic only.

The model receives exactly the eight documented image features from the
degraded image. It does not receive the target, corruption family, severity,
font, text length, source identifier, split or reference image.

Splits are assigned to **clean source groups** before pixels are generated.
Every variant of a source stays in one split. The source group is the
independent unit for bootstrap intervals and the unit of analysis for the
cross-font benchmark.

## 3. Scope and system boundaries

**In scope**

- Synthetic 192×64 grayscale text images from pinned local fonts.
- Gaussian blur, additive Gaussian noise, thin-line interference, arc-stroke
  interference and bounded elastic warp.
- Conventional regression on eight handcrafted features.
- Grouped train/validation/test evaluation.
- Zero-shot held-out-font evaluation of a frozen champion.
- Matched cross-font evaluation with the same text cases.
- Provenance, replay, lifecycle, integrity and preregistration checks.
- Read-only exploratory error analysis.

**Out of scope**

- Human readability, legibility, solving or response-time outcomes.
- CAPTCHA security, attack success or production robustness.
- Real captcha challenges, live services or captured text.
- Causal font, feature or mechanism claims.
- New experiments after the M9C stop decision.
- CNN, transformer or dashboard results.

The repository contains generated datasets and model artifacts only as local,
Git-ignored evidence. A fresh clone contains code, configs, documentation,
tests and the committed final report packages.

## 4. System architecture

```text
configs
   │
   ▼
render.py ──► clean reference
   │                │
   ▼                │
corruptions.py      │
interference.py     │
geometric.py        │
   │                │
   ▼                ▼
degraded image ──► quality_metrics.py ──► ssim_target / psnr
   │
   ▼
features.py ──► dataset.py + splits.py
   │
   ├──► benchmark.py ─────────────► M3 baseline / M5C champion
   ├──► robustness_benchmark.py ──► M4B
   ├──► geometric_benchmark.py ───► M5B
   ├──► retention/study runners ──► M5C–M6D
   ├──► heldout_font_benchmark.py ► M7B
   └──► matched_font_benchmark.py ► M8C
                                      │
                                      ▼
                       replay / lifecycle / integrity / analysis
                       M8D, M8E, M9A, M9B, M9C
```

There is no single orchestrator that runs the whole history. Each milestone has
its own frozen config, runner, report and verification path. The shared code is
the renderer, corruption operators, metric implementation, feature extractor,
dataset writer, split assigner and grouped evaluation utilities.

The detailed component map is in [`ARCHITECTURE.md`](ARCHITECTURE.md). The
methodology is in [`METHODOLOGY.md`](METHODOLOGY.md).

## 5. Data and artifact contracts

A generated dataset under `data/processed/<name>_<version>/` contains:

| File | Role |
| --- | --- |
| `manifest.csv` | One row per sample: identifiers, split, family, severity, parameters, paths, SSIM/PSNR targets and hashes; no raw text |
| `sources.csv` | One row per clean source with split and clean-image hash |
| `ground_truth.csv` | Internal `source_id,text` table; Git-ignored and never inspected for this release |
| `features.csv` | `sample_id` plus the eight documented features |
| `dataset_summary.json` | Counts, split policy, hashes and summary metadata |
| `images/` | Clean and corruption PNGs |

The public manifest deliberately omits raw text. The literal text is kept only
in the internal ground-truth file for replay. The SSIM/PSNR targets are computed
from the image pair, never from the text table.

The M8C benchmark uses a matched design: 10 fonts × 30 text cases = 300
sources, 21 rows per source, 6,300 rows total (5,700 primary and 600
diagnostic). The clean and 0.5 px elastic rows are diagnostic only.

## 6. Repository navigation

| Path | What a reader should use it for |
| --- | --- |
| `README.md` | Release overview, headline results, setup, limitations and research status |
| `docs/RESEARCH_SUMMARY.md` | Concise research narrative and claims audit |
| `docs/METHODOLOGY.md` | Full methods from M0 through M9C |
| `docs/EXPERIMENT_LOG.md` | Chronological milestone and experiment record |
| `docs/ARCHITECTURE.md` | Component map and execution boundaries |
| `docs/REPRODUCIBILITY.md` | Safe copy-paste verification workflow and regeneration boundary |
| `docs/BENCHMARK_INTEGRITY.md` | M8E historical-evidence and future-execution contract |
| `reports/matched_font_benchmark_v0_1_0/report.md` | Committed M8C benchmark report and metrics |
| `reports/matched_font_benchmark_provenance_audit_v0_1_0/report.md` | M8D audit and known provenance boundaries |
| `reports/interference_error_analysis_v0_1_0/README.md` | M9A exploratory error analysis |
| `reports/interference_experiment_protocol_v0_1_0/README.md` | M9B preregistered, unexecuted protocol |
| `reports/research_prioritization_v0_1_0/README.md` | M9C Path A decision and disposition |
| `docs/RELEASE_NOTES_v1.0.0.md` | Release scope, verification status and limitations |
| `docs/RELEASE_CHECKLIST_v1.0.0.md` | Final closure checklist with evidence |

The M8C, M8D, M9A, M9B and M9C packages are committed. Earlier M3–M7 reports,
generated datasets, model artifacts and font binaries are local-only and
Git-ignored. References to those paths therefore describe evidence that must be
present in the local evidence workspace; they are not clone-only links.

## 7. Milestone map

| Milestone | Scope | Outcome | Evidence |
| --- | --- | --- | --- |
| M0 | Project scaffold and target decision | SSIM selected; PSNR diagnostic; MS-SSIM deferred | `docs/EXPERIMENT_LOG.md` |
| M1 | Reference-metric validation pilot | SSIM passes both blur/noise families; PSNR fails the blur gate because of ink coverage | `docs/EXPERIMENT_LOG.md` |
| M2 | Synthetic dataset and features | 60-source reference dataset, source-grouped splits, eight features | `docs/EXPERIMENT_LOG.md` |
| M3 | Conventional benchmark | Random forest test MAE 0.0157, RMSE 0.0226, R² 0.9865 on 108 rows/12 sources | `docs/EXPERIMENT_LOG.md` |
| M4A/M4B | Interference pilot and benchmark | Pilot passes feasibility; frozen M3 fails zero-shot, adaptation succeeds | `docs/EXPERIMENT_LOG.md` |
| M5A/M5B | Geometric pilot and benchmark | Pilot conditional pass; adaptation improves geometric error but costs clean retention | `docs/EXPERIMENT_LOG.md` |
| M5C/M5D | Retention-aware selection and training | Expanded random forest becomes champion; training-objective change does not improve it | `docs/EXPERIMENT_LOG.md` |
| M6A–M6D | Feature studies | Spatial, resolution, local and structure-tensor features do not justify replacing the champion | `docs/EXPERIMENT_LOG.md` |
| M7A–M7E | Dataset audit, font benchmark, analysis, design, licence audit | Held-out-font evidence and matched-design plan; fonts cleared for evaluation-only use | `docs/EXPERIMENT_LOG.md` |
| M8A/M8B | Protocol, lifecycle, deterministic cases, preflight | Execution and lifecycle frozen; cases and identities proven without materialising the benchmark | `docs/METHODOLOGY.md` |
| M8C | Matched cross-font benchmark | 6,300 rows/300 sources/10 fonts; frozen champion MAE 0.059607, RMSE 0.088610, R² 0.732898 | [`reports/matched_font_benchmark_v0_1_0/report.md`](../reports/matched_font_benchmark_v0_1_0/report.md) |
| M8D | Provenance and robustness audit | Verified with documented limitations; no erratum | [`reports/matched_font_benchmark_provenance_audit_v0_1_0/report.md`](../reports/matched_font_benchmark_provenance_audit_v0_1_0/report.md) |
| M8E | Integrity contract | 17/17 pinned modules; 36/36 pinned input rows; exact ground-truth hash check without text exposure | [`BENCHMARK_INTEGRITY.md`](BENCHMARK_INTEGRITY.md) |
| M9A | Interference error analysis | Read-only exploratory analysis; broad, compressed interference response | [`reports/interference_error_analysis_v0_1_0/README.md`](../reports/interference_error_analysis_v0_1_0/README.md) |
| M9B | Preregistered protocol | Three hypotheses with decision gates; not executed; all `not_tested` | [`reports/interference_experiment_protocol_v0_1_0/README.md`](../reports/interference_experiment_protocol_v0_1_0/README.md) |
| M9C | Prioritization and closure | Path A selected; experimentation stopped; package the research | [`reports/research_prioritization_v0_1_0/README.md`](../reports/research_prioritization_v0_1_0/README.md) |

M9D is the documentation and release milestone that produced this `v1.0.0`
package. It is not a research milestone and does not authorize further
experiments.

## 8. Final research status

- The experiment phase is closed.
- No further experiment, pilot, dataset generation, model fit, calibration,
  feature change or benchmark change is authorized by this release.
- H1 is the first candidate if the project is formally reopened with new
  independent data and an approved protocol. It is not authorized now.
- H2 and H3 are deprioritized.
- M9B's hypotheses remain `not_tested`.
- No claim is made about human readability, CAPTCHA solvability, security or
  production robustness.
