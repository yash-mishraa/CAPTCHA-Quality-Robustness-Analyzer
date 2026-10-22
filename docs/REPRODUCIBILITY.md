# Reproducibility Guide

**Release:** `v1.0.0`
**Scope:** safe verification of the completed M0–M9C evidence

This guide distinguishes three different things:

1. **A fresh clone.** Contains code, configs, documentation, tests and the
   committed M8C/M8D/M9A/M9B/M9C evidence packages. It does **not** contain the
   generated datasets, model artifacts or font binaries.
2. **The prepared evidence workspace.** Contains the local Git-ignored datasets,
   models and pinned fonts needed to replay M8C and run all integrity checks.
3. **A full regeneration.** Recreates datasets, images, predictions or models
   from configs. This is **not** part of the release check and was not run for
   `v1.0.0`.

Full regeneration is not claimed from a fresh clone alone. The historical
benchmark depends on local evidence, pinned Apple-bundled font builds, the frozen
model artifact and the recorded execution environment.

## 1. Environment

- Python **3.11+**.
- Historical benchmark environment: CPython **3.13.16**, macOS **27.0.1
  (arm64)**, with the package versions recorded in
  `reports/matched_font_benchmark_v0_1_0/run_metadata.json`.
- Dependencies are declared in `pyproject.toml`.
- The historical report was run with the package version `0.1.0`. The `v1.0.0`
  release changes only release/documentation metadata; it does not re-version or
  rewrite the historical dataset, model, protocol or report.

Using `uv`:

```bash
uv sync --extra dev --extra image --extra ml
```

Using `venv` and `pip`:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,image,ml]'
```

The examples below use `uv run`. If you use the virtual environment directly,
replace `uv run python` with `.venv/bin/python`.

## 2. Read-only release verification

These commands do not generate datasets, images, predictions, models or reports.
Run them from the repository root.

### Tests and lint

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Expected on the release workspace:

- pytest: all tests pass, with skips only for local historical artifacts that
  are absent.
- Ruff check: `All checks passed!`
- Ruff format check: all files already formatted.
- `git diff --check`: no output and exit status 0.

### M8E historical integrity

```bash
uv run python -m captcha_quality.benchmark_integrity verify
```

Expected result:

- `verification: PASS`
- `17 / 17` pinned files match.
- `36 / 36` input rows are pinned and matched; `35` unique keys with one
  declared historical duplicate.
- The exact ground-truth SHA-256
  `426fa41949223ebc481e88b0c42a22770df80d30d3ed8f951fa0f52b92b39dfd` is
  checked without reading the text.
- The historical revision remains `75c8ef4-dirty`.

### M8C lifecycle state

```bash
uv run python -m captcha_quality.benchmark_lifecycle verify \
  --config configs/benchmark_lifecycle_v0_3_0.toml --state post-execution
```

Expected result: `verification: PASS` with `11/11` checks, including exact
dataset/report membership, 6,300 images and the post-execution state.

A pre-execution assertion on this workspace fails by design because the
authorised M8C outputs now exist. A failure there is an expected planning-state
guard, not a broken release.

### M8C replay

```bash
uv run python -m captcha_quality.matched_font_benchmark verify \
  --lifecycle-post configs/benchmark_lifecycle_v0_3_0.toml
```

Expected result:

- `verification: PASS`
- `16/16` replay checks.
- 6,300 images decoded and re-hashed.
- 6,300 predictions replayed.
- Thirteen tables and `report.md` replayed exactly.
- Frozen champion artifact and card hashes match.

This command requires the local M8C dataset, the local frozen model artifact and
the pinned font files. Without them it fails closed. Do not weaken the check.

### M9B static protocol validator

```bash
uv run python -m captcha_quality.interference_experiment_protocol verify
```

Expected result:

- 26/26 checks passed.
- All three hypotheses report `not_tested`.
- No execution entry point exists.
- The frozen settings are seed `20261023`, 2,000 replicates and simultaneous
  confidence 98.333%.

### M8C pre-execution planning-state check

The lifecycle contract can also be asked to assert its historical planning
state:

```bash
uv run python -m captcha_quality.benchmark_lifecycle verify \
  --config configs/benchmark_lifecycle_v0_3_0.toml --state pre-execution
```

On the completed workspace this is expected to fail because the authorised
outputs exist. Record it as a guard outcome, not as a release failure.

## 3. What each verifier actually checks

| Verifier | Read-only? | Requires local evidence? | Purpose |
| --- | --- | --- | --- |
| `benchmark_integrity verify` | Yes | Yes: M8C dataset, model/card, pinned modules, packages | M8E historical evidence and exact ground-truth hash |
| `benchmark_lifecycle verify --state post-execution` | Yes | Yes: M8C dataset/report and declared inputs | Output membership and lifecycle state |
| `matched_font_benchmark verify` | Yes | Yes: images, features, predictions, model, fonts | Full M8C replay and report/table comparison |
| `interference_experiment_protocol verify` | Yes | Yes: recorded provenance inputs | Static M9B protocol and provenance checks |
| `pytest` | Yes | Some tests skip local-only artifacts if absent | Code, schema and evidence checks |
| `ruff` / `git diff --check` | Yes | No | Style and whitespace hygiene |

## 4. Inspecting results without regeneration

The committed M8C evidence package is safe to read directly:

- `reports/matched_font_benchmark_v0_1_0/report.md`
- `reports/matched_font_benchmark_v0_1_0/overall_metrics.csv`
- `reports/matched_font_benchmark_v0_1_0/family_metrics.csv`
- `reports/matched_font_benchmark_v0_1_0/font_metrics.csv`
- `reports/matched_font_benchmark_v0_1_0/predictions.csv`
- `reports/matched_font_benchmark_provenance_audit_v0_1_0/report.md`
- `reports/interference_error_analysis_v0_1_0/README.md`
- `reports/interference_experiment_protocol_v0_1_0/README.md`
- `reports/research_prioritization_v0_1_0/README.md`

The M3–M7 reports and generated datasets are local-only. If they are present,
inspect them as evidence; they are not part of a fresh clone.

## 5. Ground-truth policy

`ground_truth.csv` is Git-ignored and untracked. This release:

- did not read, print, copy or materialise its text;
- checked its tracked state and exact hash only through approved verifiers;
- did not add its contents to documentation, tests or reports.

Do not open, paste or commit the text. If you need to verify it, use the M8E
historical-evidence verifier, which reports path, count and digest rather than
text.

## 6. Artifact-generating commands (not run for this release)

The following command families can generate datasets, images, predictions,
models or reports. Do **not** run them as part of release verification. They may
overwrite local evidence or create new artifacts that are not part of the frozen
record.

```text
captcha-dataset generate ...
captcha-pilot ...
captcha-benchmark run ...
captcha-robustness-pilot run ...
captcha-robustness-benchmark run ...
captcha-geometric-pilot run ...
captcha-geometric-benchmark run ...
captcha-retention-aware run ...
captcha-retention-training run ...
captcha-spatial-study run ...
captcha-resolution-study run ...
captcha-local-structure-study run ...
captcha-structure-tensor-study run ...
captcha-dataset-audit ...
captcha-heldout-font-benchmark run ...
captcha-cross-font-analysis ...
captcha-evaluation-plan ...
captcha-font-roster-audit ...
captcha-matched-font-benchmark run ...
captcha-interference-error-analysis run ...
```

`captcha-benchmark verify-model` and the `verify` subcommands are read-only with
respect to the research evidence. `captcha-benchmark-integrity preflight` is a
guard, not an execution command.

## 7. Full-regeneration boundary

A full regeneration would require:

- the exact pinned local font files and their licensing/eligibility conditions;
- the Git-ignored source datasets and model artifacts;
- the pinned Python, package versions, operating system and rasteriser;
- the frozen protocol and text-case manifests;
- the recorded seeds and split definitions.

Even with those inputs, regeneration is a new execution and must not be
presented as the original one-shot result. The historical M8C run records
`75c8ef4-dirty`; a fresh clean-tree run would be a new execution under the M8E
future-execution guard, not a rewrite of history.

The release therefore claims:

- the committed M8C predictions, tables and report can be replayed in the
  prepared evidence workspace;
- the historical inputs are pinned and the verifiers fail closed;
- the code, configs, documentation and tests are clone-visible;
- the complete benchmark is **not** regenerateable from a fresh clone without
  the local inputs and environment.

## 8. Release-tag check

The annotated tag `v1.0.0` is created only after all release gates pass. Verify
it with:

```bash
git rev-parse v1.0.0
git show --stat --oneline v1.0.0
```

The tag points at the release commit. No GitHub Release was published by this
milestone.
