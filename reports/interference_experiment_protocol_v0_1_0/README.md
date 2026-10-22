# Interference experiment protocol v0.1.0 (M9B)

## Purpose

M9A identified three plausible explanations for the frozen model's large
thin-line and arc-stroke interference error:

1. **H1 — attenuated response:** predictions cover too little of the target
   SSIM range, especially at severe interference.
2. **H2 — edge/sharpness feature conflation:** the frozen eight-feature
   representation may not separate interference-induced edge changes from
   benign image-quality variation.
3. **H3 — font/glyph-specific difficulty:** apparent font differences may
   reflect target difficulty, glyph composition, renderer effects or other
   confounders rather than a font effect.

M9A was descriptive and exploratory. This package is a **preregistered design
for future tests**, not a result. It fixes what would count as support,
falsification or an inconclusive result before any new confirmatory outcome is
collected.

## Status

- Milestone: **M9B**
- Status: **preregistered, not executed**
- Design revision: `e3d8ed36b497347b87606536000114746d27d709`
- M9A analysis revision: `37bc86dba3cac8d7cfe2d9e84d83da300974866f`
- M9A report revision: `e3d8ed36b497347b87606536000114746d27d709`

No experiment was run. No model was trained, refit, tuned, calibrated or
selected. No feature was added, removed or reordered. No benchmark data,
prediction or model artifact was generated. No M8C, M8D, M8E or M9A artifact
was modified.

## Package contents

| File | Purpose |
| --- | --- |
| `README.md` | scope, status, rationale and limitations |
| `experiment_protocol.md` | complete preregistered design for H1, H2 and H3 |
| `hypothesis_matrix.csv` | one row per hypothesis: predictions, thresholds, confounders and follow-ups |
| `statistical_analysis_plan.json` | machine-readable estimands, units, splits, seeds, resampling, multiplicity, missing-data rules and decision states |
| `data_requirements.md` | exact existing inputs and the new independent data required for confirmatory tests |
| `decision_gates.md` | explicit rules for proceeding, stopping, revising or deprioritising |
| `verification_checks.csv` | checks, expected outcomes, actual outcomes and limitations |

## Core design decisions

- **The target remains `ssim_target`.** The residual convention is
  `ssim_target - prediction`; positive means underprediction and negative means
  overprediction. PSNR is not a primary outcome. All primary estimands are in
  SSIM units: H1 is a dimensionless slope difference, and H2/H3 are absolute
  residual differences on the 0-1 SSIM scale.
- **The source group is the independent unit.** One source is one font rendered
  with one committed text case. All corruption variants of a source share a
  clean reference and are never treated as independent.
- **H1 uses target SSIM as the independent variable, not severity.** Severity
  is an ordinal design label with unequal target gaps, so it is secondary and
  diagnostic.
- **H1 is a paired slope-deficit test.** For each source, the within-source
  slope of prediction on target for interference is compared with the
  corresponding slope for the original blur/noise reference. This removes
  source-level mean bias and tests whether any attenuation is specific to
  interference.
- **H2 uses all eight frozen features as matching covariates.** No feature is
  chosen, ranked, added or removed from the M9A associations. The primary test
  is a source-grouped matched absolute-residual excess after matching on the
  eight features and target SSIM within source.
- **H3 uses new fonts and matched cases.** A predeclared new thin-stroke group
  is compared with a predeclared new thick-stroke group after matching on text
  case, interference family/severity and target-SSIM bins. Font identity remains
  entangled with glyph shape, font metrics, hinting, the renderer and target
  construction; no causal font effect or font-population claim is permitted.
- **Primary multiplicity control is Bonferroni simultaneous 98.333% percentile
  intervals across the three hypotheses.** Secondary and subgroup estimates are
  exploratory and use Benjamini-Hochberg q-values where applicable. They cannot
  change a primary decision.
- **All confirmatory tests require a new, independently generated source set
  C.** M8C and M9A are observed data and are labelled exploratory only. A pilot
  P estimates variance, matching overlap and attainable precision but is never
  used to adjudicate a hypothesis.
- **Sample size is not invented here.** It is determined by a precision
  simulation from the pilot and frozen before C is generated. No statistical
  power is claimed and no optional stopping is permitted.
- **The frozen M5C champion remains frozen.** This protocol does not authorise
  a model change, calibration, feature engineering or a new target.

## What would count as evidence

Each hypothesis has a predeclared primary estimand, a support rule, a
falsification rule and an inconclusive rule. The short version is:

| Hypothesis | Primary estimand | Support | Falsify |
| --- | --- | --- | --- |
| H1 | paired within-source slope deficit: reference slope minus interference slope | simultaneous CI lower bound > 0.10, interference slope upper bound < 0.80, both family slopes < 1.0 | slope-deficit CI upper bound <= 0.00, or interference slope lower bound >= 0.90 |
| H2 | source-grouped matched absolute-residual excess | simultaneous CI lower bound > 0.02 with predeclared balance and coverage checks | simultaneous CI upper bound <= 0.02 |
| H3 | case-cluster matched thin-minus-thick font-group absolute-residual contrast | simultaneous CI lower bound > 0.02 and every predeclared per-font sign check | simultaneous CI upper bound <= 0.02 |

The full definitions, units and guards are in
`statistical_analysis_plan.json` and `experiment_protocol.md`.

## Explicit non-claims

This package does not claim, and the proposed experiments cannot establish:

- human readability, CAPTCHA solvability, CAPTCHA security or production
  robustness;
- a causal mechanism for interference error;
- a causal font effect or a font-population effect;
- that any feature causes the model error or that adding a feature would fix it;
- that synthetic-reference SSIM is a perceptual or human-legibility measure;
- that the frozen champion should be replaced.

## Validation

The protocol artifacts are validated without running an experiment:

```bash
.venv/bin/python -m captcha_quality.interference_experiment_protocol verify
```

The validator checks the package membership, JSON schema, unique hypothesis
IDs, required fields, allowed decision states, matrix/plan consistency,
exploratory labelling of M8C-derived testing, provenance hashes and the absence
of model-fitting or benchmark-generation entry points. Focused tests live in
`tests/test_interference_experiment_protocol.py`.

## Reproduction boundary

This is a static design package. It deliberately has no `run`, `fit`, `predict`,
`generate`, `calibrate` or `select` command. A future confirmatory milestone
must generate new data under a separate frozen execution protocol and must keep
the historical M8C, M8D, M8E and M9A artifacts unchanged.
