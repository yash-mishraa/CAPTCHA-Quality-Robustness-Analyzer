# Preregistered experiment protocol

- **Milestone:** M9B
- **Version:** v0.1.0
- **Status:** preregistered, not executed
- **Design revision:** `e3d8ed36b497347b87606536000114746d27d709`
- **M9A analysis revision:** `37bc86dba3cac8d7cfe2d9e84d83da300974866f`

## 1. Scope

This protocol specifies how the three hypotheses in
`reports/interference_error_analysis_v0_1_0/hypotheses_and_followups.md` could
be tested fairly. It is a planning artifact. It does not run an experiment, fit
or calibrate a model, add or select a feature, generate benchmark data, produce
predictions or modify any historical artifact.

The target is synthetic-reference SSIM (`ssim_target`). The residual convention
is

```text
residual = ssim_target - prediction
```

A positive residual is underprediction; a negative residual is overprediction.
PSNR is not a primary outcome. No result in this protocol is a claim about
human readability, CAPTCHA solvability, security or production robustness.

## 2. Why a new confirmatory source set is required

M8C has already been observed. M9A used the M8C predictions to generate the
three hypotheses, inspect subgroup patterns and record exploratory feature
associations. Therefore:

- M8C cannot serve as a pristine, untouched confirmatory test set for
  hypotheses generated from M9A.
- Any M8C-based follow-up is **exploratory**, even if it is predeclared in this
  package.
- Confirmatory claims require a new, independently generated source set C with
  new cases, new render seeds, new corruption seeds and, for H3, new fonts.
- A pilot P may be generated to estimate variance, matching overlap and
  attainable precision, but P may not be used to adjudicate a hypothesis and P
  must be disjoint from C.

The data regimes are:

| Regime | Role | Confirmatory? |
| --- | --- | --- |
| M8C observed data | motivation, exploratory description, planning variance | no |
| Pilot P | precision, matching feasibility, sample-size determination | no |
| Confirmatory set C | one-shot preregistered tests | yes |

## 3. Frozen model, features and target

The frozen M5C champion is evaluated zero-shot. The artifact is
`models/retention_aware_selection_v0_1_0/selected_policy.joblib` with SHA-256
`c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`. The model
card SHA-256 is
`9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8`.

No fit, partial fit, calibration, selector, threshold change, target change or
feature change is permitted. The model is loaded once and predicts once per
evaluation run. The target is `ssim_target`, computed as SSIM between the
corrupted image and its source-specific clean reference.

The feature set is exactly the frozen eight-feature v1 set:

```text
intensity_mean
intensity_std
contrast_rms
ink_fraction
laplacian_variance
edge_density
histogram_entropy
noise_sigma_immerkaer
```

H2 uses **all eight** features as matching covariates. M9A feature-error
associations are exploratory and cannot be used to choose, rank, add or remove a
feature.

## 4. Independent unit and grouping

The independent unit is the **source group**: one pinned font rendered with one
committed text case. Each source has 21 rows: 19 primary rows and 2 diagnostic
rows. The eight interference rows are thin lines s1-s4 and arc strokes s1-s4.
The eight original reference rows are Gaussian blur s1-s4 and Gaussian noise
s1-s4. The clean reference and the 0.5 px elastic diagnostic are excluded from
primary estimands.

All corruption variants of a source share the same clean reference, text, font
and case. Individual corrupted images are never treated as independent samples.
H1 and H2 resample whole source groups. H3 uses matched case clusters and
resamples whole cases, which keeps every font, source and variant of a case
together.

## 5. H1: attenuated or range-compressed response

### 5.1 Question and operationalisation

The test asks whether the frozen model's prediction response to target SSIM is
more compressed on thin-line and arc-stroke interference than on the original
blur/noise reference conditions, after removing source-level mean differences.

Target SSIM is the independent variable. Severity is an ordinal design label
with unequal target-SSIM gaps and is used only as a secondary diagnostic.

For each eligible source group `s`:

- Fit an ordinary least-squares line of `prediction` on `ssim_target` across the
  eight interference rows. Record its slope `b_int,s`.
- Fit the corresponding line across the eight original blur/noise rows. Record
  its slope `b_ref,s`.
- Define the paired slope deficit `Delta_s = b_ref,s - b_int,s`.

The primary estimand is

```text
theta_H1 = mean over eligible sources of Delta_s
```

with each source weighted equally. Its units are SSIM per SSIM, a dimensionless
slope. A positive value means interference is more compressed than the original
reference. The primary metric is the source-grouped mean paired calibration
slope difference.

Because both slopes are computed within source on centred variables, a simple
source-level mean bias cannot create a positive `theta_H1`. The mean
interference slope is also tested directly against one-to-one response.

### 5.2 Support, falsification and inconclusive rules

**Support** requires all of the following:

1. the simultaneous 98.333% source-grouped percentile CI for `theta_H1` has a
   lower bound greater than `0.10`;
2. the simultaneous 98.333% CI for the mean interference slope has an upper
   bound less than `0.80`;
3. both the thin-line and arc-stroke mean slopes have point estimates less than
   `1.0`.

The third condition is a qualitative family guard. It prevents a pooled result
from being declared support when one interference family shows no attenuation.
It cannot by itself support H1.

**Falsification** occurs if either:

- the simultaneous 98.333% CI for `theta_H1` has an upper bound less than or
  equal to `0.00`; or
- the simultaneous 98.333% CI for the mean interference slope has a lower bound
  greater than or equal to `0.90`.

**Inconclusive** covers every result that satisfies neither rule, including a
CI that crosses a threshold, failure to meet the precision target, or fewer than
90% of planned sources being eligible.

### 5.3 Distinguishing bias, condition-specific error and source composition

- Mean bias is not part of the primary estimand. Source-level centring removes
  it. Bias and intercepts are reported as secondary diagnostics.
- Condition-specific behaviour is tested by the paired reference contrast and
  by the thin/arc family guard.
- Source composition is controlled by within-source slopes and by source-grouped
  resampling. Sources are never pooled as independent images.
- A positive result would not establish that attenuation is the only cause of
  interference error. It would establish a compressed conditional response
  relative to the reference conditions.

### 5.4 Secondary and diagnostic metrics

Reported but not used for the H1 decision:

- mean within-source interference slope for thin lines and arcs separately;
- mean within-source original blur/noise slope;
- source-level intercepts and mean bias;
- residual-target Spearman correlation by family;
- s4 target-prediction gap;
- s4 large-error proportion at the predeclared exploratory threshold of `0.20`;
- interference MAE, RMSE, R2 and bias;
- source-level squared-error concentration.

Severity-specific calibration curves are diagnostic. A linear slope is a summary
and may miss nonlinear calibration.

## 6. H2: edge/sharpness feature conflation

### 6.1 Question and operationalisation

The test asks whether the frozen eight-feature representation leaves a
systematic interference-specific absolute-error excess after matching
interference and benign rows on the frozen features and on target SSIM within
source.

The primary outcome is `|ssim_target - prediction|`. The independent variable is
condition membership: interference versus benign, after matching.

### 6.2 Matching design

Matching is observational and uses covariates only. Predictions and residuals
are never used to form matches. The matching procedure is fixed before C is
generated:

- Match within the same source group, so font, case, text and clean reference
  are held constant.
- Benign candidates are the original Gaussian blur and Gaussian noise primary
  rows from that source. Geometric primary rows are a secondary reference only.
- Require the same target-SSIM bin of width `0.05`.
- Standardise the eight features using pilot-fixed means and standard
  deviations. Do not standardise using C outcomes.
- Select the nearest benign row by Euclidean distance in the eight-dimensional
  standardised feature space, with a caliper of `0.25` SD and a deterministic
  tie-break by sample identifier.
- Match one-to-one without replacement within source and target bin.

The primary estimand is

```text
theta_H2 = mean over eligible sources of
           (mean matched |residual| for interference
            minus mean matched |residual| for benign)
```

Each source contributes equally. Positive means interference has larger
absolute error after matching. Units are SSIM units on the 0-1 scale.

### 6.3 Support, falsification and inconclusive rules

**Support** requires all of the following:

1. the simultaneous 98.333% source-grouped percentile CI for `theta_H2` has a
   lower bound greater than `0.02`;
2. post-match absolute standardised mean difference is at most `0.10` for target
   SSIM and for every one of the eight frozen features;
3. at least 80% of eligible interference rows are matched;
4. at least 90% of eligible source groups contribute at least one matched pair.

**Falsification** occurs if the simultaneous 98.333% CI for `theta_H2` has an
upper bound less than or equal to `0.02`.

**Inconclusive** covers every result that satisfies neither rule, plus any
matching overlap or balance failure, coverage below the predeclared thresholds,
or failure to meet the precision target.

### 6.4 What this test can and cannot show

A supported result would show that condition membership still predicts an
absolute-error excess after matching on the frozen features and target. It would
be consistent with feature inadequacy or conflation, but it would not identify
which feature is responsible, prove that the random forest uses a feature as a
benign cue, or prove that adding a feature would repair the error. It also does
not separate feature inadequacy from training-distribution shift.

A falsified result would weaken the conflation explanation in this design. It
would not prove that the features are sufficient for every purpose.

### 6.5 Secondary and diagnostic metrics

- signed matched residual difference;
- per-feature and target post-match standardised mean differences;
- geometric-primary secondary reference;
- per-font and per-severity descriptive matched contrasts;
- exploratory within-font and within-severity rank associations of
  `laplacian_variance` and `edge_density` with absolute error;
- feature correlation structure and redundancy;
- unmatched-row counts and reasons.

The feature associations are exploratory. They cannot support H2 on their own
and cannot be used to select a feature.

## 7. H3: font/glyph-specific difficulty

### 7.1 Question and operationalisation

The test asks whether a predeclared new thin-stroke font group has a larger
matched interference absolute error than a predeclared new thick-stroke font
group after matching on text case, interference family/severity and target-SSIM
bin. This is a conditional association under the fixed renderer, not a causal
font effect.

Font groups are fixed before C is generated using an external, predeclared
stroke-width measure applied to clean renderings. At least two new thin-stroke
and two new thick-stroke fonts are required. All fonts must be licence-eligible,
hash-pinned and unused by M5C training, M8C and pilot P. The exact roster is
frozen after the pilot and before C generation.

### 7.2 Matching and unit of analysis

- Cases are matched by construction: the same new case is rendered in every
  predeclared font.
- Condition is matched by interference family and severity.
- Target SSIM is matched within a bin of width `0.05`.
- Rows are matched one-to-one without replacement within case x family x
  severity x target bin.
- The case is the analysis cluster. Every font, source and variant of a case
  stays together during resampling.

For each eligible matched case `c`, compute the mean matched absolute residual
for the thin group and for the thick group. The primary estimand is

```text
theta_H3 = mean over eligible cases of
           (mean matched |residual| for thin fonts
            minus mean matched |residual| for thick fonts)
```

Fonts contribute equally within each group. Each case contributes equally.
Positive means the thin-stroke group has larger matched absolute error. Units are
SSIM units on the 0-1 scale.

### 7.3 Support, falsification and inconclusive rules

**Support** requires both:

1. the simultaneous 98.333% case-cluster percentile CI for `theta_H3` has a
   lower bound greater than `0.02`;
2. every predeclared per-font sign check holds: each thin font has a higher
   matched mean absolute residual than the mean of all thick fonts, and each
   thick font has a lower matched mean absolute residual than the mean of all
   thin fonts.

The sign checks are a replication guard against a single font driving the
contrast. They are not additional significance tests. If the confirmatory
roster contains more than two fonts per group, the same rule applies to every
font in the group; the roster may not be trimmed after outcomes are seen.

**Falsification** occurs if the simultaneous 98.333% case-cluster CI for
`theta_H3` has an upper bound less than or equal to `0.02`.

**Inconclusive** covers every result that satisfies neither rule, plus
target-overlap failure, sign-check inconsistency, fewer than 90% of planned
matched cases, or failure to meet the precision target.

### 7.4 What cannot be separated

Font identity is fully entangled with glyph shape, stroke width, font metrics,
hinting, rasterisation and the font's own clean reference. The renderer is
fixed, so renderer effects are not varied. A small purposive roster of at least
two fonts per group is not a font population.

This design therefore cannot support:

- a causal claim that a font property causes the error;
- a claim about thin or thick fonts in general;
- a claim about unseen renderers;
- a claim about human difficulty or CAPTCHA strength.

A supported result is a conditional contrast under the fixed renderer and the
predeclared new-font roster.

### 7.5 Secondary and diagnostic metrics

- per-font matched mean absolute residuals and pairwise font contrasts;
- target-SSIM distributions by font and severity;
- severity-specific font-group contrasts;
- source-grouped bootstrap sensitivity;
- interference MAE, bias and large-error proportion by font;
- clean-reference and s1 target distributions by font;
- text-length and case-composition checks;
- descriptive comparison with the observed M8C font ordering, explicitly
  exploratory.

## 8. Statistical analysis plan

### 8.1 Primary inference

There are three primary hypotheses. To control the family-wise error rate at
`0.05`, the primary decisions use **Bonferroni simultaneous 98.333% percentile
bootstrap intervals**:

```text
lower percentile = 0.8333333333333333
upper percentile = 99.16666666666667
```

Each primary interval is based on `2,000` bootstrap replicates with seed
`20261023`.

- H1 and H2 resample whole `source_id` groups.
- H3 resamples whole `case_id` clusters.
- Every replicate retains all rows, fonts and variants belonging to a sampled
  source or case.

The percentile method is used because it makes no normal-theory assumption and
matches the existing repository bootstrap convention. The interval is a
description of uncertainty over the observed source/case distribution, not a
population guarantee.

### 8.2 Reporting intervals and secondary analyses

Ninety-five percent intervals may be reported for secondary and descriptive
estimates. They are not the primary decision intervals. If p-values are
reported for secondary or subgroup comparisons, Benjamini-Hochberg q-values are
used within the declared secondary family.

Secondary and subgroup estimates cannot:

- change a primary support/falsification decision;
- be substituted for a failed primary result;
- be used to choose a feature, threshold, font or model;
- be described as confirmatory.

### 8.3 Missing data and invalid predictions

- No imputation is performed.
- A missing or non-finite prediction is invalid.
- Finite predictions outside `[0, 1]` are retained and reported; clipping would
  be a post hoc transformation.
- A source contributes to a primary estimand only when every required row is
  present, finite and joins exactly once. Otherwise the whole source is excluded
  from that estimand.
- A primary result is `not_testable` if fewer than 90% of planned eligible
  sources remain.
- H2 is `not_testable` if the predeclared balance or coverage checks fail.
- H3 is `not_testable` if fewer than 90% of planned matched cases remain or
  target-overlap checks fail.
- Every exclusion is counted and reported. No row is silently replaced.

### 8.4 Sample size and precision

No sample size is asserted from the existing benchmark. The current 300 M8C
sources, ten fonts and thirty cases have already been observed and are not a
pristine confirmatory set. Ten purposive fonts are not a font population.

Sample size is determined by a precision simulation from pilot P:

- estimate source-level and case-level variance, matching yield, target overlap
  and attainable interval width;
- search a prespecified candidate source-group grid;
- select the smallest candidate that meets all three precision targets with at
  least 80% expected matching coverage and at least 90% expected complete-source
  retention;
- add 10% inflation for attrition;
- freeze the sample size, roster and analysis code before C is generated.

The precision targets for the simultaneous-interval half-width are:

| Hypothesis | Maximum half-width |
| --- | --- |
| H1 | `0.05` |
| H2 | `0.01` |
| H3 | `0.01` |

These targets are chosen to resolve the predeclared decision thresholds. They
are not a power guarantee, and no power claim is made. If the pilot cannot
achieve them within a feasible dataset, confirmatory data collection does not
proceed.

### 8.5 One-shot analysis and analysis lock

Before C is generated, the protocol, sample size, font roster, case manifest,
matching code, analysis code, seed and thresholds are frozen. C is analysed
once for the primary decisions. No optional stopping, no peeking, no endpoint
change and no threshold tuning are permitted. Any post hoc analysis is labelled
exploratory and cannot be used for model development.

## 9. Leakage and test-set reuse controls

| Control | Rule |
| --- | --- |
| Source-text overlap | C uses a new case manifest with text hashes disjoint from M8C and pilot P. Literal text stays in the Git-ignored ground-truth file and is never exposed. |
| Related variants | All variants of a source stay together. H1/H2 resample source groups; H3 resamples case clusters. |
| Font and renderer | H3 uses new fonts not used in M5C training, M8C or pilot P. The renderer is fixed. Unseen-renderer questions require a separate factor. |
| Threshold selection | All primary thresholds, bins, calipers, features and endpoints are fixed here. Pilot data may inform precision and feasibility only. |
| Feature selection | H2 uses all eight frozen features. M9A associations cannot choose or rank a feature. |
| Test-set reuse | M8C is exploratory only. C is used once for primary decisions and is not reused for model development. |
| Frozen champion | The M5C champion stays frozen. Any model development requires a separately approved protocol and independent data. |
| Ground truth | `ground_truth.csv` stays Git-ignored and untracked. Hash checks are allowed; text is never read into outputs. |

## 10. Decision gates

The complete gate matrix is in `decision_gates.md`. In summary:

- **Support** justifies planning a separately approved follow-up experiment. It
  does not automatically authorise a model or feature change.
- **Falsification** deprioritises that mechanism. No calibration, feature or
  model intervention should be built on a falsified hypothesis.
- **Inconclusive** requires a better pilot, overlap or precision design. It is
  not support.
- **Not testable** means the current data or design cannot identify the effect;
  a new dataset or experimental factor is required.
- M9C is not automatically a feature-engineering or model-improvement
  milestone. Any next step needs a separate decision, a written prioritisation
  memo and a new protocol version.

## 11. Verification boundary

This package is static. The validation command

```bash
.venv/bin/python -m captcha_quality.interference_experiment_protocol verify
```

checks package membership, schema, unique IDs, required fields, allowed
decision states, matrix/plan consistency, exploratory labelling of M8C-derived
testing, provenance hashes and the absence of model-fitting or
benchmark-generation entry points. It does not run an experiment.

## 12. Limitations

- No experiment was run; all thresholds are design decisions, not empirical
  results.
- M8C and M9A are observed and cannot provide a pristine confirmatory test.
- The proposed confirmatory set C does not exist. Its size, roster and matching
  feasibility must be fixed by a pilot.
- H1 uses a linear within-source slope, which is a summary and may miss
  nonlinear calibration.
- H2 matching is observational and cannot identify a causal feature.
- H3 cannot separate font identity from glyph shape, font metrics, hinting,
  rasterisation, target construction or case composition.
- The fixed renderer means no unseen-renderer claim is possible.
- All claims are about synthetic-reference SSIM, not human readability,
  solvability, security or production robustness.
