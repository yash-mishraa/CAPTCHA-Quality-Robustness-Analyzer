# Interference error analysis v0.1.0

## Purpose and scope

This is a read-only analysis of the frozen M8C cross-font benchmark. It asks
where prediction error is concentrated under thin-line and arc-stroke
interference, whether particular fonts or source groups drive the failure,
which existing features are associated with the error, and whether the pattern
is specific to interference.

The target is `ssim_target`. PSNR is diagnostic only and is not analysed. The
independent unit is the **source group**: one pinned font rendered with one
committed text case. Whole source groups are resampled; individual corrupted
images are never treated as independent.

This analysis did **not** train, refit, tune, calibrate, select or replace a
model; did not add or select a feature; did not generate a dataset; and did not
write into any M8C, M8D or M8E artifact. It does not measure human readability,
CAPTCHA solvability, security or production robustness.

## Frozen inputs and provenance

| Input | Path | SHA-256 prefix | Rows |
| --- | --- | --- | --- |
| predictions | `reports/matched_font_benchmark_v0_1_0/predictions.csv` | e6c1f1ba83bbb05c… | 6300 |
| manifest | `data/processed/matched_font_benchmark_v0.1.0/manifest.csv` | 35ae405fd084070b… | 6300 |
| features | `data/processed/matched_font_benchmark_v0.1.0/features.csv` | fa2d79c85d682b26… | 6300 |
| sources | `data/processed/matched_font_benchmark_v0.1.0/sources.csv` | a6f721f8e849d447… | 300 |
| source_cases | `data/processed/matched_font_benchmark_v0.1.0/source_cases.csv` | 5e34d6a826b5f444… | 300 |
| ground_truth | `data/processed/matched_font_benchmark_v0.1.0/ground_truth.csv` | 426fa41949223ebc… | 300 |
| run_metadata | `reports/matched_font_benchmark_v0_1_0/run_metadata.json` | 930ce8d8767a3e52… | — |
| model_artifact | `models/retention_aware_selection_v0_1_0/selected_policy.joblib` | c9b1625de3bb503f… | — |
| model_card | `models/retention_aware_selection_v0_1_0/selected_policy_card.json` | 9a75725ba1ccf82a… | — |
| m8e_config | `configs/benchmark_integrity_v0_1_0.toml` | 0b018004f41c2373… | — |

The M5C champion artifact is hash-checked but never loaded. Its recorded
`inference.model_fit_performed`, `calibration_performed` and
`feature_change_performed` flags are all false, and its eight feature names are
exactly the frozen M8C feature set. The full input hashes, commit, working-tree
state, software versions, bootstrap seed and verifier outcomes are recorded in
`analysis_protocol.json`.

The local `ground_truth.csv` is integrity-pinned but only its `source_id`,
`case_id` and `font_id` columns are read. The text column is never loaded or
written to the report.

## Main findings

### 1. Interference is worse than the other primary conditions, with opposite bias

| Scope | Rows / sources | MAE (95% CI) | RMSE | Bias | R² | Large error ≥ 0.20 |
| --- | --- | --- | --- | --- | --- | --- |
| all_primary | 5700 / 300 | 0.0596 [0.0583, 0.0610] | 0.089 | -0.031 | 0.7329 | 5.67% |
| original_blur_noise | 2400 / 300 | 0.0327 [0.0289, 0.0371] | 0.057 | 0.016 | 0.8997 | 2.29% |
| gaussian_blur | 1200 / 300 | 0.0265 [0.0234, 0.0301] | 0.050 | 0.007 | 0.9431 | 1.92% |
| gaussian_noise | 1200 / 300 | 0.0390 [0.0342, 0.0444] | 0.064 | 0.026 | 0.7357 | 2.67% |
| interference | 2400 / 300 | 0.0908 [0.0874, 0.0940] | 0.119 | -0.087 | -0.4302 | 11.08% |
| thin_lines | 1200 / 300 | 0.1021 [0.0981, 0.1058] | 0.132 | -0.097 | -0.5752 | 15.17% |
| arc_strokes | 1200 / 300 | 0.0796 [0.0765, 0.0827] | 0.103 | -0.077 | -0.2989 | 7.00% |
| geometric_primary | 900 / 300 | 0.0480 [0.0464, 0.0498] | 0.060 | -0.009 | 0.1625 | 0.22% |
| clean_reference | 300 / 300 | 0.0074 [0.0061, 0.0087] | 0.014 | 0.007 | n/a (suppressed) | 0.00% |

Interference MAE is 0.091 versus
0.026 for blur, 0.039 for noise and
0.048 for primary geometric distortion. The
interference bias is -0.087: predictions are too
high (overprediction) on average. Blur and noise have positive bias, meaning
underprediction. The negative interference R²
(-0.430) means the model is worse than a constant
condition-mean predictor in squared-error terms on this slice.

### 2. Severe interference drives the large errors

At the predeclared exploratory threshold `|residual| >= 0.20`, thin lines have
15.17% large errors and arcs
7.00%. The failure is concentrated
at the highest rung: thin s4 has a large-error proportion of
55.3% and arc s4
28.0%. At thin s4 the mean
target is 0.696 while the mean prediction is
0.904; at arc s4 the corresponding values are
0.741 and 0.906.

| Family | Severity | Mean target | Mean prediction | MAE | Bias | Large error |
| --- | --- | --- | --- | --- | --- | --- |
| thin_lines | 1 | 0.945 | 0.973 | 0.036 | -0.027 | 0.0% |
| thin_lines | 2 | 0.893 | 0.942 | 0.058 | -0.050 | 0.7% |
| thin_lines | 3 | 0.804 | 0.909 | 0.106 | -0.105 | 4.7% |
| thin_lines | 4 | 0.696 | 0.904 | 0.208 | -0.208 | 55.3% |
| arc_strokes | 1 | 0.955 | 0.981 | 0.031 | -0.026 | 0.0% |
| arc_strokes | 2 | 0.912 | 0.954 | 0.046 | -0.042 | 0.0% |
| arc_strokes | 3 | 0.838 | 0.914 | 0.076 | -0.076 | 0.0% |
| arc_strokes | 4 | 0.741 | 0.906 | 0.165 | -0.164 | 28.0% |

### 3. The model responds monotonically but with compressed amplitude

Within each source group, thin-line predictions have a median prediction/target
SD ratio of 0.320
and a median slope of
0.241; arc values are
0.397 and
0.337. Blur and noise are
near 1.0. The within-source rank correlations are nevertheless high
(0.792 for thin and
0.885 for arc), and
residual-target Spearman correlations are
0.916 and
0.863. The pattern is attenuation or
range compression, not a completely condition-insensitive constant predictor.
The attenuation is descriptive; its cause is not identified here.

### 4. Font differences are large but entangled with glyph and target difficulty

| Font | MAE (95% CI) | Bias | R² | Large error | Paired diff vs other fonts (95% CI) | Cases worse |
| --- | --- | --- | --- | --- | --- | --- |
| impact | 0.0358 [0.0281, 0.0449] | -0.032 | 0.224 | 0.4% | -0.0611 [-0.0690, -0.0527] | 0/30 |
| luminari | 0.0846 [0.0775, 0.0922] | -0.081 | -0.392 | 9.6% | -0.0070 [-0.0138, 0.0001] | 11/30 |
| chalkduster | 0.0861 [0.0807, 0.0920] | -0.063 | -0.256 | 7.9% | -0.0052 [-0.0110, 0.0005] | 12/30 |
| monaco | 0.0903 [0.0818, 0.0994] | -0.089 | -0.458 | 7.1% | -0.0006 [-0.0083, 0.0067] | 14/30 |
| trebuchet_ms | 0.0904 [0.0824, 0.0992] | -0.089 | -0.441 | 9.6% | -0.0005 [-0.0061, 0.0053] | 13/30 |
| georgia | 0.0924 [0.0870, 0.0977] | -0.091 | -0.468 | 10.4% | 0.0017 [-0.0026, 0.0061] | 16/30 |
| arial | 0.0948 [0.0879, 0.1018] | -0.094 | -0.512 | 13.3% | 0.0044 [-0.0020, 0.0104] | 21/30 |
| verdana | 0.1009 [0.0944, 0.1073] | -0.101 | -0.598 | 12.1% | 0.0112 [0.0060, 0.0167] | 22/30 |
| times_new_roman | 0.1028 [0.0971, 0.1081] | -0.102 | -0.711 | 14.2% | 0.0133 [0.0086, 0.0181] | 26/30 |
| courier_new | 0.1302 [0.1256, 0.1347] | -0.130 | -1.056 | 26.2% | 0.0438 [0.0371, 0.0495] | 29/30 |

`courier_new` has the largest interference MAE
(0.130); `impact` has the smallest
(0.036). The paired-by-case contrast is
0.044 for Courier New
(95% CI [0.037,
0.050]; worse on
29/30 cases) and
-0.061 for Impact
(95% CI [-0.069,
-0.053]; worse on
0/30). However, the target itself is less
damaged for Impact: its interference target mean is
0.920 versus
0.819 for Courier New, and its s1-to-s4 target
drop is 0.128 versus
0.276. Font identity, glyph shape,
renderer and target degradation are entangled; this is not a causal font effect.

### 5. Interference error is broad across source groups, unlike blur/noise

| Scope | Top 5% source SSE share | Top 10% source SSE share | Source MAE median | Source MAE q90 |
| --- | --- | --- | --- | --- |
| gaussian_blur | 51.5% | 78.5% | 0.016 | 0.050 |
| gaussian_noise | 49.7% | 74.2% | 0.025 | 0.076 |
| interference | 10.6% | 19.8% | 0.091 | 0.131 |
| thin_lines | 11.1% | 20.4% | 0.102 | 0.147 |
| arc_strokes | 12.1% | 21.7% | 0.081 | 0.115 |
| geometric_primary | 15.0% | 25.4% | 0.047 | 0.069 |

The top 5% of interference sources account for
10.6%
of interference squared error, compared with
51.5%
for blur and
49.7%
for noise. Interference is worse than blur for
92.3%
of sources and worse than geometric distortion for
89.7%.
This is a broad, systematic error, not a small number of outlying sources.

### 6. Existing feature associations are strong but exploratory

| Scope | Feature | Spearman rho vs abs residual | BH q | Rows |
| --- | --- | --- | --- | --- |
| thin_lines | edge_density | +0.796 | 1.85e-262 | 1200 |
| interference | laplacian_variance | +0.792 | 0.00e+00 | 2400 |
| thin_lines | laplacian_variance | +0.792 | 2.69e-257 | 1200 |
| arc_strokes | laplacian_variance | +0.791 | 1.90e-256 | 1200 |
| arc_strokes_s4 | laplacian_variance | +0.784 | 6.51e-63 | 300 |
| interference | edge_density | +0.755 | 0.00e+00 | 2400 |
| arc_strokes | noise_sigma_immerkaer | +0.736 | 2.35e-204 | 1200 |
| arc_strokes_s4 | histogram_entropy | -0.719 | 3.62e-48 | 300 |

Within interference and within severity strata, `laplacian_variance`,
`edge_density` and `noise_sigma_immerkaer` tend to be positively associated with
absolute error, while `histogram_entropy`, contrast and ink fraction tend to be
negatively associated. These are descriptive rank associations from the
existing eight-feature table. They are not source-clustered inferential tests,
they do not establish that the model uses a feature in a particular way, and
they are not a feature-selection result.

## Figures

- `figures/fig1_condition_error_profile.svg` — MAE and bias by primary condition.
- `figures/fig2_interference_severity_calibration.svg` — mean target versus mean prediction across interference severities.
- `figures/fig3_font_interference_mae.svg` — per-font interference MAE with within-font source-grouped intervals.
- `figures/fig4_feature_error_associations.svg` — exploratory Spearman heatmap for the eight existing features.
- `figures/fig5_source_error_concentration.svg` — cumulative squared-error concentration by source group.

## Files

| File | Purpose |
| --- | --- |
| `summary_metrics.csv` | scope-level metrics, intervals, calibration and concentration |
| `condition_calibration.csv` | target/prediction calibration by condition |
| `font_condition_metrics.csv` | font x condition metrics and paired-by-case contrasts |
| `severity_metrics.csv` | primary family/severity metrics; elastic 0.5 px is diagnostic and excluded |
| `source_group_metrics.csv` | one row per source group and scope |
| `case_metrics.csv` | text-case interference summaries |
| `source_characteristic_associations.csv` | exploratory background/text-length associations |
| `feature_error_associations.csv` | exploratory feature rank associations |
| `paired_condition_contrasts.csv` | source-paired condition comparisons |
| `error_concentration.csv` | source-level squared-error concentration |
| `hypotheses_and_followups.md` | at most three testable hypotheses and future acceptance criteria |
| `verification_checks.csv` | fail-closed checks and actual outcomes |
| `analysis_protocol.json` | inputs, hashes, decisions, seed, software and output hashes |

## Reproduction

From the repository root, on the pinned environment:

```bash
.venv/bin/python -m captcha_quality.interference_error_analysis run \
  --config configs/interference_error_analysis_v0_1_0.toml
```

The run refuses to overwrite an existing report unless `--overwrite` is given.
It refuses to write anywhere outside a new
`reports/interference_error_analysis_v*` directory. It runs the M8E integrity
verifier, the M8C post-execution lifecycle verifier and the full M8C replay
before writing the report.

## Verification

82/82 pre-package analysis checks passed.
`verification_checks.csv` records the complete package check set as well,
including output membership, immutability and the external historical
verifiers. The package records the exact Git revision, the clean/dirty state at
the start, the input hashes, the model artifact and card hashes, and the
external verifier outcomes. The external verifiers are also run again by the
final project-level checks.

## Limitations

- Synthetic image-similarity analysis only; SSIM is not human readability,
  solvability, security or production robustness.
- Ten purposively selected fonts and thirty committed text cases are not a
  random sample of fonts, users or deployments. Font identity is entangled with
  glyph shape, renderer and target degradation.
- Source-grouped intervals describe this source distribution; they do not
  estimate a font or user population.
- The 0.20 large-error threshold was fixed in the protocol before these subgroup
  tables were generated, but earlier reconnaissance had inspected aggregate and
  severity summaries. Every threshold-based subgroup proportion is therefore
  labelled exploratory.
- Feature associations are descriptive and exploratory. They are not corrected
  for source clustering, do not identify mechanism and do not justify selecting
  or adding a feature.
- The elastic 0.5 px rung is diagnostic and is excluded from the primary
  severity table. Clean-reference R² is suppressed because its target is
  constant.
- M3/M5C metrics are not treated as like-for-like comparisons to M8C. M8C is a
  new cross-font benchmark with a different source, font and condition
  composition.
