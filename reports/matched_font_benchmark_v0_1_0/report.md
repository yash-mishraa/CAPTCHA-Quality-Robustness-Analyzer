# Matched cross-font generalisation benchmark (M8C)

## Scope and question

This report evaluates the exact frozen M5C champion zero-shot on a matched
cross-font benchmark. The target is **SSIM between each corrupted synthetic
image and its source-specific clean reference**. PSNR is recorded as a
diagnostic only. Nothing in this report measures human readability, CAPTCHA
solvability, CAPTCHA security or production robustness.

The central generalisation limitation is the number of **font identities**, not
the number of text cases: seven held-out identities are observed over thirty
matched cases. Three hundred sources are not three hundred independent fonts.

## Frozen identity and protocol chain

| Item | Value |
| --- | --- |
| Milestone | M8C (execution of the M8B frozen protocol) |
| Execution protocol | `configs/cross_font_benchmark_protocol_v0_2_0.toml` |
| Protocol SHA-256 | `059388572f264fb2332f3427da774f1ab30f1339a27f19fd3e98e98163d4574c` |
| Generation configuration hash | `147df5002e93e5f280d60ed10ed417d3100c8c9ccb24191b065f486d3b580e06` |
| Text-case manifest | `configs/cross_font_text_cases_v0_1_0.toml` |
| Text-case file SHA-256 | `fa9a2c57f7aec6dda214a933f3b639160ef2f2f2996129d64b225f2e60f46f48` |
| Canonical text-case manifest hash | `d1d55b841d96d507f5e8031268ce454bbec238e6bfc32f9aaa47322f3eeeb010` |
| Champion | `expanded_random_forest` / calibration `none` |
| Champion artifact SHA-256 | `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0` |
| Champion card SHA-256 | `9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8` |
| Feature set | `v1`: intensity_mean, intensity_std, contrast_rms, ink_fraction, laplacian_variance, edge_density, histogram_entropy, noise_sigma_immerkaer |
| Target | `ssim_target` |
| Dataset version | `matched-font-benchmark-v0.1.0` |
| Dataset manifest SHA-256 | `35ae405fd084070bf4d6c235638cbda6671573389e880dcca2a63d8b66049743` |

The champion was loaded with `joblib` and used unchanged. No fit, partial fit,
calibration, selection, thresholding or refitting was performed. No feature was
added, removed, reordered or recomputed with a different definition.

## Dataset composition

| component | expected | realised | matches |
| --- | --- | --- | --- |
| fonts | 10 | 10 | yes |
| cases | 30 | 30 | yes |
| sources | 300 | 300 | yes |
| rows | 6300 | 6300 | yes |
| primary_rows | 5700 | 5700 | yes |
| diagnostic_rows | 600 | 600 | yes |
| clean_rows | 300 | 300 | yes |
| rows:gaussian_blur | 1200 | 1200 | yes |
| rows:gaussian_noise | 1200 | 1200 | yes |
| rows:thin_lines | 1200 | 1200 | yes |
| rows:arc_strokes | 1200 | 1200 | yes |
| rows:elastic_primary | 900 | 900 | yes |
| rows:elastic_diagnostic | 300 | 300 | yes |
| images | 6300 | 6300 | yes |
| feature_rows | 6300 | 6300 | yes |
| source_cases_rows | 300 | 300 | yes |
| ground_truth_rows | 300 | 300 | yes |
| duplicate_row_ids | 0 | 0 | yes |
| duplicate_sample_ids | 0 | 0 | yes |
| distinct_source_ids | 300 | 300 | yes |
| min_rows_per_source | 21 | 21 | yes |
| max_rows_per_source | 21 | 21 | yes |
| missing_row_ids | 0 | 0 | yes |
| missing_sample_ids | 0 | 0 | yes |
| missing_image_paths | 0 | 0 | yes |
| missing_feature_values | 0 | 0 | yes |
| nonfinite_ssim_targets | 0 | 0 | yes |
| clean_psnr_infinite_rows | 300 | 300 | yes |
| sources:arial | 30 | 30 | yes |
| rows:arial | 630 | 630 | yes |
| sources:courier_new | 30 | 30 | yes |
| rows:courier_new | 630 | 630 | yes |
| sources:times_new_roman | 30 | 30 | yes |
| rows:times_new_roman | 630 | 630 | yes |
| sources:georgia | 30 | 30 | yes |
| rows:georgia | 630 | 630 | yes |
| sources:monaco | 30 | 30 | yes |
| rows:monaco | 630 | 630 | yes |
| sources:impact | 30 | 30 | yes |
| rows:impact | 630 | 630 | yes |
| sources:verdana | 30 | 30 | yes |
| rows:verdana | 630 | 630 | yes |
| sources:trebuchet_ms | 30 | 30 | yes |
| rows:trebuchet_ms | 630 | 630 | yes |
| sources:chalkduster | 30 | 30 | yes |
| rows:chalkduster | 630 | 630 | yes |
| sources:luminari | 30 | 30 | yes |
| rows:luminari | 630 | 630 | yes |
| fonts:familiar | 3 | 3 | yes |
| rows:familiar | 1890 | 1890 | yes |
| fonts:m7b_held_out | 3 | 3 | yes |
| rows:m7b_held_out | 1890 | 1890 | yes |
| fonts:new_held_out | 4 | 4 | yes |
| rows:new_held_out | 2520 | 2520 | yes |
| text_length:4 | yes | yes | yes |
| text_length:5 | yes | yes | yes |
| text_length:6 | yes | yes | yes |
| split | eval | eval | yes |
| distinct_image_pixel_hashes | 6300 | 6300 | yes |
| distinct_reference_pixel_hashes | 300 | 300 | yes |

The dataset contains exactly 6300 rows, 5700 primary rows and 600 diagnostic rows. Clean-reference rows and the 0.5 px elastic rows are diagnostic only and are excluded from every primary metric and from the primary bootstrap.

## Metric definitions and aggregation

- MAE is the mean absolute error, `mean(|y - y_hat|)`.
- RMSE is the population root mean squared error, `sqrt(mean((y - y_hat)^2))`.
- R-squared is `1 - SS_res / SS_tot`, where `SS_tot` is taken around the target
  mean. It is undefined for a constant target and is suppressed when the target
  standard deviation is below 0.005. Suppressed and undefined
  slices remain in the tables with an explicit status.
- Primary scopes are the M7D/M8B primary scopes: all primary rows, original
  blur and noise, each blur and noise family, interference, thin lines, arc
  strokes and geometric primary. Clean-reference and 0.5 px elastic rows are
  diagnostic and are excluded from every primary metric, primary endpoint and
  primary bootstrap.
- Overall metrics aggregate all ten fonts and all 300 sources for the named
  scope. Font metrics restrict the same primary rows to one font; group,
  family and severity metrics apply the corresponding subset. Per-source
  metrics use the 19 primary rows of one source.
- Uncertainty uses 2,000 percentile-bootstrap replicates. The source-grouped
  scheme resamples whole source groups; the matched-case scheme resamples whole
  cases and keeps all ten font renders of a case together. Neither interval is
  a significance test or a font-population estimate.

## Primary results

| scope | n_rows | n_sources | mae | rmse | r2 | r2_status |
| --- | --- | --- | --- | --- | --- | --- |
| all_primary | 5700 | 300 | 0.059607 | 0.088610 | 0.732898 | reported |
| original_blur_noise | 2400 | 300 | 0.032735 | 0.057131 | 0.899708 | reported |
| gaussian_blur | 1200 | 300 | 0.026499 | 0.049943 | 0.943091 | reported |
| gaussian_noise | 1200 | 300 | 0.038971 | 0.063510 | 0.735669 | reported |
| interference | 2400 | 300 | 0.090825 | 0.118525 | -0.430242 | reported |
| thin_lines | 1200 | 300 | 0.102073 | 0.132114 | -0.575158 | reported |
| arc_strokes | 1200 | 300 | 0.079578 | 0.103162 | -0.298911 | reported |
| geometric_primary | 900 | 300 | 0.048018 | 0.059687 | 0.162498 | reported |

By font:

| font_id | font_family | font_group | n_rows | mae | rmse | r2 | r2_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arial | Arial | familiar | 570 | 0.053974 | 0.082186 | 0.765520 | reported |
| georgia | Georgia | m7b_held_out | 570 | 0.054207 | 0.081849 | 0.762135 | reported |
| trebuchet_ms | Trebuchet MS | new_held_out | 570 | 0.054890 | 0.080456 | 0.774350 | reported |
| luminari | Luminari | new_held_out | 570 | 0.055256 | 0.081057 | 0.761790 | reported |
| monaco | Monaco | m7b_held_out | 570 | 0.056597 | 0.080589 | 0.777521 | reported |
| verdana | Verdana | new_held_out | 570 | 0.056819 | 0.085557 | 0.740090 | reported |
| chalkduster | Chalkduster | new_held_out | 570 | 0.057998 | 0.084146 | 0.735029 | reported |
| times_new_roman | Times New Roman | familiar | 570 | 0.059806 | 0.088395 | 0.708774 | reported |
| courier_new | Courier New | familiar | 570 | 0.068656 | 0.106395 | 0.565558 | reported |
| impact | Impact | m7b_held_out | 570 | 0.077869 | 0.109507 | 0.723482 | reported |

By font group:

| font_group | n_fonts | n_rows | n_sources | mae | rmse | r2 | r2_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| familiar | 3 | 1710 | 90 | 0.060812 | 0.092894 | 0.683543 | reported |
| m7b_held_out | 3 | 1710 | 90 | 0.062891 | 0.091626 | 0.750091 | reported |
| new_held_out | 4 | 2280 | 120 | 0.056241 | 0.082831 | 0.753114 | reported |

By corruption family:

| variant_family | n_rows | n_sources | mae | rmse | r2 | r2_status |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian_blur | 1200 | 300 | 0.026499 | 0.049943 | 0.943091 | reported |
| gaussian_noise | 1200 | 300 | 0.038971 | 0.063510 | 0.735669 | reported |
| thin_lines | 1200 | 300 | 0.102073 | 0.132114 | -0.575158 | reported |
| arc_strokes | 1200 | 300 | 0.079578 | 0.103162 | -0.298911 | reported |
| elastic | 900 | 300 | 0.048018 | 0.059687 | 0.162498 | reported |

By corruption severity (primary slices only):

| variant_family | severity_level | severity_parameter | n_rows | mae | rmse | r2 | r2_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_blur | 1 | 1.000000 | 300 | 0.035999 | 0.077504 | -18.653829 | reported |
| gaussian_blur | 2 | 2.000000 | 300 | 0.023596 | 0.034531 | 0.453591 | reported |
| gaussian_blur | 3 | 3.500000 | 300 | 0.018875 | 0.025796 | 0.785115 | reported |
| gaussian_blur | 4 | 5.000000 | 300 | 0.027527 | 0.045962 | 0.556401 | reported |
| gaussian_noise | 1 | 6.000000 | 300 | 0.047469 | 0.085314 | -6.471875 | reported |
| gaussian_noise | 2 | 12.000000 | 300 | 0.038352 | 0.060836 | -0.837642 | reported |
| gaussian_noise | 3 | 20.000000 | 300 | 0.039978 | 0.057722 | -0.535194 | reported |
| gaussian_noise | 4 | 32.000000 | 300 | 0.030085 | 0.042696 | 0.174234 | reported |
| thin_lines | 1 | 1.000000 | 300 | 0.036032 | 0.047481 | -2.413947 | reported |
| thin_lines | 2 | 2.000000 | 300 | 0.058144 | 0.072745 | -2.672975 | reported |
| thin_lines | 3 | 4.000000 | 300 | 0.106234 | 0.117428 | -4.497036 | reported |
| thin_lines | 4 | 7.000000 | 300 | 0.207882 | 0.220183 | -10.885486 | reported |
| arc_strokes | 1 | 1.000000 | 300 | 0.030695 | 0.035490 | -2.184325 | reported |
| arc_strokes | 2 | 2.000000 | 300 | 0.046433 | 0.053559 | -2.290097 | reported |
| arc_strokes | 3 | 4.000000 | 300 | 0.076290 | 0.084842 | -3.048105 | reported |
| arc_strokes | 4 | 7.000000 | 300 | 0.164892 | 0.176757 | -8.421074 | reported |
| elastic | 2 | 1.000000 | 300 | 0.045825 | 0.050685 | -27.922726 | reported |
| elastic | 3 | 2.000000 | 300 | 0.024068 | 0.030019 | -0.232572 | reported |
| elastic | 4 | 3.000000 | 300 | 0.074161 | 0.084955 | -3.123149 | reported |

## Clean-image and diagnostic results

These rows are reported separately and are never merged into a primary metric.

| scope | n_rows | n_sources | mae | rmse | r2 | r2_status |
| --- | --- | --- | --- | --- | --- | --- |
| clean_reference | 300 | 300 | 0.007352 | 0.013863 | n/a | suppressed_near_degenerate_target |
| diagnostic_0_5px | 300 | 300 | 0.029109 | 0.036415 | n/a | suppressed_near_degenerate_target |

The clean reference has a constant target of 1.0, so its target standard
deviation is zero and R-squared is undefined; MAE and RMSE are still reported.
The 0.5 px elastic row is diagnostic because the M7B/M7D agreement excludes
that near-identity condition from primary scoring.

## Uncertainty

Source-grouped and matched-case percentile bootstrap intervals, 2,000
replicates, resampling whole source groups or whole matched cases. The intervals
are descriptive uncertainty summaries over the observed sources and font
identities. They are not significance tests and do not estimate a font
population.

| scheme | scope | metric | estimate | ci_low | ci_high | n_groups | n_samples | r2_defined_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_grouped | all_primary | mae | 0.059607 | 0.058304 | 0.060987 | 300 | 5700 | n/a |
| source_grouped | all_primary | rmse | 0.088610 | 0.086638 | 0.090613 | 300 | 5700 | n/a |
| source_grouped | all_primary | r2 | 0.732898 | 0.721910 | 0.744091 | 300 | 5700 | 1.000000 |
| matched_case | all_primary | mae | 0.059607 | 0.058122 | 0.061184 | 30 | 5700 | n/a |
| matched_case | all_primary | rmse | 0.088610 | 0.086270 | 0.091153 | 30 | 5700 | n/a |
| matched_case | all_primary | r2 | 0.732898 | 0.714721 | 0.749193 | 30 | 5700 | 1.000000 |

## Recorded familiar-font baseline

The recorded M5C familiar-font test-slice metrics are shown beside the new
results exactly as read from `reports/retention_aware_selection_v0_1_0/test_slice_metrics.csv`. They were not recomputed,
refitted or selected. The new primary aggregate is over all ten fonts, so it is
not a like-for-like familiar-font comparison. The new `group:familiar` row is the
closest new analogue, but it is still a different matched-case dataset.

| source | subset | matched_scope | protocol_role | mae | rmse | r2 | n_rows | n_sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recorded_m5c_familiar_baseline | clean_reference | clean_reference | diagnostic | 0.001267 | 0.002275 | n/a | 12 | 12 |
| recorded_m5c_familiar_baseline | original_blur_noise | original_blur_noise | primary | 0.017410 | 0.023328 | 0.981409 | 96 | 12 |
| recorded_m5c_familiar_baseline | m3_test_overall | all_primary | primary | 0.015616 | 0.022007 | 0.987157 | 108 | 12 |
| recorded_m5c_familiar_baseline | diagnostic_0_5 | diagnostic_0_5px | diagnostic | 0.026559 | 0.035917 | -197.406358 | 12 | 12 |
| recorded_m5c_familiar_baseline | original_blur_noise_gaussian_blur | gaussian_blur | primary | 0.016146 | 0.024243 | 0.984108 | 48 | 12 |
| recorded_m5c_familiar_baseline | original_blur_noise_gaussian_noise | gaussian_noise | primary | 0.018674 | 0.022374 | 0.971882 | 48 | 12 |
| new_matched_font_benchmark | all_primary | all_primary | primary | 0.059607 | 0.088610 | 0.732898 | 5700 | 300 |
| new_matched_font_benchmark | original_blur_noise | original_blur_noise | primary | 0.032735 | 0.057131 | 0.899708 | 2400 | 300 |
| new_matched_font_benchmark | gaussian_blur | gaussian_blur | primary | 0.026499 | 0.049943 | 0.943091 | 1200 | 300 |
| new_matched_font_benchmark | gaussian_noise | gaussian_noise | primary | 0.038971 | 0.063510 | 0.735669 | 1200 | 300 |
| new_matched_font_benchmark | clean_reference | clean_reference | diagnostic | 0.007352 | 0.013863 | n/a | 300 | 300 |
| new_matched_font_benchmark | diagnostic_0_5px | diagnostic_0_5px | diagnostic | 0.029109 | 0.036415 | n/a | 300 | 300 |
| new_matched_font_benchmark | group:familiar | all_primary | primary | 0.060812 | 0.092894 | 0.683543 | 1710 | 90 |
| new_matched_font_benchmark | group:m7b_held_out | all_primary | primary | 0.062891 | 0.091626 | 0.750091 | 1710 | 90 |
| new_matched_font_benchmark | group:new_held_out | all_primary | primary | 0.056241 | 0.082831 | 0.753114 | 2280 | 120 |

## Findings

- The ten-font primary MAE is 0.059607 against a recorded familiar-font M5C test-slice MAE of 0.015616; the new value is higher. The two are not a like-for-like comparison: the recorded slice contains familiar fonts only.

- Font-level MAE ranges from 0.053974 (arial) to 0.077869 (impact). These are descriptive differences over the observed matched cases, not a font-population effect.

- R-squared is negative for thin_lines (-0.575158), arc_strokes (-0.298911): on those families the frozen champion is worse than predicting the slice mean. This mixed result is reported as observed and is not tuned away.

- Georgia, Monaco and Impact were already evaluated in M7B; these results are not an independent replication of the M7B three-font finding. Verdana, Trebuchet MS, Chalkduster and Luminari are genuinely new to project evaluation.

## Interpretation limits

- The target is synthetic-reference SSIM. PSNR is diagnostic only.
- No claim is made about human readability, CAPTCHA solvability, security or
  production robustness.
- Seven held-out font identities, not 300 cases, are the central
  generalisation limitation. Font-level differences are descriptive for the
  observed matched cases.
- Georgia, Monaco and Impact were already evaluated in M7B and are not an
  independent new-font replication. Verdana, Trebuchet MS, Chalkduster and
  Luminari are genuinely new held-out identities for this project.
- The ten faces are pinned Apple-bundled macOS builds. The result does not
  establish cross-platform generalisation.
- No post-hoc threshold, endpoint change, model selection or new champion was
  introduced after seeing these results.
- R-squared is suppressed where the target standard deviation is below
  0.005; those slices are shown with an explicit status rather
  than silently dropped.

## Provenance and environment

The report was generated by the M8C execution module. The full machine-readable
environment, input hashes, prediction counts and reproducibility comparison are
recorded in `run_metadata.json`.

| Item | Value |
| --- | --- |
| Python | `3.13.16` |
| Platform | `macOS-27.0.1-arm64-arm-64bit-Mach-O` |
| scikit-learn | `1.9.1` |
| numpy | `2.5.4` |
| pandas | `3.0.6` |
| Pillow | `12.3.0` |
| Code revision at execution | recorded in `run_metadata.json` |
| Inputs verified | 36 / 36 |

## Output membership

The dataset has exactly six top-level files plus `images/` with 6,300 PNGs in
six subdirectories. The report directory has exactly fifteen files. No font
binary is present in the repository, dataset or report.

## Reproducibility

The prescribed check was performed by repeating the complete generation,
inference and report pipeline in an isolated temporary directory and comparing
identities, image bytes, tables and report text. The outcome, including any
metadata-only difference, is recorded in `run_metadata.json`.
