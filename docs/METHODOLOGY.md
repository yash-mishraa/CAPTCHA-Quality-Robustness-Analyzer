# Methodology

**Status:** Milestones 0–3, M4A, M4B, M5A, M5B, M5C, M5D, M6A, M6B, M6C,
M6D, M7A, M7B and M7C complete. The target and M3 dataset are frozen, the
conventional benchmark, grouped interference/geometric benchmarks,
handcrafted-feature follow-ups, the read-only dataset audit, the held-out-font
generalisation benchmark and the read-only cross-font evaluation analysis are
implemented, and the CNN comparison and dashboard remain planned.
**Last updated:** 2026-10-21

## 1. Research framing

**Question.** Can image quality and corruption severity in a synthetic
CAPTCHA-style corpus be predicted from image content alone, and how does
predictive performance vary across corruption families?

**Primary task (implemented).** Supervised regression of a defined
reference-based image-quality score.

**Secondary baseline task (planned).** Classification of corruption family and,
separately, severity level. These are treated as diagnostic baselines, not the
project's main claim.

**What is not claimed.** No human solvability prediction, no CAPTCHA-strength
ranking, and no generalisation claim to production systems. Human responses are
absent from the dataset, so any statement about human difficulty would be
unfounded.

## 2. Synthetic data generation

### 2.1 Rendering

Each source starts as a clean rendering of a known string:

- **Character set:** configurable; visually ambiguous glyphs (`0/O`, `1/I/L`)
  are excluded from the default set. Characters are drawn uniformly.
- **Text length:** drawn from an explicit distribution. The reference dataset
  uses lengths `[4, 5, 6]` with weights `[0.3, 0.4, 0.3]`; the pilot uses length
  5 only.
- **Fonts:** logical IDs resolve to documented system paths. A font is either
  available or reported unavailable with its searched paths; it is **never**
  silently substituted. Dataset generation requires every configured font,
  while the M1 pilot allowed `min_required`. The M3 dataset uses Arial, Courier
  New and Times New Roman. M7B uses a separate, pinned set of fonts that do not
  appear in M3: Georgia (proportional serif), Monaco (monospaced humanist sans)
  and Impact (condensed heavy display sans). Font files are verified by SHA-256
  and glyph rendering before generation; a missing or changed file fails
  closed.
- **Layout:** canvas 192x64 grayscale, glyphs rasterised at 4x and downsampled
  with LANCZOS. Per-character vertical jitter is configurable; the dataset adds
  optional per-character spacing jitter and whole-block placement jitter.
- **Background policy:** `white` (constant 255, the M1 condition) or
  `flat_random`, which draws one constant gray level per source from a
  configured inclusive range. The reference dataset uses `[240, 255]`. Texture
  backgrounds are deliberately not supported because they would confound the
  additive-noise family.
- **Bounds:** dataset configurations enable `center_ink` (centre the actual ink
  bounding box, not the draw anchor) and `strict_bounds` (refuse a render whose
  ink touches the canvas edge).
- **Out of scope for the frozen dataset:** rotation, colour, texture backgrounds
  and occlusion. The separate M4A pilot adds two occlusion/visual-interference
  variants without changing the frozen dataset.

Rendering is deterministic given the source seed and configuration. The M1
pilot keeps the original defaults (`center_ink=False`, `strict_bounds=False`,
white background, no extra jitter), so `EXP-0001` remains byte-reproducible.

### 2.2 Distortion families

Each family is applied with a numeric parameter from an ordered severity ladder.
Severity 0 is the identity condition and is represented once per source as the
shared clean sample, not duplicated for every family.

| Family | Parameterisation | Status |
| --- | --- | --- |
| Gaussian blur | Pillow Gaussian blur radius in pixels; `[0, 1, 2, 3.5, 5]` | implemented |
| Additive Gaussian noise | Gaussian sigma on the 0–255 scale; `[0, 6, 12, 20, 32]`; independent seeded realisation per `(source, family, severity)` | implemented |
| Contrast / brightness, geometric warp, compression | — | deferred for the frozen dataset |
| Occlusion / visual interference (`thin_lines`, `arc_strokes`) | density ladder `[1, 2, 4, 7]` strokes; fixed 1 px black; seeded geometry | M4A pilot only (§5.9), not in the frozen dataset |
| Smooth elastic warp | maximum displacement ladder `[0.5, 1.0, 2.0, 3.0]` px; 5x13 smoothed field; bilinear, edge padding | M5A pilot only (§5.11), not in the frozen dataset |

The ladders are not re-tuned per image. Each variant records family, severity,
numeric parameters, and the seed used, so effects are attributable. In the
dataset pipeline, severity levels must be strictly increasing; a ladder that
would produce an identical image raises an error.

### 2.3 Metadata and manifest

The public manifest (`manifest.csv`) has one row per sample and the following
fields:

`sample_id`, `source_id`, `variant_id`, `split_group_id`, `split`,
`text_length`, `font_id`, `render_seed`, `variant_seed`, `corruption_type`,
`severity_level`, `corruption_parameters`, `image_path`, `reference_path`,
`ssim_target`, `psnr_target`, `psnr_infinite`, `height`, `width`,
`dataset_version`, `generation_config_hash`, `text_sha256`, `image_sha256`,
`reference_sha256`.

Identifier rules:

- `source_id = "src-" + first16(sha256(canonical_json(generation_config_hash,
  font_id, index, text, text_length, render_seed)))`.
- `variant_id` is `clean_s0` or `<family>_s<severity>`.
- `sample_id = source_id + "__" + variant_id`.
- `split_group_id = source_id`, so every variant inherits its source's split.

**Raw text policy.** Text is not stored in the public manifest, `sources.csv`,
or `dataset_summary.json`. A `text_sha256` identifier is public. The raw string
lives only in `ground_truth.csv`, an internal file inside the generated dataset
directory, used for reproduction and verification. The dataset directory is
Git-ignored and contains synthetic strings only.

**Targets.** `ssim_target` is computed from the exact reference/variant pair.
Finite PSNR is recorded when meaningful. For clean identity pairs PSNR is
`+inf`, which CSV cannot represent faithfully; those rows set
`psnr_target` empty and `psnr_infinite = true`. Verification recomputes every
row and rejects any mismatch.

**Hashing and atomicity.** Each image and reference stores a SHA-256 of its raw
pixels. Generation writes to a sibling `.partial` directory and renames it into
place only on success; an existing target is never overwritten unless
`--overwrite` is explicit. Manifests and summaries contain no timestamps, so
repeated runs are byte-comparable.

### 2.4 Dataset scale and splits

The initial corpus is deliberately small enough to inspect and regenerate on a
single machine.

| Configuration | Fonts x sources/font | Sources | Samples | Purpose |
| --- | --- | --- | ---: | --- |
| `configs/dataset_pilot.toml` | 3 x 3 | 9 | 81 | End-to-end pipeline check under M1-like rendering |
| `configs/dataset_v0_1_0.toml` | 3 x 20 | 60 | 540 | Reference dataset for M2 diagnostics |
| `configs/heldout_font_benchmark_v0_1_0.toml` | 3 held-out fonts x 20 | 60 | 1,260 | M7B zero-shot font-generalisation benchmark; one `eval` split |

**Split strategy (fixed before generation).**

- One group per clean source; all of its variants share the group.
- One group appears in exactly one of train/val/test. There is no variant-level
  split.
- Ratios: `0.6 / 0.2 / 0.2`. Explicit split seed `20261021`, independent of the
  dataset seed.
- Minimum two sources per split, validated before generation.
- Font-stratified allocation: for each font, groups are shuffled with a seed
  derived from `(split seed, font)`, then allocated by largest remainder. If a
  split would receive zero groups from a font, the allocation is repaired by
  taking a group from the largest split; if that is impossible, generation
  fails with an explicit message.
- Because every source has every family/severity, split-level family and
  severity composition is proportional by construction. Font, text-length, and
  background distributions are reported per split rather than forced to match.
- A held-out **font** or **configuration** split is intentionally reserved for a
  later generalisation check. The random grouped split used here is not evidence
  of cross-generator generalisation.

The reference dataset splits contain 36/12/12 sources and 324/108/108 samples;
each split contains all three fonts (12/4/4 sources per font). M7B deliberately
does **not** create a train/validation/test split: its 60 held-out-font sources
form one untouched `eval` split, because the model is frozen and no selection or
calibration is permitted. The source group remains the independent unit.

## 3. Reference-based quality targets

### 3.1 Definition

For a clean reference image `R` and degraded image `D` of the same text, compute
full-reference metrics such as SSIM, MS-SSIM, and PSNR. The primary target is a
single, explicitly documented scalar derived from these metrics; candidates are:

1. SSIM alone (bounded, interpretable, standard).
2. A fixed weighted combination of normalised SSIM, MS-SSIM, and PSNR.
3. A learned or PCA-derived composite (only if M1 shows the simple targets are
   inadequate; introduces extra complexity).

The choice is fixed **before** model comparison and recorded in the experiment
log. Targets are computed at a documented colour space, bit depth, and
downsampling convention, because SSIM and PSNR are sensitive to all three.

### 3.2 Why reference-based

- Available by construction for every synthetic sample.
- Deterministic and cheap to recompute.
- Directly interpretable as "how far did this transformation move the image
  from its clean source?"

### 3.3 Known limitations

- Full-reference metrics assume pixel correspondence; aggressive geometric
  warps can punish semantically legible images.
- SSIM/PSNR correlate imperfectly with human perception, especially for text.
- A clean reference is unavailable at inference in a real deployment; this
  project predicts the score from the degraded image, but the *training label*
  still depends on a reference. That distinction must be stated in reports.
- Targets may saturate at high severity (all images equally destroyed), creating
  a ceiling that compresses the regression range.

### 3.4 M1 validation gate (predeclared)

**Status:** implemented and evaluated in Milestone 1. Thresholds were fixed in
`configs/pilot_target_validation.toml` and in this section **before** the
recorded pilot run; no threshold was changed after inspecting results.

#### 3.4.1 Discrepancy resolution

The Milestone 0 version of this section listed five qualitative criteria
(sensitivity, separation, stability, non-degeneracy, sanity check). The
Milestone 0 report described it the same way, so there is no contradiction
between the report and the document, but the criteria were not operational:
there were no statistics, no numeric thresholds, and no verdict classes.

Milestone 1 resolves this by mapping each criterion to a computable statistic
and a predeclared threshold (below). One original sub-criterion could **not** be
evaluated as written: "not almost perfectly predicted by a single trivial
feature (for example global contrast)" requires the handcrafted feature set that
is not built until Milestone 2. For M1 it is replaced by the one trivial feature
available today — the clean image's **ink coverage** (`ink_fraction`) — and the
full feature-predictability check was explicitly deferred to M2. It is now
implemented and reported in §4.5.

#### 3.4.2 Operational criteria

Each criterion is evaluated separately for every
`(corruption family, metric)` combination.

| ID | Criterion | Statistic | Threshold | Rationale |
| --- | --- | --- | --- | --- |
| `direction` | Score falls as severity rises | Spearman rho between severity and score over all pairs (ties averaged, `+inf` ranked highest) | rho <= **-0.90** | A near-monotone response is the minimum for an ordinal target; -0.90 tolerates small rank noise but rejects flat or ambiguous trends. |
| `aggregate_trend` | Mean score falls at every step | Level means (finite values only) | Strict decrease for every adjacent pair | The severity ladder must be meaningful on average, independent of individual noise realisations. |
| `adjacent_separation` | Adjacent levels are distinguishable | Paired Cohen's d of the within-trial difference (same clean reference) | d <= **-0.50** for every evaluable pair | Cohen's medium-effect convention: the next severity must move the score by at least half a standard deviation of paired differences. Pairs with no finite difference (PSNR's `+inf` baseline) are excluded and counted. |
| `violation_rate` | Individual samples rarely reverse | Fraction of trials whose score increases between adjacent levels | <= **0.35** per step | Noise realisations may occasionally invert. Beyond roughly one in three, an individual sample's severity ordering is not trustworthy. Ties are not violations. |
| `stability_font` | Trend is not font-driven | Spearman rho recomputed within each font | rho <= **-0.80** in every font (>= 2 fonts) | A target that only works for one typeface cannot support later cross-font generalisation checks. |
| `stability_seed` | Trend is not seed-driven | Spearman rho recomputed within each repeat seed | rho <= **-0.80** in every seed (>= 2 seeds) | The direction must survive independent render/noise realisations. |
| `non_degeneracy` | Enough variation, not a restatement of a trivial feature | Eta squared for severity; span of level means; strongest within-level correlation with ink coverage | eta^2 >= **0.50**; span >= **0.10** (SSIM) or **3.0 dB** (PSNR); abs(r_ink) <= **0.95** | At least half the score variance should come from the manipulated factor; the mean must move by a practically visible amount; a score explained almost perfectly by ink coverage adds nothing beyond a trivial feature. |
| `clean_sanity` | Perfect self-similarity | Clean-vs-clean SSIM, PSNR, byte identity; severity-0 identity rows | SSIM == 1.0 exactly; PSNR == `+inf`; byte-identical | Full-reference metrics must be exactly perfect for identical inputs, otherwise later scores are not interpretable. |

Threshold provenance: the numeric values were chosen from conventional effect
sizes and measurement reasoning (above), not from observed pilot results. The
adjacent-separation threshold uses Cohen's "medium" convention; the rank
thresholds require a strong association without demanding perfection; the
variance thresholds are set well above plausible floating-point or resampling
noise.

#### 3.4.3 Verdict mapping

| Verdict | Rule |
| --- | --- |
| `PASS` | Every criterion passes. |
| `CONDITIONAL` | `direction`, `aggregate_trend`, `non_degeneracy`, and `clean_sanity` pass, but at least one separation or stability criterion fails. The metric is usable with the stated limitation. |
| `FAIL` | Any prerequisite criterion (`direction`, `aggregate_trend`, `non_degeneracy`, `clean_sanity`) fails for that family/metric combination. |
| `DEFERRED` | Not evaluated in this milestone, with a recorded reason (MS-SSIM). |

A `PASS` means the metric is a **candidate image-quality target under the tested
conditions**. It does not establish human readability, perceptual validity, or
that a model will generalise.

#### 3.4.4 Metrics and conventions

- **SSIM** (scikit-image `structural_similarity`): 2-D grayscale `uint8`,
  `data_range=255.0`, `win_size=11`, `gaussian_weights=True`, `sigma=1.5`,
  `use_sample_covariance=False` (paper-style Gaussian window).
- **PSNR** (scikit-image `peak_signal_noise_ratio`): `data_range=255.0`.
  Identical images return `math.inf` exactly and are computed before calling
  scikit-image (which would otherwise divide by zero). Infinite values are
  excluded from numeric aggregates and counted explicitly; `+inf` ranks highest
  for rank statistics.
- **MS-SSIM:** `DEFERRED`. scikit-image 0.26 provides no MS-SSIM, and the
  maintained implementations (`pytorch-msssim`, `torchmetrics`) require PyTorch,
  which is a declared but uninstalled optional extra. Adding PyTorch solely for
  one diagnostic metric is not justified in M1; it is revisited with the CNN work.
- **Alignment contract:** both images must be 2-D `uint8`, at least 11x11 px,
  and identically shaped. Violations raise `ImageContractError`.

### 3.5 M1 pilot implementation

The pilot is intentionally smaller than the planned Milestone 2 pipeline. It
renders, corrupts, and scores a few dozen source images with two families:

- **Canvas:** 192x64 px grayscale, rendered at 4x and downsampled with LANCZOS.
- **Text:** 5 characters from a 31-character set with ambiguous glyphs removed;
  deterministic per `(run seed, font, text index)`.
- **Fonts:** Arial, Courier New, Times New Roman from documented system paths.
  Missing fonts are recorded as unavailable and never substituted; fewer than
  `min_required` available fonts aborts the run.
- **Trials:** 3 fonts x 4 texts x 3 repeat seeds = **36 clean images**; each is
  evaluated at 5 severity levels in 2 families = **360 corrupted-pair rows**,
  plus **36 clean self-comparisons**. Severity 0 is the identity baseline.
- **Blur ladder:** radius `[0, 1, 2, 3.5, 5]` px (Pillow Gaussian blur).
- **Noise ladder:** sigma `[0, 6, 12, 20, 32]` on the 0-255 scale, with an
  independent seeded realisation per `(trial, family, severity)`.
- **Reproducibility:** every text, jitter, and noise draw derives from a hashed
  seed; two runs with the same configuration produce byte-identical tables.
- **Outputs:** `reports/pilot/` (Git-ignored) contains `trials.csv`,
  `sanity.csv`, summary tables, `gate_results.json`, `summary.md`, and contact
  sheets for visual inspection.

Limitations of this pilot: one canvas size, one text length, three fonts, two
families, and 36 source images. It validates metric behaviour, not dataset
scale. Because clean images are read-only and corruption functions copy their
input, the reference used at every severity is byte-identical; the runner also
verifies this invariant after each trial.

### 3.6 Results and decision

Recorded in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), experiment `EXP-0001`.


## 4. Handcrafted features and M2 diagnostics

### 4.1 Feature definitions

Features are computed from the **degraded image only**; the extractor has no
access to the reference, targets, or generation metadata. All features are
scalars, deterministic, and returned in a fixed order
(`FEATURE_SET_VERSION = "v1"`).

| Feature | Definition | Rationale / known failure mode |
| --- | --- | --- |
| `intensity_mean` | Mean pixel intensity in `[0, 255]` | Global brightness; shifts with the background policy and heavy clipping. |
| `intensity_std` | Population standard deviation in gray levels | Global spread; not a reliable noise estimate because 8-bit clipping compresses extreme noise. |
| `contrast_rms` | `std / (mean + 1e-6)` (coefficient of variation) | Scale-normalised contrast. Chosen over Michelson because percentile-based contrast saturates at 1 on clean dark-on-light text. Unbounded above; can move non-monotonically under noise. |
| `ink_fraction` | Fraction of pixels `<=` the Otsu threshold | Glyph coverage. On light backgrounds the darker Otsu class is ink. Exactly uniform images return 0 (no distinguishable ink). |
| `laplacian_variance` | Variance of a 4-neighbour Laplacian with edge-replicated padding, in gray levels | Sharpness proxy; responds to both edges and high-frequency noise. |
| `edge_density` | Fraction of pixels marked by Canny on `[0,1]` with `sigma=1.0`, `low=0.1`, `high=0.2` | Structural edge presence; zero when blur removes all detectable edges. |
| `histogram_entropy` | Shannon entropy (bits) of a 256-bin histogram over `[0,256)`, normalised to sum to 1 | Global intensity diversity; rises with noise, falls with blur/saturation. |
| `noise_sigma_immerkaer` | Immerkær's 3x3 Laplacian estimator over interior pixels, reported as a gray-level standard deviation | Noise-plus-texture estimate. It is **not** a pure noise measure on text images; it also responds to sharp glyph edges. |

**Validity and edge cases.**

- Input must be 2-D `uint8` with at least 11 pixels per side; violations raise
  `ImageContractError`.
- Exactly uniform images take an explicit shortcut: mean is preserved, all
  structural features are 0. This avoids undefined Otsu/entropy ratios.
- Every computed vector is checked for NaN/inf; extraction raises rather than
  emitting a silently invalid row.
- Batch extraction preserves column order (`sample_id` then `FEATURE_NAMES`) and
  rejects duplicate or empty IDs.
- No feature uses SSIM, PSNR, severity, corruption type, font, or text length.

### 4.2 Leakage safeguards

- Targets are computed after corruption from the exact reference/variant pair and
  stored separately from features.
- Grouped splits are assigned before generation; no source crosses splits.
- Diagnostics fit every coefficient on `train` only. Standardisation statistics
  for the multi-feature model are also estimated on `train` only.
- One-feature models are selected by validation R²; the test split is reported
  once and never used for selection.
- The diagnostic is a baseline study, not a model claim; no hyperparameter
  search is performed on test.

### 4.3 Diagnostic baselines

All models are ordinary least squares with an intercept, implemented with
`numpy.linalg.lstsq` to avoid adding a dependency for diagnostics.

| Model | Inputs | Purpose |
| --- | --- | --- |
| `naive_train_mean` | none | Floor: predicts the training mean. |
| `metadata_severity_family` | severity + family one-hot | Oracle metadata baseline; shows how much the generation design explains. Not a proposed model input. |
| `metadata_font` | font one-hot | Tests font confounding. |
| `metadata_text_length` | text length | Tests length confounding. |
| `single_<feature>` | one image feature | Tests whether the target is trivially predictable. |
| `all_features_ols` | all eight features (standardised on train) | Feature-only ceiling for later models. |

Correlations are reported overall, within each corruption family, and within
each `(family, severity)` group with at least `min_group_rows = 20` training
rows. Clean rows are excluded from family-conditional correlations because their
target is constant.

### 4.4 Predeclared triviality thresholds

These thresholds were committed in the dataset configs before the M2 runs:

- **potentially trivial:** the best single feature reaches R² >= **0.90** on
  *both* validation and test.
- **strong confound:** either split reaches R² >= **0.70**.
- otherwise: **no single-feature triviality**.

Rationale: R² 0.90 means one statistic almost completely determines the target;
0.70 indicates that a single-feature baseline explains most held-out variance.
These are diagnostic labels, not gates on the milestone. Separately, the
all-feature baseline is classified as `feature_only_near_solved` when both splits
reach 0.90, `feature_only_high` at 0.70, otherwise `feature_only_limited`.

### 4.5 M2 results and decision

Recorded in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), experiment `EXP-0002`.

On the 60-source reference dataset:

- Best single feature (selected on validation): `histogram_entropy`,
  validation R² **0.680**, test R² **0.644** -> **no single-feature triviality**.
- All-feature OLS: validation R² **0.968**, test R² **0.962** ->
  **feature_only_near_solved**.
- Metadata severity + family: validation/test R² **0.924 / 0.911**; font
  **-0.001 / 0.002**; text length **0.000 / 0.000**. Font and text length are
  not confounds in this split; corruption severity is, as expected.
- Aggregate correlations hide family-conditional reversals, e.g.
  `laplacian_variance` has overall r = 0.066 but within-blur r = 0.883 and
  within-noise r = -0.861. Within-family rows are the honest summary.
- The 9-source pilot gives entropy R² 0.822/0.801 (`strong_confound`), but its
  three training sources make that estimate too noisy to prefer over the larger
  run.

**Decision carried into M3:** the regression target is highly predictable from
simple image statistics. A CNN is not justified by target predictability alone.
Any later model must be compared against the eight-feature linear baseline and
must report gains that are not explained by the same statistics, under grouped
splits. This does not invalidate the dataset: it makes the later modelling
question sharper and prevents a triviality claim from being discovered only
after expensive training.

## 5. Conventional ML baselines (Milestone 3)

The Milestone 3 question is narrow: **given only the degraded image's eight
features, how well can a conventional model predict SSIM on sources and fonts it
did not train on?** The dataset, target and split are frozen from M2. No
hyperparameter, threshold or feature definition was changed after seeing test
results. This section records the protocol and the actual result; the full run
is `EXP-0003`.

### 5.1 Frozen evaluation protocol

| Item | Frozen value |
| --- | --- |
| Dataset | `data/processed/captcha_quality_v0.1.0/` (60 sources, 540 samples) |
| Generation hash | `820cce7a77b5e95761f033b75e8cb2b304b6ad6293f67b5bc27bcf0823a6cef9` |
| Split | source-grouped, font-stratified; assignment sha256 `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232` |
| Train / val / test | 36 / 12 / 12 sources; 324 / 108 / 108 samples |
| Primary inputs | the eight image features in §4.1, in fixed order |
| Target | `ssim_target` (SSIM in `[0, 1]`); PSNR is not modelled |
| Fitting data | train split only |
| Selection data | validation split only |
| Test use | once, after every candidate and family winner was fixed |
| Benchmark config | `configs/benchmark_v0_1_0.toml`, hash `e93d2660329a4f8b7e6b99739651250e2104ca9a26c76574db6db1e2e6bc8fa5` |
| Seed | `20261022` (model seeds and bootstrap seeds derived from it) |
| Environment | CPython 3.13.16, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6 |

Each run recomputes the split-assignment hash from `sources.csv`, requires it to
match `dataset_summary.json`, and refuses to run if a source group crosses
splits, a source has multiple fonts, the feature table is not exactly
`sample_id` plus the eight features, a feature is non-finite, or the dataset
feature-set version differs from the extractor. All 540 rows are merged 1:1 on
`sample_id`.

The test set is never used to choose a feature, preprocessing statistic,
hyperparameter, family winner or model. The candidate table written to disk
contains training and validation metrics only; test metrics exist only for the
validation-selected model and the validation-selected winner of each model
family. This is a deliberate structural guard, not a convention.

### 5.2 Candidate models and search grid

Only scikit-learn is used; no deep-learning framework is installed for this
milestone. The naive baseline is the training mean. The learned grid has 23
candidates:

| Family | Specification | Candidates |
| --- | --- | ---: |
| Naive | `DummyRegressor(strategy="mean")`, baseline only | 1 |
| Linear | OLS on eight standardised features | 1 |
| Ridge | L2 penalty, `alpha` in `{0.01, 0.1, 1, 10, 100, 1000}`, features standardised | 6 |
| Random forest | 300 trees; depth `{4, 6, 8, unlimited}` x leaf size `{1, 3, 5}`; raw features | 12 |
| Histogram gradient boosting | 200 iterations; learning rate `{0.05, 0.1}` x max leaves `{15, 31}`; min leaf 10; L2 0 | 4 |

Scaling is a `StandardScaler` inside a scikit-learn `Pipeline`, so it is fitted
on the training rows passed to `fit`. Tree models intentionally receive raw
features. Random forests use `n_jobs=1` and a derived 32-bit seed for
determinism. The grid is small on purpose: 540 rows and 60 source groups cannot
support a large search, and every extra candidate increases the chance of
validation overfitting.

**Selection rule (predeclared).** The lowest validation RMSE wins. Ties are
broken by fewer inputs, then by model name. The best configuration in each
family is selected the same way; this lets the report compare linear and
nonlinear families on test without post-hoc tuning. The overall selected model
is the best validation candidate across families. Test metrics for all family
winners are reported for transparency; they did not influence selection.

### 5.3 Metrics and grouped uncertainty

For every reported model and slice:

- **MAE** = mean absolute error, in SSIM units.
- **RMSE** = population root mean squared error, in SSIM units.
- **R²** = `1 - SS_res / SS_tot`. It is reported as undefined (`nan`, with
  `r2_defined = false`) when `SS_tot = 0`, notably the constant clean-only
  subset. It is never fabricated as zero or one.
- `n_rows` and `n_sources` are always reported. Variants are not independent
  observations.

Uncertainty is a **source-group bootstrap**: 1,000 replicates resample the
source groups (not individual variants) with replacement, concatenate all rows
of each sampled group, and recompute MAE/RMSE/R² with the fitted predictions held
fixed. The 2.5th and 97.5th percentiles are descriptive intervals. With 12 test
groups and 20 held-out-font groups they are necessarily wide and unstable; no
significance test or confidence claim is made. The same procedure is applied to
the held-out-font results.

### 5.4 Explicitly diagnostic metadata experiments

These models use generated labels and are **not** the intended image-only
model. They exist to measure how much of the target the synthetic generation
design explains and to expose confounding.

| Diagnostic | Inputs | Purpose |
| --- | --- | --- |
| `metadata_severity_family` | severity + corruption-family indicators (3 columns) | oracle-label baseline |
| `metadata_font_text_length` | font indicators + text length (3 columns) | font/length control |
| `image_plus_metadata` | eight standardised features + severity + family indicators (11 columns) | combined diagnostic reference, not a deployable model |

Category levels and scaling statistics come from the training split. An unseen
test category is an error rather than a silent all-zero encoding. These designs
are never used to select the primary model, and the combined model is not
guaranteed to upper-bound a nonlinear image-only model; it is one diagnostic
reference.

### 5.5 Stratified analysis

After selection, the frozen test split is described by:

- corruption family (`gaussian_blur`, `gaussian_noise`, clean);
- severity level 0–4;
- ink-coverage quartiles, with edges fixed from the **training** split's
  `ink_fraction` quantiles (`0, 0.25, 0.5, 0.75, 1.0`); test values outside the
  training range fall in the open outer bin. Bins are sample-level: because
  severity changes ink coverage, one source can contribute to multiple bins, so
  coverage-row group counts are not additive;
- font, for descriptive comparison only.

These rows are not used to re-select or tune a model. Small subgroups are shown
with their group and sample counts and labelled descriptive. Within-family
feature/target correlations are computed on training rows. Aggregate
correlations are not substituted for the family-conditional ones, because
Laplacian variance and ink coverage reverse sign between blur and noise.

### 5.6 Held-out-font stress test

The held-out font was fixed before fitting: **Times New Roman**, the only serif
face in the three-font configuration. The choice uses font metadata only and is
recorded in `configs/benchmark_v0_1_0.toml`; no performance result informed it.

- Training fonts: Arial and Courier New.
- Training data: their primary `train` sources — 24 sources, 216 samples.
- Selection data: their primary `val` sources — 8 sources, 72 samples.
- Held-out evaluation: all 20 Times New Roman sources — 180 samples, all
  variants grouped.
- No Times New Roman row enters fitting, scaling, category levels, early
  stopping or hyperparameter selection. The same candidate grid is re-selected
  using only the two training fonts; the held-out result is then evaluated once.
- The naive baseline is the mean SSIM of the two-font training rows.

This is a **preliminary font-shift stress test**, not proof of broad
generalisation: only three fonts exist in total, only two remain for training,
and the held-out font still shares the renderer, canvas, text distribution and
corruption design. It is stronger than the random grouped split but far weaker
than a cross-generator or real-image evaluation.

### 5.7 Results and target-meaningfulness decision

Full configuration, predictions, residual and stratified tables are in the
Git-ignored `reports/benchmark_v0_1_0/`; the model cards and joblib artifacts are
in `models/benchmark_v0_1_0/`. Headline test results (108 samples, 12 source
groups):

| Model | MAE | RMSE | R² | Note |
| --- | ---: | ---: | ---: | --- |
| Naive train mean | 0.1664 | 0.1942 | -0.0000 | baseline |
| OLS (8 features) | 0.0302 | 0.0380 | 0.9618 | validation-selected linear family |
| Ridge (`alpha=0.01`) | 0.0302 | 0.0379 | 0.9619 | validation-selected ridge family |
| Random forest (depth 8, leaf 1) | 0.0157 | 0.0226 | 0.9865 | validation-selected overall |
| Histogram gradient boosting (lr 0.1, 15 leaves) | 0.0174 | 0.0256 | 0.9826 | validation-selected boosting family |

The selected model reduced test RMSE by 88.4% relative to the naive mean, above
the predeclared 5% materiality rule. The random forest's test RMSE was 0.0153
(40.4%) lower than the best linear family winner; this is descriptive evidence
that the feature/target relationship is nonlinear on this split, not a general
claim about forests. Both tree families overfit the training split (HGB train
RMSE 0.0021 versus validation 0.0216), which the train/validation columns expose
rather than hide.

Metadata and gap results on test:

| Comparison | RMSE | R² | Gap versus selected image-only model |
| --- | ---: | ---: | --- |
| Image-only selected RF | 0.0226 | 0.9865 | — |
| Severity + family only | 0.0579 | 0.9112 | +0.0353 RMSE, -0.0752 R² |
| Font + text length only | 0.1940 | 0.0023 | effectively no signal |
| Image + metadata (OLS) | 0.0242 | 0.9844 | +0.0016 RMSE, -0.0020 R² |

The metadata-only result confirms that the synthetic corruption design explains
most of the target. The image-only model still beats it, and the combined
diagnostic does not beat the image-only random forest. Font and text length are
not confounds in this split. These facts make the image-only task genuine but
easier than real-world quality assessment; they do not make it a readability
measure.

Family and coverage slices of the selected model's test predictions:

- Blur: RMSE 0.0255, R² 0.9824 (48 samples, 12 groups).
- Noise: RMSE 0.0223, R² 0.9721 (48 samples, 12 groups).
- Clean: RMSE 0.0012; R² undefined because the target is constant (12 samples).
- Coverage quartiles Q1–Q4: RMSE 0.0189, 0.0148, 0.0252, 0.0288. The highest
  coverage quartile is hardest, descriptively.
- Severity 1–4: RMSE 0.0210, 0.0312, 0.0205, 0.0214; the R² values at severity
  1–2 (0.60, 0.70) are low because within-level target variance is small, not
  because absolute error is large. Counts are shown in the report.

Grouped bootstrap 95% intervals for the selected test model: RMSE
[0.0157, 0.0309], MAE [0.0116, 0.0209], R² [0.9757, 0.9934]. These are
descriptive and unstable at 12 groups.

Held-out font (Times New Roman, 180 samples, 20 groups):

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Naive (two-font train mean) | 0.1628 | 0.1910 | -0.0006 |
| Validation-selected HGB (lr 0.05, 31 leaves) | 0.0193 | 0.0271 | 0.9799 |
| Random-forest family winner | 0.0184 | 0.0248 | 0.9832 |
| OLS family winner | 0.0326 | 0.0398 | 0.9566 |

The selected model survives this stress test (bootstrap R² interval
[0.9712, 0.9876]), and the random-forest family winner is descriptively slightly
better on the held-out font even though HGB won on the two-font validation
split. That comparison is reported as-is; the model was not re-selected on the
held-out result.

**Decision.** For this synthetic SSIM target:

1. A conventional model improves materially over the naive mean (88.4% RMSE
   reduction).
2. A nonlinear forest beats OLS/Ridge on held-out sources by a descriptive
   margin (40.4% lower RMSE), so model class matters here.
3. The result is broadly stable across blur and noise, but error and R² vary by
   severity and coverage; the aggregate hides that.
4. Metadata alone explains 91.1% of test variance, so the synthetic design is a
   strong confound; the image-only model is still better.
5. The selected model survives the three-font held-out-font stress test, with
   the caveats above.
6. **A CNN is not justified for this target.** The selected random forest
   already reconstructs 98.6% of held-out SSIM variance, corruption metadata
   alone explains 91.1%, and nonlinearity is already captured by a conventional
   forest. A CNN would mainly add capacity, training cost and
   opacity. It would only be justified on a harder, predeclared shift if it beats
   this frozen baseline.

A high R² does not make the target scientifically interesting, and nothing here
measures human readability or CAPTCHA solvability.

### 5.8 CNN comparison criteria (Milestone 4, not implemented)

If a CNN is attempted later, it must use the same frozen split, target and
held-out-font partition; report MAE/RMSE/R² and grouped uncertainty; and be
compared against the saved feature-only random forest and the metadata
diagnostics on the same rows. A small from-scratch CNN and a documented
transfer-learning variant are sufficient. Parameter count, training time and
hardware must be reported. The CNN is retained only if it improves held-out
performance, especially under font or corruption shift, without test-set
tuning. No CNN code or dashboard exists in Milestone 3. M4B evaluated a conventional
adaptation and still found no capacity-bound limitation; see §5.10.

### 5.9 M4A occlusion/interference pilot (not a robustness benchmark)

Milestone 4A asks a narrower question than a benchmark: before adding a new
corruption family to the grouped protocol, does it produce a reproducible,
non-degenerate SSIM target with an interpretable severity response and a feature
pattern that is not merely a reparameterisation of blur or noise? The pilot is
implemented in `interference.py` and `robustness_pilot.py` and is configured by
`configs/robustness_pilot_occlusion_v0_1_0.toml`.

**Isolation.** The pilot reads the frozen `v0.1.0` dataset, verifies its hashes,
and re-renders all 60 clean sources from `configs/dataset_v0_1_0.toml` to prove
the references are the existing renderer's images. It writes only to
`data/processed/occlusion_interference_pilot_v0.1.0/` and
`reports/robustness_pilot/` (both Git-ignored). It does not modify the frozen
dataset, split, M3 reports or model artifacts, and it does not train or tune a
model.

**Variants.** Both use a 1 px black stroke (intensity 0) on the existing
192x64 canvas:

- `thin_lines`: straight strokes through a seeded pivot inside the clean image's
  ink bounding box, extended to the canvas boundary in both directions. The
  line angle is drawn from `[-60, 60]` degrees.
- `arc_strokes`: elliptical arcs through a seeded ink pivot, with sweep drawn
  from `[120, 240]` degrees, horizontal radius `0.20-0.55` of image width, and
  vertical/horizontal aspect `0.50-1.50`.

Severity levels are 1, 2, 4 and 7 strokes. For each `(source, variant)` the
pilot plans a shared seven-stroke pool from an explicit seed derived from the
pilot seed `20261023`, the source ID and the variant. Severity `k` draws the
first `k` strokes and every severity is rendered from the unmodified clean
reference. This makes severity a nested density dose rather than an independent
redraw. The full geometry is recorded per sample, and verification replays every
corrupted image from that geometry without consulting an RNG.

**Target and gate.** SSIM uses the same conventions as §3.1 and is structural
similarity to the clean reference, not human readability or CAPTCHA
solvability. PSNR is a secondary sanity diagnostic. The predeclared gate before
generation was:

| Criterion | Threshold |
| --- | --- |
| Reproducibility | Two builds identical; all 60 clean re-renders exact; dataset verification passes |
| Non-degeneracy | Corrupted SSIM SD ≥ 0.02; per-variant level-mean range ≥ 0.10; per-level SD ≥ 0.005 |
| Interpretable trend | Median within-source Spearman(severity, SSIM) ≤ -0.80; increasing adjacent-step fraction ≤ 0.15 |
| Feature distinctness | At highest severity, mean absolute Cohen's d ≥ 0.50 versus both frozen blur and noise, with at least two features at \|d\| ≥ 0.80 |
| Duplicate-free | Duplicate corrupted-image fraction ≤ 0.0 |

**Counts.** 60 frozen sources, all three fonts, split 36/12/12
train/val/test. Samples: 540 = 60 clean + 480 corrupted (2 variants × 4
severities × 60 sources). Pilot generation hash
`dc1a9b1557b3e8e49e86dc4e59beabf1d35d53c9f3a59300399a9f5a4e49e6f0`;
manifest sha256 `d08c1824ba81e3bc40a585b3047eec6c5f9aa2da0b0b9eb126057ab27fdcc92b`.

**Results.** Mean SSIM by severity (SD in parentheses):

| Variant | 1 stroke | 2 strokes | 4 strokes | 7 strokes |
| --- | ---: | ---: | ---: | ---: |
| `thin_lines` | 0.943 (0.020) | 0.887 (0.032) | 0.785 (0.041) | 0.662 (0.044) |
| `arc_strokes` | 0.947 (0.022) | 0.902 (0.029) | 0.818 (0.037) | 0.715 (0.042) |

Across all 480 corrupted samples, SSIM mean was 0.832, SD 0.105, range
0.545-0.979. Every one of the 120 source-variant ladders had Spearman -1.0;
there were 0 increasing and 0 tied adjacent steps out of 360. PSNR fell from
22.20 to 13.15 dB for lines and 22.47 to 13.96 dB for arcs, consistently with
SSIM. At the highest severity, mean absolute Cohen's d was 8.29 versus blur and
4.02 versus noise for lines, and 7.06 versus blur and 3.58 versus noise for
arcs; 7-8 of the 8 features exceeded \|d\| = 0.8. All 13 verification checks
passed, including exact clean-reference re-rendering, 480 geometry replays, and
zero duplicate images. Two independent builds produced identical manifests,
features and image hashes. The predeclared gate returned **PASS**.

**Decision.** Proceed to a separate grouped robustness benchmark for this
family. The pilot supports non-degeneracy, determinism, an interpretable
density response and a feature pattern distinct from blur/noise. It does **not**
show that the family is harder for humans, does not measure solvability, and
does not establish cross-generator generalisation. The full benchmark must use a
separate grouped protocol, keep the family out of the frozen M3 test set, and
predeclare its model comparison; the pilot itself trains no model. Marginal SSIM
drop per added stroke declines with density (lines -0.057 to -0.041; arcs -0.046
to -0.034), so a full ladder should retain the saturation caveat.

### 5.10 M4B grouped occlusion/interference robustness benchmark

Milestone 4B asks two questions on the frozen M4A dataset: does the unmodified
M3 model transfer zero-shot to the held-out interference sources, and does a
conventional model trained on M3 plus M4A training-source rows improve held-out
performance? It is implemented in `robustness_benchmark.py` and configured by
`configs/robustness_benchmark_v0_1_0.toml`.

**Protocol freeze.** The M3 dataset, split, model bundle/card, reports and M4A
pilot are read-only inputs. Their pinned hashes are recorded in the config and
rechecked by `verify`. The benchmark writes only to
`reports/robustness_benchmark_v0_1_0/` and
`models/robustness_benchmark_v0_1_0/` (both Git-ignored). The feature schema is
the existing eight image-only features; the target is the unchanged
`ssim_target`. SSIM, PSNR, severity, variant family, font, text length, source
ID, split and generation metadata are excluded from primary model inputs.

**Predeclared interpretation rules.** The protocol was fixed before test
evaluation. Zero-shot transfer is assessed from the clean-versus-corrupted
residual pattern and R², not from a universal pass threshold: near-perfect clean
predictions plus negative corrupted R² means absolute calibration did not
transfer. Adaptation is reported as beneficial when the validation-selected
model has lower held-out RMSE than frozen M3 and the paired source-grouped
difference is consistent with improvement. Limited generalisation is flagged
when variant or high-density slices remain worse, even if the aggregate
improves. A CNN is justified only if a meaningful limitation persists after the
conventional comparison; a lower aggregate loss alone is not enough. No
universal numeric pass/fail threshold is introduced after seeing test results.

**Source assignment and merge.** The M4A source assignment is the existing
36/12/12 train/validation/test split. Every variant and severity of a source
stays in its split. Expanded training uses all M3 training rows plus M4A
corrupted rows from training sources only. Expanded validation uses M3
validation rows plus M4A corrupted rows from validation sources only. M4A clean
rows are excluded from fitting and selection because all 60 are exact
duplicates of the M3 clean references; the 12 M4A clean test rows remain only in
a separately reported clean-reference evaluation slice. The merged frames are:

| Frame | Rows | Sources | Composition |
| --- | ---: | ---: | --- |
| Expanded train | 612 | 36 | 324 M3 (36 clean + 288 blur/noise) + 288 M4A corrupted |
| Expanded validation | 204 | 12 | 108 M3 (12 clean + 96 blur/noise) + 96 M4A corrupted |
| M4A test | 108 | 12 | 12 clean + 96 corrupted (48 per variant) |
| M3 test reference | 108 | 12 | 12 clean + 96 blur/noise |

The merge audit verifies source isolation in both datasets, identical source
IDs/splits/fonts/text lengths, exact clean-duplicate provenance, no duplicate
sample IDs or image hashes in the expanded fit frames, and exclusion of M4A
clean rows from fitting. All six checks pass.

**Experiment A — zero-shot frozen M3.** The frozen M3 model is loaded through
the existing bundle verifier and evaluated once on the M4A test sources. Its
comparison baseline is the M3 training-mean target, which keeps the naive
baseline fixed to the frozen model's own training distribution. Experiment B
uses a separate expanded training-mean baseline.

| Model | Slice | n | MAE | RMSE | R² | Mean residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen M3 | overall | 108 | 0.1295 | 0.1612 | -1.0455 | -0.1294 |
| Frozen M3 | clean | 12 | 0.0004 | 0.0012 | n/a | 0.0004 |
| Frozen M3 | corrupted | 96 | 0.1456 | 0.1710 | -1.6031 | -0.1456 |
| Frozen M3 | `thin_lines` | 48 | 0.1590 | 0.1876 | -1.6062 | -0.1590 |
| Frozen M3 | `arc_strokes` | 48 | 0.1322 | 0.1527 | -1.7624 | -0.1322 |
| M3 training mean | overall | 108 | 0.1688 | 0.1948 | -1.9853 | 0.1589 |
| M3 training mean | corrupted | 96 | 0.1516 | 0.1759 | -1.7545 | 0.1404 |

The frozen model is near-perfect on clean references but under-predicts with
increasing density: MAE 0.0496, 0.0990, 0.1690 and 0.2649 at severities 1-4.
Its overall MAE advantage over the M3 training mean is therefore driven by the
clean slice. The paired source-grouped MAE difference on corrupted rows is
-0.0060 with 95% interval [-0.0307, 0.0129], which includes zero; on
`thin_lines` the frozen model is descriptively worse (+0.0141, interval
[-0.0124, 0.0388]). Negative R² means it is worse than a constant test-mean
predictor in squared-error terms. Zero-shot absolute calibration does not
transfer.

**Experiment B — controlled conventional adaptation.** A fixed 12-candidate
grid was declared before test evaluation: OLS, Ridge with three alphas, four
random forests and four histogram gradient boosting configurations. It is
fitted on expanded train only, selected on expanded validation RMSE only, and
the selected model is evaluated once on M4A test. No scaler, calibration,
feature selection or hyperparameter decision uses test rows.

Validation selected `random_forest_depth_unlimited_leaf_1` (validation RMSE
0.0201, R² 0.9864). Test results:

| Model | Slice | n | MAE | RMSE | R² | Mean residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Expanded model | overall | 108 | 0.0131 | 0.0178 | 0.9752 | 0.0047 |
| Expanded model | clean | 12 | 0.0024 | 0.0074 | n/a | 0.0024 |
| Expanded model | corrupted | 96 | 0.0144 | 0.0187 | 0.9690 | 0.0050 |
| Expanded model | `thin_lines` | 48 | 0.0162 | 0.0213 | 0.9664 | 0.0053 |
| Expanded model | `arc_strokes` | 48 | 0.0126 | 0.0156 | 0.9712 | 0.0047 |
| Expanded training mean | overall | 108 | 0.1261 | 0.1466 | -0.6912 | 0.0937 |
| Expanded training mean | corrupted | 96 | 0.1117 | 0.1300 | -0.5044 | 0.0753 |

The adapted model improves over frozen M3 by -0.1164 MAE (95% interval
[-0.1277, -0.1003]) and -0.1435 RMSE ([-0.1558, -0.1256]); the paired
difference excludes zero at every severity and both variants. The clean slice
is slightly worse than frozen M3 (+0.0021 MAE, interval [-0.0000, 0.0056]) but
both are close to perfect and the interval includes zero. Severity RMSE rises
from 0.0143 to 0.0245 and mean signed residual from +0.0027 to +0.0100, so the
high-density end remains hardest. Test R² is unstable at low severity (0.445 at
severity 1), where target variance is small despite low MAE.

On the original M3 blur/noise test rows, the expanded model remains close to the
frozen reference: overall MAE/RMSE/R² 0.0158/0.0239/0.9849 versus
0.0157/0.0226/0.9865 for frozen M3. It does not sacrifice the original families
to gain interference performance.

**Target distribution.** Expanded training targets have mean 0.7587 (SD 0.1717,
range 0.3174-1.0); M4A test targets have mean 0.8524 (SD 0.1133, range
0.5532-1.0). M4A corrupted test mean is 0.8340 and the expanded corrupted-only
training mean is 0.8319; the lower full-expanded mean is caused by severe M3
blur/noise rows. The shift is disclosed and is not used for tuning. The naive
expanded-mean baseline (0.1261/0.1466/-0.6912) is reported so that the expanded
target mean is not mistaken for learned corruption response.

**Decision.** On this benchmark, conventional modelling is sufficient. The
frozen model's failure is a target/covariate shift that controlled retraining
addresses; there is no evidence that model capacity is the binding limitation.
A CNN is not justified by these results. With 12 held-out sources, all intervals
are preliminary and no significance or universal pass/fail threshold is claimed.
The result does not measure human readability or CAPTCHA solvability, does not
generalise beyond the synthetic renderer, and does not cover a cross-generator
or human-labelled shift.

### 5.11 M5A geometric-distortion feasibility pilot (not a robustness benchmark)

Milestone 5A asks the same pre-benchmark question as M4A for one narrowly scoped
geometric family: before adding it to a grouped protocol, is a bounded smooth
elastic warp reproducible, replayable, non-degenerate, interpretable and not
merely a reparameterisation of the existing corruption families? The
implementation is `geometric.py` and `geometric_pilot.py`, configured by
`configs/geometric_distortion_pilot_v0_1_0.toml`. It trains no model.

**Isolation and frozen inputs.** The pilot verifies and re-renders all 60 clean
M3 sources from `configs/dataset_v0_1_0.toml`; the M3 dataset, split, models and
reports, the M4A pilot and the M4B benchmark/model are read-only. It writes only
to `data/processed/geometric_distortion_pilot_v0.1.0/` and
`reports/geometric_distortion_pilot/` (both Git-ignored). One seeded unit-maximum
displacement field is generated per source and shared by every severity and
control, so severity is a nested amplitude dose rather than an independent
redraw. The field is zero-mean before normalisation, removing global translation.
Every transformed row stores the full field specification and an exact SHA-256
of the float64 field; verification regenerates the field and pixels without
consulting the target.

**Transformation.** The primary family is a smooth elastic displacement:

- 192x64 8-bit grayscale input, bilinear resampling;
- 5x13 coarse control grid including boundary points, with separable Gaussian
  smoothing at sigma 2.0 px;
- maximum displacement ladder `[0.5, 1.0, 2.0, 3.0]` px, plus a 0.25 px minimal
  control; the largest value is under 5% of the 64 px canvas height;
- border mode `edge` (out-of-canvas source coordinates clamp to the nearest
  valid sample); the alternative `reflect` mode is implemented and tested but
  not used in the reported run;
- identity control bypasses resampling; zero-field control passes integer
  coordinates through the bilinear sampler; minimal control applies 0.25 px of
  the same field.

All controls and severities are generated from the unchanged clean reference.
The target is the unchanged `ssim_target`; PSNR is secondary. Features are the
existing eight image-only features in fixed order.

**Counts and hashes.** 60 sources, all three fonts, frozen 36/12/12 split.
Samples: 480 = 60 clean + 180 controls (60 identity, 60 zero-field, 60 minimal)
+ 240 elastic (4 severities x 60). Pilot generation hash
`e5901ce89fda1854d606aa41b6258222aa77e9bde061c9082971d7660a924c2a`;
manifest sha256
`c63be1088db591181f23fa306f955b3ba16fe80ffb2000c4a5e3db7656fa3806`;
features sha256
`cb82ae60d8749f21d9d5f6dd324cd075e907544aa54bcdb62bbcab988472bb65`.
Identity and zero-field controls reproduced all clean pixels exactly. The
minimal control had mean SSIM 0.99803 (minimum 0.99662) and changed 17.1% of
pixels.

**Results.** Mean SSIM by severity (SD in parentheses):

| Severity | Amplitude (px) | Mean SSIM (SD) | Range | Mean PSNR (dB) | Changed-pixel fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5 | 0.9923 (0.0025) | 0.9868-0.9975 | 31.41 | 0.212 |
| 2 | 1.0 | 0.9713 (0.0091) | 0.9514-0.9900 | 25.54 | 0.255 |
| 3 | 2.0 | 0.9070 (0.0260) | 0.8526-0.9625 | 19.95 | 0.290 |
| 4 | 3.0 | 0.8372 (0.0396) | 0.7571-0.9242 | 17.13 | 0.307 |

Across all 240 elastic rows, SSIM mean was 0.9270, SD 0.0653, range
0.7571-0.9975. All 60 source ladders had Spearman -1.0; all 180 adjacent steps
decreased, with 0 tied and 0 increasing. Grouped bootstrap 95% intervals for
mean SSIM were 0.9917-0.9929, 0.9689-0.9735, 0.9010-0.9134 and 0.8273-0.8470.
No elastic row was near-identity at SSIM >= 0.9999, no row was degenerate, and
no elastic image duplicated another or a clean reference. At severity 4, mean
outside-canvas/border-affected fraction was 0.0261, bilinear-clipped-sample
fraction 0.0046, and collapsed 8x8 block fraction 0.0426. These are reported as
confounds, not subtracted from SSIM.

**Cross-family comparison.** Frozen blur/noise and M4A interference features were
contrasted at the same severity. At severity 4, mean absolute Cohen's d was 5.21
against `gaussian_blur`, 6.49 against `gaussian_noise`, 5.67 against
`thin_lines` and 4.28 against `arc_strokes`; 4-8 of the eight features exceeded
|d| = 0.8 in each contrast. Pooled correlations were not treated as evidence of
a stable feature/target relationship: for example `laplacian_variance` had
pooled Spearman 0.616 but mean within-severity Spearman -0.075, and
`edge_density` had pooled -0.095 but within-severity -0.304. Full distributions
and correlations are in the ignored report tables.

**Gate.** The predeclared gate returned **CONDITIONAL PASS**. Reproducibility,
identity/zero-field exactness, overall target spread, trend, interpolation/
boundary bounds, feature distinctness and duplicate-freedom passed. The one
failed margin was per-level SD at severity 1: 0.0025 against the predeclared
0.005. That rung is close to the interpolation floor, so its between-source
spread is small even though the overall target is non-degenerate and every
source trajectory is monotonic. The failed criterion is retained in the report;
the verdict is not an unqualified PASS.

**Decision.** The family is correctly implemented, deterministic, exactly
replayable, non-destructive and monotonic on all 60 sources, and its features
are strongly separated from the existing families at high severity. It does not
support a benchmark claim at the 0.5 px rung. A later benchmark should drop that
rung or use a larger minimum amplitude, and must carry the field specification,
controls, source grouping and interpolation/border caveat forward. No model was
trained. Nothing here measures human readability or CAPTCHA solvability, and no
CNN is justified by this pilot.

### 5.12 M5B grouped geometric-robustness benchmark

Milestone 5B asks whether the frozen M3 model transfers zero-shot to the held-out
geometric sources, and whether a controlled conventional model trained on
representative geometric data improves held-out geometric performance while
retaining the original blur/noise task. The implementation is
`geometric_benchmark.py`, configured by
`configs/geometric_robustness_benchmark_v0_1_0.toml`. It writes only to
`reports/geometric_robustness_benchmark_v0_1_0/` and
`models/geometric_robustness_benchmark_v0_1_0/` (both Git-ignored).

**Zero-shot transfer and controlled adaptation are different questions.**
Zero-shot transfer asks whether the M3 model, frozen and never refitted,
predicts geometric SSIM correctly for sources it has never seen in any form.
It is a measurement of cross-corruption generalisation and nothing may be fitted
on the geometric test rows. Controlled adaptation asks a different question:
whether a conventional model given representative geometric training rows
learns a better mapping on held-out geometric sources, and what that costs on
the original task. The two are reported side by side on identical test rows,
but a poor zero-shot result is not evidence about the adapted model, and a good
adapted result is not evidence that the original model generalised.

**Frozen inputs.** The M3 dataset, split, primary model and model card, the M4A
pilot, the M4B benchmark and its model, and the M5A pilot dataset are read-only
and hash-verified before and after the run. The frozen M3 primary model is
loaded through its existing card/artifact verification path, including the
feature-order, target, dependency and smoke-prediction checks.

**Split.** The frozen M3 source-grouped assignment is reused verbatim
(36 train / 12 validation / 12 test sources, assignment sha256
`70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`). M5A
inherits the same source assignment; every variant of a source stays in that
source's split, and no sample is assigned to a split independently. No
test-source image, feature, target or metadata enters training, preprocessing
or model selection.

**Severity protocol.** The primary geometric score uses the 1.0, 2.0 and 3.0 px
rungs (severities 2, 3, 4). The 0.5 px rung is excluded from primary training,
validation selection and scoring, and is reported only as a separately labelled
diagnostic, because the M5A gate found its between-source SSIM SD (0.0025) below
the predeclared 0.005 minimum-variation margin. It is neither silently dropped
nor merged into the primary score. Clean and control rows (the 60 clean, 60
identity, 60 zero-field and 60 minimal rows) are excluded from every fit frame
and from scoring: the clean rows duplicate the M3 clean references, and the
identity/zero-field controls are exact duplicates of them.

**Features and target.** Model inputs are exactly the eight existing image-only
features in their existing order. Severity, corruption family, font, text
length, source and split identifiers, SSIM/PSNR diagnostics and all provenance
metadata are excluded from primary inputs, and the fit frames are checked
programmatically for their absence. The target is the unchanged `ssim_target`,
computed by the existing SSIM implementation against the exact clean reference.

**Experiment A - frozen zero-shot.** The unmodified M3 model is evaluated on
M5A test sources only. The naive baseline is the M3 training target mean. No
preprocessing, calibration, threshold or parameter is fitted on geometric test
results.

**Experiment B - controlled adaptation.** The expanded training frame is the M3
blur/noise training rows plus the M5A 1.0-3.0 px elastic rows from the same
training sources (432 rows / 36 sources); the expanded validation frame is the
corresponding validation sources only (144 rows / 12 sources). A small
predeclared conventional grid (OLS, ridge at three alphas, random forest at four
depth/leaf settings, histogram gradient boosting at four learning-rate/leaf
settings; 12 candidates) is fitted on the expanded training frame and selected
on expanded validation RMSE only. Test sources are never passed to the fit or
the selector. The comparison set is the frozen M3 model, the validation-selected
adapted model, and a naive baseline fitted to the expanded training target
distribution.

**Experiment C - original-task retention.** The selected adapted model is
evaluated on the untouched M3 test split and compared with the frozen M3 model
on exactly the same rows and metrics. These results are reported, not used to
choose or retune the adapted model.

**Data integrity and leakage checks.** Before any model is evaluated the
benchmark verifies: each source belongs to exactly one split in both datasets;
M3 and M5A agree on every source's split, font and text length; all variants of
a source stay together; no test-source row enters training or validation; every
geometric variant derives from its recorded clean reference; clean-reference
duplicates are identified and excluded explicitly; no duplicate sample ID or
unintended duplicate feature/target row enters a fit frame; the 0.5 px rung is
absent from training, validation and the primary test score; the feature
schema and target implementation are unchanged; and the frozen M3/M4A/M4B/M5A
assets are byte-identical to their recorded pins. Evaluation does not proceed
if any check fails.

**Uncertainty.** Source-grouped bootstrap intervals resample whole source groups,
preserving all rows within a selected group, with 1000 replicates and paired
differences computed on identical rows. The test set contains only 12 source
groups, so every interval is labelled preliminary and descriptive. Overlapping
or non-overlapping intervals are not treated as significance tests, and no
universal performance threshold is introduced after observing test results.

**Results.** On the primary 1.0-3.0 px geometric test slice (36 rows / 12
sources), the frozen M3 model has MAE 0.0777, RMSE 0.0967 and R2 -1.1858 with a
mean residual of -0.0717, versus the naive M3 training-mean baseline at MAE
0.2030, RMSE 0.2132 and R2 -9.6363. The adapted model has MAE 0.0544, RMSE
0.0659 and R2 -0.0165; the expanded-training-mean baseline has MAE 0.1495 and
RMSE 0.1632. The aggregate paired RMSE difference between adapted and frozen is
-0.0307 (preliminary interval -0.0472 to -0.0155), a 31.8% reduction, but the
adapted R2 remains near zero, so the aggregate error falls without the model
explaining between-source variation at fixed severity. The gain is not uniform
across severity: adapted MAE is worse at 1.0 px (0.0520 versus 0.0259) and
better at 2.0 px (0.0363 versus 0.0662) and 3.0 px (0.0749 versus 0.1409). At
the diagnostic 0.5 px rung the two models are nearly indistinguishable (MAE
0.0170 versus 0.0171), which carries no primary claim.

**Original-task retention.** On the same 108-row / 12-source M3 test split, the
frozen M3 model has MAE 0.0157, RMSE 0.0226 and R2 0.9865; the adapted model
has MAE 0.0179, RMSE 0.0243 and R2 0.9843. The aggregate paired RMSE difference
is 0.0017 (preliminary interval -0.0022 to 0.0042), which includes zero, but
the clean-reference slice degrades clearly (MAE 0.0134 versus 0.0004; paired
difference 0.0130, interval 0.0074 to 0.0186). The adapted model is therefore
not a drop-in replacement.

**Decision.** Conventional feature-based ML remains the benchmark family. The
frozen model fails zero-shot geometric calibration and controlled adaptation
with representative geometric training data substantially reduces aggregate
geometric error without test leakage, while the original blur/noise task is
largely retained at the aggregate level. The clean-reference calibration
regression and the near-zero adapted R2 are unresolved: the natural next
modelling question is a predeclared retention-aware selection or calibration
protocol, not a larger model. The evidence does not justify a CNN, and a new
corruption family does not by itself justify one. These results cover one
synthetic renderer, one canvas size, three fonts and 12 held-out source groups;
they are not a claim about real-world CAPTCHA robustness, human readability or
solvability.

### 5.13 M5C retention-aware model selection and calibration

Milestone 5B left a specific trade-off open: the adapted histogram gradient
boosting model improved the aggregate geometric score but lost clean-reference
calibration and came close to losing the original blur/noise task. M5C asks
whether a conventional model-selection or calibration policy can improve
geometric prediction while retaining performance on clean references and on the
original task. The implementation is `retention_aware.py`, configured by
`configs/retention_aware_selection_v0_1_0.toml`. It writes only to
`reports/retention_aware_selection_v0_1_0/` and
`models/retention_aware_selection_v0_1_0/` (both Git-ignored).

**Zero-shot transfer, controlled adaptation and retention-aware selection are
three different questions.** Zero-shot transfer measures whether a frozen model
generalises to a corruption family it has never seen. Controlled adaptation
measures whether a model trained on representative rows of that family does
better on held-out sources of it. Retention-aware selection measures a third
thing: whether any candidate in a predeclared set can be preferred under a rule
that prices geometric improvement against retention. M5C reports all three
against the same frozen reference, because a poor zero-shot result says nothing
about the adapted candidates and a good adapted result says nothing about
whether the frozen model was the right choice.

**Frozen inputs.** M5C reads the M5B configuration, verifies its hash and
protocol hash, and reuses the M5B verified frame-building path, so both
milestones operate on identical frames, the identical frozen M3 source-grouped
split (36/12/12 sources) and the identical M5B isolation checks. The frozen M3
model, the M5A pilot, the M4A pilot, the M4B benchmark and the M3 dataset are
read-only and re-verified before and after the run.

**Predeclared candidate policies.** Six candidates were fixed before any test
prediction: the frozen M3 model, the M5B expanded HGB, and one expanded
conventional alternative, each used raw and with a single affine calibration.
The alternative is `random_forest_depth_unlimited_leaf_1`, justified from the
existing M5B validation table rather than from any M5C test result: it is the
best non-HGB candidate there (validation RMSE 0.0295848, 1.5% behind the
selected HGB at 0.0291514) and a bagged tree ensemble quantises and
extrapolates differently from boosting. M5C re-derives the M5B validation
table at run time and refuses to proceed if the alternative is no longer the
best non-HGB candidate, so the justification cannot silently rot.

**Calibration procedure.** The calibration is univariate,
`target = intercept + slope * prediction`, and uses no metadata: severity,
family, font, text length, source and split identifiers never enter it. It is
fitted only on the expanded validation frame, whose predictions from every base
model are out-of-sample because the base models are trained only on training
sources. Validation scoring of a calibrated candidate uses leave-one-fold-out
calibration predictions over four source-grouped folds, so a candidate is
never scored on rows that contributed to its own calibration; the frozen policy
uses the calibration fitted on the whole validation frame. `clip_output` is
false: clipping to [0, 1] would silently absorb the clean-reference residual
and hide exactly the offset under study. A slice-conditional calibration was
deliberately not attempted, because it would require the corruption family at
inference time.

**Retention-aware selection rule.** The rule was declared before any test
evaluation. All three slice errors are RMSE in SSIM units, so they are directly
comparable without normalisation, and the score is the change relative to the
frozen M3 reference plus an explicit extra penalty on retention regressions:

```text
score = (rmse_geo - rmse_geo_ref)
      + w_clean * (rmse_clean - rmse_clean_ref)
      + w_original * (rmse_original - rmse_original_ref)
      + penalty * ( max(0, rmse_clean - rmse_clean_ref - tolerance)
                  + max(0, rmse_original - rmse_original_ref - tolerance) )
```

with `w_geometric = w_clean = w_original = 1.0`, `penalty_regression = 1.0` and
`tolerance = 0.0`. Lower is better and the frozen M3 model scores exactly 0. A
geometric gain and a retention loss of the same size cancel, and a retention
regression is charged twice, so it must be more than twice the geometric gain
to be preferred. Individual slice metrics are always reported next to the
composite score, so the aggregate cannot conceal a serious regression. No
weight or threshold was chosen after seeing a test result.

**Results.** The selected policy is the uncalibrated expanded random forest.
On the primary 1.0-3.0 px geometric test slice (36 rows / 12 sources) it
reaches MAE 0.0493, RMSE 0.0590 and R2 0.1855, against the frozen M3 model's
0.0777 / 0.0967 / -1.1858: a 39.0% RMSE reduction, and the only positive
geometric R2 among the candidates. The improvement holds on 11 of 12 held-out
sources and the single best source accounts for 22.9% of the total gain, so it
is not driven by one outlier. By severity it is worse than the frozen model at
1.0 px (MAE 0.0570 versus 0.0259) and better at 2.0 px (0.0195 versus 0.0662)
and 3.0 px (0.0714 versus 0.1409). On the untouched M3 test split it stays
level with the frozen model (MAE 0.0156 versus 0.0157, RMSE 0.0220 versus
0.0226, R2 0.9872 versus 0.9865). The clean-reference slice costs a little:
RMSE 0.0023 against the frozen 0.0012, a paired difference of 0.0011
(preliminary interval 0.0006 to 0.0017). On the original blur/noise slice the
paired RMSE difference is -0.0006 (interval -0.0026 to 0.0006), and the
per-source gains and losses nearly cancel, so no broad improvement is claimed
there.

**What the calibration did and did not do.** The affine calibration was tested
on every base model and was selected for none of them. It reduces the M5B
HGB's clean-reference offset (mean residual +0.0122 to +0.0038), confirming
that the M5B clean error is partly a systematic offset, but a single global
affine map cannot remove it without moving the other slices: the calibrated
HGB still leaves clean-reference RMSE at 0.0126, far above the frozen 0.0012.
The trade-off was therefore resolved mainly by the model family rather than by
the calibration layer.

**Decision.** The retention trade-off is resolved on this protocol, mostly by
the model family: a bagged-tree expansion retains the original task and clean
references far better than the boosted expansion M5B selected, while scoring
better on the geometric slice. Conventional feature-based ML remains the
benchmark champion. The remaining small clean-reference cost and the loss at
the mildest included severity are unresolved; the natural next question is
whether a retention-aware *training* objective, rather than a post-hoc
selection or calibration policy, can remove them. The evidence does not
justify a CNN, and a new corruption family does not by itself justify one.
These results cover one synthetic renderer, one canvas size, three fonts and 12
held-out source groups, and the frozen reference's own validation clean slice
is unusually noisy relative to its test clean slice, so the measured clean
regression is an upper bound. Nothing here measures human readability or
CAPTCHA solvability.

### 5.14 M5D retention-aware training objectives

M5C resolved the retention trade-off mainly by changing the *model family* and
showed that a post-hoc affine calibration did not help. M5D asks whether a
retention-aware *training objective* can do better. The implementation is
`retention_aware_training.py`, configured by
`configs/retention_aware_training_v0_1_0.toml`. It writes only to
`reports/retention_aware_training_v0_1_0/` and
`models/retention_aware_training_v0_1_0/` (both Git-ignored).

**Design.** M5D is a controlled training-objective experiment. The model family,
the hyperparameters, the training rows and the seed are held fixed across every
candidate, so the *only* thing that varies is the per-example sample weight.
That makes any observed difference attributable to the objective and to nothing
else. Four candidates were predeclared: the frozen M5C champion (loaded, never
refitted), an unweighted retrained control, a severity-aware weighting and a
retention-aware weighting. The control is deliberately included: because it
shares the champion's seed and data, it reproduces the champion's predictions
exactly, which turns it into both a control on the effect of retraining and a
reproducibility check on the frozen bundle.

**Grouping and weighting.** Each training example is assigned to exactly one
group using metadata already present in the training frame: `clean` (the M3
clean references), `original` (the M3 blur and noise rows) and `geometric` (the
M5A elastic rows at the primary rungs). The weight of an example is

```text
w(x) = group_weight[group(x)] * severity_factor[group(x), severity(x)]
```

where `severity_factor` is 1 for the clean and original groups and
`geometric_severity_ratio ** (severity - min_primary_severity)` for the
geometric group. The weights are rescaled so their mean over the training
frame is exactly 1, so a larger nominal scale cannot change their
interpretation, and every weight must be finite and strictly positive. The
weights are passed only to `fit(sample_weight=...)` and never enter the
predictor's feature matrix. The group assignment is verified to be unchanged
when the target column is removed, which proves it does not depend on labels.

Two properties are worth separating. The *per-row* weight is what the estimator
receives; the *share* of total training weight a group carries is what
determines that group's effective influence. A small group with a high per-row
weight can still carry a smaller share than a large group, so both are recorded
and reported. Under the severity-aware scheme the shares are clean 6.2%,
original 49.6%, geometric 44.2%; under the retention-aware scheme they are
clean 7.4%, original 59.3%, geometric 33.3%. The retention-aware scheme
therefore protects both retention slices relative to the severity-aware one
while still emphasising geometric examples per row.

**Selection rule.** The rule was declared before any test evaluation. The
champion is the reference. A candidate qualifies only if it regresses neither
the clean slice, nor the original-task slice, nor any single primary geometric
severity beyond the declared tolerances, and the per-severity constraint is
what stops an aggregate geometric gain from concealing a severe regression on
one rung. Among the qualifying candidates the score is

```text
score = w_geometric * (rmse_geo - rmse_geo_ref)
      + w_clean     * max(0, rmse_clean - rmse_clean_ref)
      + w_original  * max(0, rmse_orig  - rmse_orig_ref)
      + w_severity   * sum_s max(0, rmse_s - rmse_s_ref)
```

with `w_geometric = 1.0`, `w_clean = 2.0`, `w_original = 2.0`,
`w_severity = 1.0`, `tolerance_clean = 0.50`, `tolerance_original = 0.10` and
`tolerance_severity = 0.25`. Lower is better and the champion scores exactly 0.
Retention regressions are charged at twice the geometric rate per SSIM unit
because those slices are the incumbent's defining behaviour, and a per-severity
excess is charged again. Ties are broken by lower aggregate geometric RMSE and
then by candidate name, and the frozen champion is the fallback if nothing
qualifies.

**Results.** No retrained candidate was selected. The unweighted control
reproduces the champion exactly, with an identical artifact SHA-256. The
severity-aware scheme is *ineligible*: it improves the severe geometric rung
(severity 4 RMSE 0.05374 against the champion's 0.06240) but degrades the mild
one badly (severity 2 RMSE 0.07152 against 0.05254, 36% worse), and its
aggregate geometric RMSE (0.05542) is only 0.0005 different from the
champion's on the test slice. The per-severity constraint is what rejects it;
selecting on the aggregate alone would have hidden the regression. The
retention-aware scheme is eligible but scores +0.00618 against the champion's
0: it improves the clean slice (validation RMSE 0.00180 against 0.00223) but
costs 0.00066 on the aggregate geometric slice and 0.00132 on the original
task.

**What the negative result means.** Within a single model family, a single
shared feature representation and a single shared sample of 432 training rows,
sample weighting redistributes error across slices rather than removing it. The
severity-aware scheme buys accuracy at the severe rung by paying for it at the
mild rung; the retention-aware scheme buys clean-reference accuracy by paying
for it on the original task. The remaining headroom is more likely in the
*representation* than in the objective: the eight image-only features are
global whole-image statistics that do not describe *where* on the canvas a
distortion occurs, and a geometric warp moves ink locally while leaving many of
those statistics nearly unchanged. This is consistent with the near-zero
between-source R2 that M5B and M5C both observed, and it explains why
reweighting cannot help — reweighting cannot add spatial information the
representation never exposed. This milestone does not isolate the cause and
cannot rule out sample size or the objective itself.

**Decision.** The retention trade-off was **not** improved by a retention-aware
training objective, and the frozen M5C champion is retained. Conventional
feature-based ML remains the benchmark champion. The evidence does not justify
a CNN, and a new corruption family does not by itself justify one. If the line
continues, the defensible next question is whether a representation carrying
*spatial* information about the distortion can lift the near-zero
between-source R2, declared and evaluated on the same frozen split, rather than
another search over objectives or model families. These results cover one
synthetic renderer, one canvas size, three fonts and 12 held-out source groups;
nothing here measures human readability or CAPTCHA solvability.

### 5.15 M6A spatial-aware feature study

M5D concluded that the feature *representation*, not the training objective,
was the likely binding constraint: the eight image-only features are global
whole-image statistics that do not describe *where* on the canvas a distortion
occurs, and a geometric warp moves ink locally while leaving many of those
statistics nearly unchanged. M6A tests that directly, by asking whether adding
spatially localized features improves geometric-distortion prediction while
preserving clean-reference and original-task performance. The implementation
is `spatial_features.py` and `spatial_study.py`, configured by
`configs/spatial_feature_study_v0_1_0.toml`. It writes only to
`reports/spatial_feature_study_v0_1_0/` and
`models/spatial_feature_study_v0_1_0/` (both Git-ignored).

**Isolation from the frozen pipeline.** The eight global features and their
extraction code are unchanged, and `FEATURE_SET_VERSION` remains `v1`. The
spatial features live in a separate module with their own version
(`spatial-v1`) and their own names, so prior model inputs and artefacts cannot
be silently altered. The study reuses the M5C verified path, so the frames,
the source-grouped split and the M5B/M5C merge and isolation checks are
identical to the preceding milestones.

**Feature design.** The canonical 192x64 image is divided into a fixed 4x2
grid of eight 16x96-pixel cells, visited in row-major order. Two families are
implemented, both computed from the degraded image alone:

- `ink_occupancy` - the fraction of pixels at or below the Otsu threshold of
  the *whole* image, counted within the cell. The threshold is computed once
  per image, so the cell values share one ink definition and are directly
  comparable with the global `ink_fraction` feature.
- `edge_density` - the fraction of Canny-marked pixels within the cell, using
  the repository's existing edge convention (sigma=1.0, low_threshold=0.1,
  high_threshold=0.2 on the image scaled to [0, 1]). The edge map is computed
  once per image, so the cell values share one edge definition.

Regional intensity statistics were considered and rejected: on training rows
the per-cell mean intensity is largely redundant with the per-cell ink
occupancy (maximum |correlation| 0.873 across cell pairs), so adding it would
inflate the feature count without adding information. A 4x2 grid was chosen
over 2x4 because it has lower internal redundancy among the spatial features
(mean pairwise |r| 0.38 against 0.52) at a comparable correlation with the
target. Both decisions were made from training rows only, before any test
evaluation.

The extractor never sees the clean reference, the SSIM/PSNR targets, the source
identifier, the corruption family, the severity, the file name or the split
label. An exactly uniform image has no distinguishable ink class and no edges,
so all its spatial features are zero; an empty cell also yields zero rather
than a division by zero.

**Ablations and selection.** Three ablations are predeclared. The model
family, the hyperparameters, the seed and the training partition are held
fixed across all of them, so the only thing that varies is the input
representation: `global_baseline` (the eight global features, using the frozen
M5C champion without refitting), `spatial_only` (the 16 spatial features,
refitted) and `combined` (8 + 16 = 24 features, refitted). Selection uses the
M5D retention-aware rule with identical coefficients and tolerances, so the
two studies are directly comparable, and the frozen M5C champion is the
fallback. No coefficient or tolerance was chosen by inspecting a test result.

**Results.** The champion is retained. The `combined` ablation improves
aggregate geometric validation RMSE by 0.00056 but regresses the original-task
slice by 15.2%, beyond the 10% tolerance, so it is ineligible. The
`spatial_only` ablation is ineligible on four slices and far worse everywhere:
it discards the global summary that actually tracks blur and noise. On the
test set the `combined` geometric RMSE is 0.0584 against the champion's
0.0590, an improvement whose paired interval includes zero, while its
original-task and M3-retention regressions have intervals that exclude zero.

**Why: a grid-resolution diagnostic.** The headline metrics cannot say whether
the declared grid is even capable of resolving the distortion, so the study
also measures, per source and on training rows only, the per-cell
ink-occupancy change between a clean reference and its distorted image. The
grid resolves blur about 25 times more strongly than the elastic family (mean
absolute occupancy change 0.136 against 0.005), and the correlation between
that change and the SSIM drop is weaker for the elastic family (+0.57) than
for blur (+0.89). A 1-3 px warp moves ink by a fraction of a 16x96-pixel
cell, so a coarse partition averages most of the effect away: the spatial
features largely re-encode how much ink the image has, which the global
features already state. This is the mechanism behind both the ineligibility of
`combined` and the failure of `spatial_only`, and it is a statement about the
*declared resolution*, not about localization in general.

**Decision.** Spatial features did not improve geometric-distortion prediction
under the predeclared retention-aware rule, and the frozen M5C champion is
retained. Conventional feature-based ML remains the benchmark champion. The
negative result is informative rather than discouraging: it rules out coarse
regional summaries as the missing information and identifies the resolution at
which a spatial feature would have to operate to see a 1-3 px warp at all. The
evidence does not justify a CNN, and a new corruption family does not by itself
justify one. If the line continues, the defensible next experiment is a
predeclared *resolution* study that holds the feature families and the model
fixed and varies only the grid resolution, so the point at which a spatial
feature begins to see the warp is measured on the same frozen split rather than
assumed. These results cover one synthetic renderer, one canvas size, three
fonts and 12 held-out source groups; nothing here measures human readability
or CAPTCHA solvability.

### 5.16 M6B spatial resolution study

M6A found that a coarse 4x2 grid of spatial features did not help, and measured
why: the grid resolved blur about 25 times more strongly than a 1-3 px warp.
M6B varies only the grid resolution to test whether that resolution mismatch
was the blocker. The implementation is `spatial_resolution_study.py`,
configured by `configs/spatial_resolution_study_v0_1_0.toml`. It writes only to
`reports/spatial_resolution_study_v0_1_0/` and
`models/spatial_resolution_study_v0_1_0/` (both Git-ignored).

**Design.** The two M6A feature families are reused with no change to their
mathematical definitions: regional ink occupancy uses the per-image Otsu
threshold, and regional edge density uses the repository's Canny convention
(sigma=1.0, low_threshold=0.1, high_threshold=0.2 on the image scaled to
[0, 1]). Both maps are computed once per image, so all cells at a given
resolution share one definition. The grayscale representation, the canonical
192x64 canvas and the row-major cell ordering are unchanged. Cells are bounded
by floor(i*height/r) .. floor((i+1)*height/r), which tiles the canonical canvas
evenly at every declared resolution and gives any remainder to the later cells;
an empty cell yields zero for every feature and an exactly uniform image has no
distinguishable ink class and no edges, so all its features are zero.

Four resolutions are predeclared, each in two representations. The model
family, hyperparameters, seed, training rows and objective are identical for
every candidate, so only the input representation and its resolution differ:

| grid | cells | cell size (px) | spatial features | combined features |
| --- | ---: | ---: | ---: | ---: |
| 2x2 | 4 | 32x96 | 8 | 16 |
| 4x2 | 8 | 16x96 | 16 | 24 |
| 4x4 | 16 | 16x48 | 32 | 40 |
| 8x8 | 64 | 8x24 | 128 | 136 |

For an r x c grid each family contributes r*c features, so the spatial-only
representation has 2*r*c features and the global-plus-spatial representation
has 8 + 2*r*c. Because the forest considers every feature at each split
(`max_features` is 1.0), a larger feature set dilutes the split search
directly; the redundancy and train-versus-validation diagnostics are therefore
reported for every resolution rather than tuned around.

**Selection.** The M6A rule is used unchanged, with the same coefficients and
tolerances, so the two studies are directly comparable, and the frozen M5C
champion is the fallback. The 4x2 grid is a reference condition, not a reason
to favour or reject that resolution in advance. As a cross-milestone check,
the 4x2 candidates reproduce the M6A 4x2 candidates bit-for-bit.

**Results.** No spatial candidate qualifies. Every combined candidate breaches
the 10% original-task tolerance, and every spatial-only candidate breaches it
plus the clean-slice and per-severity tolerances. The aggregate geometric
validation RMSE is best at the *coarsest* grid, and no resolution produces a
demonstrable geometric gain: every paired geometric interval includes zero.
The retention regressions are systematic and grow monotonically with the feature
count, with the clean-slice and original-task intervals excluding zero from
4x2 upwards.

**Why.** M6A's diagnosis is confirmed and its obvious follow-up fails. The
training-row grid diagnostic shows the finer grid does resolve the geometric
warp: the per-cell occupancy change against the same source's clean reference
rises from 0.00424 at 2x2 to 0.01068 at 8x8, and its correlation with the SSIM
drop rises from +0.453 to +0.828, closing most of the gap to blur's +0.894. The
information is therefore present, but the model cannot use it. The extra
features are informative about the geometric family and largely uninformative
about blur, noise and clean references, so they dilute the search for the
splits that matter on the other families. Redundancy is not the mechanism: mean
absolute pairwise correlation *falls* with resolution (0.585 at 2x2 to 0.259 at
8x8) and no feature is constant or duplicated at any grid, while the
overfitting gap grows (0.01739 to 0.01867 for the combined candidates).

**Decision.** The retention trade-off was not improved by increasing the
spatial resolution, and the frozen M5C champion is retained. The negative
result is sharper than M6A's because it isolates the mechanism: the blocker is
not the resolution of the partition but the **relevance** of the resulting
features to the other corruption families. Conventional feature-based ML
remains the benchmark champion. The evidence does not justify a CNN, and a new
corruption family does not by itself justify one. If the line continues, the
defensible next experiment follows from the mechanism: whether a feature
specifically sensitive to local *displacement*, rather than to local ink
quantity, can add information about the geometric family without diluting the
splits that support blur, noise and clean references. These results cover one
synthetic renderer, one canvas size, three fonts and 12 held-out source groups;
nothing here measures human readability or CAPTCHA solvability.

### 5.17 M6C local structural feature study

M6B isolated the mechanism behind the M6A/M6B negative results: the blocker was
not the resolution of the spatial partition but the relevance of regional ink
summaries to the other corruption families. M6C tests the follow-up directly:
whether local *structural* features, computed from the corrupted image alone,
add predictive information beyond the eight global features while preserving
clean-reference and original-task performance. The implementation is
`local_structure_features.py` and `local_structure_study.py`, configured by
`configs/local_structure_study_v0_1_0.toml`. It writes only to
`reports/local_structure_study_v0_1_0/` and
`models/local_structure_study_v0_1_0/` (both Git-ignored).

**Feature design.** Fourteen features are declared in two interpretable
families. The image is scaled to `[0, 1]`; Sobel gradients are computed with
scikit-image's normalised 3x3 operator and reflected boundaries.

| family | features | definition |
| --- | ---: | --- |
| `gradient_magnitude` | 4 | mean, population standard deviation, median and 0.90 quantile of `hypot(gx, gy)` |
| `gradient_orientation` | 10 | eight equal-width bins over `[0, pi)` of `arctan2(gy, gx)` folded into `[0, pi)`, plus the circular resultant length of the doubled angles (coherence) and the normalised Shannon entropy of the histogram |

The orientation family excludes pixels whose gradient magnitude is below a
fixed weak-gradient threshold of `0.10` on the `[0, 1]` scale; the surviving
counts are normalised to sum to one. The folded angle is undirected, because an
edge orientation has no sign. The coherence is in `[0, 1]` and is 1 when every
strong edge shares one orientation; the entropy is in `[0, 1]` and is 0 when
one bin holds all the mass. The families were chosen a priori for
interpretability and cost, not by searching a feature space: gradient magnitude
is the canonical measure of local-structure strength and gradient orientation
is the canonical description of stroke direction and anisotropy. Regional
intensity statistics, edge-continuity graphs and learned filters are not used.

Degenerate inputs are explicit. An exactly uniform image has zero gradient
everywhere, so all magnitude statistics are zero and, because no pixel clears
the weak-gradient threshold, the orientation histogram, coherence and entropy
are all zero. An empty strong-gradient set is a defined zero, not a division by
zero. Every output is finite and deterministic; a non-empty orientation
histogram sums to one and an empty one sums to zero by construction. The
feature path is a fixed Sobel operator on the single input image: there is no
registration, optical-flow or difference-map step, and no second image is
read.

**No-reference boundary.** The extractor signature is
`extract_local_structure_features(image, spec=None)`. The only required input is
the corrupted 2-D `uint8` image array. There is no parameter for, and no code
path to, the clean reference, the SSIM/PSNR targets, the source identifier, the
corruption family or severity, the font, the split label, the file path or any
metadata. The clean reference is used only by the separate evaluation pipeline
that computes the SSIM label. The boundary is enforced structurally on every
run, not by checking output column names:

1. A signature check confirms that the extractor accepts exactly `image` and an
   optional `spec`, with no `*args`, no `**kwargs` and no other required input.
2. An AST scan of the feature module's code confirms that no target, reference,
   source, severity, family, font, split or path identifier appears as a
   variable, argument, attribute or import. Prose that explains the boundary is
   ignored, because prose is not a data path.
3. A schema check confirms the extracted table contains only `sample_id` and
   the declared image features, and a leakage check confirms no target or
   metadata column is present.

The tests repeat the signature and AST checks independently, verify that a
second image cannot be passed positionally, and verify that an opaque
`sample_id` never changes a feature value. The extractor imports only the
shared image-contract validator from the repository; it cannot reach the
feature, modelling, retention or dataset modules, and it uses no dynamic
import or file I/O.

**Candidates and selection.** Three candidates are predeclared. The frozen M5C
champion is loaded without refitting; the local-only and global-plus-local
candidates are refitted on the same 432 expanded training rows with the same
model family, hyperparameters, seed and objective (random forest, 300 trees,
unlimited depth, leaf size 1, seed 798117391). No hyperparameter is tuned, no
post-hoc calibration is applied and no feature is removed in response to a
diagnostic. The M6A/M6B retention-aware selection rule is used unchanged, with
the same coefficients and tolerances, and the frozen champion is the fallback.
The 0.5 px rung remains diagnostic only and never enters fitting, selection or
the primary score. The M6B regional grids are not used by any candidate.

**Results.** Both local candidates are ineligible on validation.
`global_plus_local` breaches the 50% clean-reference tolerance (validation RMSE
0.00621 versus 0.00223). `local_only` breaches the 10% original-task tolerance
(0.02850 versus 0.02007), the 25% severity-3 tolerance (0.03145 versus 0.02494)
and the 25% severity-4 tolerance (0.08451 versus 0.06240). The aggregate
geometric validation RMSE is 0.04925 for the champion, 0.05319 for the combined
candidate and 0.05957 for the local-only candidate. The champion is retained.

On the untouched test slices the champion reaches geometric RMSE 0.05901
(R2 0.18553), clean RMSE 0.00228, original-task RMSE 0.02333 and M3 test RMSE
0.02201 (R2 0.98716), reproducing the pinned M5C values. The local candidates
do not improve the aggregate geometric slice: `global_plus_local` is 0.00438
worse (paired interval [-0.00015, 0.00891]) and `local_only` is 0.00711 worse
([0.00165, 0.01204]). Both candidates improve the 1 px rung and regress the 2 px
and 3 px rungs; the 1 px intervals include zero. Retention is clearly worse for
both: the clean, original-task and M3 paired intervals exclude zero for
`local_only`, and for `global_plus_local` the clean, original-task and M3
intervals exclude zero. There are 12 held-out test source groups, so all
intervals are preliminary and descriptive.

**Why.** The training-only diagnostics show that the local family is largely
redundant with the global eight. The maximum absolute pairwise correlation among
the local features is 0.9926 (the four magnitude statistics duplicate one
another), and the maximum absolute local-versus-global correlation in the
combined candidate is 0.8743. Random-forest importance (descriptive, not causal)
gives the combined candidate a local share of 0.2283, with `histogram_entropy`
alone at 0.6303 and `grad_mag_std` at 0.1521; the local-only model is dominated
by `grad_mag_median` (0.6421) and `grad_mag_std` (0.2508). The
training-to-validation geometric RMSE gap is 0.02645 for the champion, 0.03093
for the combined candidate and 0.03799 for the local-only candidate, so the
larger feature sets also overfit more. The local family partly re-expresses the
edge and variance information the global features already carry, and its
genuinely new content is not sufficient to improve the geometric slice without
costing retention.

**Decision.** The frozen M5C champion is retained. Local structural features do
not improve geometric-distortion SSIM prediction beyond the global eight while
preserving clean-reference and original-task performance. Conventional
feature-based ML remains the benchmark champion. The evidence does not justify
a CNN, and a new corruption family does not by itself justify one. The most
defensible follow-up is a representation explicitly targeted at local
*displacement* (for example a multi-scale structure-tensor anisotropy or local
autocorrelation family) under the same no-reference boundary and the same
predeclared rule; if that also fails, the accumulated evidence points to a
learned representation as a separate milestone. These results cover one
synthetic renderer, one canvas size, three fonts and 12 held-out source groups;
nothing here measures human readability or CAPTCHA solvability.

### 5.18 M6D multi-scale structure-tensor study

M6C pointed to a representation explicitly sensitive to local *displacement*
rather than local gradient strength or orientation. M6D tests the most direct
handcrafted candidate: a compact multi-scale structure tensor computed from the
corrupted image alone. The implementation is `structure_tensor_features.py` and
`structure_tensor_study.py`, configured by
`configs/structure_tensor_study_v0_1_0.toml`. It writes only to
`reports/structure_tensor_study_v0_1_0/` and
`models/structure_tensor_study_v0_1_0/` (both Git-ignored).

**Feature design.** The image is scaled to `[0, 1]`. Horizontal and vertical
gradients are computed with scikit-image's fixed 3x3 Sobel operator with
reflected boundaries; no extra normalisation is applied. The tensor components
`Jxx = gx*gx`, `Jyy = gy*gy` and `Jxy = gx*gy` are smoothed by a Gaussian with
the declared scale, reflected boundaries and truncation at four standard
deviations. At each of exactly two predeclared scales, `sigma = 1.0` and
`sigma = 3.0` px, the symmetric 2x2 tensor has non-negative eigenvalues

```text
lambda_1 = (trace + sqrt((Jxx - Jyy)^2 + 4*Jxy^2)) / 2
lambda_2 = (trace - sqrt((Jxx - Jyy)^2 + 4*Jxy^2)) / 2
trace    = Jxx + Jyy
```

with `lambda_2` clipped to `[0, lambda_1]` for round-off. The summaries are:

| statistic | definition | range |
| --- | --- | --- |
| coherence | `(lambda_1 - lambda_2) / (lambda_1 + lambda_2 + epsilon)` | `[0, 1]` |
| anisotropy | `1 - lambda_2 / (lambda_1 + epsilon)` | `[0, 1]` |
| energy | `lambda_1 + lambda_2` | `[0, inf)` |

`epsilon = 1e-12` is fixed. A zero-energy tensor is an explicit zero, not a
division by zero. The four summaries per scale are mean coherence, the 0.90
coherence quantile, mean anisotropy and the 0.90 energy quantile; quantiles use
NumPy's linear interpolation. Mean tensor energy is reported once as
`st_energy_mean` at the finest scale, because Gaussian smoothing preserves the
mean and repeating it at both scales would add a duplicate column. The nine
features, in fixed order, are `st_s1_coh_mean`, `st_s1_coh_p90`,
`st_s1_aniso_mean`, `st_s1_energy_p90`, `st_s2_coh_mean`, `st_s2_coh_p90`,
`st_s2_aniso_mean`, `st_s2_energy_p90` and `st_energy_mean`.

Degenerate inputs are explicit. An exactly uniform image has zero gradient
everywhere and produces nine defined zeros. A single faint step produces finite,
bounded summaries. Every output is finite and deterministic, and the extractor
does not mutate its input. A known sinusoidal grating at vertical, horizontal or
diagonal orientation yields coherence and anisotropy above 0.99 at both scales;
the high-energy tail is larger at the finer scale, as expected from smoothing.
Isotropic noise is less coherent than a grating. These are tests of the declared
definition, not claims about CAPTCHA images.

**No-reference boundary.** The extractor signature is
`extract_structure_tensor_features(image, spec=None)`. The only required input is
the corrupted 2-D `uint8` image array. There is no parameter for, and no code
path to, the clean reference, the SSIM/PSNR targets, the source identifier, the
corruption family or severity, the font, the split label, the file path or any
metadata. The clean reference is used only by the separate evaluation pipeline
that computes the SSIM label. The boundary is enforced structurally on every
run:

1. A signature check confirms that the extractor accepts exactly `image` and an
   optional `spec`, with no `*args`, no `**kwargs` and no other required input.
2. An AST scan of the feature module's code confirms that no target, reference,
   source, severity, family, font, split or path identifier appears as a
   variable, argument, attribute or import.
3. A schema check confirms the extracted table contains only `sample_id` and the
   declared image features, and a leakage check confirms no target or metadata
   column is present.

Tests repeat the signature and AST checks independently, verify that a second
image cannot be passed positionally, and verify that an opaque `sample_id` never
changes a feature value. The extractor imports only the shared image-contract
validator from the repository; it cannot reach the feature, modelling,
retention or dataset modules, and it uses no dynamic import or file I/O.

**Candidates and selection.** Three candidates are predeclared. The frozen M5C
champion is loaded without refitting; the structure-only and
global-plus-structure candidates are refitted on the same 432 expanded training
rows with the same model family, hyperparameters, seed and objective (random
forest, 300 trees, unlimited depth, leaf size 1, seed 798117391). No
hyperparameter is tuned, no post-hoc calibration is applied and no feature is
removed in response to a diagnostic. The M6A/M6B/M6C retention-aware selection
rule is used unchanged, with the same coefficients and tolerances, and the
frozen champion is the fallback. The 0.5 px rung remains diagnostic only and
never enters fitting, selection or the primary score.

**Validation results.** The aggregate geometric validation RMSE is 0.04925 for
the champion, 0.04935 for `global_plus_structure` and 0.05189 for
`structure_only`. Both new candidates are ineligible. `global_plus_structure`
breaches the 50% clean-reference tolerance (0.00516 versus 0.00223), and
`structure_only` breaches both the 50% clean-reference tolerance (0.01540 versus
0.00223) and the 10% original-task tolerance (0.03315 versus 0.02007). The
per-severity constraints are satisfied by both new candidates on validation.
The champion is retained.

**Test results.** The champion reproduces the pinned M5C values: geometric RMSE
0.05901 (R2 0.18553), clean RMSE 0.00228, original-task RMSE 0.02333 and M3
overall RMSE 0.02201 (R2 0.98716). The combined candidate reaches geometric
RMSE 0.06036 (R2 0.14780), clean 0.00328, original 0.02526 and M3 0.02384. The
structure-only candidate reaches geometric 0.06532 (R2 0.00199), clean 0.01822,
original 0.03879 and M3 0.03707. The aggregate geometric paired differences
versus the champion are +0.00135 [-0.00390, 0.00575] for the combined candidate
and +0.00631 [-0.00019, 0.01181] for the structure-only candidate. The combined
candidate's clean interval includes zero (+0.00101 [-0.00026, 0.00188]), while
its original-task and M3 intervals also include zero; the structure-only
candidate's clean, original-task and M3 intervals exclude zero. The combined
candidate improves severity 2 (paired -0.01894, interval excludes zero) but
regresses severity 3 (+0.01486, excludes zero) and severity 4 (+0.00937,
includes zero). There are 12 held-out test source groups, so every interval is
preliminary and descriptive.

**Diagnostics.** On 432 training rows, none of the nine features is constant,
duplicated or non-finite. The mean absolute pairwise correlation among the new
features is 0.6439 and the maximum is 0.9977, between `st_s2_coh_mean` and
`st_s2_aniso_mean`. The energy summaries are also strongly redundant:
`st_s1_energy_p90` versus `st_s2_energy_p90` 0.9935, versus `st_energy_mean`
0.9854, and `st_s2_energy_p90` versus `st_energy_mean` 0.9820. In the combined
candidate the maximum absolute new-versus-global correlation is 0.9387, between
`st_s1_coh_mean` and `ink_fraction`. Random-forest importance (descriptive, not
causal) gives the combined candidate a structure-tensor share of 0.1362, with
`histogram_entropy` alone at 0.6292. The structure-only candidate is dominated
by `st_s1_coh_p90` (0.3376), `st_s2_energy_p90` (0.2889) and
`st_s1_energy_p90` (0.2837). The training-to-validation geometric RMSE gap is
0.02645 for the champion, 0.02894 for the combined candidate and 0.03166 for the
structure-only candidate.

**Decision.** The frozen M5C champion is retained. Multi-scale structure-tensor
features do not improve geometric-distortion SSIM prediction beyond the global
eight while preserving clean-reference and original-task performance. The
structure tensor is a different representation from M6C's gradient summaries,
but it is not a different enough source of information: its coherence,
anisotropy and energy summaries are strongly redundant with one another and with
the global edge/ink statistics. Conventional feature-based ML remains the
benchmark champion. The evidence does not justify a CNN on this result alone.
The M6A-M6D series now covers coarse regional summaries, finer regional grids,
whole-image gradient summaries and multi-scale structure-tensor summaries; none
adds a retention-safe geometric gain. These results cover one synthetic
renderer, one canvas size, three fonts and 12 held-out source groups; nothing
here measures human readability or CAPTCHA solvability.

### 5.19 M7A dataset-adequacy and evaluation audit

M7A is a **read-only audit** of the frozen synthetic dataset, target
definitions, source-grouped split, M5C champion and M6A–M6D feature studies. It
does not generate images, train or recalibrate a model, change a target, change
a split, add a corruption family or select a new champion. Its purpose is to
establish what the existing benchmark can and cannot support before any further
modelling.

**Pinned inputs and isolation.** `configs/dataset_adequacy_audit_v0_1_0.toml`
pins 24 upstream files by SHA-256, including the M3/M5A configs and manifests,
the M5B protocol and split assignment, the M5C champion artifact/card and
prediction/metric/selection tables, and the M3 held-out-font comparison. The
audit loads the frames through the existing M5B/M5C verified path, replays the
champion's stored test predictions from the serialized model, and checks the
stored test-slice metrics against that replay. It fingerprints the protected
asset tree before and after the run and refuses to write into any frozen path.
The independent unit is `source_id`; image variants are correlated and are
never treated as independent observations.

**Dataset composition.** The audit reports, per source and split, the number of
M3 and M5A rows, distinct images, clean references, blur/noise severities,
control types, elastic variants, font, text length, resolution, pixel format,
renderer and clean-image hash. It accounts for every row in M3, M5A, the
expanded training/validation frames and the evaluation scopes. It verifies
source disjointness, M3/M5A source identity, shared font sets, clean-reference
hash equality, exact control relationships and the declared split counts. The
key facts are 60 sources, 36/12/12 split, 540 M3 rows, 480 M5A rows, three shared
fonts, one renderer, one 192x64 resolution and one 8-bit grayscale PNG format.
M5A has 360 distinct images because identity and zero-field controls are exact
clean duplicates and the minimal control is near-clean.

**Target distributions.** For clean, original, each original severity, primary
geometric severities 2–4 and the 0.5 px diagnostic, the audit reports counts,
min/max/mean/SD, 5/25/50/75/95 quantiles, variance, between-source variance,
mean within-source variance, a descriptive between-variance share and a
near-degeneracy flag. The predeclared near-degeneracy margin is target SD
`< 0.005`, inherited from M5A. The clean target is constant at 1.0, so its
variance is zero and R² is undefined. The 0.5 px diagnostic has SD 0.00266 and
is reported as diagnostic only.

**Residuals and influence.** The audit reports per-source MAE/RMSE, mean and SD
of residuals, per-source/per-severity RMSE where the slice supports it,
per-font residuals, squared-error contribution and concentration (top-1, top-k,
HHI, effective source count), and descriptive leave-one-source-out RMSE/MAE
changes. No model is refit during these calculations, so they are influence
measures on the frozen predictions, not new generalisation estimates.

**Uncertainty.** For every declared scope, the audit runs 2000 source-grouped
percentile bootstrap replicates with seed `20261108` (scope-specific derived
seeds). Each replicate resamples whole source groups with replacement and keeps
all variants of a sampled source together. The tables record the method, seed,
replicate count, confidence level, number of groups, number of rows and a
12-group caveat. Intervals are descriptive and unstable with so few groups; no
significance test is performed.

**Feature diagnostics.** On the 432 expanded training rows from 36 sources, the
audit reports constants, exact duplicates, scales and distributions, Pearson and
Spearman correlations, frozen-model impurity importance, condition number, VIF
and family/severity feature means. These diagnostics are descriptive only. No
feature is removed, added or redesigned from test outcomes. The strongest
training-only correlation is |r| 0.991 between `intensity_std` and
`contrast_rms`; `histogram_entropy` has the largest impurity importance
(0.633), which is not a causal claim.

**Claim table and direction comparison.** Twelve claims are classified as
`supported`, `suggestive` or `unsupported` against the frozen benchmark. Three
are supported, two suggestive and seven unsupported. The audit compares four
predeclared next directions on hypothesis, scientific value, data needs,
implementation needs, leakage/generalisation risk, success measurement and
whether current data can answer the question, and recommends exactly one
without executing it. M7A recommends **B: add explicitly held-out fonts or
rendering conditions in a future versioned dataset**.

**Verification.** `captcha-dataset-audit run` writes only to the Git-ignored
`reports/dataset_adequacy_audit_v0_1_0/` directory. `captcha-dataset-audit
verify` reloads the pinned inputs, replays the audit core, compares every table
and the report text, checks recorded output hashes, rechecks frozen-asset
fingerprints, and validates the claim and recommendation schemas. Two runs from
the same clean commit are byte-identical except for the runtime-only
`runtime_seconds` metadata field. M7B (§5.20) implements the recommended
held-out-font direction as a separate versioned evaluation dataset.

### 5.20 M7B held-out-font generalisation benchmark

M7A recommended adding explicitly held-out fonts or rendering conditions. M7B
implements the font part of that recommendation as a **separate versioned
evaluation dataset**; it does not modify M3, M5A, the frozen splits, the target,
the corruption definitions, the feature extractor or the champion. The question
is narrow: how well does the frozen M5C champion predict SSIM on text rendered
in fonts that were absent from model development?

**Held-out condition and isolation.** The configured fonts are Georgia
(proportional serif), Monaco (monospaced humanist sans) and Impact (condensed
heavy display sans). Their files are pinned by SHA-256, checked for a usable
glyph set, and rejected if missing or changed. None of the three keys or file
hashes appears in M3. Source IDs are generated under the M7B generation hash and
are checked for disjointness from the M3 source IDs. The dataset contains 20
sources per font and 60 sources total. Every source contributes one clean image,
four blur levels, four noise levels, four thin-line levels, four arc-stroke
levels and four elastic levels (0.5, 1, 2 and 3 px), for 21 rows per source and
1,260 rows total. All rows are in one untouched `eval` split. The source is the
independent unit; all variants of a source stay together.

**Frozen model and preprocessing.** The primary model is the exact M5C
`expanded_random_forest` champion with calibration `none`. The artifact SHA-256
is `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`
and the card SHA-256 is
`9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8`. The
existing eight-feature extractor, feature order, target (`ssim_target`) and
renderer are reused unchanged. The code loads the serialized estimator, checks
its feature schema and scikit-learn major/minor compatibility, and never calls
`fit`, `partial_fit`, a calibrator or a selector. The dataset directory is
checked to contain no model artifact. The frozen M3–M7A asset tree is
fingerprinted before and after generation/evaluation and must be unchanged.

**Target and references.** SSIM is computed between every corrupted image and
the exact source-specific clean reference. Clean references are preserved as
separate PNG files and hashed; PSNR is recorded as a diagnostic only. The
0.5 px elastic rung and the constant clean target are diagnostic, not primary.
R² is suppressed when target SD is below the predeclared 0.005 margin, even if
it is numerically defined.

**Results.** The primary held-out slice has 1,140 rows and 60 sources. MAE is
0.0634, RMSE 0.0930 and R² 0.7435; the source-grouped 95% intervals are
[0.0592, 0.0678] for MAE and [0.0867, 0.0986] for RMSE (2,000 whole-source
bootstrap replicates, seed `20261109`). The constant clean target has MAE
0.0094 and RMSE 0.0142, with R² undefined. The 0.5 px diagnostic has target SD
0.00338 and is near-degenerate.

| Slice | Rows | Sources | Target mean | Target SD | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All primary | 1,140 | 60 | 0.7778 | 0.1835 | 0.0634 | 0.0930 | 0.7435 |
| Original blur/noise | 480 | 60 | 0.6439 | 0.1964 | 0.0601 | 0.0959 | 0.7614 |
| Gaussian blur | 240 | 60 | 0.5657 | 0.2263 | 0.0488 | 0.0847 | 0.8599 |
| Gaussian noise | 240 | 60 | 0.7221 | 0.1170 | 0.0713 | 0.1060 | 0.1796 |
| Interference | 480 | 60 | 0.8681 | 0.0924 | 0.0727 | 0.1001 | -0.1744 |
| Thin lines | 240 | 60 | 0.8573 | 0.0980 | 0.0816 | 0.1110 | -0.2821 |
| Arc strokes | 240 | 60 | 0.8789 | 0.0850 | 0.0638 | 0.0878 | -0.0690 |
| Primary geometric | 180 | 60 | 0.8941 | 0.0687 | 0.0476 | 0.0588 | 0.2657 |
| Clean (diagnostic) | 60 | 60 | 1.0000 | 0.0000 | 0.0094 | 0.0142 | undefined |
| 0.5 px elastic (diagnostic) | 60 | 60 | 0.9909 | 0.0034 | 0.0325 | 0.0385 | undefined |

The per-font primary results are uneven: Georgia MAE/RMSE/R²
0.0569/0.0852/0.7439, Monaco 0.0528/0.0761/0.7998 and Impact
0.0807/0.1134/0.7055. On original blur/noise, Georgia and Monaco are much
stronger (RMSE 0.0279 and 0.0346) than Impact (0.1601, R² 0.4625). On
interference, Georgia and Monaco have negative R² (-0.5523 and -0.3767) while
Impact is less bad (0.0864); this is a per-font reversal, not evidence that
Impact is generally better. Primary geometric RMSE is similar across fonts
(0.0543, 0.0525 and 0.0684).

**Comparison with the familiar M5C benchmark.** The familiar benchmark uses 12
M3 test sources and Arial, Courier New and Times New Roman. M7B uses 60
held-out sources and three different fonts; M7B also adds thin-line and
arc-stroke interference. Target distributions are similar for the original
blur/noise slice (M7B mean 0.6439 / SD 0.1964 versus familiar 0.6560 / 0.1711)
and geometric primary (0.8941 / 0.0687 versus 0.8965 / 0.0654), but the source
composition and font condition differ. The comparison is therefore
**descriptive, not a matched experiment or significance test**. With that
caveat, original blur/noise RMSE is 0.0959 versus 0.0233 and both MAE and RMSE
lie outside the familiar-font source-grouped intervals. Geometric primary RMSE
is 0.0588 versus 0.0590 and lies inside the familiar interval. The 1 px
geometric rung has R² -20.96 versus -44.10 on the familiar benchmark; both
values reflect very small target variance and should not be read as a stable
ranking, and the held-out point estimate is outside the familiar interval only
because both estimates are unstable.

**Interpretation.** The evidence supports a descriptive statement that
zero-shot SSIM prediction is worse on these three unseen fonts, most clearly
for the original blur/noise task and for interference, while aggregate geometric
error is similar. It does not estimate a font population, identify a causal
font property, prove that the champion fails on arbitrary fonts, or support
claims about human readability, CAPTCHA solvability, security or production
robustness. The three typefaces are a purposive sample, not a random sample.
The 60 sources give tighter intervals than the 12-source familiar benchmark,
but the comparison between benchmarks remains unmatched.

**Verification and reproducibility.** `captcha-heldout-font-benchmark run`
writes only to the Git-ignored dataset and report directories. `verify` reloads
the dataset, recomputes every image hash, SSIM/PSNR target and feature value,
replays the frozen champion, rebuilds every table and the report text, checks
recorded output and dataset hashes, rechecks frozen-asset fingerprints and fails
if any output, font, model or protected asset differs. Generation is
byte-reproducible. Two runs from the same clean commit differ only in
`runtime_seconds`; the committed inputs are the TOML configuration, code, tests
and documentation.

### 5.21 M7C cross-font evaluation analysis

M7C is a **read-only analysis** of the frozen M7B held-out-font predictions and
the frozen M5C familiar-font predictions. It does not generate a dataset, train,
refit, recalibrate, select or tune a model, change a target, change a split, or
write into any prior dataset/model/report directory. Its question is descriptive:
what does the existing evidence say about per-font performance, corruption
families, geometric severity, source-level error concentration, uncertainty and
familiar-versus-held-out differences, and which claims does that evidence
support?

**Pinned inputs and isolation.** `configs/cross_font_analysis_v0_1_0.toml`
pins 31 upstream files by SHA-256: the M7B protocol, manifest, source identities,
feature table, dataset summary, predictions, metrics, uncertainty, comparison,
target distributions, composition, isolation checks, provenance and report; the
M5C protocol, familiar-font predictions, metrics, uncertainty, paired interval
convention, provenance, report and champion card/artifact; the M5A protocol and
severity manifest; and the M7A protocol, uncertainty and concentration tables.
Every path is read-only. The analysis refuses to write outside the Git-ignored
`reports/` tree or into any protected prior report directory. The M7C run
fingerprinted 360 protected non-image files under `configs/`, `data/processed/`,
`models/` and the prior report directories before and after the run and failed
if any byte changed. That run-time guard has since been scoped to the declared
protected-dependency membership described in the *Protected-dependency
contract* paragraph of this section; the recorded 360-file snapshot is
unchanged.

**Independent unit and scopes.** The independent unit is `source_id`, not an
image variant. Every bootstrap replicate resamples whole source groups with
replacement and keeps all rows of a sampled source together. The primary scopes
are all primary rows, original blur/noise, Gaussian blur, Gaussian noise,
interference, thin lines, arc strokes, primary geometric and the three primary
geometric severities. Clean reference and the 0.5 px elastic rung are
diagnostic-only. The M7B data contain 60 held-out sources (20 per font), 1,260
rows (1,140 primary and 120 diagnostic) and 1,260 distinct images. The familiar
M5C comparison uses the frozen selected `expanded_random_forest` predictions on
the M5C test slices (12 familiar sources); it is not matched to M7B.

**Metrics and residual convention.** M7C reproduces MAE, RMSE, R², target
mean/SD/quantiles, mean and SD residual, row/source/image counts and
near-degeneracy flags from the frozen targets and predictions. The residual is
`ssim_target - prediction`; a positive mean residual means the target SSIM is
higher than the prediction (under-prediction of image quality). R² is
suppressed when target SD is below the predeclared 0.005 margin. The 0.5 px
diagnostic has target SD 0.003376, so its R² is suppressed; the clean target is
constant, so its R² is undefined. All-primary MAE is 0.063427, RMSE 0.092952
and R² 0.743537 over 1,140 rows and 60 sources.

**Per-font and family results.** All-primary RMSE is 0.085198 for Georgia,
0.076126 for Monaco and 0.113431 for Impact. Original blur/noise RMSE is
0.027892, 0.034646 and 0.160079, respectively, versus 0.023328 on the familiar
M5C benchmark. Interference RMSE is 0.123919 for Georgia, 0.107367 for Monaco
and 0.056379 for Impact, a per-font reversal rather than evidence that Impact is
generally better. Primary geometric RMSE is 0.054292, 0.052548 and 0.068389.
These are descriptive summaries of the observed sources; the font groups are
unpaired and the three typefaces are a purposive sample.

**Geometric severity.** The M7B severity labels 2, 3 and 4 map to 1, 2 and 3 px;
severity 1 is the 0.5 px diagnostic rung. The mapping is verified against the
frozen M5A manifest. All-font primary geometric RMSE is 0.053384 at 1 px,
0.025868 at 2 px and 0.082867 at 3 px; the max/min ratio is 3.20. The target
ordering is monotonic in aggregate and in 60/60 sources, but the model error is
not monotonic: the 2 px rung is the strongest and the 3 px rung the weakest.
The aggregate geometric RMSE of 0.058839 therefore conceals substantial
severity-specific differences.

**Source-level concentration.** For all-primary rows the top-1 source accounts
for 3.48% of squared error, the top-3 share is 9.91%, the HHI-based effective
source count is 48.41 of 60 and 19 sources account for 50% of squared error.
The predeclared concentration rule (top-3 share ≥ 0.25 and effective fraction
≤ 0.50) is not met. Error is therefore distributed broadly across the observed
sources rather than dominated by a small number; this is a descriptive influence
summary, not a source-exclusion or model-refit procedure.

**Uncertainty and paired contrasts.** Every scope and font uses 2,000
source-grouped percentile bootstrap replicates with seed `20261110` and
scope-specific derived seeds. The all-primary 95% intervals are [0.059244,
0.067633] for MAE and [0.087121, 0.098730] for RMSE over 60 groups. Paired
within-source contrasts keep the same source on both sides; font differences
and familiar-versus-held-out differences are unpaired because they have
different source groups. Key paired RMSE differences are Gaussian noise minus
Gaussian blur (positive by +0.021254 [0.013953, 0.027877]), thin lines minus
arc strokes (positive by +0.023165 [0.015263, 0.031223]), interference minus
original blur/noise (+0.004181 [-0.019616, 0.029927]), geometric primary minus
original blur/noise (-0.037084 [-0.052294, -0.020216]), 2 px minus 1 px
(-0.027517 [-0.033407, -0.021562]), 3 px minus 2 px (positive by +0.057000
[0.046016, 0.067258]) and 3 px minus 1 px (+0.029483 [0.018276, 0.041545]).
Intervals are descriptive uncertainty summaries; no
significance test is performed, and an interval that excludes zero is not
treated as proof of a causal effect.

**Font and benchmark comparisons.** All-primary RMSE differences are
Georgia-minus-Monaco +0.009072 [-0.000708, 0.018519], Georgia-minus-Impact
-0.028233 [-0.038275, -0.018025] and Monaco-minus-Impact -0.037305
[-0.046567, -0.027865]. The Georgia–Monaco interval includes zero, so the
evidence supports a per-font difference for at least one pair, not for every
pair. The familiar-versus-held-out comparison reproduces the frozen M5C and M7B
values: original blur/noise RMSE 0.095923 versus 0.023328, geometric primary
0.058839 versus 0.059006, clean 0.014155 versus 0.002275 and 0.5 px diagnostic
0.038513 versus 0.035917. The benchmarks differ in fonts, source composition
and counts, and partially in corruption mix and target distributions, so these
are unmatched descriptive comparisons, not causal effects or equivalence tests.

**Claim audit.** Thirteen predeclared claims are classified as `supported`,
`suggestive` or `unsupported` against the frozen evidence. Three are supported:
C01 (per-font all-primary difference for at least one pair), C02 (blur/noise
RMSE above the familiar point estimate in all three held-out fonts) and C04
(aggregate geometric performance conceals severity-specific differences). None
are suggestive. Ten are unsupported, including arbitrary unseen fonts, unseen
renderers/resolutions/formats, human readability or solve rates, security or
production robustness, a causal font effect, geometric equivalence from close
aggregate RMSE, stable 0.5 px R² or ranking, and font-population generalisation.
The claim table records the evidence, limitations, rule and rule outcome for
every row.

**Protected-dependency contract (maintenance, 2026-10-21).** M7C originally
fingerprinted three *shared, growing* trees — `configs/`, `data/processed/` and
`models/` — together with the historical milestone report roots. That
directory-level fingerprint was too broad: it made M7C's verification depend on
the complete future contents of directories M7C never declared as inputs, so
adding an unrelated milestone configuration (the M7D planning config) failed
M7C verification even though all 31 of M7C's hash-pinned inputs, every
prediction, table, metric, protocol and report artifact were byte-identical.
The same over-broad rule also covered transient, Git-ignored scratch copies
(`data/processed/_m7b_probe*`, `data/processed/repro_*`) that are generated
outputs rather than frozen dependencies.

The contract is now scoped to an explicitly declared membership
(`PROTECTED_DEPENDENCY_MEMBERSHIP_VERSION = "m7c-scoped-dependencies-v1"`)
held in `cross_font_analysis.py`:

| Membership member | Scope | Semantics |
| --- | --- | --- |
| `config_files` | 5 files: M7C's own protocol plus the M7A, M5A, M7B and M5C protocols it depends on | Each file is hashed by path |
| `dataset_roots` | the M3, M3-pilot, M4A-pilot, M5A-pilot and M7B datasets | The set of non-image files under each root must match the recorded set exactly |
| `model_roots` | the nine frozen upstream model directories | Exact non-image file set per root |
| `report_roots` | the eighteen historical milestone report directories | Exact non-image file set per root |
| `ignored_roots` | the eight transient scratch directories listed explicitly | Excluded as generated outputs, so the exclusion is auditable rather than implicit |
| `output_files` | the fifteen artifacts M7C itself writes | Exact file set, per-file hashes and full recomputation from the pinned inputs |

The expected hashes are **never re-derived from the current tree**. They are the
hashes M7C recorded when it ran, restricted to the declared membership, so a
modified, deleted, missing or substituted declared dependency is still
detected. A file added later anywhere else in the repository — another
milestone's configuration, a new dataset root, a new model directory — is not
an M7C dependency and no longer invalidates M7C.

The integrity guarantees that matter are unchanged. The result-affecting
dependencies are M7C's 31 hash-pinned inputs, which live in the Git-tracked
protocol file and are re-verified byte-for-byte on every run; every table, the
protocol and the report are recomputed from those pins and compared with the
recorded artifacts; the recorded per-file output hashes, the config SHA-256 and
the config identity hash are still checked; and the recorded asset snapshot is
still checked for internal consistency, so editing the manifest fails closed.
The verifier now additionally checks the exact set of M7C output files, which
previously left `run_metadata.json` outside the per-file hash check.

This is a maintenance change to verifier logic only. It does not alter M7C's
metrics, predictions, datasets, protocol, report, tables, claim audit or any
scientific conclusion, and it does not rewrite or regenerate any historical
artifact. `EXP-0016` and `EXP-0017` remain valid as recorded. The regression
tests that protect the new behaviour are in
`tests/test_cross_font_analysis_dependencies.py`: verification passes with the
declared dependencies intact; an unrelated configuration file does not
invalidate M7C; a modified pinned input, a modified pinned report input, a
deleted pinned input, a deleted declared-but-not-pinned dependency, an altered
M7C output artifact and an unexpected output artifact all fail verification; a
missing, malformed or tampered dependency manifest fails closed; verification is
deterministic across repeated runs; and M7D's configuration, hashes and
verifier are unaffected. The tests use isolated temporary shadow trees and
never mutate the real frozen repository artifacts.

**Verification and reproducibility.** `captcha-cross-font-analysis run` writes
only to `reports/cross_font_analysis_v0_1_0/`. `verify` reloads the pinned
inputs, recomputes every deterministic table and the report, compares the
protocol and claim table, checks recorded output hashes, rechecks the input
hashes and the scoped protected-dependency membership, and fails on any
mismatch. Two runs from the same clean commit are byte-identical except for the
runtime-only `elapsed_seconds` metadata field. The committed inputs are the
TOML configuration, code, tests and documentation.

### 5.22 M7D evaluation design and dataset-expansion plan

M7D is a **read-only planning milestone**. It selects the next benchmark, fixes
its protocol, sample-size rationale, leakage controls, endpoints, success
criteria and stopping rules, and then stops. It does not generate a dataset,
render an image, train, refit, recalibrate, select or tune a model, change a
prediction, change a target or split, or modify any frozen benchmark artifact.
Everything it proposes is *planned, not completed*.

**Pinned inputs and isolation.** `configs/evaluation_design_plan_v0_1_0.toml`
pins 38 upstream files by SHA-256: the M3 dataset configuration, summary,
source table and manifest; the M5C configuration, champion model card, champion
artifact, test predictions, test-slice metrics, run metadata and report; the
M7A configuration, next-direction comparison, claim evidence and run metadata;
the M7B configuration, dataset summary, source table, manifest, predictions,
overall/per-font/family metrics, grouped uncertainty, familiar comparison, run
metadata and report; and the M7C configuration, per-font/family/severity/source
metrics, error concentration, grouped uncertainty, benchmark comparison, claim
evidence, run metadata and report. Every path is read-only. The plan refuses to
write outside the Git-ignored `reports/` tree or into any protected prior
report directory, and fingerprinted 376 protected non-image files under
`configs/`, `data/processed/`, `models/` and the prior report directories before
and after the run. That run-time guard has since been scoped to the declared
protected-dependency membership described in the *Protected-dependency
contract* paragraph of this section; the recorded 376-file snapshot is
unchanged.

**Evidence reproduction.** 46 numerical claims are recomputed from the pinned
frames rather than copied from prior prose: M3 and M7B source/row/split counts,
M7B and M7C all-primary MAE/RMSE/R², per-font all-primary and blur/noise RMSE,
geometric severity RMSE at 1/2/3 px, source-level error concentration
(top-3 share 0.099141, effective source count 48.409), the M7C
familiar-versus-held-out differences (+0.072595 blur/noise, -0.000167
geometric), the M5C familiar-font champion metrics, the champion artifact and
card hashes, and the M7A/M7C claim classifications (3/2/7 and 3/0/10). All 46
reproduce.

**Direction comparison and decision.** Three directions are compared. **A** is a
broader unseen-font benchmark: it expands the font-identity axis, targets the
M7C unsupported claims about arbitrary unseen fonts and font-population
generalisation, but is only partially resolvable because the roster remains
purposive. **B** is an unseen-renderer benchmark: it addresses the unsupported
unseen-renderer claim, but confounds renderer with antialiasing, glyph
placement, canvas and format, and two renderers are not a renderer sample; it
is **deferred** to a separate versioned protocol. **C** is a matched
font-transfer benchmark: it has the highest internal validity and directly
repairs the unmatched familiar-versus-held-out comparison, but reuses the same
three held-out fonts, so it cannot broaden the font population on its own. The
decision is **A**, with C's matched-case and matched-corruption logic
**incorporated into A as a design control** rather than discarded.

**Planned dataset expansion.** The planned benchmark has 10 typefaces (3
familiar: Arial, Courier New, Times New Roman; 7 held-out: Georgia, Monaco,
Impact, Verdana, Trebuchet MS, Chalkduster, Luminari), 30 matched text cases
sampled once from the frozen charset and length distribution (4/5/6 characters,
weights 0.30/0.40/0.30) and rendered in every font, 300 font-specific sources,
6,300 rows, 5,700 primary rows and 600 diagnostic rows. Georgia, Monaco and
Impact were already used in M7B as an *evaluation* set — the champion was never
fit, calibrated or selected on them — so they remain valid held-out
identities and keep three of the seven directly comparable with M7B, but only
**four** of the seven held-out faces (Verdana, Trebuchet MS, Chalkduster,
Luminari) are genuinely new. The roster therefore widens the held-out set
rather than replacing it, and the M7B three-font pattern is *not* independently
re-tested by this design; the per-font `m7b_status` column in
`font_inventory.csv` records the distinction for every face. The renderer is
the frozen Pillow ImageDraw + FreeType path at 192×64, 8-bit grayscale PNG. The
corruption protocol is the frozen M7B ladder: Gaussian blur 1/2/3.5/5 px,
Gaussian noise 6/12/20/32 sigma, thin lines 1/2/4/7, arc strokes 1/2/4/7 and
elastic warp 1/2/3 px, with the clean reference and the 0.5 px elastic rung
diagnostic only. Every font file is verified by path, SHA-256 and full charset
glyph coverage; licensing is **not** asserted, and the recorded status
`review_required_before_generation` blocks generation until a local-use review
exists. A missing font, a changed hash, lost glyph coverage or an unresolved
license stops the run: no substitute font may be selected after outcomes are
seen.

**Independent units.** The plan separates nine units that are often conflated:
*font identity* (one typeface file in the purposive roster), *matched text case*
(one string rendered in every font with the same render and corruption seed
schedule), *source* (one clean render of one case in one font), *clean
reference* (the exact PNG anchor for one source), *corrupted variant* (one
family/severity applied to one source, nested within it), *renderer* (one fixed
condition, not a sampled factor), *benchmark row* (one manifest row), *font
pair* (one held-out/familiar pair on the same cases) and *source group* (all
rows of one font-specific source, the bootstrap resampling unit). Font-level
inference is descriptive: source grouping alone cannot establish
font-population generalisation, because it resamples cases within the observed
fonts rather than new fonts.

**Split hierarchy and leakage controls.** The proposed benchmark is eval-only:
all 300 sources sit in one untouched `eval` split and no model is fit. Held-out
font identities never appear in training, validation, feature selection or
calibration. All font renders of a case stay together if a case-level partition
is ever needed; all variants of a source stay together in every split, metric
and bootstrap replicate. Any future authorized training experiment must
reserve at least two held-out font identities and disjoint text cases for a
final test set that is never used for fitting, calibration or selection. Twelve
declarative controls (L01–L12) cover font-level holdout, frozen model identity,
no test-label fitting, matched-case grouping, source grouping, no row-level
independence, future-training separation, manifest/split hashes, no font
substitution, target recomputation, output protection, and no post-hoc
endpoint search.

**Pre-registered endpoints.** The primary endpoint P1 is the matched
held-out-minus-familiar all-primary RMSE contrast, averaged over the 21
held-out/familiar font pairs, with a matched-case grouped percentile bootstrap
(2,000 replicates, 95%). P1d is the supporting per-font profile with
leave-one-font-out range. Secondary endpoints cover original blur/noise, each
predeclared corruption family, geometric primary and its 1/2/3 px rungs, paired
within-source contrasts, font-level range/SD and source concentration.
Diagnostics cover the clean reference, the 0.5 px rung and target distributions.
R² is suppressed when the target SD is below the predeclared 0.005 margin or
the target is constant; the clean and 0.5 px slices cannot support primary
ranking claims. Endpoint F1 (a predeclared future improvement margin) is
explicitly **not authorized** by this milestone.

**Sample size and precision.** The planning source-level SD is 0.022634, the
observed M7C all-primary per-source RMSE SD over 60 sources. With 30 matched
cases the normal-approximation SE for a mean source RMSE is 0.004132 and the
95% half-width is 0.008099. Because no matched cross-font data exist yet, the
paired-difference SD is *not* estimated: the plan reports sensitivity across
assumed SDs of 0.005, 0.010 and 0.020, giving half-widths of 0.001789,
0.003578 and 0.007157 at 30 cases. Font-level precision is shown for
illustrative between-font SDs of 0.010, 0.020 and 0.030 at 10 fonts (half-widths
0.006198, 0.012396, 0.018594), with the explicit note that a normal multiplier
understates small-df t uncertainty and that the three observed M7B fonts cannot
supply a font-population variance. These are **precision scenarios, not power
calculations**; no effect size is fabricated, and the sample is not extended
after results are seen.

**Frozen model and baseline policy.** The baseline is exactly the M5C
`expanded_random_forest` champion with calibration `none`, artifact SHA-256
`c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0` and card
SHA-256 `9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8`.
M7D does not load it. The future evaluator may load the serialized estimator and
predict; it must not call `fit`, `partial_fit`, a calibrator or a selector, and
must not use held-out labels to choose features, thresholds, endpoints or a
model. Any future model comparison requires a separate authorized experiment
with untouched test groups and a predeclared practical margin. M7D defines **no
universal SSIM improvement threshold**.

**Success criteria, stopping rules and interpretation.** Fourteen criteria
(G01–G14) gate the work: input integrity, font availability and hash/glyph
verification, licensing review, exact dataset counts, matched coverage,
font-level and source-level leakage isolation, byte-reproducible generation with
target replay, non-degenerate primary targets, frozen-model identity, complete
pre-registered reporting, reproducible artifacts, disciplined interpretation of
null/mixed/negative findings, and a predeclared margin for any future
improvement claim. Eight stopping rules (R01–R08) stop on data or protocol
failure rather than on whether results look favourable. A null contrast means
no detected matched difference on this roster and these corruptions; it is
**not** equivalence, universal robustness or font-population homogeneity,
because no equivalence margin was predeclared. A mixed result is a valid
conditional finding, but the plan reports family and severity slices separately
and does not average them into a universal ranking. A negative result is a
conditional descriptive result and does not establish CAPTCHA insecurity or a
causal font effect. A positive result is still conditional on the ten-font
roster, the fixed renderer, the synthetic target and the observed cases.

**Claim limits.** Twelve claims are recorded with an explicit status. Six
limitations are structural: ten typefaces are a purposive roster rather than a
font-population sample; three of the seven held-out faces were already used in
M7B as an evaluation set, so only four held-out identities are genuinely new
and the M7B three-font pattern is not independently re-tested; the roster
deliberately includes two display/script stress faces and is not representative
of production CAPTCHA fonts; 30 matched cases limit source-level precision and
the paired SD is unknown before generation; one renderer, one canvas and one
format are fixed; and SSIM against a synthetic clean reference is not human
readability, solve rate, security or production robustness. Matching reduces
but does not eliminate composition differences, because glyph metrics and ink
distribution differ by font.

**Protected-dependency contract (maintenance, 2026-10-21).** M7D originally
fingerprinted three *shared, growing* trees — `configs/`, `data/processed/` and
`models/` — together with the historical milestone report roots, and compared
that whole-tree fingerprint to the snapshot recorded when M7D ran. That
directory-level fingerprint was too broad: it made M7D verification depend on
the complete future contents of directories M7D never declared as inputs, so
adding an unrelated milestone configuration would fail M7D verification even
though all 38 of M7D's hash-pinned inputs and every plan table, manifest and
report artifact were byte-identical. The same rule also covered transient,
Git-ignored scratch copies that are generated outputs rather than frozen
dependencies.

M7D's dependencies are a *file* set rather than a tree: every upstream
dependency is individually hash-pinned in the committed M7D planning
configuration, and that configuration's own SHA-256 is recorded. The contract
is therefore scoped at file level
(`M7D_PROTECTED_DEPENDENCY_MEMBERSHIP_VERSION = "m7d-scoped-dependencies-v1"`),
which is the role-appropriate granularity for this milestone:

| Membership member | Scope | Semantics |
| --- | --- | --- |
| `config_files` | the M7D planning configuration `configs/evaluation_design_plan_v0_1_0.toml` | Must be present and match the SHA-256 recorded when M7D ran; the existing `config_sha256` and `config_hash` checks are unchanged and already verify this exact file rather than the `configs/` directory |
| `input_files` | the 38 upstream evidence files M7D pins: the M3, M5C, M7A, M7B and M7C protocols, dataset tables, predictions, metrics, evidence tables, metadata and reports | Each must be present and match its configuration-pinned SHA-256 |
| `output_files` | the 17 artifacts M7D itself writes | The file set must be exact; the sixteen non-manifest artifacts must match their recorded hashes; every table, the plan manifest and the report are recomputed from the pinned inputs and compared |
| `ignored_paths` | the eight transient scratch directories and the two *planned* future benchmark output paths (`data/processed/matched_font_benchmark_v0.1.0`, `reports/matched_font_benchmark_v0_1_0`) | Not M7D dependencies; not required to exist; excluded so the exclusion is auditable rather than implicit |

The expected hashes are **never regenerated from the current tree**. They come
from two independent anchors that must agree: the committed M7D planning
configuration (the 38 configuration-pinned input hashes) and the M7D run-time
snapshot recorded in `run_metadata.json` (the planning configuration's own
recorded hash plus the same 38 input hashes). Three checks enforce this:
`frozen_dependency_manifest` verifies that the recorded snapshot is internally
consistent with its own recorded hash, `frozen_dependency_anchors` verifies
that the recorded snapshot and the committed configuration agree on all 39
declared dependencies, and `frozen_dependencies` verifies that every declared
dependency is present, is a regular file, and hashes to its expected value.
A missing, malformed or tampered manifest therefore fails closed, and a
manifest rewritten to be self-consistent still disagrees with the committed
configuration. The new `output_file_set` check additionally requires the plan
directory to contain exactly the declared artifacts, which previously left
`run_metadata.json` outside the per-file hash check.

The integrity guarantees are preserved. The result-affecting dependencies are
the 38 configuration-pinned inputs, which live in a Git-tracked file and are
re-verified byte-for-byte on every run; every table, the plan manifest and the
report are recomputed from those pins and compared with the recorded
artifacts; the recorded per-file output hashes, the configuration SHA-256 and
the configuration identity hash are all still checked; and the plan itself is
protected by the Git history of its configuration. Files that M7D never read —
other milestones' model bundles and report outputs, the thirteen other
configuration files and the transient scratch copies — are no longer part of
M7D's own fingerprint; they remain protected by their own milestone verifiers
and by M7C's scoped membership, which is unchanged.

This is a maintenance change to verifier logic only. It does not alter the M7D
research plan, the proposed dataset size, the font roster, the split design,
the sample-size rationale, the chosen direction, the success criteria, the
recorded tables, the plan manifest, the report or any scientific conclusion,
and it does not rewrite or regenerate any historical artifact. The regression
tests that protect the new behaviour are in
`tests/test_evaluation_plan_dependencies.py`: verification passes with the
declared dependencies intact; an unrelated configuration, dataset root, model
directory or report directory does not invalidate M7D; a modified or deleted
planning configuration, a modified or deleted declared upstream dependency and
a modified or deleted protected output all fail verification; an unexpected
protected output fails the exact-membership check; a missing, malformed or
tampered dependency manifest fails closed, including a manifest rewritten to be
self-consistent; the M7C scoped verifier is unaffected and still detects a
modified dependency; and repeated M7D verification is deterministic. The tests
use isolated temporary shadow trees and never mutate the real historical
artifacts.

One forward-compatibility caveat is unchanged by this maintenance work and is
recorded here for the next milestone: the M7D isolation check
`no_planned_outputs_created` asserts that the *planned* future benchmark paths
`data/processed/matched_font_benchmark_v0.1.0` and
`reports/matched_font_benchmark_v0_1_0` do not yet exist. That check is part of
the recorded plan artifact, so it is deliberately left untouched; it will need
its own versioned treatment when the next benchmark is actually built.

**Verification and reproducibility.** `captcha-evaluation-plan run` writes
only to `reports/evaluation_design_plan_v0_1_0/`; `verify` reloads the pinned
inputs, recomputes every deterministic table, the protocol manifest and the
report, compares them byte-for-byte, and rechecks the input hashes, the output
hashes and the scoped protected-dependency membership. Two runs from the same
clean commit are byte-identical except for the runtime-only metadata field. The
committed inputs are the TOML configuration, code, tests and documentation.

**Frozen-verifier note.** The M7C `verify` command originally fingerprinted the
*entire* `configs/` tree, so adding the M7D configuration — a new, separately
versioned planning file, not a modification of any frozen artifact — changed
that tree-level fingerprint. After M7D, `captcha-cross-font-analysis verify`
reported exactly one difference: the added file
`configs/evaluation_design_plan_v0_1_0.toml`. Every one of the 360 files
recorded in the M7C run was byte-identical, all 31 M7C pinned inputs still
matched their recorded SHA-256 values, and all M7C output artifacts were
unchanged. M7C code and artifacts were deliberately **not** modified to silence
the check, and M7D re-verifies the M7C configuration, tables, metadata and
report directly by content. The underlying verifier contract has since been
repaired by a maintenance change that scopes M7C's protected-dependency
fingerprint to the dependencies M7C actually declares; see
[METHODOLOGY.md](METHODOLOGY.md) §5.21. That change alters verifier logic
only: it does not alter M7C's metrics, predictions, datasets, protocol, report,
claim audit or any scientific conclusion, and it does not rewrite or regenerate
any historical artifact.

### 5.23 M7E font-roster licensing and availability audit

M7E is a **read-only audit** of the ten typefaces the M7D plan proposes for the
planned matched cross-font benchmark. It re-verifies each face against its
pinned path and SHA-256, records the licence evidence for that specific face,
and re-derives an eligibility decision from that evidence. It generates no
images, no text cases, no dataset rows, no predictions and no model artifacts.
It does not change the M7D plan, the M5C champion or any historical output, and
it does not implement the transition of the planned benchmark paths into actual
outputs.

**Scope frozen.** The planned benchmark design is unchanged: ten typefaces
(three familiar, seven held-out), 30 matched text cases, 300 sources, 6,300
rows, 5,700 primary rows, 600 diagnostic rows, the M7B corruption ladder, the
frozen renderer at 192×64 8-bit grayscale PNG, and zero-shot evaluation of the
frozen M5C champion. The intended use under audit is to render synthetic text
strings into PNG images, distribute the rendered images and derived tables as
a Git-ignored dataset, and evaluate the frozen champion. The target boundary
is unchanged: synthetic SSIM against an exact source-specific clean reference,
not human readability, CAPTCHA solvability, security or production robustness.

**Protocol and inputs.** `configs/font_roster_audit_v0_1_0.toml` is the
versioned protocol; `configs/font_roster_audit_v0_2_0.toml` is its
superseding version and is described in the *Protocol version 0.2.0* paragraph
below. It pins three upstream files by SHA-256 — the M7D planning
configuration, the M7D machine-readable plan manifest and the M7D font
inventory — and it keeps *evidence facts* and *eligibility decisions* in two
separate tables. The evidence table records, for every face, the planned
family, the exact face and style, the version string, the PostScript name, the
copyright and rights holder, the OS/2 embedding bits, the availability source
and scope, the licence name and scope, a source URL, a retrieval date, a
verbatim evidence excerpt, and explicit answers to four questions: is rendering
permitted, is redistribution of the rendered output permitted, is
redistribution of the font file permitted, and does an attribution or notice
condition apply. The decision table records the status, the reason, the
conditions and the open questions. Evidence is never silently merged with the
decision.

**Statuses and derivation.** Four statuses are used, with the exact meanings
recorded in the protocol. `eligible` means the exact face was verified present
with the pinned SHA-256, the recorded licence evidence supports rendering the
intended output and permits distributing that output, no attribution or notice
condition is unresolved, the font file is not redistributed, and the face and
version are pinned reproducibly. `ineligible` means the evidence shows a
positive prohibition. `review_required` means the face is available but the
licence position for the intended use is not fully established, or a condition
such as attribution, a purchase requirement or a notice obligation remains
unresolved. `unavailable` means the face could not be obtained or verified.

The derivation is re-run on every verification. The declared status may be
*more* restrictive than the derived status, but a declared `eligible` is
rejected unless the derived status is also `eligible`; a declared `ineligible`
requires a positive prohibition in the evidence; a declared `unavailable`
requires the face to be unobtainable. Expected hashes are anchored in the
committed protocol and the pinned M7D inputs, and are never regenerated from
the current tree.

**Evidence standard.** Authoritative sources were used wherever the terms
could be checked: the macOS 27 Software License Agreement (EA2005) section 2E
and its important note; the Microsoft Typography *Font redistribution FAQ*; the
Monotype *Font Software For Desktop End User License Agreement*; the *CANADA
TYPE EULA* v4.193; Apple's published system-font listing; and the licence
strings, copyright notices, version strings and OS/2 embedding bits embedded
in the font files themselves. Search results and third-party summaries were
not accepted as sufficient evidence where the original terms were available.
Where a source could not be accessed or the terms remain ambiguous, the
uncertainty is recorded and the face is marked `review_required`; permission is
not inferred.

**The font-file boundary.** No licence in the roster permits redistributing a
font file, and the planned workflow does not need to: the dataset manifest
records the font identifier and the font SHA-256, never the font file; the
repository ignores `data/`, `models/` and `reports/`; and the audit verifies
at run time that no font binary exists anywhere in the working tree. The
audit therefore distinguishes the one redistribution the workflow performs —
the rendered images — from the one it never performs.

**Result of protocol version 0.1.0.** Nine of the ten faces were `eligible`
for the intended workflow and one, Luminari, was `review_required`. The Microsoft designs (Georgia, Verdana,
Trebuchet MS) and the Monotype designs (Arial, Courier New, Times New Roman,
Impact) are supported by a foundry source that expressly permits distributing
materials made with the font, subject to two conditions the benchmark
satisfies: the material must not embed the font, and a static graphic image
must not correspond to individually addressed glyphs. The Apple-published
designs (Monaco, Chalkduster) are supported by the product licence, which
grants use to display and print content and extends to publishing and
distributing materials the licensee is authorised to distribute; the evidence
grade records that this rests on the product licence rather than on an express
clause naming redistribution of rendered images. Luminari is unresolved: the
macOS product licence grants the use that creates the output but is silent on
redistributing it, while the Canada Type terms that do cover the rendered
output permit static images in png, jpeg and gif formats but only take effect
once Canada Type has received full payment, and they require the licensee to
inform anyone with access of the terms. The face is retained in the roster
with `review_required`; it is not removed and no similarly named family is
substituted.

**Availability and reproducibility.** Every face was verified present on the
audit machine with the M7D-pinned SHA-256, and the family, style, version
string, PostScript name and embedding bits were read back from the font file
itself. Availability is nonetheless Apple-platform scoped: these are the
builds Apple ships with its operating systems, and the commonly used
alternative distribution of the core fonts on Linux ships different versions
under the older Core Fonts for the Web end-user licence, so the pinned hashes
do not transfer and a cross-platform re-pin would need a fresh licence review.

**Project-level findings.** No proposed face is released under an open
licence, so the benchmark is reproducible from a pinned Apple environment but
not from freely redistributable inputs. Separately, Monotype's direct licence
contains a restriction on using the font software, or any rendered glyphs,
outlines, kerning data or other visual elements, in connection with developing,
training or fine-tuning artificial-intelligence models. The macOS-bundled
copies audited here are governed by the macOS Software License Agreement,
which contains no such clause, and the planned benchmark is evaluation-only.
The finding is nevertheless material, because the frozen M3 development data
was rendered with three of these faces and the M5C champion was trained on it:
any future move to a directly licensed Monotype copy, or any training on
font-rendered data under such a licence, would need a fresh licence review
first. Finally, Georgia, Monaco and Impact were already evaluated in M7B. They
remain valid held-out identities because the champion was never fitted,
calibrated or selected on them, but they are not fonts that are new to the
project evaluation, and the roster coverage table keeps that distinction
visible so that a future result on them is never described as an independent
replication of the M7B three-font finding.

**Gates under protocol version 0.1.0.** The audit gate was *partially
passed*: nine faces were eligible, one was review_required, none was ineligible
or unavailable. The **benchmark generation gate was closed**: every proposed
face had to carry a defensible eligibility decision before the benchmark could
be generated, and clearing Luminari — by a written determination or by a
purchased foundry licence — or a deliberately separately versioned roster
decision was a prerequisite. M7D's `no_planned_outputs_created` check is
preserved and still passes; the audit creates nothing at the planned benchmark
output paths. Protocol version 0.2.0 clears Luminari and opens the gate; see
below.

**Protocol version 0.2.0 — Luminari licence resolution (2026-10-21).** The
M7E protocol has two versions. Version 0.1.0 is preserved unchanged as the
historical record and remains independently verifiable; version 0.2.0
(`configs/font_roster_audit_v0_2_0.toml`) supersedes it for **one face only,
Luminari**, and for the benchmark generation gate. The roster, the ten
typefaces, the 30 matched text cases, the 300 sources, the 6,300 rows, the
5,700 primary rows, the 600 diagnostic rows, the corruption ladder, the
renderer, the evidence standard and the four statuses are unchanged, and the
M7D plan is untouched. The version 0.2.0 output directory is
`reports/font_roster_audit_v0_2_0/`; the version 0.1.0 report carries a
supersession banner so a reader of the older record is told which audit is
current.

**What was wrong with the version 0.1.0 reasoning.** Version 0.1.0 applied
the Canada Type end-user licence to the macOS-bundled copy of Luminari and
therefore read that licence's section 2 payment trigger as if it governed the
installed font. It also quoted sections 3 and 8 of that licence but not
section 11, which is the provision that settles the question. The contractual
connection runs the other way round:

- The copy on the audit machine is part of the *Apple Software*. Section 1.A of
  the macOS 27 Software License Agreement (EA2005, 07/11/2026) defines the
  Apple Software to include "fonts ... accompanying this License", including
  third-party fonts.
- The licence that governs the *user's* use of that copy is the macOS
  Software License Agreement. Section 2E grants the use of the bundled fonts
  "to display and print content while running the Apple Software", with
  embedding governed by the font's own embedding restrictions, which this
  workflow does not exercise. Section 1's important note extends the licence to
  the "publication and distribution" of "materials you are authorized or legally
  permitted to reproduce, modify, publish or distribute".
- The Canada Type end-user licence v4.193 governs a **direct download from
  Canada Type**, not the macOS-bundled copy. Its section 2 payment trigger
  therefore does not attach to the installed font. It is recorded as evidence of
  the rights holder's position on the design, and that position is consistent
  with the product licence.

**The decisive provisions.** With the full text of the Canada Type EULA
v4.193 in hand, the intended use is established rather than inferred:

- *Section 3 (Desktop & Print)* defines an output device to include "any other
  machine that uses Canada Type font data to output forms rasterized and
  generated from the computer where the font data is installed", and lists as
  permitted usage "to create and print documents or make static images in
  raster digital formats, such as png, jpeg, gif". The benchmark's rendering
  step is a software rasteriser on the machine where the font data is
  installed, producing PNG images, so it falls inside the rights holder's own
  definition of a licensed use and output format.
- *Section 7* confirms the intended use is not one of the separately priced
  extensions (broadcast, HTML5 banner ads, desktop application embedding).
- *Section 11* restricts the distribution of the **font software**; *section 8*
  and *section 12* restrict copies, back-ups and modification of the
  software. None of them restricts the rendered output. *Section 14* imposes a
  duty to inform persons "having access to the Canada Type Software and copies
  thereof", which is not engaged by distributing rasterised images that contain
  no font software.
- *Section 1* of the macOS SLA together with Apple's published position that
  content made with bundled fonts "may be used for either personal or
  commercial purposes", with only the font files restricted, corroborate the
  same conclusion.

**Scope of the permission and binding conditions.** The permission is: render
synthetic text with the pinned face on the pinned Apple environment, retain the
rendered images, and distribute those images and the derived tables. The font
binary may not be redistributed, embedded, modified or converted. Four
conditions are recorded on the decision and are binding: (1) generate only on
the pinned Apple environment with the face identified by the M7D SHA-256 and
re-verify that hash before every run; (2) never redistribute, embed, modify or
convert the font file and keep the working tree free of font binaries, which
the audit checks on every run; (3) use the face for **evaluation only** and do
not train or fine-tune any model on font-rendered data that includes this face
without a fresh licence review, because the authorised distributor's summary of
the same desktop licence lists an artificial-intelligence training restriction
that the rights holder's own published end-user licence does not contain; and
(4) if the benchmark is ever moved off the macOS product licence — to a
directly licensed or a Linux-distributed build of Luminari — a fresh licence
review is required first.

**Residual uncertainties.** Three are recorded rather than resolved: the
licence Apple holds from Canada Type is not published, so the chain from the
rights holder to the grant Apple makes cannot be inspected; the permission to
redistribute the rendered output follows from the product licence's important
note and Apple's published position rather than from an express clause, and
Apple's published position is a 2017 community-thread quotation of an Apple
email rather than a published Apple policy document; and the rights holder's
published text permits making static png, jpeg and gif images without using
the word "distribute" in relation to that output, although the authorised
distributor's summary of the same desktop licence states that it allows the
licensee to create, print and share flat, non-editable designs. The decision
is an interpretation of documentary terms and not a legal opinion.

**Result.** Luminari is **eligible** for the intended evaluation-only
workflow, with the four conditions above. All ten faces are eligible, none is
review_required, ineligible or unavailable, the audit gate is **passed** and
the **benchmark generation gate is open**. No roster change is proposed: the
ten-font design, the matched text cases, the row counts, the split structure
and the generalization claims are exactly as the M7D plan records them, and no
manifest, hash or verification contract outside the M7E protocol needs a
versioned update. The M7D `no_planned_outputs_created` check is preserved and
still passes, and this milestone creates nothing at the planned benchmark
output paths.

**Verification and reproducibility.** `captcha-font-roster-audit run` writes
only to the protocol's own `reports/font_roster_audit_v0_2_0/` directory
(`reports/font_roster_audit_v0_1_0/` for the superseded version); `verify`
reloads the pinned inputs, recomputes every deterministic table, the audit
manifest and the report, compares them byte-for-byte, and rechecks the declared
dependencies, the font hashes and the exact output file set. The dependency
fingerprint is file-exact over the three declared inputs, so an unrelated
configuration, dataset, model or report added to a shared directory later does
not invalidate the audit, while a modified or deleted declared input still
fails. Two runs from the same clean commit are byte-identical except for the
runtime-only metadata field. The regression tests in
`tests/test_font_roster_audit.py` cover a valid evidence record and decision,
missing licence evidence, an unsupported or malformed licence status, an
`eligible` decision without adequate evidence, a missing or duplicated roster
member, modified and deleted declared evidence, modified and unexpected
protected outputs, a missing, malformed or tampered dependency manifest,
unrelated additions to shared directories, deterministic output and
verification, and the continued tamper detection of the M7C and M7D verifiers.
The follow-up tests in `tests/test_font_roster_audit_v2.py` cover the resolved
decision and its recorded evidence and scope, the rejection of an ambiguous
`eligible` decision, the unchanged roster membership and provenance between
protocol versions, the fact that only the Luminari record changed, the
preservation of the superseded version's own conclusion and its continued
verifiability, the historical immutability of the M7D artifacts and the
superseded protocol, the unchanged absence of the planned benchmark output
paths, the continued pass and tamper detection of the M7C and M7D verifiers,
and deterministic verification of the current protocol.

## 6. Data leakage risks and controls

| Risk | Why it matters | Control |
| --- | --- | --- |
| Same text/seed in train and test | Near-duplicate images inflate scores | Group splits by generation seed/text; verify overlap is zero |
| Augmented variants crossing splits | Model memorises source images | Assign splits before augmentation; group by source |
| Corruption parameters as features | Directly encode the target's cause | Exclude severity/family/params from primary regression inputs |
| Fitting scalers or imputers on all data | Test statistics leak into training | Fit preprocessing inside CV folds / train only |
| Tuning on the test set | Optimistic bias | Fixed validation strategy; test used once per experiment |
| Selecting target definition after seeing test scores | Target leakage through iteration | Freeze target in M1; record any change and re-run everything |
| Duplicate images from different configs | Hidden overlap | Hash images and manifest records; report duplicates |
| Clean references duplicated across datasets | Double-counting or accidental clean leakage | Exact sample/image hash comparison; exclude M4A clean rows from expanded fitting and report the clean test slice separately |
| Font memorisation | Model learns font identity | M3 held-out-font stress test; M7B evaluates the frozen champion on a separate Georgia/Monaco/Impact dataset whose fonts and source IDs are disjoint from M3 |
| Shared rendering condition across splits | A within-condition score is misread as condition generalisation | M7A audits font/renderer/resolution/format coverage; M7B supplies one held-out-font probe but does not test unseen renderers, resolutions or formats |
| Metric code drift | Scores not comparable across runs | Pin metric implementation, log version/config hash |

## 7. Evaluation strategy

### 7.1 Splits and resampling

- Frozen train/validation/test split, grouped by clean source (`source_id`).
- M3 uses the fixed validation split for model selection; it does **not** run
  grouped k-fold on the training portion. With 36 training source groups, a
  fixed validation split is simpler and keeps the selection protocol identical
  to the M2 diagnostic. The limitation is a noisier selection signal.
- Bootstrap intervals resample held-out source groups, not variants, for
  headline metrics.
- Report the number of independent groups, not just the number of images. M3 and
  M4B test intervals use 12 groups; the M3 held-out-font stress test uses 20;
  M7B uses 60 groups (20 per held-out font).
- The M3 held-out-font experiment removes one font from fitting and validation
  before selection. It is a stress test, not a replacement for the primary
  split.
- M7B is a separate, frozen-model evaluation dataset. It has no training split,
  no validation selection and no calibration; its 60 held-out-font sources form
  one untouched `eval` split, and every interval resamples whole source groups.
- M7C reuses the M7B 60 source groups and the M5C 12 familiar source groups. Its
  paired contrasts resample whole sources on both sides; font differences and
  familiar-versus-held-out differences are unpaired and descriptive. The three
  held-out fonts have 20 source groups each and are not a font population.

### 7.2 Regression metrics

- MAE and RMSE in SSIM units (primary).
- R² for explained variance, with `nan`/`r2_defined = false` when the target is
  constant; never zero-filled.
- Sample and source-group counts on every row.
- Spearman rank correlation for monotonic association in diagnostics.
- Residual analysis by corruption family, severity, coverage and font; no
  calibration plot is produced in M3, only tables.
- Training, validation and test metrics are reported together so overfitting is
  visible; test comparisons are labelled as held-out.

### 7.3 Classification metrics (secondary, not run in M3)

- Macro-F1 and balanced accuracy for corruption family.
- Confusion matrix with per-class support.
- Ordinal MAE and quadratic weighted kappa for severity.
- Explicit chance baselines.

M3 focuses on SSIM regression; corruption labels are used only as diagnostic
metadata, not as a classification target. These metrics remain planned.

### 7.4 Robustness analysis

- Performance per corruption family and severity level.
- Cross-family generalisation: train on some families, test on held-out ones.
- Worst-group and tail error reporting, not only aggregate means.
- Qualitative examples of large residuals, with generation metadata for context.

### 7.5 Reporting standards

Every experiment records: config hash, dataset version/hash, split hash, code
revision, package and scikit-learn versions, random seeds, metrics with grouped
intervals where available, and runtime. Negative and inconclusive results are
recorded with the same detail as positive ones. No result is reported without
its split definition. M3 additionally records the model-card hashes and a
machine-readable `run_metadata.json`; generated artifacts remain Git-ignored.

## 8. Threats to validity

- **Construct validity:** reference-based metrics are proxies, not ground-truth
  legibility. SSIM is a synthetic image-similarity target; it is not a
  readability or solvability label.
- **Internal validity:** leakage is controlled by source grouping, a frozen
  split, train-only preprocessing and explicit feature columns. Undiscovered
  near-duplicates remain possible; duplicate checks are mandatory.
- **Selection instability:** model selection uses 12 validation source groups.
  The top two random-forest candidates differ by less than 0.0001 validation
  RMSE, so the exact winner should not be treated as a stable ranking. The
  family-level comparison is more defensible than the individual forest
  configuration.
- **External validity:** synthetic renderings, three fonts, one canvas size and
  two corruption families do not represent production CAPTCHAs. The held-out
  font (Times New Roman) is a preliminary stress test, not broad generalisation.
- **Confounding:** severity + family metadata alone reaches test R² 0.911.
  Image features recover more (0.986), but the synthetic design makes the target
  easier than real-world quality assessment.
- **Conclusion validity:** small held-out group counts limit statistical power;
  bootstrap intervals are wide and unstable. No significance claim is made.
- **Target saturation:** clean samples have SSIM exactly 1 and a constant-target
  subgroup, so R² is undefined there and the pooled R² is inflated relative to
  corrupted-only performance. Corrupted-only and family slices are reported.
- **Persistence safety:** joblib/pickle artifacts are not safe for untrusted
  files. The verification step checks schema and hashes, but only locally
  generated artifacts should be loaded.
- **Reproducibility:** results depend on library versions and hardware; these are
  pinned and logged where they affect numerical output.

## 9. Ethical and legal considerations

- No live CAPTCHA systems are queried or bypassed.
- Only synthetic images and redistributable fonts are used.
- No human-subject data is collected, so no solvability labels exist.
- The project does not provide or evaluate solving tools.
