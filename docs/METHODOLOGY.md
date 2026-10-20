# Methodology

**Status:** Milestones 0–3, M4A and M4B complete. The target and dataset are
frozen, the conventional benchmark and grouped interference benchmark are
implemented, and the CNN comparison and dashboard remain planned.
**Last updated:** 2026-10-20

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
  while the M1 pilot allowed `min_required`.
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
each split contains all three fonts (12/4/4 sources per font).

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
| Font memorisation | Model learns font identity | Font-held-out evaluation slice |
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
  M4B test intervals use 12 groups; held-out-font intervals use 20.
- The held-out-font experiment removes one font from fitting and validation
  before selection. It is a stress test, not a replacement for the primary
  split.

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
