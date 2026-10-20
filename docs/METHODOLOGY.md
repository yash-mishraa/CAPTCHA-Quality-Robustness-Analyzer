# Methodology

**Status:** Milestones 0–2 complete. The target is validated for the pilot
conditions, and a reproducible dataset/feature pipeline exists. Modelling and
CNN comparison remain planned.
**Last updated:** 2026-10-20

## 1. Research framing

**Question.** Can image quality and corruption severity in a synthetic
CAPTCHA-style corpus be predicted from image content alone, and how does
predictive performance vary across corruption families?

**Primary task (planned).** Supervised regression of a defined reference-based
image-quality score.

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
- **Out of scope:** rotation, colour, texture backgrounds, occlusion.

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
| Contrast / brightness, geometric warp, occlusion, compression | — | deferred; not needed for the current question |

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

## 5. ML baselines


### 5.1 Feature-based regression

Inputs are the eight documented image-only features in §4.1: mean and standard
deviation of intensity, RMS contrast, Otsu ink coverage, Laplacian variance,
Canny edge density, histogram entropy, and Immerkær's noise estimate. Generation
metadata (severity, corruption family, seed) is **not** an input to the primary
regressor.

Models, in increasing complexity:

1. Dummy regressor (mean/median) as a floor.
2. Linear / ridge regression on standardised features. The M2 diagnostic already
   establishes this baseline (test R² 0.962), so M3 models must beat it.
3. Random forest or histogram gradient boosting.
4. Optional: gradient boosting with modest hyperparameter search.

Preprocessing (imputation, scaling) is fit inside each training fold only.

### 5.2 Corruption-type and severity baselines

- Multi-class classifier for corruption family.
- Ordinal or multi-class model for severity, evaluated with ordinal-aware
  metrics.
- These experiments test whether the corpus encodes the intended factors. They
  are **not** the primary quality model and must not be presented as evidence of
  perceptual understanding.

### 5.3 CNN comparison

- A small CNN trained from scratch (appropriate for a modest dataset).
- A pretrained backbone with a replaced regression head, using documented
  preprocessing and partial or full fine-tuning.
- Identical splits, target definition, and early-stopping protocol as the
  feature-based models.
- Report parameter count, training time, and hardware to make the cost/benefit
  comparison explicit.

The CNN is justified only if it improves held-out performance or reveals
structure the baselines miss. Otherwise the simpler model is retained.

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
| Font memorisation | Model learns font identity | Font-held-out evaluation slice |
| Metric code drift | Scores not comparable across runs | Pin metric implementation, log version/config hash |

## 7. Evaluation strategy

### 7.1 Splits and resampling

- Frozen train/validation/test split, grouped by clean source (`source_id`).
- Grouped k-fold cross-validation on the training portion for model selection.
- Bootstrap confidence intervals over held-out groups for headline metrics.
- Report the number of independent groups, not just the number of images.

### 7.2 Regression metrics

- MAE and RMSE (primary, in target units).
- R² for explained variance, interpreted cautiously on bounded targets.
- Spearman rank correlation for monotonic association.
- Residual analysis by corruption family, severity, font, and text length.
- Calibration-style plot of predicted vs. reference scores.

### 7.3 Classification metrics (secondary)

- Macro-F1 and balanced accuracy for corruption family.
- Confusion matrix with per-class support.
- Ordinal MAE and quadratic weighted kappa for severity.
- Explicit chance baselines.

### 7.4 Robustness analysis

- Performance per corruption family and severity level.
- Cross-family generalisation: train on some families, test on held-out ones.
- Worst-group and tail error reporting, not only aggregate means.
- Qualitative examples of large residuals, with generation metadata for context.

### 7.5 Reporting standards

Every experiment records: config hash, dataset version/hash, split hash, code
revision, random seeds, metrics with confidence intervals, hardware, and
runtime. Negative and inconclusive results are recorded with the same detail as
positive ones. No result is reported without its split definition.

## 8. Threats to validity

- **Construct validity:** reference-based metrics are proxies, not ground-truth
  legibility.
- **Internal validity:** leakage is controlled by grouping and frozen splits,
  but undiscovered near-duplicates remain possible; duplicate checks are
  mandatory.
- **External validity:** synthetic renderings and a small font set do not
  represent production CAPTCHAs.
- **Conclusion validity:** small held-out group counts limit statistical power;
  confidence intervals are reported and over-interpretation is avoided.
- **Reproducibility:** results depend on library versions and hardware; these are
  pinned and logged where they affect numerical output.

## 9. Ethical and legal considerations

- No live CAPTCHA systems are queried or bypassed.
- Only synthetic images and redistributable fonts are used.
- No human-subject data is collected, so no solvability labels exist.
- The project does not provide or evaluate solving tools.
