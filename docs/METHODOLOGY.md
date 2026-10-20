# Methodology

**Status:** Proposed methodology for planned milestones. No experiments have
been run and no target has been validated yet.
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

Each sample starts as a clean rendering of a known string:

- Character set configurable; visually ambiguous characters (`0/O`, `1/l/I`)
  excluded by default to avoid confounding legibility with distortion.
- Text length sampled from a configured distribution.
- Font sampled from a licence-compatible set; font file and size recorded.
- Seeded layout: position jitter, rotation, per-character offset, foreground and
  background colour.
- Fixed canvas size and colour mode per dataset version.
- Optional background texture, recorded as a separate factor.

Rendering is deterministic given `(seed, config)` and the recorded RNG stream.
The renderer version is part of every sample record so images can be
regenerated exactly.

### 2.2 Distortion families

Each family is applied with a numeric parameter derived from an ordered severity
ladder, not hand-tuned per image:

| Family | Example parameterisation | Notes |
| --- | --- | --- |
| Gaussian noise | σ at fixed severity levels | additive, per-pixel |
| Blur | Gaussian kernel σ or motion length/angle | separable cases logged |
| Contrast / brightness | multiplicative and additive factors | clipped to valid range |
| Geometric warp | rotation angle, elastic displacement magnitude | interpolation method recorded |
| Occlusion / clutter | line count, width, opacity | deterministic geometry from seed |
| Compression | JPEG quality or downscale factor | applied last in composition |

Composition rules are explicit: families may be composed, and the applied order
and every numeric parameter are stored. Severity ladders are calibrated in M1 so
that adjacent levels produce measurably different quality scores; otherwise the
"severity" label is not meaningful.

### 2.3 Metadata and manifest

Per sample: `sample_id`, dataset version, code revision, RNG seed, clean text,
font, font size, render parameters, ordered corruption chain, severity labels,
and relative paths to the clean reference and degraded image. Quality targets
are computed and stored separately so metric code can be re-run.

### 2.4 Dataset scale and splits

The initial corpus is intentionally small enough to inspect and regenerate on a
single machine. Size is chosen after the M1 target-validation gate, not before.

Splits are assigned by **generation group** (seed and text family), stratified by
corruption family where sample counts permit. The split file is frozen and
reused by every model and every experiment. A font-held-out split is reserved
for a generalisation check.

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

### 3.4 M1 validation gate

Before dataset scale-up:

1. **Sensitivity:** mean target decreases monotonically with severity within
   each family (with tolerance for noise; report effect sizes).
2. **Separation:** adjacent severity levels are distinguishable relative to
   within-level variance.
3. **Stability:** scores are stable across fonts and seeds at fixed parameters.
4. **Non-degeneracy:** the target has meaningful variance and is not almost
   perfectly predicted by a single trivial feature (for example global contrast).
5. **Sanity check:** clean-vs-clean comparisons score at the metric's maximum.

If these fail, the target definition or corruption ladder is revised before any
model is trained.

## 4. ML baselines

### 4.1 Feature-based regression

Inputs are handcrafted image-quality features only: contrast/standard deviation,
entropy, sharpness (variance of Laplacian), edge density, colour statistics,
noise estimate, frequency-band energy, and reference-difference summaries where
appropriate. Generation metadata (severity, corruption family, seed) is **not**
an input to the primary regressor.

Models, in increasing complexity:

1. Dummy regressor (mean/median) as a floor.
2. Ridge / linear regression on standardised features.
3. Random forest or histogram gradient boosting.
4. Optional: gradient boosting with modest hyperparameter search.

Preprocessing (imputation, scaling) is fit inside each training fold only.

### 4.2 Corruption-type and severity baselines

- Multi-class classifier for corruption family.
- Ordinal or multi-class model for severity, evaluated with ordinal-aware
  metrics.
- These experiments test whether the corpus encodes the intended factors. They
  are **not** the primary quality model and must not be presented as evidence of
  perceptual understanding.

### 4.3 CNN comparison

- A small CNN trained from scratch (appropriate for a modest dataset).
- A pretrained backbone with a replaced regression head, using documented
  preprocessing and partial or full fine-tuning.
- Identical splits, target definition, and early-stopping protocol as the
  feature-based models.
- Report parameter count, training time, and hardware to make the cost/benefit
  comparison explicit.

The CNN is justified only if it improves held-out performance or reveals
structure the baselines miss. Otherwise the simpler model is retained.

## 5. Data leakage risks and controls

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

## 6. Evaluation strategy

### 6.1 Splits and resampling

- Frozen train/validation/test split, grouped by generation family.
- Grouped k-fold cross-validation on the training portion for model selection.
- Bootstrap confidence intervals over held-out groups for headline metrics.
- Report the number of independent groups, not just the number of images.

### 6.2 Regression metrics

- MAE and RMSE (primary, in target units).
- R² for explained variance, interpreted cautiously on bounded targets.
- Spearman rank correlation for monotonic association.
- Residual analysis by corruption family, severity, font, and text length.
- Calibration-style plot of predicted vs. reference scores.

### 6.3 Classification metrics (secondary)

- Macro-F1 and balanced accuracy for corruption family.
- Confusion matrix with per-class support.
- Ordinal MAE and quadratic weighted kappa for severity.
- Explicit chance baselines.

### 6.4 Robustness analysis

- Performance per corruption family and severity level.
- Cross-family generalisation: train on some families, test on held-out ones.
- Worst-group and tail error reporting, not only aggregate means.
- Qualitative examples of large residuals, with generation metadata for context.

### 6.5 Reporting standards

Every experiment records: config hash, dataset version/hash, split hash, code
revision, random seeds, metrics with confidence intervals, hardware, and
runtime. Negative and inconclusive results are recorded with the same detail as
positive ones. No result is reported without its split definition.

## 7. Threats to validity

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

## 8. Ethical and legal considerations

- No live CAPTCHA systems are queried or bypassed.
- Only synthetic images and redistributable fonts are used.
- No human-subject data is collected, so no solvability labels exist.
- The project does not provide or evaluate solving tools.
