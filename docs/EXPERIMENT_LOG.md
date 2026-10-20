# Experiment Log

**Status:** `EXP-0001` and `EXP-0002` completed. No other experiments have been run.

This file is the authoritative record of experiments. An experiment is logged
only if it can be reproduced from the recorded configuration, dataset version,
split, and code revision.

## How to use this log

1. Copy the template into the **Entries** section, newest first.
2. Assign the next sequential ID: `EXP-0001`, `EXP-0002`, ...
3. Record the exact code revision (`git rev-parse --short HEAD`) and note whether
   the working tree was clean.
4. Link the config path and the dataset/split identifiers; do not paste large
   tables here. Generated outputs belong under `reports/` (Git-ignored).
5. Record negative, failed, and inconclusive runs. Mark them as such.
6. Never edit a closed entry to "improve" a result. Supersede it with a new ID
   and cross-reference the correction.

## Conventions

| Field | Values |
| --- | --- |
| Status | `planned`, `running`, `completed`, `failed`, `inconclusive`, `superseded` |
| Task | `target-validation`, `dataset-generation`, `quality-regression`, `corruption-classification`, `severity-classification`, `cnn-comparison`, `robustness-analysis` |
| Split | Frozen split ID plus hash, for example `split-v1` (`<hash>`) |
| Dataset | Dataset version plus manifest hash |
| Metrics | Name, value, unit, and confidence interval where applicable |

## Index

| ID | Date | Task | Status | One-line outcome |
| --- | --- | --- | --- | --- |
| `EXP-0002` | 2026-10-20 | `dataset-generation` | `completed` | Two deterministic datasets generated; grouped splits verified; no single-feature triviality, but an eight-feature linear baseline reaches test R² 0.962 |
| `EXP-0001` | 2026-10-20 | `target-validation` | `completed` | SSIM passes in both families; PSNR passes for noise but fails for blur on an ink-coverage confound; MS-SSIM deferred |

## Entries

<!-- Copy the template below. Newest entries first. -->

```markdown
### EXP-XXXX — <short, specific title>

- **Date:**
- **Status:**
- **Task:**
- **Hypothesis / question:**
- **Code revision:** `<sha>` (working tree clean: yes/no)
- **Environment:** Python `<version>`, key library versions, hardware
- **Configuration:** `configs/<file>.yaml` (config hash: `<hash>`)
- **Dataset:** `<dataset-version>` (manifest hash: `<hash>`)
- **Split:** `<split-id>` (hash: `<hash>`), groups: `<n>`, images: `<n>`
- **Seeds:**
- **Runtime:**

**Method.** What was run, and what differed from the previous experiment.

**Metrics.**

| Metric | Value | 95% CI | Notes |
| --- | --- | --- | --- |
| | | | |

**Observations.** What happened, including unexpected behaviour, instability,
or data issues. Facts only; interpretation goes below.

**Conclusion.** What this supports or rules out, and how confident we are.

**Limitations.** Threats specific to this run (small groups, metric saturation,
class imbalance, and so on).

**Next step.** The concrete follow-up, or "none".

**Artifacts.** Paths to Git-ignored reports/figures, if any.
```

---

### EXP-0002 — Reproducible synthetic dataset, grouped splits and feature diagnostic

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `dataset-generation`
- **Hypothesis / question:** Can the two validated corruption families be turned
  into a deterministic, versioned dataset with stable IDs, one clean reference
  per sample, grouped splits fixed before generation, and typed image-only
  features — and do those features make the SSIM target trivially predictable on
  held-out groups? This is a dataset-validity and leakage experiment, not a
  human-solvability study.
- **Code revision:** base `92ecf33` plus the Milestone 2 implementation; this
  entry and that implementation are committed together in
  `feat: add reproducible dataset and image features`. Working tree: clean at
  run apart from the files being committed.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; Pillow 12.3.0;
  scikit-image 0.26.0; SciPy 1.18.1; macOS 26 (arm64). No GPU, no internet and
  no Kaggle access were required.
- **Configurations:**
  - `configs/dataset_pilot.toml`, sha256
    `64fb26f78cffecdb9dc8543c600a0bae8fc1c0a66e85adb72bcdb96341e73512`;
    generation hash `f9e66dc9ca109a7b1cf8a47e98f19c0214cf723933047583edbda0faec198b1d`.
  - `configs/dataset_v0_1_0.toml`, sha256
    `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`;
    generation hash `820cce7a77b5e95761f033b75e8cb2b304b6ad6293f67b5bc27bcf0823a6cef9`.
  - The M1 config `configs/pilot_target_validation.toml` is unchanged (sha256
    `6e9bdf961f36bbf70f5806e41b53661e1577bf15e13eb833348632359b886715`).
- **Datasets:** `v0.1.0-pilot` (9 sources, 81 samples; manifest sha256
  `2a5984c3f12efa14caf853c171dbe16a31ac8f399b8d741845be181b2f501f44`) and
  `v0.1.0` (60 sources, 540 samples; manifest sha256
  `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`).
- **Split:** derived from the generation hash plus split seed `20261021`;
  assignment sha256 for `v0.1.0` is
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`.
  Groups: 60 sources (36/12/12 train/val/test); samples: 540 (324/108/108).
- **Seeds:** dataset seed `20261020`; split seed `20261021`. Per-source text,
  render and corruption seeds are derived independently from the dataset seed,
  font, source index, family and severity. No global RNG state.
- **Runtime:** pilot generation ~0.6 s; reference generation ~1.2 s in-process
  (1.4 s wall including CLI startup); full verification 0.9 s; diagnostics
  0.2 s. Disk usage: 503 KiB (pilot) and 3.4 MiB (reference).

**Method.** Loaded a strict TOML configuration, planned one clean source per
`(font, index)` with text length drawn from the configured distribution, and
assigned font-stratified grouped splits **before** rendering. Rendered a clean
192x64 grayscale reference per source, then every non-zero severity of both
families (blur radii `[1, 2, 3.5, 5]` px; noise sigma `[6, 12, 20, 32]`), with
an independent seeded noise realisation per variant. Computed SSIM (primary)
and PSNR (secondary) from the exact reference/variant pair. Extracted eight
documented image-only features from every degraded image. Wrote a public
manifest, a separate internal `ground_truth.csv`, `sources.csv`, `features.csv`
and `dataset_summary.json` through a staging directory, then renamed it into
place.

Verification recomputed every target from disk, checked raw-pixel hashes, path
containment, reference↔source matching, conditional PSNR, split isolation and
feature-table integrity. Two additional independent runs were compared
byte-for-byte. Diagnostics fitted only on `train`: a naive mean, three metadata
baselines, one OLS per feature, and an all-feature OLS; one-feature selection
used validation R² and test was reported once. The pilot dataset was run first,
then the larger reference dataset.

**Metrics.**

| Metric | Value | Notes |
| --- | --- | --- |
| Sources / samples | 60 / 540 | 1 clean + 4 blur + 4 noise per source |
| Split sources (train/val/test) | 36 / 12 / 12 | all three fonts in every split (12/4/4) |
| Split samples (train/val/test) | 324 / 108 / 108 | family and severity composition proportional |
| Feature rows x features | 540 x 8 | finite; stable names and order |
| Verification checks | 11 / 11 PASS | includes 540 recomputed target pairs |
| Leakage checks | 4 / 4 PASS | target exclusion, group isolation, disjointness, train-only fitting |
| Best single feature (val/test R²) | `histogram_entropy` 0.680 / 0.644 | selected on validation only |
| All-feature OLS (val/test R²) | 0.968 / 0.962 | standardised on train |
| Metadata severity + family (val/test R²) | 0.924 / 0.911 | oracle label baseline, not a model input |
| Metadata font (val/test R²) | -0.001 / 0.002 | no font confounding in this split |
| Metadata text length (val/test R²) | 0.000 / 0.000 | no length confounding in this split |
| Reproducibility | byte-identical | two independent runs; manifest and features hashes equal |

Mean SSIM by severity on `v0.1.0` (n = 60 per cell; clean = 1.000):

| Family | s1 | s2 | s3 | s4 |
| --- | --- | --- | --- | --- |
| Gaussian blur | 0.897 (0.013) | 0.675 (0.024) | 0.475 (0.034) | 0.411 (0.053) |
| Gaussian noise | 0.871 (0.035) | 0.742 (0.045) | 0.635 (0.043) | 0.540 (0.042) |

The pilot dataset gave the same ordering: blur 0.901/0.672/0.461/0.395,
noise 0.928/0.804/0.680/0.572.

**Observations.** Facts only:

1. **Determinism held end-to-end.** Repeated generation produced identical
   manifests (`a6a2cb8654eb…`), feature tables (`d3715e7975cf…`), source and
   ground-truth tables, and 540 identical image pixel hashes. No timestamps are
   written into any artifact.
2. **Split allocation needed a floor repair.** With three sources per font and
   0.6/0.2/0.2 ratios, plain largest-remainder allocation gives `2/1/0`, silently
   excluding a split from a font. The implemented rule reserves one group per
   split and takes the remainder from the largest split, producing `1/1/1` and
   satisfying the predeclared font-stratification requirement.
3. **Text-length distribution is not balanced per split.** Train has 11/14/11
   sources at lengths 4/5/6, validation 5/7/0, test 2/6/4. This is reported, not
   corrected: only font was stratified by design.
4. **The target is strongly predictable from simple statistics.** One feature
   (`histogram_entropy`) reaches validation/test R² 0.680/0.644, below the
   predeclared 0.70 "strong confound" threshold, so the single-feature
   classification is **no single-feature triviality**. However, the eight-feature
   linear baseline reaches **0.968/0.962**, classified
   `feature_only_near_solved`.
5. **Family-conditional relationships reverse sign.** `laplacian_variance` has
   overall Pearson r = 0.066, but r = 0.883 within blur and r = -0.861 within
   noise. `ink_fraction` is r = -0.930 within blur and r = 0.182 within noise.
   Aggregate correlations are misleading here; the report includes within-family
   and within-severity tables.
6. **8-bit clipping still compresses heavy noise.** At nominal sigma 20
   (severity 3), mean `intensity_std` fell from 75.29 on the clean renders to
   73.14 on the noisy variants (58 of 60 sources), while the Immerkær estimate
   rose from 3.77 to 15.84 on every source. Intensity variance is therefore not a
   monotone noise indicator.
7. **High blur removes all Canny edges for some sources.**
   `edge_density` reaches exactly 0 on strongly blurred images, which is a valid
   but discrete response; the feature table records it rather than hiding it.
8. **The pilot is too small to override the larger result.** With three training
   sources, pilot entropy R² is 0.822/0.801 (`strong_confound`) and the
   all-feature baseline 0.985/0.974. The larger run is the reference for M3.

**Conclusion.** The pipeline produces a valid, versioned, byte-reproducible
synthetic dataset with one clean reference per sample, correct conditional PSNR
semantics, stable IDs, and grouped splits that never cross sources. The target
and features are leak-free by construction and by programmatic checks. The
diagnostic supports a specific decision: the SSIM target is **not** trivially
predictable from any single feature, but it is almost completely predictable
from the eight simple statistics together. A CNN is therefore not justified by
target predictability alone; M3 baselines must beat the feature-only linear
baseline under grouped evaluation, and any claimed gain must not be explained by
the same statistics. These findings concern synthetic image similarity only and
say nothing about human readability or CAPTCHA solvability.

**Limitations.**

- Three fonts, one canvas size, flat backgrounds only, two corruption families,
  and 60 synthetic sources. No cross-generator or cross-font claim is made; a
  held-out font/configuration split is reserved for later.
- The split is a random grouped split with 12 held-out source groups per split;
  R² values are noisy and are diagnostics, not model-performance estimates.
- Text length is sampled from a distribution but not stratified; validation has
  no length-6 sources by chance.
- PSNR remains secondary and is only interpreted for the noise family (M1).
  MS-SSIM is still deferred.
- The noise family is compressed by 8-bit clipping, so nominal sigma is not the
  realised noise standard deviation.
- Features describe the degraded image only; they cannot capture
  reference-relative structure, which is why the all-feature linear baseline
  still leaves ~4% of held-out variance.
- System fonts are recorded by path and logical ID, not by binary content hash.
- Single-machine run; PNG bytes may vary across Pillow/zlib versions, so
  reproducibility is asserted on decoded pixel hashes and CSV contents, not on
  PNG file bytes.

**Next step.** Milestone 3: grouped conventional baselines (naive, ridge/linear,
tree ensemble) trained on `features.csv`, evaluated on the frozen split, with the
feature-only OLS baseline as the floor. Add coverage-stratified reporting and a
held-out-font check. No CNN or dashboard work before that comparison is
recorded.

**Artifacts.** Git-ignored: `data/processed/captcha_quality_v0.1.0-pilot/`,
`data/processed/captcha_quality_v0.1.0/`, `data/processed/repro_a/`,
`data/processed/repro_b/`, `reports/captcha_quality_v0.1.0-pilot/` and
`reports/captcha_quality_v0.1.0/` (diagnostics markdown plus four CSV tables).

---

### EXP-0001 — Reference-metric validation pilot (SSIM, PSNR, blur, noise)

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `target-validation`
- **Hypothesis / question:** Do SSIM and PSNR respond predictably and consistently
  to increasing severity of Gaussian blur and additive Gaussian noise, with
  adjacent severities distinguishable and the response stable across fonts and
  seeds? This is a metric-validation experiment, not a human-solvability study.
- **Code revision:** base `d0e5256` plus the Milestone 1 implementation; this
  entry and that implementation are committed together in
  `feat: validate synthetic image quality targets`. Working tree: clean at run.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; Pillow 12.3.0;
  scikit-image 0.26.0; SciPy 1.18.1; macOS 26 (arm64). OpenCV was installed via
  the `image` extra but is not used by this pilot.
- **Configuration:** `configs/pilot_target_validation.toml`, sha256
  `6e9bdf961f36bbf70f5806e41b53661e1577bf15e13eb833348632359b886715`.
  Thresholds were fixed in the config and in `docs/METHODOLOGY.md` §3.4 **before**
  the recorded run and were not edited afterwards.
- **Dataset:** synthetic, generated in-process; no stored image collection. The
  Git-ignored `reports/pilot/` directory holds tables and contact sheets only.
- **Split:** not applicable (descriptive validation; no model is trained).
- **Seeds:** run seed `20261020`; repeat seeds `[0, 1, 2]`, each an independent
  render-jitter and noise realisation.
- **Runtime:** 0.82 s wall-clock for the full run (324 unique image contents).

**Method.** Rendered 5-character strings (charset `23456789ABCDEFGHJKMNPQRSTUVWXYZ`,
ambiguous glyphs removed) at 192x64 px, using 4x supersampling with LANCZOS
downsampling and seeded per-character vertical jitter. Three system fonts
(Arial, Courier New, Times New Roman) x 4 texts each x 3 repeat seeds = **36 clean
reference images** (12 unique strings). Each clean image was scored against:

- itself (36 clean self-comparisons), and
- each severity of two families, reusing the byte-identical reference:
  Gaussian blur radius `[0, 1, 2, 3.5, 5]` px and additive Gaussian noise sigma
  `[0, 6, 12, 20, 32]` on the 0-255 scale, with an independent noise draw per
  `(trial, severity)`.

That yields **360 corrupted-pair rows** at 5 severity levels (severity 0 is the
identity baseline), plus 36 sanity rows. SSIM used `data_range=255`,
`win_size=11`, Gaussian weights (`sigma=1.5`), no sample-covariance correction.
PSNR used `data_range=255`; identical images return `math.inf` explicitly.
MS-SSIM was not evaluated (see limitations).

**Metric summaries.**

Level means (n = 36 per cell; SD in parentheses):

| Family | Metric | s0 | s1 | s2 | s3 | s4 |
| --- | --- | --- | --- | --- | --- | --- |
| blur | SSIM | 1.000 | 0.913 (0.009) | 0.712 (0.021) | 0.530 (0.025) | 0.469 (0.043) |
| noise | SSIM | 1.000 | 0.918 (0.007) | 0.781 (0.019) | 0.645 (0.027) | 0.529 (0.033) |
| blur | PSNR (dB) | +inf | 20.663 (0.670) | 15.577 (0.428) | 12.963 (0.449) | 11.955 (0.661) |
| noise | PSNR (dB) | +inf | 35.142 (0.070) | 29.219 (0.077) | 24.813 (0.107) | 20.760 (0.079) |

Spearman rho (severity vs. score, n = 180): blur/SSIM **-0.973**, blur/PSNR
**-0.946**, noise/SSIM **-0.983**, noise/PSNR **-0.968**.

Adjacent-level paired differences and effect sizes (same clean reference; `d` is
paired Cohen's d):

| Family / metric | s0->s1 | s1->s2 | s2->s3 | s3->s4 |
| --- | --- | --- | --- | --- |
| blur / SSIM | -0.087 (d=-9.89) | -0.201 (d=-14.89) | -0.183 (d=-5.59) | -0.061 (d=-2.80) |
| noise / SSIM | -0.082 (d=-11.77) | -0.137 (d=-11.28) | -0.136 (d=-13.03) | -0.116 (d=-15.27) |
| blur / PSNR | not evaluable (+inf baseline) | -5.086 dB (d=-15.44) | -2.614 dB (d=-4.16) | -1.008 dB (d=-3.67) |
| noise / PSNR | not evaluable (+inf baseline) | -5.923 dB (d=-57.51) | -4.406 dB (d=-29.27) | -4.053 dB (d=-30.75) |

Trial-level violation rate was **0.000** for every evaluable adjacent pair in both
families and both metrics. Eta squared (severity share of finite-score variance):
blur/SSIM 0.987, noise/SSIM 0.986, blur/PSNR 0.974, noise/PSNR 1.000. Level-mean
ranges: 0.531 SSIM, 0.471 SSIM, 8.708 dB, 14.382 dB respectively.

Per-font rho ranged over [-0.984, -0.979] (SSIM) and [-0.968, -0.964] (PSNR);
per-seed rho over [-0.984, -0.983] (noise/SSIM), [-0.974, -0.972] (blur/SSIM),
and [-0.968, -0.946] (PSNR). Every font and seed group had strictly decreasing
level means.

Strongest within-level correlation with ink coverage (`|r|`, signed value and
level): blur/SSIM 0.874 (-0.874 at s4), noise/SSIM 0.735 (+0.735 at s4),
blur/PSNR **0.975** (-0.975 at s4), noise/PSNR 0.139 (+0.139 at s4).

**Gate results** (predeclared rule, `docs/METHODOLOGY.md` §3.4):

| Family | Metric | Verdict | Failing criterion |
| --- | --- | --- | --- |
| gaussian_blur | ssim | **PASS** | — |
| gaussian_blur | psnr | **FAIL** | `non_degeneracy` (`|r_ink|` 0.975 > 0.95) |
| gaussian_noise | ssim | **PASS** | — |
| gaussian_noise | psnr | **PASS** | — |
| any | ms_ssim | **DEFERRED** | no lightweight implementation available without PyTorch |

All four combinations passed direction, aggregate trend, adjacent separation,
violation rate, cross-font/cross-seed stability, and clean-reference sanity
(36/36 exact SSIM=1.0, PSNR=+inf, byte-identical; severity-0 identity rows exact).

**Observations.** The response is smooth and near-monotone for all four
combinations. No trial-level reversals occurred at these ladder spacings. The
clean baseline is exactly perfect for both metrics. Two behaviours are worth
recording:

1. **PSNR under blur is largely an ink-coverage measure at fixed severity.**
   Within the highest blur level, PSNR correlates -0.975 with the clean image's
   ink fraction: images with more ink (e.g. Arial, 0.150 coverage) lose more
   edge area than sparse ones (Courier New, 0.089). SSIM shows the same direction
   at a lower magnitude (-0.874).
2. **The noise ladder is compressed by 8-bit clipping.** Realised noise standard
   deviations were 4.10, 8.08, 13.39, and 21.31 for nominal sigma 6, 12, 20, 32;
   44-46% of pixels hit a rail (mostly the white background). This is why
   measured PSNR exceeds the naive `20 log10(255/sigma)` values, and it matters
   for interpreting absolute severity.

No threshold was changed after inspecting these results.

**Conclusion.** SSIM is a suitable candidate primary target for both families
under the tested conditions. PSNR is suitable for additive noise but not for
blur as a standalone target, because at fixed blur severity it mostly restates
how much ink the image contains. MS-SSIM remains unevaluated. These findings
concern synthetic image similarity only; they say nothing about human
readability or CAPTCHA solvability.

**Limitations.**

- One canvas size, one text length, three fonts, two corruption families, and
  36 source images. Results may not transfer to other renderer settings.
- SSIM and PSNR are full-reference proxies for degradation, not perceptual or
  human-legibility measures.
- Geometric warps, occlusion, and compression are not tested; PSNR's geometric
  sensitivity is therefore not characterised here.
- PSNR's `+inf` clean level is excluded from finite aggregates by design, so the
  s0->s1 PSNR separation could not be quantified with the paired statistic
  (the level means still show the drop).
- Within-level ink coverage explains a substantial share of variance at high
  severity even for SSIM (0.874 at the strongest blur), so font/coverage
  confounding must be tracked in later modelling.
- MS-SSIM deferred: scikit-image provides no MS-SSIM and the maintained
  implementations require PyTorch, which is not installed in this milestone.
- Single-machine run on macOS with system fonts; font binaries and versions are
  recorded by path, not by content hash.

**Next step.** Proceed to Milestone 2 with SSIM as the primary reference target
and PSNR as a secondary diagnostic (noise-family comparisons only). When the
feature pipeline exists, re-run the full non-degeneracy check including
trivial-feature predictability, and add coverage-stratified reporting to track
the ink confound. Revisit MS-SSIM when PyTorch is installed for the CNN work.

**Artifacts.** Git-ignored: `reports/pilot/summary.md`, `gate_results.json`,
`trials.csv`, `sanity.csv`, `levels.csv`, `separation.csv`, `stability` tables,
`ink.csv`, `variance.csv`, and `examples_gaussian_{blur,noise}.png`.

---

## Milestone 0 note

Milestone 0 ran no experiments; it established tooling and documentation only.
