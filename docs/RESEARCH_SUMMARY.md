# Research Summary

**Project:** CAPTCHA Quality & Robustness Analyzer
**Release:** `v1.0.0` research package
**Status:** M0–M9C complete; M9C selected **Path A — stop experimentation and package the research**
**Final evidence cut:** `81710e6c839f4e7f66752d4045c765df573acbe2`

## Research question

Can a conventional model predict the structural similarity of a degraded
synthetic CAPTCHA-style text image to its exact clean reference from whole-image
features alone, and how does that prediction behave across corruption families,
severities and fonts?

The target is **SSIM against the clean reference**. PSNR is a diagnostic. The
project does not measure human readability, CAPTCHA solvability, security or
production robustness.

## Experimental setup

- **Images:** synthetic 192×64 8-bit grayscale PNGs from a fixed renderer.
- **Fonts:** ten pinned Apple-bundled macOS faces: Arial, Courier New, Times New
  Roman (familiar); Georgia, Monaco, Impact (evaluated as held-out in M7B); and
  Verdana, Trebuchet MS, Chalkduster, Luminari (new held-out identities in M8C).
- **Corruptions:** Gaussian blur, additive Gaussian noise, thin-line
  interference, arc-stroke interference and bounded smooth elastic warp.
- **Features:** eight documented whole-image features:
  `intensity_mean`, `intensity_std`, `contrast_rms`, `ink_fraction`,
  `laplacian_variance`, `edge_density`, `histogram_entropy` and
  `noise_sigma_immerkaer`.
- **Model:** the frozen M5C champion `expanded_random_forest`, calibration
  `none`, artifact SHA-256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`.
- **Splits:** grouped by clean source. All variants of a source stay in one
  split. M3/M5C use 12 test source groups; M7B uses 60; M8C uses 300 source
  groups and 30 matched cases.
- **Uncertainty:** source-grouped and matched-case bootstrap intervals. These
  are descriptive, not significance tests or population estimates.

The M8C design is matched by construction: every font sees the same thirty text
cases. It has 300 sources and 6,300 rows: 5,700 primary and 600 diagnostic. The
clean-reference and 0.5 px elastic rows are diagnostic only and are excluded
from every primary metric.

## Main findings

### 1. The original synthetic task is easy for the frozen champion

On the M3 test split (108 rows, 12 sources, familiar fonts, blur/noise plus
clean), the selected random forest reaches MAE `0.0157`, RMSE `0.0226` and
R² `0.9865`. The M5C champion is similar on the same slice: MAE `0.0156`,
RMSE `0.0220`, R² `0.9872`. This is a within-renderer, within-font-family
result, not broad generalisation.

The M5C champion improves the geometric test slice to MAE `0.0493`, RMSE
`0.0590` and R² `0.1855`, while keeping clean-reference RMSE at `0.0023` and
original blur/noise RMSE at `0.0233`. Those slices are small: 36 geometric rows
and 12 clean rows from 12 source groups.

### 2. Held-out fonts are harder in the unmatched M7B comparison

M7B evaluates the frozen champion on 60 sources from Georgia, Monaco and Impact
(1,140 primary rows). All-primary MAE is `0.063427`, RMSE `0.092952` and
R² `0.743537`. Original blur/noise RMSE is `0.095923`, compared with `0.023328`
on the recorded familiar-font slice. Geometric aggregate RMSE is nearly
unchanged at `0.058839` versus `0.059006`.

This comparison is **unmatched** in fonts, source composition and corruption
mixture. It is a descriptive difference, not a font effect or a significance
test.

### 3. The matched M8C benchmark does not show a uniform held-out penalty

M8C evaluates the same frozen champion on 300 sources from ten fonts and 30
matched cases. All-primary MAE is `0.059607`, RMSE `0.088610` and R² `0.732898`.
By font group:

| Font group | RMSE | R² |
| --- | ---: | ---: |
| Familiar | 0.092894 | 0.683543 |
| M7B held-out | 0.091626 | 0.750091 |
| New held-out | 0.082831 | 0.753114 |

Per-font MAE ranges from `0.053974` (Arial) to `0.077869` (Impact). The matched
design does not reproduce a simple "held-out fonts are uniformly worse" story.
The differences are descriptive over seven observed held-out identities, and
target difficulty is entangled with font, glyph shape and rasterisation.

### 4. Thin-line and arc interference are the weak conditions

M8C primary results:

| Scope | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| All primary | 0.059607 | 0.088610 | 0.732898 |
| Original blur/noise | 0.032735 | 0.057131 | 0.899708 |
| Interference | 0.090825 | 0.118525 | -0.430242 |
| Thin lines | 0.102073 | 0.132114 | -0.575158 |
| Arc strokes | 0.079578 | 0.103162 | -0.298911 |
| Geometric primary | 0.048018 | 0.059687 | 0.162498 |

Negative R² means the model is worse than a condition-mean predictor in
squared-error terms on that slice. The result is reported, not tuned away.

### 5. M9A shows broad error concentration and amplitude compression

M9A is a read-only exploratory analysis of the frozen M8C predictions. It finds:

- Interference MAE `0.0908`, bias `-0.087` (overprediction) and R² `-0.4302`.
- Large errors at `|residual| >= 0.20` on 15.17% of thin-line rows and 7.00% of
  arc rows; at severity 4 these rise to 55.3% and 28.0%.
- Within-source thin-line slope `0.241` and prediction/target SD ratio `0.320`;
  arc slope `0.337` and ratio `0.397`. Blur and noise are near 1.0.
- The top 5% of interference source groups account for `10.6%` of interference
  squared error, versus `51.5%` for blur and `49.7%` for noise.
- Font interference MAE ranges from `0.0358` (Impact) to `0.1302` (Courier New),
  but font identity is entangled with target difficulty.

These findings are exploratory. The 0.20 threshold was fixed before the
subgroup tables, but earlier reconnaissance had seen aggregate and severity
summaries, so every threshold-based subgroup proportion is labelled exploratory.
Feature associations are descriptive rank correlations, not source-clustered
inference and not a feature-selection result.

## Methodological contributions

1. **Source-grouped evaluation.** Splits and bootstrap intervals operate on clean
   source groups, not individual corrupted images. This prevents near-duplicate
   variants from crossing evaluation boundaries.
2. **A frozen champion.** M5C is selected once and then evaluated zero-shot in
   M7B and M8C. No evaluation result is used to fit, calibrate, tune or replace
   it.
3. **Matched cross-font design.** M8C pairs every font with the same 30 text
   cases, reducing text-composition confounding relative to M7B. It still
   cannot separate font identity from glyph shape, rendering and target
   difficulty.
4. **Provenance and replay.** M8C replay, lifecycle checks, M8E integrity and
   M8D audit tie the reported metrics to committed predictions and images. The
   historical dirty revision is preserved, not rewritten.
5. **Preregistration and a stopping rule.** M9B fixes what would count as
   support, falsification or inconclusive evidence before any confirmatory data
   exist. M9C uses the existing evidence to justify a stop rather than an
   automatic extension.

## Negative and uncertain results

- M5D training-objective changes did not beat the M5C champion.
- M6A spatial features, M6B finer resolution, M6C local structural features and
  M6D multi-scale structure-tensor features did not justify replacing the
  champion. Some candidates were neutral or worse on retention.
- M8C thin-line and arc R² are negative. The model ranks and responds
  monotonically within sources but compresses the amplitude at severe
  interference.
- The M7B large held-out-font blur/noise gap did not appear as a uniform
  group-level penalty in the matched M8C benchmark. The two designs answer
  different descriptive questions.
- Font-level and feature-level differences remain confounded. No causal
  mechanism is identified.

## Limitations

- **Synthetic target.** SSIM is not human readability, legibility, solvability
  or security.
- **One renderer and platform.** The data are tied to a fixed 192×64 renderer,
  one macOS environment and pinned Apple-bundled font builds.
- **Constrained font roster.** Ten purposively selected fonts are not a font
  population. M8C has seven held-out identities, not 300 independent fonts.
- **Limited independent groups.** M3/M5C have 12 test source groups; M7B has 60;
  M8C has 300. Intervals are descriptive.
- **Confounding.** Font, glyph shape, rasterisation and target difficulty are
  entangled. M8C cannot estimate a causal font effect.
- **Diagnostic slices.** Clean-reference R² is undefined; the 0.5 px elastic rung
  is near-degenerate. Neither is a primary result.
- **Provenance boundaries.** Six imported modules were not pinned in the
  historical M8C run, and the M8C input manifest has one declared duplicate key.
  These are documented, not silently repaired.
- **No human or security validation.** There are no human labels, solver data,
  live CAPTCHA experiments or production deployment evidence.
- **No confirmatory follow-up.** M9B is a protocol, not a result.

## Final decision

M9C selected **Path A — stop experimentation and package the research**. The
experiment phase is closed. H1 (attenuated response) is the first candidate if
the project is reopened with new independent data and an approved protocol; it
is not authorized now. H2 and H3 are deprioritized. M9B was not executed, and
H1, H2 and H3 remain `not_tested`.

No further experiment, pilot, dataset generation, model fit, calibration,
feature change, benchmark change or champion replacement is authorized by this
release.

## Deferred work (not authorized)

- H1 confirmatory study with independent source set C.
- H2 feature-conflation diagnostic with predeclared matching and balance rules.
- H3 new-font characterization with licensed thin/thick font groups.
- Unseen renderer, resolution or format study.
- Human readability or solvability study with ethics approval and human labels.
- CNN comparison or dashboard.
- New feature family or model development.

Each would require a new milestone, a new protocol and, where applicable, new
independent data.

## Non-claims

This project does not claim, and the current evidence cannot establish:

- human readability, legibility, solvability or response-time outcomes;
- CAPTCHA security, attack success or production robustness;
- a causal font, feature or corruption mechanism;
- a font-population effect;
- that synthetic-reference SSIM is a perceptual or human-legibility measure;
- that H1, H2 or H3 is supported or falsified.
