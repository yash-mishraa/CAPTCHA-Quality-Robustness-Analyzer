# Hypotheses and follow-ups

These are three testable explanations motivated by the descriptive M9A results.
They are not confirmed mechanisms. None was tested by fitting, calibrating or
selecting a model in M9A. Confidence labels reflect the strength and consistency
of the observed evidence, not intuition about the model.

## H1. Attenuated response to severe interference

**Claim.** The frozen model tracks the ordering of interference severity but
compresses the magnitude of its response, so severe thin-line and arc-stroke
damage is predicted as near-clean quality. This attenuation is the main source
of the interference MAE, negative bias and negative R².

**Evidence.**
- Interference MAE is 0.091 versus
  0.026 for blur and 0.039 for noise;
  interference bias is -0.087.
- At thin s4 the mean target is 0.696 but the mean
  prediction is 0.904; at arc s4 the values
  are 0.741 and
  0.906.
- The within-source prediction/target SD ratio is
  0.320 for thin
  lines and 0.397
  for arcs, compared with
  0.968 for blur
  and 0.975 for
  noise. The corresponding within-source slopes are
  0.241 and
  0.337.
- Residual-target Spearman correlations are
  0.916 and
  0.863: lower targets are
  overpredicted and higher targets are underpredicted.
- The top 5% of sources account for only
  10.6%
  of interference squared error, so this is not carried by a few extreme
  sources.

**Evidence against / alternatives.** Within-source rank correlations are still
0.792 for thin lines
and 0.885 for arcs, so
the model is not condition-blind. Mild s1-s3 rungs have few large errors; most
of the failure is at s4. The frozen M5C training description covers original
blur/noise and geometric groups, while prior M7B documentation describes thin
and arc interference as new out-of-distribution slices; training-distribution
shift is therefore an alternative explanation for the attenuation, not a proven
cause. Impact has a smaller target drop and a much smaller error, so a
target-difficulty explanation also contributes.

**Confidence.** High for the descriptive attenuation; moderate for the
out-of-distribution training explanation.

**Falsifying follow-up.** In a future, predeclared experiment, fit a monotone
calibration or a retrained model on source-disjoint *training* sources that
include the thin/arc ladder, then evaluate once on held-out sources. If
interference error is purely an attenuation problem, the intervention should
reduce interference MAE/RMSE and the s4 gap without damaging retention slices.
If it does not, attenuation alone is insufficient.

**Independent unit and split.** Source group. Split train/validation/test by
source group with no case or font overlap between training and test; hold out
at least two fonts and a disjoint set of text cases. Never resample images.

**Metric and predeclared acceptance criterion.** Primary metric: interference
RMSE on held-out source groups; secondary: s4 thin/arc MAE and large-error
proportion. Accept only if interference RMSE falls by at least 20% relative to
the frozen champion with a source-grouped 95% interval lower bound above zero,
while blur/noise RMSE increases by no more than 5% relative and clean-reference
MAE remains at or below 0.005.

## H2. Font/glyph-specific difficulty modifies the failure

**Claim.** Interference failure is not uniform across fonts. Thin-stroke or
otherwise difficult glyph sets (notably Courier New, Times New Roman and
Verdana) have larger interference error and larger target degradation, while
Impact is comparatively protected. Font identity is a modifier of the failure,
not an isolated cause.

**Evidence.**
- Courier New interference MAE is 0.130, Times New Roman
  is 0.103 and Impact is 0.036.
- The paired-by-case contrast versus the other nine fonts is
  0.044 for Courier New
  (95% CI [0.037,
  0.050]; worse on
  29/30 cases) and
  -0.061 for Impact
  (95% CI [-0.069,
  -0.053]; worse on
  0/30).
- The interference target is less damaged for Impact
  (mean 0.920;
  s1-to-s4 drop 0.128) than for Courier
  New (mean 0.819;
  drop 0.276).

**Evidence against / alternatives.** Font identity is entangled with glyph
shape, stroke width, renderer and the SSIM target itself. The per-font error
ordering partly follows target difficulty, so the evidence does not identify a
causal font property. All fonts still show negative interference bias (Impact
included), and the font table is descriptive. A font effect that disappears
after matching on target severity would not support a font-specific mechanism.

**Confidence.** Moderate.

**Falsifying follow-up.** Predeclare a held-out-font experiment with at least
two new thin-stroke and two new thick-stroke fonts, render the same cases and
corruptions, and compare interference error after matching on target severity
and text length. If the Courier-vs-Impact contrast disappears after matching,
the font-specific hypothesis is falsified.

**Independent unit and split.** Source group, with fonts held out entirely from
model fitting or calibration. Cases and fonts must be disjoint across
train/validation/test; do not reuse a source group across splits.

**Metric and predeclared acceptance criterion.** Metric: paired-by-case
interference MAE difference between thin-stroke and thick-stroke font groups,
matched on target severity. Accept only if the source-grouped 95% interval
excludes zero and the direction replicates in at least two independently held-
out fonts. Do not claim a font-population effect from ten purposive fonts.

## H3. Existing features conflate interference with benign edge/sharpness variation

**Claim.** The existing eight image features describe edge and sharpness
structure, but the two features most associated with interference error are
`laplacian_variance` and `edge_density`. The model may therefore represent
interference as a sharpness/edge phenomenon that overlaps with benign texture,
rather than as a distinct severe degradation.

**Evidence.**
- Within thin lines, `laplacian_variance` has Spearman rho
  +0.792 and `edge_density`
  +0.796 with absolute error; for arcs the
  corresponding values are +0.791 and
  +0.700 for the same edge family.
- `histogram_entropy` is negatively associated with absolute error
  (-0.282 thin, -0.394 arc),
  and the associations persist inside several severity strata.
- The strongest associations are edge/sharpness features
  (`laplacian_variance`, `edge_density`), which are plausible benign-texture
  cues. This is consistent with, but does not demonstrate, conflation.

**Evidence against / alternatives.** These are descriptive rank associations,
not a demonstration that the random forest uses a feature as a benign cue. The
features are also correlated with source content and severity, and the analysis
does not inspect tree splits, importances or counterfactual feature changes.
Within-source prediction-target rank correlations remain high, which is evidence
against a complete failure to represent interference. Feature associations may
be consequences of the target/source composition rather than causes of error.
No feature is recommended, added or selected by M9A.

**Confidence.** Low.

**Falsifying follow-up.** On held-out fonts and source groups, compare
interference and benign-condition rows matched on the eight existing features and
on target SSIM. If matched residuals show no systematic interference excess, the
conflation hypothesis is falsified. A second, independent test can ask whether
the feature-error associations survive within font x severity strata; if they
vanish, source composition explains them.

**Independent unit and split.** Source group; held-out fonts and disjoint cases
for any model-fitting or calibration step. Matching must be done without using
test outcomes to choose features or thresholds.

**Metric and predeclared acceptance criterion.** Metric: source-grouped mean
residual difference between matched interference and benign rows, and the
within-font/severity partial rank association of `laplacian_variance` and
`edge_density` with absolute error. Accept the hypothesis only if the matched
residual difference is at least 0.02 SSIM with a 95% interval excluding zero and
the feature association replicates in at least two held-out fonts. Otherwise
treat it as unsupported.

## What M9A does not establish

- It does not identify a causal mechanism for the interference error.
- It does not establish that any font is intrinsically worse for CAPTCHA
  readability, solvability or security.
- It does not justify adding a feature, changing the target, or replacing the
  frozen champion.
- It does not treat the M3 or M5C test metrics as like-for-like comparisons to
  M8C.
