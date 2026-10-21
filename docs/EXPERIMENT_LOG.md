# Experiment Log

**Status:** `EXP-0001` through `EXP-0017` completed. No other experiments have
been run.

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
| Task | `target-validation`, `dataset-generation`, `quality-regression`, `conventional-benchmark`, `corruption-classification`, `severity-classification`, `cnn-comparison`, `robustness-analysis` |
| Split | Frozen split ID plus hash, for example `split-v1` (`<hash>`) |
| Dataset | Dataset version plus manifest hash |
| Metrics | Name, value, unit, and confidence interval where applicable |

## Index

| ID | Date | Task | Status | One-line outcome |
| --- | --- | --- | --- | --- |
| `EXP-0017` | 2026-10-21 | `robustness-analysis` | `completed` | Read-only evaluation-design plan (no dataset, image, model or prediction produced): 38 pinned inputs and 46 reproduced evidence values; Direction **A** (broader unseen-font) selected with Direction C's matched text-case control incorporated and Direction B (unseen-renderer) deferred; planned design is 10 fonts × 30 matched text cases = 300 sources / 6,300 rows (5,700 primary, 600 diagnostic) with the frozen M5C champion evaluated zero-shot; font-level inference stays descriptive because ten purposive fonts are not a font population |
| `EXP-0016` | 2026-10-21 | `robustness-analysis` | `completed` | Read-only cross-font analysis: all-primary MAE/RMSE/R² 0.0634/0.0930/0.7435; per-font RMSE 0.0852/0.0761/0.1134 with the Georgia–Monaco interval including zero; blur/noise RMSE 0.0279/0.0346/0.1601 versus familiar 0.0233 but a 5.74× font range; geometric aggregate 0.0588 versus 0.0590 while severity RMSE spans 0.0259–0.0829 (3.20×); top-3 source share 0.099 and 48.4 effective sources; 13-claim audit yields 3 supported / 0 suggestive / 10 unsupported |
| `EXP-0015` | 2026-10-21 | `robustness-analysis` | `completed` | Held-out-font benchmark: the frozen M5C champion reaches primary MAE/RMSE/R² 0.0634/0.0930/0.7435 on 60 Georgia/Monaco/Impact sources; blur/noise RMSE is 0.0959 versus 0.0233 on the familiar-font benchmark (point estimates outside the familiar intervals), geometric aggregate RMSE is similar (0.0588 versus 0.0590), interference is poorly calibrated (RMSE 0.1001, R² -0.1744), and the comparison is descriptive rather than matched |
| `EXP-0014` | 2026-10-21 | `robustness-analysis` | `completed` | Dataset/evaluation audit: the benchmark supports within-renderer SSIM claims and the M5C champion comparison, but not unseen-font, unseen-renderer, real-world, human-readability or broad-generalisation claims; 12 test source groups, a near-degenerate 0.5 px rung and a strongly redundant eight-feature representation bound the evidence, and the recommended next step is explicitly held-out fonts or rendering conditions |
| `EXP-0013` | 2026-10-21 | `robustness-analysis` | `completed` | Multi-scale structure-tensor features do not improve geometric prediction: the combined candidate is neutral on the geometric aggregate (RMSE +0.0014, interval includes zero) but breaches the clean-reference tolerance, the structure-only candidate is worse on every retention slice, the new features are strongly redundant with each other and with the global eight, and the champion is retained |
| `EXP-0012` | 2026-10-21 | `robustness-analysis` | `completed` | Local structural features do not improve geometric prediction: global-plus-local is neutral on the geometric aggregate (RMSE +0.0044, interval includes zero) but regresses clean and original-task retention, local-only is worse everywhere except the 1 px rung, and the local family is strongly redundant with the global one, so the champion is retained |
| `EXP-0011` | 2026-10-21 | `robustness-analysis` | `completed` | Finer grids do resolve the geometric warp (occupancy/SSIM correlation rises from +0.45 at 2x2 to +0.83 at 8x8) but the model cannot use it: every spatial candidate is ineligible on retention, the geometric gains are within noise, and the retention cost grows monotonically with feature count, so the champion is retained and the blocker is feature relevance, not resolution |
| `EXP-0010` | 2026-10-20 | `robustness-analysis` | `completed` | Spatial features do not improve geometric prediction: the combined ablation gains only 0.0006 geometric RMSE and regresses the original task, the spatial-only ablation is far worse everywhere, and a grid diagnostic shows why — a 4x2 partition resolves blur 25x better than a 1-3 px warp, so the champion is retained and localization at this resolution is ruled out |
| `EXP-0009` | 2026-10-20 | `robustness-analysis` | `completed` | Retention-aware training objectives do not beat the M5C champion: the severity-aware scheme is ineligible on a per-severity regression the aggregate score would hide, and the retention-aware scheme is eligible but scores worse, so the champion is retained and the eight global whole-image features are implicated as the binding constraint |
| `EXP-0008` | 2026-10-20 | `robustness-analysis` | `completed` | A retention-aware rule on validation picks an expanded random forest that cuts primary geometric RMSE 39.0% (0.0967 to 0.0590, 11/12 sources) while keeping the original M3 test RMSE at 0.0220 versus the frozen 0.0226, at a small clean-reference cost (RMSE 0.0023 versus 0.0012); conventional ML remains the champion and no CNN is justified |
| `EXP-0007` | 2026-10-20 | `robustness-analysis` | `completed` | Frozen M3 fails zero-shot geometric calibration (R² -1.186); adapted HGB cuts primary RMSE 31.8% to 0.0659 but leaves R² near zero and degrades the clean-reference slice, so conventional ML remains the family with a retention caveat and no CNN is justified |
| `EXP-0006` | 2026-10-20 | `robustness-analysis` | `completed` | Smooth elastic warp is deterministic, replayable and monotonic on 60/60 sources; gate is CONDITIONAL PASS because the 0.5 px rung has between-source SD 0.0025 below the 0.005 margin |
| `EXP-0005` | 2026-10-20 | `robustness-analysis` | `completed` | Frozen M3 fails zero-shot absolute calibration on M4A test (R² -1.045); validation-selected retraining cuts RMSE from 0.161 to 0.018, so conventional adaptation is sufficient here |
| `EXP-0004` | 2026-10-20 | `robustness-analysis` | `completed` | Occlusion/interference pilot is deterministic, non-degenerate and feature-distinct from blur/noise (gate PASS); no difficulty or solvability claim |
| `EXP-0003` | 2026-10-20 | `conventional-benchmark` | `completed` | A validation-selected random forest reaches test R² 0.986 (RMSE 0.023) versus naive 0.194; held-out-font R² 0.980; CNN not justified for this target |
| `EXP-0002` | 2026-10-20 | `dataset-generation` | `completed` | Two deterministic datasets generated; grouped splits verified; no single-feature triviality, but an eight-feature linear baseline reaches test R² 0.962 |
| `EXP-0001` | 2026-10-20 | `target-validation` | `completed` | SSIM passes in both families; PSNR passes for noise but fails for blur on an ink-coverage confound; MS-SSIM deferred |

## Maintenance notes

Maintenance changes are not experiments and are deliberately kept out of the
`EXP-` sequence. They never alter a recorded metric, prediction, dataset,
model, report or scientific conclusion, and they never rewrite a closed entry.

| Date | Scope | Change | Effect on recorded results |
| --- | --- | --- | --- |
| 2026-10-21 | M7C verifier (`cross_font_analysis.py`) | The protected-asset fingerprint was scoped from whole-directory trees (`configs/`, `data/processed/`, `models/`) to an explicitly declared membership of the dependencies M7C actually depends on: its own protocol plus the M7A/M5A/M7B/M5C protocols, the five frozen dataset roots, the nine frozen model roots, the eighteen historical report roots, the exact set of M7C output files, and an explicit list of transient scratch roots that are excluded as generated outputs. Expected hashes continue to come from the snapshot M7C recorded when it ran, so a modified, deleted, missing or substituted declared dependency is still detected. | None. M7C's metrics, predictions, datasets, protocol, report, tables and 13-claim audit are unchanged and were not recomputed, regenerated or re-hashed. The change was required because the M7D planning configuration (an unrelated file added to the shared `configs/` tree) had invalidated the M7C tree fingerprint even though all 31 M7C pinned inputs and every M7C output artifact were byte-identical. `EXP-0016` and `EXP-0017` remain valid as recorded; see [`docs/METHODOLOGY.md`](METHODOLOGY.md) §5.21. |

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

### EXP-0017 — Evaluation design and dataset-expansion plan for the next cross-font benchmark

- **Date:** 2026-10-21
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Which benchmark should come next, and what exactly
  should it measure? M7D compares a broader unseen-font benchmark (Direction A),
  an unseen-renderer benchmark (Direction B) and a matched font-transfer
  benchmark (Direction C), selects one primary direction, and pre-registers the
  dataset, endpoints, uncertainty, sample-size rationale, leakage controls,
  success criteria and stopping rules. M7D is a **read-only planning
  milestone**: it generates no dataset, renders no image, trains/refits/
  recalibrates/selects/tunes no model, changes no prediction, and modifies no
  frozen benchmark artifact. The target remains synthetic SSIM against an exact
  source-specific clean reference; this is not human readability, CAPTCHA
  solvability, security or production robustness.
- **Code revision:** base `d60643d` (M7C) plus the M7D implementation, committed
  as `docs: plan next cross-font generalization benchmark`. The reported run was
  repeated from the clean commit so provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; joblib 1.6.0; Pillow 12.3.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/evaluation_design_plan_v0_1_0.toml`; config sha256
  `e9f236b383172217e85096c1e06e81adcbb3fd85dc5c18f6537d824e1c6300d3`; identity
  hash
  `6464989d0c165d4749d2ece04bd7f8435d3499edf38e4dcd5f2097abd44b4537`.
  Seed `20261111`; 2,000 matched-case grouped percentile bootstrap replicates;
  confidence level 0.95; near-degeneracy margin target SD `< 0.005`.
- **Upstream frozen inputs:** 38 files pinned by SHA-256, including the M3
  manifest
  `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`,
  the M5C test predictions
  `6d286af061bd54e70b7c8fd547d8da7079731c502eb5255d848856edf2dec906`,
  the M7B predictions
  `660b9e8e816b22f9da4fcfeddc6f01108e23eb2b327bf8e8549fe9014e50312d`,
  the M7C font metrics
  `3975376b1162d1a391d8d80dbde6bb8016d2a4716f8be2030f32019739226cf1` and the
  M7C claim evidence
  `3e4dd2c05da2af15306e733dc0a2e5506cd9d024263950234e232512c3b2abe0`.
  Champion artifact sha256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`
  (`expanded_random_forest`, calibration `none`); card sha256
  `9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8`.
  The complete list is in the generated `input_manifest.csv`.
- **Dataset:** none created. The frozen inputs are M3 `captcha-quality-v0.1.0`
  (60 sources, 540 rows, 36/12/12 split) and M7B `heldout_font_benchmark-v0.1.0`
  (60 sources, 1,260 rows, 1,140 primary, 120 diagnostic, one `eval` split). The
  *planned* dataset would be `matched-font-v0.1.0` at
  `data/processed/matched_font_benchmark_v0.1.0/` with the benchmark report at
  `reports/matched_font_benchmark_v0_1_0/`; neither path exists.
- **Split:** none created. The plan fixes a single untouched `eval` split over
  all 300 planned sources, with font identity held out from development and all
  variants of a source kept together.
- **Seeds:** plan seed `20261111`; the future benchmark derives render and
  corruption seeds per matched text case so that fonts are paired.
- **Runtime:** recorded in `run_metadata.json`; two clean-commit runs differ
  only in `runtime_seconds`.

**Method.** The run verifies all 38 pinned files by SHA-256 and checks their
schemas, then recomputes 46 numerical claims from the frozen frames rather
than copying prior prose: M3 and M7B counts, M7B/M7C all-primary MAE/RMSE/R²,
per-font all-primary and blur/noise RMSE, geometric severity RMSE at 1/2/3 px,
source-level concentration, the M7C familiar-versus-held-out differences, the
M5C familiar champion metrics, the champion hashes and the M7A/M7C claim
classifications. It verifies the ten planned font files by path, SHA-256 and
charset glyph coverage. It then compares Directions A, B and C against explicit
scientific-value, uncertainty-resolved, independent-unit, cost, leakage and
reproducibility criteria; fixes the roster, corruption ladder, renderer, counts
and manifest requirements; separates nine independent units; declares the split
hierarchy and twelve leakage controls; pre-registers one primary, one
supporting, eight secondary/diagnostic and one deferred endpoint; computes
source-, paired- and font-level precision scenarios; and records fourteen
success criteria, eight stopping rules and twelve claim limits. It writes
deterministic CSV/JSON/Markdown outputs and provides `run` and `verify`
commands. It fingerprints 376 protected non-image files before and after the
run and refuses to write outside the Git-ignored report tree or into any prior
report directory.

**Metrics.**

| Metric | Value | 95% CI | Notes |
| --- | --- | --- | --- |
| Evidence rows reproduced | 46 / 46 | not a statistical test | recomputed from the pinned frames |
| M7C all-primary MAE / RMSE / R² | 0.063427 / 0.092952 / 0.743537 | descriptive | 1,140 rows, 60 sources |
| M7C per-font all-primary RMSE, Georgia / Monaco / Impact | 0.085198 / 0.076126 / 0.113431 | descriptive | unpaired font groups |
| M7C blur/noise RMSE, Georgia / Monaco / Impact | 0.027892 / 0.034646 / 0.160079 | descriptive | familiar M5C RMSE 0.023328 |
| M7C familiar-vs-held-out difference, blur/noise / geometric | +0.072595 / -0.000167 | descriptive | unmatched comparison |
| M7C severity RMSE, 1 / 2 / 3 px | 0.053384 / 0.025868 / 0.082867 | descriptive | max/min 3.20 |
| M7C all-primary top-3 share / effective sources | 0.099141 / 48.409 | descriptive | of 60 sources |
| M5C familiar champion RMSE, M3 test / original / geometric / clean | 0.022007 / 0.023328 / 0.059006 / 0.002275 | descriptive | 12 familiar test sources |
| M7A / M7C claim classifications | 3 / 2 / 7 and 3 / 0 / 10 | not a statistical test | supported / suggestive / unsupported |
| Planned fonts / matched text cases / sources | 10 / 30 / 300 | planned, not executed | 3 familiar, 7 held-out |
| Held-out faces reused from M7B / genuinely new | 3 / 4 | planned, not executed | M7B reuse was evaluation-only; the M7B three-font pattern is not independently re-tested |
| Planned rows / primary / diagnostic | 6,300 / 5,700 / 600 | planned, not executed | 21 rows per source |
| Source-level planning SD | 0.022634 | observed M7C per-source RMSE SD | descriptive planning input |
| Mean source RMSE precision at 30 cases | SE 0.004132, 95% half-width 0.008099 | normal approximation | not a power calculation |
| Paired-contrast half-width at 30 cases, SD 0.005 / 0.010 / 0.020 | 0.001789 / 0.003578 / 0.007157 | normal approximation | illustrative sensitivity only |
| Font-level half-width at 10 fonts, SD 0.010 / 0.020 / 0.030 | 0.006198 / 0.012396 / 0.018594 | normal approximation | descriptive roster precision |

**Observations.** The M7C evidence is internally consistent and reproduces
exactly: the familiar-versus-held-out blur/noise gap (+0.072595 RMSE) is large
but unmatched in fonts, source composition and corruption mix, so it cannot be
read as a font effect, while the geometric gap (-0.000167) is negligible in
aggregate but conceals a non-monotonic severity profile. All ten planned font
files were present on the current machine, matched their recorded SHA-256 and
covered the full project charset, but their licensing is **not** asserted: the
recorded status is `review_required_before_generation` for all ten, and
generation must fail closed until a local-use review exists. Direction A is
the only option that expands the font-identity axis, so it is selected;
Direction C's matched-case and matched-corruption logic is adopted inside A
because it is what repairs the unmatched comparison; Direction B is deferred
because a second renderer confounds renderer with antialiasing, placement,
canvas and format, and two renderers are not a renderer sample. The paired
within-case SD does not exist yet, so the plan reports a sensitivity grid
rather than a fabricated effect size. The roster widens the held-out set rather
than replacing it: Georgia, Monaco and Impact were already used in M7B as an
*evaluation* set (never for fitting, calibration or selection), so only four of
the seven held-out faces — Verdana, Trebuchet MS, Chalkduster and Luminari —
are genuinely new typefaces, and the design does not independently re-test the
M7B three-font pattern. The per-font `m7b_status` column records this for every
face. One frozen-verifier interaction is worth recording: the M7C `verify`
command fingerprints the entire `configs/` tree, so adding the new M7D
configuration changes that tree-level fingerprint. The
difference is exactly one added file; all 360 files recorded in the M7C run
remain byte-identical, all 31 M7C pinned inputs still match their recorded
SHA-256 values, and all M7C output artifacts are unchanged. No M7C code or
artifact was modified to silence the check, and M7D re-verifies the M7C
configuration, tables, metadata and report directly by content instead.

**Conclusion.** M7D establishes a pre-registered, implementation-ready design
for the next benchmark: Direction A, ten pre-specified typefaces, 30 matched
text cases, 300 font-specific sources, 6,300 rows, a frozen corruption ladder,
a frozen renderer, one primary matched endpoint with source-grouped
uncertainty, a full leakage and split contract, transparent precision
scenarios, and explicit success, stopping and interpretation rules. It does
**not** establish that the champion generalises to arbitrary fonts, to a font
population, to unseen renderers, or to human readers; it does not produce any
new measurement; and it does not authorize a model change.

**Limitations.** This is a plan, not a result. Ten typefaces are a purposive
maximum-variation roster, not a random sample of a font population, and two of
them are display/script stress faces rather than representative CAPTCHA
fonts. Three of the seven held-out faces were already used in M7B as an
evaluation set, so only four held-out identities are genuinely new and the M7B
three-font pattern is not independently re-tested. Thirty matched text cases
bound source-level precision, and the paired difference SD is unknown until the
data exist. One renderer, one 192x64 canvas
and one 8-bit grayscale PNG format are fixed. Font paths, availability and
licensing are machine-dependent. Matching reduces but does not eliminate
composition differences, because glyph metrics and ink distribution differ by
font. No equivalence or non-inferiority margin is predeclared, so a future
null result cannot be read as equivalence, and M7D deliberately defines no
universal SSIM improvement threshold.

**Next step.** Stop for review. If the plan is approved, the next milestone is
a separately versioned dataset generator and zero-shot benchmark that consumes
this roster, count, corruption-ladder, manifest and fail-closed requirements,
with any model change requiring its own authorized protocol and untouched test
set.

**Artifacts.** Git-ignored report directory
`reports/evaluation_design_plan_v0_1_0/` (input manifest, font inventory,
direction assessment, research questions, independent units, split hierarchy,
endpoints, sample-size scenarios, leakage controls, success criteria, stopping
rules, claim limits, evidence snapshot, isolation checks, plan manifest,
Markdown report and run metadata). The committed config, code, tests and
documentation are the reproducible protocol.

---

### EXP-0016 — Cross-font evaluation analysis of frozen held-out and familiar predictions

- **Date:** 2026-10-21
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** What does the frozen M7B held-out-font evidence say
  about per-font performance, corruption families, geometric severity,
  source-level error concentration, uncertainty and familiar-versus-held-out
  differences, and which claims does it support? M7C is a read-only analysis. It
  does not train, refit, recalibrate, select, tune, generate a dataset, change a
  target, change a split, or write into any prior dataset/model/report
  directory. The target remains synthetic SSIM against an exact source-specific
  clean reference; this is not human readability, CAPTCHA solvability, security
  or production robustness.
- **Code revision:** base `6631df5` plus the M7C implementation, committed as
  `feat: analyze cross-font evaluation results`. The reported run was repeated
  from the clean commit so provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; joblib 1.6.0; Pillow 12.3.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/cross_font_analysis_v0_1_0.toml`; config sha256
  `e5a4ae80b079ef8e3a44537bbc9194beef863b460c7ab86d798fadbd874241f0`;
  identity hash
  `36b81eb89299bf5ae9b7d1923d951c8e6e9a7e9ff05acb95b3b49182b57b000f`.
  Seed `20261110`; 2,000 source-grouped percentile bootstrap replicates per
  scope; confidence level 0.95; near-degeneracy margin target SD `< 0.005`;
  independent unit `source_id`.
- **Upstream frozen inputs:** 31 files pinned by SHA-256, including M7B
  predictions
  `660b9e8e816b22f9da4fcfeddc6f01108e23eb2b327bf8e8549fe9014e50312d`,
  M7B manifest
  `e751468e8159d95bf39bda478509e303f95b0177c2b9df3e3212fa8439594bc0`,
  M7B uncertainty
  `9844962584611d497ff659813f8db4ee94cf2247d465bd3abea27cdabd9f4651`,
  M5C familiar predictions
  `6d286af061bd54e70b7c8fd547d8da7079731c502eb5255d848856edf2dec906`,
  M5C test metrics
  `1dd6d630ca61caeabf907a53c758031019eeef47c5b7898f501b020322c9c9e1`,
  M5C uncertainty
  `8b7606bce32872f89c8defdee42c21c85b3c763a674b3ab1e2d86d0a174145f0`,
  M5A severity manifest
  `c63be1088db591181f23fa306f955b3ba16fe80ffb2000c4a5e3db7656fa3806`
  and M7A uncertainty
  `c350233fd65c4957c622a3b9b855c2b5e81c1728e9ea399d7b77936a29fd7e5b`.
  Champion artifact sha256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`
  (`expanded_random_forest`, calibration `none`); card sha256
  `9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8`.
  The complete list is in the generated `input_manifest.csv`.
- **Dataset:** frozen M7B `heldout_font_benchmark_v0.1.0`: 60 sources (20 per
  font), 1,260 rows (1,140 primary, 120 diagnostic), 1,260 distinct images, one
  `eval` split. Familiar comparison: frozen M5C test predictions from 12 M3
  test sources.
- **Split:** M7B one untouched `eval` split with 60 source groups; M5C 12
  familiar test groups. Font groups are disjoint and unpaired.
- **Seeds:** M7C seed `20261110`; bootstrap seeds are derived per scope/font
  with `derive_noise_seed(seed, "m7c_bootstrap", scope, font_id)` and per
  contrast with separate `m7c_paired` / `m7c_font_difference` /
  `m7c_benchmark_*` namespaces.
- **Runtime:** recorded in `run_metadata.json`; two clean-commit runs differ
  only in `elapsed_seconds`.

**Method.** The run verifies all 31 pinned files and their schemas, then
reproduces the frozen M7B and M5C metrics from the stored per-row predictions
and targets. It computes all-primary, per-font, per-family, per-severity and
per-source MAE/RMSE/R², target statistics, residuals and near-degeneracy flags.
It runs 2,000 whole-source bootstrap replicates for every declared scope and
font, preserving all rows of a sampled source. Paired within-source contrasts
use the same sampled source on both sides for Gaussian noise versus blur, thin
lines versus arcs, interference versus original blur/noise, geometric versus
original blur/noise and adjacent geometric severities. Font differences and
familiar-versus-held-out comparisons are unpaired and descriptive. It builds a
13-claim evidence table, writes deterministic CSV/JSON/Markdown outputs and
provides `run` and `verify` commands. It fingerprints 360 protected non-image
files before and after the run and refuses to write outside the Git-ignored
report tree or into any prior report directory.

**Metrics.**

| Metric | Value | 95% source-grouped CI | Notes |
| --- | --- | --- | --- |
| All-primary MAE / RMSE / R² | 0.063427 / 0.092952 / 0.743537 | MAE [0.059244, 0.067633]; RMSE [0.087121, 0.098730] | 1,140 rows, 60 sources, 1,140 images |
| Georgia all-primary MAE / RMSE / R² | 0.056859 / 0.085198 / 0.743928 | MAE [0.052435, 0.061945]; RMSE [0.077809, 0.092987] | 20 sources |
| Monaco all-primary MAE / RMSE / R² | 0.052758 / 0.076126 / 0.799809 | MAE [0.047722, 0.057987]; RMSE [0.069726, 0.082745] | 20 sources |
| Impact all-primary MAE / RMSE / R² | 0.080664 / 0.113431 / 0.705511 | MAE [0.075569, 0.085411]; RMSE [0.106749, 0.119681] | 20 sources |
| Original blur/noise RMSE, Georgia / Monaco / Impact | 0.027892 / 0.034646 / 0.160079 | Georgia [0.021698, 0.033616]; Monaco [0.026474, 0.041292]; Impact [0.144685, 0.173375] | familiar M5C RMSE 0.023328 |
| Interference RMSE, Georgia / Monaco / Impact | 0.123919 / 0.107367 / 0.056379 | descriptive per-font intervals | per-font reversal; Impact R² 0.086387 |
| Primary geometric RMSE, Georgia / Monaco / Impact | 0.054292 / 0.052548 / 0.068389 | descriptive per-font intervals | familiar 0.059006 |
| Geometric severity RMSE, 1 / 2 / 3 px | 0.053384 / 0.025868 / 0.082867 | descriptive source-grouped intervals | all-font; target SD 0.011393 / 0.030939 / 0.046608 |
| All-primary top-1 / top-3 source share | 0.034765 / 0.099141 | descriptive | 60 sources; effective sources 48.409 of 60 |
| Source count for 50% / 80% of SSE | 19 / 38 | descriptive | broad distribution |
| Claim audit | 3 supported / 0 suggestive / 10 unsupported | not a statistical test | 13 predeclared claims |

Paired RMSE contrasts (all fonts pooled): Gaussian noise minus blur is positive
by +0.021254 [0.013953, 0.027877]; thin lines minus arcs +0.023165 [0.015263,
0.031223]; interference minus original blur/noise +0.004181 [-0.019616,
0.029927]; geometric primary minus original blur/noise -0.037084 [-0.052294,
-0.020216]; 2 px minus 1 px -0.027517 [-0.033407, -0.021562]; 3 px minus 2 px
is positive by +0.057000 [0.046016, 0.067258]; 3 px minus 1 px +0.029483
[0.018276, 0.041545].
All-primary RMSE font differences are Georgia-minus-Monaco +0.009072
[-0.000708, 0.018519], Georgia-minus-Impact -0.028233 [-0.038275, -0.018025]
and Monaco-minus-Impact -0.037305 [-0.046567, -0.027865]. The familiar
blur/noise comparison is 0.095923 versus 0.023328, geometric primary is 0.058839
versus 0.059006, clean is 0.014155 versus 0.002275 and the 0.5 px diagnostic is
0.038513 versus 0.035917. The 0.5 px diagnostic target SD is 0.003376, below
the 0.005 margin, so its R² is suppressed.

**Observations.** The frozen champion reproduces the M7B headline metrics. Error
is not uniform across fonts: Impact is worse on all-primary and especially on
blur/noise, while Georgia and Monaco are comparatively strong; the Georgia–Monaco
all-primary interval includes zero, so only the two Impact comparisons clearly
separate on the observed sources. Blur/noise direction is consistent (all three
fonts above the familiar point estimate), but the magnitude is very uneven
(max/min 5.74), so a homogeneity claim is not supported. The geometric aggregate
is almost identical to the familiar benchmark (0.058839 versus 0.059006), but
severity RMSE is non-monotonic: 2 px is the strongest rung (0.025868), 1 px is
0.053384 and 3 px is the weakest (0.082867), a 3.20× range. Error is distributed
broadly across the 60 sources: the top-3 share is 0.099, the effective source
count is 48.409 and 19 sources account for 50% of squared error. The paired
contrasts show clear within-source differences for noise versus blur, thin lines
versus arcs, geometric versus original blur/noise and adjacent geometric
severities; the interference-versus-original interval includes zero.

**Conclusion.** The evidence supports three descriptive claims: a per-font
all-primary difference for at least one pair on the observed sources, a
blur/noise direction in all three held-out fonts relative to the familiar point
estimate, and aggregate geometric performance concealing severity-specific
differences. It does not support arbitrary unseen fonts, unseen
renderers/resolutions/formats, human readability or solve rates, security or
production robustness, a causal font effect, geometric equivalence from close
aggregate RMSE, stable 0.5 px R² or ranking, or font-population generalisation.
The familiar comparison is unmatched in fonts, source composition/counts and
partially corruption mix, so it is not a causal effect estimate. No significance
testing is performed; intervals are descriptive uncertainty summaries.

**Limitations.** Three purposively chosen typefaces are not a random font
sample; only one renderer, one 192x64 canvas, one grayscale PNG format and the
existing corruption definitions are represented; the 60 sources are synthetic
strings; font groups have 20 sources each and are unpaired; the familiar
benchmark has 12 source groups; the clean target is constant and the 0.5 px rung
is near-degenerate; and source-grouped intervals do not represent human users,
deployments or a font population. A difference interval that excludes zero is
not proof of a causal effect, and the results do not justify a CNN or a
production claim.

**Next step.** Stop for review. A future version could add held-out renderers,
resolutions, image formats or a designed font sample, but that is a separate
milestone and is not justified by this result alone.

**Artifacts.** Git-ignored report directory
`reports/cross_font_analysis_v0_1_0/` (input manifest, per-font/family/severity/
source tables, concentration, uncertainty, paired and font-difference tables,
benchmark comparison, claim evidence, isolation checks, protocol, Markdown
report and run metadata). The committed config, code, tests and documentation
are the reproducible protocol.

---

### EXP-0015 — Held-out-font generalisation of the frozen M5C champion

- **Date:** 2026-10-21
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** How well does the frozen M5C champion predict SSIM
  on text rendered in fonts that were absent from model development? M7B
  implements the M7A recommendation as a separate versioned evaluation dataset.
  It changes no M3–M7A dataset, target, split, model, report or config. The
  target remains SSIM against the exact source-specific clean reference; PSNR is
  diagnostic only. This is synthetic image similarity, not human readability,
  CAPTCHA solvability, security or production robustness.
- **Code revision:** base `ae0baba` plus the M7B implementation, committed as
  `feat: benchmark held-out font generalization`. The reported run was repeated
  from the clean commit so provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; joblib 1.6.0; Pillow 12.3.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/heldout_font_benchmark_v0_1_0.toml`; config sha256
  `5eb97fc6bc0ed1e7c2b17a9e8cc6f53ab89e529d17496a800e9b6c0b73a041ee`;
  identity hash
  `391642bc5f3d14c43791b8c299c79f275886d6ea387c2dde525c988be7bd100b`;
  generation hash
  `21ea470eee5d4b3697af685b939eb3f7fb4306fd887bb42d86811f3f3e95cb29`.
  Seed `20261109`; 2,000 source-grouped percentile bootstrap replicates per
  declared scope; confidence level 0.95; near-degeneracy margin target SD
  `< 0.005`; evidence rule
  `heldout_point_outside_familiar_interval`.
- **Upstream frozen inputs:** M3 config sha256
  `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`,
  M3 sources sha256
  `07d50255cdc82ea77087fc64b398d62db49f96156eea9cf5aee59bc070ab5a37`;
  M5C config sha256
  `48f3f148121537ea53a685bda961f01e758e5ac196b89b9e0181587d545086eb`,
  test metrics sha256
  `1dd6d630ca61caeabf907a53c758031019eeef47c5b7898f501b020322c9c9e1`,
  test predictions sha256
  `6d286af061bd54e70b7c8fd547d8da7079731c502eb5255d848856edf2dec906`
  and uncertainty sha256
  `8b7606bce32872f89c8defdee42c21c85b3c763a674b3ab1e2d86d0a174145f0`.
  Champion artifact sha256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`
  (`expanded_random_forest`, calibration `none`); card sha256
  `9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8`.
- **Held-out fonts:** Georgia
  (`/System/Library/Fonts/Supplemental/Georgia.ttf`, sha256
  `4f54eb299fccea7f103edeb0d92437359bfd4441811d53222b82b335369f6218`),
  Monaco (`/System/Library/Fonts/Monaco.ttf`, sha256
  `c6e990f6429fae28eb85321375dd335252f1122aa64c19fec553619a140724ef`)
  and Impact (`/System/Library/Fonts/Supplemental/Impact.ttf`, sha256
  `57e95627fd36fe2740e5ee8a91ff1ae74fb216df9580bd23e30d6772814727be`).
  None is in M3; font files are local and not redistributed.
- **Dataset:** `heldout_font_benchmark_v0.1.0`; 60 sources (20 per font), 1,260
  rows, 1,260 distinct images, one `eval` split, 21 rows per source. Manifest
  sha256
  `e751468e8159d95bf39bda478509e303f95b0177c2b9df3e3212fa8439594bc0`;
  sources sha256
  `ad50ed063552ed3b00b7562944602e3ed6b58cc17d4785488315140e1f1d3702`;
  features sha256
  `5b0ff62b8d37a8738f0b22026f43497629431833ca02cdab1ad5ddaffb4eb011`;
  summary sha256
  `8cf5d3281a0353182c81689bb26ba94f29685239e9920f3b4fcd8d229cfae569`.
- **Split:** one untouched `eval` split; 60 source groups (20 per font); the
  source is the independent unit and all variants of a source stay together.
- **Seeds:** dataset seed `20261109`; bootstrap seeds are derived per scope
  with `derive_noise_seed(seed, "m7b-bootstrap", scope)`.
- **Runtime:** recorded in `run_metadata.json`; two clean-commit runs differ
  only in `runtime_seconds`.

**Method.** The run verifies the three pinned fonts, the M3/M5C upstream hashes
and the champion artifact/card; generates one clean reference plus 20 corrupted
variants per source (blur, noise, thin lines, arc strokes and elastic 0.5/1/2/3
px); computes SSIM against each source-specific clean reference and PSNR as a
diagnostic; extracts the frozen eight features; and evaluates the serialized
champion without fitting or calibration. It checks that all 1,260 images are
distinct, no source or font overlaps M3, the dataset contains no model artifact,
and the protected M3–M7A asset tree is unchanged before and after. It writes
per-font, per-family, per-severity, per-source and target-distribution tables,
source-grouped bootstrap intervals, an explicit familiar-font comparison, a
claim/limitation report and a run/verify CLI.

**Metrics.**

| Metric | Value | 95% source-grouped CI | Notes |
| --- | --- | --- | --- |
| All primary MAE / RMSE / R² | 0.06343 / 0.09295 / 0.74354 | MAE [0.05917, 0.06781]; RMSE [0.08668, 0.09860] | 1,140 rows, 60 sources |
| Clean-reference MAE / RMSE | 0.00940 / 0.01416 | MAE [0.00684, 0.01221]; RMSE [0.01065, 0.01762] | target constant 1.0; R² undefined |
| Original blur/noise MAE / RMSE / R² | 0.06008 / 0.09592 / 0.76136 | MAE [0.04648, 0.07503]; RMSE [0.07797, 0.11199] | 480 rows; familiar RMSE 0.02333 |
| Gaussian blur RMSE / R² | 0.08471 / 0.85986 | RMSE [0.06892, 0.10011] | 240 rows |
| Gaussian noise RMSE / R² | 0.10596 / 0.17960 | RMSE [0.08408, 0.12448] | 240 rows |
| Interference MAE / RMSE / R² | 0.07273 / 0.10010 / -0.17439 | MAE [0.06346, 0.08205]; RMSE [0.09012, 0.10964] | 480 rows |
| Thin lines RMSE / R² | 0.11101 / -0.28206 | RMSE [0.09948, 0.12151] | 240 rows |
| Arc strokes RMSE / R² | 0.08785 / -0.06902 | RMSE [0.07804, 0.09752] | 240 rows |
| Primary geometric MAE / RMSE / R² | 0.04757 / 0.05884 / 0.26573 | MAE [0.04396, 0.05126]; RMSE [0.05396, 0.06358] | 180 rows; familiar RMSE 0.05901 |
| 0.5 px elastic MAE / RMSE | 0.03246 / 0.03851 | MAE [0.02704, 0.03790]; RMSE [0.03286, 0.04408] | diagnostic; target SD 0.00338 |
| Georgia / Monaco / Impact primary RMSE | 0.08520 / 0.07613 / 0.11343 | descriptive per-font intervals | 20 sources each |
| Georgia / Monaco / Impact blur/noise RMSE | 0.02789 / 0.03465 / 0.16008 | descriptive per-font intervals | Impact R² 0.46247 |
| Georgia / Monaco / Impact interference RMSE | 0.12392 / 0.10737 / 0.05638 | descriptive per-font intervals | R² -0.55230 / -0.37669 / 0.08639 |

**Observations.** The held-out primary slice is clearly worse than the familiar
M5C benchmark on the original blur/noise task: RMSE 0.09592 versus 0.02333, and
both MAE and RMSE point estimates lie outside the familiar-font source-grouped
intervals. The effect is strongly font-dependent: Georgia and Monaco remain
comparatively strong on blur/noise, while Impact is much worse (RMSE 0.16008,
R² 0.46247). Aggregate primary geometric error is similar to the familiar
benchmark (RMSE 0.05884 versus 0.05901), but the ladder is uneven: the 1 px rung
has R² -20.96 on very small target variance, the 2 px rung is strong (RMSE
0.02587, R² 0.30098), and the 3 px rung is worse than the familiar benchmark
(0.08287 versus 0.07784). Thin-line and arc interference are new
out-of-distribution slices and are poorly calibrated (combined R² -0.17439);
thin lines are worse than arcs. Clean-reference error is small in absolute terms
but nonzero, and is largest for Impact (RMSE 0.02084). All 1,260 images are
distinct; the 0.5 px elastic target SD is 0.00338, below the 0.005 margin, so it
is diagnostic only; no font or source overlaps M3; and the frozen-asset
fingerprints are unchanged.

**Conclusion.** The evidence supports a descriptive statement that zero-shot
SSIM prediction by the frozen M5C champion is worse on these three unseen fonts,
most clearly for the original blur/noise task and for interference, while
aggregate geometric error is similar. It does not estimate a font population,
identify a causal font property, prove failure on arbitrary fonts, or support
claims about human readability, CAPTCHA solvability, security or production
robustness. The familiar comparison is not matched: it differs in font, source
count, source composition and corruption mix, so the differences are not causal
effect estimates or significance tests.

**Limitations.** Three purposively chosen typefaces are not a random font
sample; only one renderer, one 192x64 canvas, one grayscale PNG format and the
existing corruption definitions are tested; sources are synthetic strings, so
source-grouped intervals do not represent human users or deployments; the clean
target is constant and the 0.5 px rung is near-degenerate; and the familiar
benchmark has only 12 source groups while M7B has 60. The comparison is
descriptive, and no result here justifies a CNN or a production claim.

**Next step.** Stop for review. A future version could add held-out renderers,
resolutions or image formats, but that is a separate milestone and is not
justified by this result alone.

**Artifacts.** Git-ignored dataset
`data/processed/heldout_font_benchmark_v0.1.0/` and reports
`reports/heldout_font_benchmark_v0_1_0/` (overall, per-font, family, source,
target-distribution, uncertainty, comparison, isolation and prediction tables;
`run_metadata.json`; Markdown report). The committed config, code, tests and
documentation are the reproducible protocol.

---

### EXP-0014 — Dataset-adequacy and evaluation audit: what the frozen benchmark can and cannot support

- **Date:** 2026-10-21
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Does the frozen synthetic dataset and evaluation
  design support the claims made for the M5C champion? Where do champion errors
  concentrate, which target slices are informative, and which conclusions
  require additional independent data? M7A is a read-only audit, not a new model
  or dataset. It changes no target, split, corruption parameter, feature or
  champion. The target is SSIM against the exact clean reference; nothing here
  measures human readability, CAPTCHA solvability, security or production
  robustness.
- **Code revision:** base `c2bfb40` plus the M7A implementation, committed as
  `feat: audit dataset adequacy and evaluation`. The reported run was repeated
  from the clean commit so provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; joblib 1.6.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/dataset_adequacy_audit_v0_1_0.toml`; config sha256
  `23f2af7ede523bec7e648043a519e0eca7d352bbd9f15445255a3a304b0364b9`;
  identity hash
  `6a1a7989cccd24dab9cc1365a5d308d53da29e31df2a20bbb8bd30639e705a48`.
  Seed `20261108`; 2000 source-grouped percentile bootstrap replicates per
  declared scope; confidence level 0.95.
- **Upstream frozen inputs:** M3 config sha256
  `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`,
  manifest sha256
  `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`;
  M5A config sha256
  `00b8c5caad610fb4aaf159bb8c99e329ad217aeff6450e7cd3e566387e9f0a97`,
  manifest sha256
  `c63be1088db591181f23fa306f955b3ba16fe80ffb2000c4a5e3db7656fa3806`;
  M5B config sha256
  `fe5c66614cee83c72a80c20b807202deadd8a604e045f7432795a8abe706ddb1`,
  protocol hash
  `77e41dd5de84e0bf8751c38a49b510735095f86edc8b00f6c19a6d8fde0d13ce`;
  M5C config sha256
  `48f3f148121537ea53a685bda961f01e758e5ac196b89b9e0181587d545086eb`,
  protocol hash
  `048b6d990749ca27358ce4c0d0ccb9a91cc0400e1da89ea48bea6661465e3e1a`;
  M5C champion artifact sha256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`
  (`expanded_random_forest`, no calibration).
- **Dataset:** frozen M3 `captcha_quality_v0.1.0` (540 rows / 60 sources) and
  M5A `geometric_distortion_pilot_v0.1.0` (480 rows / 60 sources). M5A is
  derived from the same sources, not an independent sample. M3 has 540 distinct
  image files; M5A has 480 rows but 360 distinct images because identity and
  zero-field controls are exact clean duplicates and the minimal control is
  near-clean.
- **Split:** frozen source-grouped `train/val/test` = 36/12/12 sources. M3 rows
  are 324/108/108; M5A rows are 288/96/96. Expanded fitting uses 432 train and
  144 validation rows. Primary geometric test is 36 rows; the 0.5 px diagnostic
  is 12 rows; the M3 test is 108 rows. The independent unit is the **12 test
  source groups**, not the 36 primary rows or 108 M3 rows.
- **Seeds:** audit seed `20261108`; bootstrap seeds are derived per scope from
  that seed with `derive_noise_seed(seed, "dataset_audit_bootstrap", scope)`.
- **Runtime:** 2.77 s wall-clock for the full audit with pixel-level image
  verification; the two clean-commit runs differed only in
  `runtime_seconds`.

**Method.** The audit pins 24 upstream files by SHA-256, loads the frozen M3 and
M5A frames through the existing M5B/M5C verified path, and replays the champion's
stored test predictions from the serialized model. It verifies that stored
test-slice metrics match the replay. It then builds:

1. source-level inventory and long-form composition tables, including fonts,
   text lengths, resolution, pixel format, renderer, corruption families,
   severities and derived variants;
2. split accounting for M3, M5A, expanded fitting and evaluation scopes;
3. target distributions for clean, original, every original severity, primary
   geometric severities 2–4 and the 0.5 px diagnostic, with quantiles,
   variance and a descriptive between/within-source decomposition;
4. per-source and per-severity champion residuals, per-font residuals, squared
   error concentration and descriptive leave-one-source-out influence;
5. source-grouped percentile bootstrap intervals that keep every variant of a
   sampled source together;
6. training-only feature diagnostics (constants, duplicates, correlations,
   scales, frozen-model impurity importance, VIF/condition number and
   family/severity means); and
7. a claim-to-evidence table and a comparison of four predeclared next
   directions.

No model was refit, no calibration was fitted, no feature was selected and no
test target entered a training-only diagnostic. The audit writes only to the
Git-ignored `reports/dataset_adequacy_audit_v0_1_0/` directory. It verifies 1766
frozen-asset fingerprints before and after the run and refuses to write into any
frozen path.

**Metrics.**

| Metric | Value | Notes |
| --- | --- | --- |
| M3 test MAE / RMSE / R² | 0.01562 / 0.02201 / 0.98716 | 108 rows, 12 sources |
| Clean-reference RMSE | 0.00227 | target constant at 1.0; R² undefined |
| Original blur/noise RMSE | 0.02333 | 96 rows, 12 sources |
| Primary geometric RMSE / R² | 0.05901 / 0.18553 | 36 rows, 12 sources |
| Geometric severity 2 / 3 / 4 R² | -44.10 / 0.21288 / -2.74628 | low target variance at the mildest and hardest rungs |
| 0.5 px diagnostic RMSE / R² | 0.03592 / -197.41 | target SD 0.00266 below the 0.005 margin |
| Geometric top-1 source SSE share | 0.17667 | descriptive concentration |
| Geometric top-3 source SSE share | 0.42704 | descriptive concentration |
| Maximum training feature \|r\| | 0.99106 | `intensity_std` vs `contrast_rms` |
| Maximum VIF / condition number | 518.5 / 62.15 | redundancy warning, not a selection rule |
| Top frozen-model importance | 0.63345 (`histogram_entropy`) | impurity importance, descriptive |
| Claims | 3 supported, 2 suggestive, 7 unsupported | predeclared schema |
| Recommended direction | **B** | add explicitly held-out fonts or rendering conditions |

The target-distribution table includes counts, min/max/mean/SD, 5/25/50/75/95
quantiles, variance, between-source variance, mean within-source variance and
the descriptive between-variance share for every declared slice. The clean
target variance is exactly zero. The 0.5 px diagnostic is flagged
near-degenerate by the predeclared 0.005 SD margin. The primary geometric
between-source share is about 0.113, so most of its variance is within source;
this is why source-level ranking is unstable even when aggregate RMSE is
reported.

**Observations.** The dataset has 60 sources, but only 12 test sources and one
renderer, one resolution (192x64), one pixel format (8-bit grayscale PNG) and
three fonts. Every font appears in every split, so the champion is never
evaluated on an unseen font. M5A's elastic variants are derived from the same
clean references as M3; identity and zero-field controls are exact clean
duplicates. The audit therefore treats source groups, not image variants, as
independent observations.

The champion reproduces the pinned M5C test values. Its M3 blur/noise and
clean-reference performance is strong, but the primary geometric slice is
weak and uneven. The top geometric source contributes 17.7% of squared error
and the top three contribute 42.7%. Leave-one-source-out RMSE changes are
small and descriptive; no model was refit and no new generalisation estimate is
claimed. The 0.5 px diagnostic has near-zero target variance, so its very
negative R² is a denominator artefact rather than evidence of a stable
predictive failure.

The eight global features are strongly redundant on the 432 training rows:
`intensity_std` and `contrast_rms` have |r| 0.991, `laplacian_variance` and
`noise_sigma_immerkaer` have |r| 0.956, and two pairs cross the predeclared
0.90 redundancy threshold. `histogram_entropy` dominates impurity importance
(0.633), but correlated features share importance and the ranking is not causal.
Feature means also differ by family and severity, so global scales are not
stationary across the corruption mix.

**Claim audit.** The current benchmark supports three claims: (C01) the champion
predicts SSIM on the current synthetic blur/noise benchmark; (C08) the M5C
champion improves geometric RMSE versus frozen M3 without a material clean or
original-task regression on this benchmark; and (C10) geometric error is
unevenly concentrated across the 12 test sources. Two claims are suggestive:
(C02) the champion predicts 1–3 px elastic SSIM and (C03) it generalises to
unseen source images from the same rendering setup; both are weakened by low
aggregate R², negative severity R² and only 12 groups. Seven claims are
unsupported: unseen fonts, unseen renderers/resolutions/formats, real-world
CAPTCHA quality, human readability/solve rate/security, uniform feature
adequacy across primary severities, stable 0.5 px R² and broad generalisation
beyond the synthetic distribution.

**Conclusion.** The benchmark is adequate for a **condition-specific**
synthetic SSIM evaluation and for the M5C within-benchmark champion comparison.
It is not adequate for unseen-font, unseen-renderer, real-world, human-outcome
or broad-generalisation claims. The binding limitations are condition coverage
and 12 independent test groups, not the absence of a more complex model. The
recommended next direction is **B: add explicitly held-out fonts or rendering
conditions in a future versioned dataset**. It directly tests the largest
unsupported claim, preserves the target and champion, and can be evaluated
zero-shot with source-grouped intervals. Direction A would improve precision
but not condition generalisation; C changes the research question; D risks
overfitting the same 12 test groups without a concrete missing representation.
No direction is executed in M7A.

**Limitations.**

- Only 12 independent test source groups; bootstrap intervals are wide and
  descriptive, not significance tests.
- All fonts, the renderer, the resolution and the pixel format are shared
  across splits. There is no unseen-condition test.
- The target is synthetic SSIM against a known clean reference, not a human
  readability, solve-rate or security outcome.
- The 0.5 px rung is near-degenerate and must remain diagnostic only.
- Feature importance is impurity-based and descriptive; correlated features
  share importance and the ranking is not causal.
- Leave-one-source-out results are descriptive influence measures, not refitted
  generalisation estimates.
- The audit reuses frozen artifacts and cannot repair upstream limitations such
  as the single renderer or the small source count.

**Next step.** Design a future versioned dataset with at least one explicitly
held-out font and/or a second rendering condition, predeclare the zero-shot
evaluation and source-grouped intervals, and keep the target definition
unchanged. Do not run it as part of M7A.

**Artifacts.** Git-ignored:
`reports/dataset_adequacy_audit_v0_1_0/report.md`, `run_metadata.json`, and 20
CSV tables including `dataset_inventory.csv`, `dataset_composition.csv`,
`split_accounting.csv`, `target_distributions.csv`, `test_slice_metrics.csv`,
`source_metrics.csv`, `source_severity_metrics.csv`,
`leave_one_source_out.csv`, `error_concentration.csv`, `uncertainty.csv`,
`feature_summary.csv`, `feature_correlations.csv`, `feature_importance.csv`,
`claim_evidence.csv`, `next_direction_comparison.csv` and
`isolation_checks.csv`.

---

### EXP-0013 — Multi-scale structure-tensor features: no gain beyond the global eight

- **Date:** 2026-10-21
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Do compact multi-scale structure-tensor features
  computed from the corrupted image alone improve geometric-distortion SSIM
  prediction beyond the existing eight global image features while preserving
  clean-reference and original-task performance? M6C found that whole-image
  gradient-magnitude and orientation summaries did not qualify under the
  retention rule and pointed to a representation explicitly sensitive to local
  *displacement*. M6D tests that handcrafted follow-up directly. This is a
  synthetic image-similarity benchmark, not a human-readability,
  CAPTCHA-solvability or security study.
- **Code revision:** base `715f7b0` plus the M6D implementation, committed as
  `feat: evaluate multi-scale structure tensor features`. The reported run was
  repeated after the commit so generated provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; scikit-image 0.26.0; joblib 1.6.0; macOS 27.0.1 (arm64). No GPU was
  used.
- **Configuration:** `configs/structure_tensor_study_v0_1_0.toml`; config
  sha256
  `1cab9a9eef6b9585fc2047051f43d5092389010d11b810b72257140aa5797134`;
  protocol hash
  `219330344062f0be31994a0fba315d035eaa0c4b7c5272d770a4dec3630db8bb`.
  Seed `20261107`; 1000 source-grouped bootstrap replicates.
- **Upstream frozen inputs:** M5C config sha256
  `48f3f148121537ea53a685bda961f01e758e5ac196b89b9e0181587d545086eb`, M5C
  protocol hash
  `048b6d990749ca27358ce4c0d0ccb9a91cc0400e1da89ea48bea6661465e3e1a`, M5C
  champion artifact sha256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`; M6A
  config sha256
  `e72a9e34926573b959959b8e62956cb6070a5c4564c05357d1e1d6b65266e909`, M6A
  protocol hash
  `d744248e0ebde54f83a005e05dd1a37bfdaa297a34e8d9f1299514e613e89bc9`,
  spatial feature set `spatial-v1`; M6B config sha256
  `9376ecef3e5d66dc7b8a7415f615de28330a02471c0476d4cf231d8ebe01a8a0`, M6B
  protocol hash
  `fafd818694cafff51ae5129453592a1793d754bc312248a7f843cb9b5013b152`; M6C
  config sha256
  `8ab83394bc07bcb3f372d71cad669dddab739dd18b0ae151175f4da789f61934`, M6C
  protocol hash
  `7c436dc5e179d9efb309bdb2b749ca1fd53fdc29e1bc4963765c7babb615c3d1`.
  The M3 dataset, split and model, the M4A pilot, the M4B benchmark, the M5A
  pilot, the M5B benchmark, the M5C study, the M5D study, the M6A study, the M6B
  study and the M6C study are unchanged and re-verified from disk. The M6C
  gradient summaries are **not** used by any M6D candidate.
- **Split:** the frozen M3 source-grouped assignment, reused verbatim through
  the M5C verified path (assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`):
  36 train / 12 validation / 12 test sources. Expanded training 432 rows /
  36 sources; expanded validation 144 rows / 12 sources; primary geometric test
  36 rows / 12 sources; diagnostic 0.5 px test 12 rows / 12 sources; M3 test
  108 rows / 12 sources.
- **Runtime:** 9.93 s in process on the recorded machine (not a benchmark); it
  varies slightly between runs.

**Method.** Nine structure-tensor features are computed from the corrupted
grayscale image alone at exactly two predeclared Gaussian scales,
`sigma = 1.0` and `sigma = 3.0` px. The image is scaled to `[0, 1]`;
horizontal and vertical gradients use scikit-image's fixed 3x3 Sobel operator
with reflected boundaries and no extra normalisation. The tensor components
`Jxx = gx*gx`, `Jyy = gy*gy` and `Jxy = gx*gy` are smoothed by a Gaussian with
the declared scale, reflected boundaries and truncation at four standard
deviations.

The symmetric 2x2 tensor eigenvalues are

```text
lambda_1 = (trace + sqrt((Jxx - Jyy)^2 + 4*Jxy^2)) / 2
lambda_2 = (trace - sqrt((Jxx - Jyy)^2 + 4*Jxy^2)) / 2
trace    = Jxx + Jyy
```

with `lambda_2` clipped to `[0, lambda_1]` for round-off. Coherence is
`(lambda_1 - lambda_2) / (lambda_1 + lambda_2 + epsilon)`, in `[0, 1]`.
Anisotropy is `1 - lambda_2 / (lambda_1 + epsilon)`, in `[0, 1]`. Tensor energy
is `lambda_1 + lambda_2`, non-negative and unbounded above in squared-gradient
units. `epsilon = 1e-12` is fixed; a zero-energy tensor is an explicit zero, not
a division by zero. Quantiles use NumPy's linear interpolation.

Four summaries are reported at each scale: mean coherence, the 0.90 coherence
quantile, mean anisotropy and the 0.90 energy quantile. Mean tensor energy is
reported once as `st_energy_mean` at the finest scale, because Gaussian
smoothing preserves the mean and repeating it at both scales would duplicate a
column. The nine features, in fixed order, are `st_s1_coh_mean`,
`st_s1_coh_p90`, `st_s1_aniso_mean`, `st_s1_energy_p90`, `st_s2_coh_mean`,
`st_s2_coh_p90`, `st_s2_aniso_mean`, `st_s2_energy_p90` and `st_energy_mean`.
Every output is finite and deterministic. An exactly uniform image has zero
gradient everywhere and therefore produces nine defined zeros. A single faint
step produces finite, bounded summaries.

The **no-reference boundary** is strict and structural. The extractor signature
is `extract_structure_tensor_features(image, spec=None)`: the only required
input is the corrupted 2-D `uint8` image array, and there is no parameter for a
clean reference, target, source identifier, family, severity, font, split or
path. Every run checks the signature and scans the feature module's code for
forbidden identifiers; tests repeat both checks independently. The extractor
imports only the shared image-contract validator from the repository, uses no
dynamic import and performs no file I/O. The clean reference is used only by the
separate evaluation pipeline that computes the SSIM label.

Three candidates are predeclared. The frozen M5C champion is loaded without
refitting; the structure-only and global-plus-structure candidates are refitted
on the same 432 expanded training rows with the same model family,
hyperparameters, seed and objective (random forest, 300 trees, unlimited depth,
leaf size 1, seed 798117391). No hyperparameter is tuned, no post-hoc
calibration is applied and no feature is removed in response to a diagnostic.
The M6A/M6B/M6C retention-aware selection rule is used unchanged, with the same
coefficients and tolerances, and the frozen champion is the fallback. The
0.5 px rung is diagnostic only.

**Validation results by slice (selection evidence).**

| candidate | features | slice | MAE | RMSE | R2 | mean residual | residual SD |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `global_baseline` | 8 | geometric primary | 0.04196 | 0.04925 | 0.28670 | +0.00224 | 0.04989 |
| `global_baseline` | 8 | clean reference | 0.00127 | 0.00223 | n/a | +0.00127 | 0.00192 |
| `global_baseline` | 8 | original blur/noise | 0.01589 | 0.02007 | 0.98660 | +0.00121 | 0.02014 |
| `global_baseline` | 8 | overall | 0.02119 | 0.02959 | 0.97701 | +0.00147 | 0.02965 |
| `structure_only` | 9 | geometric primary | 0.03909 | 0.05189 | 0.20796 | -0.01375 | 0.05075 |
| `structure_only` | 9 | clean reference | 0.01030 | 0.01540 | n/a | +0.01030 | 0.01196 |
| `structure_only` | 9 | original blur/noise | 0.02417 | 0.03315 | 0.96344 | -0.00286 | 0.03320 |
| `structure_only` | 9 | overall | 0.02674 | 0.03776 | 0.96256 | -0.00449 | 0.03762 |
| `global_plus_structure` | 17 | geometric primary | 0.03913 | 0.04935 | 0.28372 | -0.00667 | 0.04959 |
| `global_plus_structure` | 17 | clean reference | 0.00240 | 0.00516 | n/a | +0.00240 | 0.00477 |
| `global_plus_structure` | 17 | original blur/noise | 0.01429 | 0.01904 | 0.98793 | +0.00045 | 0.01914 |
| `global_plus_structure` | 17 | overall | 0.01951 | 0.02920 | 0.97760 | -0.00117 | 0.02928 |

`global_plus_structure` breaches the 50% clean-reference tolerance (validation
RMSE 0.00516 versus 0.00223). `structure_only` breaches the 50% clean-reference
tolerance (0.01540 versus 0.00223) and the 10% original-task tolerance (0.03315
versus 0.02007). The per-severity constraints are satisfied by both new
candidates on validation: severity 2 RMSE 0.04687 and 0.04742, severity 3 RMSE
0.02238 and 0.02852, severity 4 RMSE 0.06789 and 0.07084, all within 25% of the
champion's 0.05254, 0.02494 and 0.06240. The aggregate geometric validation
RMSE is 0.04925 for the champion, 0.04935 for the combined candidate and 0.05189
for the structure-only candidate. The champion is retained.

**Test results by slice (after selection was frozen).**

| candidate | slice | MAE | RMSE | R2 | mean residual | residual SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `global_baseline` | geometric primary | 0.04928 | 0.05901 | 0.18553 | -0.00459 | 0.05966 |
| `global_baseline` | clean reference | 0.00127 | 0.00228 | n/a | +0.00127 | 0.00197 |
| `global_baseline` | original blur/noise | 0.01741 | 0.02333 | 0.98141 | +0.00479 | 0.02295 |
| `global_baseline` | M3 test overall | 0.01562 | 0.02201 | 0.98716 | +0.00440 | 0.02166 |
| `structure_only` | geometric primary | 0.05213 | 0.06532 | 0.00199 | -0.01610 | 0.06420 |
| `structure_only` | clean reference | 0.01172 | 0.01822 | n/a | +0.01172 | 0.01457 |
| `structure_only` | original blur/noise | 0.02741 | 0.03879 | 0.94860 | -0.00146 | 0.03897 |
| `structure_only` | M3 test overall | 0.02566 | 0.03707 | 0.96355 | +0.00000 | 0.03724 |
| `global_plus_structure` | geometric primary | 0.04969 | 0.06036 | 0.14780 | -0.01225 | 0.05994 |
| `global_plus_structure` | clean reference | 0.00159 | 0.00328 | n/a | +0.00159 | 0.00300 |
| `global_plus_structure` | original blur/noise | 0.01704 | 0.02526 | 0.97820 | +0.00228 | 0.02529 |
| `global_plus_structure` | M3 test overall | 0.01533 | 0.02384 | 0.98492 | +0.00221 | 0.02385 |

The champion's reference metrics reproduce the pinned M5C values exactly:
geometric MAE/RMSE/R2 0.04928/0.05901/0.18553, clean RMSE 0.00228, original
RMSE 0.02333, M3 test RMSE 0.02201/R2 0.98716. Neither new candidate improves
the aggregate geometric slice. `global_plus_structure` is 0.00135 worse in RMSE
and `structure_only` is 0.00631 worse.

The geometric effect is mixed across the primary ladder:

| candidate | sev 2 RMSE | sev 3 RMSE | sev 4 RMSE |
| --- | ---: | ---: | ---: |
| `global_baseline` | 0.06187 | 0.02362 | 0.07784 |
| `structure_only` | 0.04704 (-0.01483) | 0.04319 (+0.01957) | 0.09339 (+0.01555) |
| `global_plus_structure` | 0.04293 (-0.01894) | 0.03848 (+0.01486) | 0.08721 (+0.00937) |

The combined candidate's severity-2 improvement and severity-3 regression have
paired intervals that exclude zero; its severity-4 interval includes zero. The
structure-only candidate's severity-3 and severity-4 regressions exclude zero,
while its severity-2 interval includes zero. The aggregate geometric intervals
include zero for both candidates.

**Paired source-grouped differences (candidate minus champion, RMSE).**

| slice | `structure_only` | `global_plus_structure` |
| --- | --- | --- |
| geometric primary | +0.00631 [-0.00019, 0.01181] | +0.00135 [-0.00390, 0.00575] |
| clean reference | +0.01594 [0.00648, 0.02342] | +0.00101 [-0.00026, 0.00188] |
| original blur/noise | +0.01546 [0.00398, 0.02553] | +0.00194 [-0.00240, 0.00848] |
| M3 test overall | +0.01507 [0.00567, 0.02508] | +0.00184 [-0.00219, 0.00821] |

There are 12 held-out test source groups, so these intervals are preliminary and
descriptive; no significance claim is made. The pattern is that the combined
candidate is approximately neutral on aggregate geometric error while costing a
small clean-reference regression that the validation rule already rejected, and
the structure-only candidate is clearly worse on retention.

**Training-only diagnostics.** On 432 training rows, none of the nine
structure-tensor features is constant, duplicated or non-finite. The mean
absolute pairwise correlation among the new features is 0.6439 and the maximum
is 0.9977, between `st_s2_coh_mean` and `st_s2_aniso_mean`; the energy summaries
are also strongly redundant (`st_s1_energy_p90` versus `st_s2_energy_p90`
0.9935, versus `st_energy_mean` 0.9854, and `st_s2_energy_p90` versus
`st_energy_mean` 0.9820). In the combined candidate the maximum absolute
correlation between a new feature and a global feature is 0.9387, between
`st_s1_coh_mean` and `ink_fraction`. Random-forest importance (descriptive, not
causal) gives the combined candidate a structure-tensor share of 0.1362;
`histogram_entropy` alone accounts for 0.6292, `ink_fraction` 0.0823,
`noise_sigma_immerkaer` 0.0751 and `laplacian_variance` 0.0451. The
structure-only candidate is dominated by `st_s1_coh_p90` (0.3376),
`st_s2_energy_p90` (0.2889) and `st_s1_energy_p90` (0.2837). The
training-to-validation geometric RMSE gap is 0.02645 for the champion, 0.02894
for the combined candidate and 0.03166 for the structure-only candidate, so the
larger feature sets also overfit more.

**Observations.** The structure tensor is a genuinely different representation
from the M6C gradient summaries, but it is not a different enough *source of
information*. Coherence and anisotropy are nearly interchangeable at the coarser
scale, the energy quantiles are nearly interchangeable across scales, and the
finest coherence summary is strongly correlated with `ink_fraction`. The
combined candidate's validation geometric RMSE is essentially unchanged
(0.04935 versus 0.04925) and its test geometric RMSE is slightly worse
(+0.00135, interval includes zero), while its clean-reference validation RMSE
more than doubles. The structure-only candidate is worse on every aggregate
slice. No candidate qualifies under the predeclared retention rule, and the rule
was not weakened to force a winner.

**Conclusion.** The hypothesis is not supported. Multi-scale structure-tensor
features computed from the corrupted image alone do not improve
geometric-distortion SSIM prediction beyond the eight global features while
preserving clean-reference and original-task performance. The frozen M5C
champion is retained. This is a negative result, reported as such. The
accumulated M6A-M6D evidence now covers coarse regional summaries, finer
regional grids, whole-image gradient summaries and multi-scale structure-tensor
summaries; none adds a retention-safe geometric gain. Conventional feature-based
ML remains the benchmark champion, and the evidence does not justify a CNN on
this result alone.

**Limitations.**

- One synthetic renderer, one 192x64 canvas, three fonts and 12 held-out test
  source groups. All grouped intervals are preliminary.
- Only two Gaussian scales and four summaries per scale were tested. The result
  does not rule out every possible structure-tensor configuration.
- The feature set contains near-duplicate columns by construction (coherence and
  anisotropy at the same scale, energy quantiles across scales). This limits the
  independent information available to the forest.
- The features are whole-image summaries of local tensors. They do not register
  or compare a region against its own clean reference, so they cannot directly
  measure displacement.
- SSIM is a full-reference structural proxy, not human readability or CAPTCHA
  solvability. This is not a security evaluation.
- No CNN, dashboard, deployment, external CAPTCHA dataset or additional
  corruption family is part of this milestone.

**Next step.** No follow-up is started in this milestone. M6D is the last
planned handcrafted-feature study; a learned representation would be a separate
predeclared milestone and requires review before any test-driven iteration.

**Artifacts.** Git-ignored: `reports/structure_tensor_study_v0_1_0/` (tables,
diagnostics, report, metadata) and
`models/structure_tensor_study_v0_1_0/` (refitted bundles and cards). Two runs
from the clean committed revision produced byte-identical reports, metric
tables, manifests, model bundles and model cards; only the in-process runtime
value and the recorded artifact paths differed.

---

### EXP-0012 — Local structural features: no gain beyond the global eight

- **Date:** 2026-10-21
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Do local structural features computed from a
  corrupted image alone improve geometric-distortion SSIM prediction beyond the
  existing eight global image features, while preserving clean-reference and
  original-task performance? M6B concluded that the blocker for its regional
  features was their *relevance*, not the resolution of the partition, and
  suggested a representation sensitive to local structure rather than regional
  ink quantity. M6C tests that follow-up directly. This is a synthetic
  image-similarity benchmark, not a human-readability, CAPTCHA-solvability or
  security study.
- **Code revision:** base `32f7b69` plus the M6C implementation, committed as
  `feat: evaluate local structural features`. The reported run was repeated
  after the commit so generated provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; scikit-image 0.26.0; joblib 1.6.0; macOS 27.0.1 (arm64). No GPU was
  used.
- **Configuration:** `configs/local_structure_study_v0_1_0.toml`; config sha256
  `8ab83394bc07bcb3f372d71cad669dddab739dd18b0ae151175f4da789f61934`;
  protocol hash
  `7c436dc5e179d9efb309bdb2b749ca1fd53fdc29e1bc4963765c7babb615c3d1`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates.
- **Upstream frozen inputs:** M5C config sha256
  `48f3f148121537ea53a685bda961f01e758e5ac196b89b9e0181587d545086eb`, M5C
  protocol hash
  `048b6d990749ca27358ce4c0d0ccb9a91cc0400e1da89ea48bea6661465e3e1a`, M5C
  champion artifact sha256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`; M6A
  config sha256
  `e72a9e34926573b959959b8e62956cb6070a5c4564c05357d1e1d6b65266e909`, M6A
  protocol hash
  `d744248e0ebde54f83a005e05dd1a37bfdaa297a34e8d9f1299514e613e89bc9`,
  spatial feature set `spatial-v1`; M6B config sha256
  `9376ecef3e5d66dc7b8a7415f615de28330a02471c0476d4cf231d8ebe01a8a0`, M6B
  protocol hash
  `fafd818694cafff51ae5129453592a1793d754bc312248a7f843cb9b5013b152`.
  The M3 dataset, split and model, the M4A pilot, the M4B benchmark, the M5A
  pilot, the M5B benchmark, the M5C study, the M5D study, the M6A study and the
  M6B study are unchanged and re-verified from disk. The M6B regional grids are
  **not** used by any M6C candidate.
- **Split:** the frozen M3 source-grouped assignment, reused verbatim through
  the M5C verified path (assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`):
  36 train / 12 validation / 12 test sources. Expanded training 432 rows /
  36 sources; expanded validation 144 rows / 12 sources; primary geometric test
  36 rows / 12 sources; diagnostic 0.5 px test 12 rows / 12 sources; M3 test
  108 rows / 12 sources.
- **Runtime:** 6.6 s in process on the recorded machine (not a benchmark); it
  varies slightly between runs.

**Method.** Fourteen local features are computed from the corrupted grayscale
image alone, in two predeclared families. The image is scaled to `[0, 1]` and
Sobel gradients are computed with scikit-image's normalised 3x3 operator and
reflected boundaries.

| family | features | definition |
| --- | ---: | --- |
| `gradient_magnitude` | 4 | mean, population standard deviation, median and 0.90 quantile of `hypot(gx, gy)` |
| `gradient_orientation` | 10 | eight equal-width bins over `[0, pi)` of `arctan2(gy, gx)` folded into `[0, pi)`, plus the circular resultant length of the doubled angles (coherence) and the normalised Shannon entropy of the histogram |

Pixels whose gradient magnitude is below `0.10` on the `[0, 1]` scale carry no
reliable orientation and are excluded from the orientation family. The
surviving counts are normalised to sum to one. An exactly uniform image has
zero gradient everywhere, so all magnitude statistics are zero and, because no
pixel clears the weak-gradient threshold, the orientation histogram, coherence
and entropy are all zero. An empty strong-gradient set is a defined zero, not a
division by zero. Every output is finite and deterministic.

The **no-reference boundary** is strict and structural. The extractor signature
is `extract_local_structure_features(image, spec=None)`: the only required
input is the corrupted image array, and there is no parameter for a clean
reference, target, source identifier, family, severity, font, split or path.
Every run checks the signature and scans the feature module's code for
forbidden identifiers; tests repeat both checks independently. The clean
reference is used only by the separate evaluation pipeline that computes the
SSIM label.

Three candidates are predeclared. The frozen M5C champion is loaded without
refitting; the local-only and global-plus-local candidates are refitted on the
same 432 expanded training rows with the same model family, hyperparameters,
seed and objective (random forest, 300 trees, unlimited depth, leaf size 1,
seed 798117391). No hyperparameter is tuned, no post-hoc calibration is
applied and no feature is removed in response to a diagnostic. The M6A/M6B
retention-aware selection rule is used unchanged, with the same coefficients
and tolerances, and the frozen champion is the fallback. The 0.5 px rung is
diagnostic only.

**Validation results by slice (selection evidence).**

| candidate | features | geometric RMSE | clean RMSE | original RMSE | sev 2 RMSE | sev 3 RMSE | sev 4 RMSE | score | eligible | selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `global_baseline` | 8 | 0.04925 | 0.00223 | 0.02007 | 0.05254 | 0.02494 | 0.06240 | 0.00000 | yes | **yes** |
| `local_only` | 14 | 0.05957 | 0.00310 | 0.02850 | 0.05015 | 0.03145 | 0.08451 | 0.05754 | no | no |
| `global_plus_local` | 22 | 0.05319 | 0.00621 | 0.01945 | 0.04998 | 0.02681 | 0.07260 | 0.02396 | no | no |

Both local candidates are ineligible. `global_plus_local` breaches the 50%
clean-reference tolerance (0.00621 versus 0.00223). `local_only` breaches the
10% original-task tolerance (0.02850 versus 0.02007), the 25% severity-3
tolerance (0.03145 versus 0.02494) and the 25% severity-4 tolerance (0.08451
versus 0.06240). The per-severity and retention constraints are doing the work
the aggregate score would not: the aggregate geometric validation RMSE of
`global_plus_local` is only 0.00394 worse than the champion, but the clean
slice is nearly three times worse.

**Test results by slice (after selection was frozen).**

| candidate | geometric RMSE | geometric R2 | clean RMSE | original RMSE | M3 overall RMSE | M3 overall R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `global_baseline` | 0.05901 | 0.18553 | 0.00228 | 0.02333 | 0.02201 | 0.98716 |
| `local_only` | 0.06612 | -0.02254 | 0.02021 | 0.03600 | 0.03461 | 0.96824 |
| `global_plus_local` | 0.06338 | 0.06022 | 0.00931 | 0.03112 | 0.02950 | 0.97692 |

The champion's reference metrics reproduce the pinned M5C values exactly:
geometric MAE/RMSE/R2 0.04928/0.05901/0.18553, clean RMSE 0.00228, original
RMSE 0.02333, M3 test RMSE 0.02201/R2 0.98716. No local candidate improves the
aggregate geometric slice. `global_plus_local` is 0.00438 worse in RMSE and
`local_only` is 0.00711 worse.

The geometric slice is not uniform across the primary ladder. On test, both
local candidates improve severity 2 (1 px) and regress severities 3 and 4:

| candidate | sev 2 RMSE | sev 3 RMSE | sev 4 RMSE |
| --- | ---: | ---: | ---: |
| `global_baseline` | 0.06187 | 0.02362 | 0.07784 |
| `local_only` | 0.04277 (-0.01910) | 0.04027 (+0.01665) | 0.09830 (+0.02045) |
| `global_plus_local` | 0.05355 (-0.00832) | 0.03384 (+0.01022) | 0.08967 (+0.01182) |

The paired severity-2 intervals include zero for both candidates, so the 1 px
improvement is not clear. The severity-3 and severity-4 regressions exclude
zero for both candidates.

**Paired source-grouped differences (candidate minus champion, RMSE).**

| slice | `local_only` | `global_plus_local` |
| --- | --- | --- |
| geometric primary | +0.00711 [0.00165, 0.01204] | +0.00438 [-0.00015, 0.00891] |
| clean reference | +0.01794 [0.00376, 0.02911] | +0.00704 [0.00036, 0.01192] |
| original blur/noise | +0.01267 [0.00741, 0.01831] | +0.00779 [0.00299, 0.01248] |
| M3 test overall | +0.01260 [0.00713, 0.01805] | +0.00749 [0.00253, 0.01171] |

There are 12 held-out test source groups, so these intervals are preliminary
and descriptive; no significance claim is made. The pattern is consistent: the
local family clearly costs retention, and the combined candidate's aggregate
geometric interval includes zero. It does not demonstrate a geometric gain.

**Training-only diagnostics.** On 432 training rows, none of the 14 local
features is constant, duplicated or non-finite. The mean absolute pairwise
correlation among the local features is 0.3925 and the maximum is 0.9926: the
four gradient-magnitude statistics are largely redundant with one another. In
the combined candidate the maximum absolute correlation between a local and a
global feature is 0.8743, so the local family partly re-expresses information
the global features already carry. Random-forest importance (descriptive, not
causal) gives the combined candidate a global share of 0.7717 and a local share
of 0.2283; `histogram_entropy` alone accounts for 0.6303 and `grad_mag_std` for
0.1521. In the local-only candidate, `grad_mag_median` (0.6421) and
`grad_mag_std` (0.2508) dominate. The training-to-validation geometric RMSE gap
is 0.02645 for the champion, 0.03093 for the combined candidate and 0.03799 for
the local-only candidate, so the larger feature sets overfit more.

**Observations.** The local family is not useless: it improves the 1 px rung on
test for both candidates, and `global_plus_local` is marginally better than the
champion on the validation original-task slice (0.01945 versus 0.02007).
Neither effect survives the predeclared rule. The 1 px gain is within the
paired interval, and the original-task gain reverses on test. The dominant
pattern is that the local features trade retention for a small and inconsistent
geometric effect. The feature-importance and correlation diagnostics explain
why: the orientation histogram's useful content overlaps the existing
`histogram_entropy`, and the magnitude statistics overlap `laplacian_variance`
and `edge_density`.

**Conclusion.** The hypothesis is not supported. Local structural features
computed from the corrupted image alone do not improve geometric-distortion
SSIM prediction beyond the eight global features while preserving
clean-reference and original-task performance. Both local candidates are
ineligible on validation, and the frozen M5C champion is retained. This is a
negative result, reported as such; the selection rule was not weakened to force
a winner. Conventional feature-based ML remains the benchmark champion, and
the evidence does not justify a CNN.

**Limitations.**

- One synthetic renderer, one 192x64 canvas, three fonts and 12 held-out test
  source groups. All grouped intervals are preliminary.
- Only two declared feature families and 14 features were tested. The result
  does not rule out every possible local representation.
- The gradient-magnitude statistics are strongly redundant with each other and
  with the existing global edge/variance features, which limits the
  information the family can add.
- The orientation histogram is a global summary of local orientations; it
  cannot localise a warp or compare a region against its own reference.
- SSIM is a full-reference structural proxy, not human readability or CAPTCHA
  solvability. This is not a security evaluation.
- No CNN, dashboard, deployment, external CAPTCHA dataset or additional
  corruption family is part of this milestone.

**Next step.** The most defensible follow-up is a representation explicitly
targeted at local *displacement* rather than local gradient strength or
orientation, such as a multi-scale structure-tensor anisotropy or local
autocorrelation family, evaluated under the same no-reference boundary and the
same predeclared retention-aware rule. If that also fails, the accumulated
evidence points to a learned representation as the next distinct milestone.
Starting a CNN is not justified by this result alone.

**Artifacts.** Git-ignored: `reports/local_structure_study_v0_1_0/` (tables,
diagnostics, report, metadata) and
`models/local_structure_study_v0_1_0/` (refitted bundles and cards).

---

### EXP-0011 — Spatial resolution study: the information is present, but the model cannot use it

- **Date:** 2026-10-21
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Does increasing the spatial resolution of the
  existing occupancy and edge-density feature families improve
  geometric-distortion SSIM prediction while preserving clean-reference and
  original-task performance? M6A added a coarse 4x2 grid of spatial features
  and found they did not help, then measured that the grid resolved blur about
  25 times more strongly than a 1-3 px warp. M6B varies only the grid
  resolution to test whether that mismatch was the blocker. This is a controlled
  resolution experiment, not a search for the lowest test score. This is a
  synthetic image-similarity benchmark, not a human-readability,
  CAPTCHA-solvability or security study.
- **Code revision:** base `e7314ec` plus the M6B implementation, committed as
  `feat: evaluate spatial feature resolutions`. The reported run was repeated
  after the commit so generated provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; scikit-image 0.26.0; joblib 1.6.0; macOS 27.0.1 (arm64). No GPU was
  used.
- **Configuration:** `configs/spatial_resolution_study_v0_1_0.toml`; config
  sha256
  `9376ecef3e5d66dc7b8a7415f615de28330a02471c0476d4cf231d8ebe01a8a0`;
  protocol hash
  `fafd818694cafff51ae5129453592a1793d754bc312248a7f843cb9b5013b152`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates.
- **Upstream frozen inputs:** M5C config sha256
  `48f3f148121537ea53a685bda961f01e758e5ac196b89b9e0181587d545086eb`, M5C
  protocol hash
  `048b6d990749ca27358ce4c0d0ccb9a91cc0400e1da89ea48bea6661465e3e1a`, M5C
  champion artifact sha256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`; M6A
  config sha256
  `e72a9e34926573b959959b8e62956cb6070a5c4564c05357d1e1d6b65266e909`, M6A
  protocol hash
  `d744248e0ebde54f83a005e05dd1a37bfdaa297a34e8d9f1299514e613e89bc9`,
  spatial feature set `spatial-v1`. The M3 dataset, split and model, the M4A
  pilot, the M4B benchmark, the M5A pilot, the M5B benchmark, the M5C study, the
  M5D study and the M6A study are unchanged and re-verified from disk.
- **Split:** the frozen M3 source-grouped assignment, reused verbatim through
  the M5C verified path (assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`):
  36 train / 12 validation / 12 test sources. Expanded training 432 rows /
  36 sources; expanded validation 144 rows / 12 sources; primary geometric test
  36 rows / 12 sources; diagnostic 0.5 px test 12 rows / 12 sources; M3 test
  108 rows / 12 sources.
- **Runtime:** 18.12 s in process on the recorded machine (not a benchmark).

**Method.** The two M6A feature families are reused with no change to their
mathematical definitions: regional ink occupancy uses the per-image Otsu
threshold, and regional edge density uses the repository's Canny convention
(sigma=1.0, low_threshold=0.1, high_threshold=0.2 on the image scaled to
[0, 1]). Both maps are computed once per image. The grayscale representation, the
canonical 192x64 canvas and the row-major cell ordering are unchanged. Cells
are bounded by floor(i*height/r) .. floor((i+1)*height/r), which tiles the
canonical canvas evenly at every declared resolution and gives any remainder to
the later cells; an empty cell yields zero for every feature and an exactly
uniform image has no distinguishable ink class and no edges, so all its
features are zero.

Four resolutions were predeclared, and each is evaluated in two
representations. The model family, hyperparameters, seed, training rows and
objective are identical for every candidate, so only the input representation
and its resolution differ.

| grid | cells | cell size (px) | spatial features | combined features |
| --- | ---: | ---: | ---: | ---: |
| 2x2 | 4 | 32x96 | 8 | 16 |
| 4x2 | 8 | 16x96 | 16 | 24 |
| 4x4 | 16 | 16x48 | 32 | 40 |
| 8x8 | 64 | 8x24 | 128 | 136 |

Selection uses the M6A rule unchanged, with the same coefficients
(`w_geometric=1.0`, `w_clean=2.0`, `w_original=2.0`, `w_severity=1.0`) and the
same tolerances (clean 0.50, original 0.10, per-severity 0.25), so the two
studies are directly comparable. The 4x2 grid is a reference condition, not a
reason to favour or reject that resolution in advance. The frozen M5C
champion is the fallback.

**Cross-milestone reproducibility.** The 4x2 candidates reproduce the M6A
4x2 candidates *bit-for-bit*: both test slice metrics agree to 0.0e+00 on
every slice. This is a strong check that M6B is a strict extension of M6A
rather than a re-parameterisation of it.

**Validation results by slice (selection evidence).**

| candidate | features | geometric RMSE | clean RMSE | original RMSE | score | eligible | selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `global_baseline` | 8 | 0.04925 | 0.00223 | 0.02007 | 0.00000 | yes | **yes** |
| `combined_2x2` | 16 | 0.04892 | 0.00228 | 0.02281 | 0.00666 | no | no |
| `combined_4x2` | 24 | 0.04869 | 0.00282 | 0.02312 | 0.01030 | no | no |
| `combined_4x4` | 40 | 0.05037 | 0.00311 | 0.02398 | 0.01792 | no | no |
| `combined_8x8` | 136 | 0.05202 | 0.00337 | 0.02428 | 0.02146 | no | no |
| `spatial_only_2x2` | 8 | 0.08563 | 0.14601 | 0.07415 | 0.55004 | no | no |
| `spatial_only_4x2` | 16 | 0.10069 | 0.15846 | 0.08907 | 0.65204 | no | no |
| `spatial_only_4x4` | 32 | 0.09041 | 0.15448 | 0.07846 | 0.58967 | no | no |
| `spatial_only_8x8` | 128 | 0.14773 | 0.21312 | 0.08791 | 0.94528 | no | no |

Every spatial candidate is ineligible. The combined candidates all breach the
10% original-task tolerance, and the spatial-only candidates breach it plus the
clean-slice and per-severity tolerances. Notably the *coarsest* grid (2x2) has
the best aggregate geometric RMSE among the combined candidates and the finest
(8x8) the worst, so resolution is not monotone in the useful direction.

**Final test results by slice (frozen evaluation, 12 held-out sources).**

| candidate | geometric MAE / RMSE / R² | clean RMSE | original RMSE | M3 test MAE / RMSE / R² |
| --- | --- | ---: | ---: | --- |
| `global_baseline` | 0.0493 / 0.0590 / 0.1855 | 0.0023 | 0.0233 | 0.0156 / 0.0220 / 0.9872 |
| `combined_2x2` | 0.0498 / 0.0592 / 0.1805 | 0.0026 | 0.0241 | 0.0167 / 0.0227 / 0.9863 |
| `combined_4x2` | 0.0485 / 0.0584 / 0.2015 | 0.0025 | 0.0250 | 0.0171 / 0.0236 / 0.9852 |
| `combined_4x4` | 0.0511 / 0.0615 / 0.1164 | 0.0033 | 0.0269 | 0.0184 / 0.0254 / 0.9829 |
| `combined_8x8` | 0.0504 / 0.0599 / 0.1602 | 0.0049 | 0.0264 | 0.0185 / 0.0249 / 0.9835 |
| `spatial_only_2x2` | 0.0885 / 0.1072 / -1.6861 | 0.1886 | 0.0805 | 0.0696 / 0.0986 / 0.7423 |
| `spatial_only_4x2` | 0.0762 / 0.0930 / -1.0233 | 0.1689 | 0.0795 | 0.0679 / 0.0937 / 0.7671 |
| `spatial_only_4x4` | 0.0758 / 0.0939 / -1.0610 | 0.1685 | 0.0801 | 0.0694 / 0.0941 / 0.7652 |
| `spatial_only_8x8` | 0.0870 / 0.1075 / -1.7047 | 0.1895 | 0.0850 | 0.0716 / 0.1020 / 0.7239 |

**Paired source-grouped differences against the frozen champion (RMSE).** Whole
source groups are resampled; differences are on identical rows. There are 12
held-out test source groups, so these intervals are preliminary and descriptive,
and no significance claim is made from interval overlap or non-overlap.

| candidate | geometric primary | clean reference | original blur/noise | M3 test overall |
| --- | --- | --- | --- | --- |
| `combined_2x2` | +0.0002 [-0.0034, +0.0030] | +0.0003 [+0.0001, +0.0007] | +0.0007 [-0.0003, +0.0020] | +0.0007 [-0.0002, +0.0020] |
| `combined_4x2` | -0.0006 [-0.0042, +0.0027] | +0.0003 [-0.0001, +0.0012] | +0.0017 [+0.0007, +0.0030] | +0.0016 [+0.0007, +0.0029] |
| `combined_4x4` | +0.0025 [-0.0020, +0.0067] | +0.0010 [+0.0002, +0.0021] | +0.0036 [+0.0021, +0.0049] | +0.0034 [+0.0020, +0.0046] |
| `combined_8x8` | +0.0009 [-0.0032, +0.0043] | +0.0026 [+0.0016, +0.0037] | +0.0031 [+0.0016, +0.0048] | +0.0029 [+0.0015, +0.0047] |
| `spatial_only_2x2` | +0.0482 [+0.0216, +0.0726] | +0.1863 [+0.1411, +0.2265] | +0.0572 [+0.0473, +0.0675] | +0.0766 [+0.0661, +0.0871] |
| `spatial_only_4x2` | +0.0340 [+0.0131, +0.0547] | +0.1666 [+0.1377, +0.1924] | +0.0562 [+0.0465, +0.0674] | +0.0717 [+0.0653, +0.0788] |
| `spatial_only_4x4` | +0.0349 [+0.0129, +0.0581] | +0.1662 [+0.1399, +0.1955] | +0.0568 [+0.0518, +0.0625] | +0.0721 [+0.0668, +0.0775] |
| `spatial_only_8x8` | +0.0485 [+0.0164, +0.0785] | +0.1873 [+0.1512, +0.2237] | +0.0617 [+0.0448, +0.0775] | +0.0800 [+0.0608, +0.0987] |

Every geometric interval includes zero, so no resolution produces a
demonstrable geometric gain. The retention regressions are systematic: for the
combined candidates the clean-slice and original-task intervals exclude zero
from 4x2 upwards and grow with the feature count.

**Training-only diagnostics.** Computed on the training rows; reported, not
acted upon. No feature was removed, no grid changed and no model modified in
response to them.

| candidate | features | constant | duplicated | mean \|r\| | max \|r\| | train RMSE | validation RMSE | overfitting gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `global_baseline` | 8 | 0 | 0 | n/a | n/a | 0.01333 | 0.02958 | 0.01626 |
| `combined_2x2` | 16 | 0 | 0 | 0.585 | 0.962 | 0.01336 | 0.03075 | 0.01739 |
| `combined_4x2` | 24 | 0 | 0 | 0.430 | 0.958 | 0.01338 | 0.03082 | 0.01744 |
| `combined_4x4` | 40 | 0 | 0 | 0.387 | 0.932 | 0.01373 | 0.03191 | 0.01818 |
| `combined_8x8` | 136 | 0 | 0 | 0.259 | 0.953 | 0.01405 | 0.03272 | 0.01867 |
| `spatial_only_2x2` | 8 | 0 | 0 | 0.585 | 0.962 | 0.03367 | 0.08530 | 0.05162 |
| `spatial_only_4x2` | 16 | 0 | 0 | 0.430 | 0.958 | 0.03492 | 0.09958 | 0.06466 |
| `spatial_only_4x4` | 32 | 0 | 0 | 0.387 | 0.932 | 0.03611 | 0.09020 | 0.05409 |
| `spatial_only_8x8` | 128 | 0 | 0 | 0.259 | 0.953 | 0.03432 | 0.11997 | 0.08565 |

No feature is constant or duplicated at any resolution, and redundancy
*decreases* with resolution (mean |r| 0.585 at 2x2 to 0.259 at 8x8), so
redundancy is not the mechanism. The overfitting gap does grow with the
feature count (0.01739 to 0.01867 for the combined candidates, and 0.05162 to
0.08565 for the spatial-only ones), and the spatial-only candidates are the
most affected.

**Grid-resolution diagnostic (training rows only, descriptive).** Per-cell ink
occupancy of a distorted image against the same source's clean reference. This
measures what each grid can resolve, independently of any model.

| grid | family | pairs | mean abs occupancy change | correlation with SSIM drop |
| --- | --- | ---: | ---: | ---: |
| 2x2 | elastic | 108 | 0.00424 | +0.453 |
| 2x2 | gaussian_blur | 144 | 0.13561 | +0.893 |
| 2x2 | gaussian_noise | 144 | 0.00143 | +0.558 |
| 4x2 | elastic | 108 | 0.00536 | +0.573 |
| 4x2 | gaussian_blur | 144 | 0.13565 | +0.893 |
| 4x2 | gaussian_noise | 144 | 0.00165 | +0.611 |
| 4x4 | elastic | 108 | 0.00703 | +0.705 |
| 4x4 | gaussian_blur | 144 | 0.13570 | +0.893 |
| 4x4 | gaussian_noise | 144 | 0.00197 | +0.685 |
| 8x8 | elastic | 108 | 0.01068 | +0.828 |
| 8x8 | gaussian_blur | 144 | 0.13600 | +0.894 |
| 8x8 | gaussian_noise | 144 | 0.00266 | +0.789 |

**Observations.**

1. *M6A's diagnosis was correct and the obvious follow-up failed.* M6A found
   that a 4x2 grid resolves blur about 25 times more strongly than a 1-3 px
   warp, and M6B confirms that the mismatch is indeed resolvable by raising the
   resolution: the elastic occupancy signal strengthens monotonically, and its
   correlation with the SSIM drop rises from +0.453 at 2x2 to +0.828 at 8x8,
   closing most of the gap to blur's +0.894.
2. *The information is present but the model cannot use it.* Despite the
   diagnostic improvement, no resolution produces a demonstrable geometric gain
   (every paired geometric interval includes zero), and the aggregate
   geometric validation RMSE is best at the *coarsest* grid. Meanwhile the
   retention slices degrade monotonically with the feature count, because the
   forest considers every feature at each split (`max_features` is 1.0) and the
   extra spatial features are informative about the geometric family but largely
   uninformative about blur, noise and clean references. The extra features
   therefore dilute the search for the splits that matter on the other
   families.
3. *Redundancy is not the mechanism.* Mean absolute pairwise correlation
   *falls* with resolution, and no feature is constant or duplicated at any
   grid, so the failure is not that the finer features repeat each other.
4. *Resolution is not a free lever.* The validation rule rejects every spatial
   candidate, and the test set agrees: the clean-slice and original-task
   regressions are systematic and their intervals exclude zero from 4x2 upwards.

**Conclusion.** The retention trade-off was **not** improved by increasing the
spatial resolution, and the frozen M5C champion is retained. The negative
result is sharper than M6A's because it isolates the mechanism: the finer grid
*does* resolve the geometric warp, so the blocker is not the resolution of the
partition but the **relevance** of the resulting features to the other
corruption families. Conventional feature-based ML remains the benchmark
champion. The evidence does not justify a CNN, and a new corruption family does
not by itself justify one.

**Limitations.** One synthetic renderer, one 192x64 canvas, three fonts, one
smooth elastic-warp family and 60 sources. Only 12 held-out test source groups,
so all grouped intervals are preliminary. The 0.5 px rung is
interpolation-dominated and diagnostic only. A grid partition can show that a
distortion is spatially uneven, but it cannot localise a warp to a glyph, and
the per-cell occupancy it exposes is a summary statistic rather than a
displacement field. Only four resolutions and two representations were tested,
all predeclared; the resolution set was not searched. The 4x2 reference
condition is the M6A configuration, so a resolution below 2x2 is untested.
Feature importance is descriptive, not causal. SSIM is a full-reference
similarity proxy, not human readability or CAPTCHA solvability.

**Next step.** Stop M6B here. Do not start CNN training, another corruption
family, a UI, deployment or presentation work. If this line continues, the
defensible next experiment follows directly from the mechanism: since the
spatial signal is present but not *relevant* to the other families, the
question is whether a feature that is specifically sensitive to local
displacement — rather than to local ink quantity — can be defined so that it
adds information about the geometric family without diluting the splits that
support blur, noise and clean references. That should be predeclared and
evaluated on the same frozen split. Do not retune on the M5A/M3 test rows.

**Artifacts.** Git-ignored:
`models/spatial_resolution_study_v0_1_0/` (one bundle per refitted candidate,
each with a JSON card) and
`reports/spatial_resolution_study_v0_1_0/` (`report.md`, validation slice
metrics, selection table, test slice metrics, predictions, uncertainty, paired
uncertainty, isolation checks, the spatial feature manifest and
`run_metadata.json`). Full artifact hashes are in `run_metadata.json`;
`combined_2x2.joblib` has sha256
`60c6e5e5d678ad3f38143d5c0dbc19734b0f72f6025c4d7fcc39f45948522af5` and
`combined_4x2.joblib` has sha256
`596e497afcf6f7031ff7e7db16155e88212aae7d6b23fcd7554a01aaec0d5b33`.
Regenerate with `uv run --extra image --extra ml captcha-resolution-study run
--config configs/spatial_resolution_study_v0_1_0.toml`; verify with
`captcha-resolution-study verify --config
configs/spatial_resolution_study_v0_1_0.toml`.

---

### EXP-0010 — Spatial-aware feature study: a coarse grid cannot resolve a 1-3 px warp, so the champion is retained

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Does adding spatially localized image features
  improve geometric-distortion SSIM prediction compared with the existing eight
  global image features, while preserving clean-reference and original-task
  performance? M5D concluded that the feature *representation*, not the
  training objective, was the likely binding constraint; M6A tests that
  directly. This is a controlled feature-ablation experiment, not a new model
  family. This is a synthetic image-similarity benchmark, not a
  human-readability, CAPTCHA-solvability or security study.
- **Code revision:** base `4f00965` plus the M6A implementation, committed as
  `feat: evaluate spatial-aware image features`. The reported run was repeated
  after the commit so generated provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; scikit-image 0.26.0; joblib 1.6.0; macOS 27.0.1 (arm64). No GPU was
  used.
- **Configuration:** `configs/spatial_feature_study_v0_1_0.toml`; config sha256
  `0b54c42eec9b9a09bedd776c5345c7b02f96736308bd83e7d2706ff708f4dc1c`;
  protocol hash
  `d744248e0ebde54f83a005e05dd1a37bfdaa297a34e8d9f1299514e613e89bc9`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates.
- **Upstream frozen inputs:** M5C config sha256
  `48f3f148121537ea53a685bda961f01e758e5ac196b89b9e0181587d545086eb`, M5C
  protocol hash
  `048b6d990749ca27358ce4c0d0ccb9a91cc0400e1da89ea48bea6661465e3e1a`, M5C
  champion artifact sha256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`. The M3
  dataset, split and model, the M4A pilot, the M4B benchmark, the M5A pilot,
  the M5B benchmark, the M5C study and the M5D study are unchanged and
  re-verified from disk.
- **Split:** the frozen M3 source-grouped assignment, reused verbatim through
  the M5C verified path (assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`):
  36 train / 12 validation / 12 test sources. Expanded training 432 rows /
  36 sources; expanded validation 144 rows / 12 sources; primary geometric test
  36 rows / 12 sources; diagnostic 0.5 px test 12 rows / 12 sources; M3 test
  108 rows / 12 sources.
- **Runtime:** 7.04 s in process on the recorded machine (not a benchmark).

**Method.** The eight frozen global features and their extraction code are
unchanged. The spatial features live in a separate, versioned module
(`spatial_features.py`, feature-set version `spatial-v1`) with their own names,
so prior model inputs and artefacts cannot be silently altered. The canonical
192x64 image is divided into a fixed 4x2 grid of eight 16x96-pixel cells,
visited in row-major order, and two families are implemented:

- `ink_occupancy` - the fraction of pixels at or below the Otsu threshold of
  the *whole* image, counted within the cell. The threshold is computed once
  per image, so the cell values share one ink definition and are directly
  comparable with the global `ink_fraction` feature.
- `edge_density` - the fraction of Canny-marked pixels within the cell, using
  the repository's existing edge convention (sigma=1.0, low_threshold=0.1,
  high_threshold=0.2 on the image scaled to [0, 1]).

That is 16 spatial features. Regional intensity statistics were considered and
rejected: on training rows the per-cell mean intensity is largely redundant
with the per-cell ink occupancy (maximum |correlation| 0.873 across cell pairs),
so adding it would inflate the feature count without adding information. A 4x2
grid was chosen over 2x4 because it has lower internal redundancy among the
spatial features (mean pairwise |r| 0.38 against 0.52) at a comparable
correlation with the target. Both decisions were made from training rows only.

Three ablations were predeclared. The model family, the hyperparameters, the
seed and the training partition are held fixed across all of them, so the only
thing that varies is the input representation:

| ablation | feature set | features | fitted |
| --- | --- | ---: | --- |
| `global_baseline` | the eight frozen global features | 8 | no — the frozen M5C champion, loaded without refitting |
| `spatial_only` | the spatial family alone | 16 | yes |
| `combined` | global plus spatial | 24 | yes |

Selection uses the M5D retention-aware rule with identical coefficients and
tolerances, so the two studies are directly comparable: the baseline is the
reference, a candidate qualifies only if it regresses neither the clean slice,
nor the original-task slice, nor any single primary geometric severity beyond
the declared tolerances, and among the qualifying candidates the score rewards
aggregate geometric RMSE and charges clean, original-task and per-severity
regressions. The frozen M5C champion is the fallback. No coefficient or
tolerance was chosen by inspecting a test result.

**Validation results by slice (selection evidence).**

| ablation | geometric RMSE | clean RMSE | original RMSE | severity 2 | severity 3 | severity 4 | score | eligible | selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `global_baseline` | 0.04925 | 0.00223 | 0.02007 | 0.05254 | 0.02494 | 0.06240 | 0.00000 | yes | yes |
| `combined` | 0.04869 | 0.00282 | 0.02312 | 0.04786 | 0.02163 | 0.06598 | 0.01030 | **no** | no |
| `spatial_only` | 0.10069 | 0.15846 | 0.08907 | 0.13465 | 0.08801 | 0.06737 | 0.65204 | **no** | no |

`combined` is rejected because its original-task RMSE regresses by 15.2%,
beyond the 10% tolerance, even though its aggregate geometric RMSE improves by
0.00056. `spatial_only` is rejected on four counts: clean, original-task,
severity 2 and severity 3.

**Final test results by slice (frozen evaluation, 12 held-out sources).**

| ablation | geometric MAE / RMSE / R² | clean RMSE | original RMSE | M3 test MAE / RMSE / R² |
| --- | --- | ---: | ---: | --- |
| `global_baseline` | 0.0493 / 0.0590 / 0.1855 | 0.0023 | 0.0233 | 0.0156 / 0.0220 / 0.9872 |
| `combined` | 0.0485 / 0.0584 / 0.2015 | 0.0025 | 0.0250 | 0.0171 / 0.0236 / 0.9852 |
| `spatial_only` | 0.0762 / 0.0930 / -1.0233 | 0.1689 | 0.0795 | 0.0679 / 0.0937 / 0.7671 |

**Paired source-grouped differences against the frozen baseline.** Whole source
groups are resampled; differences are on identical rows. There are 12 held-out
test source groups, so these intervals are preliminary and descriptive, and no
significance claim is made from interval overlap or non-overlap.

| ablation | scope | metric | estimate | 95% low | 95% high |
| --- | --- | --- | ---: | ---: | ---: |
| `combined` | geometric primary | RMSE | -0.0006 | -0.0043 | +0.0029 |
| `combined` | severity 2 | RMSE | -0.0043 | -0.0116 | +0.0040 |
| `combined` | severity 3 | RMSE | -0.0020 | -0.0083 | +0.0052 |
| `combined` | severity 4 | RMSE | +0.0025 | -0.0055 | +0.0104 |
| `combined` | clean reference | RMSE | +0.0003 | -0.0001 | +0.0012 |
| `combined` | original blur/noise | RMSE | +0.0017 | +0.0007 | +0.0029 |
| `combined` | M3 test overall | RMSE | +0.0016 | +0.0006 | +0.0028 |
| `spatial_only` | geometric primary | RMSE | +0.0340 | +0.0136 | +0.0539 |
| `spatial_only` | clean reference | RMSE | +0.1666 | +0.1378 | +0.1942 |
| `spatial_only` | original blur/noise | RMSE | +0.0562 | +0.0463 | +0.0670 |

The `combined` geometric improvement is small and its interval includes zero;
its original-task and M3-retention regressions have intervals that exclude
zero, so the trade-off is a real one rather than noise.

**Grid-resolution diagnostic (training rows only, descriptive).** The headline
metrics cannot say whether the declared grid is even capable of resolving the
distortion, so the study also measures, per source, the per-cell ink-occupancy
change between a clean reference and its distorted image:

| family | pairs | mean abs occupancy change | max abs occupancy change | correlation with SSIM drop |
| --- | ---: | ---: | ---: | ---: |
| `gaussian_blur` | 144 | 0.13565 | 0.33813 | +0.893 |
| `elastic` | 108 | 0.00536 | 0.01066 | +0.573 |
| `gaussian_noise` | 144 | 0.00165 | 0.00456 | +0.611 |

A 1-3 px warp moves ink by a fraction of a 16x96-pixel cell, so the coarse
partition averages most of the effect away: the grid resolves blur about 25
times more strongly than the elastic family. The spatial features therefore
largely re-encode how much ink the image has, which the global features
already state, and the residual spatial signal is too weak to separate the
geometric family from the others. This is the mechanism behind both the
ineligibility of `combined` and the failure of `spatial_only`.

Feature diagnostics on the training rows: no duplicated and no constant
spatial features; mean absolute pairwise correlation among the 16 spatial
features 0.43, maximum 0.96. In the `combined` model the global features
dominate the fitted importance (histogram_entropy 0.63, laplacian_variance
0.11, ink_fraction 0.10), and the highest-ranked spatial feature reaches only
0.0035; in the `spatial_only` model the spatial features are used, but they
cannot stand in for the global summary. Importance is descriptive, not causal.

**Observations.**

1. *A coarse spatial partition cannot resolve this distortion.* The grid
   diagnostic, computed without reference to any model, shows the elastic
   signal is 25x weaker than the blur signal at the same partition. This is the
   concrete explanation for the negative result, and it is a statement about
   the *declared resolution*, not about localization in general.
2. *The combined ablation traded rather than won.* It improved aggregate
   geometric RMSE by 0.00056 on validation while regressing the original task
   by 15.2%, so the retention-aware rule rejected it. The improvement's paired
   interval includes zero; the retention regression's does not.
3. *Discarding the global summary is costly.* The spatial-only ablation is far
   worse on every slice, including the geometric one, because the global
   features carry the information that tracks blur and noise.
4. *M5D's reading is confirmed but narrowed.* M5D concluded that the
   representation was the likely constraint. M6A confirms that a *coarse*
   spatial summary does not relieve it, and localises the reason to the
   resolution mismatch between a 1-3 px warp and a 16x96-pixel cell.

**Conclusion.** Spatial features did not improve geometric-distortion
prediction under the predeclared retention-aware rule, and the frozen M5C
champion is retained. The negative result is informative rather than
discouraging: it rules out coarse regional summaries as the missing
information, and it identifies the resolution at which a spatial feature
would have to operate to see a 1-3 px warp at all. Conventional feature-based
ML remains the benchmark champion. The evidence does not justify a CNN, and a
new corruption family does not by itself justify one.

**Limitations.** One synthetic renderer, one 192x64 canvas, three fonts, one
smooth elastic-warp family and 60 sources. Only 12 held-out test source
groups, so all grouped intervals are preliminary. The 0.5 px rung is
interpolation-dominated and diagnostic only. A 4x2 grid is a coarse partition:
it can show that a distortion is spatially uneven, but it cannot localise a
warp to a glyph. The spatial features are additive to the global ones, not a
replacement, so a gain in the combined model cannot be attributed to the
spatial family alone. Only one grid resolution and one alternative family were
tested, both predeclared; a different resolution would change what the spatial
family can see, and was deliberately not searched within this milestone.
Feature importance is descriptive, not causal. SSIM is a full-reference
similarity proxy, not human readability or CAPTCHA solvability.

**Next step.** Stop M6A here. Do not start CNN training, another corruption
family, a UI, deployment or presentation work. If this line continues, the
defensible next experiment is a predeclared *resolution* study: hold the
feature families and the model fixed and vary only the grid resolution, so
that the point at which a spatial feature begins to see a 1-3 px warp is
measured on the same frozen split rather than assumed. Do not retune on the
M5A/M3 test rows.

**Artifacts.** Git-ignored:
`models/spatial_feature_study_v0_1_0/` (`spatial_only.joblib` and
`combined.joblib`, each with a JSON card) and
`reports/spatial_feature_study_v0_1_0/` (`report.md`, validation slice metrics,
selection table, test slice metrics, predictions, uncertainty, paired
uncertainty, isolation checks, the spatial feature manifest and
`run_metadata.json`). Spatial-only artifact sha256
`26c8074a5d376e17...`; combined artifact sha256 `596e497afcf6f703...` (full
values in `run_metadata.json`). Regenerate with `uv run --extra image --extra ml
captcha-spatial-study run --config
configs/spatial_feature_study_v0_1_0.toml`; verify with
`captcha-spatial-study verify --config
configs/spatial_feature_study_v0_1_0.toml`.

---

### EXP-0009 — Retention-aware training objectives: weighting moves error between slices rather than removing it

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Can a retention-aware *training objective* improve
  geometric-distortion SSIM prediction relative to the M5C champion, without
  materially degrading clean-reference and original-task performance? M5C found
  that a retention-aware *selection* rule could pick a model that improves the
  geometric slice while keeping the original task, but that model was a
  different family from the one M5B selected, and the post-hoc affine
  calibration M5C tested did not help. M5D therefore holds the model family,
  the hyperparameters, the training rows and the seed fixed and varies only the
  per-example sample weight, so that any difference is attributable to the
  training objective. This is a synthetic image-similarity benchmark, not a
  human-readability, CAPTCHA-solvability or security study.
- **Code revision:** base `c2f0b37` plus the M5D implementation, committed as
  `feat: evaluate retention-aware training objectives`. The reported run was
  repeated after the commit so generated provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; scikit-image 0.26.0; joblib 1.6.0; macOS 27.0.1 (arm64). No GPU was
  used.
- **Configuration:** `configs/retention_aware_training_v0_1_0.toml`; config
  sha256
  `8842099b00844a71222099f1b88b12735e32c2fd81d27da3dc899ccec60bdea8`;
  protocol hash
  `a163650392ced7b5ec90f5896a74ff345e2cbed36fba28092426445501802383`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates.
- **Upstream frozen inputs:** M5C config sha256
  `48f3f148121537ea53a685bda961f01e758e5ac196b89b9e0181587d545086eb`, M5C
  protocol hash
  `048b6d990749ca27358ce4c0d0ccb9a91cc0400e1da89ea48bea6661465e3e1a`, M5C
  champion artifact sha256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`. The M3
  dataset, split and model, the M4A pilot, the M4B benchmark, the M5A pilot,
  the M5B benchmark and the M5C study are unchanged and re-verified from disk.
- **Split:** the frozen M3 source-grouped assignment, reused verbatim through
  the M5C verified path (assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`):
  36 train / 12 validation / 12 test sources. Expanded training 432 rows /
  36 sources; expanded validation 144 rows / 12 sources; primary geometric test
  36 rows / 12 sources; diagnostic 0.5 px test 12 rows / 12 sources; M3 test
  108 rows / 12 sources.
- **Runtime:** 9.90 s in process on the recorded machine (not a benchmark).

**Method.** Four candidate policies were predeclared. All share the champion's
model family (`random_forest`, n_estimators=300, max_depth unlimited,
min_samples_leaf=1, seed 798117391) and the same 432 training rows; only the
per-example sample weight differs.

| candidate | scheme | fit |
| --- | --- | --- |
| `frozen_m5c_champion` | none | loaded from the frozen M5C bundle, never refitted |
| `retrained_control` | none | refitted with the standard unweighted objective |
| `severity_aware` | severity_aware | refitted with the declared sample weights |
| `retention_aware` | retention_aware | refitted with the declared sample weights |

Every training example is assigned to exactly one group using metadata already
present in the training frame: `clean` (36 M3 clean references), `original` (288
M3 blur/noise rows) and `geometric` (108 M5A elastic rows at the primary rungs).
The weight of an example is

```text
w(x) = group_weight[group(x)] * severity_factor[group(x), severity(x)]
```

where `severity_factor` is 1 for the clean and original groups and
`geometric_severity_ratio ** (severity - min_primary_severity)` for the
geometric group. The weights are rescaled so their mean over the training frame
is exactly 1, every weight is finite and strictly positive, and the weights are
passed only to `fit(sample_weight=...)` — they never enter the predictor's
feature matrix. The assignment is checked to be unchanged when the target column
is removed, which proves it does not depend on labels.

| scheme | clean | original | geometric base | geometric severity ratio |
| --- | ---: | ---: | ---: | ---: |
| `severity_aware` | 1.0 | 1.0 | 1.5 | 1.5 |
| `retention_aware` | 2.0 | 2.0 | 3.0 | 1.0 |

Realised group influence (share of total training weight): under
`severity_aware`, clean 6.2%, original 49.6%, geometric 44.2%; under
`retention_aware`, clean 7.4%, original 59.3%, geometric 33.3%. The
retention-aware scheme therefore protects both retention slices relative to the
severity-aware one while still emphasising geometric examples per row.

**Selection rule, declared before test evaluation.** The champion is the
reference. A candidate qualifies only if it regresses neither the clean slice,
nor the original-task slice, nor any single primary geometric severity beyond
the declared tolerances (`tolerance_clean` 0.50, `tolerance_original` 0.10,
`tolerance_severity` 0.25). Among the qualifying candidates the score is

```text
score = w_geometric * (rmse_geo - rmse_geo_ref)
      + w_clean     * max(0, rmse_clean - rmse_clean_ref)
      + w_original  * max(0, rmse_orig  - rmse_orig_ref)
      + w_severity   * sum_s max(0, rmse_s - rmse_s_ref)
```

with `w_geometric=1.0`, `w_clean=2.0`, `w_original=2.0`, `w_severity=1.0`.
Lower is better, the champion scores exactly 0, ties are broken by lower
aggregate geometric RMSE and then by candidate name, and the frozen champion is
the fallback if nothing qualifies. No coefficient or tolerance was chosen by
inspecting a test result.

**Control result.** The unweighted retrained control reproduces the frozen
champion's validation predictions *exactly* and its persisted artifact has the
same SHA-256 as the champion bundle
(`c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`). Every
difference reported below is therefore attributable to the sample weighting and
to nothing else. Note that passing an all-ones `sample_weight` is *not*
equivalent to omitting the argument, because the tree split criterion is then
evaluated on weighted statistics; the control is fitted with the standard
objective.

**Validation results by slice (selection evidence).**

| candidate | geometric RMSE | clean RMSE | original RMSE | severity 2 | severity 3 | severity 4 | score | eligible | selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `frozen_m5c_champion` | 0.04925 | 0.00223 | 0.02007 | 0.05254 | 0.02494 | 0.06240 | 0.00000 | yes | yes |
| `retrained_control` | 0.04925 | 0.00223 | 0.02007 | 0.05254 | 0.02494 | 0.06240 | 0.00000 | yes | no |
| `severity_aware` | 0.05542 | 0.00236 | 0.02173 | 0.07152 | 0.03481 | 0.05374 | 0.03859 | **no** | no |
| `retention_aware` | 0.04991 | 0.00180 | 0.02139 | 0.05361 | 0.02674 | 0.06232 | 0.00618 | yes | no |

`severity_aware` is rejected by the per-severity constraint: severity 2 RMSE
0.07152 against the champion's 0.05254 and severity 3 RMSE 0.03481 against
0.02494, both more than 25% worse. The aggregate geometric RMSE alone would
not have revealed this. `retention_aware` is eligible but scores +0.00618
against the champion's 0: it improves the clean slice by 0.00043 but costs
0.00066 on the aggregate geometric slice and 0.00132 on the original task.

**Final test results by slice (frozen evaluation, 12 held-out sources).**

| candidate | geometric MAE / RMSE / R² | clean RMSE | original RMSE | M3 test MAE / RMSE / R² |
| --- | --- | ---: | ---: | --- |
| `frozen_m5c_champion` | 0.0493 / 0.0590 / 0.1855 | 0.0023 | 0.0233 | 0.0156 / 0.0220 / 0.9872 |
| `retrained_control` | 0.0493 / 0.0590 / 0.1855 | 0.0023 | 0.0233 | 0.0156 / 0.0220 / 0.9872 |
| `severity_aware` | 0.0490 / 0.0595 / 0.1706 | 0.0027 | 0.0253 | 0.0167 / 0.0238 / 0.9849 |
| `retention_aware` | 0.0484 / 0.0580 / 0.2121 | 0.0019 | 0.0241 | 0.0161 / 0.0227 / 0.9863 |

**Paired source-grouped differences against the champion.** Whole source groups
are resampled; differences are on identical rows. There are 12 held-out test
source groups, so these intervals are preliminary and descriptive, and no
significance claim is made from interval overlap or non-overlap.

| candidate | scope | metric | estimate | 95% low | 95% high |
| --- | --- | --- | ---: | ---: | ---: |
| `severity_aware` | geometric primary | RMSE | +0.0005 | -0.0024 | +0.0033 |
| `severity_aware` | severity 2 | RMSE | +0.0099 | +0.0045 | +0.0143 |
| `severity_aware` | severity 3 | RMSE | -0.0007 | -0.0072 | +0.0047 |
| `severity_aware` | severity 4 | RMSE | -0.0074 | -0.0123 | -0.0030 |
| `severity_aware` | clean reference | RMSE | +0.0005 | -0.0000 | +0.0011 |
| `severity_aware` | original blur/noise | RMSE | +0.0019 | +0.0009 | +0.0029 |
| `retention_aware` | geometric primary | RMSE | -0.0010 | -0.0024 | +0.0006 |
| `retention_aware` | clean reference | RMSE | -0.0003 | -0.0007 | +0.0001 |
| `retention_aware` | original blur/noise | RMSE | +0.0008 | +0.0002 | +0.0013 |

The severity-aware scheme is the clearest illustration of why the per-severity
constraint exists: it is better at the severe rung (severity 4 RMSE -0.0074)
and worse at the mild one (severity 2 +0.0099), while the aggregate number is
nearly unchanged (+0.0005). Selecting on the aggregate alone would have
hidden a real regression on the 1.0 px rung.

**Observations.**

1. *Weighting moves error between slices rather than removing it.* The
   severity-aware scheme buys accuracy at the severe geometric rung by paying
   for it at the mild rung and on the original task; the retention-aware scheme
   buys clean-reference accuracy by paying for it on the original task and on
   the aggregate geometric slice. Neither is a net improvement.
2. *The aggregate geometric score alone would have been misleading.* The
   severity-aware candidate's aggregate geometric RMSE differs from the
   champion's by +0.0005 while its severity-2 RMSE is 36% worse. The
   predeclared per-severity constraint is what rejects it.
3. *The control isolates the effect cleanly.* The unweighted retrained control
   is bit-identical to the champion, so the only thing that differed in the
   weighted candidates was the sample weight.
4. *The eight image-only features are a plausible binding constraint.* They are
   global whole-image statistics: overall intensity, contrast, ink coverage,
   sharpness, edge density, entropy and noise. They do not describe *where* on
   the canvas a distortion occurs, and a geometric warp moves ink locally while
   leaving many whole-image statistics nearly unchanged. This is consistent with
   the near-zero between-source R2 that M5B and M5C both observed, and with
   weighting being unable to help: reweighting cannot add spatial information
   the representation never exposed. This milestone does not isolate the cause
   and cannot rule out sample size or the objective itself.

**Conclusion.** The retention trade-off was **not** improved by a
retention-aware training objective. The frozen M5C champion is retained. The
negative result is informative: within a single model family, a single shared
feature representation and a single shared sample of 432 training rows, sample
weighting redistributes error across slices rather than removing it. The
remaining headroom is more likely in the *representation* than in the
objective, which points at a representation that carries spatial information
(where the distortion is) rather than at a further search over objectives or
model families.

**Limitations.** One synthetic renderer, one 192x64 canvas, three fonts, one
smooth elastic-warp family and 60 sources. Only 12 held-out test source
groups, so all grouped intervals are preliminary. The 0.5 px rung is
interpolation-dominated and diagnostic only. Both weighted schemes are single
points in the weight space; they were predeclared, not searched, so a different
setting might behave differently, but no setting was selected using test
results. The feature-representation explanation is a hypothesis consistent with
the evidence, not an isolated cause. SSIM is a full-reference similarity proxy,
not human readability or CAPTCHA solvability.

**Next step.** Stop M5D here. Do not start CNN training, another corruption
family, a UI, deployment or presentation work. If this line continues, the
defensible next question is whether a representation that carries *spatial*
information about the distortion — rather than further objective or family
search — can lift the near-zero between-source R2, and it should be predeclared
and evaluated on the same frozen split. Do not retune on the M5A/M3 test rows.

**Artifacts.** Git-ignored:
`models/retention_aware_training_v0_1_0/` (`retrained_control.joblib`,
`severity_aware.joblib` and `retention_aware.joblib`, each with a JSON card)
and `reports/retention_aware_training_v0_1_0/` (`report.md`, validation slice
metrics, selection table, test slice metrics, predictions, uncertainty, paired
uncertainty, isolation checks and `run_metadata.json`). Retrained control
artifact sha256
`c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`; severity-aware
artifact sha256
`962f9cf051fc389f195a54a4370490450843d77225ce81070603bc7103ae3bc1`; retention-aware
artifact sha256
`a7121236d9f440cccd41a2d9d51011325dbb11e457a3a711c2c50c100cabe1cc`. Regenerate
with `uv run --extra image --extra ml captcha-retention-training run --config
configs/retention_aware_training_v0_1_0.toml`; verify with
`captcha-retention-training verify --config
configs/retention_aware_training_v0_1_0.toml`.

---

### EXP-0008 — Retention-aware model selection and calibration: a slice-aware policy improves geometric prediction and keeps the original task

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Can a conventional model-selection or calibration
  policy improve geometric-distortion prediction while retaining performance on
  clean references and on the original blur/noise task? M5B left a specific
  trade-off open: the adapted histogram gradient boosting model improved the
  aggregate geometric score but lost clean-reference calibration and came close
  to losing the original task. M5C is a controlled selection and calibration
  experiment, not a new model-development campaign. This is a synthetic
  image-similarity benchmark, not a human-readability, CAPTCHA-solvability or
  security study.
- **Code revision:** base `bb3c50a` plus the M5C implementation, committed as
  `feat: evaluate retention-aware model selection`. The reported run was
  repeated after the commit so generated provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; scikit-image 0.26.0; joblib 1.6.0; macOS 27.0.1 (arm64). No GPU was
  used.
- **Configuration:** `configs/retention_aware_selection_v0_1_0.toml`; config
  sha256
  `48f3f148121537ea53a685bda961f01e758e5ac196b89b9e0181587d545086eb`;
  protocol hash
  `048b6d990749ca27358ce4c0d0ccb9a91cc0400e1da89ea48bea6661465e3e1a`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates.
- **Upstream frozen inputs:** M5B config sha256
  `fe5c66614cee83c72a80c20b807202deadd8a604e045f7432795a8abe706ddb1`, M5B
  protocol hash
  `77e41dd5de84e0bf8751c38a49b510735095f86edc8b00f6c19a6d8fde0d13ce`, M5B
  expanded model artifact sha256
  `360b7d18ca5e196b5f3fcd0cbe35c3c1f1f8b1ca0cf69672eb55a61be6479f9d`. The M3
  dataset, split, model, the M4A pilot, the M4B benchmark and the M5A pilot are
  unchanged and re-verified from disk.
- **Split:** the frozen M3 source-grouped assignment, reused verbatim via the
  M5B verified path (assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`):
  36 train / 12 validation / 12 test sources. Expanded training 432 rows /
  36 sources; expanded validation 144 rows / 12 sources; primary geometric test
  36 rows / 12 sources; diagnostic 0.5 px test 12 rows / 12 sources; M3 test
  108 rows / 12 sources.
- **Runtime:** 6.75 s in process on the recorded machine (not a benchmark).

**Method.** Six candidate policies were predeclared: the frozen M3 model, the
M5B expanded HGB and one expanded conventional alternative, each used raw and
with a single affine calibration (`target = intercept + slope * prediction`).
The alternative is `random_forest_depth_unlimited_leaf_1`, justified from the
existing M5B validation table (validation RMSE 0.0295848, third overall and
1.5% behind the selected HGB at 0.0291514); M5C re-derives that table at run
time and refuses to proceed if the alternative is no longer the best non-HGB
candidate. No M5C test result was used to justify any candidate.

The calibration is univariate, uses no metadata, and is fitted only on the
expanded validation frame, whose predictions from every base model are
out-of-sample because the base models are trained only on training sources.
Validation scoring of a calibrated candidate uses leave-one-fold-out
(4 source-grouped folds) calibration predictions, so a candidate is never
scored on rows that contributed to its own calibration. The frozen policy uses
the calibration fitted on the whole validation frame. `clip_output` is false,
so the map is a pure affine transform: clipping to [0, 1] would silently absorb
the clean-reference residual and hide the offset under study.

Selection used a predeclared retention-aware rule computed on validation data
only. All three slice errors are RMSE in SSIM units, so they are directly
comparable. The score is the change relative to the frozen M3 reference, plus
an explicit extra penalty charged on retention regressions:

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
to be preferred. No weight or threshold was chosen after seeing a test result.
The 0.5 px rung never entered fitting, selection or the primary score.

**Validation results by slice (selection evidence).**

| candidate | geometric RMSE | clean RMSE | original RMSE | score | selected |
| --- | ---: | ---: | ---: | ---: | --- |
| `expanded_random_forest` | 0.0492 | 0.0022 | 0.0201 | -0.04544 | yes |
| `m5b_expanded_hgb__affine` | 0.0463 | 0.0068 | 0.0209 | -0.04214 | no |
| `expanded_random_forest__affine` | 0.0493 | 0.0067 | 0.0195 | -0.04151 | no |
| `m5b_expanded_hgb` | 0.0457 | 0.0099 | 0.0219 | -0.03762 | no |
| `frozen_m3` | 0.0808 | 0.0162 | 0.0200 | 0.00000 | no |
| `frozen_m3__affine` | 0.0730 | 0.0332 | 0.0249 | 0.03581 | no |

**Final test results by slice (frozen evaluation, 12 held-out sources).**

| candidate | geometric MAE / RMSE / R² | clean MAE / RMSE | original MAE / RMSE | M3 test MAE / RMSE / R² |
| --- | --- | --- | --- | --- |
| `frozen_m3` | 0.0777 / 0.0967 / -1.1858 | 0.0004 / 0.0012 | 0.0176 / 0.0240 | 0.0157 / 0.0226 / 0.9865 |
| `m5b_expanded_hgb` | 0.0544 / 0.0659 / -0.0165 | 0.0134 / 0.0170 | 0.0184 / 0.0251 | 0.0179 / 0.0243 / 0.9843 |
| `expanded_random_forest` (selected) | 0.0493 / 0.0590 / 0.1855 | 0.0013 / 0.0023 | 0.0174 / 0.0233 | 0.0156 / 0.0220 / 0.9872 |

The selected policy reduces primary geometric RMSE by 39.0% against the frozen
reference and is the only candidate whose geometric R² is positive. By
severity it is worse than the frozen model at 1.0 px (MAE 0.0570 versus
0.0259) and much better at 2.0 px (0.0195 versus 0.0662) and 3.0 px (0.0714
versus 0.1409). It also beats the M5B HGB at 2.0 px (0.0195 versus 0.0363) and
3.0 px (0.0714 versus 0.0749). On the original M3 test rows it stays level
with the frozen model (MAE 0.0156 versus 0.0157, RMSE 0.0220 versus 0.0226,
R² 0.9872 versus 0.9865), so the original task is retained. The aggregate gain
therefore comes with a real cost at the mildest included rung, where the
frozen model is still competitive.

**Paired source-grouped differences against the frozen M3 model.** Whole source
groups are resampled; differences are on identical rows. With 12 held-out
sources these intervals are preliminary and descriptive, and no significance
claim is made from interval overlap or non-overlap.

| scope | metric | estimate | 95% low | 95% high |
| --- | --- | ---: | ---: | ---: |
| geometric primary | MAE | -0.0284 | -0.0423 | -0.0159 |
| geometric primary | RMSE | -0.0377 | -0.0524 | -0.0232 |
| clean reference | MAE | +0.0009 | +0.0004 | +0.0015 |
| clean reference | RMSE | +0.0011 | +0.0006 | +0.0017 |
| original blur/noise | RMSE | -0.0006 | -0.0026 | +0.0006 |
| M3 test overall | RMSE | -0.0006 | -0.0024 | +0.0005 |
| diagnostic 0.5 px | RMSE | +0.0128 | -0.0016 | +0.0309 |

**Consistency across source groups.** The geometric improvement holds on 11 of
12 held-out sources, and the single best source accounts for 22.9% of the total
gain, so it is not driven by one outlier. On the original-task slices the
per-source gains and losses nearly cancel, so no broad improvement or
regression is claimed there.

**Observations on the three candidate explanations.**

1. *The expanded model learned the new distribution but lost original-task
   behaviour.* The M5B HGB does lose clean-reference accuracy (RMSE 0.0170
   against the frozen 0.0012) and original-task accuracy (0.0251 against
   0.0240), so the M5B result is consistent with this explanation. The
   alternative model family does not show that loss.
2. *The clean-reference errors reflect a correctable systematic offset.* The
   M5B HGB's clean-reference error is a near-constant offset (mean residual
   +0.0122), and the affine calibration reduces it to +0.0038, so the offset is
   partly correctable. But a single global affine map cannot remove it without
   moving the other slices: the calibrated HGB still leaves clean-reference
   RMSE at 0.0126, well above the frozen 0.0012. Calibration alone does not
   resolve the trade-off.
3. *A single model across corruption families may not be supportable.* The
   selected policy keeps the original blur/noise RMSE at 0.0233 against the
   frozen 0.0240 while improving the geometric slice, so the eight image-only
   features do support one model across both corruption families. The residual
   difficulty is between-source variation at fixed severity, which the
   aggregate score does not remove.

A caveat on the reference: the frozen M3 model was itself selected using the
M3 validation split, and its clean-reference validation slice (RMSE 0.0162) is
unusually noisy relative to its clean-reference test slice (0.0012). The
retention deltas are therefore measured against a reference whose own
validation clean error exceeds its test clean error, which makes the measured
clean regression of the selected policy an upper bound rather than a point
estimate.

**Conclusion.** The retention trade-off was resolved on this protocol, and
mostly by the *model family* rather than by calibration: a bagged-tree
expansion retains the original task and clean references far better than the
boosted expansion M5B selected, while scoring better on the geometric slice.
The affine calibration was tested and did not help the selected policy, and it
does not rescue the M5B HGB's clean-reference loss. Conventional feature-based
ML remains the benchmark champion. The evidence does not justify a CNN, and a
new corruption family does not by itself justify one.

**Limitations.** One synthetic renderer, one 192x64 canvas, three fonts, one
smooth elastic-warp family and 60 sources. Only 12 held-out test sources, so
all grouped intervals are preliminary and should not be read as significance
tests. The 0.5 px rung is close to the interpolation floor and is diagnostic
only. The calibration is univariate and global, so it can only correct a
systematic offset or slope, not a slice-dependent distortion; a
slice-conditional calibration was deliberately not attempted because it would
require the corruption family at inference time. The frozen reference's
validation clean slice is noisy, as described above. SSIM is a full-reference
similarity proxy, not human readability or CAPTCHA solvability. No external
CAPTCHA data or production system was used.

**Next step.** Stop M5C here. Do not start CNN training, another corruption
family, a UI, deployment or presentation work. If this line continues, the
next question is whether a retention-aware *training* objective, rather than a
post-hoc selection or calibration policy, can remove the remaining small
clean-reference cost without giving up the geometric gain. Do not retune on
the M5A/M3 test rows.

**Artifacts.** Git-ignored:
`models/retention_aware_selection_v0_1_0/` (`expanded_random_forest.joblib` and
card, `selected_policy.joblib` and card) and
`reports/retention_aware_selection_v0_1_0/` (`report.md`, validation slice
metrics, selection table, test slice metrics, predictions, uncertainty, paired
uncertainty, isolation checks and `run_metadata.json`). Expanded model
artifact sha256
`c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0`;
alternative card sha256
`520c0331ff8d7b6a8f3e1cd0dbf0687c9736fc97af9d7e94192bd33134156ad6`;
selected policy card sha256
`c1a89fb141df5f308d9438030bc006251abad1df9cb41824f5c06d3a754abd2e`.
Regenerate with `uv run --extra image --extra ml captcha-retention-aware run
--config configs/retention_aware_selection_v0_1_0.toml`; verify with
`captcha-retention-aware verify --config
configs/retention_aware_selection_v0_1_0.toml`.

---

### EXP-0007 — Grouped geometric-robustness benchmark: adaptation improves the primary score but not clean retention

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Does the frozen M3 feature-only model transfer
  zero-shot to the held-out M5A smooth-elastic test sources, and does a
  controlled conventional model trained on M3 plus M5A training-source rows
  improve held-out geometric performance without fitting anything on test?
  Does adaptation preserve the original M3 blur/noise task? This is a synthetic
  image-similarity benchmark, not a human-readability, CAPTCHA-solvability or
  security study.
- **Code revision:** base `5213eae` plus the M5B implementation, committed as
  `feat: benchmark geometric distortion robustness`. The reported run was
  repeated after the commit so generated provenance records a clean revision.- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; scikit-image 0.26.0; joblib 1.6.0; macOS 27.0.1 (arm64). No GPU was
  used.
- **Configuration:** `configs/geometric_robustness_benchmark_v0_1_0.toml`; config
  sha256
  `fe5c66614cee83c72a80c20b807202deadd8a604e045f7432795a8abe706ddb1`;
  protocol hash
  `77e41dd5de84e0bf8751c38a49b510735095f86edc8b00f6c19a6d8fde0d13ce`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates; selection on
  expanded validation RMSE among 12 predeclared conventional candidates.
- **Dataset:** frozen M3 `v0.1.0` (manifest
  `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`)
  plus M5A `v0.1.0` (manifest
  `c63be1088db591181f23fa306f955b3ba16fe80ffb2000c4a5e3db7656fa3806`;
  features
  `cb82ae60d8749f21d9d5f6dd324cd075e907544aa54bcdb62bbcab988472bb65`;
  pilot generation hash
  `e5901ce89fda1854d606aa41b6258222aa77e9bde061c9082971d7660a924c2a`;
  image-pixel-hash set
  `33d14063be82a33cd6518b8fd79c4a78838f601b859b2eede56de36b2a33264b`;
  M5A gate `CONDITIONAL PASS`).
- **Split:** M3 assignment
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`,
  grouped by clean source: 36 train / 12 validation / 12 test sources.
  Expanded training: 432 rows (324 M3 + 108 M5A primary geometric);
  expanded validation: 144 rows (108 M3 + 36 M5A primary geometric).
  Primary geometric test: 36 rows / 12 sources at 1.0–3.0 px. Diagnostic
  0.5 px test: 12 rows / 12 sources. Every M5A variant of a source stays in
  that source's frozen split.
- **Seeds:** benchmark seed `20261024`; derived candidate and bootstrap seeds.
- **Runtime:** 5.99 s in process on the recorded machine (not a benchmark).

**Method.** Experiment A evaluates the frozen M3 primary bundle, loaded through
its existing card/artifact verifier, on M5A test sources with no fitting,
scaling, calibration or preprocessing on those rows. The M3 training mean is
the naive baseline. Experiment B fits the predeclared OLS/ridge/random-forest/
histogram-gradient-boosting grid on M3 train rows plus M5A 1.0–3.0 px elastic
rows from training sources, selects on expanded validation sources only, and
evaluates once on test. M5A clean rows duplicate the M3 clean references; the
identity and zero-field controls are exact duplicates and the minimal control is
near-clean. All 240 clean/control rows are excluded from fit and scoring.
Experiment C evaluates the selected adapted model on the untouched M3 test rows
and compares it with frozen M3 on exactly the same rows, without retuning.

The primary geometric score uses only the 1.0, 2.0 and 3.0 px rungs. The 0.5 px
rung is reported separately because the M5A gate found its between-source SD
(0.0025) below the predeclared 0.005 margin. It never enters training,
validation selection or the primary score. Inputs remain exactly the eight
image-only features in their existing order; target remains `ssim_target`.

**Metrics.**

| Metric | Value | 95% grouped interval | Notes |
| --- | --- | --- | --- |
| Frozen M3 primary MAE | 0.0777 | 0.0629–0.0941 | 36 rows / 12 sources, 1–3 px |
| Frozen M3 primary RMSE | 0.0967 | 0.0784–0.1151 | Preliminary; 12 groups |
| Frozen M3 primary R² | -1.1858 | -1.8663–-0.6351 | Fails absolute calibration |
| M3 training-mean primary MAE | 0.2030 | — | Naive baseline |
| Adapted primary MAE | 0.0544 | 0.0449–0.0636 | Selected HGB |
| Adapted primary RMSE | 0.0659 | 0.0548–0.0753 | 31.8% lower than frozen M3 |
| Adapted primary R² | -0.0165 | -0.2856–0.2241 | Near zero; between-source variation unresolved |
| Paired primary RMSE difference | -0.0307 | -0.0472–-0.0155 | Adapted minus frozen; descriptive |
| Frozen 0.5 px diagnostic MAE | 0.0171 | 0.0091–0.0268 | Not in primary score |
| Adapted 0.5 px diagnostic MAE | 0.0170 | 0.0107–0.0240 | Not in primary score |
| Frozen M3 test MAE | 0.0157 | 0.0116–0.0203 | Original-task retention, 108 rows |
| Adapted M3 test MAE | 0.0179 | 0.0138–0.0231 | Same rows |
| Frozen M3 test RMSE | 0.0226 | 0.0156–0.0302 | Same rows |
| Adapted M3 test RMSE | 0.0243 | 0.0164–0.0333 | +7.7% point estimate |
| Paired M3 RMSE difference | 0.0017 | -0.0022–0.0042 | Includes zero; not a significance test |
| Frozen clean-slice MAE | 0.00036 | — | M3 test clean rows |
| Adapted clean-slice MAE | 0.01337 | — | Large retention regression |
| Clean paired MAE difference | 0.0130 | 0.0074–0.0186 | Adapted minus frozen; descriptive |

Validation selected `hist_gradient_boosting_lr_0.1_leaves_31` (validation RMSE
`0.0490`; the exact value is in `candidate_validation.csv`). The expanded model
artifact sha256 is
`360b7d18ca5e196b5f3fcd0cbe35c3c1f1f8b1ca0cf69672eb55a61be6479f9d`.
The estimator bytes are revision-independent and were byte-identical across two
independent runs. The JSON model card embeds the code revision, so its sha256
is recorded in the generated card and `run_metadata.json` rather than pinned
here; the final run was repeated after the commit so those generated files
record a clean revision. The frozen M3 primary model was loaded unmodified:
card sha256
`26d89292a59964846acf3e8bab4b84a2bf41b17863608c620c35d75975ecca73`,
artifact sha256
`d727bdd48b1511c2e8e59a072bf43868c5dd997d71e6a884aa505c7b275fa085`.

**Observations.** Frozen M3 is dramatically better than the M3 training mean
(61.7% lower MAE) but still fails absolute calibration: it under-predicts
geometric SSIM by 0.072 on average, and its R² is -1.186. The adapted model
lowers aggregate MAE by 30.0% and RMSE by 31.8%, but its R² remains essentially
zero (-0.017), so it explains almost none of the between-source variation at
fixed severity. The gain is not uniform: adapted MAE is worse at 1.0 px
(0.052 versus 0.026), better at 2.0 px (0.036 versus 0.066) and much better at
3.0 px (0.075 versus 0.141). At the diagnostic 0.5 px rung the two models are
nearly indistinguishable (MAE 0.0170 versus 0.0171); that rung is
interpolation-dominated and carries no primary claim. On the original M3 test
rows the adapted model's MAE rises from 0.0157 to 0.0179 and RMSE from 0.0226
to 0.0243, while R² remains 0.9843. The aggregate paired retention RMSE
difference includes zero, but the clean-reference slice is much worse
(MAE 0.0134 versus 0.00036). No significance claim is made from interval
overlap or non-overlap.

**Conclusion.** Conventional feature-based ML remains the benchmark family:
adaptation substantially reduces aggregate geometric error without test
leakage, and the frozen M3 model remains strong on the original blur/noise
task. The adapted model is **not** a drop-in replacement, because it loses
clean-reference calibration and only improves the low-severity rung at the cost
of the 1.0 px rung. The evidence does not justify moving to a CNN on this split.
The next modelling question is retention-aware selection or calibration under a
new predeclared protocol, not a larger model.

**Limitations.** One synthetic renderer, one canvas size, three fonts, one
smooth elastic-warp family and 60 sources. Only 12 held-out test sources, so all
grouped intervals are preliminary and should not be read as significance tests.
The 0.5 px rung is close to the interpolation floor and is diagnostic only.
Bilinear resampling and edge padding remain part of the corruption; the M5A
minimal control and border diagnostics quantify but do not remove that confound.
The adapted model's training target distribution differs from the geometric test
distribution, and the clean-slice retention result shows the consequence.
SSIM is a full-reference similarity proxy, not human readability or CAPTCHA
solvability. No external CAPTCHA data or production system was used.

**Next step.** Stop M5B here. Do not start CNN training, another corruption
family, a UI, deployment or presentation work. If this line continues, first
predeclare a protocol that treats clean-reference retention as a reported
validation objective alongside geometric error, and re-evaluate conventional
models under that protocol. Do not retune on the M5A/M3 test rows.

**Artifacts.** Git-ignored:
`models/geometric_robustness_benchmark_v0_1_0/`
(`expanded_geometric_model.joblib`,
`expanded_geometric_model_card.json`) and
`reports/geometric_robustness_benchmark_v0_1_0/` (`benchmark.md`, data
accounting, split assignment, isolation checks, zero-shot/adaptation metrics,
predictions, residuals, candidate validation, M3 reference/retention metrics,
retention comparison, target distribution, uncertainty and
`run_metadata.json`). Regenerate with `uv run --extra image --extra ml
captcha-geometric-benchmark run --config
configs/geometric_robustness_benchmark_v0_1_0.toml`; verify with
`captcha-geometric-benchmark verify --config
configs/geometric_robustness_benchmark_v0_1_0.toml`.

---

### EXP-0006 — Geometric-distortion feasibility pilot: replayable and monotonic, conditional pass on the lowest rung

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Before adding a geometric family to the grouped
  robustness protocol, does a bounded smooth elastic warp produce a valid,
  deterministic and replayable SSIM target with an interpretable severity
  response and a feature pattern that is not merely a reparameterisation of
  blur, noise or occlusion/interference? This is a synthetic image-similarity
  pilot, not a human-readability, CAPTCHA-solvability or security study.
- **Code revision:** base `968d85b` plus the M5A implementation, committed as
  `feat: pilot geometric distortion robustness`. The reported run was repeated
  after the commit so generated provenance records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-image
  0.26.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/geometric_distortion_pilot_v0_1_0.toml`; config
  sha256 `00b8c5caad610fb4aaf159bb8c99e329ad217aeff6450e7cd3e566387e9f0a97`;
  pilot generation hash
  `e5901ce89fda1854d606aa41b6258222aa77e9bde061c9082971d7660a924c2a`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates. The geometry was
  fixed before generation: amplitudes `[0.5, 1.0, 2.0, 3.0]` px maximum
  displacement, minimal control `0.25` px, 5x13 coarse control grid, 2.0 px
  Gaussian smoothing, bilinear interpolation, edge border mode.
- **Datasets:** frozen M3 `v0.1.0` used as the clean reference: 60 sources, 540
  rows, manifest sha256
  `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`,
  features sha256
  `d3715e7975cf41c41e49fa2338ca4e961a7a59981318be19149c78e89804d798`,
  split-assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`.
  New pilot `v0.1.0`: 60 sources, 480 rows = 60 clean + 180 controls (60
  identity, 60 zero-field, 60 minimal) + 240 elastic (4 severities x 60);
  manifest sha256
  `c63be1088db591181f23fa306f955b3ba16fe80ffb2000c4a5e3db7656fa3806`,
  sources sha256
  `b565aed1f3d84d7d9c2685e8145868208d77c24b831a1f8c205778e2196a7ed0`,
  features sha256
  `cb82ae60d8749f21d9d5f6dd324cd075e907544aa54bcdb62bbcab988472bb65`,
  image-pixel-hash set
  `33d14063be82a33cd6518b8fd79c4a78838f601b859b2eede56de36b2a33264b`.
  Frozen M4A comparison: manifest
  `d08c1824ba81e3bc40a585b3047eec6c5f9aa2da0b0b9eb126057ab27fdcc92b`,
  features
  `261cfe9a5e2c3910f08361c800f7bcd7252c86d7b56c32006c7970bfb63a1ad9`,
  summary
  `cd9dc97ff3a25c85976b26367da4a5b5f475d8dc8aeba671a55aecb981705ed4`.
- **Split:** frozen source-grouped split, 36/12/12 train/validation/test
  sources; every clean source and all its controls/severities stay in its
  assigned split. All 60 clean references were re-rendered exactly.
- **Seeds:** pilot seed `20261024`; per-source field seed derived from pilot
  seed, `geometric`, and `source_id`; 1000 source-grouped bootstrap replicates.
- **Runtime:** 5.7 s through the CLI, including frozen-asset verification,
  exact clean re-rendering, two independent builds, 420 transformed-image
  replays, analysis, gate and report.

**Method.** Loaded and hash-checked the frozen M3 dataset, M4A pilot and M4B
artifacts; re-rendered all 60 clean sources from
`configs/dataset_v0_1_0.toml` and required exact pixel equality. Generated one
seeded unit-maximum displacement field per source, then every severity and
control directly from the clean reference by scaling that same field. The field
is zero-mean before normalisation, so there is no global translation. Stored the
full deterministic field specification plus an exact SHA-256 of the float64
field; verification regenerates the field and transformed pixels without
consulting the target. Target and features are unchanged: `ssim_target` against
the exact clean reference (PSNR secondary) and the existing eight image-only
features in their fixed order. Controls: identity bypasses resampling;
zero-field passes integer coordinates through the bilinear sampler; minimal
applies 0.25 px of the same field. Two independent builds were compared by
manifest, source table, feature table, targets and decoded image hashes, and the
written dataset was re-verified from disk.

**Target behaviour.**

| Severity | Amplitude (px) | n | SSIM mean (SD) | SSIM range | PSNR mean (dB) | Changed pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5 | 60 | 0.9923 (0.0025) | 0.9868-0.9975 | 31.41 | 0.212 |
| 2 | 1.0 | 60 | 0.9713 (0.0091) | 0.9514-0.9900 | 25.54 | 0.255 |
| 3 | 2.0 | 60 | 0.9070 (0.0260) | 0.8526-0.9625 | 19.95 | 0.290 |
| 4 | 3.0 | 60 | 0.8372 (0.0396) | 0.7571-0.9242 | 17.13 | 0.307 |

Across all 240 elastic rows: SSIM mean 0.9270, SD 0.0653, range
0.7571-0.9975. Every one of the 60 source ladders had Spearman -1.0; all 180
adjacent steps decreased, with 0 tied and 0 increasing. Mean adjacent drops were
-0.0210, -0.0643 and -0.0697 from severity 1 to 4. No elastic row was
near-identity at `SSIM >= 0.9999`, no row was degenerate, and no elastic image
duplicated another or a clean reference. Grouped bootstrap 95% intervals for
mean SSIM were 0.9917-0.9929, 0.9689-0.9735, 0.9010-0.9134 and 0.8273-0.8470 for
severities 1-4. Font and text-length slices were descriptively similar; all
included groups showed the same ordering.

**Controls and interpolation/boundary diagnostics.** Identity and zero-field
controls reproduced all 60 clean images exactly (SSIM 1.0, no pixels changed).
The 0.25 px minimal control had mean SSIM 0.99803 (minimum 0.99662), changed
17.1% of pixels, and touched 2.09% of pixels through the border policy. At
severity 4, mean outside-canvas/border-affected fraction was 0.0261, mean
bilinear-clipped-sample fraction 0.0046, and mean collapsed 8x8 block fraction
0.0426. These are reported as a confound, not subtracted from SSIM: the minimal
control is a direct near-identity resampling reference, while the border and
block diagnostics isolate different failure modes.

**Cross-family comparison.** Frozen blur/noise and M4A thin-line/arc features
were loaded and hash-verified; the elastic feature means were contrasted at the
same severity. At severity 4, mean absolute Cohen's d was 5.21 against
`gaussian_blur`, 6.49 against `gaussian_noise`, 5.67 against `thin_lines` and
4.28 against `arc_strokes`; 4-8 of the 8 features exceeded |d| = 0.8 in each
contrast. Pooled feature/target correlations still mix severity and are not the
honest summary: for example `laplacian_variance` had pooled Spearman 0.616 but
mean within-severity Spearman -0.075, and `edge_density` had pooled -0.095 but
within-severity -0.304. The eight features clearly change under elastic warp,
but the direction of the target relationship is severity-dependent; this is a
feature-pattern result, not proof of semantic distinctness or human difficulty.

**Gate.** The predeclared gate returned **CONDITIONAL PASS**. Reproducibility,
identity/zero-field exactness, overall target spread (SD 0.065, level-mean range
0.155), trend, interpolation/boundary bounds, feature distinctness and
duplicate-freedom passed. The single failed margin was per-level SD at severity
1: 0.0025 against the predeclared 0.005. That rung is close to the interpolation
floor (0.25 px control SSIM 0.998), so its between-source spread is small even
though the overall target is non-degenerate and every source trajectory is
monotonic. The verdict is therefore conditional, not an unqualified PASS, and
the failed criterion is retained in `gate_results.json` and the report.

**Conclusion.** The smooth elastic family is correctly implemented, deterministic,
exactly replayable, non-destructive, and produces an interpretable monotonic
SSIM response across all 60 frozen sources. Its features are strongly separated
from blur, noise and interference families at high severity. The lowest rung is
too close to identity to support a between-source variance claim, so a separate
benchmark should either drop that rung, use a larger minimum amplitude, or treat
the low end as interpolation-dominated. No model was trained and no CNN claim
follows from this pilot.

**Limitations.**

- One synthetic renderer, one 192x64 canvas, three fonts and 60 sources.
- Bilinear interpolation and edge padding necessarily change pixels; the
  minimal control and border fractions quantify but cannot eliminate the
  confound.
- SSIM is a full-reference structural proxy, not readability or solvability.
- The elastic family is one smooth field design; it does not cover rotation,
  perspective, compression or all geometric distortions.
- The lowest severity has low between-source variance; aggregate means at that
  rung should not be over-interpreted.
- Feature contrasts use descriptive Cohen's d with 60 sources and no
  significance claim.

**Next step.** Stop M5A here. Do not start M5B, retrain a model, train a CNN,
build a demo or create the final presentation. If a separate geometric benchmark
is later approved, predeclare a revised low-end ladder or drop the 0.5 px rung,
keep the source-grouped split, carry the field specification and controls, and
report the interpolation confound explicitly.

**Artifacts.** Git-ignored: `data/processed/geometric_distortion_pilot_v0.1.0/`
(480 images, manifest, sources, features, summary) and
`reports/geometric_distortion_pilot/` (`pilot.md`, level/uncertainty/trend/
control/group/feature tables, `feature_effect_sizes.csv`,
`feature_distinctness.csv`, `feature_target_correlations.csv`,
`gate_results.json`, `examples_elastic.png`). Regenerate with
`uv run --extra image captcha-geometric-pilot run --config
configs/geometric_distortion_pilot_v0_1_0.toml`; verify with
`captcha-geometric-pilot verify --config
configs/geometric_distortion_pilot_v0_1_0.toml`.

---

### EXP-0005 — Grouped occlusion/interference benchmark: zero-shot failure, conventional adaptation success

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Does the frozen M3 feature-only model transfer
  zero-shot to the held-out M4A occlusion/interference test sources, and does a
  controlled conventional model trained on M3 plus M4A training-source data
  improve held-out performance without fitting anything on test? This is a
  synthetic image-similarity benchmark, not a human-readability or CAPTCHA
  solvability study.
- **Code revision:** base `fd98fcf` plus the M4B implementation, committed as
  `feat: benchmark interference robustness`. The reported run was repeated
  after the commit so `run_metadata.json` records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; scikit-image 0.26.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/robustness_benchmark_v0_1_0.toml`; config sha256
  `da1b9f95bebcf359dc920498563579fc5b56ec395c092c17689212160c8326f4`;
  protocol hash
  `bf85b329a27020e244651d49d695fb04c2c29e9f3c8bd66d4a3913bf86792d53`.
  Seed `20261024`; 1000 source-grouped bootstrap replicates; selection on
  validation RMSE. The frozen configs were not edited: dataset config sha256
  `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`;
  M3 benchmark config sha256
  `bea99c4b5ec94bc603c381c2d70366a1a0f411c67c94f61c05df64639e92ac57`;
  M4A pilot config sha256
  `2ef025220beb1b17aab4593e862fbbd11d2a6618eb8d3ed43bc532f6e3924917`.
- **Datasets:** frozen M3 `v0.1.0`, 540 rows / 60 sources, manifest sha256
  `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`,
  features sha256
  `d3715e7975cf41c41e49fa2338ca4e961a7a59981318be19149c78e89804d798`;
  M4A pilot `v0.1.0`, 540 rows / 60 sources (60 clean + 480 corrupted),
  manifest sha256
  `d08c1824ba81e3bc40a585b3047eec6c5f9aa2da0b0b9eb126057ab27fdcc92b`,
  features sha256
  `261cfe9a5e2c3910f08361c800f7bcd7252c86d7b56c32006c7970bfb63a1ad9`.
- **Split:** frozen source-grouped split, assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`;
  36/12/12 train/val/test sources for both datasets. All variants and severity
  levels of a source remain in its assigned split.
- **Merged data:** expanded train 612 rows / 36 sources = 324 M3 rows (36 clean
  + 288 blur/noise) + 288 M4A corrupted rows; expanded validation 204 rows /
  12 sources = 108 M3 + 96 M4A corrupted rows. M4A clean rows were excluded
  from expanded train/validation because all 60 are exact duplicates of the M3
  clean references. M4A test evaluation used 108 rows / 12 sources, including
  12 clean-reference rows. No duplicate sample IDs or image hashes in the
  expanded fit frames; all six isolation/leakage checks passed.
- **Model:** frozen M3 primary model, card sha256
  `26d89292a59964846acf3e8bab4b84a2bf41b17863608c620c35d75975ecca73`,
  artifact sha256
  `d727bdd48b1511c2e8e59a072bf43868c5dd997d71e6a884aa505c7b275fa085`.
  Expanded grid: OLS, 3 Ridge alphas, 4 random forests and 4 histogram gradient
  boosting configurations (12 candidates), fixed before test evaluation.
- **Runtime:** 8.07 s through the CLI, including frozen-image verification,
  M3/M4A target replay, 12-candidate fitting, grouped bootstrap, report and
  model persistence.

**Method.** Loaded and hash-checked both datasets and the frozen M3 bundle
through the existing verifier. Verified M3 with the full dataset verifier and
M4A with the pilot verifier (13 checks), then cross-checked that the two source
tables agree on IDs, splits, fonts and text lengths and that the 60 M4A clean
rows are exact image/feature/target duplicates of the M3 clean rows. Used the
eight existing image-only features and the unchanged SSIM target. Experiment A
evaluated the frozen M3 model and an M3-training-mean baseline on M4A test.
Experiment B fitted the 12-candidate grid on expanded train rows, selected on
expanded validation rows only, and evaluated the selected model plus an
expanded-training-mean baseline once on M4A test. No scaler, calibration,
feature-selection or hyperparameter decision used test rows. Source-grouped
bootstrap resampled the 12 test sources, not images.

**Experiment A — frozen M3 zero-shot on M4A test (108 rows / 12 sources).**

| Model | Slice | n | MAE | RMSE | R² | Mean signed residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| frozen M3 | overall | 108 | 0.1295 | 0.1612 | -1.0455 | -0.1294 |
| frozen M3 | clean | 12 | 0.0004 | 0.0012 | n/a | 0.0004 |
| frozen M3 | corrupted | 96 | 0.1456 | 0.1710 | -1.6031 | -0.1456 |
| frozen M3 | `thin_lines` | 48 | 0.1590 | 0.1876 | -1.6062 | -0.1590 |
| frozen M3 | `arc_strokes` | 48 | 0.1322 | 0.1527 | -1.7624 | -0.1322 |
| M3 training mean | overall | 108 | 0.1688 | 0.1948 | -1.9853 | 0.1589 |
| M3 training mean | corrupted | 96 | 0.1516 | 0.1759 | -1.7545 | 0.1404 |

The frozen model was near-perfect on clean references but under-predicted
increasingly with density: MAE 0.0496, 0.0990, 0.1690 and 0.2649 at severities
1-4. Its overall MAE advantage over the M3 training mean is therefore driven by
the clean slice. On corrupted rows the paired source-grouped MAE difference was
-0.0060 with 95% interval [-0.0307, 0.0129], which includes zero; on
`thin_lines` the frozen model was descriptively worse (+0.0141, interval
[-0.0124, 0.0388]). Negative R² means it was worse than a constant test-mean
predictor in squared-error terms. Zero-shot absolute calibration does not
transfer to this interference distribution.

**Experiment B — conventional adaptation on M4A test.**

| Model | Slice | n | MAE | RMSE | R² | Mean signed residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| expanded model | overall | 108 | 0.0131 | 0.0178 | 0.9752 | 0.0047 |
| expanded model | clean | 12 | 0.0024 | 0.0074 | n/a | 0.0024 |
| expanded model | corrupted | 96 | 0.0144 | 0.0187 | 0.9690 | 0.0050 |
| expanded model | `thin_lines` | 48 | 0.0162 | 0.0213 | 0.9664 | 0.0053 |
| expanded model | `arc_strokes` | 48 | 0.0126 | 0.0156 | 0.9712 | 0.0047 |
| expanded training mean | overall | 108 | 0.1261 | 0.1466 | -0.6912 | 0.0937 |
| expanded training mean | corrupted | 96 | 0.1117 | 0.1300 | -0.5044 | 0.0753 |

Validation selected `random_forest_depth_unlimited_leaf_1` (validation RMSE
0.0201; R² 0.9864). On test, the adapted model improved over frozen M3 by
-0.1164 MAE (95% interval [-0.1277, -0.1003]) and -0.1435 RMSE
([-0.1558, -0.1256]); the paired difference excluded zero at every severity and
both variants. The clean slice was slightly worse than frozen M3 (+0.0021 MAE,
interval [-0.0000, 0.0056]), but both were close to perfect and the interval
includes zero. Severity RMSE rose from 0.0143 to 0.0245 and mean signed residual
from +0.0027 to +0.0100, so the high-density end remains the hardest slice.
Test R² is unstable at low severity (0.445 at severity 1), where target variance
is small, despite low MAE.

On the original M3 blur/noise test rows, the expanded model remained close to
the frozen M3 reference: overall MAE/RMSE/R² 0.0158/0.0239/0.9849 versus
0.0157/0.0226/0.9865 for frozen M3. It did not sacrifice the original families
to gain interference performance.

**Target distribution.** Expanded training targets had mean 0.7587 (SD 0.1717,
range 0.3174-1.0); M4A test targets had mean 0.8524 (SD 0.1133, range
0.5532-1.0). M4A corrupted test mean was 0.8340 and expanded corrupted-only
train mean 0.8319; the lower full-expanded mean is caused by the severe M3 blur
and noise rows. The shift is disclosed and was not used to tune anything.

**Conclusion.** The frozen M3 model does not transfer useful absolute SSIM
calibration to M4A interference, despite near-perfect clean-reference
predictions; the failure is a target/covariate shift, not evidence about human
difficulty. A small conventional model trained on M3 plus training-source M4A
corrupted rows, selected on validation sources only, removes most of the error
and generalises descriptively across both variants and all four severities. On
this benchmark, conventional modelling is sufficient; a CNN is not justified by
these results. The 12-source intervals are preliminary and are not significance
claims.

**Limitations.**

- One synthetic renderer, one 192x64 canvas, three fonts and 60 sources.
- Only 12 held-out test sources; all intervals are wide and preliminary.
- M4A severity is nested stroke density, not an independent redraw at each
  level.
- The expanded model is trained on a mixture whose target distribution differs
  from M4A test; the shift is reported but not corrected.
- SSIM is a full-reference structural proxy, not readability or solvability.
- No cross-generator, real-image, human or security evaluation was performed.
- The clean-reference R² is undefined by construction and is not reported as a
  perfect fit.

**Next step.** Close M4B as a conventional-modelling result. Do not start a CNN,
dashboard or deployment on the basis of this benchmark. A future shift study
would need a new predeclared protocol, ideally with independent generators or
human labels, before model capacity is treated as the limiting factor.

**Artifacts.** Git-ignored: `reports/robustness_benchmark_v0_1_0/`
(`benchmark.md`, data accounting, isolation checks, zero-shot/adaptation
metrics, predictions, residuals, candidate validation, target distribution,
uncertainty, `run_metadata.json`) and
`models/robustness_benchmark_v0_1_0/` (`expanded_model.joblib` + card).
Regenerate with `uv run --extra image --extra ml captcha-robustness-benchmark
run --config configs/robustness_benchmark_v0_1_0.toml`; verify with
`captcha-robustness-benchmark verify --config
configs/robustness_benchmark_v0_1_0.toml`.

---

### EXP-0003 — Conventional image-quality models, leakage-safe comparison and held-out-font stress test

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `conventional-benchmark`
- **Hypothesis / question:** On the frozen 540-sample synthetic dataset, do
  conventional regressors on the eight image-only features improve materially
  over a training-mean baseline; does a nonlinear model beat OLS/Ridge on
  held-out sources; how do results vary by corruption family, severity and ink
  coverage; how much of the target is explained by corruption metadata alone;
  and does the selected model survive a genuinely held-out font? This is a
  synthetic image-similarity benchmark, not a human-readability or
  CAPTCHA-solvability study.
- **Code revision:** base `b9d6686` plus the Milestone 3 implementation; committed
  together in `feat: benchmark conventional image quality models`. The reported
  run was repeated after the commit so `run_metadata.json` and both model cards
  record a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; scikit-learn
  1.9.1; joblib 1.6.0; macOS 26 (arm64). No GPU, internet or Kaggle access was
  required.
- **Configuration:** `configs/benchmark_v0_1_0.toml`; benchmark config hash
  `e93d2660329a4f8b7e6b99739651250e2104ca9a26c76574db6db1e2e6bc8fa5`.
  The frozen dataset configs were not edited:
  `configs/dataset_v0_1_0.toml` sha256
  `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`;
  `configs/pilot_target_validation.toml` sha256
  `6e9bdf961f36bbf70f5806e41b53661e1577bf15e13eb833348632359b886715`.
- **Dataset:** `v0.1.0`; generation hash
  `820cce7a77b5e95761f033b75e8cb2b304b6ad6293f67b5bc27bcf0823a6cef9`;
  manifest sha256 `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`.
- **Split:** frozen source-grouped, font-stratified split; assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`.
  Sources: 36/12/12 train/val/test; samples: 324/108/108. Every source group is
  in exactly one split; no variant crosses a boundary.
- **Seeds:** benchmark seed `20261022`; model seeds derived per candidate from
  `(seed, "model", candidate)` and reduced to 32 bits for scikit-learn;
  bootstrap seeds derived from `(seed, "bootstrap", scope)`.
- **Runtime:** ~5.5 s in process, ~6.2 s wall through the CLI on this machine.
  The run fits 23 learned candidates on train, evaluates them on validation,
  then evaluates four family winners plus the naive baseline on test.

**Method.** Loaded the frozen dataset through `load_dataset`, enforced a 1:1
manifest/feature join, exact feature schema (`sample_id` + eight features in
documented order), finite values, target range, split membership, source-group
disjointness and the recorded split hash. Built a predeclared 23-candidate grid:
OLS (1), Ridge with six alphas (6), random forest with four depths x three leaf
sizes (12), and histogram gradient boosting with two learning rates x two leaf
counts (4). Each candidate was fitted on the training split only; scalers were
inside a pipeline fitted on those rows. Selection used validation RMSE only,
with ties broken by fewer inputs then name. The best candidate in each family
and the overall best candidate were then evaluated once on test. The candidate
table written to disk has no test columns.

Separately fitted diagnostic OLS models for severity+family, font+text length,
and image+metadata. These use generated labels, train-only category levels and
train-only scaling, and are not deployable image-only models. Post-selection
test slices were computed by corruption family, severity, train-quantile
ink-coverage quartile and font. Residuals and grouped bootstrap intervals
(resampling source groups, not variants) were reported. A separate held-out-font
experiment removed Times New Roman before fitting and validation, re-ran the
same grid on Arial + Courier New train/validation sources, selected on their
validation rows only, and evaluated once on all 20 Times New Roman sources (180
samples). The held-out font was chosen a priori because it is the only serif
face in the configuration; no performance information informed the choice.

**Metrics.**

Primary held-out test (108 samples, 12 source groups):

| Model | MAE | RMSE | R² | Notes |
| --- | ---: | ---: | ---: | --- |
| Naive train mean | 0.1664 | 0.1942 | -0.0000 | baseline |
| OLS, 8 features | 0.0302 | 0.0380 | 0.9618 | linear family winner |
| Ridge, alpha 0.01 | 0.0302 | 0.0379 | 0.9619 | ridge family winner |
| Random forest, depth 8, leaf 1 | 0.0157 | 0.0226 | 0.9865 | overall validation-selected |
| Histogram gradient boosting, lr 0.1, 15 leaves | 0.0174 | 0.0256 | 0.9826 | boosting family winner |

Selected-model train/validation/test RMSE: 0.0089 / 0.0196 / 0.0226. The
random forest's validation-selected configuration was nearly tied with unlimited
depth (validation RMSE 0.019635 vs 0.019665), so the exact forest winner is
unstable even though the family-level comparison is clear. HGB overfit strongly:
train RMSE 0.0021, validation 0.0216, test 0.0256.

Metadata diagnostics on test:

| Model | Inputs | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Severity + family | 3 | 0.0458 | 0.0579 | 0.9112 |
| Font + text length | 3 | 0.1660 | 0.1940 | 0.0023 |
| Image + metadata (OLS) | 11 | 0.0202 | 0.0242 | 0.9844 |

Gaps relative to the selected image-only model on test: metadata-only has
+0.0353 RMSE and -0.0752 R²; combined image+metadata has +0.0016 RMSE and
-0.0020 R². The combined diagnostic did not beat the selected random forest.

Test slices for the selected model (all counts in the generated tables):

| Slice | Samples | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Overall | 108 | 0.0157 | 0.0226 | 0.9865 |
| Corrupted only | 96 | 0.0176 | 0.0240 | 0.9804 |
| Blur | 48 | 0.0170 | 0.0255 | 0.9824 |
| Noise | 48 | 0.0182 | 0.0223 | 0.9721 |
| Clean | 12 | 0.0004 | 0.0012 | undefined |
| Coverage Q1 | 38 | 0.0140 | 0.0189 | 0.9882 |
| Coverage Q2 | 20 | 0.0106 | 0.0148 | 0.9902 |
| Coverage Q3 | 22 | 0.0202 | 0.0252 | 0.9630 |
| Coverage Q4 | 28 | 0.0180 | 0.0288 | 0.9456 |

Severity slices 1–4 had RMSE 0.0210 / 0.0312 / 0.0205 / 0.0214. Their R² values
(0.60 / 0.70 / 0.95 / 0.93) are lower at low severity because within-level target
variance is small, not because absolute error is larger; this is why both
absolute error and R² are reported. The clean gap is a constant-target subgroup,
so R² is undefined and is not filled in.

Grouped bootstrap (1,000 source-group replicates):

| Scope | Metric | Estimate | 95% interval | Groups |
| --- | --- | ---: | --- | ---: |
| Primary test | MAE | 0.0157 | [0.0116, 0.0209] | 12 |
| Primary test | RMSE | 0.0226 | [0.0157, 0.0309] | 12 |
| Primary test | R² | 0.9865 | [0.9757, 0.9934] | 12 |
| Held-out font | MAE | 0.0193 | [0.0157, 0.0233] | 20 |
| Held-out font | RMSE | 0.0271 | [0.0211, 0.0325] | 20 |
| Held-out font | R² | 0.9799 | [0.9712, 0.9876] | 20 |

Held-out font (Times New Roman; 20 sources, 180 samples; training fonts Arial +
Courier New, 24 training sources / 216 samples, 8 validation sources / 72
samples):

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Naive two-font train mean | 0.1628 | 0.1910 | -0.0006 |
| Validation-selected HGB, lr 0.05, 31 leaves | 0.0193 | 0.0271 | 0.9799 |
| Random-forest family winner | 0.0184 | 0.0248 | 0.9832 |
| OLS family winner | 0.0326 | 0.0398 | 0.9566 |
| Ridge family winner | 0.0326 | 0.0399 | 0.9564 |

The random-forest family winner was descriptively better on the held-out font
than the HGB configuration selected on the two-font validation split. The result
is reported as-is; the model was not re-selected on the held-out data.

**Observations.** Facts only:

1. **Conventional models beat the mean by a large margin.** Test RMSE fell from
   0.1942 to 0.0226 (88.4% relative), above the predeclared 5% materiality rule.
   This is expected because SSIM varies strongly with image structure, not
   evidence of a novel method.
2. **The feature/target relationship is nonlinear on this split.** The best
   nonlinear test RMSE (random forest, 0.0226) was 40.4% lower than the best
   linear one (ridge, 0.0379; OLS 0.0380). This is descriptive on 108 test
   samples.
3. **The exact forest winner is unstable.** The top two forest configurations
   differed by 0.00003 validation RMSE. The family-level result should be
   preferred over the exact depth/leaf choice.
4. **Tree models overfit.** HGB's training RMSE was 0.0021 versus validation
   0.0216; random forest training was 0.0089 versus validation 0.0196. The
   report exposes train/validation columns instead of reporting only test.
5. **Metadata explains most of the target.** Severity + family alone reached test
   R² 0.9112. Font + text length reached 0.0023, confirming no font/length
   shortcut in this split. Image features still improved on metadata, and the
   combined OLS diagnostic (0.9844) did not beat the image-only random forest
   (0.9865).
6. **Aggregate performance hides family and coverage variation.** Blur RMSE
   0.0255, noise 0.0223; coverage Q4 (highest ink) was hardest at 0.0288. Clean
   samples are near-perfect and make the pooled R² optimistic relative to
   corrupted-only performance (0.9804).
7. **The held-out-font stress test survives.** A model trained without Times New
   Roman reached R² 0.9799 on its 180 samples (selected HGB), and the random
   forest family winner reached 0.9832. This is not broad generalisation: only
   three fonts exist, and the held-out font shares the renderer and corruption
   design.
8. **Model verification passes.** Both persisted bundles passed card, schema,
   target, dataset-hash, split-hash, scikit-learn compatibility, artifact-hash,
   load and 8-feature smoke-test checks.
9. **Reproducibility.** The documented command was repeated; the report CSVs
   (excluding runtime metadata) and model predictions were compared. The
   benchmark is deterministic under the recorded seed; model artifact bytes are
   not claimed to be byte-identical across library versions.

**Conclusion.** On the frozen synthetic SSIM target, a conventional random
forest is materially better than the naive mean and descriptively better than
OLS/Ridge. The result is broadly stable across blur and noise but not uniform
across severity or coverage. Metadata alone explains 91.1% of held-out variance,
so the synthetic design is a strong confound; image-only features still explain
98.6%. The model survives a preliminary held-out-font stress test with R² 0.980.
For this target, a CNN is **not** justified: the conventional forest already
reconstructs almost all held-out variance and captures the nonlinearity, so a
CNN would mainly add cost and opacity. A CNN would be justified only on a harder,
predeclared shift where it beats this frozen feature-only baseline. A high R²
does not make the target scientifically interesting, and none of these numbers
measure human readability or CAPTCHA solvability.

**Limitations.**

- Three fonts, one canvas, two corruption families, 60 synthetic sources. Not a
  cross-generator or real-image evaluation.
- Test has 12 source groups; the grouped bootstrap intervals are wide and
  unstable. The exact forest configuration is nearly tied with alternatives.
- Validation and test share all three fonts in the primary benchmark. The
  held-out-font experiment removes one but leaves only two for training.
- Severity-1/2 R² is low because within-level variance is small; absolute errors
  are not correspondingly large.
- Clean samples have constant SSIM = 1, so R² is undefined there and pooled R²
  is not a corrupted-only estimate.
- Metadata diagnostics use generated labels and are not deployable inputs; the
  combined model is one reference, not a guaranteed upper bound.
- The model cards record scikit-learn's major/minor version; joblib artifacts
  are not safe for untrusted files and only locally generated bundles should be
  loaded.
- Text length is not stratified; validation contains no length-6 sources.

**Next step.** Do not start a CNN on this target. If Milestone 4 is requested,
first define a harder predeclared shift (for example a new corruption family or a
small documented real-image set) and require any CNN to beat the saved random
forest on the same grouped protocol. Otherwise retain the conventional model and
move to robustness analysis (M5).

**Artifacts.** Git-ignored: `reports/benchmark_v0_1_0/` (benchmark markdown,
candidate/comparison tables, predictions, residuals, strata, coverage,
correlations, held-out-font tables, uncertainty, `run_metadata.json`) and
`models/benchmark_v0_1_0/` (`primary_model.joblib` + card,
`heldout_font_model.joblib` + card). The benchmark CLI is
`uv run --extra image --extra ml captcha-benchmark run --config
configs/benchmark_v0_1_0.toml --dataset data/processed/captcha_quality_v0.1.0`;
verification is `captcha-benchmark verify-model`.

---

### EXP-0004 — Occlusion/interference pilot: reproducible and distinct, not a difficulty result

- **Date:** 2026-10-20
- **Status:** `completed`
- **Task:** `robustness-analysis`
- **Hypothesis / question:** Before adding a new corruption family to a grouped
  robustness benchmark, do two occlusion/interference variants produce a
  reproducible, non-degenerate SSIM target with an interpretable density
  response, and do their image features differ from the existing blur and noise
  families? This is a synthetic image-similarity pilot, not a human-readability,
  solvability or security study.
- **Code revision:** base `c781938` plus the M4A implementation, committed as
  `feat: pilot occlusion and interference robustness`. The reported run was
  repeated after the commit so `pilot_summary.json` records a clean revision.
- **Environment:** CPython 3.13.16; NumPy 2.5.4; pandas 3.0.6; Pillow 12.3.0;
  scikit-image 0.26.0; macOS 27.0.1 (arm64). No GPU was used.
- **Configuration:** `configs/robustness_pilot_occlusion_v0_1_0.toml`; config
  sha256 `2ef025220beb1b17…`; pilot generation hash
  `dc1a9b1557b3e8e49e86dc4e59beabf1d35d53c9f3a59300399a9f5a4e49e6f0`.
  The frozen configs were not edited: `configs/dataset_v0_1_0.toml` sha256
  `72a945af829b2b61a8f748f1712b5231254f0ef480591f10ea595d69a9ff812a`;
  `configs/benchmark_v0_1_0.toml` sha256
  `bea99c4b5ec94bc603c381c2d70366a1a0f411c67c94f61c05df64639e92ac57`.
- **Dataset:** frozen `v0.1.0`; generation hash
  `820cce7a77b5e95761f033b75e8cb2b304b6ad6293f67b5bc27bcf0823a6cef9`;
  manifest sha256 `a6a2cb8654eb51055e11e4aec6dbc94c2a59a6fdf8ba6b4946135bffc8b4cf37`.
  Pilot dataset: manifest sha256
  `d08c1824ba81e3bc40a585b3047eec6c5f9aa2da0b0b9eb126057ab27fdcc92b`,
  features sha256
  `261cfe9a5e2c3910f08361c800f7bcd7252c86d7b56c32006c7970bfb63a1ad9`,
  60 clean + 480 corrupted images.
- **Split:** frozen source-grouped, font-stratified split; assignment sha256
  `70c07d171d2e2b3032db86ce8c2942f1ca5dd1c3c01bbcd30a0fac0d7dda7232`.
  36/12/12 train/val/test sources; 324/108/108 frozen samples. The pilot
  preserves the source split and adds no pilot sample to the frozen test set.
- **Seeds:** pilot seed `20261023`; per-source/per-variant stroke-pool seeds
  derived from `(seed, "interference", source_id, variant)`; bootstrap seeds
  derived from `(seed, "bootstrap", variant, severity)`. All randomness is
  explicit; no global RNG state.
- **Runtime:** 2.91 s wall through the CLI on this machine, including the
  frozen-reference audit, two full in-memory builds, dataset verification,
  480 provenance replays, bootstrap, report generation and contact sheets.

**Method.** Verified every frozen artifact hash, re-rendered all 60 clean
sources from the frozen dataset config, and compared source IDs, render seeds,
text hashes, split labels and pixels. Generated two variants with 1 px black
strokes on the existing 192x64 canvas: `thin_lines` through an ink pivot with
line angles in [-60, 60]°, and `arc_strokes` with sweep 120-240°, radius
0.20-0.55 of width and aspect 0.50-1.50. Severity levels were 1, 2, 4 and 7
strokes. A shared seven-stroke pool was planned per `(source, variant)`, and
severity `k` used its first `k` strokes; every severity was rendered from the
same clean reference, never from another corrupted image. Geometry, seed,
width, intensity, hashes, changed-pixel fractions and border contact were
recorded. SSIM used the frozen M1 conventions; PSNR was secondary. The pilot
trained no model and did not use the target or model predictions to choose
parameters. The predeclared gate covered reproducibility, non-degeneracy,
trend, feature distinctness from blur/noise, and duplicate freedom.

**Metrics.** Source-grouped bootstrap used 1000 replicates over 60 sources, not
variants.

| Variant | Severity | Strokes | SSIM mean | SD | 95% CI | PSNR mean (dB) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `thin_lines` | 1 | 1 | 0.943 | 0.020 | [0.938, 0.947] | 22.20 |
| `thin_lines` | 2 | 2 | 0.887 | 0.032 | [0.879, 0.895] | 18.84 |
| `thin_lines` | 3 | 4 | 0.785 | 0.041 | [0.775, 0.795] | 15.65 |
| `thin_lines` | 4 | 7 | 0.662 | 0.044 | [0.651, 0.673] | 13.15 |
| `arc_strokes` | 1 | 1 | 0.947 | 0.022 | [0.942, 0.952] | 22.47 |
| `arc_strokes` | 2 | 2 | 0.902 | 0.029 | [0.894, 0.909] | 19.38 |
| `arc_strokes` | 3 | 4 | 0.818 | 0.037 | [0.809, 0.827] | 16.33 |
| `arc_strokes` | 4 | 7 | 0.715 | 0.042 | [0.705, 0.725] | 13.96 |

Across all 480 corrupted samples: SSIM mean 0.832, SD 0.105, min 0.545, max
0.979. The clean baseline is exactly SSIM = 1.0 by construction. All 120
source-variant ladders had Spearman(severity, SSIM) = -1.0; 0/360 adjacent
steps increased and 0 were tied. Level-mean PSNR decreased monotonically with
severity; per-source PSNR exceptions were not part of the gate.

At the highest severity, Cohen's d against the frozen families at the same
severity:

| Variant | Versus | Mean \|d\| | Features with \|d\| ≥ 0.8 |
| --- | --- | ---: | ---: |
| `thin_lines` | gaussian_blur | 8.29 | 8/8 |
| `thin_lines` | gaussian_noise | 4.02 | 7/8 |
| `arc_strokes` | gaussian_blur | 7.06 | 8/8 |
| `arc_strokes` | gaussian_noise | 3.58 | 7/8 |

Predeclared gate:

| Criterion | Threshold | Observed | Verdict |
| --- | --- | --- | --- |
| Reproducibility | Two builds identical; clean re-renders exact; verification passes | identical; 60/60 exact; 13/13 checks | PASS |
| Non-degeneracy | SD ≥ 0.02; level-mean range ≥ 0.10; level SD ≥ 0.005 | SD 0.105; ranges 0.281/0.233; min level SD 0.0197 | PASS |
| Interpretable trend | Median Spearman ≤ -0.80; increasing fraction ≤ 0.15 | -1.000; 0.000 | PASS |
| Feature distinctness | Mean \|d\| ≥ 0.50 and ≥2 features at \|d\| ≥ 0.80 vs both families | 3.58-8.29; 7-8 features | PASS |
| Duplicate-free | Duplicate fraction ≤ 0.0 | 0.000000 | PASS |

**Observations.** The family is non-degenerate at every level and the severity
response is strictly monotone for every source and variant under the nested
design. Marginal SSIM drop per added stroke declines with density (`thin_lines`
-0.057 to -0.041; `arc_strokes` -0.046 to -0.034), so the high end is mildly
saturated but not collapsed. Every corrupted image touches the canvas border by
construction; changed pixels cover 0.7-5.8% of the canvas and 0.6-4.8% of the
clean ink. Fixed black intensity means a stroke that overlaps existing glyph
cores can change fewer pixels, which is a real confound and a potential source
of flat steps. Feature/target associations also differ from blur: for example,
pooled Pearson between Laplacian variance and SSIM was +0.883 for blur but
-0.964 for `thin_lines` and -0.987 for `arc_strokes`; the within-severity signs
are reported in the tables, so the pooled values are not treated as causal.

**Conclusion.** The pilot supports proceeding to a separate grouped robustness
benchmark for this family: it is deterministic, exactly replayable from recorded
geometry, non-degenerate, interpretable in density, and clearly distinct from
blur/noise in the eight-feature space. It does **not** show that occlusion is
objectively harder or more human-solvable; those claims require a different
study. The pilot also does not establish cross-generator or real-image
generalisation.

**Limitations.**

- One synthetic renderer, one 192x64 canvas, three fonts and 60 sources. The
  feature contrast may change with other canvases, fonts or text lengths.
- Density is the only varied parameter. Width, intensity, colour and coverage
  are fixed; a full benchmark should predeclare whether they vary.
- The nested pool makes the severity response a dose of the same early strokes.
  This improves interpretability but is not an independent sample of geometry
  at each severity.
- Border contact is 100% by construction and is not text clipping, but it does
  mean edge effects are not varied.
- Fixed black strokes can disappear into glyph cores; changed-pixel fraction is
  a better manipulation check than nominal stroke count.
- SSIM is a full-reference structural proxy. It is not human readability or
  CAPTCHA solvability.
- No model was trained, so the pilot says nothing about model performance on
  this family.

**Next step.** If M4 proceeds, run a separate grouped robustness benchmark with
a predeclared protocol, keep these samples out of the frozen M3 test set, and
compare any model against the saved feature-only baseline. Do not treat the
pilot's SSIM levels as a difficulty ordering.

**Artifacts.** Git-ignored: `data/processed/occlusion_interference_pilot_v0.1.0/`
(manifest, sources, features, summary, 540 PNGs) and `reports/robustness_pilot/`
(`pilot.md`, level/uncertainty/trend/feature/saturation tables,
`gate_results.json`, contact sheets). Regenerate with
`uv run --extra image captcha-robustness-pilot run --config
configs/robustness_pilot_occlusion_v0_1_0.toml`.

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
