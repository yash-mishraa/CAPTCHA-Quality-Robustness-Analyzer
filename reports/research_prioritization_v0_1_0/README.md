# Research prioritization and project closure decision v0.1.0 (M9C)

## Decision

**Path A — stop experimentation and package the research.**

The project should freeze its scientific work at the M9B preregistered protocol,
finish the documentation and reproducibility release, and present the existing
evidence as a completed measurement and robustness study. It should not start
the H1, H2 or H3 experiments, should not run a new pilot, and should not begin a
confirmatory data-collection programme at this time.

This memo is a **decision and prioritization record**. It does **not** authorize
experiment execution, new data generation, model training or refitting,
calibration, feature engineering, model selection, benchmark modification, or a
replacement of the frozen M5C champion. Any future execution requires a new,
separately approved milestone and protocol.

## Status and evidence cut

- Milestone: **M9C**
- Status: **decision complete; no experiment executed**
- Decision: **Path A**
- Evidence cut: `81710e6c839f4e7f66752d4045c765df573acbe2`
- Evidence cut was `main == origin/main`, with a clean working tree.
- M8C is observed evidence and remains **exploratory** for the M9A hypotheses.
- M9A is a read-only exploratory analysis, not a confirmatory test.
- M9B is a preregistered design, **not executed**. All three hypotheses remain
  `not_tested`.
- The M5C champion remains frozen at artifact SHA-256
  `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0` and
  card SHA-256
  `9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8`.

## Why Path A

The evidence supports a credible stopping point, not an automatic extension of
the research programme.

1. **M8C and M9A already answer the project's stated descriptive question well
   enough for a portfolio release.** The frozen champion has a large,
   broad-based interference error: interference MAE is `0.0908`, thin-line MAE
   is `0.1021`, arc-stroke MAE is `0.0796`, and interference R-squared is
   `-0.4302` over 2,400 primary rows from 300 source groups. M9A shows that the
   error is not a handful of outliers: the top 5% of sources account for only
   `10.6%` of interference squared error. That is a defensible descriptive
   finding, explicitly labelled exploratory where it is used to motivate
   hypotheses.
2. **The strongest explanatory candidate, H1, still cannot support a causal or
   interventional claim.** M9A's within-source attenuation is large and
   consistent, but M9B's H1 design can only establish a compressed conditional
   response relative to blur/noise. It explicitly cannot separate training
   distribution shift, model capacity, target construction or feature
   limitations. A supported H1 would justify a *separate* intervention-design
   decision, not a model change.
3. **H2 has low explanatory leverage and high design risk.** The feature
   associations are descriptive, the eight features are strongly redundant, and
   prior feature studies (M6A-M6D) found no defensible gain from added
   features. H2 matching may be `not_testable`, and even support would not
   identify a feature or justify feature engineering.
4. **H3 is the most externally dependent and least causally identifiable.**
   It requires new licence-eligible thin and thick fonts, new cases, a
   predeclared stroke-width classification and matched case clusters. Even then
   it can only support a conditional association under the fixed renderer, not
   a causal font effect or a font-population claim.
5. **Path B would produce planning information, not a scientific answer.** A
   pilot is useful only if a confirmatory C programme is already justified.
   Running a pilot merely to keep the project active would add data and
   maintenance without changing any current claim.
6. **Path C is expensive relative to the marginal value.** It could provide one
   confirmatory H1 result, but it requires a pilot, an independently generated
   source set C, a frozen sample size, a one-shot analysis and potentially new
   fonts for H3. The project already has a strong, audited descriptive result;
   confirmatory evidence about a synthetic SSIM proxy would not establish human
   readability, CAPTCHA solvability, security or production robustness.

The recommendation is therefore not a judgement that the hypotheses are
unimportant. It is a judgement that the **current evidence and the project's
scope justify closure**, while preserving H1 as the first candidate if the
project is ever reopened with genuinely independent data and explicit approval.

## Package contents

| File | Purpose |
| --- | --- |
| `README.md` | decision summary, evidence snapshot and verification boundary |
| `research_prioritization.md` | full memo: hypothesis assessment, path comparison, recommendation and reopening rules |
| `hypothesis_comparison.csv` | one row per hypothesis with the requested criteria and disposition |
| `project_completion_checklist.md` | mandatory closure requirements, optional future work and definition of done |
| `verification_checks.csv` | checks performed for this memo, with actual outcomes |

There is no `run`, `fit`, `predict`, `generate`, `calibrate` or `select` entry
point in this package. It is a static decision artifact.

## Evidence snapshot

| Evidence | Observed value used here | Role |
| --- | --- | --- |
| M8C all-primary MAE | `0.0596` | descriptive benchmark context |
| M8C interference MAE | `0.0908` | largest primary family error |
| M8C thin-line MAE | `0.1021` | severe interference failure |
| M8C arc-stroke MAE | `0.0796` | severe interference failure |
| M8C interference R-squared | `-0.4302` | worse than the slice mean |
| M8C interference bias | `-0.087` | systematic overprediction |
| M8C thin s4 large-error share | `55.3%` at `|residual| >= 0.20` | failure concentration |
| M8C arc s4 large-error share | `28.0%` at `|residual| >= 0.20` | failure concentration |
| M8C top-5% interference SSE share | `10.6%` | broad, not outlier-driven |
| M9A thin within-source SD ratio | `0.320` | attenuation description |
| M9A arc within-source SD ratio | `0.397` | attenuation description |
| M9A thin within-source slope | `0.241` | attenuation description |
| M9A arc within-source slope | `0.337` | attenuation description |
| M9A thin residual-target Spearman | `0.916` | lower targets overpredicted |
| M9A arc residual-target Spearman | `0.863` | lower targets overpredicted |
| M9B confirmatory requirement | new independent set C | M8C cannot be confirmatory |
| M9B simultaneous intervals | 98.333% Bonferroni, 2,000 replicates, seed `20261023` | frozen protocol settings |

These values are read from M8C and M9A; they are not new analyses and are not
confirmatory evidence for H1, H2 or H3.

## Hypothesis disposition

| Hypothesis | Disposition | Reason |
| --- | --- | --- |
| H1 — attenuated response | **Best candidate if the project is reopened; not authorized now** | Largest error, highest explanatory discrimination, fewest external dependencies, but still requires new independent C and cannot identify cause. |
| H2 — edge/sharpness conflation | **Deprioritize** | Low confidence, high matching/confounding risk, and support would not identify a feature or justify feature engineering. |
| H3 — font/glyph difficulty | **Deprioritize** | Requires new licensed fonts and still yields only a conditional association under a fixed renderer; cannot support a causal or population claim. |

## Path disposition

| Path | Disposition | Reason |
| --- | --- | --- |
| A — stop and package | **Selected** | Highest integrity, clearest stopping point, preserves the audited evidence and gives the best portfolio narrative without diluting it. |
| B — one limited pilot | Not selected | Could de-risk H1 precision only if C were already justified; produces no adjudication and adds data/maintenance. |
| C — confirmatory research | Not selected | Potentially valuable for H1, but high effort, new independent data, unresolved sample size and, for H3, new font licensing; no human or security claim follows. |

## Mandatory closure work

The repository is not yet fully packaged merely because this memo exists. The
root `README.md` still presents its status through M8B and does not list M8C,
M8D, M8E, M9A or M9B in its milestone/status tables. The project overview,
methodology and experiment log likewise need a final status update or an
explicit freeze note. These are documentation and release tasks, not reasons to
run another experiment.

The mandatory closure requirements, optional future work and the exact
definition of done are in `project_completion_checklist.md`. In summary, closure
requires a frozen decision record, byte-identical historical artifacts, passing
integrity and replay checks, an updated README and research summary, honest
limitations, a clean repository, passing tests and lint, and a versioned release.

## Verification

The following read-only checks were run for this memo:

```bash
.venv/bin/python -m captcha_quality.benchmark_integrity verify
.venv/bin/python -m captcha_quality.benchmark_lifecycle verify \
  --config configs/benchmark_lifecycle_v0_3_0.toml --state post-execution
.venv/bin/python -m captcha_quality.matched_font_benchmark verify \
  --lifecycle-post configs/benchmark_lifecycle_v0_3_0.toml
.venv/bin/python -m captcha_quality.interference_experiment_protocol verify
```

All passed. The M8E historical-evidence verifier reports 17/17 pinned modules,
36/36 pinned input rows, the exact ground-truth SHA-256, and the recorded dirty
revision `75c8ef4-dirty` without relabelling it. The M8C full replay reports
16/16 checks; the post-execution lifecycle reports 11/11. The M9B validator
reports 26/26 checks and its recorded package table reports 38/38. The detailed
checks and final test/lint outcomes are in `verification_checks.csv`.

The local `ground_truth.csv` was not read. Its tracked state, ignore rule and
hash were checked only; no text from it appears in this package.

## Non-claims

This memo does not claim, and the project cannot establish from the current
evidence:

- human readability, CAPTCHA solvability, CAPTCHA security or production
  robustness;
- a causal mechanism for interference error;
- a causal font effect or a font-population effect;
- that a feature causes the error or that adding a feature would fix it;
- that synthetic-reference SSIM is a perceptual or human-legibility measure;
- that the frozen champion should be replaced or that a better model score is
  a research result;
- that H1, H2 or H3 has been tested. All three remain `not_tested`.
