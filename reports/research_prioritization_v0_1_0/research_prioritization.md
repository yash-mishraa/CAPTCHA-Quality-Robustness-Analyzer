# M9C — Research prioritization and project closure decision

- **Milestone:** M9C
- **Version:** v0.1.0
- **Status:** decision memo; no experiment executed
- **Decision:** Path A — stop experimentation and package the research
- **Evidence cut:** `81710e6c839f4e7f66752d4045c765df573acbe2`
- **M9A analysis revision:** `37bc86dba3cac8d7cfe2d9e84d83da300974866f`
- **M9B design revision:** `e3d8ed36b497347b87606536000114746d27d709`

## 1. Decision and authority

This memo selects **Path A: stop experimentation, freeze the scientific work at
M9B, document the limitations and package a reproducible research release.**

It does not authorize execution of H1, H2 or H3. It does not authorize a pilot,
new data, a new prediction, model training or refitting, calibration, feature
engineering, model selection, a benchmark change or a champion replacement. It
also does not authorize a new claim about human readability, CAPTCHA
solvability, security or production robustness.

The decision is deliberately conservative about scientific authority. M9B is a
preregistration, not a result. M8C is observed data that motivated the
hypotheses. M9A is exploratory. No confirmatory test exists for any hypothesis,
and the confirmatory set C does not exist.

## 2. Git state and evidence inspected

At the evidence cut:

- branch `main` was level with `origin/main`;
- the working tree was clean;
- `HEAD` was `81710e6c839f4e7f66752d4045c765df573acbe2`;
- the M5C champion artifact and card hashes matched their frozen values;
- the M8C predictions, report, M9A report and M9B protocol artifacts matched
  their recorded or pinned hashes.

Evidence reviewed directly from the repository:

- `reports/interference_error_analysis_v0_1_0/README.md`,
  `hypotheses_and_followups.md`, `analysis_protocol.json` and
  `verification_checks.csv` (M9A);
- `reports/interference_experiment_protocol_v0_1_0/README.md`,
  `experiment_protocol.md`, `hypothesis_matrix.csv`,
  `statistical_analysis_plan.json`, `data_requirements.md`,
  `decision_gates.md` and `verification_checks.csv` (M9B);
- `reports/matched_font_benchmark_v0_1_0/report.md`,
  `predictions.csv` and `run_metadata.json` (M8C);
- `reports/matched_font_benchmark_provenance_audit_v0_1_0/report.md` and
  `provenance.json` (M8D);
- `configs/benchmark_integrity_v0_1_0.toml` and
  `configs/benchmark_lifecycle_v0_3_0.toml` (M8E and lifecycle);
- `models/retention_aware_selection_v0_1_0/selected_policy_card.json` (M5C
  champion metadata);
- `README.md`, `docs/PROJECT_OVERVIEW.md`, `docs/METHODOLOGY.md`,
  `docs/EXPERIMENT_LOG.md`, `docs/BENCHMARK_INTEGRITY.md`, and the relevant
  focused tests.

The read-only verifiers were run before this memo was written:

| Check | Outcome |
| --- | --- |
| M8E historical-evidence integrity | PASS; 17/17 pinned modules, 36/36 input rows, exact ground-truth hash |
| M8C post-execution lifecycle | PASS; 11/11 |
| M8C full replay with post-execution lifecycle | PASS; 16/16 |
| M9B static protocol validator | PASS; 26/26 |
| M9B recorded package checks | PASS; 38/38 |
| M5C artifact and card hashes | match the frozen values |
| Ground truth policy | ignored and untracked; hash checked, text not read |

## 3. What the evidence does and does not establish

### 3.1 M8C: a real, broad interference failure

M8C is a 300-source, 10-font, 30-case cross-font benchmark with 6,300 rows
(5,700 primary and 600 diagnostic). The frozen M5C champion was evaluated
zero-shot; no fitting, calibration or selection was performed.

The primary results show a clear interference problem:

| Scope | MAE | RMSE | R-squared | Bias |
| --- | ---: | ---: | ---: | ---: |
| all primary | 0.0596 | 0.0886 | 0.7329 | -0.031 |
| interference | 0.0908 | 0.1185 | -0.4302 | -0.087 |
| thin lines | 0.1021 | 0.1321 | -0.5752 | -0.097 |
| arc strokes | 0.0796 | 0.1032 | -0.2989 | -0.077 |
| gaussian blur | 0.0265 | 0.0499 | 0.9431 | 0.007 |
| gaussian noise | 0.0390 | 0.0635 | 0.7357 | 0.026 |

The error is broad rather than driven by a few sources: the top 5% of
interference sources account for `10.6%` of interference squared error. At the
predeclared exploratory `|residual| >= 0.20` threshold, thin s4 has a `55.3%`
large-error share and arc s4 has `28.0%`.

This is enough to describe the failure, but it is **not** confirmatory evidence
for H1, H2 or H3: M8C's outcomes were observed before the hypotheses were
formalized, and M9A used those same outcomes to generate the hypotheses.

### 3.2 M9A: a strong descriptive attenuation signal

M9A reports that the frozen model tracks interference ordering but compresses
the response range:

- within-source prediction/target SD ratio: `0.320` for thin lines and `0.397`
  for arcs, versus `0.968` for blur and `0.975` for noise;
- within-source slopes: `0.241` for thin lines and `0.337` for arcs;
- residual-target Spearman correlations: `0.916` for thin lines and `0.863`
  for arcs;
- within-source rank correlations remain `0.792` (thin) and `0.885` (arcs), so
  the model is not condition-blind.

This is the strongest descriptive case among the three hypotheses. It supports
the *possibility* of attenuation, but M9A explicitly does not identify whether
the cause is training-distribution shift, model capacity, target construction
or feature limitations. Its confidence label is high for description and only
moderate for the out-of-distribution explanation.

### 3.3 Font differences are descriptive and confounded

M9A's per-font interference MAE ranges from `0.0358` for Impact to `0.1302` for
Courier New. The paired-by-case contrast is `-0.0611` for Impact and `0.0438`
for Courier New. But the target itself is less damaged for Impact (interference
target mean `0.920`, s1-to-s4 drop `0.128`) than for Courier New (mean `0.819`,
drop `0.276`). Font identity is entangled with glyph shape, stroke width,
renderer, case composition and target degradation. M9A calls this a modifier,
not a cause.

### 3.4 Feature associations are exploratory and partially redundant

M9A's strongest feature associations with interference absolute error are
`laplacian_variance` (Spearman `+0.792`) and `edge_density` (`+0.796` for thin
lines), with negative associations for histogram entropy and contrast. These
are descriptive rank associations, not source-clustered inference or a feature
selection result.

Existing project evidence also limits the value of a feature-conflation
programme:

- M7A found maximum training feature correlation `0.9911`, with
  `intensity_std`/`contrast_rms` and `laplacian_variance`/`noise_sigma_immerkaer`
  strongly redundant;
- M6A-M6D found no defensible gain from added spatial, local-structure or
  structure-tensor features under retention constraints;
- M8D's audit found the M8C evidence verified with documented limitations, and
  M8E now pins the complete execution closure and historical ground-truth hash.

None of this proves H2 false. It does mean H2 has low expected information gain
relative to the effort and a high chance of an uninformative matching failure.

### 3.5 M9B is a design, not evidence

M9B fixes:

- a new, independently generated confirmatory source set C;
- a disjoint pilot P for precision and matching feasibility;
- source-grouped H1/H2 resampling and case-cluster H3 resampling;
- Bonferroni simultaneous 98.333% percentile intervals;
- seed `20261023` and 2,000 replicates;
- one-shot analysis, no optional stopping, no threshold changes after C;
- new fonts for H3 and a new case manifest disjoint from M8C and P.

It explicitly states that M8C cannot be confirmatory, that a pilot cannot
adjudicate a hypothesis, and that M9C is not automatic. No sample size, font
roster or analysis code for C has been fixed. The protocol is therefore not
execution-ready.

## 4. Hypothesis assessment

### 4.1 H1 — attenuated response

**Scientific importance: high.** H1 targets the largest recorded error and the
most coherent explanation: the model compresses severe interference toward
near-clean quality. The descriptive attenuation is large, broad and consistent
across thin lines and arcs.

**Falsifiability: high.** M9B predeclares a paired within-source slope deficit,
a simultaneous interval, a mean interference-slope guard and family-specific
slope guards. A clean falsification or support decision is possible in principle.

**Explanatory discrimination: high, but bounded.** Pairing slopes within source
removes source-level mean bias and asks whether compression is specific to
interference relative to blur/noise. It can distinguish interference-specific
attenuation from a general source-level offset. It cannot separate training
distribution shift from model capacity, target construction or feature
limitations.

**Data and compute: moderate.** It needs one new evaluation-only source set and
a pilot; the champion is loaded once and predicts once. There is no training,
refitting or calibration. The renderer and corruption ladder already exist. The
main cost is disciplined data generation, provenance and one-shot analysis, not
model compute.

**New cases or fonts: new cases and seeds, no new fonts required.** H1 can use
new text cases and new render/corruption seeds under the fixed renderer.

**Leakage and confounding risk: moderate.** The main risks are source
composition, unequal target gaps across severity rungs, nonlinear calibration
being summarized by a linear slope, and out-of-distribution training shift.
M9B's source grouping and family guards mitigate but do not remove them.

**Expected information gain per unit effort: highest of the three.** A support
result would justify a separate intervention-design decision; a falsification
would retire the attenuation explanation; an inconclusive result would expose a
precision or design limit rather than merely adding another descriptive table.

**Relevance to the project: high.** H1 addresses the error pattern that
motivated M9B and is central to the project's stated robustness question.

**Value of a negative or inconclusive result: high.** A clear falsification is
scientifically useful, and a well-documented precision failure is preferable to
an underpowered claim.

**Key unresolved assumptions.** The within-source linear slope is an adequate
summary; C would be independent and large enough; at least 90% of sources would
remain eligible; the H1 precision target (`0.05` simultaneous half-width) is
attainable; the M8C-derived effect would generalize to new cases and seeds.

**Implementation complexity: moderate.** A new C/P data path, a slope-deficit
estimator, a source-grouped bootstrap, a one-shot analysis lock and immutability
checks are required. None is authorized by this memo.

**Recommendation.** H1 is the only hypothesis worth considering if the project
is reopened, but it is **not authorized now**. It should be the first candidate
only after a separately approved protocol, independent C and the M9B precision
and leakage gates are satisfied.

### 4.2 H2 — edge/sharpness feature conflation

**Scientific importance: medium.** The question of whether the frozen eight
features conflate interference with benign edge and sharpness variation is
relevant to interpreting the model, but it is a diagnostic question rather than
the project's primary measurement question.

**Falsifiability: high in principle, fragile in practice.** M9B predeclares a
matched absolute-residual excess and explicit balance and coverage guards.
However, a failed match yields `not_testable`, not a clean null result.

**Explanatory discrimination: low to moderate.** A matched residual excess could
be consistent with feature inadequacy, but it cannot identify which feature is
responsible, prove that the random forest uses it as a benign cue, or show that
an omitted feature would repair the error. It also cannot separate feature
inadequacy from training-distribution shift.

**Data and compute: moderate to high.** C is required, plus feature
standardisation fixed from the pilot, target-SSIM binning, a nearest-neighbour
caliper and balance diagnostics. There is no training, but matching and
sensitivity analysis add substantial design and analysis work.

**New cases or fonts: new cases and seeds; no new fonts required.**

**Leakage and confounding risk: high.** The eight features are correlated with
target SSIM, source content and severity; matching can induce selection and
regression-to-the-mean effects; benign blur/noise and interference differ in
mechanism, not only feature values; the feature vector may omit the relevant
texture information entirely.

**Expected information gain per unit effort: low to moderate.** Support would
not authorize a feature change and could be explained by several mechanisms. A
matching failure would be expensive and uninformative. Existing M6A-M6D results
already argue against a feature-intervention programme.

**Relevance to the project: medium.** Feature representation is important, but
the project has already tested several feature families without a defensible
gain, and M7A documented severe redundancy.

**Value of a negative or inconclusive result: moderate.** A clean falsification
would weaken the conflation explanation; a `not_testable` result would mostly
describe the matching design.

**Key unresolved assumptions.** Adequate feature/target overlap exists within
source; the caliper and standardisation can be fixed without C outcomes;
matching identifies a meaningful excess rather than a selection artefact; the
eight features are the right representation for the question.

**Implementation complexity: high.** Matching, balance diagnostics,
deterministic tie-breaking, coverage accounting and a source-grouped bootstrap
must all be frozen before C.

**Recommendation.** **Deprioritize.** Do not fund H2 before H1, and do not treat
it as a route to feature engineering. If a specific feature question later
becomes important, it needs a new protocol and independent development data.

### 4.3 H3 — font/glyph-specific difficulty

**Scientific importance: medium.** Font and glyph differences are a real part
of the observed error and a natural cross-font question, but the project can
already describe them.

**Falsifiability: high on paper, conditional in practice.** M9B predeclares a
matched thin-minus-thick case-cluster contrast and per-font sign checks, with
target-overlap and coverage guards. Font availability, licensing and overlap can
make the result `not_testable`.

**Explanatory discrimination: low to moderate.** A supported result is a
conditional association under the fixed renderer and the predeclared roster. It
cannot separate font identity from glyph shape, stroke width, font metrics,
hinting, rasterisation, target construction or case composition. It cannot
support a causal font effect or a font-population claim.

**Data and compute: high.** It requires C, a pilot, at least two new thin-stroke
and two new thick-stroke fonts, new cases, licence evidence, pinned font
hashes, a predeclared external stroke-width classification, target-SSIM
matching and case-cluster resampling. No model training is required, but the
external dependencies are substantial.

**New cases or fonts: yes — new fonts are the defining requirement.** The fonts
must be licence-eligible and unused by M5C training, M8C and pilot P.

**Leakage and confounding risk: high.** Font identity is fully entangled with
glyph shape, metrics, hinting, rasterisation and the font-specific clean
reference. The renderer is fixed, and a small purposive roster is not a font
population.

**Expected information gain per unit effort: low to moderate.** A supported
result would be a narrow conditional contrast. The project already has
descriptive font differences from M7B and M8C; H3 would not turn them into a
mechanism or a population claim.

**Relevance to the project: medium.** Cross-font behaviour is part of the
project's research narrative, but the existing M7B/M8C evidence already covers
the descriptive question within its stated limits.

**Value of a negative or inconclusive result: moderate.** A falsification would
constrain the font story; a font-licensing or overlap failure would be costly
and would not answer the scientific question.

**Key unresolved assumptions.** A licence-eligible roster can be assembled;
stroke-width classification is stable and external to outcomes; target overlap
is adequate within case, family, severity and bin; the fixed renderer supports a
useful conditional contrast; the roster is not treated as a population.

**Implementation complexity: very high.** New fonts, licence audit, new cases,
matching, case-cluster bootstrap and a separate characterization protocol are
all required.

**Recommendation.** **Deprioritize.** Consider it only after a separate
font-licensing and roster decision, and never as a justification for a model or
feature change.

## 5. Path comparison

| Criterion | Path A — stop and package | Path B — one limited pilot | Path C — confirmatory research |
| --- | --- | --- | --- |
| Scientific value | Preserves a strong descriptive and audited result; adds no new causal evidence, but prevents overclaiming. | Produces planning information only; cannot adjudicate H1, H2 or H3. | Could provide a confirmatory H1 result on independent data; H2/H3 remain conditional and resource-dependent. |
| Effort | Moderate documentation, release and verification work; no new data. | Moderate: new pilot data, protocol, generation and analysis; no confirmatory outcome. | High: pilot, C, sample-size freeze, font roster for H3, one-shot analysis, provenance. |
| Risk | Low; historical artifacts remain frozen. | Low-to-moderate; new data and analysis increase surface area without a scientific decision. | Moderate-to-high; leakage, matching failure, font availability and one-shot execution risk. |
| Reproducibility | High; existing verifiers and replay paths are already in place. | High only if a new pilot protocol is frozen and audited. | High in principle, but a larger and more fragile chain of custody. |
| Portfolio value | High if README, limitations and research summary are completed; the story is coherent and honest. | Moderate technical appendix; not a headline result. | Potentially high scientific value, but slower and conditional; does not add human or security claims. |
| Clarity of stopping point | Clear and immediate. | Defers the decision; creates another milestone. | Distant and conditional; may still end inconclusive or `not_testable`. |

### 5.1 Path A — stop experimentation

Path A freezes the scientific work at M9B, documents that H1-H3 are untested,
preserves the M8C/M9A evidence and the M9B design, and prepares a reproducible
release. It gives the project a defensible answer to "what did you establish?"
without pretending that a descriptive synthetic-SSIM result is a causal or
human-facing finding.

The work is real: update the root README and project documents, write a concise
research summary, reconcile status and milestone tables, verify every frozen
hash, run the repository checks and tag a release. None of that requires a new
experiment.

### 5.2 Path B — one limited pilot

The only credible pilot is an **H1 precision and feasibility pilot**: generate a
new source set disjoint from M8C, evaluate the frozen champion zero-shot, and
estimate source-level slope variance, complete-source retention and the
attainable simultaneous half-width. It may also record whether the H1 precision
target (`0.05`) is reachable. H2 matching yield and H3 target overlap could be
recorded as secondary feasibility, but they must not be used to adjudicate any
hypothesis.

Prerequisites: explicit approval of a pilot protocol, new cases and seeds,
disjointness from M8C and any future C, a frozen precision target, an analysis
lock and a documented stop/go rule. Stop if the precision target is not feasible
within a defensible data size, if complete-source retention falls below `90%`,
or if overlap for the chosen pilot question is inadequate.

Why not select it: a pilot adds data and maintenance and produces no supported
or falsified hypothesis. It is only worth doing if a confirmatory C programme
has already been approved. The project should not run a pilot merely to defer
the closure decision.

### 5.3 Path C — confirmatory research

Path C requires all of the following before execution:

- an approved, versioned protocol that supersedes or extends M9B without
  changing its historical artifacts;
- a pilot P with new cases, render seeds and corruption seeds, disjoint from
  M8C and C;
- a precision simulation that freezes the sample size before C exists;
- an independent C with 21 rows per source (19 primary, 2 diagnostic), new
  cases and seeds and, for H3, at least two new thin and two new thick
  licence-eligible fonts unused by M5C, M8C and P;
- a frozen case manifest, font roster, matching rules, calipers, bins, analysis
  code, seed `20261023` and 2,000 replicates;
- a one-shot analysis lock, historical hash verification, clean-tree guard and
  no threshold or endpoint changes after C is inspected.

Path C would add confirmatory evidence for H1 and, if feasible, H2/H3. It would
not establish human readability, solvability, security or production
robustness, and it would not by itself justify a model or feature change. Its
marginal value over Path A is one or more conditional synthetic-SSIM
confirmations; its marginal cost is a new data programme and, for H3, a new
font-licensing effort.

## 6. Recommendation

**Select Path A. Stop experimentation and package the research.**

The decision is based on scientific information gain, feasibility and
defensibility, not on the possibility of a better model score. M8C and M9A
already provide a coherent descriptive account of a real interference failure.
M9B is an honest preregistration of what a confirmatory test would require.
That is a respectable research endpoint. Continuing to C would be justified
only by a specific, independently funded question and a feasible independent
data programme, not by a general desire to keep the project moving.

If the project is reopened, **H1 is the first and only hypothesis to consider**.
H2 and H3 should remain deprioritized unless a new, independently justified
question makes their additional data and confounders worthwhile.

No part of this recommendation authorizes execution. The next action is
documentation and release, not a new experiment.

## 7. Conditions for reopening

### 7.1 Triggers

The project may be reopened only if at least one of the following holds:

1. **A verified integrity or correctness problem** is found in a historical
   artifact, a pinned hash, a replay, a target computation or a reported claim.
   The response is a correction and, if necessary, a new protocol, not a silent
   repair.
2. **A genuinely independent confirmatory data programme is approved and
   funded**, with new cases, new render/corruption seeds, a frozen sample size
   and a protocol that satisfies the M9B gates. H1 is the first candidate.
3. **An external replication or peer review identifies a concrete, testable
   error** in the current descriptive claims, with a specific question that new
   independent data could answer.
4. **A new research question is independently justified** and cannot be
   answered by the existing evidence; it must come with its own protocol,
   independent data and stopping rule.

### 7.2 Non-triggers

The following are **not** sufficient reasons to reopen the scientific work:

- a desire for a better model score, a new champion or a higher R-squared;
- a new feature, model family, calibration idea or hyperparameter idea;
- a new dashboard, visualization or portfolio formatting request;
- adding more fonts, cases or corruption levels without an independent
  confirmatory protocol;
- rerunning M8C, M9A or M9B on the same observed data and calling it
  confirmatory;
- a reviewer asking "what about a CNN?", "what about a transformer?" or "what
  about more features?" without a specific, independently testable question;
- a deadline, a desire to extend the CV narrative, or the availability of
  compute;
- any result from C being used post hoc to choose a threshold, feature, font or
  model.

A minor documentation or formatting correction does not reopen the scientific
work; it is maintenance.

## 8. Project completion

The mandatory definition of done is in `project_completion_checklist.md`. It
separates:

- **mandatory closure requirements** — decision record, frozen evidence,
  reproducible verification, complete documentation, honest limitations, clean
  repository, passing tests and release;
- **optional future research** — H1 confirmation, H2 diagnostics, H3 font
  characterization, renderer/resolution extensions, human studies and
  CNN/dashboard work, all separately authorized and out of the current scope.

The repository is not complete merely because this memo is committed. The root
README, project overview, methodology and experiment log still need a final
status update or a freeze note covering M8C-M9C. That is closure work, not a
reason to run another experiment.

## 9. Limitations of this memo

- It is a decision document, not an experiment, systematic review or
  meta-analysis.
- It relies on recorded M8C/M9A values and M9B design text. It does not
  recompute new effect estimates, fit models or generate data.
- It does not estimate costs, timelines, sample sizes or expected improvements
  beyond the qualitative ranges supported by the repository.
- It cannot establish that H1 is the true mechanism; it only ranks H1 as the
  most informative candidate if independent research is reopened.
- It cannot support human readability, CAPTCHA solvability, security or
  production robustness claims.
- The local `ground_truth.csv` was not read; only its ignore rule, tracked state
  and SHA-256 were checked.
