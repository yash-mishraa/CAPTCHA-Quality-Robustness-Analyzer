# Project completion checklist

**Scope:** whole repository, not only the M9C memo package.
**Decision:** Path A — stop experimentation and package the research.
**Status:** M9C decision complete; the repository is not yet at final closure
because the documentation, research-summary and release items below are still
open.

This checklist separates **mandatory closure requirements** from **optional
future research**. Optional work does not block closure. A checked mandatory
item needs the evidence named beside it; a checked box is not a substitute for
that evidence.

## Definition of done

The repository is complete when every mandatory item below is checked, the
historical evidence is still byte-identical, the full test and lint suite passes,
and the final release records the same decision. Completion is **not** defined
by a better model score, a new feature, a new experiment or a larger benchmark.

## Mandatory closure requirements

### M-01 — Freeze the scientific decision and scope

- [x] The M9C memo records **Path A** and explains why H1/H2/H3 are not being
  executed.
- [x] The memo states that M8C and M9A are exploratory for the hypotheses and
  that M9B is a preregistration, not evidence.
- [x] The memo does not authorize execution, data generation, training,
  calibration, feature engineering, model selection or benchmark changes.
- [x] The three hypotheses remain `not_tested`; no result is reported as support
  or falsification.

**Evidence:** `README.md`, `research_prioritization.md`,
`hypothesis_comparison.csv`.

### M-02 — Preserve historical artifacts and provenance

- [x] M5C champion artifact and card match their frozen SHA-256 values.
- [x] M8C predictions, report, run metadata and dataset evidence match their
  recorded hashes or are covered by the M8E contract.
- [x] M9A artifacts and M9B artifacts remain byte-identical.
- [x] M8E, M8C lifecycle and M8C replay verifiers pass.
- [x] M9B static validator and recorded package checks pass.
- [x] No historical artifact, integrity contract or pinned hash was modified to
  make the memo pass.

**Evidence:** `verification_checks.csv`; M8D audit; M8E contract; M9B
`verification_checks.csv`.

### M-03 — Reproducible benchmark and documented limitations

- [x] The M8C dataset/report can be replayed with the repository's verify
  command.
- [x] The post-execution lifecycle is verified in its executed state.
- [x] The M9B design can be validated without executing an experiment.
- [x] The known M8C limitations are documented: fixed renderer, synthetic SSIM
  target, ten purposive fonts, seven held-out identities, no font-population
  claim, negative R-squared on interference, and diagnostic-only rows.
- [x] The M8D limitations are retained: honest dirty revision, historical
  verifier boundaries and no silent repair.

**Evidence:** M8C report, M8D report, M9A README, M9B protocol, M9C
`verification_checks.csv`.

### M-04 — Ground-truth and leakage policy

- [x] `ground_truth.csv` is Git-ignored and untracked.
- [x] Its integrity hash is checked without reading its text.
- [x] No literal ground-truth text appears in this package or in the memo.
- [x] The feature/prediction path remains feature-only in the recorded M8C
  evidence.
- [x] The M9B leakage controls remain the reference for any future work.

**Evidence:** `.gitignore`, M8E contract, M8D ground-truth isolation checks, M9B
`data_requirements.md`.

### M-05 — Update the repository documentation

- [ ] Root `README.md` status, planned-milestone table and current-status section
  are updated to include M8C, M8D, M8E, M9A, M9B and M9C. At the M9C evidence
  cut, the README still described the project through M8B and did not list
  M9A/M9B.
- [ ] `docs/PROJECT_OVERVIEW.md` and `docs/METHODOLOGY.md` are updated, or carry
  an explicit freeze note that points to the later M8C-M9C evidence.
- [ ] `docs/EXPERIMENT_LOG.md` records M8C, M9A and M9B (or a final index entry
  explains why they are reported in the versioned report packages instead).
- [ ] The README explains setup, architecture, workflow, verification commands,
  main results, limitations and non-claims without implying human or security
  validity.
- [ ] Stale links, milestone labels and "planned" statements are reconciled with
  the final state.

**Evidence required at closure:** a clean documentation diff, with no historical
artifact edits.

### M-06 — Research summary and portfolio packaging

- [ ] A concise research summary (one page or equivalent) states the question,
  data, method, key results, limitations and non-claims.
- [ ] The summary uses only supported numbers from the repository (for example
  M8C all-primary and interference metrics) and labels M9A/M9B appropriately.
- [ ] The summary does not claim human readability, CAPTCHA solvability,
  security, production robustness, a causal font effect or a feature fix.
- [ ] The portfolio framing explains that the contribution is a reproducible
  measurement and robustness study, not a new CAPTCHA-strength benchmark.
- [ ] The M9C decision and the "not executed" status of M9B are visible.

**Evidence required at closure:** the summary file and a final claims check
against `hypothesis_comparison.csv` and the M9C memo.

### M-07 — Final claims and limitations audit

- [ ] Every headline claim in the README, project overview, methodology,
  experiment log, research summary and report packages is traceable to a
  recorded artifact.
- [ ] Exploratory, descriptive, diagnostic and confirmatory labels are used
  correctly. M8C/M9A are not relabelled as confirmatory.
- [ ] Negative, mixed and inconclusive results are retained where they exist.
- [ ] No claim is made about H1/H2/H3 being supported.
- [ ] The limitations section names the fixed renderer, synthetic target,
  purposive font roster, source grouping, lack of human labels and lack of
  production evidence.
- [ ] The M8D limitations and the M8C dirty-revision record remain visible.

**Evidence required at closure:** a claims-to-evidence table or equivalent audit.

### M-08 — Code quality, tests and repository hygiene

- [x] Focused memo tests pass (`9 passed`).
- [x] The full test suite passes (`854 passed`).
- [x] Ruff check and Ruff format check pass.
- [x] `git diff --check` passes.
- [ ] The working tree is clean after the release commit.
- [x] No generated dataset, prediction table, model artifact, font binary or
  ground-truth text is tracked by this memo.
- [x] The versioned memo package is force-added deliberately, because
  `reports/*` is ignored by default.

**Evidence required at closure:** the final command transcript and the release
commit. M9C will record the checks it actually ran in
`verification_checks.csv`; this item remains open until the final release
commit is reviewed.

### M-09 — Versioned release and reproducible archive

- [ ] A release tag or equivalent immutable version points at the closure
  commit.
- [ ] The release notes state the decision, the frozen evidence cut, the
  verifier commands and the known limitations.
- [ ] The historical artifact hashes and the M9C package are included in the
  release record.
- [ ] A fresh checkout can follow the README to install, run the applicable
  checks and understand which artifacts are local versus committed.
- [ ] The release does not require an unreviewed experiment or an ignored
  local file to substantiate a claim.

**Evidence required at closure:** tag/release metadata and a final verification
run.

### M-10 — No open scientific execution

- [x] The project is not waiting on an unapproved H1/H2/H3 run.
- [x] No pilot is scheduled or implied by this memo.
- [x] No model, feature or benchmark change is pending.
- [x] The next action is documentation/release work, not experimentation.

**Evidence:** M9C decision and `research_prioritization.md`.

## Optional future research (not required for closure)

These are explicitly **not authorized** by M9C. Each would require a new
milestone, a new protocol and, where applicable, independent data.

| ID | Optional work | Preconditions | Why it is optional |
| --- | --- | --- | --- |
| O-01 | H1 confirmatory study | Approved protocol; pilot P; independent C; frozen sample size, analysis code and one-shot lock; all M9B gates. | Best candidate if the project reopens; not needed to close the current descriptive study. |
| O-02 | H2 feature-conflation diagnostic | Approved protocol; independent C; adequate feature/target overlap; frozen matching and balance rules. | Low expected information gain; cannot identify a feature or justify feature engineering. |
| O-03 | H3 new-font characterization | Approved font roster; licence audit; at least two new thin and two new thick fonts; new cases; matched case clusters. | Highest external dependency; conditional association only. |
| O-04 | Unseen-renderer, resolution or format study | New factor and protocol; independent data; explicit confounding analysis. | Changes the research question; deferred by M7D. |
| O-05 | Human readability or solvability study | Ethics/consent, human labels and a new research programme. | Current project has no human outcomes and cannot claim them. |
| O-06 | CNN comparison or dashboard | Separate justification and evaluation protocol. | M3/M5C evidence does not justify a CNN on this target; dashboard is presentation, not science. |
| O-07 | New feature family or model development | Independent development data; source-grouped splits; predeclared retention and evaluation criteria. | M6A-M6D already found no defensible gain; a score chase is not a reopening trigger. |

## Reopening rules

### What would trigger reopening

1. A verified integrity, provenance or correctness failure in a historical
   artifact or reported claim.
2. An approved, independently funded confirmatory programme with new cases,
   new seeds and a frozen protocol; H1 is the first candidate.
3. An external replication or peer review identifying a concrete, testable
   error in the current descriptive claims.
4. A genuinely new research question that cannot be answered by the existing
   evidence and comes with its own independent data and stopping rule.

### What would not trigger reopening

- A desire for a better model score, a higher R-squared or a new champion.
- A new feature, model family, calibration idea or hyperparameter idea.
- A new dashboard, visualization or portfolio formatting request.
- More fonts, cases or corruption levels without an independent confirmatory
  protocol.
- Rerunning M8C, M9A or M9B on the same observed data.
- A reviewer asking "what about a CNN/transformer/more features?" without a
  specific, independently testable question.
- A deadline, a desire to extend the CV narrative or the availability of
  compute.
- Any post hoc use of C to choose a threshold, feature, font or model.

## Final closure statement

Closure is a documentation and integrity achievement, not a modelling result.
When all mandatory items are checked, the repository can state:

> This project built and audited a reproducible synthetic CAPTCHA-style
> image-quality benchmark. It found a broad interference-specific error in a
> frozen conventional model, documented the limits of that evidence, and
> preregistered—but did not execute—three follow-up hypotheses. It makes no
> claim about human readability, CAPTCHA solvability, security or production
> robustness.

Until then, the M9C decision is complete but the repository is not yet fully
packaged.
