# v1.0.0 Release Closure Checklist

**Scope:** whole repository, not only the documentation files.
**Decision:** M9C Path A — stop experimentation and package the research.
**Release:** `v1.0.0`
**Status:** **PROJECT COMPLETE — v1.0.0 ready** if every mandatory item below
is `pass`. Any `blocked` item prevents tagging.

This is a new closure checklist for the release. The historical M9C checklist in
`reports/research_prioritization_v0_1_0/` is preserved unchanged.

## Mandatory closure requirements

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| C-01 | Root README covers M0–M9C, including M8C, M8D, M8E, M9A, M9B and M9C | `pass` | `README.md` status, contributions, results, limitations and research-status sections; reviewed path and content audit |
| C-02 | Project overview is complete and accurate | `pass` | `docs/PROJECT_OVERVIEW.md` includes purpose, formulation, scope, boundaries, architecture, navigation and M0–M9C map |
| C-03 | Methodology documents the actual methods and separates baseline, robustness, cross-font, exploratory and preregistered work | `pass` | `docs/METHODOLOGY.md` §1–§5.31, especially §5.26–§5.31 |
| C-04 | Experiment log is complete and source-grounded | `pass` | `docs/EXPERIMENT_LOG.md` index and entries EXP-0022–EXP-0026; M8E maintenance note |
| C-05 | Research summary is complete and technically credible | `pass` | `docs/RESEARCH_SUMMARY.md` |
| C-06 | Reproducibility instructions are verified and distinguish safe checks from artifact generation | `pass` | `docs/REPRODUCIBILITY.md`; read-only verifier commands and expected outcomes |
| C-07 | Important architecture and report links work | `pass` | Focused link test; committed report packages under `reports/` |
| C-08 | All headline claims and metrics are audited against authoritative artifacts | `pass` | Metric reconciliation against M8C, M8D, M9A and M8C comparison tables; focused documentation tests |
| C-09 | Known limitations and untested hypotheses are explicit | `pass` | README limitations, research summary, methodology §8, release notes; H1/H2/H3 marked `not_tested` |
| C-10 | No historical benchmark artifact, integrity contract, pinned hash, lifecycle state or acceptance criterion changed | `pass` | `git diff` reviewed paths; M8E/M8C hashes and verifiers unchanged |
| C-11 | M8E, M8C lifecycle, M8C replay and M9B verification behave as expected | `pass` | M8E `verification: PASS`; lifecycle `11/11`; replay `16/16`; M9B `26/26` |
| C-12 | Full test suite passes | `pass` | `866 passed in 537.91s` (`pytest -q`) |
| C-13 | Ruff lint and format checks pass | `pass` | `ruff check .`: `All checks passed!`; `ruff format --check .`: `101 files already formatted` |
| C-14 | `git diff --check` passes | `pass` | No output, exit 0 |
| C-15 | Working tree is clean after the release commit | `pass` | Post-commit `git status --porcelain` empty; checked before tag |
| C-16 | A versioned release is prepared only if every mandatory gate passes | `pass` | `docs/RELEASE_NOTES_v1.0.0.md`; package version `1.0.0`; annotated tag `v1.0.0` created after checks |
| C-17 | Ground truth remains ignored and untracked, and its text was not read or exposed | `pass` | `.gitignore`; `git ls-files` shows no `ground_truth.csv`; M8E checks exact hash without text |
| C-18 | No dataset, image, prediction, model fit, calibration, feature change or experiment was generated or executed for the release | `pass` | Only documentation, release metadata, package-version metadata and documentation tests changed; no runner invoked |
| C-19 | The frozen M5C champion and artifact/card hashes are unchanged | `pass` | Artifact `c9b1625d...`; card `9a75725b...`; focused tests and M8C replay |
| C-20 | The M8C, M8D, M9A, M9B and M9C historical artifacts remain unchanged | `pass` | Tracked hashes unchanged; no report edits; focused historical-hash tests |
| C-21 | Documentation describes clone-visible versus local-only evidence honestly | `pass` | README, `PROJECT_OVERVIEW.md`, `REPRODUCIBILITY.md`, release notes |
| C-22 | The M9C decision and end of experimentation are visible | `pass` | README research status; `RESEARCH_SUMMARY.md`; release notes §8 |
| C-23 | H1/H2/H3 remain `not_tested` and M9B remains unexecuted | `pass` | README, research summary, M9B validator, release notes |
| C-24 | No unsupported human-readability, solvability, security or production claim is made | `pass` | Claims audit and focused tests; non-claims repeated in README and summary |
| C-25 | No GitHub Release was published or externally announced | `not_applicable` | Not authorized by the repository workflow and not part of this milestone |

## Verification record

| Check | Result |
| --- | --- |
| M8E integrity | PASS — 17/17 pinned modules; 36/36 input rows; exact ground-truth hash; historical `75c8ef4-dirty` preserved |
| M8C post-execution lifecycle | PASS — 11/11 checks |
| M8C replay | PASS — 16/16 checks; 6,300 images and predictions; 13 tables and report replayed |
| M9B protocol validator | PASS — 26/26 checks; all hypotheses `not_tested` |
| Focused documentation tests | PASS — `12 passed` (`tests/test_release_documentation.py`) |
| Full test suite | PASS — `866 passed in 537.91s` (`pytest -q`) |
| Ruff check | PASS — `All checks passed!` |
| Ruff format check | PASS — `101 files already formatted` |
| `git diff --check` | PASS — no output, exit 0 |
| Ground-truth text | Not read, not printed, not committed |
| Historical artifact hashes | Unchanged |
| Release tag | `v1.0.0` created only after the mandatory checks passed |

## Blockers

None.

## Optional future work

Optional work is **not** required for closure and is **not authorized** by M9C:

- H1 confirmatory study with new independent source set C.
- H2 feature-conflation diagnostic.
- H3 new-font characterization with licensed thin/thick font groups.
- Unseen-renderer, resolution or format study.
- Human readability or solvability study.
- CNN comparison or dashboard.
- New feature family or model development.

Each would require a separately approved milestone, a new protocol and, where
applicable, independent data and ethics/licensing review.

## Final statement

The project built and audited a reproducible synthetic CAPTCHA-style
image-quality benchmark. It found a broad interference-specific error in a frozen
conventional model, documented the limits of that evidence, and preregistered —
but did not execute — three follow-up hypotheses. It makes no claim about human
readability, CAPTCHA solvability, security or production robustness.

**Final status: PROJECT COMPLETE — v1.0.0 ready.**
