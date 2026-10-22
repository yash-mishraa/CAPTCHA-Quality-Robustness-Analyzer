# Release Notes — v1.0.0

**Release date:** 2026-10-22
**Package version:** `1.0.0`
**Tag:** annotated `v1.0.0` on the release commit
**Status:** **PROJECT COMPLETE — v1.0.0 ready** if every mandatory closure item
in [`RELEASE_CHECKLIST_v1.0.0.md`](RELEASE_CHECKLIST_v1.0.0.md) passes.

## 1. What this release is

This is the documentation and evidence package for a completed measurement and
robustness study of synthetic CAPTCHA-style image quality. It covers M0–M9C and
does not start a new research milestone.

The release closes the experiment phase under **M9C Path A — stop
experimentation and package the research**. No model was trained, refitted,
tuned, calibrated or replaced. No dataset, image, prediction or benchmark result
was generated for this release.

The package version is `1.0.0`. Historical datasets, models, protocols and
reports retain their recorded `v0.1.0`, `v0.2.0` or milestone labels. This release
does not re-version scientific evidence.

## 2. Completed research scope

- **M0–M3:** target validation, deterministic synthetic dataset, grouped splits,
  eight features and a leakage-safe conventional benchmark.
- **M4–M5:** occlusion/interference and geometric-distortion pilots and grouped
  robustness benchmarks; retention-aware selection and training-objective
  studies.
- **M6:** spatial-aware, spatial-resolution, local-structural and multi-scale
  structure-tensor feature studies. None justified replacing the champion.
- **M7:** dataset-adequacy audit, held-out-font benchmark, cross-font analysis,
  next-benchmark design and font-roster licensing audit.
- **M8A–M8C:** frozen execution protocol, output lifecycle, deterministic matched
  cases, generator preflight and the matched cross-font benchmark.
- **M8D–M8E:** independent provenance/robustness audit and integrity contract.
- **M9A–M9C:** read-only interference error analysis, preregistered follow-up
  protocol and the Path A closure decision.
- **M9D:** final documentation, claims audit, release packaging and tag
  preparation.

## 3. Key results

| Result | Evaluation context | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| M3 baseline | 108 rows / 12 sources; familiar fonts; clean + blur/noise | 0.0157 | 0.0226 | 0.9865 |
| M5C champion, original task | 108 rows / 12 sources | 0.0156 | 0.0220 | 0.9872 |
| M5C champion, geometric | 36 rows / 12 sources; 1–3 px warp | 0.0493 | 0.0590 | 0.1855 |
| M7B held-out fonts | 1,140 primary rows / 60 sources | 0.063427 | 0.092952 | 0.743537 |
| M8C all primary | 5,700 rows / 300 sources / 10 fonts | 0.059607 | 0.088610 | 0.732898 |
| M8C interference | 2,400 rows / 300 sources | 0.090825 | 0.118525 | -0.430242 |
| M8C thin lines | 1,200 rows / 300 sources | 0.102073 | 0.132114 | -0.575158 |
| M8C arc strokes | 1,200 rows / 300 sources | 0.079578 | 0.103162 | -0.298911 |

M9A adds an exploratory description of the interference failure: the top 5% of
interference source groups account for 10.6% of interference squared error, and
within-source thin/arc slopes are 0.241/0.337 with prediction/target SD ratios
0.320/0.397. These are descriptive, not causal or confirmatory.

Do not infer human readability, CAPTCHA solvability, security or production
robustness from any of these numbers. The target is synthetic-reference SSIM.

## 4. Frozen artifacts

| Artifact | SHA-256 |
| --- | --- |
| M5C champion `selected_policy.joblib` | `c9b1625de3bb503f22fa14f9b6468b028e609b6c7febe7c307dafd0e3e1156e0` |
| M5C model card `selected_policy_card.json` | `9a75725ba1ccf82ab42f009aa744a102cce86a6a9bb1d1aa86431362213507f8` |
| M8C predictions | `e6c1f1ba83bbb05c18fed8a9b0dd1c87014a6f6c3667475971b2834dfedd4dd4` |
| M8C report | `ef266df7d0e3022b7eb0b6ccf08bb97782a8d2741e4f980f51facf94be7ad85f` |
| M8C ground truth (not tracked, not read) | `426fa41949223ebc481e88b0c42a22770df80d30d3ed8f951fa0f52b92b39dfd` |

The M5C artifact and card are local, Git-ignored files. The M8C predictions and
report are committed. The ground-truth table is ignored and untracked; only its
hash and structure were checked.

## 5. Verification status

| Check | Result |
| --- | --- |
| M8E integrity | PASS; 17/17 modules; 36/36 input rows; exact ground-truth hash |
| M8C lifecycle (post-execution) | PASS; 11/11 |
| M8C replay | PASS; 16/16; 6,300 images and predictions; thirteen tables and report replayed |
| M9B protocol validator | PASS; 26/26; all hypotheses `not_tested` |
| Focused documentation tests | PASS; 12 passed (`tests/test_release_documentation.py`) |
| Full test suite | PASS; 866 passed in 537.91s |
| Ruff lint | PASS; `All checks passed!` |
| Ruff format check | PASS; 101 files already formatted |
| `git diff --check` | PASS |
| Historical artifact hashes | Unchanged |
| Ground truth | Ignored/untracked; text not read or exposed |
| Working tree after release commit | Clean |

The release does not weaken or skip a check to make the tag pass. If a mandatory
item had remained blocked, no tag would have been created.

## 6. Known limitations and untested hypotheses

- Synthetic-reference SSIM is not human readability, solvability or security.
- One fixed renderer, one platform and ten purposively selected fonts.
- M8C has seven held-out font identities; 300 sources are not 300 fonts.
- Font identity is entangled with glyph shape, rasterisation and target
  difficulty.
- M3/M5C have only 12 test source groups; uncertainty intervals are descriptive.
- Clean and 0.5 px elastic rows are diagnostic only.
- The M8C run recorded `75c8ef4-dirty`; M8D documents the provenance boundary.
- Six imported modules were not pinned in the historical run.
- No human labels, solver data, live CAPTCHA experiments or production evidence.
- M9B was not executed. H1, H2 and H3 remain `not_tested`. H1 is the first
  candidate only if the project is reopened with new independent data and an
  approved protocol; H2 and H3 are deprioritized.

## 7. Reproducibility boundary

The committed M8C predictions, tables and report can be replayed in the prepared
evidence workspace using the M8E, lifecycle and M8C verifiers described in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

A fresh clone does not contain the Git-ignored datasets, model artifact, font
binaries or earlier M3–M7 reports. The complete benchmark is therefore not
claimed to be regenerable from a fresh clone alone. Full regeneration would be a
new execution under the clean-tree guard, not a reconstruction of the historical
one-shot run.

## 8. Explicit end of experimentation

No further experiment, pilot, dataset generation, model fit, calibration,
feature change, benchmark change or champion replacement is authorized by this
release. M9C selected Path A. Any future work requires a new milestone, a new
protocol and, where applicable, independent data and ethics/licensing review.

## 9. Release record

- Release tag: `v1.0.0` (annotated).
- No GitHub Release was published.
- The tag is created only after the mandatory closure checklist is fully
  satisfied.
- The closure evidence is in
  [`RELEASE_CHECKLIST_v1.0.0.md`](RELEASE_CHECKLIST_v1.0.0.md).
