# M8D provenance and robustness audit of the M8C cross-font benchmark

**Audit version:** v0.1.0<br>
**Milestone:** M8D (read-only audit of M8C)<br>
**Audited commit:** `ce5963f3bbc53bfaa4675357b3dee023de5599ce`<br>
**Recorded execution revision:** `75c8ef4-dirty`<br>
**Verdict:** **verified with documented limitations; no erratum required**

This audit did not regenerate the benchmark, alter a prediction or target, train or tune a
model, change a frozen protocol, rewrite a historical artifact, or create a new benchmark row,
image or prediction. Every check is either a read of the committed artifacts, an existing
verifier, or a recomputation from the committed predictions. Shadow-tree tamper tests operate on
temporary copies only.

## 1. Question and scope

The audit asks four questions.

1. **Provenance.** Can the committed M8C predictions and metrics be tied to the code that
   actually ran, given the recorded revision `75c8ef4-dirty`?
2. **Ground-truth isolation.** Was `ground_truth.csv` kept out of the feature, prediction,
   calibration, fitting and selection paths? Do its hashes, row counts and joins match the
   recorded evidence, and was it ever committed or exposed?
3. **Metric correctness.** Do the committed metrics follow the frozen definitions, scopes,
   exclusions, aggregation, R² suppression rule and bootstrap settings when recomputed
   independently from `predictions.csv` without calling the model?
4. **Lifecycle integrity.** Are the M7C–M7E, M8A, M8B and M8C lifecycle contracts intact, does
   the M8C post-execution transition behave as declared, and do tampered shadow trees fail
   closed?

## 2. Preflight

| Check | Result |
| --- | --- |
| Branch | `main` |
| `HEAD` | `ce5963f3bbc53bfaa4675357b3dee023de5599ce` |
| `origin/main` | identical to `HEAD` |
| Working tree at audit start | clean (after the audit test file is added, only that untracked file is new) |
| M8C commit present | yes, 6,332 paths, including 6,300 images and 15 report files |
| `ground_truth.csv` in the commit | **no** |
| `ground_truth.csv` Git state | ignored by `data/processed/*`; not tracked; not in `HEAD` |
| M8C dataset tracked files | 5 tables/summary plus 6,300 images; `ground_truth.csv` is the untracked sixth dataset file |
| Historical M7C/M7D/M7E reports | local, Git-ignored artifacts; not in the commit |

The audit used the repository's installed package (`captcha-quality-analyzer` 0.1.0) with
Python 3.13.16, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1 and pytest 9.1.1.

## 3. Executed-code provenance

### 3.1 What `75c8ef4-dirty` means

This is a verified interpretation, not a relabelling of the historical record.

| Proposition | Status | Evidence |
| --- | --- | --- |
| The recorded short revision is `75c8ef4` | verified | `code.revision == "75c8ef4-dirty"` in `run_metadata.json` |
| Exactly one commit matches the abbreviation | verified | `git rev-parse --disambiguate=75c8ef4` returns only `75c8ef428703fc2a57e157cf90653d0389c5de3f` |
| That commit was the run-time `HEAD` | verified | it is the direct parent of the execution commit; the run timestamp (2026-10-22T04:14Z) precedes the commit (2026-10-22T05:10Z) |
| The working tree was dirty | verified | the module did not exist at `75c8ef4`; `provenance.code_revision` emits `-dirty` only for a non-empty `git status --porcelain` |
| The executed module is the committed module | verified | recorded module SHA-256 `3ec0e93f…` equals both the working file and the `ce5963f` blob |
| Every hash-pinned execution input is the committed file | verified | all 34 rows that carry an expected hash match; all 36 observed hashes match current files |
| The nine core pipeline modules are unchanged across the commit | verified | `features`, `render`, `corruptions`, `dataset`, `splits`, `modelling`, `heldout_font_benchmark`, `source_identity`, `benchmark_preflight` |

The commit delta identifies the non-output files that differ between `75c8ef4` and the execution
commit:

`configs/benchmark_lifecycle_v0_3_0.toml`, `pyproject.toml`,
`src/captcha_quality/benchmark_lifecycle.py`,
`src/captcha_quality/matched_font_benchmark.py`, and the test files.

Only the execution module and lifecycle v0.3.0 are independently proven to have existed at run
time: the module is absent at `75c8ef4`, and the run metadata records v0.3.0's SHA-256 while
v0.3.0 is absent at `75c8ef4`. The test and packaging edits in this delta may have been made
after the run. The exact dirty-file list was not captured by the run.

### 3.2 Code-hash ledger

`provenance.json` and `execution_stack.csv` record the complete execution-relevant stack. The
distinction that matters is pinning.

| Class | Modules | Consequence |
| --- | --- | --- |
| Hash-pinned in `run_metadata` | 12 code paths: features, render, corruptions, dataset, splits, modelling, inference, source identity, M7C/M7D/M7E verifiers, execution module | run-time bytes are fixed by recorded hashes |
| Imported but **not** pinned | `benchmark_lifecycle.py`, `benchmark_preflight.py`, `geometric.py`, `interference.py`, `quality_metrics.py`, `provenance.py` | run-time bytes are not independently recorded |
| Unpinned, changed by `ce5963f` | `benchmark_lifecycle.py` (state parsing only) | does not render, extract features, predict or compute metrics |
| Unpinned, unchanged across the commit | preflight, geometric, interference, quality metrics, provenance | tracked, last changed before `75c8ef4`, unchanged by `ce5963f` |

The six unpinned modules are the only residual provenance gap. There is no evidence that any of
them differed at run time, but the run did not record their hashes. Two mitigations limit the
gap:

* the current verifier re-derives every SSIM target from the committed images using the current
  `quality_metrics.py`, and replays every prediction from the committed features using the
  pinned inference path;
* the committed `geometric.py` and `interference.py` are unchanged since before the recorded
  revision, and the tree is clean, so a transient edit would have had to be reverted before the
  commit without leaving a trace.

A full regeneration would be the only way to prove the corruption operators **without relying on
the recorded reproducibility run**, but regeneration is outside this audit's mandate and was
therefore not performed.

### 3.3 Findings

* The dirty revision is **honestly recorded** and is explained by uncommitted M8C work. It must
  not be rewritten as a clean revision.
* The metadata is internally consistent with every committed output. It is not signed, so
  pre-commit byte history of `run_metadata.json` is unavailable.
* `input_manifest.csv` has **36 rows but 35 unique keys**: `text_case_manifest` is added twice.
  Both rows are identical and match. The report's “36 / 36 pinned inputs” is accurate as a row
  count but overstates the number of distinct pins: 35 unique keys, of which 34 carry a
  non-vacuous expected hash. The two self-pins (`execution_module`, `lifecycle_protocol`) have an
  empty expected hash and record only the observed value. This is cosmetic, not a validity issue.

## 4. Ground-truth isolation and integrity

`ground_truth.csv` is **not the regression target**. It contains `source_id`, `case_id`,
`font_id` and the literal source text. The target is `ssim_target` in `manifest.csv`, computed
between each authorised corrupted image and its source-specific clean reference.

| Check | Result |
| --- | --- |
| SHA-256 | `426fa41949223ebc481e88b0c42a22770df80d30d3ed8f951fa0f52b92b39dfd`; matches `run_metadata.dataset.ground_truth_sha256` |
| Rows | 300; 300 unique `source_id`; no duplicates |
| Columns | exactly `source_id, case_id, font_id, text` |
| Source join | exact against `sources.csv` |
| Case/font lineage | exact against `sources.csv` |
| Text length | matches the planned length for all 300 rows |
| Text hash | SHA-256 of each text matches `sources.text_sha256` for all 300 rows |
| Charset/normalisation | ASCII, NFC, project charset only |
| Committed | **no** |
| Ever in reachable Git history | **no**; `git rev-list --all --objects` contains no `ground_truth.csv` object |
| Exposed through a committed artifact | no; the literal texts are inputs in the committed text-case manifest, but the `ground_truth.csv` table itself is absent |
| Git-ignored | yes, by `data/processed/*` |

The benchmark joins are exact and one-to-one:

| Table | Rows / keys | Join result |
| --- | --- | --- |
| Frozen text cases | 30 cases, 10 fonts | every case-font pair present |
| Sources | 300 unique `source_id` | one row per source |
| Manifest | 6,300 rows | 5,700 primary + 600 diagnostic; 300 clean, 1,200 per corruption family |
| Images | 6,300 files in exactly six subdirectories | every manifest `image_path` exists |
| Features | 6,300 rows | one-to-one with the manifest by `row_id` |
| Predictions | 6,300 rows | one-to-one with the manifest by `row_id` and `sample_id` |
| Ground truth | 300 rows | joins exactly to `sources.csv` and the case/font lineage |

The data-flow evidence is both static and behavioural:

* `_merge_features` merges `manifest.csv` and `features.csv` only.
* `model_inputs` selects exactly the eight image features; `predict` receives no target, text or
  identifier column.
* `_build_prediction_frame` adds `ssim_target` only after prediction, for reporting.
* The execution path calls `predict` only. The `inference:no_fit` check records that no `fit`,
  `partial_fit`, calibration, thresholding, selection or refitting is performed.
* A fake estimator that records its input receives a `(6300, 8)` matrix identical to the eight
  feature columns, and its predictions are bit-identical after the ground-truth text is mutated
  in memory.
* Metrics use only `ssim_target` and `prediction`.

Failure behaviour in an isolated shadow copy:

| Tamper | Result |
| --- | --- |
| Delete `ground_truth.csv` | dataset loader raises `missing M8C ground_truth` |
| Alter one text | `source_tables` fails (`ground-truth text mismatch`) and `tables_replay` fails |
| Append a duplicate row | `source_tables` fails (301 rows, expected 300) |
| Replace a source identifier | `source_tables` fails (identifiers do not match the plan) |
| Change a case identifier | `source_tables` fails (lineage mismatch) |

**Documented gap.** `verify_outputs` re-derives ground-truth content from the frozen plan but does
not compare the file with `run_metadata.dataset.ground_truth_sha256`. A semantics-preserving row
reordering therefore **passes** verification while the recorded byte hash becomes stale. This is
a byte-integrity gap, not a benchmark-validity gap: no label or join changes. It is recorded here
and exercised by a regression test; the historical verifier was not changed because that would
change the pinned module hash.

## 5. Independent metric recomputation

The recomputation is implemented directly with NumPy/pandas from `predictions.csv` and
`manifest.csv`. It does not import `matched_font_benchmark.py`, does not call the champion, and
does not use any committed metric table as an input.

| Result | Value |
| --- | --- |
| Comparisons | **1,656** |
| Failures | **0** |
| Maximum absolute difference | **6.04 × 10⁻¹⁴** |
| Tolerance | 10⁻¹² |
| Report table cells checked | **159** |
| Report cell failures | **0** |
| Report tolerance | 5 × 10⁻⁷ (six-decimal rounding) |

Coverage: overall scopes (10), all ten fonts across primary/clean/diagnostic scopes (90
comparisons), the three font groups (27), families and the diagnostic/clean rows (21), all
severity rows (63), all 300 source metrics (900), and both bootstrap schemes with their estimates
and percentile bounds (525).

Headline values:

| Scope | n | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Overall primary | 5,700 | 0.059607206010939655 | 0.08861042886468502 | 0.7328983709293053 |
| Familiar primary | 1,710 | 0.06081219566692343 | 0.09289448224297654 | 0.6835434898374908 |
| M7B held-out primary | 1,710 | 0.06289088383152783 | 0.09162579879145862 | 0.7500913926567446 |
| New held-out primary | 2,280 | 0.056240705403510685 | 0.08283125245000134 | 0.7531142215536742 |
| Thin lines | 1,200 | 0.10207297853853917 | 0.13211369955002794 | −0.5751582086858849 |
| Arc strokes | 1,200 | 0.07957756011535612 | 0.10316171406852775 | −0.29891134048257606 |

Bootstrap (2,000 replicates, seed 20260821, 2.5/97.5 percentiles):

| Scheme | MAE 95% CI | R² 95% CI |
| --- | --- | --- |
| Source-grouped (300 groups) | [0.05830400135688277, 0.06098737030295556] | [0.7219096637796057, 0.7440907727939307] |
| Matched-case (30 groups) | [0.05812168236562061, 0.06118362183131131] | [0.7147209433117239, 0.7491932388898961] |

R² suppression was checked independently:

* clean reference: every target is exactly `1.0`, population SD `0.0`, R² suppressed;
* diagnostic elastic 0.5 px: population SD `0.002791318324466318`, below the frozen `0.005`
  threshold, R² suppressed.

The 300 clean rows have missing `psnr_target` because PSNR is infinite for an identical image;
`psnr_infinite` is `True` and PSNR is diagnostic only. This is not missing prediction or target
data.

## 6. Lifecycle and verifier results

| Verifier | Live repository | Historical pre-execution shadow |
| --- | --- | --- |
| M8C `verify_outputs` | **16/16 pass**, including 6,300 image re-hashes, target/feature replay, 15-report membership, 6,300 predictions | not applicable (executed state only) |
| M8C lifecycle v0.3.0 post-execution | **11/11 pass** | **fails closed** (5 expected missing-output checks) |
| M7C analysis | **23/23 pass** | not required for this audit |
| M7D plan | **24/26**, only the two expected planning-state checks fail | **26/26 pass** |
| M7E roster v0.2.0 | expected failure: planned benchmark path now exists | **passes** |
| M8A v0.1.0 lifecycle | **8/8 pass** | **8/8 pass** |
| M8A v0.2.0 lifecycle | **11/11 pass** | **11/11 pass** |
| M8B preflight with v0.2.0 | **11/11 pass** | **11/11 pass** |
| M8B preflight with v0.3.0 | expected `lifecycle_contract` failure (v0.3.0 is executed, not awaiting execution) | n/a |

The M7D live failure is exactly the expected transition: the planning-state check
`no_planned_outputs_created` no longer holds because the authorised outputs now exist. The
lifecycle v0.3.0 verifier classifies that transition and requires every other M7D artifact check
to pass.

`verification_checks.csv` contains 212 rows. Fifteen are expected failures (M7D transition, M7E
live pre-execution guard, M8B v0.3.0 by-design mismatch, M8C v0.3.0 fail-closed, and the six
detected tamper scenarios). **Zero rows are unexpected.**

## 7. Tamper detection in isolated shadow trees

| Scenario | Observed outcome |
| --- | --- |
| Unmodified shadow copy | all checks pass |
| One prediction changed by 0.001 | `tables_replay` and `run_metadata` fail |
| `ground_truth.csv` deleted | loader raises |
| One ground-truth text altered | `source_tables` and `tables_replay` fail |
| Duplicate ground-truth row appended | `source_tables` fails |
| Unexpected report file added | `report_membership` fails |
| One decoded image pixel changed | `image_content` and `tables_replay` fail |
| Non-pixel trailing PNG byte flipped | **not detected**; pixel content is unchanged |
| Ground-truth rows reversed | **not detected**; semantics are unchanged, recorded byte hash becomes stale |

The last two are deliberate boundary cases, not failures of the benchmark:

* image integrity is verified at the **decoded pixel** level, and all 6,300 images are tracked
  and covered by Git's byte-level protection; a non-pixel edit cannot change a target or feature.
* the ground-truth hash gap is documented in §4. The audit's regression test pins the live hash
  to the recorded value, so a future semantics-preserving rewrite is detectable outside the
  historical verifier.

The lifecycle suite (`tests/test_benchmark_lifecycle.py`) exercises the declared-input,
M7D historical-state, self-consistent-tamper, roster, unrelated-addition and font-binary
scenarios: **21 passed**. The new provenance/isolation suite:
**8 passed**.

## 8. Immutability

| Evidence | Result |
| --- | --- |
| `git status --porcelain` | clean before the audit additions |
| `git diff --check` | exit 0 |
| Authoritative hashes checked | **70 artifacts, 0 mismatches** |
| M8C dataset tables | 6/6 match `run_metadata.dataset` |
| M8C report files | 14 hashed files match `run_metadata.outputs.hashes`; `run_metadata.json` is not self-hashed |
| M8C images | 6,300 files tracked at `ce5963f`; `image_content` re-hashes every decoded pixel; image pixel-hash set `c39a9fb…` and reference set `fccb970…` independently recomputed from the manifest |
| M5C champion | joblib and card hashes match the lifecycle pins |
| Lifecycle/protocol/text-case configs | match the audited commit's Git blobs |
| M7C/M7D/M7E outputs | 40 files match their own recorded output hashes; key M7D/M7E files are also pinned by M8C lifecycle v0.3.0 |

The M8C dataset and report files are force-added and tracked. M7C/M7D/M7E reports are local,
Git-ignored artifacts: their byte identity is established against their recorded output hashes
and the committed lifecycle pins, not against Git blobs. A consistent rewrite of an entire
untracked report directory could not be excluded by Git alone; no such rewrite is evidenced, and
the hashes match the records.

## 9. Tests and lint

| Command | Result |
| --- | --- |
| `python -m pytest tests/test_benchmark_provenance_audit.py -q` | 8 passed |
| `python -m pytest tests/test_benchmark_lifecycle.py -q` | 21 passed |
| `python -m pytest -q` | **791 passed in 560.58s (0:09:20)** |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 87 files already formatted |

The new tests are read-only: they never call the champion, render an image or write to an
authoritative artifact.

## 10. Findings and dispositions

| ID | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| F1 | informational | Run recorded `75c8ef4-dirty`; exact dirty-file list and unpinned-module hashes were not recorded. | Keep the honest record. This audit adds a versioned provenance supplement and interpreter notes. Future runs should commit source first or record the full dirty-state manifest. |
| F2 | limitation | Six imported modules are not hash-pinned (`geometric`, `interference`, `quality_metrics`, `provenance`, `benchmark_preflight`, `benchmark_lifecycle`). | No evidence of divergence. Current verifier behaviourally re-exercises target and prediction code. A future execution should pin the full import closure. Do not change M8C's module hash. |
| F3 | low | `verify_outputs` does not re-check `ground_truth_sha256`; a semantics-preserving reorder passes with a stale recorded hash. | Documented; regression test pins the live hash. A future verifier version should check all six recorded dataset hashes. No historical rewrite. |
| F4 | informational | `input_manifest.csv` has 36 rows / 35 unique keys / 34 expected pins because `text_case_manifest` is duplicated. | Cosmetic. All rows match. Deduplicate in a future verifier version only. |
| F5 | informational | Image verification is pixel-level; a non-pixel PNG byte change is not detected. | Acceptable: targets and features depend on decoded pixels, and tracked bytes are protected by Git. |
| F6 | informational | The recorded reproducibility run used the same dirty working tree, so it proves determinism, not committed-code equivalence. | State the limitation; do not claim a clean-run reproduction. |

No finding alters a prediction, target, metric, join, protocol or historical artifact. **No
erratum is issued.** The correct classification is *verified with documented limitations*.

## 11. Reproduction

```
python -m captcha_quality.matched_font_benchmark verify \
  --lifecycle-post configs/benchmark_lifecycle_v0_3_0.toml
python -m pytest tests/test_benchmark_provenance_audit.py -q
python -m pytest tests/test_benchmark_lifecycle.py -q
ruff check .
ruff format --check .
git diff --check
```

The independent metric comparison is recorded row-for-row in `metric_recomputation.csv`
(1,656 rows). The full verifier transcript is in `verification_checks.csv` (212 rows). The
provenance ledger is in `provenance.json` and `execution_stack.csv`; immutable artifact hashes
are in `immutability.csv`; ground-truth metadata is in `ground_truth_integrity.json`. This audit
artifact is versioned separately and does not modify the M8C run metadata or any historical
artifact.
