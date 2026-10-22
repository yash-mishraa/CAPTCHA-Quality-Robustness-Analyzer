# Benchmark provenance and integrity contract (M8E)

The M8E maintenance contract is an addition around the completed M8C
matched-font benchmark. It does not replace the frozen M8C protocol, change an
historical artifact, regenerate a dataset, or run a model.

| Item | Value |
| --- | --- |
| Contract | `m8e-integrity-v1` |
| Configuration schema | `m8e-integrity-config-v1` |
| Configuration | `configs/benchmark_integrity_v0_1_0.toml` |
| Implementation | `src/captcha_quality/benchmark_integrity.py` |
| CLI | `captcha-benchmark-integrity` |

## Historical-evidence mode

`verify_historical_evidence()` re-checks the recorded M8C evidence:

* the exact `ground_truth.csv` SHA-256, regular-file status, row count, exact
  schema and unique source identifiers;
* semantic joins and lineage against the frozen execution protocol and the
  committed text-case manifest;
* the exact historical input-manifest bytes, row count and key multiplicity;
* the exact execution-module bytes and the complete in-repository import
  closure;
* the pinned external package versions and the Python/platform constraints;
* the historical protocol, lifecycle and run-metadata bytes.

The expected ground-truth digest is a literal committed configuration value
copied from the recorded M8C evidence. It is never recomputed from the current
file. A pure row reordering changes the file digest and therefore fails even
when parsed rows remain semantically equivalent. Error details contain paths,
counts and digests, not ground-truth text strings.

Historical mode deliberately does **not** claim that the original execution
used a clean tree. `run_metadata.json` records `75c8ef4-dirty`; the verifier
checks that honest revision rather than rewriting it.

## Future-execution mode

`run_guarded_execution()` and `verify_future_preflight()` require:

* a full committed `HEAD` equal to the expected commit;
* an empty `git status --porcelain --untracked-files=normal` result;
* absent future output paths;
* the complete pinned dependency closure and matching runtime packages.

The dirty-tree check runs before the caller action and before any output path is
created. A successful guard returns an `ExecutionTicket` containing the exact
revision, branch, configuration digest and output paths so a future runner can
record them in its own metadata.

Future input manifests must have unique keys. `verify_future_input_manifest()`
has no legacy duplicate allowance.

## Dependency closure

The committed configuration pins all 17 in-repository modules reachable from
`src/captcha_quality/matched_font_benchmark.py`, not only the six modules M8D
found unpinned. Membership is exact by declared file path, not by a directory
fingerprint. An unrelated file added to `src/captcha_quality` or a shared
directory therefore does not invalidate verification, while a missing,
modified or newly imported declared-path module does.

The verifier also compares external import roots with the declared package
modules and checks the recorded versions of fonttools, joblib, numpy,
opencv-python-headless, pandas, pillow, scikit-image, scikit-learn and scipy.

## The historical duplicate input-manifest key

The M8C `input_manifest.csv` has 36 rows but 35 unique keys. The duplicate is
`text_case_manifest`. `_input_manifest()` iterates the lifecycle inputs (which
already include `text_case_manifest`) and then unconditionally adds
`text_case_manifest` again. The two rows name the same file, record the same
observed and expected hash, agree on the match flag and kind, and differ only
in descriptive role prose. The duplicate is therefore a cosmetic construction
defect, not an ambiguous resource identity, and it did not change the M8C
outputs or their verification.

The historical manifest and report are preserved byte-for-byte. The M8E
configuration explicitly declares the one allowed multiplicity, its count and
the columns that must agree. Historical mode validates that declaration;
future mode rejects every duplicate key. The smallest correction for future
runs is therefore the new unique-key verifier, not a rewrite of the historical
file.

## Commands

```bash
captcha-benchmark-integrity verify
captcha-benchmark-integrity preflight \
  --expected-commit "$(git rev-parse HEAD)" \
  --output data/processed/<future-dataset> \
  --output reports/<future-report>
```

The M8C verifier remains the authority for replaying the 6,300 predictions,
images and report tables. The M8E verifier adds the independent byte-level
ground-truth pin, the full execution closure, the clean-tree guard and the
future manifest key rule.
