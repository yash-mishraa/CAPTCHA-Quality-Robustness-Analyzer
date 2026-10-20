# Experiment Log

**Status:** Template only. No experiments have been run.

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
| Task | `target-validation`, `quality-regression`, `corruption-classification`, `severity-classification`, `cnn-comparison`, `robustness-analysis` |
| Split | Frozen split ID plus hash, for example `split-v1` (`<hash>`) |
| Dataset | Dataset version plus manifest hash |
| Metrics | Name, value, unit, and confidence interval where applicable |

## Index

| ID | Date | Task | Status | One-line outcome |
| --- | --- | --- | --- | --- |
| _none yet_ | | | | |

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

## Milestone 0 note

No experiment entries exist. Milestone 0 established tooling and documentation
only; the first planned entry is the Milestone 1 reference-target validation.
