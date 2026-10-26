# Live Demo Plan — CAPTCHA Quality & Robustness Analyzer

**Status:** D0 planning document. Implementation, deployment, and publication have
not started.

**Baseline:** research release `v1.0.0` at commit
`9b7b62ff4669c39712b799d22d50b6bbb3e5b9c3`, also `origin/main`.

**Target:** a read-only Streamlit Community Cloud presentation layer over the
frozen M8C benchmark and M9A exploratory error analysis, intended for public
deployment **only after explicit public-display clearance**.

**Review rule:** do not implement the app, deploy it, publish a URL, commit this
plan, or push anything until the repository owner has reviewed and approved this
document. A public launch also requires a separate public-display/licensing
decision; plan approval alone is not public-display clearance.

This document has the 13 requested sections. It is deliberately a plan, not an
application design hidden inside code.

---

## 1. Executive summary

### 1.1 Recommendation

Build a three-view **research explorer** intended for public presentation, not a
CAPTCHA solver and not a model playground:

1. **Research Overview** — orient a first-time visitor: the question, the frozen
   design, the primary result, the main failure, and the non-claims.
2. **Benchmark Explorer** — let a visitor select one real benchmark record, see
   the clean reference and the degraded image side by side, and inspect that
   record's target, precomputed prediction, residual, and eight model features.
3. **Robustness & Error Analysis** — show where error concentrates: family,
   severity, calibration attenuation, source concentration, and font-level
   differences, with the exploratory status of M9A stated on the page.

The app is a **presentation layer**. It reads frozen files. It does not train,
fit, calibrate, select, generate, score uploads, or execute M9B. It never loads
the frozen model artifact because the predictions are already committed in
`predictions.csv`.

### 1.2 Why this is feasible now

- The GitHub repository is public and MIT-licensed.
- The M8C dataset is present in Git: 6,305 tracked files, including all 6,300 PNG
  images and the manifest/features tables.
- The M8C and M9A report tables needed by the app are tracked.
- The 6,300 `sample_id` values join one-to-one across manifest, features, and
  predictions.
- The M7E audit supports rendered-output redistribution for the intended
  evaluation workflow and prohibits redistribution of font binaries. It does
  **not** clear use in a product, a model, or a deployment; public launch is
  therefore a separate gate, not an automatic consequence of the audit.
- The data footprint is small enough for Community Cloud: GitHub reports the
  repository at approximately 33 MB; the M8C dataset is approximately 45 MB on
  disk. Community Cloud documents a 50 GB storage maximum.
- The app can use pinned precomputed predictions and avoid `scikit-learn`,
  `joblib`, and model loading entirely.

### 1.3 What must not be built

- No arbitrary image upload or live scoring.
- No CAPTCHA solver, OCR, readability study, or security evaluation.
- No new benchmark, prediction, model, dataset, image, or metric.
- No raw text exposure from `ground_truth.csv`.
- No display of local filesystem paths, font paths, or font binaries.
- No causal or confirmatory claims from M9A.
- No H1/H2/H3 result claims. M9B remains unexecuted.
- No public deployment or public URL until an explicit public-display/licensing
  decision is recorded. A local/private review build is not a public launch.

### 1.4 Success criteria

A visitor should be able to answer, without reading the repository:

- What was measured?
- How well did the frozen model do on the primary benchmark?
- Where did it fail?
- What does that failure not prove?
- Can I inspect a real case and its provenance?

A reviewer should be able to verify that every visible number comes from a named
tracked table, every image comes from the tracked M8C dataset, and every
limitation is stated in the interface.

### 1.5 Requirement coverage

| Requirement | Where it is handled |
| --- | --- |
| Documentation-only D0 planning | This document; §13 review gate |
| Preserve `v1.0.0` and frozen artifacts | §3, §11, §13 |
| Three read-only experiences (public launch gated) | §5–§8 |
| Real values only | §4, §6–§8, §10 |
| Side-by-side reference/corrupted images | §7 |
| No arbitrary upload / no live model | §7, §10, §13 |
| Explain reference-based SSIM | §7, §10 |
| No raw text or local paths | §4, §10, §11 |
| Editorial, research-specific design | §9 |
| Streamlit Community Cloud | §12 |
| Feature readiness classification | §13 |
| No commit/deploy without review | Review rule above; §13 |

---

## 2. Goals, audience, and non-goals

### 2.1 Experience goals

1. **Make the research legible in 30 seconds.** The first screen states what was
   measured, the primary result, and the most important limitation.
2. **Make one record inspectable in 90 seconds.** A visitor can choose a font,
   case, corruption family, and severity, then see the paired images and the
   record-level target/prediction/residual.
3. **Make the failure pattern honest.** The interface shows that thin-line and
   arc interference have negative R² and large overprediction, without turning
   that into a causal claim.
4. **Make provenance visible.** Each chart names its source table, scope, and
   row count. The app never silently recomputes published aggregates.
5. **Make the presentation safe.** No uploads, no secrets, no raw text, no local
   paths, no font binaries, no model execution, and no public URL before the
   public-display gate is cleared.

### 2.2 Audience

| Audience | What they need | Design implication |
| --- | --- | --- |
| Research reviewer | Exact values, scope, source paths, limitations | Source lines, report links, n, status labels |
| Hiring manager / recruiter | A clear, credible 60-second story | Strong first screen, restrained visuals, no jargon wall |
| ML / CV practitioner | Sample-level evidence and feature context | Record explorer, paired images, eight-feature table |
| Curious non-specialist | Plain-language explanation of SSIM and failure | Short definitions, expansions, no unexplained acronyms |
| Repository maintainer | A read-only, low-maintenance app | Pinned dependencies, hash-pinned inputs, no writes |

### 2.3 Non-goals

- **Not a product.** No accounts, uploads, persistence, API, or database.
- **Not a solver.** No text prediction, OCR, or CAPTCHA bypass.
- **Not a security study.** No claim about real CAPTCHA security or attack cost.
- **Not a human study.** No readability, legibility, or solver-difficulty claim.
- **Not a model demo.** The frozen M5C champion is not loaded or run.
- **Not a new experiment.** No training, fitting, calibration, selection, or
  prediction generation.
- **Not a confirmatory analysis.** M9A remains exploratory; M9B remains
  preregistered and unexecuted.
- **Not a leaderboard.** No "best font," "worst model," or cherry-picked ranking.
- **Not a replacement for the reports.** The app links to the authoritative
  reports and uses them as the source of truth.

### 2.4 What would count as a failure

- A polished page that implies human readability or security.
- A chart that hides negative R² values.
- A sample browser that shows a "representative" case without saying it is
  illustrative.
- An upload box that cannot compute a valid reference SSIM.
- A public app that exposes a local path, raw text, or a font binary.
- A new metric that looks authoritative but is not in a tracked report.
- Any change to the frozen research artifacts.

---

## 3. Verified starting state

### 3.1 Repository and release

| Check | Verified state |
| --- | --- |
| Working branch | `main` |
| HEAD | `9b7b62ff4669c39712b799d22d50b6bbb3e5b9c3` |
| `origin/main` | Same commit |
| Annotated tag | `v1.0.0` at the same commit |
| Working tree | Clean at the start of D0 |
| GitHub visibility | Public |
| License | MIT |
| Repository URL | <https://github.com/yash-mishraa/CAPTCHA-Quality-Robustness-Analyzer> |
| GitHub release | Not a GitHub Release; the release is the annotated tag and repository package |
| Local `app/` | Only `.gitkeep`; no Streamlit application exists |
| Local `models/` | Only `.gitkeep`; the frozen model artifact is not tracked |

### 3.2 Research status

- M0–M9C are complete.
- M9C selected **Path A — stop experimentation and package the research**.
- The experiment phase is closed.
- M9B is a preregistered but unexecuted protocol.
- H1, H2 and H3 remain `not_tested`.
- The dashboard was not implemented in the release and is not part of `v1.0.0`.
  This plan describes a post-release presentation layer, not a new research
  milestone.
- No model was trained, refitted, tuned, calibrated, selected, or replaced for
  this plan.

### 3.3 Important repository-state discrepancy

The public documentation currently says generated datasets and images are not
tracked. That is not true for M8C:

- `README.md`, `data/README.md`, `docs/ARCHITECTURE.md`, and
  `docs/REPRODUCIBILITY.md` describe generated data as Git-ignored.
- The actual M8C dataset has **6,305 tracked files**, including **6,300 tracked
  PNGs**, despite `data/processed/*` being ignored.
- `ground_truth.csv` remains untracked and ignored.

**Decision for D0:** do not rewrite release documentation in this planning
change. Record the discrepancy, rely on `git ls-files` and the actual files for
the demo, and make a separate post-review documentation correction if the owner
wants one. The app must not claim that a fresh clone lacks the benchmark, because
it does not.

### 3.4 Why the demo does not need the model

`reports/matched_font_benchmark_v0_1_0/predictions.csv` contains the frozen M5C
champion's precomputed predictions for all 6,300 rows. Loading a joblib model
would add dependencies, memory, and a new execution surface without changing the
displayed values. The app should therefore be a **read-only presentation** of
recorded outputs.

### 3.5 Verified data facts

- M8C dataset: 6,300 rows, 10 fonts, 30 cases, 300 sources, 21 rows per source.
- 5,700 primary rows and 600 diagnostic rows.
- 300 clean diagnostic rows and 300 elastic-0.5 px diagnostic rows.
- All rows are `eval`; there is no training split.
- 6,300 manifest image paths are relative, exist, and resolve to 192×64 grayscale
  PNGs.
- The manifest, features, and predictions join perfectly by `sample_id`.
- M8C reports and M9A tables are tracked.
- M3–M7 raw reports and the model artifact are local-only and are not part of the
  presentation layer.

---

## 4. Verified data and artifact inventory

### 4.1 Public, app-readable artifacts

| Artifact | Shape | App use | Status |
| --- | --- | --- | --- |
| `data/processed/matched_font_benchmark_v0.1.0/manifest.csv` | 6,300 × 31 | Public record metadata, image/reference paths, target, scope | Ready |
| `data/processed/matched_font_benchmark_v0.1.0/features.csv` | 6,300 × 9 | Eight selected-sample features | Ready |
| `data/processed/matched_font_benchmark_v0.1.0/images/**` | 6,300 PNGs | Paired reference/corrupted display | Ready for local/private display; public deployment gated by M7E scope |
| `reports/matched_font_benchmark_v0_1_0/predictions.csv` | 6,300 × 17 | Target, prediction, residual, scope, font group | Ready |
| `reports/matched_font_benchmark_v0_1_0/overall_metrics.csv` | 10 rows | Headline and scope metrics | Ready |
| `reports/matched_font_benchmark_v0_1_0/family_metrics.csv` | 7 rows | Family comparison | Ready |
| `reports/matched_font_benchmark_v0_1_0/severity_metrics.csv` | 21 rows | Severity paths | Ready |
| `reports/matched_font_benchmark_v0_1_0/font_metrics.csv` | 30 rows | Descriptive font comparison | Ready |
| `reports/matched_font_benchmark_v0_1_0/group_metrics.csv` | 9 rows | Font-group context | Ready |
| `reports/matched_font_benchmark_v0_1_0/uncertainty.csv` | 105 rows | Published source-grouped intervals | Ready |
| `reports/matched_font_benchmark_v0_1_0/comparison.csv` | 15 rows | Recorded baseline context, with caveats | Ready for context only |
| `reports/matched_font_benchmark_v0_1_0/dataset_composition.csv` | 60 rows | Integrity/composition checks | Ready |
| `reports/interference_error_analysis_v0_1_0/summary_metrics.csv` | 9 rows | M9A scope metrics | Ready |
| `reports/interference_error_analysis_v0_1_0/severity_metrics.csv` | 19 rows | Interference severity means | Ready |
| `reports/interference_error_analysis_v0_1_0/condition_calibration.csv` | 9 rows | Spread and slope summary | Ready |
| `reports/interference_error_analysis_v0_1_0/error_concentration.csv` | 9 rows | Source-concentration comparison | Ready |
| `reports/interference_error_analysis_v0_1_0/font_condition_metrics.csv` | 80 rows | Font × condition error table | Ready |
| `reports/interference_error_analysis_v0_1_0/paired_condition_contrasts.csv` | 4 rows | Descriptive paired contrasts | Ready, with caveats |
| `reports/font_roster_audit_v0_2_0/report.md` | Report | Licensing and attribution evidence | Ready to link |

### 4.2 Artifacts the app must not read

These files are either private, contain local paths, or are unnecessary. The app
should enforce an allowlist, not a blocklist.

| File | Reason |
| --- | --- |
| `data/processed/matched_font_benchmark_v0.1.0/ground_truth.csv` | Raw text; untracked and ignored; must not be read, tracked, exposed, or inferred from |
| `data/processed/matched_font_benchmark_v0.1.0/sources.csv` | Contains an absolute local font path |
| `data/processed/matched_font_benchmark_v0.1.0/dataset_summary.json` | Contains absolute font paths and unnecessary provenance |
| `data/processed/matched_font_benchmark_v0.1.0/source_cases.csv` | Not needed; manifest has the public record metadata |
| `reports/matched_font_benchmark_v0_1_0/run_metadata.json` | Environment and execution metadata; not needed for display |
| `reports/matched_font_benchmark_v0_1_0/input_manifest.csv` | Internal provenance; not needed and may contain paths |
| `reports/matched_font_benchmark_provenance_audit_v0_1_0/provenance.json` | Internal provenance; not needed |
| `models/**` | No tracked model artifact; no inference |
| Any font binary (`.ttf`, `.otf`, `.ttc`, `.woff`, `.woff2` from the benchmark roster) | M7E prohibits font-file redistribution |

### 4.3 Verified schemas

**Manifest:** 31 columns, including `row_id`, `sample_id`, `source_id`,
`split_group_id`, `split`, `case_id`, `case_index`, `font_id`, `font_sha256`,
`text_length`, `text_sha256`, `render_seed`, `variant_id`, `variant_family`,
`severity_level`, `amplitude_px`, `variant_seed`, `image_path`,
`reference_path`, `ssim_target`, `psnr_target`, `psnr_infinite`, `height`,
`width`, `image_sha256`, `reference_sha256`, `dataset_version`,
`generation_config_hash`, `protocol_scope`, `primary`,
`corruption_parameters_json`.

The app should load only a UI allowlist. It should not display `text_sha256`,
`font_sha256`, `render_seed`, `generation_config_hash`, or any path.

**Features:** `sample_id` plus the eight frozen features:

- `intensity_mean`
- `intensity_std`
- `contrast_rms`
- `ink_fraction`
- `laplacian_variance`
- `edge_density`
- `histogram_entropy`
- `noise_sigma_immerkaer`

**Predictions:** `row_id`, `sample_id`, `source_id`, `case_id`, `case_index`,
`font_id`, `font_group`, `variant_family`, `severity_level`, `amplitude_px`,
`protocol_scope`, `ssim_target`, `psnr_target`, `psnr_infinite`, `prediction`,
`residual`, `abs_residual`.

### 4.4 Verified composition

| Component | Value |
| --- | --- |
| Fonts | 10 |
| Cases | 30 |
| Sources | 300 |
| Rows | 6,300 |
| Primary | 5,700 |
| Diagnostic | 600 |
| Images | 6,300 |
| Feature rows | 6,300 |
| Rows per source | 21 |
| Split | `eval` only |
| Fonts in familiar group | 3 |
| Fonts in M7B held-out group | 3 |
| Fonts in new held-out group | 4 |

Families: `clean`, `gaussian_blur`, `gaussian_noise`, `thin_lines`,
`arc_strokes`, `elastic`. The four severity levels for blur, noise, thin, and arc
are primary. Elastic severities 2–4 are primary; elastic 0.5 px is diagnostic;
clean is diagnostic.

### 4.5 Verified headline values

These are the values the app should display, read at runtime from the named
tables rather than hardcoded into chart code.

| Scope | Rows | Sources | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| All primary | 5,700 | 300 | 0.059607 | 0.088610 | 0.732898 |
| Gaussian blur | 1,200 | 300 | 0.026499 | 0.049943 | 0.943091 |
| Gaussian noise | 1,200 | 300 | 0.038971 | 0.063510 | 0.735669 |
| Thin lines | 1,200 | 300 | 0.102073 | 0.132114 | −0.575158 |
| Arc strokes | 1,200 | 300 | 0.079578 | 0.103162 | −0.298911 |
| Elastic (primary) | 900 | 300 | 0.048018 | 0.059687 | 0.162498 |
| Clean (diagnostic) | 300 | 300 | 0.007352 | 0.013863 | Suppressed |
| Elastic 0.5 px (diagnostic) | 300 | 300 | 0.029109 | 0.036415 | Suppressed |

Published source-grouped 95% intervals for all primary: MAE `0.058304`–`0.060987`,
RMSE `0.086638`–`0.090613`, R² `0.721910`–`0.744091`. These are descriptive
percentile intervals, not significance tests or font-population estimates.

### 4.6 Verified M9A exploratory values

| Scope | MAE | R² | Large errors (≥ 0.20) | Top 5% source SSE share |
| --- | ---: | ---: | ---: | ---: |
| All primary | 0.059607 | 0.732898 | 5.67% | 9.82% |
| Interference | 0.090825 | −0.430242 | 11.08% | 10.56% |
| Thin lines | 0.102073 | −0.575158 | 15.17% | 11.13% |
| Arc strokes | 0.079578 | −0.298911 | 7.00% | 12.06% |
| Gaussian blur | 0.026499 | 0.943091 | 1.92% | 51.46% |
| Gaussian noise | 0.038971 | 0.735669 | 2.67% | 49.74% |

The fixed large-error threshold is `0.20` SSIM. M9A labels subgroup
large-error proportions exploratory because earlier reconnaissance had inspected
aggregate and severity summaries.

At the severe rungs:

| Rung | Target mean | Prediction mean | MAE | Large errors |
| --- | ---: | ---: | ---: | ---: |
| Thin lines s4 | 0.696463 | 0.904139 | 0.207882 | 55.33% |
| Arc strokes s4 | 0.741238 | 0.905631 | 0.164892 | 28.00% |

Within-source spread ratios are `0.320` for thin lines and `0.397` for arcs,
versus `0.968` for blur and `0.975` for noise. Median within-source slopes are
`0.241` and `0.337`. The interference bias is `−0.087180`.

Font interference MAE is descriptive, not causal: it ranges from `0.0358`
(Impact) to `0.1302` (Courier New) across the ten purposeful fonts. The contrast
is entangled with glyph shape, target difficulty, rasterisation, and source
composition.

### 4.7 Integrity and hash pinning

The app should fail closed if the core tables do not match the release. At
minimum, verify:

- `manifest.csv` SHA-256:
  `35ae405fd084070bf4d6c235638cbda6671573389e880dcca2a63d8b66049743`
- `features.csv` SHA-256:
  `fa2d79c85d682b26545dc2c4e3ec5ece2a41d85aea7e1ea1070f7256ff8519bb`
- `predictions.csv` SHA-256:
  `e6c1f1ba83bbb05c18fed8a9b0dd1c87014a6f6c3667475971b2834dfedd4dd4`
- `overall_metrics.csv` SHA-256:
  `abbc28d5c73e15091edbf79cc086449d171c91af4233964529811634ed0a79fa`
- `M9A summary_metrics.csv` SHA-256:
  `6d68e465887e3053e20a7c582f79e8318900e8cd14fb48005f15a54c224f1303`

For the selected record only, verify the image bytes against the manifest's
`image_sha256` and `reference_sha256`. Do not hash all 6,300 images on startup.

### 4.8 Licensing boundary

M7E records all ten faces as `eligible` for the intended **local,
evaluation-only workflow**, with rendered-output redistribution permitted and
font-file redistribution prohibited. Its limitations section also says, in
terms, that the audit "does not clear any face for use in a product, a model, or
a deployment." A public Streamlit app is a deployment-like use. This is a
material condition, not a footnote.

Important consequences:

- The app may display the tracked rendered PNGs in a local/private review build
  under the recorded conditions.
- A public deployed demo is **not** automatically cleared by M7E. It requires an
  explicit public-display/licensing decision by the owner before launch.
- If that decision is not recorded, keep the app local/private and publish no
  URL.
- The app must not distribute font binaries, font subsets, or converted font
  files.
- The benchmark is evaluation-only. The demo must not add training or
  fine-tuning.
- Font names are attribution, not an endorsement.
- Luminari has recorded residual licence uncertainties; the app should link to
  the audit rather than overstate certainty.
- The repository is MIT-licensed as code. The rendered images come from
  proprietary fonts and should not be described as MIT-licensed, public domain,
  or openly licensed data.
- This plan is not legal advice and does not supply the missing public-display
  clearance.

**Public-display gate:** a public launch is blocked until the owner records an
explicit decision that the deployed research demonstration is within the
permitted use, or obtains the necessary permission. The technical work can
proceed behind that gate, but no public URL should exist before it is resolved.

### 4.9 Primary evidence links

- [Repository README](../README.md)
- [Research summary](RESEARCH_SUMMARY.md)
- [Release notes](RELEASE_NOTES_v1.0.0.md)
- [M8C benchmark report](../reports/matched_font_benchmark_v0_1_0/report.md)
- [M9A exploratory report](../reports/interference_error_analysis_v0_1_0/README.md)
- [M7E font licensing audit](../reports/font_roster_audit_v0_2_0/report.md)
- [MIT license](../LICENSE)
- [Public GitHub repository](https://github.com/yash-mishraa/CAPTCHA-Quality-Robustness-Analyzer)

The app should link to the frozen evidence, not copy report prose into Python
strings. The plan uses relative links so they resolve from `docs/`.

---

## 5. Experience architecture and navigation

### 5.1 Three experiences, one shared frame

| Experience | Job to be done | Primary evidence |
| --- | --- | --- |
| Research Overview | Understand the study and the headline result | M8C overall/family/composition |
| Benchmark Explorer | Inspect one real record | Manifest + predictions + features + images |
| Robustness & Error Analysis | Understand where and how error concentrates | M9A tables + M8C family/font/uncertainty |

There is no fourth experience. "Data & limits" is an expander/section within the
Overview and a footer link, not a separate page.

### 5.2 Navigation

Use `st.navigation` with three explicit routes:

| Route | Page | Default |
| --- | --- | --- |
| `/overview` | Research Overview | Yes |
| `/benchmark` | Benchmark Explorer | No |
| `/error-analysis` | Robustness & Error Analysis | No |

Sidebar content:

```text
CAPTCHA Quality
& Robustness
Analyzer

v1.0.0 research release

Research overview
Benchmark explorer
Robustness & error

──────────────────
Frozen 6,300-image benchmark
10 fonts · 30 cases · 300 sources
Synthetic · evaluation-only

GitHub ↗
Data & licensing ↗
```

The sidebar is a table of contents and provenance panel, not a product menu. The
app must not use a custom JavaScript router.

### 5.3 Shared page anatomy

Every page follows the same reading order:

1. **Eyebrow** — scope and release status.
2. **Title** — an editorial statement, not a product slogan.
3. **Deck** — two sentences maximum: what this page shows and what it does not.
4. **Evidence block** — chart, specimen, or ledger.
5. **Reading note** — one sentence explaining what to notice.
6. **Source line** — table name, scope, n, and report link.
7. **Caveat/footer** — the non-claim relevant to the page.

Every page ends with a "What this does not show" block. The footer repeats the
global non-claims.

### 5.4 State and routing rules

- Page state is widget state; no persistent user accounts or cookies.
- Filter selections are validated against loaded tables.
- Optional deep-link query parameters may identify a page and a `sample_id`,
  but never a filesystem path or an arbitrary record id.
- If a selection has no public row, show an empty state. Do not substitute a
  different record.
- If a data contract fails, show a generic integrity error and stop. Do not
  render partial charts or fallback mock data.

### 5.5 Error and empty states

| State | Message | Behaviour |
| --- | --- | --- |
| Contract failure | "The frozen benchmark files did not pass the read-only integrity check. No substitute data has been generated." | Stop the page; link to repository integrity docs |
| Missing image | "This record's image is unavailable." | Show no image; no stock placeholder |
| Image hash mismatch | "This record failed its image integrity check." | Do not display the image |
| No matching rows | "No public row matches this selection." | Keep filters visible |
| Unavailable metric | "Not reported for this scope." | Do not interpolate or infer |
| Diagnostic selection | "Diagnostic record: excluded from primary metrics." | Show record only; no primary aggregate |

---

## 6. Research Overview

### 6.1 Purpose

Answer four questions in one scroll: what was measured, how well it worked, where
it failed, and what it does not mean.

### 6.2 Wireframe

```text
┌ sidebar ───────────┬ main ──────────────────────────────────────────────┐
│ wordmark           │ RESEARCH EXPLORER / v1.0.0                         │
│ nav                │ A measured look at how controlled degradation      │
│ release metadata   │ changes synthetic CAPTCHA-style images.            │
│                    │ [two-sentence deck]                                │
│                    │ ─────────────────────────────────────────────────  │
│                    │ [illustrative reference] [corrupted]  [ledger]    │
│                    │ ─────────────────────────────────────────────────  │
│                    │ PRIMARY RESULT    5,700 rows · 300 source groups   │
│                    │ 0.0596 MAE   0.0886 RMSE   0.733 R²                │
│                    │ ─────────────────────────────────────────────────  │
│                    │ WHERE IT FAILS                                     │
│                    │ [family MAE dot plot]  [negative R² note]          │
│                    │ ─────────────────────────────────────────────────  │
│                    │ WHAT THIS IS NOT                                   │
│                    │ [three short statements]                           │
│                    │ [Data, status & limitations expander]              │
└────────────────────┴────────────────────────────────────────────────────┘
```

### 6.3 Copy direction

**Eyebrow:** `RESEARCH EXPLORER / v1.0.0`

**Working title:** "A measured look at how controlled degradation changes
synthetic CAPTCHA-style images."

**Deck:** "A frozen image-quality model was evaluated zero-shot on 6,300
rendered variants across ten typefaces. The target is SSIM against each source's
clean reference. This is not a solver, a security test, or a human-readability
study."

**Primary-result label:** "Primary benchmark · 5,700 rows · 300 source groups"

**Primary ledger:** MAE `0.0596` · RMSE `0.0886` · R² `0.733`

**Failure note:** "Thin-line and arc-stroke interference are the weak point.
At thin-line severity 4, the mean target is 0.696 and the mean prediction is
0.904. R² is negative; the model is worse than the mean target predictor on that
slice."

**Non-claim block:**

- "No human labels, solver data, or production CAPTCHA evidence."
- "No causal mechanism for the interference error."
- "No confirmatory test. M9B is unexecuted; H1, H2, and H3 remain `not_tested`."

### 6.4 Illustrative specimen

Use one real record in the hero, clearly labelled as illustrative:

- `sample_id`: `src-4c20a455606b18eb__thin_lines_s4`
- Font: Arial
- Case: `case-000`
- Family: thin lines
- Severity: 4
- Target: `0.768919`
- Prediction: `0.892940`
- Residual: `−0.124021`

Microcopy: "Illustrative case, not a random sample. The image pair and values are
from the frozen M8C dataset and predictions table."

This specimen is chosen because it makes the failure visible. The label prevents
it from being mistaken for a typical or representative result.

### 6.5 Primary-result data

Read from `reports/matched_font_benchmark_v0_1_0/overall_metrics.csv`, scope
`all_primary`, and `uncertainty.csv`, scheme `source_grouped`.

Display:

- MAE `0.0596` with source-grouped interval `0.0583`–`0.0610`.
- RMSE `0.0886` with interval `0.0866`–`0.0906`.
- R² `0.733` with interval `0.722`–`0.744`.
- n = 5,700 primary rows; 300 source groups; 10 fonts; 30 cases.

### 6.6 Where it fails

A compact family MAE dot plot, sorted in benchmark order, with source-grouped
intervals where published. Direct labels show MAE and R². The chart must show
negative R² values rather than clipping them.

Reading note: "Blur and noise are the strongest slices. Thin lines and arcs have
negative R²: the model's predictions are worse than a mean-target baseline on
those observed slices."

No causal language. No "thin lines cause failure." The correct framing is
descriptive: "the error is larger and biased in this observed benchmark slice."

### 6.7 Data, status, and limitations expander

Include:

- **Scope:** synthetic 192×64 grayscale, fixed renderer, ten purposeful fonts,
  thirty cases, five corruption families, evaluation-only.
- **Target:** reference-based SSIM. PSNR is diagnostic only.
- **Unit:** source group, not individual image.
- **Diagnostics:** clean and 0.5 px elastic are excluded from primary metrics.
- **Baseline caveat:** the recorded M3 familiar-font baseline is not
  like-for-like with M8C. Do not chart them as an improvement/decline story.
- **M9A:** exploratory, descriptive, not causal, not a hypothesis test.
- **M9B:** protocol only; not executed.
- **H1/H2/H3:** `not_tested`.
- **Licensing:** rendered images are shown under M7E conditions in a
  local/private build; font binaries are not redistributed. Public deployment is
  gated by an explicit public-display decision, because M7E does not clear use in
  a deployment.
- **Data availability:** the tracked M8C dataset is present in Git; see the
  repository-state discrepancy in §3.3.
- Links: M8C report, M9A report, M7E audit, release notes, GitHub.

### 6.8 Acceptance criteria

- A visitor can state the primary MAE/RMSE/R² and the main failure in under a
  minute.
- Negative R² is visible and explained in-place.
- The illustrative specimen is labelled as illustrative.
- The page does not use a grid of stock KPI cards.
- Every number has a source line.
- The page never implies human readability, solvability, or security.

---

## 7. Benchmark Explorer

### 7.1 Purpose

Inspect one real benchmark record without turning the app into a leaderboard.
The explorer is for evidence and provenance, not for ranking models or fonts.

### 7.2 Wireframe

```text
┌ sidebar ───────────┬ main ──────────────────────────────────────────────┐
│ nav                │ BENCHMARK EXPLORER                                 │
│                    │ Inspect one case, not a leaderboard.               │
│                    │ ─────────────────────────────────────────────────  │
│                    │ FONT [Arial v]  CASE [case-000 v]                  │
│                    │ FAMILY [Thin lines v]  SEVERITY [4 v]              │
│                    │ [ ] include diagnostic records                     │
│                    │ ─────────────────────────────────────────────────  │
│                    │ REFERENCE              CORRUPTED                  │
│                    │ [clean PNG]            [thin-line PNG]            │
│                    │ TARGET 0.769  PRED 0.893  RESID −0.124             │
│                    │ ─────────────────────────────────────────────────  │
│                    │ WHAT THE MODEL SAW                                 │
│                    │ [8 feature rows]                                   │
│                    │ ─────────────────────────────────────────────────  │
│                    │ WHERE THIS RECORD SITS                             │
│                    │ [prediction vs target scatter]                     │
│                    │ ─────────────────────────────────────────────────  │
│                    │ [record metadata / provenance expander]            │
└────────────────────┴────────────────────────────────────────────────────┘
```

### 7.3 Filters

Use native Streamlit controls:

| Control | Options | Default |
| --- | --- | --- |
| Font | Ten font labels, grouped by familiar / M7B held-out / new held-out | Arial |
| Case | `case-000` … `case-029`, with text length shown | `case-000` |
| Family | Gaussian blur, Gaussian noise, Thin lines, Arc strokes, Elastic, and optionally Clean | Gaussian blur |
| Severity | Family-dependent: 1–4; elastic 2–4; optional diagnostic 0 or 0.5 px | 2 |
| Include diagnostic records | Checkbox | Off |

Rules:

- Default to primary rows.
- When diagnostics are enabled, label them clearly and never mix them into a
  primary aggregate.
- Filtering by family must not silently change the case or font.
- If a combination is impossible, show an empty state; do not fall back.
- Do not sort by residual or offer a "worst cases" ranking.

### 7.4 Specimen panel

Two images side by side, equal scale:

- **Reference:** "Clean source-specific reference."
- **Corrupted:** the selected variant.
- Both are 192×64 grayscale PNGs.
- Scale up with pixel-preserving rendering; do not smooth the pixels.
- Captions include an accessible text description and the record's sample ID.
- For a clean diagnostic row, show one image and label it "Clean reference
  (diagnostic)" rather than duplicating it.
- Do not show the relative image path in the UI.
- Do not show raw text. The dataset's literal text is not public in this app.

### 7.5 Record ledger

Custom editorial ledger, not `st.metric` cards:

| Label | Value | Note |
| --- | ---: | --- |
| Target SSIM | 0.769 | SSIM against the clean reference |
| Prediction | 0.893 | Frozen M5C champion, precomputed |
| Residual | −0.124 | Prediction minus target |
| Absolute error | 0.124 | Displayed for convenience |

Microcopy: "Residual = prediction − target. Negative values mean the model
overestimated image quality. Target is not human judgment."

Condition chips: font, font group, family, severity, protocol scope, case ID.
Show the sample ID in monospace.

### 7.6 What the model saw

A compact table of the eight frozen features for the selected row:

| Feature | Value | Plain-language note |
| --- | ---: | --- |
| Intensity mean | … | Average pixel intensity |
| Intensity standard deviation | … | Pixel spread |
| RMS contrast | … | Relative contrast |
| Ink fraction | … | Darker Otsu-class fraction |
| Laplacian variance | … | Edge/sharpness energy |
| Edge density | … | Canny edge fraction |
| Histogram entropy | … | Intensity-distribution entropy |
| Immerkaer noise sigma | … | Noise-plus-texture estimate |

Notes:

- Values come from `features.csv`; the app does not recompute them.
- The features are computed on the degraded image only.
- The model did not receive font, severity, text, source, target, or PSNR.
- Feature labels use the research definitions, not invented explanations.
- The table is small and readable; do not load the full 6,300-row feature table
  into the browser.

### 7.7 Prediction-vs-target context

A scatter of all 5,700 primary rows:

- x = target SSIM, y = prediction.
- Diagonal y = x reference.
- Selected record highlighted with a ring if it is primary.
- Color by family with direct labels; never color-only encoding.
- Tooltips show sample ID, family, severity, target, prediction, residual.
- Caption: "All primary rows. The highlighted point is the selected record.
  This is context, not a new metric."

If the selected record is diagnostic, hide the highlight and label the record as
diagnostic. Do not add a diagnostic point to a primary-only scatter without a
clear legend and an explicit note.

### 7.8 Reference SSIM explanation

Place a short, visible note in the explorer, not only in the footer:

> "SSIM here means similarity to the exact clean source image. The benchmark
> pairs every corrupted image with its known reference. An arbitrary upload would
> not have that reference, and the frozen model is not deployed in this app, so
> upload scoring is intentionally unavailable."

This explanation is required. It prevents the most likely misunderstanding of a
CAPTCHA demo.

### 7.9 Data and provenance

The selected row is a one-to-one join:

- `predictions.csv` on `sample_id`
- `manifest.csv` on `sample_id`
- `features.csv` on `sample_id`

All three tables have 6,300 unique `sample_id` values. The app must validate:

- 6,300 manifest rows, 6,300 feature rows, 6,300 prediction rows.
- No duplicate or missing `sample_id`.
- 5,700 primary and 600 diagnostic rows.
- One row per selected `(font, case, family, severity, scope)` combination.
- Image and reference paths are relative, under `images/`, and resolve inside the
  dataset root.
- Selected image bytes match the manifest SHA-256.

Do not read `sources.csv`, `dataset_summary.json`, `source_cases.csv`,
`ground_truth.csv`, or any font binary.

### 7.10 Optional enhancement: severity filmstrip

If schedule and layout allow, show the selected source across severities
(s1–s4) as a four-frame filmstrip. This is a real, traceable view of the frozen
corrupted images. It must be labelled "same source, different severity" and must
not imply a new analysis. Defer if it makes the mobile layout too dense.

### 7.11 Acceptance criteria

- All 6,300 rows are addressable through valid filter combinations, with
  diagnostics clearly separated.
- Reference and corrupted images display side by side and match their hashes.
- The record ledger uses precomputed values, not recomputed SSIM.
- The eight-feature table matches `features.csv`.
- No raw text, local path, font file, or upload control is present.
- The explorer never ranks fonts or cases.
- A visitor can explain why arbitrary uploads are unsupported.

---

## 8. Robustness and Error Analysis

### 8.1 Purpose

Show where the error is, not just how large it is. The page should make three
points:

1. Blur and noise are comparatively well predicted.
2. Thin-line and arc interference fail in a structured, broad way: larger MAE,
   negative bias, compressed prediction spread, and negative R².
3. The font differences are descriptive and entangled; they are not a causal
   font effect.

### 8.2 Wireframe

```text
┌ sidebar ───────────┬ main ──────────────────────────────────────────────┐
│ nav                │ ROBUSTNESS & ERROR ANALYSIS                        │
│                    │ Error is not uniform.                              │
│                    │ ─────────────────────────────────────────────────  │
│                    │ FAMILY ERROR PROFILE                               │
│                    │ [MAE dot plot with intervals] [R² strip]           │
│                    │ ─────────────────────────────────────────────────  │
│                    │ SEVERITY PATHS                                     │
│                    │ [small multiples: target vs prediction means]      │
│                    │ ─────────────────────────────────────────────────  │
│                    │ CALIBRATION & CONCENTRATION                        │
│                    │ [spread ratio]  [top 5/10/20% source SSE]          │
│                    │ ─────────────────────────────────────────────────  │
│                    │ FONT DIFFERENCES                                   │
│                    │ [interference MAE dot plot]                        │
│                    │ ─────────────────────────────────────────────────  │
│                    │ WHAT THIS DOES NOT SHOW                            │
│                    │ [exploratory / no causal / M9B unexecuted]         │
└────────────────────┴────────────────────────────────────────────────────┘
```

### 8.3 Chart 1: Family error profile

Data: `reports/matched_font_benchmark_v0_1_0/family_metrics.csv` primary scopes;
intervals from `uncertainty.csv`, scheme `source_grouped`.

Design:

- Horizontal dot plot for MAE, sorted in a stable descriptive order.
- Source-grouped interval lines where published.
- Direct value labels.
- A companion R² strip with a zero line. Do not hide negative R².
- Axis labels: "MAE (SSIM units, lower is better)" and "R² (1 is best; negative is
  worse than mean predictor)."
- No dual axis. No 3D. No rainbow palette.

Reading note: "Thin lines and arcs have both higher MAE and negative R². Blur and
noise are the comparatively strong slices."

### 8.4 Chart 2: Severity paths

Data: `reports/interference_error_analysis_v0_1_0/severity_metrics.csv` for
target mean and prediction mean by family and severity.

Design:

- Six small multiples: blur, noise, thin, arc, elastic, and optionally clean as a
  diagnostic reference.
- Two lines per panel: target mean (teal) and prediction mean (ochre).
- Connect only recorded severity levels with straight segments. Do not smooth or
  interpolate beyond recorded points.
- Mark s4 for thin and arc with the actual means:
  - Thin s4: target `0.696`, prediction `0.904`.
  - Arc s4: target `0.741`, prediction `0.906`.
- Caption: "Means of observed rows. Individual records vary; these are not
  model calibration curves."

### 8.5 Chart 3: Calibration attenuation

Data: `condition_calibration.csv`.

Show two published summaries:

- Prediction/target standard-deviation ratio, with a reference line at `1.0`.
- Within-source median slope, with a reference line at `1.0`.

Values: thin `0.320` / `0.241`; arc `0.397` / `0.337`; blur `0.968` / `0.967`;
noise `0.975` / `0.972`. Use precise CSV values, not the rounded labels, in
tooltips.

Reading note: "At severe interference the model compresses its response: the
predicted spread is much smaller than the target spread. This is descriptive, not
a causal mechanism."

### 8.6 Chart 4: Source concentration

Data: `error_concentration.csv`.

Show top-5%, top-10%, and top-20% source SSE shares for blur, noise, thin, and
arc. A compact horizontal bar or dot plot is appropriate.

Key comparison:

- Interference top-5% source share: `10.56%`.
- Blur: `51.46%`.
- Noise: `49.74%`.

Reading note: "Blur and noise error is concentrated in a relatively small number
of sources. Interference error is broad across sources, not carried by a handful
of outliers."

### 8.7 Chart 5: Font differences

Data: `font_condition_metrics.csv`, scope `interference`, and published
`mae_ci_low` / `mae_ci_high`.

Design:

- Horizontal dot plot of interference MAE by font, with intervals.
- Color by font group, but label each font directly.
- Sort by observed MAE only if the chart is explicitly called a descriptive
  ordering.
- Do not use "best" or "worst" in labels.
- Annotate the observed range: Impact `0.0358` to Courier New `0.1302`.

Reading note: "Font-level error differs descriptively. Glyph shape, stroke
width, target difficulty, and source composition are entangled. This is not a
causal font effect and not a recommendation."

### 8.8 What this does not show

A prominent, non-dismissible block:

- M9A is exploratory and descriptive.
- It does not test H1, H2, or H3.
- M9B is a protocol, not a result.
- No causal mechanism is identified.
- No human readability, solvability, security, or production robustness is
  measured.
- Font differences are not a font-population estimate.
- Large-error proportions are exploratory and use a fixed `0.20` threshold.
- Associations are not feature-selection results.

### 8.9 Data and acceptance

- Read every value from the named M9A/M8C table.
- Do not recompute slopes, intervals, p-values, or concentration.
- Display n and scope on every chart.
- Provide a data-table alternative for every chart.
- Never label a difference "significant."
- Never describe a font or corruption family as "safe," "secure," or "better
  for CAPTCHAs."

---

## 9. Visual design direction

### 9.1 Design thesis

**Editorial lab note, not an AI dashboard.** The app should feel like a
well-set research report that happens to be interactive: warm paper, dark ink,
hairline rules, specimen plates, marginal notes, and precise mono numerals.

The one strong visual move is the **paired specimen plate**: clean reference and
corrupted image presented as evidence, with a compact target/prediction/residual
ledger underneath. Everything else supports that.

### 9.2 Anti-patterns

Do not use:

- Gradients, glassmorphism, blurred orbs, neon glows, or "AI" sparkles.
- A grid of stock KPI cards with icons and colored arrows.
- Default Inter/Roboto/Arial as the primary typeface.
- Decorative emoji, emoji bullets, or emoji navigation icons.
- Generic SaaS hero copy ("Unlock insights," "Supercharge your workflow").
- Rounded 20 px cards, drop shadows, or floating panels.
- Pie charts, donuts, 3D charts, dual axes, or rainbow categorical palettes.
- Animated charts, autoplay, parallax, or scroll-triggered effects.
- A dark "cyber" theme that implies security tooling.
- A chat interface, upload dropzone, or "Run" button.

### 9.3 Typography

Bundle open-license fonts so the app is deterministic and does not call a font
CDN:

| Role | Face | Fallback | Use |
| --- | --- | --- | --- |
| Display / headings | Source Serif 4 | Georgia, serif | Page titles, large statements |
| UI / body | IBM Plex Sans | system sans-serif | Body, labels, controls |
| Numerals / code | IBM Plex Mono | SFMono-Regular, Consolas, monospace | Ledger values, IDs, feature values |

Rules:

- Body: 17 px, line-height 1.6, maximum 70–75 characters per line.
- Hero: `clamp(2.4rem, 5vw, 4.25rem)`, line-height 1.02, weight 600.
- Section title: 1.75–2.1 rem, weight 600.
- Eyebrow: 0.72 rem, uppercase, letter-spacing 0.12em, weight 600.
- Caption: 0.82 rem, line-height 1.4, muted ink.
- Ledger: mono, tabular numerals, 1.9–2.8 rem.
- Do not use more than three weights per family.
- Do not use display type for data tables or long paragraphs.
- Include the OFL license files with the bundled fonts.
- Do not bundle any of the ten benchmark fonts.

### 9.4 Palette

Semantic tokens, tested on the paper background:

| Token | Hex | Use | Contrast on paper |
| --- | --- | --- | --- |
| Paper | `#F5F0E6` | Page background | — |
| Paper alt | `#EBE3D5` | Quiet section fill | — |
| Ink | `#1B1B17` | Primary text | 15.2:1 |
| Ink muted | `#5E5A52` | Captions, secondary text | 6.0:1 |
| Rule | `#CFC5B5` | Hairlines, dividers | Non-text |
| Rust | `#A6402F` | Failure/emphasis, primary accent | 5.4:1 |
| Teal | `#1E5B57` | Target/reference, links | 6.9:1 |
| Ochre | `#8A5519` | Prediction | 5.5:1 |
| Blue | `#3B6E8F` | Blur family | 4.9:1 |
| Umber | `#6C5B3A` | Alternate family | 5.8:1 |
| Olive | `#4F6B4A` | Elastic family | 5.2:1 |
| Gray | `#6E6A61` | Clean/diagnostic | 4.7:1 |

Rules:

- Use color plus direct labels, line styles, or position. Never color alone.
- Do not use red/green as a success/failure dichotomy.
- Do not use color to imply a causal grouping.
- Text contrast must meet WCAG AA for normal text; verify at implementation time.
- Charts use paper background, thin rules, and dark labels.

### 9.5 Layout and grid

- Maximum content width: 1,180 px.
- Page gutters: 32 px desktop, 20 px mobile.
- 12-column mental model, implemented with native `st.columns`; avoid CSS grid
  hacks.
- Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64, 80 px.
- Section separation: 64–80 px desktop, 40–48 px mobile.
- Hairline rules separate sections; cards are used only for the specimen plate,
  and even that should be a border/rule treatment rather than a shadowed card.
- Images have no decorative frame. They sit on paper with a caption and a thin
  rule.
- Use generous whitespace and short line lengths.
- Keep the first viewport content-led: no marketing hero with a stock image.

### 9.6 Image treatment

- Display at 3× on desktop (576×192) and 2× on mobile (384×128) where space
  allows.
- Use pixel-preserving scaling (`image-rendering: pixelated` or equivalent) so
  thin strokes are not blurred by browser interpolation.
- Side-by-side labels: "Reference" and "Corrupted".
- For clean diagnostics, show one image labelled "Clean reference (diagnostic)".
- Captions are part of the evidence, not decoration.
- No zoom, pan, or image carousel in V1.
- No AI-generated exemplars, stock captchas, or unrelated imagery.

### 9.7 Chart language

- Prefer dot plots, interval plots, slope charts, small multiples, and scatter.
- No dual axes. No 3D. No pie/donut. No smoothed lines.
- Axis labels include units and direction where relevant.
- Show `n` in the chart subtitle or source line.
- Use direct labels on family/font series.
- Show zero and one reference lines where meaningful.
- Display negative R² honestly.
- Do not truncate axes to exaggerate differences. If an axis must be clipped,
  label it and explain.
- Include a "View data" expander with the exact plotted table for accessibility.
- Tooltips show recorded values, not recomputed ones.

### 9.8 Motion

- No entrance animations, parallax, autoplay, or animated chart transitions.
- Native Streamlit interactions only.
- Respect `prefers-reduced-motion`; if any CSS transition is used, keep it
  under 120 ms and non-essential.
- No animated loading art beyond the native spinner.

### 9.9 Component vocabulary

| Component | Implementation | Rule |
| --- | --- | --- |
| Eyebrow | `st.html` static block | Uppercase, small, source/status context |
| Display title | `st.html` static block | Serif, one strong statement |
| Deck | `st.markdown` / `st.html` | Two sentences maximum |
| Ledger | Custom semantic HTML | No stock metric cards |
| Specimen plate | `st.columns` + `st.image` + captions | Equal scale, hash-verified |
| Chart block | `st.altair_chart` + source caption + data expander | No chart without source/n |
| Feature table | `st.dataframe` | Eight rows, formatted, no index |
| Caveat | Custom callout via `st.html` | Visible, not hidden in tooltip |
| Source line | `st.caption` | Table, scope, n, report link |
| Footer | `st.html` | Global non-claims and licensing |

### 9.10 Responsive behaviour

- Test at 390×844, 768×1024, and 1440×900.
- At < 720 px: hero type shrinks, columns stack, images use full width, ledger
  becomes a 2×2 grid, charts reduce height but keep labels.
- No horizontal scrolling.
- Touch targets at least 44 px.
- Sidebar collapses natively.
- No hover-only information.
- Tables must wrap or use a vertical layout; never force a wide table on mobile.

### 9.11 Accessibility

- Semantic heading order; one `h1` per page.
- Descriptive image captions and adjacent textual record data.
- Every chart has an adjacent data table or text summary.
- Keyboard-operable native controls.
- Visible focus states.
- AA contrast, including muted captions.
- No color-only encoding.
- No motion necessary to understand content.
- Screen-reader pass on the specimen plate, ledger, and charts.
- Test with browser zoom at 200%.

### 9.12 Design acceptance

- It reads as a research report, not a SaaS product.
- It still works in grayscale except where color labels are redundant.
- A reader can identify the main result and failure without hovering.
- No gradients, glass, orbs, stock icons, or emoji.
- The first screen contains evidence, not a marketing slogan.
- Every number is in mono and visually distinct from prose.
- The design is restrained across all three pages.

---

## 10. Content, claims, and provenance policy

### 10.1 Non-negotiable claims

Use these framings consistently:

- "Synthetic CAPTCHA-style images."
- "Reference-based SSIM against each source's clean synthetic reference."
- "A frozen image-quality model evaluated zero-shot."
- "M8C is a matched cross-font benchmark."
- "The independent unit is the source group."
- "M9A is an exploratory descriptive analysis."
- "M9B is preregistered and unexecuted."
- "H1, H2, and H3 remain `not_tested`."
- "The demo is a presentation layer over frozen outputs."
- "Predictions are precomputed; the model is not deployed."

### 10.2 Required disclaimers

Place on every relevant page and in the footer:

> "Synthetic benchmark. Not a CAPTCHA solver. Not a security evaluation. Not a
> human-readability study."

> "The target is reference-based SSIM, not human judgment or CAPTCHA difficulty."

> "Rows within a source are not independent; the source group is the independent
> unit."

> "The ten fonts are a purposeful roster, not a sample of fonts."

### 10.3 Banned or dangerous claims

Do not write:

- "Solves CAPTCHAs," "breaks CAPTCHAs," or "bypasses CAPTCHAs."
- "Measures CAPTCHA security," "robustness in the wild," or "production-ready."
- "Human readability," "legibility," or "solver difficulty."
- "State-of-the-art," "AI-powered," or "generalizes to CAPTCHAs."
- "Thin lines cause the failure" or "Courier New is a bad font."
- "H1 is confirmed," "H2 is supported," or "H3 is falsified."
- "M9B shows…" — M9B was not executed.
- "The model is robust to interference." It is not, on this benchmark.
- "The model is condition-blind." M9A reports positive within-source rank
  correlations; the failure is attenuation, not blindness.
- "The dataset is MIT-licensed" or "open data." Code is MIT; rendered images come
  from proprietary fonts and are displayed under M7E conditions for the intended
  workflow. M7E does not clear public deployment.
- "Upload an image to score it." That view is blocked.

### 10.4 Metric interpretation rules

- MAE and RMSE are in SSIM units; lower is better.
- R² = 1 is perfect; 0 is a mean-target predictor; negative is worse than the
  mean-target predictor. Do not clip or hide negative values.
- A negative R² can coexist with a positive rank correlation. It reflects bias,
  scale, and spread, not necessarily a complete failure to rank conditions.
- The 95% intervals are descriptive source-grouped percentile intervals. They are
  not significance tests and not font-population estimates.
- The large-error threshold is 0.20 SSIM; subgroup proportions are exploratory.
- PSNR is diagnostic only and is not a target.
- Clean-reference R² is undefined/suppressed because the target is
  near-degenerate. Do not report it as a model result.
- Diagnostic 0.5 px elastic is not a primary result.
- M3 and M8C are not like-for-like. Do not chart them as an improvement or
  decline.

### 10.5 Provenance rules

- Every chart has a source line: table, scope, n, report link.
- Every record has a sample ID and source ID in the provenance expander.
- Every displayed metric is read from a tracked artifact.
- Do not recompute SSIM, predictions, residuals, slopes, intervals, or p-values.
- Do not compute new aggregate metrics for filtered views. Show counts and
  record-level values only.
- Do not silently drop rows or fill missing values.
- Do not show a fallback chart or invented value when a table is unavailable.
- Report links should point at the frozen `v1.0.0` evidence, not a moving branch,
  where possible.

### 10.6 Privacy and exposure rules

- Never read or display `ground_truth.csv`.
- Never display raw CAPTCHA text.
- Never display literal text hashes in the UI.
- Never display `font_path`, `clean_image_path`, or any absolute path.
- Never display environment variables, secrets, or user directories.
- Never display font binaries.
- Never accept an arbitrary path from a query parameter.
- Never log data rows or absolute paths in error messages.
- No accounts, uploads, cookies, or custom analytics.

### 10.7 Licensing and attribution copy

> "Rendered benchmark images are displayed under the M7E audit conditions for
> the intended evaluation workflow. Font binaries are not distributed. The ten
> typefaces are proprietary; their names identify the rendered material only.
> Public deployment requires explicit clearance. This is not a legal opinion."

Link to the M7E report. Do not claim an open-data licence for the images. Do not
treat the audit as public-display approval.

### 10.8 Claims test matrix

| Statement | Allowed? | Why |
| --- | --- | --- |
| "The model overpredicts SSIM at severe thin-line and arc interference." | Yes | Supported by target/prediction means and bias |
| "Thin lines cause the failure." | No | Causal claim not tested |
| "Blur and noise are the strongest observed slices." | Yes | Descriptive family metrics |
| "The model is robust to real CAPTCHAs." | No | Not measured |
| "Courier New is a worse CAPTCHA font." | No | Font effect descriptive and confounded |
| "M9A motivates H1–H3 but does not test them." | Yes | Correct status |
| "The benchmark has 300 independent fonts." | No | 10 fonts; 300 sources |
| "Upload an image to see its SSIM." | No | No clean reference; model not deployed |
| "The demo recomputes the model." | No | Predictions are frozen and precomputed |
| "The demo is a read-only presentation layer." | Yes | Accurate |

---

## 11. Technical architecture and data access

### 11.1 Architecture

```text
public GitHub repository
  ├── data/processed/matched_font_benchmark_v0.1.0/   (tracked, read-only)
  ├── reports/matched_font_benchmark_v0_1_0/          (tracked, read-only)
  ├── reports/interference_error_analysis_v0_1_0/     (tracked, read-only)
  ├── app/
  │   ├── streamlit_app.py
  │   ├── requirements.txt
  │   ├── views/
  │   │   ├── overview.py
  │   │   ├── benchmark.py
  │   │   └── error_analysis.py
  │   ├── lib/
  │   │   ├── contracts.py
  │   │   ├── data.py
  │   │   ├── charts.py
  │   │   ├── format.py
  │   │   ├── theme.py
  │   │   └── ui.py
  │   ├── content.py
  │   ├── assets/
  │   │   ├── fonts/          (open-license UI fonts only)
  │   │   └── OFL.txt
  │   └── tests/
  └── .streamlit/config.toml
```

No copies of M8C data. No duplicated model. No generated JSON cache. The app
reads the repository's frozen files directly and writes nothing.

### 11.2 Runtime data flow

```text
predictions.csv ─┐
manifest.csv ────┼─ one-to-one join on sample_id ─→ in-memory record table
features.csv ────┘                                  │
                                                    ├─ filters → one sample
                                                    ├─ image path + SHA-256
                                                    └─ feature table

report tables ─→ page-level selectors ─→ published aggregate charts
```

The app does not:

- import `captcha_quality`,
- load joblib/scikit-learn,
- read all images,
- recompute SSIM,
- run a model,
- generate datasets,
- write to the repository.

### 11.3 Loader contract

`app/lib/data.py` should provide cached loaders:

- `load_core()` — read `predictions.csv`, `manifest.csv`, `features.csv` with
  explicit column allowlists; validate row counts, unique keys, scopes, and
  one-to-one merge; return a single frame.
- `load_report_table(name)` — read a named report CSV from an allowlist; validate
  required columns; return a copy.
- `load_selected_image(sample_id)` — validate relative path, resolve inside the
  dataset root, read bytes, verify SHA-256, verify PNG dimensions and grayscale
  mode, return an image object.
- `load_report_markdown` — optional; not needed for display.

Loading rules:

- Use `@st.cache_data` with bounded entries.
- Use pandas only for tabular data. Use Pillow only for image validation/loading.
- Preserve identifiers as strings.
- Use `validate="one_to_one"` and fail on any mismatch.
- Apply the manifest column allowlist before display.
- Do not read the forbidden files listed in §4.2.
- Do not use `os.walk`, directory globs, or recursive image scans.
- Do not use `st.cache_resource` for a mutable DataFrame.
- Do not cache all images.

### 11.4 Path safety

For every selected image:

1. Accept only `image_path` or `reference_path` from the validated manifest row.
2. Parse with `PurePosixPath`.
3. Reject absolute paths, `..`, empty parts, and unexpected suffixes.
4. Require the first part to be `images`.
5. Join to the dataset root.
6. Resolve and confirm the result is inside the dataset root.
7. Read bytes; compare SHA-256 to the manifest.
8. Open with Pillow; require 192×64 and a grayscale-compatible mode.
9. Display only after all checks pass.

No user-supplied path ever reaches the filesystem.

### 11.5 Hash and contract validation

At minimum, validate:

- The core CSV SHA-256 values in §4.7.
- 6,300 unique `sample_id` values in each core table.
- 5,700 primary and 600 diagnostic rows.
- 10 fonts, 30 cases, 300 sources.
- All required columns present.
- No missing required values.
- `protocol_scope` only `primary` or `diagnostic`.
- `variant_family` in the six known values.
- Selected image/reference hashes match the manifest.

Fail closed. A generic user-facing error should say the frozen files did not pass
the read-only integrity check. Do not show a Python traceback or an absolute
path.

### 11.6 Chart implementation

Use Altair 5.5.0 directly. Streamlit 1.50.0 requires Altair `<6` and excludes
5.4.0/5.4.1; 5.5.0 satisfies that constraint.

- Build chart specs in `app/lib/charts.py`.
- Use one shared Altair theme.
- Read plotted values from report tables; do not hardcode them.
- Use `st.altair_chart(..., use_container_width=True)`.
- Add a data-table expander under each chart.
- Add source, scope, and n captions.
- Do not use `st.metric` for the main ledger.
- Do not use `st.tabs` for navigation.

If Altair causes a Cloud dependency problem, fall back to native Streamlit charts
and update the plan. Do not install Altair 6 while pinned to Streamlit 1.50.

### 11.7 Theme implementation

- Root `.streamlit/config.toml` sets the light base theme, paper background, ink
  text, rust primary color, and usage-statistics opt-out.
- `app/lib/theme.py` injects one static CSS block with `st.html`.
- Target custom classes plus minimal, documented Streamlit theme variables.
- Do not target generated `st-emotion-cache-*` class names.
- Do not use `!important` except, if unavoidable, for the app font/background and
  document why.
- Do not use a CSS reset that breaks native controls.
- Keep light mode as the V1 theme. Dark mode is optional and must be tested
  separately if added.
- Bundle fonts locally; do not call an external font CDN.

### 11.8 Dependencies

`app/requirements.txt` (planned exact pins):

```text
streamlit==1.50.0
pandas==2.3.3
pillow==11.3.0
altair==5.5.0
```

Rationale:

- `streamlit==1.50.0` is available from the configured index and pins the
  component/CSS surface.
- `altair==5.5.0` is required by Streamlit 1.50's `<6` constraint.
- `pandas` and `pillow` are direct app dependencies.
- `numpy`, `pyarrow`, and other transitive packages are installed by Streamlit;
  the app does not import them directly.
- No `scikit-learn`, `joblib`, `scikit-image`, `opencv`, `matplotlib`, `seaborn`,
  or FastAPI.
- No root-level `requirements.txt`; keep the research package's
  `pyproject.toml` untouched.
- Do not add a `packages.txt`, `runtime.txt`, or Dockerfile unless a verified
  Cloud build requires one.

### 11.9 Python version

Community Cloud defaults to Python 3.12 and supports released versions with
security updates. The repository requires Python 3.11+. Plan to deploy on
**Python 3.12** and test locally on 3.12 and 3.13.

### 11.10 Performance and memory

- Target warm page load under 3 seconds; cold Cloud start under 15 seconds.
- Keep resident app memory well under the 2.7 GB maximum; target under 250 MB.
- Load only the tables required by the active page.
- Load only one selected image pair at a time.
- Do not ship a 6,300-row table to the browser.
- The 5,700-point scatter is acceptable; test render time and reduce opacity/size
  before considering subsampling. Do not silently subsample.
- Cache CSV loads and selected image bytes with bounded caches.
- No background jobs, timers, or model initialization.

### 11.11 Security and privacy

- No authentication; the intended app is public and read-only, subject to B11
  public-display clearance.
- No uploads, forms, or user content.
- No secrets, API keys, or environment variables.
- No external API calls at runtime.
- No custom analytics or cookies.
- No `st.secrets`.
- No writes to disk.
- No directory listing.
- No arbitrary file reads.
- Escape any dynamic string interpolated into HTML.
- Do not use `st.markdown(..., unsafe_allow_html=True)` for data; use `st.html`
  only for static layout strings.
- Keep error messages generic; do not log rows or paths.

### 11.12 Research boundary

Allowed future app changes:

- New files under `app/`.
- `.streamlit/config.toml`.
- `docs/LIVE_DEMO_PLAN.md`.
- App-specific tests under `app/tests/`.
- Optional post-review README/docs link and artifact-policy correction.

Forbidden:

- Changes to `data/`, `reports/`, `models/`, `src/`, `configs/`, or public
  research values.
- New datasets, images, predictions, models, or benchmark outputs.
- Running benchmark generation, fitting, calibration, selection, or M9B.
- Re-tagging `v1.0.0`.
- Deleting or rewriting historical evidence.
- Reading or tracking `ground_truth.csv`.

---

## 12. Hosting and deployment

### 12.1 Feasibility

**Technically feasible; public launch conditionally blocked.** The public GitHub
repository, tracked M8C data, pinned dependencies, and static read-only
architecture fit Streamlit Community Cloud.

Technical feasibility is not public-display clearance. M7E's own limitations
section says it does not clear any face for use in a product, a model, or a
deployment. A public Streamlit app is a deployment-like use. D5 therefore has two
gates:

1. a technically successful Cloud build and smoke test, and
2. an explicit public-display/licensing decision recorded by the owner.

D0 authorizes neither gate.

Verified constraints from the official documentation:

| Constraint | Value |
| --- | --- |
| Repository | Public GitHub repository |
| Entry point | `app/streamlit_app.py` |
| Python default | 3.12; selectable in Advanced settings |
| Dependency file | `app/requirements.txt` in the entrypoint directory |
| Root `pyproject.toml` | Research package file; not the app dependency file |
| Community Cloud CPU | 0.078 cores minimum, 2 cores maximum |
| Community Cloud memory | 690 MB minimum, 2.7 GB maximum |
| Community Cloud storage | 50 GB maximum |
| Hibernation | Apps without traffic for 12 hours go to sleep |
| Secrets | None required |
| Database | None |
| Custom server | None |

### 12.2 Deployment configuration

- Repository: `yash-mishraa/CAPTCHA-Quality-Robustness-Analyzer`
- Branch: `main` after the reviewed app commit
- Main file path: `app/streamlit_app.py`
- Python: 3.12
- Requirements: `app/requirements.txt`
- Secrets: none
- Config: root `.streamlit/config.toml`
- No `packages.txt` unless a build failure proves a system package is required.

### 12.3 Deployment steps

0. **Do not create a public app until the public-display gate in §12.6 is
   cleared.** Build and validate locally/privately first.
1. Merge the reviewed app commit to `main`.
2. Open Streamlit Community Cloud and create an app from the public repository.
3. Select branch `main` and entrypoint `app/streamlit_app.py`.
4. In Advanced settings, select Python 3.12.
5. Confirm the build log uses `app/requirements.txt` and does not attempt to
   install the research package from the root `pyproject.toml`.
6. Choose an available public subdomain; do not promise a URL before it is
   created.
7. Wait for the app to become healthy.
8. Run the post-deploy checklist in §13.
9. Only after the public-display decision is recorded and the smoke test passes,
   link the URL from the repository or a resume.

### 12.4 Public URL policy

- Do not claim a live URL in this plan.
- Do not create or publish a public URL until the public-display gate is cleared.
- Use a stable, descriptive subdomain if available.
- The URL is public; do not add authentication.
- Do not put secrets or private data behind the app.
- If the subdomain is unavailable, choose another; do not break links later.

### 12.5 Operations

- The app is stateless and read-only.
- No cron, queue, database, or background worker.
- Streamlit Cloud hibernates after 12 hours without traffic; first visit may be
  slower. Say "the app may take a moment to wake" in the README, not on the
  landing page.
- Use the Cloud logs for build/runtime errors.
- Do not log data rows or absolute paths.
- To update, commit a reviewed app change; Cloud redeploys from `main`.
- To roll back, revert the app commit; do not rewrite the research tag.
- Keep the app dependency pins stable. Re-test after any Streamlit upgrade.

### 12.6 Deployment risks and conditions

| Condition | Status | Action |
| --- | --- | --- |
| Public image display | Rendered-output redistribution is supported for the intended evaluation workflow; **public deployment is not cleared by M7E** | Record an explicit public-display/licensing decision before creating the public app; keep local/private until then |
| Luminari residual licence uncertainty | Recorded, not resolved | Link audit; do not overstate; obtain review if needed |
| No separate data licence was identified | Open documentation item | Do not call it MIT/open data; add attribution note |
| Docs say data is untracked | Known inconsistency | Correct post-review; app relies on actual tracked files |
| Root `pyproject.toml` dependency ambiguity | Controllable | Use `app/requirements.txt`; verify build log |
| CSS brittleness across Streamlit versions | Controllable | Pin 1.50.0; targeted selectors; smoke tests |
| Cloud cold starts | Accepted | Mention in README; no app-side workaround |
| No formal legal review | Not clearance | Do not make legal assurances; public launch requires an explicit decision |

**No hard technical blocker** prevents the three-experience read-only app. There
is, however, a **public-launch licensing/approval blocker**: M7E does not clear
deployment, so no public URL should be created until that is resolved.

### 12.7 What not to do in deployment

- Do not deploy during D0.
- Do not add a backend, database, auth, or API.
- Do not upload data to another service.
- Do not expose the local model or raw text.
- Do not use Cloud secrets for data access.
- Do not add telemetry or custom analytics.
- Do not make the app depend on a local font path or a local file outside the
  repository.

---

## 13. Validation, feature readiness, risks, and phased delivery

### 13.1 Feature readiness matrix

Status definitions:

- **ready** — supported by verified public artifacts and within scope.
- **optional** — useful, but not required for the first public release.
- **blocked** — cannot be built without violating scope, privacy, licensing, or
  the frozen release; do not replace with mock data.

| ID | Feature | Status | Evidence / constraint |
| --- | --- | --- | --- |
| R01 | Research Overview | **ready** | M8C/M9A tables tracked; copy can be sourced |
| R02 | Headline MAE/RMSE/R² ledger | **ready** | `overall_metrics.csv`; exact values verified |
| R03 | Family, severity, group metrics | **ready** | `family_metrics.csv`, `severity_metrics.csv`, `group_metrics.csv` |
| R04 | Record explorer with filters | **ready** | 6,300 joined rows; 10 fonts, 30 cases |
| R05 | Side-by-side reference/corrupted images | **ready** | 6,300 tracked PNGs; M7E supports rendered-output distribution for the intended workflow; public display depends on B11 |
| R06 | Selected-record feature table | **ready** | `features.csv`; eight frozen features |
| R07 | Prediction-vs-target scatter | **ready** | `predictions.csv`; 5,700 primary rows |
| R08 | M9A error analysis | **ready** | M9A tables tracked; label exploratory |
| R09 | Provenance and limitation panel | **ready** | Reports and release docs tracked |
| R10 | Responsive editorial theme | **ready** | Native Streamlit components plus targeted CSS |
| R11 | Read-only path/hash contracts | **ready** | Relative paths verified; selected image hashes available |
| R12 | Three-route navigation | **ready** | `st.navigation`/`st.Page` in pinned Streamlit |
| R13 | Local smoke/contract tests | **ready** | App tests can run without network or model |
| R14 | Streamlit Cloud deployment mechanics | **ready** | Public repo, 33 MB, pinned deps, no secrets; build path planned but not yet run |
| O01 | Severity filmstrip | **optional** | Real data; defer if mobile density suffers |
| O02 | Copy/download one record | **optional** | Only public fields; no raw text |
| O03 | Existing M9A SVG figures as appendix | **optional** | Tracked figures; style may differ |
| O04 | Dark theme | **optional** | Must be separately contrast-tested |
| O05 | Query-param sample deep links | **optional** | Validate sample_id; never a path |
| O06 | GitHub Actions app checks | **optional** | No CI exists today; add later if desired |
| O07 | Inline glossary/tooltips | **optional** | Improve non-specialist comprehension |
| B01 | Arbitrary upload scoring | **blocked** | No clean reference; model not deployed; would be new inference |
| B02 | Live model inference / load frozen M5C | **blocked** | Model artifact local-only; predictions precomputed |
| B03 | Raw ground-truth text | **blocked** | `ground_truth.csv` untracked/ignored; must not be read |
| B04 | Font binary preview/download | **blocked** | M7E prohibits font-file redistribution |
| B05 | Historical M3–M7 raw drilldowns | **blocked** | Those reports are local-only and not tracked |
| B06 | New metrics, predictions, or benchmark runs | **blocked** | Would change the frozen research boundary |
| B07 | M9B execution / H1–H3 tests | **blocked** | Preregistered, unexecuted; no authorization |
| B08 | Human readability/security claims | **blocked** | Not measured; out of scope |
| B09 | Accounts, uploads, persistence, analytics | **blocked** | Unnecessary; privacy and scope risk |
| B10 | Solver or security-tool framing | **blocked** | Misrepresents the research |
| B11 | Public launch / public URL under current M7E clearance | **blocked** | M7E explicitly does not clear use in a product, model, or deployment; explicit public-display/licensing decision required |

### 13.2 Phased delivery

**D0 — Plan (this document).**
- Inspect release, data, reports, licensing, and hosting constraints.
- Write this plan.
- Do not implement, deploy, commit, or push without review.
- Exit: owner approves the plan.

**D1 — Foundation.**
- Create `app/` scaffold, pinned requirements, theme, contracts, loaders.
- Implement the Research Overview with real M8C values.
- Add local smoke tests.
- Exit: Overview loads, values match reports, no forbidden reads.

**D2 — Explorer.**
- Implement filters, paired image display, record ledger, feature table, scatter.
- Add path/hash guards and empty/error states.
- Exit: representative records across all families/fonts display correctly;
  diagnostics are separated.

**D3 — Error Analysis.**
- Implement family/severity/calibration/concentration/font views.
- Add source lines, data-table alternatives, and caveats.
- Exit: every displayed value matches its M9A/M8C table; no causal claims.

**D4 — Hardening.**
- Responsive and accessibility passes.
- Performance checks.
- Contract, claims, link, and privacy tests.
- Visual review at 390/768/1440.
- Exit: all acceptance criteria pass locally.

**D5 — Deployment.**
- Precondition: B11 public-display/licensing decision is recorded. Without it,
  stop at a local/private validation build.
- Deploy to Streamlit Community Cloud on Python 3.12 only after that clearance.
- Verify dependency resolution, three routes, images, charts, mobile, and logs.
- Exit: if and only if B11 is cleared, a public URL works and no
  secrets/paths/raw text are exposed; otherwise the exit is a local/private build
  with no published URL.

**Post-deploy (optional).**
- Add a README link in a separate reviewed commit only after public-display
  clearance and a successful public smoke test.
- Correct the tracked-data documentation inconsistency.
- Add dark mode, filmstrip, or data downloads only after the core app is stable.

### 13.3 Testing plan

**Data contracts**

- Core CSV hashes match §4.7.
- 6,300 rows in each core table; unique `sample_id`.
- One-to-one join across manifest/features/predictions.
- 5,700 primary, 600 diagnostic.
- 10 fonts, 30 cases, 300 sources, 21 rows per source.
- Required columns present; no missing required values.
- All manifest image/reference paths relative and under `images/`.
- All paths resolve inside the dataset root.
- No selected image hash mismatch.
- No reading of forbidden files (code review plus an allowlist test).

**Metrics and content**

- Headline values equal `overall_metrics.csv`.
- Family values equal `family_metrics.csv`.
- Severity values equal `severity_metrics.csv` and M9A `severity_metrics.csv`.
- Font values equal `font_condition_metrics.csv`.
- Concentration values equal `error_concentration.csv`.
- No negative R² is suppressed.
- Required disclaimers present.
- Banned claims absent from UI copy.
- M9B/H1/H2/H3 statuses correct.

**Page tests**

- `AppTest` or direct view tests: no uncaught exception on each route.
- Explorer selects a valid record for each family and font.
- Diagnostic toggle changes scope and labels.
- Image pair renders; clean diagnostic shows one image.
- Empty state appears for an impossible selection.
- Contract failure produces a generic error, not a traceback.
- Data-table expanders render.

**Accessibility and responsive**

- 200% zoom.
- 390, 768, 1440 widths.
- Keyboard-only navigation.
- Screen-reader pass on specimen, ledger, and charts.
- Contrast check for text and chart labels.
- No horizontal scroll.
- Charts have adjacent data tables/captions.
- No color-only meaning.

**Performance**

- Warm page load < 3 s; cold start < 15 s.
- Selected image load < 500 ms in Cloud.
- App memory target < 250 MB.
- No full image-directory scan.
- No full 6,300-row table in the browser.

**Privacy/security**

- No `ground_truth`, `font_path`, `sources.csv`, or `dataset_summary.json` reads.
- No absolute paths in rendered HTML or logs.
- No secrets.
- No uploads.
- No external runtime API calls.
- No telemetry.
- No `unsafe_allow_html` interpolation of data.

**Repository integrity**

- `git diff --name-only` shows only `app/`, `.streamlit/`, and
  `docs/LIVE_DEMO_PLAN.md` (plus a separately reviewed docs link later).
- Research artifacts and hashes unchanged.
- `v1.0.0` still points at the release commit.
- `git diff --check` passes.
- Existing `pytest` suite passes.
- No new tag.
- No benchmark/model commands were run.

### 13.4 Acceptance criteria

The app is ready for a local/private review build when the technical criteria
below pass. It is ready for **public launch** only when the technical criteria
pass **and** the public-display gate (B11) is explicitly cleared.

- [ ] The owner has reviewed and approved this plan.
- [ ] The owner has recorded an explicit public-display/licensing decision for the
      deployed app, or the app remains local/private with no public URL.
- [ ] The three routes work without login in the local/private build.
- [ ] All headline, family, severity, font, and concentration values match named
      tracked tables.
- [ ] Every chart has a source line and a data-table alternative.
- [ ] Reference/corrupted images display side by side and pass hash checks.
- [ ] The record ledger uses precomputed target/prediction/residual values.
- [ ] The eight-feature table matches `features.csv`.
- [ ] Diagnostic rows are clearly separated from primary metrics.
- [ ] Negative R² is visible and explained.
- [ ] M9A is labelled exploratory; M9B is labelled unexecuted; H1/H2/H3 are
      labelled `not_tested`.
- [ ] No raw text, local path, font binary, secret, upload, or model execution is
      present.
- [ ] The app is readable and usable at 390, 768, and 1440 px.
- [ ] Accessibility checks pass.
- [ ] The Cloud deployment, if cleared, uses `app/requirements.txt` and Python
      3.12.
- [ ] If a public URL is created, it has been smoke-tested and contains no errors.
- [ ] Research artifacts and `v1.0.0` are unchanged.
- [ ] The plan and app changes were reviewed before commit/push.

### 13.5 Risk register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Misread as a CAPTCHA solver/security tool | Medium | High | Repeated non-claims; no upload; no solver language |
| Misread high SSIM as readability | Medium | High | Define SSIM; state non-claims next to metrics |
| Negative R² hidden or misunderstood | Medium | High | Show negative values; explain mean-baseline meaning |
| M9A treated as causal/confirmatory | Medium | High | Exploratory labels; H1/H2/H3 status; no causal copy |
| Font differences treated as recommendations | Medium | High | Descriptive language; confounding note |
| Public image licensing challenge | Medium | High | M7E does not clear deployment; keep local/private until explicit public-display clearance; no font files; no legal overclaim |
| Raw text or local path exposure | Low | High | Allowlist loaders; no raw text; path guards; tests |
| Data drift / accidental edits | Low | High | Hash pinning; fail closed; no writes |
| Streamlit CSS breakage on upgrade | Medium | Medium | Pin 1.50.0; targeted selectors; smoke tests |
| Root `pyproject.toml` dependency confusion | Medium | Medium | `app/requirements.txt`; verify Cloud build log |
| Cloud cold start / hibernation | High | Low | Document; no app-side workaround |
| Memory/performance from images | Low | Medium | Lazy selected images; bounded caches |
| Docs say data is untracked | High | Medium | Correct after review; app uses actual tracked files |
| No maintenance after launch | Medium | Medium | Pin deps; minimal app; document update path |
| Accessibility gaps | Medium | Medium | Contrast, captions, tables, keyboard, 200% zoom |

### 13.6 Open decisions for review

| Decision | Recommendation | Why |
| --- | --- | --- |
| Public deployment | Technical path is ready; **public launch remains blocked** until the owner records an explicit public-display/licensing decision | M7E does not clear use in a product, model, or deployment |
| UI font bundle | Bundle Source Serif 4 + IBM Plex Sans/Mono (OFL) | Deterministic, no CDN, deliberate typography |
| Theme scope | Light editorial theme in V1 | Focused visual system; dark mode is optional later |
| Filmstrip | Optional after core explorer | Adds value but increases mobile density |
| README live-demo link | Add only after public-display clearance and a successful public smoke test, in a separate commit | Keeps D0 documentation-only |
| Data-policy correction | Separate post-review docs commit | Fix the tracked-data discrepancy without rewriting history |
| Formal data licence | Do not claim one; add attribution note | Code is MIT; images come from proprietary fonts |
| Custom domain | Not required | Use an available Streamlit subdomain |
| Analytics | None | Privacy and scope; Cloud built-ins only |

### 13.7 Review gate and commit policy

At the time of writing:

- This plan is **uncommitted**.
- No app file exists.
- No deployment has been created.
- No public URL has been published.
- No public-display/licensing clearance has been recorded.
- No model, benchmark, dataset, prediction, or history has changed.

After owner review:

1. Commit only `docs/LIVE_DEMO_PLAN.md` if the plan is approved.
2. Do not push without explicit approval.
3. Do not create a new tag.
4. Do not modify `v1.0.0`.
5. Begin D1 only after approval.

If the plan is not approved, leave the repository unchanged.

### 13.8 Final guardrail

The presentation layer must remain what the research is: a careful, bounded,
read-only measurement of synthetic CAPTCHA-style image degradation. It must never
become a solver, a security claim, a model-training interface, or a showcase of
numbers that cannot be traced back to a frozen table. It must not become a public
deployment until the public-display gate is explicitly cleared.

---

*End of D0 plan. No implementation, deployment, or publication is authorized by
this document until the repository owner reviews and approves it.*
