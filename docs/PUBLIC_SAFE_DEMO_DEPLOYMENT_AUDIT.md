# D3C — Public-Safe Dashboard Deployment-Candidate Audit

- **Milestone:** D3C
- **Repository:** `CAPTCHA-Quality-Robustness-Analyzer`
- **Subject:** D3B Results-Only Public-Safe Dashboard Implementation
- **Audit Date:** 2026-10-26
- **Status:** Complete (Independent Technical Audit)
- **Final Classification:** `deployment_candidate_ready_for_owner_review`

---

## Executive Summary

This milestone independently audits the results-only public-safe profile implemented in Milestone D3B (`feat: implement results-only public-safe dashboard profile`, commit `20b5892d65ad2be3e12d84e9b4cee37dfb9516d8`).

The objective is to determine whether the D3B code, data-access paths, dependencies, and configuration enforce the documented public-safe boundary, specifically ensuring that:
1. No benchmark CAPTCHA images or proprietary font binaries are loaded, decoded, hashed, served, or exposed.
2. The internal answer key (`ground_truth.csv`) and trained model artifact (`model.joblib`) are completely excluded and unreferenced.
3. User-visible outputs and charts omit local filesystem paths, image hashes, and non-public metadata.
4. All headline metrics and analytical findings match authoritative $v1.0.0$ report packages exactly.
5. Telemetry and network calls are disabled, and offline self-contained font assets (SIL OFL-1.1) are bundled.

**Audit Conclusion:**
The implementation successfully enforces the public-safe results-only boundary. All 85 unit and integration tests pass, lifecycle/release verification tests pass (33/33), and Ruff check/format checks pass. The application starts and renders cleanly in headless mode.

The public-safe profile is classified as **`deployment_candidate_ready_for_owner_review`**.
*Notice:* A successful technical audit does not constitute legal advice or formal deployment authorization. Public hosting on Streamlit Community Cloud remains subject to explicit repository owner authorization.

---

## 1. Audited Commit and Release-Tag Identities

- **Audited Commit (HEAD):** [`20b5892d65ad2be3e12d84e9b4cee37dfb9516d8`](https://github.com/yash-mishraa/CAPTCHA-Quality-Robustness-Analyzer/commit/20b5892d65ad2be3e12d84e9b4cee37dfb9516d8) (`origin/main`)
- **Audited Commit Message:** `feat: implement results-only public-safe dashboard profile`
- **Release Tag:** `v1.0.0` points to commit `9b7b62ff4669c39712b799d22d50b6bbb3e5b9c3` (Annotated tag object: `dd2fcf120597259292f0d826ada6e1ee76a15e45`)
- **Parent Commit:** `6b2a4a7` (`docs: add public-safe demo feasibility and asset separation plan`)
- **Working Tree State:** Clean (`nothing to commit, working tree clean`)
- **Research Release Preservation:** Release tag `v1.0.0`, the frozen M5C champion, all historical milestone reports (M3–M9), and pinned artifact hashes remain strictly unaltered.

---

## 2. Public-Safe Execution-Path Trace

The execution path of the application under `PUBLIC_SAFE_MODE=true` was traced from entry point to rendered HTML/Vega-Lite outputs:

```
[Entry Point: app/streamlit_app.py]
  │
  ├─► 1. Environment & Path Initialization
  │     - Sets sys.path[0] to repo root
  │     - Checks contracts.is_public_safe_mode() -> True
  │     - Configures Streamlit page: layout="wide", initial_sidebar_state="auto"
  │
  ├─► 2. Offline Theme Injection (app/lib/theme.py)
  │     - Inlines bundled WOFF2 fonts as base64 data URIs (IBMPlexSans, IBMPlexMono, SourceSerif4)
  │     - Zero network/CDN requests
  │     - Hides Deploy button & toolbar via CSS
  │
  ├─► 3. Sidebar Rendering
  │     - Renders ".sidebar-badge.public-safe-badge": "Results-Only Profile · Public-Safe"
  │     - Renders metadata: "Frozen benchmark results · 5,700 primary evaluation rows"
  │     - Displays public-safe disclaimer note
  │
  ├─► 4. Route Execution (via st.navigation)
  │     │
  │     ├─► Route A: /overview (app/views/overview.py)
  │     │     - data.load_core() -> read_core(public_safe=True)
  │     │         - Reads manifest.csv, features.csv, predictions.csv
  │     │         - Bypasses _validate_manifest_paths (no images/ folder needed)
  │     │         - Strips image_path, reference_path, image_sha256, reference_sha256
  │     │         - Validates finite values across all 8 features and predictions
  │     │     - Loads M8 overall, uncertainty, family, composition tables
  │     │     - Loads M9 severity metrics
  │     │     - Specimen Panel Branch (contracts.is_public_safe_mode() == True):
  │     │         - Suppresses st.image() and load_selected_image()
  │     │         - Renders Illustrative Record Exhibit card (SSIM target 0.769, pred 0.893, res -0.124)
  │     │         - Renders Handcrafted Feature Summary table
  │     │     - Displays research scope, headline metrics ledger, limitations, and footer
  │     │
  │     ├─► Route B: /benchmark (app/views/benchmark.py)
  │     │     - Reads sanitized core DataFrame (6,300 rows, 0 image paths/hashes)
  │     │     - Interactive filter controls (Font, Case, Family, Severity, Diagnostics)
  │     │     - Specimen Branch (contracts.is_public_safe_mode() == True):
  │     │         - Suppresses st.image() and load_selected_image()
  │     │         - Renders Selected Record Exhibit card with Record Ledger & Chips
  │     │     - Features Panel: Renders 8 handcrafted spatial features in tabular format
  │     │     - Scatter Plot: charts.prediction_scatter subsets only 7 non-path columns
  │     │         (ssim_target, prediction, residual, font_id, variant_family, severity_level, is_selected)
  │     │         Zero image paths, hashes, or filesystem paths in Vega-Lite JSON payload
  │     │
  │     └─► Route C: /error-analysis (app/views/error_analysis.py)
  │           - Loads M9 summary, severity, calibration, concentration, and font metrics
  │           - Renders headline ledger, Altair family/severity curves, calibration histograms
  │           - Completely image-free by design
  │           - Displays M9A exploratory disclaimer and non-claims caveats
  │
  └─► 5. Page Completion: ui.footer() renders PUBLIC_SAFE_BUILD_NOTE and evidence links
```

**Trace Verdict:** There is zero code path reachable from the Streamlit UI under `PUBLIC_SAFE_MODE=true` that opens, decodes, serves, or references benchmark PNG files.

---

## 3. Exact File/Column Allowlist and Join Validation

### 3.1 Input File Allowlist
Every file accessed by the dashboard in public-safe mode is explicitly allowlisted and hash-pinned where applicable:

| File Path | Pinned SHA-256 Hash | Allowed Columns in Public-Safe Mode |
| :--- | :--- | :--- |
| `data/processed/matched_font_benchmark_v0.1.0/manifest.csv` | `35ae405fd084070bf4d6c235638cbda6671573389e880dcca2a63d8b66049743` | `sample_id`, `source_id`, `case_id`, `case_index`, `font_id`, `text_length`, `variant_family`, `severity_level`, `amplitude_px`, `ssim_target`, `height`, `width`, `protocol_scope` *(13 columns; image paths/hashes discarded)* |
| `data/processed/matched_font_benchmark_v0.1.0/features.csv` | `fa2d79c85d682b26545dc2c4e3ec5ece2a41d85aea7e1ea1070f7256ff8519bb` | `sample_id`, `intensity_mean`, `intensity_std`, `contrast_rms`, `ink_fraction`, `laplacian_variance`, `edge_density`, `histogram_entropy`, `noise_sigma_immerkaer` *(9 columns)* |
| `reports/matched_font_benchmark_v0_1_0/predictions.csv` | `e6c1f1ba83bbb05c18fed8a9b0dd1c87014a6f6c3667475971b2834dfedd4dd4` | `sample_id`, `font_id`, `font_group`, `ssim_target`, `prediction`, `residual`, `abs_residual` *(7 columns)* |
| `reports/matched_font_benchmark_v0_1_0/overall_metrics.csv` | `abbc28d5c73e15091edbf79cc086449d171c91af4233964529811634ed0a79fa` | `scope`, `protocol_role`, `n_rows`, `n_sources`, `n_fonts`, `mae`, `rmse`, `r2`, `r2_status` |
| `reports/matched_font_benchmark_v0_1_0/family_metrics.csv` | *(Checked in M8C)* | `variant_family`, `scope`, `protocol_role`, `n_rows`, `n_sources`, `mae`, `rmse`, `r2`, `r2_status` |
| `reports/matched_font_benchmark_v0_1_0/uncertainty.csv` | *(Checked in M8C)* | `scheme`, `scope`, `metric`, `estimate`, `ci_low`, `ci_high`, `n_groups`, `n_samples`, `note` |
| `reports/matched_font_benchmark_v0_1_0/dataset_composition.csv` | *(Checked in M8C)* | `component`, `expected`, `realised`, `matches`, `detail` |
| `reports/interference_error_analysis_v0_1_0/summary_metrics.csv` | `6d68e465887e3053e20a7c582f79e8318900e8cd14fb48005f15a54c224f1303` | `scope`, `protocol_role`, `n_rows`, `n_sources`, `mae`, `rmse`, `r2`, `large_error_proportion`, `top_5pct_source_sse_share` |
| `reports/interference_error_analysis_v0_1_0/severity_metrics.csv` | *(Checked in M9A)* | `family`, `severity_level`, `scope`, `n_rows`, `n_sources`, `target_mean`, `prediction_mean`, `mae`, `large_error_proportion` |
| `reports/interference_error_analysis_v0_1_0/condition_calibration.csv` | *(Checked in M9A)* | `scope`, `protocol_role`, `n_rows`, `n_sources`, `prediction_target_sd_ratio`, `calibration_slope`, `within_source_prediction_target_sd_ratio_median`, `within_source_calibration_slope_median` |
| `reports/interference_error_analysis_v0_1_0/error_concentration.csv` | *(Checked in M9A)* | `scope`, `n_sources`, `top_5pct_source_sse_share`, `top_10pct_source_sse_share`, `top_20pct_source_sse_share` |
| `reports/interference_error_analysis_v0_1_0/font_condition_metrics.csv` | *(Checked in M9A)* | `font_id`, `font_group`, `scope`, `n_rows`, `n_sources`, `mae`, `mae_ci_low`, `mae_ci_high` |

### 3.2 Join Validation & Integrity
- **Join Key:** `sample_id` (String).
- **Cardinality:** Exactly 1-to-1 between `manifest.csv`, `features.csv`, and `predictions.csv`.
- **Row Counts:** Exactly 6,300 joined rows (5,700 primary, 600 diagnostic).
- **Target Cross-Validation:** $| \text{ssim\_target}_{\text{manifest}} - \text{ssim\_target}_{\text{predictions}} | \le 10^{-9}$.
- **Font ID Cross-Validation:** $\text{font\_id}_{\text{manifest}} == \text{font\_id}_{\text{predictions}}$.
- **Residual Invariance:** Precomputed residuals verified against recorded formula:
  $$| \text{residual} - (\text{ssim\_target} - \text{prediction}) | \le 10^{-9}$$
  $$| \text{abs\_residual} - |\text{residual}| | \le 10^{-9}$$
- **Finite Numeric Validation:** `features.csv`, `predictions.csv`, and all report numeric columns are validated with `np.isinf(...)`. Non-finite numbers immediately raise `IntegrityError` and fail closed.

---

## 4. Image, Ground-Truth, and Model-Access Findings

| Security / Integrity Dimension | Finding | Verification Mechanism |
| :--- | :--- | :--- |
| **Benchmark Image Decoding** | **Zero Access** | In public-safe mode, `resolve_image_path()`, `load_verified_image()`, and `load_selected_image()` immediately raise `SecurityError`. Tested in `test_public_safe_prohibits_image_loading_and_resolution`. |
| **Filesystem Independence** | **Zero Dependency** | `read_core(public_safe=True)` tested in a directory with NO `images/` directory; executes and joins cleanly (`test_public_safe_works_without_images_directory`). |
| **Ground Truth Answer Key** | **Zero Access / Excluded** | `ground_truth.csv` is untracked in Git (`git ls-files` returns empty). Listed in `contracts.FORBIDDEN_FILES`. Reading it raises `IntegrityError` (`test_forbidden_files_fail_closed`). |
| **Model Estimators & Weights** | **Zero Access / Excluded** | `models/` directory is Git-ignored and untracked. No estimator is loaded or fitted. `test_app_sources_do_not_import_training_or_network_stacks` statically confirms no references to `sklearn`, `joblib`, `torch`, `tensorflow`, or `pickle`. |
| **Path Traversal Prevention** | **Fail-Closed** | `read_report_table()` validates that filenames contain no directory separators (`/`, `\`, `..`), rejecting directory escapes (`test_forbidden_files_fail_closed`). |
| **User-Visible Metadata** | **Sanitized** | Returned core DataFrame in public-safe mode drops all 4 image columns (`image_path`, `reference_path`, `image_sha256`, `reference_sha256`). Scatter plot Vega-Lite spec excludes paths and hashes (`test_scatter_payload_excludes_paths_hashes_and_forbidden_fields`). |

---

## 5. Dependency, Telemetry, Network, and Secret-Exposure Findings

### 5.1 Direct and Transitive Dependencies
The deployment profile uses exclusively pinned, minimal dependencies from [app/requirements.txt](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/app/requirements.txt):
- `streamlit==1.50.0`
- `pandas==2.3.3`
- `pillow==11.3.0`
- `altair==5.5.0`
- `pytest==8.4.2`

All transitive packages (e.g. `numpy==2.5.4`, `pyarrow==26.0.0`, `tornado==6.6`, `jsonschema==4.26.0`, `narwhals==2.27.1`) are standard production wheels. No machine learning frameworks (`scikit-learn`, `PyTorch`, `TensorFlow`), compiled accelerators, or live data scrapers are present.

### 5.2 Network Calls and Telemetry
- **Streamlit Telemetry:** Explicitly disabled in [.streamlit/config.toml](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/.streamlit/config.toml):
  ```toml
  [browser]
  gatherUsageStats = false
  ```
- **Font Network Calls:** Zero. Interface fonts are embedded offline via base64 data URIs in [app/lib/theme.py](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/app/lib/theme.py). No calls to Google Fonts, Typekit, or external CDNs exist.
- **External Network Requests:** No HTTP clients (`requests`, `urllib.request`, `httpx`) are imported or executed by the application code.

### 5.3 Secrets, Credentials, and Local Absolute Paths
- **Zero Secrets / Environment Credentials:** No API keys, database credentials, tokens, or private `.env` files exist in the repository or application code.
- **Path Portability:** `REPO_ROOT` resolves dynamically via `Path(__file__).resolve().parents[2]` in `contracts.py` and `parents[1]` in `streamlit_app.py`. No hardcoded local paths (`/Users/...`, `/home/...`) are present in the runtime code.

---

## 6. Verification Results and Coverage Limitations

### 6.1 Test Execution Results
All test suites pass cleanly:
1. **Dashboard Test Suite (`pytest app/tests`):** **85 passed in 2.95s**.
   - `test_public_safe_mode.py` (9 passed): overview/benchmark/entrypoint render without images; image resolution prohibited; functions without `images/` directory; forbidden files fail closed; non-finite metrics fail closed; default profile preserves images.
   - `test_data_contracts.py` (19 passed): core shapes, pinned hashes, schema validation, duplicate key rejection, one-to-one joins.
   - `test_images.py` (27 passed): image geometry, path traversal, pixel hash verification.
   - `test_views.py` (12 passed): AppTest view rendering, headline values match authoritative reports, forbidden import checks.
   - `test_analysis_format.py` (9 passed): numeric formatting, interval formatting, negative $R^2$ rendering.
   - `test_audit.py` (4 passed): D2 audit document and gate validation.
   - `test_public_safe_plan.py` (5 passed): D3A plan document and strategy validation.
2. **Lifecycle & Release Suite (`pytest tests/test_benchmark_lifecycle.py tests/test_release_documentation.py`):** **33 passed in 15.79s**.
3. **Static Analysis & Formatting:**
   - `ruff check app`: All checks passed.
   - `ruff format --check app`: 26 files already formatted.
   - `git diff --check`: Clean (no whitespace errors).
4. **Local Startup Smoke Test:**
   - Executed headless on port 8598 with `PUBLIC_SAFE_MODE=true`: clean startup and graceful shutdown with 0 warnings or errors.

### 6.2 Coverage Breakdown
- **Actually Exercised in Automated Tests:**
  - Full UI rendering of Overview, Benchmark, and Error Analysis views via `streamlit.testing.v1.AppTest`.
  - Zero-image rendering assertions (`len(at.get("imgs")) == 0`).
  - Strict rejection of image loading functions in public-safe mode.
  - Execution in a filesystem mock with no `images/` directory.
  - Fail-closed handling for non-finite metrics, path traversal, and unallowlisted files.
- **Supported by Static Inspection:**
  - Absence of training/network libraries in `app/` via AST/source token inspection.
  - Offline font data URI decoding and embedding.
- **Unverified in This Local Audit:**
  - Actual container deployment on Streamlit Community Cloud servers (requires owner authorization to connect GitHub repo to Cloud).

---

## 7. Scientific-Claims Review

The dashboard copy and visualizations were reviewed against `docs/RESEARCH_SUMMARY.md`, `docs/METHODOLOGY.md`, and authoritative report packages:

| Evaluation Dimension | Dashboard Claim / Copy | Research Record Compliance |
| :--- | :--- | :--- |
| **Target Variable** | "Reference-based SSIM against exact clean synthetic reference." | **Compliant.** Explicitly disclaims human judgment, CAPTCHA solvability, security, or production difficulty. |
| **Headline Metrics** | MAE: `0.0596`, RMSE: `0.0886`, $R^2$: `0.733` across 5,700 primary rows and 300 sources. | **Compliant.** Exactly matches `overall_metrics.csv` (`all_primary` scope). |
| **Interference Findings** | Thin lines ($R^2 = -0.575$) and Arc strokes ($R^2 = -0.299$) fail catastrophically; Gaussian blur ($R^2 = 0.943$) predicts well. | **Compliant.** Exactly matches `family_metrics.csv` and M8C report. |
| **M9A Characterization** | "Exploratory, descriptive, and not causal. Does not test H1, H2, or H3." | **Compliant.** Prominent caveats in UI and footer. |
| **Font Generalization** | "Ten fonts are a purposeful roster, not a sample of fonts." | **Compliant.** Avoids overclaiming font-population validity. |
| **Interactive Scoring** | "Upload scoring is intentionally unavailable." | **Compliant.** Clearly explains why arbitrary uploads cannot be scored without a clean reference. |

---

## 8. Remaining Licensing and Deployment-Authorization Questions

1. **Proprietary Font Rendered Images (M7E §11):**
   - The original image-serving dashboard remains **blocked** from public hosting due to M7E's limitation against "use in a product, a model, or a deployment."
   - **Public-Safe Resolution:** Under `PUBLIC_SAFE_MODE=true`, zero benchmark images are loaded, decoded, served, or bundled. This eliminates the image copyright redistribution risk for the public demo.
2. **Interface Typography (OFL-1.1):**
   - IBM Plex Sans, IBM Plex Mono, and Source Serif 4 are licensed under SIL Open Font License 1.1 with licenses bundled in `app/assets/OFL.txt`. Offline inlined data URIs comply with distribution terms.
3. **Owner Authorization Gate:**
   - Technical readiness is not a substitute for deployment authorization. The repository owner must formally review this audit and authorize connecting the repository to Streamlit Community Cloud.

---

## 9. Final Classification

$$\mathbf{CLASSIFICATION:\; deployment\_candidate\_ready\_for\_owner\_review}$$

### Justification:
The results-only public-safe profile (`PUBLIC_SAFE_MODE=true`) satisfies all documented data-integrity, security, privacy, and scientific constraints:
- All proprietary font CAPTCHA images are completely excluded from the public runtime.
- Ground truth text and model weights are inaccessible.
- Pinned hash contracts and finite numeric validations fail closed on invalid inputs.
- All 85 dashboard tests, 33 lifecycle tests, Ruff checks, and headless startup tests pass cleanly.

The profile is verified as a technically sound, legally defensible deployment candidate, ready for final review and deployment authorization by the repository owner.
