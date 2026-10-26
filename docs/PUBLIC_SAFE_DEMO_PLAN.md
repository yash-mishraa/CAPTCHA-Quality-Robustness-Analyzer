# D3A — Public-Safe Demo Feasibility and Asset Separation Plan

- **Milestone:** D3A
- **Repository:** `CAPTCHA-Quality-Robustness-Analyzer`
- **Specification:** Public-Safe Architecture and Asset Separation
- **Target Platform:** Streamlit Community Cloud (Public Demo)
- **Status:** Complete (Feasibility and Design Specification)
- **Feasibility Recommendation:** `public_safe_demo_feasible` (for asset-separated Results-Only architecture; direct deployment of the existing image-serving app remains `public_launch_blocked`)

---

## Executive Summary

Milestone D2 established that the existing D1 Streamlit research dashboard cannot be deployed publicly in its current form (`public_launch_blocked`), because it directly serves 6,300 benchmark PNG images rendered from ten proprietary font families whose governing audit (M7E §11) explicitly disclaims clearance for "use in a product, a model, or a deployment" (Risk B11).

This specification investigates whether a publicly hostable, copyright-compliant version of the research dashboard can be deployed on Streamlit Community Cloud by cleanly separating public-safe assets from proprietary-derived assets.

Three strategies were evaluated:
1. **Option A (Results-Only Architecture):** Present all verified statistical findings, interactive Altair charts, feature distributions, and precomputed predictions without serving or displaying any rendered benchmark images.
2. **Option B (Synthetic Open-Font Replacement Images):** Generate a new set of illustrative images using open-source (OFL-1.1) fonts degraded with the frozen corruption ladder.
3. **Option C (Hybrid Architecture):** Combine verified aggregate results with an explicitly cleared, standalone gallery of open-source pedagogical illustrations.

**Recommendation:** **Option A (Results-Only Architecture)** is selected as the safest, simplest, and most legally robust approach for a public demonstration. It allows 100% of the core scientific contributions—the primary M8C champion evaluation ($R^2 = 0.733$), the thin-line/arc-stroke breakdown, error calibration curves, and source concentration analyses—to be interactively explored without distributing a single proprietary-font-derived bitmap.

Under Option A, a public deployment is **`public_safe_demo_feasible`**.

---

## 1. Evaluation of Public-Data Strategies

### 1.1 Strategy Comparison Matrix

| Dimension | Option A: Results-Only (Recommended) | Option B: Replacement Illustrative Images | Option C: Hybrid Architecture |
| :--- | :--- | :--- | :--- |
| **Concept** | Numerical, statistical, and chart exhibits only; zero image rendering. | Generate new illustrative images from open fonts degraded with frozen corruptions. | Authoritative benchmark metrics paired with a separate open-license pedagogical gallery. |
| **Proprietary Font Images** | **None (0 PNGs)**. Complete exclusion of `data/processed/.../images/**`. | **None (0 PNGs)**. Only newly generated open-license images. | **None (0 PNGs)**. Authoritative metrics + open-font illustrations. |
| **Licensing Risk** | **Zero**. Tabular metrics and chart payloads contain no proprietary glyph geometry. | **Low**. Requires verifying the EULA of each chosen open font for image redistribution. | **Low**. Same as Option B for the illustrative portion. |
| **Scientific Fidelity** | **100% exact on benchmark metrics**. Preserves all frozen M8C and M9A values. | **Risk of confusion**. New images do not match historical benchmark observations. | **High**, provided pedagogical samples are strictly labeled as non-benchmark. |
| **Implementation Effort** | **Minimal (1–2 days)**. Suppress specimen plates; add metric summary cards. | **High (4–6 days)**. Generation script, font validation, caching, and layout testing. | **Moderate to High (4–5 days)**. Both Option A views and Option B asset management. |
| **Maintenance Burden** | **Very Low**. Self-contained static CSVs; no image asset pipeline. | **Moderate**. New asset versioning and storage overhead. | **Moderate**. Must manage two separate data contexts. |
| **Public Deployment Gate** | **`public_safe_demo_feasible`** | **`public_safe_demo_feasible`** (conditional on asset audit) | **`public_safe_demo_feasible`** (conditional on asset audit) |

---

### 1.2 Detailed Option Analysis

#### Option A: Results-Only Dashboard (Recommended)
- **Concept:** Present the research findings through interactive quantitative exhibits:
  - Overview: Primary M8C metrics ledger (MAE, RMSE, $R^2$), perturbation family breakdown, and thin-line/arc-stroke degradation profiles.
  - Explorer: Interactive Altair scatter plot (5,700 primary prediction-vs-target points), filterable feature tables (all eight handcrafted spatial features), and residual analysis cards.
  - Robustness & Error Analysis: M9A exploratory error tables, severity response curves, calibration distributions, and source concentration metrics.
- **Required Assets & Provenance:**
  - `reports/matched_font_benchmark_v0_1_0/overall_metrics.csv`
  - `reports/matched_font_benchmark_v0_1_0/family_metrics.csv`
  - `reports/matched_font_benchmark_v0_1_0/severity_metrics.csv`
  - `reports/matched_font_benchmark_v0_1_0/group_metrics.csv`
  - `reports/matched_font_benchmark_v0_1_0/predictions.csv` (joined with `features.csv`, excluding image paths)
  - `reports/interference_error_analysis_v0_1_0/m9a_summary.csv`
  - Bundled interface fonts (`app/assets/fonts/*.woff2`)
- **Licensing & Legal Basis:**
  - Tabular datasets and precomputed scalar metrics contain no copyrightable font outline data, font bytecode, or rasterized proprietary glyphs.
  - Metrics represent empirical evaluation measurements covered by the repository's open MIT license.
  - Interface fonts are SIL Open Font License 1.1 assets.
- **Scientific Strengths:** Perfectly represents the verified quantitative evidence of the $v1.0.0$ research release. No proxy data or surrogate samples.
- **Limitations:** Qualitative visual specimen inspection ("what does a severe thin-line perturbation look like?") is not shown visually; it is conveyed purely through textual and tabular descriptions of feature values (e.g., stroke width, edge density, total ink).

#### Option B: Replacement Illustrative Images
- **Concept:** Render a small set (e.g., 20–30) of synthetic illustrative text images using open-source fonts (e.g., Roboto, Open Sans, Source Serif 4) and corrupt them using the frozen corruption ladder (`src/captcha_quality/corruptions.py`).
- **Required Assets & Provenance:**
  - Upstream SIL OFL-1.1 font binaries.
  - Synthetic strings generated from an independent seed.
  - Pre-rendered 192×64 grayscale PNGs stored in `app/assets/illustrative/`.
- **Licensing & Legal Basis:**
  - SIL Open Font License 1.1 permits royalty-free creation and distribution of rendered images.
- **Scientific Strengths:** Restores the visual exhibit experience without using proprietary font assets.
- **Risks & Limitations:**
  - High risk of user confusion: visitors may incorrectly assume the reported M8C benchmark scores were measured on these open fonts rather than the 10 proprietary fonts audited in M7E.
  - Creating new images and computing model predictions for them would introduce new inference outputs into the repository, conflicting with the frozen milestone contract.

#### Option C: Hybrid Architecture
- **Concept:** Retain Option A's strict results-only views for the authoritative benchmark, and add a clearly demarcated "Methodology & Visual Exhibit" tab displaying 5 static illustrative degradation examples rendered with open-license fonts.
- **Evaluation:** While pedagogically attractive, Option C introduces unnecessary complexity and dependency management for an initial public resume demonstration. It is better treated as a prospective Phase 2 enhancement.

---

## 2. Source-by-Source Public Asset Inventory

The following inventory specifies the legal status and deployment eligibility of every asset class:

| Asset Class | Files / Paths | Origin / License | Public Web Hosting Status | Justification & Legal Basis |
| :--- | :--- | :--- | :--- | :--- |
| **Interface Typography** | `app/assets/fonts/IBMPlexSans-Variable.woff2`<br>`app/assets/fonts/IBMPlexMono-*.woff2`<br>`app/assets/fonts/SourceSerif4-Variable.woff2` | Google Fonts / IBM / Adobe;<br>SIL Open Font License 1.1 | **CLEARED FOR PUBLIC HOSTING** | Bundled WOFF2 fonts with OFL notices in `app/assets/OFL.txt`. Offline inlined data URIs. |
| **Aggregate Metric Tables** | `reports/matched_font_benchmark_v0_1_0/overall_metrics.csv`<br>`family_metrics.csv`<br>`severity_metrics.csv`<br>`group_metrics.csv`<br>`uncertainty.csv` | Empirical evaluation calculations from M8C; MIT codebase license | **CLEARED FOR PUBLIC HOSTING** | Scalar evaluation metrics; contain no proprietary code, font outlines, or personal data. |
| **M9A Error Tables** | `reports/interference_error_analysis_v0_1_0/m9a_summary.csv`<br>`family_severity_summary.csv`<br>`font_condition_metrics.csv`<br>`error_calibration_summary.csv`<br>`source_concentration_summary.csv` | Empirical error analyses from M9A; MIT codebase license | **CLEARED FOR PUBLIC HOSTING** | Analytical summaries; contain no proprietary font data. |
| **Precomputed Predictions & Features** | `reports/matched_font_benchmark_v0_1_0/predictions.csv`<br>`data/processed/.../features.csv` (excluding paths) | Precomputed M5C champion model predictions and 8 handcrafted spatial features | **CLEARED FOR PUBLIC HOSTING** | Numerical features and predictions; no raw text, no image paths, no hashes. |
| **Dataset Metadata Tables** | `data/processed/.../manifest.csv`<br>`data/processed/.../sources.csv` | M8C dataset indexing tables | **CLEARED FOR PUBLIC HOSTING** (sanitized) | Manifest metadata may be used for filtering, but image path fields must not be resolved to disk. |
| **Benchmark Rendered Images** | `data/processed/matched_font_benchmark_v0.1.0/images/**`<br>(6,300 PNGs) | Rendered from 10 proprietary fonts under M7E audit | **STRICTLY EXCLUDED** | Blocked by M7E §11 ("does not clear any face for use in a product, a model, or a deployment"). |
| **Benchmark Font Files** | System font binaries (Arial, Georgia, etc.) | Proprietary OS EULAs | **STRICTLY EXCLUDED** | Redistribution forbidden; absent from repository. |
| **Ground Truth Answer Key** | `data/processed/.../ground_truth.csv` | Internal generator answer key | **STRICTLY EXCLUDED** | Git-ignored, untracked, forbidden by contract. |
| **Trained Model Checkpoint** | `models/robustness_benchmark_v0_1_0/model.joblib` | M5C champion trained model | **STRICTLY EXCLUDED** | Local-only, untracked; precomputed predictions suffice. |

---

## 3. Recommended Architecture: Results-Only Public Dashboard

### 3.1 Design Principles
1. **Zero Proprietary Bitmaps:** The public application must not load, read, or serve any file from `data/processed/matched_font_benchmark_v0.1.0/images/`.
2. **Zero Model Inference:** The app runs entirely on static precomputed CSV tables cached with `@st.cache_data`.
3. **Scientific Completeness:** The app communicates all headline findings:
   - Overall champion performance: MAE $0.0596$, RMSE $0.0886$, $R^2 = 0.733$.
   - The primary research discovery: Gaussian blur ($R^2 = 0.943$) and Gaussian noise ($R^2 = 0.736$) are predicted accurately, whereas high-frequency structural perturbations—thin lines ($R^2 = -0.575$) and arc strokes ($R^2 = -0.299$)—cause catastrophic failure.
   - Severe thin lines ($s4$): MAE $0.2079$, large-error rate $55.33\%$.
   - Calibration and error concentration: The top 5% of source images account for over 50% of sum of squared errors in blur and noise, but error is widely distributed in interference perturbations.
4. **Editorial Aesthetic Integrity:** Retains the warm-paper palette, IBM Plex Sans / Mono typography, Source Serif 4 headings, hairline dividers, and custom Altair visualizations.

### 3.2 View-by-View Public Layout

```
+--------------------------------------------------------------------------------+
|  RESEARCH OVERVIEW (/overview)                                                 |
|  - Title, Research Question, and Executive Summary                             |
|  - Headline Metric Ledger: MAE (0.0596), RMSE (0.0886), R2 (0.733)             |
|  - Perturbation Family Performance Table with 95% Confidence Intervals        |
|  - The Structural Failure Mode Exhibit (Thin lines & Arc strokes callout card)  |
|  - Interactive Altair Family Bar Chart (MAE and R2 by condition)               |
+--------------------------------------------------------------------------------+
|  BENCHMARK EXPLORER (/benchmark)                                               |
|  - Filter Bar: Font (10), Case (30), Family (5), Severity (1-4)               |
|  - Selected Record Metric Card (Target SSIM, Prediction, Residual)             |
|  - Eight Handcrafted Spatial Features Ledger (Edge density, stroke width, etc.)|
|  - Full Benchmark Scatter Plot (5,700 primary predictions vs target SSIM)      |
|  - Scope & Diagnostic Information Card (no image plates rendered)              |
+--------------------------------------------------------------------------------+
|  ROBUSTNESS & ERROR ANALYSIS (/error-analysis)                                 |
|  - M9A Exploratory Analysis Notice (H1-H3 preregistered, unexecuted)            |
|  - Severity Ladder Curves (Altair line chart across s1-s4)                     |
|  - Error Calibration Histogram & Table                                         |
|  - Source Error Concentration Ledger (Top 5% SSE Share)                        |
|  - Font-by-Font Performance Breakdown                                          |
+--------------------------------------------------------------------------------+
```

---

## 4. Exact Files Cleared for Public Display

When running in public-safe mode, the application will read **only** the following tracked files:

```text
.streamlit/config.toml
app/requirements.txt
app/streamlit_app.py
app/content.py
app/lib/theme.py
app/lib/contracts.py
app/lib/data.py
app/lib/charts.py
app/lib/format.py
app/lib/analysis.py
app/lib/ui.py
app/views/overview.py
app/views/benchmark.py
app/views/error_analysis.py
app/assets/OFL.txt
app/assets/fonts/IBMPlexSans-Variable.woff2
app/assets/fonts/IBMPlexMono-400.woff2
app/assets/fonts/IBMPlexMono-500.woff2
app/assets/fonts/IBMPlexMono-600.woff2
app/assets/fonts/SourceSerif4-Variable.woff2
reports/matched_font_benchmark_v0_1_0/overall_metrics.csv
reports/matched_font_benchmark_v0_1_0/family_metrics.csv
reports/matched_font_benchmark_v0_1_0/severity_metrics.csv
reports/matched_font_benchmark_v0_1_0/group_metrics.csv
reports/matched_font_benchmark_v0_1_0/predictions.csv
reports/matched_font_benchmark_v0_1_0/uncertainty.csv
reports/interference_error_analysis_v0_1_0/m9a_summary.csv
reports/interference_error_analysis_v0_1_0/family_severity_summary.csv
reports/interference_error_analysis_v0_1_0/font_condition_metrics.csv
reports/interference_error_analysis_v0_1_0/error_calibration_summary.csv
reports/interference_error_analysis_v0_1_0/source_concentration_summary.csv
data/processed/matched_font_benchmark_v0.1.0/features.csv
data/processed/matched_font_benchmark_v0.1.0/manifest.csv (tabular join only)
```

### Strictly Excluded Assets
- `data/processed/matched_font_benchmark_v0.1.0/images/**` (all 6,300 PNGs)
- `data/processed/matched_font_benchmark_v0.1.0/ground_truth.csv` (raw text answers)
- `models/robustness_benchmark_v0_1_0/model.joblib` (serialized model binary)
- Any local system font binaries (`.ttf`, `.otf`, `.ttc`)

---

## 5. Required Application Changes for Public-Safe Mode

To implement Option A without altering the local review capabilities of D1, the application will introduce an explicit **deployment profile switch**:

1. **Environment Flag / Configuration:**
   - Add a configuration toggle in `app/lib/contracts.py` (e.g., `PUBLIC_SAFE_MODE = os.getenv("PUBLIC_SAFE_MODE", "false").lower() == "true"` or via `.streamlit/config.toml`).
   - When `PUBLIC_SAFE_MODE=True`:
     - Image loading functions in `app/lib/data.py` (`load_verified_image`, `resolve_image_path`) are disabled.
     - `app/views/overview.py` replaces the specimen image comparison plate with an **Editorial Metric Ledger Card** explaining the illustrative sample values.
     - `app/views/benchmark.py` replaces the specimen plate with a **Feature & Metric Ledger Card** showing the 8 spatial features and the prediction residual.
     - Sidebar displays a prominent badge: `Mode: Public Safe (Results & Analytics Only; Specimen Images Excluded)`.
   - When `PUBLIC_SAFE_MODE=False` (default for local runs):
     - The dashboard operates exactly as built in Milestone D1, loading and verifying local PNG images.

2. **Integrity Guarantees:**
   - In public-safe mode, any attempt by the application to access the `images/` directory will fail closed immediately.
   - Pinned SHA-256 validation for all ingested CSVs remains active.

---

## 6. Validation and Acceptance Criteria

Before any public deployment under Option A can be authorized:

1. **Code & Contract Checks:**
   - `PUBLIC_SAFE_MODE=True` must pass all existing tests adapted for both modes.
   - Zero calls to `st.image()` or `PIL.Image.open()` on benchmark images when running in public-safe mode.
   - `git diff --check` must be clean.
   - `ruff check app` and `ruff format --check app` must pass cleanly.
2. **Data & Privacy Verification:**
   - Automated tests must prove that `data/processed/.../images/` is never accessed in public-safe mode.
   - Scatter plot Vega-Lite specs must be verified to contain zero image paths or file hashes.
   - `ground_truth.csv` must remain untracked and blocked.
3. **Visual & UX Acceptance:**
   - In public-safe mode, layout must remain balanced and visually engaging without empty image placeholders or broken container borders.
   - The editorial aesthetic (Source Serif 4, IBM Plex fonts, rust/teal accents) must be preserved.

---

## 7. Explicit Licensing Assumptions and Unresolved Blockers

### 7.1 Licensing Assumptions (Option A)
1. **Tabular Research Data:** Precomputed scalar metrics, handcrafted statistical features, and prediction residuals derived from synthetic experiments do not constitute copyright-infringing derivatives of proprietary font software when distributed without glyph outlines or pixel bitmaps.
2. **Interface Typography:** The bundled interface fonts (IBM Plex Sans, IBM Plex Mono, Source Serif 4) are distributed strictly in compliance with the SIL Open Font License 1.1 with all copyright and license notices preserved.
3. **Attribution:** Font family names (Arial, Georgia, etc.) are cited strictly as scientific metadata to identify experimental conditions, which constitutes fair nominative use.

### 7.2 Status of Original Blockers
- **Blocker B11 (Proprietary Rendered Image Deployment):**
  - For the full image-serving dashboard: **REMAINS BLOCKED**. M7E §11 has not been cleared by legal counsel.
  - For the proposed Option A Results-Only dashboard: **RESOLVED BY ASSET EXCLUSION**. Since zero proprietary font images are loaded or served, Blocker B11 does not apply to Option A.

---

## 8. Final Recommendation

- **Recommendation:** `public_safe_demo_feasible`

$$\mathbf{RECOMMENDATION:\; public\_safe\_demo\_feasible}$$

### Summary of Recommendation:
- A public demonstration of the research project is **feasible and legally defensible** under **Option A (Results-Only Architecture)**.
- Public deployment of the original D1 image-serving dashboard remains **`public_launch_blocked`**.
- Proceeding with Option A will allow the repository owner to deploy an interactive research dashboard to Streamlit Community Cloud that showcases the full analytical rigour, ML findings, and visual design of the project without incurring any proprietary font licensing liabilities.
