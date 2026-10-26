# Local research dashboard

A read-only Streamlit presentation layer over the frozen `v1.0.0` M8C
benchmark and M9A exploratory error analysis.

**Status:** local/private review build. Public deployment, a public URL, and
hosted distribution of the benchmark images are blocked pending an explicit
public-display and licensing decision. This app does not deploy itself.

## What it shows

- **Research overview** — the research question, the primary M8C result, and
  the observed thin-line / arc-stroke failure, with an illustrative image pair.
- **Benchmark explorer** — one manifest-backed record at a time: clean
  reference, degraded image, precomputed prediction, residual, and the eight
  frozen image features.
- **Robustness & error analysis** — descriptive M9A views of family, severity,
  calibration, source concentration, and font-level error. M9A is exploratory;
  H1, H2, and H3 remain untested.

## Read-only guarantees

- Reads only allowlisted frozen CSVs and tracked PNGs.
- Verifies the pinned SHA-256 hashes for the manifest, features, predictions,
  overall metrics, and M9A summary.
- Rejects duplicate or missing keys, invalid joins, missing columns, unexpected
  scopes, malformed paths, and image hash mismatches.
- Selected images are resolved inside the dataset root and verified against
  the manifest before display.
- Never reads `ground_truth.csv` or another untracked drilldown.
- Never loads, fits, or runs the frozen model. Predictions are precomputed.
- Generates no data, metrics, images, or benchmark results.

Reference-based SSIM requires the known clean synthetic reference. Arbitrary
upload scoring is therefore unavailable, and the ten-font roster is not a
sample of fonts.

## Profiles: Local Full vs. Results-Only Public-Safe

The application supports two operating profiles selected by the `PUBLIC_SAFE_MODE` environment variable:

### 1. Default Profile (Local Full Review)
- **Activation:** default (`PUBLIC_SAFE_MODE=false`).
- **Features:** Displays both quantitative metrics and local specimen CAPTCHA image pairs (clean reference and degraded variant).
- **Security & Integrity:** Resolves local relative PNG paths, verifies decoded pixel SHA-256 hashes against the manifest, and validates schemas.
- **Licensing Constraint:** Local/private execution only. Public deployment and hosted image distribution remain blocked under M7E §11.

### 2. Results-Only Profile (Public-Safe)
- **Activation:** `PUBLIC_SAFE_MODE=true`.
- **Command:**
  ```bash
  PUBLIC_SAFE_MODE=true .venv-dashboard/bin/streamlit run app/streamlit_app.py
  ```
- **Features:** Exposes 100% of the verified quantitative research findings (headline $R^2 = 0.733$, family breakdown, thin-line/arc-stroke degradation patterns, interactive 5,700-point Altair scatter plot, handcrafted feature tables, error calibration, and source concentration) using tabular data only.
- **Excluded Assets & Operations:**
  - **Zero Benchmark Images:** Never loads, reads, decodes, hashes, or serves files from `data/processed/.../images/**`. Specimen plates are replaced with clean editorial record and feature ledger cards.
  - **Zero Ground Truth:** `ground_truth.csv` is untracked, forbidden, and never accessed.
  - **Zero Model Inference:** Does not load `model.joblib` or execute training, fitting, or benchmark replay. Runs exclusively on precomputed predictions.
  - **Zero Proprietary Fonts:** No font binaries or local font paths are accessed or bundled. Interface fonts are open SIL OFL-1.1 assets.
  - **Zero External Dependencies:** Public-safe mode does not require the local `images/` directory, `ground_truth.csv`, or model checkpoint to be present on disk.
- **Artifacts Read in Public-Safe Mode:**
  - `data/processed/matched_font_benchmark_v0.1.0/manifest.csv` (tabular join only; image paths/hashes stripped)
  - `data/processed/matched_font_benchmark_v0.1.0/features.csv`
  - `reports/matched_font_benchmark_v0_1_0/predictions.csv`
  - `reports/matched_font_benchmark_v0_1_0/overall_metrics.csv`
  - `reports/matched_font_benchmark_v0_1_0/family_metrics.csv`
  - `reports/matched_font_benchmark_v0_1_0/uncertainty.csv`
  - `reports/matched_font_benchmark_v0_1_0/dataset_composition.csv`
  - `reports/interference_error_analysis_v0_1_0/summary_metrics.csv`
  - `reports/interference_error_analysis_v0_1_0/severity_metrics.csv`
  - `reports/interference_error_analysis_v0_1_0/condition_calibration.csv`
  - `reports/interference_error_analysis_v0_1_0/error_concentration.csv`
  - `reports/interference_error_analysis_v0_1_0/font_condition_metrics.csv`
- **Scientific Nature:** An explanatory research interface over precomputed scientific benchmark data, not a deployed model or live CAPTCHA solver. SSIM measures synthetic pixel similarity, not human solvability or security.

## Run locally

```bash
python3 -m venv .venv-dashboard
.venv-dashboard/bin/python -m pip install -r app/requirements.txt

# Run default profile (with local images):
.venv-dashboard/bin/streamlit run app/streamlit_app.py

# Run public-safe profile (results-only, zero images):
PUBLIC_SAFE_MODE=true .venv-dashboard/bin/streamlit run app/streamlit_app.py
```

The three routes are `/overview` (the bare root redirects here), `/benchmark`,
and `/error-analysis`. The app is designed for a local browser session. It
makes no runtime font CDN request: the open-license interface fonts are bundled
under `app/assets/fonts/` and embedded as data URIs.

## Tests

```bash
.venv-dashboard/bin/python -m pytest app/tests
```

Focused tests cover schema and join validation, duplicate and missing-key
rejection, path traversal, manifest-backed image hashes, empty optional data,
known metric rendering, and the absence of startup training or generation.

## Licensing

Dashboard code is covered by the repository's MIT license. Bundled interface
fonts are SIL Open Font License 1.1; see `app/assets/OFL.txt` and the license
texts beside the WOFF2 files.

Rendered benchmark images are displayed under the M7E audit conditions for the
intended evaluation workflow. Font binaries from the benchmark roster are not
distributed. The ten typefaces are proprietary; their names identify rendered
material only. This is not a legal opinion. See the
[M7E licensing audit](https://github.com/yash-mishraa/CAPTCHA-Quality-Robustness-Analyzer/blob/v1.0.0/reports/font_roster_audit_v0_2_0/report.md).
