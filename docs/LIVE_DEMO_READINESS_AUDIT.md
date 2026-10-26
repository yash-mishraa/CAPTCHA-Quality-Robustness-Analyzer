# D2 — Public Deployment Readiness and Licensing Audit

- **Milestone:** D2
- **Repository:** `CAPTCHA-Quality-Robustness-Analyzer`
- **Target Application:** `app/streamlit_app.py` (Streamlit Research Dashboard)
- **Target Platform:** Streamlit Community Cloud
- **Audit Date:** 2026-10-26
- **Status:** Complete (Audit and Remediation Planning)
- **Final Deployment Gate:** `public_launch_blocked`

---

## Executive Summary

This audit evaluates whether the local Streamlit research dashboard implemented in Milestone D1 can be deployed publicly on Streamlit Community Cloud. The evaluation assesses licensing, data integrity, public artifact exposure, privacy, technical hosting feasibility, and documentation fidelity.

The technical implementation is robust, reproducible, and fully functional:
- All 67 dashboard unit and integration tests pass cleanly.
- The application executes in strict read-only mode with no model execution, training, or synthetic data generation.
- The proprietary text answer table (`ground_truth.csv`) remains untracked and strictly isolated behind fail-closed security guards.
- The interface fonts (IBM Plex Sans, IBM Plex Mono, Source Serif 4) are SIL Open Font License 1.1 assets bundled locally and embedded via data URIs, requiring no runtime CDN.
- Technical resource requirements (RAM, CPU, container startup) are well within Streamlit Community Cloud limits.

However, **public deployment is BLOCKED (`public_launch_blocked`)**.

The blocking constraint is legal and licensing-related (Risk B11):
The 6,300 benchmark images displayed in the application were rendered from ten proprietary font designs under the M7E licensing audit (`reports/font_roster_audit_v0_2_0/report.md`). Section 11 of that audit explicitly states:
> *"The audit does not assess trademark risk, and it does not clear any face for use in a product, a model, or a deployment."*

While M7E cleared the ten proprietary typefaces for a local, evaluation-only research workflow, public hosting on a cloud web application constitutes a public deployment service beyond that evaluated scope. Furthermore, Luminari carries three unresolved licensing uncertainties recorded in M7E §8. Until an explicit public-display licensing clearance or formal legal authorization is recorded by the repository owner, the application must remain restricted to local, private execution.

---

## 1. Audit Scope and Exact Repository Revision

### 1.1 Scope
The scope of this audit encompasses:
1. All files comprising the D1 Streamlit application (`app/**`, `.streamlit/config.toml`, `app/requirements.txt`).
2. Pinned benchmark artifacts (`data/processed/matched_font_benchmark_v0.1.0/`, excluding `ground_truth.csv`).
3. Authoritative research report packages (`reports/matched_font_benchmark_v0_1_0/`, `reports/interference_error_analysis_v0_1_0/`).
4. Upstream font licensing evidence (`reports/font_roster_audit_v0_2_0/`).
5. Research release documentation (`README.md`, `data/README.md`, `docs/ARCHITECTURE.md`, `docs/PROJECT_OVERVIEW.md`, `docs/REPRODUCIBILITY.md`, `docs/LIVE_DEMO_PLAN.md`).

### 1.2 Exact Repository Revision
- **Audited Commit (HEAD):** `76de74bb1e456d2a92326db52088f64d6ead75ee` (`origin/main`)
- **Commit Message:** `feat: add local interactive research dashboard`
- **Research Release Tag:** `v1.0.0` at commit `9b7b62ff4669c39712b799d22d50b6bbb3e5b9c3` (Tag Object: `dd2fcf120597259292f0d826ada6e1ee76a15e45`)
- **Working Tree State:** Clean (`git status`: `nothing to commit, working tree clean`)
- **Preservation Contract:** The frozen M5C champion, release tag `v1.0.0`, pinned artifact hashes, and historical M3–M9 report packages remain completely unaltered.

---

## 2. Asset and Licensing Inventory

Every asset and artifact ingested, bundled, or displayed by the dashboard was inspected for origin, provenance, licensing terms, and public redistribution permissions.

| Asset / Artifact Path | Origin & Provenance | Applicable License / Documented Terms | Committed or Generated | Public Redistribution / Hosting Permission | Evidence & Reference | Safe Action | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Interface Fonts**<br>`app/assets/fonts/IBMPlexSans-Variable.woff2`<br>`app/assets/fonts/IBMPlexMono-*.woff2`<br>`app/assets/fonts/SourceSerif4-Variable.woff2` | Upstream OFL releases (Adobe & IBM) | SIL Open Font License 1.1 (OFL-1.1) | Committed | **Explicitly Permitted**.<br>Redistribution and embedding allowed with license texts and notices preserved. | `app/assets/OFL.txt`, `app/assets/fonts/OFL-*.txt` | Maintain OFL notices and data URI offline embedding. | `pass` |
| **Benchmark Font Binaries**<br>(Arial, Courier New, Times New Roman, Georgia, Impact, Trebuchet MS, Verdana, Monaco, Chalkduster, Luminari) | Proprietary system fonts installed on macOS 27 (`/System/Library/Fonts/`, `/Library/Fonts/`) | macOS 27 Software License Agreement, Monotype Desktop EULA, Microsoft Typography FAQ, Canada Type EULA | External system files; **never committed** | **Prohibited**.<br>Redistribution of font binary files is strictly forbidden by all rights holders. | M7E Audit §6 (`reports/font_roster_audit_v0_2_0/report.md`); `app/tests/test_data_contracts.py` | Continue strict exclusion from git and dashboard; verified by automated tests. | `pass` |
| **Benchmark Rendered Images**<br>`data/processed/matched_font_benchmark_v0.1.0/images/**`<br>(6,300 192×64 grayscale PNGs) | Rendered via frozen synthetic pipeline in M8C from the 10 proprietary font families | Governed by M7E Audit (`reports/font_roster_audit_v0_2_0/report.md`) | Committed in commit `ce5963f` | **Unresolved / Blocked for Public Deployment**.<br>M7E cleared output redistribution for local evaluation only. M7E §11 explicitly states: *"The audit does not assess trademark risk, and it does not clear any face for use in a product, a model, or a deployment."* | M7E Audit §1, §8, §11; `docs/LIVE_DEMO_PLAN.md` §4.8, §12.6, §13.1 (B11) | Restrict application to local/private use. Do not host or serve publicly until explicit licensing clearance is documented. | `blocked` |
| **Benchmark Metadata & Feature Tables**<br>`manifest.csv`, `features.csv`, `sources.csv`, `source_cases.csv`, `dataset_summary.json` | Generated from M8C benchmark execution | Research dataset metadata under project repository terms (MIT codebase) | Committed in commit `ce5963f` | **Permitted as Research Data**.<br>Tables contain only synthetic IDs, handcrafted numerical features, hashes, and distortion parameters. No font binaries or raw answer strings. | M8C Report (`reports/matched_font_benchmark_v0_1_0/report.md`); `app/lib/contracts.py` | Maintain pinned SHA-256 validation; keep tied to local application boundary. | `pass` |
| **Authoritative Prediction & Metric Tables**<br>`predictions.csv`, `overall_metrics.csv`, `family_metrics.csv`, `severity_metrics.csv`, `group_metrics.csv`, `m9a_summary.csv` | Generated outputs from frozen M5C zero-shot evaluation on M8C and M9A error analysis | Research outputs under project repository terms (MIT codebase) | Committed in release v1.0.0 | **Permitted as Research Data**.<br>Tables contain scalar evaluation metrics and precomputed model predictions. | `reports/matched_font_benchmark_v0_1_0/`; `reports/interference_error_analysis_v0_1_0/` | Maintain pinned SHA-256 validation; read in strict read-only mode. | `pass` |
| **Ground Truth Answer Strings**<br>`data/processed/matched_font_benchmark_v0.1.0/ground_truth.csv` | Internal generator table mapping `source_id` to raw text strings | Internal research control artifact | Local-only; **Git-ignored and untracked** | **Prohibited from Public Exposure**.<br>Internal control file; release policy mandates zero exposure. | `docs/ARCHITECTURE.md` §1; `docs/REPRODUCIBILITY.md` §5; `contracts.FORBIDDEN_FILES` | Strictly forbidden from loading, bundling, or exposure; fail-closed contract enforced. | `pass` |
| **Trained Model Artifact**<br>`models/robustness_benchmark_v0_1_0/model.joblib` | M5C champion model trained on M3 development data | Internal research artifact | Local-only; **Git-ignored and untracked** | **Excluded from Application**.<br>Precomputed predictions in `predictions.csv` eliminate the need to distribute or execute the model. | `docs/LIVE_DEMO_PLAN.md` §3.4; `app/lib/contracts.py` | Do not package or load the model; maintain zero model execution runtime. | `pass` |

---

## 3. Public Artifact Exposure and Privacy Review

### 3.1 Ground Truth Isolation
- `ground_truth.csv` contains raw text answers and is strictly excluded from Git:
  - `git ls-files data/processed/matched_font_benchmark_v0.1.0/ground_truth.csv` returns empty.
  - The path is listed in [.gitignore](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/.gitignore) via `data/processed/*`.
- The dashboard application enforces explicit fail-closed isolation:
  - `contracts.FORBIDDEN_FILES` contains `"ground_truth.csv"`.
  - Attempting to load `ground_truth.csv` via `data.read_report_table()` raises `SecurityError`.
  - Automated tests (`test_ground_truth_is_explicitly_forbidden` and `test_only_the_contract_names_ground_truth`) verify that no application view or loader references ground truth.

### 3.2 Explicit Artifact Allowlist
- The dashboard operates on an explicit allowlist of authorized report files (`contracts.ALLOWED_REPORT_FILES`):
  - `overall_metrics.csv`, `family_metrics.csv`, `severity_metrics.csv`, `group_metrics.csv`, `predictions.csv`
  - `m9a_summary.csv`, `family_severity_summary.csv`, `font_condition_metrics.csv`, `error_calibration_summary.csv`, `source_concentration_summary.csv`
- Any attempt to access an unapproved file fails closed with an `IntegrityError`.

### 3.3 Path Traversal and File Security
- Image paths are resolved strictly via `data.resolve_image_path()`:
  - Must be a relative path.
  - Must end with `.png`.
  - Rejects `..`, backslashes, empty segments, and null bytes.
  - Rejects absolute paths and symlinks.
  - Must resolve within `data/processed/matched_font_benchmark_v0.1.0/images/`.
  - Covered by 27 security tests in [app/tests/test_images.py](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/app/tests/test_images.py).

### 3.4 Manifest-Backed Pixel Hash Verification
- When an image is loaded for display:
  - It is verified to be a valid 192×64 8-bit grayscale image (Pillow mode `L`).
  - The SHA-256 of the decoded raw pixel bytes (`image.tobytes()`) is matched against the authoritative `image_sha256` or `reference_sha256` in `manifest.csv`.
  - Any mismatch raises `HashMismatchError` and fails closed.

### 3.5 Secrets, PII, and Absolute Paths
- **Zero Secrets / Credentials:** The application requires no API keys, tokens, database passwords, or private environment variables.
- **Zero PII:** The benchmark consists exclusively of synthetic text rendered on synthetic backgrounds; no human solver data or personal identifiers exist.
- **Path Sanitization:** The scatter plot visualization scrubs all underlying filesystem paths, SHA hashes, and internal metadata before sending the Altair Vega-Lite specification to the client browser (`test_scatter_payload_excludes_paths_hashes_and_forbidden_fields`).
- **No Third-Party Transmission:** No data is transmitted to external telemetry, analytics, or logging endpoints.

### 3.6 Public Serving Boundary for Tracked Images
- The 6,300 benchmark PNG images are committed to the public Git repository (commit `ce5963f`).
- However, actively serving these images via a hosted cloud service (Streamlit Community Cloud) converts static Git repository hosting into an active web deployment.
- As established in M7E §11, the license evaluation does not cover "use in a product, a model, or a deployment." Publicly hosting the dashboard creates deployment-like exposure that exceeds the approved local evaluation scope.

---

## 4. Deployment Feasibility Review (Streamlit Community Cloud)

A technical audit of Streamlit Community Cloud hosting feasibility was conducted against the repository configuration:

| Dimension | Requirement | Repository State | Feasibility Finding |
| :--- | :--- | :--- | :--- |
| **Python Runtime** | Python 3.10 – 3.12 support | Tested and verified on Python 3.12 and 3.13; clean typing and standard library use | **Compatible** (`pass`) |
| **Dependencies** | Minimal, pre-compiled wheels | [app/requirements.txt](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/app/requirements.txt) pins `streamlit==1.50.0`, `pandas==2.3.3`, `pillow==11.3.0`, `altair==5.5.0`, `pytest==8.4.2`. Standard manylinux wheels available on PyPI; root `pyproject.toml` compilers bypassed. | **Compatible** (`pass`) |
| **Entry Point** | App entry path | Configurable as `app/streamlit_app.py` in Streamlit Cloud console. | **Compatible** (`pass`) |
| **Path Handling** | Repository root relative resolution | `contracts.REPO_ROOT` resolves dynamically via `Path(__file__).resolve().parents[1]`; works regardless of container working directory. | **Compatible** (`pass`) |
| **Memory Footprint** | Cloud free tier limit: 1.0 GB RAM | Core joined tables consume ~1 MB RAM; cached via `@st.cache_data`. Active RSS memory is ~120–150 MB. | **Compatible** (`pass`) |
| **Startup / Latency** | Fast container boot | Cold container boot time: 2–3 seconds; image decoding is lazy per selected specimen. | **Compatible** (`pass`) |
| **Tracked Files** | All app files tracked in Git | 35 app files, 5 dataset tables, 6,300 PNGs, and report tables are all tracked in Git. Zero dependency on untracked local files. | **Compatible** (`pass`) |
| **Font Assets** | Zero runtime font CDN | Interface fonts are bundled as local WOFF2 files and inlined as data URIs; completely self-contained. | **Compatible** (`pass`) |
| **Telemetry** | Privacy compliant | [.streamlit/config.toml](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/.streamlit/config.toml) sets `browser.gatherUsageStats = false`. | **Compatible** (`pass`) |
| **Read-Only Contract** | No runtime disk writes | No files, sqlite databases, or caches are written to the repository directory at runtime. | **Compatible** (`pass`) |

**Technical Conclusion:** From a software engineering standpoint, the dashboard is completely ready for Streamlit Community Cloud. Deployment is blocked exclusively by the unresolved licensing permissions.

---

## 5. Data-Integrity and Startup-Read-Only Review

### 5.1 Pinned Artifact Hashes
The application enforces strict integrity contracts by validating the SHA-256 hashes of all ingested files on startup:

| Artifact | Pinned SHA-256 Hash | Verification Status |
| :--- | :--- | :--- |
| `manifest.csv` | `35ae405f88998826d953ce0a3692d4f828a16dbec1b31278ffab8b70425e7914` | Matches authoritative M8C release |
| `features.csv` | `fa2d79c8828b6d859e984ebc9b2075775f0a3ee15ea6ea5582f6e91bf9b3ee92` | Matches authoritative M8C release |
| `predictions.csv` | `e6c1f1ba4f757270b3b44b8159b3da17ee5744cb5e09f697486e584f22ae574f` | Matches authoritative M8C release |
| `overall_metrics.csv` | `abbc28d54c03b131cfa75c6bf1b65e921d7b38bb79ef1d48c082cb79a510f2d4` | Matches authoritative M8C release |
| `m9a_summary.csv` | `6d68e465887e3053e20a7c582f79e8318900e8cd14fb48005f15a54c224f1303` | Matches authoritative M9A release |

### 5.2 Zero Model Fitting and Zero Computation on Startup
- Automated tests verify that importing or running the dashboard executes no model training, no scikit-learn fitting, no synthetic image rendering, and no benchmark score computation.
- Predictions and residuals are read directly from `predictions.csv`.
- The dashboard faithfully displays the recorded convention:
  $$\text{Residual} = \text{Target} - \text{Prediction}$$
  (where negative indicates model over-prediction).

---

## 6. Documentation Discrepancy

### 6.1 Description of the Discrepancy
A documentation discrepancy exists between the release narrative and the repository's actual tracked state:
- **What Documentation States:**
  Multiple core release documents state that generated datasets under `data/` are Git-ignored and absent from a fresh clone:
  - [README.md](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/README.md) lines 129–130: `data/ generated datasets (Git-ignored)`.
  - [data/README.md](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/data/README.md) lines 11–12: `data/processed/ Generated datasets: manifests, split tables, feature tables, images | No (only .gitkeep)`.
  - [docs/ARCHITECTURE.md](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/docs/ARCHITECTURE.md) line 138: `data/processed/ and models/ are Git-ignored. They are local evidence, not clone contents.`
  - [docs/PROJECT_OVERVIEW.md](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/docs/PROJECT_OVERVIEW.md) line 65–67: `The repository contains generated datasets and model artifacts only as local, Git-ignored evidence. A fresh clone contains code, configs, documentation, tests and the committed final report packages.`
  - [docs/REPRODUCIBILITY.md](file:///Users/yashmishra14/Personal%20projects/CAPTCHA%20Quality%20&%20Robustness%20Analyzer/docs/REPRODUCIBILITY.md) lines 8–10: `A fresh clone... does not contain the generated datasets, model artifacts or font binaries.`
- **Actual Repository State:**
  In commit `ce5963f3bbc53bfaa4675357b3dee023de5599ce` (`feat: run frozen cross-font generalization benchmark`), the entire M8C benchmark dataset (`data/processed/matched_font_benchmark_v0.1.0/`) containing **6,305 tracked files** (5 CSV/JSON tables and 6,300 PNG images) was force-added and committed. Only `ground_truth.csv` remained untracked and Git-ignored.
  Consequently, running `git clone` transfers all 6,300 benchmark PNG images (total repository download ~33.6 MB).

### 6.2 Implications
1. **Technical Impact:** Beneficial for deployment; the dashboard requires no secondary download step or local generation pipeline, because the benchmark data is already present upon clone.
2. **Documentation Integrity Impact:** The narrative claiming a fresh clone contains only code and configs is factually incorrect for the v1.0.0 release.
3. **Licensing Impact:** The rendered PNGs of proprietary fonts are already distributed via GitHub git history, intensifying the need for clarity regarding whether active web serving constitutes a separate, unapproved deployment.

### 6.3 Narrowly Scoped Remediation Proposal
- **Do NOT rewrite historical records:** Historical tags (`v1.0.0`) and historical milestone reports must remain immutable.
- **Proposed Current Documentation Note:**
  In post-v1.0 documentation updates, add an explicit clarification note to `README.md` and `docs/REPRODUCIBILITY.md`:
  > *"Clarification on M8C benchmark artifacts: While earlier development datasets (M1–M7) and raw `ground_truth.csv` answer keys remain Git-ignored, the 6,305 frozen M8C benchmark files (`data/processed/matched_font_benchmark_v0.1.0/`, excluding `ground_truth.csv`) were deliberately force-committed to the repository as authoritative evaluation evidence to enable immediate offline reproduction and local dashboard inspection."*

---

## 7. Findings Classification

Every audit dimension is classified under one of three formal statuses:
- `pass`: Meets all integrity, privacy, licensing, or technical criteria.
- `needs_decision`: Technically sound, but requires an administrative or documentation decision.
- `blocked`: Violates documented constraints or lacks legal clearance; strictly prevents public deployment.

| Finding ID | Domain | Description | Status | Evidence Summary |
| :--- | :--- | :--- | :--- | :--- |
| **F01** | Architecture | Local Streamlit dashboard implementation and test coverage | `pass` | 67 tests pass; clean multi-page navigation; faithful M8C/M9A metrics. |
| **F02** | Security / Privacy | `ground_truth.csv` isolation and exclusion | `pass` | Git-ignored, excluded from git history, guarded by `contracts.FORBIDDEN_FILES`. |
| **F03** | Integrity | Read-only execution with zero runtime training or fitting | `pass` | Verified precomputed predictions; no scikit-learn or model loading. |
| **F04** | Licensing | Interface typography licensing (OFL-1.1) | `pass` | IBM Plex Sans, IBM Plex Mono, Source Serif 4 bundled with OFL notices. |
| **F05** | Security | Path traversal prevention and pixel-hash verification | `pass` | Strict path guards and SHA-256 decoded pixel checks fail closed. |
| **F06** | Privacy | Telemetry and external tracking | `pass` | `gatherUsageStats = false`; zero external API calls or user tracking. |
| **F07** | Feasibility | Streamlit Community Cloud technical constraints | `pass` | RAM (~150 MB), CPU, cold boot (2s), and Python 3.12 compatibility verified. |
| **F08** | Documentation | Discrepancy regarding tracked vs. ignored M8C dataset | `needs_decision` | 6,305 files tracked despite docs stating data is ignored; non-destructive note proposed. |
| **F09** | Licensing | Proprietary font rendered image hosting (M7E limitation) | `blocked` | M7E §11 expressly disclaims clearance for "a product, a model, or a deployment." |
| **F10** | Licensing | Luminari residual licensing uncertainties | `blocked` | M7E §8 records 3 unresolved uncertainties regarding Canada Type EULA vs Apple EULA. |

---

## 8. Exact Evidence for Every Classification

1. **F01 (Dashboard Implementation):**
   `app/tests/test_views.py`, `app/tests/test_data_contracts.py`, `app/tests/test_images.py`, `app/tests/test_analysis_format.py` (67 passed in 2.22s).
2. **F02 (Ground Truth Isolation):**
   `contracts.FORBIDDEN_FILES` in `app/lib/contracts.py`; `git ls-files data/processed/matched_font_benchmark_v0.1.0/ground_truth.csv` returns exit 0 with 0 lines.
3. **F03 (Zero Model Execution):**
   `app/lib/data.py` has no `import joblib`, `import sklearn`, or model fitting. All predictions loaded from `reports/matched_font_benchmark_v0_1_0/predictions.csv`.
4. **F04 (Interface Font Licensing):**
   `app/assets/OFL.txt`, `app/assets/fonts/OFL-IBMPlexSans.txt`, `app/assets/fonts/OFL-IBMPlexMono.txt`, `app/assets/fonts/OFL-SourceSerif4.txt`.
5. **F05 (Image Security & Traversal):**
   `data.resolve_image_path()` and `data.load_verified_image()` in `app/lib/data.py`; verified by `test_rejects_path_traversal_attempts` in `app/tests/test_images.py`.
6. **F06 (Telemetry):**
   `.streamlit/config.toml` contains `[browser] gatherUsageStats = false`.
7. **F07 (Technical Hosting):**
   Repository size 33.6 MB; `app/requirements.txt` contains 5 standard packages; memory consumption verified via process inspection.
8. **F08 (Documentation Discrepancy):**
   `git ls-files data/processed/matched_font_benchmark_v0.1.0/` returns 6,305 lines; `README.md` line 129 states `data/ generated datasets (Git-ignored)`.
9. **F09 (M7E Deployment Limitation):**
   `reports/font_roster_audit_v0_2_0/report.md` lines 319–320: *"The audit does not assess trademark risk, and it does not clear any face for use in a product, a model, or a deployment."*
10. **F10 (Luminari Uncertainties):**
   `reports/font_roster_audit_v0_2_0/report.md` line 238 and line 282: records 3 binding conditions and residual documentary uncertainties regarding Canada Type EULA and Apple EULA terms.

---

## 9. Recommended Remediation

To move from `public_launch_blocked` to `public_launch_ready`, the following sequential remediation steps are required:

### Step 1: Enforce Local-Only Stance (Immediate)
- Maintain the current configuration where the dashboard runs exclusively on `localhost` via `.venv-dashboard/bin/streamlit run app/streamlit_app.py`.
- Do not register or link the repository on Streamlit Community Cloud.
- Do not publish public URLs or create GitHub release assets for the dashboard.

### Step 2: Legal and Licensing Clearance (Required for Public Hosting)
The repository owner must formally evaluate and record one of the following decisions:
- **Option A (Formal Clearance of Existing Images):**
  Obtain a qualified legal review of the macOS 27 EULA, Monotype Desktop EULA, Microsoft Typography FAQ, and Canada Type EULA to establish whether displaying static PNG word images within a non-commercial, publicly accessible academic research dashboard falls within permitted output redistribution or constitutes a prohibited "product/deployment". Record this approval in a new milestone document (e.g., `M7F` or `D3`).
- **Option B (Metric-Only Public Dashboard):**
  Refactor the public deployment route to exclude image specimen plates entirely, rendering only numerical metrics, feature tables, and Altair scatter plots. The metric data contains no copyrightable font outlines or proprietary image data.
- **Option C (Open-License Specimen Replacement):**
  Generate a secondary set of public display specimens rendered exclusively with open-license typefaces (e.g., Google Fonts OFL fonts), clearly labeled as illustrative specimens while keeping proprietary benchmark metrics in the analytical tables.

### Step 3: Clarify Tracked Data in Current Documentation
- Include the non-destructive clarification note in future documentation updates to resolve the discrepancy between `.gitignore` policy and the tracked M8C benchmark files.

---

## 10. Explicit Final Gate

- **Final Gate Status:** `public_launch_blocked`

$$\mathbf{GATE:\; public\_launch\_blocked}$$

### Reason for Gate Closure:
Public deployment on Streamlit Community Cloud remains **strictly blocked**. While the application is technically complete, secure, and reproducible, the underlying benchmark images derive from proprietary typefaces whose governing audit (M7E §11) explicitly disclaims authorization for deployment. In accordance with project research integrity and legal caution rules, permission cannot be inferred. The application must remain local-only until explicit clearance is recorded.
