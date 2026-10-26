# D5 — Controlled Deployment Report

- **Milestone:** D5
- **Repository:** `CAPTCHA-Quality-Robustness-Analyzer`
- **Target Platform:** Streamlit Community Cloud
- **Target Architecture:** Option A Results-Only Public-Safe Dashboard Profile
- **Status:** Phase 1 Complete · Phase 2 Authorization Gate Held
- **Deployment Authorization Status:** `deployment_pending_explicit_owner_authorization`
- **Report Date:** 2026-10-26

---

## Executive Summary

Milestone D5 executes the pre-launch verification and authorization protocol defined in [`docs/CONTROLLED_DEPLOYMENT_PLAN.md`](./CONTROLLED_DEPLOYMENT_PLAN.md).

All Phase 1 pre-launch verification gates have been evaluated and confirmed passing:
- The working tree is clean and synchronized with `origin/main` at commit `4ea286d`.
- Release tag `v1.0.0` points strictly to commit `9b7b62f` and remains frozen.
- The dashboard test suite passes 85/85 tests, and lifecycle/release tests pass 33/33 tests.
- Ruff linting and formatting pass across all 26 application files with zero diagnostics.
- A fresh-clone simulation verified that the application initializes, loads all 12 allowlisted CSVs, and executes all three views with zero errors—even when the `images/` directory is completely removed from the filesystem.

In strict compliance with the **Phase 2 Authorization Gate**, deployment has **not** been launched, no cloud accounts have been connected, and no public URL has been created. Explicit repository owner authorization is required before initiating the controlled launch.

---

## 1. Pre-Launch Verification Results (Phase 1)

| Verification Check | Target / Criterion | Result | Evidence & Notes |
| :--- | :--- | :---: | :--- |
| **Git Working Tree** | Clean working tree, in sync with `origin/main` | **PASS** | On branch `main`, 0 uncommitted changes. |
| **Release Tag Immutability** | Tag `v1.0.0` unmodified | **PASS** | `v1.0.0^{commit}` matches `9b7b62ff4669c39712b799d22d50b6bbb3e5b9c3`. |
| **Runtime Dependency Boundary** | Only production dependencies in `app/requirements.txt` | **PASS** | Pinned: `streamlit==1.50.0`, `pandas==2.3.3`, `pillow==11.3.0`, `altair==5.5.0`. ML stacks (`scikit-learn`, `joblib`, `torch`) excluded. |
| **Production Import Audit** | Zero test/development imports in runtime code | **PASS** | AST import audit across all `app/**/*.py` confirmed zero imports of `pytest`, `joblib`, or training stacks. |
| **Fresh-Clone Boot Test** | Clean boot without benchmark images, ground truth, or models | **PASS** | Headless `AppTest` passed 100% in isolated clone with `images/` directory removed. |
| **Dashboard Test Suite** | All unit and integration tests pass | **PASS** | `pytest app/tests` passed 85/85 in 2.96s. |
| **Lifecycle & Release Suite** | Benchmark lifecycle contracts pass | **PASS** | `pytest tests/test_benchmark_lifecycle.py tests/test_release_documentation.py` passed 33/33 in 16.79s. |
| **Code Hygiene** | Ruff linter and formatter | **PASS** | `ruff check app` (0 diagnostics) and `ruff format --check app` (26 files already formatted). |
| **Data Boundary Enforcement** | Public-safe mode strips paths and hashes | **PASS** | In `PUBLIC_SAFE_MODE=true`, `data.read_core` drops `image_path`, `reference_path`, `image_sha256`, and `reference_sha256`. |

---

## 2. Authorization Gate Status (Phase 2)

- **Gate Status:** **HELD**
- **Deployment Status:** `deployment_pending_explicit_owner_authorization`
- **Hosted Application URL:** `None` (Deployment not initiated)
- **Account Actions:** No cloud credentials connected; no hosting services invoked.

In accordance with project governance:
> A successful test suite, an existing deployment plan, or an earlier technical classification does not constitute deployment authorization. Only explicit repository owner authorization may open the Phase 2 gate.

---

## 3. Deployment Configuration Specification (When Authorized)

When the repository owner authorizes the controlled launch, the deployment should be executed on Streamlit Community Cloud using the exact parameters below:

```yaml
platform: "Streamlit Community Cloud"
repository: "yash-mishraa/CAPTCHA-Quality-Robustness-Analyzer"
branch: "main"
main_file_path: "app/streamlit_app.py"
python_version: "3.12"
environment_secrets:
  PUBLIC_SAFE_MODE: "true"
server_config:
  headless: true
  gatherUsageStats: false
runtime_requirements: "app/requirements.txt"
```

### Streamlit Community Cloud Dependency Resolution Note
Streamlit Community Cloud automatically searches for a requirements manifest in the entry point directory first (`app/requirements.txt`). This ensures that the root `pyproject.toml` (which includes scikit-learn, joblib, and dev dependencies) is ignored by the cloud runner, guaranteeing a minimal container environment.

---

## 4. Concise Owner Launch Checklist

To deploy the application to Streamlit Community Cloud, follow these steps:

1. **Access Console:**
   - Log into [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account (`yash-mishraa`).
2. **Create New App:**
   - Click **"Create app"** > **"I already have an app"**.
3. **Configure Repository Coordinates:**
   - **Repository:** `yash-mishraa/CAPTCHA-Quality-Robustness-Analyzer`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL (optional):** Choose an available custom slug (e.g. `captcha-robustness-analyzer`).
4. **Configure Advanced Settings:**
   - Click **"Advanced settings"**.
   - **Python version:** Select `3.12`.
   - **Secrets (Environment Variables):** In the Secrets TOML editor, enter:
     ```toml
     PUBLIC_SAFE_MODE = "true"
     ```
5. **Deploy:**
   - Click **"Deploy!"**.
6. **Execute Remote Smoke Tests (Phase 4):**
   - Once the app builds, run the remote verification protocol outlined in Section 5.

---

## 5. Post-Deployment Smoke-Testing Protocol (Phase 4)

Immediately after hosting succeeds, the owner should execute these verification checks:

1. **Header & Badge:**
   - Confirm the sidebar contains the badge: `Results-Only Profile · Public-Safe`.
   - Confirm the sidebar metadata card displays: `Frozen benchmark results · 5,700 primary evaluation rows`.
2. **Overview View (`/overview`):**
   - Verify primary metric cards match frozen v1.0.0 values: $R^2 = 0.828$, $\text{RMSE} = 0.052$, $\text{MAE} = 0.038$.
   - Confirm the Specimen Exhibit panel displays the Illustrative Record Exhibit Card (tabular values only; zero images).
3. **Benchmark Explorer View (`/benchmark`):**
   - Test font, variant family, and severity filters.
   - Verify the Altair scatter plot renders all 5,700 primary evaluation points.
   - Inspect browser DevTools Network tab: confirm zero image files or media blobs are downloaded.
4. **Error Analysis View (`/error-analysis`):**
   - Verify M9 summary metrics (large error proportion $1.4\%$, top-5% source SSE share $38.4\%$).
   - Verify calibration slope curves and Pareto charts render.
   - Verify the M9A exploratory disclaimer is prominently displayed.
5. **Mobile Viewport Check:**
   - Open on a mobile device or browser responsive mode at $390\text{px}$ width. Confirm sidebar collapses into a drawer and tables scroll horizontally.
6. **Network & DOM Sanitization:**
   - Confirm DevTools shows zero requests to external CDNs or Google Fonts (all fonts load from inline data URIs).
   - Confirm Streamlit "Deploy" button and toolbar menu are suppressed via CSS.

---

## 6. Rollback Readiness (Phase 5)

If any remote smoke test fails or if any benchmark image, local path, or image hash is exposed:
1. **Immediate Teardown:** In the Streamlit Cloud dashboard, click the app menu (⋮) > **"Delete App"** or **"Pause App"** (< 60 seconds).
2. **Killswitch Alternative:** Delete `PUBLIC_SAFE_MODE = "true"` from Secrets; the application will restart in an unconfigured state and block access.
3. **Root Cause Analysis:** Inspect container logs in the Streamlit Cloud console. Do not modify the frozen `v1.0.0` release tag to resolve hosting issues.

---

## 7. Residual Technical & Licensing Considerations

- **Licensing Boundary:** Option A achieves complete compliance with M7E §11 Risk B11 by omitting 100% of benchmark PNGs and proprietary font binaries. Only public aggregate metrics, feature values, model predictions, and SIL OFL-1.1 bundled fonts are served. This technical boundary does not constitute a formal legal opinion.
- **Account Credentials:** No GitHub OAuth tokens or Streamlit Cloud account credentials were used or stored by this agent.
- **Container Footprint:** The application requires ~50MB RAM at runtime for the read-only Pandas DataFrames, well within Streamlit Cloud's 1GB limit.

---

## 8. Milestone Conclusion

- **Pre-Launch Verification:** Complete and verified (**ALL PASS**).
- **Authorization Gate:** Held (**Awaiting owner approval**).
- **Deliverable:** Committed and pushed to `main`.
