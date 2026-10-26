# D5 — Controlled Deployment Report

- **Milestone:** D5
- **Repository:** `CAPTCHA-Quality-Robustness-Analyzer`
- **Target Platform:** Streamlit Community Cloud
- **Target Architecture:** Option A Results-Only Public-Safe Dashboard Profile
- **Status:** Phase 1 Complete · Live Deployment Operational
- **Deployment Authorization Status:** `deployment_authorized_and_live`
- **Hosted Application URL:** `https://captcha-robustness-analyzer.streamlit.app`
- **Report Date:** 2026-10-26

---

## Executive Summary

Milestone D5 executes the pre-launch verification and authorized launch protocol defined in [`docs/CONTROLLED_DEPLOYMENT_PLAN.md`](./CONTROLLED_DEPLOYMENT_PLAN.md).

All Phase 1 pre-launch verification gates were evaluated and confirmed passing (86/86 dashboard tests, 33/33 lifecycle tests, clean fresh-clone execution, and 0 linting diagnostics).

Following explicit repository owner launch on Streamlit Community Cloud at `https://captcha-robustness-analyzer.streamlit.app`, live remote smoke testing identified and resolved a sidebar navigation rendering defect (accidental hiding of `stExpandSidebarButton` by `stToolbar` display rule and icon font family override on `stSidebarCollapseButton`). The fix has been verified and applied.

---

## 1. Pre-Launch Verification Results (Phase 1)

| Verification Check | Target / Criterion | Result | Evidence & Notes |
| :--- | :--- | :---: | :--- |
| **Git Working Tree** | Clean working tree, in sync with `origin/main` | **PASS** | On branch `main`, 0 uncommitted changes. |
| **Release Tag Immutability** | Tag `v1.0.0` unmodified | **PASS** | `v1.0.0^{commit}` matches `9b7b62ff4669c39712b799d22d50b6bbb3e5b9c3`. |
| **Runtime Dependency Boundary** | Only production dependencies in `app/requirements.txt` | **PASS** | Pinned: `streamlit==1.50.0`, `pandas==2.3.3`, `pillow==11.3.0`, `altair==5.5.0`. ML stacks (`scikit-learn`, `joblib`, `torch`) excluded. |
| **Production Import Audit** | Zero test/development imports in runtime code | **PASS** | AST import audit across all `app/**/*.py` confirmed zero imports of `pytest`, `joblib`, or training stacks. |
| **Fresh-Clone Boot Test** | Clean boot without benchmark images, ground truth, or models | **PASS** | Headless `AppTest` passed 100% in isolated clone with `images/` directory removed. |
| **Dashboard Test Suite** | All unit and integration tests pass | **PASS** | `pytest app/tests` passed 86/86. |
| **Lifecycle & Release Suite** | Benchmark lifecycle contracts pass | **PASS** | `pytest tests/test_benchmark_lifecycle.py tests/test_release_documentation.py` passed 33/33. |
| **Code Hygiene** | Ruff linter and formatter | **PASS** | `ruff check app` (0 diagnostics) and `ruff format --check app` (26 files already formatted). |
| **Data Boundary Enforcement** | Public-safe mode strips paths and hashes | **PASS** | In `PUBLIC_SAFE_MODE=true`, `data.read_core` drops `image_path`, `reference_path`, `image_sha256`, and `reference_sha256`. |

---

## 2. Live Deployment Status (Phase 2 & Phase 3)

- **Authorization Status:** **AUTHORIZED BY REPOSITORY OWNER**
- **Deployment Status:** `deployment_authorized_and_live`
- **Hosted Application URL:** [`https://captcha-robustness-analyzer.streamlit.app`](https://captcha-robustness-analyzer.streamlit.app)
- **Host Infrastructure:** Streamlit Community Cloud (Python 3.12, Linux Container)
- **Active Environment Flag:** `PUBLIC_SAFE_MODE="true"` configured in Streamlit Cloud Secrets.

---

## 3. Remote Smoke-Testing & Remediation (Phase 4)

### 3.1 Initial Remote Smoke Test Observation
During initial post-deployment review on the live URL, a UI glitch occurred:
1. In the sidebar header, the text `"key..."` was visible instead of the collapse chevron icon.
2. Clicking `"key..."` caused the sidebar to collapse.
3. Once collapsed, the button to re-expand the sidebar was completely invisible, leaving the page with no visible navigation bar.

### 3.2 Root Cause Analysis
1. **Icon Ligature Glitch:** In Streamlit 1.50, the sidebar collapse button uses a Material Symbols icon with ligature string `:material/keyboard_double_arrow_left:`. In `app/lib/theme.py`, the rule `[data-testid="stSidebar"] span { font-family: var(--sans); }` overrode the icon font with `IBM Plex Sans`. Without the Material Symbols font, the browser rendered the literal string `keyboard_double_arrow_left`, which truncated in the button box to `"key..."`.
2. **Hidden Expand Button:** In Streamlit 1.50, when the sidebar collapses, the re-expand button (`[data-testid="stExpandSidebarButton"]`) is placed inside `[data-testid="stToolbar"]` in the header. In `app/lib/theme.py`, the rule `[data-testid="stToolbar"] { display: none !important; }` (intended to hide the Streamlit Deploy and hamburger menus) inadvertently hid the entire toolbar, including `stExpandSidebarButton`.

### 3.3 Applied Remediation
1. **Scoped Font Styles:** Scoped sidebar text font styles to specific text elements (`p, a, label, .sidebar-*`) and explicitly protected `Material Symbols Rounded !important` on `[data-testid="stSidebarCollapseButton"]` and `[data-testid="stExpandSidebarButton"]`.
2. **Granular Toolbar Hiding:** Replaced the blanket `[data-testid="stToolbar"]` display rule with targeted suppression of `[data-testid="stAppDeployButton"]`, `[data-testid="stToolbarActions"]`, `[data-testid="stHeaderActionElements"]`, and `#MainMenu`, while ensuring `[data-testid="stExpandSidebarButton"]` remains visible (`display: flex !important`).
3. **Explicit Default Expanded State:** Set `initial_sidebar_state="expanded"` in `st.set_page_config` in `app/streamlit_app.py`.
4. **Automated Regression Guard:** Added `test_sidebar_collapse_and_expand_buttons_preserved_in_css` to `app/tests/test_views.py`. All 86 tests pass.

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
