from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import content
from app.lib import charts, contracts, data
from app.views import error_analysis, overview

APP_ROOT = Path(__file__).resolve().parents[1]


def _app_test(module: str) -> AppTest:
    source = (
        f"import sys\nsys.path.insert(0, '.')\nfrom app.views import {module}\n{module}.render()\n"
    )
    return AppTest.from_string(source, default_timeout=60)


def test_overview_context_matches_authoritative_values() -> None:
    context = overview.build_context()
    overall = context["overall"]

    assert float(overall["mae"]) == pytest.approx(0.0596072060109396)
    assert float(overall["rmse"]) == pytest.approx(0.088610428864685)
    assert float(overall["r2"]) == pytest.approx(0.7328983709293053)
    assert context["counts"] == {
        "rows": 6_300,
        "primary_rows": 5_700,
        "diagnostic_rows": 600,
        "fonts": 10,
        "cases": 30,
        "sources": 300,
    }
    assert context["family"]["label"].tolist() == [
        "Gaussian blur",
        "Gaussian noise",
        "Thin lines",
        "Arc strokes",
        "Elastic",
    ]
    assert float(context["thin_s4"]["mae"]) == pytest.approx(0.207882)
    assert float(context["thin_s4"]["large_error_proportion"]) == pytest.approx(0.553333)
    assert float(context["arc_s4"]["mae"]) == pytest.approx(0.164892, abs=1e-6)
    assert float(context["arc_s4"]["large_error_proportion"]) == pytest.approx(0.280000, abs=1e-6)

    specimen = context["specimen"]
    assert specimen["sample_id"] == content.ILLUSTRATIVE_SAMPLE_ID
    assert float(specimen["ssim_target"]) == pytest.approx(0.7689188329609628)
    assert float(specimen["prediction"]) == pytest.approx(0.8929395114034177)
    assert float(specimen["residual"]) == pytest.approx(-0.1240206784424547)


def test_error_context_matches_m9a_values() -> None:
    context = error_analysis.build_context()
    interference = context["interference"]

    assert float(interference["mae"]) == pytest.approx(0.0908252693269)
    assert float(interference["r2"]) == pytest.approx(-0.430242012594)
    assert float(interference["large_error_proportion"]) == pytest.approx(0.110833333333)
    assert float(interference["top_5pct_source_sse_share"]) == pytest.approx(0.105589448181)
    assert len(context["severity"]) == 19
    assert len(context["severity_long"]) == 38
    assert len(context["calibration"]) == 4
    assert len(context["concentration"]) == 12
    assert len(context["fonts"]) == 10


def test_overview_renders_without_exception() -> None:
    at = _app_test("overview")
    at.run()

    assert not list(at.exception)
    assert len(at.get("imgs")) == 2
    captions = " ".join(caption.value for caption in at.caption)
    assert "Residual = target \u2212 prediction" in captions
    assert "Sample ID" in captions


def test_entrypoint_routes_bare_root_to_overview() -> None:
    at = AppTest.from_file(str(APP_ROOT / "streamlit_app.py"), default_timeout=60)
    at.run()

    assert not list(at.exception)
    assert len(at.get("imgs")) == 2
    assert "5,700 primary rows" in " ".join(caption.value for caption in at.caption)


def test_benchmark_renders_and_switches_records() -> None:
    at = _app_test("benchmark")
    at.run()
    assert not list(at.exception)
    assert len(at.get("imgs")) == 2

    at.session_state["bench_font"] = "impact"
    at.session_state["bench_case"] = "case-000"
    at.session_state["bench_family"] = "thin_lines"
    at.session_state["bench_severity"] = 4
    at.run()

    assert not list(at.exception)
    image_captions = " ".join(
        image.proto.imgs[0].caption for image in at.get("imgs") if len(image.proto.imgs)
    )
    captions = " ".join(caption.value for caption in at.caption)
    assert "Thin lines, s4" in image_captions
    assert "Residual = target \u2212 prediction" in captions


def test_benchmark_labels_diagnostic_records() -> None:
    at = _app_test("benchmark")
    at.run()
    at.session_state["bench_font"] = "arial"
    at.session_state["bench_case"] = "case-000"
    at.session_state["bench_diag"] = True
    at.run()
    at.session_state["bench_family"] = "clean"
    at.run()

    assert not list(at.exception)
    assert len(at.get("imgs")) == 1
    assert "Diagnostic record" in " ".join(caption.value for caption in at.caption)


def test_error_analysis_renders_without_exception() -> None:
    at = _app_test("error_analysis")
    at.run()

    assert not list(at.exception)
    assert len(at.dataframe) == 5
    assert "M9A interference slice" in " ".join(caption.value for caption in at.caption)


def test_scatter_payload_excludes_paths_hashes_and_forbidden_fields() -> None:
    core = data.load_core()
    primary = core[core["protocol_scope"] == "primary"]
    spec = charts.prediction_scatter(primary, primary.iloc[0]).to_dict()
    text = str(spec)

    for forbidden in (
        "image_path",
        "reference_path",
        "image_sha256",
        "reference_sha256",
        "ground_truth",
        "text_sha256",
        "source_cases",
    ):
        assert forbidden not in text
    assert len(text) < 4_000_000


def test_app_sources_do_not_import_training_or_network_stacks() -> None:
    forbidden = (
        "sklearn",
        "joblib",
        "pickle",
        "torch",
        "tensorflow",
        "requests",
        "urllib.request",
        "subprocess",
        "models/",
        "models\\",
    )
    for path in APP_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        for token in forbidden:
            assert token not in lowered, f"{token} found in {path}"


def test_app_sources_do_not_write_or_generate_artifacts() -> None:
    forbidden_calls = (
        r"\.fit\s*\(",
        r"\.predict\s*\(",
        r"\.predict_proba\s*\(",
        r"\.to_csv\s*\(",
        r"\.to_parquet\s*\(",
        r"\.savefig\s*\(",
        r"\.save\s*\(",
        r"urlopen\s*\(",
        r"requests\.",
        r"subprocess\.",
    )
    for path in APP_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden_calls:
            assert not re.search(pattern, source), f"{pattern} found in {path}"


def test_only_the_contract_names_ground_truth() -> None:
    for path in APP_ROOT.rglob("*.py"):
        if "tests" in path.parts or path.name == "contracts.py":
            continue
        source = path.read_text(encoding="utf-8").lower()
        assert "ground_truth" not in source, f"ground truth referenced in {path}"


def test_contracts_forbid_untracked_tables() -> None:
    assert "ground_truth.csv" in contracts.FORBIDDEN_FILES
    assert "run_metadata.json" in contracts.FORBIDDEN_FILES
    assert "input_manifest.csv" in contracts.FORBIDDEN_FILES


def test_sidebar_collapse_and_expand_buttons_preserved_in_css() -> None:
    from app.lib import theme

    css = theme.build_css()
    # Ensure stToolbar as a whole is not hidden so the sidebar expand button remains usable
    assert '[data-testid="stToolbar"] {\n  display: none !important;' not in css
    # Ensure stExpandSidebarButton is explicitly visible
    assert 'data-testid="stExpandSidebarButton"' in css
    # Ensure Material Symbols icon font is preserved for collapse and expand buttons
    assert "Material Symbols Rounded" in css
    assert 'data-testid="stSidebarCollapseButton"' in css
