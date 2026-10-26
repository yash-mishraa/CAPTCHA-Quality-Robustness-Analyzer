from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.lib import contracts, data
from app.tests.helpers import raises_integrity

APP_ROOT = Path(__file__).resolve().parents[1]


def _app_test(module: str) -> AppTest:
    source = (
        f"import sys\nsys.path.insert(0, '.')\nfrom app.views import {module}\n{module}.render()\n"
    )
    return AppTest.from_string(source, default_timeout=60)


def test_public_safe_overview_renders_without_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_SAFE_MODE", "true")
    assert contracts.is_public_safe_mode()

    at = _app_test("overview")
    at.run()

    assert not list(at.exception)
    # Zero images served in public-safe mode
    assert len(at.get("imgs")) == 0
    captions = " ".join(caption.value for caption in at.caption)
    assert "Residual = target \u2212 prediction" in captions
    assert "Sample ID" in captions


def test_public_safe_benchmark_renders_without_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_SAFE_MODE", "true")
    assert contracts.is_public_safe_mode()

    at = _app_test("benchmark")
    at.run()
    assert not list(at.exception)
    assert len(at.get("imgs")) == 0

    at.session_state["bench_font"] = "impact"
    at.session_state["bench_case"] = "case-000"
    at.session_state["bench_family"] = "thin_lines"
    at.session_state["bench_severity"] = 4
    at.run()

    assert not list(at.exception)
    assert len(at.get("imgs")) == 0
    captions = " ".join(caption.value for caption in at.caption)
    assert "Residual = target \u2212 prediction" in captions


def test_public_safe_error_analysis_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_SAFE_MODE", "true")
    assert contracts.is_public_safe_mode()

    at = _app_test("error_analysis")
    at.run()

    assert not list(at.exception)
    assert len(at.dataframe) == 5
    assert len(at.get("imgs")) == 0


def test_public_safe_entrypoint_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLIC_SAFE_MODE", "true")
    assert contracts.is_public_safe_mode()

    at = AppTest.from_file(str(APP_ROOT / "streamlit_app.py"), default_timeout=60)
    at.run()

    assert not list(at.exception)
    assert len(at.get("imgs")) == 0


def test_public_safe_prohibits_image_loading_and_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PUBLIC_SAFE_MODE", "true")
    assert contracts.is_public_safe_mode()

    with raises_integrity("prohibited in public-safe mode"):
        data.resolve_image_path("images/clean/src-test.png")

    with raises_integrity("prohibited in public-safe mode"):
        data.load_verified_image(
            pd.Series({"image_path": "images/clean/x.png", "image_sha256": "0" * 64}),
            "image",
        )

    with raises_integrity("prohibited in public-safe mode"):
        data.load_selected_image("src-test", "reference")


def test_public_safe_works_without_images_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUBLIC_SAFE_MODE", "true")
    assert contracts.is_public_safe_mode()

    data_dir = tmp_path / "matched_font_benchmark_v0.1.0"
    data_dir.mkdir(parents=True)
    # Copy only manifest and features; NO images directory!
    shutil.copy2(contracts.DATA_ROOT / "manifest.csv", data_dir / "manifest.csv")
    shutil.copy2(contracts.DATA_ROOT / "features.csv", data_dir / "features.csv")
    assert not (data_dir / "images").exists()

    core = data.read_core(data_root=data_dir, public_safe=True)
    assert len(core) == contracts.EXPECTED_ROWS
    # Image paths and hashes stripped
    for col in ("image_path", "reference_path", "image_sha256", "reference_sha256"):
        assert col not in core.columns


def test_forbidden_files_fail_closed() -> None:
    assert "ground_truth.csv" in contracts.FORBIDDEN_FILES

    for forbidden in contracts.FORBIDDEN_FILES:
        with raises_integrity("not allowlisted"):
            data.read_report_table("m8", forbidden)

    # Path traversal attempts
    with raises_integrity("path traversal"):
        data.read_report_table("m8", "../overall_metrics.csv")
    with raises_integrity("path traversal"):
        data.read_report_table("m8", "sub/overall_metrics.csv")


def test_non_finite_metrics_fail_closed(tmp_path: Path) -> None:
    report_dir = tmp_path / "m8_report"
    report_dir.mkdir()
    df = pd.read_csv(contracts.M8_REPORT_DIR / "overall_metrics.csv")
    df.loc[0, "mae"] = np.inf
    df.to_csv(report_dir / "overall_metrics.csv", index=False)

    old_dir = contracts.REPORT_DIRS["m8"]
    try:
        contracts.REPORT_DIRS["m8"] = report_dir
        with raises_integrity("non-finite"):
            data.read_report_table("m8", "overall_metrics.csv", verify_hashes=False)
    finally:
        contracts.REPORT_DIRS["m8"] = old_dir


def test_default_mode_preserves_image_exhibits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUBLIC_SAFE_MODE", raising=False)
    assert not contracts.is_public_safe_mode()

    at = _app_test("overview")
    at.run()

    assert not list(at.exception)
    # Default mode displays reference and corrupted images
    assert len(at.get("imgs")) == 2
