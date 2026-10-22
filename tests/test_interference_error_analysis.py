"""Focused tests for the read-only M9A interference error analysis."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from captcha_quality import interference_error_analysis as iea
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_point_metrics_known_values_and_residual_sign() -> None:
    target = np.array([0.9, 0.8, 0.7], dtype=float)
    prediction = np.array([0.9, 0.7, 0.6], dtype=float)
    metrics = iea._point_metrics(
        target,
        prediction,
        near_degenerate_sd=0.005,
        large_error_threshold=0.20,
    )
    assert metrics["n_rows"] == 3
    assert metrics["mae"] == pytest.approx(0.06666666666666667)
    assert metrics["rmse"] == pytest.approx(math.sqrt(0.02 / 3.0))
    assert metrics["bias"] == pytest.approx(0.06666666666666667)
    assert metrics["r2"] == pytest.approx(0.0)
    assert metrics["r2_defined"] is True
    assert metrics["large_error_count"] == 0
    assert metrics["target_mean"] == pytest.approx(0.8)
    assert metrics["prediction_mean"] == pytest.approx(0.7333333333333333)


def test_point_metrics_overprediction_is_negative() -> None:
    target = np.array([0.80, 0.81, 0.82], dtype=float)
    prediction = np.array([0.90, 0.91, 0.92], dtype=float)
    metrics = iea._point_metrics(
        target,
        prediction,
        near_degenerate_sd=0.005,
        large_error_threshold=0.20,
    )
    assert metrics["bias"] < 0.0
    assert metrics["residual_q50"] < 0.0


def test_r2_is_suppressed_for_constant_and_near_constant_targets() -> None:
    constant = iea._point_metrics(
        np.array([1.0, 1.0, 1.0]),
        np.array([0.9, 0.9, 0.9]),
        near_degenerate_sd=0.005,
        large_error_threshold=0.20,
    )
    assert math.isnan(constant["r2"])
    assert constant["r2_defined"] is False
    assert constant["r2_suppressed"] is True
    assert constant["r2_status"].startswith("suppressed_target_sd_lt")

    near_constant = iea._point_metrics(
        np.array([1.0, 1.0, 1.004]),
        np.array([0.9, 0.9, 0.9]),
        near_degenerate_sd=0.005,
        large_error_threshold=0.20,
    )
    assert math.isnan(near_constant["r2"])
    assert near_constant["r2_status"].startswith("suppressed_target_sd_lt")


def test_large_error_threshold_is_inclusive_at_exactly_point_twenty() -> None:
    metrics = iea._point_metrics(
        np.array([0.8, 0.9]),
        np.array([0.6, 0.9]),
        near_degenerate_sd=0.005,
        large_error_threshold=0.20,
    )
    assert metrics["large_error_count"] == 1
    assert metrics["large_error_proportion"] == pytest.approx(0.5)


def test_point_metrics_rejects_empty_or_non_finite() -> None:
    with pytest.raises(iea.AnalysisError):
        iea._point_metrics(
            np.array([]),
            np.array([]),
            near_degenerate_sd=0.005,
            large_error_threshold=0.20,
        )
    with pytest.raises(iea.AnalysisError):
        iea._point_metrics(
            np.array([0.9, math.nan]),
            np.array([0.8, 0.8]),
            near_degenerate_sd=0.005,
            large_error_threshold=0.20,
        )


def test_require_unique_and_missing_helpers_fail_closed() -> None:
    frame = pd.DataFrame({"source_id": ["a", "a"], "value": [1, 2]})
    with pytest.raises(iea.AnalysisError, match="duplicate"):
        iea._require_unique(frame, "source_id", "test frame")

    with pytest.raises(iea.AnalysisError, match="missing column"):
        iea._require_no_missing(frame, ["value", "absent"], "test frame")

    missing = pd.DataFrame({"source_id": ["a", ""], "value": [1, 2]})
    with pytest.raises(iea.AnalysisError, match="missing value"):
        iea._require_no_missing(missing, ["source_id"], "test frame")


def test_same_values_and_float_values_fail_closed() -> None:
    iea._same_values(pd.Series(["a", "b"]), pd.Series(["a", "b"]), "same")
    with pytest.raises(iea.AnalysisError, match="differ"):
        iea._same_values(pd.Series(["a", "b"]), pd.Series(["a", "c"]), "different")

    iea._same_float_values(pd.Series([0.1, 0.2]), pd.Series([0.1, 0.2]), "same")
    with pytest.raises(iea.AnalysisError, match="differ"):
        iea._same_float_values(pd.Series([0.1, 0.2]), pd.Series([0.1, 0.3]), "different")


def _small_tables() -> dict[str, pd.DataFrame]:
    predictions = pd.DataFrame(
        {
            "row_id": ["r1", "r2"],
            "sample_id": ["s1", "s2"],
            "source_id": ["src1", "src1"],
        }
    )
    manifest = predictions.copy()
    features = pd.DataFrame({"sample_id": ["s1", "s2"]})
    sources = pd.DataFrame({"source_id": ["src1"]})
    source_cases = pd.DataFrame({"source_id": ["src1"]})
    ground_truth = pd.DataFrame({"source_id": ["src1"]})
    return {
        "predictions": predictions,
        "manifest": manifest,
        "features": features,
        "sources": sources,
        "source_cases": source_cases,
        "ground_truth": ground_truth,
    }


def test_join_key_validation_rejects_duplicate_keys() -> None:
    tables = _small_tables()
    tables["predictions"] = pd.concat(
        [tables["predictions"], tables["predictions"].iloc[[0]]], ignore_index=True
    )
    checks: list[iea.Check] = []
    with pytest.raises(iea.AnalysisError, match="duplicate"):
        iea._validate_table_keys(
            tables["predictions"],
            tables["manifest"],
            tables["features"],
            tables["sources"],
            tables["source_cases"],
            tables["ground_truth"],
            checks,
        )


def test_join_coverage_fails_closed_when_prediction_rows_are_missing() -> None:
    predictions = pd.DataFrame({"row_id": ["r1"], "sample_id": ["s1"]})
    manifest = pd.DataFrame({"row_id": ["r1", "r2"], "sample_id": ["s1", "s2"]})
    checks: list[iea.Check] = []
    with pytest.raises(iea.AnalysisError, match="coverage"):
        iea._validate_row_coverage(predictions, manifest, checks)


def test_reference_coverage_rejects_extra_and_missing_keys() -> None:
    tables = _small_tables()
    tables["features"] = pd.concat(
        [tables["features"], pd.DataFrame({"sample_id": ["s3"]})], ignore_index=True
    )
    checks: list[iea.Check] = []
    with pytest.raises(iea.AnalysisError, match="coverage_features_sample_id"):
        iea._validate_reference_coverage(
            tables["predictions"],
            tables["features"],
            tables["sources"],
            tables["source_cases"],
            tables["ground_truth"],
            checks,
        )

    tables = _small_tables()
    tables["sources"] = pd.DataFrame({"source_id": []})
    checks = []
    with pytest.raises(iea.AnalysisError, match="coverage_sources_source_id"):
        iea._validate_reference_coverage(
            tables["predictions"],
            tables["features"],
            tables["sources"],
            tables["source_cases"],
            tables["ground_truth"],
            checks,
        )


def test_read_csv_pinned_checks_hash_rows_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    digest = sha256_file(path)
    frame = iea._read_csv_pinned(
        path,
        expected_sha256=digest,
        expected_rows=2,
        expected_columns=("a", "b"),
    )
    assert frame.shape == (2, 2)

    with pytest.raises(iea.AnalysisError, match="hash mismatch"):
        iea._read_csv_pinned(
            path,
            expected_sha256="0" * 64,
            expected_rows=2,
            expected_columns=("a", "b"),
        )
    with pytest.raises(iea.AnalysisError, match="row-count"):
        iea._read_csv_pinned(
            path,
            expected_sha256=digest,
            expected_rows=3,
            expected_columns=("a", "b"),
        )
    with pytest.raises(iea.AnalysisError, match="schema"):
        iea._read_csv_pinned(
            path,
            expected_sha256=digest,
            expected_rows=2,
            expected_columns=("a", "c"),
        )


def test_scope_filtering_rejects_unknown_and_empty_scopes() -> None:
    frame = pd.DataFrame(
        {
            "protocol_scope": ["primary"],
            "variant_family": ["thin_lines"],
            "severity_level": [1],
        }
    )
    assert iea.scope_frame(frame, "thin_lines_s1").shape == (1, 3)
    with pytest.raises(iea.AnalysisError, match="unknown scope"):
        iea.scope_frame(frame, "not_a_scope")
    with pytest.raises(iea.AnalysisError, match="empty"):
        iea.scope_frame(frame, "arc_strokes")


def _severity_frame() -> pd.DataFrame:
    rows = []
    index = 0
    for family, severities in iea.PRIMARY_SEVERITIES.items():
        for severity in severities:
            target = 0.90 - 0.01 * severity
            prediction = target + 0.01
            rows.append(
                {
                    "source_id": f"src-{index:03d}",
                    "font_id": "font-a",
                    "case_id": "case-000",
                    "protocol_scope": "primary",
                    "variant_family": family,
                    "severity_level": severity,
                    "ssim_target": target,
                    "prediction": prediction,
                    "abs_residual": abs(target - prediction),
                }
            )
            index += 1
    return pd.DataFrame(rows)


def test_severity_subgroup_handling_is_complete_and_fail_closed() -> None:
    config = iea.load_config()
    config = dataclasses.replace(
        config,
        run=dataclasses.replace(config.run, n_bootstrap=20),
    )
    frame = _severity_frame()
    result = iea.build_severity_metrics(frame, config)
    assert result.shape[0] == 19
    assert set(result["family"]) == set(iea.PRIMARY_SEVERITIES)

    missing = frame.loc[~(frame["variant_family"].eq("thin_lines") & frame["severity_level"].eq(4))]
    with pytest.raises(iea.AnalysisError, match="missing primary severity"):
        iea.build_severity_metrics(missing, config)


def test_source_bootstrap_is_deterministic_and_keeps_whole_groups() -> None:
    source_ids = [f"src-{index}" for index in range(8)] * 3
    first = iea.source_bootstrap_indices(source_ids, n_bootstrap=25, seed=123)
    second = iea.source_bootstrap_indices(source_ids, n_bootstrap=25, seed=123)
    other = iea.source_bootstrap_indices(source_ids, n_bootstrap=25, seed=124)
    assert len(first) == len(second) == len(other) == 25
    assert all(len(draw) == 8 for draw in first)
    assert all(set(draw).issubset(set(source_ids)) for draw in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert any(not np.array_equal(a, b) for a, b in zip(first, other, strict=True))


def test_bootstrap_metric_ci_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "source_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "ssim_target": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "prediction": [0.85, 0.82, 0.72, 0.58, 0.48, 0.45],
        }
    )
    first = iea._bootstrap_metric_ci(
        frame,
        n_bootstrap=50,
        seed=7,
        confidence_level=0.95,
        near_degenerate_sd=0.005,
    )
    second = iea._bootstrap_metric_ci(
        frame,
        n_bootstrap=50,
        seed=7,
        confidence_level=0.95,
        near_degenerate_sd=0.005,
    )
    assert first == second
    assert first["n_bootstrap_sources"] == 3
    assert first["mae_ci_low"] <= first["mae_ci_high"]


def test_paired_bootstrap_ci_is_deterministic() -> None:
    values = [0.1, 0.2, -0.05, 0.3, 0.15]
    first = iea.paired_bootstrap_ci(values, n_bootstrap=100, seed=11, confidence_level=0.95)
    second = iea.paired_bootstrap_ci(values, n_bootstrap=100, seed=11, confidence_level=0.95)
    assert first == second
    assert first[0] <= first[1]


def test_bh_qvalues_known_example_and_nan_preservation() -> None:
    qvalues = iea._bh_qvalues([0.01, 0.04, 0.03, math.nan])
    assert qvalues[0] == pytest.approx(0.03)
    assert qvalues[1] == pytest.approx(0.04)
    assert qvalues[2] == pytest.approx(0.04)
    assert math.isnan(qvalues[3])


def test_within_source_response_uses_source_groups() -> None:
    frame = pd.DataFrame(
        {
            "source_id": ["s1", "s1", "s2", "s2"],
            "ssim_target": [0.9, 0.8, 0.7, 0.6],
            "prediction": [0.9, 0.8, 0.7, 0.6],
        }
    )
    result = iea._within_source_response(frame)
    assert result["within_source_prediction_target_sd_ratio_median"] == pytest.approx(1.0)
    assert result["within_source_calibration_slope_median"] == pytest.approx(1.0)
    assert result["within_source_prediction_target_corr_median"] == pytest.approx(1.0)


def test_output_directory_guard_rejects_unsafe_paths(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (tmp_path / "data").mkdir()
    (reports / "matched_font_benchmark_v0_1_0").mkdir()

    valid = reports / "interference_error_analysis_v0_1_0"
    assert iea.validate_output_directory(tmp_path, valid) == valid.resolve()

    with pytest.raises(iea.AnalysisConfigError, match="must start with"):
        iea.validate_output_directory(tmp_path, reports / "other_report")
    with pytest.raises(iea.AnalysisConfigError, match="below"):
        iea.validate_output_directory(
            tmp_path, tmp_path / "data" / "interference_error_analysis_v0_1_0"
        )
    with pytest.raises(iea.AnalysisConfigError, match="forbidden"):
        iea.validate_output_directory(
            tmp_path,
            reports / "matched_font_benchmark_v0_1_0" / "interference_error_analysis_v0_1_0",
        )


def test_static_scope_guard_rejects_no_model_fit_or_generation() -> None:
    checks: list[iea.Check] = []
    iea._static_analysis_scope_check(checks)
    assert len(checks) == 1
    assert checks[0].check_id == "analysis_code_scope"
    assert checks[0].passed is True


def test_forbidden_imports_and_calls_are_declared() -> None:
    assert "sklearn" in iea.FORBIDDEN_IMPORTS
    assert "joblib" in iea.FORBIDDEN_IMPORTS
    assert "fit" in iea.FORBIDDEN_CALL_NAMES
    assert "predict" in iea.FORBIDDEN_CALL_NAMES
    assert "generate_dataset" in iea.FORBIDDEN_CALL_NAMES


def test_config_loads_with_frozen_seed_and_thresholds() -> None:
    config = iea.load_config(REPO_ROOT / "configs/interference_error_analysis_v0_1_0.toml")
    assert config.run.seed == 20261022
    assert config.run.n_bootstrap == 2000
    assert config.analysis.near_degenerate_sd == 0.005
    assert config.analysis.large_error_threshold == 0.20
    assert config.analysis.group_column == "source_id"
    assert set(pin.key for pin in config.inputs) == set(iea.REQUIRED_INPUT_KEYS)
