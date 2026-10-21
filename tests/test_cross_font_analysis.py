"""Tests for the M7C cross-font evaluation analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from captcha_quality import cross_font_analysis as cfa
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.modelling import TARGET, regression_metrics
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "cross_font_analysis_v0_1_0.toml"
M7B_DATA_ROOT = REPO_ROOT / "data" / "processed" / "heldout_font_benchmark_v0.1.0"

if not M7B_DATA_ROOT.exists():  # pragma: no cover - depends on generated artifacts
    pytest.skip(
        "frozen M7B inputs are not present; generate the upstream artifacts first",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def bundle():
    """Load the frozen inputs and run the deterministic M7C core once."""
    config = cfa.load_config(CONFIG_PATH)
    inputs = cfa.load_inputs(config)
    result = cfa.analyze(inputs)
    report = cfa.build_report(inputs, result)
    checks = cfa.build_isolation_checks(inputs, result, report)
    return config, inputs, result, report, checks


def _family_row(result: cfa.AnalysisResult, scope: str, font_id: str = "all") -> pd.Series:
    return cfa._metric_row(result.family_metrics, scope=scope, font_id=font_id)


def _frame_for(inputs: cfa.AnalysisInputs, scope: str, font_id: str = "all") -> pd.DataFrame:
    return cfa._scope_frame(inputs.m7b_predictions, scope, font_id)


def test_metric_calculations(bundle):
    _, inputs, result, _, _ = bundle
    frame = _frame_for(inputs, "all_primary")
    target = frame[TARGET].to_numpy(dtype=np.float64)
    predicted = frame["prediction"].to_numpy(dtype=np.float64)
    expected = regression_metrics(target, predicted)
    row = _family_row(result, "all_primary")
    assert int(row["n_rows"]) == len(frame)
    assert int(row["n_sources"]) == frame["source_id"].nunique()
    assert int(row["n_distinct_images"]) == frame["image_sha256"].nunique()
    assert math.isclose(float(row["mae"]), float(expected["mae"]), rel_tol=1e-12)
    assert math.isclose(float(row["rmse"]), float(expected["rmse"]), rel_tol=1e-12)
    assert math.isclose(float(row["r2"]), float(expected["r2"]), rel_tol=1e-12)
    assert math.isclose(
        float(row["target_mean"]),
        float(np.mean(target)),
        rel_tol=1e-12,
    )
    assert math.isclose(
        float(row["target_sd"]),
        float(np.std(target, ddof=0)),
        rel_tol=1e-12,
    )


def test_residual_sign_convention(bundle):
    _, inputs, result, _, _ = bundle
    predictions = inputs.m7b_predictions
    residual = predictions[TARGET] - predictions["prediction"]
    assert np.allclose(residual, predictions["residual"], atol=1e-12)
    positive = predictions.loc[predictions["residual"] > 0].iloc[0]
    negative = predictions.loc[predictions["residual"] < 0].iloc[0]
    assert positive["prediction"] < positive[TARGET]
    assert negative["prediction"] > negative[TARGET]
    all_primary = _family_row(result, "all_primary")
    primary_residual = predictions.loc[predictions["protocol_scope"] == "primary", "residual"]
    assert math.isclose(
        float(all_primary["mean_residual"]),
        float(np.mean(primary_residual)),
        rel_tol=1e-12,
    )


def test_source_aggregation_and_sse(bundle):
    _, inputs, result, _, _ = bundle
    frame = _frame_for(inputs, "all_primary")
    source_metrics = result.source_metrics.loc[result.source_metrics["scope"] == "all_primary"]
    assert len(source_metrics) == 60
    assert source_metrics["source_id"].nunique() == 60
    for _, row in source_metrics.iterrows():
        group = frame.loc[frame["source_id"] == row["source_id"]]
        expected_sse = float(
            np.sum((group[TARGET].to_numpy() - group["prediction"].to_numpy()) ** 2)
        )
        assert math.isclose(float(row["sse"]), expected_sse, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(
            float(row["mae"]),
            float(np.mean(np.abs(group[TARGET] - group["prediction"]))),
            rel_tol=1e-12,
        )
    assert math.isclose(float(source_metrics["sse_share"].sum()), 1.0, rel_tol=1e-12)
    assert set(source_metrics["sse_rank"]) == set(range(1, 61))


def test_source_concentration_accounting(bundle):
    _, _, result, _, _ = bundle
    concentration = result.error_concentration.loc[
        (result.error_concentration["scope"] == "all_primary")
        & (result.error_concentration["font_id"] == "all")
    ].iloc[0]
    source_metrics = result.source_metrics.loc[result.source_metrics["scope"] == "all_primary"]
    top_shares = source_metrics["sse_share"].nlargest(int(concentration["top_k"]))
    assert math.isclose(
        float(concentration["top_k_share"]),
        float(top_shares.sum()),
        rel_tol=1e-12,
    )
    assert 0.0 < float(concentration["effective_fraction"]) <= 1.0
    assert int(concentration["sources_for_50pct"]) <= int(concentration["n_sources"])


def test_bootstrap_determinism(bundle):
    _, inputs, _, _, _ = bundle
    frame = _frame_for(inputs, "all_primary", "georgia")
    seed = derive_noise_seed(20261110, "test_bootstrap", "all_primary", "georgia")
    first = cfa._bootstrap_group_metrics(frame, n_bootstrap=200, seed=seed, confidence_level=0.95)
    second = cfa._bootstrap_group_metrics(frame, n_bootstrap=200, seed=seed, confidence_level=0.95)
    other = cfa._bootstrap_group_metrics(
        frame, n_bootstrap=200, seed=seed + 1, confidence_level=0.95
    )
    assert first == second
    assert first["n_groups"] == 20
    assert first["mae_ci_low"] <= first["mae_ci_high"]
    assert first["rmse_ci_low"] <= first["rmse_ci_high"]
    assert (first["mae_ci_low"], first["mae_ci_high"]) != (
        other["mae_ci_low"],
        other["mae_ci_high"],
    )


def test_paired_eligibility_and_differences(bundle):
    _, inputs, result, _, _ = bundle
    paired = result.paired_differences
    assert bool(paired["eligible"].all())
    assert bool(paired["paired"].all())
    assert set(paired.loc[paired["font_id"] == "all", "n_groups"]) == {60}
    assert set(paired.loc[paired["font_id"] != "all", "n_groups"]) == {20}
    assert np.allclose(
        paired["difference"],
        paired["estimate_a"] - paired["estimate_b"],
        atol=1e-12,
        rtol=0.0,
    )
    frame_a = _frame_for(inputs, "gaussian_blur", "georgia").iloc[:10].copy()
    frame_b = _frame_for(inputs, "gaussian_noise", "monaco").iloc[:10].copy()
    ineligible = cfa._paired_metric_difference(
        frame_a,
        frame_b,
        metric="rmse",
        n_bootstrap=10,
        seed=1,
        confidence_level=0.95,
    )
    assert ineligible["eligible"] is False
    assert ineligible["reason"]


def test_severity_accounting(bundle):
    _, inputs, result, _, _ = bundle
    severity = result.severity_metrics
    assert set(severity["amplitude_px"]) == {0.5, 1.0, 2.0, 3.0}
    mapping = (
        severity[["amplitude_px", "severity_level"]]
        .drop_duplicates()
        .set_index("amplitude_px")["severity_level"]
        .to_dict()
    )
    assert mapping == {0.5: 1, 1.0: 2, 2.0: 3, 3.0: 4}
    for font_id in ("all", *cfa.FONT_ORDER):
        subset = severity.loc[severity["font_id"] == font_id]
        assert set(subset["n_sources"]) == {60 if font_id == "all" else 20}
        primary = subset.loc[subset["protocol_role"] == "primary"]
        assert set(primary["amplitude_px"]) == {1.0, 2.0, 3.0}
        assert bool(primary["aggregate_target_order_consistent"].all())
        assert (
            primary["n_sources_monotonic"].to_numpy() == primary["n_sources_primary"].to_numpy()
        ).all()
    diagnostic = severity.loc[
        (severity["font_id"] == "all") & (severity["amplitude_px"] == 0.5)
    ].iloc[0]
    assert float(diagnostic["target_sd"]) < 0.005
    assert bool(diagnostic["r2_suppressed"]) is True


def test_degenerate_target_handling(bundle):
    _, inputs, result, _, _ = bundle
    clean = _family_row(result, "clean_reference")
    diagnostic = _family_row(result, "diagnostic_0_5px")
    assert bool(clean["near_degenerate"]) is True
    assert bool(clean["r2_defined"]) is False
    assert math.isnan(float(clean["r2"]))
    assert bool(diagnostic["near_degenerate"]) is True
    assert bool(diagnostic["r2_suppressed"]) is True
    constant = pd.DataFrame(
        {
            "source_id": ["a", "b"],
            "image_sha256": ["i1", "i2"],
            TARGET: [1.0, 1.0],
            "prediction": [0.99, 1.01],
        }
    )
    values = cfa._metric_values(constant, near_degenerate_sd=0.005)
    assert values["near_degenerate"] is True
    assert values["r2_defined"] is False
    assert math.isnan(values["r2"])


def test_claim_table_completeness(bundle):
    _, _, result, report, checks = bundle
    claims = result.claim_evidence
    assert tuple(claims["claim_id"]) == cfa.CLAIM_IDS
    assert set(claims["classification"]).issubset(cfa.CLASSIFICATIONS)
    assert claims["evidence"].str.len().gt(0).all()
    assert claims["limitations"].str.len().gt(0).all()
    assert claims["rule_met"].map(type).eq(bool).all()
    required_claims = {
        "C01": "differs across the three tested held-out fonts",
        "C02": "Blur/noise RMSE is higher",
        "C03": "magnitude of blur/noise degradation is similar",
        "C04": "conceals severity-specific differences",
        "C05": "small number of source groups dominate",
        "C06": "arbitrary unseen fonts",
        "C07": "unseen renderers",
        "C08": "human readability",
        "C09": "security or production robustness",
        "C10": "caused by changing the font",
        "C11": "geometric equivalence",
        "C12": "0.5 px elastic rung",
        "C13": "population of all fonts",
    }
    for claim_id, text in required_claims.items():
        row = claims.loc[claims["claim_id"] == claim_id].iloc[0]
        assert text in str(row["claim"])
        assert claim_id in report
    assert checks["passed"].all()
    assert (result.claim_evidence["classification"] == "supported").sum() == 3
    assert (result.claim_evidence["classification"] == "unsupported").sum() == 10


def test_benchmark_comparison_reproduces_frozen_values(bundle):
    _, _, result, _, _ = bundle
    comparison = result.benchmark_comparison
    assert set(comparison["comparison_type"]) == {"unpaired_descriptive"}
    assert bool(comparison["paired"].any()) is False
    assert bool(comparison["eligible"].all())
    blur = comparison.loc[
        (comparison["m7b_scope"] == "original_blur_noise") & (comparison["metric"] == "rmse")
    ].iloc[0]
    geometric = comparison.loc[
        (comparison["m7b_scope"] == "geometric_primary") & (comparison["metric"] == "rmse")
    ].iloc[0]
    assert math.isclose(float(blur["familiar_estimate"]), 0.023328, rel_tol=1e-4)
    assert math.isclose(float(blur["m7b_estimate"]), 0.095923, rel_tol=1e-4)
    assert math.isclose(float(geometric["familiar_estimate"]), 0.059006, rel_tol=1e-4)
    assert math.isclose(float(geometric["m7b_estimate"]), 0.058839, rel_tol=1e-4)
    assert int(blur["familiar_n_sources"]) == 12
    assert int(blur["m7b_n_sources"]) == 60


def test_frozen_input_immutability(bundle):
    config, _, _, _, _ = bundle
    for pin in config.inputs:
        path = REPO_ROOT / pin.path
        assert sha256_file(path) == pin.sha256
    with pytest.raises(cfa.CrossFontError):
        cfa._safe_output_path(REPO_ROOT / "configs" / "not_allowed")
    with pytest.raises(cfa.CrossFontError):
        cfa._safe_output_path(REPO_ROOT / "data" / "processed" / "not_allowed")
    with pytest.raises(cfa.CrossFontError):
        cfa._safe_output_path(REPO_ROOT / "models" / "not_allowed")
    with pytest.raises(cfa.CrossFontError):
        cfa._safe_output_path(
            REPO_ROOT / "reports" / "heldout_font_benchmark_v0_1_0" / "not_allowed"
        )
    with pytest.raises(cfa.CrossFontError):
        cfa._safe_output_path(REPO_ROOT)
    allowed = cfa._safe_output_path(REPO_ROOT / "reports" / "_m7c_test_output")
    assert allowed.name == "_m7c_test_output"


def test_report_consistency(bundle):
    config, inputs, result, report, checks = bundle
    assert cfa.build_report(inputs, result) == report
    assert all(section in report for section in cfa.REPORT_SECTIONS)
    assert "EXP-0016" in report
    assert "read-only analysis" in report
    assert "human readability" in report
    assert "not a font-population effect" in report
    all_primary = _family_row(result, "all_primary")
    assert f"{float(all_primary['rmse']):.6f}" in report
    assert "verification" not in report.lower() or "verify" in report.lower()
    assert checks["check"].is_unique
    assert checks["passed"].all()


def test_config_identity_and_protocol(bundle):
    config, _, result, _, _ = bundle
    assert config.config_hash
    assert config.config_sha256
    assert result.protocol["independent_unit"] == "source_id"
    assert result.protocol["bootstrap"]["n_bootstrap"] == 2000
    assert result.protocol["bootstrap"]["confidence_level"] == 0.95
    assert result.protocol["claim_ids"] == list(cfa.CLAIM_IDS)
    assert result.protocol["claim_rules"]["concentration"].startswith("top3_share")


def test_metadata_is_json_safe(bundle):
    _, inputs, result, _, _ = bundle
    output_hashes = {"report.md": "a" * 64}
    frozen = {"configs/example.toml": "b" * 64}
    metadata = cfa._build_metadata(
        inputs,
        result,
        output_hashes=output_hashes,
        frozen_assets=frozen,
        frozen_assets_hash=cfa._fingerprint_hash(frozen),
        elapsed_seconds=1.0,
    )
    encoded = json.dumps(metadata, sort_keys=True)
    assert "NaN" not in encoded
    assert metadata["summary"]["claims"]["supported"] == 3
    assert metadata["model"]["model_family"] == "expanded_random_forest"
