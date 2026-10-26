from __future__ import annotations

import math

import pandas as pd
import pytest

from app.lib import analysis, contracts, data
from app.lib import format as fmt
from app.tests.helpers import raises_integrity


def test_known_metrics_render_without_recomputation() -> None:
    assert fmt.fmt_metric("mae", 0.0596072060109396) == "0.0596"
    assert fmt.fmt_metric("rmse", 0.088610428864685) == "0.0886"
    assert fmt.fmt_metric("r2", 0.7328983709293053) == "0.733"
    assert fmt.fmt_residual(-0.1240206784424547) == "\u22120.1240"
    assert fmt.fmt_interval(0.0583040013568827, 0.0609873703029555) == ("0.0583\u20130.0610")
    assert fmt.fmt_pct(0.110833333333) == "11.08%"
    assert fmt.fmt_int(5700) == "5,700"


def test_negative_r2_is_not_clipped() -> None:
    assert fmt.fmt_metric("r2", -0.575158) == "\u22120.575"
    assert fmt.fmt(-0.575158, 3) == "\u22120.575"


def test_optional_or_missing_values_are_labelled_not_imputed() -> None:
    for value in (None, float("nan"), math.inf, -math.inf, pd.NA):
        assert fmt.fmt(value) == "Not reported"
        assert fmt.fmt_feature(value) == "Not reported"

    assert fmt.severity_label("elastic", 1, None) == "s1 \u00b7 0.5 px (diagnostic)"
    assert fmt.severity_label("elastic", 2, 1.25) == "s2 \u00b7 1.25 px"
    assert fmt.severity_label("gaussian_blur", 3, None) == "s3"
    assert fmt.severity_label("clean", 0, None) == "Clean reference (diagnostic)"
    assert fmt.case_label("case-000", None) == "case-000"


def test_family_profile_uses_published_intervals() -> None:
    family = data.read_report_table("m8", "family_metrics.csv")
    uncertainty = data.read_report_table("m8", "uncertainty.csv")
    profile = analysis.family_profile(family, uncertainty)

    assert profile["label"].tolist() == [
        "Gaussian blur",
        "Gaussian noise",
        "Thin lines",
        "Arc strokes",
        "Elastic",
    ]
    assert profile["mae"].tolist() == pytest.approx(
        [0.026499, 0.038971, 0.102073, 0.079578, 0.048018],
        abs=1e-5,
    )
    assert profile["r2"].tolist() == pytest.approx(
        [0.943091, 0.735669, -0.575158, -0.298911, 0.162498],
        abs=1e-5,
    )
    assert (profile["mae_ci_low"] <= profile["mae"]).all()
    assert (profile["mae"] <= profile["mae_ci_high"]).all()


def test_family_profile_fails_on_missing_intervals() -> None:
    family = data.read_report_table("m8", "family_metrics.csv")
    uncertainty = data.read_report_table("m8", "uncertainty.csv")
    missing = uncertainty[
        ~(
            (uncertainty["scheme"] == "source_grouped")
            & (uncertainty["scope"] == "thin_lines")
            & (uncertainty["metric"] == "r2")
        )
    ]

    with raises_integrity("intervals are incomplete"):
        analysis.family_profile(family, missing)


def test_empty_family_and_severity_tables_fail_closed() -> None:
    family = data.read_report_table("m8", "family_metrics.csv")
    uncertainty = data.read_report_table("m8", "uncertainty.csv")
    with raises_integrity("family metrics are empty"):
        analysis.family_profile(family.iloc[0:0], uncertainty.iloc[0:0])

    severity = data.read_report_table("m9", "severity_metrics.csv")
    with raises_integrity("severity metrics are empty"):
        analysis.severity_long(severity.iloc[0:0])


def test_severity_calibration_concentration_and_fonts() -> None:
    severity = data.read_report_table("m9", "severity_metrics.csv")
    long = analysis.severity_long(severity)
    assert len(severity) == 19
    assert len(long) == 38
    assert set(long["series"]) == {"Target mean", "Prediction mean"}
    assert set(long["family"]) == set(fmt.PRIMARY_FAMILY_ORDER)

    calibration = analysis.calibration(data.read_report_table("m9", "condition_calibration.csv"))
    assert len(calibration) == 4
    assert calibration["label"].tolist() == [
        "Gaussian blur",
        "Gaussian noise",
        "Thin lines",
        "Arc strokes",
    ]
    assert calibration["within_source_prediction_target_sd_ratio_median"].min() == pytest.approx(
        0.319897815607, abs=1e-9
    )

    concentration = analysis.concentration(data.read_report_table("m9", "error_concentration.csv"))
    assert len(concentration) == 12
    assert concentration["tier"].tolist().count("Top 5%") == 4
    top_five = concentration[concentration["tier"] == "Top 5%"]
    assert top_five["share"].max() == pytest.approx(0.514609, abs=1e-6)

    fonts = analysis.font_interference(data.read_report_table("m9", "font_condition_metrics.csv"))
    assert len(fonts) == 10
    assert fonts.iloc[0]["label"] == "Impact"
    assert fonts.iloc[-1]["label"] == "Courier New"
    assert fonts.iloc[0]["mae"] == pytest.approx(0.035837, abs=1e-6)
    assert fonts.iloc[-1]["mae"] == pytest.approx(0.130245, abs=1e-6)


def test_optional_tables_fail_closed_when_incomplete() -> None:
    calibration = data.read_report_table("m9", "condition_calibration.csv")
    with raises_integrity("calibration table is missing"):
        analysis.calibration(calibration[calibration["scope"] != "thin_lines"])

    concentration = data.read_report_table("m9", "error_concentration.csv")
    with raises_integrity("concentration table is missing"):
        analysis.concentration(concentration[concentration["scope"] != "arc_strokes"])

    fonts = data.read_report_table("m9", "font_condition_metrics.csv")
    with raises_integrity("interference font table is empty"):
        analysis.font_interference(fonts.iloc[0:0])

    short_roster = fonts[fonts["scope"] == "interference"].iloc[:-1]
    with raises_integrity("full roster"):
        analysis.font_interference(short_roster)


def test_contract_constants_match_the_release() -> None:
    assert contracts.EXPECTED_ROWS == 6_300
    assert contracts.EXPECTED_PRIMARY_ROWS == 5_700
    assert contracts.EXPECTED_DIAGNOSTIC_ROWS == 600
    assert contracts.EXPECTED_SOURCES == 300
    assert contracts.EXPECTED_HEIGHT == 64
    assert contracts.EXPECTED_WIDTH == 192
    assert set(contracts.FONT_IDS) == {
        "arial",
        "courier_new",
        "times_new_roman",
        "georgia",
        "monaco",
        "impact",
        "verdana",
        "trebuchet_ms",
        "chalkduster",
        "luminari",
    }
