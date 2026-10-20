"""Grouped, leakage-safe feature/target diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from captcha_quality.config import load_config
from captcha_quality.dataset import DatasetError, generate_dataset
from captcha_quality.diagnostics import (
    _classify,
    _modelling_value,
    diagnose_dataset,
    run_diagnostics,
)
from captcha_quality.features import FEATURE_NAMES


@pytest.fixture
def diagnostic_dataset(dataset_config_writer, tmp_path: Path):
    config = load_config(
        dataset_config_writer(output_dir=tmp_path / "diagnostic", sources_per_font=6)
    )
    return config, generate_dataset(config)


def test_diagnostics_run_with_all_leakage_checks_passing(
    diagnostic_dataset, tmp_path: Path
) -> None:
    config, result = diagnostic_dataset
    report = run_diagnostics(
        result.manifest,
        result.features,
        config.diagnostics,
        dataset_name="test",
        dataset_version="v0",
        output_dir=tmp_path / "reports",
    )
    assert report.leakage_safe
    assert all(check.passed for check in report.leakage_checks)
    assert report.report_path is not None and report.report_path.is_file()
    text = report.report_path.read_text(encoding="utf-8")
    assert "not a measurement of human readability" in text
    assert "Classification:" in text


def test_model_table_covers_all_models_and_splits(diagnostic_dataset) -> None:
    config, result = diagnostic_dataset
    report = run_diagnostics(
        result.manifest,
        result.features,
        config.diagnostics,
        dataset_name="test",
        dataset_version="v0",
    )
    models = report.models
    assert set(models["split"]) == {"train", "val", "test"}
    expected = {
        "naive_train_mean",
        "metadata_severity_family",
        "metadata_font",
        "metadata_text_length",
        "all_features_ols",
    } | {f"single_{feature}" for feature in FEATURE_NAMES}
    assert set(models["model"]) == expected
    held_out = models[models["split"].isin(("val", "test"))]
    assert held_out[["rmse", "mae", "r2"]].notna().all().all()


def test_best_single_feature_is_chosen_on_validation(diagnostic_dataset) -> None:
    config, result = diagnostic_dataset
    report = run_diagnostics(
        result.manifest,
        result.features,
        config.diagnostics,
        dataset_name="test",
        dataset_version="v0",
    )
    singles = report.models[report.models["kind"] == "single_feature"]
    val = singles[singles["split"] == "val"].set_index("model")["r2"]
    expected = val.idxmax().removeprefix("single_")
    assert report.best_single_feature == expected
    assert math.isclose(report.best_single_val_r2, float(val.max()))
    test_value = singles[(singles["split"] == "test") & (singles["model"] == f"single_{expected}")][
        "r2"
    ].iloc[0]
    assert math.isclose(report.best_single_test_r2, float(test_value))


def test_target_level_summary_includes_clean_and_corrupted(diagnostic_dataset) -> None:
    config, result = diagnostic_dataset
    report = run_diagnostics(
        result.manifest,
        result.features,
        config.diagnostics,
        dataset_name="test",
        dataset_version="v0",
    )
    levels = report.target_levels
    assert "none" in set(levels["corruption_type"])
    assert set(levels["severity_level"]) >= {0, 1, 2}
    clean = levels[levels["corruption_type"] == "none"].iloc[0]
    assert clean["mean_ssim"] == pytest.approx(1.0)


def test_within_family_correlations_exclude_clean(diagnostic_dataset) -> None:
    config, result = diagnostic_dataset
    report = run_diagnostics(
        result.manifest,
        result.features,
        config.diagnostics,
        dataset_name="test",
        dataset_version="v0",
    )
    families = set(
        report.correlations.loc[report.correlations["scope"] == "train_family", "family"]
    )
    assert "none" not in families


def test_group_overlap_is_rejected(diagnostic_dataset) -> None:
    config, result = diagnostic_dataset
    manifest = result.manifest.copy()
    test_source = manifest.loc[manifest["split"] == "test", "source_id"].iloc[0]
    index = manifest.index[manifest["source_id"] == test_source][0]
    manifest.loc[index, "split"] = "train"
    with pytest.raises(DatasetError, match="leakage checks failed"):
        run_diagnostics(
            manifest,
            result.features,
            config.diagnostics,
            dataset_name="test",
            dataset_version="v0",
        )


def test_missing_feature_rows_are_rejected(diagnostic_dataset) -> None:
    config, result = diagnostic_dataset
    features = result.features.iloc[:-1]
    with pytest.raises(DatasetError, match="feature table"):
        run_diagnostics(
            result.manifest,
            features,
            config.diagnostics,
            dataset_name="test",
            dataset_version="v0",
        )


def test_diagnose_dataset_reads_from_disk(diagnostic_dataset, tmp_path: Path) -> None:
    config, result = diagnostic_dataset
    report = diagnose_dataset(result.root, config.diagnostics, output_dir=tmp_path / "out")
    assert report.leakage_safe
    assert (tmp_path / "out" / "feature_correlations.csv").is_file()
    assert (tmp_path / "out" / "model_diagnostics.csv").is_file()


def test_classification_thresholds() -> None:
    from captcha_quality.config import DiagnosticsSettings

    settings = DiagnosticsSettings(
        trivial_single_feature_r2=0.90, strong_confound_r2=0.70, min_group_rows=3
    )
    assert _classify(0.95, 0.92, settings) == "potentially_trivial"
    assert _classify(0.95, 0.80, settings) == "strong_confound"
    assert _classify(0.60, 0.71, settings) == "strong_confound"
    assert _classify(0.60, 0.50, settings) == "no_single_feature_triviality"
    assert _classify(math.nan, math.nan, settings) == "undetermined"


def test_modelling_value_thresholds() -> None:
    assert _modelling_value(0.95, 0.93) == "feature_only_near_solved"
    assert _modelling_value(0.95, 0.80) == "feature_only_high"
    assert _modelling_value(0.55, 0.50) == "feature_only_limited"


def test_report_does_not_claim_human_solvability(diagnostic_dataset) -> None:
    config, result = diagnostic_dataset
    report = run_diagnostics(
        result.manifest,
        result.features,
        config.diagnostics,
        dataset_name="test",
        dataset_version="v0",
    )
    lowered = report.report_markdown.lower()
    assert "human readability" in lowered
    assert "not a measurement" in lowered
    assert "solvable by humans" not in lowered


def test_correlation_table_has_stable_feature_order(diagnostic_dataset) -> None:
    config, result = diagnostic_dataset
    report = run_diagnostics(
        result.manifest,
        result.features,
        config.diagnostics,
        dataset_name="test",
        dataset_version="v0",
    )
    overall = report.correlations[report.correlations["scope"] == "train_overall"]
    assert tuple(overall["feature"]) == FEATURE_NAMES
    assert pd.api.types.is_float_dtype(overall["pearson"])
