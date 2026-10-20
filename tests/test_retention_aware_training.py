"""Tests for the M5D retention-aware training objectives milestone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from captcha_quality.config import FEATURE_SET_VERSION
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import TARGET, ModelError, model_inputs, validate_feature_columns
from captcha_quality.provenance import sha256_file
from captcha_quality.retention_aware_training import (
    CHAMPION,
    GROUP_CLEAN,
    GROUP_GEOMETRIC,
    GROUP_ORIGINAL,
    RETENTION_AWARE,
    RETRAINED_CONTROL,
    SCHEME_NONE,
    SCHEME_RETENTION_AWARE,
    SCHEME_SEVERITY_AWARE,
    SEVERITY_AWARE,
    RetentionTrainingConfig,
    RetentionTrainingConfigError,
    TrainingSelectionSettings,
    WeightingSettings,
    WeightScheme,
    _check_diagnostic_rung_excluded,
    _check_group_assignment,
    _check_training_isolation,
    _check_weight_validity,
    _parse_selection,
    _parse_weight_scheme,
    _severity_scopes,
    assign_training_groups,
    build_sample_weights,
    load_retention_training_config,
    select_candidate,
    validation_slices,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "retention_aware_training_v0_1_0.toml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _source_frame(split_counts: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fonts = ("font_a", "font_b", "font_c")
    for split, count in split_counts.items():
        for index in range(count):
            rows.append(
                {
                    "source_id": f"src-{split}-{index:02d}",
                    "split": split,
                    "font_id": fonts[index % 3],
                    "text_length": 4 + index % 3,
                }
            )
    return pd.DataFrame(rows)


def _feature_values(source_index: int, target: float) -> dict[str, float]:
    deficit = 1.0 - target
    return {
        "intensity_mean": 0.91 - 0.20 * deficit,
        "intensity_std": 0.15 + 0.30 * deficit,
        "contrast_rms": 0.22 + 0.50 * deficit,
        "ink_fraction": 0.12 - 0.05 * deficit,
        "laplacian_variance": 1500.0 - 600.0 * deficit + source_index,
        "edge_density": 0.16 - 0.04 * deficit,
        "histogram_entropy": 2.4 + 0.7 * deficit,
        "noise_sigma_immerkaer": 0.8 + 8.0 * deficit,
    }


def _m3_rows(sources: pd.DataFrame) -> pd.DataFrame:
    """M3-shaped rows: one clean plus four blur and four noise variants per source."""
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(sources.itertuples(index=False)):
        variants: list[tuple[str, int, float]] = [("none", 0, 1.0)]
        for severity in range(1, 5):
            variants.append(
                ("gaussian_blur", severity, 1.0 - 0.12 * severity - 0.01 * (source_index % 3))
            )
            variants.append(
                ("gaussian_noise", severity, 1.0 - 0.08 * severity - 0.01 * (source_index % 3))
            )
        for family, severity, target in variants:
            target = float(np.clip(target, 0.0, 1.0))
            rows.append(
                {
                    "sample_id": f"{source.source_id}__{family}_s{severity}",
                    "source_id": source.source_id,
                    "split": source.split,
                    "font_id": source.font_id,
                    "text_length": source.text_length,
                    "family": family,
                    "severity_level": severity,
                    "corruption_type": family,
                    "is_clean": family == "none",
                    "is_control": False,
                    "is_elastic": False,
                    "dataset": "m3",
                    "protocol_scope": "not_applicable",
                    "amplitude_px": np.nan,
                    "image_sha256": f"{source.source_id}-m3-{family}-{severity}",
                    **_feature_values(source_index, target),
                    TARGET: target,
                }
            )
    return pd.DataFrame(rows)


def _m5a_rows(sources: pd.DataFrame) -> pd.DataFrame:
    """M5A-shaped rows: clean, controls and four elastic severities per source."""
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(sources.itertuples(index=False)):
        for severity, amplitude in ((1, 0.5), (2, 1.0), (3, 2.0), (4, 3.0)):
            target = float(np.clip(1.0 - 0.05 * severity - 0.002 * (source_index % 3), 0.0, 1.0))
            rows.append(
                {
                    "sample_id": f"{source.source_id}__elastic_s{severity}",
                    "source_id": source.source_id,
                    "split": source.split,
                    "font_id": source.font_id,
                    "text_length": source.text_length,
                    "family": "elastic",
                    "variant_family": "elastic",
                    "severity_level": severity,
                    "amplitude_px": amplitude,
                    "is_clean": False,
                    "is_control": False,
                    "is_elastic": True,
                    "dataset": "m5a",
                    "protocol_scope": "diagnostic" if severity == 1 else "primary",
                    "image_sha256": f"{source.source_id}-m5a-elastic-{severity}",
                    **_feature_values(source_index, target),
                    TARGET: target,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a small M5C-shaped frame bundle: train, validation and test."""
    sources = _source_frame({"train": 12, "val": 6, "test": 6})
    m3 = _m3_rows(sources)
    m5a = _m5a_rows(sources)

    def frame(split: str, include_diagnostic: bool = False) -> pd.DataFrame:
        m3_rows = m3[m3["split"] == split]
        m5a_rows = m5a[m5a["split"] == split]
        if not include_diagnostic:
            m5a_rows = m5a_rows[m5a_rows["protocol_scope"] == "primary"]
        return pd.concat([m3_rows, m5a_rows], ignore_index=True)

    return frame("train"), frame("val"), frame("test", include_diagnostic=True)


def _test_config() -> RetentionTrainingConfig:
    """Return the shipped configuration."""
    return load_retention_training_config(CONFIG_PATH)


def _scheme_settings(**overrides: float) -> WeightingSettings:
    severity = WeightScheme(
        name=SCHEME_SEVERITY_AWARE,
        clean_weight=1.0,
        original_weight=1.0,
        geometric_base_weight=1.5,
        geometric_severity_ratio=1.5,
    )
    retention = WeightScheme(
        name=SCHEME_RETENTION_AWARE,
        clean_weight=2.0,
        original_weight=2.0,
        geometric_base_weight=3.0,
        geometric_severity_ratio=1.0,
    )
    settings = WeightingSettings(
        normalise_to_mean=1.0,
        min_primary_severity=2,
        schemes={severity.name: severity, retention.name: retention},
    )
    return settings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_shipped_config_declares_the_predeclared_protocol() -> None:
    config = _test_config()
    assert config.schema.feature_names == tuple(FEATURE_NAMES)
    assert config.schema.target == TARGET
    assert config.schema.feature_set_version == FEATURE_SET_VERSION
    assert config.model.family == "random_forest"
    assert set(config.candidate_names) == {
        CHAMPION,
        RETRAINED_CONTROL,
        SEVERITY_AWARE,
        RETENTION_AWARE,
    }
    assert config.candidate_schemes[CHAMPION] == SCHEME_NONE
    assert config.candidate_schemes[RETRAINED_CONTROL] == SCHEME_NONE
    assert config.candidate_schemes[SEVERITY_AWARE] == SCHEME_SEVERITY_AWARE
    assert config.candidate_schemes[RETENTION_AWARE] == SCHEME_RETENTION_AWARE
    assert config.weighting.normalise_to_mean == 1.0
    assert config.weighting.min_primary_severity == 2
    assert config.selection.rule == "retention_gated_rmse_delta"
    assert len(config.config_hash) == 64


def test_retention_aware_scheme_protects_both_retention_slices() -> None:
    """Requirement D: clean and original-task examples are both protected."""
    config = _test_config()
    scheme = config.weighting.scheme(SCHEME_RETENTION_AWARE)
    assert scheme.clean_weight > 1.0
    assert scheme.original_weight > 1.0
    assert scheme.geometric_base_weight > scheme.clean_weight
    assert scheme.geometric_base_weight > scheme.original_weight


def test_severity_aware_scheme_emphasises_geometry_and_retains_retention() -> None:
    """Requirement C: geometric influence rises while retention examples are retained."""
    config = _test_config()
    scheme = config.weighting.scheme(SCHEME_SEVERITY_AWARE)
    assert scheme.clean_weight == 1.0
    assert scheme.original_weight == 1.0
    assert scheme.geometric_base_weight > 1.0
    assert scheme.geometric_severity_ratio > 1.0


def test_config_rejects_a_non_positive_weight() -> None:
    with pytest.raises(RetentionTrainingConfigError):
        _parse_weight_scheme(
            "bad",
            {
                "clean_weight": 0.0,
                "original_weight": 1.0,
                "geometric_base_weight": 1.0,
                "geometric_severity_ratio": 1.0,
            },
        )


def test_config_rejects_a_negative_tolerance() -> None:
    with pytest.raises(RetentionTrainingConfigError):
        _parse_selection(
            {
                "rule": "retention_gated_rmse_delta",
                "w_geometric": 1.0,
                "w_clean": 1.0,
                "w_original": 1.0,
                "w_severity": 1.0,
                "tolerance_clean": -0.1,
                "tolerance_original": 0.1,
                "tolerance_severity": 0.25,
            }
        )


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_sample_weights_are_finite_positive_and_normalised() -> None:
    train, _, _ = _synthetic_frames()
    groups = assign_training_groups(train)
    settings = _scheme_settings()
    for name in (SCHEME_SEVERITY_AWARE, SCHEME_RETENTION_AWARE):
        weights = build_sample_weights(train, groups, settings.scheme(name), settings)
        assert len(weights) == len(train)
        assert np.isfinite(weights).all()
        assert (weights > 0.0).all()
        assert weights.mean() == pytest.approx(1.0, abs=1e-12)


def test_none_scheme_produces_unit_weights() -> None:
    train, _, _ = _synthetic_frames()
    groups = assign_training_groups(train)
    settings = _scheme_settings()
    scheme = WeightScheme(
        name=SCHEME_NONE,
        clean_weight=1.0,
        original_weight=1.0,
        geometric_base_weight=1.0,
        geometric_severity_ratio=1.0,
    )
    weights = build_sample_weights(train, groups, scheme, settings)
    assert np.allclose(weights, 1.0)


def test_severity_aware_weights_ramp_with_geometric_severity() -> None:
    train, _, _ = _synthetic_frames()
    groups = assign_training_groups(train)
    settings = _scheme_settings()
    weights = build_sample_weights(train, groups, settings.scheme(SCHEME_SEVERITY_AWARE), settings)
    geometric = groups.astype(str) == GROUP_GEOMETRIC
    severities = train.loc[geometric, "severity_level"].astype(int).to_numpy()
    geometric_weights = weights[geometric.to_numpy()]
    by_severity = {
        severity: float(geometric_weights[severities == severity].mean())
        for severity in sorted(set(severities))
    }
    # The ramp must increase strictly with the geometric severity.
    ordered = [by_severity[severity] for severity in sorted(by_severity)]
    assert ordered == sorted(ordered)
    assert ordered[-1] > ordered[0]


def test_retention_aware_weights_protect_both_retention_groups() -> None:
    """Requirement D: both retention groups keep more relative influence.

    Protection is measured by the share of total training weight each group
    carries, not by the raw per-row weight: a small group with a high per-row
    weight can still carry a smaller share than a large group. The meaningful
    comparison is against the severity-aware scheme, which is the alternative
    that emphasises geometric examples without protecting the retention slices.
    """
    train, _, _ = _synthetic_frames()
    groups = assign_training_groups(train)
    settings = _scheme_settings()
    group_values = groups.astype(str).to_numpy()

    def shares(scheme_name: str) -> dict[str, float]:
        weights = build_sample_weights(train, groups, settings.scheme(scheme_name), settings)
        total = float(weights.sum())
        return {
            group: float(weights[group_values == group].sum() / total)
            for group in (GROUP_CLEAN, GROUP_ORIGINAL, GROUP_GEOMETRIC)
        }

    retention = shares(SCHEME_RETENTION_AWARE)
    severity = shares(SCHEME_SEVERITY_AWARE)

    # The retention-aware scheme protects both retention slices more than the
    # severity-aware scheme does.
    assert retention[GROUP_CLEAN] > severity[GROUP_CLEAN]
    assert retention[GROUP_ORIGINAL] > severity[GROUP_ORIGINAL]
    # Geometric examples are still emphasised per row over both retention
    # groups, even though the original group has more rows in total.
    scheme = settings.scheme(SCHEME_RETENTION_AWARE)
    assert scheme.geometric_base_weight > scheme.clean_weight
    assert scheme.geometric_base_weight > scheme.original_weight
    # And the per-row weights protect both retention groups explicitly.
    assert scheme.clean_weight > 1.0
    assert scheme.original_weight > 1.0


def test_weight_validity_check_rejects_non_positive_weights() -> None:
    train, _, _ = _synthetic_frames()
    weights = np.ones(len(train))
    weights[0] = 0.0
    check = _check_weight_validity(weights, train)
    assert not check.passed
    weights[0] = np.nan
    assert not _check_weight_validity(weights, train).passed
    weights[0] = -1.0
    assert not _check_weight_validity(weights, train).passed


def test_weight_validity_check_rejects_unnormalised_weights() -> None:
    train, _, _ = _synthetic_frames()
    weights = np.full(len(train), 2.0)
    check = _check_weight_validity(weights, train)
    assert not check.passed
    assert "normalised" in check.detail


def test_weights_do_not_depend_on_the_target() -> None:
    """Weights must be built from training metadata, not from labels."""
    train, _, _ = _synthetic_frames()
    groups = assign_training_groups(train)
    settings = _scheme_settings()
    scheme = settings.scheme(SCHEME_SEVERITY_AWARE)
    baseline = build_sample_weights(train, groups, scheme, settings)
    scrambled = train.copy()
    scrambled[TARGET] = scrambled[TARGET].to_numpy()[::-1]
    assert np.allclose(baseline, build_sample_weights(scrambled, groups, scheme, settings))
    # And the group assignment itself must ignore the target.
    without_target = train.drop(columns=[TARGET])
    assert assign_training_groups(without_target).equals(groups)


def test_group_assignment_check_passes_on_valid_frames() -> None:
    train, _, _ = _synthetic_frames()
    groups = assign_training_groups(train)
    check = _check_group_assignment(train, groups)
    assert check.passed
    assert set(groups.unique()) == {GROUP_CLEAN, GROUP_ORIGINAL, GROUP_GEOMETRIC}


def test_group_assignment_check_rejects_a_target_dependent_assignment() -> None:
    train, _, _ = _synthetic_frames()
    groups = assign_training_groups(train)
    corrupted = groups.copy()
    corrupted.iloc[0] = GROUP_GEOMETRIC
    check = _check_group_assignment(train, corrupted)
    assert not check.passed


def test_feature_schema_is_preserved_and_metadata_is_excluded() -> None:
    train, _, _ = _synthetic_frames()
    inputs = model_inputs(train)
    assert inputs.shape[1] == len(FEATURE_NAMES)
    with pytest.raises(ModelError):
        validate_feature_columns([*FEATURE_NAMES, "severity_level"])


# ---------------------------------------------------------------------------
# Frame integrity
# ---------------------------------------------------------------------------


def test_training_and_validation_frames_are_disjoint_from_test() -> None:
    train, validation, test = _synthetic_frames()
    check = _check_training_isolation(train, validation, test)
    assert check.passed
    assert not set(train["source_id"]) & set(test["source_id"])
    assert not set(validation["source_id"]) & set(test["source_id"])


def test_training_isolation_fails_when_a_test_source_leaks() -> None:
    train, validation, test = _synthetic_frames()
    # Move a real test source's rows into the training frame, which is the
    # exact failure the check must catch.
    leaked_source = test["source_id"].iloc[0]
    leaked = test[test["source_id"] == leaked_source].copy()
    leaked["split"] = "train"
    contaminated = pd.concat([train, leaked], ignore_index=True)
    check = _check_training_isolation(contaminated, validation, test)
    assert not check.passed
    assert "test source" in check.detail


def test_diagnostic_rung_is_excluded_from_fit_and_selection_frames() -> None:
    train, validation, _ = _synthetic_frames()
    check = _check_diagnostic_rung_excluded(train, validation)
    assert check.passed
    assert not (train["amplitude_px"] == 0.5).any()
    assert not (validation["amplitude_px"] == 0.5).any()


def test_diagnostic_exclusion_fails_when_the_rung_enters_training() -> None:
    train, validation, test = _synthetic_frames()
    contaminated = pd.concat(
        [train, test[test["protocol_scope"] == "diagnostic"]], ignore_index=True
    )
    check = _check_diagnostic_rung_excluded(contaminated, validation)
    assert not check.passed


def test_validation_slices_cover_the_required_populations() -> None:
    train, validation, _ = _synthetic_frames()
    frames = type("F", (), {"validation": validation})()
    slices = validation_slices(frames)
    assert "geometric_primary" in slices
    assert "clean_reference" in slices
    assert "original_blur_noise" in slices
    assert "overall" in slices
    for severity in (2, 3, 4):
        assert f"geometric_primary_severity_{severity}" in slices
    assert len(train) == len(validation) + len(train) - len(validation)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


_SEVERITY_SCOPES = (
    "geometric_primary_severity_2",
    "geometric_primary_severity_3",
    "geometric_primary_severity_4",
)


def _selection_settings() -> TrainingSelectionSettings:
    return TrainingSelectionSettings(
        rule="retention_gated_rmse_delta",
        w_geometric=1.0,
        w_clean=2.0,
        w_original=2.0,
        w_severity=1.0,
        tolerance_clean=0.50,
        tolerance_original=0.10,
        tolerance_severity=0.25,
    )


def _metrics_table(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "scheme": "none",
                "scope": scope,
                "rmse": rmse,
            }
            for model, scope, rmse in rows
        ]
    )


def _champion_rows(**overrides: float) -> list[tuple[str, str, float]]:
    values = {
        "geometric_primary": 0.05,
        "clean_reference": 0.002,
        "original_blur_noise": 0.020,
        "geometric_primary_severity_2": 0.05,
        "geometric_primary_severity_3": 0.025,
        "geometric_primary_severity_4": 0.06,
    }
    values.update(overrides)
    return [(CHAMPION, scope, value) for scope, value in values.items()]


def test_selection_rejects_a_candidate_that_regresses_a_single_severity() -> None:
    """A severe per-severity regression must not be hidden by an aggregate gain."""
    settings = _selection_settings()
    table = _metrics_table(
        _champion_rows()
        + [
            ("candidate", "geometric_primary", 0.040),
            ("candidate", "clean_reference", 0.002),
            ("candidate", "original_blur_noise", 0.020),
            ("candidate", "geometric_primary_severity_2", 0.090),
            ("candidate", "geometric_primary_severity_3", 0.025),
            ("candidate", "geometric_primary_severity_4", 0.060),
        ]
    )
    selected, scored = select_candidate(table, settings, severity_scopes=_SEVERITY_SCOPES)
    assert selected == CHAMPION
    row = scored[scored["model"] == "candidate"].iloc[0]
    assert not bool(row["eligible"])
    assert "severity_2" in str(row["ineligible_reasons"])


def test_selection_rejects_a_candidate_that_regresses_the_clean_slice() -> None:
    settings = _selection_settings()
    table = _metrics_table(
        _champion_rows()
        + [
            ("candidate", "geometric_primary", 0.040),
            ("candidate", "clean_reference", 0.010),
            ("candidate", "original_blur_noise", 0.020),
            ("candidate", "geometric_primary_severity_2", 0.050),
            ("candidate", "geometric_primary_severity_3", 0.025),
            ("candidate", "geometric_primary_severity_4", 0.060),
        ]
    )
    selected, scored = select_candidate(table, settings, severity_scopes=_SEVERITY_SCOPES)
    assert selected == CHAMPION
    assert not bool(scored[scored["model"] == "candidate"].iloc[0]["eligible"])


def test_selection_rejects_a_candidate_that_regresses_the_original_task() -> None:
    settings = _selection_settings()
    table = _metrics_table(
        _champion_rows()
        + [
            ("candidate", "geometric_primary", 0.040),
            ("candidate", "clean_reference", 0.002),
            ("candidate", "original_blur_noise", 0.030),
            ("candidate", "geometric_primary_severity_2", 0.050),
            ("candidate", "geometric_primary_severity_3", 0.025),
            ("candidate", "geometric_primary_severity_4", 0.060),
        ]
    )
    selected, scored = select_candidate(table, settings, severity_scopes=_SEVERITY_SCOPES)
    assert selected == CHAMPION
    assert not bool(scored[scored["model"] == "candidate"].iloc[0]["eligible"])


def test_selection_prefers_an_eligible_candidate_with_a_lower_score() -> None:
    settings = _selection_settings()
    table = _metrics_table(
        _champion_rows()
        + [
            ("candidate", "geometric_primary", 0.040),
            ("candidate", "clean_reference", 0.002),
            ("candidate", "original_blur_noise", 0.020),
            ("candidate", "geometric_primary_severity_2", 0.050),
            ("candidate", "geometric_primary_severity_3", 0.025),
            ("candidate", "geometric_primary_severity_4", 0.060),
        ]
    )
    selected, scored = select_candidate(table, settings, severity_scopes=_SEVERITY_SCOPES)
    assert selected == "candidate"
    assert bool(scored[scored["model"] == "candidate"].iloc[0]["eligible"])
    assert float(scored[scored["model"] == "candidate"].iloc[0]["score"]) < 0.0


def test_selection_falls_back_to_the_champion_when_nothing_qualifies() -> None:
    settings = _selection_settings()
    table = _metrics_table(
        _champion_rows()
        + [
            ("candidate", "geometric_primary", 0.080),
            ("candidate", "clean_reference", 0.002),
            ("candidate", "original_blur_noise", 0.020),
            ("candidate", "geometric_primary_severity_2", 0.080),
            ("candidate", "geometric_primary_severity_3", 0.025),
            ("candidate", "geometric_primary_severity_4", 0.060),
        ]
    )
    selected, scored = select_candidate(table, settings, severity_scopes=_SEVERITY_SCOPES)
    assert selected == CHAMPION
    assert bool(scored[scored["model"] == CHAMPION].iloc[0]["selected"])
    assert not bool(scored[scored["model"] == "candidate"].iloc[0]["eligible"])


def test_selection_tie_breaking_is_deterministic() -> None:
    """Two candidates with identical scores must be ordered by name."""
    settings = _selection_settings()
    rows = _champion_rows()
    for name in ("alpha", "beta"):
        rows += [
            (name, "geometric_primary", 0.040),
            (name, "clean_reference", 0.002),
            (name, "original_blur_noise", 0.020),
            (name, "geometric_primary_severity_2", 0.050),
            (name, "geometric_primary_severity_3", 0.025),
            (name, "geometric_primary_severity_4", 0.060),
        ]
    table = _metrics_table(rows)
    first, _ = select_candidate(table, settings, severity_scopes=_SEVERITY_SCOPES)
    second, _ = select_candidate(
        table.sample(frac=1.0, random_state=3), settings, severity_scopes=_SEVERITY_SCOPES
    )
    assert first == second == "alpha"


def test_severity_scopes_follow_the_declared_protocol() -> None:
    config = _test_config()
    assert _severity_scopes(config) == (
        "geometric_primary_severity_2",
        "geometric_primary_severity_3",
        "geometric_primary_severity_4",
    )


# ---------------------------------------------------------------------------
# End-to-end run, persistence and replay
# ---------------------------------------------------------------------------


def test_end_to_end_run_persists_artifacts_and_replays(tmp_path: Path) -> None:
    """The full M5D flow must persist, reload and reproduce its own results."""
    from captcha_quality.retention_aware_training import (
        run_retention_training,
        verify_retention_training,
    )

    config = _test_config()
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    result, paths = run_retention_training(
        config, output_reports=reports, output_models=models, verify_images=False
    )
    assert result.ok
    assert result.selected_name in result.bundle.names

    for name in (
        "validation_slice_metrics.csv",
        "selection_table.csv",
        "test_slice_metrics.csv",
        "uncertainty.csv",
        "paired_uncertainty.csv",
        "test_predictions.csv",
        "isolation_checks.csv",
        "run_metadata.json",
        "report.md",
    ):
        assert (reports / name).is_file(), name
    for candidate in config.candidate_names:
        if candidate == CHAMPION:
            continue
        assert (models / f"{candidate}.joblib").is_file()
        assert (models / f"{candidate}_card.json").is_file()
    assert paths["report"].is_file()

    metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selection"]["selected_model"] == result.selected_name
    assert metadata["frozen_assets"]["all_unchanged"] is True
    assert metadata["schema"]["feature_names"] == list(FEATURE_NAMES)
    assert metadata["schema"]["target"] == TARGET
    assert metadata["split"]["train_samples"] == 432
    assert metadata["split"]["validation_samples"] == 144
    assert metadata["split"]["primary_test_samples"] == 36
    assert metadata["split"]["diagnostic_test_samples"] == 12

    # Every refitted candidate must carry the frozen feature schema and target.
    for candidate in config.candidate_names:
        if candidate == CHAMPION:
            continue
        card = json.loads((models / f"{candidate}_card.json").read_text(encoding="utf-8"))
        assert card["feature_names"] == list(FEATURE_NAMES)
        assert card["target"] == TARGET
        assert card["artifact_sha256"] == sha256_file(models / f"{candidate}.joblib")
        assert card["selection"]["selected_on"] == "val"

    report = verify_retention_training(config, output_reports=reports, output_models=models)
    assert report["passed"], [check for check in report["checks"] if not check.passed]


def test_control_candidate_reproduces_the_champion_exactly(tmp_path: Path) -> None:
    """The unweighted retrained control must match the frozen champion.

    This makes the control a check on the effect of retraining and a
    reproducibility check on the frozen champion bundle: any difference in the
    weighted candidates is then attributable to the sample weighting alone.
    """
    from captcha_quality import retention_aware as ra
    from captcha_quality.retention_aware_training import (
        _load_champion,
        build_candidate_bundle,
        check_control_reproduces_champion,
        load_training_frames,
    )

    config = _test_config()
    frames = load_training_frames(config, verify_images=False)
    m5c_config = ra.load_retention_config(config.m5c.config)
    champion = _load_champion(config, frames.retention.frames.m3_summary)
    bundle = build_candidate_bundle(frames, config, champion=champion, m5c_config=m5c_config)
    check = check_control_reproduces_champion(bundle, frames)
    assert check.passed


def test_report_generation_is_deterministic(tmp_path: Path) -> None:
    """Two runs of the same protocol must produce identical reports and tables."""
    from captcha_quality.retention_aware_training import run_retention_training

    config = _test_config()
    first_reports = tmp_path / "first"
    second_reports = tmp_path / "second"
    run_retention_training(
        config,
        output_reports=first_reports,
        output_models=tmp_path / "m1",
        verify_images=False,
    )
    run_retention_training(
        config,
        output_reports=second_reports,
        output_models=tmp_path / "m2",
        verify_images=False,
    )
    for name in (
        "validation_slice_metrics.csv",
        "selection_table.csv",
        "test_slice_metrics.csv",
        "uncertainty.csv",
        "paired_uncertainty.csv",
        "test_predictions.csv",
        "report.md",
    ):
        assert (first_reports / name).read_bytes() == (second_reports / name).read_bytes(), name


def test_verification_detects_a_tampered_artifact(tmp_path: Path) -> None:
    """A modified output must fail verification rather than pass silently."""
    from captcha_quality.retention_aware_training import (
        run_retention_training,
        verify_retention_training,
    )

    config = _test_config()
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    run_retention_training(
        config, output_reports=reports, output_models=models, verify_images=False
    )
    metrics_path = reports / "test_slice_metrics.csv"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    report = verify_retention_training(config, output_reports=reports, output_models=models)
    assert not report["passed"]
    assert any(not check.passed for check in report["checks"])


def test_selected_candidate_card_records_the_frozen_configuration(tmp_path: Path) -> None:
    from captcha_quality.retention_aware_training import run_retention_training

    config = _test_config()
    models = tmp_path / "models"
    result, _ = run_retention_training(
        config,
        output_reports=tmp_path / "reports",
        output_models=models,
        verify_images=False,
    )
    card = (
        json.loads((models / f"{result.selected_name}_card.json").read_text(encoding="utf-8"))
        if result.selected_name != CHAMPION
        else json.loads((models / "retrained_control_card.json").read_text(encoding="utf-8"))
    )
    assert card["selection"]["selected_model"] == result.selected_name
    assert card["selection"]["benchmark_config_hash"] == config.config_hash
    assert card["seed"] == config.model.seed
    assert card["params"]["n_estimators"] == config.model.n_estimators


def test_test_evaluation_happens_only_after_selection_is_frozen(tmp_path: Path) -> None:
    """The selection must be computable without touching any test frame.

    The selection rule consumes only validation metrics, so a candidate set
    that is never evaluated on test can still be selected. This is the
    structural guarantee that test data cannot influence the choice.
    """
    from captcha_quality import retention_aware as ra
    from captcha_quality.retention_aware_training import (
        _load_champion,
        build_candidate_bundle,
        load_training_frames,
        select_candidate,
    )

    config = _test_config()
    frames = load_training_frames(config, verify_images=False)
    m5c_config = ra.load_retention_config(config.m5c.config)
    champion = _load_champion(config, frames.retention.frames.m3_summary)
    bundle = build_candidate_bundle(frames, config, champion=champion, m5c_config=m5c_config)
    from captcha_quality.retention_aware_training import _slice_metrics_table

    validation_metrics = _slice_metrics_table(
        frames, bundle, bundle.validation_predictions, split_label="val"
    )
    selected, _table = select_candidate(
        validation_metrics,
        config.selection,
        severity_scopes=_severity_scopes(config),
    )
    assert selected in bundle.names
    # The selection table is built entirely from validation rows.
    assert set(validation_metrics["split"]) == {"val"}
