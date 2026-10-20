"""Tests for the M6A spatial-aware feature study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")
pytest.importorskip("skimage")

from captcha_quality.features import FEATURE_NAMES
from captcha_quality.imaging import ImageContractError
from captcha_quality.modelling import TARGET
from captcha_quality.provenance import sha256_file
from captcha_quality.spatial_features import (
    FAMILY_EDGE,
    FAMILY_INK,
    SPATIAL_FEATURE_SET_VERSION,
    SpatialFeatureError,
    SpatialFeatureSpec,
    describe_feature_set,
    extract_spatial_feature_table,
    extract_spatial_features,
    spatial_feature_manifest,
)
from captcha_quality.spatial_study import (
    COMBINED,
    FEATURE_SET_COMBINED,
    FEATURE_SET_GLOBAL,
    FEATURE_SET_SPATIAL,
    GLOBAL_BASELINE,
    SPATIAL_ONLY,
    SpatialStudyConfigError,
    StudySelectionSettings,
    _check_no_metadata_in_spatial_inputs,
    _check_spatial_feature_frame,
    load_spatial_study_config,
    select_ablation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "spatial_feature_study_v0_1_0.toml"


# ---------------------------------------------------------------------------
# Hand-verifiable images
# ---------------------------------------------------------------------------


def _blank(height: int = 64, width: int = 192, value: int = 255) -> np.ndarray:
    """A uniform image."""
    return np.full((height, width), value, dtype=np.uint8)


def _half_dark(height: int = 64, width: int = 192) -> np.ndarray:
    """A left/right split: the left half is dark, the right half is light.

    With a 4x2 grid the left column of cells is entirely dark (ink occupancy
    1.0) and the right column entirely light (ink occupancy 0.0), because the
    Otsu threshold of this image falls between the two levels.
    """
    image = _blank(height, width)
    image[:, : width // 2] = 0
    return image


def _uniform_gray(height: int = 64, width: int = 192, value: int = 128) -> np.ndarray:
    """A uniform image that is neither blank-white nor black."""
    return np.full((height, width), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------


def test_feature_names_order_and_count_are_exact() -> None:
    spec = SpatialFeatureSpec(grid_rows=4, grid_cols=2)
    names = spec.feature_names
    assert len(names) == 16
    assert len(set(names)) == 16
    # Row-major order: the row index changes slowest.
    assert names[0] == "ink_occupancy_r0_c0"
    assert names[1] == "ink_occupancy_r0_c1"
    assert names[2] == "ink_occupancy_r1_c0"
    assert names[7] == "ink_occupancy_r3_c1"
    assert names[8] == "edge_density_r0_c0"
    assert names[15] == "edge_density_r3_c1"
    assert spec.n_cells == 8


def test_feature_family_order_follows_the_declared_families() -> None:
    spec = SpatialFeatureSpec(families=(FAMILY_EDGE, FAMILY_INK))
    names = spec.feature_names
    assert names[0].startswith("edge_density")
    assert names[8].startswith("ink_occupancy")


def test_feature_manifest_matches_the_feature_names() -> None:
    spec = SpatialFeatureSpec()
    manifest = spatial_feature_manifest(spec)
    assert list(manifest["feature_name"]) == list(spec.feature_names)
    assert set(manifest["family"]) == {FAMILY_INK, FAMILY_EDGE}
    assert manifest["grid_row"].max() == spec.grid_rows - 1
    assert manifest["grid_col"].max() == spec.grid_cols - 1


def test_feature_version_is_declared() -> None:
    assert SPATIAL_FEATURE_SET_VERSION == "spatial-v1"


def test_spec_rejects_an_unknown_family() -> None:
    with pytest.raises(SpatialFeatureError):
        SpatialFeatureSpec(families=("not_a_family",))


def test_spec_rejects_a_non_positive_grid() -> None:
    with pytest.raises(SpatialFeatureError):
        SpatialFeatureSpec(grid_rows=0)


def test_spec_rejects_an_out_of_range_edge_threshold() -> None:
    with pytest.raises(SpatialFeatureError):
        SpatialFeatureSpec(edge_low_threshold=1.5)


# ---------------------------------------------------------------------------
# Extraction behaviour on hand-verifiable images
# ---------------------------------------------------------------------------


def test_uniform_image_yields_all_zero_spatial_features() -> None:
    """A blank image has no distinguishable ink class and no edges."""
    spec = SpatialFeatureSpec()
    features = extract_spatial_features(_blank(), spec)
    assert set(features) == set(spec.feature_names)
    assert all(value == 0.0 for value in features.values())


def test_uniform_gray_image_also_yields_zeros() -> None:
    spec = SpatialFeatureSpec()
    features = extract_spatial_features(_uniform_gray(), spec)
    assert all(value == 0.0 for value in features.values())


def test_half_dark_image_has_ink_only_in_the_left_column() -> None:
    """The left half is dark, so the left cells are full of ink."""
    spec = SpatialFeatureSpec()
    features = extract_spatial_features(_half_dark(), spec)
    for row in range(spec.grid_rows):
        left = features[f"{FAMILY_INK}_r{row}_c0"]
        right = features[f"{FAMILY_INK}_r{row}_c1"]
        assert left == pytest.approx(1.0)
        assert right == pytest.approx(0.0)


def test_extraction_is_deterministic() -> None:
    spec = SpatialFeatureSpec()
    image = _half_dark()
    first = extract_spatial_features(image, spec)
    second = extract_spatial_features(image, spec)
    assert first == second


def test_extracted_values_are_finite_and_in_range() -> None:
    spec = SpatialFeatureSpec()
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, size=(64, 192), dtype=np.uint8)
    features = extract_spatial_features(image, spec)
    for name, value in features.items():
        assert np.isfinite(value), name
        assert 0.0 <= value <= 1.0, name


def test_extraction_rejects_a_violating_image() -> None:
    with pytest.raises(ImageContractError):
        extract_spatial_features(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ImageContractError):
        extract_spatial_features(np.zeros((64, 192), dtype=np.float64))


def test_empty_or_degenerate_grid_cell_yields_zero_not_an_error() -> None:
    """A grid finer than the image would produce empty cells; they must be zero."""
    spec = SpatialFeatureSpec(grid_rows=64, grid_cols=192)
    features = extract_spatial_features(_half_dark(height=16, width=32), spec)
    assert all(np.isfinite(value) for value in features.values())
    assert all(0.0 <= value <= 1.0 for value in features.values())


def test_batch_extraction_preserves_column_order() -> None:
    spec = SpatialFeatureSpec()
    table = extract_spatial_feature_table([("a", _half_dark()), ("b", _blank())], spec)
    assert list(table.columns) == ["sample_id", *spec.feature_names]
    assert list(table["sample_id"]) == ["a", "b"]


def test_batch_extraction_rejects_duplicate_sample_ids() -> None:
    spec = SpatialFeatureSpec()
    with pytest.raises(SpatialFeatureError):
        extract_spatial_feature_table([("a", _blank()), ("a", _blank())], spec)


def test_batch_extraction_rejects_an_empty_sample_id() -> None:
    spec = SpatialFeatureSpec()
    with pytest.raises(SpatialFeatureError):
        extract_spatial_feature_table([("", _blank())], spec)


# ---------------------------------------------------------------------------
# Leakage prevention and schema validation
# ---------------------------------------------------------------------------


def test_spatial_feature_frame_check_accepts_a_valid_table() -> None:
    spec = SpatialFeatureSpec()
    table = extract_spatial_feature_table([("a", _half_dark()), ("b", _blank())], spec)
    check = _check_spatial_feature_frame(table, spec)
    assert check.passed


def test_spatial_feature_frame_check_rejects_a_wrong_column_order() -> None:
    spec = SpatialFeatureSpec()
    table = extract_spatial_feature_table([("a", _half_dark())], spec)
    shuffled = table[list(reversed(table.columns))]
    assert not _check_spatial_feature_frame(shuffled, spec).passed


def test_spatial_feature_frame_check_rejects_non_finite_values() -> None:
    spec = SpatialFeatureSpec()
    table = extract_spatial_feature_table([("a", _half_dark())], spec)
    broken = table.copy()
    broken.iloc[0, 1] = np.nan
    assert not _check_spatial_feature_frame(broken, spec).passed


def test_spatial_feature_frame_check_rejects_out_of_range_values() -> None:
    spec = SpatialFeatureSpec()
    table = extract_spatial_feature_table([("a", _half_dark())], spec)
    broken = table.copy()
    broken.iloc[0, 1] = 5.0
    assert not _check_spatial_feature_frame(broken, spec).passed


def test_metadata_and_target_leakage_is_detected() -> None:
    spec = SpatialFeatureSpec()
    table = extract_spatial_feature_table([("a", _half_dark())], spec)
    assert _check_no_metadata_in_spatial_inputs(table, spec).passed
    for column in (TARGET, "severity_level", "source_id", "split", "font_id", "image_path"):
        leaked = table.copy()
        leaked[column] = "x"
        assert not _check_no_metadata_in_spatial_inputs(leaked, spec).passed, column


def test_the_frozen_global_feature_schema_is_untouched() -> None:
    """The eight global features and their order must not have changed."""
    assert FEATURE_NAMES == (
        "intensity_mean",
        "intensity_std",
        "contrast_rms",
        "ink_fraction",
        "laplacian_variance",
        "edge_density",
        "histogram_entropy",
        "noise_sigma_immerkaer",
    )


def test_describe_feature_set_counts_the_families() -> None:
    spec = SpatialFeatureSpec()
    described = describe_feature_set(spec.feature_names, list(FEATURE_NAMES))
    assert described["n_features"] == 16
    assert described["n_global"] == 0
    assert described["n_spatial"] == 16
    combined = describe_feature_set(
        list(FEATURE_NAMES) + list(spec.feature_names), list(FEATURE_NAMES)
    )
    assert combined["n_features"] == 24
    assert combined["n_global"] == 8
    assert combined["n_spatial"] == 16


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_shipped_config_declares_the_predeclared_protocol() -> None:
    config = load_spatial_study_config(CONFIG_PATH)
    assert config.global_feature_names == tuple(FEATURE_NAMES)
    assert config.global_feature_set_version == "v1"
    assert config.target == TARGET
    assert config.model.family == "random_forest"
    assert config.features.grid_rows == 4
    assert config.features.grid_cols == 2
    assert set(config.features.families) == {FAMILY_INK, FAMILY_EDGE}
    assert set(config.ablation_names) == {GLOBAL_BASELINE, SPATIAL_ONLY, COMBINED}
    assert config.ablation_feature_sets[GLOBAL_BASELINE] == FEATURE_SET_GLOBAL
    assert config.ablation_feature_sets[SPATIAL_ONLY] == FEATURE_SET_SPATIAL
    assert config.ablation_feature_sets[COMBINED] == FEATURE_SET_COMBINED
    assert config.ablation_refit[GLOBAL_BASELINE] is False
    assert config.ablation_refit[SPATIAL_ONLY] is True
    assert config.ablation_refit[COMBINED] is True
    assert config.selection.rule == "retention_gated_rmse_delta"
    assert len(config.config_hash) == 64


def test_config_rejects_a_modified_global_feature_schema() -> None:
    """The eight frozen features must not be altered by this study."""
    import tomllib

    from captcha_quality.spatial_study import (
        load_spatial_study_config,
    )

    payload = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["schema"]["global_feature_names"] = [
        "intensity_mean",
        "intensity_std",
    ]
    broken = CONFIG_PATH.parent / "broken_spatial.toml"
    broken.write_text(_dump_toml(payload), encoding="utf-8")
    try:
        with pytest.raises(SpatialStudyConfigError):
            load_spatial_study_config(broken)
    finally:
        broken.unlink()


def test_config_rejects_refitting_the_global_baseline() -> None:
    import tomllib

    from captcha_quality.spatial_study import load_spatial_study_config

    payload = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["ablations"]["refit"]["global_baseline"] = True
    broken = CONFIG_PATH.parent / "broken_spatial2.toml"
    broken.write_text(_dump_toml(payload), encoding="utf-8")
    try:
        with pytest.raises(SpatialStudyConfigError):
            load_spatial_study_config(broken)
    finally:
        broken.unlink()


def _dump_toml(payload: dict) -> str:
    """Serialise the small config payload back to TOML for the negative tests."""
    lines: list[str] = []
    for section, body in payload.items():
        lines.append(f"[{section}]")
        for key, value in body.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    lines.append(f"{key}.{inner_key} = {_toml_value(inner_value)}")
            else:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def _settings() -> StudySelectionSettings:
    return StudySelectionSettings(
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
                "feature_set": "global",
                "n_features": 8,
                "scope": scope,
                "rmse": rmse,
            }
            for model, scope, rmse in rows
        ]
    )


_SEVERITY_SCOPES = (
    "geometric_primary_severity_2",
    "geometric_primary_severity_3",
    "geometric_primary_severity_4",
)


def _baseline_rows(**overrides: float) -> list[tuple[str, str, float]]:
    values = {
        "geometric_primary": 0.05,
        "clean_reference": 0.002,
        "original_blur_noise": 0.020,
        "geometric_primary_severity_2": 0.05,
        "geometric_primary_severity_3": 0.025,
        "geometric_primary_severity_4": 0.06,
    }
    values.update(overrides)
    return [(GLOBAL_BASELINE, scope, value) for scope, value in values.items()]


def test_selection_rejects_a_candidate_that_regresses_a_single_severity() -> None:
    table = _metrics_table(
        _baseline_rows()
        + [
            ("candidate", "geometric_primary", 0.040),
            ("candidate", "clean_reference", 0.002),
            ("candidate", "original_blur_noise", 0.020),
            ("candidate", "geometric_primary_severity_2", 0.090),
            ("candidate", "geometric_primary_severity_3", 0.025),
            ("candidate", "geometric_primary_severity_4", 0.060),
        ]
    )
    selected, scored = select_ablation(table, _settings(), severity_scopes=_SEVERITY_SCOPES)
    assert selected == GLOBAL_BASELINE
    row = scored[scored["model"] == "candidate"].iloc[0]
    assert not bool(row["eligible"])
    assert "severity_2" in str(row["ineligible_reasons"])


def test_selection_rejects_a_candidate_that_regresses_the_original_task() -> None:
    table = _metrics_table(
        _baseline_rows()
        + [
            ("candidate", "geometric_primary", 0.040),
            ("candidate", "clean_reference", 0.002),
            ("candidate", "original_blur_noise", 0.030),
            ("candidate", "geometric_primary_severity_2", 0.050),
            ("candidate", "geometric_primary_severity_3", 0.025),
            ("candidate", "geometric_primary_severity_4", 0.060),
        ]
    )
    selected, scored = select_ablation(table, _settings(), severity_scopes=_SEVERITY_SCOPES)
    assert selected == GLOBAL_BASELINE
    assert not bool(scored[scored["model"] == "candidate"].iloc[0]["eligible"])


def test_selection_prefers_an_eligible_candidate_with_a_lower_score() -> None:
    table = _metrics_table(
        _baseline_rows()
        + [
            ("candidate", "geometric_primary", 0.040),
            ("candidate", "clean_reference", 0.002),
            ("candidate", "original_blur_noise", 0.020),
            ("candidate", "geometric_primary_severity_2", 0.050),
            ("candidate", "geometric_primary_severity_3", 0.025),
            ("candidate", "geometric_primary_severity_4", 0.060),
        ]
    )
    selected, scored = select_ablation(table, _settings(), severity_scopes=_SEVERITY_SCOPES)
    assert selected == "candidate"
    assert float(scored[scored["model"] == "candidate"].iloc[0]["score"]) < 0.0


def test_selection_falls_back_to_the_baseline_when_nothing_qualifies() -> None:
    table = _metrics_table(
        _baseline_rows()
        + [
            ("candidate", "geometric_primary", 0.080),
            ("candidate", "clean_reference", 0.002),
            ("candidate", "original_blur_noise", 0.020),
            ("candidate", "geometric_primary_severity_2", 0.080),
            ("candidate", "geometric_primary_severity_3", 0.025),
            ("candidate", "geometric_primary_severity_4", 0.060),
        ]
    )
    selected, scored = select_ablation(table, _settings(), severity_scopes=_SEVERITY_SCOPES)
    assert selected == GLOBAL_BASELINE
    assert bool(scored[scored["model"] == GLOBAL_BASELINE].iloc[0]["selected"])


def test_selection_tie_breaking_is_deterministic() -> None:
    rows = _baseline_rows()
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
    first, _ = select_ablation(table, _settings(), severity_scopes=_SEVERITY_SCOPES)
    second, _ = select_ablation(
        table.sample(frac=1.0, random_state=7), _settings(), severity_scopes=_SEVERITY_SCOPES
    )
    assert first == second == "alpha"


# ---------------------------------------------------------------------------
# End-to-end run, persistence and replay
# ---------------------------------------------------------------------------


def test_end_to_end_run_persists_artifacts_and_replays(tmp_path: Path) -> None:
    """The full M6A flow must persist, reload and reproduce its own results."""
    from captcha_quality.spatial_study import (
        run_spatial_study,
        verify_spatial_study,
    )

    config = load_spatial_study_config(CONFIG_PATH)
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    result, paths = run_spatial_study(
        config, output_reports=reports, output_models=models, verify_images=False
    )
    assert result.ok
    assert result.selected_name in {ablation.name for ablation in result.ablations}

    for name in (
        "validation_slice_metrics.csv",
        "selection_table.csv",
        "test_slice_metrics.csv",
        "uncertainty.csv",
        "paired_uncertainty.csv",
        "test_predictions.csv",
        "isolation_checks.csv",
        "spatial_feature_manifest.csv",
        "run_metadata.json",
        "report.md",
    ):
        assert (reports / name).is_file(), name
    for ablation in config.ablation_names:
        if not config.ablation_refit[ablation]:
            continue
        assert (models / f"{ablation}.joblib").is_file()
        assert (models / f"{ablation}_card.json").is_file()
    assert paths["report"].is_file()

    metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selection"]["selected_model"] == result.selected_name
    assert metadata["frozen_assets"]["all_unchanged"] is True
    assert metadata["schema"]["global_feature_names"] == list(FEATURE_NAMES)
    assert metadata["schema"]["target"] == TARGET
    assert metadata["schema"]["spatial_feature_set_version"] == SPATIAL_FEATURE_SET_VERSION
    assert metadata["split"]["train_samples"] == 432
    assert metadata["split"]["validation_samples"] == 144
    assert metadata["split"]["primary_test_samples"] == 36
    assert metadata["split"]["diagnostic_test_samples"] == 12

    # Each refitted bundle must record the exact declared feature schema.
    for ablation in config.ablation_names:
        if not config.ablation_refit[ablation]:
            continue
        card = json.loads((models / f"{ablation}_card.json").read_text(encoding="utf-8"))
        expected = config.feature_names_for(config.ablation_feature_sets[ablation])
        assert card["feature_names"] == list(expected)
        assert card["target"] == TARGET
        assert card["artifact_sha256"] == sha256_file(models / f"{ablation}.joblib")
        assert card["selection"]["selected_on"] == "val"

    report = verify_spatial_study(config, output_reports=reports, output_models=models)
    assert report["passed"], [check for check in report["checks"] if not check.passed]


def test_report_generation_is_deterministic(tmp_path: Path) -> None:
    """Two runs of the same protocol must produce identical reports and tables."""
    from captcha_quality.spatial_study import run_spatial_study

    config = load_spatial_study_config(CONFIG_PATH)
    first_reports = tmp_path / "first"
    second_reports = tmp_path / "second"
    run_spatial_study(
        config,
        output_reports=first_reports,
        output_models=tmp_path / "m1",
        verify_images=False,
    )
    run_spatial_study(
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
        "spatial_feature_manifest.csv",
        "report.md",
    ):
        assert (first_reports / name).read_bytes() == (second_reports / name).read_bytes(), name


def test_verification_detects_a_tampered_artifact(tmp_path: Path) -> None:
    """A modified output must fail verification rather than pass silently."""
    from captcha_quality.spatial_study import (
        run_spatial_study,
        verify_spatial_study,
    )

    config = load_spatial_study_config(CONFIG_PATH)
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    run_spatial_study(config, output_reports=reports, output_models=models, verify_images=False)
    metrics_path = reports / "test_slice_metrics.csv"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    report = verify_spatial_study(config, output_reports=reports, output_models=models)
    assert not report["passed"]
    assert any(not check.passed for check in report["checks"])


def test_source_groups_never_cross_splits(tmp_path: Path) -> None:
    """Every variant of a source must stay in one partition."""
    from captcha_quality.spatial_study import load_study_frames

    config = load_spatial_study_config(CONFIG_PATH)
    frames = load_study_frames(config, verify_images=False)
    retention = frames.retention
    for frame in (
        retention.expanded_train,
        retention.expanded_val,
        retention.primary_test,
        retention.diagnostic_test,
        retention.m3_test,
    ):
        assert frame.groupby("source_id")["split"].nunique().max() == 1


def test_diagnostic_rung_is_excluded_from_the_primary_slices(tmp_path: Path) -> None:
    """The 0.5 px rung must not appear in the fit or primary scoring frames."""
    from captcha_quality.spatial_study import load_study_frames

    config = load_spatial_study_config(CONFIG_PATH)
    frames = load_study_frames(config, verify_images=False)
    retention = frames.retention
    for frame in (retention.expanded_train, retention.expanded_val, retention.primary_test):
        m5a = frame[frame["dataset"] == "m5a"]
        if m5a.empty:
            continue
        assert not (m5a["amplitude_px"] == 0.5).any()
    assert (retention.diagnostic_test["amplitude_px"] == 0.5).all()


def test_spatial_features_never_enter_the_global_ablation(tmp_path: Path) -> None:
    """The global baseline must consume exactly the eight frozen features."""
    config = load_spatial_study_config(CONFIG_PATH)
    global_names = config.feature_names_for(FEATURE_SET_GLOBAL)
    assert global_names == tuple(FEATURE_NAMES)
    for name in config.features.feature_names:
        assert name not in global_names
    spatial_names = config.feature_names_for(FEATURE_SET_SPATIAL)
    assert set(spatial_names).isdisjoint(set(global_names))
    combined = config.feature_names_for(FEATURE_SET_COMBINED)
    assert combined == tuple(FEATURE_NAMES) + tuple(config.features.feature_names)


def test_validation_selection_does_not_touch_test_metrics(tmp_path: Path) -> None:
    """The selection must be computable from validation metrics alone."""
    from captcha_quality import retention_aware as ra
    from captcha_quality.spatial_study import (
        _load_champion,
        _severity_scopes,
        _slice_metrics_table,
        build_ablations,
        load_study_frames,
    )

    config = load_spatial_study_config(CONFIG_PATH)
    frames = load_study_frames(config, verify_images=False)
    m5c_config = ra.load_retention_config(config.m5c.config)
    champion = _load_champion(config)
    ablations = build_ablations(frames, config, champion=champion, m5c_config=m5c_config)
    validation = frames.with_spatial(frames.retention.expanded_val)
    predictions = {ablation.name: ablation.predict(validation) for ablation in ablations}
    metrics = _slice_metrics_table(frames, ablations, predictions, split_label="val")
    assert set(metrics["split"]) == {"val"}
    selected, _table = select_ablation(
        metrics, config.selection, severity_scopes=_severity_scopes(config)
    )
    assert selected in {ablation.name for ablation in ablations}
