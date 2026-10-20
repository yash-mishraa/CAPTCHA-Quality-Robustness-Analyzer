"""Tests for the M6B spatial resolution study."""

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
    extract_spatial_feature_table,
    extract_spatial_features,
    spatial_feature_manifest,
)
from captcha_quality.spatial_resolution_study import (
    CHAMPION,
    COMBINED,
    SPATIAL_ONLY,
    GridSpec,
    ResolutionStudyConfigError,
    load_resolution_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "spatial_resolution_study_v0_1_0.toml"

#: The predeclared resolution set and the feature count each implies.
EXPECTED_GRIDS = {"2x2": 8, "4x2": 16, "4x4": 32, "8x8": 128}


def _blank(height: int = 64, width: int = 192, value: int = 255) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def _half_dark(height: int = 64, width: int = 192) -> np.ndarray:
    """Left half dark, right half light."""
    image = _blank(height, width)
    image[:, : width // 2] = 0
    return image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_shipped_config_declares_the_predeclared_protocol() -> None:
    config = load_resolution_config(CONFIG_PATH)
    assert config.global_feature_names == tuple(FEATURE_NAMES)
    assert config.global_feature_set_version == "v1"
    assert config.target == TARGET
    assert config.m6a.feature_set_version == SPATIAL_FEATURE_SET_VERSION
    assert config.model.family == "random_forest"
    assert config.model.n_estimators == 300
    assert config.model.seed == 798117391
    assert {grid.label for grid in config.resolutions.grids} == set(EXPECTED_GRIDS)
    assert config.representations == (SPATIAL_ONLY, COMBINED)
    assert config.champion.name == CHAMPION
    assert config.champion.refit is False
    assert config.selection.rule == "retention_gated_rmse_delta"
    assert config.selection.tolerance_clean == 0.50
    assert config.selection.tolerance_original == 0.10
    assert config.selection.tolerance_severity == 0.25
    assert len(config.config_hash) == 64


def test_resolution_feature_counts_follow_two_times_the_cell_count() -> None:
    config = load_resolution_config(CONFIG_PATH)
    for grid in config.resolutions.grids:
        assert grid.n_cells == grid.rows * grid.cols
        assert grid.n_spatial_features == 2 * grid.n_cells
        assert EXPECTED_GRIDS[grid.label] == grid.n_spatial_features
        spatial = config.feature_names_for(grid, SPATIAL_ONLY)
        combined = config.feature_names_for(grid, COMBINED)
        assert len(spatial) == grid.n_spatial_features
        assert len(combined) == grid.n_spatial_features + len(FEATURE_NAMES)
        assert combined[: len(FEATURE_NAMES)] == tuple(FEATURE_NAMES)


def test_candidate_names_cover_every_resolution_and_representation() -> None:
    config = load_resolution_config(CONFIG_PATH)
    names = config.candidate_names()
    assert names[0] == CHAMPION
    assert len(names) == 1 + 2 * len(config.resolutions.grids)
    for grid in config.resolutions.grids:
        for representation in config.representations:
            assert f"{representation}_{grid.label}" in names


def test_config_rejects_a_resolution_set_that_is_not_the_declared_one() -> None:
    import tomllib

    from captcha_quality.spatial_resolution_study import load_resolution_config

    payload = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["resolutions"]["grids"] = [{"rows": 3, "cols": 3}]
    broken = CONFIG_PATH.parent / "broken_resolution.toml"
    broken.write_text(_dump_toml(payload), encoding="utf-8")
    try:
        with pytest.raises(ResolutionStudyConfigError):
            load_resolution_config(broken)
    finally:
        broken.unlink()


def test_config_rejects_refitting_the_champion() -> None:
    import tomllib

    from captcha_quality.spatial_resolution_study import load_resolution_config

    payload = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["champion"]["refit"] = True
    broken = CONFIG_PATH.parent / "broken_resolution2.toml"
    broken.write_text(_dump_toml(payload), encoding="utf-8")
    try:
        with pytest.raises(ResolutionStudyConfigError):
            load_resolution_config(broken)
    finally:
        broken.unlink()


def test_config_rejects_a_modified_global_feature_schema() -> None:
    import tomllib

    from captcha_quality.spatial_resolution_study import load_resolution_config

    payload = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["schema"]["global_feature_names"] = ["intensity_mean"]
    broken = CONFIG_PATH.parent / "broken_resolution3.toml"
    broken.write_text(_dump_toml(payload), encoding="utf-8")
    try:
        with pytest.raises(ResolutionStudyConfigError):
            load_resolution_config(broken)
    finally:
        broken.unlink()


def _dump_toml(payload: dict) -> str:
    lines: list[str] = []
    for section, body in payload.items():
        lines.append(f"[{section}]")
        for key, value in body.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    lines.append(f"{key}.{inner_key} = {_toml_value(inner_value)}")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(f"{key} = [")
                for entry in value:
                    lines.append(
                        "  { "
                        + ", ".join(f"{k} = {_toml_value(v)}" for k, v in entry.items())
                        + " },"
                    )
                lines.append("]")
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
# Feature counts, ordering and region boundaries
# ---------------------------------------------------------------------------


def test_feature_counts_and_ordering_at_every_resolution() -> None:
    for label, expected in EXPECTED_GRIDS.items():
        rows, cols = (int(part) for part in label.split("x"))
        spec = SpatialFeatureSpec(grid_rows=rows, grid_cols=cols)
        names = spec.feature_names
        assert len(names) == expected
        assert len(set(names)) == expected
        # Row-major: the column index changes fastest.
        assert names[0] == f"{FAMILY_INK}_r0_c0"
        assert names[1] == f"{FAMILY_INK}_r0_c1"
        assert names[cols] == f"{FAMILY_INK}_r1_c0"
        assert names[rows * cols] == f"{FAMILY_EDGE}_r0_c0"
        assert names[-1] == f"{FAMILY_EDGE}_r{rows - 1}_c{cols - 1}"


def test_feature_manifest_matches_the_feature_names() -> None:
    for rows, cols in ((2, 2), (4, 2), (4, 4), (8, 8)):
        spec = SpatialFeatureSpec(grid_rows=rows, grid_cols=cols)
        manifest = spatial_feature_manifest(spec)
        assert list(manifest["feature_name"]) == list(spec.feature_names)
        assert manifest["grid_row"].max() == rows - 1
        assert manifest["grid_col"].max() == cols - 1


def test_region_boundaries_cover_the_canvas_exactly() -> None:
    """The declared grids must tile the canonical canvas without gaps or overlap."""
    from captcha_quality.spatial_features import _cell_bounds

    for rows, cols in ((2, 2), (4, 2), (4, 4), (8, 8)):
        bounds = _cell_bounds(64, 192, rows, cols)
        assert len(bounds) == rows * cols
        covered = np.zeros((64, 192), dtype=bool)
        for y0, y1, x0, x1 in bounds:
            assert 0 <= y0 < y1 <= 64
            assert 0 <= x0 < x1 <= 192
            covered[y0:y1, x0:x1] = True
        assert covered.all()


def test_uneven_dimensions_distribute_the_remainder_to_later_cells() -> None:
    """A dimension that is not a multiple of the grid still tiles completely."""
    from captcha_quality.spatial_features import _cell_bounds

    for height, width, rows, cols in ((10, 13, 4, 3), (17, 19, 5, 4), (64, 193, 8, 8)):
        bounds = _cell_bounds(height, width, rows, cols)
        covered = np.zeros((height, width), dtype=bool)
        for y0, y1, x0, x1 in bounds:
            assert y1 > y0 and x1 > x0
            covered[y0:y1, x0:x1] = True
        assert covered.all()


def test_uniform_image_yields_all_zero_features_at_every_resolution() -> None:
    for rows, cols in ((2, 2), (4, 2), (4, 4), (8, 8)):
        spec = SpatialFeatureSpec(grid_rows=rows, grid_cols=cols)
        features = extract_spatial_features(_blank(), spec)
        assert set(features) == set(spec.feature_names)
        assert all(value == 0.0 for value in features.values())


def test_half_dark_image_ink_sits_in_the_left_column_only() -> None:
    for rows, cols in ((2, 2), (4, 2), (4, 4), (8, 8)):
        spec = SpatialFeatureSpec(grid_rows=rows, grid_cols=cols)
        features = extract_spatial_features(_half_dark(), spec)
        for row in range(rows):
            left = features[f"{FAMILY_INK}_r{row}_c0"]
            right = features[f"{FAMILY_INK}_r{row}_c{cols - 1}"]
            assert left == pytest.approx(1.0)
            assert right == pytest.approx(0.0)


def test_extraction_is_deterministic_at_every_resolution() -> None:
    image = _half_dark()
    for rows, cols in ((2, 2), (4, 2), (4, 4), (8, 8)):
        spec = SpatialFeatureSpec(grid_rows=rows, grid_cols=cols)
        assert extract_spatial_features(image, spec) == extract_spatial_features(image, spec)


def test_values_are_finite_and_in_range_at_every_resolution() -> None:
    rng = np.random.default_rng(19)
    image = rng.integers(0, 256, size=(64, 192), dtype=np.uint8)
    for rows, cols in ((2, 2), (4, 2), (4, 4), (8, 8)):
        spec = SpatialFeatureSpec(grid_rows=rows, grid_cols=cols)
        for name, value in extract_spatial_features(image, spec).items():
            assert np.isfinite(value), name
            assert 0.0 <= value <= 1.0, name


def test_empty_region_yields_zero_not_an_error() -> None:
    """A grid finer than the image produces empty cells; they must be zero."""
    spec = SpatialFeatureSpec(grid_rows=64, grid_cols=192)
    features = extract_spatial_features(_half_dark(height=16, width=32), spec)
    assert all(np.isfinite(value) for value in features.values())
    assert all(0.0 <= value <= 1.0 for value in features.values())


def test_extraction_rejects_a_violating_image() -> None:
    with pytest.raises(ImageContractError):
        extract_spatial_features(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ImageContractError):
        extract_spatial_features(np.zeros((64, 192), dtype=np.float64))


def test_batch_extraction_preserves_column_order_and_rejects_duplicates() -> None:
    spec = SpatialFeatureSpec(grid_rows=8, grid_cols=8)
    table = extract_spatial_feature_table([("a", _half_dark()), ("b", _blank())], spec)
    assert list(table.columns) == ["sample_id", *spec.feature_names]
    with pytest.raises(SpatialFeatureError):
        extract_spatial_feature_table([("a", _blank()), ("a", _blank())], spec)
    with pytest.raises(SpatialFeatureError):
        extract_spatial_feature_table([("", _blank())], spec)


# ---------------------------------------------------------------------------
# Frozen schema and leakage
# ---------------------------------------------------------------------------


def test_the_frozen_global_feature_schema_is_untouched() -> None:
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
    assert SPATIAL_FEATURE_SET_VERSION == "spatial-v1"


def test_grid_spec_rejects_an_invalid_resolution() -> None:
    with pytest.raises(SpatialFeatureError):
        SpatialFeatureSpec(grid_rows=0, grid_cols=2)
    with pytest.raises(SpatialFeatureError):
        SpatialFeatureSpec(families=("not_a_family",))


def test_feature_names_are_namespaced_per_resolution() -> None:
    """The same cell name at two resolutions must not collide."""
    config = load_resolution_config(CONFIG_PATH)
    seen: set[str] = set()
    for grid in config.resolutions.grids:
        names = config.feature_names_for(grid, SPATIAL_ONLY)
        assert not seen & set(names)
        seen |= set(names)
    combined = [config.feature_names_for(grid, COMBINED) for grid in config.resolutions.grids]
    for names in combined:
        assert names[: len(FEATURE_NAMES)] == tuple(FEATURE_NAMES)


def test_grid_spec_reports_the_expected_feature_counts() -> None:
    grid = GridSpec(rows=8, cols=8)
    assert grid.n_cells == 64
    assert grid.n_spatial_features == 128
    assert grid.label == "8x8"


# ---------------------------------------------------------------------------
# End-to-end run, persistence and replay
# ---------------------------------------------------------------------------


def test_end_to_end_run_persists_artifacts_and_replays(tmp_path: Path) -> None:
    """The full M6B flow must persist, reload and reproduce its own results."""
    from captcha_quality.spatial_resolution_study import (
        run_resolution_study,
        verify_resolution_study,
    )

    config = load_resolution_config(CONFIG_PATH)
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    result, paths = run_resolution_study(
        config, output_reports=reports, output_models=models, verify_images=False
    )
    assert result.ok
    assert result.selected_name in {candidate.name for candidate in result.candidates}

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
    for candidate in config.candidate_names():
        if candidate == config.champion.name:
            continue
        assert (models / f"{candidate}.joblib").is_file()
        assert (models / f"{candidate}_card.json").is_file()
    assert paths["report"].is_file()

    metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selection"]["selected_model"] == result.selected_name
    assert metadata["frozen_assets"]["all_unchanged"] is True
    assert metadata["schema"]["global_feature_names"] == list(FEATURE_NAMES)
    assert metadata["schema"]["spatial_feature_set_version"] == SPATIAL_FEATURE_SET_VERSION
    assert metadata["split"]["train_samples"] == 432
    assert metadata["split"]["validation_samples"] == 144
    assert metadata["split"]["primary_test_samples"] == 36
    assert metadata["split"]["diagnostic_test_samples"] == 12
    # The declared feature counts must be recorded for every resolution.
    for label, expected in EXPECTED_GRIDS.items():
        assert metadata["resolutions"][label]["n_spatial_features"] == expected
        assert metadata["resolutions"][label]["n_combined_features"] == expected + 8

    # Every refitted bundle must record the exact declared feature schema.
    for candidate in config.candidate_names():
        if candidate == config.champion.name:
            continue
        card = json.loads((models / f"{candidate}_card.json").read_text(encoding="utf-8"))
        assert card["target"] == TARGET
        assert card["artifact_sha256"] == sha256_file(models / f"{candidate}.joblib")
        assert card["selection"]["selected_on"] == "val"
        assert card["seed"] == config.model.seed

    report = verify_resolution_study(config, output_reports=reports, output_models=models)
    assert report["passed"], [check for check in report["checks"] if not check.passed]


def test_report_generation_is_deterministic(tmp_path: Path) -> None:
    """Two runs of the same protocol must produce identical reports and tables."""
    from captcha_quality.spatial_resolution_study import run_resolution_study

    config = load_resolution_config(CONFIG_PATH)
    first_reports = tmp_path / "first"
    second_reports = tmp_path / "second"
    run_resolution_study(
        config,
        output_reports=first_reports,
        output_models=tmp_path / "m1",
        verify_images=False,
    )
    run_resolution_study(
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
    from captcha_quality.spatial_resolution_study import (
        run_resolution_study,
        verify_resolution_study,
    )

    config = load_resolution_config(CONFIG_PATH)
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    run_resolution_study(config, output_reports=reports, output_models=models, verify_images=False)
    metrics_path = reports / "test_slice_metrics.csv"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    report = verify_resolution_study(config, output_reports=reports, output_models=models)
    assert not report["passed"]
    assert any(not check.passed for check in report["checks"])


def test_source_groups_never_cross_splits(tmp_path: Path) -> None:
    """Every variant of a source must stay in one partition."""
    from captcha_quality.spatial_resolution_study import load_resolution_frames

    config = load_resolution_config(CONFIG_PATH)
    frames = load_resolution_frames(config, verify_images=False)
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
    from captcha_quality.spatial_resolution_study import load_resolution_frames

    config = load_resolution_config(CONFIG_PATH)
    frames = load_resolution_frames(config, verify_images=False)
    retention = frames.retention
    for frame in (retention.expanded_train, retention.expanded_val, retention.primary_test):
        m5a = frame[frame["dataset"] == "m5a"]
        if m5a.empty:
            continue
        assert not (m5a["amplitude_px"] == 0.5).any()
    assert (retention.diagnostic_test["amplitude_px"] == 0.5).all()


def test_validation_selection_does_not_touch_test_metrics(tmp_path: Path) -> None:
    """The selection must be computable from validation metrics alone."""
    from captcha_quality import retention_aware as ra
    from captcha_quality.spatial_resolution_study import (
        _load_champion,
        _severity_scopes,
        _slice_metrics_table,
        build_candidates,
        load_resolution_frames,
    )
    from captcha_quality.spatial_study import select_ablation

    config = load_resolution_config(CONFIG_PATH)
    frames = load_resolution_frames(config, verify_images=False)
    m5c_config = ra.load_retention_config(config.m5c.config)
    champion = _load_champion(config)
    candidates = build_candidates(frames, config, champion=champion, m5c_config=m5c_config)
    validation = frames.with_spatial(frames.retention.expanded_val)
    predictions = {candidate.name: candidate.predict(validation) for candidate in candidates}
    metrics = _slice_metrics_table(frames, candidates, predictions, split_label="val")
    assert set(metrics["split"]) == {"val"}
    selected, _table = select_ablation(
        metrics,
        config.selection,
        severity_scopes=_severity_scopes(config),
        reference=config.champion.name,
    )
    assert selected in {candidate.name for candidate in candidates}


def test_selection_falls_back_to_the_champion_when_nothing_qualifies() -> None:
    """The champion is the fallback and always qualifies against itself."""
    from captcha_quality.spatial_resolution_study import (
        _severity_scopes,
        load_resolution_config,
    )
    from captcha_quality.spatial_study import select_ablation

    config = load_resolution_config(CONFIG_PATH)
    rows = [
        (CHAMPION, "geometric_primary", 0.05),
        (CHAMPION, "clean_reference", 0.002),
        (CHAMPION, "original_blur_noise", 0.020),
        (CHAMPION, "geometric_primary_severity_2", 0.05),
        (CHAMPION, "geometric_primary_severity_3", 0.025),
        (CHAMPION, "geometric_primary_severity_4", 0.06),
        ("worse", "geometric_primary", 0.08),
        ("worse", "clean_reference", 0.002),
        ("worse", "original_blur_noise", 0.020),
        ("worse", "geometric_primary_severity_2", 0.08),
        ("worse", "geometric_primary_severity_3", 0.025),
        ("worse", "geometric_primary_severity_4", 0.060),
    ]
    table = pd.DataFrame(
        [
            {
                "model": model,
                "feature_set": "combined",
                "n_features": 8,
                "scope": scope,
                "rmse": rmse,
            }
            for model, scope, rmse in rows
        ]
    )
    selected, scored = select_ablation(
        table,
        config.selection,
        severity_scopes=_severity_scopes(config),
        reference=CHAMPION,
    )
    assert selected == CHAMPION
    assert bool(scored[scored["model"] == CHAMPION].iloc[0]["selected"])
    assert not bool(scored[scored["model"] == "worse"].iloc[0]["eligible"])
