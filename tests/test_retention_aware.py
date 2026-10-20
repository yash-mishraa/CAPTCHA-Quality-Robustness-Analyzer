"""Tests for the M5C retention-aware model selection and calibration milestone."""

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
from captcha_quality.geometric_benchmark import (
    DIAGNOSTIC_AMPLITUDES_PX,
    PRIMARY_AMPLITUDES_PX,
    CheckResult,
)
from captcha_quality.modelling import (
    TARGET,
    ModelError,
    model_inputs,
    regression_metrics,
    validate_feature_columns,
)
from captcha_quality.provenance import sha256_file
from captcha_quality.retention_aware import (
    ALTERNATIVE_MODEL_STEM,
    CALIBRATION_AFFINE,
    CALIBRATION_NONE,
    CLEAN_LABEL,
    EXPANDED_ALTERNATIVE,
    FROZEN_M3,
    GEO_LABEL,
    M5B_EXPANDED_HGB,
    ORIGINAL_LABEL,
    SELECTED_POLICY_STEM,
    AffineCalibration,
    RetentionConfig,
    RetentionFrames,
    RetentionSelectionConfigError,
    RetentionSelectionError,
    SelectionSettings,
    _check_alternative_justification,
    _check_calibration_isolation,
    _check_low_amplitude_excluded,
    _check_test_immutability,
    _cross_fit_calibration,
    _fit_affine,
    _retention_score,
    build_candidate_set,
    build_test_slices,
    load_retention_config,
    select_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "retention_aware_selection_v0_1_0.toml"


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
                    "reference_sha256": f"{source.source_id}-clean",
                    **_feature_values(source_index, target),
                    TARGET: target,
                }
            )
    return pd.DataFrame(rows)


def _m5a_rows(sources: pd.DataFrame) -> pd.DataFrame:
    """M5A-shaped rows: clean, three controls and four elastic severities per source."""
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
                    "reference_sha256": f"{source.source_id}-clean",
                    **_feature_values(source_index, target),
                    TARGET: target,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_frames() -> RetentionFrames:
    """Build a small, self-consistent M5C frame bundle for unit tests."""
    sources = _source_frame({"train": 12, "val": 6, "test": 6})
    m3 = _m3_rows(sources)
    m5a = _m5a_rows(sources)
    m3_train = m3[m3["split"] == "train"]
    m3_val = m3[m3["split"] == "val"]
    m3_test = m3[m3["split"] == "test"]
    m5a_primary_train = m5a[(m5a["split"] == "train") & (m5a["protocol_scope"] == "primary")]
    m5a_primary_val = m5a[(m5a["split"] == "val") & (m5a["protocol_scope"] == "primary")]
    primary_test = m5a[(m5a["split"] == "test") & (m5a["protocol_scope"] == "primary")]
    diagnostic_test = m5a[(m5a["split"] == "test") & (m5a["protocol_scope"] == "diagnostic")]
    expanded_train = pd.concat([m3_train, m5a_primary_train], ignore_index=True)
    expanded_val = pd.concat([m3_val, m5a_primary_val], ignore_index=True)
    checks = (CheckResult("synthetic", True, "synthetic frames"),)
    return RetentionFrames(
        frames=None,  # type: ignore[arg-type]
        expanded_train=expanded_train,
        expanded_val=expanded_val,
        primary_test=primary_test,
        diagnostic_test=diagnostic_test,
        m3_test=m3_test,
        m3_frame=m3,
        m5a_frame=m5a,
        checks=checks,
        alternative_spec="random_forest_depth_unlimited_leaf_1",
        alternative_validation={
            "m5b_grid_candidates": 12,
            "m5b_selected_on_validation": "hist_gradient_boosting_lr_0.1_leaves_31",
            "m5b_selected_val_rmse": 0.02915142119876432,
            "alternative": "random_forest_depth_unlimited_leaf_1",
            "alternative_val_rmse": 0.029584799760657622,
            "alternative_val_mae": 0.02118611133481319,
            "alternative_val_r2": 0.9770077763606754,
            "alternative_val_gap_fraction": 0.014866464277620573,
        },
    )


def _test_config(tmp_path: Path) -> RetentionConfig:
    """Build an in-memory M5C configuration for unit tests."""
    config = load_retention_config(CONFIG_PATH)
    return RetentionConfig(
        run=config.run,
        schema=config.schema,
        m5b=config.m5b,
        protocol=config.protocol,
        candidates=config.candidates,
        calibration=config.calibration,
        selection=config.selection,
        config_path=tmp_path / "retention.toml",
        config_sha256="0" * 64,
    )


def _dummy_estimator(scale: float = 1.0):
    """Return a deterministic linear estimator over the eight features."""

    class _Linear:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return scale * np.asarray(values, dtype=np.float64)[:, 0]

    return _Linear()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_shipped_config_declares_the_predeclared_protocol() -> None:
    config = load_retention_config(CONFIG_PATH)
    assert config.schema.feature_names == tuple(FEATURE_NAMES)
    assert config.schema.target == TARGET
    assert config.schema.feature_set_version == FEATURE_SET_VERSION
    assert config.protocol.primary_amplitudes_px == PRIMARY_AMPLITUDES_PX
    assert config.protocol.diagnostic_amplitudes_px == DIAGNOSTIC_AMPLITUDES_PX
    assert set(config.candidates.base_models) == {FROZEN_M3, M5B_EXPANDED_HGB, EXPANDED_ALTERNATIVE}
    assert set(config.candidates.calibration_forms) == {CALIBRATION_NONE, CALIBRATION_AFFINE}
    assert config.calibration.fit_scope == "expanded_validation"
    assert config.calibration.clip_output is False
    assert config.selection.rule == "retention_penalised_rmse_delta"
    assert config.selection.tolerance == 0.0
    assert len(config.config_hash) == 64


def test_config_rejects_a_diagnostic_amplitude_in_the_primary_ladder() -> None:
    from captcha_quality.retention_aware import _parse_protocol

    with pytest.raises(RetentionSelectionConfigError):
        _parse_protocol(
            {
                "primary_amplitudes_px": [0.5, 1.0, 2.0, 3.0],
                "diagnostic_amplitudes_px": [0.5],
                "primary_severities": [1, 2, 3, 4],
                "diagnostic_severities": [1],
            }
        )


def test_config_rejects_an_unknown_candidate_base_model() -> None:
    from captcha_quality.retention_aware import _parse_candidates

    with pytest.raises(RetentionSelectionConfigError):
        _parse_candidates(
            {
                "base_models": [FROZEN_M3, M5B_EXPANDED_HGB],
                "calibration_forms": [CALIBRATION_NONE, CALIBRATION_AFFINE],
                "alternative": {"spec": "x", "justification": "y"},
            }
        )


def test_config_rejects_a_non_validation_calibration_scope() -> None:
    from captcha_quality.retention_aware import _parse_calibration

    with pytest.raises(RetentionSelectionConfigError):
        _parse_calibration(
            {
                "form": "affine",
                "fit_scope": "test",
                "cross_fit_folds": 4,
                "clip_output": False,
            }
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_affine_calibration_recovers_a_known_linear_map() -> None:
    predicted = np.array([0.1, 0.4, 0.6, 0.9])
    target = 0.05 + 1.25 * predicted
    calibration = _fit_affine(predicted, target)
    assert calibration.slope == pytest.approx(1.25)
    assert calibration.intercept == pytest.approx(0.05)
    assert np.allclose(calibration.apply(predicted), target)


def test_affine_calibration_rejects_constant_predictions() -> None:
    with pytest.raises(RetentionSelectionError):
        _fit_affine(np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.2, 0.3]))


def test_affine_calibration_does_not_clip_when_clipping_is_disabled() -> None:
    calibration = AffineCalibration(intercept=0.0, slope=1.0, clip_output=False)
    assert calibration.apply(np.array([1.2]))[0] == pytest.approx(1.2)
    clipped = AffineCalibration(intercept=0.0, slope=1.0, clip_output=True)
    assert clipped.apply(np.array([1.2]))[0] == pytest.approx(1.0)


def test_cross_fit_calibration_is_grouped_and_out_of_sample() -> None:
    frames = _synthetic_frames()
    val = frames.expanded_val
    # A non-linear, source-dependent relationship so that each fold's
    # calibration genuinely differs from the full-sample fit.
    rng = np.random.default_rng(3)
    unique_sources = sorted(val["source_id"].astype(str).unique())
    offsets = rng.normal(0.0, 0.02, size=len(unique_sources))
    by_source = dict(zip(unique_sources, offsets, strict=True))
    predicted = (
        val[TARGET].to_numpy(dtype=np.float64) * 0.9
        + 0.05
        + val["source_id"].astype(str).map(by_source).to_numpy(dtype=np.float64)
    )
    calibration, cross_fitted = _cross_fit_calibration(
        val, predicted, n_folds=3, seed=7, clip_output=False
    )
    assert cross_fitted.shape == (len(val),)
    assert np.isfinite(cross_fitted).all()
    # The cross-fitted predictions must differ from a plain in-sample fit,
    # otherwise the candidate would be scored on rows that contributed to its
    # own calibration.
    in_sample = calibration.apply(predicted)
    assert not np.allclose(cross_fitted, in_sample)
    # Calibration improves on the raw predictions.
    raw_rmse = regression_metrics(val[TARGET].to_numpy(dtype=np.float64), predicted)["rmse"]
    fitted_rmse = regression_metrics(val[TARGET].to_numpy(dtype=np.float64), cross_fitted)["rmse"]
    assert fitted_rmse < raw_rmse


def test_cross_fit_calibration_keeps_every_source_in_one_fold() -> None:
    frames = _synthetic_frames()
    val = frames.expanded_val
    predicted = val[TARGET].to_numpy(dtype=np.float64) * 0.8 + 0.1
    _, cross_fitted = _cross_fit_calibration(val, predicted, n_folds=3, seed=5)
    # Every source contributes rows to a single fold, so no source can be
    # split between the calibration fit and the scored rows.
    assert len(cross_fitted) == len(val)
    assert np.isfinite(cross_fitted).all()


def test_cross_fit_calibration_requires_enough_source_groups() -> None:
    frames = _synthetic_frames()
    val = frames.expanded_val.head(2)
    with pytest.raises(RetentionSelectionError):
        _cross_fit_calibration(val, val[TARGET].to_numpy(dtype=np.float64), n_folds=4, seed=1)


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def _metric_table(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "base_model": model.split("__")[0],
                "calibration": "none",
                "scope": scope,
                "rmse": rmse,
            }
            for model, scope, rmse in rows
        ]
    )


def _slice_rmse(**values: float) -> dict[str, dict[str, float]]:
    """Build the nested slice mapping the retention rule consumes."""
    return {scope: {"rmse": value} for scope, value in values.items()}


def test_retention_score_is_zero_for_the_frozen_reference() -> None:
    settings = SelectionSettings(
        rule="retention_penalised_rmse_delta",
        w_geometric=1.0,
        w_clean=1.0,
        w_original=1.0,
        penalty_regression=1.0,
        tolerance=0.0,
    )
    reference = _slice_rmse(**{GEO_LABEL: 0.08, CLEAN_LABEL: 0.01, ORIGINAL_LABEL: 0.02})
    scored = _retention_score(reference, reference, settings)
    assert scored["score"] == pytest.approx(0.0)


def test_retention_score_charges_a_retention_regression_twice() -> None:
    settings = SelectionSettings(
        rule="retention_penalised_rmse_delta",
        w_geometric=1.0,
        w_clean=1.0,
        w_original=1.0,
        penalty_regression=1.0,
        tolerance=0.0,
    )
    reference = _slice_rmse(**{GEO_LABEL: 0.08, CLEAN_LABEL: 0.01, ORIGINAL_LABEL: 0.02})
    candidate = _slice_rmse(**{GEO_LABEL: 0.08, CLEAN_LABEL: 0.02, ORIGINAL_LABEL: 0.02})
    scored = _retention_score(candidate, reference, settings)
    # A clean regression of +0.01 is charged once as the weighted term and
    # once as the explicit penalty.
    assert scored["score"] == pytest.approx(0.02)
    assert scored["excess_clean_rmse"] == pytest.approx(0.01)


def test_retention_score_rewards_a_geometric_gain() -> None:
    settings = SelectionSettings(
        rule="retention_penalised_rmse_delta",
        w_geometric=1.0,
        w_clean=1.0,
        w_original=1.0,
        penalty_regression=1.0,
        tolerance=0.0,
    )
    reference = _slice_rmse(**{GEO_LABEL: 0.08, CLEAN_LABEL: 0.01, ORIGINAL_LABEL: 0.02})
    candidate = _slice_rmse(**{GEO_LABEL: 0.06, CLEAN_LABEL: 0.01, ORIGINAL_LABEL: 0.02})
    scored = _retention_score(candidate, reference, settings)
    assert scored["score"] == pytest.approx(-0.02)
    assert scored["delta_geometric_rmse"] == pytest.approx(-0.02)


def test_selection_prefers_the_lowest_retention_aware_score() -> None:
    settings = SelectionSettings(
        rule="retention_penalised_rmse_delta",
        w_geometric=1.0,
        w_clean=1.0,
        w_original=1.0,
        penalty_regression=1.0,
        tolerance=0.0,
    )
    table = _metric_table(
        [
            (FROZEN_M3, GEO_LABEL, 0.08),
            (FROZEN_M3, CLEAN_LABEL, 0.01),
            (FROZEN_M3, ORIGINAL_LABEL, 0.02),
            ("candidate", GEO_LABEL, 0.05),
            ("candidate", CLEAN_LABEL, 0.01),
            ("candidate", ORIGINAL_LABEL, 0.02),
            ("regressive", GEO_LABEL, 0.04),
            ("regressive", CLEAN_LABEL, 0.05),
            ("regressive", ORIGINAL_LABEL, 0.02),
        ]
    )
    selected, scored = select_policy(table, settings)
    assert selected == "candidate"
    assert bool(scored[scored["model"] == "candidate"]["selected"].iloc[0]) is True
    assert bool(scored[scored["model"] == "regressive"]["selected"].iloc[0]) is False


def test_selection_requires_every_candidate_to_cover_the_same_slices() -> None:
    settings = SelectionSettings(
        rule="retention_penalised_rmse_delta",
        w_geometric=1.0,
        w_clean=1.0,
        w_original=1.0,
        penalty_regression=1.0,
        tolerance=0.0,
    )
    table = _metric_table(
        [
            (FROZEN_M3, GEO_LABEL, 0.08),
            (FROZEN_M3, CLEAN_LABEL, 0.01),
            (FROZEN_M3, ORIGINAL_LABEL, 0.02),
            ("partial", GEO_LABEL, 0.05),
        ]
    )
    with pytest.raises(RetentionSelectionError):
        select_policy(table, settings)


# ---------------------------------------------------------------------------
# Frame integrity and leakage
# ---------------------------------------------------------------------------


def test_calibration_fit_frame_is_validation_only_and_disjoint_from_train() -> None:
    frames = _synthetic_frames()
    check = _check_calibration_isolation(frames.expanded_val, frames.expanded_train)
    assert check.passed
    train_sources = set(frames.expanded_train["source_id"])
    val_sources = set(frames.expanded_val["source_id"])
    assert not train_sources & val_sources
    assert set(frames.expanded_val["split"]) == {"val"}


def test_calibration_isolation_fails_when_a_source_crosses_splits() -> None:
    frames = _synthetic_frames()
    contaminated = frames.expanded_val.copy()
    source = contaminated.loc[contaminated["split"] == "val", "source_id"].iloc[0]
    contaminated.loc[contaminated["source_id"] == source, "split"] = "train"
    check = _check_calibration_isolation(contaminated, frames.expanded_train)
    assert not check.passed


def test_test_frames_are_immutable_and_disjoint_from_fit_frames() -> None:
    frames = _synthetic_frames()
    check = _check_test_immutability(frames.primary_test, frames.diagnostic_test, frames.m3_test)
    assert check.passed
    test_ids = set(frames.primary_test["sample_id"]) | set(frames.diagnostic_test["sample_id"])
    fit_ids = set(frames.expanded_train["sample_id"]) | set(frames.expanded_val["sample_id"])
    assert not test_ids & fit_ids


def test_test_immutability_fails_when_a_test_row_is_reused() -> None:
    frames = _synthetic_frames()
    primary = pd.concat([frames.primary_test, frames.diagnostic_test.head(1)], ignore_index=True)
    check = _check_test_immutability(primary, frames.diagnostic_test, frames.m3_test)
    assert not check.passed


def test_low_amplitude_rung_is_excluded_from_fit_and_primary_frames() -> None:
    frames = _synthetic_frames()
    check = _check_low_amplitude_excluded(
        frames.expanded_val, frames.primary_test, frames.diagnostic_test
    )
    assert check.passed
    assert not (frames.expanded_val["amplitude_px"] == 0.5).any()
    assert not (frames.primary_test["amplitude_px"] == 0.5).any()
    assert (frames.diagnostic_test["amplitude_px"] == 0.5).all()


def test_low_amplitude_exclusion_fails_when_the_rung_enters_the_fit_frame() -> None:
    frames = _synthetic_frames()
    contaminated = pd.concat(
        [frames.expanded_val, frames.diagnostic_test.head(1)], ignore_index=True
    )
    check = _check_low_amplitude_excluded(contaminated, frames.primary_test, frames.diagnostic_test)
    assert not check.passed


def test_feature_schema_is_preserved_and_metadata_is_excluded() -> None:
    frames = _synthetic_frames()
    inputs = model_inputs(frames.expanded_train)
    assert inputs.shape[1] == len(FEATURE_NAMES)
    with pytest.raises(ModelError):
        validate_feature_columns([*FEATURE_NAMES, "severity_level"])


def test_all_variants_of_a_source_stay_in_one_split() -> None:
    frames = _synthetic_frames()
    for frame in (frames.expanded_train, frames.expanded_val, frames.primary_test):
        assert frame.groupby("source_id")["split"].nunique().max() == 1


# ---------------------------------------------------------------------------
# Candidate construction and evaluation
# ---------------------------------------------------------------------------


def test_candidate_set_covers_every_predeclared_base_and_calibration() -> None:
    frames = _synthetic_frames()
    config = _test_config(Path("/tmp"))
    candidates = build_candidate_set(
        frames,
        config,
        frozen_m3=_dummy_estimator(1.0),
        m5b_expanded=_dummy_estimator(0.9),
        alternative=_dummy_estimator(0.8),
    )
    assert set(candidates.names) == {
        FROZEN_M3,
        "frozen_m3__affine",
        M5B_EXPANDED_HGB,
        "m5b_expanded_hgb__affine",
        EXPANDED_ALTERNATIVE,
        "expanded_random_forest__affine",
    }
    assert len(candidates.policies) == 6


def test_calibrated_candidates_are_scored_with_cross_fitted_predictions() -> None:
    frames = _synthetic_frames()
    config = _test_config(Path("/tmp"))
    candidates = build_candidate_set(
        frames,
        config,
        frozen_m3=_dummy_estimator(1.0),
        m5b_expanded=_dummy_estimator(0.9),
        alternative=_dummy_estimator(0.8),
    )
    diagnostics = candidates.calibration_diagnostics["frozen_m3__affine"]
    assert diagnostics["calibration"] == CALIBRATION_AFFINE
    assert diagnostics["fit_rows"] == len(frames.expanded_val)
    assert diagnostics["scored_with"] == "cross-fitted validation predictions"
    # The cross-fitted predictions must not be identical to the raw ones.
    raw = candidates.validation_predictions[FROZEN_M3]
    assert not np.allclose(raw, candidates.validation_predictions["frozen_m3__affine"])


def test_test_slices_are_scored_on_their_own_frames() -> None:
    frames = _synthetic_frames()
    config = _test_config(Path("/tmp"))
    candidates = build_candidate_set(
        frames,
        config,
        frozen_m3=_dummy_estimator(1.0),
        m5b_expanded=_dummy_estimator(0.9),
        alternative=_dummy_estimator(0.8),
    )
    slices = build_test_slices(frames, candidates)
    scopes = {item.scope for item in slices}
    assert GEO_LABEL in scopes
    assert CLEAN_LABEL in scopes
    assert ORIGINAL_LABEL in scopes
    assert "diagnostic_0_5" in scopes
    assert "geometric_primary_severity_2" in scopes
    for item in slices:
        for name, predicted in item.predictions.items():
            assert len(predicted) == len(item.frame), name


def test_calibration_never_sees_a_test_target() -> None:
    frames = _synthetic_frames()
    config = _test_config(Path("/tmp"))
    candidates = build_candidate_set(
        frames,
        config,
        frozen_m3=_dummy_estimator(1.0),
        m5b_expanded=_dummy_estimator(0.9),
        alternative=_dummy_estimator(0.8),
    )
    for entry in candidates.calibration_diagnostics.values():
        if entry["calibration"] == CALIBRATION_NONE:
            continue
        assert entry["fit_rows"] == len(frames.expanded_val)
        assert entry["fit_sources"] == frames.expanded_val["source_id"].nunique()


def test_alternative_justification_uses_the_m5b_validation_table() -> None:
    """The alternative must be justified by validation, never by a test result."""
    from captcha_quality import geometric_benchmark as gb

    config = load_retention_config(CONFIG_PATH)
    m5b_config = gb.load_geometric_benchmark_config(config.m5b.config)
    if not m5b_config.model.model_dir.is_dir():  # pragma: no cover - depends on disk state
        pytest.skip("the frozen M3 model bundle is not available")
    details = _check_alternative_justification(m5b_config, config.candidates.alternative_spec)
    assert details["alternative"] == config.candidates.alternative_spec
    assert details["m5b_selected_on_validation"] != config.candidates.alternative_spec
    assert details["alternative_val_rmse"] > details["m5b_selected_val_rmse"]
    assert details["m5b_grid_candidates"] == 12


# ---------------------------------------------------------------------------
# End-to-end run, persistence and replay
# ---------------------------------------------------------------------------


def test_end_to_end_run_persists_artifacts_and_replays(tmp_path: Path) -> None:
    """The full M5C flow must persist, reload and reproduce its own results."""
    from captcha_quality.retention_aware import (
        run_retention_aware,
        verify_retention_aware,
    )

    config = load_retention_config(CONFIG_PATH)
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    result, paths = run_retention_aware(
        config, output_reports=reports, output_models=models, verify_images=False
    )
    assert result.ok
    assert result.selected_name in result.candidates.names

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
    assert (models / f"{ALTERNATIVE_MODEL_STEM}.joblib").is_file()
    assert (models / f"{ALTERNATIVE_MODEL_STEM}_card.json").is_file()
    assert (models / f"{SELECTED_POLICY_STEM}.joblib").is_file()
    assert (models / f"{SELECTED_POLICY_STEM}_card.json").is_file()
    assert paths["report"].is_file()

    metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selection"]["selected_model"] == result.selected_name
    assert metadata["frozen_assets"]["all_unchanged"] is True
    assert metadata["schema"]["feature_names"] == list(FEATURE_NAMES)
    assert metadata["schema"]["target"] == TARGET

    card = json.loads((models / f"{SELECTED_POLICY_STEM}_card.json").read_text(encoding="utf-8"))
    assert card["feature_names"] == list(FEATURE_NAMES)
    assert card["target"] == TARGET
    assert card["artifact_sha256"] == sha256_file(models / f"{SELECTED_POLICY_STEM}.joblib")
    assert card["selection"]["selected_on"] == "val"

    report = verify_retention_aware(config, output_reports=reports, output_models=models)
    assert report["passed"], [check for check in report["checks"] if not check.passed]


def test_report_generation_is_deterministic(tmp_path: Path) -> None:
    """Two runs of the same protocol must produce identical reports and tables."""
    from captcha_quality.retention_aware import run_retention_aware

    config = load_retention_config(CONFIG_PATH)
    first_reports = tmp_path / "first"
    second_reports = tmp_path / "second"
    run_retention_aware(
        config,
        output_reports=first_reports,
        output_models=tmp_path / "m1",
        verify_images=False,
    )
    run_retention_aware(
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
    from captcha_quality.retention_aware import (
        run_retention_aware,
        verify_retention_aware,
    )

    config = load_retention_config(CONFIG_PATH)
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    run_retention_aware(config, output_reports=reports, output_models=models, verify_images=False)
    metrics_path = reports / "test_slice_metrics.csv"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    report = verify_retention_aware(config, output_reports=reports, output_models=models)
    assert not report["passed"]
    assert any(not check.passed for check in report["checks"])


def test_selected_policy_card_records_the_frozen_calibration_parameters(tmp_path: Path) -> None:
    from captcha_quality.retention_aware import run_retention_aware

    config = load_retention_config(CONFIG_PATH)
    models = tmp_path / "models"
    result, _ = run_retention_aware(
        config,
        output_reports=tmp_path / "reports",
        output_models=models,
        verify_images=False,
    )
    card = json.loads((models / f"{SELECTED_POLICY_STEM}_card.json").read_text(encoding="utf-8"))
    assert card["selection"]["selected_model"] == result.selected_name
    assert card["selection"]["n_candidates"] == len(result.candidates.policies)
    assert card["selection"]["benchmark_config_hash"] == config.config_hash
    if result.selected.calibration is None:
        assert card["calibration_parameters"] is None
    else:
        parameters = card["calibration_parameters"]
        assert set(parameters) == {"intercept", "slope", "clip_output"}
        assert parameters["clip_output"] is config.calibration.clip_output
