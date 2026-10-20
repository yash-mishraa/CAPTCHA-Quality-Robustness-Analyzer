"""Unit tests for conventional models, metrics and model persistence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import (
    CandidateSpec,
    ModelError,
    bootstrap_metrics,
    fit_candidate,
    load_model_bundle,
    model_inputs,
    predict,
    regression_metrics,
    save_model_bundle,
    select_best,
    sklearn_version,
    validate_feature_columns,
)


def _frame(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(n, len(FEATURE_NAMES)))
    frame = pd.DataFrame(values, columns=list(FEATURE_NAMES))
    frame.insert(0, "sample_id", [f"sample-{i:03d}" for i in range(n)])
    frame.insert(1, "source_id", [f"source-{i // 3:02d}" for i in range(n)])
    frame["corruption_type"] = "gaussian_noise"
    frame["severity_level"] = 2
    frame["font_id"] = "font_a"
    frame["text_length"] = 5
    frame["ssim_target"] = 0.9 - 0.1 * frame["ink_fraction"] + rng.normal(scale=0.01, size=n)
    frame["psnr_target"] = 20.0
    return frame


def _ols_spec() -> CandidateSpec:
    return CandidateSpec(name="ols", family="ols", params={}, n_inputs=len(FEATURE_NAMES))


def _rf_spec() -> CandidateSpec:
    return CandidateSpec(
        name="rf",
        family="random_forest",
        params={
            "n_estimators": 15,
            "max_depth": 3,
            "min_samples_leaf": 1,
        },
        n_inputs=len(FEATURE_NAMES),
    )


def _card(dataset_hash: str = "a" * 64, split_hash: str = "b" * 64) -> dict:
    return {
        "model_name": "ols",
        "model_family": "ols",
        "params": {},
        "target": "ssim_target",
        "feature_names": list(FEATURE_NAMES),
        "feature_set_version": "v1",
        "n_features": len(FEATURE_NAMES),
        "dataset": {
            "generation_config_hash": dataset_hash,
            "split_assignment_sha256": split_hash,
        },
        "sklearn_version": sklearn_version(),
    }


def _summary(dataset_hash: str = "a" * 64, split_hash: str = "b" * 64) -> dict:
    return {
        "feature_set_version": "v1",
        "generation_config_hash": dataset_hash,
        "split_policy": {"assignment_sha256": split_hash},
    }


def test_feature_only_inputs_exclude_target_and_metadata() -> None:
    frame = _frame()
    inputs = model_inputs(frame)
    assert inputs.shape == (len(frame), len(FEATURE_NAMES))
    np.testing.assert_allclose(inputs, frame[list(FEATURE_NAMES)].to_numpy())
    # The merged frame may carry metadata; the explicit selection never does.
    assert "ssim_target" not in FEATURE_NAMES
    assert "severity_level" not in FEATURE_NAMES


def test_feature_table_schema_rejects_extra_columns() -> None:
    with pytest.raises(ModelError, match="forbidden columns"):
        validate_feature_columns(["sample_id", *FEATURE_NAMES, "ssim_target"])
    with pytest.raises(ModelError, match="forbidden columns"):
        validate_feature_columns(["sample_id", *FEATURE_NAMES, "font_id"])
    assert validate_feature_columns(["sample_id", *FEATURE_NAMES]) == FEATURE_NAMES


def test_regression_metrics_and_edge_cases() -> None:
    perfect = regression_metrics([0.2, 0.4, 0.6], [0.2, 0.4, 0.6])
    assert perfect["mae"] == 0.0
    assert perfect["rmse"] == 0.0
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["r2_defined"] is True

    constant = regression_metrics([0.5, 0.5], [0.4, 0.6])
    assert math.isnan(constant["r2"])
    assert constant["r2_defined"] is False
    assert constant["mae"] == pytest.approx(0.1)

    empty = regression_metrics([], [])
    assert empty["n_rows"] == 0
    assert math.isnan(empty["rmse"])
    assert empty["r2_defined"] is False

    with pytest.raises(ModelError, match="shapes differ"):
        regression_metrics([0.1, 0.2], [0.1])


def test_selection_uses_validation_only() -> None:
    rows = [
        {"name": "a", "family": "ols", "n_inputs": 8, "val_rmse": 0.10, "test_rmse": 99.0},
        {"name": "b", "family": "ridge", "n_inputs": 8, "val_rmse": 0.20, "test_rmse": 0.0},
    ]
    assert select_best(rows, "rmse")["name"] == "a"
    # Naive is a reference, never a selected learned candidate.
    with pytest.raises(ModelError, match="no learned candidates"):
        select_best(
            [{"name": "naive", "family": "naive", "n_inputs": 0, "val_rmse": 0.1}],
            "rmse",
        )


def test_scaler_is_fitted_on_training_rows_only() -> None:
    frame = _frame(12)
    train = frame.iloc[:6].copy()
    fitted = fit_candidate(_ols_spec(), train, seed=3)
    scaler = fitted.estimator.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, train[list(FEATURE_NAMES)].to_numpy().mean(axis=0))
    full_mean = frame[list(FEATURE_NAMES)].to_numpy().mean(axis=0)
    # A full-data scaler would differ; this dataset is constructed to expose that.
    assert not np.allclose(scaler.mean_, full_mean)


def test_random_forest_fit_is_deterministic() -> None:
    frame = _frame()
    first = fit_candidate(_rf_spec(), frame, seed=11)
    second = fit_candidate(_rf_spec(), frame, seed=11)
    np.testing.assert_allclose(predict(first.estimator, frame), predict(second.estimator, frame))


def test_bootstrap_resamples_source_groups_deterministically() -> None:
    frame = _frame(12).iloc[:9].copy()
    frame["source_id"] = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    predicted = frame["ssim_target"].to_numpy() + 0.01
    first = bootstrap_metrics(frame, predicted, n_bootstrap=50, seed=4, scope="unit")
    second = bootstrap_metrics(frame, predicted, n_bootstrap=50, seed=4, scope="unit")
    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {"mae", "rmse", "r2"}
    assert (first["n_groups"] == 3).all()
    assert (first["n_samples"] == 9).all()
    assert first.loc[first["metric"] == "r2", "r2_defined_rate"].iloc[0] == 1.0


def test_persistence_roundtrip_and_schema_rejection(tmp_path: Path) -> None:
    frame = _frame()
    fitted = fit_candidate(_ols_spec(), frame, seed=5)
    paths = save_model_bundle(tmp_path, "model", fitted.estimator, _card())
    assert paths["artifact"].is_file()
    assert paths["card"].is_file()

    model, card, checks = load_model_bundle(tmp_path, "model", dataset_summary=_summary())
    assert model is not None
    assert card["model_name"] == "ols"
    assert all(check.passed for check in checks)
    hash_check = next(check for check in checks if check.name == "artifact_hash")
    assert hash_check.detail.startswith("artifact SHA-256 ")

    # Wrong feature order/schema is rejected before deserialisation is trusted.
    tampered_card = json.loads(paths["card"].read_text(encoding="utf-8"))
    tampered_card["feature_names"] = list(reversed(FEATURE_NAMES))
    paths["card"].write_text(json.dumps(tampered_card), encoding="utf-8")
    model, _, checks = load_model_bundle(tmp_path, "model", dataset_summary=_summary())
    assert model is None
    assert any(not check.passed and check.name == "feature_schema" for check in checks)

    # Rewrite a clean card, then corrupt the artifact.
    save_model_bundle(tmp_path, "model", fitted.estimator, _card())
    with paths["artifact"].open("ab") as handle:
        handle.write(b"tampered")
    model, _, checks = load_model_bundle(tmp_path, "model", dataset_summary=_summary())
    assert model is None
    assert any(not check.passed and check.name == "artifact_hash" for check in checks)


def test_persistence_rejects_mismatched_dataset_identity(tmp_path: Path) -> None:
    frame = _frame()
    fitted = fit_candidate(_ols_spec(), frame, seed=5)
    save_model_bundle(tmp_path, "model", fitted.estimator, _card())
    model, _, checks = load_model_bundle(
        tmp_path, "model", dataset_summary=_summary(dataset_hash="c" * 64)
    )
    assert model is None
    assert any(not check.passed and check.name == "dataset_identity" for check in checks)

    model, _, checks = load_model_bundle(
        tmp_path,
        "model",
        dataset_summary={
            **_summary(),
            "feature_set_version": "v2",
        },
    )
    assert model is None
    assert any(not check.passed and check.name == "feature_schema" for check in checks)


def test_persistence_rejects_missing_card(tmp_path: Path) -> None:
    model, card, checks = load_model_bundle(tmp_path, "absent", dataset_summary=_summary())
    assert model is None
    assert card == {}
    assert checks[0].name == "model_card"
    assert not checks[0].passed
