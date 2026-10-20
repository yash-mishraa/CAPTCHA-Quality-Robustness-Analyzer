"""Tests for the M5B grouped geometric-robustness benchmark."""

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
    DIAGNOSTIC_LABEL,
    DIAGNOSTIC_SEVERITIES,
    EXPANDED_MODEL_NAME,
    EXPANDED_MODEL_STEM,
    FROZEN_MODEL_NAME,
    M3_NAIVE_NAME,
    PRIMARY_AMPLITUDES_PX,
    PRIMARY_LABEL,
    PRIMARY_SEVERITIES,
    BenchmarkFrames,
    GeometricBenchmarkConfig,
    GeometricBenchmarkConfigError,
    M5APin,
    ProtocolSettings,
    _bootstrap_row,
    _check_feature_leakage,
    _check_severity_protocol,
    _expanded_model_card,
    _parse_protocol,
    _report_markdown,
    build_frame_bundle,
    load_geometric_benchmark_config,
    run_benchmark_core,
    write_benchmark_outputs,
)
from captcha_quality.modelling import (
    TARGET,
    CandidateSpec,
    ModelError,
    fit_candidate,
    load_model_bundle,
    model_inputs,
    regression_metrics,
    validate_feature_columns,
)
from captcha_quality.provenance import sha256_file
from captcha_quality.robustness_benchmark import (
    GridSettings,
    HistGradientBoostingGrid,
    M3Pin,
    ModelPin,
    RandomForestGrid,
    RidgeGrid,
    RunSettings,
    SchemaSettings,
    _metric_lookup,
    metrics_for_frame,
)


def _source_frame(split_counts: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fonts = ("font_a", "font_b", "font_c")
    for split, count in split_counts.items():
        for index in range(count):
            rows.append(
                {
                    "source_id": f"src-{split}-{index:02d}",
                    "split": split,
                    "font_id": fonts[index % len(fonts)],
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
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(sources.itertuples(index=False)):
        clean_hash = f"{source.source_id}-clean-pixels"
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
            image_hash = (
                clean_hash if family == "none" else f"{source.source_id}-m3-{family}-{severity}"
            )
            rows.append(
                {
                    "sample_id": f"{source.source_id}__{family}_s{severity}",
                    "source_id": source.source_id,
                    "split": source.split,
                    "font_id": source.font_id,
                    "text_length": source.text_length,
                    "severity_level": severity,
                    "ssim_target": target,
                    "reference_sha256": clean_hash,
                    "image_sha256": image_hash,
                    "corruption_type": family,
                    **_feature_values(source_index, target),
                }
            )
    return pd.DataFrame(rows)


def _m5a_rows(sources: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    amplitude_by_severity = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0}
    for source_index, source in enumerate(sources.itertuples(index=False)):
        clean_hash = f"{source.source_id}-clean-pixels"
        rows.append(
            {
                "sample_id": f"{source.source_id}__clean",
                "source_id": source.source_id,
                "split": source.split,
                "font_id": source.font_id,
                "text_length": source.text_length,
                "variant_family": "clean",
                "control_id": "none",
                "severity_level": 0,
                "amplitude_px": 0.0,
                "reference_sha256": clean_hash,
                "image_sha256": clean_hash,
                "ssim_target": 1.0,
                **_feature_values(source_index, 1.0),
            }
        )
        for control, target in (
            ("identity", 1.0),
            ("zero_field", 1.0),
            ("minimal", 0.999),
        ):
            rows.append(
                {
                    "sample_id": f"{source.source_id}__{control}",
                    "source_id": source.source_id,
                    "split": source.split,
                    "font_id": source.font_id,
                    "text_length": source.text_length,
                    "variant_family": "control",
                    "control_id": control,
                    "severity_level": 0,
                    "amplitude_px": 0.0,
                    "reference_sha256": clean_hash,
                    "image_sha256": (
                        clean_hash
                        if control in {"identity", "zero_field"}
                        else f"{source.source_id}-minimal"
                    ),
                    "ssim_target": target,
                    **_feature_values(source_index, target),
                }
            )
        for severity in range(1, 5):
            target = (
                0.992 - 0.001 * (source_index % 3)
                if severity == 1
                else 1.0 - 0.04 * severity - 0.005 * (source_index % 3)
            )
            target = float(np.clip(target, 0.0, 1.0))
            rows.append(
                {
                    "sample_id": f"{source.source_id}__elastic_s{severity}",
                    "source_id": source.source_id,
                    "split": source.split,
                    "font_id": source.font_id,
                    "text_length": source.text_length,
                    "variant_family": "elastic",
                    "control_id": "none",
                    "severity_level": severity,
                    "amplitude_px": amplitude_by_severity[severity],
                    "reference_sha256": clean_hash,
                    "image_sha256": f"{source.source_id}-m5a-elastic-{severity}",
                    "ssim_target": target,
                    **_feature_values(source_index, target),
                }
            )
    return pd.DataFrame(rows)


def _summary() -> dict[str, object]:
    return {
        "dataset_name": "synthetic-m5b",
        "dataset_version": "v-test",
        "feature_set_version": FEATURE_SET_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "generation_config_hash": "a" * 64,
        "split_policy": {"assignment_sha256": "b" * 64},
        "counts": {"sources": 30, "samples": 240},
        "target": {"name": TARGET},
    }


def _synthetic_frames() -> BenchmarkFrames:
    m3_sources = _source_frame({"train": 18, "val": 6, "test": 6})
    m5a_sources = m3_sources.copy()
    m3 = _m3_rows(m3_sources)
    m5a = _m5a_rows(m5a_sources)
    return build_frame_bundle(m3, m5a, m3_sources, m5a_sources, _summary(), _summary())


def _test_config(tmp_path: Path) -> GeometricBenchmarkConfig:
    return GeometricBenchmarkConfig(
        run=RunSettings(
            name="synthetic-m5b",
            version="v-test",
            seed=17,
            n_bootstrap=20,
            selection_metric="rmse",
            output_reports=tmp_path / "reports",
            output_models=tmp_path / "models",
        ),
        m3=M3Pin(
            dataset=tmp_path / "m3",
            config=tmp_path / "m3.toml",
            dataset_version="v-test",
            generation_config_hash="a" * 64,
            manifest_sha256="1" * 64,
            sources_sha256="2" * 64,
            features_sha256="3" * 64,
            ground_truth_sha256="4" * 64,
            summary_sha256="5" * 64,
            split_assignment_sha256="b" * 64,
        ),
        m5a=M5APin(
            dataset=tmp_path / "m5a",
            config=tmp_path / "m5a.toml",
            dataset_version="v-test",
            config_sha256="6" * 64,
            pilot_config_hash="c" * 64,
            manifest_sha256="7" * 64,
            sources_sha256="8" * 64,
            features_sha256="9" * 64,
            summary_sha256="d" * 64,
            image_pixel_hash_set="e" * 64,
            n_samples=240,
            n_sources=30,
            gate_verdict="CONDITIONAL PASS",
        ),
        model=ModelPin(
            model_dir=tmp_path / "frozen",
            model_stem="primary_model",
            card_sha256="f" * 64,
            artifact_sha256="0" * 64,
        ),
        schema=SchemaSettings(
            feature_set_version=FEATURE_SET_VERSION,
            target=TARGET,
            feature_names=FEATURE_NAMES,
            split_names=("train", "val", "test"),
        ),
        grid=GridSettings(
            include_ols=True,
            ridge=RidgeGrid(alphas=(0.1,)),
            random_forest=RandomForestGrid(
                n_estimators=20, max_depths=(4,), min_samples_leafs=(1,)
            ),
            hist_gradient_boosting=HistGradientBoostingGrid(
                max_iter=20,
                learning_rates=(0.1,),
                max_leaf_nodes=(15,),
                min_samples_leaf=10,
                l2_regularization=0.0,
            ),
        ),
        protocol=ProtocolSettings(
            primary_amplitudes_px=PRIMARY_AMPLITUDES_PX,
            diagnostic_amplitudes_px=DIAGNOSTIC_AMPLITUDES_PX,
            primary_severities=PRIMARY_SEVERITIES,
            diagnostic_severities=DIAGNOSTIC_SEVERITIES,
            exclude_control_rows_from_fit=True,
        ),
        config_path=tmp_path / "benchmark.toml",
        config_sha256="0" * 64,
    )


def _frozen_estimator(frames: BenchmarkFrames):
    spec = CandidateSpec(
        name="frozen",
        family="random_forest",
        params={"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 1},
        n_inputs=len(FEATURE_NAMES),
    )
    train = frames.m3_frame[frames.m3_frame["split"] == "train"]
    return fit_candidate(spec, train, seed=3).estimator


def test_frame_bundle_excludes_diagnostic_and_clean_controls() -> None:
    frames = _synthetic_frames()
    assert frames.ok
    assert len(frames.expanded_train) == 18 * 12
    assert len(frames.expanded_val) == 6 * 12
    assert len(frames.primary_test) == 6 * 3
    assert len(frames.diagnostic_test) == 6
    assert set(frames.primary_test["severity_level"]) == {2, 3, 4}
    assert set(frames.diagnostic_test["severity_level"]) == {1}
    assert set(frames.expanded_train["source_id"]).isdisjoint(frames.primary_test["source_id"])
    assert not (
        (frames.expanded_train["dataset"] == "m5a") & frames.expanded_train["is_clean"]
    ).any()
    assert not (
        (frames.expanded_train["dataset"] == "m5a") & frames.expanded_train["is_control"]
    ).any()
    assert not (frames.expanded_train["amplitude_px"] == 0.5).any()
    assert not (frames.expanded_val["amplitude_px"] == 0.5).any()
    assert frames.expanded_train["sample_id"].is_unique
    assert frames.expanded_train["image_sha256"].is_unique
    assert set(frames.expanded_train["protocol_scope"]) == {"primary", "not_applicable"}


def test_tampered_source_split_fails_isolation() -> None:
    frames = _synthetic_frames()
    m5a = frames.m5a_frame.copy()
    source = m5a.loc[m5a["split"] == "test", "source_id"].iloc[0]
    m5a.loc[m5a["source_id"] == source, "split"] = "train"
    broken = build_frame_bundle(
        frames.m3_frame,
        m5a,
        frames.m3_sources,
        frames.m5a_sources,
        frames.m3_summary,
        frames.m5a_summary,
    )
    assert not broken.ok
    assert any(
        check.name == "frame_source_assignment" and not check.passed for check in broken.checks
    )


def test_diagnostic_rung_is_rejected_from_primary_fit() -> None:
    frames = _synthetic_frames()
    contaminated_fit = pd.concat([frames.expanded_train, frames.diagnostic_test], ignore_index=True)
    check = _check_severity_protocol(
        frames.m5a_frame,
        contaminated_fit,
        frames.expanded_val,
        frames.primary_test,
        frames.diagnostic_test,
    )
    assert not check.passed
    assert "non-primary" in check.detail or "0.5" in check.detail


def test_feature_leakage_guard_rejects_metadata_columns() -> None:
    frames = _synthetic_frames()
    valid = _check_feature_leakage(
        frames.m3_frame[["sample_id", *FEATURE_NAMES]],
        frames.m5a_frame[["sample_id", *FEATURE_NAMES]],
        frames.expanded_train,
    )
    assert valid.passed
    assert model_inputs(frames.expanded_train).shape[1] == len(FEATURE_NAMES)
    with pytest.raises(ModelError):
        validate_feature_columns([*FEATURE_NAMES, "severity_level"])
    leaked = frames.m5a_frame[["sample_id", *FEATURE_NAMES]].copy()
    leaked["severity_level"] = frames.m5a_frame["severity_level"].to_numpy()
    broken = _check_feature_leakage(
        frames.m3_frame[["sample_id", *FEATURE_NAMES]],
        leaked,
        frames.expanded_train,
    )
    assert not broken.passed


def test_config_rejects_diagnostic_amplitude_in_primary() -> None:
    with pytest.raises(GeometricBenchmarkConfigError):
        _parse_protocol(
            {
                "primary_amplitudes_px": [0.5, 1.0, 2.0, 3.0],
                "diagnostic_amplitudes_px": [0.5],
                "primary_severities": [1, 2, 3, 4],
                "diagnostic_severities": [1],
                "exclude_control_rows_from_fit": True,
            }
        )


def test_shipped_config_declares_primary_and_diagnostic_protocol() -> None:
    config = load_geometric_benchmark_config("configs/geometric_robustness_benchmark_v0_1_0.toml")
    assert config.protocol.primary_amplitudes_px == PRIMARY_AMPLITUDES_PX
    assert config.protocol.diagnostic_amplitudes_px == DIAGNOSTIC_AMPLITUDES_PX
    assert config.m5a.gate_verdict == "CONDITIONAL PASS"
    assert len(json.dumps(config.identity_payload())) > 0


def test_metric_correctness_for_primary_frame() -> None:
    frames = _synthetic_frames()
    target = frames.primary_test[TARGET].to_numpy(dtype=np.float64)
    scores = regression_metrics(target, target.copy())
    assert scores["mae"] == pytest.approx(0.0)
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["r2"] == pytest.approx(1.0)

    table = metrics_for_frame(
        frames.primary_test,
        target + 0.01,
        model="probe",
        dataset=PRIMARY_LABEL,
        split="test",
    )
    overall = table[(table["scope"] == "overall") & (table["level"] == "all")].iloc[0]
    assert overall["n_samples"] == len(frames.primary_test)
    assert overall["n_sources"] == 6
    assert overall["mae"] == pytest.approx(0.01)
    assert overall["mean_residual"] == pytest.approx(-0.01)
    assert (table["scope"] == "severity").sum() == 3


def test_grouped_bootstrap_is_deterministic_and_grouped() -> None:
    frames = _synthetic_frames()
    predictions = frames.primary_test[TARGET].to_numpy(dtype=np.float64) - 0.01
    first = _bootstrap_row(
        frames.primary_test,
        predictions,
        model="probe",
        dataset=PRIMARY_LABEL,
        scope="overall",
        level="all",
        n_bootstrap=25,
        seed=11,
        evaluation="primary_geometric",
    )
    second = _bootstrap_row(
        frames.primary_test,
        predictions,
        model="probe",
        dataset=PRIMARY_LABEL,
        scope="overall",
        level="all",
        n_bootstrap=25,
        seed=11,
        evaluation="primary_geometric",
    )
    pd.testing.assert_frame_equal(first, second)
    mae = first[first["metric"] == "mae"].iloc[0]
    assert int(mae["n_groups"]) == 6
    assert mae["ci_low"] <= mae["estimate"] <= mae["ci_high"]
    assert mae["estimate"] == pytest.approx(0.01)


def test_core_experiments_and_persistence_replay(tmp_path: Path) -> None:
    config = _test_config(tmp_path)
    frames = _synthetic_frames()
    frozen = _frozen_estimator(frames)
    result = run_benchmark_core(
        frames,
        config,
        frozen_estimator=frozen,
        frozen_card={"feature_names": FEATURE_NAMES, "target": TARGET},
    )
    assert result.ok
    for dataset in (PRIMARY_LABEL, DIAGNOSTIC_LABEL):
        models = set(
            result.adaptation_metrics[result.adaptation_metrics["dataset"] == dataset]["model"]
        )
        assert {FROZEN_MODEL_NAME, EXPANDED_MODEL_NAME}.issubset(models)
    assert set(result.m3_retention_metrics["model"]) == {
        FROZEN_MODEL_NAME,
        EXPANDED_MODEL_NAME,
        M3_NAIVE_NAME,
        "naive_expanded_train_mean",
    }
    assert len(result.m3_retention_metrics) > 0
    assert len(result.retention_comparison) == 5
    assert set(result.retention_comparison["scope"]) == {
        "overall",
        "clean",
        "corrupted",
        "family",
    }
    adapted = _metric_lookup(
        result.adaptation_metrics, EXPANDED_MODEL_NAME, PRIMARY_LABEL, "overall", "all"
    )
    assert adapted["n_samples"] == len(frames.primary_test)
    assert adapted["n_sources"] == 6
    diagnostic = _metric_lookup(
        result.adaptation_metrics, EXPANDED_MODEL_NAME, DIAGNOSTIC_LABEL, "overall", "all"
    )
    assert diagnostic["n_samples"] == len(frames.diagnostic_test)
    assert len(result.uncertainty) > 0
    assert (result.uncertainty["evaluation"] == "diagnostic_0_5").any()

    report_first = _report_markdown(config, result)
    report_second = _report_markdown(config, result)
    assert report_first == report_second
    assert "Experiment A" in report_first
    assert "Experiment B" in report_first
    assert "Experiment C" in report_first
    assert "0.5 px" in report_first

    reports, models, paths = write_benchmark_outputs(
        result,
        config,
        frozen_before={"frozen": {"sha256": "a"}},
        frozen_after={"frozen": {"sha256": "a"}},
        elapsed_seconds=1.25,
    )
    metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["expanded_model"]["artifact_sha256"] == sha256_file(
        models / f"{EXPANDED_MODEL_STEM}.joblib"
    )
    assert metadata["frozen_assets"]["all_unchanged"] is True
    assert paths["report"].is_file()

    loaded, card, checks = load_model_bundle(
        models, EXPANDED_MODEL_STEM, dataset_summary=frames.m3_summary
    )
    assert loaded is not None
    assert all(check.passed for check in checks)
    assert card["feature_names"] == list(FEATURE_NAMES)
    primary_predictions = pd.read_csv(reports / "adaptation_predictions.csv")
    selected = primary_predictions[
        (primary_predictions["model"] == EXPANDED_MODEL_NAME)
        & (primary_predictions["dataset_label"] == PRIMARY_LABEL)
    ]
    assert len(selected) == len(frames.primary_test)
    assert np.allclose(
        selected["prediction"].to_numpy(),
        loaded.predict(model_inputs(frames.primary_test)),
        rtol=0.0,
        atol=1e-9,
    )

    card_payload = _expanded_model_card(config, result)
    assert card_payload["dataset"]["primary_amplitudes_px"] == list(PRIMARY_AMPLITUDES_PX)
    assert card_payload["dataset"]["diagnostic_amplitudes_px"] == list(DIAGNOSTIC_AMPLITUDES_PX)
