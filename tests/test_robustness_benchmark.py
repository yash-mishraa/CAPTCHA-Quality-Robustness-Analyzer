"""Tests for the M4B grouped occlusion/interference robustness benchmark."""

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
from captcha_quality.modelling import (
    TARGET,
    CandidateSpec,
    ModelError,
    fit_candidate,
    load_model_bundle,
    model_inputs,
    validate_feature_columns,
)
from captcha_quality.provenance import sha256_bytes
from captcha_quality.robustness_benchmark import (
    EXPANDED_MODEL_NAME,
    EXPANDED_MODEL_STEM,
    FROZEN_MODEL_NAME,
    BenchmarkFrames,
    GridSettings,
    HistGradientBoostingGrid,
    M3Pin,
    M4APin,
    ModelPin,
    RandomForestGrid,
    RidgeGrid,
    RobustnessConfig,
    RobustnessConfigError,
    RunSettings,
    SchemaSettings,
    build_frame_bundle,
    candidate_specs_from_config,
    fit_and_select,
    load_robustness_config,
    metrics_for_frame,
    paired_grouped_bootstrap,
    run_benchmark_core,
    write_benchmark_outputs,
)


def _source_frame(split_counts: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fonts = ("font_a", "font_b", "font_c")
    for split, count in split_counts.items():
        for index in range(count):
            source_id = f"src-{split}-{index:02d}"
            rows.append(
                {
                    "source_id": source_id,
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


def _frame_rows(
    sources: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(sources.itertuples(index=False)):
        clean_hash = f"{source.source_id}-clean-pixels"
        variants: list[tuple[str, int, float]] = [("none", 0, 1.0)]
        if dataset == "m3":
            for severity in range(1, 5):
                variants.append(
                    ("gaussian_blur", severity, 1.0 - 0.12 * severity - 0.01 * (source_index % 3))
                )
                variants.append(
                    ("gaussian_noise", severity, 1.0 - 0.08 * severity - 0.01 * (source_index % 3))
                )
        else:
            for severity in range(1, 5):
                variants.append(
                    ("thin_lines", severity, 1.0 - 0.07 * severity - 0.01 * (source_index % 3))
                )
                variants.append(
                    ("arc_strokes", severity, 1.0 - 0.06 * severity - 0.01 * (source_index % 3))
                )
        for family, severity, target in variants:
            target = float(np.clip(target, 0.0, 1.0))
            sample_id = f"{source.source_id}__{family}_s{severity}"
            image_hash = (
                clean_hash
                if family == "none"
                else f"{source.source_id}-{dataset}-{family}-{severity}"
            )
            row = {
                "sample_id": sample_id,
                "source_id": source.source_id,
                "split": source.split,
                "font_id": source.font_id,
                "text_length": source.text_length,
                "severity_level": severity,
                "ssim_target": target,
                "reference_sha256": clean_hash,
                "image_sha256": image_hash,
                **_feature_values(source_index, target),
            }
            if dataset == "m3":
                row["corruption_type"] = family
            else:
                row["variant_family"] = family
            rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_frames() -> BenchmarkFrames:
    m3_sources = _source_frame({"train": 18, "val": 6, "test": 6})
    m4a_sources = m3_sources.copy()
    m3 = _frame_rows(m3_sources, dataset="m3")
    m4a = _frame_rows(m4a_sources, dataset="m4a")
    summary = {
        "dataset_name": "synthetic-m4b",
        "dataset_version": "v-test",
        "feature_set_version": FEATURE_SET_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "generation_config_hash": "a" * 64,
        "split_policy": {"assignment_sha256": "b" * 64},
        "counts": {"sources": 30, "samples": 270},
    }
    return build_frame_bundle(m3, m4a, m3_sources, m4a_sources, summary, summary)


def _test_config(tmp_path: Path) -> RobustnessConfig:
    return RobustnessConfig(
        run=RunSettings(
            name="synthetic-m4b",
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
        m4a=M4APin(
            dataset=tmp_path / "m4a",
            config=tmp_path / "m4a.toml",
            dataset_version="v-test",
            pilot_config_hash="c" * 64,
            manifest_sha256="6" * 64,
            sources_sha256="7" * 64,
            features_sha256="8" * 64,
            summary_sha256="9" * 64,
            image_pixel_hash_set="d" * 64,
            n_samples=270,
            n_sources=30,
        ),
        model=ModelPin(
            model_dir=tmp_path / "frozen",
            model_stem="primary_model",
            card_sha256="e" * 64,
            artifact_sha256="f" * 64,
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


def test_frame_bundle_keeps_sources_grouped_and_drops_m4a_clean_from_fit() -> None:
    frames = _synthetic_frames()
    assert frames.ok
    assert len(frames.expanded_train) == 18 * 17
    assert len(frames.expanded_val) == 6 * 17
    assert len(frames.m4a_test) == 6 * 9
    assert set(frames.expanded_train["source_id"]).isdisjoint(frames.m4a_test["source_id"])
    assert not (
        (frames.expanded_train["dataset"] == "m4a") & frames.expanded_train["is_clean"]
    ).any()
    assert not ((frames.expanded_val["dataset"] == "m4a") & frames.expanded_val["is_clean"]).any()
    assert frames.expanded_train["sample_id"].is_unique
    assert frames.expanded_train["image_sha256"].is_unique

    broken = frames.m4a_frame.copy()
    train_source = frames.m4a_sources.loc[frames.m4a_sources["split"] == "train", "source_id"].iloc[
        0
    ]
    row_index = broken.index[(broken["source_id"] == train_source) & (broken["split"] == "train")][
        0
    ]
    broken.loc[row_index, "split"] = "test"
    bad_frames = build_frame_bundle(
        frames.m3_frame,
        broken,
        frames.m3_sources,
        frames.m4a_sources,
        frames.m3_summary,
        frames.m4a_summary,
    )
    assert not bad_frames.ok
    assert any(
        check.name == "m4a_source_isolation" and not check.passed for check in bad_frames.checks
    )


def test_feature_inputs_are_exactly_the_eight_image_features() -> None:
    frames = _synthetic_frames()
    validate_feature_columns(["sample_id", *FEATURE_NAMES])
    with pytest.raises(ModelError):
        validate_feature_columns(["sample_id", *FEATURE_NAMES, "severity_level"])
    assert model_inputs(frames.expanded_train).shape == (18 * 17, 8)
    assert TARGET not in list(FEATURE_NAMES)
    assert not set(FEATURE_NAMES) & {
        "severity_level",
        "family",
        "font_id",
        "text_length",
        "source_id",
        "split",
    }


def test_metrics_keep_clean_r2_undefined(tmp_path: Path) -> None:
    frames = _synthetic_frames()
    estimator = _frozen_estimator(frames)
    from captcha_quality.modelling import predict

    clean = frames.m4a_test[frames.m4a_test["is_clean"]]
    table = metrics_for_frame(
        clean,
        predict(estimator, clean),
        model=FROZEN_MODEL_NAME,
        dataset="m4a_test",
        split="test",
    )
    clean_row = table[(table["scope"] == "clean") & (table["level"] == "clean")].iloc[0]
    assert not bool(clean_row["r2_defined"])
    assert np.isnan(float(clean_row["r2"]))
    assert int(clean_row["n_sources"]) == 6


def test_candidate_selection_and_bootstrap_are_deterministic(tmp_path: Path) -> None:
    frames = _synthetic_frames()
    config = _test_config(tmp_path)
    candidates_a, fitted_a, selected_a, model_a = fit_and_select(
        frames.expanded_train, frames.expanded_val, config
    )
    candidates_b, _, selected_b, model_b = fit_and_select(
        frames.expanded_train, frames.expanded_val, config
    )
    pd.testing.assert_frame_equal(candidates_a, candidates_b)
    assert selected_a == selected_b
    assert selected_a["name"] in fitted_a
    assert candidates_a["selected"].sum() == 1
    assert set(candidates_a["val_n"]) == {len(frames.expanded_val)}
    assert set(candidates_a["val_n_sources"]) == {6}
    from captcha_quality.modelling import predict

    assert np.allclose(
        predict(model_a.estimator, frames.expanded_val),
        predict(model_b.estimator, frames.expanded_val),
        rtol=0.0,
        atol=0.0,
    )

    a = np.asarray(model_a.estimator.predict(model_inputs(frames.m4a_test)))
    b = a - 0.01
    table = paired_grouped_bootstrap(
        frames.m4a_test,
        a,
        b,
        model_a="a",
        model_b="b",
        dataset="m4a_test",
        scope="overall",
        level="all",
        n_bootstrap=20,
        seed=config.run.seed,
    )
    assert set(table["metric"]) == {"mae", "rmse"}
    assert (table["ci_low"] <= table["estimate"]).all()
    assert (table["estimate"] <= table["ci_high"]).all()
    assert set(table["n_groups"]) == {6}


def test_run_report_outputs_and_model_reload_are_reproducible(tmp_path: Path) -> None:
    frames = _synthetic_frames()
    config = _test_config(tmp_path)
    estimator = _frozen_estimator(frames)
    card = {"feature_names": list(FEATURE_NAMES), "target": TARGET}

    first = run_benchmark_core(frames, config, frozen_estimator=estimator, frozen_card=card)
    second = run_benchmark_core(frames, config, frozen_estimator=estimator, frozen_card=card)
    pd.testing.assert_frame_equal(first.candidates, second.candidates)
    pd.testing.assert_frame_equal(first.zero_shot_metrics, second.zero_shot_metrics)
    pd.testing.assert_frame_equal(first.adaptation_metrics, second.adaptation_metrics)
    pd.testing.assert_frame_equal(first.uncertainty, second.uncertainty)
    assert first.selected["name"] == second.selected["name"]

    reports, models, paths = write_benchmark_outputs(
        first,
        config,
        output_reports=tmp_path / "reports",
        output_models=tmp_path / "models",
    )
    report = (reports / "benchmark.md").read_text(encoding="utf-8")
    assert "Experiment A" in report
    assert "Experiment B" in report
    assert "clean-reference" in report
    assert "R2" in report
    metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["benchmark_config_hash"] == config.config_hash
    assert metadata["expanded_model"]["artifact_sha256"]
    assert (models / f"{EXPANDED_MODEL_STEM}.joblib").is_file()

    loaded, loaded_card, checks = load_model_bundle(
        models, EXPANDED_MODEL_STEM, dataset_summary=frames.m3_summary
    )
    assert all(check.passed for check in checks)
    assert loaded is not None
    assert loaded_card["feature_names"] == list(FEATURE_NAMES)
    actual = np.asarray(loaded.predict(model_inputs(frames.m4a_test)))
    recorded = first.adaptation_predictions[
        first.adaptation_predictions["model"] == EXPANDED_MODEL_NAME
    ]["prediction"].to_numpy(dtype=np.float64)
    assert np.allclose(actual, recorded, rtol=0.0, atol=1e-12)
    assert paths["report"].is_file()


def test_config_loader_and_candidate_grid(tmp_path: Path) -> None:
    config = load_robustness_config("configs/robustness_benchmark_v0_1_0.toml")
    assert config.config_sha256
    assert config.config_hash
    assert len(candidate_specs_from_config(config)) == 12
    assert config.run.n_bootstrap == 1000

    text = Path("configs/robustness_benchmark_v0_1_0.toml").read_text(encoding="utf-8")
    bad = text.replace(
        'name = "occlusion-interference-robustness-benchmark"', 'name = "x"\nunknown_key = 1'
    )
    path = tmp_path / "bad.toml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(RobustnessConfigError):
        load_robustness_config(path)

    bad_schema = text.replace('target = "ssim_target"', 'target = "psnr_target"')
    path.write_text(bad_schema, encoding="utf-8")
    with pytest.raises(RobustnessConfigError):
        load_robustness_config(path)


def test_metadata_hash_uses_canonical_protocol(tmp_path: Path) -> None:
    config = _test_config(tmp_path)
    assert config.config_hash == sha256_bytes(
        json.dumps(config.identity_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    assert config.config_hash == config.config_hash
