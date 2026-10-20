"""End-to-end protocol tests on a small synthetic three-font dataset.

The fixture writes CSV/JSON files directly (no image rendering) so the benchmark
logic can be tested quickly. It deliberately keeps every source group inside one
split and makes the target depend on corruption severity/family, with only weak
font and length effects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from captcha_quality.benchmark import (
    BenchmarkError,
    coverage_assignment,
    load_benchmark_config,
    load_benchmark_dataset,
    main,
    run_benchmark,
    run_heldout_font,
    run_metadata_experiments,
    split_assignment_hash,
    verify_models,
)
from captcha_quality.config import FEATURE_SET_VERSION, ConfigError
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import ModelError

FONTS = ("font_a", "font_b", "font_c")
GENERATION_HASH = "a" * 64
DATASET_VERSION = "v-test"


def _build_synthetic_dataset(root: Path) -> Path:
    manifest_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    source_index = 0
    for _font_index, font in enumerate(FONTS):
        for source_in_font in range(4):
            source_id = f"src-{source_index:03d}"
            split = "train" if source_in_font < 2 else "val" if source_in_font == 2 else "test"
            text_length = (4, 5, 6)[source_index % 3]
            render_seed = 1000 + source_index
            source_rows.append(
                {
                    "source_id": source_id,
                    "font_id": font,
                    "text_length": text_length,
                    "split": split,
                    "render_seed": render_seed,
                }
            )
            base_ink = 0.10 + 0.015 * (source_index % 4)
            base_mean = 214.0 + 2.0 * (source_index % 5)
            base_std = 58.0 + 1.0 * (source_index % 3)
            base_contrast = 0.26 + 0.005 * (source_index % 4)
            base_lap = 4200.0 + 150.0 * (source_index % 6)
            base_edge = 0.080 + 0.004 * (source_index % 4)
            base_entropy = 4.25 + 0.03 * (source_index % 5)
            base_noise = 2.0 + 0.2 * (source_index % 3)

            variants = [("none", 0)]
            variants += [("gaussian_blur", severity) for severity in range(1, 5)]
            variants += [("gaussian_noise", severity) for severity in range(1, 5)]
            for family, severity in variants:
                if family == "none":
                    variant_id = "clean_s0"
                    ink = base_ink
                    mean = base_mean
                    std = base_std
                    contrast = base_contrast
                    laplacian = base_lap
                    edge = base_edge
                    entropy = base_entropy
                    noise = base_noise
                    target = 1.0
                    psnr = 60.0
                    infinite = True
                elif family == "gaussian_blur":
                    variant_id = f"gaussian_blur_s{severity}"
                    ink = base_ink + 0.035 * severity
                    mean = base_mean - 2.0 * severity
                    std = base_std * (1.0 - 0.01 * severity)
                    contrast = base_contrast * (1.0 - 0.02 * severity)
                    laplacian = base_lap * (0.55**severity)
                    edge = base_edge * (0.40**severity)
                    entropy = base_entropy - 0.18 * severity
                    noise = base_noise
                    target = 0.99 - 0.135 * severity + 0.03 * (base_ink - 0.15)
                    target += 0.004 * float(np.sin(float(source_index + severity)))
                    psnr = 23.0 - 1.5 * severity
                    infinite = False
                else:
                    variant_id = f"gaussian_noise_s{severity}"
                    ink = base_ink + 0.004 * severity
                    mean = base_mean + 0.5 * severity
                    std = base_std + 1.5 * severity
                    contrast = base_contrast + 0.003 * severity
                    laplacian = base_lap * (1.0 + 0.15 * severity)
                    edge = base_edge * (1.0 + 0.02 * severity)
                    entropy = base_entropy + 0.03 * severity
                    noise = base_noise + 5.5 * severity
                    target = 0.99 - 0.115 * severity + 0.02 * (base_ink - 0.15)
                    target += 0.004 * float(np.sin(float(source_index + severity)))
                    psnr = 22.0 - 1.2 * severity
                    infinite = False
                target = float(np.clip(target, 0.0, 1.0))
                sample_id = f"{source_id}__{variant_id}"
                image_path = (
                    f"images/clean/{sample_id}.png"
                    if family == "none"
                    else f"images/{family}/{sample_id}.png"
                )
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "variant_id": variant_id,
                        "split_group_id": source_id,
                        "split": split,
                        "text_length": text_length,
                        "font_id": font,
                        "render_seed": render_seed,
                        "corruption_type": family,
                        "severity_level": severity,
                        "corruption_parameters": "{}",
                        "image_path": image_path,
                        "reference_path": f"images/clean/{source_id}.png",
                        "ssim_target": target,
                        "psnr_target": psnr,
                        "psnr_infinite": infinite,
                        "dataset_version": DATASET_VERSION,
                        "generation_config_hash": GENERATION_HASH,
                    }
                )
                feature_rows.append(
                    {
                        "sample_id": sample_id,
                        "intensity_mean": mean,
                        "intensity_std": std,
                        "contrast_rms": contrast,
                        "ink_fraction": float(np.clip(ink, 0.01, 0.80)),
                        "laplacian_variance": laplacian,
                        "edge_density": float(np.clip(edge, 0.0, 0.5)),
                        "histogram_entropy": float(np.clip(entropy, 0.0, 8.0)),
                        "noise_sigma_immerkaer": noise,
                    }
                )
            source_index += 1

    manifest = pd.DataFrame(manifest_rows)
    features = pd.DataFrame(feature_rows)[["sample_id", *FEATURE_NAMES]]
    sources = pd.DataFrame(source_rows)
    root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(root / "manifest.csv", index=False, lineterminator="\n")
    features.to_csv(root / "features.csv", index=False, lineterminator="\n")
    sources.to_csv(root / "sources.csv", index=False, lineterminator="\n")
    pd.DataFrame({"source_id": sources["source_id"], "text": "AB23"}).to_csv(
        root / "ground_truth.csv", index=False, lineterminator="\n"
    )
    summary = {
        "dataset_name": "synthetic-benchmark-test",
        "dataset_version": DATASET_VERSION,
        "generation_config_hash": GENERATION_HASH,
        "feature_set_version": FEATURE_SET_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "split_policy": {"assignment_sha256": split_assignment_hash(sources)},
        "counts": {
            "sources": int(sources["source_id"].nunique()),
            "samples": int(len(manifest)),
        },
    }
    (root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _write_config(path: Path, output: Path, models: Path) -> Path:
    text = f"""
[benchmark]
name = "test_benchmark"
version = "v0"
seed = 7
held_out_font = "font_c"
held_out_font_reason = "Fixture choice: the third font, recorded before fitting."
n_bootstrap = 40
coverage_quantiles = [0.0, 0.5, 1.0]
selection_metric = "rmse"
material_improvement_fraction = 0.05
output_dir = "{output.as_posix()}"
model_dir = "{models.as_posix()}"

[ridge]
alphas = [0.1, 1.0]

[random_forest]
n_estimators = 20
max_depths = [2, 0]
min_samples_leafs = [1, 3]

[hist_gradient_boosting]
max_iter = 30
learning_rates = [0.1]
max_leaf_nodes = [7]
min_samples_leaf = 5
l2_regularization = 0.0
""".strip()
    path.write_text(text + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("benchmark-fixture")
    root = _build_synthetic_dataset(base / "dataset")
    config_path = _write_config(base / "benchmark.toml", base / "reports", base / "models")
    return {"root": root, "config": config_path, "base": base}


@pytest.fixture(scope="module")
def benchmark_run(synthetic: dict[str, Path]):
    config = load_benchmark_config(synthetic["config"])
    return run_benchmark(synthetic["root"], config)


def test_dataset_split_is_grouped_and_frozen(synthetic: dict[str, Path]) -> None:
    bundle = load_benchmark_dataset(synthetic["root"])
    per_source = bundle.frame.groupby("source_id")["split"].nunique()
    assert (per_source == 1).all()
    assert set(bundle.frame["split"]) == {"train", "val", "test"}
    assert bundle.split_assignment_sha256 == split_assignment_hash(bundle.sources)


def test_primary_selection_is_validation_only(benchmark_run) -> None:
    result = benchmark_run.primary
    candidates = result.candidates
    assert "test_rmse" not in candidates.columns
    assert "test_r2" not in candidates.columns
    learned = candidates[candidates["family"] != "naive"]
    expected = learned.sort_values(["val_rmse", "n_inputs", "model"]).iloc[0]["model"]
    assert result.selected["name"] == expected
    assert candidates.loc[candidates["selected"], "model"].tolist() == [expected]
    assert set(result.family_metrics["split"]) == {"train", "val", "test"}
    selected_test = result.family_metrics[
        result.family_metrics["role"].str.contains("overall_selected")
        & (result.family_metrics["split"] == "test")
    ]
    assert len(selected_test) == 1
    assert selected_test["n_rows"].iloc[0] == 27
    assert selected_test["n_sources"].iloc[0] == 3
    assert len(result.bootstrap) == 3


def test_metadata_diagnostics_are_separate(benchmark_run) -> None:
    metrics = benchmark_run.metadata.metrics
    assert set(metrics["model"]) == {
        "metadata_severity_family",
        "metadata_font_text_length",
        "image_plus_metadata",
    }
    assert set(metrics["role"]) == {"diagnostic_only"}
    assert set(metrics["split"]) == {"train", "val", "test"}
    severity = metrics[
        (metrics["model"] == "metadata_severity_family") & (metrics["split"] == "test")
    ].iloc[0]
    assert severity["n_inputs"] == 3  # severity + two family indicators
    font_length = metrics[
        (metrics["model"] == "metadata_font_text_length") & (metrics["split"] == "test")
    ].iloc[0]
    assert font_length["n_inputs"] == 3  # two font indicators + text length
    assert set(benchmark_run.gaps["comparison"]) == {
        "selected_image_only_vs_naive",
        "metadata_severity_family_vs_selected",
        "image_plus_metadata_vs_selected",
    }


def test_heldout_font_isolation(synthetic: dict[str, Path]) -> None:
    bundle = load_benchmark_dataset(synthetic["root"])
    config = load_benchmark_config(synthetic["config"])
    result = run_heldout_font(bundle, config)
    assert result.font == "font_c"
    assert result.train_fonts == ("font_a", "font_b")
    assert result.train_rows == 36  # 2 fonts x 2 sources x 9 variants
    assert result.val_rows == 18  # 2 fonts x 1 source x 9 variants
    assert result.heldout_rows == 36  # 1 font x 4 sources x 9 variants
    assert result.heldout_sources == 4
    assert set(result.predictions["font_id"]) == {"font_c"}
    assert len(result.predictions) == 36
    assert "test_rmse" not in result.candidates.columns
    learned = result.candidates[result.candidates["family"] != "naive"]
    expected = learned.sort_values(["val_rmse", "n_inputs", "model"]).iloc[0]["model"]
    assert result.selected["name"] == expected
    train_rows = result.metrics[result.metrics["split"] == "train_fonts_train"]
    assert int(train_rows["n_sources"].max()) == 4
    heldout_rows = result.metrics[result.metrics["split"] == "heldout_font"]
    assert int(heldout_rows["n_sources"].max()) == 4
    assert "font_c" in set(result.stratified.loc[result.stratified["scope"] == "font", "level"])


def test_metadata_models_use_train_only_categories(synthetic: dict[str, Path]) -> None:
    bundle = load_benchmark_dataset(synthetic["root"])
    result = run_metadata_experiments(bundle)
    font_length = result.metrics[
        (result.metrics["model"] == "metadata_font_text_length")
        & (result.metrics["split"] == "test")
    ].iloc[0]
    assert font_length["n_rows"] == 27
    assert np.isfinite(font_length["rmse"])


def test_coverage_reference_can_exclude_a_font() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "train", "train", "test", "test"],
            "font_id": ["font_a", "font_a", "font_b", "font_c", "font_c", "font_c"],
            "ink_fraction": [0.01, 0.02, 0.03, 0.04, 0.50, 0.60],
        }
    )
    quantiles = (0.0, 0.5, 1.0)
    _, default = coverage_assignment(frame, quantiles)
    _, strict = coverage_assignment(
        frame, quantiles, train_frame=frame[frame["font_id"] != "font_c"]
    )
    assert default["edge_low"].tolist() != strict["edge_low"].tolist()
    assert default["edge_high"].tolist() != strict["edge_high"].tolist()


def test_primary_and_heldout_artifacts_exist(benchmark_run) -> None:
    assert benchmark_run.report_path.is_file()
    assert benchmark_run.metadata_path.is_file()
    for key in ("primary_artifact", "primary_card", "heldout_artifact", "heldout_card"):
        assert benchmark_run.model_paths[key].is_file()
    for filename in (
        "model_comparison.csv",
        "family_test_metrics.csv",
        "metadata_experiments.csv",
        "performance_gaps.csv",
        "stratified_metrics.csv",
        "coverage_groups.csv",
        "feature_correlations.csv",
        "residual_summary.csv",
        "heldout_font_candidates.csv",
        "heldout_font_metrics.csv",
        "heldout_font_predictions.csv",
        "uncertainty.csv",
        "primary_predictions.csv",
    ):
        assert (benchmark_run.output_dir / filename).is_file()


def test_verify_models_report_points_at_dataset(benchmark_run, synthetic: dict[str, Path]) -> None:
    report = verify_models(synthetic["root"], benchmark_run.model_dir)
    assert report.ok
    assert report.dataset == str(synthetic["root"])


def test_report_states_scope_and_checks(benchmark_run) -> None:
    text = benchmark_run.report_path.read_text(encoding="utf-8")
    assert "not a measure of human readability" in text
    assert "PASS** `group_isolation`" in text
    assert "Target-meaningfulness decision" in text
    assert "Held-out-font stress test" in text
    assert "font_c" in text
    assert "no significance test is claimed" in text


def test_run_metadata_records_hashes_and_candidates(benchmark_run) -> None:
    payload = json.loads(benchmark_run.metadata_path.read_text(encoding="utf-8"))
    assert payload["dataset"]["generation_config_hash"] == GENERATION_HASH
    assert payload["dataset"]["split_assignment_sha256"]
    assert payload["target"] == "ssim_target"
    assert payload["feature_names"] == list(FEATURE_NAMES)
    assert payload["selection"]["selected_on"] == "val"
    assert payload["selection"]["n_candidates"] == 8
    assert payload["artifacts"]


def test_cli_run_and_verify_models(
    synthetic: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(
        tmp_path / "cli.toml",
        tmp_path / "reports",
        tmp_path / "models",
    )
    assert main(["run", "--config", str(config_path), "--dataset", str(synthetic["root"])]) == 0
    output = capsys.readouterr().out
    assert "selected:" in output
    assert "test: RMSE" in output
    assert (tmp_path / "reports" / "benchmark.md").is_file()
    assert (tmp_path / "models" / "primary_model.joblib").is_file()

    assert (
        main(
            [
                "verify-model",
                "--dataset",
                str(synthetic["root"]),
                "--model-dir",
                str(tmp_path / "models"),
            ]
        )
        == 0
    )
    assert "model verification: PASS" in capsys.readouterr().out

    card_path = tmp_path / "models" / "primary_model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["dataset"]["generation_config_hash"] = "0" * 64
    card_path.write_text(json.dumps(card), encoding="utf-8")
    assert (
        main(
            [
                "verify-model",
                "--dataset",
                str(synthetic["root"]),
                "--model-dir",
                str(tmp_path / "models"),
            ]
        )
        == 1
    )
    assert "model verification: FAIL" in capsys.readouterr().out


def test_cli_config_error_returns_two(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[benchmark]\nname = 'bad'\n", encoding="utf-8")
    assert main(["run", "--config", str(bad), "--dataset", str(tmp_path)]) == 2
    assert "error:" in capsys.readouterr().err


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(
        """
[benchmark]
name = "x"
version = "v0"
seed = 1
held_out_font = "font_a"
held_out_font_reason = "x"
n_bootstrap = 0
coverage_quantiles = [0.0, 0.5, 1.0]
selection_metric = "rmse"
material_improvement_fraction = 0.05
output_dir = "reports/x"
model_dir = "models/x"
unexpected = true

[ridge]
alphas = [1.0]

[random_forest]
n_estimators = 1
max_depths = [2]
min_samples_leafs = [1]

[hist_gradient_boosting]
max_iter = 1
learning_rates = [0.1]
max_leaf_nodes = [2]
min_samples_leaf = 1
l2_regularization = 0.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_benchmark_config(config)


def test_heldout_font_must_exist(synthetic: dict[str, Path]) -> None:
    bundle = load_benchmark_dataset(synthetic["root"])
    config = load_benchmark_config(synthetic["config"])
    from dataclasses import replace

    config = replace(config, held_out_font="absent_font")
    with pytest.raises(BenchmarkError, match="not in"):
        run_heldout_font(bundle, config)


def test_validate_feature_columns_still_rejects_leaks() -> None:
    from captcha_quality.modelling import validate_feature_columns

    with pytest.raises(ModelError):
        validate_feature_columns(["sample_id", *FEATURE_NAMES, "severity_level"])
