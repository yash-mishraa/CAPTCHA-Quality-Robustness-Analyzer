"""Tests for the M7B held-out-font generalisation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from captcha_quality import heldout_font_benchmark as hfb
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import TARGET, bootstrap_metrics
from captcha_quality.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "heldout_font_benchmark_v0_1_0.toml"


def _fonts_available(config: hfb.HeldoutConfig) -> bool:
    return all(font.path.is_file() for font in config.fonts)


@pytest.fixture(scope="module")
def config() -> hfb.HeldoutConfig:
    loaded = hfb.load_config(CONFIG_PATH)
    if not _fonts_available(loaded):  # pragma: no cover - platform dependent
        pytest.skip("M7B held-out fonts are not installed on this machine")
    return loaded


@pytest.fixture(scope="module")
def benchmark_bundle(
    tmp_path_factory: pytest.TempPathFactory, config: hfb.HeldoutConfig
) -> tuple[hfb.DatasetResult, hfb.EvaluationResult, Path]:
    root = tmp_path_factory.mktemp("m7b")
    dataset = hfb.generate_dataset(config, output_dataset=root / "dataset", overwrite=True)
    evaluation = hfb.evaluate_dataset(config, dataset_root=dataset.root, verify_images=True)
    reports = root / "reports"
    hfb._write_outputs(evaluation, config, output_reports=reports, overwrite=True)
    return dataset, evaluation, reports


def _mutated_config(tmp_path: Path, old: str, new: str) -> Path:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "mutated_m7b.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Configuration, fonts and output isolation
# ---------------------------------------------------------------------------


def test_config_parses_pinned_protocol(config: hfb.HeldoutConfig) -> None:
    assert config.run.name == "heldout-font-generalization"
    assert config.run.version == "v0.1.0"
    assert config.run.seed == 20261109
    assert config.run.n_bootstrap == 2000
    assert config.schema.feature_names == FEATURE_NAMES
    assert config.schema.target == TARGET
    assert config.schema.split_names == ("eval",)
    assert config.font_keys == ("georgia", "monaco", "impact")
    assert config.config_sha256 == sha256_file(CONFIG_PATH)
    assert len(config.generation_hash) == 64
    assert config.generation_hash != config.config_hash
    assert config.protocol.near_degenerate_sd == 0.005
    assert config.protocol.min_sources_per_font == 10


def test_config_rejects_unknown_keys_and_wrong_schema(tmp_path: Path) -> None:
    unknown = _mutated_config(tmp_path, "[run]\n", "[run]\nunknown_key = 1\n")
    with pytest.raises(hfb.HeldoutFontConfigError, match="unknown key"):
        hfb.load_config(unknown)

    wrong_feature = _mutated_config(
        tmp_path,
        'feature_names = [\n  "intensity_mean",',
        'feature_names = [\n  "not_a_frozen_feature",',
    )
    with pytest.raises(hfb.HeldoutFontConfigError, match="feature_names"):
        hfb.load_config(wrong_feature)

    wrong_split = _mutated_config(tmp_path, 'split_names = ["eval"]', 'split_names = ["test"]')
    with pytest.raises(hfb.HeldoutFontConfigError, match="split_names"):
        hfb.load_config(wrong_split)

    missing = tmp_path / "missing.toml"
    with pytest.raises(hfb.HeldoutFontConfigError, match="missing M7B configuration"):
        hfb.load_config(missing)


def test_output_paths_cannot_target_frozen_assets() -> None:
    with pytest.raises(hfb.HeldoutFontError, match="protected frozen path"):
        hfb._safe_output_path("configs", "test")
    with pytest.raises(hfb.HeldoutFontError, match="protected frozen path"):
        hfb._safe_output_path("reports/retention_aware_selection_v0_1_0", "test")
    with pytest.raises(hfb.HeldoutFontError, match="repository root"):
        hfb._safe_output_path(REPO_ROOT, "test")


def test_font_identity_and_missing_font_fail_closed(
    config: hfb.HeldoutConfig, tmp_path: Path
) -> None:
    resolution = hfb.resolve_heldout_fonts(config)
    assert [font.key for font in resolution] == ["georgia", "monaco", "impact"]
    for font in config.fonts:
        assert font.sha256 not in hfb.FROZEN_FONT_HASHES
        assert font.key not in hfb.FROZEN_FONT_KEYS

    missing_path = tmp_path / "Georgia.ttf"
    mutated = _mutated_config(
        tmp_path,
        'path = "/System/Library/Fonts/Supplemental/Georgia.ttf"',
        f'path = "{missing_path}"',
    )
    broken = hfb.load_config(mutated)
    with pytest.raises(hfb.HeldoutFontError, match="missing"):
        hfb.resolve_heldout_fonts(broken)


# ---------------------------------------------------------------------------
# Dataset accounting, grouping and target/reference integrity
# ---------------------------------------------------------------------------


def test_dataset_counts_and_source_grouping(
    benchmark_bundle: tuple[hfb.DatasetResult, hfb.EvaluationResult, Path],
) -> None:
    dataset, _, _ = benchmark_bundle
    manifest = dataset.manifest
    sources = dataset.sources
    assert len(sources) == 60
    assert sources["source_id"].is_unique
    assert sources["split"].eq("eval").all()
    assert set(sources["font_id"]) == {"georgia", "monaco", "impact"}
    assert sources.groupby("font_id")["source_id"].nunique().to_dict() == {
        "georgia": 20,
        "monaco": 20,
        "impact": 20,
    }
    assert len(manifest) == 1260
    assert manifest["sample_id"].is_unique
    assert manifest["split"].eq("eval").all()
    assert manifest.groupby("source_id")["split"].nunique().max() == 1
    assert manifest.groupby("source_id").size().eq(21).all()
    assert manifest.loc[manifest["variant_family"] == "clean", "image_sha256"].is_unique
    assert manifest["image_sha256"].nunique() == 1260
    assert dataset.summary["counts"]["duplicate_image_fraction"] == 0.0
    assert dataset.summary["counts"]["sources"] == 60
    assert dataset.summary["counts"]["rows"] == 1260
    assert dataset.summary["counts"]["distinct_images"] == 1260
    assert set(manifest["variant_family"]) == {
        "clean",
        "gaussian_blur",
        "gaussian_noise",
        "thin_lines",
        "arc_strokes",
        "elastic",
    }
    assert manifest["protocol_scope"].value_counts().to_dict() == {
        "primary": 1140,
        "diagnostic": 120,
    }


def test_clean_reference_and_target_accounting(
    benchmark_bundle: tuple[hfb.DatasetResult, hfb.EvaluationResult, Path],
) -> None:
    dataset, evaluation, _ = benchmark_bundle
    manifest = dataset.manifest
    sources = dataset.sources.set_index("source_id")
    clean = manifest[manifest["variant_family"] == "clean"]
    assert len(clean) == 60
    assert clean["ssim_target"].eq(1.0).all()
    for row in manifest.itertuples(index=False):
        source = sources.loc[row.source_id]
        assert row.reference_path == source.clean_image_path
        assert row.reference_sha256 == source.clean_image_sha256
    assert np.isfinite(manifest["ssim_target"]).all()
    assert manifest["ssim_target"].between(0.0, 1.0).all()
    assert evaluation.isolation_checks["passed"].all()
    check_names = set(evaluation.isolation_checks["check"])
    assert "content:ssim_recomputed" in check_names
    assert "content:features_recomputed" in check_names
    assert "isolation:frozen_assets_unchanged" in check_names
    assert "isolation:source_ids_disjoint_from_m3" in check_names


def test_features_and_targets_are_exact(
    benchmark_bundle: tuple[hfb.DatasetResult, hfb.EvaluationResult, Path],
) -> None:
    dataset, evaluation, _ = benchmark_bundle
    assert list(dataset.features.columns) == ["sample_id", *FEATURE_NAMES]
    assert len(dataset.features) == len(dataset.manifest)
    assert set(dataset.features["sample_id"]) == set(dataset.manifest["sample_id"])
    assert np.isfinite(dataset.features[list(FEATURE_NAMES)].to_numpy()).all()
    assert np.isfinite(evaluation.predictions).all()
    assert len(evaluation.predictions) == len(evaluation.frame) == 1260


# ---------------------------------------------------------------------------
# Metric calculations, degeneracy and grouped uncertainty
# ---------------------------------------------------------------------------


def test_metric_calculations_and_r2_suppression() -> None:
    frame = pd.DataFrame(
        {
            "ssim_target": [1.0, 0.8, 0.6, 0.4],
            "source_id": ["s1", "s1", "s2", "s2"],
            "image_sha256": ["a", "b", "c", "d"],
        }
    )
    predicted = np.array([1.0, 0.9, 0.5, 0.5])
    values = hfb._metric_values(frame, predicted, near_degenerate_sd=0.005)
    assert values["n_rows"] == 4
    assert values["n_sources"] == 2
    assert values["mae"] == pytest.approx(0.075)
    assert values["rmse"] == pytest.approx(np.sqrt(0.0075))
    assert values["r2"] == pytest.approx(0.85)
    assert values["r2_defined"] is True
    assert values["near_degenerate"] is False

    constant = frame.assign(ssim_target=[0.9, 0.9, 0.9, 0.9])
    suppressed = hfb._metric_values(constant, predicted, near_degenerate_sd=0.005)
    assert suppressed["near_degenerate"] is True
    assert suppressed["r2_defined"] is False
    assert np.isnan(suppressed["r2"])


def test_grouped_bootstrap_uses_sources(
    benchmark_bundle: tuple[hfb.DatasetResult, hfb.EvaluationResult, Path],
) -> None:
    _, evaluation, _ = benchmark_bundle
    subset = evaluation.frame[evaluation.frame["protocol_scope"] == "primary"]
    predictions = evaluation.predictions[subset.index.to_numpy()]
    table = bootstrap_metrics(
        subset,
        predictions,
        n_bootstrap=50,
        seed=1,
        scope="test_primary",
    )
    assert set(table["metric"]) == {"mae", "rmse", "r2"}
    assert table["n_groups"].eq(60).all()
    assert table["n_samples"].eq(1140).all()
    assert table["n_bootstrap"].eq(50).all()
    assert table.loc[table["metric"] == "mae", "ci_low"].notna().all()


def test_evaluation_reports_primary_and_diagnostic_slices(
    benchmark_bundle: tuple[hfb.DatasetResult, hfb.EvaluationResult, Path],
) -> None:
    _, evaluation, _ = benchmark_bundle
    metrics = evaluation.metrics
    all_primary = metrics[(metrics["scope"] == "all_primary") & (metrics["font_id"] == "all")].iloc[
        0
    ]
    assert all_primary["n_rows"] == 1140
    assert all_primary["n_sources"] == 60
    assert all_primary["target_sd"] > 0.005
    assert np.isfinite(all_primary["mae"])
    assert np.isfinite(all_primary["rmse"])
    assert np.isfinite(all_primary["r2"])

    clean = metrics[(metrics["scope"] == "clean_reference") & (metrics["font_id"] == "all")].iloc[0]
    assert clean["target_sd"] == 0.0
    assert not bool(clean["r2_defined"])
    assert np.isnan(clean["r2"])

    diagnostic = metrics[
        (metrics["scope"] == "diagnostic_0_5px") & (metrics["font_id"] == "all")
    ].iloc[0]
    assert bool(diagnostic["near_degenerate"])
    assert not bool(diagnostic["r2_defined"])
    assert np.isnan(diagnostic["r2"])

    assert set(evaluation.font_metrics["font_id"]) == {"georgia", "monaco", "impact"}
    assert evaluation.uncertainty["n_groups"].isin([20, 60]).all()
    assert evaluation.comparison["caveat"].str.contains("Not a matched comparison").all()
    assert set(evaluation.comparison["m7b_scope"]).issuperset(
        {"original_blur_noise", "geometric_primary", "clean_reference"}
    )


# ---------------------------------------------------------------------------
# Reproducibility, report consistency and tamper detection
# ---------------------------------------------------------------------------


def test_generation_is_byte_reproducible(tmp_path: Path, config: hfb.HeldoutConfig) -> None:
    first = hfb.generate_dataset(config, output_dataset=tmp_path / "first", overwrite=True)
    second = hfb.generate_dataset(config, output_dataset=tmp_path / "second", overwrite=True)
    first_files = {
        path.relative_to(first.root).as_posix(): sha256_file(path)
        for path in sorted(first.root.rglob("*"))
        if path.is_file()
    }
    second_files = {
        path.relative_to(second.root).as_posix(): sha256_file(path)
        for path in sorted(second.root.rglob("*"))
        if path.is_file()
    }
    assert first_files == second_files


def test_report_and_outputs_are_consistent(
    benchmark_bundle: tuple[hfb.DatasetResult, hfb.EvaluationResult, Path],
    config: hfb.HeldoutConfig,
) -> None:
    dataset, evaluation, reports = benchmark_bundle
    report = (reports / "report.md").read_text(encoding="utf-8")
    assert "# M7B: Held-Out Font Generalisation Benchmark" in report
    assert "## Held-out fonts" in report
    assert "## Source-grouped uncertainty" in report
    assert "Not a matched comparison" in report
    assert "does not measure human readability" in report
    assert "No declared font was missing" in report
    assert evaluation.dataset.root == dataset.root
    metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
    for name, expected_hash in metadata["outputs"].items():
        assert sha256_file(reports / name) == expected_hash
    for name, expected_hash in metadata["dataset"]["file_sha256"].items():
        assert sha256_file(dataset.root / name) == expected_hash


def test_verify_detects_report_tampering(
    benchmark_bundle: tuple[hfb.DatasetResult, hfb.EvaluationResult, Path],
    config: hfb.HeldoutConfig,
    tmp_path: Path,
) -> None:
    dataset, _, reports = benchmark_bundle
    copied = tmp_path / "tampered_reports"
    copied.mkdir()
    for path in reports.iterdir():
        if path.is_file():
            (copied / path.name).write_bytes(path.read_bytes())
    table = copied / "overall_metrics.csv"
    frame = pd.read_csv(table)
    frame.loc[0, "mae"] = float(frame.loc[0, "mae"]) + 1.0
    frame.to_csv(table, index=False)
    report = hfb.verify_benchmark(config, dataset_root=dataset.root, reports_root=copied)
    assert report.ok is False
    assert any(
        check.name == "table:overall_metrics.csv" and not check.passed for check in report.checks
    )


def test_cli_verify_passes(
    benchmark_bundle: tuple[hfb.DatasetResult, hfb.EvaluationResult, Path],
    config: hfb.HeldoutConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset, _, reports = benchmark_bundle
    code = hfb.main(
        [
            "verify",
            "--config",
            str(CONFIG_PATH),
            "--dataset",
            str(dataset.root),
            "--reports",
            str(reports),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "verification: PASS" in captured.out
