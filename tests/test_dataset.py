"""Dataset generation, manifest integrity, verification and reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from captcha_quality.config import load_config
from captcha_quality.dataset import (
    CLEAN_CORRUPTION_TYPE,
    MANIFEST_COLUMNS,
    SOURCE_COLUMNS,
    DatasetError,
    DatasetExistsError,
    compare_datasets,
    generate_dataset,
    load_dataset,
    plan_sources,
    verify_dataset,
    write_feature_table,
)
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.render import resolve_fonts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_writes_all_artifacts(tiny_dataset) -> None:
    _, result = tiny_dataset
    root = result.root
    for name in (
        "manifest.csv",
        "sources.csv",
        "ground_truth.csv",
        "features.csv",
        "dataset_summary.json",
    ):
        assert (root / name).is_file(), name
    assert (root / "images" / "clean").is_dir()
    assert not list(root.parent.glob("*.partial"))


def test_counts_match_configuration(tiny_dataset) -> None:
    config, result = tiny_dataset
    assert result.summary["counts"]["sources"] == 3
    assert result.summary["counts"]["samples"] == 15
    assert result.summary["counts"]["clean_samples"] == 3
    assert result.summary["counts"]["feature_rows"] == 15
    assert config.n_samples == 15


def test_plan_sources_is_deterministic_across_calls(dataset_config_writer, tmp_path: Path) -> None:
    config = load_config(dataset_config_writer(output_dir=tmp_path / "x", sources_per_font=3))
    resolution = resolve_fonts(config.fonts.keys, min_required=1)
    first = plan_sources(config, resolution)
    second = plan_sources(config, resolution)
    assert [plan.source_id for plan in first] == [plan.source_id for plan in second]
    assert [plan.split for plan in first] == [plan.split for plan in second]
    assert {plan.split for plan in first} == {"train", "val", "test"}


def test_manifest_schema_and_no_raw_text(tiny_dataset) -> None:
    _, result = tiny_dataset
    manifest = result.manifest
    assert tuple(manifest.columns) == MANIFEST_COLUMNS
    assert "text" not in manifest.columns
    assert manifest["sample_id"].is_unique
    assert manifest["generation_config_hash"].nunique() == 1


def test_ground_truth_is_separate_and_consistent(tiny_dataset) -> None:
    _, result = tiny_dataset
    ground_truth = pd.read_csv(result.root / "ground_truth.csv")
    assert list(ground_truth.columns) == ["source_id", "text"]
    assert len(ground_truth) == 3
    merged = result.manifest.merge(ground_truth, on="source_id", how="left")
    for row in merged.drop_duplicates("source_id").itertuples(index=False):
        assert hashlib.sha256(str(row.text).encode()).hexdigest() == row.text_sha256


def test_paths_reference_and_targets_verify(tiny_dataset) -> None:
    _, result = tiny_dataset
    report = verify_dataset(result.root)
    assert report.ok, [check for check in report.failures()]
    assert set(result.manifest["corruption_type"]) == {
        CLEAN_CORRUPTION_TYPE,
        "gaussian_blur",
        "gaussian_noise",
    }


def test_clean_rows_are_identity_pairs(tiny_dataset) -> None:
    _, result = tiny_dataset
    clean = result.manifest[result.manifest["corruption_type"] == CLEAN_CORRUPTION_TYPE]
    assert len(clean) == 3
    assert (clean["ssim_target"] == 1.0).all()
    assert clean["psnr_target"].isna().all()
    assert clean["psnr_infinite"].all()
    assert (clean["image_path"] == clean["reference_path"]).all()
    assert clean["corruption_parameters"].eq("{}").all()


def test_corrupted_rows_record_parameters_and_seeds(tiny_dataset) -> None:
    _, result = tiny_dataset
    corrupted = result.manifest[result.manifest["corruption_type"] != CLEAN_CORRUPTION_TYPE]
    assert (corrupted["psnr_target"] > 0).all()
    assert not corrupted["psnr_infinite"].any()
    parameters = corrupted["corruption_parameters"]
    assert parameters.str.contains("radius_px").any()
    assert parameters.str.contains("sigma").any()
    assert corrupted["variant_seed"].nunique() == len(corrupted)
    assert (corrupted["severity_level"] >= 1).all()


def test_split_isolation_and_stratification(tiny_dataset) -> None:
    _, result = tiny_dataset
    sources = result.sources
    assert sources["split_group_id"].equals(sources["source_id"])
    assert set(sources["split"]) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert sources.loc[sources["split"] == split, "font_id"].nunique() == 1
    per_source = result.manifest.groupby("source_id")["split"].nunique()
    assert (per_source == 1).all()


def test_repeated_generation_is_byte_identical(dataset_config_writer, tmp_path: Path) -> None:
    config = load_config(dataset_config_writer(output_dir=tmp_path / "unused"))
    first = generate_dataset(config, output_dir=tmp_path / "run_a")
    second = generate_dataset(config, output_dir=tmp_path / "run_b")
    assert _sha256(first.root / "manifest.csv") == _sha256(second.root / "manifest.csv")
    assert _sha256(first.root / "features.csv") == _sha256(second.root / "features.csv")
    assert _sha256(first.root / "sources.csv") == _sha256(second.root / "sources.csv")
    report = compare_datasets(first.root, second.root)
    assert report.ok, [check for check in report.failures()]


def test_compare_detects_a_feature_difference(tiny_dataset, tmp_path: Path) -> None:
    config, result = tiny_dataset
    other = generate_dataset(config, output_dir=tmp_path / "other")
    features = pd.read_csv(other.root / "features.csv")
    features.loc[0, FEATURE_NAMES[0]] = features.loc[0, FEATURE_NAMES[0]] + 1.0
    features.to_csv(other.root / "features.csv", index=False, lineterminator="\n")
    report = compare_datasets(result.root, other.root)
    assert not report.ok
    assert any(check.name == "features.csv" for check in report.failures())


def test_existing_output_requires_overwrite(dataset_config_writer, tmp_path: Path) -> None:
    config = load_config(dataset_config_writer(output_dir=tmp_path / "d"))
    generate_dataset(config)
    with pytest.raises(DatasetExistsError, match="already exists"):
        generate_dataset(config)
    replaced = generate_dataset(config, overwrite=True)
    assert replaced.summary["counts"]["samples"] == 15


def test_failed_generation_leaves_no_partial(
    dataset_config_writer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from captcha_quality import dataset as dataset_module
    from captcha_quality.render import render_clean as real_render_clean

    config = load_config(dataset_config_writer(output_dir=tmp_path / "d"))
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("boom")
        return real_render_clean(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "render_clean", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        generate_dataset(config)
    assert not (tmp_path / "d").exists()
    assert not (tmp_path / "d.partial").exists()


def test_verify_detects_tampered_target(tiny_dataset) -> None:
    _, result = tiny_dataset
    manifest = pd.read_csv(result.root / "manifest.csv")
    index = manifest.index[manifest["corruption_type"] != CLEAN_CORRUPTION_TYPE][0]
    manifest.loc[index, "ssim_target"] = 0.123
    manifest.to_csv(result.root / "manifest.csv", index=False, lineterminator="\n")
    report = verify_dataset(result.root)
    assert not report.ok
    assert any(check.name == "targets_recomputed" for check in report.failures())


def test_verify_detects_missing_image(tiny_dataset) -> None:
    _, result = tiny_dataset
    manifest = pd.read_csv(result.root / "manifest.csv")
    target = result.root / str(manifest.loc[0, "image_path"])
    target.unlink()
    report = verify_dataset(result.root)
    assert not report.ok
    assert any(check.name == "manifest_paths_exist" for check in report.failures())


def test_verify_detects_split_crossing(tiny_dataset) -> None:
    _, result = tiny_dataset
    manifest = pd.read_csv(result.root / "manifest.csv")
    source = manifest.loc[0, "source_id"]
    rows = manifest["source_id"] == source
    current = str(manifest.loc[rows, "split"].iloc[0])
    other = next(split for split in ("train", "val", "test") if split != current)
    manifest.loc[rows, "split"] = other
    manifest.to_csv(result.root / "manifest.csv", index=False, lineterminator="\n")
    report = verify_dataset(result.root)
    assert not report.ok
    assert any(check.name == "split_isolation" for check in report.failures())


def test_verify_detects_text_in_manifest(tiny_dataset) -> None:
    _, result = tiny_dataset
    manifest = pd.read_csv(result.root / "manifest.csv")
    manifest["text"] = "LEAK"
    manifest.to_csv(result.root / "manifest.csv", index=False, lineterminator="\n")
    report = verify_dataset(result.root)
    assert not report.ok
    assert any(check.name == "manifest_schema" for check in report.failures())


def test_summary_is_timestamp_free_and_consistent(tiny_dataset) -> None:
    _, result = tiny_dataset
    summary = json.loads((result.root / "dataset_summary.json").read_text(encoding="utf-8"))
    assert not any("time" in key or "date" in key for key in summary)
    assert summary["feature_names"] == list(FEATURE_NAMES)
    assert summary["artifacts"]["manifest_sha256"] == _sha256(result.root / "manifest.csv")
    assert summary["text_policy"]["public_manifest_contains_text"] is False
    assert summary["counts"]["disk_usage_bytes"] > 0


def test_sources_table_schema(tiny_dataset) -> None:
    _, result = tiny_dataset
    assert tuple(result.sources.columns) == SOURCE_COLUMNS
    assert result.sources["source_id"].is_unique
    assert result.sources["background_level"].between(0, 255).all()


def test_write_feature_table_reproduces_bytes(tiny_dataset) -> None:
    _, result = tiny_dataset
    before = _sha256(result.root / "features.csv")
    features = write_feature_table(result.root)
    assert _sha256(result.root / "features.csv") == before
    assert len(features) == 15


def test_generate_without_features_then_rebuild(dataset_config_writer, tmp_path: Path) -> None:
    config = load_config(dataset_config_writer(output_dir=tmp_path / "d"))
    result = generate_dataset(config, extract_features_from_images=False)
    assert result.features.empty
    rebuilt = write_feature_table(result.root)
    assert len(rebuilt) == 15
    assert set(rebuilt["sample_id"]) == set(result.manifest["sample_id"])


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="missing required file"):
        load_dataset(tmp_path)


def test_text_lengths_come_from_the_configured_distribution(tiny_dataset) -> None:
    config, result = tiny_dataset
    assert set(result.sources["text_length"]).issubset(set(config.dataset.text_lengths))
    assert set(result.manifest["font_id"]).issubset(set(config.fonts.keys))
