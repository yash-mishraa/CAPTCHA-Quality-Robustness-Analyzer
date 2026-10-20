"""Tests for the M4A occlusion/interference pilot runner and analysis."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from captcha_quality.features import FEATURE_NAMES
from captcha_quality.robustness_pilot import (
    PILOT_FEATURE_COLUMNS,
    PILOT_MANIFEST_COLUMNS,
    PILOT_SOURCE_COLUMNS,
    PilotConfig,
    PilotConfigError,
    PilotError,
    PilotExistsError,
    PilotGateThresholds,
    PilotSettings,
    ReferenceSettings,
    _duplicate_fraction,
    evaluate_gate,
    feature_effect_sizes,
    feature_summary_table,
    feature_target_correlations,
    fingerprint_assets,
    grouped_bootstrap_levels,
    load_frozen_reference,
    run_pilot,
    saturation_table,
    summarise_levels,
    trend_by_source,
    trend_violations,
    verify_pilot_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pilot_config(
    path: Path,
    *,
    reference_root: Path,
    reference_config: Path,
    output_dataset: Path,
    output_reports: Path,
    counts: tuple[int, ...] = (1, 2),
    bootstrap: int = 20,
    seed: int = 101,
    manifest_sha: str | None = None,
) -> Path:
    summary = json.loads((reference_root / "dataset_summary.json").read_text(encoding="utf-8"))
    split_hash = summary["split_policy"]["assignment_sha256"]
    manifest_sha = manifest_sha or _sha256(reference_root / "manifest.csv")
    path.write_text(
        f"""
[pilot]
name = "test-occlusion-pilot"
version = "v0.0.0-test"
seed = {seed}
bootstrap_replicates = {bootstrap}
output_dataset = {str(output_dataset)!r}
output_reports = {str(output_reports)!r}

[reference]
dataset = {str(reference_root)!r}
config = {str(reference_config)!r}
dataset_version = {summary["dataset_version"]!r}
generation_config_hash = {summary["generation_config_hash"]!r}
manifest_sha256 = {manifest_sha!r}
sources_sha256 = {_sha256(reference_root / "sources.csv")!r}
features_sha256 = {_sha256(reference_root / "features.csv")!r}
ground_truth_sha256 = {_sha256(reference_root / "ground_truth.csv")!r}
summary_sha256 = {_sha256(reference_root / "dataset_summary.json")!r}
split_assignment_sha256 = {split_hash!r}

[interference]
variants = ["thin_lines", "arc_strokes"]
stroke_width_px = 1
stroke_intensity = 0

[interference.thin_lines]
stroke_counts = [{", ".join(str(value) for value in counts)}]
line_angle_deg_min = -45.0
line_angle_deg_max = 45.0

[interference.arc_strokes]
stroke_counts = [{", ".join(str(value) for value in counts)}]
arc_sweep_deg_min = 90.0
arc_sweep_deg_max = 180.0
arc_radius_fraction_min = 0.20
arc_radius_fraction_max = 0.50
arc_aspect_ratio_min = 0.50
arc_aspect_ratio_max = 1.20

[gate]
ssim_std_min = 0.01
level_mean_range_min = 0.01
per_level_sd_min = 0.001
median_source_spearman_max = -0.80
increasing_step_fraction_max = 0.20
feature_effect_min = 0.10
feature_strong_effect_min = 0.20
feature_strong_effect_count_min = 1
duplicate_fraction_max = 0.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def pilot_config(tmp_path: Path, tiny_dataset) -> PilotConfig:
    _, result = tiny_dataset
    path = _write_pilot_config(
        tmp_path / "pilot.toml",
        reference_root=result.root,
        reference_config=result.config.config_path,
        output_dataset=tmp_path / "pilot-dataset",
        output_reports=tmp_path / "pilot-reports",
    )
    return PilotConfig.from_toml(path)


def test_config_round_trip_records_generation_identity(pilot_config: PilotConfig) -> None:
    assert pilot_config.pilot.seed == 101
    assert [ladder.variant for ladder in pilot_config.ladders] == [
        "thin_lines",
        "arc_strokes",
    ]
    assert pilot_config.ladder("thin_lines").stroke_counts == (1, 2)
    assert len(pilot_config.pilot_config_hash) == 64
    assert len(pilot_config.config_sha256) == 64
    payload = pilot_config.provenance()
    assert payload["generation_config"]["seed"] == 101
    assert payload["gate"]["duplicate_fraction_max"] == 0.0


def test_config_rejects_unknown_keys(pilot_config: PilotConfig, tmp_path: Path) -> None:
    content = pilot_config.config_path.read_text(encoding="utf-8").replace(
        "[pilot]", "[pilot]\nunexpected = true"
    )
    path = tmp_path / "bad.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PilotConfigError, match="unknown key"):
        PilotConfig.from_toml(path)


def test_config_rejects_unknown_variant(pilot_config: PilotConfig, tmp_path: Path) -> None:
    content = pilot_config.config_path.read_text(encoding="utf-8").replace(
        'variants = ["thin_lines", "arc_strokes"]',
        'variants = ["thin_lines", "jpeg"]',
    )
    path = tmp_path / "bad.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PilotConfigError, match="unsupported interference variant"):
        PilotConfig.from_toml(path)


def test_config_rejects_non_increasing_counts(pilot_config: PilotConfig, tmp_path: Path) -> None:
    content = pilot_config.config_path.read_text(encoding="utf-8").replace(
        "stroke_counts = [1, 2]", "stroke_counts = [2, 2]"
    )
    path = tmp_path / "bad.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PilotConfigError, match="strictly increasing"):
        PilotConfig.from_toml(path)


def test_settings_and_thresholds_validate() -> None:
    with pytest.raises(PilotConfigError, match="seed"):
        PilotSettings("name", "v", -1, 1, Path("a"), Path("b"))
    with pytest.raises(PilotConfigError, match="seed"):
        PilotSettings("name", "v", 2**64, 1, Path("a"), Path("b"))
    with pytest.raises(PilotConfigError, match="bootstrap_replicates"):
        PilotSettings("name", "v", 1, 0, Path("a"), Path("b"))
    with pytest.raises(PilotConfigError, match="median_source_spearman_max"):
        PilotGateThresholds(0.01, 0.01, 0.001, 0.5, 0.1, 0.5, 0.8, 1, 0.0)
    with pytest.raises(PilotConfigError, match="feature_strong_effect_min"):
        PilotGateThresholds(0.01, 0.01, 0.001, -0.8, 0.1, 0.5, 0.2, 1, 0.0)


def test_reference_settings_require_hashes() -> None:
    with pytest.raises(PilotConfigError, match="lowercase SHA-256"):
        ReferenceSettings(
            Path("data"),
            Path("config.toml"),
            "v0.1.0",
            "not-a-hash",
            "0" * 64,
            "0" * 64,
            "0" * 64,
            "0" * 64,
            "0" * 64,
            "0" * 64,
        )


def test_load_frozen_reference_rejects_wrong_manifest_hash(
    pilot_config: PilotConfig, tmp_path: Path
) -> None:
    path = _write_pilot_config(
        tmp_path / "wrong.toml",
        reference_root=pilot_config.reference.dataset,
        reference_config=pilot_config.reference.config,
        output_dataset=tmp_path / "out",
        output_reports=tmp_path / "reports",
        manifest_sha="0" * 64,
    )
    wrong = PilotConfig.from_toml(path)
    with pytest.raises(PilotError, match="hash mismatch"):
        load_frozen_reference(wrong)


def test_fingerprint_assets_records_missing_paths(tmp_path: Path) -> None:
    present = tmp_path / "present.txt"
    present.write_text("hello", encoding="utf-8")
    result = fingerprint_assets([present, tmp_path / "missing.txt"])
    assert result[str(present)]["exists"] is True
    assert result[str(present)]["sha256"] == _sha256(present)
    assert result[str(tmp_path / "missing.txt")]["exists"] is False


def _synthetic_manifest() -> pd.DataFrame:
    rows = []
    for source_id, base in (("s1", 0.95), ("s2", 0.90)):
        for severity, (strokes, drop) in enumerate(((1, 0.02), (2, 0.10), (4, 0.25)), start=1):
            ssim_value = base - drop
            rows.append(
                {
                    "sample_id": f"{source_id}__thin_lines_s{severity}",
                    "source_id": source_id,
                    "variant_id": f"thin_lines_s{severity}",
                    "split": "train",
                    "font_id": "arial",
                    "text_length": 4,
                    "variant_family": "thin_lines",
                    "severity_level": severity,
                    "stroke_count": strokes,
                    "stroke_width_px": 1,
                    "stroke_intensity": 0,
                    "variant_seed": 1,
                    "geometry_json": "[]",
                    "image_path": f"images/{source_id}_{severity}.png",
                    "reference_path": f"images/{source_id}.png",
                    "ssim_target": ssim_value,
                    "psnr_target": 20.0 + ssim_value,
                    "psnr_infinite": False,
                    "height": 48,
                    "width": 128,
                    "image_sha256": f"{source_id}-{severity}",
                    "reference_sha256": f"{source_id}-clean",
                    "changed_pixel_fraction": 0.01 * strokes,
                    "changed_ink_fraction": 0.005 * strokes,
                    "changed_background_fraction": 0.02 * strokes,
                    "border_touched": True,
                    "pilot_version": "v0",
                    "pilot_config_hash": "0" * 64,
                }
            )
    return pd.DataFrame(rows, columns=PILOT_MANIFEST_COLUMNS)


def test_level_summary_and_trend_functions() -> None:
    manifest = _synthetic_manifest()
    levels = summarise_levels(manifest)
    assert {
        "variant_family",
        "severity_level",
        "stroke_count",
        "ssim_mean",
        "ssim_sd",
        "ssim_mean_step_delta",
    } <= set(levels.columns)
    assert len(levels) == 3
    assert (
        levels.loc[levels["severity_level"] == 2, "ssim_mean"].iloc[0]
        < levels.loc[levels["severity_level"] == 1, "ssim_mean"].iloc[0]
    )
    assert levels["ssim_mean_step_delta"].isna().sum() == 1

    trends = trend_by_source(manifest)
    assert (trends["spearman"] == -1.0).all()
    assert int(trends["n_increasing_steps"].sum()) == 0
    assert len(trend_violations(manifest)) == 0

    saturation = saturation_table(levels)
    assert len(saturation) == 2
    assert saturation["added_strokes"].tolist() == [1, 2]


def test_trend_violations_are_reported_not_dropped() -> None:
    manifest = _synthetic_manifest()
    mask = (manifest["source_id"] == "s1") & (manifest["severity_level"] == 2)
    manifest.loc[mask, "ssim_target"] = 0.999
    violations = trend_violations(manifest)
    assert len(violations) == 1
    assert violations["source_id"].iloc[0] == "s1"
    assert violations["increase"].iloc[0] > 0.0
    assert int(trend_by_source(manifest).loc[0, "n_increasing_steps"]) == 1


def test_duplicate_fraction_and_grouped_bootstrap() -> None:
    manifest = _synthetic_manifest()
    assert _duplicate_fraction(manifest) == 0.0
    duplicated = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    assert _duplicate_fraction(duplicated) > 0.0
    with pytest.raises(PilotError, match="n_bootstrap"):
        grouped_bootstrap_levels(manifest, n_bootstrap=0, seed=1)
    table = grouped_bootstrap_levels(manifest, n_bootstrap=20, seed=1)
    assert list(table.columns) == [
        "variant_family",
        "severity_level",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_groups",
        "n_samples",
        "n_bootstrap",
    ]
    assert len(table) == 3
    assert (table["ci_low"] <= table["estimate"]).all()
    assert (table["estimate"] <= table["ci_high"]).all()


def _feature_table(manifest: pd.DataFrame, base: float) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(manifest.itertuples(index=False)):
        values = {feature: base + index * 0.01 for feature in FEATURE_NAMES}
        rows.append({"sample_id": row.sample_id, **values})
    return pd.DataFrame(rows, columns=PILOT_FEATURE_COLUMNS)


def test_feature_analysis_functions_have_stable_schemas() -> None:
    pilot_manifest = _synthetic_manifest()
    pilot_features = _feature_table(pilot_manifest, 1.0)
    frozen_rows = []
    for source_id in ("f1", "f2"):
        for family in ("gaussian_blur", "gaussian_noise"):
            for severity in (1, 2, 3):
                frozen_rows.append(
                    {
                        "sample_id": f"{source_id}_{family}_{severity}",
                        "source_id": source_id,
                        "variant_id": f"{family}_s{severity}",
                        "split": "train",
                        "font_id": "arial",
                        "text_length": 4,
                        "corruption_type": family,
                        "severity_level": severity,
                        "ssim_target": 0.95 - 0.1 * severity,
                    }
                )
    frozen_manifest = pd.DataFrame(frozen_rows)
    frozen_features = _feature_table(frozen_manifest, 10.0)

    summary = feature_summary_table(
        pilot_manifest, pilot_features, frozen_manifest, frozen_features
    )
    assert set(summary.columns) == {
        "family_label",
        "severity_level",
        "feature",
        "n_samples",
        "mean",
        "sd",
        "min",
        "max",
    }
    effects_long, effects = feature_effect_sizes(
        pilot_manifest, pilot_features, frozen_manifest, frozen_features
    )
    assert len(effects_long) == 3 * 2 * 8
    assert set(effects.columns) == {
        "variant_family",
        "severity_level",
        "reference_family",
        "mean_abs_cohens_d",
        "max_abs_cohens_d",
        "n_features_abs_d_ge_0_8",
        "n_features_abs_d_ge_1_0",
        "n_features_abs_d_ge_strong",
        "n_features",
    }
    correlations = feature_target_correlations(
        pilot_manifest, pilot_features, frozen_manifest, frozen_features
    )
    assert len(correlations) == 3 * 8
    assert correlations["pooled_pearson"].notna().all()


def _gate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = _synthetic_manifest()
    levels = summarise_levels(manifest)
    trends = trend_by_source(manifest)
    effects = pd.DataFrame(
        [
            {
                "variant_family": "thin_lines",
                "severity_level": 3,
                "reference_family": "gaussian_blur",
                "mean_abs_cohens_d": 1.5,
                "max_abs_cohens_d": 2.0,
                "n_features_abs_d_ge_0_8": 4,
                "n_features_abs_d_ge_1_0": 3,
                "n_features_abs_d_ge_strong": 4,
                "n_features": 8,
            },
            {
                "variant_family": "thin_lines",
                "severity_level": 3,
                "reference_family": "gaussian_noise",
                "mean_abs_cohens_d": 1.2,
                "max_abs_cohens_d": 1.8,
                "n_features_abs_d_ge_0_8": 3,
                "n_features_abs_d_ge_1_0": 2,
                "n_features_abs_d_ge_strong": 3,
                "n_features": 8,
            },
        ]
    )
    return manifest, levels, trends, effects


def _gate_config(pilot_config: PilotConfig) -> PilotConfig:
    return PilotConfig(
        pilot=pilot_config.pilot,
        reference=pilot_config.reference,
        ladders=pilot_config.ladders,
        gate=PilotGateThresholds(0.01, 0.01, 0.001, -0.8, 0.2, 0.5, 0.8, 2, 0.0),
        config_path=pilot_config.config_path,
    )


def test_evaluate_gate_pass_and_fail(pilot_config: PilotConfig) -> None:
    manifest, levels, trends, effects = _gate_inputs()
    config = _gate_config(pilot_config)
    passed = evaluate_gate(
        config,
        manifest=manifest,
        levels=levels,
        trends=trends,
        effects_summary=effects,
        duplicate_fraction=0.0,
        reproducibility_ok=True,
        verification_ok=True,
        clean_reference_exact=True,
    )
    assert passed.verdict == "PASS"
    failed = evaluate_gate(
        config,
        manifest=manifest,
        levels=levels,
        trends=trends,
        effects_summary=effects.head(1),
        duplicate_fraction=0.5,
        reproducibility_ok=False,
        verification_ok=False,
        clean_reference_exact=False,
    )
    assert failed.verdict == "FAIL"
    assert "reproducibility" in failed.failed_criteria
    assert "feature_distinctness" in failed.failed_criteria


def test_run_pilot_end_to_end_is_reproducible_and_verifiable(
    pilot_config: PilotConfig, tmp_path: Path
) -> None:
    dataset_before = _sha256(pilot_config.reference.dataset / "manifest.csv")
    result = run_pilot(pilot_config, overwrite=False, verify_reproduction=True)
    assert _sha256(pilot_config.reference.dataset / "manifest.csv") == dataset_before
    assert result.verification.ok
    assert result.summary["determinism"]["ok"] is True
    assert result.summary["determinism"]["repeated_build"] is True
    assert result.summary["counts"]["sources"] == 3
    assert result.summary["counts"]["samples"] == 15
    assert result.summary["counts"]["corrupted_samples"] == 12
    assert result.summary["counts"]["clean_samples"] == 3
    assert result.report_path.is_file()
    assert result.gate.verdict in {"PASS", "CONDITIONAL", "FAIL"}

    manifest = pd.read_csv(result.output_dataset / "manifest.csv")
    sources = pd.read_csv(result.output_dataset / "sources.csv")
    features = pd.read_csv(result.output_dataset / "features.csv")
    assert tuple(manifest.columns) == PILOT_MANIFEST_COLUMNS
    assert tuple(sources.columns) == PILOT_SOURCE_COLUMNS
    assert tuple(features.columns) == PILOT_FEATURE_COLUMNS
    assert len(manifest) == 15
    assert len(sources) == 3
    assert len(features) == 15
    assert manifest["sample_id"].is_unique
    assert manifest["image_sha256"].nunique() == 15
    assert (manifest["ssim_target"] <= 1.0).all()
    assert (manifest["ssim_target"] >= 0.0).all()
    assert np.isfinite(features[list(FEATURE_NAMES)].to_numpy()).all()
    assert (result.output_reports / "pilot.md").is_file()
    assert (result.output_reports / "gate_results.json").is_file()
    assert (result.output_reports / "examples_thin_lines.png").is_file()
    assert (result.output_reports / "examples_arc_strokes.png").is_file()

    verified = verify_pilot_dataset(
        result.output_dataset, reference=result.reference, recompute_targets=True
    )
    assert verified.ok

    with pytest.raises(PilotExistsError):
        run_pilot(pilot_config, overwrite=False, verify_reproduction=False)

    unsafe_config = _write_pilot_config(
        tmp_path / "unsafe.toml",
        reference_root=pilot_config.reference.dataset,
        reference_config=pilot_config.reference.config,
        output_dataset=pilot_config.reference.dataset,
        output_reports=tmp_path / "unsafe-reports",
    )
    with pytest.raises(PilotError, match="frozen reference dataset"):
        run_pilot(
            PilotConfig.from_toml(unsafe_config),
            overwrite=True,
            verify_reproduction=False,
        )
    nested_config = _write_pilot_config(
        tmp_path / "nested.toml",
        reference_root=pilot_config.reference.dataset,
        reference_config=pilot_config.reference.config,
        output_dataset=pilot_config.reference.dataset / "nested-pilot",
        output_reports=tmp_path / "nested-reports",
    )
    with pytest.raises(PilotError, match="frozen reference dataset"):
        run_pilot(
            PilotConfig.from_toml(nested_config),
            overwrite=False,
            verify_reproduction=False,
        )


def test_verify_detects_tampered_target(pilot_config: PilotConfig, tmp_path: Path) -> None:
    result = run_pilot(pilot_config, verify_reproduction=False)
    copied = tmp_path / "tampered"
    shutil.copytree(result.output_dataset, copied)
    manifest_path = copied / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest.loc[manifest["variant_family"] != "none", "ssim_target"] += 0.05
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    report = verify_pilot_dataset(copied, reference=result.reference, recompute_targets=True)
    assert not report.ok
    assert "targets_recomputed" in {check.name for check in report.failures()}


def test_verify_detects_clean_reference_mutation(pilot_config: PilotConfig, tmp_path: Path) -> None:
    result = run_pilot(pilot_config, verify_reproduction=False)
    copied = tmp_path / "mutated"
    shutil.copytree(result.output_dataset, copied)
    source = pd.read_csv(copied / "sources.csv").iloc[0]
    image_path = copied / source["pilot_clean_image_path"]
    image = np.array(Image.open(image_path))
    image[0, 0] = 0 if image[0, 0] != 0 else 255
    Image.fromarray(image).save(image_path)
    report = verify_pilot_dataset(copied, reference=result.reference, recompute_targets=False)
    assert not report.ok
    assert "clean_reference_immutability" in {check.name for check in report.failures()}
