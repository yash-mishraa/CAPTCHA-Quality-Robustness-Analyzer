"""Tests for the M5A geometric-distortion pilot runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from captcha_quality.features import FEATURE_NAMES
from captcha_quality.geometric_pilot import (
    PILOT_FEATURE_COLUMNS,
    PILOT_MANIFEST_COLUMNS,
    PILOT_SOURCE_COLUMNS,
    GeometricPilotConfig,
    GeometricPilotConfigError,
    GeometricPilotError,
    verify_pilot_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pilot_config(
    path: Path,
    *,
    dataset: Path,
    dataset_config: Path,
    dataset_version: str,
    generation_hash: str,
    manifest_hash: str,
    sources_hash: str,
    features_hash: str,
    ground_truth_hash: str,
    summary_hash: str,
    split_hash: str,
    output_dataset: Path,
    output_reports: Path,
) -> Path:
    path.write_text(
        f"""
[pilot]
name = "geometric-distortion-pilot-test"
version = "v-test"
seed = 7
bootstrap_replicates = 5
output_dataset = "{output_dataset.as_posix()}"
output_reports = "{output_reports.as_posix()}"

[reference]
dataset = "{dataset.as_posix()}"
config = "{dataset_config.as_posix()}"
dataset_version = "{dataset_version}"
generation_config_hash = "{generation_hash}"
manifest_sha256 = "{manifest_hash}"
sources_sha256 = "{sources_hash}"
features_sha256 = "{features_hash}"
ground_truth_sha256 = "{ground_truth_hash}"
summary_sha256 = "{summary_hash}"
split_assignment_sha256 = "{split_hash}"

[comparison]
include_interference = false

[geometry]
amplitudes_px = [0.5, 1.0]
minimal_amplitude_px = 0.25
grid_rows = 5
grid_cols = 13
smoothing_sigma_px = 2.0
interpolation = "bilinear"
border_mode = "edge"

[analysis]
near_identity_ssim_min = 0.9999
min_group_samples = 1

[gate]
ssim_std_min = 0.0001
level_mean_range_min = 0.0001
per_level_sd_min = 0.0001
median_source_spearman_max = -0.1
increasing_step_fraction_max = 1.0
minimal_control_ssim_min = 0.5
border_affected_fraction_max = 1.0
collapsed_block_fraction_max = 1.0
feature_effect_min = 0.01
feature_strong_effect_min = 0.01
feature_strong_effect_count_min = 1
duplicate_fraction_max = 1.0
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def pilot_setup(tmp_path: Path, tiny_dataset) -> tuple[GeometricPilotConfig, Path, Path]:
    dataset_config, dataset_result = tiny_dataset
    summary = dataset_result.summary
    artifacts = summary["artifacts"]
    config_path = _write_pilot_config(
        tmp_path / "geometric.toml",
        dataset=dataset_result.root,
        dataset_config=dataset_config.config_path,
        dataset_version=str(summary["dataset_version"]),
        generation_hash=str(summary["generation_config_hash"]),
        manifest_hash=str(artifacts["manifest_sha256"]),
        sources_hash=str(artifacts["sources_sha256"]),
        features_hash=str(artifacts["features_sha256"]),
        ground_truth_hash=str(artifacts["ground_truth_sha256"]),
        summary_hash=_sha256(dataset_result.root / "dataset_summary.json"),
        split_hash=str(summary["split_policy"]["assignment_sha256"]),
        output_dataset=tmp_path / "pilot",
        output_reports=tmp_path / "reports",
    )
    return GeometricPilotConfig.from_toml(config_path), dataset_result.root, tmp_path


def test_end_to_end_pilot_is_verified_and_preserves_frozen_clean_sources(
    pilot_setup: tuple[GeometricPilotConfig, Path, Path],
) -> None:
    from captcha_quality.geometric_pilot import run_pilot

    config, frozen_root, tmp_path = pilot_setup
    clean_hashes_before = {
        row.source_id: row.clean_image_sha256
        for row in pd.read_csv(frozen_root / "sources.csv").itertuples(index=False)
    }
    result = run_pilot(
        config,
        output_dataset=tmp_path / "pilot",
        output_reports=tmp_path / "reports",
        overwrite=True,
    )

    assert result.verification.ok
    assert len(result.manifest) == 3 * (1 + 3 + 2)
    assert tuple(result.manifest.columns) == PILOT_MANIFEST_COLUMNS
    assert tuple(result.sources.columns) == PILOT_SOURCE_COLUMNS
    assert tuple(result.features.columns) == PILOT_FEATURE_COLUMNS
    assert tuple(result.features.columns[1:]) == FEATURE_NAMES
    assert result.summary["frozen_assets"]["all_unchanged"] is True
    assert result.summary["counts"]["sources"] == 3
    assert result.summary["counts"]["elastic_samples"] == 6
    assert result.summary["counts"]["control_samples"] == 9
    assert (tmp_path / "reports" / "pilot.md").is_file()
    assert (tmp_path / "reports" / "examples_elastic.png").is_file()

    clean_rows = result.manifest[result.manifest["variant_family"] == "clean"]
    assert len(clean_rows) == 3
    assert (clean_rows["image_sha256"] == clean_rows["reference_sha256"]).all()
    identity_rows = result.manifest[result.manifest["control_id"] == "identity"]
    zero_rows = result.manifest[result.manifest["control_id"] == "zero_field"]
    minimal_rows = result.manifest[result.manifest["control_id"] == "minimal"]
    assert (identity_rows["image_sha256"] == identity_rows["reference_sha256"]).all()
    assert (zero_rows["image_sha256"] == zero_rows["reference_sha256"]).all()
    assert not (minimal_rows["image_sha256"] == minimal_rows["reference_sha256"]).any()

    root = tmp_path / "pilot"
    report = verify_pilot_dataset(root, recompute_targets=True)
    assert report.ok

    clean_hashes_after = {
        row.source_id: row.clean_image_sha256
        for row in pd.read_csv(frozen_root / "sources.csv").itertuples(index=False)
    }
    assert clean_hashes_after == clean_hashes_before


def test_two_runs_have_identical_tables_and_image_hashes(
    pilot_setup: tuple[GeometricPilotConfig, Path, Path],
) -> None:
    from captcha_quality.geometric_pilot import run_pilot

    config, _frozen_root, tmp_path = pilot_setup
    first = run_pilot(
        config,
        output_dataset=tmp_path / "pilot",
        output_reports=tmp_path / "reports",
        overwrite=True,
    )
    first_manifest = first.manifest.copy()
    first_features = first.features.copy()
    first_hashes = first.manifest.set_index("sample_id")["image_sha256"].to_dict()

    second = run_pilot(
        config,
        output_dataset=tmp_path / "pilot",
        output_reports=tmp_path / "reports",
        overwrite=True,
    )
    pd.testing.assert_frame_equal(first_manifest, second.manifest)
    pd.testing.assert_frame_equal(first_features, second.features)
    second_hashes = second.manifest.set_index("sample_id")["image_sha256"].to_dict()
    assert first_hashes == second_hashes
    assert second.summary["determinism"]["ok"] is True


def test_tampered_image_fails_verification(
    pilot_setup: tuple[GeometricPilotConfig, Path, Path],
) -> None:
    from captcha_quality.geometric_pilot import run_pilot

    config, _frozen_root, tmp_path = pilot_setup
    result = run_pilot(
        config,
        output_dataset=tmp_path / "pilot",
        output_reports=tmp_path / "reports",
        overwrite=True,
    )
    elastic_row = result.manifest[result.manifest["variant_family"] == "elastic"].iloc[0]
    path = tmp_path / "pilot" / str(elastic_row["image_path"])
    image = np.array(Image.open(path))
    Image.fromarray(np.zeros_like(image)).save(path, format="PNG")

    report = verify_pilot_dataset(tmp_path / "pilot", recompute_targets=False)
    assert not report.ok
    assert "image_content_hashes" in {failure.name for failure in report.failures()}


def test_tampered_feature_table_fails_verification(
    pilot_setup: tuple[GeometricPilotConfig, Path, Path],
) -> None:
    from captcha_quality.geometric_pilot import run_pilot

    config, _frozen_root, tmp_path = pilot_setup
    run_pilot(
        config,
        output_dataset=tmp_path / "pilot",
        output_reports=tmp_path / "reports",
        overwrite=True,
    )
    features_path = tmp_path / "pilot" / "features.csv"
    features = pd.read_csv(features_path)
    features.loc[0, "laplacian_variance"] += 1.0
    features.to_csv(
        features_path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )

    report = verify_pilot_dataset(tmp_path / "pilot", recompute_targets=False)
    assert not report.ok
    assert "features_recomputed" in {failure.name for failure in report.failures()}


def test_config_is_strict_and_rejects_invalid_geometry(pilot_setup) -> None:
    config, _frozen_root, tmp_path = pilot_setup
    text = config.config_path.read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(text.replace("[geometry]\n", "[geometry]\nunknown = 1\n"), encoding="utf-8")
    with pytest.raises(GeometricPilotConfigError):
        GeometricPilotConfig.from_toml(unknown)

    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        text.replace("amplitudes_px = [0.5, 1.0]", "amplitudes_px = [0.5, 0.25]"),
        encoding="utf-8",
    )
    with pytest.raises(GeometricPilotConfigError):
        GeometricPilotConfig.from_toml(invalid)

    invalid_border = tmp_path / "invalid-border.toml"
    invalid_border.write_text(
        text.replace('border_mode = "edge"', 'border_mode = "wrap"'),
        encoding="utf-8",
    )
    with pytest.raises(GeometricPilotConfigError):
        GeometricPilotConfig.from_toml(invalid_border)


def test_provenance_hash_is_stable_and_generation_config_is_content_only(
    pilot_setup,
) -> None:
    config, _frozen_root, tmp_path = pilot_setup
    first = GeometricPilotConfig.from_toml(config.config_path)
    second = GeometricPilotConfig.from_toml(config.config_path)
    assert first.pilot_config_hash == second.pilot_config_hash
    assert first.config_sha256 == second.config_sha256
    assert "output_dataset" not in first.generation_config
    assert "gate" not in first.generation_config
    assert first.provenance()["geometry"]["amplitudes_px"] == [0.5, 1.0]

    moved = tmp_path / "moved.toml"
    moved.write_text(
        config.config_path.read_text(encoding="utf-8")
        .replace(
            'output_dataset = "' + str(tmp_path / "pilot") + '"',
            'output_dataset = "' + str(tmp_path / "other") + '"',
        )
        .replace(
            'output_reports = "' + str(tmp_path / "reports") + '"',
            'output_reports = "' + str(tmp_path / "other-reports") + '"',
        ),
        encoding="utf-8",
    )
    moved_config = GeometricPilotConfig.from_toml(moved)
    assert moved_config.pilot_config_hash == first.pilot_config_hash


def test_output_overlap_with_frozen_dataset_is_rejected(
    pilot_setup: tuple[GeometricPilotConfig, Path, Path],
) -> None:
    from captcha_quality.geometric_pilot import run_pilot

    config, frozen_root, tmp_path = pilot_setup
    with pytest.raises(GeometricPilotError):
        run_pilot(
            config,
            output_dataset=frozen_root,
            output_reports=tmp_path / "reports",
            overwrite=True,
        )


def test_manifest_json_is_canonical_and_field_spec_replays(
    pilot_setup: tuple[GeometricPilotConfig, Path, Path],
) -> None:
    from captcha_quality.geometric import parse_field_spec
    from captcha_quality.geometric_pilot import run_pilot

    config, _frozen_root, tmp_path = pilot_setup
    result = run_pilot(
        config,
        output_dataset=tmp_path / "pilot",
        output_reports=tmp_path / "reports",
        overwrite=True,
    )
    row = result.manifest[result.manifest["variant_family"] == "elastic"].iloc[0]
    payload = json.loads(str(row["field_spec_json"]))
    spec, field_hash = parse_field_spec(payload)
    assert spec.height == int(row["height"])
    assert spec.width == int(row["width"])
    assert field_hash == row["field_sha256"]
