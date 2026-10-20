"""Configuration parsing, validation and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from captcha_quality.config import SPLIT_NAMES, ConfigError, load_config
from captcha_quality.features import FEATURE_NAMES


def _mutate(path: Path, old: str, new: str) -> Path:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"test setup error: {old!r} not found"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_committed_pilot_config_loads() -> None:
    config = load_config("configs/dataset_pilot.toml")
    assert config.n_sources == 9
    assert config.n_samples == 81
    assert config.splits.ratios == (("train", 0.56), ("val", 0.22), ("test", 0.22))
    assert config.dataset.version == "v0.1.0-pilot"


def test_committed_dataset_config_loads() -> None:
    config = load_config("configs/dataset_v0_1_0.toml")
    assert config.n_sources == 60
    assert config.n_samples == 540
    assert config.dataset.text_lengths == (4, 5, 6)
    assert config.render.background_mode == "flat_random"
    assert config.families.n_variants_per_source == 8


def test_generation_hash_ignores_output_dir(dataset_config_writer, tmp_path: Path) -> None:
    first = load_config(dataset_config_writer(output_dir=tmp_path / "a"))
    second_path = dataset_config_writer(output_dir=tmp_path / "b")
    second = load_config(second_path)
    assert first.generation_config_hash == second.generation_config_hash


def test_generation_hash_changes_with_render_setting(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a")
    baseline = load_config(path).generation_config_hash
    _mutate(path, "jitter_fraction = 0.05", "jitter_fraction = 0.02")
    assert load_config(path).generation_config_hash != baseline


def test_generation_hash_ignores_diagnostic_thresholds(
    dataset_config_writer, tmp_path: Path
) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a")
    baseline = load_config(path).generation_config_hash
    _mutate(
        path,
        "trivial_single_feature_r2 = 0.90",
        "trivial_single_feature_r2 = 0.95",
    )
    assert load_config(path).generation_config_hash == baseline


def test_unknown_key_is_rejected(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a")
    _mutate(path, "[dataset]\n", "[dataset]\nmystery = 1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_missing_key_is_rejected(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a")
    _mutate(path, "text_length_weights = [0.5, 0.5]\n", "")
    with pytest.raises(ConfigError, match="text_length_weights"):
        load_config(path)


def test_text_length_weights_must_sum_to_one(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a", text_length_weights=(0.5, 0.6))
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)


def test_text_length_weights_must_match_lengths(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a", text_length_weights=(1.0,))
    with pytest.raises(ConfigError, match="one entry per text length"):
        load_config(path)


def test_split_ratios_must_sum_to_one(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a")
    _mutate(path, "train = 0.6", "train = 0.7")
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)


def test_zero_split_ratio_is_rejected(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a")
    _mutate(path, "val = 0.2", "val = 0.0")
    with pytest.raises(ConfigError, match="must be in"):
        load_config(path)


def test_duplicate_severity_levels_are_rejected(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a")
    _mutate(path, "levels = [0.0, 1.0, 2.5]", "levels = [0.0, 1.0, 1.0]")
    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(path)


def test_unsupported_family_is_rejected(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a")
    _mutate(path, 'names = ["gaussian_blur", "gaussian_noise"]', 'names = ["elastic"]')
    with pytest.raises(ConfigError, match="unsupported family"):
        load_config(path)


def test_stratified_splits_need_three_sources_per_font(
    dataset_config_writer, tmp_path: Path
) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a", sources_per_font=2)
    with pytest.raises(ConfigError, match="sources_per_font >= 3"):
        load_config(path)


def test_minimum_split_size_must_be_feasible(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(
        output_dir=tmp_path / "a", sources_per_font=3, min_sources_per_split=2
    )
    with pytest.raises(ConfigError, match="at least 6"):
        load_config(path)


def test_invalid_background_mode_is_rejected(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(output_dir=tmp_path / "a", background_mode="texture")
    with pytest.raises(ConfigError, match="background_mode"):
        load_config(path)


def test_background_range_must_be_ordered(dataset_config_writer, tmp_path: Path) -> None:
    path = dataset_config_writer(
        output_dir=tmp_path / "a",
        background_mode="flat_random",
        background_min=250,
        background_max=240,
    )
    with pytest.raises(ConfigError, match="background_min/background_max"):
        load_config(path)


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


def test_invalid_toml_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("[dataset\nname='x'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_split_names_are_fixed() -> None:
    assert SPLIT_NAMES == ("train", "val", "test")


def test_feature_names_are_stable_and_target_free() -> None:
    assert len(FEATURE_NAMES) == 8
    lowered = {name.lower() for name in FEATURE_NAMES}
    assert not lowered & {"ssim", "psnr", "ms_ssim"}
    assert not lowered & {"font_id", "text_length", "severity_level", "corruption_type"}
