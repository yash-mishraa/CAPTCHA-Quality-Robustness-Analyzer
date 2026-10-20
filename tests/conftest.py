"""Shared fixtures for the Milestone 1 test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from captcha_quality.pilot import PilotConfig
from captcha_quality.render import (
    FontResolutionError,
    RenderConfig,
    ResolvedFont,
    render_clean,
    resolve_fonts,
)


@pytest.fixture(scope="session")
def font() -> ResolvedFont:
    """Return the first available pilot font, skipping if none is installed."""
    try:
        resolution = resolve_fonts(["arial", "courier_new", "times_new_roman"], min_required=1)
    except FontResolutionError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no pilot font available on this machine: {exc}")
    return resolution.available[0]


@pytest.fixture
def render_config() -> RenderConfig:
    """Small canvas that keeps the test suite fast and still fits SSIM's window."""
    return RenderConfig(canvas=(128, 48), text_length=4)


@pytest.fixture
def clean(font: ResolvedFont, render_config: RenderConfig):
    """A deterministic clean render for corruption and metric tests."""
    return render_clean(font, "AB23", render_config, rng=np.random.default_rng(1234))


@pytest.fixture
def clean_ladder():
    """Short blur ladder used where only severity 0 matters."""
    from captcha_quality.corruptions import FAMILY_BLUR, SeverityLadder

    return SeverityLadder(FAMILY_BLUR, (0.0, 1.0, 2.0), "px")


def write_pilot_config(
    path: Path,
    *,
    output_dir: Path,
    font_key: str,
    repeat_seeds: tuple[int, ...] = (0, 1),
) -> Path:
    """Write a minimal valid pilot configuration for tests."""
    seeds = ", ".join(str(seed) for seed in repeat_seeds)
    path.write_text(
        f"""
[run]
name = "test-pilot"
seed = 7
repeat_seeds = [{seeds}]
texts_per_font = 1
metrics = ["ssim", "psnr"]
deferred_metrics = ["ms_ssim"]
output_dir = {str(output_dir)!r}

[render]
canvas_width = 128
canvas_height = 48
charset = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
text_length = 4
fit_fraction = 0.86
vertical_padding = 0.10
jitter_fraction = 0.05

[fonts]
keys = ["{font_key}"]
min_required = 1

[families]
names = ["gaussian_blur", "gaussian_noise"]

[families.gaussian_blur]
unit = "px"
levels = [0.0, 1.0, 2.5]

[families.gaussian_noise]
unit = "sigma"
levels = [0.0, 8.0, 20.0]

[gate]
spearman_max = -0.90
paired_effect_max = -0.50
violation_rate_max = 0.35
group_spearman_max = -0.80
eta_squared_min = 0.50
ssim_range_min = 0.10
psnr_range_min_db = 3.0
ink_correlation_max = 0.95
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def pilot_config_writer(tmp_path: Path):
    """Return a helper that writes a minimal pilot config into ``tmp_path``."""

    def _write(
        *,
        output_dir: Path,
        font_key: str,
        repeat_seeds: tuple[int, ...] = (0, 1),
    ) -> Path:
        return write_pilot_config(
            tmp_path / "pilot.toml",
            output_dir=output_dir,
            font_key=font_key,
            repeat_seeds=repeat_seeds,
        )

    return _write


@pytest.fixture
def tiny_config(tmp_path: Path, font: ResolvedFont) -> PilotConfig:
    """A one-font, one-text, six-image pilot configuration."""
    config_path = write_pilot_config(
        tmp_path / "pilot.toml",
        output_dir=tmp_path / "out",
        font_key=font.key,
    )
    return PilotConfig.from_toml(config_path)


# ---------------------------------------------------------------------------
# Milestone 2: dataset pipeline fixtures
# ---------------------------------------------------------------------------


def write_dataset_config(
    path: Path,
    *,
    output_dir: Path,
    font_key: str,
    sources_per_font: int = 3,
    text_lengths: tuple[int, ...] = (3, 4),
    text_length_weights: tuple[float, ...] = (0.5, 0.5),
    min_sources_per_split: int = 1,
    background_mode: str = "white",
    background_min: int = 255,
    background_max: int = 255,
    center_ink: bool = True,
    strict_bounds: bool = True,
    version: str = "v0.0.0-test",
) -> Path:
    """Write a minimal valid dataset configuration for tests."""
    lengths = ", ".join(str(length) for length in text_lengths)
    weights = ", ".join(str(weight) for weight in text_length_weights)
    path.write_text(
        f"""
[dataset]
name = "test-dataset"
version = "{version}"
seed = 11
output_dir = {str(output_dir)!r}
sources_per_font = {sources_per_font}
text_lengths = [{lengths}]
text_length_weights = [{weights}]
charset = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

[render]
canvas_width = 128
canvas_height = 48
fit_fraction = 0.86
vertical_padding = 0.10
jitter_fraction = 0.05
spacing_jitter_fraction = 0.0
placement_jitter_fraction = 0.0
background_mode = "{background_mode}"
background_min = {background_min}
background_max = {background_max}
center_ink = {str(center_ink).lower()}
strict_bounds = {str(strict_bounds).lower()}

[fonts]
keys = ["{font_key}"]
min_required = 1

[families]
names = ["gaussian_blur", "gaussian_noise"]

[families.gaussian_blur]
unit = "px"
levels = [0.0, 1.0, 2.5]

[families.gaussian_noise]
unit = "sigma"
levels = [0.0, 8.0, 20.0]

[splits]
train = 0.6
val = 0.2
test = 0.2
seed = 5
min_sources_per_split = {min_sources_per_split}
stratify_by_font = true

[diagnostics]
trivial_single_feature_r2 = 0.90
strong_confound_r2 = 0.70
min_group_rows = 3
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def dataset_config_writer(tmp_path: Path, font: ResolvedFont):
    """Return a helper that writes a minimal dataset config into ``tmp_path``."""

    def _write(**overrides) -> Path:
        output_dir = overrides.pop("output_dir", tmp_path / "dataset")
        font_key = overrides.pop("font_key", font.key)
        return write_dataset_config(
            tmp_path / "dataset.toml",
            output_dir=output_dir,
            font_key=font_key,
            **overrides,
        )

    return _write


@pytest.fixture
def tiny_dataset(dataset_config_writer, tmp_path: Path):
    """A generated one-font, three-source dataset (15 samples)."""
    from captcha_quality.config import load_config
    from captcha_quality.dataset import generate_dataset

    config = load_config(dataset_config_writer(output_dir=tmp_path / "tiny"))
    return config, generate_dataset(config)
