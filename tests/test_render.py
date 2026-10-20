"""Tests for deterministic rendering, the image contract, and font resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from captcha_quality.imaging import ImageContractError, validate_image, validate_pair
from captcha_quality.render import (
    MIN_INK_FRACTION,
    FontResolutionError,
    FontSpec,
    RenderConfig,
    RenderError,
    render_clean,
    resolve_fonts,
    sample_text,
)


def test_render_is_reproducible(font, render_config: RenderConfig) -> None:
    first = render_clean(font, "AB23", render_config, rng=np.random.default_rng(99))
    second = render_clean(font, "AB23", render_config, rng=np.random.default_rng(99))
    assert first.sha256 == second.sha256
    assert np.array_equal(first.image, second.image)


def test_render_varies_with_jitter_seed(font, render_config: RenderConfig) -> None:
    first = render_clean(font, "AB23", render_config, rng=np.random.default_rng(1))
    second = render_clean(font, "AB23", render_config, rng=np.random.default_rng(2))
    assert first.sha256 != second.sha256


def test_clean_image_matches_documented_contract(clean, render_config: RenderConfig) -> None:
    image = clean.image
    assert image.shape == (render_config.canvas[1], render_config.canvas[0])
    assert image.dtype == np.uint8
    assert image.min() >= 0 and image.max() <= 255
    assert len(clean.text) == render_config.text_length
    assert MIN_INK_FRACTION < clean.ink_fraction < 1.0
    assert clean.font_key == clean.font_key.lower()


def test_clean_image_is_read_only(clean) -> None:
    assert not clean.image.flags.writeable
    with pytest.raises(ValueError):
        clean.image[0, 0] = 0


def test_sample_text_is_deterministic_and_in_charset(render_config: RenderConfig) -> None:
    first = sample_text(np.random.default_rng(5), render_config)
    second = sample_text(np.random.default_rng(5), render_config)
    assert first == second
    assert len(first) == render_config.text_length
    assert set(first) <= set(render_config.charset)


def test_render_rejects_unknown_characters(font, render_config: RenderConfig) -> None:
    with pytest.raises(RenderError, match="outside the charset"):
        render_clean(font, "AB0O", render_config, rng=np.random.default_rng(0))


def test_render_rejects_empty_text(font, render_config: RenderConfig) -> None:
    with pytest.raises(RenderError, match="must not be empty"):
        render_clean(font, "", render_config, rng=np.random.default_rng(0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"canvas": (10, 64)},
        {"canvas": (0, 64)},
        {"charset": "A"},
        {"charset": "AAB"},
        {"text_length": 0},
        {"fit_fraction": 0.0},
        {"vertical_padding": 0.6},
        {"jitter_fraction": 0.6},
    ],
)
def test_render_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(RenderError):
        RenderConfig(**kwargs)  # type: ignore[arg-type]


def test_font_resolution_skips_missing_fonts_without_substitution(tmp_path: Path, font) -> None:
    missing = FontSpec(key="missing", candidates=(tmp_path / "Missing.ttf",))
    present = FontSpec(key="present", candidates=(font.path,))
    resolution = resolve_fonts(["missing", "present"], specs=(missing, present), min_required=1)
    assert [item.key for item in resolution.available] == ["present"]
    assert [item.key for item in resolution.unavailable] == ["missing"]
    assert str(tmp_path / "Missing.ttf") in resolution.describe_unavailable()
    assert resolution.available[0].path == font.path


def test_font_resolution_fails_clearly_when_too_few_fonts(tmp_path: Path) -> None:
    missing = FontSpec(key="missing", candidates=(tmp_path / "Missing.ttf",))
    with pytest.raises(FontResolutionError, match="only 0 of 1 requested fonts are available"):
        resolve_fonts(["missing"], specs=(missing,), min_required=1)


def test_font_resolution_rejects_unknown_keys(font) -> None:
    with pytest.raises(FontResolutionError, match="unknown font key"):
        resolve_fonts(["comic_sans"], min_required=1)


def test_image_contract_rejects_invalid_arrays() -> None:
    with pytest.raises(ImageContractError, match="2-D"):
        validate_image(np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(ImageContractError, match="dtype uint8"):
        validate_image(np.zeros((16, 16), dtype=np.float32))
    with pytest.raises(ImageContractError, match="at least 11x11"):
        validate_image(np.zeros((10, 16), dtype=np.uint8))
    with pytest.raises(ImageContractError, match="numpy array"):
        validate_image([[0, 1], [2, 3]])  # type: ignore[arg-type]


def test_image_pair_contract_rejects_misalignment() -> None:
    reference = np.zeros((16, 16), dtype=np.uint8)
    distorted = np.zeros((16, 17), dtype=np.uint8)
    with pytest.raises(ImageContractError, match="identical shapes"):
        validate_pair(reference, distorted)
