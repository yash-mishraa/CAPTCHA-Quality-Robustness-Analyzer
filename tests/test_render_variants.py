"""Dataset-renderer extensions: backgrounds, layout variation and bounds."""

from __future__ import annotations

import numpy as np
import pytest

from captcha_quality.render import (
    BACKGROUND,
    RenderConfig,
    RenderError,
    render_clean,
    sample_text,
)


def _config(**overrides) -> RenderConfig:
    defaults = {"canvas": (128, 48), "text_length": 4, "center_ink": True, "strict_bounds": True}
    defaults.update(overrides)
    return RenderConfig(**defaults)


def test_defaults_preserve_the_milestone_one_condition() -> None:
    config = RenderConfig()
    assert config.canvas == (192, 64)
    assert config.text_length == 5
    assert config.background_mode == "white"
    assert config.center_ink is False
    assert config.strict_bounds is False
    assert config.spacing_jitter_fraction == 0.0
    assert config.placement_jitter_fraction == 0.0


def test_white_background_is_constant(font) -> None:
    render = render_clean(font, "AB23", _config(), rng=np.random.default_rng(0))
    assert render.background_level == BACKGROUND
    assert int(np.median(render.image)) == BACKGROUND


def test_flat_random_background_is_seeded_and_in_range(font) -> None:
    config = _config(background_mode="flat_random", background_range=(240, 250))
    first = render_clean(font, "AB23", config, rng=np.random.default_rng(12))
    second = render_clean(font, "AB23", config, rng=np.random.default_rng(12))
    assert 240 <= first.background_level <= 250
    assert first.background_level == second.background_level
    assert np.array_equal(first.image, second.image)


def test_layout_jitter_changes_the_image(font) -> None:
    strict = _config(spacing_jitter_fraction=0.03, placement_jitter_fraction=0.02)
    plain = _config()
    jittered = render_clean(font, "AB23", strict, rng=np.random.default_rng(3))
    centred = render_clean(font, "AB23", plain, rng=np.random.default_rng(3))
    assert not np.array_equal(jittered.image, centred.image)
    assert jittered.ink_fraction > 0.0


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(RenderError, match="background_mode"):
        RenderConfig(background_mode="texture")
    with pytest.raises(RenderError, match="background_range"):
        RenderConfig(background_range=(255, 250))
    with pytest.raises(RenderError, match="spacing_jitter_fraction"):
        RenderConfig(spacing_jitter_fraction=0.9)
    with pytest.raises(RenderError, match="placement_jitter_fraction"):
        RenderConfig(placement_jitter_fraction=-0.1)
    with pytest.raises(RenderError, match="center_ink"):
        RenderConfig(center_ink="yes")  # type: ignore[arg-type]
    with pytest.raises(RenderError, match="strict_bounds"):
        RenderConfig(strict_bounds=1)  # type: ignore[arg-type]


def test_strict_bounds_rejects_clipping(font) -> None:
    config = _config(jitter_fraction=0.4, strict_bounds=True)
    with pytest.raises(RenderError, match="clipped"):
        render_clean(font, "AB23", config, rng=np.random.default_rng(4))


def test_center_ink_moves_the_block_away_from_the_edge(font) -> None:
    anchor_config = _config(center_ink=False, strict_bounds=False)
    centred_config = _config(center_ink=True)
    anchored = render_clean(font, "AB23", anchor_config, rng=np.random.default_rng(8))
    centred = render_clean(font, "AB23", centred_config, rng=np.random.default_rng(8))
    assert not np.array_equal(anchored.image, centred.image)
    ink = centred.image < 128
    assert not ink[0].any() and not ink[-1].any()
    assert not ink[:, 0].any() and not ink[:, -1].any()


def test_sample_text_length_override_is_deterministic() -> None:
    config = _config(text_length=3)
    first = sample_text(np.random.default_rng(1), config, length=6)
    second = sample_text(np.random.default_rng(1), config, length=6)
    assert len(first) == 6
    assert first == second
