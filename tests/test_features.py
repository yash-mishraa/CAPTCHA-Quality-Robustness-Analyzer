"""Documented handcrafted image features: definitions, validity and stability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from captcha_quality.corruptions import (
    FAMILY_BLUR,
    FAMILY_NOISE,
    SeverityLadder,
    apply_corruption,
)
from captcha_quality.features import (
    FEATURE_NAMES,
    FeatureError,
    extract_feature_table,
    extract_features,
)
from captcha_quality.imaging import ImageContractError
from captcha_quality.render import RenderConfig, render_clean

EXPECTED_FEATURE_ORDER = (
    "intensity_mean",
    "intensity_std",
    "contrast_rms",
    "ink_fraction",
    "laplacian_variance",
    "edge_density",
    "histogram_entropy",
    "noise_sigma_immerkaer",
)


def _uniform(value: int, size: tuple[int, int] = (32, 48)) -> np.ndarray:
    return np.full(size, value, dtype=np.uint8)


def test_feature_names_and_order_are_stable() -> None:
    assert FEATURE_NAMES == EXPECTED_FEATURE_ORDER


def test_feature_dict_keys_follow_declared_order(font) -> None:
    image = render_clean(
        font, "AB2", RenderConfig(canvas=(128, 48)), rng=np.random.default_rng(0)
    ).image
    features = extract_features(image)
    assert tuple(features) == FEATURE_NAMES
    assert all(isinstance(value, float) for value in features.values())
    assert all(math.isfinite(value) for value in features.values())


def test_extraction_is_deterministic(font) -> None:
    image = render_clean(
        font, "AB2", RenderConfig(canvas=(128, 48)), rng=np.random.default_rng(7)
    ).image
    assert extract_features(image) == extract_features(image)


def test_uniform_image_is_handled_without_nan() -> None:
    for value in (0, 128, 255):
        features = extract_features(_uniform(value))
        assert all(math.isfinite(feature) for feature in features.values())
        assert features["intensity_mean"] == pytest.approx(float(value))
        assert features["intensity_std"] == 0.0
        assert features["contrast_rms"] == 0.0
        assert features["ink_fraction"] == 0.0
        assert features["laplacian_variance"] == 0.0
        assert features["edge_density"] == 0.0
        assert features["histogram_entropy"] == 0.0
        assert features["noise_sigma_immerkaer"] == 0.0


def test_near_uniform_image_is_finite() -> None:
    image = np.full((24, 24), 200, dtype=np.uint8)
    image[12, 12] = 201
    features = extract_features(image)
    assert all(math.isfinite(value) for value in features.values())


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ImageContractError):
        extract_features(np.zeros((32, 48), dtype=np.float64))
    with pytest.raises(ImageContractError):
        extract_features(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(ImageContractError):
        extract_features(np.zeros((32, 48, 3), dtype=np.uint8))


def test_ink_fraction_matches_simple_threshold_measure(font) -> None:
    image = render_clean(
        font, "AB23", RenderConfig(canvas=(128, 48)), rng=np.random.default_rng(3)
    ).image
    simple = float(np.mean(image < 128))
    otsu = extract_features(image)["ink_fraction"]
    assert otsu == pytest.approx(simple, abs=0.08)


def test_blur_reduces_high_frequency_features(font) -> None:
    clean = render_clean(font, "AB23", RenderConfig(canvas=(128, 48)), rng=np.random.default_rng(4))
    ladder = SeverityLadder(FAMILY_BLUR, (0.0, 1.0, 3.0), "px")
    blurred = apply_corruption(clean.image, ladder, 2).image
    clean_features = extract_features(clean.image)
    blurred_features = extract_features(blurred)
    assert blurred_features["laplacian_variance"] < clean_features["laplacian_variance"]
    assert blurred_features["edge_density"] < clean_features["edge_density"]


def test_noise_increases_noise_estimate_and_entropy(font) -> None:
    clean = render_clean(font, "AB23", RenderConfig(canvas=(128, 48)), rng=np.random.default_rng(5))
    ladder = SeverityLadder(FAMILY_NOISE, (0.0, 20.0), "sigma")
    noisy = apply_corruption(clean.image, ladder, 1, noise_seed=99).image
    clean_features = extract_features(clean.image)
    noisy_features = extract_features(noisy)
    assert noisy_features["noise_sigma_immerkaer"] > clean_features["noise_sigma_immerkaer"]
    assert noisy_features["laplacian_variance"] > clean_features["laplacian_variance"]
    # 8-bit clipping compresses extreme noise (see M1 evidence), so intensity_std
    # is not a reliable noise indicator; entropy and the Laplacian are.
    assert noisy_features["histogram_entropy"] > clean_features["histogram_entropy"]


def test_batch_table_columns_and_order(font) -> None:
    first = _uniform(255)
    second = _uniform(0)
    table = extract_feature_table([("b", second), ("a", first)])
    assert tuple(table.columns) == ("sample_id", *FEATURE_NAMES)
    assert list(table["sample_id"]) == ["b", "a"]
    assert len(table) == 2


def test_batch_rejects_duplicate_and_empty_ids() -> None:
    with pytest.raises(FeatureError, match="duplicate sample_id"):
        extract_feature_table([("x", _uniform(255)), ("x", _uniform(0))])
    with pytest.raises(FeatureError, match="non-empty string"):
        extract_feature_table([("", _uniform(255))])


def test_feature_table_is_reproducible(font) -> None:
    image = render_clean(
        font, "AB2", RenderConfig(canvas=(128, 48)), rng=np.random.default_rng(9)
    ).image
    first = extract_feature_table([("s1", image)])
    second = extract_feature_table([("s1", image)])
    assert first.equals(second)
