"""Tests for corruption ladders, determinism, and non-mutation guarantees."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from captcha_quality.corruptions import (
    DEFAULT_BLUR_RADII,
    DEFAULT_NOISE_SIGMAS,
    FAMILY_BLUR,
    FAMILY_NOISE,
    CorruptionError,
    SeverityLadder,
    apply_corruption,
    default_ladders,
    derive_noise_seed,
)
from captcha_quality.imaging import ImageContractError


def test_identity_severity_returns_equal_copy(clean, clean_ladder) -> None:
    result = apply_corruption(clean.image, clean_ladder, 0)
    assert np.array_equal(result.image, clean.image)
    assert result.image is not clean.image
    assert result.image.flags.writeable
    assert result.spec.is_identity
    assert result.clipped_fraction == 0.0
    assert result.realised_noise_std == 0.0


def test_blur_is_deterministic(clean) -> None:
    ladder = SeverityLadder(FAMILY_BLUR, DEFAULT_BLUR_RADII, "px")
    first = apply_corruption(clean.image, ladder, 3).image
    second = apply_corruption(clean.image, ladder, 3).image
    assert np.array_equal(first, second)
    assert not np.array_equal(first, clean.image)


def test_noise_is_deterministic_given_seed(clean) -> None:
    ladder = SeverityLadder(FAMILY_NOISE, DEFAULT_NOISE_SIGMAS, "sigma")
    seed = derive_noise_seed("trial", 2)
    first = apply_corruption(clean.image, ladder, 3, noise_seed=seed).image
    second = apply_corruption(clean.image, ladder, 3, noise_seed=seed).image
    third = apply_corruption(clean.image, ladder, 3, noise_seed=seed + 1).image
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


@pytest.mark.parametrize("severity", [0, 1, 2, 3, 4])
def test_corruption_never_mutates_the_clean_reference(clean, severity: int) -> None:
    before = hashlib.sha256(clean.image.tobytes()).hexdigest()
    for ladder in default_ladders().values():
        result = apply_corruption(
            clean.image, ladder, severity, noise_seed=derive_noise_seed(severity)
        )
        assert result.image.shape == clean.image.shape
        assert result.image.dtype == np.uint8
        assert result.image.min() >= 0 and result.image.max() <= 255
    after = hashlib.sha256(clean.image.tobytes()).hexdigest()
    assert before == after == clean.sha256


def test_noise_has_expected_direction_and_magnitude(clean) -> None:
    ladder = SeverityLadder(FAMILY_NOISE, DEFAULT_NOISE_SIGMAS, "sigma")
    result = apply_corruption(clean.image, ladder, 4, noise_seed=derive_noise_seed("mag"))
    difference = result.image.astype(np.float64) - clean.image.astype(np.float64)
    assert np.mean(np.abs(difference)) > 10.0
    assert result.realised_noise_std > 10.0
    assert 0.0 < result.clipped_fraction < 1.0


def test_blur_reduces_gradient_energy_monotonically(clean) -> None:
    ladder = SeverityLadder(FAMILY_BLUR, DEFAULT_BLUR_RADII, "px")
    energies = []
    for severity in range(ladder.n_levels):
        image = apply_corruption(clean.image, ladder, severity).image.astype(np.float64)
        energy = float(
            np.mean(np.abs(np.diff(image, axis=0))) + np.mean(np.abs(np.diff(image, axis=1)))
        )
        energies.append(energy)
    assert all(later < earlier for earlier, later in zip(energies, energies[1:], strict=False))


def test_blur_preserves_image_shape_and_range(clean) -> None:
    ladder = SeverityLadder(FAMILY_BLUR, DEFAULT_BLUR_RADII, "px")
    result = apply_corruption(clean.image, ladder, 4)
    assert result.image.shape == clean.image.shape
    assert result.image.dtype == np.uint8
    assert result.image.max() <= 255


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ((0.0,), "at least 2 levels"),
        ((1.0, 2.0), "identity baseline"),
        ((0.0, -1.0), ">= 0"),
        ((0.0, 2.0, 1.0), "non-decreasing"),
        ((0.5, 1.0, 2.0), "identity baseline"),
    ],
)
def test_ladder_rejects_invalid_parameters(parameters, match: str) -> None:
    with pytest.raises(CorruptionError, match=match):
        SeverityLadder(FAMILY_BLUR, parameters, "px")


def test_ladder_rejects_unknown_family() -> None:
    with pytest.raises(CorruptionError, match="unknown corruption family"):
        SeverityLadder("jpeg", (0.0, 1.0), "quality")


def test_ladder_rejects_out_of_range_severity(clean) -> None:
    ladder = SeverityLadder(FAMILY_BLUR, (0.0, 1.0), "px")
    with pytest.raises(CorruptionError, match="out of range"):
        ladder.parameter(2)
    with pytest.raises(CorruptionError, match="out of range"):
        apply_corruption(clean.image, ladder, 2)


def test_noise_requires_an_explicit_seed(clean) -> None:
    ladder = SeverityLadder(FAMILY_NOISE, (0.0, 8.0), "sigma")
    with pytest.raises(CorruptionError, match="requires an explicit noise_seed"):
        apply_corruption(clean.image, ladder, 1)


def test_corruption_rejects_invalid_images() -> None:
    ladder = SeverityLadder(FAMILY_BLUR, (0.0, 1.0), "px")
    with pytest.raises(ImageContractError):
        apply_corruption(np.zeros((16, 16), dtype=np.float32), ladder, 1)


def test_derive_noise_seed_is_stable_and_sensitive_to_inputs() -> None:
    assert derive_noise_seed("a", 1) == derive_noise_seed("a", 1)
    assert derive_noise_seed("a", 1) != derive_noise_seed("a", 2)
    assert 0 <= derive_noise_seed("a", 1) < 2**64
