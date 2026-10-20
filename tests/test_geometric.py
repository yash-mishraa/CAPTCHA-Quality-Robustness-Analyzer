"""Focused tests for the bounded elastic-warp primitive."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from captcha_quality.geometric import (
    BORDER_EDGE,
    BORDER_REFLECT,
    ElasticFieldSpec,
    ElasticLadder,
    GeometricError,
    apply_elastic,
    field_sha256,
    field_spec_json,
    generate_displacement_field,
    parse_field_spec,
    verify_field_replay,
)
from captcha_quality.imaging import ImageContractError


def _text_like_image(height: int = 64, width: int = 192) -> np.ndarray:
    image = np.full((height, width), 250, dtype=np.uint8)
    for column in range(8, width - 8, 13):
        image[18:46, column : column + 5] = 0
        image[24:30, column : column + 12] = 0
    return image


def _spec(seed: int = 123, **overrides: object) -> ElasticFieldSpec:
    values: dict[str, object] = {
        "height": 64,
        "width": 192,
        "seed": seed,
        "grid_rows": 5,
        "grid_cols": 13,
        "smoothing_sigma_px": 2.0,
        "interpolation": "bilinear",
        "border_mode": BORDER_EDGE,
    }
    values.update(overrides)
    return ElasticFieldSpec(**values)  # type: ignore[arg-type]


def test_field_is_deterministic_and_seed_sensitive() -> None:
    image = _text_like_image()
    spec = _spec()
    first = apply_elastic(image, field_spec=spec, amplitude_px=2.0)
    second = apply_elastic(image, field_spec=spec, amplitude_px=2.0)
    other = apply_elastic(image, field_spec=_spec(seed=124), amplitude_px=2.0)

    assert np.array_equal(first.field, second.field)
    assert np.array_equal(first.image, second.image)
    assert first.image.tobytes() == second.image.tobytes()
    assert field_sha256(first.field) == field_sha256(second.field)
    assert not np.array_equal(first.field, other.field)
    assert not np.array_equal(first.image, other.image)


def test_zero_amplitude_and_zero_field_control_are_exact_identity() -> None:
    image = _text_like_image()
    spec = _spec()
    bypassed = apply_elastic(image, field_spec=spec, amplitude_px=0.0)
    sampled = apply_elastic(image, field_spec=spec, amplitude_px=0.0, force_resample=True)

    assert not bypassed.resampled
    assert bypassed.resampled is False
    assert sampled.resampled is True
    assert np.array_equal(bypassed.image, image)
    assert np.array_equal(sampled.image, image)
    assert bypassed.changed_pixel_fraction == 0.0
    assert sampled.changed_pixel_fraction == 0.0
    assert bypassed.outside_fraction == 0.0


def test_minimal_displacement_changes_pixels_without_destroying_structure() -> None:
    image = _text_like_image()
    result = apply_elastic(image, field_spec=_spec(), amplitude_px=0.25)

    assert result.resampled
    assert 0.0 < result.changed_pixel_fraction < 0.5
    assert 0.0 < result.changed_ink_fraction <= 1.0
    assert result.intensity_std > 1.0
    assert not result.degenerate
    assert not np.array_equal(result.image, image)
    assert result.changed_pixel_fraction > 0.0
    assert result.outside_fraction >= 0.0


def test_field_specification_round_trips_and_replays_exact_pixels() -> None:
    image = _text_like_image()
    spec = _spec(seed=991)
    field = generate_displacement_field(spec)
    payload = field_spec_json(spec, field)
    parsed_spec, expected_hash = parse_field_spec(json.loads(payload))

    assert parsed_spec == spec
    assert expected_hash == field_sha256(field)
    replayed = verify_field_replay(
        image,
        json.loads(payload),
        amplitude_px=2.0,
        force_resample=False,
    )
    original = apply_elastic(image, field_spec=spec, amplitude_px=2.0)
    assert np.array_equal(replayed.image, original.image)
    assert field_sha256(replayed.field) == expected_hash


def test_border_modes_are_recorded_and_finite() -> None:
    image = np.arange(64 * 192, dtype=np.uint16).reshape(64, 192).astype(np.uint8)
    for mode in (BORDER_EDGE, BORDER_REFLECT):
        result = apply_elastic(
            image,
            field_spec=_spec(seed=5, border_mode=mode),
            amplitude_px=3.0,
        )
        assert result.field_spec.border_mode == mode
        assert result.image.dtype == np.uint8
        assert result.image.shape == image.shape
        assert np.isfinite(result.image.astype(np.float64)).all()
        assert result.outside_fraction > 0.0
        assert result.border_affected_fraction == result.outside_fraction


def test_amplitude_and_image_contract_are_enforced() -> None:
    image = _text_like_image()
    with pytest.raises(GeometricError):
        apply_elastic(image, field_spec=_spec(), amplitude_px=-0.1)
    with pytest.raises(GeometricError):
        apply_elastic(image, field_spec=_spec(), amplitude_px=math.nan)
    with pytest.raises(GeometricError):
        apply_elastic(image, field_spec=_spec(), amplitude_px=7.0)
    with pytest.raises(ImageContractError):
        apply_elastic(np.zeros((64, 192), dtype=np.float32), field_spec=_spec(), amplitude_px=1.0)
    with pytest.raises(GeometricError):
        apply_elastic(np.zeros((32, 32), dtype=np.uint8), field_spec=_spec(), amplitude_px=1.0)


def test_invalid_field_parameters_are_rejected() -> None:
    with pytest.raises(GeometricError):
        _spec(grid_rows=1)
    with pytest.raises(GeometricError):
        _spec(grid_rows=65)
    with pytest.raises(GeometricError):
        _spec(smoothing_sigma_px=math.nan)
    with pytest.raises(GeometricError):
        _spec(smoothing_sigma_px=-1.0)
    with pytest.raises(GeometricError):
        _spec(interpolation="bicubic")
    with pytest.raises(GeometricError):
        _spec(border_mode="wrap")
    with pytest.raises(GeometricError):
        _spec(seed=-1)


def test_ladder_is_strictly_increasing_and_minimal_is_smaller() -> None:
    ladder = ElasticLadder(amplitudes_px=(0.5, 1.0, 2.0), minimal_amplitude_px=0.25)
    assert ladder.n_levels == 3
    assert ladder.amplitude(2) == 1.0
    with pytest.raises(GeometricError):
        ladder.amplitude(0)
    with pytest.raises(GeometricError):
        ElasticLadder(amplitudes_px=(1.0, 1.0))
    with pytest.raises(GeometricError):
        ElasticLadder(amplitudes_px=(0.5,), minimal_amplitude_px=0.5)
    with pytest.raises(GeometricError):
        ElasticLadder(amplitudes_px=(0.0, 1.0))
