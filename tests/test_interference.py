"""Tests for the occlusion/visual-interference pilot corruption family."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from captcha_quality.imaging import ImageContractError
from captcha_quality.interference import (
    DEFAULT_STROKE_COUNTS,
    SUPPORTED_VARIANTS,
    VARIANT_ARC_STROKES,
    VARIANT_THIN_LINES,
    ArcStroke,
    InterferenceError,
    InterferenceLadder,
    LineStroke,
    apply_interference,
    apply_strokes,
    default_ladders,
    geometry_to_json,
    plan_stroke_pool,
    strokes_from_json,
)
from captcha_quality.quality_metrics import ssim


def _ladder(variant: str = VARIANT_THIN_LINES) -> InterferenceLadder:
    return InterferenceLadder(variant=variant, stroke_counts=(1, 2, 4))


def test_default_ladders_cover_both_variants() -> None:
    ladders = default_ladders()
    assert set(ladders) == set(SUPPORTED_VARIANTS)
    for ladder in ladders.values():
        assert ladder.stroke_counts == DEFAULT_STROKE_COUNTS


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_interference_is_deterministic_given_seed(clean, variant: str) -> None:
    ladder = _ladder(variant)
    first = apply_interference(clean.image, ladder, 2, seed=42)
    second = apply_interference(clean.image, ladder, 2, seed=42)
    third = apply_interference(clean.image, ladder, 2, seed=43)
    assert np.array_equal(first.image, second.image)
    assert first.geometry == second.geometry
    assert not np.array_equal(first.image, third.image)


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_severities_are_nested_prefixes_of_one_pool(clean, variant: str) -> None:
    ladder = _ladder(variant)
    pool = plan_stroke_pool(clean.image, ladder, seed=7)
    assert len(pool) == ladder.max_count
    for severity in range(1, ladder.n_levels + 1):
        result = apply_interference(clean.image, ladder, severity, seed=7)
        assert len(result.geometry) == ladder.stroke_counts[severity - 1]
        assert result.geometry == pool[: ladder.stroke_counts[severity - 1]]


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_interference_preserves_contract_and_does_not_mutate(clean, variant: str) -> None:
    ladder = _ladder(variant)
    before = hashlib.sha256(clean.image.tobytes()).hexdigest()
    result = apply_interference(clean.image, ladder, 3, seed=99)
    after = hashlib.sha256(clean.image.tobytes()).hexdigest()
    assert before == after == clean.sha256
    assert result.image.shape == clean.image.shape
    assert result.image.dtype == np.uint8
    assert result.image.min() >= 0 and result.image.max() <= 255
    assert result.image.flags.writeable
    assert 0.0 < result.changed_pixel_fraction < 1.0
    assert 0.0 <= result.changed_ink_fraction <= 1.0
    assert 0.0 <= result.changed_background_fraction <= 1.0


@pytest.mark.parametrize("variant", SUPPORTED_VARIANTS)
def test_more_strokes_do_not_increase_ssim(clean, variant: str) -> None:
    ladder = _ladder(variant)
    values = [
        ssim(clean.image, apply_interference(clean.image, ladder, severity, seed=123).image)
        for severity in range(1, ladder.n_levels + 1)
    ]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:], strict=False))
    assert values[-1] < values[0]


def test_strokes_are_one_pixel_and_reach_the_border(clean) -> None:
    ladder = _ladder(VARIANT_THIN_LINES)
    result = apply_interference(clean.image, ladder, 2, seed=5)
    assert all(isinstance(stroke, LineStroke) for stroke in result.geometry)
    assert result.border_touched
    for stroke in result.geometry:
        assert 0 <= stroke.start[0] < clean.image.shape[1]
        assert 0 <= stroke.start[1] < clean.image.shape[0]
        assert 0 <= stroke.end[0] < clean.image.shape[1]
        assert 0 <= stroke.end[1] < clean.image.shape[0]


def test_arc_geometry_is_recorded_and_replayable(clean) -> None:
    ladder = _ladder(VARIANT_ARC_STROKES)
    result = apply_interference(clean.image, ladder, 3, seed=11)
    assert all(isinstance(stroke, ArcStroke) for stroke in result.geometry)
    payload = geometry_to_json(list(result.geometry))
    replayed = apply_strokes(clean.image, list(strokes_from_json(payload)), width_px=1, intensity=0)
    assert np.array_equal(replayed, result.image)
    assert json.loads(json.dumps(payload)) == payload


def test_geometry_round_trip_preserves_both_stroke_types(clean) -> None:
    lines = apply_interference(clean.image, _ladder(VARIANT_THIN_LINES), 1, seed=3)
    arcs = apply_interference(clean.image, _ladder(VARIANT_ARC_STROKES), 1, seed=3)
    combined = [*lines.geometry, *arcs.geometry]
    assert strokes_from_json(geometry_to_json(combined)) == tuple(combined)


def test_apply_strokes_with_no_strokes_returns_a_copy(clean) -> None:
    copied = apply_strokes(clean.image, [], width_px=1, intensity=0)
    assert np.array_equal(copied, clean.image)
    assert copied is not clean.image


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"stroke_counts": ()}, "must not be empty"),
        ({"stroke_counts": (0, 1)}, ">= 1"),
        ({"stroke_counts": (2, 1)}, "strictly increasing"),
        ({"stroke_counts": (1, 1)}, "strictly increasing"),
        ({"stroke_counts": (1, 2), "width_px": 0}, "width_px"),
        ({"stroke_counts": (1, 2), "width_px": 9}, "width_px"),
        ({"stroke_counts": (1, 2), "intensity": -1}, "intensity"),
        ({"stroke_counts": (1, 2), "intensity": 256}, "intensity"),
        ({"stroke_counts": (1, 2), "line_angle_deg": (10.0, -10.0)}, "line_angle_deg"),
        ({"stroke_counts": (1, 2), "arc_sweep_deg": (0.0, 90.0)}, "arc_sweep_deg"),
        ({"stroke_counts": (1, 2), "arc_radius_fraction": (0.0, 0.5)}, "arc_radius_fraction"),
        ({"stroke_counts": (1, 2), "arc_aspect_ratio": (2.0, 1.0)}, "arc_aspect_ratio"),
    ],
)
def test_ladder_rejects_invalid_parameters(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(InterferenceError, match=match):
        InterferenceLadder(variant=VARIANT_THIN_LINES, **kwargs)  # type: ignore[arg-type]


def test_ladder_rejects_unknown_variant() -> None:
    with pytest.raises(InterferenceError, match="unknown interference variant"):
        InterferenceLadder(variant="jpeg", stroke_counts=(1,))


def test_severity_is_one_based_and_validated(clean) -> None:
    ladder = _ladder()
    assert ladder.spec(1).stroke_count == 1
    assert ladder.spec(3).stroke_count == 4
    with pytest.raises(InterferenceError, match="out of range"):
        ladder.spec(0)
    with pytest.raises(InterferenceError, match="out of range"):
        ladder.spec(4)
    with pytest.raises(InterferenceError, match="out of range"):
        apply_interference(clean.image, ladder, 4, seed=1)


@pytest.mark.parametrize("seed", [-1, 2**64, 1.5, True])
def test_seed_is_explicit_and_validated(clean, seed: object) -> None:
    with pytest.raises(InterferenceError, match="seed"):
        plan_stroke_pool(clean.image, _ladder(), seed=seed)  # type: ignore[arg-type]


def test_interference_rejects_invalid_images() -> None:
    ladder = _ladder()
    with pytest.raises(ImageContractError):
        apply_interference(np.zeros((16, 16), dtype=np.float32), ladder, 1, seed=1)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        [{}],
        [{"kind": "line", "start": [0, 0]}],
        [{"kind": "bezier"}],
        [{"kind": "arc", "center": [0, 0], "radii": [1, 1]}],
    ],
)
def test_invalid_geometry_payloads_are_rejected(payload: object) -> None:
    with pytest.raises(InterferenceError):
        strokes_from_json(payload)


def test_apply_strokes_validates_width_and_intensity(clean) -> None:
    strokes = plan_stroke_pool(clean.image, _ladder(), seed=1)[:1]
    with pytest.raises(InterferenceError, match="width_px"):
        apply_strokes(clean.image, list(strokes), width_px=0, intensity=0)
    with pytest.raises(InterferenceError, match="intensity"):
        apply_strokes(clean.image, list(strokes), width_px=1, intensity=300)
