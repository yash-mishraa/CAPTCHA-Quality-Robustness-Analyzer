"""Tests for the reference-based metrics and their validation behaviour."""

from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import pytest

from captcha_quality.imaging import ImageContractError
from captcha_quality.quality_metrics import (
    DEFERRED_METRICS,
    MetricUnavailableError,
    UnknownMetricError,
    compute_pair_metrics,
    describe_metrics,
    psnr,
    ssim,
)


def test_ssim_of_identical_images_is_exactly_one(clean) -> None:
    assert ssim(clean.image, clean.image) == 1.0


def test_psnr_of_identical_images_is_infinite_without_warning(clean) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = psnr(clean.image, clean.image)
    assert math.isinf(value) and value > 0
    assert [item for item in caught if item.category is RuntimeWarning] == []


def test_psnr_matches_analytic_value_for_unit_offset() -> None:
    reference = np.full((16, 16), 100, dtype=np.uint8)
    distorted = (reference.astype(np.int16) + 1).astype(np.uint8)
    expected = 20.0 * math.log10(255.0)  # MSE = 1, data_range = 255
    assert psnr(reference, distorted) == pytest.approx(expected, abs=1e-9)


def test_ssim_is_high_but_below_one_for_brightness_offset() -> None:
    reference = np.full((16, 16), 100, dtype=np.uint8)
    distorted = (reference.astype(np.int16) + 1).astype(np.uint8)
    value = ssim(reference, distorted)
    assert 0.99 < value < 1.0


def test_metrics_detect_degradation(clean) -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 25.0, size=clean.image.shape)
    degraded = np.rint(np.clip(clean.image + noise, 0, 255)).astype(np.uint8)
    assert ssim(clean.image, degraded) < 0.95
    assert psnr(clean.image, degraded) < 30.0


def test_compute_pair_metrics_returns_requested_metrics(clean) -> None:
    values = compute_pair_metrics(clean.image, clean.image, ("ssim", "psnr"))
    assert values["ssim"] == 1.0
    assert math.isinf(values["psnr"])


def test_metrics_reject_mismatched_or_invalid_inputs(clean) -> None:
    with pytest.raises(ImageContractError, match="identical shapes"):
        ssim(clean.image, clean.image[:, :-1])
    with pytest.raises(ImageContractError, match="dtype uint8"):
        psnr(clean.image, clean.image.astype(np.float32))
    with pytest.raises(ImageContractError, match="2-D"):
        ssim(np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8))


def test_unknown_and_deferred_metrics_are_rejected(clean) -> None:
    with pytest.raises(UnknownMetricError, match="unknown metric"):
        compute_pair_metrics(clean.image, clean.image, ("bogus",))
    with pytest.raises(UnknownMetricError, match="PyTorch"):
        compute_pair_metrics(clean.image, clean.image, ("ms_ssim",))


def test_ms_ssim_is_recorded_as_deferred() -> None:
    assert "ms_ssim" in DEFERRED_METRICS
    assert "scikit-image" in DEFERRED_METRICS["ms_ssim"]


def test_metric_descriptions_document_the_range() -> None:
    descriptions = describe_metrics()
    assert "data_range=255" in descriptions["ssim"]
    assert "+inf" in descriptions["psnr"]


def test_missing_scikit_image_raises_actionable_error(clean, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "skimage", None)
    monkeypatch.delitem(sys.modules, "skimage.metrics", raising=False)
    with pytest.raises(MetricUnavailableError, match="uv sync --extra image"):
        ssim(clean.image, clean.image)
