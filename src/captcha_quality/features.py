"""Typed, deterministic handcrafted image features.

The feature set is intentionally small and fully documented. Every feature is a
scalar computed from a single 2-D ``uint8`` grayscale image; the extractor never
sees the clean reference, the SSIM/PSNR targets, or generation metadata. That is
the leakage boundary: features describe the degraded image a model would
receive, and the target is measured separately against the clean reference.

Definitions (all on the degraded image):

``intensity_mean``
    Mean pixel intensity in ``[0, 255]``.
``intensity_std``
    Population standard deviation of pixel intensity, in gray levels.
``contrast_rms``
    RMS contrast, ``std / (mean + 1e-6)``, i.e. the coefficient of variation of
    intensity. Unlike percentile-based Michelson contrast it does not saturate
    at 1 for clean dark-on-light renders, so it keeps discriminating blurred
    and noisy images. Zero for a uniform image.
``ink_fraction``
    Fraction of pixels at or below the Otsu threshold. On light backgrounds the
    darker class is the glyph ink. Zero for an exactly uniform image, where no
    ink can be distinguished.
``laplacian_variance``
    4-neighbour Laplacian applied with edge-replicated padding, in gray levels;
    variance over all pixels.
``edge_density``
    Fraction of pixels marked by Canny on the image scaled to ``[0, 1]`` with
    ``sigma=1.0``, ``low_threshold=0.1``, ``high_threshold=0.2``.
``histogram_entropy``
    Shannon entropy in bits of a 256-bin histogram over ``[0, 256)``, bins
    normalised to sum to 1. Maximum 8 bits; zero for a uniform image.
``noise_sigma_immerkaer``
    Immerkær's fast noise-variance estimate, reported as a standard deviation
    in gray levels, using the 3x3 Laplacian mask and interior pixels only.
    On these renders it responds to additive noise but also to high-frequency
    glyph edges; it is a noise-plus-texture estimate, not a pure noise measure.

The estimator does not use SSIM, PSNR, corruption severity, font, text length, or
any other target/metadata field, and no feature is selected using the test split.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Final

import numpy as np
import pandas as pd

from captcha_quality.imaging import validate_image

#: Stable feature order. Changing names or order requires bumping
#: ``captcha_quality.config.FEATURE_SET_VERSION`` so old feature tables remain
#: distinguishable.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "intensity_mean",
    "intensity_std",
    "contrast_rms",
    "ink_fraction",
    "laplacian_variance",
    "edge_density",
    "histogram_entropy",
    "noise_sigma_immerkaer",
)

#: 256 bins over the full 8-bit range, one bin per integer intensity.
HISTOGRAM_BINS: Final[int] = 256

#: Canny settings on the image scaled to [0, 1].
EDGE_SIGMA: Final[float] = 1.0
EDGE_LOW_THRESHOLD: Final[float] = 0.1
EDGE_HIGH_THRESHOLD: Final[float] = 0.2

#: Small denominator guard for contrast; documented rather than implicit.
CONTRAST_EPSILON: Final[float] = 1e-6


class FeatureError(ValueError):
    """Raised when a feature cannot be computed from the supplied input."""


class FeatureDependencyError(RuntimeError):
    """Raised when the optional scikit-image dependency is unavailable."""


def _threshold_otsu(image: np.ndarray) -> float:
    try:
        from skimage.filters import threshold_otsu
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise FeatureDependencyError(
            "scikit-image is required for Otsu thresholding and Canny edges. "
            "Install the image extra, e.g. 'uv sync --extra dev --extra image'."
        ) from exc
    return float(threshold_otsu(image))


def _canny(image: np.ndarray) -> np.ndarray:
    try:
        from skimage.feature import canny
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise FeatureDependencyError(
            "scikit-image is required for Canny edge detection. "
            "Install the image extra, e.g. 'uv sync --extra dev --extra image'."
        ) from exc
    return canny(
        image,
        sigma=EDGE_SIGMA,
        low_threshold=EDGE_LOW_THRESHOLD,
        high_threshold=EDGE_HIGH_THRESHOLD,
    )


def _ink_fraction(image: np.ndarray) -> float:
    """Fraction of pixels in the darker Otsu class."""
    threshold = _threshold_otsu(image)
    if not math.isfinite(threshold):
        return 0.0
    return float(np.mean(image <= threshold))


def _laplacian_variance(image: np.ndarray) -> float:
    values = image.astype(np.float64)
    padded = np.pad(values, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(laplacian.var())


def _edge_density(image: np.ndarray) -> float:
    edges = _canny(image.astype(np.float64) / 255.0)
    return float(np.mean(edges))


def _histogram_entropy(image: np.ndarray) -> float:
    counts, _ = np.histogram(image, bins=HISTOGRAM_BINS, range=(0, HISTOGRAM_BINS))
    probabilities = counts.astype(np.float64) / float(image.size)
    nonzero = probabilities[probabilities > 0.0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _noise_sigma_immerkaer(image: np.ndarray) -> float:
    """Immerkær noise standard-deviation estimate in gray levels."""
    height, width = image.shape
    if height < 3 or width < 3:  # pragma: no cover - validate_image enforces >= 11
        return 0.0
    values = image.astype(np.float64)
    top_left = values[:-2, :-2]
    top_mid = values[:-2, 1:-1]
    top_right = values[:-2, 2:]
    mid_left = values[1:-1, :-2]
    centre = values[1:-1, 1:-1]
    mid_right = values[1:-1, 2:]
    bottom_left = values[2:, :-2]
    bottom_mid = values[2:, 1:-1]
    bottom_right = values[2:, 2:]
    response = (
        top_left
        - 2.0 * top_mid
        + top_right
        - 2.0 * mid_left
        + 4.0 * centre
        - 2.0 * mid_right
        + bottom_left
        - 2.0 * bottom_mid
        + bottom_right
    )
    scale = math.sqrt(math.pi / 2.0) / (6.0 * (height - 2) * (width - 2))
    return float(scale * np.abs(response).sum())


def extract_features(image: np.ndarray) -> dict[str, float]:
    """Compute the documented feature vector for one grayscale image.

    Parameters
    ----------
    image:
        2-D ``uint8`` array with at least 11 pixels per side (the shared image
        contract used by the metric code).

    Returns
    -------
    dict
        Feature name to ``float``, in :data:`FEATURE_NAMES` order. Exactly
        uniform images take an explicit shortcut that avoids undefined
        thresholding/entropy ratios and yields all-zero structure features.

    Raises
    ------
    ImageContractError
        If ``image`` violates the shared image contract.
    FeatureDependencyError
        If scikit-image is not installed.
    FeatureError
        If a computed value is unexpectedly non-finite.
    """
    validate_image(image, name="feature image")

    intensity = image.astype(np.float64)
    intensity_mean = float(intensity.mean())
    intensity_std = float(intensity.std(ddof=0))

    if image.min() == image.max():
        features = {
            "intensity_mean": intensity_mean,
            "intensity_std": intensity_std,
            "contrast_rms": 0.0,
            "ink_fraction": 0.0,
            "laplacian_variance": 0.0,
            "edge_density": 0.0,
            "histogram_entropy": 0.0,
            "noise_sigma_immerkaer": 0.0,
        }
    else:
        features = {
            "intensity_mean": intensity_mean,
            "intensity_std": intensity_std,
            "contrast_rms": float(intensity_std / (intensity_mean + CONTRAST_EPSILON)),
            "ink_fraction": _ink_fraction(image),
            "laplacian_variance": _laplacian_variance(image),
            "edge_density": _edge_density(image),
            "histogram_entropy": _histogram_entropy(image),
            "noise_sigma_immerkaer": _noise_sigma_immerkaer(image),
        }

    non_finite = [name for name, value in features.items() if not math.isfinite(value)]
    if non_finite:
        raise FeatureError(f"non-finite feature(s) computed: {non_finite}")
    if tuple(features) != FEATURE_NAMES:  # pragma: no cover - structural guard
        raise FeatureError(f"feature order {tuple(features)} does not match {FEATURE_NAMES}")
    return {name: float(features[name]) for name in FEATURE_NAMES}


def extract_feature_table(
    samples: Iterable[tuple[str, np.ndarray]],
) -> pd.DataFrame:
    """Extract features for a batch of ``(sample_id, image)`` pairs.

    The returned frame has ``sample_id`` followed by :data:`FEATURE_NAMES`, so
    batching cannot change column order or definitions. Duplicate or empty
    sample identifiers are rejected.
    """
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for sample_id, image in samples:
        if not isinstance(sample_id, str) or not sample_id:
            raise FeatureError(f"sample_id must be a non-empty string, got {sample_id!r}")
        if sample_id in seen:
            raise FeatureError(f"duplicate sample_id in feature batch: {sample_id!r}")
        seen.add(sample_id)
        row: dict[str, object] = {"sample_id": sample_id}
        row.update(extract_features(image))
        rows.append(row)
    return pd.DataFrame(rows, columns=("sample_id", *FEATURE_NAMES))
