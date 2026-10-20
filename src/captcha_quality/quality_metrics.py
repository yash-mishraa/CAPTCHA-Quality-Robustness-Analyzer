"""Reference-based image-quality metrics used as candidate targets.

This module wraps established implementations from scikit-image rather than
re-implementing metric formulas. Documented conventions for the pilot:

* Images are 2-D ``uint8`` grayscale in ``[0, 255]``.
* ``data_range = 255.0`` for both metrics.
* SSIM uses the paper-style Gaussian window: ``win_size=11``,
  ``gaussian_weights=True``, ``sigma=1.5``, ``use_sample_covariance=False``
  (the configuration used by Wang et al.'s reference implementation).
* PSNR of two identical images is mathematically infinite. It is returned as
  ``math.inf`` and computed *before* calling scikit-image, which would otherwise
  divide by zero and emit a ``RuntimeWarning``.

These metrics measure similarity to a clean reference image. They are **not**
measurements of human readability or CAPTCHA solvability, and nothing in this
project treats them as such.

MS-SSIM is intentionally deferred; see :data:`DEFERRED_METRICS`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np

from captcha_quality.imaging import DATA_RANGE, validate_pair

#: SSIM window size in pixels (paper default).
SSIM_WINDOW: Final[int] = 11

#: Standard deviation of the Gaussian weighting window used by SSIM.
SSIM_GAUSSIAN_SIGMA: Final[float] = 1.5

#: Paper-style normalisation: no sample covariance correction.
SSIM_USE_SAMPLE_COVARIANCE: Final[bool] = False

SUPPORTED_METRICS: Final[tuple[str, ...]] = ("ssim", "psnr")

#: Metrics deliberately not evaluated in Milestone 1, with the reason recorded.
DEFERRED_METRICS: Final[dict[str, str]] = {
    "ms_ssim": (
        "No maintained, lightweight implementation is available in the installed "
        "stack: scikit-image 0.26 provides SSIM and PSNR but not MS-SSIM, and the "
        "maintained implementations (pytorch-msssim, torchmetrics) require PyTorch, "
        "which is a declared but uninstalled optional extra. Deferred until the "
        "deep-learning extra is installed for the CNN comparison."
    )
}


class MetricUnavailableError(RuntimeError):
    """Raised when an optional metric dependency is not installed."""


class UnknownMetricError(ValueError):
    """Raised when an unsupported or deferred metric is requested."""


def _import_metric_functions() -> tuple[Callable[..., float], Callable[..., float]]:
    """Import scikit-image's metric functions, with an actionable error.

    Imported lazily so that the core package (and its tests) works without the
    optional ``image`` extra, and so unavailability can be tested directly.
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise MetricUnavailableError(
            "reference metrics require scikit-image, which is not installed. "
            "Install the optional image extra with: uv sync --extra image"
        ) from exc
    return structural_similarity, peak_signal_noise_ratio


def ssim(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Structural similarity between two aligned images.

    Returns
    -------
    float
        SSIM in ``[-1, 1]``; identical images score ``1.0``.

    Raises
    ------
    ImageContractError
        If the images violate the contract or have different shapes.
    MetricUnavailableError
        If scikit-image is not installed.
    """
    validate_pair(reference, distorted)
    structural_similarity, _ = _import_metric_functions()
    value = structural_similarity(
        reference,
        distorted,
        data_range=DATA_RANGE,
        win_size=SSIM_WINDOW,
        gaussian_weights=True,
        sigma=SSIM_GAUSSIAN_SIGMA,
        use_sample_covariance=SSIM_USE_SAMPLE_COVARIANCE,
    )
    # Guard against tiny floating-point excursions beyond the theoretical range.
    return float(np.clip(value, -1.0, 1.0))


def psnr(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Peak signal-to-noise ratio between two aligned images, in decibels.

    Returns
    -------
    float
        PSNR in dB. Identical images return ``math.inf`` (the mathematically
        exact value), which callers must handle explicitly when aggregating.

    Raises
    ------
    ImageContractError
        If the images violate the contract or have different shapes.
    MetricUnavailableError
        If scikit-image is not installed.
    """
    validate_pair(reference, distorted)
    if np.array_equal(reference, distorted):
        return math.inf
    _, peak_signal_noise_ratio = _import_metric_functions()
    return float(peak_signal_noise_ratio(reference, distorted, data_range=DATA_RANGE))


_METRIC_FUNCTIONS: Final[dict[str, Callable[[np.ndarray, np.ndarray], float]]] = {
    "ssim": ssim,
    "psnr": psnr,
}


def compute_pair_metrics(
    reference: np.ndarray,
    distorted: np.ndarray,
    metrics: Sequence[str] = SUPPORTED_METRICS,
) -> dict[str, float]:
    """Compute several reference metrics for one aligned pair.

    Parameters
    ----------
    reference, distorted:
        Validated, aligned images. Validated once here, then passed to the
        individual metric functions.
    metrics:
        Metric names to compute.

    Returns
    -------
    dict of str to float
        Metric name to value. PSNR may be ``math.inf`` for identical images.

    Raises
    ------
    UnknownMetricError
        If a metric is unsupported, with the deferral reason when applicable.
    """
    requested = tuple(dict.fromkeys(metrics))
    for name in requested:
        if name in DEFERRED_METRICS:
            raise UnknownMetricError(f"metric {name!r} is deferred: {DEFERRED_METRICS[name]}")
        if name not in _METRIC_FUNCTIONS:
            raise UnknownMetricError(
                f"unknown metric {name!r}; supported metrics are {SUPPORTED_METRICS}"
            )

    validate_pair(reference, distorted)
    return {name: _METRIC_FUNCTIONS[name](reference, distorted) for name in requested}


def describe_metrics() -> dict[str, str]:
    """Return a short human-readable description of each supported metric."""
    return {
        "ssim": (
            "Structural similarity vs. the clean reference (data_range=255, "
            f"win_size={SSIM_WINDOW}, gaussian sigma={SSIM_GAUSSIAN_SIGMA}, "
            "no sample-covariance correction). Higher is better."
        ),
        "psnr": (
            "Peak signal-to-noise ratio in dB vs. the clean reference "
            "(data_range=255). Higher is better; identical images are +inf."
        ),
    }
