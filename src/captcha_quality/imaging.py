"""Shared image-array contract for the pilot.

Every image handled by this project is a 2-D ``uint8`` grayscale array with
shape ``(height, width)`` and values in ``[0, 255]``. Enforcing one contract in
one place keeps the renderer, the corruption module, and the reference metrics
comparable.

The contract exists because full-reference metrics are only meaningful when the
reference and the distorted image have identical shape, dtype, and value range,
and because SSIM's default 11x11 window requires a minimum image size.
"""

from __future__ import annotations

import numpy as np

#: Inclusive valid range for ``uint8`` samples.
VALUE_MIN = 0
VALUE_MAX = 255

#: Value range passed to image-quality metrics (``uint8`` -> 255).
DATA_RANGE = 255.0

#: SSIM's default window is 11x11, so each side must be at least this large.
MIN_SIDE = 11


class ImageContractError(ValueError):
    """Raised when an array does not satisfy the project image contract."""


def validate_image(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    """Return ``image`` unchanged, or raise if it violates the contract.

    Parameters
    ----------
    image:
        Candidate array.
    name:
        Label used in error messages to identify the offending argument.

    Raises
    ------
    ImageContractError
        If the array is not a 2-D ``uint8`` array of at least ``MIN_SIDE`` by
        ``MIN_SIDE`` pixels.
    """
    if not isinstance(image, np.ndarray):
        raise ImageContractError(f"{name} must be a numpy array, got {type(image).__name__}")
    if image.ndim != 2:
        raise ImageContractError(
            f"{name} must be a 2-D grayscale array (height, width), got {image.ndim} dimension(s)"
        )
    if image.dtype != np.uint8:
        raise ImageContractError(
            f"{name} must have dtype uint8 in [0, 255], got dtype {image.dtype}"
        )
    if min(image.shape) < MIN_SIDE:
        raise ImageContractError(
            f"{name} must be at least {MIN_SIDE}x{MIN_SIDE} pixels "
            f"(SSIM window size), got shape {image.shape}"
        )
    return image


def validate_pair(
    reference: np.ndarray,
    distorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a reference/distorted pair and their pixel alignment.

    Returns
    -------
    tuple of numpy.ndarray
        The two validated arrays.

    Raises
    ------
    ImageContractError
        If either array violates :func:`validate_image`, or if the two shapes
        differ (a misaligned pair makes full-reference metrics meaningless).
    """
    validate_image(reference, name="reference")
    validate_image(distorted, name="distorted")
    if reference.shape != distorted.shape:
        raise ImageContractError(
            "reference and distorted images must have identical shapes for "
            f"full-reference metrics, got {reference.shape} and {distorted.shape}"
        )
    return reference, distorted


def copy_image(image: np.ndarray) -> np.ndarray:
    """Return a writable, independent copy of a validated image."""
    validate_image(image)
    return np.array(image, dtype=np.uint8, copy=True)


def freeze_image(image: np.ndarray) -> np.ndarray:
    """Return the same array marked read-only.

    Used for clean references so that an accidental in-place corruption raises
    instead of silently mutating the reference used by every severity level.
    """
    validate_image(image)
    image.setflags(write=False)
    return image


def as_float(image: np.ndarray) -> np.ndarray:
    """Return a validated image converted to ``float64`` without rescaling."""
    validate_image(image)
    return image.astype(np.float64)
