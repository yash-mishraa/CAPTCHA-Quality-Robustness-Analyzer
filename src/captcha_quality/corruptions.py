"""Controlled corruption families for the target-validation pilot.

Only two families are implemented, because the pilot's job is to test whether
metrics respond predictably to severity — not to build the Milestone 2
distortion engine. Both families are per-pixel operations on an existing image,
which guarantees that the degraded image has exactly the same shape as the
clean reference and stays pixel-aligned with it.

Severity is expressed as an ordered ladder of *explicit* numeric parameters:

* ``gaussian_blur``: Pillow's Gaussian blur radius in pixels.
* ``gaussian_noise``: standard deviation of additive Gaussian noise on the
  0-255 intensity scale.

Parameter index 0 of every ladder is the identity (``0.0``), which is the
clean/no-corruption baseline condition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image, ImageFilter

from captcha_quality.imaging import VALUE_MAX, VALUE_MIN, copy_image, validate_image

FAMILY_BLUR: Final[str] = "gaussian_blur"
FAMILY_NOISE: Final[str] = "gaussian_noise"

#: Blur draws no random numbers, so its results are fully determined by the input.
DETERMINISTIC_FAMILIES: Final[frozenset[str]] = frozenset({FAMILY_BLUR})

#: Noise needs an explicit RNG seed per severity level.
STOCHASTIC_FAMILIES: Final[frozenset[str]] = frozenset({FAMILY_NOISE})

SUPPORTED_FAMILIES: Final[tuple[str, ...]] = (FAMILY_BLUR, FAMILY_NOISE)

#: Blur radii per severity level. Chosen to bracket the useful range for a
#: 64 px-tall glyph canvas: 1.0 px is visible but mild, 5.0 px is close to
#: illegible for five characters at this size.
DEFAULT_BLUR_RADII: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 3.5, 5.0)

#: Noise standard deviations per severity level on the 0-255 intensity scale.
#: The ladder is roughly geometric, which spreads PSNR out by roughly 4-6 dB
#: per step while keeping the highest level inside the 8-bit range.
DEFAULT_NOISE_SIGMAS: Final[tuple[float, ...]] = (0.0, 6.0, 12.0, 20.0, 32.0)


class CorruptionError(ValueError):
    """Raised when a corruption specification or ladder is invalid."""


@dataclass(frozen=True, slots=True)
class SeverityLadder:
    """An ordered ladder of corruption parameters for one family.

    Attributes
    ----------
    family:
        One of :data:`SUPPORTED_FAMILIES`.
    parameters:
        Non-decreasing parameters, one per severity level. ``parameters[0]``
        must be ``0.0`` so that severity 0 is the clean baseline condition.
    unit:
        Physical unit of the parameters, recorded for reporting.
    """

    family: str
    parameters: tuple[float, ...]
    unit: str

    def __post_init__(self) -> None:
        if self.family not in SUPPORTED_FAMILIES:
            raise CorruptionError(
                f"unknown corruption family {self.family!r}; supported: {SUPPORTED_FAMILIES}"
            )
        if len(self.parameters) < 2:
            raise CorruptionError(
                f"ladder for {self.family!r} needs at least 2 levels (clean + 1), "
                f"got {len(self.parameters)}"
            )
        if any(parameter < 0.0 for parameter in self.parameters):
            raise CorruptionError(f"ladder parameters must be >= 0, got {self.parameters}")
        if self.parameters[0] != 0.0:
            raise CorruptionError(
                f"severity 0 must be the identity baseline (0.0), got {self.parameters[0]}"
            )
        for earlier, later in zip(self.parameters, self.parameters[1:], strict=False):
            if later < earlier:
                raise CorruptionError(
                    f"ladder parameters must be non-decreasing with severity, got {self.parameters}"
                )

    @property
    def n_levels(self) -> int:
        """Number of severity levels, including the clean baseline."""
        return len(self.parameters)

    def parameter(self, severity: int) -> float:
        """Return the corruption parameter for ``severity``.

        Raises
        ------
        CorruptionError
            If ``severity`` is outside ``[0, n_levels)``.
        """
        if not 0 <= severity < self.n_levels:
            raise CorruptionError(
                f"severity {severity} is out of range for a ladder with {self.n_levels} levels"
            )
        return self.parameters[severity]

    def spec(self, severity: int) -> CorruptionSpec:
        """Return an explicit :class:`CorruptionSpec` for ``severity``."""
        return CorruptionSpec(
            family=self.family,
            severity=severity,
            parameter=self.parameter(severity),
            unit=self.unit,
        )


@dataclass(frozen=True, slots=True)
class CorruptionSpec:
    """A fully explicit corruption request."""

    family: str
    severity: int
    parameter: float
    unit: str

    @property
    def is_identity(self) -> bool:
        """True when this spec leaves the image unchanged."""
        return self.parameter == 0.0

    @property
    def label(self) -> str:
        """Compact label used in records and reports."""
        return f"{self.family}[{self.severity}]={self.parameter:g}{self.unit}"


@dataclass(frozen=True, slots=True)
class CorruptionResult:
    """A corrupted image plus diagnostics about how it was produced."""

    image: np.ndarray
    spec: CorruptionSpec
    #: Fraction of pixels whose additive noise draw was clipped at 0 or 255.
    #: Only meaningful for the additive-noise family; ``0.0`` for blur.
    clipped_fraction: float = 0.0
    #: Standard deviation of the *realised* change (corrupted minus clean),
    #: measured after clipping. This shows how 8-bit clipping compresses the
    #: effective noise level at high severity. ``0.0`` for blur.
    realised_noise_std: float = 0.0

    def __post_init__(self) -> None:
        validate_image(self.image, name="corrupted image")
        if not 0.0 <= self.clipped_fraction <= 1.0:
            raise CorruptionError(
                f"clipped_fraction must be in [0, 1], got {self.clipped_fraction}"
            )
        if self.realised_noise_std < 0.0:
            raise CorruptionError(f"realised_noise_std must be >= 0, got {self.realised_noise_std}")


def default_ladders() -> dict[str, SeverityLadder]:
    """Return the pilot's default ladders (mirrored by the pilot config file)."""
    return {
        FAMILY_BLUR: SeverityLadder(FAMILY_BLUR, DEFAULT_BLUR_RADII, "px"),
        FAMILY_NOISE: SeverityLadder(FAMILY_NOISE, DEFAULT_NOISE_SIGMAS, "sigma"),
    }


def derive_noise_seed(*parts: object) -> int:
    """Derive a stable 64-bit seed from arbitrary identifying parts.

    Each severity level uses an *independent* noise realisation derived from the
    trial identity, so severity levels are not artificially correlated by a
    shared random field. This mirrors how the Milestone 2 dataset will sample
    separate corrupted images.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def apply_corruption(
    image: np.ndarray,
    ladder: SeverityLadder,
    severity: int,
    *,
    noise_seed: int | None = None,
) -> CorruptionResult:
    """Apply one severity level of one family and return a new image.

    The input image is never modified: the function validates it, works on a
    copy, and returns a fresh ``uint8`` array with the same shape.

    Parameters
    ----------
    image:
        Clean image satisfying the project image contract.
    ladder:
        Severity ladder for the family.
    severity:
        Level index into ``ladder``.
    noise_seed:
        Required for stochastic families; ignored for deterministic ones.

    Returns
    -------
    CorruptionResult
        The corrupted image (dtype ``uint8``, values clipped to ``[0, 255]``)
        plus the specification used and, for noise, the clipped-pixel fraction.

    Raises
    ------
    CorruptionError
        If the severity is out of range, or a stochastic family is requested
        without ``noise_seed``.
    """
    validate_image(image)
    spec = ladder.spec(severity)

    if spec.is_identity:
        return CorruptionResult(image=copy_image(image), spec=spec)

    if ladder.family == FAMILY_BLUR:
        blurred = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=spec.parameter))
        return CorruptionResult(image=np.array(blurred, dtype=np.uint8).copy(), spec=spec)

    if ladder.family == FAMILY_NOISE:
        if noise_seed is None:
            raise CorruptionError(
                f"family {FAMILY_NOISE!r} is stochastic and requires an explicit noise_seed"
            )
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(loc=0.0, scale=spec.parameter, size=image.shape)
        raw = image.astype(np.float64) + noise
        clipped_fraction = float(np.mean((raw < VALUE_MIN) | (raw > VALUE_MAX)))
        noisy = np.rint(np.clip(raw, VALUE_MIN, VALUE_MAX)).astype(np.uint8)
        realised_noise_std = float(np.std(noisy.astype(np.float64) - image.astype(np.float64)))
        return CorruptionResult(
            image=noisy,
            spec=spec,
            clipped_fraction=clipped_fraction,
            realised_noise_std=realised_noise_std,
        )

    raise CorruptionError(f"unsupported corruption family {ladder.family!r}")


def max_absolute_deviation(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Largest absolute per-pixel difference between two aligned images."""
    validate_image(image_a, name="image_a")
    validate_image(image_b, name="image_b")
    if image_a.shape != image_b.shape:
        raise CorruptionError(
            f"images must be aligned to compare, got {image_a.shape} and {image_b.shape}"
        )
    return float(np.max(np.abs(image_a.astype(np.float64) - image_b.astype(np.float64))))
