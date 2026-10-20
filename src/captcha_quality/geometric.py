"""Bounded smooth elastic-warp corruption for the M5A feasibility pilot.

The transformation is a single, deliberately small family: a smooth elastic
displacement field applied to a clean image with bilinear resampling. The field
is generated from an explicit seed, a coarse random control grid, and a
separable Gaussian smoother. Its maximum displacement is normalised to one
pixel, so a severity amplitude is exactly the maximum displacement applied to
the image.

The module has three properties the pilot relies on:

* every severity and control for a source is generated directly from the same
  frozen clean reference;
* the field, not just the RNG seed, is deterministic and hashable, so a stored
  field specification can replay the exact decoded pixels;
* the transform is bounded: callers supply a maximum amplitude, validated
  against a conservative fraction of the image's smaller side.

Only bilinear interpolation and two explicit border policies are implemented.
That is intentional. A large family of geometric transforms would make the
pilot's attribution problem worse, not better.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from captcha_quality.imaging import VALUE_MAX, VALUE_MIN, copy_image, validate_image
from captcha_quality.provenance import canonical_json

INTERPOLATION_BILINEAR: Final[str] = "bilinear"
BORDER_EDGE: Final[str] = "edge"
BORDER_REFLECT: Final[str] = "reflect"

SUPPORTED_INTERPOLATIONS: Final[tuple[str, ...]] = (INTERPOLATION_BILINEAR,)
SUPPORTED_BORDER_MODES: Final[tuple[str, ...]] = (BORDER_EDGE, BORDER_REFLECT)

#: Defaults chosen before looking at any pilot targets. A 5 x 13 control grid
#: on a 64 x 192 canvas gives roughly 16 px horizontal and vertical spacing;
#: sigma 2 px smooths the piecewise-linear field without erasing its shape.
DEFAULT_GRID_ROWS: Final[int] = 5
DEFAULT_GRID_COLS: Final[int] = 13
DEFAULT_SMOOTHING_SIGMA_PX: Final[float] = 2.0

#: Predeclared amplitude ladder in pixels. These are the maximum displacement
#: of a unit-normalised field, not an average displacement. On a 64 px-tall
#: canvas the largest value is 3 px, under 5% of the image height.
DEFAULT_AMPLITUDES_PX: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 3.0)

#: Smallest non-zero displacement used to isolate interpolation/border effects.
DEFAULT_MINIMAL_AMPLITUDE_PX: Final[float] = 0.25

#: A displacement larger than this fraction of the smaller image side is
#: rejected rather than silently producing an uninterpretable image.
MAX_AMPLITUDE_FRACTION: Final[float] = 0.10

#: No single smoothing kernel should span more than this many pixels.
MAX_SMOOTHING_SIGMA_PX: Final[float] = 16.0


class GeometricError(ValueError):
    """Raised when a geometric field, amplitude, or image is invalid."""


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GeometricError(f"seed must be an integer, got {seed!r}")
    if not 0 <= seed < 2**64:
        raise GeometricError(f"seed must be in [0, 2**64), got {seed}")


def _validate_positive_int(value: int, label: str, *, minimum: int = 2) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeometricError(f"{label} must be an integer, got {value!r}")
    if value < minimum:
        raise GeometricError(f"{label} must be >= {minimum}, got {value}")


@dataclass(frozen=True, slots=True)
class ElasticFieldSpec:
    """Deterministic specification of one smooth displacement field.

    Attributes
    ----------
    height, width:
        Image canvas on which the field is defined.
    seed:
        Explicit 64-bit seed for the coarse control grid.
    grid_rows, grid_cols:
        Number of control points including both boundary points.
    smoothing_sigma_px:
        Standard deviation of the separable Gaussian smoother. ``0`` disables
        smoothing, leaving the bilinear control-grid interpolation.
    interpolation:
        Only ``"bilinear"`` is supported.
    border_mode:
        ``"edge"`` clamps out-of-bounds source coordinates to the nearest valid
        sample. ``"reflect"`` reflects fractional coordinates at the image
        boundary. Both are recorded and replayed explicitly.
    """

    height: int
    width: int
    seed: int
    grid_rows: int = DEFAULT_GRID_ROWS
    grid_cols: int = DEFAULT_GRID_COLS
    smoothing_sigma_px: float = DEFAULT_SMOOTHING_SIGMA_PX
    interpolation: str = INTERPOLATION_BILINEAR
    border_mode: str = BORDER_EDGE

    def __post_init__(self) -> None:
        _validate_positive_int(self.height, "height")
        _validate_positive_int(self.width, "width")
        _validate_seed(self.seed)
        _validate_positive_int(self.grid_rows, "grid_rows")
        _validate_positive_int(self.grid_cols, "grid_cols")
        if self.grid_rows > self.height:
            raise GeometricError(
                f"grid_rows must be <= height, got {self.grid_rows} for height {self.height}"
            )
        if self.grid_cols > self.width:
            raise GeometricError(
                f"grid_cols must be <= width, got {self.grid_cols} for width {self.width}"
            )
        if (
            isinstance(self.smoothing_sigma_px, bool)
            or not isinstance(self.smoothing_sigma_px, int | float)
            or not math.isfinite(float(self.smoothing_sigma_px))
            or not 0.0 <= float(self.smoothing_sigma_px) <= MAX_SMOOTHING_SIGMA_PX
        ):
            raise GeometricError(
                "smoothing_sigma_px must be finite and in "
                f"[0, {MAX_SMOOTHING_SIGMA_PX}], got {self.smoothing_sigma_px!r}"
            )
        if self.interpolation not in SUPPORTED_INTERPOLATIONS:
            raise GeometricError(
                f"interpolation must be one of {SUPPORTED_INTERPOLATIONS}, "
                f"got {self.interpolation!r}"
            )
        if self.border_mode not in SUPPORTED_BORDER_MODES:
            raise GeometricError(
                f"border_mode must be one of {SUPPORTED_BORDER_MODES}, got {self.border_mode!r}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Field shape as ``(height, width)``."""
        return (self.height, self.width)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-safe replay specification (without the field hash)."""
        return {
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "smoothing_sigma_px": float(self.smoothing_sigma_px),
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
        }

    def with_field_hash(self, field: np.ndarray) -> dict[str, object]:
        """Return the replay specification plus the exact field hash."""
        payload = self.as_dict()
        payload["field_sha256"] = _sha256_array(field)
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> ElasticFieldSpec:
        """Parse a strict replay specification."""
        if not isinstance(payload, dict):
            raise GeometricError("field specification must be a JSON object")
        allowed = {
            "height",
            "width",
            "seed",
            "grid_rows",
            "grid_cols",
            "smoothing_sigma_px",
            "interpolation",
            "border_mode",
            "field_sha256",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise GeometricError(f"unknown field-specification key(s): {unknown}")
        required = allowed - {"field_sha256"}
        missing = sorted(required - set(payload))
        if missing:
            raise GeometricError(f"missing field-specification key(s): {missing}")
        try:
            return cls(
                height=payload["height"],  # type: ignore[arg-type]
                width=payload["width"],  # type: ignore[arg-type]
                seed=payload["seed"],  # type: ignore[arg-type]
                grid_rows=payload["grid_rows"],  # type: ignore[arg-type]
                grid_cols=payload["grid_cols"],  # type: ignore[arg-type]
                smoothing_sigma_px=payload["smoothing_sigma_px"],  # type: ignore[arg-type]
                interpolation=payload["interpolation"],  # type: ignore[arg-type]
                border_mode=payload["border_mode"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError) as exc:  # pragma: no cover - defensive
            raise GeometricError(f"invalid field specification: {exc}") from exc


def _gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _smooth_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if kernel.size == 1:
        return values
    pad = kernel.size // 2
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(values, pad_width, mode="reflect")
    result = np.zeros_like(values, dtype=np.float64)
    for index, weight in enumerate(kernel):
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(index, index + values.shape[axis])
        result += float(weight) * padded[tuple(slices)]
    return result


def _smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma)
    smoothed = _smooth_axis(field, kernel, axis=0)
    smoothed = _smooth_axis(smoothed, kernel, axis=1)
    return smoothed


def _interpolate_control_grid(coarse: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinearly expand a ``(grid_rows, grid_cols, 2)`` control grid."""
    grid_rows, grid_cols, channels = coarse.shape
    rows = np.linspace(0.0, height - 1.0, grid_rows)
    cols = np.linspace(0.0, width - 1.0, grid_cols)
    full_rows = np.arange(height, dtype=np.float64)
    full_cols = np.arange(width, dtype=np.float64)

    vertical = np.empty((height, grid_cols, channels), dtype=np.float64)
    for column in range(grid_cols):
        for channel in range(channels):
            vertical[:, column, channel] = np.interp(full_rows, rows, coarse[:, column, channel])

    field = np.empty((height, width, channels), dtype=np.float64)
    for channel in range(channels):
        for row in range(height):
            field[row, :, channel] = np.interp(full_cols, cols, vertical[row, :, channel])
    return field


def generate_displacement_field(spec: ElasticFieldSpec) -> np.ndarray:
    """Generate the unit-maximum ``(2, height, width)`` displacement field.

    Channel 0 is the vertical displacement and channel 1 is horizontal. The
    field is smoothed, centred to zero mean (no global translation), and
    normalised so its largest vector magnitude is exactly one. Multiplication
    by an amplitude in pixels therefore gives a bounded maximum displacement.
    """
    rng = np.random.default_rng(spec.seed)
    coarse = rng.normal(loc=0.0, scale=1.0, size=(spec.grid_rows, spec.grid_cols, 2))
    field = _interpolate_control_grid(coarse, spec.height, spec.width)
    field = _smooth_field(field, float(spec.smoothing_sigma_px))
    field -= field.mean(axis=(0, 1), keepdims=True)
    norms = np.sqrt(np.sum(field**2, axis=2))
    maximum = float(norms.max())
    if not math.isfinite(maximum) or maximum <= 0.0:  # pragma: no cover - random guard
        raise GeometricError("generated displacement field is degenerate")
    field /= maximum
    result = np.transpose(field, (2, 0, 1)).astype(np.float64, copy=False)
    if not np.isfinite(result).all():  # pragma: no cover - numerical guard
        raise GeometricError("generated displacement field contains non-finite values")
    return result


def field_sha256(field: np.ndarray) -> str:
    """Exact SHA-256 of a displacement field's decoded float64 bytes."""
    if not isinstance(field, np.ndarray) or field.ndim != 3 or field.shape[0] != 2:
        raise GeometricError("field must have shape (2, height, width)")
    if field.dtype != np.float64:
        raise GeometricError(f"field must have dtype float64, got {field.dtype}")
    if not np.isfinite(field).all():
        raise GeometricError("field contains non-finite values")
    return _sha256_array(np.ascontiguousarray(field))


def _reflect_coordinates(values: np.ndarray, size: int) -> np.ndarray:
    if size == 1:
        return np.zeros_like(values)
    period = 2 * size - 2
    wrapped = np.mod(values, period)
    return np.where(wrapped < size, wrapped, period - wrapped)


def _border_coordinates(values: np.ndarray, size: int, mode: str) -> np.ndarray:
    if mode == BORDER_EDGE:
        return np.clip(values, 0.0, size - 1.0)
    if mode == BORDER_REFLECT:
        return _reflect_coordinates(values, size)
    raise GeometricError(f"unsupported border mode {mode!r}")  # pragma: no cover


@dataclass(frozen=True, slots=True)
class ElasticResult:
    """A warped image plus the diagnostics needed to interpret its target."""

    image: np.ndarray
    field_spec: ElasticFieldSpec
    amplitude_px: float
    resampled: bool
    field: np.ndarray
    outside_fraction: float
    border_affected_fraction: float
    changed_pixel_fraction: float
    changed_ink_fraction: float
    changed_background_fraction: float
    border_touched: bool
    clipped_fraction: float
    intensity_std: float
    collapsed_block_fraction: float
    degenerate: bool

    def __post_init__(self) -> None:
        validate_image(self.image, name="warped image")
        if not math.isfinite(self.amplitude_px) or self.amplitude_px < 0.0:
            raise GeometricError(f"amplitude_px must be finite and >= 0, got {self.amplitude_px!r}")
        for value, label in (
            (self.outside_fraction, "outside_fraction"),
            (self.border_affected_fraction, "border_affected_fraction"),
            (self.changed_pixel_fraction, "changed_pixel_fraction"),
            (self.changed_ink_fraction, "changed_ink_fraction"),
            (self.changed_background_fraction, "changed_background_fraction"),
            (self.clipped_fraction, "clipped_fraction"),
            (self.collapsed_block_fraction, "collapsed_block_fraction"),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise GeometricError(f"{label} must be finite and in [0, 1], got {value!r}")
        if not math.isfinite(self.intensity_std) or self.intensity_std < 0.0:
            raise GeometricError("intensity_std must be finite and >= 0")


def _validate_amplitude(amplitude_px: float, shape: tuple[int, int]) -> float:
    if isinstance(amplitude_px, bool) or not isinstance(amplitude_px, int | float):
        raise GeometricError(f"amplitude_px must be a number, got {amplitude_px!r}")
    amplitude = float(amplitude_px)
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise GeometricError(f"amplitude_px must be finite and >= 0, got {amplitude!r}")
    maximum = MAX_AMPLITUDE_FRACTION * min(shape)
    if amplitude > maximum:
        raise GeometricError(
            f"amplitude_px={amplitude} exceeds the safe bound {maximum:.2f} "
            f"({MAX_AMPLITUDE_FRACTION:.0%} of the smaller image side)"
        )
    return amplitude


def _block_std(image: np.ndarray, block: int = 8) -> np.ndarray:
    height, width = image.shape
    rows = height // block
    columns = width // block
    if rows == 0 or columns == 0:
        return np.array([float(image.std(ddof=0))], dtype=np.float64)
    cropped = image[: rows * block, : columns * block].astype(np.float64)
    reshaped = cropped.reshape(rows, block, columns, block).transpose(0, 2, 1, 3)
    return reshaped.reshape(rows, columns, block * block).std(axis=2, ddof=0)


def _collapsed_block_fraction(clean: np.ndarray, warped: np.ndarray) -> float:
    """Fraction of non-blank clean blocks that become nearly blank."""
    clean_std = _block_std(clean)
    warped_std = _block_std(warped)
    active = clean_std >= 1.0
    if not np.any(active):
        return 0.0
    return float(np.mean((warped_std < 1.0)[active]))


def _warp(
    image: np.ndarray,
    field: np.ndarray,
    amplitude_px: float,
    spec: ElasticFieldSpec,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    height, width = image.shape
    rows, columns = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    displacement_y = field[0] * amplitude_px
    displacement_x = field[1] * amplitude_px
    source_y = rows + displacement_y
    source_x = columns + displacement_x

    outside = (
        (source_y < 0.0) | (source_y > height - 1.0) | (source_x < 0.0) | (source_x > width - 1.0)
    )
    outside_fraction = float(np.mean(outside))

    safe_y = _border_coordinates(source_y, height, spec.border_mode)
    safe_x = _border_coordinates(source_x, width, spec.border_mode)
    y0 = np.floor(safe_y).astype(np.int64)
    x0 = np.floor(safe_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = safe_y - y0
    wx = safe_x - x0

    # Every sample whose mapped coordinate falls outside the canvas needs a
    # border policy. This is the fraction of pixels affected by clamping or
    # reflection before interpolation.
    border_affected_fraction = float(np.mean(outside))

    values = image.astype(np.float64)
    raw = (
        values[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + values[y0, x1] * (1.0 - wy) * wx
        + values[y1, x0] * wy * (1.0 - wx)
        + values[y1, x1] * wy * wx
    )
    clipped_fraction = float(np.mean((raw < VALUE_MIN) | (raw > VALUE_MAX)))
    warped = np.rint(np.clip(raw, VALUE_MIN, VALUE_MAX)).astype(np.uint8)
    if warped.shape != image.shape:  # pragma: no cover - structural guard
        raise GeometricError("warp changed the image shape")
    if not np.isfinite(raw).all():  # pragma: no cover - numerical guard
        raise GeometricError("warp produced non-finite samples")
    changed = warped != image
    ink = image < 128
    ink_count = int(np.count_nonzero(ink))
    background_count = int(image.size - ink_count)
    border_touched = bool(
        np.any(changed[0, :])
        or np.any(changed[-1, :])
        or np.any(changed[:, 0])
        or np.any(changed[:, -1])
    )
    return warped, {
        "outside_fraction": outside_fraction,
        "border_affected_fraction": border_affected_fraction,
        "changed_pixel_fraction": float(np.count_nonzero(changed)) / float(image.size),
        "changed_ink_fraction": (
            float(np.count_nonzero(changed & ink)) / ink_count if ink_count else 0.0
        ),
        "changed_background_fraction": (
            float(np.count_nonzero(changed & ~ink)) / background_count if background_count else 0.0
        ),
        "border_touched": border_touched,
        "clipped_fraction": clipped_fraction,
    }


def apply_elastic(
    image: np.ndarray,
    *,
    field_spec: ElasticFieldSpec,
    amplitude_px: float,
    force_resample: bool = False,
) -> ElasticResult:
    """Apply a bounded elastic warp to a clean image.

    Parameters
    ----------
    image:
        Clean 2-D ``uint8`` image satisfying the project contract.
    field_spec:
        Deterministic field specification. Its canvas must match ``image``.
    amplitude_px:
        Maximum displacement in pixels. ``0`` returns an exact copy unless
        ``force_resample`` is true, which routes the identity through the
        bilinear sampler as an interpolation control.
    force_resample:
        Request the sampler even for zero displacement. For a zero field this
        is mathematically identical to the clean image and is used to test the
        resampling path.

    Returns
    -------
    ElasticResult
        Warped image plus field, changed-pixel, border and degeneracy
        diagnostics.

    Raises
    ------
    ImageContractError
        If ``image`` violates the shared image contract.
    GeometricError
        If the field does not match the image, the amplitude is invalid, or a
        generated field is non-finite.
    """
    validate_image(image)
    if field_spec.shape != image.shape:
        raise GeometricError(
            f"field shape {field_spec.shape} does not match image shape {image.shape}"
        )
    amplitude = _validate_amplitude(amplitude_px, image.shape)
    field = generate_displacement_field(field_spec)

    if amplitude == 0.0 and not force_resample:
        copied = copy_image(image)
        return ElasticResult(
            image=copied,
            field_spec=field_spec,
            amplitude_px=0.0,
            resampled=False,
            field=field,
            outside_fraction=0.0,
            border_affected_fraction=0.0,
            changed_pixel_fraction=0.0,
            changed_ink_fraction=0.0,
            changed_background_fraction=0.0,
            border_touched=False,
            clipped_fraction=0.0,
            intensity_std=float(copied.astype(np.float64).std(ddof=0)),
            collapsed_block_fraction=0.0,
            degenerate=False,
        )

    warped, diagnostics = _warp(image, field, amplitude, field_spec)
    intensity_std = float(warped.astype(np.float64).std(ddof=0))
    collapsed = _collapsed_block_fraction(image, warped)
    degenerate = bool(intensity_std < 1.0 or collapsed > 0.5)
    if not np.isfinite(warped.astype(np.float64)).all():  # pragma: no cover - dtype guard
        raise GeometricError("warp produced non-finite pixel values")
    return ElasticResult(
        image=warped,
        field_spec=field_spec,
        amplitude_px=amplitude,
        resampled=True,
        field=field,
        outside_fraction=float(diagnostics["outside_fraction"]),
        border_affected_fraction=float(diagnostics["border_affected_fraction"]),
        changed_pixel_fraction=float(diagnostics["changed_pixel_fraction"]),
        changed_ink_fraction=float(diagnostics["changed_ink_fraction"]),
        changed_background_fraction=float(diagnostics["changed_background_fraction"]),
        border_touched=bool(diagnostics["border_touched"]),
        clipped_fraction=float(diagnostics["clipped_fraction"]),
        intensity_std=intensity_std,
        collapsed_block_fraction=collapsed,
        degenerate=degenerate,
    )


@dataclass(frozen=True, slots=True)
class ElasticLadder:
    """Predeclared amplitude ladder for the pilot's geometric family."""

    amplitudes_px: tuple[float, ...] = DEFAULT_AMPLITUDES_PX
    minimal_amplitude_px: float = DEFAULT_MINIMAL_AMPLITUDE_PX

    def __post_init__(self) -> None:
        if not self.amplitudes_px:
            raise GeometricError("amplitude ladder must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in self.amplitudes_px
        ):
            raise GeometricError(f"amplitudes_px must be numbers, got {self.amplitudes_px}")
        values = tuple(float(value) for value in self.amplitudes_px)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise GeometricError(f"amplitudes_px must be finite and > 0, got {values}")
        for earlier, later in zip(values, values[1:], strict=False):
            if later <= earlier:
                raise GeometricError(f"amplitudes_px must be strictly increasing, got {values}")
        if (
            isinstance(self.minimal_amplitude_px, bool)
            or not isinstance(self.minimal_amplitude_px, int | float)
            or not math.isfinite(float(self.minimal_amplitude_px))
            or float(self.minimal_amplitude_px) <= 0.0
        ):
            raise GeometricError(
                f"minimal_amplitude_px must be finite and > 0, got {self.minimal_amplitude_px!r}"
            )
        if float(self.minimal_amplitude_px) >= values[0]:
            raise GeometricError(
                "minimal_amplitude_px must be smaller than the first ladder amplitude, "
                f"got {self.minimal_amplitude_px} >= {values[0]}"
            )

    @property
    def n_levels(self) -> int:
        """Number of corrupted severity levels (severity is 1-based)."""
        return len(self.amplitudes_px)

    def amplitude(self, severity: int) -> float:
        """Return the amplitude for a 1-based severity."""
        if not 1 <= severity <= self.n_levels:
            raise GeometricError(f"severity {severity} is out of range for {self.n_levels} levels")
        return float(self.amplitudes_px[severity - 1])


def field_spec_json(spec: ElasticFieldSpec, field: np.ndarray) -> str:
    """Canonical JSON for a field specification and exact field hash."""
    return canonical_json(spec.with_field_hash(field))


def parse_field_spec(payload: object) -> tuple[ElasticFieldSpec, str]:
    """Parse a recorded field specification and its expected field hash."""
    if not isinstance(payload, dict):
        raise GeometricError("field specification must be a JSON object")
    expected = payload.get("field_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise GeometricError("field specification must contain a 64-character field_sha256")
    spec = ElasticFieldSpec.from_dict(payload)
    return spec, expected


def verify_field_replay(
    image: np.ndarray,
    payload: object,
    *,
    amplitude_px: float,
    force_resample: bool,
) -> ElasticResult:
    """Regenerate a field from its specification and check its exact hash."""
    spec, expected = parse_field_spec(payload)
    field = generate_displacement_field(spec)
    actual = field_sha256(field)
    if actual != expected:
        raise GeometricError(f"field replay hash mismatch: expected {expected}, got {actual}")
    result = apply_elastic(
        image,
        field_spec=spec,
        amplitude_px=amplitude_px,
        force_resample=force_resample,
    )
    if field_sha256(result.field) != expected:
        raise GeometricError("replayed field hash does not match the stored specification")
    return result


def sha256_field(field: np.ndarray) -> str:
    """Public alias for the exact field hash used by provenance records."""
    return field_sha256(field)
