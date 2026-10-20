"""Controlled occlusion and visual-interference corruption for the M4A pilot.

Two variants are implemented, both drawn directly onto a copy of a clean
reference image:

``thin_lines``
    One-pixel dark lines that cross the text region. Each line passes through a
    seeded pivot inside the reference's ink bounding box, then extends to the
    canvas boundary in both directions.
``arc_strokes``
    Curved elliptical strokes that pass through a seeded pivot inside the ink
    bounding box. The centre, radii and angular sweep are seeded.

The two variants share a *stroke pool* per ``(image, variant, seed)``. Severity
``k`` uses the first ``k`` strokes of that pool, and every severity is rendered
from the same clean reference. This nested design has two properties the pilot
relies on:

* the transformation is a genuine dose: a higher severity adds strokes rather
  than replacing the previous geometry;
* every stroke is drawn against the same clean reference, so severity is a
  density dose rather than a replacement of earlier geometry. SSIM usually
  decreases as strokes are added, but ties are possible when a stroke lies
  entirely on existing dark ink, and local metric exceptions are checked in the
  pilot rather than assumed away.

The module does not use SSIM, PSNR, model predictions or any metadata to choose
geometry. It sees only the clean image and an explicit seed. It also does not
modify its input; the clean reference is copied before drawing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image, ImageDraw

from captcha_quality.imaging import copy_image, validate_image

VARIANT_THIN_LINES: Final[str] = "thin_lines"
VARIANT_ARC_STROKES: Final[str] = "arc_strokes"
SUPPORTED_VARIANTS: Final[tuple[str, ...]] = (VARIANT_THIN_LINES, VARIANT_ARC_STROKES)

#: Default stroke counts for the four corrupted severity levels.
DEFAULT_STROKE_COUNTS: Final[tuple[int, ...]] = (1, 2, 4, 7)

#: A stroke is always thin; wider strokes would change the family's character.
DEFAULT_STROKE_WIDTH_PX: Final[int] = 1

#: Pure black, matching the renderer's foreground. Drawing over glyph cores is
#: deliberately possible; it is one source of ties in the severity trend.
DEFAULT_STROKE_INTENSITY: Final[int] = 0

#: Lines are mostly horizontal/diagonal, as in common CAPTCHA interference.
DEFAULT_LINE_ANGLE_DEG: Final[tuple[float, float]] = (-60.0, 60.0)

#: Arc sweep range. A short arc looks like a scratch; a long one looks like a
#: partial ellipse. Both cross the text region because the pivot is in the ink.
DEFAULT_ARC_SWEEP_DEG: Final[tuple[float, float]] = (120.0, 240.0)

#: Ellipse radii as a fraction of image width, and the vertical/horizontal ratio.
DEFAULT_ARC_RADIUS_FRACTION: Final[tuple[float, float]] = (0.20, 0.55)
DEFAULT_ARC_ASPECT_RATIO: Final[tuple[float, float]] = (0.50, 1.50)


class InterferenceError(ValueError):
    """Raised when an interference specification or stroke is invalid."""


def _validate_range(
    values: tuple[float, float],
    label: str,
    *,
    low_limit: float,
    high_limit: float,
) -> None:
    low, high = values
    if not low_limit <= low <= high <= high_limit:
        raise InterferenceError(
            f"{label} must satisfy {low_limit} <= min <= max <= {high_limit}, got {values}"
        )


@dataclass(frozen=True, slots=True)
class InterferenceLadder:
    """A predeclared stroke-count ladder for one interference variant.

    Parameters
    ----------
    variant:
        One of :data:`SUPPORTED_VARIANTS`.
    stroke_counts:
        Strictly increasing positive stroke counts, one per severity level.
        Severity is 1-based, matching the frozen dataset's convention.
    width_px:
        Stroke width in pixels.
    intensity:
        Stroke gray level in ``[0, 255]``.
    line_angle_deg:
        Inclusive range of line angles in degrees, measured from horizontal.
    arc_sweep_deg:
        Inclusive range of arc sweeps in degrees.
    arc_radius_fraction:
        Inclusive range of the horizontal ellipse radius as a fraction of width.
    arc_aspect_ratio:
        Inclusive range of ``vertical_radius / horizontal_radius``.
    """

    variant: str
    stroke_counts: tuple[int, ...]
    width_px: int = DEFAULT_STROKE_WIDTH_PX
    intensity: int = DEFAULT_STROKE_INTENSITY
    line_angle_deg: tuple[float, float] = DEFAULT_LINE_ANGLE_DEG
    arc_sweep_deg: tuple[float, float] = DEFAULT_ARC_SWEEP_DEG
    arc_radius_fraction: tuple[float, float] = DEFAULT_ARC_RADIUS_FRACTION
    arc_aspect_ratio: tuple[float, float] = DEFAULT_ARC_ASPECT_RATIO

    def __post_init__(self) -> None:
        if self.variant not in SUPPORTED_VARIANTS:
            raise InterferenceError(
                f"unknown interference variant {self.variant!r}; supported: {SUPPORTED_VARIANTS}"
            )
        if not self.stroke_counts:
            raise InterferenceError("stroke_counts must not be empty")
        if any(
            isinstance(count, bool) or not isinstance(count, int) for count in self.stroke_counts
        ):
            raise InterferenceError(f"stroke_counts must be integers, got {self.stroke_counts}")
        if any(count < 1 for count in self.stroke_counts):
            raise InterferenceError(f"stroke_counts must be >= 1, got {self.stroke_counts}")
        for earlier, later in zip(self.stroke_counts, self.stroke_counts[1:], strict=False):
            if later <= earlier:
                raise InterferenceError(
                    "stroke_counts must be strictly increasing with severity, got "
                    f"{self.stroke_counts}"
                )
        if isinstance(self.width_px, bool) or not isinstance(self.width_px, int):
            raise InterferenceError(f"width_px must be an integer, got {self.width_px!r}")
        if not 1 <= self.width_px <= 8:
            raise InterferenceError(f"width_px must be in [1, 8], got {self.width_px}")
        if isinstance(self.intensity, bool) or not isinstance(self.intensity, int):
            raise InterferenceError(f"intensity must be an integer, got {self.intensity!r}")
        if not 0 <= self.intensity <= 255:
            raise InterferenceError(f"intensity must be in [0, 255], got {self.intensity}")
        _validate_range(self.line_angle_deg, "line_angle_deg", low_limit=-89.9, high_limit=89.9)
        _validate_range(self.arc_sweep_deg, "arc_sweep_deg", low_limit=1.0, high_limit=360.0)
        _validate_range(
            self.arc_radius_fraction,
            "arc_radius_fraction",
            low_limit=0.01,
            high_limit=1.0,
        )
        _validate_range(
            self.arc_aspect_ratio,
            "arc_aspect_ratio",
            low_limit=0.05,
            high_limit=4.0,
        )

    @property
    def n_levels(self) -> int:
        """Number of corrupted severity levels (severity is 1-based)."""
        return len(self.stroke_counts)

    @property
    def max_count(self) -> int:
        """Largest stroke count, i.e. the size of the shared stroke pool."""
        return self.stroke_counts[-1]

    def spec(self, severity: int) -> InterferenceSpec:
        """Return the explicit specification for a 1-based severity."""
        if not 1 <= severity <= self.n_levels:
            raise InterferenceError(
                f"severity {severity} is out of range for a ladder with "
                f"{self.n_levels} corrupted levels (valid: 1..{self.n_levels})"
            )
        return InterferenceSpec(
            variant=self.variant,
            severity=severity,
            stroke_count=self.stroke_counts[severity - 1],
            width_px=self.width_px,
            intensity=self.intensity,
        )


@dataclass(frozen=True, slots=True)
class InterferenceSpec:
    """A fully explicit interference request."""

    variant: str
    severity: int
    stroke_count: int
    width_px: int
    intensity: int

    @property
    def label(self) -> str:
        """Compact label used in records and reports."""
        return f"{self.variant}[{self.severity}]={self.stroke_count}strokes"


@dataclass(frozen=True, slots=True)
class LineStroke:
    """A straight stroke from ``start`` to ``end`` with its planned pivot."""

    start: tuple[int, int]
    end: tuple[int, int]
    pivot: tuple[float, float]
    angle_deg: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable description."""
        return {
            "kind": "line",
            "start": list(self.start),
            "end": list(self.end),
            "pivot": list(self.pivot),
            "angle_deg": self.angle_deg,
        }


@dataclass(frozen=True, slots=True)
class ArcStroke:
    """An elliptical arc with its planned pivot."""

    center: tuple[float, float]
    radii: tuple[float, float]
    start_angle_deg: float
    end_angle_deg: float
    pivot: tuple[float, float]
    sweep_deg: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable description."""
        return {
            "kind": "arc",
            "center": list(self.center),
            "radii": list(self.radii),
            "start_angle_deg": self.start_angle_deg,
            "end_angle_deg": self.end_angle_deg,
            "pivot": list(self.pivot),
            "sweep_deg": self.sweep_deg,
        }


Stroke = LineStroke | ArcStroke


@dataclass(frozen=True, slots=True)
class InterferenceResult:
    """A corrupted image plus diagnostics about how it was produced."""

    image: np.ndarray
    spec: InterferenceSpec
    geometry: tuple[Stroke, ...]
    seed: int
    #: Fraction of all pixels changed relative to the clean reference.
    changed_pixel_fraction: float
    #: Fraction of the clean reference's ink pixels changed by the strokes.
    changed_ink_fraction: float
    #: Fraction of the clean reference's non-ink pixels changed by the strokes.
    changed_background_fraction: float
    #: True when at least one changed pixel lies on the canvas border.
    border_touched: bool

    def __post_init__(self) -> None:
        validate_image(self.image, name="interference image")
        if not self.geometry:
            raise InterferenceError("an interference result must record at least one stroke")
        for name in (
            "changed_pixel_fraction",
            "changed_ink_fraction",
            "changed_background_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InterferenceError(f"{name} must be in [0, 1], got {value}")
        if self.seed < 0:
            raise InterferenceError(f"seed must be >= 0, got {self.seed}")


def default_ladders() -> dict[str, InterferenceLadder]:
    """Return the pilot's default ladders (mirrored by the pilot config file)."""
    return {
        variant: InterferenceLadder(variant=variant, stroke_counts=DEFAULT_STROKE_COUNTS)
        for variant in SUPPORTED_VARIANTS
    }


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InterferenceError(f"seed must be an integer, got {seed!r}")
    if not 0 <= seed < 2**64:
        raise InterferenceError(f"seed must be in [0, 2**64), got {seed}")


def _ink_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive ``(x_min, y_min, x_max, y_max)`` of dark pixels."""
    ink = image < 128
    if not np.any(ink):
        height, width = image.shape
        return 0, 0, width - 1, height - 1
    rows, columns = np.nonzero(ink)
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def _boundary_distance(
    pivot: tuple[float, float],
    direction: tuple[float, float],
    shape: tuple[int, int],
) -> float:
    """Distance from an interior pivot to the image boundary along ``direction``."""
    height, width = shape
    px, py = pivot
    dx, dy = direction
    candidates: list[float] = []
    if dx > 0.0:
        candidates.append((width - 1 - px) / dx)
    elif dx < 0.0:
        candidates.append((0.0 - px) / dx)
    if dy > 0.0:
        candidates.append((height - 1 - py) / dy)
    elif dy < 0.0:
        candidates.append((0.0 - py) / dy)
    positive = [distance for distance in candidates if distance >= 0.0]
    if not positive:  # pragma: no cover - a non-zero direction always exits
        raise InterferenceError("could not find a boundary along the requested direction")
    return min(positive)


def _clip_point(point: tuple[float, float], shape: tuple[int, int]) -> tuple[int, int]:
    height, width = shape
    x = int(round(point[0]))
    y = int(round(point[1]))
    return min(max(x, 0), width - 1), min(max(y, 0), height - 1)


def _plan_line(
    rng: np.random.Generator,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    ladder: InterferenceLadder,
) -> LineStroke:
    x_min, y_min, x_max, y_max = bbox
    pivot = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))
    angle_deg = float(rng.uniform(*ladder.line_angle_deg))
    radians = math.radians(angle_deg)
    direction = (math.cos(radians), math.sin(radians))

    forward = _boundary_distance(pivot, direction, shape)
    backward = _boundary_distance(pivot, (-direction[0], -direction[1]), shape)
    start = _clip_point(
        (pivot[0] - backward * direction[0], pivot[1] - backward * direction[1]), shape
    )
    end = _clip_point((pivot[0] + forward * direction[0], pivot[1] + forward * direction[1]), shape)
    return LineStroke(start=start, end=end, pivot=pivot, angle_deg=angle_deg)


def _plan_arc(
    rng: np.random.Generator,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    ladder: InterferenceLadder,
) -> ArcStroke:
    height, width = shape
    x_min, y_min, x_max, y_max = bbox
    pivot = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))

    radius_x = float(
        rng.uniform(
            ladder.arc_radius_fraction[0] * width,
            ladder.arc_radius_fraction[1] * width,
        )
    )
    aspect = float(rng.uniform(*ladder.arc_aspect_ratio))
    radius_y = radius_x * aspect

    # The arc passes through the pivot by construction: the pivot is a point on
    # the ellipse at angle phi, so the centre is the pivot minus that radius.
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    center = (
        pivot[0] - radius_x * math.cos(phi),
        pivot[1] - radius_y * math.sin(phi),
    )
    sweep = float(rng.uniform(*ladder.arc_sweep_deg))
    start_angle = math.degrees(phi) - sweep / 2.0
    end_angle = math.degrees(phi) + sweep / 2.0
    return ArcStroke(
        center=center,
        radii=(radius_x, radius_y),
        start_angle_deg=start_angle,
        end_angle_deg=end_angle,
        pivot=pivot,
        sweep_deg=sweep,
    )


def plan_stroke_pool(
    image: np.ndarray,
    ladder: InterferenceLadder,
    *,
    seed: int,
) -> tuple[Stroke, ...]:
    """Plan the shared stroke pool for one image and variant.

    The pool length is ``ladder.max_count``. Severity ``k`` uses the first
    ``ladder.stroke_counts[k - 1]`` strokes, so severities are nested and all
    are applied to the unmodified clean reference.

    Parameters
    ----------
    image:
        Clean reference satisfying the shared image contract.
    ladder:
        Variant and geometry configuration.
    seed:
        Explicit non-negative seed. Every random draw derives from it.

    Returns
    -------
    tuple of Stroke
        Planned strokes, in draw order.
    """
    validate_image(image)
    _validate_seed(seed)
    rng = np.random.default_rng(seed)
    bbox = _ink_bbox(image)
    planner = _plan_line if ladder.variant == VARIANT_THIN_LINES else _plan_arc
    return tuple(planner(rng, bbox, image.shape, ladder) for _ in range(ladder.max_count))


def apply_strokes(
    image: np.ndarray,
    strokes: tuple[Stroke, ...] | list[Stroke],
    *,
    width_px: int,
    intensity: int,
) -> np.ndarray:
    """Draw already-planned strokes onto a copy of ``image``.

    This is the replay primitive: given the geometry recorded in a pilot
    manifest, it reproduces the corrupted pixels exactly without consulting an
    RNG. The input image is never modified.
    """
    validate_image(image)
    if not 1 <= width_px <= 8:
        raise InterferenceError(f"width_px must be in [1, 8], got {width_px}")
    if not 0 <= intensity <= 255:
        raise InterferenceError(f"intensity must be in [0, 255], got {intensity}")
    if not strokes:
        return copy_image(image)

    canvas = Image.fromarray(copy_image(image))
    draw = ImageDraw.Draw(canvas)
    height, width = image.shape
    for stroke in strokes:
        if isinstance(stroke, LineStroke):
            draw.line([stroke.start, stroke.end], fill=intensity, width=width_px)
        elif isinstance(stroke, ArcStroke):
            center_x, center_y = stroke.center
            radius_x, radius_y = stroke.radii
            box = (
                int(round(center_x - radius_x)),
                int(round(center_y - radius_y)),
                int(round(center_x + radius_x)),
                int(round(center_y + radius_y)),
            )
            draw.arc(
                box,
                start=stroke.start_angle_deg,
                end=stroke.end_angle_deg,
                fill=intensity,
                width=width_px,
            )
        else:  # pragma: no cover - guarded by the Stroke union and parsers
            raise InterferenceError(f"unsupported stroke type {type(stroke).__name__}")
    result = np.array(canvas, dtype=np.uint8).copy()
    if result.shape != (height, width):  # pragma: no cover - structural guard
        raise InterferenceError(
            f"drawing changed the image shape from {(height, width)} to {result.shape}"
        )
    return result


def apply_interference(
    image: np.ndarray,
    ladder: InterferenceLadder,
    severity: int,
    *,
    seed: int,
) -> InterferenceResult:
    """Apply one severity of one interference variant.

    The stroke pool is planned once from ``seed`` and the first
    ``spec.stroke_count`` strokes are drawn. Calling this for increasing
    severities with the same seed yields nested corruptions of the same clean
    reference.
    """
    validate_image(image)
    spec = ladder.spec(severity)
    pool = plan_stroke_pool(image, ladder, seed=seed)
    strokes = pool[: spec.stroke_count]
    corrupted = apply_strokes(image, strokes, width_px=spec.width_px, intensity=spec.intensity)

    difference = corrupted != image
    changed_pixels = int(np.count_nonzero(difference))
    ink = image < 128
    ink_count = int(np.count_nonzero(ink))
    background_count = int(image.size - ink_count)
    border = bool(
        np.any(difference[0, :])
        or np.any(difference[-1, :])
        or np.any(difference[:, 0])
        or np.any(difference[:, -1])
    )
    return InterferenceResult(
        image=corrupted,
        spec=spec,
        geometry=tuple(strokes),
        seed=seed,
        changed_pixel_fraction=changed_pixels / float(image.size),
        changed_ink_fraction=(
            float(np.count_nonzero(difference & ink)) / ink_count if ink_count else 0.0
        ),
        changed_background_fraction=(
            float(np.count_nonzero(difference & ~ink)) / background_count
            if background_count
            else 0.0
        ),
        border_touched=border,
    )


def geometry_to_json(strokes: tuple[Stroke, ...] | list[Stroke]) -> list[dict[str, object]]:
    """Return a JSON-serialisable list of stroke descriptions."""
    return [stroke.as_dict() for stroke in strokes]


def strokes_from_json(payload: object) -> tuple[Stroke, ...]:
    """Rebuild strokes from a manifest's recorded geometry.

    Raises
    ------
    InterferenceError
        If the payload is malformed or contains an unknown stroke kind.
    """
    if not isinstance(payload, list) or not payload:
        raise InterferenceError("geometry must be a non-empty list")
    strokes: list[Stroke] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise InterferenceError(f"geometry[{index}] must be an object")
        kind = item.get("kind")
        try:
            if kind == "line":
                strokes.append(
                    LineStroke(
                        start=_int_pair(item["start"], f"geometry[{index}].start"),
                        end=_int_pair(item["end"], f"geometry[{index}].end"),
                        pivot=_float_pair(item["pivot"], f"geometry[{index}].pivot"),
                        angle_deg=float(item["angle_deg"]),
                    )
                )
            elif kind == "arc":
                strokes.append(
                    ArcStroke(
                        center=_float_pair(item["center"], f"geometry[{index}].center"),
                        radii=_float_pair(item["radii"], f"geometry[{index}].radii"),
                        start_angle_deg=float(item["start_angle_deg"]),
                        end_angle_deg=float(item["end_angle_deg"]),
                        pivot=_float_pair(item["pivot"], f"geometry[{index}].pivot"),
                        sweep_deg=float(item["sweep_deg"]),
                    )
                )
            else:
                raise InterferenceError(f"geometry[{index}] has unknown kind {kind!r}")
        except (KeyError, TypeError, ValueError) as exc:
            raise InterferenceError(f"invalid geometry[{index}]: {exc}") from exc
    return tuple(strokes)


def _int_pair(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise InterferenceError(f"{label} must be a two-element list")
    return int(value[0]), int(value[1])


def _float_pair(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise InterferenceError(f"{label} must be a two-element list")
    return float(value[0]), float(value[1])
