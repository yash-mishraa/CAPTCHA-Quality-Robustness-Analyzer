"""Deterministic synthetic text rendering for the target-validation pilot.

This is the smallest renderer that satisfies the pilot's needs: a fixed canvas,
a small configurable font set, deterministic text selection, and deterministic
per-character jitter. It is deliberately *not* the Milestone 2 renderer: there
are no backgrounds, colours, rotations, or text-length distributions yet.

The clean render is the reference image for every reference-based metric. It is
returned read-only (see :mod:`captcha_quality.imaging`) so that corruption code
cannot mutate the reference in place.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from captcha_quality.imaging import freeze_image, validate_image

#: Fixed canvas size (width, height) in pixels for the whole pilot.
DEFAULT_CANVAS: Final[tuple[int, int]] = (192, 64)

#: Lowest ``ink_fraction`` that still counts as "text was actually drawn".
MIN_INK_FRACTION: Final[float] = 0.005

#: Uppercase/digit character set with visually ambiguous glyphs removed
#: (no ``0``/``O``, ``1``/``I``/``L``) so legibility is not confounded by
#: glyph ambiguity before any distortion is applied.
DEFAULT_CHARSET: Final[str] = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"

BACKGROUND: Final[int] = 255
FOREGROUND: Final[int] = 0

#: Supersampling factor used when rasterising glyphs, then downsampled with
#: LANCZOS for anti-aliased edges.
SUPERSAMPLE: Final[int] = 4


class FontResolutionError(RuntimeError):
    """Raised when too few of the requested fonts are available locally."""


class RenderError(RuntimeError):
    """Raised when the renderer cannot satisfy its configuration."""


@dataclass(frozen=True, slots=True)
class FontSpec:
    """A logical font name mapped to known system font locations.

    Several candidate paths are allowed per logical name because the same font
    ships in different directories across platforms. If none of them exists the
    font is reported as unavailable; it is never silently replaced.
    """

    key: str
    candidates: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ResolvedFont:
    """A font that exists on this machine."""

    key: str
    path: Path


@dataclass(frozen=True, slots=True)
class UnresolvedFont:
    """A requested font that could not be located, with the reason recorded."""

    key: str
    searched: tuple[Path, ...]

    @property
    def reason(self) -> str:
        paths = ", ".join(str(path) for path in self.searched)
        return f"no candidate path exists (searched: {paths})"


@dataclass(frozen=True, slots=True)
class FontResolution:
    """Result of resolving requested fonts against the local filesystem."""

    available: tuple[ResolvedFont, ...]
    unavailable: tuple[UnresolvedFont, ...]

    def describe_unavailable(self) -> str:
        """Return a human-readable summary of every skipped font."""
        return "; ".join(f"{font.key}: {font.reason}" for font in self.unavailable)


#: Fonts available on the development machine. Arial (sans-serif), Courier New
#: (monospace), and Times New Roman (serif) were chosen to span stroke weight
#: and serif style, which is what the cross-font stability check probes.
CANDIDATE_FONTS: Final[tuple[FontSpec, ...]] = (
    FontSpec(
        key="arial",
        candidates=(
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        ),
    ),
    FontSpec(
        key="courier_new",
        candidates=(
            Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
            Path("/Library/Fonts/Courier New.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf"),
        ),
    ),
    FontSpec(
        key="times_new_roman",
        candidates=(
            Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
            Path("/Library/Fonts/Times New Roman.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
        ),
    ),
)


def resolve_fonts(
    keys: Sequence[str] | None = None,
    *,
    specs: Sequence[FontSpec] = CANDIDATE_FONTS,
    min_required: int = 1,
) -> FontResolution:
    """Resolve logical font names to existing local font files.

    Parameters
    ----------
    keys:
        Logical font names to resolve. ``None`` requests every font in ``specs``.
    specs:
        Font registry to resolve against.
    min_required:
        Minimum number of successfully resolved fonts. Fewer than this raises,
        because a pilot with too few fonts cannot test cross-font stability.

    Returns
    -------
    FontResolution
        Resolved and unresolved fonts. Unresolved fonts carry the searched paths
        so the skip reason is explicit and reportable.

    Raises
    ------
    FontResolutionError
        If an unknown key is requested, or fewer than ``min_required`` fonts
        resolve.
    """
    registry = {spec.key: spec for spec in specs}
    requested = list(registry) if keys is None else list(dict.fromkeys(keys))

    unknown = [key for key in requested if key not in registry]
    if unknown:
        raise FontResolutionError(
            f"unknown font key(s) {unknown}; known keys are {sorted(registry)}"
        )

    available: list[ResolvedFont] = []
    unavailable: list[UnresolvedFont] = []
    for key in requested:
        spec = registry[key]
        found = next((path for path in spec.candidates if path.is_file()), None)
        if found is None:
            unavailable.append(UnresolvedFont(key=key, searched=spec.candidates))
        else:
            available.append(ResolvedFont(key=key, path=found))

    if len(available) < min_required:
        raise FontResolutionError(
            f"only {len(available)} of {len(requested)} requested fonts are available, "
            f"but {min_required} are required. Unavailable -> "
            f"{'; '.join(f'{font.key}: {font.reason}' for font in unavailable) or 'none'}"
        )

    return FontResolution(available=tuple(available), unavailable=tuple(unavailable))


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Fixed, documented rendering parameters for the pilot.

    Attributes
    ----------
    canvas:
        Canvas size ``(width, height)`` in pixels.
    charset:
        Characters eligible for text selection.
    text_length:
        Number of characters per sample.
    fit_fraction:
        Target fraction of the canvas width that the text block should occupy
        once the font size is auto-fitted.
    vertical_padding:
        Fraction of canvas height kept clear at the top and bottom.
    jitter_fraction:
        Per-character vertical jitter amplitude as a fraction of canvas height.
    """

    canvas: tuple[int, int] = DEFAULT_CANVAS
    charset: str = DEFAULT_CHARSET
    text_length: int = 5
    fit_fraction: float = 0.86
    vertical_padding: float = 0.10
    jitter_fraction: float = 0.05

    def __post_init__(self) -> None:
        width, height = self.canvas
        if width < 1 or height < 1:
            raise RenderError(f"canvas must be positive, got {self.canvas}")
        if min(self.canvas) < 11:
            raise RenderError(
                f"canvas {self.canvas} is too small: each side must be at least 11 px "
                "so that SSIM's 11x11 window fits"
            )
        if len(self.charset) < 2:
            raise RenderError("charset must contain at least two characters")
        if len(set(self.charset)) != len(self.charset):
            raise RenderError("charset must not contain duplicate characters")
        if self.text_length < 1:
            raise RenderError(f"text_length must be >= 1, got {self.text_length}")
        if not 0.0 < self.fit_fraction <= 1.0:
            raise RenderError(f"fit_fraction must be in (0, 1], got {self.fit_fraction}")
        if not 0.0 <= self.vertical_padding < 0.5:
            raise RenderError(f"vertical_padding must be in [0, 0.5), got {self.vertical_padding}")
        if not 0.0 <= self.jitter_fraction < 0.5:
            raise RenderError(f"jitter_fraction must be in [0, 0.5), got {self.jitter_fraction}")


@dataclass(frozen=True, slots=True)
class CleanRender:
    """A clean (uncorrupted) rendering and its recorded provenance."""

    image: np.ndarray
    text: str
    font_key: str
    font_size_px: int
    ink_fraction: float
    sha256: str

    def __post_init__(self) -> None:
        validate_image(self.image, name="clean render")


def sample_text(rng: np.random.Generator, config: RenderConfig) -> str:
    """Draw a deterministic text label from ``config.charset``."""
    indices = rng.integers(0, len(config.charset), size=config.text_length)
    return "".join(config.charset[int(index)] for index in indices)


def _text_block_size(font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int]:
    """Return the ``(width, height)`` of the rendered text block."""
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def _fit_font_size(path: Path, text: str, config: RenderConfig) -> int:
    """Largest supersampled font size whose text block fits the canvas.

    Binary search keeps this deterministic and cheap. The search is bounded by
    the supersampled canvas height so the returned size is stable.
    """
    width, height = config.canvas
    hi_scale = SUPERSAMPLE
    max_size = height * hi_scale
    target_width = width * hi_scale * config.fit_fraction
    target_height = height * hi_scale * (1.0 - 2.0 * config.vertical_padding)

    best = 1
    low, high = 1, max_size
    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(str(path), mid)
        block_width, block_height = _text_block_size(font, text)
        if block_width <= target_width and block_height <= target_height:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    if best == 1:
        raise RenderError(
            f"could not fit text {text!r} into canvas {config.canvas} with font {path.name}"
        )
    return best


def render_clean(
    font: ResolvedFont,
    text: str,
    config: RenderConfig,
    *,
    rng: np.random.Generator,
) -> CleanRender:
    """Render a clean text image.

    The glyphs are rasterised on a supersampled canvas and downsampled with
    LANCZOS, which yields anti-aliased edges and a smooth grayscale histogram.

    Parameters
    ----------
    font:
        Resolved font to draw with.
    text:
        Text label to draw; every character must occur in ``config.charset``.
    config:
        Rendering parameters.
    rng:
        Seeded generator controlling per-character vertical jitter. Passing a
        seeded generator is what makes repeated runs reproducible.

    Returns
    -------
    CleanRender
        The read-only clean image plus recorded provenance (text, font, fitted
        size, ink fraction, content hash).
    """
    if not text:
        raise RenderError("text must not be empty")
    unknown = sorted(set(text) - set(config.charset))
    if unknown:
        raise RenderError(f"text {text!r} contains characters outside the charset: {unknown}")

    width, height = config.canvas
    scale = SUPERSAMPLE
    size_hi = _fit_font_size(font.path, text, config)
    pil_font = ImageFont.truetype(str(font.path), size_hi)

    canvas_hi = Image.new("L", (width * scale, height * scale), color=BACKGROUND)
    draw = ImageDraw.Draw(canvas_hi)

    block_width, block_height = _text_block_size(pil_font, text)
    cursor_x = (width * scale - block_width) / 2.0
    baseline_y = (height * scale - block_height) / 2.0
    jitter_px = config.jitter_fraction * height * scale

    for character in text:
        advance = pil_font.getlength(character)
        offset_y = float(rng.uniform(-jitter_px, jitter_px))
        draw.text((cursor_x, baseline_y + offset_y), character, font=pil_font, fill=FOREGROUND)
        cursor_x += advance

    canvas = canvas_hi.resize((width, height), resample=Image.LANCZOS)
    image = np.array(canvas, dtype=np.uint8).copy()
    image = freeze_image(image)

    ink_fraction = float(np.count_nonzero(image < 128) / image.size)
    if ink_fraction < MIN_INK_FRACTION:
        raise RenderError(
            f"rendered text is effectively empty (ink_fraction={ink_fraction:.5f}); "
            "check the font, canvas size, and charset"
        )

    return CleanRender(
        image=image,
        text=text,
        font_key=font.key,
        font_size_px=size_hi // scale,
        ink_fraction=ink_fraction,
        sha256=hashlib.sha256(image.tobytes()).hexdigest(),
    )
