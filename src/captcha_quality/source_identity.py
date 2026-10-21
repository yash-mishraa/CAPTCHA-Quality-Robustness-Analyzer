"""Canonical source-identity scheme for the matched cross-font benchmark.

This module owns the identity rules that the Milestone 8B execution protocol
pins. It defines exactly one identifier per level of the benchmark hierarchy
and one documented hash for each canonical record:

``case_id``
    The matched text case. Stable, unique, and committed in the versioned
    text-case manifest. The same ``case_id`` is used by all ten typefaces,
    which is what makes the design *matched*.

``source_id``
    The clean rendered source image for one (text case, typeface) pair. It
    covers the text case, the font identity and pinned font hash, the render
    configuration and its version, the generation configuration, the render
    seed and the exact text.

``variant_id``
    The corruption variant of a source: the clean baseline or one
    (family, severity) pair.

``sample_id``
    ``<source_id>__<variant_id>``. This is the Milestone 2 / Milestone 7B
    convention and the frozen Milestone 8A protocol requires it.

``row_id``
    The canonical identity of one planned dataset row. It additionally
    distinguishes the corruption family, the corruption severity, the derived
    variant seed and the primary/diagnostic status, none of which the
    ``sample_id`` encodes on its own.

``case_hash`` / ``manifest_hash``
    The documented SHA-256 digests of the canonical case record and of the
    canonical ordered manifest.

The Milestone 2 and Milestone 7B identifier helpers are reused verbatim. The
dataset module is frozen by the Milestone 8A lifecycle pins, so the
established scheme is imported from it rather than re-implemented here.

Every digest is taken over a *canonical serialisation*: JSON with sorted keys,
compact separators, no non-finite floats, encoded as UTF-8. No identifier
depends on directory traversal order, wall-clock time, absolute paths, runtime
randomness or row ordering. The only randomness in the whole scheme is the
explicitly derived noise seed, and it is a pure function of documented inputs.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from captcha_quality.corruptions import derive_noise_seed

# The Milestone 2 / Milestone 7B identifier helpers are reused verbatim. The
# module is frozen by the Milestone 8A lifecycle pins, so the established
# scheme is imported rather than re-implemented here.
from captcha_quality.dataset import _stable_id, _variant_id
from captcha_quality.provenance import canonical_json, sha256_bytes

__all__ = [
    "CANONICAL_SERIALIZATION",
    "CLEAN_FAMILY",
    "CORRUPTION_FAMILIES",
    "IDENTITY_SCHEME_VERSION",
    "RENDER_CONFIG_VERSION",
    "GlyphCoverageError",
    "IdentityError",
    "TextCase",
    "canonical_bytes",
    "canonical_hash",
    "canonical_case_record",
    "case_record_hash",
    "font_codepoints",
    "generation_config_hash",
    "generation_config_payload",
    "image_relpath",
    "manifest_hash",
    "missing_glyphs",
    "normalization_policy",
    "render_config_hash",
    "render_config_payload",
    "render_seed",
    "row_id",
    "source_id",
    "stable_case_id",
    "validate_stored_text",
    "variant_id",
    "variant_seed",
]

#: Version of the canonical identity scheme itself.
IDENTITY_SCHEME_VERSION: Final[str] = "v1"

#: Version of the renderer configuration that the render-config hash covers.
RENDER_CONFIG_VERSION: Final[str] = "v1"

#: The documented canonical serialisation used for every digest.
CANONICAL_SERIALIZATION: Final[str] = (
    "json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False) encoded as UTF-8"
)

#: The image subdirectory that holds the clean reference renders.
CLEAN_FAMILY: Final[str] = "clean"

#: The corruption families, in the order the protocol declares them.
CORRUPTION_FAMILIES: Final[tuple[str, ...]] = (
    "gaussian_blur",
    "gaussian_noise",
    "thin_lines",
    "arc_strokes",
    "elastic",
)

#: The explicit normalisation policy for the stored text cases.
#:
#: The stored strings are used *verbatim*. Nothing is normalised at runtime.
#: The stored strings are pure ASCII, so the Unicode normal forms agree by
#: construction; the verifier asserts that they do rather than applying a
#: normalisation and hiding a difference.
normalization_policy: Final[str] = "none"

#: The normalisation forms the stored text must already satisfy.
REQUIRED_NORMALIZATION_FORMS: Final[tuple[str, ...]] = ("NFC", "NFD", "NFKC", "NFKD")


class IdentityError(ValueError):
    """Raised when an identity or provenance rule is violated."""


class GlyphCoverageError(IdentityError):
    """Raised when a pinned face cannot render a required character."""


@dataclass(frozen=True, slots=True)
class TextCase:
    """One matched text case, exactly as stored in the committed manifest."""

    case_id: str
    case_index: int
    text: str
    text_utf8_hex: str
    text_sha256: str
    text_length: int
    case_hash: str

    def record(self) -> dict[str, Any]:
        """The canonical record of this case."""
        return canonical_case_record(
            {
                "case_id": self.case_id,
                "case_index": self.case_index,
                "normalization": normalization_policy,
                "text": self.text,
                "text_length": self.text_length,
                "text_sha256": self.text_sha256,
                "text_utf8_hex": self.text_utf8_hex,
            }
        )


# ---------------------------------------------------------------------------
# canonical serialisation and hashing
# ---------------------------------------------------------------------------


def canonical_bytes(payload: object) -> bytes:
    """Serialise ``payload`` with the documented canonical serialisation."""
    return canonical_json(payload).encode("utf-8")


def canonical_hash(payload: object) -> str:
    """The SHA-256 digest of the canonical serialisation of ``payload``."""
    return sha256_bytes(canonical_bytes(payload))


def canonical_case_record(case: Mapping[str, object]) -> dict[str, Any]:
    """The exact canonical record of one matched text case.

    The record carries every property that defines the case, so its digest is
    a complete fingerprint of the case. The record never contains its own
    digest, which keeps the hash non-circular.
    """
    try:
        return {
            "case_id": str(case["case_id"]),
            "case_index": int(case["case_index"]),  # type: ignore[arg-type]
            "normalization": str(case.get("normalization", normalization_policy)),
            "text": str(case["text"]),
            "text_length": int(case["text_length"]),  # type: ignore[arg-type]
            "text_sha256": str(case["text_sha256"]),
            "text_utf8_hex": str(case["text_utf8_hex"]),
        }
    except KeyError as error:
        raise IdentityError(f"text case is missing the required field {error}") from error


def case_record_hash(case: Mapping[str, object]) -> str:
    """The documented SHA-256 digest of one canonical case record."""
    return canonical_hash(canonical_case_record(case))


def manifest_hash(cases: Sequence[Mapping[str, object]]) -> str:
    """The documented SHA-256 digest of the ordered canonical case list.

    The order of ``cases`` is part of the digest, so the manifest ordering is
    pinned by the digest itself.
    """
    return canonical_hash([canonical_case_record(case) for case in cases])


# ---------------------------------------------------------------------------
# text validation and glyph coverage
# ---------------------------------------------------------------------------


def validate_stored_text(text: str, *, charset: str) -> None:
    """Validate a stored text case without ever normalising it.

    The stored string is used verbatim at runtime. This function only
    *asserts* the invariants the manifest promises, and raises
    :class:`IdentityError` when one of them does not hold. It never rewrites
    the string, so a difference can never be hidden by a silent normalisation.
    """
    if not isinstance(text, str) or not text:
        raise IdentityError("a stored text case must be a non-empty string")
    for form in REQUIRED_NORMALIZATION_FORMS:
        if not unicodedata.is_normalized(form, text):
            raise IdentityError(
                f"stored text {text!r} is not {form}-normalised; the manifest requires "
                "the stored string to be used verbatim"
            )
    outside = sorted({character for character in text if character not in charset})
    if outside:
        raise IdentityError(
            f"stored text {text!r} contains characters outside the project charset: {outside}"
        )
    if any(ord(character) > 0x7F for character in text):
        raise IdentityError(
            f"stored text {text!r} contains non-ASCII characters; the manifest stores "
            "pure ASCII so the UTF-8 encoding is unambiguous"
        )


def font_codepoints(path: Path | str) -> frozenset[int]:
    """The Unicode codepoints a font file declares in its character map.

    The coverage evidence comes from the font's own ``cmap`` table, read with
    fontTools. A character that is absent from the map has no glyph in that
    face; it is never silently substituted with ``.notdef`` or a question mark.
    """
    from fontTools.ttLib import TTFont

    font_path = Path(path)
    if not font_path.is_file():
        raise IdentityError(f"font file is missing: {font_path}")
    try:
        with TTFont(font_path, lazy=True, fontNumber=0) as font:
            cmap = font.getBestCmap() or {}
    except Exception as error:  # noqa: BLE001 - any unreadable font fails closed
        raise IdentityError(f"could not read the character map of {font_path}: {error}") from error
    return frozenset(int(codepoint) for codepoint in cmap)


def missing_glyphs(codepoints: Iterable[int], text: str) -> list[str]:
    """The characters of ``text`` that the face cannot render."""
    supported = set(codepoints)
    return sorted({character for character in text if ord(character) not in supported})


# ---------------------------------------------------------------------------
# configuration hashes
# ---------------------------------------------------------------------------


def render_config_payload(
    renderer: Mapping[str, Any],
    *,
    charset: str,
    text_length: int,
) -> dict[str, Any]:
    """The canonical payload of the frozen renderer configuration."""
    return {
        "background_max": int(renderer["background_max"]),
        "background_min": int(renderer["background_min"]),
        "background_mode": str(renderer["background_mode"]),
        "canvas_height": int(renderer["canvas_height"]),
        "canvas_width": int(renderer["canvas_width"]),
        "center_ink": bool(renderer["center_ink"]),
        "charset": str(charset),
        "fit_fraction": float(renderer["fit_fraction"]),
        "jitter_fraction": float(renderer["jitter_fraction"]),
        "placement_jitter_fraction": float(renderer["placement_jitter_fraction"]),
        "render_config_version": RENDER_CONFIG_VERSION,
        "spacing_jitter_fraction": float(renderer["spacing_jitter_fraction"]),
        "strict_bounds": bool(renderer["strict_bounds"]),
        "text_length": int(text_length),
        "vertical_padding": float(renderer["vertical_padding"]),
    }


def render_config_hash(payload: Mapping[str, Any]) -> str:
    """The SHA-256 digest of a canonical renderer configuration."""
    return canonical_hash(dict(payload))


def generation_config_payload(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical payload of the frozen generation configuration.

    The payload covers the seed, the seed namespaces, the renderer
    configuration, the corruption ladder and the planned counts. It therefore
    changes whenever any generation parameter changes, which is what makes it
    usable as a provenance anchor for the source identifiers.
    """
    text_cases = protocol["text_cases"]
    renderer = protocol["renderer"]
    corruptions = protocol["corruptions"]
    counts = protocol["counts"]
    return {
        "case_namespace": str(text_cases["case_namespace"]),
        "charset": str(text_cases["charset"]),
        "corruptions": {
            "arc_radius_fraction": list(corruptions["arc_radius_fraction"]),
            "arc_sweep_deg": list(corruptions["arc_sweep_deg"]),
            "arc_stroke_counts": list(corruptions["arc_stroke_counts"]),
            "blur_radii_px": list(corruptions["blur_radii_px"]),
            "elastic_diagnostic_amplitudes_px": list(
                corruptions["elastic_diagnostic_amplitudes_px"]
            ),
            "elastic_primary_amplitudes_px": list(corruptions["elastic_primary_amplitudes_px"]),
            "elastic_severity_amplitudes_px": list(corruptions["elastic_severity_amplitudes_px"]),
            "grid_cols": int(corruptions["grid_cols"]),
            "grid_rows": int(corruptions["grid_rows"]),
            "interpolation": str(corruptions["interpolation"]),
            "line_angle_deg": list(corruptions["line_angle_deg"]),
            "noise_sigmas": list(corruptions["noise_sigmas"]),
            "smoothing_sigma_px": float(corruptions["smoothing_sigma_px"]),
            "stroke_intensity": int(corruptions["stroke_intensity"]),
            "stroke_width_px": int(corruptions["stroke_width_px"]),
            "thin_line_stroke_counts": list(corruptions["thin_line_stroke_counts"]),
        },
        "counts": {
            "n_cases_per_font": int(counts["n_cases_per_font"]),
            "n_fonts": int(counts["n_fonts"]),
            "n_rows": int(counts["total_rows"]),
            "n_sources": int(counts["n_sources"]),
        },
        "dataset_seed": int(text_cases["dataset_seed"]),
        "render": {
            "background_max": int(renderer["background_max"]),
            "background_min": int(renderer["background_min"]),
            "background_mode": str(renderer["background_mode"]),
            "canvas_height": int(renderer["canvas_height"]),
            "canvas_width": int(renderer["canvas_width"]),
            "center_ink": bool(renderer["center_ink"]),
            "fit_fraction": float(renderer["fit_fraction"]),
            "jitter_fraction": float(renderer["jitter_fraction"]),
            "placement_jitter_fraction": float(renderer["placement_jitter_fraction"]),
            "spacing_jitter_fraction": float(renderer["spacing_jitter_fraction"]),
            "strict_bounds": bool(renderer["strict_bounds"]),
            "vertical_padding": float(renderer["vertical_padding"]),
        },
        "render_config_version": RENDER_CONFIG_VERSION,
        "render_namespace": str(text_cases["render_namespace"]),
        "variant_namespace": str(text_cases["variant_namespace"]),
    }


def generation_config_hash(protocol: Mapping[str, Any]) -> str:
    """The SHA-256 digest of the canonical generation configuration."""
    return canonical_hash(generation_config_payload(protocol))


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------


def stable_case_id(index: int) -> str:
    """The stable identifier of the matched text case at ``index``."""
    return f"case-{int(index):03d}"


def render_seed(dataset_seed: int, font_key: str, case_index: int) -> int:
    """The render seed of one (font, case) pair.

    The seed depends on the dataset seed, the font key and the case index.
    It does not depend on the text, so changing the render jitter can never
    change a text case, and the same case keeps the same render jitter in
    every typeface.
    """
    return derive_noise_seed(dataset_seed, "render", font_key, str(int(case_index)))


def source_id(
    *,
    case: Mapping[str, Any],
    font_id: str,
    font_sha256: str,
    render_config_hash: str,
    render_seed: int,
    generation_config_hash: str,
) -> str:
    """The stable identifier of one clean source image.

    The payload distinguishes the text case, the typeface and its pinned
    hash, the render configuration and its version, the generation
    configuration, the render seed and the exact text.
    """
    payload = {
        "case_id": str(case["case_id"]),
        "font_id": str(font_id),
        "font_sha256": str(font_sha256),
        "generation_config_hash": str(generation_config_hash),
        "render_config_hash": str(render_config_hash),
        "render_seed": int(render_seed),
        "text": str(case["text"]),
        "text_length": int(case["text_length"]),
    }
    return _stable_id("src", payload)


def variant_id(family: str, severity: int) -> str:
    """The identifier of one corruption variant of a source.

    The clean baseline keeps the Milestone 2 identifier ``clean_s0``; every
    other variant is ``<family>_s<severity>``.
    """
    return _variant_id(family, int(severity))


def variant_seed(dataset_seed: int, source: str, family: str, severity: int) -> int:
    """The derived noise seed of one corruption realisation.

    The seed is a pure function of the dataset seed, the source identity, the
    corruption family and the severity level, so the same source and
    corruption configuration always reproduce the same realisation and two
    severity levels never share a random field.
    """
    return derive_noise_seed(dataset_seed, "variant", source, family, int(severity))


def row_id(
    *,
    source: str,
    sample: str,
    variant: str,
    case: Mapping[str, Any],
    font_id: str,
    font_sha256: str,
    render_config_hash: str,
    render_seed: int,
    generation_config_hash: str,
    family: str,
    severity: int,
    variant_seed: int,
    primary: bool,
) -> str:
    """The canonical identity of one planned dataset row.

    The payload distinguishes the text case, the typeface and its pinned
    hash, the render configuration and its version, the generation
    configuration, the source image identity, the corruption family, the
    corruption severity, the derived noise seed and the primary/diagnostic
    status.
    """
    payload = {
        "case_id": str(case["case_id"]),
        "corruption_family": str(family),
        "corruption_severity": int(severity),
        "font_id": str(font_id),
        "font_sha256": str(font_sha256),
        "generation_config_hash": str(generation_config_hash),
        "primary": bool(primary),
        "render_config_hash": str(render_config_hash),
        "render_seed": int(render_seed),
        "sample_id": str(sample),
        "source_id": str(source),
        "variant_id": str(variant),
        "variant_seed": int(variant_seed),
    }
    return _stable_id("row", payload)


def image_relpath(family: str, name: str) -> str:
    """The repository-relative path of a rendered image."""
    return f"images/{family}/{name}.png"
