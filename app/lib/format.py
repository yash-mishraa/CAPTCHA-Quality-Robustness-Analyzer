"""Display formatting for research values.

All numbers originate in frozen report tables.  These helpers only change how
they are written, never what they mean.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.lib import contracts

MINUS = "\u2212"
EN_DASH = "\u2013"

FONT_LABELS = {
    "arial": "Arial",
    "courier_new": "Courier New",
    "times_new_roman": "Times New Roman",
    "georgia": "Georgia",
    "monaco": "Monaco",
    "impact": "Impact",
    "verdana": "Verdana",
    "trebuchet_ms": "Trebuchet MS",
    "chalkduster": "Chalkduster",
    "luminari": "Luminari",
}

FONT_ORDER = (
    "arial",
    "courier_new",
    "times_new_roman",
    "georgia",
    "monaco",
    "impact",
    "verdana",
    "trebuchet_ms",
    "chalkduster",
    "luminari",
)

GROUP_LABELS = {
    "familiar": "Familiar",
    "m7b_held_out": "M7B held-out",
    "new_held_out": "New held-out",
}

GROUP_ORDER = ("familiar", "m7b_held_out", "new_held_out")

FAMILY_LABELS = {
    "gaussian_blur": "Gaussian blur",
    "gaussian_noise": "Gaussian noise",
    "thin_lines": "Thin lines",
    "arc_strokes": "Arc strokes",
    "elastic": "Elastic",
    "clean": "Clean",
}

PRIMARY_FAMILY_ORDER = (
    "gaussian_blur",
    "gaussian_noise",
    "thin_lines",
    "arc_strokes",
    "elastic",
)

FAMILY_ORDER = (*PRIMARY_FAMILY_ORDER, "clean")

SCOPE_LABELS = {
    "primary": "Primary",
    "diagnostic": "Diagnostic",
}

FEATURE_LABELS = {
    "intensity_mean": ("Intensity mean", "Average pixel intensity"),
    "intensity_std": ("Intensity standard deviation", "Pixel spread"),
    "contrast_rms": ("RMS contrast", "Relative contrast"),
    "ink_fraction": ("Ink fraction", "Darker Otsu-class fraction"),
    "laplacian_variance": ("Laplacian variance", "Edge and sharpness energy"),
    "edge_density": ("Edge density", "Canny edge fraction"),
    "histogram_entropy": ("Histogram entropy", "Intensity-distribution entropy"),
    "noise_sigma_immerkaer": ("Immerkaer noise sigma", "Noise-plus-texture estimate"),
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return False


def _clean_negative_zero(text: str) -> str:
    if text in {"-0", "-0.0", "-0.00", "-0.000", "-0.0000"}:
        return text[1:]
    return text


def fmt(value: Any, digits: int = 4) -> str:
    """Format a finite number with a typographic minus sign."""

    if _is_missing(value):
        return "Not reported"
    number = float(value)
    if not math.isfinite(number):
        return "Not reported"
    if abs(number) < 0.5 * (10**-digits):
        number = 0.0
    text = f"{number:.{digits}f}"
    text = _clean_negative_zero(text)
    return text.replace("-", MINUS)


def fmt_metric(metric: str, value: Any) -> str:
    digits = {"mae": 4, "rmse": 4, "r2": 3}.get(metric, 4)
    return fmt(value, digits)


def fmt_signed(value: Any, digits: int = 4) -> str:
    if _is_missing(value):
        return "Not reported"
    number = float(value)
    if not math.isfinite(number):
        return "Not reported"
    if number > 0:
        return f"+{fmt(number, digits)}"
    return fmt(number, digits)


def fmt_residual(value: Any) -> str:
    """Residuals read naturally without a leading plus for positive values."""

    return fmt(value, 4)


def fmt_pct(value: Any, digits: int = 2) -> str:
    if _is_missing(value):
        return "Not reported"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_int(value: Any) -> str:
    if _is_missing(value):
        return "Not reported"
    return f"{int(value):,}"


def fmt_interval(low: Any, high: Any, digits: int = 4) -> str:
    return f"{fmt(low, digits)}{EN_DASH}{fmt(high, digits)}"


def fmt_feature(value: Any) -> str:
    """Readable feature values without losing the recorded precision."""

    if _is_missing(value):
        return "Not reported"
    number = float(value)
    if not math.isfinite(number):
        return "Not reported"
    if abs(number) >= 1_000:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:,.2f}"
    return f"{number:.4f}"


def font_label(font_id: str) -> str:
    return FONT_LABELS.get(font_id, font_id.replace("_", " ").title())


def font_label_with_group(font_id: str, font_group: str) -> str:
    return f"{font_label(font_id)} \u00b7 {group_label(font_group)}"


def group_label(font_group: str) -> str:
    return GROUP_LABELS.get(font_group, font_group.replace("_", " ").title())


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.replace("_", " ").title())


def scope_label(scope: str) -> str:
    return SCOPE_LABELS.get(scope, scope.title())


def case_label(case_id: str, text_length: Any) -> str:
    if _is_missing(text_length):
        return case_id
    return f"{case_id} \u00b7 {int(text_length)} characters"


def severity_label(family: str, severity: Any, amplitude: Any = None) -> str:
    if _is_missing(severity):
        return "Not reported"
    level = int(severity)
    if family == "clean":
        return "Clean reference (diagnostic)"
    if family == "elastic" and level == 1:
        return "s1 \u00b7 0.5 px (diagnostic)"
    if family == "elastic" and not _is_missing(amplitude):
        return f"s{level} \u00b7 {float(amplitude):g} px"
    return f"s{level}"


def sort_font_options(core: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (font_id, font_group) pairs in roster order."""

    pairs = (
        core[["font_id", "font_group"]]
        .drop_duplicates()
        .set_index("font_id")["font_group"]
        .to_dict()
    )
    group_rank = {group: index for index, group in enumerate(GROUP_ORDER)}
    font_rank = {font: index for index, font in enumerate(FONT_ORDER)}
    return sorted(
        pairs.items(),
        key=lambda item: (group_rank.get(item[1], 99), font_rank.get(item[0], 99)),
    )


def family_options(include_diagnostics: bool) -> list[str]:
    options = list(PRIMARY_FAMILY_ORDER)
    if include_diagnostics:
        options.append("clean")
    return options


def severity_options(
    core: pd.DataFrame,
    *,
    font_id: str,
    case_id: str,
    family: str,
    include_diagnostics: bool,
) -> list[int]:
    selected = core[
        (core["font_id"] == font_id)
        & (core["case_id"] == case_id)
        & (core["variant_family"] == family)
    ]
    if not include_diagnostics:
        selected = selected[selected["protocol_scope"] == "primary"]
    return sorted(int(value) for value in selected["severity_level"].unique())


def default_severity(options: list[int]) -> int | None:
    if not options:
        return None
    return 2 if 2 in options else options[0]


def known_fonts() -> tuple[str, ...]:
    return contracts.FONT_IDS
