"""Typed configuration for the Milestone 2 synthetic dataset pipeline.

The configuration is a single TOML file so that the full experiment definition
(rendering, fonts, corruption ladders, source counts, seeds, split ratios,
dataset version, output directory and diagnostic thresholds) can be reviewed
and committed before generation. Parsing is strict: unknown or missing keys
fail with an explicit message instead of silently falling back to a default.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from captcha_quality.corruptions import SUPPORTED_FAMILIES, SeverityLadder
from captcha_quality.provenance import canonical_json, sha256_bytes
from captcha_quality.render import BACKGROUND_MODES, RenderConfig, RenderError

#: Split names in the fixed order used for allocation and reporting.
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "val", "test")

#: Feature-set identifier recorded in datasets and reports. Bump when feature
#: definitions or their order change so old feature tables are distinguishable.
FEATURE_SET_VERSION: Final[str] = "v1"

_RATIO_TOLERANCE: Final[float] = 1e-6


class ConfigError(ValueError):
    """Raised when a dataset configuration is missing, malformed, or infeasible."""


def _check_keys(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{name}] table")
    return value


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"missing required key {section_name}.{key}")
    return section[key]


def _as_str(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ConfigError(f"{label} must not be empty")
    return value


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be a boolean, got {value!r}")
    return value


def _as_sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a TOML array, got {type(value).__name__}")
    return value


def _as_str_tuple(value: Any, label: str) -> tuple[str, ...]:
    items = tuple(_as_str(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items:
        raise ConfigError(f"{label} must not be empty")
    if len(set(items)) != len(items):
        raise ConfigError(f"{label} must not contain duplicates, got {items}")
    return items


def _as_positive_int(value: Any, label: str) -> int:
    result = _as_int(value, label)
    if result < 1:
        raise ConfigError(f"{label} must be >= 1, got {result}")
    return result


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Dataset identity and source-sampling configuration."""

    name: str
    version: str
    seed: int
    output_dir: Path
    sources_per_font: int
    text_lengths: tuple[int, ...]
    text_length_weights: tuple[float, ...]
    charset: str

    def __post_init__(self) -> None:
        if not self.text_lengths:
            raise ConfigError("dataset.text_lengths must not be empty")
        if len(set(self.text_lengths)) != len(self.text_lengths):
            raise ConfigError(f"dataset.text_lengths must be unique, got {self.text_lengths}")
        if any(length < 1 for length in self.text_lengths):
            raise ConfigError(f"dataset.text_lengths must be >= 1, got {self.text_lengths}")
        if len(self.text_length_weights) != len(self.text_lengths):
            raise ConfigError(
                "dataset.text_length_weights must have one entry per text length, got "
                f"{len(self.text_length_weights)} weights for {len(self.text_lengths)} lengths"
            )
        if any(weight <= 0.0 for weight in self.text_length_weights):
            raise ConfigError(
                f"dataset.text_length_weights must be positive, got {self.text_length_weights}"
            )
        total = sum(self.text_length_weights)
        if abs(total - 1.0) > _RATIO_TOLERANCE:
            raise ConfigError(
                f"dataset.text_length_weights must sum to 1.0 (±{_RATIO_TOLERANCE}), got {total}"
            )
        if len(self.charset) < 2:
            raise ConfigError("dataset.charset must contain at least two characters")
        if len(set(self.charset)) != len(self.charset):
            raise ConfigError("dataset.charset must not contain duplicate characters")
        if self.seed < 0:
            raise ConfigError(f"dataset.seed must be >= 0, got {self.seed}")
        if self.sources_per_font < 1:
            raise ConfigError(f"dataset.sources_per_font must be >= 1, got {self.sources_per_font}")


@dataclass(frozen=True, slots=True)
class RenderSettings:
    """Dataset-level rendering policy, converted to a per-source RenderConfig."""

    canvas: tuple[int, int]
    fit_fraction: float
    vertical_padding: float
    jitter_fraction: float
    spacing_jitter_fraction: float
    placement_jitter_fraction: float
    background_mode: str
    background_range: tuple[int, int]
    center_ink: bool
    strict_bounds: bool

    def __post_init__(self) -> None:
        if self.background_mode not in BACKGROUND_MODES:
            raise ConfigError(
                f"render.background_mode must be one of {BACKGROUND_MODES}, "
                f"got {self.background_mode!r}"
            )
        low, high = self.background_range
        if not 0 <= low <= high <= 255:
            raise ConfigError(
                "render.background_min/background_max must satisfy "
                f"0 <= min <= max <= 255, got ({low}, {high})"
            )
        # Delegating to RenderConfig centralises the remaining geometry checks.
        try:
            self.to_render_config(text_length=1, charset="AB")
        except RenderError as exc:
            raise ConfigError(f"invalid render settings: {exc}") from exc

    def to_render_config(self, *, text_length: int, charset: str) -> RenderConfig:
        """Build the per-source render configuration."""
        return RenderConfig(
            canvas=self.canvas,
            charset=charset,
            text_length=text_length,
            fit_fraction=self.fit_fraction,
            vertical_padding=self.vertical_padding,
            jitter_fraction=self.jitter_fraction,
            spacing_jitter_fraction=self.spacing_jitter_fraction,
            placement_jitter_fraction=self.placement_jitter_fraction,
            background_mode=self.background_mode,
            background_range=self.background_range,
            center_ink=self.center_ink,
            strict_bounds=self.strict_bounds,
        )


@dataclass(frozen=True, slots=True)
class FontSettings:
    """Requested logical fonts and the minimum number that must resolve."""

    keys: tuple[str, ...]
    min_required: int

    def __post_init__(self) -> None:
        if not self.keys:
            raise ConfigError("fonts.keys must not be empty")
        if not 1 <= self.min_required <= len(self.keys):
            raise ConfigError(
                "fonts.min_required must be in [1, len(fonts.keys)], got "
                f"{self.min_required} for {len(self.keys)} keys"
            )


@dataclass(frozen=True, slots=True)
class FamilySettings:
    """Ordered corruption ladders used for variant generation."""

    ladders: tuple[SeverityLadder, ...]

    def __post_init__(self) -> None:
        if not self.ladders:
            raise ConfigError("at least one corruption family must be enabled")
        families = [ladder.family for ladder in self.ladders]
        if len(set(families)) != len(families):
            raise ConfigError(f"duplicate corruption family configured: {families}")
        for ladder in self.ladders:
            if ladder.family not in SUPPORTED_FAMILIES:
                raise ConfigError(f"unsupported corruption family {ladder.family!r}")
            if len(set(ladder.parameters)) != len(ladder.parameters):
                raise ConfigError(
                    f"families.{ladder.family}.levels must be strictly increasing (no duplicate "
                    f"severities), got {ladder.parameters}"
                )

    @property
    def names(self) -> tuple[str, ...]:
        """Family names in configured order."""
        return tuple(ladder.family for ladder in self.ladders)

    def ladder(self, family: str) -> SeverityLadder:
        """Return the ladder for ``family`` or raise a clear configuration error."""
        for ladder in self.ladders:
            if ladder.family == family:
                return ladder
        raise ConfigError(
            f"corruption family {family!r} is not configured; choose from {self.names}"
        )

    @property
    def n_variants_per_source(self) -> int:
        """Corrupted variants per source (the clean sample is counted separately)."""
        return sum(ladder.n_levels - 1 for ladder in self.ladders)


@dataclass(frozen=True, slots=True)
class SplitSettings:
    """Grouped split policy. Ratios are fixed before generation."""

    ratios: tuple[tuple[str, float], ...]
    seed: int
    min_sources_per_split: int
    stratify_by_font: bool

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.ratios)
        if names != SPLIT_NAMES:
            raise ConfigError(f"split ratios must be provided for {SPLIT_NAMES}, got {names}")
        for name, ratio in self.ratios:
            if not 0.0 < ratio < 1.0:
                raise ConfigError(f"splits.{name} must be in (0, 1), got {ratio}")
        total = sum(ratio for _, ratio in self.ratios)
        if abs(total - 1.0) > _RATIO_TOLERANCE:
            raise ConfigError(f"splits ratios must sum to 1.0 (±{_RATIO_TOLERANCE}), got {total}")
        if self.seed < 0:
            raise ConfigError(f"splits.seed must be >= 0, got {self.seed}")
        if self.min_sources_per_split < 1:
            raise ConfigError(
                f"splits.min_sources_per_split must be >= 1, got {self.min_sources_per_split}"
            )

    def ratio(self, split: str) -> float:
        """Return the configured ratio for ``split``."""
        for name, value in self.ratios:
            if name == split:
                return value
        raise ConfigError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")


@dataclass(frozen=True, slots=True)
class DiagnosticsSettings:
    """Predeclared thresholds for the feature/target diagnostic.

    ``trivial_single_feature_r2`` and ``strong_confound_r2`` are decision aids,
    not gates on the milestone: they classify how much of the SSIM target a
    single image statistic can reconstruct on held-out groups.
    """

    trivial_single_feature_r2: float
    strong_confound_r2: float
    min_group_rows: int

    def __post_init__(self) -> None:
        if not 0.0 < self.strong_confound_r2 < 1.0:
            raise ConfigError(
                f"diagnostics.strong_confound_r2 must be in (0, 1), got {self.strong_confound_r2}"
            )
        if not 0.0 < self.trivial_single_feature_r2 <= 1.0:
            raise ConfigError(
                "diagnostics.trivial_single_feature_r2 must be in (0, 1], got "
                f"{self.trivial_single_feature_r2}"
            )
        if self.trivial_single_feature_r2 <= self.strong_confound_r2:
            raise ConfigError(
                "diagnostics.trivial_single_feature_r2 must be greater than "
                "diagnostics.strong_confound_r2"
            )
        if self.min_group_rows < 3:
            raise ConfigError(f"diagnostics.min_group_rows must be >= 3, got {self.min_group_rows}")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Fully validated dataset experiment definition."""

    dataset: DatasetSettings
    render: RenderSettings
    fonts: FontSettings
    families: FamilySettings
    splits: SplitSettings
    diagnostics: DiagnosticsSettings
    config_path: Path

    def __post_init__(self) -> None:
        if self.splits.stratify_by_font and self.dataset.sources_per_font < len(SPLIT_NAMES):
            raise ConfigError(
                "splits.stratify_by_font requires dataset.sources_per_font >= 3 so that every "
                f"font can appear in every split; got {self.dataset.sources_per_font}"
            )
        total_sources = self.n_sources
        required = len(SPLIT_NAMES) * self.splits.min_sources_per_split
        if total_sources < required:
            raise ConfigError(
                f"only {total_sources} sources would be generated, but at least {required} "
                "are needed to satisfy splits.min_sources_per_split for train/val/test"
            )

    @property
    def n_sources(self) -> int:
        """Number of clean source images (one split group each)."""
        return len(self.fonts.keys) * self.dataset.sources_per_font

    @property
    def n_samples(self) -> int:
        """Number of manifest rows: clean sample plus corrupted variants per source."""
        return self.n_sources * (1 + self.families.n_variants_per_source)

    @property
    def generation_config(self) -> dict[str, Any]:
        """Configuration fields that define generated content and its frozen split policy.

        ``output_dir`` and ``[diagnostics]`` are excluded so that moving the
        output directory or retuning the analysis thresholds does not change
        source/sample identifiers or the generation hash.
        """
        return {
            "dataset": {
                "name": self.dataset.name,
                "version": self.dataset.version,
                "seed": self.dataset.seed,
                "sources_per_font": self.dataset.sources_per_font,
                "text_lengths": list(self.dataset.text_lengths),
                "text_length_weights": list(self.dataset.text_length_weights),
                "charset": self.dataset.charset,
            },
            "render": {
                "canvas": list(self.render.canvas),
                "fit_fraction": self.render.fit_fraction,
                "vertical_padding": self.render.vertical_padding,
                "jitter_fraction": self.render.jitter_fraction,
                "spacing_jitter_fraction": self.render.spacing_jitter_fraction,
                "placement_jitter_fraction": self.render.placement_jitter_fraction,
                "background_mode": self.render.background_mode,
                "background_range": list(self.render.background_range),
                "center_ink": self.render.center_ink,
                "strict_bounds": self.render.strict_bounds,
            },
            "fonts": {"keys": list(self.fonts.keys), "min_required": self.fonts.min_required},
            "families": {
                ladder.family: {
                    "unit": ladder.unit,
                    "levels": list(ladder.parameters),
                }
                for ladder in self.families.ladders
            },
            "splits": {
                "ratios": [[name, ratio] for name, ratio in self.splits.ratios],
                "seed": self.splits.seed,
                "min_sources_per_split": self.splits.min_sources_per_split,
                "stratify_by_font": self.splits.stratify_by_font,
            },
        }

    @property
    def generation_config_hash(self) -> str:
        """Stable SHA-256 of the content-defining configuration."""
        return sha256_bytes(canonical_json(self.generation_config).encode("utf-8"))

    def describe(self) -> str:
        """One-line human-readable summary for CLI output."""
        return (
            f"{self.dataset.name} {self.dataset.version}: {self.n_sources} sources, "
            f"{self.n_samples} samples, {len(self.fonts.keys)} fonts, "
            f"families={self.families.names}, config_hash={self.generation_config_hash[:12]}"
        )


DATASET_KEYS: Final[set[str]] = {
    "name",
    "version",
    "seed",
    "output_dir",
    "sources_per_font",
    "text_lengths",
    "text_length_weights",
    "charset",
}
RENDER_KEYS: Final[set[str]] = {
    "canvas_width",
    "canvas_height",
    "fit_fraction",
    "vertical_padding",
    "jitter_fraction",
    "spacing_jitter_fraction",
    "placement_jitter_fraction",
    "background_mode",
    "background_min",
    "background_max",
    "center_ink",
    "strict_bounds",
}
FONT_KEYS: Final[set[str]] = {"keys", "min_required"}
SPLIT_KEYS: Final[set[str]] = {
    "train",
    "val",
    "test",
    "seed",
    "min_sources_per_split",
    "stratify_by_font",
}
DIAGNOSTIC_KEYS: Final[set[str]] = {
    "trivial_single_feature_r2",
    "strong_confound_r2",
    "min_group_rows",
}
ROOT_KEYS: Final[set[str]] = {
    "dataset",
    "render",
    "fonts",
    "families",
    "splits",
    "diagnostics",
}


def _parse_dataset(raw: dict[str, Any]) -> DatasetSettings:
    section = _section(raw, "dataset")
    _check_keys(section, DATASET_KEYS, "dataset")
    lengths = tuple(
        _as_positive_int(item, "dataset.text_lengths[]")
        for item in _as_sequence(
            _required(section, "text_lengths", "dataset"), "dataset.text_lengths"
        )
    )
    weights = tuple(
        _as_float(item, "dataset.text_length_weights[]")
        for item in _as_sequence(
            _required(section, "text_length_weights", "dataset"), "dataset.text_length_weights"
        )
    )
    return DatasetSettings(
        name=_as_str(_required(section, "name", "dataset"), "dataset.name"),
        version=_as_str(_required(section, "version", "dataset"), "dataset.version"),
        seed=_as_int(_required(section, "seed", "dataset"), "dataset.seed"),
        output_dir=Path(_as_str(_required(section, "output_dir", "dataset"), "dataset.output_dir")),
        sources_per_font=_as_int(
            _required(section, "sources_per_font", "dataset"), "dataset.sources_per_font"
        ),
        text_lengths=lengths,
        text_length_weights=weights,
        charset=_as_str(_required(section, "charset", "dataset"), "dataset.charset"),
    )


def _parse_render(raw: dict[str, Any]) -> RenderSettings:
    section = _section(raw, "render")
    _check_keys(section, RENDER_KEYS, "render")
    canvas = (
        _as_positive_int(_required(section, "canvas_width", "render"), "render.canvas_width"),
        _as_positive_int(_required(section, "canvas_height", "render"), "render.canvas_height"),
    )
    return RenderSettings(
        canvas=canvas,
        fit_fraction=_as_float(_required(section, "fit_fraction", "render"), "render.fit_fraction"),
        vertical_padding=_as_float(
            _required(section, "vertical_padding", "render"), "render.vertical_padding"
        ),
        jitter_fraction=_as_float(
            _required(section, "jitter_fraction", "render"), "render.jitter_fraction"
        ),
        spacing_jitter_fraction=_as_float(
            _required(section, "spacing_jitter_fraction", "render"),
            "render.spacing_jitter_fraction",
        ),
        placement_jitter_fraction=_as_float(
            _required(section, "placement_jitter_fraction", "render"),
            "render.placement_jitter_fraction",
        ),
        background_mode=_as_str(
            _required(section, "background_mode", "render"), "render.background_mode"
        ),
        background_range=(
            _as_int(_required(section, "background_min", "render"), "render.background_min"),
            _as_int(_required(section, "background_max", "render"), "render.background_max"),
        ),
        center_ink=_as_bool(_required(section, "center_ink", "render"), "render.center_ink"),
        strict_bounds=_as_bool(
            _required(section, "strict_bounds", "render"), "render.strict_bounds"
        ),
    )


def _parse_fonts(raw: dict[str, Any]) -> FontSettings:
    section = _section(raw, "fonts")
    _check_keys(section, FONT_KEYS, "fonts")
    return FontSettings(
        keys=_as_str_tuple(_required(section, "keys", "fonts"), "fonts.keys"),
        min_required=_as_int(_required(section, "min_required", "fonts"), "fonts.min_required"),
    )


def _parse_families(raw: dict[str, Any]) -> FamilySettings:
    section = _section(raw, "families")
    names = _as_str_tuple(_required(section, "names", "families"), "families.names")
    for name in names:
        if name not in SUPPORTED_FAMILIES:
            raise ConfigError(
                f"families.names contains unsupported family {name!r}; "
                f"supported: {SUPPORTED_FAMILIES}"
            )
    _check_keys(section, {"names", *names}, "families")
    ladders: list[SeverityLadder] = []
    for name in names:
        family_section = _section(section, name)
        _check_keys(family_section, {"unit", "levels"}, f"families.{name}")
        levels = tuple(
            _as_float(item, f"families.{name}.levels[]")
            for item in _as_sequence(
                _required(family_section, "levels", f"families.{name}"),
                f"families.{name}.levels",
            )
        )
        unit = _as_str(
            _required(family_section, "unit", f"families.{name}"), f"families.{name}.unit"
        )
        ladders.append(SeverityLadder(family=name, parameters=levels, unit=unit))
    return FamilySettings(ladders=tuple(ladders))


def _parse_splits(raw: dict[str, Any]) -> SplitSettings:
    section = _section(raw, "splits")
    _check_keys(section, SPLIT_KEYS, "splits")
    ratios = tuple(
        (name, _as_float(_required(section, name, "splits"), f"splits.{name}"))
        for name in SPLIT_NAMES
    )
    return SplitSettings(
        ratios=ratios,
        seed=_as_int(_required(section, "seed", "splits"), "splits.seed"),
        min_sources_per_split=_as_int(
            _required(section, "min_sources_per_split", "splits"),
            "splits.min_sources_per_split",
        ),
        stratify_by_font=_as_bool(
            _required(section, "stratify_by_font", "splits"), "splits.stratify_by_font"
        ),
    )


def _parse_diagnostics(raw: dict[str, Any]) -> DiagnosticsSettings:
    section = _section(raw, "diagnostics")
    _check_keys(section, DIAGNOSTIC_KEYS, "diagnostics")
    return DiagnosticsSettings(
        trivial_single_feature_r2=_as_float(
            _required(section, "trivial_single_feature_r2", "diagnostics"),
            "diagnostics.trivial_single_feature_r2",
        ),
        strong_confound_r2=_as_float(
            _required(section, "strong_confound_r2", "diagnostics"),
            "diagnostics.strong_confound_r2",
        ),
        min_group_rows=_as_int(
            _required(section, "min_group_rows", "diagnostics"), "diagnostics.min_group_rows"
        ),
    )


def load_config(path: Path | str) -> DatasetConfig:
    """Load and validate a dataset configuration from ``path``.

    Relative ``dataset.output_dir`` values are resolved against the current
    working directory, which is the repository root in the documented commands.

    Raises
    ------
    ConfigError
        If the file is missing, not valid TOML, or fails any validation rule.
    """
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root in {config_path} must be a table")
    _check_keys(raw, ROOT_KEYS, "root")

    return DatasetConfig(
        dataset=_parse_dataset(raw),
        render=_parse_render(raw),
        fonts=_parse_fonts(raw),
        families=_parse_families(raw),
        splits=_parse_splits(raw),
        diagnostics=_parse_diagnostics(raw),
        config_path=config_path,
    )
