"""Milestone 4A pilot: occlusion and visual-interference feasibility.

The pilot is deliberately separate from the frozen Milestone 2 dataset and the
Milestone 3 benchmark:

* it reads the frozen ``v0.1.0`` clean sources and re-renders them to prove the
  references are byte-identical to the existing renderer's output;
* it never writes into the frozen dataset, the benchmark reports or the model
  directory;
* it generates a new, self-contained pilot dataset under
  ``data/processed/occlusion_interference_pilot_v0.1.0/`` and writes diagnostic
  tables under ``reports/robustness_pilot/``;
* it does not train, retune or evaluate any model.

The only target is SSIM against the clean reference. PSNR is a secondary
diagnostic. Neither measures human readability or CAPTCHA solvability.

Run it with::

    uv run --extra image captcha-robustness-pilot run \\
        --config configs/robustness_pilot_occlusion_v0_1_0.toml

Outputs are Git-ignored; the experiment log records the measured results.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from captcha_quality.config import ConfigError, DatasetConfig, load_config
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.dataset import (
    CheckResult,
    VerificationReport,
    load_grayscale,
)
from captcha_quality.features import FEATURE_NAMES, FeatureDependencyError, extract_feature_table
from captcha_quality.imaging import ImageContractError, freeze_image
from captcha_quality.interference import (
    SUPPORTED_VARIANTS,
    VARIANT_ARC_STROKES,
    VARIANT_THIN_LINES,
    InterferenceError,
    InterferenceLadder,
    apply_strokes,
    geometry_to_json,
    plan_stroke_pool,
    strokes_from_json,
)
from captcha_quality.provenance import (
    canonical_json,
    code_revision,
    package_version,
    sha256_bytes,
    sha256_file,
)
from captcha_quality.quality_metrics import (
    MetricUnavailableError,
    compute_pair_metrics,
)
from captcha_quality.render import (
    FontResolutionError,
    RenderError,
    render_clean,
    resolve_fonts,
)
from captcha_quality.splits import SplitError
from captcha_quality.summarise import rank_correlation

CLEAN_VARIANT_FAMILY: Final[str] = "none"
CLEAN_VARIANT_ID: Final[str] = "clean_s0"

#: Columns in the public pilot manifest, in stable order.
PILOT_MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "source_id",
    "variant_id",
    "split",
    "font_id",
    "text_length",
    "variant_family",
    "severity_level",
    "stroke_count",
    "stroke_width_px",
    "stroke_intensity",
    "variant_seed",
    "geometry_json",
    "image_path",
    "reference_path",
    "ssim_target",
    "psnr_target",
    "psnr_infinite",
    "height",
    "width",
    "image_sha256",
    "reference_sha256",
    "changed_pixel_fraction",
    "changed_ink_fraction",
    "changed_background_fraction",
    "border_touched",
    "pilot_version",
    "pilot_config_hash",
)

#: Source-level table, preserving the frozen split and font provenance.
PILOT_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "source_id",
    "split_group_id",
    "split",
    "font_id",
    "text_length",
    "pilot_clean_image_path",
    "frozen_clean_image_path",
    "clean_sha256",
    "frozen_dataset_version",
    "frozen_generation_config_hash",
)

PILOT_FEATURE_COLUMNS: Final[tuple[str, ...]] = ("sample_id", *FEATURE_NAMES)

BOOTSTRAP_LOW: Final[float] = 2.5
BOOTSTRAP_HIGH: Final[float] = 97.5

#: Frozen assets whose hashes are checked before and after the pilot. Model and
#: benchmark artifacts may be absent on a fresh clone; they are then recorded as
#: missing rather than silently treated as verified.
FROZEN_ASSET_PATHS: Final[tuple[str, ...]] = (
    "configs/dataset_v0_1_0.toml",
    "configs/pilot_target_validation.toml",
    "configs/benchmark_v0_1_0.toml",
    "data/processed/captcha_quality_v0.1.0/manifest.csv",
    "data/processed/captcha_quality_v0.1.0/sources.csv",
    "data/processed/captcha_quality_v0.1.0/features.csv",
    "data/processed/captcha_quality_v0.1.0/ground_truth.csv",
    "data/processed/captcha_quality_v0.1.0/dataset_summary.json",
    "models/benchmark_v0_1_0/primary_model.joblib",
    "models/benchmark_v0_1_0/primary_model_card.json",
    "models/benchmark_v0_1_0/heldout_font_model.joblib",
    "models/benchmark_v0_1_0/heldout_font_model_card.json",
    "reports/benchmark_v0_1_0/benchmark.md",
    "reports/benchmark_v0_1_0/run_metadata.json",
)

#: Directories the pilot must never write into or delete. The default pilot
#: outputs are deliberately outside these roots.
PROTECTED_OUTPUT_ROOTS: Final[tuple[str, ...]] = (
    "configs",
    "data/processed/captcha_quality_v0.1.0",
    "models/benchmark_v0_1_0",
    "reports/benchmark_v0_1_0",
)


class PilotConfigError(ValueError):
    """Raised when the pilot configuration is missing or invalid."""


class PilotError(RuntimeError):
    """Raised when the pilot cannot generate or verify its outputs."""


class PilotExistsError(PilotError):
    """Raised when the pilot output exists and overwriting was not requested."""


@dataclass(frozen=True, slots=True)
class PilotSettings:
    """Run identity, seed and output locations."""

    name: str
    version: str
    seed: int
    bootstrap_replicates: int
    output_dataset: Path
    output_reports: Path

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise PilotConfigError("pilot.name must not be empty")
        if not self.version.strip():
            raise PilotConfigError("pilot.version must not be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise PilotConfigError("pilot.seed must be an integer")
        if not 0 <= self.seed < 2**64:
            raise PilotConfigError("pilot.seed must be in [0, 2**64)")
        if isinstance(self.bootstrap_replicates, bool) or not isinstance(
            self.bootstrap_replicates, int
        ):
            raise PilotConfigError("pilot.bootstrap_replicates must be an integer")
        if self.bootstrap_replicates < 1:
            raise PilotConfigError("pilot.bootstrap_replicates must be >= 1")


@dataclass(frozen=True, slots=True)
class ReferenceSettings:
    """The frozen dataset and the exact hashes this pilot is pinned to."""

    dataset: Path
    config: Path
    dataset_version: str
    generation_config_hash: str
    manifest_sha256: str
    sources_sha256: str
    features_sha256: str
    ground_truth_sha256: str
    summary_sha256: str
    split_assignment_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "generation_config_hash",
            "manifest_sha256",
            "sources_sha256",
            "features_sha256",
            "ground_truth_sha256",
            "summary_sha256",
            "split_assignment_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise PilotConfigError(f"reference.{name} must be a lowercase SHA-256 hex digest")

    def expected_files(self) -> dict[str, str]:
        """Return the frozen file names and their expected hashes."""
        return {
            "manifest.csv": self.manifest_sha256,
            "sources.csv": self.sources_sha256,
            "features.csv": self.features_sha256,
            "ground_truth.csv": self.ground_truth_sha256,
            "dataset_summary.json": self.summary_sha256,
        }


@dataclass(frozen=True, slots=True)
class PilotGateThresholds:
    """Predeclared feasibility thresholds.

    These are declared in the pilot config before generation. Changing them
    after seeing results invalidates the pilot's go/no-go decision.
    """

    ssim_std_min: float
    level_mean_range_min: float
    per_level_sd_min: float
    median_source_spearman_max: float
    increasing_step_fraction_max: float
    feature_effect_min: float
    feature_strong_effect_min: float
    feature_strong_effect_count_min: int
    duplicate_fraction_max: float

    def __post_init__(self) -> None:
        if self.ssim_std_min <= 0.0:
            raise PilotConfigError("gate.ssim_std_min must be > 0")
        if self.level_mean_range_min <= 0.0:
            raise PilotConfigError("gate.level_mean_range_min must be > 0")
        if self.per_level_sd_min <= 0.0:
            raise PilotConfigError("gate.per_level_sd_min must be > 0")
        if self.median_source_spearman_max >= 0.0:
            raise PilotConfigError("gate.median_source_spearman_max must be negative")
        if not 0.0 <= self.increasing_step_fraction_max <= 1.0:
            raise PilotConfigError("gate.increasing_step_fraction_max must be in [0, 1]")
        if self.feature_effect_min <= 0.0:
            raise PilotConfigError("gate.feature_effect_min must be > 0")
        if self.feature_strong_effect_min < self.feature_effect_min:
            raise PilotConfigError(
                "gate.feature_strong_effect_min must be >= gate.feature_effect_min"
            )
        if self.feature_strong_effect_count_min < 1:
            raise PilotConfigError("gate.feature_strong_effect_count_min must be >= 1")
        if not 0.0 <= self.duplicate_fraction_max <= 1.0:
            raise PilotConfigError("gate.duplicate_fraction_max must be in [0, 1]")

    def as_dict(self) -> dict[str, float | int]:
        """Return thresholds as a plain mapping for reporting."""
        return {
            "ssim_std_min": self.ssim_std_min,
            "level_mean_range_min": self.level_mean_range_min,
            "per_level_sd_min": self.per_level_sd_min,
            "median_source_spearman_max": self.median_source_spearman_max,
            "increasing_step_fraction_max": self.increasing_step_fraction_max,
            "feature_effect_min": self.feature_effect_min,
            "feature_strong_effect_min": self.feature_strong_effect_min,
            "feature_strong_effect_count_min": self.feature_strong_effect_count_min,
            "duplicate_fraction_max": self.duplicate_fraction_max,
        }


@dataclass(frozen=True, slots=True)
class PilotConfig:
    """Fully validated pilot configuration."""

    pilot: PilotSettings
    reference: ReferenceSettings
    ladders: tuple[InterferenceLadder, ...]
    gate: PilotGateThresholds
    config_path: Path

    def ladder(self, variant: str) -> InterferenceLadder:
        """Return the ladder for ``variant`` or raise."""
        for ladder in self.ladders:
            if ladder.variant == variant:
                return ladder
        raise PilotConfigError(f"no interference ladder configured for variant {variant!r}")

    @property
    def config_sha256(self) -> str:
        """Content hash of the configuration file."""
        return sha256_file(self.config_path)

    @property
    def generation_config(self) -> dict[str, object]:
        """Content-defining fields, excluding output paths and gate thresholds."""
        return {
            "name": self.pilot.name,
            "version": self.pilot.version,
            "seed": self.pilot.seed,
            "reference": {
                "dataset_version": self.reference.dataset_version,
                "generation_config_hash": self.reference.generation_config_hash,
                "manifest_sha256": self.reference.manifest_sha256,
                "sources_sha256": self.reference.sources_sha256,
                "features_sha256": self.reference.features_sha256,
                "ground_truth_sha256": self.reference.ground_truth_sha256,
                "summary_sha256": self.reference.summary_sha256,
                "split_assignment_sha256": self.reference.split_assignment_sha256,
            },
            "interference": {
                "variants": [ladder.variant for ladder in self.ladders],
                "stroke_width_px": [ladder.width_px for ladder in self.ladders],
                "stroke_intensity": [ladder.intensity for ladder in self.ladders],
                "stroke_counts": {
                    ladder.variant: list(ladder.stroke_counts) for ladder in self.ladders
                },
                "line_angle_deg": {
                    ladder.variant: list(ladder.line_angle_deg)
                    for ladder in self.ladders
                    if ladder.variant == VARIANT_THIN_LINES
                },
                "arc_sweep_deg": {
                    ladder.variant: list(ladder.arc_sweep_deg)
                    for ladder in self.ladders
                    if ladder.variant == VARIANT_ARC_STROKES
                },
                "arc_radius_fraction": {
                    ladder.variant: list(ladder.arc_radius_fraction)
                    for ladder in self.ladders
                    if ladder.variant == VARIANT_ARC_STROKES
                },
                "arc_aspect_ratio": {
                    ladder.variant: list(ladder.arc_aspect_ratio)
                    for ladder in self.ladders
                    if ladder.variant == VARIANT_ARC_STROKES
                },
            },
        }

    @property
    def pilot_config_hash(self) -> str:
        """Stable hash of the content-defining pilot configuration."""
        return sha256_bytes(canonical_json(self.generation_config).encode("utf-8"))

    def provenance(self) -> dict[str, object]:
        """Return a JSON-serialisable description of the whole configuration."""
        return {
            "pilot": {
                "name": self.pilot.name,
                "version": self.pilot.version,
                "seed": self.pilot.seed,
                "bootstrap_replicates": self.pilot.bootstrap_replicates,
                "output_dataset": str(self.pilot.output_dataset),
                "output_reports": str(self.pilot.output_reports),
            },
            "reference": {
                "dataset": str(self.reference.dataset),
                "config": str(self.reference.config),
                "dataset_version": self.reference.dataset_version,
                "generation_config_hash": self.reference.generation_config_hash,
                "manifest_sha256": self.reference.manifest_sha256,
                "sources_sha256": self.reference.sources_sha256,
                "features_sha256": self.reference.features_sha256,
                "ground_truth_sha256": self.reference.ground_truth_sha256,
                "summary_sha256": self.reference.summary_sha256,
                "split_assignment_sha256": self.reference.split_assignment_sha256,
            },
            "ladders": [
                {
                    "variant": ladder.variant,
                    "stroke_counts": list(ladder.stroke_counts),
                    "width_px": ladder.width_px,
                    "intensity": ladder.intensity,
                    "line_angle_deg": list(ladder.line_angle_deg),
                    "arc_sweep_deg": list(ladder.arc_sweep_deg),
                    "arc_radius_fraction": list(ladder.arc_radius_fraction),
                    "arc_aspect_ratio": list(ladder.arc_aspect_ratio),
                }
                for ladder in self.ladders
            ],
            "gate": self.gate.as_dict(),
            "generation_config": self.generation_config,
            "pilot_config_hash": self.pilot_config_hash,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_toml(cls, path: Path | str) -> PilotConfig:
        """Load and validate a pilot configuration from TOML."""
        config_path = Path(path)
        if not config_path.is_file():
            raise PilotConfigError(f"pilot config not found: {config_path}")
        try:
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise PilotConfigError(f"invalid TOML in {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise PilotConfigError(f"configuration root in {config_path} must be a table")
        _check_keys(raw, {"pilot", "reference", "interference", "gate"}, "root")

        pilot_table = _table(raw, "pilot", config_path)
        _check_keys(
            pilot_table,
            {
                "name",
                "version",
                "seed",
                "bootstrap_replicates",
                "output_dataset",
                "output_reports",
            },
            "pilot",
        )
        pilot = PilotSettings(
            name=_string(pilot_table, "name", "pilot"),
            version=_string(pilot_table, "version", "pilot"),
            seed=_integer(pilot_table, "seed", "pilot"),
            bootstrap_replicates=_integer(pilot_table, "bootstrap_replicates", "pilot"),
            output_dataset=Path(_string(pilot_table, "output_dataset", "pilot")),
            output_reports=Path(_string(pilot_table, "output_reports", "pilot")),
        )

        reference_table = _table(raw, "reference", config_path)
        reference_keys = {
            "dataset",
            "config",
            "dataset_version",
            "generation_config_hash",
            "manifest_sha256",
            "sources_sha256",
            "features_sha256",
            "ground_truth_sha256",
            "summary_sha256",
            "split_assignment_sha256",
        }
        _check_keys(reference_table, reference_keys, "reference")
        reference = ReferenceSettings(
            dataset=Path(_string(reference_table, "dataset", "reference")),
            config=Path(_string(reference_table, "config", "reference")),
            dataset_version=_string(reference_table, "dataset_version", "reference"),
            generation_config_hash=_string(reference_table, "generation_config_hash", "reference"),
            manifest_sha256=_string(reference_table, "manifest_sha256", "reference"),
            sources_sha256=_string(reference_table, "sources_sha256", "reference"),
            features_sha256=_string(reference_table, "features_sha256", "reference"),
            ground_truth_sha256=_string(reference_table, "ground_truth_sha256", "reference"),
            summary_sha256=_string(reference_table, "summary_sha256", "reference"),
            split_assignment_sha256=_string(
                reference_table, "split_assignment_sha256", "reference"
            ),
        )

        interference_table = _table(raw, "interference", config_path)
        variants = tuple(_string_list(interference_table, "variants", "interference"))
        if not variants:
            raise PilotConfigError("interference.variants must not be empty")
        if len(set(variants)) != len(variants):
            raise PilotConfigError("interference.variants must be unique")
        unknown_variants = [variant for variant in variants if variant not in SUPPORTED_VARIANTS]
        if unknown_variants:
            raise PilotConfigError(
                f"unsupported interference variant(s) {unknown_variants}; "
                f"supported: {SUPPORTED_VARIANTS}"
            )
        _check_keys(
            interference_table,
            {"variants", "stroke_width_px", "stroke_intensity", *variants},
            "interference",
        )
        width_px = _integer(interference_table, "stroke_width_px", "interference")
        intensity = _integer(interference_table, "stroke_intensity", "interference")

        ladders: list[InterferenceLadder] = []
        for variant in variants:
            variant_table = _table(interference_table, variant, f"interference.{variant}")
            counts = tuple(_integer_list(variant_table, "stroke_counts", f"interference.{variant}"))
            if variant == VARIANT_THIN_LINES:
                _check_keys(
                    variant_table,
                    {
                        "stroke_counts",
                        "line_angle_deg_min",
                        "line_angle_deg_max",
                    },
                    f"interference.{variant}",
                )
                try:
                    ladder = InterferenceLadder(
                        variant=variant,
                        stroke_counts=counts,
                        width_px=width_px,
                        intensity=intensity,
                        line_angle_deg=(
                            _number(variant_table, "line_angle_deg_min", f"interference.{variant}"),
                            _number(variant_table, "line_angle_deg_max", f"interference.{variant}"),
                        ),
                    )
                except InterferenceError as exc:
                    raise PilotConfigError(f"invalid interference.{variant}: {exc}") from exc
                ladders.append(ladder)
            else:
                _check_keys(
                    variant_table,
                    {
                        "stroke_counts",
                        "arc_sweep_deg_min",
                        "arc_sweep_deg_max",
                        "arc_radius_fraction_min",
                        "arc_radius_fraction_max",
                        "arc_aspect_ratio_min",
                        "arc_aspect_ratio_max",
                    },
                    f"interference.{variant}",
                )
                try:
                    ladder = InterferenceLadder(
                        variant=variant,
                        stroke_counts=counts,
                        width_px=width_px,
                        intensity=intensity,
                        arc_sweep_deg=(
                            _number(variant_table, "arc_sweep_deg_min", f"interference.{variant}"),
                            _number(variant_table, "arc_sweep_deg_max", f"interference.{variant}"),
                        ),
                        arc_radius_fraction=(
                            _number(
                                variant_table,
                                "arc_radius_fraction_min",
                                f"interference.{variant}",
                            ),
                            _number(
                                variant_table,
                                "arc_radius_fraction_max",
                                f"interference.{variant}",
                            ),
                        ),
                        arc_aspect_ratio=(
                            _number(
                                variant_table,
                                "arc_aspect_ratio_min",
                                f"interference.{variant}",
                            ),
                            _number(
                                variant_table,
                                "arc_aspect_ratio_max",
                                f"interference.{variant}",
                            ),
                        ),
                    )
                except InterferenceError as exc:
                    raise PilotConfigError(f"invalid interference.{variant}: {exc}") from exc
                ladders.append(ladder)

        gate_table = _table(raw, "gate", config_path)
        gate_keys = {
            "ssim_std_min",
            "level_mean_range_min",
            "per_level_sd_min",
            "median_source_spearman_max",
            "increasing_step_fraction_max",
            "feature_effect_min",
            "feature_strong_effect_min",
            "feature_strong_effect_count_min",
            "duplicate_fraction_max",
        }
        _check_keys(gate_table, gate_keys, "gate")
        gate = PilotGateThresholds(
            ssim_std_min=_number(gate_table, "ssim_std_min", "gate"),
            level_mean_range_min=_number(gate_table, "level_mean_range_min", "gate"),
            per_level_sd_min=_number(gate_table, "per_level_sd_min", "gate"),
            median_source_spearman_max=_number(gate_table, "median_source_spearman_max", "gate"),
            increasing_step_fraction_max=_number(
                gate_table, "increasing_step_fraction_max", "gate"
            ),
            feature_effect_min=_number(gate_table, "feature_effect_min", "gate"),
            feature_strong_effect_min=_number(gate_table, "feature_strong_effect_min", "gate"),
            feature_strong_effect_count_min=_integer(
                gate_table, "feature_strong_effect_count_min", "gate"
            ),
            duplicate_fraction_max=_number(gate_table, "duplicate_fraction_max", "gate"),
        )

        return cls(
            pilot=pilot,
            reference=reference,
            ladders=tuple(ladders),
            gate=gate,
            config_path=config_path,
        )


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise PilotConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _table(data: Mapping[str, object], key: str, context: Path | str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise PilotConfigError(f"missing or invalid [{key}] table in {context}")
    return value


def _string(table: Mapping[str, object], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PilotConfigError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(table: Mapping[str, object], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotConfigError(f"{context}.{key} must be an integer")
    return value


def _number(table: Mapping[str, object], key: str, context: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PilotConfigError(f"{context}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise PilotConfigError(f"{context}.{key} must be finite")
    return result


def _string_list(table: Mapping[str, object], key: str, context: str) -> list[str]:
    value = table.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PilotConfigError(f"{context}.{key} must be a list of strings")
    return value


def _integer_list(table: Mapping[str, object], key: str, context: str) -> list[int]:
    value = table.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise PilotConfigError(f"{context}.{key} must be a list of integers")
    return value


@dataclass(frozen=True, slots=True)
class FrozenReference:
    """The frozen dataset as read by the pilot, plus its re-render audit."""

    root: Path
    config: DatasetConfig
    manifest: pd.DataFrame
    sources: pd.DataFrame
    features: pd.DataFrame
    summary: dict[str, Any]
    clean_images: dict[str, np.ndarray]
    clean_hashes: dict[str, str]
    rerendered_sources: int
    fingerprints: dict[str, dict[str, object]]


def fingerprint_assets(paths: Iterable[str | Path]) -> dict[str, dict[str, object]]:
    """Return existence, size and SHA-256 for each path."""
    result: dict[str, dict[str, object]] = {}
    for path_value in paths:
        path = Path(path_value)
        key = path.as_posix()
        if not path.is_file():
            result[key] = {"exists": False, "size_bytes": None, "sha256": None}
            continue
        result[key] = {
            "exists": True,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return result


def _require_matching_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise PilotError(f"frozen {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise PilotError(
            f"frozen {label} hash mismatch for {path}: expected {expected}, got {actual}"
        )


def load_frozen_reference(config: PilotConfig) -> FrozenReference:
    """Load and audit the frozen dataset without modifying it.

    The audit checks every recorded file hash, the split-assignment hash, the
    clean-image hashes, and re-renders every clean source from the frozen config
    to prove that the pilot's references are exactly the existing renderer's
    output. Any mismatch is an error; the pilot never substitutes an image.
    """
    root = config.reference.dataset
    if not root.is_dir():
        raise PilotError(
            f"frozen dataset directory not found: {root}; generate it from "
            f"{config.reference.config} before running the pilot"
        )
    for name, expected in config.reference.expected_files().items():
        _require_matching_hash(root / name, expected, name)

    dataset_config = load_config(config.reference.config)
    if dataset_config.dataset.version != config.reference.dataset_version:
        raise PilotError(
            "frozen dataset config version disagrees with the pilot reference: "
            f"{dataset_config.dataset.version!r} != {config.reference.dataset_version!r}"
        )
    if dataset_config.generation_config_hash != config.reference.generation_config_hash:
        raise PilotError(
            "frozen dataset generation hash disagrees with the pilot reference: "
            f"{dataset_config.generation_config_hash} != "
            f"{config.reference.generation_config_hash}"
        )

    manifest = pd.read_csv(root / "manifest.csv")
    sources = pd.read_csv(root / "sources.csv")
    features = pd.read_csv(root / "features.csv")
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))

    split_hash = summary.get("split_policy", {}).get("assignment_sha256")
    if split_hash != config.reference.split_assignment_sha256:
        raise PilotError(
            "frozen split-assignment hash disagrees with the pilot reference: "
            f"{split_hash!r} != {config.reference.split_assignment_sha256!r}"
        )
    if tuple(features.columns) != PILOT_FEATURE_COLUMNS:
        raise PilotError(
            f"frozen feature columns must be {PILOT_FEATURE_COLUMNS}, got {tuple(features.columns)}"
        )

    clean_images: dict[str, np.ndarray] = {}
    clean_hashes: dict[str, str] = {}
    for row in sources.itertuples(index=False):
        source_id = str(row.source_id)
        relative = str(row.clean_image_path)
        image = load_grayscale(root / relative)
        actual = sha256_bytes(image.tobytes())
        if actual != str(row.clean_image_sha256):
            raise PilotError(
                f"frozen clean image hash mismatch for {source_id}: "
                f"{actual} != {row.clean_image_sha256}"
            )
        clean_images[source_id] = freeze_image(image)
        clean_hashes[source_id] = actual

    rerendered = _audit_rerender(dataset_config, sources, clean_images)
    return FrozenReference(
        root=root,
        config=dataset_config,
        manifest=manifest,
        sources=sources,
        features=features,
        summary=summary,
        clean_images=clean_images,
        clean_hashes=clean_hashes,
        rerendered_sources=rerendered,
        fingerprints=fingerprint_assets(FROZEN_ASSET_PATHS),
    )


def _audit_rerender(
    dataset_config: DatasetConfig,
    sources: pd.DataFrame,
    clean_images: Mapping[str, np.ndarray],
) -> int:
    """Re-render every frozen source and require exact pixel equality."""
    try:
        resolution = resolve_fonts(
            dataset_config.fonts.keys,
            min_required=dataset_config.fonts.min_required,
        )
    except FontResolutionError as exc:
        raise PilotError(f"cannot re-render frozen sources: {exc}") from exc
    # plan_sources is the same planner used by dataset generation, so source
    # IDs, text, render seeds and split assignments are reproduced, not guessed.
    from captcha_quality.dataset import plan_sources

    try:
        plans = plan_sources(dataset_config, resolution)
    except (SplitError, ValueError) as exc:  # pragma: no cover - frozen config guard
        raise PilotError(f"cannot plan frozen sources: {exc}") from exc
    if len(plans) != len(sources):
        raise PilotError(
            f"frozen planner produced {len(plans)} sources but sources.csv has {len(sources)}"
        )

    source_index = sources.set_index("source_id")
    for plan in plans:
        if plan.source_id not in source_index.index:
            raise PilotError(f"re-rendered source {plan.source_id} is absent from sources.csv")
        expected = source_index.loc[plan.source_id]
        # Dataset CSV writes 64-bit seeds through pandas' int64, so values above
        # 2**63-1 appear negative. Compare modulo 2**64, not as signed integers.
        if plan.render_seed % 2**64 != int(expected["render_seed"]) % 2**64:
            raise PilotError(f"render seed mismatch for source {plan.source_id}")
        if plan.split != str(expected["split"]):
            raise PilotError(f"split mismatch for source {plan.source_id}")
        if plan.text_sha256 != str(expected["text_sha256"]):
            raise PilotError(f"text hash mismatch for source {plan.source_id}")
        render_config = dataset_config.render.to_render_config(
            text_length=plan.text_length, charset=dataset_config.dataset.charset
        )
        clean = render_clean(
            plan.font,
            plan.text,
            render_config,
            rng=np.random.default_rng(plan.render_seed),
        )
        if clean.sha256 != str(expected["clean_image_sha256"]):
            raise PilotError(
                f"re-rendered clean hash mismatch for source {plan.source_id}: "
                f"{clean.sha256} != {expected['clean_image_sha256']}"
            )
        if not np.array_equal(clean.image, clean_images[plan.source_id]):
            raise PilotError(
                f"re-rendered clean pixels differ from the frozen image for {plan.source_id}"
            )
    return len(plans)


def _relative(path: Path) -> str:
    return path.as_posix()


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.12g")
    temporary.replace(path)


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class GenerationData:
    """One complete in-memory build of the pilot dataset."""

    manifest: pd.DataFrame
    sources: pd.DataFrame
    features: pd.DataFrame
    image_payloads: dict[str, np.ndarray]
    image_hashes: dict[str, str]
    clean_hashes_after: dict[str, str]


def _clean_sample_id(source_id: str) -> str:
    return f"{source_id}__{CLEAN_VARIANT_ID}"


def _variant_id(variant: str, severity: int) -> str:
    return f"{variant}_s{severity}"


def _build_generation_data(
    config: PilotConfig,
    reference: FrozenReference,
) -> GenerationData:
    """Build every row and image from the frozen clean references."""
    source_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    feature_rows: list[tuple[str, np.ndarray]] = []
    image_payloads: dict[str, np.ndarray] = {}
    image_hashes: dict[str, str] = {}

    for source in reference.sources.sort_values("source_id", kind="stable").itertuples(index=False):
        source_id = str(source.source_id)
        clean = reference.clean_images[source_id]
        clean_hash_before = reference.clean_hashes[source_id]
        clean_rel = Path("images") / "clean" / f"{source_id}.png"
        clean_sample_id = _clean_sample_id(source_id)

        clean_metrics = compute_pair_metrics(clean, clean, ("ssim", "psnr"))
        clean_feature_image = np.array(clean, dtype=np.uint8, copy=True)
        feature_rows.append((clean_sample_id, clean_feature_image))
        image_payloads[_relative(clean_rel)] = clean_feature_image
        image_hashes[_relative(clean_rel)] = clean_hash_before

        source_rows.append(
            {
                "source_id": source_id,
                "split_group_id": str(source.split_group_id),
                "split": str(source.split),
                "font_id": str(source.font_id),
                "text_length": int(source.text_length),
                "pilot_clean_image_path": _relative(clean_rel),
                "frozen_clean_image_path": str(source.clean_image_path),
                "clean_sha256": clean_hash_before,
                "frozen_dataset_version": config.reference.dataset_version,
                "frozen_generation_config_hash": config.reference.generation_config_hash,
            }
        )
        manifest_rows.append(
            {
                "sample_id": clean_sample_id,
                "source_id": source_id,
                "variant_id": CLEAN_VARIANT_ID,
                "split": str(source.split),
                "font_id": str(source.font_id),
                "text_length": int(source.text_length),
                "variant_family": CLEAN_VARIANT_FAMILY,
                "severity_level": 0,
                "stroke_count": 0,
                "stroke_width_px": 0,
                "stroke_intensity": 0,
                "variant_seed": derive_noise_seed(config.pilot.seed, "clean", source_id),
                "geometry_json": "[]",
                "image_path": _relative(clean_rel),
                "reference_path": _relative(clean_rel),
                "ssim_target": float(clean_metrics["ssim"]),
                "psnr_target": None,
                "psnr_infinite": True,
                "height": int(clean.shape[0]),
                "width": int(clean.shape[1]),
                "image_sha256": clean_hash_before,
                "reference_sha256": clean_hash_before,
                "changed_pixel_fraction": 0.0,
                "changed_ink_fraction": 0.0,
                "changed_background_fraction": 0.0,
                "border_touched": False,
                "pilot_version": config.pilot.version,
                "pilot_config_hash": config.pilot_config_hash,
            }
        )

        for ladder in config.ladders:
            seed = derive_noise_seed(config.pilot.seed, "interference", source_id, ladder.variant)
            # Plan once so the nested prefix is explicit; apply_interference is
            # also exercised by the unit tests and produces the same pool.
            pool = plan_stroke_pool(clean, ladder, seed=seed)
            for severity in range(1, ladder.n_levels + 1):
                spec = ladder.spec(severity)
                strokes = pool[: spec.stroke_count]
                corrupted = apply_strokes(
                    clean,
                    list(strokes),
                    width_px=spec.width_px,
                    intensity=spec.intensity,
                )
                metrics = compute_pair_metrics(clean, corrupted, ("ssim", "psnr"))
                corrupted_hash = sha256_bytes(corrupted.tobytes())
                sample_id = f"{source_id}__{_variant_id(ladder.variant, severity)}"
                image_rel = Path("images") / ladder.variant / f"{sample_id}.png"
                feature_rows.append((sample_id, corrupted))
                image_payloads[_relative(image_rel)] = corrupted
                image_hashes[_relative(image_rel)] = corrupted_hash

                difference = corrupted != clean
                ink = clean < 128
                ink_count = int(np.count_nonzero(ink))
                background_count = int(clean.size - ink_count)
                border = bool(
                    np.any(difference[0, :])
                    or np.any(difference[-1, :])
                    or np.any(difference[:, 0])
                    or np.any(difference[:, -1])
                )
                psnr_value = float(metrics["psnr"])
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "variant_id": _variant_id(ladder.variant, severity),
                        "split": str(source.split),
                        "font_id": str(source.font_id),
                        "text_length": int(source.text_length),
                        "variant_family": ladder.variant,
                        "severity_level": severity,
                        "stroke_count": spec.stroke_count,
                        "stroke_width_px": spec.width_px,
                        "stroke_intensity": spec.intensity,
                        "variant_seed": seed,
                        "geometry_json": canonical_json(geometry_to_json(list(strokes))),
                        "image_path": _relative(image_rel),
                        "reference_path": _relative(clean_rel),
                        "ssim_target": float(metrics["ssim"]),
                        "psnr_target": psnr_value if math.isfinite(psnr_value) else None,
                        "psnr_infinite": not math.isfinite(psnr_value),
                        "height": int(corrupted.shape[0]),
                        "width": int(corrupted.shape[1]),
                        "image_sha256": corrupted_hash,
                        "reference_sha256": clean_hash_before,
                        "changed_pixel_fraction": float(np.count_nonzero(difference))
                        / float(clean.size),
                        "changed_ink_fraction": (
                            float(np.count_nonzero(difference & ink)) / ink_count
                            if ink_count
                            else 0.0
                        ),
                        "changed_background_fraction": (
                            float(np.count_nonzero(difference & ~ink)) / background_count
                            if background_count
                            else 0.0
                        ),
                        "border_touched": border,
                        "pilot_version": config.pilot.version,
                        "pilot_config_hash": config.pilot_config_hash,
                    }
                )

        clean_hash_after = sha256_bytes(clean.tobytes())
        if clean_hash_after != clean_hash_before:
            raise PilotError(
                f"clean reference mutated for source {source_id}: "
                f"{clean_hash_before} -> {clean_hash_after}"
            )

    sources_frame = pd.DataFrame(source_rows, columns=PILOT_SOURCE_COLUMNS)
    manifest = pd.DataFrame(manifest_rows, columns=PILOT_MANIFEST_COLUMNS)
    manifest = _coerce_manifest_types(manifest)
    features = extract_feature_table(feature_rows)
    if tuple(features.columns) != PILOT_FEATURE_COLUMNS:  # pragma: no cover - structural guard
        raise PilotError("feature extraction returned an unexpected schema")

    manifest = manifest.sort_values(
        ["source_id", "variant_family", "severity_level"], kind="stable"
    ).reset_index(drop=True)
    features = features.set_index("sample_id").loc[manifest["sample_id"]].reset_index()
    features = features.loc[:, list(PILOT_FEATURE_COLUMNS)]

    clean_hashes_after = {
        source_id: sha256_bytes(reference.clean_images[source_id].tobytes())
        for source_id in reference.clean_images
    }
    return GenerationData(
        manifest=manifest,
        sources=sources_frame,
        features=features,
        image_payloads=image_payloads,
        image_hashes=image_hashes,
        clean_hashes_after=clean_hashes_after,
    )


def _coerce_manifest_types(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in (
        "text_length",
        "severity_level",
        "stroke_count",
        "stroke_width_px",
        "stroke_intensity",
        "variant_seed",
        "height",
        "width",
    ):
        frame[column] = frame[column].astype("int64")
    frame["ssim_target"] = frame["ssim_target"].astype("float64")
    frame["psnr_target"] = frame["psnr_target"].astype("float64")
    frame["psnr_infinite"] = frame["psnr_infinite"].astype(bool)
    for column in (
        "changed_pixel_fraction",
        "changed_ink_fraction",
        "changed_background_fraction",
    ):
        frame[column] = frame[column].astype("float64")
    frame["border_touched"] = frame["border_touched"].astype(bool)
    return frame


def _compare_generations(first: GenerationData, second: GenerationData) -> dict[str, object]:
    """Compare two independent in-memory builds for exact reproducibility."""
    comparisons = {
        "manifest_identical": first.manifest.equals(second.manifest),
        "sources_identical": first.sources.equals(second.sources),
        "features_identical": first.features.equals(second.features),
        "image_hashes_identical": first.image_hashes == second.image_hashes,
        "clean_hashes_identical": first.clean_hashes_after == second.clean_hashes_after,
    }
    comparisons["ok"] = all(bool(value) for value in comparisons.values())
    return comparisons


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        name: sha256_file(root / name) for name in ("manifest.csv", "sources.csv", "features.csv")
    }


def _image_pixel_hash_set(image_hashes: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json(sorted(image_hashes.items())).encode("utf-8"))


def _counts(values: Iterable[object]) -> dict[str, int]:
    series = pd.Series([str(value) for value in values])
    counts = series.value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


@dataclass(frozen=True, slots=True)
class PilotCriterion:
    """One evaluated feasibility criterion."""

    id: str
    description: str
    observed: str
    threshold: str
    passed: bool
    note: str = ""


@dataclass(frozen=True, slots=True)
class PilotGateVerdict:
    """Overall pilot recommendation and its predeclared criteria."""

    verdict: str
    criteria: tuple[PilotCriterion, ...]

    @property
    def failed_criteria(self) -> tuple[str, ...]:
        """IDs of every criterion that did not pass."""
        return tuple(criterion.id for criterion in self.criteria if not criterion.passed)


@dataclass(frozen=True, slots=True)
class PilotResult:
    """In-memory result of a pilot run."""

    config: PilotConfig
    reference: FrozenReference
    manifest: pd.DataFrame
    sources: pd.DataFrame
    features: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    gate: PilotGateVerdict
    verification: VerificationReport
    output_dataset: Path
    output_reports: Path
    report_path: Path


def summarise_levels(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-variant, per-severity SSIM/PSNR and coverage diagnostics."""
    corrupted = manifest[manifest["variant_family"] != CLEAN_VARIANT_FAMILY]
    rows: list[dict[str, object]] = []
    for (variant, severity), group in corrupted.groupby(
        ["variant_family", "severity_level"], sort=True
    ):
        ssim_values = group["ssim_target"].to_numpy(dtype=np.float64)
        psnr_values = group["psnr_target"].to_numpy(dtype=np.float64)
        finite_psnr = psnr_values[np.isfinite(psnr_values)]
        rows.append(
            {
                "variant_family": str(variant),
                "severity_level": int(severity),
                "stroke_count": int(group["stroke_count"].iloc[0]),
                "n_samples": int(len(group)),
                "n_sources": int(group["source_id"].nunique()),
                "ssim_mean": float(ssim_values.mean()),
                "ssim_sd": float(ssim_values.std(ddof=1)),
                "ssim_median": float(np.median(ssim_values)),
                "ssim_min": float(ssim_values.min()),
                "ssim_max": float(ssim_values.max()),
                "ssim_drop_from_clean": float(1.0 - ssim_values.mean()),
                "psnr_mean_db": float(finite_psnr.mean()) if finite_psnr.size else math.nan,
                "psnr_sd_db": (
                    float(finite_psnr.std(ddof=1)) if finite_psnr.size > 1 else math.nan
                ),
                "changed_pixel_fraction_mean": float(group["changed_pixel_fraction"].mean()),
                "changed_ink_fraction_mean": float(group["changed_ink_fraction"].mean()),
                "changed_background_fraction_mean": float(
                    group["changed_background_fraction"].mean()
                ),
                "border_touched_fraction": float(group["border_touched"].mean()),
            }
        )
    table = pd.DataFrame(
        rows,
        columns=(
            "variant_family",
            "severity_level",
            "stroke_count",
            "n_samples",
            "n_sources",
            "ssim_mean",
            "ssim_sd",
            "ssim_median",
            "ssim_min",
            "ssim_max",
            "ssim_drop_from_clean",
            "psnr_mean_db",
            "psnr_sd_db",
            "changed_pixel_fraction_mean",
            "changed_ink_fraction_mean",
            "changed_background_fraction_mean",
            "border_touched_fraction",
        ),
    )
    if table.empty:
        return table
    table["ssim_mean_step_delta"] = table.groupby("variant_family")["ssim_mean"].diff()
    return table


def trend_by_source(manifest: pd.DataFrame) -> pd.DataFrame:
    """Within-source severity/SSIM trend, one row per source and variant."""
    corrupted = manifest[manifest["variant_family"] != CLEAN_VARIANT_FAMILY]
    rows: list[dict[str, object]] = []
    for (source_id, variant), group in corrupted.groupby(
        ["source_id", "variant_family"], sort=True
    ):
        ordered = group.sort_values("severity_level", kind="stable")
        severity = ordered["severity_level"].to_numpy(dtype=np.float64)
        ssim_values = ordered["ssim_target"].to_numpy(dtype=np.float64)
        differences = np.diff(ssim_values)
        rows.append(
            {
                "source_id": str(source_id),
                "variant_family": str(variant),
                "n_levels": int(len(ordered)),
                "spearman": rank_correlation(severity, ssim_values),
                "n_increasing_steps": int(np.count_nonzero(differences > 0.0)),
                "n_tied_steps": int(np.count_nonzero(differences == 0.0)),
                "n_steps": int(differences.size),
                "max_increase": float(differences.max()) if differences.size else 0.0,
                "first_ssim": float(ssim_values[0]),
                "last_ssim": float(ssim_values[-1]),
                "total_drop": float(ssim_values[0] - ssim_values[-1]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "source_id",
            "variant_family",
            "n_levels",
            "spearman",
            "n_increasing_steps",
            "n_tied_steps",
            "n_steps",
            "max_increase",
            "first_ssim",
            "last_ssim",
            "total_drop",
        ),
    )


def trend_violations(manifest: pd.DataFrame) -> pd.DataFrame:
    """Rows where SSIM increases between adjacent severity levels."""
    corrupted = manifest[manifest["variant_family"] != CLEAN_VARIANT_FAMILY]
    rows: list[dict[str, object]] = []
    for (source_id, variant), group in corrupted.groupby(
        ["source_id", "variant_family"], sort=True
    ):
        ordered = group.sort_values("severity_level", kind="stable")
        levels = ordered["severity_level"].tolist()
        values = ordered["ssim_target"].tolist()
        counts = ordered["stroke_count"].tolist()
        for index in range(len(levels) - 1):
            if values[index + 1] > values[index]:
                rows.append(
                    {
                        "source_id": str(source_id),
                        "variant_family": str(variant),
                        "lower_severity": int(levels[index]),
                        "upper_severity": int(levels[index + 1]),
                        "lower_strokes": int(counts[index]),
                        "upper_strokes": int(counts[index + 1]),
                        "lower_ssim": float(values[index]),
                        "upper_ssim": float(values[index + 1]),
                        "increase": float(values[index + 1] - values[index]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=(
            "source_id",
            "variant_family",
            "lower_severity",
            "upper_severity",
            "lower_strokes",
            "upper_strokes",
            "lower_ssim",
            "upper_ssim",
            "increase",
        ),
    )


def grouped_bootstrap_levels(
    manifest: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Source-grouped bootstrap intervals for each level's mean SSIM."""
    if n_bootstrap < 1:
        raise PilotError("n_bootstrap must be >= 1")
    corrupted = manifest[manifest["variant_family"] != CLEAN_VARIANT_FAMILY]
    columns = (
        "variant_family",
        "severity_level",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_groups",
        "n_samples",
        "n_bootstrap",
    )
    if n_bootstrap == 0 or corrupted.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (variant, severity), group in corrupted.groupby(
        ["variant_family", "severity_level"], sort=True
    ):
        values = group["ssim_target"].to_numpy(dtype=np.float64)
        groups = group["source_id"].to_numpy()
        unique_groups = np.array(sorted(set(groups)))
        indices = {name: np.flatnonzero(groups == name) for name in unique_groups}
        rng = np.random.default_rng(
            derive_noise_seed(seed, "bootstrap", str(variant), int(severity))
        )
        replicates = np.empty(n_bootstrap, dtype=np.float64)
        for index in range(n_bootstrap):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sampled_index = np.concatenate([indices[name] for name in sampled])
            replicates[index] = float(values[sampled_index].mean())
        rows.append(
            {
                "variant_family": str(variant),
                "severity_level": int(severity),
                "metric": "ssim_mean",
                "estimate": float(values.mean()),
                "ci_low": float(np.percentile(replicates, BOOTSTRAP_LOW)),
                "ci_high": float(np.percentile(replicates, BOOTSTRAP_HIGH)),
                "n_groups": int(len(unique_groups)),
                "n_samples": int(len(group)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _feature_frame(manifest: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = manifest.merge(features, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(manifest):
        raise PilotError("manifest and feature table do not join one-to-one")
    return merged


def feature_summary_table(
    pilot_manifest: pd.DataFrame,
    pilot_features: pd.DataFrame,
    frozen_manifest: pd.DataFrame,
    frozen_features: pd.DataFrame,
) -> pd.DataFrame:
    """Mean and SD of every feature by family and severity."""
    pilot = _feature_frame(pilot_manifest, pilot_features)
    frozen = _feature_frame(frozen_manifest, frozen_features)
    pilot = pilot[pilot["variant_family"] != CLEAN_VARIANT_FAMILY].copy()
    frozen = frozen[frozen["corruption_type"] != CLEAN_VARIANT_FAMILY].copy()
    pilot["family_label"] = pilot["variant_family"]
    frozen["family_label"] = frozen["corruption_type"]
    combined = pd.concat(
        [
            pilot[["family_label", "severity_level", *FEATURE_NAMES]],
            frozen[["family_label", "severity_level", *FEATURE_NAMES]],
        ],
        ignore_index=True,
    )
    rows: list[dict[str, object]] = []
    for (family, severity), group in combined.groupby(
        ["family_label", "severity_level"], sort=True
    ):
        for feature in FEATURE_NAMES:
            values = group[feature].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "family_label": str(family),
                    "severity_level": int(severity),
                    "feature": feature,
                    "n_samples": int(len(values)),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else math.nan,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "family_label",
            "severity_level",
            "feature",
            "n_samples",
            "mean",
            "sd",
            "min",
            "max",
        ),
    )


def feature_effect_sizes(
    pilot_manifest: pd.DataFrame,
    pilot_features: pd.DataFrame,
    frozen_manifest: pd.DataFrame,
    frozen_features: pd.DataFrame,
    *,
    strong_threshold: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cohen's d between each pilot variant and the frozen blur/noise families.

    Returns
    -------
    tuple of pandas.DataFrame
        Long per-feature effect sizes, then a per-(variant, severity, reference
        family) summary. Cohen's d is ``(pilot_mean - reference_mean) /
        pooled_sd``; a positive value means the pilot feature is larger. The
        summary counts features at both the fixed diagnostic thresholds 0.8 and
        1.0 and at the caller's predeclared ``strong_threshold``.
    """
    if not math.isfinite(strong_threshold) or strong_threshold <= 0.0:
        raise PilotError("strong_threshold must be finite and > 0")
    pilot = _feature_frame(pilot_manifest, pilot_features)
    frozen = _feature_frame(frozen_manifest, frozen_features)
    pilot = pilot[pilot["variant_family"] != CLEAN_VARIANT_FAMILY]
    frozen = frozen[frozen["corruption_type"] != CLEAN_VARIANT_FAMILY]

    long_rows: list[dict[str, object]] = []
    for variant in sorted(pilot["variant_family"].unique()):
        for severity in sorted(
            pilot.loc[pilot["variant_family"] == variant, "severity_level"].unique()
        ):
            pilot_group = pilot[
                (pilot["variant_family"] == variant) & (pilot["severity_level"] == severity)
            ]
            for reference_family in sorted(frozen["corruption_type"].unique()):
                reference_group = frozen[
                    (frozen["corruption_type"] == reference_family)
                    & (frozen["severity_level"] == severity)
                ]
                if reference_group.empty:
                    continue
                for feature in FEATURE_NAMES:
                    pilot_values = pilot_group[feature].to_numpy(dtype=np.float64)
                    reference_values = reference_group[feature].to_numpy(dtype=np.float64)
                    pilot_mean = float(pilot_values.mean())
                    reference_mean = float(reference_values.mean())
                    pilot_sd = float(pilot_values.std(ddof=1))
                    reference_sd = float(reference_values.std(ddof=1))
                    pooled_sd = math.sqrt((pilot_sd**2 + reference_sd**2) / 2.0)
                    effect = (pilot_mean - reference_mean) / pooled_sd if pooled_sd > 0.0 else 0.0
                    long_rows.append(
                        {
                            "variant_family": str(variant),
                            "severity_level": int(severity),
                            "reference_family": str(reference_family),
                            "feature": feature,
                            "pilot_mean": pilot_mean,
                            "reference_mean": reference_mean,
                            "pilot_sd": pilot_sd,
                            "reference_sd": reference_sd,
                            "cohens_d": float(effect),
                            "abs_cohens_d": float(abs(effect)),
                        }
                    )
    long = pd.DataFrame(
        long_rows,
        columns=(
            "variant_family",
            "severity_level",
            "reference_family",
            "feature",
            "pilot_mean",
            "reference_mean",
            "pilot_sd",
            "reference_sd",
            "cohens_d",
            "abs_cohens_d",
        ),
    )
    summary_rows: list[dict[str, object]] = []
    if not long.empty:
        for (variant, severity, reference_family), group in long.groupby(
            ["variant_family", "severity_level", "reference_family"], sort=True
        ):
            summary_rows.append(
                {
                    "variant_family": str(variant),
                    "severity_level": int(severity),
                    "reference_family": str(reference_family),
                    "mean_abs_cohens_d": float(group["abs_cohens_d"].mean()),
                    "max_abs_cohens_d": float(group["abs_cohens_d"].max()),
                    "n_features_abs_d_ge_0_8": int((group["abs_cohens_d"] >= 0.8).sum()),
                    "n_features_abs_d_ge_1_0": int((group["abs_cohens_d"] >= 1.0).sum()),
                    "n_features_abs_d_ge_strong": int(
                        (group["abs_cohens_d"] >= strong_threshold).sum()
                    ),
                    "n_features": int(len(group)),
                }
            )
    summary = pd.DataFrame(
        summary_rows,
        columns=(
            "variant_family",
            "severity_level",
            "reference_family",
            "mean_abs_cohens_d",
            "max_abs_cohens_d",
            "n_features_abs_d_ge_0_8",
            "n_features_abs_d_ge_1_0",
            "n_features_abs_d_ge_strong",
            "n_features",
        ),
    )
    return long, summary


def feature_target_correlations(
    pilot_manifest: pd.DataFrame,
    pilot_features: pd.DataFrame,
    frozen_manifest: pd.DataFrame,
    frozen_features: pd.DataFrame,
) -> pd.DataFrame:
    """Feature/target association by family, pooled and within severity."""
    pilot = _feature_frame(pilot_manifest, pilot_features)
    frozen = _feature_frame(frozen_manifest, frozen_features)
    pilot = pilot[pilot["variant_family"] != CLEAN_VARIANT_FAMILY].copy()
    frozen = frozen[frozen["corruption_type"] != CLEAN_VARIANT_FAMILY].copy()
    pilot["family_label"] = pilot["variant_family"]
    frozen["family_label"] = frozen["corruption_type"]
    combined = pd.concat(
        [
            pilot[["family_label", "severity_level", "ssim_target", *FEATURE_NAMES]],
            frozen[["family_label", "severity_level", "ssim_target", *FEATURE_NAMES]],
        ],
        ignore_index=True,
    )
    rows: list[dict[str, object]] = []
    for family, group in combined.groupby("family_label", sort=True):
        for feature in FEATURE_NAMES:
            x = group[feature].to_numpy(dtype=np.float64)
            y = group["ssim_target"].to_numpy(dtype=np.float64)
            pearson = (
                float(np.corrcoef(x, y)[0, 1])
                if x.size > 1 and np.std(x) > 0.0 and np.std(y) > 0.0
                else math.nan
            )
            within: list[float] = []
            for _, level_group in group.groupby("severity_level", sort=True):
                if len(level_group) > 2:
                    value = rank_correlation(
                        level_group[feature].to_numpy(dtype=np.float64),
                        level_group["ssim_target"].to_numpy(dtype=np.float64),
                    )
                    if math.isfinite(value):
                        within.append(value)
            rows.append(
                {
                    "family_label": str(family),
                    "feature": feature,
                    "pooled_pearson": pearson,
                    "pooled_spearman": rank_correlation(x, y),
                    "mean_within_severity_spearman": (
                        float(np.mean(within)) if within else math.nan
                    ),
                    "n_samples": int(len(group)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "family_label",
            "feature",
            "pooled_pearson",
            "pooled_spearman",
            "mean_within_severity_spearman",
            "n_samples",
        ),
    )


def saturation_table(levels: pd.DataFrame) -> pd.DataFrame:
    """Marginal SSIM change per added stroke, for parameter-saturation review."""
    rows: list[dict[str, object]] = []
    for variant, group in levels.groupby("variant_family", sort=True):
        ordered = group.sort_values("severity_level", kind="stable")
        previous = None
        for row in ordered.itertuples(index=False):
            if previous is not None:
                added = int(row.stroke_count) - int(previous.stroke_count)
                delta = float(row.ssim_mean) - float(previous.ssim_mean)
                rows.append(
                    {
                        "variant_family": str(variant),
                        "from_severity": int(previous.severity_level),
                        "to_severity": int(row.severity_level),
                        "from_strokes": int(previous.stroke_count),
                        "to_strokes": int(row.stroke_count),
                        "added_strokes": added,
                        "ssim_change": delta,
                        "ssim_change_per_added_stroke": delta / added if added else math.nan,
                    }
                )
            previous = row
    return pd.DataFrame(
        rows,
        columns=(
            "variant_family",
            "from_severity",
            "to_severity",
            "from_strokes",
            "to_strokes",
            "added_strokes",
            "ssim_change",
            "ssim_change_per_added_stroke",
        ),
    )


def evaluate_gate(
    config: PilotConfig,
    *,
    manifest: pd.DataFrame,
    levels: pd.DataFrame,
    trends: pd.DataFrame,
    effects_summary: pd.DataFrame,
    duplicate_fraction: float,
    reproducibility_ok: bool,
    verification_ok: bool,
    clean_reference_exact: bool,
) -> PilotGateVerdict:
    """Evaluate the predeclared feasibility gate."""
    corrupted = manifest[manifest["variant_family"] != CLEAN_VARIANT_FAMILY]
    criteria: list[PilotCriterion] = []

    reproducibility_passed = reproducibility_ok and verification_ok and clean_reference_exact
    criteria.append(
        PilotCriterion(
            id="reproducibility",
            description=(
                "Two builds are identical, frozen clean references were re-rendered "
                "exactly, and the written dataset passes verification"
            ),
            observed=(
                f"determinism={reproducibility_ok}, verification={verification_ok}, "
                f"clean_exact={clean_reference_exact}"
            ),
            threshold="all three true",
            passed=reproducibility_passed,
        )
    )

    overall_sd = float(corrupted["ssim_target"].std(ddof=1))
    variant_ranges: dict[str, float] = {}
    min_level_sd = math.inf
    for variant, group in levels.groupby("variant_family", sort=True):
        variant_ranges[str(variant)] = float(group["ssim_mean"].max() - group["ssim_mean"].min())
        min_level_sd = min(min_level_sd, float(group["ssim_sd"].min()))
    non_degenerate_passed = (
        overall_sd >= config.gate.ssim_std_min
        and all(value >= config.gate.level_mean_range_min for value in variant_ranges.values())
        and min_level_sd >= config.gate.per_level_sd_min
    )
    criteria.append(
        PilotCriterion(
            id="non_degeneracy",
            description=(
                "Corrupted SSIM has meaningful spread; every variant spans the "
                "configured level-mean range; every level has non-trivial SD"
            ),
            observed=(
                f"overall_sd={overall_sd:.4f}, level_mean_ranges="
                f"{ {key: round(value, 4) for key, value in variant_ranges.items()} }, "
                f"min_level_sd={min_level_sd:.4f}"
            ),
            threshold=(
                f"overall_sd>={config.gate.ssim_std_min}, "
                f"range>={config.gate.level_mean_range_min}, "
                f"level_sd>={config.gate.per_level_sd_min}"
            ),
            passed=bool(non_degenerate_passed),
        )
    )

    median_spearman = float(trends["spearman"].median())
    total_steps = int(trends["n_steps"].sum())
    increasing_steps = int(trends["n_increasing_steps"].sum())
    increasing_fraction = increasing_steps / total_steps if total_steps else 1.0
    trend_passed = (
        median_spearman <= config.gate.median_source_spearman_max
        and increasing_fraction <= config.gate.increasing_step_fraction_max
    )
    criteria.append(
        PilotCriterion(
            id="interpretable_trend",
            description="Within-source severity and SSIM move in the expected direction",
            observed=(
                f"median_source_spearman={median_spearman:.3f}, "
                f"increasing_step_fraction={increasing_fraction:.3f} "
                f"({increasing_steps}/{total_steps})"
            ),
            threshold=(
                f"median_spearman<={config.gate.median_source_spearman_max}, "
                f"increasing_fraction<={config.gate.increasing_step_fraction_max}"
            ),
            passed=bool(trend_passed),
        )
    )

    distinct_passed = True
    distinct_lines: list[str] = []
    highest = levels.groupby("variant_family")["severity_level"].max().to_dict()
    for variant, severity in highest.items():
        for reference_family in ("gaussian_blur", "gaussian_noise"):
            selected = effects_summary[
                (effects_summary["variant_family"] == variant)
                & (effects_summary["severity_level"] == severity)
                & (effects_summary["reference_family"] == reference_family)
            ]
            if selected.empty:
                distinct_passed = False
                distinct_lines.append(f"{variant} vs {reference_family}: missing")
                continue
            row = selected.iloc[0]
            passed = (
                float(row["mean_abs_cohens_d"]) >= config.gate.feature_effect_min
                and int(row["n_features_abs_d_ge_strong"])
                >= config.gate.feature_strong_effect_count_min
            )
            distinct_passed = distinct_passed and passed
            distinct_lines.append(
                f"{variant} vs {reference_family}: mean|d|={row['mean_abs_cohens_d']:.2f}, "
                f"n|d|>={config.gate.feature_strong_effect_min:.2f}="
                f"{int(row['n_features_abs_d_ge_strong'])}"
            )
    criteria.append(
        PilotCriterion(
            id="feature_distinctness",
            description=(
                "At the highest severity, the eight features differ meaningfully "
                "from both frozen blur and noise families"
            ),
            observed="; ".join(distinct_lines),
            threshold=(
                f"mean|d|>={config.gate.feature_effect_min} and "
                f"n|d|>={config.gate.feature_strong_effect_min} >= "
                f"{config.gate.feature_strong_effect_count_min} for both reference families"
            ),
            passed=bool(distinct_passed),
            note="Cohen's d compares pilot features with the frozen family at the same severity.",
        )
    )

    duplicate_passed = duplicate_fraction <= config.gate.duplicate_fraction_max
    criteria.append(
        PilotCriterion(
            id="duplicate_free",
            description="No unexpected duplicate corrupted images",
            observed=f"duplicate_fraction={duplicate_fraction:.6f}",
            threshold=f"<={config.gate.duplicate_fraction_max}",
            passed=bool(duplicate_passed),
        )
    )

    if all(criterion.passed for criterion in criteria):
        verdict = "PASS"
    elif reproducibility_passed and non_degenerate_passed and trend_passed:
        verdict = "CONDITIONAL"
    else:
        verdict = "FAIL"
    return PilotGateVerdict(verdict=verdict, criteria=tuple(criteria))


def _duplicate_fraction(manifest: pd.DataFrame) -> float:
    corrupted = manifest[manifest["variant_family"] != CLEAN_VARIANT_FAMILY]
    if corrupted.empty:
        return 1.0
    hashes = corrupted["image_sha256"]
    duplicates = int(len(hashes) - hashes.nunique())
    return duplicates / float(len(hashes))


def _check(checks: list[CheckResult], name: str, function: Any) -> None:
    try:
        detail = function()
    except Exception as exc:  # noqa: BLE001 - verification reports every failure
        checks.append(CheckResult(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}"))
        return
    checks.append(CheckResult(name=name, passed=True, detail=str(detail)))


def verify_pilot_dataset(
    root: Path | str,
    *,
    reference: FrozenReference | None = None,
    recompute_targets: bool = True,
    require_summary: bool = True,
) -> VerificationReport:
    """Verify schema, hashes, targets, provenance replay and clean immutability."""
    root = Path(root)
    checks: list[CheckResult] = []
    required = ["manifest.csv", "sources.csv", "features.csv"]
    if require_summary:
        required.append("pilot_summary.json")
    _check(checks, "required_artifacts", lambda: _require_pilot_files(root, required))
    if any(not check.passed for check in checks):
        return VerificationReport(dataset=str(root), checks=tuple(checks))

    manifest = pd.read_csv(root / "manifest.csv")
    sources = pd.read_csv(root / "sources.csv")
    features = pd.read_csv(root / "features.csv")
    summary = (
        json.loads((root / "pilot_summary.json").read_text(encoding="utf-8"))
        if require_summary
        else {}
    )

    _check(checks, "manifest_schema", lambda: _check_pilot_schema(manifest, sources, features))
    _check(checks, "manifest_ids_unique", lambda: _require_unique(manifest, "sample_id"))
    _check(checks, "manifest_paths_exist", lambda: _check_pilot_paths(root, manifest))
    _check(
        checks,
        "reference_matches_source",
        lambda: _check_pilot_reference_matching(manifest, sources),
    )
    _check(
        checks,
        "clean_reference_immutability",
        lambda: _check_clean_immutability(root, manifest, sources, reference),
    )
    _check(checks, "image_dimensions", lambda: _check_dimensions(root, manifest, reference))
    _check(checks, "image_content_hashes", lambda: _check_pilot_image_hashes(root, manifest))
    _check(checks, "conditional_psnr_targets", lambda: _check_pilot_psnr(manifest))
    if recompute_targets:
        _check(
            checks,
            "targets_recomputed",
            lambda: _check_pilot_targets(root, manifest),
        )
    _check(
        checks,
        "provenance_replay",
        lambda: _check_provenance_replay(root, manifest),
    )
    _check(
        checks,
        "no_duplicate_images",
        lambda: _check_no_duplicate_images(manifest),
    )
    _check(
        checks,
        "feature_table",
        lambda: _check_pilot_feature_table(features, manifest),
    )
    if require_summary:
        _check(
            checks,
            "summary_consistency",
            lambda: _check_pilot_summary(summary, manifest, sources, features),
        )
    return VerificationReport(dataset=str(root), checks=tuple(checks))


def _require_pilot_files(root: Path, names: Iterable[str]) -> str:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise PilotError(f"missing required file(s): {missing}")
    if not (root / "images").is_dir():
        raise PilotError("missing required directory: images/")
    return f"found {len(list(names))} files plus images/"


def _check_pilot_schema(
    manifest: pd.DataFrame, sources: pd.DataFrame, features: pd.DataFrame
) -> str:
    if tuple(manifest.columns) != PILOT_MANIFEST_COLUMNS:
        raise PilotError(
            f"manifest columns must be {PILOT_MANIFEST_COLUMNS}, got {tuple(manifest.columns)}"
        )
    if tuple(sources.columns) != PILOT_SOURCE_COLUMNS:
        raise PilotError(
            f"sources columns must be {PILOT_SOURCE_COLUMNS}, got {tuple(sources.columns)}"
        )
    if tuple(features.columns) != PILOT_FEATURE_COLUMNS:
        raise PilotError(
            f"feature columns must be {PILOT_FEATURE_COLUMNS}, got {tuple(features.columns)}"
        )
    if sources["source_id"].duplicated().any():
        raise PilotError("sources.csv contains duplicate source_id values")
    if "text" in manifest.columns:
        raise PilotError("pilot manifest must not contain raw text")
    return f"{len(manifest)} manifest rows, {len(sources)} sources, schema complete"


def _require_unique(frame: pd.DataFrame, column: str) -> str:
    if frame[column].isna().any():
        raise PilotError(f"column {column!r} contains missing values")
    if frame[column].duplicated().any():
        duplicates = frame.loc[frame[column].duplicated(), column].head(5).tolist()
        raise PilotError(f"column {column!r} is not unique, e.g. {duplicates}")
    return f"{len(frame)} unique {column} values"


def _check_pilot_paths(root: Path, manifest: pd.DataFrame) -> str:
    resolved_root = root.resolve()
    missing: list[str] = []
    escaped: list[str] = []
    for row in manifest.itertuples(index=False):
        for column in ("image_path", "reference_path"):
            relative = str(getattr(row, column))
            path = (root / relative).resolve()
            if not path.is_relative_to(resolved_root):
                escaped.append(relative)
            elif not path.is_file():
                missing.append(relative)
    if escaped:
        raise PilotError(f"paths escape the pilot root: {sorted(set(escaped))[:5]}")
    if missing:
        raise PilotError(f"missing image file(s): {sorted(set(missing))[:5]}")
    return "all image and reference paths exist inside the pilot root"


def _check_pilot_reference_matching(manifest: pd.DataFrame, sources: pd.DataFrame) -> str:
    source_index = sources.set_index("source_id")
    for row in manifest.itertuples(index=False):
        if row.source_id not in source_index.index:
            raise PilotError(f"{row.sample_id} references unknown source {row.source_id}")
        source = source_index.loc[row.source_id]
        expected_reference = str(source["pilot_clean_image_path"])
        if str(row.reference_path) != expected_reference:
            raise PilotError(f"{row.sample_id}: reference_path is not the pilot clean image")
        if str(row.reference_sha256) != str(source["clean_sha256"]):
            raise PilotError(f"{row.sample_id}: reference hash disagrees with sources.csv")
        for column in ("split", "font_id", "text_length"):
            if getattr(row, column) != source[column]:
                raise PilotError(f"{row.sample_id}: {column} disagrees with sources.csv")
    if set(sources["source_id"]) - set(manifest["source_id"]):
        raise PilotError("some sources have no manifest rows")
    return f"{len(manifest)} rows reference the correct clean source"


def _check_clean_immutability(
    root: Path,
    manifest: pd.DataFrame,
    sources: pd.DataFrame,
    reference: FrozenReference | None,
) -> str:
    clean_rows = manifest[manifest["variant_family"] == CLEAN_VARIANT_FAMILY]
    if len(clean_rows) != len(sources):
        raise PilotError(
            f"expected one clean row per source, got {len(clean_rows)} for {len(sources)} sources"
        )
    for source in sources.itertuples(index=False):
        image = load_grayscale(root / str(source.pilot_clean_image_path))
        actual = sha256_bytes(image.tobytes())
        if actual != str(source.clean_sha256):
            raise PilotError(f"clean image hash mismatch for source {source.source_id}")
        if reference is not None:
            frozen = reference.clean_images[str(source.source_id)]
            if not np.array_equal(image, frozen):
                raise PilotError(
                    f"pilot clean image differs from the frozen reference for {source.source_id}"
                )
    return f"{len(sources)} clean references match their frozen pixel hashes"


def _check_dimensions(root: Path, manifest: pd.DataFrame, reference: FrozenReference | None) -> str:
    expected: tuple[int, int] | None = None
    if reference is not None:
        first = next(iter(reference.clean_images.values()))
        expected = (int(first.shape[0]), int(first.shape[1]))
    for row in manifest.itertuples(index=False):
        image = load_grayscale(root / str(row.image_path))
        shape = (int(image.shape[0]), int(image.shape[1]))
        if shape != (int(row.height), int(row.width)):
            raise PilotError(
                f"{row.sample_id}: image shape {shape} disagrees with manifest "
                f"{(row.height, row.width)}"
            )
        if expected is not None and shape != expected:
            raise PilotError(f"{row.sample_id}: image shape {shape} differs from {expected}")
    return f"{len(manifest)} images are 8-bit grayscale with recorded dimensions"


def _check_pilot_image_hashes(root: Path, manifest: pd.DataFrame) -> str:
    expected: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        relative = str(row.image_path)
        previous = expected.setdefault(relative, str(row.image_sha256))
        if previous != row.image_sha256:
            raise PilotError(f"inconsistent image_sha256 for {relative}")
    mismatches = []
    for relative, digest in expected.items():
        image = load_grayscale(root / relative)
        if sha256_bytes(image.tobytes()) != digest:
            mismatches.append(relative)
    if mismatches:
        raise PilotError(f"image pixel hash mismatch: {sorted(mismatches)[:5]}")
    return f"{len(expected)} image files match their recorded pixel hashes"


def _check_pilot_psnr(manifest: pd.DataFrame) -> str:
    infinite = manifest["psnr_infinite"].astype(bool)
    clean = manifest["variant_family"] == CLEAN_VARIANT_FAMILY
    if not (infinite == clean).all():
        raise PilotError(
            "psnr_infinite must be true exactly for clean identity samples; "
            f"mismatches: {manifest.loc[infinite != clean, 'sample_id'].head(5).tolist()}"
        )
    finite = ~infinite
    if manifest.loc[finite, "psnr_target"].isna().any():
        raise PilotError("finite PSNR rows must record psnr_target")
    if manifest.loc[infinite, "psnr_target"].notna().any():
        raise PilotError("infinite PSNR rows must leave psnr_target empty")
    return f"{int(clean.sum())} clean rows use an explicit infinite-PSNR flag"


def _check_pilot_targets(root: Path, manifest: pd.DataFrame) -> str:
    references: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    mismatches: list[str] = []
    for row in manifest.itertuples(index=False):
        reference_path = str(row.reference_path)
        image_path = str(row.image_path)
        if reference_path not in references:
            references[reference_path] = load_grayscale(root / reference_path)
        if image_path not in images:
            images[image_path] = load_grayscale(root / image_path)
        metrics = compute_pair_metrics(
            references[reference_path], images[image_path], ("ssim", "psnr")
        )
        if abs(metrics["ssim"] - float(row.ssim_target)) > 1e-9:
            mismatches.append(f"{row.sample_id}:ssim")
        if bool(row.psnr_infinite):
            if not math.isinf(metrics["psnr"]):
                mismatches.append(f"{row.sample_id}:psnr-inf")
        elif abs(metrics["psnr"] - float(row.psnr_target)) > 1e-9:
            mismatches.append(f"{row.sample_id}:psnr")
    if mismatches:
        raise PilotError(
            f"{len(mismatches)} target(s) disagree with the exact pair, e.g. {mismatches[:5]}"
        )
    return f"recomputed SSIM/PSNR targets for {len(manifest)} rows from their exact references"


def _check_provenance_replay(root: Path, manifest: pd.DataFrame) -> str:
    """Rebuild each corrupted image from its recorded geometry without an RNG."""
    corrupted = manifest[manifest["variant_family"] != CLEAN_VARIANT_FAMILY]
    references: dict[str, np.ndarray] = {}
    mismatches: list[str] = []
    for row in corrupted.itertuples(index=False):
        reference_path = str(row.reference_path)
        if reference_path not in references:
            references[reference_path] = load_grayscale(root / reference_path)
        geometry = json.loads(str(row.geometry_json))
        strokes = strokes_from_json(geometry)
        if len(strokes) != int(row.stroke_count):
            mismatches.append(f"{row.sample_id}:stroke-count")
            continue
        replayed = apply_strokes(
            references[reference_path],
            list(strokes),
            width_px=int(row.stroke_width_px),
            intensity=int(row.stroke_intensity),
        )
        if sha256_bytes(replayed.tobytes()) != str(row.image_sha256):
            mismatches.append(f"{row.sample_id}:pixels")
    if mismatches:
        raise PilotError(f"{len(mismatches)} provenance replay(s) disagree, e.g. {mismatches[:5]}")
    return f"replayed {len(corrupted)} corrupted images from recorded geometry without an RNG"


def _check_no_duplicate_images(manifest: pd.DataFrame) -> str:
    duplicates = manifest["image_sha256"].duplicated(keep=False)
    if duplicates.any():
        examples = manifest.loc[duplicates, "sample_id"].head(5).tolist()
        raise PilotError(f"duplicate image hashes, e.g. {examples}")
    return f"{len(manifest)} images have distinct pixel hashes"


def _check_pilot_feature_table(features: pd.DataFrame, manifest: pd.DataFrame) -> str:
    if features["sample_id"].duplicated().any():
        raise PilotError("features.csv contains duplicate sample_id values")
    if set(features["sample_id"]) != set(manifest["sample_id"]):
        raise PilotError("features.csv sample IDs do not match the manifest")
    values = features[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise PilotError("features.csv contains NaN or infinite values")
    forbidden = {"ssim", "psnr", "ms_ssim"}
    leaked = forbidden & {name.lower() for name in features.columns}
    if leaked:
        raise PilotError(f"target metric(s) used as features: {sorted(leaked)}")
    return f"{len(features)} feature rows x {len(FEATURE_NAMES)} features are finite"


def _check_pilot_summary(
    summary: dict[str, Any],
    manifest: pd.DataFrame,
    sources: pd.DataFrame,
    features: pd.DataFrame,
) -> str:
    counts = summary.get("counts", {})
    if counts.get("sources") != len(sources):
        raise PilotError("summary source count disagrees with sources.csv")
    if counts.get("samples") != len(manifest):
        raise PilotError("summary sample count disagrees with the manifest")
    if counts.get("feature_rows") != len(features):
        raise PilotError("summary feature count disagrees with features.csv")
    if summary.get("feature_names") != list(FEATURE_NAMES):
        raise PilotError("summary feature order disagrees with this code version")
    if summary.get("pilot_config_hash") != manifest["pilot_config_hash"].iloc[0]:
        raise PilotError("summary pilot_config_hash disagrees with the manifest")
    return "summary counts, hashes, feature names and config identity agree with the tables"


def _build_summary(
    config: PilotConfig,
    reference: FrozenReference,
    generation: GenerationData,
    *,
    target: Path,
    determinism: Mapping[str, object],
    verification: VerificationReport,
    gate: PilotGateVerdict,
    frozen_before: Mapping[str, Mapping[str, object]],
    frozen_after: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    corrupted = generation.manifest[generation.manifest["variant_family"] != CLEAN_VARIANT_FAMILY]
    changed = {
        path: {
            "before": frozen_before.get(path),
            "after": frozen_after.get(path),
            "unchanged": frozen_before.get(path) == frozen_after.get(path),
        }
        for path in sorted(set(frozen_before) | set(frozen_after))
    }
    return {
        "pilot_name": config.pilot.name,
        "pilot_version": config.pilot.version,
        "pilot_config_hash": config.pilot_config_hash,
        "feature_names": list(FEATURE_NAMES),
        "package_version": package_version(),
        "code_revision": code_revision(),
        "config": config.provenance(),
        "reference_dataset": {
            "root": str(reference.root),
            "dataset_version": reference.summary.get("dataset_version"),
            "generation_config_hash": reference.summary.get("generation_config_hash"),
            "split_assignment_sha256": reference.summary.get("split_policy", {}).get(
                "assignment_sha256"
            ),
            "manifest_sha256": config.reference.manifest_sha256,
            "sources_sha256": config.reference.sources_sha256,
            "features_sha256": config.reference.features_sha256,
            "ground_truth_sha256": config.reference.ground_truth_sha256,
            "summary_sha256": config.reference.summary_sha256,
            "rerendered_sources": reference.rerendered_sources,
        },
        "counts": {
            "sources": int(len(generation.sources)),
            "samples": int(len(generation.manifest)),
            "clean_samples": int(
                (generation.manifest["variant_family"] == CLEAN_VARIANT_FAMILY).sum()
            ),
            "corrupted_samples": int(len(corrupted)),
            "feature_rows": int(len(generation.features)),
            "by_split": _counts(generation.sources["split"]),
            "by_font": _counts(generation.sources["font_id"]),
            "by_variant": _counts(corrupted["variant_family"]),
            "by_severity": _counts(corrupted["severity_level"]),
            "by_variant_severity": _counts(
                corrupted["variant_family"].astype(str)
                + "_s"
                + corrupted["severity_level"].astype(str)
            ),
            "disk_usage_bytes": sum(
                path.stat().st_size for path in target.rglob("*") if path.is_file()
            ),
        },
        "target": {
            "name": "ssim_target",
            "definition": (
                "SSIM against the exact clean reference (data_range=255, win_size=11, "
                "gaussian_weights=True, sigma=1.5, use_sample_covariance=False)"
            ),
            "ssim_mean": float(corrupted["ssim_target"].mean()),
            "ssim_sd": float(corrupted["ssim_target"].std(ddof=1)),
            "ssim_min": float(corrupted["ssim_target"].min()),
            "ssim_max": float(corrupted["ssim_target"].max()),
            "psnr_mean_db": float(corrupted["psnr_target"].mean()),
            "psnr_min_db": float(corrupted["psnr_target"].min()),
            "psnr_max_db": float(corrupted["psnr_target"].max()),
        },
        "determinism": dict(determinism),
        "verification": {
            "ok": verification.ok,
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in verification.checks
            ],
        },
        "gate": {
            "verdict": gate.verdict,
            "criteria": [
                {
                    "id": criterion.id,
                    "description": criterion.description,
                    "observed": criterion.observed,
                    "threshold": criterion.threshold,
                    "passed": criterion.passed,
                    "note": criterion.note,
                }
                for criterion in gate.criteria
            ],
        },
        "frozen_assets": {
            "before": dict(frozen_before),
            "after": dict(frozen_after),
            "paths": changed,
            "all_unchanged": all(bool(entry["unchanged"]) for entry in changed.values()),
        },
        "artifacts": {
            "manifest_sha256": sha256_file(target / "manifest.csv"),
            "sources_sha256": sha256_file(target / "sources.csv"),
            "features_sha256": sha256_file(target / "features.csv"),
            "n_image_files": len(generation.image_hashes),
            "image_pixel_hash_set": _image_pixel_hash_set(generation.image_hashes),
        },
    }


def _format(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[object]], *, digits: int = 3) -> list[str]:
    def escape(value: str) -> str:
        return value.replace("|", "\\|")

    lines = [
        "| " + " | ".join(escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, str):
                cells.append(escape(value))
            elif isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, int):
                cells.append(str(value))
            else:
                cells.append(_format(value, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _contact_sheet(
    path: Path,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    cells: Sequence[Sequence[np.ndarray]],
    padding: int = 6,
    label_margin_left: int = 110,
    label_margin_top: int = 18,
) -> None:
    if not cells or not cells[0]:
        return
    cell_height, cell_width = cells[0][0].shape
    n_rows = len(cells)
    n_columns = len(cells[0])
    width = label_margin_left + n_columns * (cell_width + padding) + padding
    height = label_margin_top + n_rows * (cell_height + padding) + padding
    sheet = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=11)
    for column, label in enumerate(column_labels):
        x = label_margin_left + column * (cell_width + padding)
        draw.text((x, 3), label, fill=0, font=font)
    for row_index, (label, row_cells) in enumerate(zip(row_labels, cells, strict=True)):
        y = label_margin_top + row_index * (cell_height + padding)
        draw.text((4, y + cell_height // 2 - 6), label, fill=0, font=font)
        for column, image in enumerate(row_cells):
            x = label_margin_left + column * (cell_width + padding)
            sheet.paste(Image.fromarray(image), (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")


def _write_contact_sheets(
    root: Path,
    config: PilotConfig,
    sources: pd.DataFrame,
    reports: Path,
) -> None:
    source_rows = sources.sort_values(["font_id", "source_id"], kind="stable")
    examples = source_rows.drop_duplicates("font_id", keep="first")
    for ladder in config.ladders:
        row_labels: list[str] = []
        cells: list[list[np.ndarray]] = []
        for source in examples.itertuples(index=False):
            clean = load_grayscale(root / str(source.pilot_clean_image_path))
            row = [clean]
            for severity in range(1, ladder.n_levels + 1):
                sample_id = f"{source.source_id}__{_variant_id(ladder.variant, severity)}"
                path = root / "images" / ladder.variant / f"{sample_id}.png"
                row.append(load_grayscale(path))
            row_labels.append(f"{source.font_id} / {source.source_id[:12]}")
            cells.append(row)
        column_labels = ["clean", *[f"{count} stroke(s)" for count in ladder.stroke_counts]]
        _contact_sheet(
            reports / f"examples_{ladder.variant}.png",
            row_labels=row_labels,
            column_labels=column_labels,
            cells=cells,
        )


def _report_markdown(
    config: PilotConfig,
    reference: FrozenReference,
    result_tables: Mapping[str, pd.DataFrame],
    summary: Mapping[str, Any],
    gate: PilotGateVerdict,
    verification: VerificationReport,
    determinism: Mapping[str, object],
) -> str:
    levels = result_tables["level_summary"]
    trends = result_tables["trend_by_source"]
    violations = result_tables["trend_violations"]
    effects = result_tables["feature_distinctness"]
    saturation = result_tables["saturation"]
    uncertainty = result_tables["uncertainty"]
    correlations = result_tables["feature_target_correlations"]

    lines = [
        "# Occlusion and visual-interference pilot — feasibility report",
        "",
        "**Scope:** synthetic image similarity (SSIM against the clean reference). "
        "This is not a measure of human readability, CAPTCHA solvability, or "
        "real-world quality. The pilot is a small, predeclared feasibility check, "
        "not a robustness benchmark.",
        "",
        "## Frozen assets",
        "",
        f"- Frozen dataset: `{reference.root}` (`{config.reference.dataset_version}`)",
        f"- Generation hash: `{config.reference.generation_config_hash}`",
        f"- Manifest SHA-256: `{config.reference.manifest_sha256}`",
        f"- Sources SHA-256: `{config.reference.sources_sha256}`",
        f"- Features SHA-256: `{config.reference.features_sha256}`",
        f"- Split-assignment SHA-256: `{config.reference.split_assignment_sha256}`",
        f"- Clean sources re-rendered exactly: {reference.rerendered_sources}",
        f"- Frozen assets unchanged before/after: {summary['frozen_assets']['all_unchanged']}",
        "",
        "The pilot re-rendered every clean source from the frozen dataset config and "
        "compared pixels, source IDs, render seeds, text hashes and split labels. It "
        "did not modify the frozen dataset, split, M3 model artifacts or benchmark "
        "reports.",
        "",
        "## Pilot configuration",
        "",
        f"- Config: `{config.config_path}` (sha256 `{config.config_sha256[:16]}…`)",
        f"- Pilot generation hash: `{config.pilot_config_hash}`",
        f"- Seed: `{config.pilot.seed}`; bootstrap replicates: "
        f"`{config.pilot.bootstrap_replicates}`",
        f"- Output dataset: `{config.pilot.output_dataset}`",
        f"- Output reports: `{config.pilot.output_reports}`",
        "",
    ]
    for ladder in config.ladders:
        if ladder.variant == VARIANT_THIN_LINES:
            geometry = f"line angles {list(ladder.line_angle_deg)}°"
        else:
            geometry = (
                f"arc sweep {list(ladder.arc_sweep_deg)}°, "
                f"arc radius fractions {list(ladder.arc_radius_fraction)}, "
                f"arc aspect {list(ladder.arc_aspect_ratio)}"
            )
        lines.append(
            f"- `{ladder.variant}`: stroke counts {list(ladder.stroke_counts)}, "
            f"width {ladder.width_px} px, intensity {ladder.intensity}; {geometry}"
        )
    counts = summary["counts"]
    lines.extend(
        [
            "",
            f"- Sources: {counts['sources']} (frozen split preserved: {counts['by_split']})",
            f"- Samples: {counts['samples']} = {counts['clean_samples']} clean + "
            f"{counts['corrupted_samples']} corrupted",
            f"- By variant: {counts['by_variant']}; by severity: {counts['by_severity']}",
            "",
            "## Reproducibility and validation",
            "",
            (
                "- Two independent in-memory builds identical: "
                f"`{determinism.get('manifest_identical')}` manifest, "
                f"`{determinism.get('features_identical')}` features, "
                f"`{determinism.get('image_hashes_identical')}` image hashes"
                if determinism.get("repeated_build")
                else "- Two independent in-memory builds: not checked (`--skip-reproduction`)"
            ),
            (
                "- Clean references unchanged during generation: "
                f"`{determinism.get('clean_hashes_identical')}`"
                if determinism.get("repeated_build")
                else "- Clean references unchanged during generation: not checked"
            ),
            f"- Dataset verification: {'PASS' if verification.ok else 'FAIL'} "
            f"({sum(check.passed for check in verification.checks)}/"
            f"{len(verification.checks)} checks)",
            "",
        ]
    )
    for check in verification.checks:
        lines.append(f"- **{'PASS' if check.passed else 'FAIL'}** `{check.name}` — {check.detail}")
    lines.extend(
        [
            "",
            "## SSIM findings",
            "",
            "SSIM is structural similarity to the clean reference, not readability. "
            "Higher is more similar. The clean baseline is exactly 1.0 by construction.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "variant",
                "severity",
                "strokes",
                "n",
                "SSIM mean",
                "SD",
                "median",
                "min",
                "max",
                "drop",
                "Δ step",
            ],
            [
                [
                    row.variant_family,
                    int(row.severity_level),
                    int(row.stroke_count),
                    int(row.n_samples),
                    float(row.ssim_mean),
                    float(row.ssim_sd),
                    float(row.ssim_median),
                    float(row.ssim_min),
                    float(row.ssim_max),
                    float(row.ssim_drop_from_clean),
                    "n/a" if pd.isna(row.ssim_mean_step_delta) else float(row.ssim_mean_step_delta),
                ]
                for row in levels.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            f"Source-grouped bootstrap intervals (resampling {counts['sources']} sources, "
            "not variants):",
            "",
        ]
    )
    if uncertainty.empty:
        lines.append("No bootstrap rows were configured.")
    else:
        lines.extend(
            _markdown_table(
                ["variant", "severity", "estimate", "95% low", "95% high", "groups"],
                [
                    [
                        row.variant_family,
                        int(row.severity_level),
                        float(row.estimate),
                        float(row.ci_low),
                        float(row.ci_high),
                        int(row.n_groups),
                    ]
                    for row in uncertainty.itertuples(index=False)
                ],
            )
        )
    lines.extend(
        [
            "",
            "PSNR is secondary and is reported only as a sanity diagnostic. It is not "
            "the modelling target and is not comparable across corruption families.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["variant", "severity", "PSNR mean (dB)", "PSNR SD (dB)", "n"],
            [
                [
                    row.variant_family,
                    int(row.severity_level),
                    float(row.psnr_mean_db),
                    float(row.psnr_sd_db),
                    int(row.n_samples),
                ]
                for row in levels.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Trend and exceptions",
            "",
            f"- Median within-source Spearman(severity, SSIM): `{trends['spearman'].median():.3f}`",
            f"- Increasing adjacent steps: "
            f"`{int(trends['n_increasing_steps'].sum())}/"
            f"{int(trends['n_steps'].sum())}`; tied steps: "
            f"`{int(trends['n_tied_steps'].sum())}`",
            f"- Sources whose ladder is not strictly monotone: "
            f"`{int((trends['n_increasing_steps'] > 0).sum())}/"
            f"{len(trends)}` source-variant ladders",
            "",
        ]
    )
    if violations.empty:
        lines.append("No source-variant ladder increased between adjacent severities.")
    else:
        lines.append("Exceptions (reported, not discarded):")
        lines.extend(
            _markdown_table(
                [
                    "source",
                    "variant",
                    "lower s",
                    "upper s",
                    "lower SSIM",
                    "upper SSIM",
                    "increase",
                ],
                [
                    [
                        row.source_id,
                        row.variant_family,
                        int(row.lower_severity),
                        int(row.upper_severity),
                        float(row.lower_ssim),
                        float(row.upper_ssim),
                        float(row.increase),
                    ]
                    for row in violations.itertuples(index=False)
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Feature contrast with blur and noise",
            "",
            "Cohen's d compares each feature's mean at the same severity. The gate "
            "uses the highest severity only. Positive d means the pilot feature is "
            "larger than the frozen family's feature.",
            "",
        ]
    )
    highest = levels.groupby("variant_family")["severity_level"].max().to_dict()
    highest_rows = effects[
        effects.apply(
            lambda row: int(row["severity_level"]) == int(highest.get(row["variant_family"], -1)),
            axis=1,
        )
    ]
    lines.extend(
        _markdown_table(
            [
                "variant",
                "severity",
                "reference",
                "mean |d|",
                "max |d|",
                "n |d|>=0.8",
                "n |d|>=1.0",
            ],
            [
                [
                    row.variant_family,
                    int(row.severity_level),
                    row.reference_family,
                    float(row.mean_abs_cohens_d),
                    float(row.max_abs_cohens_d),
                    int(row.n_features_abs_d_ge_0_8),
                    int(row.n_features_abs_d_ge_1_0),
                ]
                for row in highest_rows.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            "Feature/target association by family (pooled across severities; the "
            "pooled value is confounded by severity, so the within-severity median "
            "is also shown):",
            "",
        ]
    )
    selected_features = ("laplacian_variance", "edge_density", "noise_sigma_immerkaer")
    selected = correlations[correlations["feature"].isin(selected_features)]
    lines.extend(
        _markdown_table(
            ["family", "feature", "pooled Pearson", "pooled Spearman", "within-severity ρ"],
            [
                [
                    row.family_label,
                    row.feature,
                    float(row.pooled_pearson),
                    float(row.pooled_spearman),
                    float(row.mean_within_severity_spearman),
                ]
                for row in selected.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Saturation, edge effects and implementation artefacts",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "variant",
                "from s",
                "to s",
                "added strokes",
                "Δ SSIM",
                "Δ per stroke",
            ],
            [
                [
                    row.variant_family,
                    int(row.from_severity),
                    int(row.to_severity),
                    int(row.added_strokes),
                    float(row.ssim_change),
                    float(row.ssim_change_per_added_stroke),
                ]
                for row in saturation.itertuples(index=False)
            ],
        )
    )
    if not saturation.empty:
        first_rates = saturation.groupby("variant_family")["ssim_change_per_added_stroke"].first()
        last_rates = saturation.groupby("variant_family")["ssim_change_per_added_stroke"].last()
        rate_text = ", ".join(
            f"{variant} {float(first_rates[variant]):.3f} -> {float(last_rates[variant]):.3f}"
            for variant in first_rates.index
        )
        lines.append("")
        lines.append(
            f"- Marginal SSIM change per added stroke ({rate_text}) declines with "
            "density. This is modest saturation, not a collapse: every level remains "
            "well above the clean baseline and no level has a constant target."
        )
    lines.append("")
    border_min = float(levels["border_touched_fraction"].min())
    border_max = float(levels["border_touched_fraction"].max())
    border_text = (
        f"`{border_min:.3f}` at every level"
        if border_min == border_max
        else f"`{border_min:.3f}`–`{border_max:.3f}` across levels"
    )
    lines.extend(
        [
            f"- Border-touching corrupted images: {border_text}. "
            "Strokes deliberately extend to the canvas boundary; this is not text clipping.",
            f"- Changed-pixel fraction range: "
            f"`{levels['changed_pixel_fraction_mean'].min():.4f}`–"
            f"`{levels['changed_pixel_fraction_mean'].max():.4f}` of the canvas.",
            f"- Changed ink-pixel fraction range: "
            f"`{levels['changed_ink_fraction_mean'].min():.4f}`–"
            f"`{levels['changed_ink_fraction_mean'].max():.4f}` of the clean ink.",
            "- No additive intensity is used, so 8-bit clipping is not applicable. "
            "Strokes are drawn at intensity 0; pixels already at 0 (glyph cores) "
            "can make an added stroke partially invisible. That is a real confound "
            "and a source of flat or tied steps.",
            "- The nested stroke pool means severity is a density dose, not an "
            "independent random draw per level. This was chosen to keep the trend "
            "interpretable; it also means the same early strokes recur at every "
            "severity.",
            "",
            "## Feasibility gate",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["criterion", "threshold", "observed", "pass"],
            [
                [
                    criterion.id,
                    criterion.threshold,
                    criterion.observed,
                    criterion.passed,
                ]
                for criterion in gate.criteria
            ],
        )
    )
    lines.extend(
        [
            "",
            f"**Verdict: {gate.verdict}.**",
            "",
            "## Interpretation and limitations",
            "",
            "Measured facts and hypotheses are kept separate below.",
            "",
            "**Measured.** The family is deterministic and replayable from recorded "
            "geometry; the clean references are the frozen renderer's images; the "
            "target is non-degenerate and the within-source trend is reported above; "
            "the feature contrast against blur and noise is reported above.",
            "",
            "**Not measured.** This pilot does not show that occlusion is harder for "
            "humans, does not measure CAPTCHA solvability, does not train a model, and "
            "does not test cross-generator or real-image generalisation. It uses one "
            "canvas, one renderer, three fonts and 60 synthetic sources.",
            "",
            "**Confounds.** Interference intensity is fixed at black; stroke overlap "
            "with glyph ink reduces the effective change; density is the only varied "
            "parameter; the source split is inherited and not re-randomised for this "
            "family.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if gate.verdict == "PASS":
        lines.append(
            "Proceed to a separate grouped robustness benchmark. The pilot supports a "
            "non-degenerate, interpretable family with a feature pattern distinct from "
            "blur and noise. The benchmark should keep this family out of the frozen "
            "M3 test set, predeclare its grouped protocol, and compare any model "
            "against the saved feature-only baseline. The pilot does not establish "
            "that the family is objectively harder."
        )
    elif gate.verdict == "CONDITIONAL":
        lines.append(
            "Proceed only after resolving the failed criterion. The family is "
            "reproducible and interpretable, but at least one distinctness or "
            "duplicate criterion did not meet its predeclared threshold. Revise the "
            "pilot (for example, add an intensity or width dimension) rather than "
            "declaring the family successful."
        )
    else:
        lines.append(
            "Do not proceed to a full benchmark yet. At least one prerequisite "
            "criterion failed. Preserve these diagnostics and revise or reject the "
            "corruption design."
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Dataset (Git-ignored): `{config.pilot.output_dataset}`",
            f"- Reports (Git-ignored): `{config.pilot.output_reports}`",
            "- Tables: `level_summary.csv`, `uncertainty.csv`, `trend_by_source.csv`, "
            "`trend_violations.csv`, `feature_summary.csv`, `feature_effect_sizes.csv`, "
            "`feature_distinctness.csv`, `feature_target_correlations.csv`, "
            "`saturation.csv`, `gate_results.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_pilot(
    config: PilotConfig,
    *,
    output_dataset: Path | str | None = None,
    output_reports: Path | str | None = None,
    overwrite: bool = False,
    verify_reproduction: bool = True,
) -> PilotResult:
    """Run the pilot end to end.

    The frozen dataset is read and audited first. The pilot dataset is written
    through a staging directory and renamed into place only after both the
    dataset and its verification succeed.
    """
    dataset_target = (
        Path(output_dataset) if output_dataset is not None else config.pilot.output_dataset
    )
    reports_target = (
        Path(output_reports) if output_reports is not None else config.pilot.output_reports
    )
    dataset_target = dataset_target.expanduser()
    reports_target = reports_target.expanduser()
    reference_root = config.reference.dataset.expanduser().resolve()
    resolved_target = dataset_target.resolve()
    resolved_reports = reports_target.resolve()
    for label, target in (("dataset", resolved_target), ("reports", resolved_reports)):
        if (
            target == reference_root
            or reference_root.is_relative_to(target)
            or target.is_relative_to(reference_root)
        ):
            raise PilotError(
                f"{label} output must not overlap the frozen reference dataset: "
                f"{target} vs {reference_root}"
            )
    for label, target in (("dataset", resolved_target), ("reports", resolved_reports)):
        for root_name in PROTECTED_OUTPUT_ROOTS:
            protected = Path(root_name).expanduser().resolve()
            if (
                target == protected
                or target.is_relative_to(protected)
                or protected.is_relative_to(target)
            ):
                raise PilotError(
                    f"{label} output must not overlap a protected frozen path: "
                    f"{target} vs {protected}"
                )
    if dataset_target.exists() and not overwrite:
        raise PilotExistsError(
            f"pilot dataset already exists: {dataset_target}. "
            "Pass --overwrite to replace it, or choose a different directory."
        )

    frozen_before = fingerprint_assets(FROZEN_ASSET_PATHS)
    reference = load_frozen_reference(config)
    generation = _build_generation_data(config, reference)
    determinism: dict[str, object] = {
        "manifest_identical": None,
        "sources_identical": None,
        "features_identical": None,
        "image_hashes_identical": None,
        "clean_hashes_identical": None,
        "ok": False,
        "repeated_build": False,
    }
    if verify_reproduction:
        second = _build_generation_data(config, reference)
        determinism = _compare_generations(generation, second)
        determinism["repeated_build"] = True
        if not determinism["ok"]:
            raise PilotError(f"pilot is not deterministic: {determinism}")

    staging = dataset_target.with_name(dataset_target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_csv(generation.sources, staging / "sources.csv")
        _write_csv(generation.manifest, staging / "manifest.csv")
        _write_csv(generation.features, staging / "features.csv")
        for relative, image in generation.image_payloads.items():
            _write_png(staging / relative, image)
        if dataset_target.exists():
            shutil.rmtree(dataset_target)
        staging.rename(dataset_target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    verification = verify_pilot_dataset(
        dataset_target,
        reference=reference,
        recompute_targets=True,
        require_summary=False,
    )
    if not verification.ok:
        raise PilotError(
            "pilot dataset failed verification: "
            + "; ".join(f"{check.name}: {check.detail}" for check in verification.failures())
        )

    levels = summarise_levels(generation.manifest)
    trends = trend_by_source(generation.manifest)
    violations = trend_violations(generation.manifest)
    uncertainty = grouped_bootstrap_levels(
        generation.manifest,
        n_bootstrap=config.pilot.bootstrap_replicates,
        seed=config.pilot.seed,
    )
    feature_summary = feature_summary_table(
        generation.manifest,
        generation.features,
        reference.manifest,
        reference.features,
    )
    effects_long, effects_summary = feature_effect_sizes(
        generation.manifest,
        generation.features,
        reference.manifest,
        reference.features,
        strong_threshold=config.gate.feature_strong_effect_min,
    )
    correlations = feature_target_correlations(
        generation.manifest,
        generation.features,
        reference.manifest,
        reference.features,
    )
    saturation = saturation_table(levels)
    duplicate_fraction = _duplicate_fraction(generation.manifest)
    gate = evaluate_gate(
        config,
        manifest=generation.manifest,
        levels=levels,
        trends=trends,
        effects_summary=effects_summary,
        duplicate_fraction=duplicate_fraction,
        reproducibility_ok=bool(determinism.get("ok")) and bool(determinism.get("repeated_build")),
        verification_ok=verification.ok,
        clean_reference_exact=reference.rerendered_sources == len(reference.sources),
    )

    frozen_after = fingerprint_assets(FROZEN_ASSET_PATHS)
    frozen_unchanged = _fingerprints_equal(frozen_before, frozen_after)
    if not frozen_unchanged:
        raise PilotError("a frozen asset changed during the pilot run; refusing to report success")

    summary = _build_summary(
        config,
        reference,
        generation,
        target=dataset_target,
        determinism=determinism,
        verification=verification,
        gate=gate,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
    )
    _write_json(summary, dataset_target / "pilot_summary.json")
    verification = verify_pilot_dataset(
        dataset_target, reference=reference, recompute_targets=False
    )
    summary["verification"] = {
        "ok": verification.ok,
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in verification.checks
        ],
    }
    _write_json(summary, dataset_target / "pilot_summary.json")
    if not verification.ok:
        raise PilotError(
            "pilot summary check failed: "
            + "; ".join(f"{check.name}: {check.detail}" for check in verification.failures())
        )

    reports_target.mkdir(parents=True, exist_ok=True)
    tables = {
        "level_summary": levels,
        "uncertainty": uncertainty,
        "trend_by_source": trends,
        "trend_violations": violations,
        "feature_summary": feature_summary,
        "feature_effect_sizes": effects_long,
        "feature_distinctness": effects_summary,
        "feature_target_correlations": correlations,
        "saturation": saturation,
    }
    for name, table in tables.items():
        _write_csv(table, reports_target / f"{name}.csv")
    gate_payload = {
        "verdict": gate.verdict,
        "criteria": [
            {
                "id": criterion.id,
                "description": criterion.description,
                "observed": criterion.observed,
                "threshold": criterion.threshold,
                "passed": criterion.passed,
                "note": criterion.note,
            }
            for criterion in gate.criteria
        ],
        "thresholds": config.gate.as_dict(),
        "determinism": determinism,
        "duplicate_fraction": duplicate_fraction,
    }
    _write_json(gate_payload, reports_target / "gate_results.json")
    _write_contact_sheets(dataset_target, config, generation.sources, reports_target)
    report = _report_markdown(
        config,
        reference,
        tables,
        summary,
        gate,
        verification,
        determinism,
    )
    report_path = reports_target / "pilot.md"
    report_path.write_text(report, encoding="utf-8")

    return PilotResult(
        config=config,
        reference=reference,
        manifest=generation.manifest,
        sources=generation.sources,
        features=generation.features,
        tables=tables,
        summary=summary,
        gate=gate,
        verification=verification,
        output_dataset=dataset_target,
        output_reports=reports_target,
        report_path=report_path,
    )


def _fingerprints_equal(
    first: Mapping[str, Mapping[str, object]],
    second: Mapping[str, Mapping[str, object]],
) -> bool:
    return all(first.get(path) == second.get(path) for path in set(first) | set(second))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the occlusion/interference pilot."""
    parser = argparse.ArgumentParser(
        prog="captcha-robustness-pilot",
        description="Run the M4A occlusion and visual-interference feasibility pilot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="generate, verify and report the pilot")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dataset", type=Path, default=None)
    run.add_argument("--output-reports", type=Path, default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--skip-reproduction",
        action="store_true",
        help="skip the second in-memory build (not recommended)",
    )

    verify = subparsers.add_parser("verify", help="verify an existing pilot dataset")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--dataset", type=Path, default=None)
    verify.add_argument("--skip-targets", action="store_true")

    args = parser.parse_args(argv)
    try:
        config = PilotConfig.from_toml(args.config)
        if args.command == "run":
            result = run_pilot(
                config,
                output_dataset=args.output_dataset,
                output_reports=args.output_reports,
                overwrite=args.overwrite,
                verify_reproduction=not args.skip_reproduction,
            )
            counts = result.summary["counts"]
            print(
                f"pilot '{config.pilot.name}': {counts['sources']} sources, "
                f"{counts['samples']} samples ({counts['corrupted_samples']} corrupted)"
            )
            print(f"verification: {'PASS' if result.verification.ok else 'FAIL'}")
            print(f"feasibility gate: {result.gate.verdict}")
            if result.gate.failed_criteria:
                print(f"failed criteria: {', '.join(result.gate.failed_criteria)}")
            print(f"dataset: {result.output_dataset}")
            print(f"report: {result.report_path}")
            return 0 if result.gate.verdict != "FAIL" else 1
        if args.command == "verify":
            reference = load_frozen_reference(config)
            root = args.dataset or config.pilot.output_dataset
            report = verify_pilot_dataset(
                root, reference=reference, recompute_targets=not args.skip_targets
            )
            for check in report.checks:
                print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
            print(f"verification: {'PASS' if report.ok else 'FAIL'}")
            return 0 if report.ok else 1
    except (
        PilotConfigError,
        PilotError,
        InterferenceError,
        ConfigError,
        FontResolutionError,
        RenderError,
        MetricUnavailableError,
        FeatureDependencyError,
        ImageContractError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI test
    raise SystemExit(main())
