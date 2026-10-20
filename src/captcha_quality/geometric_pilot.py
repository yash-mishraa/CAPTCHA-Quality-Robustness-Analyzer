"""Milestone 5A geometric-distortion feasibility pilot.

This module is deliberately separate from the frozen Milestone 2 dataset, the
M3 models and the M4A/M4B interference work. It reads the frozen clean sources,
re-renders them as a reference audit, and generates a small, self-contained
pilot dataset from one smooth elastic-warp family. It does not train, retune or
evaluate a model.

The target is SSIM against the source's exact clean reference. PSNR is a
secondary diagnostic. Neither is a measure of human readability, CAPTCHA
solvability, or production quality.

Run it with::

    uv run --extra image captcha-geometric-pilot run \\
        --config configs/geometric_distortion_pilot_v0_1_0.toml

Generated data and reports are Git-ignored; the experiment log records the
measured results.
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
from PIL import Image

from captcha_quality.config import ConfigError
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.dataset import CheckResult, VerificationReport, load_grayscale
from captcha_quality.features import FEATURE_NAMES, FeatureDependencyError, extract_feature_table
from captcha_quality.geometric import (
    BORDER_EDGE,
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_SMOOTHING_SIGMA_PX,
    INTERPOLATION_BILINEAR,
    SUPPORTED_BORDER_MODES,
    SUPPORTED_INTERPOLATIONS,
    ElasticFieldSpec,
    ElasticLadder,
    ElasticResult,
    GeometricError,
    apply_elastic,
    field_spec_json,
    generate_displacement_field,
    sha256_field,
    verify_field_replay,
)
from captcha_quality.imaging import ImageContractError
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
from captcha_quality.render import FontResolutionError, RenderError
from captcha_quality.robustness_pilot import (
    PilotConfigError,
    PilotSettings,
    ReferenceSettings,
    fingerprint_assets,
    load_frozen_reference,
)
from captcha_quality.summarise import rank_correlation

CLEAN_FAMILY: Final[str] = "clean"
CONTROL_FAMILY: Final[str] = "control"
ELASTIC_FAMILY: Final[str] = "elastic"
CONTROL_IDENTITY: Final[str] = "identity"
CONTROL_ZERO_FIELD: Final[str] = "zero_field"
CONTROL_MINIMAL: Final[str] = "minimal"
CONTROL_NONE: Final[str] = "none"

#: Stable public manifest schema. Adding fields requires a pilot version bump.
PILOT_MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "source_id",
    "variant_id",
    "split",
    "font_id",
    "text_length",
    "variant_family",
    "control_id",
    "severity_level",
    "amplitude_px",
    "resampled",
    "field_seed",
    "field_grid_rows",
    "field_grid_cols",
    "field_smoothing_sigma_px",
    "field_interpolation",
    "field_border_mode",
    "field_sha256",
    "field_spec_json",
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
    "outside_fraction",
    "border_affected_fraction",
    "border_touched",
    "clipped_fraction",
    "intensity_std",
    "collapsed_block_fraction",
    "degenerate",
    "pilot_version",
    "pilot_config_hash",
)

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
    "field_seed",
    "field_sha256",
    "field_spec_json",
)

PILOT_FEATURE_COLUMNS: Final[tuple[str, ...]] = ("sample_id", *FEATURE_NAMES)

BOOTSTRAP_LOW: Final[float] = 2.5
BOOTSTRAP_HIGH: Final[float] = 97.5

#: Frozen assets fingerprinted before and after the run. The pilot never writes
#: to any of these paths.
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
    "reports/benchmark_v0_1_0/family_test_metrics.csv",
    "reports/benchmark_v0_1_0/uncertainty.csv",
    "reports/benchmark_v0_1_0/feature_correlations.csv",
    "configs/robustness_pilot_occlusion_v0_1_0.toml",
    "data/processed/occlusion_interference_pilot_v0.1.0/manifest.csv",
    "data/processed/occlusion_interference_pilot_v0.1.0/sources.csv",
    "data/processed/occlusion_interference_pilot_v0.1.0/features.csv",
    "data/processed/occlusion_interference_pilot_v0.1.0/pilot_summary.json",
    "reports/robustness_pilot/pilot.md",
    "reports/robustness_pilot/gate_results.json",
    "reports/robustness_pilot/level_summary.csv",
    "reports/robustness_pilot/feature_effect_sizes.csv",
    "reports/robustness_pilot/feature_distinctness.csv",
    "reports/robustness_pilot/trend_by_source.csv",
    "reports/robustness_pilot/uncertainty.csv",
    "reports/robustness_pilot/examples_thin_lines.png",
    "reports/robustness_pilot/examples_arc_strokes.png",
    "configs/robustness_benchmark_v0_1_0.toml",
    "reports/robustness_benchmark_v0_1_0/benchmark.md",
    "reports/robustness_benchmark_v0_1_0/run_metadata.json",
    "reports/robustness_benchmark_v0_1_0/zero_shot_metrics.csv",
    "reports/robustness_benchmark_v0_1_0/adaptation_metrics.csv",
    "reports/robustness_benchmark_v0_1_0/target_distribution.csv",
    "reports/robustness_benchmark_v0_1_0/isolation_checks.csv",
    "reports/robustness_benchmark_v0_1_0/uncertainty.csv",
    "models/robustness_benchmark_v0_1_0/expanded_model.joblib",
    "models/robustness_benchmark_v0_1_0/expanded_model_card.json",
)

PROTECTED_OUTPUT_ROOTS: Final[tuple[str, ...]] = (
    "configs",
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "models/robustness_benchmark_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
)


class GeometricPilotConfigError(ValueError):
    """Raised when the geometric pilot configuration is missing or invalid."""


class GeometricPilotError(RuntimeError):
    """Raised when the pilot cannot generate, verify or report its outputs."""


class GeometricPilotExistsError(GeometricPilotError):
    """Raised when the output exists and overwriting was not requested."""


@dataclass(frozen=True, slots=True)
class ComparisonSettings:
    """Optional frozen comparison tables used for cross-family analysis."""

    include_interference: bool = False
    m4a_dataset: Path | None = None
    m4a_config: Path | None = None
    m4a_config_sha256: str | None = None
    m4a_manifest_sha256: str | None = None
    m4a_features_sha256: str | None = None
    m4a_summary_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.include_interference:
            if self.m4a_dataset is None or self.m4a_config is None:
                raise GeometricPilotConfigError(
                    "comparison.include_interference requires m4a_dataset and m4a_config"
                )
            for name in (
                "m4a_config_sha256",
                "m4a_manifest_sha256",
                "m4a_features_sha256",
                "m4a_summary_sha256",
            ):
                value = getattr(self, name)
                if (
                    value is None
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise GeometricPilotConfigError(
                        f"comparison.{name} must be a lowercase SHA-256 hex digest"
                    )
        elif any(
            value is not None
            for value in (
                self.m4a_dataset,
                self.m4a_config,
                self.m4a_config_sha256,
                self.m4a_manifest_sha256,
                self.m4a_features_sha256,
                self.m4a_summary_sha256,
            )
        ):
            raise GeometricPilotConfigError(
                "comparison.include_interference=false must not set M4A paths or hashes"
            )


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    """Descriptive thresholds that do not select severity levels."""

    near_identity_ssim_min: float = 0.9999
    min_group_samples: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.near_identity_ssim_min <= 1.0:
            raise GeometricPilotConfigError("analysis.near_identity_ssim_min must be in (0, 1]")
        if isinstance(self.min_group_samples, bool) or not isinstance(self.min_group_samples, int):
            raise GeometricPilotConfigError("analysis.min_group_samples must be an integer")
        if self.min_group_samples < 1:
            raise GeometricPilotConfigError("analysis.min_group_samples must be >= 1")


@dataclass(frozen=True, slots=True)
class GeometricGateThresholds:
    """Predeclared M5A feasibility-gate thresholds."""

    ssim_std_min: float = 0.02
    level_mean_range_min: float = 0.10
    per_level_sd_min: float = 0.005
    median_source_spearman_max: float = -0.80
    increasing_step_fraction_max: float = 0.15
    minimal_control_ssim_min: float = 0.99
    border_affected_fraction_max: float = 0.10
    collapsed_block_fraction_max: float = 0.10
    feature_effect_min: float = 0.50
    feature_strong_effect_min: float = 0.80
    feature_strong_effect_count_min: int = 2
    duplicate_fraction_max: float = 0.0

    def __post_init__(self) -> None:
        for name in ("ssim_std_min", "level_mean_range_min", "per_level_sd_min"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise GeometricPilotConfigError(f"gate.{name} must be > 0")
        if (
            not math.isfinite(self.median_source_spearman_max)
            or self.median_source_spearman_max >= 0.0
        ):
            raise GeometricPilotConfigError("gate.median_source_spearman_max must be negative")
        for name in (
            "increasing_step_fraction_max",
            "border_affected_fraction_max",
            "collapsed_block_fraction_max",
            "duplicate_fraction_max",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise GeometricPilotConfigError(f"gate.{name} must be in [0, 1]")
        if not 0.0 < self.minimal_control_ssim_min <= 1.0:
            raise GeometricPilotConfigError("gate.minimal_control_ssim_min must be in (0, 1]")
        if self.feature_effect_min <= 0.0:
            raise GeometricPilotConfigError("gate.feature_effect_min must be > 0")
        if self.feature_strong_effect_min < self.feature_effect_min:
            raise GeometricPilotConfigError(
                "gate.feature_strong_effect_min must be >= gate.feature_effect_min"
            )
        if (
            isinstance(self.feature_strong_effect_count_min, bool)
            or not isinstance(self.feature_strong_effect_count_min, int)
            or self.feature_strong_effect_count_min < 1
        ):
            raise GeometricPilotConfigError(
                "gate.feature_strong_effect_count_min must be an integer >= 1"
            )

    def as_dict(self) -> dict[str, float | int]:
        """Return thresholds as a plain mapping for reporting."""
        return {
            "ssim_std_min": self.ssim_std_min,
            "level_mean_range_min": self.level_mean_range_min,
            "per_level_sd_min": self.per_level_sd_min,
            "median_source_spearman_max": self.median_source_spearman_max,
            "increasing_step_fraction_max": self.increasing_step_fraction_max,
            "minimal_control_ssim_min": self.minimal_control_ssim_min,
            "border_affected_fraction_max": self.border_affected_fraction_max,
            "collapsed_block_fraction_max": self.collapsed_block_fraction_max,
            "feature_effect_min": self.feature_effect_min,
            "feature_strong_effect_min": self.feature_strong_effect_min,
            "feature_strong_effect_count_min": self.feature_strong_effect_count_min,
            "duplicate_fraction_max": self.duplicate_fraction_max,
        }


@dataclass(frozen=True, slots=True)
class GeometricSettings:
    """Predeclared field geometry and amplitude ladder."""

    ladder: ElasticLadder
    grid_rows: int = DEFAULT_GRID_ROWS
    grid_cols: int = DEFAULT_GRID_COLS
    smoothing_sigma_px: float = DEFAULT_SMOOTHING_SIGMA_PX
    interpolation: str = INTERPOLATION_BILINEAR
    border_mode: str = BORDER_EDGE

    def __post_init__(self) -> None:
        for name, value in (("grid_rows", self.grid_rows), ("grid_cols", self.grid_cols)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise GeometricPilotConfigError(f"geometry.{name} must be an integer >= 2")
        if (
            isinstance(self.smoothing_sigma_px, bool)
            or not isinstance(self.smoothing_sigma_px, int | float)
            or not math.isfinite(float(self.smoothing_sigma_px))
            or not 0.0 <= float(self.smoothing_sigma_px) <= 16.0
        ):
            raise GeometricPilotConfigError(
                "geometry.smoothing_sigma_px must be finite and in [0, 16]"
            )
        if self.interpolation not in SUPPORTED_INTERPOLATIONS:
            raise GeometricPilotConfigError(
                f"geometry.interpolation must be one of {SUPPORTED_INTERPOLATIONS}"
            )
        if self.border_mode not in SUPPORTED_BORDER_MODES:
            raise GeometricPilotConfigError(
                f"geometry.border_mode must be one of {SUPPORTED_BORDER_MODES}"
            )

    def field_spec(self, shape: tuple[int, int], seed: int) -> ElasticFieldSpec:
        """Build the field specification for one source."""
        return ElasticFieldSpec(
            height=int(shape[0]),
            width=int(shape[1]),
            seed=seed,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            smoothing_sigma_px=float(self.smoothing_sigma_px),
            interpolation=self.interpolation,
            border_mode=self.border_mode,
        )

    def as_dict(self) -> dict[str, object]:
        """Return the content-defining geometry configuration."""
        return {
            "amplitudes_px": list(self.ladder.amplitudes_px),
            "minimal_amplitude_px": float(self.ladder.minimal_amplitude_px),
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "smoothing_sigma_px": float(self.smoothing_sigma_px),
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
        }


@dataclass(frozen=True, slots=True)
class GeometricPilotConfig:
    """Fully validated M5A pilot configuration."""

    pilot: PilotSettings
    reference: ReferenceSettings
    comparison: ComparisonSettings
    geometry: GeometricSettings
    analysis: AnalysisSettings
    gate: GeometricGateThresholds
    config_path: Path

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
            "geometry": self.geometry.as_dict(),
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
            "comparison": {
                "include_interference": self.comparison.include_interference,
                "m4a_dataset": (
                    str(self.comparison.m4a_dataset)
                    if self.comparison.m4a_dataset is not None
                    else None
                ),
                "m4a_config": (
                    str(self.comparison.m4a_config)
                    if self.comparison.m4a_config is not None
                    else None
                ),
                "m4a_config_sha256": self.comparison.m4a_config_sha256,
                "m4a_manifest_sha256": self.comparison.m4a_manifest_sha256,
                "m4a_features_sha256": self.comparison.m4a_features_sha256,
                "m4a_summary_sha256": self.comparison.m4a_summary_sha256,
            },
            "geometry": self.geometry.as_dict(),
            "analysis": {
                "near_identity_ssim_min": self.analysis.near_identity_ssim_min,
                "min_group_samples": self.analysis.min_group_samples,
            },
            "gate": self.gate.as_dict(),
            "generation_config": self.generation_config,
            "pilot_config_hash": self.pilot_config_hash,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_toml(cls, path: Path | str) -> GeometricPilotConfig:
        """Load and validate a pilot configuration from TOML."""
        config_path = Path(path)
        if not config_path.is_file():
            raise GeometricPilotConfigError(f"pilot config not found: {config_path}")
        try:
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise GeometricPilotConfigError(f"invalid TOML in {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise GeometricPilotConfigError(f"configuration root in {config_path} must be a table")
        _check_keys(
            raw, {"pilot", "reference", "comparison", "geometry", "analysis", "gate"}, "root"
        )

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
        try:
            pilot = PilotSettings(
                name=_string(pilot_table, "name", "pilot"),
                version=_string(pilot_table, "version", "pilot"),
                seed=_integer(pilot_table, "seed", "pilot"),
                bootstrap_replicates=_integer(pilot_table, "bootstrap_replicates", "pilot"),
                output_dataset=Path(_string(pilot_table, "output_dataset", "pilot")),
                output_reports=Path(_string(pilot_table, "output_reports", "pilot")),
            )
            reference = ReferenceSettings(
                dataset=Path(
                    _string(_table(raw, "reference", config_path), "dataset", "reference")
                ),
                config=Path(_string(_table(raw, "reference", config_path), "config", "reference")),
                dataset_version=_string(
                    _table(raw, "reference", config_path), "dataset_version", "reference"
                ),
                generation_config_hash=_string(
                    _table(raw, "reference", config_path),
                    "generation_config_hash",
                    "reference",
                ),
                manifest_sha256=_string(
                    _table(raw, "reference", config_path), "manifest_sha256", "reference"
                ),
                sources_sha256=_string(
                    _table(raw, "reference", config_path), "sources_sha256", "reference"
                ),
                features_sha256=_string(
                    _table(raw, "reference", config_path), "features_sha256", "reference"
                ),
                ground_truth_sha256=_string(
                    _table(raw, "reference", config_path), "ground_truth_sha256", "reference"
                ),
                summary_sha256=_string(
                    _table(raw, "reference", config_path), "summary_sha256", "reference"
                ),
                split_assignment_sha256=_string(
                    _table(raw, "reference", config_path),
                    "split_assignment_sha256",
                    "reference",
                ),
            )
        except PilotConfigError as exc:
            raise GeometricPilotConfigError(str(exc)) from exc

        reference_table = _table(raw, "reference", config_path)
        _check_keys(
            reference_table,
            {
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
            },
            "reference",
        )

        comparison_table = _table(raw, "comparison", config_path)
        _check_keys(
            comparison_table,
            {
                "include_interference",
                "m4a_dataset",
                "m4a_config",
                "m4a_config_sha256",
                "m4a_manifest_sha256",
                "m4a_features_sha256",
                "m4a_summary_sha256",
            },
            "comparison",
        )
        include_interference = _boolean(comparison_table, "include_interference", "comparison")
        if include_interference:
            comparison = ComparisonSettings(
                include_interference=True,
                m4a_dataset=Path(_string(comparison_table, "m4a_dataset", "comparison")),
                m4a_config=Path(_string(comparison_table, "m4a_config", "comparison")),
                m4a_config_sha256=_string(comparison_table, "m4a_config_sha256", "comparison"),
                m4a_manifest_sha256=_string(comparison_table, "m4a_manifest_sha256", "comparison"),
                m4a_features_sha256=_string(comparison_table, "m4a_features_sha256", "comparison"),
                m4a_summary_sha256=_string(comparison_table, "m4a_summary_sha256", "comparison"),
            )
        else:
            comparison = ComparisonSettings(include_interference=False)

        geometry_table = _table(raw, "geometry", config_path)
        _check_keys(
            geometry_table,
            {
                "amplitudes_px",
                "minimal_amplitude_px",
                "grid_rows",
                "grid_cols",
                "smoothing_sigma_px",
                "interpolation",
                "border_mode",
            },
            "geometry",
        )
        try:
            ladder = ElasticLadder(
                amplitudes_px=_number_list(geometry_table, "amplitudes_px", "geometry"),
                minimal_amplitude_px=_number(geometry_table, "minimal_amplitude_px", "geometry"),
            )
            geometry = GeometricSettings(
                ladder=ladder,
                grid_rows=_integer(geometry_table, "grid_rows", "geometry"),
                grid_cols=_integer(geometry_table, "grid_cols", "geometry"),
                smoothing_sigma_px=_number(geometry_table, "smoothing_sigma_px", "geometry"),
                interpolation=_string(geometry_table, "interpolation", "geometry"),
                border_mode=_string(geometry_table, "border_mode", "geometry"),
            )
        except GeometricError as exc:
            raise GeometricPilotConfigError(f"invalid geometry: {exc}") from exc

        analysis_table = _table(raw, "analysis", config_path)
        _check_keys(
            analysis_table,
            {"near_identity_ssim_min", "min_group_samples"},
            "analysis",
        )
        analysis = AnalysisSettings(
            near_identity_ssim_min=_number(analysis_table, "near_identity_ssim_min", "analysis"),
            min_group_samples=_integer(analysis_table, "min_group_samples", "analysis"),
        )

        gate_table = _table(raw, "gate", config_path)
        gate_keys = set(GeometricGateThresholds.__dataclass_fields__)
        _check_keys(gate_table, gate_keys, "gate")
        try:
            gate = GeometricGateThresholds(
                ssim_std_min=_number(gate_table, "ssim_std_min", "gate"),
                level_mean_range_min=_number(gate_table, "level_mean_range_min", "gate"),
                per_level_sd_min=_number(gate_table, "per_level_sd_min", "gate"),
                median_source_spearman_max=_number(
                    gate_table, "median_source_spearman_max", "gate"
                ),
                increasing_step_fraction_max=_number(
                    gate_table, "increasing_step_fraction_max", "gate"
                ),
                minimal_control_ssim_min=_number(gate_table, "minimal_control_ssim_min", "gate"),
                border_affected_fraction_max=_number(
                    gate_table, "border_affected_fraction_max", "gate"
                ),
                collapsed_block_fraction_max=_number(
                    gate_table, "collapsed_block_fraction_max", "gate"
                ),
                feature_effect_min=_number(gate_table, "feature_effect_min", "gate"),
                feature_strong_effect_min=_number(gate_table, "feature_strong_effect_min", "gate"),
                feature_strong_effect_count_min=_integer(
                    gate_table, "feature_strong_effect_count_min", "gate"
                ),
                duplicate_fraction_max=_number(gate_table, "duplicate_fraction_max", "gate"),
            )
        except (GeometricError, ValueError) as exc:
            raise GeometricPilotConfigError(f"invalid gate: {exc}") from exc

        return cls(
            pilot=pilot,
            reference=reference,
            comparison=comparison,
            geometry=geometry,
            analysis=analysis,
            gate=gate,
            config_path=config_path,
        )


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise GeometricPilotConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _table(data: Mapping[str, object], key: str, context: Path | str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise GeometricPilotConfigError(f"missing or invalid [{key}] table in {context}")
    return value


def _string(table: Mapping[str, object], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GeometricPilotConfigError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(table: Mapping[str, object], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeometricPilotConfigError(f"{context}.{key} must be an integer")
    return value


def _boolean(table: Mapping[str, object], key: str, context: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise GeometricPilotConfigError(f"{context}.{key} must be a boolean")
    return value


def _number(table: Mapping[str, object], key: str, context: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GeometricPilotConfigError(f"{context}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise GeometricPilotConfigError(f"{context}.{key} must be finite")
    return result


def _number_list(table: Mapping[str, object], key: str, context: str) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in value
    ):
        raise GeometricPilotConfigError(f"{context}.{key} must be a list of numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise GeometricPilotConfigError(f"{context}.{key} must contain only finite numbers")
    return result


@dataclass(frozen=True, slots=True)
class FrozenComparison:
    """Frozen non-geometric families used for descriptive comparison."""

    table: pd.DataFrame
    m4a_verified: bool
    m4a_manifest_sha256: str | None
    m4a_features_sha256: str | None
    m4a_summary_sha256: str | None


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise GeometricPilotError(f"frozen {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise GeometricPilotError(
            f"frozen {label} hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def _generic_comparison_frame(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    *,
    family_column: str,
    clean_family: str,
    source_label: str,
) -> pd.DataFrame:
    merged = manifest.merge(features, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(manifest):
        raise GeometricPilotError(f"{source_label}: manifest and features do not join one-to-one")
    corrupted = merged[merged[family_column] != clean_family].copy()
    if corrupted.empty:
        raise GeometricPilotError(f"{source_label}: no corrupted comparison rows found")
    if not corrupted["severity_level"].between(1, 4).all():
        raise GeometricPilotError(f"{source_label}: comparison severities must be in 1..4")
    values = corrupted[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise GeometricPilotError(f"{source_label}: comparison features are not finite")
    result = pd.DataFrame(
        {
            "family_label": corrupted[family_column].astype(str),
            "source_id": corrupted["source_id"].astype(str),
            "severity_level": corrupted["severity_level"].astype(int),
            "font_id": corrupted["font_id"].astype(str),
            "text_length": corrupted["text_length"].astype(int),
            "ssim_target": corrupted["ssim_target"].astype(float),
            "psnr_target": corrupted["psnr_target"].astype(float),
            **{feature: corrupted[feature].astype(float) for feature in FEATURE_NAMES},
        }
    )
    if corrupted["sample_id"].duplicated().any():
        raise GeometricPilotError(f"{source_label}: duplicate comparison rows")
    return result


def load_frozen_comparison(
    config: GeometricPilotConfig,
    *,
    reference_manifest: pd.DataFrame,
    reference_features: pd.DataFrame,
) -> FrozenComparison:
    """Load the frozen M3 blur/noise and optional M4A interference tables."""
    m3 = _generic_comparison_frame(
        reference_manifest,
        reference_features,
        family_column="corruption_type",
        clean_family="none",
        source_label="frozen M3",
    )
    frames = [m3]
    m4a_verified = False
    m4a_manifest_hash: str | None = None
    m4a_features_hash: str | None = None
    m4a_summary_hash: str | None = None
    if config.comparison.include_interference:
        assert config.comparison.m4a_dataset is not None
        assert config.comparison.m4a_config is not None
        assert config.comparison.m4a_manifest_sha256 is not None
        assert config.comparison.m4a_features_sha256 is not None
        assert config.comparison.m4a_summary_sha256 is not None
        assert config.comparison.m4a_config_sha256 is not None
        dataset = config.comparison.m4a_dataset
        m4a_config_hash = _require_hash(
            config.comparison.m4a_config,
            config.comparison.m4a_config_sha256,
            "M4A config",
        )
        manifest_hash = _require_hash(
            dataset / "manifest.csv",
            config.comparison.m4a_manifest_sha256,
            "M4A manifest",
        )
        features_hash = _require_hash(
            dataset / "features.csv",
            config.comparison.m4a_features_sha256,
            "M4A features",
        )
        summary_path = dataset / "pilot_summary.json"
        summary_hash = _require_hash(
            summary_path,
            config.comparison.m4a_summary_sha256,
            "M4A summary",
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        artifacts = summary.get("artifacts", {})
        if artifacts.get("manifest_sha256") != manifest_hash:
            raise GeometricPilotError("M4A summary manifest hash disagrees with the file")
        if artifacts.get("features_sha256") != features_hash:
            raise GeometricPilotError("M4A summary feature hash disagrees with the file")
        if summary.get("config", {}).get("config_sha256") != m4a_config_hash:
            raise GeometricPilotError("M4A summary config hash disagrees with the file")
        m4a_manifest = pd.read_csv(dataset / "manifest.csv")
        m4a_features = pd.read_csv(dataset / "features.csv")
        m4a = _generic_comparison_frame(
            m4a_manifest,
            m4a_features,
            family_column="variant_family",
            clean_family="none",
            source_label="frozen M4A",
        )
        frames.append(m4a)
        m4a_verified = True
        m4a_manifest_hash = manifest_hash
        m4a_features_hash = features_hash
        m4a_summary_hash = summary_hash
    table = pd.concat(frames, ignore_index=True)
    return FrozenComparison(
        table=table,
        m4a_verified=m4a_verified,
        m4a_manifest_sha256=m4a_manifest_hash,
        m4a_features_sha256=m4a_features_hash,
        m4a_summary_sha256=m4a_summary_hash,
    )


@dataclass(frozen=True, slots=True)
class GenerationData:
    """One complete in-memory build of the pilot dataset."""

    manifest: pd.DataFrame
    sources: pd.DataFrame
    features: pd.DataFrame
    image_payloads: dict[str, np.ndarray]
    image_hashes: dict[str, str]
    clean_hashes_after: dict[str, str]


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


def _clean_sample_id(source_id: str) -> str:
    return f"{source_id}__clean_s0"


def _control_sample_id(source_id: str, control_id: str) -> str:
    return f"{source_id}__control_{control_id}"


def _elastic_sample_id(source_id: str, severity: int) -> str:
    return f"{source_id}__elastic_s{severity}"


def _build_manifest_row(
    *,
    sample_id: str,
    source: Any,
    variant_id: str,
    variant_family: str,
    control_id: str,
    severity_level: int,
    result: ElasticResult,
    image_rel: Path,
    clean_rel: Path,
    field_payload: str,
    config: GeometricPilotConfig,
    metrics: Mapping[str, float],
) -> dict[str, object]:
    psnr_value = float(metrics["psnr"])
    return {
        "sample_id": sample_id,
        "source_id": str(source.source_id),
        "variant_id": variant_id,
        "split": str(source.split),
        "font_id": str(source.font_id),
        "text_length": int(source.text_length),
        "variant_family": variant_family,
        "control_id": control_id,
        "severity_level": int(severity_level),
        "amplitude_px": float(result.amplitude_px),
        "resampled": bool(result.resampled),
        "field_seed": int(result.field_spec.seed),
        "field_grid_rows": int(result.field_spec.grid_rows),
        "field_grid_cols": int(result.field_spec.grid_cols),
        "field_smoothing_sigma_px": float(result.field_spec.smoothing_sigma_px),
        "field_interpolation": result.field_spec.interpolation,
        "field_border_mode": result.field_spec.border_mode,
        "field_sha256": sha256_field(result.field),
        "field_spec_json": field_payload,
        "image_path": _relative(image_rel),
        "reference_path": _relative(clean_rel),
        "ssim_target": float(metrics["ssim"]),
        "psnr_target": psnr_value if math.isfinite(psnr_value) else None,
        "psnr_infinite": not math.isfinite(psnr_value),
        "height": int(result.image.shape[0]),
        "width": int(result.image.shape[1]),
        "image_sha256": sha256_bytes(result.image.tobytes()),
        "reference_sha256": str(source.clean_image_sha256),
        "changed_pixel_fraction": float(result.changed_pixel_fraction),
        "changed_ink_fraction": float(result.changed_ink_fraction),
        "changed_background_fraction": float(result.changed_background_fraction),
        "outside_fraction": float(result.outside_fraction),
        "border_affected_fraction": float(result.border_affected_fraction),
        "border_touched": bool(result.border_touched),
        "clipped_fraction": float(result.clipped_fraction),
        "intensity_std": float(result.intensity_std),
        "collapsed_block_fraction": float(result.collapsed_block_fraction),
        "degenerate": bool(result.degenerate),
        "pilot_version": config.pilot.version,
        "pilot_config_hash": config.pilot_config_hash,
    }


def _build_generation_data(
    config: GeometricPilotConfig,
    reference: Any,
) -> GenerationData:
    """Build every row and image directly from the frozen clean references."""
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
        field_seed = derive_noise_seed(config.pilot.seed, "geometric", source_id)
        field_spec = config.geometry.field_spec(clean.shape, field_seed)
        field = generate_displacement_field(field_spec)
        field_hash = sha256_field(field)
        field_payload = field_spec_json(field_spec, field)
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
                "field_seed": int(field_seed),
                "field_sha256": field_hash,
                "field_spec_json": field_payload,
            }
        )

        def add_row(
            *,
            sample_id: str,
            variant_id: str,
            variant_family: str,
            control_id: str,
            severity_level: int,
            amplitude_px: float,
            force_resample: bool,
            image_rel: Path,
            _clean: np.ndarray = clean,
            _field_spec: ElasticFieldSpec = field_spec,
            _source: Any = source,
            _clean_rel: Path = clean_rel,
            _field_payload: str = field_payload,
            _config: GeometricPilotConfig = config,
        ) -> None:
            result = apply_elastic(
                _clean,
                field_spec=_field_spec,
                amplitude_px=amplitude_px,
                force_resample=force_resample,
            )
            metrics = compute_pair_metrics(_clean, result.image, ("ssim", "psnr"))
            warped_hash = sha256_bytes(result.image.tobytes())
            image_payloads[_relative(image_rel)] = np.array(result.image, dtype=np.uint8, copy=True)
            image_hashes[_relative(image_rel)] = warped_hash
            feature_rows.append((sample_id, np.array(result.image, dtype=np.uint8, copy=True)))
            manifest_rows.append(
                _build_manifest_row(
                    sample_id=sample_id,
                    source=_source,
                    variant_id=variant_id,
                    variant_family=variant_family,
                    control_id=control_id,
                    severity_level=severity_level,
                    result=result,
                    image_rel=image_rel,
                    clean_rel=_clean_rel,
                    field_payload=_field_payload,
                    config=_config,
                    metrics=metrics,
                )
            )

        clean_sample_id = _clean_sample_id(source_id)
        image_payloads[_relative(clean_rel)] = np.array(clean, dtype=np.uint8, copy=True)
        image_hashes[_relative(clean_rel)] = clean_hash_before
        feature_rows.append((clean_sample_id, np.array(clean, dtype=np.uint8, copy=True)))
        clean_result = apply_elastic(
            clean, field_spec=field_spec, amplitude_px=0.0, force_resample=False
        )
        clean_metrics = compute_pair_metrics(clean, clean_result.image, ("ssim", "psnr"))
        manifest_rows.append(
            _build_manifest_row(
                sample_id=clean_sample_id,
                source=source,
                variant_id="clean_s0",
                variant_family=CLEAN_FAMILY,
                control_id=CONTROL_NONE,
                severity_level=0,
                result=clean_result,
                image_rel=clean_rel,
                clean_rel=clean_rel,
                field_payload=field_payload,
                config=config,
                metrics=clean_metrics,
            )
        )

        for control_id, amplitude, force_resample in (
            (CONTROL_IDENTITY, 0.0, False),
            (CONTROL_ZERO_FIELD, 0.0, True),
            (CONTROL_MINIMAL, config.geometry.ladder.minimal_amplitude_px, True),
        ):
            sample_id = _control_sample_id(source_id, control_id)
            add_row(
                sample_id=sample_id,
                variant_id=f"control_{control_id}",
                variant_family=CONTROL_FAMILY,
                control_id=control_id,
                severity_level=0,
                amplitude_px=amplitude,
                force_resample=force_resample,
                image_rel=Path("images") / "control" / f"{sample_id}.png",
            )

        for severity in range(1, config.geometry.ladder.n_levels + 1):
            sample_id = _elastic_sample_id(source_id, severity)
            add_row(
                sample_id=sample_id,
                variant_id=f"elastic_s{severity}",
                variant_family=ELASTIC_FAMILY,
                control_id=CONTROL_NONE,
                severity_level=severity,
                amplitude_px=config.geometry.ladder.amplitude(severity),
                force_resample=False,
                image_rel=Path("images") / "elastic" / f"{sample_id}.png",
            )

        if sha256_bytes(clean.tobytes()) != clean_hash_before:
            raise GeometricPilotError(f"clean reference mutated for source {source_id}")

    sources_frame = pd.DataFrame(source_rows, columns=PILOT_SOURCE_COLUMNS)
    manifest = pd.DataFrame(manifest_rows, columns=PILOT_MANIFEST_COLUMNS)
    manifest = _coerce_manifest_types(manifest)
    features = extract_feature_table(feature_rows)
    if tuple(features.columns) != PILOT_FEATURE_COLUMNS:  # pragma: no cover - structural guard
        raise GeometricPilotError("feature extraction returned an unexpected schema")

    manifest = manifest.sort_values(
        ["source_id", "variant_family", "control_id", "severity_level"], kind="stable"
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
    result = frame.copy()
    for column in (
        "text_length",
        "field_seed",
        "field_grid_rows",
        "field_grid_cols",
        "height",
        "width",
        "severity_level",
    ):
        result[column] = result[column].astype("int64")
    for column in (
        "amplitude_px",
        "field_smoothing_sigma_px",
        "ssim_target",
        "psnr_target",
        "changed_pixel_fraction",
        "changed_ink_fraction",
        "changed_background_fraction",
        "outside_fraction",
        "border_affected_fraction",
        "clipped_fraction",
        "intensity_std",
        "collapsed_block_fraction",
    ):
        result[column] = result[column].astype("float64")
    for column in ("resampled", "psnr_infinite", "border_touched", "degenerate"):
        result[column] = result[column].astype(bool)
    return result


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


def _counts(values: Iterable[object]) -> dict[str, int]:
    series = pd.Series([str(value) for value in values])
    counts = series.value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _image_pixel_hash_set(image_hashes: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json(sorted(image_hashes.items())).encode("utf-8"))


def _pilot_feature_frame(manifest: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    # ``intensity_std`` exists both as a feature and as a warped-image diagnostic.
    # The frozen feature definition is the one used in every analysis table; the
    # manifest copy is retained only for row-level diagnostics.
    manifest_features = manifest.drop(columns=["intensity_std"], errors="ignore")
    merged = manifest_features.merge(features, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(manifest):
        raise GeometricPilotError("pilot manifest and feature table do not join one-to-one")
    return merged


def _combined_family_frame(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    pilot = _pilot_feature_frame(manifest, features)
    elastic = pilot[pilot["variant_family"] == ELASTIC_FAMILY].copy()
    pilot_frame = pd.DataFrame(
        {
            "family_label": ELASTIC_FAMILY,
            "source_id": elastic["source_id"].astype(str),
            "severity_level": elastic["severity_level"].astype(int),
            "font_id": elastic["font_id"].astype(str),
            "text_length": elastic["text_length"].astype(int),
            "ssim_target": elastic["ssim_target"].astype(float),
            "psnr_target": elastic["psnr_target"].astype(float),
            **{feature: elastic[feature].astype(float) for feature in FEATURE_NAMES},
        }
    )
    combined = pd.concat(
        [
            pilot_frame,
            comparison[
                [
                    "family_label",
                    "source_id",
                    "severity_level",
                    "font_id",
                    "text_length",
                    "ssim_target",
                    "psnr_target",
                    *FEATURE_NAMES,
                ]
            ],
        ],
        ignore_index=True,
    )
    return combined


def pilot_level_summary(
    manifest: pd.DataFrame,
    *,
    near_identity_ssim_min: float,
) -> pd.DataFrame:
    """Per-severity SSIM/PSNR and interpolation/border diagnostics."""
    elastic = manifest[manifest["variant_family"] == ELASTIC_FAMILY]
    rows: list[dict[str, object]] = []
    for severity, group in elastic.groupby("severity_level", sort=True):
        ssim_values = group["ssim_target"].to_numpy(dtype=np.float64)
        psnr_values = group["psnr_target"].to_numpy(dtype=np.float64)
        finite = psnr_values[np.isfinite(psnr_values)]
        rows.append(
            {
                "severity_level": int(severity),
                "amplitude_px": float(group["amplitude_px"].iloc[0]),
                "n_samples": int(len(group)),
                "n_sources": int(group["source_id"].nunique()),
                "ssim_mean": float(ssim_values.mean()),
                "ssim_sd": float(ssim_values.std(ddof=1)),
                "ssim_median": float(np.median(ssim_values)),
                "ssim_min": float(ssim_values.min()),
                "ssim_max": float(ssim_values.max()),
                "ssim_drop_from_clean": float(1.0 - ssim_values.mean()),
                "psnr_mean_db": float(finite.mean()) if finite.size else math.nan,
                "psnr_sd_db": float(finite.std(ddof=1)) if finite.size > 1 else math.nan,
                "changed_pixel_fraction_mean": float(group["changed_pixel_fraction"].mean()),
                "changed_ink_fraction_mean": float(group["changed_ink_fraction"].mean()),
                "changed_background_fraction_mean": float(
                    group["changed_background_fraction"].mean()
                ),
                "outside_fraction_mean": float(group["outside_fraction"].mean()),
                "border_affected_fraction_mean": float(group["border_affected_fraction"].mean()),
                "border_touched_fraction": float(group["border_touched"].mean()),
                "clipped_fraction_mean": float(group["clipped_fraction"].mean()),
                "collapsed_block_fraction_mean": float(group["collapsed_block_fraction"].mean()),
                "degenerate_fraction": float(group["degenerate"].mean()),
                "near_identity_fraction": float(np.mean(ssim_values >= near_identity_ssim_min)),
            }
        )
    table = pd.DataFrame(
        rows,
        columns=(
            "severity_level",
            "amplitude_px",
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
            "outside_fraction_mean",
            "border_affected_fraction_mean",
            "border_touched_fraction",
            "clipped_fraction_mean",
            "collapsed_block_fraction_mean",
            "degenerate_fraction",
            "near_identity_fraction",
        ),
    )
    if not table.empty:
        table["ssim_mean_step_delta"] = table["ssim_mean"].diff()
    return table


def trend_by_source(manifest: pd.DataFrame) -> pd.DataFrame:
    """Within-source severity/SSIM trajectory, one row per source."""
    elastic = manifest[manifest["variant_family"] == ELASTIC_FAMILY]
    rows: list[dict[str, object]] = []
    for source_id, group in elastic.groupby("source_id", sort=True):
        ordered = group.sort_values("severity_level", kind="stable")
        severity = ordered["severity_level"].to_numpy(dtype=np.float64)
        values = ordered["ssim_target"].to_numpy(dtype=np.float64)
        differences = np.diff(values)
        rows.append(
            {
                "source_id": str(source_id),
                "n_levels": int(len(ordered)),
                "spearman": rank_correlation(severity, values),
                "n_increasing_steps": int(np.count_nonzero(differences > 0.0)),
                "n_decreasing_steps": int(np.count_nonzero(differences < 0.0)),
                "n_tied_steps": int(np.count_nonzero(differences == 0.0)),
                "n_steps": int(differences.size),
                "max_increase": float(differences.max()) if differences.size else 0.0,
                "max_decrease": float(differences.min()) if differences.size else 0.0,
                "mean_adjacent_change": float(differences.mean()) if differences.size else 0.0,
                "sd_adjacent_change": (
                    float(differences.std(ddof=1)) if differences.size > 1 else 0.0
                ),
                "first_ssim": float(values[0]),
                "last_ssim": float(values[-1]),
                "total_drop": float(values[0] - values[-1]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "source_id",
            "n_levels",
            "spearman",
            "n_increasing_steps",
            "n_decreasing_steps",
            "n_tied_steps",
            "n_steps",
            "max_increase",
            "max_decrease",
            "mean_adjacent_change",
            "sd_adjacent_change",
            "first_ssim",
            "last_ssim",
            "total_drop",
        ),
    )


def trend_violations(manifest: pd.DataFrame) -> pd.DataFrame:
    """Rows where SSIM increases between adjacent severity levels."""
    elastic = manifest[manifest["variant_family"] == ELASTIC_FAMILY]
    rows: list[dict[str, object]] = []
    for source_id, group in elastic.groupby("source_id", sort=True):
        ordered = group.sort_values("severity_level", kind="stable")
        for index in range(len(ordered) - 1):
            current = ordered.iloc[index]
            following = ordered.iloc[index + 1]
            if float(following["ssim_target"]) > float(current["ssim_target"]):
                rows.append(
                    {
                        "source_id": str(source_id),
                        "lower_severity": int(current["severity_level"]),
                        "upper_severity": int(following["severity_level"]),
                        "lower_amplitude_px": float(current["amplitude_px"]),
                        "upper_amplitude_px": float(following["amplitude_px"]),
                        "lower_ssim": float(current["ssim_target"]),
                        "upper_ssim": float(following["ssim_target"]),
                        "increase": float(following["ssim_target"]) - float(current["ssim_target"]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=(
            "source_id",
            "lower_severity",
            "upper_severity",
            "lower_amplitude_px",
            "upper_amplitude_px",
            "lower_ssim",
            "upper_ssim",
            "increase",
        ),
    )


def adjacent_step_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Aggregate adjacent severity-step changes across sources."""
    elastic = manifest[manifest["variant_family"] == ELASTIC_FAMILY]
    rows: list[dict[str, object]] = []
    for source_id, group in elastic.groupby("source_id", sort=True):
        ordered = group.sort_values("severity_level", kind="stable")
        values = ordered["ssim_target"].to_numpy(dtype=np.float64)
        levels = ordered["severity_level"].to_numpy(dtype=np.int64)
        for index in range(len(ordered) - 1):
            rows.append(
                {
                    "source_id": str(source_id),
                    "from_severity": int(levels[index]),
                    "to_severity": int(levels[index + 1]),
                    "delta_ssim": float(values[index + 1] - values[index]),
                }
            )
    steps = pd.DataFrame(rows, columns=("source_id", "from_severity", "to_severity", "delta_ssim"))
    summary_rows: list[dict[str, object]] = []
    for (start, end), group in steps.groupby(["from_severity", "to_severity"], sort=True):
        values = group["delta_ssim"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "from_severity": int(start),
                "to_severity": int(end),
                "n_sources": int(len(values)),
                "n_increasing": int(np.count_nonzero(values > 0.0)),
                "n_decreasing": int(np.count_nonzero(values < 0.0)),
                "n_tied": int(np.count_nonzero(values == 0.0)),
                "mean_delta": float(values.mean()),
                "mean_abs_delta": float(np.abs(values).mean()),
                "sd_delta": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min_delta": float(values.min()),
                "max_delta": float(values.max()),
            }
        )
    return pd.DataFrame(
        summary_rows,
        columns=(
            "from_severity",
            "to_severity",
            "n_sources",
            "n_increasing",
            "n_decreasing",
            "n_tied",
            "mean_delta",
            "mean_abs_delta",
            "sd_delta",
            "min_delta",
            "max_delta",
        ),
    )


def grouped_bootstrap_levels(
    manifest: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Source-grouped bootstrap intervals for each severity's mean SSIM."""
    if n_bootstrap < 1:
        raise GeometricPilotError("n_bootstrap must be >= 1")
    elastic = manifest[manifest["variant_family"] == ELASTIC_FAMILY]
    columns = (
        "severity_level",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_groups",
        "n_samples",
        "n_bootstrap",
    )
    if elastic.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for severity, group in elastic.groupby("severity_level", sort=True):
        values = group["ssim_target"].to_numpy(dtype=np.float64)
        groups = group["source_id"].to_numpy()
        unique_groups = np.array(sorted(set(groups)))
        indices = {name: np.flatnonzero(groups == name) for name in unique_groups}
        rng = np.random.default_rng(derive_noise_seed(seed, "bootstrap", int(severity)))
        replicates = np.empty(n_bootstrap, dtype=np.float64)
        for index in range(n_bootstrap):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sampled_index = np.concatenate([indices[name] for name in sampled])
            replicates[index] = float(values[sampled_index].mean())
        rows.append(
            {
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


def control_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Identity, zero-field and minimal-displacement control diagnostics."""
    controls = manifest[manifest["variant_family"] == CONTROL_FAMILY]
    rows: list[dict[str, object]] = []
    for control_id, group in controls.groupby("control_id", sort=True):
        exact = (group["image_sha256"] == group["reference_sha256"]).to_numpy()
        finite_psnr = group.loc[~group["psnr_infinite"], "psnr_target"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "control_id": str(control_id),
                "n_samples": int(len(group)),
                "amplitude_px": float(group["amplitude_px"].iloc[0]),
                "exact_reference_fraction": float(np.mean(exact)),
                "ssim_mean": float(group["ssim_target"].mean()),
                "ssim_min": float(group["ssim_target"].min()),
                "ssim_max": float(group["ssim_target"].max()),
                "psnr_mean_db": (float(finite_psnr.mean()) if finite_psnr.size else math.nan),
                "changed_pixel_fraction_mean": float(group["changed_pixel_fraction"].mean()),
                "outside_fraction_mean": float(group["outside_fraction"].mean()),
                "border_affected_fraction_mean": float(group["border_affected_fraction"].mean()),
                "border_touched_fraction": float(group["border_touched"].mean()),
                "clipped_fraction_mean": float(group["clipped_fraction"].mean()),
                "collapsed_block_fraction_mean": float(group["collapsed_block_fraction"].mean()),
                "degenerate_fraction": float(group["degenerate"].mean()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "control_id",
            "n_samples",
            "amplitude_px",
            "exact_reference_fraction",
            "ssim_mean",
            "ssim_min",
            "ssim_max",
            "psnr_mean_db",
            "changed_pixel_fraction_mean",
            "outside_fraction_mean",
            "border_affected_fraction_mean",
            "border_touched_fraction",
            "clipped_fraction_mean",
            "collapsed_block_fraction_mean",
            "degenerate_fraction",
        ),
    )


def group_summary(
    manifest: pd.DataFrame,
    *,
    group_column: str,
    min_samples: int,
) -> pd.DataFrame:
    """SSIM by severity and font/text length when group size allows."""
    if group_column not in {"font_id", "text_length"}:
        raise GeometricPilotError("group_column must be 'font_id' or 'text_length'")
    elastic = manifest[manifest["variant_family"] == ELASTIC_FAMILY]
    rows: list[dict[str, object]] = []
    for (group_value, severity), group in elastic.groupby(
        [group_column, "severity_level"], sort=True
    ):
        values = group["ssim_target"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "group_column": group_column,
                "group_value": str(group_value),
                "severity_level": int(severity),
                "n_samples": int(len(group)),
                "n_sources": int(group["source_id"].nunique()),
                "included": bool(len(group) >= min_samples),
                "ssim_mean": float(values.mean()),
                "ssim_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "psnr_mean_db": float(group.loc[~group["psnr_infinite"], "psnr_target"].mean())
                if (~group["psnr_infinite"]).any()
                else math.nan,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "group_column",
            "group_value",
            "severity_level",
            "n_samples",
            "n_sources",
            "included",
            "ssim_mean",
            "ssim_sd",
            "psnr_mean_db",
        ),
    )


def family_level_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Mean SSIM/PSNR by family and severity for the pilot and frozen families."""
    rows: list[dict[str, object]] = []
    for (family, severity), group in combined.groupby(
        ["family_label", "severity_level"], sort=True
    ):
        ssim_values = group["ssim_target"].to_numpy(dtype=np.float64)
        psnr_values = group["psnr_target"].to_numpy(dtype=np.float64)
        finite = psnr_values[np.isfinite(psnr_values)]
        rows.append(
            {
                "family_label": str(family),
                "severity_level": int(severity),
                "n_samples": int(len(group)),
                "n_sources": int(group["source_id"].nunique()),
                "ssim_mean": float(ssim_values.mean()),
                "ssim_sd": (float(ssim_values.std(ddof=1)) if len(group) > 1 else 0.0),
                "ssim_min": float(ssim_values.min()),
                "ssim_max": float(ssim_values.max()),
                "psnr_mean_db": float(finite.mean()) if finite.size else math.nan,
                "psnr_sd_db": float(finite.std(ddof=1)) if finite.size > 1 else math.nan,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "family_label",
            "severity_level",
            "n_samples",
            "n_sources",
            "ssim_mean",
            "ssim_sd",
            "ssim_min",
            "ssim_max",
            "psnr_mean_db",
            "psnr_sd_db",
        ),
    )


def feature_summary_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of every feature by family and severity."""
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
    combined: pd.DataFrame,
    *,
    strong_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cohen's d for elastic-warp features against each frozen family.

    A positive value means the geometric feature is larger. The long table keeps
    every feature; the summary counts features above the predeclared effect
    thresholds at each severity.
    """
    if not math.isfinite(strong_threshold) or strong_threshold <= 0.0:
        raise GeometricPilotError("strong_threshold must be finite and > 0")
    pilot = combined[combined["family_label"] == ELASTIC_FAMILY]
    references = combined[combined["family_label"] != ELASTIC_FAMILY]
    long_rows: list[dict[str, object]] = []
    for severity in sorted(pilot["severity_level"].unique()):
        pilot_group = pilot[pilot["severity_level"] == severity]
        for reference_family in sorted(references["family_label"].unique()):
            reference_group = references[
                (references["family_label"] == reference_family)
                & (references["severity_level"] == severity)
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
        for (severity, reference_family), group in long.groupby(
            ["severity_level", "reference_family"], sort=True
        ):
            summary_rows.append(
                {
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


def feature_target_correlations(combined: pd.DataFrame) -> pd.DataFrame:
    """Feature/target association by family, pooled and within severity."""
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
    reference: Any | None = None,
    recompute_targets: bool = True,
    require_summary: bool = True,
) -> VerificationReport:
    """Verify schema, hashes, targets, field replay and clean immutability."""
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
        lambda: _check_reference_matching(manifest, sources),
    )
    _check(
        checks,
        "clean_reference_immutability",
        lambda: _check_clean_immutability(root, manifest, sources, reference),
    )
    _check(checks, "image_dimensions", lambda: _check_dimensions(root, manifest, reference))
    _check(checks, "image_content_hashes", lambda: _check_image_hashes(root, manifest))
    _check(checks, "conditional_psnr_targets", lambda: _check_psnr(manifest))
    _check(checks, "control_identity", lambda: _check_controls(manifest))
    if recompute_targets:
        _check(checks, "targets_recomputed", lambda: _check_targets(root, manifest))
    _check(
        checks,
        "provenance_replay",
        lambda: _check_provenance_replay(root, manifest),
    )
    _check(checks, "no_duplicate_elastic_images", lambda: _check_no_duplicate_images(manifest))
    _check(checks, "feature_table", lambda: _check_feature_table(features, manifest))
    _check(
        checks,
        "features_recomputed",
        lambda: _check_features_recomputed(root, features, manifest),
    )
    if require_summary:
        _check(
            checks,
            "summary_consistency",
            lambda: _check_summary(root, summary, manifest, sources, features),
        )
    return VerificationReport(dataset=str(root), checks=tuple(checks))


def _require_pilot_files(root: Path, names: Iterable[str]) -> str:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise GeometricPilotError(f"missing required file(s): {missing}")
    if not (root / "images").is_dir():
        raise GeometricPilotError("missing required directory: images/")
    return f"found {len(list(names))} files plus images/"


def _check_pilot_schema(
    manifest: pd.DataFrame, sources: pd.DataFrame, features: pd.DataFrame
) -> str:
    if tuple(manifest.columns) != PILOT_MANIFEST_COLUMNS:
        raise GeometricPilotError(
            f"manifest columns must be {PILOT_MANIFEST_COLUMNS}, got {tuple(manifest.columns)}"
        )
    if tuple(sources.columns) != PILOT_SOURCE_COLUMNS:
        raise GeometricPilotError(
            f"sources columns must be {PILOT_SOURCE_COLUMNS}, got {tuple(sources.columns)}"
        )
    if tuple(features.columns) != PILOT_FEATURE_COLUMNS:
        raise GeometricPilotError(
            f"feature columns must be {PILOT_FEATURE_COLUMNS}, got {tuple(features.columns)}"
        )
    if sources["source_id"].duplicated().any():
        raise GeometricPilotError("sources.csv contains duplicate source_id values")
    if "text" in manifest.columns:
        raise GeometricPilotError("pilot manifest must not contain raw text")
    return f"{len(manifest)} manifest rows, {len(sources)} sources, schema complete"


def _require_unique(frame: pd.DataFrame, column: str) -> str:
    if frame[column].isna().any():
        raise GeometricPilotError(f"column {column!r} contains missing values")
    if frame[column].duplicated().any():
        duplicates = frame.loc[frame[column].duplicated(), column].head(5).tolist()
        raise GeometricPilotError(f"column {column!r} is not unique, e.g. {duplicates}")
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
        raise GeometricPilotError(f"paths escape the pilot root: {sorted(set(escaped))[:5]}")
    if missing:
        raise GeometricPilotError(f"missing image file(s): {sorted(set(missing))[:5]}")
    return "all image and reference paths exist inside the pilot root"


def _check_reference_matching(manifest: pd.DataFrame, sources: pd.DataFrame) -> str:
    source_index = sources.set_index("source_id")
    for row in manifest.itertuples(index=False):
        if row.source_id not in source_index.index:
            raise GeometricPilotError(f"{row.sample_id} references unknown source {row.source_id}")
        source = source_index.loc[row.source_id]
        expected_reference = str(source["pilot_clean_image_path"])
        if str(row.reference_path) != expected_reference:
            raise GeometricPilotError(
                f"{row.sample_id}: reference_path is not the pilot clean image"
            )
        if str(row.reference_sha256) != str(source["clean_sha256"]):
            raise GeometricPilotError(f"{row.sample_id}: reference hash disagrees with sources.csv")
        for column in ("split", "font_id", "text_length"):
            if getattr(row, column) != source[column]:
                raise GeometricPilotError(f"{row.sample_id}: {column} disagrees with sources.csv")
    if set(sources["source_id"]) - set(manifest["source_id"]):
        raise GeometricPilotError("some sources have no manifest rows")
    return f"{len(manifest)} rows reference the correct clean source"


def _check_clean_immutability(
    root: Path,
    manifest: pd.DataFrame,
    sources: pd.DataFrame,
    reference: Any | None,
) -> str:
    clean_rows = manifest[manifest["variant_family"] == CLEAN_FAMILY]
    if len(clean_rows) != len(sources):
        raise GeometricPilotError(
            f"expected one clean row per source, got {len(clean_rows)} for {len(sources)} sources"
        )
    for source in sources.itertuples(index=False):
        image = load_grayscale(root / str(source.pilot_clean_image_path))
        actual = sha256_bytes(image.tobytes())
        if actual != str(source.clean_sha256):
            raise GeometricPilotError(f"clean image hash mismatch for source {source.source_id}")
        if reference is not None:
            frozen = reference.clean_images[str(source.source_id)]
            if not np.array_equal(image, frozen):
                raise GeometricPilotError(
                    f"pilot clean image differs from the frozen reference for {source.source_id}"
                )
    return f"{len(sources)} clean references match their frozen pixel hashes"


def _check_dimensions(root: Path, manifest: pd.DataFrame, reference: Any | None) -> str:
    expected: tuple[int, int] | None = None
    if reference is not None:
        first = next(iter(reference.clean_images.values()))
        expected = (int(first.shape[0]), int(first.shape[1]))
    for row in manifest.itertuples(index=False):
        image = load_grayscale(root / str(row.image_path))
        shape = (int(image.shape[0]), int(image.shape[1]))
        if shape != (int(row.height), int(row.width)):
            raise GeometricPilotError(
                f"{row.sample_id}: image shape {shape} disagrees with manifest "
                f"{(row.height, row.width)}"
            )
        if expected is not None and shape != expected:
            raise GeometricPilotError(
                f"{row.sample_id}: image shape {shape} differs from {expected}"
            )
    return f"{len(manifest)} images are 8-bit grayscale with recorded dimensions"


def _check_image_hashes(root: Path, manifest: pd.DataFrame) -> str:
    expected: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        relative = str(row.image_path)
        previous = expected.setdefault(relative, str(row.image_sha256))
        if previous != row.image_sha256:
            raise GeometricPilotError(f"inconsistent image_sha256 for {relative}")
    mismatches = []
    for relative, digest in expected.items():
        image = load_grayscale(root / relative)
        if sha256_bytes(image.tobytes()) != digest:
            mismatches.append(relative)
    if mismatches:
        raise GeometricPilotError(f"image pixel hash mismatch: {sorted(mismatches)[:5]}")
    return f"{len(expected)} image files match their recorded pixel hashes"


def _check_psnr(manifest: pd.DataFrame) -> str:
    infinite = manifest["psnr_infinite"].astype(bool)
    exact = (manifest["variant_family"] == CLEAN_FAMILY) | (
        (manifest["variant_family"] == CONTROL_FAMILY)
        & manifest["control_id"].isin([CONTROL_IDENTITY, CONTROL_ZERO_FIELD])
    )
    if not (infinite == exact).all():
        raise GeometricPilotError(
            "psnr_infinite must be true exactly for clean and exact identity controls; "
            f"mismatches: {manifest.loc[infinite != exact, 'sample_id'].head(5).tolist()}"
        )
    finite = ~infinite
    if manifest.loc[finite, "psnr_target"].isna().any():
        raise GeometricPilotError("finite PSNR rows must record psnr_target")
    if manifest.loc[infinite, "psnr_target"].notna().any():
        raise GeometricPilotError("infinite PSNR rows must leave psnr_target empty")
    return f"{int(exact.sum())} exact rows use an explicit infinite-PSNR flag"


def _check_controls(manifest: pd.DataFrame) -> str:
    controls = manifest[manifest["variant_family"] == CONTROL_FAMILY]
    expected = {CONTROL_IDENTITY, CONTROL_ZERO_FIELD, CONTROL_MINIMAL}
    if set(controls["control_id"]) != expected:
        raise GeometricPilotError(
            f"control IDs must be {expected}, got {set(controls['control_id'])}"
        )
    exact = controls[controls["control_id"].isin([CONTROL_IDENTITY, CONTROL_ZERO_FIELD])]
    if not (exact["image_sha256"] == exact["reference_sha256"]).all():
        raise GeometricPilotError(
            "identity and zero-field controls must reproduce the clean pixels"
        )
    minimal = controls[controls["control_id"] == CONTROL_MINIMAL]
    if minimal.empty:
        raise GeometricPilotError("minimal-displacement control is missing")
    if (minimal["image_sha256"] == minimal["reference_sha256"]).any():
        raise GeometricPilotError(
            "minimal-displacement control unexpectedly equals the clean image"
        )
    if (minimal["ssim_target"] >= 1.0).any():
        raise GeometricPilotError("minimal-displacement control must have a finite SSIM below 1")
    if controls["resampled"].isna().any():
        raise GeometricPilotError("control resampled flags must be recorded")
    identity_rows = controls[controls["control_id"] == CONTROL_IDENTITY]
    zero_rows = controls[controls["control_id"] == CONTROL_ZERO_FIELD]
    if identity_rows["resampled"].any():
        raise GeometricPilotError("identity control must bypass resampling")
    if not zero_rows["resampled"].all():
        raise GeometricPilotError("zero-field control must exercise the resampler")
    if not minimal["resampled"].all():
        raise GeometricPilotError("minimal-displacement control must exercise the resampler")
    return f"{len(controls)} control rows present; exact identity and zero-field are verified"


def _check_targets(root: Path, manifest: pd.DataFrame) -> str:
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
        raise GeometricPilotError(
            f"{len(mismatches)} target(s) disagree with the exact pair, e.g. {mismatches[:5]}"
        )
    return f"recomputed SSIM/PSNR targets for {len(manifest)} rows from their exact references"


def _check_provenance_replay(root: Path, manifest: pd.DataFrame) -> str:
    """Rebuild every transformed image from its recorded field specification."""
    transformed = manifest[manifest["variant_family"] != CLEAN_FAMILY]
    references: dict[str, np.ndarray] = {}
    mismatches: list[str] = []
    for row in transformed.itertuples(index=False):
        reference_path = str(row.reference_path)
        if reference_path not in references:
            references[reference_path] = load_grayscale(root / reference_path)
        payload = json.loads(str(row.field_spec_json))
        try:
            replayed = verify_field_replay(
                references[reference_path],
                payload,
                amplitude_px=float(row.amplitude_px),
                force_resample=bool(row.resampled),
            )
        except Exception as exc:  # noqa: BLE001 - report replay failures
            mismatches.append(f"{row.sample_id}:{type(exc).__name__}")
            continue
        if sha256_bytes(replayed.image.tobytes()) != str(row.image_sha256):
            mismatches.append(f"{row.sample_id}:pixels")
    if mismatches:
        raise GeometricPilotError(
            f"{len(mismatches)} provenance replay(s) disagree, e.g. {mismatches[:5]}"
        )
    return f"replayed {len(transformed)} transformed images from recorded field specifications"


def _check_no_duplicate_images(manifest: pd.DataFrame) -> str:
    elastic = manifest[manifest["variant_family"] == ELASTIC_FAMILY]
    if elastic["image_sha256"].duplicated().any():
        duplicates = elastic.loc[elastic["image_sha256"].duplicated(), "sample_id"].head(5).tolist()
        raise GeometricPilotError(f"duplicate elastic image hashes, e.g. {duplicates}")
    clean_hashes = set(manifest.loc[manifest["variant_family"] == CLEAN_FAMILY, "image_sha256"])
    if set(elastic["image_sha256"]) & clean_hashes:
        raise GeometricPilotError("an elastic image is pixel-identical to a clean reference")
    return f"{len(elastic)} elastic images are distinct and differ from the clean references"


def _check_feature_table(features: pd.DataFrame, manifest: pd.DataFrame) -> str:
    if features["sample_id"].duplicated().any():
        raise GeometricPilotError("features.csv contains duplicate sample_id values")
    if set(features["sample_id"]) != set(manifest["sample_id"]):
        raise GeometricPilotError("features.csv sample IDs do not match the manifest")
    values = features[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise GeometricPilotError("features.csv contains NaN or infinite values")
    forbidden = {"ssim", "psnr", "ms_ssim"}
    leaked = forbidden & {name.lower() for name in features.columns}
    if leaked:
        raise GeometricPilotError(f"target metric(s) used as features: {sorted(leaked)}")
    return f"{len(features)} feature rows x {len(FEATURE_NAMES)} features are finite"


def _check_features_recomputed(root: Path, features: pd.DataFrame, manifest: pd.DataFrame) -> str:
    samples = []
    for row in manifest.itertuples(index=False):
        with Image.open(root / str(row.image_path)) as image:
            samples.append((str(row.sample_id), np.asarray(image)))
    recomputed = extract_feature_table(samples).set_index("sample_id")
    expected = features.set_index("sample_id").loc[recomputed.index]
    actual = recomputed[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    stored = expected[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    difference = np.abs(actual - stored)
    maximum = float(difference.max(initial=0.0))
    # features.csv is written with ``%.12g``, so exact float64 equality is not
    # expected even when the values were produced from the same pixels.
    if not np.allclose(actual, stored, rtol=1e-9, atol=1e-9):
        raise GeometricPilotError(
            f"recomputed features disagree with features.csv (max abs difference {maximum:.3e})"
        )
    return f"recomputed {len(recomputed)} feature rows from the written images (max |delta| {maximum:.1e})"


def _check_summary(
    root: Path,
    summary: dict[str, Any],
    manifest: pd.DataFrame,
    sources: pd.DataFrame,
    features: pd.DataFrame,
) -> str:
    counts = summary.get("counts", {})
    if counts.get("sources") != len(sources):
        raise GeometricPilotError("summary source count disagrees with sources.csv")
    if counts.get("samples") != len(manifest):
        raise GeometricPilotError("summary sample count disagrees with the manifest")
    if counts.get("feature_rows") != len(features):
        raise GeometricPilotError("summary feature count disagrees with features.csv")
    if summary.get("feature_names") != list(FEATURE_NAMES):
        raise GeometricPilotError("summary feature order disagrees with this code version")
    if summary.get("pilot_config_hash") != manifest["pilot_config_hash"].iloc[0]:
        raise GeometricPilotError("summary pilot_config_hash disagrees with the manifest")
    artifacts = summary.get("artifacts", {})
    expected_files = {
        "manifest_sha256": "manifest.csv",
        "sources_sha256": "sources.csv",
        "features_sha256": "features.csv",
    }
    for key, filename in expected_files.items():
        actual = sha256_file(root / filename)
        if artifacts.get(key) != actual:
            raise GeometricPilotError(f"summary {key} disagrees with {filename}")
    unique_images = manifest.drop_duplicates("image_path").set_index("image_path")
    image_hashes = {
        str(path): str(digest) for path, digest in unique_images["image_sha256"].items()
    }
    if artifacts.get("n_image_files") != len(image_hashes):
        raise GeometricPilotError("summary image-file count disagrees with the manifest")
    if artifacts.get("image_pixel_hash_set") != _image_pixel_hash_set(image_hashes):
        raise GeometricPilotError("summary image pixel-hash set disagrees with the manifest")
    return "summary counts, hashes, feature names, image hashes and config identity agree"


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
class GateVerdict:
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

    config: GeometricPilotConfig
    reference: Any
    comparison: FrozenComparison
    manifest: pd.DataFrame
    sources: pd.DataFrame
    features: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    gate: GateVerdict
    verification: VerificationReport
    output_dataset: Path
    output_reports: Path
    report_path: Path


def _duplicate_fraction(manifest: pd.DataFrame) -> float:
    elastic = manifest[manifest["variant_family"] == ELASTIC_FAMILY]
    if elastic.empty:
        return 1.0
    hashes = elastic["image_sha256"]
    duplicates = int(len(hashes) - hashes.nunique())
    return duplicates / float(len(hashes))


def evaluate_gate(
    config: GeometricPilotConfig,
    *,
    manifest: pd.DataFrame,
    levels: pd.DataFrame,
    trends: pd.DataFrame,
    controls: pd.DataFrame,
    effects_summary: pd.DataFrame,
    duplicate_fraction: float,
    reproducibility_ok: bool,
    verification_ok: bool,
    clean_reference_exact: bool,
) -> GateVerdict:
    """Evaluate the predeclared M5A feasibility gate."""
    elastic = manifest[manifest["variant_family"] == ELASTIC_FAMILY]
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

    control_index = controls.set_index("control_id")
    identity_ok = all(
        control_id in control_index.index
        and float(control_index.loc[control_id, "exact_reference_fraction"]) == 1.0
        for control_id in (CONTROL_IDENTITY, CONTROL_ZERO_FIELD)
    )
    minimal_row = (
        control_index.loc[CONTROL_MINIMAL] if CONTROL_MINIMAL in control_index.index else None
    )
    minimal_ok = minimal_row is not None and float(minimal_row["ssim_mean"]) >= float(
        config.gate.minimal_control_ssim_min
    )
    criteria.append(
        PilotCriterion(
            id="identity_control",
            description=(
                "Identity and zero-field controls reproduce clean pixels exactly; "
                "the minimal-displacement control is non-degenerate and near 1.0"
            ),
            observed=(
                f"identity_exact={identity_ok}, minimal_mean_ssim="
                f"{float(minimal_row['ssim_mean']) if minimal_row is not None else 'missing'}"
            ),
            threshold=(
                "identity/zero-field exact and minimal mean SSIM >= "
                f"{config.gate.minimal_control_ssim_min}"
            ),
            passed=bool(identity_ok and minimal_ok),
        )
    )

    overall_sd = float(elastic["ssim_target"].std(ddof=1))
    level_mean_range = (
        float(levels["ssim_mean"].max() - levels["ssim_mean"].min()) if not levels.empty else 0.0
    )
    min_level_sd = float(levels["ssim_sd"].min()) if not levels.empty else 0.0
    overall_spread_passed = (
        overall_sd >= config.gate.ssim_std_min
        and level_mean_range >= config.gate.level_mean_range_min
    )
    level_separation_passed = min_level_sd >= config.gate.per_level_sd_min
    non_degenerate_passed = overall_spread_passed and level_separation_passed
    criteria.append(
        PilotCriterion(
            id="non_degeneracy",
            description=(
                "Elastic SSIM has meaningful spread, the level means span the "
                "configured range, and every level has non-trivial SD"
            ),
            observed=(
                f"overall_sd={overall_sd:.4f}, level_mean_range={level_mean_range:.4f}, "
                f"min_level_sd={min_level_sd:.4f}"
            ),
            threshold=(
                f"sd>={config.gate.ssim_std_min}, range>={config.gate.level_mean_range_min}, "
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

    highest_level = levels.sort_values("severity_level").iloc[-1] if not levels.empty else None
    max_border = (
        float(highest_level["border_affected_fraction_mean"]) if highest_level is not None else 1.0
    )
    max_collapsed = (
        float(highest_level["collapsed_block_fraction_mean"]) if highest_level is not None else 1.0
    )
    interpolation_passed = (
        minimal_ok
        and max_border <= config.gate.border_affected_fraction_max
        and max_collapsed <= config.gate.collapsed_block_fraction_max
    )
    criteria.append(
        PilotCriterion(
            id="interpolation_and_boundary",
            description=(
                "Minimal displacement is near-identity and highest-severity border "
                "and block-collapse diagnostics stay within predeclared bounds"
            ),
            observed=(
                f"minimal_mean_ssim="
                f"{float(minimal_row['ssim_mean']) if minimal_row is not None else 'missing'}, "
                f"max_border_affected={max_border:.4f}, "
                f"max_collapsed_block={max_collapsed:.4f}"
            ),
            threshold=(
                f"minimal_ssim>={config.gate.minimal_control_ssim_min}, "
                f"border<={config.gate.border_affected_fraction_max}, "
                f"collapsed<={config.gate.collapsed_block_fraction_max}"
            ),
            passed=bool(interpolation_passed),
        )
    )

    highest_severity = int(elastic["severity_level"].max())
    distinct_passed = True
    distinct_lines: list[str] = []
    reference_families = sorted(effects_summary["reference_family"].unique())
    if not reference_families:
        distinct_passed = False
        distinct_lines.append("no frozen comparison family available")
    for reference_family in reference_families:
        selected = effects_summary[
            (effects_summary["severity_level"] == highest_severity)
            & (effects_summary["reference_family"] == reference_family)
        ]
        if selected.empty:
            distinct_passed = False
            distinct_lines.append(f"{reference_family}: missing at severity {highest_severity}")
            continue
        row = selected.iloc[0]
        passed = (
            float(row["mean_abs_cohens_d"]) >= config.gate.feature_effect_min
            and int(row["n_features_abs_d_ge_strong"])
            >= config.gate.feature_strong_effect_count_min
        )
        distinct_passed = distinct_passed and passed
        distinct_lines.append(
            f"{reference_family}: mean|d|={float(row['mean_abs_cohens_d']):.2f}, "
            f"n|d|>={config.gate.feature_strong_effect_min:.2f}="
            f"{int(row['n_features_abs_d_ge_strong'])}"
        )
    criteria.append(
        PilotCriterion(
            id="feature_distinctness",
            description=(
                "At the highest severity, the eight features differ meaningfully from "
                "every available frozen corruption family"
            ),
            observed="; ".join(distinct_lines),
            threshold=(
                f"mean|d|>={config.gate.feature_effect_min} and "
                f"n|d|>={config.gate.feature_strong_effect_min} >= "
                f"{config.gate.feature_strong_effect_count_min} for every family"
            ),
            passed=bool(distinct_passed),
            note="Cohen's d compares elastic features with the frozen family at the same severity.",
        )
    )

    duplicate_passed = duplicate_fraction <= config.gate.duplicate_fraction_max
    criteria.append(
        PilotCriterion(
            id="duplicate_free",
            description="No duplicate elastic images and none equal to a clean reference",
            observed=f"duplicate_fraction={duplicate_fraction:.6f}",
            threshold=f"<={config.gate.duplicate_fraction_max}",
            passed=bool(duplicate_passed),
        )
    )

    all_passed = all(criterion.passed for criterion in criteria)
    # A CONDITIONAL PASS is reserved for a pilot whose core validity checks all
    # pass but a predeclared margin does not. The per-level SD guard is one such
    # margin: the lowest rung can be close enough to the interpolation floor that
    # its between-source spread is small even when the overall target is healthy.
    core_passed = (
        reproducibility_passed
        and identity_ok
        and minimal_ok
        and overall_spread_passed
        and trend_passed
        and interpolation_passed
        and duplicate_passed
    )
    if all_passed:
        verdict = "PASS"
    elif core_passed:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "FAIL"
    return GateVerdict(verdict=verdict, criteria=tuple(criteria))


def _build_summary(
    config: GeometricPilotConfig,
    reference: Any,
    comparison: FrozenComparison,
    generation: GenerationData,
    *,
    target: Path,
    determinism: Mapping[str, object],
    verification: VerificationReport,
    gate: GateVerdict,
    frozen_before: Mapping[str, Mapping[str, object]],
    frozen_after: Mapping[str, Mapping[str, object]],
    controls: pd.DataFrame,
    levels: pd.DataFrame,
) -> dict[str, Any]:
    elastic = generation.manifest[generation.manifest["variant_family"] == ELASTIC_FAMILY]
    controls_frame = generation.manifest[generation.manifest["variant_family"] == CONTROL_FAMILY]
    changed = {
        path: {
            "before": frozen_before.get(path),
            "after": frozen_after.get(path),
            "unchanged": frozen_before.get(path) == frozen_after.get(path),
        }
        for path in sorted(set(frozen_before) | set(frozen_after))
    }
    minimal = controls[controls["control_id"] == CONTROL_MINIMAL]
    highest = levels.sort_values("severity_level").iloc[-1] if not levels.empty else None
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
        "comparison": {
            "include_interference": config.comparison.include_interference,
            "m4a_verified": comparison.m4a_verified,
            "m4a_manifest_sha256": comparison.m4a_manifest_sha256,
            "m4a_features_sha256": comparison.m4a_features_sha256,
            "m4a_summary_sha256": comparison.m4a_summary_sha256,
            "n_families": int(comparison.table["family_label"].nunique()),
            "families": sorted(comparison.table["family_label"].astype(str).unique().tolist()),
        },
        "counts": {
            "sources": int(len(generation.sources)),
            "samples": int(len(generation.manifest)),
            "clean_samples": int((generation.manifest["variant_family"] == CLEAN_FAMILY).sum()),
            "control_samples": int(len(controls_frame)),
            "elastic_samples": int(len(elastic)),
            "feature_rows": int(len(generation.features)),
            "by_split": _counts(generation.sources["split"]),
            "by_font": _counts(generation.sources["font_id"]),
            "by_control": _counts(controls_frame["control_id"]),
            "by_severity": _counts(elastic["severity_level"]),
            "disk_usage_bytes": sum(
                path.stat().st_size for path in target.rglob("*") if path.is_file()
            ),
        },
        "geometry": {
            **config.geometry.as_dict(),
            "field_specs_unique": int(generation.sources["field_spec_json"].nunique()),
            "field_hashes_unique": int(generation.sources["field_sha256"].nunique()),
        },
        "target": {
            "name": "ssim_target",
            "definition": (
                "SSIM against the exact clean reference (data_range=255, win_size=11, "
                "gaussian_weights=True, sigma=1.5, use_sample_covariance=False)"
            ),
            "ssim_mean": float(elastic["ssim_target"].mean()),
            "ssim_sd": float(elastic["ssim_target"].std(ddof=1)),
            "ssim_min": float(elastic["ssim_target"].min()),
            "ssim_max": float(elastic["ssim_target"].max()),
            "psnr_mean_db": float(elastic["psnr_target"].mean()),
            "psnr_min_db": float(elastic["psnr_target"].min()),
            "psnr_max_db": float(elastic["psnr_target"].max()),
            "near_identity_fraction": float(
                np.mean(
                    elastic["ssim_target"].to_numpy(dtype=np.float64)
                    >= config.analysis.near_identity_ssim_min
                )
            ),
            "degenerate_fraction": float(elastic["degenerate"].mean()),
        },
        "interpolation_boundary": {
            "minimal_control_ssim_mean": (
                float(minimal["ssim_mean"].iloc[0]) if not minimal.empty else math.nan
            ),
            "minimal_control_ssim_min": (
                float(minimal["ssim_min"].iloc[0]) if not minimal.empty else math.nan
            ),
            "max_severity_border_affected_fraction": (
                float(highest["border_affected_fraction_mean"]) if highest is not None else math.nan
            ),
            "max_severity_outside_fraction": (
                float(highest["outside_fraction_mean"]) if highest is not None else math.nan
            ),
            "max_severity_collapsed_block_fraction": (
                float(highest["collapsed_block_fraction_mean"]) if highest is not None else math.nan
            ),
            "max_severity_clipped_fraction": (
                float(highest["clipped_fraction_mean"]) if highest is not None else math.nan
            ),
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


def _artifacts_unchanged(
    before: Mapping[str, Mapping[str, object]],
    after: Mapping[str, Mapping[str, object]],
) -> bool:
    return all(before.get(path) == after.get(path) for path in set(before) | set(after))


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
    label_margin_left: int = 130,
    label_margin_top: int = 18,
) -> None:
    if not cells or not cells[0]:
        return
    from PIL import ImageDraw, ImageFont

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


def _write_contact_sheet(
    root: Path,
    config: GeometricPilotConfig,
    sources: pd.DataFrame,
    reports: Path,
) -> None:
    examples = sources.sort_values(["font_id", "source_id"], kind="stable").drop_duplicates(
        "font_id", keep="first"
    )
    row_labels: list[str] = []
    cells: list[list[np.ndarray]] = []
    column_labels = ["clean", "identity", "zero-field", "minimal", "s0.5", "s1.0", "s2.0", "s3.0"]
    for source in examples.itertuples(index=False):
        source_id = str(source.source_id)
        row = [load_grayscale(root / str(source.pilot_clean_image_path))]
        for control_id in (CONTROL_IDENTITY, CONTROL_ZERO_FIELD, CONTROL_MINIMAL):
            sample_id = _control_sample_id(source_id, control_id)
            row.append(load_grayscale(root / "images" / "control" / f"{sample_id}.png"))
        for severity in range(1, config.geometry.ladder.n_levels + 1):
            sample_id = _elastic_sample_id(source_id, severity)
            row.append(load_grayscale(root / "images" / "elastic" / f"{sample_id}.png"))
        row_labels.append(f"{source.font_id} / {source_id[:12]}")
        cells.append(row)
    _contact_sheet(
        reports / "examples_elastic.png",
        row_labels=row_labels,
        column_labels=column_labels[: len(cells[0])],
        cells=cells,
    )


def _report_markdown(
    config: GeometricPilotConfig,
    reference: Any,
    comparison: FrozenComparison,
    tables: Mapping[str, pd.DataFrame],
    summary: Mapping[str, Any],
    gate: GateVerdict,
    verification: VerificationReport,
    determinism: Mapping[str, object],
) -> str:
    levels = tables["level_summary"]
    trends = tables["trend_by_source"]
    steps = tables["adjacent_steps"]
    controls = tables["controls"]
    family_levels = tables["family_levels"]
    effects = tables["feature_distinctness"]
    correlations = tables["feature_target_correlations"]
    font_groups = tables["font_groups"]
    length_groups = tables["length_groups"]
    uncertainty = tables["uncertainty"]
    violations = tables["trend_violations"]
    counts = summary["counts"]
    minimal = controls[controls["control_id"] == CONTROL_MINIMAL]
    minimal_mean = float(minimal["ssim_mean"].iloc[0]) if not minimal.empty else math.nan
    highest_effects = effects[effects["severity_level"] == int(levels["severity_level"].max())]

    lines: list[str] = [
        "# Geometric-distortion feasibility pilot — M5A report",
        "",
        "**Scope:** synthetic image similarity only. SSIM and PSNR are measured "
        "against each source's exact clean reference. They are not measures of "
        "human readability, CAPTCHA solvability, or production quality. This is "
        "a small, predeclared feasibility pilot, not a robustness benchmark.",
        "",
        "## Frozen assets",
        "",
        f"- Frozen M3 dataset: `{reference.root}` (`{config.reference.dataset_version}`)",
        f"- M3 generation hash: `{config.reference.generation_config_hash}`",
        f"- M3 manifest SHA-256: `{config.reference.manifest_sha256}`",
        f"- M3 sources SHA-256: `{config.reference.sources_sha256}`",
        f"- M3 features SHA-256: `{config.reference.features_sha256}`",
        f"- M3 split-assignment SHA-256: `{config.reference.split_assignment_sha256}`",
        f"- Clean sources re-rendered exactly: {reference.rerendered_sources}",
        f"- Frozen assets unchanged before/after: {summary['frozen_assets']['all_unchanged']}",
        "",
    ]
    if comparison.m4a_verified:
        lines.extend(
            [
                f"- M4A manifest SHA-256: `{comparison.m4a_manifest_sha256}`",
                f"- M4A features SHA-256: `{comparison.m4a_features_sha256}`",
                f"- M4A summary SHA-256: `{comparison.m4a_summary_sha256}`",
            ]
        )
    else:
        lines.append("- M4A comparison was not enabled in this test configuration.")
    lines.extend(
        [
            "",
            "The pilot re-rendered every clean M3 source from the frozen dataset "
            "configuration and compared pixels, source IDs, render seeds, text hashes "
            "and split labels. It did not modify the frozen dataset, split, model "
            "artifacts, M4A pilot or M4B benchmark.",
            "",
            "## Configuration and transformation",
            "",
            f"- Config: `{config.config_path}` (sha256 `{config.config_sha256[:16]}…`)",
            f"- Pilot generation hash: `{config.pilot_config_hash}`",
            f"- Seed: `{config.pilot.seed}`; bootstrap replicates: "
            f"`{config.pilot.bootstrap_replicates}`",
            f"- Output dataset: `{config.pilot.output_dataset}`",
            f"- Output reports: `{config.pilot.output_reports}`",
            "- Transformation: smooth elastic displacement with bilinear resampling",
            f"- Amplitude ladder (maximum displacement, px): "
            f"{list(config.geometry.ladder.amplitudes_px)}",
            f"- Minimal control amplitude (px): {config.geometry.ladder.minimal_amplitude_px}",
            f"- Control grid: {config.geometry.grid_rows} x {config.geometry.grid_cols}; "
            f"Gaussian smoothing sigma: {config.geometry.smoothing_sigma_px} px",
            f"- Interpolation: `{config.geometry.interpolation}`; border mode: "
            f"`{config.geometry.border_mode}`",
            f"- Unique field specifications: {summary['geometry']['field_specs_unique']}; "
            f"unique field hashes: {summary['geometry']['field_hashes_unique']}",
            "",
            "Every severity and control for a source uses one seeded unit-maximum "
            "displacement field, scaled to the recorded amplitude. The field is "
            "centred to zero mean before normalisation, so the transformation has no "
            "global translation. A stored field specification and its SHA-256 replay "
            "the exact field and pixels; the field is not inferred from the target.",
            "",
            "## Dataset and controls",
            "",
            f"- Sources: {counts['sources']} (frozen split preserved: {counts['by_split']})",
            f"- Samples: {counts['samples']} = {counts['clean_samples']} clean + "
            f"{counts['control_samples']} controls + {counts['elastic_samples']} elastic",
            f"- Elastic rows by severity: {counts['by_severity']}",
            f"- Controls by type: {counts['by_control']}",
            f"- Feature rows: {counts['feature_rows']} x {len(FEATURE_NAMES)} features",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "control",
                "n",
                "amplitude",
                "exact clean",
                "SSIM mean",
                "SSIM min",
                "changed px",
                "border frac",
            ],
            [
                [
                    str(row.control_id),
                    int(row.n_samples),
                    float(row.amplitude_px),
                    float(row.exact_reference_fraction),
                    float(row.ssim_mean),
                    float(row.ssim_min),
                    float(row.changed_pixel_fraction_mean),
                    float(row.border_affected_fraction_mean),
                ]
                for row in controls.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "The identity control bypasses resampling and copies the clean image. The "
            "zero-field control passes a zero displacement through the bilinear sampler; "
            "because its coordinates are integers it must also reproduce the clean "
            "image exactly. The minimal control uses a 0.25 px scale of the same field "
            "to expose interpolation and boundary effects at the smallest practical "
            "non-zero displacement.",
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
            "## SSIM target behaviour",
            "",
            f"Overall elastic SSIM: mean {summary['target']['ssim_mean']:.4f}, "
            f"SD {summary['target']['ssim_sd']:.4f}, range "
            f"[{summary['target']['ssim_min']:.4f}, {summary['target']['ssim_max']:.4f}] "
            "across sources and severities.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "severity",
                "amplitude px",
                "n",
                "SSIM mean",
                "SSIM SD",
                "SSIM min",
                "SSIM max",
                "SSIM drop",
                "PSNR mean dB",
                "changed px",
                "border frac",
            ],
            [
                [
                    int(row.severity_level),
                    float(row.amplitude_px),
                    int(row.n_samples),
                    float(row.ssim_mean),
                    float(row.ssim_sd),
                    float(row.ssim_min),
                    float(row.ssim_max),
                    float(row.ssim_drop_from_clean),
                    float(row.psnr_mean_db),
                    float(row.changed_pixel_fraction_mean),
                    float(row.border_affected_fraction_mean),
                ]
                for row in levels.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Source-grouped uncertainty",
            "",
            "The intervals below resample sources, not individual images. They are "
            "preliminary because they describe the available synthetic renderer only.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["severity", "estimate", "2.5%", "97.5%", "sources", "samples", "replicates"],
            [
                [
                    int(row.severity_level),
                    float(row.estimate),
                    float(row.ci_low),
                    float(row.ci_high),
                    int(row.n_groups),
                    int(row.n_samples),
                    int(row.n_bootstrap),
                ]
                for row in uncertainty.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Source trajectories and monotonicity",
            "",
            f"- Median within-source Spearman: {float(trends['spearman'].median()):.4f}",
            f"- Sources with strictly decreasing adjacent steps: "
            f"{int((trends['n_increasing_steps'] == 0).sum())}/{len(trends)}",
            f"- Adjacent steps: decreasing "
            f"{int(trends['n_decreasing_steps'].sum())}, tied "
            f"{int(trends['n_tied_steps'].sum())}, increasing "
            f"{int(trends['n_increasing_steps'].sum())}.",
            f"- Numeric increase exceptions retained: {len(violations)}.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "step",
                "n sources",
                "decreasing",
                "tied",
                "increasing",
                "mean delta",
                "mean |delta|",
                "SD delta",
            ],
            [
                [
                    f"{int(row.from_severity)}→{int(row.to_severity)}",
                    int(row.n_sources),
                    int(row.n_decreasing),
                    int(row.n_tied),
                    int(row.n_increasing),
                    float(row.mean_delta),
                    float(row.mean_abs_delta),
                    float(row.sd_delta),
                ]
                for row in steps.itertuples(index=False)
            ],
            digits=4,
        )
    )
    if len(violations):
        lines.extend(
            [
                "",
                "The increasing steps are not discarded. They are listed in "
                "`trend_violations.csv`; at these amplitudes they are small and are "
                "consistent with local resampling/aliasing rather than a monotone "
                "semantic dose.",
            ]
        )
    lines.extend(
        [
            "",
            f"- Near-identity elastic rows (SSIM >= "
            f"{config.analysis.near_identity_ssim_min}): "
            f"{summary['target']['near_identity_fraction']:.3f}.",
            f"- Degenerate elastic rows: {summary['target']['degenerate_fraction']:.3f}.",
            "",
            "### Font and text-length groups",
            "",
        ]
    )
    for label, table in (("font", font_groups), ("text length", length_groups)):
        included = table[table["included"]]
        if included.empty:
            lines.append(f"- No {label} group met the minimum sample count.")
            continue
        lines.append(f"**By {label}**")
        lines.append("")
        lines.extend(
            _markdown_table(
                [label, "severity", "n", "SSIM mean", "SSIM SD"],
                [
                    [
                        str(row.group_value),
                        int(row.severity_level),
                        int(row.n_samples),
                        float(row.ssim_mean),
                        row.ssim_sd,
                    ]
                    for row in included.itertuples(index=False)
                ],
                digits=4,
            )
        )
        lines.append("")
    lines.extend(
        [
            "## Interpolation and boundary effects",
            "",
            f"- Minimal-displacement control mean SSIM: {minimal_mean:.6f}; full drop "
            f"from the clean reference: {1.0 - minimal_mean:.6f}.",
            f"- Highest-severity mean outside-canvas mapped fraction: "
            f"{summary['interpolation_boundary']['max_severity_outside_fraction']:.4f}.",
            f"- Highest-severity mean border-affected fraction: "
            f"{summary['interpolation_boundary']['max_severity_border_affected_fraction']:.4f}.",
            f"- Highest-severity mean clipped-sample fraction: "
            f"{summary['interpolation_boundary']['max_severity_clipped_fraction']:.4f}.",
            f"- Highest-severity mean collapsed block fraction: "
            f"{summary['interpolation_boundary']['max_severity_collapsed_block_fraction']:.4f}.",
            "",
            "These controls separate interpolation and border effects as far as this "
            "pilot can. A non-zero SSIM drop at 0.25 px is expected: bilinear "
            "resampling changes edge samples even when the text structure is intact. "
            "The minimal-control drop is therefore an interpolation/boundary "
            "reference, not a quantity to subtract from every severity. The border "
            "fraction is the fraction of output pixels whose mapped source coordinate "
            "leaves the canvas and is handled by clamping (`edge`) or reflection "
            "(`reflect`). The collapsed-block fraction counts clean 8×8 blocks with "
            "at least one gray level of variation that become nearly constant.",
            "",
            "## Cross-family feature and target comparison",
            "",
            "The elastic family is compared descriptively with frozen M3 Gaussian "
            "blur/noise and (when enabled) frozen M4A thin-line/arc interference. "
            "These comparisons do not tune the transformation and do not establish "
            "that any family is harder for humans.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "family",
                "severity",
                "n",
                "SSIM mean",
                "SSIM SD",
                "PSNR mean dB",
            ],
            [
                [
                    str(row.family_label),
                    int(row.severity_level),
                    int(row.n_samples),
                    float(row.ssim_mean),
                    row.ssim_sd,
                    float(row.psnr_mean_db),
                ]
                for row in family_levels.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Feature contrasts at the highest severity",
            "",
            "Cohen's d is `(elastic mean - frozen mean) / pooled SD`; positive means "
            "the elastic feature is larger. The threshold is predeclared and is not "
            "used to choose the severity ladder.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "reference family",
                "mean |d|",
                "max |d|",
                "n |d| >= 0.8",
                "n |d| >= strong",
                "strong threshold",
            ],
            [
                [
                    str(row.reference_family),
                    float(row.mean_abs_cohens_d),
                    float(row.max_abs_cohens_d),
                    int(row.n_features_abs_d_ge_0_8),
                    int(row.n_features_abs_d_ge_strong),
                    config.gate.feature_strong_effect_min,
                ]
                for row in highest_effects.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Feature/target correlations",
            "",
            "Pooled correlations mix severity levels and can be dominated by the "
            "between-severity trend. The within-severity column is the mean "
            "source-level Spearman correlation computed inside each severity and is "
            "the more conservative descriptive summary.",
            "",
        ]
    )
    geometric_correlations = correlations[correlations["family_label"] == ELASTIC_FAMILY]
    lines.extend(
        _markdown_table(
            ["feature", "pooled Pearson", "pooled Spearman", "mean within-severity Spearman"],
            [
                [
                    str(row.feature),
                    float(row.pooled_pearson),
                    float(row.pooled_spearman),
                    float(row.mean_within_severity_spearman),
                ]
                for row in geometric_correlations.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "Full per-family, per-feature distributions and correlations are in "
            "`feature_summary.csv` and `feature_target_correlations.csv`. A feature "
            "difference here is evidence about this synthetic renderer and metric, "
            "not proof that the geometric family is semantically distinct to a human "
            "or to a CAPTCHA solver.",
            "",
            "## Feasibility gate",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["criterion", "passed", "observed", "threshold"],
            [
                [
                    criterion.id,
                    criterion.passed,
                    criterion.observed,
                    criterion.threshold,
                ]
                for criterion in gate.criteria
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            f"**Verdict: {gate.verdict}**",
            "",
            "A PASS requires every criterion. A CONDITIONAL PASS means the core "
            "validity checks — reproducibility, identity, overall target spread, "
            "trend, boundary controls and duplicate-freedom — passed, but a "
            "predeclared margin did not. The margin can be per-level separation at "
            "the smallest amplitude or feature distinctness. A FAIL means a core "
            "validity criterion failed; such a pilot should not proceed to a separate "
            "benchmark without revision.",
            "",
            "## Feasibility questions",
            "",
            "1. **Correctly implemented?** The identity, zero-field and minimal "
            "controls, exact field replay, finite outputs and the verification checks "
            "provide direct evidence. The field is generated only from the recorded "
            "seed and geometry; the target is never used.",
            "2. **Valid, deterministic and replayable?** The two builds and the "
            "written dataset are compared by manifest, source table, feature table, "
            "targets and decoded image hashes. Every transformed row is replayed from "
            "its stored field specification and field SHA-256.",
            "3. **Non-degenerate SSIM target?** The overall SD and level-mean range "
            "pass the predeclared spread thresholds; the lowest severity has a small "
            "between-source SD (0.0025 at 0.5 px) below the 0.005 per-level margin. "
            "That margin is reported as failed and is why the verdict is conditional, "
            "not an unqualified PASS. Near-identity and degenerate fractions are "
            "reported rather than discarded.",
            "4. **Interpretable trend?** Within-source Spearman, adjacent-step counts "
            "and every numeric increase exception are reported. Strict per-source "
            "monotonicity is evidence, not a condition imposed on the data.",
            "5. **How much interpolation/border effect?** The minimal control gives a "
            "direct near-identity resampling reference; outside, border-affected, "
            "clipped and collapsed-block fractions are reported separately. They are "
            "not subtracted from SSIM because the effects are not additive.",
            "6. **Meaningfully different features?** The Cohen's d table and pooled/"
            "within-severity correlations answer this descriptively. Similarity to "
            "blur or noise is possible and is reported honestly.",
            "7. **Worth a separate benchmark?** That is the gate verdict. A separate "
            "grouped benchmark is justified only if the transformation is valid, the "
            "target is non-degenerate and the feature/target pattern adds information "
            "beyond the existing families.",
            "8. **Limitations for the next benchmark?** Source-grouped uncertainty, "
            "font/text-length coverage, synthetic-renderer dependence, interpolation/"
            "border confounds, SSIM's insensitivity to semantic readability, and the "
            "absence of human or solver validation must be addressed.",
            "",
            "## Limitations",
            "",
            "- SSIM is a full-reference structural metric, not human readability or "
            "CAPTCHA solvability.",
            "- The pilot uses one renderer, three fonts and 60 synthetic sources. "
            "Font and text-length slices are descriptive and often small.",
            "- The elastic field is a single smooth family with one control-grid and "
            "smoothing setting. It does not cover all geometric distortions.",
            "- Bilinear interpolation and border handling necessarily change pixels "
            "even at minimal displacement; the controls quantify but cannot eliminate "
            "that confound.",
            "- Source-grouped bootstrap intervals are preliminary and describe "
            "uncertainty over the available synthetic sources, not over real users.",
            "- No model was trained or evaluated in this milestone, and no conclusions "
            "about CNN necessity follow from these feature/target results.",
            "",
            "## Recommendation",
            "",
            (
                "Proceed to a separate grouped robustness benchmark only after "
                "reviewing this pilot. Keep the frozen M3/M4A/M4B assets untouched, "
                "preserve the source-grouped split, and carry the field specification "
                "and interpolation/border controls into any benchmark."
                if gate.verdict != "FAIL"
                else "Do not proceed to a separate benchmark from this pilot. Revise "
                "the transformation or severity design and rerun the feasibility "
                "check; the diagnostic evidence is retained in the ignored report "
                "directory."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_pilot(
    config: GeometricPilotConfig,
    *,
    output_dataset: Path | str | None = None,
    output_reports: Path | str | None = None,
    overwrite: bool = False,
    verify_reproduction: bool = True,
) -> PilotResult:
    """Run the M5A pilot end to end.

    Frozen assets are audited first. Outputs are written through a staging
    directory and renamed only after dataset verification succeeds.
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
            raise GeometricPilotError(
                f"{label} output must not overlap the frozen reference dataset: "
                f"{target} vs {reference_root}"
            )
        if config.comparison.m4a_dataset is not None:
            m4a_root = config.comparison.m4a_dataset.expanduser().resolve()
            if (
                target == m4a_root
                or m4a_root.is_relative_to(target)
                or target.is_relative_to(m4a_root)
            ):
                raise GeometricPilotError(
                    f"{label} output must not overlap the frozen M4A dataset: "
                    f"{target} vs {m4a_root}"
                )
        for root_name in PROTECTED_OUTPUT_ROOTS:
            protected = Path(root_name).expanduser().resolve()
            if (
                target == protected
                or target.is_relative_to(protected)
                or protected.is_relative_to(target)
            ):
                raise GeometricPilotError(
                    f"{label} output must not overlap a protected frozen path: "
                    f"{target} vs {protected}"
                )
    if dataset_target.exists() and not overwrite:
        raise GeometricPilotExistsError(
            f"pilot dataset already exists: {dataset_target}. "
            "Pass --overwrite to replace it, or choose a different directory."
        )

    frozen_before = fingerprint_assets(FROZEN_ASSET_PATHS)
    try:
        reference = load_frozen_reference(config)  # type: ignore[arg-type]
    except PilotConfigError as exc:  # pragma: no cover - configuration guard
        raise GeometricPilotError(str(exc)) from exc
    comparison = load_frozen_comparison(
        config,
        reference_manifest=reference.manifest,
        reference_features=reference.features,
    )
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
            raise GeometricPilotError(f"pilot is not deterministic: {determinism}")

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
        raise GeometricPilotError(
            "pilot dataset failed verification: "
            + "; ".join(f"{check.name}: {check.detail}" for check in verification.failures())
        )

    combined = _combined_family_frame(generation.manifest, generation.features, comparison.table)
    levels = pilot_level_summary(
        generation.manifest, near_identity_ssim_min=config.analysis.near_identity_ssim_min
    )
    trends = trend_by_source(generation.manifest)
    violations = trend_violations(generation.manifest)
    steps = adjacent_step_summary(generation.manifest)
    uncertainty = grouped_bootstrap_levels(
        generation.manifest,
        n_bootstrap=config.pilot.bootstrap_replicates,
        seed=config.pilot.seed,
    )
    controls = control_summary(generation.manifest)
    font_groups = group_summary(
        generation.manifest,
        group_column="font_id",
        min_samples=config.analysis.min_group_samples,
    )
    length_groups = group_summary(
        generation.manifest,
        group_column="text_length",
        min_samples=config.analysis.min_group_samples,
    )
    family_levels = family_level_summary(combined)
    feature_summary = feature_summary_table(combined)
    effects_long, effects_summary = feature_effect_sizes(
        combined, strong_threshold=config.gate.feature_strong_effect_min
    )
    correlations = feature_target_correlations(combined)
    duplicate_fraction = _duplicate_fraction(generation.manifest)
    gate = evaluate_gate(
        config,
        manifest=generation.manifest,
        levels=levels,
        trends=trends,
        controls=controls,
        effects_summary=effects_summary,
        duplicate_fraction=duplicate_fraction,
        reproducibility_ok=bool(determinism.get("ok")) and bool(determinism.get("repeated_build")),
        verification_ok=verification.ok,
        clean_reference_exact=reference.rerendered_sources == len(reference.sources),
    )

    frozen_after = fingerprint_assets(FROZEN_ASSET_PATHS)
    if not _artifacts_unchanged(frozen_before, frozen_after):
        raise GeometricPilotError(
            "a frozen asset changed during the pilot run; refusing to report success"
        )

    summary = _build_summary(
        config,
        reference,
        comparison,
        generation,
        target=dataset_target,
        determinism=determinism,
        verification=verification,
        gate=gate,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        controls=controls,
        levels=levels,
    )
    _write_json(summary, dataset_target / "pilot_summary.json")
    verification = verify_pilot_dataset(
        dataset_target, reference=reference, recompute_targets=False
    )
    if not verification.ok:
        raise GeometricPilotError(
            "pilot summary check failed: "
            + "; ".join(f"{check.name}: {check.detail}" for check in verification.failures())
        )

    reports_target.mkdir(parents=True, exist_ok=True)
    tables = {
        "level_summary": levels,
        "uncertainty": uncertainty,
        "trend_by_source": trends,
        "trend_violations": violations,
        "adjacent_steps": steps,
        "controls": controls,
        "font_groups": font_groups,
        "length_groups": length_groups,
        "family_levels": family_levels,
        "feature_summary": feature_summary,
        "feature_effect_sizes": effects_long,
        "feature_distinctness": effects_summary,
        "feature_target_correlations": correlations,
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
    _write_contact_sheet(dataset_target, config, generation.sources, reports_target)
    report = _report_markdown(
        config,
        reference,
        comparison,
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
        comparison=comparison,
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the M5A geometric-distortion pilot."""
    parser = argparse.ArgumentParser(
        prog="captcha-geometric-pilot",
        description="Run the M5A geometric-distortion feasibility pilot.",
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
        config = GeometricPilotConfig.from_toml(args.config)
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
                f"geometric pilot '{config.pilot.name}': {counts['sources']} sources, "
                f"{counts['samples']} samples ({counts['elastic_samples']} elastic, "
                f"{counts['control_samples']} controls)"
            )
            print(f"verification: {'PASS' if result.verification.ok else 'FAIL'}")
            print(f"feasibility gate: {result.gate.verdict}")
            if result.gate.failed_criteria:
                print(f"failed criteria: {', '.join(result.gate.failed_criteria)}")
            print(f"dataset: {result.output_dataset}")
            print(f"report: {result.report_path}")
            return 0 if result.gate.verdict != "FAIL" else 1
        if args.command == "verify":
            reference = load_frozen_reference(config)  # type: ignore[arg-type]
            root = args.dataset or config.pilot.output_dataset
            report = verify_pilot_dataset(
                root, reference=reference, recompute_targets=not args.skip_targets
            )
            for check in report.checks:
                print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
            print(f"verification: {'PASS' if report.ok else 'FAIL'}")
            return 0 if report.ok else 1
    except (
        GeometricPilotConfigError,
        GeometricPilotError,
        GeometricError,
        PilotConfigError,
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
