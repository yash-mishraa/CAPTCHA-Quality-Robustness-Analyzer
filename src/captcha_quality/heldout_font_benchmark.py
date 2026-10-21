"""Milestone M7B held-out font generalisation benchmark.

M7B asks how well the frozen M5C champion predicts SSIM against an exact clean
reference when the text is rendered in fonts that were absent from model
development. It is a zero-shot evaluation of a frozen model, not a training or
model-selection experiment.

The benchmark is deliberately separate from the frozen M3-M6 artifacts:

* it creates a new versioned dataset under ``data/processed/``;
* it reuses the existing renderer, corruption definitions, target definition
  and eight-feature extractor unchanged;
* it loads the exact M5C ``selected_policy`` artifact and verifies its hash,
  feature schema, target and calibration;
* it never calls a model ``fit`` method or writes a model artifact;
* every row belongs to one untouched evaluation split, and the independent unit
  is the source group, not an image variant.

The target is SSIM between a corrupted synthetic image and its source-specific
clean reference. PSNR is diagnostic only. Nothing here measures human
readability, CAPTCHA solvability, security or real-world robustness.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFont
from PIL import __version__ as pillow_version

from captcha_quality import robustness_benchmark as rb
from captcha_quality.corruptions import (
    FAMILY_BLUR,
    FAMILY_NOISE,
    CorruptionSpec,
    SeverityLadder,
    apply_corruption,
    derive_noise_seed,
)
from captcha_quality.dataset import CheckResult, VerificationReport, load_grayscale
from captcha_quality.features import FEATURE_NAMES, extract_feature_table
from captcha_quality.geometric import (
    SUPPORTED_BORDER_MODES,
    SUPPORTED_INTERPOLATIONS,
    ElasticFieldSpec,
    apply_elastic,
    field_spec_json,
    generate_displacement_field,
)
from captcha_quality.imaging import ImageContractError
from captcha_quality.interference import (
    VARIANT_ARC_STROKES,
    VARIANT_THIN_LINES,
    InterferenceLadder,
    apply_strokes,
    geometry_to_json,
    plan_stroke_pool,
)
from captcha_quality.modelling import (
    TARGET,
    ModelError,
    bootstrap_metrics,
    predict,
    regression_metrics,
    sklearn_version,
)
from captcha_quality.provenance import (
    canonical_json,
    code_revision,
    json_safe,
    package_version,
    sha256_bytes,
    sha256_file,
)
from captcha_quality.quality_metrics import compute_pair_metrics
from captcha_quality.render import (
    CANDIDATE_FONTS,
    FontResolutionError,
    RenderConfig,
    RenderError,
    ResolvedFont,
    render_clean,
    sample_text,
)

AUDIT_VERSION: Final[str] = "v0.1.0"
CLEAN_FAMILY: Final[str] = "clean"
ELASTIC_FAMILY: Final[str] = "elastic"
INTERFERENCE_FAMILIES: Final[tuple[str, ...]] = (
    VARIANT_THIN_LINES,
    VARIANT_ARC_STROKES,
)
PRIMARY_FAMILIES: Final[tuple[str, ...]] = (
    FAMILY_BLUR,
    FAMILY_NOISE,
    VARIANT_THIN_LINES,
    VARIANT_ARC_STROKES,
    ELASTIC_FAMILY,
)
FAMILY_ORDER: Final[tuple[str, ...]] = (
    CLEAN_FAMILY,
    FAMILY_BLUR,
    FAMILY_NOISE,
    VARIANT_THIN_LINES,
    VARIANT_ARC_STROKES,
    ELASTIC_FAMILY,
)
SCOPE_ORDER: Final[tuple[str, ...]] = (
    "all_primary",
    "clean_reference",
    "original_blur_noise",
    "gaussian_blur",
    "gaussian_noise",
    "interference",
    "thin_lines",
    "arc_strokes",
    "geometric_primary",
    "geometric_primary_1px",
    "geometric_primary_2px",
    "geometric_primary_3px",
    "diagnostic_0_5px",
)
PRIMARY_SCOPE_SET: Final[frozenset[str]] = frozenset(
    {
        "all_primary",
        "original_blur_noise",
        "gaussian_blur",
        "gaussian_noise",
        "interference",
        "thin_lines",
        "arc_strokes",
        "geometric_primary",
        "geometric_primary_1px",
        "geometric_primary_2px",
        "geometric_primary_3px",
    }
)
DIAGNOSTIC_SCOPE_SET: Final[frozenset[str]] = frozenset({"clean_reference", "diagnostic_0_5px"})
FROZEN_FONT_KEYS: Final[frozenset[str]] = frozenset({"arial", "courier_new", "times_new_roman"})
FROZEN_FONT_HASHES: Final[frozenset[str]] = frozenset(
    sha256_file(spec.candidates[0])
    for spec in CANDIDATE_FONTS
    if spec.candidates and spec.candidates[0].is_file()
)

MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "source_id",
    "split_group_id",
    "split",
    "font_id",
    "font_sha256",
    "text_length",
    "render_seed",
    "variant_id",
    "variant_family",
    "severity_level",
    "amplitude_px",
    "corruption_parameters",
    "variant_seed",
    "field_seed",
    "field_sha256",
    "field_spec_json",
    "stroke_geometry_json",
    "image_path",
    "reference_path",
    "ssim_target",
    "psnr_target",
    "psnr_infinite",
    "height",
    "width",
    "image_sha256",
    "reference_sha256",
    "text_sha256",
    "dataset_version",
    "generation_config_hash",
    "protocol_scope",
)

SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "source_id",
    "split_group_id",
    "split",
    "font_id",
    "font_path",
    "font_sha256",
    "text_length",
    "render_seed",
    "background_level",
    "height",
    "width",
    "clean_image_path",
    "clean_image_sha256",
    "text_sha256",
    "dataset_version",
    "generation_config_hash",
)

FEATURE_COLUMNS: Final[tuple[str, ...]] = ("sample_id", *FEATURE_NAMES)

OUTPUT_TABLE_NAMES: Final[tuple[str, ...]] = (
    "dataset_composition.csv",
    "target_distributions.csv",
    "overall_metrics.csv",
    "font_metrics.csv",
    "family_metrics.csv",
    "source_metrics.csv",
    "predictions.csv",
    "uncertainty.csv",
    "comparison.csv",
    "isolation_checks.csv",
)

#: Frozen assets whose bytes must not change during M7B. Missing directories are
#: recorded as absent; the check never treats an absent optional output as a
#: reason to skip a path that is present.
FROZEN_ASSET_PATHS: Final[tuple[str, ...]] = (
    "configs/dataset_v0_1_0.toml",
    "configs/benchmark_v0_1_0.toml",
    "configs/robustness_pilot_occlusion_v0_1_0.toml",
    "configs/robustness_benchmark_v0_1_0.toml",
    "configs/geometric_distortion_pilot_v0_1_0.toml",
    "configs/geometric_robustness_benchmark_v0_1_0.toml",
    "configs/retention_aware_selection_v0_1_0.toml",
    "configs/retention_aware_training_v0_1_0.toml",
    "configs/spatial_feature_study_v0_1_0.toml",
    "configs/spatial_resolution_study_v0_1_0.toml",
    "configs/local_structure_study_v0_1_0.toml",
    "configs/structure_tensor_study_v0_1_0.toml",
    "configs/dataset_adequacy_audit_v0_1_0.toml",
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "data/processed/geometric_distortion_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "models/robustness_benchmark_v0_1_0",
    "models/geometric_robustness_benchmark_v0_1_0",
    "models/retention_aware_selection_v0_1_0",
    "models/retention_aware_training_v0_1_0",
    "models/spatial_feature_study_v0_1_0",
    "models/spatial_resolution_study_v0_1_0",
    "models/local_structure_study_v0_1_0",
    "models/structure_tensor_study_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
    "reports/geometric_distortion_pilot",
    "reports/geometric_robustness_benchmark_v0_1_0",
    "reports/retention_aware_selection_v0_1_0",
    "reports/retention_aware_training_v0_1_0",
    "reports/spatial_feature_study_v0_1_0",
    "reports/spatial_resolution_study_v0_1_0",
    "reports/local_structure_study_v0_1_0",
    "reports/structure_tensor_study_v0_1_0",
    "reports/dataset_adequacy_audit_v0_1_0",
)

PROTECTED_OUTPUT_ROOTS: Final[tuple[str, ...]] = (
    "configs",
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "data/processed/geometric_distortion_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "models/robustness_benchmark_v0_1_0",
    "models/geometric_robustness_benchmark_v0_1_0",
    "models/retention_aware_selection_v0_1_0",
    "models/retention_aware_training_v0_1_0",
    "models/spatial_feature_study_v0_1_0",
    "models/spatial_resolution_study_v0_1_0",
    "models/local_structure_study_v0_1_0",
    "models/structure_tensor_study_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
    "reports/geometric_distortion_pilot",
    "reports/geometric_robustness_benchmark_v0_1_0",
    "reports/retention_aware_selection_v0_1_0",
    "reports/retention_aware_training_v0_1_0",
    "reports/spatial_feature_study_v0_1_0",
    "reports/spatial_resolution_study_v0_1_0",
    "reports/local_structure_study_v0_1_0",
    "reports/structure_tensor_study_v0_1_0",
    "reports/dataset_adequacy_audit_v0_1_0",
)

_SHA256_PATTERN: Final[str] = "^[0-9a-f]{64}$"
_EXPECTED_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "run",
        "schema",
        "dataset",
        "render",
        "fonts",
        "corruptions",
        "geometry",
        "protocol",
        "model",
        "reference",
        "expectations",
    }
)


class HeldoutFontError(RuntimeError):
    """Raised when M7B cannot be generated, evaluated or verified as declared."""


class HeldoutFontConfigError(ValueError):
    """Raised when the M7B configuration is missing or malformed."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    name: str
    version: str
    seed: int
    n_bootstrap: int
    output_dataset: Path
    output_reports: Path


@dataclass(frozen=True, slots=True)
class SchemaSettings:
    feature_set_version: str
    feature_names: tuple[str, ...]
    target: str
    split_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    name: str
    version: str
    sources_per_font: int
    text_lengths: tuple[int, ...]
    text_length_weights: tuple[float, ...]
    charset: str


@dataclass(frozen=True, slots=True)
class FontSettings:
    key: str
    path: Path
    sha256: str
    family: str
    rationale: str


@dataclass(frozen=True, slots=True)
class RenderSettings:
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

    def to_render_config(self, *, text_length: int, charset: str) -> RenderConfig:
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
class CorruptionSettings:
    blur_radii_px: tuple[float, ...]
    noise_sigmas: tuple[float, ...]
    thin_line_stroke_counts: tuple[int, ...]
    arc_stroke_counts: tuple[int, ...]
    stroke_width_px: int
    stroke_intensity: int
    line_angle_deg: tuple[float, float]
    arc_sweep_deg: tuple[float, float]
    arc_radius_fraction: tuple[float, float]
    arc_aspect_ratio: tuple[float, float]
    elastic_primary_amplitudes_px: tuple[float, ...]
    elastic_diagnostic_amplitudes_px: tuple[float, ...]

    def blur_ladder(self) -> SeverityLadder:
        return SeverityLadder(FAMILY_BLUR, (0.0, *self.blur_radii_px), "px")

    def noise_ladder(self) -> SeverityLadder:
        return SeverityLadder(FAMILY_NOISE, (0.0, *self.noise_sigmas), "sigma")

    def interference_ladder(self, variant: str) -> InterferenceLadder:
        counts = (
            self.thin_line_stroke_counts
            if variant == VARIANT_THIN_LINES
            else self.arc_stroke_counts
        )
        return InterferenceLadder(
            variant=variant,
            stroke_counts=counts,
            width_px=self.stroke_width_px,
            intensity=self.stroke_intensity,
            line_angle_deg=self.line_angle_deg,
            arc_sweep_deg=self.arc_sweep_deg,
            arc_radius_fraction=self.arc_radius_fraction,
            arc_aspect_ratio=self.arc_aspect_ratio,
        )

    def elastic_amplitudes(self) -> tuple[float, ...]:
        return tuple(
            sorted(
                {
                    *self.elastic_diagnostic_amplitudes_px,
                    *self.elastic_primary_amplitudes_px,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class GeometrySettings:
    grid_rows: int
    grid_cols: int
    smoothing_sigma_px: float
    interpolation: str
    border_mode: str

    def field_spec(self, shape: tuple[int, int], seed: int) -> ElasticFieldSpec:
        return ElasticFieldSpec(
            height=int(shape[0]),
            width=int(shape[1]),
            seed=seed,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            smoothing_sigma_px=self.smoothing_sigma_px,
            interpolation=self.interpolation,
            border_mode=self.border_mode,
        )


@dataclass(frozen=True, slots=True)
class ProtocolSettings:
    primary_scopes: tuple[str, ...]
    diagnostic_scopes: tuple[str, ...]
    bootstrap_scopes: tuple[str, ...]
    per_font_bootstrap_scopes: tuple[str, ...]
    near_degenerate_sd: float
    min_sources_per_font: int
    evidence_rule: str


@dataclass(frozen=True, slots=True)
class ModelSettings:
    model_dir: Path
    model_stem: str
    artifact_sha256: str
    card_sha256: str
    expected_model_family: str
    expected_calibration: str


@dataclass(frozen=True, slots=True)
class ReferenceSettings:
    m3_dataset: Path
    m3_config: Path
    m3_config_sha256: str
    m3_sources: Path
    m3_sources_sha256: str
    m5c_config: Path
    m5c_config_sha256: str
    m5c_test_metrics: Path
    m5c_test_metrics_sha256: str
    m5c_test_predictions: Path
    m5c_test_predictions_sha256: str
    m5c_uncertainty: Path
    m5c_uncertainty_sha256: str


@dataclass(frozen=True, slots=True)
class Expectations:
    n_sources: int
    sources_per_font: int
    n_fonts: int
    rows_per_source: int
    total_rows: int
    clean_rows: int
    original_blur_rows: int
    original_noise_rows: int
    thin_line_rows: int
    arc_stroke_rows: int
    elastic_primary_rows: int
    diagnostic_rows: int


@dataclass(frozen=True, slots=True)
class HeldoutConfig:
    run: RunSettings
    schema: SchemaSettings
    dataset: DatasetSettings
    render: RenderSettings
    fonts: tuple[FontSettings, ...]
    corruptions: CorruptionSettings
    geometry: GeometrySettings
    protocol: ProtocolSettings
    model: ModelSettings
    reference: ReferenceSettings
    expectations: Expectations
    config_path: Path
    config_sha256: str

    @property
    def font_keys(self) -> tuple[str, ...]:
        return tuple(font.key for font in self.fonts)

    def font(self, key: str) -> FontSettings:
        for font in self.fonts:
            if font.key == key:
                return font
        raise HeldoutFontConfigError(f"unknown held-out font key {key!r}")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "run": {
                "name": self.run.name,
                "version": self.run.version,
                "seed": self.run.seed,
                "n_bootstrap": self.run.n_bootstrap,
            },
            "schema": {
                "feature_set_version": self.schema.feature_set_version,
                "feature_names": list(self.schema.feature_names),
                "target": self.schema.target,
                "split_names": list(self.schema.split_names),
            },
            "dataset": {
                "name": self.dataset.name,
                "version": self.dataset.version,
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
            "fonts": [
                {
                    "key": font.key,
                    "path": str(font.path),
                    "sha256": font.sha256,
                    "family": font.family,
                    "rationale": font.rationale,
                }
                for font in self.fonts
            ],
            "corruptions": {
                "blur_radii_px": list(self.corruptions.blur_radii_px),
                "noise_sigmas": list(self.corruptions.noise_sigmas),
                "thin_line_stroke_counts": list(self.corruptions.thin_line_stroke_counts),
                "arc_stroke_counts": list(self.corruptions.arc_stroke_counts),
                "stroke_width_px": self.corruptions.stroke_width_px,
                "stroke_intensity": self.corruptions.stroke_intensity,
                "line_angle_deg": list(self.corruptions.line_angle_deg),
                "arc_sweep_deg": list(self.corruptions.arc_sweep_deg),
                "arc_radius_fraction": list(self.corruptions.arc_radius_fraction),
                "arc_aspect_ratio": list(self.corruptions.arc_aspect_ratio),
                "elastic_primary_amplitudes_px": list(
                    self.corruptions.elastic_primary_amplitudes_px
                ),
                "elastic_diagnostic_amplitudes_px": list(
                    self.corruptions.elastic_diagnostic_amplitudes_px
                ),
            },
            "geometry": {
                "grid_rows": self.geometry.grid_rows,
                "grid_cols": self.geometry.grid_cols,
                "smoothing_sigma_px": self.geometry.smoothing_sigma_px,
                "interpolation": self.geometry.interpolation,
                "border_mode": self.geometry.border_mode,
            },
            "protocol": {
                "primary_scopes": list(self.protocol.primary_scopes),
                "diagnostic_scopes": list(self.protocol.diagnostic_scopes),
                "bootstrap_scopes": list(self.protocol.bootstrap_scopes),
                "per_font_bootstrap_scopes": list(self.protocol.per_font_bootstrap_scopes),
                "near_degenerate_sd": self.protocol.near_degenerate_sd,
                "min_sources_per_font": self.protocol.min_sources_per_font,
                "evidence_rule": self.protocol.evidence_rule,
            },
            "model": {
                "model_dir": str(self.model.model_dir),
                "model_stem": self.model.model_stem,
                "artifact_sha256": self.model.artifact_sha256,
                "card_sha256": self.model.card_sha256,
                "expected_model_family": self.model.expected_model_family,
                "expected_calibration": self.model.expected_calibration,
            },
            "reference": {
                "m3_dataset": str(self.reference.m3_dataset),
                "m3_config": str(self.reference.m3_config),
                "m3_config_sha256": self.reference.m3_config_sha256,
                "m3_sources": str(self.reference.m3_sources),
                "m3_sources_sha256": self.reference.m3_sources_sha256,
                "m5c_config": str(self.reference.m5c_config),
                "m5c_config_sha256": self.reference.m5c_config_sha256,
                "m5c_test_metrics": str(self.reference.m5c_test_metrics),
                "m5c_test_metrics_sha256": self.reference.m5c_test_metrics_sha256,
                "m5c_test_predictions": str(self.reference.m5c_test_predictions),
                "m5c_test_predictions_sha256": self.reference.m5c_test_predictions_sha256,
                "m5c_uncertainty": str(self.reference.m5c_uncertainty),
                "m5c_uncertainty_sha256": self.reference.m5c_uncertainty_sha256,
            },
            "expectations": {
                "n_sources": self.expectations.n_sources,
                "sources_per_font": self.expectations.sources_per_font,
                "n_fonts": self.expectations.n_fonts,
                "rows_per_source": self.expectations.rows_per_source,
                "total_rows": self.expectations.total_rows,
                "clean_rows": self.expectations.clean_rows,
                "original_blur_rows": self.expectations.original_blur_rows,
                "original_noise_rows": self.expectations.original_noise_rows,
                "thin_line_rows": self.expectations.thin_line_rows,
                "arc_stroke_rows": self.expectations.arc_stroke_rows,
                "elastic_primary_rows": self.expectations.elastic_primary_rows,
                "diagnostic_rows": self.expectations.diagnostic_rows,
            },
        }

    def generation_payload(self) -> dict[str, Any]:
        payload = self.identity_payload()
        return {
            "dataset": payload["dataset"],
            "render": payload["render"],
            "fonts": payload["fonts"],
            "corruptions": payload["corruptions"],
            "geometry": payload["geometry"],
        }

    @property
    def generation_hash(self) -> str:
        return sha256_bytes(canonical_json(self.generation_payload()).encode("utf-8"))

    @property
    def config_hash(self) -> str:
        return sha256_bytes(canonical_json(self.identity_payload()).encode("utf-8"))

    def describe(self) -> str:
        return (
            f"{self.run.name} {self.run.version} "
            f"(seed {self.run.seed}, bootstrap {self.run.n_bootstrap})"
        )


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise HeldoutFontConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise HeldoutFontConfigError(f"missing or invalid [{name}] table")
    return value


def _required(section: Mapping[str, object], key: str, section_name: str) -> object:
    if key not in section:
        raise HeldoutFontConfigError(f"missing required key {section_name}.{key}")
    return section[key]


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HeldoutFontConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HeldoutFontConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HeldoutFontConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise HeldoutFontConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise HeldoutFontConfigError(f"{label} must be a boolean, got {value!r}")
    return value


def _as_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise HeldoutFontConfigError(f"{label} must be a TOML array")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    items = tuple(_as_str(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise HeldoutFontConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_int_tuple(value: object, label: str) -> tuple[int, ...]:
    items = tuple(_as_int(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise HeldoutFontConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_float_tuple(
    value: object, label: str, *, allow_duplicates: bool = False
) -> tuple[float, ...]:
    items = tuple(_as_float(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items:
        raise HeldoutFontConfigError(f"{label} must be non-empty")
    if not allow_duplicates and len(set(items)) != len(items):
        raise HeldoutFontConfigError(f"{label} must be duplicate-free")
    return items


def _as_float_pair(value: object, label: str) -> tuple[float, float]:
    items = _as_float_tuple(value, label)
    if len(items) != 2:
        raise HeldoutFontConfigError(f"{label} must contain exactly two numbers")
    return items[0], items[1]


def _sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise HeldoutFontConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _parse_run(raw: Mapping[str, object]) -> RunSettings:
    _check_keys(
        raw,
        {"name", "version", "seed", "n_bootstrap", "output_dataset", "output_reports"},
        "run",
    )
    settings = RunSettings(
        name=_as_str(raw["name"], "run.name"),
        version=_as_str(raw["version"], "run.version"),
        seed=_as_int(raw["seed"], "run.seed"),
        n_bootstrap=_as_int(raw["n_bootstrap"], "run.n_bootstrap"),
        output_dataset=Path(_as_str(raw["output_dataset"], "run.output_dataset")),
        output_reports=Path(_as_str(raw["output_reports"], "run.output_reports")),
    )
    if settings.seed < 0:
        raise HeldoutFontConfigError("run.seed must be >= 0")
    if settings.n_bootstrap < 1:
        raise HeldoutFontConfigError("run.n_bootstrap must be >= 1")
    return settings


def _parse_schema(raw: Mapping[str, object]) -> SchemaSettings:
    _check_keys(
        raw,
        {"feature_set_version", "feature_names", "target", "split_names"},
        "schema",
    )
    settings = SchemaSettings(
        feature_set_version=_as_str(raw["feature_set_version"], "schema.feature_set_version"),
        feature_names=_as_str_tuple(raw["feature_names"], "schema.feature_names"),
        target=_as_str(raw["target"], "schema.target"),
        split_names=_as_str_tuple(raw["split_names"], "schema.split_names"),
    )
    if settings.feature_set_version != "v1":
        raise HeldoutFontConfigError("schema.feature_set_version must be 'v1'")
    if settings.feature_names != FEATURE_NAMES:
        raise HeldoutFontConfigError(
            f"schema.feature_names must be the frozen eight features, got {settings.feature_names}"
        )
    if settings.target != TARGET:
        raise HeldoutFontConfigError(f"schema.target must be {TARGET!r}")
    if settings.split_names != ("eval",):
        raise HeldoutFontConfigError("schema.split_names must be exactly ['eval']")
    return settings


def _parse_dataset(raw: Mapping[str, object]) -> DatasetSettings:
    _check_keys(
        raw,
        {
            "name",
            "version",
            "sources_per_font",
            "text_lengths",
            "text_length_weights",
            "charset",
        },
        "dataset",
    )
    settings = DatasetSettings(
        name=_as_str(raw["name"], "dataset.name"),
        version=_as_str(raw["version"], "dataset.version"),
        sources_per_font=_as_int(raw["sources_per_font"], "dataset.sources_per_font"),
        text_lengths=_as_int_tuple(raw["text_lengths"], "dataset.text_lengths"),
        text_length_weights=_as_float_tuple(
            raw["text_length_weights"],
            "dataset.text_length_weights",
            allow_duplicates=True,
        ),
        charset=_as_str(raw["charset"], "dataset.charset"),
    )
    if settings.sources_per_font < 1:
        raise HeldoutFontConfigError("dataset.sources_per_font must be >= 1")
    if len(settings.text_lengths) != len(settings.text_length_weights):
        raise HeldoutFontConfigError("dataset text lengths and weights must have equal length")
    if any(length < 1 for length in settings.text_lengths):
        raise HeldoutFontConfigError("dataset.text_lengths must be >= 1")
    if any(weight <= 0.0 for weight in settings.text_length_weights):
        raise HeldoutFontConfigError("dataset.text_length_weights must be positive")
    if abs(sum(settings.text_length_weights) - 1.0) > 1e-6:
        raise HeldoutFontConfigError("dataset.text_length_weights must sum to 1.0")
    if len(settings.charset) < 2 or len(set(settings.charset)) != len(settings.charset):
        raise HeldoutFontConfigError("dataset.charset must be non-empty and duplicate-free")
    return settings


def _parse_render(raw: Mapping[str, object]) -> RenderSettings:
    _check_keys(
        raw,
        {
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
        },
        "render",
    )
    width = _as_int(raw["canvas_width"], "render.canvas_width")
    height = _as_int(raw["canvas_height"], "render.canvas_height")
    low = _as_int(raw["background_min"], "render.background_min")
    high = _as_int(raw["background_max"], "render.background_max")
    settings = RenderSettings(
        canvas=(width, height),
        fit_fraction=_as_float(raw["fit_fraction"], "render.fit_fraction"),
        vertical_padding=_as_float(raw["vertical_padding"], "render.vertical_padding"),
        jitter_fraction=_as_float(raw["jitter_fraction"], "render.jitter_fraction"),
        spacing_jitter_fraction=_as_float(
            raw["spacing_jitter_fraction"], "render.spacing_jitter_fraction"
        ),
        placement_jitter_fraction=_as_float(
            raw["placement_jitter_fraction"], "render.placement_jitter_fraction"
        ),
        background_mode=_as_str(raw["background_mode"], "render.background_mode"),
        background_range=(low, high),
        center_ink=_as_bool(raw["center_ink"], "render.center_ink"),
        strict_bounds=_as_bool(raw["strict_bounds"], "render.strict_bounds"),
    )
    try:
        settings.to_render_config(text_length=1, charset="AB")
    except RenderError as exc:
        raise HeldoutFontConfigError(f"invalid render settings: {exc}") from exc
    return settings


def _parse_fonts(raw: object) -> tuple[FontSettings, ...]:
    if not isinstance(raw, list) or not raw:
        raise HeldoutFontConfigError("[[fonts]] must be a non-empty array of tables")
    fonts: list[FontSettings] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HeldoutFontConfigError(f"fonts[{index}] must be a table")
        _check_keys(item, {"key", "path", "sha256", "family", "rationale"}, f"fonts[{index}]")
        font = FontSettings(
            key=_as_str(item["key"], f"fonts[{index}].key"),
            path=Path(_as_str(item["path"], f"fonts[{index}].path")),
            sha256=_sha256(item["sha256"], f"fonts[{index}].sha256"),
            family=_as_str(item["family"], f"fonts[{index}].family"),
            rationale=_as_str(item["rationale"], f"fonts[{index}].rationale"),
        )
        if font.key in FROZEN_FONT_KEYS:
            raise HeldoutFontConfigError(
                f"font {font.key!r} is one of the frozen M3 development fonts and is not held out"
            )
        if any(existing.key == font.key for existing in fonts):
            raise HeldoutFontConfigError(f"duplicate font key {font.key!r}")
        fonts.append(font)
    return tuple(fonts)


def _parse_corruptions(raw: Mapping[str, object]) -> CorruptionSettings:
    _check_keys(
        raw,
        {
            "blur_radii_px",
            "noise_sigmas",
            "thin_line_stroke_counts",
            "arc_stroke_counts",
            "stroke_width_px",
            "stroke_intensity",
            "line_angle_deg",
            "arc_sweep_deg",
            "arc_radius_fraction",
            "arc_aspect_ratio",
            "elastic_primary_amplitudes_px",
            "elastic_diagnostic_amplitudes_px",
        },
        "corruptions",
    )
    settings = CorruptionSettings(
        blur_radii_px=_as_float_tuple(raw["blur_radii_px"], "corruptions.blur_radii_px"),
        noise_sigmas=_as_float_tuple(raw["noise_sigmas"], "corruptions.noise_sigmas"),
        thin_line_stroke_counts=_as_int_tuple(
            raw["thin_line_stroke_counts"], "corruptions.thin_line_stroke_counts"
        ),
        arc_stroke_counts=_as_int_tuple(raw["arc_stroke_counts"], "corruptions.arc_stroke_counts"),
        stroke_width_px=_as_int(raw["stroke_width_px"], "corruptions.stroke_width_px"),
        stroke_intensity=_as_int(raw["stroke_intensity"], "corruptions.stroke_intensity"),
        line_angle_deg=_as_float_pair(raw["line_angle_deg"], "corruptions.line_angle_deg"),
        arc_sweep_deg=_as_float_pair(raw["arc_sweep_deg"], "corruptions.arc_sweep_deg"),
        arc_radius_fraction=_as_float_pair(
            raw["arc_radius_fraction"], "corruptions.arc_radius_fraction"
        ),
        arc_aspect_ratio=_as_float_pair(raw["arc_aspect_ratio"], "corruptions.arc_aspect_ratio"),
        elastic_primary_amplitudes_px=_as_float_tuple(
            raw["elastic_primary_amplitudes_px"], "corruptions.elastic_primary_amplitudes_px"
        ),
        elastic_diagnostic_amplitudes_px=_as_float_tuple(
            raw["elastic_diagnostic_amplitudes_px"],
            "corruptions.elastic_diagnostic_amplitudes_px",
        ),
    )
    if any(value <= 0.0 for value in settings.blur_radii_px):
        raise HeldoutFontConfigError("corruptions.blur_radii_px must be > 0")
    if any(value <= 0.0 for value in settings.noise_sigmas):
        raise HeldoutFontConfigError("corruptions.noise_sigmas must be > 0")
    if any(value <= 0.0 for value in settings.elastic_primary_amplitudes_px):
        raise HeldoutFontConfigError("primary elastic amplitudes must be > 0")
    if any(value <= 0.0 for value in settings.elastic_diagnostic_amplitudes_px):
        raise HeldoutFontConfigError("diagnostic elastic amplitudes must be > 0")
    if set(settings.elastic_primary_amplitudes_px) & set(settings.elastic_diagnostic_amplitudes_px):
        raise HeldoutFontConfigError("primary and diagnostic elastic amplitudes must be disjoint")
    if not 1 <= settings.stroke_width_px <= 8:
        raise HeldoutFontConfigError("corruptions.stroke_width_px must be in [1, 8]")
    if not 0 <= settings.stroke_intensity <= 255:
        raise HeldoutFontConfigError("corruptions.stroke_intensity must be in [0, 255]")
    try:
        settings.blur_ladder()
        settings.noise_ladder()
        settings.interference_ladder(VARIANT_THIN_LINES)
        settings.interference_ladder(VARIANT_ARC_STROKES)
    except Exception as exc:
        raise HeldoutFontConfigError(f"invalid corruption ladders: {exc}") from exc
    return settings


def _parse_geometry(raw: Mapping[str, object]) -> GeometrySettings:
    _check_keys(
        raw,
        {"grid_rows", "grid_cols", "smoothing_sigma_px", "interpolation", "border_mode"},
        "geometry",
    )
    settings = GeometrySettings(
        grid_rows=_as_int(raw["grid_rows"], "geometry.grid_rows"),
        grid_cols=_as_int(raw["grid_cols"], "geometry.grid_cols"),
        smoothing_sigma_px=_as_float(raw["smoothing_sigma_px"], "geometry.smoothing_sigma_px"),
        interpolation=_as_str(raw["interpolation"], "geometry.interpolation"),
        border_mode=_as_str(raw["border_mode"], "geometry.border_mode"),
    )
    if settings.interpolation not in SUPPORTED_INTERPOLATIONS:
        raise HeldoutFontConfigError(
            f"geometry.interpolation must be one of {SUPPORTED_INTERPOLATIONS}"
        )
    if settings.border_mode not in SUPPORTED_BORDER_MODES:
        raise HeldoutFontConfigError(
            f"geometry.border_mode must be one of {SUPPORTED_BORDER_MODES}"
        )
    try:
        settings.field_spec((64, 192), 0)
    except Exception as exc:
        raise HeldoutFontConfigError(f"invalid geometry settings: {exc}") from exc
    return settings


def _parse_protocol(raw: Mapping[str, object]) -> ProtocolSettings:
    _check_keys(
        raw,
        {
            "primary_scopes",
            "diagnostic_scopes",
            "bootstrap_scopes",
            "per_font_bootstrap_scopes",
            "near_degenerate_sd",
            "min_sources_per_font",
            "evidence_rule",
        },
        "protocol",
    )
    settings = ProtocolSettings(
        primary_scopes=_as_str_tuple(raw["primary_scopes"], "protocol.primary_scopes"),
        diagnostic_scopes=_as_str_tuple(raw["diagnostic_scopes"], "protocol.diagnostic_scopes"),
        bootstrap_scopes=_as_str_tuple(raw["bootstrap_scopes"], "protocol.bootstrap_scopes"),
        per_font_bootstrap_scopes=_as_str_tuple(
            raw["per_font_bootstrap_scopes"], "protocol.per_font_bootstrap_scopes"
        ),
        near_degenerate_sd=_as_float(raw["near_degenerate_sd"], "protocol.near_degenerate_sd"),
        min_sources_per_font=_as_int(raw["min_sources_per_font"], "protocol.min_sources_per_font"),
        evidence_rule=_as_str(raw["evidence_rule"], "protocol.evidence_rule"),
    )
    unknown = (set(settings.primary_scopes) | set(settings.diagnostic_scopes)) - set(SCOPE_ORDER)
    if unknown:
        raise HeldoutFontConfigError(f"unknown protocol scope(s): {sorted(unknown)}")
    if set(settings.primary_scopes) & set(settings.diagnostic_scopes):
        raise HeldoutFontConfigError("primary and diagnostic scopes must be disjoint")
    if not set(settings.primary_scopes).issubset(PRIMARY_SCOPE_SET):
        raise HeldoutFontConfigError("protocol.primary_scopes contains a diagnostic-only scope")
    if not set(settings.diagnostic_scopes).issubset(DIAGNOSTIC_SCOPE_SET):
        raise HeldoutFontConfigError("protocol.diagnostic_scopes contains a primary scope")
    if not set(settings.bootstrap_scopes).issubset(set(SCOPE_ORDER)):
        raise HeldoutFontConfigError("protocol.bootstrap_scopes contains an unknown scope")
    if not set(settings.per_font_bootstrap_scopes).issubset(set(settings.bootstrap_scopes)):
        raise HeldoutFontConfigError(
            "protocol.per_font_bootstrap_scopes must be a subset of bootstrap_scopes"
        )
    if settings.near_degenerate_sd <= 0.0:
        raise HeldoutFontConfigError("protocol.near_degenerate_sd must be > 0")
    if settings.min_sources_per_font < 1:
        raise HeldoutFontConfigError("protocol.min_sources_per_font must be >= 1")
    if settings.evidence_rule != "heldout_point_outside_familiar_interval":
        raise HeldoutFontConfigError(
            "protocol.evidence_rule must be 'heldout_point_outside_familiar_interval'"
        )
    return settings


def _parse_model(raw: Mapping[str, object]) -> ModelSettings:
    _check_keys(
        raw,
        {
            "model_dir",
            "model_stem",
            "artifact_sha256",
            "card_sha256",
            "expected_model_family",
            "expected_calibration",
        },
        "model",
    )
    settings = ModelSettings(
        model_dir=Path(_as_str(raw["model_dir"], "model.model_dir")),
        model_stem=_as_str(raw["model_stem"], "model.model_stem"),
        artifact_sha256=_sha256(raw["artifact_sha256"], "model.artifact_sha256"),
        card_sha256=_sha256(raw["card_sha256"], "model.card_sha256"),
        expected_model_family=_as_str(raw["expected_model_family"], "model.expected_model_family"),
        expected_calibration=_as_str(raw["expected_calibration"], "model.expected_calibration"),
    )
    if settings.expected_calibration != "none":
        raise HeldoutFontConfigError(
            "M7B evaluates the uncalibrated champion; model.expected_calibration must be 'none'"
        )
    return settings


def _parse_reference(raw: Mapping[str, object]) -> ReferenceSettings:
    _check_keys(
        raw,
        {
            "m3_dataset",
            "m3_config",
            "m3_config_sha256",
            "m3_sources",
            "m3_sources_sha256",
            "m5c_config",
            "m5c_config_sha256",
            "m5c_test_metrics",
            "m5c_test_metrics_sha256",
            "m5c_test_predictions",
            "m5c_test_predictions_sha256",
            "m5c_uncertainty",
            "m5c_uncertainty_sha256",
        },
        "reference",
    )
    return ReferenceSettings(
        m3_dataset=Path(_as_str(raw["m3_dataset"], "reference.m3_dataset")),
        m3_config=Path(_as_str(raw["m3_config"], "reference.m3_config")),
        m3_config_sha256=_sha256(raw["m3_config_sha256"], "reference.m3_config_sha256"),
        m3_sources=Path(_as_str(raw["m3_sources"], "reference.m3_sources")),
        m3_sources_sha256=_sha256(raw["m3_sources_sha256"], "reference.m3_sources_sha256"),
        m5c_config=Path(_as_str(raw["m5c_config"], "reference.m5c_config")),
        m5c_config_sha256=_sha256(raw["m5c_config_sha256"], "reference.m5c_config_sha256"),
        m5c_test_metrics=Path(_as_str(raw["m5c_test_metrics"], "reference.m5c_test_metrics")),
        m5c_test_metrics_sha256=_sha256(
            raw["m5c_test_metrics_sha256"], "reference.m5c_test_metrics_sha256"
        ),
        m5c_test_predictions=Path(
            _as_str(raw["m5c_test_predictions"], "reference.m5c_test_predictions")
        ),
        m5c_test_predictions_sha256=_sha256(
            raw["m5c_test_predictions_sha256"],
            "reference.m5c_test_predictions_sha256",
        ),
        m5c_uncertainty=Path(_as_str(raw["m5c_uncertainty"], "reference.m5c_uncertainty")),
        m5c_uncertainty_sha256=_sha256(
            raw["m5c_uncertainty_sha256"], "reference.m5c_uncertainty_sha256"
        ),
    )


def _parse_expectations(raw: Mapping[str, object]) -> Expectations:
    keys = {
        "n_sources",
        "sources_per_font",
        "n_fonts",
        "rows_per_source",
        "total_rows",
        "clean_rows",
        "original_blur_rows",
        "original_noise_rows",
        "thin_line_rows",
        "arc_stroke_rows",
        "elastic_primary_rows",
        "diagnostic_rows",
    }
    _check_keys(raw, keys, "expectations")
    values = {key: _as_int(raw[key], f"expectations.{key}") for key in keys}
    if any(value < 0 for value in values.values()):
        raise HeldoutFontConfigError("expectations values must be non-negative")
    return Expectations(**values)


def load_config(path: Path | str) -> HeldoutConfig:
    """Load and strictly validate the M7B configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise HeldoutFontConfigError(f"missing M7B configuration: {config_path}")
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise HeldoutFontConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    _check_keys(payload, set(_EXPECTED_TOP_KEYS), "M7B configuration")
    fonts = _parse_fonts(payload.get("fonts"))
    config = HeldoutConfig(
        run=_parse_run(_section(payload, "run")),
        schema=_parse_schema(_section(payload, "schema")),
        dataset=_parse_dataset(_section(payload, "dataset")),
        render=_parse_render(_section(payload, "render")),
        fonts=fonts,
        corruptions=_parse_corruptions(_section(payload, "corruptions")),
        geometry=_parse_geometry(_section(payload, "geometry")),
        protocol=_parse_protocol(_section(payload, "protocol")),
        model=_parse_model(_section(payload, "model")),
        reference=_parse_reference(_section(payload, "reference")),
        expectations=_parse_expectations(_section(payload, "expectations")),
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )
    if len(config.fonts) != config.expectations.n_fonts:
        raise HeldoutFontConfigError(
            f"expected {config.expectations.n_fonts} fonts, got {len(config.fonts)}"
        )
    if config.dataset.sources_per_font != config.expectations.sources_per_font:
        raise HeldoutFontConfigError(
            "dataset.sources_per_font does not match expectations.sources_per_font"
        )
    return config


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise HeldoutFontError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise HeldoutFontError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def _safe_output_path(path: Path | str, label: str) -> Path:
    resolved = Path(path).resolve()
    if resolved == Path.cwd().resolve():
        raise HeldoutFontError(f"refusing to write {label} into the repository root")
    for protected_text in PROTECTED_OUTPUT_ROOTS:
        protected = Path(protected_text).resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise HeldoutFontError(
                f"refusing to write {label} into protected frozen path: {resolved} "
                f"({protected_text})"
            )
    return resolved


def _stable_id(prefix: str, payload: Mapping[str, object]) -> str:
    digest = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return f"{prefix}-{digest[:16]}"


def _amplitude_label(amplitude: float) -> str:
    return f"{amplitude:g}".replace(".", "_")


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _counts(values: Iterable[object]) -> dict[str, int]:
    counter = Counter(str(value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


@dataclass(frozen=True, slots=True)
class SourcePlan:
    index: int
    font: ResolvedFont
    text: str
    text_length: int
    render_seed: int
    source_id: str

    @property
    def text_sha256(self) -> str:
        return sha256_bytes(self.text.encode("utf-8"))


def resolve_heldout_fonts(config: HeldoutConfig) -> tuple[ResolvedFont, ...]:
    """Resolve, hash and glyph-check the declared held-out fonts."""
    resolved: list[ResolvedFont] = []
    for font in config.fonts:
        if not font.path.is_file():
            raise HeldoutFontError(
                f"held-out font {font.key!r} is missing: {font.path}; no substitution is allowed"
            )
        actual = sha256_file(font.path)
        if actual != font.sha256:
            raise HeldoutFontError(
                f"held-out font {font.key!r} hash mismatch: expected {font.sha256}, "
                f"found {actual} ({font.path})"
            )
        if font.sha256 in FROZEN_FONT_HASHES:
            raise HeldoutFontError(
                f"held-out font {font.key!r} has the same file hash as an M3 development font"
            )
        try:
            pil_font = ImageFont.truetype(str(font.path), 32)
        except OSError as exc:
            raise HeldoutFontError(f"cannot load held-out font {font.key!r}: {exc}") from exc
        missing = []
        for character in config.dataset.charset:
            bbox = pil_font.getbbox(character)
            if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                missing.append(character)
        if missing:
            raise HeldoutFontError(
                f"held-out font {font.key!r} does not render charset character(s) {missing}"
            )
        resolved.append(ResolvedFont(key=font.key, path=font.path))
    if len(resolved) != len(config.fonts):
        raise HeldoutFontError("not every declared held-out font resolved")
    return tuple(resolved)


def plan_sources(
    config: HeldoutConfig, resolution: Sequence[ResolvedFont]
) -> tuple[SourcePlan, ...]:
    """Plan deterministic sources in one untouched evaluation split."""
    plans: list[SourcePlan] = []
    seen: set[str] = set()
    weights = np.asarray(config.dataset.text_length_weights, dtype=np.float64)
    for font in resolution:
        for index in range(config.dataset.sources_per_font):
            rng = np.random.default_rng(
                derive_noise_seed(config.run.seed, "m7b-source", font.key, index)
            )
            length = int(
                rng.choice(
                    np.asarray(config.dataset.text_lengths, dtype=np.int64),
                    p=weights / weights.sum(),
                )
            )
            render_config = config.render.to_render_config(
                text_length=length, charset=config.dataset.charset
            )
            text = sample_text(rng, render_config)
            render_seed = derive_noise_seed(config.run.seed, "m7b-render", font.key, index)
            source_id = _stable_id(
                "heldout-src",
                {
                    "generation_config_hash": config.generation_hash,
                    "font_id": font.key,
                    "font_sha256": config.font(font.key).sha256,
                    "index": index,
                    "text": text,
                    "text_length": length,
                    "render_seed": render_seed,
                },
            )
            if source_id in seen:
                raise HeldoutFontError(f"source_id collision for {font.key!r} index {index}")
            seen.add(source_id)
            plans.append(
                SourcePlan(
                    index=index,
                    font=font,
                    text=text,
                    text_length=length,
                    render_seed=render_seed,
                    source_id=source_id,
                )
            )
    return tuple(plans)


def _corruption_parameters(spec: CorruptionSpec) -> dict[str, float]:
    if spec.family == FAMILY_BLUR:
        return {"radius_px": spec.parameter}
    if spec.family == FAMILY_NOISE:
        return {"sigma": spec.parameter}
    raise HeldoutFontError(f"unexpected corruption family {spec.family!r}")


def _target_stats(values: pd.Series) -> dict[str, float | int | bool]:
    array = values.to_numpy(dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "min": math.nan,
            "max": math.nan,
            "q05": math.nan,
            "q25": math.nan,
            "q50": math.nan,
            "q75": math.nan,
            "q95": math.nan,
            "variance": math.nan,
        }
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q05": float(np.percentile(array, 5)),
        "q25": float(np.percentile(array, 25)),
        "q50": float(np.percentile(array, 50)),
        "q75": float(np.percentile(array, 75)),
        "q95": float(np.percentile(array, 95)),
        "variance": float(np.var(array, ddof=0)),
    }


def _manifest_row(
    *,
    config: HeldoutConfig,
    plan: SourcePlan,
    sample_id: str,
    variant_id: str,
    variant_family: str,
    severity_level: int,
    amplitude_px: float | None,
    corruption_parameters: Mapping[str, float],
    variant_seed: int,
    field_seed: int | None,
    field_sha256: str,
    field_payload: str,
    stroke_payload: str,
    image_rel: Path,
    reference_rel: Path,
    image: np.ndarray,
    reference: np.ndarray,
    protocol_scope: str,
) -> dict[str, object]:
    metrics = compute_pair_metrics(reference, image, ("ssim", "psnr"))
    psnr_value = metrics["psnr"]
    psnr_infinite = math.isinf(psnr_value)
    return {
        "sample_id": sample_id,
        "source_id": plan.source_id,
        "split_group_id": plan.source_id,
        "split": "eval",
        "font_id": plan.font.key,
        "font_sha256": config.font(plan.font.key).sha256,
        "text_length": plan.text_length,
        "render_seed": plan.render_seed,
        "variant_id": variant_id,
        "variant_family": variant_family,
        "severity_level": severity_level,
        "amplitude_px": amplitude_px,
        "corruption_parameters": canonical_json(dict(corruption_parameters)),
        "variant_seed": variant_seed,
        "field_seed": field_seed,
        "field_sha256": field_sha256,
        "field_spec_json": field_payload,
        "stroke_geometry_json": stroke_payload,
        "image_path": image_rel.as_posix(),
        "reference_path": reference_rel.as_posix(),
        "ssim_target": float(metrics["ssim"]),
        "psnr_target": None if psnr_infinite else float(psnr_value),
        "psnr_infinite": bool(psnr_infinite),
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "image_sha256": sha256_bytes(image.tobytes()),
        "reference_sha256": sha256_bytes(reference.tobytes()),
        "text_sha256": plan.text_sha256,
        "dataset_version": config.dataset.version,
        "generation_config_hash": config.generation_hash,
        "protocol_scope": protocol_scope,
    }


@dataclass(frozen=True, slots=True)
class DatasetResult:
    root: Path
    config: HeldoutConfig
    manifest: pd.DataFrame
    sources: pd.DataFrame
    features: pd.DataFrame
    summary: dict[str, Any]
    checks: tuple[CheckResult, ...] = ()


def _coerce_manifest_types(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "text_length",
        "render_seed",
        "severity_level",
        "variant_seed",
        "height",
        "width",
    ):
        result[column] = result[column].astype("int64")
    for column in ("amplitude_px", "ssim_target", "psnr_target"):
        result[column] = result[column].astype("float64")
    result["psnr_infinite"] = result["psnr_infinite"].astype(bool)
    return result


def _coerce_source_types(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("text_length", "render_seed", "background_level", "height", "width"):
        result[column] = result[column].astype("int64")
    return result


def _summary_targets(manifest: pd.DataFrame, config: HeldoutConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        subset = manifest[manifest["variant_family"] == family]
        if subset.empty:
            continue
        result[family] = {
            "rows": int(len(subset)),
            "sources": int(subset["source_id"].nunique()),
            "target": _target_stats(subset["ssim_target"]),
            "by_severity": {
                str(int(severity)): _target_stats(group["ssim_target"])
                for severity, group in subset.groupby("severity_level", sort=True)
            },
        }
    return result


def _summary_fonts(
    manifest: pd.DataFrame, sources: pd.DataFrame, config: HeldoutConfig
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for font in config.fonts:
        font_sources = sources[sources["font_id"] == font.key]
        font_rows = manifest[manifest["font_id"] == font.key]
        result[font.key] = {
            "path": str(font.path),
            "sha256": font.sha256,
            "family": font.family,
            "rationale": font.rationale,
            "sources": int(len(font_sources)),
            "rows": int(len(font_rows)),
            "text_lengths": _counts(font_sources["text_length"]),
            "families": _counts(font_rows["variant_family"]),
            "severities": _counts(font_rows["severity_level"]),
        }
    return result


def _build_summary(
    *,
    config: HeldoutConfig,
    manifest: pd.DataFrame,
    sources: pd.DataFrame,
    features: pd.DataFrame,
    image_hashes: Mapping[str, str],
    clean_hashes: Sequence[str],
) -> dict[str, Any]:
    distinct_images = len(set(image_hashes.values()))
    duplicate_fraction = 1.0 - distinct_images / len(image_hashes) if image_hashes else 0.0
    counts = {
        "sources": int(sources["source_id"].nunique()),
        "rows": int(len(manifest)),
        "images": int(len(image_hashes)),
        "distinct_images": int(distinct_images),
        "duplicate_image_fraction": float(duplicate_fraction),
        "clean_sources": int((manifest["variant_family"] == CLEAN_FAMILY).sum()),
        "clean_distinct_hashes": int(len(set(clean_hashes))),
        "fonts": int(sources["font_id"].nunique()),
        "splits": _counts(sources["split"]),
        "families": _counts(manifest["variant_family"]),
        "severities": _counts(manifest["severity_level"]),
        "text_lengths": _counts(sources["text_length"]),
        "background_levels": _counts(sources["background_level"]),
        "features": int(len(features)),
        "feature_names": list(features.columns[1:]),
    }
    summary = {
        "dataset_name": config.dataset.name,
        "dataset_version": config.dataset.version,
        "generation_config_hash": config.generation_hash,
        "config_sha256": config.config_sha256,
        "seed": config.run.seed,
        "split_policy": {
            "names": list(config.schema.split_names),
            "independent_unit": "source_id",
            "all_sources_in_eval": bool((sources["split"] == "eval").all()),
            "source_grouped": bool((sources.groupby("source_id")["split"].nunique() == 1).all()),
        },
        "target": {
            "name": TARGET,
            "definition": "SSIM between each corrupted image and its exact source-specific clean reference",
            "psnr_role": "diagnostic only",
        },
        "counts": counts,
        "fonts": _summary_fonts(manifest, sources, config),
        "targets": _summary_targets(manifest, config),
        "artifacts": {
            "image_pixel_hash_set": sha256_bytes(
                canonical_json(sorted(image_hashes.items())).encode("utf-8")
            ),
            "clean_pixel_hash_set": sha256_bytes(
                canonical_json(sorted(set(clean_hashes))).encode("utf-8")
            ),
        },
    }
    return summary


def generate_dataset(
    config: HeldoutConfig,
    *,
    output_dataset: Path | str | None = None,
    overwrite: bool = False,
) -> DatasetResult:
    """Generate the versioned held-out-font dataset."""
    target = _safe_output_path(output_dataset or config.run.output_dataset, "held-out-font dataset")
    if target.exists() and not overwrite:
        raise HeldoutFontError(
            f"dataset directory already exists: {target}; pass --overwrite or choose another path"
        )
    if target.exists():
        shutil.rmtree(target)
    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        _require_file_hash(
            config.reference.m3_sources,
            config.reference.m3_sources_sha256,
            "M3 sources",
        )
        _require_file_hash(
            config.reference.m3_config,
            config.reference.m3_config_sha256,
            "M3 config",
        )
        frozen_sources = pd.read_csv(config.reference.m3_sources)
        frozen_ids = set(frozen_sources["source_id"].astype(str))
        frozen_fonts = set(frozen_sources["font_id"].astype(str))
        if not frozen_fonts.issubset(FROZEN_FONT_KEYS):
            raise HeldoutFontError("the pinned M3 source table contains unexpected font keys")

        resolution = resolve_heldout_fonts(config)
        plans = plan_sources(config, resolution)
        source_ids = {plan.source_id for plan in plans}
        if source_ids & frozen_ids:
            raise HeldoutFontError("held-out source IDs overlap frozen M3 source IDs")

        manifest_rows: list[dict[str, object]] = []
        source_rows: list[dict[str, object]] = []
        feature_rows: list[tuple[str, np.ndarray]] = []
        image_hashes: dict[str, str] = {}
        clean_hashes: list[str] = []
        blur_ladder = config.corruptions.blur_ladder()
        noise_ladder = config.corruptions.noise_ladder()
        interference_ladders = {
            variant: config.corruptions.interference_ladder(variant)
            for variant in INTERFERENCE_FAMILIES
        }

        for plan in plans:
            render_config = config.render.to_render_config(
                text_length=plan.text_length, charset=config.dataset.charset
            )
            clean = render_clean(
                plan.font,
                plan.text,
                render_config,
                rng=np.random.default_rng(plan.render_seed),
            )
            clean_rel = Path("images") / "clean" / f"{plan.source_id}.png"
            _write_png(staging / clean_rel, clean.image)
            image_hashes[clean_rel.as_posix()] = clean.sha256
            clean_hashes.append(clean.sha256)

            source_rows.append(
                {
                    "source_id": plan.source_id,
                    "split_group_id": plan.source_id,
                    "split": "eval",
                    "font_id": plan.font.key,
                    "font_path": str(config.font(plan.font.key).path),
                    "font_sha256": config.font(plan.font.key).sha256,
                    "text_length": plan.text_length,
                    "render_seed": plan.render_seed,
                    "background_level": clean.background_level,
                    "height": int(clean.image.shape[0]),
                    "width": int(clean.image.shape[1]),
                    "clean_image_path": clean_rel.as_posix(),
                    "clean_image_sha256": clean.sha256,
                    "text_sha256": plan.text_sha256,
                    "dataset_version": config.dataset.version,
                    "generation_config_hash": config.generation_hash,
                }
            )

            clean_sample_id = f"{plan.source_id}__clean_s0"
            manifest_rows.append(
                _manifest_row(
                    config=config,
                    plan=plan,
                    sample_id=clean_sample_id,
                    variant_id="clean_s0",
                    variant_family=CLEAN_FAMILY,
                    severity_level=0,
                    amplitude_px=None,
                    corruption_parameters={},
                    variant_seed=derive_noise_seed(
                        config.run.seed, "m7b-variant", plan.source_id, CLEAN_FAMILY, 0
                    ),
                    field_seed=None,
                    field_sha256="",
                    field_payload="",
                    stroke_payload="",
                    image_rel=clean_rel,
                    reference_rel=clean_rel,
                    image=clean.image,
                    reference=clean.image,
                    protocol_scope="diagnostic",
                )
            )
            feature_rows.append((clean_sample_id, clean.image))

            for ladder in (blur_ladder, noise_ladder):
                for severity in range(1, ladder.n_levels):
                    variant_seed = derive_noise_seed(
                        config.run.seed,
                        "m7b-variant",
                        plan.source_id,
                        ladder.family,
                        severity,
                    )
                    result = apply_corruption(
                        clean.image, ladder, severity, noise_seed=variant_seed
                    )
                    sample_id = f"{plan.source_id}__{ladder.family}_s{severity}"
                    image_rel = Path("images") / ladder.family / f"{sample_id}.png"
                    _write_png(staging / image_rel, result.image)
                    image_hashes[image_rel.as_posix()] = sha256_bytes(result.image.tobytes())
                    manifest_rows.append(
                        _manifest_row(
                            config=config,
                            plan=plan,
                            sample_id=sample_id,
                            variant_id=f"{ladder.family}_s{severity}",
                            variant_family=ladder.family,
                            severity_level=severity,
                            amplitude_px=None,
                            corruption_parameters=_corruption_parameters(result.spec),
                            variant_seed=variant_seed,
                            field_seed=None,
                            field_sha256="",
                            field_payload="",
                            stroke_payload="",
                            image_rel=image_rel,
                            reference_rel=clean_rel,
                            image=result.image,
                            reference=clean.image,
                            protocol_scope="primary",
                        )
                    )
                    feature_rows.append((sample_id, result.image))

            for variant, ladder in interference_ladders.items():
                stroke_seed = derive_noise_seed(
                    config.run.seed, "m7b-interference", plan.source_id, variant
                )
                pool = plan_stroke_pool(clean.image, ladder, seed=stroke_seed)
                for severity in range(1, ladder.n_levels + 1):
                    spec = ladder.spec(severity)
                    strokes = pool[: spec.stroke_count]
                    corrupted = apply_strokes(
                        clean.image,
                        list(strokes),
                        width_px=spec.width_px,
                        intensity=spec.intensity,
                    )
                    sample_id = f"{plan.source_id}__{variant}_s{severity}"
                    image_rel = Path("images") / variant / f"{sample_id}.png"
                    _write_png(staging / image_rel, corrupted)
                    image_hashes[image_rel.as_posix()] = sha256_bytes(corrupted.tobytes())
                    manifest_rows.append(
                        _manifest_row(
                            config=config,
                            plan=plan,
                            sample_id=sample_id,
                            variant_id=f"{variant}_s{severity}",
                            variant_family=variant,
                            severity_level=severity,
                            amplitude_px=None,
                            corruption_parameters={
                                "stroke_count": float(spec.stroke_count),
                                "stroke_width_px": float(spec.width_px),
                                "stroke_intensity": float(spec.intensity),
                            },
                            variant_seed=stroke_seed,
                            field_seed=None,
                            field_sha256="",
                            field_payload="",
                            stroke_payload=canonical_json(geometry_to_json(list(strokes))),
                            image_rel=image_rel,
                            reference_rel=clean_rel,
                            image=corrupted,
                            reference=clean.image,
                            protocol_scope="primary",
                        )
                    )
                    feature_rows.append((sample_id, corrupted))

            field_seed = derive_noise_seed(config.run.seed, "m7b-elastic", plan.source_id)
            field_spec = config.geometry.field_spec(clean.image.shape, field_seed)
            field = generate_displacement_field(field_spec)
            field_hash = sha256_bytes(np.ascontiguousarray(field).tobytes())
            field_payload = field_spec_json(field_spec, field)
            for amplitude in config.corruptions.elastic_amplitudes():
                severity = (
                    int(round(amplitude * 2.0)) if amplitude < 1.0 else int(round(amplitude)) + 1
                )
                if amplitude in config.corruptions.elastic_diagnostic_amplitudes_px:
                    protocol_scope = "diagnostic"
                else:
                    protocol_scope = "primary"
                result = apply_elastic(clean.image, field_spec=field_spec, amplitude_px=amplitude)
                variant_id = f"elastic_{_amplitude_label(amplitude)}px"
                sample_id = f"{plan.source_id}__{variant_id}"
                image_rel = Path("images") / ELASTIC_FAMILY / f"{sample_id}.png"
                _write_png(staging / image_rel, result.image)
                image_hashes[image_rel.as_posix()] = sha256_bytes(result.image.tobytes())
                manifest_rows.append(
                    _manifest_row(
                        config=config,
                        plan=plan,
                        sample_id=sample_id,
                        variant_id=variant_id,
                        variant_family=ELASTIC_FAMILY,
                        severity_level=severity,
                        amplitude_px=float(amplitude),
                        corruption_parameters={"amplitude_px": float(amplitude)},
                        variant_seed=derive_noise_seed(
                            config.run.seed,
                            "m7b-variant",
                            plan.source_id,
                            ELASTIC_FAMILY,
                            severity,
                        ),
                        field_seed=field_seed,
                        field_sha256=field_hash,
                        field_payload=field_payload,
                        stroke_payload="",
                        image_rel=image_rel,
                        reference_rel=clean_rel,
                        image=result.image,
                        reference=clean.image,
                        protocol_scope=protocol_scope,
                    )
                )
                feature_rows.append((sample_id, result.image))

        manifest = _coerce_manifest_types(pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS))
        sources = _coerce_source_types(pd.DataFrame(source_rows, columns=SOURCE_COLUMNS))
        features = extract_feature_table(feature_rows)
        if list(features.columns) != list(FEATURE_COLUMNS):
            raise HeldoutFontError(
                f"feature extractor returned {list(features.columns)}, expected {list(FEATURE_COLUMNS)}"
            )
        summary = _build_summary(
            config=config,
            manifest=manifest,
            sources=sources,
            features=features,
            image_hashes=image_hashes,
            clean_hashes=clean_hashes,
        )
        _write_csv(manifest, staging / "manifest.csv")
        _write_csv(sources, staging / "sources.csv")
        _write_csv(features, staging / "features.csv")
        _write_json(summary, staging / "dataset_summary.json")

        checks = dataset_checks(config, manifest, sources, features, summary)
        _require_all_checks(checks, "generated held-out-font dataset")
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return DatasetResult(
        root=target,
        config=config,
        manifest=manifest,
        sources=sources,
        features=features,
        summary=summary,
        checks=checks,
    )


def _require_all_checks(checks: Sequence[CheckResult], label: str) -> None:
    failures = [check for check in checks if not check.passed]
    if failures:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failures)
        raise HeldoutFontError(f"{label} failed: {detail}")


def dataset_checks(
    config: HeldoutConfig,
    manifest: pd.DataFrame,
    sources: pd.DataFrame,
    features: pd.DataFrame,
    summary: Mapping[str, Any],
) -> tuple[CheckResult, ...]:
    """Return the predeclared dataset count and isolation checks."""
    checks: list[CheckResult] = []
    expected = config.expectations
    counts = summary.get("counts", {})
    observed = {
        "n_sources": int(sources["source_id"].nunique()),
        "sources_per_font": int(sources.groupby("font_id")["source_id"].nunique().min()),
        "n_fonts": int(sources["font_id"].nunique()),
        "total_rows": int(len(manifest)),
        "clean_rows": int((manifest["variant_family"] == CLEAN_FAMILY).sum()),
        "original_blur_rows": int((manifest["variant_family"] == FAMILY_BLUR).sum()),
        "original_noise_rows": int((manifest["variant_family"] == FAMILY_NOISE).sum()),
        "thin_line_rows": int((manifest["variant_family"] == VARIANT_THIN_LINES).sum()),
        "arc_stroke_rows": int((manifest["variant_family"] == VARIANT_ARC_STROKES).sum()),
        "elastic_primary_rows": int(
            (
                (manifest["variant_family"] == ELASTIC_FAMILY)
                & (manifest["protocol_scope"] == "primary")
            ).sum()
        ),
        "diagnostic_rows": int((manifest["protocol_scope"] == "diagnostic").sum()),
    }
    expected_values = {
        "n_sources": expected.n_sources,
        "sources_per_font": expected.sources_per_font,
        "n_fonts": expected.n_fonts,
        "total_rows": expected.total_rows,
        "clean_rows": expected.clean_rows,
        "original_blur_rows": expected.original_blur_rows,
        "original_noise_rows": expected.original_noise_rows,
        "thin_line_rows": expected.thin_line_rows,
        "arc_stroke_rows": expected.arc_stroke_rows,
        "elastic_primary_rows": expected.elastic_primary_rows,
        "diagnostic_rows": expected.diagnostic_rows,
    }
    for name, expected_value in expected_values.items():
        actual = observed[name]
        checks.append(
            CheckResult(
                name=f"count_{name}",
                passed=actual == expected_value,
                detail=f"observed={actual}, expected={expected_value}",
            )
        )
    checks.append(
        CheckResult(
            name="source_grouping",
            passed=bool((sources.groupby("source_id")["split"].nunique() == 1).all()),
            detail="each source appears in exactly one split",
        )
    )
    checks.append(
        CheckResult(
            name="single_eval_split",
            passed=bool((sources["split"] == "eval").all()),
            detail=f"splits={_counts(sources['split'])}",
        )
    )
    checks.append(
        CheckResult(
            name="clean_reference_identity",
            passed=bool(
                (
                    manifest.loc[manifest["variant_family"] == CLEAN_FAMILY, "image_path"]
                    == manifest.loc[manifest["variant_family"] == CLEAN_FAMILY, "reference_path"]
                ).all()
            ),
            detail="every clean row references its own clean image",
        )
    )
    clean_hashes = manifest.loc[manifest["variant_family"] == CLEAN_FAMILY, "image_sha256"].tolist()
    checks.append(
        CheckResult(
            name="clean_images_unique",
            passed=len(clean_hashes) == len(set(clean_hashes)),
            detail=f"{len(clean_hashes)} clean rows, {len(set(clean_hashes))} distinct hashes",
        )
    )
    checks.append(
        CheckResult(
            name="manifest_sample_ids_unique",
            passed=manifest["sample_id"].is_unique,
            detail=f"{manifest['sample_id'].nunique()} distinct sample IDs for {len(manifest)} rows",
        )
    )
    checks.append(
        CheckResult(
            name="feature_schema",
            passed=list(features.columns) == list(FEATURE_COLUMNS),
            detail=f"columns={list(features.columns)}",
        )
    )
    checks.append(
        CheckResult(
            name="feature_sample_ids_unique",
            passed=features["sample_id"].is_unique,
            detail=f"{features['sample_id'].nunique()} distinct feature sample IDs",
        )
    )
    checks.append(
        CheckResult(
            name="feature_count",
            passed=len(features) == len(manifest),
            detail=f"features={len(features)}, rows={len(manifest)}",
        )
    )
    checks.append(
        CheckResult(
            name="feature_sample_ids_match",
            passed=set(features["sample_id"]) == set(manifest["sample_id"]),
            detail="feature sample IDs exactly match manifest sample IDs",
        )
    )
    checks.append(
        CheckResult(
            name="source_ids_unique",
            passed=sources["source_id"].is_unique,
            detail=f"{sources['source_id'].nunique()} distinct source IDs",
        )
    )
    checks.append(
        CheckResult(
            name="targets_finite_bounded",
            passed=bool(
                np.isfinite(manifest["ssim_target"]).all()
                and (manifest["ssim_target"] >= 0.0).all()
                and (manifest["ssim_target"] <= 1.0).all()
            ),
            detail=f"target range=[{manifest['ssim_target'].min():.6f}, "
            f"{manifest['ssim_target'].max():.6f}]",
        )
    )
    checks.append(
        CheckResult(
            name="font_identity",
            passed=all(
                set(manifest.loc[manifest["font_id"] == font.key, "font_sha256"]) == {font.sha256}
                for font in config.fonts
            ),
            detail="every row records the pinned font file hash",
        )
    )
    checks.append(
        CheckResult(
            name="duplicate_image_fraction",
            passed=float(counts.get("duplicate_image_fraction", 1.0)) == 0.0,
            detail=f"duplicate_image_fraction={counts.get('duplicate_image_fraction')}",
        )
    )
    return tuple(checks)


def _read_csv_exact(path: Path, columns: Sequence[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise HeldoutFontError(f"missing {label}: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != list(columns):
        raise HeldoutFontError(
            f"{label} columns differ: expected {list(columns)}, got {list(frame.columns)}"
        )
    return frame


def _verify_reference_inputs(config: HeldoutConfig) -> tuple[CheckResult, ...]:
    checks: list[CheckResult] = []
    for path, expected, label in (
        (config.reference.m3_config, config.reference.m3_config_sha256, "M3 config"),
        (config.reference.m3_sources, config.reference.m3_sources_sha256, "M3 sources"),
        (config.reference.m5c_config, config.reference.m5c_config_sha256, "M5C config"),
        (
            config.reference.m5c_test_metrics,
            config.reference.m5c_test_metrics_sha256,
            "M5C test metrics",
        ),
        (
            config.reference.m5c_test_predictions,
            config.reference.m5c_test_predictions_sha256,
            "M5C test predictions",
        ),
        (
            config.reference.m5c_uncertainty,
            config.reference.m5c_uncertainty_sha256,
            "M5C uncertainty table",
        ),
    ):
        try:
            _require_file_hash(path, expected, label)
            checks.append(CheckResult(f"reference:{label}", True, f"sha256={expected}"))
        except HeldoutFontError as exc:
            checks.append(CheckResult(f"reference:{label}", False, str(exc)))
    return tuple(checks)


def _verify_dataset_content(dataset: DatasetResult) -> tuple[CheckResult, ...]:
    """Recompute image hashes, targets and features from the written PNGs."""
    checks: list[CheckResult] = []
    root = dataset.root
    reference_cache: dict[str, np.ndarray] = {}
    image_rows: list[tuple[str, np.ndarray]] = []
    target_errors: list[float] = []
    psnr_errors: list[float] = []
    hash_errors: list[str] = []
    shape_errors: list[str] = []

    source_clean: dict[str, tuple[Path, str]] = {
        str(row.source_id): (root / str(row.clean_image_path), str(row.clean_image_sha256))
        for row in dataset.sources.itertuples(index=False)
    }

    for row in dataset.manifest.itertuples(index=False):
        image_path = root / str(row.image_path)
        reference_path = root / str(row.reference_path)
        try:
            image = load_grayscale(image_path)
        except Exception as exc:  # noqa: BLE001 - report the exact file and error
            shape_errors.append(f"{row.sample_id}: cannot load image: {exc}")
            continue
        try:
            reference = reference_cache.get(str(row.reference_path))
            if reference is None:
                reference = load_grayscale(reference_path)
                reference_cache[str(row.reference_path)] = reference
        except Exception as exc:  # noqa: BLE001 - report the exact file and error
            shape_errors.append(f"{row.sample_id}: cannot load reference: {exc}")
            continue
        if image.shape != (int(row.height), int(row.width)):
            shape_errors.append(
                f"{row.sample_id}: image shape {image.shape} != {(int(row.height), int(row.width))}"
            )
        if reference.shape != image.shape:
            shape_errors.append(
                f"{row.sample_id}: reference shape {reference.shape} != image shape {image.shape}"
            )
        image_hash = sha256_bytes(image.tobytes())
        reference_hash = sha256_bytes(reference.tobytes())
        if image_hash != str(row.image_sha256):
            hash_errors.append(f"{row.sample_id}: image hash mismatch")
        if reference_hash != str(row.reference_sha256):
            hash_errors.append(f"{row.sample_id}: reference hash mismatch")
        source_path, source_hash = source_clean[str(row.source_id)]
        if reference_path.resolve() != source_path.resolve():
            hash_errors.append(f"{row.sample_id}: reference path is not the source clean image")
        if reference_hash != source_hash:
            hash_errors.append(f"{row.sample_id}: reference hash is not the source clean hash")
        metrics = compute_pair_metrics(reference, image, ("ssim", "psnr"))
        target_errors.append(abs(float(metrics["ssim"]) - float(row.ssim_target)))
        expected_psnr = float(row.psnr_target) if not bool(row.psnr_infinite) else math.inf
        actual_psnr = float(metrics["psnr"])
        if math.isinf(expected_psnr) and math.isinf(actual_psnr):
            psnr_errors.append(0.0)
        elif math.isfinite(expected_psnr) and math.isfinite(actual_psnr):
            psnr_errors.append(abs(actual_psnr - expected_psnr))
        else:
            psnr_errors.append(math.inf)
        image_rows.append((str(row.sample_id), image))

    checks.append(
        CheckResult(
            "content:image_files",
            not shape_errors,
            f"{len(shape_errors)} image/reference problem(s)"
            + (f"; first={shape_errors[0]}" if shape_errors else ""),
        )
    )
    checks.append(
        CheckResult(
            "content:image_hashes",
            not hash_errors,
            f"{len(hash_errors)} hash/reference problem(s)"
            + (f"; first={hash_errors[0]}" if hash_errors else ""),
        )
    )
    checks.append(
        CheckResult(
            "content:ssim_recomputed",
            bool(target_errors) and max(target_errors, default=math.inf) <= 1e-9,
            f"max |recomputed SSIM - stored target|={max(target_errors, default=math.inf):.3e}",
        )
    )
    checks.append(
        CheckResult(
            "content:psnr_recomputed",
            bool(psnr_errors) and max(psnr_errors, default=math.inf) <= 1e-8,
            f"max |recomputed PSNR - stored diagnostic|={max(psnr_errors, default=math.inf):.3e}",
        )
    )
    if image_rows:
        recomputed = extract_feature_table(image_rows)
        expected = dataset.features.set_index("sample_id").loc[
            [sample_id for sample_id, _ in image_rows]
        ]
        if list(recomputed.columns) != list(FEATURE_COLUMNS):
            checks.append(
                CheckResult(
                    "content:features_recomputed",
                    False,
                    f"recomputed feature columns differ: {list(recomputed.columns)}",
                )
            )
        else:
            actual_values = recomputed[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
            expected_values = expected[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
            max_feature_error = float(np.max(np.abs(actual_values - expected_values)))
            checks.append(
                CheckResult(
                    "content:features_recomputed",
                    max_feature_error <= 1e-9,
                    f"max |recomputed feature - stored feature|={max_feature_error:.3e}",
                )
            )
    else:
        checks.append(
            CheckResult("content:features_recomputed", False, "no images available to recompute")
        )
    return tuple(checks)


def load_dataset(
    config: HeldoutConfig,
    *,
    dataset_root: Path | str | None = None,
    verify_images: bool = False,
) -> DatasetResult:
    """Load and validate a generated M7B dataset."""
    root = Path(dataset_root or config.run.output_dataset).resolve()
    if not root.is_dir():
        raise HeldoutFontError(
            f"held-out-font dataset not found: {root}; run the generate/run command first"
        )
    manifest = _read_csv_exact(root / "manifest.csv", MANIFEST_COLUMNS, "M7B manifest")
    sources = _read_csv_exact(root / "sources.csv", SOURCE_COLUMNS, "M7B sources")
    features = _read_csv_exact(root / "features.csv", FEATURE_COLUMNS, "M7B features")
    summary_path = root / "dataset_summary.json"
    if not summary_path.is_file():
        raise HeldoutFontError(f"missing M7B dataset summary: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HeldoutFontError(f"invalid M7B dataset summary: {exc}") from exc
    if summary.get("dataset_version") != config.dataset.version:
        raise HeldoutFontError(
            f"dataset version mismatch: {summary.get('dataset_version')!r} "
            f"!= {config.dataset.version!r}"
        )
    if summary.get("generation_config_hash") != config.generation_hash:
        raise HeldoutFontError("dataset generation hash does not match the configuration")
    if summary.get("config_sha256") != config.config_sha256:
        raise HeldoutFontError("dataset config SHA-256 does not match the current configuration")

    checks = list(dataset_checks(config, manifest, sources, features, summary))
    if verify_images:
        checks.extend(
            _verify_dataset_content(
                DatasetResult(
                    root=root,
                    config=config,
                    manifest=manifest,
                    sources=sources,
                    features=features,
                    summary=summary,
                )
            )
        )
    _require_all_checks(checks, "loaded held-out-font dataset")
    return DatasetResult(
        root=root,
        config=config,
        manifest=manifest,
        sources=sources,
        features=features,
        summary=summary,
        checks=tuple(checks),
    )


def _load_champion(
    config: HeldoutConfig,
) -> tuple[Any, dict[str, Any], tuple[CheckResult, ...]]:
    artifact = config.model.model_dir / f"{config.model.model_stem}.joblib"
    card_path = config.model.model_dir / f"{config.model.model_stem}_card.json"
    checks: list[CheckResult] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append(CheckResult(name, passed, detail))

    try:
        _require_file_hash(artifact, config.model.artifact_sha256, "M5C champion artifact")
        record("model:artifact_hash", True, f"sha256={config.model.artifact_sha256}")
    except HeldoutFontError as exc:
        record("model:artifact_hash", False, str(exc))
    try:
        _require_file_hash(card_path, config.model.card_sha256, "M5C champion card")
        record("model:card_hash", True, f"sha256={config.model.card_sha256}")
    except HeldoutFontError as exc:
        record("model:card_hash", False, str(exc))

    card: dict[str, Any] = {}
    if card_path.is_file():
        try:
            loaded = json.loads(card_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("model card must be a JSON object")
            card = loaded
            record("model:card_parse", True, f"loaded {card_path.name}")
        except Exception as exc:  # noqa: BLE001 - report the exact card error
            record("model:card_parse", False, f"{type(exc).__name__}: {exc}")
    else:
        record("model:card_parse", False, f"missing {card_path}")

    def check_card() -> None:
        if card.get("artifact_file") != artifact.name:
            record(
                "model:artifact_file",
                False,
                f"card names {card.get('artifact_file')!r}, expected {artifact.name!r}",
            )
        else:
            record("model:artifact_file", True, artifact.name)
        if card.get("artifact_sha256") != config.model.artifact_sha256:
            record("model:card_artifact_hash", False, "card artifact hash does not match config")
        else:
            record("model:card_artifact_hash", True, "card artifact hash matches config")
        if card.get("feature_names") != list(FEATURE_NAMES):
            record(
                "model:feature_schema",
                False,
                f"card feature names {card.get('feature_names')} != {list(FEATURE_NAMES)}",
            )
        elif int(card.get("n_features", -1)) != len(FEATURE_NAMES):
            record("model:feature_schema", False, "card n_features does not match the extractor")
        elif card.get("feature_set_version") != config.schema.feature_set_version:
            record(
                "model:feature_schema",
                False,
                f"card feature set {card.get('feature_set_version')!r} != "
                f"{config.schema.feature_set_version!r}",
            )
        else:
            record("model:feature_schema", True, "feature names, order and version match")
        if card.get("target") != TARGET:
            record("model:target", False, f"card target {card.get('target')!r} != {TARGET!r}")
        else:
            record("model:target", True, TARGET)
        if card.get("calibration") != config.model.expected_calibration:
            record(
                "model:calibration",
                False,
                f"card calibration {card.get('calibration')!r} != "
                f"{config.model.expected_calibration!r}",
            )
        else:
            record("model:calibration", True, config.model.expected_calibration)
        if card.get("model_family") != config.model.expected_model_family:
            record(
                "model:family",
                False,
                f"card model family {card.get('model_family')!r} != "
                f"{config.model.expected_model_family!r}",
            )
        else:
            record("model:family", True, config.model.expected_model_family)
        recorded = str(card.get("sklearn_version", ""))
        current = sklearn_version()
        if recorded.split(".")[:2] != current.split(".")[:2]:
            record(
                "model:sklearn_compatibility",
                False,
                f"persisted with scikit-learn {recorded}, current is {current}",
            )
        else:
            record(
                "model:sklearn_compatibility",
                True,
                f"scikit-learn {current} (major.minor matches {recorded})",
            )

    if card:
        check_card()
    else:
        record("model:card_schema", False, "model card could not be parsed")

    if not all(check.passed for check in checks):
        return None, card, tuple(checks)

    try:
        estimator = joblib.load(artifact)
    except Exception as exc:  # noqa: BLE001 - report the exact load error
        record("model:load", False, f"{type(exc).__name__}: {exc}")
        return None, card, tuple(checks)
    record("model:load", True, type(estimator).__name__)
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features != len(FEATURE_NAMES):
        record(
            "model:n_features",
            False,
            f"estimator expects {n_features} inputs, extractor provides {len(FEATURE_NAMES)}",
        )
        return None, card, tuple(checks)
    record("model:n_features", True, f"{n_features} inputs")
    return estimator, card, tuple(checks)


def _eval_frame(dataset: DatasetResult) -> pd.DataFrame:
    frame = dataset.manifest.merge(
        dataset.features, on="sample_id", how="left", validate="one_to_one"
    )
    if len(frame) != len(dataset.manifest):
        raise HeldoutFontError("merging features changed the row count")
    if frame[list(FEATURE_NAMES)].isna().any().any():
        raise HeldoutFontError("feature merge left missing values")
    if list(frame.columns[: len(MANIFEST_COLUMNS)]) != list(MANIFEST_COLUMNS):
        raise HeldoutFontError("evaluation frame lost its manifest column order")
    return frame.reset_index(drop=True)


def _scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    family = frame["variant_family"]
    amplitude = frame["amplitude_px"]
    if scope == "all_primary":
        return frame["protocol_scope"] == "primary"
    if scope == "clean_reference":
        return family == CLEAN_FAMILY
    if scope == "original_blur_noise":
        return family.isin([FAMILY_BLUR, FAMILY_NOISE])
    if scope == "gaussian_blur":
        return family == FAMILY_BLUR
    if scope == "gaussian_noise":
        return family == FAMILY_NOISE
    if scope == "interference":
        return family.isin(list(INTERFERENCE_FAMILIES))
    if scope == "thin_lines":
        return family == VARIANT_THIN_LINES
    if scope == "arc_strokes":
        return family == VARIANT_ARC_STROKES
    if scope == "geometric_primary":
        return (family == ELASTIC_FAMILY) & (amplitude > 0.5)
    if scope == "geometric_primary_1px":
        return (family == ELASTIC_FAMILY) & (amplitude == 1.0)
    if scope == "geometric_primary_2px":
        return (family == ELASTIC_FAMILY) & (amplitude == 2.0)
    if scope == "geometric_primary_3px":
        return (family == ELASTIC_FAMILY) & (amplitude == 3.0)
    if scope == "diagnostic_0_5px":
        return (family == ELASTIC_FAMILY) & (amplitude == 0.5)
    raise HeldoutFontError(f"unknown evaluation scope {scope!r}")


def _metric_values(
    subset: pd.DataFrame, predicted: np.ndarray, *, near_degenerate_sd: float
) -> dict[str, Any]:
    target = subset["ssim_target"].to_numpy(dtype=np.float64)
    metrics = regression_metrics(target, predicted)
    stats = _target_stats(subset["ssim_target"])
    near_degenerate = bool(int(stats["n"]) > 0 and float(stats["sd"]) < float(near_degenerate_sd))
    r2_defined = bool(metrics["r2_defined"]) and not near_degenerate
    residual = target - predicted
    return {
        "n_rows": int(len(subset)),
        "n_sources": int(subset["source_id"].nunique()),
        "n_distinct_images": int(subset["image_sha256"].nunique()),
        "target_mean": float(stats["mean"]),
        "target_sd": float(stats["sd"]),
        "target_min": float(stats["min"]),
        "target_max": float(stats["max"]),
        "target_q05": float(stats["q05"]),
        "target_q50": float(stats["q50"]),
        "target_q95": float(stats["q95"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2_raw": float(metrics["r2"]),
        "r2": float(metrics["r2"]) if r2_defined else math.nan,
        "r2_defined": r2_defined,
        "r2_suppressed": bool(metrics["r2_defined"] and near_degenerate),
        "near_degenerate": near_degenerate,
        "mean_residual": float(np.mean(residual)) if residual.size else math.nan,
        "sd_residual": float(np.std(residual, ddof=0)) if residual.size else math.nan,
    }


def _metrics_table(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    config: HeldoutConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    font_ids = ("all", *config.font_keys)
    for scope in SCOPE_ORDER:
        mask = _scope_mask(frame, scope)
        subset = frame.loc[mask]
        pred = predicted[mask.to_numpy()]
        for font_id in font_ids:
            if font_id == "all":
                chosen = subset
                chosen_pred = pred
            else:
                chosen_mask = subset["font_id"] == font_id
                chosen = subset.loc[chosen_mask]
                chosen_pred = pred[chosen_mask.to_numpy()]
            values = _metric_values(
                chosen, chosen_pred, near_degenerate_sd=config.protocol.near_degenerate_sd
            )
            values.update(
                {
                    "scope": scope,
                    "font_id": font_id,
                    "protocol_role": (
                        "diagnostic" if scope in config.protocol.diagnostic_scopes else "primary"
                    ),
                }
            )
            rows.append(values)
    return pd.DataFrame(rows)


def _target_distributions(frame: pd.DataFrame, config: HeldoutConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        subset = frame.loc[_scope_mask(frame, scope)]
        for font_id in ("all", *config.font_keys):
            chosen = subset if font_id == "all" else subset[subset["font_id"] == font_id]
            stats = _target_stats(chosen["ssim_target"])
            rows.append(
                {
                    "scope": scope,
                    "font_id": font_id,
                    "protocol_role": (
                        "diagnostic" if scope in config.protocol.diagnostic_scopes else "primary"
                    ),
                    "n_rows": int(len(chosen)),
                    "n_sources": int(chosen["source_id"].nunique()),
                    "n_distinct_images": int(chosen["image_sha256"].nunique()),
                    "target_mean": float(stats["mean"]),
                    "target_sd": float(stats["sd"]),
                    "target_min": float(stats["min"]),
                    "target_max": float(stats["max"]),
                    "target_q05": float(stats["q05"]),
                    "target_q25": float(stats["q25"]),
                    "target_q50": float(stats["q50"]),
                    "target_q75": float(stats["q75"]),
                    "target_q95": float(stats["q95"]),
                    "near_degenerate": bool(
                        int(stats["n"]) > 0
                        and float(stats["sd"]) < config.protocol.near_degenerate_sd
                    ),
                }
            )
    return pd.DataFrame(rows)


def _family_metrics(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    config: HeldoutConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_mask = frame["variant_family"] == family
        family_subset = frame.loc[family_mask]
        family_pred = predicted[family_mask.to_numpy()]
        severity_values = (
            [0]
            if family == CLEAN_FAMILY
            else sorted(int(value) for value in family_subset["severity_level"].unique())
        )
        groups: list[tuple[str, pd.Series, np.ndarray]] = [("all", family_mask, family_pred)]
        for severity in severity_values:
            mask = family_mask & (frame["severity_level"] == severity)
            groups.append((str(severity), mask, predicted[mask.to_numpy()]))
        for severity_label, mask, pred in groups:
            subset = frame.loc[mask]
            for font_id in ("all", *config.font_keys):
                if font_id == "all":
                    chosen = subset
                    chosen_pred = pred
                else:
                    chosen_mask = subset["font_id"] == font_id
                    chosen = subset.loc[chosen_mask]
                    chosen_pred = pred[chosen_mask.to_numpy()]
                values = _metric_values(
                    chosen,
                    chosen_pred,
                    near_degenerate_sd=config.protocol.near_degenerate_sd,
                )
                if family == CLEAN_FAMILY or (family == ELASTIC_FAMILY and severity_label == "1"):
                    role = "diagnostic"
                elif severity_label == "all":
                    role = "mixed" if family == ELASTIC_FAMILY else "primary"
                else:
                    role = "primary"
                values.update(
                    {
                        "family": family,
                        "severity": severity_label,
                        "font_id": font_id,
                        "protocol_role": role,
                    }
                )
                rows.append(values)
    return pd.DataFrame(rows)


def _source_metrics(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    config: HeldoutConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = ("all_primary", "original_blur_noise", "interference", "geometric_primary")
    for source_id in sorted(frame["source_id"].unique()):
        source_mask = frame["source_id"] == source_id
        source = frame.loc[source_mask]
        source_pred = predicted[source_mask.to_numpy()]
        for scope in scopes:
            mask = _scope_mask(source, scope)
            subset = source.loc[mask]
            pred = source_pred[mask.to_numpy()]
            values = _metric_values(
                subset, pred, near_degenerate_sd=config.protocol.near_degenerate_sd
            )
            values.update(
                {
                    "source_id": source_id,
                    "font_id": str(source["font_id"].iloc[0]),
                    "scope": scope,
                    "protocol_role": (
                        "diagnostic" if scope in config.protocol.diagnostic_scopes else "primary"
                    ),
                }
            )
            rows.append(values)
    return pd.DataFrame(rows)


def _bootstrap_table(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    config: HeldoutConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in config.protocol.bootstrap_scopes:
        mask = _scope_mask(frame, scope)
        subset = frame.loc[mask]
        pred = predicted[mask.to_numpy()]
        if subset.empty:
            continue
        overall_label = scope
        table = bootstrap_metrics(
            subset,
            pred,
            n_bootstrap=config.run.n_bootstrap,
            seed=derive_noise_seed(config.run.seed, "m7b-bootstrap", scope),
            scope=overall_label,
        )
        for record in table.to_dict("records"):
            record["scope"] = scope
            record["font_id"] = "all"
            record["bootstrap_scope"] = overall_label
            rows.append(record)
        if scope not in config.protocol.per_font_bootstrap_scopes:
            continue
        for font_id in config.font_keys:
            font_mask = subset["font_id"] == font_id
            font_subset = subset.loc[font_mask]
            font_pred = pred[font_mask.to_numpy()]
            if font_subset.empty:
                continue
            label = f"{scope}__{font_id}"
            table = bootstrap_metrics(
                font_subset,
                font_pred,
                n_bootstrap=config.run.n_bootstrap,
                seed=derive_noise_seed(config.run.seed, "m7b-bootstrap", label),
                scope=label,
            )
            for record in table.to_dict("records"):
                record["scope"] = scope
                record["font_id"] = font_id
                record["bootstrap_scope"] = label
                rows.append(record)
    return pd.DataFrame(rows)


_COMPARISON_MAP: Final[dict[str, tuple[str, str]]] = {
    "original_blur_noise": ("original_blur_noise", "original_blur_noise"),
    "gaussian_blur": (
        "original_blur_noise_gaussian_blur",
        "original_blur_noise_gaussian_blur",
    ),
    "gaussian_noise": (
        "original_blur_noise_gaussian_noise",
        "original_blur_noise_gaussian_noise",
    ),
    "geometric_primary": ("geometric_primary", "geometric_primary"),
    "geometric_primary_1px": (
        "geometric_primary_severity_2",
        "geometric_primary_severity_2",
    ),
    "geometric_primary_2px": (
        "geometric_primary_severity_3",
        "geometric_primary_severity_3",
    ),
    "geometric_primary_3px": (
        "geometric_primary_severity_4",
        "geometric_primary_severity_4",
    ),
    "clean_reference": ("clean_reference", "clean_reference"),
    "diagnostic_0_5px": ("diagnostic_0_5", "diagnostic_0_5"),
}


def _comparison_table(
    config: HeldoutConfig,
    metrics: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    familiar = pd.read_csv(config.reference.m5c_test_metrics)
    familiar_predictions = pd.read_csv(config.reference.m5c_test_predictions)
    familiar_uncertainty = pd.read_csv(config.reference.m5c_uncertainty)
    model = config.model.expected_model_family
    familiar = familiar[familiar["model"] == model]
    familiar_predictions = familiar_predictions[familiar_predictions["model"] == model]
    familiar_uncertainty = familiar_uncertainty[familiar_uncertainty["model"] == model]
    rows: list[dict[str, Any]] = []
    for m7b_scope, (familiar_scope, uncertainty_scope) in _COMPARISON_MAP.items():
        m7b_row = metrics[(metrics["scope"] == m7b_scope) & (metrics["font_id"] == "all")]
        familiar_row = familiar[familiar["scope"] == familiar_scope]
        if m7b_row.empty or familiar_row.empty:
            continue
        m7b = m7b_row.iloc[0]
        fam = familiar_row.iloc[0]
        fam_target = familiar_predictions[familiar_predictions["scope"] == familiar_scope][
            "ssim_target"
        ]
        fam_stats = _target_stats(fam_target)
        unc = familiar_uncertainty[familiar_uncertainty["dataset"] == uncertainty_scope]
        unc_by_metric = {str(row.metric): row for row in unc.itertuples(index=False)}
        m7b_unc = uncertainty[
            (uncertainty["scope"] == m7b_scope) & (uncertainty["font_id"] == "all")
        ]
        m7b_unc_by_metric = {str(row.metric): row for row in m7b_unc.itertuples(index=False)}
        row: dict[str, Any] = {
            "m7b_scope": m7b_scope,
            "familiar_scope": familiar_scope,
            "m7b_protocol_role": str(m7b["protocol_role"]),
            "m7b_r2_defined": bool(m7b["r2_defined"]),
            "familiar_r2_defined": bool(fam["r2_defined"]),
            "m7b_font_condition": "held-out fonts: Georgia, Monaco, Impact",
            "familiar_font_condition": "familiar fonts: Arial, Courier New, Times New Roman",
            "m7b_sources": int(m7b["n_sources"]),
            "familiar_sources": int(fam["n_sources"]),
            "m7b_rows": int(m7b["n_rows"]),
            "familiar_rows": int(fam["n_samples"]),
            "m7b_target_mean": float(m7b["target_mean"]),
            "m7b_target_sd": float(m7b["target_sd"]),
            "familiar_target_mean": float(fam_stats["mean"]),
            "familiar_target_sd": float(fam_stats["sd"]),
            "m7b_mae": float(m7b["mae"]),
            "familiar_mae": float(fam["mae"]),
            "delta_mae": float(m7b["mae"] - fam["mae"]),
            "m7b_rmse": float(m7b["rmse"]),
            "familiar_rmse": float(fam["rmse"]),
            "delta_rmse": float(m7b["rmse"] - fam["rmse"]),
            "m7b_r2": float(m7b["r2"]) if bool(m7b["r2_defined"]) else math.nan,
            "familiar_r2": float(fam["r2"]) if bool(fam["r2_defined"]) else math.nan,
            "delta_r2": (
                float(m7b["r2"] - fam["r2"])
                if bool(m7b["r2_defined"]) and bool(fam["r2_defined"])
                else math.nan
            ),
            "caveat": (
                "Not a matched comparison: fonts, source composition and corruption mix "
                "differ; differences are descriptive, not causal or significance tests."
            ),
        }
        for metric in ("mae", "rmse", "r2"):
            familiar_row_unc = unc_by_metric.get(metric)
            heldout_row_unc = m7b_unc_by_metric.get(metric)
            if familiar_row_unc is None or heldout_row_unc is None:
                row[f"familiar_{metric}_ci_low"] = math.nan
                row[f"familiar_{metric}_ci_high"] = math.nan
                row[f"heldout_{metric}_ci_low"] = math.nan
                row[f"heldout_{metric}_ci_high"] = math.nan
                row[f"outside_familiar_{metric}_interval"] = False
                continue
            low = float(familiar_row_unc.ci_low)
            high = float(familiar_row_unc.ci_high)
            estimate = float(heldout_row_unc.estimate)
            row[f"familiar_{metric}_ci_low"] = low
            row[f"familiar_{metric}_ci_high"] = high
            row[f"heldout_{metric}_ci_low"] = float(heldout_row_unc.ci_low)
            row[f"heldout_{metric}_ci_high"] = float(heldout_row_unc.ci_high)
            metric_defined = True
            if metric == "r2":
                metric_defined = bool(m7b["r2_defined"]) and bool(fam["r2_defined"])
            row[f"outside_familiar_{metric}_interval"] = bool(
                metric_defined
                and math.isfinite(low)
                and math.isfinite(high)
                and (estimate < low or estimate > high)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _fingerprint_assets(paths: Iterable[str | Path] = FROZEN_ASSET_PATHS) -> dict[str, Any]:
    return rb._fingerprint_assets(paths)  # noqa: SLF001 - shared frozen-asset audit helper


def _isolation_checks(
    config: HeldoutConfig,
    dataset: DatasetResult,
    model_checks: Sequence[CheckResult],
    reference_checks: Sequence[CheckResult],
    *,
    frozen_before: Mapping[str, Any],
    frozen_after: Mapping[str, Any],
) -> tuple[CheckResult, ...]:
    checks: list[CheckResult] = [*dataset.checks, *reference_checks, *model_checks]
    m3_sources = pd.read_csv(config.reference.m3_sources)
    m3_ids = set(m3_sources["source_id"].astype(str))
    m3_fonts = set(m3_sources["font_id"].astype(str))
    checks.append(
        CheckResult(
            "isolation:source_ids_disjoint_from_m3",
            not (set(dataset.sources["source_id"].astype(str)) & m3_ids),
            "held-out source IDs do not overlap M3 source IDs",
        )
    )
    checks.append(
        CheckResult(
            "isolation:font_keys_disjoint_from_m3",
            not (set(dataset.sources["font_id"].astype(str)) & m3_fonts),
            f"held-out fonts={sorted(set(dataset.sources['font_id']))}, M3 fonts={sorted(m3_fonts)}",
        )
    )
    checks.append(
        CheckResult(
            "isolation:single_eval_split",
            bool((dataset.sources["split"] == "eval").all()),
            f"splits={_counts(dataset.sources['split'])}",
        )
    )
    checks.append(
        CheckResult(
            "isolation:frozen_assets_unchanged",
            dict(frozen_before) == dict(frozen_after),
            "every fingerprinted M3-M7A asset has the same bytes before and after M7B",
        )
    )
    checks.append(
        CheckResult(
            "isolation:no_model_artifact_written",
            all(
                not path.is_file()
                for path in (
                    dataset.root / "selected_policy.joblib",
                    dataset.root / "model.joblib",
                )
            ),
            "the dataset directory contains no model artifact",
        )
    )
    return tuple(checks)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    dataset: DatasetResult
    frame: pd.DataFrame
    predictions: np.ndarray
    metrics: pd.DataFrame
    font_metrics: pd.DataFrame
    family_metrics: pd.DataFrame
    source_metrics: pd.DataFrame
    target_distributions: pd.DataFrame
    uncertainty: pd.DataFrame
    comparison: pd.DataFrame
    isolation_checks: pd.DataFrame
    champion_card: dict[str, Any]
    frozen_before: dict[str, Any]
    frozen_after: dict[str, Any]


def evaluate_dataset(
    config: HeldoutConfig,
    *,
    dataset_root: Path | str | None = None,
    verify_images: bool = True,
) -> EvaluationResult:
    """Evaluate the frozen M5C champion zero-shot on the held-out-font dataset."""
    frozen_before = _fingerprint_assets()
    reference_checks = _verify_reference_inputs(config)
    _require_all_checks(reference_checks, "frozen reference inputs")
    dataset = load_dataset(config, dataset_root=dataset_root, verify_images=verify_images)
    estimator, card, model_checks = _load_champion(config)
    _require_all_checks(model_checks, "frozen M5C champion")
    if estimator is None:
        raise HeldoutFontError("M5C champion could not be loaded")
    frame = _eval_frame(dataset)
    predicted = predict(estimator, frame)
    frame = frame.copy()
    frame["prediction"] = predicted
    frame["residual"] = frame["ssim_target"] - frame["prediction"]
    frame["abs_residual"] = frame["residual"].abs()

    metrics = _metrics_table(frame, predicted, config)
    font_metrics = metrics[metrics["font_id"] != "all"].copy()
    family_metrics = _family_metrics(frame, predicted, config)
    source_metrics = _source_metrics(frame, predicted, config)
    target_distributions = _target_distributions(frame, config)
    uncertainty = _bootstrap_table(frame, predicted, config)
    comparison = _comparison_table(config, metrics, uncertainty)
    frozen_after = _fingerprint_assets()
    checks = _isolation_checks(
        config,
        dataset,
        model_checks,
        reference_checks,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
    )
    _require_all_checks(checks, "M7B isolation checks")
    check_frame = pd.DataFrame(
        [
            {"check": check.name, "passed": bool(check.passed), "detail": check.detail}
            for check in checks
        ]
    )
    return EvaluationResult(
        dataset=dataset,
        frame=frame,
        predictions=predicted,
        metrics=metrics,
        font_metrics=font_metrics,
        family_metrics=family_metrics,
        source_metrics=source_metrics,
        target_distributions=target_distributions,
        uncertainty=uncertainty,
        comparison=comparison,
        isolation_checks=check_frame,
        champion_card=card,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
    )


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _dataset_composition(dataset: DatasetResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (font_id, family, severity, protocol_scope), group in dataset.manifest.groupby(
        ["font_id", "variant_family", "severity_level", "protocol_scope"], sort=True
    ):
        rows.append(
            {
                "font_id": font_id,
                "variant_family": family,
                "severity_level": int(severity),
                "protocol_scope": protocol_scope,
                "n_rows": int(len(group)),
                "n_sources": int(group["source_id"].nunique()),
                "n_distinct_images": int(group["image_sha256"].nunique()),
                "target_mean": float(group["ssim_target"].mean()),
                "target_sd": float(group["ssim_target"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def _report_text(evaluation: EvaluationResult, config: HeldoutConfig) -> str:
    dataset = evaluation.dataset
    summary = dataset.summary
    counts = summary["counts"]
    metrics = evaluation.metrics
    overall = metrics[metrics["font_id"] == "all"]
    fonts = evaluation.font_metrics
    comparison = evaluation.comparison
    uncertainty = evaluation.uncertainty
    lines: list[str] = []
    add = lines.append

    add("# M7B: Held-Out Font Generalisation Benchmark")
    add("")
    add("## Scope and question")
    add("")
    add(
        "This benchmark asks how well the frozen M5C champion predicts SSIM against an "
        "exact source-specific clean reference when the text is rendered in fonts that "
        "were absent from model development. It is a zero-shot evaluation of a frozen "
        "model, not a training, calibration or model-selection experiment."
    )
    add("")
    add(
        "The target is SSIM between a corrupted synthetic image and its corresponding "
        "clean reference. PSNR is reported as a diagnostic only. This is a synthetic "
        "image-similarity study; it does not measure human readability, CAPTCHA "
        "solvability, security or real-world robustness."
    )
    add("")
    add("## Pre-registered hypothesis")
    add("")
    add(
        "> Performance on unseen fonts will differ from performance on the existing "
        "familiar-font benchmark, and the difference may vary across fonts and "
        "corruption families."
    )
    add("")
    add(
        "The evidence rule was fixed before evaluation: a held-out point estimate is "
        "described as differing when it lies outside the familiar-font source-grouped "
        "95% interval. This is descriptive, not a significance test."
    )
    add("")
    add("## Held-out fonts")
    add("")
    add("| Font | Family | File | SHA-256 | Rationale |")
    add("| --- | --- | --- | --- | --- |")
    for font in config.fonts:
        add(f"| {font.key} | {font.family} | `{font.path}` | `{font.sha256}` | {font.rationale} |")
    add("")
    add(
        "The three M3 development fonts (Arial, Courier New and Times New Roman) were "
        "excluded. Font files are pinned by SHA-256 and are used locally; they are not "
        "redistributed by this repository. A missing or changed file fails closed."
    )
    add("")
    add("## Rendering and target protocol")
    add("")
    add(
        f"The existing renderer is reused unchanged: {config.render.canvas[0]}×"
        f"{config.render.canvas[1]} grayscale canvas, text fitted to "
        f"{config.render.fit_fraction:.0%} of the width, vertical padding "
        f"{config.render.vertical_padding:.0%}, per-character jitter "
        f"{config.render.jitter_fraction:.0%}, spacing jitter "
        f"{config.render.spacing_jitter_fraction:.0%}, placement jitter "
        f"{config.render.placement_jitter_fraction:.0%}, and a constant per-source "
        f"background drawn from [{config.render.background_range[0]}, "
        f"{config.render.background_range[1]}]. Glyphs are rasterised on a supersampled "
        "canvas and downsampled with LANCZOS; images are stored as 8-bit grayscale PNG."
    )
    add("")
    add(
        f"Library versions in this run: Pillow `{pillow_version}`, NumPy `{np.__version__}`, "
        f"scikit-learn `{sklearn_version()}`. The target is SSIM between each corrupted "
        "image and its exact source-specific clean reference; PSNR is diagnostic only."
    )
    add("")
    add(
        "Corruption ladders: Gaussian blur radii "
        f"{list(config.corruptions.blur_radii_px)} px; additive Gaussian noise sigma "
        f"{list(config.corruptions.noise_sigmas)}; thin-line and arc-stroke counts "
        f"{list(config.corruptions.thin_line_stroke_counts)} and "
        f"{list(config.corruptions.arc_stroke_counts)}; elastic amplitudes "
        f"{list(config.corruptions.elastic_primary_amplitudes_px)} px primary and "
        f"{list(config.corruptions.elastic_diagnostic_amplitudes_px)} px diagnostic. "
        "All variants of a source share that source's clean reference and stay in the "
        "same split."
    )
    add("")
    add("## Dataset")
    add("")
    add(
        f"Dataset version `{summary['dataset_version']}`; generation hash "
        f"`{summary['generation_config_hash']}`; seed `{summary['seed']}`. "
        f"There are **{counts['sources']} sources**, **{counts['rows']} rows**, "
        f"**{counts['distinct_images']} distinct images** and "
        f"**{counts['fonts']} held-out fonts**. All sources are in one untouched "
        "`eval` split; the independent unit is the source group."
    )
    add("")
    add("| Font | Sources | Rows | Text lengths |")
    add("| --- | ---: | ---: | --- |")
    for font_id in config.font_keys:
        font_summary = summary["fonts"][font_id]
        add(
            f"| {font_id} | {font_summary['sources']} | {font_summary['rows']} | "
            f"{font_summary['text_lengths']} |"
        )
    add("")
    add(
        "Each source contributes one clean image, four blur levels, four noise levels, "
        "four thin-line levels, four arc-stroke levels and four elastic levels "
        "(0.5, 1, 2 and 3 px). The 0.5 px elastic rung and the constant clean target are "
        "diagnostic only. The primary slices exclude both."
    )
    add("")
    add("| Family | Rows | Target mean | Target SD | Target range |")
    add("| --- | ---: | ---: | ---: | --- |")
    for family in FAMILY_ORDER:
        stats = summary["targets"].get(family)
        if not stats:
            continue
        target = stats["target"]
        add(
            f"| {family} | {stats['rows']} | {_fmt(target['mean'])} | "
            f"{_fmt(target['sd'])} | [{_fmt(target['min'])}, {_fmt(target['max'])}] |"
        )
    add("")
    add(
        f"Near-degeneracy threshold: target SD < {config.protocol.near_degenerate_sd:g}. "
        "R² is suppressed for slices below that threshold even when it is numerically "
        "defined."
    )
    add("")
    diagnostic_sd = summary["targets"][ELASTIC_FAMILY]["by_severity"]["1"]["sd"]
    add(
        "No declared font was missing, no source was excluded and no generation or "
        f"evaluation failure occurred. The 0.5 px elastic slice has target SD "
        f"{_fmt(diagnostic_sd)} and is retained as diagnostic only. Every primary "
        "family/severity slice exceeds the near-degeneracy threshold; per-slice values "
        "are in `target_distributions.csv` and `family_metrics.csv`."
    )
    add("")
    add("## Frozen model")
    add("")
    add(
        f"The primary model is the exact M5C champion `{config.model.expected_model_family}` "
        f"with calibration `{config.model.expected_calibration}`. The artifact SHA-256 is "
        f"`{config.model.artifact_sha256}` and the card SHA-256 is "
        f"`{config.model.card_sha256}`. The eight-feature extractor, feature order and "
        "target definition were reused unchanged. No `fit`, `partial_fit`, recalibration "
        "or feature selection was performed."
    )
    add("")
    add("## Zero-shot results")
    add("")
    add("| Scope | Role | Sources | Rows | Target mean | Target SD | MAE | RMSE | R² |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for scope in SCOPE_ORDER:
        row = overall[overall["scope"] == scope]
        if row.empty:
            continue
        record = row.iloc[0]
        add(
            f"| {scope} | {record['protocol_role']} | {int(record['n_sources'])} | "
            f"{int(record['n_rows'])} | {_fmt(record['target_mean'])} | "
            f"{_fmt(record['target_sd'])} | {_fmt(record['mae'])} | "
            f"{_fmt(record['rmse'])} | {_fmt(record['r2'])} |"
        )
    add("")
    add("### Per-font primary performance")
    add("")
    add("| Font | Scope | Sources | Rows | MAE | RMSE | R² |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for font_id in config.font_keys:
        for scope in ("all_primary", "original_blur_noise", "interference", "geometric_primary"):
            row = fonts[(fonts["font_id"] == font_id) & (fonts["scope"] == scope)]
            if row.empty:
                continue
            record = row.iloc[0]
            add(
                f"| {font_id} | {scope} | {int(record['n_sources'])} | "
                f"{int(record['n_rows'])} | {_fmt(record['mae'])} | "
                f"{_fmt(record['rmse'])} | {_fmt(record['r2'])} |"
            )
    add("")
    add("### Per-corruption family and severity")
    add("")
    add("| Family | Severity | Role | Rows | MAE | RMSE | R² |")
    add("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    family_all = evaluation.family_metrics[evaluation.family_metrics["font_id"] == "all"]
    for family in FAMILY_ORDER:
        for severity in family_all[family_all["family"] == family]["severity"].tolist():
            record = family_all[
                (family_all["family"] == family) & (family_all["severity"] == severity)
            ].iloc[0]
            add(
                f"| {family} | {severity} | {record['protocol_role']} | "
                f"{int(record['n_rows'])} | {_fmt(record['mae'])} | "
                f"{_fmt(record['rmse'])} | {_fmt(record['r2'])} |"
            )
    add("")
    add("## Source-grouped uncertainty")
    add("")
    add(
        f"Uncertainty uses {config.run.n_bootstrap} source-grouped bootstrap replicates. "
        "Every replicate resamples sources, not image variants, so all variants of a "
        "source stay together. Intervals are descriptive and wide with 60 groups."
    )
    add("")
    add("| Scope | Font | Metric | Estimate | 95% interval | Groups |")
    add("| --- | --- | --- | ---: | --- | ---: |")
    for scope in config.protocol.bootstrap_scopes:
        for font_id in ("all", *config.font_keys):
            subset = uncertainty[
                (uncertainty["scope"] == scope) & (uncertainty["font_id"] == font_id)
            ]
            if subset.empty:
                continue
            for metric in ("mae", "rmse"):
                record = subset[subset["metric"] == metric]
                if record.empty:
                    continue
                item = record.iloc[0]
                add(
                    f"| {scope} | {font_id} | {metric} | {_fmt(item['estimate'])} | "
                    f"[{_fmt(item['ci_low'])}, {_fmt(item['ci_high'])}] | "
                    f"{int(item['n_groups'])} |"
                )
    add("")
    add("## Comparison with the familiar M5C benchmark")
    add("")
    add(
        "The familiar benchmark uses 12 M3 test sources and Arial, Courier New and Times "
        "New Roman. M7B uses 60 held-out sources and Georgia, Monaco and Impact. The "
        "corruption mix also differs: M7B adds thin-line and arc-stroke interference. "
        "Not a matched comparison: the table below is descriptive, the differences are "
        "not causal estimates, and they are not significance tests."
    )
    add("")
    add(
        "| Scope | M7B target mean | M7B target SD | Familiar target mean | "
        "Familiar target SD | M7B MAE | Familiar MAE | Δ MAE | M7B RMSE | "
        "Familiar RMSE | Δ RMSE | M7B R² | Familiar R² |"
    )
    add(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |"
    )
    for row in comparison.itertuples(index=False):
        add(
            f"| {row.m7b_scope} | {_fmt(row.m7b_target_mean)} | "
            f"{_fmt(row.m7b_target_sd)} | {_fmt(row.familiar_target_mean)} | "
            f"{_fmt(row.familiar_target_sd)} | {_fmt(row.m7b_mae)} | "
            f"{_fmt(row.familiar_mae)} | {_fmt(row.delta_mae)} | "
            f"{_fmt(row.m7b_rmse)} | {_fmt(row.familiar_rmse)} | "
            f"{_fmt(row.delta_rmse)} | {_fmt(row.m7b_r2)} | {_fmt(row.familiar_r2)} |"
        )
    add("")
    add("### Pre-registered evidence assessment")
    add("")
    if comparison.empty:
        add("No common comparison scopes were available.")
    else:
        primary_comparison = comparison[comparison["m7b_protocol_role"] == "primary"]
        for row in primary_comparison.itertuples(index=False):
            flags = []
            for metric in ("mae", "rmse", "r2"):
                if getattr(row, f"outside_familiar_{metric}_interval", False):
                    flags.append(metric.upper())
            if flags:
                add(
                    f"- `{row.m7b_scope}`: the held-out point estimate lies outside the "
                    f"familiar-font 95% interval for {', '.join(flags)}."
                )
            else:
                add(
                    f"- `{row.m7b_scope}`: the held-out point estimate lies inside the "
                    "familiar-font 95% intervals that are defined."
                )
        add("")
        add(
            "The diagnostic clean and 0.5 px elastic scopes are shown in the comparison "
            "table but are not used in this primary evidence assessment."
        )
    add("")
    add(
        "These statements are descriptive. They do not establish that the champion "
        "generalises to arbitrary fonts, renderers, resolutions or real CAPTCHAs, and "
        "they do not establish that it fails to do so."
    )
    add("")
    add("## Interpretation and limitations")
    add("")
    add(
        "- The held-out condition is the particular typefaces Georgia, Monaco and Impact "
        "under the existing renderer, canvas, grayscale/PNG pipeline and corruption "
        "definitions. Other fonts, renderers, resolutions, image formats and real-world "
        "captchas remain outside scope."
    )
    add(
        "- The three fonts are not a random sample of all fonts. Per-font differences "
        "describe these fonts, not a font population."
    )
    add(
        "- Sources are synthetic text strings; source-grouped uncertainty reflects "
        "variation across those synthetic sources, not across human users or real "
        "CAPTCHA deployments."
    )
    add(
        "- Clean and 0.5 px elastic slices are diagnostic only. R² is not interpreted "
        "where target variance is near zero or undefined."
    )
    add(
        "- The M5C comparison differs in font, source count, source composition and "
        "corruption mix. It is a reference point, not a controlled effect estimate."
    )
    add(
        "- A poor result would not prove that all feature-based models fail; a strong "
        "result would not prove generalisation to arbitrary fonts or renderers."
    )
    add("")
    add("## Reproduce")
    add("")
    add("```bash")
    add(
        "uv run --extra image --extra ml captcha-heldout-font-benchmark run "
        "--config configs/heldout_font_benchmark_v0_1_0.toml"
    )
    add(
        "uv run --extra image --extra ml captcha-heldout-font-benchmark verify "
        "--config configs/heldout_font_benchmark_v0_1_0.toml"
    )
    add("```")
    add("")
    add(
        "Generated datasets and reports are Git-ignored. The committed configuration, "
        "code, tests and documentation are the reproducible protocol."
    )
    add("")
    return "\n".join(lines)


def _frame_equivalent(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    for column in left.columns:
        a = left[column]
        b = right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(dtype=np.float64)
            bv = b.to_numpy(dtype=np.float64)
            if not np.allclose(av, bv, rtol=1e-10, atol=1e-12, equal_nan=True):
                return False
        else:
            if not a.astype(str).equals(b.astype(str)):
                return False
    return True


def _write_outputs(
    evaluation: EvaluationResult,
    config: HeldoutConfig,
    *,
    output_reports: Path | str | None = None,
    overwrite: bool = False,
    runtime_seconds: float = 0.0,
) -> dict[str, Any]:
    reports_root = _safe_output_path(
        output_reports or config.run.output_reports,
        "held-out-font reports",
    )
    if reports_root.exists() and not overwrite:
        raise HeldoutFontError(
            f"report directory already exists: {reports_root}; pass --overwrite or choose another path"
        )
    if reports_root.exists():
        shutil.rmtree(reports_root)
    staging = reports_root.with_name(reports_root.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        tables: dict[str, pd.DataFrame] = {
            "dataset_composition.csv": _dataset_composition(evaluation.dataset),
            "target_distributions.csv": evaluation.target_distributions,
            "overall_metrics.csv": evaluation.metrics[evaluation.metrics["font_id"] == "all"],
            "font_metrics.csv": evaluation.font_metrics,
            "family_metrics.csv": evaluation.family_metrics,
            "source_metrics.csv": evaluation.source_metrics,
            "predictions.csv": evaluation.frame[
                [
                    "sample_id",
                    "source_id",
                    "font_id",
                    "text_length",
                    "variant_family",
                    "severity_level",
                    "amplitude_px",
                    "protocol_scope",
                    "image_sha256",
                    "ssim_target",
                    "prediction",
                    "residual",
                    "abs_residual",
                ]
            ],
            "uncertainty.csv": evaluation.uncertainty,
            "comparison.csv": evaluation.comparison,
            "isolation_checks.csv": evaluation.isolation_checks,
        }
        for name, frame in tables.items():
            _write_csv(frame, staging / name)
        report = _report_text(evaluation, config)
        (staging / "report.md").write_text(report, encoding="utf-8")
        dataset_hashes = {
            name: sha256_file(evaluation.dataset.root / name)
            for name in ("manifest.csv", "sources.csv", "features.csv", "dataset_summary.json")
        }
        output_hashes = {
            name: sha256_file(staging / name) for name in (*OUTPUT_TABLE_NAMES, "report.md")
        }
        metadata = {
            "experiment": {
                "name": config.run.name,
                "version": config.run.version,
                "audit_version": AUDIT_VERSION,
                "seed": config.run.seed,
                "n_bootstrap": config.run.n_bootstrap,
                "config_sha256": config.config_sha256,
                "identity_hash": config.config_hash,
                "generation_config_hash": config.generation_hash,
                "evidence_rule": config.protocol.evidence_rule,
            },
            "code_revision": code_revision(),
            "package_version": package_version(),
            "platform": {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "sklearn_version": sklearn_version(),
            "runtime_seconds": float(runtime_seconds),
            "dataset": {
                "name": evaluation.dataset.summary["dataset_name"],
                "version": evaluation.dataset.summary["dataset_version"],
                "counts": evaluation.dataset.summary["counts"],
                "fonts": evaluation.dataset.summary["fonts"],
                "targets": evaluation.dataset.summary["targets"],
                "artifacts": evaluation.dataset.summary["artifacts"],
                "file_sha256": dataset_hashes,
            },
            "model": {
                "artifact_sha256": config.model.artifact_sha256,
                "card_sha256": config.model.card_sha256,
                "model_family": config.model.expected_model_family,
                "calibration": config.model.expected_calibration,
                "card": evaluation.champion_card,
            },
            "frozen_assets": json_safe(evaluation.frozen_before),
            "checks": evaluation.isolation_checks.to_dict("records"),
            "outputs": output_hashes,
            "report_sha256": output_hashes["report.md"],
        }
        _write_json(json_safe(metadata), staging / "run_metadata.json")
        staging.rename(reports_root)
        return {
            "reports_root": reports_root,
            "outputs": output_hashes,
            "dataset_hashes": dataset_hashes,
            "report": report,
            "metadata": metadata,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_benchmark(
    config: HeldoutConfig,
    *,
    output_dataset: Path | str | None = None,
    output_reports: Path | str | None = None,
    overwrite: bool = False,
    verify_images: bool = True,
) -> tuple[EvaluationResult, dict[str, Any]]:
    """Generate the dataset, evaluate the frozen champion and write reports."""
    started = time.perf_counter()
    frozen_before = _fingerprint_assets()
    dataset = generate_dataset(config, output_dataset=output_dataset, overwrite=overwrite)
    evaluation = evaluate_dataset(config, dataset_root=dataset.root, verify_images=verify_images)
    frozen_after = _fingerprint_assets()
    if dict(frozen_before) != dict(frozen_after):
        raise HeldoutFontError("a frozen M3-M7A asset changed during M7B generation/evaluation")
    outputs = _write_outputs(
        evaluation,
        config,
        output_reports=output_reports,
        overwrite=overwrite,
        runtime_seconds=time.perf_counter() - started,
    )
    return evaluation, outputs


def verify_benchmark(
    config: HeldoutConfig,
    *,
    dataset_root: Path | str | None = None,
    reports_root: Path | str | None = None,
) -> VerificationReport:
    """Replay generation/evaluation and verify every stored M7B artifact."""
    root = Path(dataset_root or config.run.output_dataset).resolve()
    reports = Path(reports_root or config.run.output_reports).resolve()
    checks: list[CheckResult] = []
    frozen_before = _fingerprint_assets()
    try:
        dataset = load_dataset(config, dataset_root=root, verify_images=True)
    except Exception as exc:  # noqa: BLE001 - verification reports the exact failure
        checks.append(CheckResult("dataset", False, f"{type(exc).__name__}: {exc}"))
        return VerificationReport(dataset=str(root), checks=tuple(checks))
    checks.append(CheckResult("dataset", True, f"loaded and rehashed {dataset.root}"))
    try:
        evaluation = evaluate_dataset(config, dataset_root=root, verify_images=True)
    except Exception as exc:  # noqa: BLE001 - verification reports the exact failure
        checks.append(CheckResult("evaluation", False, f"{type(exc).__name__}: {exc}"))
        return VerificationReport(dataset=str(root), checks=tuple(checks))
    checks.append(CheckResult("evaluation", True, "frozen champion replay succeeded"))
    if not reports.is_dir():
        checks.append(CheckResult("reports", False, f"missing report directory: {reports}"))
        return VerificationReport(dataset=str(root), checks=tuple(checks))

    expected_tables: dict[str, pd.DataFrame] = {
        "dataset_composition.csv": _dataset_composition(dataset),
        "target_distributions.csv": evaluation.target_distributions,
        "overall_metrics.csv": evaluation.metrics[evaluation.metrics["font_id"] == "all"],
        "font_metrics.csv": evaluation.font_metrics,
        "family_metrics.csv": evaluation.family_metrics,
        "source_metrics.csv": evaluation.source_metrics,
        "predictions.csv": evaluation.frame[
            [
                "sample_id",
                "source_id",
                "font_id",
                "text_length",
                "variant_family",
                "severity_level",
                "amplitude_px",
                "protocol_scope",
                "image_sha256",
                "ssim_target",
                "prediction",
                "residual",
                "abs_residual",
            ]
        ],
        "uncertainty.csv": evaluation.uncertainty,
        "comparison.csv": evaluation.comparison,
        "isolation_checks.csv": evaluation.isolation_checks,
    }
    for name, expected in expected_tables.items():
        path = reports / name
        if not path.is_file():
            checks.append(CheckResult(f"table:{name}", False, f"missing {path}"))
            continue
        stored = pd.read_csv(path)
        checks.append(
            CheckResult(
                f"table:{name}",
                _frame_equivalent(stored, expected),
                f"{len(expected)} row(s)",
            )
        )
    expected_report = _report_text(evaluation, config)
    report_path = reports / "report.md"
    if report_path.is_file():
        stored_report = report_path.read_text(encoding="utf-8")
        checks.append(
            CheckResult(
                "report",
                stored_report == expected_report,
                f"{len(expected_report)} character(s) reproduced exactly",
            )
        )
    else:
        checks.append(CheckResult("report", False, f"missing {report_path}"))

    metadata_path = reports / "run_metadata.json"
    if not metadata_path.is_file():
        checks.append(CheckResult("metadata", False, f"missing {metadata_path}"))
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            recorded_outputs = metadata.get("outputs", {})
            mismatched = []
            for name, expected_hash in recorded_outputs.items():
                path = reports / name
                if not path.is_file() or sha256_file(path) != expected_hash:
                    mismatched.append(name)
            checks.append(
                CheckResult(
                    "metadata:output_hashes",
                    not mismatched,
                    f"mismatched={mismatched}"
                    if mismatched
                    else "all recorded output hashes match",
                )
            )
            recorded_dataset = metadata.get("dataset", {}).get("file_sha256", {})
            mismatched_dataset = []
            for name, expected_hash in recorded_dataset.items():
                path = root / name
                if not path.is_file() or sha256_file(path) != expected_hash:
                    mismatched_dataset.append(name)
            checks.append(
                CheckResult(
                    "metadata:dataset_hashes",
                    not mismatched_dataset,
                    (
                        f"mismatched={mismatched_dataset}"
                        if mismatched_dataset
                        else "all recorded dataset hashes match"
                    ),
                )
            )
            checks.append(
                CheckResult(
                    "metadata:frozen_assets",
                    dict(metadata.get("frozen_assets", {})) == dict(frozen_before),
                    "recorded frozen-asset fingerprints match the current files",
                )
            )
            checks.append(
                CheckResult(
                    "metadata:checks",
                    all(bool(item.get("passed")) for item in metadata.get("checks", [])),
                    "all recorded isolation checks passed",
                )
            )
        except Exception as exc:  # noqa: BLE001 - verification reports the exact failure
            checks.append(CheckResult("metadata", False, f"{type(exc).__name__}: {exc}"))
    checks.append(
        CheckResult(
            "frozen_assets_unchanged",
            dict(evaluation.frozen_before) == dict(evaluation.frozen_after),
            "frozen assets unchanged across the verification replay",
        )
    )
    return VerificationReport(dataset=str(root), checks=tuple(checks))


def _print_report(report: VerificationReport) -> None:
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print("verification: PASS" if report.ok else "verification: FAIL")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-heldout-font-benchmark",
        description=(
            "M7B held-out-font generalisation benchmark: generate an isolated dataset "
            "and evaluate the frozen M5C champion zero-shot."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="generate the dataset, evaluate and write reports")
    run.add_argument("--config", required=True, help="M7B TOML configuration")
    run.add_argument("--output-dataset", default=None, help="override dataset directory")
    run.add_argument("--output-reports", default=None, help="override report directory")
    run.add_argument("--overwrite", action="store_true", help="replace existing M7B outputs")
    run.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="skip recomputing image hashes, targets and features during evaluation",
    )

    evaluate = subparsers.add_parser("evaluate", help="evaluate an existing dataset")
    evaluate.add_argument("--config", required=True, help="M7B TOML configuration")
    evaluate.add_argument("--dataset", default=None, help="dataset directory")
    evaluate.add_argument("--output-reports", default=None, help="override report directory")
    evaluate.add_argument("--overwrite", action="store_true", help="replace existing reports")
    evaluate.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="skip recomputing image hashes, targets and features during evaluation",
    )

    verify = subparsers.add_parser("verify", help="replay and verify stored M7B artifacts")
    verify.add_argument("--config", required=True, help="M7B TOML configuration")
    verify.add_argument("--dataset", default=None, help="dataset directory")
    verify.add_argument("--reports", default=None, help="report directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "run":
            evaluation, outputs = run_benchmark(
                config,
                output_dataset=args.output_dataset,
                output_reports=args.output_reports,
                overwrite=args.overwrite,
                verify_images=not args.skip_image_verification,
            )
            overall = evaluation.metrics[
                (evaluation.metrics["scope"] == "all_primary")
                & (evaluation.metrics["font_id"] == "all")
            ].iloc[0]
            print(f"dataset: {evaluation.dataset.root}")
            print(f"reports: {outputs['reports_root']}")
            print(
                "all_primary: "
                f"MAE={_fmt(overall['mae'])} RMSE={_fmt(overall['rmse'])} "
                f"R2={_fmt(overall['r2'])} rows={int(overall['n_rows'])} "
                f"sources={int(overall['n_sources'])}"
            )
            print("run: PASS")
            return 0
        if args.command == "evaluate":
            evaluation = evaluate_dataset(
                config,
                dataset_root=args.dataset,
                verify_images=not args.skip_image_verification,
            )
            outputs = _write_outputs(
                evaluation,
                config,
                output_reports=args.output_reports,
                overwrite=args.overwrite,
            )
            overall = evaluation.metrics[
                (evaluation.metrics["scope"] == "all_primary")
                & (evaluation.metrics["font_id"] == "all")
            ].iloc[0]
            print(f"reports: {outputs['reports_root']}")
            print(
                "all_primary: "
                f"MAE={_fmt(overall['mae'])} RMSE={_fmt(overall['rmse'])} "
                f"R2={_fmt(overall['r2'])} rows={int(overall['n_rows'])} "
                f"sources={int(overall['n_sources'])}"
            )
            print("evaluate: PASS")
            return 0
        if args.command == "verify":
            report = verify_benchmark(config, dataset_root=args.dataset, reports_root=args.reports)
            _print_report(report)
            return 0 if report.ok else 1
    except (
        HeldoutFontError,
        HeldoutFontConfigError,
        ModelError,
        RenderError,
        FontResolutionError,
        ImageContractError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
