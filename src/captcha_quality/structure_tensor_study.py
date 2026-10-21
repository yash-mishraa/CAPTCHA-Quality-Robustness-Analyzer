"""Milestone M6D multi-scale structure-tensor study.

M6A found that coarse regional ink-occupancy features did not improve
retention-aware geometric-distortion prediction. M6B varied only the grid
resolution and found that finer grids resolve an elastic warp better without
producing a reliable predictive gain. M6C tested whole-image summaries of local
gradient magnitude and orientation; neither local-only nor global-plus-local
features qualified under the predeclared retention rule, and both regressed
clean-reference or original-task retention.

M6D tests a distinct hypothesis:

    Do compact multi-scale structure-tensor features computed from the corrupted
    image alone improve geometric-distortion SSIM prediction while preserving
    clean-reference and original-task performance?

A structure tensor summarizes the local arrangement and directional coherence
of gradients, not merely their average strength. The nine features in
:mod:`captcha_quality.structure_tensor_features` are computed at two
predeclared Gaussian scales (sigma = 1.0 and 3.0 px) from one corrupted image.
The three predeclared candidates are the frozen M5C champion (not refitted), a
structure-tensor-only model, and a global-plus-structure model. The model
family, hyperparameters, seed, training rows and objective are identical for
the two refitted candidates, so a difference between them is attributable to
the input representation alone. No hyperparameter is tuned, no post-hoc
calibration is applied and no feature is removed in response to a diagnostic.

**No-reference boundary.** Every primary feature is a pure function of one
corrupted grayscale image. The extractor accepts no clean reference, target,
identifier, family, severity, font, split or path value, and the module's code
contains no reference to any of them. The clean reference is used only by the
separate evaluation pipeline that computes the SSIM label. This boundary is
enforced by a runtime check on every run and by structural tests, not merely by
inspecting output column names.

The retention-aware eligibility constraints, selection score, tolerances and
tie-breaking are the M6A/M6B/M6C rule unchanged, so all four studies are
directly comparable, and the frozen M5C champion is the fallback. The 0.5 px
geometric rung remains diagnostic only.

The target is synthetic image similarity (SSIM against the exact clean
reference). Nothing here measures human readability, CAPTCHA solvability,
security or real-world robustness.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import re
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from captcha_quality import geometric_benchmark as gb
from captcha_quality import local_structure_study as lss
from captcha_quality import retention_aware as ra
from captcha_quality import robustness_benchmark as rb
from captcha_quality import spatial_resolution_study as srs
from captcha_quality import spatial_study as ss
from captcha_quality import structure_tensor_features as stf
from captcha_quality.config import FEATURE_SET_VERSION
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.dataset import CheckResult, load_grayscale
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import (
    TARGET,
    CandidateSpec,
    ModelError,
    save_model_bundle,
    sklearn_version,
)
from captcha_quality.provenance import (
    canonical_json,
    code_revision,
    package_version,
    sha256_bytes,
    sha256_file,
)

#: Representation names, matching the declared feature sets.
FEATURE_SET_GLOBAL: Final[str] = "global"
FEATURE_SET_STRUCTURE: Final[str] = "structure"
FEATURE_SET_COMBINED: Final[str] = "combined"

#: Candidate names.
GLOBAL_BASELINE: Final[str] = "global_baseline"
STRUCTURE_ONLY: Final[str] = "structure_only"
GLOBAL_PLUS_STRUCTURE: Final[str] = "global_plus_structure"

#: Slice labels, matching the M5C, M5D, M6A and M6B conventions.
GEO_LABEL: Final[str] = "geometric_primary"
CLEAN_LABEL: Final[str] = "clean_reference"
ORIGINAL_LABEL: Final[str] = "original_blur_noise"
OVERALL_LABEL: Final[str] = "overall"

_EXPECTED_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "run",
        "schema",
        "m5c",
        "m6a",
        "m6b",
        "m6c",
        "model",
        "protocol",
        "features",
        "ablations",
        "selection",
    }
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Identifiers that must never appear in the structure-tensor feature module's
#: code. The clean reference, the targets and every metadata field are outside
#: the feature boundary.
_FORBIDDEN_FEATURE_IDENTIFIERS: Final[frozenset[str]] = frozenset(
    {
        "ssim",
        "ssim_target",
        "psnr",
        "psnr_target",
        "target",
        "reference",
        "reference_image",
        "ref",
        "clean",
        "clean_image",
        "source_id",
        "severity",
        "severity_level",
        "corruption_type",
        "font_id",
        "split",
        "image_path",
        "metadata",
    }
)

#: Frozen inputs audited before and after the milestone. M6D's own outputs are
#: not in this list.
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
)


class StructureTensorStudyError(RuntimeError):
    """Raised when the M6D protocol cannot be honoured."""


class StructureTensorConfigError(ValueError):
    """Raised when the M6D configuration is invalid."""


@dataclass(frozen=True, slots=True)
class M5CPin:
    """Pins for the upstream M5C study and its champion bundle."""

    config: Path
    config_sha256: str
    protocol_hash: str
    champion_dir: Path
    champion_stem: str
    champion_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class M6APin:
    """Pins for the M6A study whose spatial definitions must stay unchanged."""

    config: Path
    config_sha256: str
    protocol_hash: str
    feature_set_version: str


@dataclass(frozen=True, slots=True)
class M6BPin:
    """Pins for the M6B resolution study whose configuration must stay unchanged."""

    config: Path
    config_sha256: str
    protocol_hash: str


@dataclass(frozen=True, slots=True)
class M6CPin:
    """Pins for the immediately preceding local-structure study."""

    config: Path
    config_sha256: str
    protocol_hash: str


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """The fixed model family and hyperparameters shared by every candidate."""

    family: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    seed: int


@dataclass(frozen=True, slots=True)
class AblationSettings:
    """The predeclared candidate representations."""

    names: tuple[str, ...]
    feature_sets: Mapping[str, str]
    refit: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class StructureTensorConfig:
    """The complete M6D protocol."""

    run: rb.RunSettings
    global_feature_set_version: str
    global_feature_names: tuple[str, ...]
    target: str
    m5c: M5CPin
    m6a: M6APin
    m6b: M6BPin
    m6c: M6CPin
    model: ModelSettings
    protocol: ra.ProtocolSettings
    features: stf.StructureTensorSpec
    ablations: AblationSettings
    selection: ss.StudySelectionSettings
    config_path: Path
    config_sha256: str

    def identity_payload(self) -> dict[str, Any]:
        """Return the declared protocol, excluding the file location and hash."""
        return {
            "run": {
                "name": self.run.name,
                "version": self.run.version,
                "seed": self.run.seed,
                "n_bootstrap": self.run.n_bootstrap,
                "output_reports": str(self.run.output_reports),
                "output_models": str(self.run.output_models),
            },
            "schema": {
                "global_feature_set_version": self.global_feature_set_version,
                "global_feature_names": list(self.global_feature_names),
                "target": self.target,
                "split_names": ["train", "val", "test"],
            },
            "m5c": {
                "config": str(self.m5c.config),
                "config_sha256": self.m5c.config_sha256,
                "protocol_hash": self.m5c.protocol_hash,
                "champion_dir": str(self.m5c.champion_dir),
                "champion_stem": self.m5c.champion_stem,
                "champion_artifact_sha256": self.m5c.champion_artifact_sha256,
            },
            "m6a": {
                "config": str(self.m6a.config),
                "config_sha256": self.m6a.config_sha256,
                "protocol_hash": self.m6a.protocol_hash,
                "feature_set_version": self.m6a.feature_set_version,
            },
            "m6b": {
                "config": str(self.m6b.config),
                "config_sha256": self.m6b.config_sha256,
                "protocol_hash": self.m6b.protocol_hash,
            },
            "m6c": {
                "config": str(self.m6c.config),
                "config_sha256": self.m6c.config_sha256,
                "protocol_hash": self.m6c.protocol_hash,
            },
            "model": {
                "family": self.model.family,
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "min_samples_leaf": self.model.min_samples_leaf,
                "seed": self.model.seed,
            },
            "protocol": {
                "primary_amplitudes_px": list(self.protocol.primary_amplitudes_px),
                "diagnostic_amplitudes_px": list(self.protocol.diagnostic_amplitudes_px),
                "primary_severities": list(self.protocol.primary_severities),
                "diagnostic_severities": list(self.protocol.diagnostic_severities),
            },
            "features": self.features.identity_payload(),
            "ablations": {
                "names": list(self.ablations.names),
                "feature_sets": {
                    name: self.ablations.feature_sets[name]
                    for name in sorted(self.ablations.feature_sets)
                },
                "refit": {
                    name: self.ablations.refit[name] for name in sorted(self.ablations.refit)
                },
            },
            "selection": {
                "rule": self.selection.rule,
                "w_geometric": self.selection.w_geometric,
                "w_clean": self.selection.w_clean,
                "w_original": self.selection.w_original,
                "w_severity": self.selection.w_severity,
                "tolerance_clean": self.selection.tolerance_clean,
                "tolerance_original": self.selection.tolerance_original,
                "tolerance_severity": self.selection.tolerance_severity,
            },
        }

    @property
    def config_hash(self) -> str:
        return sha256_bytes(canonical_json(self.identity_payload()).encode("utf-8"))

    def describe(self) -> str:
        return (
            f"{self.run.name} {self.run.version} "
            f"(seed {self.run.seed}, bootstrap {self.run.n_bootstrap})"
        )

    def feature_names_for(self, feature_set: str) -> tuple[str, ...]:
        """Return the ordered feature names for one declared feature set."""
        if feature_set == FEATURE_SET_GLOBAL:
            return tuple(self.global_feature_names)
        if feature_set == FEATURE_SET_STRUCTURE:
            return tuple(self.features.feature_names)
        if feature_set == FEATURE_SET_COMBINED:
            return tuple(self.global_feature_names) + tuple(self.features.feature_names)
        raise StructureTensorConfigError(f"unknown feature set {feature_set!r}")


# ---------------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------------


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise StructureTensorConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise StructureTensorConfigError(f"missing or invalid [{name}] table")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructureTensorConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StructureTensorConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StructureTensorConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise StructureTensorConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise StructureTensorConfigError(f"{label} must be a boolean, got {value!r}")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise StructureTensorConfigError(f"{label} must be a TOML array")
    items = tuple(_as_str(item, f"{label}[]") for item in value)
    if not items or len(set(items)) != len(items):
        raise StructureTensorConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_float_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise StructureTensorConfigError(f"{label} must be a TOML array")
    items = tuple(_as_float(item, f"{label}[]") for item in value)
    if not items or len(set(items)) != len(items):
        raise StructureTensorConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise StructureTensorConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _parse_run(raw: Mapping[str, object]) -> rb.RunSettings:
    _check_keys(
        raw,
        {"name", "version", "seed", "n_bootstrap", "output_reports", "output_models"},
        "run",
    )
    return rb.RunSettings(
        name=_as_str(raw["name"], "run.name"),
        version=_as_str(raw["version"], "run.version"),
        seed=_as_int(raw["seed"], "run.seed"),
        n_bootstrap=_as_int(raw["n_bootstrap"], "run.n_bootstrap"),
        selection_metric="rmse",
        output_reports=Path(_as_str(raw["output_reports"], "run.output_reports")),
        output_models=Path(_as_str(raw["output_models"], "run.output_models")),
    )


def _parse_m5c(raw: Mapping[str, object]) -> M5CPin:
    _check_keys(
        raw,
        {
            "config",
            "config_sha256",
            "protocol_hash",
            "champion_dir",
            "champion_stem",
            "champion_artifact_sha256",
        },
        "m5c",
    )
    return M5CPin(
        config=Path(_as_str(raw["config"], "m5c.config")),
        config_sha256=_as_sha256(raw["config_sha256"], "m5c.config_sha256"),
        protocol_hash=_as_sha256(raw["protocol_hash"], "m5c.protocol_hash"),
        champion_dir=Path(_as_str(raw["champion_dir"], "m5c.champion_dir")),
        champion_stem=_as_str(raw["champion_stem"], "m5c.champion_stem"),
        champion_artifact_sha256=_as_sha256(
            raw["champion_artifact_sha256"], "m5c.champion_artifact_sha256"
        ),
    )


def _parse_m6a(raw: Mapping[str, object]) -> M6APin:
    _check_keys(raw, {"config", "config_sha256", "protocol_hash", "feature_set_version"}, "m6a")
    return M6APin(
        config=Path(_as_str(raw["config"], "m6a.config")),
        config_sha256=_as_sha256(raw["config_sha256"], "m6a.config_sha256"),
        protocol_hash=_as_sha256(raw["protocol_hash"], "m6a.protocol_hash"),
        feature_set_version=_as_str(raw["feature_set_version"], "m6a.feature_set_version"),
    )


def _parse_m6b(raw: Mapping[str, object]) -> M6BPin:
    _check_keys(raw, {"config", "config_sha256", "protocol_hash"}, "m6b")
    return M6BPin(
        config=Path(_as_str(raw["config"], "m6b.config")),
        config_sha256=_as_sha256(raw["config_sha256"], "m6b.config_sha256"),
        protocol_hash=_as_sha256(raw["protocol_hash"], "m6b.protocol_hash"),
    )


def _parse_m6c(raw: Mapping[str, object]) -> M6CPin:
    _check_keys(raw, {"config", "config_sha256", "protocol_hash"}, "m6c")
    return M6CPin(
        config=Path(_as_str(raw["config"], "m6c.config")),
        config_sha256=_as_sha256(raw["config_sha256"], "m6c.config_sha256"),
        protocol_hash=_as_sha256(raw["protocol_hash"], "m6c.protocol_hash"),
    )


def _parse_model(raw: Mapping[str, object]) -> ModelSettings:
    _check_keys(raw, {"family", "n_estimators", "max_depth", "min_samples_leaf", "seed"}, "model")
    settings = ModelSettings(
        family=_as_str(raw["family"], "model.family"),
        n_estimators=_as_int(raw["n_estimators"], "model.n_estimators"),
        max_depth=_as_int(raw["max_depth"], "model.max_depth"),
        min_samples_leaf=_as_int(raw["min_samples_leaf"], "model.min_samples_leaf"),
        seed=_as_int(raw["seed"], "model.seed"),
    )
    if settings.family != "random_forest":
        raise StructureTensorConfigError(
            f"model.family must be 'random_forest', got {settings.family!r}"
        )
    if settings.n_estimators < 1:
        raise StructureTensorConfigError("model.n_estimators must be at least 1")
    if settings.max_depth < 0:
        raise StructureTensorConfigError("model.max_depth must be zero (unlimited) or positive")
    if settings.min_samples_leaf < 1:
        raise StructureTensorConfigError("model.min_samples_leaf must be at least 1")
    return settings


def _parse_protocol(raw: Mapping[str, object]) -> ra.ProtocolSettings:
    _check_keys(
        raw,
        {
            "primary_amplitudes_px",
            "diagnostic_amplitudes_px",
            "primary_severities",
            "diagnostic_severities",
        },
        "protocol",
    )
    settings = ra.ProtocolSettings(
        primary_amplitudes_px=tuple(
            _as_float(item, "protocol.primary_amplitudes_px[]")
            for item in raw["primary_amplitudes_px"]
        ),
        diagnostic_amplitudes_px=tuple(
            _as_float(item, "protocol.diagnostic_amplitudes_px[]")
            for item in raw["diagnostic_amplitudes_px"]
        ),
        primary_severities=tuple(
            _as_int(item, "protocol.primary_severities[]") for item in raw["primary_severities"]
        ),
        diagnostic_severities=tuple(
            _as_int(item, "protocol.diagnostic_severities[]")
            for item in raw["diagnostic_severities"]
        ),
    )
    if set(settings.primary_amplitudes_px) & set(settings.diagnostic_amplitudes_px):
        raise StructureTensorConfigError(
            "the diagnostic amplitude must not also appear in the primary ladder"
        )
    if set(settings.primary_severities) & set(settings.diagnostic_severities):
        raise StructureTensorConfigError(
            "the diagnostic severity must not also appear in the primary ladder"
        )
    return settings


def _parse_features(raw: Mapping[str, object]) -> stf.StructureTensorSpec:
    _check_keys(
        raw,
        {
            "scales",
            "coherence_quantile",
            "energy_quantile",
            "epsilon",
            "gaussian_truncate",
            "gradient_operator",
            "smoothing_kernel",
            "boundary",
        },
        "features",
    )
    scales = _as_float_tuple(raw["scales"], "features.scales")
    if len(scales) != 2:
        raise StructureTensorConfigError("features.scales must declare exactly two scales")
    if max(scales) / min(scales) < 2.0:
        raise StructureTensorConfigError(
            "features.scales must be meaningfully different (max/min >= 2.0)"
        )
    try:
        return stf.StructureTensorSpec(
            scales=scales,
            coherence_quantile=_as_float(raw["coherence_quantile"], "features.coherence_quantile"),
            energy_quantile=_as_float(raw["energy_quantile"], "features.energy_quantile"),
            epsilon=_as_float(raw["epsilon"], "features.epsilon"),
            gaussian_truncate=_as_float(raw["gaussian_truncate"], "features.gaussian_truncate"),
            gradient_operator=_as_str(raw["gradient_operator"], "features.gradient_operator"),
            smoothing_kernel=_as_str(raw["smoothing_kernel"], "features.smoothing_kernel"),
            boundary=_as_str(raw["boundary"], "features.boundary"),
        )
    except stf.StructureTensorFeatureError as exc:
        raise StructureTensorConfigError(str(exc)) from exc


def _parse_ablations(raw: Mapping[str, object]) -> AblationSettings:
    _check_keys(raw, {"names", "feature_sets", "refit"}, "ablations")
    names = _as_str_tuple(raw["names"], "ablations.names")
    if set(names) != {GLOBAL_BASELINE, STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE}:
        raise StructureTensorConfigError(
            "ablations.names must be exactly "
            f"{[GLOBAL_BASELINE, STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE]}, got {sorted(names)}"
        )
    raw_feature_sets = raw["feature_sets"]
    raw_refit = raw["refit"]
    if not isinstance(raw_feature_sets, dict) or not isinstance(raw_refit, dict):
        raise StructureTensorConfigError(
            "ablations.feature_sets and ablations.refit must be tables"
        )
    _check_keys(raw_feature_sets, set(names), "ablations.feature_sets")
    _check_keys(raw_refit, set(names), "ablations.refit")
    feature_sets = {
        name: _as_str(raw_feature_sets[name], f"ablations.feature_sets.{name}") for name in names
    }
    refit = {name: _as_bool(raw_refit[name], f"ablations.refit.{name}") for name in names}
    expected = {
        GLOBAL_BASELINE: FEATURE_SET_GLOBAL,
        STRUCTURE_ONLY: FEATURE_SET_STRUCTURE,
        GLOBAL_PLUS_STRUCTURE: FEATURE_SET_COMBINED,
    }
    for name, feature_set in expected.items():
        if feature_sets[name] != feature_set:
            raise StructureTensorConfigError(
                f"ablations.feature_sets.{name} must be {feature_set!r}, got {feature_sets[name]!r}"
            )
    if refit[GLOBAL_BASELINE]:
        raise StructureTensorConfigError(
            "the frozen champion must not be refitted; it is the reference policy"
        )
    if not refit[STRUCTURE_ONLY] or not refit[GLOBAL_PLUS_STRUCTURE]:
        raise StructureTensorConfigError(
            "the structure-only and global-plus-structure ablations must be refitted"
        )
    return AblationSettings(names=names, feature_sets=feature_sets, refit=refit)


def load_structure_tensor_config(path: Path | str) -> StructureTensorConfig:
    """Load and validate the M6D configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise StructureTensorConfigError(f"missing configuration file: {config_path}")
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    unknown = sorted(set(payload) - _EXPECTED_TOP_KEYS)
    if unknown:
        raise StructureTensorConfigError(
            f"unknown top-level key(s): {unknown}; allowed keys are {sorted(_EXPECTED_TOP_KEYS)}"
        )
    try:
        global_version, global_names, target = ss._parse_schema(  # noqa: SLF001
            _section(payload, "schema")
        )
        selection = ss._parse_selection(  # noqa: SLF001
            _section(payload, "selection")
        )
    except ss.SpatialStudyConfigError as exc:
        raise StructureTensorConfigError(str(exc)) from exc
    config = StructureTensorConfig(
        run=_parse_run(_section(payload, "run")),
        global_feature_set_version=global_version,
        global_feature_names=global_names,
        target=target,
        m5c=_parse_m5c(_section(payload, "m5c")),
        m6a=_parse_m6a(_section(payload, "m6a")),
        m6b=_parse_m6b(_section(payload, "m6b")),
        m6c=_parse_m6c(_section(payload, "m6c")),
        model=_parse_model(_section(payload, "model")),
        protocol=_parse_protocol(_section(payload, "protocol")),
        features=_parse_features(_section(payload, "features")),
        ablations=_parse_ablations(_section(payload, "ablations")),
        selection=selection,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )
    if config.global_feature_names != tuple(FEATURE_NAMES):
        raise StructureTensorConfigError(
            "schema.global_feature_names must match the frozen extractor "
            f"{list(FEATURE_NAMES)}; the eight-feature pipeline must not be altered"
        )
    if config.global_feature_set_version != FEATURE_SET_VERSION:
        raise StructureTensorConfigError(
            f"schema.global_feature_set_version must be {FEATURE_SET_VERSION!r}, got "
            f"{config.global_feature_set_version!r}"
        )
    if config.target != TARGET:
        raise StructureTensorConfigError(f"schema.target must be {TARGET!r}")
    if config.m6a.feature_set_version != "spatial-v1":
        raise StructureTensorConfigError(
            "m6a.feature_set_version must be 'spatial-v1'; the frozen spatial "
            "definitions must be reused unchanged"
        )
    if config.protocol.primary_amplitudes_px != gb.PRIMARY_AMPLITUDES_PX:
        raise StructureTensorConfigError(
            "protocol.primary_amplitudes_px must match the M5B primary ladder "
            f"{list(gb.PRIMARY_AMPLITUDES_PX)}"
        )
    if config.protocol.diagnostic_amplitudes_px != gb.DIAGNOSTIC_AMPLITUDES_PX:
        raise StructureTensorConfigError(
            "protocol.diagnostic_amplitudes_px must match the M5B diagnostic ladder "
            f"{list(gb.DIAGNOSTIC_AMPLITUDES_PX)}"
        )
    # The shipped protocol declares one fixed structure-tensor design. A
    # different scale, quantile, tolerance or operator would be a different
    # predeclaration, so it is rejected here rather than silently accepted.
    if config.features.scales != stf.DEFAULT_SCALES:
        raise StructureTensorConfigError(
            f"features.scales must be exactly {list(stf.DEFAULT_SCALES)}"
        )
    if config.features.coherence_quantile != stf.DEFAULT_COHERENCE_QUANTILE:
        raise StructureTensorConfigError(
            f"features.coherence_quantile must be {stf.DEFAULT_COHERENCE_QUANTILE}"
        )
    if config.features.energy_quantile != stf.DEFAULT_ENERGY_QUANTILE:
        raise StructureTensorConfigError(
            f"features.energy_quantile must be {stf.DEFAULT_ENERGY_QUANTILE}"
        )
    if config.features.epsilon != stf.DEFAULT_EPSILON:
        raise StructureTensorConfigError(f"features.epsilon must be {stf.DEFAULT_EPSILON}")
    if config.features.gaussian_truncate != stf.DEFAULT_GAUSSIAN_TRUNCATE:
        raise StructureTensorConfigError(
            f"features.gaussian_truncate must be {stf.DEFAULT_GAUSSIAN_TRUNCATE}"
        )
    if config.features.gradient_operator != "sobel_3x3":
        raise StructureTensorConfigError("features.gradient_operator must be 'sobel_3x3'")
    if config.features.smoothing_kernel != "gaussian":
        raise StructureTensorConfigError("features.smoothing_kernel must be 'gaussian'")
    if config.features.boundary != "reflect":
        raise StructureTensorConfigError("features.boundary must be 'reflect'")
    expected_features = stf.StructureTensorSpec().feature_names
    if config.features.feature_names != expected_features:
        raise StructureTensorConfigError(
            f"features must declare exactly {list(expected_features)}, got "
            f"{list(config.features.feature_names)}"
        )
    return config


# ---------------------------------------------------------------------------
# Frames, the no-reference boundary and candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructureTensorFrames:
    """The frozen frames plus the structure-tensor feature table."""

    retention: ra.RetentionFrames
    structure_features: pd.DataFrame
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def with_structure(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` joined with the structure-tensor features."""
        if frame["sample_id"].duplicated().any():
            raise StructureTensorStudyError("frame has duplicate sample_id values")
        missing_columns = [
            column for column in ("sample_id", TARGET) if column not in frame.columns
        ]
        if missing_columns:  # pragma: no cover - structural guard
            raise StructureTensorStudyError(
                f"frame is missing required column(s) {missing_columns}"
            )
        merged = frame.merge(
            self.structure_features, on="sample_id", how="left", validate="one_to_one"
        )
        missing = merged[self.structure_features.columns[1:]].isna().any().any()
        if missing:
            raise StructureTensorStudyError(
                "some rows are missing extracted structure-tensor features"
            )
        return merged


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise StructureTensorStudyError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise StructureTensorStudyError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def _feature_module_source() -> str:
    """Return the source of the structure-tensor feature module for auditing."""
    return Path(stf.__file__).read_text(encoding="utf-8")


def _forbidden_identifiers_in_feature_module() -> list[str]:
    """Return forbidden identifiers that appear in the feature module's code.

    Only code identifiers are considered; docstrings and comments are ignored,
    because prose that *explains* the boundary is not a data path. The scan is
    intentionally conservative: any variable, argument or attribute name that
    matches a target or metadata field fails the audit.
    """
    tree = ast.parse(_feature_module_source())
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.arg):
            used.add(node.arg)
        elif isinstance(node, ast.alias):
            used.add((node.asname or node.name).split(".")[-1])
    return sorted(_FORBIDDEN_FEATURE_IDENTIFIERS & used)


def _check_no_reference_signature() -> CheckResult:
    """Confirm the extractor accepts one image and no other data source."""
    try:
        signature = inspect.signature(stf.extract_structure_tensor_features)
        parameters = list(signature.parameters.values())
        names = tuple(parameter.name for parameter in parameters)
        if names != ("image", "spec"):
            raise StructureTensorStudyError(
                f"the extractor signature must be (image, spec), got {names}"
            )
        for parameter in parameters:
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise StructureTensorStudyError(
                    f"the extractor must not accept *args/**kwargs, found {parameter.name!r}"
                )
        required = [
            parameter
            for parameter in parameters
            if parameter.default is inspect.Parameter.empty
            and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        ]
        if [parameter.name for parameter in required] != ["image"]:
            raise StructureTensorStudyError(
                "the only required extractor input must be the corrupted image array"
            )
        return CheckResult(
            name="no_reference_signature",
            passed=True,
            detail=(
                "extract_structure_tensor_features(image, spec=None) accepts exactly one "
                "required input, the corrupted image array, and no *args/**kwargs"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="no_reference_signature",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_no_reference_identifiers() -> CheckResult:
    """Confirm the feature module's code cannot reach a target or metadata field."""
    try:
        forbidden = _forbidden_identifiers_in_feature_module()
        if forbidden:
            raise StructureTensorStudyError(
                "the structure-tensor feature module's code uses forbidden "
                f"identifier(s): {forbidden}"
            )
        return CheckResult(
            name="no_reference_identifiers",
            passed=True,
            detail=(
                "the structure-tensor feature module's code contains no target, "
                "reference, source, "
                "severity, family, font, split or path identifier; those names appear "
                "only in prose explaining the boundary"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="no_reference_identifiers",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_structure_feature_frame(
    features: pd.DataFrame, spec: stf.StructureTensorSpec
) -> CheckResult:
    """Confirm the extracted table has the exact declared schema and is finite."""
    try:
        expected = ("sample_id", *spec.feature_names)
        if tuple(features.columns) != expected:
            raise StructureTensorStudyError(
                f"structure-tensor feature columns must be exactly {list(expected)}, got "
                f"{list(features.columns)}"
            )
        if features["sample_id"].duplicated().any():
            raise StructureTensorStudyError(
                "structure-tensor feature table has duplicate sample_id values"
            )
        values = features[list(spec.feature_names)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise StructureTensorStudyError("structure-tensor features contain non-finite values")
        bounded = [name for name in spec.feature_names if "_coh_" in name or "_aniso_" in name]
        if bounded:
            summaries = features[bounded].to_numpy(dtype=np.float64)
            if (summaries < 0.0).any() or (summaries > 1.0).any():
                raise StructureTensorStudyError(
                    "coherence and anisotropy features must lie inside [0, 1]"
                )
        energies = [name for name in spec.feature_names if "_energy_" in name]
        if energies:
            energy_values = features[energies].to_numpy(dtype=np.float64)
            if (energy_values < 0.0).any():
                raise StructureTensorStudyError("tensor-energy features must be non-negative")
        return CheckResult(
            name="structure_tensor_feature_schema",
            passed=True,
            detail=(
                f"{len(features)} rows with {len(spec.feature_names)} structure-tensor "
                "features in the declared order, all finite, coherence/anisotropy inside "
                "[0, 1] and tensor energy non-negative"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="structure_tensor_feature_schema",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_no_metadata_in_structure_inputs(
    features: pd.DataFrame, spec: stf.StructureTensorSpec
) -> CheckResult:
    """Confirm the structure-tensor table carries no target or metadata column."""
    try:
        forbidden = {
            TARGET,
            "psnr_target",
            "corruption_type",
            "severity_level",
            "font_id",
            "text_length",
            "source_id",
            "split",
            "image_path",
            "reference_path",
            "variant_family",
            "family",
            "amplitude_px",
            "protocol_scope",
            "is_clean",
            "is_control",
            "is_elastic",
        }
        present = sorted(forbidden & set(features.columns))
        if present:
            raise StructureTensorStudyError(
                f"forbidden columns present in the structure-tensor feature table: {present}"
            )
        if len(spec.feature_names) != len(set(spec.feature_names)):
            raise StructureTensorStudyError("duplicate structure-tensor feature names")
        return CheckResult(
            name="structure_tensor_feature_leakage",
            passed=True,
            detail=(
                "the structure-tensor feature table contains only sample_id and the "
                "declared image features; no target, severity, family, font, source, "
                "split or path column is present"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="structure_tensor_feature_leakage",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _verify_pinned_upstream(config: StructureTensorConfig) -> ra.RetentionConfig:
    """Verify the M5C, M6A, M6B and M6C pins and return the M5C configuration."""
    _require_file_hash(config.m5c.config, config.m5c.config_sha256, "M5C config")
    m5c_config = ra.load_retention_config(config.m5c.config)
    if m5c_config.config_hash != config.m5c.protocol_hash:
        raise StructureTensorStudyError(
            "M5C protocol hash mismatch: expected "
            f"{config.m5c.protocol_hash}, found {m5c_config.config_hash}"
        )
    _require_file_hash(config.m6a.config, config.m6a.config_sha256, "M6A config")
    m6a_config = ss.load_spatial_study_config(config.m6a.config)
    if m6a_config.config_hash != config.m6a.protocol_hash:
        raise StructureTensorStudyError(
            "M6A protocol hash mismatch: expected "
            f"{config.m6a.protocol_hash}, found {m6a_config.config_hash}"
        )
    _require_file_hash(config.m6b.config, config.m6b.config_sha256, "M6B config")
    m6b_config = srs.load_resolution_config(config.m6b.config)
    if m6b_config.config_hash != config.m6b.protocol_hash:
        raise StructureTensorStudyError(
            "M6B protocol hash mismatch: expected "
            f"{config.m6b.protocol_hash}, found {m6b_config.config_hash}"
        )
    _require_file_hash(config.m6c.config, config.m6c.config_sha256, "M6C config")
    m6c_config = lss.load_local_structure_config(config.m6c.config)
    if m6c_config.config_hash != config.m6c.protocol_hash:
        raise StructureTensorStudyError(
            "M6C protocol hash mismatch: expected "
            f"{config.m6c.protocol_hash}, found {m6c_config.config_hash}"
        )
    return m5c_config


def load_structure_tensor_frames(
    config: StructureTensorConfig, *, verify_images: bool = True
) -> StructureTensorFrames:
    """Load the frozen frames through M5C and extract structure-tensor features."""
    m5c_config = _verify_pinned_upstream(config)
    retention = ra.load_retention_frames(m5c_config, verify_images=verify_images)
    roots = ss._dataset_roots(m5c_config)  # noqa: SLF001

    # Every row that will be fitted or scored needs its structure-tensor
    # features. The
    # extraction is deterministic and receives only the corrupted image array.
    needed = (
        retention.expanded_train,
        retention.expanded_val,
        retention.m3_test,
        retention.primary_test,
        retention.diagnostic_test,
    )
    samples: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for frame in needed:
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            if sample_id in seen:
                continue
            seen.add(sample_id)
            dataset = str(row.dataset)
            root = roots.get(dataset)
            if root is None:
                raise StructureTensorStudyError(f"no image root configured for dataset {dataset!r}")
            samples.append((sample_id, load_grayscale(root / str(row.image_path))))
    structure_values = stf.extract_structure_tensor_feature_table(samples, config.features)

    checks = (
        *retention.checks,
        _check_no_reference_signature(),
        _check_no_reference_identifiers(),
        _check_structure_feature_frame(structure_values, config.features),
        _check_no_metadata_in_structure_inputs(structure_values, config.features),
    )
    rb._verify_checks(checks, "M6D frame checks")  # noqa: SLF001
    return StructureTensorFrames(
        retention=retention, structure_features=structure_values, checks=checks
    )


def _champion_spec(config: StructureTensorConfig, m5c_config: ra.RetentionConfig) -> CandidateSpec:
    """Return the champion's grid specification and confirm it matches the config."""
    m5b_config = ra._load_m5b_config(m5c_config)  # noqa: SLF001
    specs = rb.candidate_specs_from_config(m5b_config)  # type: ignore[arg-type]
    spec = next(
        (item for item in specs if item.name == m5c_config.candidates.alternative_spec), None
    )
    if spec is None:
        raise StructureTensorStudyError(
            f"the M5C champion specification {m5c_config.candidates.alternative_spec!r} is "
            "not in the M5B grid"
        )
    if spec.family != config.model.family:
        raise StructureTensorStudyError(
            f"model.family is {config.model.family!r} but the champion was fitted from a "
            f"{spec.family!r} specification"
        )
    for key, value in (
        ("n_estimators", config.model.n_estimators),
        ("max_depth", config.model.max_depth),
        ("min_samples_leaf", config.model.min_samples_leaf),
    ):
        if spec.params.get(key) != value:
            raise StructureTensorStudyError(
                f"model.{key} is {value!r} but the M5B grid specification has "
                f"{spec.params.get(key)!r}"
            )
    seed = derive_noise_seed(m5b_config.run.seed, "model", spec.name) % (2**32)
    if seed != config.model.seed:
        raise StructureTensorStudyError(
            f"model.seed is {config.model.seed} but the M5B grid derives {seed}"
        )
    return spec


def _load_champion(config: StructureTensorConfig) -> Any:
    """Load the frozen M5C champion bundle and confirm it is unchanged."""
    directory = config.m5c.champion_dir
    stem = config.m5c.champion_stem
    artifact = directory / f"{stem}.joblib"
    card_path = directory / f"{stem}_card.json"
    _require_file_hash(artifact, config.m5c.champion_artifact_sha256, "M5C champion artifact")
    if not card_path.is_file():
        raise StructureTensorStudyError(f"missing M5C champion card: {card_path}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("feature_names") != list(config.global_feature_names):
        raise StructureTensorStudyError("the M5C champion uses a different feature schema")
    if card.get("target") != config.target:
        raise StructureTensorStudyError(f"the M5C champion target is not {config.target!r}")
    recorded = str(card.get("selection", {}).get("benchmark_config_hash", ""))
    if recorded != config.m5c.protocol_hash:
        raise StructureTensorStudyError(
            f"the M5C champion was produced under protocol {recorded!r}, not the pinned "
            f"{config.m5c.protocol_hash!r}"
        )
    if card.get("artifact_sha256") != sha256_file(artifact):
        raise StructureTensorStudyError("the M5C champion artifact hash does not match its card")
    if str(card.get("sklearn_version", "")).split(".")[:2] != sklearn_version().split(".")[:2]:
        raise StructureTensorStudyError(
            f"the M5C champion was persisted with scikit-learn {card.get('sklearn_version')}, "
            f"current is {sklearn_version()}"
        )
    import joblib

    estimator = joblib.load(artifact)
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features != len(config.global_feature_names):
        raise StructureTensorStudyError(
            f"the M5C champion expects {n_features} inputs, the frozen schema has "
            f"{len(config.global_feature_names)}"
        )
    return estimator


def build_candidates(
    frames: StructureTensorFrames,
    config: StructureTensorConfig,
    *,
    champion: Any,
    m5c_config: ra.RetentionConfig,
) -> tuple[ss.Ablation, ...]:
    """Fit every predeclared candidate on the expanded training rows.

    The model family, hyperparameters, seed, training rows and objective are
    identical for the two refitted candidates; only the input representation
    differs. The champion is loaded, never refitted.
    """
    spec = _champion_spec(config, m5c_config)
    train = frames.with_structure(frames.retention.expanded_train)
    candidates: list[ss.Ablation] = []
    for name in config.ablations.names:
        feature_set = config.ablations.feature_sets[name]
        feature_names = config.feature_names_for(feature_set)
        if not config.ablations.refit[name]:
            candidates.append(
                ss.Ablation(
                    name=name,
                    feature_set=feature_set,
                    feature_names=feature_names,
                    estimator=champion,
                    refitted=False,
                    train_importances=None,
                )
            )
            continue
        estimator = spec.build(seed=config.model.seed)
        inputs = train[list(feature_names)].to_numpy(dtype=np.float64)
        target = train[TARGET].to_numpy(dtype=np.float64)
        estimator.fit(inputs, target)
        importances = getattr(estimator, "feature_importances_", None)
        candidates.append(
            ss.Ablation(
                name=name,
                feature_set=feature_set,
                feature_names=feature_names,
                estimator=estimator,
                refitted=True,
                train_importances=(
                    np.asarray(importances, dtype=np.float64) if importances is not None else None
                ),
            )
        )
    return tuple(candidates)


# ---------------------------------------------------------------------------
# Validation scoring and selection
# ---------------------------------------------------------------------------


def validation_slices(frames: StructureTensorFrames) -> dict[str, pd.DataFrame]:
    """Return the predeclared validation slices, unchanged from M6A/M6B."""
    validation = frames.with_structure(frames.retention.expanded_val)
    m5a = validation[validation["dataset"] == gb.M5A_LABEL]
    m3 = validation[validation["dataset"] == gb.M3_LABEL]
    slices: dict[str, pd.DataFrame] = {
        GEO_LABEL: m5a,
        CLEAN_LABEL: m3[m3["is_clean"]],
        ORIGINAL_LABEL: m3[~m3["is_clean"]],
        OVERALL_LABEL: validation,
    }
    for severity, subset in m5a.groupby("severity_level", sort=True):
        slices[f"{GEO_LABEL}_severity_{int(severity)}"] = subset
    return slices


def _severity_scopes(config: StructureTensorConfig) -> tuple[str, ...]:
    return tuple(
        f"{GEO_LABEL}_severity_{severity}" for severity in config.protocol.primary_severities
    )


def _slice_metrics_table(
    frames: StructureTensorFrames,
    ablations: Sequence[ss.Ablation],
    predictions: Mapping[str, np.ndarray],
    *,
    split_label: str,
) -> pd.DataFrame:
    """Score every ablation on every validation slice."""
    validation = frames.with_structure(frames.retention.expanded_val)
    slices = validation_slices(frames)
    rows: list[dict[str, Any]] = []
    for ablation in ablations:
        predicted = np.asarray(predictions[ablation.name], dtype=np.float64)
        if predicted.size != len(validation):
            raise StructureTensorStudyError(
                f"predictions for {ablation.name!r} do not align with the validation frame"
            )
        for scope, subset in slices.items():
            if subset.empty:
                continue
            mask = validation.index.get_indexer(subset.index)
            scores = ra._slice_scores(subset, predicted[mask])  # noqa: SLF001
            rows.append(
                {
                    "model": ablation.name,
                    "feature_set": ablation.feature_set,
                    "n_features": len(ablation.feature_names),
                    "split": split_label,
                    "scope": scope,
                    **scores,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------


def build_test_slices(
    frames: StructureTensorFrames, ablations: Sequence[ss.Ablation]
) -> list[ss.TestSlice]:
    """Assemble the frozen evaluation slices scored after selection is complete."""
    retention = frames.retention
    geometric = frames.with_structure(retention.primary_test)
    m3_test = frames.with_structure(retention.m3_test)
    original = m3_test[~m3_test["is_clean"]]
    diagnostic = frames.with_structure(retention.diagnostic_test)

    def predict_on(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {ablation.name: ablation.predict(frame) for ablation in ablations}

    slices: list[ss.TestSlice] = [
        ss.TestSlice(GEO_LABEL, geometric, predict_on(geometric)),
        ss.TestSlice(
            CLEAN_LABEL, m3_test[m3_test["is_clean"]], predict_on(m3_test[m3_test["is_clean"]])
        ),
        ss.TestSlice(ORIGINAL_LABEL, original, predict_on(original)),
        ss.TestSlice("m3_test_overall", m3_test, predict_on(m3_test)),
        ss.TestSlice("diagnostic_0_5", diagnostic, predict_on(diagnostic)),
    ]
    for severity, subset in geometric.groupby("severity_level", sort=True):
        slices.append(
            ss.TestSlice(f"{GEO_LABEL}_severity_{int(severity)}", subset, predict_on(subset))
        )
    for family, subset in original.groupby("family", sort=True):
        slices.append(ss.TestSlice(f"{ORIGINAL_LABEL}_{family}", subset, predict_on(subset)))
    return [item for item in slices if not item.frame.empty]


def _test_metrics_table(slices: Sequence[ss.TestSlice]) -> pd.DataFrame:
    """Score every ablation on every frozen test slice."""
    rows: list[dict[str, Any]] = []
    for item in slices:
        for name, predicted in item.predictions.items():
            scores = ra._slice_scores(item.frame, predicted)  # noqa: SLF001
            rows.append({"model": name, "split": "test", "scope": item.scope, **scores})
    return pd.DataFrame(rows)


def _uncertainty_table(
    slices: Sequence[ss.TestSlice], *, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    """Source-grouped bootstrap intervals for every ablation and slice."""
    rows: list[pd.DataFrame] = []
    for item in slices:
        for name, predicted in item.predictions.items():
            table = gb._bootstrap_row(  # noqa: SLF001
                item.frame,
                predicted,
                model=name,
                dataset=item.scope,
                scope="overall",
                level="all",
                n_bootstrap=n_bootstrap,
                seed=derive_noise_seed(seed, "m6d_bootstrap", name, item.scope) % (2**64),
                evaluation="m6d_test",
            )
            rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _paired_uncertainty_table(
    slices: Sequence[ss.TestSlice],
    *,
    n_bootstrap: int,
    seed: int,
    reference: str = GLOBAL_BASELINE,
) -> pd.DataFrame:
    """Paired source-grouped differences against the frozen baseline."""
    rows: list[pd.DataFrame] = []
    for item in slices:
        if reference not in item.predictions:
            continue
        reference_predictions = item.predictions[reference]
        for name, predicted in item.predictions.items():
            if name == reference:
                continue
            table = gb._paired_row(  # noqa: SLF001
                item.frame,
                predicted,
                reference_predictions,
                model_a=name,
                model_b=reference,
                dataset=item.scope,
                scope="overall",
                level="all",
                n_bootstrap=n_bootstrap,
                seed=derive_noise_seed(seed, "m6d_paired", name, item.scope) % (2**64),
                evaluation="m6d_test",
            )
            rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _prediction_table(slices: Sequence[ss.TestSlice], model_names: Sequence[str]) -> pd.DataFrame:
    """Row-level test predictions for the requested ablations."""
    frames: list[pd.DataFrame] = []
    for item in slices:
        for name in model_names:
            if name not in item.predictions:
                continue
            working = item.frame.copy()
            working["model"] = name
            working["scope"] = item.scope
            working["split"] = "test"
            working["prediction"] = item.predictions[name]
            working["residual"] = working[TARGET] - working["prediction"]
            working["abs_residual"] = working["residual"].abs()
            frames.append(working)
    if not frames:
        return pd.DataFrame()
    columns = [
        "model",
        "scope",
        "split",
        "sample_id",
        "source_id",
        "font_id",
        "text_length",
        "family",
        "severity_level",
        TARGET,
        "prediction",
        "residual",
        "abs_residual",
    ]
    return pd.concat(frames, ignore_index=True)[columns]


# ---------------------------------------------------------------------------
# Training-only diagnostics
# ---------------------------------------------------------------------------


def _duplicate_columns(values: pd.DataFrame) -> list[str]:
    """Return ``name==other`` labels for exactly duplicated numeric columns."""
    seen: dict[bytes, str] = {}
    duplicates: list[str] = []
    for name in values.columns:
        payload = np.ascontiguousarray(values[name].to_numpy(dtype=np.float64)).tobytes()
        if payload in seen:
            duplicates.append(f"{name}=={seen[payload]}")
        else:
            seen[payload] = name
    return duplicates


def _correlation_summary(values: pd.DataFrame) -> tuple[float, float]:
    """Return the mean and maximum absolute off-diagonal correlation."""
    variable = [name for name in values.columns if float(values[name].std(ddof=0)) > 0.0]
    if len(variable) < 2:
        return 0.0, 0.0
    correlation = values[variable].corr().abs().to_numpy(dtype=np.float64)
    mask = ~np.eye(len(variable), dtype=bool)
    off_diagonal = correlation[mask]
    off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
    if off_diagonal.size == 0:
        return 0.0, 0.0
    return float(np.mean(off_diagonal)), float(np.max(off_diagonal))


def _correlation_pairs(values: pd.DataFrame, top_n: int = 5) -> list[dict[str, object]]:
    """Return the strongest absolute off-diagonal feature pairs."""
    variable = [name for name in values.columns if float(values[name].std(ddof=0)) > 0.0]
    if len(variable) < 2:
        return []
    correlation = values[variable].corr().abs()
    pairs: list[dict[str, object]] = []
    for index, left in enumerate(variable):
        for right in variable[index + 1 :]:
            value = float(correlation.loc[left, right])
            if math.isfinite(value):
                pairs.append({"feature_a": left, "feature_b": right, "abs_correlation": value})
    pairs.sort(key=lambda item: (-float(item["abs_correlation"]), str(item["feature_a"])))
    return pairs[:top_n]


def _cross_block_correlation(
    values: pd.DataFrame, left: Sequence[str], right: Sequence[str]
) -> float:
    """Return the maximum absolute correlation between two feature blocks."""
    left_variable = [name for name in left if float(values[name].std(ddof=0)) > 0.0]
    right_variable = [name for name in right if float(values[name].std(ddof=0)) > 0.0]
    if not left_variable or not right_variable:
        return 0.0
    left_values = values[left_variable].to_numpy(dtype=np.float64)
    right_values = values[right_variable].to_numpy(dtype=np.float64)
    correlations: list[float] = []
    for index in range(left_values.shape[1]):
        for other in range(right_values.shape[1]):
            a = left_values[:, index]
            b = right_values[:, other]
            if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
                continue
            correlations.append(abs(float(np.corrcoef(a, b)[0, 1])))
    finite = [value for value in correlations if math.isfinite(value)]
    return float(max(finite)) if finite else 0.0


def _feature_family(name: str, global_names: Sequence[str]) -> str:
    if name in set(global_names):
        return "global"
    if name.startswith("st_"):
        return stf.FAMILY_STRUCTURE_TENSOR
    raise StructureTensorStudyError(f"unrecognised feature name {name!r}")


def training_diagnostics(
    frames: StructureTensorFrames,
    ablations: Sequence[ss.Ablation],
    validation_metrics: pd.DataFrame,
    config: StructureTensorConfig,
) -> dict[str, Any]:
    """Collect descriptive diagnostics from the training rows only.

    Nothing here is used to redesign the feature set within this milestone: the
    diagnostics are reported, not acted upon. In particular the feature set is
    not pruned, reweighted or re-binned in response to any value below.
    """
    train = frames.with_structure(frames.retention.expanded_train)
    structure_names = list(config.features.feature_names)
    global_names = list(config.global_feature_names)
    structure_values = train[structure_names]

    constant_structure = [
        name for name in structure_names if float(structure_values[name].std(ddof=0)) == 0.0
    ]
    duplicated_structure = _duplicate_columns(structure_values)
    non_finite_structure = [
        name
        for name in structure_names
        if not np.isfinite(structure_values[name].to_numpy(dtype=np.float64)).all()
    ]
    structure_correlation_mean, structure_correlation_max = _correlation_summary(structure_values)
    structure_correlation_top_pairs = _correlation_pairs(structure_values)

    candidates: dict[str, Any] = {}
    for ablation in ablations:
        values = train[list(ablation.feature_names)]
        correlation_mean, correlation_max = _correlation_summary(values)
        constant = [
            name for name in ablation.feature_names if float(values[name].std(ddof=0)) == 0.0
        ]
        duplicated = _duplicate_columns(values)
        non_finite = [
            name
            for name in ablation.feature_names
            if not np.isfinite(values[name].to_numpy(dtype=np.float64)).all()
        ]

        structure_in_model = [
            name for name in ablation.feature_names if name in set(structure_names)
        ]
        global_in_model = [name for name in ablation.feature_names if name in set(global_names)]
        cross_block = (
            _cross_block_correlation(values, structure_in_model, global_in_model)
            if structure_in_model and global_in_model
            else 0.0
        )

        train_predictions = ablation.predict(train)
        train_scores = ra._slice_scores(train, train_predictions)  # noqa: SLF001
        geometric_train = train[train["dataset"] == gb.M5A_LABEL]
        geometric_train_scores = ra._slice_scores(  # noqa: SLF001
            geometric_train, ablation.predict(geometric_train)
        )
        validation_row = validation_metrics[
            (validation_metrics["model"] == ablation.name)
            & (validation_metrics["scope"] == GEO_LABEL)
        ]
        if validation_row.empty:
            raise StructureTensorStudyError(
                f"ablation {ablation.name!r} has no validation geometric metric"
            )
        validation_rmse = float(validation_row.iloc[0]["rmse"])

        importances: list[dict[str, Any]] = []
        importance_global = 0.0
        importance_structure = 0.0
        if ablation.train_importances is not None:
            order = np.argsort(-ablation.train_importances, kind="stable")
            for rank, index in enumerate(order, start=1):
                name = ablation.feature_names[int(index)]
                value = float(ablation.train_importances[int(index)])
                family = _feature_family(name, global_names)
                if family == "global":
                    importance_global += value
                else:
                    importance_structure += value
                importances.append(
                    {
                        "rank": rank,
                        "feature_name": name,
                        "family": family,
                        "importance": value,
                    }
                )
        total_importance = importance_global + importance_structure

        candidates[ablation.name] = {
            "feature_set": ablation.feature_set,
            "n_features": len(ablation.feature_names),
            "n_global_features": len(global_in_model),
            "n_structure_features": len(structure_in_model),
            "constant_features": constant,
            "duplicated_features": duplicated,
            "non_finite_features": non_finite,
            "correlation_mean": correlation_mean,
            "correlation_max": correlation_max,
            "structure_vs_global_max_abs_correlation": cross_block,
            "train_rmse_overall": float(train_scores["rmse"]),
            "train_rmse_geometric": float(geometric_train_scores["rmse"]),
            "validation_rmse_geometric": validation_rmse,
            "overfit_gap_geometric_rmse": float(validation_rmse - geometric_train_scores["rmse"]),
            "importance_global_share": (
                float(importance_global / total_importance) if total_importance > 0.0 else None
            ),
            "importance_structure_share": (
                float(importance_structure / total_importance) if total_importance > 0.0 else None
            ),
            "feature_importances": importances,
        }

    return {
        "n_train_rows": int(len(train)),
        "n_global_features": len(global_names),
        "n_structure_features": len(structure_names),
        "structure_tensor_feature_set_version": stf.STRUCTURE_TENSOR_FEATURE_SET_VERSION,
        "structure_tensor_feature_names": structure_names,
        "constant_structure_features": constant_structure,
        "duplicated_structure_features": duplicated_structure,
        "non_finite_structure_features": non_finite_structure,
        "structure_correlation_mean": structure_correlation_mean,
        "structure_correlation_max": structure_correlation_max,
        "structure_correlation_top_pairs": structure_correlation_top_pairs,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructureTensorResult:
    """The complete M6D outcome."""

    frames: StructureTensorFrames
    checks: tuple[CheckResult, ...]
    ablations: tuple[ss.Ablation, ...]
    validation_metrics: pd.DataFrame
    selection_table: pd.DataFrame
    selected_name: str
    test_metrics: pd.DataFrame
    uncertainty: pd.DataFrame
    paired_uncertainty: pd.DataFrame
    test_predictions: pd.DataFrame
    diagnostics: dict[str, Any]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def selected(self) -> ss.Ablation:
        for ablation in self.ablations:
            if ablation.name == self.selected_name:
                return ablation
        raise StructureTensorStudyError(f"selected ablation {self.selected_name!r} is missing")


def run_structure_tensor_core(
    frames: StructureTensorFrames,
    config: StructureTensorConfig,
    *,
    champion: Any,
    m5c_config: ra.RetentionConfig,
) -> StructureTensorResult:
    """Run the M6D study in the predeclared order.

    The order is fixed: fit on training data only, score validation, select,
    freeze, and only then score the untouched test slices.
    """
    rb._verify_checks(frames.checks, "M6D frame checks")  # noqa: SLF001

    ablations = build_candidates(frames, config, champion=champion, m5c_config=m5c_config)
    validation = frames.with_structure(frames.retention.expanded_val)
    validation_predictions = {ablation.name: ablation.predict(validation) for ablation in ablations}
    validation_metrics = _slice_metrics_table(
        frames, ablations, validation_predictions, split_label="val"
    )
    selected_name, selection_table = ss.select_ablation(
        validation_metrics,
        config.selection,
        severity_scopes=_severity_scopes(config),
    )

    # The selected ablation and its complete configuration are frozen here.
    # The test slices are only touched after this point.
    slices = build_test_slices(frames, ablations)
    test_metrics = _test_metrics_table(slices)
    uncertainty = _uncertainty_table(
        slices, n_bootstrap=config.run.n_bootstrap, seed=config.run.seed
    )
    paired = _paired_uncertainty_table(
        slices, n_bootstrap=config.run.n_bootstrap, seed=config.run.seed
    )
    predictions = _prediction_table(slices, [a.name for a in ablations])
    diagnostics = training_diagnostics(frames, ablations, validation_metrics, config)

    return StructureTensorResult(
        frames=frames,
        checks=frames.checks,
        ablations=ablations,
        validation_metrics=validation_metrics,
        selection_table=selection_table,
        selected_name=selected_name,
        test_metrics=test_metrics,
        uncertainty=uncertainty,
        paired_uncertainty=paired,
        test_predictions=predictions,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _ablation_model_card(
    result: StructureTensorResult,
    config: StructureTensorConfig,
    ablation: ss.Ablation,
) -> dict[str, Any]:
    """Build the model card for one refitted candidate."""
    metrics = result.test_metrics[result.test_metrics["model"] == ablation.name]
    metric_payload: dict[str, Any] = {}
    for row in metrics.itertuples(index=False):
        metric_payload[str(row.scope)] = {
            "mae": float(row.mae),
            "rmse": float(row.rmse),
            "r2": float(row.r2),
            "r2_defined": bool(row.r2_defined),
            "n_rows": int(row.n_samples),
            "n_sources": int(row.n_sources),
            "mean_residual": float(row.mean_residual),
            "sd_residual": float(row.sd_residual),
        }
    validation = result.validation_metrics[result.validation_metrics["model"] == ablation.name]
    validation_payload: dict[str, Any] = {}
    for row in validation.itertuples(index=False):
        validation_payload[str(row.scope)] = {
            "mae": float(row.mae),
            "rmse": float(row.rmse),
            "r2": float(row.r2),
            "n_rows": int(row.n_samples),
            "n_sources": int(row.n_sources),
        }
    return {
        "model_name": ablation.name,
        "model_family": config.model.family,
        "feature_set": ablation.feature_set,
        "n_features": len(ablation.feature_names),
        "feature_names": list(ablation.feature_names),
        "structure_tensor_feature_set_version": stf.STRUCTURE_TENSOR_FEATURE_SET_VERSION,
        "structure_features": config.features.identity_payload(),
        "params": {
            "n_estimators": config.model.n_estimators,
            "max_depth": config.model.max_depth,
            "min_samples_leaf": config.model.min_samples_leaf,
        },
        "seed": config.model.seed,
        "refitted": ablation.refitted,
        "preprocessing": "none",
        "feature_set_version": config.global_feature_set_version,
        "target": config.target,
        "package_version": package_version(),
        "sklearn_version": sklearn_version(),
        "code_revision": code_revision(),
        "selection": {
            "rule": config.selection.rule,
            "selected_on": "val",
            "selected_model": result.selected_name,
            "n_ablations": len(result.ablations),
            "benchmark_config_hash": config.config_hash,
        },
        "validation_metrics": validation_payload,
        "metrics": metric_payload,
    }


def write_structure_tensor_outputs(
    result: StructureTensorResult,
    config: StructureTensorConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    frozen_before: Mapping[str, Mapping[str, object]] | None = None,
    frozen_after: Mapping[str, Mapping[str, object]] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Path]:
    """Persist the M6D tables, model bundles, metadata and report."""
    reports = Path(output_reports) if output_reports is not None else config.run.output_reports
    models = Path(output_models) if output_models is not None else config.run.output_models
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    result.validation_metrics.to_csv(reports / "validation_slice_metrics.csv", index=False)
    result.selection_table.to_csv(reports / "selection_table.csv", index=False)
    result.test_metrics.to_csv(reports / "test_slice_metrics.csv", index=False)
    result.uncertainty.to_csv(reports / "uncertainty.csv", index=False)
    result.paired_uncertainty.to_csv(reports / "paired_uncertainty.csv", index=False)
    result.test_predictions.to_csv(reports / "test_predictions.csv", index=False)
    pd.DataFrame(
        [
            {"check": check.name, "passed": check.passed, "detail": check.detail}
            for check in result.checks
        ]
    ).to_csv(reports / "isolation_checks.csv", index=False)
    stf.structure_tensor_manifest(config.features).to_csv(
        reports / "structure_tensor_manifest.csv", index=False
    )
    paths.update(
        {
            "validation_metrics": reports / "validation_slice_metrics.csv",
            "selection_table": reports / "selection_table.csv",
            "test_metrics": reports / "test_slice_metrics.csv",
            "uncertainty": reports / "uncertainty.csv",
            "paired_uncertainty": reports / "paired_uncertainty.csv",
            "test_predictions": reports / "test_predictions.csv",
            "isolation_checks": reports / "isolation_checks.csv",
            "feature_manifest": reports / "structure_tensor_manifest.csv",
        }
    )

    for ablation in result.ablations:
        if not ablation.refitted:
            continue
        saved = save_model_bundle(
            models,
            ablation.name,
            ablation.estimator,
            _ablation_model_card(result, config, ablation),
        )
        paths[f"{ablation.name}_artifact"] = saved["artifact"]
        paths[f"{ablation.name}_card"] = saved["card"]

    metadata = {
        "benchmark_config": config.identity_payload(),
        "benchmark_config_hash": config.config_hash,
        "benchmark_config_sha256": config.config_sha256,
        "schema": {
            "global_feature_set_version": config.global_feature_set_version,
            "global_feature_names": list(config.global_feature_names),
            "structure_tensor_feature_set_version": stf.STRUCTURE_TENSOR_FEATURE_SET_VERSION,
            "structure_tensor_feature_names": list(config.features.feature_names),
            "target": config.target,
            "split_names": ["train", "val", "test"],
        },
        "no_reference_boundary": {
            "extractor": "extract_structure_tensor_features(image, spec=None)",
            "required_inputs": ["image"],
            "clean_reference_used_for_features": False,
            "target_or_metadata_inputs": [],
            "checks": {
                check.name: {"passed": check.passed, "detail": check.detail}
                for check in result.checks
                if check.name.startswith("no_reference")
            },
        },
        "runtime": {
            "code_revision": code_revision(),
            "package_version": package_version(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn_version(),
            "elapsed_seconds_in_process": (
                round(float(elapsed_seconds), 6) if elapsed_seconds is not None else None
            ),
        },
        "split": {
            "train_sources": int(result.frames.retention.expanded_train["source_id"].nunique()),
            "val_sources": int(result.frames.retention.expanded_val["source_id"].nunique()),
            "test_sources": int(result.frames.retention.primary_test["source_id"].nunique()),
            "train_samples": int(len(result.frames.retention.expanded_train)),
            "validation_samples": int(len(result.frames.retention.expanded_val)),
            "primary_test_samples": int(len(result.frames.retention.primary_test)),
            "diagnostic_test_samples": int(len(result.frames.retention.diagnostic_test)),
            "m3_test_samples": int(len(result.frames.retention.m3_test)),
        },
        "ablations": {
            ablation.name: {
                "feature_set": ablation.feature_set,
                "n_features": len(ablation.feature_names),
                "feature_names": list(ablation.feature_names),
                "refitted": ablation.refitted,
            }
            for ablation in result.ablations
        },
        "selection": {
            "rule": config.selection.rule,
            "selected_model": result.selected_name,
            "n_ablations": len(result.ablations),
            "selected_on": "val",
        },
        "diagnostics": result.diagnostics,
        "frozen_assets": {
            "before": dict(frozen_before or {}),
            "after": dict(frozen_after or {}),
            "all_unchanged": ss._frozen_unchanged(  # noqa: SLF001
                frozen_before, frozen_after
            ),
        },
        "artifacts": {},
    }
    for key, path in paths.items():
        if path.is_file():
            metadata["artifacts"][key] = {
                "name": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
            }
    metadata_path = reports / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    paths["metadata"] = metadata_path

    report_path = reports / "report.md"
    report_path.write_text(_report_markdown(result, config), encoding="utf-8")
    paths["report"] = report_path
    return paths


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_FEATURE_DEFINITIONS: Final[dict[str, str]] = {
    "st_s1_coh_mean": "mean local coherence at scale 1 (sigma = 1.0 px), in [0, 1]",
    "st_s1_coh_p90": "0.90 quantile of local coherence at scale 1, in [0, 1]",
    "st_s1_aniso_mean": "mean local anisotropy at scale 1, in [0, 1]",
    "st_s1_energy_p90": "0.90 quantile of local tensor energy at scale 1",
    "st_s2_coh_mean": "mean local coherence at scale 2 (sigma = 3.0 px), in [0, 1]",
    "st_s2_coh_p90": "0.90 quantile of local coherence at scale 2, in [0, 1]",
    "st_s2_aniso_mean": "mean local anisotropy at scale 2, in [0, 1]",
    "st_s2_energy_p90": "0.90 quantile of local tensor energy at scale 2",
    "st_energy_mean": "mean local tensor energy, reported once at the finest scale",
}


def _feature_definition(name: str) -> str:
    if name in _FEATURE_DEFINITIONS:
        return _FEATURE_DEFINITIONS[name]
    raise StructureTensorStudyError(f"no report definition for feature {name!r}")


def _manifest_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    return str(value)


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    float_digits: int = 5,
) -> list[str]:
    """Render a deterministic Markdown table with a fixed column order."""
    if frame.empty:
        return ["_No rows._", ""]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in frame[list(columns)].itertuples(index=False):
        cells: list[str] = []
        for value in row:
            if isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, (float, np.floating)):
                if not math.isfinite(float(value)):
                    cells.append("n/a")
                else:
                    cells.append(f"{float(value):.{float_digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _scope_order(scope: str) -> tuple[int, str]:
    if scope == GEO_LABEL:
        return (0, scope)
    if scope.startswith(f"{GEO_LABEL}_severity_"):
        return (1, scope)
    if scope == CLEAN_LABEL:
        return (2, scope)
    if scope == ORIGINAL_LABEL:
        return (3, scope)
    if scope == OVERALL_LABEL:
        return (4, scope)
    return (5, scope)


def _metric_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    ordered = metrics.copy()
    ordered["_order"] = ordered["scope"].map(_scope_order)
    ordered = ordered.sort_values(["model", "_order", "scope"], kind="stable")
    return ordered.drop(columns="_order")


def _report_markdown(result: StructureTensorResult, config: StructureTensorConfig) -> str:
    """Render the deterministic M6D report."""
    models = [ablation.name for ablation in result.ablations]
    selected = result.selected_name
    diagnostics = result.diagnostics
    lines: list[str] = [
        "# M6D multi-scale structure-tensor feature study",
        "",
        "**Scope:** synthetic image similarity, measured as SSIM against the exact clean "
        "reference. This is not a measure of human readability, CAPTCHA solvability, "
        "security, or real-world quality.",
        "",
        "## Question",
        "",
        "Do compact multi-scale structure-tensor features computed from the corrupted "
        "image alone improve geometric-distortion SSIM prediction beyond the existing "
        "eight global image features, while preserving clean-reference and original-task "
        "performance?",
        "",
        "## Frozen protocol",
        "",
        f"- Benchmark: `{config.run.name}` `{config.run.version}`",
        f"- Config: `{config.config_path}` (sha256 `{config.config_sha256}`)",
        f"- Protocol hash: `{config.config_hash}`",
        f"- Seed: `{config.run.seed}`; grouped bootstrap replicates: `{config.run.n_bootstrap}`",
        f"- Upstream M5C config sha256: `{config.m5c.config_sha256}`",
        f"- Upstream M5C protocol hash: `{config.m5c.protocol_hash}`",
        f"- Frozen champion artifact sha256: `{config.m5c.champion_artifact_sha256}`",
        f"- M6A config sha256: `{config.m6a.config_sha256}`; M6B: `{config.m6b.config_sha256}`; "
        f"M6C: `{config.m6c.config_sha256}`",
        f"- Global schema: `{config.global_feature_set_version}` with "
        f"{len(config.global_feature_names)} features",
        f"- Target: `{config.target}`",
        f"- Primary amplitudes: {list(config.protocol.primary_amplitudes_px)} px; "
        f"diagnostic amplitudes: {list(config.protocol.diagnostic_amplitudes_px)} px",
        f"- Primary severities: {list(config.protocol.primary_severities)}; "
        f"diagnostic severities: {list(config.protocol.diagnostic_severities)}",
        "",
        "## No-reference boundary",
        "",
        "Every primary feature is a pure function of one corrupted 2-D `uint8` grayscale "
        "image. The extractor's declared interface is:",
        "",
        "```python",
        "extract_structure_tensor_features(image: np.ndarray, spec: StructureTensorSpec | None = None)",
        "```",
        "",
        "It has exactly one required input and no `*args` or `**kwargs`. It cannot receive "
        "the clean reference, SSIM or PSNR, a target, a source identifier, a filename, a "
        "corruption family or severity, font metadata, split membership, or any other "
        "sample metadata. The clean reference is used only by the separate evaluation "
        "pipeline that computes the SSIM label. The runtime checks and the independent "
        "structural tests are summarised below.",
        "",
        "| check | passed | detail |",
        "| --- | --- | --- |",
    ]
    for check in result.checks:
        if check.name.startswith(("no_reference", "structure_tensor_feature")):
            lines.append(f"| `{check.name}` | {'yes' if check.passed else 'no'} | {check.detail} |")
    lines.extend(
        [
            "",
            "## Feature definitions",
            "",
            "The feature set is `structure-tensor-v1`. The image is scaled to `[0, 1]`. "
            "Gradients use scikit-image's fixed 3x3 Sobel operator (`[-1, 0, 1]` edge "
            "weights and `[1, 2, 1]` smoothing) with reflected boundaries; no additional "
            "normalisation is applied. Tensor components `Jxx = gx*gx`, `Jyy = gy*gy` and "
            "`Jxy = gx*gy` are smoothed by a Gaussian with reflected boundaries and "
            "truncation at four standard deviations.",
            "",
            "The symmetric 2x2 tensor eigenvalues are",
            "",
            "```text",
            "lambda_1 = (trace + sqrt((Jxx - Jyy)^2 + 4*Jxy^2)) / 2",
            "lambda_2 = (trace - sqrt((Jxx - Jyy)^2 + 4*Jxy^2)) / 2",
            "trace    = Jxx + Jyy",
            "```",
            "",
            "`lambda_2` is clipped to `[0, lambda_1]` for round-off. Coherence is "
            "`(lambda_1 - lambda_2) / (lambda_1 + lambda_2 + epsilon)`, in `[0, 1]`. "
            "Anisotropy is `1 - lambda_2 / (lambda_1 + epsilon)`, in `[0, 1]`. Tensor "
            "energy is `lambda_1 + lambda_2`, non-negative and unbounded above in squared "
            "gradient units. The fixed tolerance is `epsilon = 1e-12`; a zero-energy "
            "tensor is an explicit zero, not a division by zero. Quantiles use NumPy's "
            "linear interpolation.",
            "",
            f"The two predeclared scales are sigma = {config.features.scales[0]:.1f} px "
            f"and {config.features.scales[1]:.1f} px. Four summaries are reported at each "
            "scale: mean coherence, the 0.90 coherence quantile, mean anisotropy and the "
            "0.90 energy quantile. Mean tensor energy is reported once as "
            "`st_energy_mean` at the finest scale, because Gaussian smoothing preserves "
            "the mean and repeating it would duplicate a column. The nine features are:",
            "",
        ]
    )
    for name in config.features.feature_names:
        lines.append(f"- `{name}`: {_feature_definition(name)}")
    lines.extend(
        [
            "",
            "## Candidate representations",
            "",
            f"- `{GLOBAL_BASELINE}`: frozen M5C Random Forest, loaded without refitting.",
            f"- `{STRUCTURE_ONLY}`: the same Random Forest family trained on the nine "
            "structure-tensor features only.",
            f"- `{GLOBAL_PLUS_STRUCTURE}`: the same Random Forest family trained on the "
            "unchanged eight global features concatenated with the nine structure-tensor "
            "features.",
            "",
            "All refitted candidates use the same training rows, model family, "
            "hyperparameters, random seed and objective. There is no sample weighting, "
            "calibration, hyperparameter search or test-driven feature editing.",
            "",
            "## Validation metrics",
            "",
            "The validation partition is the predeclared source-grouped holdout. The "
            "diagnostic 0.5 px rung is excluded from fitting, selection and the primary "
            "score.",
            "",
        ]
    )
    validation = _metric_rows(result.validation_metrics)
    lines.extend(
        _markdown_table(
            validation,
            [
                "model",
                "scope",
                "mae",
                "rmse",
                "r2",
                "r2_defined",
                "mean_residual",
                "sd_residual",
                "n_samples",
                "n_sources",
            ],
        )
    )
    lines.extend(
        [
            "## Selection rule",
            "",
            "The frozen M6A/M6B/M6C retention-aware rule is reused unchanged. The champion "
            "is the reference. A candidate is eligible only if validation clean-reference "
            "RMSE, original-task RMSE and every primary geometric severity RMSE stay within "
            "the predeclared tolerances. Among eligible candidates the score is",
            "",
            "```text",
            "score = w_geometric * (rmse_geo - rmse_geo_ref)",
            "      + w_clean     * max(0, rmse_clean - rmse_clean_ref)",
            "      + w_original  * max(0, rmse_orig - rmse_orig_ref)",
            "      + w_severity  * sum_s max(0, rmse_s - rmse_s_ref)",
            "```",
            "",
            f"Coefficients: geometric {config.selection.w_geometric}, clean "
            f"{config.selection.w_clean}, original {config.selection.w_original}, "
            f"severity {config.selection.w_severity}. Tolerances: clean "
            f"{config.selection.tolerance_clean:.0%}, original "
            f"{config.selection.tolerance_original:.0%}, per severity "
            f"{config.selection.tolerance_severity:.0%}. Ties break on lower aggregate "
            "geometric RMSE and then candidate name. If no candidate qualifies, the "
            "frozen M5C champion is retained.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            result.selection_table,
            [
                "model",
                "feature_set",
                "n_features",
                "eligible",
                "score",
                "delta_geometric_rmse",
                "delta_clean_rmse",
                "delta_original_rmse",
                "max_severity_excess_rmse",
                "selected",
            ],
            float_digits=6,
        )
    )
    lines.extend(
        [
            f"**Selected policy: `{selected}`.**",
            "",
            "## Test metrics",
            "",
            "The test slices are scored only after the validation selection is frozen. "
            "These are descriptive comparisons of all three predeclared candidates; the "
            "policy is the validation-selected candidate above.",
            "",
        ]
    )
    test = _metric_rows(result.test_metrics)
    lines.extend(
        _markdown_table(
            test,
            [
                "model",
                "scope",
                "mae",
                "rmse",
                "r2",
                "r2_defined",
                "mean_residual",
                "sd_residual",
                "n_samples",
                "n_sources",
            ],
        )
    )
    lines.extend(
        [
            "## Paired source-grouped uncertainty",
            "",
            "Intervals resample source groups, not individual image variants. The test set "
            "contains only 12 independent source groups, so these intervals are "
            "preliminary and should not be read as broad generalisation or statistical "
            "significance.",
            "",
        ]
    )
    paired = result.paired_uncertainty.copy()
    if not paired.empty:
        paired = paired[paired["metric"].isin(["rmse", "mae"])]
        paired["_order"] = paired["dataset"].map(_scope_order)
        paired = paired.sort_values(
            ["comparison", "_order", "dataset", "metric"], kind="stable"
        ).drop(columns="_order")
    lines.extend(
        _markdown_table(
            paired,
            [
                "comparison",
                "dataset",
                "metric",
                "estimate",
                "ci_low",
                "ci_high",
                "n_groups",
                "n_samples",
            ],
            float_digits=6,
        )
    )
    lines.extend(
        [
            "## Diagnostics",
            "",
            "Diagnostics use training rows only and are descriptive, not causal. They were "
            "not used to prune, reweight or redesign the feature set in this milestone.",
            "",
            f"- Training rows: {diagnostics['n_train_rows']}",
            f"- Global features: {diagnostics['n_global_features']}",
            f"- Structure-tensor features: {diagnostics['n_structure_features']}",
            f"- Feature-set version: `{diagnostics['structure_tensor_feature_set_version']}`",
            f"- Feature names: {diagnostics['structure_tensor_feature_names']}",
            f"- Constant structure-tensor features: "
            f"{diagnostics['constant_structure_features'] or 'none'}",
            f"- Duplicated structure-tensor features: "
            f"{diagnostics['duplicated_structure_features'] or 'none'}",
            f"- Non-finite structure-tensor features: "
            f"{diagnostics['non_finite_structure_features'] or 'none'}",
            f"- Mean absolute pairwise correlation among structure-tensor features: "
            f"{diagnostics['structure_correlation_mean']:.4f}",
            f"- Maximum absolute pairwise correlation among structure-tensor features: "
            f"{diagnostics['structure_correlation_max']:.4f}",
            "",
            "Strongest absolute pairwise correlations among the new features:",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            pd.DataFrame(diagnostics["structure_correlation_top_pairs"]),
            ["feature_a", "feature_b", "abs_correlation"],
        )
    )
    candidate_rows: list[dict[str, object]] = []
    for name in models:
        entry = diagnostics["candidates"][name]
        candidate_rows.append(
            {
                "model": name,
                "feature_set": entry["feature_set"],
                "n_features": entry["n_features"],
                "structure_vs_global_max_abs_r": entry["structure_vs_global_max_abs_correlation"],
                "train_rmse_geometric": entry["train_rmse_geometric"],
                "validation_rmse_geometric": entry["validation_rmse_geometric"],
                "overfit_gap_geometric_rmse": entry["overfit_gap_geometric_rmse"],
                "importance_structure_share": entry["importance_structure_share"],
            }
        )
    lines.extend(
        _markdown_table(
            pd.DataFrame(candidate_rows),
            [
                "model",
                "feature_set",
                "n_features",
                "structure_vs_global_max_abs_r",
                "train_rmse_geometric",
                "validation_rmse_geometric",
                "overfit_gap_geometric_rmse",
                "importance_structure_share",
            ],
            float_digits=5,
        )
    )
    lines.extend(
        [
            "Random Forest feature importances are descriptive only. They describe how "
            "this fitted forest used its inputs, not a causal effect of any image property.",
            "",
            "## Limitations",
            "",
            "- The target is SSIM against a synthetic clean reference, not human "
            "readability, CAPTCHA solvability, security or production robustness.",
            "- The test partition contains only 12 independent source groups. Paired "
            "intervals are preliminary.",
            "- Structure-tensor features summarise local gradient geometry but not stroke "
            "identity, glyph semantics or human perception.",
            "- The global features remain strong for blur and noise; a compact structural "
            "representation may still be redundant with them even when it is genuinely "
            "local in construction.",
            "- This milestone does not test a CNN, a new corruption family, a dashboard, "
            "deployment or a presentation.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run --extra image --extra ml captcha-structure-tensor-study run \\",
            "    --config configs/structure_tensor_study_v0_1_0.toml",
            "uv run --extra image --extra ml captcha-structure-tensor-study verify \\",
            "    --config configs/structure_tensor_study_v0_1_0.toml",
            "```",
            "",
            "## Research integrity",
            "",
            "All conclusions are limited to the synthetic benchmark, the frozen source-"
            "grouped splits and the SSIM proxy target. No result here establishes human "
            "readability, solve rate, security or real-world CAPTCHA robustness.",
            "",
        ]
    )
    return "\n".join(lines)


def _diagnostic_lines(result: StructureTensorResult, config: StructureTensorConfig) -> list[str]:
    """Render compact training-only diagnostics for the console."""
    diagnostics = result.diagnostics
    lines = [
        f"- Structure-tensor features (`{diagnostics['structure_tensor_feature_set_version']}`): "
        f"{diagnostics['n_structure_features']}",
        f"- Constant structure-tensor features: "
        f"{diagnostics['constant_structure_features'] or 'none'}",
        f"- Duplicated structure-tensor features: "
        f"{diagnostics['duplicated_structure_features'] or 'none'}",
        f"- Non-finite structure-tensor features: "
        f"{diagnostics['non_finite_structure_features'] or 'none'}",
        f"- Mean absolute pairwise correlation among structure-tensor features: "
        f"{diagnostics['structure_correlation_mean']:.4f}",
        f"- Maximum absolute pairwise correlation among structure-tensor features: "
        f"{diagnostics['structure_correlation_max']:.4f}",
        "",
        "| model | structure-vs-global max abs r | train geometric RMSE | "
        "validation geometric RMSE | gap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in [ablation.name for ablation in result.ablations]:
        entry = diagnostics["candidates"][name]
        lines.append(
            f"| {name} | {entry['structure_vs_global_max_abs_correlation']:.4f} | "
            f"{entry['train_rmse_geometric']:.5f} | "
            f"{entry['validation_rmse_geometric']:.5f} | "
            f"{entry['overfit_gap_geometric_rmse']:.5f} |"
        )
    return lines


def _decision_lines(result: StructureTensorResult, config: StructureTensorConfig) -> list[str]:
    """Render the validation-selected policy and the descriptive test comparisons."""
    selected = result.selected_name
    selection = result.selection_table
    selected_row = selection[selection["model"] == selected].iloc[0]
    lines = [
        f"Selected policy: {selected}",
        f"Selection score: {float(selected_row['score']):.6f}",
        f"Validation geometric RMSE: {float(selected_row['rmse_geometric']):.5f}",
    ]
    if selected == GLOBAL_BASELINE:
        lines.append(
            "No structure-tensor candidate qualified under the predeclared retention "
            "rule; the frozen M5C champion is retained."
        )
    else:
        lines.append(
            f"The selected candidate is `{selected}`; it qualified on validation and is "
            "frozen before test scoring."
        )
    test = result.test_metrics
    for scope in (GEO_LABEL, CLEAN_LABEL, ORIGINAL_LABEL):
        rows = test[test["scope"] == scope]
        if rows.empty:
            continue
        champion = rows[rows["model"] == GLOBAL_BASELINE]
        if champion.empty:
            continue
        champion_rmse = float(champion.iloc[0]["rmse"])
        for name in (STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE):
            candidate = rows[rows["model"] == name]
            if candidate.empty:
                continue
            delta = float(candidate.iloc[0]["rmse"]) - champion_rmse
            lines.append(
                f"Test {scope}: {name} RMSE delta versus champion = {delta:+.5f} "
                "(descriptive, not the selection criterion)."
            )
    return lines


def run_structure_tensor_study(
    config: StructureTensorConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    verify_images: bool = True,
) -> tuple[StructureTensorResult, dict[str, Path]]:
    """Load, verify, run and persist the complete M6D study."""
    started = time.perf_counter()
    before = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    frames = load_structure_tensor_frames(config, verify_images=verify_images)
    m5c_config = ra.load_retention_config(config.m5c.config)
    champion = _load_champion(config)

    result = run_structure_tensor_core(frames, config, champion=champion, m5c_config=m5c_config)
    after = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    if not ss._frozen_unchanged(before, after):  # noqa: SLF001
        raise StructureTensorStudyError("a frozen asset changed during the M6D run")
    paths = write_structure_tensor_outputs(
        result,
        config,
        output_reports=output_reports,
        output_models=output_models,
        frozen_before=before,
        frozen_after=after,
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, paths


def verify_structure_tensor_study(
    config: StructureTensorConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
) -> dict[str, Any]:
    """Verify a written M6D run: pins, boundary, selection and prediction replay."""
    reports = Path(output_reports) if output_reports is not None else config.run.output_reports
    models = Path(output_models) if output_models is not None else config.run.output_models
    checks: list[CheckResult] = []

    def record(name: str, function: Any) -> None:
        try:
            detail = function()
        except Exception as exc:  # noqa: BLE001
            checks.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))
        else:
            checks.append(CheckResult(name, True, str(detail)))

    def load_inputs() -> str:
        _verify_pinned_upstream(config)
        _require_file_hash(
            config.m5c.champion_dir / f"{config.m5c.champion_stem}.joblib",
            config.m5c.champion_artifact_sha256,
            "M5C champion artifact",
        )
        frames = load_structure_tensor_frames(config, verify_images=False)
        return (
            f"{len(frames.retention.expanded_train)} training rows, "
            f"{len(frames.retention.expanded_val)} validation rows, "
            f"{len(frames.retention.primary_test)} primary geometric test rows, "
            f"{len(frames.retention.diagnostic_test)} diagnostic test rows and "
            f"{len(frames.structure_features)} extracted structure-tensor feature rows"
        )

    def boundary_checks() -> str:
        signature = _check_no_reference_signature()
        identifiers = _check_no_reference_identifiers()
        if not signature.passed or not identifiers.passed:
            raise StructureTensorStudyError(f"{signature.detail}; {identifiers.detail}")
        return (
            "the extractor accepts one corrupted image and no other data source, and the "
            "feature module's code contains no target or metadata identifier"
        )

    def metadata_checks() -> str:
        metadata_path = reports / "run_metadata.json"
        if not metadata_path.is_file():
            raise StructureTensorStudyError(f"missing run metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("benchmark_config_sha256") != config.config_sha256:
            raise StructureTensorStudyError("run metadata config hash does not match")
        if metadata.get("benchmark_config_hash") != config.config_hash:
            raise StructureTensorStudyError("run metadata protocol hash does not match")
        if not metadata.get("frozen_assets", {}).get("all_unchanged"):
            raise StructureTensorStudyError("run metadata reports a changed frozen asset")
        boundary = metadata.get("no_reference_boundary", {})
        if boundary.get("clean_reference_used_for_features"):
            raise StructureTensorStudyError(
                "run metadata says the clean reference was used for features"
            )
        for name, entry in metadata.get("artifacts", {}).items():
            if not isinstance(entry, dict):
                raise StructureTensorStudyError(f"artifact entry {name!r} is not an object")
            recorded = Path(entry["path"])
            # Prefer the artifact beside the outputs being verified, so a run
            # copied to another directory is checked against its own files
            # rather than against the absolute paths recorded at run time.
            path = reports / recorded.name
            if not path.is_file():
                path = models / recorded.name
            if not path.is_file():
                path = recorded
            if not path.is_file():
                raise StructureTensorStudyError(f"missing recorded artifact: {path}")
            if sha256_file(path) != entry["sha256"]:
                raise StructureTensorStudyError(f"artifact hash mismatch: {name}")
        return f"{len(metadata.get('artifacts', {}))} artifact hashes match"

    def selection_checks() -> str:
        table_path = reports / "selection_table.csv"
        if not table_path.is_file():
            raise StructureTensorStudyError(f"missing selection table: {table_path}")
        table = pd.read_csv(table_path)
        selected = table[table["selected"]]
        if len(selected) != 1:
            raise StructureTensorStudyError("the selection table must mark exactly one winner")
        metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
        recorded = metadata.get("selection", {}).get("selected_model")
        if recorded != str(selected.iloc[0]["model"]):
            raise StructureTensorStudyError(
                f"selection table winner {selected.iloc[0]['model']!r} does not match the "
                f"recorded selection {recorded!r}"
            )
        if recorded not in {GLOBAL_BASELINE, STRUCTURE_ONLY, GLOBAL_PLUS_STRUCTURE}:
            raise StructureTensorStudyError(f"unexpected selected candidate {recorded!r}")
        return f"selected candidate {recorded!r} is consistent"

    def model_checks() -> str:
        for name in config.ablations.names:
            if not config.ablations.refit[name]:
                continue
            card_path = models / f"{name}_card.json"
            artifact = models / f"{name}.joblib"
            if not card_path.is_file() or not artifact.is_file():
                raise StructureTensorStudyError(f"missing model bundle for {name!r}")
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("artifact_sha256") != sha256_file(artifact):
                raise StructureTensorStudyError(f"artifact hash mismatch for {name!r}")
            expected = config.feature_names_for(config.ablations.feature_sets[name])
            if card.get("feature_names") != list(expected):
                raise StructureTensorStudyError(f"feature schema mismatch for {name!r}")
            if card.get("target") != config.target:
                raise StructureTensorStudyError(f"target mismatch for {name!r}")
            if card.get("structure_tensor_feature_set_version") != (
                stf.STRUCTURE_TENSOR_FEATURE_SET_VERSION
            ):
                raise StructureTensorStudyError(
                    f"structure-tensor feature version mismatch for {name!r}"
                )
        return "every refitted bundle matches its card and the declared schema"

    def replay_checks() -> str:
        frames = load_structure_tensor_frames(config, verify_images=False)
        m5c_config = ra.load_retention_config(config.m5c.config)
        champion = _load_champion(config)
        ablations = build_candidates(frames, config, champion=champion, m5c_config=m5c_config)
        slices = build_test_slices(frames, ablations)
        replayed = _test_metrics_table(slices)
        stored = pd.read_csv(reports / "test_slice_metrics.csv")
        merged = replayed.merge(stored, on=["model", "scope"], suffixes=("_replay", "_stored"))
        if len(merged) != len(replayed):
            raise StructureTensorStudyError("replayed metrics do not cover the stored rows")
        for column in ("mae", "rmse"):
            if not np.allclose(
                merged[f"{column}_replay"],
                merged[f"{column}_stored"],
                rtol=0.0,
                atol=1e-9,
            ):
                raise StructureTensorStudyError(f"replayed {column} does not match the stored run")
        return f"{len(replayed)} slice metrics reproduce exactly"

    def report_checks() -> str:
        report_path = reports / "report.md"
        if not report_path.is_file():
            raise StructureTensorStudyError(f"missing report: {report_path}")
        text = report_path.read_text(encoding="utf-8")
        required = (
            "## No-reference boundary",
            "## Feature definitions",
            "## Candidate representations",
            "## Validation metrics",
            "## Selection rule",
            "## Test metrics",
            "## Paired source-grouped uncertainty",
            "## Diagnostics",
            "## Limitations",
            "## Reproduce",
            "## Research integrity",
        )
        missing = [section for section in required if section not in text]
        if missing:
            raise StructureTensorStudyError(f"report is missing sections: {missing}")
        return "the report contains every required section"

    record("inputs_and_pins", load_inputs)
    record("no_reference_boundary", boundary_checks)
    record("run_metadata", metadata_checks)
    record("selection_consistency", selection_checks)
    record("model_bundles", model_checks)
    record("prediction_replay", replay_checks)
    record("report", report_checks)
    return {
        "checks": checks,
        "passed": all(check.passed for check in checks),
        "config": config,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-structure-tensor-study",
        description="Run or verify the M6D multi-scale structure-tensor study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the structure-tensor study")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path, default=None)
    run_parser.add_argument("--output-models", type=Path, default=None)

    verify_parser = subparsers.add_parser("verify", help="verify a written structure-tensor run")
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path, default=None)
    verify_parser.add_argument("--output-models", type=Path, default=None)
    return parser


def _print_report(report: Mapping[str, Any]) -> None:
    for check in report["checks"]:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"structure-tensor study verification: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_structure_tensor_config(args.config)
        if args.command == "run":
            result, paths = run_structure_tensor_study(
                config,
                output_reports=args.output_reports,
                output_models=args.output_models,
            )
            print(f"benchmark: {config.describe()}")
            print(f"protocol hash: {config.config_hash}")
            print(
                "rows: train "
                f"{len(result.frames.retention.expanded_train)}, validation "
                f"{len(result.frames.retention.expanded_val)}, primary test "
                f"{len(result.frames.retention.primary_test)}, diagnostic 0.5 px "
                f"{len(result.frames.retention.diagnostic_test)}"
            )
            print(f"selected candidate: {result.selected_name}")
            for scope in (GEO_LABEL, CLEAN_LABEL, ORIGINAL_LABEL, "m3_test_overall"):
                row = ss._metric_row_for(  # noqa: SLF001
                    result.test_metrics, result.selected_name, scope
                )
                baseline = ss._metric_row_for(  # noqa: SLF001
                    result.test_metrics, GLOBAL_BASELINE, scope
                )
                if row is None or baseline is None:
                    continue
                print(
                    f"  {scope}: selected MAE {row['mae']:.4f} RMSE {row['rmse']:.4f} | "
                    f"baseline MAE {baseline['mae']:.4f} RMSE {baseline['rmse']:.4f}"
                )
            print(f"report: {paths['report']}")
            return 0
        report = verify_structure_tensor_study(
            config,
            output_reports=args.output_reports,
            output_models=args.output_models,
        )
        _print_report(report)
        return 0 if report["passed"] else 1
    except (
        StructureTensorStudyError,
        StructureTensorConfigError,
        stf.StructureTensorFeatureError,
        stf.StructureTensorDependencyError,
        ModelError,
        ra.RetentionSelectionError,
        ra.RetentionSelectionConfigError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
