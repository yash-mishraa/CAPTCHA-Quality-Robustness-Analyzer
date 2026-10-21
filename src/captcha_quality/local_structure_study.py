"""Milestone M6C local structural feature study.

M6A added spatially localized ink-occupancy and edge-density features on a
fixed 4x2 grid and found that they did not improve retention-aware
geometric-distortion prediction. M6B varied only the grid resolution, found
that finer grids resolve an elastic warp better but still produced no
demonstrable predictive gain, and concluded that the blocker is the *relevance*
of regional ink summaries to the other corruption families rather than the
resolution of the partition.

M6C tests a different hypothesis:

    Do local structural features computed from a corrupted image alone improve
    geometric-distortion SSIM prediction beyond the existing global features,
    while preserving clean-reference and original-task performance?

Two interpretable, inexpensive families are predeclared in
``configs/local_structure_study_v0_1_0.toml`` and implemented in
:mod:`captcha_quality.local_structure_features`:

* ``gradient_magnitude`` - Sobel gradient magnitude summarised by four
  statistics (mean, population standard deviation, median and the 0.90
  quantile).
* ``gradient_orientation`` - an eight-bin orientation histogram over
  ``[0, pi)`` plus the circular resultant length and the normalised Shannon
  entropy of that histogram. Pixels below a fixed weak-gradient threshold are
  excluded.

Fourteen local features result. The three predeclared candidates are the frozen
M5C champion (not refitted), a local-only model, and a global-plus-local model.
The model family, hyperparameters, seed, training rows and objective are
identical for the two refitted candidates, so a difference between them is
attributable to the input representation alone. No hyperparameter is tuned, no
post-hoc calibration is applied and no feature is removed in response to a
diagnostic.

**No-reference boundary.** Every local feature is a pure function of one
corrupted grayscale image. The extractor accepts no clean reference, target,
identifier, family, severity, font, split or path value, and the module's code
contains no reference to any of them. The clean reference is used only by the
separate evaluation pipeline that computes the SSIM label. This boundary is
enforced by a runtime check on every run and by structural tests, not merely by
inspecting output column names.

The retention-aware eligibility constraints, selection score, tolerances and
tie-breaking are the M6A/M6B rule unchanged, so all three studies are directly
comparable, and the frozen M5C champion is the fallback. The 0.5 px geometric
rung remains diagnostic only.

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
from captcha_quality import local_structure_features as lsf
from captcha_quality import retention_aware as ra
from captcha_quality import robustness_benchmark as rb
from captcha_quality import spatial_resolution_study as srs
from captcha_quality import spatial_study as ss
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
FEATURE_SET_LOCAL: Final[str] = "local"
FEATURE_SET_COMBINED: Final[str] = "combined"

#: Candidate names.
GLOBAL_BASELINE: Final[str] = "global_baseline"
LOCAL_ONLY: Final[str] = "local_only"
GLOBAL_PLUS_LOCAL: Final[str] = "global_plus_local"

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
        "model",
        "protocol",
        "features",
        "ablations",
        "selection",
    }
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Identifiers that must never appear in the local feature module's code. The
#: clean reference, the targets and every metadata field are outside the
#: feature boundary.
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

#: Frozen inputs audited before and after the milestone. M6C's own outputs are
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
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
    "reports/geometric_distortion_pilot",
    "reports/geometric_robustness_benchmark_v0_1_0",
    "reports/retention_aware_selection_v0_1_0",
    "reports/retention_aware_training_v0_1_0",
    "reports/spatial_feature_study_v0_1_0",
    "reports/spatial_resolution_study_v0_1_0",
)


class LocalStructureStudyError(RuntimeError):
    """Raised when the M6C protocol cannot be honoured."""


class LocalStructureConfigError(ValueError):
    """Raised when the M6C configuration is invalid."""


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
class LocalStructureConfig:
    """The complete M6C protocol."""

    run: rb.RunSettings
    global_feature_set_version: str
    global_feature_names: tuple[str, ...]
    target: str
    m5c: M5CPin
    m6a: M6APin
    m6b: M6BPin
    model: ModelSettings
    protocol: ra.ProtocolSettings
    features: lsf.LocalStructureSpec
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
        if feature_set == FEATURE_SET_LOCAL:
            return tuple(self.features.feature_names)
        if feature_set == FEATURE_SET_COMBINED:
            return tuple(self.global_feature_names) + tuple(self.features.feature_names)
        raise LocalStructureConfigError(f"unknown feature set {feature_set!r}")


# ---------------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------------


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise LocalStructureConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise LocalStructureConfigError(f"missing or invalid [{name}] table")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LocalStructureConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LocalStructureConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LocalStructureConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise LocalStructureConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise LocalStructureConfigError(f"{label} must be a boolean, got {value!r}")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise LocalStructureConfigError(f"{label} must be a TOML array")
    items = tuple(_as_str(item, f"{label}[]") for item in value)
    if not items or len(set(items)) != len(items):
        raise LocalStructureConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise LocalStructureConfigError(f"{label} must be a lowercase SHA-256 hex digest")
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
        raise LocalStructureConfigError(
            f"model.family must be 'random_forest', got {settings.family!r}"
        )
    if settings.n_estimators < 1:
        raise LocalStructureConfigError("model.n_estimators must be at least 1")
    if settings.max_depth < 0:
        raise LocalStructureConfigError("model.max_depth must be zero (unlimited) or positive")
    if settings.min_samples_leaf < 1:
        raise LocalStructureConfigError("model.min_samples_leaf must be at least 1")
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
        raise LocalStructureConfigError(
            "the diagnostic amplitude must not also appear in the primary ladder"
        )
    if set(settings.primary_severities) & set(settings.diagnostic_severities):
        raise LocalStructureConfigError(
            "the diagnostic severity must not also appear in the primary ladder"
        )
    return settings


def _parse_features(raw: Mapping[str, object]) -> lsf.LocalStructureSpec:
    _check_keys(
        raw,
        {"families", "orient_bins", "orient_min_magnitude", "magnitude_quantile"},
        "features",
    )
    return lsf.LocalStructureSpec(
        families=_as_str_tuple(raw["families"], "features.families"),
        orient_bins=_as_int(raw["orient_bins"], "features.orient_bins"),
        orient_min_magnitude=_as_float(
            raw["orient_min_magnitude"], "features.orient_min_magnitude"
        ),
        magnitude_quantile=_as_float(raw["magnitude_quantile"], "features.magnitude_quantile"),
    )


def _parse_ablations(raw: Mapping[str, object]) -> AblationSettings:
    _check_keys(raw, {"names", "feature_sets", "refit"}, "ablations")
    names = _as_str_tuple(raw["names"], "ablations.names")
    if set(names) != {GLOBAL_BASELINE, LOCAL_ONLY, GLOBAL_PLUS_LOCAL}:
        raise LocalStructureConfigError(
            "ablations.names must be exactly "
            f"{[GLOBAL_BASELINE, LOCAL_ONLY, GLOBAL_PLUS_LOCAL]}, got {sorted(names)}"
        )
    raw_feature_sets = raw["feature_sets"]
    raw_refit = raw["refit"]
    if not isinstance(raw_feature_sets, dict) or not isinstance(raw_refit, dict):
        raise LocalStructureConfigError("ablations.feature_sets and ablations.refit must be tables")
    _check_keys(raw_feature_sets, set(names), "ablations.feature_sets")
    _check_keys(raw_refit, set(names), "ablations.refit")
    feature_sets = {
        name: _as_str(raw_feature_sets[name], f"ablations.feature_sets.{name}") for name in names
    }
    refit = {name: _as_bool(raw_refit[name], f"ablations.refit.{name}") for name in names}
    expected = {
        GLOBAL_BASELINE: FEATURE_SET_GLOBAL,
        LOCAL_ONLY: FEATURE_SET_LOCAL,
        GLOBAL_PLUS_LOCAL: FEATURE_SET_COMBINED,
    }
    for name, feature_set in expected.items():
        if feature_sets[name] != feature_set:
            raise LocalStructureConfigError(
                f"ablations.feature_sets.{name} must be {feature_set!r}, got {feature_sets[name]!r}"
            )
    if refit[GLOBAL_BASELINE]:
        raise LocalStructureConfigError(
            "the frozen champion must not be refitted; it is the reference policy"
        )
    if not refit[LOCAL_ONLY] or not refit[GLOBAL_PLUS_LOCAL]:
        raise LocalStructureConfigError(
            "the local-only and global-plus-local ablations must be refitted"
        )
    return AblationSettings(names=names, feature_sets=feature_sets, refit=refit)


def load_local_structure_config(path: Path | str) -> LocalStructureConfig:
    """Load and validate the M6C configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise LocalStructureConfigError(f"missing configuration file: {config_path}")
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    unknown = sorted(set(payload) - _EXPECTED_TOP_KEYS)
    if unknown:
        raise LocalStructureConfigError(
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
        raise LocalStructureConfigError(str(exc)) from exc
    config = LocalStructureConfig(
        run=_parse_run(_section(payload, "run")),
        global_feature_set_version=global_version,
        global_feature_names=global_names,
        target=target,
        m5c=_parse_m5c(_section(payload, "m5c")),
        m6a=_parse_m6a(_section(payload, "m6a")),
        m6b=_parse_m6b(_section(payload, "m6b")),
        model=_parse_model(_section(payload, "model")),
        protocol=_parse_protocol(_section(payload, "protocol")),
        features=_parse_features(_section(payload, "features")),
        ablations=_parse_ablations(_section(payload, "ablations")),
        selection=selection,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )
    if config.global_feature_names != tuple(FEATURE_NAMES):
        raise LocalStructureConfigError(
            "schema.global_feature_names must match the frozen extractor "
            f"{list(FEATURE_NAMES)}; the eight-feature pipeline must not be altered"
        )
    if config.global_feature_set_version != FEATURE_SET_VERSION:
        raise LocalStructureConfigError(
            f"schema.global_feature_set_version must be {FEATURE_SET_VERSION!r}, got "
            f"{config.global_feature_set_version!r}"
        )
    if config.target != TARGET:
        raise LocalStructureConfigError(f"schema.target must be {TARGET!r}")
    if config.m6a.feature_set_version != "spatial-v1":
        raise LocalStructureConfigError(
            "m6a.feature_set_version must be 'spatial-v1'; the frozen spatial "
            "definitions must be reused unchanged"
        )
    if config.protocol.primary_amplitudes_px != gb.PRIMARY_AMPLITUDES_PX:
        raise LocalStructureConfigError(
            "protocol.primary_amplitudes_px must match the M5B primary ladder "
            f"{list(gb.PRIMARY_AMPLITUDES_PX)}"
        )
    if config.protocol.diagnostic_amplitudes_px != gb.DIAGNOSTIC_AMPLITUDES_PX:
        raise LocalStructureConfigError(
            "protocol.diagnostic_amplitudes_px must match the M5B diagnostic ladder "
            f"{list(gb.DIAGNOSTIC_AMPLITUDES_PX)}"
        )
    # The shipped protocol declares one fixed feature family. A different bin
    # count, weak-gradient threshold or quantile would be a different
    # predeclaration, so it is rejected here rather than silently accepted.
    if config.features.families != (lsf.FAMILY_MAGNITUDE, lsf.FAMILY_ORIENTATION):
        raise LocalStructureConfigError(
            f"features.families must be exactly {[lsf.FAMILY_MAGNITUDE, lsf.FAMILY_ORIENTATION]}"
        )
    if config.features.orient_bins != lsf.DEFAULT_ORIENT_BINS:
        raise LocalStructureConfigError(f"features.orient_bins must be {lsf.DEFAULT_ORIENT_BINS}")
    if config.features.orient_min_magnitude != lsf.DEFAULT_ORIENT_MIN_MAGNITUDE:
        raise LocalStructureConfigError(
            f"features.orient_min_magnitude must be {lsf.DEFAULT_ORIENT_MIN_MAGNITUDE}"
        )
    if config.features.magnitude_quantile != lsf.DEFAULT_MAGNITUDE_QUANTILE:
        raise LocalStructureConfigError(
            f"features.magnitude_quantile must be {lsf.DEFAULT_MAGNITUDE_QUANTILE}"
        )
    expected_features = (
        "grad_mag_mean",
        "grad_mag_std",
        "grad_mag_median",
        "grad_mag_p90",
        *(f"orient_hist_{index:02d}" for index in range(config.features.orient_bins)),
        "orient_coherence",
        "orient_entropy",
    )
    if config.features.feature_names != expected_features:
        raise LocalStructureConfigError(
            f"features must declare exactly {list(expected_features)}, got "
            f"{list(config.features.feature_names)}"
        )
    return config


# ---------------------------------------------------------------------------
# Frames, the no-reference boundary and candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalStructureFrames:
    """The frozen frames plus the local structural feature table."""

    retention: ra.RetentionFrames
    local_features: pd.DataFrame
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def with_local(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` joined with the local structural features."""
        if frame["sample_id"].duplicated().any():
            raise LocalStructureStudyError("frame has duplicate sample_id values")
        missing_columns = [
            column for column in ("sample_id", TARGET) if column not in frame.columns
        ]
        if missing_columns:  # pragma: no cover - structural guard
            raise LocalStructureStudyError(f"frame is missing required column(s) {missing_columns}")
        merged = frame.merge(self.local_features, on="sample_id", how="left", validate="one_to_one")
        missing = merged[self.local_features.columns[1:]].isna().any().any()
        if missing:
            raise LocalStructureStudyError("some rows are missing extracted local features")
        return merged


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise LocalStructureStudyError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise LocalStructureStudyError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def _feature_module_source() -> str:
    """Return the source of the local feature module, for the boundary audit."""
    return Path(lsf.__file__).read_text(encoding="utf-8")


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
        signature = inspect.signature(lsf.extract_local_structure_features)
        parameters = list(signature.parameters.values())
        names = tuple(parameter.name for parameter in parameters)
        if names != ("image", "spec"):
            raise LocalStructureStudyError(
                f"the extractor signature must be (image, spec), got {names}"
            )
        for parameter in parameters:
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise LocalStructureStudyError(
                    f"the extractor must not accept *args/**kwargs, found {parameter.name!r}"
                )
        required = [
            parameter
            for parameter in parameters
            if parameter.default is inspect.Parameter.empty
            and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        ]
        if [parameter.name for parameter in required] != ["image"]:
            raise LocalStructureStudyError(
                "the only required extractor input must be the corrupted image array"
            )
        return CheckResult(
            name="no_reference_signature",
            passed=True,
            detail=(
                "extract_local_structure_features(image, spec=None) accepts exactly one "
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
            raise LocalStructureStudyError(
                f"the local feature module's code uses forbidden identifier(s): {forbidden}"
            )
        return CheckResult(
            name="no_reference_identifiers",
            passed=True,
            detail=(
                "the local feature module's code contains no target, reference, source, "
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


def _check_local_feature_frame(features: pd.DataFrame, spec: lsf.LocalStructureSpec) -> CheckResult:
    """Confirm the extracted table has the exact declared schema and is finite."""
    try:
        expected = ("sample_id", *spec.feature_names)
        if tuple(features.columns) != expected:
            raise LocalStructureStudyError(
                f"local feature columns must be exactly {list(expected)}, got "
                f"{list(features.columns)}"
            )
        if features["sample_id"].duplicated().any():
            raise LocalStructureStudyError("local feature table has duplicate sample_id values")
        values = features[list(spec.feature_names)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise LocalStructureStudyError("local features contain non-finite values")
        magnitude = [name for name in spec.feature_names if name.startswith("grad_mag_")]
        if magnitude:
            magnitudes = features[magnitude].to_numpy(dtype=np.float64)
            if (magnitudes < 0.0).any():
                raise LocalStructureStudyError("gradient magnitudes must be non-negative")
        bounded = [name for name in spec.feature_names if name.startswith("orient_")]
        if bounded:
            orientations = features[bounded].to_numpy(dtype=np.float64)
            if (orientations < 0.0).any() or (orientations > 1.0).any():
                raise LocalStructureStudyError("orientation features must lie inside [0, 1]")
        histograms = [name for name in spec.feature_names if name.startswith("orient_hist_")]
        if histograms:
            totals = features[histograms].sum(axis=1).to_numpy(dtype=np.float64)
            # A row with no strong-gradient pixel has an explicitly zero
            # histogram; every other row is normalised to one.
            normalised = np.isclose(totals, 1.0, rtol=0.0, atol=1e-9)
            empty = np.isclose(totals, 0.0, rtol=0.0, atol=1e-12)
            if not np.logical_or(normalised, empty).all():
                raise LocalStructureStudyError(
                    "an orientation histogram must sum to one, or to zero when no "
                    "pixel clears the weak-gradient threshold"
                )
        return CheckResult(
            name="local_feature_schema",
            passed=True,
            detail=(
                f"{len(features)} rows with {len(spec.feature_names)} local features in the "
                "declared order, all finite, magnitudes non-negative, orientation summaries "
                "inside [0, 1] and normalised histograms summing to one"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="local_feature_schema", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _check_no_metadata_in_local_inputs(
    features: pd.DataFrame, spec: lsf.LocalStructureSpec
) -> CheckResult:
    """Confirm the local feature table carries no target or metadata column."""
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
            raise LocalStructureStudyError(
                f"forbidden columns present in the local feature table: {present}"
            )
        if len(spec.feature_names) != len(set(spec.feature_names)):
            raise LocalStructureStudyError("duplicate local feature names")
        return CheckResult(
            name="local_feature_leakage",
            passed=True,
            detail=(
                "the local feature table contains only sample_id and the declared image "
                "features; no target, severity, family, font, source, split or path column "
                "is present"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="local_feature_leakage", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _verify_pinned_upstream(config: LocalStructureConfig) -> ra.RetentionConfig:
    """Verify the M5C, M6A and M6B pins and return the M5C configuration."""
    _require_file_hash(config.m5c.config, config.m5c.config_sha256, "M5C config")
    m5c_config = ra.load_retention_config(config.m5c.config)
    if m5c_config.config_hash != config.m5c.protocol_hash:
        raise LocalStructureStudyError(
            "M5C protocol hash mismatch: expected "
            f"{config.m5c.protocol_hash}, found {m5c_config.config_hash}"
        )
    _require_file_hash(config.m6a.config, config.m6a.config_sha256, "M6A config")
    m6a_config = ss.load_spatial_study_config(config.m6a.config)
    if m6a_config.config_hash != config.m6a.protocol_hash:
        raise LocalStructureStudyError(
            "M6A protocol hash mismatch: expected "
            f"{config.m6a.protocol_hash}, found {m6a_config.config_hash}"
        )
    _require_file_hash(config.m6b.config, config.m6b.config_sha256, "M6B config")
    m6b_config = srs.load_resolution_config(config.m6b.config)
    if m6b_config.config_hash != config.m6b.protocol_hash:
        raise LocalStructureStudyError(
            "M6B protocol hash mismatch: expected "
            f"{config.m6b.protocol_hash}, found {m6b_config.config_hash}"
        )
    return m5c_config


def load_local_structure_frames(
    config: LocalStructureConfig, *, verify_images: bool = True
) -> LocalStructureFrames:
    """Load the frozen frames through the M5C path and extract local features."""
    m5c_config = _verify_pinned_upstream(config)
    retention = ra.load_retention_frames(m5c_config, verify_images=verify_images)
    roots = ss._dataset_roots(m5c_config)  # noqa: SLF001

    # Every row that will be fitted or scored needs its local features. The
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
                raise LocalStructureStudyError(f"no image root configured for dataset {dataset!r}")
            samples.append((sample_id, load_grayscale(root / str(row.image_path))))
    local = lsf.extract_local_structure_feature_table(samples, config.features)

    checks = (
        *retention.checks,
        _check_no_reference_signature(),
        _check_no_reference_identifiers(),
        _check_local_feature_frame(local, config.features),
        _check_no_metadata_in_local_inputs(local, config.features),
    )
    rb._verify_checks(checks, "M6C frame checks")  # noqa: SLF001
    return LocalStructureFrames(retention=retention, local_features=local, checks=checks)


def _champion_spec(config: LocalStructureConfig, m5c_config: ra.RetentionConfig) -> CandidateSpec:
    """Return the champion's grid specification and confirm it matches the config."""
    m5b_config = ra._load_m5b_config(m5c_config)  # noqa: SLF001
    specs = rb.candidate_specs_from_config(m5b_config)  # type: ignore[arg-type]
    spec = next(
        (item for item in specs if item.name == m5c_config.candidates.alternative_spec), None
    )
    if spec is None:
        raise LocalStructureStudyError(
            f"the M5C champion specification {m5c_config.candidates.alternative_spec!r} is "
            "not in the M5B grid"
        )
    if spec.family != config.model.family:
        raise LocalStructureStudyError(
            f"model.family is {config.model.family!r} but the champion was fitted from a "
            f"{spec.family!r} specification"
        )
    for key, value in (
        ("n_estimators", config.model.n_estimators),
        ("max_depth", config.model.max_depth),
        ("min_samples_leaf", config.model.min_samples_leaf),
    ):
        if spec.params.get(key) != value:
            raise LocalStructureStudyError(
                f"model.{key} is {value!r} but the M5B grid specification has "
                f"{spec.params.get(key)!r}"
            )
    seed = derive_noise_seed(m5b_config.run.seed, "model", spec.name) % (2**32)
    if seed != config.model.seed:
        raise LocalStructureStudyError(
            f"model.seed is {config.model.seed} but the M5B grid derives {seed}"
        )
    return spec


def _load_champion(config: LocalStructureConfig) -> Any:
    """Load the frozen M5C champion bundle and confirm it is unchanged."""
    directory = config.m5c.champion_dir
    stem = config.m5c.champion_stem
    artifact = directory / f"{stem}.joblib"
    card_path = directory / f"{stem}_card.json"
    _require_file_hash(artifact, config.m5c.champion_artifact_sha256, "M5C champion artifact")
    if not card_path.is_file():
        raise LocalStructureStudyError(f"missing M5C champion card: {card_path}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("feature_names") != list(config.global_feature_names):
        raise LocalStructureStudyError("the M5C champion uses a different feature schema")
    if card.get("target") != config.target:
        raise LocalStructureStudyError(f"the M5C champion target is not {config.target!r}")
    recorded = str(card.get("selection", {}).get("benchmark_config_hash", ""))
    if recorded != config.m5c.protocol_hash:
        raise LocalStructureStudyError(
            f"the M5C champion was produced under protocol {recorded!r}, not the pinned "
            f"{config.m5c.protocol_hash!r}"
        )
    if card.get("artifact_sha256") != sha256_file(artifact):
        raise LocalStructureStudyError("the M5C champion artifact hash does not match its card")
    if str(card.get("sklearn_version", "")).split(".")[:2] != sklearn_version().split(".")[:2]:
        raise LocalStructureStudyError(
            f"the M5C champion was persisted with scikit-learn {card.get('sklearn_version')}, "
            f"current is {sklearn_version()}"
        )
    import joblib

    estimator = joblib.load(artifact)
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features != len(config.global_feature_names):
        raise LocalStructureStudyError(
            f"the M5C champion expects {n_features} inputs, the frozen schema has "
            f"{len(config.global_feature_names)}"
        )
    return estimator


def build_candidates(
    frames: LocalStructureFrames,
    config: LocalStructureConfig,
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
    train = frames.with_local(frames.retention.expanded_train)
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


def validation_slices(frames: LocalStructureFrames) -> dict[str, pd.DataFrame]:
    """Return the predeclared validation slices, unchanged from M6A/M6B."""
    validation = frames.with_local(frames.retention.expanded_val)
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


def _severity_scopes(config: LocalStructureConfig) -> tuple[str, ...]:
    return tuple(
        f"{GEO_LABEL}_severity_{severity}" for severity in config.protocol.primary_severities
    )


def _slice_metrics_table(
    frames: LocalStructureFrames,
    ablations: Sequence[ss.Ablation],
    predictions: Mapping[str, np.ndarray],
    *,
    split_label: str,
) -> pd.DataFrame:
    """Score every ablation on every validation slice."""
    validation = frames.with_local(frames.retention.expanded_val)
    slices = validation_slices(frames)
    rows: list[dict[str, Any]] = []
    for ablation in ablations:
        predicted = np.asarray(predictions[ablation.name], dtype=np.float64)
        if predicted.size != len(validation):
            raise LocalStructureStudyError(
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
    frames: LocalStructureFrames, ablations: Sequence[ss.Ablation]
) -> list[ss.TestSlice]:
    """Assemble the frozen evaluation slices scored after selection is complete."""
    retention = frames.retention
    geometric = frames.with_local(retention.primary_test)
    m3_test = frames.with_local(retention.m3_test)
    original = m3_test[~m3_test["is_clean"]]
    diagnostic = frames.with_local(retention.diagnostic_test)

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
                seed=derive_noise_seed(seed, "m6c_bootstrap", name, item.scope) % (2**64),
                evaluation="m6c_test",
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
                seed=derive_noise_seed(seed, "m6c_paired", name, item.scope) % (2**64),
                evaluation="m6c_test",
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
    if name.startswith("grad_mag_"):
        return lsf.FAMILY_MAGNITUDE
    if name.startswith("orient_"):
        return lsf.FAMILY_ORIENTATION
    raise LocalStructureStudyError(f"unrecognised feature name {name!r}")


def training_diagnostics(
    frames: LocalStructureFrames,
    ablations: Sequence[ss.Ablation],
    validation_metrics: pd.DataFrame,
    config: LocalStructureConfig,
) -> dict[str, Any]:
    """Collect descriptive diagnostics from the training rows only.

    Nothing here is used to redesign the feature set within this milestone: the
    diagnostics are reported, not acted upon. In particular the feature set is
    not pruned, reweighted or re-binned in response to any value below.
    """
    train = frames.with_local(frames.retention.expanded_train)
    local_names = list(config.features.feature_names)
    global_names = list(config.global_feature_names)
    local_values = train[local_names]

    constant_local = [name for name in local_names if float(local_values[name].std(ddof=0)) == 0.0]
    duplicated_local = _duplicate_columns(local_values)
    non_finite_local = [
        name
        for name in local_names
        if not np.isfinite(local_values[name].to_numpy(dtype=np.float64)).all()
    ]
    local_correlation_mean, local_correlation_max = _correlation_summary(local_values)

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

        local_in_model = [name for name in ablation.feature_names if name in set(local_names)]
        global_in_model = [name for name in ablation.feature_names if name in set(global_names)]
        cross_block = (
            _cross_block_correlation(values, local_in_model, global_in_model)
            if local_in_model and global_in_model
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
            raise LocalStructureStudyError(
                f"ablation {ablation.name!r} has no validation geometric metric"
            )
        validation_rmse = float(validation_row.iloc[0]["rmse"])

        importances: list[dict[str, Any]] = []
        importance_global = 0.0
        importance_local = 0.0
        if ablation.train_importances is not None:
            order = np.argsort(-ablation.train_importances, kind="stable")
            for rank, index in enumerate(order, start=1):
                name = ablation.feature_names[int(index)]
                value = float(ablation.train_importances[int(index)])
                family = _feature_family(name, global_names)
                if family == "global":
                    importance_global += value
                else:
                    importance_local += value
                importances.append(
                    {
                        "rank": rank,
                        "feature_name": name,
                        "family": family,
                        "importance": value,
                    }
                )
        total_importance = importance_global + importance_local

        candidates[ablation.name] = {
            "feature_set": ablation.feature_set,
            "n_features": len(ablation.feature_names),
            "n_global_features": len(global_in_model),
            "n_local_features": len(local_in_model),
            "constant_features": constant,
            "duplicated_features": duplicated,
            "non_finite_features": non_finite,
            "correlation_mean": correlation_mean,
            "correlation_max": correlation_max,
            "local_vs_global_max_abs_correlation": cross_block,
            "train_rmse_overall": float(train_scores["rmse"]),
            "train_rmse_geometric": float(geometric_train_scores["rmse"]),
            "validation_rmse_geometric": validation_rmse,
            "overfit_gap_geometric_rmse": float(validation_rmse - geometric_train_scores["rmse"]),
            "importance_global_share": (
                float(importance_global / total_importance) if total_importance > 0.0 else None
            ),
            "importance_local_share": (
                float(importance_local / total_importance) if total_importance > 0.0 else None
            ),
            "feature_importances": importances,
        }

    return {
        "n_train_rows": int(len(train)),
        "n_global_features": len(global_names),
        "n_local_features": len(local_names),
        "local_feature_set_version": lsf.LOCAL_STRUCTURE_FEATURE_SET_VERSION,
        "local_feature_names": local_names,
        "constant_local_features": constant_local,
        "duplicated_local_features": duplicated_local,
        "non_finite_local_features": non_finite_local,
        "local_correlation_mean": local_correlation_mean,
        "local_correlation_max": local_correlation_max,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalStructureResult:
    """The complete M6C outcome."""

    frames: LocalStructureFrames
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
        raise LocalStructureStudyError(f"selected ablation {self.selected_name!r} is missing")


def run_local_structure_core(
    frames: LocalStructureFrames,
    config: LocalStructureConfig,
    *,
    champion: Any,
    m5c_config: ra.RetentionConfig,
) -> LocalStructureResult:
    """Run the M6C study in the predeclared order.

    The order is fixed: fit on training data only, score validation, select,
    freeze, and only then score the untouched test slices.
    """
    rb._verify_checks(frames.checks, "M6C frame checks")  # noqa: SLF001

    ablations = build_candidates(frames, config, champion=champion, m5c_config=m5c_config)
    validation = frames.with_local(frames.retention.expanded_val)
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

    return LocalStructureResult(
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
    result: LocalStructureResult,
    config: LocalStructureConfig,
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
        "local_feature_set_version": lsf.LOCAL_STRUCTURE_FEATURE_SET_VERSION,
        "local_features": config.features.identity_payload(),
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


def write_local_structure_outputs(
    result: LocalStructureResult,
    config: LocalStructureConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    frozen_before: Mapping[str, Mapping[str, object]] | None = None,
    frozen_after: Mapping[str, Mapping[str, object]] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Path]:
    """Persist the M6C tables, model bundles, metadata and report."""
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
    lsf.local_structure_manifest(config.features).to_csv(
        reports / "local_feature_manifest.csv", index=False
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
            "feature_manifest": reports / "local_feature_manifest.csv",
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
            "local_feature_set_version": lsf.LOCAL_STRUCTURE_FEATURE_SET_VERSION,
            "local_feature_names": list(config.features.feature_names),
            "target": config.target,
            "split_names": ["train", "val", "test"],
        },
        "no_reference_boundary": {
            "extractor": "extract_local_structure_features(image, spec=None)",
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
    "grad_mag_mean": "mean Sobel gradient magnitude on the [0, 1] scale",
    "grad_mag_std": "population standard deviation of the Sobel gradient magnitude",
    "grad_mag_median": "median Sobel gradient magnitude",
    "grad_mag_p90": "0.90 quantile of the Sobel gradient magnitude",
    "orient_coherence": (
        "circular resultant length of the doubled strong-gradient orientations, in [0, 1]"
    ),
    "orient_entropy": (
        "normalised Shannon entropy of the strong-gradient orientation histogram, in [0, 1]"
    ),
}


def _feature_definition(name: str, bins: int) -> str:
    if name in _FEATURE_DEFINITIONS:
        return _FEATURE_DEFINITIONS[name]
    if name.startswith("orient_hist_"):
        index = int(name.rsplit("_", 1)[1])
        width = math.pi / bins
        return (
            f"share of strong-gradient pixels with orientation in "
            f"[{index * width:.4f}, {(index + 1) * width:.4f}) rad"
        )
    raise LocalStructureStudyError(f"no report definition for feature {name!r}")


def _manifest_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    return str(value)


def _report_markdown(result: LocalStructureResult, config: LocalStructureConfig) -> str:
    """Render the deterministic M6C report."""
    models = [ablation.name for ablation in result.ablations]
    selected = result.selected_name
    lines: list[str] = [
        "# M6C local structural feature study",
        "",
        "**Scope:** synthetic image similarity, measured as SSIM against the exact clean "
        "reference. This is not a measure of human readability, CAPTCHA solvability, "
        "security, or real-world quality.",
        "",
        "## Question",
        "",
        "Do local structural features computed from a corrupted image alone improve "
        "geometric-distortion SSIM prediction beyond the existing eight global image "
        "features, while preserving clean-reference and original-task performance?",
        "",
        "## Frozen protocol",
        "",
        f"- Benchmark: `{config.run.name}` `{config.run.version}`",
        f"- Config: `{config.config_path}` (sha256 `{config.config_sha256}`)",
        f"- Protocol hash: `{config.config_hash}`",
        f"- Seed: `{config.run.seed}`; grouped bootstrap replicates: `{config.run.n_bootstrap}`",
        f"- Upstream M5C config sha256: `{config.m5c.config_sha256}`; M5C protocol hash: "
        f"`{config.m5c.protocol_hash}`",
        f"- M5C champion artifact sha256: `{config.m5c.champion_artifact_sha256}`",
        f"- Pinned M6A config sha256: `{config.m6a.config_sha256}`; protocol hash: "
        f"`{config.m6a.protocol_hash}`",
        f"- Pinned M6B config sha256: `{config.m6b.config_sha256}`; protocol hash: "
        f"`{config.m6b.protocol_hash}`",
        f"- Model family: `{config.model.family}` "
        f"(n_estimators={config.model.n_estimators}, max_depth={config.model.max_depth}, "
        f"min_samples_leaf={config.model.min_samples_leaf}, seed={config.model.seed})",
        f"- Frozen global feature set: `{config.global_feature_set_version}`, "
        f"{len(config.global_feature_names)} features, unchanged",
        f"- Local feature set: `{lsf.LOCAL_STRUCTURE_FEATURE_SET_VERSION}`, "
        f"{len(config.features.feature_names)} features",
        "",
        "The M3 dataset, split, frozen M3 model, M4A pilot, M4B benchmark, M5A pilot, "
        "M5B benchmark, M5C selection study, M5D training study, M6A spatial-aware study "
        "and M6B resolution study are read-only and hash-verified before and after the "
        "run. The M6A `spatial-v1` definitions and the M6B configuration are pinned and "
        "verified unchanged. The M6B regional grids are not used by any candidate here.",
        "",
        "### Severity protocol",
        "",
        f"- Primary geometric score: amplitudes "
        f"`{list(config.protocol.primary_amplitudes_px)}` px (severities "
        f"`{list(config.protocol.primary_severities)}`).",
        f"- Diagnostic only: amplitude `{list(config.protocol.diagnostic_amplitudes_px)}` px "
        f"(severity `{list(config.protocol.diagnostic_severities)}`).",
        "- The 0.5 px rung never enters fitting, validation selection or the primary score.",
        "",
        "## No-reference feature boundary",
        "",
        "Every local feature is a pure function of one corrupted 2-D `uint8` grayscale "
        "image. The extractor is",
        "",
        "```text",
        "extract_local_structure_features(image: np.ndarray, spec: LocalStructureSpec | None = None)",
        "```",
        "",
        "The only required input is the image array. There is no parameter for, and no "
        "code path to, the clean reference, the SSIM/PSNR targets, the source identifier, "
        "the corruption family or severity, the font, the split label, the file path or "
        "any metadata. The clean reference is used only by the separate evaluation "
        "pipeline that computes the SSIM label.",
        "",
        "This is enforced structurally, not by inspecting output column names:",
        "",
        "1. A signature check confirms that the extractor accepts exactly `image` and an "
        "optional `spec`, with no `*args`, no `**kwargs` and no other required input.",
        "2. An AST scan of the feature module's code confirms that no target, reference, "
        "source, severity, family, font, split or path identifier appears as a variable, "
        "argument, attribute or import. Prose that explains the boundary is ignored, "
        "because prose is not a data path.",
        "3. A schema check confirms the extracted table contains only `sample_id` and the "
        "declared image features, and a leakage check confirms no target or metadata "
        "column is present.",
        "",
        "| boundary check | result | detail |",
        "| --- | --- | --- |",
    ]
    for check in result.checks:
        if not check.name.startswith(("no_reference", "local_feature")):
            continue
        lines.append(
            f"| `{check.name}` | {'pass' if check.passed else 'fail'} | "
            f"{check.detail.replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Local feature definitions",
            "",
            "Two predeclared, interpretable families are computed on the corrupted image "
            "alone. The image is scaled to `[0, 1]` and Sobel gradients are computed with "
            "scikit-image's normalised 3x3 operator and reflected boundaries.",
            "",
            f"- **`{lsf.FAMILY_MAGNITUDE}`** (4 features): whole-image summaries of the "
            "local gradient magnitude `hypot(gx, gy)`. They describe how strong local "
            "structure is, rather than only how much ink is present.",
            f"- **`{lsf.FAMILY_ORIENTATION}`** "
            f"({config.features.orient_bins + 2} features): the local gradient "
            "orientation `arctan2(gy, gx)` folded into `[0, pi)`, because an edge "
            "orientation is undirected. Pixels whose gradient magnitude is below "
            f"`{config.features.orient_min_magnitude}` carry no reliable orientation and "
            f"are excluded. The survivors are binned into {config.features.orient_bins} "
            "equal-width bins over `[0, pi)` and the counts are normalised to sum to one. "
            "Two scalar summaries accompany the histogram: the circular resultant length "
            "of the doubled angles (coherence in `[0, 1]`) and the normalised Shannon "
            "entropy of the histogram (0 to 1).",
            "",
            "The two families were declared before any test evaluation. They were chosen "
            "for interpretability and cost, not by searching a feature space: gradient "
            "magnitude is the canonical local-structure strength measure, and gradient "
            "orientation is the canonical description of stroke direction and anisotropy. "
            "Regional intensity statistics, edge-continuity graphs and learned filters "
            "were not used.",
            "",
            "Degenerate inputs are explicit. An exactly uniform image has zero gradient "
            "everywhere, so all magnitude statistics are zero and, because no pixel clears "
            "the weak-gradient threshold, the orientation histogram, coherence and entropy "
            "are all zero. An empty strong-gradient set is a defined zero, not a division "
            "by zero. Every output is finite and deterministic. A non-empty orientation "
            "histogram sums to one; an empty one sums to zero by construction.",
            "",
            "### Feature manifest",
            "",
            "| feature | family | definition |",
            "| --- | --- | --- |",
        ]
    )
    manifest = lsf.local_structure_manifest(config.features)
    for row in manifest.itertuples(index=False):
        lines.append(
            f"| `{row.feature_name}` | {row.family} | "
            f"{_feature_definition(str(row.feature_name), config.features.orient_bins)} |"
        )
    lines.extend(
        [
            "",
            "| feature | family | statistic | bin index | bin lower (rad) | bin upper (rad) |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in manifest.itertuples(index=False):
        lower = _manifest_cell(row.bin_lower_rad)
        upper = _manifest_cell(row.bin_upper_rad)
        if lower != "n/a":
            lower = f"{float(lower):.6f}"
            upper = f"{float(upper):.6f}"
        lines.append(
            f"| `{row.feature_name}` | {row.family} | {row.statistic} | "
            f"{_manifest_cell(row.bin_index)} | {lower} | {upper} |"
        )

    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "The model family, the hyperparameters, the seed, the training rows and the "
            "objective are held fixed across all three candidates, so the only thing that "
            "varies is the input representation. No hyperparameter is tuned, no post-hoc "
            "calibration is applied and no feature is removed in response to a diagnostic.",
            "",
            "| candidate | feature set | features | fitted |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for ablation in result.ablations:
        lines.append(
            f"| `{ablation.name}` | {ablation.feature_set} | "
            f"{len(ablation.feature_names)} | "
            f"{'refitted' if ablation.refitted else 'frozen champion, not refitted'} |"
        )

    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            "The rule was declared before any test evaluation and is identical to the "
            "M6A/M6B rule, so the studies are directly comparable. The baseline is the "
            "reference. A candidate qualifies only if it regresses neither the clean "
            "slice, nor the original-task slice, nor any single primary geometric severity "
            "beyond the declared tolerances:",
            "",
            "```text",
            "eligible(c) <=> clean RMSE    <= baseline clean RMSE    * (1 + tolerance_clean)",
            "               original RMSE <= baseline original RMSE * (1 + tolerance_original)",
            "               for each primary severity s:",
            "                   RMSE_s     <= baseline RMSE_s       * (1 + tolerance_severity)",
            "```",
            "",
            "Among the qualifying candidates the score is",
            "",
            "```text",
            "score = w_geometric * (rmse_geo - rmse_geo_ref)",
            "      + w_clean     * max(0, rmse_clean - rmse_clean_ref)",
            "      + w_original  * max(0, rmse_orig  - rmse_orig_ref)",
            "      + w_severity   * sum_s max(0, rmse_s - rmse_s_ref)",
            "```",
            "",
            f"With `w_geometric={config.selection.w_geometric}`, "
            f"`w_clean={config.selection.w_clean}`, "
            f"`w_original={config.selection.w_original}`, "
            f"`w_severity={config.selection.w_severity}`, "
            f"`tolerance_clean={config.selection.tolerance_clean}`, "
            f"`tolerance_original={config.selection.tolerance_original}` and "
            f"`tolerance_severity={config.selection.tolerance_severity}`. Lower is better "
            "and the baseline scores exactly 0. Ties are broken by lower aggregate "
            "geometric RMSE and then by candidate name. If no candidate qualifies, the "
            "frozen M5C champion is kept. No coefficient or tolerance was chosen by "
            "inspecting a test result.",
            "",
            "## Validation results by slice",
            "",
            "Validation only. This is the evidence the selection rule consumed.",
            "",
        ]
    )
    for scope, title in (
        (GEO_LABEL, "Aggregate geometric validation slice (1.0-3.0 px)"),
        (CLEAN_LABEL, "Clean-reference validation slice"),
        (ORIGINAL_LABEL, "Original blur/noise validation slice"),
        (OVERALL_LABEL, "Overall validation population (secondary)"),
    ):
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(ss._slice_table(result.validation_metrics, scope, models))  # noqa: SLF001
        lines.append("")

    lines.append("### Geometric validation slice by severity")
    lines.append("")
    for severity in config.protocol.primary_severities:
        scope = f"{GEO_LABEL}_severity_{severity}"
        lines.append(f"**Severity {severity}**")
        lines.append("")
        lines.extend(ss._slice_table(result.validation_metrics, scope, models))  # noqa: SLF001
        lines.append("")

    lines.extend(
        [
            "### Eligibility and selection scores",
            "",
            "| candidate | eligible | score | geometric RMSE | clean RMSE | original RMSE | "
            "selected |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result.selection_table.itertuples(index=False):
        lines.append(
            f"| `{row.model}` | {'yes' if row.eligible else 'no'} | {row.score:.5f} | "
            f"{row.rmse_geometric:.5f} | {row.rmse_clean:.5f} | "
            f"{row.rmse_original:.5f} | {'yes' if row.selected else 'no'} |"
        )
    ineligible = result.selection_table[~result.selection_table["eligible"]]
    if not ineligible.empty:
        lines.append("")
        lines.append("Ineligible candidates:")
        lines.append("")
        for row in ineligible.itertuples(index=False):
            lines.append(f"- `{row.model}`: {row.ineligible_reasons}")
    lines.extend(
        [
            "",
            f"Selected candidate: **`{selected}`**.",
            "",
            "## Training-only diagnostics",
            "",
            "These diagnostics are computed on the training rows only. They are reported "
            "to assess whether the local family carries information and whether the "
            "models overfit. They are not used to redesign the feature set within this "
            "milestone, and feature importance is descriptive, not causal.",
            "",
        ]
    )
    lines.extend(_diagnostic_lines(result, config))
    lines.extend(
        [
            "",
            "## Final test results by slice",
            "",
            "The selected candidate and its complete configuration were frozen before "
            "these rows were scored. The M3 test split and the M5B/M5C geometric test "
            "split are untouched frozen evaluations. All three candidates are reported, "
            "as planned, so the descriptive ablations are visible alongside the selected "
            "candidate.",
            "",
        ]
    )
    for scope, title in (
        (GEO_LABEL, "Primary geometric test slice (1.0-3.0 px)"),
        (CLEAN_LABEL, "Clean-reference test slice"),
        (ORIGINAL_LABEL, "Original blur/noise test slice"),
        ("m3_test_overall", "Original M3 test slice (overall)"),
        ("diagnostic_0_5", "Diagnostic 0.5 px test slice (not in the primary score)"),
    ):
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(ss._slice_table(result.test_metrics, scope, models))  # noqa: SLF001
        lines.append("")

    lines.append("### Geometric test slice by severity")
    lines.append("")
    for severity in config.protocol.primary_severities:
        scope = f"{GEO_LABEL}_severity_{severity}"
        lines.append(f"**Severity {severity}**")
        lines.append("")
        lines.extend(ss._slice_table(result.test_metrics, scope, models))  # noqa: SLF001
        lines.append("")

    lines.extend(
        [
            "## Paired source-grouped differences against the frozen baseline",
            "",
            "Whole source groups are resampled, preserving all rows within each selected "
            "group. Differences are candidate minus the frozen baseline on identical "
            "rows. There are 12 held-out test source groups, so these intervals are "
            "preliminary and descriptive. Interval overlap or non-overlap is not a "
            "significance test, and no significance claim is made.",
            "",
        ]
    )
    paired_scopes = [
        (GEO_LABEL, "Primary geometric test slice"),
        (CLEAN_LABEL, "Clean-reference test slice"),
        (ORIGINAL_LABEL, "Original blur/noise test slice"),
        ("m3_test_overall", "Original M3 test slice (overall)"),
    ]
    paired_scopes.extend(
        (f"{GEO_LABEL}_severity_{severity}", f"Geometric test severity {severity}")
        for severity in config.protocol.primary_severities
    )
    for scope, title in paired_scopes:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| candidate | metric | estimate | 95% low | 95% high |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for model in models:
            if model == GLOBAL_BASELINE:
                continue
            for metric in ("mae", "rmse"):
                found = ss._paired_interval_for(  # noqa: SLF001
                    result.paired_uncertainty, model, scope, metric
                )
                if found is None:
                    continue
                estimate, low, high = found
                lines.append(
                    f"| `{model}` | {metric} | {estimate:+.4f} | {low:+.4f} | {high:+.4f} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Consistency across source groups",
            "",
            "A paired change driven by one or two sources is weaker evidence than a "
            "change that holds across the group. This counts the held-out sources on "
            "which a candidate has a lower absolute error than the frozen baseline.",
            "",
            "| scope | candidate | sources improved | worst source change | best source "
            "change | largest single-source share of the total gain |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope in (GEO_LABEL, ORIGINAL_LABEL, "m3_test_overall"):
        consistency = ss._source_consistency(result, scope)  # noqa: SLF001
        for candidate, entry in consistency.get("candidates", {}).items():
            share = entry["largest_single_source_share"]
            share_text = f"{share * 100:.1f}%" if math.isfinite(share) else "n/a"
            lines.append(
                f"| {scope} | `{candidate}` | "
                f"{entry['n_sources_better']}/{entry['n_sources']} | "
                f"{entry['worst_delta']:+.4f} | {entry['best_delta']:+.4f} | "
                f"{share_text} |"
            )
    lines.extend(
        [
            "",
            "`n/a` marks a slice where the per-source gains and losses nearly cancel, so "
            "no single-source share is meaningful.",
            "",
            "## Interpretation",
            "",
        ]
    )
    lines.extend(_decision_lines(result, config))
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- One synthetic renderer, one 192x64 canvas, three fonts and 60 sources.",
            "- Only 12 held-out test source groups, so all grouped intervals are "
            "preliminary and descriptive.",
            "- The 0.5 px rung is close to the interpolation floor and is diagnostic only.",
            "- The local features are whole-image summaries of a local operator. They can "
            "describe the strength and anisotropy of structure, but they cannot localise "
            "a warp to a glyph or compare a region against its own reference.",
            "- The global features are strong for blur and noise. If the local family "
            "mainly re-expresses those effects, it will add little on the geometric slice "
            "even if it is genuinely local in construction.",
            "- Feature importance is descriptive, not causal, and is reported only to "
            "assess whether the local family carries information.",
            "- SSIM is a full-reference structural proxy, not human readability or CAPTCHA "
            "solvability. This is not a security evaluation.",
            "- No CNN, dashboard, deployment, external CAPTCHA dataset or additional "
            "corruption family is part of this milestone.",
            "",
            "## Reproducibility",
            "",
            "Run and verify with:",
            "",
            "```text",
            "uv run --extra image --extra ml captcha-local-structure-study run \\",
            "    --config configs/local_structure_study_v0_1_0.toml",
            "uv run --extra image --extra ml captcha-local-structure-study verify \\",
            "    --config configs/local_structure_study_v0_1_0.toml",
            "```",
            "",
            "Generated tables, the model bundles and this report live in Git-ignored paths.",
        ]
    )
    return "\n".join(lines) + "\n"


def _diagnostic_lines(result: LocalStructureResult, config: LocalStructureConfig) -> list[str]:
    """Render the descriptive training-only diagnostics."""
    diagnostics = result.diagnostics
    lines = [
        f"- Training rows: {diagnostics['n_train_rows']}",
        f"- Global features: {diagnostics['n_global_features']}",
        f"- Local features (`{diagnostics['local_feature_set_version']}`): "
        f"{diagnostics['n_local_features']}",
        f"- Constant local features: {diagnostics['constant_local_features'] or 'none'}",
        f"- Duplicated local features: {diagnostics['duplicated_local_features'] or 'none'}",
        f"- Non-finite local features: {diagnostics['non_finite_local_features'] or 'none'}",
        f"- Mean absolute pairwise correlation among the local features: "
        f"{diagnostics['local_correlation_mean']:.4f}",
        f"- Maximum absolute pairwise correlation among the local features: "
        f"{diagnostics['local_correlation_max']:.4f}",
        "",
        "| candidate | features | constant | duplicated | non-finite | max abs r | "
        "local-vs-global max abs r | train geometric RMSE | validation geometric RMSE | "
        "gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in [ablation.name for ablation in result.ablations]:
        entry = diagnostics["candidates"][name]
        lines.append(
            f"| `{name}` | {entry['n_features']} | "
            f"{len(entry['constant_features'])} | "
            f"{len(entry['duplicated_features'])} | "
            f"{len(entry['non_finite_features'])} | "
            f"{entry['correlation_max']:.4f} | "
            f"{entry['local_vs_global_max_abs_correlation']:.4f} | "
            f"{entry['train_rmse_geometric']:.5f} | "
            f"{entry['validation_rmse_geometric']:.5f} | "
            f"{entry['overfit_gap_geometric_rmse']:+.5f} |"
        )
    lines.append("")
    for name in [ablation.name for ablation in result.ablations]:
        entry = diagnostics["candidates"][name]
        importances = entry.get("feature_importances") or []
        if not importances:
            lines.append(f"### `{name}` feature importance (frozen champion, not refitted)")
            lines.append("")
            continue
        lines.append(f"### `{name}` feature importance (training rows, descriptive)")
        lines.append("")
        lines.append(
            f"Global share of total importance: {entry['importance_global_share']:.4f}; "
            f"local share: {entry['importance_local_share']:.4f}."
        )
        lines.append("")
        lines.append("| rank | feature | family | importance |")
        lines.append("| ---: | --- | --- | ---: |")
        for item in importances[:15]:
            lines.append(
                f"| {item['rank']} | `{item['feature_name']}` | {item['family']} | "
                f"{item['importance']:.4f} |"
            )
        lines.append("")
    return lines


def _decision_lines(result: LocalStructureResult, config: LocalStructureConfig) -> list[str]:
    """Derive the descriptive interpretation from the observed results."""
    selected = result.selected_name
    lines: list[str] = []

    def metric(model: str, scope: str) -> dict[str, Any] | None:
        return ss._metric_row_for(result.test_metrics, model, scope)  # noqa: SLF001

    baseline_geo = metric(GLOBAL_BASELINE, GEO_LABEL)
    selected_geo = metric(selected, GEO_LABEL)
    baseline_clean = metric(GLOBAL_BASELINE, CLEAN_LABEL)
    selected_clean = metric(selected, CLEAN_LABEL)
    baseline_original = metric(GLOBAL_BASELINE, ORIGINAL_LABEL)
    selected_original = metric(selected, ORIGINAL_LABEL)
    baseline_m3 = metric(GLOBAL_BASELINE, "m3_test_overall")
    selected_m3 = metric(selected, "m3_test_overall")

    if baseline_geo and selected_geo:
        change = (
            (selected_geo["rmse"] - baseline_geo["rmse"]) / baseline_geo["rmse"] * 100
            if baseline_geo["rmse"] > 0.0
            else math.nan
        )
        lines.append(
            f"- **Primary geometric slice.** The frozen baseline reaches MAE "
            f"{baseline_geo['mae']:.4f}, RMSE {baseline_geo['rmse']:.4f} and R2 "
            f"{baseline_geo['r2']:.4f} on {baseline_geo['n_samples']} rows from "
            f"{baseline_geo['n_sources']} sources. The selected candidate `{selected}` "
            f"reaches MAE {selected_geo['mae']:.4f}, RMSE {selected_geo['rmse']:.4f} and "
            f"R2 {selected_geo['r2']:.4f}, a {change:+.1f}% change in RMSE against the "
            "baseline."
        )
    if baseline_clean and selected_clean:
        lines.append(
            f"- **Clean-reference slice.** Baseline MAE {baseline_clean['mae']:.4f} / "
            f"RMSE {baseline_clean['rmse']:.4f}; selected MAE "
            f"{selected_clean['mae']:.4f} / RMSE {selected_clean['rmse']:.4f}."
        )
    if baseline_original and selected_original:
        lines.append(
            f"- **Original blur/noise slice.** Baseline MAE "
            f"{baseline_original['mae']:.4f} / RMSE {baseline_original['rmse']:.4f}; "
            f"selected MAE {selected_original['mae']:.4f} / RMSE "
            f"{selected_original['rmse']:.4f}."
        )
    if baseline_m3 and selected_m3:
        lines.append(
            f"- **Original-task retention.** On the same untouched M3 test rows the "
            f"baseline reaches MAE {baseline_m3['mae']:.4f} / RMSE "
            f"{baseline_m3['rmse']:.4f} / R2 {baseline_m3['r2']:.4f} and the selected "
            f"candidate reaches MAE {selected_m3['mae']:.4f} / RMSE "
            f"{selected_m3['rmse']:.4f} / R2 {selected_m3['r2']:.4f}."
        )

    for row in result.selection_table.itertuples(index=False):
        if row.model == GLOBAL_BASELINE or bool(row.eligible):
            continue
        lines.append(
            f"- **`{row.model}` was ineligible.** {row.ineligible_reasons}. The "
            "per-severity and retention constraints are what reject it; the aggregate "
            "geometric score alone would not have revealed the regression."
        )

    baseline_row = result.selection_table[result.selection_table["model"] == GLOBAL_BASELINE]
    baseline_geo_rmse = (
        float(baseline_row.iloc[0]["rmse_geometric"]) if not baseline_row.empty else math.nan
    )
    for row in result.selection_table.itertuples(index=False):
        if row.model == GLOBAL_BASELINE:
            continue
        lines.append(
            f"- **`{row.model}` descriptive ablation.** Validation geometric RMSE "
            f"{row.rmse_geometric:.5f} against the baseline's {baseline_geo_rmse:.5f} "
            f"(delta {row.delta_geometric_rmse:+.5f}); clean delta "
            f"{row.delta_clean_rmse:+.5f}; original-task delta "
            f"{row.delta_original_rmse:+.5f}; score {row.score:+.5f}; eligible: "
            f"{'yes' if row.eligible else 'no'}."
        )

    for name in (LOCAL_ONLY, GLOBAL_PLUS_LOCAL):
        parts: list[str] = []
        improved: list[int] = []
        worsened: list[int] = []
        for severity in config.protocol.primary_severities:
            scope = f"{GEO_LABEL}_severity_{severity}"
            baseline = metric(GLOBAL_BASELINE, scope)
            candidate = metric(name, scope)
            if baseline is None or candidate is None:
                continue
            delta = candidate["rmse"] - baseline["rmse"]
            parts.append(
                f"severity {severity}: {candidate['rmse']:.5f} vs "
                f"{baseline['rmse']:.5f} ({delta:+.5f})"
            )
            if delta < 0.0:
                improved.append(severity)
            elif delta > 0.0:
                worsened.append(severity)
        if parts:
            lines.append(f"- **`{name}` by primary severity (test).** " + "; ".join(parts) + ".")
        if improved and worsened:
            lines.append(
                f"- **`{name}` is mixed across the primary ladder.** It improves "
                f"severity {improved} but regresses severity {worsened}, so its aggregate "
                "change is not a uniform improvement."
            )
        elif worsened and not improved:
            lines.append(
                f"- **`{name}` regresses every primary severity on test.** The aggregate "
                "loss is broad-based rather than a single-rung artefact."
            )

    diagnostics = result.diagnostics
    combined_diagnostics = diagnostics["candidates"].get(GLOBAL_PLUS_LOCAL, {})
    lines.append(
        f"- **Redundancy.** The maximum absolute pairwise correlation among the local "
        f"features is {diagnostics['local_correlation_max']:.4f}, and the maximum "
        f"absolute correlation between a local and a global feature in the combined "
        f"candidate is "
        f"{combined_diagnostics.get('local_vs_global_max_abs_correlation', 0.0):.4f}. "
        "High values mean the local family partly re-expresses information the global "
        "features already carry, which limits what it can add."
    )

    if selected == GLOBAL_BASELINE:
        lines.append(
            "- **Outcome.** No local-structure candidate was selected: the frozen M5C "
            "champion is retained. This is reported as a negative result rather than a "
            "manufactured winner. The retention constraints were not weakened to force a "
            "different answer."
        )
        lines.append(
            "- **What the local family did descriptively.** The local-only and "
            "global-plus-local candidates are reported in full above. Their validation "
            "and test deltas show whether the family carried information beyond the "
            "global features; the paired source-grouped intervals are preliminary and "
            "descriptive because there are only 12 held-out source groups."
        )
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark "
            "champion, and the M5C champion remains the best geometric-robustness policy "
            "found so far. Adding these local structural features did not improve the "
            "held-out geometric slice under the predeclared retention-aware rule. The "
            "evidence does not justify a CNN, and a new corruption family does not by "
            "itself justify one."
        )
    else:
        lines.append(
            f"- **Outcome.** A local-structure candidate was selected: `{selected}`. It "
            "met every predeclared retention constraint on validation and improved the "
            "aggregate geometric validation score."
        )
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark "
            f"champion, now with the local-structure policy `{selected}` rather than the "
            "frozen baseline alone. The evidence does not justify a CNN, and a new "
            "corruption family does not by itself justify one."
        )
    lines.append(
        "- **Most defensible next step.** The result is specific to these two declared "
        "families and this synthetic dataset. If the local family does not help, the "
        "next defensible step is to test a qualitatively different representation "
        "before considering a learned model; if it does help, the next step is to "
        "replicate the gain on an independent renderer or dataset. Neither step is part "
        "of this milestone."
    )
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_local_structure_study(
    config: LocalStructureConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    verify_images: bool = True,
) -> tuple[LocalStructureResult, dict[str, Path]]:
    """Load, verify, run and persist the complete M6C study."""
    started = time.perf_counter()
    before = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    frames = load_local_structure_frames(config, verify_images=verify_images)
    m5c_config = ra.load_retention_config(config.m5c.config)
    champion = _load_champion(config)

    result = run_local_structure_core(frames, config, champion=champion, m5c_config=m5c_config)
    after = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    if not ss._frozen_unchanged(before, after):  # noqa: SLF001
        raise LocalStructureStudyError("a frozen asset changed during the M6C run")
    paths = write_local_structure_outputs(
        result,
        config,
        output_reports=output_reports,
        output_models=output_models,
        frozen_before=before,
        frozen_after=after,
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, paths


def verify_local_structure_study(
    config: LocalStructureConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
) -> dict[str, Any]:
    """Verify a written M6C run: pins, boundary, selection and prediction replay."""
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
        frames = load_local_structure_frames(config, verify_images=False)
        return (
            f"{len(frames.retention.expanded_train)} training rows, "
            f"{len(frames.retention.expanded_val)} validation rows, "
            f"{len(frames.retention.primary_test)} primary geometric test rows, "
            f"{len(frames.retention.diagnostic_test)} diagnostic test rows and "
            f"{len(frames.local_features)} extracted local feature rows"
        )

    def boundary_checks() -> str:
        signature = _check_no_reference_signature()
        identifiers = _check_no_reference_identifiers()
        if not signature.passed or not identifiers.passed:
            raise LocalStructureStudyError(f"{signature.detail}; {identifiers.detail}")
        return (
            "the extractor accepts one corrupted image and no other data source, and the "
            "feature module's code contains no target or metadata identifier"
        )

    def metadata_checks() -> str:
        metadata_path = reports / "run_metadata.json"
        if not metadata_path.is_file():
            raise LocalStructureStudyError(f"missing run metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("benchmark_config_sha256") != config.config_sha256:
            raise LocalStructureStudyError("run metadata config hash does not match")
        if metadata.get("benchmark_config_hash") != config.config_hash:
            raise LocalStructureStudyError("run metadata protocol hash does not match")
        if not metadata.get("frozen_assets", {}).get("all_unchanged"):
            raise LocalStructureStudyError("run metadata reports a changed frozen asset")
        boundary = metadata.get("no_reference_boundary", {})
        if boundary.get("clean_reference_used_for_features"):
            raise LocalStructureStudyError(
                "run metadata says the clean reference was used for features"
            )
        for name, entry in metadata.get("artifacts", {}).items():
            if not isinstance(entry, dict):
                raise LocalStructureStudyError(f"artifact entry {name!r} is not an object")
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
                raise LocalStructureStudyError(f"missing recorded artifact: {path}")
            if sha256_file(path) != entry["sha256"]:
                raise LocalStructureStudyError(f"artifact hash mismatch: {name}")
        return f"{len(metadata.get('artifacts', {}))} artifact hashes match"

    def selection_checks() -> str:
        table_path = reports / "selection_table.csv"
        if not table_path.is_file():
            raise LocalStructureStudyError(f"missing selection table: {table_path}")
        table = pd.read_csv(table_path)
        selected = table[table["selected"]]
        if len(selected) != 1:
            raise LocalStructureStudyError("the selection table must mark exactly one winner")
        metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
        recorded = metadata.get("selection", {}).get("selected_model")
        if recorded != str(selected.iloc[0]["model"]):
            raise LocalStructureStudyError(
                f"selection table winner {selected.iloc[0]['model']!r} does not match the "
                f"recorded selection {recorded!r}"
            )
        if recorded not in {GLOBAL_BASELINE, LOCAL_ONLY, GLOBAL_PLUS_LOCAL}:
            raise LocalStructureStudyError(f"unexpected selected candidate {recorded!r}")
        return f"selected candidate {recorded!r} is consistent"

    def model_checks() -> str:
        for name in config.ablations.names:
            if not config.ablations.refit[name]:
                continue
            card_path = models / f"{name}_card.json"
            artifact = models / f"{name}.joblib"
            if not card_path.is_file() or not artifact.is_file():
                raise LocalStructureStudyError(f"missing model bundle for {name!r}")
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("artifact_sha256") != sha256_file(artifact):
                raise LocalStructureStudyError(f"artifact hash mismatch for {name!r}")
            expected = config.feature_names_for(config.ablations.feature_sets[name])
            if card.get("feature_names") != list(expected):
                raise LocalStructureStudyError(f"feature schema mismatch for {name!r}")
            if card.get("target") != config.target:
                raise LocalStructureStudyError(f"target mismatch for {name!r}")
            if card.get("local_feature_set_version") != (lsf.LOCAL_STRUCTURE_FEATURE_SET_VERSION):
                raise LocalStructureStudyError(f"local feature version mismatch for {name!r}")
        return "every refitted bundle matches its card and the declared schema"

    def replay_checks() -> str:
        frames = load_local_structure_frames(config, verify_images=False)
        m5c_config = ra.load_retention_config(config.m5c.config)
        champion = _load_champion(config)
        ablations = build_candidates(frames, config, champion=champion, m5c_config=m5c_config)
        slices = build_test_slices(frames, ablations)
        replayed = _test_metrics_table(slices)
        stored = pd.read_csv(reports / "test_slice_metrics.csv")
        merged = replayed.merge(stored, on=["model", "scope"], suffixes=("_replay", "_stored"))
        if len(merged) != len(replayed):
            raise LocalStructureStudyError("replayed metrics do not cover the stored rows")
        for column in ("mae", "rmse"):
            if not np.allclose(
                merged[f"{column}_replay"],
                merged[f"{column}_stored"],
                rtol=0.0,
                atol=1e-9,
            ):
                raise LocalStructureStudyError(f"replayed {column} does not match the stored run")
        return f"{len(replayed)} slice metrics reproduce exactly"

    def report_checks() -> str:
        report_path = reports / "report.md"
        if not report_path.is_file():
            raise LocalStructureStudyError(f"missing report: {report_path}")
        text = report_path.read_text(encoding="utf-8")
        required = (
            "No-reference feature boundary",
            "Local feature definitions",
            "Feature manifest",
            "Candidates",
            "Selection rule",
            "Validation results by slice",
            "Training-only diagnostics",
            "Final test results by slice",
            "Paired source-grouped differences against the frozen baseline",
            "Reproducibility",
        )
        missing = [section for section in required if section not in text]
        if missing:
            raise LocalStructureStudyError(f"report is missing sections: {missing}")
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
        prog="captcha-local-structure-study",
        description="Run or verify the M6C local structural feature study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the local structural study")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path, default=None)
    run_parser.add_argument("--output-models", type=Path, default=None)

    verify_parser = subparsers.add_parser("verify", help="verify a written local structural run")
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path, default=None)
    verify_parser.add_argument("--output-models", type=Path, default=None)
    return parser


def _print_report(report: Mapping[str, Any]) -> None:
    for check in report["checks"]:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"local structural study verification: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_local_structure_config(args.config)
        if args.command == "run":
            result, paths = run_local_structure_study(
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
        report = verify_local_structure_study(
            config,
            output_reports=args.output_reports,
            output_models=args.output_models,
        )
        _print_report(report)
        return 0 if report["passed"] else 1
    except (
        LocalStructureStudyError,
        LocalStructureConfigError,
        lsf.LocalStructureFeatureError,
        lsf.LocalStructureDependencyError,
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
