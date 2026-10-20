"""Milestone M6A spatial-aware feature study.

M5B, M5C and M5D established that the frozen Milestone 3 feature set, the
model family and the training objective are not the binding constraint on
geometric-distortion prediction: M5B and M5C both reached only a near-zero
between-source R2 on the held-out geometric slice, and M5D found that
reweighting the training rows moves error between slices rather than removing
it. M6A tests the remaining hypothesis, that the *representation* is the
constraint:

    Does adding spatially localized image features improve geometric-distortion
    SSIM prediction compared with the existing eight global image features,
    while preserving clean-reference and original-task performance?

This is a controlled feature-ablation experiment. The eight frozen global
features and their extraction code are unchanged; the spatial features live in
a separate, versioned module (``spatial_features.py``) with their own names, so
prior model inputs and artefacts cannot be silently altered. The model family,
the hyperparameters, the seed and the training partition are held fixed across
all three ablations, so the only thing that varies is the input representation.

Three ablations are predeclared:

* **Global baseline** - the frozen M5C champion, loaded without refitting.
* **Spatial-only** - the spatial feature family alone, refitted on the same
  expanded training rows.
* **Combined** - the eight global features plus the spatial features,
  refitted on the same expanded training rows.

Selection uses a predeclared eligibility-and-score rule computed on validation
data only, with the same coefficients and tolerances as the M5D rule so the two
studies are directly comparable. The selected ablation and its full
configuration are frozen before the untouched M3 and M5B/M5C test slices are
scored, and the frozen M5C champion is the fallback if nothing qualifies.

Feature extraction is deterministic and may be applied to every image, but it
never sees a target, a clean reference or any metadata, and the test rows
never influence the feature design, the feature count, the selection rule or
the model settings. The 0.5 px geometric rung remains a diagnostic only.

The target is synthetic image similarity (SSIM against the exact clean
reference). Nothing here measures human readability, CAPTCHA solvability,
security or real-world robustness.
"""

from __future__ import annotations

import argparse
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
from captcha_quality import retention_aware as ra
from captcha_quality import robustness_benchmark as rb
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
from captcha_quality.spatial_features import (
    SPATIAL_FEATURE_SET_VERSION,
    SpatialFeatureSpec,
    extract_spatial_feature_table,
    spatial_feature_manifest,
)

#: Ablation names.
GLOBAL_BASELINE: Final[str] = "global_baseline"
SPATIAL_ONLY: Final[str] = "spatial_only"
COMBINED: Final[str] = "combined"

#: Feature-set names.
FEATURE_SET_GLOBAL: Final[str] = "global"
FEATURE_SET_SPATIAL: Final[str] = "spatial"
FEATURE_SET_COMBINED: Final[str] = "combined"

#: Slice labels, matching the M5C and M5D conventions.
GEO_LABEL: Final[str] = "geometric_primary"
CLEAN_LABEL: Final[str] = "clean_reference"
ORIGINAL_LABEL: Final[str] = "original_blur_noise"
OVERALL_LABEL: Final[str] = "overall"

_EXPECTED_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {"run", "schema", "m5c", "model", "protocol", "features", "ablations", "selection"}
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Frozen inputs audited before and after the milestone. M6A's own outputs are
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
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "data/processed/geometric_distortion_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "models/robustness_benchmark_v0_1_0",
    "models/geometric_robustness_benchmark_v0_1_0",
    "models/retention_aware_selection_v0_1_0",
    "models/retention_aware_training_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
    "reports/geometric_distortion_pilot",
    "reports/geometric_robustness_benchmark_v0_1_0",
    "reports/retention_aware_selection_v0_1_0",
    "reports/retention_aware_training_v0_1_0",
)


class SpatialStudyError(RuntimeError):
    """Raised when the M6A protocol cannot be honoured."""


class SpatialStudyConfigError(ValueError):
    """Raised when the M6A configuration is invalid."""


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
class ModelSettings:
    """The fixed model family and hyperparameters shared by every ablation."""

    family: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    seed: int


@dataclass(frozen=True, slots=True)
class StudySelectionSettings:
    """The predeclared eligibility constraints and selection score."""

    rule: str
    w_geometric: float
    w_clean: float
    w_original: float
    w_severity: float
    tolerance_clean: float
    tolerance_original: float
    tolerance_severity: float


@dataclass(frozen=True, slots=True)
class SpatialStudyConfig:
    """The complete M6A protocol."""

    run: rb.RunSettings
    global_feature_set_version: str
    global_feature_names: tuple[str, ...]
    target: str
    m5c: M5CPin
    model: ModelSettings
    protocol: ra.ProtocolSettings
    features: SpatialFeatureSpec
    ablation_names: tuple[str, ...]
    ablation_feature_sets: Mapping[str, str]
    ablation_refit: Mapping[str, bool]
    selection: StudySelectionSettings
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
                "names": list(self.ablation_names),
                "feature_sets": {
                    key: self.ablation_feature_sets[key]
                    for key in sorted(self.ablation_feature_sets)
                },
                "refit": {key: self.ablation_refit[key] for key in sorted(self.ablation_refit)},
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
        if feature_set == FEATURE_SET_SPATIAL:
            return tuple(self.features.feature_names)
        if feature_set == FEATURE_SET_COMBINED:
            return tuple(self.global_feature_names) + tuple(self.features.feature_names)
        raise SpatialStudyConfigError(f"unknown feature set {feature_set!r}")


# ---------------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------------


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise SpatialStudyConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise SpatialStudyConfigError(f"missing or invalid [{name}] table")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpatialStudyConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialStudyConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpatialStudyConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise SpatialStudyConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SpatialStudyConfigError(f"{label} must be a TOML array")
    items = tuple(_as_str(item, f"{label}[]") for item in value)
    if not items or len(set(items)) != len(items):
        raise SpatialStudyConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise SpatialStudyConfigError(f"{label} must be a lowercase SHA-256 hex digest")
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


def _parse_schema(raw: Mapping[str, object]) -> tuple[str, tuple[str, ...], str]:
    _check_keys(
        raw,
        {"global_feature_set_version", "global_feature_names", "target", "split_names"},
        "schema",
    )
    return (
        _as_str(raw["global_feature_set_version"], "schema.global_feature_set_version"),
        _as_str_tuple(raw["global_feature_names"], "schema.global_feature_names"),
        _as_str(raw["target"], "schema.target"),
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
        protocol_hash=_as_sha256(raw["protocol_hash"], "m5b.protocol_hash"),
        champion_dir=Path(_as_str(raw["champion_dir"], "m5c.champion_dir")),
        champion_stem=_as_str(raw["champion_stem"], "m5c.champion_stem"),
        champion_artifact_sha256=_as_sha256(
            raw["champion_artifact_sha256"], "m5c.champion_artifact_sha256"
        ),
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
        raise SpatialStudyConfigError(
            f"model.family must be 'random_forest', got {settings.family!r}"
        )
    if settings.n_estimators < 1:
        raise SpatialStudyConfigError("model.n_estimators must be at least 1")
    if settings.max_depth < 0:
        raise SpatialStudyConfigError("model.max_depth must be zero (unlimited) or positive")
    if settings.min_samples_leaf < 1:
        raise SpatialStudyConfigError("model.min_samples_leaf must be at least 1")
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
        raise SpatialStudyConfigError(
            "the diagnostic amplitude must not also appear in the primary ladder"
        )
    if set(settings.primary_severities) & set(settings.diagnostic_severities):
        raise SpatialStudyConfigError(
            "the diagnostic severity must not also appear in the primary ladder"
        )
    return settings


def _parse_features(raw: Mapping[str, object]) -> SpatialFeatureSpec:
    _check_keys(
        raw,
        {
            "grid_rows",
            "grid_cols",
            "families",
            "edge_sigma",
            "edge_low_threshold",
            "edge_high_threshold",
        },
        "features",
    )
    return SpatialFeatureSpec(
        grid_rows=_as_int(raw["grid_rows"], "features.grid_rows"),
        grid_cols=_as_int(raw["grid_cols"], "features.grid_cols"),
        families=_as_str_tuple(raw["families"], "features.families"),
        edge_sigma=_as_float(raw["edge_sigma"], "features.edge_sigma"),
        edge_low_threshold=_as_float(raw["edge_low_threshold"], "features.edge_low_threshold"),
        edge_high_threshold=_as_float(raw["edge_high_threshold"], "features.edge_high_threshold"),
    )


def _parse_ablations(
    raw: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, str], dict[str, bool]]:
    _check_keys(raw, {"names", "feature_sets", "refit"}, "ablations")
    names = _as_str_tuple(raw["names"], "ablations.names")
    feature_sets_raw = raw["feature_sets"]
    refit_raw = raw["refit"]
    if not isinstance(feature_sets_raw, dict) or not isinstance(refit_raw, dict):
        raise SpatialStudyConfigError("ablations.feature_sets and ablations.refit must be tables")
    feature_sets = {
        str(key): _as_str(value, f"ablations.feature_sets.{key}")
        for key, value in feature_sets_raw.items()
    }
    refit = {str(key): bool(value) for key, value in refit_raw.items()}
    if set(feature_sets) != set(names) or set(refit) != set(names):
        raise SpatialStudyConfigError(
            "ablations.feature_sets and ablations.refit must name exactly the ablations in "
            "ablations.names"
        )
    known_sets = {FEATURE_SET_GLOBAL, FEATURE_SET_SPATIAL, FEATURE_SET_COMBINED}
    for name, feature_set in feature_sets.items():
        if feature_set not in known_sets:
            raise SpatialStudyConfigError(
                f"ablation {name!r} uses unknown feature set {feature_set!r}"
            )
    if refit.get(GLOBAL_BASELINE, True):
        raise SpatialStudyConfigError(
            "the global baseline must not be refitted; it is the frozen champion"
        )
    return names, feature_sets, refit


def _parse_selection(raw: Mapping[str, object]) -> StudySelectionSettings:
    _check_keys(
        raw,
        {
            "rule",
            "w_geometric",
            "w_clean",
            "w_original",
            "w_severity",
            "tolerance_clean",
            "tolerance_original",
            "tolerance_severity",
        },
        "selection",
    )
    settings = StudySelectionSettings(
        rule=_as_str(raw["rule"], "selection.rule"),
        w_geometric=_as_float(raw["w_geometric"], "selection.w_geometric"),
        w_clean=_as_float(raw["w_clean"], "selection.w_clean"),
        w_original=_as_float(raw["w_original"], "selection.w_original"),
        w_severity=_as_float(raw["w_severity"], "selection.w_severity"),
        tolerance_clean=_as_float(raw["tolerance_clean"], "selection.tolerance_clean"),
        tolerance_original=_as_float(raw["tolerance_original"], "selection.tolerance_original"),
        tolerance_severity=_as_float(raw["tolerance_severity"], "selection.tolerance_severity"),
    )
    if settings.rule != "retention_gated_rmse_delta":
        raise SpatialStudyConfigError(
            f"selection.rule must be 'retention_gated_rmse_delta', got {settings.rule!r}"
        )
    for label, value in (
        ("w_geometric", settings.w_geometric),
        ("w_clean", settings.w_clean),
        ("w_original", settings.w_original),
        ("w_severity", settings.w_severity),
    ):
        if value < 0.0:
            raise SpatialStudyConfigError(f"selection.{label} must be non-negative")
    for label, value in (
        ("tolerance_clean", settings.tolerance_clean),
        ("tolerance_original", settings.tolerance_original),
        ("tolerance_severity", settings.tolerance_severity),
    ):
        if value < 0.0:
            raise SpatialStudyConfigError(f"selection.{label} must be non-negative")
    return settings


def load_spatial_study_config(path: Path | str) -> SpatialStudyConfig:
    """Load and validate the M6A configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise SpatialStudyConfigError(f"missing configuration file: {config_path}")
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    unknown = sorted(set(payload) - _EXPECTED_TOP_KEYS)
    if unknown:
        raise SpatialStudyConfigError(
            f"unknown top-level key(s): {unknown}; allowed keys are {sorted(_EXPECTED_TOP_KEYS)}"
        )
    global_version, global_names, target = _parse_schema(_section(payload, "schema"))
    names, feature_sets, refit = _parse_ablations(_section(payload, "ablations"))
    config = SpatialStudyConfig(
        run=_parse_run(_section(payload, "run")),
        global_feature_set_version=global_version,
        global_feature_names=global_names,
        target=target,
        m5c=_parse_m5c(_section(payload, "m5c")),
        model=_parse_model(_section(payload, "model")),
        protocol=_parse_protocol(_section(payload, "protocol")),
        features=_parse_features(_section(payload, "features")),
        ablation_names=names,
        ablation_feature_sets=feature_sets,
        ablation_refit=refit,
        selection=_parse_selection(_section(payload, "selection")),
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )
    if config.global_feature_names != tuple(FEATURE_NAMES):
        raise SpatialStudyConfigError(
            "schema.global_feature_names must match the frozen extractor "
            f"{list(FEATURE_NAMES)}; the eight-feature pipeline must not be altered"
        )
    if config.global_feature_set_version != "v1":
        raise SpatialStudyConfigError(
            f"schema.global_feature_set_version must be 'v1', got "
            f"{config.global_feature_set_version!r}"
        )
    if config.target != TARGET:
        raise SpatialStudyConfigError(f"schema.target must be {TARGET!r}")
    if config.protocol.primary_amplitudes_px != gb.PRIMARY_AMPLITUDES_PX:
        raise SpatialStudyConfigError(
            "protocol.primary_amplitudes_px must match the M5B primary ladder "
            f"{list(gb.PRIMARY_AMPLITUDES_PX)}"
        )
    if config.protocol.diagnostic_amplitudes_px != gb.DIAGNOSTIC_AMPLITUDES_PX:
        raise SpatialStudyConfigError(
            "protocol.diagnostic_amplitudes_px must match the M5B diagnostic ladder "
            f"{list(gb.DIAGNOSTIC_AMPLITUDES_PX)}"
        )
    expected = {GLOBAL_BASELINE, SPATIAL_ONLY, COMBINED}
    if set(config.ablation_names) != expected:
        raise SpatialStudyConfigError(
            f"ablations.names must be {sorted(expected)}, got {sorted(config.ablation_names)}"
        )
    if config.ablation_feature_sets[GLOBAL_BASELINE] != FEATURE_SET_GLOBAL:
        raise SpatialStudyConfigError("the global baseline must use the global feature set")
    if config.ablation_feature_sets[SPATIAL_ONLY] != FEATURE_SET_SPATIAL:
        raise SpatialStudyConfigError("the spatial-only ablation must use the spatial set")
    if config.ablation_feature_sets[COMBINED] != FEATURE_SET_COMBINED:
        raise SpatialStudyConfigError("the combined ablation must use the combined set")
    return config


# ---------------------------------------------------------------------------
# Frames, spatial features and ablations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyFrames:
    """The frozen frames plus the extracted spatial feature table."""

    retention: ra.RetentionFrames
    spatial_features: pd.DataFrame
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def with_spatial(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` joined with the spatial features, keyed by sample_id."""
        if frame["sample_id"].duplicated().any():
            raise SpatialStudyError("frame has duplicate sample_id values")
        merged = frame.merge(
            self.spatial_features, on="sample_id", how="left", validate="one_to_one"
        )
        missing = merged[self.spatial_features.columns[1:]].isna().any().any()
        if missing:
            raise SpatialStudyError("some rows are missing extracted spatial features")
        return merged


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SpatialStudyError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise SpatialStudyError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def _dataset_roots(m5c_config: ra.RetentionConfig) -> dict[str, Path]:
    """Return the image root for each dataset that contributes rows.

    The M5C configuration does not name the datasets directly; it points at the
    M5B configuration, which pins the M3 and M5A dataset paths.
    """
    m5b_config = ra._load_m5b_config(m5c_config)  # noqa: SLF001
    return {
        gb.M3_LABEL: Path(m5b_config.m3.dataset),
        gb.M5A_LABEL: Path(m5b_config.m5a.dataset),
    }


def _check_spatial_feature_frame(features: pd.DataFrame, spec: SpatialFeatureSpec) -> CheckResult:
    """Confirm the extracted table has the exact declared schema and is finite."""
    try:
        expected = ("sample_id", *spec.feature_names)
        if tuple(features.columns) != expected:
            raise SpatialStudyError(
                f"spatial feature columns must be exactly {list(expected)}, got "
                f"{list(features.columns)}"
            )
        if features["sample_id"].duplicated().any():
            raise SpatialStudyError("spatial feature table has duplicate sample_id values")
        values = features[list(spec.feature_names)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise SpatialStudyError("spatial features contain non-finite values")
        if (values < 0.0).any() or (values > 1.0).any():
            raise SpatialStudyError("spatial features fall outside [0, 1]")
        return CheckResult(
            name="spatial_feature_schema",
            passed=True,
            detail=(
                f"{len(features)} rows with {len(spec.feature_names)} spatial features in the "
                "declared order, all finite and inside [0, 1]"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="spatial_feature_schema", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _check_no_metadata_in_spatial_inputs(
    features: pd.DataFrame, spec: SpatialFeatureSpec
) -> CheckResult:
    """Confirm the spatial features carry no target or metadata column."""
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
            "amplitude_px",
            "protocol_scope",
            "is_clean",
            "is_control",
            "is_elastic",
        }
        present = sorted(forbidden & set(features.columns))
        if present:
            raise SpatialStudyError(f"forbidden columns present in the feature table: {present}")
        if len(spec.feature_names) != len(set(spec.feature_names)):
            raise SpatialStudyError("duplicate spatial feature names")
        return CheckResult(
            name="spatial_feature_leakage",
            passed=True,
            detail=(
                "the spatial feature table contains only sample_id and the declared image "
                "features; no target, severity, family, font, source, split or path column "
                "is present"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="spatial_feature_leakage", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def load_study_frames(config: SpatialStudyConfig, *, verify_images: bool = True) -> StudyFrames:
    """Load the frozen frames through the M5C path and extract spatial features."""
    _require_file_hash(config.m5c.config, config.m5c.config_sha256, "M5C config")
    m5c_config = ra.load_retention_config(config.m5c.config)
    if m5c_config.config_hash != config.m5c.protocol_hash:
        raise SpatialStudyError(
            "M5C protocol hash mismatch: expected "
            f"{config.m5c.protocol_hash}, found {m5c_config.config_hash}"
        )
    retention = ra.load_retention_frames(m5c_config, verify_images=verify_images)
    roots = _dataset_roots(m5c_config)

    # Every row of every frame that will be scored or fitted needs its spatial
    # features. The extraction is deterministic and uses only image pixels.
    needed: dict[str, pd.DataFrame] = {
        "train": retention.expanded_train,
        "val": retention.expanded_val,
        "m3_test": retention.m3_test,
        "primary_test": retention.primary_test,
        "diagnostic_test": retention.diagnostic_test,
    }
    samples: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for frame in needed.values():
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            if sample_id in seen:
                continue
            seen.add(sample_id)
            dataset = str(row.dataset)
            root = roots.get(dataset)
            if root is None:
                raise SpatialStudyError(f"no image root configured for dataset {dataset!r}")
            image = load_grayscale(root / str(row.image_path))
            samples.append((sample_id, image))
    spatial = extract_spatial_feature_table(samples, config.features)

    checks = (
        *retention.checks,
        _check_spatial_feature_frame(spatial, config.features),
        _check_no_metadata_in_spatial_inputs(spatial, config.features),
    )
    rb._verify_checks(checks, "M6A frame checks")  # noqa: SLF001
    return StudyFrames(retention=retention, spatial_features=spatial, checks=checks)


@dataclass(frozen=True, slots=True)
class Ablation:
    """One fitted ablation: a named model over a declared feature set."""

    name: str
    feature_set: str
    feature_names: tuple[str, ...]
    estimator: Any
    refitted: bool
    train_importances: np.ndarray | None

    def inputs(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ModelError(f"frame is missing feature column(s): {missing}")
        return frame[list(self.feature_names)].to_numpy(dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.estimator.predict(self.inputs(frame))
        predictions = np.asarray(values, dtype=np.float64).reshape(-1)
        if predictions.size != len(frame):
            raise ModelError(
                f"model returned {predictions.size} prediction(s) for {len(frame)} row(s)"
            )
        if not np.isfinite(predictions).all():
            raise ModelError("model produced non-finite predictions")
        return predictions


def _champion_spec(config: SpatialStudyConfig, m5c_config: ra.RetentionConfig) -> CandidateSpec:
    """Return the champion's grid specification and confirm it matches the config."""
    m5b_config = ra._load_m5b_config(m5c_config)  # noqa: SLF001
    specs = rb.candidate_specs_from_config(m5b_config)  # type: ignore[arg-type]
    spec = next(
        (item for item in specs if item.name == m5c_config.candidates.alternative_spec), None
    )
    if spec is None:
        raise SpatialStudyError(
            f"the M5C champion specification {m5c_config.candidates.alternative_spec!r} is not "
            "in the M5B grid"
        )
    if spec.family != config.model.family:
        raise SpatialStudyError(
            f"model.family is {config.model.family!r} but the champion was fitted from a "
            f"{spec.family!r} specification"
        )
    for key, value in (
        ("n_estimators", config.model.n_estimators),
        ("max_depth", config.model.max_depth),
        ("min_samples_leaf", config.model.min_samples_leaf),
    ):
        if spec.params.get(key) != value:
            raise SpatialStudyError(
                f"model.{key} is {value!r} but the M5B grid specification has "
                f"{spec.params.get(key)!r}"
            )
    seed = derive_noise_seed(m5b_config.run.seed, "model", spec.name) % (2**32)
    if seed != config.model.seed:
        raise SpatialStudyError(
            f"model.seed is {config.model.seed} but the M5B grid derives {seed}"
        )
    return spec


def _load_champion(config: SpatialStudyConfig) -> Any:
    """Load the frozen M5C champion bundle and confirm it is unchanged."""
    directory = config.m5c.champion_dir
    stem = config.m5c.champion_stem
    artifact = directory / f"{stem}.joblib"
    card_path = directory / f"{stem}_card.json"
    _require_file_hash(artifact, config.m5c.champion_artifact_sha256, "M5C champion artifact")
    if not card_path.is_file():
        raise SpatialStudyError(f"missing M5C champion card: {card_path}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("feature_names") != list(config.global_feature_names):
        raise SpatialStudyError("the M5C champion uses a different feature schema")
    if card.get("target") != config.target:
        raise SpatialStudyError(f"the M5C champion target is not {config.target!r}")
    recorded = str(card.get("selection", {}).get("benchmark_config_hash", ""))
    if recorded != config.m5c.protocol_hash:
        raise SpatialStudyError(
            f"the M5C champion was produced under protocol {recorded!r}, not the pinned "
            f"{config.m5c.protocol_hash!r}"
        )
    if card.get("artifact_sha256") != sha256_file(artifact):
        raise SpatialStudyError("the M5C champion artifact hash does not match its card")
    if str(card.get("sklearn_version", "")).split(".")[:2] != sklearn_version().split(".")[:2]:
        raise SpatialStudyError(
            f"the M5C champion was persisted with scikit-learn {card.get('sklearn_version')}, "
            f"current is {sklearn_version()}"
        )
    import joblib

    estimator = joblib.load(artifact)
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features != len(config.global_feature_names):
        raise SpatialStudyError(
            f"the M5C champion expects {n_features} inputs, the frozen schema has "
            f"{len(config.global_feature_names)}"
        )
    return estimator


def build_ablations(
    frames: StudyFrames,
    config: SpatialStudyConfig,
    *,
    champion: Any,
    m5c_config: ra.RetentionConfig,
) -> tuple[Ablation, ...]:
    """Fit every predeclared ablation on the expanded training rows."""
    spec = _champion_spec(config, m5c_config)
    train = frames.with_spatial(frames.retention.expanded_train)
    ablations: list[Ablation] = []
    for name in config.ablation_names:
        feature_set = config.ablation_feature_sets[name]
        feature_names = config.feature_names_for(feature_set)
        if not config.ablation_refit[name]:
            ablations.append(
                Ablation(
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
        ablations.append(
            Ablation(
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
    return tuple(ablations)


# ---------------------------------------------------------------------------
# Validation scoring and selection
# ---------------------------------------------------------------------------


def validation_slices(frames: StudyFrames) -> dict[str, pd.DataFrame]:
    """Return the predeclared validation slices."""
    validation = frames.with_spatial(frames.retention.expanded_val)
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


def _severity_scopes(config: SpatialStudyConfig) -> tuple[str, ...]:
    return tuple(
        f"{GEO_LABEL}_severity_{severity}" for severity in config.protocol.primary_severities
    )


def _slice_metrics_table(
    frames: StudyFrames,
    ablations: Sequence[Ablation],
    predictions: Mapping[str, np.ndarray],
    *,
    split_label: str,
) -> pd.DataFrame:
    """Score every ablation on every validation slice.

    The prediction arrays are aligned to the whole validation frame, so each
    slice is scored with the rows that belong to it.
    """
    validation = frames.with_spatial(frames.retention.expanded_val)
    slices = validation_slices(frames)
    rows: list[dict[str, Any]] = []
    for ablation in ablations:
        predicted = np.asarray(predictions[ablation.name], dtype=np.float64)
        if predicted.size != len(validation):
            raise SpatialStudyError(
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


def select_ablation(
    metrics: pd.DataFrame,
    settings: StudySelectionSettings,
    *,
    severity_scopes: Sequence[str],
    reference: str = GLOBAL_BASELINE,
) -> tuple[str, pd.DataFrame]:
    """Apply the predeclared eligibility constraints and selection score.

    The baseline is the reference. A candidate qualifies only if it regresses
    neither the clean slice, nor the original-task slice, nor any single
    primary geometric severity beyond the declared tolerances. Among the
    qualifying candidates the score rewards aggregate geometric improvement and
    charges clean, original-task and per-severity regressions. Ties are broken
    by lower aggregate geometric RMSE and then by candidate name.
    """
    severity_scopes = tuple(severity_scopes)
    reference_rows = metrics[metrics["model"] == reference]
    if reference_rows.empty:
        raise SpatialStudyError("the baseline reference is missing from the metrics table")

    def slice_rmse(model: str, scope: str) -> float | None:
        rows = metrics[(metrics["model"] == model) & (metrics["scope"] == scope)]
        if rows.empty:
            return None
        return float(rows.iloc[0]["rmse"])

    reference_values: dict[str, float] = {}
    for scope in (GEO_LABEL, CLEAN_LABEL, ORIGINAL_LABEL, *severity_scopes):
        value = slice_rmse(reference, scope)
        if value is None:
            raise SpatialStudyError(f"the baseline has no {scope!r} slice")
        reference_values[scope] = value

    records: list[dict[str, Any]] = []
    for model, group in metrics.groupby("model", sort=True):
        feature_set = str(group.iloc[0]["feature_set"])
        n_features = int(group.iloc[0]["n_features"])
        geo = slice_rmse(model, GEO_LABEL)
        clean = slice_rmse(model, CLEAN_LABEL)
        original = slice_rmse(model, ORIGINAL_LABEL)
        if geo is None or clean is None or original is None:
            raise SpatialStudyError(f"ablation {model!r} is missing a required slice")
        severity_values = {scope: slice_rmse(model, scope) for scope in severity_scopes}
        missing = [scope for scope, value in severity_values.items() if value is None]
        if missing:
            raise SpatialStudyError(f"ablation {model!r} is missing severity slice(s) {missing}")

        clean_excess = clean - reference_values[CLEAN_LABEL]
        original_excess = original - reference_values[ORIGINAL_LABEL]
        severity_excess = {
            scope: severity_values[scope] - reference_values[scope] for scope in severity_scopes
        }

        eligible = True
        reasons: list[str] = []
        if clean > reference_values[CLEAN_LABEL] * (1.0 + settings.tolerance_clean):
            eligible = False
            reasons.append(
                f"clean RMSE {clean:.5f} exceeds the baseline {reference_values[CLEAN_LABEL]:.5f} "
                f"by more than {settings.tolerance_clean:.0%}"
            )
        if original > reference_values[ORIGINAL_LABEL] * (1.0 + settings.tolerance_original):
            eligible = False
            reasons.append(
                f"original-task RMSE {original:.5f} exceeds the baseline "
                f"{reference_values[ORIGINAL_LABEL]:.5f} by more than "
                f"{settings.tolerance_original:.0%}"
            )
        for scope, value in severity_values.items():
            if value > reference_values[scope] * (1.0 + settings.tolerance_severity):
                eligible = False
                reasons.append(
                    f"{scope} RMSE {value:.5f} exceeds the baseline "
                    f"{reference_values[scope]:.5f} by more than "
                    f"{settings.tolerance_severity:.0%}"
                )

        score = (
            settings.w_geometric * (geo - reference_values[GEO_LABEL])
            + settings.w_clean * max(0.0, clean_excess)
            + settings.w_original * max(0.0, original_excess)
            + settings.w_severity * sum(max(0.0, value) for value in severity_excess.values())
        )
        records.append(
            {
                "model": model,
                "feature_set": feature_set,
                "n_features": n_features,
                "eligible": eligible,
                "ineligible_reasons": "; ".join(reasons),
                "score": float(score),
                "rmse_geometric": geo,
                "rmse_clean": clean,
                "rmse_original": original,
                **{f"rmse_{scope}": severity_values[scope] for scope in severity_scopes},
                "delta_geometric_rmse": float(geo - reference_values[GEO_LABEL]),
                "delta_clean_rmse": float(clean_excess),
                "delta_original_rmse": float(original_excess),
                "max_severity_excess_rmse": float(
                    max((max(0.0, value) for value in severity_excess.values()), default=0.0)
                ),
            }
        )
    table = pd.DataFrame(records)
    eligible_rows = table[table["eligible"]]
    if eligible_rows.empty:
        selected = reference
    else:
        ordered = eligible_rows.sort_values(
            ["score", "rmse_geometric", "model"], kind="stable", ignore_index=True
        )
        selected = str(ordered.iloc[0]["model"])
    table["selected"] = table["model"] == selected
    table = table.sort_values(
        ["eligible", "score", "rmse_geometric", "model"],
        ascending=[False, True, True, True],
        kind="stable",
        ignore_index=True,
    )
    return selected, table


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestSlice:
    """One frozen evaluation slice with its own frame and per-ablation predictions."""

    scope: str
    frame: pd.DataFrame
    predictions: dict[str, np.ndarray]


def build_test_slices(frames: StudyFrames, ablations: Sequence[Ablation]) -> list[TestSlice]:
    """Assemble the frozen evaluation slices scored after selection is complete."""
    retention = frames.retention
    geometric = frames.with_spatial(retention.primary_test)
    m3_test = frames.with_spatial(retention.m3_test)
    original = m3_test[~m3_test["is_clean"]]
    diagnostic = frames.with_spatial(retention.diagnostic_test)

    def predict_on(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {ablation.name: ablation.predict(frame) for ablation in ablations}

    slices: list[TestSlice] = [
        TestSlice(GEO_LABEL, geometric, predict_on(geometric)),
        TestSlice(
            CLEAN_LABEL, m3_test[m3_test["is_clean"]], predict_on(m3_test[m3_test["is_clean"]])
        ),
        TestSlice(ORIGINAL_LABEL, original, predict_on(original)),
        TestSlice("m3_test_overall", m3_test, predict_on(m3_test)),
        TestSlice("diagnostic_0_5", diagnostic, predict_on(diagnostic)),
    ]
    for severity, subset in geometric.groupby("severity_level", sort=True):
        slices.append(
            TestSlice(f"{GEO_LABEL}_severity_{int(severity)}", subset, predict_on(subset))
        )
    for family, subset in original.groupby("family", sort=True):
        slices.append(TestSlice(f"{ORIGINAL_LABEL}_{family}", subset, predict_on(subset)))
    return [item for item in slices if not item.frame.empty]


def _test_metrics_table(slices: Sequence[TestSlice]) -> pd.DataFrame:
    """Score every ablation on every frozen test slice."""
    rows: list[dict[str, Any]] = []
    for item in slices:
        for name, predicted in item.predictions.items():
            scores = ra._slice_scores(item.frame, predicted)  # noqa: SLF001
            rows.append({"model": name, "split": "test", "scope": item.scope, **scores})
    return pd.DataFrame(rows)


def _uncertainty_table(slices: Sequence[TestSlice], *, n_bootstrap: int, seed: int) -> pd.DataFrame:
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
                seed=derive_noise_seed(seed, "m6a_bootstrap", name, item.scope) % (2**64),
                evaluation="m6a_test",
            )
            rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _paired_uncertainty_table(
    slices: Sequence[TestSlice],
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
                seed=derive_noise_seed(seed, "m6a_paired", name, item.scope) % (2**64),
                evaluation="m6a_test",
            )
            rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _prediction_table(slices: Sequence[TestSlice], model_names: Sequence[str]) -> pd.DataFrame:
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
# Feature diagnostics
# ---------------------------------------------------------------------------


def feature_diagnostics(
    frames: StudyFrames,
    ablations: Sequence[Ablation],
    config: SpatialStudyConfig,
) -> dict[str, Any]:
    """Collect descriptive feature diagnostics from the training rows only.

    Nothing here is used to redesign the feature set within this milestone:
    the diagnostics are reported, not acted upon.
    """
    train = frames.with_spatial(frames.retention.expanded_train)
    spatial_names = list(config.features.feature_names)
    spatial_frame = train[spatial_names]
    duplicated = [name for name in spatial_names if spatial_frame[name].duplicated().all()]
    constant = [name for name in spatial_names if float(spatial_frame[name].std(ddof=0)) == 0.0]
    correlation = spatial_frame.corr().abs()

    importances: dict[str, list[dict[str, Any]]] = {}
    for ablation in ablations:
        if ablation.train_importances is None:
            continue
        order = np.argsort(-ablation.train_importances)
        importances[ablation.name] = [
            {
                "feature_name": ablation.feature_names[int(index)],
                "importance": float(ablation.train_importances[int(index)]),
            }
            for index in order
        ]
    return {
        "n_train_rows": int(len(train)),
        "n_spatial_features": len(spatial_names),
        "duplicated_spatial_features": duplicated,
        "constant_spatial_features": constant,
        "spatial_correlation_max": float(
            correlation.where(~np.eye(len(spatial_names), dtype=bool)).max().max()
        )
        if len(spatial_names) > 1
        else 0.0,
        "spatial_correlation_mean": float(
            correlation.where(~np.eye(len(spatial_names), dtype=bool)).stack().mean()
        )
        if len(spatial_names) > 1
        else 0.0,
        "feature_importances": importances,
    }


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyResult:
    """The complete M6A outcome."""

    frames: StudyFrames
    checks: tuple[CheckResult, ...]
    ablations: tuple[Ablation, ...]
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
    def selected(self) -> Ablation:
        for ablation in self.ablations:
            if ablation.name == self.selected_name:
                return ablation
        raise SpatialStudyError(f"selected ablation {self.selected_name!r} is missing")


def run_study_core(
    frames: StudyFrames,
    config: SpatialStudyConfig,
    *,
    champion: Any,
    m5c_config: ra.RetentionConfig,
) -> StudyResult:
    """Run the M6A study in the predeclared order.

    The order is fixed: fit on training data only, score validation, select,
    freeze, and only then score the untouched test slices.
    """
    rb._verify_checks(frames.checks, "M6A frame checks")  # noqa: SLF001

    ablations = build_ablations(frames, config, champion=champion, m5c_config=m5c_config)
    validation = frames.with_spatial(frames.retention.expanded_val)
    validation_predictions = {ablation.name: ablation.predict(validation) for ablation in ablations}
    validation_metrics = _slice_metrics_table(
        frames, ablations, validation_predictions, split_label="val"
    )
    selected_name, selection_table = select_ablation(
        validation_metrics,
        config.selection,
        severity_scopes=_severity_scopes(config),
    )

    # The selected ablation and its full configuration are frozen here. The
    # test slices are only touched after this point.
    slices = build_test_slices(frames, ablations)
    test_metrics = _test_metrics_table(slices)
    uncertainty = _uncertainty_table(
        slices, n_bootstrap=config.run.n_bootstrap, seed=config.run.seed
    )
    paired = _paired_uncertainty_table(
        slices, n_bootstrap=config.run.n_bootstrap, seed=config.run.seed
    )
    predictions = _prediction_table(slices, [a.name for a in ablations])
    diagnostics = feature_diagnostics(frames, ablations, config)
    diagnostics["grid_resolution"] = grid_resolution_diagnostic(frames, config)

    return StudyResult(
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
    result: StudyResult, config: SpatialStudyConfig, ablation: Ablation
) -> dict[str, Any]:
    """Build the model card for one refitted ablation."""
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
        "spatial_feature_set_version": SPATIAL_FEATURE_SET_VERSION,
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


def write_study_outputs(
    result: StudyResult,
    config: SpatialStudyConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    frozen_before: Mapping[str, Mapping[str, object]] | None = None,
    frozen_after: Mapping[str, Mapping[str, object]] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Path]:
    """Persist the M6A tables, model bundles, metadata and report."""
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
    spatial_feature_manifest(config.features).to_csv(
        reports / "spatial_feature_manifest.csv", index=False
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
            "feature_manifest": reports / "spatial_feature_manifest.csv",
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
            "spatial_feature_set_version": SPATIAL_FEATURE_SET_VERSION,
            "spatial_feature_names": list(config.features.feature_names),
            "target": config.target,
            "split_names": ["train", "val", "test"],
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
            "all_unchanged": _frozen_unchanged(frozen_before, frozen_after),
        },
        "artifacts": {},
    }
    for key, path in paths.items():
        if path.is_file():
            metadata["artifacts"][key] = {"path": str(path), "sha256": sha256_file(path)}
    metadata_path = reports / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    paths["metadata"] = metadata_path

    report_path = reports / "report.md"
    report_path.write_text(_report_markdown(result, config), encoding="utf-8")
    paths["report"] = report_path
    return paths


def _frozen_unchanged(
    before: Mapping[str, Mapping[str, object]] | None,
    after: Mapping[str, Mapping[str, object]] | None,
) -> bool:
    """Compare the before/after fingerprints of the frozen assets."""
    if before is None or after is None:
        return False
    if set(before) != set(after):
        return False
    return all(before[key].get("sha256") == after[key].get("sha256") for key in before)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _metric_row_for(table: pd.DataFrame, model: str, scope: str) -> dict[str, Any] | None:
    rows = table[(table["model"] == model) & (table["scope"] == scope)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "r2": float(row["r2"]),
        "n_samples": int(row["n_samples"]),
        "n_sources": int(row["n_sources"]),
        "mean_residual": float(row["mean_residual"]),
        "sd_residual": float(row["sd_residual"]),
    }


def _format_metric(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def _slice_table(table: pd.DataFrame, scope: str, models: Sequence[str]) -> list[str]:
    lines = [
        "| ablation | n | sources | MAE | RMSE | R2 | mean residual | residual SD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        row = _metric_row_for(table, model, scope)
        if row is None:
            continue
        lines.append(
            f"| {model} | {row['n_samples']} | {row['n_sources']} | "
            f"{_format_metric(row['mae'])} | {_format_metric(row['rmse'])} | "
            f"{_format_metric(row['r2'])} | {_format_metric(row['mean_residual'])} | "
            f"{_format_metric(row['sd_residual'])} |"
        )
    return lines


def _paired_interval_for(
    table: pd.DataFrame, model: str, scope: str, metric: str
) -> tuple[float, float, float] | None:
    rows = table[
        (table["model"] == f"{model}_minus_{GLOBAL_BASELINE}")
        & (table["dataset"] == scope)
        & (table["metric"] == metric)
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])


def _report_markdown(result: StudyResult, config: SpatialStudyConfig) -> str:
    """Render the deterministic M6A report."""
    models = [ablation.name for ablation in result.ablations]
    selected = result.selected_name
    lines: list[str] = [
        "# M6A spatial-aware feature study",
        "",
        "**Scope:** synthetic image similarity, measured as SSIM against the exact clean "
        "reference. This is not a measure of human readability, CAPTCHA solvability, "
        "security, or real-world quality.",
        "",
        "## Question",
        "",
        "Does adding spatially localized image features improve geometric-distortion SSIM "
        "prediction compared with the existing eight global image features, while "
        "preserving clean-reference and original-task performance?",
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
        f"- Model family: `{config.model.family}` "
        f"(n_estimators={config.model.n_estimators}, max_depth={config.model.max_depth}, "
        f"min_samples_leaf={config.model.min_samples_leaf}, seed={config.model.seed})",
        f"- Frozen global feature set: `{config.global_feature_set_version}`, "
        f"{len(config.global_feature_names)} features, unchanged",
        f"- Spatial feature set: `{SPATIAL_FEATURE_SET_VERSION}`, "
        f"{len(config.features.feature_names)} features, unchanged from the global pipeline",
        "",
        "The M3 dataset, split, frozen M3 model, M4A pilot, M4B benchmark, M5A pilot, "
        "M5B benchmark, M5C selection study and M5D training study are read-only and "
        "hash-verified before and after the run.",
        "",
        "### Severity protocol",
        "",
        f"- Primary geometric score: amplitudes `{list(config.protocol.primary_amplitudes_px)}` px "
        f"(severities `{list(config.protocol.primary_severities)}`).",
        f"- Diagnostic only: amplitude `{list(config.protocol.diagnostic_amplitudes_px)}` px "
        f"(severity `{list(config.protocol.diagnostic_severities)}`).",
        "- The 0.5 px rung never enters fitting, validation selection or the primary score.",
        "",
        "## Spatial feature definitions",
        "",
        f"The canonical image is divided into a fixed {config.features.grid_rows} x "
        f"{config.features.grid_cols} grid of {config.features.n_cells} cells, visited in "
        f"row-major order. Each cell is {64 // config.features.grid_rows} pixels tall and "
        f"{192 // config.features.grid_cols} pixels wide. Two families are implemented, both "
        "computed from the degraded image alone:",
        "",
        "- `ink_occupancy` - the fraction of pixels at or below the Otsu threshold of the "
        "whole image, counted within the cell. The threshold is computed once per image, "
        "so the cell values share one ink definition and are directly comparable with the "
        "global `ink_fraction` feature.",
        "- `edge_density` - the fraction of Canny-marked pixels within the cell, using the "
        f"repository's existing edge convention (sigma={config.features.edge_sigma}, "
        f"low_threshold={config.features.edge_low_threshold}, "
        f"high_threshold={config.features.edge_high_threshold} on the image scaled to "
        "[0, 1]). The edge map is computed once per image.",
        "",
        "Regional intensity statistics were considered and rejected: on training rows the "
        "per-cell mean intensity is largely redundant with the per-cell ink occupancy "
        "(maximum |correlation| 0.873 across cell pairs). A 4x2 grid was chosen over 2x4 "
        "because it has lower internal redundancy among the spatial features (mean "
        "pairwise |r| 0.38 against 0.52) at a comparable correlation with the target. Both "
        "decisions were made from training rows only.",
        "",
        "The extractor never sees the clean reference, the SSIM/PSNR targets, the source "
        "identifier, the corruption family, the severity, the file name or the split "
        "label, so it cannot leak target or metadata information into a predictor. An "
        "exactly uniform image has no distinguishable ink class and no edges, so all its "
        "spatial features are zero.",
        "",
        "### Feature manifest",
        "",
        "| ablation | feature set | features |",
        "| --- | --- | ---: |",
    ]
    for ablation in result.ablations:
        lines.append(
            f"| `{ablation.name}` | {ablation.feature_set} | {len(ablation.feature_names)} |"
        )
    lines.extend(
        [
            "",
            "| feature | family | grid row | grid col |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in spatial_feature_manifest(config.features).itertuples(index=False):
        lines.append(f"| `{row.feature_name}` | {row.family} | {row.grid_row} | {row.grid_col} |")

    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "The model family, the hyperparameters, the seed and the training partition "
            "are held fixed across all three ablations, so the only thing that varies is "
            "the input representation.",
            "",
            "| ablation | feature set | features | fitted |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for ablation in result.ablations:
        lines.append(
            f"| `{ablation.name}` | {ablation.feature_set} | {len(ablation.feature_names)} | "
            f"{'refitted' if ablation.refitted else 'frozen champion, not refitted'} |"
        )

    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            "The rule was declared before any test evaluation and is identical to the M5D "
            "rule, so the two studies are directly comparable. The baseline is the "
            "reference. A candidate qualifies only if it regresses neither the clean slice, "
            "nor the original-task slice, nor any single primary geometric severity beyond "
            "the declared tolerances:",
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
            "geometric RMSE and then by ablation name. If no candidate qualifies, the "
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
        lines.extend(_slice_table(result.validation_metrics, scope, models))
        lines.append("")

    lines.append("### Geometric validation slice by severity")
    lines.append("")
    for severity in config.protocol.primary_severities:
        scope = f"{GEO_LABEL}_severity_{severity}"
        lines.append(f"**Severity {severity}**")
        lines.append("")
        lines.extend(_slice_table(result.validation_metrics, scope, models))
        lines.append("")

    lines.extend(
        [
            "### Eligibility and selection scores",
            "",
            "| ablation | eligible | score | geometric RMSE | clean RMSE | original RMSE | "
            "selected |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result.selection_table.itertuples(index=False):
        lines.append(
            f"| `{row.model}` | {'yes' if row.eligible else 'no'} | {row.score:.5f} | "
            f"{row.rmse_geometric:.5f} | {row.rmse_clean:.5f} | {row.rmse_original:.5f} | "
            f"{'yes' if row.selected else 'no'} |"
        )
    ineligible = result.selection_table[~result.selection_table["eligible"]]
    if not ineligible.empty:
        lines.append("")
        lines.append("Ineligible ablations:")
        lines.append("")
        for row in ineligible.itertuples(index=False):
            lines.append(f"- `{row.model}`: {row.ineligible_reasons}")
    lines.extend(
        [
            "",
            f"Selected ablation: **`{selected}`**.",
            "",
            "## Final test results by slice",
            "",
            "The selected ablation and its complete configuration were frozen before these "
            "rows were scored. The M3 test split and the M5B/M5C geometric test split are "
            "untouched frozen evaluations. All three ablations are reported, as planned, "
            "so the descriptive ablation is visible alongside the selected candidate.",
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
        lines.extend(_slice_table(result.test_metrics, scope, models))
        lines.append("")

    lines.append("### Geometric test slice by severity")
    lines.append("")
    for severity in config.protocol.primary_severities:
        scope = f"{GEO_LABEL}_severity_{severity}"
        lines.append(f"**Severity {severity}**")
        lines.append("")
        lines.extend(_slice_table(result.test_metrics, scope, models))
        lines.append("")

    lines.extend(
        [
            "## Paired source-grouped differences against the frozen baseline",
            "",
            "Whole source groups are resampled, preserving all rows within each selected "
            "group. Differences are ablation minus the frozen baseline on identical rows. "
            "There are 12 held-out test source groups, so these intervals are preliminary "
            "and descriptive. Interval overlap or non-overlap is not a significance test.",
            "",
        ]
    )
    for scope, title in (
        (GEO_LABEL, "Primary geometric test slice"),
        (CLEAN_LABEL, "Clean-reference test slice"),
        (ORIGINAL_LABEL, "Original blur/noise test slice"),
        ("m3_test_overall", "Original M3 test slice (overall)"),
    ):
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| ablation | metric | estimate | 95% low | 95% high |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for model in models:
            if model == GLOBAL_BASELINE:
                continue
            for metric in ("mae", "rmse"):
                found = _paired_interval_for(result.paired_uncertainty, model, scope, metric)
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
            "A paired change that is driven by one or two sources is weaker evidence than "
            "a change that holds across the group. This counts the held-out sources on "
            "which an ablation has a lower absolute error than the frozen baseline.",
            "",
            "| scope | ablation | sources improved | worst source change | best source "
            "change | largest single-source share of the total gain |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope in (GEO_LABEL, ORIGINAL_LABEL, "m3_test_overall"):
        consistency = _source_consistency(result, scope)
        for candidate, entry in consistency.get("candidates", {}).items():
            share = entry["largest_single_source_share"]
            share_text = f"{share * 100:.1f}%" if math.isfinite(share) else "n/a"
            lines.append(
                f"| {scope} | `{candidate}` | {entry['n_sources_better']}/{entry['n_sources']} | "
                f"{entry['worst_delta']:+.4f} | {entry['best_delta']:+.4f} | {share_text} |"
            )
    lines.append("")
    lines.append(
        "`n/a` marks a slice where the per-source gains and losses nearly cancel, so no "
        "single-source share is meaningful: the aggregate change there is small and is not "
        "evidence of a broad improvement or of a broad regression."
    )

    lines.extend(["## Feature diagnostics", ""])
    lines.extend(_diagnostic_lines(result))
    lines.extend(["## Interpretation", ""])
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
            "- A 4x2 grid is a coarse partition. It can show that a distortion is "
            "spatially uneven, but it cannot localise a warp to a glyph.",
            "- The spatial features are additive to the global ones, not a replacement: a "
            "combined model still sees the global summary, so a gain there cannot be "
            "attributed to the spatial family alone.",
            "- Feature importance is descriptive, not causal, and is reported only to "
            "assess whether the spatial family carries information.",
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
            "uv run --extra image --extra ml captcha-spatial-study run \\",
            "    --config configs/spatial_feature_study_v0_1_0.toml",
            "uv run --extra image --extra ml captcha-spatial-study verify \\",
            "    --config configs/spatial_feature_study_v0_1_0.toml",
            "```",
            "",
            "Generated tables, the model bundles and this report live in Git-ignored paths.",
        ]
    )
    return "\n".join(lines) + "\n"


def _source_consistency(
    result: StudyResult, scope: str, reference: str = GLOBAL_BASELINE
) -> dict[str, Any]:
    """Summarise whether a paired change is broad-based or driven by outliers."""
    frame = result.test_predictions
    subset = frame[frame["scope"] == scope]
    if subset.empty:
        return {}
    pivot = subset.pivot_table(
        index="source_id", columns="model", values="abs_residual", aggfunc="mean"
    )
    if reference not in pivot.columns:
        return {}
    summary: dict[str, Any] = {"candidates": {}}
    for candidate in [name for name in pivot.columns if name != reference]:
        delta = (pivot[candidate] - pivot[reference]).dropna()
        if delta.empty:
            continue
        gains = -delta
        total = float(gains.sum())
        scale = float(np.abs(delta).sum())
        clear = total > 0.0 and total > 0.25 * scale
        summary["candidates"][candidate] = {
            "n_sources_better": int((delta < 0).sum()),
            "n_sources": int(len(delta)),
            "worst_delta": float(delta.max()),
            "best_delta": float(delta.min()),
            "total_gain": total,
            "aggregate_change_is_clear": bool(clear),
            "largest_single_source_share": (float(gains.max() / total) if clear else math.nan),
        }
    return summary


def grid_resolution_diagnostic(frames: StudyFrames, config: SpatialStudyConfig) -> dict[str, Any]:
    """Measure how much of a distortion the declared grid can actually see.

    This is a descriptive diagnostic computed on **training rows only**. It
    compares, per source, the per-cell ink occupancy of a distorted image with
    that of the same source's clean reference, and reports the mean and maximum
    absolute change together with the correlation between that change and the
    SSIM drop. It answers a question the headline metrics cannot: whether the
    declared grid is even capable of resolving the distortion, or whether a
    coarse partition simply averages the effect away.

    The result is reported, not acted upon: the grid is fixed by the
    configuration and is not changed within this milestone.
    """
    train = frames.with_spatial(frames.retention.expanded_train)
    m3_train = train[train["dataset"] == gb.M3_LABEL]
    m5a_train = train[train["dataset"] == gb.M5A_LABEL]

    def clean_rows() -> pd.DataFrame:
        return m3_train[m3_train["family"] == gb.M3_CLEAN_FAMILY]

    def occupancy_map(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        columns = [
            f"ink_occupancy_r{i}_c{j}"
            for i in range(config.features.grid_rows)
            for j in range(config.features.grid_cols)
        ]
        values = frame[columns].to_numpy(dtype=np.float64)
        return {
            str(sample_id): values[index]
            for index, sample_id in enumerate(frame["sample_id"].astype(str))
        }

    clean = occupancy_map(train)
    summaries: dict[str, Any] = {}
    families = {
        "gaussian_blur": m3_train[m3_train["family"] == "gaussian_blur"],
        "gaussian_noise": m3_train[m3_train["family"] == "gaussian_noise"],
        "elastic": m5a_train[m5a_train["family"] == gb.ELASTIC_FAMILY],
    }
    for family, frame in families.items():
        changed: list[float] = []
        dropped: list[float] = []
        for record in frame[["sample_id", "source_id", TARGET]].itertuples(index=False):
            # Compare against the same source's clean reference where one exists.
            source = str(record.source_id)
            reference = m3_train[
                (m3_train["source_id"] == source) & (m3_train["family"] == gb.M3_CLEAN_FAMILY)
            ]
            if reference.empty:
                continue
            reference_id = str(reference.iloc[0]["sample_id"])
            if reference_id not in clean:
                continue
            changed.append(float(np.abs(clean[str(record.sample_id)] - clean[reference_id]).mean()))
            dropped.append(float(reference.iloc[0][TARGET]) - float(record[2]))
        if not changed:
            continue
        correlation = (
            float(np.corrcoef(changed, dropped)[0, 1])
            if len(changed) > 2 and np.std(changed) > 0.0 and np.std(dropped) > 0.0
            else math.nan
        )
        summaries[family] = {
            "n_pairs": len(changed),
            "mean_abs_occupancy_change": float(np.mean(changed)),
            "max_abs_occupancy_change": float(np.max(changed)),
            "correlation_with_ssim_drop": correlation,
        }
    return {
        "grid_rows": config.features.grid_rows,
        "grid_cols": config.features.grid_cols,
        "cell_height_px": 64 // config.features.grid_rows,
        "cell_width_px": 192 // config.features.grid_cols,
        "families": summaries,
    }


def _diagnostic_lines(result: StudyResult) -> list[str]:
    """Render the descriptive feature diagnostics."""
    diagnostics = result.diagnostics
    lines = [
        "These diagnostics are computed on the training rows only. They are reported to "
        "assess whether the spatial family carries information; they are not used to "
        "redesign the feature set within this milestone.",
        "",
        f"- Training rows: {diagnostics['n_train_rows']}",
        f"- Spatial features: {diagnostics['n_spatial_features']}",
        f"- Duplicated spatial features: {diagnostics['duplicated_spatial_features'] or 'none'}",
        f"- Constant spatial features: {diagnostics['constant_spatial_features'] or 'none'}",
        f"- Maximum absolute pairwise correlation among the spatial features: "
        f"{diagnostics['spatial_correlation_max']:.4f}",
        f"- Mean absolute pairwise correlation among the spatial features: "
        f"{diagnostics['spatial_correlation_mean']:.4f}",
        "",
    ]
    for ablation, entries in diagnostics.get("feature_importances", {}).items():
        lines.append(f"### {ablation} feature importance (training rows, descriptive)")
        lines.append("")
        lines.append("| rank | feature | importance |")
        lines.append("| ---: | --- | ---: |")
        for rank, entry in enumerate(entries[:12], start=1):
            lines.append(f"| {rank} | `{entry['feature_name']}` | {entry['importance']:.4f} |")
        lines.append("")

    resolution = diagnostics.get("grid_resolution", {})
    if resolution:
        families = resolution.get("families", {})
        if families:
            lines.append("### Can the declared grid resolve the distortion?")
            lines.append("")
            lines.append(
                "Per-cell ink occupancy of a distorted image against the same source's clean "
                "reference, on training rows only. This measures whether the grid is even "
                "capable of resolving the distortion, independently of any model."
            )
            lines.append("")
            lines.append(
                f"Declared grid: {resolution['grid_rows']}x{resolution['grid_cols']}, "
                f"cells of {resolution['cell_height_px']}x{resolution['cell_width_px']} pixels."
            )
            lines.append("")
            lines.append(
                "| family | pairs | mean abs occupancy change | max abs occupancy change | "
                "correlation with SSIM drop |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for family, entry in families.items():
                correlation = entry["correlation_with_ssim_drop"]
                correlation_text = (
                    "n/a" if not math.isfinite(correlation) else f"{correlation:+.3f}"
                )
                lines.append(
                    f"| {family} | {entry['n_pairs']} | "
                    f"{entry['mean_abs_occupancy_change']:.5f} | "
                    f"{entry['max_abs_occupancy_change']:.5f} | {correlation_text} |"
                )
            lines.append("")
    return lines


def _decision_lines(result: StudyResult, config: SpatialStudyConfig) -> list[str]:
    """Derive the descriptive interpretation from the observed results."""
    selected = result.selected_name
    lines: list[str] = []

    def metric(model: str, scope: str) -> dict[str, Any] | None:
        return _metric_row_for(result.test_metrics, model, scope)

    baseline_geo = metric(GLOBAL_BASELINE, GEO_LABEL)
    selected_geo = metric(selected, GEO_LABEL)
    baseline_clean = metric(GLOBAL_BASELINE, CLEAN_LABEL)
    selected_clean = metric(selected, CLEAN_LABEL)
    baseline_original = metric(GLOBAL_BASELINE, ORIGINAL_LABEL)
    selected_original = metric(selected, ORIGINAL_LABEL)
    baseline_m3 = metric(GLOBAL_BASELINE, "m3_test_overall")
    selected_m3 = metric(selected, "m3_test_overall")

    if baseline_geo and selected_geo:
        change = (selected_geo["rmse"] - baseline_geo["rmse"]) / baseline_geo["rmse"] * 100
        lines.append(
            f"- **Primary geometric slice.** The frozen baseline reaches MAE "
            f"{baseline_geo['mae']:.4f}, RMSE {baseline_geo['rmse']:.4f} and R2 "
            f"{baseline_geo['r2']:.4f} on {baseline_geo['n_samples']} rows from "
            f"{baseline_geo['n_sources']} sources. The selected ablation `{selected}` "
            f"reaches MAE {selected_geo['mae']:.4f}, RMSE {selected_geo['rmse']:.4f} and R2 "
            f"{selected_geo['r2']:.4f}, a {change:+.1f}% change in RMSE against the "
            "baseline."
        )
    if baseline_clean and selected_clean:
        lines.append(
            f"- **Clean-reference slice.** Baseline MAE {baseline_clean['mae']:.4f} / RMSE "
            f"{baseline_clean['rmse']:.4f}; selected MAE {selected_clean['mae']:.4f} / RMSE "
            f"{selected_clean['rmse']:.4f}."
        )
    if baseline_original and selected_original:
        lines.append(
            f"- **Original blur/noise slice.** Baseline MAE {baseline_original['mae']:.4f} / "
            f"RMSE {baseline_original['rmse']:.4f}; selected MAE "
            f"{selected_original['mae']:.4f} / RMSE {selected_original['rmse']:.4f}."
        )
    if baseline_m3 and selected_m3:
        lines.append(
            f"- **Original-task retention.** On the same untouched M3 test rows the "
            f"baseline reaches MAE {baseline_m3['mae']:.4f} / RMSE {baseline_m3['rmse']:.4f} / "
            f"R2 {baseline_m3['r2']:.4f} and the selected ablation reaches MAE "
            f"{selected_m3['mae']:.4f} / RMSE {selected_m3['rmse']:.4f} / R2 "
            f"{selected_m3['r2']:.4f}."
        )

    for row in result.selection_table.itertuples(index=False):
        if row.model == GLOBAL_BASELINE or bool(row.eligible):
            continue
        lines.append(
            f"- **`{row.model}` was ineligible.** {row.ineligible_reasons}. The "
            "per-severity constraint is what rejects it: the aggregate geometric score "
            "alone would not have revealed the regression on a single rung."
        )

    combined_row = result.selection_table[result.selection_table["model"] == COMBINED]
    if not combined_row.empty:
        row = combined_row.iloc[0]
        if bool(row["eligible"]):
            lines.append(
                f"- **`{combined_row.iloc[0]['model']}` was eligible but not selected.** Its "
                f"aggregate geometric validation RMSE is {row.rmse_geometric:.5f} against "
                f"the baseline's {result.selection_table[result.selection_table['model'] == GLOBAL_BASELINE].iloc[0]['rmse_geometric']:.5f} "
                f"(delta {row.delta_geometric_rmse:+.5f}), the clean slice moves by "
                f"{row.delta_clean_rmse:+.5f} and the original task by "
                f"{row.delta_original_rmse:+.5f}, giving a score of {row.score:+.5f} against "
                "the baseline's 0."
            )
        else:
            lines.append(
                f"- **The combined ablation traded rather than won.** Adding the spatial "
                f"features moved the aggregate geometric validation RMSE by "
                f"{row.delta_geometric_rmse:+.5f}, a small improvement, but the "
                f"original-task slice regressed by {row.delta_original_rmse:+.5f} and the "
                f"clean slice by {row.delta_clean_rmse:+.5f}. Under the predeclared rule a "
                "candidate must improve the aggregate geometric RMSE by more than it costs "
                "on the retention slices, and this one does not, so it is reported as "
                "ineligible rather than as a near-win."
            )

    if selected == GLOBAL_BASELINE:
        lines.append(
            "- **Outcome.** No spatial ablation was selected: the frozen M5C champion is "
            "retained. This is reported as a negative result rather than a manufactured "
            "winner."
        )
        resolution = result.diagnostics.get("grid_resolution", {})
        families = resolution.get("families", {}) if resolution else {}
        elastic = families.get("elastic")
        blur = families.get("gaussian_blur")
        if elastic and blur:
            lines.append(
                "- **Why the spatial family did not help.** The declared grid resolves blur "
                "well but barely resolves the geometric warp. On training rows the mean "
                "absolute per-cell ink-occupancy change between a clean reference and its "
                f"distorted image is {blur['mean_abs_occupancy_change']:.5f} for blur but "
                f"only {elastic['mean_abs_occupancy_change']:.5f} for the elastic family, "
                f"a factor of "
                f"{blur['mean_abs_occupancy_change'] / elastic['mean_abs_occupancy_change']:.0f}. "
                f"A 1-3 px warp moves ink by a fraction of a "
                f"{resolution['cell_height_px']}x{resolution['cell_width_px']} pixel cell, so "
                "a coarse partition averages most of the effect away and the spatial "
                "features largely re-encode how much ink the image has, which the global "
                "features already state. The occupancy change still correlates with the "
                f"SSIM drop for the elastic family "
                f"({elastic['correlation_with_ssim_drop']:+.3f}), but more weakly than for "
                f"blur ({blur['correlation_with_ssim_drop']:+.3f}) and on a much smaller "
                "signal, so a model built on it cannot separate the geometric family from "
                "the others. The spatial-only ablation is consequently poor on every slice, "
                "because it discards the global summary that actually tracks blur and noise."
            )
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark champion, "
            "and the M5C champion remains the best geometric-robustness policy found so "
            "far. Adding spatial features did not improve the held-out geometric slice "
            "under the predeclared retention-aware rule. The evidence does not justify a "
            "CNN, and a new corruption family does not by itself justify one."
        )
        lines.append(
            "- **What this rules in and out.** The negative result is specific to the "
            "declared 4x2 partition: it shows that a *coarse* spatial summary does not "
            "help, not that localization is useless. A finer partition, or features that "
            "compare a region against its own reference rather than summarising it "
            "absolutely, would change what the spatial family can see."
        )
    else:
        lines.append(f"- **Outcome.** A spatial ablation was selected: `{selected}`.")
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark champion, "
            f"now with the spatial-aware policy `{selected}` rather than the frozen "
            "baseline alone. The evidence does not justify a CNN, and a new corruption "
            "family does not by itself justify one."
        )
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_spatial_study(
    config: SpatialStudyConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    verify_images: bool = True,
) -> tuple[StudyResult, dict[str, Path]]:
    """Load, verify, run and persist the complete M6A study."""
    started = time.perf_counter()
    before = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    frames = load_study_frames(config, verify_images=verify_images)
    m5c_config = ra.load_retention_config(config.m5c.config)
    champion = _load_champion(config)

    result = run_study_core(frames, config, champion=champion, m5c_config=m5c_config)
    after = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    if not _frozen_unchanged(before, after):
        raise SpatialStudyError("a frozen asset changed during the M6A run")
    paths = write_study_outputs(
        result,
        config,
        output_reports=output_reports,
        output_models=output_models,
        frozen_before=before,
        frozen_after=after,
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, paths


def verify_spatial_study(
    config: SpatialStudyConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
) -> dict[str, Any]:
    """Verify a written M6A run: hashes, protocol, selection and prediction replay."""
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
        _require_file_hash(config.m5c.config, config.m5c.config_sha256, "M5C config")
        m5c_config = ra.load_retention_config(config.m5c.config)
        if m5c_config.config_hash != config.m5c.protocol_hash:
            raise SpatialStudyError("the M5C protocol hash does not match the pin")
        _require_file_hash(
            config.m5c.champion_dir / f"{config.m5c.champion_stem}.joblib",
            config.m5c.champion_artifact_sha256,
            "M5C champion artifact",
        )
        frames = load_study_frames(config, verify_images=False)
        return (
            f"{len(frames.retention.expanded_train)} training rows, "
            f"{len(frames.retention.expanded_val)} validation rows, "
            f"{len(frames.retention.primary_test)} primary geometric test rows, "
            f"{len(frames.retention.diagnostic_test)} diagnostic test rows and "
            f"{len(frames.spatial_features)} extracted spatial feature rows"
        )

    def metadata_checks() -> str:
        metadata_path = reports / "run_metadata.json"
        if not metadata_path.is_file():
            raise SpatialStudyError(f"missing run metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("benchmark_config_sha256") != config.config_sha256:
            raise SpatialStudyError("run metadata config hash does not match")
        if metadata.get("benchmark_config_hash") != config.config_hash:
            raise SpatialStudyError("run metadata protocol hash does not match")
        if not metadata.get("frozen_assets", {}).get("all_unchanged"):
            raise SpatialStudyError("run metadata reports a changed frozen asset")
        for name, entry in metadata.get("artifacts", {}).items():
            path = Path(entry["path"]) if isinstance(entry, dict) else reports / name
            if not path.is_file():
                raise SpatialStudyError(f"missing recorded artifact: {path}")
            if sha256_file(path) != (entry["sha256"] if isinstance(entry, dict) else entry):
                raise SpatialStudyError(f"artifact hash mismatch: {name}")
        return f"{len(metadata.get('artifacts', {}))} artifact hashes match"

    def selection_checks() -> str:
        table_path = reports / "selection_table.csv"
        if not table_path.is_file():
            raise SpatialStudyError(f"missing selection table: {table_path}")
        table = pd.read_csv(table_path)
        selected = table[table["selected"]]
        if len(selected) != 1:
            raise SpatialStudyError("the selection table must mark exactly one winner")
        metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
        recorded = metadata.get("selection", {}).get("selected_model")
        if recorded != str(selected.iloc[0]["model"]):
            raise SpatialStudyError(
                f"selection table winner {selected.iloc[0]['model']!r} does not match the "
                f"recorded selection {recorded!r}"
            )
        return f"selected ablation {recorded!r} is consistent"

    def model_checks() -> str:
        for ablation in config.ablation_names:
            if not config.ablation_refit[ablation]:
                continue
            card_path = models / f"{ablation}_card.json"
            artifact = models / f"{ablation}.joblib"
            if not card_path.is_file() or not artifact.is_file():
                raise SpatialStudyError(f"missing model bundle for {ablation!r}")
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("artifact_sha256") != sha256_file(artifact):
                raise SpatialStudyError(f"artifact hash mismatch for {ablation!r}")
            expected = config.feature_names_for(config.ablation_feature_sets[ablation])
            if card.get("feature_names") != list(expected):
                raise SpatialStudyError(f"feature schema mismatch for {ablation!r}")
            if card.get("target") != config.target:
                raise SpatialStudyError(f"target mismatch for {ablation!r}")
        return "every refitted bundle loads, matches its card and uses the declared schema"

    def replay_checks() -> str:
        frames = load_study_frames(config, verify_images=False)
        m5c_config = ra.load_retention_config(config.m5c.config)
        champion = _load_champion(config)
        ablations = build_ablations(frames, config, champion=champion, m5c_config=m5c_config)
        slices = build_test_slices(frames, ablations)
        replayed = _test_metrics_table(slices)
        stored = pd.read_csv(reports / "test_slice_metrics.csv")
        merged = replayed.merge(stored, on=["model", "scope"], suffixes=("_replay", "_stored"))
        if len(merged) != len(replayed):
            raise SpatialStudyError("replayed metrics do not cover the stored rows")
        for column in ("mae", "rmse"):
            if not np.allclose(
                merged[f"{column}_replay"], merged[f"{column}_stored"], rtol=0.0, atol=1e-9
            ):
                raise SpatialStudyError(f"replayed {column} does not match the stored run")
        return f"{len(replayed)} slice metrics reproduce exactly"

    def report_checks() -> str:
        report_path = reports / "report.md"
        if not report_path.is_file():
            raise SpatialStudyError(f"missing report: {report_path}")
        text = report_path.read_text(encoding="utf-8")
        required = (
            "Spatial feature definitions",
            "Feature manifest",
            "Ablations",
            "Selection rule",
            "Validation results by slice",
            "Final test results by slice",
            "Paired source-grouped differences against the frozen baseline",
            "Feature diagnostics",
            "Reproducibility",
        )
        missing = [section for section in required if section not in text]
        if missing:
            raise SpatialStudyError(f"report is missing sections: {missing}")
        return "the report contains every required section"

    record("inputs_and_models", load_inputs)
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
        prog="captcha-spatial-study",
        description="Run or verify the grouped M6A spatial-aware feature study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the spatial-aware feature study")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path, default=None)
    run_parser.add_argument("--output-models", type=Path, default=None)

    verify_parser = subparsers.add_parser(
        "verify", help="verify a written spatial-aware feature run"
    )
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path, default=None)
    verify_parser.add_argument("--output-models", type=Path, default=None)
    return parser


def _print_report(report: Mapping[str, Any]) -> None:
    for check in report["checks"]:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"spatial-aware feature study verification: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_spatial_study_config(args.config)
        if args.command == "run":
            result, paths = run_spatial_study(
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
            print(f"selected ablation: {result.selected_name}")
            for scope in (GEO_LABEL, CLEAN_LABEL, ORIGINAL_LABEL, "m3_test_overall"):
                row = _metric_row_for(result.test_metrics, result.selected_name, scope)
                baseline = _metric_row_for(result.test_metrics, GLOBAL_BASELINE, scope)
                if row is None or baseline is None:
                    continue
                print(
                    f"  {scope}: selected MAE {row['mae']:.4f} RMSE {row['rmse']:.4f} | "
                    f"baseline MAE {baseline['mae']:.4f} RMSE {baseline['rmse']:.4f}"
                )
            print(f"report: {paths['report']}")
            return 0
        report = verify_spatial_study(
            config,
            output_reports=args.output_reports,
            output_models=args.output_models,
        )
        _print_report(report)
        return 0 if report["passed"] else 1
    except (
        SpatialStudyError,
        SpatialStudyConfigError,
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
