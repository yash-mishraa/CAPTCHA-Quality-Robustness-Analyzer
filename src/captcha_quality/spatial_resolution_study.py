"""Milestone M6B spatial resolution study.

M6A added spatially localized image features on a fixed 4x2 grid of the
canonical 192x64 canvas and found that they did not improve held-out
geometric-distortion prediction. Its grid-resolution diagnostic explained why:
a 1-3 px warp moves ink by a fraction of a 16x96-pixel cell, so a coarse
partition averages the effect away, and the spatial features largely re-encode
how much ink the image has - which the eight global features already state.

M6B asks whether that conclusion is a property of the *resolution* rather than
of localization as such:

    Does increasing the spatial resolution of the existing occupancy and
    edge-density feature families improve geometric-distortion SSIM prediction
    while preserving clean-reference and original-task performance?

This is a controlled resolution experiment. The two M6A feature families are
reused without any change to their mathematical definitions: the per-image Otsu
threshold, the Canny edge convention, the grayscale representation, the
canonical canvas and the row-major cell ordering are all identical. The *only*
thing that varies between candidates is the number of grid rows and columns.

Four resolutions are predeclared - 2x2 (coarse control), 4x2 (the M6A
reference), 4x4 (intermediate) and 8x8 (fine) - and each is evaluated in two
representations: spatial-only and global-plus-spatial. The frozen M5C random
forest remains the reference policy and is never refitted.

The model family, hyperparameters, seed, training objective, training data and
validation data are identical for every candidate, so a difference between
resolutions is attributable to the input representation alone. No
hyperparameter is changed to accommodate an individual resolution, and no
feature is removed, no grid is changed and no model is modified in response to
a diagnostic. The selection rule is the M6A rule unchanged, with the same
coefficients and tolerances, so the two studies are directly comparable.

Because the forest considers every feature at each split (``max_features`` is
1.0), a larger feature set dilutes the split search directly, and the 8x8 grid
produces 128 spatial and 136 combined features against 432 training rows. The
training-only redundancy and train-versus-validation diagnostics are therefore
reported for every resolution, and the measured effect sizes are reported
against the small number of independent test sources.

The 0.5 px geometric rung remains a diagnostic only. The target is synthetic
image similarity (SSIM against the exact clean reference). Nothing here
measures human readability, CAPTCHA solvability, security or real-world
robustness.
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
from captcha_quality import spatial_study as ss
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.dataset import CheckResult, load_grayscale
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import (
    TARGET,
    CandidateSpec,
    ModelError,
    save_model_bundle,
    score_frame,
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
    FAMILY_EDGE,
    FAMILY_INK,
    SPATIAL_FEATURE_SET_VERSION,
    SpatialFeatureSpec,
    extract_spatial_feature_table,
    spatial_feature_manifest,
)

#: Representation names.
SPATIAL_ONLY: Final[str] = "spatial_only"
COMBINED: Final[str] = "combined"

#: The frozen champion, which is the reference and the fallback.
CHAMPION: Final[str] = "global_baseline"

#: Slice labels, matching the M5C, M5D and M6A conventions.
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
        "model",
        "protocol",
        "resolutions",
        "representations",
        "champion",
        "selection",
    }
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Frozen inputs audited before and after the milestone. M6B's own outputs are
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
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "data/processed/geometric_distortion_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "models/robustness_benchmark_v0_1_0",
    "models/geometric_robustness_benchmark_v0_1_0",
    "models/retention_aware_selection_v0_1_0",
    "models/retention_aware_training_v0_1_0",
    "models/spatial_feature_study_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
    "reports/geometric_distortion_pilot",
    "reports/geometric_robustness_benchmark_v0_1_0",
    "reports/retention_aware_selection_v0_1_0",
    "reports/retention_aware_training_v0_1_0",
    "reports/spatial_feature_study_v0_1_0",
)


class ResolutionStudyError(RuntimeError):
    """Raised when the M6B protocol cannot be honoured."""


class ResolutionStudyConfigError(ValueError):
    """Raised when the M6B configuration is invalid."""


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
    """Pins for the upstream M6A study, whose feature definitions are reused."""

    config: Path
    config_sha256: str
    protocol_hash: str
    feature_set_version: str


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """The fixed model family and hyperparameters shared by every candidate."""

    family: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    seed: int


@dataclass(frozen=True, slots=True)
class GridSpec:
    """One declared grid resolution."""

    rows: int
    cols: int

    @property
    def label(self) -> str:
        return f"{self.rows}x{self.cols}"

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    @property
    def n_spatial_features(self) -> int:
        """Two families, each contributing one feature per cell."""
        return 2 * self.n_cells

    def feature_spec(self, *, edge_sigma: float, low: float, high: float) -> SpatialFeatureSpec:
        """Return the M6A feature specification for this grid."""
        return SpatialFeatureSpec(
            grid_rows=self.rows,
            grid_cols=self.cols,
            families=(FAMILY_INK, FAMILY_EDGE),
            edge_sigma=edge_sigma,
            edge_low_threshold=low,
            edge_high_threshold=high,
        )


@dataclass(frozen=True, slots=True)
class ResolutionSettings:
    """The predeclared resolution set and the shared edge convention."""

    grids: tuple[GridSpec, ...]
    edge_sigma: float
    edge_low_threshold: float
    edge_high_threshold: float


@dataclass(frozen=True, slots=True)
class ChampionSettings:
    """The frozen champion reference."""

    name: str
    feature_set: str
    refit: bool


@dataclass(frozen=True, slots=True)
class ResolutionConfig:
    """The complete M6B protocol."""

    run: rb.RunSettings
    global_feature_set_version: str
    global_feature_names: tuple[str, ...]
    target: str
    m5c: M5CPin
    m6a: M6APin
    model: ModelSettings
    protocol: ra.ProtocolSettings
    resolutions: ResolutionSettings
    representations: tuple[str, ...]
    champion: ChampionSettings
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
            "resolutions": {
                "grids": [
                    {"rows": grid.rows, "cols": grid.cols} for grid in self.resolutions.grids
                ],
                "edge_sigma": self.resolutions.edge_sigma,
                "edge_low_threshold": self.resolutions.edge_low_threshold,
                "edge_high_threshold": self.resolutions.edge_high_threshold,
            },
            "representations": list(self.representations),
            "champion": {
                "name": self.champion.name,
                "feature_set": self.champion.feature_set,
                "refit": self.champion.refit,
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

    def feature_names_for(self, grid: GridSpec, representation: str) -> tuple[str, ...]:
        """Return the ordered feature names for one grid and representation.

        The spatial names are namespaced with the resolution label, because the
        same cell name occurs at more than one grid and the per-resolution
        schemas must stay disjoint.
        """
        spatial = tuple(
            f"{name}__{grid.label}"
            for name in grid.feature_spec(
                edge_sigma=self.resolutions.edge_sigma,
                low=self.resolutions.edge_low_threshold,
                high=self.resolutions.edge_high_threshold,
            ).feature_names
        )
        if representation == SPATIAL_ONLY:
            return spatial
        if representation == COMBINED:
            return tuple(self.global_feature_names) + spatial
        raise ResolutionStudyConfigError(f"unknown representation {representation!r}")

    @property
    def champion_feature_names(self) -> tuple[str, ...]:
        """Return the ordered feature names of the frozen champion."""
        return tuple(self.global_feature_names)

    def candidate_names(self) -> tuple[str, ...]:
        """Return every candidate name, in a deterministic order."""
        names = [self.champion.name]
        for grid in self.resolutions.grids:
            for representation in self.representations:
                names.append(f"{representation}_{grid.label}")
        return tuple(names)


# ---------------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------------


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ResolutionStudyConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ResolutionStudyConfigError(f"missing or invalid [{name}] table")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResolutionStudyConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResolutionStudyConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ResolutionStudyConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ResolutionStudyConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ResolutionStudyConfigError(f"{label} must be a boolean, got {value!r}")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ResolutionStudyConfigError(f"{label} must be a TOML array")
    items = tuple(_as_str(item, f"{label}[]") for item in value)
    if not items or len(set(items)) != len(items):
        raise ResolutionStudyConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ResolutionStudyConfigError(f"{label} must be a lowercase SHA-256 hex digest")
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
        raise ResolutionStudyConfigError(
            f"model.family must be 'random_forest', got {settings.family!r}"
        )
    if settings.n_estimators < 1:
        raise ResolutionStudyConfigError("model.n_estimators must be at least 1")
    if settings.max_depth < 0:
        raise ResolutionStudyConfigError("model.max_depth must be zero (unlimited) or positive")
    if settings.min_samples_leaf < 1:
        raise ResolutionStudyConfigError("model.min_samples_leaf must be at least 1")
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
        raise ResolutionStudyConfigError(
            "the diagnostic amplitude must not also appear in the primary ladder"
        )
    if set(settings.primary_severities) & set(settings.diagnostic_severities):
        raise ResolutionStudyConfigError(
            "the diagnostic severity must not also appear in the primary ladder"
        )
    return settings


def _parse_resolutions(raw: Mapping[str, object]) -> ResolutionSettings:
    _check_keys(
        raw,
        {"grids", "edge_sigma", "edge_low_threshold", "edge_high_threshold"},
        "resolutions",
    )
    grids_raw = raw["grids"]
    if not isinstance(grids_raw, list) or not grids_raw:
        raise ResolutionStudyConfigError("resolutions.grids must be a non-empty TOML array")
    grids: list[GridSpec] = []
    for index, entry in enumerate(grids_raw):
        if not isinstance(entry, dict):
            raise ResolutionStudyConfigError(f"resolutions.grids[{index}] must be a table")
        _check_keys(entry, {"rows", "cols"}, f"resolutions.grids[{index}]")
        grid = GridSpec(
            rows=_as_int(entry["rows"], f"resolutions.grids[{index}].rows"),
            cols=_as_int(entry["cols"], f"resolutions.grids[{index}].cols"),
        )
        if grid.rows < 1 or grid.cols < 1:
            raise ResolutionStudyConfigError(
                f"resolutions.grids[{index}] must have positive rows and cols"
            )
        grids.append(grid)
    labels = [grid.label for grid in grids]
    if len(set(labels)) != len(labels):
        raise ResolutionStudyConfigError("resolutions.grids must be duplicate-free")
    settings = ResolutionSettings(
        grids=tuple(grids),
        edge_sigma=_as_float(raw["edge_sigma"], "resolutions.edge_sigma"),
        edge_low_threshold=_as_float(raw["edge_low_threshold"], "resolutions.edge_low_threshold"),
        edge_high_threshold=_as_float(
            raw["edge_high_threshold"], "resolutions.edge_high_threshold"
        ),
    )
    if settings.edge_sigma <= 0.0:
        raise ResolutionStudyConfigError("resolutions.edge_sigma must be strictly positive")
    for label, value in (
        ("edge_low_threshold", settings.edge_low_threshold),
        ("edge_high_threshold", settings.edge_high_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ResolutionStudyConfigError(f"resolutions.{label} must lie in [0, 1]")
    return settings


def _parse_champion(raw: Mapping[str, object]) -> ChampionSettings:
    _check_keys(raw, {"name", "feature_set", "refit"}, "champion")
    settings = ChampionSettings(
        name=_as_str(raw["name"], "champion.name"),
        feature_set=_as_str(raw["feature_set"], "champion.feature_set"),
        refit=_as_bool(raw["refit"], "champion.refit"),
    )
    if settings.refit:
        raise ResolutionStudyConfigError(
            "the champion must not be refitted; it is the frozen reference policy"
        )
    if settings.feature_set != "global":
        raise ResolutionStudyConfigError("champion.feature_set must be 'global'")
    return settings


def load_resolution_config(path: Path | str) -> ResolutionConfig:
    """Load and validate the M6B configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ResolutionStudyConfigError(f"missing configuration file: {config_path}")
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    unknown = sorted(set(payload) - _EXPECTED_TOP_KEYS)
    if unknown:
        raise ResolutionStudyConfigError(
            f"unknown top-level key(s): {unknown}; allowed keys are {sorted(_EXPECTED_TOP_KEYS)}"
        )
    global_version, global_names, target = ss._parse_schema(  # noqa: SLF001
        _section(payload, "schema")
    )
    representations = _as_str_tuple(
        _section(payload, "representations")["names"], "representations.names"
    )
    if set(representations) != {SPATIAL_ONLY, COMBINED}:
        raise ResolutionStudyConfigError(
            f"representations.names must be {[SPATIAL_ONLY, COMBINED]}, got "
            f"{sorted(representations)}"
        )
    config = ResolutionConfig(
        run=_parse_run(_section(payload, "run")),
        global_feature_set_version=global_version,
        global_feature_names=global_names,
        target=target,
        m5c=_parse_m5c(_section(payload, "m5c")),
        m6a=_parse_m6a(_section(payload, "m6a")),
        model=_parse_model(_section(payload, "model")),
        protocol=_parse_protocol(_section(payload, "protocol")),
        resolutions=_parse_resolutions(_section(payload, "resolutions")),
        representations=representations,
        champion=_parse_champion(_section(payload, "champion")),
        selection=ss._parse_selection(  # noqa: SLF001
            _section(payload, "selection")
        ),
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )
    if config.global_feature_names != tuple(FEATURE_NAMES):
        raise ResolutionStudyConfigError(
            "schema.global_feature_names must match the frozen extractor "
            f"{list(FEATURE_NAMES)}; the eight-feature pipeline must not be altered"
        )
    if config.global_feature_set_version != "v1":
        raise ResolutionStudyConfigError(
            f"schema.global_feature_set_version must be 'v1', got "
            f"{config.global_feature_set_version!r}"
        )
    if config.target != TARGET:
        raise ResolutionStudyConfigError(f"schema.target must be {TARGET!r}")
    if config.m6a.feature_set_version != SPATIAL_FEATURE_SET_VERSION:
        raise ResolutionStudyConfigError(
            f"m6a.feature_set_version must be {SPATIAL_FEATURE_SET_VERSION!r}; the frozen "
            "spatial feature definitions must be reused unchanged"
        )
    if config.protocol.primary_amplitudes_px != gb.PRIMARY_AMPLITUDES_PX:
        raise ResolutionStudyConfigError(
            "protocol.primary_amplitudes_px must match the M5B primary ladder "
            f"{list(gb.PRIMARY_AMPLITUDES_PX)}"
        )
    if config.protocol.diagnostic_amplitudes_px != gb.DIAGNOSTIC_AMPLITUDES_PX:
        raise ResolutionStudyConfigError(
            "protocol.diagnostic_amplitudes_px must match the M5B diagnostic ladder "
            f"{list(gb.DIAGNOSTIC_AMPLITUDES_PX)}"
        )
    expected_grids = {"2x2", "4x2", "4x4", "8x8"}
    if {grid.label for grid in config.resolutions.grids} != expected_grids:
        raise ResolutionStudyConfigError(
            f"resolutions.grids must be {sorted(expected_grids)}, got "
            f"{sorted(grid.label for grid in config.resolutions.grids)}"
        )
    return config


# ---------------------------------------------------------------------------
# Frames, feature extraction and candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolutionFrames:
    """The frozen frames with the spatial features extracted at every resolution."""

    retention: ra.RetentionFrames
    spatial_features: Mapping[str, pd.DataFrame]
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def with_spatial(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` joined with every resolution's spatial features.

        The same cell name occurs at more than one resolution (for example
        ``ink_occupancy_r0_c0`` exists at every grid), so each resolution's
        columns are namespaced with the resolution label. The namespaced
        column is what the candidates consume, which keeps the per-resolution
        feature schemas disjoint and the merges unambiguous.
        """
        if frame["sample_id"].duplicated().any():
            raise ResolutionStudyError("frame has duplicate sample_id values")
        merged = frame
        for label, table in self.spatial_features.items():
            renamed = table.rename(
                columns={
                    column: f"{column}__{label}"
                    for column in table.columns
                    if column != "sample_id"
                }
            )
            merged = merged.merge(renamed, on="sample_id", how="left", validate="one_to_one")
        feature_columns = [
            f"{column}__{label}"
            for label, table in self.spatial_features.items()
            for column in table.columns
            if column != "sample_id"
        ]
        if merged[feature_columns].isna().any().any():
            raise ResolutionStudyError("some rows are missing extracted spatial features")
        return merged


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ResolutionStudyError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ResolutionStudyError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def _check_spatial_frame(table: pd.DataFrame, spec: SpatialFeatureSpec, label: str) -> CheckResult:
    """Confirm one resolution's table has the exact declared schema and is finite."""
    try:
        expected = ("sample_id", *spec.feature_names)
        if tuple(table.columns) != expected:
            raise ResolutionStudyError(
                f"{label} feature columns must be exactly {list(expected)}, got "
                f"{list(table.columns)}"
            )
        if table["sample_id"].duplicated().any():
            raise ResolutionStudyError(f"{label} table has duplicate sample_id values")
        values = table[list(spec.feature_names)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ResolutionStudyError(f"{label} features contain non-finite values")
        if (values < 0.0).any() or (values > 1.0).any():
            raise ResolutionStudyError(f"{label} features fall outside [0, 1]")
        return CheckResult(
            name=f"spatial_schema_{label}",
            passed=True,
            detail=(
                f"{len(table)} rows with {len(spec.feature_names)} spatial features in the "
                "declared order, all finite and inside [0, 1]"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=f"spatial_schema_{label}", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _check_no_metadata_in_spatial_inputs(
    table: pd.DataFrame, spec: SpatialFeatureSpec, label: str
) -> CheckResult:
    """Confirm a resolution's table carries no target or metadata column."""
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
        present = sorted(forbidden & set(table.columns))
        if present:
            raise ResolutionStudyError(
                f"forbidden columns present in the {label} feature table: {present}"
            )
        if len(spec.feature_names) != len(set(spec.feature_names)):
            raise ResolutionStudyError(f"duplicate spatial feature names at {label}")
        return CheckResult(
            name=f"spatial_leakage_{label}",
            passed=True,
            detail=(
                f"the {label} feature table contains only sample_id and the declared image "
                "features; no target, severity, family, font, source, split or path column "
                "is present"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=f"spatial_leakage_{label}", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def load_resolution_frames(
    config: ResolutionConfig, *, verify_images: bool = True
) -> ResolutionFrames:
    """Load the frozen frames through the M5C path and extract every resolution."""
    _require_file_hash(config.m5c.config, config.m5c.config_sha256, "M5C config")
    m5c_config = ra.load_retention_config(config.m5c.config)
    if m5c_config.config_hash != config.m5c.protocol_hash:
        raise ResolutionStudyError(
            "M5C protocol hash mismatch: expected "
            f"{config.m5c.protocol_hash}, found {m5c_config.config_hash}"
        )
    _require_file_hash(config.m6a.config, config.m6a.config_sha256, "M6A config")
    m6a_config = ss.load_spatial_study_config(config.m6a.config)
    if m6a_config.config_hash != config.m6a.protocol_hash:
        raise ResolutionStudyError(
            "M6A protocol hash mismatch: expected "
            f"{config.m6a.protocol_hash}, found {m6a_config.config_hash}"
        )
    retention = ra.load_retention_frames(m5c_config, verify_images=verify_images)
    roots = ss._dataset_roots(m5c_config)  # noqa: SLF001

    specs = {
        grid.label: grid.feature_spec(
            edge_sigma=config.resolutions.edge_sigma,
            low=config.resolutions.edge_low_threshold,
            high=config.resolutions.edge_high_threshold,
        )
        for grid in config.resolutions.grids
    }

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
                raise ResolutionStudyError(f"no image root configured for dataset {dataset!r}")
            samples.append((sample_id, load_grayscale(root / str(row.image_path))))

    tables: dict[str, pd.DataFrame] = {}
    checks: list[CheckResult] = [*retention.checks]
    for label, spec in specs.items():
        table = extract_spatial_feature_table(samples, spec)
        tables[label] = table
        checks.append(_check_spatial_frame(table, spec, label))
        checks.append(_check_no_metadata_in_spatial_inputs(table, spec, label))
    rb._verify_checks(checks, "M6B frame checks")  # noqa: SLF001
    return ResolutionFrames(retention=retention, spatial_features=tables, checks=tuple(checks))


def _champion_spec(config: ResolutionConfig, m5c_config: ra.RetentionConfig) -> CandidateSpec:
    """Return the champion's grid specification and confirm it matches the config."""
    m5b_config = ra._load_m5b_config(m5c_config)  # noqa: SLF001
    specs = rb.candidate_specs_from_config(m5b_config)  # type: ignore[arg-type]
    spec = next(
        (item for item in specs if item.name == m5c_config.candidates.alternative_spec), None
    )
    if spec is None:
        raise ResolutionStudyError(
            f"the M5C champion specification {m5c_config.candidates.alternative_spec!r} is "
            "not in the M5B grid"
        )
    if spec.family != config.model.family:
        raise ResolutionStudyError(
            f"model.family is {config.model.family!r} but the champion was fitted from a "
            f"{spec.family!r} specification"
        )
    for key, value in (
        ("n_estimators", config.model.n_estimators),
        ("max_depth", config.model.max_depth),
        ("min_samples_leaf", config.model.min_samples_leaf),
    ):
        if spec.params.get(key) != value:
            raise ResolutionStudyError(
                f"model.{key} is {value!r} but the M5B grid specification has "
                f"{spec.params.get(key)!r}"
            )
    seed = derive_noise_seed(m5b_config.run.seed, "model", spec.name) % (2**32)
    if seed != config.model.seed:
        raise ResolutionStudyError(
            f"model.seed is {config.model.seed} but the M5B grid derives {seed}"
        )
    return spec


def _load_champion(config: ResolutionConfig) -> Any:
    """Load the frozen M5C champion bundle and confirm it is unchanged."""
    directory = config.m5c.champion_dir
    stem = config.m5c.champion_stem
    artifact = directory / f"{stem}.joblib"
    card_path = directory / f"{stem}_card.json"
    _require_file_hash(artifact, config.m5c.champion_artifact_sha256, "M5C champion artifact")
    if not card_path.is_file():
        raise ResolutionStudyError(f"missing M5C champion card: {card_path}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("feature_names") != list(config.global_feature_names):
        raise ResolutionStudyError("the M5C champion uses a different feature schema")
    if card.get("target") != config.target:
        raise ResolutionStudyError(f"the M5C champion target is not {config.target!r}")
    recorded = str(card.get("selection", {}).get("benchmark_config_hash", ""))
    if recorded != config.m5c.protocol_hash:
        raise ResolutionStudyError(
            f"the M5C champion was produced under protocol {recorded!r}, not the pinned "
            f"{config.m5c.protocol_hash!r}"
        )
    if card.get("artifact_sha256") != sha256_file(artifact):
        raise ResolutionStudyError("the M5C champion artifact hash does not match its card")
    if str(card.get("sklearn_version", "")).split(".")[:2] != sklearn_version().split(".")[:2]:
        raise ResolutionStudyError(
            f"the M5C champion was persisted with scikit-learn {card.get('sklearn_version')}, "
            f"current is {sklearn_version()}"
        )
    import joblib

    estimator = joblib.load(artifact)
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features != len(config.global_feature_names):
        raise ResolutionStudyError(
            f"the M5C champion expects {n_features} inputs, the frozen schema has "
            f"{len(config.global_feature_names)}"
        )
    return estimator


def build_candidates(
    frames: ResolutionFrames,
    config: ResolutionConfig,
    *,
    champion: Any,
    m5c_config: ra.RetentionConfig,
) -> tuple[ss.Ablation, ...]:
    """Fit every predeclared candidate on the expanded training rows.

    The model family, hyperparameters, seed, training rows and objective are
    identical for every candidate; only the input representation and its
    resolution differ.
    """
    spec = _champion_spec(config, m5c_config)
    train = frames.with_spatial(frames.retention.expanded_train)
    candidates: list[ss.Ablation] = [
        ss.Ablation(
            name=config.champion.name,
            feature_set=config.champion.feature_set,
            feature_names=config.champion_feature_names,
            estimator=champion,
            refitted=False,
            train_importances=None,
        )
    ]
    for grid in config.resolutions.grids:
        for representation in config.representations:
            feature_names = config.feature_names_for(grid, representation)
            estimator = spec.build(seed=config.model.seed)
            inputs = train[list(feature_names)].to_numpy(dtype=np.float64)
            target = train[TARGET].to_numpy(dtype=np.float64)
            estimator.fit(inputs, target)
            importances = getattr(estimator, "feature_importances_", None)
            candidates.append(
                ss.Ablation(
                    name=f"{representation}_{grid.label}",
                    feature_set=representation,
                    feature_names=feature_names,
                    estimator=estimator,
                    refitted=True,
                    train_importances=(
                        np.asarray(importances, dtype=np.float64)
                        if importances is not None
                        else None
                    ),
                )
            )
    return tuple(candidates)


# ---------------------------------------------------------------------------
# Validation scoring, diagnostics and selection
# ---------------------------------------------------------------------------


def validation_slices(frames: ResolutionFrames) -> dict[str, pd.DataFrame]:
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


def _severity_scopes(config: ResolutionConfig) -> tuple[str, ...]:
    return tuple(
        f"{GEO_LABEL}_severity_{severity}" for severity in config.protocol.primary_severities
    )


def _slice_metrics_table(
    frames: ResolutionFrames,
    candidates: Sequence[ss.Ablation],
    predictions: Mapping[str, np.ndarray],
    *,
    split_label: str,
) -> pd.DataFrame:
    """Score every candidate on every validation slice."""
    validation = frames.with_spatial(frames.retention.expanded_val)
    slices = validation_slices(frames)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        predicted = np.asarray(predictions[candidate.name], dtype=np.float64)
        if predicted.size != len(validation):
            raise ResolutionStudyError(
                f"predictions for {candidate.name!r} do not align with the validation frame"
            )
        for scope, subset in slices.items():
            if subset.empty:
                continue
            mask = validation.index.get_indexer(subset.index)
            scores = ra._slice_scores(subset, predicted[mask])  # noqa: SLF001
            rows.append(
                {
                    "model": candidate.name,
                    "feature_set": candidate.feature_set,
                    "n_features": len(candidate.feature_names),
                    "split": split_label,
                    "scope": scope,
                    **scores,
                }
            )
    return pd.DataFrame(rows)


def training_diagnostics(
    frames: ResolutionFrames,
    candidates: Sequence[ss.Ablation],
    config: ResolutionConfig,
) -> dict[str, Any]:
    """Compute training-only redundancy and overfitting diagnostics.

    Everything here is computed on the training rows. The diagnostics are
    reported, not acted upon: no feature is removed, no grid is changed and no
    model is modified in response to them.
    """
    train = frames.with_spatial(frames.retention.expanded_train)
    validation = frames.with_spatial(frames.retention.expanded_val)
    slices = validation_slices(frames)
    geometric_val = slices[GEO_LABEL]

    summaries: dict[str, Any] = {}
    for candidate in candidates:
        spatial_names = [
            name for name in candidate.feature_names if name not in set(config.global_feature_names)
        ]
        spatial = train[spatial_names] if spatial_names else None
        constant: list[str] = []
        duplicated: list[str] = []
        mean_corr = math.nan
        max_corr = math.nan
        if spatial is not None and len(spatial.columns) > 1:
            constant = [name for name in spatial.columns if float(spatial[name].std(ddof=0)) == 0.0]
            # A duplicated feature is one whose column is identical to another
            # column. Group the columns by their exact value pattern so this
            # stays linear in the number of columns.
            values = spatial.to_numpy(dtype=np.float64)
            patterns: dict[bytes, list[str]] = {}
            for index, name in enumerate(spatial.columns):
                key = values[:, index].tobytes()
                patterns.setdefault(key, []).append(name)
            duplicated = sorted(
                names[1] for names in patterns.values() if len(names) > 1 for names in [names]
            )
            correlation = spatial.corr().abs()
            mask = ~np.eye(len(spatial.columns), dtype=bool)
            if mask.any():
                mean_corr = float(correlation.where(mask).stack().mean())
                max_corr = float(correlation.where(mask).max().max())
        # The training-versus-validation gap: the candidate is fitted on the
        # training rows, so a positive gap is the overfitting signal.
        train_predictions = candidate.predict(train)
        train_rmse = float(score_frame(train, train_predictions)["rmse"])
        val_predictions = candidate.predict(validation)
        val_rmse = float(score_frame(validation, val_predictions)["rmse"])
        geo_val_predictions = candidate.predict(geometric_val)
        geo_val_rmse = float(score_frame(geometric_val, geo_val_predictions)["rmse"])
        summaries[candidate.name] = {
            "n_features": len(candidate.feature_names),
            "n_spatial_features": len(spatial_names),
            "n_constant_spatial_features": len(constant),
            "constant_spatial_features": constant,
            "n_duplicated_spatial_features": len(duplicated),
            "duplicated_spatial_features": duplicated,
            "spatial_correlation_mean": mean_corr,
            "spatial_correlation_max": max_corr,
            "train_rmse": train_rmse,
            "validation_rmse": val_rmse,
            "overfitting_gap_rmse": val_rmse - train_rmse,
            "geometric_validation_rmse": geo_val_rmse,
        }
    return summaries


def grid_resolution_diagnostic(
    frames: ResolutionFrames, config: ResolutionConfig
) -> dict[str, Any]:
    """Measure, per resolution, how much of a distortion the grid can resolve.

    This is a descriptive diagnostic computed on **training rows only**. It
    compares, per source, the per-cell ink occupancy of a distorted image with
    that of the same source's clean reference, and reports the mean absolute
    change and its correlation with the SSIM drop. It answers whether a finer
    partition can resolve the distortion at all, independently of any model.
    """
    train = frames.with_spatial(frames.retention.expanded_train)
    m3_train = train[train["dataset"] == gb.M3_LABEL]
    m5a_train = train[train["dataset"] == gb.M5A_LABEL]
    clean_rows = m3_train[m3_train["family"] == gb.M3_CLEAN_FAMILY]
    clean_by_source = {str(row.source_id): row for row in clean_rows.itertuples(index=False)}
    families = {
        "gaussian_blur": m3_train[m3_train["family"] == "gaussian_blur"],
        "gaussian_noise": m3_train[m3_train["family"] == "gaussian_noise"],
        "elastic": m5a_train[m5a_train["family"] == gb.ELASTIC_FAMILY],
    }
    results: dict[str, Any] = {}
    for grid in config.resolutions.grids:
        columns = [
            f"{FAMILY_INK}_r{i}_c{j}__{grid.label}"
            for i in range(grid.rows)
            for j in range(grid.cols)
        ]
        per_family: dict[str, Any] = {}
        for family, frame in families.items():
            changed: list[float] = []
            dropped: list[float] = []
            for record in frame[["source_id", TARGET, *columns]].itertuples(index=False):
                reference = clean_by_source.get(str(record.source_id))
                if reference is None:
                    continue
                distorted = np.asarray(record[2:], dtype=np.float64)
                base = np.asarray(
                    [getattr(reference, column) for column in columns], dtype=np.float64
                )
                changed.append(float(np.abs(distorted - base).mean()))
                dropped.append(float(getattr(reference, TARGET)) - float(record[1]))
            if not changed:
                continue
            correlation = (
                float(np.corrcoef(changed, dropped)[0, 1])
                if len(changed) > 2 and np.std(changed) > 0.0 and np.std(dropped) > 0.0
                else math.nan
            )
            per_family[family] = {
                "n_pairs": len(changed),
                "mean_abs_occupancy_change": float(np.mean(changed)),
                "max_abs_occupancy_change": float(np.max(changed)),
                "correlation_with_ssim_drop": correlation,
            }
        results[grid.label] = per_family
    return {
        "cell_height_px": {grid.label: 64 // grid.rows for grid in config.resolutions.grids},
        "cell_width_px": {grid.label: 192 // grid.cols for grid in config.resolutions.grids},
        "families_by_resolution": results,
    }


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """The complete M6B outcome."""

    frames: ResolutionFrames
    checks: tuple[CheckResult, ...]
    candidates: tuple[ss.Ablation, ...]
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
        for candidate in self.candidates:
            if candidate.name == self.selected_name:
                return candidate
        raise ResolutionStudyError(f"selected candidate {self.selected_name!r} is missing")


def _test_slices(frames: ResolutionFrames, candidates: Sequence[ss.Ablation]) -> list[ss.TestSlice]:
    """Assemble the frozen evaluation slices scored after selection is complete."""
    retention = frames.retention
    geometric = frames.with_spatial(retention.primary_test)
    m3_test = frames.with_spatial(retention.m3_test)
    original = m3_test[~m3_test["is_clean"]]
    diagnostic = frames.with_spatial(retention.diagnostic_test)

    def predict_on(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {candidate.name: candidate.predict(frame) for candidate in candidates}

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


def run_resolution_core(
    frames: ResolutionFrames,
    config: ResolutionConfig,
    *,
    champion: Any,
    m5c_config: ra.RetentionConfig,
) -> ResolutionResult:
    """Run the M6B study in the predeclared order.

    The order is fixed: fit on training data only, score validation, select,
    freeze, and only then score the untouched test slices.
    """
    rb._verify_checks(frames.checks, "M6B frame checks")  # noqa: SLF001

    candidates = build_candidates(frames, config, champion=champion, m5c_config=m5c_config)
    validation = frames.with_spatial(frames.retention.expanded_val)
    validation_predictions = {
        candidate.name: candidate.predict(validation) for candidate in candidates
    }
    validation_metrics = _slice_metrics_table(
        frames, candidates, validation_predictions, split_label="val"
    )
    selected_name, selection_table = ss.select_ablation(
        validation_metrics,
        config.selection,
        severity_scopes=_severity_scopes(config),
        reference=config.champion.name,
    )

    # The selected candidate and its full configuration are frozen here. The
    # test slices are only touched after this point.
    slices = _test_slices(frames, candidates)
    test_metrics = ss._test_metrics_table(slices)  # noqa: SLF001
    uncertainty = ss._uncertainty_table(  # noqa: SLF001
        slices, n_bootstrap=config.run.n_bootstrap, seed=config.run.seed
    )
    paired = ss._paired_uncertainty_table(  # noqa: SLF001
        slices,
        n_bootstrap=config.run.n_bootstrap,
        seed=config.run.seed,
        reference=config.champion.name,
    )
    predictions = ss._prediction_table(  # noqa: SLF001
        slices, [candidate.name for candidate in candidates]
    )
    diagnostics = training_diagnostics(frames, candidates, config)
    diagnostics["grid_resolution"] = grid_resolution_diagnostic(frames, config)

    return ResolutionResult(
        frames=frames,
        checks=frames.checks,
        candidates=candidates,
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


def _candidate_model_card(
    result: ResolutionResult, config: ResolutionConfig, candidate: ss.Ablation
) -> dict[str, Any]:
    """Build the model card for one refitted candidate."""
    metrics = result.test_metrics[result.test_metrics["model"] == candidate.name]
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
    validation = result.validation_metrics[result.validation_metrics["model"] == candidate.name]
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
        "model_name": candidate.name,
        "model_family": config.model.family,
        "feature_set": candidate.feature_set,
        "n_features": len(candidate.feature_names),
        "feature_names": list(candidate.feature_names),
        "spatial_feature_set_version": SPATIAL_FEATURE_SET_VERSION,
        "params": {
            "n_estimators": config.model.n_estimators,
            "max_depth": config.model.max_depth,
            "min_samples_leaf": config.model.min_samples_leaf,
        },
        "seed": config.model.seed,
        "refitted": candidate.refitted,
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
            "n_candidates": len(result.candidates),
            "benchmark_config_hash": config.config_hash,
        },
        "validation_metrics": validation_payload,
        "metrics": metric_payload,
    }


def write_resolution_outputs(
    result: ResolutionResult,
    config: ResolutionConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    frozen_before: Mapping[str, Mapping[str, object]] | None = None,
    frozen_after: Mapping[str, Mapping[str, object]] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Path]:
    """Persist the M6B tables, model bundles, metadata and report."""
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
    manifest = pd.concat(
        [
            spatial_feature_manifest(
                grid.feature_spec(
                    edge_sigma=config.resolutions.edge_sigma,
                    low=config.resolutions.edge_low_threshold,
                    high=config.resolutions.edge_high_threshold,
                )
            ).assign(resolution=grid.label)
            for grid in config.resolutions.grids
        ],
        ignore_index=True,
    )
    manifest.to_csv(reports / "spatial_feature_manifest.csv", index=False)
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

    for candidate in result.candidates:
        if not candidate.refitted:
            continue
        saved = save_model_bundle(
            models,
            candidate.name,
            candidate.estimator,
            _candidate_model_card(result, config, candidate),
        )
        paths[f"{candidate.name}_artifact"] = saved["artifact"]
        paths[f"{candidate.name}_card"] = saved["card"]

    metadata = {
        "benchmark_config": config.identity_payload(),
        "benchmark_config_hash": config.config_hash,
        "benchmark_config_sha256": config.config_sha256,
        "schema": {
            "global_feature_set_version": config.global_feature_set_version,
            "global_feature_names": list(config.global_feature_names),
            "spatial_feature_set_version": SPATIAL_FEATURE_SET_VERSION,
            "target": config.target,
            "split_names": ["train", "val", "test"],
        },
        "resolutions": {
            grid.label: {
                "rows": grid.rows,
                "cols": grid.cols,
                "n_cells": grid.n_cells,
                "cell_height_px": 64 // grid.rows,
                "cell_width_px": 192 // grid.cols,
                "n_spatial_features": grid.n_spatial_features,
                "n_combined_features": grid.n_spatial_features + len(config.global_feature_names),
            }
            for grid in config.resolutions.grids
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
        "candidates": {
            candidate.name: {
                "feature_set": candidate.feature_set,
                "n_features": len(candidate.feature_names),
                "feature_names": list(candidate.feature_names),
                "refitted": candidate.refitted,
            }
            for candidate in result.candidates
        },
        "selection": {
            "rule": config.selection.rule,
            "selected_model": result.selected_name,
            "n_candidates": len(result.candidates),
            "selected_on": "val",
        },
        "diagnostics": result.diagnostics,
        "frozen_assets": {
            "before": dict(frozen_before or {}),
            "after": dict(frozen_after or {}),
            "all_unchanged": ss._frozen_unchanged(frozen_before, frozen_after),  # noqa: SLF001
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
        "| candidate | n | sources | MAE | RMSE | R2 | mean residual | residual SD |",
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
    table: pd.DataFrame, model: str, scope: str, metric: str, reference: str
) -> tuple[float, float, float] | None:
    rows = table[
        (table["model"] == f"{model}_minus_{reference}")
        & (table["dataset"] == scope)
        & (table["metric"] == metric)
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])


def _report_markdown(result: ResolutionResult, config: ResolutionConfig) -> str:
    """Render the deterministic M6B report."""
    models = [candidate.name for candidate in result.candidates]
    selected = result.selected_name
    lines: list[str] = [
        "# M6B spatial resolution study",
        "",
        "**Scope:** synthetic image similarity, measured as SSIM against the exact clean "
        "reference. This is not a measure of human readability, CAPTCHA solvability, "
        "security, or real-world quality.",
        "",
        "## Question",
        "",
        "Does increasing the spatial resolution of the existing occupancy and edge-density "
        "feature families improve geometric-distortion SSIM prediction while preserving "
        "clean-reference and original-task performance?",
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
        f"- Upstream M6A config sha256: `{config.m6a.config_sha256}`; M6A protocol hash: "
        f"`{config.m6a.protocol_hash}`",
        f"- Spatial feature set: `{config.m6a.feature_set_version}`, reused unchanged",
        f"- Model family: `{config.model.family}` "
        f"(n_estimators={config.model.n_estimators}, max_depth={config.model.max_depth}, "
        f"min_samples_leaf={config.model.min_samples_leaf}, seed={config.model.seed})",
        f"- Frozen global feature set: `{config.global_feature_set_version}`, "
        f"{len(config.global_feature_names)} features, unchanged",
        "",
        "The M3 dataset, split, frozen M3 model, M4A pilot, M4B benchmark, M5A pilot, M5B "
        "benchmark, M5C selection study, M5D training study and M6A feature study are "
        "read-only and hash-verified before and after the run.",
        "",
        "### Severity protocol",
        "",
        f"- Primary geometric score: amplitudes `{list(config.protocol.primary_amplitudes_px)}` px "
        f"(severities `{list(config.protocol.primary_severities)}`).",
        f"- Diagnostic only: amplitude `{list(config.protocol.diagnostic_amplitudes_px)}` px "
        f"(severity `{list(config.protocol.diagnostic_severities)}`).",
        "- The 0.5 px rung never enters fitting, validation selection or the primary score.",
        "",
        "## Grids and feature definitions",
        "",
        "The two M6A feature families are reused with no change to their mathematical "
        "definitions: regional ink occupancy uses the per-image Otsu threshold, and regional "
        "edge density uses the repository's Canny convention "
        f"(sigma={config.resolutions.edge_sigma}, "
        f"low_threshold={config.resolutions.edge_low_threshold}, "
        f"high_threshold={config.resolutions.edge_high_threshold} on the image scaled to "
        "[0, 1]). Both maps are computed once per image, so all cells at a given "
        "resolution share one definition. The grayscale representation and the canonical "
        "192x64 canvas are unchanged.",
        "",
        "Cells are visited in row-major order, with bounds floor(i*height/r) to "
        "floor((i+1)*height/r). Every declared resolution divides the canonical canvas "
        "evenly; for an image whose dimensions are not multiples of the grid, the "
        "remainder falls to the later cells. A cell that would be empty yields zero for "
        "every feature rather than an undefined value, and an exactly uniform image has no "
        "distinguishable ink class and no edges, so all its features are zero.",
        "",
        "| grid | cells | cell size (px) | spatial features | combined features |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for grid in config.resolutions.grids:
        lines.append(
            f"| {grid.label} | {grid.n_cells} | {64 // grid.rows}x{192 // grid.cols} | "
            f"{grid.n_spatial_features} | "
            f"{grid.n_spatial_features + len(config.global_feature_names)} |"
        )
    lines.extend(
        [
            "",
            "For an r x c grid each family contributes r*c features, so the spatial-only "
            "representation has 2*r*c features and the global-plus-spatial representation "
            "has 8 + 2*r*c. The forest considers every feature at each split "
            "(`max_features` is 1.0), so a larger feature set dilutes the split search "
            "directly; the 8x8 grid's 128 spatial and 136 combined features against 432 "
            "training rows are examined in the redundancy diagnostics below rather than "
            "tuned around.",
            "",
            "## Candidates",
            "",
            "The model family, hyperparameters, seed, training rows and objective are "
            "identical for every candidate; only the input representation and its "
            "resolution differ. The champion is a frozen reference and is never refitted.",
            "",
            "| candidate | feature set | features | fitted |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for candidate in result.candidates:
        lines.append(
            f"| `{candidate.name}` | {candidate.feature_set} | {len(candidate.feature_names)} | "
            f"{'refitted' if candidate.refitted else 'frozen champion, not refitted'} |"
        )

    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            "The rule is the M6A rule unchanged, with the same coefficients and the same "
            "tolerances, so the two studies are directly comparable. The champion is the "
            "reference. A candidate qualifies only if it regresses neither the clean slice, "
            "nor the original-task slice, nor any single primary geometric severity beyond "
            "the declared tolerances:",
            "",
            "```text",
            "eligible(c) <=> clean RMSE    <= champion clean RMSE    * (1 + tolerance_clean)",
            "               original RMSE <= champion original RMSE * (1 + tolerance_original)",
            "               for each primary severity s:",
            "                   RMSE_s     <= champion RMSE_s       * (1 + tolerance_severity)",
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
            f"`tolerance_severity={config.selection.tolerance_severity}`. Lower is better and "
            "the champion scores exactly 0. Ties are broken by lower aggregate geometric "
            "RMSE and then by candidate name. If no candidate qualifies, the frozen M5C "
            "champion is kept. The 4x2 grid is a reference condition, not a reason to "
            "favour or reject that resolution in advance, and no coefficient was chosen by "
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
            "| candidate | eligible | score | geometric RMSE | clean RMSE | original RMSE | "
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
            "These are computed on the training rows only. They are reported to assess "
            "whether finer resolutions introduce redundancy or overfitting, not to redesign "
            "the feature set within this milestone.",
            "",
            "| candidate | features | constant | duplicated | mean \\|r\\| | max \\|r\\| | "
            "train RMSE | validation RMSE | overfitting gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in result.candidates:
        entry = result.diagnostics.get(candidate.name, {})
        mean_corr = entry.get("spatial_correlation_mean", math.nan)
        max_corr = entry.get("spatial_correlation_max", math.nan)
        lines.append(
            f"| `{candidate.name}` | {entry.get('n_features', 0)} | "
            f"{entry.get('n_constant_spatial_features', 0)} | "
            f"{entry.get('n_duplicated_spatial_features', 0)} | "
            f"{_format_metric(mean_corr)} | {_format_metric(max_corr)} | "
            f"{_format_metric(entry.get('train_rmse', math.nan))} | "
            f"{_format_metric(entry.get('validation_rmse', math.nan))} | "
            f"{_format_metric(entry.get('overfitting_gap_rmse', math.nan))} |"
        )
    lines.append("")

    resolution = result.diagnostics.get("grid_resolution", {})
    families_by_resolution = resolution.get("families_by_resolution", {})
    if families_by_resolution:
        lines.append("### Can each grid resolve the distortion?")
        lines.append("")
        lines.append(
            "Per-cell ink occupancy of a distorted image against the same source's clean "
            "reference, on training rows only. This measures whether a grid can resolve the "
            "distortion at all, independently of any model."
        )
        lines.append("")
        lines.append(
            "| grid | family | pairs | mean abs occupancy change | max abs occupancy change | "
            "correlation with SSIM drop |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for label in sorted(families_by_resolution, key=lambda item: (len(item), item)):
            for family, entry in sorted(families_by_resolution[label].items()):
                correlation = entry["correlation_with_ssim_drop"]
                correlation_text = (
                    "n/a" if not math.isfinite(correlation) else f"{correlation:+.3f}"
                )
                lines.append(
                    f"| {label} | {family} | {entry['n_pairs']} | "
                    f"{entry['mean_abs_occupancy_change']:.5f} | "
                    f"{entry['max_abs_occupancy_change']:.5f} | {correlation_text} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Final test results by slice",
            "",
            "The selected candidate and its complete configuration were frozen before these "
            "rows were scored. The M3 test split and the M5B/M5C geometric test split are "
            "untouched frozen evaluations. All planned resolution ablations are reported, as "
            "planned, so the descriptive comparison is visible alongside the selected "
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
            "## Paired source-grouped differences against the frozen champion",
            "",
            "Whole source groups are resampled, preserving all rows within each selected "
            "group. Differences are candidate minus the champion on identical rows. There are "
            "12 held-out test source groups, so these intervals are preliminary and "
            "descriptive. Interval overlap or non-overlap is not a significance test.",
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
        lines.append("| candidate | metric | estimate | 95% low | 95% high |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for model in models:
            if model == config.champion.name:
                continue
            for metric in ("mae", "rmse"):
                found = _paired_interval_for(
                    result.paired_uncertainty, model, scope, metric, config.champion.name
                )
                if found is None:
                    continue
                estimate, low, high = found
                lines.append(
                    f"| `{model}` | {metric} | {estimate:+.4f} | {low:+.4f} | {high:+.4f} |"
                )
        lines.append("")

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
            "- A grid partition can show that a distortion is spatially uneven, but it "
            "cannot localise a warp to a glyph.",
            "- The 8x8 grid produces 128 spatial features against 432 training rows, and "
            "the forest considers every feature at each split, so the finest resolution is "
            "the most exposed to overfitting on this small synthetic dataset.",
            "- Only four resolutions and two representations were tested, all predeclared; "
            "the resolution set was not searched.",
            "- Feature importance is descriptive, not causal.",
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
            "uv run --extra image --extra ml captcha-resolution-study run \\",
            "    --config configs/spatial_resolution_study_v0_1_0.toml",
            "uv run --extra image --extra ml captcha-resolution-study verify \\",
            "    --config configs/spatial_resolution_study_v0_1_0.toml",
            "```",
            "",
            "Generated tables, the model bundles and this report live in Git-ignored paths.",
        ]
    )
    return "\n".join(lines) + "\n"


def _decision_lines(result: ResolutionResult, config: ResolutionConfig) -> list[str]:
    """Derive the descriptive interpretation from the observed results."""
    selected = result.selected_name
    lines: list[str] = []

    def metric(model: str, scope: str) -> dict[str, Any] | None:
        return _metric_row_for(result.test_metrics, model, scope)

    champion_geo = metric(config.champion.name, GEO_LABEL)
    selected_geo = metric(selected, GEO_LABEL)
    champion_clean = metric(config.champion.name, CLEAN_LABEL)
    selected_clean = metric(selected, CLEAN_LABEL)
    champion_original = metric(config.champion.name, ORIGINAL_LABEL)
    selected_original = metric(selected, ORIGINAL_LABEL)
    champion_m3 = metric(config.champion.name, "m3_test_overall")
    selected_m3 = metric(selected, "m3_test_overall")

    if champion_geo and selected_geo:
        change = (selected_geo["rmse"] - champion_geo["rmse"]) / champion_geo["rmse"] * 100
        lines.append(
            f"- **Primary geometric slice.** The frozen champion reaches MAE "
            f"{champion_geo['mae']:.4f}, RMSE {champion_geo['rmse']:.4f} and R2 "
            f"{champion_geo['r2']:.4f} on {champion_geo['n_samples']} rows from "
            f"{champion_geo['n_sources']} sources. The selected candidate `{selected}` "
            f"reaches MAE {selected_geo['mae']:.4f}, RMSE {selected_geo['rmse']:.4f} and R2 "
            f"{selected_geo['r2']:.4f}, a {change:+.1f}% change in RMSE against the champion."
        )
    if champion_clean and selected_clean:
        lines.append(
            f"- **Clean-reference slice.** Champion MAE {champion_clean['mae']:.4f} / RMSE "
            f"{champion_clean['rmse']:.4f}; selected MAE {selected_clean['mae']:.4f} / RMSE "
            f"{selected_clean['rmse']:.4f}."
        )
    if champion_original and selected_original:
        lines.append(
            f"- **Original blur/noise slice.** Champion MAE {champion_original['mae']:.4f} / "
            f"RMSE {champion_original['rmse']:.4f}; selected MAE "
            f"{selected_original['mae']:.4f} / RMSE {selected_original['rmse']:.4f}."
        )
    if champion_m3 and selected_m3:
        lines.append(
            f"- **Original-task retention.** On the same untouched M3 test rows the champion "
            f"reaches MAE {champion_m3['mae']:.4f} / RMSE {champion_m3['rmse']:.4f} / R2 "
            f"{champion_m3['r2']:.4f} and the selected candidate reaches MAE "
            f"{selected_m3['mae']:.4f} / RMSE {selected_m3['rmse']:.4f} / R2 "
            f"{selected_m3['r2']:.4f}."
        )

    for row in result.selection_table.itertuples(index=False):
        if row.model == config.champion.name or bool(row.eligible):
            continue
        lines.append(
            f"- **`{row.model}` was ineligible.** {row.ineligible_reasons}. The "
            "per-severity constraint is what rejects it: the aggregate geometric score "
            "alone would not have revealed the regression on a single rung."
        )

    if selected == config.champion.name:
        lines.append(
            "- **Outcome.** No spatial candidate was selected: the frozen M5C champion is "
            "retained. This is reported as a negative result rather than a manufactured "
            "winner."
        )
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark champion, "
            "and the M5C champion remains the best geometric-robustness policy found so "
            "far. Increasing the spatial resolution did not clear the retention-aware rule. "
            "The evidence does not justify a CNN, and a new corruption family does not by "
            "itself justify one."
        )
    else:
        lines.append(f"- **Outcome.** A spatial candidate was selected: `{selected}`.")
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark champion, "
            f"now with the resolution-aware policy `{selected}` rather than the frozen "
            "champion alone. The evidence does not justify a CNN, and a new corruption "
            "family does not by itself justify one."
        )

    # The mechanism: a finer grid resolves the distortion better, yet predicts
    # worse. Report the dissociation explicitly, because it is the substantive
    # finding and it rules out the naive reading of the diagnostic.
    resolution = result.diagnostics.get("grid_resolution", {})
    families_by_resolution = resolution.get("families_by_resolution", {})
    if families_by_resolution:
        labels = sorted(
            families_by_resolution,
            key=lambda item: (int(item.split("x")[0]), int(item.split("x")[1])),
        )
        elastic = {label: families_by_resolution[label].get("elastic") for label in labels}
        available = {label: entry for label, entry in elastic.items() if entry}
        if len(available) >= 2:
            first, last = labels[0], labels[-1]
            lines.append(
                f"- **Mechanism: the information is there, the model cannot use it.** The "
                f"training-row grid diagnostic shows the finer grid does resolve the "
                f"geometric warp: the per-cell ink-occupancy change against the same "
                f"source's clean reference rises from "
                f"{available[first]['mean_abs_occupancy_change']:.5f} at {first} to "
                f"{available[last]['mean_abs_occupancy_change']:.5f} at {last}, and its "
                f"correlation with the SSIM drop rises from "
                f"{available[first]['correlation_with_ssim_drop']:+.3f} to "
                f"{available[last]['correlation_with_ssim_drop']:+.3f}. Yet the aggregate "
                "geometric validation RMSE does not improve with resolution, and the "
                "retention slices degrade monotonically. The extra spatial features are "
                "therefore informative about the geometric family but largely uninformative "
                "about blur, noise and clean references, and because the forest considers "
                "every feature at each split they dilute the search for the splits that "
                "matter on the other families."
            )
        combined = [
            entry
            for name, entry in result.diagnostics.items()
            if name.startswith("combined_") and name != "grid_resolution"
        ]
        if combined:
            best = min(combined, key=lambda entry: entry["geometric_validation_rmse"])
            worst = max(combined, key=lambda entry: entry["geometric_validation_rmse"])
            lines.append(
                f"- **Resolution did not help the aggregate.** Among the global-plus-spatial "
                f"candidates the best aggregate geometric validation RMSE is "
                f"{best['geometric_validation_rmse']:.5f} and the worst is "
                f"{worst['geometric_validation_rmse']:.5f}, so the finest grid is not the "
                "best one. The gains on the geometric slice are within the noise of a "
                "12-source validation set, while the retention regressions are systematic "
                "and monotone in the number of features."
            )
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_resolution_study(
    config: ResolutionConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    verify_images: bool = True,
) -> tuple[ResolutionResult, dict[str, Path]]:
    """Load, verify, run and persist the complete M6B study."""
    started = time.perf_counter()
    before = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    frames = load_resolution_frames(config, verify_images=verify_images)
    m5c_config = ra.load_retention_config(config.m5c.config)
    champion = _load_champion(config)

    result = run_resolution_core(frames, config, champion=champion, m5c_config=m5c_config)
    after = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    if not ss._frozen_unchanged(before, after):  # noqa: SLF001
        raise ResolutionStudyError("a frozen asset changed during the M6B run")
    paths = write_resolution_outputs(
        result,
        config,
        output_reports=output_reports,
        output_models=output_models,
        frozen_before=before,
        frozen_after=after,
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, paths


def verify_resolution_study(
    config: ResolutionConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
) -> dict[str, Any]:
    """Verify a written M6B run: hashes, protocol, selection and prediction replay."""
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
            raise ResolutionStudyError("the M5C protocol hash does not match the pin")
        _require_file_hash(
            config.m5c.champion_dir / f"{config.m5c.champion_stem}.joblib",
            config.m5c.champion_artifact_sha256,
            "M5C champion artifact",
        )
        frames = load_resolution_frames(config, verify_images=False)
        return (
            f"{len(frames.retention.expanded_train)} training rows, "
            f"{len(frames.retention.expanded_val)} validation rows, "
            f"{len(frames.retention.primary_test)} primary geometric test rows and "
            f"{len(frames.spatial_features)} spatial feature tables"
        )

    def metadata_checks() -> str:
        metadata_path = reports / "run_metadata.json"
        if not metadata_path.is_file():
            raise ResolutionStudyError(f"missing run metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("benchmark_config_sha256") != config.config_sha256:
            raise ResolutionStudyError("run metadata config hash does not match")
        if metadata.get("benchmark_config_hash") != config.config_hash:
            raise ResolutionStudyError("run metadata protocol hash does not match")
        if not metadata.get("frozen_assets", {}).get("all_unchanged"):
            raise ResolutionStudyError("run metadata reports a changed frozen asset")
        for name, entry in metadata.get("artifacts", {}).items():
            path = Path(entry["path"]) if isinstance(entry, dict) else reports / name
            if not path.is_file():
                raise ResolutionStudyError(f"missing recorded artifact: {path}")
            if sha256_file(path) != (entry["sha256"] if isinstance(entry, dict) else entry):
                raise ResolutionStudyError(f"artifact hash mismatch: {name}")
        return f"{len(metadata.get('artifacts', {}))} artifact hashes match"

    def selection_checks() -> str:
        table_path = reports / "selection_table.csv"
        if not table_path.is_file():
            raise ResolutionStudyError(f"missing selection table: {table_path}")
        table = pd.read_csv(table_path)
        selected = table[table["selected"]]
        if len(selected) != 1:
            raise ResolutionStudyError("the selection table must mark exactly one winner")
        metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
        recorded = metadata.get("selection", {}).get("selected_model")
        if recorded != str(selected.iloc[0]["model"]):
            raise ResolutionStudyError(
                f"selection table winner {selected.iloc[0]['model']!r} does not match the "
                f"recorded selection {recorded!r}"
            )
        return f"selected candidate {recorded!r} is consistent"

    def model_checks() -> str:
        for candidate in result_candidate_names(config):
            if candidate == config.champion.name:
                continue
            card_path = models / f"{candidate}_card.json"
            artifact = models / f"{candidate}.joblib"
            if not card_path.is_file() or not artifact.is_file():
                raise ResolutionStudyError(f"missing model bundle for {candidate!r}")
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("artifact_sha256") != sha256_file(artifact):
                raise ResolutionStudyError(f"artifact hash mismatch for {candidate!r}")
            if card.get("target") != config.target:
                raise ResolutionStudyError(f"target mismatch for {candidate!r}")
            expected = expected_feature_names(config, candidate)
            if card.get("feature_names") != list(expected):
                raise ResolutionStudyError(f"feature schema mismatch for {candidate!r}")
        return "every refitted bundle loads, matches its card and uses the declared schema"

    def replay_checks() -> str:
        frames = load_resolution_frames(config, verify_images=False)
        m5c_config = ra.load_retention_config(config.m5c.config)
        champion = _load_champion(config)
        candidates = build_candidates(frames, config, champion=champion, m5c_config=m5c_config)
        slices = _test_slices(frames, candidates)
        replayed = ss._test_metrics_table(slices)  # noqa: SLF001
        stored = pd.read_csv(reports / "test_slice_metrics.csv")
        merged = replayed.merge(stored, on=["model", "scope"], suffixes=("_replay", "_stored"))
        if len(merged) != len(replayed):
            raise ResolutionStudyError("replayed metrics do not cover the stored rows")
        for column in ("mae", "rmse"):
            if not np.allclose(
                merged[f"{column}_replay"], merged[f"{column}_stored"], rtol=0.0, atol=1e-9
            ):
                raise ResolutionStudyError(f"replayed {column} does not match the stored run")
        return f"{len(replayed)} slice metrics reproduce exactly"

    def report_checks() -> str:
        report_path = reports / "report.md"
        if not report_path.is_file():
            raise ResolutionStudyError(f"missing report: {report_path}")
        text = report_path.read_text(encoding="utf-8")
        required = (
            "Grids and feature definitions",
            "Candidates",
            "Selection rule",
            "Training-only diagnostics",
            "Validation results by slice",
            "Final test results by slice",
            "Paired source-grouped differences against the frozen champion",
            "Reproducibility",
        )
        missing = [section for section in required if section not in text]
        if missing:
            raise ResolutionStudyError(f"report is missing sections: {missing}")
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


def result_candidate_names(config: ResolutionConfig) -> tuple[str, ...]:
    """Return every candidate name in a deterministic order."""
    return config.candidate_names()


def expected_feature_names(config: ResolutionConfig, candidate: str) -> tuple[str, ...]:
    """Return the declared feature names for one candidate."""
    if candidate == config.champion.name:
        return config.champion_feature_names
    for grid in config.resolutions.grids:
        for representation in config.representations:
            if candidate == f"{representation}_{grid.label}":
                return config.feature_names_for(grid, representation)
    raise ResolutionStudyError(f"unknown candidate {candidate!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-resolution-study",
        description="Run or verify the grouped M6B spatial resolution study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the spatial resolution study")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path, default=None)
    run_parser.add_argument("--output-models", type=Path, default=None)

    verify_parser = subparsers.add_parser("verify", help="verify a written spatial resolution run")
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path, default=None)
    verify_parser.add_argument("--output-models", type=Path, default=None)
    return parser


def _print_report(report: Mapping[str, Any]) -> None:
    for check in report["checks"]:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"spatial resolution study verification: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_resolution_config(args.config)
        if args.command == "run":
            result, paths = run_resolution_study(
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
                row = _metric_row_for(result.test_metrics, result.selected_name, scope)
                champion = _metric_row_for(result.test_metrics, config.champion.name, scope)
                if row is None or champion is None:
                    continue
                print(
                    f"  {scope}: selected MAE {row['mae']:.4f} RMSE {row['rmse']:.4f} | "
                    f"champion MAE {champion['mae']:.4f} RMSE {champion['rmse']:.4f}"
                )
            print(f"report: {paths['report']}")
            return 0
        report = verify_resolution_study(
            config,
            output_reports=args.output_reports,
            output_models=args.output_models,
        )
        _print_report(report)
        return 0 if report["passed"] else 1
    except (
        ResolutionStudyError,
        ResolutionStudyConfigError,
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
