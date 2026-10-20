"""Milestone 5D retention-aware training objectives.

M5C showed that a retention-aware *selection* rule can pick a conventional
model that improves geometric-distortion prediction while keeping the original
blur/noise task, but the model it selected was a different model family from
the one M5B had chosen, and the post-hoc affine calibration it tested did not
help. M5D asks a narrower question:

    Can a retention-aware *training objective* improve geometric-distortion SSIM
    prediction relative to the M5C champion, without materially degrading
    clean-reference and original-task performance?

This is a controlled training-objective experiment, not a new model-development
campaign. The model family, the hyperparameters, the training rows and the
seed are held fixed; the only thing that varies between the retrained
candidates is the per-example sample weight used during fitting. That makes any
difference attributable to the weighting scheme and to nothing else.

Four candidates are predeclared:

* **Frozen M5C champion** - the M5C random forest, loaded from its persisted
  bundle and never refitted. It is the primary reference and the fallback.
* **Retrained control** - the same model family, the same training rows and
  the same seed, fitted with the standard unweighted objective. Because the
  seed and the data are identical it is expected to reproduce the champion
  exactly; the run asserts this, which makes it both a control and a
  reproducibility check.
* **Severity-aware weighting** - geometric training rows receive an increased
  weight that ramps with the geometric severity, while clean and original-task
  rows keep unit weight.
* **Retention-aware weighting** - clean references are explicitly protected with
  a higher weight, and geometric rows still receive more weight than the
  original task.

Weights are built only from metadata already present in the training frame
(group membership and severity). They are never added to the predictor's
feature matrix, and the eight image-only features and their order are
unchanged. The weights are rescaled to mean one over the training frame so a
larger nominal scale cannot change their interpretation, and every weight must
be finite and strictly positive.

Selection uses a predeclared eligibility-and-score rule computed on validation
data only. A candidate must not regress the clean slice, the original-task
slice, or any single primary geometric severity beyond the declared
tolerances. Among the qualifying candidates the score rewards aggregate
geometric improvement and charges clean, original-task and per-severity
regressions. The selected candidate and its full configuration are frozen
before the untouched M3 and M5B/M5C test slices are scored.

The 0.5 px geometric rung remains a diagnostic only. It never enters fitting,
selection or the primary score. The target is synthetic image similarity (SSIM
against the exact clean reference). Nothing here measures human readability,
CAPTCHA solvability, security or real-world robustness.
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

import joblib
import numpy as np
import pandas as pd

from captcha_quality import geometric_benchmark as gb
from captcha_quality import retention_aware as ra
from captcha_quality import robustness_benchmark as rb
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.dataset import CheckResult
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import (
    TARGET,
    CandidateSpec,
    ModelError,
    model_inputs,
    predict,
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

#: Candidate policy names.
CHAMPION: Final[str] = "frozen_m5c_champion"
RETRAINED_CONTROL: Final[str] = "retrained_control"
SEVERITY_AWARE: Final[str] = "severity_aware"
RETENTION_AWARE: Final[str] = "retention_aware"

#: Weighting scheme names.
SCHEME_NONE: Final[str] = "none"
SCHEME_SEVERITY_AWARE: Final[str] = "severity_aware"
SCHEME_RETENTION_AWARE: Final[str] = "retention_aware"

#: Training group names.
GROUP_CLEAN: Final[str] = "clean"
GROUP_ORIGINAL: Final[str] = "original"
GROUP_GEOMETRIC: Final[str] = "geometric"

#: Slice labels.
GEO_LABEL: Final[str] = "geometric_primary"
CLEAN_LABEL: Final[str] = "clean_reference"
ORIGINAL_LABEL: Final[str] = "original_blur_noise"
OVERALL_LABEL: Final[str] = "overall"

_EXPECTED_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {"run", "schema", "m5c", "model", "protocol", "candidates", "weighting", "selection"}
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Frozen inputs audited before and after the milestone. M5D's own outputs are
#: not in this list.
FROZEN_ASSET_PATHS: Final[tuple[str, ...]] = (
    "configs/dataset_v0_1_0.toml",
    "configs/benchmark_v0_1_0.toml",
    "configs/robustness_pilot_occlusion_v0_1_0.toml",
    "configs/robustness_benchmark_v0_1_0.toml",
    "configs/geometric_distortion_pilot_v0_1_0.toml",
    "configs/geometric_robustness_benchmark_v0_1_0.toml",
    "configs/retention_aware_selection_v0_1_0.toml",
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "data/processed/geometric_distortion_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "models/robustness_benchmark_v0_1_0",
    "models/geometric_robustness_benchmark_v0_1_0",
    "models/retention_aware_selection_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
    "reports/geometric_distortion_pilot",
    "reports/geometric_robustness_benchmark_v0_1_0",
    "reports/retention_aware_selection_v0_1_0",
)


class RetentionTrainingError(RuntimeError):
    """Raised when the M5D protocol cannot be honoured."""


class RetentionTrainingConfigError(ValueError):
    """Raised when the M5D configuration is invalid."""


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
    """The fixed model family and hyperparameters shared by every candidate."""

    family: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    seed: int


@dataclass(frozen=True, slots=True)
class WeightScheme:
    """One predeclared sample-weighting scheme."""

    name: str
    clean_weight: float
    original_weight: float
    geometric_base_weight: float
    geometric_severity_ratio: float


@dataclass(frozen=True, slots=True)
class WeightingSettings:
    """The weighting design: normalisation target and the scheme table."""

    normalise_to_mean: float
    min_primary_severity: int
    schemes: Mapping[str, WeightScheme]

    def scheme(self, name: str) -> WeightScheme:
        try:
            return self.schemes[name]
        except KeyError as exc:
            raise RetentionTrainingConfigError(
                f"unknown weighting scheme {name!r}; known schemes are {sorted(self.schemes)}"
            ) from exc


@dataclass(frozen=True, slots=True)
class TrainingSelectionSettings:
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
class RetentionTrainingConfig:
    """The complete M5D protocol."""

    run: rb.RunSettings
    schema: rb.SchemaSettings
    m5c: M5CPin
    model: ModelSettings
    protocol: ra.ProtocolSettings
    candidate_names: tuple[str, ...]
    candidate_schemes: Mapping[str, str]
    weighting: WeightingSettings
    selection: TrainingSelectionSettings
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
                "feature_set_version": self.schema.feature_set_version,
                "target": self.schema.target,
                "feature_names": list(self.schema.feature_names),
                "split_names": list(self.schema.split_names),
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
            "candidates": {
                "names": list(self.candidate_names),
                "schemes": {
                    key: self.candidate_schemes[key] for key in sorted(self.candidate_schemes)
                },
            },
            "weighting": {
                "normalise_to_mean": self.weighting.normalise_to_mean,
                "min_primary_severity": self.weighting.min_primary_severity,
                "schemes": {
                    name: {
                        "clean_weight": scheme.clean_weight,
                        "original_weight": scheme.original_weight,
                        "geometric_base_weight": scheme.geometric_base_weight,
                        "geometric_severity_ratio": scheme.geometric_severity_ratio,
                    }
                    for name, scheme in sorted(self.weighting.schemes.items())
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


# ---------------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------------


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise RetentionTrainingConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise RetentionTrainingConfigError(f"missing or invalid [{name}] table")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetentionTrainingConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetentionTrainingConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RetentionTrainingConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RetentionTrainingConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RetentionTrainingConfigError(f"{label} must be a boolean, got {value!r}")
    return value


def _as_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RetentionTrainingConfigError(f"{label} must be a TOML array")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    items = tuple(_as_str(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise RetentionTrainingConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_int_tuple(value: object, label: str) -> tuple[int, ...]:
    items = tuple(_as_int(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise RetentionTrainingConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_float_tuple(value: object, label: str) -> tuple[float, ...]:
    items = tuple(_as_float(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise RetentionTrainingConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise RetentionTrainingConfigError(f"{label} must be a lowercase SHA-256 hex digest")
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


def _parse_schema(raw: Mapping[str, object]) -> rb.SchemaSettings:
    _check_keys(raw, {"feature_set_version", "target", "feature_names", "split_names"}, "schema")
    return rb.SchemaSettings(
        feature_set_version=_as_str(raw["feature_set_version"], "schema.feature_set_version"),
        target=_as_str(raw["target"], "schema.target"),
        feature_names=_as_str_tuple(raw["feature_names"], "schema.feature_names"),
        split_names=_as_str_tuple(raw["split_names"], "schema.split_names"),
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
        raise RetentionTrainingConfigError(
            f"model.family must be 'random_forest', got {settings.family!r}"
        )
    if settings.n_estimators < 1:
        raise RetentionTrainingConfigError("model.n_estimators must be at least 1")
    if settings.max_depth < 0:
        raise RetentionTrainingConfigError("model.max_depth must be zero (unlimited) or positive")
    if settings.min_samples_leaf < 1:
        raise RetentionTrainingConfigError("model.min_samples_leaf must be at least 1")
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
        primary_amplitudes_px=_as_float_tuple(
            raw["primary_amplitudes_px"], "protocol.primary_amplitudes_px"
        ),
        diagnostic_amplitudes_px=_as_float_tuple(
            raw["diagnostic_amplitudes_px"], "protocol.diagnostic_amplitudes_px"
        ),
        primary_severities=_as_int_tuple(raw["primary_severities"], "protocol.primary_severities"),
        diagnostic_severities=_as_int_tuple(
            raw["diagnostic_severities"], "protocol.diagnostic_severities"
        ),
    )
    if set(settings.primary_amplitudes_px) & set(settings.diagnostic_amplitudes_px):
        raise RetentionTrainingConfigError(
            "the diagnostic amplitude must not also appear in the primary ladder"
        )
    if set(settings.primary_severities) & set(settings.diagnostic_severities):
        raise RetentionTrainingConfigError(
            "the diagnostic severity must not also appear in the primary ladder"
        )
    return settings


def _parse_weight_scheme(name: str, raw: Mapping[str, object]) -> WeightScheme:
    _check_keys(
        raw,
        {
            "clean_weight",
            "original_weight",
            "geometric_base_weight",
            "geometric_severity_ratio",
        },
        f"weighting.schemes.{name}",
    )
    scheme = WeightScheme(
        name=name,
        clean_weight=_as_float(raw["clean_weight"], f"{name}.clean_weight"),
        original_weight=_as_float(raw["original_weight"], f"{name}.original_weight"),
        geometric_base_weight=_as_float(
            raw["geometric_base_weight"], f"{name}.geometric_base_weight"
        ),
        geometric_severity_ratio=_as_float(
            raw["geometric_severity_ratio"], f"{name}.geometric_severity_ratio"
        ),
    )
    for label, value in (
        ("clean_weight", scheme.clean_weight),
        ("original_weight", scheme.original_weight),
        ("geometric_base_weight", scheme.geometric_base_weight),
        ("geometric_severity_ratio", scheme.geometric_severity_ratio),
    ):
        if value <= 0.0:
            raise RetentionTrainingConfigError(
                f"{name}.{label} must be strictly positive so every weight is valid"
            )
    return scheme


def _parse_weighting(raw: Mapping[str, object]) -> WeightingSettings:
    _check_keys(raw, {"normalise_to_mean", "min_primary_severity", "schemes"}, "weighting")
    schemes_raw = raw["schemes"]
    if not isinstance(schemes_raw, dict):
        raise RetentionTrainingConfigError("weighting.schemes must be a table")
    schemes = {
        str(name): _parse_weight_scheme(str(name), value)
        for name, value in schemes_raw.items()
        if isinstance(value, dict)
    }
    if not schemes:
        raise RetentionTrainingConfigError("weighting.schemes must not be empty")
    settings = WeightingSettings(
        normalise_to_mean=_as_float(raw["normalise_to_mean"], "weighting.normalise_to_mean"),
        min_primary_severity=_as_int(raw["min_primary_severity"], "weighting.min_primary_severity"),
        schemes=schemes,
    )
    if settings.normalise_to_mean <= 0.0:
        raise RetentionTrainingConfigError("weighting.normalise_to_mean must be positive")
    return settings


def _parse_selection(raw: Mapping[str, object]) -> TrainingSelectionSettings:
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
    settings = TrainingSelectionSettings(
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
        raise RetentionTrainingConfigError(
            f"selection.rule must be 'retention_gated_rmse_delta', got {settings.rule!r}"
        )
    for label, value in (
        ("w_geometric", settings.w_geometric),
        ("w_clean", settings.w_clean),
        ("w_original", settings.w_original),
        ("w_severity", settings.w_severity),
    ):
        if value < 0.0:
            raise RetentionTrainingConfigError(f"selection.{label} must be non-negative")
    for label, value in (
        ("tolerance_clean", settings.tolerance_clean),
        ("tolerance_original", settings.tolerance_original),
        ("tolerance_severity", settings.tolerance_severity),
    ):
        if value < 0.0:
            raise RetentionTrainingConfigError(f"selection.{label} must be non-negative")
    return settings


def load_retention_training_config(path: Path | str) -> RetentionTrainingConfig:
    """Load and validate the M5D configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise RetentionTrainingConfigError(f"missing configuration file: {config_path}")
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    unknown = sorted(set(payload) - _EXPECTED_TOP_KEYS)
    if unknown:
        raise RetentionTrainingConfigError(
            f"unknown top-level key(s): {unknown}; allowed keys are {sorted(_EXPECTED_TOP_KEYS)}"
        )
    candidates_raw = _section(payload, "candidates")
    _check_keys(candidates_raw, {"names", "schemes"}, "candidates")
    names = _as_str_tuple(candidates_raw["names"], "candidates.names")
    schemes_raw = candidates_raw["schemes"]
    if not isinstance(schemes_raw, dict):
        raise RetentionTrainingConfigError("candidates.schemes must be a table")
    schemes = {
        str(key): _as_str(value, f"candidates.schemes.{key}") for key, value in schemes_raw.items()
    }
    if set(schemes) != set(names):
        raise RetentionTrainingConfigError(
            "candidates.schemes must name exactly the candidates in candidates.names"
        )

    config = RetentionTrainingConfig(
        run=_parse_run(_section(payload, "run")),
        schema=_parse_schema(_section(payload, "schema")),
        m5c=_parse_m5c(_section(payload, "m5c")),
        model=_parse_model(_section(payload, "model")),
        protocol=_parse_protocol(_section(payload, "protocol")),
        candidate_names=names,
        candidate_schemes=schemes,
        weighting=_parse_weighting(_section(payload, "weighting")),
        selection=_parse_selection(_section(payload, "selection")),
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )
    if config.schema.feature_names != tuple(FEATURE_NAMES):
        raise RetentionTrainingConfigError(
            f"schema.feature_names must match the extractor {list(FEATURE_NAMES)}"
        )
    if config.schema.target != TARGET:
        raise RetentionTrainingConfigError(f"schema.target must be {TARGET!r}")
    if config.protocol.primary_amplitudes_px != gb.PRIMARY_AMPLITUDES_PX:
        raise RetentionTrainingConfigError(
            "protocol.primary_amplitudes_px must match the M5B primary ladder "
            f"{list(gb.PRIMARY_AMPLITUDES_PX)}"
        )
    if config.protocol.diagnostic_amplitudes_px != gb.DIAGNOSTIC_AMPLITUDES_PX:
        raise RetentionTrainingConfigError(
            "protocol.diagnostic_amplitudes_px must match the M5B diagnostic ladder "
            f"{list(gb.DIAGNOSTIC_AMPLITUDES_PX)}"
        )
    expected = {CHAMPION, RETRAINED_CONTROL, SEVERITY_AWARE, RETENTION_AWARE}
    if set(config.candidate_names) != expected:
        raise RetentionTrainingConfigError(
            f"candidates.names must be {sorted(expected)}, got {sorted(config.candidate_names)}"
        )
    if config.candidate_schemes[CHAMPION] != SCHEME_NONE:
        raise RetentionTrainingConfigError(
            "the champion candidate must use the 'none' scheme; it is loaded, never refitted"
        )
    if config.candidate_schemes[RETRAINED_CONTROL] != SCHEME_NONE:
        raise RetentionTrainingConfigError("the retrained control must use the 'none' scheme")
    for name in (SEVERITY_AWARE, RETENTION_AWARE):
        if config.candidate_schemes[name] not in config.weighting.schemes:
            raise RetentionTrainingConfigError(
                f"candidate {name!r} uses unknown scheme {config.candidate_schemes[name]!r}"
            )
    return config


# ---------------------------------------------------------------------------
# Frames, groups and sample weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingFrames:
    """The frozen frames and the training-group assignment used for weighting."""

    retention: ra.RetentionFrames
    train: pd.DataFrame
    validation: pd.DataFrame
    groups: pd.Series
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RetentionTrainingError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RetentionTrainingError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def assign_training_groups(frame: pd.DataFrame) -> pd.Series:
    """Assign each training row to its weighting group.

    The assignment uses only metadata that is already present in the training
    frame: the dataset a row came from, whether it is a clean reference, and
    its corruption family. It never reads the target, the image features, the
    test split or any test-derived value.
    """
    dataset = frame["dataset"].astype(str)
    family = frame["family"].astype(str)
    is_clean = frame["is_clean"].astype(bool)
    groups = pd.Series(index=frame.index, dtype=object)
    groups.loc[(dataset == gb.M3_LABEL) & is_clean] = GROUP_CLEAN
    groups.loc[(dataset == gb.M3_LABEL) & ~is_clean] = GROUP_ORIGINAL
    groups.loc[(dataset == gb.M5A_LABEL) & (family == gb.ELASTIC_FAMILY)] = GROUP_GEOMETRIC
    if groups.isna().any():
        unassigned = sorted(set(frame.loc[groups.isna(), "sample_id"].astype(str)))
        raise RetentionTrainingError(
            f"{len(unassigned)} training row(s) could not be assigned to a weighting group "
            f"(first: {unassigned[0]})"
        )
    return groups


def _check_group_assignment(frame: pd.DataFrame, groups: pd.Series) -> CheckResult:
    """Confirm the group assignment is complete, disjoint and target-independent."""
    try:
        if groups.isna().any():
            raise RetentionTrainingError("some training rows have no weighting group")
        if set(groups.unique()) != {GROUP_CLEAN, GROUP_ORIGINAL, GROUP_GEOMETRIC}:
            raise RetentionTrainingError(f"unexpected group set {sorted(set(groups.unique()))}")
        if len(frame) != len(groups):
            raise RetentionTrainingError("group vector length does not match the frame")
        # The assignment must be a function of the frame's metadata alone: every
        # row sharing the same metadata pattern must receive the same group.
        metadata = frame[["dataset", "family", "is_clean"]].astype(str).agg("|".join, axis=1)
        by_pattern = groups.groupby(metadata).nunique()
        if (by_pattern > 1).any():
            conflicting = sorted(by_pattern[by_pattern > 1].index)[:3]
            raise RetentionTrainingError(
                f"identical metadata received different groups (patterns: {conflicting})"
            )
        # The assignment must not depend on the target: removing the target
        # column entirely must produce exactly the same grouping.
        without_target = frame.drop(columns=[TARGET])
        recomputed = assign_training_groups(without_target)
        if not recomputed.astype(str).equals(groups.astype(str)):
            raise RetentionTrainingError(
                "the group assignment changes when the target column is removed, so it "
                "depends on the target"
            )
        return CheckResult(
            name="weight_group_assignment",
            passed=True,
            detail=(
                f"{len(frame)} training rows assigned to "
                f"{', '.join(f'{name}={int((groups == name).sum())}' for name in sorted(set(groups)))} "
                "using dataset, family and clean-flag metadata only; the assignment is "
                "unchanged when the target column is removed"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="weight_group_assignment", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _check_weight_validity(weights: np.ndarray, frame: pd.DataFrame) -> CheckResult:
    """Confirm every training weight is finite, strictly positive and normalised."""
    try:
        if len(weights) != len(frame):
            raise RetentionTrainingError("weight vector length does not match the frame")
        if not np.isfinite(weights).all():
            raise RetentionTrainingError("weights contain non-finite values")
        if (weights <= 0.0).any():
            raise RetentionTrainingError("weights contain non-positive values")
        mean = float(weights.mean())
        if not math.isclose(mean, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RetentionTrainingError(
                f"weights are not normalised to mean 1.0 (mean is {mean:.12f})"
            )
        return CheckResult(
            name="weight_validity",
            passed=True,
            detail=(
                f"{len(weights)} finite, strictly positive weights with mean "
                f"{mean:.6f} and range {weights.min():.6f}-{weights.max():.6f}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="weight_validity", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def build_sample_weights(
    frame: pd.DataFrame,
    groups: pd.Series,
    scheme: WeightScheme,
    settings: WeightingSettings,
) -> np.ndarray:
    """Build the normalised per-example training weights for one scheme.

    The weight of an example is
    ``group_weight[group] * severity_factor[group, severity]``. The severity
    factor is 1 for the clean and original groups and
    ``geometric_severity_ratio ** (severity - min_primary_severity)`` for the
    geometric group. The result is rescaled so the mean weight is exactly 1.
    """
    if scheme.name == SCHEME_NONE:
        return np.ones(len(frame), dtype=np.float64)
    severity = frame["severity_level"].astype(int).to_numpy()
    group_values = groups.astype(str).to_numpy()
    weights = np.empty(len(frame), dtype=np.float64)
    for index, group in enumerate(group_values):
        if group == GROUP_CLEAN:
            weights[index] = scheme.clean_weight
        elif group == GROUP_ORIGINAL:
            weights[index] = scheme.original_weight
        elif group == GROUP_GEOMETRIC:
            exponent = int(severity[index]) - settings.min_primary_severity
            if exponent < 0:
                raise RetentionTrainingError(
                    f"geometric severity {severity[index]} is below the declared minimum "
                    f"{settings.min_primary_severity}"
                )
            weights[index] = scheme.geometric_base_weight * (
                scheme.geometric_severity_ratio**exponent
            )
        else:  # pragma: no cover - assign_training_groups already rejects this
            raise RetentionTrainingError(f"unknown weighting group {group!r}")
    if not np.isfinite(weights).all():
        raise RetentionTrainingError("computed weights contain non-finite values")
    if (weights <= 0.0).any():
        raise RetentionTrainingError("computed weights contain non-positive values")
    mean = float(weights.mean())
    if mean <= 0.0:
        raise RetentionTrainingError("computed weights have a non-positive mean")
    return weights * (settings.normalise_to_mean / mean)


def _check_training_isolation(
    train: pd.DataFrame, validation: pd.DataFrame, primary_test: pd.DataFrame
) -> CheckResult:
    """Confirm no test row or test source can reach the training frame."""
    try:
        if set(train["split"]) != {"train"}:
            raise RetentionTrainingError("training frame is not train-only")
        if set(validation["split"]) != {"val"}:
            raise RetentionTrainingError("validation frame is not val-only")
        train_sources = set(train["source_id"].astype(str))
        validation_sources = set(validation["source_id"].astype(str))
        test_sources = set(primary_test["source_id"].astype(str))
        if train_sources & test_sources:
            raise RetentionTrainingError("a test source appears in the training frame")
        if validation_sources & test_sources:
            raise RetentionTrainingError("a test source appears in the validation frame")
        if train_sources & validation_sources:
            raise RetentionTrainingError("a source appears in both train and validation")
        if set(train["sample_id"]) & set(validation["sample_id"]):
            raise RetentionTrainingError("a sample appears in both train and validation")
        return CheckResult(
            name="training_isolation",
            passed=True,
            detail=(
                f"{len(train)} training rows from {len(train_sources)} sources and "
                f"{len(validation)} validation rows from {len(validation_sources)} sources "
                f"are disjoint from the {len(test_sources)} held-out test sources"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="training_isolation", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _check_diagnostic_rung_excluded(train: pd.DataFrame, validation: pd.DataFrame) -> CheckResult:
    """Confirm the 0.5 px rung is absent from every fit and selection frame."""
    try:
        for frame, label in ((train, "training"), (validation, "validation")):
            m5a_rows = frame[frame["dataset"] == gb.M5A_LABEL]
            if (
                not m5a_rows.empty
                and gb._amplitude_mask(  # noqa: SLF001
                    m5a_rows, gb.DIAGNOSTIC_AMPLITUDES_PX
                ).any()
            ):
                raise RetentionTrainingError(f"0.5 px rows entered the {label} frame")
            if not m5a_rows.empty and set(m5a_rows["severity_level"].astype(int)) & set(
                gb.DIAGNOSTIC_SEVERITIES
            ):
                raise RetentionTrainingError(f"the diagnostic severity entered the {label} frame")
        return CheckResult(
            name="diagnostic_rung_exclusion",
            passed=True,
            detail=(
                "the 0.5 px rung is absent from the training and validation frames and "
                "appears only in the separately labelled diagnostic test frame"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="diagnostic_rung_exclusion", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def load_training_frames(
    config: RetentionTrainingConfig, *, verify_images: bool = True
) -> TrainingFrames:
    """Load the frozen M3/M5A inputs through the M5C verified path."""
    _require_file_hash(config.m5c.config, config.m5c.config_sha256, "M5C config")
    m5c_config = ra.load_retention_config(config.m5c.config)
    if m5c_config.config_hash != config.m5c.protocol_hash:
        raise RetentionTrainingError(
            "M5C protocol hash mismatch: expected "
            f"{config.m5c.protocol_hash}, found {m5c_config.config_hash}"
        )
    retention = ra.load_retention_frames(m5c_config, verify_images=verify_images)
    train = retention.expanded_train
    validation = retention.expanded_val
    groups = assign_training_groups(train)
    checks = (
        *retention.checks,
        _check_group_assignment(train, groups),
        _check_training_isolation(train, validation, retention.primary_test),
        _check_diagnostic_rung_excluded(train, validation),
    )
    rb._verify_checks(checks, "M5D frame checks")  # noqa: SLF001
    return TrainingFrames(
        retention=retention,
        train=train,
        validation=validation,
        groups=groups,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingCandidate:
    """One predeclared candidate training policy."""

    name: str
    scheme: str
    estimator: Any
    weights: np.ndarray | None
    refitted: bool

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return predict(self.estimator, frame)


@dataclass(frozen=True, slots=True)
class CandidateBundle:
    """All candidates with their validation predictions and weight diagnostics."""

    candidates: tuple[TrainingCandidate, ...]
    validation_predictions: dict[str, np.ndarray]
    weight_diagnostics: dict[str, dict[str, Any]]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(candidate.name for candidate in self.candidates)


def _champion_spec(
    config: RetentionTrainingConfig, m5c_config: ra.RetentionConfig
) -> CandidateSpec:
    """Return the champion's grid specification and confirm it matches the config.

    The M5C champion was fitted from a specification in the M5B grid, and the
    M5C configuration records which one it was. M5D re-derives that
    specification and checks it against the pinned M5D model settings, so an
    upstream drift is detected rather than silently changing the comparison.
    """
    m5b_config = ra._load_m5b_config(m5c_config)  # noqa: SLF001
    specs = rb.candidate_specs_from_config(m5b_config)  # type: ignore[arg-type]
    spec_name = m5c_config.candidates.alternative_spec
    spec = next((item for item in specs if item.name == spec_name), None)
    if spec is None:
        raise RetentionTrainingError(
            f"the M5C champion specification {spec_name!r} is not in the M5B grid"
        )
    if spec.family != config.model.family:
        raise RetentionTrainingError(
            f"M5D model.family is {config.model.family!r} but the M5C champion was "
            f"fitted from a {spec.family!r} specification"
        )
    expected = {
        "n_estimators": config.model.n_estimators,
        "max_depth": config.model.max_depth,
        "min_samples_leaf": config.model.min_samples_leaf,
    }
    for key, value in expected.items():
        if spec.params.get(key) != value:
            raise RetentionTrainingError(
                f"M5D model.{key} is {value!r} but the M5B grid specification has "
                f"{spec.params.get(key)!r}"
            )
    seed = derive_noise_seed(m5b_config.run.seed, "model", spec.name) % (2**32)
    if seed != config.model.seed:
        raise RetentionTrainingError(
            f"M5D model.seed is {config.model.seed} but the M5B grid derives {seed}"
        )
    return spec


def _fit_with_weights(
    spec: CandidateSpec,
    frame: pd.DataFrame,
    weights: np.ndarray | None,
    *,
    seed: int,
    use_sample_weight: bool = True,
) -> Any:
    """Fit one estimator on the training rows, optionally with sample weights.

    ``use_sample_weight`` is false for the unweighted control, which must use
    the standard training objective exactly. Passing an all-ones
    ``sample_weight`` is not equivalent to omitting the argument, because the
    tree split criterion is then evaluated on weighted statistics.
    """
    estimator = spec.build(seed=seed)
    target = frame[TARGET].to_numpy(dtype=np.float64)
    inputs = model_inputs(frame)
    if weights is None or not use_sample_weight:
        estimator.fit(inputs, target)
    else:
        estimator.fit(inputs, target, sample_weight=weights)
    return estimator


def _load_champion(config: RetentionTrainingConfig, m3_summary: Mapping[str, Any]) -> Any:
    """Load the frozen M5C champion bundle and confirm it is unchanged.

    The champion is verified against the M5C protocol it was produced under,
    not against the M3 dataset summary. The champion was fitted on the
    expanded M3+M5A frame rather than on the M3 dataset alone, so the M3
    dataset identity check does not apply to it. The checks below confirm the
    artifact hash, the frozen feature schema and target, the scikit-learn
    compatibility, the recorded M5C protocol hash and a smoke prediction.
    """
    directory = config.m5c.champion_dir
    stem = config.m5c.champion_stem
    artifact = directory / f"{stem}.joblib"
    card_path = directory / f"{stem}_card.json"
    _require_file_hash(artifact, config.m5c.champion_artifact_sha256, "M5C champion artifact")
    if not card_path.is_file():
        raise RetentionTrainingError(f"missing M5C champion card: {card_path}")
    card = json.loads(card_path.read_text(encoding="utf-8"))

    if card.get("feature_names") != list(FEATURE_NAMES):
        raise RetentionTrainingError("the M5C champion uses a different feature schema")
    if card.get("feature_set_version") != config.schema.feature_set_version:
        raise RetentionTrainingError("the M5C champion uses a different feature-set version")
    if int(card.get("n_features", -1)) != len(FEATURE_NAMES):
        raise RetentionTrainingError("the M5C champion n_features does not match the extractor")
    if card.get("target") != TARGET:
        raise RetentionTrainingError(f"the M5C champion target is not {TARGET!r}")
    recorded = str(card.get("selection", {}).get("benchmark_config_hash", ""))
    if recorded != config.m5c.protocol_hash:
        raise RetentionTrainingError(
            "the M5C champion was produced under protocol "
            f"{recorded!r}, not the pinned {config.m5c.protocol_hash!r}"
        )
    if card.get("artifact_sha256") != sha256_file(artifact):
        raise RetentionTrainingError("the M5C champion artifact hash does not match its card")
    if str(card.get("sklearn_version", "")).split(".")[:2] != sklearn_version().split(".")[:2]:
        raise RetentionTrainingError(
            f"the M5C champion was persisted with scikit-learn "
            f"{card.get('sklearn_version')}, current is {sklearn_version()}"
        )
    estimator = joblib.load(artifact)
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features != len(FEATURE_NAMES):
        raise RetentionTrainingError(
            f"the M5C champion expects {n_features} inputs, the extractor provides "
            f"{len(FEATURE_NAMES)}"
        )
    probe = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
    prediction = np.asarray(estimator.predict(probe), dtype=np.float64)
    if prediction.shape != (1,) or not np.isfinite(prediction).all():
        raise RetentionTrainingError("the M5C champion smoke prediction is not a finite scalar")
    return estimator


def build_candidate_bundle(
    frames: TrainingFrames,
    config: RetentionTrainingConfig,
    *,
    champion: Any,
    m5c_config: ra.RetentionConfig,
) -> CandidateBundle:
    """Build every predeclared candidate and generate its validation predictions."""
    spec = _champion_spec(config, m5c_config)
    validation = frames.validation
    candidates: list[TrainingCandidate] = []
    validation_predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    for name in config.candidate_names:
        scheme_name = config.candidate_schemes[name]
        if name == CHAMPION:
            estimator = champion
            weights = None
            refitted = False
        else:
            if scheme_name == SCHEME_NONE:
                weights = np.ones(len(frames.train), dtype=np.float64)
                use_sample_weight = False
            else:
                scheme = config.weighting.scheme(scheme_name)
                weights = build_sample_weights(
                    frames.train, frames.groups, scheme, config.weighting
                )
                use_sample_weight = True
            _check_weight_validity(weights, frames.train)
            estimator = _fit_with_weights(
                spec,
                frames.train,
                weights,
                seed=config.model.seed,
                use_sample_weight=use_sample_weight,
            )
            refitted = True
        candidates.append(
            TrainingCandidate(
                name=name,
                scheme=scheme_name,
                estimator=estimator,
                weights=weights,
                refitted=refitted,
            )
        )
        validation_predictions[name] = predict(estimator, validation)
        if weights is None:
            uniform = np.ones(len(frames.train), dtype=np.float64)
            groups = frames.groups.astype(str)
            diagnostics[name] = {
                "scheme": SCHEME_NONE,
                "refitted": refitted,
                "n_weighted_rows": 0,
                "mean_weight": math.nan,
                "min_weight": math.nan,
                "max_weight": math.nan,
                "group_weight_means": {
                    group: float(uniform[groups == group].mean()) for group in sorted(set(groups))
                },
                "group_weight_shares": {
                    group: float(uniform[groups == group].sum() / uniform.sum())
                    for group in sorted(set(groups))
                },
            }
        else:
            groups = frames.groups.astype(str)
            group_means = {
                group: float(weights[groups == group].mean()) for group in sorted(set(groups))
            }
            # The share of total training weight each group carries. Mean-one
            # normalisation keeps the overall scale comparable, but a group with
            # many rows can still carry more total influence than a small group
            # with a high per-row weight, so the share is recorded explicitly.
            total = float(weights.sum())
            group_shares = {
                group: float(weights[groups == group].sum() / total)
                for group in sorted(set(groups))
            }
            diagnostics[name] = {
                "scheme": scheme_name,
                "refitted": refitted,
                "n_weighted_rows": int(len(weights)),
                "mean_weight": float(weights.mean()),
                "min_weight": float(weights.min()),
                "max_weight": float(weights.max()),
                "group_weight_means": group_means,
                "group_weight_shares": group_shares,
            }
    return CandidateBundle(
        candidates=tuple(candidates),
        validation_predictions=validation_predictions,
        weight_diagnostics=diagnostics,
    )


def check_control_reproduces_champion(
    bundle: CandidateBundle, frames: TrainingFrames
) -> CheckResult:
    """Confirm the unweighted retrained control reproduces the champion.

    The control uses the champion's family, rows and seed with the standard
    objective, so it must match the champion's predictions exactly. This makes
    it a control on the effect of retraining and a reproducibility check on the
    frozen champion bundle.
    """
    try:
        control = next((item for item in bundle.candidates if item.name == RETRAINED_CONTROL), None)
        if control is None:
            raise RetentionTrainingError("the retrained control candidate is missing")
        champion = next(item for item in bundle.candidates if item.name == CHAMPION)
        validation = frames.validation
        control_predictions = predict(control.estimator, validation)
        champion_predictions = predict(champion.estimator, validation)
        if not np.array_equal(control_predictions, champion_predictions):
            maximum = float(np.abs(control_predictions - champion_predictions).max())
            raise RetentionTrainingError(
                "the unweighted retrained control does not reproduce the champion "
                f"(max absolute difference {maximum:.3e})"
            )
        return CheckResult(
            name="control_reproduces_champion",
            passed=True,
            detail=(
                "the unweighted retrained control reproduces the frozen champion's "
                "validation predictions exactly, so every later difference is "
                "attributable to the sample weighting"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="control_reproduces_champion",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Validation scoring and selection
# ---------------------------------------------------------------------------


def validation_slices(frames: TrainingFrames) -> dict[str, pd.DataFrame]:
    """Return the predeclared validation slices.

    The geometric slice is the aggregate over the primary rungs; each primary
    severity is also a slice of its own so a regression on one rung cannot be
    hidden by the aggregate. The clean and original-task slices come from the
    M3 rows. The overall population is a secondary summary.
    """
    validation = frames.validation
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


def _slice_metrics_table(
    frames: TrainingFrames,
    bundle: CandidateBundle,
    predictions: Mapping[str, np.ndarray],
    *,
    split_label: str,
) -> pd.DataFrame:
    """Score every candidate on every validation slice."""
    slices = validation_slices(frames)
    validation = frames.validation
    rows: list[dict[str, Any]] = []
    for candidate in bundle.candidates:
        predicted = predictions[candidate.name]
        for scope, subset in slices.items():
            if subset.empty:
                continue
            mask = validation.index.get_indexer(subset.index)
            scores = ra._slice_scores(subset, np.asarray(predicted)[mask])  # noqa: SLF001
            rows.append(
                {
                    "model": candidate.name,
                    "scheme": candidate.scheme,
                    "split": split_label,
                    "scope": scope,
                    **scores,
                }
            )
    return pd.DataFrame(rows)


def _severity_scopes(config: RetentionTrainingConfig) -> tuple[str, ...]:
    return tuple(
        f"{GEO_LABEL}_severity_{severity}" for severity in config.protocol.primary_severities
    )


def select_candidate(
    metrics: pd.DataFrame,
    settings: TrainingSelectionSettings,
    *,
    severity_scopes: Sequence[str],
) -> tuple[str, pd.DataFrame]:
    """Apply the predeclared eligibility constraints and selection score.

    The champion is the reference. A candidate qualifies only if it regresses
    neither the clean slice, nor the original-task slice, nor any single
    primary geometric severity beyond the declared tolerances. Among the
    qualifying candidates the score rewards aggregate geometric improvement and
    charges clean, original-task and per-severity regressions. Ties are broken
    by lower aggregate geometric RMSE and then by candidate name.
    """
    severity_scopes = tuple(severity_scopes)
    reference_name = CHAMPION
    reference_rows = metrics[metrics["model"] == reference_name]
    if reference_rows.empty:
        raise RetentionTrainingError("the champion reference is missing from the metrics table")

    def slice_rmse(model: str, scope: str) -> float | None:
        rows = metrics[(metrics["model"] == model) & (metrics["scope"] == scope)]
        if rows.empty:
            return None
        return float(rows.iloc[0]["rmse"])

    reference: dict[str, float] = {}
    for scope in (GEO_LABEL, CLEAN_LABEL, ORIGINAL_LABEL, *severity_scopes):
        value = slice_rmse(reference_name, scope)
        if value is None:
            raise RetentionTrainingError(f"the champion has no {scope!r} slice")
        reference[scope] = value

    records: list[dict[str, Any]] = []
    for model, group in metrics.groupby("model", sort=True):
        scheme = str(group.iloc[0]["scheme"])
        geo = slice_rmse(model, GEO_LABEL)
        clean = slice_rmse(model, CLEAN_LABEL)
        original = slice_rmse(model, ORIGINAL_LABEL)
        if geo is None or clean is None or original is None:
            raise RetentionTrainingError(f"candidate {model!r} is missing a required slice")
        severity_values = {scope: slice_rmse(model, scope) for scope in severity_scopes}
        missing = [scope for scope, value in severity_values.items() if value is None]
        if missing:
            raise RetentionTrainingError(
                f"candidate {model!r} is missing severity slice(s) {missing}"
            )

        clean_excess = clean - reference[CLEAN_LABEL]
        original_excess = original - reference[ORIGINAL_LABEL]
        severity_excess = {
            scope: severity_values[scope] - reference[scope] for scope in severity_scopes
        }

        eligible = True
        reasons: list[str] = []
        if clean > reference[CLEAN_LABEL] * (1.0 + settings.tolerance_clean):
            eligible = False
            reasons.append(
                f"clean RMSE {clean:.5f} exceeds the champion {reference[CLEAN_LABEL]:.5f} "
                f"by more than {settings.tolerance_clean:.0%}"
            )
        if original > reference[ORIGINAL_LABEL] * (1.0 + settings.tolerance_original):
            eligible = False
            reasons.append(
                f"original-task RMSE {original:.5f} exceeds the champion "
                f"{reference[ORIGINAL_LABEL]:.5f} by more than "
                f"{settings.tolerance_original:.0%}"
            )
        for scope, value in severity_values.items():
            if value > reference[scope] * (1.0 + settings.tolerance_severity):
                eligible = False
                reasons.append(
                    f"{scope} RMSE {value:.5f} exceeds the champion "
                    f"{reference[scope]:.5f} by more than "
                    f"{settings.tolerance_severity:.0%}"
                )

        score = (
            settings.w_geometric * (geo - reference[GEO_LABEL])
            + settings.w_clean * max(0.0, clean_excess)
            + settings.w_original * max(0.0, original_excess)
            + settings.w_severity * sum(max(0.0, value) for value in severity_excess.values())
        )
        records.append(
            {
                "model": model,
                "scheme": scheme,
                "eligible": eligible,
                "ineligible_reasons": "; ".join(reasons),
                "score": float(score),
                "rmse_geometric": geo,
                "rmse_clean": clean,
                "rmse_original": original,
                **{f"rmse_{scope}": severity_values[scope] for scope in severity_scopes},
                "delta_geometric_rmse": float(geo - reference[GEO_LABEL]),
                "delta_clean_rmse": float(clean_excess),
                "delta_original_rmse": float(original_excess),
                "max_severity_excess_rmse": float(
                    max((max(0.0, value) for value in severity_excess.values()), default=0.0)
                ),
            }
        )
    table = pd.DataFrame(records)
    eligible = table[table["eligible"]]
    if eligible.empty:
        # The fallback is the frozen champion, which is always eligible because
        # every delta is zero by construction.
        selected = reference_name
    else:
        ordered = eligible.sort_values(
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
    """One frozen evaluation slice with its own frame and per-candidate predictions."""

    scope: str
    frame: pd.DataFrame
    predictions: dict[str, np.ndarray]


def build_test_slices(frames: TrainingFrames, bundle: CandidateBundle) -> list[TestSlice]:
    """Assemble the frozen evaluation slices scored after selection is complete."""
    retention = frames.retention
    geometric = retention.primary_test
    m3_test = retention.m3_test
    original = m3_test[~m3_test["is_clean"]]

    def predict_on(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {candidate.name: candidate.predict(frame) for candidate in bundle.candidates}

    slices: list[TestSlice] = [
        TestSlice(GEO_LABEL, geometric, predict_on(geometric)),
        TestSlice(
            CLEAN_LABEL, m3_test[m3_test["is_clean"]], predict_on(m3_test[m3_test["is_clean"]])
        ),
        TestSlice(ORIGINAL_LABEL, original, predict_on(original)),
        TestSlice("m3_test_overall", m3_test, predict_on(m3_test)),
        TestSlice(
            "diagnostic_0_5", retention.diagnostic_test, predict_on(retention.diagnostic_test)
        ),
    ]
    for severity, subset in geometric.groupby("severity_level", sort=True):
        slices.append(
            TestSlice(f"{GEO_LABEL}_severity_{int(severity)}", subset, predict_on(subset))
        )
    for family, subset in original.groupby("family", sort=True):
        slices.append(TestSlice(f"{ORIGINAL_LABEL}_{family}", subset, predict_on(subset)))
    return [item for item in slices if not item.frame.empty]


def _test_metrics_table(slices: Sequence[TestSlice]) -> pd.DataFrame:
    """Score every candidate on every frozen test slice."""
    rows: list[dict[str, Any]] = []
    for item in slices:
        for name, predicted in item.predictions.items():
            scores = ra._slice_scores(item.frame, predicted)  # noqa: SLF001
            rows.append({"model": name, "split": "test", "scope": item.scope, **scores})
    return pd.DataFrame(rows)


def _uncertainty_table(slices: Sequence[TestSlice], *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """Source-grouped bootstrap intervals for every candidate and slice."""
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
                seed=derive_noise_seed(seed, "m5d_bootstrap", name, item.scope) % (2**64),
                evaluation="m5d_test",
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
    reference: str,
) -> pd.DataFrame:
    """Paired source-grouped differences against the champion."""
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
                seed=derive_noise_seed(seed, "m5d_paired", name, item.scope) % (2**64),
                evaluation="m5d_test",
            )
            rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _prediction_table(slices: Sequence[TestSlice], model_names: Sequence[str]) -> pd.DataFrame:
    """Row-level test predictions for the requested models."""
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
# Experiment execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The complete M5D outcome."""

    frames: TrainingFrames
    checks: tuple[CheckResult, ...]
    bundle: CandidateBundle
    validation_metrics: pd.DataFrame
    selection_table: pd.DataFrame
    selected_name: str
    test_metrics: pd.DataFrame
    uncertainty: pd.DataFrame
    paired_uncertainty: pd.DataFrame
    test_predictions: pd.DataFrame
    champion_card: dict[str, Any]
    weight_diagnostics: dict[str, dict[str, Any]]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def selected(self) -> TrainingCandidate:
        for candidate in self.bundle.candidates:
            if candidate.name == self.selected_name:
                return candidate
        raise RetentionTrainingError(f"selected candidate {self.selected_name!r} is missing")


def run_training_core(
    frames: TrainingFrames,
    config: RetentionTrainingConfig,
    *,
    champion: Any,
    m5c_config: ra.RetentionConfig,
) -> TrainingResult:
    """Run the M5D study in the predeclared order.

    The order is fixed: fit on training data only, score validation, select,
    freeze, and only then score the untouched test slices. The test slices are
    not read until :func:`select_candidate` has returned.
    """
    rb._verify_checks(frames.checks, "M5D frame checks")  # noqa: SLF001

    bundle = build_candidate_bundle(frames, config, champion=champion, m5c_config=m5c_config)
    control_check = check_control_reproduces_champion(bundle, frames)
    rb._verify_checks([control_check], "M5D control check")  # noqa: SLF001

    # Validation only. This is the evidence the selection rule consumes.
    validation_metrics = _slice_metrics_table(
        frames, bundle, bundle.validation_predictions, split_label="val"
    )
    selected_name, selection_table = select_candidate(
        validation_metrics,
        config.selection,
        severity_scopes=_severity_scopes(config),
    )

    # The selected candidate and its full configuration are frozen here. The
    # test slices are only touched after this point.
    slices = build_test_slices(frames, bundle)
    test_metrics = _test_metrics_table(slices)
    uncertainty = _uncertainty_table(
        slices, n_bootstrap=config.run.n_bootstrap, seed=config.run.seed
    )
    paired = _paired_uncertainty_table(
        slices,
        n_bootstrap=config.run.n_bootstrap,
        seed=config.run.seed,
        reference=CHAMPION,
    )
    predictions = _prediction_table(slices, bundle.names)

    return TrainingResult(
        frames=frames,
        checks=(*frames.checks, control_check),
        bundle=bundle,
        validation_metrics=validation_metrics,
        selection_table=selection_table,
        selected_name=selected_name,
        test_metrics=test_metrics,
        uncertainty=uncertainty,
        paired_uncertainty=paired,
        test_predictions=predictions,
        champion_card={},
        weight_diagnostics=bundle.weight_diagnostics,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _candidate_model_card(
    result: TrainingResult,
    config: RetentionTrainingConfig,
    candidate: TrainingCandidate,
) -> dict[str, Any]:
    """Build the model card for one candidate."""
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
    scheme = config.weighting.scheme(candidate.scheme) if candidate.scheme != SCHEME_NONE else None
    return {
        "model_name": candidate.name,
        "model_family": config.model.family,
        "weighting_scheme": candidate.scheme,
        "weighting_parameters": (
            {
                "clean_weight": scheme.clean_weight,
                "original_weight": scheme.original_weight,
                "geometric_base_weight": scheme.geometric_base_weight,
                "geometric_severity_ratio": scheme.geometric_severity_ratio,
            }
            if scheme is not None
            else None
        ),
        "weight_diagnostics": result.weight_diagnostics.get(candidate.name, {}),
        "params": {
            "n_estimators": config.model.n_estimators,
            "max_depth": config.model.max_depth,
            "min_samples_leaf": config.model.min_samples_leaf,
        },
        "seed": config.model.seed,
        "refitted": candidate.refitted,
        "feature_names": list(FEATURE_NAMES),
        "feature_set_version": config.schema.feature_set_version,
        "n_features": len(FEATURE_NAMES),
        "target": TARGET,
        "package_version": package_version(),
        "sklearn_version": sklearn_version(),
        "code_revision": code_revision(),
        "selection": {
            "rule": config.selection.rule,
            "selected_on": "val",
            "selected_model": result.selected_name,
            "n_candidates": len(result.bundle.candidates),
            "benchmark_config_hash": config.config_hash,
        },
        "validation_metrics": validation_payload,
        "metrics": metric_payload,
    }


def write_training_outputs(
    result: TrainingResult,
    config: RetentionTrainingConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    frozen_before: Mapping[str, Mapping[str, object]] | None = None,
    frozen_after: Mapping[str, Mapping[str, object]] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Path]:
    """Persist the M5D tables, model bundles, metadata and report."""
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
    paths.update(
        {
            "validation_metrics": reports / "validation_slice_metrics.csv",
            "selection_table": reports / "selection_table.csv",
            "test_metrics": reports / "test_slice_metrics.csv",
            "uncertainty": reports / "uncertainty.csv",
            "paired_uncertainty": reports / "paired_uncertainty.csv",
            "test_predictions": reports / "test_predictions.csv",
            "isolation_checks": reports / "isolation_checks.csv",
        }
    )

    for candidate in result.bundle.candidates:
        if not candidate.refitted:
            # The champion is the frozen M5C bundle; it is not rewritten here.
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
            "feature_names": list(FEATURE_NAMES),
            "feature_set_version": config.schema.feature_set_version,
            "target": TARGET,
            "split_names": list(config.schema.split_names),
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
            "train_sources": int(result.frames.train["source_id"].nunique()),
            "val_sources": int(result.frames.validation["source_id"].nunique()),
            "test_sources": int(result.frames.retention.primary_test["source_id"].nunique()),
            "train_samples": int(len(result.frames.train)),
            "validation_samples": int(len(result.frames.validation)),
            "primary_test_samples": int(len(result.frames.retention.primary_test)),
            "diagnostic_test_samples": int(len(result.frames.retention.diagnostic_test)),
            "m3_test_samples": int(len(result.frames.retention.m3_test)),
        },
        "weighting": {
            "groups": {
                group: int((result.frames.groups.astype(str) == group).sum())
                for group in sorted(set(result.frames.groups.astype(str)))
            },
            "diagnostics": result.weight_diagnostics,
        },
        "selection": {
            "rule": config.selection.rule,
            "selected_model": result.selected_name,
            "selected_scheme": result.selected.scheme,
            "n_candidates": len(result.bundle.candidates),
            "selected_on": "val",
        },
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
        "| model | n | sources | MAE | RMSE | R2 | mean residual | residual SD |",
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
        (table["model"] == f"{model}_minus_{CHAMPION}")
        & (table["dataset"] == scope)
        & (table["metric"] == metric)
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])


def _report_markdown(result: TrainingResult, config: RetentionTrainingConfig) -> str:
    """Render the deterministic M5D report."""
    models = list(result.bundle.names)
    selected = result.selected_name
    lines: list[str] = [
        "# M5D retention-aware training objectives",
        "",
        "**Scope:** synthetic image similarity, measured as SSIM against the exact clean "
        "reference. This is not a measure of human readability, CAPTCHA solvability, "
        "security, or real-world quality.",
        "",
        "## Question",
        "",
        "Can a retention-aware *training objective* improve geometric-distortion SSIM "
        "prediction relative to the M5C champion, without materially degrading "
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
        f"- Model family: `{config.model.family}` "
        f"(n_estimators={config.model.n_estimators}, max_depth={config.model.max_depth}, "
        f"min_samples_leaf={config.model.min_samples_leaf}, seed={config.model.seed})",
        f"- Feature schema: `{config.schema.feature_set_version}`, target `{TARGET}`, "
        f"{len(FEATURE_NAMES)} image-only features in their existing order",
        "",
        "The M3 dataset, split, frozen M3 model, M4A pilot, M4B benchmark, M5A pilot, "
        "M5B benchmark and M5C selection study are read-only and hash-verified before and "
        "after the run.",
        "",
        "### Severity protocol",
        "",
        f"- Primary geometric score: amplitudes `{list(config.protocol.primary_amplitudes_px)}` px "
        f"(severities `{list(config.protocol.primary_severities)}`).",
        f"- Diagnostic only: amplitude `{list(config.protocol.diagnostic_amplitudes_px)}` px "
        f"(severity `{list(config.protocol.diagnostic_severities)}`).",
        "- The 0.5 px rung never enters fitting, validation selection or the primary score.",
        "",
        "## Candidate training policies",
        "",
        "The model family, the hyperparameters, the training rows and the seed are held "
        "fixed across all candidates. Only the per-example sample weight changes.",
        "",
        "| candidate | weighting scheme | refitted |",
        "| --- | --- | --- |",
    ]
    for candidate in result.bundle.candidates:
        lines.append(
            f"| `{candidate.name}` | {candidate.scheme} | {'yes' if candidate.refitted else 'no'} |"
        )

    lines.extend(
        [
            "",
            "### Weighting formulas",
            "",
            "Every training example is assigned to exactly one group using metadata already "
            "present in the training frame, and the weight of an example is",
            "",
            "```text",
            "w(x) = group_weight[group(x)] * severity_factor[group(x), severity(x)]",
            "```",
            "",
            "where `severity_factor` is 1 for the clean and original groups and "
            "`geometric_severity_ratio ** (severity - min_primary_severity)` for the "
            "geometric group. The weights are rescaled so their mean over the training frame "
            "is exactly 1, and every weight must be finite and strictly positive. The "
            "weights are passed only to `fit(sample_weight=...)`; they never enter the "
            "predictor's feature matrix.",
            "",
            "| scheme | clean | original | geometric base | geometric severity ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in sorted(config.weighting.schemes):
        scheme = config.weighting.scheme(name)
        lines.append(
            f"| {name} | {scheme.clean_weight:g} | {scheme.original_weight:g} | "
            f"{scheme.geometric_base_weight:g} | {scheme.geometric_severity_ratio:g} |"
        )
    groups = result.frames.groups.astype(str)
    counts = {group: int((groups == group).sum()) for group in sorted(set(groups))}
    lines.extend(
        [
            "",
            "Training groups: "
            + ", ".join(f"{group}={count}" for group, count in counts.items())
            + f" over {len(result.frames.train)} training rows.",
            "",
            "### Realised training weights",
            "",
            "The per-row weight is what the estimator receives. The share is the "
            "fraction of the total training weight each group carries, which is what "
            "determines a group's effective influence on the fit: a small group with a "
            "high per-row weight can still carry a smaller share than a large group.",
            "",
            "| candidate | mean | min | max | clean mean / share | original mean / share | "
            "geometric mean / share |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in models:
        entry = result.weight_diagnostics.get(name, {})
        if not entry or entry.get("scheme") == SCHEME_NONE:
            lines.append(f"| `{name}` | — | — | — | — | — | — |")
            continue
        group_means = entry.get("group_weight_means", {})
        group_shares = entry.get("group_weight_shares", {})
        lines.append(
            f"| `{name}` | {entry['mean_weight']:.4f} | {entry['min_weight']:.4f} | "
            f"{entry['max_weight']:.4f} | "
            f"{group_means.get(GROUP_CLEAN, float('nan')):.4f} / "
            f"{group_shares.get(GROUP_CLEAN, float('nan')) * 100:.1f}% | "
            f"{group_means.get(GROUP_ORIGINAL, float('nan')):.4f} / "
            f"{group_shares.get(GROUP_ORIGINAL, float('nan')) * 100:.1f}% | "
            f"{group_means.get(GROUP_GEOMETRIC, float('nan')):.4f} / "
            f"{group_shares.get(GROUP_GEOMETRIC, float('nan')) * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Selection rule",
            "",
            "The rule was declared before any test evaluation. The champion is the "
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
            "RMSE and then by candidate name. If no candidate qualifies, the frozen champion "
            "is kept. No coefficient or tolerance was chosen by inspecting a test result.",
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
            f"Selected candidate: **`{selected}`** (scheme `{result.selected.scheme}`).",
            "",
            "## Final test results by slice",
            "",
            "The selected candidate and its complete configuration were frozen before these "
            "rows were scored. The M3 test split and the M5B/M5C geometric test split are "
            "untouched frozen evaluations.",
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
            "## Paired source-grouped differences against the champion",
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
            if model == CHAMPION:
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
            "- Sample weighting can only rebalance the existing training distribution; it "
            "cannot add information the eight image-only features do not contain.",
            "- The weighted candidates are fitted on the same 432 training rows as the "
            "champion, so they cannot recover accuracy that the shared feature "
            "representation or the shared sample size does not support.",
            "- The eight image-only features are global whole-image statistics. They "
            "describe overall intensity, contrast, ink coverage, sharpness, edge density, "
            "entropy and noise, but they do not describe *where* on the canvas the "
            "distortion occurs. A geometric warp moves ink locally while leaving many "
            "whole-image statistics nearly unchanged, which is consistent with the "
            "near-zero between-source R2 that M5B and M5C both observed. The results here "
            "are consistent with the feature representation, not the training objective, "
            "being the binding constraint, but this milestone does not isolate that cause.",
            "- The frozen M3 reference was itself selected on the M3 validation split, and "
            "its clean-reference validation slice is noisier than its test slice.",
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
            "uv run --extra image --extra ml captcha-retention-training run \\",
            "    --config configs/retention_aware_training_v0_1_0.toml",
            "uv run --extra image --extra ml captcha-retention-training verify \\",
            "    --config configs/retention_aware_training_v0_1_0.toml",
            "```",
            "",
            "Generated tables, the model bundles and this report live in Git-ignored paths.",
        ]
    )
    return "\n".join(lines) + "\n"


def _decision_lines(result: TrainingResult, config: RetentionTrainingConfig) -> list[str]:
    """Derive the descriptive interpretation from the observed results."""
    selected = result.selected_name
    lines: list[str] = []

    def metric(model: str, scope: str) -> dict[str, Any] | None:
        return _metric_row_for(result.test_metrics, model, scope)

    champion_geo = metric(CHAMPION, GEO_LABEL)
    selected_geo = metric(selected, GEO_LABEL)
    champion_clean = metric(CHAMPION, CLEAN_LABEL)
    selected_clean = metric(selected, CLEAN_LABEL)
    champion_original = metric(CHAMPION, ORIGINAL_LABEL)
    selected_original = metric(selected, ORIGINAL_LABEL)
    champion_m3 = metric(CHAMPION, "m3_test_overall")
    selected_m3 = metric(selected, "m3_test_overall")

    if champion_geo and selected_geo:
        change = (selected_geo["rmse"] - champion_geo["rmse"]) / champion_geo["rmse"] * 100
        lines.append(
            f"- **Primary geometric slice.** The M5C champion reaches MAE "
            f"{champion_geo['mae']:.4f}, RMSE {champion_geo['rmse']:.4f} and R2 "
            f"{champion_geo['r2']:.4f} on {champion_geo['n_samples']} rows from "
            f"{champion_geo['n_sources']} sources. The selected candidate `{selected}` "
            f"reaches MAE {selected_geo['mae']:.4f}, RMSE {selected_geo['rmse']:.4f} and R2 "
            f"{selected_geo['r2']:.4f}, a {change:+.1f}% change in RMSE against the "
            "champion."
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

    geometric_improved = bool(
        selected_geo and champion_geo and selected_geo["rmse"] < champion_geo["rmse"]
    )
    if selected == CHAMPION:
        lines.append(
            "- **Outcome.** No retrained candidate was selected. The frozen M5C champion is "
            "retained. This is reported as a negative result rather than a manufactured "
            "winner."
        )
        # Explain each rejected candidate individually.
        for row in result.selection_table.itertuples(index=False):
            if row.model == CHAMPION or bool(row.eligible):
                continue
            lines.append(
                f"- **`{row.model}` was ineligible.** {row.ineligible_reasons}. The "
                "per-severity constraint is what rejects it: the aggregate geometric score "
                "alone would not have revealed the regression on a single rung."
            )
        eligible_losers = result.selection_table[
            result.selection_table["eligible"] & ~result.selection_table["selected"]
        ]
        for row in eligible_losers.itertuples(index=False):
            lines.append(
                f"- **`{row.model}` was eligible but not selected.** Its aggregate geometric "
                f"validation RMSE is {row.rmse_geometric:.5f} against the champion's "
                f"{result.selection_table[result.selection_table['model'] == CHAMPION].iloc[0]['rmse_geometric']:.5f} "
                f"(delta {row.delta_geometric_rmse:+.5f}), the clean slice moves by "
                f"{row.delta_clean_rmse:+.5f} and the original task by "
                f"{row.delta_original_rmse:+.5f}, giving a score of {row.score:+.5f} against "
                "the champion's 0. Under the predeclared rule a candidate must improve the "
                "aggregate geometric RMSE by more than it costs on the retention slices, and "
                "this one does not."
            )
        lines.append(
            "- **Why weighting did not help.** Sample weighting can only rebalance the "
            "existing 432 training rows. Every candidate shares the champion's model "
            "family, its hyperparameters, its seed and its training rows, so the only "
            "lever is how much each row counts. Reweighting the clean and original-task "
            "rows more heavily protects them but leaves less capacity for the geometric "
            "rungs, and reweighting the geometric rungs more heavily costs accuracy on the "
            "rungs that were not up-weighted. With a single model family and a single "
            "shared representation, the weighting moves error between slices rather than "
            "removing it."
        )
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark champion, "
            "and the M5C champion remains the best geometric-robustness policy found so "
            "far. Changing the training objective within the existing feature "
            "representation and sample size did not help. The evidence does not justify a "
            "CNN, and a new corruption family does not by itself justify one."
        )
    else:
        lines.append(
            "- **Outcome.** A weighted training objective was selected: "
            + (
                "it improves the aggregate geometric slice while staying inside the "
                "predeclared clean, original-task and per-severity tolerances."
                if geometric_improved
                else "it was selected on the aggregate geometric score, but the geometric "
                "improvement itself is small or absent."
            )
        )
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark champion, "
            f"now with the retention-aware training objective `{result.selected.scheme}` "
            f"(`{selected}`) rather than the M5C champion alone. The evidence does not "
            "justify a CNN, and a new corruption family does not by itself justify one."
        )
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_retention_training(
    config: RetentionTrainingConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    verify_images: bool = True,
) -> tuple[TrainingResult, dict[str, Path]]:
    """Load, verify, run and persist the complete M5D study."""
    started = time.perf_counter()
    before = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    frames = load_training_frames(config, verify_images=verify_images)
    m5c_config = ra.load_retention_config(config.m5c.config)
    champion = _load_champion(config, frames.retention.frames.m3_summary)

    result = run_training_core(frames, config, champion=champion, m5c_config=m5c_config)
    after = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    if not _frozen_unchanged(before, after):
        raise RetentionTrainingError("a frozen asset changed during the M5D run")
    paths = write_training_outputs(
        result,
        config,
        output_reports=output_reports,
        output_models=output_models,
        frozen_before=before,
        frozen_after=after,
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, paths


def verify_retention_training(
    config: RetentionTrainingConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
) -> dict[str, Any]:
    """Verify a written M5D run: hashes, protocol, selection and prediction replay."""
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
            raise RetentionTrainingError("the M5C protocol hash does not match the pin")
        _require_file_hash(
            config.m5c.champion_dir / f"{config.m5c.champion_stem}.joblib",
            config.m5c.champion_artifact_sha256,
            "M5C champion artifact",
        )
        frames = load_training_frames(config, verify_images=False)
        return (
            f"{len(frames.train)} training rows, {len(frames.validation)} validation rows, "
            f"{len(frames.retention.primary_test)} primary geometric test rows and "
            f"{len(frames.retention.diagnostic_test)} diagnostic test rows"
        )

    def metadata_checks() -> str:
        metadata_path = reports / "run_metadata.json"
        if not metadata_path.is_file():
            raise RetentionTrainingError(f"missing run metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("benchmark_config_sha256") != config.config_sha256:
            raise RetentionTrainingError("run metadata config hash does not match")
        if metadata.get("benchmark_config_hash") != config.config_hash:
            raise RetentionTrainingError("run metadata protocol hash does not match")
        if not metadata.get("frozen_assets", {}).get("all_unchanged"):
            raise RetentionTrainingError("run metadata reports a changed frozen asset")
        for name, entry in metadata.get("artifacts", {}).items():
            path = Path(entry["path"]) if isinstance(entry, dict) else reports / name
            if not path.is_file():
                raise RetentionTrainingError(f"missing recorded artifact: {path}")
            if sha256_file(path) != (entry["sha256"] if isinstance(entry, dict) else entry):
                raise RetentionTrainingError(f"artifact hash mismatch: {name}")
        return f"{len(metadata.get('artifacts', {}))} artifact hashes match"

    def selection_checks() -> str:
        table_path = reports / "selection_table.csv"
        if not table_path.is_file():
            raise RetentionTrainingError(f"missing selection table: {table_path}")
        table = pd.read_csv(table_path)
        selected = table[table["selected"]]
        if len(selected) != 1:
            raise RetentionTrainingError("the selection table must mark exactly one winner")
        metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
        recorded = metadata.get("selection", {}).get("selected_model")
        if recorded != str(selected.iloc[0]["model"]):
            raise RetentionTrainingError(
                f"selection table winner {selected.iloc[0]['model']!r} does not match the "
                f"recorded selection {recorded!r}"
            )
        return f"selected candidate {recorded!r} is consistent"

    def model_checks() -> str:
        for candidate in config.candidate_names:
            if candidate == CHAMPION:
                continue
            card_path = models / f"{candidate}_card.json"
            artifact = models / f"{candidate}.joblib"
            if not card_path.is_file() or not artifact.is_file():
                raise RetentionTrainingError(f"missing model bundle for {candidate!r}")
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("artifact_sha256") != sha256_file(artifact):
                raise RetentionTrainingError(f"artifact hash mismatch for {candidate!r}")
            if card.get("feature_names") != list(FEATURE_NAMES):
                raise RetentionTrainingError(f"feature schema mismatch for {candidate!r}")
            if card.get("target") != TARGET:
                raise RetentionTrainingError(f"target mismatch for {candidate!r}")
        return "every refitted model bundle loads, matches its card and uses the frozen schema"

    def replay_checks() -> str:
        frames = load_training_frames(config, verify_images=False)
        m5c_config = ra.load_retention_config(config.m5c.config)
        champion = _load_champion(config, frames.retention.frames.m3_summary)
        bundle = build_candidate_bundle(frames, config, champion=champion, m5c_config=m5c_config)
        slices = build_test_slices(frames, bundle)
        replayed = _test_metrics_table(slices)
        stored = pd.read_csv(reports / "test_slice_metrics.csv")
        merged = replayed.merge(stored, on=["model", "scope"], suffixes=("_replay", "_stored"))
        if len(merged) != len(replayed):
            raise RetentionTrainingError("replayed metrics do not cover the stored rows")
        for column in ("mae", "rmse"):
            if not np.allclose(
                merged[f"{column}_replay"], merged[f"{column}_stored"], rtol=0.0, atol=1e-9
            ):
                raise RetentionTrainingError(f"replayed {column} does not match the stored run")
        return f"{len(replayed)} slice metrics reproduce exactly"

    def report_checks() -> str:
        report_path = reports / "report.md"
        if not report_path.is_file():
            raise RetentionTrainingError(f"missing report: {report_path}")
        text = report_path.read_text(encoding="utf-8")
        required = (
            "Candidate training policies",
            "Weighting formulas",
            "Selection rule",
            "Validation results by slice",
            "Final test results by slice",
            "Paired source-grouped differences against the champion",
            "Reproducibility",
        )
        missing = [section for section in required if section not in text]
        if missing:
            raise RetentionTrainingError(f"report is missing sections: {missing}")
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
        prog="captcha-retention-training",
        description="Run or verify the grouped M5D retention-aware training study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the retention-aware training study")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path, default=None)
    run_parser.add_argument("--output-models", type=Path, default=None)

    verify_parser = subparsers.add_parser(
        "verify", help="verify a written retention-aware training run"
    )
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path, default=None)
    verify_parser.add_argument("--output-models", type=Path, default=None)
    return parser


def _print_report(report: Mapping[str, Any]) -> None:
    for check in report["checks"]:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"retention-aware training verification: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_retention_training_config(args.config)
        if args.command == "run":
            result, paths = run_retention_training(
                config,
                output_reports=args.output_reports,
                output_models=args.output_models,
            )
            print(f"benchmark: {config.describe()}")
            print(f"protocol hash: {config.config_hash}")
            print(
                "rows: train "
                f"{len(result.frames.train)}, validation "
                f"{len(result.frames.validation)}, primary test "
                f"{len(result.frames.retention.primary_test)}, diagnostic 0.5 px "
                f"{len(result.frames.retention.diagnostic_test)}"
            )
            print(f"selected candidate: {result.selected_name}")
            for scope in (GEO_LABEL, CLEAN_LABEL, ORIGINAL_LABEL, "m3_test_overall"):
                row = _metric_row_for(result.test_metrics, result.selected_name, scope)
                champion = _metric_row_for(result.test_metrics, CHAMPION, scope)
                if row is None or champion is None:
                    continue
                print(
                    f"  {scope}: selected MAE {row['mae']:.4f} RMSE {row['rmse']:.4f} | "
                    f"champion MAE {champion['mae']:.4f} RMSE {champion['rmse']:.4f}"
                )
            print(f"report: {paths['report']}")
            return 0
        report = verify_retention_training(
            config,
            output_reports=args.output_reports,
            output_models=args.output_models,
        )
        _print_report(report)
        return 0 if report["passed"] else 1
    except (
        RetentionTrainingError,
        RetentionTrainingConfigError,
        ra.RetentionSelectionError,
        ra.RetentionSelectionConfigError,
        ModelError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
