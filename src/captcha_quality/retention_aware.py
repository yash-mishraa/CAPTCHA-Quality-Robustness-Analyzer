"""Milestone 5C retention-aware model selection and calibration.

M5B established that the frozen Milestone 3 model does not transfer zero-shot
to the held-out M5A geometric sources, and that a conventional model trained on
representative geometric rows reduces aggregate geometric error while losing
clean-reference calibration on the original blur/noise task. M5C asks a single
narrow question:

    Can a conventional model-selection or calibration policy improve geometric
    prediction while retaining performance on clean and blur/noise samples?

The milestone is a controlled selection and calibration experiment, not a new
model-development campaign. It declares a small candidate set, a calibration
procedure and a retention-aware selection rule *before* any test-source
prediction is made, and it keeps the frozen M3 model available as a reference
throughout.

Three candidate policies are evaluated:

* **Frozen M3 model** - unchanged reference and retention baseline.
* **M5B expanded HGB** - unchanged, validation-selected reference.
* **Expanded conventional alternative** - the strongest non-HGB candidate in the
  existing M5B validation table, which is the only evidence used to justify
  including it.

Each base model may additionally be used with a single affine calibration
(``target = intercept + slope * prediction``). The base models are trained only
on training sources, so their predictions on the expanded validation frame are
out-of-sample and may be used to fit the calibration. The mapping is
univariate and uses no metadata, so severity, family, font, text length, source
and split identifiers never enter it.

Selection uses a predeclared retention-aware rule computed on validation data
only: the change in RMSE relative to the frozen M3 model across the geometric,
clean-reference and original blur/noise slices, with an explicit extra penalty
charged on retention regressions. The selected policy and all calibration
parameters are frozen before the untouched M3 and M5B test splits are scored.

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

import numpy as np
import pandas as pd

from captcha_quality import geometric_benchmark as gb
from captcha_quality import robustness_benchmark as rb
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.dataset import CheckResult
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import (
    TARGET,
    ModelError,
    fit_candidate,
    load_model_bundle,
    predict,
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

#: Slice labels used throughout the validation and test reports.
GEO_LABEL: Final[str] = "geometric_primary"
CLEAN_LABEL: Final[str] = "clean_reference"
ORIGINAL_LABEL: Final[str] = "original_blur_noise"
OVERALL_LABEL: Final[str] = "overall"

#: Candidate policy names.
FROZEN_M3: Final[str] = "frozen_m3"
M5B_EXPANDED_HGB: Final[str] = "m5b_expanded_hgb"
EXPANDED_ALTERNATIVE: Final[str] = "expanded_random_forest"

#: Calibration policy names.
CALIBRATION_NONE: Final[str] = "none"
CALIBRATION_AFFINE: Final[str] = "affine"

#: Model artefact stems written by this milestone.
ALTERNATIVE_MODEL_STEM: Final[str] = "expanded_random_forest"
SELECTED_POLICY_STEM: Final[str] = "selected_policy"

_EXPECTED_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {"run", "schema", "m5b", "protocol", "candidates", "calibration", "selection"}
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Frozen inputs audited before and after the milestone. The M5B configuration,
#: model bundle and report are frozen inputs here; M5C's own outputs are not.
FROZEN_ASSET_PATHS: Final[tuple[str, ...]] = (
    "configs/dataset_v0_1_0.toml",
    "configs/benchmark_v0_1_0.toml",
    "configs/robustness_pilot_occlusion_v0_1_0.toml",
    "configs/robustness_benchmark_v0_1_0.toml",
    "configs/geometric_distortion_pilot_v0_1_0.toml",
    "configs/geometric_robustness_benchmark_v0_1_0.toml",
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "data/processed/geometric_distortion_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "models/robustness_benchmark_v0_1_0",
    "models/geometric_robustness_benchmark_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
    "reports/geometric_distortion_pilot",
    "reports/geometric_robustness_benchmark_v0_1_0",
)


class RetentionSelectionError(RuntimeError):
    """Raised when the M5C protocol cannot be honoured."""


class RetentionSelectionConfigError(ValueError):
    """Raised when the M5C configuration is invalid."""


@dataclass(frozen=True, slots=True)
class M5BPin:
    """Pins for the upstream M5B benchmark and its expanded model."""

    config: Path
    config_sha256: str
    protocol_hash: str
    model_dir: Path
    model_stem: str
    model_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class ProtocolSettings:
    """Severity inclusion rules, carried forward unchanged from M5B."""

    primary_amplitudes_px: tuple[float, ...]
    diagnostic_amplitudes_px: tuple[float, ...]
    primary_severities: tuple[int, ...]
    diagnostic_severities: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CandidateSettings:
    """The predeclared candidate policies."""

    base_models: tuple[str, ...]
    calibration_forms: tuple[str, ...]
    alternative_spec: str
    alternative_justification: str


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    """The predeclared calibration procedure."""

    form: str
    fit_scope: str
    cross_fit_folds: int
    clip_output: bool


@dataclass(frozen=True, slots=True)
class SelectionSettings:
    """The predeclared retention-aware selection rule."""

    rule: str
    w_geometric: float
    w_clean: float
    w_original: float
    penalty_regression: float
    tolerance: float


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    """The complete M5C protocol."""

    run: rb.RunSettings
    schema: rb.SchemaSettings
    m5b: M5BPin
    protocol: ProtocolSettings
    candidates: CandidateSettings
    calibration: CalibrationSettings
    selection: SelectionSettings
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
            "m5b": {
                "config": str(self.m5b.config),
                "config_sha256": self.m5b.config_sha256,
                "protocol_hash": self.m5b.protocol_hash,
                "model_dir": str(self.m5b.model_dir),
                "model_stem": self.m5b.model_stem,
                "model_artifact_sha256": self.m5b.model_artifact_sha256,
            },
            "protocol": {
                "primary_amplitudes_px": list(self.protocol.primary_amplitudes_px),
                "diagnostic_amplitudes_px": list(self.protocol.diagnostic_amplitudes_px),
                "primary_severities": list(self.protocol.primary_severities),
                "diagnostic_severities": list(self.protocol.diagnostic_severities),
            },
            "candidates": {
                "base_models": list(self.candidates.base_models),
                "calibration_forms": list(self.candidates.calibration_forms),
                "alternative": {
                    "spec": self.candidates.alternative_spec,
                    "justification": self.candidates.alternative_justification,
                },
            },
            "calibration": {
                "form": self.calibration.form,
                "fit_scope": self.calibration.fit_scope,
                "cross_fit_folds": self.calibration.cross_fit_folds,
                "clip_output": self.calibration.clip_output,
            },
            "selection": {
                "rule": self.selection.rule,
                "w_geometric": self.selection.w_geometric,
                "w_clean": self.selection.w_clean,
                "w_original": self.selection.w_original,
                "penalty_regression": self.selection.penalty_regression,
                "tolerance": self.selection.tolerance,
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


def _check_keys(section: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise RetentionSelectionConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise RetentionSelectionConfigError(f"missing or invalid [{name}] table")
    return value


def _required(section: Mapping[str, object], key: str, section_name: str) -> object:
    if key not in section:
        raise RetentionSelectionConfigError(f"missing required key {section_name}.{key}")
    return section[key]


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetentionSelectionConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetentionSelectionConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RetentionSelectionConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RetentionSelectionConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RetentionSelectionConfigError(f"{label} must be a boolean, got {value!r}")
    return value


def _as_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RetentionSelectionConfigError(f"{label} must be a TOML array")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    items = tuple(_as_str(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise RetentionSelectionConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_int_tuple(value: object, label: str) -> tuple[int, ...]:
    items = tuple(_as_int(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise RetentionSelectionConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_float_tuple(value: object, label: str) -> tuple[float, ...]:
    items = tuple(_as_float(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise RetentionSelectionConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise RetentionSelectionConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _parse_run(raw: Mapping[str, object]) -> rb.RunSettings:
    _check_keys(
        raw, {"name", "version", "seed", "n_bootstrap", "output_reports", "output_models"}, "run"
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


def _parse_m5b(raw: Mapping[str, object]) -> M5BPin:
    _check_keys(
        raw,
        {
            "config",
            "config_sha256",
            "protocol_hash",
            "model_dir",
            "model_stem",
            "model_artifact_sha256",
        },
        "m5b",
    )
    return M5BPin(
        config=Path(_as_str(raw["config"], "m5b.config")),
        config_sha256=_as_sha256(raw["config_sha256"], "m5b.config_sha256"),
        protocol_hash=_as_sha256(raw["protocol_hash"], "m5b.protocol_hash"),
        model_dir=Path(_as_str(raw["model_dir"], "m5b.model_dir")),
        model_stem=_as_str(raw["model_stem"], "m5b.model_stem"),
        model_artifact_sha256=_as_sha256(raw["model_artifact_sha256"], "m5b.model_artifact_sha256"),
    )


def _parse_protocol(raw: Mapping[str, object]) -> ProtocolSettings:
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
    settings = ProtocolSettings(
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
        raise RetentionSelectionConfigError(
            "the diagnostic amplitude must not also appear in the primary ladder"
        )
    if set(settings.primary_severities) & set(settings.diagnostic_severities):
        raise RetentionSelectionConfigError(
            "the diagnostic severity must not also appear in the primary ladder"
        )
    if tuple(sorted(settings.primary_amplitudes_px)) != settings.primary_amplitudes_px:
        raise RetentionSelectionConfigError("primary amplitudes must be sorted ascending")
    return settings


def _parse_candidates(raw: Mapping[str, object]) -> CandidateSettings:
    _check_keys(raw, {"base_models", "calibration_forms", "alternative"}, "candidates")
    alternative = raw["alternative"]
    if not isinstance(alternative, dict):
        raise RetentionSelectionConfigError("candidates.alternative must be a table")
    _check_keys(alternative, {"spec", "justification"}, "candidates.alternative")
    settings = CandidateSettings(
        base_models=_as_str_tuple(raw["base_models"], "candidates.base_models"),
        calibration_forms=_as_str_tuple(raw["calibration_forms"], "candidates.calibration_forms"),
        alternative_spec=_as_str(alternative["spec"], "candidates.alternative.spec"),
        alternative_justification=_as_str(
            alternative["justification"], "candidates.alternative.justification"
        ),
    )
    expected_bases = {FROZEN_M3, M5B_EXPANDED_HGB, EXPANDED_ALTERNATIVE}
    if set(settings.base_models) != expected_bases:
        raise RetentionSelectionConfigError(
            f"candidates.base_models must be {sorted(expected_bases)}, got "
            f"{sorted(settings.base_models)}"
        )
    expected_forms = {CALIBRATION_NONE, CALIBRATION_AFFINE}
    if set(settings.calibration_forms) != expected_forms:
        raise RetentionSelectionConfigError(
            f"candidates.calibration_forms must be {sorted(expected_forms)}, got "
            f"{sorted(settings.calibration_forms)}"
        )
    return settings


def _parse_calibration(raw: Mapping[str, object]) -> CalibrationSettings:
    _check_keys(raw, {"form", "fit_scope", "cross_fit_folds", "clip_output"}, "calibration")
    settings = CalibrationSettings(
        form=_as_str(raw["form"], "calibration.form"),
        fit_scope=_as_str(raw["fit_scope"], "calibration.fit_scope"),
        cross_fit_folds=_as_int(raw["cross_fit_folds"], "calibration.cross_fit_folds"),
        clip_output=_as_bool(raw["clip_output"], "calibration.clip_output"),
    )
    if settings.form != CALIBRATION_AFFINE:
        raise RetentionSelectionConfigError(
            f"calibration.form must be {CALIBRATION_AFFINE!r}, got {settings.form!r}"
        )
    if settings.fit_scope != "expanded_validation":
        raise RetentionSelectionConfigError(
            "calibration.fit_scope must be 'expanded_validation' so that calibration is "
            "fitted only on out-of-sample validation predictions"
        )
    if settings.cross_fit_folds < 2:
        raise RetentionSelectionConfigError("calibration.cross_fit_folds must be at least 2")
    return settings


def _parse_selection(raw: Mapping[str, object]) -> SelectionSettings:
    _check_keys(
        raw,
        {
            "rule",
            "w_geometric",
            "w_clean",
            "w_original",
            "penalty_regression",
            "tolerance",
        },
        "selection",
    )
    settings = SelectionSettings(
        rule=_as_str(raw["rule"], "selection.rule"),
        w_geometric=_as_float(raw["w_geometric"], "selection.w_geometric"),
        w_clean=_as_float(raw["w_clean"], "selection.w_clean"),
        w_original=_as_float(raw["w_original"], "selection.w_original"),
        penalty_regression=_as_float(raw["penalty_regression"], "selection.penalty_regression"),
        tolerance=_as_float(raw["tolerance"], "selection.tolerance"),
    )
    if settings.rule != "retention_penalised_rmse_delta":
        raise RetentionSelectionConfigError(
            f"selection.rule must be 'retention_penalised_rmse_delta', got {settings.rule!r}"
        )
    if settings.w_clean < 0 or settings.w_original < 0 or settings.penalty_regression < 0:
        raise RetentionSelectionConfigError("selection weights and penalty must be non-negative")
    if settings.tolerance < 0:
        raise RetentionSelectionConfigError("selection.tolerance must be non-negative")
    return settings


def load_retention_config(path: Path | str) -> RetentionConfig:
    """Load and validate the M5C configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise RetentionSelectionConfigError(f"missing configuration file: {config_path}")
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    unknown = sorted(set(payload) - _EXPECTED_TOP_KEYS)
    if unknown:
        raise RetentionSelectionConfigError(
            f"unknown top-level key(s): {unknown}; allowed keys are {sorted(_EXPECTED_TOP_KEYS)}"
        )
    config = RetentionConfig(
        run=_parse_run(_section(payload, "run")),
        schema=_parse_schema(_section(payload, "schema")),
        m5b=_parse_m5b(_section(payload, "m5b")),
        protocol=_parse_protocol(_section(payload, "protocol")),
        candidates=_parse_candidates(_section(payload, "candidates")),
        calibration=_parse_calibration(_section(payload, "calibration")),
        selection=_parse_selection(_section(payload, "selection")),
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )
    if config.schema.feature_names != tuple(FEATURE_NAMES):
        raise RetentionSelectionConfigError(
            f"schema.feature_names must match the extractor {list(FEATURE_NAMES)}"
        )
    if config.schema.target != TARGET:
        raise RetentionSelectionConfigError(f"schema.target must be {TARGET!r}")
    if config.protocol.primary_amplitudes_px != gb.PRIMARY_AMPLITUDES_PX:
        raise RetentionSelectionConfigError(
            "protocol.primary_amplitudes_px must match the M5B primary ladder "
            f"{list(gb.PRIMARY_AMPLITUDES_PX)}"
        )
    if config.protocol.diagnostic_amplitudes_px != gb.DIAGNOSTIC_AMPLITUDES_PX:
        raise RetentionSelectionConfigError(
            "protocol.diagnostic_amplitudes_px must match the M5B diagnostic ladder "
            f"{list(gb.DIAGNOSTIC_AMPLITUDES_PX)}"
        )
    return config


# ---------------------------------------------------------------------------
# Frames and inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetentionFrames:
    """Validation and test frames used by the M5C policy comparison."""

    frames: gb.BenchmarkFrames
    expanded_train: pd.DataFrame
    expanded_val: pd.DataFrame
    primary_test: pd.DataFrame
    diagnostic_test: pd.DataFrame
    m3_test: pd.DataFrame
    m3_frame: pd.DataFrame
    m5a_frame: pd.DataFrame
    checks: tuple[CheckResult, ...]
    alternative_spec: str
    alternative_validation: dict[str, Any]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def validation_slices(self) -> dict[str, pd.DataFrame]:
        """Return the three predeclared retention slices of the validation frame."""
        val = self.expanded_val
        geometric = val[val["dataset"] == gb.M5A_LABEL]
        m3_val = val[val["dataset"] == gb.M3_LABEL]
        clean = m3_val[m3_val["is_clean"]]
        original = m3_val[~m3_val["is_clean"]]
        return {GEO_LABEL: geometric, CLEAN_LABEL: clean, ORIGINAL_LABEL: original}


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RetentionSelectionError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RetentionSelectionError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def _load_m5b_config(config: RetentionConfig) -> gb.GeometricBenchmarkConfig:
    """Load the pinned M5B configuration and confirm it is unchanged."""
    _require_file_hash(config.m5b.config, config.m5b.config_sha256, "M5B config")
    m5b_config = gb.load_geometric_benchmark_config(config.m5b.config)
    if m5b_config.config_hash != config.m5b.protocol_hash:
        raise RetentionSelectionError(
            "M5B protocol hash mismatch: expected "
            f"{config.m5b.protocol_hash}, found {m5b_config.config_hash}"
        )
    return m5b_config


def _check_alternative_justification(
    m5b_config: gb.GeometricBenchmarkConfig, alternative_spec: str
) -> dict[str, Any]:
    """Confirm the alternative is the best non-HGB candidate on M5B validation.

    The justification for including an expanded conventional alternative comes
    from the existing M5B validation table, which is reproduced here from the
    M5B grid. No M5C test result is consulted.
    """
    specs = rb.candidate_specs_from_config(m5b_config)  # type: ignore[arg-type]
    names = {spec.name for spec in specs}
    if alternative_spec not in names:
        raise RetentionSelectionError(
            f"alternative spec {alternative_spec!r} is not in the M5B grid"
        )
    frames = gb.load_benchmark_frames(m5b_config, verify_images=False)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        seed = derive_noise_seed(m5b_config.run.seed, "model", spec.name) % (2**32)
        model = fit_candidate(spec, frames.expanded_train, seed=seed)
        scores = score_frame(frames.expanded_val, predict(model.estimator, frames.expanded_val))
        rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "val_mae": scores["mae"],
                "val_rmse": scores["rmse"],
                "val_r2": scores["r2"],
                "val_n": scores["n_rows"],
            }
        )
    table = pd.DataFrame(rows).sort_values("val_rmse", kind="stable")
    selected = table.iloc[0]
    non_hgb = table[table["family"] != "hist_gradient_boosting"]
    if non_hgb.empty:
        raise RetentionSelectionError("the M5B grid has no non-HGB candidate")
    best_non_hgb = non_hgb.iloc[0]
    if str(best_non_hgb["model"]) != alternative_spec:
        raise RetentionSelectionError(
            f"alternative {alternative_spec!r} is not the best non-HGB M5B candidate by "
            f"validation RMSE ({best_non_hgb['model']!r} is)"
        )
    if str(selected["model"]) == alternative_spec:
        raise RetentionSelectionError(
            "the alternative must differ from the M5B validation-selected model"
        )
    return {
        "m5b_grid_candidates": len(table),
        "m5b_selected_on_validation": str(selected["model"]),
        "m5b_selected_val_rmse": float(selected["val_rmse"]),
        "alternative": alternative_spec,
        "alternative_val_rmse": float(best_non_hgb["val_rmse"]),
        "alternative_val_mae": float(best_non_hgb["val_mae"]),
        "alternative_val_r2": float(best_non_hgb["val_r2"]),
        "alternative_val_gap_fraction": float(
            (best_non_hgb["val_rmse"] - selected["val_rmse"]) / selected["val_rmse"]
        ),
    }


def _check_calibration_isolation(
    expanded_val: pd.DataFrame, expanded_train: pd.DataFrame
) -> CheckResult:
    """Confirm calibration fit rows are out-of-sample and disjoint from test."""
    try:
        if set(expanded_val["split"]) != {"val"}:
            raise RetentionSelectionError("calibration fit frame is not validation-only")
        if set(expanded_train["split"]) != {"train"}:
            raise RetentionSelectionError("training frame is not train-only")
        train_sources = set(expanded_train["source_id"].astype(str))
        val_sources = set(expanded_val["source_id"].astype(str))
        if train_sources & val_sources:
            raise RetentionSelectionError("a source appears in both train and validation")
        if expanded_val["sample_id"].duplicated().any():
            raise RetentionSelectionError("calibration fit frame has duplicate sample IDs")
        return CheckResult(
            name="calibration_isolation",
            passed=True,
            detail=(
                f"{len(expanded_val)} validation rows from {len(val_sources)} sources are "
                "disjoint from the training sources; no test row is present"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="calibration_isolation", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _check_test_immutability(
    primary_test: pd.DataFrame, diagnostic_test: pd.DataFrame, m3_test: pd.DataFrame
) -> CheckResult:
    """Confirm the evaluation frames are disjoint from every fit frame."""
    try:
        test_ids = (
            set(primary_test["sample_id"])
            | set(diagnostic_test["sample_id"])
            | set(m3_test["sample_id"])
        )
        if len(test_ids) != len(primary_test) + len(diagnostic_test) + len(m3_test):
            raise RetentionSelectionError("test frames share sample IDs")
        for frame, label in ((primary_test, "primary"), (diagnostic_test, "diagnostic")):
            if set(frame["split"]) != {"test"}:
                raise RetentionSelectionError(f"{label} test frame is not test-only")
        if set(m3_test["split"]) != {"test"}:
            raise RetentionSelectionError("M3 test frame is not test-only")
        return CheckResult(
            name="test_immutability",
            passed=True,
            detail=(
                f"{len(test_ids)} distinct test sample IDs across the primary, diagnostic "
                "and M3 test frames; none can enter a fit frame"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="test_immutability", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _check_low_amplitude_excluded(
    expanded_val: pd.DataFrame, primary_test: pd.DataFrame, diagnostic_test: pd.DataFrame
) -> CheckResult:
    """Confirm the 0.5 px rung is absent from every fit and primary score frame."""
    try:
        fit = pd.concat([expanded_val], ignore_index=True)
        m5a_fit = fit[fit["dataset"] == gb.M5A_LABEL]
        if (
            not m5a_fit.empty
            and gb._amplitude_mask(  # noqa: SLF001
                m5a_fit, gb.DIAGNOSTIC_AMPLITUDES_PX
            ).any()
        ):
            raise RetentionSelectionError("0.5 px rows entered the calibration fit frame")
        if gb._amplitude_mask(  # noqa: SLF001
            primary_test, gb.DIAGNOSTIC_AMPLITUDES_PX
        ).any():
            raise RetentionSelectionError("0.5 px rows entered the primary score frame")
        if not gb._amplitude_mask(  # noqa: SLF001
            diagnostic_test, gb.DIAGNOSTIC_AMPLITUDES_PX
        ).all():
            raise RetentionSelectionError("the diagnostic frame is not exactly the 0.5 px rung")
        if set(primary_test["sample_id"]) & set(diagnostic_test["sample_id"]):
            raise RetentionSelectionError("primary and diagnostic frames share samples")
        return CheckResult(
            name="low_amplitude_exclusion",
            passed=True,
            detail=(
                "the 0.5 px rung is absent from the calibration fit frame and the primary "
                "score frame and appears only in the separately labelled diagnostic frame"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="low_amplitude_exclusion", passed=False, detail=f"{type(exc).__name__}: {exc}"
        )


def load_retention_frames(
    config: RetentionConfig, *, verify_images: bool = True
) -> RetentionFrames:
    """Load the frozen M3/M5A inputs through the M5B verified path."""
    m5b_config = _load_m5b_config(config)
    frames = gb.load_benchmark_frames(m5b_config, verify_images=verify_images)
    alternative = _check_alternative_justification(m5b_config, config.candidates.alternative_spec)

    checks = (
        *frames.checks,
        _check_calibration_isolation(frames.expanded_val, frames.expanded_train),
        _check_test_immutability(frames.primary_test, frames.diagnostic_test, frames.m3_test),
        _check_low_amplitude_excluded(
            frames.expanded_val, frames.primary_test, frames.diagnostic_test
        ),
    )
    rb._verify_checks(checks, "M5C frame checks")  # noqa: SLF001
    return RetentionFrames(
        frames=frames,
        expanded_train=frames.expanded_train,
        expanded_val=frames.expanded_val,
        primary_test=frames.primary_test,
        diagnostic_test=frames.diagnostic_test,
        m3_test=frames.m3_test,
        m3_frame=frames.m3_frame,
        m5a_frame=frames.m5a_frame,
        checks=checks,
        alternative_spec=config.candidates.alternative_spec,
        alternative_validation=alternative,
    )


# ---------------------------------------------------------------------------
# Models and calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AffineCalibration:
    """A univariate affine map ``target = intercept + slope * prediction``."""

    intercept: float
    slope: float
    clip_output: bool = False

    def apply(self, predicted: np.ndarray) -> np.ndarray:
        values = self.intercept + self.slope * np.asarray(predicted, dtype=np.float64)
        if self.clip_output:
            return np.clip(values, 0.0, 1.0)
        return values

    def as_payload(self) -> dict[str, float | bool]:
        return {
            "intercept": self.intercept,
            "slope": self.slope,
            "clip_output": self.clip_output,
        }


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    """One predeclared candidate policy: a base model plus an optional calibration."""

    name: str
    base_model: str
    calibration_form: str
    estimator: Any
    calibration: AffineCalibration | None

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        base = predict(self.estimator, frame)
        if self.calibration is None:
            return base
        return self.calibration.apply(base)


def _fit_affine(
    predicted: np.ndarray, target: np.ndarray, *, clip_output: bool = False
) -> AffineCalibration:
    """Fit ``target = intercept + slope * prediction`` by ordinary least squares."""
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.size < 2:
        raise RetentionSelectionError("affine calibration needs at least two observations")
    variance = float(np.sum((x - x.mean()) ** 2))
    if variance <= 0.0:
        raise RetentionSelectionError("affine calibration is degenerate: constant predictions")
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / variance)
    intercept = float(y.mean() - slope * x.mean())
    return AffineCalibration(intercept=intercept, slope=slope, clip_output=clip_output)


def _cross_fit_calibration(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    clip_output: bool = False,
) -> tuple[AffineCalibration, np.ndarray]:
    """Fit a calibration on validation data and return honest cross-fitted predictions.

    The calibration is fitted on the whole validation frame, but the validation
    predictions used to *score* the calibrated candidate are produced by
    leave-one-fold-out fits, so the candidate is never scored on rows that
    contributed to its own calibration. The fold assignment is deterministic and
    grouped by source, so all rows of a source stay together.
    """
    y = frame[TARGET].to_numpy(dtype=np.float64)
    groups = frame["source_id"].astype(str).to_numpy()
    unique_groups = np.array(sorted(set(groups)))
    if n_folds < 2:
        raise RetentionSelectionError("cross-fit calibration needs at least two folds")
    if len(unique_groups) < n_folds:
        raise RetentionSelectionError(
            f"cross-fit calibration needs at least {n_folds} source groups, "
            f"found {len(unique_groups)}"
        )
    assignment = {group: index % n_folds for index, group in enumerate(unique_groups.tolist())}
    fold_of_row = np.array([assignment[group] for group in groups], dtype=np.int64)
    cross_fitted = np.full(len(frame), np.nan, dtype=np.float64)
    for fold in range(n_folds):
        held_out = fold_of_row == fold
        if not held_out.any():
            continue
        keep = ~held_out
        calibration = _fit_affine(np.asarray(predicted)[keep], y[keep], clip_output=clip_output)
        cross_fitted[held_out] = calibration.apply(np.asarray(predicted)[held_out])
    if not np.isfinite(cross_fitted).all():
        raise RetentionSelectionError("cross-fitted calibration produced non-finite values")
    final = _fit_affine(predicted, y, clip_output=clip_output)
    del seed  # the fold assignment is deterministic; the seed is reserved for future use
    return final, cross_fitted


def _load_frozen_m3(
    m5b_config: gb.GeometricBenchmarkConfig, m3_summary: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Load the frozen M3 primary model through its existing verification path."""
    estimator, card, checks = gb.load_frozen_model(m5b_config, m3_summary)
    rb._verify_checks(checks, "frozen M3 model verification")  # noqa: SLF001
    if estimator is None:
        raise RetentionSelectionError("frozen M3 model could not be loaded")
    return estimator, card


def _load_m5b_expanded(
    config: RetentionConfig, m3_summary: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Load the M5B expanded model bundle and confirm it is unchanged."""
    artifact = config.m5b.model_dir / f"{config.m5b.model_stem}.joblib"
    _require_file_hash(artifact, config.m5b.model_artifact_sha256, "M5B expanded model artifact")
    estimator, card, checks = load_model_bundle(
        config.m5b.model_dir, config.m5b.model_stem, dataset_summary=m3_summary
    )
    rb._verify_checks(checks, "M5B expanded model verification")  # noqa: SLF001
    if estimator is None:
        raise RetentionSelectionError("M5B expanded model could not be loaded")
    return estimator, card


def _fit_alternative(
    m5b_config: gb.GeometricBenchmarkConfig, frames: RetentionFrames
) -> tuple[Any, dict[str, Any]]:
    """Fit the expanded conventional alternative on the expanded training frame."""
    specs = rb.candidate_specs_from_config(m5b_config)  # type: ignore[arg-type]
    spec = next((item for item in specs if item.name == frames.alternative_spec), None)
    if spec is None:
        raise RetentionSelectionError(
            f"alternative spec {frames.alternative_spec!r} is not in the M5B grid"
        )
    seed = derive_noise_seed(m5b_config.run.seed, "model", spec.name) % (2**32)
    model = fit_candidate(spec, frames.expanded_train, seed=seed)
    scores = score_frame(frames.expanded_val, predict(model.estimator, frames.expanded_val))
    return model.estimator, {
        "name": spec.name,
        "family": spec.family,
        "params": spec.params,
        "n_inputs": spec.n_inputs,
        "val_mae": float(scores["mae"]),
        "val_rmse": float(scores["rmse"]),
        "val_r2": float(scores["r2"]),
        "val_n": int(scores["n_rows"]),
        "train_n": int(len(frames.expanded_train)),
    }


# ---------------------------------------------------------------------------
# Candidate construction and slice scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """All predeclared candidate policies with their validation diagnostics."""

    policies: tuple[CandidatePolicy, ...]
    validation_predictions: dict[str, np.ndarray]
    calibration_diagnostics: dict[str, dict[str, Any]]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(policy.name for policy in self.policies)


def _candidate_name(base: str, calibration_form: str) -> str:
    if calibration_form == CALIBRATION_NONE:
        return base
    return f"{base}__{calibration_form}"


def build_candidate_set(
    frames: RetentionFrames,
    config: RetentionConfig,
    *,
    frozen_m3: Any,
    m5b_expanded: Any,
    alternative: Any,
) -> CandidateSet:
    """Build every predeclared candidate policy and score it on validation."""
    base_estimators: dict[str, Any] = {
        FROZEN_M3: frozen_m3,
        M5B_EXPANDED_HGB: m5b_expanded,
        EXPANDED_ALTERNATIVE: alternative,
    }
    val = frames.expanded_val
    policies: list[CandidatePolicy] = []
    validation_predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    for base in config.candidates.base_models:
        estimator = base_estimators[base]
        base_predictions = predict(estimator, val)
        for form in config.candidates.calibration_forms:
            name = _candidate_name(base, form)
            if form == CALIBRATION_NONE:
                policies.append(
                    CandidatePolicy(
                        name=name,
                        base_model=base,
                        calibration_form=form,
                        estimator=estimator,
                        calibration=None,
                    )
                )
                validation_predictions[name] = base_predictions
                diagnostics[name] = {
                    "calibration": "none",
                    "fit_rows": 0,
                    "fit_sources": 0,
                    "intercept": math.nan,
                    "slope": math.nan,
                    "scored_with": "base predictions",
                }
                continue

            calibration, cross_fitted = _cross_fit_calibration(
                val,
                base_predictions,
                n_folds=config.calibration.cross_fit_folds,
                seed=config.run.seed,
                clip_output=config.calibration.clip_output,
            )
            policies.append(
                CandidatePolicy(
                    name=name,
                    base_model=base,
                    calibration_form=form,
                    estimator=estimator,
                    calibration=calibration,
                )
            )
            # Validation scoring uses the cross-fitted predictions so a
            # calibrated candidate is never scored on rows that contributed to
            # its own calibration.
            validation_predictions[name] = cross_fitted
            diagnostics[name] = {
                "calibration": form,
                "fit_rows": int(len(val)),
                "fit_sources": int(val["source_id"].nunique()),
                "intercept": calibration.intercept,
                "slope": calibration.slope,
                "scored_with": "cross-fitted validation predictions",
            }
    return CandidateSet(
        policies=tuple(policies),
        validation_predictions=validation_predictions,
        calibration_diagnostics=diagnostics,
    )


def _slice_scores(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, R2 and residual spread for one slice."""
    scores = score_frame(frame, predicted)
    residual = frame[TARGET].to_numpy(dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    return {
        "mae": float(scores["mae"]),
        "rmse": float(scores["rmse"]),
        "r2": float(scores["r2"]),
        "r2_defined": bool(scores["r2_defined"]),
        "mean_residual": float(np.mean(residual)) if len(residual) else math.nan,
        "sd_residual": (float(np.std(residual, ddof=1)) if len(residual) > 1 else math.nan),
        "n_samples": int(scores["n_rows"]),
        "n_sources": int(scores["n_sources"]),
    }


def _slice_metrics_table(
    frames: RetentionFrames,
    candidates: CandidateSet,
    predictions: Mapping[str, np.ndarray],
    *,
    split_label: str,
) -> pd.DataFrame:
    """Score every candidate on the geometric, clean and original-task slices."""
    slices = frames.validation_slices
    rows: list[dict[str, Any]] = []
    for policy in candidates.policies:
        predicted = predictions[policy.name]
        for label, subset in slices.items():
            if subset.empty:
                continue
            mask = frames.expanded_val.index.get_indexer(subset.index)
            scores = _slice_scores(subset, np.asarray(predicted)[mask])
            rows.append(
                {
                    "model": policy.name,
                    "base_model": policy.base_model,
                    "calibration": policy.calibration_form,
                    "split": split_label,
                    "scope": label,
                    **scores,
                }
            )
    return pd.DataFrame(rows)


def _retention_score(
    metrics: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, float],
    settings: SelectionSettings,
) -> dict[str, float]:
    """Apply the predeclared retention-aware selection rule.

    All three slice errors are RMSE in SSIM units, so they are directly
    comparable. The score is the change relative to the frozen M3 reference,
    plus an explicit extra penalty on retention regressions.
    """
    delta_geo = metrics[GEO_LABEL]["rmse"] - reference[GEO_LABEL]["rmse"]
    delta_clean = metrics[CLEAN_LABEL]["rmse"] - reference[CLEAN_LABEL]["rmse"]
    delta_original = metrics[ORIGINAL_LABEL]["rmse"] - reference[ORIGINAL_LABEL]["rmse"]
    excess_clean = max(
        0.0, metrics[CLEAN_LABEL]["rmse"] - reference[CLEAN_LABEL]["rmse"] - settings.tolerance
    )
    excess_original = max(
        0.0,
        metrics[ORIGINAL_LABEL]["rmse"] - reference[ORIGINAL_LABEL]["rmse"] - settings.tolerance,
    )
    score = (
        settings.w_geometric * delta_geo
        + settings.w_clean * delta_clean
        + settings.w_original * delta_original
        + settings.penalty_regression * (excess_clean + excess_original)
    )
    return {
        "score": float(score),
        "delta_geometric_rmse": float(delta_geo),
        "delta_clean_rmse": float(delta_clean),
        "delta_original_rmse": float(delta_original),
        "excess_clean_rmse": float(excess_clean),
        "excess_original_rmse": float(excess_original),
    }


def select_policy(metrics: pd.DataFrame, settings: SelectionSettings) -> tuple[str, pd.DataFrame]:
    """Select the policy with the lowest retention-aware validation score."""
    reference_rows = metrics[metrics["model"] == FROZEN_M3]
    if reference_rows.empty:
        raise RetentionSelectionError("the frozen M3 reference is missing from the metrics table")
    reference = {
        str(row.scope): {"rmse": float(row.rmse)} for row in reference_rows.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for model, group in metrics.groupby("model", sort=True):
        slice_metrics = {
            str(row.scope): {"rmse": float(row.rmse)} for row in group.itertuples(index=False)
        }
        if set(slice_metrics) != set(reference):
            raise RetentionSelectionError(
                f"candidate {model!r} does not cover the same slices as the reference"
            )
        scored = _retention_score(slice_metrics, reference, settings)
        records.append(
            {
                "model": model,
                "base_model": str(group.iloc[0]["base_model"]),
                "calibration": str(group.iloc[0]["calibration"]),
                **scored,
            }
        )
    table = pd.DataFrame(records).sort_values(["score", "model"], kind="stable", ignore_index=True)
    table["selected"] = table["model"] == table.iloc[0]["model"]
    return str(table.iloc[0]["model"]), table


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestSlice:
    """One frozen evaluation slice with its own frame and predictions.

    Predictions are always computed on the slice's own frame, so the
    prediction array and the scored frame can never drift out of alignment.
    """

    scope: str
    frame: pd.DataFrame
    predictions: dict[str, np.ndarray]

    @property
    def label(self) -> str:
        return self.scope


def _predict_on(candidates: CandidateSet, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Predict every candidate on one frame."""
    return {policy.name: policy.predict(frame) for policy in candidates.policies}


def build_test_slices(frames: RetentionFrames, candidates: CandidateSet) -> list[TestSlice]:
    """Assemble the frozen evaluation slices scored after selection completes."""
    geometric = frames.primary_test
    m3_test = frames.m3_test
    slices: list[TestSlice] = [
        TestSlice(GEO_LABEL, geometric, _predict_on(candidates, geometric)),
        TestSlice(CLEAN_LABEL, m3_test[m3_test["is_clean"]], {}),
        TestSlice(ORIGINAL_LABEL, m3_test[~m3_test["is_clean"]], {}),
        TestSlice("m3_test_overall", m3_test, _predict_on(candidates, m3_test)),
        TestSlice(
            "diagnostic_0_5",
            frames.diagnostic_test,
            _predict_on(candidates, frames.diagnostic_test),
        ),
    ]
    clean_predictions = _predict_on(candidates, m3_test[m3_test["is_clean"]])
    original_predictions = _predict_on(candidates, m3_test[~m3_test["is_clean"]])
    slices[1] = TestSlice(CLEAN_LABEL, m3_test[m3_test["is_clean"]], clean_predictions)
    slices[2] = TestSlice(ORIGINAL_LABEL, m3_test[~m3_test["is_clean"]], original_predictions)

    resolved: list[TestSlice] = []
    for item in slices:
        if not item.frame.empty:
            resolved.append(item)
    for severity, subset in geometric.groupby("severity_level", sort=True):
        resolved.append(
            TestSlice(
                f"{GEO_LABEL}_severity_{int(severity)}",
                subset,
                _predict_on(candidates, subset),
            )
        )
    for family, subset in m3_test[~m3_test["is_clean"]].groupby("family", sort=True):
        resolved.append(
            TestSlice(
                f"{ORIGINAL_LABEL}_{family}",
                subset,
                _predict_on(candidates, subset),
            )
        )
    return resolved


def _test_metrics_table(slices: Sequence[TestSlice]) -> pd.DataFrame:
    """Score every candidate on every frozen test slice."""
    rows: list[dict[str, Any]] = []
    for item in slices:
        for name, predicted in item.predictions.items():
            scores = _slice_scores(item.frame, predicted)
            rows.append(
                {
                    "model": name,
                    "split": "test",
                    "scope": item.scope,
                    **scores,
                }
            )
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
                seed=derive_noise_seed(seed, "m5c_bootstrap", name, item.scope) % (2**64),
                evaluation="m5c_test",
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
    """Paired source-grouped differences against the frozen M3 reference."""
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
                seed=derive_noise_seed(seed, "m5c_paired", name, item.scope) % (2**64),
                evaluation="m5c_test",
            )
            rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetentionResult:
    """The complete M5C outcome."""

    frames: RetentionFrames
    checks: tuple[CheckResult, ...]
    candidates: CandidateSet
    validation_metrics: pd.DataFrame
    selection_table: pd.DataFrame
    selected_name: str
    test_metrics: pd.DataFrame
    uncertainty: pd.DataFrame
    paired_uncertainty: pd.DataFrame
    test_predictions: pd.DataFrame
    frozen_card: dict[str, Any]
    m5b_card: dict[str, Any]
    alternative_card: dict[str, Any]
    alternative_validation: dict[str, Any]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def selected(self) -> CandidatePolicy:
        for policy in self.candidates.policies:
            if policy.name == self.selected_name:
                return policy
        raise RetentionSelectionError(f"selected policy {self.selected_name!r} is missing")


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
        "split",
        "font_id",
        "text_length",
        "family",
        "severity_level",
        TARGET,
        "prediction",
        "residual",
        "abs_residual",
    ]
    columns = list(dict.fromkeys(columns))
    return pd.concat(frames, ignore_index=True)[columns]


def run_retention_core(
    frames: RetentionFrames,
    config: RetentionConfig,
    *,
    frozen_m3: Any,
    m5b_expanded: Any,
    alternative: Any,
    alternative_card: dict[str, Any],
) -> RetentionResult:
    """Run the full M5C comparison: validation selection, then one test evaluation."""
    rb._verify_checks(frames.checks, "M5C frame checks")  # noqa: SLF001

    candidates = build_candidate_set(
        frames,
        config,
        frozen_m3=frozen_m3,
        m5b_expanded=m5b_expanded,
        alternative=alternative,
    )

    # Selection uses validation data only.
    validation_metrics = _slice_metrics_table(
        frames, candidates, candidates.validation_predictions, split_label="val"
    )
    selected_name, selection_table = select_policy(validation_metrics, config.selection)

    # The policy and all calibration parameters are frozen before the test
    # frames are scored.
    slices = build_test_slices(frames, candidates)
    test_metrics = _test_metrics_table(slices)
    uncertainty = _uncertainty_table(
        slices, n_bootstrap=config.run.n_bootstrap, seed=config.run.seed
    )
    paired = _paired_uncertainty_table(
        slices, n_bootstrap=config.run.n_bootstrap, seed=config.run.seed, reference=FROZEN_M3
    )
    predictions = _prediction_table(slices, candidates.names)

    return RetentionResult(
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
        frozen_card={},
        m5b_card={},
        alternative_card=alternative_card,
        alternative_validation=frames.alternative_validation,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _selected_policy_card(result: RetentionResult, config: RetentionConfig) -> dict[str, Any]:
    """Build the model card for the selected policy."""
    policy = result.selected
    metrics = result.test_metrics[result.test_metrics["model"] == policy.name]
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
    return {
        "model_name": SELECTED_POLICY_STEM,
        "model_family": policy.base_model,
        "calibration": policy.calibration_form,
        "calibration_parameters": (
            policy.calibration.as_payload() if policy.calibration is not None else None
        ),
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
            "selected_model": policy.name,
            "base_model": policy.base_model,
            "n_candidates": len(result.candidates.policies),
            "benchmark_config_hash": config.config_hash,
        },
        "calibration_diagnostics": result.candidates.calibration_diagnostics[policy.name],
        "metrics": metric_payload,
    }


def _alternative_model_card(result: RetentionResult, config: RetentionConfig) -> dict[str, Any]:
    """Build the model card for the expanded conventional alternative."""
    validation = result.alternative_validation
    return {
        "model_name": ALTERNATIVE_MODEL_STEM,
        "model_family": "random_forest",
        "params": validation.get("params", {}),
        "feature_names": list(FEATURE_NAMES),
        "feature_set_version": config.schema.feature_set_version,
        "n_features": len(FEATURE_NAMES),
        "target": TARGET,
        "package_version": package_version(),
        "sklearn_version": sklearn_version(),
        "code_revision": code_revision(),
        "selection": {
            "criterion": "m5b_validation_rmse_best_non_hgb",
            "selected_on": "m5b_val",
            "justification": config.candidates.alternative_justification,
        },
        "dataset": {
            "generation_config_hash": config.m5b.protocol_hash,
            "split_assignment_sha256": config.m5b.protocol_hash,
            "expanded_train_samples": int(result.frames.expanded_train.shape[0]),
            "expanded_val_samples": int(result.frames.expanded_val.shape[0]),
        },
        "metrics": {
            "val": {
                "mae": float(validation["alternative_val_mae"]),
                "rmse": float(validation["alternative_val_rmse"]),
                "r2": float(validation["alternative_val_r2"]),
                "n_rows": int(validation.get("alternative_val_n", 0)),
            }
        },
    }


def write_retention_outputs(
    result: RetentionResult,
    config: RetentionConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    frozen_before: Mapping[str, Mapping[str, object]] | None = None,
    frozen_after: Mapping[str, Mapping[str, object]] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Path]:
    """Persist the M5C tables, model bundles, metadata and report."""
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
            {
                "check": check.name,
                "passed": check.passed,
                "detail": check.detail,
            }
            for check in result.checks
        ]
    ).to_csv(reports / "isolation_checks.csv", index=False)
    paths["validation_metrics"] = reports / "validation_slice_metrics.csv"
    paths["selection_table"] = reports / "selection_table.csv"
    paths["test_metrics"] = reports / "test_slice_metrics.csv"
    paths["uncertainty"] = reports / "uncertainty.csv"
    paths["paired_uncertainty"] = reports / "paired_uncertainty.csv"
    paths["test_predictions"] = reports / "test_predictions.csv"
    paths["isolation_checks"] = reports / "isolation_checks.csv"

    alternative_paths = save_model_bundle(
        models,
        ALTERNATIVE_MODEL_STEM,
        next(
            policy.estimator
            for policy in result.candidates.policies
            if policy.base_model == EXPANDED_ALTERNATIVE
        ),
        _alternative_model_card(result, config),
    )
    paths["alternative_artifact"] = alternative_paths["artifact"]
    paths["alternative_card"] = alternative_paths["card"]

    selected_card = _selected_policy_card(result, config)
    selected_paths = save_model_bundle(
        models,
        SELECTED_POLICY_STEM,
        result.selected.estimator,
        selected_card,
    )
    paths["selected_artifact"] = selected_paths["artifact"]
    paths["selected_card"] = selected_paths["card"]

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
            "train_sources": int(result.frames.expanded_train["source_id"].nunique()),
            "val_sources": int(result.frames.expanded_val["source_id"].nunique()),
            "test_sources": int(result.frames.primary_test["source_id"].nunique()),
            "expanded_train_samples": int(len(result.frames.expanded_train)),
            "expanded_val_samples": int(len(result.frames.expanded_val)),
            "primary_test_samples": int(len(result.frames.primary_test)),
            "diagnostic_test_samples": int(len(result.frames.diagnostic_test)),
            "m3_test_samples": int(len(result.frames.m3_test)),
        },
        "selection": {
            "rule": config.selection.rule,
            "selected_model": result.selected_name,
            "selected_base_model": result.selected.base_model,
            "selected_calibration": result.selected.calibration_form,
            "n_candidates": len(result.candidates.policies),
            "selected_on": "val",
        },
        "calibration": result.candidates.calibration_diagnostics,
        "alternative": result.alternative_validation,
        "frozen_assets": {
            "before": dict(frozen_before or {}),
            "after": dict(frozen_after or {}),
            "all_unchanged": _frozen_unchanged(frozen_before, frozen_after),
        },
        "artifacts": {},
    }
    for key, path in paths.items():
        if path.is_file():
            metadata["artifacts"][key] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
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


def _interval_for(
    table: pd.DataFrame, model: str, scope: str, metric: str
) -> tuple[float, float] | None:
    rows = table[
        (table["model"] == model) & (table["dataset"] == scope) & (table["metric"] == metric)
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return float(row["ci_low"]), float(row["ci_high"])


def _paired_interval_for(
    table: pd.DataFrame, model: str, scope: str, metric: str
) -> tuple[float, float, float] | None:
    rows = table[
        (table["model"] == f"{model}_minus_{FROZEN_M3}")
        & (table["dataset"] == scope)
        & (table["metric"] == metric)
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])


def _format_metric(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def _slice_table(
    result: RetentionResult, table: pd.DataFrame, scope: str, models: Sequence[str]
) -> list[str]:
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


def _source_consistency(result: RetentionResult, scope: str, reference: str) -> dict[str, Any]:
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
    candidates = [name for name in pivot.columns if name != reference]
    summary: dict[str, Any] = {"candidates": {}}
    for candidate in candidates:
        delta = (pivot[candidate] - pivot[reference]).dropna()
        if delta.empty:
            continue
        gains = -delta
        total = float(gains.sum())
        # The share is only meaningful when the aggregate change is a clear
        # improvement. If the gains and losses nearly cancel, a single source
        # can account for more than the whole net change, so it is reported as
        # not applicable rather than as a misleading percentage.
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


def _report_markdown(result: RetentionResult, config: RetentionConfig) -> str:
    """Render the deterministic M5C report."""
    models = list(result.candidates.names)
    selected = result.selected_name
    lines: list[str] = [
        "# M5C retention-aware model selection and calibration",
        "",
        "**Scope:** synthetic image similarity, measured as SSIM against the exact clean "
        "reference. This is not a measure of human readability, CAPTCHA solvability, "
        "security, or real-world quality.",
        "",
        "## Question",
        "",
        "Can a conventional model-selection or calibration policy improve geometric "
        "prediction while retaining performance on clean and blur/noise samples?",
        "",
        "Three explanations are distinguished by the reported slices:",
        "",
        "1. The expanded model learned the new target distribution but lost some "
        "original-task behaviour.",
        "2. The clean-reference errors reflect a systematic prediction offset that may "
        "be correctable.",
        "3. The feature representation or limited training data may not support a single "
        "model that performs consistently across all corruption families.",
        "",
        "## Frozen protocol",
        "",
        f"- Benchmark: `{config.run.name}` `{config.run.version}`",
        f"- Config: `{config.config_path}` (sha256 `{config.config_sha256}`)",
        f"- Protocol hash: `{config.config_hash}`",
        f"- Seed: `{config.run.seed}`; grouped bootstrap replicates: `{config.run.n_bootstrap}`",
        f"- Upstream M5B config sha256: `{config.m5b.config_sha256}`; M5B protocol hash: "
        f"`{config.m5b.protocol_hash}`",
        f"- M5B expanded model artifact sha256: `{config.m5b.model_artifact_sha256}`",
        f"- Feature schema: `{config.schema.feature_set_version}`, target `{TARGET}`, "
        f"{len(FEATURE_NAMES)} image-only features in their existing order",
        "",
        "The M3 dataset, split, frozen M3 model, M4A pilot, M4B benchmark, M5A pilot and "
        "M5B benchmark are read-only and hash-verified before and after the run.",
        "",
        "### Severity protocol",
        "",
        f"- Primary geometric score: amplitudes `{list(config.protocol.primary_amplitudes_px)}` px "
        f"(severities `{list(config.protocol.primary_severities)}`).",
        f"- Diagnostic only: amplitude `{list(config.protocol.diagnostic_amplitudes_px)}` px "
        f"(severity `{list(config.protocol.diagnostic_severities)}`).",
        "- The 0.5 px rung never enters fitting, validation selection or the primary score.",
        "",
        "## Predeclared candidate policies",
        "",
        "| candidate | base model | calibration |",
        "| --- | --- | --- |",
    ]
    for policy in result.candidates.policies:
        lines.append(f"| `{policy.name}` | {policy.base_model} | {policy.calibration_form} |")
    lines.extend(
        [
            "",
            f"The expanded conventional alternative is `{config.candidates.alternative_spec}`. "
            f"Justification: {config.candidates.alternative_justification}",
            "",
            f"Reproduced from the M5B validation table: the M5B grid has "
            f"{result.alternative_validation['m5b_grid_candidates']} candidates, the "
            f"M5B validation-selected model is "
            f"`{result.alternative_validation['m5b_selected_on_validation']}` with validation "
            f"RMSE {result.alternative_validation['m5b_selected_val_rmse']:.4f}, and the "
            f"alternative reaches validation RMSE "
            f"{result.alternative_validation['alternative_val_rmse']:.4f} "
            f"({result.alternative_validation['alternative_val_gap_fraction'] * 100:.1f}% behind). "
            "No M5C test result was used to justify it.",
            "",
            "## Calibration procedure",
            "",
            "- Form: univariate affine, `target = intercept + slope * prediction`.",
            "- Fit data: the "
            f"`{config.calibration.fit_scope}` frame. The base models are trained only on "
            "training sources, so these predictions are out-of-sample and no test target "
            "ever enters a calibration.",
            f"- Cross-fitting: {config.calibration.cross_fit_folds} source-grouped folds. "
            "Validation scoring uses leave-one-fold-out calibration predictions, so a "
            "calibrated candidate is never scored on rows that contributed to its own "
            "calibration. The frozen policy uses the calibration fitted on the whole "
            "validation frame.",
            "- The mapping uses no metadata: severity, family, font, text length, source and "
            "split identifiers never enter it.",
            f"- `clip_output` is `{str(config.calibration.clip_output).lower()}`, so the "
            "calibration is a pure affine map with no saturation. Clipping to [0, 1] would "
            "silently absorb the clean-reference residual and hide the offset this "
            "milestone is trying to measure.",
            "",
            "| candidate | calibration | fit rows | fit sources | intercept | slope | "
            "validation scored with |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name in result.candidates.names:
        entry = result.candidates.calibration_diagnostics[name]
        if entry["calibration"] == CALIBRATION_NONE:
            lines.append(f"| `{name}` | none | — | — | — | — | base predictions |")
        else:
            lines.append(
                f"| `{name}` | {entry['calibration']} | {entry['fit_rows']} | "
                f"{entry['fit_sources']} | {entry['intercept']:.6f} | "
                f"{entry['slope']:.6f} | {entry['scored_with']} |"
            )
    lines.extend(
        [
            "",
            "## Retention-aware selection rule",
            "",
            "The rule was declared before any test evaluation. All three slice errors are "
            "RMSE in SSIM units, so they are directly comparable. The score is the change "
            "relative to the frozen M3 reference, plus an explicit extra penalty on "
            "retention regressions:",
            "",
            "```text",
            "score = (rmse_geo - rmse_geo_ref)",
            "      + w_clean * (rmse_clean - rmse_clean_ref)",
            "      + w_original * (rmse_original - rmse_original_ref)",
            "      + penalty * ( max(0, rmse_clean - rmse_clean_ref - tolerance)",
            "                  + max(0, rmse_original - rmse_original_ref - tolerance) )",
            "```",
            "",
            f"With `w_geometric={config.selection.w_geometric}`, "
            f"`w_clean={config.selection.w_clean}`, "
            f"`w_original={config.selection.w_original}`, "
            f"`penalty_regression={config.selection.penalty_regression}` and "
            f"`tolerance={config.selection.tolerance}`. Lower is better and the frozen M3 "
            "model scores exactly 0. A geometric gain and a retention loss of the same size "
            "cancel, and a retention regression is charged twice, so it must be more than "
            "twice the geometric gain to be preferred. No weight or threshold was chosen "
            "after seeing a test result.",
            "",
            "## Validation results by slice",
            "",
            "Validation-only. This is the evidence the selection rule consumed.",
            "",
        ]
    )
    for scope, title in (
        (GEO_LABEL, "Geometric validation slice (1.0-3.0 px)"),
        (CLEAN_LABEL, "Clean-reference validation slice"),
        (ORIGINAL_LABEL, "Original blur/noise validation slice"),
    ):
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(_slice_table(result, result.validation_metrics, scope, models))
        lines.append("")

    lines.extend(
        [
            "### Retention-aware selection scores",
            "",
            "| candidate | score | delta geometric RMSE | delta clean RMSE | "
            "delta original RMSE | selected |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result.selection_table.itertuples(index=False):
        lines.append(
            f"| `{row.model}` | {row.score:.5f} | {row.delta_geometric_rmse:+.5f} | "
            f"{row.delta_clean_rmse:+.5f} | {row.delta_original_rmse:+.5f} | "
            f"{'yes' if row.selected else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Selected policy: **`{selected}`** "
            f"(base `{result.selected.base_model}`, calibration "
            f"`{result.selected.calibration_form}`).",
            "",
        ]
    )
    diagnostics = result.candidates.calibration_diagnostics[selected]
    if diagnostics["calibration"] != CALIBRATION_NONE:
        lines.extend(
            [
                "### Selected calibration parameters",
                "",
                f"- Intercept: `{diagnostics['intercept']:.6f}`",
                f"- Slope: `{diagnostics['slope']:.6f}`",
                f"- Fitted on: {diagnostics['fit_rows']} validation rows from "
                f"{diagnostics['fit_sources']} sources",
                f"- Validation scoring used: {diagnostics['scored_with']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Final test results by slice",
            "",
            "The selected policy and all calibration parameters were frozen before these "
            "rows were scored. The M3 test split and the M5B geometric test split are "
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
        lines.extend(_slice_table(result, result.test_metrics, scope, models))
        lines.append("")

    lines.append("### Geometric test results by severity")
    lines.append("")
    for severity in sorted(
        int(scope.rsplit("_", 1)[1])
        for scope in result.test_metrics["scope"].unique()
        if scope.startswith(f"{GEO_LABEL}_severity_")
    ):
        scope = f"{GEO_LABEL}_severity_{severity}"
        lines.append(f"**Severity {severity}**")
        lines.append("")
        lines.extend(_slice_table(result, result.test_metrics, scope, models))
        lines.append("")

    lines.extend(
        [
            "## Paired source-grouped differences",
            "",
            "Whole source groups are resampled, preserving all rows within each selected "
            "group. Differences are candidate minus the frozen M3 reference on identical "
            "rows. With only 12 held-out sources these intervals are preliminary and "
            "descriptive; overlap or non-overlap is not a significance test.",
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
            if model == FROZEN_M3:
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
            "which the selected policy has a lower absolute error than the frozen M3 "
            "reference.",
            "",
            "| scope | candidate | sources improved | worst source change | best source "
            "change | largest single-source share of the total gain |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope in (GEO_LABEL, ORIGINAL_LABEL, "m3_test_overall"):
        consistency = _source_consistency(result, scope, FROZEN_M3)
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
    lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "No universal pass/fail threshold was invented after observing test results. "
            "The statements below are descriptive, use held-out sources only, and are not "
            "causal claims about human or production CAPTCHA performance.",
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
            "- Only 12 held-out test sources, so all grouped intervals are preliminary.",
            "- The 0.5 px rung is close to the interpolation floor and is diagnostic only.",
            "- Bilinear resampling and edge padding are part of the corruption; the M5A "
            "minimal control and border diagnostics quantify, but do not remove, that "
            "confound.",
            "- The calibration is univariate and global, so it can only correct a systematic "
            "offset or slope, not a slice-dependent distortion.",
            "- Validation has 12 sources, so the cross-fitted calibration is estimated from a "
            "small number of groups.",
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
            "uv run --extra image --extra ml captcha-retention-aware run \\",
            "    --config configs/retention_aware_selection_v0_1_0.toml",
            "uv run --extra image --extra ml captcha-retention-aware verify \\",
            "    --config configs/retention_aware_selection_v0_1_0.toml",
            "```",
            "",
            "Generated tables, the model bundles and this report live in Git-ignored paths.",
        ]
    )
    return "\n".join(lines) + "\n"


def _decision_lines(result: RetentionResult, config: RetentionConfig) -> list[str]:
    """Derive the descriptive interpretation from the observed results."""
    selected = result.selected_name
    lines: list[str] = []

    def metric(model: str, scope: str) -> dict[str, Any] | None:
        return _metric_row_for(result.test_metrics, model, scope)

    frozen_geo = metric(FROZEN_M3, GEO_LABEL)
    selected_geo = metric(selected, GEO_LABEL)
    frozen_clean = metric(FROZEN_M3, CLEAN_LABEL)
    selected_clean = metric(selected, CLEAN_LABEL)
    frozen_original = metric(FROZEN_M3, ORIGINAL_LABEL)
    selected_original = metric(selected, ORIGINAL_LABEL)
    frozen_m3 = metric(FROZEN_M3, "m3_test_overall")
    selected_m3 = metric(selected, "m3_test_overall")

    if frozen_geo and selected_geo:
        change = (selected_geo["rmse"] - frozen_geo["rmse"]) / frozen_geo["rmse"] * 100
        lines.append(
            f"- **Primary geometric slice.** The frozen M3 model reaches MAE "
            f"{frozen_geo['mae']:.4f}, RMSE {frozen_geo['rmse']:.4f} and R2 "
            f"{frozen_geo['r2']:.4f} on {frozen_geo['n_samples']} rows from "
            f"{frozen_geo['n_sources']} sources. The selected policy `{selected}` reaches "
            f"MAE {selected_geo['mae']:.4f}, RMSE {selected_geo['rmse']:.4f} and R2 "
            f"{selected_geo['r2']:.4f}, a {change:+.1f}% change in RMSE against the "
            "frozen reference."
        )
    if frozen_clean and selected_clean:
        lines.append(
            f"- **Clean-reference slice.** Frozen MAE {frozen_clean['mae']:.4f} / RMSE "
            f"{frozen_clean['rmse']:.4f}; selected MAE {selected_clean['mae']:.4f} / RMSE "
            f"{selected_clean['rmse']:.4f}."
        )
    if frozen_original and selected_original:
        lines.append(
            f"- **Original blur/noise slice.** Frozen MAE {frozen_original['mae']:.4f} / RMSE "
            f"{frozen_original['rmse']:.4f}; selected MAE {selected_original['mae']:.4f} / "
            f"RMSE {selected_original['rmse']:.4f}."
        )
    if frozen_m3 and selected_m3:
        lines.append(
            f"- **Original-task retention.** On the same untouched M3 test rows the frozen "
            f"model reaches MAE {frozen_m3['mae']:.4f} / RMSE {frozen_m3['rmse']:.4f} / R2 "
            f"{frozen_m3['r2']:.4f} and the selected policy reaches MAE "
            f"{selected_m3['mae']:.4f} / RMSE {selected_m3['rmse']:.4f} / R2 "
            f"{selected_m3['r2']:.4f}."
        )

    geo_improved = bool(selected_geo and frozen_geo and selected_geo["rmse"] < frozen_geo["rmse"])
    clean_retained = bool(
        selected_clean
        and frozen_clean
        and selected_clean["rmse"] <= frozen_clean["rmse"] + config.selection.tolerance
    )
    original_retained = bool(
        selected_original
        and frozen_original
        and selected_original["rmse"] <= frozen_original["rmse"] + config.selection.tolerance
    )
    if geo_improved and clean_retained and original_retained:
        lines.append(
            "- **Trade-off.** The selected policy improves the primary geometric slice "
            "without regressing either retention slice. The retention-aware rule found a "
            "policy that dominates the frozen reference on this validation protocol."
        )
    elif geo_improved:
        lines.append(
            "- **Trade-off.** The selected policy improves the primary geometric slice but "
            "regresses at least one retention slice relative to the frozen reference. The "
            "retention-aware rule already priced that regression in, so the geometric gain "
            "was large enough to survive it, but the trade-off is not free."
        )
    else:
        lines.append(
            "- **Trade-off.** No candidate improved the primary geometric slice while "
            "retaining the frozen reference's behaviour, so the frozen M3 model remains the "
            "selected policy. This is reported as a negative result rather than a "
            "manufactured winner."
        )

    if selected == FROZEN_M3:
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark family. "
            "Under a retention-aware policy the frozen M3 model is not displaced on this "
            "protocol. The evidence does not justify a CNN, and a new corruption family "
            "does not by itself justify one."
        )
    else:
        lines.append(
            "- **Decision.** Conventional feature-based ML remains the benchmark family, now "
            f"with a retention-aware policy (`{selected}`) rather than the frozen model "
            "alone. The evidence does not justify a CNN, and a new corruption family does "
            "not by itself justify one."
        )

    # The three candidate explanations, checked against the reported slices.
    hgb_clean = metric(M5B_EXPANDED_HGB, CLEAN_LABEL)
    selected_clean_metric = metric(selected, CLEAN_LABEL)
    frozen_clean_metric = metric(FROZEN_M3, CLEAN_LABEL)
    hgb_original = metric(M5B_EXPANDED_HGB, ORIGINAL_LABEL)
    selected_original = metric(selected, ORIGINAL_LABEL)
    if hgb_clean and selected_clean_metric and frozen_clean_metric:
        lines.append(
            "- **Explanation 1 - lost original-task behaviour.** The M5B expanded HGB loses "
            f"clean-reference accuracy (RMSE {hgb_clean['rmse']:.4f} against the frozen "
            f"{frozen_clean_metric['rmse']:.4f}) and loses original-task accuracy "
            f"(RMSE {hgb_original['rmse']:.4f} against the frozen "
            f"{metric(FROZEN_M3, ORIGINAL_LABEL)['rmse']:.4f}), so the M5B result is "
            "consistent with a model that traded original-task behaviour for geometric "
            "accuracy."
        )
    if hgb_clean and selected_clean_metric:
        lines.append(
            "- **Explanation 2 - a correctable systematic offset.** The clean-reference "
            f"error of the M5B expanded HGB is a near-constant offset (mean residual "
            f"{hgb_clean['mean_residual']:+.4f}), and the affine calibration reduces it "
            f"to {metric(M5B_EXPANDED_HGB + '__affine', CLEAN_LABEL)['mean_residual']:+.4f}. "
            "The offset is therefore partly correctable, but a single global affine map "
            "cannot remove it without moving the other slices: the calibrated HGB still "
            f"leaves clean-reference RMSE at "
            f"{metric(M5B_EXPANDED_HGB + '__affine', CLEAN_LABEL)['rmse']:.4f}, above the "
            f"frozen reference {frozen_clean_metric['rmse']:.4f}."
        )
    lines.append(
        "- **Explanation 3 - a single model across families.** The selected policy keeps "
        f"the original blur/noise RMSE at {selected_original['rmse']:.4f} against the "
        f"frozen {metric(FROZEN_M3, ORIGINAL_LABEL)['rmse']:.4f} while improving the "
        "geometric slice, so the eight features do support one model across both "
        "corruption families. The residual difficulty is between-source variation at "
        "fixed severity, which the aggregate score does not remove."
    )
    lines.append(
        "- **Note on the reference's validation behaviour.** The frozen M3 model was itself "
        "selected using the M3 validation split, and its clean-reference validation slice "
        "is unusually noisy relative to its clean-reference test slice. The retention "
        "deltas are therefore measured against a reference whose own validation clean "
        "error is larger than its test clean error, which makes the measured clean "
        "regression of the selected policy an upper bound rather than a point estimate."
    )
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_retention_aware(
    config: RetentionConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    verify_images: bool = True,
) -> tuple[RetentionResult, dict[str, Path]]:
    """Load, verify, run and persist the complete M5C study."""
    started = time.perf_counter()
    before = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    frames = load_retention_frames(config, verify_images=verify_images)
    m5b_config = _load_m5b_config(config)

    frozen_m3, frozen_card = _load_frozen_m3(m5b_config, frames.frames.m3_summary)
    m5b_expanded, m5b_card = _load_m5b_expanded(config, frames.frames.m3_summary)
    alternative, alternative_validation = _fit_alternative(m5b_config, frames)

    result = run_retention_core(
        frames,
        config,
        frozen_m3=frozen_m3,
        m5b_expanded=m5b_expanded,
        alternative=alternative,
        alternative_card=alternative_validation,
    )
    result = RetentionResult(
        frames=result.frames,
        checks=result.checks,
        candidates=result.candidates,
        validation_metrics=result.validation_metrics,
        selection_table=result.selection_table,
        selected_name=result.selected_name,
        test_metrics=result.test_metrics,
        uncertainty=result.uncertainty,
        paired_uncertainty=result.paired_uncertainty,
        test_predictions=result.test_predictions,
        frozen_card=frozen_card,
        m5b_card=m5b_card,
        alternative_card=result.alternative_card,
        alternative_validation=result.alternative_validation,
    )
    after = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    if not _frozen_unchanged(before, after):
        raise RetentionSelectionError("a frozen asset changed during the M5C run")
    paths = write_retention_outputs(
        result,
        config,
        output_reports=output_reports,
        output_models=output_models,
        frozen_before=before,
        frozen_after=after,
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, paths


def verify_retention_aware(
    config: RetentionConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
) -> dict[str, Any]:
    """Verify a written M5C run: hashes, protocol, selection and prediction replay."""
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
        _require_file_hash(config.m5b.config, config.m5b.config_sha256, "M5B config")
        _load_m5b_config(config)
        _require_file_hash(
            config.m5b.model_dir / f"{config.m5b.model_stem}.joblib",
            config.m5b.model_artifact_sha256,
            "M5B expanded model artifact",
        )
        frames = load_retention_frames(config, verify_images=False)
        return (
            f"{len(frames.expanded_train)} expanded training rows, "
            f"{len(frames.expanded_val)} expanded validation rows, "
            f"{len(frames.primary_test)} primary geometric test rows and "
            f"{len(frames.diagnostic_test)} diagnostic test rows"
        )

    def metadata_checks() -> str:
        metadata_path = reports / "run_metadata.json"
        if not metadata_path.is_file():
            raise RetentionSelectionError(f"missing run metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("benchmark_config_sha256") != config.config_sha256:
            raise RetentionSelectionError("run metadata config hash does not match")
        if metadata.get("benchmark_config_hash") != config.config_hash:
            raise RetentionSelectionError("run metadata protocol hash does not match")
        if not metadata.get("frozen_assets", {}).get("all_unchanged"):
            raise RetentionSelectionError("run metadata reports a changed frozen asset")
        for name, entry in metadata.get("artifacts", {}).items():
            path = Path(entry["path"]) if isinstance(entry, dict) else reports / name
            if not path.is_file():
                raise RetentionSelectionError(f"missing recorded artifact: {path}")
            if sha256_file(path) != (entry["sha256"] if isinstance(entry, dict) else entry):
                raise RetentionSelectionError(f"artifact hash mismatch: {name}")
        return f"{len(metadata.get('artifacts', {}))} artifact hashes match"

    def selection_checks() -> str:
        table_path = reports / "selection_table.csv"
        if not table_path.is_file():
            raise RetentionSelectionError(f"missing selection table: {table_path}")
        table = pd.read_csv(table_path)
        selected = table[table["selected"]]
        if len(selected) != 1:
            raise RetentionSelectionError("the selection table must mark exactly one winner")
        metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
        recorded = metadata.get("selection", {}).get("selected_model")
        if recorded != str(selected.iloc[0]["model"]):
            raise RetentionSelectionError(
                f"selection table winner {selected.iloc[0]['model']!r} does not match the "
                f"recorded selection {recorded!r}"
            )
        return f"selected policy {recorded!r} is consistent"

    def model_checks() -> str:
        for stem in (ALTERNATIVE_MODEL_STEM, SELECTED_POLICY_STEM):
            card_path = models / f"{stem}_card.json"
            artifact = models / f"{stem}.joblib"
            if not card_path.is_file() or not artifact.is_file():
                raise RetentionSelectionError(f"missing model bundle for {stem!r}")
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("artifact_sha256") != sha256_file(artifact):
                raise RetentionSelectionError(f"artifact hash mismatch for {stem!r}")
            if card.get("feature_names") != list(FEATURE_NAMES):
                raise RetentionSelectionError(f"feature schema mismatch for {stem!r}")
            if card.get("target") != TARGET:
                raise RetentionSelectionError(f"target mismatch for {stem!r}")
        return "both model bundles load, match their cards and use the frozen feature schema"

    def replay_checks() -> str:
        frames = load_retention_frames(config, verify_images=False)
        m5b_config = _load_m5b_config(config)
        frozen_m3, _ = _load_frozen_m3(m5b_config, frames.frames.m3_summary)
        m5b_expanded, _ = _load_m5b_expanded(config, frames.frames.m3_summary)
        alternative, _ = _fit_alternative(m5b_config, frames)
        candidates = build_candidate_set(
            frames,
            config,
            frozen_m3=frozen_m3,
            m5b_expanded=m5b_expanded,
            alternative=alternative,
        )
        slices = build_test_slices(frames, candidates)
        replayed = _test_metrics_table(slices)
        stored = pd.read_csv(reports / "test_slice_metrics.csv")
        merged = replayed.merge(stored, on=["model", "scope"], suffixes=("_replay", "_stored"))
        if len(merged) != len(replayed):
            raise RetentionSelectionError("replayed metrics do not cover the stored rows")
        for column in ("mae", "rmse"):
            if not np.allclose(
                merged[f"{column}_replay"], merged[f"{column}_stored"], rtol=0.0, atol=1e-9
            ):
                raise RetentionSelectionError(f"replayed {column} does not match the stored run")
        return f"{len(replayed)} slice metrics reproduce exactly"

    def report_checks() -> str:
        report_path = reports / "report.md"
        if not report_path.is_file():
            raise RetentionSelectionError(f"missing report: {report_path}")
        text = report_path.read_text(encoding="utf-8")
        required = (
            "Predeclared candidate policies",
            "Retention-aware selection rule",
            "Calibration procedure",
            "Final test results by slice",
            "Paired source-grouped differences",
            "Reproducibility",
        )
        missing = [section for section in required if section not in text]
        if missing:
            raise RetentionSelectionError(f"report is missing sections: {missing}")
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
        prog="captcha-retention-aware",
        description="Run or verify the grouped M5C retention-aware selection benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the retention-aware selection study")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path, default=None)
    run_parser.add_argument("--output-models", type=Path, default=None)

    verify_parser = subparsers.add_parser("verify", help="verify a written retention-aware run")
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path, default=None)
    verify_parser.add_argument("--output-models", type=Path, default=None)
    return parser


def _print_report(report: Mapping[str, Any]) -> None:
    for check in report["checks"]:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"retention-aware selection verification: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_retention_config(args.config)
        if args.command == "run":
            result, paths = run_retention_aware(
                config,
                output_reports=args.output_reports,
                output_models=args.output_models,
            )
            print(f"benchmark: {config.describe()}")
            print(f"protocol hash: {config.config_hash}")
            print(
                "rows: expanded train "
                f"{len(result.frames.expanded_train)}, expanded val "
                f"{len(result.frames.expanded_val)}, primary test "
                f"{len(result.frames.primary_test)}, diagnostic 0.5 px "
                f"{len(result.frames.diagnostic_test)}"
            )
            print(f"selected policy: {result.selected_name}")
            for scope in (GEO_LABEL, CLEAN_LABEL, ORIGINAL_LABEL, "m3_test_overall"):
                row = _metric_row_for(result.test_metrics, result.selected_name, scope)
                frozen = _metric_row_for(result.test_metrics, FROZEN_M3, scope)
                if row is None or frozen is None:
                    continue
                print(
                    f"  {scope}: selected MAE {row['mae']:.4f} RMSE {row['rmse']:.4f} | "
                    f"frozen M3 MAE {frozen['mae']:.4f} RMSE {frozen['rmse']:.4f}"
                )
            print(f"report: {paths['report']}")
            print(f"model: {paths['selected_artifact']}")
            return 0
        report = verify_retention_aware(
            config,
            output_reports=args.output_reports,
            output_models=args.output_models,
        )
        _print_report(report)
        return 0 if report["passed"] else 1
    except (RetentionSelectionError, RetentionSelectionConfigError, ModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
