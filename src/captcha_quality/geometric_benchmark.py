"""Milestone 5B grouped geometric-robustness benchmark.

The benchmark keeps the Milestone 3 task fixed: predict ``ssim_target`` from the
same eight image-only features, using the frozen source-grouped split and the
frozen M3 model bundle. It adds three experiments on the M5A smooth elastic
warp dataset:

* **Experiment A (zero-shot).** The unmodified M3 model is evaluated on the
  held-out M5A test sources. No preprocessing, calibration or model parameter is
  fitted on those rows.
* **Experiment B (controlled adaptation).** Conventional models are fitted on
  M3 train rows plus M5A primary geometric rows from training sources, selected
  on M5A validation sources only, and then evaluated once on M5A test sources.
* **Experiment C (original-task retention).** The selected adapted model is
  scored on the untouched M3 blur/noise test split and compared with the frozen
  M3 model on the same rows.

The primary geometric score uses the 1.0, 2.0 and 3.0 px maximum-displacement
rungs. The 0.5 px rung is evaluated and reported separately as a diagnostic; it
never enters primary training, validation selection or the primary score. M5A's
predeclared gate was CONDITIONAL PASS because that rung's between-source SSIM
standard deviation was below the declared minimum-variation margin.

M5A clean and control rows duplicate or near-duplicate the frozen M3 clean
references. They are excluded from expanded training, validation and model
selection. The target is synthetic image similarity (SSIM to a clean reference).
Nothing here measures human readability, CAPTCHA solvability or security.
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

from captcha_quality import robustness_benchmark as rb
from captcha_quality.benchmark import load_benchmark_dataset
from captcha_quality.config import FEATURE_SET_VERSION, ConfigError
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.dataset import (
    CheckResult,
    DatasetError,
    VerificationReport,
    verify_dataset,
)
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.geometric_pilot import GeometricPilotError, verify_pilot_dataset
from captcha_quality.modelling import (
    TARGET,
    FittedModel,
    ModelError,
    fit_candidate,
    load_model_bundle,
    naive_spec,
    predict,
    save_model_bundle,
    score_frame,
    select_best,
    sklearn_version,
    validate_feature_columns,
)
from captcha_quality.provenance import (
    canonical_json,
    code_revision,
    package_version,
    sha256_bytes,
    sha256_file,
)

M3_LABEL: Final[str] = "m3"
M5A_LABEL: Final[str] = "m5a"
PRIMARY_LABEL: Final[str] = "m5a_primary_test"
DIAGNOSTIC_LABEL: Final[str] = "m5a_diagnostic_0_5_test"
M3_TEST_LABEL: Final[str] = "m3_test"

FROZEN_MODEL_NAME: Final[str] = "frozen_m3"
M3_NAIVE_NAME: Final[str] = "naive_m3_train_mean"
EXPANDED_MODEL_NAME: Final[str] = "expanded_geometric_model"
EXPANDED_NAIVE_NAME: Final[str] = "naive_expanded_train_mean"
EXPANDED_MODEL_STEM: Final[str] = "expanded_geometric_model"

PRIMARY_AMPLITUDES_PX: Final[tuple[float, ...]] = (1.0, 2.0, 3.0)
DIAGNOSTIC_AMPLITUDES_PX: Final[tuple[float, ...]] = (0.5,)
PRIMARY_SEVERITIES: Final[tuple[int, ...]] = (2, 3, 4)
DIAGNOSTIC_SEVERITIES: Final[tuple[int, ...]] = (1,)
CLEAN_FAMILY: Final[str] = "clean"
CONTROL_FAMILY: Final[str] = "control"
ELASTIC_FAMILY: Final[str] = "elastic"
M3_CLEAN_FAMILY: Final[str] = "none"

_SELECTION_METRICS: Final[tuple[str, ...]] = ("rmse", "mae")
_EXPECTED_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {"run", "m3", "m5a", "model", "schema", "grid", "protocol"}
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Frozen inputs audited before and after the benchmark. Directories are
#: fingerprinted recursively; generated M5B outputs are not in this list.
FROZEN_ASSET_PATHS: Final[tuple[str, ...]] = (
    "configs/dataset_v0_1_0.toml",
    "configs/benchmark_v0_1_0.toml",
    "configs/robustness_pilot_occlusion_v0_1_0.toml",
    "configs/robustness_benchmark_v0_1_0.toml",
    "configs/geometric_distortion_pilot_v0_1_0.toml",
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "data/processed/geometric_distortion_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "models/robustness_benchmark_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
    "reports/geometric_distortion_pilot",
)

PROTECTED_OUTPUT_ROOTS: Final[tuple[str, ...]] = (
    "configs",
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "data/processed/geometric_distortion_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "models/robustness_benchmark_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
    "reports/robustness_benchmark_v0_1_0",
    "reports/geometric_distortion_pilot",
)


class GeometricBenchmarkConfigError(ValueError):
    """Raised when the M5B configuration is missing or malformed."""


class GeometricBenchmarkError(RuntimeError):
    """Raised when the geometric benchmark cannot run as declared."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M5APin:
    dataset: Path
    config: Path
    dataset_version: str
    config_sha256: str
    pilot_config_hash: str
    manifest_sha256: str
    sources_sha256: str
    features_sha256: str
    summary_sha256: str
    image_pixel_hash_set: str
    n_samples: int
    n_sources: int
    gate_verdict: str


@dataclass(frozen=True, slots=True)
class ProtocolSettings:
    primary_amplitudes_px: tuple[float, ...]
    diagnostic_amplitudes_px: tuple[float, ...]
    primary_severities: tuple[int, ...]
    diagnostic_severities: tuple[int, ...]
    exclude_control_rows_from_fit: bool


@dataclass(frozen=True, slots=True)
class GeometricBenchmarkConfig:
    run: rb.RunSettings
    m3: rb.M3Pin
    m5a: M5APin
    model: rb.ModelPin
    schema: rb.SchemaSettings
    grid: rb.GridSettings
    protocol: ProtocolSettings
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
                "selection_metric": self.run.selection_metric,
                "output_reports": str(self.run.output_reports),
                "output_models": str(self.run.output_models),
            },
            "m3": {
                "dataset": str(self.m3.dataset),
                "config": str(self.m3.config),
                "dataset_version": self.m3.dataset_version,
                "generation_config_hash": self.m3.generation_config_hash,
                "manifest_sha256": self.m3.manifest_sha256,
                "sources_sha256": self.m3.sources_sha256,
                "features_sha256": self.m3.features_sha256,
                "ground_truth_sha256": self.m3.ground_truth_sha256,
                "summary_sha256": self.m3.summary_sha256,
                "split_assignment_sha256": self.m3.split_assignment_sha256,
            },
            "m5a": {
                "dataset": str(self.m5a.dataset),
                "config": str(self.m5a.config),
                "dataset_version": self.m5a.dataset_version,
                "config_sha256": self.m5a.config_sha256,
                "pilot_config_hash": self.m5a.pilot_config_hash,
                "manifest_sha256": self.m5a.manifest_sha256,
                "sources_sha256": self.m5a.sources_sha256,
                "features_sha256": self.m5a.features_sha256,
                "summary_sha256": self.m5a.summary_sha256,
                "image_pixel_hash_set": self.m5a.image_pixel_hash_set,
                "n_samples": self.m5a.n_samples,
                "n_sources": self.m5a.n_sources,
                "gate_verdict": self.m5a.gate_verdict,
            },
            "model": {
                "model_dir": str(self.model.model_dir),
                "model_stem": self.model.model_stem,
                "card_sha256": self.model.card_sha256,
                "artifact_sha256": self.model.artifact_sha256,
            },
            "schema": {
                "feature_set_version": self.schema.feature_set_version,
                "target": self.schema.target,
                "feature_names": list(self.schema.feature_names),
                "split_names": list(self.schema.split_names),
            },
            "grid": {
                "include_ols": self.grid.include_ols,
                "ridge": {"alphas": list(self.grid.ridge.alphas)},
                "random_forest": {
                    "n_estimators": self.grid.random_forest.n_estimators,
                    "max_depths": list(self.grid.random_forest.max_depths),
                    "min_samples_leafs": list(self.grid.random_forest.min_samples_leafs),
                },
                "hist_gradient_boosting": {
                    "max_iter": self.grid.hist_gradient_boosting.max_iter,
                    "learning_rates": list(self.grid.hist_gradient_boosting.learning_rates),
                    "max_leaf_nodes": list(self.grid.hist_gradient_boosting.max_leaf_nodes),
                    "min_samples_leaf": self.grid.hist_gradient_boosting.min_samples_leaf,
                    "l2_regularization": (self.grid.hist_gradient_boosting.l2_regularization),
                },
            },
            "protocol": {
                "primary_amplitudes_px": list(self.protocol.primary_amplitudes_px),
                "diagnostic_amplitudes_px": list(self.protocol.diagnostic_amplitudes_px),
                "primary_severities": list(self.protocol.primary_severities),
                "diagnostic_severities": list(self.protocol.diagnostic_severities),
                "exclude_control_rows_from_fit": (self.protocol.exclude_control_rows_from_fit),
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
        raise GeometricBenchmarkConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise GeometricBenchmarkConfigError(f"missing or invalid [{name}] table")
    return value


def _required(section: Mapping[str, object], key: str, section_name: str) -> object:
    if key not in section:
        raise GeometricBenchmarkConfigError(f"missing required key {section_name}.{key}")
    return section[key]


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeometricBenchmarkConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeometricBenchmarkConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GeometricBenchmarkConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise GeometricBenchmarkConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise GeometricBenchmarkConfigError(f"{label} must be a boolean, got {value!r}")
    return value


def _as_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise GeometricBenchmarkConfigError(f"{label} must be a TOML array")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    items = tuple(_as_str(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise GeometricBenchmarkConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_int_tuple(value: object, label: str) -> tuple[int, ...]:
    items = tuple(_as_int(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise GeometricBenchmarkConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_float_tuple(value: object, label: str) -> tuple[float, ...]:
    items = tuple(_as_float(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise GeometricBenchmarkConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise GeometricBenchmarkConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _parse_run(raw: Mapping[str, object]) -> rb.RunSettings:
    try:
        return rb._parse_run(raw)  # noqa: SLF001 - same frozen protocol grammar
    except rb.RobustnessConfigError as exc:
        raise GeometricBenchmarkConfigError(str(exc)) from exc


def _parse_m3(raw: Mapping[str, object]) -> rb.M3Pin:
    try:
        return rb._parse_m3(raw)  # noqa: SLF001
    except rb.RobustnessConfigError as exc:
        raise GeometricBenchmarkConfigError(str(exc)) from exc


def _parse_model(raw: Mapping[str, object]) -> rb.ModelPin:
    try:
        return rb._parse_model(raw)  # noqa: SLF001
    except rb.RobustnessConfigError as exc:
        raise GeometricBenchmarkConfigError(str(exc)) from exc


def _parse_schema(raw: Mapping[str, object]) -> rb.SchemaSettings:
    try:
        return rb._parse_schema(raw)  # noqa: SLF001
    except rb.RobustnessConfigError as exc:
        raise GeometricBenchmarkConfigError(str(exc)) from exc


def _parse_grid(raw: Mapping[str, object]) -> rb.GridSettings:
    try:
        return rb._parse_grid(raw)  # noqa: SLF001
    except rb.RobustnessConfigError as exc:
        raise GeometricBenchmarkConfigError(str(exc)) from exc


def _parse_m5a(raw: Mapping[str, object]) -> M5APin:
    _check_keys(
        raw,
        {
            "dataset",
            "config",
            "dataset_version",
            "config_sha256",
            "pilot_config_hash",
            "manifest_sha256",
            "sources_sha256",
            "features_sha256",
            "summary_sha256",
            "image_pixel_hash_set",
            "n_samples",
            "n_sources",
            "gate_verdict",
        },
        "m5a",
    )
    n_samples = _as_int(_required(raw, "n_samples", "m5a"), "m5a.n_samples")
    n_sources = _as_int(_required(raw, "n_sources", "m5a"), "m5a.n_sources")
    if n_samples < 1 or n_sources < 1:
        raise GeometricBenchmarkConfigError("m5a.n_samples and m5a.n_sources must be >= 1")
    return M5APin(
        dataset=Path(_as_str(_required(raw, "dataset", "m5a"), "m5a.dataset")),
        config=Path(_as_str(_required(raw, "config", "m5a"), "m5a.config")),
        dataset_version=_as_str(_required(raw, "dataset_version", "m5a"), "m5a.dataset_version"),
        config_sha256=_as_sha256(_required(raw, "config_sha256", "m5a"), "m5a.config_sha256"),
        pilot_config_hash=_as_sha256(
            _required(raw, "pilot_config_hash", "m5a"), "m5a.pilot_config_hash"
        ),
        manifest_sha256=_as_sha256(_required(raw, "manifest_sha256", "m5a"), "m5a.manifest_sha256"),
        sources_sha256=_as_sha256(_required(raw, "sources_sha256", "m5a"), "m5a.sources_sha256"),
        features_sha256=_as_sha256(_required(raw, "features_sha256", "m5a"), "m5a.features_sha256"),
        summary_sha256=_as_sha256(_required(raw, "summary_sha256", "m5a"), "m5a.summary_sha256"),
        image_pixel_hash_set=_as_sha256(
            _required(raw, "image_pixel_hash_set", "m5a"),
            "m5a.image_pixel_hash_set",
        ),
        n_samples=n_samples,
        n_sources=n_sources,
        gate_verdict=_as_str(_required(raw, "gate_verdict", "m5a"), "m5a.gate_verdict"),
    )


def _parse_protocol(raw: Mapping[str, object]) -> ProtocolSettings:
    _check_keys(
        raw,
        {
            "primary_amplitudes_px",
            "diagnostic_amplitudes_px",
            "primary_severities",
            "diagnostic_severities",
            "exclude_control_rows_from_fit",
        },
        "protocol",
    )
    primary = _as_float_tuple(
        _required(raw, "primary_amplitudes_px", "protocol"),
        "protocol.primary_amplitudes_px",
    )
    diagnostic = _as_float_tuple(
        _required(raw, "diagnostic_amplitudes_px", "protocol"),
        "protocol.diagnostic_amplitudes_px",
    )
    primary_severities = _as_int_tuple(
        _required(raw, "primary_severities", "protocol"),
        "protocol.primary_severities",
    )
    diagnostic_severities = _as_int_tuple(
        _required(raw, "diagnostic_severities", "protocol"),
        "protocol.diagnostic_severities",
    )
    exclude_controls = _as_bool(
        _required(raw, "exclude_control_rows_from_fit", "protocol"),
        "protocol.exclude_control_rows_from_fit",
    )
    if primary != PRIMARY_AMPLITUDES_PX:
        raise GeometricBenchmarkConfigError(
            f"protocol.primary_amplitudes_px must be {PRIMARY_AMPLITUDES_PX}"
        )
    if diagnostic != DIAGNOSTIC_AMPLITUDES_PX:
        raise GeometricBenchmarkConfigError(
            f"protocol.diagnostic_amplitudes_px must be {DIAGNOSTIC_AMPLITUDES_PX}"
        )
    if primary_severities != PRIMARY_SEVERITIES:
        raise GeometricBenchmarkConfigError(
            f"protocol.primary_severities must be {PRIMARY_SEVERITIES}"
        )
    if diagnostic_severities != DIAGNOSTIC_SEVERITIES:
        raise GeometricBenchmarkConfigError(
            f"protocol.diagnostic_severities must be {DIAGNOSTIC_SEVERITIES}"
        )
    if set(primary) & set(diagnostic):
        raise GeometricBenchmarkConfigError("primary and diagnostic amplitudes overlap")
    if not exclude_controls:
        raise GeometricBenchmarkConfigError(
            "protocol.exclude_control_rows_from_fit must be true for this benchmark"
        )
    return ProtocolSettings(
        primary_amplitudes_px=primary,
        diagnostic_amplitudes_px=diagnostic,
        primary_severities=primary_severities,
        diagnostic_severities=diagnostic_severities,
        exclude_control_rows_from_fit=exclude_controls,
    )


def load_geometric_benchmark_config(path: Path | str) -> GeometricBenchmarkConfig:
    """Load and strictly validate the M5B TOML configuration."""
    path = Path(path)
    if not path.is_file():
        raise GeometricBenchmarkConfigError(f"geometric benchmark config not found: {path}")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise GeometricBenchmarkConfigError(f"invalid TOML in {path}: {exc}") from exc
    _check_keys(payload, set(_EXPECTED_TOP_KEYS), "geometric benchmark file")
    config = GeometricBenchmarkConfig(
        run=_parse_run(_section(payload, "run")),
        m3=_parse_m3(_section(payload, "m3")),
        m5a=_parse_m5a(_section(payload, "m5a")),
        model=_parse_model(_section(payload, "model")),
        schema=_parse_schema(_section(payload, "schema")),
        grid=_parse_grid(_section(payload, "grid")),
        protocol=_parse_protocol(_section(payload, "protocol")),
        config_path=path,
        config_sha256=sha256_file(path),
    )
    specs = rb.candidate_specs_from_config(config)  # type: ignore[arg-type]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise GeometricBenchmarkConfigError("model grid produces duplicate candidate names")
    return config


# ---------------------------------------------------------------------------
# Dataset loading, merging and isolation checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkFrames:
    """Frozen and leakage-safe frames used by all three experiments."""

    m3_frame: pd.DataFrame
    m5a_frame: pd.DataFrame
    m3_sources: pd.DataFrame
    m5a_sources: pd.DataFrame
    m3_summary: dict[str, Any]
    m5a_summary: dict[str, Any]
    expanded_train: pd.DataFrame
    expanded_val: pd.DataFrame
    primary_test: pd.DataFrame
    diagnostic_test: pd.DataFrame
    m3_test: pd.DataFrame
    data_accounting: pd.DataFrame
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise GeometricBenchmarkError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise GeometricBenchmarkError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def _amplitude_mask(frame: pd.DataFrame, amplitudes: Sequence[float]) -> np.ndarray:
    if "amplitude_px" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    values = frame["amplitude_px"].to_numpy(dtype=np.float64)
    mask = np.zeros(len(frame), dtype=bool)
    for amplitude in amplitudes:
        mask |= np.isclose(values, float(amplitude), rtol=0.0, atol=1e-9)
    return mask


def _normalise_m3(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["dataset"] = M3_LABEL
    result["family"] = result["corruption_type"].astype(str)
    result["is_clean"] = result["family"] == M3_CLEAN_FAMILY
    result["is_control"] = False
    result["is_elastic"] = ~result["is_clean"]
    result["protocol_scope"] = "not_applicable"
    return result


def _normalise_m5a(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["dataset"] = M5A_LABEL
    result["family"] = result["variant_family"].astype(str)
    result["corruption_type"] = result["variant_family"].astype(str)
    result["is_clean"] = result["family"] == CLEAN_FAMILY
    result["is_control"] = result["family"] == CONTROL_FAMILY
    result["is_elastic"] = result["family"] == ELASTIC_FAMILY
    result["protocol_scope"] = "excluded"
    primary = result["is_elastic"] & _amplitude_mask(result, PRIMARY_AMPLITUDES_PX)
    diagnostic = result["is_elastic"] & _amplitude_mask(result, DIAGNOSTIC_AMPLITUDES_PX)
    result.loc[primary, "protocol_scope"] = "primary"
    result.loc[diagnostic, "protocol_scope"] = "diagnostic"
    return result


def _check_source_identity(m3_sources: pd.DataFrame, m5a_sources: pd.DataFrame) -> CheckResult:
    try:
        left = m3_sources.set_index("source_id").sort_index()
        right = m5a_sources.set_index("source_id").sort_index()
        if not left.index.equals(right.index):
            raise GeometricBenchmarkError("M3 and M5A source IDs differ")
        for column in ("split", "font_id", "text_length"):
            if column not in right.columns:
                raise GeometricBenchmarkError(f"M5A sources missing {column!r}")
            if not left[column].equals(right[column]):
                raise GeometricBenchmarkError(f"M3 and M5A sources disagree on {column!r}")
        return CheckResult(
            name="source_identity",
            passed=True,
            detail=(
                f"{len(left)} source IDs match across datasets, with identical "
                "split, font and text-length assignments"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="source_identity",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_frame_source_assignment(
    m3_frame: pd.DataFrame,
    m5a_frame: pd.DataFrame,
    m3_sources: pd.DataFrame,
    m5a_sources: pd.DataFrame,
) -> CheckResult:
    try:
        for label, frame, sources in (
            ("M3", m3_frame, m3_sources),
            ("M5A", m5a_frame, m5a_sources),
        ):
            expected = sources.set_index("source_id").sort_index()
            observed = (
                frame.groupby("source_id")
                .agg(
                    split=("split", lambda values: tuple(sorted(set(values)))),
                    font_id=("font_id", lambda values: tuple(sorted(set(values)))),
                    text_length=("text_length", lambda values: tuple(sorted(set(values)))),
                )
                .sort_index()
            )
            if not observed.index.equals(expected.index):
                raise GeometricBenchmarkError(
                    f"{label} frame source IDs do not match its sources table"
                )
            if (observed["split"].map(len) != 1).any():
                raise GeometricBenchmarkError(f"{label} frame source crosses splits")
            if (observed["font_id"].map(len) != 1).any():
                raise GeometricBenchmarkError(f"{label} frame source has multiple fonts")
            if (observed["text_length"].map(len) != 1).any():
                raise GeometricBenchmarkError(f"{label} frame source has multiple text lengths")
            for source_id, row in observed.iterrows():
                expected_row = expected.loc[source_id]
                if row["split"][0] != str(expected_row["split"]):
                    raise GeometricBenchmarkError(
                        f"{label} frame source {source_id} has split {row['split'][0]!r}, "
                        f"sources.csv says {expected_row['split']!r}"
                    )
                if row["font_id"][0] != str(expected_row["font_id"]):
                    raise GeometricBenchmarkError(
                        f"{label} frame source {source_id} font disagrees with sources.csv"
                    )
                if int(row["text_length"][0]) != int(expected_row["text_length"]):
                    raise GeometricBenchmarkError(
                        f"{label} frame source {source_id} text length disagrees with sources.csv"
                    )
        m3_assignment = m3_sources.set_index("source_id")["split"].sort_index()
        m5a_assignment = m5a_sources.set_index("source_id")["split"].sort_index()
        if not m3_assignment.equals(m5a_assignment):
            raise GeometricBenchmarkError("M3 and M5A source split assignments differ")
        return CheckResult(
            name="frame_source_assignment",
            passed=True,
            detail=(
                "every row's source split, font and text length agree with sources.csv; "
                "M3 and M5A assign all 60 sources to the same split"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="frame_source_assignment",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_clean_duplicate_provenance(
    m3_frame: pd.DataFrame, m5a_frame: pd.DataFrame
) -> CheckResult:
    try:
        m3_clean = m3_frame[m3_frame["is_clean"]].set_index("source_id").sort_index()
        m5a_clean = m5a_frame[m5a_frame["is_clean"]].set_index("source_id").sort_index()
        if not m3_clean.index.equals(m5a_clean.index):
            raise GeometricBenchmarkError("clean source IDs differ between datasets")
        for column in ("image_sha256", "reference_sha256", "ssim_target"):
            if not m3_clean[column].equals(m5a_clean[column]):
                raise GeometricBenchmarkError(
                    f"clean rows disagree on {column!r}; the merge would duplicate or "
                    "silently replace a clean reference"
                )
        for feature in FEATURE_NAMES:
            if not np.array_equal(
                m3_clean[feature].to_numpy(dtype=np.float64),
                m5a_clean[feature].to_numpy(dtype=np.float64),
            ):
                raise GeometricBenchmarkError(f"clean rows disagree on feature {feature!r}")

        controls = m5a_frame[m5a_frame["is_control"]]
        expected_controls = {"identity", "zero_field", "minimal"}
        if set(controls["control_id"]) != expected_controls:
            raise GeometricBenchmarkError(
                f"M5A controls are {sorted(set(controls['control_id']))}, "
                f"expected {sorted(expected_controls)}"
            )
        exact = controls[controls["control_id"].isin(["identity", "zero_field"])]
        if not (exact["image_sha256"] == exact["reference_sha256"]).all():
            raise GeometricBenchmarkError("identity/zero-field controls differ from clean pixels")
        minimal = controls[controls["control_id"] == "minimal"]
        if minimal.empty or (minimal["ssim_target"] >= 1.0).any():
            raise GeometricBenchmarkError("minimal control is missing or exactly clean")
        return CheckResult(
            name="clean_duplicate_provenance",
            passed=True,
            detail=(
                f"{len(m3_clean)} M5A clean rows match M3 clean pixels, targets and "
                "features exactly; 60 identity and 60 zero-field controls are exact "
                "clean duplicates and the 60 minimal controls are near-clean. All 240 "
                "clean/control rows are excluded from expanded train/val and scoring."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="clean_duplicate_provenance",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_severity_protocol(
    m5a_frame: pd.DataFrame,
    expanded_train: pd.DataFrame,
    expanded_val: pd.DataFrame,
    primary_test: pd.DataFrame,
    diagnostic_test: pd.DataFrame,
) -> CheckResult:
    try:
        elastic = m5a_frame[m5a_frame["is_elastic"]]
        mapping = (
            elastic[["severity_level", "amplitude_px"]]
            .drop_duplicates()
            .sort_values("severity_level")
        )
        expected = {
            1: 0.5,
            2: 1.0,
            3: 2.0,
            4: 3.0,
        }
        actual = {
            int(row.severity_level): round(float(row.amplitude_px), 9)
            for row in mapping.itertuples(index=False)
        }
        if actual != expected:
            raise GeometricBenchmarkError(
                f"severity/amplitude mapping is {actual}, expected {expected}"
            )
        fit = pd.concat([expanded_train, expanded_val], ignore_index=True)
        if (fit["dataset"] == M5A_LABEL).any():
            m5a_fit = fit[fit["dataset"] == M5A_LABEL]
            if set(m5a_fit["protocol_scope"]) != {"primary"}:
                raise GeometricBenchmarkError("non-primary M5A rows entered train/val")
            if _amplitude_mask(m5a_fit, DIAGNOSTIC_AMPLITUDES_PX).any():
                raise GeometricBenchmarkError("0.5 px rows entered train/val")
            if set(m5a_fit["severity_level"].astype(int)) & set(DIAGNOSTIC_SEVERITIES):
                raise GeometricBenchmarkError("diagnostic severity entered train/val")
            if m5a_fit["is_clean"].any() or m5a_fit["is_control"].any():
                raise GeometricBenchmarkError("M5A clean/control rows entered train/val")
        if set(primary_test["protocol_scope"]) != {"primary"}:
            raise GeometricBenchmarkError("primary test frame contains non-primary rows")
        if _amplitude_mask(primary_test, DIAGNOSTIC_AMPLITUDES_PX).any():
            raise GeometricBenchmarkError("0.5 px row entered the primary test score")
        if set(diagnostic_test["protocol_scope"]) != {"diagnostic"}:
            raise GeometricBenchmarkError("diagnostic frame contains non-diagnostic rows")
        if not _amplitude_mask(diagnostic_test, DIAGNOSTIC_AMPLITUDES_PX).all():
            raise GeometricBenchmarkError("diagnostic frame is not exactly the 0.5 px rung")
        if set(primary_test["sample_id"]) & set(diagnostic_test["sample_id"]):
            raise GeometricBenchmarkError("primary and diagnostic frames share samples")
        return CheckResult(
            name="severity_protocol",
            passed=True,
            detail=(
                f"primary amplitudes {PRIMARY_AMPLITUDES_PX} at severities "
                f"{PRIMARY_SEVERITIES}; diagnostic {DIAGNOSTIC_AMPLITUDES_PX} at "
                f"severity {DIAGNOSTIC_SEVERITIES}; 0.5 px excluded from all fit, "
                "selection and primary scoring frames"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="severity_protocol",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_merge_contract(
    expanded_train: pd.DataFrame,
    expanded_val: pd.DataFrame,
    primary_test: pd.DataFrame,
    diagnostic_test: pd.DataFrame,
    m3_test: pd.DataFrame,
) -> CheckResult:
    try:
        train_ids = set(expanded_train["sample_id"].astype(str))
        val_ids = set(expanded_val["sample_id"].astype(str))
        primary_ids = set(primary_test["sample_id"].astype(str))
        diagnostic_ids = set(diagnostic_test["sample_id"].astype(str))
        m3_test_ids = set(m3_test["sample_id"].astype(str))
        if len(train_ids) != len(expanded_train):
            raise GeometricBenchmarkError("expanded train has duplicate sample_id values")
        if len(val_ids) != len(expanded_val):
            raise GeometricBenchmarkError("expanded val has duplicate sample_id values")
        if train_ids & val_ids:
            raise GeometricBenchmarkError("expanded train and val share sample_id values")
        if (train_ids | val_ids) & (primary_ids | diagnostic_ids | m3_test_ids):
            raise GeometricBenchmarkError("test sample IDs entered expanded train/val")
        train_hashes = expanded_train["image_sha256"].astype(str)
        val_hashes = expanded_val["image_sha256"].astype(str)
        if train_hashes.duplicated().any():
            raise GeometricBenchmarkError("expanded train has duplicate image hashes")
        if val_hashes.duplicated().any():
            raise GeometricBenchmarkError("expanded val has duplicate image hashes")
        if set(train_hashes) & set(val_hashes):
            raise GeometricBenchmarkError("expanded train and val share image hashes")
        train_sources = set(expanded_train["source_id"].astype(str))
        val_sources = set(expanded_val["source_id"].astype(str))
        test_sources = set(primary_test["source_id"].astype(str))
        diagnostic_sources = set(diagnostic_test["source_id"].astype(str))
        if test_sources != diagnostic_sources:
            raise GeometricBenchmarkError("primary and diagnostic test sources differ")
        if (
            train_sources & val_sources
            or train_sources & test_sources
            or val_sources & test_sources
        ):
            raise GeometricBenchmarkError("source groups overlap across expanded/evaluation sets")
        return CheckResult(
            name="merge_contract",
            passed=True,
            detail=(
                f"train {len(expanded_train)} rows/{len(train_sources)} sources, "
                f"val {len(expanded_val)} rows/{len(val_sources)} sources, "
                f"primary test {len(primary_test)} rows/{len(test_sources)} sources, "
                f"diagnostic test {len(diagnostic_test)} rows/{len(diagnostic_sources)} sources; "
                "no duplicate sample IDs or image hashes and no test-source rows in fit"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="merge_contract",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_feature_leakage(
    m3_features: pd.DataFrame, m5a_features: pd.DataFrame, expanded_train: pd.DataFrame
) -> CheckResult:
    try:
        validate_feature_columns(list(m3_features.columns))
        validate_feature_columns(list(m5a_features.columns))
        inputs = expanded_train[list(FEATURE_NAMES)]
        if inputs.shape[1] != len(FEATURE_NAMES):
            raise GeometricBenchmarkError("expanded training input width is not 8")
        forbidden = {
            TARGET,
            "psnr_target",
            "severity_level",
            "amplitude_px",
            "family",
            "variant_family",
            "control_id",
            "font_id",
            "text_length",
            "source_id",
            "split",
            "dataset",
            "protocol_scope",
            "generation_config_hash",
            "pilot_config_hash",
        }
        leaked = sorted(forbidden & set(inputs.columns))
        if leaked:
            raise GeometricBenchmarkError(f"forbidden columns in model inputs: {leaked}")
        return CheckResult(
            name="feature_leakage",
            passed=True,
            detail=(
                "feature tables expose exactly sample_id plus the eight image features; "
                "primary inputs contain no target, PSNR, severity, amplitude, family, "
                "control, font, length, source, split or provenance metadata"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="feature_leakage",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _accounting_row(dataset: str, split: str, frame: pd.DataFrame) -> dict[str, Any]:
    family = frame["family"].astype(str)
    return {
        "dataset": dataset,
        "split": split,
        "sources": int(frame["source_id"].nunique()),
        "samples": int(len(frame)),
        "clean": int(frame["is_clean"].sum()),
        "controls": int(frame.get("is_control", pd.Series(False, index=frame.index)).sum()),
        "primary_elastic": int((frame["protocol_scope"] == "primary").sum()),
        "diagnostic_elastic": int((frame["protocol_scope"] == "diagnostic").sum()),
        "gaussian_blur": int((family == "gaussian_blur").sum()),
        "gaussian_noise": int((family == "gaussian_noise").sum()),
    }


def _data_accounting(
    m3_frame: pd.DataFrame,
    m5a_frame: pd.DataFrame,
    expanded_train: pd.DataFrame,
    expanded_val: pd.DataFrame,
    primary_test: pd.DataFrame,
    diagnostic_test: pd.DataFrame,
    m3_test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        rows.append(_accounting_row("m3", split, m3_frame[m3_frame["split"] == split]))
        rows.append(_accounting_row("m5a", split, m5a_frame[m5a_frame["split"] == split]))
    rows.append(_accounting_row("expanded", "train", expanded_train))
    rows.append(_accounting_row("expanded", "val", expanded_val))
    rows.append(_accounting_row("evaluation", "m5a_primary_test", primary_test))
    rows.append(_accounting_row("evaluation", "m5a_diagnostic_0_5_test", diagnostic_test))
    rows.append(_accounting_row("evaluation", "m3_test", m3_test))
    return pd.DataFrame(rows)


def build_frame_bundle(
    m3_frame: pd.DataFrame,
    m5a_frame: pd.DataFrame,
    m3_sources: pd.DataFrame,
    m5a_sources: pd.DataFrame,
    m3_summary: Mapping[str, Any],
    m5a_summary: Mapping[str, Any],
) -> BenchmarkFrames:
    """Normalise, audit and merge the frozen and M5A frames.

    This function is deliberately pure with respect to the filesystem. File
    hashes, image verification and target replay happen in
    :func:`load_benchmark_frames` before this merge is used.
    """
    m3 = _normalise_m3(m3_frame)
    m5a = _normalise_m5a(m5a_frame)
    if m3["sample_id"].duplicated().any() or m5a["sample_id"].duplicated().any():
        raise GeometricBenchmarkError("sample_id must be unique within each dataset")
    if m3_sources["source_id"].duplicated().any() or m5a_sources["source_id"].duplicated().any():
        raise GeometricBenchmarkError("source_id must be unique within sources.csv")

    m3_train = m3[m3["split"] == "train"].copy()
    m3_val = m3[m3["split"] == "val"].copy()
    m3_test = m3[m3["split"] == "test"].copy()
    m5a_primary_train = m5a[(m5a["split"] == "train") & (m5a["protocol_scope"] == "primary")].copy()
    m5a_primary_val = m5a[(m5a["split"] == "val") & (m5a["protocol_scope"] == "primary")].copy()
    primary_test = m5a[(m5a["split"] == "test") & (m5a["protocol_scope"] == "primary")].copy()
    diagnostic_test = m5a[(m5a["split"] == "test") & (m5a["protocol_scope"] == "diagnostic")].copy()
    expanded_train = pd.concat([m3_train, m5a_primary_train], ignore_index=True)
    expanded_val = pd.concat([m3_val, m5a_primary_val], ignore_index=True)

    checks = (
        rb._check_source_isolation(m3, "m3"),  # noqa: SLF001
        rb._check_source_isolation(m5a, "m5a"),  # noqa: SLF001
        _check_source_identity(m3_sources, m5a_sources),
        _check_frame_source_assignment(m3, m5a, m3_sources, m5a_sources),
        _check_clean_duplicate_provenance(m3, m5a),
        _check_severity_protocol(m5a, expanded_train, expanded_val, primary_test, diagnostic_test),
        _check_merge_contract(expanded_train, expanded_val, primary_test, diagnostic_test, m3_test),
        _check_feature_leakage(
            m3[["sample_id", *FEATURE_NAMES]],
            m5a[["sample_id", *FEATURE_NAMES]],
            expanded_train,
        ),
    )
    accounting = _data_accounting(
        m3, m5a, expanded_train, expanded_val, primary_test, diagnostic_test, m3_test
    )
    return BenchmarkFrames(
        m3_frame=m3,
        m5a_frame=m5a,
        m3_sources=m3_sources.copy(),
        m5a_sources=m5a_sources.copy(),
        m3_summary=dict(m3_summary),
        m5a_summary=dict(m5a_summary),
        expanded_train=expanded_train,
        expanded_val=expanded_val,
        primary_test=primary_test,
        diagnostic_test=diagnostic_test,
        m3_test=m3_test,
        data_accounting=accounting,
        checks=checks,
    )


def load_benchmark_frames(
    config: GeometricBenchmarkConfig, *, verify_images: bool = True
) -> BenchmarkFrames:
    """Load pinned M3/M5A inputs, verify provenance and build the merged frames."""
    if not config.m3.config.is_file():
        raise GeometricBenchmarkError(f"missing M3 dataset config: {config.m3.config}")
    _require_file_hash(config.m3.dataset / "manifest.csv", config.m3.manifest_sha256, "M3 manifest")
    _require_file_hash(config.m3.dataset / "sources.csv", config.m3.sources_sha256, "M3 sources")
    _require_file_hash(config.m3.dataset / "features.csv", config.m3.features_sha256, "M3 features")
    _require_file_hash(
        config.m3.dataset / "ground_truth.csv",
        config.m3.ground_truth_sha256,
        "M3 ground truth",
    )
    _require_file_hash(
        config.m3.dataset / "dataset_summary.json", config.m3.summary_sha256, "M3 summary"
    )
    _require_file_hash(config.m5a.config, config.m5a.config_sha256, "M5A config")
    _require_file_hash(
        config.m5a.dataset / "manifest.csv", config.m5a.manifest_sha256, "M5A manifest"
    )
    _require_file_hash(config.m5a.dataset / "sources.csv", config.m5a.sources_sha256, "M5A sources")
    _require_file_hash(
        config.m5a.dataset / "features.csv", config.m5a.features_sha256, "M5A features"
    )
    _require_file_hash(
        config.m5a.dataset / "pilot_summary.json",
        config.m5a.summary_sha256,
        "M5A summary",
    )
    _require_file_hash(
        config.model.model_dir / f"{config.model.model_stem}_card.json",
        config.model.card_sha256,
        "M3 model card",
    )
    _require_file_hash(
        config.model.model_dir / f"{config.model.model_stem}.joblib",
        config.model.artifact_sha256,
        "M3 model artifact",
    )

    m3_bundle = load_benchmark_dataset(config.m3.dataset)
    m5a_manifest = pd.read_csv(config.m5a.dataset / "manifest.csv")
    m5a_sources = pd.read_csv(config.m5a.dataset / "sources.csv")
    m5a_features = pd.read_csv(config.m5a.dataset / "features.csv")
    m5a_summary = json.loads(
        (config.m5a.dataset / "pilot_summary.json").read_text(encoding="utf-8")
    )
    validate_feature_columns(list(m5a_features.columns))
    if len(m5a_manifest) != config.m5a.n_samples:
        raise GeometricBenchmarkError(
            f"M5A manifest has {len(m5a_manifest)} rows, expected {config.m5a.n_samples}"
        )
    if m5a_sources["source_id"].nunique() != config.m5a.n_sources:
        raise GeometricBenchmarkError(
            f"M5A has {m5a_sources['source_id'].nunique()} sources, expected {config.m5a.n_sources}"
        )
    if m5a_summary.get("pilot_config_hash") != config.m5a.pilot_config_hash:
        raise GeometricBenchmarkError("M5A pilot_summary config hash does not match the pin")
    if m5a_summary.get("feature_names") != list(FEATURE_NAMES):
        raise GeometricBenchmarkError("M5A summary feature names do not match the extractor")
    if (
        m5a_summary.get("artifacts", {}).get("image_pixel_hash_set")
        != config.m5a.image_pixel_hash_set
    ):
        raise GeometricBenchmarkError("M5A image pixel hash set does not match the pin")
    if m5a_summary.get("pilot_version") != config.m5a.dataset_version:
        raise GeometricBenchmarkError("M5A pilot version does not match the pin")
    if m5a_summary.get("target", {}).get("name") != TARGET:
        raise GeometricBenchmarkError("M5A target does not match ssim_target")
    if m5a_summary.get("gate", {}).get("verdict") != config.m5a.gate_verdict:
        raise GeometricBenchmarkError("M5A gate verdict does not match the benchmark pin")
    if m5a_manifest["pilot_config_hash"].nunique() != 1:
        raise GeometricBenchmarkError("M5A manifest has multiple pilot_config_hash values")
    if m5a_manifest["pilot_config_hash"].iloc[0] != config.m5a.pilot_config_hash:
        raise GeometricBenchmarkError("M5A manifest config hash does not match the pin")

    # The M5A manifest carries a diagnostic column named ``intensity_std`` that
    # is not the feature of the same name. Drop manifest copies of feature names
    # before the join so the feature table remains the sole feature source.
    manifest_for_merge = m5a_manifest.drop(
        columns=[name for name in FEATURE_NAMES if name in m5a_manifest.columns]
    )
    m5a_frame = manifest_for_merge.merge(
        m5a_features, on="sample_id", how="left", validate="one_to_one"
    )
    if len(m5a_frame) != len(m5a_manifest) or m5a_frame[list(FEATURE_NAMES)].isna().any().any():
        raise GeometricBenchmarkError("M5A manifest and feature rows do not align")
    if not np.isfinite(m5a_frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)).all():
        raise GeometricBenchmarkError("M5A features contain non-finite values")
    if not np.isfinite(m5a_frame[TARGET].to_numpy(dtype=np.float64)).all():
        raise GeometricBenchmarkError("M5A target contains non-finite values")
    if not m5a_frame[TARGET].between(0.0, 1.0).all():
        raise GeometricBenchmarkError("M5A ssim_target is outside [0, 1]")

    if verify_images:
        m3_report = verify_dataset(config.m3.dataset, recompute_targets=True)
        rb._verify_checks(m3_report.checks, "M3 dataset verification")  # noqa: SLF001
        m5a_report = verify_pilot_dataset(
            config.m5a.dataset, reference=None, recompute_targets=True
        )
        rb._verify_checks(m5a_report.checks, "M5A dataset verification")  # noqa: SLF001

    frames = build_frame_bundle(
        m3_bundle.frame,
        m5a_frame,
        m3_bundle.sources,
        m5a_sources,
        m3_bundle.summary,
        m5a_summary,
    )
    rb._verify_checks(frames.checks, "M3/M5A merge checks")  # noqa: SLF001
    return frames


def load_frozen_model(
    config: GeometricBenchmarkConfig, m3_summary: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], tuple[CheckResult, ...]]:
    """Load the frozen M3 primary bundle through the existing verifier."""
    return load_model_bundle(
        config.model.model_dir,
        config.model.model_stem,
        dataset_summary=dict(m3_summary),
    )


# ---------------------------------------------------------------------------
# Metrics, adaptation and grouped uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeometricBenchmarkResult:
    frames: BenchmarkFrames
    checks: tuple[CheckResult, ...]
    frozen_card: dict[str, Any]
    zero_shot_metrics: pd.DataFrame
    zero_shot_residuals: pd.DataFrame
    zero_shot_predictions: pd.DataFrame
    m3_reference_metrics: pd.DataFrame
    candidates: pd.DataFrame
    selected: dict[str, Any]
    selected_model: FittedModel
    adaptation_metrics: pd.DataFrame
    adaptation_residuals: pd.DataFrame
    adaptation_predictions: pd.DataFrame
    m3_retention_metrics: pd.DataFrame
    m3_retention_predictions: pd.DataFrame
    retention_comparison: pd.DataFrame
    target_distribution: pd.DataFrame
    uncertainty: pd.DataFrame

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)


def _fit_naive(train: pd.DataFrame, *, seed: int) -> FittedModel:
    return fit_candidate(naive_spec(), train, seed=seed)


def _candidate_table(rows: list[dict[str, Any]], selected_name: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "model": row["name"],
                "family": row["family"],
                "params": json.dumps(row["params"], sort_keys=True, separators=(",", ":")),
                "n_inputs": row["n_inputs"],
                "train_mae": row["train_mae"],
                "train_rmse": row["train_rmse"],
                "train_r2": row["train_r2"],
                "train_n": row["train_n"],
                "val_mae": row["val_mae"],
                "val_rmse": row["val_rmse"],
                "val_r2": row["val_r2"],
                "val_n": row["val_n"],
                "val_n_sources": row["val_n_sources"],
                "selected": row["name"] == selected_name,
                "selected_on": "val",
            }
        )
    return pd.DataFrame(records)


def fit_and_select(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: GeometricBenchmarkConfig,
) -> tuple[pd.DataFrame, dict[str, FittedModel], dict[str, Any], FittedModel]:
    """Fit the predeclared grid on train and select on validation only."""
    specs = rb.candidate_specs_from_config(config)  # type: ignore[arg-type]
    rows: list[dict[str, Any]] = []
    fitted: dict[str, FittedModel] = {}
    for spec in specs:
        seed = derive_noise_seed(config.run.seed, "model", spec.name) % (2**32)
        model = fit_candidate(spec, train, seed=seed)
        train_scores = score_frame(train, predict(model.estimator, train))
        val_scores = score_frame(val, predict(model.estimator, val))
        fitted[spec.name] = model
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "params": spec.params,
                "n_inputs": spec.n_inputs,
                "train_mae": train_scores["mae"],
                "train_rmse": train_scores["rmse"],
                "train_r2": train_scores["r2"],
                "train_n": train_scores["n_rows"],
                "val_mae": val_scores["mae"],
                "val_rmse": val_scores["rmse"],
                "val_r2": val_scores["r2"],
                "val_n": val_scores["n_rows"],
                "val_n_sources": val_scores["n_sources"],
            }
        )
    selected = select_best(rows, config.run.selection_metric)
    selected_model = fitted[str(selected["name"])]
    return _candidate_table(rows, str(selected["name"])), fitted, selected, selected_model


def _bootstrap_row(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    model: str,
    dataset: str,
    scope: str,
    level: str,
    n_bootstrap: int,
    seed: int,
    evaluation: str,
) -> pd.DataFrame:
    table = rb._bootstrap_rows(  # noqa: SLF001
        frame,
        predictions,
        model=model,
        dataset=dataset,
        scope=scope,
        level=level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    table["evaluation"] = evaluation
    table["comparison_scope"] = ""
    return table


def _paired_row(
    frame: pd.DataFrame,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    *,
    model_a: str,
    model_b: str,
    dataset: str,
    scope: str,
    level: str,
    n_bootstrap: int,
    seed: int,
    evaluation: str,
) -> pd.DataFrame:
    table = rb.paired_grouped_bootstrap(
        frame,
        predictions_a,
        predictions_b,
        model_a=model_a,
        model_b=model_b,
        dataset=dataset,
        scope=scope,
        level=level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    table["evaluation"] = evaluation
    table["comparison_scope"] = evaluation
    return table


def _uncertainty_table(
    frames: BenchmarkFrames,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    model_names = (
        FROZEN_MODEL_NAME,
        M3_NAIVE_NAME,
        EXPANDED_MODEL_NAME,
        EXPANDED_NAIVE_NAME,
    )
    primary = frames.primary_test
    diagnostic = frames.diagnostic_test
    m3_test = frames.m3_test
    primary_scopes = [
        ("overall", "all"),
        ("severity", "2"),
        ("severity", "3"),
        ("severity", "4"),
    ]
    primary_scopes.extend(("font", font) for font in sorted(primary["font_id"].unique()))
    diagnostic_scopes = [("overall", "all"), ("severity", "1")]
    m3_scopes = [
        ("overall", "all"),
        ("clean", "clean"),
        ("corrupted", "all"),
        ("family", "gaussian_blur"),
        ("family", "gaussian_noise"),
    ]
    for model in model_names:
        for scope, level in primary_scopes:
            rows.append(
                _bootstrap_row(
                    primary,
                    predictions["primary"][model],
                    model=model,
                    dataset=PRIMARY_LABEL,
                    scope=scope,
                    level=level,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    evaluation="primary_geometric",
                )
            )
        for scope, level in diagnostic_scopes:
            rows.append(
                _bootstrap_row(
                    diagnostic,
                    predictions["diagnostic"][model],
                    model=model,
                    dataset=DIAGNOSTIC_LABEL,
                    scope=scope,
                    level=level,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    evaluation="diagnostic_0_5",
                )
            )
        for scope, level in m3_scopes:
            rows.append(
                _bootstrap_row(
                    m3_test,
                    predictions["m3_test"][model],
                    model=model,
                    dataset=M3_TEST_LABEL,
                    scope=scope,
                    level=level,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    evaluation="m3_retention",
                )
            )
    for model_a, model_b, _label in (
        (EXPANDED_MODEL_NAME, FROZEN_MODEL_NAME, "adapted_minus_frozen"),
        (FROZEN_MODEL_NAME, M3_NAIVE_NAME, "frozen_minus_m3_naive"),
        (EXPANDED_MODEL_NAME, EXPANDED_NAIVE_NAME, "adapted_minus_expanded_naive"),
    ):
        for scope, level in primary_scopes:
            rows.append(
                _paired_row(
                    primary,
                    predictions["primary"][model_a],
                    predictions["primary"][model_b],
                    model_a=model_a,
                    model_b=model_b,
                    dataset=PRIMARY_LABEL,
                    scope=scope,
                    level=level,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    evaluation="primary_geometric",
                )
            )
        for scope, level in diagnostic_scopes:
            rows.append(
                _paired_row(
                    diagnostic,
                    predictions["diagnostic"][model_a],
                    predictions["diagnostic"][model_b],
                    model_a=model_a,
                    model_b=model_b,
                    dataset=DIAGNOSTIC_LABEL,
                    scope=scope,
                    level=level,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    evaluation="diagnostic_0_5",
                )
            )
    for scope, level in m3_scopes:
        rows.append(
            _paired_row(
                m3_test,
                predictions["m3_test"][EXPANDED_MODEL_NAME],
                predictions["m3_test"][FROZEN_MODEL_NAME],
                model_a=EXPANDED_MODEL_NAME,
                model_b=FROZEN_MODEL_NAME,
                dataset=M3_TEST_LABEL,
                scope=scope,
                level=level,
                n_bootstrap=n_bootstrap,
                seed=seed,
                evaluation="m3_retention",
            )
        )
    return pd.concat(rows, ignore_index=True)


def _target_distribution(frames: BenchmarkFrames) -> pd.DataFrame:
    scopes: list[tuple[str, str, str, pd.DataFrame]] = []
    for split in ("train", "val", "test"):
        scopes.append(("m3", split, "all", frames.m3_frame[frames.m3_frame["split"] == split]))
        scopes.append(("m5a", split, "all", frames.m5a_frame[frames.m5a_frame["split"] == split]))
    scopes.append(("expanded", "train", "all", frames.expanded_train))
    scopes.append(("expanded", "val", "all", frames.expanded_val))
    scopes.append(("evaluation", "primary_test", "all", frames.primary_test))
    scopes.append(("evaluation", "diagnostic_0_5_test", "all", frames.diagnostic_test))
    scopes.append(("evaluation", "m3_test", "all", frames.m3_test))
    rows: list[dict[str, Any]] = []
    for dataset, split, scope, frame in scopes:
        for family in sorted(frame["family"].astype(str).unique()):
            for severity in sorted(frame["severity_level"].astype(int).unique()):
                group = frame[(frame["family"] == family) & (frame["severity_level"] == severity)]
                if group.empty:
                    continue
                values = group[TARGET].to_numpy(dtype=np.float64)
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "scope": scope,
                        "family": family,
                        "severity_level": int(severity),
                        "amplitude_px": (
                            float(group["amplitude_px"].iloc[0])
                            if "amplitude_px" in group.columns
                            and math.isfinite(float(group["amplitude_px"].iloc[0]))
                            else None
                        ),
                        "n_samples": int(len(group)),
                        "n_sources": int(group["source_id"].nunique()),
                        "mean": float(np.mean(values)),
                        "sd": (float(np.std(values, ddof=1)) if len(values) > 1 else math.nan),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                )
    return pd.DataFrame(rows)


def _retention_comparison(metrics: pd.DataFrame, config: GeometricBenchmarkConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, level in (
        ("overall", "all"),
        ("clean", "clean"),
        ("corrupted", "all"),
        ("family", "gaussian_blur"),
        ("family", "gaussian_noise"),
    ):
        frozen = rb._metric_lookup(  # noqa: SLF001
            metrics, FROZEN_MODEL_NAME, M3_TEST_LABEL, scope, level
        )
        adapted = rb._metric_lookup(  # noqa: SLF001
            metrics, EXPANDED_MODEL_NAME, M3_TEST_LABEL, scope, level
        )
        rows.append(
            {
                "dataset": M3_TEST_LABEL,
                "scope": scope,
                "level": level,
                "n_samples": int(frozen["n_samples"]),
                "n_sources": int(frozen["n_sources"]),
                "frozen_mae": float(frozen["mae"]),
                "adapted_mae": float(adapted["mae"]),
                "mae_delta": float(adapted["mae"]) - float(frozen["mae"]),
                "frozen_rmse": float(frozen["rmse"]),
                "adapted_rmse": float(adapted["rmse"]),
                "rmse_delta": float(adapted["rmse"]) - float(frozen["rmse"]),
                "frozen_r2": float(frozen["r2"]),
                "adapted_r2": float(adapted["r2"]),
                "selection_metric": config.run.selection_metric,
            }
        )
    return pd.DataFrame(rows)


def run_benchmark_core(
    frames: BenchmarkFrames,
    config: GeometricBenchmarkConfig,
    *,
    frozen_estimator: Any,
    frozen_card: Mapping[str, Any],
) -> GeometricBenchmarkResult:
    """Run all three experiments and return in-memory tables (no writes)."""
    rb._verify_checks(frames.checks, "M3/M5A merge checks")  # noqa: SLF001
    train = frames.expanded_train
    val = frames.expanded_val
    primary = frames.primary_test
    diagnostic = frames.diagnostic_test
    m3_test = frames.m3_test
    for held_out in (primary, diagnostic, m3_test):
        if set(train["source_id"]) & set(held_out["source_id"]):
            raise GeometricBenchmarkError("test sources entered expanded training")
        if set(val["source_id"]) & set(held_out["source_id"]):
            raise GeometricBenchmarkError("test sources entered expanded validation")
    if set(train["split"]) != {"train"} or set(val["split"]) != {"val"}:
        raise GeometricBenchmarkError("expanded fit/selection frames have wrong split labels")

    m3_train = frames.m3_frame[frames.m3_frame["split"] == "train"]
    m3_naive = _fit_naive(
        m3_train, seed=derive_noise_seed(config.run.seed, "naive", "m3") % (2**32)
    )
    frozen_primary = predict(frozen_estimator, primary)
    frozen_diagnostic = predict(frozen_estimator, diagnostic)
    frozen_m3 = predict(frozen_estimator, m3_test)
    m3_naive_primary = predict(m3_naive.estimator, primary)
    m3_naive_diagnostic = predict(m3_naive.estimator, diagnostic)
    m3_naive_m3 = predict(m3_naive.estimator, m3_test)

    zero_shot = rb._concat_evaluations(  # noqa: SLF001
        [
            rb._model_eval(  # noqa: SLF001
                primary,
                frozen_estimator,
                model=FROZEN_MODEL_NAME,
                dataset=PRIMARY_LABEL,
                split="test",
            ),
            rb._model_eval(  # noqa: SLF001
                primary,
                m3_naive.estimator,
                model=M3_NAIVE_NAME,
                dataset=PRIMARY_LABEL,
                split="test",
            ),
            rb._model_eval(  # noqa: SLF001
                diagnostic,
                frozen_estimator,
                model=FROZEN_MODEL_NAME,
                dataset=DIAGNOSTIC_LABEL,
                split="test",
            ),
            rb._model_eval(  # noqa: SLF001
                diagnostic,
                m3_naive.estimator,
                model=M3_NAIVE_NAME,
                dataset=DIAGNOSTIC_LABEL,
                split="test",
            ),
        ]
    )
    m3_reference = rb._concat_evaluations(  # noqa: SLF001
        [
            rb._model_eval(  # noqa: SLF001
                m3_test,
                frozen_estimator,
                model=FROZEN_MODEL_NAME,
                dataset=M3_TEST_LABEL,
                split="test",
            ),
            rb._model_eval(  # noqa: SLF001
                m3_test,
                m3_naive.estimator,
                model=M3_NAIVE_NAME,
                dataset=M3_TEST_LABEL,
                split="test",
            ),
        ]
    ).metrics

    candidates, _, selected, selected_model = fit_and_select(train, val, config)
    expanded_naive = _fit_naive(
        train, seed=derive_noise_seed(config.run.seed, "naive", "expanded") % (2**32)
    )
    adapted_primary = predict(selected_model.estimator, primary)
    adapted_diagnostic = predict(selected_model.estimator, diagnostic)
    adapted_m3 = predict(selected_model.estimator, m3_test)
    expanded_naive_primary = predict(expanded_naive.estimator, primary)
    expanded_naive_diagnostic = predict(expanded_naive.estimator, diagnostic)
    expanded_naive_m3 = predict(expanded_naive.estimator, m3_test)

    adaptation = rb._concat_evaluations(  # noqa: SLF001
        [
            rb._model_eval(  # noqa: SLF001
                primary,
                frozen_estimator,
                model=FROZEN_MODEL_NAME,
                dataset=PRIMARY_LABEL,
                split="test",
            ),
            rb._model_eval(  # noqa: SLF001
                primary,
                selected_model.estimator,
                model=EXPANDED_MODEL_NAME,
                dataset=PRIMARY_LABEL,
                split="test",
            ),
            rb._model_eval(  # noqa: SLF001
                primary,
                expanded_naive.estimator,
                model=EXPANDED_NAIVE_NAME,
                dataset=PRIMARY_LABEL,
                split="test",
            ),
            rb._model_eval(  # noqa: SLF001
                diagnostic,
                frozen_estimator,
                model=FROZEN_MODEL_NAME,
                dataset=DIAGNOSTIC_LABEL,
                split="test",
            ),
            rb._model_eval(  # noqa: SLF001
                diagnostic,
                selected_model.estimator,
                model=EXPANDED_MODEL_NAME,
                dataset=DIAGNOSTIC_LABEL,
                split="test",
            ),
            rb._model_eval(  # noqa: SLF001
                diagnostic,
                expanded_naive.estimator,
                model=EXPANDED_NAIVE_NAME,
                dataset=DIAGNOSTIC_LABEL,
                split="test",
            ),
        ]
    )

    m3_reference = pd.concat(
        [
            m3_reference,
            rb._model_eval(  # noqa: SLF001
                m3_test,
                selected_model.estimator,
                model=EXPANDED_MODEL_NAME,
                dataset=M3_TEST_LABEL,
                split="test",
            ).metrics,
            rb._model_eval(  # noqa: SLF001
                m3_test,
                expanded_naive.estimator,
                model=EXPANDED_NAIVE_NAME,
                dataset=M3_TEST_LABEL,
                split="test",
            ).metrics,
        ],
        ignore_index=True,
    )
    m3_retention = m3_reference[
        m3_reference["model"].isin(
            (FROZEN_MODEL_NAME, EXPANDED_MODEL_NAME, M3_NAIVE_NAME, EXPANDED_NAIVE_NAME)
        )
    ].copy()
    retention_predictions = pd.concat(
        [
            rb._model_eval(  # noqa: SLF001
                m3_test,
                frozen_estimator,
                model=FROZEN_MODEL_NAME,
                dataset=M3_TEST_LABEL,
                split="test",
            ).predictions,
            rb._model_eval(  # noqa: SLF001
                m3_test,
                selected_model.estimator,
                model=EXPANDED_MODEL_NAME,
                dataset=M3_TEST_LABEL,
                split="test",
            ).predictions,
            rb._model_eval(  # noqa: SLF001
                m3_test,
                m3_naive.estimator,
                model=M3_NAIVE_NAME,
                dataset=M3_TEST_LABEL,
                split="test",
            ).predictions,
            rb._model_eval(  # noqa: SLF001
                m3_test,
                expanded_naive.estimator,
                model=EXPANDED_NAIVE_NAME,
                dataset=M3_TEST_LABEL,
                split="test",
            ).predictions,
        ],
        ignore_index=True,
    )

    predictions = {
        "primary": {
            FROZEN_MODEL_NAME: frozen_primary,
            M3_NAIVE_NAME: m3_naive_primary,
            EXPANDED_MODEL_NAME: adapted_primary,
            EXPANDED_NAIVE_NAME: expanded_naive_primary,
        },
        "diagnostic": {
            FROZEN_MODEL_NAME: frozen_diagnostic,
            M3_NAIVE_NAME: m3_naive_diagnostic,
            EXPANDED_MODEL_NAME: adapted_diagnostic,
            EXPANDED_NAIVE_NAME: expanded_naive_diagnostic,
        },
        "m3_test": {
            FROZEN_MODEL_NAME: frozen_m3,
            M3_NAIVE_NAME: m3_naive_m3,
            EXPANDED_MODEL_NAME: adapted_m3,
            EXPANDED_NAIVE_NAME: expanded_naive_m3,
        },
    }
    uncertainty = _uncertainty_table(
        frames,
        predictions,
        n_bootstrap=config.run.n_bootstrap,
        seed=config.run.seed,
    )
    return GeometricBenchmarkResult(
        frames=frames,
        checks=frames.checks,
        frozen_card=dict(frozen_card),
        zero_shot_metrics=zero_shot.metrics,
        zero_shot_residuals=zero_shot.residuals,
        zero_shot_predictions=zero_shot.predictions,
        m3_reference_metrics=m3_reference,
        candidates=candidates,
        selected=selected,
        selected_model=selected_model,
        adaptation_metrics=adaptation.metrics,
        adaptation_residuals=adaptation.residuals,
        adaptation_predictions=adaptation.predictions,
        m3_retention_metrics=m3_retention,
        m3_retention_predictions=retention_predictions,
        retention_comparison=_retention_comparison(m3_retention, config),
        target_distribution=_target_distribution(frames),
        uncertainty=uncertainty,
    )


# ---------------------------------------------------------------------------
# Report and persistence
# ---------------------------------------------------------------------------


def _safe_output_path(path: Path) -> Path:
    resolved = path.resolve()
    for protected_text in PROTECTED_OUTPUT_ROOTS:
        protected = Path(protected_text).resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise GeometricBenchmarkError(
                f"refusing to write into protected frozen path: {path} ({protected_text})"
            )
    return resolved


def _expanded_model_card(
    config: GeometricBenchmarkConfig, result: GeometricBenchmarkResult
) -> dict[str, Any]:
    train = result.frames.expanded_train
    val = result.frames.expanded_val
    selected_row = result.candidates[result.candidates["selected"]].iloc[0]
    primary = rb._metric_lookup(  # noqa: SLF001
        result.adaptation_metrics, EXPANDED_MODEL_NAME, PRIMARY_LABEL, "overall", "all"
    )
    diagnostic = rb._metric_lookup(  # noqa: SLF001
        result.adaptation_metrics, EXPANDED_MODEL_NAME, DIAGNOSTIC_LABEL, "overall", "all"
    )
    retention = rb._metric_lookup(  # noqa: SLF001
        result.m3_retention_metrics, EXPANDED_MODEL_NAME, M3_TEST_LABEL, "overall", "all"
    )
    metrics: dict[str, dict[str, Any]] = {
        "train": {
            "n_rows": int(len(train)),
            "n_sources": int(train["source_id"].nunique()),
            "mae": float(selected_row["train_mae"]),
            "rmse": float(selected_row["train_rmse"]),
            "r2": float(selected_row["train_r2"]),
            "r2_defined": math.isfinite(float(selected_row["train_r2"])),
        },
        "val": {
            "n_rows": int(len(val)),
            "n_sources": int(val["source_id"].nunique()),
            "mae": float(selected_row["val_mae"]),
            "rmse": float(selected_row["val_rmse"]),
            "r2": float(selected_row["val_r2"]),
            "r2_defined": math.isfinite(float(selected_row["val_r2"])),
        },
        "test_geometric_primary": {
            "n_rows": int(primary["n_samples"]),
            "n_sources": int(primary["n_sources"]),
            "mae": float(primary["mae"]),
            "rmse": float(primary["rmse"]),
            "r2": float(primary["r2"]),
            "r2_defined": bool(primary["r2_defined"]),
        },
        "test_geometric_diagnostic_0_5": {
            "n_rows": int(diagnostic["n_samples"]),
            "n_sources": int(diagnostic["n_sources"]),
            "mae": float(diagnostic["mae"]),
            "rmse": float(diagnostic["rmse"]),
            "r2": float(diagnostic["r2"]),
            "r2_defined": bool(diagnostic["r2_defined"]),
        },
        "test_m3_retention": {
            "n_rows": int(retention["n_samples"]),
            "n_sources": int(retention["n_sources"]),
            "mae": float(retention["mae"]),
            "rmse": float(retention["rmse"]),
            "r2": float(retention["r2"]),
            "r2_defined": bool(retention["r2_defined"]),
        },
    }
    return {
        "model_name": EXPANDED_MODEL_NAME,
        "model_family": result.selected["family"],
        "params": result.selected["params"],
        "target": TARGET,
        "feature_names": list(FEATURE_NAMES),
        "feature_set_version": FEATURE_SET_VERSION,
        "n_features": len(FEATURE_NAMES),
        "preprocessing": (
            "standard_scaler_fit_on_expanded_training_rows"
            if result.selected["family"] in {"ols", "ridge"}
            else "none"
        ),
        "dataset": {
            "dataset_name": "captcha-quality-expanded-geometric-distortion",
            "dataset_version": config.run.version,
            "feature_set_version": FEATURE_SET_VERSION,
            "generation_config_hash": config.m3.generation_config_hash,
            "split_assignment_sha256": config.m3.split_assignment_sha256,
            "target": TARGET,
            "m3_manifest_sha256": config.m3.manifest_sha256,
            "m5a_manifest_sha256": config.m5a.manifest_sha256,
            "m5a_pilot_config_hash": config.m5a.pilot_config_hash,
            "m5a_image_pixel_hash_set": config.m5a.image_pixel_hash_set,
            "expanded_train_samples": int(len(train)),
            "expanded_val_samples": int(len(val)),
            "excluded_rows": (
                "M5A clean and control rows (duplicate/near-duplicate M3 clean "
                "references) and the entire 0.5 px diagnostic rung"
            ),
            "primary_amplitudes_px": list(config.protocol.primary_amplitudes_px),
            "diagnostic_amplitudes_px": list(config.protocol.diagnostic_amplitudes_px),
            "primary_severities": list(config.protocol.primary_severities),
            "diagnostic_severities": list(config.protocol.diagnostic_severities),
        },
        "training_scope": "expanded_train_sources_m3_plus_m5a_primary_geometric",
        "training_fonts": sorted(train["font_id"].astype(str).unique().tolist()),
        "selection": {
            "criterion": f"val_{config.run.selection_metric}",
            "selected_on": "val",
            "n_candidates": int(len(result.candidates)),
            "benchmark_config_hash": config.config_hash,
        },
        "metrics": metrics,
        "package_version": package_version(),
        "sklearn_version": sklearn_version(),
        "code_revision": code_revision(),
    }


def _split_assignment_table(frames: BenchmarkFrames) -> pd.DataFrame:
    """Explicit source-to-split assignment and row counts for the run metadata."""
    sources = frames.m5a_sources[["source_id", "split", "font_id", "text_length"]].copy()
    m3_counts = frames.m3_frame.groupby("source_id").size().rename("m3_samples").reset_index()
    m5a_counts = (
        frames.m5a_frame.assign(
            primary=lambda frame: (frame["protocol_scope"] == "primary").astype(int),
            diagnostic=lambda frame: (frame["protocol_scope"] == "diagnostic").astype(int),
            clean_controls=lambda frame: (frame["is_clean"] | frame["is_control"]).astype(int),
        )
        .groupby("source_id")
        .agg(
            m5a_samples=("sample_id", "size"),
            m5a_primary=("primary", "sum"),
            m5a_diagnostic=("diagnostic", "sum"),
            m5a_clean_controls=("clean_controls", "sum"),
        )
        .reset_index()
    )
    table = sources.merge(m3_counts, on="source_id", how="left").merge(
        m5a_counts, on="source_id", how="left"
    )
    order = {"train": 0, "val": 1, "test": 2}
    table["_order"] = table["split"].map(order)
    table = table.sort_values(["_order", "source_id"]).drop(columns="_order")
    return table.reset_index(drop=True)


def write_benchmark_outputs(
    result: GeometricBenchmarkResult,
    config: GeometricBenchmarkConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    frozen_before: Mapping[str, Mapping[str, object]] | None = None,
    frozen_after: Mapping[str, Mapping[str, object]] | None = None,
    elapsed_seconds: float | None = None,
) -> tuple[Path, Path, dict[str, Path]]:
    """Write tables, the report, the model bundle and run metadata."""
    reports = Path(output_reports) if output_reports is not None else config.run.output_reports
    models = Path(output_models) if output_models is not None else config.run.output_models
    _safe_output_path(reports)
    _safe_output_path(models)
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    tables = {
        "data_accounting.csv": result.frames.data_accounting,
        "split_assignment.csv": _split_assignment_table(result.frames),
        "isolation_checks.csv": pd.DataFrame(
            [
                {"check": check.name, "passed": check.passed, "detail": check.detail}
                for check in result.checks
            ]
        ),
        "zero_shot_metrics.csv": result.zero_shot_metrics,
        "zero_shot_residuals.csv": result.zero_shot_residuals,
        "zero_shot_predictions.csv": result.zero_shot_predictions,
        "m3_reference_metrics.csv": result.m3_reference_metrics,
        "candidate_validation.csv": result.candidates,
        "adaptation_metrics.csv": result.adaptation_metrics,
        "adaptation_residuals.csv": result.adaptation_residuals,
        "adaptation_predictions.csv": result.adaptation_predictions,
        "m3_retention_metrics.csv": result.m3_retention_metrics,
        "m3_retention_predictions.csv": result.m3_retention_predictions,
        "retention_comparison.csv": result.retention_comparison,
        "target_distribution.csv": result.target_distribution,
        "uncertainty.csv": result.uncertainty,
    }
    for filename, table in tables.items():
        rb._write_csv(table, reports / filename)  # noqa: SLF001

    report = _report_markdown(config, result)
    report_path = reports / "benchmark.md"
    report_path.write_text(report, encoding="utf-8")

    card = _expanded_model_card(config, result)
    paths = save_model_bundle(models, EXPANDED_MODEL_STEM, result.selected_model.estimator, card)
    model_metadata = {
        "artifact_sha256": sha256_file(paths["artifact"]),
        "card_sha256": sha256_file(paths["card"]),
        "artifact_file": paths["artifact"].name,
        "card_file": paths["card"].name,
    }
    frozen_state = {
        "before": dict(frozen_before or {}),
        "after": dict(frozen_after or {}),
        "all_unchanged": dict(frozen_before or {}) == dict(frozen_after or {}),
    }
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(reports.iterdir())
        if path.is_file() and path.name != "run_metadata.json"
    }
    artifact_hashes.update(
        {
            f"models/{path.name}": sha256_file(path)
            for path in sorted(models.iterdir())
            if path.is_file()
        }
    )
    metadata_payload = {
        "benchmark_config_hash": config.config_hash,
        "benchmark_config_sha256": config.config_sha256,
        "benchmark_config": config.identity_payload(),
        "dataset": {
            "m3": {
                "root": str(config.m3.dataset),
                "dataset_version": config.m3.dataset_version,
                "generation_config_hash": config.m3.generation_config_hash,
                "manifest_sha256": config.m3.manifest_sha256,
                "sources_sha256": config.m3.sources_sha256,
                "features_sha256": config.m3.features_sha256,
                "ground_truth_sha256": config.m3.ground_truth_sha256,
                "summary_sha256": config.m3.summary_sha256,
                "split_assignment_sha256": config.m3.split_assignment_sha256,
            },
            "m5a": {
                "root": str(config.m5a.dataset),
                "dataset_version": config.m5a.dataset_version,
                "config_sha256": config.m5a.config_sha256,
                "pilot_config_hash": config.m5a.pilot_config_hash,
                "manifest_sha256": config.m5a.manifest_sha256,
                "sources_sha256": config.m5a.sources_sha256,
                "features_sha256": config.m5a.features_sha256,
                "summary_sha256": config.m5a.summary_sha256,
                "image_pixel_hash_set": config.m5a.image_pixel_hash_set,
                "gate_verdict": config.m5a.gate_verdict,
            },
        },
        "schema": {
            "feature_set_version": config.schema.feature_set_version,
            "feature_names": list(config.schema.feature_names),
            "target": config.schema.target,
            "split_names": list(config.schema.split_names),
        },
        "protocol": {
            "primary_amplitudes_px": list(config.protocol.primary_amplitudes_px),
            "diagnostic_amplitudes_px": list(config.protocol.diagnostic_amplitudes_px),
            "primary_severities": list(config.protocol.primary_severities),
            "diagnostic_severities": list(config.protocol.diagnostic_severities),
            "excluded_from_primary": ("0.5 px diagnostic rung, clean rows and all control rows"),
        },
        "split": {
            "assignment_sha256": config.m3.split_assignment_sha256,
            "train_sources": int(result.frames.expanded_train["source_id"].nunique()),
            "val_sources": int(result.frames.expanded_val["source_id"].nunique()),
            "primary_test_sources": int(result.frames.primary_test["source_id"].nunique()),
            "diagnostic_test_sources": int(result.frames.diagnostic_test["source_id"].nunique()),
            "m3_test_sources": int(result.frames.m3_test["source_id"].nunique()),
        },
        "selection": {
            "selected_model": result.selected["name"],
            "selected_family": result.selected["family"],
            "criterion": f"val_{config.run.selection_metric}",
            "selected_on": "val",
            "n_candidates": int(len(result.candidates)),
        },
        "frozen_model": {
            "model_dir": str(config.model.model_dir),
            "model_stem": config.model.model_stem,
            "card_sha256": config.model.card_sha256,
            "artifact_sha256": config.model.artifact_sha256,
            "card": dict(result.frozen_card),
        },
        "expanded_model": model_metadata,
        "frozen_assets": frozen_state,
        "runtime": {
            "package_version": package_version(),
            "sklearn_version": sklearn_version(),
            "numpy_version": rb._dependency_version("numpy"),  # noqa: SLF001
            "pandas_version": rb._dependency_version("pandas"),  # noqa: SLF001
            "scikit_image_version": rb._dependency_version("scikit-image"),  # noqa: SLF001
            "joblib_version": rb._dependency_version("joblib"),  # noqa: SLF001
            "python_version": sys.version.split()[0],
            "code_revision": code_revision(),
            "elapsed_seconds_in_process": (
                round(float(elapsed_seconds), 6) if elapsed_seconds is not None else None
            ),
        },
        "artifacts": artifact_hashes,
    }
    metadata_path = reports / "run_metadata.json"
    rb._write_json(rb._json_ready(metadata_payload), metadata_path)  # noqa: SLF001
    return (
        reports,
        models,
        {
            "report": report_path,
            "metadata": metadata_path,
            "model_artifact": paths["artifact"],
            "model_card": paths["card"],
        },
    )


def _metric_table(
    table: pd.DataFrame,
    *,
    dataset: str,
    models: Sequence[str],
    scopes: Sequence[tuple[str, str]],
) -> list[list[object]]:
    rows: list[list[object]] = []
    for model in models:
        for scope, level in scopes:
            matches = table[
                (table["model"] == model)
                & (table["dataset"] == dataset)
                & (table["scope"] == scope)
                & (table["level"] == level)
            ]
            if len(matches) != 1:
                continue
            row = matches.iloc[0]
            rows.append(
                [
                    model,
                    f"{scope}:{level}",
                    int(row["n_samples"]),
                    int(row["n_sources"]),
                    float(row["mae"]),
                    float(row["rmse"]),
                    float(row["r2"]),
                    float(row["mean_residual"]),
                ]
            )
    return rows


def _decision_lines(result: GeometricBenchmarkResult) -> list[str]:
    primary = result.adaptation_metrics
    diagnostic = result.adaptation_metrics
    m3 = result.m3_retention_metrics
    frozen = rb._metric_lookup(primary, FROZEN_MODEL_NAME, PRIMARY_LABEL, "overall", "all")  # noqa: SLF001
    adapted = rb._metric_lookup(primary, EXPANDED_MODEL_NAME, PRIMARY_LABEL, "overall", "all")  # noqa: SLF001
    m3_naive = rb._metric_lookup(  # noqa: SLF001
        result.zero_shot_metrics, M3_NAIVE_NAME, PRIMARY_LABEL, "overall", "all"
    )
    expanded_naive = rb._metric_lookup(  # noqa: SLF001
        primary, EXPANDED_NAIVE_NAME, PRIMARY_LABEL, "overall", "all"
    )
    diagnostic_frozen = rb._metric_lookup(  # noqa: SLF001
        diagnostic, FROZEN_MODEL_NAME, DIAGNOSTIC_LABEL, "overall", "all"
    )
    diagnostic_adapted = rb._metric_lookup(  # noqa: SLF001
        diagnostic, EXPANDED_MODEL_NAME, DIAGNOSTIC_LABEL, "overall", "all"
    )
    retention_frozen = rb._metric_lookup(  # noqa: SLF001
        m3, FROZEN_MODEL_NAME, M3_TEST_LABEL, "overall", "all"
    )
    retention_adapted = rb._metric_lookup(  # noqa: SLF001
        m3, EXPANDED_MODEL_NAME, M3_TEST_LABEL, "overall", "all"
    )
    clean_frozen = rb._metric_lookup(  # noqa: SLF001
        m3, FROZEN_MODEL_NAME, M3_TEST_LABEL, "clean", "clean"
    )
    clean_adapted = rb._metric_lookup(  # noqa: SLF001
        m3, EXPANDED_MODEL_NAME, M3_TEST_LABEL, "clean", "clean"
    )
    retention_pair = result.uncertainty[
        (result.uncertainty["evaluation"] == "m3_retention")
        & (result.uncertainty["scope"] == "overall")
        & (result.uncertainty["level"] == "all")
        & (result.uncertainty["model"] == f"{EXPANDED_MODEL_NAME}_minus_{FROZEN_MODEL_NAME}")
        & (result.uncertainty["metric"] == "rmse")
    ].iloc[0]
    severity_rows = {
        level: (
            rb._metric_lookup(primary, FROZEN_MODEL_NAME, PRIMARY_LABEL, "severity", level),  # noqa: SLF001
            rb._metric_lookup(primary, EXPANDED_MODEL_NAME, PRIMARY_LABEL, "severity", level),  # noqa: SLF001
        )
        for level in ("2", "3", "4")
    }
    zero_shot_reduction = (
        (float(m3_naive["mae"]) - float(frozen["mae"])) / float(m3_naive["mae"])
        if float(m3_naive["mae"])
        else math.nan
    )
    adaptation_change = (
        (float(frozen["rmse"]) - float(adapted["rmse"])) / float(frozen["rmse"])
        if float(frozen["rmse"])
        else math.nan
    )
    retention_change = (
        (float(retention_adapted["rmse"]) - float(retention_frozen["rmse"]))
        / float(retention_frozen["rmse"])
        if float(retention_frozen["rmse"])
        else math.nan
    )
    primary_improves = float(adapted["rmse"]) < float(frozen["rmse"])
    clean_regresses = float(clean_adapted["mae"]) > float(clean_frozen["mae"])
    if primary_improves and not clean_regresses:
        decision = (
            "The adapted conventional model has lower held-out primary geometric RMSE "
            "than frozen M3 without a large clean-reference retention regression. "
            "Conventional ML remains the benchmark champion for this controlled task; "
            "the result is consistent with distribution shift that representative "
            "training data can address. No CNN is justified by this evidence."
        )
    elif primary_improves:
        decision = (
            "The adapted conventional model improves the primary geometric score but "
            "substantially degrades clean-reference retention. Keep frozen M3 for the "
            "original task and treat the adapted conventional model as a geometric-"
            "robustness candidate, not a drop-in replacement. Conventional ML remains "
            "the benchmark family, but the clean-calibration trade-off must be fixed "
            "under a new predeclared protocol before broader claims. This evidence "
            "does not justify a CNN."
        )
    else:
        decision = (
            "Controlled adaptation does not improve primary geometric RMSE over the "
            "frozen M3 model on this split. Before proposing a more complex model, "
            "inspect target behaviour, the eight-feature representation, test-source "
            "count and the interpolation/border confound. A CNN is not justified merely "
            "because this corruption family is new."
        )
    return [
        "### Interpretation and decision gate",
        "",
        "No universal pass/fail threshold was invented after observing test results. "
        "The statements below are descriptive, use held-out sources only and are not "
        "causal claims about human or production CAPTCHA performance.",
        "",
        f"- **Primary score (1.0-3.0 px, 36 rows/12 sources).** Frozen M3 has MAE "
        f"{rb._fmt(frozen['mae'])}, RMSE {rb._fmt(frozen['rmse'])} and R2 "
        f"{rb._fmt(frozen['r2'])}. The M3 training-mean baseline has MAE "
        f"{rb._fmt(m3_naive['mae'])}, RMSE {rb._fmt(m3_naive['rmse'])} and R2 "
        f"{rb._fmt(m3_naive['r2'])}; frozen M3 changes MAE by "
        f"{rb._fmt(100 * zero_shot_reduction, 1)}% relative to that constant. The frozen "
        f"mean signed residual is {rb._fmt(frozen['mean_residual'])} with residual SD "
        f"{rb._fmt(frozen['sd_residual'])}.",
        f"- **Adaptation.** Validation selected `{result.selected['name']}` "
        f"({result.selected['family']}). It has MAE {rb._fmt(adapted['mae'])}, RMSE "
        f"{rb._fmt(adapted['rmse'])} and R2 {rb._fmt(adapted['r2'])} on the same primary "
        f"rows; RMSE changes by {rb._fmt(100 * adaptation_change, 1)}% versus frozen M3. "
        f"The expanded training-mean baseline has MAE {rb._fmt(expanded_naive['mae'])} "
        f"and RMSE {rb._fmt(expanded_naive['rmse'])}. Adapted R2 ({rb._fmt(adapted['r2'])}) "
        "remains near zero: the aggregate error is lower, but the model still explains "
        "almost none of the between-source variation at fixed severity.",
        f"- **Severity trade-off.** Adapted MAE is "
        f"{rb._fmt(severity_rows['2'][1]['mae'])} at 1.0 px versus "
        f"{rb._fmt(severity_rows['2'][0]['mae'])} for frozen M3 (worse at the low end), "
        f"{rb._fmt(severity_rows['3'][1]['mae'])} versus "
        f"{rb._fmt(severity_rows['3'][0]['mae'])} at 2.0 px, and "
        f"{rb._fmt(severity_rows['4'][1]['mae'])} versus "
        f"{rb._fmt(severity_rows['4'][0]['mae'])} at 3.0 px (better at the high end). "
        "The aggregate primary improvement is therefore not uniform across severity.",
        f"- **0.5 px diagnostic (12 rows/12 sources, not in the primary score).** Frozen "
        f"MAE {rb._fmt(diagnostic_frozen['mae'])}, RMSE "
        f"{rb._fmt(diagnostic_frozen['rmse'])}; adapted MAE "
        f"{rb._fmt(diagnostic_adapted['mae'])}, RMSE "
        f"{rb._fmt(diagnostic_adapted['rmse'])}. This rung is close to the interpolation "
        "floor and is reported only as a diagnostic; it is not used to select the model "
        "or to compute the primary geometric score.",
        f"- **Original-task retention.** On the same 108-row/12-source M3 test split, "
        f"frozen M3 has MAE {rb._fmt(retention_frozen['mae'])} and RMSE "
        f"{rb._fmt(retention_frozen['rmse'])}; the adapted model has MAE "
        f"{rb._fmt(retention_adapted['mae'])} and RMSE "
        f"{rb._fmt(retention_adapted['rmse'])}. Aggregate retention RMSE changes by "
        f"{rb._fmt(100 * retention_change, 1)}%; the paired source-grouped RMSE "
        f"difference is {rb._fmt(retention_pair['estimate'])}, with a preliminary "
        f"interval [{rb._fmt(retention_pair['ci_low'])}, "
        f"{rb._fmt(retention_pair['ci_high'])}] that includes zero and is not a "
        f"significance test. The clean-reference slice is much worse: MAE "
        f"{rb._fmt(clean_adapted['mae'])} versus {rb._fmt(clean_frozen['mae'])} for "
        "frozen M3. Retention was not used to choose or retune the adapted model.",
        f"- **Decision.** {decision}",
        "- **Uncertainty.** The intervals resample whole source groups, but there are "
        "only 12 held-out sources. Overlapping or non-overlapping intervals are not "
        "treated as significance tests.",
    ]


def _report_markdown(config: GeometricBenchmarkConfig, result: GeometricBenchmarkResult) -> str:
    frames = result.frames
    zero_rows = _metric_table(
        result.zero_shot_metrics,
        dataset=PRIMARY_LABEL,
        models=(FROZEN_MODEL_NAME, M3_NAIVE_NAME),
        scopes=(
            ("overall", "all"),
            ("severity", "2"),
            ("severity", "3"),
            ("severity", "4"),
            ("font", "arial"),
            ("font", "courier_new"),
            ("font", "times_new_roman"),
        ),
    )
    diagnostic_rows = _metric_table(
        result.zero_shot_metrics,
        dataset=DIAGNOSTIC_LABEL,
        models=(FROZEN_MODEL_NAME, M3_NAIVE_NAME),
        scopes=(("overall", "all"), ("severity", "1")),
    )
    adaptation_rows = _metric_table(
        result.adaptation_metrics,
        dataset=PRIMARY_LABEL,
        models=(FROZEN_MODEL_NAME, EXPANDED_MODEL_NAME, EXPANDED_NAIVE_NAME),
        scopes=(
            ("overall", "all"),
            ("severity", "2"),
            ("severity", "3"),
            ("severity", "4"),
            ("font", "arial"),
            ("font", "courier_new"),
            ("font", "times_new_roman"),
        ),
    )
    diagnostic_adaptation_rows = _metric_table(
        result.adaptation_metrics,
        dataset=DIAGNOSTIC_LABEL,
        models=(FROZEN_MODEL_NAME, EXPANDED_MODEL_NAME, EXPANDED_NAIVE_NAME),
        scopes=(("overall", "all"), ("severity", "1")),
    )
    m3_reference_rows = _metric_table(
        result.m3_reference_metrics,
        dataset=M3_TEST_LABEL,
        models=(FROZEN_MODEL_NAME, M3_NAIVE_NAME),
        scopes=(
            ("overall", "all"),
            ("clean", "clean"),
            ("corrupted", "all"),
            ("family", "gaussian_blur"),
            ("family", "gaussian_noise"),
        ),
    )
    retention_rows = _metric_table(
        result.m3_retention_metrics,
        dataset=M3_TEST_LABEL,
        models=(FROZEN_MODEL_NAME, EXPANDED_MODEL_NAME, M3_NAIVE_NAME),
        scopes=(
            ("overall", "all"),
            ("clean", "clean"),
            ("corrupted", "all"),
            ("family", "gaussian_blur"),
            ("family", "gaussian_noise"),
        ),
    )
    selected_uncertainty = result.uncertainty[
        result.uncertainty["model"].isin(
            [
                FROZEN_MODEL_NAME,
                EXPANDED_MODEL_NAME,
                M3_NAIVE_NAME,
                EXPANDED_NAIVE_NAME,
                f"{EXPANDED_MODEL_NAME}_minus_{FROZEN_MODEL_NAME}",
                f"{FROZEN_MODEL_NAME}_minus_{M3_NAIVE_NAME}",
                f"{EXPANDED_MODEL_NAME}_minus_{EXPANDED_NAIVE_NAME}",
            ]
        )
    ]
    uncertainty_rows = selected_uncertainty[selected_uncertainty["metric"].isin(["mae", "rmse"])]
    residual_rows = result.adaptation_residuals[
        (result.adaptation_residuals["model"] == EXPANDED_MODEL_NAME)
        & (result.adaptation_residuals["dataset"] == PRIMARY_LABEL)
    ]
    target_rows = result.target_distribution[
        result.target_distribution["dataset"].isin(["expanded", "evaluation", "m5a"])
    ]

    lines = [
        "# M5B grouped geometric-robustness benchmark",
        "",
        "**Scope:** synthetic image similarity, measured as SSIM against the exact "
        "clean reference. This is not a measure of human readability, CAPTCHA "
        "solvability, security, or real-world quality.",
        "",
        "## Frozen protocol",
        "",
        f"- Benchmark: `{config.run.name}` `{config.run.version}`",
        f"- Config: `{config.config_path}` (sha256 `{config.config_sha256}`)",
        f"- Protocol hash: `{config.config_hash}`",
        f"- Seed: `{config.run.seed}`; grouped bootstrap replicates: `{config.run.n_bootstrap}`",
        f"- Selection metric: `val_{config.run.selection_metric}`",
        f"- M3 dataset: `{config.m3.dataset}`; generation "
        f"`{config.m3.generation_config_hash}`; split assignment "
        f"`{config.m3.split_assignment_sha256}`",
        f"- M3 manifest `{config.m3.manifest_sha256}`; sources "
        f"`{config.m3.sources_sha256}`; features `{config.m3.features_sha256}`",
        f"- M5A dataset: `{config.m5a.dataset}`; pilot generation hash "
        f"`{config.m5a.pilot_config_hash}`; manifest `{config.m5a.manifest_sha256}`",
        f"- M5A sources `{config.m5a.sources_sha256}`; features "
        f"`{config.m5a.features_sha256}`; image-pixel-hash set "
        f"`{config.m5a.image_pixel_hash_set}`",
        f"- M5A gate: `{config.m5a.gate_verdict}`; the 0.5 px rung is diagnostic only",
        f"- Frozen model: `{config.model.model_dir}/{config.model.model_stem}`; card "
        f"`{config.model.card_sha256}`; artifact `{config.model.artifact_sha256}`",
        f"- Feature schema: `{config.schema.feature_set_version}`, target "
        f"`{config.schema.target}`, {len(config.schema.feature_names)} image-only features",
        "- Explicit source-to-split assignments: `split_assignment.csv`",
        "",
        "The split is grouped by clean source. Every M5A variant and severity of a "
        "source stays in the frozen M3 split. No test-source image, feature, target "
        "or metadata is used for fitting, preprocessing, thresholding or selection.",
        "",
        "### Severity inclusion rule",
        "",
        f"- Primary geometric score: amplitudes "
        f"`{list(config.protocol.primary_amplitudes_px)}` px "
        f"(severities `{list(config.protocol.primary_severities)}`).",
        f"- Diagnostic only: amplitude "
        f"`{list(config.protocol.diagnostic_amplitudes_px)}` px "
        f"(severity `{list(config.protocol.diagnostic_severities)}`).",
        "- The 0.5 px rung is excluded from expanded training, validation selection and "
        "the primary score. Clean and control rows are excluded from all fit frames.",
        "",
        "## Data accounting and split isolation",
        "",
    ]
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            [
                "dataset",
                "split",
                "sources",
                "samples",
                "clean",
                "controls",
                "primary",
                "diagnostic",
                "blur",
                "noise",
            ],
            [
                [
                    row.dataset,
                    row.split,
                    row.sources,
                    row.samples,
                    row.clean,
                    row.controls,
                    row.primary_elastic,
                    row.diagnostic_elastic,
                    row.gaussian_blur,
                    row.gaussian_noise,
                ]
                for row in frames.data_accounting.itertuples(index=False)
            ],
        )
    )
    lines.extend(["", "Isolation checks:", ""])
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            ["check", "result", "detail"],
            [
                [check.name, "PASS" if check.passed else "FAIL", check.detail]
                for check in result.checks
            ],
        )
    )
    lines.extend(
        [
            "",
            "M5A clean rows match the M3 clean pixels, targets and features exactly. "
            "The identity and zero-field controls are exact duplicates; the minimal "
            "control is near-clean. All 240 clean/control rows are excluded from the "
            "expanded fit frames and from scoring. Primary training adds only the 1.0, "
            "2.0 and 3.0 px elastic rows from M3-assigned training/validation sources.",
            "",
            "## Experiment A - frozen M3 zero-shot on M5A test sources",
            "",
            "The frozen M3 model was loaded through the existing card/artifact verifier. "
            "No parameter, scaler or calibration was fitted on these rows.",
            "",
            "### Primary geometric score (1.0-3.0 px)",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            ["model", "scope", "n", "sources", "MAE", "RMSE", "R2", "mean residual"],
            zero_rows,
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Diagnostic 0.5 px rung (not in the primary score)",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            ["model", "scope", "n", "sources", "MAE", "RMSE", "R2", "mean residual"],
            diagnostic_rows,
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "`R2` is reported as `n/a` when the target is constant; that is expected "
            "for exact-clean slices and is not treated as a perfect fit.",
            "",
            "### Original M3 blur/noise test reference",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            ["model", "scope", "n", "sources", "MAE", "RMSE", "R2", "mean residual"],
            m3_reference_rows,
            digits=4,
        )
    )
    frozen_primary = rb._metric_lookup(  # noqa: SLF001
        result.zero_shot_metrics, FROZEN_MODEL_NAME, PRIMARY_LABEL, "overall", "all"
    )
    frozen_m3 = rb._metric_lookup(  # noqa: SLF001
        result.m3_reference_metrics, FROZEN_MODEL_NAME, M3_TEST_LABEL, "overall", "all"
    )
    lines.extend(
        [
            "",
            f"Frozen M3 primary geometric MAE is {rb._fmt(frozen_primary['mae'])} "
            f"and RMSE {rb._fmt(frozen_primary['rmse'])}, versus M3 blur/noise test "
            f"MAE {rb._fmt(frozen_m3['mae'])} and RMSE {rb._fmt(frozen_m3['rmse'])}. "
            "This is a descriptive comparison across different task mixtures, not a "
            "threshold or a claim that one corruption is intrinsically harder.",
            "",
            "## Experiment B - controlled conventional adaptation",
            "",
            "Candidates were fitted on M3 train rows plus M5A primary geometric rows "
            "from training sources, and selected on M5A validation sources only. The "
            "selected model was then evaluated once on M5A test sources.",
            "",
            "### Validation-only candidate selection",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            ["model", "family", "n inputs", "train RMSE", "val RMSE", "val R2", "selected"],
            [
                [
                    row.model,
                    row.family,
                    row.n_inputs,
                    row.train_rmse,
                    row.val_rmse,
                    row.val_r2,
                    bool(row.selected),
                ]
                for row in result.candidates.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            f"Selected: `{result.selected['name']}` (family "
            f"`{result.selected['family']}`), chosen on validation "
            f"`{config.run.selection_metric}`.",
            "",
            "### Held-out primary geometric comparison",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            ["model", "scope", "n", "sources", "MAE", "RMSE", "R2", "mean residual"],
            adaptation_rows,
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Diagnostic 0.5 px comparison (not in the primary score)",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            ["model", "scope", "n", "sources", "MAE", "RMSE", "R2", "mean residual"],
            diagnostic_adaptation_rows,
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Residual analysis for the selected adapted model (primary)",
            "",
            "Signed residual is `target - prediction`; positive values mean under-prediction.",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            ["family", "severity", "n", "sources", "mean residual", "median", "MAE", "RMSE", "R2"],
            [
                [
                    row.family,
                    row.severity_level,
                    row.n_samples,
                    row.n_sources,
                    row.mean_residual,
                    row.median_residual,
                    row.mae,
                    row.rmse,
                    row.r2,
                ]
                for row in residual_rows.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "## Experiment C - original-task retention",
            "",
            "The selected adapted model and frozen M3 model were evaluated on exactly "
            "the same untouched M3 test rows. Retention results were not used to choose "
            "or retune the adapted model.",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            ["model", "scope", "n", "sources", "MAE", "RMSE", "R2", "mean residual"],
            retention_rows,
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Paired retention comparison (adapted - frozen)",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            [
                "scope",
                "n",
                "sources",
                "frozen MAE",
                "adapted MAE",
                "MAE delta",
                "frozen RMSE",
                "adapted RMSE",
                "RMSE delta",
            ],
            [
                [
                    f"{row.scope}:{row.level}",
                    row.n_samples,
                    row.n_sources,
                    row.frozen_mae,
                    row.adapted_mae,
                    row.mae_delta,
                    row.frozen_rmse,
                    row.adapted_rmse,
                    row.rmse_delta,
                ]
                for row in result.retention_comparison.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "## Target distribution",
            "",
            "The table discloses target shift; no test statistic is used to tune a model.",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            [
                "dataset",
                "split",
                "family",
                "severity",
                "amplitude",
                "n",
                "mean",
                "sd",
                "min",
                "max",
            ],
            [
                [
                    row.dataset,
                    row.split,
                    row.family,
                    row.severity_level,
                    row.amplitude_px,
                    row.n_samples,
                    row.mean,
                    row.sd,
                    row.min,
                    row.max,
                ]
                for row in target_rows.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "## Grouped uncertainty",
            "",
            "Intervals resample the 12 held-out source groups, preserving all rows "
            "within each selected group. They are preliminary and descriptive at this "
            "group count. `comparison` rows are paired source-grouped differences "
            "(`a - b`) for MAE and RMSE. Interval overlap is not a significance test.",
            "",
        ]
    )
    lines.extend(
        rb._markdown_table(  # noqa: SLF001
            [
                "model/comparison",
                "evaluation",
                "dataset",
                "scope",
                "level",
                "metric",
                "estimate",
                "95% low",
                "95% high",
            ],
            [
                [
                    row.model,
                    row.evaluation,
                    row.dataset,
                    row.scope,
                    row.level,
                    row.metric,
                    row.estimate,
                    row.ci_low,
                    row.ci_high,
                ]
                for row in uncertainty_rows.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(["", *_decision_lines(result)])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- One synthetic renderer, one 192x64 canvas, three fonts and 60 sources.",
            "- Only 12 held-out test sources; grouped intervals are preliminary.",
            "- The 0.5 px rung is interpolation-dominated and is not part of the primary score.",
            "- Bilinear resampling and edge padding are part of the corruption; the M5A "
            "minimal control and border diagnostics quantify, but do not remove, that confound.",
            "- The adapted model is trained on a mixture whose target distribution differs "
            "from the geometric test distribution; the target-distribution table discloses it.",
            "- SSIM is a full-reference structural proxy, not human readability or CAPTCHA "
            "solvability. This is not a security evaluation.",
            "- No CNN, dashboard, deployment, external CAPTCHA dataset or frozen-model "
            "retuning is part of this benchmark.",
            "",
            "## Reproducibility",
            "",
            "Run and verify with:",
            "",
            "```text",
            "uv run --extra image --extra ml captcha-geometric-benchmark run \\",
            "    --config configs/geometric_robustness_benchmark_v0_1_0.toml",
            "uv run --extra image --extra ml captcha-geometric-benchmark verify \\",
            "    --config configs/geometric_robustness_benchmark_v0_1_0.toml",
            "```",
            "",
            "Generated tables, the model bundle and this report live in Git-ignored paths.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_geometric_benchmark(
    config: GeometricBenchmarkConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    verify_images: bool = True,
) -> tuple[GeometricBenchmarkResult, Path, Path, dict[str, Path]]:
    """Load, verify, run and persist the complete M5B benchmark."""
    started = time.perf_counter()
    before = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    frames = load_benchmark_frames(config, verify_images=verify_images)
    frozen_model, frozen_card, model_checks = load_frozen_model(config, frames.m3_summary)
    rb._verify_checks(model_checks, "frozen M3 model verification")  # noqa: SLF001
    if frozen_model is None:
        raise GeometricBenchmarkError("frozen M3 model could not be loaded")
    result = run_benchmark_core(
        frames,
        config,
        frozen_estimator=frozen_model,
        frozen_card=frozen_card,
    )
    after = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    if before != after:
        changed = sorted(
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        )
        raise GeometricBenchmarkError(f"frozen assets changed during the run: {changed[:5]}")
    reports, models, paths = write_benchmark_outputs(
        result,
        config,
        output_reports=output_reports,
        output_models=output_models,
        frozen_before=before,
        frozen_after=after,
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, reports, models, paths


# ---------------------------------------------------------------------------
# Verification CLI
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GeometricBenchmarkError(f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GeometricBenchmarkError(f"JSON artifact is not an object: {path}")
    return payload


def _check_hash(path: Path, expected: object, label: str) -> None:
    if not path.is_file():
        raise GeometricBenchmarkError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise GeometricBenchmarkError(f"{label} hash mismatch: expected {expected}, found {actual}")


def verify_benchmark(
    config: GeometricBenchmarkConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    verify_images: bool = True,
) -> VerificationReport:
    """Verify a written run against its configuration and provenance."""
    reports = Path(output_reports) if output_reports is not None else config.run.output_reports
    models = Path(output_models) if output_models is not None else config.run.output_models
    checks: list[CheckResult] = []

    def run(name: str, function: Any) -> None:
        try:
            detail = function()
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            checks.append(
                CheckResult(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")
            )
        else:
            checks.append(CheckResult(name=name, passed=True, detail=str(detail)))

    frames: BenchmarkFrames | None = None
    frozen_model: Any = None

    def load_inputs() -> str:
        nonlocal frames, frozen_model
        frames = load_benchmark_frames(config, verify_images=verify_images)
        frozen_model, _, model_checks = load_frozen_model(config, frames.m3_summary)
        rb._verify_checks(model_checks, "frozen M3 model verification")  # noqa: SLF001
        return f"{len(frames.m3_frame)} M3 rows, {len(frames.m5a_frame)} M5A rows, model loaded"

    run("inputs_and_model", load_inputs)

    def metadata_checks() -> str:
        metadata = _load_json(reports / "run_metadata.json")
        if metadata.get("benchmark_config_hash") != config.config_hash:
            raise GeometricBenchmarkError("run_metadata config hash does not match the config")
        if metadata.get("benchmark_config_sha256") != config.config_sha256:
            raise GeometricBenchmarkError("run_metadata config file hash does not match")
        _check_hash(
            models / f"{EXPANDED_MODEL_STEM}.joblib",
            metadata.get("expanded_model", {}).get("artifact_sha256"),
            "expanded model artifact",
        )
        _check_hash(
            models / f"{EXPANDED_MODEL_STEM}_card.json",
            metadata.get("expanded_model", {}).get("card_sha256"),
            "expanded model card",
        )
        artifacts = metadata.get("artifacts", {})
        if not isinstance(artifacts, dict) or not artifacts:
            raise GeometricBenchmarkError("run_metadata has no artifact hash table")
        for relative, expected in sorted(artifacts.items()):
            if relative.startswith("models/"):
                path = models / relative.removeprefix("models/")
            else:
                path = reports / relative
            _check_hash(path, expected, relative)
        if metadata.get("frozen_assets", {}).get("all_unchanged") is not True:
            raise GeometricBenchmarkError("run_metadata reports changed frozen assets")
        return f"{len(artifacts)} artifact hashes match and frozen inputs were unchanged"

    run("run_metadata", metadata_checks)

    def severity_protocol_check() -> str:
        if frames is None:
            raise GeometricBenchmarkError("frames are unavailable")
        if (frames.expanded_train["dataset"] == M5A_LABEL).any():
            fit = pd.concat([frames.expanded_train, frames.expanded_val], ignore_index=True)
            m5a_fit = fit[fit["dataset"] == M5A_LABEL]
            if _amplitude_mask(m5a_fit, DIAGNOSTIC_AMPLITUDES_PX).any():
                raise GeometricBenchmarkError("0.5 px rows are present in the fit frames")
            if set(m5a_fit["protocol_scope"]) != {"primary"}:
                raise GeometricBenchmarkError("non-primary M5A rows are present in fit frames")
        if _amplitude_mask(frames.primary_test, DIAGNOSTIC_AMPLITUDES_PX).any():
            raise GeometricBenchmarkError("0.5 px rows are present in the primary test frame")
        if not _amplitude_mask(frames.diagnostic_test, DIAGNOSTIC_AMPLITUDES_PX).all():
            raise GeometricBenchmarkError("diagnostic frame is not the 0.5 px rung")
        return (
            "0.5 px rung is absent from train/val and the primary test score; "
            "it is present only in the diagnostic frame"
        )

    run("severity_protocol", severity_protocol_check)

    def expanded_model_checks() -> str:
        if frames is None:
            raise GeometricBenchmarkError("frames are unavailable")
        model, _, model_checks = load_model_bundle(
            models,
            EXPANDED_MODEL_STEM,
            dataset_summary=frames.m3_summary,
        )
        rb._verify_checks(model_checks, "expanded model verification")  # noqa: SLF001
        if model is None:
            raise GeometricBenchmarkError("expanded model could not be loaded")
        expected = pd.read_csv(reports / "adaptation_predictions.csv")
        for label, frame, dataset_label in (
            ("primary", frames.primary_test, PRIMARY_LABEL),
            ("diagnostic", frames.diagnostic_test, DIAGNOSTIC_LABEL),
        ):
            selected = expected[
                (expected["model"] == EXPANDED_MODEL_NAME)
                & (expected["dataset_label"] == dataset_label)
            ]
            if len(selected) != len(frame):
                raise GeometricBenchmarkError(
                    f"adaptation_predictions.csv has the wrong {label} row count"
                )
            actual = predict(model, frame)
            if not np.allclose(
                selected["prediction"].to_numpy(dtype=np.float64),
                actual,
                rtol=0.0,
                atol=1e-9,
            ):
                raise GeometricBenchmarkError(f"expanded model {label} predictions do not replay")
        return "model reloads and reproduces primary and diagnostic predictions"

    run("expanded_model_replay", expanded_model_checks)

    def zero_shot_replay() -> str:
        if frames is None or frozen_model is None:
            raise GeometricBenchmarkError("frozen model or frames are unavailable")
        expected = pd.read_csv(reports / "zero_shot_predictions.csv")
        for label, frame, dataset_label in (
            ("primary", frames.primary_test, PRIMARY_LABEL),
            ("diagnostic", frames.diagnostic_test, DIAGNOSTIC_LABEL),
        ):
            selected = expected[
                (expected["model"] == FROZEN_MODEL_NAME)
                & (expected["dataset_label"] == dataset_label)
            ]
            if len(selected) != len(frame):
                raise GeometricBenchmarkError(
                    f"zero_shot_predictions.csv has the wrong {label} row count"
                )
            actual = predict(frozen_model, frame)
            if not np.allclose(
                selected["prediction"].to_numpy(dtype=np.float64),
                actual,
                rtol=0.0,
                atol=1e-9,
            ):
                raise GeometricBenchmarkError(f"frozen model {label} predictions do not replay")
        return "frozen model reproduces primary and diagnostic zero-shot predictions"

    run("zero_shot_replay", zero_shot_replay)

    def report_exists() -> str:
        report = reports / "benchmark.md"
        if not report.is_file() or not report.read_text(encoding="utf-8").strip():
            raise GeometricBenchmarkError("benchmark.md is missing or empty")
        text = report.read_text(encoding="utf-8")
        for marker in ("Experiment A", "Experiment B", "Experiment C", "0.5 px"):
            if marker not in text:
                raise GeometricBenchmarkError(f"benchmark.md is missing {marker!r}")
        return f"{report} is present and contains the required experiment sections"

    run("report", report_exists)
    return VerificationReport(dataset=str(reports), checks=tuple(checks))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-geometric-benchmark",
        description="Run or verify the grouped M5B geometric-robustness benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "run the grouped geometric benchmark"),
        ("verify", "verify a written benchmark run"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output-reports", type=Path, default=None)
        command.add_argument("--output-models", type=Path, default=None)
    return parser


def _print_report(report: VerificationReport) -> None:
    for check in report.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print(
        "geometric benchmark verification: PASS"
        if report.ok
        else "geometric benchmark verification: FAIL"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the M5B CLI; returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_geometric_benchmark_config(args.config)
        if args.command == "run":
            result, _, _, paths = run_geometric_benchmark(
                config,
                output_reports=args.output_reports,
                output_models=args.output_models,
            )
            print(f"benchmark: {config.describe()}")
            print(f"protocol hash: {config.config_hash}")
            print(
                f"rows: train {len(result.frames.expanded_train)}, "
                f"val {len(result.frames.expanded_val)}, "
                f"primary test {len(result.frames.primary_test)}, "
                f"diagnostic 0.5 px {len(result.frames.diagnostic_test)}"
            )
            frozen = rb._metric_lookup(  # noqa: SLF001
                result.zero_shot_metrics, FROZEN_MODEL_NAME, PRIMARY_LABEL, "overall", "all"
            )
            selected = rb._metric_lookup(  # noqa: SLF001
                result.adaptation_metrics, EXPANDED_MODEL_NAME, PRIMARY_LABEL, "overall", "all"
            )
            diagnostic = rb._metric_lookup(  # noqa: SLF001
                result.adaptation_metrics,
                EXPANDED_MODEL_NAME,
                DIAGNOSTIC_LABEL,
                "overall",
                "all",
            )
            retention = rb._metric_lookup(  # noqa: SLF001
                result.m3_retention_metrics, EXPANDED_MODEL_NAME, M3_TEST_LABEL, "overall", "all"
            )
            print(
                f"zero-shot frozen M3 primary: MAE {frozen['mae']:.4f}, "
                f"RMSE {frozen['rmse']:.4f}, R2 {frozen['r2']:.4f}"
            )
            print(
                f"adapted {result.selected['name']} primary: MAE {selected['mae']:.4f}, "
                f"RMSE {selected['rmse']:.4f}, R2 {selected['r2']:.4f}"
            )
            print(
                f"adapted diagnostic 0.5 px: MAE {diagnostic['mae']:.4f}, "
                f"RMSE {diagnostic['rmse']:.4f}, R2 {diagnostic['r2']:.4f}"
            )
            print(
                f"adapted M3 retention: MAE {retention['mae']:.4f}, "
                f"RMSE {retention['rmse']:.4f}, R2 {retention['r2']:.4f}"
            )
            print(f"report: {paths['report']}")
            print(f"model: {paths['model_artifact']}")
            return 0 if result.ok else 1
        if args.command == "verify":
            report = verify_benchmark(
                config,
                output_reports=args.output_reports,
                output_models=args.output_models,
            )
            _print_report(report)
            return 0 if report.ok else 1
        raise GeometricBenchmarkError(f"unknown command {args.command!r}")  # pragma: no cover
    except (
        ConfigError,
        DatasetError,
        GeometricBenchmarkConfigError,
        GeometricBenchmarkError,
        GeometricPilotError,
        ModelError,
        rb.RobustnessBenchmarkError,
        rb.RobustnessConfigError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
