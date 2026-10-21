"""Milestone M7A dataset-adequacy and evaluation audit.

M7A is a read-only audit. It does not generate a dataset, train or recalibrate a
model, change a target, change a split or add a corruption family. It consumes
the frozen M3 dataset, the M5A geometric-distortion pilot, the M5B grouped
benchmark, the M5C champion and the M6A-M6D study artifacts, verifies their
hashes, and reports:

* dataset composition and source independence,
* target distributions and between/within-source variance,
* champion residuals and descriptive leave-one-source-out influence,
* training-only feature redundancy and importance diagnostics,
* source-grouped bootstrap intervals for the frozen champion,
* a claim-to-evidence table, and
* one predeclared next direction without executing it.

The independent unit is a source group. Hundreds of correlated image variants
are not hundreds of independent test observations. Every conclusion is bounded
by the synthetic distribution: SSIM against a known clean reference is not
human readability, CAPTCHA solvability, security or production robustness.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

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
    bootstrap_metrics,
    predict,
    regression_metrics,
    score_frame,
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

AUDIT_VERSION: Final[str] = "v0.1.0"
CLAIM_SCHEMA_VERSION: Final[str] = "v1"
CLASSIFICATIONS: Final[tuple[str, ...]] = ("supported", "suggestive", "unsupported")
DIRECTION_IDS: Final[tuple[str, ...]] = ("A", "B", "C", "D")
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")
SCOPE_ORDER: Final[tuple[str, ...]] = (
    "m3_test_overall",
    "clean_reference",
    "original_blur_noise",
    "original_blur_noise_gaussian_blur",
    "original_blur_noise_gaussian_noise",
    "geometric_primary",
    "geometric_primary_severity_2",
    "geometric_primary_severity_3",
    "geometric_primary_severity_4",
    "diagnostic_0_5",
)
REPORT_SECTIONS: Final[tuple[str, ...]] = (
    "## Executive summary",
    "## 1. Audit protocol and pinned inputs",
    "## 2. Dataset inventory and independence",
    "## 3. Target distributions and variance",
    "## 4. Champion residuals by source and slice",
    "## 5. Source-grouped uncertainty",
    "## 6. Feature information and redundancy",
    "## 7. Claim-to-evidence table",
    "## 8. Recommended next experiment",
    "## 9. Limitations",
    "## 10. Reproduce and verify",
    "## 11. Research-integrity boundary",
)
OUTPUT_TABLE_NAMES: Final[tuple[str, ...]] = (
    "dataset_inventory.csv",
    "dataset_composition.csv",
    "split_accounting.csv",
    "target_distributions.csv",
    "test_slice_metrics.csv",
    "source_metrics.csv",
    "source_severity_metrics.csv",
    "source_family_metrics.csv",
    "font_metrics.csv",
    "leave_one_source_out.csv",
    "error_concentration.csv",
    "uncertainty.csv",
    "feature_summary.csv",
    "feature_correlations.csv",
    "feature_importance.csv",
    "feature_slice_means.csv",
    "feature_target_associations.csv",
    "claim_evidence.csv",
    "next_direction_comparison.csv",
    "isolation_checks.csv",
)

#: Frozen assets that must be byte-identical before and after an audit. M7A's
#: own outputs are deliberately absent.
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
)

_SHA256_PATTERN: Final[str] = "^[0-9a-f]{64}$"


class DatasetAuditError(RuntimeError):
    """Raised when the audit cannot run as declared."""


class DatasetAuditConfigError(ValueError):
    """Raised when the audit configuration is missing or malformed."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    name: str
    version: str
    seed: int
    n_bootstrap: int
    output_reports: Path


@dataclass(frozen=True, slots=True)
class SchemaSettings:
    feature_set_version: str
    feature_names: tuple[str, ...]
    target: str
    split_names: tuple[str, ...]
    claim_schema_version: str
    allowed_classifications: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class M3Pin:
    dataset: Path
    config: Path
    config_sha256: str
    manifest_sha256: str
    sources_sha256: str
    features_sha256: str
    ground_truth_sha256: str
    summary_sha256: str


@dataclass(frozen=True, slots=True)
class M5APin:
    dataset: Path
    config: Path
    config_sha256: str
    manifest_sha256: str
    sources_sha256: str
    features_sha256: str
    summary_sha256: str


@dataclass(frozen=True, slots=True)
class M5CPin:
    config: Path
    config_sha256: str
    protocol_hash: str
    champion_dir: Path
    champion_stem: str
    champion_artifact_sha256: str
    champion_card_sha256: str
    predictions: Path
    predictions_sha256: str
    test_metrics: Path
    test_metrics_sha256: str
    selection_table: Path
    selection_table_sha256: str
    run_metadata: Path
    run_metadata_sha256: str


@dataclass(frozen=True, slots=True)
class M5BPin:
    config: Path
    config_sha256: str
    data_accounting: Path
    data_accounting_sha256: str
    split_assignment: Path
    split_assignment_sha256: str


@dataclass(frozen=True, slots=True)
class M3BenchmarkPin:
    config: Path
    config_sha256: str
    heldout_font_metrics: Path
    heldout_font_metrics_sha256: str
    model_comparison: Path
    model_comparison_sha256: str


@dataclass(frozen=True, slots=True)
class AuditSettings:
    group_column: str
    bootstrap_method: str
    confidence_level: float
    near_degenerate_sd: float
    redundancy_abs_correlation: float
    top_k_sources: int
    min_sources_for_interval: int
    feature_slice_dimensions: tuple[str, ...]
    bootstrap_scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Expectations:
    m3_sources: tuple[int, int, int]
    m3_samples: tuple[int, int, int]
    m5a_sources: tuple[int, int, int]
    m5a_samples: tuple[int, int, int]
    expanded_train_rows: int
    expanded_val_rows: int
    primary_test_rows: int
    diagnostic_test_rows: int
    m3_test_rows: int
    n_sources: int
    n_fonts: int


@dataclass(frozen=True, slots=True)
class RecommendationSettings:
    selected_direction: str
    directions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditConfig:
    run: RunSettings
    schema: SchemaSettings
    m3: M3Pin
    m5a: M5APin
    m5c: M5CPin
    m5b: M5BPin
    m3_benchmark: M3BenchmarkPin
    audit: AuditSettings
    expectations: Expectations
    recommendation: RecommendationSettings
    config_path: Path
    config_sha256: str

    def identity_payload(self) -> dict[str, Any]:
        """Return the declared audit protocol, excluding the file hash."""
        return {
            "run": {
                "name": self.run.name,
                "version": self.run.version,
                "seed": self.run.seed,
                "n_bootstrap": self.run.n_bootstrap,
                "output_reports": str(self.run.output_reports),
            },
            "schema": {
                "feature_set_version": self.schema.feature_set_version,
                "feature_names": list(self.schema.feature_names),
                "target": self.schema.target,
                "split_names": list(self.schema.split_names),
                "claim_schema_version": self.schema.claim_schema_version,
                "allowed_classifications": list(self.schema.allowed_classifications),
            },
            "inputs": {
                "m3": {
                    "dataset": str(self.m3.dataset),
                    "config": str(self.m3.config),
                    "config_sha256": self.m3.config_sha256,
                    "manifest_sha256": self.m3.manifest_sha256,
                    "sources_sha256": self.m3.sources_sha256,
                    "features_sha256": self.m3.features_sha256,
                    "ground_truth_sha256": self.m3.ground_truth_sha256,
                    "summary_sha256": self.m3.summary_sha256,
                },
                "m5a": {
                    "dataset": str(self.m5a.dataset),
                    "config": str(self.m5a.config),
                    "config_sha256": self.m5a.config_sha256,
                    "manifest_sha256": self.m5a.manifest_sha256,
                    "sources_sha256": self.m5a.sources_sha256,
                    "features_sha256": self.m5a.features_sha256,
                    "summary_sha256": self.m5a.summary_sha256,
                },
                "m5c": {
                    "config": str(self.m5c.config),
                    "config_sha256": self.m5c.config_sha256,
                    "protocol_hash": self.m5c.protocol_hash,
                    "champion_dir": str(self.m5c.champion_dir),
                    "champion_stem": self.m5c.champion_stem,
                    "champion_artifact_sha256": self.m5c.champion_artifact_sha256,
                    "champion_card_sha256": self.m5c.champion_card_sha256,
                    "predictions": str(self.m5c.predictions),
                    "predictions_sha256": self.m5c.predictions_sha256,
                    "test_metrics": str(self.m5c.test_metrics),
                    "test_metrics_sha256": self.m5c.test_metrics_sha256,
                    "selection_table": str(self.m5c.selection_table),
                    "selection_table_sha256": self.m5c.selection_table_sha256,
                    "run_metadata": str(self.m5c.run_metadata),
                    "run_metadata_sha256": self.m5c.run_metadata_sha256,
                },
                "m5b": {
                    "config": str(self.m5b.config),
                    "config_sha256": self.m5b.config_sha256,
                    "data_accounting": str(self.m5b.data_accounting),
                    "data_accounting_sha256": self.m5b.data_accounting_sha256,
                    "split_assignment": str(self.m5b.split_assignment),
                    "split_assignment_sha256": self.m5b.split_assignment_sha256,
                },
                "m3_benchmark": {
                    "config": str(self.m3_benchmark.config),
                    "config_sha256": self.m3_benchmark.config_sha256,
                    "heldout_font_metrics": str(self.m3_benchmark.heldout_font_metrics),
                    "heldout_font_metrics_sha256": self.m3_benchmark.heldout_font_metrics_sha256,
                    "model_comparison": str(self.m3_benchmark.model_comparison),
                    "model_comparison_sha256": self.m3_benchmark.model_comparison_sha256,
                },
            },
            "audit": {
                "group_column": self.audit.group_column,
                "bootstrap_method": self.audit.bootstrap_method,
                "confidence_level": self.audit.confidence_level,
                "near_degenerate_sd": self.audit.near_degenerate_sd,
                "redundancy_abs_correlation": self.audit.redundancy_abs_correlation,
                "top_k_sources": self.audit.top_k_sources,
                "min_sources_for_interval": self.audit.min_sources_for_interval,
                "feature_slice_dimensions": list(self.audit.feature_slice_dimensions),
                "bootstrap_scopes": list(self.audit.bootstrap_scopes),
            },
            "expectations": {
                "m3_sources": list(self.expectations.m3_sources),
                "m3_samples": list(self.expectations.m3_samples),
                "m5a_sources": list(self.expectations.m5a_sources),
                "m5a_samples": list(self.expectations.m5a_samples),
                "expanded_train_rows": self.expectations.expanded_train_rows,
                "expanded_val_rows": self.expectations.expanded_val_rows,
                "primary_test_rows": self.expectations.primary_test_rows,
                "diagnostic_test_rows": self.expectations.diagnostic_test_rows,
                "m3_test_rows": self.expectations.m3_test_rows,
                "n_sources": self.expectations.n_sources,
                "n_fonts": self.expectations.n_fonts,
            },
            "recommendation": {
                "selected_direction": self.recommendation.selected_direction,
                "directions": list(self.recommendation.directions),
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
    missing = sorted(allowed - set(section))
    if unknown:
        raise DatasetAuditConfigError(f"unknown key(s) in [{name}]: {unknown}")
    if missing:
        raise DatasetAuditConfigError(f"missing key(s) in [{name}]: {missing}")


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise DatasetAuditConfigError(f"missing or invalid [{name}] section")
    return dict(value)


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetAuditConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetAuditConfigError(f"{label} must be an integer")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetAuditConfigError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise DatasetAuditConfigError(f"{label} must be finite")
    return number


def _as_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DatasetAuditConfigError(f"{label} must be a list")
    return list(value)


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_as_str(item, f"{label} item") for item in _as_sequence(value, label))


def _as_int_tuple(value: object, label: str) -> tuple[int, ...]:
    return tuple(_as_int(item, f"{label} item") for item in _as_sequence(value, label))


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DatasetAuditConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _pin_path(section: Mapping[str, object], key: str, label: str) -> Path:
    return Path(_as_str(section.get(key), f"{label}.{key}"))


def load_audit_config(path: Path | str) -> AuditConfig:
    """Load and strictly validate the M7A audit configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise DatasetAuditConfigError(f"missing audit configuration: {config_path}")
    try:
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:  # pragma: no cover - defensive
        raise DatasetAuditConfigError(f"invalid TOML in {config_path}: {error}") from error

    allowed_sections = {
        "run",
        "schema",
        "inputs",
        "audit",
        "expectations",
        "recommendation",
    }
    _check_keys(payload, allowed_sections, "top-level")

    run_section = _section(payload, "run")
    _check_keys(
        run_section,
        {"name", "version", "seed", "n_bootstrap", "output_reports"},
        "run",
    )
    run = RunSettings(
        name=_as_str(run_section["name"], "run.name"),
        version=_as_str(run_section["version"], "run.version"),
        seed=_as_int(run_section["seed"], "run.seed"),
        n_bootstrap=_as_int(run_section["n_bootstrap"], "run.n_bootstrap"),
        output_reports=Path(_as_str(run_section["output_reports"], "run.output_reports")),
    )
    if run.n_bootstrap < 1:
        raise DatasetAuditConfigError("run.n_bootstrap must be positive")

    schema_section = _section(payload, "schema")
    _check_keys(
        schema_section,
        {
            "feature_set_version",
            "feature_names",
            "target",
            "split_names",
            "claim_schema_version",
            "allowed_classifications",
        },
        "schema",
    )
    schema = SchemaSettings(
        feature_set_version=_as_str(
            schema_section["feature_set_version"], "schema.feature_set_version"
        ),
        feature_names=_as_str_tuple(schema_section["feature_names"], "schema.feature_names"),
        target=_as_str(schema_section["target"], "schema.target"),
        split_names=_as_str_tuple(schema_section["split_names"], "schema.split_names"),
        claim_schema_version=_as_str(
            schema_section["claim_schema_version"], "schema.claim_schema_version"
        ),
        allowed_classifications=_as_str_tuple(
            schema_section["allowed_classifications"], "schema.allowed_classifications"
        ),
    )
    if schema.feature_names != FEATURE_NAMES:
        raise DatasetAuditConfigError(
            f"schema.feature_names must match the frozen eight-feature schema {FEATURE_NAMES}"
        )
    if schema.target != TARGET:
        raise DatasetAuditConfigError(f"schema.target must be {TARGET!r}")
    if schema.split_names != SPLITS:
        raise DatasetAuditConfigError(f"schema.split_names must be {SPLITS}")
    if schema.allowed_classifications != CLASSIFICATIONS:
        raise DatasetAuditConfigError(f"schema.allowed_classifications must be {CLASSIFICATIONS}")
    if schema.claim_schema_version != CLAIM_SCHEMA_VERSION:
        raise DatasetAuditConfigError(
            f"schema.claim_schema_version must be {CLAIM_SCHEMA_VERSION!r}"
        )

    inputs_section = _section(payload, "inputs")
    _check_keys(inputs_section, {"m3", "m5a", "m5c", "m5b", "m3_benchmark"}, "inputs")

    m3_section = _section(inputs_section, "m3")
    _check_keys(
        m3_section,
        {
            "dataset",
            "config",
            "config_sha256",
            "manifest_sha256",
            "sources_sha256",
            "features_sha256",
            "ground_truth_sha256",
            "summary_sha256",
        },
        "inputs.m3",
    )
    m3 = M3Pin(
        dataset=_pin_path(m3_section, "dataset", "inputs.m3"),
        config=_pin_path(m3_section, "config", "inputs.m3"),
        config_sha256=_as_sha256(m3_section["config_sha256"], "inputs.m3.config_sha256"),
        manifest_sha256=_as_sha256(m3_section["manifest_sha256"], "inputs.m3.manifest_sha256"),
        sources_sha256=_as_sha256(m3_section["sources_sha256"], "inputs.m3.sources_sha256"),
        features_sha256=_as_sha256(m3_section["features_sha256"], "inputs.m3.features_sha256"),
        ground_truth_sha256=_as_sha256(
            m3_section["ground_truth_sha256"], "inputs.m3.ground_truth_sha256"
        ),
        summary_sha256=_as_sha256(m3_section["summary_sha256"], "inputs.m3.summary_sha256"),
    )

    m5a_section = _section(inputs_section, "m5a")
    _check_keys(
        m5a_section,
        {
            "dataset",
            "config",
            "config_sha256",
            "manifest_sha256",
            "sources_sha256",
            "features_sha256",
            "summary_sha256",
        },
        "inputs.m5a",
    )
    m5a = M5APin(
        dataset=_pin_path(m5a_section, "dataset", "inputs.m5a"),
        config=_pin_path(m5a_section, "config", "inputs.m5a"),
        config_sha256=_as_sha256(m5a_section["config_sha256"], "inputs.m5a.config_sha256"),
        manifest_sha256=_as_sha256(m5a_section["manifest_sha256"], "inputs.m5a.manifest_sha256"),
        sources_sha256=_as_sha256(m5a_section["sources_sha256"], "inputs.m5a.sources_sha256"),
        features_sha256=_as_sha256(m5a_section["features_sha256"], "inputs.m5a.features_sha256"),
        summary_sha256=_as_sha256(m5a_section["summary_sha256"], "inputs.m5a.summary_sha256"),
    )

    m5c_section = _section(inputs_section, "m5c")
    _check_keys(
        m5c_section,
        {
            "config",
            "config_sha256",
            "protocol_hash",
            "champion_dir",
            "champion_stem",
            "champion_artifact_sha256",
            "champion_card_sha256",
            "predictions",
            "predictions_sha256",
            "test_metrics",
            "test_metrics_sha256",
            "selection_table",
            "selection_table_sha256",
            "run_metadata",
            "run_metadata_sha256",
        },
        "inputs.m5c",
    )
    m5c = M5CPin(
        config=_pin_path(m5c_section, "config", "inputs.m5c"),
        config_sha256=_as_sha256(m5c_section["config_sha256"], "inputs.m5c.config_sha256"),
        protocol_hash=_as_sha256(m5c_section["protocol_hash"], "inputs.m5c.protocol_hash"),
        champion_dir=_pin_path(m5c_section, "champion_dir", "inputs.m5c"),
        champion_stem=_as_str(m5c_section["champion_stem"], "inputs.m5c.champion_stem"),
        champion_artifact_sha256=_as_sha256(
            m5c_section["champion_artifact_sha256"], "inputs.m5c.champion_artifact_sha256"
        ),
        champion_card_sha256=_as_sha256(
            m5c_section["champion_card_sha256"], "inputs.m5c.champion_card_sha256"
        ),
        predictions=_pin_path(m5c_section, "predictions", "inputs.m5c"),
        predictions_sha256=_as_sha256(
            m5c_section["predictions_sha256"], "inputs.m5c.predictions_sha256"
        ),
        test_metrics=_pin_path(m5c_section, "test_metrics", "inputs.m5c"),
        test_metrics_sha256=_as_sha256(
            m5c_section["test_metrics_sha256"], "inputs.m5c.test_metrics_sha256"
        ),
        selection_table=_pin_path(m5c_section, "selection_table", "inputs.m5c"),
        selection_table_sha256=_as_sha256(
            m5c_section["selection_table_sha256"], "inputs.m5c.selection_table_sha256"
        ),
        run_metadata=_pin_path(m5c_section, "run_metadata", "inputs.m5c"),
        run_metadata_sha256=_as_sha256(
            m5c_section["run_metadata_sha256"], "inputs.m5c.run_metadata_sha256"
        ),
    )

    m5b_section = _section(inputs_section, "m5b")
    _check_keys(
        m5b_section,
        {
            "config",
            "config_sha256",
            "data_accounting",
            "data_accounting_sha256",
            "split_assignment",
            "split_assignment_sha256",
        },
        "inputs.m5b",
    )
    m5b = M5BPin(
        config=_pin_path(m5b_section, "config", "inputs.m5b"),
        config_sha256=_as_sha256(m5b_section["config_sha256"], "inputs.m5b.config_sha256"),
        data_accounting=_pin_path(m5b_section, "data_accounting", "inputs.m5b"),
        data_accounting_sha256=_as_sha256(
            m5b_section["data_accounting_sha256"], "inputs.m5b.data_accounting_sha256"
        ),
        split_assignment=_pin_path(m5b_section, "split_assignment", "inputs.m5b"),
        split_assignment_sha256=_as_sha256(
            m5b_section["split_assignment_sha256"], "inputs.m5b.split_assignment_sha256"
        ),
    )

    m3_benchmark_section = _section(inputs_section, "m3_benchmark")
    _check_keys(
        m3_benchmark_section,
        {
            "config",
            "config_sha256",
            "heldout_font_metrics",
            "heldout_font_metrics_sha256",
            "model_comparison",
            "model_comparison_sha256",
        },
        "inputs.m3_benchmark",
    )
    m3_benchmark = M3BenchmarkPin(
        config=_pin_path(m3_benchmark_section, "config", "inputs.m3_benchmark"),
        config_sha256=_as_sha256(
            m3_benchmark_section["config_sha256"], "inputs.m3_benchmark.config_sha256"
        ),
        heldout_font_metrics=_pin_path(
            m3_benchmark_section, "heldout_font_metrics", "inputs.m3_benchmark"
        ),
        heldout_font_metrics_sha256=_as_sha256(
            m3_benchmark_section["heldout_font_metrics_sha256"],
            "inputs.m3_benchmark.heldout_font_metrics_sha256",
        ),
        model_comparison=_pin_path(m3_benchmark_section, "model_comparison", "inputs.m3_benchmark"),
        model_comparison_sha256=_as_sha256(
            m3_benchmark_section["model_comparison_sha256"],
            "inputs.m3_benchmark.model_comparison_sha256",
        ),
    )

    audit_section = _section(payload, "audit")
    _check_keys(
        audit_section,
        {
            "group_column",
            "bootstrap_method",
            "confidence_level",
            "near_degenerate_sd",
            "redundancy_abs_correlation",
            "top_k_sources",
            "min_sources_for_interval",
            "feature_slice_dimensions",
            "bootstrap_scopes",
        },
        "audit",
    )
    audit = AuditSettings(
        group_column=_as_str(audit_section["group_column"], "audit.group_column"),
        bootstrap_method=_as_str(audit_section["bootstrap_method"], "audit.bootstrap_method"),
        confidence_level=_as_float(audit_section["confidence_level"], "audit.confidence_level"),
        near_degenerate_sd=_as_float(
            audit_section["near_degenerate_sd"], "audit.near_degenerate_sd"
        ),
        redundancy_abs_correlation=_as_float(
            audit_section["redundancy_abs_correlation"], "audit.redundancy_abs_correlation"
        ),
        top_k_sources=_as_int(audit_section["top_k_sources"], "audit.top_k_sources"),
        min_sources_for_interval=_as_int(
            audit_section["min_sources_for_interval"], "audit.min_sources_for_interval"
        ),
        feature_slice_dimensions=_as_str_tuple(
            audit_section["feature_slice_dimensions"], "audit.feature_slice_dimensions"
        ),
        bootstrap_scopes=_as_str_tuple(audit_section["bootstrap_scopes"], "audit.bootstrap_scopes"),
    )
    if audit.group_column != "source_id":
        raise DatasetAuditConfigError("audit.group_column must be 'source_id'")
    if audit.bootstrap_method != "source_grouped_percentile":
        raise DatasetAuditConfigError("audit.bootstrap_method must be 'source_grouped_percentile'")
    if not 0.0 < audit.confidence_level < 1.0:
        raise DatasetAuditConfigError("audit.confidence_level must be between 0 and 1")
    if audit.near_degenerate_sd <= 0.0:
        raise DatasetAuditConfigError("audit.near_degenerate_sd must be positive")
    if not 0.0 < audit.redundancy_abs_correlation <= 1.0:
        raise DatasetAuditConfigError("audit.redundancy_abs_correlation must be in (0, 1]")
    if audit.top_k_sources < 1:
        raise DatasetAuditConfigError("audit.top_k_sources must be positive")
    if audit.min_sources_for_interval < 2:
        raise DatasetAuditConfigError("audit.min_sources_for_interval must be at least 2")
    unknown_scopes = sorted(set(audit.bootstrap_scopes) - set(SCOPE_ORDER))
    if unknown_scopes:
        raise DatasetAuditConfigError(f"unknown bootstrap scope(s): {unknown_scopes}")

    expectations_section = _section(payload, "expectations")
    _check_keys(
        expectations_section,
        {
            "m3_sources",
            "m3_samples",
            "m5a_sources",
            "m5a_samples",
            "expanded_train_rows",
            "expanded_val_rows",
            "primary_test_rows",
            "diagnostic_test_rows",
            "m3_test_rows",
            "n_sources",
            "n_fonts",
        },
        "expectations",
    )
    expectations = Expectations(
        m3_sources=_as_int_tuple(expectations_section["m3_sources"], "expectations.m3_sources"),
        m3_samples=_as_int_tuple(expectations_section["m3_samples"], "expectations.m3_samples"),
        m5a_sources=_as_int_tuple(expectations_section["m5a_sources"], "expectations.m5a_sources"),
        m5a_samples=_as_int_tuple(expectations_section["m5a_samples"], "expectations.m5a_samples"),
        expanded_train_rows=_as_int(
            expectations_section["expanded_train_rows"], "expectations.expanded_train_rows"
        ),
        expanded_val_rows=_as_int(
            expectations_section["expanded_val_rows"], "expectations.expanded_val_rows"
        ),
        primary_test_rows=_as_int(
            expectations_section["primary_test_rows"], "expectations.primary_test_rows"
        ),
        diagnostic_test_rows=_as_int(
            expectations_section["diagnostic_test_rows"], "expectations.diagnostic_test_rows"
        ),
        m3_test_rows=_as_int(expectations_section["m3_test_rows"], "expectations.m3_test_rows"),
        n_sources=_as_int(expectations_section["n_sources"], "expectations.n_sources"),
        n_fonts=_as_int(expectations_section["n_fonts"], "expectations.n_fonts"),
    )
    for label, values in (
        ("expectations.m3_sources", expectations.m3_sources),
        ("expectations.m3_samples", expectations.m3_samples),
        ("expectations.m5a_sources", expectations.m5a_sources),
        ("expectations.m5a_samples", expectations.m5a_samples),
    ):
        if len(values) != 3:
            raise DatasetAuditConfigError(f"{label} must contain three split counts")

    recommendation_section = _section(payload, "recommendation")
    _check_keys(
        recommendation_section,
        {"selected_direction", "directions"},
        "recommendation",
    )
    recommendation = RecommendationSettings(
        selected_direction=_as_str(
            recommendation_section["selected_direction"], "recommendation.selected_direction"
        ),
        directions=_as_str_tuple(recommendation_section["directions"], "recommendation.directions"),
    )
    if recommendation.directions != DIRECTION_IDS:
        raise DatasetAuditConfigError(f"recommendation.directions must be {DIRECTION_IDS}")
    if recommendation.selected_direction not in recommendation.directions:
        raise DatasetAuditConfigError(
            "recommendation.selected_direction must be one of the declared directions"
        )

    return AuditConfig(
        run=run,
        schema=schema,
        m3=m3,
        m5a=m5a,
        m5c=m5c,
        m5b=m5b,
        m3_benchmark=m3_benchmark,
        audit=audit,
        expectations=expectations,
        recommendation=recommendation,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )


@dataclass(frozen=True, slots=True)
class AuditInputs:
    """Verified frozen inputs consumed by the read-only audit."""

    config: AuditConfig
    frames: gb.BenchmarkFrames
    m5c_config: ra.RetentionConfig
    m5b_config: gb.GeometricBenchmarkConfig
    champion_model: Any
    champion_card: dict[str, Any]
    champion_name: str
    predictions: pd.DataFrame
    stored_test_metrics: pd.DataFrame
    selection_table: pd.DataFrame
    heldout_font_metrics: pd.DataFrame
    model_comparison: pd.DataFrame
    m3_features: pd.DataFrame
    m5a_features: pd.DataFrame
    input_hashes: dict[str, str]
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise DatasetAuditError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise DatasetAuditError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise DatasetAuditError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetAuditError(f"missing {label}: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DatasetAuditError(f"{label} must contain a JSON object")
    return payload


def _require_all_checks(checks: Sequence[CheckResult], label: str) -> None:
    failures = [check for check in checks if not check.passed]
    if failures:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failures)
        raise DatasetAuditError(f"{label} failed: {detail}")


def scope_frames(frames: gb.BenchmarkFrames) -> dict[str, pd.DataFrame]:
    """Return the frozen evaluation scopes in their predeclared order."""
    m3_test = frames.m3_test
    primary = frames.primary_test
    original = m3_test[~m3_test["is_clean"]]
    result = {
        "m3_test_overall": m3_test,
        "clean_reference": m3_test[m3_test["is_clean"]],
        "original_blur_noise": original,
        "original_blur_noise_gaussian_blur": original[original["family"] == "gaussian_blur"],
        "original_blur_noise_gaussian_noise": original[original["family"] == "gaussian_noise"],
        "geometric_primary": primary,
        "diagnostic_0_5": frames.diagnostic_test,
    }
    for severity in (2, 3, 4):
        result[f"geometric_primary_severity_{severity}"] = primary[
            primary["severity_level"] == severity
        ]
    return {scope: result[scope] for scope in SCOPE_ORDER}


def _feature_schema_check(features: pd.DataFrame, label: str) -> CheckResult:
    expected = ["sample_id", *FEATURE_NAMES]
    passed = list(features.columns) == expected
    forbidden = sorted(
        set(features.columns)
        & {
            "ssim_target",
            "psnr_target",
            "target",
            "severity_level",
            "family",
            "font_id",
            "source_id",
            "split",
        }
    )
    if forbidden:
        passed = False
    finite = bool(np.isfinite(features[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)).all())
    passed = passed and finite and features["sample_id"].is_unique
    return CheckResult(
        name=f"{label}_feature_schema",
        passed=passed,
        detail=(
            f"{len(features)} rows x {len(FEATURE_NAMES)} features; "
            f"columns match={list(features.columns) == expected}; "
            f"forbidden={forbidden}; finite={finite}; unique_ids={features['sample_id'].is_unique}"
        ),
    )


def _source_split_check(sources: pd.DataFrame, label: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    crossing = sources.groupby("source_id")["split"].nunique()
    checks.append(
        CheckResult(
            name=f"{label}_source_split_isolation",
            passed=bool((crossing == 1).all()),
            detail=f"{crossing.size} source(s); max splits per source={int(crossing.max())}",
        )
    )
    counts = sources.groupby("split")["source_id"].nunique().reindex(SPLITS, fill_value=0)
    checks.append(
        CheckResult(
            name=f"{label}_split_source_counts",
            passed=bool((counts > 0).all()),
            detail=", ".join(f"{split}={int(counts[split])}" for split in SPLITS),
        )
    )
    font_sets = {
        split: set(sources.loc[sources["split"] == split, "font_id"].astype(str))
        for split in SPLITS
    }
    shared = set.intersection(*font_sets.values())
    checks.append(
        CheckResult(
            name=f"{label}_font_partition_overlap",
            passed=bool(shared),
            detail=(
                "fonts per split: "
                + "; ".join(f"{split}={sorted(font_sets[split])}" for split in SPLITS)
                + f"; shared={sorted(shared)}"
            ),
        )
    )
    return checks


def _verify_prediction_replay(
    inputs_predictions: pd.DataFrame,
    frames: gb.BenchmarkFrames,
    model: Any,
    champion_name: str,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    scopes = scope_frames(frames)
    for scope, frame in scopes.items():
        stored = inputs_predictions[
            (inputs_predictions["model"] == champion_name) & (inputs_predictions["scope"] == scope)
        ].sort_values("sample_id")
        expected_frame = frame.sort_values("sample_id").reset_index(drop=True)
        same_ids = stored["sample_id"].tolist() == expected_frame["sample_id"].tolist()
        max_prediction_delta = math.nan
        max_target_delta = math.nan
        residual_ok = False
        if same_ids and len(stored) == len(expected_frame):
            replayed = predict(model, expected_frame)
            max_prediction_delta = float(
                np.max(np.abs(replayed - stored["prediction"].to_numpy(dtype=np.float64)))
            )
            max_target_delta = float(
                np.max(
                    np.abs(
                        expected_frame[TARGET].to_numpy(dtype=np.float64)
                        - stored[TARGET].to_numpy(dtype=np.float64)
                    )
                )
            )
            residual = stored[TARGET].to_numpy(dtype=np.float64) - stored["prediction"].to_numpy(
                dtype=np.float64
            )
            residual_ok = bool(
                np.allclose(residual, stored["residual"].to_numpy(dtype=np.float64), atol=1e-12)
                and np.allclose(
                    np.abs(residual),
                    stored["abs_residual"].to_numpy(dtype=np.float64),
                    atol=1e-12,
                )
            )
        passed = (
            same_ids
            and len(stored) == len(expected_frame)
            and np.isfinite(max_prediction_delta)
            and max_prediction_delta <= 1e-9
            and max_target_delta <= 1e-12
            and residual_ok
        )
        checks.append(
            CheckResult(
                name=f"prediction_replay_{scope}",
                passed=bool(passed),
                detail=(
                    f"n={len(stored)}/{len(expected_frame)}; same_ids={same_ids}; "
                    f"max_prediction_delta={max_prediction_delta:.3g}; "
                    f"max_target_delta={max_target_delta:.3g}; residual_ok={residual_ok}"
                ),
            )
        )
    return checks


def _verify_stored_metrics(
    predictions: pd.DataFrame,
    frames: gb.BenchmarkFrames,
    stored_metrics: pd.DataFrame,
    champion_name: str,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    scopes = scope_frames(frames)
    for scope, frame in scopes.items():
        stored = predictions[
            (predictions["model"] == champion_name) & (predictions["scope"] == scope)
        ].sort_values("sample_id")
        metric_row = stored_metrics[
            (stored_metrics["model"] == champion_name) & (stored_metrics["scope"] == scope)
        ]
        if len(metric_row) != 1 or len(stored) != len(frame):
            checks.append(
                CheckResult(
                    name=f"stored_metrics_{scope}",
                    passed=False,
                    detail=f"metric_rows={len(metric_row)}; prediction_rows={len(stored)}",
                )
            )
            continue
        aligned = frame.sort_values("sample_id").reset_index(drop=True)
        replayed = score_frame(aligned, stored["prediction"].to_numpy(dtype=np.float64))
        row = metric_row.iloc[0]
        mae_delta = abs(float(replayed["mae"]) - float(row["mae"]))
        rmse_delta = abs(float(replayed["rmse"]) - float(row["rmse"]))
        r2_delta = abs(float(replayed["r2"]) - float(row["r2"])) if bool(row["r2_defined"]) else 0.0
        passed = (
            int(replayed["n_rows"]) == int(row["n_samples"])
            and int(replayed["n_sources"]) == int(row["n_sources"])
            and mae_delta <= 1e-9
            and rmse_delta <= 1e-9
            and r2_delta <= 1e-9
        )
        checks.append(
            CheckResult(
                name=f"stored_metrics_{scope}",
                passed=bool(passed),
                detail=(
                    f"n={int(row['n_samples'])}; mae_delta={mae_delta:.3g}; "
                    f"rmse_delta={rmse_delta:.3g}; r2_delta={r2_delta:.3g}"
                ),
            )
        )
    return checks


def load_audit_inputs(config: AuditConfig, *, verify_images: bool = False) -> AuditInputs:
    """Verify every pinned input and load the frozen frames and champion."""
    checks: list[CheckResult] = []

    pinned: list[tuple[Path, str, str]] = [
        (config.m3.config, config.m3.config_sha256, "M3 config"),
        (config.m3.dataset / "manifest.csv", config.m3.manifest_sha256, "M3 manifest"),
        (config.m3.dataset / "sources.csv", config.m3.sources_sha256, "M3 sources"),
        (config.m3.dataset / "features.csv", config.m3.features_sha256, "M3 features"),
        (config.m3.dataset / "ground_truth.csv", config.m3.ground_truth_sha256, "M3 ground truth"),
        (config.m3.dataset / "dataset_summary.json", config.m3.summary_sha256, "M3 summary"),
        (config.m5a.config, config.m5a.config_sha256, "M5A config"),
        (config.m5a.dataset / "manifest.csv", config.m5a.manifest_sha256, "M5A manifest"),
        (config.m5a.dataset / "sources.csv", config.m5a.sources_sha256, "M5A sources"),
        (config.m5a.dataset / "features.csv", config.m5a.features_sha256, "M5A features"),
        (config.m5a.dataset / "pilot_summary.json", config.m5a.summary_sha256, "M5A summary"),
        (config.m5c.config, config.m5c.config_sha256, "M5C config"),
        (
            config.m5c.champion_dir / f"{config.m5c.champion_stem}.joblib",
            config.m5c.champion_artifact_sha256,
            "M5C champion artifact",
        ),
        (
            config.m5c.champion_dir / f"{config.m5c.champion_stem}_card.json",
            config.m5c.champion_card_sha256,
            "M5C champion card",
        ),
        (config.m5c.predictions, config.m5c.predictions_sha256, "M5C predictions"),
        (config.m5c.test_metrics, config.m5c.test_metrics_sha256, "M5C test metrics"),
        (config.m5c.selection_table, config.m5c.selection_table_sha256, "M5C selection table"),
        (config.m5c.run_metadata, config.m5c.run_metadata_sha256, "M5C run metadata"),
        (config.m5b.config, config.m5b.config_sha256, "M5B config"),
        (config.m5b.data_accounting, config.m5b.data_accounting_sha256, "M5B data accounting"),
        (config.m5b.split_assignment, config.m5b.split_assignment_sha256, "M5B split assignment"),
        (config.m3_benchmark.config, config.m3_benchmark.config_sha256, "M3 benchmark config"),
        (
            config.m3_benchmark.heldout_font_metrics,
            config.m3_benchmark.heldout_font_metrics_sha256,
            "M3 held-out-font metrics",
        ),
        (
            config.m3_benchmark.model_comparison,
            config.m3_benchmark.model_comparison_sha256,
            "M3 model comparison",
        ),
    ]
    input_hashes: dict[str, str] = {}
    for path, expected, label in pinned:
        _require_file_hash(path, expected, label)
        input_hashes[str(path)] = sha256_file(path)
    checks.append(
        CheckResult(
            name="pinned_input_hashes",
            passed=True,
            detail=f"{len(pinned)} pinned file(s) match their SHA-256 values",
        )
    )

    m5c_config = ra.load_retention_config(config.m5c.config)
    if m5c_config.config_hash != config.m5c.protocol_hash:
        raise DatasetAuditError(
            "M5C protocol hash mismatch: "
            f"expected {config.m5c.protocol_hash}, found {m5c_config.config_hash}"
        )
    checks.append(
        CheckResult(
            name="m5c_protocol_hash",
            passed=True,
            detail=m5c_config.config_hash,
        )
    )

    m5b_config = gb.load_geometric_benchmark_config(config.m5b.config)
    if m5b_config.config_hash != m5c_config.m5b.protocol_hash:
        raise DatasetAuditError(
            "M5B protocol hash does not match the M5C pin: "
            f"{m5b_config.config_hash} != {m5c_config.m5b.protocol_hash}"
        )
    checks.append(
        CheckResult(
            name="m5b_protocol_hash",
            passed=True,
            detail=m5b_config.config_hash,
        )
    )

    frames = gb.load_benchmark_frames(m5b_config, verify_images=verify_images)
    checks.extend(frames.checks)
    checks.append(
        CheckResult(
            name="frozen_benchmark_frames",
            passed=frames.ok,
            detail=(
                f"M3={len(frames.m3_frame)} rows/{frames.m3_frame['source_id'].nunique()} sources; "
                f"M5A={len(frames.m5a_frame)} rows/{frames.m5a_frame['source_id'].nunique()} sources"
            ),
        )
    )

    m3_features = _read_csv(config.m3.dataset / "features.csv", "M3 features")
    m5a_features = _read_csv(config.m5a.dataset / "features.csv", "M5A features")
    checks.append(_feature_schema_check(m3_features, "m3"))
    checks.append(_feature_schema_check(m5a_features, "m5a"))

    m3_sources = frames.m3_sources
    m5a_sources = frames.m5a_sources
    checks.extend(_source_split_check(m3_sources, "m3"))
    checks.extend(_source_split_check(m5a_sources, "m5a"))

    m3_source_set = set(m3_sources["source_id"].astype(str))
    m5a_source_set = set(m5a_sources["source_id"].astype(str))
    checks.append(
        CheckResult(
            name="source_identity_across_datasets",
            passed=m3_source_set == m5a_source_set,
            detail=(
                f"M3={len(m3_source_set)} sources; M5A={len(m5a_source_set)} sources; "
                f"symmetric_difference={len(m3_source_set ^ m5a_source_set)}"
            ),
        )
    )
    joined = m3_sources[["source_id", "split", "font_id", "text_length"]].merge(
        m5a_sources[["source_id", "split", "font_id", "text_length"]],
        on="source_id",
        suffixes=("_m3", "_m5a"),
        validate="one_to_one",
    )
    metadata_match = bool(
        (joined["split_m3"] == joined["split_m5a"]).all()
        and (joined["font_id_m3"] == joined["font_id_m5a"]).all()
        and (joined["text_length_m3"] == joined["text_length_m5a"]).all()
    )
    checks.append(
        CheckResult(
            name="source_metadata_consistency",
            passed=metadata_match,
            detail=f"{len(joined)} shared sources; split/font/text_length identical={metadata_match}",
        )
    )

    expected = config.expectations
    m3_counts = (
        frames.m3_sources.groupby("split")["source_id"].nunique().reindex(SPLITS, fill_value=0)
    )
    m5a_counts = (
        frames.m5a_sources.groupby("split")["source_id"].nunique().reindex(SPLITS, fill_value=0)
    )
    m3_rows = frames.m3_frame.groupby("split").size().reindex(SPLITS, fill_value=0)
    m5a_rows = frames.m5a_frame.groupby("split").size().reindex(SPLITS, fill_value=0)
    split_counts_ok = (
        tuple(int(value) for value in m3_counts) == expected.m3_sources
        and tuple(int(value) for value in m5a_counts) == expected.m5a_sources
        and tuple(int(value) for value in m3_rows) == expected.m3_samples
        and tuple(int(value) for value in m5a_rows) == expected.m5a_samples
    )
    checks.append(
        CheckResult(
            name="declared_split_counts",
            passed=split_counts_ok,
            detail=(
                f"M3 sources={tuple(int(value) for value in m3_counts)}, rows={tuple(int(value) for value in m3_rows)}; "
                f"M5A sources={tuple(int(value) for value in m5a_counts)}, rows={tuple(int(value) for value in m5a_rows)}"
            ),
        )
    )

    expanded_counts_ok = (
        len(frames.expanded_train) == expected.expanded_train_rows
        and len(frames.expanded_val) == expected.expanded_val_rows
        and len(frames.primary_test) == expected.primary_test_rows
        and len(frames.diagnostic_test) == expected.diagnostic_test_rows
        and len(frames.m3_test) == expected.m3_test_rows
        and frames.m3_frame["source_id"].nunique() == expected.n_sources
        and frames.m3_sources["font_id"].nunique() == expected.n_fonts
    )
    checks.append(
        CheckResult(
            name="declared_frame_counts",
            passed=expanded_counts_ok,
            detail=(
                f"expanded_train={len(frames.expanded_train)}; expanded_val={len(frames.expanded_val)}; "
                f"primary_test={len(frames.primary_test)}; diagnostic_test={len(frames.diagnostic_test)}; "
                f"m3_test={len(frames.m3_test)}; sources={frames.m3_frame['source_id'].nunique()}; "
                f"fonts={frames.m3_sources['font_id'].nunique()}"
            ),
        )
    )

    m3_test = frames.m3_test
    m5a_test = frames.m5a_frame[frames.m5a_frame["split"] == "test"]
    m3_clean = int((m3_test["family"] == gb.M3_CLEAN_FAMILY).sum())
    m3_original = int((m3_test["family"] != gb.M3_CLEAN_FAMILY).sum())
    m5a_clean = int((m5a_test["family"] == gb.CLEAN_FAMILY).sum())
    m5a_control = int((m5a_test["family"] == gb.CONTROL_FAMILY).sum())
    m5a_primary = int(
        (
            (m5a_test["family"] == gb.ELASTIC_FAMILY)
            & (m5a_test["amplitude_px"].isin(gb.PRIMARY_AMPLITUDES_PX))
        ).sum()
    )
    m5a_diagnostic = int(
        (
            (m5a_test["family"] == gb.ELASTIC_FAMILY)
            & (m5a_test["amplitude_px"].isin(gb.DIAGNOSTIC_AMPLITUDES_PX))
        ).sum()
    )
    target_accounting_ok = (
        m3_clean + m3_original == len(m3_test) == expected.m3_test_rows
        and m5a_clean + m5a_control + m5a_primary + m5a_diagnostic == len(m5a_test) == 96
        and m5a_primary == expected.primary_test_rows
        and m5a_diagnostic == expected.diagnostic_test_rows
    )
    checks.append(
        CheckResult(
            name="target_slice_accounting",
            passed=target_accounting_ok,
            detail=(
                f"M3 test clean={m3_clean}, original={m3_original}, total={len(m3_test)}; "
                f"M5A test clean={m5a_clean}, control={m5a_control}, primary={m5a_primary}, "
                f"diagnostic={m5a_diagnostic}, total={len(m5a_test)}"
            ),
        )
    )

    clean_target_constant = bool(
        np.allclose(m3_test.loc[m3_test["is_clean"], TARGET].to_numpy(dtype=np.float64), 1.0)
        and np.allclose(
            m5a_test.loc[m5a_test["family"] == gb.CLEAN_FAMILY, TARGET].to_numpy(dtype=np.float64),
            1.0,
        )
    )
    checks.append(
        CheckResult(
            name="clean_target_constant",
            passed=clean_target_constant,
            detail="all test clean-reference targets equal 1.0 (zero variance)",
        )
    )

    control_identity = m5a_test[m5a_test["control_id"] == "identity"]
    control_zero = m5a_test[m5a_test["control_id"] == "zero_field"]
    control_minimal = m5a_test[m5a_test["control_id"] == "minimal"]
    controls_ok = bool(
        len(control_identity) == 12
        and len(control_zero) == 12
        and len(control_minimal) == 12
        and np.allclose(control_identity[TARGET], 1.0)
        and np.allclose(control_zero[TARGET], 1.0)
        and (control_minimal[TARGET] < 1.0).all()
        and (control_minimal[TARGET] > 0.99).all()
    )
    checks.append(
        CheckResult(
            name="control_relationship",
            passed=controls_ok,
            detail=(
                f"identity={len(control_identity)}, zero_field={len(control_zero)}, "
                f"minimal={len(control_minimal)}; identity/zero_field targets=1.0; "
                "minimal is near-clean but not exact"
            ),
        )
    )

    split_assignment = _read_csv(config.m5b.split_assignment, "M5B split assignment")
    assignment_match = True
    if {"source_id", "split"}.issubset(split_assignment.columns):
        recorded = split_assignment.set_index("source_id")["split"].astype(str).to_dict()
        observed = m3_sources.set_index("source_id")["split"].astype(str).to_dict()
        assignment_match = recorded == observed
    checks.append(
        CheckResult(
            name="m5b_split_assignment_matches_manifests",
            passed=assignment_match,
            detail=f"{len(split_assignment)} recorded source assignment(s)",
        )
    )

    selection_table = _read_csv(config.m5c.selection_table, "M5C selection table")
    selected = selection_table[selection_table["selected"].astype(bool)]
    if len(selected) != 1:
        raise DatasetAuditError(
            f"M5C selection table must select exactly one model, found {len(selected)}"
        )
    champion_name = str(selected.iloc[0]["model"])
    checks.append(
        CheckResult(
            name="champion_selection",
            passed=True,
            detail=f"selected model={champion_name}",
        )
    )

    import joblib  # noqa: PLC0415 - optional ML dependency is loaded only on demand

    card_path = config.m5c.champion_dir / f"{config.m5c.champion_stem}_card.json"
    champion_card = _read_json(card_path, "M5C champion card")
    if champion_card.get("artifact_sha256") != config.m5c.champion_artifact_sha256:
        raise DatasetAuditError("champion card artifact hash does not match the audit pin")
    if champion_card.get("feature_names") != list(FEATURE_NAMES):
        raise DatasetAuditError("champion card feature names do not match the frozen schema")
    if champion_card.get("target") != TARGET:
        raise DatasetAuditError("champion card target does not match the frozen target")
    if champion_card.get("calibration") != "none":
        raise DatasetAuditError("M7A expects the uncalibrated champion")
    champion_model = joblib.load(config.m5c.champion_dir / f"{config.m5c.champion_stem}.joblib")
    if getattr(champion_model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise DatasetAuditError("champion model input width does not match the frozen schema")
    if not hasattr(champion_model, "feature_importances_"):
        raise DatasetAuditError("champion model does not expose feature_importances_")
    checks.append(
        CheckResult(
            name="champion_artifact",
            passed=True,
            detail=(
                f"artifact={config.m5c.champion_artifact_sha256}; "
                f"features={champion_model.n_features_in_}; calibration=none"
            ),
        )
    )

    predictions_all = _read_csv(config.m5c.predictions, "M5C predictions")
    if champion_name not in set(predictions_all["model"].astype(str)):
        raise DatasetAuditError(f"champion {champion_name!r} is absent from the prediction table")
    predictions = predictions_all[predictions_all["model"] == champion_name].copy()
    stored_test_metrics = _read_csv(config.m5c.test_metrics, "M5C test metrics")
    stored_test_metrics = stored_test_metrics[stored_test_metrics["model"] == champion_name].copy()
    if stored_test_metrics.empty:
        raise DatasetAuditError(f"champion {champion_name!r} is absent from the test metrics")

    checks.extend(_verify_prediction_replay(predictions, frames, champion_model, champion_name))
    checks.extend(_verify_stored_metrics(predictions, frames, stored_test_metrics, champion_name))

    heldout_font_metrics = _read_csv(
        config.m3_benchmark.heldout_font_metrics, "M3 held-out-font metrics"
    )
    model_comparison = _read_csv(config.m3_benchmark.model_comparison, "M3 model comparison")

    _require_all_checks(checks, "M7A input verification")
    return AuditInputs(
        config=config,
        frames=frames,
        m5c_config=m5c_config,
        m5b_config=m5b_config,
        champion_model=champion_model,
        champion_card=champion_card,
        champion_name=champion_name,
        predictions=predictions,
        stored_test_metrics=stored_test_metrics,
        selection_table=selection_table,
        heldout_font_metrics=heldout_font_metrics,
        model_comparison=model_comparison,
        m3_features=m3_features,
        m5a_features=m5a_features,
        input_hashes=input_hashes,
        checks=tuple(checks),
    )


def _count(frame: pd.DataFrame, mask: pd.Series | np.ndarray | None = None) -> int:
    if mask is None:
        return int(len(frame))
    return int(np.asarray(mask, dtype=bool).sum())


def _font_paths(m3_summary: Mapping[str, Any]) -> dict[str, str]:
    fonts = m3_summary.get("fonts", {})
    resolved = fonts.get("resolved", []) if isinstance(fonts, Mapping) else []
    result: dict[str, str] = {}
    for entry in resolved:
        if isinstance(entry, Mapping):
            result[str(entry.get("key"))] = str(entry.get("path"))
    return result


def build_dataset_inventory(inputs: AuditInputs) -> pd.DataFrame:
    """Build one source-level inventory row per synthetic source."""
    frames = inputs.frames
    m3 = frames.m3_frame
    m5a = frames.m5a_frame
    m3_sources = frames.m3_sources.set_index("source_id")
    m5a_sources = frames.m5a_sources.set_index("source_id")
    font_paths = _font_paths(frames.m3_summary)

    rows: list[dict[str, Any]] = []
    for source_id in sorted(m3_sources.index):
        m3_source = m3_sources.loc[source_id]
        m5a_source = m5a_sources.loc[source_id]
        m3_rows = m3[m3["source_id"] == source_id]
        m5a_rows = m5a[m5a["source_id"] == source_id]
        field_spec: dict[str, Any] = {}
        try:
            field_spec = json.loads(str(m5a_source.get("field_spec_json", "{}")))
        except json.JSONDecodeError:  # pragma: no cover - verified upstream
            field_spec = {}
        m3_clean = m3_rows[m3_rows["family"] == gb.M3_CLEAN_FAMILY]
        m5a_clean = m5a_rows[m5a_rows["family"] == gb.CLEAN_FAMILY]
        m5a_control = m5a_rows[m5a_rows["family"] == gb.CONTROL_FAMILY]
        m5a_elastic = m5a_rows[m5a_rows["family"] == gb.ELASTIC_FAMILY]
        m5a_primary = m5a_elastic[m5a_elastic["amplitude_px"].isin(gb.PRIMARY_AMPLITUDES_PX)]
        m5a_diagnostic = m5a_elastic[m5a_elastic["amplitude_px"].isin(gb.DIAGNOSTIC_AMPLITUDES_PX)]
        rows.append(
            {
                "source_id": source_id,
                "split": str(m3_source["split"]),
                "font_id": str(m3_source["font_id"]),
                "font_path": font_paths.get(str(m3_source["font_id"]), ""),
                "text_length": int(m3_source["text_length"]),
                "render_seed": int(m3_source["render_seed"]),
                "background_level": int(m3_source["background_level"]),
                "height": int(m3_source["height"]),
                "width": int(m3_source["width"]),
                "clean_sha256": str(m3_source["clean_image_sha256"]),
                "m3_samples": len(m3_rows),
                "m3_clean": len(m3_clean),
                "m3_gaussian_blur": int((m3_rows["family"] == "gaussian_blur").sum()),
                "m3_gaussian_noise": int((m3_rows["family"] == "gaussian_noise").sum()),
                "m3_distinct_images": int(m3_rows["image_sha256"].nunique()),
                "m5a_samples": len(m5a_rows),
                "m5a_clean": len(m5a_clean),
                "m5a_control": len(m5a_control),
                "m5a_control_identity": int((m5a_control["control_id"] == "identity").sum()),
                "m5a_control_zero_field": int((m5a_control["control_id"] == "zero_field").sum()),
                "m5a_control_minimal": int((m5a_control["control_id"] == "minimal").sum()),
                "m5a_elastic": len(m5a_elastic),
                "m5a_primary": len(m5a_primary),
                "m5a_diagnostic": len(m5a_diagnostic),
                "m5a_distinct_images": int(m5a_rows["image_sha256"].nunique()),
                "field_grid_rows": int(field_spec.get("grid_rows", 0)),
                "field_grid_cols": int(field_spec.get("grid_cols", 0)),
                "field_smoothing_sigma_px": float(field_spec.get("smoothing_sigma_px", math.nan)),
                "field_interpolation": str(field_spec.get("interpolation", "")),
                "field_border_mode": str(field_spec.get("border_mode", "")),
                "field_seed": int(field_spec.get("seed", 0)),
                "field_sha256": str(field_spec.get("field_sha256", "")),
                "clean_hash_matches": bool(
                    str(m3_source["clean_image_sha256"]) == str(m5a_source["clean_sha256"])
                ),
                "renderer": "Pillow ImageDraw + FreeType",
                "pixel_format": "8-bit grayscale PNG",
                "source_shared_across_datasets": True,
            }
        )
    order = {split: index for index, split in enumerate(SPLITS)}
    inventory = pd.DataFrame(rows)
    inventory["_split_order"] = inventory["split"].map(order)
    return (
        inventory.sort_values(["_split_order", "font_id", "source_id"])
        .drop(columns=["_split_order"])
        .reset_index(drop=True)
    )


def _composition_row(
    *,
    dataset: str,
    split: str,
    dimension: str,
    value: str,
    n_samples: int,
    n_sources: int,
    severity_level: int = -1,
    amplitude_px: float = math.nan,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "dimension": dimension,
        "value": value,
        "severity_level": severity_level,
        "amplitude_px": amplitude_px,
        "n_samples": n_samples,
        "n_sources": n_sources,
        "detail": detail,
    }


def build_dataset_composition(inputs: AuditInputs) -> pd.DataFrame:
    """Build a long-form composition table for both frozen datasets."""
    frames = inputs.frames
    m3 = frames.m3_frame
    m5a = frames.m5a_frame
    rows: list[dict[str, Any]] = []
    for split in ("all", *SPLITS):
        m3_part = m3 if split == "all" else m3[m3["split"] == split]
        m5a_part = m5a if split == "all" else m5a[m5a["split"] == split]
        for dataset, part, family_column in (
            ("m3", m3_part, "family"),
            ("m5a", m5a_part, "family"),
        ):
            if part.empty:
                continue
            for dimension in ("font_id", "text_length"):
                for value, group in part.groupby(dimension, dropna=False):
                    rows.append(
                        _composition_row(
                            dataset=dataset,
                            split=split,
                            dimension=dimension,
                            value=str(value),
                            n_samples=len(group),
                            n_sources=int(group["source_id"].nunique()),
                        )
                    )
            rows.append(
                _composition_row(
                    dataset=dataset,
                    split=split,
                    dimension="resolution",
                    value="192x64",
                    n_samples=len(part),
                    n_sources=int(part["source_id"].nunique()),
                    detail="single canvas size; no alternative resolution in the frozen data",
                )
            )
            rows.append(
                _composition_row(
                    dataset=dataset,
                    split=split,
                    dimension="pixel_format",
                    value="8-bit grayscale PNG",
                    n_samples=len(part),
                    n_sources=int(part["source_id"].nunique()),
                    detail="single image contract; no RGB, JPEG or alternative bit depth",
                )
            )
            rows.append(
                _composition_row(
                    dataset=dataset,
                    split=split,
                    dimension="renderer",
                    value="Pillow ImageDraw + FreeType",
                    n_samples=len(part),
                    n_sources=int(part["source_id"].nunique()),
                    detail="single renderer family; fonts are system TrueType files",
                )
            )
            if dataset == "m3":
                for (family, severity), group in part.groupby([family_column, "severity_level"]):
                    rows.append(
                        _composition_row(
                            dataset=dataset,
                            split=split,
                            dimension="corruption_family",
                            value=str(family),
                            severity_level=int(severity),
                            n_samples=len(group),
                            n_sources=int(group["source_id"].nunique()),
                            detail="clean reference or blur/noise corruption ladder",
                        )
                    )
            else:
                for (family, severity, amplitude), group in part.groupby(
                    [family_column, "severity_level", "amplitude_px"], dropna=False
                ):
                    detail = "clean reference or derived elastic variant"
                    if family == gb.CONTROL_FAMILY:
                        detail = "identity and zero-field controls are exact clean duplicates; minimal is near-clean"
                    rows.append(
                        _composition_row(
                            dataset=dataset,
                            split=split,
                            dimension="variant_family",
                            value=str(family),
                            severity_level=int(severity),
                            amplitude_px=float(amplitude),
                            n_samples=len(group),
                            n_sources=int(group["source_id"].nunique()),
                            detail=detail,
                        )
                    )
                for control_id, group in part[part["family"] == gb.CONTROL_FAMILY].groupby(
                    "control_id"
                ):
                    rows.append(
                        _composition_row(
                            dataset=dataset,
                            split=split,
                            dimension="control_type",
                            value=str(control_id),
                            n_samples=len(group),
                            n_sources=int(group["source_id"].nunique()),
                            detail=(
                                "exact clean duplicate"
                                if control_id in {"identity", "zero_field"}
                                else "near-clean control"
                            ),
                        )
                    )
    composition = pd.DataFrame(rows)
    split_order = {"all": 0, **{split: index + 1 for index, split in enumerate(SPLITS)}}
    composition["_split_order"] = composition["split"].map(split_order)
    return (
        composition.sort_values(
            ["dataset", "_split_order", "dimension", "value", "severity_level", "amplitude_px"]
        )
        .drop(columns=["_split_order"])
        .reset_index(drop=True)
    )


def _account_row(
    *,
    dataset: str,
    split: str,
    frame: pd.DataFrame,
    expected_sources: int,
    expected_samples: int,
) -> dict[str, Any]:
    clean = int((frame["family"].isin([gb.M3_CLEAN_FAMILY, gb.CLEAN_FAMILY])).sum())
    control = int((frame["family"] == gb.CONTROL_FAMILY).sum())
    amplitude = (
        frame["amplitude_px"]
        if "amplitude_px" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    primary = int(
        ((frame["family"] == gb.ELASTIC_FAMILY) & (amplitude.isin(gb.PRIMARY_AMPLITUDES_PX))).sum()
    )
    diagnostic = int(
        (
            (frame["family"] == gb.ELASTIC_FAMILY) & (amplitude.isin(gb.DIAGNOSTIC_AMPLITUDES_PX))
        ).sum()
    )
    blur = int((frame["family"] == "gaussian_blur").sum())
    noise = int((frame["family"] == "gaussian_noise").sum())
    sources = int(frame["source_id"].nunique())
    return {
        "dataset": dataset,
        "split": split,
        "n_sources": sources,
        "n_samples": len(frame),
        "clean": clean,
        "control": control,
        "primary_elastic": primary,
        "diagnostic_elastic": diagnostic,
        "gaussian_blur": blur,
        "gaussian_noise": noise,
        "expected_sources": expected_sources,
        "expected_samples": expected_samples,
        "matches_expectation": bool(sources == expected_sources and len(frame) == expected_samples),
    }


def build_split_accounting(inputs: AuditInputs) -> pd.DataFrame:
    """Account for every frozen row and source by split and evaluation scope."""
    frames = inputs.frames
    expected = inputs.config.expectations
    rows: list[dict[str, Any]] = []
    for index, split in enumerate(SPLITS):
        rows.append(
            _account_row(
                dataset="m3",
                split=split,
                frame=frames.m3_frame[frames.m3_frame["split"] == split],
                expected_sources=expected.m3_sources[index],
                expected_samples=expected.m3_samples[index],
            )
        )
        rows.append(
            _account_row(
                dataset="m5a",
                split=split,
                frame=frames.m5a_frame[frames.m5a_frame["split"] == split],
                expected_sources=expected.m5a_sources[index],
                expected_samples=expected.m5a_samples[index],
            )
        )
    rows.append(
        _account_row(
            dataset="expanded",
            split="train",
            frame=frames.expanded_train,
            expected_sources=expected.m3_sources[0],
            expected_samples=expected.expanded_train_rows,
        )
    )
    rows.append(
        _account_row(
            dataset="expanded",
            split="val",
            frame=frames.expanded_val,
            expected_sources=expected.m3_sources[1],
            expected_samples=expected.expanded_val_rows,
        )
    )
    rows.append(
        _account_row(
            dataset="evaluation",
            split="primary_test",
            frame=frames.primary_test,
            expected_sources=expected.m3_sources[2],
            expected_samples=expected.primary_test_rows,
        )
    )
    rows.append(
        _account_row(
            dataset="evaluation",
            split="diagnostic_0_5",
            frame=frames.diagnostic_test,
            expected_sources=expected.m3_sources[2],
            expected_samples=expected.diagnostic_test_rows,
        )
    )
    rows.append(
        _account_row(
            dataset="evaluation",
            split="m3_test",
            frame=frames.m3_test,
            expected_sources=expected.m3_sources[2],
            expected_samples=expected.m3_test_rows,
        )
    )
    return pd.DataFrame(rows)


def _distribution_stats(frame: pd.DataFrame, *, near_degenerate_sd: float) -> dict[str, Any]:
    values = frame[TARGET].to_numpy(dtype=np.float64)
    n_samples = int(values.size)
    n_sources = int(frame["source_id"].nunique())
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n_samples > 1 else math.nan
    variance = float(np.var(values, ddof=1)) if n_samples > 1 else math.nan
    grouped = frame.groupby("source_id")[TARGET]
    source_means = grouped.mean()
    source_variances = grouped.var(ddof=1)
    between = float(source_means.var(ddof=1)) if n_sources > 1 else math.nan
    finite_within = source_variances.dropna()
    within = float(finite_within.mean()) if not finite_within.empty else math.nan
    between_share = math.nan
    if math.isfinite(between) and math.isfinite(within) and between + within > 0.0:
        between_share = between / (between + within)
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n_samples": n_samples,
        "n_sources": n_sources,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": mean,
        "sd": sd,
        "variance": variance,
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "between_source_variance": between,
        "within_source_variance": within,
        "between_variance_share": between_share,
        "r2_denominator_ss": float(np.sum((values - mean) ** 2)),
        "near_degenerate": bool(math.isfinite(sd) and sd < near_degenerate_sd),
    }


def _target_slice_specs(
    frames: gb.BenchmarkFrames,
) -> list[tuple[str, str, str, pd.DataFrame, str, int, float]]:
    specs: list[tuple[str, str, str, pd.DataFrame, str, int, float]] = []
    for split in ("all", *SPLITS):
        m3 = (
            frames.m3_frame
            if split == "all"
            else frames.m3_frame[frames.m3_frame["split"] == split]
        )
        m5a = (
            frames.m5a_frame
            if split == "all"
            else frames.m5a_frame[frames.m5a_frame["split"] == split]
        )
        specs.append(("m3", split, "m3_overall", m3, "all", -1, math.nan))
        specs.append(
            (
                "m3",
                split,
                "clean_reference",
                m3[m3["family"] == gb.M3_CLEAN_FAMILY],
                gb.M3_CLEAN_FAMILY,
                0,
                math.nan,
            )
        )
        original = m3[m3["family"] != gb.M3_CLEAN_FAMILY]
        specs.append(("m3", split, "original_blur_noise", original, "all", -1, math.nan))
        for family in ("gaussian_blur", "gaussian_noise"):
            family_frame = original[original["family"] == family]
            for severity in (1, 2, 3, 4):
                specs.append(
                    (
                        "m3",
                        split,
                        f"original_blur_noise_{family}",
                        family_frame[family_frame["severity_level"] == severity],
                        family,
                        severity,
                        math.nan,
                    )
                )
        specs.append(("m5a", split, "m5a_overall", m5a, "all", -1, math.nan))
        specs.append(
            (
                "m5a",
                split,
                "m5a_clean",
                m5a[m5a["family"] == gb.CLEAN_FAMILY],
                gb.CLEAN_FAMILY,
                0,
                math.nan,
            )
        )
        controls = m5a[m5a["family"] == gb.CONTROL_FAMILY]
        specs.append(("m5a", split, "m5a_control", controls, gb.CONTROL_FAMILY, 0, math.nan))
        for control_id in ("identity", "zero_field", "minimal"):
            specs.append(
                (
                    "m5a",
                    split,
                    f"m5a_control_{control_id}",
                    controls[controls["control_id"] == control_id],
                    gb.CONTROL_FAMILY,
                    0,
                    math.nan,
                )
            )
        elastic = m5a[m5a["family"] == gb.ELASTIC_FAMILY]
        specs.append(
            (
                "m5a",
                split,
                "geometric_primary",
                elastic[elastic["amplitude_px"].isin(gb.PRIMARY_AMPLITUDES_PX)],
                gb.ELASTIC_FAMILY,
                -1,
                math.nan,
            )
        )
        for severity, amplitude in ((2, 1.0), (3, 2.0), (4, 3.0)):
            specs.append(
                (
                    "m5a",
                    split,
                    f"geometric_primary_severity_{severity}",
                    elastic[elastic["amplitude_px"] == amplitude],
                    gb.ELASTIC_FAMILY,
                    severity,
                    amplitude,
                )
            )
        specs.append(
            (
                "m5a",
                split,
                "diagnostic_0_5",
                elastic[elastic["amplitude_px"].isin(gb.DIAGNOSTIC_AMPLITUDES_PX)],
                gb.ELASTIC_FAMILY,
                1,
                0.5,
            )
        )
    return specs


def build_target_distributions(inputs: AuditInputs) -> pd.DataFrame:
    """Summarise target distributions for every declared slice and split."""
    threshold = inputs.config.audit.near_degenerate_sd
    rows: list[dict[str, Any]] = []
    for dataset, split, slice_name, frame, family, severity, amplitude in _target_slice_specs(
        inputs.frames
    ):
        if frame.empty:
            continue
        stats = _distribution_stats(frame, near_degenerate_sd=threshold)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "slice": slice_name,
                "family": family,
                "severity_level": severity,
                "amplitude_px": amplitude,
                **stats,
            }
        )
    result = pd.DataFrame(rows)
    dataset_order = {"m3": 0, "m5a": 1}
    split_order = {"all": 0, **{split: index + 1 for index, split in enumerate(SPLITS)}}
    slice_order = {
        name: index
        for index, name in enumerate(
            (
                "m3_overall",
                "clean_reference",
                "original_blur_noise",
                "original_blur_noise_gaussian_blur",
                "original_blur_noise_gaussian_noise",
                "m5a_overall",
                "m5a_clean",
                "m5a_control",
                "m5a_control_identity",
                "m5a_control_zero_field",
                "m5a_control_minimal",
                "geometric_primary",
                "geometric_primary_severity_2",
                "geometric_primary_severity_3",
                "geometric_primary_severity_4",
                "diagnostic_0_5",
            )
        )
    }
    result["_dataset_order"] = result["dataset"].map(dataset_order)
    result["_split_order"] = result["split"].map(split_order)
    result["_slice_order"] = result["slice"].map(slice_order)
    return (
        result.sort_values(
            ["_dataset_order", "_split_order", "_slice_order", "severity_level", "amplitude_px"]
        )
        .drop(columns=["_dataset_order", "_split_order", "_slice_order"])
        .reset_index(drop=True)
    )


def _scope_predictions(inputs: AuditInputs, scope: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = scope_frames(inputs.frames)[scope].sort_values("sample_id").reset_index(drop=True)
    stored = inputs.predictions[
        (inputs.predictions["model"] == inputs.champion_name)
        & (inputs.predictions["scope"] == scope)
    ].sort_values("sample_id")
    if stored["sample_id"].tolist() != frame["sample_id"].tolist():
        raise DatasetAuditError(f"prediction rows for scope {scope!r} do not align with the frame")
    return frame, stored["prediction"].to_numpy(dtype=np.float64)


def _metric_row(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, Any]:
    metrics = regression_metrics(frame[TARGET].to_numpy(dtype=np.float64), predicted)
    residual = frame[TARGET].to_numpy(dtype=np.float64) - predicted
    return {
        "n_samples": int(metrics["n_rows"]),
        "n_sources": int(frame["source_id"].nunique()),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "r2_defined": bool(metrics["r2_defined"]),
        "mean_residual": float(np.mean(residual)),
        "sd_residual": float(np.std(residual, ddof=1)) if residual.size > 1 else math.nan,
        "sse": float(np.sum(residual**2)),
    }


def build_test_slice_metrics(inputs: AuditInputs) -> pd.DataFrame:
    """Copy and enrich the frozen champion's test-slice metrics."""
    rows: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        frame, predicted = _scope_predictions(inputs, scope)
        metrics = _metric_row(frame, predicted)
        stats = _distribution_stats(
            frame, near_degenerate_sd=inputs.config.audit.near_degenerate_sd
        )
        rows.append(
            {
                "model": inputs.champion_name,
                "scope": scope,
                "n_samples": metrics["n_samples"],
                "n_sources": metrics["n_sources"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "r2_defined": metrics["r2_defined"],
                "mean_residual": metrics["mean_residual"],
                "sd_residual": metrics["sd_residual"],
                "target_sd": stats["sd"],
                "target_variance": stats["variance"],
                "target_near_degenerate": stats["near_degenerate"],
                "rmse_over_target_sd": (
                    metrics["rmse"] / stats["sd"]
                    if math.isfinite(stats["sd"]) and stats["sd"] > 0.0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_source_metrics(inputs: AuditInputs) -> pd.DataFrame:
    """Per-source champion MAE, RMSE, residual spread and squared-error share."""
    rows: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        frame, predicted = _scope_predictions(inputs, scope)
        residual = frame[TARGET].to_numpy(dtype=np.float64) - predicted
        working = frame[["source_id", "font_id", TARGET]].copy()
        working["prediction"] = predicted
        working["residual"] = residual
        working["squared_error"] = residual**2
        total_sse = float(working["squared_error"].sum())
        for source_id, group in working.groupby("source_id", sort=True):
            metrics = regression_metrics(
                group[TARGET].to_numpy(dtype=np.float64),
                group["prediction"].to_numpy(dtype=np.float64),
            )
            sse = float(group["squared_error"].sum())
            rows.append(
                {
                    "scope": scope,
                    "source_id": str(source_id),
                    "font_id": str(group["font_id"].iloc[0]),
                    "n_samples": len(group),
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "mean_residual": float(group["residual"].mean()),
                    "sd_residual": (
                        float(group["residual"].std(ddof=1)) if len(group) > 1 else math.nan
                    ),
                    "sse": sse,
                    "sse_share": sse / total_sse if total_sse > 0.0 else math.nan,
                }
            )
    result = pd.DataFrame(rows)
    result["sse_rank"] = (
        result.groupby("scope")["sse"].rank(method="first", ascending=False).astype(int)
    )
    scope_order = {scope: index for index, scope in enumerate(SCOPE_ORDER)}
    result["_scope_order"] = result["scope"].map(scope_order)
    return (
        result.sort_values(["_scope_order", "sse_rank", "source_id"])
        .drop(columns=["_scope_order"])
        .reset_index(drop=True)
    )


def build_source_severity_metrics(inputs: AuditInputs) -> pd.DataFrame:
    """Per-source, per-severity residual summaries for geometric and original slices."""
    rows: list[dict[str, Any]] = []
    specifications = (
        ("geometric_primary", "geometric_primary"),
        ("original_blur_noise", "original_blur_noise"),
    )
    for scope, _ in specifications:
        frame, predicted = _scope_predictions(inputs, scope)
        columns = ["source_id", "font_id", "family", "severity_level"]
        if "amplitude_px" in frame.columns:
            columns.append("amplitude_px")
        working = frame[columns].copy()
        working["residual"] = frame[TARGET].to_numpy(dtype=np.float64) - predicted
        working["squared_error"] = working["residual"] ** 2
        for (source_id, family, severity), group in working.groupby(
            ["source_id", "family", "severity_level"], sort=True
        ):
            amplitude = (
                float(group["amplitude_px"].iloc[0])
                if "amplitude_px" in group.columns and pd.notna(group["amplitude_px"].iloc[0])
                else math.nan
            )
            rows.append(
                {
                    "scope": scope,
                    "source_id": str(source_id),
                    "font_id": str(group["font_id"].iloc[0]),
                    "family": str(family),
                    "severity_level": int(severity),
                    "amplitude_px": amplitude,
                    "n_samples": len(group),
                    "mae": float(group["residual"].abs().mean()),
                    "rmse": float(np.sqrt(group["squared_error"].mean())),
                    "mean_residual": float(group["residual"].mean()),
                    "sd_residual": (
                        float(group["residual"].std(ddof=1)) if len(group) > 1 else math.nan
                    ),
                }
            )
    result = pd.DataFrame(rows)
    scope_order = {scope: index for index, scope in enumerate(SCOPE_ORDER)}
    result["_scope_order"] = result["scope"].map(scope_order)
    return (
        result.sort_values(["_scope_order", "source_id", "family", "severity_level"])
        .drop(columns=["_scope_order"])
        .reset_index(drop=True)
    )


def build_source_family_metrics(inputs: AuditInputs) -> pd.DataFrame:
    """Per-source, per-family residual summaries for the original M3 task."""
    frame, predicted = _scope_predictions(inputs, "original_blur_noise")
    working = frame[["source_id", "font_id", "family"]].copy()
    working["residual"] = frame[TARGET].to_numpy(dtype=np.float64) - predicted
    working["squared_error"] = working["residual"] ** 2
    rows: list[dict[str, Any]] = []
    for (source_id, family), group in working.groupby(["source_id", "family"], sort=True):
        rows.append(
            {
                "scope": "original_blur_noise",
                "source_id": str(source_id),
                "font_id": str(group["font_id"].iloc[0]),
                "family": str(family),
                "n_samples": len(group),
                "mae": float(group["residual"].abs().mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "mean_residual": float(group["residual"].mean()),
                "sd_residual": float(group["residual"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_id", "family"]).reset_index(drop=True)


def build_font_metrics(inputs: AuditInputs) -> pd.DataFrame:
    """Per-font champion metrics for every evaluation scope."""
    rows: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        frame, predicted = _scope_predictions(inputs, scope)
        working = frame[["source_id", "font_id"]].copy()
        working["target"] = frame[TARGET].to_numpy(dtype=np.float64)
        working["prediction"] = predicted
        working["residual"] = working["target"] - working["prediction"]
        working["squared_error"] = working["residual"] ** 2
        for font_id, group in working.groupby("font_id", sort=True):
            rows.append(
                {
                    "scope": scope,
                    "font_id": str(font_id),
                    "n_sources": int(group["source_id"].nunique()),
                    "n_samples": len(group),
                    "mae": float(group["residual"].abs().mean()),
                    "rmse": float(np.sqrt(group["squared_error"].mean())),
                    "mean_residual": float(group["residual"].mean()),
                    "sd_residual": (
                        float(group["residual"].std(ddof=1)) if len(group) > 1 else math.nan
                    ),
                }
            )
    result = pd.DataFrame(rows)
    scope_order = {scope: index for index, scope in enumerate(SCOPE_ORDER)}
    result["_scope_order"] = result["scope"].map(scope_order)
    return (
        result.sort_values(["_scope_order", "font_id"])
        .drop(columns=["_scope_order"])
        .reset_index(drop=True)
    )


def build_error_concentration(inputs: AuditInputs) -> pd.DataFrame:
    """Describe how squared error is distributed over the available source groups."""
    rows: list[dict[str, Any]] = []
    top_k = inputs.config.audit.top_k_sources
    for scope in SCOPE_ORDER:
        frame, predicted = _scope_predictions(inputs, scope)
        residual = frame[TARGET].to_numpy(dtype=np.float64) - predicted
        working = frame[["source_id"]].copy()
        working["squared_error"] = residual**2
        source_sse = (
            working.groupby("source_id")["squared_error"].sum().sort_values(ascending=False)
        )
        total = float(source_sse.sum())
        shares = source_sse / total if total > 0.0 else source_sse * math.nan
        hhi = float(np.sum(shares**2)) if total > 0.0 else math.nan
        cumulative = shares.cumsum()
        rows.append(
            {
                "scope": scope,
                "n_sources": int(source_sse.size),
                "total_sse": total,
                "top1_source": str(source_sse.index[0]),
                "top1_share": float(shares.iloc[0]) if total > 0.0 else math.nan,
                f"top{top_k}_share": float(shares.head(top_k).sum()) if total > 0.0 else math.nan,
                "hhi": hhi,
                "effective_sources": 1.0 / hhi if math.isfinite(hhi) and hhi > 0.0 else math.nan,
                "sources_for_50pct": int((cumulative < 0.5).sum() + 1) if total > 0.0 else 0,
                "sources_for_80pct": int((cumulative < 0.8).sum() + 1) if total > 0.0 else 0,
                "source_sse_cv": (
                    float(source_sse.std(ddof=1) / source_sse.mean())
                    if source_sse.size > 1 and source_sse.mean() > 0.0
                    else math.nan
                ),
                "interpretation": "descriptive concentration; not a significance test",
            }
        )
    result = pd.DataFrame(rows)
    scope_order = {scope: index for index, scope in enumerate(SCOPE_ORDER)}
    result["_scope_order"] = result["scope"].map(scope_order)
    return result.sort_values("_scope_order").drop(columns=["_scope_order"]).reset_index(drop=True)


def build_leave_one_source_out(inputs: AuditInputs) -> pd.DataFrame:
    """Descriptive leave-one-source-out influence on aggregate RMSE and MAE."""
    rows: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        frame, predicted = _scope_predictions(inputs, scope)
        target = frame[TARGET].to_numpy(dtype=np.float64)
        full = regression_metrics(target, predicted)
        working = frame[["source_id"]].copy()
        working["squared_error"] = (target - predicted) ** 2
        total_sse = float(working["squared_error"].sum())
        for source_id in sorted(frame["source_id"].unique()):
            keep = (frame["source_id"] != source_id).to_numpy()
            without = regression_metrics(target[keep], predicted[keep])
            omitted_sse = float(working.loc[~keep, "squared_error"].sum())
            rows.append(
                {
                    "scope": scope,
                    "omitted_source_id": str(source_id),
                    "n_sources_remaining": int(frame.loc[keep, "source_id"].nunique()),
                    "n_samples_remaining": int(keep.sum()),
                    "rmse_full": float(full["rmse"]),
                    "rmse_without": float(without["rmse"]),
                    "delta_rmse": float(without["rmse"]) - float(full["rmse"]),
                    "mae_full": float(full["mae"]),
                    "mae_without": float(without["mae"]),
                    "delta_mae": float(without["mae"]) - float(full["mae"]),
                    "omitted_sse_share": omitted_sse / total_sse if total_sse > 0.0 else math.nan,
                    "interpretation": "descriptive influence; no refitting or new generalisation estimate",
                }
            )
    result = pd.DataFrame(rows)
    scope_order = {scope: index for index, scope in enumerate(SCOPE_ORDER)}
    result["_scope_order"] = result["scope"].map(scope_order)
    return (
        result.sort_values(["_scope_order", "omitted_source_id"])
        .drop(columns=["_scope_order"])
        .reset_index(drop=True)
    )


def build_uncertainty(inputs: AuditInputs) -> pd.DataFrame:
    """Source-grouped percentile bootstrap for every declared scope."""
    config = inputs.config
    rows: list[pd.DataFrame] = []
    for scope in config.audit.bootstrap_scopes:
        frame, predicted = _scope_predictions(inputs, scope)
        seed = derive_noise_seed(config.run.seed, "dataset_audit_bootstrap", scope)
        table = bootstrap_metrics(
            frame,
            predicted,
            n_bootstrap=config.run.n_bootstrap,
            seed=seed,
            scope=scope,
        )
        stats = _distribution_stats(frame, near_degenerate_sd=config.audit.near_degenerate_sd)
        table["dataset"] = "m3" if scope.startswith(("m3_", "clean_", "original_")) else "m5a"
        table["target_sd"] = stats["sd"]
        table["target_variance"] = stats["variance"]
        table["target_near_degenerate"] = stats["near_degenerate"]
        table["method"] = config.audit.bootstrap_method
        table["confidence_level"] = config.audit.confidence_level
        table["independent_groups"] = table["n_groups"]
        table["caveat"] = (
            f"{int(table['n_groups'].iloc[0])} source groups; descriptive percentile interval; "
            "not a significance test or a new generalisation estimate"
        )
        rows.append(table)
    result = pd.concat(rows, ignore_index=True)
    metric_order = {"mae": 0, "rmse": 1, "r2": 2}
    scope_order = {scope: index for index, scope in enumerate(SCOPE_ORDER)}
    result["_scope_order"] = result["scope"].map(scope_order)
    result["_metric_order"] = result["metric"].map(metric_order)
    return (
        result.sort_values(["_scope_order", "_metric_order"])
        .drop(columns=["_scope_order", "_metric_order"])
        .reset_index(drop=True)
    )


def _training_features(inputs: AuditInputs) -> pd.DataFrame:
    frame = inputs.frames.expanded_train
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise DatasetAuditError(f"expanded training frame is missing features: {missing}")
    return frame


def build_feature_summary(inputs: AuditInputs) -> pd.DataFrame:
    """Training-only scale, distribution and duplicate diagnostics."""
    frame = _training_features(inputs)
    rows: list[dict[str, Any]] = []
    previous: list[tuple[str, np.ndarray]] = []
    for feature in FEATURE_NAMES:
        values = frame[feature].to_numpy(dtype=np.float64)
        series = pd.Series(values)
        duplicate_of = ""
        for prior_name, prior_values in previous:
            if np.array_equal(values, prior_values):
                duplicate_of = prior_name
                break
        previous.append((feature, values))
        rows.append(
            {
                "feature": feature,
                "n_samples": int(values.size),
                "n_unique": int(series.nunique(dropna=False)),
                "constant": bool(series.nunique(dropna=False) <= 1),
                "duplicate_of": duplicate_of,
                "finite": bool(np.isfinite(values).all()),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "q50": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
                "zero_fraction": float(np.mean(values == 0.0)),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurt()),
            }
        )
    return pd.DataFrame(rows)


def build_feature_correlations(inputs: AuditInputs) -> pd.DataFrame:
    """Training-only Pearson and Spearman correlations between frozen features."""
    frame = _training_features(inputs)
    values = frame[list(FEATURE_NAMES)]
    pearson = values.corr(method="pearson")
    spearman = values.corr(method="spearman")
    threshold = inputs.config.audit.redundancy_abs_correlation
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(FEATURE_NAMES):
        for right in FEATURE_NAMES[left_index + 1 :]:
            correlation = float(pearson.loc[left, right])
            rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "pearson": correlation,
                    "abs_pearson": abs(correlation),
                    "spearman": float(spearman.loc[left, right]),
                    "redundant": bool(abs(correlation) >= threshold),
                    "pair_type": "global-global",
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["abs_pearson", "feature_a", "feature_b"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def build_feature_importance(inputs: AuditInputs) -> pd.DataFrame:
    """Descriptive impurity importance from the frozen champion, not a causal claim."""
    importances = np.asarray(inputs.champion_model.feature_importances_, dtype=np.float64)
    if importances.size != len(FEATURE_NAMES):
        raise DatasetAuditError("champion feature importance width does not match the schema")
    frame = pd.DataFrame({"feature": FEATURE_NAMES, "importance": importances})
    frame = frame.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(
        drop=True
    )
    frame["importance_rank"] = np.arange(1, len(frame) + 1)
    frame["cumulative_importance"] = frame["importance"].cumsum()
    frame["descriptive_only"] = True
    frame["interpretation"] = (
        "impurity importance from the frozen champion; descriptive, not causal"
    )
    return frame


def build_feature_slice_means(inputs: AuditInputs) -> pd.DataFrame:
    """Training-only feature means across existing families and severities."""
    frame = _training_features(inputs)
    rows: list[dict[str, Any]] = []
    for dimension in inputs.config.audit.feature_slice_dimensions:
        if dimension not in frame.columns:
            continue
        for value, group in frame.groupby(dimension, sort=True):
            for feature in FEATURE_NAMES:
                values = group[feature].to_numpy(dtype=np.float64)
                rows.append(
                    {
                        "dimension": dimension,
                        "value": str(value),
                        "feature": feature,
                        "n_samples": int(values.size),
                        "n_sources": int(group["source_id"].nunique()),
                        "mean": float(np.mean(values)),
                        "sd": float(np.std(values, ddof=1)) if values.size > 1 else math.nan,
                        "median": float(np.median(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "range": float(np.max(values) - np.min(values)),
                    }
                )
    result = pd.DataFrame(rows)
    dimension_order = {
        dimension: index
        for index, dimension in enumerate(inputs.config.audit.feature_slice_dimensions)
    }
    feature_order = {feature: index for index, feature in enumerate(FEATURE_NAMES)}
    result["_dimension_order"] = result["dimension"].map(dimension_order)
    result["_feature_order"] = result["feature"].map(feature_order)
    return (
        result.sort_values(["_dimension_order", "value", "_feature_order"])
        .drop(columns=["_dimension_order", "_feature_order"])
        .reset_index(drop=True)
    )


def build_feature_target_associations(inputs: AuditInputs) -> pd.DataFrame:
    """Training-only descriptive association between each feature and the target."""
    frame = _training_features(inputs)
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_NAMES:
        pearson = float(frame[[feature, TARGET]].corr(method="pearson").iloc[0, 1])
        spearman = float(frame[[feature, TARGET]].corr(method="spearman").iloc[0, 1])
        rows.append(
            {
                "feature": feature,
                "n_samples": len(frame),
                "n_sources": int(frame["source_id"].nunique()),
                "pearson": pearson,
                "spearman": spearman,
                "abs_spearman": abs(spearman),
                "interpretation": (
                    "training-only descriptive association; not feature selection or causal evidence"
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["abs_spearman", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def feature_diagnostics_summary(inputs: AuditInputs) -> dict[str, Any]:
    """Return compact numeric diagnostics used by the report and metadata."""
    frame = _training_features(inputs)
    values = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    correlations = frame[list(FEATURE_NAMES)].corr(method="pearson").to_numpy()
    abs_correlations = np.abs(correlations.copy())
    np.fill_diagonal(abs_correlations, 0.0)
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0, ddof=1)
    safe = standard_deviations > 0.0
    condition_number = math.nan
    if safe.all():
        standardised = (values - means) / standard_deviations
        condition_number = float(np.linalg.cond(standardised))
    correlation_matrix = frame[list(FEATURE_NAMES)].corr(method="pearson").to_numpy()
    vif = np.full(len(FEATURE_NAMES), math.nan)
    try:
        inverse = np.linalg.inv(correlation_matrix)
        vif = np.diag(inverse)
    except np.linalg.LinAlgError:  # pragma: no cover - only for degenerate inputs
        inverse = np.linalg.pinv(correlation_matrix)
        vif = np.diag(inverse)
    threshold = inputs.config.audit.redundancy_abs_correlation
    return {
        "n_rows": len(frame),
        "n_sources": int(frame["source_id"].nunique()),
        "n_features": len(FEATURE_NAMES),
        "constant_features": [
            feature for feature in FEATURE_NAMES if frame[feature].nunique(dropna=False) <= 1
        ],
        "max_abs_correlation": float(abs_correlations.max()),
        "mean_abs_correlation": float(
            abs_correlations[np.triu_indices_from(abs_correlations, k=1)].mean()
        ),
        "n_redundant_pairs": int((abs_correlations >= threshold).sum() // 2),
        "redundancy_threshold": threshold,
        "condition_number": condition_number,
        "max_vif": float(np.nanmax(vif)),
        "top_importance_feature": str(
            FEATURE_NAMES[int(np.argmax(inputs.champion_model.feature_importances_))]
        ),
        "top_importance": float(np.max(inputs.champion_model.feature_importances_)),
    }


def _metric_row_for_scope(metrics: pd.DataFrame, scope: str) -> pd.Series:
    row = metrics[metrics["scope"] == scope]
    if len(row) != 1:
        raise DatasetAuditError(f"expected exactly one metric row for scope {scope!r}")
    return row.iloc[0]


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "undefined"
        return f"{number:.{digits}f}"
    return str(value)


def build_claim_evidence(
    inputs: AuditInputs,
    test_slice_metrics: pd.DataFrame,
    error_concentration: pd.DataFrame,
    feature_correlations: pd.DataFrame,
    feature_diagnostics: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the predeclared claim-to-evidence classification table."""
    m3 = _metric_row_for_scope(test_slice_metrics, "m3_test_overall")
    clean = _metric_row_for_scope(test_slice_metrics, "clean_reference")
    original = _metric_row_for_scope(test_slice_metrics, "original_blur_noise")
    geometric = _metric_row_for_scope(test_slice_metrics, "geometric_primary")
    severity_2 = _metric_row_for_scope(test_slice_metrics, "geometric_primary_severity_2")
    severity_3 = _metric_row_for_scope(test_slice_metrics, "geometric_primary_severity_3")
    severity_4 = _metric_row_for_scope(test_slice_metrics, "geometric_primary_severity_4")
    diagnostic = _metric_row_for_scope(test_slice_metrics, "diagnostic_0_5")
    concentration = error_concentration[error_concentration["scope"] == "geometric_primary"].iloc[0]
    top_correlation = feature_correlations.iloc[0]

    selected = inputs.selection_table[inputs.selection_table["selected"].astype(bool)].iloc[0]
    heldout = inputs.heldout_font_metrics
    heldout_candidates = heldout[
        (heldout["split"] == "heldout_font")
        & (heldout["role"].astype(str).str.contains("selected|family_best", regex=True))
    ]
    if heldout_candidates.empty:
        heldout_candidates = heldout[heldout["split"] == "heldout_font"]
    heldout_best = heldout_candidates.sort_values("r2", ascending=False).iloc[0]

    claims: list[dict[str, Any]] = [
        {
            "claim_id": "C01",
            "claim": "The frozen champion predicts SSIM on the current synthetic blur/noise benchmark.",
            "classification": "supported",
            "evidence": (
                f"M3 test: RMSE {_fmt(m3['rmse'])}, R² {_fmt(m3['r2'])}, "
                f"{int(m3['n_samples'])} rows from {int(m3['n_sources'])} sources."
            ),
            "limitations": "Same renderer, resolution and font set as training; 12 test source groups.",
            "evidence_scope": "m3_test_overall",
            "evidence_metric": "rmse/r2",
            "evidence_value": f"{_fmt(m3['rmse'])}/{_fmt(m3['r2'])}",
            "benchmark_coverage": "within_current_benchmark",
        },
        {
            "claim_id": "C02",
            "claim": "The frozen champion predicts SSIM for 1-3 px elastic geometric distortions on held-out sources from the same renderer.",
            "classification": "suggestive",
            "evidence": (
                f"Geometric primary: RMSE {_fmt(geometric['rmse'])}, R² {_fmt(geometric['r2'])}, "
                f"{int(geometric['n_samples'])} rows from {int(geometric['n_sources'])} sources; "
                f"severity 4 R² {_fmt(severity_4['r2'])}."
            ),
            "limitations": "Low aggregate R², negative R² at severity 2 and 4, and only 12 independent source groups.",
            "evidence_scope": "geometric_primary",
            "evidence_metric": "rmse/r2",
            "evidence_value": f"{_fmt(geometric['rmse'])}/{_fmt(geometric['r2'])}",
            "benchmark_coverage": "within_current_benchmark",
        },
        {
            "claim_id": "C03",
            "claim": "The frozen champion generalises to unseen source images from the same rendering setup.",
            "classification": "suggestive",
            "evidence": (
                f"Source-grouped test: clean RMSE {_fmt(clean['rmse'])}, original RMSE {_fmt(original['rmse'])}, "
                f"geometric RMSE {_fmt(geometric['rmse'])}; 12 test source groups."
            ),
            "limitations": "The clean and original slices support the claim; the geometric slice is much weaker. Twelve groups cannot support a precise generalisation interval.",
            "evidence_scope": "clean_reference/original_blur_noise/geometric_primary",
            "evidence_metric": "rmse",
            "evidence_value": f"{_fmt(clean['rmse'])}/{_fmt(original['rmse'])}/{_fmt(geometric['rmse'])}",
            "benchmark_coverage": "within_current_benchmark",
        },
        {
            "claim_id": "C04",
            "claim": "The frozen champion generalises to unseen fonts.",
            "classification": "unsupported",
            "evidence": (
                "All three fonts appear in train, validation and test, so the champion is never evaluated on a font it did not see. "
                f"The separate M3 held-out-font model reaches R² {_fmt(heldout_best['r2'])} on "
                f"{int(heldout_best['n_sources'])} held-out-font sources, but it is a different model."
            ),
            "limitations": "No champion held-out-font test exists; the M3 stress test cannot be transferred to the champion.",
            "evidence_scope": "heldout_font",
            "evidence_metric": "coverage",
            "evidence_value": "font shared across all splits",
            "benchmark_coverage": "not_covered",
        },
        {
            "claim_id": "C05",
            "claim": "The frozen champion generalises to unseen renderers, resolutions or image formats.",
            "classification": "unsupported",
            "evidence": (
                "The frozen data use one renderer (Pillow ImageDraw + FreeType), one resolution (192x64) "
                "and one pixel format (8-bit grayscale PNG). No alternative condition exists in any split."
            ),
            "limitations": "No unseen-renderer, unseen-resolution or unseen-format data; no basis for a condition-generalisation claim.",
            "evidence_scope": "dataset_composition",
            "evidence_metric": "condition_coverage",
            "evidence_value": "1 renderer / 1 resolution / 1 format",
            "benchmark_coverage": "not_covered",
        },
        {
            "claim_id": "C06",
            "claim": "The frozen champion measures real-world CAPTCHA image quality.",
            "classification": "unsupported",
            "evidence": "Targets are SSIM against a known synthetic clean reference; no real CAPTCHA images, capture pipeline or deployment sample is present.",
            "limitations": "Synthetic-to-real distribution shift is unmeasured.",
            "evidence_scope": "target_definition",
            "evidence_metric": "construct_coverage",
            "evidence_value": "synthetic only",
            "benchmark_coverage": "not_covered",
        },
        {
            "claim_id": "C07",
            "claim": "The frozen champion measures human readability, solve rate or CAPTCHA security.",
            "classification": "unsupported",
            "evidence": "No human study, solver outcome or security evaluation is part of the dataset or audit; SSIM is full-reference image similarity.",
            "limitations": "Construct validity for human or security outcomes is untested.",
            "evidence_scope": "target_definition",
            "evidence_metric": "construct_coverage",
            "evidence_value": "no human/solver outcomes",
            "benchmark_coverage": "not_covered",
        },
        {
            "claim_id": "C08",
            "claim": "The champion improves geometric prediction over the frozen M3 model without materially regressing clean or original-task performance on this benchmark.",
            "classification": "supported",
            "evidence": (
                f"M5C selection versus frozen M3: geometric RMSE delta "
                f"{_fmt(float(selected['delta_geometric_rmse']))}, clean RMSE delta "
                f"{_fmt(float(selected['delta_clean_rmse']))}, original RMSE delta "
                f"{_fmt(float(selected['delta_original_rmse']))}. The frozen reference row is "
                "the zero-delta baseline."
            ),
            "limitations": "This is a within-benchmark comparison on the same 12 test source groups, not an external generalisation result.",
            "evidence_scope": "m5c_selection_table",
            "evidence_metric": "delta_rmse",
            "evidence_value": f"{_fmt(float(selected['delta_geometric_rmse']))}",
            "benchmark_coverage": "within_current_benchmark",
        },
        {
            "claim_id": "C09",
            "claim": "The global eight-feature representation is adequate for every primary geometric severity.",
            "classification": "unsupported",
            "evidence": (
                f"Severity 2 R² {_fmt(severity_2['r2'])}, severity 3 R² {_fmt(severity_3['r2'])}, "
                f"severity 4 R² {_fmt(severity_4['r2'])}; the strongest feature pair has |r| "
                f"{_fmt(feature_diagnostics['max_abs_correlation'])} "
                f"({top_correlation['feature_a']} vs {top_correlation['feature_b']})."
            ),
            "limitations": "Negative R² at the easiest and hardest rungs indicates the representation and model do not capture the displacement signal uniformly.",
            "evidence_scope": "geometric_primary_severity_2/3/4",
            "evidence_metric": "r2",
            "evidence_value": f"{_fmt(severity_2['r2'])}/{_fmt(severity_3['r2'])}/{_fmt(severity_4['r2'])}",
            "benchmark_coverage": "within_current_benchmark",
        },
        {
            "claim_id": "C10",
            "claim": "Geometric error is unevenly concentrated across the 12 held-out source groups.",
            "classification": "supported",
            "evidence": (
                f"Top source contributes {_fmt(float(concentration['top1_share']))} of squared error; "
                f"top 3 contribute {_fmt(float(concentration['top3_share']))}; "
                f"effective source count {_fmt(float(concentration['effective_sources']), 2)} of 12."
            ),
            "limitations": "Descriptive only; with 12 groups the ranking is unstable and leave-one-source-out deltas are small.",
            "evidence_scope": "geometric_primary",
            "evidence_metric": "sse_share",
            "evidence_value": f"{_fmt(float(concentration['top3_share']))}",
            "benchmark_coverage": "within_current_benchmark",
        },
        {
            "claim_id": "C11",
            "claim": "The 0.5 px diagnostic rung supports a stable R² estimate.",
            "classification": "unsupported",
            "evidence": (
                f"Target SD {_fmt(diagnostic['target_sd'], 5)} is below the predeclared 0.005 margin; "
                f"R² {_fmt(diagnostic['r2'])} and RMSE {_fmt(diagnostic['rmse'])}. "
                "A near-constant target makes R² dominated by tiny denominator variance."
            ),
            "limitations": "Keep the rung as a diagnostic only; do not use it as primary evidence.",
            "evidence_scope": "diagnostic_0_5",
            "evidence_metric": "target_sd/r2",
            "evidence_value": f"{_fmt(diagnostic['target_sd'], 5)}/{_fmt(diagnostic['r2'])}",
            "benchmark_coverage": "within_current_benchmark",
        },
        {
            "claim_id": "C12",
            "claim": "The current benchmark supports broad generalisation claims beyond the synthetic distribution.",
            "classification": "unsupported",
            "evidence": "One renderer, one resolution, three shared fonts, 12 test source groups and synthetic SSIM targets cannot identify condition or population generalisation.",
            "limitations": "Requires additional independent data under explicitly held-out conditions.",
            "evidence_scope": "benchmark_coverage",
            "evidence_metric": "coverage",
            "evidence_value": "condition-specific",
            "benchmark_coverage": "not_covered",
        },
    ]
    frame = pd.DataFrame(claims)
    if not set(frame["classification"]).issubset(set(CLASSIFICATIONS)):
        raise DatasetAuditError("claim table contains an undeclared classification")
    return frame


@dataclass(frozen=True, slots=True)
class DirectionSpec:
    direction: str
    title: str
    hypothesis: str
    scientific_value: str
    data_needs: str
    implementation_needs: str
    leakage_generalization_risk: str
    success_measure: str
    current_data_can_answer: str
    rationale: str


def build_direction_comparison(config: AuditConfig) -> pd.DataFrame:
    """Compare the four predeclared next directions and mark exactly one."""
    directions = [
        DirectionSpec(
            direction="A",
            title="Expand independent source data within the current rendering setup",
            hypothesis="More independent sources will narrow source-grouped intervals and test whether the observed per-source error ranking persists.",
            scientific_value="Moderate: improves precision and source-level inference, but does not test any new rendering condition.",
            data_needs="New independent synthetic sources and clean references under the existing renderer, font set and corruption ladders; no new conditions.",
            implementation_needs="Versioned dataset configuration, source-disjoint grouped splits, frozen-champion zero-shot evaluation and grouped uncertainty; no model changes.",
            leakage_generalization_risk="Low if new sources are disjoint and never used for selection; generalisation remains limited to the current renderer and font set.",
            success_measure="Narrower source-grouped intervals and a stable per-source error ranking; aggregate claims should change only through precision, not through a new condition result.",
            current_data_can_answer="Partially: the existing 12 test groups are too few for precise intervals, but the data already show the direction of the effect.",
            rationale="Useful but not the highest-value next step because it cannot resolve the unsupported unseen-font or unseen-renderer claims.",
        ),
        DirectionSpec(
            direction="B",
            title="Add explicitly held-out fonts or rendering conditions in a future versioned dataset",
            hypothesis="The frozen champion's SSIM prediction transfers to fonts or rendering conditions that never appear in training; if it does not, current claims are condition-specific.",
            scientific_value="High: directly tests the largest unsupported generalisation claims and separates source generalisation from condition generalisation.",
            data_needs="At least one font not used in training and/or a second renderer, resolution or pixel format, with clean references and the same corruption protocol; source-disjoint splits with a held-out condition.",
            implementation_needs="New versioned dataset and manifests with condition metadata, frozen-champion zero-shot evaluation, and a predeclared adaptation protocol only if zero-shot transfer fails.",
            leakage_generalization_risk="Moderate: the held-out condition must be truly unseen, not merely absent from test, and must never be used for model or feature selection. Keep the target definition unchanged.",
            success_measure="Predeclared zero-shot RMSE/R² on the held-out condition, compared with the same-condition test, with source-grouped intervals and clean/original retention slices.",
            current_data_can_answer="No: all three fonts and the single renderer/resolution appear in every split.",
            rationale="Recommended: it addresses the largest unsupported claim, is directly motivated by the audit's coverage gap, and can be evaluated without changing the target or retraining the champion.",
        ),
        DirectionSpec(
            direction="C",
            title="Reconsider the target and evaluation through a separate research design",
            hypothesis="SSIM against a clean reference is not the right construct for readability or security; human judgments or solver outcomes would answer a different question.",
            scientific_value="High for construct validity, but it changes the research question and is outside the current synthetic-image scope.",
            data_needs="Human ratings or solver outcomes on real CAPTCHA samples, with appropriate consent and reliability protocols.",
            implementation_needs="New target pipeline, new metrics, new protocol and a separate validation study; the current benchmark cannot be reused as-is.",
            leakage_generalization_risk="High if human labels leak into model selection or if the new target is optimized directly without held-out raters.",
            success_measure="Reliability and validity of the new target, plus association with independent human or solver outcomes.",
            current_data_can_answer="No: the current data contain no human or solver outcomes.",
            rationale="Scientifically important but not the next incremental step; it requires a different research programme and new data collection.",
        ),
        DirectionSpec(
            direction="D",
            title="Continue feature engineering only if the audit identifies a concrete testable gap",
            hypothesis="A specific missing representation, such as a displacement-sensitive feature, can improve the hardest geometric severity without retention loss.",
            scientific_value="Low to moderate: M6A-M6D already tested four feature families and found redundancy with the global eight; the audit does not identify a concrete missing feature.",
            data_needs="No new data, but a genuinely new representation and a predeclared retention rule.",
            implementation_needs="New feature module, model refit and repeated evaluation on the same test groups.",
            leakage_generalization_risk="High: repeated feature search on the same 12 test groups inflates false-positive risk and does not create new independent evidence.",
            success_measure="Retention-safe geometric RMSE gain with a source-grouped interval excluding zero and no clean/original regression, confirmed on fresh sources.",
            current_data_can_answer="Not credibly: the same 12 test groups have already been used for M6A-M6D.",
            rationale="Not recommended without a concrete gap and fresh test data; it risks overfitting the small test set.",
        ),
    ]
    frame = pd.DataFrame([asdict(direction) for direction in directions])
    frame["recommended"] = frame["direction"] == config.recommendation.selected_direction
    if int(frame["recommended"].sum()) != 1:
        raise DatasetAuditError("exactly one next direction must be recommended")
    return frame


@dataclass(frozen=True, slots=True)
class AuditResult:
    """All deterministic tables and checks produced by the audit core."""

    inputs: AuditInputs
    inventory: pd.DataFrame
    composition: pd.DataFrame
    split_accounting: pd.DataFrame
    target_distributions: pd.DataFrame
    test_slice_metrics: pd.DataFrame
    source_metrics: pd.DataFrame
    source_severity_metrics: pd.DataFrame
    source_family_metrics: pd.DataFrame
    font_metrics: pd.DataFrame
    leave_one_source_out: pd.DataFrame
    error_concentration: pd.DataFrame
    uncertainty: pd.DataFrame
    feature_summary: pd.DataFrame
    feature_correlations: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_slice_means: pd.DataFrame
    feature_target_associations: pd.DataFrame
    claim_evidence: pd.DataFrame
    next_direction_comparison: pd.DataFrame
    isolation_checks: pd.DataFrame
    feature_diagnostics: dict[str, Any]
    summary: dict[str, Any]

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return the deterministic CSV tables in output order."""
        return {
            "dataset_inventory.csv": self.inventory,
            "dataset_composition.csv": self.composition,
            "split_accounting.csv": self.split_accounting,
            "target_distributions.csv": self.target_distributions,
            "test_slice_metrics.csv": self.test_slice_metrics,
            "source_metrics.csv": self.source_metrics,
            "source_severity_metrics.csv": self.source_severity_metrics,
            "source_family_metrics.csv": self.source_family_metrics,
            "font_metrics.csv": self.font_metrics,
            "leave_one_source_out.csv": self.leave_one_source_out,
            "error_concentration.csv": self.error_concentration,
            "uncertainty.csv": self.uncertainty,
            "feature_summary.csv": self.feature_summary,
            "feature_correlations.csv": self.feature_correlations,
            "feature_importance.csv": self.feature_importance,
            "feature_slice_means.csv": self.feature_slice_means,
            "feature_target_associations.csv": self.feature_target_associations,
            "claim_evidence.csv": self.claim_evidence,
            "next_direction_comparison.csv": self.next_direction_comparison,
            "isolation_checks.csv": self.isolation_checks,
        }


def _claim_schema_check(claims: pd.DataFrame) -> CheckResult:
    expected = {
        "claim_id",
        "claim",
        "classification",
        "evidence",
        "limitations",
        "evidence_scope",
        "evidence_metric",
        "evidence_value",
        "benchmark_coverage",
    }
    schema_ok = set(claims.columns) == expected
    ids_unique = bool(claims["claim_id"].is_unique)
    classes_ok = set(claims["classification"]).issubset(set(CLASSIFICATIONS))
    all_classes = set(CLASSIFICATIONS).issubset(set(claims["classification"]))
    return CheckResult(
        name="claim_schema",
        passed=bool(schema_ok and ids_unique and classes_ok and all_classes),
        detail=(
            f"columns={sorted(claims.columns)}; unique_ids={ids_unique}; "
            f"classes={sorted(set(claims['classification']))}; all_classes={all_classes}"
        ),
    )


def _recommendation_check(directions: pd.DataFrame, config: AuditConfig) -> CheckResult:
    recommended = directions[directions["recommended"].astype(bool)]
    return CheckResult(
        name="recommendation_single",
        passed=bool(len(recommended) == 1),
        detail=(
            f"recommended={recommended['direction'].tolist()}; "
            f"declared={config.recommendation.selected_direction}"
        ),
    )


def run_audit_core(inputs: AuditInputs) -> AuditResult:
    """Run the deterministic read-only audit over verified inputs."""
    inventory = build_dataset_inventory(inputs)
    composition = build_dataset_composition(inputs)
    split_accounting = build_split_accounting(inputs)
    target_distributions = build_target_distributions(inputs)
    test_slice_metrics = build_test_slice_metrics(inputs)
    source_metrics = build_source_metrics(inputs)
    source_severity_metrics = build_source_severity_metrics(inputs)
    source_family_metrics = build_source_family_metrics(inputs)
    font_metrics = build_font_metrics(inputs)
    leave_one_source_out = build_leave_one_source_out(inputs)
    error_concentration = build_error_concentration(inputs)
    uncertainty = build_uncertainty(inputs)
    feature_summary = build_feature_summary(inputs)
    feature_correlations = build_feature_correlations(inputs)
    feature_importance = build_feature_importance(inputs)
    feature_slice_means = build_feature_slice_means(inputs)
    feature_target_associations = build_feature_target_associations(inputs)
    feature_diagnostics = feature_diagnostics_summary(inputs)
    claims = build_claim_evidence(
        inputs,
        test_slice_metrics,
        error_concentration,
        feature_correlations,
        feature_diagnostics,
    )
    directions = build_direction_comparison(inputs.config)

    checks = list(inputs.checks)
    checks.append(_claim_schema_check(claims))
    checks.append(_recommendation_check(directions, inputs.config))

    expected_groups = inputs.config.expectations.m3_sources[2]
    grouping_ok = bool(
        (uncertainty["n_groups"] == expected_groups).all()
        and (uncertainty["independent_groups"] == expected_groups).all()
    )
    checks.append(
        CheckResult(
            name="source_grouped_bootstrap",
            passed=grouping_ok,
            detail=(
                f"scopes={uncertainty['scope'].nunique()}; "
                f"groups={sorted(int(value) for value in uncertainty['n_groups'].unique())}; "
                f"bootstrap={inputs.config.run.n_bootstrap}"
            ),
        )
    )

    clean_row = target_distributions[
        (target_distributions["dataset"] == "m3")
        & (target_distributions["split"] == "test")
        & (target_distributions["slice"] == "clean_reference")
    ].iloc[0]
    diagnostic_row = target_distributions[
        (target_distributions["dataset"] == "m5a")
        & (target_distributions["split"] == "test")
        & (target_distributions["slice"] == "diagnostic_0_5")
    ].iloc[0]
    checks.append(
        CheckResult(
            name="target_variance_boundaries",
            passed=bool(
                float(clean_row["variance"]) == 0.0
                and float(diagnostic_row["sd"]) < inputs.config.audit.near_degenerate_sd
            ),
            detail=(
                f"clean variance={float(clean_row['variance']):.3g}; "
                f"diagnostic sd={float(diagnostic_row['sd']):.6f} "
                f"< {inputs.config.audit.near_degenerate_sd}"
            ),
        )
    )

    sse_share_ok = True
    for scope in SCOPE_ORDER:
        subset = source_metrics[source_metrics["scope"] == scope]
        total_share = float(subset["sse_share"].sum())
        sse_share_ok = sse_share_ok and abs(total_share - 1.0) <= 1e-9
    checks.append(
        CheckResult(
            name="per_source_sse_accounting",
            passed=sse_share_ok,
            detail="per-source squared-error shares sum to 1 for every scope",
        )
    )

    loo_ok = True
    for scope in SCOPE_ORDER:
        subset = leave_one_source_out[leave_one_source_out["scope"] == scope]
        scope_sources = int(source_metrics[source_metrics["scope"] == scope]["source_id"].nunique())
        loo_ok = loo_ok and bool(
            len(subset) == scope_sources
            and (subset["n_sources_remaining"] == scope_sources - 1).all()
        )
    checks.append(
        CheckResult(
            name="leave_one_source_out_accounting",
            passed=loo_ok,
            detail="one omission per source and n_sources_remaining = n_sources - 1",
        )
    )

    feature_ok = bool(
        feature_diagnostics["n_rows"] == inputs.config.expectations.expanded_train_rows
        and feature_diagnostics["n_sources"] == inputs.config.expectations.m3_sources[0]
        and feature_diagnostics["n_features"] == len(FEATURE_NAMES)
    )
    checks.append(
        CheckResult(
            name="feature_diagnostics_training_only",
            passed=feature_ok,
            detail=(
                f"rows={feature_diagnostics['n_rows']}; sources={feature_diagnostics['n_sources']}; "
                f"features={feature_diagnostics['n_features']}; max_abs_corr="
                f"{feature_diagnostics['max_abs_correlation']:.6f}"
            ),
        )
    )

    _require_all_checks(checks, "M7A audit checks")

    m3 = _metric_row_for_scope(test_slice_metrics, "m3_test_overall")
    clean = _metric_row_for_scope(test_slice_metrics, "clean_reference")
    original = _metric_row_for_scope(test_slice_metrics, "original_blur_noise")
    geometric = _metric_row_for_scope(test_slice_metrics, "geometric_primary")
    diagnostic = _metric_row_for_scope(test_slice_metrics, "diagnostic_0_5")
    concentration = error_concentration[error_concentration["scope"] == "geometric_primary"].iloc[0]
    recommended = directions[directions["recommended"].astype(bool)].iloc[0]
    summary = {
        "champion": inputs.champion_name,
        "m3_test_rmse": float(m3["rmse"]),
        "m3_test_r2": float(m3["r2"]),
        "clean_rmse": float(clean["rmse"]),
        "original_rmse": float(original["rmse"]),
        "geometric_rmse": float(geometric["rmse"]),
        "geometric_r2": float(geometric["r2"]),
        "diagnostic_rmse": float(diagnostic["rmse"]),
        "diagnostic_r2": float(diagnostic["r2"]),
        "diagnostic_target_sd": float(diagnostic["target_sd"]),
        "geometric_top1_share": float(concentration["top1_share"]),
        "geometric_top3_share": float(concentration["top3_share"]),
        "feature_max_abs_correlation": float(feature_diagnostics["max_abs_correlation"]),
        "feature_top_importance": str(feature_diagnostics["top_importance_feature"]),
        "feature_top_importance_value": float(feature_diagnostics["top_importance"]),
        "n_sources": int(inputs.frames.m3_frame["source_id"].nunique()),
        "n_test_sources": int(geometric["n_sources"]),
        "n_bootstrap": inputs.config.run.n_bootstrap,
        "recommended_direction": str(recommended["direction"]),
        "recommended_title": str(recommended["title"]),
    }

    return AuditResult(
        inputs=inputs,
        inventory=inventory,
        composition=composition,
        split_accounting=split_accounting,
        target_distributions=target_distributions,
        test_slice_metrics=test_slice_metrics,
        source_metrics=source_metrics,
        source_severity_metrics=source_severity_metrics,
        source_family_metrics=source_family_metrics,
        font_metrics=font_metrics,
        leave_one_source_out=leave_one_source_out,
        error_concentration=error_concentration,
        uncertainty=uncertainty,
        feature_summary=feature_summary,
        feature_correlations=feature_correlations,
        feature_importance=feature_importance,
        feature_slice_means=feature_slice_means,
        feature_target_associations=feature_target_associations,
        claim_evidence=claims,
        next_direction_comparison=directions,
        isolation_checks=pd.DataFrame(
            [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                }
                for check in checks
            ]
        ),
        feature_diagnostics=feature_diagnostics,
        summary=summary,
    )


def _cell(value: object, *, float_digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "undefined"
        return f"{number:.{float_digits}f}"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    float_digits: int = 4,
    limit: int | None = None,
) -> str:
    if frame.empty:
        return "_none_"
    selected = frame.loc[:, list(columns)]
    if limit is not None:
        selected = selected.head(limit)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in selected.iterrows():
        lines.append(
            "| "
            + " | ".join(_cell(row[column], float_digits=float_digits) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def _report(result: AuditResult, config: AuditConfig) -> str:
    summary = result.summary
    claims = result.claim_evidence
    claim_counts = claims["classification"].value_counts().to_dict()
    test_targets = result.target_distributions[
        (result.target_distributions["split"] == "test")
        & (
            result.target_distributions["slice"].isin(
                (
                    "clean_reference",
                    "original_blur_noise",
                    "geometric_primary",
                    "geometric_primary_severity_2",
                    "geometric_primary_severity_3",
                    "geometric_primary_severity_4",
                    "diagnostic_0_5",
                )
            )
        )
    ]
    inventory_by_split = (
        result.inventory.groupby("split", sort=False)
        .agg(
            n_sources=("source_id", "nunique"),
            m3_rows=("m3_samples", "sum"),
            m5a_rows=("m5a_samples", "sum"),
            fonts=("font_id", "nunique"),
        )
        .reset_index()
    )
    top_sources = result.source_metrics[result.source_metrics["scope"] == "geometric_primary"].head(
        5
    )
    loo = result.leave_one_source_out[
        result.leave_one_source_out["scope"] == "geometric_primary"
    ].copy()
    loo["abs_delta"] = loo["delta_rmse"].abs()
    loo = loo.sort_values("abs_delta", ascending=False).head(5)
    concentration = result.error_concentration[
        result.error_concentration["scope"] == "geometric_primary"
    ]
    uncertainty_rmse = result.uncertainty[result.uncertainty["metric"] == "rmse"]
    top_correlations = result.feature_correlations.head(8)
    recommended = result.next_direction_comparison[
        result.next_direction_comparison["recommended"].astype(bool)
    ].iloc[0]
    direction_table = result.next_direction_comparison[
        ["direction", "title", "recommended", "current_data_can_answer"]
    ]

    lines = [
        "# M7A dataset-adequacy and evaluation audit",
        "",
        "**Scope:** read-only audit of the frozen synthetic dataset, targets, grouped splits, "
        "the frozen M5C champion and the M6A-M6D feature studies. No model was trained, "
        "recalibrated or selected in this milestone. The target is SSIM against the exact clean "
        "reference; this is not human readability, CAPTCHA solvability, security or real-world robustness.",
        "",
        f"**Audit config:** `{config.config_path}`  ",
        f"**Config SHA-256:** `{config.config_sha256}`  ",
        f"**Config identity hash:** `{config.config_hash}`  ",
        f"**Seed:** {config.run.seed}  ",
        f"**Bootstrap:** {config.run.n_bootstrap} source-grouped percentile replicates  ",
        f"**Champion:** `{summary['champion']}`  ",
        "",
        "## Executive summary",
        "",
        f"- The frozen data contain **{summary['n_sources']} synthetic sources**, "
        f"**540 M3 rows** and **480 M5A rows**. The split is 36/12/12 sources and "
        "324/108/108 M3 rows; M5A is 288/96/96 rows. Only **12 test source groups** are "
        "independent observations.",
        f"- The champion predicts the M3 blur/noise target well on this benchmark: "
        f"RMSE **{summary['m3_test_rmse']:.4f}**, R² **{summary['m3_test_r2']:.4f}**. "
        f"Clean-reference RMSE is **{summary['clean_rmse']:.4f}** and original-task RMSE is "
        f"**{summary['original_rmse']:.4f}**.",
        f"- The primary geometric slice is weaker and uncertain: RMSE "
        f"**{summary['geometric_rmse']:.4f}**, R² **{summary['geometric_r2']:.4f}**. "
        f"Severity 4 is the hardest rung; severity 2 and 4 have negative R² because the target "
        "has little variance relative to the residual.",
        f"- Geometric squared error is uneven across sources: the top source contributes "
        f"**{summary['geometric_top1_share']:.2%}** and the top three contribute "
        f"**{summary['geometric_top3_share']:.2%}**. Leave-one-source-out RMSE changes are small "
        "and descriptive; they are not a new generalisation estimate.",
        f"- The 0.5 px diagnostic is near-degenerate: target SD "
        f"**{summary['diagnostic_target_sd']:.5f}** is below the predeclared 0.005 margin, so its "
        f"R² (**{summary['diagnostic_r2']:.1f}**) must not be interpreted as stable predictive skill.",
        f"- The eight global features are strongly redundant: the maximum absolute training "
        f"correlation is **{summary['feature_max_abs_correlation']:.4f}**; "
        f"`{summary['feature_top_importance']}` has the largest frozen-model impurity importance "
        f"(**{summary['feature_top_importance_value']:.4f}**), but this is descriptive, not causal.",
        f"- Claim audit: **{claim_counts.get('supported', 0)} supported**, "
        f"**{claim_counts.get('suggestive', 0)} suggestive/uncertain**, "
        f"**{claim_counts.get('unsupported', 0)} unsupported**. The current data cannot test "
        "unseen fonts, renderers, resolutions or formats.",
        f"- Recommended next direction: **{summary['recommended_direction']} — "
        f"{summary['recommended_title']}**. It is not executed in M7A.",
        "",
        "## 1. Audit protocol and pinned inputs",
        "",
        "Every upstream file is pinned by SHA-256 before the audit and re-verified after it. "
        "The audit loads the M3 and M5A manifests through the frozen M5B/M5C configuration path, "
        "replays the champion's stored test predictions from the serialized model, and checks "
        "that the stored test-slice metrics match the replayed predictions.",
        "",
        f"- Pinned files verified: **{len(result.inputs.input_hashes)}**.",
        f"- M3 manifest: `{config.m3.manifest_sha256}`.",
        f"- M5A manifest: `{config.m5a.manifest_sha256}`.",
        f"- Champion artifact: `{config.m5c.champion_artifact_sha256}`.",
        f"- M5C predictions: `{config.m5c.predictions_sha256}`.",
        f"- M5C selection table: `{config.m5c.selection_table_sha256}`.",
        "- The independent unit is the source group; all variants of a sampled source are kept "
        "together in the bootstrap. No target, severity, font or split value enters the model "
        "feature matrix.",
        "",
        "## 2. Dataset inventory and independence",
        "",
        "The M3 and M5A datasets use the same 60 synthetic sources and the same source-disjoint "
        "split. M5A adds derived elastic variants and controls; it is not an independent sample of "
        "new sources.",
        "",
        _md_table(
            inventory_by_split,
            ["split", "n_sources", "m3_rows", "m5a_rows", "fonts"],
            float_digits=0,
        ),
        "",
        "- M3 has **540 distinct image files** (60 clean + 480 corruptions). M5A has **480 rows "
        "but only 360 distinct images**: clean, identity-control and zero-field-control rows are "
        "exact clean duplicates; the minimal control is near-clean; elastic rows are derived.",
        "- Fonts: **3** (`arial`, `courier_new`, `times_new_roman`), each present in every split. "
        "This is font overlap, not a held-out-font test.",
        "- Renderer: one Pillow/FreeType renderer; resolution: one 192x64 canvas; pixel format: "
        "one 8-bit grayscale PNG. There is no alternative rendering condition in any split.",
        f"- Source IDs are identical across M3 and M5A, and no source crosses a split. "
        f"Clean-image hashes match across the two datasets for all "
        f"**{int(result.inventory['clean_hash_matches'].sum())}** sources.",
        "",
        "## 3. Target distributions and variance",
        "",
        "The table below isolates the test slices used by the champion. `between_share` is a "
        "descriptive variance decomposition over source means and within-source variance, not an "
        "inferential ICC.",
        "",
        _md_table(
            test_targets,
            [
                "dataset",
                "slice",
                "severity_level",
                "n_samples",
                "n_sources",
                "mean",
                "sd",
                "min",
                "max",
                "variance",
                "between_variance_share",
                "near_degenerate",
            ],
            float_digits=5,
        ),
        "",
        "- The clean target is exactly 1.0 for every clean reference, so its variance is zero and "
        "R² is undefined. Clean RMSE is still a useful absolute error.",
        "- The 0.5 px diagnostic has a target SD below the predeclared 0.005 margin. Its R² is "
        "dominated by a tiny denominator and is reported as a diagnostic only.",
        "- M3 original corruption targets have much larger variance than the 1-3 px elastic "
        "targets, which is why the original-task R² is high while the geometric R² is low.",
        "",
        "## 4. Champion residuals by source and slice",
        "",
        _md_table(
            result.test_slice_metrics,
            [
                "scope",
                "n_samples",
                "n_sources",
                "mae",
                "rmse",
                "r2",
                "mean_residual",
                "sd_residual",
                "target_sd",
                "target_near_degenerate",
            ],
            float_digits=5,
        ),
        "",
        "Top geometric error sources:",
        "",
        _md_table(
            top_sources,
            ["source_id", "font_id", "n_samples", "mae", "rmse", "mean_residual", "sse_share"],
            float_digits=5,
        ),
        "",
        "Largest descriptive leave-one-source-out RMSE changes (geometric primary):",
        "",
        _md_table(
            loo,
            [
                "omitted_source_id",
                "rmse_full",
                "rmse_without",
                "delta_rmse",
                "delta_mae",
                "omitted_sse_share",
            ],
            float_digits=5,
        ),
        "",
        _md_table(
            concentration,
            [
                "scope",
                "n_sources",
                "top1_share",
                "top3_share",
                "hhi",
                "effective_sources",
                "sources_for_50pct",
                "sources_for_80pct",
            ],
            float_digits=5,
        ),
        "",
        "Per-font and per-family tables are in `font_metrics.csv`, `source_severity_metrics.csv` "
        "and `source_family_metrics.csv`. They show that aggregate performance is not uniform "
        "across fonts or severities, but the 12 test source groups are too few for stable "
        "font-level inference.",
        "",
        "## 5. Source-grouped uncertainty",
        "",
        "Percentile intervals below resample the 12 source groups with replacement and keep every "
        "variant of a sampled source. They are descriptive, not significance tests.",
        "",
        _md_table(
            uncertainty_rmse,
            [
                "scope",
                "n_groups",
                "n_samples",
                "estimate",
                "ci_low",
                "ci_high",
                "target_sd",
                "target_near_degenerate",
            ],
            float_digits=5,
        ),
        "",
        "The intervals are wide because there are only 12 independent groups. A clean R² is "
        "undefined; the bootstrap reports the defined-rate for R² in `uncertainty.csv`.",
        "",
        "## 6. Feature information and redundancy",
        "",
        "These diagnostics use only the 432 expanded training rows from 36 sources. Test targets "
        "are not used for feature selection, feature redesign or model fitting.",
        "",
        _md_table(
            result.feature_summary,
            [
                "feature",
                "mean",
                "sd",
                "min",
                "max",
                "skewness",
                "constant",
                "duplicate_of",
            ],
            float_digits=4,
        ),
        "",
        "Strongest training-only feature correlations:",
        "",
        _md_table(
            top_correlations,
            ["feature_a", "feature_b", "pearson", "abs_pearson", "spearman", "redundant"],
            float_digits=4,
        ),
        "",
        _md_table(
            result.feature_importance,
            ["feature", "importance", "importance_rank", "cumulative_importance"],
            float_digits=5,
        ),
        "",
        f"- Training rows: **{result.feature_diagnostics['n_rows']}**, sources: "
        f"**{result.feature_diagnostics['n_sources']}**.",
        f"- Maximum |Pearson r|: **{result.feature_diagnostics['max_abs_correlation']:.4f}**; "
        f"mean |r|: **{result.feature_diagnostics['mean_abs_correlation']:.4f}**; "
        f"pairs at or above |r|={result.feature_diagnostics['redundancy_threshold']:.2f}: "
        f"**{result.feature_diagnostics['n_redundant_pairs']}**.",
        f"- Standardized design condition number: "
        f"**{result.feature_diagnostics['condition_number']:.1f}**; maximum VIF: "
        f"**{result.feature_diagnostics['max_vif']:.1f}**. This is a redundancy warning, not a "
        "feature-selection instruction.",
        "- Feature means differ by family and severity (`feature_slice_means.csv`), so global "
        "feature scales are not stationary across the corruption mix.",
        "",
        "## 7. Claim-to-evidence table",
        "",
        _md_table(
            claims,
            ["claim_id", "claim", "classification", "evidence", "limitations"],
            float_digits=4,
        ),
        "",
        "The classification is deliberately conservative: `supported` means the current frozen "
        "benchmark directly measures the claim; `suggestive` means the evidence points in that "
        "direction but the 12-group design or low target variance is too weak; `unsupported` means "
        "the current data do not contain the required condition or outcome.",
        "",
        "## 8. Recommended next experiment",
        "",
        _md_table(
            direction_table,
            ["direction", "title", "recommended", "current_data_can_answer"],
            float_digits=4,
        ),
        "",
        f"**Selected: {recommended['direction']} — {recommended['title']}.**",
        "",
        str(recommended["rationale"]),
        "",
        "The other directions are compared on hypothesis, scientific value, data needs, "
        "implementation needs, leakage/generalisation risk and success measurement in "
        "`next_direction_comparison.csv`. Exactly one direction is recommended and none is run "
        "in M7A.",
        "",
        "## 9. Limitations",
        "",
        "- The independent sample size is 12 test source groups, not 108 or 36 image variants.",
        "- All fonts, the renderer, the resolution and the pixel format are shared across splits; "
        "the audit cannot identify unseen-condition generalisation.",
        "- The target is synthetic SSIM against a known clean reference. It is not a human "
        "readability, solve-rate or security outcome.",
        "- The geometric slice has low target variance at some severities, making R² unstable and "
        "sometimes strongly negative.",
        "- Feature importance is impurity-based and descriptive; correlated features share "
        "importance and the ranking is not causal.",
        "- Leave-one-source-out results are descriptive influence measures; no model was refit and "
        "no new generalisation estimate is claimed.",
        "",
        "## 10. Reproduce and verify",
        "",
        "```bash",
        "uv run --extra image --extra ml captcha-dataset-audit run \\",
        "    --config configs/dataset_adequacy_audit_v0_1_0.toml",
        "uv run --extra image --extra ml captcha-dataset-audit verify \\",
        "    --config configs/dataset_adequacy_audit_v0_1_0.toml",
        "```",
        "",
        "The run writes only to the Git-ignored audit report directory. `verify` re-loads the "
        "pinned inputs, replays the audit core, compares every table and the report text, and "
        "checks the recorded output hashes.",
        "",
        "## 11. Research-integrity boundary",
        "",
        "This audit can support claims about the frozen synthetic benchmark. It cannot establish "
        "that the model predicts human readability, CAPTCHA solve rate, security, or performance "
        "on real or unseen rendering conditions. Those claims require additional independent data "
        "and a separate protocol.",
        "",
    ]
    return "\n".join(lines)


def _safe_output_path(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    if resolved == Path.cwd().resolve():
        raise DatasetAuditError("refusing to write audit outputs into the repository root")
    for protected_text in PROTECTED_OUTPUT_ROOTS:
        protected = Path(protected_text).resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise DatasetAuditError(
                f"refusing to write into protected frozen path: {resolved} ({protected_text})"
            )
    return resolved


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _fingerprint_hash(fingerprint: Mapping[str, Mapping[str, object]]) -> str:
    return sha256_bytes(canonical_json(fingerprint).encode("utf-8"))


def _build_metadata(
    result: AuditResult,
    config: AuditConfig,
    *,
    frozen_before: Mapping[str, Mapping[str, object]],
    frozen_after: Mapping[str, Mapping[str, object]],
    elapsed_seconds: float,
    output_hashes: Mapping[str, str],
    report_sha256: str,
) -> dict[str, Any]:
    claim_counts = result.claim_evidence["classification"].value_counts().to_dict()
    recommended = result.next_direction_comparison[
        result.next_direction_comparison["recommended"].astype(bool)
    ].iloc[0]
    checks = result.isolation_checks.to_dict(orient="records")
    payload = {
        "audit": {
            "name": config.run.name,
            "version": config.run.version,
            "config": str(config.config_path),
            "config_sha256": config.config_sha256,
            "config_hash": config.config_hash,
            "claim_schema_version": config.schema.claim_schema_version,
        },
        "code_revision": code_revision(),
        "package_version": package_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn_version(),
        "runtime_seconds": round(elapsed_seconds, 6),
        "method": {
            "independent_unit": config.audit.group_column,
            "bootstrap_method": config.audit.bootstrap_method,
            "bootstrap_seed": config.run.seed,
            "n_bootstrap": config.run.n_bootstrap,
            "confidence_level": config.audit.confidence_level,
            "interval": "percentile",
            "near_degenerate_sd": config.audit.near_degenerate_sd,
            "redundancy_abs_correlation": config.audit.redundancy_abs_correlation,
            "top_k_sources": config.audit.top_k_sources,
            "feature_slice_dimensions": list(config.audit.feature_slice_dimensions),
            "bootstrap_scopes": list(config.audit.bootstrap_scopes),
        },
        "dataset": {
            "m3_sources": int(result.inputs.frames.m3_frame["source_id"].nunique()),
            "m3_rows": len(result.inputs.frames.m3_frame),
            "m5a_sources": int(result.inputs.frames.m5a_frame["source_id"].nunique()),
            "m5a_rows": len(result.inputs.frames.m5a_frame),
            "expanded_train_rows": len(result.inputs.frames.expanded_train),
            "expanded_val_rows": len(result.inputs.frames.expanded_val),
            "primary_test_rows": len(result.inputs.frames.primary_test),
            "diagnostic_test_rows": len(result.inputs.frames.diagnostic_test),
            "m3_test_rows": len(result.inputs.frames.m3_test),
            "n_test_sources": int(result.summary["n_test_sources"]),
        },
        "champion": {
            "name": result.inputs.champion_name,
            "artifact_sha256": config.m5c.champion_artifact_sha256,
            "card_sha256": config.m5c.champion_card_sha256,
            "calibration": result.inputs.champion_card.get("calibration"),
            "feature_names": result.inputs.champion_card.get("feature_names"),
            "target": result.inputs.champion_card.get("target"),
        },
        "inputs": result.inputs.input_hashes,
        "summary": result.summary,
        "feature_diagnostics": result.feature_diagnostics,
        "claims": {
            "counts": {str(key): int(value) for key, value in claim_counts.items()},
            "classifications": list(CLASSIFICATIONS),
            "schema_version": config.schema.claim_schema_version,
        },
        "recommendation": {
            "selected_direction": str(recommended["direction"]),
            "title": str(recommended["title"]),
            "rationale": str(recommended["rationale"]),
            "executed": False,
        },
        "frozen_assets": {
            "n_files": len(frozen_before),
            "before_sha256": _fingerprint_hash(frozen_before),
            "after_sha256": _fingerprint_hash(frozen_after),
            "unchanged": frozen_before == frozen_after,
        },
        "outputs": dict(sorted(output_hashes.items())),
        "report_sha256": report_sha256,
        "checks": checks,
    }
    return json_safe(payload)


def write_audit_outputs(
    result: AuditResult,
    config: AuditConfig,
    *,
    output_reports: Path | None = None,
    frozen_before: Mapping[str, Mapping[str, object]],
    frozen_after: Mapping[str, Mapping[str, object]],
    elapsed_seconds: float,
) -> dict[str, Path]:
    """Write deterministic tables, the report and run metadata."""
    output_dir = _safe_output_path(output_reports or config.run.output_reports)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in result.tables().items():
        path = output_dir / name
        _write_csv(frame, path)
        paths[name] = path
    report_path = output_dir / "report.md"
    report_text = _report(result, config)
    report_path.write_text(report_text, encoding="utf-8")
    paths["report.md"] = report_path

    output_hashes = {name: sha256_file(path) for name, path in paths.items()}
    metadata = _build_metadata(
        result,
        config,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        elapsed_seconds=elapsed_seconds,
        output_hashes=output_hashes,
        report_sha256=sha256_file(report_path),
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["run_metadata.json"] = metadata_path
    return paths


def run_dataset_audit(
    config: AuditConfig,
    *,
    output_reports: Path | None = None,
    verify_images: bool = True,
) -> tuple[AuditResult, dict[str, Path]]:
    """Load, verify, run and persist the complete M7A audit."""
    started = time.perf_counter()
    before = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    inputs = load_audit_inputs(config, verify_images=verify_images)
    result = run_audit_core(inputs)
    after = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    if before != after:
        raise DatasetAuditError("a frozen asset changed during the M7A audit")
    paths = write_audit_outputs(
        result,
        config,
        output_reports=output_reports,
        frozen_before=before,
        frozen_after=after,
        elapsed_seconds=time.perf_counter() - started,
    )
    return result, paths


def _normalise_for_compare(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise CSV empty-string/NaN round-tripping without hiding numeric drift."""
    result = frame.copy()
    for column in result.columns:
        if result[column].isna().all():
            result[column] = ""
            continue
        if not pd.api.types.is_numeric_dtype(result[column]) and not pd.api.types.is_bool_dtype(
            result[column]
        ):
            result[column] = (
                result[column].astype(object).where(result[column].notna(), "").astype(str)
            )
    return result


def _compare_frames(stored: pd.DataFrame, expected: pd.DataFrame, label: str) -> CheckResult:
    try:
        pd.testing.assert_frame_equal(
            _normalise_for_compare(stored),
            _normalise_for_compare(expected),
            check_dtype=False,
            check_like=False,
            atol=1e-6,
            rtol=1e-9,
        )
    except AssertionError as error:
        return CheckResult(
            name=f"output_replay_{label}",
            passed=False,
            detail=str(error).splitlines()[0][:300],
        )
    return CheckResult(
        name=f"output_replay_{label}",
        passed=True,
        detail=f"{len(expected)} rows x {len(expected.columns)} columns match the replay",
    )


def verify_dataset_audit(
    config: AuditConfig,
    *,
    output_reports: Path | None = None,
    verify_images: bool = False,
) -> dict[str, Any]:
    """Verify a written audit against the pinned inputs and a fresh replay."""
    output_dir = _safe_output_path(output_reports or config.run.output_reports)
    checks: list[CheckResult] = []
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return {
            "ok": False,
            "checks": (
                CheckResult(
                    name="run_metadata",
                    passed=False,
                    detail=f"missing {metadata_path}",
                ),
            ),
        }
    metadata = _read_json(metadata_path, "audit run metadata")

    config_hash_ok = (
        metadata.get("audit", {}).get("config_sha256") == config.config_sha256
        and metadata.get("audit", {}).get("config_hash") == config.config_hash
    )
    checks.append(
        CheckResult(
            name="config_identity",
            passed=config_hash_ok,
            detail=f"config_sha256={config.config_sha256}; config_hash={config.config_hash}",
        )
    )

    expected_outputs = metadata.get("outputs", {})
    output_hash_checks: list[CheckResult] = []
    if not isinstance(expected_outputs, Mapping):
        expected_outputs = {}
    for name in (*OUTPUT_TABLE_NAMES, "report.md"):
        path = output_dir / name
        expected_hash = expected_outputs.get(name)
        if not path.is_file() or not isinstance(expected_hash, str):
            output_hash_checks.append(
                CheckResult(
                    name=f"output_hash_{name}",
                    passed=False,
                    detail=f"missing file or recorded hash for {name}",
                )
            )
            continue
        actual = sha256_file(path)
        output_hash_checks.append(
            CheckResult(
                name=f"output_hash_{name}",
                passed=actual == expected_hash,
                detail=f"{actual} == {expected_hash}",
            )
        )
    checks.extend(output_hash_checks)

    before = rb._fingerprint_assets(FROZEN_ASSET_PATHS)  # noqa: SLF001
    frozen = metadata.get("frozen_assets", {})
    frozen_ok = (
        isinstance(frozen, Mapping)
        and frozen.get("after_sha256") == _fingerprint_hash(before)
        and bool(frozen.get("unchanged", False))
    )
    checks.append(
        CheckResult(
            name="frozen_assets_unchanged",
            passed=frozen_ok,
            detail=(
                f"n_files={len(before)}; recorded_after={frozen.get('after_sha256') if isinstance(frozen, Mapping) else None}"
            ),
        )
    )

    inputs = load_audit_inputs(config, verify_images=verify_images)
    replayed = run_audit_core(inputs)
    for name, expected in replayed.tables().items():
        stored_path = output_dir / name
        if not stored_path.is_file():
            checks.append(
                CheckResult(
                    name=f"output_replay_{name}",
                    passed=False,
                    detail=f"missing {stored_path}",
                )
            )
            continue
        stored = pd.read_csv(stored_path)
        checks.append(_compare_frames(stored, expected, name))

    report_path = output_dir / "report.md"
    report_ok = report_path.is_file() and report_path.read_text(encoding="utf-8") == _report(
        replayed, config
    )
    checks.append(
        CheckResult(
            name="report_replay",
            passed=report_ok,
            detail=f"report text matches replay={report_ok}",
        )
    )

    claims_ok = _claim_schema_check(replayed.claim_evidence).passed
    recommendation_ok = _recommendation_check(replayed.next_direction_comparison, config).passed
    checks.append(
        CheckResult(
            name="claim_and_recommendation_schema",
            passed=bool(claims_ok and recommendation_ok),
            detail=f"claims={claims_ok}; recommendation={recommendation_ok}",
        )
    )

    return {
        "ok": all(check.passed for check in checks),
        "checks": tuple(checks),
        "metadata": metadata,
        "result": replayed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the M7A dataset-adequacy and evaluation audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the audit and write Git-ignored outputs")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--output-reports", type=Path, default=None)
    run_parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="skip pixel-level image hash verification (manifest hashes are still checked)",
    )

    verify_parser = subparsers.add_parser(
        "verify", help="replay the audit and verify its written outputs"
    )
    verify_parser.add_argument("--config", required=True, type=Path)
    verify_parser.add_argument("--output-reports", type=Path, default=None)
    verify_parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="skip pixel-level image hash verification (manifest hashes are still checked)",
    )

    args = parser.parse_args(argv)
    try:
        config = load_audit_config(args.config)
    except DatasetAuditConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    try:
        if args.command == "run":
            result, paths = run_dataset_audit(
                config,
                output_reports=args.output_reports,
                verify_images=not args.skip_image_verification,
            )
            summary = result.summary
            print(
                f"audit: {config.describe()}; champion={summary['champion']}; "
                f"recommended={summary['recommended_direction']}"
            )
            print(
                f"M3 test RMSE={summary['m3_test_rmse']:.6f}, R2={summary['m3_test_r2']:.6f}; "
                f"geometric RMSE={summary['geometric_rmse']:.6f}, R2={summary['geometric_r2']:.6f}"
            )
            print(
                f"top-3 geometric SSE share={summary['geometric_top3_share']:.4f}; "
                f"max feature |r|={summary['feature_max_abs_correlation']:.4f}"
            )
            print(f"report: {paths['report.md']}")
            print(f"metadata: {paths['run_metadata.json']}")
            return 0
        verification = verify_dataset_audit(
            config,
            output_reports=args.output_reports,
            verify_images=not args.skip_image_verification,
        )
        for check in verification["checks"]:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
        print(f"verification: {'PASS' if verification['ok'] else 'FAIL'}")
        return 0 if verification["ok"] else 1
    except (DatasetAuditError, gb.GeometricBenchmarkError, ra.RetentionSelectionError) as error:
        print(f"audit error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
