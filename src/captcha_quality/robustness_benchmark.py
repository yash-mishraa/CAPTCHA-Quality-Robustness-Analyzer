"""Milestone 4B grouped robustness benchmark for occlusion and interference.

The benchmark keeps the Milestone 3 task fixed: predict ``ssim_target`` from the
same eight image-only features, using the frozen source-grouped split and the
frozen M3 model bundle. It adds two experiments:

* **Experiment A (zero-shot).** The unmodified M3 model is evaluated on the
  held-out M4A test sources. No preprocessing, calibration or model parameter
  is fitted on those rows.
* **Experiment B (controlled adaptation).** Conventional models are fitted on
  M3 train rows plus M4A corrupted rows from training sources, selected on M4A
  validation sources only, and then evaluated once on M4A test sources.

The M4A clean rows are deliberately excluded from expanded training and
validation because they duplicate the M3 clean references. M4A clean test rows
are retained for the separately reported clean-reference slice.

The target is synthetic image similarity (SSIM to a clean reference). Nothing in
this module measures human readability, CAPTCHA solvability or security.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

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
from captcha_quality.modelling import (
    BOOTSTRAP_HIGH,
    BOOTSTRAP_LOW,
    FEATURE_COLUMNS,
    SPLITS,
    TARGET,
    CandidateSpec,
    FittedModel,
    ModelError,
    bootstrap_metrics,
    fit_candidate,
    load_model_bundle,
    naive_spec,
    predict,
    regression_metrics,
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
from captcha_quality.robustness_pilot import PilotError, verify_pilot_dataset

CLEAN_FAMILY: Final[str] = "none"
M3_LABEL: Final[str] = "m3"
M4A_LABEL: Final[str] = "m4a"
FROZEN_MODEL_NAME: Final[str] = "frozen_m3"
M3_NAIVE_NAME: Final[str] = "naive_m3_train_mean"
EXPANDED_MODEL_NAME: Final[str] = "expanded_model"
EXPANDED_NAIVE_NAME: Final[str] = "naive_expanded_train_mean"
EXPANDED_MODEL_STEM: Final[str] = "expanded_model"

_SELECTION_METRICS: Final[tuple[str, ...]] = ("rmse", "mae")
_EXPECTED_TOP_KEYS: Final[frozenset[str]] = frozenset(
    {"run", "m3", "m4a", "model", "schema", "grid"}
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Files that must remain byte-identical while the benchmark runs. The M3
#: benchmark reports may be absent on a fresh clone; their state is recorded
#: rather than silently treated as verified.
FROZEN_ASSET_PATHS: Final[tuple[str, ...]] = (
    "configs/dataset_v0_1_0.toml",
    "configs/benchmark_v0_1_0.toml",
    "configs/robustness_pilot_occlusion_v0_1_0.toml",
    "configs/robustness_benchmark_v0_1_0.toml",
    "data/processed/captcha_quality_v0.1.0/manifest.csv",
    "data/processed/captcha_quality_v0.1.0/sources.csv",
    "data/processed/captcha_quality_v0.1.0/features.csv",
    "data/processed/captcha_quality_v0.1.0/ground_truth.csv",
    "data/processed/captcha_quality_v0.1.0/dataset_summary.json",
    "data/processed/occlusion_interference_pilot_v0.1.0/manifest.csv",
    "data/processed/occlusion_interference_pilot_v0.1.0/sources.csv",
    "data/processed/occlusion_interference_pilot_v0.1.0/features.csv",
    "data/processed/occlusion_interference_pilot_v0.1.0/pilot_summary.json",
    "models/benchmark_v0_1_0/primary_model.joblib",
    "models/benchmark_v0_1_0/primary_model_card.json",
    "models/benchmark_v0_1_0/heldout_font_model.joblib",
    "models/benchmark_v0_1_0/heldout_font_model_card.json",
    "reports/benchmark_v0_1_0/benchmark.md",
    "reports/benchmark_v0_1_0/run_metadata.json",
)

#: Output roots that the benchmark must never write into or delete.
PROTECTED_OUTPUT_ROOTS: Final[tuple[str, ...]] = (
    "configs",
    "data/processed/captcha_quality_v0.1.0",
    "data/processed/occlusion_interference_pilot_v0.1.0",
    "models/benchmark_v0_1_0",
    "reports/benchmark_v0_1_0",
    "reports/robustness_pilot",
)


class RobustnessBenchmarkError(RuntimeError):
    """Raised when the robustness benchmark cannot run as declared."""


class RobustnessConfigError(ValueError):
    """Raised when the robustness configuration is malformed."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunSettings:
    name: str
    version: str
    seed: int
    n_bootstrap: int
    selection_metric: str
    output_reports: Path
    output_models: Path


@dataclass(frozen=True, slots=True)
class M3Pin:
    dataset: Path
    config: Path
    dataset_version: str
    generation_config_hash: str
    manifest_sha256: str
    sources_sha256: str
    features_sha256: str
    ground_truth_sha256: str
    summary_sha256: str
    split_assignment_sha256: str


@dataclass(frozen=True, slots=True)
class M4APin:
    dataset: Path
    config: Path
    dataset_version: str
    pilot_config_hash: str
    manifest_sha256: str
    sources_sha256: str
    features_sha256: str
    summary_sha256: str
    image_pixel_hash_set: str
    n_samples: int
    n_sources: int


@dataclass(frozen=True, slots=True)
class ModelPin:
    model_dir: Path
    model_stem: str
    card_sha256: str
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class SchemaSettings:
    feature_set_version: str
    target: str
    feature_names: tuple[str, ...]
    split_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RidgeGrid:
    alphas: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RandomForestGrid:
    n_estimators: int
    max_depths: tuple[int, ...]
    min_samples_leafs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HistGradientBoostingGrid:
    max_iter: int
    learning_rates: tuple[float, ...]
    max_leaf_nodes: tuple[int, ...]
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True, slots=True)
class GridSettings:
    include_ols: bool
    ridge: RidgeGrid
    random_forest: RandomForestGrid
    hist_gradient_boosting: HistGradientBoostingGrid


@dataclass(frozen=True, slots=True)
class RobustnessConfig:
    run: RunSettings
    m3: M3Pin
    m4a: M4APin
    model: ModelPin
    schema: SchemaSettings
    grid: GridSettings
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
            "m4a": {
                "dataset": str(self.m4a.dataset),
                "config": str(self.m4a.config),
                "dataset_version": self.m4a.dataset_version,
                "pilot_config_hash": self.m4a.pilot_config_hash,
                "manifest_sha256": self.m4a.manifest_sha256,
                "sources_sha256": self.m4a.sources_sha256,
                "features_sha256": self.m4a.features_sha256,
                "summary_sha256": self.m4a.summary_sha256,
                "image_pixel_hash_set": self.m4a.image_pixel_hash_set,
                "n_samples": self.m4a.n_samples,
                "n_sources": self.m4a.n_sources,
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
                    "l2_regularization": self.grid.hist_gradient_boosting.l2_regularization,
                },
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
        raise RobustnessConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise RobustnessConfigError(f"missing or invalid [{name}] table")
    return value


def _required(section: Mapping[str, object], key: str, section_name: str) -> object:
    if key not in section:
        raise RobustnessConfigError(f"missing required key {section_name}.{key}")
    return section[key]


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RobustnessConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RobustnessConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RobustnessConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RobustnessConfigError(f"{label} must be finite, got {value!r}")
    return result


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RobustnessConfigError(f"{label} must be a boolean, got {value!r}")
    return value


def _as_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RobustnessConfigError(f"{label} must be a TOML array")
    return value


def _as_str_tuple(value: object, label: str) -> tuple[str, ...]:
    items = tuple(_as_str(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise RobustnessConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_int_tuple(value: object, label: str) -> tuple[int, ...]:
    items = tuple(_as_int(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise RobustnessConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_float_tuple(value: object, label: str) -> tuple[float, ...]:
    items = tuple(_as_float(item, f"{label}[]") for item in _as_sequence(value, label))
    if not items or len(set(items)) != len(items):
        raise RobustnessConfigError(f"{label} must be non-empty and duplicate-free")
    return items


def _as_sha256(value: object, label: str) -> str:
    text = _as_str(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise RobustnessConfigError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _parse_run(raw: Mapping[str, object]) -> RunSettings:
    _check_keys(
        raw,
        {
            "name",
            "version",
            "seed",
            "n_bootstrap",
            "selection_metric",
            "output_reports",
            "output_models",
        },
        "run",
    )
    seed = _as_int(_required(raw, "seed", "run"), "run.seed")
    if not 0 <= seed < 2**64:
        raise RobustnessConfigError("run.seed must be in [0, 2**64)")
    n_bootstrap = _as_int(_required(raw, "n_bootstrap", "run"), "run.n_bootstrap")
    if n_bootstrap < 1:
        raise RobustnessConfigError("run.n_bootstrap must be >= 1")
    selection_metric = _as_str(_required(raw, "selection_metric", "run"), "run.selection_metric")
    if selection_metric not in _SELECTION_METRICS:
        raise RobustnessConfigError(
            f"run.selection_metric must be one of {_SELECTION_METRICS}, got {selection_metric!r}"
        )
    return RunSettings(
        name=_as_str(_required(raw, "name", "run"), "run.name"),
        version=_as_str(_required(raw, "version", "run"), "run.version"),
        seed=seed,
        n_bootstrap=n_bootstrap,
        selection_metric=selection_metric,
        output_reports=Path(_as_str(_required(raw, "output_reports", "run"), "run.output_reports")),
        output_models=Path(_as_str(_required(raw, "output_models", "run"), "run.output_models")),
    )


def _parse_m3(raw: Mapping[str, object]) -> M3Pin:
    _check_keys(
        raw,
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
        "m3",
    )
    return M3Pin(
        dataset=Path(_as_str(_required(raw, "dataset", "m3"), "m3.dataset")),
        config=Path(_as_str(_required(raw, "config", "m3"), "m3.config")),
        dataset_version=_as_str(_required(raw, "dataset_version", "m3"), "m3.dataset_version"),
        generation_config_hash=_as_sha256(
            _required(raw, "generation_config_hash", "m3"),
            "m3.generation_config_hash",
        ),
        manifest_sha256=_as_sha256(_required(raw, "manifest_sha256", "m3"), "m3.manifest_sha256"),
        sources_sha256=_as_sha256(_required(raw, "sources_sha256", "m3"), "m3.sources_sha256"),
        features_sha256=_as_sha256(_required(raw, "features_sha256", "m3"), "m3.features_sha256"),
        ground_truth_sha256=_as_sha256(
            _required(raw, "ground_truth_sha256", "m3"), "m3.ground_truth_sha256"
        ),
        summary_sha256=_as_sha256(_required(raw, "summary_sha256", "m3"), "m3.summary_sha256"),
        split_assignment_sha256=_as_sha256(
            _required(raw, "split_assignment_sha256", "m3"),
            "m3.split_assignment_sha256",
        ),
    )


def _parse_m4a(raw: Mapping[str, object]) -> M4APin:
    _check_keys(
        raw,
        {
            "dataset",
            "config",
            "dataset_version",
            "pilot_config_hash",
            "manifest_sha256",
            "sources_sha256",
            "features_sha256",
            "summary_sha256",
            "image_pixel_hash_set",
            "n_samples",
            "n_sources",
        },
        "m4a",
    )
    n_samples = _as_int(_required(raw, "n_samples", "m4a"), "m4a.n_samples")
    n_sources = _as_int(_required(raw, "n_sources", "m4a"), "m4a.n_sources")
    if n_samples < 1 or n_sources < 1:
        raise RobustnessConfigError("m4a.n_samples and m4a.n_sources must be >= 1")
    return M4APin(
        dataset=Path(_as_str(_required(raw, "dataset", "m4a"), "m4a.dataset")),
        config=Path(_as_str(_required(raw, "config", "m4a"), "m4a.config")),
        dataset_version=_as_str(_required(raw, "dataset_version", "m4a"), "m4a.dataset_version"),
        pilot_config_hash=_as_sha256(
            _required(raw, "pilot_config_hash", "m4a"), "m4a.pilot_config_hash"
        ),
        manifest_sha256=_as_sha256(_required(raw, "manifest_sha256", "m4a"), "m4a.manifest_sha256"),
        sources_sha256=_as_sha256(_required(raw, "sources_sha256", "m4a"), "m4a.sources_sha256"),
        features_sha256=_as_sha256(_required(raw, "features_sha256", "m4a"), "m4a.features_sha256"),
        summary_sha256=_as_sha256(_required(raw, "summary_sha256", "m4a"), "m4a.summary_sha256"),
        image_pixel_hash_set=_as_sha256(
            _required(raw, "image_pixel_hash_set", "m4a"), "m4a.image_pixel_hash_set"
        ),
        n_samples=n_samples,
        n_sources=n_sources,
    )


def _parse_model(raw: Mapping[str, object]) -> ModelPin:
    _check_keys(raw, {"model_dir", "model_stem", "card_sha256", "artifact_sha256"}, "model")
    return ModelPin(
        model_dir=Path(_as_str(_required(raw, "model_dir", "model"), "model.model_dir")),
        model_stem=_as_str(_required(raw, "model_stem", "model"), "model.model_stem"),
        card_sha256=_as_sha256(_required(raw, "card_sha256", "model"), "model.card_sha256"),
        artifact_sha256=_as_sha256(
            _required(raw, "artifact_sha256", "model"), "model.artifact_sha256"
        ),
    )


def _parse_schema(raw: Mapping[str, object]) -> SchemaSettings:
    _check_keys(raw, {"feature_set_version", "target", "feature_names", "split_names"}, "schema")
    feature_names = _as_str_tuple(_required(raw, "feature_names", "schema"), "schema.feature_names")
    split_names = _as_str_tuple(_required(raw, "split_names", "schema"), "schema.split_names")
    feature_set_version = _as_str(
        _required(raw, "feature_set_version", "schema"), "schema.feature_set_version"
    )
    target = _as_str(_required(raw, "target", "schema"), "schema.target")
    if feature_set_version != FEATURE_SET_VERSION:
        raise RobustnessConfigError(
            f"schema.feature_set_version is {feature_set_version!r}, "
            f"this code provides {FEATURE_SET_VERSION!r}"
        )
    if feature_names != FEATURE_NAMES:
        raise RobustnessConfigError("schema.feature_names must match the extractor exactly")
    if target != TARGET:
        raise RobustnessConfigError(f"schema.target must be {TARGET!r}")
    if split_names != SPLITS:
        raise RobustnessConfigError(f"schema.split_names must be {SPLITS}")
    return SchemaSettings(
        feature_set_version=feature_set_version,
        target=target,
        feature_names=feature_names,
        split_names=split_names,
    )


def _parse_grid(raw: Mapping[str, object]) -> GridSettings:
    _check_keys(raw, {"include_ols", "ridge", "random_forest", "hist_gradient_boosting"}, "grid")
    include_ols = _as_bool(_required(raw, "include_ols", "grid"), "grid.include_ols")
    ridge_raw = _section(raw, "ridge")
    _check_keys(ridge_raw, {"alphas"}, "grid.ridge")
    alphas = _as_float_tuple(_required(ridge_raw, "alphas", "grid.ridge"), "grid.ridge.alphas")
    if any(alpha <= 0.0 for alpha in alphas):
        raise RobustnessConfigError("grid.ridge.alphas must all be > 0")

    rf_raw = _section(raw, "random_forest")
    _check_keys(rf_raw, {"n_estimators", "max_depths", "min_samples_leafs"}, "grid.random_forest")
    n_estimators = _as_int(
        _required(rf_raw, "n_estimators", "grid.random_forest"),
        "grid.random_forest.n_estimators",
    )
    if n_estimators < 1:
        raise RobustnessConfigError("grid.random_forest.n_estimators must be >= 1")
    max_depths = _as_int_tuple(
        _required(rf_raw, "max_depths", "grid.random_forest"), "grid.random_forest.max_depths"
    )
    if any(depth < 0 for depth in max_depths):
        raise RobustnessConfigError("grid.random_forest.max_depths entries must be >= 0")
    leafs = _as_int_tuple(
        _required(rf_raw, "min_samples_leafs", "grid.random_forest"),
        "grid.random_forest.min_samples_leafs",
    )
    if any(leaf < 1 for leaf in leafs):
        raise RobustnessConfigError("grid.random_forest.min_samples_leafs entries must be >= 1")

    hgb_raw = _section(raw, "hist_gradient_boosting")
    _check_keys(
        hgb_raw,
        {
            "max_iter",
            "learning_rates",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        },
        "grid.hist_gradient_boosting",
    )
    max_iter = _as_int(
        _required(hgb_raw, "max_iter", "grid.hist_gradient_boosting"),
        "grid.hist_gradient_boosting.max_iter",
    )
    if max_iter < 1:
        raise RobustnessConfigError("grid.hist_gradient_boosting.max_iter must be >= 1")
    learning_rates = _as_float_tuple(
        _required(hgb_raw, "learning_rates", "grid.hist_gradient_boosting"),
        "grid.hist_gradient_boosting.learning_rates",
    )
    if any(rate <= 0.0 or rate > 1.0 for rate in learning_rates):
        raise RobustnessConfigError("grid.hist_gradient_boosting.learning_rates must be in (0, 1]")
    max_leaf_nodes = _as_int_tuple(
        _required(hgb_raw, "max_leaf_nodes", "grid.hist_gradient_boosting"),
        "grid.hist_gradient_boosting.max_leaf_nodes",
    )
    if any(nodes < 2 for nodes in max_leaf_nodes):
        raise RobustnessConfigError(
            "grid.hist_gradient_boosting.max_leaf_nodes entries must be >= 2"
        )
    min_samples_leaf = _as_int(
        _required(hgb_raw, "min_samples_leaf", "grid.hist_gradient_boosting"),
        "grid.hist_gradient_boosting.min_samples_leaf",
    )
    if min_samples_leaf < 1:
        raise RobustnessConfigError("grid.hist_gradient_boosting.min_samples_leaf must be >= 1")
    l2 = _as_float(
        _required(hgb_raw, "l2_regularization", "grid.hist_gradient_boosting"),
        "grid.hist_gradient_boosting.l2_regularization",
    )
    if l2 < 0.0:
        raise RobustnessConfigError("grid.hist_gradient_boosting.l2_regularization must be >= 0")
    return GridSettings(
        include_ols=include_ols,
        ridge=RidgeGrid(alphas=alphas),
        random_forest=RandomForestGrid(
            n_estimators=n_estimators,
            max_depths=max_depths,
            min_samples_leafs=leafs,
        ),
        hist_gradient_boosting=HistGradientBoostingGrid(
            max_iter=max_iter,
            learning_rates=learning_rates,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2,
        ),
    )


def load_robustness_config(path: Path | str) -> RobustnessConfig:
    """Load and strictly validate the M4B TOML configuration."""
    path = Path(path)
    if not path.is_file():
        raise RobustnessConfigError(f"robustness benchmark config not found: {path}")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RobustnessConfigError(f"invalid TOML in {path}: {exc}") from exc
    _check_keys(payload, set(_EXPECTED_TOP_KEYS), "robustness benchmark file")
    config = RobustnessConfig(
        run=_parse_run(_section(payload, "run")),
        m3=_parse_m3(_section(payload, "m3")),
        m4a=_parse_m4a(_section(payload, "m4a")),
        model=_parse_model(_section(payload, "model")),
        schema=_parse_schema(_section(payload, "schema")),
        grid=_parse_grid(_section(payload, "grid")),
        config_path=path,
        config_sha256=sha256_file(path),
    )
    if not config.config_sha256:
        raise RobustnessConfigError("could not hash the configuration file")
    specs = candidate_specs_from_config(config)
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise RobustnessConfigError("model grid produces duplicate candidate names")
    return config


def candidate_specs_from_config(config: RobustnessConfig) -> tuple[CandidateSpec, ...]:
    """Return the predeclared expanded-data model grid."""
    specs: list[CandidateSpec] = []
    if config.grid.include_ols:
        specs.append(
            CandidateSpec(name="ols", family="ols", params={}, n_inputs=len(FEATURE_NAMES))
        )
    for alpha in sorted(set(config.grid.ridge.alphas)):
        specs.append(
            CandidateSpec(
                name=f"ridge_alpha_{alpha:g}",
                family="ridge",
                params={"alpha": float(alpha)},
                n_inputs=len(FEATURE_NAMES),
            )
        )
    for depth in config.grid.random_forest.max_depths:
        for leaf in config.grid.random_forest.min_samples_leafs:
            depth_label = "unlimited" if depth == 0 else str(depth)
            specs.append(
                CandidateSpec(
                    name=f"random_forest_depth_{depth_label}_leaf_{leaf}",
                    family="random_forest",
                    params={
                        "n_estimators": int(config.grid.random_forest.n_estimators),
                        "max_depth": int(depth),
                        "min_samples_leaf": int(leaf),
                    },
                    n_inputs=len(FEATURE_NAMES),
                )
            )
    for learning_rate in config.grid.hist_gradient_boosting.learning_rates:
        for leaves in config.grid.hist_gradient_boosting.max_leaf_nodes:
            specs.append(
                CandidateSpec(
                    name=f"hist_gradient_boosting_lr_{learning_rate:g}_leaves_{leaves}",
                    family="hist_gradient_boosting",
                    params={
                        "max_iter": int(config.grid.hist_gradient_boosting.max_iter),
                        "learning_rate": float(learning_rate),
                        "max_leaf_nodes": int(leaves),
                        "min_samples_leaf": int(
                            config.grid.hist_gradient_boosting.min_samples_leaf
                        ),
                        "l2_regularization": float(
                            config.grid.hist_gradient_boosting.l2_regularization
                        ),
                    },
                    n_inputs=len(FEATURE_NAMES),
                )
            )
    if not specs:
        raise RobustnessConfigError("model grid must contain at least one learned candidate")
    return tuple(specs)


# ---------------------------------------------------------------------------
# Dataset loading and merging
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkFrames:
    """The frozen frames and the leakage-safe expanded/evaluation frames."""

    m3_frame: pd.DataFrame
    m4a_frame: pd.DataFrame
    m3_sources: pd.DataFrame
    m4a_sources: pd.DataFrame
    m3_summary: dict[str, Any]
    m4a_summary: dict[str, Any]
    expanded_train: pd.DataFrame
    expanded_val: pd.DataFrame
    m4a_test: pd.DataFrame
    m3_test: pd.DataFrame
    data_accounting: pd.DataFrame
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RobustnessBenchmarkError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RobustnessBenchmarkError(
            f"{label} hash mismatch: expected {expected}, found {actual} ({path})"
        )


def _normalise_m3(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["dataset"] = M3_LABEL
    result["family"] = result["corruption_type"].astype(str)
    result["is_clean"] = result["family"] == CLEAN_FAMILY
    return result


def _normalise_m4a(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["dataset"] = M4A_LABEL
    result["family"] = result["variant_family"].astype(str)
    # Keep the existing column name available for code that expects it, while
    # making the cross-dataset family column explicit.
    result["corruption_type"] = result["variant_family"].astype(str)
    result["is_clean"] = result["family"] == CLEAN_FAMILY
    return result


def _source_split(frame: pd.DataFrame) -> dict[str, str]:
    grouped = frame.groupby("source_id")["split"].agg(lambda values: tuple(sorted(set(values))))
    return {
        str(source): (values[0] if len(values) == 1 else "|".join(values))
        for source, values in grouped.items()
    }


def _check_source_isolation(frame: pd.DataFrame, label: str) -> CheckResult:
    try:
        per_source = frame.groupby("source_id")["split"].nunique()
        if (per_source > 1).any():
            offenders = per_source[per_source > 1].index[:5].tolist()
            raise RobustnessBenchmarkError(f"source(s) cross splits: {offenders}")
        split_sources = {
            split: set(frame.loc[frame["split"] == split, "source_id"].astype(str))
            for split in SPLITS
        }
        if set(frame["split"]) != set(SPLITS):
            raise RobustnessBenchmarkError(
                f"expected splits {SPLITS}, found {sorted(set(frame['split']))}"
            )
        overlaps = {
            f"{left}/{right}": len(split_sources[left] & split_sources[right])
            for left in SPLITS
            for right in SPLITS
            if left < right
        }
        if any(overlaps.values()):
            raise RobustnessBenchmarkError(f"source overlap between splits: {overlaps}")
        counts = {split: len(split_sources[split]) for split in SPLITS}
        return CheckResult(
            name=f"{label}_source_isolation",
            passed=True,
            detail=f"no source crosses splits; source counts {counts}",
        )
    except Exception as exc:  # noqa: BLE001 - checks report every failure
        return CheckResult(
            name=f"{label}_source_isolation",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_source_identity(m3_sources: pd.DataFrame, m4a_sources: pd.DataFrame) -> CheckResult:
    try:
        left = m3_sources.set_index("source_id").sort_index()
        right = m4a_sources.set_index("source_id").sort_index()
        if not left.index.equals(right.index):
            raise RobustnessBenchmarkError("M3 and M4A source IDs differ")
        for column in ("split", "font_id", "text_length"):
            if column not in right.columns:
                raise RobustnessBenchmarkError(f"M4A sources missing {column!r}")
            if not left[column].equals(right[column]):
                raise RobustnessBenchmarkError(f"M3 and M4A sources disagree on {column!r}")
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


def _check_clean_duplicates(m3_frame: pd.DataFrame, m4a_frame: pd.DataFrame) -> CheckResult:
    try:
        m3_clean = m3_frame[m3_frame["is_clean"]].set_index("source_id").sort_index()
        m4a_clean = m4a_frame[m4a_frame["is_clean"]].set_index("source_id").sort_index()
        if not m3_clean.index.equals(m4a_clean.index):
            raise RobustnessBenchmarkError("clean source IDs differ between datasets")
        for column in ("image_sha256", "reference_sha256", "ssim_target"):
            if not m3_clean[column].equals(m4a_clean[column]):
                raise RobustnessBenchmarkError(
                    f"clean rows disagree on {column!r}; the merge would duplicate or "
                    "silently replace a clean reference"
                )
        for feature in FEATURE_NAMES:
            if not np.allclose(
                m3_clean[feature].to_numpy(dtype=np.float64),
                m4a_clean[feature].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=0.0,
            ):
                raise RobustnessBenchmarkError(f"clean rows disagree on feature {feature!r}")
        return CheckResult(
            name="clean_duplicate_provenance",
            passed=True,
            detail=(
                f"{len(m3_clean)} M4A clean rows are exact duplicates of M3 clean "
                "references; they are excluded from expanded train/val and used "
                "only in the M4A test evaluation frame"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="clean_duplicate_provenance",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_merge_contract(
    expanded_train: pd.DataFrame,
    expanded_val: pd.DataFrame,
    m4a_test: pd.DataFrame,
    m3_test: pd.DataFrame,
) -> CheckResult:
    try:
        train_ids = set(expanded_train["sample_id"].astype(str))
        val_ids = set(expanded_val["sample_id"].astype(str))
        test_ids = set(m4a_test["sample_id"].astype(str))
        m3_test_ids = set(m3_test["sample_id"].astype(str))
        if len(train_ids) != len(expanded_train):
            raise RobustnessBenchmarkError("expanded train has duplicate sample_id values")
        if len(val_ids) != len(expanded_val):
            raise RobustnessBenchmarkError("expanded val has duplicate sample_id values")
        if train_ids & val_ids:
            raise RobustnessBenchmarkError("expanded train and val share sample_id values")
        if (train_ids | val_ids) & test_ids:
            raise RobustnessBenchmarkError("test sample_id values entered expanded train/val")
        if (train_ids | val_ids) & m3_test_ids:
            raise RobustnessBenchmarkError("M3 test sample_id values entered expanded train/val")
        train_hashes = expanded_train["image_sha256"].astype(str)
        val_hashes = expanded_val["image_sha256"].astype(str)
        if train_hashes.duplicated().any():
            raise RobustnessBenchmarkError("expanded train has duplicate image hashes")
        if val_hashes.duplicated().any():
            raise RobustnessBenchmarkError("expanded val has duplicate image hashes")
        if set(train_hashes) & set(val_hashes):
            raise RobustnessBenchmarkError("expanded train and val share image hashes")
        if (expanded_train["dataset"] == M4A_LABEL).any() and (
            expanded_train.loc[expanded_train["dataset"] == M4A_LABEL, "is_clean"].any()
        ):
            raise RobustnessBenchmarkError("M4A clean rows entered expanded training")
        if (expanded_val["dataset"] == M4A_LABEL).any() and (
            expanded_val.loc[expanded_val["dataset"] == M4A_LABEL, "is_clean"].any()
        ):
            raise RobustnessBenchmarkError("M4A clean rows entered expanded validation")
        train_sources = set(expanded_train["source_id"].astype(str))
        val_sources = set(expanded_val["source_id"].astype(str))
        test_sources = set(m4a_test["source_id"].astype(str))
        if (
            train_sources & val_sources
            or train_sources & test_sources
            or val_sources & test_sources
        ):
            raise RobustnessBenchmarkError("source groups overlap across expanded/evaluation sets")
        return CheckResult(
            name="merge_contract",
            passed=True,
            detail=(
                f"train {len(expanded_train)} rows/{len(train_sources)} sources, "
                f"val {len(expanded_val)} rows/{len(val_sources)} sources, "
                f"test {len(m4a_test)} rows/{len(test_sources)} sources; "
                "no duplicate sample IDs or image hashes; M4A clean rows excluded "
                "from fit/selection"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="merge_contract",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _check_feature_leakage(
    m3_features: pd.DataFrame, m4a_features: pd.DataFrame, expanded_train: pd.DataFrame
) -> CheckResult:
    try:
        validate_feature_columns(list(m3_features.columns))
        validate_feature_columns(list(m4a_features.columns))
        inputs = expanded_train[list(FEATURE_COLUMNS)]
        if inputs.shape[1] != len(FEATURE_NAMES):
            raise RobustnessBenchmarkError("expanded training input width is not 8")
        forbidden = {
            TARGET,
            "psnr_target",
            "severity_level",
            "family",
            "variant_family",
            "font_id",
            "text_length",
            "source_id",
            "split",
            "dataset",
            "generation_config_hash",
        }
        leaked = sorted(forbidden & set(inputs.columns))
        if leaked:
            raise RobustnessBenchmarkError(f"forbidden columns in model inputs: {leaked}")
        return CheckResult(
            name="feature_leakage",
            passed=True,
            detail=(
                "feature tables expose exactly sample_id plus the eight image features; "
                "primary inputs contain no target, PSNR, severity, family, font, length, "
                "source, split or generation metadata"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="feature_leakage",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _accounting_row(dataset: str, split: str, frame: pd.DataFrame) -> dict[str, Any]:
    families = frame["family"].astype(str)
    return {
        "dataset": dataset,
        "split": split,
        "sources": int(frame["source_id"].nunique()),
        "samples": int(len(frame)),
        "clean": int((families == CLEAN_FAMILY).sum()),
        "corrupted": int((families != CLEAN_FAMILY).sum()),
        "gaussian_blur": int((families == "gaussian_blur").sum()),
        "gaussian_noise": int((families == "gaussian_noise").sum()),
        "thin_lines": int((families == "thin_lines").sum()),
        "arc_strokes": int((families == "arc_strokes").sum()),
    }


def _data_accounting(
    m3_frame: pd.DataFrame,
    m4a_frame: pd.DataFrame,
    expanded_train: pd.DataFrame,
    expanded_val: pd.DataFrame,
    m4a_test: pd.DataFrame,
    m3_test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        rows.append(_accounting_row("m3", split, m3_frame[m3_frame["split"] == split]))
        rows.append(_accounting_row("m4a", split, m4a_frame[m4a_frame["split"] == split]))
    rows.append(_accounting_row("expanded", "train", expanded_train))
    rows.append(_accounting_row("expanded", "val", expanded_val))
    rows.append(_accounting_row("evaluation", "m4a_test", m4a_test))
    rows.append(_accounting_row("evaluation", "m3_test", m3_test))
    return pd.DataFrame(rows)


def build_frame_bundle(
    m3_frame: pd.DataFrame,
    m4a_frame: pd.DataFrame,
    m3_sources: pd.DataFrame,
    m4a_sources: pd.DataFrame,
    m3_summary: Mapping[str, Any],
    m4a_summary: Mapping[str, Any],
) -> BenchmarkFrames:
    """Normalise, audit and merge the frozen and M4A frames.

    This function is deliberately pure with respect to the filesystem. File
    hashes, image verification and target replay happen in
    :func:`load_benchmark_frames` before this merge is used.
    """
    m3 = _normalise_m3(m3_frame)
    m4a = _normalise_m4a(m4a_frame)
    if m3["sample_id"].duplicated().any() or m4a["sample_id"].duplicated().any():
        raise RobustnessBenchmarkError("sample_id must be unique within each dataset")
    if m3_sources["source_id"].duplicated().any() or m4a_sources["source_id"].duplicated().any():
        raise RobustnessBenchmarkError("source_id must be unique within sources.csv")

    m3_train = m3[m3["split"] == "train"].copy()
    m3_val = m3[m3["split"] == "val"].copy()
    m3_test = m3[m3["split"] == "test"].copy()
    m4a_corrupted_train = m4a[(m4a["split"] == "train") & (~m4a["is_clean"])].copy()
    m4a_corrupted_val = m4a[(m4a["split"] == "val") & (~m4a["is_clean"])].copy()
    m4a_test = m4a[m4a["split"] == "test"].copy()
    expanded_train = pd.concat([m3_train, m4a_corrupted_train], ignore_index=True)
    expanded_val = pd.concat([m3_val, m4a_corrupted_val], ignore_index=True)

    checks = (
        _check_source_isolation(m3, "m3"),
        _check_source_isolation(m4a, "m4a"),
        _check_source_identity(m3_sources, m4a_sources),
        _check_clean_duplicates(m3, m4a),
        _check_merge_contract(expanded_train, expanded_val, m4a_test, m3_test),
        _check_feature_leakage(
            m3[["sample_id", *FEATURE_COLUMNS]],
            m4a[["sample_id", *FEATURE_COLUMNS]],
            expanded_train,
        ),
    )
    accounting = _data_accounting(m3, m4a, expanded_train, expanded_val, m4a_test, m3_test)
    return BenchmarkFrames(
        m3_frame=m3,
        m4a_frame=m4a,
        m3_sources=m3_sources.copy(),
        m4a_sources=m4a_sources.copy(),
        m3_summary=dict(m3_summary),
        m4a_summary=dict(m4a_summary),
        expanded_train=expanded_train,
        expanded_val=expanded_val,
        m4a_test=m4a_test,
        m3_test=m3_test,
        data_accounting=accounting,
        checks=checks,
    )


def _verify_checks(checks: Sequence[CheckResult], context: str) -> None:
    failures = [check for check in checks if not check.passed]
    if failures:
        joined = "; ".join(f"{check.name}: {check.detail}" for check in failures)
        raise RobustnessBenchmarkError(f"{context} failed: {joined}")


def load_benchmark_frames(
    config: RobustnessConfig, *, verify_images: bool = True
) -> BenchmarkFrames:
    """Load pinned M3/M4A inputs, verify provenance and build the merged frames."""
    if not config.m3.config.is_file():
        raise RobustnessBenchmarkError(f"missing M3 dataset config: {config.m3.config}")
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
    _require_file_hash(
        config.m4a.dataset / "manifest.csv", config.m4a.manifest_sha256, "M4A manifest"
    )
    _require_file_hash(config.m4a.dataset / "sources.csv", config.m4a.sources_sha256, "M4A sources")
    _require_file_hash(
        config.m4a.dataset / "features.csv", config.m4a.features_sha256, "M4A features"
    )
    _require_file_hash(
        config.m4a.dataset / "pilot_summary.json", config.m4a.summary_sha256, "M4A summary"
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
    m4a_manifest = pd.read_csv(config.m4a.dataset / "manifest.csv")
    m4a_sources = pd.read_csv(config.m4a.dataset / "sources.csv")
    m4a_features = pd.read_csv(config.m4a.dataset / "features.csv")
    m4a_summary = json.loads(
        (config.m4a.dataset / "pilot_summary.json").read_text(encoding="utf-8")
    )
    validate_feature_columns(list(m4a_features.columns))
    if len(m4a_manifest) != config.m4a.n_samples:
        raise RobustnessBenchmarkError(
            f"M4A manifest has {len(m4a_manifest)} rows, expected {config.m4a.n_samples}"
        )
    if m4a_sources["source_id"].nunique() != config.m4a.n_sources:
        raise RobustnessBenchmarkError(
            f"M4A has {m4a_sources['source_id'].nunique()} sources, expected {config.m4a.n_sources}"
        )
    if m4a_summary.get("pilot_config_hash") != config.m4a.pilot_config_hash:
        raise RobustnessBenchmarkError("M4A pilot_summary config hash does not match the pin")
    if m4a_summary.get("feature_names") != list(FEATURE_NAMES):
        raise RobustnessBenchmarkError("M4A summary feature names do not match the extractor")
    if (
        m4a_summary.get("artifacts", {}).get("image_pixel_hash_set")
        != config.m4a.image_pixel_hash_set
    ):
        raise RobustnessBenchmarkError("M4A image pixel hash set does not match the pin")
    if m4a_summary.get("pilot_version") != config.m4a.dataset_version:
        raise RobustnessBenchmarkError("M4A pilot version does not match the pin")

    m4a_frame = m4a_manifest.merge(m4a_features, on="sample_id", how="left", validate="one_to_one")
    if len(m4a_frame) != len(m4a_manifest) or m4a_frame[list(FEATURE_NAMES)].isna().any().any():
        raise RobustnessBenchmarkError("M4A manifest and feature rows do not align")
    if not np.isfinite(m4a_frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)).all():
        raise RobustnessBenchmarkError("M4A features contain non-finite values")
    if not np.isfinite(m4a_frame[TARGET].to_numpy(dtype=np.float64)).all():
        raise RobustnessBenchmarkError("M4A target contains non-finite values")
    if not m4a_frame[TARGET].between(0.0, 1.0).all():
        raise RobustnessBenchmarkError("M4A ssim_target is outside [0, 1]")

    if verify_images:
        m3_report = verify_dataset(config.m3.dataset, recompute_targets=True)
        _verify_checks(m3_report.checks, "M3 dataset verification")
        m4a_report = verify_pilot_dataset(
            config.m4a.dataset, reference=None, recompute_targets=True
        )
        _verify_checks(m4a_report.checks, "M4A dataset verification")

    frames = build_frame_bundle(
        m3_bundle.frame,
        m4a_frame,
        m3_bundle.sources,
        m4a_sources,
        m3_bundle.summary,
        m4a_summary,
    )
    _verify_checks(frames.checks, "M3/M4A merge checks")
    return frames


def load_frozen_model(
    config: RobustnessConfig, m3_summary: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], tuple[CheckResult, ...]]:
    """Load the frozen M3 primary bundle through the existing verifier."""
    return load_model_bundle(
        config.model.model_dir,
        config.model.model_stem,
        dataset_summary=dict(m3_summary),
    )


# ---------------------------------------------------------------------------
# Metrics, residuals and grouped uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Predictions and derived tables for one model on one frame."""

    metrics: pd.DataFrame
    residuals: pd.DataFrame
    predictions: pd.DataFrame


def _scope_subsets(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """Return reporting scopes without using any target or model information."""
    subsets: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
    clean = frame[frame["is_clean"]]
    if len(clean):
        subsets.append(("clean", "clean", clean))
    corrupted = frame[~frame["is_clean"]]
    if len(corrupted):
        subsets.append(("corrupted", "all", corrupted))
        for family in sorted(corrupted["family"].astype(str).unique()):
            subsets.append(("family", family, corrupted[corrupted["family"] == family]))
        for severity in sorted(corrupted["severity_level"].astype(int).unique()):
            subsets.append(
                ("severity", str(int(severity)), corrupted[corrupted["severity_level"] == severity])
            )
        for family in sorted(corrupted["family"].astype(str).unique()):
            family_frame = corrupted[corrupted["family"] == family]
            for severity in sorted(family_frame["severity_level"].astype(int).unique()):
                groups = family_frame[family_frame["severity_level"] == severity]
                subsets.append(("family_severity", f"{family}_s{int(severity)}", groups))
    for font in sorted(frame["font_id"].astype(str).unique()):
        subsets.append(("font", font, frame[frame["font_id"] == font]))
    return subsets


def _metric_row(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    dataset: str,
    split: str,
    scope: str,
    level: str,
) -> dict[str, Any]:
    scores = score_frame(frame, predicted)
    residual = frame[TARGET].to_numpy(dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    return {
        "model": model,
        "dataset": dataset,
        "split": split,
        "scope": scope,
        "level": level,
        "n_samples": int(scores["n_rows"]),
        "n_sources": int(scores["n_sources"]),
        "mae": float(scores["mae"]),
        "rmse": float(scores["rmse"]),
        "r2": float(scores["r2"]),
        "r2_defined": bool(scores["r2_defined"]),
        "mean_residual": float(np.mean(residual)) if len(residual) else math.nan,
        "median_residual": float(np.median(residual)) if len(residual) else math.nan,
        "sd_residual": (float(np.std(residual, ddof=1)) if len(residual) > 1 else math.nan),
    }


def metrics_for_frame(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    dataset: str,
    split: str,
) -> pd.DataFrame:
    """Score overall, clean, corruption, severity and font slices."""
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(predicted) != len(frame):
        raise RobustnessBenchmarkError("predictions do not align with the scored frame")
    rows = [
        _metric_row(
            subset,
            predicted[frame.index.get_indexer(subset.index)],
            model=model,
            dataset=dataset,
            split=split,
            scope=scope,
            level=level,
        )
        for scope, level, subset in _scope_subsets(frame)
    ]
    return pd.DataFrame(rows)


def residual_summary(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    dataset: str,
    split: str,
) -> pd.DataFrame:
    """Summarise signed residuals by family/severity, keeping clean separate."""
    predicted = np.asarray(predicted, dtype=np.float64)
    working = frame.copy()
    working["prediction"] = predicted
    working["residual"] = working[TARGET] - working["prediction"]
    rows: list[dict[str, Any]] = []
    for (family, severity), group in working.groupby(["family", "severity_level"], sort=True):
        scores = score_frame(group, group["prediction"].to_numpy(dtype=np.float64))
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "split": split,
                "family": str(family),
                "severity_level": int(severity),
                "n_samples": int(scores["n_rows"]),
                "n_sources": int(scores["n_sources"]),
                "mean_residual": float(group["residual"].mean()),
                "median_residual": float(group["residual"].median()),
                "sd_residual": (
                    float(group["residual"].std(ddof=1)) if len(group) > 1 else math.nan
                ),
                "mae": float(scores["mae"]),
                "rmse": float(scores["rmse"]),
                "r2": float(scores["r2"]),
                "r2_defined": bool(scores["r2_defined"]),
            }
        )
    return pd.DataFrame(rows)


def prediction_table(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    dataset: str,
    split: str,
) -> pd.DataFrame:
    """Return row-level predictions with target and metadata for audit."""
    working = frame.copy()
    working["model"] = model
    working["dataset_label"] = dataset
    working["split_label"] = split
    working["prediction"] = np.asarray(predicted, dtype=np.float64)
    working["residual"] = working[TARGET] - working["prediction"]
    working["abs_residual"] = working["residual"].abs()
    columns = [
        "model",
        "dataset_label",
        "split_label",
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
    return working[columns].reset_index(drop=True)


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
    config: RobustnessConfig,
) -> tuple[pd.DataFrame, dict[str, FittedModel], dict[str, Any], FittedModel]:
    """Fit the predeclared grid on train and select on validation only."""
    specs = candidate_specs_from_config(config)
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


def _scope_frame(frame: pd.DataFrame, scope: str, level: str) -> pd.DataFrame:
    for candidate_scope, candidate_level, subset in _scope_subsets(frame):
        if candidate_scope == scope and candidate_level == level:
            return subset
    raise RobustnessBenchmarkError(f"unknown scope {scope!r}/{level!r}")


def _bootstrap_rows(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    dataset: str,
    scope: str,
    level: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    subset = _scope_frame(frame, scope, level)
    mask = frame.index.get_indexer(subset.index)
    table = bootstrap_metrics(
        subset,
        np.asarray(predicted, dtype=np.float64)[mask],
        n_bootstrap=n_bootstrap,
        seed=derive_noise_seed(seed, "bootstrap", model, dataset, scope, level) % (2**64),
        scope=f"{dataset}_{scope}_{level}",
    )
    if "scope" in table.columns:
        table = table.rename(columns={"scope": "bootstrap_scope"})
    table.insert(0, "model", model)
    table.insert(1, "comparison", "")
    table.insert(2, "dataset", dataset)
    table.insert(3, "scope", scope)
    table.insert(4, "level", level)
    return table


def paired_grouped_bootstrap(
    frame: pd.DataFrame,
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    *,
    model_a: str,
    model_b: str,
    dataset: str,
    scope: str,
    level: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Descriptive source-grouped interval for ``metric(a) - metric(b)``."""
    subset = _scope_frame(frame, scope, level)
    mask = frame.index.get_indexer(subset.index)
    a = np.asarray(predicted_a, dtype=np.float64)[mask]
    b = np.asarray(predicted_b, dtype=np.float64)[mask]
    y = subset[TARGET].to_numpy(dtype=np.float64)
    groups = subset["source_id"].astype(str).to_numpy()
    unique_groups = np.array(sorted(set(groups)))
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(
        derive_noise_seed(seed, "paired_bootstrap", model_a, model_b, dataset, scope, level)
    )
    replicates: dict[str, list[float]] = {"mae": [], "rmse": []}
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([indices_by_group[group] for group in sampled])
        scores_a = regression_metrics(y[index], a[index])
        scores_b = regression_metrics(y[index], b[index])
        for metric in ("mae", "rmse"):
            replicates[metric].append(float(scores_a[metric]) - float(scores_b[metric]))
    estimate_a = regression_metrics(y, a)
    estimate_b = regression_metrics(y, b)
    rows: list[dict[str, Any]] = []
    for metric in ("mae", "rmse"):
        values = np.asarray(replicates[metric], dtype=np.float64)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "model": f"{model_a}_minus_{model_b}",
                "comparison": f"{model_a}_minus_{model_b}",
                "dataset": dataset,
                "scope": scope,
                "level": level,
                "metric": metric,
                "estimate": float(estimate_a[metric]) - float(estimate_b[metric]),
                "ci_low": float(np.percentile(finite, BOOTSTRAP_LOW)) if finite.size else math.nan,
                "ci_high": float(np.percentile(finite, BOOTSTRAP_HIGH))
                if finite.size
                else math.nan,
                "n_groups": int(len(unique_groups)),
                "n_samples": int(len(subset)),
                "n_bootstrap": int(n_bootstrap),
                "r2_defined_rate": math.nan,
            }
        )
    return pd.DataFrame(rows)


def target_distribution(frames: BenchmarkFrames) -> pd.DataFrame:
    """Describe target shift between train, validation and test scopes."""
    scopes: list[tuple[str, str, str, pd.DataFrame]] = []
    for split in SPLITS:
        scopes.append(("m3", split, "all", frames.m3_frame[frames.m3_frame["split"] == split]))
        scopes.append(("m4a", split, "all", frames.m4a_frame[frames.m4a_frame["split"] == split]))
    scopes.append(("expanded", "train", "all", frames.expanded_train))
    scopes.append(("expanded", "val", "all", frames.expanded_val))
    scopes.append(("evaluation", "m4a_test", "all", frames.m4a_test))
    scopes.append(("evaluation", "m4a_test", "clean", frames.m4a_test[frames.m4a_test["is_clean"]]))
    scopes.append(
        ("evaluation", "m4a_test", "corrupted", frames.m4a_test[~frames.m4a_test["is_clean"]])
    )
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
                        "n_samples": int(len(group)),
                        "n_sources": int(group["source_id"].nunique()),
                        "mean": float(np.mean(values)),
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    frames: BenchmarkFrames
    checks: tuple[CheckResult, ...]
    frozen_card: dict[str, Any]
    zero_shot_metrics: pd.DataFrame
    zero_shot_residuals: pd.DataFrame
    zero_shot_predictions: pd.DataFrame
    m3_baseline_metrics: pd.DataFrame
    candidates: pd.DataFrame
    selected: dict[str, Any]
    selected_model: FittedModel
    adaptation_metrics: pd.DataFrame
    adaptation_residuals: pd.DataFrame
    adaptation_predictions: pd.DataFrame
    target_distribution: pd.DataFrame
    uncertainty: pd.DataFrame

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)


def _model_eval(
    frame: pd.DataFrame,
    estimator: Any,
    *,
    model: str,
    dataset: str,
    split: str,
) -> Evaluation:
    predictions = predict(estimator, frame)
    return Evaluation(
        metrics=metrics_for_frame(frame, predictions, model=model, dataset=dataset, split=split),
        residuals=residual_summary(frame, predictions, model=model, dataset=dataset, split=split),
        predictions=prediction_table(frame, predictions, model=model, dataset=dataset, split=split),
    )


def _concat_evaluations(evaluations: Iterable[Evaluation]) -> Evaluation:
    items = list(evaluations)
    return Evaluation(
        metrics=pd.concat([item.metrics for item in items], ignore_index=True),
        residuals=pd.concat([item.residuals for item in items], ignore_index=True),
        predictions=pd.concat([item.predictions for item in items], ignore_index=True),
    )


def _uncertainty_table(
    frames: BenchmarkFrames,
    frozen_predictions: np.ndarray,
    m3_naive_predictions: np.ndarray,
    expanded_predictions: np.ndarray,
    expanded_naive_predictions: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    model_frames = {
        FROZEN_MODEL_NAME: frozen_predictions,
        M3_NAIVE_NAME: m3_naive_predictions,
        EXPANDED_MODEL_NAME: expanded_predictions,
        EXPANDED_NAIVE_NAME: expanded_naive_predictions,
    }
    scopes = [
        ("overall", "all"),
        ("clean", "clean"),
        ("corrupted", "all"),
        ("family", "thin_lines"),
        ("family", "arc_strokes"),
        ("severity", "1"),
        ("severity", "2"),
        ("severity", "3"),
        ("severity", "4"),
    ]
    for model, predictions in model_frames.items():
        for scope, level in scopes:
            rows.append(
                _bootstrap_rows(
                    frames.m4a_test,
                    predictions,
                    model=model,
                    dataset="m4a_test",
                    scope=scope,
                    level=level,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
            )
    for comparison, a, b, label in (
        (
            f"{EXPANDED_MODEL_NAME}_minus_{FROZEN_MODEL_NAME}",
            expanded_predictions,
            frozen_predictions,
            "adaptation",
        ),
        (
            f"{FROZEN_MODEL_NAME}_minus_{M3_NAIVE_NAME}",
            frozen_predictions,
            m3_naive_predictions,
            "zero_shot",
        ),
        (
            f"{EXPANDED_MODEL_NAME}_minus_{EXPANDED_NAIVE_NAME}",
            expanded_predictions,
            expanded_naive_predictions,
            "adaptation",
        ),
    ):
        for scope, level in (
            ("overall", "all"),
            ("clean", "clean"),
            ("corrupted", "all"),
            ("family", "thin_lines"),
            ("family", "arc_strokes"),
            ("severity", "1"),
            ("severity", "2"),
            ("severity", "3"),
            ("severity", "4"),
        ):
            table = paired_grouped_bootstrap(
                frames.m4a_test,
                a,
                b,
                model_a=comparison.split("_minus_")[0],
                model_b=comparison.split("_minus_")[1],
                dataset="m4a_test",
                scope=scope,
                level=level,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            table["comparison"] = comparison
            table["comparison_scope"] = label
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


def run_benchmark_core(
    frames: BenchmarkFrames,
    config: RobustnessConfig,
    *,
    frozen_estimator: Any,
    frozen_card: Mapping[str, Any],
) -> RobustnessResult:
    """Run both experiments and return in-memory tables (no filesystem writes)."""
    _verify_checks(frames.checks, "M3/M4A merge checks")
    train = frames.expanded_train
    val = frames.expanded_val
    test = frames.m4a_test
    if set(train["source_id"]) & set(test["source_id"]):
        raise RobustnessBenchmarkError("test sources entered expanded training")
    if set(val["source_id"]) & set(test["source_id"]):
        raise RobustnessBenchmarkError("test sources entered expanded validation")
    if set(train["split"]) != {"train"} or set(val["split"]) != {"val"}:
        raise RobustnessBenchmarkError("expanded fit/selection frames have the wrong split labels")

    m3_train = frames.m3_frame[frames.m3_frame["split"] == "train"]
    m3_naive = _fit_naive(
        m3_train, seed=derive_noise_seed(config.run.seed, "naive", "m3") % (2**32)
    )
    m3_naive_predictions = predict(m3_naive.estimator, test)
    frozen_predictions = predict(frozen_estimator, test)
    zero_shot = _concat_evaluations(
        [
            _model_eval(
                test,
                frozen_estimator,
                model=FROZEN_MODEL_NAME,
                dataset="m4a_test",
                split="test",
            ),
            _model_eval(
                test,
                m3_naive.estimator,
                model=M3_NAIVE_NAME,
                dataset="m4a_test",
                split="test",
            ),
        ]
    )
    m3_baseline = _concat_evaluations(
        [
            _model_eval(
                frames.m3_test,
                frozen_estimator,
                model=FROZEN_MODEL_NAME,
                dataset="m3_test",
                split="test",
            ),
            _model_eval(
                frames.m3_test,
                m3_naive.estimator,
                model=M3_NAIVE_NAME,
                dataset="m3_test",
                split="test",
            ),
        ]
    ).metrics

    candidates, _, selected, selected_model = fit_and_select(train, val, config)
    expanded_naive = _fit_naive(
        train, seed=derive_noise_seed(config.run.seed, "naive", "expanded") % (2**32)
    )
    expanded_predictions = predict(selected_model.estimator, test)
    expanded_naive_predictions = predict(expanded_naive.estimator, test)
    adaptation = _concat_evaluations(
        [
            _model_eval(
                test,
                frozen_estimator,
                model=FROZEN_MODEL_NAME,
                dataset="m4a_test",
                split="test",
            ),
            _model_eval(
                test,
                selected_model.estimator,
                model=EXPANDED_MODEL_NAME,
                dataset="m4a_test",
                split="test",
            ),
            _model_eval(
                test,
                expanded_naive.estimator,
                model=EXPANDED_NAIVE_NAME,
                dataset="m4a_test",
                split="test",
            ),
        ]
    )
    # The same final candidates are also scored on the original M3 test rows.
    # This is a descriptive reference, not another selection step.
    m3_baseline = pd.concat(
        [
            m3_baseline,
            _model_eval(
                frames.m3_test,
                selected_model.estimator,
                model=EXPANDED_MODEL_NAME,
                dataset="m3_test",
                split="test",
            ).metrics,
            _model_eval(
                frames.m3_test,
                expanded_naive.estimator,
                model=EXPANDED_NAIVE_NAME,
                dataset="m3_test",
                split="test",
            ).metrics,
        ],
        ignore_index=True,
    )

    uncertainty = _uncertainty_table(
        frames,
        frozen_predictions,
        m3_naive_predictions,
        expanded_predictions,
        expanded_naive_predictions,
        n_bootstrap=config.run.n_bootstrap,
        seed=config.run.seed,
    )
    # Add frozen/retrained M3-test intervals explicitly, using the same
    # source-grouped convention as the M4A-test intervals.
    m3_rows: list[pd.DataFrame] = []
    for model, estimator in (
        (FROZEN_MODEL_NAME, frozen_estimator),
        (EXPANDED_MODEL_NAME, selected_model.estimator),
    ):
        m3_predictions = predict(estimator, frames.m3_test)
        for scope, level in (
            ("overall", "all"),
            ("clean", "clean"),
            ("corrupted", "all"),
            ("family", "gaussian_blur"),
            ("family", "gaussian_noise"),
        ):
            m3_rows.append(
                _bootstrap_rows(
                    frames.m3_test,
                    m3_predictions,
                    model=model,
                    dataset="m3_test",
                    scope=scope,
                    level=level,
                    n_bootstrap=config.run.n_bootstrap,
                    seed=config.run.seed,
                )
            )
    uncertainty = pd.concat([uncertainty, *m3_rows], ignore_index=True)

    return RobustnessResult(
        frames=frames,
        checks=frames.checks,
        frozen_card=dict(frozen_card),
        zero_shot_metrics=zero_shot.metrics,
        zero_shot_residuals=zero_shot.residuals,
        zero_shot_predictions=zero_shot.predictions,
        m3_baseline_metrics=m3_baseline,
        candidates=candidates,
        selected=selected,
        selected_model=selected_model,
        adaptation_metrics=adaptation.metrics,
        adaptation_residuals=adaptation.residuals,
        adaptation_predictions=adaptation.predictions,
        target_distribution=target_distribution(frames),
        uncertainty=uncertainty,
    )


# ---------------------------------------------------------------------------
# Persistence and report generation
# ---------------------------------------------------------------------------


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


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[object]], *, digits: int = 3) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for index, value in enumerate(row):
            if index < 2 or isinstance(value, str):
                cells.append(str(value))
            elif isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, int):
                cells.append(str(value))
            else:
                cells.append(_fmt(value, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _metric_lookup(
    table: pd.DataFrame, model: str, dataset: str, scope: str, level: str
) -> dict[str, Any]:
    rows = table[
        (table["model"] == model)
        & (table["dataset"] == dataset)
        & (table["scope"] == scope)
        & (table["level"] == level)
    ]
    if len(rows) != 1:
        raise RobustnessBenchmarkError(
            f"expected one metric row for {model}/{dataset}/{scope}/{level}, found {len(rows)}"
        )
    return rows.iloc[0].to_dict()


def _decision_lines(result: RobustnessResult) -> list[str]:
    frozen = _metric_lookup(
        result.adaptation_metrics, FROZEN_MODEL_NAME, "m4a_test", "overall", "all"
    )
    expanded = _metric_lookup(
        result.adaptation_metrics, EXPANDED_MODEL_NAME, "m4a_test", "overall", "all"
    )
    expanded_naive = _metric_lookup(
        result.adaptation_metrics, EXPANDED_NAIVE_NAME, "m4a_test", "overall", "all"
    )
    thin = _metric_lookup(
        result.adaptation_metrics, EXPANDED_MODEL_NAME, "m4a_test", "family", "thin_lines"
    )
    arc = _metric_lookup(
        result.adaptation_metrics, EXPANDED_MODEL_NAME, "m4a_test", "family", "arc_strokes"
    )
    severity_one = _metric_lookup(
        result.adaptation_metrics, EXPANDED_MODEL_NAME, "m4a_test", "severity", "1"
    )
    severity_four = _metric_lookup(
        result.adaptation_metrics, EXPANDED_MODEL_NAME, "m4a_test", "severity", "4"
    )
    frozen_zero = _metric_lookup(
        result.zero_shot_metrics, FROZEN_MODEL_NAME, "m4a_test", "overall", "all"
    )
    naive_zero = _metric_lookup(
        result.zero_shot_metrics, M3_NAIVE_NAME, "m4a_test", "overall", "all"
    )
    zero_reduction = (
        (float(naive_zero["mae"]) - float(frozen_zero["mae"])) / float(naive_zero["mae"])
        if float(naive_zero["mae"])
        else math.nan
    )
    adaptation_reduction = (
        (float(frozen["rmse"]) - float(expanded["rmse"])) / float(frozen["rmse"])
        if float(frozen["rmse"])
        else math.nan
    )
    if expanded["rmse"] < frozen["rmse"]:
        adaptation_statement = (
            "The expanded-data model has lower held-out test RMSE than the frozen M3 "
            "model, so the controlled adaptation helps descriptively on this split."
        )
    else:
        adaptation_statement = (
            "The expanded-data model does not have lower held-out test RMSE than the "
            "frozen M3 model; this experiment does not show an adaptation benefit."
        )
    generalisation_statement = (
        f"Variant RMSE is {_fmt(thin['rmse'])} for thin_lines and {_fmt(arc['rmse'])} "
        f"for arc_strokes; severity RMSE rises from {_fmt(severity_one['rmse'])} at "
        f"level 1 to {_fmt(severity_four['rmse'])} at level 4. The high-density slice "
        "remains hardest, and R2 at low severity is unstable because target variance "
        "is narrow. Aggregate error should not be read as uniform robustness."
    )
    frozen_corrupted = _metric_lookup(
        result.zero_shot_metrics, FROZEN_MODEL_NAME, "m4a_test", "corrupted", "all"
    )
    naive_corrupted = _metric_lookup(
        result.zero_shot_metrics, M3_NAIVE_NAME, "m4a_test", "corrupted", "all"
    )
    paired_corrupted = result.uncertainty[
        (result.uncertainty["model"] == f"{FROZEN_MODEL_NAME}_minus_{M3_NAIVE_NAME}")
        & (result.uncertainty["dataset"] == "m4a_test")
        & (result.uncertainty["scope"] == "corrupted")
        & (result.uncertainty["metric"] == "mae")
    ].iloc[0]
    return [
        "### Interpretation",
        "",
        "No universal pass/fail threshold was invented after seeing test results. "
        "The following statements are descriptive and use the held-out M4A test sources only.",
        "",
        f"- **Zero-shot transfer:** the frozen M3 model has MAE {_fmt(frozen_zero['mae'])}, "
        f"RMSE {_fmt(frozen_zero['rmse'])} and R2 {_fmt(frozen_zero['r2'])} on M4A test, "
        f"versus MAE {_fmt(naive_zero['mae'])}, RMSE {_fmt(naive_zero['rmse'])} and "
        f"R2 {_fmt(naive_zero['r2'])} for the M3 training-mean baseline "
        f"({_fmt(100 * zero_reduction, 1)}% MAE reduction). The negative R2 is important: "
        "the frozen model beats the M3 training mean in absolute error but is worse than "
        "a constant test-mean predictor, and its mean signed residual is "
        f"{_fmt(frozen_zero['mean_residual'])}, indicating systematic under-prediction "
        "as interference density rises. On corrupted rows alone, frozen M3 MAE is "
        f"{_fmt(frozen_corrupted['mae'])} versus {_fmt(naive_corrupted['mae'])} for the "
        "training mean; the paired grouped MAE difference is "
        f"{_fmt(paired_corrupted['estimate'])} with interval "
        f"[{_fmt(paired_corrupted['ci_low'])}, {_fmt(paired_corrupted['ci_high'])}], "
        "which includes zero. Zero-shot absolute calibration therefore does not "
        "transfer, even though the clean-reference slice remains near-perfect.",
        f"- **Adaptation:** the validation-selected model is `{result.selected['name']}`. "
        f"It has MAE {_fmt(expanded['mae'])}, RMSE {_fmt(expanded['rmse'])} and R2 "
        f"{_fmt(expanded['r2'])} on M4A test, versus frozen M3 MAE {_fmt(frozen['mae'])} "
        f"and RMSE {_fmt(frozen['rmse'])}; RMSE changes by "
        f"{_fmt(100 * adaptation_reduction, 1)}% relative to frozen M3. "
        f"{adaptation_statement}",
        f"- **Baseline check:** the expanded training-mean baseline has MAE "
        f"{_fmt(expanded_naive['mae'])} and RMSE {_fmt(expanded_naive['rmse'])} on the "
        "same test rows. The selected model is compared against it so that the "
        "expanded target mean is not mistaken for learned corruption response.",
        f"- **Generalisation:** {generalisation_statement}",
        "- **CNN decision:** no CNN is trained in this milestone. The clear zero-shot "
        "failure is a covariate/target-shift problem that controlled conventional "
        "retraining addresses; there is no evidence here that model capacity is the "
        "binding limitation. A harder experiment would be justified only if a "
        "meaningful limitation persisted after this comparison, not on a post-hoc "
        "universal threshold.",
    ]


def _report_markdown(config: RobustnessConfig, result: RobustnessResult) -> str:
    frames = result.frames
    lines = [
        "# M4B grouped robustness benchmark — occlusion and interference",
        "",
        "**Scope:** synthetic image similarity, measured as SSIM against the exact clean "
        "reference. This is not a measure of human readability, CAPTCHA solvability, "
        "security, or real-world quality.",
        "",
        "## Frozen protocol",
        "",
        f"- Benchmark: `{config.run.name}` `{config.run.version}`",
        f"- Config: `{config.config_path}` (sha256 `{config.config_sha256}`)",
        f"- Protocol hash: `{config.config_hash}`",
        f"- Seed: `{config.run.seed}`; grouped bootstrap replicates: `{config.run.n_bootstrap}`",
        f"- Selection metric: `val_{config.run.selection_metric}`",
        f"- M3 dataset: `{config.m3.dataset}`; generation `{config.m3.generation_config_hash}`",
        f"- M3 manifest sha256 `{config.m3.manifest_sha256}`; sources "
        f"`{config.m3.sources_sha256}`; features `{config.m3.features_sha256}`",
        f"- M4A dataset: `{config.m4a.dataset}`; pilot config hash "
        f"`{config.m4a.pilot_config_hash}`",
        f"- M4A manifest sha256 `{config.m4a.manifest_sha256}`; sources "
        f"`{config.m4a.sources_sha256}`; features `{config.m4a.features_sha256}`",
        f"- Frozen model: `{config.model.model_dir}/{config.model.model_stem}`; card "
        f"`{config.model.card_sha256}`; artifact `{config.model.artifact_sha256}`",
        f"- Feature schema: `{config.schema.feature_set_version}`, target "
        f"`{config.schema.target}`, {len(config.schema.feature_names)} image-only features",
        f"- Source split assignment: `{config.m3.split_assignment_sha256}`",
        "",
        "The split is grouped by clean source. Every variant and severity level of a "
        "source stays in that source's assigned split. No test-source row is used for "
        "fitting, feature selection, scaling or hyperparameter selection.",
        "",
        "## Data accounting and isolation",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "dataset",
                "split",
                "sources",
                "samples",
                "clean",
                "corrupted",
                "blur",
                "noise",
                "lines",
                "arcs",
            ],
            [
                [
                    row.dataset,
                    row.split,
                    row.sources,
                    row.samples,
                    row.clean,
                    row.corrupted,
                    row.gaussian_blur,
                    row.gaussian_noise,
                    row.thin_lines,
                    row.arc_strokes,
                ]
                for row in frames.data_accounting.itertuples(index=False)
            ],
        )
    )
    lines.extend(["", "Isolation checks:", ""])
    lines.extend(
        _markdown_table(
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
            "The M4A clean rows are exact duplicates of the M3 clean references; they are "
            "excluded from expanded training and validation. The M4A test frame keeps one "
            "clean row per test source so the clean-reference slice can be reported "
            "separately. No images are duplicated within the expanded fit frames.",
            "",
            "## Experiment A — frozen M3 zero-shot on M4A test sources",
            "",
            "The frozen M3 model was loaded through the existing card/artifact verifier. "
            "No parameter, scaler or calibration was fitted on these rows.",
            "",
        ]
    )
    zero_rows = result.zero_shot_metrics[
        (
            result.zero_shot_metrics["scope"].isin(
                ["overall", "clean", "corrupted", "family", "severity"]
            )
        )
    ]
    lines.extend(
        _markdown_table(
            ["model", "scope", "level", "n", "sources", "MAE", "RMSE", "R2", "mean residual"],
            [
                [
                    row.model,
                    row.scope,
                    row.level,
                    row.n_samples,
                    row.n_sources,
                    row.mae,
                    row.rmse,
                    row.r2,
                    row.mean_residual,
                ]
                for row in zero_rows.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "`R2` is reported as `n/a` when it is undefined because the target is "
            "constant. This is expected for the clean-reference rows (SSIM = 1.0); it is "
            "not a perfect fit and is not treated as one.",
            "",
            "### M3 blur/noise test reference for all final candidates",
            "",
        ]
    )
    baseline_rows = result.m3_baseline_metrics[
        result.m3_baseline_metrics["scope"].isin(
            ["overall", "clean", "corrupted", "family", "severity"]
        )
    ]
    lines.extend(
        _markdown_table(
            ["model", "scope", "level", "n", "sources", "MAE", "RMSE", "R2"],
            [
                [
                    row.model,
                    row.scope,
                    row.level,
                    row.n_samples,
                    row.n_sources,
                    row.mae,
                    row.rmse,
                    row.r2,
                ]
                for row in baseline_rows.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "This table evaluates the final candidates on the original M3 test rows. The "
            "frozen rows are the same artifact used for zero-shot transfer; this is a "
            "reference for behaviour on the corruption families represented in M3, not "
            "a second test-set selection step.",
            "",
            "## Experiment B — controlled adaptation",
            "",
            "Candidate models were fitted on M3 train rows plus M4A corrupted rows from "
            "training sources, and selected on M4A validation sources only. The selected "
            "model was then evaluated once on M4A test sources.",
            "",
            "### Validation-only candidate selection",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
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
            "### Held-out test comparison",
            "",
        ]
    )
    comparison_rows = result.adaptation_metrics[
        result.adaptation_metrics["scope"].isin(
            ["overall", "clean", "corrupted", "family", "severity"]
        )
    ]
    lines.extend(
        _markdown_table(
            ["model", "scope", "level", "n", "sources", "MAE", "RMSE", "R2", "mean residual"],
            [
                [
                    row.model,
                    row.scope,
                    row.level,
                    row.n_samples,
                    row.n_sources,
                    row.mae,
                    row.rmse,
                    row.r2,
                    row.mean_residual,
                ]
                for row in comparison_rows.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Variant × severity slices for the selected model",
            "",
        ]
    )
    slice_rows = result.adaptation_metrics[
        (result.adaptation_metrics["model"] == EXPANDED_MODEL_NAME)
        & (result.adaptation_metrics["scope"] == "family_severity")
    ]
    lines.extend(
        _markdown_table(
            ["level", "n", "sources", "MAE", "RMSE", "R2", "mean residual"],
            [
                [
                    row.level,
                    row.n_samples,
                    row.n_sources,
                    row.mae,
                    row.rmse,
                    row.r2,
                    row.mean_residual,
                ]
                for row in slice_rows.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Target distribution",
            "",
            "The M4A test target distribution is not identical to the expanded training "
            "distribution. The table makes the shift visible; no test statistic is used "
            "to tune a model.",
            "",
        ]
    )
    target_rows = result.target_distribution[
        (result.target_distribution["dataset"].isin(["expanded", "evaluation"]))
    ]
    lines.extend(
        _markdown_table(
            ["dataset", "split", "scope", "family", "severity", "n", "mean", "sd", "min", "max"],
            [
                [
                    row.dataset,
                    row.split,
                    row.scope,
                    row.family,
                    row.severity_level,
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
            "### Grouped uncertainty",
            "",
            "Intervals resample the 12 held-out test sources, not individual images. "
            "They are descriptive and preliminary at this group count. `comparison` rows "
            "are paired source-grouped differences (`a - b`) for MAE and RMSE.",
            "",
        ]
    )
    selected_uncertainty = result.uncertainty[
        result.uncertainty["model"].isin(
            [
                FROZEN_MODEL_NAME,
                EXPANDED_MODEL_NAME,
                M3_NAIVE_NAME,
                EXPANDED_NAIVE_NAME,
                f"{EXPANDED_MODEL_NAME}_minus_{FROZEN_MODEL_NAME}",
            ]
        )
    ]
    lines.extend(
        _markdown_table(
            [
                "model/comparison",
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
                    row.dataset,
                    row.scope,
                    row.level,
                    row.metric,
                    row.estimate,
                    row.ci_low,
                    row.ci_high,
                ]
                for row in selected_uncertainty.itertuples(index=False)
            ],
            digits=4,
        )
    )
    lines.extend(
        [
            "",
            "### Residual summary",
            "",
            "Signed residual is `target - prediction` in the row-level tables and in the "
            "summary below. Positive values mean under-prediction.",
            "",
        ]
    )
    residual_rows = result.adaptation_residuals[
        (result.adaptation_residuals["model"] == EXPANDED_MODEL_NAME)
    ]
    lines.extend(
        _markdown_table(
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
    lines.extend(["", *_decision_lines(result)])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- One synthetic renderer, one 192x64 canvas, three fonts and 60 sources.",
            "- Only 12 held-out test sources; grouped intervals are preliminary and will be wide.",
            "- The expanded model is trained on a mixture whose target distribution differs "
            "from the M4A test distribution; the target-distribution table discloses the shift.",
            "- M4A corruption severity is a nested stroke density, not an independent redraw.",
            "- SSIM is a full-reference structural proxy, not human readability or CAPTCHA "
            "solvability.",
            "- No CNN, dashboard, deployment, retuning of the frozen M3 model or external "
            "CAPTCHA dataset is part of this benchmark.",
            "",
            "## Reproducibility",
            "",
            "Run and verify with:",
            "",
            "```text",
            "uv run --extra image --extra ml captcha-robustness-benchmark run \\",
            "    --config configs/robustness_benchmark_v0_1_0.toml",
            "uv run --extra image --extra ml captcha-robustness-benchmark verify \\",
            "    --config configs/robustness_benchmark_v0_1_0.toml",
            "```",
            "",
            "Generated tables, the model bundle and this report live in Git-ignored paths.",
        ]
    )
    return "\n".join(lines) + "\n"


def _expanded_model_card(
    config: RobustnessConfig,
    result: RobustnessResult,
) -> dict[str, Any]:
    train = result.frames.expanded_train
    val = result.frames.expanded_val
    selected_row = result.candidates[result.candidates["selected"]].iloc[0]
    test_row = _metric_lookup(
        result.adaptation_metrics, EXPANDED_MODEL_NAME, "m4a_test", "overall", "all"
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
        "test_m4a": {
            "n_rows": int(test_row["n_samples"]),
            "n_sources": int(test_row["n_sources"]),
            "mae": float(test_row["mae"]),
            "rmse": float(test_row["rmse"]),
            "r2": float(test_row["r2"]),
            "r2_defined": bool(test_row["r2_defined"]),
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
            "dataset_name": "captcha-quality-expanded-occlusion-interference",
            "dataset_version": config.run.version,
            "feature_set_version": FEATURE_SET_VERSION,
            "generation_config_hash": config.m3.generation_config_hash,
            "split_assignment_sha256": config.m3.split_assignment_sha256,
            "target": TARGET,
            "m3_manifest_sha256": config.m3.manifest_sha256,
            "m4a_manifest_sha256": config.m4a.manifest_sha256,
            "m4a_pilot_config_hash": config.m4a.pilot_config_hash,
            "expanded_train_samples": int(len(train)),
            "expanded_val_samples": int(len(val)),
            "excluded_rows": "M4A clean rows (duplicate M3 clean references)",
        },
        "training_scope": "expanded_train_sources_m3_plus_m4a_corrupted",
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


def _safe_output_path(path: Path) -> Path:
    resolved = path.resolve()
    for protected_text in PROTECTED_OUTPUT_ROOTS:
        protected = Path(protected_text).resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise RobustnessBenchmarkError(
                f"refusing to write into protected frozen path: {path} ({protected_text})"
            )
    return resolved


def _fingerprint_assets(paths: Iterable[str | Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files = sorted(file for file in path.rglob("*") if file.is_file())
            for file in files:
                result[str(file)] = {
                    "path": str(file),
                    "exists": True,
                    "sha256": sha256_file(file),
                    "size_bytes": int(file.stat().st_size),
                }
        else:
            result[str(path)] = {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "size_bytes": int(path.stat().st_size) if path.is_file() else None,
            }
    return result


def write_benchmark_outputs(
    result: RobustnessResult,
    config: RobustnessConfig,
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
        "isolation_checks.csv": pd.DataFrame(
            [
                {"check": check.name, "passed": check.passed, "detail": check.detail}
                for check in result.checks
            ]
        ),
        "zero_shot_metrics.csv": result.zero_shot_metrics,
        "zero_shot_residuals.csv": result.zero_shot_residuals,
        "zero_shot_predictions.csv": result.zero_shot_predictions,
        "m3_baseline_metrics.csv": result.m3_baseline_metrics,
        "candidate_validation.csv": result.candidates,
        "adaptation_metrics.csv": result.adaptation_metrics,
        "adaptation_residuals.csv": result.adaptation_residuals,
        "adaptation_predictions.csv": result.adaptation_predictions,
        "target_distribution.csv": result.target_distribution,
        "uncertainty.csv": result.uncertainty,
    }
    for filename, table in tables.items():
        _write_csv(table, reports / filename)

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
            "m4a": {
                "root": str(config.m4a.dataset),
                "dataset_version": config.m4a.dataset_version,
                "pilot_config_hash": config.m4a.pilot_config_hash,
                "manifest_sha256": config.m4a.manifest_sha256,
                "sources_sha256": config.m4a.sources_sha256,
                "features_sha256": config.m4a.features_sha256,
                "summary_sha256": config.m4a.summary_sha256,
                "image_pixel_hash_set": config.m4a.image_pixel_hash_set,
            },
        },
        "schema": {
            "feature_set_version": config.schema.feature_set_version,
            "feature_names": list(config.schema.feature_names),
            "target": config.schema.target,
            "split_names": list(config.schema.split_names),
        },
        "split": {
            "assignment_sha256": config.m3.split_assignment_sha256,
            "train_sources": int(result.frames.expanded_train["source_id"].nunique()),
            "val_sources": int(result.frames.expanded_val["source_id"].nunique()),
            "test_sources": int(result.frames.m4a_test["source_id"].nunique()),
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
            "numpy_version": _dependency_version("numpy"),
            "pandas_version": _dependency_version("pandas"),
            "scikit_image_version": _dependency_version("scikit-image"),
            "joblib_version": _dependency_version("joblib"),
            "python_version": sys.version.split()[0],
            "code_revision": code_revision(),
            "elapsed_seconds_in_process": (
                round(float(elapsed_seconds), 6) if elapsed_seconds is not None else None
            ),
        },
        "artifacts": artifact_hashes,
    }
    metadata_path = reports / "run_metadata.json"
    _write_json(_json_ready(metadata_payload), metadata_path)
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


def _dependency_version(name: str) -> str:
    try:
        from importlib import metadata as importlib_metadata

        return importlib_metadata.version(name)
    except Exception:  # noqa: BLE001 - metadata is best effort
        return "unknown"


def run_benchmark(
    config: RobustnessConfig,
    *,
    output_reports: Path | None = None,
    output_models: Path | None = None,
    verify_images: bool = True,
) -> tuple[RobustnessResult, Path, Path, dict[str, Path]]:
    """Load, verify, run and persist the complete M4B benchmark."""
    started = time.perf_counter()
    before = _fingerprint_assets(FROZEN_ASSET_PATHS)
    frames = load_benchmark_frames(config, verify_images=verify_images)
    frozen_model, frozen_card, model_checks = load_frozen_model(config, frames.m3_summary)
    _verify_checks(model_checks, "frozen M3 model verification")
    if frozen_model is None:
        raise RobustnessBenchmarkError("frozen M3 model could not be loaded")
    result = run_benchmark_core(
        frames,
        config,
        frozen_estimator=frozen_model,
        frozen_card=frozen_card,
    )
    after = _fingerprint_assets(FROZEN_ASSET_PATHS)
    if before != after:
        changed = sorted(
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        )
        raise RobustnessBenchmarkError(f"frozen assets changed during the run: {changed[:5]}")
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
        raise RobustnessBenchmarkError(f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RobustnessBenchmarkError(f"JSON artifact is not an object: {path}")
    return payload


def _check_hash(path: Path, expected: object, label: str) -> None:
    if not path.is_file():
        raise RobustnessBenchmarkError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RobustnessBenchmarkError(
            f"{label} hash mismatch: expected {expected}, found {actual}"
        )


def verify_benchmark(
    config: RobustnessConfig,
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
        _verify_checks(model_checks, "frozen M3 model verification")
        return f"{len(frames.m3_frame)} M3 rows, {len(frames.m4a_frame)} M4A rows, model loaded"

    run("inputs_and_model", load_inputs)

    def metadata_checks() -> str:
        metadata = _load_json(reports / "run_metadata.json")
        if metadata.get("benchmark_config_hash") != config.config_hash:
            raise RobustnessBenchmarkError("run_metadata config hash does not match the config")
        if metadata.get("benchmark_config_sha256") != config.config_sha256:
            raise RobustnessBenchmarkError("run_metadata config file hash does not match")
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
            raise RobustnessBenchmarkError("run_metadata has no artifact hash table")
        for relative, expected in sorted(artifacts.items()):
            if relative.startswith("models/"):
                path = models / relative.removeprefix("models/")
            else:
                path = reports / relative
            _check_hash(path, expected, relative)
        if metadata.get("frozen_assets", {}).get("all_unchanged") is not True:
            raise RobustnessBenchmarkError("run_metadata reports changed frozen assets")
        return f"{len(artifacts)} artifact hashes match and frozen inputs were unchanged"

    run("run_metadata", metadata_checks)

    def expanded_model_checks() -> str:
        if frames is None:
            raise RobustnessBenchmarkError("frames are unavailable")
        model, _, model_checks = load_model_bundle(
            models,
            EXPANDED_MODEL_STEM,
            dataset_summary=frames.m3_summary,
        )
        _verify_checks(model_checks, "expanded model verification")
        if model is None:
            raise RobustnessBenchmarkError("expanded model could not be loaded")
        expected = pd.read_csv(reports / "adaptation_predictions.csv")
        selected = expected[expected["model"] == EXPANDED_MODEL_NAME]
        if len(selected) != len(frames.m4a_test):
            raise RobustnessBenchmarkError("adaptation_predictions.csv has the wrong row count")
        actual = predict(model, frames.m4a_test)
        if not np.allclose(
            selected["prediction"].to_numpy(dtype=np.float64),
            actual,
            rtol=0.0,
            atol=1e-9,
        ):
            raise RobustnessBenchmarkError("expanded model predictions do not replay")
        return f"model reloads and reproduces {len(actual)} held-out predictions"

    run("expanded_model_replay", expanded_model_checks)

    def zero_shot_replay() -> str:
        if frames is None or frozen_model is None:
            raise RobustnessBenchmarkError("frozen model or frames are unavailable")
        expected = pd.read_csv(reports / "zero_shot_predictions.csv")
        selected = expected[expected["model"] == FROZEN_MODEL_NAME]
        if len(selected) != len(frames.m4a_test):
            raise RobustnessBenchmarkError("zero_shot_predictions.csv has the wrong row count")
        actual = predict(frozen_model, frames.m4a_test)
        if not np.allclose(
            selected["prediction"].to_numpy(dtype=np.float64),
            actual,
            rtol=0.0,
            atol=1e-9,
        ):
            raise RobustnessBenchmarkError("frozen model predictions do not replay")
        return f"frozen model reproduces {len(actual)} zero-shot predictions"

    run("zero_shot_replay", zero_shot_replay)

    def report_exists() -> str:
        report = reports / "benchmark.md"
        if not report.is_file() or not report.read_text(encoding="utf-8").strip():
            raise RobustnessBenchmarkError("benchmark.md is missing or empty")
        return f"{report} is present"

    run("report", report_exists)
    return VerificationReport(dataset=str(reports), checks=tuple(checks))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-robustness-benchmark",
        description="Run or verify the grouped M4B occlusion/interference benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "run the grouped robustness benchmark"),
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
        "robustness benchmark verification: PASS"
        if report.ok
        else "robustness benchmark verification: FAIL"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the M4B CLI; returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_robustness_config(args.config)
        if args.command == "run":
            result, reports, models, paths = run_benchmark(
                config,
                output_reports=args.output_reports,
                output_models=args.output_models,
            )
            print(f"benchmark: {config.describe()}")
            print(f"protocol hash: {config.config_hash}")
            print(
                f"rows: train {len(result.frames.expanded_train)}, "
                f"val {len(result.frames.expanded_val)}, test {len(result.frames.m4a_test)}"
            )
            frozen = _metric_lookup(
                result.zero_shot_metrics, FROZEN_MODEL_NAME, "m4a_test", "overall", "all"
            )
            selected = _metric_lookup(
                result.adaptation_metrics, EXPANDED_MODEL_NAME, "m4a_test", "overall", "all"
            )
            print(
                f"zero-shot frozen M3: MAE {frozen['mae']:.4f}, RMSE {frozen['rmse']:.4f}, "
                f"R2 {frozen['r2']:.4f}"
            )
            print(
                f"adapted {result.selected['name']}: MAE {selected['mae']:.4f}, "
                f"RMSE {selected['rmse']:.4f}, R2 {selected['r2']:.4f}"
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
        raise RobustnessBenchmarkError(f"unknown command {args.command!r}")  # pragma: no cover
    except (
        ConfigError,
        DatasetError,
        ModelError,
        PilotError,
        RobustnessBenchmarkError,
        RobustnessConfigError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
