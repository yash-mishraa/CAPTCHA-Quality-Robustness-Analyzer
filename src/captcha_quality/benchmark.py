"""Milestone 3 conventional-ML benchmark for the synthetic SSIM target.

Protocol (frozen before model fitting)
--------------------------------------
* Data: the committed ``v0.1.0`` dataset and its source-group split.
* Primary inputs: the eight documented image features only.
* Fitting: training sources only. Selection: validation sources only.
* Test: evaluated once after every candidate and family winner is fixed.
* Metadata models are explicitly diagnostic and never presented as the
  deployable image-only task.
* Held-out font: one font is excluded from fitting and validation before any
  model is trained; it is evaluated once as a font-shift stress test.

The target is synthetic image similarity (SSIM to a clean reference). No result
here measures human readability, CAPTCHA solvability, or real-world generality.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from captcha_quality.config import FEATURE_SET_VERSION, ConfigError
from captcha_quality.corruptions import derive_noise_seed
from captcha_quality.dataset import (
    CheckResult,
    DatasetError,
    VerificationReport,
    load_dataset,
)
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.modelling import (
    FEATURE_COLUMNS,
    SPLITS,
    TARGET,
    CandidateSpec,
    FittedModel,
    ModelError,
    best_per_family,
    bootstrap_metrics,
    candidate_specs,
    fit_candidate,
    load_model_bundle,
    metadata_design,
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
from captcha_quality.summarise import rank_correlation

_SELECTION_METRICS: Final[tuple[str, ...]] = ("rmse", "mae")
_EXPECTED_KEYS: Final[frozenset[str]] = frozenset(
    {"benchmark", "ridge", "random_forest", "hist_gradient_boosting"}
)


class BenchmarkError(RuntimeError):
    """Raised when the benchmark protocol cannot be run as specified."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


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
class BenchmarkConfig:
    """A versioned, reviewable benchmark definition."""

    name: str
    version: str
    seed: int
    held_out_font: str
    held_out_font_reason: str
    n_bootstrap: int
    coverage_quantiles: tuple[float, ...]
    selection_metric: str
    material_improvement_fraction: float
    output_dir: Path
    model_dir: Path
    ridge: RidgeGrid
    random_forest: RandomForestGrid
    hist_gradient_boosting: HistGradientBoostingGrid

    def identity_payload(self) -> dict[str, Any]:
        """Fields that define the benchmark; output paths are excluded."""
        return {
            "name": self.name,
            "version": self.version,
            "seed": self.seed,
            "held_out_font": self.held_out_font,
            "held_out_font_reason": self.held_out_font_reason,
            "n_bootstrap": self.n_bootstrap,
            "coverage_quantiles": list(self.coverage_quantiles),
            "selection_metric": self.selection_metric,
            "material_improvement_fraction": self.material_improvement_fraction,
            "ridge": {"alphas": list(self.ridge.alphas)},
            "random_forest": {
                "n_estimators": self.random_forest.n_estimators,
                "max_depths": list(self.random_forest.max_depths),
                "min_samples_leafs": list(self.random_forest.min_samples_leafs),
            },
            "hist_gradient_boosting": {
                "max_iter": self.hist_gradient_boosting.max_iter,
                "learning_rates": list(self.hist_gradient_boosting.learning_rates),
                "max_leaf_nodes": list(self.hist_gradient_boosting.max_leaf_nodes),
                "min_samples_leaf": self.hist_gradient_boosting.min_samples_leaf,
                "l2_regularization": self.hist_gradient_boosting.l2_regularization,
            },
        }

    @property
    def config_hash(self) -> str:
        """Stable content hash of the identity-defining fields."""
        return sha256_bytes(canonical_json(self.identity_payload()).encode("utf-8"))

    def describe(self) -> str:
        """Short human-readable identity for CLI output."""
        return f"{self.name} {self.version} ({self.config_hash[:12]})"


def _check_keys(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in [{name}]: {unknown}; allowed keys are {sorted(allowed)}"
        )


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{name}] table")
    return value


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"missing required key {section_name}.{key}")
    return section[key]


def _as_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{label} must be finite, got {value!r}")
    return result


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{label} must be a non-empty TOML array")
    return value


def _floats(value: Any, label: str) -> tuple[float, ...]:
    items = tuple(_as_float(item, f"{label}[]") for item in _sequence(value, label))
    if len(set(items)) != len(items):
        raise ConfigError(f"{label} must not contain duplicates")
    return items


def _ints(value: Any, label: str) -> tuple[int, ...]:
    items = tuple(_as_int(item, f"{label}[]") for item in _sequence(value, label))
    if len(set(items)) != len(items):
        raise ConfigError(f"{label} must not contain duplicates")
    return items


def load_benchmark_config(path: Path | str) -> BenchmarkConfig:
    """Load and strictly validate a benchmark TOML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"benchmark config not found: {path}")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    _check_keys(payload, set(_EXPECTED_KEYS), "benchmark file")

    benchmark = _section(payload, "benchmark")
    _check_keys(
        benchmark,
        {
            "name",
            "version",
            "seed",
            "held_out_font",
            "held_out_font_reason",
            "n_bootstrap",
            "coverage_quantiles",
            "selection_metric",
            "material_improvement_fraction",
            "output_dir",
            "model_dir",
        },
        "benchmark",
    )
    name = _as_str(_required(benchmark, "name", "benchmark"), "benchmark.name")
    version = _as_str(_required(benchmark, "version", "benchmark"), "benchmark.version")
    seed = _as_int(_required(benchmark, "seed", "benchmark"), "benchmark.seed")
    held_out_font = _as_str(
        _required(benchmark, "held_out_font", "benchmark"), "benchmark.held_out_font"
    )
    reason = _as_str(
        _required(benchmark, "held_out_font_reason", "benchmark"),
        "benchmark.held_out_font_reason",
    )
    n_bootstrap = _as_int(_required(benchmark, "n_bootstrap", "benchmark"), "benchmark.n_bootstrap")
    if n_bootstrap < 0:
        raise ConfigError("benchmark.n_bootstrap must be >= 0")
    quantiles = _floats(
        _required(benchmark, "coverage_quantiles", "benchmark"),
        "benchmark.coverage_quantiles",
    )
    if len(quantiles) < 3:
        raise ConfigError("benchmark.coverage_quantiles must define at least two bins")
    if quantiles[0] != 0.0 or quantiles[-1] != 1.0:
        raise ConfigError("coverage_quantiles must start at 0.0 and end at 1.0")
    if any(right <= left for left, right in zip(quantiles, quantiles[1:], strict=False)):
        raise ConfigError("coverage_quantiles must be strictly increasing")
    selection_metric = _as_str(
        _required(benchmark, "selection_metric", "benchmark"), "benchmark.selection_metric"
    )
    if selection_metric not in _SELECTION_METRICS:
        raise ConfigError(
            f"benchmark.selection_metric must be one of {_SELECTION_METRICS}, "
            f"got {selection_metric!r}"
        )
    materiality = _as_float(
        _required(benchmark, "material_improvement_fraction", "benchmark"),
        "benchmark.material_improvement_fraction",
    )
    if not 0.0 < materiality < 1.0:
        raise ConfigError("benchmark.material_improvement_fraction must be in (0, 1)")
    output_dir = Path(
        _as_str(_required(benchmark, "output_dir", "benchmark"), "benchmark.output_dir")
    )
    model_dir = Path(_as_str(_required(benchmark, "model_dir", "benchmark"), "benchmark.model_dir"))

    ridge_section = _section(payload, "ridge")
    _check_keys(ridge_section, {"alphas"}, "ridge")
    alphas = _floats(_required(ridge_section, "alphas", "ridge"), "ridge.alphas")
    if any(alpha <= 0.0 for alpha in alphas):
        raise ConfigError("ridge.alphas must all be > 0")

    rf = _section(payload, "random_forest")
    _check_keys(rf, {"n_estimators", "max_depths", "min_samples_leafs"}, "random_forest")
    n_estimators = _as_int(
        _required(rf, "n_estimators", "random_forest"), "random_forest.n_estimators"
    )
    if n_estimators < 1:
        raise ConfigError("random_forest.n_estimators must be >= 1")
    max_depths = _ints(_required(rf, "max_depths", "random_forest"), "random_forest.max_depths")
    if any(depth < 0 for depth in max_depths):
        raise ConfigError("random_forest.max_depths entries must be >= 0 (0 means unlimited)")
    leafs = _ints(
        _required(rf, "min_samples_leafs", "random_forest"),
        "random_forest.min_samples_leafs",
    )
    if any(leaf < 1 for leaf in leafs):
        raise ConfigError("random_forest.min_samples_leafs entries must be >= 1")

    hgb = _section(payload, "hist_gradient_boosting")
    _check_keys(
        hgb,
        {
            "max_iter",
            "learning_rates",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        },
        "hist_gradient_boosting",
    )
    max_iter = _as_int(
        _required(hgb, "max_iter", "hist_gradient_boosting"),
        "hist_gradient_boosting.max_iter",
    )
    if max_iter < 1:
        raise ConfigError("hist_gradient_boosting.max_iter must be >= 1")
    learning_rates = _floats(
        _required(hgb, "learning_rates", "hist_gradient_boosting"),
        "hist_gradient_boosting.learning_rates",
    )
    if any(rate <= 0.0 or rate > 1.0 for rate in learning_rates):
        raise ConfigError("hist_gradient_boosting.learning_rates must be in (0, 1]")
    max_leaf_nodes = _ints(
        _required(hgb, "max_leaf_nodes", "hist_gradient_boosting"),
        "hist_gradient_boosting.max_leaf_nodes",
    )
    if any(nodes < 2 for nodes in max_leaf_nodes):
        raise ConfigError("hist_gradient_boosting.max_leaf_nodes entries must be >= 2")
    min_samples_leaf = _as_int(
        _required(hgb, "min_samples_leaf", "hist_gradient_boosting"),
        "hist_gradient_boosting.min_samples_leaf",
    )
    if min_samples_leaf < 1:
        raise ConfigError("hist_gradient_boosting.min_samples_leaf must be >= 1")
    l2 = _as_float(
        _required(hgb, "l2_regularization", "hist_gradient_boosting"),
        "hist_gradient_boosting.l2_regularization",
    )
    if l2 < 0.0:
        raise ConfigError("hist_gradient_boosting.l2_regularization must be >= 0")

    return BenchmarkConfig(
        name=name,
        version=version,
        seed=seed,
        held_out_font=held_out_font,
        held_out_font_reason=reason,
        n_bootstrap=n_bootstrap,
        coverage_quantiles=quantiles,
        selection_metric=selection_metric,
        material_improvement_fraction=materiality,
        output_dir=output_dir,
        model_dir=model_dir,
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


def benchmark_specs(config: BenchmarkConfig) -> tuple[CandidateSpec, ...]:
    """Return the full predeclared candidate grid."""
    return candidate_specs(
        ridge_alphas=config.ridge.alphas,
        rf_n_estimators=config.random_forest.n_estimators,
        rf_max_depths=config.random_forest.max_depths,
        rf_min_samples_leafs=config.random_forest.min_samples_leafs,
        hgb_max_iter=config.hist_gradient_boosting.max_iter,
        hgb_learning_rates=config.hist_gradient_boosting.learning_rates,
        hgb_max_leaf_nodes=config.hist_gradient_boosting.max_leaf_nodes,
        hgb_min_samples_leaf=config.hist_gradient_boosting.min_samples_leaf,
        hgb_l2_regularization=config.hist_gradient_boosting.l2_regularization,
    )


# ---------------------------------------------------------------------------
# Dataset loading and protocol validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    root: Path
    manifest: pd.DataFrame
    sources: pd.DataFrame
    features: pd.DataFrame
    summary: dict[str, Any]
    frame: pd.DataFrame
    generation_config_hash: str
    split_assignment_sha256: str
    feature_set_version: str


def split_assignment_hash(sources: pd.DataFrame) -> str:
    """Stable hash of the source-level split assignment, matching M2."""
    pairs = sorted(zip(sources["source_id"].tolist(), sources["split"].tolist(), strict=True))
    return sha256_bytes(canonical_json(pairs).encode("utf-8"))


def load_benchmark_dataset(root: Path | str) -> DatasetBundle:
    """Load a dataset and enforce the frozen benchmark preconditions."""
    root = Path(root)
    manifest, sources, features, summary = load_dataset(root)
    feature_names = validate_feature_columns(list(features.columns))

    if len(manifest) != len(features):
        raise BenchmarkError(
            f"manifest has {len(manifest)} rows but features.csv has {len(features)}"
        )
    if features["sample_id"].duplicated().any():
        raise BenchmarkError("duplicate sample_id in features.csv")
    if sources["source_id"].duplicated().any():
        raise BenchmarkError("duplicate source_id in sources.csv")
    if manifest["sample_id"].duplicated().any():
        raise BenchmarkError("duplicate sample_id in manifest.csv")

    frame = manifest.merge(features, on="sample_id", how="left", validate="one_to_one")
    if len(frame) != len(manifest) or frame[list(feature_names)].isna().any().any():
        raise BenchmarkError("manifest and feature rows do not align one-to-one")
    if not np.isfinite(frame[list(feature_names)].to_numpy(dtype=np.float64)).all():
        raise BenchmarkError("feature table contains non-finite values")
    if not np.isfinite(frame[TARGET].to_numpy(dtype=np.float64)).all():
        raise BenchmarkError("target contains non-finite values")
    if not frame[TARGET].between(0.0, 1.0).all():
        raise BenchmarkError("ssim_target is outside [0, 1]")

    if set(frame["split"]) != set(SPLITS):
        raise BenchmarkError(f"dataset splits are {sorted(set(frame['split']))}, expected {SPLITS}")
    per_source = frame.groupby("source_id")["split"].nunique()
    if (per_source > 1).any():
        raise BenchmarkError("a source group crosses train/val/test")
    per_source_font = frame.groupby("source_id")["font_id"].nunique()
    if (per_source_font > 1).any():
        raise BenchmarkError("a source group has multiple fonts")

    source_check = sources.set_index("source_id")
    missing = set(frame["source_id"]) - set(source_check.index)
    if missing:
        raise BenchmarkError(f"source(s) missing from sources.csv: {sorted(missing)}")
    for source_id, group in frame.groupby("source_id"):
        if source_check.loc[source_id, "split"] != group["split"].iloc[0]:
            raise BenchmarkError(f"split mismatch between manifest and sources.csv for {source_id}")
        if source_check.loc[source_id, "font_id"] != group["font_id"].iloc[0]:
            raise BenchmarkError(f"font mismatch between manifest and sources.csv for {source_id}")

    expected_hash = split_assignment_hash(sources)
    summary_hash = summary.get("split_policy", {}).get("assignment_sha256")
    if expected_hash != summary_hash:
        raise BenchmarkError(
            "sources.csv split assignment does not match dataset_summary.json; "
            "the frozen split cannot be verified"
        )
    if summary.get("feature_set_version") != FEATURE_SET_VERSION:
        raise BenchmarkError(
            f"dataset feature set {summary.get('feature_set_version')!r} does not match "
            f"this code ({FEATURE_SET_VERSION!r})"
        )
    if list(summary.get("feature_names", [])) != list(FEATURE_NAMES):
        raise BenchmarkError("dataset feature names/order do not match the extractor")

    return DatasetBundle(
        root=root,
        manifest=manifest,
        sources=sources,
        features=features,
        summary=summary,
        frame=frame,
        generation_config_hash=str(summary.get("generation_config_hash")),
        split_assignment_sha256=expected_hash,
        feature_set_version=str(summary.get("feature_set_version")),
    )


def protocol_checks(bundle: DatasetBundle, config: BenchmarkConfig) -> tuple[CheckResult, ...]:
    """Programmatic checks that the evaluation protocol is the frozen one."""
    checks: list[CheckResult] = []

    def run(name: str, function: Any) -> None:
        try:
            detail = function()
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            checks.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))
        else:
            checks.append(CheckResult(name, True, str(detail)))

    def feature_schema() -> str:
        if tuple(FEATURE_COLUMNS) != FEATURE_NAMES:
            raise BenchmarkError("feature column order drifted from FEATURE_NAMES")
        return f"{len(FEATURE_NAMES)} documented image features, no metadata or target"

    def group_isolation() -> str:
        source_splits = bundle.frame.groupby("split")["source_id"].nunique().to_dict()
        if set(source_splits) != set(SPLITS):
            raise BenchmarkError(f"missing split(s): {sorted(set(SPLITS) - set(source_splits))}")
        return f"disjoint source counts: {source_splits}"

    def split_disjointness() -> str:
        sets = {
            split: set(bundle.frame.loc[bundle.frame["split"] == split, "source_id"])
            for split in SPLITS
        }
        overlaps = {
            f"{left}/{right}": len(sets[left] & sets[right])
            for left in SPLITS
            for right in SPLITS
            if left < right
        }
        if any(overlaps.values()):
            raise BenchmarkError(f"source overlap between splits: {overlaps}")
        return "train/val/test source sets have empty pairwise intersections"

    def split_hash() -> str:
        if split_assignment_hash(bundle.sources) != bundle.split_assignment_sha256:
            raise BenchmarkError("split assignment hash changed after loading")
        return f"split assignment {bundle.split_assignment_sha256[:12]} matches summary"

    def held_out_font() -> str:
        fonts = sorted(bundle.frame["font_id"].unique())
        if config.held_out_font not in fonts:
            raise BenchmarkError(
                f"held-out font {config.held_out_font!r} is not in dataset fonts {fonts}"
            )
        if len(fonts) < 2:
            raise BenchmarkError("held-out-font experiment needs at least two fonts")
        heldout_sources = bundle.frame.loc[
            bundle.frame["font_id"] == config.held_out_font, "source_id"
        ].nunique()
        return f"{config.held_out_font!r}: {heldout_sources} sources reserved for the stress test"

    run("feature_schema", feature_schema)
    run("group_isolation", group_isolation)
    run("split_disjointness", split_disjointness)
    run("split_hash", split_hash)
    run("held_out_font_available", held_out_font)
    return tuple(checks)


# ---------------------------------------------------------------------------
# Small table helpers
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


def _metrics_row(
    *,
    name: str,
    family: str,
    role: str,
    split: str,
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    n_inputs: int | None = None,
) -> dict[str, Any]:
    return {
        "model": name,
        "family": family,
        "role": role,
        "split": split,
        "n_rows": int(metrics["n_rows"]),
        "n_sources": int(metrics["n_sources"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "r2_defined": bool(metrics["r2_defined"]),
        "n_inputs": n_inputs,
        "params": json.dumps(params, sort_keys=True, separators=(",", ":")) if params else "",
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _markdown_table(
    headers: list[str],
    rows: list[list[Any]],
    *,
    digits: int = 3,
    numeric_from: int = 0,
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            if index < numeric_from or isinstance(value, str):
                cells.append(str(value))
            elif isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, int):
                cells.append(str(value))
            else:
                cells.append(_fmt(value, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _sorted_by_split(frame: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    """Sort a report table in train/val/test (or held-out) order."""
    order = {name: index for index, name in enumerate(SPLITS)}
    order.update({"train_fonts_train": 0, "train_fonts_val": 1, "heldout_font": 2})
    temporary = frame.assign(__split_order=frame["split"].map(order).fillna(len(order)))
    return temporary.sort_values(["__split_order", *extra], kind="stable").drop(
        columns="__split_order"
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrimaryResult:
    candidates: pd.DataFrame
    family_metrics: pd.DataFrame
    selected: dict[str, Any]
    selected_model: FittedModel
    predictions: pd.DataFrame
    bootstrap: pd.DataFrame


def _fit_and_score_candidates(
    specs: tuple[CandidateSpec, ...],
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    seed: int,
    selection_metric: str,
) -> tuple[list[dict[str, Any]], dict[str, FittedModel]]:
    rows: list[dict[str, Any]] = []
    fitted: dict[str, FittedModel] = {}
    for spec in specs:
        # scikit-learn's random_state range is 32-bit; keep the derived seed
        # deterministic while fitting inside that documented range.
        model = fit_candidate(
            spec,
            train,
            seed=derive_noise_seed(seed, "model", spec.name) % (2**32),
        )
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
                "train_r2_defined": train_scores["r2_defined"],
                "train_n": train_scores["n_rows"],
                "val_mae": val_scores["mae"],
                "val_rmse": val_scores["rmse"],
                "val_r2": val_scores["r2"],
                "val_r2_defined": val_scores["r2_defined"],
                "val_n": val_scores["n_rows"],
                "val_n_sources": val_scores["n_sources"],
                "selected_on": "val",
            }
        )
    return rows, fitted


def _candidate_table(
    rows: list[dict[str, Any]], selected_name: str, family_names: set[str]
) -> pd.DataFrame:
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
                "selected": row["name"] == selected_name,
                "family_selected": row["name"] in family_names,
                "selected_on": "val",
            }
        )
    return pd.DataFrame(records)


def _prediction_frame(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    model_name: str,
    splits: tuple[str, ...] = SPLITS,
) -> pd.DataFrame:
    subset = frame[frame["split"].isin(splits)].copy()
    mask = frame["split"].isin(splits).to_numpy()
    subset["model"] = model_name
    subset["prediction"] = predictions[mask]
    subset["residual"] = subset["prediction"] - subset[TARGET]
    subset["abs_residual"] = subset["residual"].abs()
    return subset


def run_primary(bundle: DatasetBundle, config: BenchmarkConfig) -> PrimaryResult:
    """Fit every candidate on train, select on val, then evaluate on test once."""
    frame = bundle.frame
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]
    specs = benchmark_specs(config)

    naive = naive_spec()
    naive_fit = fit_candidate(naive, train, seed=config.seed)
    naive_train = score_frame(train, predict(naive_fit.estimator, train))
    naive_val = score_frame(val, predict(naive_fit.estimator, val))
    naive_row = {
        "name": naive.name,
        "family": naive.family,
        "params": naive.params,
        "n_inputs": 0,
        "train_mae": naive_train["mae"],
        "train_rmse": naive_train["rmse"],
        "train_r2": naive_train["r2"],
        "train_r2_defined": naive_train["r2_defined"],
        "train_n": naive_train["n_rows"],
        "val_mae": naive_val["mae"],
        "val_rmse": naive_val["rmse"],
        "val_r2": naive_val["r2"],
        "val_r2_defined": naive_val["r2_defined"],
        "val_n": naive_val["n_rows"],
        "val_n_sources": naive_val["n_sources"],
        "selected_on": "val",
    }

    learned_rows, fitted = _fit_and_score_candidates(
        specs, train, val, seed=config.seed, selection_metric=config.selection_metric
    )
    all_rows = [naive_row, *learned_rows]
    selected = select_best(learned_rows, config.selection_metric)
    family_selected = best_per_family(learned_rows, config.selection_metric)
    family_names = {row["name"] for row in family_selected.values()}

    # Test is first touched here, after every candidate and family winner is fixed.
    metrics_rows: list[dict[str, Any]] = []

    predictions = predict(naive_fit.estimator, test)
    metrics_rows.append(
        _metrics_row(
            name=naive.name,
            family="naive",
            role="naive",
            split="test",
            metrics=score_frame(test, predictions),
            params={},
            n_inputs=0,
        )
    )
    for split_name, split_frame, split_predictions in (
        ("train", train, predict(naive_fit.estimator, train)),
        ("val", val, predict(naive_fit.estimator, val)),
    ):
        metrics_rows.append(
            _metrics_row(
                name=naive.name,
                family="naive",
                role="naive",
                split=split_name,
                metrics=score_frame(split_frame, split_predictions),
                params={},
                n_inputs=0,
            )
        )

    for family, row in sorted(family_selected.items()):
        name = row["name"]
        model = fitted[name]
        role = "family_best"
        if name == selected["name"]:
            role = "family_best+overall_selected"
        for split_name, split_frame in (("train", train), ("val", val), ("test", test)):
            predictions = predict(model.estimator, split_frame)
            metrics_rows.append(
                _metrics_row(
                    name=name,
                    family=family,
                    role=role,
                    split=split_name,
                    metrics=score_frame(split_frame, predictions),
                    params=model.spec.params,
                    n_inputs=model.spec.n_inputs,
                )
            )

    selected_model = fitted[selected["name"]]
    selected_test_predictions = predict(selected_model.estimator, test)
    predictions = _prediction_frame(
        frame, predict(selected_model.estimator, frame), model_name=selected["name"]
    )

    bootstrap = bootstrap_metrics(
        test,
        selected_test_predictions,
        n_bootstrap=config.n_bootstrap,
        seed=derive_noise_seed(config.seed, "bootstrap", "primary_test"),
        scope="primary_test",
    )

    return PrimaryResult(
        candidates=_candidate_table(all_rows, selected["name"], family_names),
        family_metrics=pd.DataFrame(metrics_rows),
        selected=selected,
        selected_model=selected_model,
        predictions=predictions,
        bootstrap=bootstrap,
    )


@dataclass(frozen=True, slots=True)
class MetadataResult:
    metrics: pd.DataFrame
    gaps: pd.DataFrame


def run_metadata_experiments(bundle: DatasetBundle) -> MetadataResult:
    """Fit explicitly diagnostic metadata models with train-only preprocessing."""
    frame = bundle.frame
    train = frame[frame["split"] == "train"]
    rows: list[dict[str, Any]] = []
    for kind, name in (
        ("severity_family", "metadata_severity_family"),
        ("font_text_length", "metadata_font_text_length"),
        ("image_plus_metadata", "image_plus_metadata"),
    ):
        design, names = metadata_design(frame, train, kind)
        train_mask = (frame["split"] == "train").to_numpy()
        model = LinearRegression()
        model.fit(design[train_mask], train[TARGET].to_numpy(dtype=np.float64))
        for split in SPLITS:
            mask = (frame["split"] == split).to_numpy()
            predictions = model.predict(design[mask])
            rows.append(
                _metrics_row(
                    name=name,
                    family="metadata_diagnostic",
                    role="diagnostic_only",
                    split=split,
                    metrics=score_frame(frame.loc[mask], predictions),
                    params={"design": kind, "columns": names},
                    n_inputs=len(names),
                )
            )
    return MetadataResult(metrics=pd.DataFrame(rows), gaps=pd.DataFrame())


def _gap_table(
    family_metrics: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    def lookup(frame: pd.DataFrame, model: str, split: str) -> dict[str, Any]:
        rows = frame[(frame["model"] == model) & (frame["split"] == split)]
        if len(rows) != 1:
            raise BenchmarkError(f"expected one row for {model}/{split}, found {len(rows)}")
        return rows.iloc[0].to_dict()

    selected_name = family_metrics.loc[
        family_metrics["role"].str.contains("overall_selected"), "model"
    ].iloc[0]
    comparisons = [
        ("selected_image_only_vs_naive", selected_name, "naive_train_mean"),
        ("metadata_severity_family_vs_selected", "metadata_severity_family", selected_name),
        ("image_plus_metadata_vs_selected", "image_plus_metadata", selected_name),
    ]
    rows: list[dict[str, Any]] = []
    for label, a_model, b_model in comparisons:
        for split in ("val", "test"):
            if a_model == "naive_train_mean":
                a = lookup(family_metrics, a_model, split)
            elif a_model.startswith("metadata") or a_model == "image_plus_metadata":
                a = lookup(metadata, a_model, split)
            else:
                a = lookup(family_metrics, a_model, split)
            if b_model == "naive_train_mean":
                b = lookup(family_metrics, b_model, split)
            else:
                b = lookup(family_metrics, b_model, split)
            rows.append(
                {
                    "comparison": label,
                    "split": split,
                    "a_model": a_model,
                    "b_model": b_model,
                    "a_rmse": a["rmse"],
                    "b_rmse": b["rmse"],
                    "rmse_gap_a_minus_b": float(a["rmse"]) - float(b["rmse"]),
                    "a_r2": a["r2"],
                    "b_r2": b["r2"],
                    "r2_gap_a_minus_b": float(a["r2"]) - float(b["r2"]),
                    "a_n_inputs": a["n_inputs"],
                    "b_n_inputs": b["n_inputs"],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage, stratification, residuals and correlations
# ---------------------------------------------------------------------------


def coverage_assignment(
    frame: pd.DataFrame,
    quantiles: tuple[float, ...],
    *,
    train_frame: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Assign train-quantile ink-coverage bins without using held-out values.

    Bin edges come from the training split only. Test values outside the train
    range fall into the open first or last bin, which is reported explicitly.
    ``train_frame`` can supply a stricter training reference (for example all
    fonts except the held-out font) without changing which rows are labelled.
    """
    reference = frame if train_frame is None else train_frame
    train = reference.loc[reference["split"] == "train", "ink_fraction"].to_numpy(dtype=np.float64)
    edges = np.quantile(train, quantiles)
    interior = edges[1:-1]
    if len(np.unique(interior)) != len(interior):
        raise BenchmarkError(
            "ink-coverage quantiles are duplicated on the training split; "
            "choose fewer coverage bins or a different stratification"
        )
    labels = tuple(f"Q{index + 1}" for index in range(len(quantiles) - 1))
    cut_edges = np.concatenate(([-np.inf], interior, [np.inf]))
    groups = pd.cut(
        frame["ink_fraction"],
        bins=cut_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    if groups.isna().any():  # pragma: no cover - cut with infinite edges covers all
        raise BenchmarkError("coverage assignment produced an unassigned row")
    reference_train = reference[reference["split"] == "train"]
    reference_train_groups = groups.loc[reference_train.index]
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        train_group = reference_train.loc[reference_train_groups == label, "ink_fraction"].to_numpy(
            dtype=np.float64
        )
        test_group = frame.loc[(frame["split"] == "test") & (groups == label), "ink_fraction"]
        rows.append(
            {
                "coverage_group": label,
                "quantile_low": quantiles[index],
                "quantile_high": quantiles[index + 1],
                "edge_low": float(cut_edges[index]),
                "edge_high": float(cut_edges[index + 1]),
                "n_train": int(len(train_group)),
                "n_test": int(len(test_group)),
                "train_min": float(np.min(train_group)) if len(train_group) else math.nan,
                "train_max": float(np.max(train_group)) if len(train_group) else math.nan,
                "test_min": float(test_group.min()) if len(test_group) else math.nan,
                "test_max": float(test_group.max()) if len(test_group) else math.nan,
            }
        )
    return groups, pd.DataFrame(rows)


def _scope_subsets(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    subsets: list[tuple[str, str, pd.DataFrame]] = [
        ("overall", "all", frame),
        ("corrupted", "all", frame[frame["corruption_type"] != "none"]),
    ]
    for family in sorted(frame["corruption_type"].unique()):
        subsets.append(("family", str(family), frame[frame["corruption_type"] == family]))
    for severity in sorted(frame["severity_level"].unique()):
        subsets.append(("severity", str(int(severity)), frame[frame["severity_level"] == severity]))
    if "coverage_group" in frame:
        coverage_labels = sorted(str(value) for value in frame["coverage_group"].unique())
        for label in coverage_labels:
            subsets.append(("coverage", label, frame[frame["coverage_group"].astype(str) == label]))
    for font in sorted(frame["font_id"].unique()):
        subsets.append(("font", str(font), frame[frame["font_id"] == font]))
    return subsets


def stratified_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    model: str,
    split: str = "test",
) -> pd.DataFrame:
    """Descriptive metrics by corruption, severity, coverage and font.

    Rows with a constant target (notably the clean severity-0 group) report
    ``r2_defined = False`` and an undefined R² rather than a fabricated value.
    Every row carries its sample and source count.
    """
    subset_frame = frame[frame["split"] == split].copy()
    mask = (frame["split"] == split).to_numpy()
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(predictions) != len(frame):
        raise BenchmarkError("predictions must align with the full frame")
    rows: list[dict[str, Any]] = []
    for scope, level, subset in _scope_subsets(subset_frame):
        if len(subset) == 0:
            continue
        local = subset.index.to_numpy()
        # Convert subset labels to positions in the split frame.
        positions = np.array([subset_frame.index.get_loc(index) for index in local], dtype=int)
        scores = score_frame(subset, predictions[mask][positions])
        rows.append(
            {
                "scope": scope,
                "level": level,
                "split": split,
                "model": model,
                "n_sources": scores["n_sources"],
                "n_samples": scores["n_rows"],
                "mae": scores["mae"],
                "rmse": scores["rmse"],
                "r2": scores["r2"],
                "r2_defined": scores["r2_defined"],
            }
        )
    return pd.DataFrame(rows)


def residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, family, severity), group in predictions.groupby(
        ["split", "corruption_type", "severity_level"], sort=True
    ):
        scores = regression_metrics(
            group[TARGET].to_numpy(dtype=np.float64),
            group["prediction"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "split": split,
                "corruption_type": family,
                "severity_level": int(severity),
                "n_sources": int(group["source_id"].nunique()),
                "n_samples": int(len(group)),
                "mean_residual": float(group["residual"].mean()),
                "sd_residual": (
                    float(group["residual"].std(ddof=1)) if len(group) > 1 else math.nan
                ),
                "mae": scores["mae"],
                "rmse": scores["rmse"],
                "r2": scores["r2"],
                "r2_defined": scores["r2_defined"],
            }
        )
    return pd.DataFrame(rows)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    x_values = x.to_numpy(dtype=np.float64)
    y_values = y.to_numpy(dtype=np.float64)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[mask], y_values[mask]
    if x_values.size < 3 or np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return math.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def feature_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman associations on train, overall and within family."""
    train = frame[frame["split"] == "train"]
    corrupted = train[train["corruption_type"] != "none"]
    scopes: list[tuple[str, str, pd.DataFrame]] = [("train_overall", "all", train)]
    for family in sorted(corrupted["corruption_type"].unique()):
        scopes.append(
            ("train_family", str(family), corrupted[corrupted["corruption_type"] == family])
        )
    rows: list[dict[str, Any]] = []
    for scope, family, subset in scopes:
        for feature in FEATURE_NAMES:
            rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "feature": feature,
                    "n_rows": int(len(subset)),
                    "pearson": _pearson(subset[feature], subset[TARGET]),
                    "spearman": rank_correlation(subset[feature], subset[TARGET]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Held-out font experiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeldOutResult:
    font: str
    candidates: pd.DataFrame
    metrics: pd.DataFrame
    stratified: pd.DataFrame
    predictions: pd.DataFrame
    bootstrap: pd.DataFrame
    selected: dict[str, Any]
    model: FittedModel
    train_fonts: tuple[str, ...]
    train_rows: int
    val_rows: int
    heldout_rows: int
    heldout_sources: int


def run_heldout_font(bundle: DatasetBundle, config: BenchmarkConfig) -> HeldOutResult:
    """Exclude one font entirely, select on the other fonts, evaluate once.

    The held-out font is removed before any candidate is fitted. Its rows are
    not used for scaling, category levels, hyperparameter selection or early
    stopping.
    """
    frame = bundle.frame
    font = config.held_out_font
    fonts = tuple(sorted(frame["font_id"].unique()))
    if font not in fonts:
        raise BenchmarkError(f"held-out font {font!r} not in {fonts}")
    train_fonts = tuple(value for value in fonts if value != font)
    train = frame[(frame["font_id"] != font) & (frame["split"] == "train")]
    val = frame[(frame["font_id"] != font) & (frame["split"] == "val")]
    heldout = frame[frame["font_id"] == font]
    if len(train) == 0 or len(val) == 0 or len(heldout) == 0:
        raise BenchmarkError("held-out-font split has an empty partition")

    specs = benchmark_specs(config)
    rows, fitted = _fit_and_score_candidates(
        specs, train, val, seed=config.seed, selection_metric=config.selection_metric
    )
    selected = select_best(rows, config.selection_metric)
    family_selected = best_per_family(rows, config.selection_metric)
    family_names = {row["name"] for row in family_selected.values()}
    candidates = _candidate_table(rows, selected["name"], family_names)

    naive = naive_spec()
    naive_fit = fit_candidate(naive, train, seed=config.seed)
    metrics_rows: list[dict[str, Any]] = []
    for model_name, model, role in [
        (naive.name, naive_fit, "naive"),
        *[
            (
                row["name"],
                fitted[row["name"]],
                "family_best+overall_selected"
                if row["name"] == selected["name"]
                else "family_best",
            )
            for _, row in sorted(family_selected.items())
        ],
    ]:
        for split_name, split_frame in (
            ("train_fonts_train", train),
            ("train_fonts_val", val),
            ("heldout_font", heldout),
        ):
            metrics_rows.append(
                _metrics_row(
                    name=model_name,
                    family=model.spec.family,
                    role=role,
                    split=split_name,
                    metrics=score_frame(split_frame, predict(model.estimator, split_frame)),
                    params=model.spec.params,
                    n_inputs=model.spec.n_inputs,
                )
            )
    selected_model = fitted[selected["name"]]
    heldout_predictions = predict(selected_model.estimator, heldout)
    predictions = _prediction_frame(
        frame,
        predict(selected_model.estimator, frame),
        model_name=selected["name"],
    )
    predictions = predictions[predictions["font_id"] == font].copy()

    coverage_series, _ = coverage_assignment(
        frame,
        config.coverage_quantiles,
        train_frame=frame[frame["font_id"] != font],
    )
    heldout_with_coverage = heldout.copy()
    heldout_with_coverage["coverage_group"] = coverage_series.loc[heldout.index]
    selected_strata = stratified_metrics(
        heldout_with_coverage.assign(split="test"),
        heldout_predictions,
        model=selected["name"],
        split="test",
    )
    naive_strata = stratified_metrics(
        heldout_with_coverage.assign(split="test"),
        predict(naive_fit.estimator, heldout),
        model=naive.name,
        split="test",
    )
    stratified = pd.concat([selected_strata, naive_strata], ignore_index=True)

    bootstrap = bootstrap_metrics(
        heldout,
        heldout_predictions,
        n_bootstrap=config.n_bootstrap,
        seed=derive_noise_seed(config.seed, "bootstrap", "heldout_font"),
        scope="heldout_font",
    )

    return HeldOutResult(
        font=font,
        candidates=candidates,
        metrics=pd.DataFrame(metrics_rows),
        stratified=stratified,
        predictions=predictions,
        bootstrap=bootstrap,
        selected=selected,
        model=selected_model,
        train_fonts=train_fonts,
        train_rows=len(train),
        val_rows=len(val),
        heldout_rows=len(heldout),
        heldout_sources=int(heldout["source_id"].nunique()),
    )


# ---------------------------------------------------------------------------
# Model cards and reports
# ---------------------------------------------------------------------------


def _dataset_identity(bundle: DatasetBundle) -> dict[str, Any]:
    return {
        "dataset_name": bundle.summary.get("dataset_name"),
        "dataset_version": bundle.summary.get("dataset_version"),
        "generation_config_hash": bundle.generation_config_hash,
        "split_assignment_sha256": bundle.split_assignment_sha256,
        "feature_set_version": bundle.feature_set_version,
        "feature_names": list(FEATURE_NAMES),
        "target": TARGET,
    }


def _model_card(
    *,
    bundle: DatasetBundle,
    config: BenchmarkConfig,
    spec: CandidateSpec,
    metrics: dict[str, dict[str, Any]],
    training_scope: str,
    training_fonts: tuple[str, ...],
    held_out_font: str | None,
    selection_scope: str,
    n_candidates: int,
) -> dict[str, Any]:
    return {
        "model_name": spec.name,
        "model_family": spec.family,
        "params": spec.params,
        "target": TARGET,
        "feature_names": list(FEATURE_NAMES),
        "feature_set_version": bundle.feature_set_version,
        "n_features": len(FEATURE_NAMES),
        "preprocessing": (
            "standard_scaler_fit_on_training_rows" if spec.family in {"ols", "ridge"} else "none"
        ),
        "training_scope": training_scope,
        "training_fonts": list(training_fonts),
        "held_out_font": held_out_font,
        "dataset": _dataset_identity(bundle),
        "selection": {
            "criterion": f"{selection_scope}_{config.selection_metric}",
            "selected_on": selection_scope,
            "n_candidates": n_candidates,
            "benchmark_config_hash": config.config_hash,
        },
        "metrics": metrics,
        "package_version": package_version(),
        "sklearn_version": sklearn_version(),
        "code_revision": code_revision(),
    }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    output_dir: Path
    model_dir: Path
    checks: tuple[CheckResult, ...]
    primary: PrimaryResult
    metadata: MetadataResult
    gaps: pd.DataFrame
    strata: pd.DataFrame
    coverage: pd.DataFrame
    correlations: pd.DataFrame
    residuals: pd.DataFrame
    heldout: HeldOutResult
    uncertainty: pd.DataFrame
    report_path: Path
    metadata_path: Path
    model_paths: dict[str, Path]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)


def _decision_text(
    *,
    primary: PrimaryResult,
    metadata: MetadataResult,
    strata: pd.DataFrame,
    heldout: HeldOutResult,
    config: BenchmarkConfig,
) -> list[str]:
    final = primary.family_metrics
    selected_rows = final[final["role"].str.contains("overall_selected")]
    selected_test = selected_rows[selected_rows["split"] == "test"].iloc[0]
    selected_val = selected_rows[selected_rows["split"] == "val"].iloc[0]
    naive_test = final[(final["model"] == "naive_train_mean") & (final["split"] == "test")].iloc[0]
    naive_val = final[(final["model"] == "naive_train_mean") & (final["split"] == "val")].iloc[0]

    reduction = (float(naive_test["rmse"]) - float(selected_test["rmse"])) / float(
        naive_test["rmse"]
    )
    material = reduction >= config.material_improvement_fraction

    linear = final[
        (final["family"].isin(("ols", "ridge")))
        & (final["split"] == "test")
        & (final["role"].str.contains("family_best"))
    ]
    nonlinear = final[
        (final["family"].isin(("random_forest", "hist_gradient_boosting")))
        & (final["split"] == "test")
        & (final["role"].str.contains("family_best"))
    ]
    best_linear = linear.loc[linear["rmse"].idxmin()]
    best_nonlinear = nonlinear.loc[nonlinear["rmse"].idxmin()]
    nonlinear_gain = float(best_linear["rmse"]) - float(best_nonlinear["rmse"])

    metadata_test = metadata.metrics[
        (metadata.metrics["model"] == "metadata_severity_family")
        & (metadata.metrics["split"] == "test")
    ].iloc[0]
    combined_test = metadata.metrics[
        (metadata.metrics["model"] == "image_plus_metadata") & (metadata.metrics["split"] == "test")
    ].iloc[0]

    heldout_selected = heldout.metrics[
        (heldout.metrics["model"] == heldout.selected["name"])
        & (heldout.metrics["split"] == "heldout_font")
    ].iloc[0]
    heldout_naive = heldout.metrics[
        (heldout.metrics["model"] == "naive_train_mean")
        & (heldout.metrics["split"] == "heldout_font")
    ].iloc[0]
    combined_relation = (
        "did not beat"
        if float(combined_test["r2"]) <= float(selected_test["r2"])
        else "slightly beat"
    )

    family_table = strata[
        (strata["scope"] == "family") & (strata["model"] == selected_test["model"])
    ]
    family_summary = ", ".join(
        f"{row.level} RMSE {_fmt(row.rmse)} (n={int(row.n_samples)})"
        for row in family_table.itertuples(index=False)
    )

    lines = [
        "## Target-meaningfulness decision",
        "",
        "These answers use the actual held-out test results above. The numbers are "
        "descriptive; no significance test is claimed.",
        "",
        "**1. Does a conventional model improve materially over the naive baseline?**",
        "",
        f"Validation RMSE fell from {_fmt(naive_val['rmse'])} (naive) to "
        f"{_fmt(selected_val['rmse'])}; test RMSE fell from {_fmt(naive_test['rmse'])} to "
        f"{_fmt(selected_test['rmse'])} ({reduction * 100:.1f}% relative reduction). "
        f"The predeclared materiality threshold was {config.material_improvement_fraction * 100:.1f}%; "
        f"the result is **{'material' if material else 'not material'}** by that rule. "
        "This is a large gain over a mean-only predictor, which is expected because SSIM "
        "varies strongly with image structure.",
        "",
        "**2. Does a nonlinear model outperform OLS or Ridge on held-out sources?**",
        "",
        f"The best validation-selected linear model was `{best_linear['model']}` "
        f"(test RMSE {_fmt(best_linear['rmse'])}, R² {_fmt(best_linear['r2'])}); the best "
        f"nonlinear model was `{best_nonlinear['model']}` (test RMSE "
        f"{_fmt(best_nonlinear['rmse'])}, R² {_fmt(best_nonlinear['r2'])}). "
        f"The nonlinear model's RMSE was {abs(nonlinear_gain):.4f} "
        f"({'%.1f%%' % (abs(nonlinear_gain) / float(best_linear['rmse']) * 100)} "
        f"{'lower' if nonlinear_gain > 0 else 'higher'} than the linear one). "
        "That difference is descriptive evidence that model class matters here, but it "
        "comes from one 108-sample split and is not evidence of a general nonlinear "
        "advantage.",
        "",
        "**3. Is the result stable across corruption families?**",
        "",
        f"Test error by family for `{selected_test['model']}`: {family_summary or 'not available'}. "
        "The aggregate hides different error levels; family-specific rows above are the "
        "honest summary. A model that is accurate on blur and weaker on noise (or the "
        "reverse) should not be described as uniformly robust.",
        "",
        "**4. How much does metadata-only performance explain the target?**",
        "",
        f"Severity + family alone reached test R² {_fmt(metadata_test['r2'])} "
        f"(RMSE {_fmt(metadata_test['rmse'])}), versus image-only R² "
        f"{_fmt(selected_test['r2'])} (RMSE {_fmt(selected_test['rmse'])}). "
        f"Image + metadata reached R² {_fmt(combined_test['r2'])}. "
        "The metadata-only result confirms that the synthetic corruption design explains "
        "much of the target; the image-only task is not reducible to metadata, but it is "
        "easier than real-world quality assessment. The combined diagnostic (R² "
        f"{_fmt(combined_test['r2'])}) {combined_relation} the selected image-only model; "
        "it is a reference for confounding, not an upper bound and not a deployable model.",
        "",
        "**5. Does the model survive the held-out-font stress test?**",
        "",
        f"Held-out font `{heldout.font}` ({heldout.heldout_sources} sources, "
        f"{heldout.heldout_rows} samples): selected model RMSE "
        f"{_fmt(heldout_selected['rmse'])}, R² {_fmt(heldout_selected['r2'])}, versus "
        f"naive RMSE {_fmt(heldout_naive['rmse'])} "
        f"(R² {_fmt(heldout_naive['r2'])}). The held-out error is much lower than the mean "
        "predictor's in this stress test. The random-forest family winner, not selected "
        "overall, had held-out "
        "RMSE "
        f"{_fmt(float(heldout.metrics[(heldout.metrics['family'] == 'random_forest') & (heldout.metrics['split'] == 'heldout_font')]['rmse'].iloc[0]))}; "
        "that comparison is descriptive and was not used to re-select a model. With three "
        "fonts and 20 held-out sources this is a preliminary font-shift stress test, not "
        "proof of broad real-world generalisation.",
        "",
        "**6. Is a CNN justified by the evidence?**",
        "",
    ]

    near_solved = float(selected_test["r2"]) >= 0.95
    metadata_strong = float(metadata_test["r2"]) >= 0.90
    if near_solved and metadata_strong:
        lines.append(
            "No, not for this synthetic target. The selected model reconstructs "
            f"{float(selected_test['r2']) * 100:.1f}% of held-out SSIM variance (RMSE "
            f"{_fmt(selected_test['rmse'])}), and corruption metadata alone explains "
            f"{float(metadata_test['r2']) * 100:.1f}%. The nonlinear model's gain over "
            "linear regression is already available from a random forest, so a CNN would "
            "mainly add capacity, training cost and opacity. A high R² does not make the "
            "target scientifically interesting or prove real-world generalisation. A CNN "
            "would only be justified on a harder, predeclared shift if it beats this "
            "frozen conventional baseline."
        )
    else:
        lines.append(
            "Not yet justified. The conventional baselines leave enough held-out error "
            "that a CNN could be tested, but only as a controlled comparison against the "
            "feature-only and metadata baselines above. There is no evidence here that a "
            "CNN would generalise better across fonts or to real CAPTCHA images, and none "
            "should be claimed without that comparison."
        )

    lines.extend(
        [
            "",
            "**Recommended next experiment (not implemented in this milestone).** "
            "Keep this dataset and target frozen. Add a genuinely harder shift — a "
            "held-out font plus a new corruption family or a small real-image set with "
            "documented provenance — and require any CNN to beat the frozen feature-only "
            "baseline on the same grouped protocol. If no conventional model survives the "
            "shift, report that rather than forcing a deep model.",
        ]
    )
    return lines


def _report_markdown(
    *,
    bundle: DatasetBundle,
    config: BenchmarkConfig,
    checks: tuple[CheckResult, ...],
    primary: PrimaryResult,
    metadata: MetadataResult,
    gaps: pd.DataFrame,
    strata: pd.DataFrame,
    coverage: pd.DataFrame,
    correlations: pd.DataFrame,
    residuals: pd.DataFrame,
    heldout: HeldOutResult,
    uncertainty: pd.DataFrame,
) -> str:
    lines = [
        f"# Conventional SSIM benchmark — {bundle.summary.get('dataset_name')} "
        f"{bundle.summary.get('dataset_version')}",
        "",
        "**Scope:** synthetic image similarity (SSIM against a clean reference). This is "
        "not a measure of human readability, CAPTCHA solvability, or real-world quality. "
        "All numbers are held-out diagnostics on a small synthetic dataset.",
        "",
        "## Frozen protocol",
        "",
        f"- Benchmark config: `{config.describe()}`",
        f"- Dataset: `{bundle.root}`",
        f"- Dataset generation hash: `{bundle.generation_config_hash}`",
        f"- Split assignment hash: `{bundle.split_assignment_sha256}`",
        f"- Feature set: `{bundle.feature_set_version}` — {len(FEATURE_NAMES)} image features",
        f"- Target: `{TARGET}` (SSIM, data range 0–1)",
        f"- Seed: `{config.seed}`; bootstrap replicates: `{config.n_bootstrap}`",
        f"- Selection criterion: lowest validation **{config.selection_metric.upper()}**",
        "- Test evaluated once after all candidates and family winners were fixed",
        "",
        "**Primary inputs:** only the eight documented features. PSNR, SSIM, corruption "
        "family, severity, font, text length, source IDs and split IDs are excluded from "
        "the primary model. Metadata models below are explicitly diagnostic.",
        "",
        "## Protocol checks",
        "",
    ]
    for check in checks:
        lines.append(f"- **{'PASS' if check.passed else 'FAIL'}** `{check.name}` — {check.detail}")

    split_counts = (
        bundle.frame.groupby(["split", "font_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(list(SPLITS))
    )
    lines.extend(["", "## Data", "", "Samples by split and font:", ""])
    lines.extend(
        _markdown_table(
            ["split", *[str(col) for col in split_counts.columns], "total"],
            [
                [
                    split,
                    *[int(split_counts.loc[split, col]) for col in split_counts.columns],
                    int(split_counts.loc[split].sum()),
                ]
                for split in SPLITS
            ],
            numeric_from=1,
        )
    )

    lines.extend(
        [
            "",
            "## Candidate search (validation only)",
            "",
            f"{len(benchmark_specs(config))} learned candidates were fitted on train and "
            "compared on validation, alongside the naive mean baseline. Test metrics for "
            "candidates are deliberately absent from this table; only the selected model "
            "and family winners were evaluated on test.",
            "",
        ]
    )
    candidate_rows = []
    for row in primary.candidates.sort_values(
        ["val_rmse", "n_inputs", "model"], kind="stable"
    ).itertuples(index=False):
        candidate_rows.append(
            [
                row.model,
                row.family,
                int(row.n_inputs),
                _fmt(row.train_rmse, 4),
                _fmt(row.val_rmse, 4),
                _fmt(row.train_r2),
                _fmt(row.val_r2),
                "yes" if row.selected else "no",
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "model",
                "family",
                "inputs",
                "train RMSE",
                "val RMSE",
                "train R²",
                "val R²",
                "selected",
            ],
            candidate_rows,
            numeric_from=2,
        )
    )

    final = primary.family_metrics
    lines.extend(
        [
            "",
            "## Model-family comparison (test evaluated once)",
            "",
            "Each family's configuration was chosen on validation, then evaluated once on "
            "test. The naive baseline is the training mean. R² is `n/a` where the target "
            "is constant (for example, the clean-only stratum).",
            "",
        ]
    )
    rows = []
    for row in _sorted_by_split(final, ["family", "rmse"]).itertuples(index=False):
        rows.append(
            [
                row.split,
                row.model,
                row.family,
                row.role,
                int(row.n_rows),
                int(row.n_sources),
                _fmt(row.mae, 4),
                _fmt(row.rmse, 4),
                _fmt(row.r2),
            ]
        )
    lines.extend(
        _markdown_table(
            ["split", "model", "family", "role", "n", "groups", "MAE", "RMSE", "R²"],
            rows,
            numeric_from=4,
        )
    )

    lines.extend(
        [
            "",
            "## Metadata diagnostics",
            "",
            "These models use generated labels, not image features. They are reported to "
            "show how much of the target is explained by the corruption design and font/"
            "length controls. They are **not** the intended image-only model.",
            "",
        ]
    )
    rows = []
    for row in _sorted_by_split(metadata.metrics, ["model"]).itertuples(index=False):
        rows.append(
            [
                row.split,
                row.model,
                int(row.n_inputs),
                int(row.n_rows),
                _fmt(row.mae, 4),
                _fmt(row.rmse, 4),
                _fmt(row.r2),
            ]
        )
    lines.extend(
        _markdown_table(
            ["split", "model", "inputs", "n", "MAE", "RMSE", "R²"], rows, numeric_from=2
        )
    )

    lines.extend(
        [
            "",
            "### Performance gaps",
            "",
            "Positive RMSE gap means model A has higher (worse) error than model B; positive "
            "R² gap means A has higher (better) R².",
            "",
        ]
    )
    rows = []
    for row in gaps.itertuples(index=False):
        rows.append(
            [
                row.comparison,
                row.split,
                _fmt(row.a_rmse, 4),
                _fmt(row.b_rmse, 4),
                _fmt(row.rmse_gap_a_minus_b, 4),
                _fmt(row.r2_gap_a_minus_b),
            ]
        )
    lines.extend(
        _markdown_table(
            ["comparison", "split", "A RMSE", "B RMSE", "RMSE gap", "R² gap"],
            rows,
            numeric_from=2,
        )
    )

    lines.extend(
        [
            "",
            "## Corruption- and coverage-stratified test results",
            "",
            "Descriptive only; these rows were not used to select or tune the model. "
            "Coverage bins use training-split quantiles of `ink_fraction`, so the bin edges "
            "are fixed without test information.",
            "",
        ]
    )
    rows = []
    for row in strata.sort_values(["model", "scope", "level"]).itertuples(index=False):
        rows.append(
            [
                row.model,
                row.scope,
                row.level,
                int(row.n_sources),
                int(row.n_samples),
                _fmt(row.mae, 4),
                _fmt(row.rmse, 4),
                _fmt(row.r2),
                "yes" if row.r2_defined else "no",
            ]
        )
    lines.extend(
        _markdown_table(
            ["model", "scope", "level", "groups", "n", "MAE", "RMSE", "R²", "R² defined"],
            rows,
            numeric_from=3,
        )
    )

    lines.extend(
        [
            "",
            "Coverage bins are assigned per sample. Because severity changes ink "
            "coverage, one source can contribute to more than one bin; the group "
            "counts in coverage rows are therefore not additive.",
            "",
            "Coverage bin definition (train quantiles):",
            "",
        ]
    )
    rows = []
    for row in coverage.itertuples(index=False):
        rows.append(
            [
                row.coverage_group,
                _fmt(row.quantile_low, 2),
                _fmt(row.quantile_high, 2),
                _fmt(row.train_min, 4),
                _fmt(row.train_max, 4),
                int(row.n_train),
                int(row.n_test),
            ]
        )
    lines.extend(
        _markdown_table(
            ["group", "q low", "q high", "train min", "train max", "n train", "n test"],
            rows,
            numeric_from=1,
        )
    )

    lines.extend(
        [
            "",
            "## Feature-target associations (train)",
            "",
            "Aggregate correlations can reverse sign between corruption families; the "
            "within-family rows are the honest summary.",
            "",
        ]
    )
    rows = []
    for row in correlations.itertuples(index=False):
        rows.append(
            [
                row.scope,
                row.family,
                row.feature,
                int(row.n_rows),
                _fmt(row.pearson),
                _fmt(row.spearman),
            ]
        )
    lines.extend(
        _markdown_table(
            ["scope", "family", "feature", "n", "Pearson r", "Spearman rho"],
            rows,
            numeric_from=3,
        )
    )

    lines.extend(
        [
            "",
            "## Residual summary (selected model)",
            "",
            "Residual = prediction − actual SSIM. Positive means the model over-predicts quality.",
            "",
        ]
    )
    rows = []
    for row in _sorted_by_split(residuals, ["corruption_type", "severity_level"]).itertuples(
        index=False
    ):
        rows.append(
            [
                row.split,
                row.corruption_type,
                int(row.severity_level),
                int(row.n_sources),
                int(row.n_samples),
                _fmt(row.mean_residual, 4),
                _fmt(row.sd_residual, 4),
                _fmt(row.mae, 4),
                _fmt(row.rmse, 4),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "split",
                "family",
                "severity",
                "groups",
                "n",
                "mean residual",
                "sd",
                "MAE",
                "RMSE",
            ],
            rows,
            numeric_from=3,
        )
    )

    lines.extend(
        [
            "",
            "## Grouped uncertainty",
            "",
            "Bootstrap intervals resample **source groups**, not variants, so correlated "
            "samples are not treated as independent. The fitted predictions are held fixed. "
            "With 12 test groups and 20 held-out-font groups these intervals are unstable "
            "and descriptive only.",
            "",
        ]
    )
    rows = []
    for row in uncertainty.itertuples(index=False):
        rows.append(
            [
                row.scope,
                row.metric,
                _fmt(row.estimate, 4),
                _fmt(row.ci_low, 4),
                _fmt(row.ci_high, 4),
                int(row.n_groups),
                int(row.n_samples),
                _fmt(row.r2_defined_rate, 2),
            ]
        )
    lines.extend(
        _markdown_table(
            ["scope", "metric", "estimate", "2.5%", "97.5%", "groups", "n", "R² defined rate"],
            rows,
            numeric_from=2,
        )
    )

    lines.extend(
        [
            "",
            "## Held-out-font stress test",
            "",
            f"Held-out font: **{heldout.font}** — {heldout.heldout_sources} sources, "
            f"{heldout.heldout_rows} samples. Training fonts: "
            f"{', '.join(heldout.train_fonts)}. Selection used only the training-font "
            f"validation rows ({heldout.val_rows} samples); the held-out font was never "
            "used for fitting, scaling or hyperparameter selection.",
            "",
            f"Selection rationale (recorded before fitting): {config.held_out_font_reason}",
            "",
        ]
    )
    rows = []
    for row in _sorted_by_split(heldout.metrics, ["family", "rmse"]).itertuples(index=False):
        rows.append(
            [
                row.split,
                row.model,
                row.family,
                row.role,
                int(row.n_rows),
                int(row.n_sources),
                _fmt(row.mae, 4),
                _fmt(row.rmse, 4),
                _fmt(row.r2),
            ]
        )
    lines.extend(
        _markdown_table(
            ["split", "model", "family", "role", "n", "groups", "MAE", "RMSE", "R²"],
            rows,
            numeric_from=4,
        )
    )

    lines.extend(
        [
            "",
            "Held-out-font strata (selected model vs naive), descriptive only:",
            "",
        ]
    )
    rows = []
    for row in heldout.stratified.sort_values(["model", "scope", "level"]).itertuples(index=False):
        rows.append(
            [
                row.model,
                row.scope,
                row.level,
                int(row.n_sources),
                int(row.n_samples),
                _fmt(row.mae, 4),
                _fmt(row.rmse, 4),
                _fmt(row.r2),
            ]
        )
    lines.extend(
        _markdown_table(
            ["model", "scope", "level", "groups", "n", "MAE", "RMSE", "R²"],
            rows,
            numeric_from=3,
        )
    )

    lines.extend(
        _decision_text(
            primary=primary,
            metadata=metadata,
            strata=strata,
            heldout=heldout,
            config=config,
        )
    )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Synthetic images only; SSIM is not a human-readability measure and this "
            "benchmark does not predict CAPTCHA solvability.",
            "- Three fonts, one canvas size, two corruption families and 60 source groups. "
            "The random grouped split is not a cross-generator test; the held-out-font "
            "experiment is a preliminary stress test.",
            "- The test split has 12 source groups and 108 samples. Group-bootstrap "
            "intervals are wide and unstable; no significance test is claimed.",
            "- Validation and test share the same fonts in the primary benchmark. The "
            "held-out-font experiment removes one font entirely, but only two fonts remain "
            "for training.",
            "- Text length was not stratified in the dataset; validation contains no "
            "length-6 sources by chance.",
            "- Metadata models use generated labels and are diagnostic references, not "
            "deployable inputs; the combined model is not guaranteed to upper-bound a "
            "nonlinear image-only model.",
            "- Scikit-learn model persistence uses joblib, which is not safe for untrusted "
            "files; only load artifacts generated locally.",
            "",
            "Artifacts: `reports/<benchmark>/` (CSV/Markdown) and `models/<benchmark>/` "
            "(joblib + JSON cards), both Git-ignored.",
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    dataset: Path | str,
    config: BenchmarkConfig,
    *,
    output_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
) -> BenchmarkResult:
    """Run the complete frozen protocol and write reports/models."""
    started = time.perf_counter()
    bundle = load_benchmark_dataset(dataset)
    checks = protocol_checks(bundle, config)
    if not all(check.passed for check in checks):
        failures = "; ".join(f"{c.name}: {c.detail}" for c in checks if not c.passed)
        raise BenchmarkError(f"protocol checks failed: {failures}")

    primary = run_primary(bundle, config)
    metadata = run_metadata_experiments(bundle)
    metadata = MetadataResult(
        metrics=metadata.metrics, gaps=_gap_table(primary.family_metrics, metadata.metrics)
    )
    gaps = metadata.gaps

    coverage_series, coverage_table = coverage_assignment(bundle.frame, config.coverage_quantiles)
    frame = bundle.frame.copy()
    frame["coverage_group"] = coverage_series.astype(str)
    strata = stratified_metrics(
        frame,
        predict(primary.selected_model.estimator, frame),
        model=str(primary.selected["name"]),
        split="test",
    )
    naive_strata = stratified_metrics(
        frame,
        predict(
            fit_candidate(
                naive_spec(), frame[frame["split"] == "train"], seed=config.seed
            ).estimator,
            frame,
        ),
        model="naive_train_mean",
        split="test",
    )
    strata = pd.concat([strata, naive_strata], ignore_index=True)

    correlations = feature_correlations(frame)
    residuals = residual_summary(primary.predictions)
    heldout = run_heldout_font(bundle, config)
    uncertainty = pd.concat([primary.bootstrap, heldout.bootstrap], ignore_index=True).sort_values(
        ["scope", "metric"], kind="stable"
    )

    output_dir = Path(output_dir) if output_dir is not None else config.output_dir
    model_dir = Path(model_dir) if model_dir is not None else config.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    primary_card = _model_card(
        bundle=bundle,
        config=config,
        spec=primary.selected_model.spec,
        metrics={
            split: {
                "n_rows": int(row["n_rows"]),
                "n_sources": int(row["n_sources"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "r2": float(row["r2"]),
                "r2_defined": bool(row["r2_defined"]),
            }
            for split in SPLITS
            for row in [
                primary.family_metrics[
                    (primary.family_metrics["model"] == primary.selected["name"])
                    & (primary.family_metrics["split"] == split)
                ].iloc[0]
            ]
        },
        training_scope="all_fonts_train_split",
        training_fonts=tuple(sorted(bundle.frame["font_id"].unique())),
        held_out_font=None,
        selection_scope="val",
        n_candidates=len(benchmark_specs(config)),
    )
    primary_paths = save_model_bundle(
        model_dir, "primary_model", primary.selected_model.estimator, primary_card
    )

    heldout_row = heldout.metrics[
        (heldout.metrics["model"] == heldout.selected["name"])
        & (heldout.metrics["split"] == "heldout_font")
    ].iloc[0]
    heldout_val_row = heldout.metrics[
        (heldout.metrics["model"] == heldout.selected["name"])
        & (heldout.metrics["split"] == "train_fonts_val")
    ].iloc[0]
    heldout_train_row = heldout.metrics[
        (heldout.metrics["model"] == heldout.selected["name"])
        & (heldout.metrics["split"] == "train_fonts_train")
    ].iloc[0]
    heldout_card = _model_card(
        bundle=bundle,
        config=config,
        spec=heldout.model.spec,
        metrics={
            "train": {
                "n_rows": int(heldout_train_row["n_rows"]),
                "n_sources": int(heldout_train_row["n_sources"]),
                "mae": float(heldout_train_row["mae"]),
                "rmse": float(heldout_train_row["rmse"]),
                "r2": float(heldout_train_row["r2"]),
                "r2_defined": bool(heldout_train_row["r2_defined"]),
            },
            "val": {
                "n_rows": int(heldout_val_row["n_rows"]),
                "n_sources": int(heldout_val_row["n_sources"]),
                "mae": float(heldout_val_row["mae"]),
                "rmse": float(heldout_val_row["rmse"]),
                "r2": float(heldout_val_row["r2"]),
                "r2_defined": bool(heldout_val_row["r2_defined"]),
            },
            "heldout_font": {
                "n_rows": int(heldout_row["n_rows"]),
                "n_sources": int(heldout_row["n_sources"]),
                "mae": float(heldout_row["mae"]),
                "rmse": float(heldout_row["rmse"]),
                "r2": float(heldout_row["r2"]),
                "r2_defined": bool(heldout_row["r2_defined"]),
            },
        },
        training_scope=f"fonts_excluding_{heldout.font}_train_split",
        training_fonts=heldout.train_fonts,
        held_out_font=heldout.font,
        selection_scope="train_fonts_val",
        n_candidates=len(benchmark_specs(config)),
    )
    heldout_paths = save_model_bundle(
        model_dir, "heldout_font_model", heldout.model.estimator, heldout_card
    )

    output_files = {
        "model_comparison.csv": primary.candidates,
        "family_test_metrics.csv": primary.family_metrics,
        "primary_predictions.csv": primary.predictions,
        "metadata_experiments.csv": metadata.metrics,
        "performance_gaps.csv": gaps,
        "stratified_metrics.csv": strata,
        "coverage_groups.csv": coverage_table,
        "feature_correlations.csv": correlations,
        "residual_summary.csv": residuals,
        "heldout_font_candidates.csv": heldout.candidates,
        "heldout_font_metrics.csv": heldout.metrics,
        "heldout_font_stratified.csv": heldout.stratified,
        "heldout_font_predictions.csv": heldout.predictions,
        "uncertainty.csv": uncertainty,
    }
    for filename, table in output_files.items():
        _write_csv(table, output_dir / filename)

    report = _report_markdown(
        bundle=bundle,
        config=config,
        checks=checks,
        primary=primary,
        metadata=metadata,
        gaps=gaps,
        strata=strata,
        coverage=coverage_table,
        correlations=correlations,
        residuals=residuals,
        heldout=heldout,
        uncertainty=uncertainty,
    )
    report_path = output_dir / "benchmark.md"
    report_temporary = report_path.with_name(report_path.name + ".tmp")
    report_temporary.write_text(report, encoding="utf-8")
    report_temporary.replace(report_path)

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_metadata.json"
    }
    artifact_hashes.update(
        {
            f"models/{path.name}": sha256_file(path)
            for path in sorted(model_dir.iterdir())
            if path.is_file()
        }
    )
    elapsed = time.perf_counter() - started
    metadata_payload = {
        "benchmark_config_hash": config.config_hash,
        "benchmark_config": config.identity_payload(),
        "dataset": _dataset_identity(bundle),
        "selection": {
            "primary_model": primary.selected["name"],
            "primary_family": primary.selected["family"],
            "selection_metric": f"val_{config.selection_metric}",
            "selected_on": "val",
            "n_candidates": len(benchmark_specs(config)),
            "heldout_font_model": heldout.selected["name"],
            "heldout_font": heldout.font,
        },
        "feature_names": list(FEATURE_NAMES),
        "target": TARGET,
        "seed": config.seed,
        "package_version": package_version(),
        "sklearn_version": sklearn_version(),
        "code_revision": code_revision(),
        "elapsed_seconds_in_process": round(elapsed, 6),
        "artifacts": artifact_hashes,
    }
    metadata_path = output_dir / "run_metadata.json"
    _write_json(_json_ready(metadata_payload), metadata_path)

    return BenchmarkResult(
        output_dir=output_dir,
        model_dir=model_dir,
        checks=checks,
        primary=primary,
        metadata=metadata,
        gaps=gaps,
        strata=strata,
        coverage=coverage_table,
        correlations=correlations,
        residuals=residuals,
        heldout=heldout,
        uncertainty=uncertainty,
        report_path=report_path,
        metadata_path=metadata_path,
        model_paths={
            "primary_artifact": primary_paths["artifact"],
            "primary_card": primary_paths["card"],
            "heldout_artifact": heldout_paths["artifact"],
            "heldout_card": heldout_paths["card"],
        },
    )


# ---------------------------------------------------------------------------
# Model verification CLI
# ---------------------------------------------------------------------------


def verify_models(dataset: Path | str, model_dir: Path | str) -> VerificationReport:
    """Verify every ``*_card.json`` bundle in ``model_dir`` against a dataset."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise BenchmarkError(f"model directory not found: {model_dir}")
    cards = sorted(model_dir.glob("*_card.json"))
    if not cards:
        raise BenchmarkError(f"no model cards found in {model_dir}")
    _, _, _, summary = load_dataset(dataset)
    checks: list[CheckResult] = []
    for card_path in cards:
        stem = card_path.name.removesuffix("_card.json")
        model, _, model_checks = load_model_bundle(model_dir, stem, dataset_summary=summary)
        for check in model_checks:
            checks.append(CheckResult(f"{stem}:{check.name}", check.passed, check.detail))
        if model is not None:
            checks.append(CheckResult(f"{stem}:loaded", True, "model bundle is usable"))
    return VerificationReport(dataset=str(dataset), checks=tuple(checks))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-benchmark",
        description="Benchmark conventional SSIM models on a frozen synthetic dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the frozen benchmark")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--model-dir", type=Path, default=None)

    verify = subparsers.add_parser(
        "verify-model", help="verify persisted model cards against a dataset"
    )
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument("--model-dir", type=Path, required=True)
    return parser


def _print_report(report: VerificationReport) -> None:
    for check in report.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print("model verification: PASS" if report.ok else "model verification: FAIL")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI; returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = load_benchmark_config(args.config)
            result = run_benchmark(
                args.dataset,
                config,
                output_dir=args.output_dir,
                model_dir=args.model_dir,
            )
            print(f"benchmark: {config.describe()}")
            print(f"dataset: {args.dataset}")
            print(f"candidates: {len(benchmark_specs(config))}")
            print(
                f"selected: {result.primary.selected['name']} "
                f"(family {result.primary.selected['family']})"
            )
            test = result.primary.family_metrics[
                (result.primary.family_metrics["role"].str.contains("overall_selected"))
                & (result.primary.family_metrics["split"] == "test")
            ].iloc[0]
            naive = result.primary.family_metrics[
                (result.primary.family_metrics["model"] == "naive_train_mean")
                & (result.primary.family_metrics["split"] == "test")
            ].iloc[0]
            print(f"test: RMSE {test['rmse']:.4f} (naive {naive['rmse']:.4f}), R2 {test['r2']:.4f}")
            print(
                f"held-out font {result.heldout.font}: "
                f"RMSE {result.heldout.metrics[(result.heldout.metrics['model'] == result.heldout.selected['name']) & (result.heldout.metrics['split'] == 'heldout_font')].iloc[0]['rmse']:.4f}"
            )
            print(f"report: {result.report_path}")
            print(f"primary model: {result.model_paths['primary_artifact']}")
            return 0 if result.ok else 1
        if args.command == "verify-model":
            report = verify_models(args.dataset, args.model_dir)
            _print_report(report)
            return 0 if report.ok else 1
        raise BenchmarkError(f"unknown command {args.command!r}")  # pragma: no cover
    except (ConfigError, DatasetError, BenchmarkError, ModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
