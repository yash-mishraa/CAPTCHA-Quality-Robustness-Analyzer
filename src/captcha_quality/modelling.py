"""Conventional regression models, grouped evaluation and model persistence.

This module is deliberately small and dependency-light: scikit-learn estimators,
NumPy metrics, and deterministic grouped bootstrap. It contains no dataset
generation and no test-set model selection.

Leakage boundary
----------------
A primary model receives exactly :data:`captcha_quality.features.FEATURE_NAMES`,
in that order. It never receives the target, PSNR, corruption family, severity,
font, text length, source identifiers or split labels. Those columns are allowed
only in the explicitly labelled metadata diagnostics in ``benchmark.py``.

Preprocessing
-------------
Scalers are inside a scikit-learn ``Pipeline`` and are fitted only on the rows
passed to ``fit`` (the training split). Tree models receive raw features, which
is intentional: they do not require scaling and adding it would only obscure
what the model sees.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from captcha_quality.dataset import CheckResult
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.provenance import sha256_file

#: The only supervised target used in this milestone.
TARGET: Final[str] = "ssim_target"

#: Split names in the reporting order used by the dataset pipeline.
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")

#: Feature-only design columns. ``sample_id`` is an identifier, never an input.
FEATURE_COLUMNS: Final[tuple[str, ...]] = FEATURE_NAMES

#: Columns that must never appear in a primary feature matrix. The explicit
#: column selection is the real guard; this set makes accidental additions fail
#: loudly and documents why each column is excluded.
FORBIDDEN_INPUT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        TARGET,
        "psnr_target",
        "psnr_infinite",
        "ms_ssim_target",
        "corruption_type",
        "severity_level",
        "corruption_parameters",
        "font_id",
        "text_length",
        "source_id",
        "split",
        "split_group_id",
        "variant_id",
        "render_seed",
        "variant_seed",
        "text_sha256",
        "image_path",
        "reference_path",
        "image_sha256",
        "reference_sha256",
    }
)

#: Bootstrap percentile interval, reported as a descriptive range.
BOOTSTRAP_LOW: Final[float] = 2.5
BOOTSTRAP_HIGH: Final[float] = 97.5


class ModelError(RuntimeError):
    """Raised when a model or its persisted metadata is incompatible."""


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One deterministic model configuration in a predeclared search grid."""

    name: str
    family: str
    params: dict[str, Any]
    n_inputs: int

    def build(self, *, seed: int) -> Any:
        """Return a fresh, unfitted scikit-learn estimator."""
        if self.family == "naive":
            return DummyRegressor(strategy="mean")
        if self.family == "ols":
            return Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            )
        if self.family == "ridge":
            return Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=float(self.params["alpha"]))),
                ]
            )
        if self.family == "random_forest":
            depth = self.params["max_depth"]
            return RandomForestRegressor(
                n_estimators=int(self.params["n_estimators"]),
                max_depth=None if depth == 0 else int(depth),
                min_samples_leaf=int(self.params["min_samples_leaf"]),
                random_state=seed,
                n_jobs=1,
            )
        if self.family == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(
                max_iter=int(self.params["max_iter"]),
                learning_rate=float(self.params["learning_rate"]),
                max_leaf_nodes=int(self.params["max_leaf_nodes"]),
                min_samples_leaf=int(self.params["min_samples_leaf"]),
                l2_regularization=float(self.params["l2_regularization"]),
                random_state=seed,
            )
        raise ModelError(f"unknown model family {self.family!r}")


@dataclass(frozen=True, slots=True)
class FittedModel:
    """A fitted estimator plus the schema it was trained against."""

    spec: CandidateSpec
    estimator: Any
    feature_names: tuple[str, ...] = field(default=FEATURE_COLUMNS)


def sklearn_version() -> str:
    """Return the installed scikit-learn version."""
    return importlib_metadata.version("scikit-learn")


def validate_feature_columns(columns: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate a feature table's columns and return the input order.

    The table must contain ``sample_id`` followed by exactly the eight documented
    features. Extra columns are rejected rather than silently ignored, so a
    future metadata or target column cannot enter a primary model by accident.
    """
    expected = ("sample_id", *FEATURE_COLUMNS)
    actual = tuple(columns)
    if actual != expected:
        leaked = sorted(FORBIDDEN_INPUT_COLUMNS & set(actual))
        raise ModelError(
            f"feature table columns must be exactly {list(expected)}, got {list(actual)}; "
            f"forbidden columns present: {leaked}"
        )
    return FEATURE_COLUMNS


def model_inputs(frame: pd.DataFrame) -> np.ndarray:
    """Return the primary feature matrix, never any metadata or target column.

    The merged benchmark frame intentionally carries identifiers, targets and
    metadata for reporting. This function selects only the eight documented
    features, in order; :func:`validate_feature_columns` separately enforces the
    exact schema of ``features.csv`` when the dataset is loaded.
    """
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ModelError(f"frame is missing feature column(s): {missing}")
    return frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)


def regression_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, float | int | bool]:
    """Compute MAE, RMSE, R² and sample count for aligned arrays.

    R² is undefined when the actual values are constant or the input is empty;
    in that case it is returned as ``nan`` and ``r2_defined`` is ``False``.
    RMSE is the population root mean squared error. No value is silently
    estimated or replaced.
    """
    y = np.asarray(actual, dtype=np.float64)
    y_hat = np.asarray(predicted, dtype=np.float64)
    if y.shape != y_hat.shape:
        raise ModelError(f"actual and predicted shapes differ: {y.shape} vs {y_hat.shape}")
    if y.size == 0:
        return {
            "n_rows": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "r2": math.nan,
            "r2_defined": False,
        }
    finite = np.isfinite(y) & np.isfinite(y_hat)
    if not finite.all():
        raise ModelError(f"{int((~finite).sum())} non-finite target/prediction value(s)")
    residual = y - y_hat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_defined = ss_tot > 0.0
    return {
        "n_rows": int(y.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - ss_res / ss_tot) if r2_defined else math.nan,
        "r2_defined": r2_defined,
    }


def score_frame(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    target: str = TARGET,
) -> dict[str, float | int | bool]:
    """Score predictions for ``frame`` and include the number of source groups."""
    metrics = regression_metrics(frame[target].to_numpy(dtype=np.float64), predicted)
    metrics["n_sources"] = int(frame["source_id"].nunique()) if "source_id" in frame else 0
    return metrics


def predict(estimator: Any, frame: pd.DataFrame) -> np.ndarray:
    """Predict from a frame after validating the feature-only schema."""
    values = estimator.predict(model_inputs(frame))
    predictions = np.asarray(values, dtype=np.float64).reshape(-1)
    if predictions.size != len(frame):
        raise ModelError(f"model returned {predictions.size} prediction(s) for {len(frame)} row(s)")
    if not np.isfinite(predictions).all():
        raise ModelError("model produced non-finite predictions")
    return predictions


def fit_candidate(spec: CandidateSpec, train: pd.DataFrame, *, seed: int) -> FittedModel:
    """Fit one candidate on exactly the rows supplied (the training split)."""
    estimator = spec.build(seed=seed)
    estimator.fit(model_inputs(train), train[TARGET].to_numpy(dtype=np.float64))
    return FittedModel(spec=spec, estimator=estimator)


def naive_spec() -> CandidateSpec:
    """The training-mean baseline, included in every primary comparison."""
    return CandidateSpec(name="naive_train_mean", family="naive", params={}, n_inputs=0)


def candidate_specs(
    *,
    ridge_alphas: tuple[float, ...],
    rf_n_estimators: int,
    rf_max_depths: tuple[int, ...],
    rf_min_samples_leafs: tuple[int, ...],
    hgb_max_iter: int,
    hgb_learning_rates: tuple[float, ...],
    hgb_max_leaf_nodes: tuple[int, ...],
    hgb_min_samples_leaf: int,
    hgb_l2_regularization: float,
) -> tuple[CandidateSpec, ...]:
    """Build the predeclared search grid in a stable order.

    ``max_depth == 0`` means unlimited depth; it is stored as ``0`` in the
    candidate metadata so the grid remains JSON/TOML friendly.
    """
    specs: list[CandidateSpec] = [
        CandidateSpec(name="ols", family="ols", params={}, n_inputs=len(FEATURE_COLUMNS))
    ]
    for alpha in sorted(set(ridge_alphas)):
        specs.append(
            CandidateSpec(
                name=f"ridge_alpha_{alpha:g}",
                family="ridge",
                params={"alpha": float(alpha)},
                n_inputs=len(FEATURE_COLUMNS),
            )
        )
    for depth in rf_max_depths:
        for leaf in rf_min_samples_leafs:
            depth_label = "unlimited" if depth == 0 else str(depth)
            specs.append(
                CandidateSpec(
                    name=f"random_forest_depth_{depth_label}_leaf_{leaf}",
                    family="random_forest",
                    params={
                        "n_estimators": int(rf_n_estimators),
                        "max_depth": int(depth),
                        "min_samples_leaf": int(leaf),
                    },
                    n_inputs=len(FEATURE_COLUMNS),
                )
            )
    for learning_rate in hgb_learning_rates:
        for leaves in hgb_max_leaf_nodes:
            specs.append(
                CandidateSpec(
                    name=(f"hist_gradient_boosting_lr_{learning_rate:g}_leaves_{leaves}"),
                    family="hist_gradient_boosting",
                    params={
                        "max_iter": int(hgb_max_iter),
                        "learning_rate": float(learning_rate),
                        "max_leaf_nodes": int(leaves),
                        "min_samples_leaf": int(hgb_min_samples_leaf),
                        "l2_regularization": float(hgb_l2_regularization),
                    },
                    n_inputs=len(FEATURE_COLUMNS),
                )
            )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):  # pragma: no cover - validated config guard
        raise ModelError(f"duplicate candidate name(s) in grid: {names}")
    return tuple(specs)


def selection_key(row: dict[str, Any], metric: str) -> tuple[bool, float, int, str]:
    """Sort key for model selection: finite first, then metric, inputs, name.

    Only validation rows are ever passed to this function. The key deliberately
    contains no test field, so test performance cannot influence selection.
    """
    value = row.get(f"val_{metric}")
    finite = isinstance(value, int | float) and math.isfinite(float(value))
    numeric = float(value) if finite else math.inf
    return (not finite, numeric, int(row.get("n_inputs", 0)), str(row.get("name", "")))


def select_best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Select the lowest-validation-error learned candidate.

    Ties are broken by fewer model inputs, then by name. The naive baseline is a
    reference, not a learned candidate, and is never returned.
    """
    learned = [row for row in rows if row.get("family") != "naive"]
    if not learned:
        raise ModelError("no learned candidates were supplied for selection")
    return min(learned, key=lambda row: selection_key(row, metric))


def best_per_family(rows: list[dict[str, Any]], metric: str) -> dict[str, dict[str, Any]]:
    """Return the validation-selected candidate for each learned family."""
    families = sorted({str(row["family"]) for row in rows if row.get("family") != "naive"})
    selected: dict[str, dict[str, Any]] = {}
    for family in families:
        members = [row for row in rows if row.get("family") == family]
        selected[family] = min(members, key=lambda row: selection_key(row, metric))
    return selected


def family_one_hot(frame: pd.DataFrame, categories: tuple[str, ...]) -> np.ndarray:
    """One-hot encode corruption family, dropping the first category."""
    values = np.zeros((len(frame), max(len(categories) - 1, 0)), dtype=np.float64)
    for index, category in enumerate(categories[1:]):
        values[:, index] = (frame["corruption_type"].to_numpy() == category).astype(np.float64)
    return values


def font_one_hot(frame: pd.DataFrame, categories: tuple[str, ...]) -> np.ndarray:
    """One-hot encode font, dropping the first category."""
    values = np.zeros((len(frame), max(len(categories) - 1, 0)), dtype=np.float64)
    for index, category in enumerate(categories[1:]):
        values[:, index] = (frame["font_id"].to_numpy() == category).astype(np.float64)
    return values


def standardise_features(
    frame: pd.DataFrame,
    train: pd.DataFrame,
) -> np.ndarray:
    """Standardise the eight features using training rows only."""
    mean = train[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64).mean(axis=0)
    scale = train[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64).std(axis=0)
    scale[scale == 0.0] = 1.0
    values = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    return (values - mean) / scale


def metadata_design(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    kind: str,
) -> tuple[np.ndarray, list[str]]:
    """Build an explicitly diagnostic metadata design.

    ``kind`` is one of:

    ``severity_family``
        Severity as a numeric column plus corruption-family indicators.
    ``font_text_length``
        Font indicators plus text length.
    ``image_plus_metadata``
        Standardised image features plus severity and corruption-family
        indicators. This is a combined diagnostic reference, not the deployed
        task and not a guaranteed upper bound on a nonlinear image-only model.

    Categories and scaling statistics come from ``train`` only. An unseen test
    category is an error rather than a silent all-zero encoding.
    """
    train_families = tuple(sorted(train["corruption_type"].unique()))
    train_fonts = tuple(sorted(train["font_id"].unique()))
    for column, categories in (
        ("corruption_type", train_families),
        ("font_id", train_fonts),
    ):
        unseen = sorted(set(frame[column].unique()) - set(categories))
        if unseen:
            raise ModelError(f"{column} value(s) {unseen} were not present in training")

    severity = frame[["severity_level"]].to_numpy(dtype=np.float64)
    if kind == "severity_family":
        design = np.column_stack([severity, family_one_hot(frame, train_families)])
        names = ["severity_level", *[f"family_{value}" for value in train_families[1:]]]
    elif kind == "font_text_length":
        design = np.column_stack(
            [
                font_one_hot(frame, train_fonts),
                frame[["text_length"]].to_numpy(dtype=np.float64),
            ]
        )
        names = [*[f"font_{value}" for value in train_fonts[1:]], "text_length"]
    elif kind == "image_plus_metadata":
        design = np.column_stack(
            [
                standardise_features(frame, train),
                severity,
                family_one_hot(frame, train_families),
            ]
        )
        names = [
            *FEATURE_COLUMNS,
            "severity_level",
            *[f"family_{value}" for value in train_families[1:]],
        ]
    else:  # pragma: no cover - guarded by callers
        raise ModelError(f"unknown metadata design {kind!r}")
    if design.shape[1] != len(names):  # pragma: no cover - structural guard
        raise ModelError("metadata design width does not match its column names")
    return design, names


def bootstrap_metrics(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    scope: str,
    target: str = TARGET,
) -> pd.DataFrame:
    """Grouped bootstrap over source groups, never over individual variants.

    The fitted model's predictions are held fixed. Each replicate resamples
    source groups with replacement and concatenates all of their rows, so
    correlated variants of one source are not treated as independent. The
    returned interval is descriptive; with few groups it is necessarily wide and
    unstable.
    """
    if n_bootstrap < 0:
        raise ModelError("n_bootstrap must be >= 0")
    if n_bootstrap == 0:
        return pd.DataFrame(
            columns=(
                "scope",
                "metric",
                "estimate",
                "ci_low",
                "ci_high",
                "n_groups",
                "n_samples",
                "n_bootstrap",
                "r2_defined_rate",
            )
        )
    if len(frame) != len(predicted):
        raise ModelError("frame and prediction lengths differ")
    y = frame[target].to_numpy(dtype=np.float64)
    groups = frame["source_id"].to_numpy()
    unique_groups = np.array(sorted(set(groups)))
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)

    replicates: list[dict[str, float | int | bool]] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([indices_by_group[group] for group in sampled])
        replicates.append(regression_metrics(y[index], predicted[index]))

    rows: list[dict[str, Any]] = []
    estimate = regression_metrics(y, predicted)
    for metric in ("mae", "rmse", "r2"):
        values = np.array([float(rep[metric]) for rep in replicates], dtype=np.float64)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "estimate": float(estimate[metric]),
                "ci_low": (
                    float(np.percentile(finite, BOOTSTRAP_LOW)) if finite.size else math.nan
                ),
                "ci_high": (
                    float(np.percentile(finite, BOOTSTRAP_HIGH)) if finite.size else math.nan
                ),
                "n_groups": int(len(unique_groups)),
                "n_samples": int(len(frame)),
                "n_bootstrap": int(n_bootstrap),
                "r2_defined_rate": (
                    float(np.mean([bool(rep["r2_defined"]) for rep in replicates]))
                    if metric == "r2"
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _json_ready(value: object) -> object:
    """Convert NumPy scalars and non-finite floats to strict JSON values."""
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


def save_model_bundle(
    directory: Path | str,
    stem: str,
    estimator: Any,
    card: dict[str, Any],
) -> dict[str, Path]:
    """Persist one estimator and its metadata card.

    ``joblib`` is scikit-learn's documented persistence format for estimators.
    It is **not** safe against untrusted files: only load artifacts this project
    generated locally. The card records the artifact SHA-256 so accidental or
    malicious replacement is detected before deserialisation is trusted.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{stem}.joblib"
    card_path = directory / f"{stem}_card.json"
    joblib.dump(estimator, artifact, compress=3)

    payload = dict(card)
    payload["artifact_file"] = artifact.name
    payload["artifact_sha256"] = sha256_file(artifact)
    card_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"artifact": artifact, "card": card_path}


def _check(checks: list[CheckResult], name: str, function: Any) -> None:
    try:
        detail = function()
    except Exception as exc:  # noqa: BLE001 - every incompatibility is reported
        checks.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(CheckResult(name, True, str(detail)))


def load_model_bundle(
    directory: Path | str,
    stem: str,
    *,
    dataset_summary: dict[str, Any],
) -> tuple[Any, dict[str, Any], tuple[CheckResult, ...]]:
    """Load and verify a persisted model against its dataset and feature schema.

    Every mismatch is collected as a failed check. The estimator is only
    deserialised after the metadata checks pass, because joblib/pickle payloads
    are not safe for untrusted input.
    """
    directory = Path(directory)
    artifact = directory / f"{stem}.joblib"
    card_path = directory / f"{stem}_card.json"
    checks: list[CheckResult] = []

    def card_loaded() -> dict[str, Any]:
        if not card_path.is_file():
            raise ModelError(f"missing model card: {card_path}")
        payload = json.loads(card_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ModelError("model card is not a JSON object")
        return payload

    card: dict[str, Any] = {}
    try:
        card = card_loaded()
    except Exception as exc:  # noqa: BLE001 - report and stop before loading
        checks.append(CheckResult("model_card", False, f"{type(exc).__name__}: {exc}"))
        return None, {}, tuple(checks)
    checks.append(CheckResult("model_card", True, f"loaded {card_path.name}"))

    def feature_schema() -> str:
        if card.get("feature_names") != list(FEATURE_NAMES):
            raise ModelError(
                f"model expects {card.get('feature_names')}, extractor provides {list(FEATURE_NAMES)}"
            )
        expected_version = dataset_summary.get("feature_set_version")
        if card.get("feature_set_version") != expected_version:
            raise ModelError(
                f"model feature set {card.get('feature_set_version')!r} does not match "
                f"dataset {expected_version!r}"
            )
        if int(card.get("n_features", -1)) != len(FEATURE_NAMES):
            raise ModelError("model n_features does not match the feature extractor")
        return "feature names, order and feature-set version match"

    def target_schema() -> str:
        if card.get("target") != TARGET:
            raise ModelError(f"model target is {card.get('target')!r}, expected {TARGET!r}")
        return f"target is {TARGET}"

    def dataset_identity() -> str:
        dataset = card.get("dataset", {})
        expected_generation = dataset_summary.get("generation_config_hash")
        expected_split = dataset_summary.get("split_policy", {}).get("assignment_sha256")
        if dataset.get("generation_config_hash") != expected_generation:
            raise ModelError("model was trained on a different dataset generation hash")
        if dataset.get("split_assignment_sha256") != expected_split:
            raise ModelError("model was trained on a different split assignment")
        return "dataset generation and split hashes match"

    def sklearn_compatible() -> str:
        recorded = str(card.get("sklearn_version", ""))
        current = sklearn_version()
        if recorded.split(".")[:2] != current.split(".")[:2]:
            raise ModelError(
                f"model was persisted with scikit-learn {recorded}, current is {current}"
            )
        return f"scikit-learn {current} (major.minor matches {recorded})"

    def artifact_hash() -> str:
        if not artifact.is_file():
            raise ModelError(f"missing model artifact: {artifact}")
        recorded = card.get("artifact_sha256")
        actual = sha256_file(artifact)
        if recorded != actual:
            raise ModelError("artifact SHA-256 does not match the model card")
        return f"artifact SHA-256 {actual[:12]} matches the card"

    _check(checks, "feature_schema", feature_schema)
    _check(checks, "target_schema", target_schema)
    _check(checks, "dataset_identity", dataset_identity)
    _check(checks, "sklearn_compatibility", sklearn_compatible)
    _check(checks, "artifact_hash", artifact_hash)

    if not all(check.passed for check in checks):
        return None, card, tuple(checks)

    def load_and_smoke_test() -> Any:
        estimator = joblib.load(artifact)
        n_features = getattr(estimator, "n_features_in_", None)
        if n_features != len(FEATURE_NAMES):
            raise ModelError(
                f"estimator expects {n_features} inputs, extractor provides {len(FEATURE_NAMES)}"
            )
        probe = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
        prediction = np.asarray(estimator.predict(probe), dtype=np.float64)
        if prediction.shape != (1,) or not np.isfinite(prediction).all():
            raise ModelError("estimator smoke-test prediction is not a finite scalar")
        return estimator

    try:
        estimator = load_and_smoke_test()
    except Exception as exc:  # noqa: BLE001 - report load/predict failures
        checks.append(CheckResult("artifact_load", False, f"{type(exc).__name__}: {exc}"))
        return None, card, tuple(checks)
    checks.append(
        CheckResult(
            "artifact_load",
            True,
            f"loaded and predicted on a {len(FEATURE_NAMES)}-feature probe",
        )
    )
    return estimator, card, tuple(checks)
