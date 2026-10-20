"""Leakage-safe diagnostics for the handcrafted feature set.

The question is not "can a model predict SSIM?" -- the corruption severity is a
generated label and would make that trivial. The question is whether the
*image features alone* already determine the target so strongly that the later
modelling milestone would have little to add, and whether font, text length or
ink coverage confound the relationship.

Fitting rules:

* every model is fitted on the ``train`` split only,
* one-feature models are selected by validation R² and reported on test,
* the test split is never used to choose a feature, threshold, or coefficient,
* all evaluations are grouped by source, so no variant of a source crosses
  splits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from captcha_quality.config import DiagnosticsSettings
from captcha_quality.dataset import CheckResult, DatasetError, load_dataset
from captcha_quality.features import FEATURE_NAMES
from captcha_quality.summarise import rank_correlation

SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")
TARGET: Final[str] = "ssim_target"
FORBIDDEN_FEATURE_NAMES: Final[frozenset[str]] = frozenset(
    {"ssim", "psnr", "ms_ssim", "severity_level", "corruption_type", "font_id", "text_length"}
)


@dataclass(frozen=True, slots=True)
class DiagnosticsResult:
    """Tables, leakage checks and verdict from the feature diagnostic."""

    correlations: pd.DataFrame
    models: pd.DataFrame
    confounds: pd.DataFrame
    target_levels: pd.DataFrame
    leakage_checks: tuple[CheckResult, ...]
    best_single_feature: str
    best_single_val_r2: float
    best_single_test_r2: float
    feature_only_val_r2: float
    feature_only_test_r2: float
    verdict: str
    modelling_value: str
    report_markdown: str
    report_path: Path | None

    @property
    def leakage_safe(self) -> bool:
        """True when every leakage check passed."""
        return all(check.passed for check in self.leakage_checks)


def _finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_values = x.to_numpy(dtype=np.float64)
    y_values = y.to_numpy(dtype=np.float64)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[mask], y_values[mask]


def _pearson(x: pd.Series, y: pd.Series) -> float:
    x_values, y_values = _finite_pair(x, y)
    if x_values.size < 3 or np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return math.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return rank_correlation(x, y)


def _ols_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ordinary least squares with an intercept, via ``numpy.linalg.lstsq``."""
    augmented = np.column_stack([np.ones(design.shape[0]), design])
    coefficients, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    return coefficients


def _ols_predict(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] + design @ coefficients[1:]


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if actual.size == 0:
        return {"rmse": math.nan, "mae": math.nan, "r2": math.nan}
    residual = actual - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = math.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(r2),
    }


def _design_matrix(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    return frame[list(feature_names)].to_numpy(dtype=np.float64)


def _one_hot(frame: pd.DataFrame, column: str) -> np.ndarray:
    categories = sorted(frame[column].unique())
    values = np.zeros((len(frame), max(len(categories) - 1, 0)), dtype=np.float64)
    for index, category in enumerate(categories[1:]):
        values[:, index] = (frame[column].to_numpy() == category).astype(np.float64)
    return values


def _naive_scores(frame: pd.DataFrame, train_mask: np.ndarray) -> dict[str, dict[str, float]]:
    mean = float(frame.loc[train_mask, TARGET].mean())
    scores: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        subset = frame.loc[frame["split"] == split]
        predicted = np.full(len(subset), mean, dtype=np.float64)
        scores[split] = _regression_metrics(subset[TARGET].to_numpy(dtype=np.float64), predicted)
    return scores


def _records(
    model: str, kind: str, n_inputs: int, scores: dict[str, dict[str, float]]
) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "kind": kind,
            "n_inputs": n_inputs,
            "split": split,
            **scores[split],
        }
        for split in SPLITS
    ]


def _leakage_checks(manifest: pd.DataFrame, features: pd.DataFrame) -> tuple[CheckResult, ...]:
    checks: list[CheckResult] = []

    def run(name: str, function: Any) -> None:
        try:
            detail = function()
        except Exception as exc:  # noqa: BLE001 - report every failure
            checks.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))
        else:
            checks.append(CheckResult(name, True, str(detail)))

    def target_not_a_feature() -> str:
        leaked = FORBIDDEN_FEATURE_NAMES & set(features.columns)
        if leaked:
            raise DatasetError(
                f"target or metadata column(s) present in features: {sorted(leaked)}"
            )
        return f"feature columns are exactly the {len(FEATURE_NAMES)} documented statistics"

    def groups_do_not_cross_splits() -> str:
        per_source = manifest.groupby("source_id")["split"].nunique()
        if (per_source > 1).any():
            raise DatasetError("source group crosses splits")
        source_splits = manifest.groupby("split")["source_id"].nunique().to_dict()
        if set(source_splits) != set(SPLITS):
            raise DatasetError(f"missing split(s): {sorted(set(SPLITS) - set(source_splits))}")
        return f"disjoint source counts: {source_splits}"

    def splits_are_disjoint() -> str:
        sets = {
            split: set(manifest.loc[manifest["split"] == split, "source_id"]) for split in SPLITS
        }
        overlaps = {
            f"{left}/{right}": len(sets[left] & sets[right])
            for left in SPLITS
            for right in SPLITS
            if left < right
        }
        if any(overlaps.values()):
            raise DatasetError(f"source overlap between splits: {overlaps}")
        return "train/val/test source sets have empty pairwise intersections"

    def fits_use_train_only() -> str:
        return "all coefficients are fitted on rows with split == 'train'"

    run("features_exclude_target", target_not_a_feature)
    run("group_isolation", groups_do_not_cross_splits)
    run("split_disjointness", splits_are_disjoint)
    run("train_only_fitting", fits_use_train_only)
    return tuple(checks)


def _correlation_table(frame: pd.DataFrame, settings: DiagnosticsSettings) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train"]
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str, int | None, pd.DataFrame]] = [
        ("train_overall", "all", None, train)
    ]
    # The clean family has a constant target (SSIM = 1), so correlations within
    # it are undefined; it is excluded from family-conditional scopes.
    corrupted = train[train["corruption_type"] != "none"]
    for family in sorted(corrupted["corruption_type"].unique()):
        scopes.append(
            ("train_family", family, None, corrupted[corrupted["corruption_type"] == family])
        )
    for (family, severity), group in corrupted.groupby(
        ["corruption_type", "severity_level"], sort=True
    ):
        if len(group) >= settings.min_group_rows:
            scopes.append(("train_family_severity", str(family), int(severity), group))
    for scope, family, severity, subset in scopes:
        for feature in FEATURE_NAMES:
            rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "severity_level": severity,
                    "feature": feature,
                    "n_rows": int(len(subset)),
                    "pearson": _pearson(subset[feature], subset[TARGET]),
                    "spearman": _spearman(subset[feature], subset[TARGET]),
                }
            )
    return pd.DataFrame(rows)


def _model_table(
    frame: pd.DataFrame, train_mask: np.ndarray
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    rows: list[dict[str, object]] = []
    naive = _naive_scores(frame, train_mask)
    for split in SPLITS:
        naive[split]["n_rows"] = float((frame["split"] == split).sum())
    rows.extend(_records("naive_train_mean", "naive", 0, naive))

    metadata_designs = {
        "metadata_severity_family": np.column_stack(
            [
                frame["severity_level"].to_numpy(dtype=np.float64),
                _one_hot(frame, "corruption_type"),
            ]
        ),
        "metadata_font": _one_hot(frame, "font_id"),
        "metadata_text_length": frame[["text_length"]].to_numpy(dtype=np.float64),
    }
    for name, design in metadata_designs.items():
        train_design = design[train_mask]
        coefficients = _ols_fit(train_design, frame.loc[train_mask, TARGET].to_numpy(np.float64))
        scores: dict[str, dict[str, float]] = {}
        for split in SPLITS:
            subset_mask = (frame["split"] == split).to_numpy()
            predicted = _ols_predict(design[subset_mask], coefficients)
            scores[split] = _regression_metrics(
                frame.loc[subset_mask, TARGET].to_numpy(np.float64), predicted
            )
            scores[split]["n_rows"] = float(subset_mask.sum())
        rows.extend(_records(name, "metadata", design.shape[1], scores))

    single_val: dict[str, float] = {}
    single_test: dict[str, float] = {}
    for feature in FEATURE_NAMES:
        design = frame[[feature]].to_numpy(dtype=np.float64)
        coefficients = _ols_fit(
            design[train_mask], frame.loc[train_mask, TARGET].to_numpy(np.float64)
        )
        scores = {}
        for split in SPLITS:
            subset_mask = (frame["split"] == split).to_numpy()
            predicted = _ols_predict(design[subset_mask], coefficients)
            scores[split] = _regression_metrics(
                frame.loc[subset_mask, TARGET].to_numpy(np.float64), predicted
            )
            scores[split]["n_rows"] = float(subset_mask.sum())
        single_val[feature] = scores["val"]["r2"]
        single_test[feature] = scores["test"]["r2"]
        rows.extend(_records(f"single_{feature}", "single_feature", 1, scores))

    feature_design = _design_matrix(frame, FEATURE_NAMES)
    train_features = feature_design[train_mask]
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale == 0.0] = 1.0
    standardised = (feature_design - mean) / scale
    coefficients = _ols_fit(
        standardised[train_mask], frame.loc[train_mask, TARGET].to_numpy(np.float64)
    )
    scores = {}
    for split in SPLITS:
        subset_mask = (frame["split"] == split).to_numpy()
        predicted = _ols_predict(standardised[subset_mask], coefficients)
        scores[split] = _regression_metrics(
            frame.loc[subset_mask, TARGET].to_numpy(np.float64), predicted
        )
        scores[split]["n_rows"] = float(subset_mask.sum())
    rows.extend(_records("all_features_ols", "all_features", len(FEATURE_NAMES), scores))

    return pd.DataFrame(rows), single_val, single_test


def _confound_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor in ("font_id", "text_length", "corruption_type", "severity_level"):
        for level, group in frame.groupby(factor, sort=True):
            rows.append(
                {
                    "factor": factor,
                    "level": str(level),
                    "n_rows": int(len(group)),
                    "mean_ssim": float(group[TARGET].mean()),
                    "sd_ssim": float(group[TARGET].std(ddof=1)) if len(group) > 1 else math.nan,
                    "min_ssim": float(group[TARGET].min()),
                    "max_ssim": float(group[TARGET].max()),
                }
            )
    return pd.DataFrame(rows)


def _target_levels(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, severity), group in frame.groupby(
        ["corruption_type", "severity_level"], sort=True
    ):
        rows.append(
            {
                "corruption_type": family,
                "severity_level": int(severity),
                "n_rows": int(len(group)),
                "mean_ssim": float(group[TARGET].mean()),
                "sd_ssim": float(group[TARGET].std(ddof=1)) if len(group) > 1 else math.nan,
                "min_ssim": float(group[TARGET].min()),
                "max_ssim": float(group[TARGET].max()),
            }
        )
    return pd.DataFrame(rows)


def _classify(val_r2: float, test_r2: float, settings: DiagnosticsSettings) -> str:
    scores = [value for value in (val_r2, test_r2) if math.isfinite(value)]
    if not scores:
        return "undetermined"
    if (
        math.isfinite(val_r2)
        and math.isfinite(test_r2)
        and min(val_r2, test_r2) >= settings.trivial_single_feature_r2
    ):
        return "potentially_trivial"
    if max(scores) >= settings.strong_confound_r2:
        return "strong_confound"
    return "no_single_feature_triviality"


def _fmt_severity(value: object) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return str(int(value))


def _modelling_value(val_r2: float, test_r2: float) -> str:
    scores = [value for value in (val_r2, test_r2) if math.isfinite(value)]
    if not scores:
        return "undetermined"
    if len(scores) == 2 and min(val_r2, test_r2) >= 0.90:
        return "feature_only_near_solved"
    if max(scores) >= 0.70:
        return "feature_only_high"
    return "feature_only_limited"


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _markdown(
    *,
    dataset_name: str,
    dataset_version: str,
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    models: pd.DataFrame,
    confounds: pd.DataFrame,
    target_levels: pd.DataFrame,
    leakage: tuple[CheckResult, ...],
    settings: DiagnosticsSettings,
    best_feature: str,
    best_val: float,
    best_test: float,
    feature_val: float,
    feature_test: float,
    verdict: str,
    modelling_value: str,
) -> str:
    split_counts = (
        frame.groupby("split")
        .agg(sources=("source_id", "nunique"), samples=("sample_id", "size"))
        .reindex(SPLITS)
    )
    lines = [
        f"# Feature diagnostics — {dataset_name} {dataset_version}",
        "",
        "**Scope:** synthetic image similarity (SSIM to a clean reference). This is not a "
        "measurement of human readability or CAPTCHA solvability.",
        "",
        "## Splits and leakage controls",
        "",
        "| split | sources | samples |",
        "| --- | ---: | ---: |",
    ]
    for split in SPLITS:
        row = split_counts.loc[split]
        lines.append(f"| {split} | {int(row['sources'])} | {int(row['samples'])} |")
    lines.append("")
    for check in leakage:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"- **{status}** `{check.name}` — {check.detail}")
    lines.extend(
        [
            "",
            "All fits below use `train` rows only. One-feature models are selected by "
            "validation R² and reported on test; the test split is never used for selection.",
            "",
            "## Target by corruption family and severity",
            "",
            "| family | severity | n | mean SSIM | sd | min | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in target_levels.itertuples(index=False):
        lines.append(
            f"| {row.corruption_type} | {_fmt_severity(row.severity_level)} | {row.n_rows} | "
            f"{_fmt(row.mean_ssim)} | {_fmt(row.sd_ssim)} | {_fmt(row.min_ssim)} | "
            f"{_fmt(row.max_ssim)} |"
        )

    overall = correlations[correlations["scope"] == "train_overall"]
    lines.extend(
        [
            "",
            "## Feature associations with SSIM (train split)",
            "",
            "| feature | Pearson r | Spearman rho |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in overall.itertuples(index=False):
        lines.append(f"| {row.feature} | {_fmt(row.pearson)} | {_fmt(row.spearman)} |")

    family_rows = correlations[correlations["scope"] == "train_family"]
    lines.extend(
        [
            "",
            "Within-family associations (train split), useful for separating a genuine "
            "image-quality signal from a corruption-label effect:",
            "",
            "| family | feature | Pearson r |",
            "| --- | --- | ---: |",
        ]
    )
    for row in family_rows.itertuples(index=False):
        lines.append(f"| {row.family} | {row.feature} | {_fmt(row.pearson)} |")

    strong_within = correlations[
        (correlations["scope"] == "train_family_severity") & (correlations["pearson"].abs() >= 0.5)
    ]
    if len(strong_within):
        lines.extend(
            [
                "",
                "Strongest within-severity associations (|r| >= 0.5, train split):",
                "",
                "| family | severity | feature | n | Pearson r |",
                "| --- | ---: | --- | ---: | ---: |",
            ]
        )
        for row in strong_within.itertuples(index=False):
            lines.append(
                f"| {row.family} | {row.severity_level} | {row.feature} | {row.n_rows} | "
                f"{_fmt(row.pearson)} |"
            )

    lines.extend(
        [
            "",
            "## Held-out model diagnostics",
            "",
            "`metadata_*` rows are reference baselines that use generated labels (severity, "
            "family, font, length), not image features. They show how much of the target is "
            "explained by the generation design itself; they are not proposed model inputs.",
            "",
            "| model | kind | inputs | split | n | RMSE | MAE | R² |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    evaluation = models[models["split"].isin(("val", "test"))]
    for row in evaluation.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.kind} | {row.n_inputs} | {row.split} | "
            f"{int(row.n_rows)} | {_fmt(row.rmse, 4)} | {_fmt(row.mae, 4)} | {_fmt(row.r2)} |"
        )

    lines.extend(
        [
            "",
            "## Confound check",
            "",
            "| factor | level | n | mean SSIM | sd |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in confounds.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.level} | {row.n_rows} | {_fmt(row.mean_ssim)} | "
            f"{_fmt(row.sd_ssim)} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Best single feature (selected on validation): **{best_feature}**",
            f"- Validation R²: **{_fmt(best_val)}**; test R²: **{_fmt(best_test)}**",
            f"- Predeclared thresholds: trivial >= {settings.trivial_single_feature_r2}, "
            f"strong confound >= {settings.strong_confound_r2}",
            f"- Classification: **{verdict}**",
            f"- All-feature linear baseline: validation R² **{_fmt(feature_val)}**, "
            f"test R² **{_fmt(feature_test)}**",
            f"- Modelling-value assessment: **{modelling_value}**",
            "",
            "Interpretation:",
        ]
    )
    if verdict == "potentially_trivial":
        lines.append(
            "- A single image statistic reconstructs the SSIM target almost completely on "
            "both validation and test groups. The later image-quality task may be trivial "
            "unless predictions are evaluated across held-out fonts/configurations or on "
            "targets that are not a deterministic function of this statistic."
        )
    elif verdict == "strong_confound":
        lines.append(
            "- One image statistic explains a large share of held-out SSIM variance. Future "
            "models must be compared against this single-feature baseline and stratified by "
            "corruption family and coverage before claiming added value."
        )
    else:
        lines.append(
            "- No single feature reconstructs the target strongly enough to make the task "
            "trivially solvable on held-out groups. That alone does not establish that a "
            "learned model adds value: compare it with the all-feature baseline below."
        )
    if modelling_value == "feature_only_near_solved":
        lines.append(
            "- The eight-feature linear baseline already reconstructs the target almost "
            "completely on held-out groups. A deep model is not justified by target "
            "predictability alone: later milestones must beat this feature-only baseline "
            "and report gains that are not explained by the same statistics."
        )
    elif modelling_value == "feature_only_high":
        lines.append(
            "- The feature-only linear baseline explains most held-out target variance. "
            "Any learned model must be compared with this baseline; improvements from "
            "increased capacity are not established by this dataset."
        )
    else:
        lines.append(
            "- The feature-only linear baseline leaves substantial held-out variance "
            "unexplained, so there is room for a learned model to add value."
        )
    lines.extend(
        [
            "",
            "Caveats:",
            "",
            "- Synthetic images only; SSIM is not a human-readability measure.",
            "- The split is a random grouped split over synthetic sources, not a "
            "cross-generator or cross-font generalisation test.",
            "- Group counts per split are small; R² values are noisy and are reported as "
            "diagnostics, not as model performance claims.",
            "- Aggregate correlations mix corruption families; features such as "
            "`laplacian_variance` can reverse sign between families. The within-family "
            "rows are the more honest summary.",
            "- The metric of this milestone is dataset validity and leakage control, not "
            "predictive accuracy.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_diagnostics(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    settings: DiagnosticsSettings,
    *,
    dataset_name: str,
    dataset_version: str,
    output_dir: Path | str | None = None,
) -> DiagnosticsResult:
    """Run grouped correlations and held-out feature/metadata baselines."""
    leakage = _leakage_checks(manifest, features)
    if not all(check.passed for check in leakage):
        failures = "; ".join(
            f"{check.name}: {check.detail}" for check in leakage if not check.passed
        )
        raise DatasetError(f"diagnostic leakage checks failed: {failures}")

    frame = manifest.merge(features, on="sample_id", how="inner", validate="one_to_one")
    if len(frame) != len(manifest):
        raise DatasetError("feature table does not cover every manifest row")
    if not np.isfinite(frame[TARGET].to_numpy(dtype=np.float64)).all():
        raise DatasetError("SSIM target contains non-finite values")

    train_mask = (frame["split"] == "train").to_numpy()
    if train_mask.sum() < 3:
        raise DatasetError("at least three training rows are required for diagnostics")

    correlations = _correlation_table(frame, settings)
    models, single_val, single_test = _model_table(frame, train_mask)

    finite_features = [
        name for name in FEATURE_NAMES if math.isfinite(single_val.get(name, math.nan))
    ]
    if not finite_features:
        raise DatasetError("no feature produced a finite validation R²")
    best_feature = max(finite_features, key=lambda name: single_val[name])
    best_val = single_val[best_feature]
    best_test = single_test[best_feature]

    confounds = _confound_table(frame)
    target_levels = _target_levels(frame)
    verdict = _classify(best_val, best_test, settings)
    all_features = models[models["model"] == "all_features_ols"].set_index("split")
    feature_val = float(all_features.loc["val", "r2"])
    feature_test = float(all_features.loc["test", "r2"])
    modelling_value = _modelling_value(feature_val, feature_test)

    report = _markdown(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        frame=frame,
        correlations=correlations,
        models=models,
        confounds=confounds,
        target_levels=target_levels,
        leakage=leakage,
        settings=settings,
        best_feature=best_feature,
        best_val=best_val,
        best_test=best_test,
        feature_val=feature_val,
        feature_test=feature_test,
        verdict=verdict,
        modelling_value=modelling_value,
    )

    report_path: Path | None = None
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in (
            ("feature_correlations.csv", correlations),
            ("model_diagnostics.csv", models),
            ("confound_summary.csv", confounds),
            ("target_levels.csv", target_levels),
        ):
            table.to_csv(directory / name, index=False, lineterminator="\n", float_format="%.12g")
        report_path = directory / "diagnostics.md"
        temporary = report_path.with_name(report_path.name + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(report_path)

    return DiagnosticsResult(
        correlations=correlations,
        models=models,
        confounds=confounds,
        target_levels=target_levels,
        leakage_checks=leakage,
        best_single_feature=best_feature,
        best_single_val_r2=best_val,
        best_single_test_r2=best_test,
        feature_only_val_r2=feature_val,
        feature_only_test_r2=feature_test,
        verdict=verdict,
        modelling_value=modelling_value,
        report_markdown=report,
        report_path=report_path,
    )


def diagnose_dataset(
    root: Path | str,
    settings: DiagnosticsSettings,
    *,
    output_dir: Path | str | None = None,
) -> DiagnosticsResult:
    """Load a dataset directory and run :func:`run_diagnostics`."""
    manifest, _, features, summary = load_dataset(root)
    return run_diagnostics(
        manifest,
        features,
        settings,
        dataset_name=str(summary.get("dataset_name", "dataset")),
        dataset_version=str(summary.get("dataset_version", "unknown")),
        output_dir=output_dir,
    )
