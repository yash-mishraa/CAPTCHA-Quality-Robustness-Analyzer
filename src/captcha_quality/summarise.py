"""Statistical summaries for the target-validation pilot.

Everything here operates on the trial table produced by
:mod:`captcha_quality.pilot`, whose rows are one ``(trial, corruption family,
severity)`` combination. The functions are deliberately explicit about
non-finite values: PSNR is ``+inf`` for a clean image compared with itself, so
aggregations over PSNR exclude infinite values and report how many were dropped.

Statistical choices (documented for the experiment log):

* **Aggregate trend** uses the mean score per severity level and requires strict
  decrease between adjacent levels.
* **Rank association** uses Spearman's rho, computed as the Pearson correlation
  of average ranks (ties get their average rank; ``+inf`` ranks highest).
* **Adjacent separation** is paired by trial, so the same clean source image is
  compared across levels. The effect size is Cohen's d for paired differences
  (mean difference divided by the standard deviation of the differences). A 95%
  interval for the mean difference uses the normal approximation
  ``mean +/- 1.96 * sd / sqrt(n)``; with n >= 30 paired trials per level pair
  this is adequate for a pilot and is labelled as approximate.
* **Violation rate** is the fraction of individual trials whose score moves in
  the *wrong* direction between two adjacent levels. Ties are not violations.
* **Non-degeneracy** uses eta squared (share of variance attributable to the
  severity factor), the range of the level means, and the strongest within-level
  correlation with the trivial ink-coverage feature.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

Z_95 = 1.959963984540054


def _finite(values: pd.Series) -> np.ndarray:
    """Return a float array of finite values."""
    array = values.to_numpy(dtype=float)
    return array[np.isfinite(array)]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks, assigning tied values their average rank."""
    uniques, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    starts = cumulative - counts
    ranks = starts + (counts + 1) / 2.0
    return ranks[inverse]


def rank_correlation(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Spearman rank correlation, ignoring non-finite pairs.

    Implemented directly (Pearson correlation of average ranks) to avoid adding
    SciPy as an explicit dependency for one statistic.
    """
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if int(mask.sum()) < 3:
        return math.nan

    ranked_x = average_ranks(x_values[mask])
    ranked_y = average_ranks(y_values[mask])
    if np.std(ranked_x) == 0.0 or np.std(ranked_y) == 0.0:
        return math.nan
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def level_table(trials: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-family, per-severity descriptive statistics for one metric."""
    rows: list[dict[str, object]] = []
    for (family, severity), group in trials.groupby(["family", "severity"], sort=True):
        values = group[metric].to_numpy(dtype=float)
        finite = _finite(group[metric])
        # A level with no finite scores is +inf only if every score is +inf
        # (the PSNR clean baseline); otherwise its summary is undefined.
        fallback = math.inf if values.size and np.all(np.isposinf(values)) else math.nan
        row: dict[str, object] = {
            "family": family,
            "metric": metric,
            "severity": int(severity),
            "parameter": float(group["parameter"].iloc[0]),
            "unit": str(group["unit"].iloc[0]),
            "n_total": int(values.size),
            "n_finite": int(finite.size),
            "n_infinite": int(values.size - finite.size),
            "mean": float(finite.mean()) if finite.size else fallback,
            "sd": float(finite.std(ddof=1)) if finite.size > 1 else math.nan,
            "median": float(np.median(finite)) if finite.size else fallback,
            "min": float(finite.min()) if finite.size else fallback,
            "max": float(finite.max()) if finite.size else fallback,
            "clipped_fraction_mean": float(group["clipped_fraction"].mean()),
            "realised_noise_std_mean": float(group["realised_noise_std"].mean()),
            "n_identical": int(group["identical"].sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_trend(trials: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Check whether level means decrease strictly with severity, per family."""
    rows: list[dict[str, object]] = []
    for family, group in trials.groupby("family", sort=True):
        means: dict[int, float] = {}
        for severity, level_group in group.groupby("severity", sort=True):
            finite = _finite(level_group[metric])
            if finite.size:
                means[int(severity)] = float(finite.mean())
            elif np.all(np.isposinf(level_group[metric].to_numpy(dtype=float))):
                means[int(severity)] = math.inf
        levels = sorted(means)
        steps = [
            {"from": a, "to": b, "from_mean": means[a], "to_mean": means[b]}
            for a, b in zip(levels, levels[1:], strict=False)
        ]
        for step in steps:
            step["decreasing"] = step["to_mean"] < step["from_mean"]
        rows.append(
            {
                "family": family,
                "metric": metric,
                "n_levels": len(levels),
                "steps": steps,
                "all_strictly_decreasing": all(step["decreasing"] for step in steps),
                "n_failed_steps": sum(not step["decreasing"] for step in steps),
            }
        )
    return pd.DataFrame(rows)


def adjacent_separation(trials: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Paired adjacent-level differences and effect sizes, per family."""
    rows: list[dict[str, object]] = []
    for family, group in trials.groupby("family", sort=True):
        wide = group.pivot_table(
            index="trial_id", columns="severity", values=metric, aggfunc="first"
        )
        levels = sorted(int(level) for level in wide.columns)
        for lower, upper in zip(levels, levels[1:], strict=False):
            differences = (wide[upper] - wide[lower]).to_numpy(dtype=float)
            valid = differences[np.isfinite(differences)]
            n_valid = int(valid.size)
            n_excluded = int(differences.size - n_valid)
            if n_valid < 2:
                rows.append(
                    {
                        "family": family,
                        "metric": metric,
                        "lower": lower,
                        "upper": upper,
                        "n_pairs": n_valid,
                        "n_excluded": n_excluded,
                        "mean_diff": math.nan,
                        "sd_diff": math.nan,
                        "paired_d": math.nan,
                        "ci95_low": math.nan,
                        "ci95_high": math.nan,
                        "expected_direction_fraction": math.nan,
                        "evaluable": False,
                    }
                )
                continue

            mean_diff = float(valid.mean())
            sd_diff = float(valid.std(ddof=1))
            if sd_diff == 0.0:
                paired_d = -math.inf if mean_diff < 0 else math.inf if mean_diff > 0 else 0.0
                half_width = 0.0
            else:
                paired_d = mean_diff / sd_diff
                half_width = Z_95 * sd_diff / math.sqrt(n_valid)
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "lower": lower,
                    "upper": upper,
                    "n_pairs": n_valid,
                    "n_excluded": n_excluded,
                    "mean_diff": mean_diff,
                    "sd_diff": sd_diff,
                    "paired_d": paired_d,
                    "ci95_low": mean_diff - half_width,
                    "ci95_high": mean_diff + half_width,
                    "expected_direction_fraction": float(np.mean(valid < 0)),
                    "evaluable": True,
                }
            )
    return pd.DataFrame(rows)


def violation_rates(trials: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Fraction of trials whose score increases between adjacent levels."""
    return (
        adjacent_separation(trials, metric)
        .assign(violation_rate=lambda frame: 1.0 - frame["expected_direction_fraction"])
        .loc[:, ["family", "metric", "lower", "upper", "n_pairs", "violation_rate"]]
    )


def variance_decomposition(trials: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Eta squared and level-mean range for the severity factor, per family."""
    rows: list[dict[str, object]] = []
    for family, group in trials.groupby("family", sort=True):
        finite_rows = group.loc[np.isfinite(group[metric].to_numpy(dtype=float))]
        values = finite_rows[metric].to_numpy(dtype=float)
        if values.size == 0:
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "n": 0,
                    "n_levels": 0,
                    "overall_sd": math.nan,
                    "level_mean_range": math.nan,
                    "eta_squared": math.nan,
                }
            )
            continue

        grand_mean = float(values.mean())
        total_ss = float(np.sum((values - grand_mean) ** 2))
        between_ss = float(
            sum(
                len(level_values) * (level_values.mean() - grand_mean) ** 2
                for level_values in (
                    _finite(level_group[metric])
                    for _, level_group in finite_rows.groupby("severity")
                )
            )
        )
        level_means = [
            float(_finite(level_group[metric]).mean())
            for _, level_group in finite_rows.groupby("severity")
        ]
        rows.append(
            {
                "family": family,
                "metric": metric,
                "n": int(values.size),
                "n_levels": len(level_means),
                "overall_sd": float(values.std(ddof=1)) if values.size > 1 else math.nan,
                "level_mean_range": max(level_means) - min(level_means),
                "eta_squared": between_ss / total_ss if total_ss > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def ink_correlation(trials: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Strongest within-level Pearson correlation between score and ink coverage.

    A metric that merely restates how much of the canvas is covered by ink would
    show a near-perfect correlation at every severity level. Only levels with at
    least three distinct ink values contribute.
    """
    rows: list[dict[str, object]] = []
    for family, group in trials.groupby("family", sort=True):
        correlations: list[tuple[int, float]] = []
        for severity, level_group in group.groupby("severity", sort=True):
            finite_rows = level_group.loc[np.isfinite(level_group[metric].to_numpy(dtype=float))]
            if finite_rows["ink_fraction"].nunique() < 3 or finite_rows[metric].nunique() < 2:
                continue
            correlation = finite_rows["ink_fraction"].corr(finite_rows[metric])
            if pd.notna(correlation):
                correlations.append((int(severity), float(correlation)))
        strongest = max(correlations, key=lambda item: abs(item[1])) if correlations else None
        rows.append(
            {
                "family": family,
                "metric": metric,
                "max_abs_ink_correlation": abs(strongest[1]) if strongest else math.nan,
                "signed_ink_correlation": strongest[1] if strongest else math.nan,
                "at_severity": strongest[0] if strongest else -1,
                "n_levels_used": len(correlations),
            }
        )
    return pd.DataFrame(rows)


def stability_table(trials: pd.DataFrame, metric: str, group_column: str) -> pd.DataFrame:
    """Per-group rank correlation and level means, to test cross-condition stability."""
    rows: list[dict[str, object]] = []
    for family, family_group in trials.groupby("family", sort=True):
        for group_value, group in family_group.groupby(group_column, sort=True):
            finite_rows = group.loc[np.isfinite(group[metric].to_numpy(dtype=float))]
            rho = rank_correlation(finite_rows["severity"], finite_rows[metric])
            level_means = [
                float(_finite(level_group[metric]).mean())
                for _, level_group in finite_rows.groupby("severity")
                if _finite(level_group[metric]).size
            ]
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "group_column": group_column,
                    "group": group_value,
                    "n": int(len(finite_rows)),
                    "spearman_rho": rho,
                    "level_means_decreasing": all(
                        later < earlier
                        for earlier, later in zip(level_means, level_means[1:], strict=False)
                    ),
                    "level_mean_min": min(level_means) if level_means else math.nan,
                    "level_mean_max": max(level_means) if level_means else math.nan,
                }
            )
    return pd.DataFrame(rows)


def summarise(trials: pd.DataFrame, metrics: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Build every summary table needed by the gate and the report."""
    levels = pd.concat([level_table(trials, metric) for metric in metrics], ignore_index=True)
    separation = pd.concat(
        [adjacent_separation(trials, metric) for metric in metrics], ignore_index=True
    )
    violations = pd.concat(
        [violation_rates(trials, metric) for metric in metrics], ignore_index=True
    )
    variance = pd.concat(
        [variance_decomposition(trials, metric) for metric in metrics], ignore_index=True
    )
    ink = pd.concat([ink_correlation(trials, metric) for metric in metrics], ignore_index=True)
    font_stability = pd.concat(
        [stability_table(trials, metric, "font") for metric in metrics], ignore_index=True
    )
    seed_stability = pd.concat(
        [stability_table(trials, metric, "seed") for metric in metrics], ignore_index=True
    )
    trend = pd.concat([aggregate_trend(trials, metric) for metric in metrics], ignore_index=True)
    rho_rows = [
        {
            "family": family,
            "metric": metric,
            "spearman_rho": rank_correlation(group["severity"], group[metric]),
            "n": int(len(group)),
        }
        for metric in metrics
        for family, group in trials.groupby("family", sort=True)
    ]
    return {
        "levels": levels,
        "trend": trend,
        "separation": separation,
        "violations": violations,
        "variance": variance,
        "ink": ink,
        "font_stability": font_stability,
        "seed_stability": seed_stability,
        "family_rho": pd.DataFrame(rho_rows),
    }
