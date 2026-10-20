"""Tests for the statistical summaries used by the gate."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from captcha_quality.summarise import (
    adjacent_separation,
    aggregate_trend,
    average_ranks,
    ink_correlation,
    level_table,
    rank_correlation,
    stability_table,
    summarise,
    variance_decomposition,
    violation_rates,
)


def make_trials(
    values_by_trial: dict[str, list[float]],
    *,
    family: str = "gaussian_blur",
    metric: str = "ssim",
) -> pd.DataFrame:
    """Build a minimal trial table accepted by every summary function."""
    rows = []
    for index, (trial_id, values) in enumerate(values_by_trial.items()):
        for severity, value in enumerate(values):
            rows.append(
                {
                    "trial_id": trial_id,
                    "font": "arial",
                    "text": "AB23",
                    "seed": 0,
                    "family": family,
                    "severity": severity,
                    "parameter": float(severity),
                    "unit": "px",
                    metric: value,
                    "ink_fraction": 0.10 + 0.01 * index,
                    "clipped_fraction": 0.0,
                    "realised_noise_std": 0.0,
                    "identical": severity == 0,
                }
            )
    return pd.DataFrame(rows)


def test_average_ranks_handles_ties() -> None:
    ranks = average_ranks(np.array([10.0, 10.0, 20.0, 30.0]))
    assert ranks.tolist() == [1.5, 1.5, 3.0, 4.0]


def test_rank_correlation_detects_perfect_orders() -> None:
    assert rank_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert rank_correlation([1, 1, 2, 2], [5, 5, 6, 6]) == pytest.approx(1.0)
    assert math.isnan(rank_correlation([1, 1, 1], [1, 2, 3]))


def test_level_table_reports_infinite_psnr_baseline() -> None:
    trials = make_trials(
        {"t0": [math.inf, 30.0, 20.0], "t1": [math.inf, 28.0, 18.0]}, metric="psnr"
    )
    table = level_table(trials, "psnr")
    clean_level = table.loc[table["severity"] == 0].iloc[0]
    assert clean_level["n_infinite"] == 2
    assert clean_level["n_finite"] == 0
    assert math.isinf(clean_level["mean"])
    assert table.loc[table["severity"] == 1, "mean"].iloc[0] == pytest.approx(29.0)


def test_aggregate_trend_requires_strict_decrease() -> None:
    decreasing = make_trials({"t0": [1.0, 0.8, 0.5], "t1": [1.0, 0.7, 0.4]})
    assert aggregate_trend(decreasing, "ssim").iloc[0]["all_strictly_decreasing"]

    flat = make_trials({"t0": [1.0, 0.8, 0.8]})
    assert not aggregate_trend(flat, "ssim").iloc[0]["all_strictly_decreasing"]


def test_variance_decomposition_is_one_for_pure_severity_effect() -> None:
    trials = make_trials({"t0": [1.0, 0.5], "t1": [1.0, 0.5]})
    row = variance_decomposition(trials, "ssim").iloc[0]
    assert row["eta_squared"] == pytest.approx(1.0)
    assert row["level_mean_range"] == pytest.approx(0.5)


def test_violation_rate_detects_reversals() -> None:
    trials = make_trials({"t0": [1.0, 0.5], "t1": [0.5, 1.0]})
    row = violation_rates(trials, "ssim").iloc[0]
    assert row["violation_rate"] == pytest.approx(0.5)
    assert row["n_pairs"] == 2


def test_adjacent_separation_handles_zero_variance_differences() -> None:
    trials = make_trials({"t0": [1.0, 0.5], "t1": [0.8, 0.3]})
    row = adjacent_separation(trials, "ssim").iloc[0]
    assert row["mean_diff"] == pytest.approx(-0.5)
    assert row["sd_diff"] == pytest.approx(0.0)
    assert row["paired_d"] == -math.inf
    assert row["ci95_low"] == pytest.approx(-0.5)
    assert row["ci95_high"] == pytest.approx(-0.5)


def test_separation_excludes_non_finite_differences() -> None:
    trials = make_trials({"t0": [math.inf, 30.0], "t1": [math.inf, 28.0]}, metric="psnr")
    row = adjacent_separation(trials, "psnr").iloc[0]
    assert not row["evaluable"]
    assert row["n_pairs"] == 0
    assert row["n_excluded"] == 2
    assert math.isnan(row["paired_d"])


def test_stability_table_groups_by_font() -> None:
    trials = make_trials({"t0": [1.0, 0.6], "t1": [1.0, 0.4]})
    table = stability_table(trials, "ssim", "font")
    assert len(table) == 1
    assert table.iloc[0]["group"] == "arial"
    assert table.iloc[0]["spearman_rho"] < -0.9
    assert table.iloc[0]["level_means_decreasing"]


def test_ink_correlation_reports_signed_strength() -> None:
    trials = make_trials({"t0": [1.0, 0.9], "t1": [1.0, 0.8], "t2": [1.0, 0.7]})
    row = ink_correlation(trials, "ssim").iloc[0]
    # Severity 0 is constant (all 1.0) and is skipped; severity 1 has three
    # distinct ink values and a perfectly decreasing score.
    assert row["at_severity"] == 1
    assert row["signed_ink_correlation"] == pytest.approx(-1.0)
    assert row["max_abs_ink_correlation"] == pytest.approx(1.0)


def test_summarise_returns_every_table() -> None:
    trials = make_trials({"t0": [1.0, 0.6], "t1": [1.0, 0.5]})
    trials["psnr"] = [40.0, 26.0, 38.0, 24.0]
    tables = summarise(trials, ("ssim", "psnr"))
    assert set(tables) == {
        "levels",
        "trend",
        "separation",
        "violations",
        "variance",
        "ink",
        "font_stability",
        "seed_stability",
        "family_rho",
    }
    assert len(tables["family_rho"]) == 2
