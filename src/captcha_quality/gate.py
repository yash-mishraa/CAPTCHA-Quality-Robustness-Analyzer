"""Predeclared acceptance gate for candidate image-quality targets.

The thresholds in :class:`GateThresholds` are declared in
``configs/pilot_target_validation.toml`` and documented in
``docs/METHODOLOGY.md`` **before** the pilot is run. Changing a threshold after
seeing results invalidates the gate for that run and requires a new experiment
ID.

Verdict semantics
-----------------
``PASS``
    Every criterion is met.
``CONDITIONAL``
    Direction, aggregate trend, non-degeneracy, and sanity checks hold, but at
    least one separation/stability criterion does not. The metric remains usable
    with the stated limitation.
``FAIL``
    A prerequisite criterion (direction, aggregate trend, non-degeneracy, or
    clean-reference sanity) fails, so the metric cannot serve as an ordinal
    quality target under these conditions.
``DEFERRED``
    Not evaluated in this milestone. Recorded with a reason.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from captcha_quality.quality_metrics import DEFERRED_METRICS

PASS: Final[str] = "PASS"
CONDITIONAL: Final[str] = "CONDITIONAL"
FAIL: Final[str] = "FAIL"
DEFERRED: Final[str] = "DEFERRED"

#: Criteria that decide whether a metric is usable at all.
PREREQUISITE_CRITERIA: Final[frozenset[str]] = frozenset(
    {"direction", "aggregate_trend", "non_degeneracy", "clean_sanity"}
)


class GateConfigurationError(ValueError):
    """Raised when gate thresholds are missing, inconsistent, or out of range."""


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Predeclared numeric acceptance thresholds.

    Attributes
    ----------
    spearman_max:
        Maximum (negative) Spearman rho between severity and score.
    paired_effect_max:
        Maximum (negative) paired Cohen's d for every adjacent level pair.
    violation_rate_max:
        Maximum fraction of trials whose score moves the wrong way between
        adjacent levels.
    group_spearman_max:
        Maximum (negative) rho required within every font and every seed group.
    eta_squared_min:
        Minimum share of score variance attributable to severity.
    ssim_range_min:
        Minimum span of SSIM level means.
    psnr_range_min_db:
        Minimum span of PSNR level means, in dB.
    ink_correlation_max:
        Maximum |Pearson r| between score and ink coverage within a level.
    """

    spearman_max: float
    paired_effect_max: float
    violation_rate_max: float
    group_spearman_max: float
    eta_squared_min: float
    ssim_range_min: float
    psnr_range_min_db: float
    ink_correlation_max: float

    def __post_init__(self) -> None:
        if self.spearman_max >= 0:
            raise GateConfigurationError("spearman_max must be negative")
        if self.paired_effect_max >= 0:
            raise GateConfigurationError("paired_effect_max must be negative")
        if self.group_spearman_max >= 0:
            raise GateConfigurationError("group_spearman_max must be negative")
        if not 0.0 <= self.violation_rate_max <= 1.0:
            raise GateConfigurationError("violation_rate_max must be in [0, 1]")
        if not 0.0 <= self.eta_squared_min <= 1.0:
            raise GateConfigurationError("eta_squared_min must be in [0, 1]")
        if self.ssim_range_min <= 0 or self.psnr_range_min_db <= 0:
            raise GateConfigurationError("level-mean range minima must be positive")
        if not 0.0 <= self.ink_correlation_max <= 1.0:
            raise GateConfigurationError("ink_correlation_max must be in [0, 1]")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> GateThresholds:
        """Build thresholds from parsed TOML, with clear errors for missing keys."""
        required = (
            "spearman_max",
            "paired_effect_max",
            "violation_rate_max",
            "group_spearman_max",
            "eta_squared_min",
            "ssim_range_min",
            "psnr_range_min_db",
            "ink_correlation_max",
        )
        missing = [key for key in required if key not in values]
        if missing:
            raise GateConfigurationError(f"gate config is missing key(s): {missing}")
        extra = sorted(set(values) - set(required))
        if extra:
            raise GateConfigurationError(f"gate config has unknown key(s): {extra}")
        return cls(**{key: float(values[key]) for key in required})

    def range_minimum(self, metric: str) -> float:
        """Minimum acceptable span of level means for ``metric``."""
        if metric == "ssim":
            return self.ssim_range_min
        if metric == "psnr":
            return self.psnr_range_min_db
        raise GateConfigurationError(f"no range threshold defined for metric {metric!r}")

    def as_dict(self) -> dict[str, float]:
        """Return the thresholds as a plain mapping for reporting."""
        return {
            "spearman_max": self.spearman_max,
            "paired_effect_max": self.paired_effect_max,
            "violation_rate_max": self.violation_rate_max,
            "group_spearman_max": self.group_spearman_max,
            "eta_squared_min": self.eta_squared_min,
            "ssim_range_min": self.ssim_range_min,
            "psnr_range_min_db": self.psnr_range_min_db,
            "ink_correlation_max": self.ink_correlation_max,
        }


@dataclass(frozen=True, slots=True)
class Criterion:
    """One evaluated acceptance criterion."""

    id: str
    description: str
    observed: str
    threshold: str
    passed: bool
    note: str = ""


@dataclass(frozen=True, slots=True)
class MetricVerdict:
    """Gate outcome for one ``(family, metric)`` combination."""

    family: str
    metric: str
    verdict: str
    criteria: tuple[Criterion, ...]

    @property
    def failed_criteria(self) -> tuple[str, ...]:
        """IDs of every criterion that did not pass."""
        return tuple(criterion.id for criterion in self.criteria if not criterion.passed)


def _format(value: object, digits: int = 3) -> str:
    """Format numbers for reports, including ``inf``/``nan`` explicitly."""
    if value is None:
        return "n/a"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _row(summary: Mapping[str, pd.DataFrame], table: str, **filters: object) -> pd.Series | None:
    """Select a single summary row, or ``None`` if it does not exist."""
    frame = summary[table]
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        if column not in frame.columns:
            return None
        mask &= frame[column] == value
    selected = frame.loc[mask]
    if selected.empty:
        return None
    return selected.iloc[0]


def _select(summary: Mapping[str, pd.DataFrame], table: str, **filters: object) -> pd.DataFrame:
    """Select summary rows matching every filter."""
    frame = summary[table]
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column] == value
    return frame.loc[mask]


def _criterion(
    identifier: str,
    description: str,
    observed: str,
    threshold: str,
    passed: bool,
    note: str = "",
) -> Criterion:
    return Criterion(
        id=identifier,
        description=description,
        observed=observed,
        threshold=threshold,
        passed=bool(passed),
        note=note,
    )


def evaluate_gate(
    summary: Mapping[str, pd.DataFrame],
    sanity: pd.DataFrame,
    thresholds: GateThresholds,
    metrics: Sequence[str],
) -> tuple[MetricVerdict, ...]:
    """Evaluate every criterion for each ``(family, metric)`` pair.

    Parameters
    ----------
    summary:
        Output of :func:`captcha_quality.summarise.summarise`.
    sanity:
        Clean-vs-clean self-comparison rows, one per trial.
    thresholds:
        Predeclared thresholds.
    metrics:
        Metrics to evaluate.

    Returns
    -------
    tuple of MetricVerdict
        One verdict per family and metric.
    """
    families = sorted(summary["levels"]["family"].unique())
    clean_ssim_exact = bool(len(sanity) > 0 and np.all(sanity["ssim"].to_numpy(dtype=float) == 1.0))
    clean_psnr_infinite = bool(
        len(sanity) > 0 and np.all(np.isposinf(sanity["psnr"].to_numpy(dtype=float)))
    )
    clean_identical = bool(len(sanity) > 0 and sanity["identical"].all())

    verdicts: list[MetricVerdict] = []
    for family in families:
        for metric in metrics:
            criteria: list[Criterion] = []

            # --- 1. Direction -------------------------------------------------
            rho_row = _row(summary, "family_rho", family=family, metric=metric)
            rho = math.nan if rho_row is None else float(rho_row["spearman_rho"])
            criteria.append(
                _criterion(
                    "direction",
                    "Score decreases as severity increases (rank association).",
                    f"Spearman rho = {_format(rho)} (n = {int(rho_row['n']) if rho_row is not None else 0})",
                    f"rho <= {thresholds.spearman_max:.2f}",
                    math.isfinite(rho) and rho <= thresholds.spearman_max,
                    "A non-finite rho means the score is constant or has too few samples."
                    if not math.isfinite(rho)
                    else "",
                )
            )

            # --- 2. Aggregate trend ------------------------------------------
            trend_row = _row(summary, "trend", family=family, metric=metric)
            trend_ok = (
                bool(trend_row["all_strictly_decreasing"]) if trend_row is not None else False
            )
            step_text = "n/a"
            if trend_row is not None:
                step_text = ", ".join(
                    f"{step['from']}->{step['to']}: "
                    f"{_format(step['from_mean'])} -> {_format(step['to_mean'])}"
                    for step in trend_row["steps"]
                )
            criteria.append(
                _criterion(
                    "aggregate_trend",
                    "Mean score decreases strictly at every adjacent severity level.",
                    step_text,
                    "strict decrease at every step",
                    trend_ok,
                )
            )

            # --- 3. Adjacent separation --------------------------------------
            separation = _select(summary, "separation", family=family, metric=metric)
            evaluable = separation.loc[separation["evaluable"]]
            excluded = separation.loc[~separation["evaluable"]]
            if evaluable.empty:
                separation_ok = False
                effect_text = "no evaluable adjacent pairs"
            else:
                binding_effect = float(evaluable["paired_d"].max())
                per_pair = ", ".join(
                    f"{int(row['lower'])}->{int(row['upper'])}: d={_format(row['paired_d'])}"
                    for _, row in evaluable.iterrows()
                )
                separation_ok = bool((evaluable["paired_d"] <= thresholds.paired_effect_max).all())
                effect_text = f"{per_pair} (closest to zero: d = {_format(binding_effect)})"
            criteria.append(
                _criterion(
                    "adjacent_separation",
                    "Every evaluable adjacent level pair is distinguishable on average.",
                    effect_text,
                    f"paired Cohen's d <= {thresholds.paired_effect_max:.2f} for every pair",
                    separation_ok,
                    f"{len(excluded)} pair(s) excluded because no finite paired differences exist."
                    if len(excluded)
                    else "",
                )
            )

            # --- 4. Trial-level violations ------------------------------------
            violations = _select(summary, "violations", family=family, metric=metric)
            violations = violations.loc[violations["n_pairs"] >= 2]
            if violations.empty:
                violation_ok = False
                violation_text = "no evaluable adjacent pairs"
            else:
                worst_rate = float(violations["violation_rate"].max())
                violation_text = ", ".join(
                    f"{int(row['lower'])}->{int(row['upper'])}: {_format(row['violation_rate'])}"
                    for _, row in violations.iterrows()
                )
                violation_ok = bool(worst_rate <= thresholds.violation_rate_max)
            criteria.append(
                _criterion(
                    "violation_rate",
                    "Individual trials rarely move the wrong way between adjacent levels.",
                    violation_text,
                    f"violation rate <= {thresholds.violation_rate_max:.2f} per step",
                    violation_ok,
                    "Steps whose paired differences are all non-finite are excluded; ties are not violations.",
                )
            )

            # --- 5. Stability across fonts and seeds --------------------------
            for group_column, criterion_id in (
                ("font", "stability_font"),
                ("seed", "stability_seed"),
            ):
                stability = _select(
                    summary,
                    f"{group_column}_stability",
                    family=family,
                    metric=metric,
                )
                if len(stability) < 2:
                    criteria.append(
                        _criterion(
                            criterion_id,
                            f"Trend holds within every {group_column}.",
                            f"only {len(stability)} group(s) available",
                            f"rho <= {thresholds.group_spearman_max:.2f} in >= 2 groups",
                            False,
                            "Fewer than two groups makes cross-condition stability untestable.",
                        )
                    )
                    continue
                rho_values = stability["spearman_rho"].to_numpy(dtype=float)
                detail = ", ".join(
                    f"{row['group']}: {_format(row['spearman_rho'])}"
                    for _, row in stability.iterrows()
                )
                criteria.append(
                    _criterion(
                        criterion_id,
                        f"Trend holds within every {group_column}.",
                        detail,
                        f"rho <= {thresholds.group_spearman_max:.2f} in every group",
                        bool(np.all(np.isfinite(rho_values)))
                        and bool((rho_values <= thresholds.group_spearman_max).all()),
                    )
                )

            # --- 6. Non-degeneracy -------------------------------------------
            variance_row = _row(summary, "variance", family=family, metric=metric)
            ink_row = _row(summary, "ink", family=family, metric=metric)
            eta_squared = math.nan if variance_row is None else float(variance_row["eta_squared"])
            mean_range = (
                math.nan if variance_row is None else float(variance_row["level_mean_range"])
            )
            ink_correlation = (
                math.nan if ink_row is None else float(ink_row["max_abs_ink_correlation"])
            )
            ink_strength = math.nan if ink_row is None else float(ink_row["signed_ink_correlation"])
            ink_severity = -1 if ink_row is None else int(ink_row["at_severity"])
            minimum_range = thresholds.range_minimum(metric)
            non_degenerate = (
                math.isfinite(eta_squared)
                and eta_squared >= thresholds.eta_squared_min
                and math.isfinite(mean_range)
                and mean_range >= minimum_range
                and math.isfinite(ink_correlation)
                and ink_correlation <= thresholds.ink_correlation_max
            )
            criteria.append(
                _criterion(
                    "non_degeneracy",
                    "The score varies with severity, not with nuisance ink coverage.",
                    f"eta^2 = {_format(eta_squared)}, "
                    f"level-mean range = {_format(mean_range)}"
                    f"{' dB' if metric == 'psnr' else ''}, "
                    f"max |r_ink| = {_format(ink_correlation)} "
                    f"(signed {_format(ink_strength)} at severity {ink_severity})",
                    f"eta^2 >= {thresholds.eta_squared_min:.2f}, "
                    f"range >= {minimum_range:g}, "
                    f"|r_ink| <= {thresholds.ink_correlation_max:.2f}",
                    non_degenerate,
                    "eta^2 uses finite scores only; PSNR's +inf clean level is excluded."
                    if metric == "psnr"
                    else "",
                )
            )

            # --- 7. Clean-reference sanity ------------------------------------
            baseline = summary["levels"].loc[summary["levels"]["severity"] == 0]
            ssim_baseline = baseline.loc[baseline["metric"] == "ssim"]
            psnr_baseline = baseline.loc[baseline["metric"] == "psnr"]
            identity_ok = bool(
                len(baseline) > 0
                and (baseline["n_identical"] == baseline["n_total"]).all()
                and (ssim_baseline.empty or np.all(ssim_baseline["mean"].to_numpy() == 1.0))
                and (
                    psnr_baseline.empty
                    or (psnr_baseline["n_infinite"] == psnr_baseline["n_total"]).all()
                )
            )
            criteria.append(
                _criterion(
                    "clean_sanity",
                    "Clean-vs-clean comparisons are exactly perfect, including the "
                    "severity-0 identity condition.",
                    f"{len(sanity)} self-comparisons: SSIM=1.0 "
                    f"{int(clean_ssim_exact) * len(sanity)}/{len(sanity)}, "
                    f"PSNR=+inf {int(clean_psnr_infinite) * len(sanity)}/{len(sanity)}, "
                    f"byte-identical {int(clean_identical) * len(sanity)}/{len(sanity)}; "
                    f"severity-0 identity rows exact: {identity_ok}",
                    "all SSIM == 1.0, all PSNR == +inf, all identical",
                    clean_ssim_exact and clean_psnr_infinite and clean_identical and identity_ok,
                )
            )

            verdict = _verdict_from(criteria)
            verdicts.append(
                MetricVerdict(
                    family=family,
                    metric=metric,
                    verdict=verdict,
                    criteria=tuple(criteria),
                )
            )

    return tuple(verdicts)


def _verdict_from(criteria: Iterable[Criterion]) -> str:
    """Map criterion outcomes to a verdict (see module docstring)."""
    criteria = tuple(criteria)
    if any(
        not criterion.passed and criterion.id in PREREQUISITE_CRITERIA for criterion in criteria
    ):
        return FAIL
    if any(not criterion.passed for criterion in criteria):
        return CONDITIONAL
    return PASS


def deferred_metrics(metrics: Sequence[str]) -> dict[str, str]:
    """Return deferral reasons for requested metrics that were not evaluated."""
    return {metric: DEFERRED_METRICS[metric] for metric in metrics if metric in DEFERRED_METRICS}


def render_markdown(
    summary: Mapping[str, pd.DataFrame],
    verdicts: Sequence[MetricVerdict],
    thresholds: GateThresholds,
    *,
    header_lines: Sequence[str] = (),
    deferred: Mapping[str, str] | None = None,
) -> str:
    """Render the pilot's human-readable results summary."""
    lines: list[str] = ["# Milestone 1 target-validation summary", ""]
    lines.extend(header_lines)
    if header_lines:
        lines.append("")

    lines.extend(
        [
            "## Gate verdicts",
            "",
            "| Family | Metric | Verdict | Failed criteria |",
            "|---|---|---|---|",
        ]
    )
    for verdict in verdicts:
        failed = ", ".join(verdict.failed_criteria) or "—"
        lines.append(f"| {verdict.family} | {verdict.metric} | **{verdict.verdict}** | {failed} |")

    lines.extend(["", "## Criterion detail", ""])
    for verdict in verdicts:
        lines.extend(
            [
                f"### {verdict.family} / {verdict.metric} — {verdict.verdict}",
                "",
                "| Criterion | Observed | Threshold | Result |",
                "|---|---|---|---|",
            ]
        )
        for criterion in verdict.criteria:
            observed = criterion.observed.replace("|", "\\|")
            result = "PASS" if criterion.passed else "FAIL"
            lines.append(f"| {criterion.id} | {observed} | {criterion.threshold} | {result} |")
            if criterion.note:
                lines.append(f"| | _{criterion.note}_ | | |")
        lines.append("")

    lines.extend(["## Level means", ""])
    for metric in sorted(summary["levels"]["metric"].unique()):
        lines.extend(
            [
                f"### {metric}",
                "",
                "| Family | Severity | Parameter | n | Mean | SD | Min | Max | +inf |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        table = summary["levels"].loc[summary["levels"]["metric"] == metric]
        for _, row in table.iterrows():
            lines.append(
                f"| {row['family']} | {int(row['severity'])} | "
                f"{_format(row['parameter'])} {row['unit']} | {int(row['n_finite'])} | "
                f"{_format(row['mean'])} | {_format(row['sd'])} | {_format(row['min'])} | "
                f"{_format(row['max'])} | {int(row['n_infinite'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Adjacent-level separation",
            "",
            "| Family | Metric | Pair | n | Mean diff | 95% CI | Paired d | Violations |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    merged = summary["separation"].merge(
        summary["violations"],
        on=["family", "metric", "lower", "upper", "n_pairs"],
        how="left",
    )
    for _, row in merged.iterrows():
        lines.append(
            f"| {row['family']} | {row['metric']} | {int(row['lower'])}->{int(row['upper'])} | "
            f"{int(row['n_pairs'])} | {_format(row['mean_diff'])} | "
            f"[{_format(row['ci95_low'])}, {_format(row['ci95_high'])}] | "
            f"{_format(row['paired_d'])} | {_format(row['violation_rate'])} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Stability",
            "",
            "| Group by | Family | Metric | Group | n | Spearman rho | Level means decreasing |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    stability = pd.concat([summary["font_stability"], summary["seed_stability"]], ignore_index=True)
    for _, row in stability.iterrows():
        lines.append(
            f"| {row['group_column']} | {row['family']} | {row['metric']} | {row['group']} | "
            f"{int(row['n'])} | {_format(row['spearman_rho'])} | "
            f"{'yes' if row['level_means_decreasing'] else 'no'} |"
        )
    lines.append("")

    if deferred:
        lines.extend(["## Deferred metrics", ""])
        for metric, reason in deferred.items():
            lines.append(f"- **{metric}** — {reason}")
        lines.append("")

    lines.extend(
        [
            "## Threshold provenance",
            "",
            "These thresholds were committed in "
            "`configs/pilot_target_validation.toml` before the pilot was run and "
            "are documented in `docs/METHODOLOGY.md`.",
            "",
            "```json",
            json.dumps(thresholds.as_dict(), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
